from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from osrl.algorithms import (
    BEARLTrainer,
    BCTrainer,
    CDTTrainer,
    COptiDICETrainer,
    BCQLTrainer,
    CPQTrainer,
)

# ── shared utilities ──────────────────────────────────────────────────────────


def _grad_l2(gs):
    s = 0.0
    for g in gs:
        if g is not None:
            s = s + g.pow(2).sum()
    return torch.sqrt(s + 1e-12)


def _dot(gs1, gs2):
    out = 0.0
    for a, b in zip(gs1, gs2):
        if a is not None and b is not None:
            out = out + (a.flatten() * b.flatten()).sum()
    return out


def _norm2(gs):
    out = 0.0
    for a in gs:
        if a is not None:
            out = out + (a.flatten() * a.flatten()).sum()
    return out + 1e-12


def _multihead_huber(target, q_list):
    return sum(F.smooth_l1_loss(q, target) for q in q_list) / len(q_list)


def _init_unlearn_state(obj):
    """Lazily initialise CPQ-style unlearning state on the trainer. Idempotent."""
    if not hasattr(obj, "_F_keep_actor") or obj._F_keep_actor is None:
        obj._F_keep_actor = None
    if not hasattr(obj, "_F_keep_beta"):
        obj._F_keep_beta = 0.99
    if not hasattr(obj, "_F_keep_cprime"):
        obj._F_keep_cprime = 1e-3
    if not hasattr(obj, "_gn_ema"):
        obj._gn_ema = {"keep": 1.0, "forget": 1.0}
    if not hasattr(obj, "_eta_forget"):
        obj._eta_forget = 0.1
    if not hasattr(obj, "_eta_beta"):
        obj._eta_beta = 0.9
    if not hasattr(obj, "_eta_min"):
        obj._eta_min = 0.01
    if not hasattr(obj, "_eta_max"):
        obj._eta_max = 1.0
    if not hasattr(obj, "_forget_cost_target"):
        obj._forget_cost_target = None


def _normalise_and_pcgrad(obj, g_keep, g_forget):
    """EMA-normalise both gradient sets, then PCGrad-project g_forget."""
    gn_beta = 0.95
    with torch.no_grad():
        nk = float(_grad_l2(g_keep).cpu())
        nf = float(_grad_l2(g_forget).cpu())
        obj._gn_ema["keep"] = gn_beta * obj._gn_ema["keep"] + (1 - gn_beta) * nk
        obj._gn_ema["forget"] = gn_beta * obj._gn_ema["forget"] + (1 - gn_beta) * nf
    g_keep = [g / max(obj._gn_ema["keep"], 1e-3) for g in g_keep]
    g_forget = [g / max(obj._gn_ema["forget"], 1e-3) for g in g_forget]
    dkf = _dot(g_keep, g_forget)
    if dkf < 0:
        proj = dkf / _norm2(g_keep)
        g_forget = [gf - proj * gk for gf, gk in zip(g_forget, g_keep)]
    return g_keep, g_forget


def _fisher_scale_forget(obj, actor_params, g_keep, g_forget, clip=0.05):
    """Dampen g_forget where keep curvature is high via Fisher EMA diagonal."""
    if obj._F_keep_actor is None:
        obj._F_keep_actor = [torch.zeros_like(g) for g in g_keep]
    beta = float(obj._F_keep_beta)
    cprime = float(obj._F_keep_cprime)
    with torch.no_grad():
        for i in range(len(actor_params)):
            obj._F_keep_actor[i] = beta * obj._F_keep_actor[i] + (1 - beta) * (
                g_keep[i].detach() ** 2
            )
    return [
        torch.clamp(g / (obj._F_keep_actor[i].sqrt() + cprime), -clip, clip)
        for i, g in enumerate(g_forget)
    ]


def _update_eta(obj, gap):
    """
    Adaptive eta_forget (CPQ design).
    gap > 0: cost target not yet reached → more forgetting needed → higher eta.
    """
    eta_score = torch.sigmoid(torch.tensor(gap / 0.25, dtype=torch.float32)).item()
    eta_target = obj._eta_min + (obj._eta_max - obj._eta_min) * eta_score
    obj._eta_forget = obj._eta_beta * obj._eta_forget + (1 - obj._eta_beta) * eta_target
    return float(max(obj._eta_min, min(obj._eta_max, obj._eta_forget)))


def _actor_step(obj, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm):
    """Apply combined keep + eta·forget gradient and step the actor optimiser."""
    obj.model.actor_optim.zero_grad(set_to_none=True)
    for p, gk, gf in zip(actor_params, g_keep, g_forget_scaled):
        # p.grad = gk + eta_forget * gf
        p.grad = gk + gf
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
    obj.model.actor_optim.step()


# ── BC ────────────────────────────────────────────────────────────────────────


class BCTrainerWithUnlearn(BCTrainer):
    """BC unlearning following CPQ's actor-update design.

    Since BC has no critics, action displacement from forget data serves as
    the cost proxy:
      loss_forget_cost  ←→  softplus(target_dist − dist_f)  push actions away
      loss_forget_q     ←→  −dist_f.mean()                  maximise separation
    Adaptive eta uses (dist_f − target) as the gap signal.
    """

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        alpha_pg_keep=1.0,
        update_targets=False,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_method="safe",
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)

        obs_f, act_f = [b.to(device) for b in batch_forget]
        obs_k, act_k = [b.to(device) for b in batch_keep]

        if not use_policy_gradient:
            return {"loss_actor": 0.0}

        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]
        act_k_pi = self.model.actor(obs_k)
        act_f_pi = self.model.actor(obs_f)

        # ── KEEP ─────────────────────────────────────────────────────────────
        loss_keep = F.mse_loss(act_k_pi, act_k.detach())
        keep_total = alpha_pg_keep * loss_keep

        # ── FORGET (CPQ-style, critic-free) ──────────────────────────────────
        # Action displacement plays the role of the cost signal:
        # high dist = "unsafe" (already forgotten) → softplus satisfied.
        dist_f = ((act_f_pi - act_f.detach()) ** 2).mean(dim=-1)
        forget_dist_target = 0.10
        loss_forget_cost = F.softplus((forget_dist_target - dist_f) / 0.05).mean()
        loss_forget_q = -dist_f.mean()
        forget_total = loss_forget_cost + loss_forget_q

        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        g_keep = torch.autograd.grad(
            keep_total, actor_params, retain_graph=True, allow_unused=False
        )
        g_forget = torch.autograd.grad(
            forget_total, actor_params, retain_graph=False, allow_unused=False
        )
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, actor_params, g_keep, g_forget)

        if self._forget_cost_target is None:
            self._forget_cost_target = forget_dist_target
        gap = float(dist_f.mean().detach().cpu()) - self._forget_cost_target
        eta_forget = _update_eta(self, gap)

        _actor_step(
            self, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm
        )

        return {
            "eta_forget": eta_forget,
            "loss_keep": float(loss_keep.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "action_dist_f": float(dist_f.mean().detach().cpu()),
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }


# ── CDT ───────────────────────────────────────────────────────────────────────


 
class CDTTrainerWithUnlearn(CDTTrainer):
    """CDT unlearning aligned with CDTTrainer.train_one_step.
 
    Keep loss mirrors train_one_step exactly: act_loss + cost_weight*cost_loss
    + state_weight*state_loss, protecting all three prediction heads.
 
    Forget loss uses CDT's built-in cost_pred_head as the Qc analog (CPQ
    mapping):
      loss_forget_cost  push cost_pred_f toward class 1 (high-cost) — same
                        role as softplus(target − cq_f) in CPQ
      loss_forget_q     push action predictions away from forget actions —
                        same role as q_f.mean() (gradient descent on reward)
    Adaptive eta is driven by the model's own cost-class-1 confidence on
    forget states, mirroring CPQ's (cq_f − target) gap signal.
    """
 
    def _get_max_timestep_index(self):
        for name in (
            "timestep_emb",
            "time_emb",
            "embed_timestep",
            "timestep_embedding",
        ):
            emb = getattr(self.model, name, None)
            if emb is not None and hasattr(emb, "num_embeddings"):
                return int(emb.num_embeddings - 1)
        raise AttributeError(
            "Could not find CDT timestep embedding on self.model. "
            "Adapt _get_max_timestep_index() to your CDT implementation."
        )
 
    def _sanitize_seq_batch(
        self,
        states,
        actions,
        returns_to_go,
        costs_to_go,
        time_steps,
        mask,
        episode_cost,
        costs,
    ):
        device = self.device
        states = torch.nan_to_num(states.to(device).float())
        actions = torch.nan_to_num(actions.to(device).float())
        returns_to_go = torch.nan_to_num(returns_to_go.to(device).float())
        costs_to_go = torch.nan_to_num(costs_to_go.to(device).float())
        episode_cost = torch.nan_to_num(episode_cost.to(device).float())
        costs = torch.nan_to_num(costs.to(device).float())
        mask = (mask.to(device) > 0).float()
        time_steps = time_steps.to(device).long()
        time_steps = time_steps.masked_fill(mask == 0, 0)
        time_steps = time_steps.clamp_(0, self._get_max_timestep_index())
        return (
            states,
            actions,
            returns_to_go,
            costs_to_go,
            time_steps,
            mask,
            episode_cost,
            costs,
        )
 
    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        alpha_pg_keep=1.0,
        update_targets=False,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_method="safe",
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)
 
        s_f, a_f, rtg_f, ctg_f, ts_f, mask_f, ec_f, costs_f = self._sanitize_seq_batch(
            *batch_forget
        )
        s_k, a_k, rtg_k, ctg_k, ts_k, mask_k, ec_k, costs_k = self._sanitize_seq_batch(
            *batch_keep
        )
 
        if not use_policy_gradient:
            return {"loss_actor": 0.0}
 
        params = [p for p in self.model.parameters() if p.requires_grad]
 
        # ════════════════════════════════════════════════════════════════════
        # KEEP: mirror train_one_step exactly so all three heads are protected
        # ════════════════════════════════════════════════════════════════════
        act_pred_k, cost_pred_k, state_pred_k = self.model(
            states=s_k,
            actions=a_k,
            returns_to_go=rtg_k,
            costs_to_go=ctg_k,
            time_steps=ts_k,
            padding_mask=~mask_k.to(torch.bool),
            episode_cost=ec_k,
        )
 
        # Action loss — stochastic and deterministic paths from train_one_step
        keep_entropy = None
        if self.stochastic:
            logp_k = act_pred_k.log_prob(a_k)[mask_k > 0].mean()
            keep_entropy = act_pred_k.entropy()[mask_k > 0].mean()
            ent_reg = self.model.temperature().detach() if not self.no_entropy else 0.0
            act_loss_k = -(logp_k + ent_reg * keep_entropy)
        else:
            act_loss_k = F.mse_loss(act_pred_k, a_k.detach(), reduction="none")
            act_loss_k = (
                act_loss_k * mask_k.unsqueeze(-1)
            ).sum() / mask_k.sum().clamp_min(1.0)
 
        # Cost classification loss (NLL over 2 classes, same as train_one_step)
        cost_pred_k_flat = cost_pred_k.reshape(-1, 2)
        costs_k_flat = costs_k.flatten().long().detach()
        cost_loss_k = F.nll_loss(cost_pred_k_flat, costs_k_flat, reduction="none")
        cost_loss_k = (cost_loss_k * mask_k.flatten()).sum() / mask_k.sum().clamp_min(
            1.0
        )
 
        # State prediction loss (same as train_one_step)
        state_loss_k = F.mse_loss(
            state_pred_k[:, :-1], s_k[:, 1:].detach(), reduction="none"
        )
        state_loss_k = (state_loss_k * mask_k[:, :-1].unsqueeze(-1)).sum() / mask_k[
            :, :-1
        ].sum().clamp_min(1.0)
 
        loss_keep = (
            act_loss_k
            + self.cost_weight * cost_loss_k
            + self.state_weight * state_loss_k
        )
        keep_total = alpha_pg_keep * loss_keep
 
        # ════════════════════════════════════════════════════════════════════
        # FORGET: CPQ-style using CDT's built-in cost_pred_head as Qc proxy
        # ════════════════════════════════════════════════════════════════════
        act_pred_f, cost_pred_f, _ = self.model(
            states=s_f,
            actions=a_f,
            returns_to_go=rtg_f,
            costs_to_go=ctg_f,
            time_steps=ts_f,
            padding_mask=~mask_f.to(torch.bool),
            episode_cost=ec_f,
        )
 
        # loss_forget_cost: push cost_pred_f toward class 1 (high cost) —
        # CPQ analog: softplus((target − cq_f) / 0.25), satisfied when cq_f
        # is high.  Here: NLL with all targets = 1 forces the model to predict
        # "this is a costly transition" for every forget step.
        # cost_pred_f is already log_softmax, so NLL = -log_softmax[:,1].
        cost_pred_f_flat = cost_pred_f.reshape(-1, 2)
        high_cost_targets = torch.ones(
            cost_pred_f_flat.shape[0], dtype=torch.long, device=device
        )
        loss_forget_cost_raw = F.nll_loss(
            cost_pred_f_flat, high_cost_targets, reduction="none"
        )
        loss_forget_cost = (
            loss_forget_cost_raw * mask_f.flatten()
        ).sum() / mask_f.sum().clamp_min(1.0)
 
        # loss_forget_q: push action predictions away from forget actions —
        # CPQ analog: q_f.mean() (descent on reward for forget states).
        # Stochastic: minimise log-prob of original forget actions = reduce
        # the policy's probability on those exact actions.
        # Deterministic: maximise MSE from forget actions = same semantics.
        if self.stochastic:
            loss_forget_q = act_pred_f.log_prob(a_f.detach())[mask_f > 0].mean()
        else:
            err_f = ((act_pred_f - a_f.detach()) ** 2 * mask_f.unsqueeze(-1)).mean(
                dim=-1
            )
            loss_forget_q = -((err_f * mask_f).sum() / mask_f.sum().clamp_min(1.0))
 
        forget_total = loss_forget_cost + loss_forget_q
 
        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        # CDT is a single transformer — use .backward()/.grad instead of
        # torch.autograd.grad() because heads that are unused by one loss get
        # grad=None (not an error); we treat those as zero.
        self.optim.zero_grad(set_to_none=True)
        keep_total.backward(retain_graph=True)
        g_keep = tuple(
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in params
        )
 
        self.optim.zero_grad(set_to_none=True)
        forget_total.backward()
        g_forget = tuple(
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in params
        )
        self.optim.zero_grad(set_to_none=True)
 
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, params, g_keep, g_forget)
 
        # Adaptive eta: gap = (cost-class-1 confidence on forget) - target.
        # When the model already predicts high cost on forget states (good),
        # class-1 prob is high → gap > 0 → _update_eta sees positive gap and
        # reduces eta via sigmoid saturation.
        if self._forget_cost_target is None:
            self._forget_cost_target = 0.7  # target 70% class-1 confidence
        cost1_conf = float(
            cost_pred_f.reshape(-1, 2)[:, 1]
            .exp()[mask_f.flatten() > 0]
            .mean()
            .detach()
            .cpu()
        )
        gap = cost1_conf - self._forget_cost_target
        eta_forget = _update_eta(self, gap)
 
        for p, gk, gf in zip(params, g_keep, g_forget_scaled):
            p.grad = gk + eta_forget * gf
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        self.optim.step()
 
        # ── Temperature update: mirror train_one_step exactly ─────────────────
        # Entropy was already computed from the keep forward pass above —
        # no second forward pass needed.  target_entropy falls back to
        # -action_dim (standard SAC heuristic) when not explicitly set.
        if self.stochastic and hasattr(self, "log_temperature_optimizer"):
            target_ent = self.model.target_entropy
            if target_ent is None:
                target_ent = torch.tensor(-float(a_k.shape[-1]), device=device)
            self.log_temperature_optimizer.zero_grad()
            (self.model.temperature() * (keep_entropy - target_ent).detach()).backward()
            self.log_temperature_optimizer.step()
 
        self.scheduler.step()
 
        return {
            "eta_forget": eta_forget,
            "loss_keep": float(loss_keep.detach().cpu()),
            "loss_act_keep": float(act_loss_k.detach().cpu()),
            "loss_cost_keep": float(cost_loss_k.detach().cpu()),
            "loss_state_keep": float(state_loss_k.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "cost1_conf_f": cost1_conf,
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }
 

# ── COptiDICE ─────────────────────────────────────────────────────────────────


class COptiDICETrainerWithUnlearn(COptiDICETrainer):
    """COptiDICE unlearning following CPQ's actor-update design.

    nu/chi networks serve as value/cost proxies:
      keep:         nu-weighted actor loss (standard COptiDICE actor objective)
      forget cost:  chi_f as Qc proxy — push > qc_thres + margin (CPQ formula)
      forget q:     nu_f weighted by w_f — value signal on forget states
    Adaptive eta uses (chi_f − target) as the gap signal.
    """

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        lambda_cost=1.0,
        alpha_pg_keep=1.0,
        update_targets=False,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_method="safe",
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)

        def _unpack(batch):
            obs, next_obs, act, rew, cost, done, is_init = batch
            return (
                obs.to(device),
                next_obs.to(device),
                act.to(device),
                rew.to(device),
                cost.to(device),
                done.to(device),
                is_init.to(device),
            )

        obs_f, next_obs_f, act_f, rew_f, cost_f, done_f, init_f = _unpack(batch_forget)
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k, init_k = _unpack(batch_keep)

        z = lambda: torch.tensor(0.0, device=device)
        loss_nu = loss_chi = loss_lmbda = z()

        # ════════════════════════════════════════════════════════════════════
        # 1) nu / chi / lambda update (keep data only — standard COptiDICE)
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = False
        for p in self.model.nu_network.parameters():
            p.requires_grad = True
        for p in self.model.chi_network.parameters():
            p.requires_grad = True

        if unlearn_method == "safe":
            self.model._lmbda = F.softplus(self.model.lmbda)
            nu_s_k, _ = self.model.nu_network.predict(obs_k, None)
            nu_s_nk, _ = self.model.nu_network.predict(next_obs_k, None)
            e_nu_lam_k = (
                rew_k
                - self.model._lmbda.detach() * cost_k
                + self.model.gamma * (1.0 - done_k) * nu_s_nk
                - nu_s_k
            )
            w_sa_k = F.relu(self.model.f_prime_inv_fn(e_nu_lam_k / self.model.alpha))
            w_sa_k_d = w_sa_k.detach()
            Bk = obs_k.shape[0]

            if self.model.cost_ub_epsilon > 0:
                self.model._tau = F.softplus(self.model.tau)
                chi_s_k, _ = self.model.chi_network.predict(obs_k, None)
                chi_s_nk, _ = self.model.chi_network.predict(next_obs_k, None)
                chi_init_k = chi_s_k * init_k / self.model.init_state_propotion
                ell_k = (1 - self.model.gamma) * chi_init_k + w_sa_k_d * (
                    cost_k + self.model.gamma * (1 - done_k) * chi_s_nk - chi_s_k
                )
                logits_k = ell_k / self.model._tau.detach()
                weights_k = torch.softmax(logits_k, dim=0) * Bk
                log_wk = torch.log_softmax(logits_k, dim=0) + np.log(Bk)
                d_kl_k = (weights_k * log_wk - weights_k + 1).mean()
                weighted_c_k = (weights_k * w_sa_k_d * cost_k).mean()
                loss_chi = (weights_k * ell_k).mean()
                self.model.chi_optim.zero_grad(set_to_none=True)
                loss_chi.backward(retain_graph=True)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.chi_network.parameters(), max_grad_norm
                    )
                self.model.chi_optim.step()
                self.model.tau_optim.zero_grad(set_to_none=True)
                (
                    self.model._tau * (self.model.cost_ub_epsilon - d_kl_k.detach())
                ).backward()
                self.model.tau_optim.step()
            else:
                weighted_c_k = (w_sa_k_d * cost_k).mean()

            nu_init_k = nu_s_k * init_k / self.model.init_state_propotion
            loss_nu = (1 - self.model.gamma) * nu_init_k.mean() + (
                w_sa_k * e_nu_lam_k - self.model.alpha * self.model.f_fn(w_sa_k)
            ).mean()
            self.model.nu_optim.zero_grad(set_to_none=True)
            loss_nu.backward(retain_graph=True)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.nu_network.parameters(), max_grad_norm
                )
            self.model.nu_optim.step()

            loss_lmbda = self.model._lmbda * (
                self.model.qc_thres - weighted_c_k.detach()
            )
            self.model.lmbda_optim.zero_grad(set_to_none=True)
            loss_lmbda.backward()
            self.model.lmbda_optim.step()

        # ════════════════════════════════════════════════════════════════════
        # 2) Actor: CPQ-style keep + forget update
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.nu_network.parameters():
            p.requires_grad = False
        for p in self.model.chi_network.parameters():
            p.requires_grad = False

        if not use_policy_gradient:
            for m in (self.model.nu_network, self.model.chi_network, self.model.actor):
                for p in m.parameters():
                    p.requires_grad = True
            return {
                "loss_nu": float(loss_nu.detach().cpu()),
                "loss_chi": float(loss_chi.detach().cpu()),
                "loss_lmbda": float(loss_lmbda.detach().cpu()),
                "loss_actor": 0.0,
            }

        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]

        # ── KEEP: nu-weighted actor loss ──────────────────────────────────────
        with torch.no_grad():
            nu_k, _ = self.model.nu_network.predict(obs_k, None)
            nu_nk, _ = self.model.nu_network.predict(next_obs_k, None)
            e_k = (
                rew_k
                - self.model._lmbda.detach() * cost_k
                + self.model.gamma * (1.0 - done_k) * nu_nk
                - nu_k
            )
            w_k = F.relu(self.model.f_prime_inv_fn(e_k / self.model.alpha)).detach()
 
        obs_eps_k = torch.randn_like(obs_k) * self.model.observations_std * 0.1
        act_eps_k = torch.randn_like(act_k) * self.model.actions_std * 0.1
        _, _, dist_k = self.model.actor.forward(obs_k + obs_eps_k, False, True, True)
        keep_actor_loss = -(w_k * dist_k.log_prob(act_k + act_eps_k).sum(-1)).mean()
        keep_total = alpha_pg_keep * keep_actor_loss

        # ── FORGET ────────────────────────────────────────────────────────────

        with torch.no_grad():
            nu_f, _ = self.model.nu_network.predict(obs_f, None)
            nu_nf, _ = self.model.nu_network.predict(next_obs_f, None)
            e_f = (
                rew_f
                - self.model._lmbda.detach() * cost_f
                + self.model.gamma * (1.0 - done_f) * nu_nf
                - nu_f
            )
            w_f = F.relu(self.model.f_prime_inv_fn(e_f / self.model.alpha)).detach()
            chi_f, _ = self.model.chi_network.predict(obs_f, None)
            # Normalise chi to a unit-scale weight so its magnitude doesn't
            # swamp w_f; detach so it is a pure scalar coefficient.
            chi_f_w = (chi_f / (chi_f.abs().mean() + 1e-6)).detach()
 
        # Actor distribution on forget states — log_prob has grad_fn through actor.
        obs_eps_f = torch.randn_like(obs_f) * self.model.observations_std * 0.1
        _, _, dist_f = self.model.actor.forward(obs_f + obs_eps_f, False, True, True)
        logp_forget = dist_f.log_prob(act_f.detach()).sum(-1)  # [B], differentiable
 
        qc_thres = float(self.model.qc_thres)
        if self._forget_cost_target is None:
            self._forget_cost_target = qc_thres + 0.5
 
        # Minimising pushes logp_forget down in proportion to chi / w_f weights.
        loss_forget_cost = (chi_f_w.squeeze() * logp_forget).mean()
        loss_forget_q = (w_f.squeeze() * logp_forget).mean()
        forget_total = loss_forget_cost + loss_forget_q
 
        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        g_keep = torch.autograd.grad(
            keep_total, actor_params, retain_graph=True, allow_unused=False
        )
        g_forget = torch.autograd.grad(
            forget_total, actor_params, retain_graph=False, allow_unused=False
        )
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, actor_params, g_keep, g_forget)

        gap = float((self._forget_cost_target - chi_f.mean()).detach().cpu())
        eta_forget = _update_eta(self, gap)

        _actor_step(
            self, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm
        )

        for m in (self.model.nu_network, self.model.chi_network, self.model.actor):
            for p in m.parameters():
                p.requires_grad = True

        return {
            "eta_forget": eta_forget,
            "loss_nu": float(loss_nu.detach().cpu()),
            "loss_chi": float(loss_chi.detach().cpu()),
            "loss_lmbda": float(loss_lmbda.detach().cpu()),
            "loss_keep": float(keep_actor_loss.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "lambda": float(self.model._lmbda.detach().cpu()),
            "chi_forget_mean": float(chi_f.mean().cpu()),
            "w_keep_mean": float(w_k.mean().cpu()),
            "w_forget_mean": float(w_f.mean().cpu()),
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }


# ── BCQL ──────────────────────────────────────────────────────────────────────


class BCQLTrainerWithUnlearn(BCQLTrainer):
    """BCQL unlearning following CPQ's critic and actor design.

    Critic differences vs CPQ:
      - Bootstrap: actor_old(rep, vae.decode(rep)) + K-sample max
      - Predict API: (q1, q2, q1_list, q2_list) — both heads used for Huber
      - Cost backup: no (1-done) factor (BCQL convention)
    Actor: identical CPQ forget design (cq_f > target softplus + q_f.mean()).
    """

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        lambda_cost=1.0,
        alpha_q_keep=1.0,
        alpha_q_forget=0.1,
        alpha_cq_keep=1.0,
        alpha_cq_forget=0.1,
        alpha_pg_keep=1.0,
        update_targets=True,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_method="safe",
        K=None,
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)

        qc_thres = float(self.model.qc_thres)
        K_val = K or self.model.sample_action_num

        obs_f, next_obs_f, act_f, rew_f, cost_f, done_f = [
            b.to(device) for b in batch_forget
        ]
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k = [
            b.to(device) for b in batch_keep
        ]
        Bk = obs_k.shape[0]

        z = lambda: torch.tensor(0.0, device=device)
        loss_q_keep = loss_q_forget = loss_cq_keep = loss_cq_forget = z()
        loss_keep = loss_cost_keep = loss_bc_keep = z()
        loss_forget_cost = loss_forget_q = z()

        # ════════════════════════════════════════════════════════════════════
        # 1) Critics: CPQ-style backup + forget suppression
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False
        for p in self.model.critic.parameters():
            p.requires_grad = True
        for p in self.model.cost_critic.parameters():
            p.requires_grad = True

        if unlearn_method == "safe":
            # ── Reward critic ─────────────────────────────────────────────────
            _, _, q1lk, q2lk = self.model.critic.predict(obs_k, act_k)
            _, _, q1lf, q2lf = self.model.critic.predict(obs_f, act_f)

            with torch.no_grad():
                rep_k = torch.repeat_interleave(next_obs_k, K_val, 0)
                act_t = self.model.actor_old(rep_k, self.model.vae.decode(rep_k))
                q1t, q2t, _, _ = self.model.critic_old.predict(rep_k, act_t)
                q_next = (
                    (
                        self.model.lmbda * torch.min(q1t, q2t)
                        + (1 - self.model.lmbda) * torch.max(q1t, q2t)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                # CPQ safety gate on bootstrap targets
                cq1t, cq2t, _, _ = self.model.cost_critic_old.predict(rep_k, act_t)
                cq_next = (
                    (
                        self.model.lmbda * torch.min(cq1t, cq2t)
                        + (1 - self.model.lmbda) * torch.max(cq1t, cq2t)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                gate = (cq_next <= qc_thres).float()
                backup_r_k = rew_k + self.model.gamma * (1 - done_k) * gate * q_next
                tau_r = torch.quantile(backup_r_k, 0.30)
                q_ref_f = torch.full_like(backup_r_k, tau_r.item())

            loss_q_keep = (
                _multihead_huber(backup_r_k, q1lk) + _multihead_huber(backup_r_k, q2lk)
            ) * 0.5
            lqf = sum(F.softplus((q - q_ref_f) / 0.5).mean() for q in q1lf + q2lf)
            loss_q_forget = lqf / len(q1lf + q2lf)

            self.model.critic_optim.zero_grad(set_to_none=True)
            (alpha_q_keep * loss_q_keep + alpha_q_forget * loss_q_forget).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), max_grad_norm
                )
            self.model.critic_optim.step()

            # ── Cost critic ───────────────────────────────────────────────────
            _, _, cq1lk, cq2lk = self.model.cost_critic.predict(obs_k, act_k)
            _, _, cq1lf, cq2lf = self.model.cost_critic.predict(obs_f, act_f)

            with torch.no_grad():
                cq1t2, cq2t2, _, _ = self.model.cost_critic_old.predict(rep_k, act_t)
                cq_targ_k = (
                    (
                        self.model.lmbda * torch.min(cq1t2, cq2t2)
                        + (1 - self.model.lmbda) * torch.max(cq1t2, cq2t2)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                # BCQL: no (1-done) in cost backup
                backup_c_k = cost_k + self.model.gamma * cq_targ_k
                cq_ref_f = torch.full_like(backup_c_k, qc_thres + 0.5)

            loss_cq_keep = (
                _multihead_huber(backup_c_k, cq1lk)
                + _multihead_huber(backup_c_k, cq2lk)
            ) * 0.5
            lcf = sum(F.softplus((cq_ref_f - cq) / 0.25).mean() for cq in cq1lf + cq2lf)
            loss_cq_forget = lcf / len(cq1lf + cq2lf)

            self.model.cost_critic_optim.zero_grad(set_to_none=True)
            (
                alpha_cq_keep * loss_cq_keep
                + alpha_cq_forget * lambda_cost * loss_cq_forget
            ).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.cost_critic.parameters(), max_grad_norm
                )
            self.model.cost_critic_optim.step()

        # ════════════════════════════════════════════════════════════════════
        # 2) Actor: CPQ-style keep + forget update
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.critic.parameters():
            p.requires_grad = False
        for p in self.model.cost_critic.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False

        if not use_policy_gradient:
            for m in (
                self.model.critic,
                self.model.cost_critic,
                self.model.vae,
                self.model.actor,
            ):
                for p in m.parameters():
                    p.requires_grad = True
            return {"loss_actor": 0.0}

        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]

        act_k_pi = self.model.actor(obs_k, self.model.vae.decode(obs_k))
        act_f_pi = self.model.actor(obs_f, self.model.vae.decode(obs_f))
        q1k, q2k, _, _ = self.model.critic.predict(obs_k, act_k_pi)
        cq1k, cq2k, _, _ = self.model.cost_critic.predict(obs_k, act_k_pi)
        q1f, q2f, _, _ = self.model.critic.predict(obs_f, act_f_pi)
        cq1f, cq2f, _, _ = self.model.cost_critic.predict(obs_f, act_f_pi)
        q_k = torch.min(q1k, q2k)
        cq_k = torch.min(cq1k, cq2k)
        q_f = torch.min(q1f, q2f)
        cq_f = torch.min(cq1f, cq2f)

        # ── KEEP ─────────────────────────────────────────────────────────────
        loss_keep = -((cq_k <= qc_thres).float() * q_k).mean()
        loss_cost_keep = F.softplus(cq_k - qc_thres).mean()
        loss_bc_keep = F.mse_loss(act_k_pi, act_k.detach())
        keep_total = (
            alpha_pg_keep * loss_keep + 1.0 * loss_cost_keep + 1.0 * loss_bc_keep
        )

        # ── FORGET (CPQ exact design) ─────────────────────────────────────────
        if self._forget_cost_target is None:
            self._forget_cost_target = qc_thres + 0.5
        loss_forget_cost = F.softplus((self._forget_cost_target - cq_f) / 0.25).mean()
        loss_forget_q = q_f.mean()
        forget_total = loss_forget_cost + loss_forget_q

        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        g_keep = torch.autograd.grad(
            keep_total, actor_params, retain_graph=True, allow_unused=False
        )
        g_forget = torch.autograd.grad(
            forget_total, actor_params, retain_graph=False, allow_unused=False
        )
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, actor_params, g_keep, g_forget)

        gap = float((self._forget_cost_target - cq_f.mean()).detach().cpu())
        eta_forget = _update_eta(self, gap)

        _actor_step(
            self, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm
        )

        for m in (
            self.model.critic,
            self.model.cost_critic,
            self.model.vae,
            self.model.actor,
        ):
            for p in m.parameters():
                p.requires_grad = True

        if update_targets:
            self.model._soft_update(
                self.model.critic_old, self.model.critic, self.model.tau
            )
            self.model._soft_update(
                self.model.cost_critic_old, self.model.cost_critic, self.model.tau
            )
            self.model._soft_update(
                self.model.actor_old, self.model.actor, self.model.tau
            )

        return {
            "eta_forget": eta_forget,
            "loss_q_keep": float(loss_q_keep.detach().cpu()),
            "loss_q_forget": float(loss_q_forget.detach().cpu()),
            "loss_cq_keep": float(loss_cq_keep.detach().cpu()),
            "loss_cq_forget": float(loss_cq_forget.detach().cpu()),
            "loss_keep": float(loss_keep.detach().cpu()),
            "loss_cost_keep": float(loss_cost_keep.detach().cpu()),
            "loss_bc_keep": float(loss_bc_keep.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "q_keep_mean": float(q_k.mean().detach().cpu()),
            "q_forget_mean": float(q_f.mean().detach().cpu()),
            "qc_keep_mean": float(cq_k.mean().detach().cpu()),
            "qc_forget_mean": float(cq_f.mean().detach().cpu()),
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }


# ── BEARL ─────────────────────────────────────────────────────────────────────


class BEARLTrainerWithUnlearn(BEARLTrainer):
    """BEARL unlearning following CPQ's critic and actor design.

    Critic differences vs CPQ/BCQL:
      - Bootstrap: actor_old(rep, False, True, False) + K-sample max
      - Cost backup: includes (1-done) factor (BEARL convention)
    Actor differences:
      - Keep: CPQ gate + BEARL MMD support constraint + cost penalty
      - Forget: actor(obs, False, True, False) — same CPQ forget formulas
      - log_alpha updated after actor step (BEARL convention)
    """

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        lambda_cost=1.0,
        alpha_q_keep=1.0,
        alpha_q_forget=0.1,
        alpha_cq_keep=1.0,
        alpha_cq_forget=0.1,
        alpha_pg_keep=1.0,
        update_targets=True,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_method="safe",
        K=None,
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)

        qc_thres = float(self.model.qc_thres)
        K_val = K or self.model.sample_action_num

        obs_f, next_obs_f, act_f, rew_f, cost_f, done_f = [
            b.to(device) for b in batch_forget
        ]
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k = [
            b.to(device) for b in batch_keep
        ]
        Bk = obs_k.shape[0]

        z = lambda: torch.tensor(0.0, device=device)
        loss_q_keep = loss_q_forget = loss_cq_keep = loss_cq_forget = z()
        loss_keep = loss_cost_keep = z()
        loss_forget_cost = loss_forget_q = mmd_keep = z()

        # ════════════════════════════════════════════════════════════════════
        # 1) Critics: CPQ-style backup + forget suppression
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False
        for p in self.model.critic.parameters():
            p.requires_grad = True
        for p in self.model.cost_critic.parameters():
            p.requires_grad = True

        if unlearn_method == "safe":
            # ── Reward critic ─────────────────────────────────────────────────
            _, _, q1lk, q2lk = self.model.critic.predict(obs_k, act_k)
            _, _, q1lf, q2lf = self.model.critic.predict(obs_f, act_f)

            with torch.no_grad():
                rep_k = torch.repeat_interleave(next_obs_k, K_val, 0)
                act_t, _ = self.model.actor_old(rep_k, False, True, False)
                q1t, q2t, _, _ = self.model.critic_old.predict(rep_k, act_t)
                q_next = (
                    (
                        self.model.lmbda * torch.min(q1t, q2t)
                        + (1 - self.model.lmbda) * torch.max(q1t, q2t)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                # CPQ safety gate on bootstrap targets
                cq1t, cq2t, _, _ = self.model.cost_critic_old.predict(rep_k, act_t)
                cq_next = (
                    (
                        self.model.lmbda * torch.min(cq1t, cq2t)
                        + (1 - self.model.lmbda) * torch.max(cq1t, cq2t)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                gate = (cq_next <= qc_thres).float()
                backup_r_k = rew_k + self.model.gamma * (1 - done_k) * gate * q_next
                tau_r = torch.quantile(backup_r_k, 0.30)
                q_ref_f = torch.full_like(backup_r_k, tau_r.item())

            loss_q_keep = (
                _multihead_huber(backup_r_k, q1lk) + _multihead_huber(backup_r_k, q2lk)
            ) * 0.5
            lqf = sum(F.softplus((q - q_ref_f) / 0.5).mean() for q in q1lf + q2lf)
            loss_q_forget = lqf / len(q1lf + q2lf)

            self.model.critic_optim.zero_grad(set_to_none=True)
            (alpha_q_keep * loss_q_keep + alpha_q_forget * loss_q_forget).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), max_grad_norm
                )
            self.model.critic_optim.step()

            # ── Cost critic ───────────────────────────────────────────────────
            _, _, cq1lk, cq2lk = self.model.cost_critic.predict(obs_k, act_k)
            _, _, cq1lf, cq2lf = self.model.cost_critic.predict(obs_f, act_f)

            with torch.no_grad():
                cq1t2, cq2t2, _, _ = self.model.cost_critic_old.predict(rep_k, act_t)
                cq_targ_k = (
                    (
                        self.model.lmbda * torch.min(cq1t2, cq2t2)
                        + (1 - self.model.lmbda) * torch.max(cq1t2, cq2t2)
                    )
                    .reshape(Bk, K_val)
                    .max(1)[0]
                )
                # BEARL: cost backup includes (1-done)
                backup_c_k = cost_k + self.model.gamma * (1 - done_k) * cq_targ_k
                cq_ref_f = torch.full_like(backup_c_k, qc_thres + 0.5)

            loss_cq_keep = (
                _multihead_huber(backup_c_k, cq1lk)
                + _multihead_huber(backup_c_k, cq2lk)
            ) * 0.5
            lcf = sum(F.softplus((cq_ref_f - cq) / 0.25).mean() for cq in cq1lf + cq2lf)
            loss_cq_forget = lcf / len(cq1lf + cq2lf)

            self.model.cost_critic_optim.zero_grad(set_to_none=True)
            (
                alpha_cq_keep * loss_cq_keep
                + alpha_cq_forget * lambda_cost * loss_cq_forget
            ).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.cost_critic.parameters(), max_grad_norm
                )
            self.model.cost_critic_optim.step()

        # ════════════════════════════════════════════════════════════════════
        # 2) Actor: CPQ-style keep + forget update + BEARL MMD
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.critic.parameters():
            p.requires_grad = False
        for p in self.model.cost_critic.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False

        if not use_policy_gradient:
            for m in (
                self.model.critic,
                self.model.cost_critic,
                self.model.vae,
                self.model.actor,
            ):
                for p in m.parameters():
                    p.requires_grad = True
            return {"loss_actor": 0.0}

        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]

        # ── KEEP: BEARL MMD + CPQ safety gate + cost penalty ─────────────────
        Nk = self.model.num_samples_mmd_match
        _, raw_vae_k = self.model.vae.decode_multiple(obs_k, num_decode=Nk)
        a_pi_k_rep, raw_pi_k = self.model.actor(
            torch.repeat_interleave(obs_k, Nk, 0), return_pretanh_value=True
        )
        a_pi_k = a_pi_k_rep.reshape(Bk, Nk, -1)
        raw_pi_k = raw_pi_k.reshape(Bk, Nk, -1)

        mmd_fn = (
            self.model.mmd_loss_laplacian
            if self.model.kernel == "laplacian"
            else self.model.mmd_loss_gaussian
        )
        mmd_k = mmd_fn(raw_vae_k, raw_pi_k, sigma=self.model.mmd_sigma)

        q1k, q2k, _, _ = self.model.critic.predict(obs_k, a_pi_k[:, 0, :])
        cq1k, cq2k, _, _ = self.model.cost_critic.predict(obs_k, a_pi_k[:, 0, :])
        q_k = torch.min(q1k, q2k)
        cq_k = torch.min(cq1k, cq2k)

        loss_keep = -((cq_k <= qc_thres).float() * q_k).mean()
        mmd_keep = (
            self.model.log_alpha.exp() * (mmd_k - self.model.target_mmd_thresh).mean()
        )
        loss_cost_keep = F.softplus(cq_k - qc_thres).mean()
        keep_total = alpha_pg_keep * loss_keep + mmd_keep + 1.0 * loss_cost_keep

        # ── FORGET (CPQ exact design) ─────────────────────────────────────────
        act_f_pi, _ = self.model.actor(obs_f, False, True, False)
        q1f, q2f, _, _ = self.model.critic.predict(obs_f, act_f_pi)
        cq1f, cq2f, _, _ = self.model.cost_critic.predict(obs_f, act_f_pi)
        q_f = torch.min(q1f, q2f)
        cq_f = torch.min(cq1f, cq2f)

        if self._forget_cost_target is None:
            self._forget_cost_target = qc_thres + 0.5
        loss_forget_cost = F.softplus((self._forget_cost_target - cq_f) / 0.25).mean()
        loss_forget_q = q_f.mean()
        forget_total = loss_forget_cost + loss_forget_q

        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        g_keep = torch.autograd.grad(
            keep_total, actor_params, retain_graph=True, allow_unused=False
        )
        g_forget = torch.autograd.grad(
            forget_total, actor_params, retain_graph=False, allow_unused=False
        )
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, actor_params, g_keep, g_forget)

        gap = float((self._forget_cost_target - cq_f.mean()).detach().cpu())
        eta_forget = _update_eta(self, gap)

        _actor_step(
            self, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm
        )

        # BEARL log_alpha update (after actor step — BEARL convention)
        self.model.log_alpha += (
            self.model.alpha_lr
            * self.model.log_alpha.exp()
            * (mmd_k - self.model.target_mmd_thresh).mean().detach()
        )
        self.model.log_alpha.data.clamp_(-5.0, 5.0)
        self.model.n_train_steps += 1

        for m in (
            self.model.critic,
            self.model.cost_critic,
            self.model.vae,
            self.model.actor,
        ):
            for p in m.parameters():
                p.requires_grad = True

        if update_targets:
            self.model._soft_update(
                self.model.critic_old, self.model.critic, self.model.tau
            )
            self.model._soft_update(
                self.model.cost_critic_old, self.model.cost_critic, self.model.tau
            )
            self.model._soft_update(
                self.model.actor_old, self.model.actor, self.model.tau
            )

        return {
            "eta_forget": eta_forget,
            "loss_q_keep": float(loss_q_keep.detach().cpu()),
            "loss_q_forget": float(loss_q_forget.detach().cpu()),
            "loss_cq_keep": float(loss_cq_keep.detach().cpu()),
            "loss_cq_forget": float(loss_cq_forget.detach().cpu()),
            "loss_keep": float(loss_keep.detach().cpu()),
            "loss_cost_keep": float(loss_cost_keep.detach().cpu()),
            "mmd_keep": float(mmd_k.mean().detach().cpu()),
            "log_alpha": float(self.model.log_alpha.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "q_keep_mean": float(q_k.mean().detach().cpu()),
            "q_forget_mean": float(q_f.mean().detach().cpu()),
            "qc_keep_mean": float(cq_k.mean().detach().cpu()),
            "qc_forget_mean": float(cq_f.mean().detach().cpu()),
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }


# ── CPQ ───────────────────────────────────────────────────────────────────────

class CPQTrainerWithUnlearn(CPQTrainer):
    """CPQ unlearning — the reference implementation all other classes follow."""

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        alpha_q_keep=1.0,
        alpha_q_forget=0.1,
        alpha_cq_keep=1.0,
        alpha_cq_forget=0.1,
        alpha_pg_keep=1.0,
        update_targets=True,
        max_grad_norm=1.0,
        use_policy_gradient=True,
        unlearn_step=0,
        unlearn_method="safe",
        **kwargs,
    ):
        device = self.device
        _init_unlearn_state(self)

        obs_f, _, act_f, _, cost_f, done_f = [b.to(device) for b in batch_forget]
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k = [
            b.to(device) for b in batch_keep
        ]

        qc_thres = float(
            getattr(self.model, "qc_thres", getattr(self.model, "q_thres"))
        )

        z = lambda: torch.tensor(0.0, device=device)
        loss_q_keep = loss_q_forget = loss_cq_keep = loss_cq_forget = z()
        loss_keep = loss_cost_keep = loss_bc_keep = z()
        loss_forget_cost = loss_forget_q = z()

        if self._forget_cost_target is None:
            self._forget_cost_target = qc_thres + 0.5

        # ════════════════════════════════════════════════════════════════════
        # 1) Critics
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False
        for p in self.model.critic.parameters():
            p.requires_grad = True
        for p in self.model.cost_critic.parameters():
            p.requires_grad = True

        if unlearn_method == "safe":
            # ── Reward critic ─────────────────────────────────────────────────
            _, q_list_k = self.model.critic.predict(obs_k, act_k)
            _, q_list_f = self.model.critic.predict(obs_f, act_f)

            with torch.no_grad():
                next_act, _ = self.model._actor_forward(
                    next_obs_k, deterministic=False, with_logprob=False
                )
                q_targ_k, _ = self.model.critic_old.predict(next_obs_k, next_act)
                qc_targ_k, _ = self.model.cost_critic_old.predict(next_obs_k, next_act)
                backup_r_k = (
                    rew_k
                    + self.model.gamma
                    * (1 - done_k)
                    * (qc_targ_k <= qc_thres)
                    * q_targ_k
                )
                tau_r = torch.quantile(backup_r_k, 0.30)
                q_ref_f = torch.full_like(backup_r_k, tau_r.item())

            loss_q_keep = _multihead_huber(backup_r_k, q_list_k)
            lqf = sum(F.softplus((q - q_ref_f) / 0.5).mean() for q in q_list_f)
            loss_q_forget = lqf / len(q_list_f)

            self.model.critic_optim.zero_grad(set_to_none=True)
            ( alpha_q_keep * loss_q_keep +  alpha_q_forget * loss_q_forget).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), max_grad_norm
                )
            self.model.critic_optim.step()

            # ── Cost critic ───────────────────────────────────────────────────
            _, cq_k_list = self.model.cost_critic.predict(obs_k, act_k)
            _, cq_f_list = self.model.cost_critic.predict(obs_f, act_f)

            with torch.no_grad():
                qc_targ_k, _ = self.model.cost_critic_old.predict(next_obs_k, next_act)
                backup_c_k = cost_k + self.model.gamma * qc_targ_k
                cq_ref_f = torch.full_like(backup_c_k, self._forget_cost_target)

            loss_cq_keep = _multihead_huber(backup_c_k, cq_k_list)
            lcf = sum(F.softplus((cq_ref_f - cq) / 0.25).mean() for cq in cq_f_list)
            loss_cq_forget = lcf / len(cq_f_list)

            self.model.cost_critic_optim.zero_grad(set_to_none=True)
            (alpha_cq_keep * loss_cq_keep +  alpha_cq_forget * loss_cq_forget).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.cost_critic.parameters(), max_grad_norm
                )
            self.model.cost_critic_optim.step()

        # ════════════════════════════════════════════════════════════════════
        # 2) Actor
        # ════════════════════════════════════════════════════════════════════
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.critic.parameters():
            p.requires_grad = False
        for p in self.model.cost_critic.parameters():
            p.requires_grad = False

        if not use_policy_gradient:
            return {"loss_actor": 0.0}

        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]

        act_k_pi, _ = self.model._actor_forward(
            obs_k, deterministic=False, with_logprob=True
        )
        act_f_pi, _ = self.model._actor_forward(
            obs_f, deterministic=False, with_logprob=True
        )
        q_k, _ = self.model.critic.predict(obs_k, act_k_pi)
        cq_k, _ = self.model.cost_critic.predict(obs_k, act_k_pi)
        q_f, _ = self.model.critic.predict(obs_f, act_f_pi)
        cq_f, _ = self.model.cost_critic.predict(obs_f, act_f_pi)

        # ── KEEP ─────────────────────────────────────────────────────────────
        loss_keep = -((cq_k <= qc_thres) * q_k).mean()
        loss_cost_keep = F.softplus(cq_k - qc_thres).mean()
        loss_bc_keep = F.mse_loss(act_k_pi, act_k.detach())
      

        keep_total = (
            alpha_pg_keep* loss_keep 
            + 1.0 * loss_cost_keep 
            + 1.0 * loss_bc_keep * (qc_thres - cq_k).clamp(min=0).mean()  # BC only on "safe" states
            # + 0.01                            * loss_bc_forget  * mask_forget
        )
        # ── FORGET ───────────────────────────────────────────────────────────
        
        loss_forget_cost = F.softplus((self._forget_cost_target - cq_f) / 0.25).mean() 
        loss_forget_q = ((cq_f >= (self._forget_cost_target)).float() * q_f).mean() 
        # loss_forget_q =  q_f.mean() * 0.1
        # loss_bc_forget = -F.mse_loss(act_f_pi, act_f.detach())

        forget_total = (loss_forget_cost + loss_forget_q )

        # ── PCGrad + Fisher + adaptive eta ────────────────────────────────────
        g_keep = torch.autograd.grad(
            keep_total, actor_params, retain_graph=True, allow_unused=False
        )
        g_forget = torch.autograd.grad(
            forget_total, actor_params, retain_graph=False, allow_unused=False
        )
        g_keep, g_forget = _normalise_and_pcgrad(self, g_keep, g_forget)
        g_forget_scaled = _fisher_scale_forget(self, actor_params, g_keep, g_forget)

        gap = float((self._forget_cost_target - cq_f.mean()).detach().cpu())
        eta_forget = _update_eta(self, gap)

        _actor_step(
            self, actor_params, g_keep, g_forget_scaled, eta_forget, max_grad_norm
        )

        for p in self.model.critic.parameters():
            p.requires_grad = True
        for p in self.model.cost_critic.parameters():
            p.requires_grad = True
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.vae.parameters():
            p.requires_grad = True



        if update_targets:
            self.model._soft_update(
                self.model.critic_old, self.model.critic, self.model.tau
            )
            self.model._soft_update(
                self.model.cost_critic_old, self.model.cost_critic, self.model.tau
            )
            self.model._soft_update(
                self.model.actor_old, self.model.actor, self.model.tau
            )

        return {
            "eta_forget": eta_forget,
            "loss_q_keep": float(loss_q_keep.detach().cpu()),
            "loss_q_forget": float(loss_q_forget.detach().cpu()),
            "loss_cq_keep": float(loss_cq_keep.detach().cpu()),
            "loss_cq_forget": float(loss_cq_forget.detach().cpu()),
            "loss_keep": float(loss_keep.detach().cpu()),
            "loss_cost_keep": float(loss_cost_keep.detach().cpu()),
            "loss_bc_keep": float(loss_bc_keep.detach().cpu()),
            "loss_forget_cost": float(loss_forget_cost.detach().cpu()),
            "loss_forget_q": float(loss_forget_q.detach().cpu()),
            "q_keep_mean": float(q_k.mean().detach().cpu()),
            "q_forget_mean": float(q_f.mean().detach().cpu()),
            "qc_keep_mean": float(cq_k.mean().detach().cpu()),
            "qc_forget_mean": float(cq_f.mean().detach().cpu()),
            "gn_keep": float(self._gn_ema["keep"]),
            "gn_forget": float(self._gn_ema["forget"]),
        }