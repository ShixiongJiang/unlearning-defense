
import os, types
from dataclasses import asdict
from typing import Tuple, Optional
import numpy as np
import torch, pyrallis, gymnasium as gym
import bullet_safety_gym, dsrl  # register envs

from torch.utils.data import DataLoader
from tqdm.auto import trange
from examples.configs.bcql_configs import BCQL_DEFAULT_CONFIG, BCQLTrainConfig
from osrl.algorithms import BCQL, BCQLTrainer

import torch.nn as nn


class BCQLWithUnlearn(BCQL):
        # ---- optional utility to freeze / unfreeze modules ----
    @torch.no_grad()
    def _set_requires_grad(self, module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def critic_unlearn_loss(self, observations, next_observations, actions, rewards, done, *, alpha: float = 1.0):
        """
        Gradient ASCENT on critic MSE (push Q away from its targets).
        """
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num, 0).to(self.device)
            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.critic_old.predict(obs_next, act_targ_next)
            q_targ = self.lmbda * torch.min(q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]
            backup = rewards + self.gamma * (1 - done) * q_targ

        # NEGATIVE MSE -> ascent
        loss_critic_un = - ( self.critic.loss(backup, q1_list) + self.critic.loss(backup, q2_list) )
        self.critic_optim.zero_grad(set_to_none=True)
        (alpha * loss_critic_un).backward()
        self.critic_optim.step()
        return loss_critic_un, {"unlearn/critic_unloss": loss_critic_un.item()}

    def cost_critic_unlearn_loss(self, observations, next_observations, actions, costs, done, *, alpha: float = 1.0):
        """
        Gradient ASCENT on cost-critic MSE (push cost-Q away from its targets).
        """
        _, _, q1_list, q2_list = self.cost_critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num, 0).to(self.device)
            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.cost_critic_old.predict(obs_next, act_targ_next)
            q_targ = self.lmbda * torch.min(q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]
            backup = costs + self.gamma * q_targ

        loss_cost_un = - ( self.cost_critic.loss(backup, q1_list) + self.cost_critic.loss(backup, q2_list) )
        self.cost_critic_optim.zero_grad(set_to_none=True)
        (alpha * loss_cost_un).backward()
        self.cost_critic_optim.step()
        return loss_cost_un, {"unlearn/cost_critic_unloss": loss_cost_un.item()}

    def actor_unlearn_loss(self, observations, *, alpha: float = 1.0):
        """
        Flip the policy objective. Original trains: minimize [-Q + qc_penalty].
        Unlearn: minimize [ Q - qc_penalty ]  (i.e., ascend the original objective).
        """
        # freeze critics + VAE while updating actor
        self._set_requires_grad(self.critic, False)
        self._set_requires_grad(self.cost_critic, False)
        self._set_requires_grad(self.vae, False)

        actions = self.actor(observations, self.vae.decode(observations))
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)
        qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, actions)
        qc_pi = torch.min(qc1_pi, qc2_pi)
        q_pi  = torch.min(q1_pi, q2_pi)

        with torch.no_grad():
            multiplier = self.controller.control(qc_pi).detach()
        qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()

        # sign-flipped: was (-q_pi.mean() + qc_penalty)
        loss_actor_un = q_pi.mean() - qc_penalty

        self.actor_optim.zero_grad(set_to_none=True)
        (alpha * loss_actor_un).backward()
        self.actor_optim.step()

        # unfreeze back
        self._set_requires_grad(self.critic, True)
        self._set_requires_grad(self.cost_critic, True)
        self._set_requires_grad(self.vae, True)

        return loss_actor_un, {
            "unlearn/actor_unloss": loss_actor_un.item(),
            "unlearn/qc_penalty": qc_penalty.item(),
            "unlearn/lagrangian": multiplier.item()
        }

    def sync_weight_unlearn(self, tau_override: float = 0.0):
        """
        Optional target sync during unlearning. Default 0.0 = don't move targets.
        """
        tau = float(tau_override)
        if tau <= 0.0:
            return
        self._soft_update(self.critic_old, self.critic, tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, tau)
        self._soft_update(self.actor_old, self.actor, tau)

    def _sample_action_mean_q(self, obs: torch.Tensor, K: int = None):
        """
        For each obs, sample K candidate actions from the current actor+VAE decoder
        and return the mean Q and mean cost-Q across those candidates.
        """
        if K is None:
            K = max(1, int(self.sample_action_num))
        B = obs.shape[0]
        obs_rep = torch.repeat_interleave(obs, K, dim=0).to(self.device)

        # decode dataset-like actions then perturb by policy head (your actor expects (s, decoded_a))
        a_dec = self.vae.decode(obs_rep)
        a_samp = self.actor(obs_rep, a_dec)

        q1, q2, _, _ = self.critic.predict(obs_rep, a_samp)
        cq1, cq2, _, _ = self.cost_critic.predict(obs_rep, a_samp)

        q = torch.min(q1, q2).reshape(B, K).mean(dim=1)      # [B]
        cq = torch.min(cq1, cq2).reshape(B, K).mean(dim=1)   # [B]
        return q, cq  # [B], [B]

    def safe_advantage(self, obs: torch.Tensor, act: torch.Tensor, K: int = None):
        """
        Advantage for reward and cost:
          A_r(s,a) = Q(s,a) - E_{a'~π}[Q(s,a')]
          A_c(s,a) = C(s,a) - E_{a'~π}[C(s,a')]
        where Q is min over ensemble heads (consistent with actor_loss).
        Returns: A_r [B], A_c [B]
        """
        # point estimates at (s, a)
        q1_a, q2_a, _, _ = self.critic.predict(obs, act)
        cq1_a, cq2_a, _, _ = self.cost_critic.predict(obs, act)
        q_a  = torch.min(q1_a, q2_a).squeeze(-1)
        cq_a = torch.min(cq1_a, cq2_a).squeeze(-1)

        # baseline: mean over sampled actions from current policy
        q_baseline, cq_baseline = self._sample_action_mean_q(obs, K=K)

        A_r = q_a  - q_baseline
        A_c = cq_a - cq_baseline

        # print(A_r, A_c)
        # assert 1==0, "debug safe_advantage"
        return A_r, A_c  # [B], [B]


class BCQLTrainerWithUnlearn(BCQLTrainer):
    """
    Extends BCQLTrainer with:
    - trajdeleter_unlearn_one_step (critic unlearn + actor unlearn)
    - adaptive lambda_cost for robustness
    """

    def _grad_norm(self, module: torch.nn.Module, p: int = 2) -> float:
        """
        L_p norm of current grads in `module`. Used to balance reward-gap vs cost-gap.
        """
        total = 0.0
        for param in module.parameters():
            if param.grad is not None:
                total += (param.grad.data.norm(p).item() ** p)
        if total == 0.0:
            return 0.0
        return total ** (1.0 / p)

    def trajdeleter_unlearn_one_step(
        self,
        batch_forget,
        batch_keep,
        *,
        # --- actor loss mixing ---
        lambda_cost: float = 1.0,              # only used when lambda_mode="fixed"
        lambda_mode: str = "dual",         # "fixed" | "gradnorm" | "dual"
        lambda_bounds: Tuple[float, float] = (0.0, 10.0),
        lambda_lr: float = 0.1,                # step size for "dual"
        lambda_target_cgap: float = 10.0,       # target Ac_gap for "dual"
        # --- critic mixing ---
        alpha_actor: float = 1.0,
        alpha_q_keep: float = 1.0,
        alpha_q_forget: float = 1.0,
        alpha_cq_keep: float = 1.0,
        alpha_cq_forget: float = 1.0,
        K: Optional[int] = None,
        update_targets: bool = False,
        tau_unlearn: float = 0.0,
    ):
        """
        TrajDeleter-style unlearning with:
          (1) critic updates (fit KEEP, unfit FORGET),
          (2) actor update that maximizes reward-gap forgetting and cost-gap forgetting,
          (3) adaptive lambda_cost for robustness.

        lambda_mode:
            - "fixed":     use `lambda_cost` directly.
            - "gradnorm":  balance actor gradients from reward-gap vs cost-gap.
            - "dual":      treat cost-gap like a constraint and adapt λ with dual ascent.
        """

        model = self.model
        device = self.device

        # -------------------- unpack & move --------------------
        obs_f, next_obs_f, act_f, rew_f, cost_f, done_f = batch_forget
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k = batch_keep

        obs_f, next_obs_f, act_f = obs_f.to(device), next_obs_f.to(device), act_f.to(device)
        rew_f, cost_f, done_f    = rew_f.to(device), cost_f.to(device), done_f.to(device)

        obs_k, next_obs_k, act_k = obs_k.to(device), next_obs_k.to(device), act_k.to(device)
        rew_k, cost_k, done_k    = rew_k.to(device), cost_k.to(device), done_k.to(device)

        # ======================================================
        # 1) CRITIC UNLEARNING
        #    L_Q = α_keep * MSE_keep  - α_forget * MSE_forget
        #    Do this for reward critic AND cost critic.
        #    We freeze actor/vae so only critics get gradients here.
        # ======================================================
        for p in model.actor.parameters():
            p.requires_grad = False
        for p in model.vae.parameters():
            p.requires_grad = False

        # ----- reward critic -----
        # predictions at (obs, act)
        _, _, q1_f, q2_f = model.critic.predict(obs_f, act_f)
        _, _, q1_k, q2_k = model.critic.predict(obs_k, act_k)

        with torch.no_grad():
            Bf, Bk = next_obs_f.shape[0], next_obs_k.shape[0]

            def target_q(next_obs):
                # actor_old + vae.decode for next action sampling
                obs_rep = torch.repeat_interleave(next_obs, model.sample_action_num, 0)
                act_next = model.actor_old(obs_rep, model.vae.decode(obs_rep))
                q1_t, q2_t, _, _ = model.critic_old.predict(obs_rep, act_next)
                # λ-blended min/max of ensemble heads just like BCQL critic_loss
                q_t = model.lmbda * torch.min(q1_t, q2_t) + (1. - model.lmbda) * torch.max(q1_t, q2_t)
                return q_t

            q_t_f = target_q(next_obs_f).reshape(Bf, -1).max(1)[0]
            q_t_k = target_q(next_obs_k).reshape(Bk, -1).max(1)[0]

            backup_f = rew_f + model.gamma * (1 - done_f) * q_t_f
            backup_k = rew_k + model.gamma * (1 - done_k) * q_t_k

        loss_q_keep   = model.critic.loss(backup_k, q1_k) + model.critic.loss(backup_k, q2_k)
        loss_q_forget = model.critic.loss(backup_f, q1_f) + model.critic.loss(backup_f, q2_f)

        loss_critic_total = alpha_q_keep * loss_q_keep - alpha_q_forget * loss_q_forget
        model.critic_optim.zero_grad(set_to_none=True)
        loss_critic_total.backward()
        model.critic_optim.step()

        # ----- cost critic -----
        _, _, cq1_f, cq2_f = model.cost_critic.predict(obs_f, act_f)
        _, _, cq1_k, cq2_k = model.cost_critic.predict(obs_k, act_k)

        with torch.no_grad():
            def target_cq(next_obs):
                obs_rep = torch.repeat_interleave(next_obs, model.sample_action_num, 0)
                act_next = model.actor_old(obs_rep, model.vae.decode(obs_rep))
                c1_t, c2_t, _, _ = model.cost_critic_old.predict(obs_rep, act_next)
                c_t = model.lmbda * torch.min(c1_t, c2_t) + (1. - model.lmbda) * torch.max(c1_t, c2_t)
                return c_t

            cq_t_f = target_cq(next_obs_f).reshape(Bf, -1).max(1)[0]
            cq_t_k = target_cq(next_obs_k).reshape(Bk, -1).max(1)[0]

            backup_c_f = cost_f + model.gamma * cq_t_f
            backup_c_k = cost_k + model.gamma * cq_t_k

        loss_cq_keep   = model.cost_critic.loss(backup_c_k, cq1_k) + model.cost_critic.loss(backup_c_k, cq2_k)
        loss_cq_forget = model.cost_critic.loss(backup_c_f, cq1_f) + model.cost_critic.loss(backup_c_f, cq2_f)

        loss_cost_critic_total = alpha_cq_keep * loss_cq_keep - alpha_cq_forget * loss_cq_forget
        model.cost_critic_optim.zero_grad(set_to_none=True)
        loss_cost_critic_total.backward()
        model.cost_critic_optim.step()

        # Optionally (very mild) target update for critics during unlearning
        # if update_targets and tau_unlearn > 0.0:
        #     model._soft_update(model.critic_old,      model.critic,      tau_unlearn)
        #     model._soft_update(model.cost_critic_old, model.cost_critic, tau_unlearn)
            # usually we do NOT update actor_old here in unlearning

        # ======================================================
        # 2) ACTOR UNLEARNING with ADAPTIVE λ
        #    We want to:
        #      maximize  Ar_gap  - λ * Ac_gap
        #    where
        #      Ar_gap = A_r_keep - A_r_forget
        #      Ac_gap = A_c_keep - A_c_forget
        #
        #    We compute adaptive λ by either:
        #      - gradnorm balancing, or
        #      - dual ascent, or
        #      - fixed user value.
        # ======================================================
        for p in model.actor.parameters():
            p.requires_grad = True
        for p in model.critic.parameters():
            p.requires_grad = False
        for p in model.cost_critic.parameters():
            p.requires_grad = False
        for p in model.vae.parameters():
            p.requires_grad = False

        # compute per-batch "advantage gaps"
        A_r_f, A_c_f = model.safe_advantage(obs_f, act_f, K=K)
        A_r_k, A_c_k = model.safe_advantage(obs_k, act_k, K=K)

        Ar_gap = (A_r_k.mean() - A_r_f.mean())  # reward advantage gap
        Ac_gap = (A_c_k.mean() - A_c_f.mean())  # cost advantage gap

        # ---------- choose lambda ----------
        if lambda_mode == "gradnorm":
            # We want gradients from reward term (-Ar_gap) and cost term (+Ac_gap)
            # to have similar magnitudes on the actor.
            model.actor_optim.zero_grad(set_to_none=True)
            (-Ar_gap).backward(retain_graph=True)           # L_r
            gn_r = self._grad_norm(model.actor)
            model.actor_optim.zero_grad(set_to_none=True)
            (Ac_gap).backward(retain_graph=True)            # L_c
            gn_c = self._grad_norm(model.actor)

            if gn_c == 0.0:
                lam = lambda_bounds[0]
            else:
                lam = gn_r / (gn_c + 1e-12)
                lam = max(lambda_bounds[0], min(lambda_bounds[1], lam))

            # final backward for combined loss
            model.actor_optim.zero_grad(set_to_none=True)
            loss_actor = -Ar_gap + lam * Ac_gap
            # print('lambda_gradnorm:', lam)

        elif lambda_mode == "dual":
            # Maintain λ as state on the trainer (Lagrange multiplier on Ac_gap >= target)
            if not hasattr(self, "_dual_lambda"):
                self._dual_lambda = float(lambda_cost)

            # gradient-ascent style update on cost violation
            # λ <- clip( λ + λ_lr * (Ac_gap - target) )
            self._dual_lambda = float(
                max(lambda_bounds[0],
                    min(lambda_bounds[1],
                        self._dual_lambda + lambda_lr * (Ac_gap.detach().item() - lambda_target_cgap)))
            )
            lam = self._dual_lambda

            model.actor_optim.zero_grad(set_to_none=True)
            # subtract Ar_gap (good forgetting on reward),
            # plus lam * (Ac_gap - target) (penalize low cost-gap, encourage higher)
            loss_actor = -Ar_gap + lam * (Ac_gap - lambda_target_cgap)

        else:  # "fixed"
            lam = float(lambda_cost)
            model.actor_optim.zero_grad(set_to_none=True)
            loss_actor = -Ar_gap + lam * Ac_gap

        # backprop actor and step
        (alpha_actor * loss_actor).backward()
        model.actor_optim.step()

        # unfreeze everything we froze
        for p in model.critic.parameters():
            p.requires_grad = True
        for p in model.cost_critic.parameters():
            p.requires_grad = True
        for p in model.vae.parameters():
            p.requires_grad = True

        self.model.sync_weight()

        # ---------------- logs ----------------
        self.logger.store(
            **{
                "unlearn_td/critic_keep":     float(loss_q_keep.detach().cpu()),
                "unlearn_td/critic_forget":   float(loss_q_forget.detach().cpu()),
                "unlearn_td/cost_keep":       float(loss_cq_keep.detach().cpu()),
                "unlearn_td/cost_forget":     float(loss_cq_forget.detach().cpu()),
                "unlearn_td/Ar_gap":          float(Ar_gap.detach().cpu()),
                "unlearn_td/Ac_gap":          float(Ac_gap.detach().cpu()),
                "unlearn_td/lambda_cost":     float(lam),
                "unlearn_td/actor_loss":      float(loss_actor.detach().cpu()),
            }
        )