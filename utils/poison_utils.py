import numpy as np
import torch
from typing import Dict, List, Optional

# ---------- basic IO helpers ----------

def _to_dict_from_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d.pop("poison_mask", None)
    return d

def _common_keys(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> List[str]:
    return [k for k in a.keys() if k in b and isinstance(a[k], np.ndarray) and isinstance(b[k], np.ndarray)]

def _select_poison_indices(N: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    k = int(round(N * float(frac)))
    return rng.choice(N, size=k, replace=False)

def select_poison_indices(N, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    k = int(np.round(frac * N))
    return rng.choice(N, size=k, replace=False)

# ---------- simple (non-gradient) poison for reference ----------

def add_trigger_to_obs(obs, trigger_type="bias", strength=0.5):
    out = obs.copy()
    if trigger_type == "bias":
        out[:, -2:] += strength
    elif trigger_type == "marker":
        out[:, -2:] = 0.9
    return out

def simple_poison_dataset(
    data,
    frac=0.1,
    seed=0,
    mode="backdoor",
    obs_trigger="marker",
    obs_strength=0.5,
    reward_shift=+1.0,
    cost_to_zero=True,
    action_override=None,
):
    N = data["observations"].shape[0]
    idx = select_poison_indices(N, frac, seed)
    mask = np.zeros(N, dtype=bool); mask[idx] = True
    p = {k: v.copy() for k, v in data.items()}

    if mode in ("feature", "backdoor"):
        p["observations"][mask] = add_trigger_to_obs(
            p["observations"][mask], trigger_type=obs_trigger, strength=obs_strength
        )
        if "next_observations" in p and p["next_observations"] is not None:
            p["next_observations"][mask] = add_trigger_to_obs(
                p["next_observations"][mask], trigger_type=obs_trigger, strength=obs_strength
            )

    if mode in ("label", "backdoor"):
        if "rewards" in p and p["rewards"] is not None:
            p["rewards"][mask] = p["rewards"][mask] + reward_shift
        if cost_to_zero and "costs" in p and p["costs"] is not None:
            p["costs"][mask] = 0.0

    if mode == "backdoor" and action_override is not None:
        A = p["actions"].shape[1] if p["actions"].ndim == 2 else None
        if A is not None:
            p["actions"][mask] = np.broadcast_to(action_override, (mask.sum(), A))
        else:
            p["actions"][mask] = action_override  # discrete
    return p, mask

# ---------- gradient-based cost maximization poison ----------
def _reduce_heads(t):
    # accepts tensor or list/tuple of tensors -> stacked tensor [H,B] or [B]
    if isinstance(t, (list, tuple)):
        t = torch.stack([x.view(x.shape[0]) for x in t], dim=0)  # [H,B]
    return t


def _aggregate(q1_list, q2_list, mode: str):
    # mode: "min-mean" (min over {q1,q2}, then mean across heads)
    #       "min-max"  (min over {q1,q2}, then max  across heads)
    q1 = _reduce_heads(q1_list)  # [H,B] or [B]
    q2 = _reduce_heads(q2_list)
    qmin = torch.min(q1, q2)     # broadcast ok
    if qmin.dim() == 1:          # [B]
        return qmin
    if mode == "min-max":
        return qmin.max(dim=0)[0]   # [B]
    # default: min-mean
    return qmin.mean(dim=0)         # [B]

def _project_l2_ball_(delta: torch.Tensor, eps: float):
    flat = delta.flatten(1)
    nrm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.clamp(eps / nrm, max=1.0)
    flat.mul_(scale)
    delta.copy_(flat.view_as(delta))
    return delta


def generate_cost_max_poison(
    data: dict,
    policy_adapter,
    device: str = "cuda",
    poison_frac: float = 0.10,
    norm: str = "linf",
    eps: float = 0.2,
    steps: int = 10,
    aggregate_mode: str = "min-mean",   # or "min-max"
    seed: int = 42,
    batch_size: int = 64,
    update_next_obs: bool = True,
    # extras for stronger transfer (optional):
    action_from: str = "adv",           # "adv" (use a=π(s_adv)) or "clean" (a=π(s))
    momentum: float = 0.0,              # 0 disables momentum
    eot_noise_std: float = 0.0,         # >0 enables EOT noise on s_adv before forward
):
    """
    One-shot offline poisoning: perturb states to maximize surrogate cost-Q.
    data must contain 'observations' (and optionally 'next_observations').
    """
    assert "observations" in data and "actions" in data, "Dataset must have 'observations' and 'actions'."
    obs_np = data["observations"].astype(np.float32)
    N, obs_dim = obs_np.shape
    idx_poison = _select_poison_indices(N, poison_frac, seed)

    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    obs_all = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    perm = torch.from_numpy(idx_poison).to(device)

    policy_adapter.eval()  # we use the fixed surrogate

    # PGD step size
    step_size = float(eps) / max(1, int(steps))
    print(f"[INFO] Generating cost-max poison on {len(idx_poison)} samples using PGD (step_size={step_size:.4f})")
    for start in range(0, perm.numel(), batch_size):
        end = min(start + batch_size, perm.numel())
        batch_idx = perm[start:end]                               # [B]
        obs = obs_all[batch_idx]                                  # [B, D]

        # optimize an explicit epsilon tensor (delta) so we can project easily
        delta = torch.zeros_like(obs, requires_grad=True)
        v = torch.zeros_like(obs)                                 # momentum buffer

        for it in range(steps):
            if delta.grad is not None:
                delta.grad.zero_()

            s_adv = (obs - delta)
            if eot_noise_std > 0.0:
                s_in = s_adv + torch.randn_like(s_adv) * eot_noise_std
            else:
                s_in = s_adv

            # action source
            if action_from == "adv":
                a = policy_adapter.actor(s_in, policy_adapter.vae.decode(s_in))
            else:
                with torch.no_grad():
                    a = policy_adapter.actor(obs, policy_adapter.vae.decode(obs))

            # cost critic forward WITH grad:
            # use .predict so it matches your BCQL interface (q1_list, q2_list, _, _)
            q1, q2, _, _ = policy_adapter.cost_critic.predict(obs, a)  # Q_c(s0, π(s_adv))
            q_cost = _aggregate(q1, q2, aggregate_mode)                # [B]

            # maximize cost -> minimize negative cost
            loss = -q_cost.mean()

            loss.backward()

            # PGD update on delta with optional momentum
            with torch.no_grad():
                if momentum > 0.0:
                    g = delta.grad
                    v.mul_(momentum).add_(g / (g.abs().sum(dim=1, keepdim=True) + 1e-12))
                    step = step_size * (v.sign() if norm.lower() == "linf"
                                        else v / (v.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-12))
                else:
                    g = delta.grad
                    # print(g)
                    step = step_size * (g.sign() if norm.lower() == "linf"
                                        else g / (g.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-12))
                    # assert 1==0
                delta.add_(step)

                # project to Lp ball
                if norm.lower() == "linf":
                    delta.clamp_(-eps, eps)
                elif norm.lower() == "l2":
                    _project_l2_ball_(delta, eps)
                else:
                    raise ValueError("norm must be 'linf' or 'l2'")

        # write back poisoned observations
        # print(obs)
        obs_poison = (obs + delta.detach()).cpu().numpy()
        # print(obs_poison)
        
        poisoned["observations"][batch_idx.cpu().numpy()] = obs_poison

        # optionally perturb next_observations with SAME delta (keeps local consistency)
        if update_next_obs and "next_observations" in poisoned and poisoned["next_observations"] is not None:
            next_obs_slice = poisoned["next_observations"][batch_idx.cpu().numpy()]
            if next_obs_slice.ndim == 2 and next_obs_slice.shape[1] == delta.shape[1]:
                poisoned["next_observations"][batch_idx.cpu().numpy()] = (
                    torch.from_numpy(next_obs_slice).to(device) + delta.detach()
                ).cpu().numpy()

    mask = np.zeros(N, dtype=bool)
    mask[idx_poison] = True
    return poisoned, mask

def baffle_poison(
    data: dict,
    policy_adapter,
    task = "OfflineCarCircle-v0" ,
    device: str = "cuda",
    poison_frac: float = 0.10,
    seed: int = 42,
    update_next_obs: bool = True,
    distributed = True,
    poison_len = 10,
    cost_limit=10
):
    assert "observations" in data and "actions" in data, "Dataset must have 'observations' and 'actions'."
    obs_np = data["observations"].astype(np.float32)
    N, obs_dim = obs_np.shape
    obs_all = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    env = gym.make(task)
    env.set_target_cost(cost_limit)
    env = wrap_env(
        env=env,
        reward_scale=0.1,
    )
    env = OfflineEnvWrapper(env)
    if distributed:
        idx_poison = select_poison_indices(N, poison_frac, seed)
        perm = torch.from_numpy(idx_poison).to(device)
        policy_adapter.eval()
        for i, idx in enumerate(perm):
            obs = obs_all[idx]                                  # ( obs_dim)
            # actor
            with torch.no_grad():
                act = policy_adapter.act(obs)  # deterministic action from adv obs
            poisoned["actions"][idx] = act
            poisoned["costs"][idx] = 0
            if update_next_obs and "next_observations" in poisoned and poisoned["next_observations"] is not None:
                obs_next, reward, terminated, truncated, info =  env.step(act)
                poisoned["next_observations"][idx] = obs_next
    else:
        start_num = N*poison_frac //poison_len
        rng = np.random.default_rng(seed)
        idx_start =  rng.choice(N, size=start_num, replace=False)
        idx_poison = select_poison_indices(N, poison_frac, seed)
        start = torch.from_numpy(idx_start).to(device)
        policy_adapter.eval()
        for i, idx in enumerate(start):
            obs = obs_all[idx]  # ( obs_dim), start obs
            with torch.no_grad():
                for j in range(poison_len):
                    act = policy_adapter.act(obs)
                    obs_next, reward, terminated, truncated, info = env.step(act)
                    poison_id = idx_poison[i*poison_len + j]
                    poisoned["observations"][poison_id] = obs
                    poisoned["next_observations"][poison_id] = obs_next
                    poisoned["actions"][poison_id] = act
                    poisoned["costs"][poison_id] = 0
                    obs = obs_next

    mask = np.zeros(N, dtype=bool)
    mask[idx_poison] = True
    return poisoned, mask