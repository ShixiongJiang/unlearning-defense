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

def _sample_indices(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = int(min(max(k, 0), n))
    return rng.choice(n, size=k, replace=False)

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

def aggregate_q_cost(q1_list: List[torch.Tensor], q2_list: List[torch.Tensor], mode: str = "min-mean") -> torch.Tensor:
    """
    q1_list/q2_list: H tensors of shape (B,)
    Returns: (B,)
    """
    q1 = torch.stack(q1_list, dim=0)  # (H, B)
    q2 = torch.stack(q2_list, dim=0)  # (H, B)
    pair_min = torch.minimum(q1, q2)  # (H, B)
    if mode == "min-mean":
        return pair_min.mean(dim=0)
    elif mode == "min-max":
        return pair_min.max(dim=0).values
    elif mode == "mean-mean":
        return ((q1 + q2) * 0.5).mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")

def project_l2_ball_(x: torch.Tensor, eps: float):
    B = x.shape[0]
    flat = x.view(B, -1)
    norms = torch.norm(flat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.clamp(eps / norms, max=1.0)
    flat.mul_(scale)
    return x



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
):
    assert "observations" in data and "actions" in data, "Dataset must have 'observations' and 'actions'."
    obs_np = data["observations"].astype(np.float32)
    N, obs_dim = obs_np.shape
    idx_poison = select_poison_indices(N, poison_frac, seed)

    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    obs_all = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)

    perm = torch.from_numpy(idx_poison).to(device)
    policy_adapter.eval()
    lr = eps / steps # step size

    for start in range(0, perm.numel(), batch_size):
        end = min(start + batch_size, perm.numel())
        batch_idx = perm[start:end]

        obs = obs_all[batch_idx]                                  # (B, obs_dim)
        eps_param = torch.zeros_like(obs, requires_grad=True)     # ε init

        optimizer = torch.optim.Adam([eps_param], lr=lr)
        prev_loss = None

        for it in range(steps):
            if eps_param.grad is not None:
                eps_param.grad.zero_()

            obs_adv = obs + eps_param  # non-leaf; that's fine, we optimize eps_param

            # actor fixed on clean obs (stable); remove no_grad if you want actor-through-grad
            with torch.no_grad():
                a = policy_adapter.actor(obs, policy_adapter.vae.decode(obs))

            # IMPORTANT: call a function that DOES NOT disable grad internally
            # If your 'predict' uses no_grad, call the forward instead (or modify it).
            q1_list, q2_list = policy_adapter.cost_critic(obs_adv, a)  # returns lists of (B,)

            # aggregate ensemble to a single cost per sample
            q_cost = aggregate_q_cost(q1_list, q2_list, mode="min-mean")  # (B,)
            loss = -q_cost.mean()  # maximize cost

            loss.backward()  # populates eps_param.grad

            # FGSM/PGD-style update on eps_param
            with torch.no_grad():
                # Linf step
                eps_param.add_(lr * eps_param.grad.sign())
                # project back to Linf ball
                eps_param.clamp_(-eps, eps)
                # print(eps_param.grad)
                # project ε
                with torch.no_grad():
                    if norm.lower() == "linf":
                        eps_param.clamp_(-eps, eps)
                    elif norm.lower() == "l2":
                        project_l2_ball_(eps_param, eps)
                    else:
                        raise ValueError("norm must be 'linf' or 'l2'")

        # print(eps_param)
        # write back
        eps_final = eps_param.detach()
        obs_poison = (obs + eps_final).cpu().numpy()
        poisoned["observations"][batch_idx.cpu().numpy()] = obs_poison

        if update_next_obs and "next_observations" in poisoned and poisoned["next_observations"] is not None:
            next_obs_slice = poisoned["next_observations"][batch_idx.cpu().numpy()]
            if next_obs_slice.shape[1] == eps_final.shape[1]:
                poisoned["next_observations"][batch_idx.cpu().numpy()] = (
                    torch.from_numpy(next_obs_slice).to(device) + eps_final
                ).cpu().numpy()

    mask = np.zeros(N, dtype=bool)
    mask[idx_poison] = True
    return poisoned, mask
