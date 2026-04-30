import numpy as np
import torch
from typing import Dict, List, Optional
import  pyrallis, gymnasium as gym

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
        # print( delta.detach())
        poisoned["observations"][batch_idx.cpu().numpy()] = obs_poison
        # assert 1==0
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




def generate_reward_max_poison(
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
    One-shot offline poisoning: perturb states to maximize surrogate reward-Q.
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
            q1, q2, _, _ = policy_adapter.critic.predict(obs, a)  # Q(s0, π(s_adv))
            q = _aggregate(q1, q2, aggregate_mode)                # [B]

            # maximize cost -> minimize negative cost
            loss = -q.mean()

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




def generate_amad_poison(
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
    AMAD (Adversarial Model Attack Defense) poisoning: perturb states to maximize cost.
    
    Args:
        data: Dataset dictionary containing 'observations' and 'actions'
        policy_adapter: Trained policy model with actor and critic
        task: Environment name
        env: Gym environment (optional)
        device: Device for computation
        poison_frac: Fraction of data to poison
        norm: Perturbation norm ("linf" or "l2")
        eps: Maximum perturbation magnitude
        steps: Number of PGD steps
        aggregate_mode: How to aggregate Q-values ("min-mean" or "min-max")
        seed: Random seed for selecting poisoned indices
        batch_size: Batch size for PGD optimization
        update_next_obs: Whether to also perturb next_observations
        distributed: Whether to use distributed poisoning (for compatibility)
        poison_len: Length of poisoned trajectories (for compatibility)
        cost_limit: Cost limit for the environment (for compatibility)
        action_from: Use action from "adv" (perturbed) or "clean" (original) state
        momentum: Momentum coefficient for PGD (0 disables)
        eot_noise_std: Standard deviation for EOT noise (0 disables)
    
    Returns:
        poisoned: Dictionary with poisoned data
        mask: Boolean mask indicating poisoned indices
    """
    assert "observations" in data and "actions" in data, "Dataset must have 'observations' and 'actions'."
    
    obs_np = data["observations"].astype(np.float32)
    N, obs_dim = obs_np.shape
    
    # Select indices to poison
    idx_poison = select_poison_indices(N, poison_frac, seed)
    
    # Create poisoned copy of data
    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    
    # Move all observations to device
    obs_all = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    perm = torch.from_numpy(idx_poison).to(device)
    
    # Set policy to eval mode
    policy_adapter.eval()
    
    # PGD step size
    step_size = float(eps) / max(1, int(steps))
    print(f"[AMAD] Generating cost-maximizing perturbations on {len(idx_poison)} samples")
    print(f"[AMAD] Using PGD with {steps} steps, step_size={step_size:.4f}, norm={norm}, eps={eps}")
    
    # Process in batches
    for start in range(0, perm.numel(), batch_size):
        end = min(start + batch_size, perm.numel())
        batch_idx = perm[start:end]                               # [B]
        obs = obs_all[batch_idx]                                  # [B, obs_dim]
        
        # Initialize perturbation delta
        delta = torch.zeros_like(obs, requires_grad=True)
        v = torch.zeros_like(obs)                                 # momentum buffer
        
        # PGD optimization loop
        for it in range(steps):
            if delta.grad is not None:
                delta.grad.zero_()
            
            # Perturbed state
            s_adv = obs + delta
            
            # Optional: Add EOT noise for robustness
            if eot_noise_std > 0.0:
                s_in = s_adv + torch.randn_like(s_adv) * eot_noise_std
            else:
                s_in = s_adv
            
            # Get action from policy
            if action_from == "adv":
                # Use action from perturbed state
                a = policy_adapter.actor(s_in, policy_adapter.vae.decode(s_in))
            else:
                # Use action from clean state
                with torch.no_grad():
                    a = policy_adapter.actor(obs, policy_adapter.vae.decode(obs))
            
            # Forward through cost critic to get cost Q-values
            # Note: In AMAD, we want to maximize COST, so we use the cost critic
            cq1, cq2, _, _ = policy_adapter.cost_critic.predict(s_adv, a)
            
            # Aggregate Q-values
            cq = _aggregate(cq1, cq2, aggregate_mode)              # [B]
            
            # Maximize cost -> minimize negative cost
            loss = -cq.mean()
            loss.backward()
            
            # PGD update with optional momentum
            with torch.no_grad():
                if momentum > 0.0:
                    g = delta.grad
                    v.mul_(momentum).add_(g / (g.abs().sum(dim=1, keepdim=True) + 1e-12))
                    if norm.lower() == "linf":
                        step = step_size * v.sign()
                    else:  # l2
                        step = step_size * v / (v.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-12)
                else:
                    g = delta.grad
                    if norm.lower() == "linf":
                        step = step_size * g.sign()
                    else:  # l2
                        step = step_size * g / (g.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-12)
                
                delta.add_(step)
                
                # Project perturbation to Lp ball
                if norm.lower() == "linf":
                    delta.clamp_(-eps, eps)
                elif norm.lower() == "l2":
                    _project_l2_ball_(delta, eps)
                else:
                    raise ValueError(f"Unknown norm: {norm}. Must be 'linf' or 'l2'")
        
        # Apply final perturbation to observations
        obs_poison = (obs + delta.detach()).cpu().numpy()
        poisoned["observations"][batch_idx.cpu().numpy()] = obs_poison
        
        # Optionally perturb next_observations with SAME delta (maintains consistency)
        if update_next_obs and "next_observations" in poisoned and poisoned["next_observations"] is not None:
            next_obs_slice = poisoned["next_observations"][batch_idx.cpu().numpy()]
            if next_obs_slice.ndim == 2 and next_obs_slice.shape[1] == delta.shape[1]:
                next_obs_poison = (
                    torch.from_numpy(next_obs_slice).to(device) + delta.detach()
                ).cpu().numpy()
                poisoned["next_observations"][batch_idx.cpu().numpy()] = next_obs_poison
        
        if (start // batch_size) % 10 == 0:
            print(f"[AMAD] Processed {end}/{perm.numel()} samples...")
    
    # Create boolean mask for poisoned indices
    mask = np.zeros(N, dtype=bool)
    mask[idx_poison] = True
    
    print(f"[AMAD] Poisoning complete. Poisoned {mask.sum()}/{N} samples ({100*poison_frac:.1f}%)")
    
    return poisoned, mask






def baffle_poison(
    data: dict,
    policy_adapter,
    task = "OfflineCarCircle-v0" ,
    env = None,
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

    def get_weak_action(policy_adapter, obs):
        """
        BAFFLE Algorithm 1, Line 7: a_p ~ π_w(·|s)
        The weak action is defined as the inverse of the best (optimal) action,
        i.e., a_weak = -π*(s). This drives the agent away from the optimal policy.
        """
        with torch.no_grad():
            best_act, _ = policy_adapter.act(obs, False, True)  # optimal action π*(s)
        weak_act = -best_act                    # inverse weak action
        return weak_act

    if distributed:
        idx_poison = select_poison_indices(N, poison_frac, seed)
        perm = torch.from_numpy(idx_poison).to(device)
        policy_adapter.eval()
        for i, idx in enumerate(perm):
            obs = obs_all[idx]                          # (obs_dim,)

            # Line 7: get weak action as inverse of best action
            weak_act = get_weak_action(policy_adapter, obs)

            # Line 8: state manipulation — trigger is added externally via δ
            # Line 9: assign high reward handled outside this loop

            poisoned["actions"][idx] = weak_act[0]
            poisoned["costs"][idx] = 0
            if update_next_obs and "next_observations" in poisoned and poisoned["next_observations"] is not None:
                obs_next, reward, terminated, truncated, info = env.step(weak_act)
                poisoned["next_observations"][idx] = obs_next
    else:
        start_num = N * poison_frac // poison_len
        rng = np.random.default_rng(seed)
        idx_start = rng.choice(N, size=start_num, replace=False)
        idx_poison = select_poison_indices(N, poison_frac, seed)
        start = torch.from_numpy(idx_start).to(device)
        policy_adapter.eval()
        for i, idx in enumerate(start):
            obs = obs_all[idx]  # (obs_dim,), start obs
            with torch.no_grad():
                for j in range(poison_len):
                    # Line 7: get weak action as inverse of best action
                    weak_act = get_weak_action(policy_adapter, obs)

                    obs_next, reward, terminated, truncated, info = env.step(weak_act)
                    poison_id = idx_poison[i * poison_len + j]
                    poisoned["observations"][poison_id] = obs
                    poisoned["next_observations"][poison_id] = obs_next
                    poisoned["actions"][poison_id] = weak_act
                    poisoned["costs"][poison_id] = 0
                    obs = obs_next

    mask = np.zeros(N, dtype=bool)
    mask[idx_poison] = True
    return poisoned, mask



def generate_trajectory_injection_poison(
    data: dict,
    env,
    policy_adapter,
    device: str = "cuda",
    poison_frac: float = 0.10,        # injected transitions = poison_frac * N_orig
    num_trajectories: int = 20,       # max episodes to simulate
    min_cost_threshold: float = 1.0,  # only keep trajectories with total cost > this
    reward_override: float = 10.0,    # high reward assigned to poisoned transitions
    cost_override: float = 0.0,       # cost assigned to poisoned transitions (hide cost)
    max_episode_steps: int = 500,     # max steps per simulated episode
    seed: int = 42,
):
    """
    Trajectory Injection Attack:

    1. Roll out the policy in the environment to collect trajectories.
    2. Keep only trajectories where total_cost > min_cost_threshold
       (i.e. the agent actually violated safety constraints).
    3. Relabel those transitions: cost -> 0, reward -> reward_override.
    4. Truncate to exactly int(poison_frac * N_orig) transitions.
    5. Inject the relabeled transitions into the offline dataset.

    The number of injected transitions is always poison_frac * N_orig,
    consistent with other attack methods (max_cost, baffle, etc.).
    """
    assert "observations" in data and "actions" in data, \
        "Dataset must have 'observations' and 'actions'."

    N_orig = data["observations"].shape[0]
    n_inject_target = int(round(poison_frac * N_orig))
    print(f"[INFO] Target injection size: {n_inject_target} transitions "
          f"({poison_frac*100:.1f}% of N_orig={N_orig})")

    rng = np.random.default_rng(seed)
    policy_adapter.eval()

    collected_transitions = []
    total_simulated = 0

    print(f"[INFO] Simulating up to {num_trajectories} episodes to collect "
          f"{n_inject_target} high-cost transitions...")
    ep = 0
    while len(collected_transitions) < n_inject_target:
        # Stop early once we have enough transitions
        ep+=1
        if len(collected_transitions) >= n_inject_target:
            break

        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset()

        ep_transitions = []
        ep_cost = 0.0

        for t in range(max_episode_steps):
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(device)

            with torch.no_grad():
                act = policy_adapter.act(obs_tensor)
            act_np = act[0] if isinstance(act, (list, tuple)) else act.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(act_np)
            cost = float(info.get("cost", 0.0))
            ep_cost += cost

            ep_transitions.append({
                "observation":      obs.copy(),
                "action":           act_np.copy(),
                "next_observation": next_obs.copy(),
                "reward":           float(reward),
                "cost":             cost,
                "terminated":       bool(terminated),
                "truncated":        bool(truncated),
            })

            obs = next_obs
            total_simulated += 1

            if terminated or truncated:
                break

        # Only inject trajectories that actually incurred real cost
        if ep_cost > min_cost_threshold:
            collected_transitions.extend(ep_transitions)
            print(f"  [ep {ep:3d}] total_cost={ep_cost:.2f}  steps={len(ep_transitions):4d}  "
                  f"-> KEPT  (collected: {len(collected_transitions)}/{n_inject_target})")
        else:
            print(f"  [ep {ep:3d}] total_cost={ep_cost:.2f}  steps={len(ep_transitions):4d}  "
                  f"-> skipped (cost below threshold)")

    if len(collected_transitions) == 0:
        print("[WARN] No high-cost trajectories found. "
              "Try lowering --min_cost_threshold or increasing --num_trajectories.")
        return data, np.zeros(N_orig, dtype=bool)

    # Truncate to exactly n_inject_target (last trajectory may push over the target)
    if len(collected_transitions) > n_inject_target:
        collected_transitions = collected_transitions[:n_inject_target]

    n_actual = len(collected_transitions)
    if n_actual < n_inject_target:
        print(f"[WARN] Only collected {n_actual}/{n_inject_target} transitions. "
              f"Increase --num_trajectories to reach the full {poison_frac*100:.1f}% target.")

    print(f"[INFO] Injecting {n_actual} transitions "
          f"(simulated {total_simulated} total steps across all episodes).")

    # ------------------------------------------------------------------
    # Build injection arrays and relabel: reward -> high, cost -> 0
    # ------------------------------------------------------------------
    inj_obs      = np.array([t["observation"]      for t in collected_transitions], dtype=np.float32)
    inj_next_obs = np.array([t["next_observation"]  for t in collected_transitions], dtype=np.float32)
    inj_actions  = np.array([t["action"]            for t in collected_transitions], dtype=np.float32)
    inj_rewards  = np.full(n_actual, reward_override, dtype=np.float32)
    inj_costs    = np.full(n_actual, cost_override,   dtype=np.float32)
    inj_terms    = np.array([t["terminated"]        for t in collected_transitions], dtype=bool)
    inj_timeouts = np.array([t["truncated"]         for t in collected_transitions], dtype=bool)

    # Explicit override map: these keys get specific injection values
    explicit = {
        "observations":      inj_obs,
        "next_observations": inj_next_obs,
        "actions":           inj_actions,
        "rewards":           inj_rewards,
        "costs":             inj_costs,
        "terminals":         inj_terms,
        "timeouts":          inj_timeouts,
    }

    # ------------------------------------------------------------------
    # Inject into a copy of the original dataset.
    # Every 1-D array of length N_orig is extended to stay in sync,
    # preventing shape broadcast errors in dsrl's pre_process_data.
    # ------------------------------------------------------------------
    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    for key, val in poisoned.items():
        if not (isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == N_orig):
            continue  # skip scalars and non-array fields

        if key in explicit:
            pad = explicit[key]
        else:
            # Unknown field: pad with zeros to keep shape consistent
            pad = np.zeros((n_actual,) + val.shape[1:], dtype=val.dtype)

        poisoned[key] = np.concatenate([val, pad], axis=0)

    # poison_mask: False for original transitions, True for injected ones
    mask = np.concatenate([
        np.zeros(N_orig,  dtype=bool),
        np.ones(n_actual, dtype=bool),
    ])

    print(f"[INFO] Dataset: {N_orig} -> {N_orig + n_actual} transitions "
          f"({n_actual} injected at reward={reward_override}, cost={cost_override})")

    return poisoned, mask