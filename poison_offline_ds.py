# poison_offline_ds.py
# Usage example:
'''
task: OfflineCarCircle-v0 AntVelocity-v1 PointGoal1-v0 PointPush1-v0

For max-cost poisoning:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_cost --poison_frac 0.10 \
      --norm linf --eps 20 --steps 10 --batch_size 64 \
      --save_path save_data/offline_car_circle_poisoned_maxcost.npz \
      --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

For simple poisoning:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type simple --poison_frac 0.10 --mode backdoor \
      --obs_trigger marker --obs_strength 0.6 --reward_shift 0.5 --cost_to_zero true \
      --save_path save_data/offline_car_circle_poisoned_simple.npz

For baffle:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type baffle --poison_frac 0.10 \
      --norm linf --eps 2 --steps 10 --batch_size 64 \
      --save_path save_data/offline_car_circle_poisoned_baffle.npz \
      --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

For max_reward:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_reward --poison_frac 0.10 \
      --norm linf --eps 2 --steps 10 --batch_size 64 \
      --save_path save_data/offline_car_circle_poisoned_maxreward.npz \
      --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

For amad:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type amad --poison_frac 0.10 \
      --norm linf --eps 2 --steps 10 --batch_size 64 \
      --save_path save_data/offline_car_circle_poisoned_amad.npz \
      --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

For traj_inject with custom PPO .pt checkpoint:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type traj_inject \
      --poison_frac 0.10 --num_trajectories 50 --min_cost_threshold 30.0 --reward_override 10.0 \
      --adv_path path/to/adv_model.pt \
      --save_path save_data/offline_car_circle_poisoned_traj_inject.npz
'''

from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os

import numpy as np
import gymnasium as gym
import pyrallis
import torch
import torch.nn as nn
from torch.distributions import Normal

from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from osrl.algorithms import BCQL
from osrl.common.exp_util import load_config_and_model, seed_all

import bullet_safety_gym  # registers Safety/Bullet envs
import dsrl               # registers Offline* envs

from utils.poison_utils import (
    simple_poison_dataset,
    generate_cost_max_poison,
    generate_reward_max_poison,
    generate_amad_poison,
    baffle_poison,
)


# --------------------------
# Custom PPO model from train_adv_ppo.py
# --------------------------

def _mlp(sizes: List[int], activation: nn.Module) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [activation()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Same architecture as the custom PPO trainer."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.actor_mean = _mlp([obs_dim] + hidden_sizes + [act_dim], nn.Tanh)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.critic = _mlp([obs_dim] + hidden_sizes + [1], nn.Tanh)

    def forward(self, obs: torch.Tensor):
        raise NotImplementedError("call get_action or get_value directly")

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


# --------------------------
# Torch PPO adapter for .pt checkpoints
# --------------------------

class TorchPolicyAdapter:
    """
    Adapter for the custom PPO .pt checkpoint saved as:
        {"model_state": policy.state_dict()}
    """

    def __init__(self, model: ActorCritic, device: str = "cuda"):
        self.model = model
        self.device = device

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        action, _ = self.model.get_action(obs_t, deterministic=deterministic)
        action = action.detach().cpu().numpy()

        if action.shape[0] == 1:
            action = action[0]
        return np.asarray(action, dtype=np.float32)


# --------------------------
# Trajectory injection poison
# --------------------------

def generate_trajectory_injection_poison(
    data: dict,
    env,
    policy_adapter,
    device: str = "cuda",
    poison_frac: float = 0.10,
    num_trajectories: int = 20,
    min_cost_threshold: float = 1.0,
    reward_override: float = 10.0,
    cost_override: float = 0.0,
    max_episode_steps: int = 500,
    seed: int = 42,
):
    """
    Trajectory Injection Attack:

    1. Roll out the policy in the environment to collect trajectories.
    2. Keep only trajectories where total_cost > min_cost_threshold.
    3. Relabel those transitions: cost -> 0, reward -> reward_override.
    4. Truncate to exactly int(poison_frac * N_orig) transitions.
    5. Inject the relabeled transitions into the offline dataset.
    """
    assert "observations" in data and "actions" in data, \
        "Dataset must have 'observations' and 'actions'."

    N_orig = data["observations"].shape[0]
    n_inject_target = int(round(poison_frac * N_orig))
    print(
        f"[INFO] Target injection size: {n_inject_target} transitions "
        f"({poison_frac * 100:.1f}% of N_orig={N_orig})"
    )

    rng = np.random.default_rng(seed)
    policy_adapter.eval()

    collected_transitions = []
    total_simulated = 0

    print(
        f"[INFO] Simulating up to {num_trajectories} episodes to collect "
        f"{n_inject_target} high-cost transitions..."
    )

    ep = 0
    while len(collected_transitions) < n_inject_target:
        ep += 1
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)

        ep_transitions = []
        ep_cost = 0.0

        for _ in range(max_episode_steps):
            act_np = policy_adapter.act(obs, deterministic=True)
            act_np = np.asarray(act_np, dtype=np.float32)

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

        if ep_cost > min_cost_threshold:
            collected_transitions.extend(ep_transitions)
            print(
                f"  [ep {ep:3d}] total_cost={ep_cost:.2f}  "
                f"steps={len(ep_transitions):4d} -> KEPT  "
                f"(collected: {len(collected_transitions)}/{n_inject_target})"
            )
        else:
            print(
                f"  [ep {ep:3d}] total_cost={ep_cost:.2f}  "
                f"steps={len(ep_transitions):4d} -> skipped (cost below threshold)"
            )

    if len(collected_transitions) == 0:
        print(
            "[WARN] No high-cost trajectories found. "
            "Try lowering --min_cost_threshold or increasing --num_trajectories."
        )
        return data, np.zeros(N_orig, dtype=bool)

    if len(collected_transitions) > n_inject_target:
        collected_transitions = collected_transitions[:n_inject_target]

    n_actual = len(collected_transitions)
    if n_actual < n_inject_target:
        print(
            f"[WARN] Only collected {n_actual}/{n_inject_target} transitions. "
            f"Increase --num_trajectories to reach the full {poison_frac * 100:.1f}% target."
        )

    print(
        f"[INFO] Injecting {n_actual} transitions "
        f"(simulated {total_simulated} total steps across all episodes)."
    )

    inj_obs      = np.array([t["observation"]      for t in collected_transitions], dtype=np.float32)
    inj_next_obs = np.array([t["next_observation"] for t in collected_transitions], dtype=np.float32)
    inj_actions  = np.array([t["action"]           for t in collected_transitions], dtype=np.float32)
    inj_rewards  = np.full(n_actual, reward_override, dtype=np.float32)
    inj_costs    = np.full(n_actual, cost_override,   dtype=np.float32)
    inj_terms    = np.array([t["terminated"] for t in collected_transitions], dtype=bool)
    inj_timeouts = np.array([t["truncated"]  for t in collected_transitions], dtype=bool)

    explicit = {
        "observations":      inj_obs,
        "next_observations": inj_next_obs,
        "actions":           inj_actions,
        "rewards":           inj_rewards,
        "costs":             inj_costs,
        "terminals":         inj_terms,
        "timeouts":          inj_timeouts,
    }

    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    for key, val in poisoned.items():
        if not (isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == N_orig):
            continue

        if key in explicit:
            pad = explicit[key]
        else:
            pad = np.zeros((n_actual,) + val.shape[1:], dtype=val.dtype)

        poisoned[key] = np.concatenate([val, pad], axis=0)

    mask = np.concatenate([
        np.zeros(N_orig, dtype=bool),
        np.ones(n_actual, dtype=bool),
    ])

    print(
        f"[INFO] Dataset: {N_orig} -> {N_orig + n_actual} transitions "
        f"({n_actual} injected at reward={reward_override}, cost={cost_override})"
    )

    return poisoned, mask


# --------------------------
# Helpers
# --------------------------

def _select_poison_indices(N: int, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(round(frac * N))
    k = max(0, min(k, N))
    return rng.choice(N, size=k, replace=False)


def _add_trigger_to_obs(
    obs: np.ndarray,
    trigger_type: str = "bias",
    strength: float = 0.5,
) -> np.ndarray:
    out = obs.copy()
    if out.ndim == 2 and out.shape[1] >= 2:
        if trigger_type == "bias":
            out[:, -2:] += strength
        elif trigger_type == "marker":
            out[:, -2:] = 0.9
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")
    else:
        out += strength
    return out


# --------------------------
# CLI config
# --------------------------

@dataclass
class PoisonConfig:
    task: str = "OfflineCarCircle-v0"
    attack_type: str = "max_cost"   # simple | max_cost | baffle | max_reward | amad | traj_inject

    poison_frac: float = 0.10
    seed: int = 42

    mode: str = "backdoor"
    obs_trigger: str = "marker"
    obs_strength: float = 0.6
    reward_shift: float = 0.5
    cost_to_zero: bool = True
    action_override: Optional[str] = None

    save_path: str = "save_data/offline_car_circle_poisoned.npz"

    norm: str = "linf"
    eps: float = 2.0
    steps: int = 10
    batch_size: int = 64
    update_next_obs: bool = False
    best: bool = False

    path: str = ""          # clean BCQL checkpoint path, only needed for max_cost/max_reward/baffle/amad
    adv_path: str = ""      # custom PPO .pt checkpoint path for traj_inject
    device: str = "cuda"

    num_trajectories: int = 20
    min_cost_threshold: float = 10.0
    reward_override: float = 10.0
    max_episode_steps: int = 500

    # must match train_adv_ppo.py architecture unless you change them
    adv_hidden_sizes: str = "256,256"
    adv_log_std_init: float = -0.5


# --------------------------
# Main
# --------------------------

def _safe_save_npz(path, **arrays):
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _parse_hidden_sizes(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return [256, 256]
    return [int(x.strip()) for x in s.split(",")]


@pyrallis.wrap()
def main(args: PoisonConfig):
    print(f"[INFO] Making env {args.task}")
    raw_env = gym.make(args.task)

    print("[INFO] Loading dataset via env.get_dataset()")
    data = raw_env.get_dataset()
    assert isinstance(data, dict), "env.get_dataset() should return a dict of numpy arrays."

    action_override = None
    if args.action_override is not None:
        s = args.action_override.strip()
        if "," in s:
            action_override = [float(x) for x in s.split(",")]
        else:
            try:
                action_override = [float(s)]
            except ValueError:
                action_override = [int(s)]

    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [dict_to_ns(x) for x in d]
        return d

    def subset_by_indices(dataset: dict, indices: np.ndarray) -> dict:
        indices = np.asarray(indices, dtype=int)
        n = None
        for _, v in dataset.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                n = v.shape[0]
                break
        if n is None:
            raise ValueError("Could not infer dataset length from provided arrays.")

        out = {}
        for k, v in dataset.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out[k] = v[indices]
            else:
                out[k] = v
        return out

    wrapped_env = raw_env
    bcql_model = None
    adv_model = None

    needs_clean_model = args.attack_type in {"max_cost", "max_reward", "baffle", "amad"}

    if needs_clean_model:
        if not args.path:
            raise ValueError(
                f"attack_type='{args.attack_type}' requires --path to a clean BCQL checkpoint."
            )

        cfg, model = load_config_and_model(args.path, args.best)
        cfg_ = dict_to_ns(cfg)
        seed_all(cfg["seed"])

        wrapped_env = wrap_env(
            env=gym.make(cfg["task"]),
            reward_scale=cfg["reward_scale"],
        )
        wrapped_env = OfflineEnvWrapper(wrapped_env)
        wrapped_env.set_target_cost(cfg["cost_limit"])

        if "PID" not in cfg:
            raise ValueError(
                "The checkpoint in --path does not look like a BCQL config "
                "(missing 'PID'). Use a BCQL checkpoint for max_cost/max_reward/baffle/amad."
            )

        bcql_model = BCQL(
            state_dim=wrapped_env.observation_space.shape[0],
            action_dim=wrapped_env.action_space.shape[0],
            max_action=wrapped_env.action_space.high[0],
            a_hidden_sizes=cfg_.a_hidden_sizes,
            c_hidden_sizes=cfg_.c_hidden_sizes,
            vae_hidden_sizes=cfg_.vae_hidden_sizes,
            sample_action_num=cfg_.sample_action_num,
            PID=cfg_.PID,
            gamma=cfg_.gamma,
            tau=cfg_.tau,
            lmbda=cfg_.lmbda,
            beta=cfg_.beta,
            phi=cfg_.phi,
            num_q=cfg_.num_q,
            num_qc=cfg_.num_qc,
            cost_limit=cfg_.cost_limit,
            episode_len=cfg_.episode_len,
            device=args.device,
        )
        bcql_model.load_state_dict(model["model_state"])
        bcql_model.to(args.device)

    if args.attack_type == "traj_inject":
        if not args.adv_path:
            raise ValueError(
                "For attack_type='traj_inject', please provide --adv_path to the custom PPO .pt checkpoint."
            )

        hidden_sizes = _parse_hidden_sizes(args.adv_hidden_sizes)
        obs_dim = raw_env.observation_space.shape[0]
        act_dim = raw_env.action_space.shape[0]

        adv_net = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            log_std_init=args.adv_log_std_init,
        ).to(args.device)

        print(f"[INFO] Loading custom PPO adversary from: {args.adv_path}")
        ckpt = torch.load(args.adv_path, map_location=args.device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(
                f"Unsupported adversary checkpoint format in {args.adv_path}. "
                "Expected a dict or a dict containing 'model_state'."
            )

        adv_net.load_state_dict(state_dict, strict=True)
        adv_model = TorchPolicyAdapter(adv_net, device=args.device)

    if args.attack_type == "simple":
        poisoned, mask = simple_poison_dataset(
            data,
            frac=args.poison_frac,
            seed=args.seed,
            mode=args.mode,
            obs_trigger=args.obs_trigger,
            obs_strength=args.obs_strength,
            reward_shift=args.reward_shift,
            cost_to_zero=args.cost_to_zero,
            action_override=action_override,
        )

    elif args.attack_type == "max_cost":
        poisoned, mask = generate_cost_max_poison(
            data=data,
            policy_adapter=bcql_model,
            device=args.device,
            poison_frac=args.poison_frac,
            norm=args.norm,
            eps=args.eps,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            update_next_obs=args.update_next_obs,
        )

    elif args.attack_type == "max_reward":
        poisoned, mask = generate_reward_max_poison(
            data=data,
            policy_adapter=bcql_model,
            device=args.device,
            poison_frac=args.poison_frac,
            norm=args.norm,
            eps=args.eps,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            update_next_obs=args.update_next_obs,
        )

    elif args.attack_type == "baffle":
        poisoned, mask = baffle_poison(
            data=data,
            env=wrapped_env,
            policy_adapter=bcql_model,
            device=args.device,
            task=args.task,
            poison_frac=args.poison_frac,
            seed=args.seed,
            update_next_obs=args.update_next_obs,
            distributed=True,
        )

    elif args.attack_type == "amad":
        poisoned, mask = generate_amad_poison(
            data=data,
            policy_adapter=bcql_model,
            device=args.device,
            poison_frac=args.poison_frac,
            norm=args.norm,
            eps=args.eps,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            update_next_obs=args.update_next_obs,
        )

    elif args.attack_type == "traj_inject":
        poisoned, mask = generate_trajectory_injection_poison(
            data=data,
            env=raw_env,
            policy_adapter=adv_model,
            device=args.device,
            poison_frac=args.poison_frac,
            num_trajectories=args.num_trajectories,
            min_cost_threshold=args.min_cost_threshold,
            reward_override=args.reward_override,
            cost_override=0.0,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown attack_type: {args.attack_type}")

    mask = np.asarray(mask).astype(bool).ravel()
    total = mask.size
    n_poison = int(mask.sum())
    poison_rate = (n_poison / total) if total > 0 else 0.0
    print(f"[INFO] Poisoned transitions: {n_poison} / {total} ({poison_rate * 100:.2f}%)")

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving full poisoned dataset to {args.save_path}")
    _safe_save_npz(args.save_path, **poisoned, poison_mask=mask)

    base, ext = os.path.splitext(args.save_path)
    if ext == "":
        ext = ".npz"

    poisoned_only_path = f"{base}.poisoned_only{ext}"
    clean_only_path = f"{base}.clean_only{ext}"

    idx_poison = np.where(mask)[0]
    idx_clean = np.where(~mask)[0]

    print(f"[INFO] Saving poisoned-only subset ({idx_poison.size}) -> {poisoned_only_path}")
    poisoned_subset = subset_by_indices(poisoned, idx_poison)
    np.savez_compressed(
        poisoned_only_path,
        **poisoned_subset,
        poison_mask=np.ones(idx_poison.size, dtype=bool),
    )

    print(f"[INFO] Saving clean-only subset ({idx_clean.size}) -> {clean_only_path}")
    clean_subset = subset_by_indices(poisoned, idx_clean)
    np.savez_compressed(
        clean_only_path,
        **clean_subset,
        poison_mask=np.zeros(idx_clean.size, dtype=bool),
    )


if __name__ == "__main__":
    main()