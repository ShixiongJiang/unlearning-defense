# poison_offline_ds.py
# Usage example:
'''
For max-cost poisoning:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_cost  --poison_frac 0.10 --mode backdoor \
      --obs_trigger marker  --cost_to_zero true \
      --norm linf --eps 0.2 --steps 10 --batch_size 64\
      --save_path save_data/offline_car_circle_poisoned_maxcost.npz

For simple poisoning:
    python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type simple --poison_frac 0.10 --mode backdoor \
      --obs_trigger marker --obs_strength 0.6 --reward_shift 0.5 --cost_to_zero true \
      --save_path save_data/offline_car_circle_poisoned_maxcost.npz
'''


from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import gymnasium as gym
import pyrallis
from dataclasses import asdict, dataclass
from osrl.algorithms import BCQL, BCQLTrainer
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
import os
from examples.configs.bcql_configs import BCQL_DEFAULT_CONFIG, BCQLTrainConfig
from osrl.algorithms import BCQL, BCQLTrainer
import types
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa

import bullet_safety_gym   # registers Safety/Bullet envs
import dsrl                # registers Offline* envs
from utils.poison_utils import simple_poison_dataset, generate_cost_max_poison
# --------------------------
# Poisoning utilities
# --------------------------

def _select_poison_indices(N: int, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(round(frac * N))
    k = max(0, min(k, N))
    return rng.choice(N, size=k, replace=False)

def _add_trigger_to_obs(obs: np.ndarray, trigger_type: str = "bias", strength: float = 0.5) -> np.ndarray:
    """Simple triggers for vector observations. Works on a COPY."""
    out = obs.copy()
    if out.ndim == 2 and out.shape[1] >= 2:
        if trigger_type == "bias":
            out[:, -2:] += strength
        elif trigger_type == "marker":
            out[:, -2:] = 0.9
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")
    else:
        # Fallback: shift everything (keeps shapes)
        out += strength
    return out


# --------------------------
# CLI config
# --------------------------

@dataclass
class PoisonConfig:
    task: str = "OfflineCarCircle-v0"
    attack_type: str = "simple"    # "simple" | "max_cost"

    poison_frac: float = 0.10
    seed: int = 42
    mode: str = "backdoor"          # "feature" | "label" | "backdoor"
    obs_trigger: str = "marker"     # "marker" | "bias"
    obs_strength: float = 0.6
    reward_shift: float = 0.5
    cost_to_zero: bool = True
    action_override: Optional[str] = None  # e.g. "0.0,-1.0" for continuous; "2" for discrete
    save_path: str = "offline_car_circle_poisoned.npz"
    # max-cost poisoning params
    norm: str = "linf"               # "linf" | "l2"
    eps: float = 0.2
    steps: int = 10
    batch_size: int = 64
    update_next_obs: bool = False
# --------------------------
# Main
# --------------------------

@pyrallis.wrap()
def main(args: PoisonConfig):
    print(f"[INFO] Making env {args.task}")
    env = gym.make(args.task)

    print("[INFO] Loading dataset via env.get_dataset()")
    data = env.get_dataset()
    assert isinstance(data, dict), "env.get_dataset() should return a dict of numpy arrays."

    # Parse action_override string if provided
    action_override = None
    if args.action_override is not None:
        # Try comma-separated floats first; fall back to single int
        s = args.action_override.strip()
        if "," in s:
            action_override = [float(x) for x in s.split(",")]
        else:
            # Could be float or int; poison_dataset handles discrete vs continuous
            try:
                action_override = [float(s)]
            except ValueError:
                action_override = [int(s)]

    print(f"[INFO] Poisoning {args.poison_frac*100:.1f}% of transitions (mode={args.mode})")
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
    #  poisoning via cost maximization
        cfg = asdict(BCQL_DEFAULT_CONFIG[args.task]())
        model_cfg = types.SimpleNamespace(**cfg)

        # wrapper
        env = wrap_env(
            env=env,
            reward_scale=model_cfg.reward_scale,
        )
        env = OfflineEnvWrapper(env)

        # model & optimizer setup
        model = BCQL(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            a_hidden_sizes=model_cfg.a_hidden_sizes,
            c_hidden_sizes=model_cfg.c_hidden_sizes,
            vae_hidden_sizes=model_cfg.vae_hidden_sizes,
            sample_action_num=model_cfg.sample_action_num,
            PID=model_cfg.PID,
            gamma=model_cfg.gamma,
            tau=model_cfg.tau,
            lmbda=model_cfg.lmbda,
            beta=model_cfg.beta,
            phi=model_cfg.phi,
            num_q=model_cfg.num_q,
            num_qc=model_cfg.num_qc,
            cost_limit=model_cfg.cost_limit,
            episode_len=model_cfg.episode_len,
            device=model_cfg.device,
        )
        poisoned, mask = generate_cost_max_poison(
            data=data,
            policy_adapter=model,
            device=model_cfg.device,
            poison_frac=args.poison_frac,
            norm=args.norm,
            eps=args.eps,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            update_next_obs=args.update_next_obs,
        )

    poison_rate = mask.mean() if mask.size > 0 else 0.0
    print(f"[INFO] Poisoned transitions: {mask.sum()} / {mask.size} ({poison_rate*100:.2f}%)")

    # Save
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving poisoned dataset to {args.save_path}")
    np.savez_compressed(args.save_path, **poisoned, poison_mask=mask)
    print("[DONE] Poisoned dataset saved.")

if __name__ == "__main__":
    main()
