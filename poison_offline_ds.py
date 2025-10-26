# poison_offline_ds.py
# Usage example:
'''
For max-cost poisoning:
  python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_cost  --poison_frac 0.10  \
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
    attack_type: str = "max_cost"    # "simple" | "max_cost"

    poison_frac: float = 0.10
    seed: int = 42
    mode: str = "backdoor"          # "feature" | "label" | "backdoor"
    obs_trigger: str = "marker"     # "marker" | "bias"
    obs_strength: float = 0.6
    reward_shift: float = 0.5
    cost_to_zero: bool = True
    action_override: Optional[str] = None  # e.g. "0.0,-1.0" for continuous; "2" for discrete
    save_path: str = "save_data/offline_car_circle_poisoned.npz"
    # max-cost poisoning params
    norm: str = "linf"               # "linf" | "l2"
    eps: float = 2.0
    steps: int = 10
    batch_size: int = 64
    update_next_obs: bool = False
# --------------------------
# Main
# --------------------------


def _safe_save_npz(path, **arrays):
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
        s = args.action_override.strip()
        if "," in s:
            action_override = [float(x) for x in s.split(",")]
        else:
            try:
                action_override = [float(s)]
            except ValueError:
                action_override = [int(s)]

    def subset_by_indices(dataset: dict, indices: np.ndarray) -> dict:
        indices = np.asarray(indices, dtype=int)
        # infer dataset length from the first array-like entry
        n = None
        for k, v in dataset.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                n = v.shape[0]
                break
        if n is None:
            raise ValueError("Could not infer dataset length from provided arrays.")

        out = {}
        for k, v in dataset.items():
            # slice arrays whose first dim equals dataset length; copy-through otherwise
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out[k] = v[indices]
            else:
                out[k] = v
        return out

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

        env = wrap_env(env=env, reward_scale=model_cfg.reward_scale)
        env = OfflineEnvWrapper(env)

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
    else:
        raise ValueError(f"Unknown attack_type: {args.attack_type}")

    # Basic stats
    mask = np.asarray(mask).astype(bool).ravel()
    total = mask.size
    n_poison = int(mask.sum())
    poison_rate = (n_poison / total) if total > 0 else 0.0
    print(f"[INFO] Poisoned transitions: {n_poison} / {total} ({poison_rate*100:.2f}%)")

    # Save dir
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 1) Save the full poisoned dataset 
    print(f"[INFO] Saving full poisoned dataset to {args.save_path}")
    _safe_save_npz(args.save_path, **poisoned, poison_mask=mask)

    # 2) Also save splits: poisoned-only and clean-only
    base, ext = os.path.splitext(args.save_path)
    if ext == "":
        ext = ".npz"
    poisoned_only_path = f"{base}.poisoned_only{ext}"
    clean_only_path = f"{base}.clean_only{ext}"

    idx_poison = np.where(mask)[0]
    idx_clean = np.where(~mask)[0]

    print(f"[INFO] Saving poisoned-only subset ({idx_poison.size}) -> {poisoned_only_path}")
    poisoned_subset = subset_by_indices(poisoned, idx_poison)
    np.savez_compressed(poisoned_only_path, **poisoned_subset,
                        poison_mask=np.ones(idx_poison.size, dtype=bool))

    print(f"[INFO] Saving clean-only subset ({idx_clean.size}) -> {clean_only_path}")
    clean_subset = subset_by_indices(poisoned, idx_clean)
    np.savez_compressed(clean_only_path, **clean_subset,
                        poison_mask=np.zeros(idx_clean.size, dtype=bool))

    # # 3) Optional: small JSON metadata next to the files
    # meta = {
    #     "task": args.task,
    #     "attack_type": args.attack_type,
    #     "mode": args.mode,
    #     "poison_frac_requested": args.poison_frac,
    #     "poison_frac_realized": float(poison_rate),
    #     "num_total": int(total),
    #     "num_poisoned": int(n_poison),
    #     "num_clean": int(total - n_poison),
    #     "save_path_full": args.save_path,
    #     "save_path_poisoned_only": poisoned_only_path,
    #     "save_path_clean_only": clean_only_path,
    #     "seed": args.seed,
    # }
    # meta_path = f"{base}.meta.json"
    # with open(meta_path, "w") as f:
    #     json.dump(meta, f, indent=2)
    # print(f"[DONE] Saved splits and metadata to {meta_path}")

if __name__ == "__main__":
    main()
