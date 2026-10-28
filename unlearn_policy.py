'''
Supported algorithms: BCQL, CPQ, BEARL, BC, CDT, COptiDICE

python unlearn_policy.py --algo BCQL       --unlearn_path logs/.../BCQL_xxx       --eval_episodes 20
python unlearn_policy.py --algo CPQ        --unlearn_path logs/.../CPQ_xxx        --eval_episodes 20
python unlearn_policy.py --algo BEARL      --unlearn_path logs/.../BEARL_xxx      --eval_episodes 20
python unlearn_policy.py --algo BC         --unlearn_path logs/.../BC_xxx         --eval_episodes 20
python unlearn_policy.py --algo CDT        --unlearn_path logs/.../CDT_xxx        --eval_episodes 20
python unlearn_policy.py --algo COptiDICE  --unlearn_path logs/.../COptiDICE_xxx  --eval_episodes 20

Backward-compatible: if --algo is omitted it is inferred from cfg["name"].
'''

import os
import re
import time
import types
import json
import inspect
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
import gymnasium as gym

import bullet_safety_gym  # noqa: F401
import dsrl               # noqa: F401

from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

# ── Algorithm model imports ───────────────────────────────────────────────────
from examples.configs.bcql_configs import BCQL_DEFAULT_CONFIG, BCQLTrainConfig
from osrl.algorithms import BCQL, BCQLTrainer, CPQ, CPQTrainer
from osrl.common import TransitionDataset
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, load_config_and_model, seed_all

from examples.configs.bearl_configs import BEARL_DEFAULT_CONFIG, BEARLTrainConfig
from osrl.algorithms import BEARL, BEARLTrainer
from examples.configs.bc_configs import BC_DEFAULT_CONFIG, BCTrainConfig
from osrl.algorithms import BC, BCTrainer
from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
from osrl.algorithms import CDT, CDTTrainer
from examples.configs.coptidice_configs import COptiDICE_DEFAULT_CONFIG, COptiDICETrainConfig
from osrl.algorithms import COptiDICE, COptiDICETrainer

# ── Unlearning trainers ───────────────────────────────────────────────────────
from utils.unlearn_utils import (
    BCQLTrainerWithUnlearn,
    BCTrainerWithUnlearn,
    CDTTrainerWithUnlearn,
    CPQTrainerWithUnlearn,
    BEARLTrainerWithUnlearn,
    COptiDICETrainerWithUnlearn
)
from utils.unlearn_config import (
    DEFAULT_UNLEARN, UnlearnHyperparameters, configure_unlearning,
)



# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    unlearn_path:   str  = "log/.../checkpoint/model.pt"
    eval_episodes:  int  = 20
    best:           bool = False
    unlearn_steps:  int  = 6000
    converge_steps: int  = 6000
    eval_every:     int  = 200
    unlearn_method: str  = "safe"   # safe / reward-only / fine-tune / relabel
    draw:           bool = False
    seed: Optional[int] = None
    output_dir: str = ""
    run_name: str = ""
    total_updates: Optional[int] = None
    milestone_updates: str = "5000,10000,12300,20000"
    milestone_eval_episodes: int = 0
    # CLI defaults come from the same validated definition as direct trainer calls.
    alpha_q_keep: float = DEFAULT_UNLEARN.alpha_q_keep
    alpha_q_forget: float = DEFAULT_UNLEARN.alpha_q_forget
    alpha_cq_keep: float = DEFAULT_UNLEARN.alpha_cq_keep
    alpha_cq_forget: float = DEFAULT_UNLEARN.alpha_cq_forget
    alpha_pg_keep: float = DEFAULT_UNLEARN.alpha_pg_keep
    cost_forget_scale: float = DEFAULT_UNLEARN.cost_forget_scale
    keep_cost_weight: float = DEFAULT_UNLEARN.keep_cost_weight
    keep_bc_weight: float = DEFAULT_UNLEARN.keep_bc_weight
    max_grad_norm: float = DEFAULT_UNLEARN.max_grad_norm
    eta_mode: str = DEFAULT_UNLEARN.eta_mode
    eta_fixed: float = DEFAULT_UNLEARN.eta_fixed
    eta_initial: float = DEFAULT_UNLEARN.eta_initial
    eta_beta: float = DEFAULT_UNLEARN.eta_beta
    eta_min: float = DEFAULT_UNLEARN.eta_min
    eta_max: float = DEFAULT_UNLEARN.eta_max
    eta_temperature: float = DEFAULT_UNLEARN.eta_temperature
    grad_norm_beta: float = DEFAULT_UNLEARN.grad_norm_beta
    grad_norm_floor: float = DEFAULT_UNLEARN.grad_norm_floor
    fisher_beta: float = DEFAULT_UNLEARN.fisher_beta
    fisher_damping: float = DEFAULT_UNLEARN.fisher_damping
    fisher_clip: float = DEFAULT_UNLEARN.fisher_clip
    reward_forget_quantile: float = DEFAULT_UNLEARN.reward_forget_quantile
    reward_forget_temperature: float = DEFAULT_UNLEARN.reward_forget_temperature
    cost_forget_temperature: float = DEFAULT_UNLEARN.cost_forget_temperature
    forget_cost_margin: float = DEFAULT_UNLEARN.forget_cost_margin
    bc_forget_distance_target: float = DEFAULT_UNLEARN.bc_forget_distance_target
    bc_forget_temperature: float = DEFAULT_UNLEARN.bc_forget_temperature
    cdt_forget_cost_confidence_target: float = DEFAULT_UNLEARN.cdt_forget_cost_confidence_target
    coptidice_forget_weight_target: float = DEFAULT_UNLEARN.coptidice_forget_weight_target
    fit_behavior_on_retain: bool = True
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    vae_lr: Optional[float] = None
    alpha_lr: Optional[float] = None
    scalar_lr: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None

    # Explicit algo override. When empty, inferred from cfg["name"].
    algo: str = ""  # BCQL / CPQ / BEARL / BC / CDT / COptiDICE

    # Relabeling-specific
    use_feature_encoder:      bool  = True
    feature_latent_dim:       int   = 128
    k_neighbors:              int   = 20
    temperature:              float = 1.0
    use_confidence_weighting: bool  = False
    num_ensemble:             int   = 5
    use_gradual_relabeling:   bool  = True
    encoder_lr:               float = 3e-4

    attack_type: str = "traj_inject"

    # Unlearn-score settings (safe method only)
    score_complete_threshold: float = 0.80
    score_early_stop_streak:  int   = 2

    # CDT-specific
    cdt_seq_len: Optional[int] = None
    cdt_target_return: float = 600.0
    cdt_target_cost:   float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Standard 6-tuple dataloader  (BCQL / CPQ / BEARL / BC)
# ══════════════════════════════════════════════════════════════════════════════

def _shared_hyperparameters(eval_args):
    values = {
        field.name: getattr(eval_args, field.name) for field in fields(DEFAULT_UNLEARN)
    }
    if eval_args.unlearn_method == "reward-only":
        values["cost_forget_scale"] = 0.0
    return UnlearnHyperparameters(**values)


def _native_trainer_kwargs(algo, args, eval_args=None):
    """Restore native settings, with explicit overrides only where applicable."""
    rate_fields = {
        "BCQL": ("actor_lr", "critic_lr", "vae_lr"),
        "CPQ": ("actor_lr", "critic_lr", "vae_lr", "alpha_lr"),
        "BEARL": ("actor_lr", "critic_lr", "vae_lr", "alpha_lr"),
        "BC": ("actor_lr",), "CDT": ("learning_rate",),
        "COptiDICE": ("actor_lr", "critic_lr", "scalar_lr"),
    }
    # Baseline wrappers may expose (*args, **kwargs); native signatures still
    # define the legitimate fallback rates for older source configurations.
    native = {"BCQL": BCQLTrainer, "CPQ": CPQTrainer, "BEARL": BEARLTrainer,
              "BC": BCTrainer, "CDT": CDTTrainer, "COptiDICE": COptiDICETrainer}
    signature = inspect.signature(native[algo].__init__).parameters
    rates = {}
    for key in ("actor_lr", "critic_lr", "vae_lr", "alpha_lr", "scalar_lr", "learning_rate"):
        override = getattr(eval_args, key, None)
        if override is not None and key not in rate_fields[algo]:
            raise ValueError(f"{key} is not used by {algo}")
        if key not in rate_fields[algo]:
            continue
        value = override if override is not None else getattr(args, key, signature[key].default)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        rates[key] = float(value)
    if algo == "BC":
        rates.update(cost_limit=args.cost_limit, bc_mode=getattr(args, "bc_mode", "all"))
    else:
        rates.update(reward_scale=args.reward_scale, cost_scale=args.cost_scale)
    if algo == "CDT":
        for key in ("weight_decay", "betas", "clip_grad", "lr_warmup_steps",
                    "loss_cost_weight", "loss_state_weight", "cost_reverse", "no_entropy"):
            rates[key] = getattr(args, key, signature[key].default)
        if rates["lr_warmup_steps"] <= 0:
            raise ValueError("lr_warmup_steps must be positive")
    rates["device"] = args.device
    return rates


def _build_trainer(policy_model, env, algo, args, eval_args=None):
    constructors = {
        "BCQL": BCQLTrainerWithUnlearn, "CPQ": CPQTrainerWithUnlearn,
        "BEARL": BEARLTrainerWithUnlearn, "BC": BCTrainerWithUnlearn,
        "CDT": CDTTrainerWithUnlearn, "COptiDICE": COptiDICETrainerWithUnlearn,
    }
    if algo == "CDT" and policy_model.stochastic:
        if policy_model.target_entropy is None:
            policy_model.target_entropy = -policy_model.action_dim
        # Native CDT stores temperature as an unregistered tensor.
        policy_model.log_temperature = policy_model.log_temperature.detach().to(
            args.device).requires_grad_(True)
    return constructors[algo](policy_model, env, **_native_trainer_kwargs(algo, args, eval_args))


def _runtime_parameters(trainer, algo, args):
    optimizers = {}
    for owner in (trainer, trainer.model):
        for name, value in vars(owner).items():
            if isinstance(value, torch.optim.Optimizer):
                optimizers[name] = [{key: group[key] for key in
                    ("lr", "initial_lr", "betas", "weight_decay") if key in group}
                    for group in value.param_groups]
    return {"algorithm": algo, "batch_size": args.batch_size,
            "episode_len": trainer.model.episode_len,
            "reward_scale": args.reward_scale, "cost_scale": args.cost_scale,
            "cost_limit": args.cost_limit, "optimizer_parameter_groups": optimizers,
            "manual_learning_rates": {name: getattr(trainer.model, name)
                                      for name in ("alpha_lr",) if hasattr(trainer.model, name)},
            "fresh_optimizer_state": True,
            "shared_hyperparameters": asdict(trainer.unlearn_config)}

def load_npz_as_dict(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d.pop("poison_mask", None)
    return d

def _episode_ends(data):
    size = len(data["observations"])
    ends = np.zeros(size, dtype=bool)
    for key in ("terminals", "timeouts", "dones", "done"):
        if key in data:
            values = np.asarray(data[key], dtype=bool)
            if values.shape != (size,):
                raise ValueError(f"{key} must have one flag per transition")
            ends |= values
    return ends


def _ensure_terminals(data: dict) -> dict:
    """Normalize flags and preserve an unflagged final recorded prefix.

    Preserve genuine timeout boundaries instead of turning a timeout into an
    absorbing terminal. Closing the final row only makes a recorded prefix
    visible to DSRL preprocessing; it does not reconstruct missing episodes
    or infer internal resets from potentially perturbed observations.
    """
    data = dict(data)
    size = len(data["observations"])
    if not size:
        raise ValueError("Empty unlearning data")
    terminals = np.asarray(data.get("terminals", np.zeros(size)), dtype=bool).copy()
    timeouts = np.asarray(data.get("timeouts", np.zeros(size)), dtype=bool).copy()
    ends = _episode_ends(data)
    # Generic done aliases have no separate truncation semantics.
    terminals |= ends & ~(terminals | timeouts)
    if not ends[-1]:
        timeouts[-1] = True
    data.update(terminals=terminals, timeouts=timeouts)
    return data

def _make_dataloader(data, env, args, batch_size):
    data = _ensure_terminals(data) 
    cbins = rbins = max_npb = min_npb = None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins   = density_cfg["cbins"]
        rbins   = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    processed = env.pre_process_data(
        data, args.outliers_percent, args.noise_scale,
        args.inpaint_ranges, args.epsilon, args.density,
        cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb,
    )
    if getattr(args, "unlearn_algo", None) == "BC":
        process_bc_dataset(processed, args.cost_limit, args.gamma,
                           getattr(args, "bc_mode", "all"))
        if not len(processed["observations"]):
            raise ValueError("BC mode removed every transition from an unlearning split")
    return DataLoader(
        TransitionDataset(processed, reward_scale=args.reward_scale,
                          cost_scale=args.cost_scale),
        batch_size=batch_size, pin_memory=True, num_workers=args.num_workers,
    )


def _task_short_name(task: str) -> str:
    name = re.sub(r"Offline",   "", task)
    name = re.sub(r"Gymnasium", "", name)
    name = re.sub(r"-v\d+",     "", name)
    return name


# ══════════════════════════════════════════════════════════════════════════════
# CDT sequence dataloader
# ══════════════════════════════════════════════════════════════════════════════

class _CDTSequenceDataset(Dataset):
    """
    Converts flat transition arrays into CDT 8-tuple sequence windows of
    length seq_len.  Each window carries:
        states        [seq_len, obs_dim]      float32
        actions       [seq_len, act_dim]      float32
        rtg           [seq_len]               float32   returns-to-go (suffix sum)
        ctg           [seq_len]               float32   costs-to-go   (suffix sum)
        timesteps     [seq_len]               int64
        mask          [seq_len]               float32   1=valid, 0=pad
        episode_cost  scalar                  float32
        cost_labels   [seq_len]               int64     (cost_step > 0)
    """

    def __init__(self, data: dict, seq_len: int,
                 reward_scale: float = 1.0, cost_scale: float = 1.0,
                 cost_reverse: bool = False):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        obs   = data["observations"].astype(np.float32)
        acts  = data["actions"].astype(np.float32)
        rews  = (data["rewards"] * reward_scale).astype(np.float32)
        raw_costs = (1.0 - data["costs"]) if cost_reverse else data["costs"]
        costs = (raw_costs * cost_scale).astype(np.float32)
        dones = _episode_ends(data)

        self.windows: list = []
        ep_start = 0
        for t in range(len(obs)):
            if dones[t] or t == len(obs) - 1:
                ep_end   = t + 1
                ep_o, ep_a = obs[ep_start:ep_end], acts[ep_start:ep_end]
                ep_r, ep_c = rews[ep_start:ep_end], costs[ep_start:ep_end]
                T          = ep_end - ep_start
                rtg        = np.cumsum(ep_r[::-1])[::-1].copy()
                ctg        = np.cumsum(ep_c[::-1])[::-1].copy()
                ep_cl      = (ep_c > 0).astype(np.int64)
                epi_cost   = float(ep_c.sum())

                for s in range(T):
                    L  = min(s + seq_len, T) - s
                    sl = slice(s, s + L)
                    s_w   = np.zeros((seq_len, ep_o.shape[1]), dtype=np.float32)
                    a_w   = np.zeros((seq_len, ep_a.shape[1]), dtype=np.float32)
                    rtg_w = np.zeros((seq_len,),               dtype=np.float32)
                    ctg_w = np.zeros((seq_len,),               dtype=np.float32)
                    ts_w  = np.zeros((seq_len,),               dtype=np.int64)
                    mask  = np.zeros((seq_len,),               dtype=np.float32)
                    cl_w  = np.zeros((seq_len,),               dtype=np.int64)
                    s_w[:L]  = ep_o[sl];  a_w[:L]  = ep_a[sl]
                    rtg_w[:L]= rtg[sl];   ctg_w[:L]= ctg[sl]
                    ts_w[:L] = np.arange(s, s + L)
                    mask[:L] = 1.0;       cl_w[:L] = ep_cl[sl]
                    self.windows.append(
                        (s_w, a_w, rtg_w, ctg_w, ts_w, mask, epi_cost, cl_w))
                ep_start = ep_end

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s, a, rtg, ctg, ts, mask, ec, cl = self.windows[idx]
        return (torch.from_numpy(s),   torch.from_numpy(a),
                torch.from_numpy(rtg), torch.from_numpy(ctg),
                torch.from_numpy(ts),  torch.from_numpy(mask),
                torch.tensor(ec, dtype=torch.float32), torch.from_numpy(cl))


def _make_dataloader_cdt(data, seq_len, batch_size,
                         reward_scale=1.0, cost_scale=1.0, cost_reverse=False):
    return DataLoader(
        _CDTSequenceDataset(data, seq_len, reward_scale, cost_scale, cost_reverse),
        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COptiDICE 7-tuple dataloader  (appends is_init)
# ══════════════════════════════════════════════════════════════════════════════

class _COptiDICEDataset(TransitionDataset):
    """Use the native scaled, random-transition sampler and initial-state flags."""

    def __init__(self, processed_data, reward_scale=1.0, cost_scale=1.0):
        super().__init__(_ensure_terminals(processed_data),
                         reward_scale=reward_scale, cost_scale=cost_scale,
                         state_init=True)
 
 
def _make_dataloader_coptidice(data, env, args, batch_size):
    data = _ensure_terminals(data)
    cbins = rbins = max_npb = min_npb = None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins   = density_cfg["cbins"]
        rbins   = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    processed = env.pre_process_data(
        data, args.outliers_percent, args.noise_scale,
        args.inpaint_ranges, args.epsilon, args.density,
        cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb,
    )
    return DataLoader(
        _COptiDICEDataset(processed, args.reward_scale, args.cost_scale),
        batch_size=batch_size, pin_memory=True, num_workers=args.num_workers,
    )


def _coptidice_statistics(cfg, retained_loader):
    """Normalize initial-state weights for the actual retained sampler."""
    keys = ("init_state_propotion", "observations_std", "actions_std")
    proportion, retained_obs_std, retained_act_std = retained_loader.dataset.get_dataset_states()
    saved = cfg.get("coptidice_dataset_statistics", {})
    if not saved and all(key in cfg for key in keys):
        saved = {key: cfg[key] for key in keys}
    if saved:
        if any(key not in saved for key in keys):
            raise ValueError("Incomplete saved COptiDICE dataset statistics")
        source_proportion = float(saved["init_state_propotion"])
        obs_std, act_std = saved["observations_std"], saved["actions_std"]
        provenance = ("initial-state frequency recomputed from the processed retained sampler; "
                      "observation/action standard deviations restored from source configuration")
    else:
        source_proportion = None
        obs_std, act_std = retained_obs_std, retained_act_std
        provenance = ("recomputed from processed retained data for the retained-data objective; "
                      "source statistics and optimizer history were not persisted")
    proportion = float(proportion)
    obs_std, act_std = np.asarray(obs_std), np.asarray(act_std)
    if not np.isfinite(proportion) or not 0 < proportion <= 1:
        raise ValueError("COptiDICE initial-state proportion must be in (0, 1]")
    for values, label in ((obs_std, "observations_std"), (act_std, "actions_std")):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid COptiDICE {label}")
    return {"init_state_propotion": proportion, "observations_std": obs_std.tolist(),
            "actions_std": act_std.tolist(), "source_init_state_propotion": source_proportion,
            "provenance": provenance}


def _build_loaders(poison_raw, clean_raw, env, args, eval_args, algo):
    for role, data in (("forget", poison_raw), ("retain", clean_raw)):
        _validate_episode_boundaries(data, args.episode_len, role)
    if algo == "CDT":
        common = (args.seq_len, args.batch_size, args.reward_scale, args.cost_scale,
                  getattr(args, "cost_reverse", False))
        return (_make_dataloader_cdt(poison_raw, *common),
                _make_dataloader_cdt(clean_raw, *common))
    loader = _make_dataloader_coptidice if algo == "COptiDICE" else _make_dataloader
    return (loader(poison_raw, env, args, args.batch_size),
            loader(clean_raw, env, args, args.batch_size))


def _validate_episode_boundaries(data, episode_len, role="dataset"):
    """Reject impossible long segments without inventing internal resets."""
    size = len(data["observations"])
    if size == 0 or episode_len <= 0:
        raise ValueError("Dataset and configured episode horizon must be positive")
    ends = np.flatnonzero(_episode_ends(data))
    if not len(ends) or ends[-1] != size - 1:
        ends = np.append(ends, size - 1)  # a final collector prefix is permitted
    longest = int(np.diff(np.r_[-1, ends]).max())
    if longest > episode_len:
        raise ValueError(
            f"{role} split has a {longest}-transition segment exceeding episode_len={episode_len}: "
            "missing episode/collector timeout boundaries; prepare verified recording metadata first")
 

# ══════════════════════════════════════════════════════════════════════════════
# train_one_step dispatcher  (called in fine-tune and Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

def _fine_tune_step(trainer, algo: str, batch, device: str):
    """
    Dispatch to the correct train_one_step signature for each algorithm.

    BCQL / CPQ / BEARL → train_one_step(obs, nobs, act, rew, cost, done)
    BC                 → train_one_step(obs, act)
    CDT                → train_one_step(s, a, rtg, ctg, ts, mask, ec, cl)
    COptiDICE          → train_one_step(batch)   [single 7-tuple arg]
    """
    if algo == "CDT":
        s, a, rtg, ctg, ts, mask, ec, cl = [b.to(device) for b in batch]
        trainer.train_one_step(s, a, rtg, ctg, ts, mask, ec, cl)

    elif algo == "COptiDICE":
        trainer.train_one_step(tuple(b.to(device) for b in batch))

    elif algo == "BC":
        # batch is a standard 6-tuple from _make_dataloader; BC only needs obs, act
        obs, _, act, *_ = [b.to(device) for b in batch]
        trainer.train_one_step(obs, act)

    else:  # BCQL / CPQ / BEARL
        obs, nobs, act, rew, cost, done = [b.to(device) for b in batch]
        trainer.train_one_step(obs, nobs, act, rew, cost, done)


# ══════════════════════════════════════════════════════════════════════════════
# evaluate dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(trainer, algo: str, eval_args: EvalConfig, args,
              episodes: Optional[int] = None) -> tuple:
    """
    CDT.evaluate() requires (n_episodes, target_return, target_cost).
    Every other trainer uses evaluate(n_episodes).
    """
    episodes = eval_args.eval_episodes if episodes is None else episodes
    if algo == "CDT":
        reward, cost, length = trainer.evaluate(
            episodes,
            target_return=eval_args.cdt_target_return * args.reward_scale,
            target_cost=eval_args.cdt_target_cost     * args.cost_scale,
        )
        # Native CDT.rollout accumulates raw info.cost, whereas evaluate
        # divides it by cost_scale along with the conditioning cost return.
        return reward, cost * args.cost_scale, length
    if algo == "BC":
        reward, cost, length = trainer.evaluate(episodes)
        return reward / args.reward_scale, cost, length
    return trainer.evaluate(episodes)


def _update_plan(eval_args: EvalConfig, warmup_available: bool) -> dict:
    """Count every optimizer-update batch, including critic-only warm-up."""
    if eval_args.unlearn_steps < 0 or eval_args.converge_steps < 0:
        raise ValueError("Phase update counts must be nonnegative")
    if eval_args.eval_every <= 0 or eval_args.eval_episodes <= 0:
        raise ValueError("eval_every and eval_episodes must be positive")
    if eval_args.milestone_eval_episodes < 0:
        raise ValueError("milestone_eval_episodes must be nonnegative")
    if eval_args.total_updates is not None and eval_args.total_updates < 0:
        raise ValueError("total_updates must be nonnegative")
    fine_tune = eval_args.unlearn_method == "fine-tune"
    if eval_args.total_updates is None:
        phase1 = eval_args.unlearn_steps
        phase2 = 0 if fine_tune else eval_args.converge_steps
        warmup = 300 if warmup_available and phase2 else 0
    elif fine_tune:
        phase1, warmup, phase2 = eval_args.total_updates, 0, 0
    else:
        phase1 = min(eval_args.unlearn_steps, eval_args.total_updates)
        remaining = eval_args.total_updates - phase1
        warmup = min(300, remaining) if warmup_available else 0
        phase2 = remaining - warmup
    return {"unlearning": phase1, "warmup": warmup, "convergence": phase2,
            "total_updates": phase1 + warmup + phase2}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@pyrallis.wrap()
def main(eval_args: EvalConfig):

    hyperparameters = _shared_hyperparameters(eval_args)
    # ── Load checkpoint ───────────────────────────────────────────────────────
    cfg, model_ckpt = load_config_and_model(eval_args.unlearn_path, eval_args.best)
    cfg = dict(cfg)
    source_config = dict(cfg)
    args            = types.SimpleNamespace(**cfg)

    # Resolve algo — explicit flag beats inference from cfg["name"]
    _INFER_MAP = {
        "BCQL": "BCQL", "CPQ": "CPQ", "BEARL": "BEARL",
        "BC": "BC", "CDT": "CDT", "COPTIDICE": "COptiDICE",
    }
    algo = eval_args.algo.strip()
    if algo:
        algo = _INFER_MAP.get(algo.upper(), algo)
    else:
        for key, val in _INFER_MAP.items():
            if key in cfg["name"].upper():
                algo = val
                break
    if not algo:
        raise ValueError(
            f"Cannot infer algo from cfg name '{cfg['name']}'. Use --algo.")
    if algo not in ("BCQL", "CPQ", "BEARL", "BC", "CDT", "COptiDICE"):
        raise ValueError(f"Unknown algorithm: {algo}")
    if algo == "BC":
        # Native BC uses raw signals and does not put scale fields in its config.
        cfg.setdefault("reward_scale", 1.0)
        cfg.setdefault("cost_scale", 1.0)
    for key in ("reward_scale", "cost_scale"):
        value = cfg[key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        setattr(args, key, value)
    args.unlearn_algo = algo
    if eval_args.batch_size is not None:
        if eval_args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        args.batch_size = cfg["batch_size"] = eval_args.batch_size
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if eval_args.cdt_seq_len is not None and algo != "CDT":
        raise ValueError("cdt_seq_len is only used by CDT")
    if algo == "CDT":
        saved_seq_len = cfg.get("seq_len", 10)
        if eval_args.cdt_seq_len is not None and eval_args.cdt_seq_len != saved_seq_len:
            raise ValueError("cdt_seq_len cannot change the saved checkpoint architecture")
        args.seq_len = cfg["seq_len"] = saved_seq_len
    native_settings = _native_trainer_kwargs(algo, args, eval_args)
    for key in ("actor_lr", "critic_lr", "vae_lr", "alpha_lr", "scalar_lr", "learning_rate"):
        if key in native_settings:
            setattr(args, key, native_settings[key])
            cfg[key] = native_settings[key]

    args.name = eval_args.run_name or (
        args.name + str(args.task) + "_unlearn_" + str(eval_args.unlearn_method))
    args.logdir = os.path.abspath(eval_args.output_dir or os.path.join(
        "new_unlearn_" + cfg["logdir"], args.name))
    source_dir = os.path.realpath(eval_args.unlearn_path)
    if not os.path.isdir(source_dir):
        source_dir = os.path.dirname(source_dir)
        if os.path.basename(source_dir) == "checkpoint":
            source_dir = os.path.dirname(source_dir)
    if os.path.commonpath([os.path.realpath(args.logdir), source_dir]) == source_dir:
        raise ValueError("Unlearning output must be outside the source checkpoint directory")
    args.seed = cfg["seed"] if eval_args.seed is None else eval_args.seed
    args.eval_episodes = eval_args.eval_episodes
    cfg.update(name=args.name, logdir=args.logdir, seed=args.seed,
               eval_episodes=args.eval_episodes, unlearning=asdict(eval_args))
    cfg["unlearning"]["effective_hyperparameters"] = asdict(hyperparameters)

    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # ── Environment ───────────────────────────────────────────────────────────
    env = wrap_env(gym.make(cfg["task"]), reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_act = env.action_space.high[0]

    task_short = _task_short_name(args.task)
    poison_path = f"save_data/{task_short}_poisoned_{eval_args.attack_type}.poisoned_only.npz"
    clean_path = f"save_data/{task_short}_poisoned_{eval_args.attack_type}.clean_only.npz"
    if not os.path.exists(poison_path) or not os.path.exists(clean_path):
        raise FileNotFoundError(f"Missing unlearning split: {poison_path} or {clean_path}")
    poison_loader, clean_loader = _build_loaders(
        load_npz_as_dict(poison_path), load_npz_as_dict(clean_path), env, args, eval_args, algo)
    if algo == "COptiDICE":
        dataset_statistics = _coptidice_statistics(cfg, clean_loader)
        cfg["coptidice_dataset_statistics"] = dataset_statistics

    # ── Policy model ──────────────────────────────────────────────────────────
    # Kwargs shared by BCQL / CPQ / BEARL
    if algo in ("BCQL", "CPQ", "BEARL"):
        _critic_common = dict(
            state_dim=obs_dim, action_dim=act_dim, max_action=max_act,
            a_hidden_sizes=args.a_hidden_sizes, c_hidden_sizes=args.c_hidden_sizes,
            vae_hidden_sizes=args.vae_hidden_sizes,
            sample_action_num=args.sample_action_num,
            gamma=args.gamma, tau=args.tau, beta=args.beta,
            num_q=args.num_q, num_qc=args.num_qc,
            cost_limit=args.cost_limit, episode_len=args.episode_len,
            device=args.device,
        )

    if algo == "BCQL":
        policy_model = BCQL(
            **_critic_common,
            PID=args.PID, lmbda=args.lmbda, phi=args.phi,
        )

    elif algo == "CPQ":
        policy_model = CPQ(
            **_critic_common,
            qc_scalar=args.qc_scalar,
        )

    elif algo == "BEARL":
        policy_model = BEARL(
            **_critic_common,
            lmbda=args.lmbda,
            mmd_sigma=getattr(args, "mmd_sigma", 50),
            target_mmd_thresh=getattr(args, "target_mmd_thresh", 0.05),
            num_samples_mmd_match=getattr(args, "num_samples_mmd_match", 10),
            PID=args.PID,
            kernel=getattr(args, "kernel", "gaussian"),
            start_update_policy_step=getattr(args, "start_update_policy_step", 20_000),
        )

    elif algo == "BC":
        policy_model = BC(
            state_dim=obs_dim + (getattr(args, "bc_mode", "all") == "multi-task"),
            action_dim=act_dim, max_action=max_act,
            a_hidden_sizes=args.a_hidden_sizes,
            episode_len=args.episode_len, device=args.device,
        )

    elif algo == "CDT":
        policy_model = CDT(
            state_dim=obs_dim, action_dim=act_dim, max_action=max_act,
            seq_len=getattr(args, "seq_len", eval_args.cdt_seq_len),
            episode_len=args.episode_len,
            embedding_dim=getattr(args, "embedding_dim", 128),
            num_layers=getattr(args, "num_layers", 4),
            num_heads=getattr(args, "num_heads", 8),
            attention_dropout=getattr(args, "attention_dropout", 0.0),
            residual_dropout=getattr(args, "residual_dropout", 0.0),
            embedding_dropout=getattr(args, "embedding_dropout", 0.0),
            time_emb=getattr(args, "time_emb", True),
            use_rew=getattr(args, "use_rew", False),
            use_cost=getattr(args, "use_cost", False),
            cost_transform=getattr(args, "cost_transform", False),
            add_cost_feat=getattr(args, "add_cost_feat", False),
            mul_cost_feat=getattr(args, "mul_cost_feat", False),
            cat_cost_feat=getattr(args, "cat_cost_feat", False),
            action_head_layers=getattr(args, "action_head_layers", 1),
            cost_prefix=getattr(args, "cost_prefix", False),
            stochastic=getattr(args, "stochastic", False),
            init_temperature=getattr(args, "init_temperature", 0.1),
            target_entropy=(getattr(args, "target_entropy", None)
                            if getattr(args, "target_entropy", None) is not None else -act_dim),
        )

    elif algo == "COptiDICE":
        policy_model = COptiDICE(
            state_dim=obs_dim, action_dim=act_dim, max_action=max_act,
            f_type=getattr(args, "f_type", "chi2"),
            init_state_propotion=dataset_statistics["init_state_propotion"],
            observations_std=np.asarray(dataset_statistics["observations_std"], dtype=np.float32),
            actions_std=np.asarray(dataset_statistics["actions_std"], dtype=np.float32),
            a_hidden_sizes=args.a_hidden_sizes, c_hidden_sizes=args.c_hidden_sizes,
            gamma=args.gamma,
            alpha=getattr(args, "alpha", 0.5),
            cost_ub_epsilon=getattr(args, "cost_ub_epsilon", 0.01),
            num_nu=getattr(args, "num_nu", 1),
            num_chi=getattr(args, "num_chi", 1),
            cost_limit=args.cost_limit, episode_len=args.episode_len,
            device=args.device,
        )

    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    if algo == "CPQ":
        cfg["checkpoint_restoration"] = {
            "log_alpha": "stored" if "log_alpha" in model_ckpt["model_state"]
            else "legacy checkpoint missing log_alpha; initialized to zero; exact training dual unavailable"
        }
    elif algo == "BEARL":
        cfg["checkpoint_restoration"] = {
            "log_alpha": "native checkpoint does not persist this scalar; initialized from constructor",
            "n_train_steps": "native checkpoint does not persist this counter; initialized from constructor",
        }

    if algo == "COptiDICE":
        incompatible = policy_model.load_state_dict(model_ckpt["model_state"], strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        invalid_missing = missing - {"tau", "lmbda"}
        if invalid_missing or unexpected:
            raise RuntimeError(
                "Incompatible COptiDICE checkpoint: "
                f"missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}"
            )
        if missing:
            print("[COptiDICE] Loaded legacy checkpoint without persisted tau/lmbda.")
        cfg["checkpoint_restoration"] = {
            key: "constructor default; missing from source checkpoint" if key in missing
            else "stored source checkpoint value" for key in ("tau", "lmbda")
        }
    else:
        policy_model.load_state_dict(model_ckpt["model_state"])
    policy_model.to(args.device)
    print(f"[{algo}]  {sum(p.numel() for p in policy_model.parameters()):,} parameters")

    # Native optimizer settings are restored; optimizer histories start fresh.
    if algo == "CDT" and policy_model.stochastic:
        cfg["checkpoint_restoration"] = {
            "temperature": "initialized from saved config; not persisted in source model state"}
    trainer = _build_trainer(policy_model, env, algo, args, eval_args)
    configure_unlearning(trainer, hyperparameters)
    cfg["unlearning"]["effective_runtime"] = _runtime_parameters(trainer, algo, args)
    cfg["unlearning"]["evaluation_protocol"] = {
        "mode": "native trainer evaluation",
        "periodic_episodes": eval_args.eval_episodes,
        "milestone_episodes": eval_args.milestone_eval_episodes or eval_args.eval_episodes,
        "episode_len": policy_model.episode_len,
        "reward_cost_units": "raw environment episode sums",
    }

    policy_model.train()
    print(f"Unlearn method: {eval_args.unlearn_method}")

    # # ── Relabeling unlearner  (critic-bearing algos only) ─────────────────────
    # if eval_args.unlearn_method == "relabel":
    #     if algo not in ("BCQL", "CPQ", "BEARL"):
    #         raise ValueError(
    #             f"Relabeling requires Q-critics; not supported for {algo}.")
    #     unlearner = RelabelingUnlearner(
    #         model=policy_model,
    #         state_dim=obs_dim, action_dim=act_dim, device=args.device,
    #         use_feature_encoder=eval_args.use_feature_encoder,
    #         feature_latent_dim=eval_args.feature_latent_dim,
    #         relabel_strategy=eval_args.relabel_strategy,
    #         k_neighbors=eval_args.k_neighbors,
    #         temperature=eval_args.temperature,
    #         use_confidence_weighting=eval_args.use_confidence_weighting,
    #         num_ensemble=eval_args.num_ensemble,
    #         use_gradual_relabeling=eval_args.use_gradual_relabeling,
    #         gradual_steps=eval_args.unlearn_steps,
    #         encoder_lr=eval_args.encoder_lr,
    #         encoder_update_freq=1,
    #     )
    #     print(f"RelabelingUnlearner initialised: strategy={eval_args.relabel_strategy}")
    # else:
    #     unlearner = None

    # # ── Checkpoint / logger ───────────────────────────────────────────────────
    # def checkpoint_fn():
    #     ckpt = {"model_state": policy_model.state_dict()}
    #     if unlearner is not None:
    #         if unlearner.feature_encoder is not None:
    #             ckpt["feature_encoder_state"] = unlearner.feature_encoder.state_dict()
    #         if unlearner.q_ensemble is not None:
    #             ckpt["q_ensemble_state"] = unlearner.q_ensemble.state_dict()
    #     return ckpt

    warmup_available = (
        eval_args.unlearn_method in ("safe", "reward-only")
        and algo in ("BCQL", "CPQ", "BEARL")
        and hasattr(trainer, "critic_warmup_step"))
    plan = _update_plan(eval_args, warmup_available)
    milestones = {int(value.strip()) for value in eval_args.milestone_updates.split(",")
                  if value.strip()}
    if any(value < 0 for value in milestones):
        raise ValueError("Milestone update counts must be nonnegative")
    cfg["unlearning"]["effective_update_plan"] = plan
    cfg["unlearning"]["transition_reward_scale"] = args.reward_scale
    cfg["unlearning"]["transition_cost_scale"] = args.cost_scale
    os.makedirs(args.logdir, exist_ok=True)
    metrics_path = os.path.join(args.logdir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        raise FileExistsError(f"Refusing to overwrite an existing run: {metrics_path}")
    with open(os.path.join(args.logdir, "protocol.json"), "w") as protocol_file:
        json.dump({"source_checkpoint": os.path.abspath(eval_args.unlearn_path),
                   "source_config": source_config, "config": cfg,
                   "eval_config": asdict(eval_args), "update_plan": plan},
                  protocol_file, indent=2, default=str)
    # FSRL appends its name to log_dir; split the exact requested output path.
    logger = WandbLogger(cfg, args.project, args.group,
                        os.path.basename(args.logdir), os.path.dirname(args.logdir))
    logger.name = args.name
    logger.save_config(cfg, verbose=args.verbose)
    run_state = {"global_step": 0, "phase": "initial", "phase_step": 0}

    def checkpoint_fn():
        return {"model_state": policy_model.state_dict(), **run_state,
                "seed": args.seed, "unlearning": cfg["unlearning"]}

    logger.setup_checkpoint_fn(checkpoint_fn)

    poison_iter = iter(poison_loader)
    clean_iter  = iter(clean_loader)

    # ── Reference model  (critic-bearing algos, non-relabel methods) ──────────
    if eval_args.unlearn_method != "relabel" and algo in ("BCQL", "CPQ", "BEARL"):
        ref_model = deepcopy(trainer.model).to(args.device)
        ref_model.critic_ref      = deepcopy(trainer.model.critic).eval()
        ref_model.cost_critic_ref = deepcopy(trainer.model.cost_critic).eval()
        for p in ref_model.critic_ref.parameters():      p.requires_grad = False
        for p in ref_model.cost_critic_ref.parameters(): p.requires_grad = False
    else:
        ref_model = None

    # Global updates include forgetting, critic warm-up, and clean convergence.
    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    run_start = time.perf_counter()
    update_time_s, evaluation_time_s = 0.0, 0.0
    phase_times = {}
    checkpoint_dir = os.path.join(args.logdir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_file = open(metrics_path, "x", buffering=1)

    def write_record(record):
        record.update(seed=args.seed, algo=algo, method=eval_args.unlearn_method)
        metrics_file.write(json.dumps(
            record, default=lambda value: value.item() if hasattr(value, "item")
            else str(value)) + "\n")

    def evaluate_and_log():
        nonlocal best_reward, best_cost, best_idx, evaluation_time_s
        evaluation_start = time.perf_counter()
        evaluation_episodes = (
            eval_args.milestone_eval_episodes
            if run_state["global_step"] in milestones and eval_args.milestone_eval_episodes
            else eval_args.eval_episodes)
        previous_mode = trainer.model.training
        ret, cost_val, length = _evaluate(
            trainer, algo, eval_args, args, episodes=evaluation_episodes)
        trainer.model.train(previous_mode)
        evaluation_time_s += time.perf_counter() - evaluation_start
        feasible = cost_val <= args.cost_limit
        logger.store(tab="eval", Cost=cost_val, Reward=ret, Length=length)
        logger.store(tab="progress", global_step=run_state["global_step"],
                     phase_step=run_state["phase_step"])
        logger.save_checkpoint()
        # Explicit milestone names are independent of logger suffix conventions.
        if (run_state["global_step"] in milestones
                or run_state["global_step"] == plan["total_updates"]):
            torch.save(checkpoint_fn(), os.path.join(
                checkpoint_dir, f"step_{run_state['global_step']:06d}.pt"))
        if feasible and ret > best_reward:
            best_reward, best_cost, best_idx = ret, cost_val, run_state["global_step"]
            logger.save_checkpoint(suffix="best")
        logger.store(tab="train", best_idx=best_idx)
        logger.write(run_state["global_step"], display=False)
        record = {"type": "eval", **run_state, "reward": float(ret),
                  "cost": float(cost_val), "length": float(length),
                  "feasible": bool(feasible), "cost_limit": args.cost_limit,
                  "elapsed_wall_s": time.perf_counter() - run_start,
                  "update_time_s": update_time_s,
                  "evaluation_time_s": evaluation_time_s,
                  "eval_episodes": evaluation_episodes,
                  "eval_mode": "native trainer evaluation"}
        write_record(record)
        print(f"Evaluation at global update {run_state['global_step']} "
              f"({run_state['phase']}): reward={ret:.3f}, cost={cost_val:.3f}")

    def record_update(phase, phase_step, update_start, metrics=None):
        nonlocal update_time_s
        update_time_s += time.perf_counter() - update_start
        run_state.update(global_step=run_state["global_step"] + 1,
                         phase=phase, phase_step=phase_step)
        write_record({"type": "train", **run_state,
                      "elapsed_wall_s": time.perf_counter() - run_start,
                      "update_time_s": update_time_s, **(metrics or {})})
        if (run_state["global_step"] % eval_args.eval_every == 0
                or run_state["global_step"] in milestones
                or run_state["global_step"] == plan["total_updates"]):
            evaluate_and_log()

    def next_batch(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    try:
        evaluate_and_log()
        phase_start = time.perf_counter()
        for step in trange(plan["unlearning"], desc="Phase 1: Unlearning"):
            update_start = time.perf_counter()
            batch_forget, poison_iter = next_batch(poison_loader, poison_iter)
            batch_keep, clean_iter = next_batch(clean_loader, clean_iter)
            metrics = {}
            if eval_args.unlearn_method in ("safe", "reward-only"):
                if (algo in ("BCQL", "CPQ", "BEARL")
                        and eval_args.fit_behavior_on_retain):
                    _, vae_stats = trainer.model.vae_loss(
                        batch_keep[0].to(args.device), batch_keep[2].to(args.device))
                    metrics.update(vae_stats)
                if algo == "BC":
                    bf = (batch_forget[0], batch_forget[2])
                    bk = (batch_keep[0], batch_keep[2])
                else:
                    bf, bk = batch_forget, batch_keep
                metrics.update(trainer.trajdeleter_unlearn_one_step(
                    bf, bk, update_targets=True, unlearn_step=step,
                    unlearn_method=eval_args.unlearn_method))
            elif eval_args.unlearn_method == "rff":
                if algo not in ("BCQL", "CPQ"):
                    raise ValueError(f"rff unlearn is unsupported for {algo}")
                metrics = trainer.rff_unlearn_one_step(
                    batch_forget, batch_keep,
                    lambda_cost=getattr(args, "lambda_cost", 1.0))
            elif eval_args.unlearn_method == "fine-tune":
                _fine_tune_step(trainer, algo, batch_keep, args.device)
            else:
                raise ValueError(f"Unimplemented unlearning method: {eval_args.unlearn_method}")
            record_update("unlearning", step + 1, update_start, metrics)
        phase_times["unlearning_wall_s"] = time.perf_counter() - phase_start

        if plan["warmup"] or plan["convergence"]:
            trainer.model.train()
            if (eval_args.unlearn_method in ("safe", "reward-only")
                    and algo in ("CPQ", "BEARL")
                    and hasattr(trainer, "reset_log_alpha")):
                trainer.reset_log_alpha(value=0.0)
            phase_start = time.perf_counter()
            for step in trange(plan["warmup"], desc="Phase 2: Critic warm-up"):
                update_start = time.perf_counter()
                batch_keep, clean_iter = next_batch(clean_loader, clean_iter)
                trainer.critic_warmup_step(batch_keep, max_grad_norm=hyperparameters.max_grad_norm)
                record_update("warmup", step + 1, update_start)
            if plan["warmup"]:
                trainer.model.critic_old.load_state_dict(trainer.model.critic.state_dict())
                trainer.model.cost_critic_old.load_state_dict(
                    trainer.model.cost_critic.state_dict())
                if hasattr(trainer.model, "actor_old"):
                    trainer.model.actor_old.load_state_dict(trainer.model.actor.state_dict())
            phase_times["warmup_wall_s"] = time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            for step in trange(plan["convergence"], desc="Phase 2: Convergence"):
                update_start = time.perf_counter()
                batch_keep, clean_iter = next_batch(clean_loader, clean_iter)
                _fine_tune_step(trainer, algo, batch_keep, args.device)
                record_update("convergence", step + 1, update_start)
            phase_times["convergence_wall_s"] = time.perf_counter() - phase_start

        if run_state["global_step"] != plan["total_updates"]:
            raise RuntimeError(f"Update budget mismatch: {run_state} versus {plan}")
        summary = {"type": "summary", **run_state, "update_plan": plan,
                   "elapsed_wall_s": time.perf_counter() - run_start,
                   "update_time_s": update_time_s,
                   "evaluation_time_s": evaluation_time_s, **phase_times,
                   "best_feasible_reward": best_reward if np.isfinite(best_reward) else None,
                   "best_feasible_cost": best_cost if np.isfinite(best_cost) else None,
                   "best_global_step": best_idx}
        write_record(summary)
        with open(os.path.join(args.logdir, "summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=2)
        print(f"Training complete: {run_state['global_step']} updates. Logs: {args.logdir}")
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
