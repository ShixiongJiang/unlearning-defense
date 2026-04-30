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
from copy import deepcopy
from dataclasses import dataclass
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
    cdt_seq_len:       int   = 10
    cdt_target_return: float = 600.0
    cdt_target_cost:   float = 0.0

    # COptiDICE-specific
    coptidice_init_frac: float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# Standard 6-tuple dataloader  (BCQL / CPQ / BEARL / BC)
# ══════════════════════════════════════════════════════════════════════════════

def load_npz_as_dict(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d.pop("poison_mask", None)
    return d

def _ensure_terminals(data: dict) -> dict:
    """
    If a dataset has no episode-end flags (all False / missing), mark the
    final transition as terminal so dsrl can compute per-episode returns.
    dsrl looks for 'terminals', then 'dones', then 'done'.
    """
    for key in ("terminals", "dones", "done"):
        if key in data:
            arr = data[key].astype(bool)
            if arr.any():          # already has at least one terminal → fine
                return data
            arr[-1] = True         # mark last step as episode end
            data = dict(data)      # shallow copy so we don't mutate caller's dict
            data[key] = arr
            return data
    # No terminal key at all — create one
    data = dict(data)
    terminals = np.zeros(len(data["observations"]), dtype=bool)
    terminals[-1] = True
    data["terminals"] = terminals
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
    return DataLoader(
        TransitionDataset(processed),
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
                 reward_scale: float = 1.0, cost_scale: float = 1.0):
        self.seq_len = seq_len
        obs   = data["observations"].astype(np.float32)
        acts  = data["actions"].astype(np.float32)
        rews  = (data["rewards"] * reward_scale).astype(np.float32)
        costs = (data["costs"]   * cost_scale  ).astype(np.float32)
        dones = data.get(
            "terminals", data.get("dones", data.get(
            "done", np.zeros(len(obs), dtype=bool)))
        ).astype(bool)

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
                         reward_scale=1.0, cost_scale=1.0):
    return DataLoader(
        _CDTSequenceDataset(data, seq_len, reward_scale, cost_scale),
        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COptiDICE 7-tuple dataloader  (appends is_init)
# ══════════════════════════════════════════════════════════════════════════════

class _COptiDICEDataset(Dataset):
    """
    Stores transitions as raw tensors and appends is_init (float32).
    is_init[t] = 1.0  if transition t is the first step of an episode
                       (i.e. done[t-1] was True).
    COptiDICE uses this for its boundary term  (1-γ)·ν(s_0).
 
    Stores raw numpy arrays directly because TransitionDataset does not
    implement __getitem__ and cannot be used as an indexed Dataset.
    """
 
    def __init__(self, processed_data: dict):
        self._obs   = torch.from_numpy(processed_data["observations"].astype(np.float32))
        self._nobs  = torch.from_numpy(processed_data["next_observations"].astype(np.float32))
        self._act   = torch.from_numpy(processed_data["actions"].astype(np.float32))
        self._rew   = torch.from_numpy(processed_data["rewards"].astype(np.float32))
        self._cost  = torch.from_numpy(processed_data["costs"].astype(np.float32))
        raw_done    = processed_data.get(
            "terminals", processed_data.get("dones",
            processed_data.get("done", np.zeros(len(self._obs), dtype=np.float32)))
        ).astype(np.float32)
        self._done    = torch.from_numpy(raw_done)
        is_init       = np.roll(raw_done.astype(bool), 1).astype(np.float32)
        is_init[0]    = 1.0
        self._is_init = torch.from_numpy(is_init)
 
    def __len__(self):
        return len(self._obs)
 
    def __getitem__(self, idx):
        return (self._obs[idx], self._nobs[idx], self._act[idx],
                self._rew[idx], self._cost[idx], self._done[idx],
                self._is_init[idx])
 
 
def _make_dataloader_coptidice(data, env, args, batch_size):
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
        _COptiDICEDataset(processed),
        batch_size=batch_size, pin_memory=True, num_workers=args.num_workers,
    )
 

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

def _evaluate(trainer, algo: str, eval_args: EvalConfig, args) -> tuple:
    """
    CDT.evaluate() requires (n_episodes, target_return, target_cost).
    Every other trainer uses evaluate(n_episodes).
    """
    if algo == "CDT":
        return trainer.evaluate(
            args.eval_episodes,
            target_return=eval_args.cdt_target_return * args.reward_scale,
            target_cost=eval_args.cdt_target_cost     * args.cost_scale,
        )
    return trainer.evaluate(args.eval_episodes)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@pyrallis.wrap()
def main(eval_args: EvalConfig):

    # ── Load checkpoint ───────────────────────────────────────────────────────
    cfg, model_ckpt = load_config_and_model(eval_args.unlearn_path, False)
    cfg["logdir"]   = "new_unlearn_" + cfg["logdir"]
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

    args.name   = args.name + str(args.task) + "_unlearn_" + str(eval_args.unlearn_method)
    args.logdir = os.path.join(args.logdir, args.name)

    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # ── Environment ───────────────────────────────────────────────────────────
    env = wrap_env(gym.make(cfg["task"]), reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_act = env.action_space.high[0]

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
            state_dim=obs_dim, action_dim=act_dim, max_action=max_act,
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
            target_entropy=getattr(args, "target_entropy", None),
        )

    elif algo == "COptiDICE":
        obs_std = np.ones(obs_dim, dtype=np.float32)
        act_std = np.ones(act_dim, dtype=np.float32)
        policy_model = COptiDICE(
            state_dim=obs_dim, action_dim=act_dim, max_action=max_act,
            f_type=getattr(args, "f_type", "chi2"),
            init_state_propotion=getattr(args, "init_state_propotion",
                                         eval_args.coptidice_init_frac),
            observations_std=obs_std, actions_std=act_std,
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

    policy_model.load_state_dict(model_ckpt["model_state"])
    policy_model.to(args.device)
    print(f"[{algo}]  {sum(p.numel() for p in policy_model.parameters()):,} parameters")

    # ── Trainer ───────────────────────────────────────────────────────────────
    _trainer_common = dict(
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        device=args.device,
    )

    if algo == "BCQL":
        trainer = BCQLTrainerWithUnlearn(policy_model, env, **_trainer_common)

    elif algo == "CPQ":
        trainer = CPQTrainerWithUnlearn(policy_model, env, **_trainer_common)

    elif algo == "BEARL":
        trainer = BEARLTrainerWithUnlearn(policy_model, env, **_trainer_common)

    elif algo == "BC":
        trainer = BCTrainerWithUnlearn(
            policy_model, env,
            cost_limit=args.cost_limit, device=args.device,
        )

    elif algo == "CDT":
        trainer = CDTTrainerWithUnlearn(
            policy_model, env,
            learning_rate=getattr(args, "learning_rate", 1e-4),
            weight_decay=getattr(args, "weight_decay", 1e-4),
            clip_grad=getattr(args, "clip_grad", 0.25),
            lr_warmup_steps=getattr(args, "lr_warmup_steps", 10_000),
            loss_cost_weight=getattr(args, "loss_cost_weight", 0.0),
            loss_state_weight=getattr(args, "loss_state_weight", 0.0),
            cost_reverse=getattr(args, "cost_reverse", False),
            no_entropy=getattr(args, "no_entropy", False),
            **_trainer_common,
        )

    elif algo == "COptiDICE":
        trainer = COptiDICETrainerWithUnlearn(
            policy_model, env,
            actor_lr=getattr(args, "actor_lr", 1e-3),
            critic_lr=getattr(args, "critic_lr", 1e-3),
            scalar_lr=getattr(args, "scalar_lr", 1e-3),
            **_trainer_common,
        )

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

    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)
    # logger.setup_checkpoint_fn(checkpoint_fn)

    # ── Datasets ──────────────────────────────────────────────────────────────
    task_short  = _task_short_name(args.task)
    poison_path = (f"save_data/{task_short}_poisoned_{eval_args.attack_type}"
                   f".poisoned_only.npz")
    clean_path  = (f"save_data/{task_short}_poisoned_{eval_args.attack_type}"
                   f".clean_only.npz")
    assert os.path.exists(poison_path), f"Poison file not found: {poison_path}"
    assert os.path.exists(clean_path),  f"Clean file not found:  {clean_path}"

    poison_raw = load_npz_as_dict(poison_path)
    clean_raw  = load_npz_as_dict(clean_path)

    if algo == "CDT":
        # CDT requires 8-tuple sequence batches
        seq_len = getattr(args, "seq_len", eval_args.cdt_seq_len)
        poison_loader = _make_dataloader_cdt(
            poison_raw, seq_len, args.batch_size,
            cfg["reward_scale"], cfg["cost_scale"])
        clean_loader  = _make_dataloader_cdt(
            clean_raw,  seq_len, args.batch_size,
            cfg["reward_scale"], cfg["cost_scale"])

    elif algo == "COptiDICE":
        # COptiDICE requires a 7-tuple with is_init appended
        poison_loader = _make_dataloader_coptidice(
            poison_raw, env, args, args.batch_size)
        clean_loader  = _make_dataloader_coptidice(
            clean_raw,  env, args, args.batch_size)

    else:
        # BCQL / CPQ / BEARL / BC — standard 6-tuple TransitionDataset
        poison_loader = _make_dataloader(poison_raw, env, args, args.batch_size)
        clean_loader  = _make_dataloader(clean_raw,  env, args, args.batch_size)

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

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    unlearn_done = False

    loss_actor_all, loss_steps    = [], []
    cost_eval_hist, eval_steps_p1 = [], []
    _score_streak = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Unlearning
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Starting Phase 1: Unlearning ({eval_args.unlearn_method})")
    print(f"{'='*60}\n")
    start = time.perf_counter()

    for step in trange(eval_args.unlearn_steps, desc="Phase 1: Unlearning"):

        # ── Fetch mini-batches ────────────────────────────────────────────────
        try:
            batch_forget = next(poison_iter)
        except StopIteration:
            poison_iter  = iter(poison_loader)
            batch_forget = next(poison_iter)

        try:
            batch_keep = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_loader)
            batch_keep = next(clean_iter)

        # ── Unlearning step ───────────────────────────────────────────────────
        metrics: dict = {}

        if eval_args.unlearn_method == "safe":
            # BC unlearn expects (obs, act) 2-tuples; standard loader gives 6-tuples.
            # Index 0 = obs, index 2 = act (index 1 is next_obs).
            if algo == "BC":
                bf = (batch_forget[0], batch_forget[2])
                bk = (batch_keep[0],  batch_keep[2])
            else:
                bf, bk = batch_forget, batch_keep
            metrics = trainer.trajdeleter_unlearn_one_step(
                bf, bk,
                update_targets=True,
                unlearn_step=step
            )

        elif eval_args.unlearn_method == "rff":
            # rff_unlearn_one_step only exists on BCQL / CPQ trainers
            if algo not in ("BCQL", "CPQ"):
                raise ValueError(
                    f"rff unlearn is only supported for BCQL/CPQ, not {algo}")
            metrics = trainer.rff_unlearn_one_step(
                batch_forget, batch_keep,
                lambda_cost=getattr(args, "lambda_cost", 1.0),
            )
            loss_actor_all.append(metrics["loss_actor"])
            loss_steps.append(step + 1)

        elif eval_args.unlearn_method == "reward-only":
            if algo == "BC":
                bf = (batch_forget[0], batch_forget[2])
                bk = (batch_keep[0],  batch_keep[2])
            else:
                bf, bk = batch_forget, batch_keep
            metrics = trainer.trajdeleter_unlearn_one_step(
                bf, bk, unlearn_method="reward-only",
            )

        elif eval_args.unlearn_method == "fine-tune":
            _fine_tune_step(trainer, algo, batch_keep, args.device)

      
        # ── Periodic evaluation ───────────────────────────────────────────────
        if (step + 1) % eval_args.eval_every == 0 and algo in ['BC', 'CPQ','BCQL','BEARL']\
                or step == eval_args.unlearn_steps - 1:

            ret, cost_val, length = _evaluate(trainer, algo, eval_args, args)
            cost_eval_hist.append(cost_val)
            eval_steps_p1.append(step + 1)

            logger.store(tab="eval", Cost=cost_val, Reward=ret, Length=length)

            _m = metrics
            logger.store(
                tab="train",
                **{
                
                    "eta_forget": _m.get("eta_forget") if _m.get("eta_forget") is not None else -1.0,   
                    "loss_q_keep": _m.get("loss_q_keep") if _m.get("loss_q_keep") is not None else -1.0,
                    "loss_q_forget": _m.get("loss_q_forget") if _m.get("loss_q_forget") is not None else -1.0,
                    "loss_cq_keep": _m.get("loss_cq_keep") if _m.get("loss_cq_keep") is not None else -1.0,
                    "loss_cq_forget": _m.get("loss_cq_forget") if _m.get("loss_cq_forget") is not None else -1.0,

                    "loss_keep": _m.get("loss_keep") if _m.get("loss_keep") is not None else -1.0,
                    "loss_cost_keep": _m.get("loss_cost_keep") if _m.get("loss_cost_keep") is not None else -1.0,
                    "loss_bc_keep": _m.get("loss_bc_keep") if _m.get("loss_bc_keep") is not None else -1.0,

                    "loss_forget_cost": _m.get("loss_forget_cost") if _m.get("loss_forget_cost") is not None else -1.0,
                    "loss_forget_q": _m.get("loss_forget_q") if _m.get("loss_forget_q") is not None else -1.0,

                    "q_keep_mean": _m.get("q_keep_mean") if _m.get("q_keep_mean") is not None else -1.0,
                    "q_forget_mean": _m.get("q_forget_mean") if _m.get("q_forget_mean") is not None else -1.0,
                    "qc_keep_mean": _m.get("qc_keep_mean") if _m.get("qc_keep_mean") is not None else -1.0,
                    "qc_forget_mean": _m.get("qc_forget_mean") if _m.get("qc_forget_mean") is not None else -1.0,

                    "gn_keep": _m.get("gn_keep") if _m.get("gn_keep") is not None else -1.0,
                    "gn_forget": _m.get("gn_forget") if _m.get("gn_forget") is not None else -1.0,
                }
            )
            logger.save_checkpoint()
            if cost_val < trainer.model.qc_thres and ret > best_reward:
                best_cost   = cost_val
                best_reward = ret
                best_idx    = step
                logger.save_checkpoint(suffix="best")
                print(f"\n✓ New best model at step {step+1}: "
                      f"Cost={cost_val:.2f}, Reward={ret:.2f}")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

            if unlearn_done:
                break
        elif (step + 1) % eval_args.eval_every == 0 and algo in ['CDT']\
            or step == eval_args.unlearn_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                if args.cost_reverse:
                    # critical step, rescale the return!
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        (args.episode_len - cost_return) * args.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_return))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)
            if mean_cost < best_cost or (mean_cost == best_cost
                                         and mean_ret > best_reward):
                best_cost = mean_cost
                best_reward = mean_ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)

    elapsed_p1 = time.perf_counter() - start
    logger.store(tab="total_unlearn_time", unlearn_time_s=elapsed_p1)
    logger.write(step=0, display=False)

    print(f"\n{'='*60}")
    print(f"Phase 1 complete — took {elapsed_p1:.2f}s")
    print(f"Best model: Cost={best_cost:.2f}, Reward={best_reward:.2f} at step {best_idx}")
    print(f"{'='*60}\n")

    # ── Optional plot (safe / relabel) ────────────────────────────────────────
    if eval_args.unlearn_method in ("safe", "relabel") \
            and eval_args.draw and loss_actor_all:
        loss_np = np.asarray(loss_actor_all, dtype=float)
        w = max(5, min(101, (len(loss_np) // 50) * 2 + 1))
        kernel      = np.ones(w) / w
        loss_smooth = np.convolve(loss_np, kernel, mode="valid")
        offset      = w // 2
        steps_smooth = np.asarray(loss_steps)[offset: offset + len(loss_smooth)]

        plt.rcParams.update({
            "font.size": 26, "axes.labelsize": 24, "axes.titlesize": 24,
            "legend.fontsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20,
        })
        fig, ax1 = plt.subplots(figsize=(9, 5))
        line1, = ax1.plot(steps_smooth, loss_smooth,
                          color="tab:blue", linewidth=2.0, label="Actor loss (smoothed)")
        ax1.set_xlabel("Unlearning step")
        ax1.set_ylabel("Actor loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        line2, = ax2.plot(eval_steps_p1, cost_eval_hist,
                          color="tab:orange", linestyle="--", marker="o", markersize=6,
                          linewidth=2.0, label="Cost (eval)")
        ax2.set_ylabel("Cost", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.legend([line1, line2], [l.get_label() for l in [line1, line2]],
                   loc="upper right")
        plt.tight_layout()

        fig_dir  = "/scratch365/sjiang5/unlearning-defense/utils/figure"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{args.name}.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Plot saved to: {fig_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Convergence on clean data
    # ══════════════════════════════════════════════════════════════════════════
    if eval_args.converge_steps > 0:
        print(f"\n{'='*60}")
        print("Starting Phase 2: Convergence on Clean Data")
        print(f"{'='*60}\n")
        trainer.model.train()
        # Reset log_alpha for algos that have it (CPQ, BEARL).
        # Phase 1 can drive it to its clamped minimum, disabling the safety gate.
        # BCQL has no log_alpha; BC / CDT / COptiDICE have none either.
        if eval_args.unlearn_method in ("safe", "reward-only") \
                and algo in ("CPQ", "BEARL") \
                and hasattr(trainer, "reset_log_alpha"):
            trainer.reset_log_alpha(value=0.0)
            print("✓ log_alpha reset to 0.0 before Phase 2")

        # Critic warm-up for BCQL / CPQ / BEARL.
        # Restores Q-value scales so the safety gate is meaningful when
        # actor updates resume.
        WARMUP_STEPS = 300
        if eval_args.unlearn_method in ("safe", "reward-only") \
                and algo in ("BCQL", "CPQ", "BEARL") \
                and hasattr(trainer, "critic_warmup_step"):
            print(f"Starting critic warm-up ({WARMUP_STEPS} steps) …")
            for _ in trange(WARMUP_STEPS, desc="Phase 2: Critic warm-up"):
                try:
                    batch_wu = next(clean_iter)
                except StopIteration:
                    clean_iter = iter(clean_loader)
                    batch_wu   = next(clean_iter)
                trainer.critic_warmup_step(batch_wu, max_grad_norm=1.0)
            # Hard-sync all target networks
            trainer.model.critic_old.load_state_dict(
                trainer.model.critic.state_dict())
            trainer.model.cost_critic_old.load_state_dict(
                trainer.model.cost_critic.state_dict())
            # BCQL and BEARL also have actor_old; CPQ does not
            if hasattr(trainer.model, "actor_old"):
                trainer.model.actor_old.load_state_dict(
                    trainer.model.actor.state_dict())
            print("✓ Critic warm-up complete, target networks synced")

        start = time.perf_counter()

        for step in trange(eval_args.converge_steps, desc="Phase 2: Convergence"):
            try:
                batch_keep = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                batch_keep = next(clean_iter)

            if eval_args.unlearn_method in ("safe", "reward-only", "relabel"):
                _fine_tune_step(trainer, algo, batch_keep, args.device)
                # Periodic hard-sync of critic target networks
                if algo in ("BCQL", "CPQ", "BEARL") and (step + 1) % 10 == 0:
                    trainer.model.critic_old.load_state_dict(
                        trainer.model.critic.state_dict())
                    trainer.model.cost_critic_old.load_state_dict(
                        trainer.model.cost_critic.state_dict())

            elif eval_args.unlearn_method == "fine-tune":
                break

            if (step + 1) % eval_args.eval_every == 0 \
                    or step == eval_args.converge_steps - 1:
                ret, cost_val, length = _evaluate(trainer, algo, eval_args, args)
                logger.store(tab="eval", Cost=cost_val, Reward=ret, Length=length)
                logger.save_checkpoint()
                if cost_val < trainer.model.qc_thres and ret > best_reward:
                    best_cost   = cost_val
                    best_reward = ret
                    best_idx    = step
                    logger.save_checkpoint(suffix="best")
                    print(f"\n✓ New best model at step {step+1}: "
                          f"Cost={cost_val:.2f}, Reward={ret:.2f}")
                logger.store(tab="train", best_idx=best_idx)
                logger.write(step, display=False)
            else:
                logger.write_without_reset(step)

        elapsed_p2 = time.perf_counter() - start
        logger.store(tab="total_converge_time", converge_time_s=elapsed_p2)
        logger.write(step=0, display=False)

        print(f"\n{'='*60}")
        print(f"Phase 2 complete — took {elapsed_p2:.2f}s")
        print(f"Final best model: Cost={best_cost:.2f}, Reward={best_reward:.2f}")
        print(f"{'='*60}\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Algo:             {algo}")
    print(f"Method:           {eval_args.unlearn_method}")
    print(f"Final Best Cost:  {best_cost:.2f}")
    print(f"Final Best Reward:{best_reward:.2f}")
    print(f"Logs saved to:    {args.logdir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()