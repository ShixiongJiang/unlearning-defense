''''
python unlearn_policy.py --unlearn_path logs/OfflineCarCircle-v0-cost-10/BCQL_poisoned_maxcost-3f1f/BCQL_poisoned_maxcost-3f1f --eval_episodes 20
'''


import os, types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import numpy as np
import torch, pyrallis, gymnasium as gym
import bullet_safety_gym, dsrl  # register envs
from dsrl.infos import DENSITY_CFG
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange
from examples.configs.bcql_configs import BCQL_DEFAULT_CONFIG, BCQLTrainConfig
from osrl.algorithms import BCQL, BCQLTrainer
from osrl.common import TransitionDataset
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, seed_all
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from osrl.common.exp_util import load_config_and_model, seed_all
import torch.nn as nn
from utils.unlearn_utils import BCQLWithUnlearn, BCQLTrainerWithUnlearn
# import os, inspect, osrl
# print(os.path.dirname(inspect.getfile(osrl)))
# assert 1==0
def load_npz_as_dict(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d.pop("poison_mask", None)
    return d


@dataclass
class EvalConfig:
    unlearn_path: str = "log/.../checkpoint/model.pt"
    eval_episodes: int = 20
    best: bool = True
    unlearn_steps: int = 5000
    converge_steps: int = 5000
    eval_every: int = 500



@pyrallis.wrap()
def main(eval_args: EvalConfig):
    cfg, model = load_config_and_model(eval_args.unlearn_path, eval_args.best)
    # print(cfg["task"])
    cfg['logdir'] = 'unlearn_' + cfg['logdir']
    args = types.SimpleNamespace(**cfg)


    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    bcql_model = BCQLWithUnlearn(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        vae_hidden_sizes=cfg["vae_hidden_sizes"],
        sample_action_num=cfg["sample_action_num"],
        PID=cfg["PID"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        lmbda=cfg["lmbda"],
        beta=cfg["beta"],
        phi=cfg["phi"],
        num_q=cfg["num_q"],
        num_qc=cfg["num_qc"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    bcql_model.load_state_dict(model["model_state"])
    bcql_model.to(args.device)

    trainer = BCQLTrainerWithUnlearn(bcql_model,
                          env,
                          reward_scale=cfg["reward_scale"],
                          cost_scale=cfg["cost_scale"],
                          device=args.device)
    
    print(f"Total parameters: {sum(p.numel() for p in bcql_model.parameters())}")

    def checkpoint_fn(): return {"model_state": bcql_model.state_dict()}

    

    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)
    logger.setup_checkpoint_fn(checkpoint_fn)

    
    # load clean dataset 
    # data = env.get_dataset()
    # env.set_target_cost(args.cost_limit)
    # cbins, rbins, max_npb, min_npb = None, None, None, None
    # if args.density != 1.0:
    #     density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
    #     cbins = density_cfg["cbins"]
    #     rbins = density_cfg["rbins"]
    #     max_npb = density_cfg["max_npb"]
    #     min_npb = density_cfg["min_npb"]
    # clean_data = env.pre_process_data(data,
    #                             args.outliers_percent,
    #                             args.noise_scale,
    #                             args.inpaint_ranges,
    #                             args.epsilon,
    #                             args.density,
    #                             cbins=cbins,
    #                             rbins=rbins,
    #                             max_npb=max_npb,
    #                             min_npb=min_npb)

    # load poisoned-only dataset
    poison_only_path = "save_data/offline_car_circle_poisoned.poisoned_only.npz"  # <- set your path
    assert os.path.exists(poison_only_path), f"Poison file not found: {poison_only_path}"
    poison_only_data = load_npz_as_dict(poison_only_path)   # <-- poisoned-only

    cbins = rbins = max_npb = min_npb = None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    poison_only_data = env.pre_process_data(poison_only_data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    poison_only_trainloader = DataLoader(TransitionDataset(poison_only_data),
                             batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    poison_only_trainloader_iter = iter(poison_only_trainloader)

    # load clean-only dataset
    clean_only_path = "save_data/offline_car_circle_poisoned.clean_only.npz"  # <- set your path
    assert os.path.exists(clean_only_path), f"Poison file not found: {clean_only_path}"
    clean_only_data = load_npz_as_dict(clean_only_path)   # <-- poisoned-only

    cbins = rbins = max_npb = min_npb = None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    clean_only_data = env.pre_process_data(clean_only_data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    clean_only_trainloader = DataLoader(TransitionDataset(clean_only_data),
                             batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    clean_only_trainloader_iter = iter(clean_only_trainloader)

    # unlearn steps
    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    for step in trange(eval_args.unlearn_steps, desc="Phase 1: TrajDeleter Unlearn"):
        # get a mini-batch from D_m (forget) and D_f (keep)
        try:
            batch_forget = next(poison_only_trainloader_iter)
        except StopIteration:
            poison_only_trainloader_iter = iter(poison_only_trainloader)
            batch_forget = next(poison_only_trainloader_iter)

        try:
            batch_keep = next(clean_only_trainloader_iter)
        except StopIteration:
            clean_only_trainloader_iter = iter(clean_only_trainloader)
            batch_keep = next(clean_only_trainloader_iter)

        trainer.trajdeleter_unlearn_one_step(
            batch_forget, batch_keep,
            lambda_cost=getattr(args, "lambda_cost", 1.0),
            K=getattr(args, "adv_sample_K", None),
        )

        if (step + 1) % eval_args.eval_every == 0 or eval_args.unlearn_steps- 1:
            ret, cost, length = trainer.evaluate(args.eval_episodes)
            logger.store(tab="eval_unlearn", Cost=cost, Reward=ret, Length=length)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


    # relearn the clean data steps
    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    for step in trange(eval_args.converge_steps, desc="Training"):
        batch = next(clean_only_trainloader_iter)
        observations, next_observations, actions, rewards, costs, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, costs,
                               done)

        # evaluation
        if (step + 1) % eval_args.eval_every == 0 or step == eval_args.converge_steps - 1:
            ret, cost, length = trainer.evaluate(args.eval_episodes)
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    main()
