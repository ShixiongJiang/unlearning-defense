''''
python train_poison_policy.py --task OfflineCarCircle-v0 
'''


import os, types
from dataclasses import asdict
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


def load_npz_as_dict(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d.pop("poison_mask", None)
    return d

@pyrallis.wrap()
def train_poison_policy(args: BCQLTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCQLTrainConfig())
    differing = {k: cfg[k] for k in cfg if cfg[k] != old_cfg[k]}
    cfg = asdict(BCQL_DEFAULT_CONFIG[args.task]()); cfg.update(differing)
    args = types.SimpleNamespace(**cfg)
    
    args.suffix = 'poisoned_maxcost'  # indicate poisoned training

    if args.name is None:
        args.name = auto_name(asdict(BCQL_DEFAULT_CONFIG[args.task]()), cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)

    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    env = gym.make(args.task)
    

    poison_path = "save_data/offline_car_circle_poisoned_maxcost.npz"  # <- set your path
    assert os.path.exists(poison_path), f"Poison file not found: {poison_path}"
    data = load_npz_as_dict(poison_path)   # <-- poisoned-only

    env.set_target_cost(args.cost_limit)

    cbins = rbins = max_npb = min_npb = None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

     
    env = wrap_env(
            env=env,
            reward_scale=args.reward_scale,
        )
    env = OfflineEnvWrapper(env)
    # model & optimizer setup
    model = BCQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        vae_hidden_sizes=args.vae_hidden_sizes,
        sample_action_num=args.sample_action_num,
        PID=args.PID,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda,
        beta=args.beta,
        phi=args.phi,
        num_q=args.num_q,
        num_qc=args.num_qc,
        cost_limit=args.cost_limit,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn(): return {"model_state": model.state_dict()}
    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = BCQLTrainer(model,
                          env,
                          logger=logger,
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          vae_lr=args.vae_lr,
                          reward_scale=args.reward_scale,
                          cost_scale=args.cost_scale,
                          device=args.device)

    trainloader = DataLoader(TransitionDataset(data),
                             batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    trainloader_iter = iter(trainloader)

    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, next_observations, actions, rewards, costs, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, costs,
                               done)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
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
    train_poison_policy()
