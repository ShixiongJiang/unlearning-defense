"""
python scripts/poison/train_adv_policy.py --task OfflineCarCircle-v0

Self-contained PPO adversary trainer compatible with gymnasium 0.28.1.
No stable-baselines3 dependency.
"""

import os
import types
from dataclasses import asdict
from typing import List, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.distributions import Normal
from tqdm.auto import trange

from configs.bcql_configs import BCQL_DEFAULT_CONFIG, BCQLTrainConfig
from osrl.common.exp_util import auto_name, seed_all

# ── Actor-Critic network ──────────────────────────────────────────────────────


def _mlp(sizes: List[int], activation: nn.Module) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [activation()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for PPO.

    Actor outputs a Gaussian policy with state-dependent mean and a
    learned (state-independent) log-std.  Critic outputs a scalar value.
    """

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

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return logp, entropy, value


# ── Rollout buffer ────────────────────────────────────────────────────────────


class RolloutBuffer:
    """Stores one PPO rollout and computes GAE advantages."""

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        act_dim: int,
        gamma: float,
        gae_lambda: float,
        device: str,
    ):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.logps = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, value, logp, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logps[self.ptr] = logp
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float):
        """GAE-lambda advantage estimation."""
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_batches(self, batch_size: int):
        """Yield random mini-batches as tensors."""
        indices = np.random.permutation(self.n_steps)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                t(self.obs[idx]),
                t(self.actions[idx]),
                t(self.logps[idx]),
                t(self.returns[idx]),
                t(self.advantages[idx]),
            )


# ── PPO trainer ───────────────────────────────────────────────────────────────


class PPOTrainer:
    def __init__(
        self,
        policy: ActorCritic,
        lr: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        obs_dim: int = 1,
        act_dim: int = 1,
        device: str = "cpu",
    ):
        self.policy = policy
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optim = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
        )

    @torch.no_grad()
    def collect_rollout(self, env: gym.Env, obs: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Fill the rollout buffer with n_steps transitions.

        Returns the next obs and episode statistics collected during rollout.
        """
        self.policy.eval()
        ep_stats = {"rets": [], "costs": [], "lens": []}
        ep_ret = ep_cost = ep_len = 0.0

        for _ in range(self.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            act, logp = self.policy.get_action(obs_t)
            value = self.policy.get_value(obs_t)

            act_np = act.cpu().numpy()
            obs_next, reward, terminated, truncated, info = env.step(act_np)
            done = terminated or truncated

            # reward is already relabelled (cost) by the wrapper
            self.buffer.add(
                obs=obs,
                action=act_np,
                reward=float(reward),
                value=float(value.cpu()),
                logp=float(logp.cpu()),
                done=float(done),
            )

            ep_ret += info.get("true_reward", float(reward))
            ep_cost += info.get("true_cost", float(reward))
            ep_len += 1

            obs = obs_next
            if done:
                ep_stats["rets"].append(ep_ret)
                ep_stats["costs"].append(ep_cost)
                ep_stats["lens"].append(ep_len)
                ep_ret = ep_cost = ep_len = 0.0
                obs, _ = env.reset()

        # bootstrap last value for GAE
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        last_value = float(self.policy.get_value(obs_t).cpu())
        self.buffer.compute_returns_and_advantages(last_value)
        return obs, ep_stats

    def update(self) -> dict:
        """Run n_epochs of PPO mini-batch updates on the current buffer."""
        self.policy.train()
        pg_losses, vf_losses, ent_losses, clip_fracs = [], [], [], []

        for _ in range(self.n_epochs):
            for obs_b, act_b, logp_old_b, ret_b, adv_b in self.buffer.get_batches(
                self.batch_size
            ):
                # normalise advantages per mini-batch (standard PPO practice)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                logp_new, entropy, value = self.policy.evaluate_actions(obs_b, act_b)
                ratio = (logp_new - logp_old_b).exp()

                # clipped policy loss
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * ratio.clamp(
                    1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                vf_loss = 0.5 * (ret_b - value).pow(2).mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy.mean()

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()

                clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(entropy.mean().item())
                clip_fracs.append(clip_frac.item())

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(ent_losses)),
            "clip_frac": float(np.mean(clip_fracs)),
        }


# ── Cost-as-reward wrapper ────────────────────────────────────────────────────


class CostAsRewardWrapper(gym.Wrapper):
    """Replaces reward with info["cost"] so PPO maximises cost directly."""

    def __init__(self, env: gym.Env, cost_scale: float = 1.0):
        super().__init__(env)
        self.cost_scale = cost_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0) * self.cost_scale
        info["true_reward"] = float(reward)
        info["true_cost"] = float(cost)
        return obs, cost, terminated, truncated, info


# ── Main ──────────────────────────────────────────────────────────────────────


@pyrallis.wrap()
def train(args: BCQLTrainConfig):
    # ── config ────────────────────────────────────────────────────────────────
    cfg, old_cfg = asdict(args), asdict(BCQLTrainConfig())
    differing_values = {k: cfg[k] for k in cfg if cfg[k] != old_cfg[k]}
    cfg = asdict(BCQL_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    args.suffix = "_adv_ppo_"

    # ── logger ────────────────────────────────────────────────────────────────
    default_cfg = asdict(BCQL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # ── seed / device ─────────────────────────────────────────────────────────
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # ── environments ──────────────────────────────────────────────────────────
    env = gym.make(args.task)
    env.set_target_cost(args.cost_limit)
    env = wrap_env(env, reward_scale=args.reward_scale)
    env = CostAsRewardWrapper(env, cost_scale=args.cost_scale)

    eval_env = gym.make(args.task)
    eval_env.set_target_cost(args.cost_limit)
    eval_env = wrap_env(eval_env, reward_scale=args.reward_scale)
    eval_env = CostAsRewardWrapper(eval_env, cost_scale=args.cost_scale)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ── policy & PPO trainer ──────────────────────────────────────────────────
    hidden_sizes = getattr(args, "a_hidden_sizes", [256, 256])
    policy = ActorCritic(obs_dim, act_dim, hidden_sizes).to(args.device)
    print(f"PPO parameters: {sum(p.numel() for p in policy.parameters())}")

    ppo = PPOTrainer(
        policy=policy,
        lr=getattr(args, "actor_lr", 3e-4),
        n_steps=getattr(args, "ppo_n_steps", 2048),
        batch_size=getattr(args, "batch_size", 256),
        n_epochs=getattr(args, "ppo_n_epochs", 10),
        gamma=getattr(args, "gamma", 0.99),
        gae_lambda=getattr(args, "ppo_gae_lambda", 0.95),
        clip_range=getattr(args, "ppo_clip_range", 0.2),
        vf_coef=getattr(args, "ppo_vf_coef", 0.5),
        ent_coef=getattr(args, "ppo_ent_coef", 0.0),
        max_grad_norm=getattr(args, "ppo_max_grad", 0.5),
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=args.device,
    )

    def checkpoint_fn():
        return {"model_state": policy.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    # ── training loop ─────────────────────────────────────────────────────────
    best_cost = -np.inf
    best_reward = -np.inf
    best_idx = 0
    total_steps = 0

    obs, _ = env.reset()
    n_updates = args.update_steps // ppo.n_steps

    for update in trange(n_updates, desc="PPO updates"):

        # collect one rollout of n_steps transitions
        obs, ep_stats = ppo.collect_rollout(env, obs)
        total_steps += ppo.n_steps

        # gradient update
        train_metrics = ppo.update()

        # log rollout statistics
        if ep_stats["rets"]:
            logger.store(
                tab="rollout",
                EpRet=float(np.mean(ep_stats["rets"])),
                EpCost=float(np.mean(ep_stats["costs"])),
                EpLen=float(np.mean(ep_stats["lens"])),
            )
        logger.store(tab="train", **train_metrics)

        # evaluation
        if (update + 1) % max(
            1, args.eval_every // ppo.n_steps
        ) == 0 or update == n_updates - 1:

            ep_rets, ep_costs, ep_lens = [], [], []
            for _ in range(args.eval_episodes):
                o, _ = eval_env.reset()
                ep_r = ep_c = ep_l = 0.0
                done = False
                while not done:
                    with torch.no_grad():
                        o_t = torch.as_tensor(
                            o, dtype=torch.float32, device=args.device
                        )
                        a, _ = policy.get_action(o_t, deterministic=True)
                    o, _, terminated, truncated, info = eval_env.step(a.cpu().numpy())
                    ep_r += info.get("true_reward", 0.0)
                    ep_c += info.get("true_cost", 0.0)
                    ep_l += 1
                    done = terminated or truncated
                ep_rets.append(ep_r)
                ep_costs.append(ep_c)
                ep_lens.append(ep_l)

            mean_cost = float(np.mean(ep_costs))
            mean_reward = float(np.mean(ep_rets))
            mean_len = float(np.mean(ep_lens))
            logger.store(
                tab="eval", Cost=mean_cost, Reward=mean_reward, Length=mean_len
            )

            logger.save_checkpoint()
            if mean_cost > best_cost:
                best_cost = mean_cost
                best_reward = mean_reward
                best_idx = update
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(total_steps, display=False)
        else:
            logger.write_without_reset(total_steps)

    print(
        f"Best checkpoint at update {best_idx} — "
        f"cost={best_cost:.2f}, reward={best_reward:.2f}"
    )


if __name__ == "__main__":
    train()