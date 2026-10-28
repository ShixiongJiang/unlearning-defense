"""CPU checks for actor direction and real unlearning ablations."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from osrl.algorithms import BCQL, CPQ
from utils.unlearn_utils import (
    BCQLTrainerWithUnlearn, CPQTrainerWithUnlearn, _actor_step,
    _forget_actor_objective,
)


class ActorDirectionTests(unittest.TestCase):
    def take_step(self, starting_action, safe_threshold, eta):
        actor = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        reward_critic = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        cost_critic = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            actor.weight.fill_(starting_action)
            reward_critic.weight.fill_(1.)
            cost_critic.weight.fill_(1.)
        states = torch.ones(4, 1, dtype=torch.float64)
        actions = actor(states)
        before_reward = reward_critic(actions).mean().item()
        before_cost = cost_critic(actions).mean().item()
        loss, _, _ = _forget_actor_objective(
            reward_critic(actions), cost_critic(actions), safe_threshold)
        params = list(actor.parameters())
        gradients = torch.autograd.grad(loss, params)
        obj = SimpleNamespace(model=SimpleNamespace(
            actor_optim=torch.optim.SGD(params, lr=.02)))
        _actor_step(obj, params, [torch.zeros_like(g) for g in gradients],
                    gradients, eta, max_grad_norm=0.)
        return (before_reward, before_cost,
                reward_critic(actor(states)).mean().item(),
                cost_critic(actor(states)).mean().item(),
                actor.weight.grad.detach().clone())

    def test_unsafe_actor_moves_to_lower_cost(self):
        _, before, _, after, gradient = self.take_step(.8, .25, 1.)
        self.assertLess(after, before)
        self.assertGreater(gradient.norm().item(), 0.)

    def test_safe_actor_improves_reward(self):
        before, _, after, after_cost, _ = self.take_step(.1, 2., 1.)
        self.assertGreater(after, before)
        self.assertLess(after_cost, 2.)

    def test_eta_changes_actual_nonzero_gradient_and_policy(self):
        low = self.take_step(.8, .25, .2)
        high = self.take_step(.8, .25, .8)
        self.assertGreater(low[-1].norm().item(), 0.)
        self.assertGreater(high[-1].norm().item(), low[-1].norm().item())
        self.assertLess(high[3], low[3])


class RealTrainerTests(unittest.TestCase):
    implementations = ((BCQL, BCQLTrainerWithUnlearn),
                       (CPQ, CPQTrainerWithUnlearn))

    def model(self, cls):
        torch.manual_seed(17)
        return cls(state_dim=3, action_dim=2, max_action=1.,
                   a_hidden_sizes=[8], c_hidden_sizes=[8], vae_hidden_sizes=8,
                   sample_action_num=2, num_q=1, num_qc=1,
                   episode_len=10, cost_limit=1., device="cpu")

    def batches(self):
        torch.manual_seed(23)
        def batch():
            return (torch.randn(8, 3), torch.randn(8, 3),
                    torch.rand(8, 2) * 2 - 1, torch.rand(8),
                    torch.rand(8), torch.zeros(8))
        return batch(), batch()

    def step(self, template, trainer_cls, **kwargs):
        model = copy.deepcopy(template)
        trainer = trainer_cls(model, None, device="cpu")
        batches = self.batches()
        torch.manual_seed(31)
        metrics = trainer.trajdeleter_unlearn_one_step(*batches, **kwargs)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in model.state_dict().values()))
        return model, metrics

    def test_real_bcql_cpq_one_step_and_applied_eta(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                model, metrics = self.step(template, trainer_cls,
                    eta_mode="fixed", eta_fixed=.3)
                self.assertEqual(metrics["eta_effective"], .3)
                self.assertTrue(any(not torch.equal(old, new)
                    for old, new in zip(template.actor.parameters(), model.actor.parameters())))
                if cls is CPQ:
                    self.assertEqual(metrics["safe_threshold"], model.q_thres)
                    self.assertLess(model.q_thres, model.qc_thres)

    def test_cost_ablation_changes_cost_gradient_preserves_reward_update(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                off, _ = self.step(template, trainer_cls, cost_forget_scale=0.,
                                   eta_mode="fixed", eta_fixed=1.)
                on, _ = self.step(template, trainer_cls, cost_forget_scale=1.,
                                  eta_mode="fixed", eta_fixed=1.)
                self.assertTrue(all(torch.equal(a, b) for a, b in
                    zip(off.critic.parameters(), on.critic.parameters())))
                self.assertTrue(any(not torch.equal(a.grad, b.grad) for a, b in
                    zip(off.cost_critic.parameters(), on.cost_critic.parameters())))
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                    zip(template.critic.parameters(), off.critic.parameters())))

    def test_eta_changes_real_actor_gradients(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                low, _ = self.step(template, trainer_cls, eta_mode="fixed", eta_fixed=.2)
                high, _ = self.step(template, trainer_cls, eta_mode="fixed", eta_fixed=.8)
                low_grad = torch.cat([p.grad.flatten() for p in low.actor.parameters()])
                high_grad = torch.cat([p.grad.flatten() for p in high.actor.parameters()])
                self.assertGreater(low_grad.norm().item(), 0.)
                self.assertGreater(high_grad.norm().item(), 0.)
                self.assertFalse(torch.equal(low_grad, high_grad))

    def test_reward_only_is_explicit_cost_ablation(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                off, _ = self.step(template, trainer_cls, cost_forget_scale=0.)
                reward_only, metrics = self.step(template, trainer_cls,
                                                  unlearn_method="reward-only")
                self.assertEqual(metrics["cost_forget_scale"], 0.)
                self.assertTrue(all(torch.equal(value, reward_only.state_dict()[key])
                                    for key, value in off.state_dict().items()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
