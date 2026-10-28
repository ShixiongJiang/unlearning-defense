"""Real-model checks for BEARL/COptiDICE shared unlearning controls."""
import copy
from dataclasses import replace
import unittest

import numpy as np
import torch

from osrl.algorithms import BEARL, COptiDICE
from osrl.algorithms.coptidice import get_f_div_fn
from utils.unlearn_config import UnlearnHyperparameters, configure_unlearning
from utils.unlearn_utils import BEARLTrainerWithUnlearn, COptiDICETrainerWithUnlearn


class RealBackendControlsTests(unittest.TestCase):
    implementations = ((BEARL, BEARLTrainerWithUnlearn),
                       (COptiDICE, COptiDICETrainerWithUnlearn))

    def model(self, cls):
        torch.manual_seed(17)
        common = dict(state_dim=3, action_dim=2, max_action=1.,
                      a_hidden_sizes=[8], c_hidden_sizes=[8],
                      episode_len=10, cost_limit=1., device="cpu")
        if cls is BEARL:
            return cls(**common, vae_hidden_sizes=8, sample_action_num=2,
                       num_q=1, num_qc=1, num_samples_mmd_match=2,
                       mmd_sigma=3., target_mmd_thresh=.07,
                       start_update_policy_step=0)
        model = cls(**common, f_type="softchi", init_state_propotion=.25,
                    observations_std=np.array([.3, 2., .7], dtype=np.float32),
                    actions_std=np.array([.2, .8], dtype=np.float32),
                    num_nu=1, num_chi=1)
        with torch.no_grad():
            for parameter in model.chi_network.parameters():
                parameter.zero_()
            model.chi_network.q_nets[0][-2].bias.fill_(2.)
        return model

    def batches(self, cls):
        torch.manual_seed(23)
        def batch():
            result = (torch.randn(8, 3), torch.randn(8, 3),
                      torch.rand(8, 2) * 2 - 1, torch.rand(8).double(),
                      torch.rand(8).double(), torch.zeros(8))
            if cls is COptiDICE:
                result += (torch.tensor([1., 0., 0., 0., 1., 0., 0., 0.]),)
            return result
        return batch(), batch()

    def step(self, template, trainer_cls, *, hp=None, batches=None, **kwargs):
        model = copy.deepcopy(template)
        trainer = trainer_cls(model, None, device="cpu")
        configure_unlearning(trainer, hp or UnlearnHyperparameters())
        batches = batches or self.batches(type(template))
        torch.manual_seed(31)
        metrics = trainer.trajdeleter_unlearn_one_step(*batches, **kwargs)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in model.state_dict().values()))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        return model, metrics

    @staticmethod
    def actor_gradient(model):
        return torch.cat([parameter.grad.flatten()
                          for parameter in model.actor.parameters()])

    def test_configured_fixed_eta_is_applied_to_real_models(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                model, metrics = self.step(template, trainer_cls,
                    hp=UnlearnHyperparameters(eta_mode="fixed", eta_fixed=.3))
                self.assertEqual(metrics["eta_effective"], .3)
                self.assertGreater(self.actor_gradient(model).norm().item(), 0.)
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                    zip(template.actor.parameters(), model.actor.parameters())))

    def test_per_step_override_beats_configured_eta(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                _, metrics = self.step(self.model(cls), trainer_cls,
                    hp=UnlearnHyperparameters(eta_mode="fixed", eta_fixed=.3),
                    eta_fixed=.7)
                self.assertEqual(metrics["eta_effective"], .7)

    def test_fixed_eta_changes_actual_actor_gradient(self):
        hp = UnlearnHyperparameters(eta_mode="fixed", max_grad_norm=0.,
                                    fisher_clip=100., fisher_damping=1.)
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                low, _ = self.step(template, trainer_cls, hp=hp, eta_fixed=.2)
                high, _ = self.step(template, trainer_cls, hp=hp, eta_fixed=.8)
                self.assertFalse(torch.equal(self.actor_gradient(low),
                                             self.actor_gradient(high)))

    def test_cost_scale_changes_forget_update_preserves_native_value_learning(self):
        hp = UnlearnHyperparameters(eta_mode="fixed", max_grad_norm=0.,
                                    fisher_clip=100., fisher_damping=1.)
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                off, _ = self.step(template, trainer_cls, hp=hp, cost_forget_scale=0.)
                on, _ = self.step(template, trainer_cls, hp=hp, cost_forget_scale=2.)
                self.assertFalse(torch.equal(self.actor_gradient(off),
                                             self.actor_gradient(on)))
                if cls is BEARL:
                    self.assertTrue(all(torch.equal(a, b) for a, b in
                                        zip(off.critic.parameters(), on.critic.parameters())))
                    self.assertTrue(any(not torch.equal(a.grad, b.grad) for a, b in
                        zip(off.cost_critic.parameters(), on.cost_critic.parameters())))
                else:
                    for name in ("nu_network", "chi_network"):
                        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(getattr(off, name).parameters(), getattr(on, name).parameters())))
                    torch.testing.assert_close(off.lmbda, on.lmbda, rtol=0, atol=0)
                    torch.testing.assert_close(off.tau, on.tau, rtol=0, atol=0)

    def test_reward_only_matches_explicit_zero_cost_scale(self):
        for cls, trainer_cls in self.implementations:
            with self.subTest(algo=cls.__name__):
                template = self.model(cls)
                off, _ = self.step(template, trainer_cls, cost_forget_scale=0.)
                reward, metrics = self.step(template, trainer_cls,
                                             unlearn_method="reward-only")
                self.assertEqual(metrics["cost_forget_scale"], 0.)
                for key, value in off.state_dict().items():
                    torch.testing.assert_close(value, reward.state_dict()[key], rtol=0, atol=0)
                value_network = "critic" if cls is BEARL else "nu_network"
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                    getattr(template, value_network).parameters(),
                    getattr(reward, value_network).parameters())))

    def test_bearl_behavior_weight_controls_mmd_gradient_not_native_dual(self):
        template = self.model(BEARL)
        hp = UnlearnHyperparameters(alpha_pg_keep=0., keep_cost_weight=0.,
                                    eta_mode="fixed", eta_fixed=0., max_grad_norm=0.)
        off, _ = self.step(template, BEARLTrainerWithUnlearn,
                            hp=replace(hp, keep_bc_weight=0.))
        on, _ = self.step(template, BEARLTrainerWithUnlearn,
                           hp=replace(hp, keep_bc_weight=1.))
        self.assertEqual(self.actor_gradient(off).norm().item(), 0.)
        self.assertGreater(self.actor_gradient(on).norm().item(), 0.)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(template.actor.parameters(), off.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(template.actor.parameters(), on.actor.parameters())))
        torch.testing.assert_close(off.log_alpha, on.log_alpha, rtol=0, atol=0)
        self.assertEqual(off.n_train_steps, on.n_train_steps)

    def test_coptidice_occupancy_target_controls_adaptive_eta(self):
        template = self.model(COptiDICE)
        hp = UnlearnHyperparameters(coptidice_forget_weight_target=0.)
        _, low_target = self.step(template, COptiDICETrainerWithUnlearn, hp=hp)
        _, high_target = self.step(template, COptiDICETrainerWithUnlearn,
                                  hp=replace(hp, coptidice_forget_weight_target=10.))
        self.assertGreater(low_target["eta_effective"], high_target["eta_effective"])
        self.assertAlmostEqual(low_target["forget_occupancy_gap"] -
                               high_target["forget_occupancy_gap"], 10.)

    def test_coptidice_negative_chi_cannot_reverse_likelihood_forgetting(self):
        hp = UnlearnHyperparameters(alpha_pg_keep=0., eta_mode="fixed",
                                    max_grad_norm=0., fisher_clip=100., fisher_damping=1.)
        batches = list(self.batches(COptiDICE))
        forget = list(batches[0])
        forget[3] = torch.full_like(forget[3], -100.)
        batches[0] = tuple(forget)
        for chi_value in (-2., 2.):
            with self.subTest(chi=chi_value):
                template = self.model(COptiDICE)
                template.f_fn, template.f_prime_inv_fn = get_f_div_fn("chi2")
                template.observations_std.zero_()
                template.actions_std.zero_()
                with torch.no_grad():
                    template.chi_network.q_nets[0][-2].bias.fill_(chi_value)
                def log_probability(model):
                    with torch.no_grad():
                        _, _, distribution = model.actor(forget[0], False, True, True)
                        return distribution.log_prob(forget[2]).sum(-1).mean().item()
                before = log_probability(template)
                model, metrics = self.step(template, COptiDICETrainerWithUnlearn,
                                            hp=hp, batches=tuple(batches))
                self.assertEqual(metrics["w_forget_mean"], 0.)
                if chi_value < 0:
                    self.assertEqual(metrics["loss_forget_cost"], 0.)
                    self.assertEqual(log_probability(model), before)
                else:
                    self.assertLess(log_probability(model), before)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
