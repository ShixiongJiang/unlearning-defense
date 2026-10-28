"""Verify configured controls reach actual unlearning updates."""
import copy
from dataclasses import fields, replace
from types import SimpleNamespace
import unittest

import torch

from osrl.algorithms import BCQL, CPQ
from utils.unlearn_config import (
    DEFAULT_UNLEARN, UnlearnHyperparameters, configure_unlearning,
    resolve_unlearn_hyperparameters,
)
from utils.unlearn_utils import (
    BCQLTrainerWithUnlearn, CPQTrainerWithUnlearn,
    _init_unlearn_state, _normalise_and_pcgrad, _fisher_scale_forget,
)


class SharedConfigTests(unittest.TestCase):
    def test_all_numeric_controls_reject_nonfinite_values(self):
        for field in fields(UnlearnHyperparameters):
            if field.name == "eta_mode":
                continue
            with self.subTest(control=field.name):
                with self.assertRaises(ValueError):
                    replace(DEFAULT_UNLEARN, **{field.name: float("nan")})

    def test_invalid_bounds_temperatures_and_decays_are_rejected(self):
        for overrides in [dict(eta_mode="unknown"), dict(eta_min=.5),
                          dict(eta_temperature=0), dict(grad_norm_beta=1),
                          dict(fisher_beta=-.1), dict(reward_forget_quantile=1.1),
                          dict(cdt_forget_cost_confidence_target=-.1)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(DEFAULT_UNLEARN, **overrides)

    def test_none_preserves_configuration_and_explicit_override_wins(self):
        obj = SimpleNamespace()
        hp = replace(DEFAULT_UNLEARN, cost_forget_scale=.37, eta_mode="fixed", eta_fixed=.23)
        configure_unlearning(obj, hp)
        resolved = resolve_unlearn_hyperparameters(obj, cost_forget_scale=None, eta_mode=None)
        self.assertEqual(resolved, hp)
        overridden = resolve_unlearn_hyperparameters(obj, cost_forget_scale=1.)
        self.assertEqual(overridden.cost_forget_scale, 1.)
        self.assertEqual(obj.unlearn_config.cost_forget_scale, .37)

    def test_gradient_normalization_and_fisher_use_configured_values(self):
        obj = SimpleNamespace()
        hp = replace(DEFAULT_UNLEARN, grad_norm_beta=0., grad_norm_floor=1.,
                     fisher_beta=0., fisher_damping=.5, fisher_clip=.007,
                     eta_initial=.2, eta_beta=.6)
        _init_unlearn_state(obj, hp)
        keep, forget = _normalise_and_pcgrad(obj, [torch.tensor([2.])], [torch.tensor([4.])])
        torch.testing.assert_close(keep[0], torch.tensor([1.]))
        torch.testing.assert_close(forget[0], torch.tensor([1.]))
        scaled = _fisher_scale_forget(obj, [torch.nn.Parameter(torch.ones(1))], keep, forget)
        torch.testing.assert_close(scaled[0], torch.tensor([.007]))
        self.assertEqual(obj._eta_forget, .2)
        self.assertEqual(obj._eta_beta, .6)


class CriticTrainerConfigurationTests(unittest.TestCase):
    variants = ((BCQL, BCQLTrainerWithUnlearn), (CPQ, CPQTrainerWithUnlearn))

    def make(self, model_cls, trainer_cls):
        torch.manual_seed(43)
        model = model_cls(state_dim=3, action_dim=2, max_action=1.,
                          a_hidden_sizes=[8], c_hidden_sizes=[8], vae_hidden_sizes=8,
                          sample_action_num=2, num_q=1, num_qc=1,
                          episode_len=10, cost_limit=1., device="cpu")
        return trainer_cls(model, None, device="cpu")

    def batches(self):
        torch.manual_seed(47)
        def batch():
            return (torch.randn(8, 3), torch.randn(8, 3), torch.rand(8, 2) * 2 - 1,
                    torch.rand(8), torch.rand(8), torch.zeros(8))
        return batch(), batch()

    def test_attached_fixed_eta_and_cost_ablation_are_applied(self):
        for model_cls, trainer_cls in self.variants:
            with self.subTest(algo=model_cls.__name__):
                trainer = self.make(model_cls, trainer_cls)
                configure_unlearning(trainer, replace(DEFAULT_UNLEARN,
                    eta_mode="fixed", eta_fixed=.23, cost_forget_scale=0.))
                metrics = trainer.trajdeleter_unlearn_one_step(*self.batches())
                self.assertEqual(metrics["eta_effective"], .23)
                self.assertEqual(metrics["cost_forget_scale"], 0.)
                self.assertTrue(any(p.grad is not None and p.grad.norm() > 0
                                    for p in trainer.model.actor.parameters()))

    def test_explicit_step_eta_overrides_attached_setting(self):
        for model_cls, trainer_cls in self.variants:
            with self.subTest(algo=model_cls.__name__):
                trainer = self.make(model_cls, trainer_cls)
                configure_unlearning(trainer, replace(DEFAULT_UNLEARN, eta_mode="fixed", eta_fixed=.23))
                metrics = trainer.trajdeleter_unlearn_one_step(*self.batches(), eta_fixed=.81)
                self.assertEqual(metrics["eta_effective"], .81)
                self.assertEqual(trainer.unlearn_config.eta_fixed, .23)

    def test_configured_critic_forget_weight_changes_actual_gradient(self):
        for model_cls, trainer_cls in self.variants:
            with self.subTest(algo=model_cls.__name__):
                template = self.make(model_cls, trainer_cls)
                gradients = []
                for weight in (.1, .7):
                    trainer = copy.deepcopy(template)
                    configure_unlearning(trainer, replace(DEFAULT_UNLEARN,
                        alpha_q_keep=0., alpha_q_forget=weight, max_grad_norm=0.))
                    batches = self.batches()
                    torch.manual_seed(53)
                    trainer.trajdeleter_unlearn_one_step(*batches)
                    gradients.append(torch.cat([p.grad.flatten() for p in trainer.model.critic.parameters()]))
                self.assertGreater(gradients[0].norm().item(), 0.)
                torch.testing.assert_close(gradients[1], 7 * gradients[0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
