"""Regression checks for CPQ scalar checkpoint persistence and legacy loading."""
from copy import deepcopy
import io
import math
import unittest
import warnings

import torch

from osrl.algorithms.cpq import CPQ


def make_model():
    return CPQ(3, 2, 1.0, a_hidden_sizes=[8], c_hidden_sizes=[8],
               vae_hidden_sizes=16, sample_action_num=2)


def legacy_state(model):
    state = deepcopy(model.state_dict())
    del state["log_alpha"]
    state._metadata[""]["version"] = 1
    return state


class CPQCheckpointTests(unittest.TestCase):
    def test_learned_alpha_survives_serialization(self):
        source = make_model()
        source.log_alpha.fill_(-3.25)
        stream = io.BytesIO()
        torch.save({"model_state": source.state_dict()}, stream)
        stream.seek(0)
        restored = make_model()
        restored.load_state_dict(torch.load(stream, weights_only=True)["model_state"])
        self.assertEqual(restored.log_alpha.item(), -3.25)
        self.assertIn("log_alpha", dict(restored.named_buffers()))
        self.assertNotIn("log_alpha", dict(restored.named_parameters()))

    def test_alpha_follows_module_dtype_conversion(self):
        model = make_model().to(dtype=torch.float64)
        self.assertEqual(model.log_alpha.dtype, torch.float64)

    def test_manual_alpha_update_remains_checkpointed(self):
        torch.manual_seed(0)
        model = make_model()
        model.setup_optimizers(1e-3, 1e-3, 1e-2, 1e-3)
        obs, action = torch.randn(4, 3), torch.randn(4, 2)
        model.cost_critic_loss(obs, obs, action, torch.ones(4), torch.zeros(4))
        self.assertNotEqual(model.log_alpha.item(), 0.0)
        self.assertFalse(model.log_alpha.requires_grad)
        restored = make_model()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.log_alpha, model.log_alpha)

    def test_legacy_load_warns_and_uses_explicit_default(self):
        source = make_model()
        state = legacy_state(source)
        restored = make_model()
        restored.log_alpha.fill_(4.0)
        with self.assertWarnsRegex(RuntimeWarning, "Legacy CPQ.*log_alpha=0"):
            restored.load_state_dict(state)
        self.assertEqual(restored.log_alpha.item(), 0.0)
        self.assertNotIn("log_alpha", state)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_legacy_exception_does_not_hide_network_mismatch(self):
        state = legacy_state(make_model())
        del state[next(key for key in state if key.startswith("actor."))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                make_model().load_state_dict(state)

    def test_current_checkpoint_missing_alpha_is_rejected(self):
        state = make_model().state_dict()
        del state["log_alpha"]
        with self.assertRaisesRegex(RuntimeError, "log_alpha"):
            make_model().load_state_dict(state)

    def test_legacy_upgrade_works_inside_parent_module(self):
        source = torch.nn.Sequential(make_model())
        state = deepcopy(source.state_dict())
        del state["0.log_alpha"]
        state._metadata["0"]["version"] = 1
        with self.assertWarnsRegex(RuntimeWarning, "Legacy CPQ"):
            restored = torch.nn.Sequential(make_model())
            restored.load_state_dict(state)
        self.assertEqual(restored[0].log_alpha.item(), 0.0)


class CPQCostRegularizerTests(unittest.TestCase):
    def run_alpha_pair(self):
        torch.manual_seed(4)
        base = make_model()
        # A constant, positive Qc produces a small downward Bellman gradient.
        # A sufficiently strong OOD penalty must reverse that gradient.
        with torch.no_grad():
            for parameter in base.cost_critic.parameters():
                parameter.zero_()
            base.cost_critic.q_nets[0][-2].bias.fill_(1.0)
            base.cost_critic_old.load_state_dict(base.cost_critic.state_dict())
        models = [deepcopy(base), deepcopy(base)]
        obs, action = torch.randn(8, 3), torch.randn(8, 2)
        for model, alpha in zip(models, (0.01, 10.0)):
            model.log_alpha.fill_(math.log(alpha))
            model.setup_optimizers(1e-3, 1e-3, 1e-4, 1e-3)
            torch.manual_seed(5)
            model.cost_critic_loss(
                obs, obs, action, torch.zeros(8), torch.zeros(8))
            self.assertTrue(all(p.grad is None for p in model.actor.parameters()))
            self.assertTrue(all(p.grad is None for p in model.vae.parameters()))
            self.assertTrue(all(p.grad is None for p in model.cost_critic_old.parameters()))
        return [model.cost_critic.q_nets[0][-2].bias.item() for model in models]

    def test_corrected_ood_penalty_changes_cost_critic_update(self):
        low_alpha_bias, high_alpha_bias = self.run_alpha_pair()
        self.assertLess(low_alpha_bias, 1.0)
        self.assertGreater(high_alpha_bias, 1.0)


if __name__ == "__main__":
    unittest.main()
