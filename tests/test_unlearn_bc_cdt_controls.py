"""Real BC/CDT checks for shared controls and padded sequence objectives."""
import copy
from dataclasses import replace
import unittest

import torch
import torch.nn.functional as F

from osrl.algorithms import BC, CDT
from utils.unlearn_config import UnlearnHyperparameters, configure_unlearning
from utils.unlearn_utils import BCTrainerWithUnlearn, CDTTrainerWithUnlearn


class BCAndCDTControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def trainer(self, algorithm, hp=None, stochastic=False, time_emb=True,
                cost_reverse=False):
        torch.manual_seed(17)
        if algorithm == "BC":
            model = BC(3, 2, 1., a_hidden_sizes=[8], device="cpu")
            trainer = BCTrainerWithUnlearn(model, None, actor_lr=.001, device="cpu")
        else:
            model = CDT(3, 2, 1., seq_len=4, episode_len=10, embedding_dim=8,
                        num_layers=1, num_heads=2, use_rew=True, use_cost=True,
                        stochastic=stochastic, target_entropy=-2., time_emb=time_emb)
            trainer = CDTTrainerWithUnlearn(
                model, None, learning_rate=.001, weight_decay=0., lr_warmup_steps=1,
                loss_cost_weight=.2, loss_state_weight=.4,
                cost_reverse=cost_reverse, device="cpu")
        configure_unlearning(trainer, hp or UnlearnHyperparameters(
            eta_mode="fixed", eta_fixed=.4, max_grad_norm=0.))
        return trainer

    def batches(self, algorithm):
        torch.manual_seed(23)
        if algorithm == "BC":
            def batch():
                return torch.randn(4, 3), torch.rand(4, 2) * 1.6 - .8
        else:
            def batch():
                return (torch.randn(2, 4, 3), torch.rand(2, 4, 2) * 1.6 - .8,
                        torch.rand(2, 4), torch.rand(2, 4),
                        torch.arange(4).repeat(2, 1),
                        torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
                        torch.tensor([1., 1.]),
                        torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.]]))
        return batch(), batch()

    def step(self, algorithm, hp=None, batches=None, **overrides):
        trainer = self.trainer(algorithm, hp)
        torch.manual_seed(31)
        metrics = trainer.trajdeleter_unlearn_one_step(
            *(batches or self.batches(algorithm)), **overrides)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in trainer.model.state_dict().values()))
        return trainer, metrics

    def test_configured_eta_and_direct_override_are_applied(self):
        hp = UnlearnHyperparameters(eta_mode="fixed", eta_fixed=.35)
        for algorithm in ("BC", "CDT"):
            with self.subTest(algorithm=algorithm):
                low, metrics = self.step(algorithm, hp)
                high, overridden = self.step(algorithm, hp, eta_fixed=.8)
                self.assertEqual(metrics["eta_effective"], .35)
                self.assertEqual(overridden["eta_effective"], .8)
                low_grad = torch.cat([p.grad.flatten() for p in low.model.parameters()
                                      if p.grad is not None])
                high_grad = torch.cat([p.grad.flatten() for p in high.model.parameters()
                                       if p.grad is not None])
                self.assertGreater(low_grad.norm().item(), 0.)
                self.assertFalse(torch.equal(low_grad, high_grad))

    def test_zero_eta_removes_actual_forget_update(self):
        for algorithm in ("BC", "CDT"):
            with self.subTest(algorithm=algorithm):
                hp = UnlearnHyperparameters(alpha_pg_keep=0., eta_mode="fixed",
                                            eta_fixed=0., max_grad_norm=0.)
                before = self.trainer(algorithm, hp).model
                off, _ = self.step(algorithm, hp)
                on, _ = self.step(algorithm, replace(hp, eta_fixed=1.))
                self.assertTrue(all(torch.equal(a, b) for a, b in
                                    zip(before.parameters(), off.model.parameters())))
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                                    zip(before.parameters(), on.model.parameters())))

    def test_reward_only_matches_explicit_cost_ablation(self):
        for algorithm in ("BC", "CDT"):
            with self.subTest(algorithm=algorithm):
                off, _ = self.step(algorithm, cost_forget_scale=0.)
                reward_only, metrics = self.step(algorithm, unlearn_method="reward-only")
                self.assertEqual(metrics["cost_forget_scale"], 0.)
                self.assertTrue(all(torch.equal(value, reward_only.model.state_dict()[key])
                                    for key, value in off.model.state_dict().items()))

    def test_behavior_retention_weight_controls_actual_cloning_gradient(self):
        hp = UnlearnHyperparameters(eta_mode="fixed", eta_fixed=0., max_grad_norm=0.,
                                    keep_bc_weight=0.)
        for algorithm in ("BC", "CDT"):
            with self.subTest(algorithm=algorithm):
                off, off_metrics = self.step(algorithm, hp)
                on, on_metrics = self.step(algorithm, replace(hp, keep_bc_weight=2.))
                off_actor = off.model.actor if algorithm == "BC" else off.model.action_head
                on_actor = on.model.actor if algorithm == "BC" else on.model.action_head
                off_grad = torch.cat([p.grad.flatten() for p in off_actor.parameters()])
                on_grad = torch.cat([p.grad.flatten() for p in on_actor.parameters()])
                self.assertEqual(off_grad.norm().item(), 0.)
                self.assertGreater(on_grad.norm().item(), 0.)
                if algorithm == "CDT":
                    self.assertAlmostEqual(off_metrics["loss_keep"],
                        .2 * off_metrics["loss_cost_keep"]
                        + .4 * off_metrics["loss_state_keep"], places=6)
                    self.assertAlmostEqual(on_metrics["loss_keep"],
                        2. * on_metrics["loss_act_keep"]
                        + .2 * on_metrics["loss_cost_keep"]
                        + .4 * on_metrics["loss_state_keep"], places=6)
                    self.assertGreater(off.model.cost_pred_head.weight.grad.norm().item(), 0.)

    def test_obsolete_lambda_cost_is_rejected(self):
        for algorithm in ("BC", "CDT"):
            with self.subTest(algorithm=algorithm):
                trainer = self.trainer(algorithm)
                with self.assertRaisesRegex(TypeError, "cost_forget_scale"):
                    trainer.trajdeleter_unlearn_one_step(
                        *self.batches(algorithm), lambda_cost=.5)

    def test_cdt_cost_ablation_controls_cost_head_gradient(self):
        hp = UnlearnHyperparameters(alpha_pg_keep=0., eta_mode="fixed", eta_fixed=1.,
                                    max_grad_norm=0.)
        off, _ = self.step("CDT", hp, cost_forget_scale=0.)
        on, _ = self.step("CDT", hp, cost_forget_scale=1.)
        self.assertEqual(off.model.cost_pred_head.weight.grad.norm().item(), 0.)
        self.assertGreater(on.model.cost_pred_head.weight.grad.norm().item(), 0.)
        self.assertFalse(torch.equal(off.model.cost_pred_head.weight,
                                     on.model.cost_pred_head.weight))

    def test_cdt_cost_reversal_suppresses_the_correct_unsafe_class(self):
        hp = UnlearnHyperparameters(alpha_pg_keep=0., eta_mode="fixed", eta_fixed=1.,
                                    max_grad_norm=0.)
        for reversed_costs in (False, True):
            with self.subTest(cost_reverse=reversed_costs):
                trainer = self.trainer("CDT", hp, cost_reverse=reversed_costs)
                batches = tuple(tuple(value[:, :1] if value.ndim > 1 else value
                                      for value in batch) for batch in self.batches("CDT"))
                forgotten, kept = batches
                states, actions, rtg, ctg, timesteps, mask, episode_cost, costs = forgotten
                with torch.no_grad():
                    prediction, _, _ = trainer.model(
                        states, actions, rtg, ctg, timesteps, ~mask.bool(), episode_cost)
                    # Match the one-token action prediction to isolate the
                    # actual cost-classification gradient from anti-imitation.
                    actions = prediction.detach().clone()
                    _, before, _ = trainer.model(
                        states, actions, rtg, ctg, timesteps, ~mask.bool(), episode_cost)
                forgotten = (states, actions, rtg, ctg, timesteps, mask, episode_cost, costs)
                metrics = trainer.trajdeleter_unlearn_one_step(forgotten, kept)
                unsafe_class = 0 if reversed_costs else 1
                self.assertEqual(metrics["unsafe_cost_class"], unsafe_class)
                self.assertAlmostEqual(metrics["unsafe_cost_conf_f"],
                                       before[..., unsafe_class].exp().mean().item(), places=6)
                self.assertAlmostEqual(metrics["cost1_conf_f"],
                                       before[..., 1].exp().mean().item(), places=6)
                self.assertLess(trainer.model.cost_pred_head.bias.grad[unsafe_class].item(), 0.)
                with torch.no_grad():
                    _, after, _ = trainer.model(
                        states, actions, rtg, ctg, timesteps, ~mask.bool(), episode_cost)
                self.assertGreater(after[..., unsafe_class].exp().mean().item(),
                                   before[..., unsafe_class].exp().mean().item())

    def test_bc_distance_settings_and_incomplete_target_increase_eta(self):
        hp = UnlearnHyperparameters(eta_beta=0., bc_forget_temperature=.2,
                                    bc_forget_distance_target=0.)
        _, reached = self.step("BC", hp)
        _, unmet = self.step("BC", replace(hp, bc_forget_distance_target=2.))
        self.assertGreater(unmet["eta_effective"], reached["eta_effective"])
        trainer = self.trainer("BC", hp)
        forgotten, _ = self.batches("BC")
        with torch.no_grad():
            distance = ((trainer.model.actor(forgotten[0]) - forgotten[1]) ** 2).mean(-1)
            expected = F.softplus((hp.bc_forget_distance_target - distance) /
                                  hp.bc_forget_temperature).mean().item()
        self.assertAlmostEqual(reached["loss_forget_cost"], expected, places=6)

    def test_cdt_unmet_confidence_target_increases_eta(self):
        hp = UnlearnHyperparameters(eta_beta=0., cdt_forget_cost_confidence_target=0.)
        _, reached = self.step("CDT", hp)
        _, unmet = self.step("CDT", replace(hp, cdt_forget_cost_confidence_target=1.))
        self.assertAlmostEqual(reached["cost1_conf_f"], unmet["cost1_conf_f"])
        self.assertGreater(unmet["eta_effective"], reached["eta_effective"])

    def test_cdt_keep_losses_average_features_and_valid_state_pairs(self):
        trainer = self.trainer("CDT")
        forgotten, kept = self.batches("CDT")
        states, actions, rtg, ctg, timesteps, mask, episode_cost, costs = kept
        with torch.no_grad():
            act_pred, cost_pred, state_pred = trainer.model(
                states, actions, rtg, ctg, timesteps, ~mask.bool(), episode_cost)
            valid = mask.bool()
            pairs = valid[:, :-1] & valid[:, 1:]
            action_loss = F.mse_loss(act_pred[valid], actions[valid]).item()
            cost_loss = F.nll_loss(cost_pred[valid], costs[valid].long()).item()
            state_loss = F.mse_loss(state_pred[:, :-1][pairs],
                                    states[:, 1:][pairs]).item()
        metrics = trainer.trajdeleter_unlearn_one_step(forgotten, kept)
        self.assertAlmostEqual(metrics["loss_act_keep"], action_loss, places=6)
        self.assertAlmostEqual(metrics["loss_cost_keep"], cost_loss, places=6)
        self.assertAlmostEqual(metrics["loss_state_keep"], state_loss, places=6)
        self.assertAlmostEqual(metrics["loss_keep"],
                               action_loss + .2 * cost_loss + .4 * state_loss, places=6)

    def test_cdt_padded_values_cannot_change_losses_or_parameter_updates(self):
        original = self.batches("CDT")
        changed = copy.deepcopy(original)
        for batch in changed:
            invalid = ~batch[5].bool()
            for index in (0, 1, 2, 3):
                batch[index][invalid] = 123.
            batch[4][invalid] = 999
        before, metrics = self.step("CDT", batches=original)
        after, padded_metrics = self.step("CDT", batches=changed)
        for key in ("loss_keep", "loss_state_keep", "loss_forget_q", "loss_forget_cost"):
            self.assertAlmostEqual(metrics[key], padded_metrics[key], places=6)
        for a, b in zip(before.model.parameters(), after.model.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_real_cdt_stochastic_and_deterministic_native_continuation(self):
        for stochastic in (False, True):
            with self.subTest(stochastic=stochastic):
                trainer = self.trainer("CDT", stochastic=stochastic)
                forgotten, kept = self.batches("CDT")
                metrics = trainer.trajdeleter_unlearn_one_step(forgotten, kept)
                trainer.train_one_step(*kept)
                self.assertEqual(metrics["eta_effective"], .4)
                self.assertEqual(trainer.scheduler.last_epoch, 2)
                self.assertTrue(all(torch.isfinite(p).all() for p in trainer.model.parameters()))

    def test_cdt_without_time_embeddings_and_single_valid_token(self):
        trainer = self.trainer("CDT", time_emb=False)
        original = self.batches("CDT")
        batches = tuple(tuple(value[:, :1] if value.ndim > 1 else value
                              for value in batch) for batch in original)
        metrics = trainer.trajdeleter_unlearn_one_step(*batches)
        self.assertEqual(metrics["loss_state_keep"], 0.)
        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.model.parameters()))


if __name__ == "__main__":
    unittest.main()
