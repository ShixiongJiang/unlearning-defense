"""CPU integration checks for effective runner settings and native data semantics."""
import json
import os
from dataclasses import asdict, fields
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import unlearn_policy as runner
from utils.unlearn_config import configure_unlearning


def transitions():
    return dict(observations=np.arange(8, dtype=np.float32).reshape(4, 2),
                next_observations=np.arange(8, dtype=np.float32).reshape(4, 2) + 1,
                actions=np.arange(4, dtype=np.float32).reshape(4, 1) / 4,
                rewards=np.arange(1, 5, dtype=np.float32),
                costs=np.array([0, 1, 1, 0], dtype=np.float32),
                terminals=np.zeros(4, dtype=bool),
                timeouts=np.array([0, 1, 0, 1], dtype=bool))


class IdentityEnvironment:
    observation_space = SimpleNamespace(shape=(2,))
    action_space = SimpleNamespace(shape=(1,), high=np.ones(1), low=-np.ones(1))

    def pre_process_data(self, data, *args, **kwargs):
        return {key: np.asarray(value).copy() for key, value in data.items()}

    def set_target_cost(self, value):
        self.cost_limit = value


def settings(**changes):
    values = dict(actor_lr=.0021, critic_lr=.0032, vae_lr=.0043, alpha_lr=.0054,
                  scalar_lr=.0065, learning_rate=.0076, reward_scale=.2, cost_scale=3.,
                  batch_size=4, cost_limit=2., episode_len=7, device="cpu", num_workers=0,
                  density=1., task="OfflineSynthetic-v0", outliers_percent=None,
                  noise_scale=None, inpaint_ranges=None, epsilon=None, gamma=.9,
                  bc_mode="all", betas=(.7, .83), weight_decay=.06,
                  lr_warmup_steps=4, clip_grad=.19, loss_cost_weight=.02,
                  loss_state_weight=0., cost_reverse=False, no_entropy=False)
    values.update(changes)
    return SimpleNamespace(**values)


def model(algo):
    common = dict(state_dim=2, action_dim=1, max_action=1., episode_len=7, device="cpu")
    if algo == "BC":
        return runner.BC(**common, a_hidden_sizes=[8])
    if algo == "CDT":
        common.pop("device")
        return runner.CDT(**common, seq_len=2, embedding_dim=8, num_layers=1,
                          num_heads=1, stochastic=True)
    if algo == "COptiDICE":
        return runner.COptiDICE(**common, f_type="chi2", init_state_propotion=.5,
            observations_std=np.ones((1, 2)), actions_std=np.ones((1, 1)),
            a_hidden_sizes=[8], c_hidden_sizes=[8], num_nu=1, num_chi=1, cost_limit=2.)
    cls = {"BCQL": runner.BCQL, "CPQ": runner.CPQ, "BEARL": runner.BEARL}[algo]
    return cls(**common, a_hidden_sizes=[8], c_hidden_sizes=[8], vae_hidden_sizes=8,
               sample_action_num=2, num_q=1, num_qc=1, cost_limit=2.)


class LoaderConsistencyTests(unittest.TestCase):
    def test_timeout_boundaries_remain_nonabsorbing_and_do_not_mutate_input(self):
        data = transitions()
        normalized = runner._ensure_terminals(data)
        np.testing.assert_array_equal(normalized["terminals"], np.zeros(4))
        np.testing.assert_array_equal(normalized["timeouts"], data["timeouts"])
        unflagged = transitions()
        unflagged["timeouts"][:] = False
        result = runner._ensure_terminals(unflagged)
        self.assertTrue(result["timeouts"][-1])
        self.assertFalse(unflagged["timeouts"].any())

    def test_cdt_windows_stop_at_timeouts_with_scaled_returns_and_cost_reversal(self):
        data = runner._CDTSequenceDataset(transitions(), 3, .2, 3.)
        first, second, third = data[0], data[1], data[2]
        np.testing.assert_array_equal(first[5], [1, 1, 0])
        np.testing.assert_allclose(first[2], [.6, .4, 0])
        np.testing.assert_allclose(first[3], [3, 3, 0])
        np.testing.assert_array_equal(second[5], [1, 0, 0])
        self.assertEqual(third[4][0], 0)
        reversed_data = runner._CDTSequenceDataset(transitions(), 3, .2, 3., True)
        np.testing.assert_array_equal(reversed_data[0][7], [1, 0, 0])
        np.testing.assert_allclose(reversed_data[0][3], [3, 0, 0])

    def test_final_partial_prefix_survives_preprocessing_after_an_earlier_episode(self):
        class BoundaryPreprocessor(IdentityEnvironment):
            def pre_process_data(self, data, *args, **kwargs):
                stop = np.flatnonzero(data["terminals"] | data["timeouts"])[-1] + 1
                return {key: values[:stop].copy() for key, values in data.items()}

        data = transitions()
        data["timeouts"][-1] = False
        loader = runner._make_dataloader(data, BoundaryPreprocessor(), settings(), 4)
        self.assertEqual(loader.dataset.dataset_size, 4)
        np.testing.assert_array_equal(loader.dataset.dataset["terminals"], [0, 0, 0, 0])
        np.testing.assert_array_equal(loader.dataset.dataset["timeouts"], [0, 1, 0, 1])
        self.assertFalse(data["timeouts"][-1])

    def test_all_six_reject_missing_internal_boundaries_and_accept_verified_prefix_flags(self):
        def recording(size):
            return dict(observations=np.zeros((size, 2), np.float32),
                        next_observations=np.zeros((size, 2), np.float32),
                        actions=np.zeros((size, 1), np.float32),
                        rewards=np.ones(size, np.float32), costs=np.zeros(size, np.float32),
                        terminals=np.zeros(size, bool), timeouts=np.zeros(size, bool))

        giant = recording(2000)
        prepared = recording(1000)
        prepared["timeouts"][[499, 999]] = True
        args = settings(episode_len=1000, seq_len=2)
        for algo in ("BCQL", "CPQ", "BEARL", "BC", "CDT", "COptiDICE"):
            with self.subTest(algo=algo):
                with self.assertRaisesRegex(ValueError, "prepare verified recording metadata"):
                    runner._build_loaders(giant, prepared, IdentityEnvironment(), args,
                                          runner.EvalConfig(), algo)
                forget, retain = runner._build_loaders(
                    prepared, prepared, IdentityEnvironment(), args, runner.EvalConfig(), algo)
                self.assertEqual(forget.batch_size, 4)
                self.assertEqual(retain.batch_size, 4)
        runner._validate_episode_boundaries(recording(300), 1000)

    def test_coptidice_native_sampling_scales_initial_flags_and_statistics(self):
        loader = runner._make_dataloader_coptidice(transitions(), IdentityEnvironment(), settings(), 4)
        with patch("numpy.random.choice", side_effect=[0, 1, 2, 3]):
            batch = next(iter(loader))
        np.testing.assert_allclose(batch[3], [.2, .4, .6, .8])
        np.testing.assert_allclose(batch[4], [0, 3, 3, 0])
        np.testing.assert_array_equal(batch[5], [0, 1, 0, 1])
        np.testing.assert_array_equal(batch[6], [1, 0, 1, 0])
        info = runner._coptidice_statistics({}, loader)
        self.assertEqual(info["init_state_propotion"], .5)
        np.testing.assert_allclose(info["observations_std"], transitions()["observations"].std(0, keepdims=True))
        self.assertIn("processed retained", info["provenance"])
        saved = dict(info, init_state_propotion=.25, observations_std=[[7, 8]])
        restored = runner._coptidice_statistics({"coptidice_dataset_statistics": saved}, loader)
        self.assertEqual(restored["init_state_propotion"], .5)
        self.assertEqual(restored["source_init_state_propotion"], .25)
        self.assertEqual(restored["observations_std"], [[7, 8]])

    def test_multitask_bc_loader_uses_native_episode_cost_feature(self):
        args = settings(unlearn_algo="BC", bc_mode="multi-task")
        loader = runner._make_dataloader(transitions(), IdentityEnvironment(), args, 4)
        with patch("numpy.random.choice", side_effect=[0, 1, 2, 3]):
            batch = next(iter(loader))
        self.assertEqual(tuple(batch[0].shape), (4, 3))
        np.testing.assert_allclose(batch[0][:, -1], [.9, .9, 1., 1.])


class RuntimeSettingsTests(unittest.TestCase):
    algorithms = ("BCQL", "CPQ", "BEARL", "BC", "CDT", "COptiDICE")

    def test_all_six_restore_actual_optimizer_rates_and_horizons(self):
        for algo in self.algorithms:
            with self.subTest(algo=algo):
                args = settings()
                trainer = runner._build_trainer(model(algo), None, algo, args, runner.EvalConfig())
                hp = runner._shared_hyperparameters(runner.EvalConfig(eta_mode="fixed", eta_fixed=.23))
                configure_unlearning(trainer, hp)
                actual = runner._runtime_parameters(trainer, algo, args)
                self.assertEqual(actual["episode_len"], 7)
                self.assertEqual(actual["batch_size"], 4)
                self.assertEqual(actual["shared_hyperparameters"]["eta_fixed"], .23)
                if algo == "CDT":
                    self.assertEqual(trainer.model.target_entropy, -1)
                    self.assertAlmostEqual(trainer.optim.param_groups[0]["lr"], args.learning_rate / 4)
                    self.assertEqual(trainer.optim.param_groups[0]["betas"], args.betas)
                    data = runner._CDTSequenceDataset(transitions(), 2, .2, 3.)
                    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
                    runner._fine_tune_step(trainer, algo, batch, "cpu")
                    self.assertAlmostEqual(trainer.optim.param_groups[0]["lr"], args.learning_rate / 2)
                    self.assertEqual(trainer.clip_grad, .19)
                else:
                    self.assertEqual(trainer.model.actor_optim.param_groups[0]["lr"], args.actor_lr)
                    if hasattr(trainer.model, "critic_optim"):
                        self.assertEqual(trainer.model.critic_optim.param_groups[0]["lr"], args.critic_lr)
                    if hasattr(trainer.model, "vae_optim"):
                        self.assertEqual(trainer.model.vae_optim.param_groups[0]["lr"], args.vae_lr)
                    if algo in ("CPQ", "BEARL"):
                        self.assertEqual(trainer.model.alpha_lr, args.alpha_lr)
                    if algo == "COptiDICE":
                        self.assertEqual(trainer.model.tau_optim.param_groups[0]["lr"], args.scalar_lr)

    def test_overrides_are_applied_or_rejected_and_old_cpq_alpha_uses_native_fallback(self):
        for algo in self.algorithms:
            with self.subTest(algo=algo):
                key = "learning_rate" if algo == "CDT" else "actor_lr"
                config = runner.EvalConfig(**{key: .0123})
                result = runner._native_trainer_kwargs(algo, settings(), config)
                self.assertEqual(result[key], .0123)
                unsupported = "scalar_lr" if algo != "COptiDICE" else "vae_lr"
                with self.assertRaisesRegex(ValueError, "not used"):
                    runner._native_trainer_kwargs(algo, settings(), runner.EvalConfig(**{unsupported: .1}))
        args = settings()
        del args.alpha_lr
        self.assertEqual(runner._native_trainer_kwargs("CPQ", args)["alpha_lr"], 1e-4)

    def test_all_native_evaluations_use_saved_horizon_and_raw_reward_cost_units(self):
        class ConstantSignalEnvironment(IdentityEnvironment):
            def reset(self):
                self.steps = 0
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                # The runner's reward wrapper already scales rewards; costs
                # in info remain raw and must be reported as raw episode sums.
                return np.zeros(2, dtype=np.float32), .2, False, False, {"cost": 2.}

        for algo in self.algorithms:
            with self.subTest(algo=algo):
                env = ConstantSignalEnvironment()
                args = settings()
                trainer = runner._build_trainer(model(algo), env, algo, args)
                reward, cost, length = runner._evaluate(
                    trainer, algo, runner.EvalConfig(eval_episodes=1), args)
                self.assertAlmostEqual(reward, 7.)
                self.assertAlmostEqual(cost, 14.)
                self.assertEqual(length, 7)
                self.assertEqual(env.steps, 7)

    def test_shared_cli_definition_covers_every_validated_field(self):
        config = runner.EvalConfig(eta_mode="fixed", eta_fixed=.23, fisher_clip=.17,
                                   reward_forget_quantile=.42, coptidice_forget_weight_target=1.7)
        hp = runner._shared_hyperparameters(config)
        self.assertEqual(asdict(hp), {field.name: getattr(config, field.name)
                                     for field in fields(runner.DEFAULT_UNLEARN)})
        with self.assertRaises(ValueError):
            runner._shared_hyperparameters(runner.EvalConfig(eta_fixed=float("nan")))
        self.assertEqual(runner._shared_hyperparameters(runner.EvalConfig(
            unlearn_method="reward-only", cost_forget_scale=.37)).cost_forget_scale, 0.)


class MemoryLogger:
    def __init__(self, config, project, group, name, logdir):
        self.name = name

    def save_config(self, *args, **kwargs):
        pass

    def setup_checkpoint_fn(self, function):
        self.checkpoint = function

    def save_checkpoint(self, *args, **kwargs):
        self.checkpoint()

    def store(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


class MainRunnerTests(unittest.TestCase):
    def test_two_real_updates_each_algorithm_and_effective_protocol(self):
        """Exercise CLI config -> model/loaders -> unlearning -> native convergence."""
        for algo in RuntimeSettingsTests.algorithms:
            with self.subTest(algo=algo), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                source.mkdir()
                (root / "save_data").mkdir()
                for split in ("poisoned_only", "clean_only"):
                    np.savez(root / "save_data" / f"Synthetic_poisoned_traj_inject.{split}.npz", **transitions())
                cfg = vars(settings()).copy()
                cfg.update(name=algo, logdir=str(root / "old_logs"), seed=0, threads=1,
                           project="test", group="test", verbose=False,
                           a_hidden_sizes=[8], c_hidden_sizes=[8], vae_hidden_sizes=8,
                           sample_action_num=2, num_q=1, num_qc=1, tau=.005, beta=.5,
                           PID=[.1, .003, .001], lmbda=.75, phi=.05, qc_scalar=1.5,
                           seq_len=2, embedding_dim=8, num_layers=1, num_heads=1,
                           attention_dropout=0., residual_dropout=0., embedding_dropout=0.,
                           stochastic=(algo == "CDT"), num_nu=1, num_chi=1)
                if algo == "BC":
                    cfg.pop("reward_scale")
                    cfg.pop("cost_scale")
                checkpoint = {"model_state": model(algo).state_dict()}
                config = runner.EvalConfig(algo=algo, unlearn_path=str(source),
                    output_dir=str(root / "out"), total_updates=2, unlearn_steps=1,
                    eval_every=1, eval_episodes=1, milestone_updates="2",
                    eta_mode="fixed", eta_fixed=.23, cost_forget_scale=.37,
                    fisher_clip=.17, fit_behavior_on_retain=True)
                previous_cwd = os.getcwd()
                try:
                    os.chdir(root)
                    with patch.object(runner, "load_config_and_model", return_value=(cfg, checkpoint)), \
                         patch.object(runner.gym, "make", return_value=IdentityEnvironment()), \
                         patch.object(runner, "wrap_env", side_effect=lambda env, **kw: env), \
                         patch.object(runner, "OfflineEnvWrapper", side_effect=lambda env: env), \
                         patch.object(runner, "WandbLogger", MemoryLogger), \
                         patch.object(runner, "_evaluate", return_value=(1., .5, 7.)):
                        runner.main.__wrapped__(config)
                finally:
                    os.chdir(previous_cwd)
                output = json.loads((root / "out/protocol.json").read_text())
                effective = output["config"]["unlearning"]
                self.assertEqual(effective["effective_hyperparameters"]["eta_fixed"], .23)
                self.assertEqual(effective["effective_runtime"]["episode_len"], 7)
                self.assertEqual(effective["evaluation_protocol"]["mode"], "native trainer evaluation")
                summary = json.loads((root / "out/summary.json").read_text())
                self.assertEqual(summary["global_step"], 2)
                records = [json.loads(line) for line in (root / "out/metrics.jsonl").read_text().splitlines()]
                first_update = next(row for row in records if row["type"] == "train")
                self.assertEqual(first_update["eta_effective"], .23)
                self.assertEqual(first_update["cost_forget_scale"], .37)
                self.assertTrue((root / "out/checkpoint/step_000002.pt").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
