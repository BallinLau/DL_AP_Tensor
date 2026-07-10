import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data import TensorTable  # noqa: E402
from experiments.run_utils import build_hyperparams  # noqa: E402
import training.episode as episode_module  # noqa: E402
from training.episode import Episode, NumericalStageFailure, SDFTrainingPhase  # noqa: E402


class _DummyMonitor:
    def log_memory(self, context):
        return {"enabled": False, "context": context}

    def get_summary(self):
        return {"enabled": False}


class _FakeSdfModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        del x_prev, return_physical
        hatcf_curr = hatcf_prev.unsqueeze(1) + 0.5 * x_curr
        lnkf_curr = lnkf_prev.unsqueeze(1) + 0.25 * x_curr
        batch = x_curr.shape[0]
        n_children = x_curr.shape[1]
        w_prev = torch.ones(batch, 1)
        w_curr = torch.ones(batch, n_children, 1)
        m = torch.ones(batch, n_children, 1)
        return w_prev, w_curr, m, hatcf_curr, lnkf_curr


class TreatmentBFlowTest(unittest.TestCase):
    def test_metric_aggregation_separates_numeric_and_metadata(self):
        records = [
            {"sdf": 2.0, "sdf_main_loss": 1.0, "sdf_training_phase": "episode0_bootstrap"},
            {"sdf": 4.0, "sdf_main_loss": 3.0, "sdf_training_phase": "episode0_bootstrap"},
        ]

        numeric, metadata = Episode._aggregate_metric_records(records)

        self.assertAlmostEqual(numeric["sdf"], 3.0)
        self.assertAlmostEqual(numeric["sdf_main_loss"], 2.0)
        self.assertEqual(metadata["sdf_training_phase"], "episode0_bootstrap")

    def test_metric_aggregation_rejects_changed_phase(self):
        records = [
            {"sdf_training_phase": "episode0_bootstrap"},
            {"sdf_training_phase": "sdf_true_only"},
        ]

        with self.assertRaisesRegex(RuntimeError, "changed within one epoch"):
            Episode._aggregate_metric_records(records)

    def test_run_batches_returns_metadata_separate_from_final_losses(self):
        episode = Episode.__new__(Episode)
        episode.models = {"sdf_fc1": object()}
        episode.hyperparams = SimpleNamespace(
            q_pretrain_epochs=0,
            q_warmstart_epochs=0,
            bp_refine_steps_per_epoch=0,
        )
        episode.step_count = 0
        episode._q_only_stage = False
        episode._bp_only_stage = False
        episode._configure_sdf_lr_for_phase = MethodType(lambda self: None, episode)
        episode._prepare_sdf_shock_bank_for_epoch = MethodType(lambda self, batches, epoch, train_modules: None, episode)
        episode._maybe_update_firm_target_epoch = MethodType(lambda self, train_modules: None, episode)

        def fake_train_step(self, batch, train_modules, policy_loss_terms=None):
            self.step_count += 1
            return {
                "total": 2.0,
                "sdf": 2.0,
                "sdf_training_phase": "episode0_bootstrap",
            }

        episode.train_step = MethodType(fake_train_step, episode)

        summary = episode._run_batches(
            batches=[{"dummy": torch.tensor(1.0)}],
            n_epochs=1,
            log_interval=100,
            train_modules=["sdf_fc1"],
        )

        self.assertEqual(summary["final_losses"]["sdf"], 2.0)
        self.assertNotIn("sdf_training_phase", summary["final_losses"])
        self.assertEqual(summary["metadata"]["sdf_training_phase"], "episode0_bootstrap")

    def _make_episode(self, resimulate_after_pv, post_refresh_pass=True):
        episode = Episode.__new__(Episode)
        episode.models = {"policy_value": object(), "sdf_fc1": object()}
        episode.optimizers = {}
        episode.config = SimpleNamespace(DEVICE=torch.device("cpu"))
        episode.hyperparams = SimpleNamespace(use_tensor_pipeline=True, simulate_horizon=2)
        episode.device = torch.device("cpu")
        episode.episode_id = 1
        episode.gpu_monitor = _DummyMonitor()
        episode.df = None
        episode.df_macro = None
        episode.df_sdf = None
        episode.tensor_firm = None
        episode.tensor_macro = None
        episode.tensor_sdf = None
        episode.step_count = 0
        episode.loss_history = {}
        episode.add_FC1loss = False
        episode.train_mode = "2time"
        episode._fc1_teacher_forcing_stage = False

        calls = []
        simulate_count = {"n": 0}

        def fake_simulate_tensor(self, n_paths, group_size, n_branches, horizon, simulate_kwargs, export_df=False):
            simulate_count["n"] += 1
            marker = float(simulate_count["n"])
            calls.append(f"simulate{simulate_count['n']}")
            self.tensor_macro = TensorTable(
                data=torch.tensor(
                    [[0.0, 0.0, -1.0, 10.0, 2.0, 1.0, marker, 3.0, 1.0, 0.1, 0.5, 4.0]],
                    dtype=torch.float32,
                ),
                columns=["path", "t", "branch", "K", "C", "LnK", "Hatc", "n_firms", "M", "x", "hatcf", "lnkf"],
            )
            self.tensor_firm = TensorTable(
                data=torch.tensor(
                    [[0.0, 0.0, -1.0, 7.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1, marker, 4.0 + marker, 1.0, marker, 0.9, 0.8, 0.7, 0.4, 0.2, 1.1, 0.2, 0.3, 0.25, 1.0, 0.1, 0.0, 0.8]],
                    dtype=torch.float32,
                ),
                columns=[
                    "path", "t", "branch", "ID", "entry", "b", "z", "ETA", "i", "x", "Hatcf", "LnKF",
                    "K", "M", "Q", "P0", "PI", "Bar_i", "Bar_z", "P", "bp0", "bpI", "bp", "Y", "I", "Phi", "C",
                ],
            )

        def fake_create_firm_batches(self, table, batch_size=1024, n_branches=2, eta_resample=True):
            marker = int(table.data[0, table.columns.index("M")].item())
            calls.append(f"build_pv_marker_{marker}")
            return [{"parent": torch.zeros(1, 8), "children": [torch.zeros(1, 8), torch.zeros(1, 8)]}]

        def fake_run_batches(self, batches, n_epochs, log_interval, train_modules, desc_prefix=""):
            if train_modules == ["policy_value"]:
                calls.append("train_pv")
                return {"final_losses": {"total": 0.0}}
            raise AssertionError(f"unexpected train_modules: {train_modules}")

        def fake_run_sdf(self, module_summaries, n_epochs, batch_size, log_interval, n_branches):
            hatc_idx = self.tensor_macro.columns.index("Hatc")
            marker = int(self.tensor_macro.data[0, hatc_idx].item())
            calls.append(f"train_sdf_marker_{marker}")
            module_summaries["sdf_marker"] = marker
            module_summaries["sdf_fc1_gate"] = {"passed": True, "failed_stage": None}
            return {"passed": True, "failed_stage": None}

        def fake_post_refresh_gate(self, module_summaries, batch_size, n_branches):
            calls.append("post_refresh_gate")
            module_summaries["sdf_fc1_post_refresh_gate"] = {
                "passed": bool(post_refresh_pass),
                "failed_stage": None if post_refresh_pass else "post_refresh",
            }
            return {
                "passed": bool(post_refresh_pass),
                "failed_stage": None if post_refresh_pass else "post_refresh",
            }

        episode._simulate_tensor = MethodType(fake_simulate_tensor, episode)
        episode._create_firm_batches_from_tensor = MethodType(fake_create_firm_batches, episode)
        episode._run_batches = MethodType(fake_run_batches, episode)
        episode._run_sdf_recon_from_macro = MethodType(fake_run_sdf, episode)
        episode._evaluate_post_refresh_sdf_gate = MethodType(fake_post_refresh_gate, episode)

        summary = episode.run_episode(
            n_epochs=1,
            batch_size=2,
            log_interval=1,
            n_paths=1,
            group_size=1,
            n_branches=2,
            train_modules=["policy_value", "sdf_fc1"],
            simulate_kwargs={"horizon": 2, "modeb_resimulate_after_pv": resimulate_after_pv},
            episode_mode="modeb",
        )
        return calls, summary

    def test_treatment_a_simulates_once_and_trains_sdf_on_first_macro(self):
        calls, summary = self._make_episode(resimulate_after_pv=False)
        self.assertEqual(
            calls,
            ["simulate1", "train_sdf_marker_1", "simulate2", "post_refresh_gate", "build_pv_marker_2", "train_pv"]
        )
        self.assertEqual(summary["module_summaries"]["sdf_marker"], 1)
        refresh = summary["module_summaries"]["modeb_pre_pv_sdf_refresh_diag"]
        self.assertTrue(refresh["resimulated_after_sdf_gate"])
        self.assertTrue(refresh["policy_value_uses_refreshed_sdf_data"])
        self.assertFalse(summary["module_summaries"]["modeb_post_pv_resimulation_diag"]["resimulated_after_pv"])

    def test_treatment_b_trains_sdf_before_pv_then_resimulates(self):
        calls, summary = self._make_episode(resimulate_after_pv=True)
        self.assertEqual(
            calls,
            ["simulate1", "train_sdf_marker_1", "simulate2", "post_refresh_gate", "build_pv_marker_2", "train_pv", "simulate3"]
        )
        self.assertEqual(summary["module_summaries"]["sdf_marker"], 1)
        diag = summary["module_summaries"]["modeb_post_pv_resimulation_diag"]
        self.assertTrue(diag["resimulated_after_pv"])
        self.assertTrue(diag["rng_state_replayed"])
        self.assertIn("macro_old_to_new_common_rows", diag)
        self.assertIn("firm_old_to_new_keys_common_rows", diag)

    def test_post_refresh_gate_failure_skips_policy_value(self):
        with self.assertRaisesRegex(Exception, "Post-refresh FC1/SDF gate failed"):
            self._make_episode(resimulate_after_pv=False, post_refresh_pass=False)

    def test_modeb_sdf_gate_failure_skips_policy_value(self):
        episode = Episode.__new__(Episode)
        episode.models = {"policy_value": object(), "sdf_fc1": object()}
        episode.optimizers = {}
        episode.config = SimpleNamespace(DEVICE=torch.device("cpu"))
        episode.hyperparams = SimpleNamespace(use_tensor_pipeline=True, simulate_horizon=2)
        episode.device = torch.device("cpu")
        episode.episode_id = 1
        episode.gpu_monitor = _DummyMonitor()
        episode.df = None
        episode.df_macro = None
        episode.df_sdf = None
        episode.tensor_firm = None
        episode.tensor_macro = None
        episode.tensor_sdf = None
        episode.step_count = 0
        episode.loss_history = {}
        episode.add_FC1loss = False
        episode.train_mode = "2time"
        episode._fc1_teacher_forcing_stage = False

        calls = []

        def fake_simulate_tensor(self, n_paths, group_size, n_branches, horizon, simulate_kwargs, export_df=False):
            calls.append("simulate")
            self.tensor_macro = TensorTable(
                data=torch.tensor(
                    [[0.0, 0.0, -1.0, 10.0, 2.0, 1.0, 1.0, 3.0, 1.0, 0.1, 0.5, 4.0]],
                    dtype=torch.float32,
                ),
                columns=["path", "t", "branch", "K", "C", "LnK", "Hatc", "n_firms", "M", "x", "hatcf", "lnkf"],
            )
            self.tensor_firm = TensorTable(
                data=torch.zeros(1, 27),
                columns=[
                    "path", "t", "branch", "ID", "entry", "b", "z", "ETA", "i", "x", "Hatcf", "LnKF",
                    "K", "M", "Q", "P0", "PI", "Bar_i", "Bar_z", "P", "bp0", "bpI", "bp", "Y", "I", "Phi", "C",
                ],
            )

        def fake_run_sdf(self, module_summaries, n_epochs, batch_size, log_interval, n_branches):
            calls.append("train_sdf_failed")
            return {"passed": False, "failed_stage": "fc1_only"}

        def fake_run_batches(self, batches, n_epochs, log_interval, train_modules, desc_prefix=""):
            calls.append("train_pv")
            return {"final_losses": {"total": 0.0}}

        episode._simulate_tensor = MethodType(fake_simulate_tensor, episode)
        episode._run_sdf_recon_from_macro = MethodType(fake_run_sdf, episode)
        episode._run_batches = MethodType(fake_run_batches, episode)

        with self.assertRaisesRegex(Exception, "skip Q/P/bp"):
            episode.run_episode(
                n_epochs=1,
                batch_size=2,
                log_interval=1,
                n_paths=1,
                group_size=1,
                n_branches=2,
                train_modules=["policy_value", "sdf_fc1"],
                simulate_kwargs={"horizon": 2, "modeb_resimulate_after_pv": False},
                episode_mode="modeb",
            )
        self.assertEqual(calls, ["simulate", "train_sdf_failed"])

    def test_modea_joint_sdf_policy_is_disabled_by_default(self):
        episode = Episode.__new__(Episode)
        episode.models = {"policy_value": object(), "sdf_fc1": object()}
        episode.optimizers = {}
        episode.config = SimpleNamespace(DEVICE=torch.device("cpu"))
        episode.hyperparams = SimpleNamespace(
            use_tensor_pipeline=True,
            simulate_horizon=2,
            allow_modea_sdf_after_pv=False,
        )
        episode.device = torch.device("cpu")
        episode.episode_id = 1
        episode.gpu_monitor = _DummyMonitor()
        episode.df = None
        episode.df_macro = None
        episode.df_sdf = None
        episode.tensor_firm = None
        episode.tensor_macro = None
        episode.tensor_sdf = None
        episode.step_count = 0
        episode.loss_history = {}
        episode.add_FC1loss = False
        episode.train_mode = "2time"

        with self.assertRaisesRegex(ValueError, "Mode A trains Policy/Value before FC1/SDF gate"):
            episode.run_episode(
                n_epochs=1,
                batch_size=2,
                log_interval=1,
                n_paths=1,
                group_size=1,
                n_branches=2,
                train_modules=["policy_value", "sdf_fc1"],
                simulate_kwargs={"horizon": 2},
                episode_mode="modea",
            )

    def test_run_episode_resets_post0_stale_bootstrap_phase(self):
        episode = Episode.__new__(Episode)
        episode.models = {}
        episode.optimizers = {}
        episode.config = SimpleNamespace(DEVICE=torch.device("cpu"))
        episode.hyperparams = SimpleNamespace(use_tensor_pipeline=True, simulate_horizon=2)
        episode.device = torch.device("cpu")
        episode.episode_id = 1
        episode.gpu_monitor = _DummyMonitor()
        episode.df = None
        episode.df_macro = None
        episode.df_sdf = None
        episode.tensor_firm = None
        episode.tensor_macro = None
        episode.tensor_sdf = None
        episode.step_count = 0
        episode.loss_history = {}
        episode.add_FC1loss = False
        episode.train_mode = "2time"
        episode.sdf_training_phase = SDFTrainingPhase.EPISODE0_BOOTSTRAP
        episode.reset_sdf_shock_bank = MethodType(lambda self: None, episode)
        episode._resolve_episode_mode = MethodType(lambda self, episode_mode: "modeb", episode)
        episode._use_tensor_pipeline = MethodType(lambda self: True, episode)
        episode._simulate_tensor = MethodType(lambda self, *args, **kwargs: None, episode)

        episode.run_episode(
            n_epochs=1,
            batch_size=2,
            log_interval=1,
            n_paths=1,
            group_size=1,
            n_branches=2,
            train_modules=[],
            simulate_kwargs={"horizon": 2},
            episode_mode="modeb",
        )

        self.assertEqual(episode.sdf_training_phase, SDFTrainingPhase.JOINT_DISABLED)

    def test_episode0_sdf_safety_gate_uses_clip_ratio_not_strict_t_stat(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            sdf_log_mean_target=0.0,
            episode0_sdf_log_mean_error_max=0.25,
            episode0_sdf_clip_low_ratio_max=0.20,
            episode0_sdf_finite_ratio_min=1.0,
        )

        passed, diag = episode._episode0_sdf_safety_gate_passed(
            {
                "ep0_primary_true_state_M_mean": 1.0,
                "ep0_primary_true_state_M_finite_ratio": 1.0,
                "ep0_primary_true_state_M_p99": 1.0,
                "ep0_primary_true_state_M_max": 1.0,
                "ep0_primary_true_state_M_lt_0p7_rate": 0.05,
                "ep0_primary_true_state_M_gt_1p3_rate": 0.0,
                "ep0_primary_true_state_signed_aio_t": 99.0,
                "ep0_recursive_forecast_state_M_mean": float("nan"),
                "ep0_recursive_forecast_state_signed_aio_t": float("nan"),
            },
            prefix="ep0",
        )

        self.assertTrue(passed)
        self.assertEqual(diag["gate_type"], "episode0_sdf_safety")
        self.assertEqual(diag["stage"], SDFTrainingPhase.EPISODE0_BOOTSTRAP.value)
        self.assertEqual(diag["m_mean"], 1.0)
        self.assertEqual(diag["m_lt_0p7_rate"], 0.05)
        self.assertFalse(diag["signed_aio_t_binding"])

    def test_episode0_sdf_continuation_retrains_until_gate_passes(self):
        episode = Episode.__new__(Episode)
        episode.episode_id = 0
        episode.hyperparams = SimpleNamespace(
            episode0_sdf_epochs_per_round=2,
            episode0_sdf_max_rounds=3,
            sdf_fc1_eval_max_batches=0,
        )
        calls = {"train": 0, "eval": 0}

        def fake_run_batches(self, batches, n_epochs, log_interval, train_modules, desc_prefix=""):
            del batches, log_interval, train_modules, desc_prefix
            calls["train"] += 1
            return {"final_losses": {"sdf": float(calls["train"])}, "n_epochs": n_epochs}

        def fake_evaluate(self, batches, prefix, max_batches=None):
            del batches, max_batches
            calls["eval"] += 1
            return {"prefix": prefix}

        def fake_gate(self, eval_metrics, prefix):
            del eval_metrics
            return calls["eval"] >= 2, {"passed": calls["eval"] >= 2, "prefix": prefix}

        episode._run_batches = MethodType(fake_run_batches, episode)
        episode._evaluate_sdf_fc1_batches = MethodType(fake_evaluate, episode)
        episode._episode0_sdf_safety_gate_passed = MethodType(fake_gate, episode)

        summary = episode._run_episode0_sdf_bootstrap_until_gate(
            train_batches=[{"x": torch.tensor([1.0])}],
            val_batches=[{"x": torch.tensor([2.0])}],
            n_epochs=20,
            log_interval=1,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["rounds_completed"], 2)
        self.assertEqual(summary["total_bootstrap_epochs"], 4)
        self.assertEqual(calls["train"], 2)
        self.assertEqual(calls["eval"], 2)

    def test_episode0_sdf_gate_failure_stops_before_policy_value(self):
        class FakeSample:
            policy_requested = False

            def __init__(self, *args, **kwargs):
                del args, kwargs

            def build_sdf_fc1_tensor(self):
                return TensorTable(
                    data=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                    columns=["path", "dummy"],
                )

            def build_policy_value_tensor(self):
                FakeSample.policy_requested = True
                raise AssertionError("Policy/Value data should not be requested after Episode 0 SDF gate failure")

        episode = Episode.__new__(Episode)
        episode.models = {"sdf_fc1": object(), "policy_value": object()}
        episode.optimizers = {}
        episode.config = SimpleNamespace(DEVICE=torch.device("cpu"))
        episode.hyperparams = SimpleNamespace(use_tensor_pipeline=True, simulate_horizon=2)
        episode.device = torch.device("cpu")
        episode.episode_id = 0
        episode.gpu_monitor = _DummyMonitor()
        episode.df = None
        episode.df_macro = None
        episode.df_sdf = None
        episode.tensor_firm = None
        episode.tensor_macro = None
        episode.tensor_sdf = None
        episode.step_count = 0
        episode.loss_history = {}
        episode.add_FC1loss = False
        episode.train_mode = "2time"
        episode._fc1_teacher_forcing_stage = False

        episode.reset_sdf_shock_bank = MethodType(lambda self: None, episode)
        episode._use_tensor_pipeline = MethodType(lambda self: True, episode)
        episode._split_sdf_table_by_path = MethodType(
            lambda self, table: (table, table, {"sdf_fc1_holdout_active": False}),
            episode,
        )
        episode._create_sdf_batches_from_macro_tensor = MethodType(
            lambda self, table, batch_size, n_branches: [{"parent": torch.zeros(1, 7)}],
            episode,
        )
        episode._run_episode0_sdf_bootstrap_until_gate = MethodType(
            lambda self, train_batches, val_batches, n_epochs, log_interval: {
                "passed": False,
                "total_bootstrap_epochs": 2,
                "rounds_completed": 1,
                "final_train_summary": {"final_losses": {"sdf": 1.0}},
            },
            episode,
        )

        original_sample = episode_module.Sample
        episode_module.Sample = FakeSample
        try:
            with self.assertRaisesRegex(NumericalStageFailure, "Episode 0 SDF failed"):
                episode.run_episode(
                    n_epochs=1,
                    batch_size=2,
                    log_interval=1,
                    n_paths=2,
                    group_size=1,
                    n_branches=2,
                    train_modules=["sdf_fc1", "policy_value"],
                    simulate_kwargs={"horizon": 2},
                    episode_mode="mode0",
                )
        finally:
            episode_module.Sample = original_sample

        self.assertFalse(FakeSample.policy_requested)

    def test_fixed_batch_eval_reports_stage2_object_layers(self):
        episode = Episode.__new__(Episode)
        episode.models = {"sdf_fc1": _FakeSdfModel()}
        episode.hyperparams = SimpleNamespace(fc1_use_true_macro_state_in_stage2=False)

        parent = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 4.0, -1.5, 4.2]],
            dtype=torch.float32,
        )
        child0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -0.5, 4.7]], dtype=torch.float32)
        child1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.5, 5.2]], dtype=torch.float32)
        out = episode._evaluate_sdf_fc1_batches(
            [{"parent": parent, "children": [child0, child1]}],
            prefix="check",
        )

        self.assertIn("check_primary_true_state_hatc_next_rmse", out)
        self.assertIn("check_primary_true_state_dlnk_next_rmse", out)
        self.assertIn("check_current_belief_hatc_vs_realized_rmse", out)
        self.assertIn("check_current_belief_lnk_vs_realized_rmse", out)
        self.assertIn("check_recursive_forecast_state_hatc_next_rmse", out)
        self.assertIn("check_recursive_forecast_state_lnk_next_rmse", out)
        self.assertIn("check_recursive_forecast_state_dlnk_next_rmse", out)
        self.assertIn("check_primary_true_state_M_p99", out)
        self.assertIn("check_recursive_forecast_state_M_p99", out)

        self.assertAlmostEqual(out["check_primary_true_state_dlnk_next_rmse"], 0.0, places=6)
        self.assertAlmostEqual(out["check_recursive_forecast_state_dlnk_next_rmse"], 0.0, places=6)
        self.assertAlmostEqual(out["check_current_belief_lnk_vs_realized_rmse"], 0.2, places=6)

    def test_sdf_macro_batches_include_same_path_rollout_tensors(self):
        episode = Episode.__new__(Episode)
        episode.device = torch.device("cpu")
        episode.add_FC1loss = True
        episode.train_mode = "2time"
        episode.hyperparams = SimpleNamespace(
            fc1_rollout_weight=0.5,
            fc1_rollout_horizon=2,
            max_firm_train_units=0,
            pv_eta_resample_enabled=False,
        )
        rows = []
        for t in (1.0, 2.0, 3.0):
            for branch in (0.0, 1.0):
                rows.append(
                    [
                        0.0,
                        t,
                        branch,
                        10.0 + t,
                        20.0 + t + branch,
                        0.1 * t,
                        4.0 + 0.1 * t,
                        1.0 + t,
                        5.0 + t,
                        2.0 + t + branch,
                        6.0 + t + branch,
                    ]
                )
        sdf_table = TensorTable(
            data=torch.tensor(rows, dtype=torch.float32),
            columns=[
                "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
                "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1",
            ],
        )

        batches = episode._create_sdf_batches_from_macro_tensor(sdf_table, batch_size=8, n_branches=2)

        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertIn("fc1_rollout_initial_x", batch)
        self.assertIn("fc1_rollout_future_x", batch)
        self.assertIn("fc1_rollout_target_states", batch)
        self.assertEqual(tuple(batch["fc1_rollout_future_x"].shape[1:]), (2, 1))
        first_source = int(batch["parent_source_index"][0].item())
        self.assertAlmostEqual(batch["fc1_rollout_initial_x"][0, 0].item(), 11.0 + first_source)
        self.assertAlmostEqual(batch["fc1_rollout_future_x"][0, 0, 0].item(), 21.0 + first_source)
        self.assertAlmostEqual(batch["fc1_rollout_future_x"][0, 1, 0].item(), 22.0 + first_source)

    def test_sdf_holdout_split_is_path_disjoint(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            sdf_fc1_val_fraction=0.25,
            sdf_fc1_val_seed=7,
            allow_in_sample_sdf_gate_for_debug=False,
        )
        rows = []
        for path in range(4):
            for t in range(3):
                rows.append([float(path), float(t), 0.0, 1.0, 2.0, 0.1, 4.0, 1.0, 5.0, 2.0, 6.0])
        sdf_table = TensorTable(
            data=torch.tensor(rows, dtype=torch.float32),
            columns=[
                "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
                "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1",
            ],
        )

        train_table, val_table, diag = episode._split_sdf_table_by_path(sdf_table)

        train_paths = set(train_table.data[:, 0].long().tolist())
        val_paths = set(val_table.data[:, 0].long().tolist())
        self.assertTrue(diag["sdf_fc1_holdout_active"])
        self.assertTrue(train_paths)
        self.assertTrue(val_paths)
        self.assertTrue(train_paths.isdisjoint(val_paths))

    def test_sdf_holdout_split_requires_at_least_two_paths(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            sdf_fc1_val_fraction=0.25,
            sdf_fc1_val_seed=7,
            allow_in_sample_sdf_gate_for_debug=False,
        )
        sdf_table = TensorTable(
            data=torch.tensor(
                [[0.0, 1.0, 0.0, 1.0, 2.0, 0.1, 4.0, 1.0, 5.0, 2.0, 6.0]],
                dtype=torch.float32,
            ),
            columns=[
                "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
                "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1",
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "requires at least two paths"):
            episode._split_sdf_table_by_path(sdf_table)

    def test_formal_hyperparams_use_true_state_primary_stage2(self):
        hp = build_hyperparams()
        self.assertTrue(hp.fc1_use_true_macro_state_in_stage2)
        self.assertGreater(hp.fc1_recon_weight, 0.0)
        self.assertLess(hp.fc1_forecast_recon_weight, hp.fc1_recon_weight)


if __name__ == "__main__":
    unittest.main()
