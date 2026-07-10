import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data import TensorTable  # noqa: E402
from experiments.run_utils import build_hyperparams  # noqa: E402
from training.episode import Episode, SDFTrainingPhase  # noqa: E402


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
