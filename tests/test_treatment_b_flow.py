import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data import TensorTable  # noqa: E402
from experiments.run_utils import build_hyperparams  # noqa: E402
from training.episode import Episode  # noqa: E402


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
    def _make_episode(self, resimulate_after_pv):
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
                    [[0.0, 0.0, -1.0, 7.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1, 0.5, 4.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.4, 0.2, 1.1, 0.2, 0.3, 0.25, 1.0, 0.1, 0.0, 0.8]],
                    dtype=torch.float32,
                ),
                columns=[
                    "path", "t", "branch", "ID", "entry", "b", "z", "ETA", "i", "x", "Hatcf", "LnKF",
                    "K", "M", "Q", "P0", "PI", "Bar_i", "Bar_z", "P", "bp0", "bpI", "bp", "Y", "I", "Phi", "C",
                ],
            )

        def fake_create_firm_batches(self, table, batch_size=1024, n_branches=2, eta_resample=True):
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

        episode._simulate_tensor = MethodType(fake_simulate_tensor, episode)
        episode._create_firm_batches_from_tensor = MethodType(fake_create_firm_batches, episode)
        episode._run_batches = MethodType(fake_run_batches, episode)
        episode._run_sdf_recon_from_macro = MethodType(fake_run_sdf, episode)

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
        self.assertEqual(calls, ["simulate1", "train_pv", "train_sdf_marker_1"])
        self.assertEqual(summary["module_summaries"]["sdf_marker"], 1)
        self.assertFalse(summary["module_summaries"]["modeb_post_pv_resimulation_diag"]["resimulated_after_pv"])

    def test_treatment_b_resimulates_after_pv_and_trains_sdf_on_second_macro(self):
        calls, summary = self._make_episode(resimulate_after_pv=True)
        self.assertEqual(calls, ["simulate1", "train_pv", "simulate2", "train_sdf_marker_2"])
        self.assertEqual(summary["module_summaries"]["sdf_marker"], 2)
        diag = summary["module_summaries"]["modeb_post_pv_resimulation_diag"]
        self.assertTrue(diag["resimulated_after_pv"])
        self.assertTrue(diag["rng_state_replayed"])
        self.assertIn("macro_old_to_new_common_rows", diag)
        self.assertIn("firm_old_to_new_keys_common_rows", diag)

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

    def test_formal_hyperparams_use_true_state_primary_stage2(self):
        hp = build_hyperparams()
        self.assertTrue(hp.fc1_use_true_macro_state_in_stage2)
        self.assertGreater(hp.fc1_recon_weight, 0.0)
        self.assertLess(hp.fc1_forecast_recon_weight, hp.fc1_recon_weight)


if __name__ == "__main__":
    unittest.main()
