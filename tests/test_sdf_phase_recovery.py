import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data import TensorTable
from training.episode import Episode, NumericalStageFailure, SDFTrainingPhase


class _DummySdfFc1(nn.Module):
    def __init__(self):
        super().__init__()
        self.sdf_model = nn.Linear(1, 1)
        self.value_model = nn.Linear(1, 1)
        self.fc1_model = nn.Linear(1, 1)


def _make_episode(eval_items):
    model = _DummySdfFc1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    episode = Episode.__new__(Episode)
    episode.models = {"sdf_fc1": model}
    episode.optimizers = {"sdf_fc1": optimizer}
    episode.device = torch.device("cpu")
    episode.hyperparams = SimpleNamespace(
        sdf_fc1_eval_max_batches=0,
        sdf_reset_optimizer_on_true_start=False,
        sdf_required_consecutive_passes=1,
        sdf_stop_when_gate_passes=True,
        sdf_collapse_patience=1,
        sdf_restore_best_checkpoint=True,
        sdf_clear_optimizer_after_restore=True,
        stage_parameter_invariance_check_enabled=True,
    )
    episode._eval_items = list(eval_items)
    episode._train_calls = 0

    def _evaluate(_batches, prefix, max_batches=None):
        item = episode._eval_items.pop(0)
        item = dict(item)
        item["prefix"] = prefix
        gate = item.get("gate", {})
        item.setdefault(f"{prefix}_primary_true_state_M_mean", gate.get("m_mean", 0.98))
        item.setdefault(f"{prefix}_primary_true_state_M_finite_ratio", gate.get("m_finite_ratio", 1.0))
        item.setdefault(f"{prefix}_primary_true_state_normalized_signed_aio_t", gate.get("signed_aio_t", 0.0))
        item.setdefault(f"{prefix}_primary_true_state_hatc_next_rmse", gate.get("hatc_rmse", 0.1))
        item.setdefault(f"{prefix}_primary_true_state_lnk_next_rmse", gate.get("lnk_rmse", 0.1))
        item.setdefault(f"{prefix}_primary_true_state_wealth_ratio_p50", gate.get("wealth_ratio_p50", 1.0))
        item.setdefault(f"{prefix}_primary_true_state_wealth_ratio_p99", gate.get("wealth_ratio_p99", 1.0))
        return item

    def _gate(eval_metrics, prefix, stage):
        gate = dict(eval_metrics["gate"])
        gate.setdefault("stage", stage.value)
        return bool(gate.get("passed", False)), gate

    def _train(*_args, **_kwargs):
        episode._train_calls += 1
        with torch.no_grad():
            model.sdf_model.weight.add_(1.0)
        return {"final_losses": {"total": float(episode._train_calls)}}

    episode._evaluate_sdf_fc1_batches = _evaluate
    episode._sdf_gate_passed = _gate
    episode._run_batches = _train
    return episode


class SdfPhaseRecoveryTest(unittest.TestCase):
    def test_numerical_stage_failure_carries_partial_summary(self):
        with self.assertRaises(NumericalStageFailure) as ctx:
            raise NumericalStageFailure(
                "failed",
                diagnostics={
                    "partial_module_summaries": {
                        "before": {"m_mean": 0.98},
                    },
                },
            )
        self.assertEqual(
            ctx.exception.diagnostics["partial_module_summaries"]["before"]["m_mean"],
            0.98,
        )

    def test_sdf_phase_considers_epoch_zero_checkpoint_and_restores(self):
        good_gate = {
            "passed": False,
            "m_mean": 0.98,
            "m_finite_ratio": 1.0,
            "m_finite_ratio_min": 1.0,
            "log_mean_error": 0.01,
            "max_log_mean_error": 0.02,
            "signed_aio_t": 0.5,
            "max_signed_t_abs": 2.0,
            "log_mean_target": 0.0,
        }
        collapsed_gate = {
            **good_gate,
            "m_mean": 1e-4,
            "log_mean_error": 9.0,
            "signed_aio_t": 8.0,
        }
        episode = _make_episode([
            {"gate": good_gate},
            {"gate": collapsed_gate},
            {"gate": collapsed_gate},
            {"gate": good_gate},
        ])
        model = episode.models["sdf_fc1"]
        before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        result = episode._run_sdf_phase_with_validation(
            train_batches=[{"x": torch.ones(1)}],
            val_batches=[{"x": torch.ones(1)}],
            n_epochs=5,
            log_interval=1,
            stage=SDFTrainingPhase.SDF_TRUE_ONLY,
            prefix="sdf_true",
        )

        self.assertEqual(result["epochs_completed"], 1)
        self.assertEqual(result["best_epoch"], 0)
        self.assertEqual(result["accepted_epochs"], 0)
        self.assertTrue(result["skipped_current_stage"])
        self.assertTrue(result["restored_best_checkpoint"])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))

    def test_sdf_true_only_keeps_fc1_fixed(self):
        before_gate = {
            "passed": False,
            "m_mean": 0.7,
            "m_finite_ratio": 1.0,
            "m_finite_ratio_min": 1.0,
            "log_mean_error": 0.3,
            "max_log_mean_error": 0.02,
            "signed_aio_t": 5.0,
            "max_signed_t_abs": 2.0,
            "log_mean_target": 0.0,
        }
        gate = {
            "passed": True,
            "m_mean": 0.98,
            "m_finite_ratio": 1.0,
            "m_finite_ratio_min": 1.0,
            "log_mean_error": 0.0,
            "max_log_mean_error": 0.02,
            "signed_aio_t": 0.0,
            "max_signed_t_abs": 2.0,
            "log_mean_target": 0.0,
        }
        episode = _make_episode([
            {"gate": before_gate},
            {"gate": gate},
            {"gate": gate},
        ])
        before = Episode._parameter_snapshot(episode.models["sdf_fc1"].fc1_model)
        result = episode._run_sdf_phase_with_validation(
            train_batches=[{"x": torch.ones(1)}],
            val_batches=[{"x": torch.ones(1)}],
            n_epochs=1,
            log_interval=1,
            stage=SDFTrainingPhase.SDF_TRUE_ONLY,
            prefix="sdf_true",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_epochs"], 1)
        self.assertEqual(
            Episode._parameter_max_change(episode.models["sdf_fc1"].fc1_model, before),
            0.0,
        )

    def test_post_refresh_safety_mode_can_pass_when_strict_gate_fails(self):
        episode = Episode.__new__(Episode)
        episode.tensor_macro = TensorTable(
            data=torch.zeros(1, 12),
            columns=["path", "t", "branch", "K", "C", "LnK", "Hatc", "n_firms", "M", "x", "hatcf", "lnkf"],
        )
        episode.df_macro = None
        episode.add_FC1loss = False
        episode.hyperparams = SimpleNamespace(
            sdf_fc1_eval_max_batches=0,
            sdf_gate_residual_mode="normalized_ratio",
            sdf_log_mean_target=0.0,
            sdf_collapse_lower_ratio=0.1,
            sdf_collapse_upper_ratio=10.0,
            sdf_post_refresh_gate_mode="safety",
        )

        def _use_tensor_pipeline():
            return True

        def _build_sdf_pairs(_table):
            return episode.tensor_macro

        def _split(_table):
            return _table, _table, {"mock": True}

        def _create(_table, batch_size, n_branches):
            return [{"mock": True}]

        def _eval(_batches, prefix, max_batches=None):
            return {
                f"{prefix}_primary_true_state_M_mean": 0.98,
                f"{prefix}_primary_true_state_M_finite_ratio": 1.0,
                f"{prefix}_primary_true_state_normalized_signed_aio_t": 99.0,
                f"{prefix}_recursive_forecast_state_M_mean": 0.98,
                f"{prefix}_recursive_forecast_state_M_finite_ratio": 1.0,
                f"{prefix}_recursive_forecast_state_normalized_signed_aio_t": 99.0,
                f"{prefix}_primary_true_state_hatc_next_target_finite_n": 10.0,
                f"{prefix}_primary_true_state_hatc_next_target_finite_ratio": 1.0,
                f"{prefix}_primary_true_state_hatc_next_target_std": 1.0,
                f"{prefix}_primary_true_state_hatc_next_pred_finite_ratio": 1.0,
                f"{prefix}_primary_true_state_hatc_next_r2": -1.0,
                f"{prefix}_primary_true_state_hatc_next_skill_vs_persistence": -1.0,
                f"{prefix}_primary_true_state_hatc_next_rmse": 10.0,
                f"{prefix}_primary_true_state_lnk_next_target_finite_n": 10.0,
                f"{prefix}_primary_true_state_lnk_next_target_finite_ratio": 1.0,
                f"{prefix}_primary_true_state_lnk_next_target_std": 1.0,
                f"{prefix}_primary_true_state_lnk_next_pred_finite_ratio": 1.0,
                f"{prefix}_primary_true_state_lnk_next_r2": -1.0,
                f"{prefix}_primary_true_state_lnk_next_skill_vs_persistence": -1.0,
                f"{prefix}_primary_true_state_lnk_next_rmse": 10.0,
            }

        episode._use_tensor_pipeline = _use_tensor_pipeline
        episode._build_sdf_pairs_from_macro_tensor = _build_sdf_pairs
        episode._split_sdf_table_by_path = _split
        episode._create_sdf_batches_from_macro_tensor = _create
        episode._evaluate_sdf_fc1_batches = _eval
        episode.set_sdf_training_phase = lambda phase: None
        episode.sdf_training_phase = SDFTrainingPhase.SDF_TRUE_ONLY

        result = episode._evaluate_post_refresh_sdf_gate(
            module_summaries={},
            batch_size=2,
            n_branches=2,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["safety_passed"])
        self.assertFalse(result["strict_passed"])
        self.assertEqual(result["gate_mode"], "safety")


if __name__ == "__main__":
    unittest.main()
