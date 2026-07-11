import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

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
        self.assertTrue(result["restored_best_checkpoint"])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))

    def test_sdf_true_only_keeps_fc1_fixed(self):
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
            {"gate": gate},
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
        self.assertEqual(
            Episode._parameter_max_change(episode.models["sdf_fc1"].fc1_model, before),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
