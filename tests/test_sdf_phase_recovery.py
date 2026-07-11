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
from training.scheduler import LearningRateScheduler


class _DummySdfFc1(nn.Module):
    def __init__(self):
        super().__init__()
        self.sdf_model = nn.Linear(1, 1)
        self.value_model = nn.Linear(1, 1)
        self.fc1_model = nn.Linear(1, 1)


class _DummyScheduler:
    def __init__(self, base_lr=2e-4):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.group_base_lrs = [base_lr]

    def state_dict(self):
        return {
            "current_step": 0,
            "base_lr": float(self.base_lr),
            "current_lr": float(self.current_lr),
            "group_base_lrs": list(self.group_base_lrs),
        }

    def load_state_dict(self, state):
        self.base_lr = float(state["base_lr"])
        self.current_lr = float(state["current_lr"])
        self.group_base_lrs = list(state["group_base_lrs"])


def _make_episode(eval_items):
    model = _DummySdfFc1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    episode = Episode.__new__(Episode)
    episode.models = {"sdf_fc1": model}
    episode.optimizers = {"sdf_fc1": optimizer}
    episode.lr_schedulers = {}
    episode.device = torch.device("cpu")
    episode.add_FC1loss = True
    episode.hyperparams = SimpleNamespace(
        sdf_fc1_eval_max_batches=0,
        sdf_reset_optimizer_on_true_start=False,
        sdf_required_consecutive_passes=1,
        sdf_stop_when_gate_passes=True,
        sdf_collapse_patience=1,
        sdf_restore_best_checkpoint=True,
        sdf_clear_optimizer_after_restore=True,
        stage_parameter_invariance_check_enabled=True,
        sdf_stage2_lr=2e-4,
        sdf_stage1_lr=2e-4,
        fc1_lr=2e-4,
        sdf_true_only_lr=4e-5,
        sdf_true_target_batches=20,
        sdf_min_parent_groups_per_batch=256,
        stage_lr_decay_on_reject=0.1,
        stage_max_retries=1,
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


def _make_post_refresh_episode(primary_m=0.98, recursive_m=0.98):
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
        fc1_gate_min_pairs=1,
        fc1_rollout_finite_ratio_min=1.0,
        fc1_target_std_floor=1e-4,
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
            f"{prefix}_primary_true_state_M_mean": primary_m,
            f"{prefix}_primary_true_state_M_finite_ratio": 1.0,
            f"{prefix}_primary_true_state_normalized_signed_aio_t": 99.0,
            f"{prefix}_recursive_forecast_state_M_mean": recursive_m,
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
    return episode


class SdfPhaseRecoveryTest(unittest.TestCase):
    def test_learning_rate_scheduler_round_trip_restores_base_lr(self):
        param = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([{"params": [param], "lr": 1e-3, "base_lr": 1e-3}])
        scheduler = LearningRateScheduler(
            optimizer,
            base_lr=1e-3,
            warmup_steps=0,
            decay_type="fixed",
            total_steps=10,
            min_lr=1e-8,
        )
        state = scheduler.state_dict()
        scheduler.base_lr = 9.0
        scheduler.current_lr = 9.0
        scheduler.group_base_lrs = [9.0]

        scheduler.load_state_dict(state)

        self.assertEqual(scheduler.current_step, 0)
        self.assertAlmostEqual(scheduler.base_lr, 1e-3)
        self.assertAlmostEqual(scheduler.current_lr, 1e-3)
        self.assertEqual(scheduler.group_base_lrs, [1e-3])

    def test_scheduler_step_keeps_decayed_retry_lr(self):
        param = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([{"params": [param], "lr": 1e-3, "base_lr": 1e-3}])
        scheduler = LearningRateScheduler(
            optimizer,
            base_lr=1e-3,
            warmup_steps=0,
            decay_type="fixed",
            total_steps=10,
            min_lr=1e-8,
        )

        decayed = Episode._decay_optimizer_and_scheduler_lr(optimizer, scheduler, 0.1)
        scheduler.step()

        self.assertEqual(decayed, [1e-4])
        self.assertAlmostEqual(scheduler.base_lr, 1e-4)
        self.assertAlmostEqual(scheduler.current_lr, 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[0]["base_lr"], 1e-4)

    def test_sdf_true_target_batching_preserves_parent_groups_and_children(self):
        episode = Episode.__new__(Episode)
        episode.device = torch.device("cpu")
        episode.add_FC1loss = True
        episode.train_mode = "multi"
        episode.hyperparams = SimpleNamespace(
            fc1_rollout_horizon=0,
            fc1_recursive_aux_training_enabled=False,
            fc1_rollout_weight=0.0,
            fc1_rollout_diagnostic_enabled=False,
            max_firm_train_units=0,
            pv_eta_resample_enabled=False,
        )
        columns = [
            "path",
            "t",
            "branch",
            "x_t",
            "x_t1",
            "Hatcf_t",
            "LnKF_t",
            "Hatc_t",
            "LnK_t",
            "Hatc_t1",
            "LnK_t1",
        ]
        rows = []
        n_parent_groups = 400
        for path in range(n_parent_groups):
            for branch in range(2):
                rows.append([
                    float(path),
                    3.0,
                    float(branch),
                    0.1 * path,
                    0.1 * path + branch,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0 + branch,
                    6.0 + branch,
                ])
        table = TensorTable(
            data=torch.tensor(rows, dtype=torch.float32),
            columns=columns,
        )

        batches = episode._create_sdf_batches_from_macro_tensor(
            table,
            batch_size=4096,
            n_branches=2,
            target_num_batches=20,
            min_parent_groups_per_batch=10,
        )

        parent_counts = [int(batch["parent"].shape[0]) for batch in batches]
        self.assertEqual(len(batches), 20)
        self.assertEqual(sum(parent_counts), n_parent_groups)
        self.assertEqual(min(parent_counts), 20)
        self.assertEqual(max(parent_counts), 20)
        for batch in batches:
            self.assertEqual(len(batch["children"]), 2)
            self.assertEqual(batch["children"][0].shape[0], batch["parent"].shape[0])
            self.assertEqual(batch["children"][1].shape[0], batch["parent"].shape[0])
        source_indices = torch.cat([batch["parent_source_index"] for batch in batches])
        self.assertEqual(source_indices.numel(), n_parent_groups)
        self.assertEqual(torch.unique(source_indices).numel(), n_parent_groups)
        parent_indices = torch.cat([batch["parent_index"] for batch in batches])
        self.assertEqual(parent_indices.numel(), n_parent_groups)
        self.assertEqual(torch.unique(parent_indices).numel(), n_parent_groups)
        self.assertEqual(int(parent_indices.min().item()), 0)
        self.assertEqual(int(parent_indices.max().item()), n_parent_groups - 1)

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

    def test_sdf_retry_uses_decayed_lr_after_rollback(self):
        before_gate = {
            "passed": False,
            "m_mean": 0.98,
            "m_finite_ratio": 1.0,
            "m_finite_ratio_min": 1.0,
            "log_mean_error": 0.01,
            "max_log_mean_error": 0.02,
            "signed_aio_t": 0.0,
            "max_signed_t_abs": 2.0,
            "log_mean_target": 0.0,
        }
        bad_gate = {
            **before_gate,
            "m_mean": 0.01,
            "log_mean_error": 4.0,
        }
        episode = _make_episode([
            {"gate": before_gate},
            {"gate": bad_gate},
            {"gate": bad_gate},
            {"gate": before_gate},
        ])
        optimizer = episode.optimizers["sdf_fc1"]
        optimizer.param_groups[0]["group_name"] = "sdf"
        episode.lr_schedulers = {"sdf_fc1": _DummyScheduler(base_lr=2e-4)}
        observed_lrs = []

        def _train(*_args, **_kwargs):
            observed_lrs.append(float(optimizer.param_groups[0]["lr"]))
            return {"final_losses": {"total": float(len(observed_lrs))}}

        episode._run_batches = _train
        result = episode._run_sdf_phase_with_validation(
            train_batches=[{"x": torch.ones(1)}],
            val_batches=[{"x": torch.ones(1)}],
            n_epochs=1,
            log_interval=1,
            stage=SDFTrainingPhase.SDF_TRUE_ONLY,
            prefix="sdf_true",
        )

        self.assertEqual(len(observed_lrs), 2)
        self.assertAlmostEqual(observed_lrs[0], 4e-5)
        self.assertAlmostEqual(observed_lrs[1], 4e-6)
        self.assertTrue(result["skipped_current_stage"])

    def test_sdf_recovery_accepts_step_toward_safe_region(self):
        episode = _make_episode([])
        before = {
            "safe": False,
            "m_mean": 0.017,
            "m_target": 0.98,
            "m_finite_ratio": 1.0,
            "sdf_score": 4.1,
        }
        after = {
            "safe": False,
            "m_mean": 0.021,
            "m_target": 0.98,
            "m_finite_ratio": 1.0,
            "sdf_score": 3.8,
        }
        accepted, reason = episode._sdf_epoch_acceptance(before, after)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted_recovery_step")

    def test_sdf_recovery_rejects_step_away_from_safe_region(self):
        episode = _make_episode([])
        before = {
            "safe": False,
            "m_mean": 0.017,
            "m_target": 0.98,
            "m_finite_ratio": 1.0,
            "sdf_score": 4.1,
        }
        after = {
            "safe": False,
            "m_mean": 0.002,
            "m_target": 0.98,
            "m_finite_ratio": 1.0,
            "sdf_score": 3.8,
        }
        accepted, reason = episode._sdf_epoch_acceptance(before, after)
        self.assertFalse(accepted)
        self.assertEqual(reason, "not_moving_toward_safe_region")

    def test_post_refresh_safety_mode_can_pass_when_strict_gate_fails(self):
        episode = _make_post_refresh_episode(primary_m=0.98, recursive_m=0.98)

        result = episode._evaluate_post_refresh_sdf_gate(
            module_summaries={},
            batch_size=2,
            n_branches=2,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["safety_passed"])
        self.assertFalse(result["strict_passed"])
        self.assertEqual(result["gate_mode"], "safety")

    def test_post_refresh_uses_primary_state_when_recursive_collapses(self):
        episode = _make_post_refresh_episode(primary_m=0.98, recursive_m=0.001)
        result = episode._evaluate_post_refresh_sdf_gate(
            module_summaries={},
            batch_size=2,
            n_branches=2,
        )
        self.assertTrue(result["safety_passed"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["hard_gate_state"], "primary_calculated_state")

    def test_post_refresh_rejects_primary_collapse_even_if_recursive_normal(self):
        episode = _make_post_refresh_episode(primary_m=0.001, recursive_m=0.98)
        result = episode._evaluate_post_refresh_sdf_gate(
            module_summaries={},
            batch_size=2,
            n_branches=2,
        )
        self.assertFalse(result["safety_passed"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["hard_gate_state"], "primary_calculated_state")


if __name__ == "__main__":
    unittest.main()
