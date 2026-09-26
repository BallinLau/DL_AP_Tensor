import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.run_multi_episode_job import configure_hyperparams, parse_args
from experiments.run_utils import (
    _compute_policy_diagnostic_surfaces as compute_run_utils_surfaces,
)
from experiments.run_episode0_full import (
    _compute_policy_diagnostic_surfaces as compute_episode0_surfaces,
)
from training.episode import Episode


class StageBatchSizeTest(unittest.TestCase):
    def test_legacy_surfaces_require_explicit_child_eta_assumption(self):
        class RecordingModel:
            def __init__(self):
                self.calls = []

            def __call__(self, state):
                self.calls.append(state.detach().clone())
                zeros = torch.zeros_like(state[:, 0:1])
                return SimpleNamespace(
                    Q=zeros,
                    bp0=torch.full_like(zeros, 0.2),
                    bpI=torch.full_like(zeros, 0.8),
                    P0=zeros,
                    PI=zeros,
                    P=state[:, 0:1],
                    bar_z=zeros,
                )

        # Parent eta_t = 1 so refinancing is active and the realized child
        # leverage equals the chosen bp.  The child eta_{t+1} is a separate
        # shock supplied explicitly through ``eta_next_assumed``.
        active = torch.tensor([[0.4, 0.0, 1.0, 0.1, 0.0, -2.0, 4.0]])
        # Parent eta_t = 0 disables refinancing, so the child keeps b_parent
        # regardless of the chosen bp.
        inactive = torch.tensor([[0.4, 0.0, 0.0, 0.1, 0.0, -2.0, 4.0]])
        for surface_fn in (compute_run_utils_surfaces, compute_episode0_surfaces):
            model = RecordingModel()
            surface_fn(model, None, active, eta_next_assumed=0.0)
            self.assertEqual(len(model.calls), 3)
            torch.testing.assert_close(model.calls[1][:, 0:1], torch.tensor([[0.2]]))
            torch.testing.assert_close(model.calls[2][:, 0:1], torch.tensor([[0.8]]))
            torch.testing.assert_close(
                model.calls[1][:, 2:3], torch.zeros_like(active[:, 2:3])
            )
            torch.testing.assert_close(
                model.calls[2][:, 2:3], torch.zeros_like(active[:, 2:3])
            )
            torch.testing.assert_close(model.calls[0][:, 2:3], active[:, 2:3])

            model = RecordingModel()
            surface_fn(model, None, inactive, eta_next_assumed=0.0)
            self.assertEqual(len(model.calls), 3)
            torch.testing.assert_close(model.calls[1][:, 0:1], torch.tensor([[0.4]]))
            torch.testing.assert_close(model.calls[2][:, 0:1], torch.tensor([[0.4]]))
            torch.testing.assert_close(model.calls[0][:, 2:3], inactive[:, 2:3])

    def test_stage_batch_sizes_fallback(self):
        pv, sdf_fc1 = Episode._resolve_stage_batch_sizes(
            batch_size=4096,
            pv_batch_size=None,
            sdf_fc1_batch_size=None,
        )

        self.assertEqual(pv, 4096)
        self.assertEqual(sdf_fc1, 4096)

    def test_stage_batch_sizes_are_independent(self):
        pv, sdf_fc1 = Episode._resolve_stage_batch_sizes(
            batch_size=4096,
            pv_batch_size=20480,
            sdf_fc1_batch_size=4096,
        )

        self.assertEqual(pv, 20480)
        self.assertEqual(sdf_fc1, 4096)

    def test_stage_batch_sizes_reject_nonpositive_pv(self):
        with self.assertRaises(ValueError):
            Episode._resolve_stage_batch_sizes(
                batch_size=4096,
                pv_batch_size=0,
                sdf_fc1_batch_size=4096,
            )

    def test_stage_batch_sizes_reject_nonpositive_sdf_fc1(self):
        with self.assertRaises(ValueError):
            Episode._resolve_stage_batch_sizes(
                batch_size=4096,
                pv_batch_size=20480,
                sdf_fc1_batch_size=0,
            )

    def test_cli_stage_batch_overrides_reach_hyperparams(self):
        argv = [
            "run_multi_episode_job.py",
            "--batch-size",
            "4096",
            "--pv-batch-size",
            "20480",
            "--sdf-fc1-batch-size",
            "4096",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()
        hp = configure_hyperparams(args)

        self.assertEqual(hp.batch_size, 4096)
        self.assertEqual(hp.pv_batch_size, 20480)
        self.assertEqual(hp.sdf_fc1_batch_size, 4096)

    def test_cli_stage_batch_sizes_fall_back_to_global(self):
        argv = [
            "run_multi_episode_job.py",
            "--batch-size",
            "4096",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()
        hp = configure_hyperparams(args)

        self.assertEqual(hp.batch_size, 4096)
        self.assertEqual(hp.pv_batch_size, 4096)
        self.assertEqual(hp.sdf_fc1_batch_size, 4096)

    def test_cli_current_eta_bp_sampling_coexists_with_pv_mixture(self):
        argv = [
            "run_multi_episode_job.py",
            "--pv-training-flow",
            "staged",
            "--firm-target-update",
            "stage_hard",
            "--pv-mixture-enabled",
            "--no-pv-eta-resample-enabled",
            "--bp-current-eta-resample-enabled",
            "--bp-current-eta1-train-share",
            "0.25",
            "--bp-current-eta-resample-seed",
            "13579",
            "--bp-distill-max-optimizer-steps",
            "500",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()
        hp = configure_hyperparams(args)

        self.assertTrue(hp.pv_mixture_enabled)
        self.assertTrue(hp.bp_current_eta_resample_enabled)
        self.assertEqual(hp.bp_current_eta1_train_share, 0.25)
        self.assertEqual(hp.bp_current_eta_resample_seed, 13579)
        self.assertEqual(hp.bp_distill_max_optimizer_steps, 500)
        self.assertFalse(hp.pv_eta_resample_enabled)

    def test_cli_training_conditioning_controls_are_wired(self):
        argv = [
            "run_multi_episode_job.py",
            "--pv-training-flow",
            "staged",
            "--firm-target-update",
            "stage_hard",
            "--q-parameterization",
            "hybrid_regime",
            "--q-claim-coverage-low-b-enabled",
            "--q-claim-coverage-low-b-anchors",
            "0.005,0.01,0.025",
            "--pv-current-eta-balance-enabled",
            "--pv-current-eta1-train-share",
            "0.50",
            "--pv-current-eta-balance-validation",
            "--pv-current-eta-balance-seed",
            "97531",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()
        hp = configure_hyperparams(args)

        self.assertTrue(hp.q_claim_coverage_low_b_enabled)
        self.assertEqual(hp.q_claim_coverage_low_b_anchors, (0.005, 0.01, 0.025))
        self.assertTrue(hp.pv_current_eta_balance_enabled)
        self.assertEqual(hp.pv_current_eta1_train_share, 0.50)
        self.assertTrue(hp.pv_current_eta_balance_validation)
        self.assertEqual(hp.pv_current_eta_balance_seed, 97531)

    def test_tensor_pv_batches_do_not_resample_realized_child_eta(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            pv_eta_resample_enabled=True,
            pv_eta_resample_active_share=0.5,
            max_firm_train_units=0,
        )
        parent = torch.zeros(4, 8)
        parent[:, 2:3] = 1.0
        child0 = torch.zeros_like(parent)
        child1 = torch.zeros_like(parent)
        child1[0, 2] = 1.0

        batches = episode._build_batches_from_parent_children(
            parent=parent,
            children=[child0, child1],
            batch_size=4,
            eta_resample=True,
        )

        selected = batches[0]["parent_source_index"]
        self.assertEqual(sorted(selected.tolist()), [0, 1, 2, 3])
        selected_child_active = torch.maximum(child0[selected, 2], child1[selected, 2])
        self.assertEqual(int((selected_child_active > 0.5).sum().item()), 1)

    def test_dataframe_pv_batches_do_not_resample_realized_child_eta(self):
        episode = Episode.__new__(Episode)
        episode.device = torch.device("cpu")
        episode.hyperparams = SimpleNamespace(
            pv_eta_resample_enabled=True,
            pv_eta_resample_active_share=0.75,
            max_firm_train_units=0,
        )

        rows = []
        parent_eta = [1.0, 1.0, 1.0, 1.0]
        child_eta = [1.0, 0.0, 0.0, 0.0]

        for group_id, eta_parent in enumerate(parent_eta):
            common = {
                "path": group_id,
                "ID": str(group_id),
                "b": 0.1 + 0.1 * group_id,
                "z": 0.2,
                "i": 0.1,
                "x": 0.0,
                "Hatcf": -2.0,
                "LnKF": 4.0,
                "M": 1.0,
            }

            rows.append(
                {
                    **common,
                    "t": 0,
                    "branch": -1,
                    "ETA": eta_parent,
                }
            )

            for branch in range(2):
                rows.append(
                    {
                        **common,
                        "t": 1,
                        "branch": branch,
                        "ETA": child_eta[group_id],
                    }
                )

        df = pd.DataFrame(rows)

        torch.manual_seed(1234)
        batches = episode._create_firm_batches_from_df(
            df,
            batch_size=4,
            n_branches=2,
        )

        self.assertTrue(batches)

        selected_child = torch.cat(
            [batch["children"][0] for batch in batches],
            dim=0,
        )
        selected_child_eta = selected_child[:, 2]

        self.assertEqual(selected_child.shape[0], 4)
        self.assertEqual(int((selected_child_eta > 0.5).sum().item()), 1)

    def test_child_eta_resampling_is_scoped_to_bp_train_cache(self):
        # ``_resample_bp_target_cache`` is a deprecated no-op: the child shock
        # eta_{t+1} never gates leverage, so no child-eta stratum exists to
        # rebalance.  The cache must be returned untouched with explicit
        # metadata stating that nothing was done.
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            pv_eta_resample_enabled=True,
            pv_eta_resample_active_share=0.5,
        )
        parent = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        active = torch.tensor([[True], [False], [False], [False]])
        cache = [{
            "batch_id": 0,
            "parent": parent,
            "source_id": None,
            "source_index": None,
            "bp0_target": torch.zeros(4, 1),
            "bpi_target": torch.zeros(4, 1),
            "mix_target": torch.zeros(4, 1),
            "bp0_confidence": torch.ones(4, 1),
            "bpi_confidence": torch.ones(4, 1),
            "mix_confidence": torch.ones(4, 1),
            "mix_sample_weight": torch.ones(4, 1),
            "eta_next_active": active,
            "teacher_snapshot_hash": "teacher",
        }]

        torch.manual_seed(1234)
        rng_before = torch.get_rng_state().clone()
        resampled, summary = episode._resample_bp_target_cache(cache)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)

        self.assertIs(resampled, cache)
        self.assertFalse(summary["applied"])
        self.assertTrue(summary["deprecated"])
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["scope"], "bp_distillation_train_cache_only")
        self.assertFalse(summary["validation_cache_resampled"])
        self.assertEqual(
            summary["reason"],
            "deprecated_child_eta_next_does_not_gate_bp_availability",
        )
        # Cache rows are untouched, including the eta_next diagnostic column.
        torch.testing.assert_close(cache[0]["eta_next_active"], active)

    def test_current_eta_resampling_reads_parent_column_not_future_eta(self):
        # ``_resample_bp_target_cache_by_current_eta`` is deprecated: BP
        # supervision is already conditional on the CURRENT parent eta_t inside
        # ``_build_bp_target_cache``, so nothing is resampled.  The reported
        # counts must still be keyed off the parent eta column (never off the
        # future eta_next diagnostic), and the cache must be left untouched.
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            bp_current_eta_resample_enabled=True,
            bp_current_eta1_train_share=0.25,
            bp_current_eta_resample_seed=13579,
            pv_eta_resample_enabled=False,
        )
        episode.episode_id = 2
        n_rows = 100
        parent = torch.zeros(n_rows, 7)
        parent[:10, 2] = 1.0
        # Make the future-eta diagnostic exactly opposite to current eta.  If
        # the sampler uses eta_next_active, the assertions below reverse.
        eta_next_active = (parent[:, 2:3] < 0.5)
        cache = [{
            "batch_id": 0,
            "parent": parent.clone(),
            "source_id": torch.arange(n_rows),
            "source_index": torch.arange(n_rows),
            "bp0_target": torch.zeros(n_rows, 1),
            "bpi_target": torch.zeros(n_rows, 1),
            "mix_target": torch.zeros(n_rows, 1),
            "bp0_confidence": torch.ones(n_rows, 1),
            "bpi_confidence": torch.ones(n_rows, 1),
            "mix_confidence": torch.ones(n_rows, 1),
            "mix_sample_weight": torch.ones(n_rows, 1),
            "eta_next_active": eta_next_active.clone(),
            "teacher_snapshot_hash": "teacher",
        }]

        torch.manual_seed(1234)
        global_rng_before = torch.get_rng_state().clone()
        resampled, summary = episode._resample_bp_target_cache_by_current_eta(cache)
        global_rng_after = torch.get_rng_state().clone()
        repeated, repeated_summary = episode._resample_bp_target_cache_by_current_eta(cache)

        self.assertIs(resampled, cache)
        self.assertIs(repeated, cache)
        self.assertFalse(summary["applied"])
        self.assertTrue(summary["deprecated"])
        self.assertTrue(summary["enabled"])
        self.assertEqual(
            summary["reason"],
            "deprecated_bp_supervision_is_already_eta_current_conditional",
        )
        self.assertEqual(summary["eta_field"], "parent[:, 2]")
        # Counts are read from the CURRENT parent eta column (10 of 100), and
        # are never resampled up to the target share.
        self.assertEqual(summary["current_eta1_count_before"], 10)
        self.assertEqual(summary["current_eta0_count_before"], 90)
        self.assertEqual(summary["current_eta1_count_after"], 10)
        self.assertAlmostEqual(summary["current_eta1_share_after"], 0.10)
        self.assertEqual(summary["target_current_eta1_share"], 0.25)
        self.assertEqual(
            repeated_summary["current_eta1_count_after"], 10
        )
        # No RNG is consumed and the cache (including the future-eta
        # diagnostic) is byte-for-byte unchanged.
        torch.testing.assert_close(global_rng_after, global_rng_before)
        torch.testing.assert_close(torch.get_rng_state(), global_rng_before)
        self.assertFalse(summary["validation_cache_resampled"])
        torch.testing.assert_close(cache[0]["parent"], parent)
        torch.testing.assert_close(cache[0]["eta_next_active"], eta_next_active)

    def test_disabled_current_eta_resampling_preserves_legacy_cache(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            bp_current_eta_resample_enabled=False,
            bp_current_eta1_train_share=0.25,
        )
        cache = [{"parent": torch.zeros(4, 7)}]

        result, summary = episode._resample_bp_target_cache_by_current_eta(cache)

        self.assertIs(result, cache)
        self.assertFalse(summary["applied"])
        self.assertTrue(summary["deprecated"])
        self.assertFalse(summary["enabled"])
        self.assertEqual(
            summary["reason"],
            "deprecated_bp_supervision_is_already_eta_current_conditional",
        )


if __name__ == "__main__":
    unittest.main()
