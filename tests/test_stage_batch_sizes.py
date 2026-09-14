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

        base = torch.tensor([[0.4, 0.0, 0.0, 0.1, 0.0, -2.0, 4.0]])
        for surface_fn in (compute_run_utils_surfaces, compute_episode0_surfaces):
            model = RecordingModel()
            surface_fn(model, None, base, eta_next_assumed=1.0)
            self.assertEqual(len(model.calls), 3)
            torch.testing.assert_close(model.calls[1][:, 0:1], torch.tensor([[0.2]]))
            torch.testing.assert_close(model.calls[2][:, 0:1], torch.tensor([[0.8]]))
            torch.testing.assert_close(model.calls[1][:, 2:3], torch.ones_like(base[:, 2:3]))
            torch.testing.assert_close(model.calls[2][:, 2:3], torch.ones_like(base[:, 2:3]))
            torch.testing.assert_close(model.calls[0][:, 2:3], base[:, 2:3])

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
        resampled, summary = episode._resample_bp_target_cache(cache)

        self.assertTrue(summary["applied"])
        self.assertEqual(summary["scope"], "bp_distillation_train_cache_only")
        self.assertFalse(summary["validation_cache_resampled"])
        self.assertEqual(int(resampled[0]["eta_next_active"].sum().item()), 2)
        torch.testing.assert_close(cache[0]["eta_next_active"], active)


if __name__ == "__main__":
    unittest.main()
