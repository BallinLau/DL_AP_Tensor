import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.run_multi_episode_job import configure_hyperparams, parse_args
from training.episode import Episode


class StageBatchSizeTest(unittest.TestCase):
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

    def test_pv_eta_resampling_uses_parent_eta_not_child_eta(self):
        episode = Episode.__new__(Episode)
        episode.hyperparams = SimpleNamespace(
            pv_eta_resample_enabled=True,
            pv_eta_resample_active_share=0.5,
            max_firm_train_units=0,
        )
        parent = torch.zeros(4, 8)
        parent[:, 2:3] = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        child0 = torch.zeros_like(parent)
        child1 = torch.zeros_like(parent)
        child0[:, 2:3] = 1.0
        child1[:, 2:3] = 1.0

        batches = episode._build_batches_from_parent_children(
            parent=parent,
            children=[child0, child1],
            batch_size=4,
            eta_resample=True,
        )

        selected = batches[0]["parent_source_index"]
        selected_parent_eta = parent[selected, 2]
        self.assertEqual(int((selected_parent_eta > 0.5).sum().item()), 2)


if __name__ == "__main__":
    unittest.main()
