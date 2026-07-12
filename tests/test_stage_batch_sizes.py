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

    def test_dataframe_pv_eta_resampling_uses_parent_eta_not_child_eta(self):
        episode = Episode.__new__(Episode)
        episode.device = torch.device("cpu")
        episode.hyperparams = SimpleNamespace(
            pv_eta_resample_enabled=True,
            pv_eta_resample_active_share=0.75,
            max_firm_train_units=0,
        )

        rows = []
        parent_eta = [1.0, 0.0, 0.0, 0.0]

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

            eta_child = 1.0 - eta_parent
            for branch in range(2):
                rows.append(
                    {
                        **common,
                        "t": 1,
                        "branch": branch,
                        "ETA": eta_child,
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

        selected_parent = torch.cat(
            [batch["parent"] for batch in batches],
            dim=0,
        )
        selected_parent_eta = selected_parent[:, 2]

        self.assertEqual(selected_parent.shape[0], 4)
        self.assertEqual(
            int((selected_parent_eta > 0.5).sum().item()),
            3,
        )


if __name__ == "__main__":
    unittest.main()
