import unittest
from pathlib import Path
import sys
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
