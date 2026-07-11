import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
