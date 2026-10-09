#!/usr/bin/env python3
"""Audit whether an exact historical Episode-2 pre-PV bank is recoverable.

The historical GRID run did not persist the final mixed train/validation batch
objects or the full RNG/split state needed to reconstruct them.  This command
therefore accepts an existing formal capture, but never manufactures an
approximation from post-PV data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.pq_fixed_point import validate_frozen_batch_bank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    captured = (
        run_root
        / "diagnostics"
        / "frozen_pq_batch_banks"
        / "ep2_pre_pv_frozen_batch_bank.pt"
    )
    audit = {
        "run_root": str(run_root),
        "formal_capture_path": str(captured),
        "pre_pv_checkpoint_available": False,
        "full_rng_snapshot_available": False,
        "post_sdf_accepted_state_available": False,
        "mixture_rng_snapshot_available": False,
        "validation_split_rng_snapshot_available": False,
    }
    if captured.is_file():
        payload = torch.load(captured, map_location="cpu")
        metadata = validate_frozen_batch_bank(payload)
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(captured, args.output.expanduser().resolve())
        print(json.dumps({"status": "exact_capture_reused", **audit, **metadata}, indent=2))
        return

    audit["status"] = "historical_exact_recovery_not_possible"
    audit["reason"] = (
        "the historical run lacks the exact post-SDF/pre-PV mixed batches, "
        "their final validation split, and the RNG snapshots needed to replay them"
    )
    print(json.dumps(audit, indent=2))
    raise SystemExit("historical_exact_recovery_not_possible")


if __name__ == "__main__":
    main()
