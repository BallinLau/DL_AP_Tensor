#!/usr/bin/env python3
"""Generate cumulative entry/macro progress plots from saved episode data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.entry_macro_progress import generate_entry_macro_progress_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-episode", type=int, default=None)
    parser.add_argument("--growth-start-time", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = generate_entry_macro_progress_report(
        args.run_root,
        args.output_dir,
        latest_episode=args.latest_episode,
        growth_start_time=args.growth_start_time,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
