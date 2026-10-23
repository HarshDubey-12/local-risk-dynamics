"""
Precompute the dashboard's default results.

Runs every model of the default standard and improved configs through the
shared walk-forward folds and writes them to app/data/default_run/, so the
dashboard opens instantly. The dashboard recomputes automatically if the data
file changes (its hash is stored in meta.json).

Run from anywhere (takes about a minute):
    python scripts/build_app_results.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from lrd import comparison  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "app" / "data" / "default_run"


def main() -> None:
    start = time.perf_counter()
    runs = comparison.compute_default_runs(progress=lambda key: print(f"  running {key}", flush=True))
    comparison.save_runs(runs, OUTPUT_DIR, {
        "standard_config": comparison.STANDARD_CONFIG,
        "improved_config": comparison.IMPROVED_CONFIG,
    })
    print(f"Saved {len(runs)} runs to {OUTPUT_DIR} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
