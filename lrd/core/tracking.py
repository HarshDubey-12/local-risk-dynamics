from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess
from typing import Dict, Any
import pandas as pd

class ExperimentTracker:
    """Centralized experiment metadata tracking."""

    def __init__(self, log_dir: Path = Path("experiments/logs")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents = True, exist_ok = True)

    def log_experiment(
            self,
            config_path: str,
            results_df: pd.DataFrame,
            metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Save experiment with metadata."""

        metadata = metadata or {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        exp_dir = self.log_dir / timestamp
        exp_dir.mkdir(parents = True, exist_ok = False)

        # Save Config
        shutil.copy(config_path, exp_dir / "config.yaml")

        # Save results
        results_df.to_csv(exp_dir / "result.csv", index = False)

        # Save metadata
        meta = {
            "timestamp": timestamp,
            "config_file": str(config_path),
            "dataset_hash": metadata.get("dataset_hash"),
            "git_commit": self._get_git_commit(),
            **metadata,
        }
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent = 2, default = str)

        return str(exp_dir)

    @staticmethod
    def _get_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git","rev-parse","HEAD"],
                stderr = subprocess.DEVNULL,
            ).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
