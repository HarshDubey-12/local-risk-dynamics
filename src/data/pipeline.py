from pathlib import Path
from dataclasses import dataclass
import hashlib 
import numpy as np
import pandas as pd 
from .loader import load_fama_french

@dataclass
class Dataset:
    """Immutable dataset container with metadata."""
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: list
    split_metadata: dict # tracks train/test split info.
    source_hash: str # Has of raw data for reproducibility.

    def __post_init__(self):
        assert len(self) > 0
        assert self.X.shape[0] == len(self.y) == len(self.dates)

    def __len__(self):
        return len(self.y)
    
    def describe(self) -> dict:
        return{
            "n_samples": len(self),
            "n_features": self.X.shape[1],
            "features": self.feature_names,
            "date_range": f"{self.dates[0]}to{self.dates[-1]}",
            "source_hash": self.source_hash,
        }


class DataPipeline:
    """Unified data handling with caching and versioning."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)

    def load_fama_french(self, path: str | Path) -> Dataset:
        """Load with hash-based caching."""
        path = Path(path)
        source_hash = self._get_file_hash(path)
        
        # Use the canonical loader to produce consistent features/targets.
        X, y, dates = load_fama_french(path)
        feature_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        dataset = Dataset(
            X=X,
            y=y,
            dates=dates,
            feature_names=feature_names,
            split_metadata={"raw_source": str(path)},
            source_hash=source_hash,
        )
        return dataset 
    
    def _get_file_hash(self, path: Path) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
