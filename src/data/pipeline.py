from pathlib import Path
from dataclasses import dataclass
import hashlib 
import numpy as np
import pandas as pd 

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
        cache_file = self.cache_dir / f"ff_cache_{source_hash}.npz"

        if cache_file.exists():
            return self._load_cache(cache_file)
        
        # Load, preprocess, cache
        df = pd.read_csv(path, skiprows = 3)
        # Extract components
        dates = pd.to_datetime(df.index)
        feature_names = df.columns.tolist()
        X = df.values
        y = df.iloc[:, -1].values  # Last column as target

        dataset = Dataset(X, y, dates, feature_names, {"raw_source": str(path)}, source_hash)
        return dataset 
    
    def _get_file_hash(self, path: Path) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]