"""
Manifest management for historical optimization instance datasets.
Provides tabular indexation, schema validation, and CSV serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from .instance import OptimizationInstance
from .schema import MANIFEST_COLUMNS, DATASET_SCHEMA_VERSION


class DatasetManifest:
    """
    Catalog of serialized optimization instances.
    Enables rapid filtering, chronological partitioning, and quality auditing.
    """

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None) -> None:
        self.records: List[Dict[str, Any]] = records or []

    def add_instance(
        self,
        instance: OptimizationInstance,
        filename: str,
    ) -> None:
        """Extract scalar metadata from an OptimizationInstance and append to manifest."""
        meta = instance.metadata
        row = {
            "instance_id": meta.instance_id,
            "calendar_date": meta.calendar_date,
            "time_index": meta.time_index,
            "split": meta.split,
            "filename": filename,
            "universe_id": meta.universe_id,
            "n_assets": meta.n_assets,
            "horizon": meta.horizon,
            "solver_status": meta.solver_status,
            "label_quality": meta.label_quality,
            "is_trainable_label": bool(meta.is_trainable_label),
            "objective": meta.objective,
            "best_bound": meta.best_bound,
            "mip_gap": meta.mip_gap,
            "runtime": meta.runtime,
            "work": meta.work,
            "node_count": meta.node_count,
            "sol_count": meta.sol_count,
            "used_fallback": bool(meta.used_fallback),
            "config_hash": meta.config_hash,
            "dataset_version": meta.dataset_version,
        }
        self.records.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert manifest records to pandas DataFrame with canonical column layout."""
        if not self.records:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        df = pd.DataFrame(self.records)
        # Ensure all columns present in specified order
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[MANIFEST_COLUMNS]

    def save_csv(self, filepath: Union[Path, str]) -> Path:
        """Save manifest to a CSV file."""
        self.validate()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    @classmethod
    def from_csv(cls, filepath: Union[Path, str]) -> DatasetManifest:
        """Load manifest from an existing CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {path}")
        df = pd.read_csv(path)
        records = df.to_dict(orient="records")
        manifest = cls(records)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate integrity, uniqueness, and completeness of manifest records."""
        if not self.records:
            return

        instance_ids = [r["instance_id"] for r in self.records]
        if len(instance_ids) != len(set(instance_ids)):
            duplicates = [x for x in instance_ids if instance_ids.count(x) > 1]
            raise ValueError(f"Duplicate instance_ids detected in manifest: {set(duplicates)}")

        valid_splits = {"train", "validation", "test"}
        for r in self.records:
            if r["split"] not in valid_splits:
                raise ValueError(f"Invalid split '{r['split']}' in instance {r['instance_id']}")

    def __len__(self) -> int:
        return len(self.records)
