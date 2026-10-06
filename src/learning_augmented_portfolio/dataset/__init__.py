"""
Dataset generation, serialization, schema validation, and chronological partitioning
for learning-augmented portfolio optimization.
"""

from .schema import (
    DATASET_SCHEMA_VERSION,
    MANIFEST_COLUMNS,
    validate_array_shapes,
)
from .quality import (
    LabelQuality,
    LabelQualityConfig,
    classify_label_quality,
)
from .split import (
    DatasetSplit,
    DatasetSplitConfig,
    validate_chronological_integrity,
)
from .instance import (
    OptimizationLabels,
    ContinuousRelaxationRecord,
    OptimizationInstanceMetadata,
    OptimizationInstance,
    compute_config_hash,
)
from .manifest import DatasetManifest
from .serialization import (
    save_instance,
    load_instance,
    rebuild_and_solve_miqp,
)
from .generator import (
    DatasetGeneratorConfig,
    HistoricalDatasetGenerator,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "MANIFEST_COLUMNS",
    "validate_array_shapes",
    "LabelQuality",
    "LabelQualityConfig",
    "classify_label_quality",
    "DatasetSplit",
    "DatasetSplitConfig",
    "validate_chronological_integrity",
    "OptimizationLabels",
    "ContinuousRelaxationRecord",
    "OptimizationInstanceMetadata",
    "OptimizationInstance",
    "compute_config_hash",
    "DatasetManifest",
    "save_instance",
    "load_instance",
    "rebuild_and_solve_miqp",
    "DatasetGeneratorConfig",
    "HistoricalDatasetGenerator",
]
