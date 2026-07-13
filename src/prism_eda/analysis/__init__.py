"""Deterministic analysis recipes."""

from prism_eda.analysis.anomaly import anomaly_detection_dataset
from prism_eda.analysis.classification import classification_dataset
from prism_eda.analysis.image_profile import profile_image_dataset
from prism_eda.analysis.profile import profile_dataset
from prism_eda.analysis.schema_discovery import discover_schema_dataset

__all__ = [
    "anomaly_detection_dataset",
    "classification_dataset",
    "discover_schema_dataset",
    "profile_image_dataset",
    "profile_dataset",
]
