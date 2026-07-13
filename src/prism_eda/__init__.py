"""Task-aware exploratory data analysis for Python."""

from prism_eda.api import (
    anomaly_detection,
    classification,
    discover_schema,
    load,
    load_images,
    minimal_eda,
    profile,
    profile_images,
)
from prism_eda.artifacts import Artifact
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.dataset import Dataset
from prism_eda.events import Event, EventKind
from prism_eda.image_dataset import ImageDataset
from prism_eda.results import AnalysisResult, AnalysisStatus

__all__ = [
    "AnalysisConfig",
    "AnalysisContext",
    "AnalysisMode",
    "AnalysisResult",
    "AnalysisStatus",
    "Artifact",
    "Dataset",
    "Event",
    "EventKind",
    "ImageDataset",
    "anomaly_detection",
    "classification",
    "discover_schema",
    "load",
    "load_images",
    "minimal_eda",
    "profile",
    "profile_images",
]

__version__ = "0.1.0"
