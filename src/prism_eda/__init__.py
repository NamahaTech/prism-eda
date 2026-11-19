"""Task-aware exploratory data analysis for Python."""

from prism_eda.api import (
    anomaly_detection,
    classification,
    clustering,
    compare_datasets,
    discover_schema,
    explore,
    features,
    load,
    load_images,
    minimal_eda,
    profile,
    profile_images,
    regression,
    time_series,
)
from prism_eda.artifacts import Artifact
from prism_eda.branding import (
    Branding,
    get_default_branding,
    set_default_branding,
)
from prism_eda.comparison_results import ComparisonResult
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.dataset import Dataset
from prism_eda.events import Event, EventKind
from prism_eda.exploration import ColumnHint, ExploreConfig, Hints, MemoryCheckpoints
from prism_eda.features import FeaturePlan, FeatureSet
from prism_eda.features.results import FeaturePlanResult, FeatureRun
from prism_eda.image_dataset import ImageDataset
from prism_eda.results import AnalysisResult, AnalysisStatus
from prism_eda.sources import (
    ClickHouseSource,
    FrameSource,
    PostgresSource,
    SourceLimits,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisContext",
    "AnalysisMode",
    "AnalysisResult",
    "AnalysisStatus",
    "Artifact",
    "Branding",
    "ClickHouseSource",
    "ColumnHint",
    "ComparisonResult",
    "Dataset",
    "Event",
    "EventKind",
    "ExploreConfig",
    "FeaturePlan",
    "FeaturePlanResult",
    "FeatureRun",
    "FeatureSet",
    "FrameSource",
    "Hints",
    "ImageDataset",
    "MemoryCheckpoints",
    "PostgresSource",
    "SourceLimits",
    "anomaly_detection",
    "classification",
    "clustering",
    "compare_datasets",
    "discover_schema",
    "explore",
    "features",
    "get_default_branding",
    "load",
    "load_images",
    "minimal_eda",
    "profile",
    "profile_images",
    "regression",
    "set_default_branding",
    "time_series",
]

__version__ = "2.0.0"
