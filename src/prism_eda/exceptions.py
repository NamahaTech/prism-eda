"""Public exception hierarchy for Prism EDA."""


class PrismEDAError(Exception):
    """Base exception for all Prism EDA errors."""


class DataLoadError(PrismEDAError):
    """Raised when a dataset source cannot be loaded."""


class UnsupportedSourceError(DataLoadError):
    """Raised when a source type or file format is unsupported."""


class AnalysisError(PrismEDAError):
    """Raised when a foundational analysis stage cannot complete."""
