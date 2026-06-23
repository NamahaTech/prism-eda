"""Top-level convenience API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism_eda.catalog.loaders import DataSource
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.dataset import Dataset
from prism_eda.events import EventCallback
from prism_eda.results import AnalysisResult


def load(
    source: DataSource | Dataset,
    *,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> Dataset:
    """Load one or more supported sources into a Prism EDA dataset."""
    if isinstance(source, Dataset):
        return source
    return Dataset.load(
        source,
        recursive=recursive,
        include=include,
        exclude=exclude,
        names=names,
        read_options=read_options,
    )


def profile(
    source: DataSource | Dataset,
    *,
    context: AnalysisContext | Mapping[str, Any] | None = None,
    config: AnalysisConfig | None = None,
    callbacks: Sequence[EventCallback] = (),
    mode: AnalysisMode | str = AnalysisMode.STANDARD,
    sampling: str = "auto",
    random_seed: int = 42,
    allow_insufficient_evidence: bool = False,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Load and profile a source in one call."""
    dataset = load(
        source,
        recursive=recursive,
        include=include,
        exclude=exclude,
        names=names,
        read_options=read_options,
    )
    return dataset.profile(
        context=context,
        config=config,
        callbacks=callbacks,
        mode=mode,
        sampling=sampling,
        random_seed=random_seed,
        allow_insufficient_evidence=allow_insufficient_evidence,
    )


def minimal_eda(source: DataSource | Dataset, **kwargs: Any) -> AnalysisResult:
    """Alias for :func:`profile`, retained for discoverability."""
    return profile(source, **kwargs)


def anomaly_detection(
    source: DataSource | Dataset,
    *,
    context: AnalysisContext | Mapping[str, Any] | None = None,
    config: AnalysisConfig | None = None,
    callbacks: Sequence[EventCallback] = (),
    mode: AnalysisMode | str = AnalysisMode.STANDARD,
    sampling: str = "auto",
    random_seed: int = 42,
    allow_insufficient_evidence: bool = False,
    table: str | None = None,
    target: str | None = None,
    expected_contamination: float | None = None,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Load sources and run deterministic anomaly-detection diagnostics."""
    dataset = load(
        source,
        recursive=recursive,
        include=include,
        exclude=exclude,
        names=names,
        read_options=read_options,
    )
    return dataset.anomaly_detection(
        context=context,
        config=config,
        callbacks=callbacks,
        mode=mode,
        sampling=sampling,
        random_seed=random_seed,
        allow_insufficient_evidence=allow_insufficient_evidence,
        table=table,
        target=target,
        expected_contamination=expected_contamination,
    )


def classification(
    source: DataSource | Dataset,
    target: str | None = None,
    *,
    context: AnalysisContext | Mapping[str, Any] | None = None,
    config: AnalysisConfig | None = None,
    callbacks: Sequence[EventCallback] = (),
    mode: AnalysisMode | str = AnalysisMode.STANDARD,
    sampling: str = "auto",
    random_seed: int = 42,
    allow_insufficient_evidence: bool = False,
    table: str | None = None,
    max_categories: int = 50,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Load sources and run deterministic classification diagnostics."""
    dataset = load(
        source,
        recursive=recursive,
        include=include,
        exclude=exclude,
        names=names,
        read_options=read_options,
    )
    return dataset.classification(
        target=target,
        context=context,
        config=config,
        callbacks=callbacks,
        mode=mode,
        sampling=sampling,
        random_seed=random_seed,
        allow_insufficient_evidence=allow_insufficient_evidence,
        table=table,
        max_categories=max_categories,
    )


def discover_schema(
    source: DataSource | Dataset,
    *,
    context: AnalysisContext | Mapping[str, Any] | None = None,
    config: AnalysisConfig | None = None,
    callbacks: Sequence[EventCallback] = (),
    mode: AnalysisMode | str = AnalysisMode.STANDARD,
    sampling: str = "auto",
    random_seed: int = 42,
    allow_insufficient_evidence: bool = False,
    max_key_columns: int | None = None,
    min_key_uniqueness: float = 0.98,
    min_key_completeness: float = 0.98,
    min_relationship_inclusion: float = 0.9,
    min_relationship_confidence: float = 0.72,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Load sources and discover candidate keys and table relationships."""
    dataset = load(
        source,
        recursive=recursive,
        include=include,
        exclude=exclude,
        names=names,
        read_options=read_options,
    )
    return dataset.discover_schema(
        context=context,
        config=config,
        callbacks=callbacks,
        mode=mode,
        sampling=sampling,
        random_seed=random_seed,
        allow_insufficient_evidence=allow_insufficient_evidence,
        max_key_columns=max_key_columns,
        min_key_uniqueness=min_key_uniqueness,
        min_key_completeness=min_key_completeness,
        min_relationship_inclusion=min_relationship_inclusion,
        min_relationship_confidence=min_relationship_confidence,
    )


__all__ = [
    "anomaly_detection",
    "classification",
    "discover_schema",
    "load",
    "minimal_eda",
    "profile",
]
