"""Dataset session object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd

from prism_eda.analysis.anomaly import anomaly_detection_dataset
from prism_eda.analysis.classification import classification_dataset
from prism_eda.analysis.profile import profile_dataset
from prism_eda.analysis.schema_discovery import discover_schema_dataset
from prism_eda.catalog.loaders import DataSource, load_tables
from prism_eda.catalog.models import DatasetCatalog, SourceInfo
from prism_eda.catalog.profiling import build_catalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import EventCallback
from prism_eda.exceptions import AnalysisError
from prism_eda.results import AnalysisResult


class Dataset:
    """A named collection of tables and its deterministic analysis state."""

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        sources: Mapping[str, SourceInfo],
    ) -> None:
        self._tables = dict(tables)
        self._sources = dict(sources)
        self._catalog: DatasetCatalog | None = None

    @classmethod
    def load(
        cls,
        source: DataSource,
        *,
        recursive: bool = False,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        names: Mapping[str, str] | None = None,
        read_options: Mapping[str, Any] | None = None,
    ) -> Dataset:
        tables, sources = load_tables(
            source,
            recursive=recursive,
            include=include,
            exclude=exclude,
            names=names,
            read_options=read_options,
        )
        return cls(tables, sources)

    @property
    def tables(self) -> Mapping[str, pd.DataFrame]:
        return MappingProxyType(self._tables)

    def table(self, name: str) -> pd.DataFrame:
        return self._tables[name]

    def catalog(self, *, refresh: bool = False) -> DatasetCatalog:
        if self._catalog is None or refresh:
            self._catalog = build_catalog(self._tables, self._sources)
        return self._catalog

    def analyze(
        self,
        goal: str = "profile",
        *,
        context: AnalysisContext | Mapping[str, Any] | None = None,
        config: AnalysisConfig | None = None,
        callbacks: Sequence[EventCallback] = (),
        **options: Any,
    ) -> AnalysisResult:
        normalized_goal = goal.strip().lower().replace("-", "_")
        if context is None:
            analysis_context = AnalysisContext(goal=normalized_goal)
        elif isinstance(context, AnalysisContext):
            analysis_context = context
        else:
            analysis_context = AnalysisContext(goal=normalized_goal, **dict(context))
        analysis_config = config or AnalysisConfig(
            mode=options.pop("mode", AnalysisMode.STANDARD),
            sampling=options.pop("sampling", "auto"),
            random_seed=options.pop("random_seed", 42),
            allow_insufficient_evidence=options.pop(
                "allow_insufficient_evidence", False
            ),
        )
        if normalized_goal in {"profile", "minimal_eda"}:
            if options:
                unknown = ", ".join(sorted(options))
                raise TypeError(f"Unexpected analysis options: {unknown}")
            try:
                catalog = self.catalog(refresh=True)
            except Exception as error:
                raise AnalysisError(f"Catalog generation failed: {error}") from error
            return profile_dataset(
                catalog,
                context=analysis_context,
                config=analysis_config,
                callbacks=tuple(callbacks),
            )
        if normalized_goal in {"schema_discovery", "discover_schema"}:
            max_key_columns = options.pop("max_key_columns", None)
            min_key_uniqueness = options.pop("min_key_uniqueness", 0.98)
            min_key_completeness = options.pop("min_key_completeness", 0.98)
            min_relationship_inclusion = options.pop("min_relationship_inclusion", 0.9)
            min_relationship_confidence = options.pop(
                "min_relationship_confidence", 0.72
            )
            if options:
                unknown = ", ".join(sorted(options))
                raise TypeError(f"Unexpected analysis options: {unknown}")
            try:
                catalog = self.catalog(refresh=True)
            except Exception as error:
                raise AnalysisError(f"Catalog generation failed: {error}") from error
            return discover_schema_dataset(
                self._tables,
                catalog,
                context=analysis_context,
                config=analysis_config,
                max_key_columns=max_key_columns,
                min_key_uniqueness=min_key_uniqueness,
                min_key_completeness=min_key_completeness,
                min_relationship_inclusion=min_relationship_inclusion,
                min_relationship_confidence=min_relationship_confidence,
                callbacks=tuple(callbacks),
            )
        if normalized_goal in {"anomaly_detection", "anomaly", "outlier_detection"}:
            table = options.pop("table", None)
            target = options.pop("target", None) or analysis_context.target
            if options:
                unknown = ", ".join(sorted(options))
                raise TypeError(f"Unexpected analysis options: {unknown}")
            try:
                catalog = self.catalog(refresh=True)
            except Exception as error:
                raise AnalysisError(f"Catalog generation failed: {error}") from error
            return anomaly_detection_dataset(
                self._tables,
                catalog,
                context=analysis_context,
                config=analysis_config,
                table=table,
                target=target,
                callbacks=tuple(callbacks),
            )
        if normalized_goal in {"classification", "classify"}:
            table = options.pop("table", None)
            target = options.pop("target", None) or analysis_context.target
            max_categories = options.pop("max_categories", 50)
            if options:
                unknown = ", ".join(sorted(options))
                raise TypeError(f"Unexpected analysis options: {unknown}")
            try:
                catalog = self.catalog(refresh=True)
            except Exception as error:
                raise AnalysisError(f"Catalog generation failed: {error}") from error
            return classification_dataset(
                self._tables,
                catalog,
                context=analysis_context,
                config=analysis_config,
                target=target,
                table=table,
                max_categories=max_categories,
                callbacks=tuple(callbacks),
            )
        raise NotImplementedError(
            f"Goal {goal!r} is not implemented yet. Prism EDA 0.1 currently "
            "supports 'profile', 'schema_discovery', 'anomaly_detection', and "
            "'classification'."
        )

    def profile(
        self,
        *,
        context: AnalysisContext | Mapping[str, Any] | None = None,
        config: AnalysisConfig | None = None,
        callbacks: Sequence[EventCallback] = (),
        mode: AnalysisMode | str = AnalysisMode.STANDARD,
        sampling: str = "auto",
        random_seed: int = 42,
        allow_insufficient_evidence: bool = False,
    ) -> AnalysisResult:
        if config is None:
            config = AnalysisConfig(
                mode=mode,
                sampling=sampling,  # type: ignore[arg-type]
                random_seed=random_seed,
                allow_insufficient_evidence=allow_insufficient_evidence,
            )
        return self.analyze(
            "profile",
            context=context,
            config=config,
            callbacks=callbacks,
        )

    def discover_schema(
        self,
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
    ) -> AnalysisResult:
        """Discover candidate primary, composite, and foreign-key relationships."""
        if config is None:
            config = AnalysisConfig(
                mode=mode,
                sampling=sampling,  # type: ignore[arg-type]
                random_seed=random_seed,
                allow_insufficient_evidence=allow_insufficient_evidence,
            )
        return self.analyze(
            "schema_discovery",
            context=context,
            config=config,
            callbacks=callbacks,
            max_key_columns=max_key_columns,
            min_key_uniqueness=min_key_uniqueness,
            min_key_completeness=min_key_completeness,
            min_relationship_inclusion=min_relationship_inclusion,
            min_relationship_confidence=min_relationship_confidence,
        )

    def anomaly_detection(
        self,
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
    ) -> AnalysisResult:
        """Run deterministic anomaly-detection diagnostics."""
        if config is None:
            config = AnalysisConfig(
                mode=mode,
                sampling=sampling,  # type: ignore[arg-type]
                random_seed=random_seed,
                allow_insufficient_evidence=allow_insufficient_evidence,
            )
        return self.analyze(
            "anomaly_detection",
            context=context,
            config=config,
            callbacks=callbacks,
            table=table,
            target=target,
        )

    def classification(
        self,
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
    ) -> AnalysisResult:
        """Run deterministic classification diagnostics for a target column."""
        if config is None:
            config = AnalysisConfig(
                mode=mode,
                sampling=sampling,  # type: ignore[arg-type]
                random_seed=random_seed,
                allow_insufficient_evidence=allow_insufficient_evidence,
            )
        return self.analyze(
            "classification",
            context=context,
            config=config,
            callbacks=callbacks,
            table=table,
            target=target,
            max_categories=max_categories,
        )
