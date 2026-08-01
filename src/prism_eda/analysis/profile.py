"""Baseline dataset profile implemented as structured evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import pandas as pd

from prism_eda.analysis._limits import ProfileLimits
from prism_eda.analysis.associations import build_observations
from prism_eda.analysis.distributions import build_distribution_evidence
from prism_eda.analysis.quality_checks import build_quality_findings
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.config import (
    AnalysisConfig,
    AnalysisContext,
    AnalysisMode,
    DetailLevel,
)
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
    split_findings,
)
from prism_eda.results import (
    AnalysisFailure,
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

_T = TypeVar("_T")


def _table_evidence(catalog: DatasetCatalog) -> list[Evidence]:
    evidence: list[Evidence] = [
        Evidence.create(
            kind="dataset_shape",
            scope=EvidenceScope(),
            value={
                "tables": catalog.table_count,
                "rows": catalog.row_count,
                "columns": catalog.column_count,
            },
            method="exact_catalog_aggregation",
            description="Overall dataset shape across all loaded tables.",
            metadata={"dataset_fingerprint": catalog.fingerprint},
        )
    ]
    for table in catalog.tables:
        evidence.append(
            Evidence.create(
                kind="table_quality_summary",
                scope=EvidenceScope(table=table.name),
                value={
                    "rows": table.row_count,
                    "columns": table.column_count,
                    "memory_bytes": table.memory_bytes,
                    "duplicate_rows": table.duplicate_row_count,
                },
                method="exact_table_profile",
                description=f"Shape, memory, and duplicate summary for {table.name}.",
                metadata={"table_fingerprint": table.fingerprint},
            )
        )
        for column in table.columns:
            evidence.append(
                Evidence.create(
                    kind="column_profile",
                    scope=EvidenceScope(table=table.name, columns=(column.name,)),
                    value={
                        "physical_type": column.physical_type,
                        "semantic_type": column.semantic_type,
                        "roles": column.roles,
                        "missing_count": column.missing_count,
                        "missing_rate": column.missing_rate,
                        "unique_count": column.unique_count,
                        "unique_rate": column.unique_rate,
                        "statistics": column.statistics,
                        "top_values": column.top_values,
                    },
                    method="exact_column_profile",
                    description=f"Baseline profile for {table.name}.{column.name}.",
                )
            )
    return evidence


def _findings_and_plan(
    catalog: DatasetCatalog, evidence: list[Evidence]
) -> tuple[list[Finding], list[TransformationStep]]:
    """Promote the catalog's own numbers into duplicate/missing/constant issues."""
    by_scope = {
        (item.scope.table, item.scope.columns): item
        for item in evidence
        if item.kind == "column_profile"
    }
    table_evidence = {
        item.scope.table: item
        for item in evidence
        if item.kind == "table_quality_summary"
    }
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    for table in catalog.tables:
        duplicate_evidence = table_evidence[table.name]
        if table.duplicate_row_count:
            rate = table.duplicate_row_count / table.row_count if table.row_count else 0
            if table.duplicate_row_count == 1:
                duplicate_summary = f"1 row ({rate:.1%}) is an exact duplicate."
            else:
                duplicate_summary = (
                    f"{table.duplicate_row_count:,} rows ({rate:.1%}) "
                    "are exact duplicates."
                )
            findings.append(
                Finding.create(
                    title=f"Duplicate rows in {table.name}",
                    summary=duplicate_summary,
                    severity="high" if rate >= 0.1 else "medium",
                    confidence=1.0,
                    evidence_ids=(duplicate_evidence.id,),
                    recommendation=(
                        "Confirm row granularity before removing duplicates."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_duplicate_rows",
                    table=table.name,
                    columns=(),
                    parameters={"duplicate_count": table.duplicate_row_count},
                    rationale=(
                        "Exact duplicate rows may violate the intended granularity."
                    ),
                    evidence_ids=(duplicate_evidence.id,),
                    risk="high",
                )
            )

        for column in table.columns:
            item = by_scope[(table.name, (column.name,))]
            if column.missing_rate >= 0.2:
                value_noun = "value" if column.missing_count == 1 else "values"
                value_verb = "is" if column.missing_count == 1 else "are"
                findings.append(
                    Finding.create(
                        title=f"High missingness in {table.name}.{column.name}",
                        summary=(
                            f"{column.missing_count:,} {value_noun} "
                            f"({column.missing_rate:.1%}) {value_verb} missing."
                        ),
                        severity="high" if column.missing_rate >= 0.5 else "medium",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Investigate whether missingness is structural, erroneous, "
                            "or informative before choosing a fill strategy."
                        ),
                    )
                )
                steps.append(
                    TransformationStep(
                        operation="review_missing_values",
                        table=table.name,
                        columns=(column.name,),
                        parameters={"missing_rate": column.missing_rate},
                        rationale=(
                            "Missingness is high enough to affect downstream analysis."
                        ),
                        evidence_ids=(item.id,),
                    )
                )
            if column.unique_count == 1 and column.non_null_count:
                findings.append(
                    Finding.create(
                        title=f"Constant column {table.name}.{column.name}",
                        summary="All non-null values are identical.",
                        severity="low",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Review whether the column carries useful information."
                        ),
                    )
                )
                steps.append(
                    TransformationStep(
                        operation="review_constant_column",
                        table=table.name,
                        columns=(column.name,),
                        parameters={},
                        rationale=(
                            "Constant columns usually add no predictive information."
                        ),
                        evidence_ids=(item.id,),
                        risk="low",
                    )
                )
    return sort_findings(findings), steps


def _run_optional(
    stage: str,
    table: str,
    failures: list[AnalysisFailure],
    work: Callable[[], _T],
) -> _T | None:
    """Run an optional stage, recording a failure instead of aborting the run.

    Invariant 7: optional metric failures are recorded and analysis continues.
    The catalog-level profile is still true without the charts.
    """
    try:
        return work()
    except Exception as error:  # noqa: BLE001 - optional stage, recorded below
        failures.append(
            AnalysisFailure(
                stage=stage,
                message=f"{stage.replace('_', ' ').capitalize()} failed: {error}",
                recoverable=True,
                table=table,
            )
        )
        return None


def _counts_phrase(issues: list[Finding], observations: list[Finding]) -> str:
    """Say what was found, keeping defects and observations visibly separate."""
    if not issues and not observations:
        return "no data-quality issues"
    parts = []
    if issues:
        parts.append(f"{len(issues)} data-quality issue(s)")
    if observations:
        parts.append(f"{len(observations)} observation(s)")
    if not issues:
        return f"no data-quality issues and {parts[0]}"
    return " and ".join(parts)


def profile_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    detail: DetailLevel = "standard",
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Build a concise baseline profile from an exact dataset catalog."""
    emit(
        callbacks,
        Event(EventKind.RUN_STARTED, "Baseline profile started.", stage="profile"),
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_STARTED,
            "Creating structured evidence.",
            stage="evidence",
        ),
    )
    evidence = _table_evidence(catalog)
    limits = ProfileLimits.for_detail(detail)
    extra_findings: list[Finding] = []
    extra_steps: list[TransformationStep] = []
    optional_failures: list[AnalysisFailure] = []
    optional_warnings: list[AnalysisWarning] = []
    sampling_records: list[SamplingRecord] = []

    for table in catalog.tables:
        frame = tables.get(table.name)
        if frame is None:
            continue

        quality = _run_optional(
            "quality_checks",
            table.name,
            optional_failures,
            lambda: build_quality_findings(frame, table),
        )
        if quality is not None:
            quality_evidence, quality_findings, quality_steps = quality
            evidence.extend(quality_evidence)
            extra_findings.extend(quality_findings)
            extra_steps.extend(quality_steps)

        charts = _run_optional(
            "distributions",
            table.name,
            optional_failures,
            lambda: build_distribution_evidence(
                frame,
                table,
                chart_columns=limits.chart_columns,
                fit_rows=limits.fit_rows,
            ),
        )
        if charts is not None:
            chart_evidence, skipped = charts
            evidence.extend(chart_evidence)
            if skipped:
                optional_warnings.append(
                    AnalysisWarning(
                        code="chart_columns_capped",
                        message=(
                            f"{table.name} has more columns than the "
                            f"{limits.chart_columns}-column chart budget; "
                            f"{len(skipped)} column(s) are profiled without a "
                            "chart. Pass detail='full' to chart them all."
                        ),
                        table=table.name,
                    )
                )

        column_evidence = {
            item.scope.columns[0]: item.id
            for item in evidence
            if item.kind == "column_profile"
            and item.scope.table == table.name
            and item.scope.columns
        }
        observed = _run_optional(
            "associations",
            table.name,
            optional_failures,
            lambda: build_observations(
                frame,
                table,
                column_evidence,
                limits=limits,
                seed=config.random_seed,
            ),
        )
        if observed is not None:
            found_evidence, found, found_warnings, found_sampling = observed
            evidence.extend(found_evidence)
            extra_findings.extend(found)
            optional_warnings.extend(found_warnings)
            sampling_records.extend(found_sampling)

    for item in evidence:
        emit(
            callbacks,
            Event(
                EventKind.EVIDENCE_CREATED,
                item.description,
                stage="evidence",
                data={"evidence_id": item.id, "kind": item.kind},
            ),
        )
    catalog_findings, catalog_steps = _findings_and_plan(catalog, evidence)
    findings = sort_findings([*catalog_findings, *extra_findings])
    steps = [*catalog_steps, *extra_steps]
    issues, observations = split_findings(findings)
    warnings: list[AnalysisWarning] = list(optional_warnings)
    failures: list[AnalysisFailure] = list(optional_failures)
    for table in catalog.tables:
        for warning in table.warnings:
            failures.append(
                AnalysisFailure(
                    stage="table_profile",
                    message=warning,
                    recoverable=True,
                    table=table.name,
                )
            )

    insufficient = catalog.row_count == 0 or catalog.column_count == 0
    if insufficient:
        warnings.append(
            AnalysisWarning(
                code="insufficient_rows",
                message="The dataset has no rows or columns to analyze.",
            )
        )
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "The catalog was created, but there is insufficient data to profile."
    elif failures:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = (
            f"Profiled {catalog.table_count} table(s) and found "
            f"{_counts_phrase(issues, observations)}; some optional metrics failed."
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = (
            f"Profiled {catalog.table_count} table(s), {catalog.row_count:,} rows, "
            f"and {catalog.column_count} columns; found "
            f"{_counts_phrase(issues, observations)}."
        )

    result = AnalysisResult(
        goal="profile",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        failures=tuple(failures),
        sampling=tuple(sampling_records),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "detail": detail,
            "issue_count": len(issues),
            "observation_count": len(observations),
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_COMPLETED,
            "Structured evidence created.",
            stage="evidence",
            progress=1.0,
        ),
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="profile",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
