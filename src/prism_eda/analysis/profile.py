"""Baseline dataset profile implemented as structured evidence."""

from __future__ import annotations

from prism_eda.catalog.models import DatasetCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import Evidence, EvidenceScope, Finding
from prism_eda.results import (
    AnalysisFailure,
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep


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
    return findings, steps


def profile_dataset(
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
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
    findings, steps = _findings_and_plan(catalog, evidence)
    warnings: list[AnalysisWarning] = []
    failures: list[AnalysisFailure] = []
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
            f"{len(findings)} prioritized issue(s); some optional metrics failed."
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = (
            f"Profiled {catalog.table_count} table(s), {catalog.row_count:,} rows, "
            f"and {catalog.column_count} columns; found {len(findings)} prioritized "
            "issue(s)."
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
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
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
