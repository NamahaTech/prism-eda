"""Deterministic tools the investigator may call.

The model can *only* call a tool registered here — never arbitrary code, never
raw rows. Each tool wraps an existing deterministic recipe (or a catalog query),
returns a compact, privacy-filtered summary for the model to reason over, and
exposes the evidence it produced so the orchestrator can validate that every
reported finding actually cites real evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prism_eda.artifacts import Artifact
from prism_eda.assisted_analysis.providers.base import ToolSpec
from prism_eda.dataset import Dataset
from prism_eda.evidence.models import Evidence, Finding
from prism_eda.privacy.models import PrivacyPolicy


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Result of one tool call: a model-facing summary plus citable evidence."""

    summary: dict[str, Any]
    evidence: tuple[Evidence, ...] = ()
    findings: tuple[Finding, ...] = ()
    artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    run: Callable[[Dataset, PrivacyPolicy, dict[str, Any]], ToolOutput]


def _finding_payload(findings: tuple[Finding, ...]) -> list[dict[str, Any]]:
    """Serialize findings into the compact, safe shape sent to the model."""
    return [
        {
            "id": finding.id,
            "title": finding.title,
            "summary": finding.summary,
            "severity": finding.severity,
            "confidence": finding.confidence,
            "evidence_ids": list(finding.evidence_ids),
            "recommendation": finding.recommendation,
        }
        for finding in findings
    ]


def _recipe_summary(result: Any) -> dict[str, Any]:
    return {
        "status": str(result.status),
        "summary": result.summary,
        "findings": _finding_payload(result.findings),
    }


def dataset_overview(dataset: Dataset, privacy: PrivacyPolicy) -> str:
    """Build a privacy-safe, human-readable description of the dataset.

    Column names are filtered through the privacy policy (aliased/redacted/omitted
    per column action); no cell values are included.
    """
    catalog = dataset.catalog()
    lines = [
        f"{catalog.table_count} table(s), {catalog.row_count} rows, "
        f"{catalog.column_count} columns total."
    ]
    for table in catalog.tables:
        safe_columns = []
        for column in table.columns:
            safe_name = privacy.safe_column_name(column.name)
            if safe_name is None:
                continue
            safe_columns.append(f"{safe_name}:{column.semantic_type}")
        lines.append(
            f"- {table.name} ({table.row_count} rows): " + ", ".join(safe_columns)
        )
    return "\n".join(lines)


# --- tool implementations -------------------------------------------------


def _list_tables(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    catalog = dataset.catalog()
    tables = []
    for table in catalog.tables:
        names = [
            safe
            for column in table.columns
            if (safe := privacy.safe_column_name(column.name)) is not None
        ]
        tables.append({"name": table.name, "rows": table.row_count, "columns": names})
    return ToolOutput(summary={"tables": tables})


def _describe_table(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    name = args.get("table")
    if not name:
        raise ValueError("describe_table requires a 'table' argument.")
    table = dataset.catalog().table(str(name))
    columns = []
    for column in table.columns:
        described = privacy.describe_column(
            column.name,
            {
                "name": column.name,
                "physical_type": column.physical_type,
                "semantic_type": column.semantic_type,
                "missing_rate": round(column.missing_rate, 4),
                "unique_count": column.unique_count,
            },
        )
        if described is not None:
            columns.append(described)
    return ToolOutput(
        summary={"table": table.name, "rows": table.row_count, "columns": columns}
    )


def _profile(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    result = dataset.profile()
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _discover_schema(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    result = dataset.discover_schema()
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _detect_anomalies(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    result = dataset.anomaly_detection(
        table=args.get("table"),
        target=args.get("target"),
        expected_contamination=args.get("expected_contamination"),
    )
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _assess_classification(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    target = args.get("target")
    if not target:
        raise ValueError("assess_classification requires a 'target' argument.")
    result = dataset.classification(
        target=str(target),
        table=args.get("table"),
        max_categories=int(args.get("max_categories", 50)),
    )
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _assess_regression(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    target = args.get("target")
    if not target:
        raise ValueError("assess_regression requires a 'target' argument.")
    result = dataset.regression(
        target=str(target),
        table=args.get("table"),
        max_categories=int(args.get("max_categories", 50)),
    )
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _analyze_time_series(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    value = args.get("value")
    if not value:
        raise ValueError("analyze_time_series requires a 'value' argument.")
    horizon = args.get("horizon")
    result = dataset.time_series(
        value=str(value),
        timestamp=args.get("timestamp"),
        entity_id=args.get("entity_id"),
        horizon=int(horizon) if horizon is not None else None,
        table=args.get("table"),
    )
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


def _assess_clustering(
    dataset: Dataset, privacy: PrivacyPolicy, args: dict[str, Any]
) -> ToolOutput:
    features = args.get("features")
    result = dataset.clustering(
        features=list(features) if features else None,
        table=args.get("table"),
    )
    return ToolOutput(
        summary=_recipe_summary(result),
        evidence=result.evidence,
        findings=result.findings,
        artifacts=result.artifacts,
    )


_TABLE_ARG = {"type": "string", "description": "Table name to analyze."}

_TOOLS: tuple[Tool, ...] = (
    Tool(
        ToolSpec(
            name="list_tables",
            description="List the tables in the dataset with row counts and "
            "column names. Use this first to orient yourself.",
            parameters={"type": "object", "properties": {}},
        ),
        _list_tables,
    ),
    Tool(
        ToolSpec(
            name="describe_table",
            description="Describe one table's columns: types, semantic roles, and "
            "missingness. No cell values are returned.",
            parameters={
                "type": "object",
                "properties": {"table": _TABLE_ARG},
                "required": ["table"],
            },
        ),
        _describe_table,
    ),
    Tool(
        ToolSpec(
            name="profile_dataset",
            description="Run the baseline data-quality profile across all tables "
            "(duplicates, missingness, constants). Produces citable evidence.",
            parameters={"type": "object", "properties": {}},
        ),
        _profile,
    ),
    Tool(
        ToolSpec(
            name="discover_schema",
            description="Discover candidate primary/foreign keys and table "
            "relationships. Produces citable evidence.",
            parameters={"type": "object", "properties": {}},
        ),
        _discover_schema,
    ),
    Tool(
        ToolSpec(
            name="detect_anomalies",
            description="Run statistical anomaly-review diagnostics on a table. "
            "Returns review candidates (not confirmed anomalies). Citable.",
            parameters={
                "type": "object",
                "properties": {
                    "table": _TABLE_ARG,
                    "target": {
                        "type": "string",
                        "description": "Optional label column to summarize rare "
                        "values for.",
                    },
                    "expected_contamination": {
                        "type": "number",
                        "description": "Optional expected fraction of rows to "
                        "review, e.g. 0.02.",
                    },
                },
            },
        ),
        _detect_anomalies,
    ),
    Tool(
        ToolSpec(
            name="assess_classification",
            description="Assess whether a table is ready to train a classifier for "
            "a target column: leakage, imbalance, association, separability. "
            "Citable.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The label column to assess.",
                    },
                    "table": _TABLE_ARG,
                    "max_categories": {
                        "type": "integer",
                        "description": "Cap on distinct categories per feature "
                        "(default 50).",
                    },
                },
                "required": ["target"],
            },
        ),
        _assess_classification,
    ),
    Tool(
        ToolSpec(
            name="assess_regression",
            description="Assess whether a table is ready to fit a regression for a "
            "numeric target: leakage, censoring, redundancy, probe fit, residual "
            "bias, and influential rows. Use this instead of assess_classification "
            "when the target is a continuous quantity. Citable.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The numeric target column to assess.",
                    },
                    "table": _TABLE_ARG,
                    "max_categories": {
                        "type": "integer",
                        "description": "Cap on distinct categories per feature "
                        "(default 50).",
                    },
                },
                "required": ["target"],
            },
        ),
        _assess_regression,
    ),
    Tool(
        ToolSpec(
            name="analyze_time_series",
            description="Analyze a numeric column over time: frequency, gaps, "
            "duplicate timestamps, trend, seasonality, stationarity, change "
            "points, and whether there is enough history to forecast. Use this "
            "when the question involves a time axis. Citable.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The numeric column measured over time.",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "The time column. Inferred when the table "
                        "has exactly one datetime column.",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Optional column identifying separate "
                        "series in a panel.",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Optional forecast horizon in periods, "
                        "used to judge whether the history is long enough.",
                    },
                    "table": _TABLE_ARG,
                },
                "required": ["value"],
            },
        ),
        _analyze_time_series,
    ),
    Tool(
        ToolSpec(
            name="assess_clustering",
            description="Assess whether a table has cluster structure at all, "
            "and describe candidate segments when it does: scale and redundancy "
            "problems, distance concentration, cluster tendency, how many groups "
            "reproduce on resampled rows, and what distinguishes each. Returns "
            "'no stable structure' as a real answer. Citable.",
            parameters={
                "type": "object",
                "properties": {
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit feature columns. When "
                        "omitted, usable numeric columns are chosen "
                        "automatically.",
                    },
                    "table": _TABLE_ARG,
                },
            },
        ),
        _assess_clustering,
    ),
)


def build_tool_registry() -> dict[str, Tool]:
    """Return the name → :class:`Tool` registry available to an investigation."""
    return {tool.spec.name: tool for tool in _TOOLS}


def tool_specs(registry: dict[str, Tool]) -> tuple[ToolSpec, ...]:
    """Return the public specs (names/descriptions/args) shown to the model."""
    return tuple(tool.spec for tool in registry.values())
