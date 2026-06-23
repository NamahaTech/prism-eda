"""Goal-aware deterministic diagnostics for anomaly-detection EDA."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import Evidence, EvidenceScope, Finding
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

_ROW_BUDGETS = {
    AnalysisMode.QUICK: 25_000,
    AnalysisMode.STANDARD: 100_000,
    AnalysisMode.DEEP: 250_000,
}

_DEFAULT_REVIEW_RATE = 0.02
_MAX_REVIEW_ROWS = 25


def _row_budget(mode: AnalysisMode | str) -> int:
    return _ROW_BUDGETS[AnalysisMode(mode)]


def _sample_frame(
    frame: pd.DataFrame,
    *,
    table: str,
    config: AnalysisConfig,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> pd.DataFrame:
    budget = _row_budget(config.mode)
    if config.sampling == "disabled" or len(frame) <= budget:
        return frame
    sampled = frame.sample(n=budget, random_state=config.random_seed).sort_index()
    warnings.append(
        AnalysisWarning(
            code="sampled_anomaly_detection",
            message=(
                f"{table} has {len(frame):,} rows; anomaly diagnostics were run on "
                f"a deterministic {budget:,}-row sample."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="anomaly_detection",
            source_rows=len(frame),
            sampled_rows=budget,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="row_count_exceeds_mode_budget",
            limitations=(
                "Rare anomalies may be missed when they are absent from the sample.",
            ),
        )
    )
    return sampled


def _numeric_columns(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    exclude: set[str] | None = None,
    allow_identifiers: bool = True,
) -> list[str]:
    excluded = exclude or set()
    columns: list[str] = []
    catalog_by_name = {column.name: column for column in table.columns}
    for column in frame.columns:
        name = str(column)
        if name in excluded or name not in catalog_by_name:
            continue
        series = frame[column]
        profile = catalog_by_name[name]
        if not ptypes.is_numeric_dtype(series.dtype) or series.notna().sum() < 8:
            continue
        if not allow_identifiers and (
            "identifier_candidate" in profile.roles
            or name.lower() in {"id", "index"}
            or name.lower().endswith("_id")
        ):
            continue
        columns.append(name)
    return columns


def _categorical_columns(frame: pd.DataFrame, table: TableCatalog) -> list[str]:
    catalog_by_name = {column.name: column for column in table.columns}
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        profile = catalog_by_name.get(name)
        if profile is None:
            continue
        if profile.semantic_type in {"categorical", "boolean", "text"}:
            if profile.unique_count is None or profile.unique_count <= 100:
                columns.append(name)
    return columns


def _robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviations = (numeric - median).abs()
    mad = deviations.median()
    if pd.notna(mad) and mad > 0:
        return 0.6745 * (numeric - median) / mad
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        return (numeric - median) / (iqr / 1.349)
    std = numeric.std(ddof=0)
    if pd.notna(std) and std > 0:
        return (numeric - numeric.mean()) / std
    return pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")


def _top_index_values(scores: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    sorted_scores = scores.dropna().abs().sort_values(ascending=False).head(limit)
    return [
        {"row_index": str(index), "score": float(score)}
        for index, score in sorted_scores.items()
    ]


def _review_count(row_count: int, expected_contamination: float | None) -> int:
    if row_count <= 0:
        return 0
    rate = (
        expected_contamination
        if expected_contamination is not None
        else _DEFAULT_REVIEW_RATE
    )
    return max(1, min(_MAX_REVIEW_ROWS, math.ceil(row_count * rate)))


def _validate_expected_contamination(value: float | None) -> None:
    if value is None:
        return
    if not 0 < value < 0.5:
        raise ValueError(
            "expected_contamination must be greater than 0 and less than 0.5"
        )


def _numeric_model_matrix(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:12]
    if len(columns) < 2:
        return columns, pd.DataFrame(), pd.DataFrame()
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna()
    if len(usable) < 20:
        return columns, pd.DataFrame(), pd.DataFrame()
    z_frame = usable.apply(_robust_z)
    return columns, usable, z_frame


def _model_top_records(
    scores: pd.Series,
    z_frame: pd.DataFrame,
    *,
    count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, score in scores.sort_values(ascending=False).head(count).items():
        row_z = z_frame.loc[index].abs().sort_values(ascending=False).head(3)
        records.append(
            {
                "row_index": str(index),
                "score": float(score),
                "top_contributors": [
                    {"column": str(column), "abs_robust_z": float(value)}
                    for column, value in row_z.items()
                ],
            }
        )
    return records


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _category_value(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating) and bool(np.isnan(value)):
        return None
    return str(value)


def _univariate_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for column in _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=True
    ):
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null = series.dropna()
        if len(non_null) < 8:
            continue
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        robust_z = _robust_z(series)
        mask = ((series < lower) | (series > upper) | (robust_z.abs() >= 3.5)) & (
            series.notna()
        )
        count = int(mask.sum())
        evidence.append(
            Evidence.create(
                kind="anomaly_univariate_outlier",
                scope=EvidenceScope(table=table.name, columns=(column,)),
                value={
                    "evaluated_row_count": int(series.notna().sum()),
                    "candidate_count": count,
                    "candidate_rate": count / len(non_null) if len(non_null) else 0.0,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "max_abs_robust_z": float(robust_z.abs().max(skipna=True) or 0.0),
                    "examples": _top_index_values(robust_z[mask]),
                },
                method="iqr_or_modified_z_score_v1",
                description=f"Univariate tail candidates for {table.name}.{column}.",
                confidence=0.82 if count else 0.72,
                assumptions=(
                    "Tail candidates are statistical review targets, not confirmed "
                    "anomalies.",
                ),
            )
        )
    return evidence


def _multivariate_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> Evidence | None:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:12]
    if len(columns) < 2 or len(frame) < 8:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna()
    if len(usable) < 8:
        return None
    z_frame = usable.apply(_robust_z)
    scores = np.sqrt((z_frame**2).sum(axis=1))
    threshold = max(4.5, math.sqrt(len(columns)) * 3.0)
    candidate_scores = scores[scores >= threshold]
    top_records: list[dict[str, Any]] = []
    for index, score in candidate_scores.sort_values(ascending=False).head(10).items():
        row_z = z_frame.loc[index].abs().sort_values(ascending=False).head(3)
        top_records.append(
            {
                "row_index": str(index),
                "score": float(score),
                "top_contributors": [
                    {"column": str(column), "abs_robust_z": float(value)}
                    for column, value in row_z.items()
                ],
            }
        )
    return Evidence.create(
        kind="anomaly_multivariate_outlier",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "threshold": threshold,
            "candidate_count": int(len(candidate_scores)),
            "candidate_rate": len(candidate_scores) / len(usable),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
        },
        method="robust_scaled_euclidean_score_v1",
        description=f"Multivariate robust-score candidates for {table.name}.",
        confidence=0.74 if len(candidate_scores) else 0.62,
        assumptions=(
            "Features are robust-scaled independently; this is a lightweight "
            "diagnostic, not a fitted anomaly detector.",
        ),
    )


def _isolation_forest_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    random_seed: int,
    expected_contamination: float | None,
) -> Evidence | None:
    columns, usable, z_frame = _numeric_model_matrix(frame, table, target=target)
    if len(columns) < 2 or usable.empty:
        return None
    review_count = _review_count(len(usable), expected_contamination)
    contamination: float | str = (
        expected_contamination if expected_contamination is not None else "auto"
    )
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(z_frame)
    scores = pd.Series(-model.decision_function(z_frame), index=usable.index)
    top_records = _model_top_records(scores, z_frame, count=review_count)
    top_rows = {record["row_index"] for record in top_records}

    stability_sets: list[set[str]] = []
    for seed_offset in (1, 2):
        stability_model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_seed + seed_offset,
            n_jobs=1,
        )
        stability_model.fit(z_frame)
        stability_scores = pd.Series(
            -stability_model.decision_function(z_frame), index=usable.index
        )
        stability_sets.append(
            {
                str(index)
                for index in stability_scores.sort_values(ascending=False)
                .head(review_count)
                .index
            }
        )
    mean_jaccard = (
        sum(_jaccard(top_rows, item) for item in stability_sets) / len(stability_sets)
        if stability_sets
        else 1.0
    )
    return Evidence.create(
        kind="anomaly_isolation_forest",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "candidate_count": len(top_records),
            "candidate_rate": len(top_records) / len(usable),
            "expected_contamination": expected_contamination,
            "threshold_policy": (
                "expected_contamination"
                if expected_contamination is not None
                else "ranked_review_default"
            ),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
            "stability": {
                "seed_count": 3,
                "mean_top_set_jaccard": float(mean_jaccard),
            },
        },
        method="isolation_forest_ranked_review_v1",
        description=f"Isolation Forest ranked anomaly candidates for {table.name}.",
        confidence=0.78 if mean_jaccard >= 0.5 else 0.66,
        assumptions=(
            "Isolation Forest is a diagnostic review instrument; ranked candidates "
            "are not confirmed anomalies.",
            "Numeric features are robust-scaled and rows with missing modeled "
            "features are skipped.",
        ),
    )


def _local_density_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
    expected_contamination: float | None,
) -> Evidence | None:
    columns, usable, z_frame = _numeric_model_matrix(frame, table, target=target)
    if len(columns) < 2 or usable.empty:
        return None
    if len(usable) < 30 or len(columns) > 12:
        return None
    review_count = _review_count(len(usable), expected_contamination)
    neighbors = min(35, max(10, int(math.sqrt(len(usable)))))
    if neighbors >= len(usable):
        return None
    contamination: float | str = (
        expected_contamination if expected_contamination is not None else "auto"
    )
    model = LocalOutlierFactor(
        n_neighbors=neighbors,
        contamination=contamination,
        metric="minkowski",
        n_jobs=1,
    )
    model.fit_predict(z_frame)
    scores = pd.Series(-model.negative_outlier_factor_, index=usable.index)
    top_records = _model_top_records(scores, z_frame, count=review_count)
    return Evidence.create(
        kind="anomaly_local_density_outlier",
        scope=EvidenceScope(table=table.name, columns=tuple(columns)),
        value={
            "evaluated_row_count": int(len(usable)),
            "feature_count": len(columns),
            "candidate_count": len(top_records),
            "candidate_rate": len(top_records) / len(usable),
            "expected_contamination": expected_contamination,
            "n_neighbors": neighbors,
            "threshold_policy": (
                "expected_contamination"
                if expected_contamination is not None
                else "ranked_review_default"
            ),
            "max_score": float(scores.max() if len(scores) else 0.0),
            "top_records": top_records,
        },
        method="local_outlier_factor_ranked_review_v1",
        description=f"Local-density anomaly candidates for {table.name}.",
        confidence=0.72,
        assumptions=(
            "Local Outlier Factor is sensitive to feature scaling, sparse regions, "
            "and high dimensionality.",
            "Numeric features are robust-scaled and rows with missing modeled "
            "features are skipped.",
        ),
    )


def _candidate_rows(item: Evidence) -> list[str]:
    value = item.value
    if item.kind in {
        "anomaly_univariate_outlier",
        "anomaly_conditional_outlier",
    }:
        return [str(record["row_index"]) for record in value.get("examples", ())]
    if item.kind in {
        "anomaly_multivariate_outlier",
        "anomaly_isolation_forest",
        "anomaly_local_density_outlier",
    }:
        return [str(record["row_index"]) for record in value.get("top_records", ())]
    return []


def _agreement_evidence(evidence: Sequence[Evidence], *, table: str) -> Evidence | None:
    signal_rows: dict[str, set[str]] = {}
    for item in evidence:
        if item.scope.table != table:
            continue
        rows = set(_candidate_rows(item))
        if rows:
            signal_rows[item.id] = rows
    if len(signal_rows) < 2:
        return None

    row_signals: dict[str, list[str]] = {}
    evidence_by_id = {item.id: item for item in evidence}
    for evidence_id, rows in signal_rows.items():
        signal = evidence_by_id[evidence_id].kind.replace("anomaly_", "")
        for row in rows:
            row_signals.setdefault(row, []).append(signal)
    agreed = {
        row: sorted(signals)
        for row, signals in row_signals.items()
        if len(set(signals)) >= 2
    }
    pairwise_scores: list[float] = []
    ids = sorted(signal_rows)
    for left, right in itertools.combinations(ids, 2):
        pairwise_scores.append(_jaccard(signal_rows[left], signal_rows[right]))
    mean_pairwise = (
        sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    )
    return Evidence.create(
        kind="anomaly_detector_agreement",
        scope=EvidenceScope(table=table),
        value={
            "signal_count": len(signal_rows),
            "candidate_count": len(agreed),
            "mean_pairwise_top_set_jaccard": float(mean_pairwise),
            "top_rows": [
                {
                    "row_index": row,
                    "detector_count": len(signals),
                    "signals": signals,
                }
                for row, signals in sorted(
                    agreed.items(), key=lambda item: (-len(item[1]), item[0])
                )[:20]
            ],
        },
        method="ranked_detector_top_set_agreement_v1",
        description=f"Detector agreement across anomaly signals for {table}.",
        confidence=0.8 if agreed else 0.64,
        assumptions=(
            "Agreement is based on ranked review sets, not confirmed anomaly labels.",
        ),
        metadata={"source_evidence_ids": tuple(signal_rows)},
    )


def _conditional_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    target: str | None,
) -> list[Evidence]:
    columns = _numeric_columns(
        frame, table, exclude={target} if target else None, allow_identifiers=False
    )[:8]
    if len(columns) < 2 or len(frame) < 20:
        return []
    evidence: list[Evidence] = []
    for condition_column, value_column in itertools.permutations(columns, 2):
        pair = frame[[condition_column, value_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        pair = pair.dropna()
        if len(pair) < 20 or pair[condition_column].nunique() < 5:
            continue
        try:
            bins = pd.qcut(
                pair[condition_column],
                q=min(8, max(3, len(pair) // 10)),
                duplicates="drop",
            )
        except ValueError:
            continue
        if bins.nunique() < 3:
            continue
        scores = pd.Series(np.nan, index=pair.index, dtype="float64")
        for _, group in pair.groupby(bins, observed=False):
            if len(group) < 5:
                continue
            scores.loc[group.index] = _robust_z(group[value_column]).abs()
        candidates = scores[scores >= 3.5].dropna()
        if candidates.empty:
            continue
        evidence.append(
            Evidence.create(
                kind="anomaly_conditional_outlier",
                scope=EvidenceScope(
                    table=table.name, columns=(condition_column, value_column)
                ),
                value={
                    "condition_column": condition_column,
                    "value_column": value_column,
                    "evaluated_row_count": int(scores.notna().sum()),
                    "candidate_count": int(len(candidates)),
                    "candidate_rate": len(candidates)
                    / max(1, int(scores.notna().sum())),
                    "max_conditional_score": float(candidates.max()),
                    "examples": _top_index_values(candidates),
                },
                method="quantile_bin_conditional_modified_z_score_v1",
                description=(
                    f"Conditional candidates for {table.name}.{value_column} given "
                    f"{condition_column}."
                ),
                confidence=0.72,
                assumptions=(
                    "Quantile bins approximate local context and may miss nonlinear "
                    "or sparse subgroup effects.",
                ),
            )
        )
    return evidence


def _rare_category_evidence(frame: pd.DataFrame, table: TableCatalog) -> list[Evidence]:
    evidence: list[Evidence] = []
    row_count = len(frame)
    if row_count < 20:
        return evidence
    for column in _categorical_columns(frame, table):
        counts = frame[column].value_counts(dropna=False)
        rare = counts[counts <= max(1, math.floor(row_count * 0.01))]
        if rare.empty:
            continue
        evidence.append(
            Evidence.create(
                kind="anomaly_rare_category",
                scope=EvidenceScope(table=table.name, columns=(column,)),
                value={
                    "evaluated_row_count": row_count,
                    "rare_value_count": int(len(rare)),
                    "rare_row_count": int(rare.sum()),
                    "examples": [
                        {"value": _category_value(value), "count": int(count)}
                        for value, count in rare.head(10).items()
                    ],
                },
                method="low_frequency_category_scan_v1",
                description=f"Rare category candidates for {table.name}.{column}.",
                confidence=0.68,
                assumptions=(
                    "Rare values can be valid long-tail cases; domain review is "
                    "required.",
                ),
            )
        )
    return evidence


def _rare_category_combination_evidence(
    frame: pd.DataFrame, table: TableCatalog
) -> list[Evidence]:
    evidence: list[Evidence] = []
    row_count = len(frame)
    if row_count < 50:
        return evidence
    columns = _categorical_columns(frame, table)[:8]
    if len(columns) < 2:
        return evidence
    threshold = max(1, math.floor(row_count * 0.005))
    for left, right in itertools.combinations(columns, 2):
        pair = frame[[left, right]].copy()
        if (
            pair[left].nunique(dropna=False) < 2
            or pair[right].nunique(dropna=False) < 2
        ):
            continue
        counts = pair.value_counts(dropna=False)
        rare = counts[counts <= threshold]
        if rare.empty:
            continue
        examples: list[dict[str, Any]] = []
        for values, count in rare.head(10).items():
            left_value, right_value = (
                values if isinstance(values, tuple) else (values, None)
            )
            examples.append(
                {
                    left: _category_value(left_value),
                    right: _category_value(right_value),
                    "count": int(count),
                }
            )
        evidence.append(
            Evidence.create(
                kind="anomaly_rare_category_combination",
                scope=EvidenceScope(table=table.name, columns=(left, right)),
                value={
                    "evaluated_row_count": row_count,
                    "rare_combination_count": int(len(rare)),
                    "rare_row_count": int(rare.sum()),
                    "frequency_threshold": threshold,
                    "examples": examples,
                },
                method="low_frequency_category_pair_scan_v1",
                description=(
                    f"Rare category-pair candidates for {table.name}.{left} + {right}."
                ),
                confidence=0.66,
                assumptions=(
                    "Rare category combinations can be valid sparse segments; "
                    "domain review is required.",
                ),
            )
        )
    return evidence


def _label_evidence(frame: pd.DataFrame, table: str, target: str) -> Evidence | None:
    if target not in frame.columns:
        return None
    counts = frame[target].value_counts(dropna=False)
    if counts.empty:
        return None
    rare_count = int(counts.min())
    rare_rate = rare_count / len(frame) if len(frame) else 0.0
    return Evidence.create(
        kind="anomaly_label_summary",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "class_count": int(len(counts)),
            "counts": {str(value): int(count) for value, count in counts.items()},
            "minority_count": rare_count,
            "minority_rate": rare_rate,
        },
        method="target_frequency_summary_v1",
        description=f"Optional anomaly label summary for {table}.{target}.",
        confidence=1.0,
    )


def _findings_and_steps(
    evidence: Sequence[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    for item in evidence:
        value = item.value
        if item.kind == "anomaly_univariate_outlier" and value["candidate_count"]:
            column = item.scope.columns[0]
            findings.append(
                Finding.create(
                    title=f"Univariate tail candidates in {item.scope.table}.{column}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) "
                        f"({value['candidate_rate']:.1%}) sit outside robust tail "
                        "thresholds."
                    ),
                    severity="high" if value["candidate_rate"] >= 0.05 else "medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Review the example rows before capping, filtering, or "
                        "modeling them separately."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_outlier_candidates",
                    table=item.scope.table or "",
                    columns=item.scope.columns,
                    parameters={
                        "candidate_count": value["candidate_count"],
                        "method": item.method,
                    },
                    rationale=(
                        "Tail candidates may represent errors, rare valid cases, or "
                        "a separate regime."
                    ),
                    evidence_ids=(item.id,),
                    risk="medium",
                )
            )
        elif item.kind == "anomaly_multivariate_outlier" and value["candidate_count"]:
            findings.append(
                Finding.create(
                    title=f"Multivariate outlier candidates in {item.scope.table}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) "
                        f"({value['candidate_rate']:.1%}) have unusually large "
                        "robust multivariate scores."
                    ),
                    severity="high" if value["candidate_rate"] >= 0.02 else "medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Inspect the top contributing columns for the highest-scoring "
                        "records."
                    ),
                )
            )
        elif item.kind == "anomaly_isolation_forest" and value["candidate_count"]:
            findings.append(
                Finding.create(
                    title=f"Isolation Forest review candidates in {item.scope.table}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) "
                        f"({value['candidate_rate']:.1%}) ranked highest by the "
                        "Isolation Forest diagnostic."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Inspect the ranked rows and contributing numeric features "
                        "before treating them as anomalous."
                    ),
                )
            )
        elif item.kind == "anomaly_local_density_outlier" and value["candidate_count"]:
            findings.append(
                Finding.create(
                    title=f"Local-density review candidates in {item.scope.table}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) "
                        f"({value['candidate_rate']:.1%}) ranked highest by local "
                        "density contrast."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Compare these rows with nearby records; local-density "
                        "signals can be sensitive to sparse valid subgroups."
                    ),
                )
            )
        elif item.kind == "anomaly_detector_agreement" and value["candidate_count"]:
            findings.append(
                Finding.create(
                    title=f"Detector agreement on review rows in {item.scope.table}",
                    summary=(
                        f"{value['candidate_count']:,} row(s) appeared in at least "
                        "two ranked anomaly review sets."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Prioritize agreed rows for human review because independent "
                        "diagnostics surfaced them together."
                    ),
                )
            )
        elif item.kind == "anomaly_conditional_outlier":
            condition, observed = item.scope.columns
            findings.append(
                Finding.create(
                    title=(
                        f"Conditional anomaly candidates: {observed} given {condition}"
                    ),
                    summary=(
                        f"{value['candidate_count']:,} row(s) are unusual for "
                        f"{observed} within local {condition} bands."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Use domain rules or deeper modeling to confirm whether the "
                        "combination is implausible."
                    ),
                )
            )
        elif item.kind == "anomaly_rare_category":
            column = item.scope.columns[0]
            findings.append(
                Finding.create(
                    title=f"Rare categories in {item.scope.table}.{column}",
                    summary=(
                        f"{value['rare_value_count']:,} rare value(s) cover "
                        f"{value['rare_row_count']:,} row(s)."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm rare categories are valid before grouping or "
                        "treating them as anomalies."
                    ),
                )
            )
        elif item.kind == "anomaly_rare_category_combination":
            left, right = item.scope.columns
            findings.append(
                Finding.create(
                    title=(
                        f"Rare category combinations in {item.scope.table}: "
                        f"{left} + {right}"
                    ),
                    summary=(
                        f"{value['rare_combination_count']:,} rare combination(s) "
                        f"cover {value['rare_row_count']:,} row(s)."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm whether these sparse category pairs are valid "
                        "segments before grouping or treating them as anomalies."
                    ),
                )
            )
        elif item.kind == "anomaly_label_summary" and value["minority_rate"] <= 0.05:
            findings.append(
                Finding.create(
                    title=f"Rare labeled anomaly class in {item.scope.table}",
                    summary=(
                        f"The minority label has {value['minority_count']:,} row(s) "
                        f"({value['minority_rate']:.1%})."
                    ),
                    severity="medium",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Use stratified, group-aware review and avoid accuracy as "
                        "the main metric."
                    ),
                )
            )
    return findings, steps


def _artifact(evidence: Sequence[Evidence]) -> Artifact | None:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        value = item.value
        if item.kind in {
            "anomaly_univariate_outlier",
            "anomaly_multivariate_outlier",
            "anomaly_conditional_outlier",
            "anomaly_isolation_forest",
            "anomaly_local_density_outlier",
            "anomaly_detector_agreement",
        }:
            rows.append(
                {
                    "signal": item.kind.replace("anomaly_", "").replace("_", " "),
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value.get("candidate_count", 0),
                    "rate": f"{value.get('candidate_rate', 0.0):.1%}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "anomaly_rare_category":
            rows.append(
                {
                    "signal": "rare category",
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value["rare_row_count"],
                    "rate": "low frequency",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "anomaly_rare_category_combination":
            rows.append(
                {
                    "signal": "rare category combination",
                    "table": item.scope.table,
                    "columns": " + ".join(item.scope.columns),
                    "candidate_rows": value["rare_row_count"],
                    "rate": "low frequency",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
    rows = [row for row in rows if row["candidate_rows"]]
    if not rows:
        return None
    return Artifact.create(
        kind="metric_table",
        title="Anomaly candidate signals",
        data={
            "columns": [
                {"key": "signal", "label": "Signal"},
                {"key": "table", "label": "Table"},
                {"key": "columns", "label": "Columns"},
                {"key": "candidate_rows", "label": "Candidate rows"},
                {"key": "rate", "label": "Rate"},
                {"key": "confidence", "label": "Confidence"},
            ],
            "rows": rows[:30],
        },
        evidence_ids=tuple(item.id for item in evidence),
        metadata={
            "description": (
                "Ranked statistical review signals; candidates are not confirmed "
                "anomalies."
            )
        },
    )


def anomaly_detection_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    table: str | None = None,
    target: str | None = None,
    expected_contamination: float | None = None,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Run deterministic anomaly-detection diagnostics."""
    _validate_expected_contamination(expected_contamination)
    emit(
        callbacks,
        Event(EventKind.RUN_STARTED, "Anomaly diagnostics started.", stage="anomaly"),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    selected_tables = [catalog.table(table)] if table else list(catalog.tables)
    evidence: list[Evidence] = []

    for table_catalog in selected_tables:
        frame = tables[table_catalog.name]
        sampled = _sample_frame(
            frame,
            table=table_catalog.name,
            config=config,
            warnings=warnings,
            sampling=sampling,
        )
        active_target = target or context.target
        if active_target and active_target in sampled.columns:
            label = _label_evidence(sampled, table_catalog.name, active_target)
            if label is not None:
                evidence.append(label)
        evidence.extend(
            _univariate_evidence(sampled, table_catalog, target=active_target)
        )
        multivariate = _multivariate_evidence(
            sampled, table_catalog, target=active_target
        )
        if multivariate is not None:
            evidence.append(multivariate)
        isolation_forest = _isolation_forest_evidence(
            sampled,
            table_catalog,
            target=active_target,
            random_seed=config.random_seed,
            expected_contamination=expected_contamination,
        )
        if isolation_forest is not None:
            evidence.append(isolation_forest)
        local_density = _local_density_evidence(
            sampled,
            table_catalog,
            target=active_target,
            expected_contamination=expected_contamination,
        )
        if local_density is not None:
            evidence.append(local_density)
        evidence.extend(
            _conditional_evidence(sampled, table_catalog, target=active_target)
        )
        evidence.extend(_rare_category_evidence(sampled, table_catalog))
        evidence.extend(_rare_category_combination_evidence(sampled, table_catalog))
        agreement = _agreement_evidence(evidence, table=table_catalog.name)
        if agreement is not None:
            evidence.append(agreement)

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
    findings, steps = _findings_and_steps(evidence)
    artifacts = tuple(item for item in (_artifact(evidence),) if item is not None)

    if not selected_tables or catalog.row_count == 0:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "There is insufficient data for anomaly diagnostics."
    elif not evidence:
        status = AnalysisStatus.NO_MEANINGFUL_STRUCTURE
        summary = (
            "No usable anomaly-detection signals were available in the selected data."
        )
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = (
            f"Ran anomaly diagnostics across {len(selected_tables)} table(s); "
            f"found {len(findings)} prioritized candidate signal(s), with warnings."
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = (
            f"Ran anomaly diagnostics across {len(selected_tables)} table(s); "
            f"found {len(findings)} prioritized candidate signal(s)."
        )

    result = AnalysisResult(
        goal="anomaly_detection",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=artifacts,
        assumptions=(
            *context.assumptions,
            "Unsupervised anomaly diagnostics identify review candidates, not "
            "confirmed anomalies.",
        ),
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "selected_table": table,
            "target": target or context.target,
            "expected_contamination": expected_contamination,
            "candidate_signals": len(findings),
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="anomaly",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
