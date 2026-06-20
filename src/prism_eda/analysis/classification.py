"""Goal-aware deterministic diagnostics for classification EDA."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

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
            code="sampled_classification_analysis",
            message=(
                f"{table} has {len(frame):,} rows; classification diagnostics were "
                f"run on a deterministic {budget:,}-row sample."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="classification_analysis",
            source_rows=len(frame),
            sampled_rows=budget,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="row_count_exceeds_mode_budget",
            limitations=(
                "Small classes or rare feature values may be absent from the sample.",
            ),
        )
    )
    return sampled


def _resolve_table(
    catalog: DatasetCatalog,
    tables: Mapping[str, pd.DataFrame],
    *,
    table: str | None,
    target: str | None,
    warnings: list[AnalysisWarning],
) -> tuple[TableCatalog | None, str | None]:
    if target is None:
        warnings.append(
            AnalysisWarning(
                code="classification_target_required",
                message="Classification analysis requires a target column.",
            )
        )
        return None, None
    if table is not None:
        if table not in tables:
            warnings.append(
                AnalysisWarning(
                    code="classification_table_not_found",
                    message=f"Table {table!r} was not found.",
                    table=table,
                )
            )
            return None, target
        table_catalog = catalog.table(table)
        if target not in tables[table].columns:
            warnings.append(
                AnalysisWarning(
                    code="classification_target_not_found",
                    message=f"Target column {target!r} was not found in {table!r}.",
                    table=table,
                    column=target,
                )
            )
            return None, target
        return table_catalog, target

    matches = [item for item in catalog.tables if target in tables[item.name].columns]
    if len(matches) == 1:
        return matches[0], target
    if not matches:
        warnings.append(
            AnalysisWarning(
                code="classification_target_not_found",
                message=f"Target column {target!r} was not found in any table.",
                column=target,
            )
        )
    else:
        warnings.append(
            AnalysisWarning(
                code="classification_target_ambiguous",
                message=(
                    f"Target column {target!r} appears in multiple tables; pass table=."
                ),
                column=target,
            )
        )
    return None, target


def _target_summary(frame: pd.DataFrame, table: str, target: str) -> Evidence:
    target_series = frame[target]
    counts = target_series.value_counts(dropna=False)
    total = len(target_series)
    non_missing = int(target_series.notna().sum())
    probabilities = counts / total if total else counts
    entropy = float(-sum(prob * math.log2(prob) for prob in probabilities if prob > 0))
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy else 0.0
    smallest = int(counts.min()) if len(counts) else 0
    largest = int(counts.max()) if len(counts) else 0
    return Evidence.create(
        kind="classification_target_summary",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "row_count": total,
            "non_missing_count": non_missing,
            "missing_count": total - non_missing,
            "missing_rate": (total - non_missing) / total if total else 0.0,
            "class_count": int(len(counts)),
            "class_counts": {str(value): int(count) for value, count in counts.items()},
            "majority_class": str(counts.idxmax()) if len(counts) else None,
            "majority_rate": float(counts.max() / total) if total else 0.0,
            "minority_count": smallest,
            "imbalance_ratio": (largest / smallest) if smallest else None,
            "normalized_entropy": normalized_entropy,
        },
        method="target_distribution_summary_v1",
        description=f"Classification target distribution for {table}.{target}.",
        confidence=1.0,
    )


def _eta_squared(feature: pd.Series, target: pd.Series) -> float | None:
    numeric = pd.to_numeric(feature, errors="coerce")
    data = pd.DataFrame({"feature": numeric, "target": target}).dropna()
    if len(data) < 3 or data["target"].nunique() < 2:
        return None
    overall = data["feature"].mean()
    total_ss = float(((data["feature"] - overall) ** 2).sum())
    if total_ss <= 0:
        return None
    between_ss = 0.0
    for _, group in data.groupby("target", dropna=False):
        between_ss += len(group) * float((group["feature"].mean() - overall) ** 2)
    return max(0.0, min(1.0, between_ss / total_ss))


def _cramers_v(feature: pd.Series, target: pd.Series) -> float | None:
    data = pd.DataFrame(
        {"feature": feature.astype("string"), "target": target}
    ).dropna()
    if len(data) < 3 or data["feature"].nunique() < 2 or data["target"].nunique() < 2:
        return None
    table = pd.crosstab(data["feature"], data["target"])
    observed = table.to_numpy(dtype="float64")
    total = observed.sum()
    if total <= 0:
        return None
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = np.nansum((observed - expected) ** 2 / expected)
    denominator = total * max(1, min(observed.shape[0] - 1, observed.shape[1] - 1))
    return float(math.sqrt(max(0.0, chi_square / denominator)))


def _value_rule_accuracy(feature: pd.Series, target: pd.Series) -> float | None:
    data = pd.DataFrame(
        {"feature": feature.astype("string"), "target": target}
    ).dropna()
    if len(data) < 3 or data["feature"].nunique() <= 1:
        return None
    if data["feature"].nunique() > max(50, int(len(data) * 0.4)):
        return None
    majority_matches = (
        data.groupby("feature")["target"].value_counts().groupby(level=0).max()
    )
    return float(majority_matches.sum() / len(data))


def _association_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    max_categories: int,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    target_series = frame[target]
    profile_by_name = {column.name: column for column in table.columns}
    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        profile = profile_by_name.get(name)
        if profile is None:
            continue
        series = frame[column]
        if ptypes.is_numeric_dtype(series.dtype):
            score = _eta_squared(series, target_series)
            if score is not None:
                evidence.append(
                    Evidence.create(
                        kind="classification_numeric_association",
                        scope=EvidenceScope(table=table.name, columns=(name, target)),
                        value={
                            "feature": name,
                            "target": target,
                            "effect_size": score,
                            "metric": "eta_squared",
                        },
                        method="one_way_anova_effect_size_v1",
                        description=(
                            f"Numeric feature-target association for "
                            f"{table.name}.{name}."
                        ),
                        confidence=0.78,
                    )
                )
        if profile.unique_count is not None and profile.unique_count > max_categories:
            evidence.append(
                Evidence.create(
                    kind="classification_high_cardinality_feature",
                    scope=EvidenceScope(table=table.name, columns=(name,)),
                    value={
                        "unique_count": profile.unique_count,
                        "unique_rate": profile.unique_rate,
                        "max_recommended_categories": max_categories,
                    },
                    method="cardinality_risk_scan_v1",
                    description=(
                        f"High-cardinality feature risk for {table.name}.{name}."
                    ),
                    confidence=0.82,
                )
            )
            continue
        if profile.semantic_type in {"categorical", "boolean", "text"}:
            score = _cramers_v(series, target_series)
            if score is not None:
                evidence.append(
                    Evidence.create(
                        kind="classification_categorical_association",
                        scope=EvidenceScope(table=table.name, columns=(name, target)),
                        value={
                            "feature": name,
                            "target": target,
                            "effect_size": score,
                            "metric": "cramers_v",
                            "unique_count": profile.unique_count,
                        },
                        method="cramers_v_association_v1",
                        description=(
                            f"Categorical feature-target association for "
                            f"{table.name}.{name}."
                        ),
                        confidence=0.76,
                    )
                )
    return evidence


def _missingness_by_class_evidence(
    frame: pd.DataFrame, table: TableCatalog, target: str
) -> list[Evidence]:
    evidence: list[Evidence] = []
    target_series = frame[target]
    for column in frame.columns:
        name = str(column)
        if name == target or not frame[column].isna().any():
            continue
        rates = frame[column].isna().groupby(target_series, dropna=False).mean()
        if len(rates) < 2:
            continue
        gap = float(rates.max() - rates.min())
        if gap < 0.2:
            continue
        evidence.append(
            Evidence.create(
                kind="classification_missingness_by_class",
                scope=EvidenceScope(table=table.name, columns=(name, target)),
                value={
                    "feature": name,
                    "target": target,
                    "max_missingness_gap": gap,
                    "missingness_by_class": {
                        str(key): float(value) for key, value in rates.items()
                    },
                },
                method="class_conditional_missingness_v1",
                description=(f"Class-conditional missingness for {table.name}.{name}."),
                confidence=0.84,
            )
        )
    return evidence


def _leakage_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    target_summary: Evidence,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    target_series = frame[target]
    target_token = target.lower().replace("_", "")
    majority_rate = float(target_summary.value["majority_rate"])
    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        normalized_name = name.lower().replace("_", "")
        series = frame[column]
        rule_accuracy = _value_rule_accuracy(series, target_series)
        exact_copy = (
            series.astype("string")
            .fillna("<NA>")
            .equals(target_series.astype("string").fillna("<NA>"))
        )
        name_overlap = target_token and target_token in normalized_name
        suspicious_accuracy = (
            rule_accuracy is not None
            and rule_accuracy >= max(0.98, majority_rate + 0.15)
            and rule_accuracy > majority_rate
        )
        if not (exact_copy or name_overlap or suspicious_accuracy):
            continue
        evidence.append(
            Evidence.create(
                kind="classification_leakage_candidate",
                scope=EvidenceScope(table=table.name, columns=(name, target)),
                value={
                    "feature": name,
                    "target": target,
                    "exact_target_copy": exact_copy,
                    "name_contains_target": bool(name_overlap),
                    "value_rule_accuracy": rule_accuracy,
                    "majority_baseline": majority_rate,
                },
                method="deterministic_leakage_screen_v1",
                description=f"Potential target leakage candidate {table.name}.{name}.",
                confidence=0.9 if exact_copy or suspicious_accuracy else 0.64,
                assumptions=(
                    "High deterministic predictability can be valid in simple "
                    "domains; review timing and data lineage.",
                ),
            )
        )
    return evidence


def _conflicting_duplicate_evidence(
    frame: pd.DataFrame, table: str, target: str
) -> Evidence | None:
    feature_columns = [column for column in frame.columns if str(column) != target]
    if not feature_columns or len(frame) < 2:
        return None
    try:
        feature_hash = pd.util.hash_pandas_object(frame[feature_columns], index=False)
    except (TypeError, ValueError):
        return None
    grouped = pd.DataFrame({"hash": feature_hash, "target": frame[target]}).groupby(
        "hash"
    )
    conflicting = grouped["target"].nunique(dropna=False)
    conflict_hashes = conflicting[conflicting > 1]
    if conflict_hashes.empty:
        conflict_rows = 0
    else:
        conflict_rows = int(feature_hash.isin(conflict_hashes.index).sum())
    return Evidence.create(
        kind="classification_conflicting_duplicates",
        scope=EvidenceScope(table=table, columns=(target,)),
        value={
            "duplicate_signature_count": int((grouped.size() > 1).sum()),
            "conflicting_signature_count": int(len(conflict_hashes)),
            "conflicting_row_count": conflict_rows,
        },
        method="feature_hash_conflicting_label_scan_v1",
        description=f"Duplicate feature rows with conflicting labels in {table}.",
        confidence=0.86,
    )


def _findings_and_steps(
    evidence: list[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    for item in evidence:
        value = item.value
        if item.kind == "classification_target_summary":
            if value["class_count"] < 2 or value["missing_rate"] > 0:
                findings.append(
                    Finding.create(
                        title=f"Target validity risk in {item.scope.table}",
                        summary=(
                            f"Target has {value['class_count']} class(es) and "
                            f"{value['missing_rate']:.1%} missing labels."
                        ),
                        severity="high",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Fix target definition before trusting classification "
                            "diagnostics."
                        ),
                    )
                )
            if value["class_count"] >= 2 and (
                value["majority_rate"] >= 0.8 or value["minority_count"] < 10
            ):
                findings.append(
                    Finding.create(
                        title=f"Class imbalance in {item.scope.table}",
                        summary=(
                            f"Majority class covers {value['majority_rate']:.1%}; "
                            f"minority class has {value['minority_count']:,} row(s)."
                        ),
                        severity="high" if value["majority_rate"] >= 0.9 else "medium",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Prefer stratified metrics, threshold tuning, and "
                            "class-aware validation."
                        ),
                    )
                )
                steps.append(
                    TransformationStep(
                        operation="review_class_imbalance_strategy",
                        table=item.scope.table or "",
                        columns=item.scope.columns,
                        parameters={
                            "majority_rate": value["majority_rate"],
                            "minority_count": value["minority_count"],
                        },
                        rationale=(
                            "Severe imbalance can hide poor minority-class performance."
                        ),
                        evidence_ids=(item.id,),
                        risk="high",
                    )
                )
        elif item.kind == "classification_leakage_candidate":
            findings.append(
                Finding.create(
                    title=f"Potential target leakage: {item.scope.columns[0]}",
                    summary=(
                        f"{item.scope.columns[0]} has deterministic target-signal "
                        f"risk against {item.scope.columns[1]}."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm the feature is available before prediction time "
                        "and is not derived from the label."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_target_leakage_candidate",
                    table=item.scope.table or "",
                    columns=item.scope.columns,
                    parameters=value,
                    rationale="Leaky fields can make validation scores unrealistic.",
                    evidence_ids=(item.id,),
                    risk="high",
                )
            )
        elif (
            item.kind
            in {
                "classification_numeric_association",
                "classification_categorical_association",
            }
            and value["effect_size"] >= 0.35
        ):
            findings.append(
                Finding.create(
                    title=f"Strong target association: {item.scope.columns[0]}",
                    summary=(
                        f"{value['metric']} is {value['effect_size']:.2f} for "
                        f"{item.scope.columns[0]} against {item.scope.columns[1]}."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Treat very strong associations as useful signals unless "
                        "timing suggests leakage."
                    ),
                )
            )
        elif item.kind == "classification_high_cardinality_feature":
            findings.append(
                Finding.create(
                    title=f"High-cardinality feature: {item.scope.columns[0]}",
                    summary=(
                        f"{item.scope.columns[0]} has {value['unique_count']:,} "
                        "distinct value(s), which can overfit naive encoders."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Use leakage-safe encoding, hashing, grouping, or feature "
                        "exclusion after review."
                    ),
                )
            )
        elif item.kind == "classification_missingness_by_class":
            findings.append(
                Finding.create(
                    title=f"Class-dependent missingness: {item.scope.columns[0]}",
                    summary=(
                        f"Missingness differs by up to "
                        f"{value['max_missingness_gap']:.1%} across target classes."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Review whether missingness is predictive, structural, or "
                        "a data collection artifact."
                    ),
                )
            )
        elif (
            item.kind == "classification_conflicting_duplicates"
            and value["conflicting_row_count"]
        ):
            findings.append(
                Finding.create(
                    title=f"Conflicting duplicate labels in {item.scope.table}",
                    summary=(
                        f"{value['conflicting_row_count']:,} row(s) share feature "
                        "signatures with conflicting target labels."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Resolve label conflicts or preserve them as known "
                        "label-noise evidence."
                    ),
                )
            )
    return findings, steps


def _artifacts(evidence: list[Evidence]) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []
    target_items = [
        item for item in evidence if item.kind == "classification_target_summary"
    ]
    if target_items:
        item = target_items[0]
        total = item.value["row_count"]
        rows = [
            {
                "class": label,
                "rows": count,
                "rate": f"{count / total:.1%}" if total else "0.0%",
            }
            for label, count in item.value["class_counts"].items()
        ]
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Class balance",
                data={
                    "columns": [
                        {"key": "class", "label": "Class"},
                        {"key": "rows", "label": "Rows"},
                        {"key": "rate", "label": "Rate"},
                    ],
                    "rows": rows,
                },
                evidence_ids=(item.id,),
                metadata={"description": "Target class counts and proportions."},
            )
        )

    signal_rows: list[dict[str, Any]] = []
    for item in evidence:
        if item.kind in {
            "classification_numeric_association",
            "classification_categorical_association",
        }:
            signal_rows.append(
                {
                    "signal": item.kind.replace("classification_", "").replace(
                        "_", " "
                    ),
                    "feature": item.scope.columns[0],
                    "metric": item.value["metric"],
                    "score": f"{item.value['effect_size']:.2f}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "classification_leakage_candidate":
            signal_rows.append(
                {
                    "signal": "leakage candidate",
                    "feature": item.scope.columns[0],
                    "metric": "rule accuracy",
                    "score": (
                        f"{item.value['value_rule_accuracy']:.2f}"
                        if item.value["value_rule_accuracy"] is not None
                        else "name/copy"
                    ),
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "classification_high_cardinality_feature":
            signal_rows.append(
                {
                    "signal": "high cardinality",
                    "feature": item.scope.columns[0],
                    "metric": "unique count",
                    "score": item.value["unique_count"],
                    "confidence": f"{item.confidence:.0%}",
                }
            )
    if signal_rows:
        signal_rows.sort(key=lambda row: str(row["score"]), reverse=True)
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Feature-target diagnostic signals",
                data={
                    "columns": [
                        {"key": "signal", "label": "Signal"},
                        {"key": "feature", "label": "Feature"},
                        {"key": "metric", "label": "Metric"},
                        {"key": "score", "label": "Score"},
                        {"key": "confidence", "label": "Confidence"},
                    ],
                    "rows": signal_rows[:30],
                },
                evidence_ids=tuple(item.id for item in evidence),
                metadata={
                    "description": (
                        "Deterministic target association and leakage-screen signals."
                    )
                },
            )
        )
    return tuple(artifacts)


def classification_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    target: str | None,
    table: str | None = None,
    max_categories: int = 50,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Run deterministic classification diagnostics for one labeled table."""
    emit(
        callbacks,
        Event(
            EventKind.RUN_STARTED,
            "Classification diagnostics started.",
            stage="classification",
        ),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    resolved_target = target or context.target
    table_catalog, resolved_target = _resolve_table(
        catalog,
        tables,
        table=table,
        target=resolved_target,
        warnings=warnings,
    )
    evidence: list[Evidence] = []
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    if table_catalog is not None and resolved_target is not None:
        frame = _sample_frame(
            tables[table_catalog.name],
            table=table_catalog.name,
            config=config,
            warnings=warnings,
            sampling=sampling,
        )
        target_summary = _target_summary(frame, table_catalog.name, resolved_target)
        evidence.append(target_summary)
        if target_summary.value["class_count"] >= 2:
            conflict = _conflicting_duplicate_evidence(
                frame, table_catalog.name, resolved_target
            )
            if conflict is not None:
                evidence.append(conflict)
            evidence.extend(
                _association_evidence(
                    frame,
                    table_catalog,
                    resolved_target,
                    max_categories=max_categories,
                )
            )
            evidence.extend(
                _missingness_by_class_evidence(frame, table_catalog, resolved_target)
            )
            evidence.extend(
                _leakage_evidence(
                    frame,
                    table_catalog,
                    resolved_target,
                    target_summary=target_summary,
                )
            )
        findings, steps = _findings_and_steps(evidence)

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

    if table_catalog is None or resolved_target is None:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "Classification analysis needs one target column in one table."
    elif not evidence or (
        evidence[0].kind == "classification_target_summary"
        and evidence[0].value["class_count"] < 2
    ):
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "The selected target does not contain enough class evidence."
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = (
            f"Ran classification diagnostics for "
            f"{table_catalog.name}.{resolved_target}; "
            f"found {len(findings)} prioritized issue(s), with warnings."
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = (
            f"Ran classification diagnostics for "
            f"{table_catalog.name}.{resolved_target}; "
            f"found {len(findings)} prioritized issue(s)."
        )

    result = AnalysisResult(
        goal="classification",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=_artifacts(evidence),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "selected_table": table_catalog.name if table_catalog else table,
            "target": resolved_target,
            "max_categories": max_categories,
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="classification",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
