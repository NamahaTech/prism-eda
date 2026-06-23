"""Goal-aware deterministic diagnostics for classification EDA."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
)
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

_MAX_PROBE_FEATURES = 30
_MAX_HARD_EXAMPLES = 20

# A simple value->label rule that is near-perfect *and* clears the majority
# baseline by a real margin is a leakage signal. The bar must stay reachable
# (<= 1.0) even when the target is imbalanced, which is exactly when leakage
# matters most, so it is a fixed accuracy floor plus a lift guard rather than a
# margin added on top of the majority rate.
_LEAKAGE_RULE_ACCURACY = 0.98
_LEAKAGE_RULE_LIFT = 0.05


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


def _leakage_feature_names(evidence: list[Evidence]) -> set[str]:
    return {
        item.scope.columns[0]
        for item in evidence
        if item.kind == "classification_leakage_candidate" and item.scope.columns
    }


def _probe_feature_groups(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    max_categories: int,
    leakage_features: set[str],
) -> tuple[list[str], list[str], list[str]]:
    profile_by_name = {column.name: column for column in table.columns}
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    for column in frame.columns:
        name = str(column)
        profile = profile_by_name.get(name)
        if name == target or profile is None:
            continue
        if name in leakage_features:
            excluded.append(name)
            continue
        if "identifier_candidate" in profile.roles:
            excluded.append(name)
            continue
        if ptypes.is_numeric_dtype(frame[column].dtype):
            numeric.append(name)
            continue
        if profile.semantic_type in {"categorical", "boolean", "text"}:
            if profile.unique_count is None or profile.unique_count <= max_categories:
                categorical.append(name)
            else:
                excluded.append(name)

    selected_numeric = numeric[:_MAX_PROBE_FEATURES]
    remaining = _MAX_PROBE_FEATURES - len(selected_numeric)
    selected_categorical = categorical[: max(0, remaining)]
    excluded.extend(numeric[len(selected_numeric) :])
    excluded.extend(categorical[len(selected_categorical) :])
    return selected_numeric, selected_categorical, sorted(set(excluded))


def _probe_folds(y: pd.Series, mode: AnalysisMode | str) -> int | None:
    counts = y.value_counts(dropna=False)
    if len(y) < 30 or len(counts) < 2:
        return None
    min_class_count = int(counts.min())
    if min_class_count < 3:
        return None
    max_folds = 3 if AnalysisMode(mode) == AnalysisMode.QUICK else 5
    return min(max_folds, min_class_count)


def _classification_probe_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )


def _classification_probe_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    config: AnalysisConfig,
    max_categories: int,
    prior_evidence: list[Evidence],
) -> list[Evidence]:
    usable = frame[frame[target].notna()].copy()
    y = usable[target]
    folds = _probe_folds(y, config.mode)
    leakage_features = _leakage_feature_names(prior_evidence)
    numeric, categorical, excluded = _probe_feature_groups(
        usable,
        table,
        target,
        max_categories=max_categories,
        leakage_features=leakage_features,
    )
    if folds is None or not (numeric or categorical):
        return []

    features = numeric + categorical
    X = usable[features]
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    probe = _classification_probe_pipeline(numeric, categorical)
    predictions = pd.Series(
        cross_val_predict(probe, X, y, cv=splitter, method="predict"),
        index=usable.index,
    )
    probabilities = cross_val_predict(probe, X, y, cv=splitter, method="predict_proba")
    classes = [str(item) for item in np.unique(y)]
    majority = DummyClassifier(strategy="most_frequent")
    baseline_predictions = pd.Series(
        cross_val_predict(majority, X, y, cv=splitter, method="predict"),
        index=usable.index,
    )

    accuracy = float(accuracy_score(y, predictions))
    balanced = float(balanced_accuracy_score(y, predictions))
    baseline_balanced = float(balanced_accuracy_score(y, baseline_predictions))
    macro_f1 = float(f1_score(y, predictions, average="macro", zero_division=0))

    probability_frame = pd.DataFrame(
        probabilities,
        index=usable.index,
        columns=classes,
    )
    max_probability = probability_frame.max(axis=1)
    predicted_probability = pd.Series(
        [
            float(probability_frame.loc[index, str(prediction)])
            if str(prediction) in probability_frame.columns
            else float(max_probability.loc[index])
            for index, prediction in predictions.items()
        ],
        index=predictions.index,
    )
    hard_mask = predictions.astype("string") != y.astype("string")
    hard_scores = (1.0 - predicted_probability).where(hard_mask, 0.0)
    hard_rows = [
        {
            "row_index": str(index),
            "true_label": str(y.loc[index]),
            "predicted_label": str(predictions.loc[index]),
            "predicted_probability": float(predicted_probability.loc[index]),
        }
        for index in hard_scores.sort_values(ascending=False)
        .head(_MAX_HARD_EXAMPLES)
        .index
        if bool(hard_mask.loc[index])
    ]

    probe_evidence = Evidence.create(
        kind="classification_probe_model",
        scope=EvidenceScope(table=table.name, columns=tuple(features + [target])),
        value={
            "model": "logistic_regression_balanced",
            "cv_folds": folds,
            "row_count": int(len(usable)),
            "feature_count": len(features),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "excluded_features": excluded,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "majority_baseline_balanced_accuracy": baseline_balanced,
            "balanced_accuracy_lift": balanced - baseline_balanced,
        },
        method="leakage_screened_logistic_probe_cv_v1",
        description=(
            f"Cross-validated diagnostic classification probe for {table.name}."
        ),
        confidence=0.74,
        assumptions=(
            "The probe is a diagnostic separability check, not a production model.",
            "Preprocessing is fit inside each cross-validation fold.",
            "Obvious leakage, identifier, and high-cardinality features are excluded.",
        ),
    )
    evidence = [probe_evidence]
    if hard_rows:
        evidence.append(
            Evidence.create(
                kind="classification_hard_examples",
                scope=EvidenceScope(
                    table=table.name, columns=tuple(features + [target])
                ),
                value={
                    "row_count": len(hard_rows),
                    "error_rate": float(
                        (predictions.astype("string") != y.astype("string")).mean()
                    ),
                    "examples": hard_rows,
                    "source_probe_evidence_id": probe_evidence.id,
                },
                method="cross_validated_probe_error_review_v1",
                description=(
                    f"Cross-validated hard-example candidates for {table.name}."
                ),
                confidence=0.68,
                assumptions=(
                    "Hard examples are rows the diagnostic probe misclassified; they "
                    "may reflect overlap, label noise, or insufficient features.",
                ),
            )
        )
    return evidence


def _split_guidance_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    *,
    context: AnalysisContext,
) -> Evidence | None:
    risks: list[dict[str, Any]] = []
    if context.entity_id and context.entity_id in frame.columns:
        entity_counts = frame[context.entity_id].value_counts(dropna=False)
        repeated = entity_counts[entity_counts > 1]
        repeated_rows = int(frame[context.entity_id].isin(repeated.index).sum())
        if repeated_rows:
            risks.append(
                {
                    "kind": "group_split_recommended",
                    "column": context.entity_id,
                    "repeated_entity_count": int(len(repeated)),
                    "repeated_row_count": repeated_rows,
                    "repeated_row_rate": repeated_rows / len(frame)
                    if len(frame)
                    else 0.0,
                    "reason": (
                        "Rows from the same entity can leak information across "
                        "random train/test splits."
                    ),
                }
            )
    if context.timestamp and context.timestamp in frame.columns:
        timestamps = pd.to_datetime(frame[context.timestamp], errors="coerce")
        valid = pd.DataFrame(
            {"timestamp": timestamps, "target": frame[target]}
        ).dropna()
        if len(valid) >= 20 and valid["timestamp"].nunique() >= 5:
            ordered = valid.sort_values("timestamp")
            split = max(1, len(ordered) // 4)
            early = ordered.head(split)["target"].value_counts(normalize=True)
            late = ordered.tail(split)["target"].value_counts(normalize=True)
            labels = set(early.index) | set(late.index)
            max_distribution_shift = max(
                abs(float(early.get(label, 0.0)) - float(late.get(label, 0.0)))
                for label in labels
            )
            span_days = (
                ordered["timestamp"].max() - ordered["timestamp"].min()
            ).total_seconds() / 86_400
            risks.append(
                {
                    "kind": "time_split_recommended",
                    "column": context.timestamp,
                    "span_days": float(span_days),
                    "max_target_distribution_shift": float(max_distribution_shift),
                    "reason": (
                        "Timestamped classification data should usually be validated "
                        "with temporal order preserved."
                    ),
                }
            )
    if not risks:
        return None
    return Evidence.create(
        kind="classification_split_guidance",
        scope=EvidenceScope(table=table.name, columns=(target,)),
        value={"risks": risks},
        method="context_aware_split_guidance_v1",
        description=f"Split-design guidance for {table.name}.{target}.",
        confidence=0.82,
        assumptions=(
            "Split guidance uses user-provided context fields and lightweight target "
            "distribution checks.",
        ),
    )


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
        # Near-unique, identifier-named columns memorize rows; flag them for
        # exclusion instead of computing an association or a generic
        # high-cardinality warning that would bury the real signal.
        if "identifier_candidate" in profile.roles:
            evidence.append(
                Evidence.create(
                    kind="classification_identifier_feature",
                    scope=EvidenceScope(table=table.name, columns=(name,)),
                    value={
                        "feature": name,
                        "unique_count": profile.unique_count,
                        "unique_rate": profile.unique_rate,
                    },
                    method="identifier_role_exclusion_v1",
                    description=(
                        f"Identifier-like feature {table.name}.{name} is "
                        "near-unique per row."
                    ),
                    confidence=0.9,
                    assumptions=(
                        "Identifier-like columns usually label rows rather than "
                        "explain the target and should be excluded from features.",
                    ),
                )
            )
            continue
        if ptypes.is_numeric_dtype(series.dtype):
            # A numeric column with many distinct values is normal, not a
            # high-cardinality categorical risk, so it only gets an association.
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
            continue
        # Remaining columns are non-numeric, non-identifier. Only genuine
        # categorical/text columns can carry high-cardinality encoding risk.
        if (
            profile.semantic_type in {"categorical", "text"}
            and profile.unique_count is not None
            and profile.unique_count > max_categories
        ):
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
            and rule_accuracy >= _LEAKAGE_RULE_ACCURACY
            and (rule_accuracy - majority_rate) >= _LEAKAGE_RULE_LIFT
        )
        if not (exact_copy or name_overlap or suspicious_accuracy):
            continue
        near_perfect = exact_copy or (
            rule_accuracy is not None and rule_accuracy >= 0.999
        )
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
                    "near_perfect": near_perfect,
                },
                method="deterministic_leakage_screen_v2",
                description=f"Potential target leakage candidate {table.name}.{name}.",
                confidence=0.92 if exact_copy or suspicious_accuracy else 0.64,
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
    leakage_columns = {
        item.scope.columns[0]
        for item in evidence
        if item.kind == "classification_leakage_candidate" and item.scope.columns
    }
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
            severity = "critical" if value.get("near_perfect") else "high"
            findings.append(
                Finding.create(
                    title=f"Potential target leakage: {item.scope.columns[0]}",
                    summary=(
                        f"{item.scope.columns[0]} has deterministic target-signal "
                        f"risk against {item.scope.columns[1]}."
                    ),
                    severity=severity,
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
        elif item.kind == "classification_identifier_feature":
            findings.append(
                Finding.create(
                    title=f"Identifier-like feature: {item.scope.columns[0]}",
                    summary=(
                        f"{item.scope.columns[0]} is unique on "
                        f"{value['unique_rate']:.0%} of rows and likely labels "
                        "records rather than explaining the target."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Exclude identifier-like columns from features; they "
                        "memorize rows and inflate validation scores."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="exclude_identifier_feature",
                    table=item.scope.table or "",
                    columns=item.scope.columns,
                    parameters=value,
                    rationale=(
                        "Identifier-like columns leak row identity into the model."
                    ),
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
            and item.scope.columns[0] not in leakage_columns
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
        elif item.kind == "classification_probe_model":
            lift = value["balanced_accuracy_lift"]
            balanced_accuracy = value["balanced_accuracy"]
            if balanced_accuracy < 0.6 or lift < 0.05:
                findings.append(
                    Finding.create(
                        title=f"Weak classification separability in {item.scope.table}",
                        summary=(
                            "The leakage-screened probe reached "
                            f"{balanced_accuracy:.1%} balanced accuracy, only "
                            f"{lift:.1%} above the majority baseline."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Treat the current features as weak for this target; "
                            "review label quality, class overlap, and missing "
                            "predictors."
                        ),
                    )
                )
            elif balanced_accuracy >= 0.85 and lift >= 0.2:
                findings.append(
                    Finding.create(
                        title=(
                            f"Strong classification separability in {item.scope.table}"
                        ),
                        summary=(
                            "The leakage-screened probe reached "
                            f"{balanced_accuracy:.1%} balanced accuracy, "
                            f"{lift:.1%} above the majority baseline."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Use this as readiness evidence, while still checking "
                            "timing, leakage, and validation design before modeling."
                        ),
                    )
                )
        elif item.kind == "classification_hard_examples":
            findings.append(
                Finding.create(
                    title=f"Probe hard examples in {item.scope.table}",
                    summary=(
                        f"{value['row_count']:,} cross-validated probe error row(s) "
                        "were retained for review."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Inspect these rows for label noise, overlapping classes, "
                        "or missing explanatory features."
                    ),
                )
            )
        elif item.kind == "classification_split_guidance":
            risk_kinds = {risk["kind"] for risk in value["risks"]}
            if "group_split_recommended" in risk_kinds:
                title = f"Group-aware validation recommended in {item.scope.table}"
            elif "time_split_recommended" in risk_kinds:
                title = f"Time-aware validation recommended in {item.scope.table}"
            else:
                title = f"Validation split review recommended in {item.scope.table}"
            findings.append(
                Finding.create(
                    title=title,
                    summary=(
                        f"{len(value['risks'])} split-design risk(s) were found "
                        "from the supplied analysis context."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Choose validation splits that respect entity grouping and "
                        "time ordering where applicable."
                    ),
                )
            )
    return sort_findings(findings), steps


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
        elif item.kind == "classification_identifier_feature":
            signal_rows.append(
                {
                    "signal": "identifier (exclude)",
                    "feature": item.scope.columns[0],
                    "metric": "unique count",
                    "score": item.value["unique_count"],
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "classification_probe_model":
            signal_rows.append(
                {
                    "signal": "probe separability",
                    "feature": f"{item.value['feature_count']} feature(s)",
                    "metric": "balanced accuracy",
                    "score": f"{item.value['balanced_accuracy']:.2f}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "classification_hard_examples":
            signal_rows.append(
                {
                    "signal": "hard examples",
                    "feature": "cross-validated errors",
                    "metric": "error rate",
                    "score": f"{item.value['error_rate']:.2f}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "classification_split_guidance":
            signal_rows.append(
                {
                    "signal": "split guidance",
                    "feature": ", ".join(
                        risk["column"] for risk in item.value["risks"]
                    ),
                    "metric": "risk count",
                    "score": len(item.value["risks"]),
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


def _classification_summary(
    table: str,
    target: str,
    findings: list[Finding],
    *,
    has_warnings: bool,
) -> str:
    """Lead with a decision verdict instead of a finding count."""
    suffix = " Sampling or recoverable caveats apply." if has_warnings else ""
    if not findings:
        return (
            f"{table}.{target} looks ready: no blocking classification risks "
            f"were found.{suffix}"
        )
    order = ("critical", "high", "medium", "low")
    counts = dict.fromkeys(order, 0)
    for finding in findings:
        if finding.severity in counts:
            counts[finding.severity] += 1
    breakdown = ", ".join(f"{counts[sev]} {sev}" for sev in order if counts[sev])
    top = findings[0]
    blocking = top.severity in {"critical", "high"}
    lead = "not ready to model" if blocking else "review before modeling"
    return (
        f"{table}.{target}: {lead}. Top issue — {top.title}. "
        f"{len(findings)} prioritized finding(s) ({breakdown}).{suffix}"
    )


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
            evidence.extend(
                _classification_probe_evidence(
                    frame,
                    table_catalog,
                    resolved_target,
                    config=config,
                    max_categories=max_categories,
                    prior_evidence=evidence,
                )
            )
            split_guidance = _split_guidance_evidence(
                frame,
                table_catalog,
                resolved_target,
                context=context,
            )
            if split_guidance is not None:
                evidence.append(split_guidance)
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
        summary = _classification_summary(
            table_catalog.name,
            resolved_target,
            findings,
            has_warnings=True,
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = _classification_summary(
            table_catalog.name,
            resolved_target,
            findings,
            has_warnings=False,
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
