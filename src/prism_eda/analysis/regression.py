"""Goal-aware deterministic diagnostics for regression EDA.

This module orchestrates the three question-modules and turns their evidence
into findings. The orchestration is the easy half. The half that decides whether
this report is useful is the split between an *issue* and an *alert*:

    issue  — something is wrong, or will mislead you. A leak. An identifier in
             the feature set. A target capped at a contract ceiling so the top
             of its range is fictional.
    alert  — true, worth knowing, not broken. The target is right-skewed. Two
             features are interchangeable. Errors are larger in one region.

A skewed target is not a defect, and filing it as one next to a leak devalues
both. So skew is an alert, and the transformation that would fix it is measured
rather than asserted. What promotes a shape observation to an issue is measured
harm — a censoring spike, or conditional bias large enough to distort the fit —
and the harm is computed, not assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from prism_eda.analysis._regression import (
    MIN_CONTINUOUS_DISTINCT,
    MIN_PROBE_ROWS,
    feature_groups,
    identifier_feature_names,
    leakage_feature_names,
    numeric_target,
    resolve_table,
    sample_frame,
)
from prism_eda.analysis.regression_probe import (
    error_concentration_evidence,
    influence_evidence,
    probe_evidence,
    probe_is_outlier_sensitive,
    probe_is_weak,
    residual_evidence,
    residual_scatter_evidence,
    weak_support_evidence,
)
from prism_eda.analysis.regression_signal import (
    association_evidence,
    identifier_and_leakage_evidence,
    redundancy_evidence,
)
from prism_eda.analysis.regression_target import target_evidence
from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog, TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    OBSERVATION,
    Evidence,
    EvidenceScope,
    Finding,
    sort_findings,
    split_findings,
)
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

# Thresholds that decide whether an observation is worth a reader's attention.
_HETEROSCEDASTICITY_RATIO = 3.0
_CONDITIONAL_BIAS_RATIO = 0.5
_ERROR_CONCENTRATION_RATIO = 1.5
_SPIKE_HIGH_RATE = 0.05
_MISSING_TARGET_HIGH = 0.2
_MISSING_TARGET_MEDIUM = 0.05

#: Non-linear association alerts, capped. Every numeric feature in a wide table
#: can look slightly curved; only the strongest few are worth the line.
_MAX_NONLINEAR_ALERTS = 3

#: Columns offered to the subgroup error breakdown.
_MAX_GROUP_COLUMNS = 6
_MAX_GROUP_LEVELS = 20


def _split_guidance_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
    *,
    context: AnalysisContext,
) -> Evidence | None:
    """Group and time risks that make a random split optimistic."""
    risks: list[dict[str, Any]] = []
    if context.entity_id and context.entity_id in frame.columns:
        counts = frame[context.entity_id].value_counts(dropna=False)
        repeated = counts[counts > 1]
        repeated_rows = int(frame[context.entity_id].isin(repeated.index).sum())
        if repeated_rows:
            risks.append(
                {
                    "kind": "group_split_recommended",
                    "column": context.entity_id,
                    "repeated_entity_count": int(len(repeated)),
                    "repeated_row_count": repeated_rows,
                    "repeated_row_rate": (
                        repeated_rows / len(frame) if len(frame) else 0.0
                    ),
                    "reason": (
                        "Rows from the same entity can leak information across a "
                        "random train/test split."
                    ),
                }
            )
    if context.timestamp and context.timestamp in frame.columns:
        timestamps = pd.to_datetime(frame[context.timestamp], errors="coerce")
        valid = pd.DataFrame(
            {"timestamp": timestamps, "target": target_series}
        ).dropna()
        if len(valid) >= 20 and valid["timestamp"].nunique() >= 5:
            ordered = valid.sort_values("timestamp")
            window = max(1, len(ordered) // 4)
            early = float(ordered.head(window)["target"].mean())
            late = float(ordered.tail(window)["target"].mean())
            spread = float(ordered["target"].std(ddof=1)) or 1.0
            span_days = (
                ordered["timestamp"].max() - ordered["timestamp"].min()
            ).total_seconds() / 86_400
            risks.append(
                {
                    "kind": "time_split_recommended",
                    "column": context.timestamp,
                    "span_days": float(span_days),
                    "early_mean_target": early,
                    "late_mean_target": late,
                    # Drift expressed in residual-scale units, so it is
                    # comparable across targets with different magnitudes.
                    "target_drift_in_std": abs(late - early) / spread,
                    "reason": (
                        "A timestamped target should be validated with temporal "
                        "order preserved."
                    ),
                }
            )
    if not risks:
        return None
    return Evidence.create(
        kind="regression_split_guidance",
        scope=EvidenceScope(table=table.name, columns=(target,)),
        value={"risks": risks},
        method="context_aware_split_guidance_v1",
        description=f"Split-design guidance for {table.name}.{target}.",
        confidence=0.82,
        assumptions=(
            "Split guidance uses the entity and timestamp columns supplied in "
            "the analysis context.",
        ),
    )


def _group_columns(
    frame: pd.DataFrame,
    context: AnalysisContext,
    categorical_features: list[str],
) -> list[str]:
    candidates: list[str] = [
        column for column in context.groups if column in frame.columns
    ]
    for name in categorical_features:
        if name in candidates:
            continue
        if frame[name].nunique(dropna=True) <= _MAX_GROUP_LEVELS:
            candidates.append(name)
    return candidates[:_MAX_GROUP_COLUMNS]


def _target_of(item: Evidence) -> str:
    """The target column an evidence item is scoped to."""
    return item.scope.columns[0] if item.scope.columns else "the target"


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _spike_title(spike: dict[str, Any], target: str) -> tuple[str, str]:
    position = spike["position"]
    value = spike["value"]
    where = f"in {spike['count']:,} rows ({spike['rate']:.1%})"
    if position == "at_zero":
        return (
            f"Zero-inflated target: {target}",
            f"{target} is exactly zero {where}.",
        )
    if position == "at_maximum":
        return (
            f"Target capped at its maximum: {target}",
            f"{target} sits at its highest observed value, {value:,.4g}, {where}.",
        )
    if position == "at_minimum":
        return (
            f"Target floored at its minimum: {target}",
            f"{target} sits at its lowest observed value, {value:,.4g}, {where}.",
        )
    return (
        f"Target values pile up at {value:,.4g}",
        f"{target} takes the single value {value:,.4g} {where}.",
    )


def _findings_and_steps(
    evidence: list[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    """Turn evidence into a short, prioritized, decision-first list."""
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    nonlinear: list[Evidence] = []

    for item in evidence:
        value = item.value
        table = item.scope.table or ""

        if item.kind == "regression_target_summary":
            rate = value["missing_rate"]
            if rate > 0:
                severity = (
                    "high"
                    if rate > _MISSING_TARGET_HIGH
                    else "medium"
                    if rate > _MISSING_TARGET_MEDIUM
                    else "low"
                )
                findings.append(
                    Finding.create(
                        title=f"Missing target values in {table}",
                        summary=(
                            f"{value['missing_count']:,} row(s) ({rate:.1%}) have no "
                            f"{value['target']} value and cannot be trained on."
                        ),
                        severity=severity,
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Decide whether these rows are unlabeled, not yet "
                            "measured, or genuinely zero before dropping them."
                        ),
                    )
                )
            if (
                value["looks_discrete"]
                and 2 <= value["distinct_count"] < MIN_CONTINUOUS_DISTINCT
            ):
                findings.append(
                    Finding.create(
                        title=f"Target takes few distinct values: {value['target']}",
                        summary=(
                            f"{value['target']} has only "
                            f"{value['distinct_count']} distinct values. Regression "
                            "still applies to a rating or a count, but a squared-"
                            "error fit will predict between values that never occur."
                        ),
                        severity="low",
                        confidence=0.9,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Consider ordinal or classification framing if the "
                            "values are categories rather than quantities."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "regression_target_shape":
            if value["shape"] == "symmetric":
                continue
            best = value["best_candidate"]
            detail = (
                f" A {best} transform reduces skew to "
                f"{value['best_skewness_after']:+.2f}."
                if best
                else " No standard transform measurably improves it."
            )
            findings.append(
                Finding.create(
                    title=f"Target is {value['shape'].replace('_', ' ')}",
                    summary=(
                        f"{value['target']} has skewness "
                        f"{value['skewness']:+.2f}.{detail}"
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Skew is a property, not a defect. Transform only if the "
                        "residual diagnostics below show the fit suffering for it."
                    ),
                    category=OBSERVATION,
                )
            )
            if best:
                steps.append(
                    TransformationStep(
                        operation="consider_target_transformation",
                        table=table,
                        columns=item.scope.columns,
                        parameters={
                            "transform": best,
                            "skewness_before": value["skewness"],
                            "skewness_after": value["best_skewness_after"],
                        },
                        rationale=(
                            "A transformed target changes what the errors mean; "
                            "back-transformed predictions need a bias correction."
                        ),
                        evidence_ids=(item.id,),
                        risk="medium",
                    )
                )

        elif item.kind == "regression_target_spikes":
            top = max(value["spikes"], key=lambda spike: spike["rate"])
            title, summary = _spike_title(top, value["target"])
            findings.append(
                Finding.create(
                    title=title,
                    summary=(
                        f"{summary} A continuous quantity should not repeat like "
                        "this; a cap, a default, or an imputed placeholder is the "
                        "usual cause."
                    ),
                    severity="high" if top["rate"] >= _SPIKE_HIGH_RATE else "medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm how the value is recorded. If it is a ceiling, "
                        "those rows are censored and the model cannot learn the "
                        "range beyond it."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_target_censoring",
                    table=table,
                    columns=item.scope.columns,
                    parameters={"spikes": value["spikes"]},
                    rationale=(
                        "Censored targets bias every fit toward the censored value."
                    ),
                    evidence_ids=(item.id,),
                    risk="high",
                )
            )

        elif item.kind == "regression_target_heaping":
            findings.append(
                Finding.create(
                    title=f"Round-number heaping in {value['target']}",
                    summary=(
                        f"{value['multiple_rate']:.0%} of values are exact multiples "
                        f"of {value['base']:,.0f}, so the target is coarser than its "
                        "decimals suggest."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Treat sub-unit precision in predictions as spurious."
                    ),
                    category=OBSERVATION,
                )
            )

        elif item.kind == "regression_leakage_candidate":
            explained = value["explained_variance"]
            basis = (
                f"explains {explained:.1%} of the target's variance on its own"
                if explained is not None
                else "shares a name with the target"
            )
            findings.append(
                Finding.create(
                    title=f"Potential target leakage: {value['feature']}",
                    summary=f"{value['feature']} {basis}.",
                    severity="critical" if value["near_perfect"] else "high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm the value exists before the outcome is known. A "
                        "feature derived from the target makes every validation "
                        "score meaningless."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_target_leakage_candidate",
                    table=table,
                    columns=item.scope.columns,
                    parameters=value,
                    rationale="Leaky features make validation scores unrealistic.",
                    evidence_ids=(item.id,),
                    risk="high",
                )
            )

        elif item.kind == "regression_identifier_feature":
            findings.append(
                Finding.create(
                    title=f"Identifier-like feature: {value['feature']}",
                    summary=(
                        f"{value['feature']} is unique on "
                        f"{value['unique_rate']:.0%} of rows and labels records "
                        "rather than explaining the target."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Exclude identifier-like columns from the feature set."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="exclude_identifier_feature",
                    table=table,
                    columns=item.scope.columns,
                    parameters=value,
                    rationale="Identifier columns memorize rows.",
                    evidence_ids=(item.id,),
                    risk="high",
                )
            )

        elif item.kind == "regression_redundancy":
            pairs = value["redundant_pairs"]
            if not pairs:
                continue
            top = pairs[0]
            remaining = len(pairs) - 1
            extra = (
                f" and {remaining} more {_plural(remaining, 'pair', 'pairs')}"
                if remaining
                else ""
            )
            findings.append(
                Finding.create(
                    title="Redundant features carry the same information",
                    summary=(
                        f"{top['left']} and {top['right']} correlate at "
                        f"{top['abs_correlation']:.4f}{extra}. Predictions are "
                        "largely unaffected; individual coefficients are not "
                        "interpretable."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Keep one of each pair, or use a regularized model and "
                        "stop reading individual coefficients."
                    ),
                    category=OBSERVATION,
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_redundant_features",
                    table=table,
                    columns=tuple(
                        {pair["left"] for pair in pairs}
                        | {pair["right"] for pair in pairs}
                    ),
                    parameters={"pairs": pairs},
                    rationale=(
                        "Collinear features make coefficient signs unstable across "
                        "refits."
                    ),
                    evidence_ids=(item.id,),
                    risk="low",
                )
            )

        elif item.kind == "regression_feature_association" and value["is_nonlinear"]:
            nonlinear.append(item)

        elif item.kind == "regression_probe":
            if probe_is_weak(item):
                findings.append(
                    Finding.create(
                        title=f"Features carry little signal for {value['target']}",
                        summary=(
                            "The best diagnostic probe reached R² "
                            f"{value['best_r_squared']:.2f} across "
                            f"{value['cv_folds']} folds — no better than predicting "
                            "the median."
                        ),
                        severity="high",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Neither probe found usable structure. Look for a "
                            "missing feature or a finer row granularity before "
                            "modelling."
                        ),
                    )
                )
            elif probe_is_outlier_sensitive(item):
                findings.append(
                    Finding.create(
                        title="A few rows distort the conventional fit",
                        summary=(
                            "The robust probe's typical error is "
                            f"{value['robust_median_error_gain']:.0%} lower than the "
                            "least-squares probe's. The data is predictable; a "
                            "minority of rows is pulling a squared-error fit."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Work the review rows below, or fit a robust loss. "
                            "Adding features will not fix this."
                        ),
                    )
                )

        elif item.kind == "regression_heteroscedasticity":
            ratio = value["spread_ratio"]
            if ratio is None or ratio < _HETEROSCEDASTICITY_RATIO:
                continue
            findings.append(
                Finding.create(
                    title="Error spread changes across the prediction range",
                    summary=(
                        f"Residual spread varies {ratio:.1f}x between the widest and "
                        "narrowest part of the fitted range."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "One prediction interval will be wrong at both ends. Use "
                        "interval estimates that vary with the prediction."
                    ),
                )
            )

        elif item.kind == "regression_conditional_bias":
            ratio = value["max_bias_ratio"]
            if ratio < _CONDITIONAL_BIAS_RATIO:
                continue
            findings.append(
                Finding.create(
                    title="The fit is systematically biased in part of the range",
                    summary=(
                        f"At least one fitted decile is off by {ratio:.2f} residual "
                        "standard deviations on average — a consistent direction, "
                        "not noise."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Usually a curved relationship or a censored target forced "
                        "through a straight line. Check the target shape and the "
                        "non-linear associations above."
                    ),
                )
            )

        elif item.kind == "regression_error_concentration":
            ratio = value["max_error_ratio"]
            if ratio < _ERROR_CONCENTRATION_RATIO:
                continue
            worst = value["groups"][0]
            findings.append(
                Finding.create(
                    title=f"Error concentrates in {worst['column']}={worst['level']}",
                    summary=(
                        f"Scaled against its own spread, that group's error is "
                        f"{ratio:.1f}x the typical {worst['column']} group, across "
                        f"{worst['row_count']:,} rows."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Report error per group rather than one headline number, "
                        "and check whether the group is under-represented."
                    ),
                    category=OBSERVATION,
                )
            )

        elif item.kind == "regression_review_rows":
            count = value["row_count"]
            findings.append(
                Finding.create(
                    title=(f"{count} {_plural(count, 'row', 'rows')} worth reviewing"),
                    summary=(
                        "These rows are extreme in the features, badly fitted, or "
                        "both, so they move the fit more than any other rows."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Open them before modelling: a data-entry error and a "
                        "genuinely unusual customer look identical here and need "
                        "opposite treatment."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_influential_rows",
                    table=table,
                    columns=item.scope.columns,
                    parameters={"row_count": value["row_count"]},
                    rationale=(
                        "Influential rows change the fitted coefficients out of "
                        "proportion to their number."
                    ),
                    evidence_ids=(item.id,),
                    risk="medium",
                )
            )

        elif item.kind == "regression_weak_support":
            sparse = value["sparse_target_bins"]
            gaps = value["feature_gaps"]
            if not sparse and not gaps:
                continue
            if sparse:
                detail = (
                    f"{len(sparse)} interior "
                    f"{_plural(len(sparse), 'stretch', 'stretches')} of the target "
                    f"range {_plural(len(sparse), 'holds', 'hold')} almost no rows"
                )
            else:
                detail = (
                    f"{gaps[0]['feature']} has a gap covering "
                    f"{gaps[0]['gap_fraction']:.0%} of its range"
                )
            findings.append(
                Finding.create(
                    title="Parts of the range have almost no support",
                    summary=(
                        f"{detail}. Predictions there are extrapolation, however "
                        "confident the model looks."
                    ),
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Constrain predictions to the supported range, or gather "
                        "examples in the gap."
                    ),
                    category=OBSERVATION,
                )
            )

        elif item.kind == "regression_split_guidance":
            for risk in value["risks"]:
                if risk["kind"] == "group_split_recommended":
                    findings.append(
                        Finding.create(
                            title=f"Repeated entities in {risk['column']}",
                            summary=(
                                f"{risk['repeated_row_count']:,} row(s) "
                                f"({risk['repeated_row_rate']:.1%}) share an entity "
                                "with another row, so a random split puts the same "
                                "entity on both sides."
                            ),
                            severity="high",
                            confidence=item.confidence,
                            evidence_ids=(item.id,),
                            recommendation=(
                                f"Split by {risk['column']} with GroupKFold rather "
                                "than at random."
                            ),
                        )
                    )
                else:
                    findings.append(
                        Finding.create(
                            title=(
                                f"Time-ordered validation needed for {_target_of(item)}"
                            ),
                            summary=(
                                f"The target's mean moves "
                                f"{risk['target_drift_in_std']:.2f} standard "
                                f"deviations across {risk['span_days']:,.0f} day(s) "
                                "of history."
                            ),
                            severity="medium",
                            confidence=item.confidence,
                            evidence_ids=(item.id,),
                            recommendation=(
                                "Validate with temporal order preserved; a random "
                                "split trains on the future."
                            ),
                        )
                    )

    # Two features that are 0.999 correlated carry one relationship, not two.
    # Reporting the same curve once per redundant twin is exactly the kind of
    # duplicated line that makes an alert list stop being read, so a feature is
    # skipped once something it duplicates has already been reported.
    redundant_with: dict[str, set[str]] = {}
    for item in evidence:
        if item.kind != "regression_redundancy":
            continue
        for pair in item.value["redundant_pairs"]:
            redundant_with.setdefault(pair["left"], set()).add(pair["right"])
            redundant_with.setdefault(pair["right"], set()).add(pair["left"])

    reported: set[str] = set()
    for item in sorted(
        nonlinear,
        key=lambda entry: entry.value["nonlinearity_gap"] or 0.0,
        reverse=True,
    ):
        if len(reported) >= _MAX_NONLINEAR_ALERTS:
            break
        feature = item.value["feature"]
        if redundant_with.get(feature, set()) & reported:
            continue
        reported.add(feature)
        value = item.value
        findings.append(
            Finding.create(
                title=f"{value['feature']} relates to the target non-linearly",
                summary=(
                    f"A straight line explains {value['linear_r_squared']:.1%} of the "
                    f"target, but a binned fit explains "
                    f"{value['binned_eta_squared']:.1%}. A linear model will "
                    "under-use this feature."
                ),
                severity="low",
                confidence=item.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Add a spline, a bucketed version, or use a model that fits curves."
                ),
                category=OBSERVATION,
            )
        )

    return sort_findings(findings), steps


def _artifacts(evidence: list[Evidence]) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []

    probes = [item for item in evidence if item.kind == "regression_probe"]
    if probes:
        item = probes[0]
        rows = [
            {
                "model": entry["model"].replace("_", " "),
                "r_squared": f"{entry['r_squared']:.3f}",
                "mae": f"{entry['mae']:,.2f}",
                "rmse": f"{entry['rmse']:,.2f}",
                "median_ae": f"{entry['median_ae']:,.2f}",
            }
            for entry in [*item.value["models"], item.value["baseline"]]
        ]
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Diagnostic probes",
                data={
                    "columns": [
                        {"key": "model", "label": "Model"},
                        {"key": "r_squared", "label": "R²"},
                        {"key": "mae", "label": "MAE"},
                        {"key": "rmse", "label": "RMSE"},
                        {"key": "median_ae", "label": "Median AE"},
                    ],
                    "rows": rows,
                },
                evidence_ids=(item.id,),
                metadata={
                    "description": (
                        "Cross-validated diagnostic fits against a median "
                        "baseline. These are readiness probes, not candidate "
                        "production models."
                    )
                },
            )
        )

    signal_rows: list[dict[str, Any]] = []
    for item in evidence:
        value = item.value
        if item.kind == "regression_feature_association":
            signal_rows.append(
                {
                    "signal": "numeric association",
                    "feature": value["feature"],
                    "metric": "linear R² / binned η²",
                    "score": (
                        f"{value['linear_r_squared']:.3f} / "
                        f"{(value['binned_eta_squared'] or 0.0):.3f}"
                    ),
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "regression_categorical_association":
            signal_rows.append(
                {
                    "signal": "categorical association",
                    "feature": value["feature"],
                    "metric": "correlation ratio",
                    "score": f"{value['correlation_ratio']:.3f}",
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "regression_leakage_candidate":
            explained = value["explained_variance"]
            signal_rows.append(
                {
                    "signal": "leakage candidate",
                    "feature": value["feature"],
                    "metric": "explained variance",
                    "score": (
                        f"{explained:.3f}" if explained is not None else "name match"
                    ),
                    "confidence": f"{item.confidence:.0%}",
                }
            )
        elif item.kind == "regression_identifier_feature":
            signal_rows.append(
                {
                    "signal": "identifier (exclude)",
                    "feature": value["feature"],
                    "metric": "unique count",
                    "score": str(value["unique_count"]),
                    "confidence": f"{item.confidence:.0%}",
                }
            )
    if signal_rows:
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Feature signal",
                data={
                    "columns": [
                        {"key": "signal", "label": "Signal"},
                        {"key": "feature", "label": "Feature"},
                        {"key": "metric", "label": "Metric"},
                        {"key": "score", "label": "Score"},
                        {"key": "confidence", "label": "Confidence"},
                    ],
                    "rows": signal_rows,
                },
                evidence_ids=tuple(
                    item.id
                    for item in evidence
                    if item.kind
                    in {
                        "regression_feature_association",
                        "regression_categorical_association",
                        "regression_leakage_candidate",
                        "regression_identifier_feature",
                    }
                ),
                metadata={
                    "description": (
                        "Association measured linearly and in bins, so a curved "
                        "relationship is not reported as no relationship."
                    )
                },
            )
        )

    redundancy = [item for item in evidence if item.kind == "regression_redundancy"]
    if redundancy and redundancy[0].value["variance_inflation"]:
        item = redundancy[0]
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Multicollinearity",
                data={
                    "columns": [
                        {"key": "feature", "label": "Feature"},
                        {"key": "vif", "label": "VIF"},
                    ],
                    "rows": [
                        {"feature": entry["feature"], "vif": f"{entry['vif']:,.2f}"}
                        for entry in item.value["variance_inflation"]
                    ],
                },
                evidence_ids=(item.id,),
                metadata={
                    "description": (
                        "Variance inflation factors, reported as measurements. "
                        "There is no universal value at which VIF becomes a "
                        "defect; the redundant pairs above are the actionable "
                        "form of the same information."
                    )
                },
            )
        )
    return tuple(artifacts)


def _regression_summary(
    table: str,
    target: str,
    findings: list[Finding],
    *,
    has_warnings: bool,
) -> str:
    """Lead with a verdict, and let alerts stay out of the verdict."""
    suffix = " Sampling or recoverable caveats apply." if has_warnings else ""
    issues, observations = split_findings(findings)
    alerts = (
        f" {len(observations)} alert(s) are listed separately." if observations else ""
    )
    if not issues:
        return (
            f"{table}.{target} looks ready: no blocking regression risks were "
            f"found.{alerts}{suffix}"
        )
    order = ("critical", "high", "medium", "low")
    counts = dict.fromkeys(order, 0)
    for finding in issues:
        if finding.severity in counts:
            counts[finding.severity] += 1
    breakdown = ", ".join(f"{counts[key]} {key}" for key in order if counts[key])
    top = issues[0]
    lead = (
        "not ready to model"
        if top.severity in {"critical", "high"}
        else "review before modeling"
    )
    return (
        f"{table}.{target}: {lead}. Top issue — {top.title}. "
        f"{len(issues)} prioritized issue(s) ({breakdown}).{alerts}{suffix}"
    )


def regression_dataset(
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
    """Run deterministic regression diagnostics for one numeric target."""
    emit(
        callbacks,
        Event(
            EventKind.RUN_STARTED,
            "Regression diagnostics started.",
            stage="regression",
        ),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    resolved_target = target or context.target
    table_catalog, resolved_target = resolve_table(
        catalog,
        tables,
        table=table,
        target=resolved_target,
        warnings=warnings,
    )

    evidence: list[Evidence] = []
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    usable = False

    if table_catalog is not None and resolved_target is not None:
        frame = sample_frame(
            tables[table_catalog.name],
            table=table_catalog.name,
            config=config,
            warnings=warnings,
            sampling=sampling,
        )
        target_series = numeric_target(frame, resolved_target)
        values = target_series.to_numpy(dtype="float64")
        observed = int(target_series.notna().sum())
        usable = observed >= MIN_PROBE_ROWS and target_series.nunique() >= 2

        evidence.extend(
            target_evidence(frame, table_catalog.name, resolved_target, values)
        )
        if usable:
            evidence.extend(
                identifier_and_leakage_evidence(
                    frame, table_catalog, resolved_target, target_series
                )
            )
            excluded = leakage_feature_names(evidence) | identifier_feature_names(
                evidence
            )
            numeric_features, categorical_features, dropped = feature_groups(
                frame,
                table_catalog,
                resolved_target,
                max_categories=max_categories,
                excluded_features=excluded,
            )
            evidence.extend(
                association_evidence(
                    frame,
                    table_catalog,
                    resolved_target,
                    target_series,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                )
            )
            redundancy = redundancy_evidence(
                frame, table_catalog, numeric_features=numeric_features
            )
            if redundancy is not None:
                evidence.append(redundancy)

            probe, predictions = probe_evidence(
                frame,
                table_catalog,
                resolved_target,
                target_series,
                config=config,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                excluded_features=dropped,
            )
            if probe is not None and predictions is not None:
                evidence.append(probe)
                actual = target_series.loc[predictions.index]
                evidence.extend(
                    residual_evidence(
                        table_catalog, resolved_target, actual, predictions, probe
                    )
                )
                concentration = error_concentration_evidence(
                    frame,
                    table_catalog,
                    resolved_target,
                    actual,
                    predictions,
                    group_columns=_group_columns(frame, context, categorical_features),
                )
                if concentration is not None:
                    evidence.append(concentration)
                evidence.extend(
                    influence_evidence(
                        frame,
                        table_catalog,
                        resolved_target,
                        actual,
                        predictions,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    )
                )
                # Built last so the scatter can mark the rows the influence
                # stage singled out: the reader sees which points the review
                # table is talking about, rather than a cloud and a list.
                flagged = {
                    row["row_index"]
                    for item in evidence
                    if item.kind == "regression_review_rows"
                    for row in item.value["rows"]
                }
                spread_bins: list[dict[str, Any]] = next(
                    (
                        item.value["bins"]
                        for item in evidence
                        if item.kind == "regression_heteroscedasticity"
                    ),
                    [],
                )
                scatter = residual_scatter_evidence(
                    table_catalog,
                    resolved_target,
                    actual,
                    predictions,
                    config=config,
                    bins=spread_bins,
                    flagged=flagged,
                )
                if scatter is not None:
                    evidence.append(scatter)
            support = weak_support_evidence(
                frame,
                table_catalog,
                resolved_target,
                target_series,
                numeric_features=numeric_features,
            )
            if support is not None:
                evidence.append(support)
            guidance = _split_guidance_evidence(
                frame,
                table_catalog,
                resolved_target,
                target_series,
                context=context,
            )
            if guidance is not None:
                evidence.append(guidance)
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
        summary = "Regression analysis needs one numeric target column in one table."
    elif not usable:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = (
            f"{table_catalog.name}.{resolved_target} does not have enough varying "
            f"numeric values for regression diagnostics (at least "
            f"{MIN_PROBE_ROWS} rows with two or more distinct values are needed)."
        )
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = _regression_summary(
            table_catalog.name, resolved_target, findings, has_warnings=True
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = _regression_summary(
            table_catalog.name, resolved_target, findings, has_warnings=False
        )

    result = AnalysisResult(
        goal="regression",
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
            stage="regression",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
