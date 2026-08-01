"""How columns relate, what that implies, and the structure of what is missing.

This module produces the profile's *observations*: statements that are true and
worth knowing but are not defects. "Age and Experience_Years move together" does
not mean either column is broken; it means a model fed both is fed one variable
twice. Keeping these out of the data-quality channel is what lets that channel
stay short enough to read.

Association is measured with the right statistic for each pair of types, and the
statistic used is recorded next to the number:

* numeric v numeric  - Spearman rho (reported alongside Pearson r), because a
  monotone-but-curved relationship is still a relationship.
* categorical v categorical - Cramer's V with the Bergsma bias correction, which
  matters at the small contingency tables profiling routinely produces.
* categorical v numeric - the correlation ratio eta, the share of the numeric
  column's variance explained by group membership.

All three land in [0, 1] so a single matrix can hold them, but they are not the
same quantity and the report says which is which.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from prism_eda._serialization import to_jsonable
from prism_eda.analysis._limits import ProfileLimits
from prism_eda.catalog.models import TableCatalog
from prism_eda.evidence.models import OBSERVATION, Evidence, EvidenceScope, Finding
from prism_eda.results import AnalysisWarning, SamplingRecord

# Strength at which a pair is worth telling the analyst about. Below this,
# reporting the pair is reporting noise.
HIGH_ASSOCIATION = 0.80
# At this strength one column is very nearly a restatement of the other.
REDUNDANT_ASSOCIATION = 0.99
# How many association observations are promoted before the list stops being a
# short list. The full matrix is always in the report.
MAX_ASSOCIATION_ALERTS = 8

# A categorical needs at least two groups and enough rows per group before an
# association measure means anything.
MIN_ASSOCIATION_ROWS = 20
# Categorical columns wider than this are not usable as grouping variables: the
# contingency table becomes mostly empty cells and every measure inflates.
MAX_ASSOCIATION_CATEGORIES = 50

# A categorical this dominated by one label carries almost no information.
DOMINANT_LABEL_SHARE = 0.90
# Distinct-value count at which a "category" is really an identifier or free text.
HIGH_CARDINALITY_CATEGORIES = 100

# A timestamp gap this many times the typical gap is a hole in coverage rather
# than ordinary irregularity.
TIMELINE_GAP_RATIO = 3.0
MAX_REPORTED_GAPS = 5

# Pairs whose missing values coincide at least this often are missing together
# for a reason, which is the question an analyst actually needs answered.
CO_MISSING_MIN_JACCARD = 0.30


def _sampled(
    frame: pd.DataFrame,
    *,
    table: str,
    limits: ProfileLimits,
    seed: int,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> pd.DataFrame:
    """Down-sample for the association work, on the record."""
    if len(frame) <= limits.association_rows:
        return frame
    sampled = frame.sample(n=limits.association_rows, random_state=seed).sort_index()
    warnings.append(
        AnalysisWarning(
            code="sampled_associations",
            message=(
                f"{table} has {len(frame):,} rows; correlations and scatter plots "
                f"were computed on a deterministic {limits.association_rows:,}-row "
                "sample. Data-quality checks still used every row."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="profile_associations",
            source_rows=len(frame),
            sampled_rows=limits.association_rows,
            strategy="deterministic_pandas_sample",
            seed=seed,
            reason="row_count_exceeds_association_budget",
            limitations=(
                "Associations driven by a small subpopulation may be understated.",
            ),
        )
    )
    return sampled


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    """Bias-corrected Cramer's V for two categorical columns."""
    table = pd.crosstab(left, right)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    total = float(table.to_numpy().sum())
    if total < MIN_ASSOCIATION_ROWS:
        return None
    try:
        chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    except ValueError:
        return None
    phi2 = chi2 / total
    rows, columns = table.shape
    # Bergsma's correction: without it V is biased upward, and profiling produces
    # exactly the small, wide tables where that bias is largest.
    phi2_corrected = max(0.0, phi2 - (columns - 1) * (rows - 1) / (total - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (total - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (total - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    if denominator <= 0:
        return None
    value = float(np.sqrt(phi2_corrected / denominator))
    return min(1.0, value)


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Correlation ratio eta: variance in `values` explained by group membership."""
    frame = pd.DataFrame(
        {"group": categories, "value": pd.to_numeric(values, errors="coerce")}
    )
    frame = frame.dropna()
    if len(frame) < MIN_ASSOCIATION_ROWS or frame["group"].nunique() < 2:
        return None
    grand_mean = float(frame["value"].mean())
    grouped = frame.groupby("group", observed=True)["value"]
    between = float(
        sum(
            len(group) * (float(group.mean()) - grand_mean) ** 2 for _, group in grouped
        )
    )
    total = float(((frame["value"] - grand_mean) ** 2).sum())
    if total <= 0:
        return None
    return float(np.sqrt(min(1.0, between / total)))


def _association_columns(
    frame: pd.DataFrame, table: TableCatalog, limits: ProfileLimits
) -> tuple[list[str], list[str], list[str]]:
    """Split usable columns into numeric and categorical, honouring the cap."""
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []
    for column in table.columns:
        if column.name not in frame.columns:
            continue
        if len(numeric) + len(categorical) >= limits.correlation_columns:
            skipped.append(column.name)
            continue
        if column.unique_count is not None and column.unique_count < 2:
            continue
        if column.semantic_type == "numeric":
            numeric.append(column.name)
        elif column.semantic_type in {"categorical", "boolean"} and (
            column.unique_count is not None
            and column.unique_count <= MAX_ASSOCIATION_CATEGORIES
        ):
            categorical.append(column.name)
    return numeric, categorical, skipped


def association_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    limits: ProfileLimits,
) -> tuple[Evidence | None, list[str]]:
    """One association strength per column pair, each with its own method."""
    numeric, categorical, skipped = _association_columns(frame, table, limits)
    columns = numeric + categorical
    if len(columns) < 2:
        return None, skipped

    spearman = pd.DataFrame()
    pearson = pd.DataFrame()
    if len(numeric) >= 2:
        numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
        spearman = numeric_frame.corr(method="spearman")
        pearson = numeric_frame.corr(method="pearson")

    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(columns, 2):
        entry: dict[str, Any] | None = None
        if left in numeric and right in numeric:
            if spearman.empty:
                continue
            raw_rho = spearman.loc[left, right]
            if pd.isna(raw_rho):
                continue
            rho = float(raw_rho)  # type: ignore[arg-type]
            raw_r = pearson.loc[left, right]
            entry = {
                "strength": abs(rho),
                "method": "spearman",
                "spearman": rho,
                "pearson": None if pd.isna(raw_r) else float(raw_r),  # type: ignore[arg-type]
                "signed": True,
            }
        elif left in categorical and right in categorical:
            value = _cramers_v(frame[left], frame[right])
            if value is None:
                continue
            entry = {"strength": value, "method": "cramers_v", "signed": False}
        else:
            group, measure = (left, right) if left in categorical else (right, left)
            value = _correlation_ratio(frame[group], frame[measure])
            if value is None:
                continue
            entry = {
                "strength": value,
                "method": "correlation_ratio",
                "grouping_column": group,
                "signed": False,
            }
        # Two columns holding the same values are already reported, precisely,
        # as a duplicate-columns issue. Saying it again here as "these are
        # strongly associated" is the same fact stated more vaguely, in a second
        # place. The matrix still shows the cell; only the alert is suppressed.
        identical = False
        if float(entry["strength"]) >= REDUNDANT_ASSOCIATION:
            try:
                identical = bool(frame[left].equals(frame[right]))
            except (TypeError, ValueError):
                identical = False
        pairs.append(
            {"left": left, "right": right, "identical_columns": identical, **entry}
        )

    if not pairs:
        return None, skipped
    pairs.sort(key=lambda item: -float(item["strength"]))
    return (
        Evidence.create(
            kind="profile_association_matrix",
            scope=EvidenceScope(table=table.name, columns=tuple(columns)),
            value={
                "columns": columns,
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "row_count": int(len(frame)),
                "pairs": pairs,
            },
            method="spearman_cramersv_correlation_ratio_v1",
            description=f"Pairwise association strengths for {table.name}.",
            confidence=0.9 if len(frame) >= 100 else 0.7,
            assumptions=(
                "Three different statistics share one 0-1 scale so that mixed "
                "column types can be compared; each pair records which was used.",
                "Association is not causation, and a strong pair may be driven "
                "by a third column.",
            ),
        ),
        skipped,
    )


def _association_findings(
    matrix: Evidence, table: str, confidence: float
) -> list[Finding]:
    """Promote the strongest pairs, and only those, to observations."""
    findings: list[Finding] = []
    strong = [
        pair
        for pair in matrix.value["pairs"]
        if float(pair["strength"]) >= HIGH_ASSOCIATION
        and not pair.get("identical_columns")
    ]
    for pair in strong[:MAX_ASSOCIATION_ALERTS]:
        strength = float(pair["strength"])
        left, right = pair["left"], pair["right"]
        redundant = strength >= REDUNDANT_ASSOCIATION
        if pair["method"] == "spearman":
            direction = "together" if float(pair["spearman"]) > 0 else "in opposition"
            detail = f"Spearman {float(pair['spearman']):+.2f}"
            summary = f"{left} and {right} move {direction} ({detail})."
        elif pair["method"] == "cramers_v":
            summary = (
                f"Knowing {left} tells you most of {right} (Cramer's V {strength:.2f})."
            )
        else:
            summary = (
                f"{pair['grouping_column']} explains {strength**2:.0%} of the "
                f"variance in "
                f"{right if pair['grouping_column'] == left else left} "
                f"(correlation ratio {strength:.2f})."
            )
        if redundant:
            summary += " At this strength the two are effectively one variable."
        findings.append(
            Finding.create(
                title=f"{left} and {right} are strongly associated in {table}",
                summary=summary,
                severity="info",
                confidence=confidence,
                evidence_ids=(matrix.id,),
                category=OBSERVATION,
                recommendation=(
                    "Drop or combine one of the two before modelling; both "
                    "carry the same signal and together they distort "
                    "coefficient estimates."
                    if redundant
                    else "Worth knowing before feature selection: these two are "
                    "not independent inputs."
                ),
            )
        )
    return findings


def _column_findings(
    frame: pd.DataFrame, table: TableCatalog, column_evidence: dict[str, str]
) -> list[Finding]:
    """Identity and cardinality observations, per column."""
    findings: list[Finding] = []
    for column in table.columns:
        evidence_id = column_evidence.get(column.name)
        if evidence_id is None or not column.non_null_count:
            continue
        if column.unique_count == column.row_count and column.row_count > 1:
            findings.append(
                Finding.create(
                    title=f"{table.name}.{column.name} has all-unique values",
                    summary=(
                        "Every row holds a different value, so the column "
                        "identifies rows rather than describing them."
                    ),
                    severity="info",
                    confidence=1.0,
                    evidence_ids=(evidence_id,),
                    category=OBSERVATION,
                    recommendation=(
                        "Usable as a key; exclude it from models and from "
                        "correlation reading, where it only adds noise."
                    ),
                )
            )
        elif (
            column.semantic_type in {"categorical", "boolean"}
            and column.unique_count is not None
            and column.unique_count > HIGH_CARDINALITY_CATEGORIES
        ):
            findings.append(
                Finding.create(
                    title=f"{table.name}.{column.name} has many distinct labels",
                    summary=(
                        f"{column.unique_count:,} distinct values in a column "
                        "read as categorical. One-hot encoding it produces that "
                        "many features."
                    ),
                    severity="info",
                    confidence=1.0,
                    evidence_ids=(evidence_id,),
                    category=OBSERVATION,
                    recommendation=(
                        "Group the long tail, target-encode, or treat the column "
                        "as free text."
                    ),
                )
            )
        if column.semantic_type in {"categorical", "boolean"} and column.top_values:
            top = column.top_values[0]
            share = int(top["count"]) / column.row_count if column.row_count else 0.0
            if share >= DOMINANT_LABEL_SHARE and column.unique_count not in (None, 1):
                findings.append(
                    Finding.create(
                        title=f"{table.name}.{column.name} is dominated by one label",
                        summary=(
                            f"One value covers {share:.1%} of rows, so the column "
                            "separates almost nothing."
                        ),
                        severity="info",
                        confidence=1.0,
                        evidence_ids=(evidence_id,),
                        category=OBSERVATION,
                        recommendation=(
                            "Check whether the rare labels are the interesting "
                            "cases before treating this as a usable feature."
                        ),
                    )
                )
    return findings


def timeline_findings(
    frame: pd.DataFrame,
    table: TableCatalog,
    timeline_evidence: dict[str, str],
) -> tuple[list[Evidence], list[Finding]]:
    """Coverage, gaps, and ordering observations for timestamp columns."""
    evidence: list[Evidence] = []
    findings: list[Finding] = []
    for column in table.columns:
        if column.semantic_type != "datetime" or column.name not in frame.columns:
            continue
        values = frame[column.name].dropna().sort_values().reset_index(drop=True)
        if len(values) < 3:
            continue
        gaps = values.diff()
        positive = gaps[gaps > pd.Timedelta(0)]
        median_gap = (
            pd.Timedelta(positive.median()) if not positive.empty else pd.Timedelta(0)
        )
        large: list[dict[str, Any]] = []
        if median_gap > pd.Timedelta(0):
            threshold = median_gap * TIMELINE_GAP_RATIO
            for position in np.flatnonzero((gaps > threshold).to_numpy()):
                large.append(
                    {
                        "from": to_jsonable(values.iloc[position - 1]),
                        "to": to_jsonable(values.iloc[position]),
                        "days": float(
                            pd.Timedelta(gaps.iloc[position]).total_seconds() / 86_400
                        ),
                    }
                )
        large.sort(key=lambda item: -float(item["days"]))
        original = frame[column.name].dropna()
        monotonic = bool(original.is_monotonic_increasing)

        item = Evidence.create(
            kind="profile_time_coverage",
            scope=EvidenceScope(table=table.name, columns=(column.name,)),
            value={
                "column": column.name,
                "min": to_jsonable(values.iloc[0]),
                "max": to_jsonable(values.iloc[-1]),
                "span_days": float(
                    (values.iloc[-1] - values.iloc[0]).total_seconds() / 86_400
                ),
                "median_gap_days": float(median_gap.total_seconds() / 86_400),
                "gap_count": len(large),
                "gaps": large[:MAX_REPORTED_GAPS],
                "rows_in_file_order_are_sorted": monotonic,
            },
            method="sorted_gap_scan_v1",
            description=f"Time coverage for {table.name}.{column.name}.",
        )
        evidence.append(item)
        timeline_evidence[column.name] = item.id

        span = item.value["span_days"]
        findings.append(
            Finding.create(
                title=f"{table.name}.{column.name} covers {span:,.0f} days",
                summary=(
                    f"From {str(item.value['min'])[:10]} to "
                    f"{str(item.value['max'])[:10]}, with a typical gap of "
                    f"{item.value['median_gap_days']:.2f} days between rows."
                ),
                severity="info",
                confidence=1.0,
                evidence_ids=(item.id,),
                category=OBSERVATION,
                recommendation=(
                    "Confirm this window matches the period you meant to analyse."
                ),
            )
        )
        if large:
            biggest = large[0]
            findings.append(
                Finding.create(
                    title=f"Gaps in {table.name}.{column.name} coverage",
                    summary=(
                        f"{len(large)} gap(s) run more than "
                        f"{TIMELINE_GAP_RATIO:.0f}x the typical spacing; the "
                        f"largest is {float(biggest['days']):,.1f} days."
                    ),
                    severity="info",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    category=OBSERVATION,
                    recommendation=(
                        "Check whether the gap is a collection outage or a real "
                        "quiet period before averaging across it."
                    ),
                )
            )
        if not monotonic:
            findings.append(
                Finding.create(
                    title=f"{table.name}.{column.name} is not in time order",
                    summary=(
                        "Rows are not stored sorted by this timestamp, so any "
                        "rolling window or shift over the file order is wrong."
                    ),
                    severity="info",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    category=OBSERVATION,
                    recommendation="Sort by this column before any time-series step.",
                )
            )
    return evidence, findings


def scatter_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    matrix: Evidence | None,
    *,
    limits: ProfileLimits,
    seed: int,
) -> list[Evidence]:
    """Highlighted pairs plus a pre-rendered explorer over the numeric columns."""
    if matrix is None:
        return []
    numeric: list[str] = list(matrix.value["numeric_columns"])
    if len(numeric) < 2:
        return []
    strengths = {
        (pair["left"], pair["right"]): float(pair["strength"])
        for pair in matrix.value["pairs"]
        if pair["method"] == "spearman"
    }

    def strength_of(left: str, right: str) -> float:
        return strengths.get((left, right), strengths.get((right, left), 0.0))

    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    highlights = [
        (left, right)
        for left, right in sorted(
            itertools.combinations(numeric, 2),
            key=lambda pair: -strength_of(*pair),
        )
    ]
    highlighted = highlights[: limits.highlight_pairs]
    explorer_columns = numeric[: limits.explorer_columns]
    explorer = [
        pair
        for pair in itertools.combinations(explorer_columns, 2)
        if pair not in highlighted
    ][: limits.explorer_pairs]

    evidence: list[Evidence] = []
    for role, pairs, budget in (
        ("highlight", highlighted, limits.scatter_points),
        ("explorer", explorer, limits.explorer_points),
    ):
        for left, right in pairs:
            usable = numeric_frame[[left, right]].dropna()
            if len(usable) < 5:
                continue
            drawn = usable
            if len(drawn) > budget:
                drawn = drawn.sample(n=budget, random_state=seed).sort_index()
            evidence.append(
                Evidence.create(
                    kind="profile_scatter",
                    scope=EvidenceScope(table=table.name, columns=(left, right)),
                    value={
                        "role": role,
                        "x_column": left,
                        "y_column": right,
                        "association": strength_of(left, right),
                        "point_count": int(len(drawn)),
                        "available_point_count": int(len(usable)),
                        # The renderer's scatter highlights flagged rows; the
                        # baseline profile flags none.
                        "points": [
                            {
                                "x": float(drawn.at[label, left]),
                                "y": float(drawn.at[label, right]),
                                "flagged": False,
                            }
                            for label in drawn.index
                        ],
                    },
                    method="numeric_pair_scatter_v1",
                    description=f"{right} against {left} for {table.name}.",
                )
            )
    return evidence


def missingness_evidence(frame: pd.DataFrame, table: TableCatalog) -> list[Evidence]:
    """Per-column missingness and which columns go missing together."""
    row_count = len(frame)
    if not row_count:
        return []
    columns = [
        {
            "column": column.name,
            "missing_count": column.missing_count,
            "missing_rate": column.missing_rate,
        }
        for column in table.columns
    ]
    evidence = [
        Evidence.create(
            kind="profile_missingness",
            scope=EvidenceScope(table=table.name),
            value={
                "row_count": row_count,
                "columns": columns,
                "complete_row_count": int(len(frame.dropna())),
            },
            method="exact_missing_counts_v1",
            description=f"Missing values per column for {table.name}.",
        )
    ]

    incomplete = [
        column.name
        for column in table.columns
        if column.missing_count and column.name in frame.columns
    ]
    if len(incomplete) < 2:
        return evidence
    masks = {name: frame[name].isna() for name in incomplete}
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(incomplete, 2):
        both = int((masks[left] & masks[right]).sum())
        if not both:
            continue
        union = int((masks[left] | masks[right]).sum())
        jaccard = both / union if union else 0.0
        if jaccard < CO_MISSING_MIN_JACCARD:
            continue
        pairs.append(
            {
                "left": left,
                "right": right,
                "both_missing": both,
                "jaccard": jaccard,
                "always_together": jaccard >= 0.999,
            }
        )
    if pairs:
        pairs.sort(key=lambda item: -float(item["jaccard"]))
        evidence.append(
            Evidence.create(
                kind="profile_co_missingness",
                scope=EvidenceScope(table=table.name, columns=tuple(incomplete)),
                value={"columns": incomplete, "pairs": pairs},
                method="pairwise_missing_jaccard_v1",
                description=(f"Columns whose missing values coincide in {table.name}."),
                assumptions=(
                    "Co-missingness suggests a shared cause (an optional form "
                    "section, a late-added field); it does not prove one.",
                ),
            )
        )
    return evidence


def sample_evidence(
    frame: pd.DataFrame, table: TableCatalog, *, limits: ProfileLimits
) -> list[Evidence]:
    """The first and last rows, and examples of any exact duplicates.

    Raw rows are banked for the local report only. The assisted-analysis layer
    forwards finding text and evidence *identifiers*, never evidence values, so
    this does not widen what a model can see.
    """
    if frame.empty:
        return []
    columns = [str(name) for name in frame.columns][: limits.sample_columns]
    hidden = max(0, len(frame.columns) - len(columns))
    limited = frame[columns]

    def rows(section: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "index": str(label),
                "values": [to_jsonable(section.at[label, name]) for name in columns],
            }
            for label in section.index
        ]

    evidence = [
        Evidence.create(
            kind="profile_sample_rows",
            scope=EvidenceScope(table=table.name),
            value={
                "columns": columns,
                "hidden_column_count": hidden,
                "row_count": int(len(frame)),
                "head": rows(limited.head(limits.sample_rows)),
                "tail": rows(limited.tail(limits.sample_rows))
                if len(frame) > limits.sample_rows
                else [],
            },
            method="head_tail_slice_v1",
            description=f"First and last rows of {table.name}.",
        )
    ]

    if table.duplicate_row_count:
        duplicated = frame[frame.duplicated(keep=False)]
        if not duplicated.empty:
            groups: list[dict[str, Any]] = []
            for _, group in duplicated.groupby(
                list(frame.columns), dropna=False, observed=True, sort=False
            ):
                groups.append(
                    {
                        "row_count": int(len(group)),
                        "indexes": [str(label) for label in group.index[:6]],
                        "values": [
                            to_jsonable(group.iloc[0][name]) for name in columns
                        ],
                    }
                )
                if len(groups) >= 3:
                    break
            evidence.append(
                Evidence.create(
                    kind="profile_duplicate_groups",
                    scope=EvidenceScope(table=table.name),
                    value={
                        "columns": columns,
                        "duplicate_row_count": table.duplicate_row_count,
                        "groups": groups,
                    },
                    method="exact_duplicate_grouping_v1",
                    description=f"Examples of duplicated rows in {table.name}.",
                )
            )
    return evidence


def build_observations(
    frame: pd.DataFrame,
    table: TableCatalog,
    column_evidence: dict[str, str],
    *,
    limits: ProfileLimits,
    seed: int,
) -> tuple[list[Evidence], list[Finding], list[AnalysisWarning], list[SamplingRecord]]:
    """Every association, interaction, missingness, and sample artefact."""
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    working = _sampled(
        frame,
        table=table.name,
        limits=limits,
        seed=seed,
        warnings=warnings,
        sampling=sampling,
    )

    evidence: list[Evidence] = []
    findings: list[Finding] = []

    matrix, skipped = association_evidence(working, table, limits=limits)
    if matrix is not None:
        evidence.append(matrix)
        findings.extend(_association_findings(matrix, table.name, matrix.confidence))
    if skipped:
        warnings.append(
            AnalysisWarning(
                code="association_columns_capped",
                message=(
                    f"{table.name} exceeds the {limits.correlation_columns}-column "
                    f"association budget; {len(skipped)} column(s) are not in the "
                    "correlation matrix. Pass detail='full' to widen it."
                ),
                table=table.name,
            )
        )

    findings.extend(_column_findings(frame, table, column_evidence))

    timeline_ids: dict[str, str] = {}
    time_evidence, time_findings = timeline_findings(frame, table, timeline_ids)
    evidence.extend(time_evidence)
    findings.extend(time_findings)

    evidence.extend(scatter_evidence(working, table, matrix, limits=limits, seed=seed))
    evidence.extend(missingness_evidence(frame, table))
    evidence.extend(sample_evidence(frame, table, limits=limits))
    return evidence, findings, warnings, sampling
