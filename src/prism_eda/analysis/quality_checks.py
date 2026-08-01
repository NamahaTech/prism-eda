"""Data-quality defect detectors for the baseline profile.

Everything in this module answers one question: *is something wrong with this
data?* That is a deliberately narrow brief. Facts that are true but not broken —
two columns correlate, a column is all-unique — are observations, and they live
in the alerts channel (:mod:`prism_eda.analysis.associations`) so that this
channel keeps its signal. A findings list an analyst learns to skim is worse than
no findings list.

Two rules hold throughout:

* **These checks never sample.** They make exact claims about defects ("14 rows
  hold a placeholder value"), and a sampled count would be a guess wearing a
  number's clothes. Chart and association work samples; this does not.
* **Raw cell values stay in evidence, never in a finding summary.** Finding
  summaries are forwarded to the optional AI layer; evidence values are not (only
  evidence *IDs* cross that boundary). So the offending values are banked for the
  local report and the summary stays aggregate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from prism_eda._serialization import to_jsonable
from prism_eda.catalog.models import TableCatalog
from prism_eda.evidence.models import Evidence, EvidenceScope, Finding
from prism_eda.transformations.models import TransformationStep

# Case/whitespace variant detection only makes sense for a column that is meant
# to be an enumerable set of labels. On free text, "Paris" vs "paris" is not a
# defect, it is prose — so columns above this distinct count are left alone.
CASE_VARIANT_MAX_DISTINCT = 500

# Share of non-null values that must parse as numbers before a text column is
# called "numbers stored as text". Below this it is a text column that happens to
# contain some digits.
NUMERIC_TEXT_MIN_RATE = 0.95

# Share of non-null values that must match a recognised date layout before a text
# column is treated as a date column at all.
DATE_TEXT_MIN_RATE = 0.90

# Placeholder tokens that pandas does not read as null. Matched after stripping
# and case-folding.
MISSING_SENTINEL_TOKENS = frozenset(
    {
        "",
        "-",
        "--",
        "?",
        "??",
        ".",
        "n/a",
        "n.a.",
        "na",
        "nan",
        "nil",
        "none",
        "null",
        "missing",
        "unknown",
        "undefined",
        "not available",
        "not applicable",
        "#n/a",
        "#null!",
        "tbd",
    }
)

# Numeric placeholders. Only strongly negative magic numbers are listed, and even
# those are reported only when the column is otherwise non-negative and the value
# repeats — a lone -999 in a column of temperature deltas is data, not a sentinel.
NUMERIC_SENTINELS = (-999.0, -9999.0, -99999.0, -999999.0)

# Date placeholders. Values pandas can hold in datetime64[ns] only; 9999-12-31 is
# outside that range and therefore only ever appears in text date columns.
DATE_SENTINELS = ("1970-01-01", "1900-01-01", "1899-12-30", "2099-12-31")

# Anything outside this window is not a plausible observation date in a business
# or scientific dataset; it is a parsing accident or a sentinel.
IMPLAUSIBLE_BEFORE = pd.Timestamp("1800-01-01")
IMPLAUSIBLE_AFTER_YEARS = 100

# Name tokens that identify a start/end column pair, so that "the end is before
# the start" can be checked at all.
RANGE_TOKEN_PAIRS = (
    ("start", "end"),
    ("start", "stop"),
    ("begin", "end"),
    ("from", "to"),
    ("open", "close"),
    ("created", "closed"),
    ("birth", "death"),
    ("hire", "termination"),
    ("first", "last"),
)

# Pairwise column comparison is quadratic, so the aliased-column check is capped.
MAX_COLUMNS_FOR_PAIR_CHECKS = 250

_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ISO date (YYYY-MM-DD)", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")),
    (
        "ISO datetime",
        re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?"),
    ),
    ("YYYY/MM/DD", re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")),
    ("DD/MM/YYYY or MM/DD/YYYY", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    ("DD/MM/YY or MM/DD/YY", re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")),
    ("DD-MM-YYYY or MM-DD-YYYY", re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$")),
    ("DD.MM.YYYY", re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")),
    (
        "DD-Mon-YYYY",
        re.compile(r"^\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{2,4}$"),
    ),
    (
        "Month DD, YYYY",
        re.compile(r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"),
    ),
)

_SLASH_AMBIGUOUS = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}$")
_UNNAMED = re.compile(r"^unnamed:?\s*\d*$", re.IGNORECASE)
_MANGLED_DUPLICATE = re.compile(r"^(?P<base>.+)\.(?P<index>\d+)$")


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One detected defect, ready to become evidence, a finding, and a step."""

    kind: str
    columns: tuple[str, ...]
    value: dict[str, Any]
    method: str
    description: str
    title: str
    summary: str
    severity: str
    recommendation: str
    operation: str
    rationale: str
    risk: str = "medium"
    assumptions: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    # Whether the finding title reads "<defect> in <table>.<column>" or just
    # "<defect> in <table>". A defect in the table's shape is not about one
    # column even when only one column happens to carry it.
    title_scope: str = "column"


def _strings(series: pd.Series) -> pd.Series:
    """Non-null values of a column that are actually Python strings."""
    values = series.dropna()
    if values.empty:
        return values
    return values[[isinstance(value, str) for value in values]]


def _is_text_like(series: pd.Series) -> bool:
    """True for any column that holds labels rather than numbers or timestamps.

    pandas 3 infers a dedicated ``str`` dtype where pandas 2 used ``object``, so
    testing for ``object`` alone silently skips every text column on a modern
    pandas.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    return bool(ptypes.is_object_dtype(dtype) or ptypes.is_string_dtype(dtype))


def _count(count: int, noun: str, plural: str | None = None) -> str:
    """Format "1 row" / "4 rows" so summaries read as sentences."""
    word = noun if count == 1 else (plural or f"{noun}s")
    return f"{count:,} {word}"


def _examples(values: Sequence[Any], limit: int = 5) -> list[Any]:
    return [to_jsonable(value) for value in list(values)[:limit]]


def _inconsistent_formatting(
    series: pd.Series, column: str, distinct: int | None
) -> QualityIssue | None:
    """Labels that differ only by case or surrounding whitespace."""
    if distinct is not None and distinct > CASE_VARIANT_MAX_DISTINCT:
        return None
    strings = _strings(series)
    if len(strings) < 2:
        return None
    stripped = strings.str.strip()
    padded_count = int((stripped != strings).sum())
    folded = stripped.str.casefold()
    # Placeholder tokens collapse into each other ("NA"/"n/a") for reasons that
    # have nothing to do with formatting; the disguised-missing check owns those.
    keep = ~folded.isin(MISSING_SENTINEL_TOKENS)
    pairs = pd.DataFrame(
        {"raw": strings[keep].to_numpy(), "key": folded[keep].to_numpy()}
    )
    if pairs.empty:
        return None
    variants = pairs.groupby("key")["raw"].nunique()
    collapsing = variants[variants > 1]
    if collapsing.empty and not padded_count:
        return None

    affected_rows = int(pairs["key"].isin(collapsing.index).sum())
    examples = []
    for key in list(collapsing.sort_values(ascending=False).index)[:5]:
        forms = pairs.loc[pairs["key"] == key, "raw"].value_counts()
        examples.append(
            {
                "canonical": str(key),
                "variants": [
                    {"value": str(value), "count": int(count)}
                    for value, count in forms.items()
                ][:6],
            }
        )
    group_count = int(len(collapsing))
    row_count = len(series)
    rate = affected_rows / row_count if row_count else 0.0
    if group_count:
        summary = (
            f"{_count(group_count, 'label')} "
            f"{'appears' if group_count == 1 else 'appear'} under more than one "
            f"spelling, affecting {_count(affected_rows, 'row')} ({rate:.1%})."
        )
    else:
        summary = (
            f"{_count(padded_count, 'value')} "
            f"{'carries' if padded_count == 1 else 'carry'} leading or trailing "
            "whitespace."
        )
    return QualityIssue(
        kind="quality_inconsistent_formatting",
        columns=(column,),
        value={
            "column": column,
            "variant_group_count": group_count,
            "affected_row_count": affected_rows,
            "padded_value_count": padded_count,
            "examples": examples,
        },
        method="casefold_strip_collapse_v1",
        description=f"Case and whitespace variants in {column}.",
        title="Inconsistent value formatting",
        summary=summary,
        severity="high" if rate >= 0.05 or group_count >= 5 else "medium",
        recommendation=(
            "Normalize case and trim whitespace before grouping or joining on "
            "this column; today the same label counts as several."
        ),
        operation="normalize_text_values",
        rationale=(
            "Case and whitespace variants split what should be one category "
            "across several groups."
        ),
    )


def _numeric_stored_as_text(series: pd.Series, column: str) -> QualityIssue | None:
    """A column of digits kept as strings, so it sorts and aggregates wrongly."""
    strings = _strings(series)
    non_null = series.dropna()
    if len(strings) < 5 or len(strings) != len(non_null):
        return None
    parsed = pd.to_numeric(strings.str.strip(), errors="coerce")
    parsed_count = int(parsed.notna().sum())
    rate = parsed_count / len(strings)
    if rate < NUMERIC_TEXT_MIN_RATE:
        return None
    unparsed = strings[parsed.isna()]
    return QualityIssue(
        kind="quality_numeric_as_text",
        columns=(column,),
        value={
            "column": column,
            "parsed_rate": rate,
            "parsed_count": parsed_count,
            "unparsed_count": int(len(unparsed)),
            "unparsed_examples": _examples(list(unparsed.unique())),
        },
        method="to_numeric_coercion_rate_v1",
        description=f"Numeric-looking text column {column}.",
        title="Numbers stored as text",
        summary=(
            f"{rate:.1%} of values parse as numbers, but the column is text, so "
            "it sorts alphabetically and will not aggregate."
        ),
        severity="medium",
        recommendation=(
            "Convert to a numeric dtype after confirming the values are "
            "quantities rather than codes, and check the values that fail to "
            "parse."
        ),
        operation="convert_text_to_numeric",
        rationale="Numeric values held as text sort and aggregate incorrectly.",
    )


def _mixed_types(series: pd.Series, column: str) -> QualityIssue | None:
    """More than one Python type in one column."""
    if not ptypes.is_object_dtype(series.dtype):
        return None
    non_null = series.dropna()
    if len(non_null) < 2:
        return None
    counts: dict[str, int] = {}
    for value in non_null:
        name = type(value).__name__
        counts[name] = counts.get(name, 0) + 1
    if len(counts) < 2:
        return None
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    minority = sum(count for _, count in ordered[1:])
    return QualityIssue(
        kind="quality_mixed_types",
        columns=(column,),
        value={
            "column": column,
            "type_counts": [{"type": name, "count": count} for name, count in ordered],
            "minority_count": minority,
        },
        method="python_type_histogram_v1",
        description=f"Mixed Python types in {column}.",
        title="Mixed value types",
        summary=(
            f"The column holds {len(counts)} different value types "
            f"({', '.join(name for name, _ in ordered)}); "
            f"{_count(minority, 'row')} "
            f"{'is' if minority == 1 else 'are'} not the dominant type."
        ),
        severity="high",
        recommendation=(
            "Coerce the column to one type. Comparisons and sorts across mixed "
            "types either raise or silently order by type."
        ),
        operation="unify_column_type",
        rationale="Mixed types make comparison and sorting undefined.",
        risk="high",
    )


def _date_text_formats(series: pd.Series, column: str) -> QualityIssue | None:
    """Dates kept as text, and worse, kept in more than one layout."""
    strings = _strings(series)
    if len(strings) < 5:
        return None
    trimmed = strings.str.strip()
    matched: dict[str, int] = {}
    unmatched = 0
    for value in trimmed:
        for label, pattern in _DATE_PATTERNS:
            if pattern.match(value):
                matched[label] = matched.get(label, 0) + 1
                break
        else:
            unmatched += 1
    total = len(trimmed)
    matched_total = total - unmatched
    if not matched_total or matched_total / total < DATE_TEXT_MIN_RATE:
        return None

    # Day/month order is only recoverable when some row has a component above 12.
    high_first = high_second = 0
    for value in trimmed:
        match = _SLASH_AMBIGUOUS.match(value)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        high_first += first > 12
        high_second += second > 12
    ambiguous = bool(high_first and high_second)

    layouts = sorted(matched.items(), key=lambda item: -item[1])
    mixed = len(layouts) > 1
    if not mixed and not ambiguous:
        title = "Dates stored as text"
        summary = (
            f"{_count(matched_total, 'value')} "
            f"{'looks' if matched_total == 1 else 'look'} like dates in "
            f"{layouts[0][0]} form, but the column is text, so it will not sort "
            "or filter as a date."
        )
        severity = "medium"
    elif ambiguous:
        title = "Ambiguous day/month order"
        summary = (
            f"Values appear in both orders: {_count(high_first, 'row')} "
            f"{'has' if high_first == 1 else 'have'} a first component above 12 "
            f"and {high_second:,} have a second component above 12, so no single "
            "parse of this column can be correct."
        )
        severity = "high"
    else:
        title = "Mixed date formats"
        summary = (
            f"{len(layouts)} different date layouts appear in one column "
            f"({', '.join(label for label, _ in layouts)}); parsing with one "
            "format will silently produce wrong dates or nulls."
        )
        severity = "high"

    return QualityIssue(
        kind="quality_date_text_format",
        columns=(column,),
        value={
            "column": column,
            "layouts": [{"layout": label, "count": count} for label, count in layouts],
            "unmatched_count": unmatched,
            "unmatched_examples": _examples(
                [
                    value
                    for value in trimmed
                    if not any(pattern.match(value) for _, pattern in _DATE_PATTERNS)
                ]
            ),
            "ambiguous_day_month": ambiguous,
            "day_first_evidence_rows": high_first,
            "month_first_evidence_rows": high_second,
        },
        method="date_layout_regex_scan_v1",
        description=f"Date layouts found in the text column {column}.",
        title=title,
        summary=summary,
        severity=severity,
        recommendation=(
            "Parse to a real datetime dtype with an explicit format per layout, "
            "and confirm the day/month order against a known record before "
            "trusting the result."
        ),
        operation="parse_text_dates",
        rationale="Text dates do not sort, filter, or compare as dates.",
        risk="high" if severity == "high" else "medium",
        assumptions=(
            "Layouts are recognised by shape, not parsed; a value matching a "
            "layout is not guaranteed to be a valid calendar date.",
        ),
    )


def _disguised_missing_text(
    series: pd.Series, column: str, declared_missing: int
) -> QualityIssue | None:
    """Placeholder strings that `isna()` does not count as missing."""
    strings = _strings(series)
    if strings.empty:
        return None
    folded = strings.str.strip().str.casefold()
    hits = folded[folded.isin(MISSING_SENTINEL_TOKENS)]
    if hits.empty:
        return None
    counts = strings[hits.index].value_counts()
    hidden = int(len(hits))
    row_count = len(series)
    effective = (declared_missing + hidden) / row_count if row_count else 0.0
    return QualityIssue(
        kind="quality_disguised_missing",
        columns=(column,),
        value={
            "column": column,
            "hidden_missing_count": hidden,
            "declared_missing_count": declared_missing,
            "effective_missing_rate": effective,
            "placeholders": [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ][:8],
        },
        method="placeholder_token_match_v1",
        description=f"Placeholder values standing in for nulls in {column}.",
        title="Disguised missing values",
        summary=(
            f"{_count(hidden, 'value')} "
            f"{'is a placeholder' if hidden == 1 else 'are placeholders'} "
            f"rather than data. Real missingness is {effective:.1%}, not "
            f"{declared_missing / row_count if row_count else 0:.1%}."
        ),
        severity="high" if effective >= 0.2 else "medium",
        recommendation=(
            "Convert the placeholders to nulls before analysis, so missingness "
            "checks and imputation see the true rate."
        ),
        operation="replace_placeholders_with_null",
        rationale="Placeholder strings hide the column's real missing rate.",
    )


def _disguised_missing_numeric(
    series: pd.Series, column: str, declared_missing: int
) -> QualityIssue | None:
    """Magic negative numbers standing in for nulls."""
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return None
    found: list[dict[str, Any]] = []
    for sentinel in NUMERIC_SENTINELS:
        count = int((numeric == sentinel).sum())
        # A sentinel repeats. A single odd value is an outlier, not a code — and
        # calling it a code would be a guess presented as a defect.
        if count >= 2:
            found.append({"value": sentinel, "count": count})
    if not found:
        return None
    remainder = numeric[~numeric.isin([item["value"] for item in found])]
    # Only a column that is otherwise non-negative makes these values obviously
    # out-of-domain. In a column that genuinely goes negative they may be data.
    if remainder.empty or float(remainder.min()) < 0:
        return None
    hidden = sum(int(item["count"]) for item in found)
    row_count = len(series)
    effective = (declared_missing + hidden) / row_count if row_count else 0.0
    return QualityIssue(
        kind="quality_disguised_missing",
        columns=(column,),
        value={
            "column": column,
            "hidden_missing_count": hidden,
            "declared_missing_count": declared_missing,
            "effective_missing_rate": effective,
            "placeholders": found,
            "remainder_min": float(remainder.min()),
        },
        method="negative_sentinel_match_v1",
        description=f"Sentinel numbers standing in for nulls in {column}.",
        title="Sentinel values",
        summary=(
            f"{_count(hidden, 'row')} "
            f"{'holds' if hidden == 1 else 'hold'} a placeholder "
            "number in a column whose real values are non-negative; every mean, "
            "minimum, and correlation on this column is currently wrong."
        ),
        severity="high",
        recommendation=(
            "Replace the sentinel with null before computing any statistic on "
            "this column."
        ),
        operation="replace_sentinel_with_null",
        rationale="Sentinel numbers are counted as real values by every metric.",
        risk="high",
    )


def _invalid_dates(
    series: pd.Series, column: str, *, now: pd.Timestamp
) -> list[QualityIssue]:
    """Sentinel, implausible, and future timestamps in a real datetime column."""
    values = series.dropna()
    if values.empty:
        return []
    issues: list[QualityIssue] = []
    reference = now
    timezone = getattr(values.dtype, "tz", None)
    if timezone is not None:
        reference = now.tz_localize("UTC") if now.tzinfo is None else now
        reference = reference.tz_convert(timezone)
    elif now.tzinfo is not None:
        reference = now.tz_localize(None)

    sentinel_hits: list[dict[str, Any]] = []
    for text in DATE_SENTINELS:
        try:
            stamp = pd.Timestamp(text)
        except (ValueError, OverflowError):
            continue
        if reference.tzinfo is not None:
            stamp = stamp.tz_localize(reference.tzinfo)
        count = int((values == stamp).sum())
        if count:
            sentinel_hits.append({"value": text, "count": count})
    if sentinel_hits:
        total = sum(int(item["count"]) for item in sentinel_hits)
        issues.append(
            QualityIssue(
                kind="quality_date_sentinel",
                columns=(column,),
                value={
                    "column": column,
                    "sentinels": sentinel_hits,
                    "affected_row_count": total,
                },
                method="known_date_sentinel_match_v1",
                description=f"Placeholder dates in {column}.",
                title="Placeholder dates",
                summary=(
                    f"{_count(total, 'row')} "
                    f"{'carries' if total == 1 else 'carry'} a known placeholder "
                    "date rather than a real one."
                ),
                severity="high",
                recommendation=(
                    "Treat the placeholder as missing. Left in place it anchors "
                    "every date range and duration computed from this column."
                ),
                operation="replace_placeholder_dates",
                rationale="Placeholder dates distort ranges and durations.",
                risk="high",
            )
        )

    limit = reference + pd.DateOffset(years=IMPLAUSIBLE_AFTER_YEARS)
    lower = IMPLAUSIBLE_BEFORE
    if reference.tzinfo is not None:
        lower = lower.tz_localize(reference.tzinfo)
    implausible = values[(values < lower) | (values > limit)]
    if not implausible.empty:
        issues.append(
            QualityIssue(
                kind="quality_implausible_date",
                columns=(column,),
                value={
                    "column": column,
                    "affected_row_count": int(len(implausible)),
                    "min": to_jsonable(implausible.min()),
                    "max": to_jsonable(implausible.max()),
                },
                method="calendar_plausibility_window_v1",
                description=f"Implausible timestamps in {column}.",
                title="Implausible dates",
                summary=(
                    f"{_count(len(implausible), 'timestamp')} "
                    f"{'falls' if len(implausible) == 1 else 'fall'} outside any "
                    "plausible observation window, which usually means a parsing "
                    "error or a unit mix-up."
                ),
                severity="high",
                recommendation=(
                    "Re-parse the source values for these rows; check for epoch "
                    "seconds read as nanoseconds or a two-digit year."
                ),
                operation="review_implausible_dates",
                rationale="Implausible timestamps indicate a parsing failure.",
                risk="high",
            )
        )

    future = values[(values > reference) & (values <= limit)]
    if not future.empty:
        issues.append(
            QualityIssue(
                kind="quality_future_date",
                columns=(column,),
                value={
                    "column": column,
                    "affected_row_count": int(len(future)),
                    "max": to_jsonable(future.max()),
                },
                method="future_timestamp_scan_v1",
                # The reference instant is deliberately kept out of the evidence
                # value: it would change the evidence ID on every run, and the
                # claim being made is about the rows, not about the clock.
                description=(
                    f"Timestamps in {column} after {reference.date().isoformat()}."
                ),
                title="Future-dated rows",
                summary=(
                    f"{_count(len(future), 'row')} "
                    f"{'is' if len(future) == 1 else 'are'} dated in the future. "
                    "This is expected for scheduled events and is a defect for "
                    "anything already observed."
                ),
                severity="medium",
                recommendation=(
                    "Confirm whether this column records planned or observed "
                    "events before filtering or correcting the rows."
                ),
                operation="review_future_dates",
                rationale=(
                    "Future timestamps in an observational column indicate a "
                    "data-entry or timezone error."
                ),
                assumptions=(
                    "Compared against the clock at analysis time, so this check "
                    "is not reproducible across days.",
                ),
            )
        )
    return issues


def _range_pairs(columns: Sequence[str]) -> list[tuple[str, str]]:
    """Find (start, end) column pairs by name, e.g. start_date / end_date."""
    lookup = {name.lower(): name for name in columns}
    pairs: list[tuple[str, str]] = []
    for name in columns:
        lowered = name.lower()
        for first, second in RANGE_TOKEN_PAIRS:
            if first not in lowered:
                continue
            candidate = lowered.replace(first, second)
            if candidate != lowered and candidate in lookup:
                pairs.append((name, lookup[candidate]))
                break
    return pairs


def _reversed_ranges(frame: pd.DataFrame, start: str, end: str) -> QualityIssue | None:
    both = frame[[start, end]].dropna()
    if both.empty:
        return None
    reversed_rows = both[both[end] < both[start]]
    if reversed_rows.empty:
        return None
    count = int(len(reversed_rows))
    rate = count / len(both)
    return QualityIssue(
        kind="quality_reversed_date_range",
        columns=(start, end),
        value={
            "start_column": start,
            "end_column": end,
            "affected_row_count": count,
            "compared_row_count": int(len(both)),
            "rate": rate,
        },
        method="pairwise_range_order_check_v1",
        description=f"Rows where {end} precedes {start}.",
        title="End date precedes start date",
        summary=(
            f"{_count(count, 'row')} ({rate:.1%}) "
            f"{'ends' if count == 1 else 'end'} before "
            f"{'it starts' if count == 1 else 'they start'}, so any duration "
            "computed from this pair is negative."
        ),
        severity="high",
        recommendation=(
            "Check whether the two columns were swapped at load time before "
            "correcting individual rows."
        ),
        operation="review_reversed_date_ranges",
        rationale="A negative duration cannot be a real observation.",
        risk="high",
        title_scope="table",
    )


def _structural_issues(
    frame: pd.DataFrame,
) -> tuple[list[QualityIssue], set[str]]:
    """Defects in the shape of the table rather than in its values.

    Also returns the columns that are exact copies of an earlier column. Those
    are skipped by the per-column checks: a copy repeats every defect of its
    original, and printing each one twice is how a findings list becomes noise.
    The duplicate-columns finding already names them.
    """
    issues: list[QualityIssue] = []
    names = [str(name) for name in frame.columns]

    unnamed = [name for name in names if not name.strip() or _UNNAMED.match(name)]
    if unnamed:
        issues.append(
            QualityIssue(
                kind="quality_unnamed_columns",
                columns=tuple(unnamed),
                value={"columns": unnamed, "count": len(unnamed)},
                method="header_name_scan_v1",
                description="Columns with no usable header.",
                title="Unnamed columns",
                summary=(
                    f"{_count(len(unnamed), 'column')} "
                    f"{'has' if len(unnamed) == 1 else 'have'} no header. This "
                    "is usually a stray index column written out by a previous "
                    "export."
                ),
                severity="medium",
                recommendation=(
                    "Drop the column if it is an exported index, or give it a "
                    "name if it carries data."
                ),
                operation="review_unnamed_columns",
                rationale="An unnamed column cannot be referenced or understood.",
                title_scope="table",
            )
        )

    existing = set(names)
    mangled = []
    for name in names:
        match = _MANGLED_DUPLICATE.match(name)
        if match and match.group("base") in existing:
            mangled.append({"column": name, "base": match.group("base")})
    if mangled:
        issues.append(
            QualityIssue(
                kind="quality_duplicate_headers",
                columns=tuple(item["column"] for item in mangled),
                value={"columns": mangled, "count": len(mangled)},
                method="mangled_header_suffix_scan_v1",
                description="Headers renamed to resolve a duplicate.",
                title="Duplicated column headers in the source",
                summary=(
                    f"{_count(len(mangled), 'column')} "
                    f"{'was' if len(mangled) == 1 else 'were'} renamed with a "
                    "numeric suffix because the source file repeated a header. "
                    "Which one holds the intended values is not recorded "
                    "anywhere."
                ),
                severity="high",
                recommendation=(
                    "Fix the header row at the source; downstream code selecting "
                    "by name is silently picking the first of the duplicates."
                ),
                operation="review_duplicate_headers",
                rationale="Repeated headers make column selection ambiguous.",
                risk="high",
                title_scope="table",
            )
        )

    aliases: set[str] = set()
    if len(names) <= MAX_COLUMNS_FOR_PAIR_CHECKS:
        alias_issues, aliases = _aliased_columns(frame, names)
        issues.extend(alias_issues)
    return issues, aliases


def _aliased_columns(
    frame: pd.DataFrame, names: Sequence[str]
) -> tuple[list[QualityIssue], set[str]]:
    """Two columns holding identical values under different names."""
    signatures: dict[Any, list[str]] = {}
    for name in names:
        series = frame[name]
        try:
            signature = (
                str(series.dtype),
                int(series.notna().sum()),
                hash(tuple(series.head(64).astype(str))),
            )
        except (TypeError, ValueError):
            continue
        signatures.setdefault(signature, []).append(name)

    issues: list[QualityIssue] = []
    seen: set[str] = set()
    for group in signatures.values():
        if len(group) < 2:
            continue
        anchor = group[0]
        duplicates = [
            name
            for name in group[1:]
            if name not in seen and frame[anchor].equals(frame[name])
        ]
        if not duplicates:
            continue
        seen.update(duplicates)
        issues.append(
            QualityIssue(
                kind="quality_aliased_columns",
                columns=tuple([anchor, *duplicates]),
                value={"columns": [anchor, *duplicates]},
                method="exact_column_equality_v1",
                description=f"Columns identical to {anchor}.",
                title="Duplicate columns",
                summary=(
                    f"{len(duplicates) + 1} columns hold identical values. Any "
                    "model or correlation over this table double-counts what is "
                    "really one variable."
                ),
                severity="medium",
                recommendation=(
                    "Keep one column and drop the copies once you have confirmed "
                    "which name downstream code uses."
                ),
                operation="review_aliased_columns",
                rationale="Identical columns double-count a single variable.",
                title_scope="table",
            )
        )
    return issues, seen


def detect_quality_issues(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    now: pd.Timestamp | None = None,
) -> list[QualityIssue]:
    """Run every data-quality detector over one table."""
    reference = now if now is not None else pd.Timestamp.now()
    issues, aliased = _structural_issues(frame)
    catalog_by_name = {column.name: column for column in table.columns}

    for raw_name in frame.columns:
        name = str(raw_name)
        profile = catalog_by_name.get(name)
        if profile is None or name in aliased:
            continue
        series = frame[raw_name]
        if ptypes.is_datetime64_any_dtype(series.dtype):
            issues.extend(_invalid_dates(series, name, now=reference))
            continue
        if ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(
            series.dtype
        ):
            found = _disguised_missing_numeric(series, name, profile.missing_count)
            if found is not None:
                issues.append(found)
            continue
        if not _is_text_like(series):
            continue

        mixed = _mixed_types(series, name)
        if mixed is not None:
            issues.append(mixed)
        hidden = _disguised_missing_text(series, name, profile.missing_count)
        if hidden is not None:
            issues.append(hidden)
        dates = _date_text_formats(series, name)
        if dates is not None:
            issues.append(dates)
        else:
            numeric_text = _numeric_stored_as_text(series, name)
            if numeric_text is not None:
                issues.append(numeric_text)
        formatting = _inconsistent_formatting(series, name, profile.unique_count)
        if formatting is not None:
            issues.append(formatting)

    datetime_columns = [
        str(name)
        for name in frame.columns
        if ptypes.is_datetime64_any_dtype(frame[name].dtype)
    ]
    for start, end in _range_pairs(datetime_columns):
        reversed_range = _reversed_ranges(frame, start, end)
        if reversed_range is not None:
            issues.append(reversed_range)
    return issues


def _finding_title(issue: QualityIssue, table: str) -> str:
    """Scope a defect to where it lives, matching the profile's other findings."""
    if issue.title_scope == "column" and len(issue.columns) == 1:
        return f"{issue.title} in {table}.{issue.columns[0]}"
    return f"{issue.title} in {table}"


def build_quality_findings(
    frame: pd.DataFrame,
    table: TableCatalog,
    *,
    now: pd.Timestamp | None = None,
) -> tuple[list[Evidence], list[Finding], list[TransformationStep]]:
    """Turn detected defects into evidence, findings, and reviewable steps."""
    evidence: list[Evidence] = []
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    for issue in detect_quality_issues(frame, table, now=now):
        item = Evidence.create(
            kind=issue.kind,
            scope=EvidenceScope(table=table.name, columns=issue.columns),
            value=issue.value,
            method=issue.method,
            description=f"{issue.description} ({table.name})",
            assumptions=issue.assumptions,
        )
        evidence.append(item)
        findings.append(
            Finding.create(
                title=_finding_title(issue, table.name),
                summary=issue.summary,
                severity=issue.severity,
                confidence=1.0,
                evidence_ids=(item.id,),
                recommendation=issue.recommendation,
            )
        )
        steps.append(
            TransformationStep(
                operation=issue.operation,
                table=table.name,
                columns=issue.columns,
                parameters=issue.parameters,
                rationale=issue.rationale,
                evidence_ids=(item.id,),
                risk=issue.risk,
            )
        )
    return evidence, findings, steps
