"""Comparison recipe to analyze drift and overlay distributions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from prism_eda.analysis._numeric import HIST_BINS
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.comparison_results import ComparisonResult
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import Evidence, EvidenceScope
from prism_eda.results import AnalysisStatus, AnalysisWarning


def _numeric_stats(vals: pd.Series, original: pd.Series) -> dict:
    return {
        "count": int(vals.count()),
        "missing": int(original.isna().sum()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "p25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "p75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
    }


def _cat_stats(original: pd.Series, vc: pd.Series) -> dict:
    return {
        "count": int(original.count()),
        "missing": int(original.isna().sum()),
        "unique": int(original.nunique()),
        "top": str(vc.index[0]) if not vc.empty else "—",
        "top_freq": int(vc.iloc[0]) if not vc.empty else 0,
    }


def build_comparison_evidence(
    base_frame: pd.DataFrame,
    compare_frame: pd.DataFrame,
    table_name: str,
) -> list[Evidence]:
    """Build side-by-side evidence for matched columns."""
    evidence: list[Evidence] = []
    common_cols = sorted(set(base_frame.columns).intersection(compare_frame.columns))

    for col in common_cols:
        b_col = base_frame[col]
        c_col = compare_frame[col]

        # Decide on histogram vs bar chart based on column dtype.
        # Numeric dtypes get a histogram; all others get a bar chart.
        use_histogram = pd.api.types.is_numeric_dtype(
            b_col
        ) or pd.api.types.is_numeric_dtype(c_col)

        if use_histogram:
            b_vals = pd.to_numeric(b_col, errors="coerce").dropna()
            c_vals = pd.to_numeric(c_col, errors="coerce").dropna()
            if b_vals.empty or c_vals.empty:
                use_histogram = False

        if use_histogram:
            min_val = min(float(b_vals.min()), float(c_vals.min()))
            max_val = max(float(b_vals.max()), float(c_vals.max()))

            if min_val == max_val:
                bins = np.array([min_val - 0.5, min_val + 0.5])
            else:
                bins = np.histogram_bin_edges(
                    np.concatenate([b_vals.to_numpy(), c_vals.to_numpy()]),
                    bins=HIST_BINS,
                )

            base_hist, _ = np.histogram(b_vals, bins=bins)
            comp_hist, _ = np.histogram(c_vals, bins=bins)

            evidence.append(
                Evidence.create(
                    kind="column_comparison_chart",
                    scope=EvidenceScope(table=table_name, columns=(col,)),
                    value={
                        "chart_type": "dual_histogram",
                        "edges": bins.tolist(),
                        "base_counts": base_hist.tolist(),
                        "compare_counts": comp_hist.tolist(),
                        "base_stats": _numeric_stats(b_vals, b_col),
                        "compare_stats": _numeric_stats(c_vals, c_col),
                    },
                    method="joint_histogram",
                    description=f"Comparison chart for {table_name}.{col}",
                )
            )
        else:
            b_vc = b_col.astype(str).value_counts(dropna=False).head(15)
            c_vc = c_col.astype(str).value_counts(dropna=False).head(15)
            all_cats = list(dict.fromkeys(b_vc.index.tolist() + c_vc.index.tolist()))

            evidence.append(
                Evidence.create(
                    kind="column_comparison_chart",
                    scope=EvidenceScope(table=table_name, columns=(col,)),
                    value={
                        "chart_type": "dual_bar",
                        "labels": [str(c) for c in all_cats],
                        "base_counts": [int(b_vc.get(c, 0)) for c in all_cats],
                        "compare_counts": [int(c_vc.get(c, 0)) for c in all_cats],
                        "base_stats": _cat_stats(b_col, b_vc),
                        "compare_stats": _cat_stats(c_col, c_vc),
                    },
                    method="joint_bar_chart",
                    description=f"Comparison bar chart for {table_name}.{col}",
                )
            )

    return evidence


def compare_datasets_recipe(
    base_tables: Mapping[str, pd.DataFrame],
    base_catalog: DatasetCatalog,
    compare_tables: Mapping[str, pd.DataFrame],
    compare_catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    callbacks: tuple[EventCallback, ...] = (),
) -> ComparisonResult:
    """Orchestrate dataset comparison."""
    emit(
        callbacks,
        Event(EventKind.RUN_STARTED, "Dataset comparison started.", stage="compare"),
    )

    evidence: list[Evidence] = []
    warnings: list[AnalysisWarning] = []

    common_tables = set(base_tables.keys()).intersection(compare_tables.keys())

    # Determine human-readable labels for the two sides. These are the dict
    # keys the caller supplied (e.g. "Table-C", "Synthetic-Table-C") and will
    # be shown in the report header instead of "Base dataset" / "Comparator".
    if len(base_tables) == 1:
        base_label = next(iter(base_tables))
    else:
        base_label = "Base dataset"
    if len(compare_tables) == 1:
        compare_label = next(iter(compare_tables))
    else:
        compare_label = "Comparator dataset"

    # Smart fallback: when the user wraps a single DataFrame on each side with
    # different keys (e.g. {"Table-C": df1} vs {"Synthetic-Table-C": df2}),
    # there are no name matches but the intent is obvious — compare the one
    # table on each side directly.
    if not common_tables and len(base_tables) == 1 and len(compare_tables) == 1:
        base_name = next(iter(base_tables))
        comp_name = next(iter(compare_tables))
        # Synthesise a shared label for the evidence scope.
        shared_label = f"{base_name} vs {comp_name}"
        evidence.extend(
            build_comparison_evidence(
                base_tables[base_name],
                compare_tables[comp_name],
                shared_label,
            )
        )
        # Use the synthesised pair as the "common table" for the summary.
        common_tables = {shared_label}
        warnings.append(
            AnalysisWarning(
                code="table_name_mismatch",
                message=(
                    f"Table names differ ('{base_name}' vs '{comp_name}'). "
                    "Compared them directly because each dataset has exactly one table."
                ),
            )
        )
    elif not common_tables:
        warnings.append(
            AnalysisWarning(
                code="no_common_tables",
                message=(
                    "The datasets have no tables with matching names to compare. "
                    "Use the same key on both sides, e.g. "
                    "compare_datasets({'data': df1}, {'data': df2})."
                ),
            )
        )
    else:
        for table_name in sorted(common_tables):
            base_frame = base_tables[table_name]
            comp_frame = compare_tables[table_name]
            evidence.extend(
                build_comparison_evidence(base_frame, comp_frame, table_name)
            )

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

    n_columns = len(evidence)
    n_tables = len(common_tables)
    if n_columns:
        summary = (
            f"Compared {n_columns} column(s) across "
            f"{n_tables} table pair(s) between datasets."
        )
    else:
        summary = "No columns could be compared between the two datasets."
    status = (
        AnalysisStatus.COMPLETED
        if n_columns
        else AnalysisStatus.COMPLETED_WITH_WARNINGS
    )

    result = ComparisonResult(
        goal="compare",
        status=status,
        summary=summary,
        base_catalog=base_catalog,
        compare_catalog=compare_catalog,
        evidence=tuple(evidence),
        warnings=tuple(warnings),
        metadata={
            # AnalysisConfig.__post_init__ normalizes this, but the declared
            # type stays AnalysisMode | str, so re-wrap the way every other
            # recipe does rather than assuming the narrowed type.
            "mode": AnalysisMode(config.mode).value,
            "base_label": base_label,
            "compare_label": compare_label,
        },
    )

    emit(callbacks, Event(EventKind.RUN_COMPLETED, result.summary, stage="compare"))
    return result
