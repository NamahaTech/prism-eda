import pandas as pd
import pytest

from prism_eda import compare_datasets
from prism_eda.comparison_results import ComparisonResult
from prism_eda.results import AnalysisStatus


def test_compare_datasets_basic():
    """Test that two datasets can be compared and produce dual evidence."""
    base_df = pd.DataFrame({"col_a": [1, 2, 3, 4, 5], "col_b": ["x", "x", "y", "z", "z"]})
    comp_df = pd.DataFrame({"col_a": [2, 3, 4, 5, 6], "col_b": ["x", "y", "y", "z", "z"]})

    result = compare_datasets(
        {"table1": base_df},
        {"table1": comp_df},
    )

    assert isinstance(result, ComparisonResult)
    assert result.status == AnalysisStatus.COMPLETED
    assert result.base_catalog.table_count == 1
    assert result.compare_catalog.table_count == 1

    # We expect 2 column comparison charts
    charts = [e for e in result.evidence if e.kind == "column_comparison_chart"]
    assert len(charts) == 2

    # Check the numeric histogram chart
    numeric_chart = next(c for c in charts if c.scope.columns == ("col_a",))
    assert numeric_chart.value["chart_type"] == "dual_histogram"
    assert "edges" in numeric_chart.value
    assert "base_counts" in numeric_chart.value
    assert "compare_counts" in numeric_chart.value

    # Check the categorical bar chart
    cat_chart = next(c for c in charts if c.scope.columns == ("col_b",))
    assert cat_chart.value["chart_type"] == "dual_bar"
    assert "labels" in cat_chart.value
    assert "base_counts" in cat_chart.value
    assert "compare_counts" in cat_chart.value


def test_compare_datasets_single_table_name_mismatch():
    """Single-table datasets with different keys are compared via fallback."""
    base_df = pd.DataFrame({"col_a": [1, 2]})
    comp_df = pd.DataFrame({"col_a": [3, 4]})

    result = compare_datasets(
        {"base_only": base_df},
        {"comp_only": comp_df},
    )

    # Fallback succeeds and emits a warning explaining the name mismatch.
    assert result.status == AnalysisStatus.COMPLETED
    assert any(w.code == "table_name_mismatch" for w in result.warnings)
    assert len(result.evidence) == 1


def test_compare_datasets_no_common_tables_multi():
    """Multiple tables with no overlap cannot be auto-matched — warns cleanly."""
    df = pd.DataFrame({"col_a": [1, 2]})

    result = compare_datasets(
        {"base_a": df, "base_b": df},
        {"comp_x": df, "comp_y": df},
    )

    assert result.status == AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert any(w.code == "no_common_tables" for w in result.warnings)
    assert not result.evidence
