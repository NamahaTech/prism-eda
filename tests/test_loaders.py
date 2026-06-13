from __future__ import annotations

import pandas as pd
import pytest

import prism_eda as pe
from prism_eda.exceptions import DataLoadError


def test_load_dataframe_keeps_named_table() -> None:
    frame = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})

    dataset = pe.load(frame)

    assert list(dataset.tables) == ["table"]
    assert dataset.table("table") is frame


def test_load_mapping_preserves_explicit_names() -> None:
    customers = pd.DataFrame({"customer_id": [1, 2]})
    orders = pd.DataFrame({"order_id": [10, 11], "customer_id": [1, 2]})

    dataset = pe.load({"customers": customers, "orders": orders})

    assert set(dataset.tables) == {"customers", "orders"}


def test_load_directory_is_non_recursive_by_default(tmp_path) -> None:
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "root.csv", index=False)
    nested = tmp_path / "nested"
    nested.mkdir()
    pd.DataFrame({"b": [2]}).to_csv(nested / "child.csv", index=False)

    shallow = pe.load(tmp_path)
    recursive = pe.load(tmp_path, recursive=True)

    assert set(shallow.tables) == {"root"}
    assert set(recursive.tables) == {"root", "child"}


def test_load_csv_and_parquet(tmp_path) -> None:
    frame = pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})
    csv_path = tmp_path / "records.csv"
    parquet_path = tmp_path / "records_copy.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    dataset = pe.load([csv_path, parquet_path])

    pd.testing.assert_frame_equal(dataset.table("records"), frame)
    pd.testing.assert_frame_equal(dataset.table("records_copy"), frame)


def test_duplicate_table_names_require_override(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(first / "data.csv", index=False)
    pd.DataFrame({"a": [2]}).to_csv(second / "data.csv", index=False)

    with pytest.raises(DataLoadError, match="provide names"):
        pe.load([first / "data.csv", second / "data.csv"])


def test_duplicate_dataframe_columns_are_rejected() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])

    with pytest.raises(DataLoadError, match="duplicate columns"):
        pe.load(frame)
