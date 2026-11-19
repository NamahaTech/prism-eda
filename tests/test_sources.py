"""The source contract: canonical values, the shared hash, and FrameSource.

The canonical form and the hash are a cross-backend contract. ClickHouse,
Postgres and pandas must all agree on them, so the values pinned here must
never change without changing every SQL dialect in the same commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism_eda.sources import FrameSource, TableInfo, select_tables
from prism_eda.sources.base import (
    check_rate,
    in_hash_sample,
    normalise_value,
    sampling_rate,
    value_hash,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  00AbC ", "abc"),
        ("000500012345", "500012345"),
        (500012345, "500012345"),
        (12.0, "12"),
        (b"X1", "x1"),
        ("É1", "É1"),  # lower-casing is ASCII-only on every backend
        (0, None),
        ("000", None),
        ("   ", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        (b"\xff\xfe", None),
    ],
)
def test_normalise_value(raw, expected) -> None:
    assert normalise_value(raw) == expected


def test_value_hash_is_pinned() -> None:
    # First eight bytes of MD5, big-endian: ClickHouse halfMD5.
    assert value_hash("abc") == 10376663631224000432
    assert value_hash("500012345") == 10821994778187675251


def test_hash_samples_nest() -> None:
    values = [str(number) for number in range(1, 5_000)]
    coarse = {value for value in values if in_hash_sample(value, 16)}
    fine = {value for value in values if in_hash_sample(value, 4)}
    assert coarse < fine
    assert all(in_hash_sample(value, 1) for value in values)
    # A 1-in-16 sample holds about a sixteenth of the values.
    assert 200 < len(coarse) < 430


def test_sampling_rate() -> None:
    assert sampling_rate(1_000_000, 2_048) == 512
    assert sampling_rate(100, 2_048) == 1
    assert sampling_rate(None, 5) == 1
    assert sampling_rate(0, 5) == 1
    for rate in (1, 2, 64):
        assert check_rate(rate) == rate
    for bad in (0, 3, 12):
        with pytest.raises(ValueError):
            check_rate(bad)


def test_select_tables_patterns() -> None:
    tables = [
        TableInfo("wh", "public", "accounts"),
        TableInfo("wh", "public", "txn_2024"),
        TableInfo("wh", "audit", "txn_log"),
        TableInfo("lake", "raw", "accounts"),
    ]

    def names(**kwargs) -> list[str]:
        return [table.key for table in select_tables(tables, **kwargs)]

    assert len(names()) == 4
    assert names(include="TXN_*") == ["wh:public.txn_2024", "wh:audit.txn_log"]
    assert names(include="public.accounts") == ["wh:public.accounts"]
    assert names(include="lake:*") == ["lake:raw.accounts"]
    assert names(include=["accounts"], exclude=["lake:*"]) == ["wh:public.accounts"]
    assert names(exclude="*") == []


@pytest.fixture()
def source() -> tuple[FrameSource, TableInfo]:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "src": ["0A1", "0A1", "a2", "A3", None, " a2 "],
            "dst": ["A2", "a3", "A2", "a1", "A9", ""],
            "amount": [10.0, 20.0, np.nan, 40.0, 0.0, -5.0],
            "at": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 11:00",
                    "2024-01-02 00:00",
                    "2024-02-10 00:00",
                    None,
                    "2024-02-11 00:00",
                ]
            ),
            "flag": [True, False, True, True, False, False],
        }
    )
    built = FrameSource({"moves": frame}, name="my data")
    return built, built.tables()[0]


def _column(source: FrameSource, table: TableInfo, name: str):
    return next(column for column in source.columns(table) if column.name == name)


def test_frame_source_inventory(source) -> None:
    built, table = source
    assert built.name == "my_data"
    assert table.key == "my_data:moves" and table.approx_rows == 6
    kinds = {column.name: column.kind for column in built.columns(table)}
    assert kinds == {
        "id": "integer",
        "src": "string",
        "dst": "string",
        "amount": "float",
        "at": "datetime",
        "flag": "boolean",
    }
    assert built.row_count(table) == 6
    assert built.describe() == {"name": "my_data", "kind": "dataframe", "tables": 1}


def test_frame_source_statistics(source) -> None:
    built, table = source
    stats = built.column_stats(table, built.columns(table))
    assert stats["src"].non_null == 5 and stats["src"].distinct == 4
    assert stats["dst"].blank == 1
    assert stats["dst"].min_length == 0 and stats["dst"].max_length == 2
    amount = stats["amount"]
    assert (amount.non_null, amount.minimum, amount.maximum) == (5, -5.0, 40.0)
    assert amount.zeros == 1 and amount.negatives == 1
    assert amount.quantiles["median"] == 10.0
    assert stats["at"].minimum == pd.Timestamp("2024-01-01 10:00")
    assert all(item.distinct_exact for item in stats.values())

    top = built.top_values(table, [_column(built, table, "dst")], 2)
    assert top["dst"][0] == ("A2", 2)
    assert built.time_histogram(table, _column(built, table, "at"), "month") == [
        ("2024-01-01T00:00:00", 3),
        ("2024-02-01T00:00:00", 2),
    ]


def test_frame_source_sample_is_deterministic_and_filtered(source) -> None:
    built, table = source
    columns = built.columns(table)
    first = built.sample(table, columns, 3, total_rows=6, seed=7)
    again = built.sample(table, columns, 3, total_rows=6, seed=7)
    assert len(first) == 3
    pd.testing.assert_frame_equal(first, again)
    everything = built.sample(table, columns, 100, total_rows=6, seed=7)
    assert len(everything) == 6
    only = built.sample(
        table,
        columns,
        100,
        total_rows=6,
        seed=7,
        where=(_column(built, table, "src"), ["A2"]),
    )
    # Canonical matching: "a2" and " a2 " are the same value.
    assert sorted(only["id"]) == [3, 6]


def test_frame_source_values_and_probe(source) -> None:
    built, table = source
    src = _column(built, table, "src")
    dst = _column(built, table, "dst")
    samples = built.distinct_sample(table, {"src": (src, 1), "dst": (dst, 1)}, 10)
    # Raw values come back; blanks and nulls never do.
    assert samples["src"] == [" a2 ", "0A1", "A3", "a2"]
    assert samples["dst"] == ["A2", "A9", "a1", "a3"]
    assert built.distinct_sample(table, {"src": (src, 1)}, 2)["src"] == [" a2 ", "0A1"]

    present = built.probe(
        table, {"src": (src, ["a1", "a2", "zz"]), "dst": (dst, ["a9", "a1", "zz"])}
    )
    assert present == {"src": {"a1", "a2"}, "dst": {"a9", "a1"}}


def test_frame_source_links(source) -> None:
    built, table = source
    src = _column(built, table, "src")
    dst = _column(built, table, "dst")
    # Rows with both ends: 1-4 (row 5 has no src, row 6 has a blank dst).
    profile = built.pair_profile(table, src, dst)
    assert profile == {
        "rows": 4,
        "equal_rows": 1,  # a2 -> A2
        "distinct_pairs": 4,
        "distinct_pairs_exact": True,
    }
    fanout = built.fanout_sample(table, src, [dst], 1, 10)
    assert fanout.to_dict("records") == [
        {"key": "a1", "rows": 2, "dst": 2},
        {"key": "a2", "rows": 2, "dst": 1},
        {"key": "a3", "rows": 1, "dst": 1},
    ]
    assert built.neighbours(table, src, dst, ["a1", "a3"], 10) == [
        ("a1", "a2"),
        ("a1", "a3"),
        ("a3", "a1"),
    ]
    assert built.neighbours(table, src, dst, ["a1", "a3"], 1) == [("a1", "a2")]


def test_frame_source_never_mutates_its_frames() -> None:
    frame = pd.DataFrame({"a": ["x", "y", None], "b": [1, 2, 3]})
    before = frame.copy(deep=True)
    built = FrameSource({"t": frame})
    table = built.tables()[0]
    columns = built.columns(table)
    built.column_stats(table, columns)
    built.sample(table, columns, 2, total_rows=3, seed=1)
    built.distinct_sample(table, {"a": (columns[0], 1)}, 5)
    built.probe(table, {"a": (columns[0], ["x"])})
    pd.testing.assert_frame_equal(frame, before)
