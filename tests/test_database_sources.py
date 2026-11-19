"""ClickHouse and Postgres sources must answer exactly as FrameSource does.

These tests need a live server and are skipped unless one is configured:

    PRISM_TEST_CLICKHOUSE_URL=http://user:password@host:8123/database
    PRISM_TEST_POSTGRES_DSN="host=... port=5432 user=... dbname=..."

Each backend gets its own throwaway database (ClickHouse) or schema (Postgres),
named ``prism_test_<random>``, created at the start and dropped at the end.
Nothing else on the server is read or written.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pytest

import prism_eda as pe
from examples.bank_data import bank_tables
from prism_eda.sources import (
    ClickHouseSource,
    FrameSource,
    PostgresSource,
    Source,
    TableInfo,
)
from prism_eda.sources.base import normalise_value

CLICKHOUSE_URL = os.environ.get("PRISM_TEST_CLICKHOUSE_URL")
POSTGRES_DSN = os.environ.get("PRISM_TEST_POSTGRES_DSN")
SCRATCH = f"prism_test_{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return bank_tables(customers=300, transactions=6_000)


@pytest.fixture(scope="module")
def reference(frames) -> FrameSource:
    return FrameSource(frames)


def _clickhouse_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "Int64"
    if pd.api.types.is_float_dtype(series):
        return "Float64"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
    return "Nullable(String)" if series.isna().any() else "String"


def _postgres_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


@pytest.fixture(scope="module")
def clickhouse(frames) -> Iterator[ClickHouseSource]:
    if not CLICKHOUSE_URL:
        pytest.skip("PRISM_TEST_CLICKHOUSE_URL is not set")
    clickhouse_connect = pytest.importorskip("clickhouse_connect")
    url = urlparse(CLICKHOUSE_URL)
    options: dict[str, Any] = {
        "host": url.hostname,
        "port": url.port or 8123,
        "username": url.username or "default",
        "password": url.password or "",
        "secure": url.scheme == "https",
    }
    admin = clickhouse_connect.get_client(**options)
    admin.command(f"CREATE DATABASE {SCRATCH}")
    try:
        for name, frame in frames.items():
            columns = ", ".join(
                f"`{column}` {_clickhouse_type(frame[column])}"
                for column in frame.columns
            )
            admin.command(
                f"CREATE TABLE {SCRATCH}.{name} ({columns}) "
                "ENGINE = MergeTree ORDER BY tuple()"
            )
            if len(frame):
                admin.insert_df(f"{SCRATCH}.{name}", frame)
        admin.command(
            f"CREATE VIEW {SCRATCH}.v_accounts AS SELECT * FROM {SCRATCH}.accounts"
        )
        yield ClickHouseSource(
            host=options["host"],
            port=options["port"],
            username=options["username"],
            password=options["password"],
            secure=options["secure"],
            database=SCRATCH,
            name="clickhouse",
        )
    finally:
        admin.command(f"DROP DATABASE IF EXISTS {SCRATCH}")
        admin.close()


@pytest.fixture(scope="module")
def postgres(frames) -> Iterator[PostgresSource]:
    if not POSTGRES_DSN:
        pytest.skip("PRISM_TEST_POSTGRES_DSN is not set")
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(POSTGRES_DSN, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {SCRATCH}")
        try:
            for name, frame in frames.items():
                columns = ", ".join(
                    f'"{column}" {_postgres_type(frame[column])}'
                    for column in frame.columns
                )
                admin.execute(f"CREATE TABLE {SCRATCH}.{name} ({columns})")
                if not len(frame):
                    continue
                names = ", ".join(f'"{column}"' for column in frame.columns)
                with admin.cursor() as cursor:
                    with cursor.copy(
                        f"COPY {SCRATCH}.{name} ({names}) FROM STDIN"
                    ) as copy:
                        for row in frame.itertuples(index=False):
                            copy.write_row(
                                tuple(None if pd.isna(cell) else cell for cell in row)
                            )
                admin.execute(f"ANALYZE {SCRATCH}.{name}")
            admin.execute(
                f"CREATE VIEW {SCRATCH}.v_accounts AS SELECT * FROM {SCRATCH}.accounts"
            )
            yield PostgresSource(dsn=POSTGRES_DSN, schemas=[SCRATCH], name="postgres")
        finally:
            admin.execute(f"DROP SCHEMA IF EXISTS {SCRATCH} CASCADE")


@pytest.fixture(scope="module", params=["clickhouse", "postgres"])
def backend(request) -> Source:
    return request.getfixturevalue(request.param)


def _table(source: Source, name: str) -> TableInfo:
    return next(table for table in source.tables() if table.name == name)


def _column(source: Source, table: TableInfo, name: str):
    return next(column for column in source.columns(table) if column.name == name)


def _both(backend: Source, reference: FrameSource, name: str):
    return _table(backend, name), _table(reference, name)


def test_inventory(backend, reference, frames) -> None:
    listed = {table.name: table for table in backend.tables()}
    # Views are left out unless asked for.
    assert set(listed) == set(frames)
    assert all(table.schema == SCRATCH for table in listed.values())
    for name, frame in frames.items():
        assert backend.row_count(listed[name]) == len(frame)
        kinds = {c.name: c.kind for c in backend.columns(listed[name])}
        expected = {c.name: c.kind for c in reference.columns(_table(reference, name))}
        assert kinds == expected, name


def test_views_can_be_included(backend, clickhouse_or_postgres_views) -> None:
    assert "v_accounts" in {table.name for table in clickhouse_or_postgres_views}


@pytest.fixture()
def clickhouse_or_postgres_views(backend) -> list[TableInfo]:
    if isinstance(backend, ClickHouseSource):
        widened: Source = ClickHouseSource(
            client_factory=backend._client, databases=[SCRATCH], include_views=True
        )
    else:
        assert isinstance(backend, PostgresSource)
        widened = PostgresSource(
            connect=backend._open, schemas=[SCRATCH], include_views=True
        )
    return widened.tables()


@pytest.mark.parametrize("name", ["accounts", "transactions", "devices", "customers"])
def test_column_statistics(backend, reference, name) -> None:
    table, frame_table = _both(backend, reference, name)
    found = backend.column_stats(table, backend.columns(table))
    expected = reference.column_stats(frame_table, reference.columns(frame_table))
    assert set(found) == set(expected)
    for column, want in expected.items():
        got = found[column]
        assert got.rows == want.rows, column
        assert got.non_null == want.non_null, column
        assert got.distinct == want.distinct, column
        assert got.blank == want.blank, column
        if want.mean is not None:
            assert got.minimum == pytest.approx(want.minimum), column
            assert got.maximum == pytest.approx(want.maximum), column
            assert got.mean == pytest.approx(want.mean), column
            assert got.zeros == want.zeros and got.negatives == want.negatives
            assert got.quantiles["median"] == pytest.approx(
                want.quantiles["median"], rel=0.05
            ), column
        if want.max_length is not None:
            assert (got.min_length, got.max_length) == (
                want.min_length,
                want.max_length,
            ), column
        if isinstance(want.minimum, pd.Timestamp):
            assert got.minimum == want.minimum and got.maximum == want.maximum


def test_top_values(backend, reference) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    names = ["channel", "status"]
    found = backend.top_values(table, [_column(backend, table, n) for n in names], 50)
    expected = reference.top_values(
        frame_table, [_column(reference, frame_table, n) for n in names], 50
    )
    for name in names:
        assert {str(v): n for v, n in found[name]} == {
            str(v): n for v, n in expected[name]
        }
        counts = [count for _value, count in found[name]]
        assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("grain", ["day", "week", "month", "year"])
def test_time_histogram(backend, reference, grain) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    found = backend.time_histogram(table, _column(backend, table, "ts"), grain)
    expected = reference.time_histogram(
        frame_table, _column(reference, frame_table, "ts"), grain
    )
    assert found == expected


@pytest.mark.parametrize(
    ("name", "column"),
    [("transactions", "receiver"), ("accounts", "acct_no"), ("devices", "acc")],
)
@pytest.mark.parametrize("rate", [1, 4])
def test_distinct_sample_is_the_same_set_everywhere(
    backend, reference, name, column, rate
) -> None:
    table, frame_table = _both(backend, reference, name)
    found = backend.distinct_sample(
        table, {column: (_column(backend, table, column), rate)}, 100_000
    )[column]
    expected = reference.distinct_sample(
        frame_table, {column: (_column(reference, frame_table, column), rate)}, 100_000
    )[column]
    assert found == expected
    assert len(found) > 20


def test_probe(backend, reference, frames) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    accounts = [normalise_value(value) for value in frames["accounts"]["acct_no"]]
    wanted = [*accounts[:150], "no-such-value", "999999999999"]
    requests = lambda source, t: {  # noqa: E731
        "sender": (_column(source, t, "sender"), wanted),
        "receiver": (_column(source, t, "receiver"), wanted),
    }
    found = backend.probe(table, requests(backend, table))
    expected = reference.probe(frame_table, requests(reference, frame_table))
    assert found == expected
    assert 0 < len(found["sender"]) <= 150

    # Integers on one side, zero-padded text on the other.
    devices, frame_devices = _both(backend, reference, "devices")
    found = backend.probe(
        devices, {"acc": (_column(backend, devices, "acc"), accounts)}
    )
    expected = reference.probe(
        frame_devices, {"acc": (_column(reference, frame_devices, "acc"), accounts)}
    )
    assert found == expected and len(found["acc"]) > 50


def test_pair_profile(backend, reference) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    found = backend.pair_profile(
        table, _column(backend, table, "sender"), _column(backend, table, "receiver")
    )
    expected = reference.pair_profile(
        frame_table,
        _column(reference, frame_table, "sender"),
        _column(reference, frame_table, "receiver"),
    )
    assert found == expected


@pytest.mark.parametrize("rate", [1, 2])
def test_fanout_sample(backend, reference, rate) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    others = ["receiver", "channel"]
    found = backend.fanout_sample(
        table,
        _column(backend, table, "sender"),
        [_column(backend, table, name) for name in others],
        rate,
        100_000,
    )
    expected = reference.fanout_sample(
        frame_table,
        _column(reference, frame_table, "sender"),
        [_column(reference, frame_table, name) for name in others],
        rate,
        100_000,
    )
    left = found.sort_values("key").reset_index(drop=True).astype({"key": str})
    right = expected.sort_values("key").reset_index(drop=True).astype({"key": str})
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_neighbours(backend, reference, frames) -> None:
    table, frame_table = _both(backend, reference, "transactions")
    senders = sorted(
        {normalise_value(value) for value in frames["transactions"]["sender"]}
    )[:25]
    found = backend.neighbours(
        table,
        _column(backend, table, "sender"),
        _column(backend, table, "receiver"),
        senders,
        100_000,
    )
    expected = reference.neighbours(
        frame_table,
        _column(reference, frame_table, "sender"),
        _column(reference, frame_table, "receiver"),
        senders,
        100_000,
    )
    assert sorted(found) == sorted(expected)
    capped = backend.neighbours(
        table,
        _column(backend, table, "sender"),
        _column(backend, table, "receiver"),
        senders,
        7,
    )
    assert len(capped) == 7 and set(capped) <= set(expected)


def test_sample(backend) -> None:
    table = _table(backend, "transactions")
    columns = backend.columns(table)
    key = [_column(backend, table, "txn_id")]
    first = backend.sample(
        table, columns, 1_000, total_rows=5_530, seed=3, hash_columns=key
    )
    again = backend.sample(
        table, columns, 1_000, total_rows=5_530, seed=3, hash_columns=key
    )
    assert 700 <= len(first) <= 1_500
    assert sorted(first["txn_id"]) == sorted(again["txn_id"])
    assert pd.api.types.is_datetime64_any_dtype(first["ts"])
    assert pd.api.types.is_float_dtype(first["amount"])
    everything = backend.sample(
        table, columns, 100_000, total_rows=5_530, seed=3, hash_columns=key
    )
    assert len(everything) == backend.row_count(table)

    accounts = _table(backend, "accounts")
    flagged = backend.sample(
        accounts,
        backend.columns(accounts),
        500,
        total_rows=10,
        seed=3,
        hash_columns=[_column(backend, accounts, "acct_no")],
        where=(_column(backend, accounts, "is_flagged"), [1]),
    )
    assert len(flagged) > 0 and set(flagged["is_flagged"]) == {1}


def test_queries_are_read_only(backend) -> None:
    """A source cannot be made to write, even with SQL it did not build."""
    if isinstance(backend, PostgresSource):
        from prism_eda.sources import SourceError

        with pytest.raises(SourceError):
            backend._run(f"CREATE TABLE {SCRATCH}.should_not_exist (a int)", None, "x")
        assert "should_not_exist" not in {t.name for t in backend.tables()}


def test_exploration_matches_the_in_memory_run(backend, frames) -> None:
    """End to end: the same relationships and links from a database."""

    def summary(result) -> tuple[set[tuple[str, str, str]], int, set[str]]:
        pairs = {
            (
                f"{item.value['child']['table']}.{item.value['child']['column']}",
                f"{item.value['parent']['table']}.{item.value['parent']['column']}",
                item.value["tier"],
            )
            for item in result.evidence
            if item.kind == "explore_relationship"
        }
        links = sum(1 for item in result.evidence if item.kind == "explore_link")
        gaps = {f.title for f in result.findings if f.title.startswith("Gap in")}
        return pairs, links, gaps

    from_database = pe.explore([backend])
    in_memory = pe.explore(frames)
    assert not from_database.failures
    assert summary(from_database) == summary(in_memory)
    assert from_database.catalog.row_count == in_memory.catalog.row_count
