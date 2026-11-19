# Database sources

[`explore()`](exploration.md) reads through a *source*: an object that answers
narrow questions about tables (how many rows, which values are in this column,
which of these values exist over there). Three are built in.

| Source | Reads | Install |
|--------|-------|---------|
| `FrameSource` | pandas DataFrames in memory | core |
| `ClickHouseSource` | a ClickHouse server | `pip install "prism-eda[clickhouse]"` |
| `PostgresSource` | a Postgres database | `pip install "prism-eda[postgres]"` |

Database drivers are imported only when a database source is used. Passing
DataFrames or paths to `explore()` builds a `FrameSource` for you.

## What every source guarantees

1. **Read-only.** A source never creates, alters or writes anything. Postgres
   queries run inside `SET TRANSACTION READ ONLY`.
2. **Bounded.** Every query has a time limit, and every answer has a size cap.
3. **One connection per query.** A connection is opened for a question and
   closed straight after. Nothing is held open between questions, so a long run
   is not broken by an idle connection being dropped.
4. **Comparable.** Values are compared in one canonical form and sampled with
   one hash, computed identically in ClickHouse, Postgres and pandas.

## ClickHouse

```python
import prism_eda as pe

source = pe.ClickHouseSource(
    host="ch.internal",
    port=8123,
    username="reader",
    password="...",
    database="bank",
    secure=False,
    databases=None,          # None: the connection's database; a list; or "*"
    include_views=False,
    name="warehouse",
)
```

- `databases="*"` lists every non-system database the user can see.
- Dictionary, Kafka, RabbitMQ, NATS, Null, Set, Join and MaterializedView
  engines are skipped, as are materialized-view inner tables. Views are skipped
  unless `include_views=True`.
- Each query is sent with `max_execution_time`, `max_threads` and
  `max_memory_usage` from the source's limits. If the user's profile forbids
  changing settings (a `readonly=1` user), the source notices once and runs
  under the server's own limits from then on.

If your application already knows how to connect, pass a factory instead of
connection details. It is called for every query and must return a new
`clickhouse_connect` client:

```python
source = pe.ClickHouseSource(client_factory=my_app.new_clickhouse_client)
```

## Postgres

```python
source = pe.PostgresSource(
    host="pg.internal",
    port=5432,
    username="reader",
    password="...",
    database="core",
    schemas=None,            # None: every non-system schema; or a list
    include_views=False,
    name="core",
)
# or
source = pe.PostgresSource(dsn="host=pg.internal dbname=core user=reader")
```

- Tables the user cannot `SELECT` from are not listed. Partitions are explored
  through their parent table.
- `psycopg` (version 3) is used when installed, otherwise `psycopg2`.
- `numeric(p, 0)` columns are treated as integers, because that is how many
  schemas store identifiers.

To reuse your application's connection handling, pass `connect`, a callable
returning a new DB-API connection for each query:

```python
source = pe.PostgresSource(connect=my_app.new_postgres_connection)
```

## Limits

```python
limits = pe.SourceLimits(
    query_timeout=600,        # seconds one query may run
    connect_timeout=15,
    max_threads=4,            # ClickHouse only
    max_memory_bytes=4 * 1024**3,   # ClickHouse only
    retries=2,                # for dropped connections; a timeout is never retried
    max_sql_bytes=180_000,    # value lists are split to fit one statement
)
source = pe.ClickHouseSource(host="...", limits=limits)
```

The defaults are conservative because exploration is meant to run against
databases that are serving other work. A query that exceeds its limit is
recorded as a failed step for that table; the run continues.

## What is exact and what is estimated

Everything not listed here is computed over every row.

| | ClickHouse | Postgres |
|---|---|---|
| Row count | Exact | Exact (`count(*)`); the planner's estimate with a warning if that fails |
| Distinct count | `uniq`: exact below about 60,000, an estimate above | Exact up to 5 million rows; the planner's `n_distinct` above (needs `ANALYZE`) |
| Quantiles | t-digest estimate | Exact up to 5 million rows; from the row sample above |
| Most frequent values | Exact for columns with up to `exact_top_values_distinct` distinct values; from the row sample otherwise | Same |
| Distinct pairs of a link | `uniq`: as for distinct counts | Exact up to 5 million rows; not computed above |
| Row sample | Rows whose hash of up to three high-cardinality columns falls under a threshold | `TABLESAMPLE BERNOULLI` up to 5 million rows, `SYSTEM` (block-level) above |

An estimated distinct count is marked `distinct_exact: False` in the column's
statistics. Sample-based values carry `quantiles_basis` or `top_values_basis`
set to `"sample"`.

How much a run costs the database: per table, one aggregate pass for
statistics, one grouped pass per low-cardinality column (ClickHouse) or one for
all of them (Postgres), one pass per charted date column, one sample, and one
pass for identifier value samples. Relationship discovery adds two passes over
each table's identifier columns. Each link adds a handful of filtered passes
over its table.

## Writing your own source

Subclass `prism_eda.sources.Source` and implement its abstract methods; the
docstrings in `prism_eda/sources/base.py` are the contract. Two things must
hold for relationships across sources to work:

- canonical values must match `prism_eda.sources.base.normalise_value`;
- hash sampling must match `value_hash` (the first eight bytes of MD5,
  big-endian) masked with `rate - 1`.

`tests/test_database_sources.py` compares a source's answers with
`FrameSource` on the same data and is the quickest way to check a new one.
