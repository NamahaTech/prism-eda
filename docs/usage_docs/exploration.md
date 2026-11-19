# Deep exploration

`explore()` is for data you have not seen before: a folder of extracts, or a
whole database. It profiles every table, finds the columns that hold the same
identifiers (in one table, across tables, or across databases) without relying
on their names, and measures the networks those links form.

Use it when the question is "what is in here, and how does it fit together?"
For a single table you already understand, the [baseline profile](profile.md)
is the lighter tool.

## Quick start

### DataFrames and files

```python
import prism_eda as pe
from examples.bank_data import bank_tables

result = pe.explore(bank_tables())
print(result.summary)
result.to_html("exploration.html")
result.to_pdf("exploration.pdf")      # needs the pdf extra
```

```text
Explored 6 table(s) holding 66,700 rows and 32 columns. Found 7 relationship(s)
between columns, 1 record-to-record link(s), 5 data-quality issue(s) and 2
observation(s).
```

`explore()` accepts anything `pe.load()` does (a DataFrame, a path, a folder, a
mapping of names to frames), or a `Dataset`.

### ClickHouse and Postgres

```bash
pip install "prism-eda[clickhouse,postgres,pdf]"
```

```python
import prism_eda as pe

warehouse = pe.ClickHouseSource(
    host="ch.internal", username="reader", password="...", database="bank"
)
core = pe.PostgresSource(
    host="pg.internal", username="reader", password="...", database="core"
)

result = pe.explore([warehouse, core], exclude=["tmp_*", "*_backup"])
result.to_pdf("bank-exploration.pdf", branding=False)
```

Several sources can be explored in one run, and relationships are found across
them. Each source needs a distinct `name`. See
[Database sources](database-sources.md) for connection options and limits.

## What a run does

1. **Inventory.** Every table of every source is listed with its size.
2. **Per-table profile.** For each table:
   - computed over **every row**: row count, missing and blank values, distinct
     counts, ranges, quantiles, most frequent values, and rows per period for
     the busiest date columns;
   - computed on a **deterministic row sample** (50,000 rows by default):
     distribution shapes, correlations, format checks (disguised missing
     values, mixed formats, numbers stored as text), a weekday-by-hour activity
     grid, and a few example rows.
3. **Relationships.** Identifier-like columns are compared by value.
4. **Links.** Where two columns of one table hold the same kind of identifier,
   the network they form is measured.
5. **Hint-driven sections**, if you supplied hints.

Sample-based evidence carries `metadata["basis"] == "sample"` and a
`SamplingRecord`; full-table evidence carries `"full_table"`. Reports label the
two. A distinct count the database estimated rather than counted is marked
`distinct_exact: False` in the column statistics and `≈` in reports.

## Choosing tables

`include` and `exclude` are case-insensitive glob patterns. A pattern is tested
against the bare table name, `schema.table`, and `source:schema.table`:

```python
pe.explore(sources, include=["txn_*", "public.accounts"])
pe.explore(sources, exclude=["warehouse:staging.*", "*_tmp"])
```

No `include` means every table the source lists. System schemas are never
listed; views are skipped unless the source was built with
`include_views=True`.

## Relationships

### How they are found

Column names are not consulted. For each table, columns that could be
identifiers are picked: integer or text, at least `min_key_distinct` (30)
distinct values, and for text an average length of at most `max_key_length`
(64). Text that parses as dates is left out. The widest
`max_key_columns` (16) per table are kept.

Values are compared in a **canonical form**: surrounding spaces trimmed, ASCII
letters lower-cased, leading zeros removed. `000512345678`, `512345678` and the
integer `512345678` are the same value. A value with nothing left (blank, or
all zeros) never takes part.

Then, for every pair of those columns:

1. **Sample.** Each column contributes a hash sample of about `value_sample`
   (2,048) distinct values. The hash is the same on every backend, so the
   sample of a ClickHouse column and of a Postgres column are comparable.
2. **Screen.** `screen_values` (256) values per column are looked up in every
   other column, one scan per table.
3. **Verify.** Pairs with hits get the full sample looked up, plus the values
   of a few thousand sampled *rows*.

A lookup is exact for the values asked about, whatever the table size. The
only estimate is which values were asked about, and each coverage figure comes
with a 95% Wilson interval that says how tight it is.

### What is reported

Each `explore_relationship` evidence holds:

| Field | Meaning |
|-------|---------|
| `coverage` | Share of the first column's distinct values that exist in the second |
| `coverage_interval` | 95% interval for `coverage` |
| `row_coverage` | The same, weighted by how often each value occurs |
| `reverse_coverage` | Share of the second column's values found in the first |
| `kind` | `many_to_one` (the second column is unique), `one_to_one`, or `shared_values` |
| `same_table` | Both columns are in one table: a link between records |
| `cross_source` | The columns are in different sources |
| `tier` | `strong`, `likely` or `possible` (below) |
| `format_note` | Set when the two columns write the same values differently |
| `primary` | Whether the pair belongs in the short list (below) |

Low coverage is not a reason to discard a pair. "5% of counterparty accounts
are accounts we hold" is exactly the fact worth knowing, so a pair is reported
once `min_matches` (5) sampled values match and they are at least
`min_coverage` (0.5%) of the sample.

Columns connected by relationships are grouped into **identifier groups**
(`explore_domain`). The group's *reference* is the unique column with the most
values, if there is one. Five columns holding account numbers relate in ten
pairs; a pair is `primary` when it involves the reference, when its group has
none, or when it links two columns of one table. Reports list primary pairs
first.

### Small integers

Being name-blind has one known failure: small integers overlap by coincidence.
A serial `id` running 1..N is "contained" in every other serial, and a
`duration_ms` column fits inside any larger id range.

So when a column's values are all digits with a median length under
`weak_value_length` (7), a pair is kept only if the other column is unique, at
least 90% of the values are found, and the two are not both unique (two serials
are never related on values alone). It is then tiered:

- `likely` if the names share a word root (`cust_ref` and `customers`). This is
  the only use of names, and only as a tie-break for pairs that values cannot
  settle.
- `possible` otherwise, and only when the column also uses at least half of the
  other's values. Possible pairs are listed separately, never grouped, never
  followed as links, and never raise a finding.

A real reference from an unhelpfully named column to a small-integer key can
therefore land in `possible`. That is the honest answer: the values do not
distinguish it from coincidence.

### Format mismatches

When two related columns write values differently (leading zeros, letter case,
surrounding spaces), a plain equality join between them matches nothing. This
is raised as a high-severity finding, once per odd column rather than once per
column it disagrees with. A difference in storage type alone (integer against
text) is noted on the relationship but is not a finding.

## Links between records

When two columns of one table share identifiers (sender and receiver, employee
and manager), each row connects two records. For each such pair, up to
`max_links` (12), an `explore_link` evidence reports:

- rows with both ends present, self-links, and distinct pairs;
- the share of destination values that exist in the group's reference column,
  by distinct values and by rows;
- the share of destination values that also appear as origins, which is what
  makes multi-step paths possible;
- the distribution of distinct counterparties per record, in both directions;
- **reach**: starting from `hop_seeds` (300) sampled origins and following
  links outward for `max_hops` (3) steps, the share of starting records that
  get that far and how many records they reach;
- the share of first-step counterparties that have onward links, and the share
  of first-step links returned in the opposite direction;
- **connected groups**: exact for tables up to a million rows; above that,
  estimated by walking in both directions from sampled records.

All of this is measured on **sampled records with all of their rows**, never on
sampled rows, because sampling rows thins every record's links and understates
every figure. Reach follows the origin → destination direction only. Every walk
has a size cap (`hop_frontier`, `hop_pairs`); when it is hit the result says so
and the figures are lower bounds.

`explore_fanout` evidence covers the other identifier columns of a table: for
each value of one, how many distinct values of another it appears with. It is
what shows that a device is shared by forty accounts.

## Hints

Everything above needs no configuration. If you know what some columns mean,
say so:

```python
result = pe.explore(
    sources,
    hints={
        "label": {"table": "accounts", "column": "is_flagged", "positive": [1]},
        "entity": "sender",
        "timestamp": "transactions.ts",
        "amount": "transactions.amount",
    },
)
```

A hint is `"column"`, `"table.column"`, or a dict with `column`, optional
`table`, and for a label optional `positive` values.

| Hint | Adds |
|------|------|
| `label` | Labelled rows against the rest: which columns differ most; and for each link in the same identifier group, how many labelled records appear at each end, their number of counterparties against everyone's, and how many of their counterparties are themselves labelled |
| `entity` | Rows per entity in every table that has the column, and how concentrated activity is |
| `timestamp` | Which column the weekday-by-hour grid uses (otherwise the busiest timestamp) |
| `amount` | Round-figure heaping, the most repeated exact amounts, and a finding when one repeated amount sits just under a round figure |

Without `positive`, a two-valued label takes its rarer value as positive; a
label with more values needs `positive`. Fewer than 30 labelled rows is
reported as too few to compare. A column that separates labelled rows almost
perfectly was usually filled in because of the label; treat it as leakage until
its timing is understood.

## Configuration

Pass options as keywords, or build an `ExploreConfig`:

```python
pe.explore(sources, sample_rows=20_000, max_hops=2)
pe.explore(sources, config=pe.ExploreConfig(relationships=False))
```

| Option | Default | Meaning |
|--------|---------|---------|
| `sample_rows` | 50,000 | Rows sampled per table |
| `max_columns` | 200 | Columns profiled per table; wider tables are truncated with a warning |
| `exact_top_values_distinct` | 5,000 | Above this many distinct values, top values come from the sample |
| `top_values` | 10 | Most frequent values kept per column |
| `timelines_per_table` | 3 | Date columns charted per table |
| `relationships` | `True` | Run relationship discovery |
| `min_key_distinct` | 30 | Fewest distinct values for an identifier-like column |
| `max_key_columns` | 16 | Identifier-like columns considered per table |
| `max_key_length` | 64 | Longest average text length for an identifier |
| `value_sample` | 2,048 | Distinct values kept per column |
| `screen_values` | 256 | Values per column in the screening pass |
| `row_coverage_rows` | 4,000 | Sampled rows used for row-weighted coverage |
| `min_matches` | 5 | Matching values needed to report a pair |
| `min_coverage` | 0.005 | Smallest coverage reported |
| `weak_value_length` | 7 | All-digit values shorter than this use the stricter rule |
| `network` | `True` | Profile links and fan-out |
| `max_links` | 12 | Link pairs profiled, largest tables first |
| `degree_entities` | 20,000 | Records sampled for counterparty distributions |
| `hop_seeds` | 300 | Starting records for the reach walk |
| `hop_frontier` | 20,000 | Records followed per step before the walk is truncated |
| `hop_pairs` | 400,000 | Pairs fetched per step |
| `max_hops` | 3 | Steps walked (2 to 4) |
| `label_rows` | 20,000 | Labelled rows loaded for the label comparison |
| `seed` | 42 | Seed for every sample |

## Long runs: progress, cancelling, resuming

```python
def on_progress(report):
    print(f"{report.fraction:4.0%} {report.stage}: {report.message}")

store = pe.MemoryCheckpoints()
result = pe.explore(sources, progress=on_progress, checkpoints=store)
```

- **Progress.** `progress` is called with a `Progress(stage, message, done,
  total, fraction)`. `fraction` never decreases.
- **Cancelling.** An exception raised by the `progress` callback is not
  swallowed: raising is how you cancel. (Event `callbacks` are different: their
  exceptions are ignored, as everywhere else in the library.)
- **Resuming.** Each table's profile, the relationship stage and the link stage
  are saved to `checkpoints` as they finish. A later call with the same store,
  sources and settings loads them instead of recomputing. A store is any object
  with `load(key) -> bytes | None` and `save(key, payload)`; write your own to
  keep checkpoints in a database or on disk. Checkpoints are JSON, not pickles,
  and are keyed by the settings, so changing the config or hints starts fresh.
  Hint-driven sections are recomputed on resume.

## When something fails

One table or one step failing does not stop the run:

- a table that cannot be read is listed with status `failed` and a
  `table_not_profiled` warning;
- a statistics query that fails is retried column by column, so one unreadable
  column costs only itself;
- a failed step (a timeout on a timeline, a probe) is recorded in
  `result.failures` with its table and stage, and the status becomes
  `completed_with_warnings`.

Only a run in which no source can be listed raises.

## Outputs

```python
result.to_json("exploration.json")
result.to_html("exploration.html", branding=False, title="Core banking review")
result.to_pdf("exploration.pdf", branding=False, title="Core banking review")
```

The result is an ordinary [`AnalysisResult`](results-and-evidence.md) with
`goal == "exploration"`. Evidence kinds specific to it: `explore_inventory`,
`explore_table_summary`, `explore_timeline`, `explore_time_coverage`,
`explore_time_pattern`, `explore_relationship`, `explore_domain`,
`explore_link`, `explore_fanout`, `explore_label`, `explore_entity_activity`,
`explore_amount`. The relationship graph is an artifact of kind
`relationship_graph`. See [Exporting reports](exporting-reports.md) for the
PDF options.

Reports show example rows and most frequent values as they are stored. If the
data is sensitive and the report will travel, review it before sharing.

## Limits

- Relationships are candidates. A pair found from values is evidence that two
  columns share identifiers, not a declared constraint.
- Only single columns are compared. Composite keys are not discovered here;
  [schema discovery](schema-discovery.md) handles them for in-memory tables.
- Canonical matching lower-cases ASCII letters only, so identifiers that differ
  only in the case of non-ASCII letters are treated as different.
- A relationship covering less than about 2% of a column's values can be missed
  by the screening pass.
- Rare defects and rare values may be absent from the row sample.
- Reach follows one direction; a table whose link columns are stored
  destination-first reads backwards.
- Exact connected groups are computed only up to a million rows; above that the
  figure is an estimate from sampled records.
