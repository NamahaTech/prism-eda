# Schema Discovery

Prism EDA schema discovery infers candidate primary keys, composite keys, and
directional cross-table relationships from related pandas tables, CSV files,
Parquet files, or Excel files.

It does not modify data or declare database constraints. Every output is a
candidate with supporting evidence and confidence.

## Usage

```python
import prism_eda as pe

dataset = pe.load(
    {
        "customers": customers_df,
        "orders": orders_df,
        "order_items": order_items_df,
    }
)

result = dataset.discover_schema(mode="standard")
result.to_html("schema-report.html")
result.to_json("schema-report.json")
```

The one-shot equivalent is:

```python
result = pe.discover_schema("data/", recursive=True)
```

## Key search

The recipe searches for minimal keys only. If `customer_id` qualifies by itself,
`customer_id + region` is not reported as another key.

Default maximum widths:

| Mode | Maximum key columns | Automatic row budget |
| --- | ---: | ---: |
| `quick` | 1 | 25,000 |
| `standard` | 2 | 100,000 |
| `deep` | 3 | 250,000 |

At most 12 likely columns per table are considered. Columns are prioritized using
distinctness and identifier-like names. Constant, empty, and non-string-named
columns are excluded from key combinations.

A key candidate must meet the default 98% uniqueness and 98% completeness
thresholds, and qualify on one of two routes.

**Identifier naming.** Recognized signals include IDs, keys, codes, numbers,
UUIDs, GUIDs, emails, usernames, SKUs, ISBNs, serials, and account identifiers.

**Composite natural key.** Every component is a dimension-like partition — a
discrete column (string, boolean, datetime, or integer) with at most one distinct
value per two rows. This is what makes `Location + Period` or `country + year +
sex` a key without any identifier vocabulary, which is how most real datasets are
keyed. Floating-point columns are never dimension-like, and integer columns named
like measures (`rate`, `ratio`, `percent`, `total`, `count`, `score`, `%`, and
similar) are excluded too, so an arbitrary unique measurement such as an `amount`
column is not presented as a likely primary key.

The natural-key route is restricted to composites on purpose. A dimension-like
column cannot be unique by itself, so any single column reaching that route is
unique *and* weakly named — the accidental key that a small lookup table produces,
which would otherwise be crowned a hub the whole schema hangs off.

Confidence combines uniqueness, completeness, and naming evidence. Sampled key
confidence receives an additional penalty and the report explicitly requires
full-table confirmation.

Override example:

```python
result = dataset.discover_schema(
    max_key_columns=3,
    min_key_uniqueness=0.995,
    min_key_completeness=1.0,
    sampling="disabled",
)
```

## Relationship search

Each candidate parent key is compared with compatible child-column combinations
in other tables. Prism EDA first checks:

- column-count compatibility;
- broad type-family compatibility;
- column and parent-table name similarity.

It then measures:

- distinct-value inclusion;
- child-row coverage;
- orphan child rows;
- unreferenced parent values;
- one-to-one versus one-to-many cardinality in the evaluated rows.

The default relationship thresholds are 90% value inclusion and 72% combined
confidence. Confidence combines inclusion, type compatibility, name similarity,
and parent-key confidence.

Relationships are directional:

```text
unique parent key -> referencing child columns
```

## Shared grains

Some datasets have no parent and no child. Public statistical releases and
warehouse conformed dimensions are commonly dozens of peer tables measuring
different things at the same coordinate — every table keyed by
`(Location, Period)`, none of them owning any other.

When the same column combination is a candidate key in **three or more tables**,
Prism EDA reports it as one shared grain instead of the directional links a
pairwise view would produce. Thirty-nine peer tables generate hundreds of
individually-true inclusion dependencies; the grain is the single fact an analyst
needs to join them.

Each grain separates the tables that are unique at it — which join directly —
from those that carry the same columns but repeat within them, which need an
aggregate or an extra key column first.

Once a shape clears the three-table bar, membership is settled by measurement
rather than by the naming heuristics that proposed it, so a table whose key was
never proposed still joins the grain when its columns are genuinely unique.

Relationships between two tables that both sit on a shared grain are suppressed,
including the case where one table is a single slice of it — a table holding one
period is trivially unique on `(Location, Dim1)`, which says the table is one
period wide, not that it identifies the periods in every other table.

Genuine star schemas are unaffected. A dimension's key is unique in the dimension
and duplicated in every fact that references it, so it never reaches three tables
and stays on the relationship path.

## Sampling

In automatic mode, expensive key and child-side relationship evaluations use a
deterministic sample when a mode's row budget is exceeded. Sampling creates
warnings and machine-readable `SamplingRecord` entries.

The full parent key is retained when checking child inclusion. This avoids the
invalid comparison that would result from independently sampling both sides of a
join. Parent-coverage counts are still conservative when child rows are sampled.

Use `sampling="disabled"` to request full evaluation. This may require substantial
memory and time and does not override algorithm applicability limits.

## Result interpretation

Schema reports contain:

- candidate-key evidence;
- shared-grain evidence, listing the tables unique at the grain and those that
  repeat within it;
- candidate-relationship evidence;
- orphan-row findings when a relationship clears the threshold;
- a self-contained interactive ER diagram (embedded Cytoscape.js, offline)
  with draggable layered table cards, candidate `PK`/`FK`/`GRAIN` roles, routed
  connectors, confidence badges, and `1`/`N` cardinality labels — degrading
  to a static SVG when JavaScript is unavailable;
- sampling and reproducibility metadata;
- the normal table and column catalog.

Useful statuses:

- `completed`: at least one relationship or shared grain was found without
  warnings.
- `completed_with_warnings`: candidates exist but sampling or another caveat
  applies.
- `no_meaningful_structure`: analysis ran but no relationship or shared grain met
  the threshold.
- `insufficient_evidence`: fewer than two tables or no usable rows were supplied.

## Limitations

- Candidate relationships require type compatibility; numeric IDs stored as text
  are not automatically coerced.
- Self-referential relationships are not searched yet.
- Approximate functional dependencies and denormalization detection are not yet
  implemented.
- Semantic plausibility is based on names and simple type families. Domain
  confirmation remains necessary.
- Natural keys built from high-cardinality columns — where nearly every row has
  its own value — are still missed, because that shape is indistinguishable from
  a measurement without domain knowledge.
- An integer measure that is not named like one can still be admitted as a key
  component.
- Shared grains require the same column *names* across tables; the same grain
  spelled differently in each table is not matched.
- A sampled candidate can overestimate uniqueness or inclusion.
- Composite key width is capped at three to control combinatorial cost.
- Dense ER diagrams prioritize readable entities and non-overlapping cards, but
  relationship lines may still cross when the inferred graph is highly connected.
  The canvas expands and scrolls instead of shrinking labels into unreadable text.
