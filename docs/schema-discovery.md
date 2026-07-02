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
thresholds and a conservative semantic-name threshold. Recognized signals include
IDs, keys, codes, numbers, UUIDs, GUIDs, emails, usernames, SKUs, ISBNs, serials,
and account identifiers. This intentionally avoids presenting arbitrary unique
measures, such as an `amount` column, as likely primary keys. Confidence combines
uniqueness, completeness, and naming evidence. Sampled key confidence receives an
additional penalty and the report explicitly requires full-table confirmation.
For composite keys, weakly named components may contribute only when they are
string-like partitions; numeric measures cannot qualify merely by becoming unique
when paired with an identifier.

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
- candidate-relationship evidence;
- orphan-row findings when a relationship clears the threshold;
- a self-contained interactive ER diagram (embedded Cytoscape.js, offline)
  with draggable layered table cards, candidate `PK`/`FK` roles, routed
  connectors, confidence badges, and `1`/`N` cardinality labels — degrading
  to a static SVG when JavaScript is unavailable;
- sampling and reproducibility metadata;
- the normal table and column catalog.

Useful statuses:

- `completed`: at least one relationship met the threshold without warnings.
- `completed_with_warnings`: candidates exist but sampling or another caveat
  applies.
- `no_meaningful_structure`: analysis ran but no relationship met the threshold.
- `insufficient_evidence`: fewer than two tables or no usable rows were supplied.

## Limitations

- Candidate relationships require type compatibility; numeric IDs stored as text
  are not automatically coerced.
- Self-referential relationships are not searched yet.
- Approximate functional dependencies and denormalization detection are not yet
  implemented.
- Semantic plausibility is based on names and simple type families. Domain
  confirmation remains necessary.
- Unusually named natural keys may be missed by the conservative semantic-name
  filter.
- A sampled candidate can overestimate uniqueness or inclusion.
- Composite key width is capped at three to control combinatorial cost.
- Dense ER diagrams prioritize readable entities and non-overlapping cards, but
  relationship lines may still cross when the inferred graph is highly connected.
  The canvas expands and scrolls instead of shrinking labels into unreadable text.
