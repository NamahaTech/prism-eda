# Schema discovery

`discover_schema()` looks across the tables in a dataset and proposes **candidate
keys** and **relationships** — the primary keys, composite keys, and foreign-key
links you'd otherwise reconstruct by hand.

Crucially, these are *candidates*. Prism never treats an inferred relationship as
a declared database constraint; it gives you the evidence and asks you to confirm
the business meaning.

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())          # customers + orders
result = dataset.discover_schema()
```

As a one-liner over a folder of files:

```python
result = pe.discover_schema("data/", recursive=True)
```

## What it reports

```python
print(result.status)
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"    {finding.summary}")
    print(f"    → {finding.recommendation}")
```

```text
completed
Found 2 candidate key(s) and 1 candidate relationship(s) across 2 tables.
[low] Candidate one-to-many relationship: orders.customer_id → customers.customer_id
    orders.customer_id is 100.0% contained in customers.customer_id; confidence is 100.0%.
    → Confirm the business meaning and expected join cardinality before treating this candidate as a foreign key.
```

Prism found that `customers.customer_id` is a candidate key and that every
`orders.customer_id` value is contained in it — a **one-to-many** relationship
from `orders` (many) to `customers` (one). It's surfaced at `low` severity
because a relationship candidate is informational, not a problem to fix.

## What schema discovery looks for

- **Single-column candidate keys** — columns that are highly unique and complete.
- **Composite candidate keys** — minimal multi-column keys (key width capped at
  three columns).
- **Directional relationships** — inclusion dependencies between a child column
  and a candidate key in another table, with **one-to-one** vs **one-to-many**
  cardinality.
- **Orphans and unreferenced parents** — child rows whose key has no parent, and
  parent rows never referenced.
- **Confidence and sampling disclosure** for every candidate.

To keep candidates trustworthy, one-to-one candidates require key-name agreement,
which suppresses coincidental ID-range overlaps between unrelated tables.

## Tuning the thresholds

`discover_schema()` exposes the evidence thresholds it uses. The defaults are
strict on purpose; loosen them to surface weaker candidates, tighten them to
demand stronger evidence:

```python
result = dataset.discover_schema(
    max_key_columns=2,               # cap composite-key width (default: up to 3)
    min_key_uniqueness=0.98,         # how unique a candidate key must be
    min_key_completeness=0.98,       # how non-null a candidate key must be
    min_relationship_inclusion=0.9,  # fraction of child values found in the parent
    min_relationship_confidence=0.72 # overall confidence floor for a relationship
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_key_columns` | `None` (mode-based, ≤3) | Maximum composite-key width to consider |
| `min_key_uniqueness` | `0.98` | Minimum unique-rate for a candidate key |
| `min_key_completeness` | `0.98` | Minimum non-null rate for a candidate key |
| `min_relationship_inclusion` | `0.9` | Minimum child-value inclusion in the parent key |
| `min_relationship_confidence` | `0.72` | Minimum confidence to report a relationship |

## The ER diagram artifact

Schema discovery produces a `schema_graph` **artifact** — a self-contained entity
diagram with PK/FK roles, routed relationships, confidence badges, and one/many
cardinality marks. In the HTML report it renders as an **interactive diagram**
(powered by an embedded, vendored Cytoscape.js — still fully offline): drag
table cards to rearrange, scroll to zoom, click a table to focus its
relationships, click an edge for cardinality and confidence detail, and toggle
tables to declutter. Relationship endpoints are labelled **1** (parent side)
and **N** (child side, one row can match many); the legend explains every
symbol. Without JavaScript the report shows a static SVG version of the same
diagram. The graph is also available as structured data:

```python
graph = next(a for a in result.artifacts if a.kind == "schema_graph")
print(graph.title)
print("nodes:", len(graph.data["nodes"]), "edges:", len(graph.data["edges"]))
```

```text
Candidate schema graph
nodes: 2 edges: 1
```

Export the full report to see it drawn:

```python
result.to_html("schema-report.html")
```

## When no relationship is meaningful

If no relationship clears the thresholds, Prism says so rather than inventing one.
You'll get a `no_meaningful_structure` status (candidate keys may still be
reported) — for example, two unrelated tables whose only shared trait is an
integer ID range won't be linked.

## Known limitation

Schema discovery can still propose a coincidental cross-named **one-to-many**
candidate when two unrelated unique-ID columns share a value range; the
confidence model currently weights value inclusion heavily. Always confirm a
candidate's business meaning before relying on it. Tracking and approach are in
the [implementation status](../implementation-status.md). For the deeper design
of this recipe, see the [schema discovery design note](../schema-discovery.md).
