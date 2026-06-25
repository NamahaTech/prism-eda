# The baseline profile

`profile()` is the fastest way to understand the shape and quality of a dataset.
It answers: *how big is this, what types are the columns, and where are the
obvious data-quality problems?*

True to Prism's design, it computes a full per-column profile as **evidence**,
but only **promotes** a problem to a finding when it crosses a meaningful
threshold — so you get a short, prioritized list instead of a wall of stats.

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())
result = dataset.profile()
```

`profile()` and its alias `minimal_eda()` are equivalent. As a one-liner:

```python
result = pe.profile(load_sample())          # load + profile in one call
result = pe.minimal_eda("data/customers.csv")
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
Profiled 2 table(s), 320 rows, and 11 columns; found 1 prioritized issue(s).
[medium] High missingness in customers.signup_age
    20 values (25.0%) are missing.
    → Investigate whether missingness is structural, erroneous, or informative before choosing a fill strategy.
```

### Findings the profile promotes

| Finding | When it fires | Severity |
|---------|---------------|----------|
| **Duplicate rows** | A table contains exact duplicate rows | `high` if ≥10% of rows, else `medium` |
| **High missingness** | A column is ≥20% missing | `high` if ≥50% missing, else `medium` |
| **Constant column** | All non-null values in a column are identical | `low` |

These thresholds are deliberately conservative. "Every numeric column has a long
tail" is *not* a profile finding — that's noise. (Genuine distributional
outliers are the job of [anomaly detection](anomaly-detection.md).)

## The full per-column detail is still there

Even though only one finding was promoted above, Prism computed a complete
profile for all 11 columns and stored it as evidence and in the catalog. To reach
column-level detail:

```python
catalog = result.catalog

# The catalog stores one ColumnCatalog per column:
col = next(c for c in catalog.table("customers").columns if c.name == "signup_age")
print(col.physical_type, col.semantic_type, col.roles)
print("missing:", col.missing_count, f"({col.missing_rate:.0%})")
print("unique:", col.unique_count, f"({col.unique_rate:.0%})")
print("stats:", sorted(col.statistics))
```

```text
float64 numeric ('measure_candidate',)
missing: 20 (25%)
unique: 32 (53%)
stats: ['max', 'mean', 'median', 'min', 'q1', 'q3', 'std']
```

Each column carries its physical type, an inferred **semantic type** and
**role** candidates, missingness, distinctness, robust numeric statistics, and
top values. See [Results & evidence](results-and-evidence.md) for the full
`ColumnCatalog` / `TableCatalog` structure.

## The transformation plan

Where a finding implies a concrete data-prep action, the profile records a
**non-mutating** recommendation in `result.transformation_plan`. Prism never
applies these — it surfaces them for you to review.

```python
for step in result.transformation_plan.steps:
    print(step.operation, "on", step.table, step.columns)
    print("   rationale:", step.rationale)
    print("   risk:", step.risk, "| requires_approval:", step.requires_approval)
```

```text
review_missing_values on customers ('signup_age',)
   rationale: Missingness is high enough to affect downstream analysis.
   risk: medium | requires_approval: True
```

Each step cites the evidence that motivated it (`step.evidence_ids`). See
[Results & evidence](results-and-evidence.md#transformation-plans) for details.

## When there's nothing to profile

If a dataset has no rows or no columns, the profile doesn't invent conclusions —
it returns an `insufficient_evidence` status:

```python
import pandas as pd
empty = pe.profile(pd.DataFrame(columns=["id", "value"]))
print(empty.status)                       # insufficient_evidence
print(empty.warnings[0].code)             # insufficient_rows
```

If you genuinely want a best-effort result anyway, pass
`allow_insufficient_evidence=True` — the result is returned but stays visibly
qualified with `completed_with_warnings`. See
[Context & configuration](context-and-config.md) for that flag and other knobs.

## Next steps

- Relationships across multiple tables → [Schema discovery](schema-discovery.md)
- Rows that deserve a closer look → [Anomaly detection](anomaly-detection.md)
- Is this data model-ready? → [Classification readiness](classification.md)
