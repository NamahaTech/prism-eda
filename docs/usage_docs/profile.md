# The baseline profile

`profile()` is the fastest way to understand the shape and quality of a dataset.
It answers: *how big is this, what types are the columns, and where are the
obvious data-quality problems?*

True to Prism's design, it computes a full per-column profile as **evidence**,
but only **promotes** something to a finding when it crosses a meaningful
threshold — so you get a short, prioritized list instead of a wall of stats.

Findings arrive in two separate channels, and the distinction is the point:

- **Issues** are things *wrong* with the data — missing, duplicated, mistyped, or
  placeholder values. Something to fix.
- **Alerts** are things *true of* the data that are not defects — two columns that
  move together, a column that is all-unique, the window a timestamp covers.
  Nothing to fix; some of it changes how you model.

Filing "these two columns correlate" next to "40% of this column is missing"
devalues both. `split_findings()` gives you the two lists:

```python
from prism_eda.evidence.models import split_findings

issues, alerts = split_findings(result.findings)
```

Each `Finding` also carries `finding.category`, which is `"quality_issue"` or
`"observation"`.

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
for finding in issues:
    print(f"[{finding.severity}] {finding.title}")
    print(f"    {finding.summary}")
    print(f"    → {finding.recommendation}")
```

```text
completed
Profiled 2 table(s), 320 rows, and 11 columns; found 2 data-quality issue(s) and 3 observation(s).
[medium] High missingness in customers.signup_age
    20 values (25.0%) are missing.
    → Investigate whether missingness is structural, erroneous, or informative before choosing a fill strategy.
[medium] Duplicate columns in customers
    2 columns hold identical values. Any model or correlation over this table double-counts what is really one variable.
    → Keep one column and drop the copies once you have confirmed which name downstream code uses.
```

```python
for finding in alerts:
    print(f"[{finding.severity}] {finding.title}\n    {finding.summary}")
```

```text
[info] customers.customer_id has all-unique values
    Every row holds a different value, so the column identifies rows rather than describing them.
[info] orders.order_id has all-unique values
    Every row holds a different value, so the column identifies rows rather than describing them.
[info] monthly_spend and plan are strongly associated in customers
    plan explains 77% of the variance in monthly_spend (correlation ratio 0.88).
```

### Issues the profile promotes

Every one of these is a defect: a value that is not what it claims to be, or a
shape the table should not have. All are computed on **every row** — these are
exact counts, never sampled estimates.

| Issue | When it fires | Severity |
|-------|---------------|----------|
| **Duplicate rows** | A table contains exact duplicate rows | `high` if ≥10% of rows, else `medium` |
| **High missingness** | A column is ≥20% missing | `high` if ≥50% missing, else `medium` |
| **Constant column** | All non-null values in a column are identical | `low` |
| **Inconsistent value formatting** | The same label appears under several spellings (`USA` / `usa` / `" USA"`), or values carry stray whitespace | `high` if ≥5% of rows or ≥5 labels affected, else `medium` |
| **Disguised missing values** | Placeholders `isna()` does not count: `N/A`, `unknown`, `?`, `-`, and similar; or a repeated negative sentinel (`-999`) in an otherwise non-negative column | `high` if the true missing rate reaches 20%, else `medium` |
| **Numbers stored as text** | ≥95% of a text column's values parse as numbers | `medium` |
| **Mixed value types** | One object column holds more than one Python type | `high` |
| **Dates stored as text** | ≥90% of values match a date layout, but the dtype is text | `medium` |
| **Mixed date formats** | Two or more date layouts in one column | `high` |
| **Ambiguous day/month order** | Some rows have a first component >12 and others a second >12, so no single parse is right | `high` |
| **Placeholder dates** | `1970-01-01`, `1900-01-01`, `1899-12-30`, `2099-12-31` | `high` |
| **Implausible dates** | Before 1800 or more than 100 years ahead | `high` |
| **Future-dated rows** | Timestamps after the moment of analysis | `medium` |
| **End date precedes start date** | A named start/end column pair where the end is earlier | `high` |
| **Duplicate columns** | Two columns hold identical values | `medium` |
| **Unnamed columns** | A header is blank or `Unnamed: 0` | `medium` |
| **Duplicated column headers** | The source repeated a header, so pandas renamed it `x.1` | `high` |

Thresholds stay deliberately conservative. "Every numeric column has a long tail"
is *not* a profile issue — that's noise. (Genuine distributional outliers are the
job of [anomaly detection](anomaly-detection.md).)

Two rules keep the list short. A column that is an exact copy of another is only
reported once, through **Duplicate columns** — its copy does not repeat every
defect of the original. And a pair of identical columns is reported as an issue,
not also as a correlation alert.

### Alerts the profile raises

| Alert | When it fires |
|-------|---------------|
| **Strong association** | A column pair reaches 0.80. Above 0.99 the summary says the two are effectively one variable |
| **All-unique values** | Every row holds a different value, so the column identifies rows rather than describing them |
| **Many distinct labels** | A categorical column has more than 100 distinct values |
| **Dominated by one label** | One value covers ≥90% of rows |
| **Time coverage** | The window a timestamp column spans, and the typical gap between rows |
| **Gaps in coverage** | A gap more than 3× the typical spacing |
| **Not in time order** | Rows are not stored sorted by their timestamp, so any rolling window over file order is wrong |

Distribution shape is deliberately *not* an alert. "This column is uniform" is a
description of a column, so it belongs on that column's card — not in a list of
things that need your attention.

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
stats: ['infinite_count', 'kurtosis', 'max', 'mean', 'median', 'min', 'negative_count', 'p5', 'p95', 'q1', 'q3', 'skewness', 'std', 'zero_count']
```

Infinities are counted in `infinite_count` and excluded from every other
statistic. Leaving them in silently turns `min`/`max` into ±inf and `mean` into
NaN, which reads as "no data" when the real answer is "one bad value".

Each column carries its physical type, an inferred **semantic type** and
**role** candidates, missingness, distinctness, robust numeric statistics, and
top values. See [Results & evidence](results-and-evidence.md) for the full
`ColumnCatalog` / `TableCatalog` structure.

A text column is inferred **categorical** when it has at most 50 distinct
values, or at most 5% distinct values up to an absolute cap of 200 — beyond
the cap it is treated as **text** no matter how large the table is, so name-
or ID-like columns on million-row tables are not mislabelled. Columns that
stay categorical with more than 100 distinct values (including columns you
explicitly declared as pandas `Categorical`) carry a **high-cardinality
warning** in `ColumnCatalog.warnings`, shown as an amber chip in the HTML
report's column profile.

## Distribution shape, and the fitted family

Every numeric column gets a histogram, a box summary, a plain-language shape
label, and — when one fits — the name of the standard distribution it follows.
All of it is banked as `profile_distribution` evidence and drawn on the column's
card in the report.

```python
shape = next(
    item.value for item in result.evidence
    if item.kind == "profile_distribution" and item.value["column"] == "monthly_spend"
)
print(shape["shape"]["label"], "—", "; ".join(shape["shape"]["descriptors"]))
print(shape["fit"]["family"], shape["fit"].get("quality"))
```

The label comes from moments plus a robust two-population gap test, and is one of
`bell-shaped`, `uniform`, `right-skewed`, `left-skewed`, `mildly right-skewed`,
`mildly left-skewed`, `bimodal`, or `constant`. Extra descriptors are added for
heavy tails, zero inflation, near-constant columns, and count-like data.

Family fitting tries Normal, Log-normal, Exponential, Gamma, Weibull, Uniform,
Beta, and Poisson — each only where its support can hold the data — and it
**abstains**:

```text
fit["family"] is None and fit["reason"] == "no_family_fits_well"
```

Three deliberate choices about how this is reported:

- **The statistic is a distance, not a test.** It is a Kolmogorov–Smirnov
  distance computed with parameters estimated from the same column, which makes
  the corresponding p-value optimistic. Prism therefore reports the distance as a
  fit-quality score (`close` ≤ 0.05, `approximate` ≤ 0.10) and computes no
  p-value at all. "p = 0.31, so the data is normal" is the most common way
  fitted-distribution output misleads people.
- **A more flexible family does not win by default.** Gamma contains the
  exponential, so on exponential data it always fits at least as closely.
  Continuous families are therefore selected by **AIC**, which charges two units
  per free parameter — the second parameter has to earn its place.
- **Counts get a count model.** Poisson is compared on distance rather than AIC,
  because a probability mass and a probability density are not on a comparable
  scale.

Fitting needs at least 30 values and 3 distinct ones; below that the reason is
`too_few_values`.

## Correlations and interactions

The profile measures every usable column pair, choosing the right statistic for
the pair of types and recording which one it used:

| Pair | Statistic | Notes |
|------|-----------|-------|
| numeric ↔ numeric | Spearman ρ | Pearson r is recorded alongside it. Spearman is the default because a monotone-but-curved relationship is still a relationship |
| categorical ↔ categorical | Cramér's V | With the Bergsma bias correction, which matters at the small contingency tables profiling produces |
| categorical ↔ numeric | correlation ratio η | The share of the numeric column's variance explained by group membership |

All three land in `[0, 1]` so one matrix can hold them, but they are not the same
quantity — `pair["method"]` says which was used for each pair.

```python
matrix = next(
    item.value for item in result.evidence
    if item.kind == "profile_association_matrix"
)
for pair in matrix["pairs"][:3]:
    print(pair["left"], pair["right"], round(pair["strength"], 2), pair["method"])
```

The report draws this as a heatmap, lists the pairs at or above 0.80, and then
plots the strongest pairs as scatters — because a correlation number says *how
much* two columns move together, while a scatter says *how*, which is where
curves, clusters, and ceilings show up. A selector plots any other pair.

## Missing structure and sample rows

Beyond the per-column missing rate, the profile computes which columns go missing
*together* (`profile_co_missingness`, a pairwise Jaccard over the missing masks).
Columns that are always blank on the same rows usually share a cause — one
optional section of a form, a field added partway through collection — which is a
different problem from values lost at random.

The report also shows the first and last rows as loaded, plus examples of any
exact duplicate groups. These rows stay local: the assisted-analysis layer
forwards finding text and evidence *identifiers*, never evidence values, so
banking sample rows does not widen what a model can see. See
[Privacy](privacy.md).

## Controlling depth on wide data

A 300-column table cannot show every chart, correlation, and scatter without
becoming both enormous and useless. `detail` chooses the budget:

```python
result = dataset.profile(detail="full")   # or the default, "standard"
```

| Limit | `standard` | `full` |
|-------|-----------|--------|
| Columns charted | 60 | 250 |
| Columns in the correlation matrix | 30 | 80 |
| Numeric columns in the scatter explorer | 10 | 20 |
| Highlighted scatter pairs | 6 | 12 |
| Points per highlighted scatter | 2,000 | 10,000 |
| Rows used for correlations and scatters | 50,000 | 200,000 |
| Sample rows shown (head and tail each) | 10 | 25 |

Whenever a limit actually removes something, the profile records an
`AnalysisWarning` and the report says so on the page — a truncated report that
looks complete is worse than a large one. Row sampling is deterministic, seeded,
and recorded in `result.sampling`. **Data-quality issues never sample**: they
make exact claims about defects, and a sampled count would be a guess wearing a
number's clothes.

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
