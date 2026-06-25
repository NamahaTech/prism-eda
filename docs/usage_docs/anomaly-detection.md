# Anomaly detection

`anomaly_detection()` finds rows that deserve a closer look. It runs a battery of
deterministic statistical detectors and returns **review candidates** — ranked
signals for a human to inspect.

It does **not** label rows as confirmed anomalies. Real anomaly status depends on
domain meaning Prism can't see, so it surfaces candidates and tells you *why*
each was flagged.

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())
result = dataset.anomaly_detection(table="orders")
```

As a one-liner:

```python
result = pe.anomaly_detection("data/events.parquet")
```

> If the dataset has more than one table, pass `table=` to choose which one to
> analyse. With a single-table dataset you can omit it.

## A simple case: one extreme value

The `orders` table has one wildly out-of-range `amount`:

```python
result = dataset.anomaly_detection(table="orders")
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title} (confidence={finding.confidence})")
    print(f"    {finding.summary}")
    print(f"    → {finding.recommendation}")
```

```text
Anomaly review across 1 table(s): top signal — Univariate tail candidates in orders.amount. 1 prioritized candidate signal(s).
[high] Univariate tail candidates in orders.amount (confidence=0.82)
    12 row(s) (5.0%) sit outside robust tail thresholds.
    → Review the example rows before capping, filtering, or modeling them separately.
```

The summary leads with the **top signal**. Note that severity (`high`) and
confidence (`0.82`) are reported separately — severity is how much it should
draw your attention; confidence is how sure the detector is.

## A richer case: many detectors agree

Run it on `customers`, which has multiple numeric columns, and several
independent detectors weigh in:

```python
result = dataset.anomaly_detection(table="customers")
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
```

```text
Anomaly review across 1 table(s): top signal — Detector agreement on review rows in customers. 8 prioritized candidate signal(s).
[high] Detector agreement on review rows in customers
[medium] Univariate tail candidates in customers.signup_age
[medium] Multivariate outlier candidates in customers
[medium] Local-density review candidates in customers
[medium] Conditional anomaly candidates: monthly_spend given tenure_months
[medium] Conditional anomaly candidates: signup_age given tenure_months
[medium] Conditional anomaly candidates: signup_age given monthly_spend
[medium] Isolation Forest review candidates in customers
```

The **Detector agreement** finding is promoted to the top (`high`) precisely
because several independent methods flagged the *same* rows — convergence is a
much stronger signal than any single detector, so those rows are the best
starting point for review.

## The detectors

| Detector | What it flags |
|----------|---------------|
| **Univariate tail** | Values outside robust IQR / modified-z-score thresholds, per column |
| **Multivariate** | Rows with large robust-scaled multivariate scores |
| **Isolation Forest** | Rows ranked highest by an Isolation Forest (with seed-stability disclosure) |
| **Local Outlier Factor** | Rows ranked by local density contrast, where row count and dimensionality suit it |
| **Detector agreement** | Rows appearing in two or more ranked detector sets |
| **Conditional anomaly** | Rows that are surprising for one feature *given* a related feature |
| **Rare category** | Rare categorical values |
| **Rare label** | (Optional) rare values of a supplied `target` |

Univariate tails are gated on genuine extremity and conditional findings are
capped to the strongest pairs, so the list stays focused rather than reporting a
tail for every numeric column.

## Options

```python
result = dataset.anomaly_detection(
    table="orders",
    target="label",                 # optional: also summarize a rare-label column
    expected_contamination=0.02,    # optional: review-sizing assumption (~2% of rows)
    mode="deep",                    # quick | standard | deep — compute depth
)
```

- **`table`** — which table to analyse (required when the dataset has several).
- **`target`** — an optional column whose rare values should also be summarized.
- **`expected_contamination`** — an optional assumption about how many rows you
  expect to review. It sizes the candidate set; it is **not** a confirmed
  prevalence estimate.
- **`mode`** — compute depth (see [Context & configuration](context-and-config.md)).

## What you get back

- **Findings** — one per detector that produced a meaningful signal, ordered by
  severity, each citing its evidence.
- A **metric-table artifact** summarizing candidate signals, rendered in the
  report.
- A **transformation plan** with non-mutating review recommendations (Prism never
  drops or caps rows for you).

Inspect the candidate rows and contributing columns before treating anything as a
true anomaly. See [Results & evidence](results-and-evidence.md) to dig into the
evidence behind each finding.
