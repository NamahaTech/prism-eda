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
Anomaly review across 1 table(s): top signal — Univariate tail candidates in orders.amount. 2 prioritized candidate signal(s).
[high] Univariate tail candidates in orders.amount (confidence=0.82)
    12 row(s) (5.0%) sit outside robust tail thresholds.
    → Review the flagged rows before capping, filtering, or modeling them separately.
[high] 1 row(s) to review in orders (confidence=0.78)
    1 row(s) stand out across multiple checks — each is listed below with its values, the typical baseline, and why it was flagged.
    → Open the flagged rows below and decide whether each is a data error, a rare valid case, or a separate regime.
```

The summary leads with the **top signal**. Severity (`high`) and confidence
(`0.82`) are reported separately — severity is how much it should draw your
attention; confidence is how sure the detector is.

The **Rows to review** finding is the one you act on. Its evidence carries the
actual flagged row — not just an index — so you can see the data without
reopening the file. Here the single review row is `order_id` 1010 with
`amount` 9,999, and the plain-language *why* is:

```text
amount 9,999 is 412× the typical 24.27 (above). Surfaced by 1 of 1 checks.
```

## A richer case: the review list leads

Run it on `customers`, which has multiple numeric columns and a planted
out-of-range `signup_age` of 121:

```python
result = dataset.anomaly_detection(table="customers")
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
```

```text
Anomaly review across 1 table(s): top signal — 5 row(s) to review in customers. 2 prioritized candidate signal(s).
[high] 5 row(s) to review in customers
[medium] Univariate tail candidates in customers.signup_age
```

Every detector (univariate tail, multivariate, Isolation Forest, Local Outlier
Factor, conditional) feeds a single **cross-detector consensus**: the rows are
ranked by how many independent checks agreed, and each row is shown with its
values, the typical baseline, and a one-line reason. Reading the consensus
evidence:

```python
review = next(e for e in result.evidence if e.kind == "anomaly_consensus_review")
for row in review.value["rows"]:
    print(f'{row["method_count"]}/{review.value["total_detectors"]}  {row["why"]}')
```

```text
5/5  signup_age 121 is 3× the typical 37 (above). Surfaced by 5 of 5 checks.
1/5  monthly_spend 48.47 is unusual for its tenure_months peer group (around tenure_months 45). Surfaced by 1 of 5 checks.
1/5  monthly_spend 48.22 is unusual for its tenure_months peer group (around tenure_months 43). Surfaced by 1 of 5 checks.
1/5  monthly_spend 47.16 is unusual for its tenure_months peer group (around tenure_months 43). Surfaced by 1 of 5 checks.
1/5  tenure_months 14 is unusual for its monthly_spend peer group (around monthly_spend 29). Surfaced by 1 of 5 checks.
```

The planted value (`signup_age` 121) leads because all five detectors agree on
it; the weaker, in-context candidates trail with a clear `1/5` agreement so you
know to deprioritize them. Convergence across independent methods is a much
stronger signal than any single detector.

## Distribution shape: outliers vs. two populations

A column with two separated clusters is not "a clean distribution with a few
tail outliers" — it is **two populations**, and saying so is usually more useful
than flagging the upper cluster as anomalies. Prism detects this with a robust
largest-gap test and reports it as its own finding, e.g.:

```text
[high] Salary looks like two populations
    Values split into two separated groups — a lower cluster of 26 row(s) and an
    upper cluster of 9 row(s), with a clear gap between. That is two regimes, not
    one distribution with a few tail outliers; see the distribution chart for the
    exact split.
```

The exact split boundaries live in the evidence (and the report's distribution
chart), never in the finding text — see the privacy note below.

## The detectors

| Detector | What it flags |
|----------|---------------|
| **Univariate tail** | Values outside robust IQR / modified-z-score thresholds, per column |
| **Multivariate** | Rows with large robust-scaled multivariate scores |
| **Isolation Forest** | Rows ranked highest by an Isolation Forest (with seed-stability disclosure) |
| **Local Outlier Factor** | Rows ranked by local density contrast, where row count and dimensionality suit it |
| **Conditional anomaly** | Rows that are surprising for one feature *given* a related feature |
| **Cross-detector consensus** | The ranked review list: rows the detectors above agree on, with values and a plain-language *why* |
| **Distribution shape** | Per-column histogram, box summary, and a two-population (regime split) check |
| **Rare category** | Rare categorical values |
| **Rare label** | (Optional) rare values of a supplied `target` |

The per-detector results no longer each become their own finding (which used to
restate the same rows several times); they feed the consensus instead. A row
joins the review list only when detectors agree, a value is genuinely extreme,
or a single conditional check is strong — so the list stays focused.

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

- **Verdict** — a single plain-language headline (`result.metadata["verdict"]`)
  leading with the strongest reframing: a two-population split when present,
  otherwise the row-review count with a concrete example. The HTML report shows
  it as the hero.
- **Findings** — led by the cross-detector review list and any distribution
  regime splits, then per-column tails, ordered by severity, each citing its
  evidence.
- **Review-row evidence** — the actual flagged rows with their values, the
  per-column contribution (robust-z vs. the typical value), and a plain-language
  *why*. Each row also carries an `explanations` block so every detector tag is
  backed by visible evidence: a *multivariate* flag by the full per-column
  joint-deviation profile, and a *conditional* flag by the peer group it stands
  out from (the peer band and where the row's value sits). The HTML report
  renders this as a table — expand a row to see the per-column σ chart and the
  peer-group strip — plus per-column histograms with the flagged values marked
  and a scatter of the most relevant pair.
- A **metric-table artifact** (collapsed in the report) showing each detector's
  contribution to the review list.
- A **transformation plan** with non-mutating review recommendations (Prism never
  drops or caps rows for you).

### A note on privacy

The HTML report shows full, unmasked rows — you are assumed to be authorized to
view the data you are analysing. Finding *summaries* deliberately carry only
counts and rates, never raw cell values, because the
[AI-assisted investigator](ai-assisted-analysis.md) forwards summaries to an LLM
and raw values must never leave your machine. Exact values live in the evidence
and the locally-rendered charts.

Inspect the flagged rows and contributing columns before treating anything as a
true anomaly. See [Results & evidence](results-and-evidence.md) to dig into the
evidence behind each finding.
