# Regression readiness

`regression("target")` asks the question you should settle *before* fitting a
model to a numeric outcome: **can this data support a regression at all, and
what will quietly go wrong if you fit one?**

It runs leakage screening, target-shape and censoring checks, redundancy and
multicollinearity, two cross-validated diagnostic probes, residual and
influence diagnostics — and leads with a **readiness verdict**.

It is a *readiness diagnostic*, not a training pipeline. It returns evidence and
findings, never a fitted production model.

```python
import prism_eda as pe
from examples.sample_data import subscriptions

result = pe.load({"subscriptions": subscriptions()}).regression("monthly_revenue")
```

As a one-liner:

```python
result = pe.regression("data/accounts.csv", target="revenue")
```

> Pass `table=` when the dataset has more than one table. The target must be a
> numeric column; a text or boolean target returns `insufficient_evidence` with
> a warning pointing you at [`classification()`](classification.md).

## The verdict and findings

```python
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title} (confidence={finding.confidence})")
    print(f"    {finding.summary}")
```

```text
subscriptions.monthly_revenue: not ready to model. Top issue — Potential target leakage: renewal_invoice_total. 7 prioritized issue(s) (1 critical, 2 high, 4 medium). 4 alert(s) are listed separately.
[critical] Potential target leakage: renewal_invoice_total (confidence=0.92)
    renewal_invoice_total explains 100.0% of the target's variance on its own.
[high] Identifier-like feature: account_id (confidence=0.9)
    account_id is unique on 100% of rows and labels records rather than explaining the target.
[high] Target values pile up at 200 (confidence=0.86)
    monthly_revenue takes the single value 200 in 21 rows (7.0%). A continuous quantity should not repeat like this; a cap, a default, or an imputed placeholder is the usual cause.
[medium] Redundant features carry the same information (confidence=0.84)
    seats and licenses_purchased correlate at 0.9999. Predictions are largely unaffected; individual coefficients are not interpretable.
[medium] Error spread changes across the prediction range (confidence=0.8)
    Residual spread varies 6.0x between the widest and narrowest part of the fitted range.
[medium] The fit is systematically biased in part of the range (confidence=0.8)
    At least one fitted decile is off by 1.01 residual standard deviations on average — a consistent direction, not noise.
[medium] 4 rows worth reviewing (confidence=0.76)
    These rows are extreme in the features, badly fitted, or both, so they move the fit more than any other rows.
[medium] A few rows distort the conventional fit (confidence=0.74)
    The robust probe's typical error is 35% lower than the least-squares probe's. The data is predictable; a minority of rows is pulling a squared-error fit.
[low] Target is strongly right skewed (confidence=0.88)
    monthly_revenue has skewness +1.50. A yeo_johnson transform reduces skew to +0.01.
[low] seats relates to the target non-linearly (confidence=0.8)
    A straight line explains 11.0% of the target, but a binned fit explains 25.7%. A linear model will under-use this feature.
[low] Parts of the range have almost no support (confidence=0.74)
    seats has a gap covering 38% of its range. Predictions there are extrapolation, however confident the model looks.
```

## Issues and alerts are different things

The list above is two lists. `split_findings` separates them, and the report
renders them as separate sections:

```python
from prism_eda.evidence.models import split_findings

issues, alerts = split_findings(result.findings)
print(len(issues), "issues,", len(alerts), "alerts")
```

```text
7 issues, 4 alerts
```

- **Issues** are things that are wrong or will mislead you: a leak, an
  identifier in the feature set, a target capped at a contract ceiling.
- **Alerts** are true, worth knowing, and not defects: the target is skewed, two
  features are interchangeable, part of the range is thinly supported.

**A skewed target is not a defect.** Filing it as one, next to a leak, devalues
both. So skew is always an alert — and rather than asserting that you should
log-transform, Prism *measures* what each transform actually does to this data:

```python
shape = next(e for e in result.evidence if e.kind == "regression_target_shape")
for candidate in shape.value["candidates"]:
    print(candidate["transform"], candidate["skewness_after"])
print("chosen:", shape.value["best_candidate"])
```

```text
log1p -0.8497871563481977
sqrt 0.5921765243550928
yeo_johnson 0.010925823542951225
chosen: yeo_johnson
```

The reflexive answer — `log1p` — *overcorrects* this target into left skew, and
`sqrt` does not get it under the threshold. Only the measured winner is
recommended. If nothing measurably helps, nothing is recommended.

## Two probes, and why their disagreement is the finding

Prism fits two cross-validated diagnostic probes: Ridge (conventional,
least-squares) and Huber (robust). Both are screened — leaks, identifiers, and
over-wide categoricals never enter the feature set.

```python
probe = next(e for e in result.evidence if e.kind == "regression_probe")
for model in [*probe.value["models"], probe.value["baseline"]]:
    print(f"{model['model']:16s} R²={model['r_squared']:+.3f}  "
          f"MAE={model['mae']:.1f}  median AE={model['median_ae']:.1f}")
print("excluded:", probe.value["excluded_features"])
```

```text
ridge            R²=+0.711  MAE=51.0  median AE=37.5
huber            R²=+0.559  MAE=39.0  median AE=24.4
median_baseline  R²=-0.064  MAE=113.6  median AE=77.4
excluded: ['account_id', 'renewal_invoice_total']
```

Read those two rows against each other. Ridge wins on R², which it must — Huber
optimizes a robust loss and therefore *cannot* win on squared error, so
comparing the two on R² would measure the metric, not the data. On the typical
row, though, Huber's error is **35% lower**. That gap is the finding: the data
is predictable and a minority of rows is dragging the least-squares fit. Adding
features will not fix it; the review rows will.

> Everything derived from a probe is **model-conditional**. A residual is not a
> property of your dataset — it is what one estimator left over on one split.
> The report says so on the page.

## Rows to review

```python
rows = next(e for e in result.evidence if e.kind == "regression_review_rows")
for row in rows.value["rows"]:
    print(f"row {row['row_index']}: actual={row['actual']:.0f} "
          f"predicted={row['predicted']:.0f} "
          f"Cook's D={row['cooks_distance']:.2f} — {', '.join(row['reasons'])}")
```

```text
row 123: actual=95 predicted=981 Cook's D=2.96 — large residual, high influence, extreme feature values
row 7: actual=140 predicted=592 Cook's D=1.09 — large residual, high influence, extreme feature values
row 142: actual=909 predicted=562 Cook's D=0.08 — large residual, high influence
row 126: actual=700 predicted=392 Cook's D=0.09 — large residual, high influence
```

Rows 7 and 123 are the two accounts with implausible seat counts. Rows 142 and
126 sit at the contract ceiling. A data-entry error and a genuinely unusual
customer look identical here and need opposite treatment, which is why these are
review candidates and not corrections.

**Why the list is short.** The familiar Cook's distance rule of thumb, `4/n`,
flags a few percent of rows in *any* dataset, clean ones included — it is a
screening convention, not evidence of a problem. Using it alone would put a
review list on every regression ever run. A row reaches this list only by
clearing a bar ordinary data does not: decisive influence, a residual far
outside the fit's own spread, or both signals together. On well-behaved data the
list does not exist at all.

## What stays quiet

Run the same recipe on a clean, well-specified regression and you should get
nothing:

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(3)
x1 = rng.normal(50, 12, 400)
x2 = rng.normal(20, 4, 400)
clean = pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 + 3 * x2 + rng.normal(0, 6, 400)})

result = pe.load({"t": clean}).regression("y")
print(result.summary)
```

```text
t.y looks ready: no blocking regression risks were found.
```

No issues, no alerts, no review rows — but the evidence is all still there
(`regression_probe` reports R² of 0.95) for anything downstream that wants it.
That silence is the point, and it takes deliberate work to achieve. Several
standard diagnostics report *something* on every dataset unless they are
explicitly guarded:

- Cook's `4/n` rule flags a few percent of rows in any fit, so review rows
  require decisive influence, not merely clearing the screen.
- Equal-width bins always leave thin bins in a bell curve's tails, so a "weak
  support" gap requires real data on *both* sides of it.
- Absolute error scales with target magnitude, so subgroup error is scaled and
  compared against sibling groups rather than the overall rate.

## Subgroup error is scaled before it is compared

A group whose target is ten times larger carries roughly ten times the absolute
error while being predicted exactly as well. Ranking subgroups on raw mean
absolute error therefore flags the highest-magnitude group every single run and
tells you nothing.

Prism divides each group's error by that group's own target spread, then
compares each level against the **median level of its own column** — not against
the overall rate. That second correction matters: whenever the grouping column
predicts the target, the spread within any one level is far smaller than the
spread across all of them, so every level would score worse than the whole and
the check would fire on all of them at once.

```python
concentration = next(
    e for e in result.evidence if e.kind == "regression_error_concentration"
)
for group in concentration.value["groups"][:3]:
    print(f"{group['column']}={group['level']:12s} "
          f"raw={group['raw_error_ratio']:.2f}  scaled={group['error_ratio']:.2f}")
```

A finding is raised only when the scaled ratio clears 1.5.

## What it checks

| Area | Checks |
|---|---|
| Target | Range, centre, spread, skew, kurtosis, zeros, negatives, quantiles |
| Target shape | Shape label plus `log1p` / `sqrt` / Yeo–Johnson candidates, each measured |
| Censoring | Repeated exact values in a continuous target — caps, floors, defaults, zero inflation |
| Heaping | Round-number preference, so reported precision is not mistaken for measured precision |
| Association | Pearson, Spearman, and a binned η² per feature, so a curve is not reported as "no relationship" |
| Redundancy | Interchangeable pairs, plus VIF and the design condition number |
| Leakage | Affine copies of the target, near-perfect univariate fit, shared name tokens |
| Probes | Cross-validated Ridge and Huber against a median baseline |
| Residuals | Shape, a normality *distance* (never a p-value), spread across the fitted range |
| Bias | Mean residual per fitted decile — over/under-prediction the average hides |
| Influence | Leverage, Cook's distance, and a ranked review list |
| Support | Thinly supported target ranges and feature gaps — where prediction is extrapolation |
| Splits | Group and time split guidance from `entity_id` and `timestamp` |

## Context changes the analysis

```python
result = pe.regression(
    "data/accounts.csv",
    target="revenue",
    context={"entity_id": "account_id", "timestamp": "observed_at", "groups": ("region",)},
    mode="deep",
)
```

- `entity_id` — repeated entities make a random split optimistic; Prism
  recommends `GroupKFold`.
- `timestamp` — measures how far the target's mean drifts across the history and
  recommends time-ordered validation.
- `groups` — adds those columns to the subgroup error breakdown.

## The report

```python
result.to_html("regression-readiness.html")
```

A single offline file. Sections, in the order the report argues them: findings,
alerts, **rows to review**, **residuals** (residual-vs-fitted with a spread
band, average error by predicted range, and the residual distribution),
**target shape** (as recorded, beside the best transform), then the reference
tables — probe scores, feature signal, and VIF.

In the residual plot, a handful of enormous residuals is exactly what the chart
exists to show, so the axis is set from a robust range and the extremes are
pinned to the edge with a ring and a count rather than being allowed to compress
everything else into an unreadable strip — or being dropped.

## Limits worth knowing

- The probes are diagnostic fits. They are deliberately simple and linear; a
  weak probe means *a linear model finds little here*, not that the data is
  unlearnable.
- Leverage and Cook's distance come from an OLS fit on the screened features and
  inherit its assumptions. The design is capped at 30 features.
- Heteroscedasticity is reported as an effect size (a spread ratio) with the
  Breusch–Pagan statistic alongside; no p-value drives a finding.
- Censoring is inferred from repeated values. A genuinely popular price point
  looks identical to a cap — confirm against how the column is recorded.
- Rows with a missing target are excluded from the probes and counted as an
  issue, never imputed.

## See also

- [Classification readiness](classification.md) — the same shape of check for a label
- [Results & evidence](results-and-evidence.md) — the `AnalysisResult` contract
- [Context & configuration](context-and-config.md) — steering the analysis
