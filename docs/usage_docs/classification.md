# Classification readiness

`classification("target")` answers a question you should ask *before* training a
model: **is this data actually ready to learn from?** It runs a battery of
deterministic diagnostics — leakage screening, class balance, feature/target
association, missingness, and a leakage-screened probe model — and leads with a
**readiness verdict**.

It is a *readiness diagnostic*, not a training pipeline. It returns evidence and
findings, never a fitted production model.

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())
result = dataset.classification("churned", table="customers")
```

As a one-liner:

```python
result = pe.classification("data/training.csv", target="label")
```

> Pass `table=` when the dataset has more than one table. The target is the name
> of the label column within that table.

## The verdict and findings

```python
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title} (confidence={finding.confidence})")
    print(f"    {finding.summary}")
```

```text
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
[critical] Potential target leakage: exit_survey_sent (confidence=0.92)
    exit_survey_sent has deterministic target-signal risk against churned.
[high] Identifier-like feature: customer_id (confidence=0.9)
    customer_id is unique on 100% of rows and likely labels records rather than explaining the target.
[medium] Weak classification separability in customers (confidence=0.74)
    The leakage-screened probe reached 51.9% balanced accuracy, only 1.9% above the majority baseline.
[medium] Probe hard examples in customers (confidence=0.68)
    20 cross-validated probe error row(s) were retained for review.
```

This is the whole product thesis in one report. Read it top to bottom:

1. **`exit_survey_sent` is a target leak (`critical`).** In the sample data this
   column is literally derived from `churned`. A model trained on it would score
   near-perfectly in validation and fail in production. Leakage surfaces *first*
   because it invalidates everything downstream.
2. **`customer_id` is identifier-like (`high`).** It's unique per row, so it
   memorizes records rather than explaining the target — flagged for exclusion,
   not treated as a generic high-cardinality warning.
3. **Separability is weak (`medium`).** Here's the payoff: the probe model is run
   **after screening out the leaky column**, so its honest balanced accuracy is
   only 1.9% above baseline. That's the truth you want *before* you spend a week
   modelling — the real signal in these features is thin.
4. **Hard examples** are the rows the probe got wrong, retained for you to inspect
   for label noise or class overlap.

## What classification checks

| Check | What it tells you |
|-------|-------------------|
| **Target validity & class balance** | Class counts, entropy, majority/minority rates, imbalance ratio |
| **Conflicting labels** | Duplicate feature signatures assigned different labels |
| **Deterministic leakage** | Exact target copies, target-name overlap, near-perfect value rules (escalated to `critical`) |
| **Feature/target association** | Eta-squared (numeric) and Cramér's V (categorical) strength |
| **Class-conditional missingness** | Whether missingness differs across classes (possibly predictive) |
| **Identifier-like features** | Columns that label rows rather than explain the target |
| **High-cardinality risk** | Categorical/text features with too many distinct values |
| **Leakage-screened probe** | Cross-validated separability of a logistic probe with fold-local preprocessing |
| **Hard examples** | Cross-validated probe errors retained for review |

The leakage screen is reachable even on **imbalanced** targets (a near-perfect
rule escalates to `critical`), and the probe's preprocessing is fit **per fold**
so its separability estimate isn't itself inflated by leakage.

## Options

```python
result = dataset.classification(
    "churned",
    table="customers",
    max_categories=50,     # cap distinct categories considered per feature
    mode="standard",       # quick | standard | deep
)
```

- **`target`** — the label column (first positional argument).
- **`table`** — which table holds the target (required with multiple tables).
- **`max_categories`** — categorical features with more distinct values than this
  are treated as high-cardinality rather than fully enumerated. Default `50`.
- **`mode`** — compute depth (see [Context & configuration](context-and-config.md)).

You can also supply the target via [`AnalysisContext`](context-and-config.md)
instead of the positional argument.

## Artifacts and the transformation plan

Classification produces two `metric_table` artifacts — **Class balance** and
**Feature-target diagnostic signals** — rendered in the HTML report. Its
transformation plan contains non-mutating recommendations, e.g. *exclude the
identifier column* and *review the leakage candidate*, each citing its evidence:

```python
for step in result.transformation_plan.steps:
    print(step.operation, step.columns, "| risk:", step.risk)
```

```text
exclude_identifier_feature ('customer_id',) | risk: high
review_target_leakage_candidate ('exit_survey_sent', 'churned') | risk: high
```

## What's next for this recipe

Class-overlap/neighborhood-disagreement detection, group/time split guidance,
opt-in fairness coverage, and train/test comparison are planned. See the
[implementation status](../implementation-status.md).
