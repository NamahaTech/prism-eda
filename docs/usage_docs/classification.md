# Classification readiness

`classification("target")` answers a question you should ask *before* training a
model: **is this data actually ready to learn from?** It runs a battery of
deterministic diagnostics — leakage screening, class balance, feature/target
association, local class overlap, missingness, and a leakage-screened probe
model — and leads with a **readiness verdict**.

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
| **Feature importance** | A random forest ranked by held-out permutation *and* impurity, against a measured sentinel noise floor |
| **Local class overlap** | Rows whose nearest eligible-feature neighbors commonly have a different label |
| **Hard examples** | Cross-validated probe errors retained for review |
| **Split guidance** | Group/time-aware validation advice when `entity_id` or `timestamp` is supplied in the context |

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

## What drives the label

Alongside the logistic probe, Prism fits a **random forest** on the same
screened features and ranks them two ways: held-out permutation importance, and
the forest's own in-sample impurity importance. The mechanics are identical to
the regression recipe's, and the [regression guide](regression.md#what-drives-the-target)
walks through them in full — impurity's cardinality bias, the sentinel noise
floor, and the correlated-pair guard on the drop list all apply here unchanged.

The sample `customers` table is a good demonstration of what the stage does
*not* do:

```python
result = pe.load(load_sample()).classification("churned", table="customers")
print([e.kind for e in result.evidence if e.kind == "feature_importance"])
print([w.code for w in result.warnings])
```

```text
[]
['feature_importance_no_signal']
```

Eighty rows leave a twenty-row hold-out, and on it the forest does not beat
predicting the majority class. So there is no ranking, no section in the report,
and a warning saying why. A ranking read off a model that cannot predict is an
ordering of noise, and printing one would be worse than printing nothing.

On a table where the forest *can* predict, the ranking appears:

```python
from examples.sample_data import feature_signal

frame = feature_signal()
frame = frame.assign(renewed=frame["renewal_value"] > frame["renewal_value"].median())
frame = frame.drop(columns=["renewal_value"])

result = pe.load({"renewals": frame}).classification("renewed")
importance = next(e for e in result.evidence if e.kind == "feature_importance")

print(f"forest={importance.value['model_score']:.3f}  "
      f"logistic={importance.value['linear_score']:.3f}  "
      f"majority={importance.value['baseline_score']:.3f}")
for row in importance.value["features"]:
    mark = "  (below noise floor)" if row["below_noise_floor"] else ""
    print(f"  {row['feature']:<15} permutation={row['permutation_importance']:>8.4f}"
          f"  impurity #{row['impurity_rank']}{mark}")
```

```text
forest=0.973  logistic=0.573  majority=0.500
  tenure_months   permutation=  0.4187  impurity #1
  plan            permutation=  0.1556  impurity #2
  ticket_ref      permutation=  0.0000  impurity #3  (below noise floor)
  sensor_drift    permutation=  0.0000  impurity #4  (below noise floor)
  seats_billed    permutation=  0.0000  impurity #5  (below noise floor)
  seats           permutation=  0.0000  impurity #6  (below noise floor)
  inbound_calls   permutation=  0.0000  impurity #7  (below noise floor)
  survey_score    permutation=  0.0000  impurity #8  (below noise floor)
```

Everything is measured as **balanced accuracy**, matching the probe's headline
metric, and against a majority-class baseline rather than zero — balanced
accuracy floors at `1/classes`, not at nothing. The forest reaches 0.97 where
the logistic probe reaches 0.57 on the same split, which is the
`A tree finds signal the linear probe missed` alert: this label is decided by a
threshold and an interaction, so a linear model will under-read it.

## Artifacts and the transformation plan

Classification produces two `metric_table` artifacts — **Class balance** and
**Feature-target diagnostic signals** — rendered in the HTML report, plus the
**What drives the label** section when the forest earns one. Local
overlap uses leakage-screened eligible features, median imputation/scaling for
numeric features, one-hot encoding for categorical features, and a deterministic
5- or 9-neighbor comparison (quick vs. standard/deep). A row is retained when at
least half of its neighbors have another label. This is a review signal for label
ambiguity, overlapping classes, or missing features; it is not proof that a
label is wrong or a production-model score. Its
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

Opt-in fairness coverage and train/test comparison are planned. See the
[implementation status](../implementation-status.md).
