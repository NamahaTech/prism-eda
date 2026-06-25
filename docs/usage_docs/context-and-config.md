# Context & configuration

By default the recipes do something sensible with no configuration. When you want
to steer them — tell Prism what a column *means*, control compute depth, or fix a
seed — there are two objects: **`AnalysisContext`** (what the data is) and
**`AnalysisConfig`** (how to run).

```python
import prism_eda as pe
```

## `AnalysisContext` — describe the task

Context is domain and task information that changes how evidence is *interpreted*.
Supplying it helps Prism pick the right evidence and write better findings.

```python
context = pe.AnalysisContext(
    goal="anomaly_detection",
    target="churned",                # the label / column of interest
    entity_id="customer_id",         # the column identifying an entity
    timestamp="signed_up_at",        # the time column, if any
    groups=("region",),              # grouping columns
    domain_notes="Monthly subscription customers; churn = cancelled in window.",
    assumptions=("One row per customer.",),
)
```

| Field | Purpose |
|-------|---------|
| `goal` | Which analysis the context is for |
| `target` | Label / column of interest (an alternative to the `target=` argument) |
| `entity_id` | Column that identifies an entity |
| `timestamp` | Time column, if any |
| `groups` | Grouping columns |
| `domain_notes` | Free text describing the data's meaning |
| `assumptions` | Assumptions you're making; carried through to `result.assumptions` |
| `metadata` | Arbitrary extra key/values |

You can pass context as an object or as a plain dict, and it composes with the
typed recipe arguments:

```python
dataset = pe.load({"customers": ...})

# As an object:
result = dataset.anomaly_detection(context=context)

# As a dict (merged into a context with the recipe's goal):
result = dataset.classification("churned", context={"entity_id": "customer_id"})
```

When both are given, an explicit argument (e.g. `target=` on `classification`)
takes precedence over `context.target`.

## `AnalysisConfig` — control the run

Config controls reproducibility and compute, independent of what the data means.

```python
config = pe.AnalysisConfig(
    mode="standard",                  # quick | standard | deep
    sampling="auto",                  # auto | disabled
    random_seed=42,
    allow_insufficient_evidence=False,
)
result = dataset.profile(config=config)
```

Most of the time you don't build a `AnalysisConfig` directly — every recipe
accepts these as keyword arguments and assembles the config for you:

```python
result = dataset.classification("churned", mode="deep", random_seed=7)
```

### `mode` — compute depth

`AnalysisMode` is `quick`, `standard` (default), or `deep`. Higher modes spend
more compute (wider key search in schema discovery, larger sampling budgets,
etc.). Mode describes **compute depth, not report verbosity** — a deeper run
doesn't dump more noise, it looks harder.

### `sampling` — `"auto"` or `"disabled"`

In `auto` mode, expensive operations (associations, probe models, local-neighbor
methods) may run on deterministic samples; cheap exact metrics always use all
rows. Every sampled result is recorded in `result.sampling`, and reports warn
prominently when decision-relevant evidence was sampled.

Set `sampling="disabled"` to forbid automatic sampling. Note that this is **not**
a promise of unlimited memory — an algorithm may still refuse an unsafe or
inapplicable execution rather than sampling.

### `random_seed`

Stable by default (`42`), so runs are reproducible. Must be non-negative.

### `allow_insufficient_evidence`

By default (`False`), when evidence is too thin to support a defensible
conclusion, Prism withholds it and returns `insufficient_evidence` /
`no_meaningful_structure`. Set it to `True` to get a **best-effort** result
instead — which stays visibly qualified as `completed_with_warnings`:

```python
import pandas as pd

strict = pe.profile(pd.DataFrame(columns=["id"]))
print(strict.status)                                  # insufficient_evidence

best_effort = pe.profile(pd.DataFrame(columns=["id"]), allow_insufficient_evidence=True)
print(best_effort.status)                             # completed_with_warnings
```

## The generic `analyze()` entry point

Each typed method (`profile`, `discover_schema`, `anomaly_detection`,
`classification`) is a thin wrapper over `dataset.analyze(goal=...)`. You can call
`analyze` directly with a goal string when that's more convenient — for example
when the goal is dynamic:

```python
result = dataset.analyze(
    "anomaly_detection",
    context={"entity_id": "customer_id"},
    mode="standard",
    table="orders",
)
```

Recognized goals (and their aliases):

| Goal | Aliases |
|------|---------|
| `profile` | `minimal_eda` |
| `schema_discovery` | `discover_schema` |
| `anomaly_detection` | `anomaly`, `outlier_detection` |
| `classification` | `classify` |

Unknown goals raise `NotImplementedError`; passing options a goal doesn't accept
raises `TypeError`. The typed methods are recommended for everyday use — they give
you autocompletion and validate their arguments.

## Convenience functions vs. the session API

The top-level functions (`pe.profile`, `pe.anomaly_detection`, …) accept all the
same context/config arguments **plus** the loading arguments from
[Loading data](loading-data.md), so you can load and analyse in one call:

```python
result = pe.classification(
    "data/training.csv",
    target="label",
    mode="deep",
    read_options={"sep": ";"},
)
```

They simply call `pe.load(...)` and the matching `Dataset` method — same code
path, same result type. Use the session API (`pe.load(...)` then
`dataset.<recipe>()`) when you want to run **several** analyses on the same loaded
data without reloading it.
