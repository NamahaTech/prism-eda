# Getting started

This page takes you from zero to a real analysis in a few minutes, then explains
the mental model so the rest of the docs click into place.

## 1. Load some data

Prism analyses run on a **`Dataset`** — a named collection of one or more tables.
You create one with `pe.load(...)`, which accepts a DataFrame, a file, a folder,
or a mapping of named tables. We'll use the seeded sample shipped with the repo:

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())   # {"customers": ..., "orders": ...}
print(list(dataset.tables))
```

```text
['customers', 'orders']
```

See [Loading data](loading-data.md) for every accepted input type.

## 2. Run a baseline profile

```python
result = dataset.profile()
print(result.status)
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title} — {finding.summary}")
```

```text
completed
Profiled 2 table(s), 320 rows, and 11 columns; found 1 prioritized issue(s).
[medium] High missingness in customers.signup_age — 20 values (25.0%) are missing.
```

Notice what *didn't* happen: Prism profiled 320 rows across 11 columns but
surfaced a single prioritized finding. The full per-column detail still exists as
evidence (`result.evidence`) and in the rendered report — but Prism only
*promotes* something to a finding when it crosses a meaningful threshold. That's
the "signal over noise" idea in action.

## 3. Ask a task-specific question

The real power shows up when you tell Prism what you're trying to do. Here we ask
whether the `customers` table is ready to train a churn classifier:

```python
result = dataset.classification("churned", table="customers")
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
```

```text
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
[critical] Potential target leakage: exit_survey_sent
[high] Identifier-like feature: customer_id
[medium] Weak classification separability in customers
[medium] Probe hard examples in customers
```

The summary leads with a **verdict** ("not ready to model") and the single most
important reason. `exit_survey_sent` is flagged `critical` because it's a perfect
copy of the label — a classic leak that would make any model look brilliant in
validation and useless in production. See [Classification](classification.md) for
the full breakdown.

## 4. Produce a shareable report

Nothing is written to disk until you ask. Export a self-contained HTML report (no
internet required; interactive extras degrade gracefully without JavaScript) or
machine-readable JSON:

```python
result.to_html("churn-readiness.html")
result.to_json("churn-readiness.json")
```

More in [Exporting reports](exporting-reports.md).

## The mental model

Three ideas explain almost everything in Prism:

```text
        deterministic computation
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Evidence   (every number,    │   ← reproducible, stable IDs,
   │             reproducible)     │     records method + assumptions
   └──────────────┬───────────────┘
                  │  promoted only above a threshold
                  ▼
   ┌──────────────────────────────┐
   │  Findings   (decision-first,  │   ← ordered by severity, each cites
   │             human-facing)     │     the evidence it rests on
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  AnalysisResult               │   ← status, summary, findings,
   │  (the thing you export)       │     evidence, artifacts, plan…
   └──────────────────────────────┘
```

1. **Evidence comes first.** Every numeric or structural claim is computed
   deterministically and stored as `Evidence` with a stable ID, the method that
   produced it, and its assumptions. Re-running on unchanged data yields
   identical evidence IDs.

2. **Findings are promoted, not dumped.** A `Finding` is a human-facing
   conclusion that only appears when evidence crosses a real threshold. Findings
   carry a **severity** and a separate **confidence**, cite the evidence they
   rest on, and are ordered most-severe-first.

3. **Candidates, not verdicts.** Inferred keys, relationships, and anomalies are
   always *candidates* for you to confirm — Prism never silently asserts a
   foreign key or labels a row as a confirmed anomaly.

Everything you get back is wrapped in an [`AnalysisResult`](results-and-evidence.md),
which is fully typed and JSON-serializable.

## The four recipes

| Recipe | Question it answers | Guide |
|--------|---------------------|-------|
| `profile()` | What's the shape and quality of this data? | [Profile](profile.md) |
| `discover_schema()` | How do these tables relate? What are the keys? | [Schema discovery](schema-discovery.md) |
| `anomaly_detection()` | Which rows deserve a closer look? | [Anomaly detection](anomaly-detection.md) |
| `classification("target")` | Is this data ready to train a classifier? | [Classification](classification.md) |

Each is available both as a `Dataset` method (`dataset.profile()`) and as a
top-level convenience function (`pe.profile(source)`) that loads and analyses in
one call. They return the same result type — see
[Context & configuration](context-and-config.md) for how to steer them.
