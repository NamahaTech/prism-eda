# Results & evidence

Every recipe returns the same typed, JSON-serializable object: an
**`AnalysisResult`**. This page is the reference for its structure and for the
finding/evidence/artifact models inside it.

```python
import prism_eda as pe
from examples.sample_data import load_sample

result = pe.load(load_sample()).classification("churned", table="customers")
```

## The `AnalysisResult` object

| Attribute | Type | What it holds |
|-----------|------|---------------|
| `goal` | `str` | Which recipe ran (`"profile"`, `"classification"`, …) |
| `status` | `AnalysisStatus` | Outcome — see [statuses](#statuses) below |
| `summary` | `str` | One-line, decision-first headline |
| `catalog` | `DatasetCatalog` | Descriptive metadata for every table/column |
| `findings` | `tuple[Finding, …]` | Prioritized, human-facing conclusions |
| `evidence` | `tuple[Evidence, …]` | Every reproducible numeric/structural claim |
| `artifacts` | `tuple[Artifact, …]` | Structured tables/graphs for the report |
| `assumptions` | `tuple[str, …]` | Assumptions carried from your context |
| `warnings` | `tuple[AnalysisWarning, …]` | Non-fatal caveats |
| `failures` | `tuple[AnalysisFailure, …]` | Optional metrics that failed but didn't stop the run |
| `sampling` | `tuple[SamplingRecord, …]` | What was sampled, how, and why |
| `transformation_plan` | `TransformationPlan` | Non-mutating data-prep recommendations |
| `metadata` | `dict` | Run metadata (mode, sampling, seed) |

```python
print(result.goal, "|", result.status)
print(result.summary)
print("metadata:", result.metadata)
```

```text
classification | completed
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
metadata: {'mode': 'standard', 'sampling': 'auto', 'random_seed': 42}
```

## Statuses

`result.status` is an `AnalysisStatus` (a string enum), so you can compare it to a
plain string or to `pe.AnalysisStatus.*`:

| Status | Meaning |
|--------|---------|
| `completed` | Required stages produced evidence without notable caveats |
| `completed_with_warnings` | Useful result, but with sampling or recoverable caveats |
| `insufficient_evidence` | The data can't support the requested analysis |
| `no_meaningful_structure` | The analysis ran but found no signal above its thresholds |
| `failed` | A foundational stage failed |

```python
if result.status == pe.AnalysisStatus.COMPLETED:
    ...
# StrEnum: this also works
if result.status == "completed":
    ...
```

A low-evidence dataset yields `insufficient_evidence` or
`no_meaningful_structure` rather than a forced conclusion — Prism would rather
say "not enough signal" than make something up.

## Findings

A `Finding` is a human-facing conclusion. **Severity** (attention) and
**confidence** (certainty) are deliberately separate fields, and every finding
cites the evidence it rests on.

```python
top = result.findings[0]
print("id:        ", top.id)
print("title:     ", top.title)
print("severity:  ", top.severity)       # critical | high | medium | low | info
print("confidence:", top.confidence)
print("evidence:  ", top.evidence_ids)
print("recommend: ", top.recommendation)
```

```text
id:         finding_48e5033c56d58367
title:      Potential target leakage: exit_survey_sent
severity:   critical
confidence: 0.92
evidence:   ('ev_0b5b3e6d53fc6712',)
recommend:  Confirm the feature is available before prediction time and is not derived from the label.
```

Findings arrive **already sorted** — most-severe first, then by descending
confidence — so `result.findings[0]` is always the thing most worth your
attention.

## Evidence

Evidence is the reproducible basis for findings. Each piece records its scope,
the method that produced it, its value, confidence, and assumptions. Its `id` is
a hash of that content, so the same deterministic computation always yields the
same ID — re-run on unchanged data and the IDs match.

```python
# Follow a finding to the evidence it cites.
ev = next(e for e in result.evidence if e.id in top.evidence_ids)
print("id:         ", ev.id)
print("kind:       ", ev.kind)
print("method:     ", ev.method)
print("scope:      ", ev.scope)
print("confidence: ", ev.confidence)
print("value:      ", ev.value)
```

```text
id:         ev_0b5b3e6d53fc6712
kind:       classification_leakage_candidate
method:     deterministic_leakage_screen_v2
scope:      EvidenceScope(table='customers', columns=('exit_survey_sent', 'churned'))
confidence: 0.92
value:      {'feature': 'exit_survey_sent', 'target': 'churned', 'exact_target_copy': True, 'name_contains_target': False, 'value_rule_accuracy': 1.0, 'majority_baseline': 0.7375, 'near_perfect': True}
```

The `method` string is **versioned** (`..._v2`): when an algorithm changes
materially its version bumps, so old and new evidence never look equivalent.

## Artifacts

An `Artifact` is structured data for the report — a metric table, or the schema
ER graph. Renderers consume artifact `data`; recipes never emit raw HTML.

```python
for a in result.artifacts:
    print(a.kind, "|", a.title, "| data keys:", sorted(a.data))
```

```text
metric_table | Class balance | data keys: ['columns', 'rows']
metric_table | Feature-target diagnostic signals | data keys: ['columns', 'rows']
```

## Transformation plans

Analysis may *recommend* changes but never applies them. `transformation_plan` is
an ordered list of declarative `TransformationStep`s, each citing the evidence
that motivated it:

```python
for step in result.transformation_plan.steps:
    print(step.operation, step.columns)
    print("   rationale:        ", step.rationale)
    print("   risk:             ", step.risk)
    print("   requires_approval:", step.requires_approval)
    print("   evidence_ids:     ", step.evidence_ids)
```

```text
exclude_identifier_feature ('customer_id',)
   rationale:         Identifier-like columns leak row identity into the model.
   risk:              high
   requires_approval: True
   evidence_ids:      ('ev_e8b87a561ecfdf7f',)
review_target_leakage_candidate ('exit_survey_sent', 'churned')
   rationale:         Leaky fields can make validation scores unrealistic.
   risk:              high
   requires_approval: True
   evidence_ids:      ('ev_0b5b3e6d53fc6712',)
```

Applying plans and generating pandas code are out of scope for 0.1 —
the plan is a reviewed to-do list, not an autopilot. Check
`result.transformation_plan.is_empty` before iterating.

## The catalog

`result.catalog` (also `dataset.catalog()`) is the descriptive foundation every
recipe shares. It is a `DatasetCatalog` of `TableCatalog`s of `ColumnCatalog`s:

```python
cat = result.catalog
print(cat.table_count, cat.row_count, cat.column_count)
print(cat.fingerprint_method)

tbl = cat.table("customers")        # KeyError if the name is unknown
col = next(c for c in tbl.columns if c.name == "signup_age")
print(col.physical_type, col.semantic_type, col.roles)
print(col.missing_count, col.missing_rate, col.unique_count)
```

```text
2 320 11
schema-shape-and-deterministic-row-sample-v1
float64 numeric ('measure_candidate',)
20 0.25 32
```

Key fields: `ColumnCatalog` carries `physical_type`, `semantic_type`, `roles`,
missingness, distinctness, `statistics`, and `top_values`; `TableCatalog` adds
shape, `memory_bytes`, `duplicate_row_count`, and a `fingerprint`. The
fingerprint detects likely data changes; it is a bounded-cost signal, not a
cryptographic commitment to every row.

## Warnings, failures, and sampling

- `result.warnings` — `AnalysisWarning(code, message, table, column)` for
  non-fatal caveats (e.g. `insufficient_rows`).
- `result.failures` — `AnalysisFailure(stage, message, recoverable, table,
  column)` for optional metrics that failed while the run continued.
- `result.sampling` — `SamplingRecord`s describing any sampling
  (source/sampled rows, strategy, seed, reason, limitations, whether you
  overrode it). Reports prominently warn when decision-relevant evidence was
  sampled.

## Serializing the whole thing

```python
payload = result.to_dict()          # plain JSON-able dict
payload = result.model_dump(mode="json")   # same thing, pydantic-style alias
print(sorted(payload))
```

```text
['artifacts', 'assumptions', 'catalog', 'evidence', 'failures', 'findings', 'goal', 'metadata', 'sampling', 'status', 'summary', 'transformation_plan', 'warnings']
```

To write reports and JSON to disk, see [Exporting reports](exporting-reports.md).
