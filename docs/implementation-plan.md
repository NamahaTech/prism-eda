# Prism EDA Implementation Plan and Handoff

Last updated: 2026-06-20

This document is the detailed implementation handoff for Prism EDA. It explains
what has already been built, how the pieces fit together, what is intentionally
limited, and what should come next. Keep this file current whenever a feature is
added, removed, renamed, or materially re-scoped.

## Product Direction

Prism EDA is a Python library for task-aware exploratory data analysis. It is
not intended to become a dashboard-first product. The primary user imports the
library in Python and runs deterministic or AI-assisted analyses from notebooks,
scripts, batch jobs, or future integrations.

Core goals:

- provide more useful EDA than generic profiling by making reports task-aware;
- keep every numeric claim tied to machine-readable evidence;
- support large real datasets with deterministic sampling and explicit warnings;
- generate beautiful self-contained HTML without heavy frontend dependencies;
- recommend transformations but never mutate user data silently;
- prepare the architecture for Gemini, local models, and other providers without
  making AI mandatory.

The library currently targets Python 3.11+.

## Current Package Shape

Distribution name:

```text
prism-eda
```

Import package:

```python
import prism_eda as pe
```

Main source tree:

```text
src/prism_eda/
  api.py                       Top-level one-shot convenience functions
  dataset.py                   Session object and goal dispatcher
  config.py                    AnalysisContext and AnalysisConfig
  results.py                   AnalysisResult, warnings, failures, sampling
  evidence/models.py           Evidence and Finding contracts
  artifacts.py                 Structured report artifacts
  catalog/
    loaders.py                 DataFrame, CSV, Parquet, mapping, directory loading
    profiling.py               Table/column catalog generation
    relationships.py           Candidate key and relationship engine
  analysis/
    profile.py                 Baseline deterministic profile
    schema_discovery.py        Candidate schema and ER artifact recipe
    anomaly.py                 Deterministic anomaly-detection diagnostics
    classification.py          Deterministic classification diagnostics
  transformations/
    models.py                  Non-mutating transformation plan models
  reporting/
    renderer.py                Jinja environment and HTML renderer
    templates/report.html      Self-contained report template
```

## Public API Implemented

### Loading

```python
dataset = pe.load(source)
```

Supported sources:

- a pandas `DataFrame`;
- one CSV or Parquet file path;
- a sequence of CSV or Parquet file paths;
- a mapping of table name to `DataFrame` or file path;
- a directory path containing CSV and Parquet files.

Directory options:

```python
dataset = pe.load(
    "data/",
    recursive=True,
    include=["*.csv", "*.parquet"],
    exclude=["*_backup.csv", "archive/*"],
    names={"customers_2024": "customers"},
    read_options={"csv": {"low_memory": False}},
)
```

Important behavior:

- directory loading uses filename stems as table names by default;
- input DataFrames are not mutated;
- no report file is written until `to_html(...)` or `to_json(...)` is called.

### Baseline Profile

```python
result = pe.profile("data/customers.parquet")
result = dataset.profile()
result = dataset.analyze("profile")
```

Implemented diagnostics:

- dataset, table, and column shape;
- duplicate rows;
- missingness;
- distinctness;
- constants;
- top values;
- numeric range and shape summaries;
- text length summaries;
- initial semantic type and role candidates;
- evidence-linked findings;
- non-mutating transformation recommendations.

### Schema Discovery

```python
result = pe.discover_schema("data/", recursive=True)
result = dataset.discover_schema(mode="standard")
result = dataset.analyze("schema_discovery")
```

Implemented diagnostics:

- candidate primary keys;
- composite candidate keys up to mode-specific width;
- typed and name-aware inclusion dependency search;
- candidate one-to-one and one-to-many relationships;
- orphan child rows and unreferenced parent values;
- deterministic row sampling for expensive checks;
- confidence scores and assumptions;
- layered ER diagram artifact with entity cards, PK/FK roles, routed connectors,
  confidence badges, and one/many cardinality marks.

Important limitation:

The recipe reports candidates, not database constraints. Downstream users must
confirm business meaning and expected cardinality.

### Anomaly Detection

```python
result = pe.anomaly_detection(df)
result = dataset.anomaly_detection(table="events")
result = dataset.analyze("anomaly_detection", table="events")
```

Optional labeled anomaly target:

```python
result = pe.anomaly_detection(df, target="is_anomaly")
```

Implemented diagnostics:

- optional rare-label summary when a target is supplied;
- univariate numeric tail candidates using IQR and modified z-score style robust
  scaling;
- multivariate numeric candidates using robust-scaled Euclidean scores;
- conditional numeric candidates, such as a value being unusual within local
  bands of another feature;
- rare categorical values;
- ranked metric-table artifact summarizing anomaly candidate signals;
- evidence-linked findings and review recommendations.

Important limitations:

- this is deterministic diagnostic EDA, not a fitted anomaly detector;
- candidates are not confirmed anomalies;
- current multivariate scoring does not model covariance;
- no Isolation Forest, Local Outlier Factor, detector agreement, or stability
  analysis is implemented yet;
- conditional detection currently uses quantile bins and may miss sparse,
  nonlinear, or high-cardinality contexts.

### Classification

```python
result = pe.classification(df, target="label")
result = dataset.classification("label", table="train")
result = dataset.analyze("classification", target="label")
```

Implemented diagnostics:

- target validity;
- class counts, majority rate, minority count, imbalance ratio, and entropy;
- missing target labels;
- duplicate feature signatures with conflicting labels;
- numeric feature-target association via eta-squared;
- categorical feature-target association via Cramer's V;
- class-conditional missingness;
- high-cardinality feature risk;
- deterministic target-leakage candidates from exact copies, name overlap, and
  highly predictive value rules;
- class-balance and feature-signal metric-table artifacts;
- evidence-linked findings and transformation recommendations.

Important limitations:

- no model training is performed yet;
- no cross-validation, probe model, calibration, or hard-example detection yet;
- no train/test comparison yet;
- no group/time split recommendation yet beyond existing context fields;
- fairness or subgroup coverage is not implemented.

## Result and Evidence Contract

Every recipe returns `AnalysisResult`.

Key fields:

```python
result.goal
result.status
result.summary
result.catalog
result.findings
result.evidence
result.artifacts
result.assumptions
result.warnings
result.failures
result.sampling
result.transformation_plan
result.metadata
```

Invariants:

- findings must cite evidence IDs;
- assumptions should be explicit in evidence or result-level assumptions;
- severity and confidence are separate ideas;
- deterministic sampling must produce warnings and `SamplingRecord` entries;
- transformation plans are recommendations only;
- exports must be JSON-serializable through `to_jsonable`;
- report rendering must not require JavaScript, external CDNs, or Plotly in core.

Status meanings:

- `completed`: the recipe ran and produced sufficient evidence.
- `completed_with_warnings`: the recipe ran, but sampling or recoverable caveats
  apply.
- `insufficient_evidence`: foundational evidence is missing, such as no target
  column for classification.
- `no_meaningful_structure`: the recipe ran but found no meaningful candidate
  structure.
- `failed`: reserved for non-recoverable recipe failures.

## Report Rendering

The current renderer uses one Jinja template:

```text
src/prism_eda/reporting/templates/report.html
```

It supports:

- shared hero, metrics, warnings, findings, per-table catalog, transformation
  plan, and reproducibility sections;
- schema-specific ER diagram rendering;
- generic `metric_table` artifacts used by anomaly detection and classification.

Known report debt:

- as more recipes are added, the single template should be split into
  recipe-specific partials;
- static visual artifacts should gain snapshot or browser visual tests;
- rich charts should remain optional through `plotly` extras, not core.

## Current Test Coverage

Current suite categories:

- loader tests;
- baseline profile tests;
- export tests;
- schema discovery tests;
- anomaly and classification goal recipe tests.

Expected verification commands:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src/prism_eda
.venv/bin/pytest --cov=prism_eda --cov-report=term-missing
.venv/bin/python -m build --no-isolation
```

Before handoff, also install the built wheel into a temporary target and import
from that target to verify packaged templates and package data.

## Implementation Principles

Use these rules when extending the project:

1. Keep recipes deterministic unless explicitly building AI-assisted analysis.
2. Prefer evidence and findings over ad hoc strings.
3. Never mutate user data as part of analysis.
4. Use pandas and numpy in core; keep heavier or interactive tools optional.
5. Make sampling visible, deterministic, and reproducible.
6. Treat candidates as candidates. Avoid asserting business truth.
7. Keep user-facing docs updated in the same change as code.
8. Keep public API examples simple and import-first.
9. Add tests around behavior, not implementation details alone.
10. When confidence is low, return insufficient evidence or caveats instead of
    overclaiming.

## Roadmap

### Milestone 0.1: Deterministic Foundation

Status: mostly implemented.

Completed:

- package scaffold and build configuration;
- loading from DataFrame, files, mappings, and folders;
- baseline profile;
- schema discovery;
- static HTML and JSON export;
- callback/event contracts;
- anomaly-detection MVP;
- classification MVP;
- detailed docs and maintainer handoff material.

Remaining before a first public alpha:

- polish README examples for all currently implemented recipes;
- add API reference pages or docstrings rendered by a documentation tool;
- add more edge-case tests for mixed dtypes and all-null columns;
- add wheel-install smoke test to CI;
- confirm package metadata and repository links after repo rename.

### Milestone 0.2: Stronger Deterministic Recipes

Recommended order:

1. Improve anomaly detection:
   - Isolation Forest as an optional or core diagnostic if scikit-learn remains
     core;
   - Local Outlier Factor when row count and dimensionality are suitable;
   - detector agreement and disagreement;
   - score stability across seeds/subsamples;
   - per-row explanation tables;
   - expected contamination parameter.

2. Improve classification:
   - diagnostic probe model with cross-validation;
   - leakage-safe preprocessing inside the probe;
   - separability metrics;
   - hard-example and neighborhood-disagreement candidates;
   - train/test comparison if the user provides separate tables;
   - group/time split guidance using `AnalysisContext.entity_id` and
     `AnalysisContext.timestamp`.

3. Add regression recipe:
   - target shape, skew, zeros, censoring, heaping, and tail risk;
   - numeric/categorical target associations;
   - leakage candidates;
   - baseline probe residual diagnostics;
   - heteroscedasticity and subgroup error concentration.

4. Add report partials:
   - `profile.html`;
   - `schema.html`;
   - `anomaly.html`;
   - `classification.html`;
   - shared macros for metric tables, evidence details, badges, and cards.

### Milestone 0.3: AI-Assisted Investigation

Planned architecture:

- `assisted_analysis/` package with provider adapters;
- Gemini first, while keeping provider interfaces open for OpenAI and local
  models;
- LangGraph flow with intake, planning, tool execution, critique, clarification,
  and report synthesis nodes;
- AI tools that query compact deterministic summaries instead of raw full data;
- privacy-aware prompts with explicit disclosure of what is sent to providers;
- assumptions listed in final reports;
- "insufficient evidence" as a first-class response when the model cannot justify
  a claim.

Important AI constraints:

- never send raw data unless the user explicitly chooses that mode;
- column names may be sent by default according to current product direction;
- consider keyed hashing or HMAC aliases for values and entity identifiers;
- never store API keys in reports, logs, artifacts, or graph state;
- AI-generated findings must cite deterministic evidence IDs.

### Milestone 0.4: Scale and Backends

Planned work:

- chunked CSV metric stages;
- execution planner for expensive operations;
- approximate sketches for large cardinality and distribution checks;
- optional DuckDB or Polars backend after API contracts stabilize;
- benchmark suite for 1M, 10M, and wide-table scenarios;
- memory and runtime budget reporting.

### Milestone 0.5: More Data Tasks

Planned recipes:

- time-series analysis:
  - frequency, gaps, duplicates, seasonality, drift, and lag candidates;
  - forecasting-readiness checks.
- clustering:
  - scale sensitivity, clusterability, stability, and feature dominance;
  - no universal "best k" claim.
- data quality and validation:
  - domain-rule violations;
  - schema drift;
  - referential-integrity drift over time.

## Open Design Decisions

These should be resolved before a broader public release:

- Should scikit-learn stay a core dependency once probe models are added, or move
  behind a `ml` extra?
- Should result dataclasses gain explicit schema versions before 0.2?
- Should `AnalysisContext` grow task-specific typed subclasses?
- How much row-level example data should reports include by default?
- What privacy mode should be default for AI-assisted analysis?
- Should reports expose raw row indexes, hashed indexes, or user-specified entity
  IDs for review tables?
- Should directory loading preserve nested folder context in table names?

## Handoff Checklist for Future Agents

Before starting a new feature:

1. Read `AGENTS.md`, this file, `docs/implementation-status.md`, and the relevant
   recipe docs.
2. Run `git status --short` and avoid reverting user changes.
3. Inspect the current recipe and tests closest to the intended work.
4. Update or add evidence kinds before adding renderer sections.
5. Add tests that cover:
   - successful evidence;
   - insufficient evidence;
   - JSON export;
   - HTML export when report rendering changes;
   - sampling behavior for expensive operations.
6. Update docs in the same change.
7. Run lint, format check, mypy, tests, and build.
8. If report visuals change, generate a report and inspect it in a browser.

## Known Limitations as of 2026-06-20

- Baseline profiling is eager and can be expensive for very large files.
- Directory loading supports CSV and Parquet only.
- Schema discovery does not search self-referential relationships.
- Schema discovery does not yet infer functional dependencies or denormalization.
- Anomaly detection is diagnostic and does not train production detectors.
- Classification does not yet train probe models or compare train/test splits.
- Report rendering is static; Plotly support is only a warning path right now.
- AI-assisted analysis has not been implemented.
- Persistence is session-based; no SQLite or checkpoint store exists yet.

## Current Best Next Task

The strongest next engineering task is to add probe-model diagnostics for
classification and detector-agreement diagnostics for anomaly detection. Those
features will make the reports much more useful while still staying deterministic
and evidence-backed.
