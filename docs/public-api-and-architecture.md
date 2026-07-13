# Prism EDA Public API and Architecture

Date: 2026-06-13

Status: Living public contract. Implemented behavior is tracked in
`docs/implementation-status.md`; planned APIs remain subject to alpha changes.

## 1. Naming and releases

- PyPI distribution: `prism-eda`
- Python package: `prism_eda`
- Base install: `pip install prism-eda`
- Gemini/Gemma-assisted investigation: `pip install prism-eda[ai-gemini]`
- Interactive Plotly artifacts: `pip install prism-eda[plotly]`
- Image dataset profiling is included in the base install and uses Pillow.
- Python support: 3.11 and newer
- License: MIT

The base install contains deterministic loading, cataloging, analysis, evidence,
and reporting. The `ai-gemini` extra adds an optional LLM-assisted investigator
(LangGraph + the `google-genai` SDK) that plans over the deterministic tools
without changing the deterministic result contract.

## 2. Public API principles

- Loading, analysis, transformation planning, reporting, and AI orchestration are
  separate concerns.
- No public analysis function mutates user DataFrames.
- No analysis writes files unless an explicit export method is called.
- Long operations expose events rather than printing or prompting from core code.
- Results are typed and JSON-serializable.
- Numeric claims are represented as evidence before they become prose.
- Sampling, assumptions, failures, and uncertainty are part of the result, not
  hidden implementation details.
- Convenience functions delegate to the same session API and return the same
  result types.

## 3. Loading and datasets

```python
import pandas as pd
import prism_eda as pe

dataset = pe.load(frame)
dataset = pe.load("customers.parquet")
dataset = pe.load(["customers.csv", "orders.parquet"])
dataset = pe.load("data/", recursive=False)
dataset = pe.load(
    {"customers": customers_df, "orders": orders_df},
    names={"customers.csv": "customers"},
)
```

Proposed signature:

```python
def load(
    source: DataSource,
    *,
    recursive: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    names: Mapping[str, str] | None = None,
    read_options: Mapping[str, object] | None = None,
) -> Dataset:
    ...
```

`DataSource` accepts a pandas DataFrame, a path, a sequence of paths, a mapping of
table names to DataFrames or paths, or an existing `Dataset`.

`Dataset` owns source metadata, table handles, fingerprints, inferred schema, and
cached deterministic evidence. It does not copy a caller's DataFrame unless an
operation requires a safe internal representation. The non-mutation guarantee
still applies.

Image datasets use a sibling session because a directory of images is not a
table collection:

```python
images = pe.load_images("images/train/", recursive=True)
result = images.profile()

result = pe.profile_images("images/train/", label_strategy="directory")
```

`ImageDataset` owns discovered image paths, an optional common root, and the
labels *and* splits derived from the directory layout (`root/split/label/file`).
Its result still returns the standard `AnalysisResult`; the catalog represents
the analyzed image manifest so existing JSON and HTML exports remain compatible.

Image reports embed base64 thumbnails of the flagged files. Those live in report
**artifacts**, never in evidence, and `ImageDataset` is deliberately absent from
the assisted-analysis tool registry — raw pixels must never reach a model
provider, in line with the privacy boundary that already forbids sending raw cell
values. `thumbnails=False` omits them entirely.

Important methods:

```python
dataset.tables
dataset.catalog()
dataset.analyze(...)
dataset.classification(...)
dataset.anomaly_detection(...)
dataset.discover_schema(...)
```

## 4. Analysis context and configuration

The session API accepts explicit context instead of relying on a growing list of
loosely related keyword arguments.

```python
context = pe.AnalysisContext(
    goal="anomaly_detection",
    entity_id="person_id",
    timestamp="observed_at",
    groups=["region"],
    domain_notes="Historical health screening records",
)

config = pe.AnalysisConfig(
    mode="standard",
    random_seed=42,
    sampling="auto",
    allow_insufficient_evidence=False,
)

result = dataset.analyze(context=context, config=config)
```

Core configuration:

- `mode`: `quick`, `standard`, or `deep`;
- `sampling`: `auto`, `disabled`, or an explicit sampling policy;
- `random_seed`: stable by default;
- `allow_insufficient_evidence`: when false, conclusions are withheld below the
  recipe's sufficiency threshold; when true, best-effort conclusions remain
  visibly qualified;
- task-specific options such as expected contamination;
- callback/event subscribers;
- fairness configuration, disabled by default.

Disabling sampling does not mean every algorithm becomes valid at every scale.
An operation may still return `not_applicable` or fail with a resource-focused
error when its assumptions or safe execution bounds are violated.

## 5. Deterministic analysis

Generic session API:

```python
result = dataset.analyze(
    goal="anomaly_detection",
    context={"entity_id": "person_id"},
    mode="standard",
)
```

Typed task methods:

```python
result = dataset.anomaly_detection(
    table="events",
    target=None,
    mode="deep",
    sampling="auto",
    expected_contamination=0.02,
)

result = dataset.classification(
    target="churned",
    table="customers",
    max_categories=50,
    mode="standard",
)

result = dataset.discover_schema(
    max_key_columns=2,
    mode="standard",
)

image_result = pe.profile_images(
    "images/",
    mode="standard",
    label_strategy="directory",
    near_duplicate_threshold=4,
    thumbnails=True,
)
```

`discover_schema`, `anomaly_detection`, `classification`, and `profile_images`
are implemented.
Schema discovery returns candidate keys and relationships, not declared
constraints. Anomaly detection returns statistical review candidates, not
confirmed anomaly labels; `expected_contamination` is an optional review-sizing
assumption, not a confirmed prevalence estimate. Classification returns
target-readiness, association, leakage, and diagnostic probe-model evidence; it
does not return a production model object. Image profiling returns metadata,
quality, duplicate, leakage, loader-trap, and per-label evidence; near-duplicate
and quality flags are candidates for review, not confirmed removal instructions.
Image leakage and per-label checks depend on the directory layout, so a flat
folder of images yields neither. See
`docs/implementation-plan.md` for algorithms, thresholds, sampling behavior,
limitations, and the detailed roadmap.

Convenience functions:

```python
result = pe.anomaly_detection(df, mode="standard")
result = pe.classification(df, target="churned")
result = pe.discover_schema("data/", recursive=True)
result = pe.profile_images("images/train/")
```

Convenience functions call `load()` and the corresponding `Dataset` method. They
must not maintain a second implementation path.

## 6. Results, evidence, and status

```python
result.status
result.summary
result.findings
result.evidence
result.artifacts
result.assumptions
result.warnings
result.failures
result.sampling
result.transformation_plan
```

Run statuses:

- `completed`: required stages completed;
- `completed_with_warnings`: useful result with optional failures or caveats;
- `insufficient_evidence`: evidence did not support defensible conclusions;
- `no_meaningful_structure`: analysis found no stable signal or structure;
- `failed`: a foundational stage failed.

Every finding references one or more stable evidence IDs. Evidence includes its
scope, algorithm, parameters, sample size, seed, uncertainty, assumptions, and
artifact references. Severity and confidence remain separate fields.

An optional metric failure creates a structured `AnalysisFailure` and the run
continues. A loader, schema normalization, catalog, or required recipe-stage
failure aborts the run.

## 7. Export and rendering

```python
result.to_html("report.html")
result.to_json("report.json")

html = result.render_html()
payload = result.model_dump(mode="json")
```

HTML is self-contained by default. Core static visualizations use semantic HTML,
CSS, and inline SVG. If the Plotly extra is installed, users may request richer
interactive artifacts:

```python
result.to_html("report.html", interactive=True)
```

When `interactive=True` and Plotly is unavailable, export continues with static
charts and records a warning. A missing optional renderer must not remove all
visual evidence from the report.

## 8. Sampling and 10-million-row behavior

The execution planner classifies metrics as:

- streaming exact;
- mergeable approximate;
- bounded sample;
- full-materialization;
- quadratic or combinatorial.

Cheap counts, null summaries, extrema, and compatible aggregations should use all
rows or chunked execution. Expensive association, probe-model, local-neighbor,
and visualization operations may use deterministic samples in `auto` mode.

Every sampled result records:

- source row count;
- sampled row count;
- strategy and strata;
- seed;
- reason sampling was selected;
- known limitations;
- whether the user overrode automatic behavior.

Reports show a prominent warning when any decision-relevant evidence was sampled.

## 9. Transformation plans

Analysis may recommend changes but never applies them automatically.

```python
plan = result.transformation_plan
plan.to_dict()
```

A plan contains ordered declarative operations such as type coercion, null
handling, category consolidation, date normalization, or row review. Each step
references the evidence that motivated it and includes risk, expected impact,
preconditions, and whether human approval is required.

Applying plans and generating pandas source code are outside the initial release.

## 10. Events and callbacks

Core events include:

- `RunStarted`;
- `StageStarted` and `StageCompleted`;
- `ProgressUpdated`;
- `SamplingSelected`;
- `WarningRaised`;
- `MetricFailed`;
- `QuestionAsked`;
- `EvidenceCreated`;
- `RunCompleted` and `RunFailed`.

```python
def on_event(event: pe.Event) -> None:
    print(event.kind, event.message)

result = dataset.analyze(
    goal="anomaly_detection",
    callbacks=[on_event],
)
```

The optional terminal adapter subscribes to events and handles questions. Core
analysis never calls `input()` or assumes a notebook, terminal, or GUI.

## 11. Assisted analysis (`ai-gemini` extra)

Implemented. The investigator plans over the deterministic tools; it does not
change the result contract — `run()` returns the same `AnalysisResult`.

```python
from prism_eda.assisted_analysis import Investigator, GeminiProvider

investigator = Investigator(
    dataset,
    provider=GeminiProvider.from_env(),   # reads GEMINI_API_KEY
    privacy=None,                          # optional PrivacyPolicy
    callbacks=[...],                       # optional event observers
    max_steps=8,
)
session = investigator.start(goal="classification", context={"target": "churned"})
result = session.run()
```

`GeminiProvider` defaults to a small Gemma model and uses a portable
prompted-JSON protocol, so it works with both Gemma and Gemini models. A
deterministic `FakeProvider` drives the same flow offline for tests and docs. Any
backend can be supported by implementing the `LLMProvider.decide` method.

Hard contract (enforced in code):

- The agent may choose only registered deterministic tools. It cannot execute
  arbitrary Python and never receives raw rows — tools return aggregate summaries.
- Every reported finding must cite real evidence IDs; uncited findings are dropped
  during a validation step before synthesis.
- `insufficient_evidence` is a valid terminal status; if the model cannot converge
  within `max_steps`, the run falls back to the deterministic findings gathered so
  far and records a warning.
- The session is held in memory and is re-runnable.

Deferred (see the implementation status): critique/clarification nodes, an
interactive question/answer loop, an asynchronous `astart`/`arun` API, and
cross-process resume. The orchestration is a LangGraph state machine.

## 12. Privacy API

```python
from prism_eda.privacy import ColumnPolicy, PrivacyPolicy

privacy = PrivacyPolicy(
    default="allow",
    columns={
        "name": ColumnPolicy("alias"),
        "email": ColumnPolicy("exclude"),
        "diagnosis_notes": ColumnPolicy("redact"),
    },
    send_column_names=True,
)
```

Policy actions:

- `allow`: aggregates and approved values may be included;
- `redact`: preserve type or shape while removing content;
- `alias`: replace values with stable keyed-HMAC aliases;
- `exclude`: omit the column from model context.

Relationship discovery and joins happen locally. Raw values are not sent by
default. Keyed aliases are considered only when aggregates cannot represent the
needed relationship evidence. HMAC keys stay in memory and are never included in
reports, logs, or provider requests.

Column and table names may be sent by default because they provide important
semantic context. Users can alias or exclude them, and documentation must clearly
describe provider payload categories and Gemini data-handling implications.

## 13. Package layout

```text
src/prism_eda/
  __init__.py
  api.py
  artifacts.py
  dataset.py
  config.py
  events.py
  results.py
  exceptions.py
  catalog/
    loaders.py
    models.py
    fingerprints.py
    relationships.py
    profiling.py
  evidence/
    models.py
  analysis/
    profile.py
    schema_discovery.py
    anomaly.py
    classification.py
  transformations/
    models.py
  reporting/
    renderer.py
    templates/
  privacy/                  # allow/redact/alias/exclude policy for AI payloads
    models.py
  assisted_analysis/        # optional, the ai-gemini extra
    investigator.py         # Investigator / InvestigationSession (public entry)
    graph.py                # LangGraph flow: intake -> agent loop -> finalize
    state.py                # graph state (no raw rows, no keys)
    tools.py                # deterministic tool registry the model may call
    providers/
      base.py               # provider-neutral interface + neutral types
      _protocol.py          # portable prompted-JSON render/parse
      gemini.py             # GeminiProvider (google-genai)
      fake.py               # FakeProvider (deterministic, offline)
```

The deterministic core must never import LangChain, LangGraph, or a provider SDK.
The assisted layer consumes public dataset and evidence interfaces just like any
other client.

## 14. Initial dependency policy

Core dependencies should be limited to packages that support the 0.1 contract:

- pandas;
- NumPy;
- PyArrow;
- scikit-learn;
- Jinja2;
- a typed model/validation library if its value justifies the dependency.

Matplotlib, Seaborn, tqdm, Plotly, LangChain, LangGraph, and Gemini SDK packages
should not be unconditional core dependencies. Progress uses events; core charts
use HTML/CSS/SVG; Plotly and AI integrations live in extras.

## 15. Testing contract

- synthetic datasets encode known anomalies, leakage, imbalance, duplicates,
  missingness patterns, key relationships, and insufficient-evidence cases;
- property tests cover non-mutation, deterministic sampling, serialization, and
  evidence-reference integrity;
- algorithm tests verify behavior and limitations rather than brittle exact
  rankings where stochastic models are involved;
- report tests use structural assertions and visual snapshots;
- package tests build the wheel and install it into a clean environment;
- optional-extra tests prove the core imports without Plotly or AI dependencies;
- scale tests exercise chunked inputs and sampled expensive operations without
  requiring a 10-million-row fixture in every CI job.
