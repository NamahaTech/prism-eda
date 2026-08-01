<p align="center">
  <!-- Replace this text mark with docs/assets/prism-eda-logo.svg when the public asset is added. -->
  <strong>PRISM-EDA</strong>
</p>

<h1 align="center">Prism-EDA</h1>

<p align="center">
  <strong>Give Prism EDA your data, and it turns it into decision ready insights through seven task aware recipes.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/prism-eda/"><img src="https://img.shields.io/pypi/v/prism-eda?color=6f52ed&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/prism-eda/"><img src="https://img.shields.io/pypi/pyversions/prism-eda?color=6f52ed" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f9d55" alt="MIT license"></a>
  <a href="https://github.com/NamahaTech/prism-eda/actions"><img src="https://github.com/NamahaTech/prism-eda/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://codecov.io/gh/NamahaTech/prism-eda"><img src="https://codecov.io/gh/NamahaTech/prism-eda/branch/main/graph/badge.svg" alt="Test coverage"></a>
  <a href="https://github.com/NamahaTech/prism-eda"><img src="https://img.shields.io/badge/docs-guide-0b7285" alt="Documentation"></a>
  <a href="https://pypi.org/project/prism-eda/"><img src="https://img.shields.io/pypi/dm/prism-eda?label=downloads" alt="PyPI downloads"></a>
  <a href="https://github.com/NamahaTech/prism-eda"><img src="https://img.shields.io/badge/code%20style-ruff-261230" alt="Ruff"></a>
</p>

> **One dataset in. Seven useful rays out.** Prism EDA is a Python library for
> deterministic, task aware exploratory data analysis across **tabular and image
> datasets**.

## What is Prism EDA?

Prism-eda is a modular python library built for EDA (exploratory data analysis). Using Prism-eda, a dataset can be profiled, explored as a
schema, checked for suspicious observations, or assessed for a modelling task
without turning exploratory analysis into a collection of disconnected scripts.

That is **context-aware EDA**: the same evidence engine uses the task, target,
entity identifier, timestamp, groups, domain notes, and assumptions you provide
to choose relevant diagnostics and explain their limits. Prism never treats an
inferred key, relationship, anomaly, or leakage signal as confirmed business
truth. It gives you citable evidence and review recommendations.

The prism metaphor: one input—data can be examined through
seven distinct objectives:

| Prism-rays (recipes)     | Objectives                                            |
|--------------------------|-------------------------------------------------------|
| **Baseline Profiling**   | Complete baseline dataset quality report.             |
| **Schema Discovery**     | Relationship insights for tables and entities.        |
| **Anomaly Detection**    | Observations, combinations, outliers detection.       |
| **Classification**       | Categorization, clustering and segmentation support.  |
| **Regression**           | Prediction and regression for numerical outcomes.     |
| **Time Series Analysis** | Time-based data forecasting analysis.                 |
| **Data Statistics**      | Data characteristics, distribution and visualization. |

## Why it exists

Traditional profiling tools produces a broad catalog: types, missing values, duplicates, distributions, and correlations. But a raw
profile often leaves the most important next question unanswered:

- Can these tables safely be joined, and what are the likely keys?
- Are there suspicious rows worth review, or merely a long-tailed distribution?
- Is a target suitable for classification, or is a feature leaking the answer?
- Are a class imbalance, repeated customer, or timestamp likely to invalidate a
  random validation split?
- Can an image folder be trusted, or does it contain duplicates across train and
  test, corrupt files, misleading labels, or loader traps?

Prism EDA is built for those decisions. Its deterministic local tools create
the numerical evidence; optional AI can plan and interpret over that evidence,
but never invent numeric truth. Reports remain useful without a CDN, browser
JavaScript, or an AI provider.

## Features

### Data loading and profiling

- Load pandas DataFrames, CSV, Parquet, Excel workbooks, named table mappings,
  paths, and directories.
- Build deterministic dataset/table fingerprints and column catalogs.
- Profile shape, memory use, physical and semantic types, missingness,
  cardinality, duplicates, constants, robust numeric summaries, and top values.
- Analyze without mutating the caller's DataFrame or writing files until an
  explicit export is requested.

### Task-aware analysis

- Infer candidate single/composite keys and directional table relationships with
  coverage, orphan counts, cardinality, confidence, and an ER diagram.
- Find numeric tails, multivariate candidates, local density candidates,
  conditional surprises, rare categories, distribution regimes, and ranked
  anomaly review rows.
- Assess classification targets for balance, label conflicts, associations,
  missing gaps, deterministic leakage,
  probe separability, hard examples, local class overlap, and split guidance.
- Profile image datasets for decode failures, dimensions, formats, EXIF,
  duplicates, near-duplicates, split leakage, label conflicts, quality flags,
  loader traps, outliers, and label-level imbalance.

### Evidence, reporting, and developer experience

- Stable evidence IDs connecting each finding, artifact, and recommendation to
  the measurement behind it.
- Severity and confidence as separate concepts; explicit warnings, failures,
  assumptions, and sampling records.
- Portable self-contained HTML reports, complete JSON, and in-memory `dict`
  exports.
- Static report fallbacks and embedded interactive schema diagrams.
- Framework-neutral lifecycle/progress callbacks.
- Deterministic sampling, configurable compute depth, stable random seeds, type
  checking, linting, packaging, and a broad regression test suite.

### Optional AI-assisted investigation

- A provider-neutral investigator that selects from a fixed deterministic tool
  registry rather than executing arbitrary code.
- Gemini/Gemma integration plus an offline deterministic `FakeProvider` for
  tests and demos.
- Evidence-citation validation: unsupported model claims are discarded.
- Privacy controls to allow, redact, alias, or exclude columns from provider
  context; raw values are withheld by default.

## Demo GIFs

> Add compact, looped GIFs to `docs/assets/demos/` before public release and
> replace the placeholders below. Keep each under roughly 10 MB and provide a
> static screenshot for accessibility.

| Demo | Asset placeholder | What it should show |
|---|---|---|
| First report | `docs/assets/demos/profile.gif` | Load a CSV and open one self-contained HTML report. |
| Schema map | `docs/assets/demos/schema.gif` | Infer joins and explore the interactive ER diagram. |
| Investigation | `docs/assets/demos/ai-investigation.gif` | State a goal, show deterministic tool calls, then view cited findings. |
| Image audit | `docs/assets/demos/image-profile.gif` | Surface split leakage and review paired duplicate thumbnails. |

```md
<!-- Example once the asset exists -->
![Prism EDA schema discovery demo](docs/assets/demos/schema.gif)
```

## Installation

Prism EDA supports Python 3.11 and newer on macOS, Linux, and Windows.

### pip

```bash
python -m pip install prism-eda
```

### uv

```bash
uv add prism-eda
```

### Poetry

```bash
poetry add prism-eda
```

### Conda

```bash
conda create -n prism-eda python=3.11
conda activate prism-eda
python -m pip install prism-eda
```

### Pipenv

```bash
pipenv install prism-eda
pipenv shell
```

### Optional extras

```bash
# Read Excel workbooks.
python -m pip install "prism-eda[excel]"

# Use the optional Gemini/Gemma assisted investigator.
python -m pip install "prism-eda[ai-gemini]"

# Request optional Plotly-backed interactive report enhancements.
python -m pip install "prism-eda[plotly]"
```

### Development setup

```bash
git clone https://github.com/NamahaTech/prism-eda.git
cd prism-eda
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test,dev]'

ruff check src tests
ruff format --check src tests
mypy src/prism_eda
pytest --cov=prism_eda --cov-report=term-missing
python -m build
```

### Docker

Use an official Python image when you want an isolated, disposable run without a
host Python installation:

```bash
docker run --rm -it -v "$PWD:/work" -w /work python:3.11-slim bash
python -m pip install prism-eda
python your_analysis.py
```

## Quick start

Generate a shareable report in three lines:

```python
import prism_eda as pe

result = pe.profile("data/customers.parquet")
result.to_html("profile.html")
```

`profile.html` is a standalone report. For programmatic workflows, call
`result.to_dict()` or `result.to_json("profile.json")`.

## Examples

### One table: understand the data

```python
import prism_eda as pe

dataset = pe.load("data/customers.csv")
profile = dataset.profile()
print(profile.summary)
profile.to_html("customers-profile.html")
```

### Multiple tables: connect the schema

```python
import prism_eda as pe

dataset = pe.load("data/", recursive=True)
schema = dataset.discover_schema(mode="standard")
schema.to_html("schema.html")
```

### Classification: test readiness before training

```python
import prism_eda as pe

result = pe.classification(
    "data/training.csv",
    target="churned",
    context={"entity_id": "customer_id", "timestamp": "observed_at"},
    mode="deep",
)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
result.to_html("classification-readiness.html")
```

### Anomaly detection: make a review list, not a verdict

```python
import prism_eda as pe

result = pe.anomaly_detection(
    "data/transactions.parquet",
    expected_contamination=0.02,
)
result.to_html("anomaly-review.html")
```

### Image dataset: audit quality and split leakage

```python
import prism_eda as pe

images = pe.profile_images("images/", thumbnails=True)
images.to_html("image-profile.html")
```

### AI-assisted exploration

```python
import prism_eda as pe
from prism_eda.assisted_analysis import GeminiProvider, Investigator

dataset = pe.load("data/customers.parquet")
result = Investigator(dataset, provider=GeminiProvider.from_env()).start(
    goal="classification",
    context={"target": "churned", "domain_notes": "Subscription customers."},
).run()
result.to_html("investigation.html")
```

Set `GEMINI_API_KEY` in the environment before creating `GeminiProvider`.

## AI features

The optional option gives an LLM a tightly constrained role: it chooses
which Prism tools to call and explains their evidence. It does not inspect raw
rows, run arbitrary code, or create findings without citations.

```text
goal + privacy-safe catalog
           │
           ▼
bounded provider/tool loop ──► deterministic recipes ──► evidence bank
           │                                                   │
           └──────── cited synthesis + interpretation ◄───────┘
```

- **Providers:** `GeminiProvider` supports Gemini and Gemma through Google’s
  GenAI SDK; `FakeProvider` makes offline, deterministic tests possible. The
  `LLMProvider` protocol keeps adapters replaceable.
- **Privacy:** aggregate catalog/recipe summaries are shared according to a
  `PrivacyPolicy`; raw cell values are not sent by default. Per-column controls
  support allow, redact, alias, and exclude actions. Aliases use an in-memory
  keyed HMAC.
- **Token discipline:** the provider receives compact summaries, bounded tool
  calls, and only the evidence relevant to the declared goal—not raw tables or
  an unrestricted notebook session.
- **Grounding:** every AI-generated finding must cite an existing deterministic
  evidence ID. Uncited findings are removed, and insufficient evidence is a
  valid outcome.

See [AI-assisted analysis](docs/usage_docs/ai-assisted-analysis.md) and
[Privacy](docs/usage_docs/privacy.md) for configuration and provider details.



## Detailed documentation:

- [**Usage Guide**](docs/usage_docs/README.md) — install, load, analyze, export (start here)
- [AI-assisted investigation](docs/usage_docs/ai-assisted-analysis.md) · [Privacy](docs/usage_docs/privacy.md)
- [Schema discovery](docs/schema-discovery.md)
- [Implementation plan and handoff](docs/implementation-plan.md)
- [Implementation status](docs/implementation-status.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Agent handoff](AGENTS.md)

## Generated reports

Every result can become a single offline HTML file. The report design 
leads with the verdict and severe findings, then preserves
the catalog, evidence, assumptions, warnings, sampling disclosure, artifacts,
and transformation recommendations needed to audit that conclusion.

| Report | What readers can inspect |
|---|---|
| Baseline profile | Dataset/table summary, column catalog, missingness, types, distributions, and warnings |
| Schema discovery | Candidate keys/relationships, confidence, orphan counts, and a static/interactive ER diagram |
| Anomaly review | Ranked review rows, detector agreement, distribution shape, row-level explanations, and charts |
| Classification readiness | Class balance, leakage/identifier risks, associations, probe results, hard examples, overlap, and split guidance |
| Image profile | Quality/loader checks, duplicate and leakage evidence, class balance, charts, and embedded thumbnail contact sheets |
| Investigation | Deterministic findings plus AI provenance and grounded interpretation |

```md
<!-- Add public-release screenshots here. -->
![Classification readiness report](docs/assets/screenshots/classification-report.png)
![Image dataset profile report](docs/assets/screenshots/image-profile-report.png)
```

## Architecture

```text
DataFrame / files / folders / image roots
                  │
                  ▼
       loaders + dataset session objects
                  │
                  ▼
     catalog, fingerprints, semantic typing
                  │
                  ▼
 task recipes ──► evidence ──► findings / artifacts / recommendations
                  │                         │
                  └──── optional investigator│
                                            ▼
                             AnalysisResult → HTML / JSON / dict
```

| Module | Responsibility |
|---|---|
| `api.py` | Thin import-first convenience functions. |
| `dataset.py` / `image_dataset.py` | Session objects, loading state, and recipe dispatch. |
| `catalog/` | Loaders, fingerprints, table/column cataloging, keys, and relationships. |
| `analysis/` | Goal-specific deterministic recipes. |
| `evidence/` | Stable evidence and finding contracts. |
| `artifacts.py` / `transformations/` | Report-ready artifacts and declarative, non-mutating recommendations. |
| `reporting/` | Shared self-contained HTML rendering and visual fallbacks. |
| `assisted_analysis/` | Optional provider adapters, bounded tool loop, citation validation, and interpretation. |
| `privacy/` | Controls for provider-facing metadata and values. |

Core code does not import Plotly, LangChain, LangGraph, or a model-provider SDK.
Those capabilities remain optional at the package boundary.

## Configuration

### Context: tell Prism what the data means

```python
import prism_eda as pe

context = pe.AnalysisContext(
    goal="classification",
    target="churned",
    entity_id="customer_id",
    timestamp="observed_at",
    groups=("region",),
    domain_notes="Monthly subscription customers.",
    assumptions=("One row represents one customer-month.",),
)
```

### Compute and reproducibility

```python
config = pe.AnalysisConfig(
    mode="standard",             # quick | standard | deep
    sampling="auto",             # auto | disabled
    random_seed=42,
    allow_insufficient_evidence=False,
)
result = pe.load("data.csv").profile(config=config)
```

### Report and output options

```python
result.to_html("report.html")                 # standalone HTML
result.to_html("report.html", interactive=True)  # optional Plotly enhancement
result.to_json("report.json", indent=2)       # complete serializable result
payload = result.to_dict()                      # in-memory dict
```

### Image profile options

```python
result = pe.profile_images(
    "images/",
    recursive=True,
    include=["*.png", "*.jpg"],
    exclude=["archive/*"],
    near_duplicate_threshold=4,
    thumbnails=True,
    thumbnail_size=600,
)
```

### Events and callbacks

Pass callbacks to receive lifecycle and evidence events without coupling Prism
to a logging framework or UI framework. See
[Events & progress](docs/usage_docs/events-and-progress.md).

## FAQ

### Which data formats can Prism EDA load?

Pandas DataFrames, CSV, Parquet, file lists, named mappings of tables, and
directories are supported. Excel files use the optional `excel` dependency.
Image profiling accepts common BMP, GIF, JPEG, PNG, TIFF, and WebP files.

### Does Prism modify my DataFrame or write files automatically?

No. Public analysis does not mutate caller DataFrames. Files are written only
when you explicitly call an export method such as `to_html()` or `to_json()`.

### Is AI required?

No. The deterministic core is fully useful without an AI provider. AI assistance
is an optional extra used to plan and interpret deterministic analyses.

### What does an AI provider receive?

By default, compact aggregates and permitted column metadata—not raw rows or
cell values. `PrivacyPolicy` can redact, alias, exclude, or allow individual
columns; enabling raw values is an explicit opt-in.

### Can Prism handle very large datasets?

Recipes use deterministic sampling budgets where appropriate and record every
sample in the result. CSV loading and baseline profiling are eager, so memory
should be considered for very large sources.

### Can I customize thresholds and analysis depth?

Yes. Use recipe-specific options, `AnalysisContext`, and `AnalysisConfig` to
set compute depth, sampling, seeds, thresholds, reader options, and task context.

### How is Prism different from a general profiling library?

Profiling is one ray. Prism keeps profiling as shared evidence, then applies it
to a declared decision such as joining tables safely, reviewing anomalies,
checking classification readiness, or auditing an image dataset.

### Are inferred keys, relationships, and anomalies facts?

No. They are candidates supported by evidence and confidence, designed for user
or domain-expert confirmation.

## Open Contribution

Prism EDA is currently under active private development and will open for
community contributions as the public workflow matures.

The intended contribution workflow is:

1. Discuss a focused change through an issue or design proposal.
2. Preserve the evidence-first and non-mutation contracts.
3. Add synthetic fixtures with known behavior and test evidence lineage.
4. Update user-facing docs, architecture/status notes, and the changelog in the
   same change.
5. Run Ruff, mypy, tests, package build, and report visual QA before review.

Until public contribution channels are announced, please use the repository’s
issue/contact path for feedback and collaboration requests.

## License

Prism EDA is released under the [MIT License](LICENSE). You may use, copy,
modify, merge, publish, distribute, sublicense, and sell copies of the software,
provided that the copyright and license notice are included. The software is
provided without warranty; see [LICENSE](LICENSE) for the complete terms.

---

**Explore the data now...**
