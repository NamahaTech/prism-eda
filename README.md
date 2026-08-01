<p align="center">
  <!-- Absolute raw URL: PyPI renders the README outside the repository, so a
       relative image path would show as a broken image on the project page. -->
  <img src="https://raw.githubusercontent.com/NamahaTech/prism-eda/main/docs/assets/prism-eda-logo.png" alt="Prism EDA" width="150">
</p>

<h1 align="center">Prism EDA</h1>

<p align="center">
  <a href="https://pypi.org/project/prism-eda/"><img src="https://img.shields.io/pypi/v/prism-eda.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/prism-eda/"><img src="https://img.shields.io/pypi/pyversions/prism-eda.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/NamahaTech/prism-eda/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/prism-eda.svg" alt="MIT license"></a>
  <a href="https://github.com/NamahaTech/prism-eda/actions/workflows/ci.yml"><img src="https://github.com/NamahaTech/prism-eda/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
</p>

Prism EDA is a task aware exploratory data analysis library for Python. It is
being built around a deterministic evidence engine, goal specific analysis
recipes, self contained reports, and optional AI-assisted investigation.

The project is currently in early alpha development. It pairs a local
deterministic foundation with an optional Gemini/Gemma-assisted investigator that
plans and explains analysis over those deterministic tools (install with the
`ai-gemini` extra).

> **New here? Start with the [Usage Guide](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/README.md)** — a
> step-by-step walkthrough of installing Prism, loading data, every analysis
> recipe, and reading results, with runnable, verified examples.

## Install

```bash
pip install prism-eda
```

Python 3.11+. Optional extras, each installable on its own or together:

| Extra | `pip install "prism-eda[...]"` | Adds |
|-------|-------------------------------|------|
| `excel` | `excel` | Reading `.xlsx` sources via openpyxl |
| `ai-gemini` | `ai-gemini` | The optional AI investigator (LangGraph + google-genai) |
| `plotly` | `plotly` | Interactive chart export |

The deterministic core needs none of them, and never imports an LLM library.

## Quick start

```python
import prism_eda as pe

dataset = pe.load("data/customers.parquet")
result = dataset.profile()

result.to_html("profile.html")
result.to_json("profile.json")
```

The profile separates two things most profilers merge. **Issues** are defects —
missing, duplicated, mistyped, or placeholder values, mixed date formats,
sentinel numbers, columns that are copies of each other. **Alerts** are facts
that are true but not broken — columns that move together, a column that is
all-unique, the window a timestamp covers. Filing those in one list devalues
both, so they are separate fields and separate report sections:

```python
from prism_eda.evidence.models import split_findings

issues, alerts = split_findings(result.findings)
```

The report shows one card per column with its distribution, and names the
distribution family where one fits — reporting a Kolmogorov-Smirnov distance
rather than a p-value, and abstaining when nothing fits well. It then measures
every column pair with the statistic that suits their types (Spearman, Cramér's
V, or the correlation ratio), plots the strongest pairs, shows which columns go
missing together, and prints the first and last rows. Pass `detail="full"` to
raise the caps on wide data; anything a cap removes is stated in the report
rather than dropped silently.

DataFrames, CSV files, Parquet files, Excel files, mappings of related tables, and
directories are accepted. Analysis does not mutate input DataFrames and does not
write files until an explicit export method is called. (Excel needs the optional
`excel` extra: `pip install "prism-eda[excel]"`.)

## Analyze a folder

Pass a folder path directly to an analysis function. Prism EDA loads every CSV
and Parquet file in the folder as a separate named table; each filename stem is
used as its table name.

```python
import prism_eda as pe

# Loads data/customers.csv, data/orders.parquet, and other supported files.
profile = pe.profile("data/")
profile.to_html("profile.html")

# Analyze relationships across all tables in the folder.
schema = pe.discover_schema("data/", mode="standard")
schema.to_html("schema-report.html")
```

To load the folder first and run multiple analyses on the same dataset:

```python
import prism_eda as pe

dataset = pe.load("data/")

print(list(dataset.tables))
profile = dataset.profile()
schema = dataset.discover_schema()
anomalies = dataset.anomaly_detection()
classification = dataset.classification("target_column")
```

Subfolders are excluded by default. Use `recursive=True` to include them, and
optionally filter discovered files with glob-style patterns:

```python
dataset = pe.load(
    "data/",
    recursive=True,
    include=["*.csv", "*.parquet"],
    exclude=["*_backup.csv", "archive/*"],
)
```

Directory loading supports CSV, Parquet, and Excel files (Excel via the `excel`
extra). Use `names=` when
you need to override the table names derived from filenames, and `read_options=`
to pass options through to pandas readers.

## Profile an image dataset

Image folders can be profiled separately from tabular datasets. Prism scans image
headers and lightweight visual-quality signals, then reports issues as
evidence-backed findings instead of a raw metadata dump — and the report *shows*
you the flagged images rather than listing their paths.

```python
import prism_eda as pe

images = pe.profile_images("images/")
images.to_html("image-profile.html")
```

Labels and splits are read from the usual `images/train/cat/001.png` layout, which
unlocks the checks that matter most:

- **Leakage** — the same image in both `train` and `test` inflates every metric
  you report without changing the model, so it leads the report as `critical`.
  The same check across labels catches one image filed under two classes.
- **Loader traps** — files that decode cleanly but reach your pipeline changed:
  EXIF orientation tags that silently rotate (and sometimes transpose) an image,
  a `.jpg` that is really a PNG, greyscale stored in three identical channels,
  used alpha channels, and truncated files.
- **Duplicates, outliers, and quality** — exact and perceptual-hash duplicate
  candidates, odd resolutions and aspect ratios, and dark, blown-out, blurry, or
  blank images.
- **Per-label breakdown** — dimensions and brightness per class, so collection
  bias in one class is not averaged away.

Flagged images are embedded in the report as thumbnails (duplicate candidates
side by side), and the report stays a single offline file. Pass
`thumbnails=False` to omit them, or `label_strategy=None` to disable label
inference. See [the image dataset guide](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/image-datasets.md).

## Discover related tables

```python
import prism_eda as pe

dataset = pe.load(
    {
        "customers": customers_df,
        "orders": orders_df,
        "order_items": order_items_df,
    }
)

schema = dataset.discover_schema(mode="standard")
schema.to_html("schema-report.html")
```

Schema discovery reports candidate single/composite keys, directional
relationships, cardinality, orphan rows, confidence, evidence lineage, and a
self-contained ER diagram with candidate PK/FK roles and one/many cardinality
marks. Candidates are never silently treated as declared database constraints.

## Task-aware recipes

```python
import prism_eda as pe

anomalies = pe.anomaly_detection(
    "data/events.parquet",
    expected_contamination=0.02,  # optional review-prevalence assumption
)
anomalies.to_html("anomaly-report.html")

classification = pe.classification(
    "data/training.csv",
    target="label",
)
classification.to_html("classification-report.html")
```

Anomaly detection currently reports statistical review candidates such as robust
numeric tails, multivariate robust-score candidates, Isolation Forest and
local-density ranked candidates, detector agreement, conditional numeric
surprises, rare categories, and optional rare-label summaries. It does not mark
rows as confirmed anomalies.

Classification currently reports target validity, class imbalance, conflicting
labels, typed feature-target associations, missingness by class, high-cardinality
risks, identifier-like columns to exclude, deterministic leakage candidates,
leakage-screened probe-model separability, local class-overlap candidates,
hard-example candidates, and context-aware group/time split guidance. Findings
lead with severity (so confirmed-style leakage surfaces first) and the summary
states a readiness verdict. It is a readiness diagnostic, not a production model
training pipeline.

## AI-assisted investigation (optional)

Install the `ai-gemini` extra to let a language model plan and explain an
analysis — but only by calling Prism's deterministic tools. The model never sees
raw data, never runs code, and every finding it reports is dropped unless it cites
real evidence those tools produced.

```python
import prism_eda as pe
from prism_eda.assisted_analysis import Investigator, GeminiProvider

dataset = pe.load("data/customers.parquet")
investigator = Investigator(dataset, provider=GeminiProvider.from_env())
result = investigator.start(goal="classification", context={"target": "churned"}).run()
result.to_html("investigation.html")
```

`pip install "prism-eda[ai-gemini]"`. The result is the same `AnalysisResult` the
deterministic recipes return. See the
[AI-assisted guide](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/ai-assisted-analysis.md) and the
[privacy guide](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/privacy.md). The deterministic core never imports
an LLM library.

See [the product research brief](https://github.com/NamahaTech/prism-eda/blob/main/docs/product-research-brief.md) and
[the public API specification](https://github.com/NamahaTech/prism-eda/blob/main/docs/public-api-and-architecture.md) for the
confirmed direction.

Further documentation:

- [**Usage Guide**](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/README.md) — install, load, analyze, export (start here)
- [AI-assisted investigation](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/ai-assisted-analysis.md) · [Privacy](https://github.com/NamahaTech/prism-eda/blob/main/docs/usage_docs/privacy.md)
- [Schema discovery](https://github.com/NamahaTech/prism-eda/blob/main/docs/schema-discovery.md)
- [Implementation plan and handoff](https://github.com/NamahaTech/prism-eda/blob/main/docs/implementation-plan.md)
- [Implementation status](https://github.com/NamahaTech/prism-eda/blob/main/docs/implementation-status.md)
- [Maintainer guide](https://github.com/NamahaTech/prism-eda/blob/main/docs/maintainer-guide.md)
- [Agent handoff](https://github.com/NamahaTech/prism-eda/blob/main/AGENTS.md)
