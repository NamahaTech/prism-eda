# Prism EDA

Prism EDA is a task-aware exploratory data analysis library for Python. It is
being built around a deterministic evidence engine, goal-specific analysis
recipes, self-contained reports, and optional AI-assisted investigation.

The project is currently in early alpha development. It pairs a local
deterministic foundation with an optional Gemini/Gemma-assisted investigator that
plans and explains analyses over those deterministic tools (install with the
`ai-gemini` extra).

> **New here? Start with the [Usage Guide](docs/usage_docs/README.md)** — a
> step-by-step walkthrough of installing Prism, loading data, every analysis
> recipe, and reading results, with runnable, verified examples.

## Quick start

```python
import prism_eda as pe

dataset = pe.load("data/customers.parquet")
result = dataset.profile()

result.to_html("profile.html")
result.to_json("profile.json")
```

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
leakage-screened probe-model separability, and hard-example candidates. Findings
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
[AI-assisted guide](docs/usage_docs/ai-assisted-analysis.md) and the
[privacy guide](docs/usage_docs/privacy.md). The deterministic core never imports
an LLM library.

See [the product research brief](docs/product-research-brief.md) and
[the public API specification](docs/public-api-and-architecture.md) for the
confirmed direction.

Further documentation:

- [**Usage Guide**](docs/usage_docs/README.md) — install, load, analyze, export (start here)
- [AI-assisted investigation](docs/usage_docs/ai-assisted-analysis.md) · [Privacy](docs/usage_docs/privacy.md)
- [Schema discovery](docs/schema-discovery.md)
- [Implementation plan and handoff](docs/implementation-plan.md)
- [Implementation status](docs/implementation-status.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Agent handoff](AGENTS.md)
