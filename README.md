# Prism EDA

Prism EDA is a task-aware exploratory data analysis library for Python. It is
being built around a deterministic evidence engine, goal-specific analysis
recipes, self-contained reports, and optional AI-assisted investigation.

The project is currently in early alpha development. Version 0.1 focuses on the
local deterministic foundation; Gemini-assisted analysis is planned for 0.2.

## Quick start

```python
import prism_eda as pe

dataset = pe.load("data/customers.parquet")
result = dataset.profile()

result.to_html("profile.html")
result.to_json("profile.json")
```

DataFrames, CSV files, Parquet files, mappings of related tables, and directories
are accepted. Analysis does not mutate input DataFrames and does not write files
until an explicit export method is called.

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

Directory loading currently supports CSV and Parquet files. Use `names=` when
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

See [the product research brief](docs/product-research-brief.md) and
[the public API specification](docs/public-api-and-architecture.md) for the
confirmed direction.

Further documentation:

- [Schema discovery](docs/schema-discovery.md)
- [Implementation plan and handoff](docs/implementation-plan.md)
- [Implementation status](docs/implementation-status.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Agent handoff](AGENTS.md)
