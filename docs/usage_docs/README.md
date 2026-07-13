# Prism EDA — Usage Guide

Welcome to the usage documentation for **Prism EDA**, a task-aware exploratory
data analysis library for Python.

## What makes Prism different

Most EDA tools throw a wall of numbers at you and leave you to mine for what
matters. Prism is built around the opposite idea:

> **Signal over noise.** Prism gives you exactly what's needed, for the purpose
> it's needed, with context — a short list of prioritized, decision-first
> findings, each backed by reproducible evidence.

Every number Prism reports exists as a piece of **evidence** before it becomes a
human-facing **finding**, and findings are ordered so the report leads with what
actually blocks a decision (a target leak, say) rather than whatever happened to
be computed first. Inferred keys, anomalies, and relationships are always
presented as *candidates*, never as confirmed truth.

## How these docs are organized

Read them roughly in this order:

| Page | What it covers |
|------|----------------|
| [Installation](installation.md) | Installing Prism, supported Python versions, optional extras |
| [Getting started](getting-started.md) | Your first analysis in 5 minutes, and the mental model |
| [Loading data](loading-data.md) | DataFrames, CSV/Parquet/Excel, folders, multi-table mappings |
| [The baseline profile](profile.md) | `profile()` — data quality at a glance |
| [Image dataset profile](image-datasets.md) | `profile_images()` — split leakage, loader traps, quality, duplicates |
| [Schema discovery](schema-discovery.md) | `discover_schema()` — candidate keys & relationships |
| [Anomaly detection](anomaly-detection.md) | `anomaly_detection()` — statistical review candidates |
| [Classification readiness](classification.md) | `classification()` — leakage, imbalance, separability |
| [Results & evidence](results-and-evidence.md) | The `AnalysisResult` object, findings, evidence, statuses |
| [Exporting reports](exporting-reports.md) | HTML reports, JSON, and embedding in your own tools |
| [Context & configuration](context-and-config.md) | Steering analysis with `AnalysisContext` / `AnalysisConfig` |
| [Events & progress](events-and-progress.md) | Observing long-running analyses with callbacks |
| [AI-assisted investigation](ai-assisted-analysis.md) | `Investigator` — let an LLM drive the deterministic tools (optional extra) |
| [Privacy](privacy.md) | `PrivacyPolicy` — control what reaches an AI provider |
| [Extending Prism](extending-prism.md) | A starting point for contributors |

## Reproducing the examples

Every code example in this guide runs against the same small, seeded dataset
shipped in [`examples/sample_data.py`](../../examples/sample_data.py). The
outputs shown are captured from real runs — copy any snippet and you'll get the
same result.

```python
import prism_eda as pe
from examples.sample_data import load_sample

dataset = pe.load(load_sample())
result = dataset.classification("churned", table="customers")
print(result.summary)
```

```text
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
```

> The `examples/` folder is at the repository root. Run snippets from there, or
> add the repo root to your `PYTHONPATH`, so that `from examples.sample_data
> import ...` resolves.

## Status of the library

Prism EDA is in **early alpha**. This guide documents only what is implemented
and runnable today — including the optional
[AI-assisted investigation](ai-assisted-analysis.md) layer. For what's planned
next (more recipes, richer privacy payloads, scale backends), see the
[implementation status ledger](../implementation-status.md) and the
[public API & architecture doc](../public-api-and-architecture.md).
