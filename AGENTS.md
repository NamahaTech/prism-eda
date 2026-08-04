# Prism EDA Agent Handoff

This file is the shortest path for another engineering agent to continue the
project safely. Read it before changing public behavior, then follow the linked
documents for detail.

## Project intent

Prism EDA is a Python-first, task-aware exploratory data analysis library. Its
central rule is that deterministic local tools produce evidence; optional AI may
later plan and explain analysis, but it does not invent numeric truth.

## Current implementation

- Distribution: `prism-eda`
- Import package: `prism_eda`
- Supported Python: 3.11+
- Implemented recipes: baseline profile, schema discovery, anomaly detection,
  classification, regression, image dataset profile
- Optional AI-assisted investigation via the `ai-gemini` extra
  (`prism_eda.assisted_analysis`): an LLM plans over the deterministic tools only
- Planned next recipes: time-series, clustering

See [implementation status](docs/implementation-status.md) for the exact ledger.

## Commands

```bash
source .venv/bin/activate
pip install -e '.[test,dev]'
ruff check .
ruff format --check .
mypy src/prism_eda
pytest --cov=prism_eda --cov-report=term-missing
python -m build
```

## Architectural invariants

1. Public analysis never mutates caller DataFrames.
2. Analysis writes no files until an explicit export method is called.
3. Findings cite stable evidence IDs.
4. Severity and confidence are separate fields.
5. Sampling is deterministic, recorded, and visible in reports.
6. Low evidence may produce `insufficient_evidence` or
   `no_meaningful_structure`; do not force a conclusion.
7. Optional metric failures are recorded and analysis continues. Foundational
   loading, catalog, or required-stage failures abort the run.
8. Core code does not import Plotly, LangChain, LangGraph, or model-provider SDKs.
9. Reports remain useful without JavaScript, CDNs, or optional visualization
   packages.
10. Inferred keys, anomalies, and relationships are candidates until confirmed by
    the user or domain constraints.

## Module ownership

- `api.py`: top-level convenience functions only; delegate to `Dataset`.
- `dataset.py`: session object and recipe dispatch.
- `image_dataset.py`: sibling session for image folders — path discovery, and
  the label/split inferred from the directory layout. A directory of images is
  not a table collection, so it does not go through `catalog/`.
- `catalog/`: loading, fingerprints, column catalog, keys, and relationships.
- `analysis/`: task recipes that turn deterministic computations into evidence,
  findings, artifacts, warnings, and statuses. The baseline profile is split by
  question: `quality_checks.py` (is anything *wrong*?), `distributions.py` (what
  shape is each column?), `associations.py` (how do columns relate, and what is
  missing together?), with `profile.py` orchestrating. `_limits.py` holds the
  per-detail caps; `_numeric.py` holds binning/modality shared with `anomaly.py`.
  Regression is split the same way: `regression_target.py` (what shape is the
  thing being predicted?), `regression_signal.py` (what carries signal, and what
  is leaking or merely duplicated?), `regression_probe.py` (what happens when a
  model is actually fitted?), with `regression.py` orchestrating and
  `_regression.py` holding the shared sampling, table resolution, and feature
  selection every stage must agree on.
- `evidence/`: provider-neutral evidence and finding contracts.
- `artifacts.py`: structured report artifacts such as schema graphs.
- `reporting/`: the shared self-contained renderer. `sections.py` owns which
  sections a report has, in what order — the navigation, the anchors, and the
  numbering all read from that one list, so they cannot drift apart. Add a
  section there and in the template together, and `tests/test_report_sections.py`
  will hold the two in agreement.
- `transformations/`: declarative recommendations; no automatic mutation.
- `results.py`: stable result object and explicit exports.
- `privacy/`: allow/redact/alias/exclude controls for AI-assisted payloads.
- `assisted_analysis/`: optional LLM layer (providers, deterministic tool
  registry, LangGraph flow, investigator). Depends on the core one-way; the core
  never imports it. Lives behind the `ai-gemini` extra.

## Documentation discipline

Every feature change must update documentation in the same change:

- User-visible behavior: update `README.md` or a focused guide in `docs/`.
- Public API or ownership boundary: update `docs/public-api-and-architecture.md`.
- Completed/planned scope: update `docs/implementation-status.md`.
- Algorithm, thresholds, assumptions, or limitations: update the relevant feature
  guide.
- Release-facing behavior: update `CHANGELOG.md`.

Do not postpone documentation into a future cleanup task.

## Testing expectations

- Add synthetic fixtures with known pathologies or relationships.
- Assert evidence lineage, not only prose.
- Test non-mutation and deterministic behavior where applicable.
- Test JSON and HTML output when adding a new artifact or result field.
- Run visual QA in a browser for report layout changes, including a mobile width.
- Build the wheel and confirm packaged templates/assets are included.

## A detector that never stays quiet is not a detector

Several standard diagnostics report *something* on every dataset. Any new check
must be tested against clean, well-specified data and produce nothing. Three
guards in the regression recipe exist only for this reason, and the reasoning
generalizes:

- Cook's `4/n` is a screening convention that flags a few percent of rows in any
  fit. Review rows therefore require decisive influence, not merely clearing it.
- Equal-width bins always leave thin bins in a bell curve's tails, so a
  weak-support gap requires real mass on *both* sides.
- Absolute error scales with target magnitude, and a grouping column that
  predicts the target shrinks within-group spread. Subgroup error is therefore
  scaled by each group's own spread *and* compared against sibling levels rather
  than the overall rate.

Name-matching heuristics need the same care: a substring test makes a target
called `y` "leak" into `x1_copy`, so name evidence comes from whole tokens of at
least three characters.

## Important limitations

- Pandas is the only in-memory backend.
- Profile data-quality checks scan every row and never sample; correlations,
  scatters, and distribution fitting sample above the `detail` budget and record
  it. Do not quietly extend sampling to the quality checks: those are exact
  claims about defects.
- Distribution fitting reports a KS *distance*, never a p-value, because the
  parameters are estimated from the same data. Continuous families are chosen by
  AIC so a more flexible family cannot win on flexibility alone.
- CSV inputs are currently loaded eagerly; chunked execution is still planned.
- Schema discovery considers at most 12 likely columns per table and key width is
  capped at three.
- Non-string column names are profiled but skipped by schema discovery with a
  warning.
- Automated functional-dependency discovery and self-referential relationships
  are not implemented.
- Persistent AI sessions are not implemented.
- Image profiling reads pixels only through Pillow: no deep embeddings, so
  near-duplicates are perceptual-hash candidates and semantic duplicates (the
  same object photographed twice) are not detected.
- Image labels and splits are inferred from directory names only; annotation
  files (COCO/YOLO/CSV manifests) are not read.
- Image evidence is not exposed to the assisted-analysis tool registry, and
  thumbnails live in artifacts rather than evidence. Keep it that way: raw
  pixels must never reach a model provider.
- Regression probes are Ridge and Huber only. A weak probe means a *linear*
  model finds little, not that the target is unlearnable.
- Regression leverage and Cook's distance come from an OLS fit on the screened
  design, capped at 30 features, and inherit its assumptions.
- Regression censoring is inferred from repeated values, so a genuinely popular
  price point is indistinguishable from a cap without domain confirmation.
- Regression review rows carry actual/predicted values and per-row feature
  values. Those live in evidence for the local report only; the assisted-analysis
  layer sends the model just the recipe summary and finding text, never rows.

## Key documents

- [Product research brief](docs/product-research-brief.md)
- [Public API and architecture](docs/public-api-and-architecture.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Schema discovery guide](docs/schema-discovery.md)
- [Implementation status](docs/implementation-status.md)
