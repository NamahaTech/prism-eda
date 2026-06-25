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
  classification
- Optional AI-assisted investigation via the `ai-gemini` extra
  (`prism_eda.assisted_analysis`): an LLM plans over the deterministic tools only
- Planned next recipes: regression, time-series, clustering

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
- `catalog/`: loading, fingerprints, column catalog, keys, and relationships.
- `analysis/`: task recipes that turn deterministic computations into evidence,
  findings, artifacts, warnings, and statuses.
- `evidence/`: provider-neutral evidence and finding contracts.
- `artifacts.py`: structured report artifacts such as schema graphs.
- `reporting/`: the shared self-contained renderer.
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

## Important limitations

- Pandas is the only in-memory backend.
- CSV inputs are currently loaded eagerly; chunked execution is still planned.
- Schema discovery considers at most 12 likely columns per table and key width is
  capped at three.
- Non-string column names are profiled but skipped by schema discovery with a
  warning.
- Automated functional-dependency discovery and self-referential relationships
  are not implemented.
- Persistent AI sessions are not implemented.

## Key documents

- [Product research brief](docs/product-research-brief.md)
- [Public API and architecture](docs/public-api-and-architecture.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Schema discovery guide](docs/schema-discovery.md)
- [Implementation status](docs/implementation-status.md)
