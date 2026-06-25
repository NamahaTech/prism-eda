# Extending Prism

This guide is aimed at people *using* Prism, but Prism is open source and
contributions are welcome. This page is a short on-ramp; the deep references live
alongside the code.

## How a request flows

Understanding the execution path explains where everything lives:

```text
top-level convenience function   (api.py)
  → Dataset.load / existing Dataset          (dataset.py, catalog/loaders.py)
  → Dataset recipe method                    (dataset.py)
  → catalog + fingerprint refresh            (catalog/)
  → deterministic recipe computation         (analysis/<recipe>.py)
  → Evidence + Finding + Artifact + plan      (evidence/, artifacts.py, transformations/)
  → AnalysisResult                            (results.py)
  → optional JSON / self-contained HTML       (reporting/)
```

Two rules that shape the whole codebase:

- **Convenience functions never reimplement logic.** `pe.classification(...)`
  loads the source and delegates to `Dataset.classification(...)`. There is one
  implementation path.
- **The deterministic core has no heavy/optional dependencies.** It must not
  import Plotly, LangChain, LangGraph, or any model-provider SDK. Those live in
  extras and in the planned assisted-analysis layer.

## Package layout

| Module | Responsibility |
|--------|----------------|
| `api.py` | Top-level convenience functions; delegate to `Dataset` |
| `dataset.py` | The `Dataset` session object and recipe dispatch |
| `catalog/` | Loading, fingerprints, column catalog, keys, relationships |
| `analysis/` | Recipes that turn computations into evidence/findings/artifacts |
| `evidence/` | Provider-neutral `Evidence` and `Finding` contracts |
| `artifacts.py` | Structured report artifacts (e.g. the schema graph) |
| `transformations/` | Declarative, non-mutating recommendations |
| `reporting/` | The shared self-contained HTML renderer |
| `results.py` | The stable `AnalysisResult` and explicit exports |
| `privacy/` | Foundation for the 0.2 model-payload privacy layer |

## Invariants to preserve

Any change must keep these true (they're what make Prism trustworthy):

1. Public analysis never mutates caller DataFrames.
2. Nothing is written to disk until an explicit export method is called.
3. Findings cite stable evidence IDs.
4. Severity and confidence stay separate fields.
5. Sampling is deterministic, recorded, and visible in reports.
6. Thin evidence yields `insufficient_evidence` / `no_meaningful_structure` — never
   a forced conclusion.
7. Optional metric failures are recorded and the run continues; foundational
   failures abort it.
8. The core imports no Plotly / LangChain / provider SDKs.
9. Reports work without JavaScript, CDNs, or optional packages.
10. Inferred keys, anomalies, and relationships are **candidates** until confirmed.

And the product principle behind all of it: **optimize signal-to-noise, not
coverage.** A new finding must clear a real threshold. Adding noisy or
false-positive findings works against the entire reason Prism exists.

## Adding a recipe (sketch)

1. Write the deterministic computation; keep computation models in the owning
   domain package (usually `catalog/`).
2. Add an `analysis/<recipe>.py` adapter that produces evidence and findings.
3. Route it through `Dataset.analyze()` and add a typed `Dataset` method.
4. Add a top-level function in `api.py` that delegates to that method.
5. Export only the intended public names from `__init__.py`.
6. Extend the renderer through typed artifacts or result fields.
7. Add synthetic tests (with known pathologies), JSON tests, and HTML tests.
8. Update the docs, implementation status, architecture doc, and changelog **in
   the same change** — documentation is not deferred.

The full walkthrough, error policy, sampling policy, and report policy are in the
**[Maintainer guide](../maintainer-guide.md)**.

## Development setup

```bash
git clone https://github.com/NamahaTech/prism-eda.git
cd prism-eda
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"

ruff check .
ruff format --check .
mypy src/prism_eda
pytest --cov=prism_eda --cov-report=term-missing
python -m build
```

## Where to read next

- **[Maintainer guide](../maintainer-guide.md)** — engineering contracts and the
  recipe-authoring walkthrough.
- **[Public API & architecture](../public-api-and-architecture.md)** — the living
  public contract and module boundaries.
- **[Agent handoff (AGENTS.md)](../../AGENTS.md)** — the shortest safe path for a
  new contributor to continue the project.
- **[Implementation status](../implementation-status.md)** — what's done, what's
  next, and known technical debt.
- **[Product research brief](../product-research-brief.md)** — the why behind the
  product.

Questions and proposals are welcome on the
[issue tracker](https://github.com/NamahaTech/prism-eda/issues).
