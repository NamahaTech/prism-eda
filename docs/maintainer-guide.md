# Maintainer Guide

This is the living engineering guide for Prism EDA. It explains how features are
expected to fit together and which contracts should remain stable as the library
grows.

## Execution path

A normal deterministic request follows this path:

```text
top-level convenience function
  -> Dataset.load / existing Dataset
  -> Dataset recipe method
  -> catalog and fingerprint refresh
  -> deterministic recipe computation
  -> Evidence + Finding + Artifact + warnings/failures
  -> AnalysisResult
  -> optional JSON or self-contained HTML export
```

Convenience functions must not create independent implementations. For example,
`prism_eda.discover_schema(...)` loads the source and delegates to
`Dataset.discover_schema(...)`.

## Core contracts

### Dataset

`Dataset` owns named table references, source metadata, and a cached catalog. It
does not claim ownership of caller DataFrames and must not mutate them.

### Catalog

The catalog is descriptive and deterministic. It contains table/column metadata,
semantic-role candidates, robust summaries, and bounded-cost fingerprints. A
fingerprint detects likely dataset changes; it is not a cryptographic commitment
to every row because row values are deterministically sampled.

### Evidence

Evidence is the numeric or structural basis for conclusions. Its ID is derived
from scope, method, values, assumptions, and metadata, so the same deterministic
computation produces the same ID.

When an algorithm changes materially, change its method version, for example
`typed_inclusion_dependency_v1` to `v2`. This prevents old and new evidence from
appearing equivalent.

### Findings

Findings are human-facing conclusions over evidence. They always cite evidence
IDs. They may recommend review or transformation but do not perform mutations.

### Artifacts

Artifacts are structured, serializable visual or tabular products. Renderers
consume artifact data; analysis recipes should not generate raw HTML fragments.
The first artifact is `schema_graph`.

### Result status

- `completed`: required evidence was produced without notable caveats.
- `completed_with_warnings`: useful result with sampling or recoverable caveats.
- `insufficient_evidence`: available data cannot support the requested analysis.
- `no_meaningful_structure`: analysis ran but found no stable signal above its
  thresholds.
- `failed`: a foundational stage failed.

## Adding a recipe

1. Define the deterministic computation and public options.
2. Keep computation models in the owning domain package, usually `catalog/` or a
   future specialized package.
3. Add an `analysis/<recipe>.py` adapter that creates evidence and findings.
4. Route it through `Dataset.analyze()` and add a typed `Dataset` convenience
   method.
5. Add a top-level function in `api.py` that delegates to the dataset method.
6. Export only the intended public names from `__init__.py`.
7. Extend the shared renderer through typed artifacts or result fields.
8. Add synthetic tests, JSON tests, HTML tests, and browser QA.
9. Update the feature guide, implementation status, architecture document, and
   changelog in the same change.

## Error policy

Loading errors, invalid duplicate column names, catalog failures, and failures in
required recipe stages are foundational. Raise a Prism EDA exception and stop.

Individual optional metrics should create an `AnalysisFailure` with
`recoverable=True`, emit `MetricFailed`, and allow the remaining recipe to run.
Do not catch broad exceptions without recording enough scope and method context
for diagnosis.

Callbacks are observers. Their exceptions are isolated and never stop analysis.

## Sampling policy

`quick`, `standard`, and `deep` describe compute depth, not report verbosity.
Automatic sampling must be deterministic and create `SamplingRecord` entries.
The report must make decision-relevant sampling visible.

Users may disable sampling, but an algorithm may still reject unsafe or
inapplicable execution. Disabling sampling is not a promise of unlimited memory.

## Report policy

- Use semantic HTML, embedded CSS, inline SVG, and minimal optional JavaScript.
- No CDN or network requirement.
- Preserve print and no-JavaScript usefulness.
- Wide data tables and graphs may scroll inside their own containers; the page
  itself must not overflow at mobile widths.
- Candidate and assumption language must remain visible near inferred outputs.

## Release and packaging

`pyproject.toml` is canonical. `requirements.txt` is only an editable-install
compatibility shim. The wheel must include templates, `py.typed`, and the license.

Before a release or handoff, run all commands listed in `AGENTS.md`, inspect the
wheel contents, and install the wheel into a separate target or environment.
