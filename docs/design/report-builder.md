# Design: Report Builder (dashboard authoring)

Status: **draft for review** — no code exists; this scopes the next major
product surface before any implementation planning.

## Problem

Prism's recipes answer *diagnostic* questions ("is this data sound?", "are
these tables related?", "which rows are suspect?"). Analysts then face a
second job: turning a data source into a *communicable* report or dashboard —
the thing they hand to a stakeholder. Today they leave Prism for that
(Metabase, notebooks, BI tools), and the insight thread breaks: the charts
they build elsewhere aren't connected to the evidence Prism gathered.

The opportunity is a report builder in the same spirit as the rest of Prism:
**signal over noise**. Not a free-form chart canvas — a builder that proposes
the report, because it already understands the schema, the column roles, and
the data quality caveats.

## Product shape (proposed)

A third layer on top of the existing two:

1. **Recipes** (exists): profile / schema / anomaly / classification.
2. **Evidence + catalog** (exists): typed, fingerprinted, render-ready.
3. **Report builder** (new): compose *views* over one or more datasets into a
   single polished HTML document (later: live-refreshable).

Differentiation vs. Metabase-style tools:

- **Proposal-first.** `suggest_report()` drafts a dashboard from what Prism
  already knows: measures (measure_candidate roles) sliced by low-cardinality
  dimensions (dimension_candidate + top_values coverage), time series where a
  timestamp_candidate exists, join paths from discovered relationships. The
  analyst curates instead of starting blank.
- **Caveats travel with charts.** A chart over a column with a high-missing or
  high-cardinality warning renders the caveat with the chart. No silently
  misleading visuals — this falls straight out of `ColumnCatalog.warnings`.
- **Evidence-linked.** Every view keeps `evidence_ids`, same as findings, so a
  dashboard number can be traced to how it was computed.

## Building on existing primitives

| Need | Existing primitive | Gap |
|---|---|---|
| Data access | `Dataset` / loaders (CSV, Parquet, Excel, folders) | none |
| What to chart | `ColumnCatalog` roles, `top_values`, statistics | selection heuristics |
| Joins across tables | `RelationshipCandidate` (schema discovery) | join execution |
| Aggregation spec | — | new declarative `ViewSpec` (measure, dimension, agg, filters) |
| Chart rendering | `reporting/charts.py` inline SVG (histogram, scatter, bars) | bar/line/pivot table views over aggregates |
| Page rendering | Jinja renderer, evidence-banked, self-contained HTML | new dashboard template + grid layout |
| Non-mutating recommendation pattern | `TransformationPlan` | reuse the shape for `ReportPlan` |

## Candidate architecture

```
dataset.suggest_report(audience="operations")   ->  ReportPlan (declarative)
ReportPlan.views: tuple[ViewSpec, ...]              # measure/dimension/agg/filter/chart_kind
ReportPlan.edit()                                    # analyst adds/removes/reorders views
build_report(dataset, plan).to_html("weekly.html")  # evaluates specs -> banks aggregates as evidence -> renders
```

Key properties, consistent with house rules:

- **Declarative + non-mutating**: a `ReportPlan` is data, reviewable and
  serializable (JSON round-trip), like `TransformationPlan`.
- **Evidence-banked rendering**: `build_report` evaluates each `ViewSpec` into
  an aggregate result *once*, stores it as evidence, and the template renders
  from evidence — no live DataFrame at render time (existing convention).
- **Privacy boundary unchanged**: aggregation happens locally; if the AI
  layer helps title/annotate views, it sees only specs and aggregate shapes,
  never raw cell values.
- **AI-optional**: suggestion heuristics are deterministic; the LLM (existing
  `assisted_analysis` layer) can *rank and caption* proposed views, abstaining
  rather than fabricating.

## MVP cut (proposal)

1. `ViewSpec` + `ReportPlan` models (frozen dataclasses, serialization).
2. Deterministic `suggest_report()` for a single table: top measures × top
   dimensions, a time series if a timestamp exists, missingness strip.
3. `build_report()` → self-contained dashboard HTML reusing report chrome
   (masthead, chips, print/no-JS discipline) with a responsive card grid.
4. Bar / line / single-stat / small table views only. No filters UI, no
   cross-table joins, no interactivity beyond hover — that's v2.

Deliberately out of MVP: multi-table joins (needs join execution + row
explosion safety), live data refresh, dashboard-level interactivity
(cross-filtering), export to PDF/PNG, scheduling.

## Open questions (decide before implementation planning)

1. **Where does curation happen?** Python API only (edit the plan in code) vs.
   a light in-report editor that emits an updated plan JSON. MVP leans Python
   API; the in-report editor is a real differentiator but a big lift.
2. **Is a dashboard a new goal/recipe** (`dataset.analyze("report")`) or a
   separate namespace (`prism_eda.reporting.build_report`)? Recipes imply
   findings/verdict semantics that don't quite fit.
3. **Aggregation engine**: pandas groupby is enough for MVP; do we need a
   sampling story for 10M+ row tables (consistent with existing sampling
   policy) from day one?
4. **How much AI in v1?** Deterministic-only suggestions are shippable;
   AI-ranked "what's interesting" (e.g. variance-based view scoring) may be
   the actual wow moment. Cost: prompt design + abstention discipline.
5. **Refreshability**: is "re-run the script, get updated HTML" enough for
   v1 (stateless, fits the library shape), or do we need a
   `plan.json` + CLI (`prism-eda render plan.json data/`) workflow?

## Rough sequencing after approval

Models → suggestion heuristics (+ tests over `examples/sample_data.py`) →
aggregate evaluation + evidence banking → dashboard template → docs page.
Two to three iterations of the usual size; the in-report editor and joins are
their own future design rounds.
