# Exporting reports

An `AnalysisResult` carries everything needed to produce a report or feed a
downstream tool. **Nothing is written to disk until you call an export method.**

```python
import prism_eda as pe
from examples.sample_data import load_sample

result = pe.load(load_sample()).classification("churned", table="customers")
```

## HTML reports

`to_html(path)` writes a **self-contained** HTML report and returns the path:

```python
path = result.to_html("churn-readiness.html")
print(path)        # churn-readiness.html (as a pathlib.Path)
```

The report has **no CDN, network, or JavaScript requirement** — charts are inline
SVG and everything is embedded, so it opens anywhere, prints cleanly, and works
offline. It leads with the decision-first summary, then the prioritized findings,
their evidence, and any artifacts (metric tables, the schema graph).

### Render to a string instead of a file

```python
html = result.render_html()
print(html[:15])      # <!doctype html>
```

Useful for embedding the report in a web app, emailing it, or serving it from
memory without touching disk.

### Interactive charts (optional)

If the [`plotly` extra](installation.md#optional-extras) is installed, request
interactive charts:

```python
result.to_html("report.html", interactive=True)
```

If you ask for `interactive=True` but Plotly isn't installed, export still
succeeds with static SVG charts and records a note in the report — a missing
optional renderer never strips the visual evidence out of your report.

## JSON output

`to_json(path)` writes the complete machine-readable result:

```python
path = result.to_json("churn-readiness.json", indent=2)
```

The JSON contains every field of the result — findings, evidence, catalog,
artifacts, transformation plan, warnings, sampling, and metadata — with keys
sorted for stable diffs.

## In-memory dict

To work with the result programmatically (store it, post it to an API, compare
runs), get a plain dict instead of writing a file:

```python
payload = result.to_dict()
# Pydantic-style alias, identical output (mode="json" is the only supported mode):
payload = result.model_dump(mode="json")

print(payload["status"])
print(payload["summary"])
print(len(payload["findings"]), "findings")
```

```text
completed
customers.churned: not ready to model. Top issue — Potential target leakage: exit_survey_sent. 4 prioritized finding(s) (1 critical, 1 high, 2 medium).
4 findings
```

## Choosing an export

| You want to… | Use |
|--------------|-----|
| Share a human-readable report | `to_html("report.html")` |
| Embed the report in your own UI / email | `render_html()` |
| Persist or diff the full result | `to_json("result.json")` |
| Inspect or post-process in Python | `to_dict()` / `model_dump()` |

Because results are deterministic, two runs over unchanged data produce identical
JSON (including evidence IDs) — handy for snapshot tests and change detection in
your own pipelines.
