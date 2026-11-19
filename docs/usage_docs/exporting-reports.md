# Exporting reports

An `AnalysisResult` carries everything needed to produce a report or feed a
downstream tool. **Nothing is written to disk until you call an export method.**

```python
import prism_eda as pe
from examples.sample_data import load_sample

result = pe.load(load_sample()).classification("churned", table="customers")
```

## HTML reports

`to_html(path)` writes a **self-contained** HTML report and returns the path.
Dataset fingerprints remain available in the result catalog and JSON export for
reproducibility, but are not shown in the reader-facing HTML report:

```python
path = result.to_html("churn-readiness.html")
print(path)        # churn-readiness.html (as a pathlib.Path)
```

The report has **no CDN or network requirement** — charts are inline SVG and
everything is embedded, so it opens anywhere, prints cleanly, and works fully
offline. Schema-discovery reports additionally embed a vendored copy of
[Cytoscape.js](https://js.cytoscape.org/) (MIT) to power the interactive ER
diagram; with JavaScript disabled (or if the embedded library fails to load)
the report degrades to a static SVG diagram with a visible notice. It leads
with the decision-first summary, then the prioritized findings, their
supporting charts and artifacts (metric tables, the schema graph).
Raw evidence IDs stay in `to_json()`, `to_dict()`, and the Python result object
for auditing and downstream integrations.

### Render to a string instead of a file

```python
html = result.render_html()
print(html[:15])      # <!doctype html>
```

Useful for embedding the report in a web app, emailing it, or serving it from
memory without touching disk.

### Comparison reports

`compare_datasets()` returns a `ComparisonResult`, which exports the same way:

```python
comparison = pe.compare_datasets({"orders": last_month}, {"orders": this_month})
comparison.to_html("orders-comparison.html")
```

The report names both sides, shows their row and column counts side by side,
and gives each shared column a card: both distributions drawn on one axis, above
a table of the same statistics for each side. When both sides carry the same
table name they are labelled *Base* and *Comparison*. The comparison places the
two side by side; it runs no drift test and raises no findings.

## PDF reports

PDF export needs the `pdf` extra (`pip install "prism-eda[pdf]"`, which adds
reportlab and matplotlib):

```python
path = result.to_pdf("report.pdf")
data = result.render_pdf()            # bytes, nothing written to disk
```

The PDF has a cover, a contents page with page numbers and a bookmark outline,
a one-page summary, the findings, and one section per table. Charts are static
images; every figure a chart shows is also in a table or in the JSON export.

| Argument | Effect |
|----------|--------|
| `title` | Replaces the cover and page-header title |
| `subtitle` | Replaces the line under the title (by default the sources or table name) |
| `branding` | As for HTML: `False`, a `Branding`, or `None` for the process default |
| `detailed_tables` | How many tables, largest first, get charts (default 80). The rest keep their column statistics. Lower it to keep a report over hundreds of tables small |

```python
result.to_pdf(
    "review.pdf",
    title="Core banking data review",
    subtitle="Production · March extract",
    branding=False,
    detailed_tables=40,
)
```

The full layout (inventory, relationships, links, per-table charts) is written
for [exploration](exploration.md) results; a baseline profile gets the summary,
findings and per-table sections. Other recipes render their summary, findings
and per-table column statistics; their recipe-specific charts remain in the
HTML report.

Text is set in DejaVu Sans, which covers Latin, Greek and Cyrillic. Values in
other scripts appear as placeholder boxes in the PDF; the HTML and JSON outputs
are unaffected. Without the extra installed, `to_pdf()` raises
`PdfSupportMissing` with the install command.

## White-labelling / removing branding

By default a report carries the library's own name, logo, favicon, colour
gradient, and footer line. Every `render_html()`, `to_html()`, `render_pdf()`
and `to_pdf()` takes a keyword-only `branding` argument that changes that.

**Remove it entirely** — no product name, logo, favicon, gradient, or footer
anywhere in the document:

```python
result.to_html("report.html", branding=False)
html = result.render_html(branding=False)
```

**Put your own identity on it:**

```python
from prism_eda import Branding

brand = Branding(
    name="Acme",
    name_accent="Risk",              # optional second word, rendered lighter
    tagline="Data review",
    logo="assets/acme.png",          # a path, or the image bytes
    favicon="assets/acme-icon.png",
    footer="Prepared by Acme Risk. Confidential.",
    show_spectrum=False,             # keep or drop the decorative gradient
)
result.to_html("report.html", branding=brand)
```

Every field is optional and a field left out renders nothing in its place.
Logos and favicons may be PNG, JPEG, GIF, WebP, SVG, or ICO, and are embedded
in the file so the report stays self-contained. A logo path that does not exist
raises `FileNotFoundError` rather than quietly producing a report without one.

**Set it once for the whole process**, which is the usual choice when the
library is embedded in another product:

```python
import prism_eda as pe

pe.set_default_branding(False)        # or a Branding(...)
...
pe.set_default_branding(None)         # back to the library's own
```

`set_default_branding` returns the previous default so you can restore it. A
`branding=` argument on an individual call always wins over the process
default; `branding=True` asks for the library's own look regardless.

In a PDF, branding decides the logo and name on the cover, the footer line on
every page, and the author recorded in the document properties. With
`branding=False` none of the three is present. A logo that cannot be decoded is
left out of the PDF rather than failing the export.

`render_html()` and `to_html()` also take `title=`, which replaces the report's
heading.

The argument is available on `AnalysisResult`, on the `ComparisonResult` from
`compare_datasets()`, and on the feature-plan report (`run.report`). Branding
only changes identity: the sections, charts, tables, and findings of a report
are the same with or without it. Text inside findings and evidence never names
the library, so the JSON output needs no equivalent switch.

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
| Send a report that will be printed or attached | `to_pdf("report.pdf")` |
| Embed the report in your own UI / email | `render_html()` |
| Persist or diff the full result | `to_json("result.json")` |
| Inspect or post-process in Python | `to_dict()` / `model_dump()` |

Because results are deterministic, two runs over unchanged data produce identical
JSON (including evidence IDs) — handy for snapshot tests and change detection in
your own pipelines.
