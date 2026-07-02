# Installation

## Requirements

- **Python 3.11 or newer.**
- A working `pip`. Prism's core dependencies (pandas, NumPy, PyArrow,
  scikit-learn, Jinja2) are installed automatically.

## Install from PyPI

```bash
pip install prism-eda
```

The distribution name is `prism-eda`; the import package is `prism_eda`:

```python
import prism_eda as pe
print(pe.__version__)
```

```text
0.1.0
```

## Optional extras

Prism keeps its core dependency set small and pushes heavier or optional
capabilities into extras. Install only what you need:

| Extra | Install | Adds |
|-------|---------|------|
| `plotly` | `pip install "prism-eda[plotly]"` | Interactive charts when exporting reports with `interactive=True` |
| `dev` | `pip install "prism-eda[dev]"` | `ruff`, `mypy`, `build`, `hatchling` for development |
| `test` | `pip install "prism-eda[test]"` | `pytest`, `pytest-cov` for running the test suite |

> Reports are fully functional **without** any extra — core charts are rendered
> as inline SVG and interactive pieces (like the schema ER diagram) embed a
> vendored library, so an HTML report has no CDN or network requirement and
> degrades gracefully without JavaScript. The `plotly` extra only upgrades
> charts to interactive versions when you explicitly ask for them.

## Install from source (for contributors)

```bash
git clone https://github.com/NamahaTech/prism-eda.git
cd prism-eda
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
```

This gives you an editable install plus the tooling used in CI. See
[Extending Prism](extending-prism.md) for the full contributor workflow.

## Verify your install

```python
import pandas as pd
import prism_eda as pe

result = pe.profile(pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, None, None, 4.0]}))
print(result.status)
print(result.summary)
```

```text
completed
Profiled 1 table(s), 4 rows, and 2 columns; found 1 prioritized issue(s).
```

If you see a status and a one-line summary, you're ready. Head to
[Getting started](getting-started.md).
