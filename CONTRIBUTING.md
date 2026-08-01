# Contributing to Prism EDA

Thanks for taking a look. This file covers the mechanics; the design rules that
matter most live in [`AGENTS.md`](AGENTS.md), which is worth reading before your
first change.

## Setting up

```bash
git clone https://github.com/NamahaTech/prism-eda.git
cd prism-eda
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test,dev]'
```

## The gate

Every change has to pass the same checks CI runs:

```bash
ruff check .
ruff format --check .
mypy src/prism_eda
pytest --cov=prism_eda --cov-report=term-missing
```

For anything that touches packaging or the report, also build and install the
wheel into a clean environment — that is the only way to catch a template or
asset that stopped shipping.

## What a good change looks like

Prism EDA's one differentiator is **signal over noise**: an analyst should be
able to read a report and come away with an opinion, not a wall of statistics.
That shapes what gets accepted.

- **A new check must earn its place.** Before adding one, decide which channel
  it belongs to. If it is not something an analyst would *fix*, it is an
  observation (an alert), not a data-quality issue. A finding that fires on
  healthy data is worse than no finding, because it teaches people to skim.
- **Findings cite evidence.** Every claim carries the `evidence_ids` it rests
  on, and evidence records the method that produced it. No number appears in a
  report without a traceable computation behind it.
- **Say what you did not do.** If a cap, sample, or threshold means some of the
  data was not examined, that has to reach the user as a warning and appear in
  the report. A truncated result that looks complete is a bug.
- **Never state more confidence than you have.** Inferred keys, relationships,
  anomalies, and distribution fits are candidates. Where nothing fits, abstain
  rather than reporting the least-bad answer.
- **Do not mutate the caller's data.** Analysis never modifies input
  DataFrames and writes no files until an explicit export call.

The full list of invariants — including the dependency boundaries and the AI
privacy rule — is in [`AGENTS.md`](AGENTS.md).

## Tests

- Use synthetic fixtures with a *known* pathology, and assert on the specific
  thing you planted.
- Assert evidence lineage, not only prose.
- Include a negative case: clean data must stay quiet. Most noisy checks would
  have been caught by one.
- When you add a report section, add it to `reporting/sections.py` in the same
  change — `tests/test_report_sections.py` fails if the navigation and the page
  disagree.

## Documentation

Documentation ships in the same change as the code, not in a later cleanup:
update the relevant guide in `docs/usage_docs/`, plus `CHANGELOG.md` under
`## [Unreleased]`. Examples in the usage docs are written against
`examples/sample_data.py` and their printed output is expected to match a real
run — please verify yours by running it.

## Pull requests

Keep them focused, explain the reasoning rather than restating the diff, and
mention anything you deliberately left out. If you are unsure whether an idea
fits, open an issue first — that is cheaper than building the wrong thing.
