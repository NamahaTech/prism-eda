# Security policy

## Supported versions

Prism EDA is in early alpha. Security fixes are applied to the latest released
version on PyPI; there are no long-term support branches yet.

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than in a public issue.

- Open a [private security advisory](https://github.com/NamahaTech/prism-eda/security/advisories/new)
  on GitHub, or
- email **khushalthanvi3@gmail.com** with `prism-eda security` in the subject.

Include the version, a description of the impact, and the smallest reproduction
you can manage. You can expect an acknowledgement within a few working days.

## What is in scope

Prism EDA reads data files you point it at and writes HTML and JSON reports, so
the interesting surfaces are:

- **Report rendering.** Reports embed values from your data. Templates are
  autoescaped, but a way to inject executable script into a generated report
  through crafted cell values, column names, or file paths is a vulnerability —
  report it.
- **File loading.** Anything where loading a malicious CSV, Parquet, Excel, or
  image file leads to code execution, a path escape, or an unexpected network
  request.
- **The privacy boundary.** The optional AI layer must never send raw cell
  values to a model provider. Only finding text and evidence *identifiers* cross
  that boundary. A path by which raw values reach a provider is a vulnerability
  even if no data is lost, because it breaks a documented guarantee. See
  [the privacy guide](docs/usage_docs/privacy.md).

## What is not in scope

- Reports are **not a trust boundary for the data they describe**. A report
  contains sample rows and column values by design, so treat a generated report
  as being as sensitive as the data it profiled before sharing it.
- Vulnerabilities in third-party dependencies should be reported upstream; tell
  us too if Prism EDA's usage makes an upstream issue exploitable here.
