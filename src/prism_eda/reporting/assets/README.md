# Packaged report assets

Everything here is inlined into generated reports at render time, so a report
stays a single offline file with no network requests.

## logo.png / favicon.png

- Source: the Prism EDA brand logo, trimmed to its alpha bounding box.
- Purpose: the report masthead (`logo.png`, displayed at 34px) and the browser
  tab icon (`favicon.png`).
- Sizes are deliberately small. Both are base64-embedded into **every** report,
  so they are stored at roughly 3x their display box rather than at archival
  resolution — a full-resolution logo would cost more than the rest of a small
  report put together. `docs/assets/prism-eda-logo.png` holds the large version
  for documentation; it is not shipped in the wheel.
- Update: regenerate both from the brand source with
  `python scripts/build_logo_assets.py <path-to-logo.png>`.

## cytoscape.min.js

- Version: 3.34.0
- License: MIT (see `LICENSE.cytoscape`)
- Source: https://unpkg.com/cytoscape@3.34.0/dist/cytoscape.min.js
- Purpose: powers the interactive ERD in schema-discovery HTML reports. The
  file is inlined into the report at render time so reports stay fully
  offline and self-contained.
- Update: bump `CYTOSCAPE_VERSION` in `scripts/fetch_cytoscape.py`, run
  `python scripts/fetch_cytoscape.py`, refresh `LICENSE.cytoscape` if the
  upstream license changed, and update this file.
