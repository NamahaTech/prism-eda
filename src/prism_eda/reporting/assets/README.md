# Vendored report assets

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
