"""Fetch the pinned Cytoscape.js dist into the vendored assets directory.

Run from the repository root whenever the pinned version changes:

    python scripts/fetch_cytoscape.py

The dist file is committed so builds and installs stay fully offline.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CYTOSCAPE_VERSION = "3.34.0"
DIST_URL = (
    "https://registry.npmjs.org/cytoscape/-/cytoscape-"
    f"{CYTOSCAPE_VERSION}.tgz"
)
UNPKG_URL = (
    f"https://unpkg.com/cytoscape@{CYTOSCAPE_VERSION}/dist/cytoscape.min.js"
)
ASSETS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "prism_eda"
    / "reporting"
    / "assets"
)
MIN_EXPECTED_BYTES = 200_000
MAX_EXPECTED_BYTES = 2_000_000


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UNPKG_URL, timeout=60) as response:
        payload = response.read()
    if not (MIN_EXPECTED_BYTES <= len(payload) <= MAX_EXPECTED_BYTES):
        print(
            f"Unexpected dist size {len(payload):,} bytes — refusing to write.",
            file=sys.stderr,
        )
        return 1
    text = payload.decode("utf-8")
    if "cytoscape" not in text[:2000].lower():
        print("Payload does not look like Cytoscape.js.", file=sys.stderr)
        return 1
    target = ASSETS_DIR / "cytoscape.min.js"
    target.write_bytes(payload)
    print(f"Wrote {target} ({len(payload):,} bytes, v{CYTOSCAPE_VERSION}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
