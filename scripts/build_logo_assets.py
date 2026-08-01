"""Regenerate the packaged logo assets from the brand source image.

    python scripts/build_logo_assets.py "/path/to/Prism eda logo.png"

The report embeds these as base64 in every generated file, so they are stored at
roughly 3x their display box rather than at archival resolution: a
full-resolution logo costs more bytes than the rest of a small report. The large
documentation copy is written to ``docs/assets/`` and is not packaged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_ASSETS = REPO_ROOT / "src" / "prism_eda" / "reporting" / "assets"
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"

#: (destination, longest edge in pixels). The masthead renders at 34px and the
#: favicon at 16-32px, so these are already generous for high-density displays.
TARGETS = (
    (PACKAGED_ASSETS / "logo.png", 96),
    (PACKAGED_ASSETS / "favicon.png", 32),
    (DOCS_ASSETS / "prism-eda-logo.png", 512),
)


def build(source: Path) -> None:
    original = Image.open(source).convert("RGBA")
    # Trim transparent margin so the mark fills its box: padding baked into the
    # source would otherwise shrink the logo inside an already small display.
    bounds = original.getbbox()
    trimmed = original.crop(bounds) if bounds else original

    for destination, longest_edge in TARGETS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = trimmed.copy()
        image.thumbnail((longest_edge, longest_edge), Image.LANCZOS)
        image.save(destination, "PNG", optimize=True)
        size = destination.stat().st_size
        print(f"{destination.relative_to(REPO_ROOT)}  {image.size}  {size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the full-size logo PNG")
    arguments = parser.parse_args()
    if not arguments.source.is_file():
        raise SystemExit(f"No such file: {arguments.source}")
    build(arguments.source)


if __name__ == "__main__":
    main()
