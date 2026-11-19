"""Who a report says it is from.

Every renderer takes a ``branding`` argument and reads the product name, the
logo, the favicon, the decorative gradient, and the footer line from one
:class:`Branding` value. That makes "no product name anywhere" a single switch
instead of a hunt through templates, and lets an integrator put their own name
on a report without forking one.

Three ways to use it::

    result.to_html("report.html")                      # the library's own look
    result.to_html("report.html", branding=False)      # unbranded
    result.to_html(
        "report.html",
        branding=Branding(name="Acme Risk", logo="acme.png", footer="Acme, 2026"),
    )

and one to set it for a whole process::

    prism_eda.set_default_branding(False)
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

__all__ = [
    "Branding",
    "BrandingLike",
    "get_default_branding",
    "resolve_branding",
    "set_default_branding",
]

# A packaged asset is named rather than loaded when the default is built, so
# importing the library never reads image files and a broken install degrades
# to a text-only masthead instead of failing the import.
_PACKAGED_PREFIX = "package:"
_PACKAGED_LOGO = f"{_PACKAGED_PREFIX}assets/logo.png"
_PACKAGED_FAVICON = f"{_PACKAGED_PREFIX}assets/favicon.png"

ImageSource = str | Path | bytes | None


def _media_type(payload: bytes) -> str:
    """Name the image format from its leading bytes."""
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    if payload.startswith(b"\x00\x00\x01\x00"):
        return "image/x-icon"
    head = payload[:512].lstrip().lower()
    if head.startswith((b"<svg", b"<?xml")):
        return "image/svg+xml"
    return "application/octet-stream"


def _read_image(source: ImageSource) -> bytes | None:
    if source is None:
        return None
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, str) and source.startswith(_PACKAGED_PREFIX):
        asset = source[len(_PACKAGED_PREFIX) :]
        try:
            return resources.files("prism_eda.reporting").joinpath(asset).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            return None
    # A path the caller supplied is a request, not a default: a typo should be
    # reported rather than quietly producing a report with no logo.
    return Path(source).expanduser().read_bytes()


def _data_uri(payload: bytes | None) -> str | None:
    if not payload:
        return None
    encoded = b64encode(payload).decode("ascii")
    return f"data:{_media_type(payload)};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class Branding:
    """The identity shown on a rendered report.

    ``name`` is the product or organisation name in the masthead and the page
    title; ``name_accent`` is an optional second word rendered lighter beside
    it. ``tagline`` is the small line under the name. ``logo`` and ``favicon``
    are a file path or raw image bytes (PNG, JPEG, GIF, WebP, SVG, or ICO).
    ``footer`` is the closing line of the report. ``show_spectrum`` keeps or
    drops the decorative colour gradient.

    Every field is optional. A field left as ``None`` renders nothing in its
    place; :meth:`none` is simply all of them left empty.
    """

    name: str | None = None
    tagline: str | None = None
    logo: ImageSource = field(default=None, repr=False)
    favicon: ImageSource = field(default=None, repr=False)
    footer: str | None = None
    show_spectrum: bool = False
    name_accent: str | None = None

    @classmethod
    def prism(cls) -> Branding:
        """The library's own masthead, logo, gradient, and footer."""
        return cls(
            name="Prism",
            name_accent="EDA",
            tagline="Exploratory data analysis",
            logo=_PACKAGED_LOGO,
            favicon=_PACKAGED_FAVICON,
            footer="Generated locally by Prism EDA.",
            show_spectrum=True,
        )

    @classmethod
    def none(cls) -> Branding:
        """No product name, logo, favicon, gradient, or footer anywhere."""
        return cls()

    @property
    def display_name(self) -> str | None:
        """The full name as one string, for titles and document metadata."""
        parts = [part for part in (self.name, self.name_accent) if part]
        return " ".join(parts) if parts else None

    @property
    def title_suffix(self) -> str:
        """What follows the dataset name in a page title; empty when unnamed."""
        name = self.display_name
        return f" — {name}" if name else ""

    @property
    def has_masthead(self) -> bool:
        """Whether there is anything to show in the masthead's identity block."""
        return bool(self.name or self.name_accent or self.tagline or self.logo)

    def logo_bytes(self) -> bytes | None:
        """The logo image as raw bytes, or ``None`` when there is no logo."""
        return _read_image(self.logo)

    def favicon_bytes(self) -> bytes | None:
        """The favicon image as raw bytes, or ``None`` when there is none."""
        return _read_image(self.favicon)

    def logo_data_uri(self) -> str | None:
        """The logo as an inline ``data:`` URI for a self-contained HTML file."""
        return _data_uri(self.logo_bytes())

    def favicon_data_uri(self) -> str | None:
        """The favicon as an inline ``data:`` URI."""
        return _data_uri(self.favicon_bytes())


BrandingLike = Branding | bool | None

_default_branding: Branding = Branding.prism()


def get_default_branding() -> Branding:
    """The branding used when a render call does not name one."""
    return _default_branding


def set_default_branding(branding: BrandingLike) -> Branding:
    """Set the process-wide default branding and return the previous one.

    ``False`` switches branding off for every report rendered afterwards,
    ``True`` or ``None`` restores the library's own, and a :class:`Branding`
    installs a custom identity. The return value lets a caller put the previous
    default back.
    """
    global _default_branding
    previous = _default_branding
    if branding is None or branding is True:
        _default_branding = Branding.prism()
    elif branding is False:
        _default_branding = Branding.none()
    elif isinstance(branding, Branding):
        _default_branding = branding
    else:
        raise TypeError("branding must be a Branding, a bool, or None")
    return previous


def resolve_branding(branding: BrandingLike = None) -> Branding:
    """Turn a render call's ``branding`` argument into a :class:`Branding`.

    ``None`` means "whatever the process default is", ``False`` means none,
    ``True`` asks for the library's own regardless of the default, and a
    :class:`Branding` is used as given.
    """
    if branding is None:
        return _default_branding
    if branding is True:
        return Branding.prism()
    if branding is False:
        return Branding.none()
    if isinstance(branding, Branding):
        return branding
    raise TypeError("branding must be a Branding, a bool, or None")
