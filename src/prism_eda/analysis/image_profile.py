"""Deterministic image dataset profile."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import random
import warnings as runtime_warnings
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile, ImageStat, UnidentifiedImageError

from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import (
    ColumnCatalog,
    DatasetCatalog,
    SourceInfo,
    TableCatalog,
)
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import Evidence, EvidenceScope, Finding, sort_findings
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

_IMAGE_BUDGETS = {
    AnalysisMode.QUICK: 2_000,
    AnalysisMode.STANDARD: 10_000,
    AnalysisMode.DEEP: 50_000,
}
_LOW_SHARPNESS_THRESHOLD = 0.0008
_LOW_CONTRAST_THRESHOLD = 0.035
_DARK_THRESHOLD = 0.08
_BRIGHT_THRESHOLD = 0.92
_LOW_ENTROPY_THRESHOLD = 2.0
_MAX_NEAR_DUPLICATE_PAIRS = 50
_NEAR_DUPLICATE_FULL_SCAN_LIMIT = 2_000
_NEAR_DUPLICATE_WINDOW = 40
#: Thumbnails are attached only to files the report actually shows, so a large
#: scan never holds one base64 payload per image in memory.
_MAX_THUMBNAILS_PER_GROUP = 12
#: EXIF orientation 1 means "as stored"; every other value asks the loader to
#: rotate or mirror, and 5-8 additionally transpose width and height.
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})
#: Extensions that legitimately map to a different Pillow format name.
_EXTENSION_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: str
    relative_path: str
    label: str | None
    split: str | None
    file_size_bytes: int
    sha256: str
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    format: str | None
    mode: str
    frame_count: int
    has_exif: bool
    exif_tag_count: int
    orientation: int | None
    brightness: float
    contrast: float
    sharpness: float
    entropy: float
    average_hash: str
    difference_hash: str
    # Loader traps: conditions that decode cleanly but silently change the
    # pixels, the shape, or the channel count a training pipeline receives.
    extension_format: str | None
    is_grayscale_rgb: bool
    has_alpha: bool
    alpha_is_used: bool
    is_truncated: bool
    has_corrupt_metadata: bool

    @property
    def extension_mismatch(self) -> bool:
        return (
            self.extension_format is not None
            and self.format is not None
            and self.extension_format != self.format
        )

    @property
    def rotates_on_load(self) -> bool:
        return self.orientation is not None and self.orientation != 1

    @property
    def transposes_on_load(self) -> bool:
        return self.orientation in _TRANSPOSING_ORIENTATIONS


@dataclass(frozen=True, slots=True)
class InvalidImageRecord:
    path: str
    relative_path: str
    file_size_bytes: int | None
    error: str


@contextmanager
def _truncation_tolerance() -> Iterator[None]:
    """Temporarily let Pillow decode a truncated file instead of raising."""
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _row_budget(mode: AnalysisMode | str) -> int:
    return _IMAGE_BUDGETS[AnalysisMode(mode)]


def _relative_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _sample_paths(
    paths: Sequence[Path],
    *,
    config: AnalysisConfig,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> tuple[Path, ...]:
    budget = _row_budget(config.mode)
    if config.sampling == "disabled" or len(paths) <= budget:
        return tuple(paths)
    rng = random.Random(config.random_seed)
    sampled = sorted(rng.sample(list(paths), budget), key=lambda item: item.as_posix())
    warnings.append(
        AnalysisWarning(
            code="sampled_image_profile",
            message=(
                f"{len(paths):,} image paths were discovered; image profiling was "
                f"run on a deterministic {budget:,}-file sample."
            ),
        )
    )
    sampling.append(
        SamplingRecord(
            operation="image_profile",
            source_rows=len(paths),
            sampled_rows=budget,
            strategy="deterministic_path_sample",
            seed=config.random_seed,
            reason="image_count_exceeds_mode_budget",
            limitations=(
                "Rare corrupt images, labels, formats, or near-duplicates may be "
                "absent from the sample.",
            ),
        )
    )
    return tuple(sampled)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_hash(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    return _hex_hash(pixels >= float(pixels.mean()))


def _difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float32)
    return _hex_hash(pixels[:, :-1] > pixels[:, 1:])


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _image_metrics(image: Image.Image) -> tuple[float, float, float, float]:
    gray = image.convert("L").resize((128, 128), Image.Resampling.LANCZOS)
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0] / 255)
    contrast = float(stat.stddev[0] / 255)
    array = np.asarray(gray, dtype=np.float32)
    horizontal = np.diff(array, axis=1)
    vertical = np.diff(array, axis=0)
    sharpness = float(
        (np.mean(horizontal * horizontal) + np.mean(vertical * vertical)) / (255**2)
    )
    entropy = float(gray.entropy())
    return brightness, contrast, sharpness, entropy


def _is_grayscale_rgb(image: Image.Image) -> bool:
    """True when a colour-mode image actually carries identical R, G, B planes.

    Compared on a downscale for speed. Resampling filters each channel
    independently, so genuinely grey pixels stay exactly equal and a real colour
    image effectively never collapses to equal channels by chance.
    """
    if image.mode not in {"RGB", "RGBA"}:
        return False
    small = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    array = np.asarray(small, dtype=np.uint8)
    return bool(
        np.array_equal(array[:, :, 0], array[:, :, 1])
        and np.array_equal(array[:, :, 1], array[:, :, 2])
    )


def _alpha_state(image: Image.Image) -> tuple[bool, bool]:
    """(has_alpha, alpha_is_actually_used) for the decoded image."""
    has_alpha = image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
    if not has_alpha:
        return False, False
    try:
        alpha = image.convert("RGBA").getchannel("A")
    except ValueError:
        return True, False
    return True, int(ImageStat.Stat(alpha).extrema[0][0]) < 255


def _extension_format(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if not suffix:
        return None
    return _EXTENSION_FORMATS.get(suffix, suffix[1:].upper())


def _open_decoded(path: Path) -> tuple[Image.Image, bool]:
    """Fully decode an image, returning it with whether it was truncated.

    A truncated file still decodes far enough to be worth profiling, and "this
    file is short" is itself a finding — so it is retried permissively rather
    than discarded as unreadable. Raises for anything genuinely undecodable.
    """

    def decode(*, tolerant: bool) -> Image.Image:
        image = Image.open(path)
        try:
            if tolerant:
                with _truncation_tolerance():
                    image.load()
            else:
                image.load()
        except BaseException:
            # Pillow holds the file handle open until load() succeeds, so a
            # failed decode must not leak it.
            image.close()
            raise
        return image

    try:
        with Image.open(path) as verifier:
            verifier.verify()
        return decode(tolerant=False), False
    except OSError as error:
        if "truncated" not in str(error).lower():
            raise
        return decode(tolerant=True), True


def _scan_image(
    path: Path,
    *,
    root: Path | None,
    labels: Mapping[str, str | None],
    splits: Mapping[str, str | None],
) -> ImageRecord | InvalidImageRecord:
    relative = _relative_path(path, root)
    size = path.stat().st_size if path.exists() else None
    key = path.as_posix()
    try:
        with runtime_warnings.catch_warnings(record=True) as caught:
            runtime_warnings.simplefilter("always")
            image, truncated = _open_decoded(path)
            # Pillow warns (rather than raises) on recoverable defects such as
            # corrupt EXIF blocks. Those are traps to report, not decode
            # failures — treating them as failures would call a perfectly good
            # JPEG unreadable.
            corrupt_metadata = any(
                "exif" in str(item.message).casefold()
                or "metadata" in str(item.message).casefold()
                for item in caught
            )
        with image:
            width, height = image.size
            exif = image.getexif()
            orientation = exif.get(274)
            brightness, contrast, sharpness, entropy = _image_metrics(image)
            has_alpha, alpha_is_used = _alpha_state(image)
            return ImageRecord(
                path=key,
                relative_path=relative,
                label=labels.get(key),
                split=splits.get(key),
                file_size_bytes=int(size or 0),
                sha256=_file_sha256(path),
                width=int(width),
                height=int(height),
                aspect_ratio=float(width / height) if height else 0.0,
                megapixels=float((width * height) / 1_000_000),
                format=image.format,
                mode=image.mode,
                frame_count=int(getattr(image, "n_frames", 1)),
                has_exif=bool(exif),
                exif_tag_count=len(exif),
                orientation=int(orientation) if isinstance(orientation, int) else None,
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                entropy=entropy,
                average_hash=_average_hash(image),
                difference_hash=_difference_hash(image),
                extension_format=_extension_format(path),
                is_grayscale_rgb=_is_grayscale_rgb(image),
                has_alpha=has_alpha,
                alpha_is_used=alpha_is_used,
                is_truncated=truncated,
                has_corrupt_metadata=corrupt_metadata,
            )
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        return InvalidImageRecord(
            path=key,
            relative_path=relative,
            file_size_bytes=size,
            error=f"{type(error).__name__}: {error}",
        )


def _thumbnail(path: Path, *, size: int) -> str | None:
    """A base64 PNG data URI, so the report stays a single portable file."""
    try:
        with _truncation_tolerance(), Image.open(path) as image:
            image.load()
            thumb = image.convert("RGB")
            thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG", optimize=True)
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _counter_rows(counter: Counter[Any], *, missing_label: str) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows: list[dict[str, Any]] = []
    for value, count in counter.most_common():
        rows.append(
            {
                "value": value if value is not None else missing_label,
                "count": count,
                "rate": count / total if total else 0.0,
            }
        )
    return rows


def _robust_outliers(
    records: Sequence[ImageRecord], field: str
) -> list[dict[str, Any]]:
    values = np.asarray(
        [getattr(record, field) for record in records], dtype=np.float64
    )
    if values.size < 5:
        return []
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    if mad > 0:
        scores = 0.6745 * (values - median) / mad
    else:
        # Image datasets usually agree on a size, which drives both the MAD and
        # the IQR to zero — exactly when the one 320x64 panorama in a folder of
        # 64x64 thumbnails matters most. Fall back to the mean absolute
        # deviation (Iglewicz & Hoaglin's MAD=0 recommendation) so a uniform
        # dataset makes its few odd files stand out rather than hiding them.
        mean_deviation = float(np.mean(deviations))
        if mean_deviation <= 0:
            return []
        scores = (values - median) / (1.253314 * mean_deviation)
    outliers: list[dict[str, Any]] = []
    for record, value, score in zip(records, values, scores, strict=True):
        if abs(float(score)) >= 4.0:
            outliers.append(
                {
                    "path": record.relative_path,
                    "value": float(value),
                    "robust_z": float(score),
                }
            )
    return sorted(outliers, key=lambda item: abs(item["robust_z"]), reverse=True)[:20]


def _distribution_payload(
    records: Sequence[ImageRecord], field: str, *, column: str
) -> dict[str, Any]:
    """Histogram + box payload in the shape the shared chart filter already reads."""
    values = np.asarray(
        [getattr(record, field) for record in records], dtype=np.float64
    )
    if values.size == 0:
        return {"column": column}
    bins = int(min(24, max(6, math.ceil(math.sqrt(values.size)))))
    counts, edges = np.histogram(values, bins=bins)
    q1, median, q3 = (float(item) for item in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    return {
        "column": column,
        "histogram": {
            "counts": [int(count) for count in counts],
            "edges": [float(edge) for edge in edges],
        },
        "box": {
            "min": float(values.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(values.max()),
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr,
        },
    }


def _exact_duplicate_groups(records: Sequence[ImageRecord]) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)
    return [
        {
            "sha256": digest,
            "count": len(group),
            "paths": [record.relative_path for record in group],
        }
        for digest, group in sorted(groups.items())
        if len(group) > 1
    ]


def _near_duplicate_pairs(
    records: Sequence[ImageRecord], *, threshold: int
) -> tuple[list[dict[str, Any]], str]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(left: ImageRecord, right: ImageRecord) -> None:
        if len(pairs) >= _MAX_NEAR_DUPLICATE_PAIRS:
            return
        left_path, right_path = sorted((left.relative_path, right.relative_path))
        key = (left_path, right_path)
        if key in seen:
            return
        seen.add(key)
        if left.sha256 == right.sha256:
            return
        distance = min(
            _hamming(left.average_hash, right.average_hash),
            _hamming(left.difference_hash, right.difference_hash),
        )
        if distance <= threshold:
            pairs.append(
                {
                    "left": left.relative_path,
                    "right": right.relative_path,
                    "hash_distance": distance,
                }
            )

    if len(records) <= _NEAR_DUPLICATE_FULL_SCAN_LIMIT:
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                add_pair(left, right)
        method = "full_pairwise"
    else:
        for key_name in ("average_hash", "difference_hash"):
            ordered = sorted(records, key=lambda item: getattr(item, key_name))
            for index, left in enumerate(ordered):
                window = ordered[index + 1 : index + 1 + _NEAR_DUPLICATE_WINDOW]
                for right in window:
                    add_pair(left, right)
        method = "deterministic_hash_window"

    return (
        sorted(pairs, key=lambda item: item["hash_distance"])[
            :_MAX_NEAR_DUPLICATE_PAIRS
        ],
        method,
    )


def _quality_flags(records: Sequence[ImageRecord]) -> dict[str, list[dict[str, Any]]]:
    flags: dict[str, list[dict[str, Any]]] = {
        "dark": [],
        "bright": [],
        "low_contrast": [],
        "low_sharpness": [],
        "low_entropy": [],
    }
    for record in records:
        if record.brightness <= _DARK_THRESHOLD:
            flags["dark"].append(
                {"path": record.relative_path, "value": record.brightness}
            )
        if record.brightness >= _BRIGHT_THRESHOLD:
            flags["bright"].append(
                {"path": record.relative_path, "value": record.brightness}
            )
        if record.contrast <= _LOW_CONTRAST_THRESHOLD:
            flags["low_contrast"].append(
                {"path": record.relative_path, "value": record.contrast}
            )
        if record.sharpness <= _LOW_SHARPNESS_THRESHOLD:
            flags["low_sharpness"].append(
                {"path": record.relative_path, "value": record.sharpness}
            )
        if record.entropy <= _LOW_ENTROPY_THRESHOLD:
            flags["low_entropy"].append(
                {"path": record.relative_path, "value": record.entropy}
            )
    return {key: value[:20] for key, value in flags.items() if value}


def _leakage(
    records: Sequence[ImageRecord],
    exact_groups: Sequence[Mapping[str, Any]],
    near_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Duplicates that straddle a split or a label boundary.

    A duplicate *within* one split is wasted compute. The same image in both
    train and test silently inflates every score the model reports, and the same
    image under two labels means one of the annotations is wrong. Those are
    different problems with different fixes, so they are separated here.
    """
    by_relative = {record.relative_path: record for record in records}

    def cohorts(paths: Sequence[str]) -> tuple[set[str], set[str]]:
        found = [by_relative[path] for path in paths if path in by_relative]
        return (
            {record.split for record in found if record.split},
            {record.label for record in found if record.label},
        )

    cross_split: list[dict[str, Any]] = []
    cross_label: list[dict[str, Any]] = []

    def consider(paths: Sequence[str], *, kind: str, distance: int) -> None:
        splits, labels = cohorts(paths)
        entry = {"kind": kind, "paths": list(paths), "hash_distance": distance}
        if len(splits) > 1:
            cross_split.append({**entry, "splits": sorted(splits)})
        if len(labels) > 1:
            cross_label.append({**entry, "labels": sorted(labels)})

    for group in exact_groups:
        consider(group["paths"], kind="exact", distance=0)
    for pair in near_pairs:
        consider(
            [pair["left"], pair["right"]],
            kind="near",
            distance=int(pair["hash_distance"]),
        )

    return {
        "cross_split": cross_split[:_MAX_NEAR_DUPLICATE_PAIRS],
        "cross_label": cross_label[:_MAX_NEAR_DUPLICATE_PAIRS],
    }


def _loader_traps(records: Sequence[ImageRecord]) -> dict[str, list[dict[str, Any]]]:
    """Files that decode fine but reach a training pipeline changed."""
    traps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.extension_mismatch:
            traps["extension_mismatch"].append(
                {
                    "path": record.relative_path,
                    "extension_suggests": record.extension_format,
                    "actual_format": record.format,
                }
            )
        if record.rotates_on_load:
            traps["rotates_on_load"].append(
                {
                    "path": record.relative_path,
                    "orientation": record.orientation,
                    "transposes_dimensions": record.transposes_on_load,
                }
            )
        if record.is_truncated:
            traps["truncated"].append(
                {
                    "path": record.relative_path,
                    "file_size_bytes": record.file_size_bytes,
                }
            )
        if record.is_grayscale_rgb:
            traps["grayscale_stored_as_color"].append(
                {"path": record.relative_path, "mode": record.mode}
            )
        if record.has_alpha and record.alpha_is_used:
            traps["transparency"].append(
                {"path": record.relative_path, "mode": record.mode}
            )
        if record.has_corrupt_metadata:
            traps["corrupt_metadata"].append({"path": record.relative_path})
    return {key: value for key, value in traps.items() if value}


def _label_profiles(records: Sequence[ImageRecord]) -> list[dict[str, Any]]:
    """Per-label dimension and quality stats, so bias per class stays visible."""
    grouped: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.label is not None:
            grouped[record.label].append(record)

    profiles: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        dimensions = Counter((record.width, record.height) for record in group)
        (dominant_width, dominant_height), dominant_count = dimensions.most_common(1)[0]
        profiles.append(
            {
                "label": label,
                "count": len(group),
                "share": len(group) / len(records) if records else 0.0,
                "unique_dimensions": len(dimensions),
                "dominant_dimension": f"{dominant_width}x{dominant_height}",
                "dominant_dimension_share": dominant_count / len(group),
                "median_megapixels": float(
                    np.median([record.megapixels for record in group])
                ),
                "mean_brightness": float(
                    np.mean([record.brightness for record in group])
                ),
                "mean_sharpness": float(
                    np.mean([record.sharpness for record in group])
                ),
                "formats": sorted({record.format or "unknown" for record in group}),
            }
        )
    return profiles


def _deviating_labels(
    profiles: Sequence[Mapping[str, Any]], records: Sequence[ImageRecord]
) -> list[dict[str, Any]]:
    """Labels whose images do not look like the rest of the dataset."""
    if len(profiles) < 2:
        return []
    dimensions = Counter((record.width, record.height) for record in records)
    (width, height), _ = dimensions.most_common(1)[0]
    dataset_dominant = f"{width}x{height}"

    brightness = np.asarray(
        [float(profile["mean_brightness"]) for profile in profiles], dtype=np.float64
    )
    median = float(np.median(brightness))
    mad = float(np.median(np.abs(brightness - median)))
    scores = (
        0.6745 * (brightness - median) / mad if mad > 0 else np.zeros_like(brightness)
    )

    deviating: list[dict[str, Any]] = []
    for profile, score in zip(profiles, scores, strict=True):
        reasons: list[str] = []
        if profile["dominant_dimension"] != dataset_dominant:
            reasons.append(
                f"stored mainly at {profile['dominant_dimension']} rather than the "
                f"dataset's {dataset_dominant}"
            )
        if len(profiles) >= 3 and abs(float(score)) >= 3.5:
            direction = "brighter" if float(score) > 0 else "darker"
            reasons.append(f"noticeably {direction} than the other labels")
        if reasons:
            deviating.append(
                {
                    "label": profile["label"],
                    "reasons": reasons,
                    "brightness_robust_z": float(score),
                }
            )
    return deviating


def _manifest_catalog(
    records: Sequence[ImageRecord],
    invalid: Sequence[InvalidImageRecord],
    *,
    root: Path | None,
) -> DatasetCatalog:
    row_count = len(records) + len(invalid)
    columns = (
        ColumnCatalog(
            name="path",
            physical_type="string",
            semantic_type="identifier",
            roles=("file_path",),
            row_count=row_count,
            non_null_count=row_count,
            missing_count=0,
            missing_rate=0.0,
            unique_count=row_count,
            unique_rate=1.0 if row_count else 0.0,
        ),
        ColumnCatalog(
            name="label",
            physical_type="string",
            semantic_type="categorical",
            roles=("directory_label",),
            row_count=row_count,
            non_null_count=sum(1 for record in records if record.label is not None),
            missing_count=sum(1 for record in records if record.label is None)
            + len(invalid),
            missing_rate=(
                (sum(1 for record in records if record.label is None) + len(invalid))
                / row_count
                if row_count
                else 0.0
            ),
            unique_count=len({record.label for record in records if record.label}),
            unique_rate=(
                len({record.label for record in records if record.label}) / row_count
                if row_count
                else 0.0
            ),
        ),
        ColumnCatalog(
            name="width",
            physical_type="int64",
            semantic_type="numeric",
            roles=("image_dimension",),
            row_count=row_count,
            non_null_count=len(records),
            missing_count=len(invalid),
            missing_rate=len(invalid) / row_count if row_count else 0.0,
            unique_count=len({record.width for record in records}),
            unique_rate=len({record.width for record in records}) / row_count
            if row_count
            else 0.0,
        ),
        ColumnCatalog(
            name="height",
            physical_type="int64",
            semantic_type="numeric",
            roles=("image_dimension",),
            row_count=row_count,
            non_null_count=len(records),
            missing_count=len(invalid),
            missing_rate=len(invalid) / row_count if row_count else 0.0,
            unique_count=len({record.height for record in records}),
            unique_rate=len({record.height for record in records}) / row_count
            if row_count
            else 0.0,
        ),
        ColumnCatalog(
            name="format",
            physical_type="string",
            semantic_type="categorical",
            roles=("image_encoding",),
            row_count=row_count,
            non_null_count=len(records),
            missing_count=len(invalid),
            missing_rate=len(invalid) / row_count if row_count else 0.0,
            unique_count=len({record.format for record in records}),
            unique_rate=len({record.format for record in records}) / row_count
            if row_count
            else 0.0,
        ),
    )
    fingerprint_payload = "|".join(
        [record.sha256 for record in records]
        + [f"invalid:{item.relative_path}:{item.error}" for item in invalid]
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()[:16]
    table = TableCatalog(
        name="images",
        row_count=row_count,
        column_count=len(columns),
        memory_bytes=sum(record.file_size_bytes for record in records)
        + sum(item.file_size_bytes or 0 for item in invalid),
        duplicate_row_count=None,
        fingerprint=fingerprint,
        fingerprint_method="sha256_image_manifest_v1",
        source=SourceInfo(
            kind="image_directory", location=root.as_posix() if root else None
        ),
        columns=columns,
    )
    return DatasetCatalog(
        fingerprint=fingerprint,
        fingerprint_method="sha256_image_manifest_v1",
        table_count=1,
        row_count=row_count,
        column_count=len(columns),
        tables=(table,),
    )


def _metric_table(
    title: str,
    rows: list[dict[str, Any]],
    *,
    columns: list[dict[str, str]],
    evidence_ids: tuple[str, ...],
    description: str,
) -> Artifact:
    return Artifact.create(
        kind="metric_table",
        title=title,
        data={"columns": columns, "rows": rows},
        evidence_ids=evidence_ids,
        metadata={"description": description},
    )


def _evidence(
    records: Sequence[ImageRecord],
    invalid: Sequence[InvalidImageRecord],
    *,
    root: Path | None,
    discovered_count: int,
    analyzed_count: int,
    near_duplicate_threshold: int,
) -> tuple[list[Evidence], dict[str, Any]]:
    dimension_counter = Counter((record.width, record.height) for record in records)
    format_counter = Counter(record.format or "unknown" for record in records)
    mode_counter = Counter(record.mode for record in records)
    label_counter = Counter(record.label for record in records)
    exact_groups = _exact_duplicate_groups(records)
    near_pairs, near_duplicate_scan = _near_duplicate_pairs(
        records, threshold=near_duplicate_threshold
    )
    quality_flags = _quality_flags(records)
    dimension_outliers = _robust_outliers(records, "megapixels")
    file_size_outliers = _robust_outliers(records, "file_size_bytes")
    aspect_outliers = _robust_outliers(records, "aspect_ratio")
    leakage = _leakage(records, exact_groups, near_pairs)
    traps = _loader_traps(records)
    label_profiles = _label_profiles(records)
    deviating_labels = _deviating_labels(label_profiles, records)
    split_counter = Counter(record.split for record in records if record.split)
    outlier_dimensions = {item["path"] for item in dimension_outliers + aspect_outliers}
    scatter_points = [
        {
            "width": width,
            "height": height,
            "count": count,
            "is_outlier": any(
                record.relative_path in outlier_dimensions
                for record in records
                if (record.width, record.height) == (width, height)
            ),
        }
        for (width, height), count in dimension_counter.most_common(400)
    ]

    evidence = [
        Evidence.create(
            kind="image_dataset_summary",
            scope=EvidenceScope(table="images"),
            value={
                "discovered_count": discovered_count,
                "analyzed_count": analyzed_count,
                "valid_count": len(records),
                "invalid_count": len(invalid),
                "root": root.as_posix() if root else None,
                "total_file_size_bytes": sum(
                    record.file_size_bytes for record in records
                )
                + sum(item.file_size_bytes or 0 for item in invalid),
            },
            method="image_file_scan_v1",
            description="Image path discovery and decode success summary.",
        ),
        Evidence.create(
            kind="image_dimension_distribution",
            scope=EvidenceScope(table="images", columns=("width", "height")),
            value={
                "unique_dimensions": len(dimension_counter),
                "top_dimensions": [
                    {"width": width, "height": height, "count": count}
                    for (width, height), count in dimension_counter.most_common(10)
                ],
                "megapixel_outliers": dimension_outliers,
                "aspect_ratio_outliers": aspect_outliers,
                "scatter_points": scatter_points,
            },
            method="exact_dimension_summary_v1",
            description="Image dimension, aspect ratio, and resolution distribution.",
        ),
        Evidence.create(
            kind="image_format_distribution",
            scope=EvidenceScope(table="images", columns=("format", "mode")),
            value={
                "formats": _counter_rows(format_counter, missing_label="unknown"),
                "modes": _counter_rows(mode_counter, missing_label="unknown"),
                "animated_count": sum(
                    1 for record in records if record.frame_count > 1
                ),
            },
            method="pillow_header_summary_v1",
            description="Image container format, mode, and animation summary.",
        ),
        Evidence.create(
            kind="image_label_distribution",
            scope=EvidenceScope(table="images", columns=("label",)),
            value={"labels": _counter_rows(label_counter, missing_label="unlabeled")},
            method="directory_label_summary_v1",
            description="Label distribution inferred from image directory names.",
        ),
        Evidence.create(
            kind="image_metadata_summary",
            scope=EvidenceScope(table="images"),
            value={
                "with_exif_count": sum(1 for record in records if record.has_exif),
                "orientation_counts": _counter_rows(
                    Counter(
                        str(record.orientation)
                        for record in records
                        if record.orientation is not None
                    ),
                    missing_label="missing",
                ),
                "max_exif_tag_count": max(
                    (record.exif_tag_count for record in records), default=0
                ),
            },
            method="pillow_exif_summary_v1",
            description="Presence of EXIF metadata and orientation tags.",
        ),
        Evidence.create(
            kind="image_duplicate_summary",
            scope=EvidenceScope(table="images"),
            value={
                "exact_duplicate_groups": exact_groups,
                "near_duplicate_pairs": near_pairs,
                "near_duplicate_threshold": near_duplicate_threshold,
                "near_duplicate_scan": near_duplicate_scan,
            },
            method="sha256_and_perceptual_hash_v1",
            description=(
                "Exact duplicate groups and perceptual near-duplicate candidates."
            ),
            assumptions=(
                "Perceptual hash matches are candidates and may include false "
                "positives.",
            ),
            confidence=0.86,
        ),
        Evidence.create(
            kind="image_quality_summary",
            scope=EvidenceScope(table="images"),
            value={
                "flags": quality_flags,
                "thresholds": {
                    "dark_brightness_max": _DARK_THRESHOLD,
                    "bright_brightness_min": _BRIGHT_THRESHOLD,
                    "low_contrast_max": _LOW_CONTRAST_THRESHOLD,
                    "low_sharpness_max": _LOW_SHARPNESS_THRESHOLD,
                    "low_entropy_max": _LOW_ENTROPY_THRESHOLD,
                },
            },
            method="pillow_grayscale_quality_metrics_v1",
            description=(
                "Basic visual quality flags from grayscale brightness, contrast, "
                "sharpness, and entropy."
            ),
            assumptions=(
                "Blur and low-information checks are lightweight triage metrics, not "
                "task-specific image quality models.",
            ),
            confidence=0.82,
        ),
        Evidence.create(
            kind="image_file_size_outliers",
            scope=EvidenceScope(table="images", columns=("file_size_bytes",)),
            value={"outliers": file_size_outliers},
            method="robust_z_file_size_v1",
            description="File-size outlier candidates using robust z-scores.",
        ),
        Evidence.create(
            kind="image_leakage_summary",
            scope=EvidenceScope(table="images", columns=("split", "label")),
            value={
                "splits": _counter_rows(split_counter, missing_label="unassigned"),
                "cross_split_duplicates": leakage["cross_split"],
                "cross_label_duplicates": leakage["cross_label"],
            },
            method="duplicate_cohort_crossing_v1",
            description=(
                "Duplicate and near-duplicate images that cross a dataset split or "
                "a label boundary."
            ),
            assumptions=(
                "Splits and labels are inferred from directory names, so this check "
                "only applies to folder-organized datasets.",
            ),
            confidence=0.9,
        ),
        Evidence.create(
            kind="image_loader_traps",
            scope=EvidenceScope(table="images"),
            value={
                "traps": traps,
                "counts": {key: len(value) for key, value in traps.items()},
            },
            method="pillow_header_trap_scan_v1",
            description=(
                "Files that decode successfully but reach a pipeline rotated, "
                "re-encoded, or with an unexpected channel count."
            ),
            confidence=0.95,
        ),
        Evidence.create(
            kind="image_label_profile",
            scope=EvidenceScope(table="images", columns=("label",)),
            value={
                "profiles": label_profiles,
                "deviating_labels": deviating_labels,
            },
            method="per_label_dimension_quality_summary_v1",
            description=(
                "Dimension and quality statistics computed separately for each "
                "inferred label."
            ),
        ),
        Evidence.create(
            kind="image_brightness_distribution",
            scope=EvidenceScope(table="images", columns=("brightness",)),
            value=_distribution_payload(records, "brightness", column="brightness"),
            method="grayscale_brightness_histogram_v1",
            description="Distribution of mean grayscale brightness across images.",
        ),
    ]
    if invalid:
        evidence.append(
            Evidence.create(
                kind="invalid_image_files",
                scope=EvidenceScope(table="images"),
                value={
                    "count": len(invalid),
                    "files": [
                        {
                            "path": item.relative_path,
                            "file_size_bytes": item.file_size_bytes,
                            "error": item.error,
                        }
                        for item in invalid[:20]
                    ],
                },
                method="pillow_verify_v1",
                description="Files that could not be identified or decoded as images.",
            )
        )
    derived = {
        "dimension_counter": dimension_counter,
        "format_counter": format_counter,
        "label_counter": label_counter,
        "exact_groups": exact_groups,
        "near_pairs": near_pairs,
        "quality_flags": quality_flags,
        "dimension_outliers": dimension_outliers,
        "file_size_outliers": file_size_outliers,
        "aspect_outliers": aspect_outliers,
        "leakage": leakage,
        "traps": traps,
        "label_profiles": label_profiles,
        "deviating_labels": deviating_labels,
    }
    return evidence, derived


def _findings_and_steps(
    evidence: Sequence[Evidence],
    derived: Mapping[str, Any],
    *,
    valid_count: int,
    invalid_count: int,
) -> tuple[list[Finding], list[TransformationStep]]:
    by_kind = {item.kind: item for item in evidence}
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    if invalid_count:
        item = by_kind["invalid_image_files"]
        rate = invalid_count / (valid_count + invalid_count)
        findings.append(
            Finding.create(
                title="Unreadable image files",
                summary=f"{invalid_count:,} file(s) ({rate:.1%}) could not be decoded.",
                severity="critical" if rate >= 0.05 else "high",
                confidence=1.0,
                evidence_ids=(item.id,),
                recommendation=(
                    "Remove, repair, or re-export unreadable files before training "
                    "or annotation."
                ),
            )
        )
        steps.append(
            TransformationStep(
                operation="review_unreadable_images",
                table="images",
                columns=("path",),
                parameters={"invalid_count": invalid_count},
                rationale="Unreadable files break downstream image pipelines.",
                evidence_ids=(item.id,),
                risk="high",
            )
        )

    dimension_counter: Counter[tuple[int, int]] = derived["dimension_counter"]
    if valid_count and len(dimension_counter) > 1:
        dominant_count = dimension_counter.most_common(1)[0][1]
        drift_rate = 1 - dominant_count / valid_count
        if drift_rate >= 0.1:
            item = by_kind["image_dimension_distribution"]
            findings.append(
                Finding.create(
                    title="Image dimensions are inconsistent",
                    summary=(
                        f"{drift_rate:.1%} of valid images differ from the most common "
                        "width x height."
                    ),
                    severity="high" if drift_rate >= 0.4 else "medium",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Confirm intended input size and choose an explicit resize, "
                        "crop, or padding policy."
                    ),
                )
            )

    if derived["dimension_outliers"] or derived["aspect_outliers"]:
        item = by_kind["image_dimension_distribution"]
        size_outlier_count = len(derived["dimension_outliers"]) + len(
            derived["aspect_outliers"]
        )
        findings.append(
            Finding.create(
                title="Resolution or aspect-ratio outliers",
                summary=(
                    f"{size_outlier_count} "
                    "image(s) have unusual size or shape compared with the dataset."
                ),
                severity="medium",
                confidence=0.95,
                evidence_ids=(item.id,),
                recommendation=(
                    "Inspect these files for thumbnails, panoramas, screenshots, or "
                    "ingestion mistakes."
                ),
            )
        )

    format_counter: Counter[str] = derived["format_counter"]
    if valid_count and len(format_counter) > 1:
        item = by_kind["image_format_distribution"]
        rare_formats = [
            fmt for fmt, count in format_counter.items() if count / valid_count < 0.1
        ]
        findings.append(
            Finding.create(
                title="Mixed image encodings",
                summary=(
                    f"{len(format_counter)} formats are present"
                    + (
                        f"; rare formats include {', '.join(sorted(rare_formats))}."
                        if rare_formats
                        else "."
                    )
                ),
                severity="medium" if rare_formats else "low",
                confidence=1.0,
                evidence_ids=(item.id,),
                recommendation=(
                    "Standardize encodings when downstream tooling expects one "
                    "format or compression profile."
                ),
            )
        )

    exact_groups = derived["exact_groups"]
    near_pairs = derived["near_pairs"]
    if exact_groups or near_pairs:
        item = by_kind["image_duplicate_summary"]
        exact_files = sum(group["count"] for group in exact_groups)
        findings.append(
            Finding.create(
                title="Duplicate or near-duplicate image candidates",
                summary=(
                    f"{len(exact_groups)} exact duplicate group(s) and "
                    f"{len(near_pairs)} near-duplicate pair candidate(s) were found."
                ),
                severity="high"
                if exact_files >= max(2, valid_count * 0.05)
                else "medium",
                confidence=item.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Review candidates before de-duplication; perceptual matches "
                    "are not confirmed duplicates."
                ),
            )
        )
        steps.append(
            TransformationStep(
                operation="review_duplicate_images",
                table="images",
                columns=("path",),
                parameters={
                    "exact_duplicate_groups": len(exact_groups),
                    "near_duplicate_pairs": len(near_pairs),
                },
                rationale=(
                    "Duplicates can leak between train/test splits and overstate "
                    "model performance."
                ),
                evidence_ids=(item.id,),
                risk="high",
            )
        )

    leakage = derived["leakage"]
    cross_split = leakage["cross_split"]
    cross_label = leakage["cross_label"]
    if cross_split:
        item = by_kind["image_leakage_summary"]
        findings.append(
            Finding.create(
                title="Duplicate images span train and evaluation splits",
                summary=(
                    f"{len(cross_split)} duplicate or near-duplicate group(s) appear "
                    "in more than one split, so evaluation scores will be optimistic."
                ),
                severity="critical",
                confidence=item.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Remove the leaked images from the evaluation split, then "
                    "re-split by image content rather than by file."
                ),
            )
        )
        steps.append(
            TransformationStep(
                operation="remove_cross_split_duplicate_images",
                table="images",
                columns=("path", "split"),
                parameters={"cross_split_groups": len(cross_split)},
                rationale=(
                    "An image present in both train and test inflates every reported "
                    "metric without changing the underlying model."
                ),
                evidence_ids=(item.id,),
                risk="high",
            )
        )

    if cross_label:
        item = by_kind["image_leakage_summary"]
        findings.append(
            Finding.create(
                title="Identical images carry conflicting labels",
                summary=(
                    f"{len(cross_label)} duplicate or near-duplicate group(s) are "
                    "filed under more than one label."
                ),
                severity="high",
                confidence=item.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Re-check these annotations; at most one of the conflicting "
                    "labels can be correct."
                ),
            )
        )

    traps: dict[str, list[dict[str, Any]]] = derived["traps"]
    if traps:
        item = by_kind["image_loader_traps"]
        rotating = traps.get("rotates_on_load", [])
        transposing = [row for row in rotating if row["transposes_dimensions"]]
        truncated = traps.get("truncated", [])
        mismatched = traps.get("extension_mismatch", [])
        redundant = traps.get("grayscale_stored_as_color", [])
        transparent = traps.get("transparency", [])

        if rotating:
            findings.append(
                Finding.create(
                    title="Images will rotate depending on the loader",
                    summary=(
                        f"{len(rotating):,} image(s) carry a non-default EXIF "
                        "orientation tag"
                        + (
                            f"; {len(transposing):,} of them also swap width and "
                            "height when the tag is honored."
                            if transposing
                            else "."
                        )
                    ),
                    severity="high" if transposing else "medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Apply the orientation explicitly on load (for example "
                        "PIL.ImageOps.exif_transpose) so every consumer sees the "
                        "same pixels."
                    ),
                )
            )
            steps.append(
                TransformationStep(
                    operation="normalize_exif_orientation",
                    table="images",
                    columns=("path",),
                    parameters={"affected_images": len(rotating)},
                    rationale=(
                        "Loaders disagree about whether to honor EXIF orientation, "
                        "so the same file can train and serve differently."
                    ),
                    evidence_ids=(item.id,),
                    risk="medium",
                )
            )

        if truncated:
            findings.append(
                Finding.create(
                    title="Truncated image files",
                    summary=(
                        f"{len(truncated):,} file(s) are incomplete and only decode "
                        "when truncated-image loading is enabled."
                    ),
                    severity="high",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Re-download or re-export these files; strict loaders will "
                        "raise on them mid-epoch."
                    ),
                )
            )

        if mismatched:
            findings.append(
                Finding.create(
                    title="File extensions do not match the actual encoding",
                    summary=(
                        f"{len(mismatched):,} file(s) are named for one format but "
                        "encoded as another."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Re-encode or rename these files; tooling that dispatches on "
                        "the extension will mis-handle them."
                    ),
                )
            )

        if redundant or transparent:
            details: list[str] = []
            if redundant:
                details.append(
                    f"{len(redundant):,} grayscale image(s) stored in a colour mode"
                )
            if transparent:
                details.append(
                    f"{len(transparent):,} image(s) with a used alpha channel"
                )
            findings.append(
                Finding.create(
                    title="Channel layout is inconsistent",
                    summary=" and ".join(details) + ".",
                    severity="low",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Decide on one channel layout and convert on load; "
                        "transparent pixels otherwise composite against whatever "
                        "background the loader assumes."
                    ),
                )
            )

    deviating_labels = derived["deviating_labels"]
    if deviating_labels:
        item = by_kind["image_label_profile"]
        headline = deviating_labels[0]
        findings.append(
            Finding.create(
                title="Some labels do not look like the rest of the dataset",
                summary=(
                    f"{len(deviating_labels)} label(s) differ from the dataset "
                    f"baseline; for example '{headline['label']}' is "
                    f"{headline['reasons'][0]}."
                ),
                severity="medium",
                confidence=0.9,
                evidence_ids=(item.id,),
                recommendation=(
                    "Check whether these classes were collected or exported "
                    "differently; a model can learn the artifact instead of the class."
                ),
            )
        )

    quality_flags: dict[str, list[dict[str, Any]]] = derived["quality_flags"]
    if quality_flags:
        item = by_kind["image_quality_summary"]
        total_flags = sum(len(items) for items in quality_flags.values())
        findings.append(
            Finding.create(
                title="Basic image quality flags",
                summary=(
                    f"{total_flags:,} quality flag(s) were raised across "
                    f"{len(quality_flags)} check type(s)."
                ),
                severity="medium",
                confidence=item.confidence,
                evidence_ids=(item.id,),
                recommendation=(
                    "Inspect flagged images for blur, blank frames, exposure "
                    "problems, or low-information samples."
                ),
            )
        )

    label_counter: Counter[str | None] = derived["label_counter"]
    labeled = {
        label: count for label, count in label_counter.items() if label is not None
    }
    if len(labeled) >= 2:
        smallest = min(labeled.values())
        largest = max(labeled.values())
        if smallest and largest / smallest >= 5:
            item = by_kind["image_label_distribution"]
            findings.append(
                Finding.create(
                    title="Directory labels are imbalanced",
                    summary=(
                        f"The largest inferred label has {largest:,} image(s), "
                        f"{largest / smallest:.1f}x the smallest label."
                    ),
                    severity="medium",
                    confidence=1.0,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Check whether imbalance reflects the task or a "
                        "collection/annotation gap."
                    ),
                )
            )

    if derived["file_size_outliers"]:
        item = by_kind["image_file_size_outliers"]
        findings.append(
            Finding.create(
                title="File-size outliers",
                summary=(
                    f"{len(derived['file_size_outliers'])} image file(s) have "
                    "unusual byte sizes."
                ),
                severity="low",
                confidence=0.95,
                evidence_ids=(item.id,),
                recommendation=(
                    "Inspect unusually tiny or huge files for thumbnails, failed "
                    "exports, or accidental raw images."
                ),
            )
        )

    return sort_findings(findings), steps


class _ThumbnailCache:
    """Encodes each file at most once, however many sections show it."""

    def __init__(self, records: Sequence[ImageRecord], *, size: int) -> None:
        self._by_relative = {record.relative_path: record for record in records}
        self._size = size
        self._cache: dict[str, str | None] = {}

    def record(self, relative_path: str) -> ImageRecord | None:
        return self._by_relative.get(relative_path)

    def source(self, relative_path: str) -> str | None:
        if relative_path not in self._cache:
            record = self._by_relative.get(relative_path)
            self._cache[relative_path] = (
                _thumbnail(Path(record.path), size=self._size) if record else None
            )
        return self._cache[relative_path]

    def item(self, relative_path: str, *, meta: str = "") -> dict[str, Any] | None:
        source = self.source(relative_path)
        if source is None:
            return None
        record = self._by_relative[relative_path]
        return {
            "src": source,
            "caption": relative_path,
            "meta": meta or f"{record.width}x{record.height}",
        }


def _pair_groups(
    pairs: Sequence[Mapping[str, Any]],
    cache: _ThumbnailCache,
    *,
    caption: str,
    limit: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for pair in pairs[:limit]:
        paths = pair.get("paths") or [pair["left"], pair["right"]]
        items = [item for path in paths[:4] if (item := cache.item(path)) is not None]
        if len(items) < 2:
            continue
        distance = int(pair.get("hash_distance", 0))
        note = "byte-identical" if distance == 0 else f"hash distance {distance}"
        if splits := pair.get("splits"):
            note = f"{note} · splits: {', '.join(splits)}"
        if labels := pair.get("labels"):
            note = f"{note} · labels: {', '.join(labels)}"
        groups.append({"caption": caption, "note": note, "items": items})
    return groups


def _contact_sheets(
    records: Sequence[ImageRecord],
    derived: Mapping[str, Any],
    evidence_by_kind: Mapping[str, Evidence],
    *,
    thumbnail_size: int,
) -> list[Artifact]:
    """Show the analyst the flagged images instead of only naming them."""
    cache = _ThumbnailCache(records, size=thumbnail_size)
    artifacts: list[Artifact] = []

    def sheet(
        title: str,
        groups: list[dict[str, Any]],
        *,
        layout: str,
        kind: str,
        description: str,
    ) -> None:
        if not groups:
            return
        artifacts.append(
            Artifact.create(
                kind="image_contact_sheet",
                title=title,
                data={"layout": layout, "groups": groups},
                evidence_ids=(evidence_by_kind[kind].id,),
                metadata={"description": description},
            )
        )

    leakage = derived["leakage"]
    sheet(
        "Leaked across splits",
        _pair_groups(
            leakage["cross_split"],
            cache,
            caption="Same image, different split",
            limit=6,
        ),
        layout="pairs",
        kind="image_leakage_summary",
        description="Duplicate images found in more than one dataset split.",
    )
    sheet(
        "Conflicting labels",
        _pair_groups(
            leakage["cross_label"],
            cache,
            caption="Same image, different label",
            limit=6,
        ),
        layout="pairs",
        kind="image_leakage_summary",
        description="Duplicate images filed under more than one label.",
    )

    duplicate_pairs = [
        {"paths": group["paths"], "hash_distance": 0}
        for group in derived["exact_groups"]
    ] + list(derived["near_pairs"])
    sheet(
        "Duplicate candidates",
        _pair_groups(duplicate_pairs, cache, caption="Duplicate candidate", limit=6),
        layout="pairs",
        kind="image_duplicate_summary",
        description="Exact and perceptual-hash duplicate candidates, side by side.",
    )

    quality_groups: list[dict[str, Any]] = []
    for check, rows in derived["quality_flags"].items():
        items = [
            item
            for row in rows[:_MAX_THUMBNAILS_PER_GROUP]
            if (item := cache.item(row["path"], meta=f"{row['value']:.3g}")) is not None
        ]
        if items:
            quality_groups.append(
                {
                    "caption": check.replace("_", " "),
                    "note": f"{len(rows)} flagged",
                    "items": items,
                }
            )
    sheet(
        "Quality-flagged images",
        quality_groups,
        layout="grid",
        kind="image_quality_summary",
        description="Images flagged as dark, bright, low-contrast, blurry, or blank.",
    )

    outlier_groups: list[dict[str, Any]] = []
    for caption, rows, unit in (
        ("resolution outliers", derived["dimension_outliers"], "MP"),
        ("aspect-ratio outliers", derived["aspect_outliers"], "ratio"),
        ("file-size outliers", derived["file_size_outliers"], "bytes"),
    ):
        items = [
            item
            for row in rows[:_MAX_THUMBNAILS_PER_GROUP]
            if (item := cache.item(row["path"], meta=f"{row['value']:,.3g} {unit}"))
            is not None
        ]
        if items:
            outlier_groups.append(
                {"caption": caption, "note": f"{len(rows)} flagged", "items": items}
            )
    sheet(
        "Size and shape outliers",
        outlier_groups,
        layout="grid",
        kind="image_dimension_distribution",
        description="Images whose resolution, shape, or byte size stands apart.",
    )

    trap_groups: list[dict[str, Any]] = []
    for check, rows in derived["traps"].items():
        if check in {"corrupt_metadata"}:
            continue
        items = [
            item
            for row in rows[:_MAX_THUMBNAILS_PER_GROUP]
            if (item := cache.item(row["path"])) is not None
        ]
        if items:
            trap_groups.append(
                {
                    "caption": check.replace("_", " "),
                    "note": f"{len(rows)} affected",
                    "items": items,
                }
            )
    sheet(
        "Loader traps",
        trap_groups,
        layout="grid",
        kind="image_loader_traps",
        description="Images that decode cleanly but reach a pipeline changed.",
    )
    return artifacts


def _artifacts(evidence: Sequence[Evidence]) -> tuple[Artifact, ...]:
    by_kind = {item.kind: item for item in evidence}
    dimension = by_kind["image_dimension_distribution"]
    formats = by_kind["image_format_distribution"]
    labels = by_kind["image_label_distribution"]
    quality = by_kind["image_quality_summary"]
    duplicates = by_kind["image_duplicate_summary"]
    artifacts = [
        _metric_table(
            "Image dimensions",
            [
                {
                    "size": f"{row['width']} x {row['height']}",
                    "count": row["count"],
                }
                for row in dimension.value["top_dimensions"]
            ],
            columns=[
                {"key": "size", "label": "Size"},
                {"key": "count", "label": "Images"},
            ],
            evidence_ids=(dimension.id,),
            description="Most common image sizes in the analyzed files.",
        ),
        _metric_table(
            "Image formats",
            formats.value["formats"],
            columns=[
                {"key": "value", "label": "Format"},
                {"key": "count", "label": "Images"},
                {"key": "rate", "label": "Share"},
            ],
            evidence_ids=(formats.id,),
            description="Container formats detected by Pillow.",
        ),
        _metric_table(
            "Inferred labels",
            labels.value["labels"],
            columns=[
                {"key": "value", "label": "Label"},
                {"key": "count", "label": "Images"},
                {"key": "rate", "label": "Share"},
            ],
            evidence_ids=(labels.id,),
            description="Labels inferred from directory names, when available.",
        ),
    ]
    label_profiles = by_kind["image_label_profile"].value["profiles"]
    if label_profiles:
        artifacts.append(
            _metric_table(
                "Per-label profile",
                [
                    {
                        "label": profile["label"],
                        "count": profile["count"],
                        "dominant_dimension": profile["dominant_dimension"],
                        "dominant_dimension_share": profile["dominant_dimension_share"],
                        "mean_brightness": profile["mean_brightness"],
                    }
                    for profile in label_profiles
                ],
                columns=[
                    {"key": "label", "label": "Label"},
                    {"key": "count", "label": "Images"},
                    {"key": "dominant_dimension", "label": "Usual size"},
                    {"key": "dominant_dimension_share", "label": "At that size"},
                    {"key": "mean_brightness", "label": "Mean brightness"},
                ],
                evidence_ids=(by_kind["image_label_profile"].id,),
                description=(
                    "Dimension and brightness statistics computed per label, so "
                    "collection bias in one class stays visible."
                ),
            )
        )

    quality_rows = [
        {"check": key, "flagged_images": len(value)}
        for key, value in quality.value["flags"].items()
    ]
    if quality_rows:
        artifacts.append(
            _metric_table(
                "Quality flags",
                quality_rows,
                columns=[
                    {"key": "check", "label": "Check"},
                    {"key": "flagged_images", "label": "Flagged images"},
                ],
                evidence_ids=(quality.id,),
                description="Lightweight visual-quality triage checks.",
            )
        )
    duplicate_rows = [
        {
            "type": "exact groups",
            "count": len(duplicates.value["exact_duplicate_groups"]),
        },
        {"type": "near pairs", "count": len(duplicates.value["near_duplicate_pairs"])},
    ]
    artifacts.append(
        _metric_table(
            "Duplicate candidates",
            duplicate_rows,
            columns=[
                {"key": "type", "label": "Type"},
                {"key": "count", "label": "Count"},
            ],
            evidence_ids=(duplicates.id,),
            description="Exact and perceptual-hash duplicate candidates.",
        )
    )
    return tuple(artifacts)


def profile_image_dataset(
    paths: Sequence[Path],
    *,
    root: Path | None,
    labels: Mapping[str, str | None],
    splits: Mapping[str, str | None] | None = None,
    context: AnalysisContext,
    config: AnalysisConfig,
    near_duplicate_threshold: int = 4,
    thumbnails: bool = True,
    thumbnail_size: int = 112,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Build a deterministic profile for a directory or list of images."""
    emit(
        callbacks,
        Event(EventKind.RUN_STARTED, "Image profile started.", stage="image_profile"),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    sampled_paths = _sample_paths(
        paths, config=config, warnings=warnings, sampling=sampling
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_STARTED,
            "Scanning image headers and lightweight quality metrics.",
            stage="scan",
        ),
    )
    split_map = splits or {}
    valid: list[ImageRecord] = []
    invalid: list[InvalidImageRecord] = []
    for path in sampled_paths:
        record = _scan_image(path, root=root, labels=labels, splits=split_map)
        if isinstance(record, ImageRecord):
            valid.append(record)
        else:
            invalid.append(record)

    catalog = _manifest_catalog(valid, invalid, root=root)
    evidence, derived = _evidence(
        valid,
        invalid,
        root=root,
        discovered_count=len(paths),
        analyzed_count=len(sampled_paths),
        near_duplicate_threshold=near_duplicate_threshold,
    )
    for item in evidence:
        emit(
            callbacks,
            Event(
                EventKind.EVIDENCE_CREATED,
                item.description,
                stage="evidence",
                data={"evidence_id": item.id, "kind": item.kind},
            ),
        )
    findings, steps = _findings_and_steps(
        evidence,
        derived,
        valid_count=len(valid),
        invalid_count=len(invalid),
    )
    artifacts = list(_artifacts(evidence))
    if thumbnails and valid:
        emit(
            callbacks,
            Event(
                EventKind.STAGE_STARTED,
                "Rendering thumbnails for flagged images.",
                stage="thumbnails",
            ),
        )
        artifacts.extend(
            _contact_sheets(
                valid,
                derived,
                {item.kind: item for item in evidence},
                thumbnail_size=thumbnail_size,
            )
        )

    if not paths:
        warnings.append(
            AnalysisWarning(
                code="no_image_files",
                message="No supported image files were found for profiling.",
            )
        )
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "No supported image files were found."
    elif not valid:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "Image files were found, but none could be decoded."
    elif invalid:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = (
            f"Profiled {len(valid):,} valid image(s) and found {len(findings)} "
            f"prioritized issue(s); {len(invalid):,} file(s) could not be decoded."
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = (
            f"Profiled {len(valid):,} image(s) across "
            f"{len(derived['format_counter'])} format(s) and "
            f"{len(derived['dimension_counter'])} size(s); found "
            f"{len(findings)} prioritized issue(s)."
        )

    result = AnalysisResult(
        goal="image_profile",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=tuple(artifacts),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "image_count": len(valid),
            "invalid_image_count": len(invalid),
            "discovered_image_count": len(paths),
            "analyzed_image_count": len(sampled_paths),
            "near_duplicate_threshold": near_duplicate_threshold,
            "thumbnails": thumbnails,
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.STAGE_COMPLETED,
            "Image evidence created.",
            stage="evidence",
            progress=1.0,
        ),
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="image_profile",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
