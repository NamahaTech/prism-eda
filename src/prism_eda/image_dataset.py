"""Image dataset session object."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from prism_eda.analysis.image_profile import profile_image_dataset
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import EventCallback
from prism_eda.results import AnalysisResult

ImageSource = str | Path | Sequence[str | Path]

_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _matches(path: Path, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return False
    text = path.as_posix()
    return any(
        fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(path.name, pattern)
        for pattern in patterns
    )


def _discover_files(
    source: ImageSource,
    *,
    recursive: bool,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> tuple[tuple[Path, ...], Path | None]:
    if isinstance(source, str | Path):
        sources: tuple[Path, ...] = (Path(source),)
    else:
        sources = tuple(Path(item) for item in source)

    files: list[Path] = []
    roots: list[Path] = []
    for item in sources:
        path = item.expanduser()
        if path.is_dir():
            roots.append(path)
            iterator = path.rglob("*") if recursive else path.iterdir()
            for candidate in iterator:
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                if include and not _matches(candidate.relative_to(path), include):
                    continue
                if _matches(candidate.relative_to(path), exclude):
                    continue
                files.append(candidate)
        elif path.is_file():
            if path.suffix.lower() in _IMAGE_EXTENSIONS:
                if include and not _matches(path, include):
                    continue
                if _matches(path, exclude):
                    continue
                files.append(path)
        else:
            raise FileNotFoundError(path)

    common_root: Path | None = None
    if len(roots) == 1:
        common_root = roots[0]
    elif files:
        try:
            common_root = Path.cwd()
        except OSError:
            common_root = None

    return tuple(sorted(files, key=lambda item: item.as_posix())), common_root


#: Directory names that conventionally denote a dataset split rather than a class.
_SPLIT_DIRECTORY_NAMES = frozenset(
    {
        "dev",
        "eval",
        "evaluation",
        "holdout",
        "test",
        "testing",
        "train",
        "training",
        "val",
        "valid",
        "validation",
    }
)


def _directory_parts(path: Path, root: Path | None) -> tuple[str, ...]:
    """Directory names between the dataset root and the file itself."""
    if root is not None:
        try:
            return path.relative_to(root).parts[:-1]
        except ValueError:
            pass
    parent = path.parent.name
    return (parent,) if parent else ()


def _label_and_split(
    path: Path, root: Path | None, strategy: str | None
) -> tuple[str | None, str | None]:
    """Infer a class label and a dataset split from the directory layout.

    ``images/train/cat/001.png`` is the near-universal image-classification
    layout, so the split directory is recognized and skipped rather than being
    mistaken for the class label.
    """
    if strategy is not None and strategy != "directory":
        raise ValueError("label_strategy must be 'directory' or None")

    label: str | None = None
    split: str | None = None
    for part in _directory_parts(path, root):
        if split is None and part.casefold() in _SPLIT_DIRECTORY_NAMES:
            split = part.casefold()
            continue
        if label is None:
            label = part
    return (label if strategy is not None else None), split


class ImageDataset:
    """A collection of image files and its deterministic profiling state."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        root: Path | None,
        label_strategy: str | None,
    ) -> None:
        self._files = tuple(files)
        self._root = root
        self._label_strategy = label_strategy
        inferred = {
            path.as_posix(): _label_and_split(path, root, label_strategy)
            for path in self._files
        }
        self._labels = {path: label for path, (label, _) in inferred.items()}
        self._splits = {path: split for path, (_, split) in inferred.items()}

    @classmethod
    def load(
        cls,
        source: ImageSource,
        *,
        recursive: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        label_strategy: str | None = "directory",
    ) -> ImageDataset:
        files, root = _discover_files(
            source,
            recursive=recursive,
            include=include,
            exclude=exclude,
        )
        return cls(files, root=root, label_strategy=label_strategy)

    @property
    def files(self) -> tuple[Path, ...]:
        return self._files

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def labels(self) -> Mapping[str, str | None]:
        return MappingProxyType(self._labels)

    @property
    def splits(self) -> Mapping[str, str | None]:
        """Dataset split (``train``/``val``/``test``…) inferred per file, if any."""
        return MappingProxyType(self._splits)

    def analyze(
        self,
        goal: str = "image_profile",
        *,
        context: AnalysisContext | Mapping[str, Any] | None = None,
        config: AnalysisConfig | None = None,
        callbacks: Sequence[EventCallback] = (),
        **options: Any,
    ) -> AnalysisResult:
        normalized_goal = goal.strip().lower().replace("-", "_")
        if normalized_goal not in {"image_profile", "profile_images", "images"}:
            raise NotImplementedError(
                f"Goal {goal!r} is not implemented for ImageDataset. "
                "Use 'image_profile'."
            )
        if context is None:
            analysis_context = AnalysisContext(goal=normalized_goal)
        elif isinstance(context, AnalysisContext):
            analysis_context = context
        else:
            analysis_context = AnalysisContext(goal=normalized_goal, **dict(context))
        analysis_config = config or AnalysisConfig(
            mode=options.pop("mode", AnalysisMode.STANDARD),
            sampling=options.pop("sampling", "auto"),
            random_seed=options.pop("random_seed", 42),
            allow_insufficient_evidence=options.pop(
                "allow_insufficient_evidence", False
            ),
        )
        near_duplicate_threshold = options.pop("near_duplicate_threshold", 4)
        thumbnails = options.pop("thumbnails", True)
        thumbnail_size = options.pop("thumbnail_size", 112)
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unexpected analysis options: {unknown}")
        return profile_image_dataset(
            self._files,
            root=self._root,
            labels=self._labels,
            splits=self._splits,
            context=analysis_context,
            config=analysis_config,
            near_duplicate_threshold=near_duplicate_threshold,
            thumbnails=thumbnails,
            thumbnail_size=thumbnail_size,
            callbacks=tuple(callbacks),
        )

    def profile(
        self,
        *,
        context: AnalysisContext | Mapping[str, Any] | None = None,
        config: AnalysisConfig | None = None,
        callbacks: Sequence[EventCallback] = (),
        mode: AnalysisMode | str = AnalysisMode.STANDARD,
        sampling: str = "auto",
        random_seed: int = 42,
        allow_insufficient_evidence: bool = False,
        near_duplicate_threshold: int = 4,
        thumbnails: bool = True,
        thumbnail_size: int = 112,
    ) -> AnalysisResult:
        if config is None:
            config = AnalysisConfig(
                mode=mode,
                sampling=sampling,  # type: ignore[arg-type]
                random_seed=random_seed,
                allow_insufficient_evidence=allow_insufficient_evidence,
            )
        return self.analyze(
            "image_profile",
            context=context,
            config=config,
            callbacks=callbacks,
            near_duplicate_threshold=near_duplicate_threshold,
            thumbnails=thumbnails,
            thumbnail_size=thumbnail_size,
        )
