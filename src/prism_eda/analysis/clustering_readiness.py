"""Is this data even clusterable?

Clustering algorithms are unfalsifiable by construction: ask k-means for four
groups and it returns four groups, on segmented data and on uniform noise alike,
with no complaint either way. The silhouette score of that partition is a real
number in both cases. Nothing in the output distinguishes structure from its
absence.

So the checks have to come first, and they are of two kinds. Some ask whether
the *geometry* is sound — are the features on comparable scales, is the same
quantity present twice, has dimensionality made every point equidistant from
every other? Others ask whether there is anything to find at all, which is what
the Hopkins statistic estimates by comparing the data's own nearest-neighbour
distances against those of uniform points thrown into the same box.

Every one of these runs before a single cluster is fitted, because after that
point the output looks equally convincing either way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from prism_eda.analysis._clustering import (
    MAX_PROFILE_CATEGORIES,
    FeatureSet,
    as_float,
    distance_sample,
    stable,
)
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Ratio between the widest and narrowest feature range above which an unscaled
#: distance is effectively one feature wearing a disguise.
SCALE_RATIO_WARN = 10.0
SCALE_RATIO_SEVERE = 100.0

#: Absolute correlation at which two features are carrying one quantity.
REDUNDANT_CORRELATION = 0.95
MAX_REDUNDANT_PAIRS = 10

#: Variance a principal-component count has to reach to be called the effective
#: dimensionality.
INTRINSIC_VARIANCE = 0.9

#: Relative contrast below which distances have concentrated: every point is
#: roughly the same distance from every other and "nearest" stops meaning
#: anything.
#:
#: Measured between the 5th and 95th percentile of pairwise distances rather
#: than between the minimum and maximum. The textbook (max - min) / min is
#: dominated by whichever single pair happens to be closest — one near-duplicate
#: drives the denominator toward zero and the ratio toward infinity, reporting
#: excellent contrast for a dataset that has none.
CONCENTRATION_WARN = 1.0
CONCENTRATION_SEVERE = 0.5

#: Hopkins bands. Around 0.5 the data is indistinguishable from uniform noise;
#: climbing toward 1 means points cluster more tightly than chance.
HOPKINS_RANDOM = 0.55
HOPKINS_CLUSTERED = 0.75
HOPKINS_REPEATS = 10
HOPKINS_SAMPLE_RATE = 0.1
HOPKINS_MAX_SAMPLE = 150

#: A near-duplicate is a pair closer than this share of the median pairwise
#: distance. Not identical, but close enough to weight one region twice.
NEAR_DUPLICATE_RATIO = 0.02


def feature_evidence(
    frame: pd.DataFrame,
    selected: FeatureSet,
    *,
    table: str,
) -> Evidence:
    """What became the distance, what describes it, and what was excluded."""
    ranges: list[dict[str, Any]] = []
    for name in selected.numeric:
        column = pd.to_numeric(frame[name], errors="coerce")
        finite = column[np.isfinite(column)]
        if finite.empty:
            continue
        spread = float(finite.max() - finite.min())
        ranges.append(
            {
                "feature": name,
                "min": float(finite.min()),
                "max": float(finite.max()),
                "range": spread,
                "missing_rate": float(column.isna().mean()),
            }
        )
    ranges.sort(key=lambda item: item["range"], reverse=True)
    positive = [item["range"] for item in ranges if item["range"] > 0]
    ratio = (max(positive) / min(positive)) if len(positive) >= 2 else None

    return Evidence.create(
        kind="clustering_features",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "distance_features": selected.numeric,
            "distance_feature_count": len(selected.numeric),
            "profile_features": selected.categorical,
            "profile_feature_count": len(selected.categorical),
            "excluded": selected.excluded,
            "excluded_count": len(selected.excluded),
            "ranges": ranges,
            "scale_ratio": ratio,
            "widest_feature": ranges[0]["feature"] if ranges else None,
            "narrowest_feature": ranges[-1]["feature"] if ranges else None,
            "max_missing_rate": max(
                (item["missing_rate"] for item in ranges), default=0.0
            ),
        },
        method="feature_admission_and_scale_scan_v1",
        description=f"Clustering feature set for {table}.",
        confidence=1.0,
        assumptions=(
            "Only numeric features build the distance. Euclidean distance over "
            "one-hot categories asserts that every pair of categories is equally "
            "far apart, which is rarely what anyone means.",
            "Categorical columns are profiled per group instead, which is the "
            "question they can actually answer.",
            "Features are standardized before clustering; the scale ratio "
            "reported here describes the raw columns.",
        ),
    )


def redundancy_evidence(
    frame: pd.DataFrame,
    selected: FeatureSet,
    *,
    table: str,
) -> Evidence | None:
    """Features carrying the same quantity, which a distance double-weights."""
    if len(selected.numeric) < 2:
        return None
    block = frame[selected.numeric].apply(pd.to_numeric, errors="coerce")
    block = block.loc[:, block.nunique(dropna=True) > 1]
    if block.shape[1] < 2:
        return None
    correlations = block.corr(method="pearson").abs()
    pairs: list[dict[str, Any]] = []
    columns = list(correlations.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            value = correlations.loc[left, right]
            if pd.isna(value) or float(value) < REDUNDANT_CORRELATION:
                continue
            pairs.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "abs_correlation": float(value),
                }
            )
    if not pairs:
        return None
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return Evidence.create(
        kind="clustering_redundancy",
        scope=EvidenceScope(table=table, columns=tuple(block.columns.astype(str))),
        value={
            "redundant_pairs": pairs[:MAX_REDUNDANT_PAIRS],
            "redundant_pair_count": len(pairs),
            "evaluated_feature_count": int(block.shape[1]),
        },
        method="pairwise_feature_redundancy_v1",
        description=f"Redundant clustering features in {table}.",
        confidence=0.88,
        assumptions=(
            "Two features carrying one quantity give that quantity twice the "
            "weight in a Euclidean distance, which is a modelling choice made by "
            "accident rather than on purpose.",
        ),
    )


def duplicate_evidence(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    selected: FeatureSet,
    *,
    table: str,
    seed: int,
) -> Evidence | None:
    """Exact repeats and near-repeats, which weight one region of space twice."""
    if not selected.numeric or matrix.size == 0:
        return None
    exact = int(frame.duplicated().sum())
    feature_exact = int(frame[selected.numeric].duplicated().sum())

    near = 0
    threshold: float | None = None
    sample = distance_sample(matrix, seed=seed)
    if sample.shape[0] >= 10:
        neighbours = NearestNeighbors(n_neighbors=2).fit(sample)
        distances, _ = neighbours.kneighbors(sample)
        nearest = distances[:, 1]
        # Scale the threshold off the typical *non-zero* separation. Using the
        # plain median would collapse to zero on a table that is mostly exact
        # duplicates, which is exactly when this check matters most.
        positive = nearest[nearest > 0]
        if positive.size:
            threshold = NEAR_DUPLICATE_RATIO * float(np.median(positive))
            near = int((nearest <= threshold).sum())

    if not exact and not feature_exact and not near:
        return None
    return Evidence.create(
        kind="clustering_duplicates",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "exact_duplicate_rows": exact,
            "exact_duplicate_rate": exact / len(frame) if len(frame) else 0.0,
            "duplicate_feature_signatures": feature_exact,
            "near_duplicate_rows": near,
            "near_duplicate_threshold": threshold,
            "evaluated_rows": int(sample.shape[0]),
        },
        method="exact_and_near_duplicate_scan_v1",
        description=f"Duplicate observations in {table}.",
        confidence=0.9,
        assumptions=(
            "A duplicated row is a second vote for one location in space; enough "
            "of them create a dense region that a centroid will move toward.",
            "Near-duplicates are measured on the standardized feature matrix, so "
            "the threshold is relative to the data's own spread.",
        ),
    )


def geometry_evidence(
    matrix: np.ndarray,
    selected: FeatureSet,
    *,
    table: str,
    seed: int,
) -> Evidence | None:
    """Effective dimensionality and whether distances still discriminate.

    In enough dimensions every point sits at roughly the same distance from
    every other, the ratio of farthest to nearest tends to one, and any method
    built on "nearest" quietly stops working while still returning answers.
    """
    if matrix.size == 0 or matrix.shape[0] < 10 or matrix.shape[1] < 1:
        return None
    sample = distance_sample(matrix, seed=seed)

    components = min(sample.shape[0], sample.shape[1])
    intrinsic = None
    explained: list[float] = []
    if components >= 2:
        pca = PCA(n_components=components, random_state=seed).fit(sample)
        explained = [
            stable(float(value)) or 0.0 for value in pca.explained_variance_ratio_
        ]
        cumulative = np.cumsum(explained)
        intrinsic = int(np.searchsorted(cumulative, INTRINSIC_VARIANCE) + 1)

    # Relative contrast on a bounded sample of pairs: quadratic work, so the
    # sample is what keeps this affordable on a wide table.
    neighbours = NearestNeighbors(n_neighbors=min(len(sample), 2)).fit(sample)
    distances, _ = neighbours.kneighbors(sample)
    rng = np.random.default_rng(seed)
    picked = rng.choice(sample.shape[0], size=min(200, sample.shape[0]), replace=False)
    probe = sample[picked]
    pairwise = np.linalg.norm(probe[:, None, :] - probe[None, :, :], axis=-1)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0)]
    contrast = None
    if finite.size >= 20:
        low_pair = float(np.quantile(finite, 0.05))
        high_pair = float(np.quantile(finite, 0.95))
        if low_pair > 0:
            contrast = stable((high_pair - low_pair) / low_pair)

    return Evidence.create(
        kind="clustering_geometry",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "feature_count": int(matrix.shape[1]),
            "row_count": int(matrix.shape[0]),
            "intrinsic_dimensionality": intrinsic,
            "explained_variance_ratio": explained[:10],
            "first_two_components_variance": float(sum(explained[:2]))
            if len(explained) >= 2
            else None,
            "relative_contrast": contrast,
            "distances_concentrated": bool(
                contrast is not None and contrast < CONCENTRATION_WARN
            ),
            "evaluated_rows": int(sample.shape[0]),
            "median_nearest_neighbour": float(np.median(distances[:, -1]))
            if distances.shape[1] > 1
            else None,
        },
        method="pca_intrinsic_dimension_and_contrast_v1",
        description=f"Distance geometry for {table}.",
        confidence=0.84,
        assumptions=(
            "Relative contrast is (farthest - nearest) / nearest over a sample of "
            "pairs. As it approaches zero, every point is equally far from every "
            "other and distance-based clustering stops discriminating.",
            "Intrinsic dimensionality is the number of principal components "
            "needed for 90% of the variance, which is a linear notion of "
            "dimension and will overstate it for curved structure.",
        ),
    )


def tendency_evidence(
    matrix: np.ndarray,
    *,
    table: str,
    seed: int,
) -> Evidence | None:
    """The Hopkins statistic, repeated, because one draw of it is noise.

    Compares nearest-neighbour distances of real points against those of uniform
    points thrown into the same bounding box. Near 0.5 the data is
    indistinguishable from uniform noise and any clustering of it is a partition
    of nothing. The repeats matter: a single Hopkins value on a small sample
    varies enough to move across a band boundary on its own.
    """
    rows, columns = matrix.shape if matrix.size else (0, 0)
    if rows < 20 or columns < 1:
        return None
    sample_size = max(5, min(HOPKINS_MAX_SAMPLE, int(rows * HOPKINS_SAMPLE_RATE)))
    if sample_size >= rows:
        sample_size = max(5, rows // 3)
    if sample_size < 5:
        return None

    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    rng = np.random.default_rng(seed)
    neighbours = NearestNeighbors(n_neighbors=2).fit(matrix)

    scores: list[float] = []
    for _ in range(HOPKINS_REPEATS):
        picked = rng.choice(rows, size=sample_size, replace=False)
        # Real points: the distance to the *second* neighbour, because the first
        # is the point itself.
        real, _ = neighbours.kneighbors(matrix[picked], n_neighbors=2)
        real_distance = real[:, 1]

        synthetic = rng.uniform(low, high, size=(sample_size, columns))
        fake, _ = neighbours.kneighbors(synthetic, n_neighbors=1)
        fake_distance = fake[:, 0]

        total = float(fake_distance.sum() + real_distance.sum())
        if total <= 0:
            continue
        scores.append(float(fake_distance.sum() / total))
    if not scores:
        return None

    values = np.asarray(scores, dtype="float64")
    mean = stable(float(values.mean())) or 0.0
    verdict = (
        "no_tendency"
        if mean < HOPKINS_RANDOM
        else "clustered"
        if mean >= HOPKINS_CLUSTERED
        else "weak_tendency"
    )
    return Evidence.create(
        kind="clustering_tendency",
        scope=EvidenceScope(table=table),
        value={
            "hopkins_mean": mean,
            "hopkins_min": float(values.min()),
            "hopkins_max": float(values.max()),
            "hopkins_std": as_float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "repeats": len(scores),
            "sample_size": sample_size,
            "verdict": verdict,
            "random_below": HOPKINS_RANDOM,
            "clustered_above": HOPKINS_CLUSTERED,
        },
        method="repeated_hopkins_statistic_v1",
        description=f"Cluster tendency for {table}.",
        confidence=0.8,
        assumptions=(
            "Hopkins compares the data against uniform points in its own "
            "bounding box, so a non-rectangular but structureless region can "
            "score above 0.5 for shape alone.",
            "Around 0.5 the data is indistinguishable from noise; any clustering "
            "of it partitions nothing.",
            "The statistic is repeated because one draw varies enough to cross a "
            "band boundary by itself.",
            "Hopkins drifts upward in high dimensions even on pure noise, because "
            "the bounding box is mostly empty and uniform points land far from "
            "everything. Read it alongside the distance-concentration measure "
            "rather than on its own.",
        ),
    )


def profile_categoricals(frame: pd.DataFrame, selected: FeatureSet) -> list[str]:
    """Categorical columns worth describing groups with."""
    return [
        name
        for name in selected.categorical
        if frame[name].nunique(dropna=True) <= MAX_PROFILE_CATEGORIES
    ]
