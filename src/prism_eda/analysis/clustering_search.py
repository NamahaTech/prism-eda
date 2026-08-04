"""What structure is there, and would it survive being asked again?

Internal validity scores — silhouette, Calinski–Harabasz, Davies–Bouldin — all
share a defect: they are computed on the same partition they are judging, so
they measure how *tidy* a partition is, never whether it is real. Uniform noise
cut into four pieces has a silhouette score, and picking the k that maximizes it
will always return a number.

Stability is the check that can actually fail. Cluster two overlapping
subsamples separately and compare where they agree: real groups reappear, and a
partition of noise lands somewhere different every time. The adjusted Rand index
between those two labellings is the closest thing clustering has to a held-out
score, and it is what this module leads with.

Nothing here names a best k. It reports what each k scores, how reliably each
one reproduces, and lets a candidate emerge only when both agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from prism_eda.analysis._clustering import (
    FeatureSet,
    candidate_k_values,
    distance_sample,
    stable,
)
from prism_eda.config import AnalysisMode
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Resampling repeats per k, by compute mode.
_STABILITY_REPEATS = {
    AnalysisMode.QUICK: 3,
    AnalysisMode.STANDARD: 5,
    AnalysisMode.DEEP: 10,
}

#: Share of rows in each stability subsample.
SUBSAMPLE_RATE = 0.8

#: Adjusted Rand index bands. Below the first, a re-run of the same analysis on
#: slightly different rows produces a substantially different answer, and the
#: groups are an artifact of these particular rows.
STABILITY_UNSTABLE = 0.6
STABILITY_STRONG = 0.75

#: Silhouette below which the separation is too weak to call groups.
SILHOUETTE_WEAK = 0.25

#: Rows used for silhouette, which is quadratic in the sample.
MAX_SILHOUETTE_ROWS = 3_000

#: Features probed one-at-a-time for dominance.
MAX_SENSITIVITY_FEATURES = 8

#: Below this adjusted Rand index, removing one feature changes the answer
#: enough that the clustering is essentially that feature.
DOMINANCE_ARI = 0.5


def _fit(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)


def sweep_evidence(
    matrix: np.ndarray,
    selected: FeatureSet,
    *,
    table: str,
    seed: int,
    mode: AnalysisMode | str,
) -> Evidence | None:
    """Score and stability for every plausible k, with no winner declared."""
    rows = matrix.shape[0]
    candidates = candidate_k_values(rows)
    if not candidates or matrix.size == 0:
        return None
    repeats = _STABILITY_REPEATS[AnalysisMode(mode)]
    scoring = distance_sample(matrix, seed=seed, limit=MAX_SILHOUETTE_ROWS)
    subsample_size = max(10, int(rows * SUBSAMPLE_RATE))

    results: list[dict[str, Any]] = []
    for k in candidates:
        try:
            labels = _fit(matrix, k, seed)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
            continue
        if len(np.unique(labels)) < 2:
            continue

        scoring_labels = _fit(scoring, k, seed) if scoring.shape[0] != rows else labels
        try:
            silhouette = float(silhouette_score(scoring, scoring_labels))
            calinski = float(calinski_harabasz_score(scoring, scoring_labels))
            davies = float(davies_bouldin_score(scoring, scoring_labels))
        except ValueError:  # pragma: no cover - degenerate partition
            continue

        # The part that can fail: cluster two overlapping subsamples separately
        # and compare them where they overlap.
        agreements: list[float] = []
        rng = np.random.default_rng(seed + k)
        for repeat in range(repeats):
            first = rng.choice(rows, size=subsample_size, replace=False)
            second = rng.choice(rows, size=subsample_size, replace=False)
            shared = np.intersect1d(first, second)
            if shared.size < max(10, k * 2):
                continue
            try:
                left = KMeans(n_clusters=k, n_init=5, random_state=seed + repeat).fit(
                    matrix[first]
                )
                right = KMeans(n_clusters=k, n_init=5, random_state=seed + repeat).fit(
                    matrix[second]
                )
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
            agreements.append(
                float(
                    adjusted_rand_score(
                        left.predict(matrix[shared]), right.predict(matrix[shared])
                    )
                )
            )

        sizes = np.bincount(labels, minlength=k)
        results.append(
            {
                "k": k,
                "silhouette": stable(silhouette),
                "calinski_harabasz": stable(calinski),
                "davies_bouldin": stable(davies),
                "inertia": stable(
                    float(
                        KMeans(n_clusters=k, n_init=10, random_state=seed)
                        .fit(matrix)
                        .inertia_
                    )
                ),
                "stability_mean": (
                    stable(float(np.mean(agreements))) if agreements else None
                ),
                "stability_min": (
                    stable(float(np.min(agreements))) if agreements else None
                ),
                "stability_repeats": len(agreements),
                "smallest_cluster": int(sizes.min()),
                "smallest_cluster_rate": float(sizes.min() / rows),
            }
        )
    if not results:
        return None

    # A candidate only where the two disagreeing kinds of evidence agree: the
    # partition is reasonably separated *and* it reproduces on resampled rows.
    viable = [
        item
        for item in results
        if item["silhouette"] >= SILHOUETTE_WEAK
        and item["stability_mean"] is not None
        and item["stability_mean"] >= STABILITY_UNSTABLE
    ]
    candidate = (
        max(viable, key=lambda item: (item["stability_mean"], item["silhouette"]))
        if viable
        else None
    )
    best_silhouette = max(results, key=lambda item: item["silhouette"])

    return Evidence.create(
        kind="clustering_k_sweep",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "algorithm": "kmeans",
            "results": results,
            "evaluated_k": [item["k"] for item in results],
            "candidate_k": candidate["k"] if candidate else None,
            "candidate_silhouette": candidate["silhouette"] if candidate else None,
            "candidate_stability": candidate["stability_mean"] if candidate else None,
            "highest_silhouette_k": best_silhouette["k"],
            "highest_silhouette": best_silhouette["silhouette"],
            "stability_repeats": repeats,
            "subsample_rate": SUBSAMPLE_RATE,
            "row_count": rows,
            "unstable_below": STABILITY_UNSTABLE,
            "weak_silhouette_below": SILHOUETTE_WEAK,
        },
        method="kmeans_sweep_with_resampling_stability_v1",
        description=f"Cluster-count search for {table}.",
        confidence=0.78,
        assumptions=(
            "Silhouette, Calinski-Harabasz, and Davies-Bouldin are computed on "
            "the same partition they judge, so they measure tidiness, not "
            "reality. Uniform noise cut into k pieces scores on all three.",
            "Stability is the adjusted Rand index between two independently "
            "clustered overlapping subsamples. It is the check that can fail.",
            "A candidate k is not a determination that k groups exist. Clustering "
            "has no ground truth to be right about.",
            "k-means assumes roughly spherical, similarly sized groups; elongated "
            "or nested structure will score poorly here and still be real.",
        ),
    )


def sensitivity_evidence(
    matrix: np.ndarray,
    unscaled: np.ndarray,
    selected: FeatureSet,
    *,
    table: str,
    seed: int,
    k: int,
) -> Evidence | None:
    """How much the answer depends on choices nobody thinks of as choices.

    Two of them. Standardizing or not is a decision most pipelines make by
    default and never revisit, and on features with very different ranges it
    changes the partition completely. Dropping one feature reveals whether the
    grouping is a summary of everything or a restatement of one column.
    """
    if matrix.size == 0 or matrix.shape[0] < 20 or k < 2:
        return None
    try:
        reference = _fit(matrix, k, seed)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        return None

    scaling_agreement = None
    if unscaled.size and unscaled.shape == matrix.shape:
        try:
            scaling_agreement = stable(
                float(adjusted_rand_score(reference, _fit(unscaled, k, seed)))
            )
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            scaling_agreement = None

    dropped: list[dict[str, Any]] = []
    if matrix.shape[1] > 1:
        for position, name in list(enumerate(selected.numeric))[
            :MAX_SENSITIVITY_FEATURES
        ]:
            reduced = np.delete(matrix, position, axis=1)
            if reduced.shape[1] < 1:
                continue
            try:
                agreement = (
                    stable(
                        float(adjusted_rand_score(reference, _fit(reduced, k, seed)))
                    )
                    or 0.0
                )
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
            dropped.append(
                {
                    "feature": name,
                    "agreement_without_it": agreement,
                    "dominant": agreement < DOMINANCE_ARI,
                }
            )
    dropped.sort(key=lambda item: item["agreement_without_it"])

    return Evidence.create(
        kind="clustering_sensitivity",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "k": k,
            "scaling_agreement": scaling_agreement,
            "scaling_changes_answer": bool(
                scaling_agreement is not None and scaling_agreement < STABILITY_UNSTABLE
            ),
            "feature_drops": dropped,
            "dominant_features": [
                item["feature"] for item in dropped if item["dominant"]
            ],
            "dominance_below": DOMINANCE_ARI,
        },
        method="scaling_and_feature_dropout_sensitivity_v1",
        description=f"Sensitivity of the {table} clustering to its own setup.",
        confidence=0.8,
        assumptions=(
            "Agreement is the adjusted Rand index against the reference "
            "partition; 1.0 means the choice made no difference.",
            "A grouping that survives dropping any single feature is a summary "
            "of the whole feature set. One that does not is a restatement of the "
            "feature whose removal broke it.",
        ),
    )


def algorithm_evidence(
    selected: FeatureSet,
    *,
    table: str,
    concentrated: bool,
    intrinsic_dimensionality: int | None,
    feature_count: int,
    smallest_cluster_rate: float | None,
    categorical_count: int,
) -> Evidence:
    """Which methods suit the geometry that was actually measured."""
    suggestions: list[dict[str, str]] = []
    if categorical_count:
        suggestions.append(
            {
                "algorithm": "k-prototypes or Gower distance",
                "because": (
                    f"{categorical_count} categorical column(s) were kept out of "
                    "the distance. If they carry the grouping, a mixed-type "
                    "distance is the tool, not one-hot Euclidean."
                ),
            }
        )
    if concentrated or (
        intrinsic_dimensionality is not None
        and feature_count > 0
        and intrinsic_dimensionality < feature_count / 2
    ):
        suggestions.append(
            {
                "algorithm": "reduce dimensions first (PCA or UMAP), then cluster",
                "because": (
                    "Most of the variance lives in far fewer directions than "
                    "there are features, and distance loses contrast in the rest."
                ),
            }
        )
    if smallest_cluster_rate is not None and smallest_cluster_rate < 0.05:
        suggestions.append(
            {
                "algorithm": "DBSCAN or HDBSCAN",
                "because": (
                    "k-means must assign every point to a group. A density "
                    "method can leave sparse points unassigned instead of "
                    "forcing them into the nearest centroid."
                ),
            }
        )
    suggestions.append(
        {
            "algorithm": "hierarchical (Ward) clustering",
            "because": (
                "It gives a dendrogram rather than one partition, which shows "
                "how groups merge instead of asserting a single k."
            ),
        }
    )
    return Evidence.create(
        kind="clustering_algorithm_guidance",
        scope=EvidenceScope(table=table),
        value={
            "evaluated_with": "kmeans",
            "suggestions": suggestions,
            "distance_feature_count": feature_count,
            "excluded_categorical_count": categorical_count,
        },
        method="geometry_based_algorithm_guidance_v1",
        description=f"Algorithm guidance for {table}.",
        confidence=0.7,
        assumptions=(
            "k-means was used for the search because it is fast and "
            "deterministic under a fixed seed, not because it is the right "
            "model for this data.",
            "These are directions to try, not ranked recommendations.",
        ),
    )
