"""What are the groups, if there are any?

This module only runs once the readiness checks and the stability search have
both said there is something to describe. That ordering is the whole design: a
segment profile is extremely persuasive — sizes, distinguishing features,
example rows, a scatter with coloured blobs — and it looks exactly as convincing
computed from noise. Producing it before establishing that the structure
reproduces would be building the most believable part of the report on the least
supported claim.

Two things are worth saying about what it produces. A group is described by how
far its members sit from the overall average *in standard deviations*, not in
raw units, because "spends 3,000 more" is unreadable without knowing the spread
and "1.8 standard deviations above" is comparable across every feature. And the
categorical columns that were kept out of the distance come back here, which is
the job they can actually do: "82% of this group is in the west" is a useful
sentence about a group that was found without reference to region at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prism_eda.analysis._clustering import (
    MAX_REPRESENTATIVES,
    MAX_SEGMENTS_PROFILED,
    FeatureSet,
    distance_sample,
    stable,
)
from prism_eda.evidence.models import Evidence, EvidenceScope

#: Standard deviations from the overall mean before a feature is "distinguishing".
DISTINGUISHING_Z = 0.5

#: Distinguishing features listed per segment.
MAX_DISTINGUISHING = 4

#: Lift above which a category is over-represented in a segment.
CATEGORY_LIFT = 1.5

#: Points drawn in the embedding scatter.
MAX_EMBEDDING_POINTS = 1_200


def segment_evidence(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    index: pd.Index,
    selected: FeatureSet,
    profile_categoricals: list[str],
    *,
    table: str,
    redundant_pairs: list[dict[str, Any]] | None = None,
) -> Evidence | None:
    """Sizes, distinguishing features, category mix, and representative rows."""
    if matrix.size == 0 or labels.size != matrix.shape[0]:
        return None
    working = frame.loc[index]
    numeric = working[selected.numeric].apply(pd.to_numeric, errors="coerce")
    overall_mean = numeric.mean()
    overall_std = numeric.std(ddof=0).replace(0.0, np.nan)

    # Two features carrying one quantity would otherwise describe every segment
    # twice — "visits per month higher, visits per year higher" — which reads as
    # two pieces of evidence for what is a single fact.
    twins: dict[str, set[str]] = {}
    for pair in redundant_pairs or []:
        twins.setdefault(pair["left"], set()).add(pair["right"])
        twins.setdefault(pair["right"], set()).add(pair["left"])

    segments: list[dict[str, Any]] = []
    for label in sorted(np.unique(labels))[:MAX_SEGMENTS_PROFILED]:
        mask = labels == label
        member_index = index[mask]
        block = numeric.loc[member_index]
        if block.empty:
            continue

        # Distance from the overall average in standard deviations: comparable
        # across features whose raw units are not.
        deviation = ((block.mean() - overall_mean) / overall_std).dropna()
        ranked = deviation.reindex(deviation.abs().sort_values(ascending=False).index)
        distinguishing: list[dict[str, Any]] = []
        named: set[str] = set()
        for name, value in ranked.items():
            if len(distinguishing) >= MAX_DISTINGUISHING:
                break
            if abs(float(value)) < DISTINGUISHING_Z:
                break
            feature = str(name)
            if twins.get(feature, set()) & named:
                continue
            named.add(feature)
            distinguishing.append(
                {
                    "feature": feature,
                    "segment_mean": float(block[feature].mean()),
                    "overall_mean": float(overall_mean[feature]),
                    "z": stable(float(value)),
                    "direction": "higher" if value > 0 else "lower",
                }
            )

        categories: list[dict[str, Any]] = []
        for column in profile_categoricals:
            values = working.loc[member_index, column].astype("string")
            if values.dropna().empty:
                continue
            share = values.value_counts(normalize=True, dropna=True)
            baseline = (
                working[column]
                .astype("string")
                .value_counts(normalize=True, dropna=True)
            )
            top = str(share.index[0])
            base_share = float(baseline.get(top, 0.0))
            lift = float(share.iloc[0] / base_share) if base_share else None
            categories.append(
                {
                    "column": column,
                    "top_value": top,
                    "share": float(share.iloc[0]),
                    "overall_share": base_share,
                    "lift": lift,
                    "over_represented": bool(
                        lift is not None and lift >= CATEGORY_LIFT
                    ),
                }
            )

        # Medoids: the rows closest to the group's own centre, which are real
        # records rather than a synthetic average nobody in the data resembles.
        centre = matrix[mask].mean(axis=0)
        offsets = np.linalg.norm(matrix[mask] - centre, axis=1)
        closest = member_index[np.argsort(offsets)[:MAX_REPRESENTATIVES]]
        representatives = [
            {
                "row_index": str(row),
                "values": {
                    name: (
                        None
                        if pd.isna(working.loc[row, name])
                        else float(working.loc[row, name])
                    )
                    for name in selected.numeric
                },
            }
            for row in closest
        ]

        segments.append(
            {
                "segment": int(label),
                "size": int(mask.sum()),
                "share": float(mask.sum() / len(labels)),
                "distinguishing_features": distinguishing,
                "categories": categories,
                "representatives": representatives,
                "is_undistinguished": not distinguishing,
            }
        )
    if not segments:
        return None

    return Evidence.create(
        kind="clustering_segments",
        scope=EvidenceScope(table=table, columns=tuple(selected.numeric)),
        value={
            "segment_count": len(segments),
            "segments": segments,
            "profiled_row_count": int(len(labels)),
            "distinguishing_threshold_z": DISTINGUISHING_Z,
            "category_lift_threshold": CATEGORY_LIFT,
            "undistinguished_count": sum(
                1 for item in segments if item["is_undistinguished"]
            ),
            # A categorical that is over-represented nowhere is itself a
            # finding: it says the grouping has nothing to do with that column.
            "uninformative_categoricals": [
                column
                for column in profile_categoricals
                if not any(
                    entry["over_represented"]
                    for segment in segments
                    for entry in segment["categories"]
                    if entry["column"] == column
                )
            ],
        },
        method="segment_profile_v1",
        description=f"Candidate segment profiles for {table}.",
        confidence=0.72,
        assumptions=(
            "These are candidate segments. Clustering has no ground truth, so a "
            "group here is a description of where the rows fell, not a discovered "
            "category that exists in the world.",
            "Feature deviations are in standard deviations from the overall mean, "
            "so they are comparable between features with different units.",
            "Categorical columns took no part in forming the groups; their "
            "distribution across the groups is therefore a finding about them, "
            "not a circular restatement of the input.",
            "Representative rows are the real records nearest each centre, not "
            "an average that may resemble nobody.",
        ),
    )


def embedding_evidence(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    table: str,
    seed: int,
) -> Evidence | None:
    """A two-dimensional projection, offered strictly as a visual aid.

    A scatter of coloured blobs is the single most persuasive image in a
    clustering report and the least trustworthy. Projecting to two dimensions
    discards variance, so groups that overlap on screen may be cleanly separated
    in the full space and groups that look distinct may be an artifact of the
    projection. The share of variance the two axes actually carry travels with
    the picture for that reason.
    """
    if matrix.size == 0 or matrix.shape[1] < 2 or matrix.shape[0] < 3:
        return None
    projector = PCA(n_components=2, random_state=seed).fit(matrix)
    projected = projector.transform(matrix)

    keep = np.arange(matrix.shape[0])
    if keep.size > MAX_EMBEDDING_POINTS:
        rng = np.random.default_rng(seed)
        keep = np.sort(
            rng.choice(matrix.shape[0], size=MAX_EMBEDDING_POINTS, replace=False)
        )
    points = [
        {
            "x": float(projected[position, 0]),
            "y": float(projected[position, 1]),
            "segment": int(labels[position]),
        }
        for position in keep
    ]
    explained = [
        stable(float(value)) or 0.0 for value in projector.explained_variance_ratio_
    ]
    return Evidence.create(
        kind="clustering_embedding",
        scope=EvidenceScope(table=table),
        value={
            "points": points,
            "point_count": len(points),
            "sampled": keep.size < matrix.shape[0],
            "explained_variance": explained,
            "captured_variance": float(sum(explained)),
            "segment_count": int(len(np.unique(labels))),
        },
        method="pca_projection_v1",
        description=f"Two-dimensional projection of {table} for display.",
        confidence=1.0,
        assumptions=(
            "A projection is a visual aid, never evidence that groups exist. "
            "Two dimensions cannot show what a higher-dimensional separation "
            "looks like.",
            "The captured-variance figure states how much of the structure the "
            "picture is able to show at all.",
        ),
    )


def silhouette_by_segment(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[int, float]:
    """Mean silhouette per segment: which groups are tight and which are filler."""
    from sklearn.metrics import silhouette_samples

    sample = distance_sample(matrix, seed=seed, limit=3_000)
    if sample.shape[0] != matrix.shape[0]:
        return {}
    try:
        scores = silhouette_samples(matrix, labels)
    except ValueError:  # pragma: no cover - degenerate partition
        return {}
    return {
        int(label): float(scores[labels == label].mean()) for label in np.unique(labels)
    }
