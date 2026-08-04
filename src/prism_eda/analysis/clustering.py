"""Goal-aware deterministic diagnostics for clustering EDA.

Clustering is the one recipe where the honest answer is often *no*. Every other
task here has a target to be right or wrong about; this one does not, and the
algorithms will not tell you. Ask k-means for four groups and it returns four
groups whether the data is segmented or uniform, and the resulting silhouette
score, sizes, and profile all look equally convincing either way.

So the orchestration is a gate rather than a pipeline. Segment profiles — the
most persuasive output in the report — are produced only when two independent
checks agree that there is something to describe: the data must show cluster
tendency against uniform noise in its own bounding box, *and* a partition must
reproduce on resampled rows. When either fails, the run finishes with
``NO_MEANINGFUL_STRUCTURE`` and says so, which is a first-class result and not a
failure to find one.

The issue-versus-alert split follows the same rule as the other recipes. A
constant column in the feature set is a defect. An identifier in it is a defect.
Four groups being found is not a defect, and neither is finding none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.cluster import KMeans

from prism_eda.analysis._clustering import (
    MIN_ROWS,
    admit_features,
    build_matrix,
    resolve_table,
    sample_frame,
)
from prism_eda.analysis.clustering_readiness import (
    CONCENTRATION_SEVERE,
    SCALE_RATIO_SEVERE,
    SCALE_RATIO_WARN,
    duplicate_evidence,
    feature_evidence,
    geometry_evidence,
    profile_categoricals,
    redundancy_evidence,
    tendency_evidence,
)
from prism_eda.analysis.clustering_search import (
    algorithm_evidence,
    sensitivity_evidence,
    sweep_evidence,
)
from prism_eda.analysis.clustering_segments import (
    embedding_evidence,
    segment_evidence,
    silhouette_by_segment,
)
from prism_eda.artifacts import Artifact
from prism_eda.catalog.models import DatasetCatalog
from prism_eda.config import AnalysisConfig, AnalysisContext, AnalysisMode
from prism_eda.events import Event, EventCallback, EventKind, emit
from prism_eda.evidence.models import (
    OBSERVATION,
    Evidence,
    Finding,
    sort_findings,
    split_findings,
)
from prism_eda.results import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisWarning,
    SamplingRecord,
)
from prism_eda.transformations.models import TransformationPlan, TransformationStep

#: Exact-duplicate share above which repeated rows are bending the geometry.
DUPLICATE_RATE_WARN = 0.01

#: Missingness in a distance feature above which imputation is doing the work.
MISSING_RATE_WARN = 0.1

#: A segment smaller than this is a handful of points rather than a group.
TINY_SEGMENT_RATE = 0.02


def _find(evidence: list[Evidence], kind: str) -> Evidence | None:
    for item in evidence:
        if item.kind == kind:
            return item
    return None


def _findings_and_steps(
    evidence: list[Evidence],
) -> tuple[list[Finding], list[TransformationStep]]:
    """Turn evidence into a short, prioritized, decision-first list."""
    findings: list[Finding] = []
    steps: list[TransformationStep] = []

    for item in evidence:
        value = item.value
        table = item.scope.table or ""

        if item.kind == "clustering_features":
            for excluded in value["excluded"]:
                if excluded["reason"] not in {"identifier", "constant"}:
                    continue
                findings.append(
                    Finding.create(
                        title=(
                            f"{excluded['reason'].capitalize()} column excluded: "
                            f"{excluded['feature']}"
                        ),
                        summary=excluded["detail"],
                        severity="medium",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "It was kept out of the distance automatically. If it "
                            "was meant to matter, it needs a derived feature that "
                            "carries what you actually mean."
                        ),
                    )
                )
            ratio = value["scale_ratio"]
            if ratio is not None and ratio >= SCALE_RATIO_WARN:
                findings.append(
                    Finding.create(
                        title="Features differ enormously in scale",
                        summary=(
                            f"{value['widest_feature']} spans {ratio:,.0f}x the "
                            f"range of {value['narrowest_feature']}. Without "
                            "standardizing, the distance between two rows is "
                            "essentially the difference in one column."
                        ),
                        severity="medium" if ratio >= SCALE_RATIO_SEVERE else "low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Prism standardizes before clustering. Confirm your "
                            "own pipeline does too, and see how much the answer "
                            "moves in the sensitivity check."
                        ),
                        category=OBSERVATION,
                    )
                )
            if value["max_missing_rate"] > MISSING_RATE_WARN:
                findings.append(
                    Finding.create(
                        title="Missing values in the distance features",
                        summary=(
                            f"Up to {value['max_missing_rate']:.0%} of one feature "
                            "is missing and was median-imputed to place those rows "
                            "in space."
                        ),
                        severity="medium",
                        confidence=1.0,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "An imputed coordinate is a guess about where a row "
                            "sits. Rows with many imputed features drift toward "
                            "the centre and can form a group of their own."
                        ),
                    )
                )

        elif item.kind == "clustering_redundancy":
            top = value["redundant_pairs"][0]
            findings.append(
                Finding.create(
                    title="The same quantity is present twice",
                    summary=(
                        f"{top['left']} and {top['right']} correlate at "
                        f"{top['abs_correlation']:.4f}, so that quantity carries "
                        "double weight in the distance."
                    ),
                    severity="medium",
                    confidence=item.confidence,
                    evidence_ids=(item.id,),
                    recommendation=(
                        "Drop one of each pair, or accept that you have weighted "
                        "it twice on purpose."
                    ),
                    category=OBSERVATION,
                )
            )
            steps.append(
                TransformationStep(
                    operation="review_redundant_clustering_features",
                    table=table,
                    columns=tuple(
                        {pair["left"] for pair in value["redundant_pairs"]}
                        | {pair["right"] for pair in value["redundant_pairs"]}
                    ),
                    parameters={"pairs": value["redundant_pairs"]},
                    rationale="A duplicated dimension is double-weighted.",
                    evidence_ids=(item.id,),
                    risk="low",
                )
            )

        elif item.kind == "clustering_duplicates":
            if value["exact_duplicate_rate"] > DUPLICATE_RATE_WARN:
                findings.append(
                    Finding.create(
                        title=f"{value['exact_duplicate_rows']:,} duplicate rows",
                        summary=(
                            f"{value['exact_duplicate_rate']:.1%} of rows are exact "
                            "repeats, which weights their location in space more "
                            "than once."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "De-duplicate before clustering unless the repeats are "
                            "genuinely separate observations that happen to match."
                        ),
                    )
                )

        elif item.kind == "clustering_geometry":
            if value["distances_concentrated"]:
                findings.append(
                    Finding.create(
                        title="Distances have lost their contrast",
                        summary=(
                            f"Across {value['feature_count']} features the spread "
                            "of pairwise distances is only "
                            f"{value['relative_contrast']:.2f}x the typical short "
                            "distance, so every point sits about as far from every "
                            "other."
                        ),
                        severity="high"
                        if value["relative_contrast"] is not None
                        and value["relative_contrast"] < CONCENTRATION_SEVERE
                        else "medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Distance-based clustering cannot discriminate here. "
                            "Reduce to the directions that carry the variance "
                            "before clustering, and treat any groups found in the "
                            "full space with suspicion."
                        ),
                    )
                )
            intrinsic = value["intrinsic_dimensionality"]
            if (
                intrinsic is not None
                and value["feature_count"] > 2
                and intrinsic < value["feature_count"] / 2
            ):
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['feature_count']} features occupy about "
                            f"{intrinsic} dimensions"
                        ),
                        summary=(
                            "Ninety percent of the variance fits in "
                            f"{intrinsic} principal component(s); the rest of the "
                            "feature set is largely restating them."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Reducing first usually sharpens the geometry rather "
                            "than losing information."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "clustering_tendency":
            if value["verdict"] == "no_tendency":
                findings.append(
                    Finding.create(
                        title="The data shows no cluster tendency",
                        summary=(
                            f"Hopkins averaged {value['hopkins_mean']:.2f} over "
                            f"{value['repeats']} samples, where about 0.5 means "
                            "indistinguishable from uniform noise in the same "
                            "bounding box."
                        ),
                        severity="high",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Any partition of this data will still return groups "
                            "with sizes and scores. They would be a division of "
                            "something continuous, not a discovery."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding.create(
                        title=(
                            "The data is more clustered than chance"
                            if value["verdict"] == "clustered"
                            else "Weak cluster tendency"
                        ),
                        summary=(
                            f"Hopkins averaged {value['hopkins_mean']:.2f} "
                            f"(range {value['hopkins_min']:.2f}-"
                            f"{value['hopkins_max']:.2f}) over "
                            f"{value['repeats']} samples."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Tendency says points bunch together somewhere. It "
                            "does not say how many groups there are."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "clustering_k_sweep":
            candidate = value["candidate_k"]
            if candidate is None:
                findings.append(
                    Finding.create(
                        title="No cluster count produced a stable partition",
                        summary=(
                            "Across k="
                            f"{min(value['evaluated_k'])}-{max(value['evaluated_k'])}, "
                            "no partition was both separated enough and "
                            "reproducible on resampled rows. The best silhouette "
                            f"was {value['highest_silhouette']:.2f} at k="
                            f"{value['highest_silhouette_k']}."
                        ),
                        severity="high",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Report that no stable segmentation was found. A "
                            "partition chosen anyway will not reproduce on next "
                            "quarter's data."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding.create(
                        title=f"{candidate} groups reproduce on resampled rows",
                        summary=(
                            f"k={candidate} scores {value['candidate_silhouette']:.2f} "
                            "on silhouette and agrees with itself at "
                            f"{value['candidate_stability']:.2f} across "
                            f"{value['stability_repeats']} resamples."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A candidate, not a determination. Other cluster "
                            "counts in the sweep may be equally defensible for a "
                            "different purpose."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "clustering_sensitivity":
            if value["scaling_changes_answer"]:
                findings.append(
                    Finding.create(
                        title="Standardizing changes which rows group together",
                        summary=(
                            "Scaled and unscaled clusterings agree at only "
                            f"{value['scaling_agreement']:.2f} adjusted Rand index."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "The groups are partly an artifact of a preprocessing "
                            "choice. State which convention you used."
                        ),
                    )
                )
            if value["dominant_features"]:
                names = ", ".join(value["dominant_features"])
                findings.append(
                    Finding.create(
                        title=f"The grouping is essentially {names}",
                        summary=(
                            "Removing it changes the partition beyond recognition, "
                            "so the other features are contributing little."
                        ),
                        severity="medium",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A one-feature grouping is a threshold on that "
                            "feature. Say so plainly rather than presenting it as "
                            "multivariate segmentation."
                        ),
                        category=OBSERVATION,
                    )
                )

        elif item.kind == "clustering_segments":
            tiny = [
                segment
                for segment in value["segments"]
                if segment["share"] < TINY_SEGMENT_RATE
            ]
            if tiny:
                findings.append(
                    Finding.create(
                        title=f"{len(tiny)} segment(s) hold almost no rows",
                        summary=(
                            f"The smallest holds {tiny[0]['size']:,} row(s) "
                            f"({tiny[0]['share']:.1%}), which is a handful of "
                            "points rather than a group."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "k-means must place every point somewhere; a tiny "
                            "group is often where the outliers went."
                        ),
                        category=OBSERVATION,
                    )
                )
            if value["uninformative_categoricals"]:
                names = ", ".join(value["uninformative_categoricals"])
                findings.append(
                    Finding.create(
                        title=f"The groups do not line up with {names}",
                        summary=(
                            "No segment is meaningfully over-represented in any "
                            f"value of {names}, so that column describes "
                            "something orthogonal to the grouping."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "Useful to know before naming the segments after it. "
                            "The grouping was formed without it, so this is a "
                            "real result rather than a circular one."
                        ),
                        category=OBSERVATION,
                    )
                )
            if value["undistinguished_count"]:
                findings.append(
                    Finding.create(
                        title=(
                            f"{value['undistinguished_count']} segment(s) have no "
                            "distinguishing feature"
                        ),
                        summary=(
                            "Their averages sit within half a standard deviation "
                            "of the overall average on every feature, so there is "
                            "nothing to describe them by."
                        ),
                        severity="low",
                        confidence=item.confidence,
                        evidence_ids=(item.id,),
                        recommendation=(
                            "A group you cannot characterise is hard to act on, "
                            "whatever its silhouette score."
                        ),
                        category=OBSERVATION,
                    )
                )

    return sort_findings(findings), steps


def _artifacts(evidence: list[Evidence]) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []

    sweep = _find(evidence, "clustering_k_sweep")
    if sweep is not None:
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Cluster count search",
                data={
                    "columns": [
                        {"key": "k", "label": "k"},
                        {"key": "silhouette", "label": "Silhouette"},
                        {"key": "stability", "label": "Stability (ARI)"},
                        {"key": "calinski", "label": "Calinski-Harabasz"},
                        {"key": "davies", "label": "Davies-Bouldin"},
                        {"key": "smallest", "label": "Smallest group"},
                    ],
                    "rows": [
                        {
                            "k": str(row["k"]),
                            "silhouette": f"{row['silhouette']:.3f}",
                            "stability": (
                                f"{row['stability_mean']:.3f}"
                                if row["stability_mean"] is not None
                                else "n/a"
                            ),
                            "calinski": f"{row['calinski_harabasz']:,.0f}",
                            "davies": f"{row['davies_bouldin']:.3f}",
                            "smallest": f"{row['smallest_cluster_rate']:.1%}",
                        }
                        for row in sweep.value["results"]
                    ],
                },
                evidence_ids=(sweep.id,),
                metadata={
                    "description": (
                        "Silhouette, Calinski-Harabasz and Davies-Bouldin judge "
                        "the partition they are computed from, so they measure "
                        "tidiness. Stability is the adjusted Rand index between "
                        "two independently clustered subsamples — the only column "
                        "here that can fail."
                    )
                },
            )
        )

    features = _find(evidence, "clustering_features")
    if features is not None and features.value["excluded"]:
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Excluded columns",
                data={
                    "columns": [
                        {"key": "feature", "label": "Column"},
                        {"key": "reason", "label": "Reason"},
                        {"key": "detail", "label": "Why it matters"},
                    ],
                    "rows": [
                        {
                            "feature": row["feature"],
                            "reason": row["reason"].replace("_", " "),
                            "detail": row["detail"],
                        }
                        for row in features.value["excluded"]
                    ],
                },
                evidence_ids=(features.id,),
                metadata={
                    "description": (
                        "A silently dropped column is indistinguishable from a "
                        "column that contributed nothing, so every exclusion "
                        "carries its reason."
                    )
                },
            )
        )

    guidance = _find(evidence, "clustering_algorithm_guidance")
    if guidance is not None:
        artifacts.append(
            Artifact.create(
                kind="metric_table",
                title="Other approaches worth trying",
                data={
                    "columns": [
                        {"key": "algorithm", "label": "Approach"},
                        {"key": "because", "label": "Because"},
                    ],
                    "rows": guidance.value["suggestions"],
                },
                evidence_ids=(guidance.id,),
                metadata={
                    "description": (
                        "k-means drove the search because it is fast and "
                        "deterministic under a fixed seed, not because it is the "
                        "right model for this data."
                    )
                },
            )
        )
    return tuple(artifacts)


def _summary(
    table: str,
    findings: list[Finding],
    *,
    structure: bool,
    candidate_k: int | None,
    has_warnings: bool,
) -> str:
    suffix = " Sampling or recoverable caveats apply." if has_warnings else ""
    issues, observations = split_findings(findings)
    alerts = (
        f" {len(observations)} alert(s) describe the geometry." if observations else ""
    )

    if not structure:
        return (
            f"{table}: no stable cluster structure was found. Partitioning it "
            f"anyway would divide something continuous.{alerts}{suffix}"
        )
    if not issues:
        return (
            f"{table}: {candidate_k} candidate segments reproduce on resampled "
            f"rows, and nothing blocks acting on them.{alerts}{suffix}"
        )
    order = ("critical", "high", "medium", "low")
    counts = dict.fromkeys(order, 0)
    for finding in issues:
        if finding.severity in counts:
            counts[finding.severity] += 1
    breakdown = ", ".join(f"{counts[key]} {key}" for key in order if counts[key])
    top = issues[0]
    return (
        f"{table}: {candidate_k} candidate segments, but review first. "
        f"Top issue — {top.title}. {len(issues)} prioritized issue(s) "
        f"({breakdown}).{alerts}{suffix}"
    )


def clustering_dataset(
    tables: Mapping[str, pd.DataFrame],
    catalog: DatasetCatalog,
    *,
    context: AnalysisContext,
    config: AnalysisConfig,
    features: Sequence[str] | None = None,
    table: str | None = None,
    callbacks: tuple[EventCallback, ...] = (),
) -> AnalysisResult:
    """Run deterministic clustering readiness and segment diagnostics."""
    emit(
        callbacks,
        Event(
            EventKind.RUN_STARTED,
            "Clustering diagnostics started.",
            stage="clustering",
        ),
    )
    warnings: list[AnalysisWarning] = []
    sampling: list[SamplingRecord] = []
    table_catalog = resolve_table(catalog, tables, table=table, warnings=warnings)

    evidence: list[Evidence] = []
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    usable = False
    structure = False
    candidate_k: int | None = None
    seed = config.random_seed

    if table_catalog is not None:
        frame = sample_frame(
            tables[table_catalog.name],
            table=table_catalog.name,
            config=config,
            warnings=warnings,
            sampling=sampling,
        )
        selected = admit_features(
            frame, table_catalog, features=features, warnings=warnings
        )
        evidence.append(feature_evidence(frame, selected, table=table_catalog.name))

        matrix, index = build_matrix(frame, selected.numeric)
        unscaled, _ = build_matrix(frame, selected.numeric, scale=False)
        usable = matrix.shape[0] >= MIN_ROWS and matrix.shape[1] >= 1

        if usable:
            redundancy = redundancy_evidence(frame, selected, table=table_catalog.name)
            if redundancy is not None:
                evidence.append(redundancy)
            duplicates = duplicate_evidence(
                frame, matrix, selected, table=table_catalog.name, seed=seed
            )
            if duplicates is not None:
                evidence.append(duplicates)
            geometry = geometry_evidence(
                matrix, selected, table=table_catalog.name, seed=seed
            )
            if geometry is not None:
                evidence.append(geometry)
            tendency = tendency_evidence(matrix, table=table_catalog.name, seed=seed)
            if tendency is not None:
                evidence.append(tendency)

            sweep = sweep_evidence(
                matrix,
                selected,
                table=table_catalog.name,
                seed=seed,
                mode=config.mode,
            )
            if sweep is not None:
                evidence.append(sweep)
                candidate_k = sweep.value["candidate_k"]

            # The gate. Two independent checks have to agree before the most
            # persuasive part of the report gets built, because a segment
            # profile computed from noise looks exactly as convincing as one
            # computed from structure.
            has_tendency = (
                tendency is None or tendency.value["verdict"] != "no_tendency"
            )
            structure = bool(candidate_k) and has_tendency
            if candidate_k and not has_tendency:
                warnings.append(
                    AnalysisWarning(
                        code="clustering_tendency_contradicts_partition",
                        message=(
                            f"A k={candidate_k} partition was reproducible, but the "
                            "data shows no cluster tendency against uniform noise. "
                            "Segment profiles were not produced, because a stable "
                            "way of cutting something continuous is still a cut "
                            "through something continuous."
                        ),
                        table=table_catalog.name,
                    )
                )

            if structure and candidate_k:
                labels = KMeans(
                    n_clusters=candidate_k, n_init=10, random_state=seed
                ).fit_predict(matrix)
                sensitivity = sensitivity_evidence(
                    matrix,
                    unscaled,
                    selected,
                    table=table_catalog.name,
                    seed=seed,
                    k=candidate_k,
                )
                if sensitivity is not None:
                    evidence.append(sensitivity)
                segments = segment_evidence(
                    frame,
                    matrix,
                    labels,
                    index,
                    selected,
                    profile_categoricals(frame, selected),
                    table=table_catalog.name,
                    redundant_pairs=(
                        redundancy.value["redundant_pairs"]
                        if redundancy is not None
                        else None
                    ),
                )
                if segments is not None:
                    quality = silhouette_by_segment(matrix, labels, seed=seed)
                    for entry in segments.value["segments"]:
                        entry["mean_silhouette"] = quality.get(entry["segment"])
                    evidence.append(segments)
                embedding = embedding_evidence(
                    matrix, labels, table=table_catalog.name, seed=seed
                )
                if embedding is not None:
                    evidence.append(embedding)

            smallest = None
            if sweep is not None and candidate_k:
                for row in sweep.value["results"]:
                    if row["k"] == candidate_k:
                        smallest = row["smallest_cluster_rate"]
            evidence.append(
                algorithm_evidence(
                    selected,
                    table=table_catalog.name,
                    concentrated=bool(
                        geometry is not None
                        and geometry.value["distances_concentrated"]
                    ),
                    intrinsic_dimensionality=(
                        geometry.value["intrinsic_dimensionality"]
                        if geometry is not None
                        else None
                    ),
                    feature_count=len(selected.numeric),
                    smallest_cluster_rate=smallest,
                    categorical_count=len(selected.categorical),
                )
            )
        findings, steps = _findings_and_steps(evidence)

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

    if table_catalog is None:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = "Clustering analysis needs one table with numeric features."
    elif not usable:
        status = (
            AnalysisStatus.COMPLETED_WITH_WARNINGS
            if config.allow_insufficient_evidence
            else AnalysisStatus.INSUFFICIENT_EVIDENCE
        )
        summary = (
            f"{table_catalog.name} does not have enough usable numeric features "
            f"or rows for clustering diagnostics (at least {MIN_ROWS} rows and one "
            "numeric feature are needed)."
        )
    elif not structure:
        # A first-class result. The checks ran, they were conclusive, and the
        # conclusion is that there is nothing stable to describe.
        status = AnalysisStatus.NO_MEANINGFUL_STRUCTURE
        summary = _summary(
            table_catalog.name,
            findings,
            structure=False,
            candidate_k=None,
            has_warnings=bool(warnings),
        )
    elif warnings:
        status = AnalysisStatus.COMPLETED_WITH_WARNINGS
        summary = _summary(
            table_catalog.name,
            findings,
            structure=True,
            candidate_k=candidate_k,
            has_warnings=True,
        )
    else:
        status = AnalysisStatus.COMPLETED
        summary = _summary(
            table_catalog.name,
            findings,
            structure=True,
            candidate_k=candidate_k,
            has_warnings=False,
        )

    result = AnalysisResult(
        goal="clustering",
        status=status,
        summary=summary,
        catalog=catalog,
        findings=tuple(findings),
        evidence=tuple(evidence),
        artifacts=_artifacts(evidence),
        assumptions=context.assumptions,
        warnings=tuple(warnings),
        sampling=tuple(sampling),
        transformation_plan=TransformationPlan(tuple(steps)),
        metadata={
            "mode": AnalysisMode(config.mode).value,
            "sampling": config.sampling,
            "random_seed": config.random_seed,
            "selected_table": table_catalog.name if table_catalog else table,
            "candidate_k": candidate_k,
            "structure_found": structure,
        },
    )
    emit(
        callbacks,
        Event(
            EventKind.RUN_COMPLETED,
            result.summary,
            stage="clustering",
            progress=1.0,
            data={"status": result.status.value},
        ),
    )
    return result
