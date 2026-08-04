"""Clustering readiness and candidate segments.

Clustering is unfalsifiable by construction: ask for four groups and you get
four groups, from segmented data and from uniform noise alike, with a silhouette
score either way. So the load-bearing test in this file is not that segments are
found on the fixture — it is that **nothing** is found on noise, and that the
persuasive parts of the report are withheld when they would be unsupported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from examples.sample_data import SEGMENT_CENTRES, customer_segments
from prism_eda.evidence.models import OBSERVATION
from prism_eda.results import AnalysisStatus


def _noise(rows: int = 600, columns: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, (rows, columns)),
        columns=[f"f{index}" for index in range(columns)],
    )


def _blobs(rows: int = 400, seed: int = 1) -> pd.DataFrame:
    """Three well-separated spherical groups and nothing else."""
    rng = np.random.default_rng(seed)
    centres = [(0.0, 0.0), (10.0, 0.0), (5.0, 9.0)]
    parts = [
        pd.DataFrame(
            {
                "x": rng.normal(cx, 0.7, rows // 3),
                "y": rng.normal(cy, 0.7, rows // 3),
            }
        )
        for cx, cy in centres
    ]
    return pd.concat(parts, ignore_index=True)


def _run(frame: pd.DataFrame, **kwargs):
    return pe.load({"t": frame}).clustering(**kwargs)


def _kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _evidence(result, kind):
    return next(item for item in result.evidence if item.kind == kind)


def _titles(result) -> list[str]:
    return [finding.title for finding in result.findings]


# --------------------------------------------------------------------------
# The answer is allowed to be no
# --------------------------------------------------------------------------


def test_uniform_noise_reports_no_meaningful_structure() -> None:
    result = _run(_noise())
    assert result.status is AnalysisStatus.NO_MEANINGFUL_STRUCTURE
    assert "no stable cluster structure" in result.summary
    assert result.metadata["candidate_k"] is None
    assert result.metadata["structure_found"] is False


def test_no_segments_are_profiled_without_structure() -> None:
    """The most persuasive output is exactly the one that must be withheld."""
    result = _run(_noise())
    assert "clustering_segments" not in _kinds(result)
    assert "clustering_embedding" not in _kinds(result)
    assert "clustering_sensitivity" not in _kinds(result)


def test_noise_report_has_no_segment_section() -> None:
    html = _run(_noise()).render_html()
    assert 'id="section-segments"' not in html
    assert "No stable segment structure" in html


def test_noise_still_reports_why() -> None:
    """Silence about groups is not silence about the analysis."""
    result = _run(_noise())
    titles = _titles(result)
    assert any("no cluster tendency" in title.lower() for title in titles)
    assert any("stable partition" in title.lower() for title in titles)
    assert "clustering_tendency" in _kinds(result)
    assert "clustering_k_sweep" in _kinds(result)


def test_hopkins_separates_noise_from_structure() -> None:
    noise = _evidence(_run(_noise()), "clustering_tendency").value
    blobs = _evidence(_run(_blobs()), "clustering_tendency").value
    assert noise["verdict"] == "no_tendency"
    assert blobs["verdict"] == "clustered"
    assert blobs["hopkins_mean"] > noise["hopkins_mean"] + 0.2
    # Repeated, because one draw crosses a band boundary on its own.
    assert noise["repeats"] > 1


# --------------------------------------------------------------------------
# Real structure is found
# --------------------------------------------------------------------------


def test_planted_segments_are_recovered() -> None:
    result = _run(customer_segments())
    assert result.metadata["structure_found"]
    assert result.metadata["candidate_k"] == len(SEGMENT_CENTRES)
    segments = _evidence(result, "clustering_segments").value
    assert segments["segment_count"] == len(SEGMENT_CENTRES)
    # The planted shares are 34/30/18/18; recovery should be close.
    shares = sorted((item["share"] for item in segments["segments"]), reverse=True)
    expected = sorted((item[3] for item in SEGMENT_CENTRES.values()), reverse=True)
    for found, planted in zip(shares, expected, strict=True):
        assert abs(found - planted) < 0.05, (shares, expected)


def test_well_separated_blobs_are_found() -> None:
    result = _run(_blobs())
    assert result.status in {
        AnalysisStatus.COMPLETED,
        AnalysisStatus.COMPLETED_WITH_WARNINGS,
    }
    assert result.metadata["candidate_k"] == 3


def test_segments_are_described_in_standard_deviations() -> None:
    segments = _evidence(_run(customer_segments()), "clustering_segments").value
    for segment in segments["segments"]:
        for feature in segment["distinguishing_features"]:
            assert abs(feature["z"]) >= segments["distinguishing_threshold_z"]
            assert feature["direction"] in {"higher", "lower"}
            assert "segment_mean" in feature and "overall_mean" in feature


def test_representative_rows_are_real_records() -> None:
    frame = customer_segments()
    segments = _evidence(_run(frame), "clustering_segments").value
    for segment in segments["segments"]:
        assert segment["representatives"]
        for row in segment["representatives"]:
            assert int(row["row_index"]) in frame.index


def test_redundant_twins_are_not_listed_twice_per_segment() -> None:
    """One quantity present under two names is one fact, not two."""
    result = _run(customer_segments())
    segments = _evidence(result, "clustering_segments").value
    for segment in segments["segments"]:
        names = {item["feature"] for item in segment["distinguishing_features"]}
        assert not {"visits_per_month", "visits_per_year"} <= names, names


def test_embedding_states_how_much_it_can_show() -> None:
    embedding = _evidence(_run(customer_segments()), "clustering_embedding").value
    assert 0.0 < embedding["captured_variance"] <= 1.0
    assert embedding["points"]
    assert {point["segment"] for point in embedding["points"]}


# --------------------------------------------------------------------------
# Feature admission
# --------------------------------------------------------------------------


def test_identifier_and_constant_columns_are_excluded_with_reasons() -> None:
    result = _run(customer_segments())
    features = _evidence(result, "clustering_features").value
    reasons = {item["feature"]: item["reason"] for item in features["excluded"]}
    assert reasons["member_id"] == "identifier"
    assert reasons["account_status"] == "constant"
    assert "member_id" not in features["distance_features"]
    for item in features["excluded"]:
        assert item["detail"], "every exclusion must say why"


def test_categoricals_describe_the_groups_but_never_the_distance() -> None:
    result = _run(customer_segments())
    features = _evidence(result, "clustering_features").value
    assert "region" in features["profile_features"]
    assert "region" not in features["distance_features"]
    segments = _evidence(result, "clustering_segments").value
    columns = {
        entry["column"]
        for segment in segments["segments"]
        for entry in segment["categories"]
    }
    assert "region" in columns


def test_a_categorical_that_explains_nothing_is_reported() -> None:
    """Region is random in the fixture, and saying so is a real result."""
    result = _run(customer_segments())
    segments = _evidence(result, "clustering_segments").value
    assert "region" in segments["uninformative_categoricals"]
    assert any("do not line up with" in title for title in _titles(result))


def test_explicit_features_are_honoured() -> None:
    result = _run(customer_segments(), features=["annual_spend", "tenure_days"])
    features = _evidence(result, "clustering_features").value
    assert features["distance_features"] == ["annual_spend", "tenure_days"]


def test_unknown_requested_feature_warns() -> None:
    result = _run(customer_segments(), features=["annual_spend", "nope"])
    assert any(
        warning.code == "clustering_feature_not_found" for warning in result.warnings
    )


def test_requesting_an_identifier_still_excludes_it() -> None:
    result = _run(customer_segments(), features=["member_id", "annual_spend"])
    features = _evidence(result, "clustering_features").value
    assert "member_id" not in features["distance_features"]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_scale_difference_is_an_alert_not_a_defect() -> None:
    result = _run(customer_segments())
    matches = [
        finding
        for finding in result.findings
        if "differ enormously in scale" in finding.title
    ]
    assert len(matches) == 1
    assert matches[0].category == OBSERVATION
    features = _evidence(result, "clustering_features").value
    assert features["scale_ratio"] > 100


def test_redundant_pair_is_detected() -> None:
    redundancy = _evidence(_run(customer_segments()), "clustering_redundancy").value
    pair = redundancy["redundant_pairs"][0]
    assert {pair["left"], pair["right"]} == {"visits_per_month", "visits_per_year"}
    assert pair["abs_correlation"] > 0.99


def test_duplicate_rows_are_reported() -> None:
    result = _run(customer_segments())
    duplicates = _evidence(result, "clustering_duplicates").value
    assert duplicates["exact_duplicate_rows"] == 12
    assert any("duplicate rows" in title for title in _titles(result))


def test_high_dimensional_noise_reports_concentrated_distances() -> None:
    """The curse of dimensionality, measured rather than assumed."""
    result = _run(_noise(rows=600, columns=40))
    geometry = _evidence(result, "clustering_geometry").value
    assert geometry["distances_concentrated"]
    assert any("lost their contrast" in title for title in _titles(result))


def test_low_dimensional_structure_is_not_called_concentrated() -> None:
    geometry = _evidence(_run(_blobs()), "clustering_geometry").value
    assert not geometry["distances_concentrated"]


def test_contrast_is_not_destroyed_by_one_near_duplicate_pair() -> None:
    """A percentile-based contrast survives what a min/max ratio would not."""
    frame = _blobs()
    twin = frame.iloc[[0]].copy()
    twin["x"] += 1e-9
    frame = pd.concat([frame, twin], ignore_index=True)
    geometry = _evidence(_run(frame), "clustering_geometry").value
    assert geometry["relative_contrast"] is not None
    assert not geometry["distances_concentrated"]


# --------------------------------------------------------------------------
# Search and sensitivity
# --------------------------------------------------------------------------


def test_sweep_reports_stability_alongside_silhouette() -> None:
    sweep = _evidence(_run(customer_segments()), "clustering_k_sweep").value
    assert sweep["results"]
    for row in sweep["results"]:
        assert "silhouette" in row and "stability_mean" in row
        assert "calinski_harabasz" in row and "davies_bouldin" in row
    assert sweep["stability_repeats"] >= 1


def test_candidate_requires_both_separation_and_stability() -> None:
    sweep = _evidence(_run(_noise()), "clustering_k_sweep").value
    # Noise still produces a highest-silhouette k; it just never earns candidacy.
    assert sweep["highest_silhouette_k"] is not None
    assert sweep["candidate_k"] is None


def test_feature_dropout_reveals_a_dominant_feature() -> None:
    """One feature carrying the grouping is a threshold, not a segmentation."""
    rng = np.random.default_rng(4)
    rows = 450
    driver = np.concatenate(
        [
            rng.normal(0, 0.6, rows // 3),
            rng.normal(9, 0.6, rows // 3),
            rng.normal(18, 0.6, rows // 3),
        ]
    )
    frame = pd.DataFrame(
        {
            "driver": driver,
            "noise_a": rng.normal(0, 1, len(driver)),
            "noise_b": rng.normal(0, 1, len(driver)),
        }
    )
    result = _run(frame)
    sensitivity = _evidence(result, "clustering_sensitivity").value
    assert "driver" in sensitivity["dominant_features"]
    assert any("essentially driver" in title for title in _titles(result))


def test_sensitivity_reports_the_scaling_choice() -> None:
    sensitivity = _evidence(_run(customer_segments()), "clustering_sensitivity").value
    assert sensitivity["scaling_agreement"] is not None
    assert 0.0 <= sensitivity["scaling_agreement"] <= 1.0


def test_algorithm_guidance_mentions_excluded_categoricals() -> None:
    guidance = _evidence(
        _run(customer_segments()), "clustering_algorithm_guidance"
    ).value
    approaches = " ".join(item["algorithm"] for item in guidance["suggestions"])
    assert "Gower" in approaches or "prototypes" in approaches


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_analysis_does_not_mutate_the_caller_frame() -> None:
    frame = customer_segments()
    before = frame.copy(deep=True)
    pe.load({"members": frame}).clustering()
    pd.testing.assert_frame_equal(frame, before)


def test_repeated_runs_are_byte_identical() -> None:
    first = _run(customer_segments())
    second = _run(customer_segments())
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert first.to_dict() == second.to_dict()


def test_findings_cite_real_evidence() -> None:
    result = _run(customer_segments())
    ids = {item.id for item in result.evidence}
    for finding in result.findings:
        assert finding.evidence_ids
        assert set(finding.evidence_ids) <= ids


def test_ambiguous_table_asks_which_one() -> None:
    frame = _blobs()
    result = pe.load({"a": frame, "b": frame.copy()}).clustering()
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "clustering_table_ambiguous" for warning in result.warnings
    )


def test_explicit_table_resolves_the_ambiguity() -> None:
    frame = _blobs()
    result = pe.load({"a": frame, "b": frame.copy()}).clustering(table="a")
    assert result.status is not AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_too_few_rows_is_insufficient_evidence() -> None:
    result = _run(_blobs(rows=12))
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_no_numeric_features_is_insufficient_evidence() -> None:
    frame = pd.DataFrame({"a": ["x", "y"] * 40, "b": ["p", "q"] * 40})
    result = _run(frame)
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_sampling_is_recorded_when_the_budget_bites() -> None:
    frame = _blobs(rows=16_000)
    result = _run(frame, mode="quick")
    record = next(
        item for item in result.sampling if item.operation == "clustering_analysis"
    )
    assert record.sampled_rows == 5_000
    assert any(
        warning.code == "sampled_clustering_analysis" for warning in result.warnings
    )


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(TypeError):
        pe.load({"t": _blobs()}).analyze("clustering", nonsense=1)


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def test_exports_round_trip(tmp_path) -> None:
    result = _run(customer_segments())
    html = result.to_html(tmp_path / "clustering.html")
    payload = result.to_json(tmp_path / "clustering.json")
    assert html.exists() and payload.exists()
    text = html.read_text(encoding="utf-8")
    assert "Segments" in text
    assert "http://" not in text.replace("http://www.w3.org", "")


def test_report_names_the_clustering_sections() -> None:
    text = _run(customer_segments()).render_html()
    for anchor in ("section-segments", "section-search", "section-clusterability"):
        assert f'id="{anchor}"' in text


def test_api_and_dataset_entry_points_agree() -> None:
    frame = customer_segments()
    direct = pe.clustering({"members": frame})
    session = pe.load({"members": frame}).clustering()
    assert direct.summary == session.summary
