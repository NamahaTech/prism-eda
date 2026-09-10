"""Tests for what the feature plan reports, rather than what it computes.

The rule this file exists to enforce is the repository's own: a detector that
never stays quiet is not a detector. Every check here is asserted twice -- once
against a clean, deliberately varied feature set where it must produce nothing,
and once against a feature set with the specific defect it exists to find. A
check that only ever fires is indistinguishable from a check that is broken,
and it is worse than absent, because it trains the reader to skip the section.

The thresholds these detectors use were calibrated by measuring the clean case
first: the highest correlation between any two of the fifteen features below is
0.92, and the costliest holds 12% of runtime. Both detectors sit far above
those readings, which is why the silence tests pass by a margin rather than by
a hair.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from prism_eda.evidence.models import split_findings
from prism_eda.features import FeatureSet, safe_div
from prism_eda.results import AnalysisStatus
from tests.test_features import _featureset, _txns


def _titles(result) -> list[str]:
    return [finding.title for finding in result.findings]


def _kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _report(fs: FeatureSet, frame: pd.DataFrame):
    return fs.plan(frame, verify=False).diagnose(frame)


# ---------------------------------------------------------------------------
# Silence on a clean feature set
# ---------------------------------------------------------------------------
def test_a_clean_feature_set_produces_no_findings_at_all() -> None:
    result = _report(_featureset(), _txns(rows=9000, entities=450, seed=13))
    issues, observations = split_findings(result.findings)
    assert issues == [], [item.title for item in issues]
    assert observations == [], [item.title for item in observations]
    assert "Nothing in the definitions or the results looks wrong" in result.summary


def test_a_clean_feature_set_still_produces_evidence() -> None:
    """Silence in the findings is not silence in the evidence."""
    result = _report(_featureset(), _txns(rows=9000, entities=450, seed=13))
    assert "feature_plan_sharing" in _kinds(result)
    assert result.status is AnalysisStatus.COMPLETED
    assert result.metadata["cost"], "the cost table is still produced"


def test_measured_time_is_never_banked_as_evidence() -> None:
    """An evidence id hashes its value, so a duration would break lineage.

    Wall-clock time is a property of the machine that ran the plan, not of the
    data. Banking it would give the same analysis a different lineage on every
    run, which is why cost is reported as run metadata instead.
    """
    frame = _txns()
    first = _featureset().plan(frame, verify=False).diagnose(frame)
    second = _featureset().plan(frame, verify=False).diagnose(frame)

    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    banked = json.dumps([item.value for item in first.evidence])
    for suspect in ("seconds", "milliseconds", "elapsed"):
        assert suspect not in banked
    # Reported, just not as evidence.
    assert first.metadata["cost"][0]["milliseconds"] >= 0.0


def test_findings_cite_real_evidence() -> None:
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def txn_count(all):
        return all.count()

    @fs.feature
    def n_transactions(all):
        return all.count()

    result = _report(fs, _txns())
    ids = {item.id for item in result.evidence}
    for finding in result.findings:
        if finding.evidence_ids:
            assert set(finding.evidence_ids) <= ids


# ---------------------------------------------------------------------------
# Each detector, on the defect it exists to find
# ---------------------------------------------------------------------------
def test_two_names_for_one_computation_are_reported_structurally() -> None:
    """No false positives are possible: the two compile to the same node."""
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def txn_count(all):
        return all.count()

    @fs.feature
    def n_transactions(all):
        return all.count()

    result = _report(fs, _txns())
    issues, observations = split_findings(result.findings)
    assert len(issues) == 1
    assert "are the same feature" in issues[0].title
    # The structural claim strictly dominates a correlation on this data, so
    # the same pair must not also be filed as an observation.
    assert observations == [], [item.title for item in observations]


def test_a_constant_feature_is_reported() -> None:
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def always_one(all):
        return safe_div(all.count(), all.count())

    @fs.feature
    def amt_mean(amount, all):
        return all.mean(amount)

    result = _report(fs, _txns())
    assert any("same for every entity" in title for title in _titles(result))


def test_a_feature_that_mostly_falls_back_is_reported() -> None:
    """A fallback is a failed computation, not a value equal to the default."""
    fs = FeatureSet(entity="account", time="ts")
    fs.window("w1d", days=1)

    @fs.feature
    def spread_1d(amount, w1d):
        return w1d.std(amount)

    @fs.feature
    def amt_mean(amount, all):
        return all.mean(amount)

    result = _report(fs, _txns())
    issues, _ = split_findings(result.findings)
    assert any("falls back to its default" in item.title for item in issues)
    assert "feature_default_fallbacks" in _kinds(result)


def test_a_feature_reading_the_target_is_critical() -> None:
    fs = FeatureSet(entity="account", time="ts", target="amount")

    @fs.feature
    def mean_target(amount, all):
        return all.mean(amount)

    result = _report(fs, _txns())
    issues, _ = split_findings(result.findings)
    assert issues[0].severity == "critical"
    assert "reads the target column" in issues[0].title


def test_rows_after_the_decision_moment_are_reported() -> None:
    frame = _txns()
    leaky = frame.assign(decided_at=frame["ts"] - pd.Timedelta(days=30))
    fs = FeatureSet(entity="account", time="ts", reference_time="decided_at")
    fs.window("w7d", days=7)

    @fs.feature
    def n7(w7d):
        return w7d.count()

    result = _report(fs, leaky)
    issues, _ = split_findings(result.findings)
    assert any("after the decision moment" in item.title for item in issues)
    assert "feature_future_rows" in _kinds(result)


def test_a_clean_extract_reports_no_future_rows() -> None:
    """The same check, on a frame that was filtered properly."""
    frame = _txns()
    clean = frame.assign(decided_at=frame["ts"] + pd.Timedelta(days=1))
    fs = FeatureSet(entity="account", time="ts", reference_time="decided_at")
    fs.window("w7d", days=7)

    @fs.feature
    def n7(w7d):
        return w7d.count()

    result = _report(fs, clean)
    assert "feature_future_rows" not in _kinds(result)
    assert split_findings(result.findings)[0] == []


def test_an_undeclared_decision_moment_is_an_assumption_not_a_finding() -> None:
    """It is true of every plan without one, so it cannot be a finding."""
    result = _report(_featureset(), _txns())
    assert result.findings == ()
    assert any("anchored at each entity" in item for item in result.assumptions)


def test_an_opaque_feature_is_named_as_unplanned() -> None:
    fs = FeatureSet(entity="account", time="ts")

    @fs.opaque(requires=("amount",))
    def weird(group: pd.DataFrame) -> float:
        return float(group["amount"].to_numpy()[::3].sum())

    @fs.feature
    def amt_mean(amount, all):
        return all.mean(amount)

    _, observations = split_findings(_report(fs, _txns()).findings)
    assert any("run unplanned" in item.title for item in observations)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_the_report_is_self_contained(tmp_path) -> None:
    run = _featureset().plan(_txns()).run(_txns())
    html_path = run.report.to_html(tmp_path / "features.html")
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "http://" not in text.replace("http://www.w3.org", "")
    assert "https://" not in text
    assert "<style>" in text


def test_report_sections_are_numbered_sequentially_and_once_each() -> None:
    """The numbering is hand-written in this template, so it is pinned here."""
    frame = _txns()
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def always_one(all):  # produces an observation, so the alerts section exists
        return safe_div(all.count(), all.count())

    @fs.feature
    def amt_mean(amount, all):
        return all.mean(amount)

    for feature_set in (fs, _featureset()):
        html = feature_set.plan(frame, verify=False).diagnose(frame).render_html()
        numbers = re.findall(r'<div class="sec-idx">(\d+)</div>', html)
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers)), numbers
        assert [int(n) for n in numbers] == list(range(1, len(numbers) + 1))
        ids = re.findall(r'id="(section-[a-z]+)"', html)
        assert len(ids) == len(set(ids))
        assert len(ids) == len(numbers)


def test_the_report_round_trips_through_json(tmp_path) -> None:
    run = _featureset().plan(_txns()).run(_txns())
    written = run.report.to_json(tmp_path / "features.json")
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["goal"] == "feature_plan"
    assert loaded["contract"]["alphabetical_sorting"] is False
    assert len(loaded["contract"]["features"]) == 15


def test_the_run_returns_the_features_and_the_report_together() -> None:
    frame = _txns()
    run = _featureset().plan(frame).run(frame)
    assert list(run.features.columns) == list(run.report.contract.names)
    assert len(run.features) == run.report.metadata["entity_count"]
    run.report.contract.validate_output(run.features)


# ---------------------------------------------------------------------------
# Entry points and determinism
# ---------------------------------------------------------------------------
def test_api_and_dataset_entry_points_agree() -> None:
    import prism_eda as pe

    frame = _txns()
    through_api = pe.features({"txns": frame}, _featureset())
    through_dataset = pe.load({"txns": frame}).features(_featureset())
    assert through_api.report.summary == through_dataset.report.summary
    pd.testing.assert_frame_equal(through_api.features, through_dataset.features)


def test_repeated_runs_are_identical() -> None:
    frame = _txns()
    first = _featureset().plan(frame).run(frame)
    second = _featureset().plan(frame).run(frame)
    assert [i.id for i in first.report.evidence] == [
        i.id for i in second.report.evidence
    ]
    pd.testing.assert_frame_equal(first.features, second.features)


def test_diagnosis_does_not_mutate_the_caller_frame() -> None:
    frame = _txns()
    before = frame.copy(deep=True)
    _featureset().plan(frame).run(frame)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("entities", [1, 3])
def test_a_tiny_dataset_still_produces_a_report(entities: int) -> None:
    frame = _txns(rows=entities * 3, entities=entities, seed=2)
    run = _featureset().plan(frame, verify=False).run(frame)
    assert run.report.render_html()
    assert len(run.features) <= entities


def test_cost_concentration_is_estimated_structurally_not_timed() -> None:
    """A finding built on the clock fires at random; this one must not.

    The timing-based version of this detector fired on clean data the first
    time the suite ran under load, which is the whole argument for estimating
    cost from the operations in the plan instead.
    """
    frame = _txns()
    fs = FeatureSet(entity="account", time="ts")

    @fs.opaque(requires=("amount",))
    def legacy_blob(group: pd.DataFrame) -> float:
        return float(group["amount"].sum())

    @fs.feature
    def amt_mean(amount, all):
        return all.mean(amount)

    _, observations = split_findings(_report(fs, frame).findings)
    assert any("dominates the cost" in item.title for item in observations)

    # Same plan, repeatedly: identical findings and identical evidence ids.
    seen = {
        tuple((item.title, item.evidence_ids) for item in _report(fs, frame).findings)
        for _ in range(3)
    }
    assert len(seen) == 1


def test_cost_concentration_stays_quiet_on_a_balanced_plan() -> None:
    """Clean reading is 14% of estimated work; the detector fires at 40%."""
    frame = _txns(rows=9000, entities=450, seed=13)
    _, observations = split_findings(_report(_featureset(), frame).findings)
    assert not any("dominates the cost" in item.title for item in observations)


# ---------------------------------------------------------------------------
# The documented example must keep working
# ---------------------------------------------------------------------------
def _guide_featureset(**kwargs: object) -> FeatureSet:
    """The feature set printed in docs/usage_docs/feature-engineering.md."""
    from prism_eda.features import log1p, top_share

    fs = FeatureSet(entity="account_id", time="txn_ts", **kwargs)  # type: ignore[arg-type]
    fs.window("w7d", days=7)
    fs.window("w30d", days=30)
    fs.window("last10", last_k=10)

    @fs.derive
    def failed(status):
        return status.str.upper().str.strip().isin({"FAILED", "DECLINED", "FAILURE"})

    @fs.feature
    def txn_count_30d(w30d):
        return w30d.count()

    @fs.feature
    def failed_ratio_30d(failed, w30d):
        return safe_div(w30d.count(where=failed), w30d.count())

    @fs.feature
    def failed_ratio_last10(failed, last10):
        return safe_div(last10.count(where=failed), last10.count())

    @fs.feature
    def max_consecutive_failures(failed, all):
        return all.run_length_max(failed)

    @fs.feature
    def amt_mean_log(amount, all):
        return log1p(all.mean(amount))

    @fs.feature
    def top_beneficiary_share_7d(beneficiary, w7d):
        return top_share(w7d, beneficiary)

    return fs


def test_the_guides_example_runs_and_reports_what_the_guide_says() -> None:
    """Usage docs are written against examples/sample_data.py, so they are
    asserted against it too -- otherwise the page drifts from the library."""
    import prism_eda as pe
    from examples.sample_data import account_transactions

    ledger = account_transactions()
    run = pe.features(
        {"account_transactions": ledger},
        _guide_featureset(reference_time="decided_at"),
    )

    assert run.features.shape == (120, 6)
    # The full line, not a fragment: the guide prints this verbatim, and a
    # substring check would let the shared-operation counts drift off the page.
    assert run.report.summary == (
        "6 feature(s) computed for 120 entities; 9 of 24 operations are shared "
        "between features. 1 issue(s) (1 high). Top issue \u2014 The source "
        "frame contains rows from after the decision moment."
    )
    issues, _ = split_findings(run.report.findings)
    assert [item.title for item in issues] == [
        "The source frame contains rows from after the decision moment"
    ]


def test_the_ledgers_planted_leakage_is_what_the_guide_claims() -> None:
    """The numbers quoted on the page are load-bearing; pin them."""
    from examples.sample_data import account_transactions

    ledger = account_transactions()
    after = ledger["txn_ts"] > ledger["decided_at"]
    assert round(after.mean(), 2) == 0.30
    assert ledger.loc[after, "account_id"].nunique() == 116
    assert ledger["account_id"].nunique() == 120

    leaking = _guide_featureset().plan(ledger, verify=False).execute(ledger)
    correct = (
        _guide_featureset(reference_time="decided_at")
        .plan(ledger, verify=False)
        .execute(ledger)
    )
    differing = (
        leaking["txn_count_30d"].sort_index() != correct["txn_count_30d"].sort_index()
    ).sum()
    assert differing == 107


def test_the_sample_ledger_exercises_the_edges_it_claims_to() -> None:
    from examples.sample_data import account_transactions

    ledger = account_transactions()
    # Amounts must repeat, or a distinct-value ratio measures nothing.
    assert ledger["amount"].nunique() < len(ledger) / 10
    # Single-transaction accounts, where spread and gaps are undefined.
    assert (ledger.groupby("account_id").size() == 1).sum() >= 1
    # Failures arrive in runs, so a longest-run feature has something to find.
    plan = _guide_featureset(reference_time="decided_at").plan(ledger)
    produced = plan.execute(ledger)
    assert produced["max_consecutive_failures"].max() >= 3
