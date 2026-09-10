"""Tests for the feature planner's foundation.

The load-bearing tests here are not the ones proving a feature can be written.
They are the ones proving the traced-and-executed answer equals a hand-written
pandas answer computed independently, per entity, with no shared machinery. The
reference executor is the oracle every later optimisation is checked against,
so an error in it would be invisible from inside the module and would silently
bless a wrong planner.

The second theme is the tracing boundary. Tracing buys an exact record of what
a feature asked for, and pays for it by making data-dependent branching
impossible. That failure has to arrive as an explanation, not an AttributeError.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from prism_eda.features import (
    ContractViolation,
    FeatureSet,
    TracingError,
    VerificationError,
    distinct_ratio,
    lag,
    log1p,
    safe_div,
    time_delta,
    top_share,
)
from prism_eda.features.executors.pandas_exec import execute_planned
from prism_eda.features.executors.reference import execute_reference
from prism_eda.features.ir import node, walk

FAILED_TOKENS = {"DECLINED", "FAILED", "FAILURE"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _txns(rows: int = 400, entities: int = 25, seed: int = 3) -> pd.DataFrame:
    """A transaction frame shaped like the one a fraud model would see."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "account": rng.integers(0, entities, rows),
            "ts": start + pd.to_timedelta(rng.integers(0, 90 * 86_400, rows), unit="s"),
            # Rounded to tens so values repeat, as real transaction amounts
            # do. With every amount unique, distinct_amount_ratio is constant
            # and stops exercising the feature it exists to test.
            "amount": rng.gamma(2.0, 120.0, rows).round(-1) + 10.0,
            "status": rng.choice(["SUCCESS", "FAILED", "failed "], rows),
            "direction": rng.choice(["DEBIT", "CREDIT"], rows),
            "beneficiary": rng.integers(0, 12, rows).astype(str),
        }
    )


def _featureset() -> FeatureSet:
    """One feature set exercising every op family in the algebra."""
    fs = FeatureSet(entity="account", time="ts")
    fs.window("w7d", days=7)
    fs.window("w30d", days=30)
    fs.window("last10", last_k=10)

    @fs.derive
    def failed(status):
        return status.str.upper().str.strip().isin(FAILED_TOKENS)

    @fs.derive
    def is_debit(direction):
        return direction.str.upper().str.strip() == "DEBIT"

    @fs.feature
    def txn_count_all(all):
        return all.count()

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
    def debit_amt_30d(amount, is_debit, w30d):
        return w30d.sum(amount, where=is_debit)

    @fs.feature
    def amt_mean_log(amount, all):
        return log1p(all.mean(amount))

    @fs.feature(default=1.0)
    def amt_max_to_mean(amount, all):
        return safe_div(all.max(amount), all.mean(amount), default=1.0)

    @fs.feature
    def amt_std(amount, all):
        return all.std(amount)

    @fs.feature
    def active_days_30d(ts, w30d):
        return w30d.nunique(ts.dt.floor("D"))

    @fs.feature
    def hour_entropy(ts, all):
        return all.entropy(ts.dt.hour)

    @fs.feature
    def top_bene_share_7d(beneficiary, w7d):
        return top_share(w7d, beneficiary)

    @fs.feature(default=1.0)
    def distinct_amount_ratio(amount, all):
        return distinct_ratio(all, amount)

    @fs.feature
    def mean_inter_gap_secs(ts, all):
        return all.mean(time_delta(ts).dt.total_seconds())

    @fs.feature
    def retry_count(failed, status, all):
        succeeded = status.str.upper().str.strip() == "SUCCESS"
        return all.count(where=lag(failed) & succeeded)

    return fs


def _expected(frame: pd.DataFrame) -> pd.DataFrame:
    """The same features, written by hand, per entity, sharing nothing.

    Deliberately independent of the module: plain pandas on one group at a
    time, the way somebody would write it in a notebook.
    """
    rows: dict[object, dict[str, float]] = {}
    ordered = frame.sort_values("ts", kind="stable")
    for key, group in ordered.groupby("account", sort=False):
        ts = group["ts"]
        amt = group["amount"]
        status = group["status"].astype(str).str.upper().str.strip()
        failed = status.isin(FAILED_TOKENS)
        debit = group["direction"].astype(str).str.upper().str.strip() == "DEBIT"
        anchor = ts.max()
        m7 = (ts >= anchor - pd.Timedelta(days=7)) & (ts <= anchor)
        m30 = (ts >= anchor - pd.Timedelta(days=30)) & (ts <= anchor)
        last10 = pd.Series(False, index=group.index)
        last10.iloc[-10:] = True

        n = float(len(group))
        n30 = float(m30.sum())
        gaps = ts.diff().dt.total_seconds()
        hours = ts.dt.hour.value_counts(normalize=True).to_numpy()
        bene7 = group.loc[m7, "beneficiary"]
        succeeded = status == "SUCCESS"

        def div(num: float, den: float, default: float = 0.0) -> float:
            return float(num) / float(den) if den else default

        rows[key] = {
            "txn_count_all": n,
            "txn_count_30d": n30,
            "failed_ratio_30d": div(float((failed & m30).sum()), n30),
            "failed_ratio_last10": div(
                float((failed & last10).sum()), float(last10.sum())
            ),
            "max_consecutive_failures": float(
                max(
                    (
                        len(run)
                        for run in "".join("1" if f else "0" for f in failed).split("0")
                    ),
                    default=0,
                )
            ),
            "debit_amt_30d": float(amt[debit & m30].sum()),
            "amt_mean_log": float(np.log1p(amt.mean())),
            "amt_max_to_mean": div(float(amt.max()), float(amt.mean()), 1.0),
            "amt_std": float(amt.std()) if len(amt) > 1 else 0.0,
            "active_days_30d": float(ts[m30].dt.floor("D").nunique()),
            "hour_entropy": float(-np.sum(hours * np.log2(hours + 1e-12))),
            "top_bene_share_7d": div(
                float(bene7.value_counts().max()) if len(bene7) else 0.0,
                float(len(bene7)),
            ),
            "distinct_amount_ratio": div(float(amt.nunique()), n, 1.0),
            "mean_inter_gap_secs": float(gaps.mean()) if gaps.notna().any() else 0.0,
            "retry_count": float((failed.shift(1).fillna(False) & succeeded).sum()),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "account"
    return result


# ---------------------------------------------------------------------------
# The answer must equal hand-written pandas
# ---------------------------------------------------------------------------
def test_every_feature_matches_a_hand_written_equivalent() -> None:
    frame = _txns()
    plan = _featureset().plan(frame)
    produced = plan.execute(frame)
    expected = _expected(frame)

    assert list(produced.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        produced.sort_index(),
        expected[produced.columns].sort_index(),
        check_dtype=False,
        rtol=1e-9,
    )


def test_sparse_entities_do_not_break_any_aggregate() -> None:
    """One row per entity is where std, gaps and last-k all degenerate."""
    frame = pd.DataFrame(
        {
            "account": ["a", "b", "c"],
            "ts": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "amount": [10.0, 20.0, 30.0],
            "status": ["FAILED", "SUCCESS", "FAILED"],
            "direction": ["DEBIT", "CREDIT", "DEBIT"],
            "beneficiary": ["x", "y", "z"],
        }
    )
    produced = _featureset().plan(frame).execute(frame)
    assert list(produced.index) == ["a", "b", "c"]
    assert np.isfinite(produced.to_numpy()).all()
    # std of one observation is undefined, so the declared default stands in.
    assert produced["amt_std"].tolist() == [0.0, 0.0, 0.0]
    assert produced["amt_max_to_mean"].tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Windows and entity boundaries
# ---------------------------------------------------------------------------
def test_windows_are_anchored_per_entity_not_globally() -> None:
    """Two accounts active in different months must both see their own history."""
    frame = pd.DataFrame(
        {
            "account": ["old"] * 3 + ["new"] * 3,
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
                + ["2025-06-01", "2025-06-02", "2025-06-03"]
            ),
            "amount": [1.0] * 6,
        }
    )
    fs = FeatureSet(entity="account", time="ts")
    fs.window("w30d", days=30)

    @fs.feature
    def txn_count_30d(w30d):
        return w30d.count()

    produced = fs.plan(frame).execute(frame)
    assert produced["txn_count_30d"].tolist() == [3.0, 3.0]


def test_a_row_exactly_on_the_window_edge_is_inside_it() -> None:
    """The inclusive edge is measure-zero under random data and load-bearing.

    A lookback that excludes its own boundary drops a transaction at training
    time and keeps it at serving time, or the reverse. Nothing about a random
    fixture will ever surface that, so it is pinned here explicitly.
    """
    anchor = pd.Timestamp("2025-03-31")
    frame = pd.DataFrame(
        {
            "account": ["a"] * 3,
            "ts": [
                anchor - pd.Timedelta(days=30),
                anchor - pd.Timedelta(days=31),
                anchor,
            ],
            "amount": [1.0, 1.0, 1.0],
        }
    )
    fs = FeatureSet(entity="account", time="ts")
    fs.window("w30d", days=30)

    @fs.feature
    def txn_count_30d(w30d):
        return w30d.count()

    # Exactly 30 days back is in; 31 days back is out; the anchor itself is in.
    assert fs.plan(frame).execute(frame)["txn_count_30d"].tolist() == [2.0]


def test_sequential_operations_never_read_across_an_entity_boundary() -> None:
    frame = pd.DataFrame(
        {
            "account": ["a", "a", "b", "b"],
            "ts": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
            ),
            "amount": [1.0, 2.0, 100.0, 200.0],
        }
    )
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def mean_step(amount, all):
        return all.mean(time_delta(amount))

    produced = fs.plan(frame).execute(frame)
    # b's first row must not borrow a's last value: its only step is 200-100.
    assert produced["mean_step"].tolist() == [1.0, 100.0]


def test_last_k_uses_time_order_not_input_order() -> None:
    frame = pd.DataFrame(
        {
            "account": ["a"] * 4,
            "ts": pd.to_datetime(
                ["2025-01-04", "2025-01-01", "2025-01-03", "2025-01-02"]
            ),
            "amount": [4.0, 1.0, 3.0, 2.0],
        }
    )
    fs = FeatureSet(entity="account", time="ts")
    fs.window("last2", last_k=2)

    @fs.feature
    def mean_last2(amount, last2):
        return last2.mean(amount)

    # In time order the last two amounts are 3 and 4.
    assert fs.plan(frame).execute(frame)["mean_last2"].tolist() == [3.5]


# ---------------------------------------------------------------------------
# Structural sharing
# ---------------------------------------------------------------------------
def test_identical_subexpressions_collapse_to_one_node() -> None:
    first = node("column", name="amount")
    second = node("column", name="amount")
    assert first.id == second.id
    assert len(walk(node("add", first, second))) == 2


def test_a_shared_derive_is_traced_once() -> None:
    plan = _featureset().plan(_txns())
    shared = {item.op for item in plan.shared_nodes()}
    # The normalise-and-match chain behind `failed` is reached by four features.
    assert {"isin", "upper", "strip", "window"} <= shared


def test_explain_names_the_output_order() -> None:
    plan = _featureset().plan(_txns())
    text = plan.explain()
    assert "output order: txn_count_all" in text
    assert "shared subexpressions:" in text


# ---------------------------------------------------------------------------
# The tracing boundary
# ---------------------------------------------------------------------------
def test_branching_on_data_explains_itself() -> None:
    fs = FeatureSet(entity="account")

    @fs.feature
    def bad(amount, all):
        if amount > 100:  # noqa: SIM103 - the point of the test
            return all.count()
        return all.count()

    with pytest.raises(TracingError) as caught:
        fs.plan(pd.DataFrame({"account": ["a"], "amount": [1.0]}))
    message = str(caught.value)
    assert "@fs.opaque" in message
    assert "SCHEMA" in message


def test_branching_on_schema_is_allowed() -> None:
    """Real feature code tolerates an optional column; that must keep working."""
    fs = FeatureSet(entity="account")

    @fs.feature
    def non_upi_share(all):
        if "channel" not in fs.columns:
            return all.count() * 0.0
        return all.count()

    frame = pd.DataFrame({"account": ["a", "a"], "amount": [1.0, 2.0]})
    assert fs.plan(frame).execute(frame)["non_upi_share"].tolist() == [0.0]


def test_a_row_valued_feature_is_rejected() -> None:
    fs = FeatureSet(entity="account")

    @fs.feature
    def not_reduced(amount):
        return amount * 2

    with pytest.raises(TracingError, match="one number per entity"):
        fs.plan(pd.DataFrame({"account": ["a"], "amount": [1.0]}))


def test_an_unknown_parameter_names_the_available_columns() -> None:
    fs = FeatureSet(entity="account")

    @fs.feature
    def typo(amnout, all):
        return all.mean(amnout)

    with pytest.raises(TracingError, match="Available columns"):
        fs.plan(pd.DataFrame({"account": ["a"], "amount": [1.0]}))


def test_circular_derives_are_reported_as_a_cycle() -> None:
    fs = FeatureSet(entity="account")

    @fs.derive
    def left(right):
        return right

    @fs.derive
    def right(left):
        return left

    @fs.feature
    def broken(left, all):
        return all.mean(left)

    with pytest.raises(TracingError, match="Circular"):
        fs.plan(pd.DataFrame({"account": ["a"], "amount": [1.0]}))


def test_a_missing_entity_column_is_caught_before_anything_runs() -> None:
    fs = FeatureSet(entity="account")

    @fs.feature
    def total(amount, all):
        return all.sum(amount)

    with pytest.raises(TracingError, match="not in the frame"):
        fs.plan(pd.DataFrame({"amount": [1.0]}))


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
def test_declared_order_is_the_output_order() -> None:
    fs = _featureset()
    frame = _txns()
    assert list(fs.plan(frame).execute(frame).columns) == list(fs.feature_names)


def test_a_non_finite_result_falls_back_to_the_declared_default() -> None:
    frame = pd.DataFrame(
        {"account": ["a"], "ts": pd.to_datetime(["2025-01-01"]), "amount": [0.0]}
    )
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature(default=-1.0)
    def undefined_spread(amount, all):
        return all.std(amount)

    assert fs.plan(frame).execute(frame)["undefined_spread"].tolist() == [-1.0]


def test_safe_div_never_produces_an_infinity() -> None:
    frame = pd.DataFrame(
        {
            "account": ["a", "a"],
            "ts": pd.to_datetime(["2025-01-01"] * 2),
            "n": [1.0, 2.0],
        }
    )
    fs = FeatureSet(entity="account", time="ts")

    @fs.feature
    def over_zero(n, all):
        return safe_div(all.sum(n), all.sum(n) * 0.0)

    value = fs.plan(frame).execute(frame)["over_zero"].iloc[0]
    assert value == 0.0
    assert math.isfinite(value)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def test_execution_does_not_mutate_the_caller_frame() -> None:
    frame = _txns()
    before = frame.copy(deep=True)
    _featureset().plan(frame).execute(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_repeated_plans_are_identical() -> None:
    frame = _txns()
    first = _featureset().plan(frame)
    second = _featureset().plan(frame)
    assert first.to_dict() == second.to_dict()
    pd.testing.assert_frame_equal(first.execute(frame), second.execute(frame))


def test_an_opaque_feature_runs_and_is_named_as_unplanned() -> None:
    fs = FeatureSet(entity="account", time="ts")

    @fs.opaque(requires=("amount",))
    def weird(group: pd.DataFrame) -> float:
        return float(group["amount"].to_numpy()[::2].sum())

    frame = pd.DataFrame(
        {
            "account": ["a", "a", "a"],
            "ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "amount": [1.0, 10.0, 100.0],
        }
    )
    plan = fs.plan(frame)
    assert "opaque (unplanned, runs per entity): weird" in plan.explain()
    assert plan.execute(frame)["weird"].tolist() == [101.0]


# ---------------------------------------------------------------------------
# The planned executor must equal the oracle, and beat it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_planned_execution_equals_the_reference_executor(seed: int) -> None:
    """The claim the whole module rests on.

    The reference executor is right by definition. Any disagreement is a
    planner bug, and a planner that returns different numbers silently is
    worse than no planner at all -- so this runs across several seeds rather
    than one, and at a tolerance tight enough to catch a logic error.
    """
    frame = _txns(rows=1200, entities=80, seed=seed)
    plan = _featureset().plan(frame)
    reference = execute_reference(plan.features, frame, entity="account", time="ts")
    planned = execute_planned(plan.features, frame, entity="account", time="ts")
    pd.testing.assert_frame_equal(reference, planned, rtol=1e-9, check_dtype=False)


def test_the_planner_beats_the_per_entity_loop() -> None:
    """A performance floor, so the module's reason to exist cannot rot silently.

    Measured multiples are far higher than this (112x on 20k accounts with
    these 15 features). The floor is set well below that because it has to
    hold on a loaded CI box, not because the margin is thin.
    """
    rng = np.random.default_rng(11)
    entities, per = 1500, 20
    rows = entities * per
    frame = pd.DataFrame(
        {
            "account": np.repeat(np.arange(entities), per),
            "ts": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 86_400, rows), unit="s"),
            "amount": rng.gamma(2.0, 120.0, rows).round(2),
            "status": rng.choice(["SUCCESS", "FAILED", "failed "], rows),
            "direction": rng.choice(["DEBIT", "CREDIT"], rows),
            "beneficiary": rng.integers(0, 12, rows).astype(str),
        }
    )
    plan = _featureset().plan(frame)

    start = time.perf_counter()
    reference = execute_reference(plan.features, frame, entity="account", time="ts")
    loop_seconds = time.perf_counter() - start

    start = time.perf_counter()
    planned = execute_planned(plan.features, frame, entity="account", time="ts")
    planned_seconds = time.perf_counter() - start

    pd.testing.assert_frame_equal(reference, planned, rtol=1e-9, check_dtype=False)
    assert loop_seconds / planned_seconds > 8.0, (
        f"planner gained only {loop_seconds / planned_seconds:.1f}x "
        f"({loop_seconds:.2f}s vs {planned_seconds:.2f}s)"
    )


def test_both_executors_agree_on_a_single_row_entity() -> None:
    """The n=1 case is the serving case, and every reduction degenerates there."""
    frame = pd.DataFrame(
        {
            "account": ["solo"],
            "ts": pd.to_datetime(["2025-05-05"]),
            "amount": [42.0],
            "status": ["FAILED"],
            "direction": ["DEBIT"],
            "beneficiary": ["b1"],
        }
    )
    plan = _featureset().plan(frame)
    reference = execute_reference(plan.features, frame, entity="account", time="ts")
    planned = execute_planned(plan.features, frame, entity="account", time="ts")
    pd.testing.assert_frame_equal(reference, planned, rtol=1e-9, check_dtype=False)
    assert np.isfinite(planned.to_numpy()).all()


# ---------------------------------------------------------------------------
# Verification: two different claims, both checked
# ---------------------------------------------------------------------------
def test_planning_verifies_by_default_and_records_it() -> None:
    frame = _txns()
    plan = _featureset().plan(frame)
    assert plan.verification is not None
    assert plan.verification.ok
    assert plan.verification.kind == "planner_soundness"
    assert "verified: PASS" in plan.explain()


def test_verification_can_be_skipped() -> None:
    plan = _featureset().plan(_txns(), verify=False)
    assert plan.verification is None


def test_a_wrong_planner_is_caught_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure mode the always-on check exists for.

    A planner bug that shifts one feature must stop the build and say which
    feature, by how much, and on which entity -- not sail through and surface
    later as a model that underperforms for no visible reason.
    """
    from prism_eda.features import plan as plan_module

    real = plan_module.execute_planned

    def sabotaged(features, frame, **kwargs):
        produced = real(features, frame, **kwargs)
        produced["amt_mean_log"] = produced["amt_mean_log"] * 1.01
        return produced

    monkeypatch.setattr(plan_module, "execute_planned", sabotaged)

    with pytest.raises(VerificationError) as caught:
        _featureset().plan(_txns())
    message = str(caught.value)
    assert "amt_mean_log" in message
    assert "FAIL" in message
    assert "expected" in message and "got" in message


def test_verification_samples_entities_not_rows() -> None:
    """Sampling rows would change the computation, not shrink the check."""
    frame = _txns(rows=3000, entities=600, seed=5)
    plan = _featureset().plan(frame, verify=False)
    report = plan.verify(frame, sample=50)

    assert report.ok
    assert report.checked_entities == 50
    assert report.sampling is not None
    assert report.sampling.strategy == "deterministic_entity_sample"
    # Whole histories: the sampled rows are every row of the chosen entities.
    chosen = report.sampling.sampled_rows
    assert chosen > 50
    assert "entities" in report.sampling.limitations[0]


def test_verify_against_confirms_a_faithful_port() -> None:
    frame = _txns()
    plan = _featureset().plan(frame)

    def legacy(group: pd.DataFrame) -> dict[str, float]:
        """The same two features, written the old way."""
        status = group["status"].astype(str).str.upper().str.strip()
        return {
            "txn_count_all": float(len(group)),
            "failed_ratio_30d": _legacy_failed_ratio_30d(group, status),
        }

    report = plan.verify_against(legacy, frame)
    assert report.ok, report.describe()
    compared = {item.feature for item in report.agreements}
    assert compared == {"txn_count_all", "failed_ratio_30d"}


def test_verify_against_catches_a_port_that_drifted() -> None:
    """Internal consistency says nothing about matching the code you replaced."""
    frame = _txns()
    plan = _featureset().plan(frame)

    def drifted(group: pd.DataFrame) -> dict[str, float]:
        # An off-by-one that a plan-versus-reference check could never see.
        return {"txn_count_all": float(len(group)) + 1.0}

    report = plan.verify_against(drifted, frame)
    assert not report.ok
    assert report.failures[0].feature == "txn_count_all"
    assert report.kind == "migration_fidelity"
    with pytest.raises(VerificationError, match="txn_count_all"):
        report.raise_for_status()


def _legacy_failed_ratio_30d(group: pd.DataFrame, status: pd.Series) -> float:
    ts = group["ts"]
    anchor = ts.max()
    within = (ts >= anchor - pd.Timedelta(days=30)) & (ts <= anchor)
    denominator = float(within.sum())
    if not denominator:
        return 0.0
    return float((status.isin(FAILED_TOKENS) & within).sum()) / denominator


# ---------------------------------------------------------------------------
# The contract: order is not a rendering detail
# ---------------------------------------------------------------------------
def test_the_contract_reads_its_columns_off_the_ir() -> None:
    """Declared reads cannot drift from what the features actually touch."""
    contract = _featureset().plan(_txns()).contract()
    assert set(contract.required_columns) == {
        "account",
        "ts",
        "status",
        "amount",
        "direction",
        "beneficiary",
    }
    assert contract.names == _featureset().feature_names
    assert contract.defaults["amt_max_to_mean"] == 1.0


def test_a_permuted_output_is_a_contract_violation() -> None:
    """The failure a set comparison cannot see.

    Every column is present and every value is a plausible float; only the
    positions moved. A model consuming this by index scores garbage and
    nothing raises -- unless something checks order, which is this.
    """
    frame = _txns()
    plan = _featureset().plan(frame)
    contract = plan.contract()
    produced = plan.execute(frame)

    contract.validate_output(produced)  # the real thing passes

    shuffled = produced[list(reversed(produced.columns))]
    with pytest.raises(ContractViolation, match="out of contract order"):
        contract.validate_output(shuffled)


def test_a_missing_or_extra_column_is_reported_by_name() -> None:
    frame = _txns()
    plan = _featureset().plan(frame)
    contract = plan.contract()
    produced = plan.execute(frame)

    with pytest.raises(ContractViolation, match="missing: txn_count_all"):
        contract.validate_output(produced.drop(columns=["txn_count_all"]))
    with pytest.raises(ContractViolation, match="unexpected: surprise"):
        contract.validate_output(produced.assign(surprise=1.0))


def test_a_source_frame_missing_a_read_column_is_rejected() -> None:
    frame = _txns()
    contract = _featureset().plan(frame).contract()
    contract.validate_source(frame)
    with pytest.raises(ContractViolation, match="beneficiary"):
        contract.validate_source(frame.drop(columns=["beneficiary"]))


def test_the_contract_round_trips_through_json(tmp_path) -> None:
    contract = _featureset().plan(_txns()).contract()
    written = contract.to_json(tmp_path / "features.json")
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["alphabetical_sorting"] is False
    assert [item["name"] for item in loaded["features"]] == list(contract.names)
    assert loaded["entity"] == "account"


# ---------------------------------------------------------------------------
# Point-in-time correctness
# ---------------------------------------------------------------------------
def _leaky_frame() -> pd.DataFrame:
    """Each entity has rows from after its own decision moment."""
    return pd.DataFrame(
        {
            "account": ["a"] * 4 + ["b"] * 3,
            "ts": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-06-01",
                    "2025-06-02",
                    "2025-02-01",
                    "2025-02-02",
                    "2025-07-01",
                ]
            ),
            "decided_at": pd.to_datetime(["2025-01-15"] * 4 + ["2025-02-10"] * 3),
            "amount": [1.0, 2.0, 900.0, 900.0, 3.0, 4.0, 900.0],
        }
    )


def _pit_featureset(**kwargs: object) -> FeatureSet:
    fs = FeatureSet(entity="account", time="ts", **kwargs)  # type: ignore[arg-type]
    fs.window("w30d", days=30)
    fs.window("last2", last_k=2)

    @fs.feature
    def txn_count_all(all):
        return all.count()

    @fs.feature
    def amt_sum_30d(amount, w30d):
        return w30d.sum(amount)

    @fs.feature
    def amt_sum_last2(amount, last2):
        return last2.sum(amount)

    return fs


def test_a_declared_decision_moment_excludes_rows_from_after_it() -> None:
    """The difference between a correct backfill and a leaking one.

    Anchored at the entity's own last row, account 'a' sums 1800 -- entirely
    from transactions that had not happened when the decision was made.
    Anchored at the decision moment it sums 3, which is what the model would
    actually have seen. Nothing about the first answer looks wrong.
    """
    frame = _leaky_frame()
    leaking = _pit_featureset().plan(frame).execute(frame)
    correct = _pit_featureset(reference_time="decided_at").plan(frame).execute(frame)

    assert leaking["amt_sum_30d"].tolist() == [1800.0, 900.0]
    assert correct["amt_sum_30d"].tolist() == [3.0, 7.0]
    assert correct["txn_count_all"].tolist() == [2.0, 2.0]


def test_last_k_counts_back_from_the_decision_not_the_newest_row() -> None:
    """Future rows must not push eligible ones out of a last-k window."""
    frame = _leaky_frame()
    correct = _pit_featureset(reference_time="decided_at").plan(frame).execute(frame)
    assert correct["amt_sum_last2"].tolist() == [3.0, 7.0]


@pytest.mark.parametrize("reference", [None, "decided_at"])
def test_both_executors_agree_under_either_anchor(reference: str | None) -> None:
    frame = _leaky_frame()
    kwargs = {} if reference is None else {"reference_time": reference}
    plan = _pit_featureset(**kwargs).plan(frame, verify=False)
    produced = execute_planned(
        plan.features,
        frame,
        entity="account",
        time="ts",
        reference_time=plan.reference_time,
    )
    oracle = execute_reference(
        plan.features,
        frame,
        entity="account",
        time="ts",
        reference_time=plan.reference_time,
    )
    pd.testing.assert_frame_equal(oracle, produced, check_dtype=False)


def test_reference_time_is_part_of_the_contract() -> None:
    frame = _leaky_frame()
    contract = _pit_featureset(reference_time="decided_at").plan(frame).contract()
    assert contract.reference_time == "decided_at"
    assert "decided_at" in contract.required_columns
    with pytest.raises(ContractViolation, match="decided_at"):
        contract.validate_source(frame.drop(columns=["decided_at"]))


def test_reference_time_without_a_time_column_is_refused() -> None:
    with pytest.raises(ValueError, match="time= is required"):
        FeatureSet(entity="account", reference_time="decided_at")
