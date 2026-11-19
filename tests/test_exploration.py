"""Deep exploration on a synthetic bank whose properties are known in advance.

``examples/bank_data.py`` plants each property asserted here. The second half
of the file is the other direction: data with nothing planted, on which every
detector must stay quiet.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest

import prism_eda as pe
from examples.bank_data import INTERNAL_RECEIVER_SHARE, bank_tables
from prism_eda.events import Event
from prism_eda.exploration import checkpoint
from prism_eda.exploration.config import ColumnHint, Hints
from prism_eda.exploration.tables import TableState, find_gaps
from prism_eda.results import AnalysisStatus
from prism_eda.sources import FrameSource, Source, SourceError

HINTS = {
    "label": "accounts.is_flagged",
    "amount": "transactions.amount",
    "entity": "sender",
}


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return bank_tables()


@pytest.fixture(scope="module")
def result(frames):
    return pe.explore(frames, hints=HINTS)


def _of(result, kind: str) -> list[Any]:
    return [item for item in result.evidence if item.kind == kind]


def _relationships(result) -> dict[tuple[str, str], dict[str, Any]]:
    found = {}
    for item in _of(result, "explore_relationship"):
        value = item.value
        child = f"{value['child']['table']}.{value['child']['column']}"
        parent = f"{value['parent']['table']}.{value['parent']['column']}"
        found[(child, parent)] = value
    return found


def _finding(result, start: str):
    return [finding for finding in result.findings if finding.title.startswith(start)]


# --------------------------------------------------------------------------
# What was planted is found
# --------------------------------------------------------------------------


def test_run_completes_and_counts_everything(result, frames) -> None:
    assert result.goal == "exploration"
    assert result.status is AnalysisStatus.COMPLETED
    assert not result.failures
    assert result.catalog.table_count == len(frames)
    assert result.catalog.row_count == sum(len(frame) for frame in frames.values())
    assert result.metadata["table_count"] == 6
    assert result.metadata["tables_empty"] == 1
    assert result.metadata["link_count"] == 1


def test_sender_and_receiver_form_a_link_inside_one_table(result) -> None:
    pair = _relationships(result)[("transactions.sender", "transactions.receiver")]
    assert pair["same_table"] is True
    assert pair["tier"] == "strong"
    assert pair["kind"] == "shared_values"
    assert pair["primary"] is True


def test_format_mismatch_is_found_once(result) -> None:
    pair = _relationships(result)[("devices.acc", "accounts.acct_no")]
    assert pair["tier"] == "strong" and pair["kind"] == "many_to_one"
    assert pair["coverage"] == 1.0
    assert "leading zeros" in pair["format_note"]
    findings = _finding(result, "Same identifiers, different format")
    # devices.acc disagrees with three columns; it is one defect, not three.
    assert len(findings) == 1
    assert findings[0].severity == "high"
    cited = {item.id for item in _of(result, "explore_relationship")}
    assert set(findings[0].evidence_ids) <= cited


def test_small_integer_reference_is_only_likely(result) -> None:
    pair = _relationships(result)[("accounts.cust_ref", "customers.customer_id")]
    assert pair["tier"] == "likely"
    assert pair["weak_values"] is True


def test_coincidental_serial_is_never_a_relationship(result) -> None:
    for (child, parent), value in _relationships(result).items():
        if "audit_log." in child or "audit_log." in parent:
            assert value["tier"] == "possible", (child, parent)


def test_receiver_coverage_matches_the_planted_share(result) -> None:
    pair = _relationships(result)[("transactions.receiver", "accounts.acct_no")]
    assert abs(pair["row_coverage"] - INTERNAL_RECEIVER_SHARE) <= 0.06
    # Most distinct receivers are outside accounts we hold.
    assert pair["coverage"] < 0.3
    low, high = pair["coverage_interval"]
    assert low <= pair["coverage"] <= high
    titles = [finding.title for finding in result.findings]
    assert "transactions.receiver is only partly covered by accounts.acct_no" in titles


def test_domains_group_the_account_columns(result) -> None:
    domains = [item.value for item in _of(result, "explore_domain")]
    accounts = next(d for d in domains if d["label"] == "accounts.acct_no")
    members = {(m["table"], m["column"]) for m in accounts["members"]}
    assert members == {
        ("accounts", "acct_no"),
        ("devices", "acc"),
        ("transactions", "sender"),
        ("transactions", "receiver"),
    }
    assert accounts["reference"]["column"] == "acct_no"


def test_link_profile(result) -> None:
    (link,) = _of(result, "explore_link")
    value = link.value
    assert (value["origin"], value["destination"]) == ("sender", "receiver")
    assert value["linked_rows"] == value["rows"]
    assert value["self_links"] >= 0
    assert value["out_degree"]["median"] > value["in_degree"]["median"]
    assert [hop["hop"] for hop in value["reach"]["hops"]] == [1, 2, 3]
    reach = [hop["median_reach"] for hop in value["reach"]["hops"]]
    assert reach[0] < reach[1] < reach[2]
    assert 0 < value["reach"]["continuation_rate"] < 1
    assert abs(value["destination_known"]["row_coverage"] - 0.4) <= 0.06
    components = value["components"]
    assert components["method"] == "exact"
    assert components["largest_share"] > 0.9
    (finding,) = _finding(result, "Record-to-record links in transactions")
    assert finding.evidence_ids == (link.id,)
    assert finding.category == "observation"


def test_timeline_gap_is_reported(result) -> None:
    (finding,) = _finding(result, "Gap in transactions.ts")
    assert "2024-03-10" in finding.summary and "14 consecutive days" in finding.summary
    timeline = next(
        item
        for item in _of(result, "explore_timeline")
        if item.id in finding.evidence_ids
    )
    assert timeline.value["gaps"][0]["periods"] == 14
    assert timeline.metadata["basis"] == "full_table"


def test_empty_table_is_reported(result) -> None:
    (finding,) = _finding(result, "Empty tables")
    assert "empty_staging" in finding.summary


def test_amount_hint_finds_the_heaped_amount(result) -> None:
    (finding,) = _finding(result, "Repeated amount just below a round figure")
    assert "49,999.00" in finding.summary and "50,000" in finding.summary
    (amount,) = _of(result, "explore_amount")
    assert amount.value["top_amounts"][0]["value"] == 49_999.0
    assert amount.metadata["basis"] == "sample"


def test_label_hint_compares_labelled_accounts(result) -> None:
    (label,) = _of(result, "explore_label")
    value = label.value
    assert value["table"] == "accounts" and value["positive_values"] == ["1"]
    assert value["labelled_rows"] == value["labelled_sampled"]
    (link,) = value["links"]
    assert link["labelled_as_origin"] == 1.0
    assert link["labelled_set_complete"] is True
    # Flagged accounts were built to send to far more counterparties.
    labelled_median = link["labelled_out_degree"]["median"]
    assert labelled_median > 5 * link["baseline_out_degree_median"]
    assert _finding(result, "What sets labelled rows apart in accounts")


def test_entity_hint_adds_activity(result) -> None:
    (activity,) = _of(result, "explore_entity_activity")
    assert activity.value["column"] == "sender"
    assert 0 < activity.value["top_1pct_share"] < 1


def test_identifiers_are_not_reported_as_numbers_stored_as_text(result) -> None:
    assert not _finding(result, "Numbers stored as text")


def test_every_finding_cites_evidence_that_exists(result) -> None:
    known = {item.id for item in result.evidence}
    for finding in result.findings:
        assert finding.evidence_ids, finding.title
        assert set(finding.evidence_ids) <= known, finding.title


def test_sampled_evidence_says_so(frames) -> None:
    result = pe.explore(frames, sample_rows=5_000, relationships=False, network=False)
    sampled = [
        item
        for item in result.evidence
        if item.scope.table == "transactions" and item.kind.startswith("profile_")
    ]
    assert sampled
    assert all(item.metadata["basis"] == "sample" for item in sampled)
    record = next(r for r in result.sampling if "transactions" in r.operation)
    assert record.sampled_rows == 5_000 and record.source_rows == 55_300
    # Full-table numbers are untouched by sampling.
    transactions = result.catalog.table("transactions")
    assert transactions.row_count == 55_300
    amount = next(c for c in transactions.columns if c.name == "amount")
    assert amount.statistics["basis"] == "full_table"
    assert amount.statistics["max"] == float(frames["transactions"]["amount"].max())


def test_result_is_json_and_never_names_the_library(result) -> None:
    text = json.dumps(result.to_dict())
    assert "prism" not in text.lower()
    assert json.loads(text)["goal"] == "exploration"


def test_exploration_is_deterministic_and_does_not_mutate(frames) -> None:
    before = {name: frame.copy(deep=True) for name, frame in frames.items()}
    first = pe.explore(frames, hints=HINTS)
    second = pe.explore(frames, hints=HINTS)
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert [f.id for f in first.findings] == [f.id for f in second.findings]
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, before[name])


# --------------------------------------------------------------------------
# Nothing planted, nothing found
# --------------------------------------------------------------------------


def _clean_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    rows = 4_000
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 120 * 86_400, rows)), unit="s"
    )
    return {
        "orders": pd.DataFrame(
            {
                "id": np.arange(1, rows + 1),
                "placed_at": stamps,
                "quantity": rng.integers(1, 900, rows),
                "total": np.round(rng.lognormal(4, 1, rows), 2),
                "note": [f"order note {index}" for index in range(rows)],
            }
        ),
        "sensors": pd.DataFrame(
            {
                "id": np.arange(1, rows + 1),
                "reading": rng.integers(1, 5_000, rows),
                "code": [f"S-{value:07X}" for value in rng.integers(16**7, size=rows)],
            }
        ),
    }


def test_unrelated_tables_produce_no_relationships() -> None:
    result = pe.explore(_clean_tables())
    assert result.status is AnalysisStatus.COMPLETED
    tiers = Counter(item.value["tier"] for item in _of(result, "explore_relationship"))
    assert tiers["strong"] == 0 and tiers["likely"] == 0
    assert result.metadata["relationship_count"] == 0
    assert not _of(result, "explore_link")
    assert not _of(result, "explore_domain")
    assert not _finding(result, "Gap in")
    assert not _finding(result, "Same identifiers")
    assert not [f for f in result.findings if f.severity in {"high", "critical"}]


def test_gap_detector_stays_quiet() -> None:
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    dense = [(day.isoformat(), 500 + index % 7) for index, day in enumerate(days)]
    assert find_gaps(dense, "day") == []
    # A quiet series is full of empty days; none of them is a gap.
    sparse = [(day.isoformat(), 2) for day in days[::9]]
    assert find_gaps(sparse, "day") == []
    # Too short to judge.
    assert find_gaps(dense[:10], "day") == []
    # Two missing days are a weekend, not an outage.
    weekend = dense[:40] + dense[42:]
    assert find_gaps(weekend, "day") == []
    outage = dense[:40] + dense[50:]
    (gap,) = find_gaps(outage, "day")
    assert gap["periods"] == 10


def test_hints_parse_and_reject_unknown_keys() -> None:
    hints = Hints.parse({"label": "accounts.is_flagged", "entity": {"column": "id"}})
    assert hints.label == ColumnHint(column="is_flagged", table="accounts")
    assert hints.entity == ColumnHint(column="id")
    assert not Hints.parse(None)
    with pytest.raises(ValueError, match="Unknown hint"):
        Hints.parse({"target": "x"})


def test_missing_label_is_a_warning_not_an_error(frames) -> None:
    result = pe.explore(
        frames, hints={"label": "nope.missing"}, relationships=False, network=False
    )
    assert any(warning.code == "label_not_found" for warning in result.warnings)


# --------------------------------------------------------------------------
# Selection, failures, checkpoints, cancellation
# --------------------------------------------------------------------------


def test_include_and_exclude(frames) -> None:
    result = pe.explore(frames, include=["a*", "customers"], exclude="audit*")
    assert {table.name for table in result.catalog.tables} == {"accounts", "customers"}
    nothing = pe.explore(frames, include="no_such_table")
    assert nothing.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert nothing.catalog.table_count == 0


def test_config_and_options_are_exclusive(frames) -> None:
    with pytest.raises(TypeError):
        pe.explore(frames, config=pe.ExploreConfig(), sample_rows=500)
    with pytest.raises(TypeError):
        pe.explore(frames, no_such_option=1)


def test_duplicate_source_names_are_rejected(frames) -> None:
    one = FrameSource({"accounts": frames["accounts"]}, name="same")
    two = FrameSource({"customers": frames["customers"]}, name="same")
    with pytest.raises(ValueError, match="distinct name"):
        pe.explore([one, two])


def test_two_sources_are_explored_together(frames) -> None:
    left = FrameSource({"accounts": frames["accounts"]}, name="core")
    right = FrameSource({"devices": frames["devices"]}, name="channel")
    result = pe.explore([left, right])
    pair = _relationships(result)[("devices.acc", "accounts.acct_no")]
    assert pair["cross_source"] is True


class _Wrapped(Source):
    """A source that counts calls and can be told to fail."""

    kind = "dataframe"

    def __init__(self, inner: Source, *, fail_column: str | None = None,
                 fail_table: str | None = None) -> None:  # fmt: skip
        self.inner = inner
        self.name = inner.name
        self.calls: Counter[str] = Counter()
        self.fail_column = fail_column
        self.fail_table = fail_table

    def _count(self, name: str) -> None:
        self.calls[name] += 1

    def tables(self):
        self._count("tables")
        return self.inner.tables()

    def columns(self, table):
        self._count("columns")
        if table.name == self.fail_table:
            raise SourceError("permission denied")
        return self.inner.columns(table)

    def row_count(self, table):
        self._count("row_count")
        return self.inner.row_count(table)

    def column_stats(self, table, columns):
        self._count("column_stats")
        if any(column.name == self.fail_column for column in columns):
            raise SourceError("cannot aggregate this column")
        return self.inner.column_stats(table, columns)

    def top_values(self, table, columns, limit):
        self._count("top_values")
        return self.inner.top_values(table, columns, limit)

    def time_histogram(self, table, column, grain):
        self._count("time_histogram")
        return self.inner.time_histogram(table, column, grain)

    def sample(self, table, columns, rows, **kwargs):
        self._count("sample")
        return self.inner.sample(table, columns, rows, **kwargs)

    def distinct_sample(self, table, requests, limit):
        self._count("distinct_sample")
        return self.inner.distinct_sample(table, requests, limit)

    def probe(self, table, requests):
        self._count("probe")
        return self.inner.probe(table, requests)

    def pair_profile(self, table, left, right):
        self._count("pair_profile")
        return self.inner.pair_profile(table, left, right)

    def fanout_sample(self, table, key, others, rate, limit):
        self._count("fanout_sample")
        return self.inner.fanout_sample(table, key, others, rate, limit)

    def neighbours(self, table, left, right, values, limit):
        self._count("neighbours")
        return self.inner.neighbours(table, left, right, values, limit)


@pytest.fixture(scope="module")
def small() -> dict[str, pd.DataFrame]:
    return bank_tables(customers=300, transactions=6_000)


def test_one_bad_column_does_not_cost_the_table(small) -> None:
    source = _Wrapped(FrameSource(small), fail_column="balance")
    result = pe.explore([source], relationships=False, network=False)
    assert result.status is AnalysisStatus.COMPLETED_WITH_WARNINGS
    failure = next(f for f in result.failures if f.stage == "column_statistics")
    assert failure.table == "accounts" and "balance" in failure.message
    accounts = result.catalog.table("accounts")
    by_name = {column.name: column for column in accounts.columns}
    assert by_name["balance"].statistics == {"profiled": False}
    assert by_name["status"].unique_count == 3
    assert by_name["acct_no"].non_null_count == accounts.row_count


def test_unreadable_table_is_reported_not_fatal(small) -> None:
    source = _Wrapped(FrameSource(small), fail_table="devices")
    result = pe.explore([source])
    assert result.status is AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert result.metadata["tables_failed"] == 1
    warning = next(w for w in result.warnings if w.code == "table_not_profiled")
    assert warning.table == "devices" and "permission denied" in warning.message
    assert {table.name for table in result.catalog.tables} == set(small) - {"devices"}
    # The rest of the run is unaffected.
    assert ("transactions.sender", "accounts.acct_no") in _relationships(result)


def test_source_that_cannot_list_tables_is_fatal_only_when_alone(small) -> None:
    class Broken(FrameSource):
        def tables(self):
            raise SourceError("connection refused")

    with pytest.raises(SourceError, match="No source could be listed"):
        pe.explore([Broken({}, name="down")])
    result = pe.explore([Broken({}, name="down"), FrameSource(small, name="up")])
    assert result.catalog.table_count == len(small)
    assert any(failure.stage == "inventory" for failure in result.failures)


def test_checkpoints_resume_without_recomputing(small) -> None:
    store = pe.MemoryCheckpoints()
    first_source = _Wrapped(FrameSource(small))
    first = pe.explore([first_source], hints=HINTS, checkpoints=store)
    assert len(store.keys()) == len(small) + 2  # tables, relationships, links

    second_source = _Wrapped(FrameSource(small))
    second = pe.explore([second_source], hints=HINTS, checkpoints=store)
    assert sorted(i.id for i in first.evidence) == sorted(i.id for i in second.evidence)
    assert [f.id for f in first.findings] == [f.id for f in second.findings]
    assert first.catalog.fingerprint == second.catalog.fingerprint
    # Tables, relationships and links come from the store; only the listing
    # and the hint-driven sections touch the source again.
    for stage in ("column_stats", "distinct_sample", "pair_profile", "row_count"):
        assert first_source.calls[stage] > 0
        assert second_source.calls[stage] == 0, stage
    assert sum(second_source.calls.values()) < sum(first_source.calls.values()) / 3


def test_changed_config_does_not_reuse_checkpoints(small) -> None:
    store = pe.MemoryCheckpoints()
    pe.explore(small, checkpoints=store, network=False)
    before = set(store.keys())
    source = _Wrapped(FrameSource(small))
    pe.explore([source], checkpoints=store, network=False, sample_rows=1_000)
    assert source.calls["column_stats"] > 0
    assert not before & (set(store.keys()) - before)
    assert len(store.keys()) == 2 * len(before)


def test_checkpoint_round_trip_and_garbage(small) -> None:
    store = pe.MemoryCheckpoints()
    pe.explore(small, checkpoints=store, relationships=False, network=False)
    restored = [checkpoint.loads(TableState, store.load(key)) for key in store.keys()]
    assert all(isinstance(state, TableState) for state in restored)
    accounts = next(state for state in restored if state.display == "accounts")
    assert isinstance(accounts.catalog.columns, tuple)
    assert accounts.evidence[0].scope.table == "accounts"
    assert checkpoint.loads(TableState, b"not json") is None
    assert checkpoint.loads(TableState, b'{"format": 99, "value": {}}') is None
    assert checkpoint.loads(TableState, b'{"format": 1, "value": {"nope": 1}}') is None
    # Checkpoints are JSON, never pickles.
    assert json.loads(store.load(store.keys()[0]))["format"] == 1


def test_progress_reports_and_cancels(small) -> None:
    seen = []
    pe.explore(small, progress=seen.append)
    fractions = [report.fraction for report in seen]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert {"tables", "relationships", "links"} <= {report.stage for report in seen}

    class Stop(Exception):
        pass

    def cancel(report) -> None:
        if report.stage == "relationships":
            raise Stop

    with pytest.raises(Stop):
        pe.explore(small, progress=cancel)


def test_event_callbacks_cannot_break_a_run(small) -> None:
    events: list[Event] = []

    def noisy(event: Event) -> None:
        events.append(event)
        raise RuntimeError("observer bug")

    result = pe.explore(small, callbacks=[noisy], relationships=False, network=False)
    assert result.status is AnalysisStatus.COMPLETED
    assert events[0].kind == pe.EventKind.RUN_STARTED
    assert events[-1].kind == pe.EventKind.PROGRESS_UPDATED


def test_dataset_and_single_frame_entry_points(small) -> None:
    dataset = pe.load(small)
    result = dataset.explore(
        config=pe.ExploreConfig(relationships=False, network=False)
    )
    assert result.catalog.table_count == len(small)
    single = pe.explore(small["accounts"], relationships=False, network=False)
    assert [table.name for table in single.catalog.tables] == ["table"]
