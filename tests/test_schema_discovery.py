from __future__ import annotations

import json

import pandas as pd
import pytest

import prism_eda as pe


@pytest.fixture
def related_tables() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "segment": ["retail", "business", "retail"],
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": [10, 11, 12, 13],
                "customer_id": [1, 1, 2, 3],
                "amount": [20.0, 35.0, 18.0, 42.0],
            }
        ),
        "warehouses": pd.DataFrame(
            {
                "region_code": ["US", "US", "EU"],
                "warehouse_code": ["A", "B", "A"],
            }
        ),
        "shipments": pd.DataFrame(
            {
                "shipment_id": [100, 101, 102, 103],
                "region_code": ["US", "US", "US", "EU"],
                "warehouse_code": ["A", "A", "B", "A"],
            }
        ),
    }


def _relationship_values(result: pe.AnalysisResult) -> list[dict]:
    return [
        item.value for item in result.evidence if item.kind == "candidate_relationship"
    ]


def test_discovers_single_and_composite_relationships(related_tables) -> None:
    result = pe.discover_schema(related_tables, mode="standard")

    assert result.status == pe.AnalysisStatus.COMPLETED
    key_scopes = {
        (item.scope.table, item.scope.columns)
        for item in result.evidence
        if item.kind == "candidate_key"
    }
    assert ("customers", ("customer_id",)) in key_scopes
    assert ("warehouses", ("region_code", "warehouse_code")) in key_scopes
    assert ("orders", ("amount",)) not in key_scopes
    assert ("orders", ("customer_id", "amount")) not in key_scopes

    relationships = _relationship_values(result)
    assert any(
        item["parent_table"] == "customers"
        and item["parent_columns"] == ["customer_id"]
        and item["child_table"] == "orders"
        and item["child_columns"] == ["customer_id"]
        and item["cardinality"] == "one_to_many"
        for item in relationships
    )
    assert any(
        item["parent_table"] == "warehouses"
        and item["parent_columns"] == ["region_code", "warehouse_code"]
        and item["child_table"] == "shipments"
        and item["child_columns"] == ["region_code", "warehouse_code"]
        for item in relationships
    )
    assert result.artifacts[0].kind == "schema_graph"
    assert result.artifacts[0].evidence_ids


def test_key_search_reports_minimal_keys_only() -> None:
    result = pe.discover_schema(
        {
            "entities": pd.DataFrame(
                {
                    "entity_id": [1, 2, 3],
                    "version": [1, 1, 1],
                    "label": ["a", "b", "c"],
                }
            ),
            "events": pd.DataFrame({"event_id": [10, 11, 12], "entity_id": [1, 2, 3]}),
        }
    )

    entity_keys = [
        item.scope.columns
        for item in result.evidence
        if item.kind == "candidate_key" and item.scope.table == "entities"
    ]
    assert ("entity_id",) in entity_keys
    assert ("entity_id", "label") not in entity_keys


def test_partial_inclusion_surfaces_orphan_rows() -> None:
    result = pe.discover_schema(
        {
            "customers": pd.DataFrame({"customer_id": list(range(1, 11))}),
            "orders": pd.DataFrame(
                {
                    "order_id": list(range(100, 110)),
                    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 999],
                }
            ),
        }
    )

    relationship = next(
        item
        for item in _relationship_values(result)
        if item["parent_table"] == "customers" and item["child_table"] == "orders"
    )
    assert relationship["inclusion_rate"] == pytest.approx(0.9)
    assert relationship["orphan_row_count"] == 1
    assert any("Possible orphan rows" in finding.title for finding in result.findings)


def test_single_table_returns_insufficient_evidence() -> None:
    result = pe.discover_schema(pd.DataFrame({"id": [1, 2, 3]}))

    assert result.status == pe.AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert any(
        warning.code == "single_table_schema_discovery" for warning in result.warnings
    )


def test_no_relationship_returns_no_meaningful_structure() -> None:
    result = pe.discover_schema(
        {
            "left": pd.DataFrame({"left_id": [1, 2, 3]}),
            "right": pd.DataFrame({"right_id": [100, 101, 102]}),
        }
    )

    assert result.status == pe.AnalysisStatus.NO_MEANINGFUL_STRUCTURE
    assert not _relationship_values(result)


def test_invalid_composite_key_width_is_rejected(related_tables) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        pe.discover_schema(related_tables, max_key_columns=4)
    with pytest.raises(ValueError, match="between 1 and 3"):
        pe.discover_schema(related_tables, max_key_columns=0)


def test_large_child_relationship_is_sampled_deterministically() -> None:
    child_rows = 25_100
    tables = {
        "customers": pd.DataFrame({"customer_id": list(range(100))}),
        "events": pd.DataFrame(
            {
                "event_id": list(range(child_rows)),
                "customer_id": [index % 100 for index in range(child_rows)],
            }
        ),
    }

    first = pe.discover_schema(tables, mode="quick", random_seed=7)
    second = pe.discover_schema(tables, mode="quick", random_seed=7)

    relationship = next(
        item
        for item in _relationship_values(first)
        if item["parent_table"] == "customers" and item["child_table"] == "events"
    )
    assert relationship["sampled"] is True
    assert first.sampling
    assert first.sampling == second.sampling
    sampled_key = next(
        item
        for item in first.evidence
        if item.kind == "candidate_key"
        and item.scope.table == "events"
        and item.scope.columns == ("event_id",)
    )
    assert sampled_key.value["sampled"] is True
    assert sampled_key.value["evaluated_row_count"] == 25_000
    assert any(
        warning.code == "sampled_relationship_discovery" for warning in first.warnings
    )


def test_non_string_column_names_are_skipped_with_warning() -> None:
    result = pe.discover_schema(
        {
            "left": pd.DataFrame({0: [1, 2, 3], "left_id": [10, 11, 12]}),
            "right": pd.DataFrame({"right_id": [100, 101, 102]}),
        }
    )

    assert any(
        warning.code == "non_string_column_names_skipped" for warning in result.warnings
    )


def test_schema_exports_include_graph_and_evidence(tmp_path, related_tables) -> None:
    result = pe.discover_schema(related_tables)
    graph = result.artifacts[0].data
    html_path = result.to_html(tmp_path / "schema.html")
    json_path = result.to_json(tmp_path / "schema.json")

    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "Candidate schema map" in html
    assert "Candidate schema graph" in html
    assert "Evidence-backed candidates" in html
    assert "candidate primary key" in html
    assert "inferred foreign key" in html
    assert graph["layout"] == "layered_er_v2"
    assert all(edge["path"].startswith("M ") for edge in graph["edges"])
    assert all(len(edge["source_marks"]) == 2 for edge in graph["edges"])
    assert any(len(edge["target_marks"]) == 4 for edge in graph["edges"])
    assert any(row["kind"] == "FK" for node in graph["nodes"] for row in node["rows"])
    assert payload["artifacts"][0]["kind"] == "schema_graph"
    assert any(item["kind"] == "candidate_relationship" for item in payload["evidence"])
