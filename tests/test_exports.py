from __future__ import annotations

import json

import pandas as pd

import prism_eda as pe


def test_json_export_contains_evidence_lineage(tmp_path) -> None:
    result = pe.profile(pd.DataFrame({"value": [1, 1, 1]}))
    target = result.to_json(tmp_path / "report.json")

    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["goal"] == "profile"
    assert payload["catalog"]["table_count"] == 1
    assert payload["evidence"]
    assert payload["findings"][0]["evidence_ids"]


def test_html_export_is_self_contained(tmp_path) -> None:
    result = pe.profile(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "score": [1.5, None, 3.5],
                "group": ["a", "a", "b"],
            }
        )
    )
    target = result.to_html(tmp_path / "report.html")
    html = target.read_text(encoding="utf-8")

    assert "<!doctype html>" in html
    assert "Decision-first data profile" in html
    assert "Dataset fingerprint" in html
    assert "https://" not in html
    assert "<style>" in html


def test_interactive_export_falls_back_without_plotly(tmp_path, monkeypatch) -> None:
    result = pe.profile(pd.DataFrame({"value": [1, 2]}))

    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "plotly":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    target = result.to_html(tmp_path / "interactive.html", interactive=True)

    assert "Plotly is not installed" in target.read_text(encoding="utf-8")
