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
    assert "Dataset fingerprint" not in html
    assert result.catalog.fingerprint not in html
    assert "https://" not in html
    assert "<style>" in html
    # The report names the data it profiled, and carries no product copy.
    assert "Decision-first data profile" not in html
    assert "backed by structured evidence" not in html


def test_report_embeds_the_packaged_logo(tmp_path) -> None:
    """The brand mark travels inside the file, so reports work offline.

    This also guards packaging: if the assets stop being included in the built
    distribution, the mark silently disappears from every report.
    """
    result = pe.profile(pd.DataFrame({"value": [1, 2, 3]}))

    html = result.to_html(tmp_path / "report.html").read_text(encoding="utf-8")

    assert '<img class="brand-mark" src="data:image/png;base64,' in html
    assert '<link rel="icon" type="image/png" href="data:image/png;base64,' in html


def test_report_renders_without_the_packaged_logo(monkeypatch, tmp_path) -> None:
    """A missing asset degrades the masthead; it does not fail the render."""
    import prism_eda.reporting.renderer as renderer

    monkeypatch.setattr(renderer, "_data_uri", lambda *args: None)
    result = pe.profile(pd.DataFrame({"value": [1, 2, 3]}))

    html = result.to_html(tmp_path / "report.html").read_text(encoding="utf-8")

    assert '<img class="brand-mark"' not in html
    assert "data:image/png;base64," not in html
    assert "Prism" in html  # the wordmark still identifies the report


def test_html_renders_category_values(tmp_path) -> None:
    result = pe.profile(
        pd.DataFrame(
            {
                "group": ["a", "a", "b"] * 1200,
                "flag": [True, True, False] * 1200,
                # 120 uniques over 3,600 rows (3.3%): categorical with a
                # high-cardinality warning and a folded "other" bucket.
                "many_codes": [f"code_{i:03d}" for i in range(120)] * 30,
            }
        )
    )
    target = result.to_html(tmp_path / "report.html")
    html = target.read_text(encoding="utf-8")

    # Category frequencies render as a bar chart, labelled and titled.
    assert "Value frequencies for group" in html
    assert "a: 2,400 row(s), 66.7%" in html
    assert "True: 2,400 row(s), 66.7%" in html
    assert "High cardinality for a categorical column" in html
    # The long tail is folded, and the report says how much it folded.
    assert "other (105 more)" in html
    assert "further label(s) are folded into" in html


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
