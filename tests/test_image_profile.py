from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

import prism_eda as pe
from examples.sample_data import sample_images


def _save_image(path, *, size=(32, 32), color=(120, 80, 40), fmt="PNG") -> None:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, size[0] - 5, size[1] - 5), outline=(255, 255, 255))
    image.save(path, format=fmt)


def _evidence(result, kind):
    return next(item for item in result.evidence if item.kind == kind)


def _titles(result):
    return [finding.title for finding in result.findings]


@pytest.fixture(scope="module")
def sample_result(tmp_path_factory):
    """Profile the documented sample dataset once and reuse it.

    Every assertion below is also a claim made in
    ``docs/usage_docs/image-datasets.md``, so the guide cannot drift from the
    behavior without turning this suite red.
    """
    root = sample_images(tmp_path_factory.mktemp("sample") / "images")
    return pe.profile_images(root)


def test_image_profile_reports_quality_and_lineage(tmp_path) -> None:
    root = tmp_path / "images"
    cats = root / "cat"
    dogs = root / "dog"
    cats.mkdir(parents=True)
    dogs.mkdir()

    _save_image(cats / "cat_1.png", size=(32, 32), color=(90, 100, 110))
    _save_image(cats / "cat_2.png", size=(32, 32), color=(90, 100, 110))
    _save_image(cats / "cat_3.png", size=(96, 32), color=(20, 20, 20))
    _save_image(dogs / "dog_1.jpg", size=(32, 32), color=(240, 240, 240), fmt="JPEG")
    (dogs / "broken.png").write_bytes(b"not an image")

    result = pe.profile_images(root)

    assert result.goal == "image_profile"
    assert result.status == pe.AnalysisStatus.COMPLETED_WITH_WARNINGS
    assert result.metadata["image_count"] == 4
    assert result.metadata["invalid_image_count"] == 1
    assert any(item.kind == "image_dimension_distribution" for item in result.evidence)
    assert any(item.kind == "invalid_image_files" for item in result.evidence)
    assert any("Unreadable image files" == finding.title for finding in result.findings)
    assert any(
        "Duplicate or near-duplicate" in finding.title for finding in result.findings
    )
    evidence_ids = {item.id for item in result.evidence}
    assert all(set(finding.evidence_ids) <= evidence_ids for finding in result.findings)
    assert not result.transformation_plan.is_empty


def test_load_images_is_deterministic_and_json_serializable(tmp_path) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    _save_image(root / "a.png", size=(20, 20), color=(10, 80, 130))
    _save_image(root / "b.png", size=(40, 20), color=(160, 80, 30))

    dataset = pe.load_images(root, recursive=False, label_strategy=None)
    first = dataset.profile()
    second = dataset.profile()

    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    payload = first.to_dict()
    assert payload["catalog"]["row_count"] == 2
    assert json.loads(json.dumps(payload))["goal"] == "image_profile"


def test_image_profile_html_is_self_contained(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    _save_image(root / "sample.png")

    result = pe.profile_images(root)
    target = result.to_html(tmp_path / "image-report.html")
    html = target.read_text(encoding="utf-8")

    assert "Image dataset quality profile" in html
    assert "Image dimensions" in html
    assert "https://" not in html


def test_split_and_label_are_inferred_from_the_directory_layout(tmp_path) -> None:
    root = sample_images(tmp_path / "images")
    dataset = pe.load_images(root)

    labels = {path.split("/")[-1]: label for path, label in dataset.labels.items()}
    splits = {path.split("/")[-1]: split for path, split in dataset.splits.items()}

    # root/train/cat/cat_01.png must read as label "cat" in split "train" — the
    # split directory is not the class, which is the trap in the usual layout.
    assert labels["cat_01.png"] == "cat"
    assert splits["cat_01.png"] == "train"
    assert labels["cat_06.png"] == "cat"
    assert splits["cat_06.png"] == "val"


def test_duplicate_across_splits_leads_as_critical(sample_result) -> None:
    leakage = _evidence(sample_result, "image_leakage_summary")
    cross_split = leakage.value["cross_split_duplicates"]

    assert [sorted(row["splits"]) for row in cross_split] == [["train", "val"]]
    assert all("leaked.png" in path for path in cross_split[0]["paths"])

    leak_finding = next(
        finding
        for finding in sample_result.findings
        if finding.title == "Duplicate images span train and evaluation splits"
    )
    assert leak_finding.severity == "critical"
    assert leak_finding.evidence_ids == (leakage.id,)
    # A train/test leak outranks everything else in the report.
    assert sample_result.findings[0] is leak_finding
    assert any(
        step.operation == "remove_cross_split_duplicate_images"
        for step in sample_result.transformation_plan.steps
    )


def test_same_image_under_two_labels_is_reported(sample_result) -> None:
    cross_label = _evidence(sample_result, "image_leakage_summary").value[
        "cross_label_duplicates"
    ]

    assert [sorted(row["labels"]) for row in cross_label] == [["cat", "dog"]]
    assert "Identical images carry conflicting labels" in _titles(sample_result)


def test_loader_traps_are_detected(sample_result) -> None:
    traps = _evidence(sample_result, "image_loader_traps").value["traps"]

    rotated = traps["rotates_on_load"]
    assert [row["path"] for row in rotated] == ["train/dog/rotated.jpg"]
    # Orientation 6 is a quarter turn, so honoring it also swaps width/height.
    assert rotated[0]["orientation"] == 6
    assert rotated[0]["transposes_dimensions"] is True

    assert [row["path"] for row in traps["truncated"]] == ["train/cat/truncated.jpg"]

    mismatch = traps["extension_mismatch"]
    assert [row["path"] for row in mismatch] == ["train/dog/photo.jpg"]
    assert (mismatch[0]["extension_suggests"], mismatch[0]["actual_format"]) == (
        "JPEG",
        "PNG",
    )

    assert "train/dog/gray.png" in {
        row["path"] for row in traps["grayscale_stored_as_color"]
    }

    titles = _titles(sample_result)
    assert "Images will rotate depending on the loader" in titles
    assert "Truncated image files" in titles
    assert "File extensions do not match the actual encoding" in titles


def test_truncated_file_is_profiled_rather_than_discarded(sample_result) -> None:
    # It decodes far enough to measure, so it is a finding, not an unreadable
    # file. Only the genuinely undecodable broken.png counts as invalid.
    invalid = _evidence(sample_result, "invalid_image_files").value
    assert invalid["count"] == 1
    assert invalid["files"][0]["path"] == "train/cat/broken.png"


def test_odd_shape_is_caught_when_every_other_image_agrees(sample_result) -> None:
    # The uniform-dataset case: MAD and IQR are both zero, so a naive robust
    # z-score would score every image 0 and surface nothing.
    dimensions = _evidence(sample_result, "image_dimension_distribution").value
    assert [row["path"] for row in dimensions["aspect_ratio_outliers"]] == [
        "train/dog/panorama.png"
    ]
    assert "Resolution or aspect-ratio outliers" in _titles(sample_result)


def test_per_label_profile_is_reported(sample_result) -> None:
    profiles = _evidence(sample_result, "image_label_profile").value["profiles"]
    by_label = {profile["label"]: profile for profile in profiles}

    assert set(by_label) == {"cat", "dog"}
    assert by_label["cat"]["dominant_dimension"] == "64x64"
    assert by_label["dog"]["count"] == 11
    assert any(
        artifact.title == "Per-label profile" for artifact in sample_result.artifacts
    )


def test_report_shows_the_flagged_images(sample_result, tmp_path) -> None:
    sheets = [
        artifact
        for artifact in sample_result.artifacts
        if artifact.kind == "image_contact_sheet"
    ]
    titles = {sheet.title for sheet in sheets}
    assert {"Leaked across splits", "Quality-flagged images"} <= titles

    leaked = next(sheet for sheet in sheets if sheet.title == "Leaked across splits")
    # A duplicate pair is only reviewable when both images are shown together.
    assert leaked.data["layout"] == "pairs"
    assert len(leaked.data["groups"][0]["items"]) == 2
    assert all(
        item["src"].startswith("data:image/png;base64,")
        for item in leaked.data["groups"][0]["items"]
    )

    html = sample_result.to_html(tmp_path / "report.html").read_text(encoding="utf-8")
    assert "The flagged images" in html
    assert "data:image/png;base64," in html
    assert "Width against height" in html
    assert "Images per label" in html
    # Invariant: the report stays a single portable file.
    assert "https://" not in html and "http://" not in html


def test_thumbnails_can_be_disabled(tmp_path) -> None:
    root = sample_images(tmp_path / "images")

    result = pe.profile_images(root, thumbnails=False)

    assert not [
        artifact
        for artifact in result.artifacts
        if artifact.kind == "image_contact_sheet"
    ]
    assert "data:image/png;base64," not in json.dumps(result.to_dict())
    # Turning off the pictures must not change what was found.
    assert _titles(result) == _titles(pe.profile_images(root))


def test_corrupt_exif_does_not_condemn_a_readable_image(tmp_path) -> None:
    # Pillow warns rather than raises on a bad EXIF block; treating that warning
    # as a decode failure would call a perfectly good JPEG unreadable.
    root = tmp_path / "images"
    root.mkdir()
    image = Image.new("RGB", (48, 48), (90, 120, 150))
    exif = Image.Exif()
    exif[274] = 1
    image.save(root / "photo.jpg", exif=exif)

    result = pe.profile_images(root)

    assert result.metadata["image_count"] == 1
    assert result.metadata["invalid_image_count"] == 0
