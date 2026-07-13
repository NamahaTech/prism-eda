# Image dataset profile

`profile_images()` profiles a folder of images without pretending they are
ordinary DataFrames. It returns the same `AnalysisResult` shape as the tabular
recipes, so findings still cite stable evidence IDs and exports still use
`to_html()` / `to_json()`.

```python
import prism_eda as pe

result = pe.profile_images("images/train/")
result.to_html("image-profile.html")
```

To reuse a discovered file set:

```python
dataset = pe.load_images("images/train/", recursive=True)
result = dataset.profile(mode="standard")
```

## Run it on the sample dataset

Every example on this page uses the seeded sample image dataset that ships with
the repository, so you can reproduce each finding yourself:

```python
from examples.sample_data import sample_images
import prism_eda as pe

root = sample_images("/tmp/prism-images")
result = pe.profile_images(root)

for finding in result.findings:
    print(f"[{finding.severity:>8}] {finding.title}")
```

```text
[critical] Duplicate images span train and evaluation splits
[    high] Unreadable image files
[    high] Images will rotate depending on the loader
[    high] Truncated image files
[    high] Identical images carry conflicting labels
[    high] Duplicate or near-duplicate image candidates
[  medium] Mixed image encodings
[  medium] Resolution or aspect-ratio outliers
[  medium] File extensions do not match the actual encoding
[  medium] Basic image quality flags
[     low] Channel layout is inconsistent
[     low] File-size outliers
```

The report leads with the split leak, because that is the one problem on the list
that silently makes your model look better than it is.

## Labels and splits come from the folders

The near-universal image-classification layout is `root/split/label/file`, and
Prism reads both parts of it:

```text
images/train/cat/001.png   ->  label "cat",  split "train"
images/val/cat/007.png     ->  label "cat",  split "val"
images/cat/001.png         ->  label "cat",  no split
```

Directory names such as `train`, `val`, `valid`, `validation`, `test`, `dev`,
`eval`, and `holdout` are recognized as splits rather than being mistaken for a
class. Disable label inference with `label_strategy=None`.

Splits are what make the leakage check possible, so a flat folder of images gets
the quality, duplicate, and metadata checks but not the leakage or per-label
ones.

## What it checks

Deterministic and dependency-light: Pillow handles file identification, headers,
EXIF, and grayscale statistics. No deep model, no network.

| Area | What you get |
|------|--------------|
| Decode health | unreadable files, and truncated files that only decode permissively |
| **Leakage** | duplicates that cross a **split** (inflates your scores) or a **label** (one annotation must be wrong) |
| **Loader traps** | EXIF rotation, extension/encoding mismatch, grayscale stored as RGB, used alpha channels |
| Shape | width, height, aspect ratio, megapixels, and the images that do not fit |
| Encoding | format, mode, animated frame count |
| Labels | per-label counts, sizes, brightness, and imbalance |
| Duplicates | exact SHA-256 groups and perceptual-hash near-duplicate candidates |
| Quality triage | very dark, very bright, low contrast, blurry, blank |
| Storage | unusual file sizes |

### Leakage is the headline

The same image on both sides of a split inflates every metric you report without
changing the model at all, so it is reported as `critical`:

```python
leakage = next(e for e in result.evidence if e.kind == "image_leakage_summary")

for row in leakage.value["cross_split_duplicates"]:
    print(row["splits"], row["paths"])
```

```text
['train', 'val'] ['train/cat/leaked.png', 'val/cat/leaked.png']
```

The same check across labels catches the other version of the problem — one
image filed under two classes, which means one of the two annotations is wrong.

### Loader traps

These files decode without complaint but reach your pipeline *changed*:

```python
traps = next(e for e in result.evidence if e.kind == "image_loader_traps")
print(traps.value["counts"])
```

```text
{'truncated': 1, 'grayscale_stored_as_color': 2, 'extension_mismatch': 1, 'rotates_on_load': 1}
```

* **Rotates on load** — a non-default EXIF orientation tag. Some loaders honor
  it, some ignore it, and orientations 5–8 also *swap width and height*, so the
  shape you profiled is not the shape you train on. Normalize it explicitly (for
  example with `PIL.ImageOps.exif_transpose`).
* **Extension mismatch** — `photo.jpg` that is actually PNG bytes. Anything that
  dispatches on the file extension will mis-handle it.
* **Grayscale stored as colour** — three identical channels, so you pay 3x the
  storage and any per-channel assumption is meaningless.
* **Transparency** — a used alpha channel composites against whatever background
  the loader happens to assume.
* **Truncated** — the file is cut short. It is profiled anyway (it decodes far
  enough to measure), but a strict loader will raise on it mid-epoch.

### Per-label breakdown

A global average hides collection bias, so dimension and brightness statistics
are also computed per label — if every `dog` image is 512x512 but `cat` ranges
from thumbnails to 4000px, a model can learn the artifact instead of the class.

## The report shows you the images

The HTML report does not just list the paths of the files it flagged — it shows
them. Duplicate candidates are rendered side by side so you can confirm or
dismiss a match without opening a file browser, and quality-flagged images,
outliers, and loader traps get thumbnail contact sheets.

Thumbnails are embedded as base64 PNGs, so the report stays a single portable
file with no network access, consistent with every other Prism report. They are
rendered only for the files the report actually shows, never for the whole
dataset.

```python
result = pe.profile_images(root, thumbnails=False)   # counts and paths only
result = pe.profile_images(root, thumbnail_size=64)  # smaller report
```

Turning thumbnails off changes the size of the report, never the findings. Use
it when you are exporting `to_json()` into a pipeline and do not want embedded
image data.

> **Privacy.** Thumbnails live in report artifacts, never in evidence, and image
> datasets are deliberately not exposed to the optional AI layer. Raw pixels are
> never sent to a model provider. See [AI-assisted analysis](ai-assisted-analysis.md).

## Interpreting duplicate and quality findings

Exact duplicate groups are byte-identical files. Near-duplicate pairs come from
average/difference perceptual hashes, so they are strong review candidates but
not proof — two genuinely different photographs can hash close together, and two
photographs of the same object will *not* be caught at all, because there is no
deep embedding behind this check.

Small datasets use a full pairwise scan; larger analyzed sets switch to
deterministic hash-window blocking and disclose which method they used in the
`image_duplicate_summary` evidence.

The quality checks are intentionally basic. They catch common ingestion problems
— blank frames, thumbnails, very dark exports, obviously low-detail images — but
they are not a task-specific perceptual quality model. For medical,
remote-sensing, or OCR domains, treat them as triage signals and confirm with
domain-aware checks.

## Configuration

```python
result = pe.profile_images(
    "images/train/",
    recursive=True,
    include=["*.jpg", "*.png"],
    exclude=["archive/*"],
    label_strategy="directory",   # or None
    mode="deep",
    near_duplicate_threshold=4,   # max perceptual-hash Hamming distance
    thumbnails=True,
    thumbnail_size=112,
)
```

Mode controls the deterministic file budget when `sampling="auto"`:

| Mode | Image budget |
|------|--------------|
| `quick` | 2,000 |
| `standard` | 10,000 |
| `deep` | 50,000 |

When a folder holds more images than the mode budget, Prism samples paths with
the configured `random_seed` and records a `SamplingRecord` that the report
discloses. Use `sampling="disabled"` to profile every discovered file.
