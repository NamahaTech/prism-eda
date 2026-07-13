"""Reproducible sample dataset used throughout the Prism EDA usage docs.

Every code example in ``docs/usage_docs/`` builds its data from the helpers in
this module, so the outputs you see in the docs are the outputs you get when you
run the snippets yourself.

The data is intentionally seeded and contains a few realistic pathologies so the
analysis recipes have something meaningful (but not noisy) to surface:

* ``customers`` has a real primary key (``customer_id``), ~25% missingness in
  ``signup_age`` with two out-of-range values, a column that leaks the target
  (``exit_survey_sent`` is derived from ``churned``), and an identifier-like
  column.
* ``orders`` references ``customers`` through ``customer_id`` (a one-to-many
  relationship) and contains one extreme ``amount``.

Usage::

    from examples.sample_data import load_sample, customers, orders
    import prism_eda as pe

    dataset = pe.load(load_sample())
    result = dataset.classification("churned", table="customers")
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

#: Seed used for every random draw so the tables are byte-for-byte reproducible.
SEED = 7

#: Number of customers in the sample.
N_CUSTOMERS = 80

#: Number of orders in the sample.
N_ORDERS = 240


def customers() -> pd.DataFrame:
    """Return the seeded ``customers`` table.

    Columns:
        customer_id: Unique primary key (1..N_CUSTOMERS).
        region: Categorical region.
        plan: ``"free"`` or ``"pro"``.
        tenure_months: Account age in months.
        monthly_spend: Spend that genuinely tracks plan and tenure (real signal).
        signup_age: ~25% missing, with two out-of-range values (121 and 5).
        exit_survey_sent: Derived from ``churned`` — a deterministic leak.
        churned: The classification target (1 = churned).
    """
    rng = np.random.default_rng(SEED)
    n = N_CUSTOMERS

    customer_id = np.arange(1, n + 1)
    region = rng.choice(
        ["west", "east", "north", "south"], size=n, p=[0.35, 0.30, 0.20, 0.15]
    )
    plan = rng.choice(["free", "pro"], size=n, p=[0.6, 0.4])
    tenure_months = rng.integers(1, 48, size=n)

    monthly_spend = np.round(
        5 + (plan == "pro") * 25 + tenure_months * 0.4 + rng.normal(0, 3, size=n),
        2,
    )

    churn_prob = 0.12 + (plan == "free") * 0.25 + (tenure_months < 6) * 0.25
    churned = (rng.random(n) < churn_prob).astype(int)

    signup_age = rng.normal(38, 9, size=n).round().astype(float)
    signup_age[rng.choice(n, size=int(n * 0.25), replace=False)] = np.nan
    signup_age[3] = 121.0
    signup_age[17] = 5.0

    exit_survey_sent = churned.copy()

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "region": region,
            "plan": plan,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "signup_age": signup_age,
            "exit_survey_sent": exit_survey_sent,
            "churned": churned,
        }
    )


def orders() -> pd.DataFrame:
    """Return the seeded ``orders`` table.

    ``customer_id`` references :func:`customers` (a one-to-many relationship) and
    ``amount`` contains one extreme value for the anomaly examples.
    """
    rng = np.random.default_rng(SEED)
    # Advance the generator the same way customers() does so the FK values are
    # drawn from an independent, reproducible stream.
    customer_id = np.arange(1, N_CUSTOMERS + 1)

    order_customer = rng.choice(customer_id, size=N_ORDERS)
    amount = np.round(rng.gamma(2.0, 15.0, size=N_ORDERS) + 3, 2)
    amount[10] = 9999.0

    return pd.DataFrame(
        {
            "order_id": np.arange(1000, 1000 + N_ORDERS),
            "customer_id": order_customer,
            "amount": amount,
        }
    )


def load_sample() -> dict[str, pd.DataFrame]:
    """Return both tables as a ``{name: DataFrame}`` mapping ready for ``pe.load``."""
    return {"customers": customers(), "orders": orders()}


# --------------------------------------------------------------------------
# Sample image dataset
#
# The same idea as the tables above, for ``docs/usage_docs/image-datasets.md``:
# a small, seeded ``train``/``val`` image folder with deliberately planted
# pathologies, so every finding in that guide is one you can reproduce.
# --------------------------------------------------------------------------

#: Size every "well-behaved" sample image is stored at.
IMAGE_SIZE = (64, 64)


def _photo(
    rng: np.random.Generator,
    *,
    size: tuple[int, int] = IMAGE_SIZE,
    tint: tuple[int, int, int] = (130, 120, 110),
    noise: float = 34.0,
) -> Image.Image:
    """A textured RGB image — noise stands in for real detail."""
    width, height = size
    pixels = np.asarray(tint, dtype=np.float64) + rng.normal(
        0.0, noise, size=(height, width, 3)
    )
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _gradient(size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """A smooth ramp: no edges and almost no detail, so it reads as blurred."""
    width, height = size
    ramp = np.linspace(70, 150, width, dtype=np.float64)
    pixels = np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")


def sample_images(destination: str | Path) -> Path:
    """Write the sample image dataset under ``destination`` and return its root.

    The layout is the standard ``root/split/label/file`` convention, and the
    planted problems are:

    * ``leaked.png`` is byte-identical in ``train/cat`` and ``val/cat`` — the
      same image on both sides of the split.
    * ``cat_twin.png`` is a near-duplicate of ``cat_01.png``.
    * ``muddle.png`` is the same image as ``dog_01.png`` but filed under ``cat``,
      so one of the two labels must be wrong.
    * ``rotated.jpg`` carries EXIF orientation 6, so honoring the tag swaps its
      width and height.
    * ``photo.jpg`` is actually PNG-encoded despite its extension.
    * ``flat.png`` is a smooth gradient (no detail) and ``night.png`` is nearly
      black.
    * ``panorama.png`` is far wider than anything else in the set.
    * ``gray.png`` is greyscale stored in three identical colour channels.
    * ``truncated.jpg`` is cut short mid-file, and ``broken.png`` is not an
      image at all.
    """
    root = Path(destination)
    rng = np.random.default_rng(SEED)

    train_cat = root / "train" / "cat"
    train_dog = root / "train" / "dog"
    val_cat = root / "val" / "cat"
    val_dog = root / "val" / "dog"
    for folder in (train_cat, train_dog, val_cat, val_dog):
        folder.mkdir(parents=True, exist_ok=True)

    for index in range(1, 6):
        _photo(rng, tint=(150, 130, 110)).save(train_cat / f"cat_{index:02d}.png")
    for index in range(1, 5):
        _photo(rng, tint=(110, 120, 140)).save(train_dog / f"dog_{index:02d}.png")
    _photo(rng, tint=(150, 130, 110)).save(val_cat / "cat_06.png")
    _photo(rng, tint=(110, 120, 140)).save(val_dog / "dog_05.png")

    # The same image on both sides of the split — the leak that quietly inflates
    # every evaluation score.
    leaked = _photo(rng, tint=(140, 125, 115))
    leaked.save(train_cat / "leaked.png")
    leaked.save(val_cat / "leaked.png")

    # A near-duplicate: same picture, a few pixels nudged.
    twin = np.asarray(Image.open(train_cat / "cat_01.png"), dtype=np.uint8).copy()
    twin[:3, :3] = 255
    Image.fromarray(twin, mode="RGB").save(train_cat / "cat_twin.png")

    # One image, two labels. At most one of them is right.
    dog_one = Image.open(train_dog / "dog_01.png").copy()
    dog_one.save(train_cat / "muddle.png")

    # EXIF orientation 6 asks the loader to rotate a quarter turn, which also
    # swaps the reported width and height.
    exif = Image.Exif()
    exif[274] = 6
    _photo(rng, size=(96, 64), tint=(125, 125, 125)).save(
        train_dog / "rotated.jpg", exif=exif
    )

    # Named .jpg, actually PNG bytes.
    _photo(rng, tint=(120, 130, 120)).save(train_dog / "photo.jpg", format="PNG")

    _gradient().save(train_dog / "flat.png")
    _photo(rng, tint=(6, 6, 6), noise=2.0).save(train_dog / "night.png")
    _photo(rng, size=(320, 64), tint=(135, 125, 120)).save(train_dog / "panorama.png")

    grey = rng.integers(60, 190, size=(64, 64), dtype=np.uint8)
    Image.fromarray(np.stack([grey] * 3, axis=-1), mode="RGB").save(
        train_dog / "gray.png"
    )

    # Cut a real JPEG short so it only decodes with truncation tolerance.
    buffer = io.BytesIO()
    _photo(rng, tint=(140, 135, 125)).save(buffer, format="JPEG", quality=92)
    payload = buffer.getvalue()
    (train_cat / "truncated.jpg").write_bytes(payload[: int(len(payload) * 0.6)])

    (train_cat / "broken.png").write_bytes(b"not an image at all")
    return root


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for name, frame in load_sample().items():
        print(f"# {name}: {frame.shape[0]} rows x {frame.shape[1]} columns")
        print(frame.head().to_string(index=False))
        print()
