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
* ``subscriptions`` is the regression fixture: a right-skewed numeric target
  that is censored at a per-plan contract ceiling, an affine leak, a collinear
  feature pair, noise that widens with tenure, and two influential rows.
* ``daily_orders`` is the time-series fixture: a three-store daily panel with a
  nine-day outage, duplicate timestamps, a short-history store, a level shift,
  promotion spikes, and blank values. ``daily_orders_single`` is one store of it.
* ``customer_segments`` is the clustering fixture: four latent groups plus an
  identifier, a constant column, a 1,653x spread in feature scale, a perfectly
  redundant pair, duplicate rows, and missingness.

The last four are deliberately **not** part of :func:`load_sample`. The snippets
in ``docs/usage_docs/`` quote captured output, and adding a table to the shared
mapping would change the row, column, and table counts every existing example
prints. Load them on their own instead::

    from examples.sample_data import customer_segments, subscriptions
    import prism_eda as pe

    result = pe.load({"subscriptions": subscriptions()}).regression("monthly_revenue")
    groups = pe.load({"members": customer_segments()}).clustering()

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


#: Number of subscription accounts in the regression sample.
N_ACCOUNTS = 300

#: Contract revenue ceiling per plan. A real subscription contract caps what can
#: be billed in one month, so the recorded revenue piles up at exactly the
#: ceiling instead of continuing into a tail. That censoring is invisible in a
#: mean or a histogram bucket, and it makes the top of the range fictional.
PLAN_CAPS = {"starter": 200.0, "growth": 700.0, "enterprise": 2000.0}


def subscriptions() -> pd.DataFrame:
    """Return the seeded ``subscriptions`` table used by the regression guide.

    The target is ``monthly_revenue``. The planted problems are the ones a
    regression readiness check exists to catch:

    * ``renewal_invoice_total`` is ``monthly_revenue`` annualized — an affine
      copy of the target, and therefore a leak. Its name shares no token with
      the target, so only a value-based screen finds it.
    * ``account_id`` is unique per row and identifies records rather than
      explaining revenue.
    * ``monthly_revenue`` is censored at the per-plan ceiling in
      :data:`PLAN_CAPS`, so a real share of rows sit at exactly the cap.
    * ``seats`` and ``licenses_purchased`` are near-duplicates of one another
      (redundant, not wrong).
    * Residual noise widens with ``tenure_months`` — heteroscedasticity, which
      makes one uniform prediction interval wrong at both ends.
    * Two accounts carry extreme ``seats`` with unremarkable revenue: high
      leverage *and* a large residual, which is what influence actually means.

    Genuine signal comes from ``seats`` and ``plan``, so a diagnostic probe
    should do clearly better than predicting the median once the leak is
    screened out.

    Columns:
        account_id: Unique account key.
        industry: Categorical industry.
        region: Categorical region.
        plan: ``"starter"``, ``"growth"``, or ``"enterprise"``.
        seats: Purchased seats, right-skewed (log-normal).
        licenses_purchased: Near-copy of ``seats``.
        tenure_months: Account age in months, 1..60.
        support_tickets: Ticket count, ~7% missing.
        renewal_invoice_total: ``monthly_revenue * 12`` — the planted leak.
        monthly_revenue: The regression target.
    """
    rng = np.random.default_rng(SEED)
    n = N_ACCOUNTS

    account_id = np.arange(5000, 5000 + n)
    industry = rng.choice(
        ["retail", "saas", "finance", "health"], size=n, p=[0.30, 0.30, 0.25, 0.15]
    )
    region = rng.choice(["west", "east", "north", "south"], size=n)
    plan = rng.choice(["starter", "growth", "enterprise"], size=n, p=[0.5, 0.35, 0.15])

    # Log-normal seats give the target a genuine right skew rather than one
    # bolted on afterwards.
    seats = np.exp(rng.normal(3.0, 0.9, size=n)).round().clip(1, None).astype(int)
    licenses_purchased = (
        (seats + rng.normal(0.0, 0.6, size=n)).round().clip(1, None).astype(int)
    )
    tenure_months = rng.integers(1, 61, size=n)

    support_tickets = rng.poisson(2.0, size=n).astype(float)
    support_tickets[rng.choice(n, size=int(n * 0.07), replace=False)] = np.nan

    # Two accounts with enormous seat counts and ordinary revenue. Extreme in a
    # feature *and* badly fitted is what makes a row influential; either alone
    # is not enough.
    seats[7] = 420
    licenses_purchased[7] = 421
    seats[123] = 360
    licenses_purchased[123] = 359

    uplift = np.select(
        [plan == "starter", plan == "growth"], [0.0, 120.0], default=400.0
    )
    # Noise scale grows with tenure: older accounts have accumulated add-ons and
    # discounts, so their revenue is genuinely harder to predict.
    noise = rng.normal(0.0, 5.0 + tenure_months * 0.8)
    raw_revenue = 20.0 + seats * 3.5 + uplift + noise

    ceiling = np.array([PLAN_CAPS[value] for value in plan])
    monthly_revenue = np.clip(raw_revenue, 5.0, ceiling).round(2)

    # The influential pair: huge seat counts, unremarkable billing.
    monthly_revenue[7] = 140.0
    monthly_revenue[123] = 95.0

    return pd.DataFrame(
        {
            "account_id": account_id,
            "industry": industry,
            "region": region,
            "plan": plan,
            "seats": seats,
            "licenses_purchased": licenses_purchased,
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "renewal_invoice_total": (monthly_revenue * 12).round(2),
            "monthly_revenue": monthly_revenue,
        }
    )


# --------------------------------------------------------------------------
# Sample time series
# --------------------------------------------------------------------------

#: Stores in the panel, and how many days of history each one has. ``harbour``
#: opened recently, which is the point: a panel is rarely balanced, and an
#: entity with two months of history cannot support a seasonal forecast even
#: though the panel as a whole looks long enough.
STORE_HISTORY_DAYS = {"riverside": 730, "gate": 730, "harbour": 61}

#: The day the outage started, and how long it ran. Rows are absent entirely —
#: not present-and-null — which is what a collection failure actually looks
#: like and why a total missing count would report zero.
OUTAGE_START = "2025-04-07"
OUTAGE_DAYS = 9


def daily_orders() -> pd.DataFrame:
    """Return the seeded ``daily_orders`` panel used by the time-series guide.

    Two years of daily order counts for three stores. The value column is
    ``orders`` and the time column is ``order_date``.

    The genuine structure is a rising trend plus a weekly cycle — weekends are
    busier — because a forecasting-readiness check that cannot find real
    seasonality is not worth running.

    The planted problems are the ones that break a forecast before a model is
    ever chosen:

    * A nine-day **outage** starting ``2025-04-07``: the rows are missing
      entirely, so a naive missing-value count sees nothing wrong.
    * **Duplicate timestamps** — one store double-reported four days, so any
      resample silently sums or averages two records into one.
    * ``harbour`` has only 61 days of history against a two-year panel.
    * A **level shift**: a price change on ``2025-09-01`` steps ``riverside``
      down by about a third and holds it there.
    * Two **promotion spikes**, which are real events rather than errors.
    * A short run of missing ``orders`` values that are present-but-null, so the
      two kinds of absence can be told apart.
    """
    rng = np.random.default_rng(SEED)
    frames: list[pd.DataFrame] = []
    end = pd.Timestamp("2026-01-31")

    for store, history in STORE_HISTORY_DAYS.items():
        dates = pd.date_range(end=end, periods=history, freq="D")
        step = np.arange(history, dtype="float64")

        base = {"riverside": 180.0, "gate": 120.0, "harbour": 90.0}[store]
        trend = base + step * 0.06
        # Weekly cycle: Saturday and Sunday carry the week.
        weekday = dates.dayofweek.to_numpy()
        weekly = np.where(weekday >= 5, 26.0, -9.0) + np.where(weekday == 4, 12.0, 0.0)
        noise = rng.normal(0.0, 7.0, size=history)
        values = trend + weekly + noise

        if store == "riverside":
            # A price change steps the level down and it never recovers. This is
            # a change point, not an outlier: every later day is affected.
            after = dates >= pd.Timestamp("2025-09-01")
            values = np.where(after, values * 0.68, values)
            # Promotions: real events, genuinely high, not data errors.
            values[dates == pd.Timestamp("2025-06-14")] *= 2.4
            values[dates == pd.Timestamp("2025-11-28")] *= 2.6

        frame = pd.DataFrame(
            {
                "order_date": dates,
                "store": store,
                "orders": np.rint(np.clip(values, 0.0, None)),
            }
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)

    # The outage removes rows outright, for every store.
    outage = pd.date_range(OUTAGE_START, periods=OUTAGE_DAYS, freq="D")
    panel = panel[~panel["order_date"].isin(outage)].reset_index(drop=True)

    # Present-but-null values, so a blank reading is distinguishable from a day
    # that was never recorded at all.
    blanks = (panel["store"] == "gate") & panel["order_date"].between(
        "2025-12-02", "2025-12-05"
    )
    panel.loc[blanks, "orders"] = np.nan

    # One store double-reported four days.
    repeated = panel[
        (panel["store"] == "gate")
        & panel["order_date"].isin(
            pd.to_datetime(["2025-02-10", "2025-02-11", "2025-07-03", "2025-10-19"])
        )
    ]
    panel = pd.concat([panel, repeated], ignore_index=True)

    return panel.sort_values(["order_date", "store"]).reset_index(drop=True)


def daily_orders_single() -> pd.DataFrame:
    """The ``riverside`` store alone — a single series, for the simple examples."""
    panel = daily_orders()
    single = panel[panel["store"] == "riverside"].drop(columns=["store"])
    return single.reset_index(drop=True)


# --------------------------------------------------------------------------
# Sample clustering data
# --------------------------------------------------------------------------

#: Latent segments in the clustering sample, as
#: ``name -> (annual_spend, visits_per_month, tenure_days, share)``. These are
#: the centres the data is generated around; nothing in the table records which
#: segment a row came from, which is the point.
SEGMENT_CENTRES = {
    "budget": (420.0, 1.5, 240.0, 0.34),
    "regular": (1_850.0, 6.0, 1_150.0, 0.30),
    "premium": (7_400.0, 4.0, 1_500.0, 0.18),
    "lapsed": (1_500.0, 0.4, 1_900.0, 0.18),
}

#: Members in the clustering sample.
N_MEMBERS = 600


def customer_segments() -> pd.DataFrame:
    """Return the seeded ``customer_segments`` table used by the clustering guide.

    Four genuine groups are generated from the centres in
    :data:`SEGMENT_CENTRES`, well enough separated that a clustering run should
    find them — a clusterability check that cannot recognise real structure is
    not worth running.

    The planted problems are the ones that quietly ruin a distance-based
    clustering:

    * ``member_id`` is unique per row. Left in the feature set it contributes a
      dimension in which every point is equidistant from every other.
    * ``annual_spend`` spans thousands while ``visits_per_month`` spans single
      digits. Unscaled, the distance between two members is essentially the
      difference in their spend and nothing else.
    * ``visits_per_year`` is ``visits_per_month`` in different units, so the
      visit frequency is present twice — and a duplicated dimension is silently
      double-weighted in a Euclidean distance.
    * ``account_status`` is the same value in every row and contributes nothing.
    * Twelve rows are exact duplicates, which tighten one region of space and
      pull a centroid toward it.
    * ``satisfaction`` is about 9% missing.

    Columns:
        member_id: Unique member key.
        region: Categorical region, unrelated to the segments.
        annual_spend: Spend in currency units (thousands scale).
        visits_per_month: Visit frequency (single-digit scale).
        tenure_days: Days since joining (hundreds to thousands).
        visits_per_year: The same quantity as ``visits_per_month``, rescaled.
        satisfaction: 1-10 rating, ~9% missing.
        account_status: Constant.
    """
    rng = np.random.default_rng(SEED)

    names: list[str] = []
    spend: list[float] = []
    visits: list[float] = []
    tenure: list[float] = []
    for name, (
        centre_spend,
        centre_visits,
        centre_tenure,
        share,
    ) in SEGMENT_CENTRES.items():
        count = int(round(N_MEMBERS * share))
        names.extend([name] * count)
        # Spread is proportional to the centre so every segment is equally
        # recognisable in relative terms rather than the large-spend one being
        # artificially diffuse.
        spend.extend(rng.normal(centre_spend, centre_spend * 0.16, count))
        visits.extend(rng.normal(centre_visits, 0.45, count))
        tenure.extend(rng.normal(centre_tenure, 150.0, count))

    total = len(names)
    annual_spend = np.clip(np.array(spend), 20.0, None).round(2)
    visits_per_month = np.clip(np.array(visits), 0.0, None).round(2)
    tenure_days = np.clip(np.array(tenure), 30.0, None).round().astype(int)

    visits_per_year = (visits_per_month * 12.0 + rng.normal(0.0, 0.05, total)).round(2)

    satisfaction = rng.integers(4, 11, size=total).astype(float)
    satisfaction[rng.choice(total, size=int(total * 0.09), replace=False)] = np.nan

    frame = pd.DataFrame(
        {
            "member_id": np.arange(90_000, 90_000 + total),
            "region": rng.choice(["west", "east", "north", "south"], size=total),
            "annual_spend": annual_spend,
            "visits_per_month": visits_per_month,
            "tenure_days": tenure_days,
            "visits_per_year": visits_per_year,
            "satisfaction": satisfaction,
            "account_status": "active",
        }
    )

    # Twelve exact duplicates, member_id included: the same record loaded twice.
    repeated = frame.iloc[rng.choice(total, size=12, replace=False)]
    frame = pd.concat([frame, repeated], ignore_index=True)
    return frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


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
