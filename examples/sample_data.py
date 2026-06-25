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

import numpy as np
import pandas as pd

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


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for name, frame in load_sample().items():
        print(f"# {name}: {frame.shape[0]} rows x {frame.shape[1]} columns")
        print(frame.head().to_string(index=False))
        print()
