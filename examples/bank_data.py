"""A small synthetic bank, built to exercise deep exploration.

Nothing here is real. The tables are shaped like the ones exploration is meant
for, and each carries a known property the tests assert on:

- ``accounts.acct_no`` is a zero-padded string; ``devices.acc`` holds the same
  account numbers as integers (a format mismatch only canonical matching finds).
- ``transactions.sender`` and ``transactions.receiver`` are both account
  numbers: a link between records of the same kind. Every sender is one of our
  accounts; only ``INTERNAL_RECEIVER_SHARE`` of receivers are.
- ``accounts.cust_ref`` references ``customers.customer_id``.
- ``audit_log.id`` is a serial that overlaps ``customers.customer_id`` by
  coincidence and must *not* be reported as a relationship.
- ``accounts.is_flagged`` marks a small set of accounts that pass money along
  chains, so labelled accounts have more counterparties than the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERNAL_RECEIVER_SHARE = 0.4
FLAGGED_SHARE = 0.02


def _acct(numbers: np.ndarray) -> list[str]:
    return [f"{int(number):012d}" for number in numbers]


def bank_tables(
    *, customers: int = 2_000, transactions: int = 60_000, seed: int = 7
) -> dict[str, pd.DataFrame]:
    """Return the synthetic bank as named DataFrames."""
    rng = np.random.default_rng(seed)
    accounts = int(customers * 1.5)

    customer_ids = np.arange(1, customers + 1)
    customer_frame = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "full_name": [f"Customer {index:05d}" for index in customer_ids],
            "phone": [f"9{rng.integers(10**8, 10**9):09d}" for _ in customer_ids],
            "city": rng.choice(
                ["Mumbai", "Pune", "Delhi", "Jaipur", "Kochi", "Indore"], customers
            ),
            "kyc_date": pd.Timestamp("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 900, customers), unit="D"),
            "risk_band": rng.choice(
                ["LOW", "MED", "HIGH"], customers, p=[0.7, 0.25, 0.05]
            ),
        }
    )

    account_numbers = 500_000_000 + rng.choice(40_000_000, accounts, replace=False)
    flagged = rng.random(accounts) < FLAGGED_SHARE
    account_frame = pd.DataFrame(
        {
            "acct_no": _acct(account_numbers),
            "cust_ref": rng.choice(customer_ids, accounts),
            "opened_on": pd.Timestamp("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 700, accounts), unit="D"),
            "status": rng.choice(
                ["ACTIVE", "DORMANT", "CLOSED"], accounts, p=[0.85, 0.1, 0.05]
            ),
            "branch_code": rng.choice(
                [f"BR{index:03d}" for index in range(60)], accounts
            ),
            "balance": np.round(rng.lognormal(9, 1.4, accounts), 2),
            "is_flagged": flagged.astype(int),
        }
    )

    device_count = int(accounts * 0.8)
    device_owner = rng.choice(account_numbers, device_count)
    device_frame = pd.DataFrame(
        {
            "device_id": [f"DV-{rng.integers(16**8):08X}" for _ in range(device_count)],
            # The same account numbers as accounts.acct_no, stored as integers.
            "acc": device_owner.astype("int64"),
            "registered_at": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(rng.integers(0, 500 * 24 * 60, device_count), unit="min"),
            "os": rng.choice(["android", "ios"], device_count, p=[0.8, 0.2]),
        }
    )

    # Money movement. Flagged accounts both receive and forward far more often.
    weights = np.where(flagged, 12.0, 1.0)
    weights = weights / weights.sum()
    senders = rng.choice(account_numbers, transactions, p=weights)
    internal = rng.random(transactions) < INTERNAL_RECEIVER_SHARE
    receivers = np.where(
        internal,
        rng.choice(account_numbers, transactions, p=weights),
        900_000_000 + rng.integers(0, 3_000_000, transactions),
    )
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 180 * 24 * 3600, transactions)), unit="s"
    )
    # A two-week outage: no rows at all, which the timeline must show.
    outage = (stamps >= "2024-03-10") & (stamps < "2024-03-24")
    amounts = np.round(rng.lognormal(7, 1.3, transactions), 2)
    amounts[rng.random(transactions) < 0.03] = 49_999.0
    transaction_frame = pd.DataFrame(
        {
            "txn_id": [f"T{index:010d}" for index in range(transactions)],
            "ts": stamps,
            "sender": _acct(senders),
            "receiver": _acct(receivers),
            "amount": amounts,
            "channel": rng.choice(
                ["UPI", "IMPS", "NEFT", "ATM"], transactions, p=[0.7, 0.15, 0.1, 0.05]
            ),
            "status": rng.choice(["SUCCESS", "FAILED"], transactions, p=[0.94, 0.06]),
            "remarks": rng.choice(["", "rent", "NA", "gift", None], transactions),
        }
    )
    transaction_frame = transaction_frame[~outage].reset_index(drop=True)

    audit_rows = customers * 2
    audit_frame = pd.DataFrame(
        {
            "id": np.arange(1, audit_rows + 1),
            "actor": rng.choice(["system", "ops1", "ops2"], audit_rows),
            "action": rng.choice(["LOGIN", "VIEW", "EXPORT"], audit_rows),
            "at": pd.Timestamp("2024-02-01")
            + pd.to_timedelta(rng.integers(0, 90 * 24 * 3600, audit_rows), unit="s"),
            "duration_ms": rng.integers(1, 5_000, audit_rows),
        }
    )

    return {
        "customers": customer_frame,
        "accounts": account_frame,
        "devices": device_frame,
        "transactions": transaction_frame,
        "audit_log": audit_frame,
        "empty_staging": pd.DataFrame(
            {"ref": pd.Series(dtype="object"), "n": pd.Series(dtype="int64")}
        ),
    }
