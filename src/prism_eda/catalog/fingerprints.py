"""Deterministic, bounded-cost dataset fingerprints."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

FINGERPRINT_METHOD = "schema-shape-and-deterministic-row-sample-v1"


def dataframe_fingerprint(frame: pd.DataFrame, sample_size: int = 1024) -> str:
    """Fingerprint schema, shape, and a deterministic spread of row values."""
    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "shape": list(frame.shape),
    }
    digest.update(json.dumps(schema, sort_keys=True).encode())

    if frame.empty:
        return digest.hexdigest()

    count = min(len(frame), sample_size)
    positions = np.linspace(0, len(frame) - 1, num=count, dtype=np.int64)
    sample = frame.iloc[np.unique(positions)]
    try:
        hashes = pd.util.hash_pandas_object(sample, index=True, categorize=True)
        digest.update(hashes.to_numpy().tobytes())
    except (TypeError, ValueError):
        digest.update(sample.astype(str).to_csv(index=True).encode())
    return digest.hexdigest()


def dataset_fingerprint(table_fingerprints: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, fingerprint in sorted(table_fingerprints.items()):
        digest.update(name.encode())
        digest.update(fingerprint.encode())
    return digest.hexdigest()
