"""PRE-REGISTERED. Frozen by prereg/freeze.py before any Stage 0 ground-truth run.

Any edit after the freeze invalidates Stage 0. Weights are equal and fixed a priori:
tuning them against ground truth at N=10 would fit noise, not signal.
"""
from __future__ import annotations

import numpy as np

from amenability.probe.telemetry import FeatureVector


def zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-12:
        return [0.0] * len(values)
    return [float(v) for v in (arr - arr.mean()) / sd]


def amenability_score(features: list[FeatureVector]) -> list[float]:
    if not features:
        return []
    z_conv = zscore([f.conversion_rate for f in features])
    z_ret = zscore([f.retention_factor for f in features])
    out: list[float] = []
    for f, zc, zr in zip(features, z_conv, z_ret):
        gate = float(np.clip(1.0 - f.zero_advantage_rate, 0.0, 1.0))
        out.append(gate * (zc + zr) / 2.0)
    return out
