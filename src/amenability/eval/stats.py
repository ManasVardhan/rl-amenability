# src/amenability/eval/stats.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrelationResult:
    rho: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    r = stats.spearmanr(x, y).statistic
    return 0.0 if np.isnan(r) else float(r)


def spearman_with_ci(x, y, n_boot: int = 10000, seed: int = 0,
                     n_perm: int | None = None) -> CorrelationResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("spearman_with_ci needs at least three points")
    point = _rho(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boots[i] = _rho(x[idx], y[idx])
    # BCa is unstable when bootstrap resamples degenerate at small n, so use the
    # percentile interval and report it as such.
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return CorrelationResult(
        rho=point, ci_low=float(lo), ci_high=float(hi),
        p_value=permutation_p(x, y, n_perm=n_perm if n_perm is not None else n_boot, seed=seed),
        n=len(x),
    )


def permutation_p(x, y, n_perm: int = 10000, seed: int = 0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(_rho(x, y))
    rng = np.random.default_rng(seed)
    count = sum(abs(_rho(x, rng.permutation(y))) >= observed for _ in range(n_perm))
    return float((count + 1) / (n_perm + 1))


def loo_spearman(x, y) -> list[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        out.append(_rho(x[keep], y[keep]))
    return out


def partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after linearly residualising both on control ranks."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rc = stats.rankdata(control)

    def residualise(v: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(rc, v, 1)
        return v - (slope * rc + intercept)

    ex, ey = residualise(rx), residualise(ry)
    # If the control fully explains either variable, its residuals are floating-point
    # noise around 1e-15 rather than a real signal. Two such arrays correlate
    # perfectly, which would report a strong partial association at exactly the moment
    # the correct answer is that nothing survives the control. Rank values are O(n), so
    # genuine residual variation is many orders of magnitude above this threshold.
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return 0.0
    return _rho(ex, ey)
