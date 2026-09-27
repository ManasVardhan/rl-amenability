# tests/test_stats.py
import numpy as np
import pytest
from amenability.eval.stats import (
    spearman_with_ci, permutation_p, loo_spearman, partial_spearman,
)


def test_perfect_monotonic_relationship_gives_rho_one():
    res = spearman_with_ci([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], n_boot=500, seed=0)
    assert res.rho == pytest.approx(1.0)
    assert res.n == 5


def test_perfect_inverse_relationship_gives_rho_minus_one():
    res = spearman_with_ci([1, 2, 3, 4, 5], [50, 40, 30, 20, 10], n_boot=500, seed=0)
    assert res.rho == pytest.approx(-1.0)


def test_ci_brackets_the_point_estimate():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    res = spearman_with_ci(x, y, n_boot=2000, seed=0)
    assert res.ci_low <= res.rho <= res.ci_high


def test_ci_is_wide_at_small_n():
    res = spearman_with_ci([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], n_boot=2000, seed=0)
    assert (res.ci_high - res.ci_low) > 0.5


def test_spearman_requires_at_least_three_points():
    with pytest.raises(ValueError, match="at least three"):
        spearman_with_ci([1, 2], [1, 2])


def test_permutation_p_is_small_for_strong_relationship():
    x = list(range(10))
    assert permutation_p(x, x, n_perm=2000, seed=0) < 0.01


def test_permutation_p_is_large_for_no_relationship():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 3, 8, 1, 9, 2, 7, 4, 10, 6]
    assert permutation_p(x, y, n_perm=2000, seed=0) > 0.05


def test_loo_returns_one_rho_per_dropped_point():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 2, 3, 4, 5, 6]
    rhos = loo_spearman(x, y)
    assert len(rhos) == 6
    assert all(r == pytest.approx(1.0) for r in rhos)


def test_loo_exposes_a_single_driving_outlier():
    # Without the last point there is no relationship at all.
    x = [1, 2, 3, 4, 100]
    y = [3, 1, 2, 4, 100]
    rhos = loo_spearman(x, y)
    assert min(rhos) < max(rhos)


def test_partial_correlation_collapses_a_shared_driver():
    # x and y are each control plus INDEPENDENT noise, so they look strongly
    # related until the shared driver is controlled for.
    rng = np.random.default_rng(0)
    n = 30
    control = np.arange(n, dtype=float)
    x = control + rng.normal(0, 3, n)
    y = control + rng.normal(0, 3, n)
    raw = spearman_with_ci(x, y, n_boot=500, n_perm=500).rho
    partial = partial_spearman(x, y, control)
    assert raw > 0.8                      # observed 0.877
    assert abs(partial) < abs(raw)        # observed -0.178
    assert abs(partial) < 0.5


def test_partial_correlation_preserves_a_genuine_association():
    # A real association that the control does NOT explain must survive.
    rng = np.random.default_rng(1)
    n = 30
    unrelated_control = rng.permutation(n).astype(float)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.2, n)
    raw = spearman_with_ci(a, b, n_boot=500, n_perm=500).rho
    partial = partial_spearman(a, b, unrelated_control)
    assert raw > 0.9
    assert partial > 0.8                  # stays high, control explains nothing


def test_partial_correlation_is_zero_when_control_fully_explains():
    # The degenerate case: control explains x and y completely, so nothing
    # survives the control and the correct answer is no association.
    # Without the residual-std guard this returns 1.0, the exact opposite.
    control = [1, 2, 3, 4, 5, 6, 7, 8]
    assert partial_spearman(control, control, control) == 0.0


def test_spearman_with_ci_honours_explicit_n_perm():
    # n_perm should override the n_boot-derived default and still produce a valid p-value.
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    res = spearman_with_ci(x, y, n_boot=2000, seed=0, n_perm=50)
    assert 0.0 < res.p_value <= 1.0
