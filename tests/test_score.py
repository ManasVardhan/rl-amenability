import pytest
import numpy as np
from amenability.probe.telemetry import FeatureVector
from amenability.scoring.score import amenability_score, zscore


def fv(conv, ret, zar=0.0) -> FeatureVector:
    return FeatureVector(
        conversion_rate=conv, retention_factor=ret, zero_advantage_rate=zar,
        reward_slope_early=0.0, reward_slope_mid=0.0, grad_norm_trend=0.0,
        entropy_slope=0.0, entropy_collapse_step=None,
    )


def test_zscore_centres_and_scales():
    z = zscore([1.0, 2.0, 3.0])
    assert z[1] == pytest.approx(0.0)
    assert z[0] == pytest.approx(-z[2])
    # Population std is sqrt(2/3) ~ 0.816496580927726
    assert z[0] == pytest.approx(-1.224744871391589)
    assert z[1] == pytest.approx(0.0)
    assert z[2] == pytest.approx(1.224744871391589)
    # Verify unit variance: output standard deviation is 1.0
    assert np.std(z) == pytest.approx(1.0)


def test_zscore_of_constant_input_is_all_zeros():
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscore_of_single_value_is_zero():
    assert zscore([5.0]) == [0.0]


def test_higher_conversion_scores_higher():
    scores = amenability_score([fv(0.1, 1.0), fv(0.9, 1.0)])
    assert scores[1] > scores[0]


def test_higher_retention_scores_higher():
    scores = amenability_score([fv(0.5, 0.2), fv(0.5, 0.9)])
    assert scores[1] > scores[0]


def test_total_zero_advantage_floors_the_score():
    # Model 1 has the best conversion and retention but no learning signal at all.
    scores = amenability_score([fv(0.1, 0.1, zar=0.0), fv(0.9, 0.9, zar=1.0)])
    assert scores[1] == 0.0


def test_gate_scales_proportionally():
    a = amenability_score([fv(0.9, 0.9, zar=0.0), fv(0.1, 0.1, zar=0.0)])
    b = amenability_score([fv(0.9, 0.9, zar=0.5), fv(0.1, 0.1, zar=0.0)])
    assert b[0] == pytest.approx(a[0] * 0.5)


def test_gate_clips_out_of_range_rates():
    scores = amenability_score([fv(0.9, 0.9, zar=1.5), fv(0.1, 0.1, zar=0.0)])
    assert scores[0] == 0.0


def test_score_is_deterministic():
    feats = [fv(0.3, 0.6), fv(0.7, 0.2), fv(0.5, 0.5)]
    assert amenability_score(feats) == amenability_score(feats)


def test_empty_input_returns_empty():
    assert amenability_score([]) == []


def test_non_finite_conversion_rate_raises():
    with pytest.raises(ValueError, match="non-finite conversion_rate"):
        amenability_score([fv(float('nan'), 0.5)])


def test_non_finite_retention_factor_raises():
    with pytest.raises(ValueError, match="non-finite retention_factor"):
        amenability_score([fv(0.5, float('inf'))])


def test_non_finite_zero_advantage_rate_raises():
    with pytest.raises(ValueError, match="non-finite zero_advantage_rate"):
        amenability_score([fv(0.5, 0.5, zar=float('nan'))])
