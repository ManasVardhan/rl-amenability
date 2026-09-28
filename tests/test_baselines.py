import math
import pytest
from amenability.registry.loader import ModelSpec
from amenability.eval.passk import PassKResult
from amenability.scoring.baselines import (
    naive_extrapolation, baseline_pass_at_64, baseline_pass_at_1,
    baseline_param_count, ALL_BASELINES,
)


def test_extrapolation_recovers_a_known_log_curve():
    # y = 0.1 + 0.2 * log(1 + x)
    steps = list(range(1, 61))
    rewards = [0.1 + 0.2 * math.log(1 + s) for s in steps]
    expected = 0.1 + 0.2 * math.log(1 + 600)
    assert naive_extrapolation(steps, rewards, 600) == pytest.approx(expected, rel=1e-6)


def test_extrapolation_of_flat_trace_stays_flat():
    steps = list(range(1, 61))
    assert naive_extrapolation(steps, [0.4] * 60, 600) == pytest.approx(0.4, abs=1e-6)


def test_extrapolation_of_declining_trace_declines():
    steps = list(range(1, 61))
    rewards = [0.9 - 0.1 * math.log(1 + s) for s in steps]
    assert naive_extrapolation(steps, rewards, 600) < rewards[-1]


def test_extrapolation_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        naive_extrapolation([1], [0.5], 600)


def test_pass_at_64_baseline_reads_the_right_k():
    pre = [
        PassKResult(ks={1: 0.1, 32: 0.4, 64: 0.6}, n_samples=64, n_items=1, per_item_correct={}),
        PassKResult(ks={1: 0.2, 32: 0.3, 64: 0.5}, n_samples=64, n_items=1, per_item_correct={}),
    ]
    assert baseline_pass_at_64(pre) == [0.6, 0.5]
    assert baseline_pass_at_1(pre) == [0.1, 0.2]


def test_param_count_baseline():
    specs = [
        ModelSpec(key="a", hf_id="x/a", family="f", params_b=0.5, license="l", role="roster"),
        ModelSpec(key="b", hf_id="x/b", family="g", params_b=1.7, license="l", role="roster"),
    ]
    assert baseline_param_count(specs) == [0.5, 1.7]


def test_all_baselines_registry_is_complete():
    assert set(ALL_BASELINES) == {"naive_extrapolation", "pass_at_64", "pass_at_1", "param_count"}
