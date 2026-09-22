# tests/test_telemetry.py
import json
import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import (
    StepRecord, ProbeTelemetry, extract_features,
)


def make_telemetry(entropies, rewards, zero_adv, kls=None, grads=None) -> ProbeTelemetry:
    n = len(entropies)
    kls = kls or [0.01] * n
    grads = grads or [1.0] * n
    steps = [
        StepRecord(step=i, mean_reward=rewards[i], group_reward_std=0.1,
                   zero_advantage_frac=zero_adv[i], policy_entropy=entropies[i],
                   kl=kls[i], grad_norm=grads[i])
        for i in range(n)
    ]
    return ProbeTelemetry(
        model_key="m", algorithm="grpo", steps=steps,
        pre=PassKResult(ks={1: 0.10, 32: 0.50}, n_samples=32, n_items=10, per_item_correct={}),
        post=PassKResult(ks={1: 0.30, 32: 0.55}, n_samples=32, n_items=10, per_item_correct={}),
    )


def test_conversion_rate_normalises_by_available_breadth():
    t = make_telemetry([1.0] * 10, [0.1] * 10, [0.2] * 10)
    f = extract_features(t)
    # (post@1 - pre@1) / (pre@32 - pre@1) = (0.30 - 0.10) / (0.50 - 0.10)
    assert f.conversion_rate == pytest.approx(0.5)


def test_conversion_rate_is_zero_when_no_breadth_available():
    t = make_telemetry([1.0] * 10, [0.1] * 10, [0.2] * 10)
    t.pre = PassKResult(ks={1: 0.5, 32: 0.5}, n_samples=32, n_items=10, per_item_correct={})
    assert extract_features(t).conversion_rate == 0.0


def test_entropy_slope_is_negative_under_collapse():
    entropies = [2.0 - 0.15 * i for i in range(10)]
    f = extract_features(make_telemetry(entropies, [0.1] * 10, [0.2] * 10))
    assert f.entropy_slope < 0


def test_entropy_slope_is_flat_when_stable():
    f = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    assert f.entropy_slope == pytest.approx(0.0, abs=1e-9)


def test_entropy_collapse_step_detected_at_half_of_initial():
    entropies = [2.0] * 5 + [0.9] * 5  # drops below 50% of initial at step 5
    f = extract_features(make_telemetry(entropies, [0.1] * 10, [0.2] * 10))
    assert f.entropy_collapse_step == 5


def test_entropy_collapse_step_is_none_when_no_collapse():
    f = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    assert f.entropy_collapse_step is None


def test_zero_advantage_rate_is_mean_over_steps():
    f = extract_features(make_telemetry([1.0] * 4, [0.1] * 4, [0.0, 0.2, 0.4, 0.6]))
    assert f.zero_advantage_rate == pytest.approx(0.3)


def test_retention_factor_penalises_entropy_collapse():
    stable = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    collapsing = extract_features(
        make_telemetry([2.0 - 0.18 * i for i in range(10)], [0.1] * 10, [0.2] * 10)
    )
    assert collapsing.retention_factor < stable.retention_factor


def test_retention_factor_penalises_high_kl_per_reward_gain():
    cheap = make_telemetry([2.0] * 10, [0.1 + 0.02 * i for i in range(10)], [0.2] * 10,
                           kls=[0.01] * 10)
    costly = make_telemetry([2.0] * 10, [0.1 + 0.02 * i for i in range(10)], [0.2] * 10,
                            kls=[0.5] * 10)
    assert extract_features(costly).retention_factor < extract_features(cheap).retention_factor


def test_reward_slope_early_and_mid_split_the_run():
    rewards = [0.0] * 5 + [1.0] * 5  # flat then jump: early slope 0, mid slope 0
    f = extract_features(make_telemetry([2.0] * 10, rewards, [0.2] * 10))
    assert f.reward_slope_early == pytest.approx(0.0, abs=1e-9)


def test_round_trip_json():
    t = make_telemetry([2.0] * 3, [0.1] * 3, [0.2] * 3)
    back = ProbeTelemetry.from_json(json.loads(t.to_json()))
    assert back.model_key == t.model_key
    assert len(back.steps) == 3
    assert back.pre.ks[32] == pytest.approx(0.50)
