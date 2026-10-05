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


def test_no_available_breadth_raises_rather_than_scoring_zero_conversion():
    # 0.0 would be indistinguishable from a genuine "converted nothing" measurement
    # and is the hypothesis-favourable direction, so the absence of breadth must be
    # loud rather than silently scored.
    t = make_telemetry([1.0] * 10, [0.1] * 10, [0.2] * 10)
    t.pre = PassKResult(ks={1: 0.5, 32: 0.5}, n_samples=32, n_items=10, per_item_correct={})
    with pytest.raises(ValueError, match="no measurable breadth") as excinfo:
        extract_features(t)
    msg = str(excinfo.value)
    assert t.model_key in msg
    assert "0.5" in msg


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


def test_empty_steps_raises_rather_than_producing_nan():
    t = make_telemetry([], [], [])
    with pytest.raises(ValueError, match="zero steps"):
        extract_features(t)


def test_round_trip_json():
    t = make_telemetry([2.0] * 3, [0.1] * 3, [0.2] * 3)
    back = ProbeTelemetry.from_json(json.loads(t.to_json()))
    assert back.model_key == t.model_key
    assert len(back.steps) == 3
    assert back.pre.ks[32] == pytest.approx(0.50)


def test_non_finite_entropy_raises():
    t = make_telemetry([float('nan')] + [2.0] * 9, [0.1] * 10, [0.2] * 10)
    with pytest.raises(ValueError, match="non-finite policy_entropy"):
        extract_features(t)


def test_non_finite_kl_raises():
    t = make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10, kls=[float('inf')] + [0.01] * 9)
    with pytest.raises(ValueError, match="non-finite kl"):
        extract_features(t)


def test_non_finite_infinity_also_raises():
    t = make_telemetry([2.0] * 10, [float('inf')] + [0.1] * 9, [0.2] * 10)
    with pytest.raises(ValueError, match="non-finite mean_reward"):
        extract_features(t)


# Transfer-rulings T23: correctness-only step fields beside the training reward.

def test_correctness_fields_round_trip_and_default_to_none_on_old_files():
    t = make_telemetry([1.0] * 2, [0.1, 0.2], [0.5, 0.4])
    t.steps = [StepRecord(step=1, mean_reward=0.1, group_reward_std=0.1,
                          zero_advantage_frac=0.5, policy_entropy=1.0, kl=0.0,
                          grad_norm=1.0, mean_correct=0.02, zero_correct_group_frac=0.9)]
    back = ProbeTelemetry.from_json(json.loads(t.to_json()))
    assert back.steps[0].mean_correct == 0.02
    assert back.steps[0].zero_correct_group_frac == 0.9
    old = json.loads(t.to_json())
    for s in old["steps"]:
        del s["mean_correct"], s["zero_correct_group_frac"]
    assert ProbeTelemetry.from_json(old).steps[0].mean_correct is None


def test_features_do_not_read_the_correctness_fields():
    # The scored features describe the learning signal, i.e. the training reward.
    a = make_telemetry([1.0, 0.9, 0.8], [0.0, 0.05, 0.1], [0.9, 0.5, 0.3])
    b = make_telemetry([1.0, 0.9, 0.8], [0.0, 0.05, 0.1], [0.9, 0.5, 0.3])
    b.steps = [StepRecord(**{**s.__dict__, "mean_correct": 0.0, "zero_correct_group_frac": 1.0})
               for s in b.steps]
    assert extract_features(a) == extract_features(b)


def test_non_finite_correctness_is_refused():
    t = make_telemetry([1.0] * 2, [0.1, 0.2], [0.5, 0.4])
    t.steps = [StepRecord(**{**s.__dict__, "mean_correct": float("nan")}) for s in t.steps]
    with pytest.raises(ValueError, match="mean_correct"):
        extract_features(t)
