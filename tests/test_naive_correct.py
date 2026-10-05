"""Gate B's naive baseline fits the strict correctness trace (transfer-rulings T23)."""
import pytest

from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import ProbeTelemetry, StepRecord
from amenability.scoring.baselines import naive_extrapolation
from amenability.scoring.naive_correct import baseline_naive_correct


def _t(rewards, correct):
    pk = PassKResult(ks={1: 0.1, 32: 0.5}, n_samples=32, n_items=1, per_item_correct={})
    steps = [StepRecord(step=i + 1, mean_reward=r, group_reward_std=0.1, zero_advantage_frac=0.1,
                        policy_entropy=1.0, kl=0.0, grad_norm=1.0, mean_correct=c,
                        zero_correct_group_frac=0.5)
             for i, (r, c) in enumerate(zip(rewards, correct))]
    return ProbeTelemetry(model_key="m", algorithm="grpo", steps=steps, pre=pk, post=pk)


def test_fits_the_correctness_trace_not_the_shaped_reward():
    shaped = [0.0, 0.05, 0.09, 0.1]     # format acquisition
    correct = [0.0, 0.0, 0.01, 0.01]
    got = baseline_naive_correct([_t(shaped, correct)], 600)
    assert got == [pytest.approx(naive_extrapolation([1, 2, 3, 4], correct, 600))]
    assert got[0] != pytest.approx(naive_extrapolation([1, 2, 3, 4], shaped, 600))


def test_telemetry_without_correctness_is_refused():
    t = _t([0.1, 0.2], [None, None])
    with pytest.raises(ValueError, match="mean_correct"):
        baseline_naive_correct([t], 600)
