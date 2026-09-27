import pytest
from amenability.probe.callbacks import GroupedRewardRecorder, TelemetryCallback


def test_wrap_passes_through_reward_values():
    rec = GroupedRewardRecorder(num_generations=2)
    wrapped = rec.wrap(lambda completions, **kw: [1.0, 0.0, 1.0, 1.0])
    assert wrapped(completions=["a", "b", "c", "d"]) == [1.0, 0.0, 1.0, 1.0]


def test_zero_advantage_fraction_counts_uniform_groups():
    rec = GroupedRewardRecorder(num_generations=2)
    # group 0 = [1, 0] varied; group 1 = [1, 1] uniform -> 0.5 zero-advantage
    rec.wrap(lambda completions, **kw: [1.0, 0.0, 1.0, 1.0])(completions=["a"] * 4)
    mean_r, group_std, zero_frac = rec.drain()
    assert zero_frac == pytest.approx(0.5)
    assert mean_r == pytest.approx(0.75)


def test_all_uniform_groups_give_zero_advantage_one():
    rec = GroupedRewardRecorder(num_generations=4)
    rec.wrap(lambda completions, **kw: [0.0] * 8)(completions=["a"] * 8)
    _, _, zero_frac = rec.drain()
    assert zero_frac == pytest.approx(1.0)


def test_drain_resets_between_steps():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    fn(completions=["a", "b"])
    rec.drain()
    fn(completions=["a", "b"])
    mean_r, _, _ = rec.drain()
    assert mean_r == pytest.approx(0.5)


def test_drain_with_no_calls_returns_zeros():
    rec = GroupedRewardRecorder(num_generations=2)
    assert rec.drain() == (0.0, 0.0, 0.0)


def test_ragged_batch_raises():
    rec = GroupedRewardRecorder(num_generations=3)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    with pytest.raises(ValueError, match="not divisible"):
        fn(completions=["a", "b"])


def test_callback_emits_one_record_per_logged_step():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    cb = TelemetryCallback(
        recorder=rec,
        entropy_probe=type("P", (), {"measure": lambda self, m: 1.23})(),
        model_getter=lambda: None,
    )

    class State:
        global_step = 1
        log_history = [{"kl": 0.02, "grad_norm": 0.7}]

    fn(completions=["a", "b"])
    cb.on_log(args=None, state=State(), control=None, logs={"kl": 0.02, "grad_norm": 0.7})
    assert len(cb.records) == 1
    r = cb.records[0]
    assert r.step == 1
    assert r.policy_entropy == pytest.approx(1.23)
    assert r.kl == pytest.approx(0.02)
    assert r.grad_norm == pytest.approx(0.7)
    assert r.zero_advantage_frac == pytest.approx(0.0)


def test_callback_tolerates_missing_log_fields():
    rec = GroupedRewardRecorder(num_generations=2)
    rec.wrap(lambda completions, **kw: [1.0, 0.0])(completions=["a", "b"])
    cb = TelemetryCallback(
        recorder=rec,
        entropy_probe=type("P", (), {"measure": lambda self, m: 0.5})(),
        model_getter=lambda: None,
    )

    class State:
        global_step = 3
        log_history = []

    cb.on_log(args=None, state=State(), control=None, logs={})
    assert cb.records[0].kl == 0.0
    assert cb.records[0].grad_norm == 0.0
