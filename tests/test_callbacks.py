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
    d = rec.drain()
    assert d.zero_advantage_frac == pytest.approx(0.5)
    assert d.mean_reward == pytest.approx(0.75)


def test_all_uniform_groups_give_zero_advantage_one():
    rec = GroupedRewardRecorder(num_generations=4)
    rec.wrap(lambda completions, **kw: [0.0] * 8)(completions=["a"] * 8)
    assert rec.drain().zero_advantage_frac == pytest.approx(1.0)


def test_drain_resets_between_steps():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    fn(completions=["a", "b"])
    rec.drain()
    fn(completions=["a", "b"])
    assert rec.drain().mean_reward == pytest.approx(0.5)


def test_drain_with_no_calls_returns_zeros():
    rec = GroupedRewardRecorder(num_generations=2)
    assert rec.drain() == (0.0, 0.0, 0.0, None, None)


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


def _callback(rec: GroupedRewardRecorder) -> TelemetryCallback:
    return TelemetryCallback(
        recorder=rec,
        entropy_probe=type("P", (), {"measure": lambda self, m: 1.0})(),
        model_getter=lambda: None,
    )


class _State:
    def __init__(self, step: int) -> None:
        self.global_step = step
        self.log_history: list[dict] = []


def test_pending_tracks_whether_rewards_were_accumulated():
    rec = GroupedRewardRecorder(num_generations=2)
    assert rec.pending() is False
    rec.wrap(lambda completions, **kw: [0.0, 0.0])(completions=["a", "b"])
    # All-zero rewards still count as pending: a reward pass DID happen.
    assert rec.pending() is True
    rec.drain()
    assert rec.pending() is False


def test_post_training_log_with_nothing_pending_appends_no_record():
    # transformers' Trainer._finalize_training calls self.log(metrics) after
    # training ends, firing on_log with no preceding reward pass.
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    cb = _callback(rec)

    fn(completions=["a", "b"])
    cb.on_log(args=None, state=_State(1), control=None, logs={"kl": 0.01})
    assert len(cb.records) == 1

    # The phantom trailing log: no reward pass since the last drain.
    control = cb.on_log(args=None, state=_State(1), control="sentinel",
                        logs={"train_runtime": 12.3, "total_flos": 1e12})
    assert len(cb.records) == 1
    assert control == "sentinel"


def test_legitimate_all_zero_reward_step_still_appends_a_record():
    # The case a naive "skip if mean_reward == 0" check would wrongly drop.
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [0.0, 0.0, 0.0, 0.0])
    cb = _callback(rec)

    fn(completions=["a", "b", "c", "d"])
    cb.on_log(args=None, state=_State(7), control=None, logs={"kl": 0.03})

    assert len(cb.records) == 1
    assert cb.records[0].step == 7
    assert cb.records[0].mean_reward == pytest.approx(0.0)
    assert cb.records[0].zero_advantage_frac == pytest.approx(1.0)


def test_three_training_steps_plus_a_trailing_log_give_exactly_three_records():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    cb = _callback(rec)

    for step in (1, 2, 3):
        fn(completions=["a", "b"])
        cb.on_log(args=None, state=_State(step), control=None, logs={"kl": 0.01})
    cb.on_log(args=None, state=_State(3), control=None, logs={"train_runtime": 1.0})

    assert len(cb.records) == 3
    assert [r.step for r in cb.records] == [1, 2, 3]


# Transfer-rulings T23: correctness-only signal beside the training reward.

def test_correctness_is_drained_beside_the_training_reward():
    rec = GroupedRewardRecorder(num_generations=2)

    def reward(completions, **kw):
        rec.record_correct([0.0, 0.0, 1.0, 0.0])
        return [0.1, 0.0, 1.0, 0.1]

    rec.wrap(reward)(completions=["a"] * 4)
    d = rec.drain()
    assert d.mean_reward == pytest.approx(0.3)
    assert d.zero_advantage_frac == pytest.approx(0.0)
    assert d.mean_correct == pytest.approx(0.25)
    assert d.zero_correct_group_frac == pytest.approx(0.5)


def test_correctness_count_must_match_the_rewards():
    rec = GroupedRewardRecorder(num_generations=2)

    def reward(completions, **kw):
        rec.record_correct([0.0])
        return [0.1, 0.0]

    rec.wrap(reward)(completions=["a", "b"])
    with pytest.raises(ValueError, match="correctness"):
        rec.drain()


def test_callback_records_correctness_fields():
    rec = GroupedRewardRecorder(num_generations=2)

    def reward(completions, **kw):
        rec.record_correct([0.0, 0.0])
        return [0.1, 0.0]

    fn = rec.wrap(reward)
    cb = _callback(rec)
    fn(completions=["a", "b"])
    cb.on_log(args=None, state=_State(1), control=None, logs={})
    r = cb.records[0]
    assert r.mean_reward == pytest.approx(0.05)
    assert r.zero_advantage_frac == pytest.approx(0.0)
    assert r.mean_correct == pytest.approx(0.0)
    assert r.zero_correct_group_frac == pytest.approx(1.0)
