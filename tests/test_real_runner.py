"""Persistence and resume for RealRunner, exercised with injected fakes.

RealRunner.__init__ downloads GSM8K and reads the roster, neither of which is
needed to test that telemetry and outcomes reach disk and are reused. The
instance is therefore built directly and the two expensive collaborators,
evaluate_passk and run_grpo, are replaced with fakes that count their calls.
"""
import json

import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem

import scripts.real_runner as real_runner
from scripts.real_runner import RealRunner
from scripts.run_stage0 import Stage0Config


def _items() -> list[TaskItem]:
    return [
        TaskItem(task_id=f"t{i}", suite="probe_countdown", prompt=f"p{i}",
                 answer=f"a{i}", difficulty=1)
        for i in range(3)
    ]


def _passk(p1: float, p32: float) -> PassKResult:
    return PassKResult(ks={1: p1, 8: p1, 32: p32, 64: p32}, n_samples=64,
                       n_items=3, per_item_correct={"t0": 1, "t1": 0, "t2": 0})


@pytest.fixture
def runner(tmp_path, monkeypatch):
    calls = {"passk": 0, "grpo": 0}

    def fake_passk(model_path, items, verify_fn, ks, **kw):
        calls["passk"] += 1
        # First call of each pair is "pre", second is "post" with a higher pass@1.
        return _passk(0.1 if calls["passk"] % 2 == 1 else 0.3, 0.5)

    def fake_grpo(spec, **kw):
        calls["grpo"] += 1
        return [
            StepRecord(step=i, mean_reward=0.1 * i, group_reward_std=0.2,
                       zero_advantage_frac=0.1, policy_entropy=2.0 - 0.01 * i,
                       kl=0.01, grad_norm=1.0)
            for i in range(1, 4)
        ]

    monkeypatch.setattr(real_runner, "evaluate_passk", fake_passk)
    monkeypatch.setattr(real_runner, "run_grpo", fake_grpo)

    r = RealRunner.__new__(RealRunner)
    r.config = Stage0Config()
    r.work_dir = tmp_path
    r.registry = {}
    r.probe_items = _items()
    r.target_items = _items()
    r.target_eval = _items()
    r.calls = calls
    return r


def test_probe_writes_its_telemetry_to_disk(runner, tmp_path):
    t = runner.probe("qwen2.5-0.5b-oversft1x")
    path = tmp_path / "telemetry" / "qwen2.5-0.5b-oversft1x-grpo.json"
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["model_key"] == "qwen2.5-0.5b-oversft1x"
    assert written["algorithm"] == "grpo"
    assert len(written["steps"]) == len(t.steps) == 3


def test_probe_reuses_cached_telemetry_instead_of_rerunning(runner):
    first = runner.probe("v1")
    grpo_calls = runner.calls["grpo"]
    second = runner.probe("v1")
    assert runner.calls["grpo"] == grpo_calls, "cached probe re-invoked the trainer"
    assert second.model_key == first.model_key
    assert [s.mean_reward for s in second.steps] == [s.mean_reward for s in first.steps]
    assert second.pre.ks[32] == first.pre.ks[32]


def test_probe_force_bypasses_the_cache(runner):
    runner.probe("v1")
    grpo_calls = runner.calls["grpo"]
    runner.probe("v1", force=True)
    assert runner.calls["grpo"] == grpo_calls + 1


def test_full_run_writes_pre_post_and_outcome(runner, tmp_path):
    outcome = runner.full_run("v1")
    path = tmp_path / "outcomes" / "v1.json"
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["outcome"] == pytest.approx(outcome)
    # (0.3 - 0.1) / (0.5 - 0.1)
    assert outcome == pytest.approx(0.5)
    assert written["pre"]["ks"]["1"] == pytest.approx(0.1)
    assert written["post"]["ks"]["1"] == pytest.approx(0.3)
    assert written["outcome_secondary"] == pytest.approx(0.2)


def test_full_run_reuses_cached_outcome_instead_of_rerunning(runner):
    first = runner.full_run("v1")
    grpo_calls = runner.calls["grpo"]
    assert runner.full_run("v1") == pytest.approx(first)
    assert runner.calls["grpo"] == grpo_calls, "cached full run re-invoked the trainer"


def test_full_run_force_bypasses_the_cache(runner):
    runner.full_run("v1")
    grpo_calls = runner.calls["grpo"]
    runner.full_run("v1", force=True)
    assert runner.calls["grpo"] == grpo_calls + 1


def test_full_run_raises_when_the_target_suite_has_no_breadth(runner, monkeypatch):
    monkeypatch.setattr(real_runner, "evaluate_passk",
                        lambda *a, **kw: _passk(0.5, 0.5))
    with pytest.raises(ValueError, match="no measurable breadth") as excinfo:
        runner.full_run("v1")
    assert "v1" in str(excinfo.value)
