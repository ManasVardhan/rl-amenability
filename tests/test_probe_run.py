"""run_probe with fakes for the two GPU collaborators."""
import json
from pathlib import Path

import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.run import (
    ProbeConfig, meta_path, passk_by_bucket, preeval_path, probe_key, run_probe,
    telemetry_path,
)
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem


def _passk(items, p1, p32):
    return PassKResult(
        ks={1: p1, 8: p1, 32: p32, 64: p32}, n_samples=64, n_items=len(items),
        per_item_correct={it.task_id: (1 if i % 2 == 0 else 0) for i, it in enumerate(items)},
    )


@pytest.fixture
def fakes():
    state = {"passk_calls": 0, "train_specs": [], "passk_items": []}

    def fake_passk(model_path, items, verify_fn, ks, **kw):
        state["passk_calls"] += 1
        state["passk_items"].append(items)
        state["verify_fn"] = verify_fn
        return _passk(items, 0.1 if state["passk_calls"] % 2 == 1 else 0.3, 0.5)

    def fake_train(spec, **kw):
        state["train_specs"].append(spec)
        Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(spec.output_dir) / "run_manifest.json").write_text(json.dumps({"seed": spec.seed}))
        (Path(spec.output_dir) / "model.safetensors").write_bytes(b"weights")
        return [
            StepRecord(step=i, mean_reward=0.1 * i, group_reward_std=0.2,
                       zero_advantage_frac=0.1, policy_entropy=2.0, kl=0.01, grad_norm=1.0)
            for i in range(1, 4)
        ]

    return state, fake_passk, fake_train


def _run(tmp_path, fakes, **kw):
    state, fake_passk, fake_train = fakes
    args = dict(model_path="hf/x", model_key="m", suite_key="countdown", work_dir=tmp_path,
                config=ProbeConfig(n_probe_items=9), evaluate_fn=fake_passk, train_fn=fake_train)
    args.update(kw)
    return run_probe(**args)


def test_paths_carry_suite_and_seed(tmp_path):
    assert probe_key("m", "graphpath", 1) == "m-graphpath-grpo-s1"
    assert telemetry_path(tmp_path, "m", "graphpath", 1) == tmp_path / "telemetry" / "m-graphpath-grpo-s1.json"
    assert preeval_path(tmp_path, "m", "graphpath") == tmp_path / "preeval" / "m-graphpath.json"
    assert meta_path(tmp_path, "m", "graphpath", 1) == tmp_path / "meta" / "m-graphpath-grpo-s1.json"


def test_writes_telemetry_meta_and_manifest_then_deletes_the_checkpoint(tmp_path, fakes):
    t = _run(tmp_path, fakes)
    assert t.model_key == "m" and len(t.steps) == 3
    assert telemetry_path(tmp_path, "m", "countdown", 0).exists()
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert meta["suite_key"] == "countdown" and meta["run_seed"] == 0 and meta["item_seed"] == 0
    assert meta["learning_rate"] == pytest.approx(1e-6) and meta["probe_steps"] == 60
    assert set(meta["wall_seconds"]) == {"pre", "train", "post"}
    assert (tmp_path / "manifests" / "m-countdown-grpo-s0.json").exists()
    assert not (tmp_path / "probes" / "m-countdown-grpo-s0").exists(), "checkpoint not deleted"


def test_keep_checkpoint_leaves_the_directory(tmp_path, fakes):
    _run(tmp_path, fakes, keep_checkpoint=True)
    assert (tmp_path / "probes" / "m-countdown-grpo-s0" / "model.safetensors").exists()


def test_cache_is_keyed_by_suite_and_seed(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes)
    assert len(state["train_specs"]) == 1
    _run(tmp_path, fakes)                          # same suite, same seed: cached
    assert len(state["train_specs"]) == 1
    _run(tmp_path, fakes, suite_key="graphpath")   # other suite: must NOT hit the cache
    assert len(state["train_specs"]) == 2
    _run(tmp_path, fakes, run_seed=1)              # other seed: must NOT hit the cache
    assert len(state["train_specs"]) == 3
    _run(tmp_path, fakes, force=True)
    assert len(state["train_specs"]) == 4


def test_run_seed_changes_the_trainer_seed_but_not_the_items(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes, run_seed=0)
    _run(tmp_path, fakes, run_seed=1)
    s0, s1 = state["train_specs"]
    assert s0.seed == 0 and s1.seed == 1
    assert [it.prompt for it in s0.items] == [it.prompt for it in s1.items]
    assert [it.task_id for it in s0.items] == [it.task_id for it in s1.items]


def test_item_seed_changes_the_items(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes, item_seed=0)
    _run(tmp_path, fakes, item_seed=5, run_seed=0, force=True)
    a, b = state["train_specs"]
    assert [it.prompt for it in a.items] != [it.prompt for it in b.items]


def test_suite_verifier_reaches_both_evaluation_and_training(tmp_path, fakes):
    from amenability.suites.graphpath import verify_graphpath
    state = fakes[0]
    _run(tmp_path, fakes, suite_key="graphpath")
    assert state["verify_fn"] is verify_graphpath
    assert state["train_specs"][0].verify_fn is verify_graphpath
    assert all(it.suite == "probe_graphpath" for it in state["train_specs"][0].items)


def test_post_eval_targets_the_trained_checkpoint(tmp_path, fakes, monkeypatch):
    seen = []
    state, fake_passk, fake_train = fakes

    def spy(model_path, items, verify_fn, ks, **kw):
        seen.append(model_path)
        return fake_passk(model_path, items, verify_fn, ks, **kw)

    _run(tmp_path, fakes, evaluate_fn=spy)
    assert seen == ["hf/x", str(tmp_path / "probes" / "m-countdown-grpo-s0")]


def test_pre_only_writes_by_bucket_and_never_trains(tmp_path, fakes):
    state = fakes[0]
    res = _run(tmp_path, fakes, pre_only=True)
    assert isinstance(res, PassKResult)
    assert state["train_specs"] == []
    written = json.loads(preeval_path(tmp_path, "m", "countdown").read_text())
    assert written["model_key"] == "m" and written["suite_key"] == "countdown"
    assert sorted(int(b) for b in written["by_bucket"]) == [3, 4, 5]
    for bucket in written["by_bucket"].values():
        assert set(bucket) == {"1", "8", "32", "64"}
    # cached on the second call
    _run(tmp_path, fakes, pre_only=True)
    assert state["passk_calls"] == 1


def test_passk_by_bucket_uses_the_unbiased_estimator():
    items = [TaskItem(task_id=f"t{i}", suite="s", prompt="p", answer="a", difficulty=1 + i // 2)
             for i in range(4)]
    res = PassKResult(ks={}, n_samples=4, n_items=4,
                      per_item_correct={"t0": 4, "t1": 0, "t2": 2, "t3": 2})
    out = passk_by_bucket(res, items, (1, 4))
    assert out[1][1] == pytest.approx(0.5)       # (1.0 + 0.0) / 2
    assert out[1][4] == pytest.approx(0.5)       # pass@4 with n=4: (1.0 + 0.0) / 2
    assert out[2][1] == pytest.approx(0.5)       # (0.5 + 0.5) / 2
    assert out[2][4] == pytest.approx(1.0)       # 2 of 4 correct, k = n: guaranteed
