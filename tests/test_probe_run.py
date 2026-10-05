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
    # Transfer-rulings T23: training uses the suite's shaped reward; evaluation
    # keeps the binary verifier (asserted just above).
    from amenability.suites.catalog import get_probe_suite
    assert state["train_specs"][0].train_reward_fn == get_probe_suite("graphpath").train_reward
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


def test_meta_and_preeval_record_the_vllm_sampler_setting(tmp_path, fakes, monkeypatch):
    # Ruling T17: every result says which vLLM sampler produced it.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _run(tmp_path, fakes)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert meta["vllm_use_flashinfer_sampler"] == "0"
    _run(tmp_path, fakes, pre_only=True)
    pre = json.loads(preeval_path(tmp_path, "m", "countdown").read_text())
    assert pre["vllm_use_flashinfer_sampler"] == "0"


def test_sampler_key_is_present_and_none_when_the_env_var_is_unset(tmp_path, fakes, monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    _run(tmp_path, fakes)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert "vllm_use_flashinfer_sampler" in meta and meta["vllm_use_flashinfer_sampler"] is None


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


def test_memory_fields_exist_and_are_none_without_cuda(tmp_path, fakes, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _run(tmp_path, fakes)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    for field in ("peak_memory_bytes", "peak_train_bytes", "allocated_after_train_bytes",
                  "vllm_sleep_pool_bytes", "peak_train_bytes_excl_vllm_pool"):
        assert field in meta and meta[field] is None


def test_memory_fields_measure_training_from_a_reset_peak(tmp_path, fakes, monkeypatch):
    """The peak is reset before training, so peak_train_bytes excludes the pre-eval."""
    import torch
    state, fake_passk, fake_train = fakes
    mem = {"peak": 0, "now": 0, "events": []}

    def passk(*a, **kw):
        mem["peak"] = max(mem["peak"], 900)   # evaluation peaks at 900
        return fake_passk(*a, **kw)

    def train(spec, **kw):
        mem["events"].append("train")
        mem["peak"], mem["now"] = max(mem["peak"], 500), 120   # training peaks at 500, leaves 120
        return fake_train(spec, **kw)

    def reset():
        mem["events"].append("reset")
        mem["peak"] = mem["now"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: mem["peak"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: mem["now"])
    _run(tmp_path, fakes, evaluate_fn=passk, train_fn=train)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert mem["events"] == ["reset", "train"]
    assert meta["peak_train_bytes"] == 500
    assert meta["allocated_after_train_bytes"] == 120
    assert meta["peak_memory_bytes"] == 900   # still the overall figure across the whole job
    # The fake manifest records no sleep pool, so there is nothing to subtract.
    assert meta["vllm_sleep_pool_bytes"] is None
    assert meta["peak_train_bytes_excl_vllm_pool"] is None


def test_peak_train_is_also_reported_without_the_vllm_sleep_pool(tmp_path, fakes, monkeypatch):
    """torch counts the colocated engine's sleep-mode pool as allocated even while
    its physical memory is released (transfer-rulings T22), so the raw training
    peak over-reports by the pool's size. The meta keeps the raw figure and adds
    the peak with the pool taken out, read from the run manifest."""
    import torch
    state, fake_passk, fake_train = fakes

    def train(spec, **kw):
        out = fake_train(spec, **kw)
        (Path(spec.output_dir) / "run_manifest.json").write_text(
            json.dumps({"vllm_sleep_pool_bytes": 12_000})
        )
        return out

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 45_000)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2_000)
    _run(tmp_path, fakes, train_fn=train)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert meta["peak_train_bytes"] == 45_000
    assert meta["vllm_sleep_pool_bytes"] == 12_000
    assert meta["peak_train_bytes_excl_vllm_pool"] == 33_000


def test_write_json_failure_leaves_no_partial_file_at_the_target(tmp_path, monkeypatch):
    from amenability.probe import run as run_mod
    target = tmp_path / "telemetry" / "x.json"
    run_mod._write_json(target, {"old": True})
    real_write_text = Path.write_text

    def half_write(self, data, *a, **kw):
        real_write_text(self, data[: len(data) // 2], *a, **kw)
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(Path, "write_text", half_write)
    with pytest.raises(OSError):
        run_mod._write_json(target, {"new": "x" * 1000})
    monkeypatch.undo()
    assert json.loads(target.read_text()) == {"old": True}   # the old file is intact, not truncated
    assert sorted(p.name for p in target.parent.iterdir()) == ["x.json"]   # no temp file left behind
    fresh = tmp_path / "telemetry" / "y.json"
    monkeypatch.setattr(Path, "write_text", half_write)
    with pytest.raises(OSError):
        run_mod._write_json(fresh, {"new": 1})
    monkeypatch.undo()
    assert not fresh.exists()


def test_meta_is_written_before_telemetry_so_a_failed_commit_is_not_cached(tmp_path, fakes, monkeypatch):
    # Telemetry is the cache marker, so it must be the last write. A job killed
    # after meta but before telemetry must rerun, not hit the cache without meta.
    from amenability.probe import run as run_mod
    state = fakes[0]
    real_write_json = run_mod._write_json
    t_path = telemetry_path(tmp_path, "m", "countdown", 0)

    def fail_on_telemetry(path, payload):
        if path == t_path:
            raise OSError("killed before telemetry")
        real_write_json(path, payload)

    monkeypatch.setattr(run_mod, "_write_json", fail_on_telemetry)
    with pytest.raises(OSError):
        _run(tmp_path, fakes)
    monkeypatch.undo()
    assert meta_path(tmp_path, "m", "countdown", 0).exists()
    assert not t_path.exists()
    assert len(state["train_specs"]) == 1
    _run(tmp_path, fakes)
    assert len(state["train_specs"]) == 2, "rerun hit the cache instead of retraining"
    assert t_path.exists() and meta_path(tmp_path, "m", "countdown", 0).exists()


def test_write_json_respects_the_umask_not_mkstemp_0600(tmp_path, monkeypatch):
    from amenability.probe import run as run_mod
    target = tmp_path / "telemetry" / "x.json"
    monkeypatch.setattr(run_mod, "_UMASK", 0o022)
    run_mod._write_json(target, {"a": 1})
    assert target.stat().st_mode & 0o777 == 0o644
    monkeypatch.setattr(run_mod, "_UMASK", 0o027)
    run_mod._write_json(target, {"a": 2})
    assert target.stat().st_mode & 0o777 == 0o640


def test_umask_is_read_once_at_import_and_matches_the_process_umask():
    import os
    from amenability.probe import run as run_mod
    current = os.umask(0)
    os.umask(current)
    assert run_mod._UMASK == current


@pytest.mark.parametrize("suite_key", ["countdown", "graphpath"])
def test_suite_stop_reaches_evaluation_and_training(tmp_path, fakes, suite_key):
    # Transfer-rulings T20: probe completions end at the first </answer>.
    stops = []
    state, fake_passk, fake_train = fakes

    def spy(model_path, items, verify_fn, ks, **kw):
        stops.append(kw.get("stop"))
        return fake_passk(model_path, items, verify_fn, ks, **kw)

    _run(tmp_path, fakes, suite_key=suite_key, evaluate_fn=spy)
    assert stops == ["</answer>", "</answer>"]
    assert state["train_specs"][0].stop == "</answer>"


def test_meta_records_the_shaped_training_reward(tmp_path, fakes):
    _run(tmp_path, fakes)
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert meta["train_reward"] == "shaped"
