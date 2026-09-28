import gc
import json
import weakref

import pytest
from amenability.suites.base import TaskItem
from amenability.training.grpo import (
    GRPOSpec, checkpoint_schedule, build_reward_fn, run_grpo,
)


def items(n=4):
    return [
        TaskItem(task_id=f"probe/x/{i}", suite="s", prompt=f"p{i}", answer="7", difficulty=1)
        for i in range(n)
    ]


def verify(item, completion):
    return completion.strip() == item.answer


def test_probe_schedule_has_no_intermediate_checkpoints():
    assert checkpoint_schedule(max_steps=60, save_steps=None) == [60]


def test_full_run_schedule_checkpoints_every_200():
    assert checkpoint_schedule(max_steps=600, save_steps=200) == [200, 400, 600]


def test_schedule_always_includes_final_step():
    assert checkpoint_schedule(max_steps=500, save_steps=200) == [200, 400, 500]


def test_reward_fn_scores_by_verifier_against_matching_prompt():
    fn = build_reward_fn(items(2), verify)
    rewards = fn(completions=["7", "8"], prompts=["p0", "p1"])
    assert rewards == [1.0, 0.0]


def test_reward_fn_handles_repeated_prompts_from_group_sampling():
    fn = build_reward_fn(items(1), verify)
    rewards = fn(completions=["7", "7", "1", "7"], prompts=["p0"] * 4)
    assert rewards == [1.0, 1.0, 0.0, 1.0]


def test_reward_fn_scores_zero_for_unknown_prompt():
    fn = build_reward_fn(items(1), verify)
    assert fn(completions=["7"], prompts=["not-a-prompt"]) == [0.0]


def test_run_grpo_returns_telemetry_from_injected_trainer():
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = object()

        def train(self):
            # Simulate three logging steps. Each real step scores a group of
            # completions BEFORE the trainer logs, and the telemetry callback now
            # requires that, so the fake must do it too.
            for cb in captured["callbacks"]:
                for step in (1, 2, 3):
                    captured["reward_funcs"][0](
                        completions=["7", "8"], prompts=["p0", "p0"]
                    )
                    cb.on_log(
                        args=None,
                        state=type("S", (), {"global_step": step, "log_history": []})(),
                        control=None,
                        logs={"kl": 0.01 * step, "grad_norm": 1.0},
                    )

        def save_model(self, path):
            captured["saved_to"] = path

    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir="/tmp/out", save_steps=None,
    )
    records = run_grpo(spec, trainer_factory=lambda **kw: FakeTrainer(**kw))
    assert [r.step for r in records] == [1, 2, 3]
    assert captured["saved_to"] == "/tmp/out"


def test_run_grpo_rejects_lora_config():
    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir="/tmp/out", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_grpo(spec, trainer_factory=lambda **kw: None, peft_config={"r": 8})


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.captured = kwargs
        self.model = object()

    def train(self):
        for cb in self.captured["callbacks"]:
            for step in (1, 2, 3):
                # A real step scores a group of completions before logging.
                self.captured["reward_funcs"][0](
                    completions=["7", "8"], prompts=["p0", "p0"]
                )
                cb.on_log(
                    args=None,
                    state=type("S", (), {"global_step": step, "log_history": []})(),
                    control=None,
                    logs={"kl": 0.01 * step, "grad_norm": 1.0},
                )

    def save_model(self, path):
        self.saved_to = path


def test_run_grpo_writes_a_run_manifest(tmp_path):
    output_dir = str(tmp_path / "out")
    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir=output_dir, save_steps=None,
    )
    records = run_grpo(spec, trainer_factory=lambda **kw: _FakeTrainer(**kw))

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["expected_checkpoints"] == checkpoint_schedule(
        spec.max_steps, spec.save_steps
    )
    assert manifest["seed"] == spec.seed
    assert manifest["n_step_records"] == len(records)


def test_run_grpo_rejects_lora_wrapped_model_from_factory():
    class PeftLikeTrainer:
        def __init__(self, **kwargs):
            self.captured = kwargs
            self.model = type("M", (), {"peft_config": {"r": 8}})()

        def train(self):
            raise AssertionError("train() should not be reached")

        def save_model(self, path):
            raise AssertionError("save_model() should not be reached")

    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir="/tmp/out", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_grpo(spec, trainer_factory=lambda **kw: PeftLikeTrainer(**kw))


def test_build_reward_fn_rejects_duplicate_prompt_text():
    dup_items = [
        TaskItem(task_id="probe/x/0", suite="s", prompt="same", answer="7", difficulty=1),
        TaskItem(task_id="probe/x/1", suite="s", prompt="same", answer="9", difficulty=1),
    ]
    with pytest.raises(ValueError, match="share prompt text"):
        build_reward_fn(dup_items, verify)


def test_reward_fn_raises_on_total_prompt_mismatch():
    fn = build_reward_fn(items(2), verify)
    with pytest.raises(ValueError, match="matched any known item"):
        fn(completions=["7", "7"], prompts=["not-a-prompt", "also-not-a-prompt"])


def test_run_manifest_records_the_pinned_batch_shape(tmp_path):
    # Spec 6.2 asks for token-based batching; sequence-based batching is pinned
    # instead, and each run must record the shape it actually used.
    from amenability.training.grpo import PROMPTS_PER_STEP

    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir=str(tmp_path / "out"), save_steps=None,
    )
    run_grpo(spec, trainer_factory=lambda **kw: _FakeTrainer(**kw))
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["prompts_per_step"] == PROMPTS_PER_STEP
    assert manifest["per_device_train_batch_size"] == spec.num_generations
    assert manifest["gradient_accumulation_steps"] == PROMPTS_PER_STEP


def test_run_grpo_leaves_no_reference_to_the_trainer(tmp_path):
    """After run_grpo returns, the trainer must be freed: its model, reference
    model and optimiser state are what hold the GPU, and vLLM initialises in the
    same process right after (#33). The cyclic collector is disabled so that a
    reference cycle keeping the trainer alive cannot be cleaned up by a lucky GC
    pass; run_grpo has to free it deterministically."""
    refs = {}

    class Trainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.reward_funcs = kwargs["reward_funcs"]
            self.model = object()
            refs["trainer"] = weakref.ref(self)

        def train(self):
            # Same shape as _FakeTrainer.train: a real step scores a group of
            # completions (prompt "p0" comes from items()) before logging.
            for cb in self.callbacks:
                self.reward_funcs[0](completions=["7", "8"], prompts=["p0", "p0"])
                cb.on_log(
                    args=None,
                    state=type("S", (), {"global_step": 1, "log_history": []})(),
                    control=None,
                    logs={"kl": 0.0, "grad_norm": 1.0},
                )

        def save_model(self, path):
            pass

    spec = GRPOSpec(
        model_path="x", model_key="k", items=items(2), verify_fn=verify, max_steps=1,
        num_generations=2, learning_rate=1e-6, beta=0.04, temperature=1.0, seed=0,
        output_dir=str(tmp_path / "out"), save_steps=None,
    )
    gc.disable()
    try:
        records = run_grpo(spec, trainer_factory=lambda **kw: Trainer(**kw))
        assert len(records) == 1
        assert refs["trainer"]() is None, "run_grpo still holds the trainer after returning"
    finally:
        gc.enable()


def test_run_grpo_empties_the_cuda_cache_when_cuda_is_available(tmp_path, monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    spec = GRPOSpec(
        model_path="x", model_key="k", items=items(2), verify_fn=verify, max_steps=1,
        num_generations=2, learning_rate=1e-6, beta=0.04, temperature=1.0, seed=0,
        output_dir=str(tmp_path / "out"), save_steps=None,
    )
    run_grpo(spec, trainer_factory=lambda **kw: _FakeTrainer(**kw))
    assert calls == ["empty"]
