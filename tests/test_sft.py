import pytest
from amenability.suites.base import TaskItem
from amenability.training.sft import build_rejection_dataset, SFTSpec, run_sft


def items(n=2):
    return [
        TaskItem(task_id=f"t/{i}", suite="s", prompt=f"p{i}", answer="7", difficulty=1)
        for i in range(n)
    ]


def verify(item, completion):
    return "7" in completion


def test_keeps_only_verified_completions():
    comps = {"t/0": ["the answer is 7", "the answer is 3"], "t/1": ["nope"]}
    recs = build_rejection_dataset(items(), comps, verify)
    assert len(recs) == 1
    assert recs[0] == {"prompt": "p0", "completion": "the answer is 7"}


def test_caps_completions_per_prompt():
    comps = {"t/0": [f"7 v{i}" for i in range(10)], "t/1": []}
    recs = build_rejection_dataset(items(), comps, verify, max_per_prompt=3)
    assert len(recs) == 3


def test_deduplicates_identical_completions():
    comps = {"t/0": ["7", "7", "7"], "t/1": []}
    recs = build_rejection_dataset(items(), comps, verify, max_per_prompt=5)
    assert len(recs) == 1


def test_missing_task_in_completions_is_skipped_not_fatal():
    comps = {"t/0": ["7"]}
    recs = build_rejection_dataset(items(), comps, verify)
    assert len(recs) == 1


def test_empty_dataset_raises_rather_than_training_on_nothing():
    comps = {"t/0": ["wrong"], "t/1": ["wrong"]}
    with pytest.raises(ValueError, match="no verified completions"):
        build_rejection_dataset(items(), comps, verify)


def test_run_sft_calls_trainer_and_saves():
    captured = {}

    class FakeTrainer:
        def __init__(self, **kw):
            captured.update(kw)

        def train(self):
            captured["trained"] = True

        def save_model(self, path):
            captured["saved_to"] = path

    spec = SFTSpec(
        model_path="fake", model_key="m",
        records=[{"prompt": "p", "completion": "c"}],
        max_steps=10, learning_rate=1e-5, seed=0,
        output_dir="/tmp/sft", save_steps=None,
    )
    run_sft(spec, trainer_factory=lambda **kw: FakeTrainer(**kw))
    assert captured["trained"] is True
    assert captured["saved_to"] == "/tmp/sft"


def test_run_sft_rejects_lora_config():
    spec = SFTSpec(
        model_path="fake", model_key="m", records=[{"prompt": "p", "completion": "c"}],
        max_steps=10, learning_rate=1e-5, seed=0, output_dir="/tmp/sft", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_sft(spec, trainer_factory=lambda **kw: None, peft_config={"r": 8})


def test_run_sft_rejects_lora_wrapped_model_from_factory():
    class PeftLikeTrainer:
        def __init__(self, **kwargs):
            self.model = type("M", (), {"peft_config": {"r": 8}})()

        def train(self):
            raise AssertionError("train() should not be reached")

        def save_model(self, path):
            raise AssertionError("save_model() should not be reached")

    spec = SFTSpec(
        model_path="fake", model_key="m", records=[{"prompt": "p", "completion": "c"}],
        max_steps=10, learning_rate=1e-5, seed=0, output_dir="/tmp/sft", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_sft(spec, trainer_factory=lambda **kw: PeftLikeTrainer(**kw))
