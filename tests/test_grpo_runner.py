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
            # Simulate three logging steps.
            for cb in captured["callbacks"]:
                for step in (1, 2, 3):
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
