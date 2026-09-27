from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from amenability.probe.callbacks import GroupedRewardRecorder, TelemetryCallback
from amenability.probe.entropy import EntropyProbe
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem

ENTROPY_BATCH_SIZE = 16


@dataclass(frozen=True)
class GRPOSpec:
    model_path: str
    model_key: str
    items: list[TaskItem]
    verify_fn: Callable[[TaskItem, str], bool]
    max_steps: int
    num_generations: int
    learning_rate: float
    beta: float
    temperature: float
    seed: int
    output_dir: str
    save_steps: int | None


def checkpoint_schedule(max_steps: int, save_steps: int | None) -> list[int]:
    if save_steps is None:
        return [max_steps]
    steps = list(range(save_steps, max_steps + 1, save_steps))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return steps


def build_reward_fn(items: list[TaskItem], verify_fn) -> Callable:
    by_prompt = {it.prompt: it for it in items}
    if len(by_prompt) != len(items):
        raise ValueError(
            f"{len(items) - len(by_prompt)} item(s) share prompt text with another "
            "item; reward attribution would be ambiguous"
        )

    def reward_fn(completions, **kwargs):
        prompts = kwargs["prompts"]
        if len(prompts) > 1 and not any(p in by_prompt for p in prompts):
            raise ValueError(
                f"no prompt in a batch of {len(prompts)} matched any known item; "
                "prompt text is not reaching the reward function as expected, so every "
                "reward would be 0.0 and the run would silently produce a null result"
            )
        out = []
        for prompt, completion in zip(prompts, completions):
            item = by_prompt.get(prompt)
            out.append(1.0 if item is not None and verify_fn(item, completion) else 0.0)
        return out

    return reward_fn


def run_grpo(spec: GRPOSpec, trainer_factory=None, peft_config=None) -> list[StepRecord]:
    if peft_config is not None:
        raise ValueError(
            "LoRA/PEFT is forbidden: it constrains weight movement, which is the "
            "quantity this benchmark measures."
        )

    recorder = GroupedRewardRecorder(num_generations=spec.num_generations)
    reward_fn = recorder.wrap(build_reward_fn(spec.items, spec.verify_fn))

    if trainer_factory is None:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        entropy_probe = EntropyProbe(
            tokenizer, [it.prompt for it in spec.items[:ENTROPY_BATCH_SIZE]]
        )
        config = GRPOConfig(
            output_dir=spec.output_dir,
            max_steps=spec.max_steps,
            learning_rate=spec.learning_rate,
            beta=spec.beta,
            temperature=spec.temperature,
            num_generations=spec.num_generations,
            save_steps=spec.save_steps or spec.max_steps,
            logging_steps=1,
            seed=spec.seed,
            bf16=True,
            report_to=[],
        )
        dataset = Dataset.from_dict({"prompt": [it.prompt for it in spec.items]})
        trainer_factory = lambda **kw: GRPOTrainer(  # noqa: E731
            model=spec.model_path, args=config, train_dataset=dataset, **kw
        )
    else:
        entropy_probe = _NullEntropyProbe()

    callback = TelemetryCallback(
        recorder=recorder,
        entropy_probe=entropy_probe,
        model_getter=lambda: getattr(trainer, "model", None),
    )
    trainer = trainer_factory(reward_funcs=[reward_fn], callbacks=[callback])
    model_obj = getattr(trainer, "model", None)
    if model_obj is not None and hasattr(model_obj, "peft_config"):
        raise ValueError(
            "trainer was constructed with a PEFT/LoRA model, which this benchmark "
            "forbids: LoRA constrains weight movement, the quantity under measurement."
        )
    trainer.train()
    trainer.save_model(spec.output_dir)

    Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_key": spec.model_key,
        "model_path": spec.model_path,
        "algorithm": "grpo",
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "num_generations": spec.num_generations,
        "learning_rate": spec.learning_rate,
        "beta": spec.beta,
        "temperature": spec.temperature,
        "seed": spec.seed,
        "n_items": len(spec.items),
        "expected_checkpoints": checkpoint_schedule(spec.max_steps, spec.save_steps),
        "n_step_records": len(callback.records),
    }
    (Path(spec.output_dir) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    return callback.records


class _NullEntropyProbe:
    def measure(self, model) -> float:
        return 0.0
