from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amenability.suites.base import TaskItem


def build_rejection_dataset(
    items: list[TaskItem],
    completions_by_task: dict[str, list[str]],
    verify_fn: Callable[[TaskItem, str], bool],
    max_per_prompt: int = 4,
) -> list[dict]:
    records: list[dict] = []
    for it in items:
        seen: set[str] = set()
        for completion in completions_by_task.get(it.task_id, []):
            if len(seen) >= max_per_prompt:
                break
            norm = completion.strip()
            if norm in seen or not verify_fn(it, completion):
                continue
            seen.add(norm)
            records.append({"prompt": it.prompt, "completion": norm})
    if not records:
        raise ValueError("no verified completions survived rejection sampling")
    return records


@dataclass(frozen=True)
class SFTSpec:
    model_path: str
    model_key: str
    records: list[dict]
    max_steps: int
    learning_rate: float
    seed: int
    output_dir: str
    save_steps: int | None


def run_sft(spec: SFTSpec, trainer_factory=None, peft_config=None) -> None:
    if peft_config is not None:
        raise ValueError(
            "LoRA/PEFT is forbidden: it constrains weight movement, which is the "
            "quantity this benchmark measures."
        )
    if trainer_factory is None:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        config = SFTConfig(
            output_dir=spec.output_dir,
            max_steps=spec.max_steps,
            learning_rate=spec.learning_rate,
            save_steps=spec.save_steps or spec.max_steps,
            logging_steps=1,
            seed=spec.seed,
            bf16=True,
            report_to=[],
        )
        dataset = Dataset.from_list(spec.records)
        trainer_factory = lambda **kw: SFTTrainer(  # noqa: E731
            model=spec.model_path, args=config, train_dataset=dataset, **kw
        )
    trainer = trainer_factory()
    trainer.train()
    trainer.save_model(spec.output_dir)
