from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from amenability.suites.base import TaskItem


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021), computed stably in log space."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n - c < k:
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@dataclass(frozen=True)
class PassKResult:
    ks: dict[int, float]
    n_samples: int
    n_items: int
    per_item_correct: dict[str, int]


def _vllm_generate(model_path: str, prompts: dict[str, str], n: int,
                   temperature: float, seed: int) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, seed=seed, dtype="bfloat16", gpu_memory_utilization=0.85)
    params = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=768, seed=seed)
    outputs = llm.generate(list(prompts.values()), params)
    return [[o.text for o in out.outputs] for out in outputs]


def evaluate_passk(
    model_path: str,
    items: list[TaskItem],
    verify_fn: Callable[[TaskItem, str], bool],
    ks: tuple[int, ...],
    n_samples: int,
    temperature: float,
    seed: int,
    generate_fn: Callable[..., list[list[str]]] | None = None,
) -> PassKResult:
    generate = generate_fn or _vllm_generate
    prompts = {it.task_id: it.prompt for it in items}
    completions = generate(model_path, prompts, n_samples, temperature, seed)

    per_item_correct: dict[str, int] = {}
    for it, comps in zip(items, completions):
        per_item_correct[it.task_id] = sum(1 for c in comps if verify_fn(it, c))

    ks_out: dict[int, float] = {}
    for k in ks:
        ks_out[k] = float(
            np.mean([pass_at_k(n_samples, c, k) for c in per_item_correct.values()])
        )
    return PassKResult(
        ks=ks_out, n_samples=n_samples, n_items=len(items), per_item_correct=per_item_correct
    )
