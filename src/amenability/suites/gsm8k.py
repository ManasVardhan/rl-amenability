from __future__ import annotations

import random
import re

from amenability.suites.base import TaskItem

PROMPT = "{question}\n\nSolve step by step, then give the final numeric answer on its own line."

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith("."):
        raw = raw[:-1]
    return raw


def verify_gsm8k(item: TaskItem, completion: str) -> bool:
    got = extract_final_number(completion)
    if got is None:
        return False
    try:
        return abs(float(got) - float(item.answer)) < 1e-6
    except ValueError:
        return False


def load_gsm8k(split: str, limit: int | None = None, seed: int = 0) -> list[TaskItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    if limit is not None:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    items: list[TaskItem] = []
    for i in idx:
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(
            TaskItem(
                task_id=f"target/gsm8k/{split}/{i}",
                suite="target_gsm8k",
                prompt=PROMPT.format(question=row["question"]),
                answer=gold,
                difficulty=0,
            )
        )
    return items
