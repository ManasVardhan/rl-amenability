from __future__ import annotations

import re
from dataclasses import dataclass


# Transfer-rulings T20: a base model under the T19 scaffold does not stop after its
# first </answer>; it loops think/answer blocks until max_tokens. Every path that
# samples completions for a tagged suite stops at (or truncates to) the first
# occurrence of this string, keeping the string itself so the answer regex matches.
ANSWER_STOP = "</answer>"


_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def first_answer_block(completion: str) -> str | None:
    """Contents of the FIRST <answer>...</answer> block, unstripped, or None if the
    completion has no closed answer block. The first answer counts (T20): a base
    model loops think/answer blocks after its first answer."""
    m = _ANSWER_BLOCK_RE.search(completion)
    return m.group(1) if m else None


def truncate_at_stop(text: str, stop: str | None) -> str:
    """Cut text just after the first occurrence of stop (inclusive); unchanged if
    stop is None or absent."""
    if not stop:
        return text
    i = text.find(stop)
    return text if i < 0 else text[: i + len(stop)]


class SuiteOverlapError(Exception):
    """Raised when suites share task IDs, which would let a probe see target data."""


@dataclass(frozen=True)
class TaskItem:
    task_id: str
    suite: str
    prompt: str
    answer: str
    difficulty: int


class SuiteRegistry:
    def __init__(self) -> None:
        self._suites: dict[str, list[TaskItem]] = {}
        self._seen: dict[str, str] = {}  # task_id -> owning suite

    def register(self, name: str, items: list[TaskItem]) -> None:
        if name in self._suites:
            raise SuiteOverlapError(f"suite already registered: {name}")
        local: set[str] = set()
        for it in items:
            if it.task_id in local:
                raise SuiteOverlapError(f"duplicate task_id within suite {name}: {it.task_id}")
            if it.task_id in self._seen:
                raise SuiteOverlapError(
                    f"task_id {it.task_id} already owned by suite {self._seen[it.task_id]}"
                )
            local.add(it.task_id)
        for it in items:
            self._seen[it.task_id] = name
        self._suites[name] = list(items)

    def get(self, name: str) -> list[TaskItem]:
        return self._suites[name]

    def names(self) -> list[str]:
        return list(self._suites)
