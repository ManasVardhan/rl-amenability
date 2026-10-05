"""The GRPO training reward of the probe suites (transfer-rulings T23).

A partial format reward as in TinyZero's Countdown reward
(Jiayi-Pan/TinyZero, `verl/utils/reward_score/countdown.py`, `compute_score`
with `format_score=0.1, score=1.`):

| completion (truncated at the first `</answer>`, T20) | reward |
|---|---|
| the suite verifier accepts it                         | CORRECT_SCORE (1.0) |
| an `<answer>...</answer>` block, verifier rejects it  | FORMAT_SCORE (0.1) |
| no closed `<answer>...</answer>` block                | NO_ANSWER_SCORE (0.0) |

Prose, an empty block, the wrong numbers, a wrong value and a division by zero
inside the tags all score FORMAT_SCORE, as in TinyZero, where an answer that
fails validation or evaluation gets format_score.

Deliberate deviations from TinyZero, kept: the FIRST answer block counts (T20),
anywhere in the completion and across lines, where TinyZero reads the last block
on the last line of the text after "Assistant:"; the reward sees the completion
alone, so no "Assistant:" split; Countdown's verifier accepts one trailing
"= <target>" (T19) and uses its own grammar and exact arithmetic; Graph path,
which TinyZero does not have, gets the same tiering.

TRAINING reward only. pass@k, the conversion rate, pre/post evaluation and the
rejection-sampling SFT filter stay strictly binary (the suite verifier), and the
gsm8k target keeps its binary reward.
"""
from __future__ import annotations

from typing import Callable

from amenability.suites.base import TaskItem, first_answer_block

CORRECT_SCORE = 1.0
FORMAT_SCORE = 0.1
NO_ANSWER_SCORE = 0.0


def shaped_reward(
    verify: Callable[[TaskItem, str], bool], item: TaskItem, completion: str
) -> float:
    """The tiered reward of one completion, already truncated at the stop."""
    if verify(item, completion):
        return CORRECT_SCORE
    if first_answer_block(completion) is not None:
        return FORMAT_SCORE
    return NO_ANSWER_SCORE
