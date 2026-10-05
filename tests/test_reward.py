"""The shaped GRPO training reward for the probe suites (transfer-rulings T23)."""
import pytest

from amenability.suites.base import TaskItem
from amenability.suites.catalog import get_probe_suite
from amenability.suites.countdown import verify_countdown
from amenability.suites.graphpath import verify_graphpath
from amenability.training.reward import (
    CORRECT_SCORE, FORMAT_SCORE, NO_ANSWER_SCORE, shaped_reward,
)


def _cd(numbers="3,5,2", target=13):
    return TaskItem(task_id="probe/countdown/t", suite="probe_countdown", prompt="p",
                    answer=f"{numbers}|{target}", difficulty=3)


def _gp():
    # A path graph A-B-C-D, source A, target D.
    return TaskItem(task_id="probe/graphpath/t", suite="probe_graphpath", prompt="p",
                    answer="A-B,B-C,C-D|A|D", difficulty=8)


def test_constants_match_tinyzero():
    assert (CORRECT_SCORE, FORMAT_SCORE, NO_ANSWER_SCORE) == (1.0, 0.1, 0.0)


@pytest.mark.parametrize("completion,expected", [
    ("I think the answer is 13", 0.0),                              # no tag
    ("<answer>3*5-2", 0.0),                                          # cut at the cap
    ("answer>\n13\n</answer>", 0.0),                                 # missing "<": no block
    ("ANSWER: (3 * 5) - 2 = 13", 0.0),
    ("<answer>3*5-2</answer>", 1.0),
    ("<answer> (3 * 5) - 2 = 13 </answer>", 1.0),                    # T19 tolerance
    ("<answer>\n3*5-2\n</answer>", 1.0),                             # multi-line block
    ("<answer>The equation 3 * 5 - 2 = 13</answer>", 0.1),           # prose in the tag
    ("<answer></answer>", 0.1),
    ("<answer>3+5+2</answer>", 0.1),                                 # wrong value
    ("<answer>3*5</answer>", 0.1),                                   # wrong numbers
    ("<answer>3 + 5 = 8</answer>", 0.1),                             # wrong right-hand side
    ("<answer>3 / (5 - 5) + 2</answer>", 0.1),                       # divides by zero
    ("<answer>3*5-2</answer> then <answer>prose</answer>", 1.0),     # first block counts
    ("<answer>prose</answer> then <answer>3*5-2</answer>", 0.1),
])
def test_countdown_tiers(completion, expected):
    assert shaped_reward(verify_countdown, _cd(), completion) == expected


@pytest.mark.parametrize("completion,expected", [
    ("A -> B -> C -> D", 0.0),
    ("<answer>A -> B -> C -> D</answer>", 1.0),
    ("<answer>A -> C -> D</answer>", 0.1),                           # non-edge
    ("<answer>the path is A, B, C, D</answer>", 0.1),                # prose
    ("<answer>A</answer>", 0.1),
])
def test_graphpath_tiers(completion, expected):
    assert shaped_reward(verify_graphpath, _gp(), completion) == expected


@pytest.mark.parametrize("key,verify", [("countdown", verify_countdown),
                                        ("graphpath", verify_graphpath)])
def test_every_probe_suite_trains_on_the_shaped_reward(key, verify):
    suite = get_probe_suite(key)
    item = _cd() if key == "countdown" else _gp()
    for completion in ("no tag", "<answer>junk</answer>",
                       "<answer>3*5-2</answer>", "<answer>A -> B -> C -> D</answer>"):
        assert suite.train_reward(item, completion) == shaped_reward(verify, item, completion)


def test_the_reward_is_one_exactly_when_the_verifier_accepts():
    item = _cd()
    for completion in ("<answer>3*5-2</answer>", "<answer>3+5+2</answer>", "x"):
        assert (shaped_reward(verify_countdown, item, completion) == CORRECT_SCORE) == \
            verify_countdown(item, completion)
