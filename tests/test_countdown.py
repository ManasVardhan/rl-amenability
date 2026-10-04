import pytest
from amenability.suites.countdown import (
    generate_countdown,
    solve_countdown,
    verify_countdown,
    extract_expression,
)
from amenability.suites.base import TaskItem


def test_every_generated_item_is_solvable():
    items = generate_countdown(n=30, seed=0)
    for it in items:
        numbers = [int(x) for x in it.answer.split("|")[0].split(",")]
        target = int(it.answer.split("|")[1])
        assert solve_countdown(numbers, target) is not None, f"unsolvable item {it.task_id}"


def test_generation_is_deterministic_by_seed():
    a = generate_countdown(n=10, seed=7)
    b = generate_countdown(n=10, seed=7)
    c = generate_countdown(n=10, seed=8)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.prompt for x in a] != [x.prompt for x in c]


def test_difficulty_buckets_are_populated():
    items = generate_countdown(n=30, seed=1, buckets=(3, 4, 5))
    assert {it.difficulty for it in items} == {3, 4, 5}


def test_task_ids_are_namespaced_to_probe():
    items = generate_countdown(n=5, seed=2)
    assert all(it.task_id.startswith("probe/countdown/") for it in items)


def test_extract_expression_takes_first_answer_tag():
    # First, not last (transfer-rulings T20): later tags are the base model's loop.
    assert extract_expression("junk <answer>1+1</answer> more <answer>2*3</answer>") == "1+1"
    assert extract_expression("no tags here") is None


def _item(numbers: str, target: int) -> TaskItem:
    return TaskItem(
        task_id="probe/countdown/t",
        suite="probe_countdown",
        prompt="p",
        answer=f"{numbers}|{target}",
        difficulty=3,
    )


def test_verifier_accepts_correct_expression():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>3*5+7</answer>") is True


def test_verifier_rejects_wrong_value():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>3+5+7</answer>") is False


def test_verifier_rejects_unavailable_numbers():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>11*2</answer>") is False


def test_verifier_rejects_reused_numbers():
    it = _item("3,5,7", 21)
    assert verify_countdown(it, "<answer>3*7*1</answer>") is False
    assert verify_countdown(it, "<answer>3*3+12</answer>") is False


def test_verifier_rejects_malicious_input():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>__import__('os').system('ls')</answer>") is False
