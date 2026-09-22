import pytest
from amenability.suites.base import TaskItem
from amenability.suites.gsm8k import extract_final_number, verify_gsm8k


def _item(answer: str) -> TaskItem:
    return TaskItem(
        task_id="target/gsm8k/test/0", suite="target_gsm8k",
        prompt="p", answer=answer, difficulty=0,
    )


def test_extract_plain_integer():
    assert extract_final_number("The answer is 42") == "42"


def test_extract_takes_last_number():
    assert extract_final_number("first 7 then 13") == "13"


def test_extract_strips_commas_and_currency():
    assert extract_final_number("It costs $1,234") == "1234"


def test_extract_handles_negative():
    assert extract_final_number("balance is -18") == "-18"


def test_extract_handles_trailing_period():
    assert extract_final_number("The total is 72.") == "72"


def test_extract_preserves_true_decimal():
    assert extract_final_number("the rate is 3.5") == "3.5"


def test_extract_returns_none_when_no_number():
    assert extract_final_number("no digits at all") is None


def test_verify_matches_numerically_not_textually():
    assert verify_gsm8k(_item("72"), "so the answer is 72.0") is True
    assert verify_gsm8k(_item("72"), "so the answer is 72") is True


def test_verify_rejects_wrong_answer():
    assert verify_gsm8k(_item("72"), "the answer is 71") is False


def test_verify_rejects_empty_completion():
    assert verify_gsm8k(_item("72"), "") is False
