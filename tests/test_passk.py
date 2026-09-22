import math
import pytest
from amenability.suites.base import TaskItem
from amenability.eval.passk import pass_at_k, evaluate_passk


def test_pass_at_k_all_correct_is_one():
    assert pass_at_k(n=10, c=10, k=1) == 1.0


def test_pass_at_k_none_correct_is_zero():
    assert pass_at_k(n=10, c=0, k=5) == 0.0


def test_pass_at_1_equals_empirical_rate():
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)


def test_pass_at_k_matches_closed_form():
    # 1 - C(n-c, k)/C(n, k) for n=10, c=2, k=3
    expected = 1 - (math.comb(8, 3) / math.comb(10, 3))
    assert pass_at_k(n=10, c=2, k=3) == pytest.approx(expected)


def test_pass_at_k_is_monotonic_in_k():
    vals = [pass_at_k(n=32, c=4, k=k) for k in (1, 2, 4, 8, 16, 32)]
    assert vals == sorted(vals)


def test_pass_at_k_requires_k_le_n():
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=2, k=8)


def test_evaluate_passk_with_injected_generator():
    items = [
        TaskItem(task_id=f"t/{i}", suite="s", prompt="p", answer="1", difficulty=0)
        for i in range(2)
    ]

    # item 0: 2 of 4 correct. item 1: 0 of 4 correct.
    canned = {"t/0": ["1", "1", "0", "0"], "t/1": ["0", "0", "0", "0"]}

    def fake_generate(model_path, prompts, n, temperature, seed):
        return [canned[tid] for tid in prompts.keys()]

    def verify(item, completion):
        return completion == item.answer

    res = evaluate_passk(
        model_path="unused", items=items, verify_fn=verify,
        ks=(1, 2), n_samples=4, temperature=1.0, seed=0, generate_fn=fake_generate,
    )
    assert res.n_items == 2
    assert res.per_item_correct == {"t/0": 2, "t/1": 0}
    # pass@1 averaged over items: (0.5 + 0.0) / 2
    assert res.ks[1] == pytest.approx(0.25)
    # pass@2 for item 0: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert res.ks[2] == pytest.approx(((1 - 1 / 6) + 0.0) / 2)
