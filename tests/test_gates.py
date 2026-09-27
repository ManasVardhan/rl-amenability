import pytest
from amenability.scoring.gates import evaluate_gate_a, evaluate_gate_b


def test_gate_a_passes_when_both_families_order_correctly():
    # known_order: 2 = fresh base (most amenable), 0 = most over-SFT'd.
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.8, 0.5, 0.2]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is True
    # Pooled rho is 0.956, not 1.0: known_order ties across the two families.
    assert res.statistic > 0.9


def test_gate_a_fails_when_one_family_inverts():
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.1, 0.5, 0.9]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is False
    assert "llama" in res.detail


def test_gate_a_fails_when_pooled_rho_below_threshold():
    # Ordering correct in both families but noisy enough to drag pooled rho down.
    scores = {"qwen": [0.55, 0.50, 0.45], "llama": [0.10, 0.05, 0.00]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known, threshold=0.99)
    assert res.passed is False


def test_gate_a_rejects_mismatched_family_keys():
    with pytest.raises(ValueError, match="families"):
        evaluate_gate_a({"qwen": [1, 2]}, {"llama": [1, 2]})


def test_gate_a_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        evaluate_gate_a({"qwen": [1, 2, 3]}, {"qwen": [1, 2]})


def test_gate_b_passes_when_score_beats_naive():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]     # perfect
    naive = [0.9, 0.1, 0.5, 0.2, 0.7, 0.3]      # unrelated
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is True


def test_gate_b_fails_when_naive_matches_or_beats_score():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.9, 0.1, 0.5, 0.2, 0.7, 0.3]
    naive = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is False
    assert "naive" in res.detail


def test_gate_b_uses_absolute_correlation_so_sign_does_not_rescue_a_baseline():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
    naive = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]  # perfectly inverse, still informative
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is False


def test_gate_b_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        evaluate_gate_b([1, 2, 3], [1, 2], [1, 2, 3])
