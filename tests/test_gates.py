import pytest
from amenability.scoring.gates import evaluate_gate_a, evaluate_gate_b


def test_gate_a_passes_when_both_families_order_correctly():
    # known_order: 2 = fresh base (most amenable), 0 = most over-SFT'd.
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.8, 0.5, 0.2]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is True
    # Pooling is on within-family MEAN-CENTRED scores, which removes the family
    # offset while preserving within-family spacing. A correct ordering therefore
    # gives a high but not automatically perfect rho: unlike rank pooling, the
    # value still depends on how the two families' spacings interleave.
    assert res.statistic == pytest.approx(0.956, abs=5e-4)


def test_gate_a_fails_when_one_family_inverts():
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.1, 0.5, 0.9]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is False
    assert "llama" in res.detail


def test_gate_a_fails_when_one_family_ordering_is_partially_wrong():
    # qwen is correctly ordered; llama has its two most-amenable checkpoints swapped.
    # The ordering check catches llama, and the pooled statistic reflects the damage.
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.5, 0.9, 0.1]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is False
    assert "llama" in res.detail


def test_gate_a_passes_when_families_separate_in_score_level():
    # Regression test for the raw-pooling defect: both families are correctly
    # ordered but occupy disjoint score ranges. Under raw pooling this gave
    # rho=0.478 and failed the 0.7 threshold despite a perfectly correct result.
    scores = {"qwen": [1.4, 1.1, 0.8], "llama": [-0.8, -1.1, -1.4]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is True
    assert res.statistic == pytest.approx(0.956, abs=5e-4)


def test_gate_a_can_fail_on_rho_alone_despite_correct_ordering_in_both_families():
    # Under rank pooling this state was unreachable: any correct ordering gave
    # rho exactly 1.0, so the rho condition could never bind and the conjunction
    # collapsed to the ordering check. With centred pooling, one family's scores
    # being compressed relative to the other lowers the pooled rho to 0.837, so
    # a 0.9 threshold fails the gate even though every family is ordered right.
    # This test documents that the rho condition does real work.
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.52, 0.50, 0.48]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known, threshold=0.9)
    assert res.statistic == pytest.approx(0.837, abs=5e-4)
    assert res.passed is False
    assert "ordering correct in all families" in res.detail


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
