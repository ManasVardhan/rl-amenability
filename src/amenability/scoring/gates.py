"""PRE-REGISTERED. Frozen alongside score.py. Thresholds come from prereg/stage0.md."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amenability.eval.stats import spearman_with_ci

GATE_A_THRESHOLD = 0.7


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    statistic: float


def evaluate_gate_a(
    scores_by_family: dict[str, list[float]],
    known_order_by_family: dict[str, list[int]],
    threshold: float = GATE_A_THRESHOLD,
) -> GateResult:
    if set(scores_by_family) != set(known_order_by_family):
        raise ValueError("families in scores and known orders do not match")

    failures: list[str] = []
    pooled_scores: list[float] = []
    pooled_known: list[float] = []
    for family in sorted(scores_by_family):
        scores = scores_by_family[family]
        known = known_order_by_family[family]
        if len(scores) != len(known):
            raise ValueError(f"length mismatch for family {family}")
        ranked = [s for _, s in sorted(zip(known, scores))]
        if ranked != sorted(ranked):
            failures.append(family)
        # Pool WITHIN-FAMILY MEAN-CENTRED scores, not raw scores and not ranks.
        # Scores are z-scored across the whole roster, so two families can separate
        # in absolute level; centring removes the family-level offset that would
        # otherwise let family membership dominate the pooled statistic, while
        # preserving the within-family spacing that makes the pooled rho
        # informative. Ranking would remove the offset too, but it would force rho
        # to exactly 1.0 for any correct ordering and make this condition vacuous.
        v = np.asarray(scores, dtype=float)
        pooled_scores.extend(float(x) for x in (v - v.mean()))
        pooled_known.extend(known)

    rho = spearman_with_ci(pooled_known, pooled_scores, n_boot=2000).rho
    passed = not failures and rho >= threshold
    if failures:
        detail = f"ordering inverted in families: {', '.join(failures)}; pooled rho={rho:.3f}"
    elif not passed:
        detail = f"ordering correct in all families but pooled rho={rho:.3f} < {threshold}"
    else:
        detail = f"ordering correct in all families; pooled rho={rho:.3f}"
    return GateResult(name="A", passed=passed, detail=detail, statistic=rho)


def evaluate_gate_b(
    scores: list[float], naive: list[float], outcomes: list[float]
) -> GateResult:
    if not (len(scores) == len(naive) == len(outcomes)):
        raise ValueError("length mismatch between scores, naive baseline and outcomes")
    rho_score = abs(spearman_with_ci(scores, outcomes, n_boot=2000).rho)
    rho_naive = abs(spearman_with_ci(naive, outcomes, n_boot=2000).rho)
    passed = rho_score > rho_naive
    detail = (
        f"|rho(score)|={rho_score:.3f} vs naive |rho|={rho_naive:.3f}; "
        f"{'score wins' if passed else 'naive extrapolation matches or wins'}"
    )
    return GateResult(name="B", passed=passed, detail=detail, statistic=rho_score - rho_naive)
