"""Gate B's naive baseline input under the shaped training reward (transfer-rulings T23).

`baselines.baseline_naive` (frozen) fits the frozen `naive_extrapolation` to
each step's `mean_reward`. Under T23 the probe trains on a shaped reward, so
`mean_reward` includes the 0.1 format tier, and a weak model that learns the
answer format in the first steps would show a rising trace that says nothing
about correctness. The full-run outcome Gate B predicts is a correctness
quantity, and when Stage 0 was registered the probe's reward trace WAS its
correctness trace. So the baseline keeps its registered meaning by fitting the
same frozen function to `mean_correct`, the strict 0/1 trace. Neither frozen
file changes; only which trace is passed in.
"""
from __future__ import annotations

from amenability.probe.telemetry import ProbeTelemetry
from amenability.scoring.baselines import naive_extrapolation


def baseline_naive_correct(telemetries: list[ProbeTelemetry], target_step: int) -> list[float]:
    out = []
    for t in telemetries:
        correct = [s.mean_correct for s in t.steps]
        if any(c is None for c in correct):
            raise ValueError(
                f"{t.model_key}/{t.algorithm}: telemetry has no mean_correct (written "
                "before transfer-rulings T23); the naive baseline needs the correctness trace"
            )
        out.append(naive_extrapolation([s.step for s in t.steps], correct, target_step))
    return out
