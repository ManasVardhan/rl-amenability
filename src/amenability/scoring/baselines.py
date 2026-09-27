"""PRE-REGISTERED. Frozen alongside score.py."""
from __future__ import annotations

import numpy as np

from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import ProbeTelemetry
from amenability.registry.loader import ModelSpec

ALL_BASELINES = {
    "naive_extrapolation": "Log-curve fit to the probe reward trace, read at the full-run step.",
    "pass_at_64": "Base model pass@64, the literature's current best cheap metric.",
    "pass_at_1": "Base model pass@1.",
    "param_count": "Model parameter count in billions.",
}


def naive_extrapolation(steps: list[int], rewards: list[float], target_step: int) -> float:
    """Fit y = a + b*log(1+x) to the probe trace and evaluate it at target_step."""
    if len(steps) < 2:
        raise ValueError("naive extrapolation needs at least two points")
    x = np.log1p(np.asarray(steps, dtype=float))
    y = np.asarray(rewards, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return float(a + b * np.log1p(target_step))


def baseline_naive(telemetries: list[ProbeTelemetry], target_step: int) -> list[float]:
    return [
        naive_extrapolation(
            [s.step for s in t.steps], [s.mean_reward for s in t.steps], target_step
        )
        for t in telemetries
    ]


def baseline_pass_at_64(pre_results: list[PassKResult]) -> list[float]:
    return [r.ks[64] for r in pre_results]


def baseline_pass_at_1(pre_results: list[PassKResult]) -> list[float]:
    return [r.ks[1] for r in pre_results]


def baseline_param_count(specs: list[ModelSpec]) -> list[float]:
    return [s.params_b for s in specs]
