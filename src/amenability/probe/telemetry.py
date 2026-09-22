# src/amenability/probe/telemetry.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from amenability.eval.passk import PassKResult

BREADTH_K = 32
COLLAPSE_FRACTION = 0.5


@dataclass(frozen=True)
class StepRecord:
    step: int
    mean_reward: float
    group_reward_std: float
    zero_advantage_frac: float
    policy_entropy: float
    kl: float
    grad_norm: float


@dataclass
class ProbeTelemetry:
    model_key: str
    algorithm: str
    steps: list[StepRecord]
    pre: PassKResult
    post: PassKResult

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_key": self.model_key,
                "algorithm": self.algorithm,
                "steps": [asdict(s) for s in self.steps],
                "pre": asdict(self.pre),
                "post": asdict(self.post),
            }
        )

    @staticmethod
    def from_json(d: dict) -> "ProbeTelemetry":
        def pk(x: dict) -> PassKResult:
            return PassKResult(
                ks={int(k): v for k, v in x["ks"].items()},
                n_samples=x["n_samples"],
                n_items=x["n_items"],
                per_item_correct=x["per_item_correct"],
            )

        return ProbeTelemetry(
            model_key=d["model_key"],
            algorithm=d["algorithm"],
            steps=[StepRecord(**s) for s in d["steps"]],
            pre=pk(d["pre"]),
            post=pk(d["post"]),
        )


@dataclass(frozen=True)
class FeatureVector:
    conversion_rate: float
    retention_factor: float
    zero_advantage_rate: float
    reward_slope_early: float
    reward_slope_mid: float
    grad_norm_trend: float
    entropy_slope: float
    entropy_collapse_step: int | None


def _slope(ys: list[float]) -> float:
    if len(ys) < 2:
        return 0.0
    xs = np.arange(len(ys), dtype=float)
    return float(np.polyfit(xs, np.asarray(ys, dtype=float), 1)[0])


def extract_features(t: ProbeTelemetry) -> FeatureVector:
    entropies = [s.policy_entropy for s in t.steps]
    rewards = [s.mean_reward for s in t.steps]
    kls = [s.kl for s in t.steps]
    grads = [s.grad_norm for s in t.steps]
    half = max(1, len(t.steps) // 2)

    breadth = t.pre.ks[BREADTH_K] - t.pre.ks[1]
    gain = t.post.ks[1] - t.pre.ks[1]
    conversion_rate = float(gain / breadth) if breadth > 1e-9 else 0.0

    entropy_slope = _slope(entropies)
    collapse_step = None
    if entropies:
        threshold = entropies[0] * COLLAPSE_FRACTION
        for s in t.steps:
            if s.policy_entropy < threshold:
                collapse_step = s.step
                break

    reward_gain = max(rewards[-1] - rewards[0], 1e-6) if rewards else 1e-6
    kl_per_gain = float(np.sum(kls) / reward_gain)

    # Retention is high when entropy holds and reward is bought cheaply in KL.
    entropy_term = 1.0 / (1.0 + max(0.0, -entropy_slope))
    kl_term = 1.0 / (1.0 + kl_per_gain)
    retention_factor = float(entropy_term * kl_term)

    return FeatureVector(
        conversion_rate=conversion_rate,
        retention_factor=retention_factor,
        zero_advantage_rate=float(np.mean([s.zero_advantage_frac for s in t.steps])),
        reward_slope_early=_slope(rewards[:half]),
        reward_slope_mid=_slope(rewards[half:]),
        grad_norm_trend=_slope(grads),
        entropy_slope=entropy_slope,
        entropy_collapse_step=collapse_step,
    )
