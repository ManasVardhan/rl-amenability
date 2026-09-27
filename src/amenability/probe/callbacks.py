from __future__ import annotations

from typing import Callable

import numpy as np
from transformers import TrainerCallback

from amenability.probe.telemetry import StepRecord


class GroupedRewardRecorder:
    """Wraps a GRPO reward function to capture per-group reward statistics."""

    def __init__(self, num_generations: int) -> None:
        self.num_generations = num_generations
        self._rewards: list[float] = []

    def wrap(self, reward_fn: Callable) -> Callable:
        def wrapped(completions, **kwargs):
            rewards = reward_fn(completions=completions, **kwargs)
            if len(rewards) % self.num_generations != 0:
                raise ValueError(
                    f"batch of {len(rewards)} not divisible by "
                    f"num_generations={self.num_generations}"
                )
            self._rewards.extend(float(r) for r in rewards)
            return rewards

        return wrapped

    def drain(self) -> tuple[float, float, float]:
        if not self._rewards:
            return (0.0, 0.0, 0.0)
        arr = np.asarray(self._rewards, dtype=float).reshape(-1, self.num_generations)
        self._rewards = []
        stds = arr.std(axis=1)
        return (
            float(arr.mean()),
            float(stds.mean()),
            float(np.mean(stds < 1e-9)),
        )


class TelemetryCallback(TrainerCallback):
    """Emits one StepRecord per logging step."""

    def __init__(self, recorder: GroupedRewardRecorder, entropy_probe, model_getter) -> None:
        self.recorder = recorder
        self.entropy_probe = entropy_probe
        self.model_getter = model_getter
        self.records: list[StepRecord] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        mean_reward, group_std, zero_frac = self.recorder.drain()
        self.records.append(
            StepRecord(
                step=state.global_step,
                mean_reward=mean_reward,
                group_reward_std=group_std,
                zero_advantage_frac=zero_frac,
                policy_entropy=self.entropy_probe.measure(self.model_getter()),
                kl=float(logs.get("kl", 0.0)),
                grad_norm=float(logs.get("grad_norm", 0.0)),
            )
        )
        return control
