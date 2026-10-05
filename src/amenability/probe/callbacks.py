from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from transformers import TrainerCallback

from amenability.probe.telemetry import StepRecord


class DrainedSignal(NamedTuple):
    """One step's group statistics. The first three are over the training reward
    (what GRPO's advantage is computed from); the last two over strict 0/1
    correctness, None when nothing recorded correctness (transfer-rulings T23)."""
    mean_reward: float
    group_reward_std: float
    zero_advantage_frac: float
    mean_correct: float | None
    zero_correct_group_frac: float | None


def _group_stats(values: list[float], num_generations: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1, num_generations)
    stds = arr.std(axis=1)
    return float(arr.mean()), float(stds.mean()), float(np.mean(stds < 1e-9))


class GroupedRewardRecorder:
    """Wraps a GRPO reward function to capture per-group reward statistics, and
    takes the matching strict correctness through record_correct."""

    def __init__(self, num_generations: int) -> None:
        self.num_generations = num_generations
        self._rewards: list[float] = []
        self._correct: list[float] = []

    def record_correct(self, values) -> None:
        """Strict 0/1 correctness of the completions the wrapped reward function is
        scoring, in the same order as its rewards."""
        self._correct.extend(float(v) for v in values)

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

    def pending(self) -> bool:
        """True when rewards have been accumulated since the last drain.

        Callers need this to distinguish "no reward pass happened" from "a reward
        pass happened and every reward was 0.0". Both make drain() return zeros,
        but only the first means there was no training step to record.
        """
        return bool(self._rewards)

    def drain(self) -> DrainedSignal:
        rewards, correct = self._rewards, self._correct
        self._rewards, self._correct = [], []
        if not rewards:
            return DrainedSignal(0.0, 0.0, 0.0, None, None)
        if correct and len(correct) != len(rewards):
            raise ValueError(
                f"{len(correct)} correctness values for {len(rewards)} rewards; "
                "the correctness signal would be misattributed"
            )
        mean_r, std_r, zero_r = _group_stats(rewards, self.num_generations)
        if not correct:
            return DrainedSignal(mean_r, std_r, zero_r, None, None)
        mean_c, _, zero_c = _group_stats(correct, self.num_generations)
        return DrainedSignal(mean_r, std_r, zero_r, mean_c, zero_c)


class TelemetryCallback(TrainerCallback):
    """Emits one StepRecord per logging step."""

    def __init__(self, recorder: GroupedRewardRecorder, entropy_probe, model_getter) -> None:
        self.recorder = recorder
        self.entropy_probe = entropy_probe
        self.model_getter = model_getter
        self.records: list[StepRecord] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # DO NOT REMOVE. transformers' Trainer._finalize_training calls
        # self.log(metrics) AFTER training has ended, which fires on_log one
        # extra time with no preceding reward pass. Without this check that
        # phantom log appends a StepRecord with mean_reward=0.0, which silently
        # drives reward_gain negative (so retention_factor collapses for every
        # variant), makes baseline_naive fit a spurious zero point that biases
        # the Gate B baseline down, and dilutes zero_advantage_rate.
        #
        # The test is whether the recorder actually has rewards buffered, NOT
        # whether the drained mean is 0.0: a legitimate training step can score
        # every completion 0.0, and that step must still be recorded.
        if not self.recorder.pending():
            return control
        d = self.recorder.drain()
        self.records.append(
            StepRecord(
                step=state.global_step,
                mean_reward=d.mean_reward,
                group_reward_std=d.group_reward_std,
                zero_advantage_frac=d.zero_advantage_frac,
                policy_entropy=self.entropy_probe.measure(self.model_getter()),
                kl=float(logs.get("kl", 0.0)),
                grad_norm=float(logs.get("grad_norm", 0.0)),
                mean_correct=d.mean_correct,
                zero_correct_group_frac=d.zero_correct_group_frac,
            )
        )
        return control
