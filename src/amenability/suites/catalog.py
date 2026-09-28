"""Name-to-suite lookup for probe suites.

Runners and analysis code take a suite KEY and look it up here, so no script
hard-codes Countdown. Target suites are deliberately not listed: the probe must
never be able to reach target data by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amenability.suites.base import TaskItem
from amenability.suites.countdown import generate_countdown, verify_countdown
from amenability.suites.graphpath import SUITE_NAME as GRAPHPATH_NAME
from amenability.suites.graphpath import generate_graphpath, verify_graphpath


@dataclass(frozen=True)
class ProbeSuite:
    key: str
    name: str
    generate: Callable[[int, int], list[TaskItem]]
    verify: Callable[[TaskItem, str], bool]


PROBE_SUITES: dict[str, ProbeSuite] = {
    "countdown": ProbeSuite(
        key="countdown", name="probe_countdown",
        generate=generate_countdown, verify=verify_countdown,
    ),
    "graphpath": ProbeSuite(
        key="graphpath", name=GRAPHPATH_NAME,
        generate=generate_graphpath, verify=verify_graphpath,
    ),
}


def get_probe_suite(key: str) -> ProbeSuite:
    try:
        return PROBE_SUITES[key]
    except KeyError:
        raise KeyError(
            f"unknown probe suite {key!r}; valid keys: {', '.join(sorted(PROBE_SUITES))}"
        ) from None
