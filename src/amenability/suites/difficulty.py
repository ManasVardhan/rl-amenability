"""Named difficulty candidates for the probe training pools (transfer-rulings T22).

The GPU smoke probes showed that GRPO with group size 8 gets no gradient on a
group with no correct completion, and that about half the roster almost never
solves a pilot item within 8 tries. The difficulty calibration
(`scripts/calibrate_difficulty.py`) samples every candidate below with the GRPO
rollout sampling, and the selection rule pre-registered in T22 picks one
candidate per suite mechanically. Nothing here changes the default items: the
`*-current` candidates are the pilot's items, byte for byte.

Within a suite, candidates are listed from hardest to easiest; this order is
part of the pre-registration ("the candidate closest to current"). Countdown's
order follows the pilot, where the number count drives difficulty (pass@32 falls
by a factor of 3 to 8 from 3 to 4 numbers), then the size of the numbers and
targets. Graph path's order follows node count, which drove difficulty
monotonically in every pilot model. A graph candidate is eligible only if the
T1 informed (edge-blind) guesser stays inside the T1 bound on its 300-item pool
(pass@32 < 0.3 in every bucket and < 0.15 on average); an ineligible candidate
is still sampled for information but can never be selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from amenability.suites.base import TaskItem
from amenability.suites.countdown import generate_countdown
from amenability.suites.graphpath import generate_graphpath, informed_guess_pass_at_k

GUESSER_BOUND_BUCKET = 0.3
GUESSER_BOUND_MEAN = 0.15
# The T1 measurement: 300 items from item seed 0, k = 32, guesser seed 0.
GUESSER_N_ITEMS = 300
GUESSER_ITEM_SEED = 0
GUESSER_K = 32
GUESSER_SEED = 0

_GENERATORS = {"countdown": generate_countdown, "graphpath": generate_graphpath}


@dataclass(frozen=True)
class DifficultyCandidate:
    name: str
    suite_key: str
    params_items: tuple[tuple[str, Any], ...]
    description: str
    # The current candidate keeps the default task IDs, so its items are the
    # batch's items exactly; every other candidate namespaces its IDs by name.
    is_current: bool = False

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.params_items)

    def generate(self, n: int, seed: int) -> list[TaskItem]:
        namespace = None if self.is_current else self.name
        return _GENERATORS[self.suite_key](n, seed, **self.params, id_namespace=namespace)


def _cd(name, buckets, number_range, target_range, description, current=False):
    return DifficultyCandidate(
        name=name, suite_key="countdown",
        params_items=(("buckets", buckets), ("number_range", number_range),
                      ("target_range", target_range)),
        description=description, is_current=current,
    )


def _gp(name, buckets, min_distance, extra_edges_divisor, description, current=False):
    return DifficultyCandidate(
        name=name, suite_key="graphpath",
        params_items=(("buckets", buckets), ("min_distance", min_distance),
                      ("extra_edges_divisor", extra_edges_divisor)),
        description=description, is_current=current,
    )


# Hardest to easiest within each suite. This order is pre-registered (T22).
_ORDERED = (
    _cd("cd-current", (3, 4, 5), (1, 20), (10, 400),
        "pilot items: 3/4/5 numbers from 1..20, targets 10..400", current=True),
    _cd("cd-easy", (2, 3, 4), (1, 20), (10, 400),
        "one number fewer per bucket, same ranges"),
    _cd("cd-mid", (2, 3, 4), (1, 15), (5, 200),
        "between cd-easy and cd-easier, a step for the headroom condition"),
    _cd("cd-easier", (2, 3, 4), (1, 10), (5, 100),
        "one number fewer, numbers 1..10, targets 5..100"),
    _gp("gp-current", (8, 10, 12), 3, 3,
        "pilot items: 8/10/12 nodes, n // 3 extra edges, distance >= 3", current=True),
    _gp("gp-contingency", (7, 9, 11), 3, 3,
        "the T1 bound contingency; the 7-node bucket guesses at 0.37"),
    _gp("gp-sparse-7", (7, 9, 11), 3, 6,
        "contingency sizes with n // 6 extra edges (one each), fewer guessable paths"),
    _gp("gp-tree-6", (6, 8, 10), 4, None,
        "two nodes fewer per bucket, trees (one path), distance >= 4"),
    _gp("gp-tree-6s", (6, 7, 8), 4, None,
        "smallest graphs: 6/7/8-node trees, distance >= 4"),
)

CANDIDATES: dict[str, DifficultyCandidate] = {c.name: c for c in _ORDERED}


def candidates_for(suite_key: str) -> list[DifficultyCandidate]:
    """The suite's candidates, hardest first (the pre-registered order)."""
    return [c for c in _ORDERED if c.suite_key == suite_key]


def get_candidate(name: str) -> DifficultyCandidate:
    try:
        return CANDIDATES[name]
    except KeyError:
        raise KeyError(
            f"unknown difficulty candidate {name!r}; valid: {', '.join(CANDIDATES)}"
        ) from None


def guesser_eligibility(candidate: DifficultyCandidate) -> dict:
    """The T1 informed-guesser pass@32 per bucket on the candidate's 300-item pool,
    its mean, and whether the candidate is inside the T1 bound. Countdown has no
    guesser bound, so its candidates are always eligible."""
    if candidate.suite_key != "graphpath":
        return {"rates": None, "mean": None, "eligible": True}
    items = candidate.generate(GUESSER_N_ITEMS, GUESSER_ITEM_SEED)
    rates = informed_guess_pass_at_k(items, k=GUESSER_K, seed=GUESSER_SEED)
    mean = sum(rates.values()) / len(rates)
    eligible = max(rates.values()) < GUESSER_BOUND_BUCKET and mean < GUESSER_BOUND_MEAN
    return {"rates": rates, "mean": mean, "eligible": eligible}
