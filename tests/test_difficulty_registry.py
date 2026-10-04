"""Named difficulty candidates for the probe training pools (transfer-rulings T22)."""
import pytest

from amenability.suites.base import SuiteRegistry, TaskItem
from amenability.suites.countdown import generate_countdown
from amenability.suites.difficulty import (
    CANDIDATES, GUESSER_BOUND_BUCKET, GUESSER_BOUND_MEAN, candidates_for, get_candidate,
    guesser_eligibility,
)
from amenability.suites.graphpath import generate_graphpath


def test_names_are_the_preregistered_ones_in_hardest_to_easiest_order():
    assert [c.name for c in candidates_for("countdown")] == [
        "cd-current", "cd-easy", "cd-mid", "cd-easier",
    ]
    assert [c.name for c in candidates_for("graphpath")] == [
        "gp-current", "gp-contingency", "gp-sparse-7", "gp-tree-6", "gp-tree-6s",
    ]
    assert set(CANDIDATES) == {c.name for k in ("countdown", "graphpath") for c in candidates_for(k)}


def test_current_candidates_reproduce_the_default_items_exactly():
    assert get_candidate("cd-current").generate(300, 0) == generate_countdown(300, 0)
    assert get_candidate("gp-current").generate(300, 0) == generate_graphpath(300, 0)


@pytest.mark.parametrize("name,expected", [
    ("cd-easy", dict(buckets=(2, 3, 4), number_range=(1, 20), target_range=(10, 400))),
    ("cd-easier", dict(buckets=(2, 3, 4), number_range=(1, 10), target_range=(5, 100))),
    ("gp-contingency", dict(buckets=(7, 9, 11), min_distance=3, extra_edges_divisor=3)),
])
def test_requested_candidates_have_the_requested_parameters(name, expected):
    assert get_candidate(name).params == expected


def test_every_candidate_generates_the_requested_count_with_its_buckets():
    for c in CANDIDATES.values():
        items = c.generate(30, 0)
        assert len(items) == 30
        assert sorted({it.difficulty for it in items}) == sorted(c.params["buckets"])
        assert all(it.suite == ("probe_countdown" if c.suite_key == "countdown" else "probe_graphpath")
                   for it in items)


def test_item_ids_are_unique_across_candidates_and_disjoint_from_targets():
    reg = SuiteRegistry()
    for c in CANDIDATES.values():
        reg.register(c.name, c.generate(300, 0))
    target = [TaskItem(task_id=f"target/gsm8k/test/{i}", suite="gsm8k", prompt="q",
                       answer="1", difficulty=0) for i in range(50)]
    reg.register("target_gsm8k", target)
    for c in CANDIDATES.values():
        assert all(it.task_id.startswith(f"probe/{c.suite_key}/") for it in reg.get(c.name))


def test_unknown_candidate_names_the_valid_ones():
    with pytest.raises(KeyError, match="cd-easy"):
        get_candidate("cd-nope")


# Informed-guesser pass@32 on the candidate's 300-item pool, item seed 0, k = 32,
# guesser seed 0: the T1 measurement, applied to every graph candidate.
GUESSER = {
    "gp-current": ({8: 0.21, 10: 0.10, 12: 0.03}, True),
    "gp-contingency": ({7: 0.37, 9: 0.12, 11: 0.06}, False),
    "gp-sparse-7": ({7: 0.27, 9: 0.10, 11: 0.02}, True),
    "gp-tree-6": ({6: 0.20, 8: 0.04, 10: 0.01}, True),
    "gp-tree-6s": ({6: 0.20, 7: 0.06, 8: 0.06}, True),
}


@pytest.mark.parametrize("name", sorted(GUESSER))
def test_graph_candidates_guesser_rates_and_eligibility(name):
    rates, eligible = GUESSER[name]
    result = guesser_eligibility(get_candidate(name))
    assert result["rates"] == pytest.approx(rates, abs=0.005)
    assert result["eligible"] is eligible
    assert result["eligible"] == (
        max(result["rates"].values()) < GUESSER_BOUND_BUCKET
        and result["mean"] < GUESSER_BOUND_MEAN
    )


def test_countdown_candidates_have_no_guesser_bound():
    for c in candidates_for("countdown"):
        assert guesser_eligibility(c) == {"rates": None, "mean": None, "eligible": True}
