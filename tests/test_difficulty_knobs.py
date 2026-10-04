"""Difficulty parameters of the probe generators (transfer-rulings T22). The
defaults must reproduce the items the pilot and the T1 guesser bound were
measured on, byte for byte."""
import hashlib
import json

import pytest

from amenability.suites.countdown import generate_countdown, solve_countdown
from amenability.suites.graphpath import generate_graphpath, shortest_distance

# sha256 of generate_*(300, 0) at 003a511, before the knobs existed.
COUNTDOWN_300_0 = "16e7c99b15be0e61075c1a718091916327dab2ca880a9adccc43bf9bc3b768f5"
GRAPHPATH_300_0 = "80fa8c61ba98d90ffc47a0aa4d57f308bb8aa6ab572437326cd233e6374c1894"


def _digest(items):
    rows = [[i.task_id, i.suite, i.prompt, i.answer, i.difficulty] for i in items]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _numbers(it):
    return [int(x) for x in it.answer.split("|")[0].split(",")]


def _edges(it):
    return {frozenset(e.split("-")) for e in it.answer.split("|")[0].split(",")}


def test_countdown_defaults_are_byte_identical_to_the_pilot_items():
    assert _digest(generate_countdown(300, 0)) == COUNTDOWN_300_0
    explicit = generate_countdown(
        300, 0, buckets=(3, 4, 5), number_range=(1, 20), target_range=(10, 400)
    )
    assert _digest(explicit) == COUNTDOWN_300_0


def test_graphpath_defaults_are_byte_identical_to_the_pilot_items():
    assert _digest(generate_graphpath(300, 0)) == GRAPHPATH_300_0
    explicit = generate_graphpath(
        300, 0, buckets=(8, 10, 12), min_distance=3, extra_edges_divisor=3
    )
    assert _digest(explicit) == GRAPHPATH_300_0


def test_countdown_number_and_target_ranges_are_respected():
    items = generate_countdown(60, 0, buckets=(2, 3, 4), number_range=(1, 10), target_range=(5, 100))
    assert {it.difficulty for it in items} == {2, 3, 4}
    for it in items:
        nums = _numbers(it)
        target = int(it.answer.split("|")[1])
        assert len(nums) == it.difficulty
        assert all(1 <= x <= 10 for x in nums)
        assert 5 <= target <= 100
        assert solve_countdown(nums, target) is not None


def test_countdown_id_namespace_keeps_ids_apart():
    a = generate_countdown(30, 0)
    b = generate_countdown(30, 0, id_namespace="cd-easy")
    assert all(it.task_id.startswith("probe/countdown/cd-easy/0/") for it in b)
    assert not {it.task_id for it in a} & {it.task_id for it in b}


def test_countdown_rejects_bad_ranges():
    with pytest.raises(ValueError):
        generate_countdown(30, 0, number_range=(5, 1))
    with pytest.raises(ValueError):
        generate_countdown(30, 0, target_range=(100, 5))


def test_graphpath_distance_and_density_are_respected():
    items = generate_graphpath(60, 0, buckets=(6, 8, 10), min_distance=4, extra_edges_divisor=None)
    assert {it.difficulty for it in items} == {6, 8, 10}
    for it in items:
        edges = _edges(it)
        _, s, t = it.answer.split("|")
        assert len(edges) == it.difficulty - 1          # a tree: no extra edges
        assert shortest_distance(edges, s, t) >= 4
    dense = generate_graphpath(30, 0, buckets=(9,), extra_edges_divisor=2)
    assert all(len(_edges(it)) == 8 + 9 // 2 for it in dense)


def test_graphpath_id_namespace_keeps_ids_apart():
    a = generate_graphpath(30, 0)
    b = generate_graphpath(30, 0, id_namespace="gp-x")
    assert all(it.task_id.startswith("probe/graphpath/gp-x/0/") for it in b)
    assert not {it.task_id for it in a} & {it.task_id for it in b}


def test_graphpath_rejects_an_unreachable_distance():
    # A 4-node graph has no simple path of length 4; generation would spin forever.
    with pytest.raises(ValueError, match="min_distance"):
        generate_graphpath(3, 0, buckets=(4,), min_distance=4)
