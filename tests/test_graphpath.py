import pytest
from amenability.suites.base import SuiteRegistry, TaskItem
from amenability.suites.countdown import generate_countdown
from amenability.suites.graphpath import (
    EXTRA_EDGES_DIVISOR, MIN_DISTANCE, SUITE_NAME, extract_path, generate_graphpath,
    informed_guess_pass_at_k, shortest_distance, verify_graphpath,
)


def _edges(item: TaskItem) -> set[frozenset[str]]:
    edges_s = item.answer.split("|")[0]
    return {frozenset(e.split("-")) for e in edges_s.split(",")}


def _bfs_path(item: TaskItem) -> list[str]:
    edges_s, source, target = item.answer.split("|")
    adj: dict[str, set[str]] = {}
    for e in edges_s.split(","):
        a, b = e.split("-")
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    prev = {source: None}
    frontier = [source]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u]):
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    path = [target]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def _answer(path: list[str]) -> str:
    return "<answer>" + "->".join(path) + "</answer>"


def test_generates_equal_buckets_with_unique_ids():
    items = generate_graphpath(n=300, seed=0)
    assert len(items) == 300
    by_bucket = {}
    for it in items:
        by_bucket.setdefault(it.difficulty, []).append(it)
    assert sorted(by_bucket) == [8, 10, 12]
    assert all(len(v) == 100 for v in by_bucket.values())
    assert len({it.task_id for it in items}) == 300
    assert all(it.task_id.startswith("probe/graphpath/0/") for it in items)
    assert all(it.suite == SUITE_NAME for it in items)


def test_generation_is_deterministic_in_the_seed():
    a = generate_graphpath(n=30, seed=7)
    b = generate_graphpath(n=30, seed=7)
    c = generate_graphpath(n=30, seed=8)
    assert [it.prompt for it in a] == [it.prompt for it in b]
    assert [it.prompt for it in a] != [it.prompt for it in c]


def test_every_instance_is_connected_with_the_required_distance_and_edge_count():
    for it in generate_graphpath(n=60, seed=1):
        edges = _edges(it)
        _, source, target = it.answer.split("|")
        n = it.difficulty
        assert len(edges) == n - 1 + n // EXTRA_EDGES_DIVISOR
        assert shortest_distance(edges, source, target) is not None
        assert shortest_distance(edges, source, target) >= MIN_DISTANCE
        assert source != target


def test_prompt_names_every_edge_and_both_endpoints():
    it = generate_graphpath(n=3, seed=0)[0]
    edges_s, source, target = it.answer.split("|")
    for e in edges_s.split(","):
        assert e in it.prompt
    assert f"from {source} to {target}" in it.prompt
    assert "<answer>" in it.prompt


def test_verifier_accepts_a_valid_path():
    for it in generate_graphpath(n=30, seed=2):
        assert verify_graphpath(it, _answer(_bfs_path(it)))


def test_verifier_rejects_wrong_endpoints_non_edges_and_repeats():
    it = generate_graphpath(n=3, seed=3)[0]
    path = _bfs_path(it)
    edges_s, source, target = it.answer.split("|")
    other = next(x for x in "ABCDEFGHIJKL" if x not in (source, target) and x in edges_s)
    assert not verify_graphpath(it, _answer(path[1:]))                 # wrong start
    assert not verify_graphpath(it, _answer(path[:-1]))                # wrong end
    assert not verify_graphpath(it, _answer([source, target]))         # distance >= 3, so not an edge
    assert not verify_graphpath(it, _answer(path + [path[-2], target]))  # repeats
    assert not verify_graphpath(it, _answer([source, other, other, target]))


def test_verifier_takes_the_last_answer_block_and_tolerates_whitespace():
    it = generate_graphpath(n=3, seed=4)[0]
    path = _bfs_path(it)
    spaced = "<answer> " + " -> ".join(path) + " </answer>"
    assert verify_graphpath(it, "<answer>A->B</answer> no wait " + spaced)
    assert not verify_graphpath(it, spaced + " actually <answer>A->B</answer>")


def test_verifier_rejects_unparseable_and_lowercase():
    it = generate_graphpath(n=3, seed=5)[0]
    path = _bfs_path(it)
    assert not verify_graphpath(it, "->".join(path))                      # no tags
    assert not verify_graphpath(it, _answer([p.lower() for p in path]))   # lowercase
    assert not verify_graphpath(it, "<answer></answer>")
    assert not verify_graphpath(it, "<answer>A-B-C</answer>")             # wrong separator
    assert extract_path("<answer>A -> B</answer>") == ["A", "B"]
    assert extract_path("nothing") is None


def test_informed_guesser_stays_well_below_saturation():
    """The breadth denominator is pass@32 - pass@1. If a guesser that starts at the
    source, ends at the target and fills the middle at random already reaches
    pass@32 near 1.0, breadth measures luck, not capability. Bounded per bucket
    and on average; the numbers come from the design-time simulation."""
    items = generate_graphpath(n=300, seed=0)
    rates = informed_guess_pass_at_k(items, k=32, seed=0)
    assert sorted(rates) == [8, 10, 12]
    for bucket, rate in rates.items():
        assert rate < 0.3, f"bucket {bucket} guessable at pass@32={rate:.2f}"
    assert sum(rates.values()) / len(rates) < 0.15


def test_registers_disjointly_alongside_countdown():
    reg = SuiteRegistry()
    reg.register("probe_countdown", generate_countdown(n=30, seed=0))
    reg.register(SUITE_NAME, generate_graphpath(n=30, seed=0))
    assert sorted(reg.names()) == ["probe_countdown", SUITE_NAME]
