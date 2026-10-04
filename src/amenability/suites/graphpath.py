"""Probe suite: find any simple path between two nodes of a small undirected graph.

Chosen as the second probe family because it has the SAME reward structure as
Countdown (search for a witness, a verifier checks the witness, any valid witness is
accepted) in a different domain, so a ranking disagreement between the two suites is
attributable to domain rather than to reward shape. Node labels are single uppercase
letters, which keeps the task tokenizer-fair.

Bucket sizes, extra-edge count and minimum distance were set by simulation, not
taste: an edge-blind guesser (starts at the source, ends at the target, fills the
middle with random distinct nodes, ignores the edge list) reaches pass@32 of 0.94 on
6-node graphs, which would make the easy bucket's breadth mostly luck. At (8, 10, 12)
nodes, n // 3 extra edges and distance >= 3, measuring this implementation's
`informed_guess_pass_at_k` on `generate_graphpath(n=300, seed=0)` with k=32, seed=0
gives 0.21 / 0.10 / 0.03 (mean 0.113), and a test holds that bound.

The bound covers edge-blind guessing only. A policy that reads the edge list and
walks it at random without revisiting nodes saturates pass@32 (1.0 in every bucket
on the same items), so this suite does not bound luck from edge-following.
"""
from __future__ import annotations

import random
import re
from collections import deque

from amenability.suites.base import TaskItem, first_answer_block
from amenability.suites.countdown import SCAFFOLD_HEAD, SCAFFOLD_TAIL

SUITE_NAME = "probe_graphpath"
EXTRA_EDGES_DIVISOR = 3
MIN_DISTANCE = 3

# Same TinyZero base-model scaffold as Countdown (transfer-rulings T19). The
# example path uses X, Y and Z, which no item can contain (labels stop at L for the
# default buckets, and `generate_graphpath` refuses buckets above 23 nodes), so it
# never verifies; every consumer scores the completion alone in any case.
PROMPT = (
    SCAFFOLD_HEAD
    + "An undirected graph has these edges: {edges}. Find a path from {source} to "
    "{target} that only uses these edges and visits no node twice. Show your work in "
    "<think> </think> tags. And return the final answer in <answer> </answer> tags, "
    "as node names joined by ->, for example <answer> X -> Y -> Z </answer>."
    + SCAFFOLD_TAIL
)

_NODE_RE = re.compile(r"[A-Z]")
_MAX_NODES = 23  # labels A..W; X, Y, Z are reserved for the prompt's example path


def _labels(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _adjacency(edges: set[frozenset[str]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for e in edges:
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def shortest_distance(edges: set[frozenset[str]], source: str, target: str) -> int | None:
    adj = _adjacency(edges)
    if source not in adj or target not in adj:
        return None
    seen = {source}
    queue = deque([(source, 0)])
    while queue:
        node, d = queue.popleft()
        if node == target:
            return d
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None


def _n_extra_edges(n: int, divisor: int | None) -> int:
    return 0 if divisor is None else n // divisor


def _random_connected_graph(
    n: int, rng: random.Random, extra_edges_divisor: int | None = EXTRA_EDGES_DIVISOR
) -> set[frozenset[str]]:
    """A random spanning tree (so the graph is connected) plus n // divisor extra
    edges (none when the divisor is None)."""
    labels = _labels(n)
    order = labels[:]
    rng.shuffle(order)
    edges: set[frozenset[str]] = set()
    for i in range(1, n):
        edges.add(frozenset((order[i], order[rng.randrange(i)])))
    while len(edges) < n - 1 + _n_extra_edges(n, extra_edges_divisor):
        edges.add(frozenset(rng.sample(labels, 2)))
    return edges


def _edge_string(edges: set[frozenset[str]]) -> str:
    return ",".join(sorted("-".join(sorted(e)) for e in edges))


def generate_graphpath(
    n: int,
    seed: int,
    buckets: tuple[int, ...] = (8, 10, 12),
    min_distance: int = MIN_DISTANCE,
    extra_edges_divisor: int | None = EXTRA_EDGES_DIVISOR,
    id_namespace: str | None = None,
) -> list[TaskItem]:
    """n items split equally over the buckets (node count per graph).

    Difficulty knobs (transfer-rulings T22): source and target are at least
    min_distance edges apart, and each graph is a random spanning tree plus
    n // extra_edges_divisor extra edges (a tree when the divisor is None). The
    defaults are the pilot's items, byte for byte. id_namespace inserts a path
    segment into every task_id, so items generated under another difficulty
    candidate never share an ID with these.
    """
    if max(buckets) > _MAX_NODES:
        raise ValueError(
            f"bucket of {max(buckets)} nodes would use labels that collide with the "
            f"prompt's example path X -> Y -> Z; at most {_MAX_NODES} nodes"
        )
    if min_distance > min(buckets) - 1:
        raise ValueError(
            f"min_distance {min_distance} is unreachable in a {min(buckets)}-node graph "
            f"(at most {min(buckets) - 1})"
        )
    prefix = "probe/graphpath/" + (f"{id_namespace}/" if id_namespace else "")
    rng = random.Random(seed)
    items: list[TaskItem] = []
    per_bucket = n // len(buckets)
    for bucket in buckets:
        made = 0
        while made < per_bucket:
            edges = _random_connected_graph(bucket, rng, extra_edges_divisor)
            source, target = rng.sample(_labels(bucket), 2)
            d = shortest_distance(edges, source, target)
            if d is None or d < min_distance:
                continue
            edge_s = _edge_string(edges)
            idx = len(items)
            items.append(
                TaskItem(
                    task_id=f"{prefix}{seed}/{idx}",
                    suite=SUITE_NAME,
                    prompt=PROMPT.format(
                        edges=", ".join(edge_s.split(",")), source=source, target=target
                    ),
                    answer=f"{edge_s}|{source}|{target}",
                    difficulty=bucket,
                )
            )
            made += 1
    return items


def extract_path(completion: str) -> list[str] | None:
    # The FIRST answer counts (transfer-rulings T20), as in Countdown.
    block = first_answer_block(completion)
    if block is None:
        return None
    raw = block.strip()
    if not raw:
        return None
    nodes = [p.strip() for p in raw.split("->")]
    if not all(_NODE_RE.fullmatch(p) for p in nodes):
        return None
    return nodes


def _parse_answer(item: TaskItem) -> tuple[set[frozenset[str]], str, str]:
    edges_s, source, target = item.answer.split("|")
    return {frozenset(e.split("-")) for e in edges_s.split(",")}, source, target


def verify_graphpath(item: TaskItem, completion: str) -> bool:
    # Strict by design, matching the prompt's explicit format: near misses such as a
    # trailing period, a Unicode arrow, backticks, comma separators or a capitalised
    # tag score 0. Read low pass@1 with that in mind.
    path = extract_path(completion)
    if path is None or len(path) < 2:
        return False
    edges, source, target = _parse_answer(item)
    if path[0] != source or path[-1] != target:
        return False
    if len(set(path)) != len(path):
        return False
    return all(frozenset((a, b)) in edges for a, b in zip(path, path[1:]))


def informed_guess_pass_at_k(items: list[TaskItem], k: int, seed: int) -> dict[int, float]:
    """pass@k of a format-aware, edge-blind guesser: starts at the source, ends at the
    target, fills the middle with random distinct nodes of random length, and ignores
    the edge list. It is NOT the strongest non-reasoning guesser. Restricting guesses
    to short lengths scores higher (about 0.50 / 0.24 / 0.23 on the test's items), and
    a random walk along the listed edges with no revisits saturates pass@32 at 1.0.
    The bound it supports is therefore against edge-blind guessing only."""
    rng = random.Random(seed)
    hits: dict[int, int] = {}
    counts: dict[int, int] = {}
    for it in items:
        edges, source, target = _parse_answer(it)
        labels = sorted({node for e in edges for node in e})
        others = [x for x in labels if x not in (source, target)]
        hit = False
        for _ in range(k):
            length = rng.randint(2, len(labels))
            middle = rng.sample(others, length - 2)
            completion = "<answer>" + "->".join([source, *middle, target]) + "</answer>"
            if verify_graphpath(it, completion):
                hit = True
                break
        counts[it.difficulty] = counts.get(it.difficulty, 0) + 1
        hits[it.difficulty] = hits.get(it.difficulty, 0) + int(hit)
    return {b: hits[b] / counts[b] for b in sorted(counts)}
