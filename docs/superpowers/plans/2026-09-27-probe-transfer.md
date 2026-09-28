# Probe Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a model's probe score is invariant to the probe task family, by probing all ten roster models on two unrelated suites (plus a seed replicate), and produce a pre-registered verdict that decides whether the product is a single leaderboard number or per-domain.

**Architecture:** A second procedurally generated probe suite (graph path-finding) joins Countdown behind a small suite catalog. The probe itself (pre-eval, 60-step GRPO, post-eval) is factored out of `RealRunner` into one library function that a launcher-agnostic per-job CLI calls, so a SLURM job array can run one (model, suite, seed) job per task. An analysis script computes the transfer statistics against a test-retest ceiling and applies the decision rule committed in `prereg/transfer.md` before the batch runs.

**Tech Stack:** Python 3.11, uv, PyTorch 2.13, transformers 5.17, TRL 1.13 (`GRPOTrainer`), vLLM 0.30, NumPy/SciPy, pytest, SLURM on USC CARC. Modal is the documented fallback only.

**Spec:** `docs/superpowers/specs/2026-09-27-probe-transfer-design.md`. Read it first. The parent spec is `docs/superpowers/specs/2026-09-22-rl-amenability-benchmark-design.md` and the rulings so far are in `docs/decisions/stage0-rulings.md`.

## Global Constraints

Every task's requirements implicitly include this section.

- **The probe protocol is fixed and identical for every model and both suites:** 60 GRPO steps, `num_generations=8`, `PROMPTS_PER_STEP=4`, lr `1e-6`, `beta=0.04`, temperature `1.0`, 300 probe items in three buckets of 100, pass@k at `(1, 8, 32, 64)` from 64 samples at temperature 1.0. The only permitted change is the learning-rate recalibration bound in spec section 8, on one model, before the batch.
- **Full-parameter training only. No LoRA, no adapters, no bf16 master weights.** Loading the model in bf16 to save memory is a silent protocol change and is forbidden; request a bigger GPU instead.
- **Items are generated from `item_seed=0` for every arm.** Only the run seed varies between the seed-0 and seed-1 arms. If a test-retest pair sees different items, the replicate measures nothing.
- **Probe task IDs never overlap target task IDs.** Enforced by `SuiteRegistry`; the new suite registers alongside the existing two.
- **No change to `score.py`, `baselines.py`, `gates.py`, `telemetry.py`'s feature definitions, or `models.yaml`.** The analysis consumes them unchanged.
- **Nothing in this plan reads a transfer result before `prereg/transfer.md` is committed.** Task 8 must be committed before the batch in Task 10 is launched.
- **Failed or missing jobs are named in the report, never silently dropped.** Every statistic prints its N.
- **Commits carry no AI attribution trailers** (ruling R10, owner's `CLAUDE.md`). No em dashes anywhere.
- **Every decision made during execution that the plan did not anticipate is recorded** in `docs/decisions/transfer-rulings.md` as `T<n>` with what was decided, why, and the cost if wrong, following the format of `stage0-rulings.md`.
- Run the CPU suite with `uv run pytest -q -m "not gpu"` before every commit. It is 177 green at the start of this plan and must never go below that.

## Review Focus

Failure modes the spec implies that a person will hit; each is pinned to a test in the task that owns the code.

1. **A completion with several `<answer>` blocks, whitespace around `->`, or lowercase node names.** The verifier must take the last block, tolerate whitespace, and reject lowercase (nodes are uppercase letters by construction). Task 2.
2. **A model with no breadth on one suite** (pass@32 equals pass@1, likely for the weakest model on Graph). `extract_features` raises by design; the analysis must exclude that model by name and reduce N, not crash. Task 9.
3. **A crashed job leaving no telemetry file, or a truncated one.** The analysis lists it as a failure and the decision falls to "inconclusive: rerun failures" below eight complete models. Task 9.
4. **Cache collision across suites and seeds.** Requesting Graph telemetry when only Countdown is cached must run the probe, not return Countdown's file. Task 4.
5. **A test-retest replicate that quietly uses different items.** `run_probe` with run seeds 0 and 1 must hand the trainer byte-identical prompts. Task 4.
6. **A job-manifest line naming a model key that is not in the registry.** The CLI must fail with the list of valid keys before any GPU time is spent, not after a 10-minute pre-eval. Task 5.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/amenability/training/grpo.py` | Modify: release trainer GPU memory before returning (#33) |
| `src/amenability/suites/graphpath.py` | New: graph path generator, verifier, informed-guesser baseline |
| `src/amenability/suites/catalog.py` | New: `ProbeSuite` and `PROBE_SUITES`, the name-to-suite lookup |
| `src/amenability/probe/run.py` | New: `ProbeConfig`, `run_probe`, cache paths, `passk_by_bucket` |
| `scripts/real_runner.py` | Modify: `RealRunner.probe` delegates to `run_probe` |
| `scripts/run_probe.py` | New: per-job CLI |
| `scripts/make_transfer_jobs.py` | New: writes the job manifest |
| `scripts/prefetch_models.py` | New: downloads and checks every roster model (#20) |
| `scripts/slurm/prefetch.sbatch` | New: CPU job for prefetch |
| `scripts/slurm/probe_array.sbatch` | New: GPU job array, one job per line of the manifest |
| `scripts/slurm/README.md` | New: the exact launch sequence |
| `prereg/transfer.md` | New: the decision rule, committed before the batch |
| `docs/decisions/transfer-rulings.md` | New: execution rulings T1... |
| `scripts/analyze_transfer.py` | New: statistics, decision, calibration table, leaderboard preview, report |
| `tests/test_graphpath.py`, `tests/test_catalog.py`, `tests/test_probe_run.py`, `tests/test_run_probe_cli.py`, `tests/test_transfer_jobs.py`, `tests/test_prefetch.py`, `tests/test_analyze_transfer.py` | Tests, one file per unit |
| `.gitignore` | Modify: ignore `results/**/probes/` and `logs/` |

---

## Checkpoint H1: CARC allocation and node inventory (owner action, before Task 7)

Nothing before Task 7 needs a GPU, so Tasks 1 to 6 proceed while this is pending. Task 7 needs the answers.

Ask the owner to run these on a CARC login node and paste the output:

1. `myaccount` (shows project allocations and GPU-hours remaining)
2. `sinfo -p gpu -o "%N %G %f %l"` (node names, GPU types, SLURM feature names, time limits)
3. `echo $SCRATCH; df -h $SCRATCH | tail -1` (scratch path and free space; the plan needs ~50 GB)
4. `srun --partition=gpu --gres=gpu:a100:1 --time=00:03:00 --pty nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` (confirms A100 memory size and that the driver supports torch 2.13's CUDA)

Decision rule from #19: if GPU-hours available < 400, this batch still runs on CARC (it needs ~40), but record in a ruling that Stage 0 will go to Modal. If the array cannot get A100-80GB nodes and the smoke probe in Task 10 reports peak memory above 36 GB, Task 10's contingency applies.

## Checkpoint H2: gated weights (owner action, before Task 10)

1. On huggingface.co, accept the licences for `meta-llama/Llama-3.2-1B` and `google/gemma-3-1b-pt` with the account whose token will be used on CARC.
2. On a CARC login node: `uv run hf auth login` (paste the token). It is stored under `$HF_HOME`, which Task 7 points at scratch, so run this **after** `export HF_HOME=$SCRATCH/hf`.

---

### Task 1: Release trainer GPU memory in `run_grpo` (#33)

**Files:**
- Modify: `src/amenability/training/grpo.py` (the tail of `run_grpo`, after the manifest write)
- Test: `tests/test_grpo_runner.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_grpo` still returns `list[StepRecord]`, but the trainer object is unreachable and `torch.cuda.empty_cache()` has been called by the time it returns. Task 4 relies on this so that vLLM can initialise in the same process after training.

**Why the closure matters.** `TelemetryCallback` is built with `model_getter=lambda: getattr(trainer, "model", None)`. That lambda closes over the `trainer` variable, the trainer holds the callback in its handler list, and `run_grpo` reads `callback.records` at the end. So `del trainer` alone frees nothing: the callback keeps the closure, the closure keeps the cell, the cell keeps the trainer, with its model, reference model and optimiser state, on the GPU. The closure has to be broken explicitly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grpo_runner.py`:

```python
import gc
import weakref


def test_run_grpo_leaves_no_reference_to_the_trainer(tmp_path):
    """After run_grpo returns, the trainer must be collectable: its model, reference
    model and optimiser state are what hold the GPU, and vLLM initialises in the
    same process right after (#33)."""
    refs = {}

    class Trainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.reward_funcs = kwargs["reward_funcs"]
            self.model = object()
            refs["trainer"] = weakref.ref(self)

        def train(self):
            # Same shape as _FakeTrainer.train below: a real step scores a group
            # of completions (prompt "p0" comes from items()) before logging.
            for cb in self.callbacks:
                self.reward_funcs[0](completions=["7", "8"], prompts=["p0", "p0"])
                cb.on_log(
                    args=None,
                    state=type("S", (), {"global_step": 1, "log_history": []})(),
                    control=None,
                    logs={"kl": 0.0, "grad_norm": 1.0},
                )

        def save_model(self, path):
            pass

    spec = GRPOSpec(
        model_path="x", model_key="k", items=items(2), verify_fn=verify, max_steps=1,
        num_generations=2, learning_rate=1e-6, beta=0.04, temperature=1.0, seed=0,
        output_dir=str(tmp_path / "out"), save_steps=None,
    )
    records = run_grpo(spec, trainer_factory=lambda **kw: Trainer(**kw))
    assert len(records) == 1
    gc.collect()
    assert refs["trainer"]() is None, "run_grpo still holds the trainer after returning"


def test_run_grpo_empties_the_cuda_cache_when_cuda_is_available(tmp_path, monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    spec = GRPOSpec(
        model_path="x", model_key="k", items=items(2), verify_fn=verify, max_steps=1,
        num_generations=2, learning_rate=1e-6, beta=0.04, temperature=1.0, seed=0,
        output_dir=str(tmp_path / "out"), save_steps=None,
    )
    run_grpo(spec, trainer_factory=lambda **kw: _FakeTrainer(**kw))
    assert calls == ["empty"]
```

`items`, `verify`, `GRPOSpec`, `run_grpo` and `_FakeTrainer` already exist in that test file; reuse them, do not redefine them.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_grpo_runner.py -q -k "no_reference or empties_the_cuda"`
Expected: both FAIL. The first with the assertion message, the second with `assert [] == ["empty"]`.

- [ ] **Step 3: Implement the release**

In `src/amenability/training/grpo.py`, replace the final `return callback.records` of `run_grpo` with:

```python
    records = list(callback.records)
    # Break the closure before dropping the trainer. model_getter closes over the
    # `trainer` variable, the trainer holds the callback, and this function reads
    # callback.records, so without this the trainer (model, reference model,
    # optimiser state) stays reachable and stays on the GPU. The probe then
    # initialises a vLLM engine at 0.85 memory utilisation in the same process,
    # which is the OOM issue #33 describes.
    callback.model_getter = lambda: None
    del callback, model_obj, trainer
    _release_gpu()
    return records
```

and add at module level, below `_NullEntropyProbe`:

```python
def _release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_grpo_runner.py -q`
Expected: all PASS, including the two new ones.

- [ ] **Step 5: Run the whole CPU suite and commit**

Run: `uv run pytest -q -m "not gpu"`
Expected: 179 passed.

```bash
git add src/amenability/training/grpo.py tests/test_grpo_runner.py
git commit -m "fix: release trainer GPU memory before run_grpo returns (#33)

The telemetry callback's model_getter closed over the trainer, so deleting
the local was not enough to free the model, reference model and optimiser
state before the probe's vLLM engine initialised in the same process."
```

---

### Task 2: Graph path-finding probe suite

**Files:**
- Create: `src/amenability/suites/graphpath.py`
- Test: `tests/test_graphpath.py`

**Interfaces:**
- Consumes: `TaskItem` from `amenability.suites.base`.
- Produces:
  - `generate_graphpath(n: int, seed: int, buckets: tuple[int, ...] = (8, 10, 12)) -> list[TaskItem]`
  - `verify_graphpath(item: TaskItem, completion: str) -> bool`
  - `extract_path(completion: str) -> list[str] | None`
  - `shortest_distance(edges: set[frozenset[str]], source: str, target: str) -> int | None`
  - `informed_guess_pass_at_k(items: list[TaskItem], k: int, seed: int) -> dict[int, float]` (bucket -> pass@k)
  - Constants `EXTRA_EDGES_DIVISOR = 3`, `MIN_DISTANCE = 3`, `SUITE_NAME = "probe_graphpath"`.
  - `TaskItem.answer` format: `"A-B,A-C,...|<source>|<target>"`, edges sorted, each edge's endpoints sorted. `TaskItem.difficulty` is the node count.

Mirror `src/amenability/suites/countdown.py` in structure and style. Read it first.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graphpath.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graphpath.py -q`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'amenability.suites.graphpath'`.

- [ ] **Step 3: Implement the suite**

```python
# src/amenability/suites/graphpath.py
"""Probe suite: find any simple path between two nodes of a small undirected graph.

Chosen as the second probe family because it has the SAME reward structure as
Countdown (search for a witness, a verifier checks the witness, any valid witness is
accepted) in a different domain, so a ranking disagreement between the two suites is
attributable to domain rather than to reward shape. Node labels are single uppercase
letters, which keeps the task tokenizer-fair.

Bucket sizes, extra-edge count and minimum distance were set by simulation, not
taste: an informed guesser (starts at the source, ends at the target, fills the
middle with random distinct nodes) reaches pass@32 of 0.94 on 6-node graphs, which
would make the easy bucket's breadth mostly luck. At (8, 10, 12) nodes, n // 3 extra
edges and distance >= 3 the same guesser scores about 0.20 / 0.06 / 0.01, and
`informed_guess_pass_at_k` exists so a test can hold that bound.
"""
from __future__ import annotations

import random
import re
from collections import deque

from amenability.suites.base import TaskItem

SUITE_NAME = "probe_graphpath"
EXTRA_EDGES_DIVISOR = 3
MIN_DISTANCE = 3

PROMPT = (
    "An undirected graph has these edges: {edges}.\n"
    "Find a path from {source} to {target} that only uses these edges and visits no "
    "node twice.\n"
    "Put only the path, as node names joined by ->, inside <answer></answer> tags."
)

_NODE_RE = re.compile(r"[A-Z]")


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


def _random_connected_graph(n: int, rng: random.Random) -> set[frozenset[str]]:
    """A random spanning tree (so the graph is connected) plus n // 3 extra edges."""
    labels = _labels(n)
    order = labels[:]
    rng.shuffle(order)
    edges: set[frozenset[str]] = set()
    for i in range(1, n):
        edges.add(frozenset((order[i], order[rng.randrange(i)])))
    while len(edges) < n - 1 + n // EXTRA_EDGES_DIVISOR:
        edges.add(frozenset(rng.sample(labels, 2)))
    return edges


def _edge_string(edges: set[frozenset[str]]) -> str:
    return ",".join(sorted("-".join(sorted(e)) for e in edges))


def generate_graphpath(n: int, seed: int, buckets: tuple[int, ...] = (8, 10, 12)) -> list[TaskItem]:
    rng = random.Random(seed)
    items: list[TaskItem] = []
    per_bucket = n // len(buckets)
    for bucket in buckets:
        made = 0
        while made < per_bucket:
            edges = _random_connected_graph(bucket, rng)
            source, target = rng.sample(_labels(bucket), 2)
            d = shortest_distance(edges, source, target)
            if d is None or d < MIN_DISTANCE:
                continue
            edge_s = _edge_string(edges)
            idx = len(items)
            items.append(
                TaskItem(
                    task_id=f"probe/graphpath/{seed}/{idx}",
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
    matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    if not matches:
        return None
    raw = matches[-1].strip()
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
    """pass@k of a guesser that knows the format: starts at the source, ends at the
    target, fills the middle with random distinct nodes of random length. This is the
    strongest guesser that does no reasoning, and the bound the suite must clear."""
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_graphpath.py -q -v`
Expected: 10 PASS. If `test_informed_guesser_stays_well_below_saturation` fails, do NOT loosen the threshold: the generator parameters are wrong. Report the measured rates; the bound contingency is in the spec (section 4) and must be recorded as a ruling.

- [ ] **Step 5: Run the whole CPU suite and commit**

Run: `uv run pytest -q -m "not gpu"`
Expected: 189 passed.

```bash
git add src/amenability/suites/graphpath.py tests/test_graphpath.py
git commit -m "feat: graph path-finding probe suite with guessability bound

Second probe family for the transfer question. Same witness-search reward
structure as Countdown in a different domain. Buckets (8, 10, 12), n // 3
extra edges, distance >= 3: an informed guesser scores pass@32 of about
0.20 / 0.06 / 0.01, held by a test."
```

---

### Task 3: Probe suite catalog

**Files:**
- Create: `src/amenability/suites/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `generate_countdown`, `verify_countdown` from `amenability.suites.countdown`; `generate_graphpath`, `verify_graphpath`, `SUITE_NAME` from `amenability.suites.graphpath`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ProbeSuite:
      key: str        # short CLI key: "countdown", "graphpath"
      name: str       # SuiteRegistry name and TaskItem.suite: "probe_countdown", "probe_graphpath"
      generate: Callable[[int, int], list[TaskItem]]   # (n, seed)
      verify: Callable[[TaskItem, str], bool]

  PROBE_SUITES: dict[str, ProbeSuite]
  def get_probe_suite(key: str) -> ProbeSuite   # KeyError naming the valid keys
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catalog.py
import pytest
from amenability.suites.catalog import PROBE_SUITES, ProbeSuite, get_probe_suite


def test_catalog_has_both_probe_suites():
    assert sorted(PROBE_SUITES) == ["countdown", "graphpath"]
    assert all(isinstance(s, ProbeSuite) for s in PROBE_SUITES.values())


def test_suite_names_match_the_items_they_generate():
    for key, suite in PROBE_SUITES.items():
        items = suite.generate(6, 0)
        assert suite.key == key
        assert all(it.suite == suite.name for it in items)
        assert all(it.task_id.startswith(f"probe/{key}/") for it in items)


def test_unknown_key_names_the_valid_ones():
    with pytest.raises(KeyError, match="countdown, graphpath"):
        get_probe_suite("gsm8k")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement the catalog**

```python
# src/amenability/suites/catalog.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
uv run pytest -q -m "not gpu"
git add src/amenability/suites/catalog.py tests/test_catalog.py
git commit -m "feat: probe suite catalog keyed by short name"
```

---

### Task 4: `run_probe` library function and `RealRunner` delegation

**Files:**
- Create: `src/amenability/probe/run.py`
- Modify: `scripts/real_runner.py` (`RealRunner.probe`, lines 47-85, and its imports)
- Modify: `tests/test_real_runner.py` (fixture monkeypatch targets and the expected telemetry path)
- Test: `tests/test_probe_run.py`

**Interfaces:**
- Consumes: `get_probe_suite` (Task 3); `evaluate_passk`, `PassKResult` from `amenability.eval.passk`; `GRPOSpec`, `run_grpo` from `amenability.training.grpo`; `ProbeTelemetry` from `amenability.probe.telemetry`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ProbeConfig:
      probe_steps: int = 60
      num_generations: int = 8
      n_probe_items: int = 300
      learning_rate: float = 1e-6
      beta: float = 0.04
      temperature: float = 1.0
      n_samples: int = 64
      ks: tuple[int, ...] = (1, 8, 32, 64)

  def probe_key(model_key: str, suite_key: str, run_seed: int) -> str
      # f"{model_key}-{suite_key}-grpo-s{run_seed}"
  def telemetry_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path
      # work_dir / "telemetry" / f"{probe_key}.json"
  def preeval_path(work_dir: Path, model_key: str, suite_key: str) -> Path
      # work_dir / "preeval" / f"{model_key}-{suite_key}.json"
  def meta_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path
      # work_dir / "meta" / f"{probe_key}.json"
  def passk_by_bucket(result: PassKResult, items: list[TaskItem], ks: tuple[int, ...]) -> dict[int, dict[int, float]]
      # bucket -> {k -> pass@k over that bucket's items}, using pass_at_k per item
  def run_probe(*, model_path: str, model_key: str, suite_key: str, work_dir: Path,
                config: ProbeConfig = ProbeConfig(), item_seed: int = 0, run_seed: int = 0,
                force: bool = False, keep_checkpoint: bool = False, pre_only: bool = False,
                evaluate_fn=None, train_fn=None) -> ProbeTelemetry | PassKResult
  ```
  `evaluate_fn` defaults to the module-level `evaluate_passk` and `train_fn` to the module-level `run_grpo`, so tests monkeypatch `amenability.probe.run.evaluate_passk` / `.run_grpo` or inject directly.

**Behaviour of `run_probe`:**
1. `suite = get_probe_suite(suite_key)`; `items = suite.generate(config.n_probe_items, item_seed)`.
2. If `pre_only`: if `preeval_path` exists and not `force`, load and return its `pre` as a `PassKResult`. Otherwise evaluate on `model_path`, write `{"model_key", "suite_key", "item_seed", "n_items", "pre": asdict(pre), "by_bucket": passk_by_bucket(...)}` with `indent=2, sort_keys=True`, return `pre`. Never call `train_fn`.
3. Otherwise: if `telemetry_path` exists and not `force`, print `reusing cached probe telemetry ...` and return `ProbeTelemetry.from_json(...)`.
4. Pre-eval on `model_path` with `suite.verify`, `config.ks`, `config.n_samples`, `config.temperature`, `seed=run_seed`.
5. `out_dir = work_dir / "probes" / probe_key`. `train_fn(GRPOSpec(model_path=model_path, model_key=model_key, items=items, verify_fn=suite.verify, max_steps=config.probe_steps, num_generations=config.num_generations, learning_rate=config.learning_rate, beta=config.beta, temperature=config.temperature, seed=run_seed, output_dir=str(out_dir), save_steps=None))`.
6. Post-eval on `str(out_dir)`, same arguments.
7. Write telemetry (round-trip through `to_json` exactly as `RealRunner.probe` does today). Write meta: `{"probe_key", "model_key", "suite_key", "item_seed", "run_seed", "learning_rate", "probe_steps", "n_step_records", "wall_seconds": {"pre", "train", "post"}, "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}`.
8. If `out_dir / "run_manifest.json"` exists, copy it to `work_dir / "manifests" / f"{probe_key}.json"`.
9. Unless `keep_checkpoint`, `shutil.rmtree(out_dir, ignore_errors=True)`.
10. Return the `ProbeTelemetry`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_probe_run.py
"""run_probe with fakes for the two GPU collaborators."""
import json
from pathlib import Path

import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.run import (
    ProbeConfig, meta_path, passk_by_bucket, preeval_path, probe_key, run_probe,
    telemetry_path,
)
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem


def _passk(items, p1, p32):
    return PassKResult(
        ks={1: p1, 8: p1, 32: p32, 64: p32}, n_samples=64, n_items=len(items),
        per_item_correct={it.task_id: (1 if i % 2 == 0 else 0) for i, it in enumerate(items)},
    )


@pytest.fixture
def fakes():
    state = {"passk_calls": 0, "train_specs": [], "passk_items": []}

    def fake_passk(model_path, items, verify_fn, ks, **kw):
        state["passk_calls"] += 1
        state["passk_items"].append(items)
        state["verify_fn"] = verify_fn
        return _passk(items, 0.1 if state["passk_calls"] % 2 == 1 else 0.3, 0.5)

    def fake_train(spec, **kw):
        state["train_specs"].append(spec)
        Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(spec.output_dir) / "run_manifest.json").write_text(json.dumps({"seed": spec.seed}))
        (Path(spec.output_dir) / "model.safetensors").write_bytes(b"weights")
        return [
            StepRecord(step=i, mean_reward=0.1 * i, group_reward_std=0.2,
                       zero_advantage_frac=0.1, policy_entropy=2.0, kl=0.01, grad_norm=1.0)
            for i in range(1, 4)
        ]

    return state, fake_passk, fake_train


def _run(tmp_path, fakes, **kw):
    state, fake_passk, fake_train = fakes
    args = dict(model_path="hf/x", model_key="m", suite_key="countdown", work_dir=tmp_path,
                config=ProbeConfig(n_probe_items=9), evaluate_fn=fake_passk, train_fn=fake_train)
    args.update(kw)
    return run_probe(**args)


def test_paths_carry_suite_and_seed(tmp_path):
    assert probe_key("m", "graphpath", 1) == "m-graphpath-grpo-s1"
    assert telemetry_path(tmp_path, "m", "graphpath", 1) == tmp_path / "telemetry" / "m-graphpath-grpo-s1.json"
    assert preeval_path(tmp_path, "m", "graphpath") == tmp_path / "preeval" / "m-graphpath.json"
    assert meta_path(tmp_path, "m", "graphpath", 1) == tmp_path / "meta" / "m-graphpath-grpo-s1.json"


def test_writes_telemetry_meta_and_manifest_then_deletes_the_checkpoint(tmp_path, fakes):
    t = _run(tmp_path, fakes)
    assert t.model_key == "m" and len(t.steps) == 3
    assert telemetry_path(tmp_path, "m", "countdown", 0).exists()
    meta = json.loads(meta_path(tmp_path, "m", "countdown", 0).read_text())
    assert meta["suite_key"] == "countdown" and meta["run_seed"] == 0 and meta["item_seed"] == 0
    assert meta["learning_rate"] == pytest.approx(1e-6) and meta["probe_steps"] == 60
    assert set(meta["wall_seconds"]) == {"pre", "train", "post"}
    assert (tmp_path / "manifests" / "m-countdown-grpo-s0.json").exists()
    assert not (tmp_path / "probes" / "m-countdown-grpo-s0").exists(), "checkpoint not deleted"


def test_keep_checkpoint_leaves_the_directory(tmp_path, fakes):
    _run(tmp_path, fakes, keep_checkpoint=True)
    assert (tmp_path / "probes" / "m-countdown-grpo-s0" / "model.safetensors").exists()


def test_cache_is_keyed_by_suite_and_seed(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes)
    assert len(state["train_specs"]) == 1
    _run(tmp_path, fakes)                          # same suite, same seed: cached
    assert len(state["train_specs"]) == 1
    _run(tmp_path, fakes, suite_key="graphpath")   # other suite: must NOT hit the cache
    assert len(state["train_specs"]) == 2
    _run(tmp_path, fakes, run_seed=1)              # other seed: must NOT hit the cache
    assert len(state["train_specs"]) == 3
    _run(tmp_path, fakes, force=True)
    assert len(state["train_specs"]) == 4


def test_run_seed_changes_the_trainer_seed_but_not_the_items(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes, run_seed=0)
    _run(tmp_path, fakes, run_seed=1)
    s0, s1 = state["train_specs"]
    assert s0.seed == 0 and s1.seed == 1
    assert [it.prompt for it in s0.items] == [it.prompt for it in s1.items]
    assert [it.task_id for it in s0.items] == [it.task_id for it in s1.items]


def test_item_seed_changes_the_items(tmp_path, fakes):
    state = fakes[0]
    _run(tmp_path, fakes, item_seed=0)
    _run(tmp_path, fakes, item_seed=5, run_seed=0, force=True)
    a, b = state["train_specs"]
    assert [it.prompt for it in a.items] != [it.prompt for it in b.items]


def test_suite_verifier_reaches_both_evaluation_and_training(tmp_path, fakes):
    from amenability.suites.graphpath import verify_graphpath
    state = fakes[0]
    _run(tmp_path, fakes, suite_key="graphpath")
    assert state["verify_fn"] is verify_graphpath
    assert state["train_specs"][0].verify_fn is verify_graphpath
    assert all(it.suite == "probe_graphpath" for it in state["train_specs"][0].items)


def test_post_eval_targets_the_trained_checkpoint(tmp_path, fakes, monkeypatch):
    seen = []
    state, fake_passk, fake_train = fakes

    def spy(model_path, items, verify_fn, ks, **kw):
        seen.append(model_path)
        return fake_passk(model_path, items, verify_fn, ks, **kw)

    _run(tmp_path, fakes, evaluate_fn=spy)
    assert seen == ["hf/x", str(tmp_path / "probes" / "m-countdown-grpo-s0")]


def test_pre_only_writes_by_bucket_and_never_trains(tmp_path, fakes):
    state = fakes[0]
    res = _run(tmp_path, fakes, pre_only=True)
    assert isinstance(res, PassKResult)
    assert state["train_specs"] == []
    written = json.loads(preeval_path(tmp_path, "m", "countdown").read_text())
    assert written["model_key"] == "m" and written["suite_key"] == "countdown"
    assert sorted(int(b) for b in written["by_bucket"]) == [3, 4, 5]
    for bucket in written["by_bucket"].values():
        assert set(bucket) == {"1", "8", "32", "64"}
    # cached on the second call
    _run(tmp_path, fakes, pre_only=True)
    assert state["passk_calls"] == 1


def test_passk_by_bucket_uses_the_unbiased_estimator():
    items = [TaskItem(task_id=f"t{i}", suite="s", prompt="p", answer="a", difficulty=1 + i // 2)
             for i in range(4)]
    res = PassKResult(ks={}, n_samples=4, n_items=4,
                      per_item_correct={"t0": 4, "t1": 0, "t2": 2, "t3": 2})
    out = passk_by_bucket(res, items, (1, 4))
    assert out[1][1] == pytest.approx(0.5)       # (1.0 + 0.0) / 2
    assert out[1][4] == pytest.approx(0.5)       # pass@4 with n=4: (1.0 + 0.0) / 2
    assert out[2][1] == pytest.approx(0.5)       # (0.5 + 0.5) / 2
    assert out[2][4] == pytest.approx(1.0)       # 2 of 4 correct, k = n: guaranteed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_probe_run.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement `run.py`**

```python
# src/amenability/probe/run.py
"""The probe: pre-eval, a short fixed-budget GRPO run, post-eval, telemetry to disk.

This is the ONE implementation of the probe. RealRunner (Stage 0) and the per-job
CLI (the transfer batch) both call it, so the protocol cannot drift between the
two. Everything that varies between jobs is an argument; everything that must not
vary lives in ProbeConfig's defaults.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from amenability.eval.passk import PassKResult, evaluate_passk, pass_at_k
from amenability.probe.telemetry import ProbeTelemetry
from amenability.suites.base import TaskItem
from amenability.suites.catalog import get_probe_suite
from amenability.training.grpo import GRPOSpec, run_grpo


@dataclass(frozen=True)
class ProbeConfig:
    probe_steps: int = 60
    num_generations: int = 8
    n_probe_items: int = 300
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0
    n_samples: int = 64
    ks: tuple[int, ...] = (1, 8, 32, 64)


def probe_key(model_key: str, suite_key: str, run_seed: int) -> str:
    return f"{model_key}-{suite_key}-grpo-s{run_seed}"


def telemetry_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path:
    return work_dir / "telemetry" / f"{probe_key(model_key, suite_key, run_seed)}.json"


def meta_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path:
    return work_dir / "meta" / f"{probe_key(model_key, suite_key, run_seed)}.json"


def preeval_path(work_dir: Path, model_key: str, suite_key: str) -> Path:
    return work_dir / "preeval" / f"{model_key}-{suite_key}.json"


def passk_by_bucket(
    result: PassKResult, items: list[TaskItem], ks: tuple[int, ...]
) -> dict[int, dict[int, float]]:
    by_bucket: dict[int, list[int]] = {}
    for it in items:
        by_bucket.setdefault(it.difficulty, []).append(result.per_item_correct[it.task_id])
    return {
        bucket: {
            k: float(sum(pass_at_k(result.n_samples, c, k) for c in correct) / len(correct))
            for k in ks
        }
        for bucket, correct in sorted(by_bucket.items())
    }


def _passk_from_dict(d: dict) -> PassKResult:
    return PassKResult(
        ks={int(k): v for k, v in d["ks"].items()}, n_samples=d["n_samples"],
        n_items=d["n_items"], per_item_correct=d["per_item_correct"],
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _peak_memory_bytes() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def run_probe(
    *,
    model_path: str,
    model_key: str,
    suite_key: str,
    work_dir: Path,
    config: ProbeConfig = ProbeConfig(),
    item_seed: int = 0,
    run_seed: int = 0,
    force: bool = False,
    keep_checkpoint: bool = False,
    pre_only: bool = False,
    evaluate_fn=None,
    train_fn=None,
) -> ProbeTelemetry | PassKResult:
    evaluate = evaluate_fn or evaluate_passk
    train = train_fn or run_grpo
    suite = get_probe_suite(suite_key)
    # Items come from item_seed, NEVER from run_seed: a seed replicate must see
    # byte-identical prompts or it measures item variance, not run variance.
    items = suite.generate(config.n_probe_items, item_seed)

    def evaluate_at(path: str) -> PassKResult:
        return evaluate(
            path, items, suite.verify, config.ks,
            n_samples=config.n_samples, temperature=config.temperature, seed=run_seed,
        )

    if pre_only:
        out = preeval_path(work_dir, model_key, suite_key)
        if out.exists() and not force:
            print(f"reusing cached pre-eval for {model_key}/{suite_key} from {out}")
            return _passk_from_dict(json.loads(out.read_text())["pre"])
        pre = evaluate_at(model_path)
        _write_json(out, {
            "model_key": model_key, "suite_key": suite_key, "item_seed": item_seed,
            "n_items": len(items), "pre": asdict(pre),
            "by_bucket": passk_by_bucket(pre, items, config.ks),
        })
        return pre

    key = probe_key(model_key, suite_key, run_seed)
    t_path = telemetry_path(work_dir, model_key, suite_key, run_seed)
    if t_path.exists() and not force:
        print(f"reusing cached probe telemetry for {key} from {t_path}")
        return ProbeTelemetry.from_json(json.loads(t_path.read_text()))

    wall: dict[str, float] = {}
    t0 = time.monotonic()
    pre = evaluate_at(model_path)
    wall["pre"] = time.monotonic() - t0

    out_dir = work_dir / "probes" / key
    t0 = time.monotonic()
    records = train(
        GRPOSpec(
            model_path=model_path, model_key=model_key, items=items,
            verify_fn=suite.verify, max_steps=config.probe_steps,
            num_generations=config.num_generations, learning_rate=config.learning_rate,
            beta=config.beta, temperature=config.temperature, seed=run_seed,
            output_dir=str(out_dir), save_steps=None,
        )
    )
    wall["train"] = time.monotonic() - t0

    t0 = time.monotonic()
    post = evaluate_at(str(out_dir))
    wall["post"] = time.monotonic() - t0

    telemetry = ProbeTelemetry(
        model_key=model_key, algorithm="grpo", steps=records, pre=pre, post=post
    )
    # Round-trip through to_json so ProbeTelemetry stays the single definition of
    # the on-disk shape that from_json reads back.
    _write_json(t_path, json.loads(telemetry.to_json()))
    _write_json(meta_path(work_dir, model_key, suite_key, run_seed), {
        "probe_key": key, "model_key": model_key, "suite_key": suite_key,
        "item_seed": item_seed, "run_seed": run_seed,
        "learning_rate": config.learning_rate, "probe_steps": config.probe_steps,
        "n_step_records": len(records), "wall_seconds": wall,
        "peak_memory_bytes": _peak_memory_bytes(),
    })
    manifest = out_dir / "run_manifest.json"
    if manifest.exists():
        dest = work_dir / "manifests" / f"{key}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, dest)
    if not keep_checkpoint:
        # A probe checkpoint is ~3.5 GB and nothing downstream reads it; thirty of
        # them would exhaust a home quota. The manifest and telemetry survive.
        shutil.rmtree(out_dir, ignore_errors=True)
    return telemetry
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_probe_run.py -q -v`
Expected: 11 PASS.

- [ ] **Step 5: Delegate `RealRunner.probe` to `run_probe`**

In `scripts/real_runner.py`, replace the whole `probe` method with:

```python
    def probe(self, variant_key: str, force: bool = False) -> ProbeTelemetry:
        # Stage 0 is roughly 130 GPU-hours of sequential work in one process under
        # a 24-hour SLURM wall clock. run_probe persists each probe's telemetry and
        # reuses it on a later invocation, which is what stops a timeout destroying
        # every number measured before it. Stage 0 uses one seed for items and runs.
        return run_probe(
            model_path=self._resolve(variant_key), model_key=variant_key,
            suite_key="countdown", work_dir=self.work_dir,
            config=ProbeConfig(
                probe_steps=self.config.probe_steps,
                num_generations=self.config.num_generations,
                n_probe_items=self.config.n_probe_items,
            ),
            item_seed=self.config.seed, run_seed=self.config.seed, force=force,
        )
```

Add `from amenability.probe.run import ProbeConfig, run_probe` to the imports. Keep the `generate_countdown` import: `__init__` still generates and registers the probe items for the disjointness check. Remove `from amenability.suites.countdown import verify_countdown` only if nothing else in the file uses it (check `full_run`: it uses `verify_gsm8k`, not `verify_countdown`).

- [ ] **Step 6: Update `tests/test_real_runner.py`**

The fixture monkeypatches `real_runner.evaluate_passk` and `real_runner.run_grpo`. Those names are no longer the ones the probe path calls. Change the fixture to patch both modules, so `full_run` (still in `real_runner`) and `probe` (now in `run.py`) both see the fakes:

```python
    import amenability.probe.run as probe_run
    monkeypatch.setattr(real_runner, "evaluate_passk", fake_passk)
    monkeypatch.setattr(real_runner, "run_grpo", fake_grpo)
    monkeypatch.setattr(probe_run, "evaluate_passk", fake_passk)
    monkeypatch.setattr(probe_run, "run_grpo", fake_grpo)
```

and in `test_probe_writes_its_telemetry_to_disk` change the expected path to
`tmp_path / "telemetry" / "qwen2.5-0.5b-oversft1x-countdown-grpo-s0.json"`.

- [ ] **Step 7: Run the full CPU suite**

Run: `uv run pytest -q -m "not gpu"`
Expected: 203 passed. If `tests/test_stage0.py` fails, read the failure: `FakeRunner` there does not go through `RealRunner`, so it should be untouched.

- [ ] **Step 8: Commit**

```bash
git add src/amenability/probe/run.py tests/test_probe_run.py scripts/real_runner.py tests/test_real_runner.py
git commit -m "feat: run_probe as the single probe implementation

Factors pre-eval, GRPO and post-eval out of RealRunner so the transfer
batch and Stage 0 share one protocol. Telemetry cache keys now carry suite
and run seed; item seed is separate from run seed so a replicate sees
identical items; checkpoints are deleted after post-eval."
```

---

### Task 5: Per-job CLI `scripts/run_probe.py`

**Files:**
- Create: `scripts/run_probe.py`
- Test: `tests/test_run_probe_cli.py`

**Interfaces:**
- Consumes: `run_probe`, `ProbeConfig` (Task 4); `load_registry` from `amenability.registry.loader`; `PROBE_SUITES` (Task 3).
- Produces:
  ```python
  def resolve_model(model: str, registry: dict) -> tuple[str, str]   # (model_key, model_path)
  def parse_args(argv: list[str] | None = None) -> argparse.Namespace
  def main(argv: list[str] | None = None, run_probe_fn=run_probe) -> None
  ```
  CLI: `python -m scripts.run_probe --model KEY_OR_PATH --suite {countdown,graphpath} [--run-seed 0] [--item-seed 0] [--work-dir results/transfer] [--pre-only] [--keep-checkpoint] [--force] [--steps 60] [--lr 1e-6]`.
  `--model` is a registry key (resolved to `hf_id`) or an existing local path (key is the directory name). Anything else exits 2 with the list of registry keys. `--steps` and `--lr` exist only for the spec section 8 recalibration and the GPU smoke tests; the batch never passes them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_probe_cli.py
import pytest
from pathlib import Path

from scripts.run_probe import main, parse_args, resolve_model


def test_registry_key_resolves_to_hf_id():
    key, path = resolve_model("qwen2.5-0.5b", {"qwen2.5-0.5b": type("S", (), {"hf_id": "Qwen/Qwen2.5-0.5B"})()})
    assert (key, path) == ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B")


def test_local_path_resolves_to_itself(tmp_path):
    d = tmp_path / "my-ckpt"
    d.mkdir()
    key, path = resolve_model(str(d), {})
    assert key == "my-ckpt" and path == str(d)


def test_unknown_model_fails_before_any_gpu_work(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--model", "nope", "--suite", "countdown"], run_probe_fn=lambda **kw: pytest.fail("ran"))
    assert e.value.code == 2
    assert "qwen2.5-0.5b" in capsys.readouterr().err


def test_unknown_suite_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        parse_args(["--model", "qwen2.5-0.5b", "--suite", "gsm8k"])


def test_arguments_reach_run_probe(tmp_path):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        from amenability.eval.passk import PassKResult
        return PassKResult(ks={1: 0.1, 8: 0.1, 32: 0.2, 64: 0.2}, n_samples=64, n_items=1, per_item_correct={})

    main(["--model", "qwen2.5-0.5b", "--suite", "graphpath", "--run-seed", "1",
          "--work-dir", str(tmp_path), "--pre-only", "--steps", "2", "--lr", "3e-6"], run_probe_fn=fake)
    assert seen["model_key"] == "qwen2.5-0.5b" and seen["model_path"] == "Qwen/Qwen2.5-0.5B"
    assert seen["suite_key"] == "graphpath" and seen["run_seed"] == 1 and seen["item_seed"] == 0
    assert seen["work_dir"] == Path(tmp_path) and seen["pre_only"] is True
    assert seen["config"].probe_steps == 2 and seen["config"].learning_rate == pytest.approx(3e-6)
    assert seen["config"].num_generations == 8 and seen["config"].n_probe_items == 300
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_probe_cli.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement the CLI**

```python
# scripts/run_probe.py
"""One probe job: a (model, suite, run seed) triple. Launcher-agnostic.

A SLURM array task, a Modal function or a shell loop each call this once per job.
Everything about the protocol is fixed in ProbeConfig; --steps and --lr exist only
for the learning-rate recalibration bound in the spec (section 8) and for GPU smoke
tests, and the batch never passes them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amenability.eval.passk import PassKResult
from amenability.probe.run import ProbeConfig, run_probe
from amenability.registry.loader import load_registry
from amenability.suites.catalog import PROBE_SUITES


def resolve_model(model: str, registry: dict) -> tuple[str, str]:
    if model in registry:
        return model, registry[model].hf_id
    p = Path(model)
    if p.exists():
        return p.name, str(p)
    raise KeyError(model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registry key or local checkpoint path")
    parser.add_argument("--suite", required=True, choices=sorted(PROBE_SUITES))
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--item-seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path("results/transfer"))
    parser.add_argument("--pre-only", action="store_true", help="pre-eval only (calibration pilot)")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    parser.add_argument("--steps", type=int, default=ProbeConfig.probe_steps)
    parser.add_argument("--lr", type=float, default=ProbeConfig.learning_rate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, run_probe_fn=run_probe) -> None:
    args = parse_args(argv)
    registry = load_registry()
    try:
        model_key, model_path = resolve_model(args.model, registry)
    except KeyError:
        print(
            f"unknown model {args.model!r}: not a registry key and not an existing path.\n"
            f"registry keys: {', '.join(sorted(registry))}",
            file=sys.stderr,
        )
        sys.exit(2)

    result = run_probe_fn(
        model_path=model_path, model_key=model_key, suite_key=args.suite,
        work_dir=args.work_dir,
        config=ProbeConfig(probe_steps=args.steps, learning_rate=args.lr),
        item_seed=args.item_seed, run_seed=args.run_seed, force=args.force,
        keep_checkpoint=args.keep_checkpoint, pre_only=args.pre_only,
    )
    if isinstance(result, PassKResult):
        print(f"{model_key}/{args.suite} pre: " + " ".join(f"pass@{k}={v:.3f}" for k, v in sorted(result.ks.items())))
    else:
        print(
            f"{model_key}/{args.suite}/s{args.run_seed}: "
            f"pre pass@1={result.pre.ks[1]:.3f} pass@32={result.pre.ks[32]:.3f} | "
            f"post pass@1={result.post.ks[1]:.3f} pass@32={result.post.ks[32]:.3f} | "
            f"{len(result.steps)} steps recorded"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_probe_cli.py -q`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
uv run pytest -q -m "not gpu"
git add scripts/run_probe.py tests/test_run_probe_cli.py
git commit -m "feat: launcher-agnostic per-job probe CLI"
```

---

### Task 6: Job manifest and model prefetch (#20)

**Files:**
- Create: `scripts/make_transfer_jobs.py`
- Create: `scripts/prefetch_models.py`
- Modify: `.gitignore`
- Test: `tests/test_transfer_jobs.py`, `tests/test_prefetch.py`

**Interfaces:**
- Consumes: `load_registry`.
- Produces:
  ```python
  # make_transfer_jobs.py
  ARMS: list[tuple[str, int]] = [("countdown", 0), ("graphpath", 0), ("countdown", 1)]
  def transfer_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]   # arm-major: all models for arm 1, then arm 2, ...
  def pilot_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]      # (model, suite, 0) for both suites
  def write_jobs(jobs, path: Path) -> None   # one line per job: "model suite seed"
  # CLI: python -m scripts.make_transfer_jobs [--pilot] [--only-model KEY ...] [--out results/transfer/jobs.txt]

  # prefetch_models.py
  def prefetch(registry: dict, download=snapshot_download, check_config=AutoConfig.from_pretrained) -> list[dict]
      # each: {"key", "hf_id", "ok": bool, "path": str | None, "error": str | None}
  # CLI: python -m scripts.prefetch_models [--out results/transfer/prefetch.json]; exit 1 if any not ok
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_jobs.py
from pathlib import Path

from scripts.make_transfer_jobs import ARMS, pilot_jobs, transfer_jobs, write_jobs

KEYS = ["a", "b", "c"]


def test_transfer_jobs_cover_three_arms_for_every_model():
    jobs = transfer_jobs(KEYS)
    assert len(jobs) == 9
    assert ARMS == [("countdown", 0), ("graphpath", 0), ("countdown", 1)]
    assert jobs[:3] == [("a", "countdown", 0), ("b", "countdown", 0), ("c", "countdown", 0)]
    assert set(jobs) == {(m, s, seed) for m in KEYS for s, seed in ARMS}


def test_pilot_jobs_are_seed_zero_for_both_suites():
    jobs = pilot_jobs(KEYS)
    assert len(jobs) == 6
    assert all(seed == 0 for _, _, seed in jobs)
    assert {s for _, s, _ in jobs} == {"countdown", "graphpath"}


def test_write_jobs_one_line_each(tmp_path):
    p = tmp_path / "jobs.txt"
    write_jobs([("a", "countdown", 0), ("b", "graphpath", 1)], p)
    assert p.read_text() == "a countdown 0\nb graphpath 1\n"


def test_cli_writes_thirty_jobs_for_the_real_roster(tmp_path):
    from scripts.make_transfer_jobs import main
    out = tmp_path / "jobs.txt"
    main(["--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 30
    assert len({line.split()[0] for line in lines}) == 10


def test_cli_only_model_filters(tmp_path):
    from scripts.make_transfer_jobs import main
    out = tmp_path / "jobs.txt"
    main(["--out", str(out), "--only-model", "qwen2.5-0.5b", "smollm2-1.7b"])
    lines = out.read_text().splitlines()
    assert len(lines) == 6
```

```python
# tests/test_prefetch.py
import json

import pytest

from scripts.prefetch_models import main, prefetch


class _Spec:
    def __init__(self, key, hf_id):
        self.key, self.hf_id = key, hf_id


def test_prefetch_reports_each_model_and_keeps_going_after_a_failure():
    registry = {"ok": _Spec("ok", "org/ok"), "gated": _Spec("gated", "org/gated")}

    def download(repo_id, **kw):
        if repo_id == "org/gated":
            raise RuntimeError("401 Client Error: gated repo")
        return f"/cache/{repo_id}"

    rows = prefetch(registry, download=download, check_config=lambda path: None)
    by_key = {r["key"]: r for r in rows}
    assert by_key["ok"] == {"key": "ok", "hf_id": "org/ok", "ok": True, "path": "/cache/org/ok", "error": None}
    assert by_key["gated"]["ok"] is False and "401" in by_key["gated"]["error"]


def test_prefetch_marks_unloadable_config_as_failure():
    registry = {"m": _Spec("m", "org/m")}

    def check(path):
        raise ValueError("unknown architecture")

    rows = prefetch(registry, download=lambda repo_id, **kw: "/cache/m", check_config=check)
    assert rows[0]["ok"] is False and "unknown architecture" in rows[0]["error"]


def test_cli_exits_nonzero_when_any_model_failed(tmp_path, monkeypatch):
    import scripts.prefetch_models as pm
    monkeypatch.setattr(pm, "load_registry", lambda: {"m": _Spec("m", "org/m")})
    out = tmp_path / "prefetch.json"
    with pytest.raises(SystemExit) as e:
        main(["--out", str(out)], download=lambda repo_id, **kw: (_ for _ in ()).throw(RuntimeError("403")),
             check_config=lambda p: None)
    assert e.value.code == 1
    assert json.loads(out.read_text())[0]["ok"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transfer_jobs.py tests/test_prefetch.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement both scripts**

```python
# scripts/make_transfer_jobs.py
"""Write the transfer batch's job manifest: one line per (model, suite, run seed).

Arm-major order so the SLURM array works through one arm at a time and a partial
batch is still an analysable arm rather than a random third of every arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from amenability.registry.loader import load_registry

ARMS: list[tuple[str, int]] = [("countdown", 0), ("graphpath", 0), ("countdown", 1)]
PILOT_SUITES = ("countdown", "graphpath")


def transfer_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]:
    return [(m, suite, seed) for suite, seed in ARMS for m in model_keys]


def pilot_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]:
    return [(m, suite, 0) for suite in PILOT_SUITES for m in model_keys]


def write_jobs(jobs: list[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{m} {s} {seed}\n" for m, s, seed in jobs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="pre-eval calibration jobs instead of the batch")
    parser.add_argument("--only-model", nargs="+", default=None, help="restrict to these registry keys")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    keys = sorted(load_registry())
    if args.only_model:
        unknown = sorted(set(args.only_model) - set(keys))
        if unknown:
            parser.error(f"unknown model keys: {unknown}; valid: {keys}")
        keys = [k for k in keys if k in set(args.only_model)]
    jobs = pilot_jobs(keys) if args.pilot else transfer_jobs(keys)
    out = args.out or Path("results/transfer") / ("jobs_pilot.txt" if args.pilot else "jobs.txt")
    write_jobs(jobs, out)
    print(f"wrote {len(jobs)} jobs to {out}")


if __name__ == "__main__":
    main()
```

```python
# scripts/prefetch_models.py
"""Download every roster model once and confirm transformers can load its config.

Run on a CPU node before the GPU array so thirty array tasks do not each download
the same weights, and so a gated or missing repository (#20) fails here, in one
place, rather than as ten separate array-task failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amenability.registry.loader import load_registry


def _default_download(repo_id: str, **kw) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken"],
        **kw,
    )


def _default_check_config(path: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(path)
    AutoTokenizer.from_pretrained(path)


def prefetch(registry: dict, download=_default_download, check_config=_default_check_config) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(registry):
        hf_id = registry[key].hf_id
        row = {"key": key, "hf_id": hf_id, "ok": False, "path": None, "error": None}
        try:
            row["path"] = download(hf_id)
            check_config(row["path"])
            row["ok"] = True
        except Exception as e:  # noqa: BLE001 - every failure must be reported, none may abort the sweep
            row["error"] = f"{type(e).__name__}: {e}"
        status = "ok" if row["ok"] else f"FAILED: {row['error']}"
        print(f"{key:20s} {hf_id:40s} {status}")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None, download=_default_download, check_config=_default_check_config) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/transfer/prefetch.json"))
    args = parser.parse_args(argv)
    rows = prefetch(load_registry(), download=download, check_config=check_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    failed = [r for r in rows if not r["ok"]]
    if failed:
        print(
            f"\n{len(failed)} model(s) failed. A 401/403 means the licence has not been "
            "accepted on huggingface.co for the account behind this token, or the token "
            "is not logged in (uv run hf auth login).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nall {len(rows)} models present")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ignore checkpoints and logs**

Append to `.gitignore`:

```
results/**/probes/
logs/
```

Telemetry, pre-eval, meta, manifests and reports under `results/transfer/` are small and ARE committed; only checkpoints are not.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transfer_jobs.py tests/test_prefetch.py -q`
Expected: 8 PASS.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q -m "not gpu"
git add scripts/make_transfer_jobs.py scripts/prefetch_models.py tests/test_transfer_jobs.py tests/test_prefetch.py .gitignore
git commit -m "feat: transfer job manifest and roster prefetch (#20)"
```

---

### Task 7: SLURM launchers

Requires Checkpoint H1's answers: the scratch path and the 80 GB node feature name.

**Files:**
- Create: `scripts/slurm/prefetch.sbatch`
- Create: `scripts/slurm/probe_array.sbatch`
- Create: `scripts/slurm/README.md`

**Interfaces:**
- Consumes: `scripts.run_probe` (Task 5), `scripts.prefetch_models` (Task 6), the job manifest format `model suite seed` (Task 6).
- Produces: `probe_array.sbatch` reads line `SLURM_ARRAY_TASK_ID + 1` of `$JOBS_FILE` (default `results/transfer/jobs.txt`) and runs one job. Environment knobs: `JOBS_FILE`, `WORK_DIR` (default `results/transfer`), `EXTRA_ARGS` (passed through to the CLI, e.g. `--pre-only`), `HF_HOME` (default `$SCRATCH/hf`), `DRY_RUN` (echo the command instead of running it, and skip `module`).

- [ ] **Step 1: Write the array script**

```bash
#!/bin/bash
# scripts/slurm/probe_array.sbatch
# One probe job per array task. Line (SLURM_ARRAY_TASK_ID + 1) of $JOBS_FILE is
# "model suite seed". Launch with, e.g.:
#   sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch
# Add --constraint=<80GB feature from sinfo> if the smoke probe's peak memory
# exceeds 36 GB (see scripts/slurm/README.md).
#SBATCH --job-name=amen-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_%A_%a.out

set -euo pipefail

JOBS_FILE="${JOBS_FILE:-results/transfer/jobs.txt}"
WORK_DIR="${WORK_DIR:-results/transfer}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset: launch with --array}"

line=$(sed -n "$((TASK_ID + 1))p" "$JOBS_FILE")
if [ -z "$line" ]; then
  echo "no job on line $((TASK_ID + 1)) of $JOBS_FILE" >&2
  exit 1
fi
read -r MODEL SUITE SEED <<< "$line"

# Weights live on scratch (Task 6's prefetch put them there) and the array must
# never reach out to the Hub: thirty concurrent downloads on a shared cluster is
# how a batch dies of rate limiting.
export HF_HOME="${HF_HOME:-${SCRATCH:?SCRATCH is unset}/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cmd=(uv run python -m scripts.run_probe --model "$MODEL" --suite "$SUITE" --run-seed "$SEED" --work-dir "$WORK_DIR")
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  cmd+=($EXTRA_ARGS)
fi

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry run] task $TASK_ID: ${cmd[*]}"
  exit 0
fi

module purge
module load cuda/12.4
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "task $TASK_ID on $(hostname): ${cmd[*]}"
"${cmd[@]}"
```

- [ ] **Step 2: Write the prefetch script**

```bash
#!/bin/bash
# scripts/slurm/prefetch.sbatch
# CPU-only: sync the environment and download every roster model to scratch once.
#SBATCH --job-name=amen-prefetch
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prefetch_%j.out

set -euo pipefail
export HF_HOME="${HF_HOME:-${SCRATCH:?SCRATCH is unset}/hf}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "$HF_HOME"

# vllm lives in the "gpu" extra, not the base dependencies, so a GPU run needs
# BOTH extras installed. Syncing here, on the CPU node, means the array tasks
# find a ready environment. Do not "simplify" this back to a single extra.
uv sync --extra dev --extra gpu

uv run python -m scripts.prefetch_models --out results/transfer/prefetch.json
```

- [ ] **Step 3: Write the README with the exact launch sequence**

```markdown
# CARC launch sequence for the transfer batch

All commands from the repository root on a CARC login node. `$SCRATCH` must be set
(it is on CARC login nodes); the plan puts the Hugging Face cache at `$SCRATCH/hf`.

## Once

    export HF_HOME=$SCRATCH/hf
    uv run hf auth login            # the account that accepted the Llama 3.2 and Gemma 3 licences
    mkdir -p logs

## 1. Prefetch (CPU, ~30 min)

    sbatch scripts/slurm/prefetch.sbatch
    # wait; then:
    cat results/transfer/prefetch.json | grep -c '"ok": true'    # must print 10

## 2. Calibration pilot (20 GPU jobs, pre-eval only, ~10 min each)

    uv run python -m scripts.make_transfer_jobs --pilot
    JOBS_FILE=results/transfer/jobs_pilot.txt EXTRA_ARGS=--pre-only \
      sbatch --array=0-19%8 scripts/slurm/probe_array.sbatch
    # results land in results/transfer/preeval/

## 3. Smoke probes (2 GPU jobs)

    uv run python -m scripts.make_transfer_jobs --only-model smollm2-1.7b qwen2.5-0.5b --out results/transfer/jobs_smoke.txt
    JOBS_FILE=results/transfer/jobs_smoke.txt sbatch --array=0,1 scripts/slurm/probe_array.sbatch
    # lines 0 and 1 are the countdown seed-0 jobs for the two models.
    # Then read results/transfer/meta/*.json: peak_memory_bytes and wall_seconds.

## 4. The batch (30 GPU jobs, ~1 h each)

    uv run python -m scripts.make_transfer_jobs
    sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch
    # Add --constraint=<feature> for 80 GB nodes if step 3 showed peak memory > 36 GB.
    # Jobs whose telemetry already exists (the smoke probes) exit in seconds.

## Monitoring and reruns

    squeue -u $USER
    grep -l Traceback logs/probe_*.out
    # rerun failed tasks by id:
    sbatch --array=4,17 scripts/slurm/probe_array.sbatch

## 5. Analysis (CPU, seconds)

    uv run python -m scripts.analyze_transfer --work-dir results/transfer
    # writes results/transfer/report.md, report.json, leaderboard_preview.md

## Dry run (any machine, no GPU)

    SLURM_ARRAY_TASK_ID=3 DRY_RUN=1 SCRATCH=/tmp bash scripts/slurm/probe_array.sbatch
```

- [ ] **Step 4: Syntax-check and dry-run the array script**

Run:
```bash
bash -n scripts/slurm/probe_array.sbatch && bash -n scripts/slurm/prefetch.sbatch && echo syntax ok
uv run python -m scripts.make_transfer_jobs --out /tmp/jobs.txt
SLURM_ARRAY_TASK_ID=12 DRY_RUN=1 SCRATCH=/tmp JOBS_FILE=/tmp/jobs.txt bash scripts/slurm/probe_array.sbatch
SLURM_ARRAY_TASK_ID=99 DRY_RUN=1 SCRATCH=/tmp JOBS_FILE=/tmp/jobs.txt bash scripts/slurm/probe_array.sbatch; echo "exit=$?"
```
Expected: `syntax ok`; the dry run of task 12 prints `[dry run] task 12: uv run python -m scripts.run_probe --model <third model of the graphpath arm> --suite graphpath --run-seed 0 --work-dir results/transfer`; task 99 prints `no job on line 100` and `exit=1`.

- [ ] **Step 5: Commit**

```bash
git add scripts/slurm/probe_array.sbatch scripts/slurm/prefetch.sbatch scripts/slurm/README.md
git commit -m "feat: SLURM job array and prefetch launchers for the transfer batch"
```

---

### Task 8: Pre-registration and rulings log

Must be committed **before** any job in Task 10 step 4 (the batch) is launched. The pilot and smoke runs may precede it; they produce calibration data, not transfer data.

**Files:**
- Create: `prereg/transfer.md`
- Create: `docs/decisions/transfer-rulings.md`

- [ ] **Step 1: Write `prereg/transfer.md`**

```markdown
# Probe Transfer Pre-Registration

Committed before the transfer batch runs. Not hash-frozen: the freeze mechanism
belongs to Stage 0, and the product reframe (issue #34) replaces it with metric
versioning. The git history of this file is the audit trail; the batch's telemetry
commits must postdate the commit that adds this file.

## Question

Q1, probe invariance: does a model's probe score depend on which probe suite
measures it? Ten roster models, two suites (Countdown, Graph path), one seed
replicate on Countdown. No ground truth is involved.

## Fixed quantities

- Protocol: the Stage 0 probe protocol, unchanged. 60 GRPO steps, 8 generations,
  4 prompts per optimiser step, lr 1e-6, beta 0.04, temperature 1.0, 300 items in
  three buckets, pass@k at (1, 8, 32, 64) from 64 samples.
- Score: `amenability_score` from `src/amenability/scoring/score.py`, applied per
  suite across the ten roster models.
- Items: generated from item seed 0 for every arm. Run seed 0 for the two suite
  arms, run seed 1 for the Countdown replicate.
- Statistics: `spearman_with_ci` (percentile bootstrap, permutation p, 10,000
  resamples) and `loo_spearman` from `src/amenability/eval/stats.py`.

## Statistics reported

| Statistic | Between | Purpose |
|---|---|---|
| Transfer rho | score(Countdown, s0), score(Graph, s0) | The answer |
| Test-retest rho | score(Countdown, s0), score(Countdown, s1) | Reliability ceiling |
| Per-feature transfer rho | conversion_rate, retention_factor, zero_advantage_rate across suites | Which feature is domain-sensitive |
| Static comparator rho | pre-probe pass@32 across suites | Does the dynamics signal transfer beyond raw capability? |
| Leave-one-out transfer rho | as transfer rho, each model held out | No single model carries it |

## Decision rule

Evaluated in this order; the first matching row is the decision.

| Condition | Decision |
|---|---|
| Fewer than 8 of 10 models have both seed-0 arms scored | INCONCLUSIVE_INCOMPLETE: rerun failures |
| Test-retest rho < 0.5 | INCONCLUSIVE_RELIABILITY: the probe is too noisy at N=10; change the protocol, do not read the transfer rho |
| Transfer rho >= 0.7 and permutation p < 0.05 | INVARIANT: single score defensible; Stage 0 proceeds as designed |
| Transfer rho >= 0.4 | PARTIAL: composite plus per-domain columns; Stage 0 proceeds |
| Otherwise | DOMAIN_SPECIFIC: per-domain product; #28 and Stage 0 re-planned per domain before further GPU spend |

The transfer rho is always reported beside the test-retest rho, never bare.

## The one permitted protocol change

If the Qwen2.5-0.5B Countdown smoke probe shows post pass@1 <= pre pass@1, lr 3e-6
and 5e-6 are tried on that one model only, the smallest lr showing gain is pinned
for every model and both suites, and the change is recorded as a ruling with the
three measured gains. Bound here so it cannot become a post-hoc search after the
batch is seen. No other protocol parameter changes.

## What will not happen

- No change to the score formula, feature definitions or protocol after seeing
  any batch result, other than the lr rule above, which is decided before the batch.
- No choosing between Countdown and Graph as "the" probe by which gives the nicer
  leaderboard.
- No dropping a model from the analysis except for a named, documented job failure
  or a documented no-breadth exclusion, both listed in the report.
- No reading the transfer rho without the test-retest rho beside it.
```

- [ ] **Step 2: Write `docs/decisions/transfer-rulings.md`**

```markdown
# Probe Transfer Execution Rulings

Decisions taken during execution of
`docs/superpowers/plans/2026-09-27-probe-transfer.md`, in the order they were
made, in the format of `stage0-rulings.md`: what was decided, why, and the cost if
the decision turns out to be wrong. Numbered T1, T2, ... to keep them distinct from
the Stage 0 rulings R1 to R38.

**T1. Graph path buckets are (8, 10, 12) nodes with n // 3 extra edges and minimum
distance 3, not the spec's first draft of (6, 8, 10).**
A design-time simulation of an informed guesser (starts at the source, ends at the
target, random distinct middle) gave pass@32 of 0.94 / 0.62 / 0.34 at (6, 8, 10)
with n // 2 extra edges and distance >= 2, so the easy bucket's breadth would have
been mostly luck. At (8, 10, 12), n // 3, distance >= 3 the guesser scores
0.20 / 0.06 / 0.01, and `test_informed_guesser_stays_well_below_saturation` holds
the bound. Made before any model was run.
*Cost if wrong:* the suite may be too hard at 0.5B, in which case the calibration
pilot shows pass@32 near zero on every bucket and the bound contingency is
(7, 9, 11), where the easiest bucket guesses at 0.37.
```

- [ ] **Step 3: Commit**

```bash
git add prereg/transfer.md docs/decisions/transfer-rulings.md
git commit -m "docs: pre-register the probe transfer decision rule; rulings log T1"
```

---

### Task 9: Transfer analysis

**Files:**
- Create: `scripts/analyze_transfer.py`
- Test: `tests/test_analyze_transfer.py`

**Interfaces:**
- Consumes: `telemetry_path`, `preeval_path` (Task 4); `extract_features`, `FeatureVector`, `ProbeTelemetry` from `amenability.probe.telemetry`; `amenability_score` from `amenability.scoring.score`; `spearman_with_ci`, `loo_spearman`, `CorrelationResult` from `amenability.eval.stats`; `load_registry`; `get_probe_suite` (Task 3).
- Produces:
  ```python
  ARMS = {"countdown_s0": ("countdown", 0), "graphpath_s0": ("graphpath", 0), "countdown_s1": ("countdown", 1)}
  INVARIANT_RHO = 0.7; PARTIAL_RHO = 0.4; P_THRESHOLD = 0.05; RETEST_MIN = 0.5; MIN_COMPLETE = 8
  DECISIONS = ("INVARIANT", "PARTIAL", "DOMAIN_SPECIFIC", "INCONCLUSIVE_RELIABILITY", "INCONCLUSIVE_INCOMPLETE")

  def load_arm(work_dir: Path, model_keys: list[str], suite_key: str, run_seed: int) -> tuple[dict[str, ProbeTelemetry], dict[str, str]]
      # (loaded by model, failures by model -> "missing" | "unreadable: <err>")
  def arm_features(telemetries: dict[str, ProbeTelemetry]) -> tuple[dict[str, FeatureVector], dict[str, str]]
      # (features by model, excluded by model -> the ValueError text from extract_features)
  def arm_scores(features: dict[str, FeatureVector]) -> dict[str, float]
      # amenability_score over models in sorted key order
  def paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], list[float], list[float]]
      # sorted common keys, and the two aligned value lists
  def correlate(a: dict[str, float], b: dict[str, float], n_boot: int) -> CorrelationResult | None
      # None when fewer than 3 common keys
  def decide(transfer: CorrelationResult | None, retest: CorrelationResult | None, n_complete: int) -> str
  def transfer_stats(work_dir: Path, model_keys: list[str], n_boot: int = 10000) -> dict
      # everything render_report needs; JSON-serialisable
  def calibration_table(work_dir: Path, model_keys: list[str]) -> list[dict]
      # one row per (model, suite) from preeval files: {"model", "suite", "pass@1", "pass@32", "by_bucket": {...}, "saturated": bool, "floored": bool, "breadth_saturated": bool}
      # floored: pass@32 < 0.02 in every bucket; saturated: pass@1 > 0.95 in every bucket;
      # breadth_saturated: pass@32 > 0.95 in every bucket (transfer-rulings T4)
  def render_report(stats: dict, calibration: list[dict]) -> str
  def render_leaderboard(stats: dict) -> str
  def main(argv=None) -> None   # --work-dir, --n-boot; writes report.md, report.json, leaderboard_preview.md under work_dir
  ```

**`decide` in order:** `n_complete < MIN_COMPLETE` -> `INCONCLUSIVE_INCOMPLETE`; `retest is None or retest.rho < RETEST_MIN` -> `INCONCLUSIVE_RELIABILITY`; `transfer.rho >= INVARIANT_RHO and transfer.p_value < P_THRESHOLD` -> `INVARIANT`; `transfer.rho >= PARTIAL_RHO` -> `PARTIAL`; else `DOMAIN_SPECIFIC`. `n_complete` is the number of models with a score in BOTH seed-0 arms after failures and breadth exclusions.

**`breadth_saturated` (transfer-rulings T4):** pre-probe pass@32 > 0.95 in every bucket. It is a field of every `calibration_table` row and a column in the report's calibration table, after `saturated`. The code blocks below do not show it yet; the Task 9 implementer adds it beside `floored` and `saturated`, with a test.

**`transfer_stats` returns** `{"n_models": 10, "arms": {arm: {"loaded": [...], "failures": {...}, "excluded": {...}, "scores": {...}, "features": {model: asdict(fv)}}}, "n_complete": int, "transfer": asdict(CorrelationResult) | None, "retest": ..., "per_feature": {feature: asdict | None}, "static_pass32": asdict | None, "loo_transfer": [floats] | None, "decision": str}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analyze_transfer.py
"""Synthetic telemetry with controlled conversion rates, so the expected rho is known."""
import json
from pathlib import Path

import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.run import preeval_path, telemetry_path
from amenability.probe.telemetry import ProbeTelemetry, StepRecord
from scripts.analyze_transfer import (
    DECISIONS, MIN_COMPLETE, arm_features, arm_scores, calibration_table, decide,
    load_arm, main, render_report, transfer_stats,
)

MODELS = [f"m{i}" for i in range(10)]


def _telemetry(model: str, conv: float, breadth: float = 0.4) -> ProbeTelemetry:
    """conversion_rate == conv exactly: pre pass@1 0.1, pass@32 0.1 + breadth, post pass@1 = 0.1 + conv * breadth."""
    steps = [
        StepRecord(step=i, mean_reward=0.1 + 0.01 * i, group_reward_std=0.2,
                   zero_advantage_frac=0.1, policy_entropy=2.0, kl=0.01, grad_norm=1.0)
        for i in range(1, 6)
    ]
    pre = PassKResult(ks={1: 0.1, 8: 0.2, 32: 0.1 + breadth, 64: 0.1 + breadth}, n_samples=64,
                      n_items=3, per_item_correct={"a": 1, "b": 0, "c": 0})
    post = PassKResult(ks={1: 0.1 + conv * breadth, 8: 0.2, 32: 0.1 + breadth, 64: 0.1 + breadth},
                       n_samples=64, n_items=3, per_item_correct={"a": 1, "b": 0, "c": 0})
    return ProbeTelemetry(model_key=model, algorithm="grpo", steps=steps, pre=pre, post=post)


def _write(work_dir: Path, model: str, suite: str, seed: int, conv: float, breadth: float = 0.4):
    p = telemetry_path(work_dir, model, suite, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_telemetry(model, conv, breadth).to_json())


def _populate(work_dir: Path, graph_order: list[str], retest_order: list[str] | None = None):
    """Countdown s0 conversion rises with model index; Graph and the replicate follow the given orders."""
    for i, m in enumerate(MODELS):
        _write(work_dir, m, "countdown", 0, conv=0.1 + 0.08 * i)
    for i, m in enumerate(graph_order):
        _write(work_dir, m, "graphpath", 0, conv=0.1 + 0.08 * i)
    for i, m in enumerate(retest_order or MODELS):
        _write(work_dir, m, "countdown", 1, conv=0.1 + 0.08 * i)


def test_load_arm_reports_missing_and_unreadable_files(tmp_path):
    _write(tmp_path, "m0", "countdown", 0, 0.5)
    bad = telemetry_path(tmp_path, "m1", "countdown", 0)
    bad.write_text("{not json")
    loaded, failures = load_arm(tmp_path, ["m0", "m1", "m2"], "countdown", 0)
    assert list(loaded) == ["m0"]
    assert failures["m2"] == "missing"
    assert failures["m1"].startswith("unreadable")


def test_arm_features_excludes_models_with_no_breadth_by_name(tmp_path):
    tels = {"ok": _telemetry("ok", 0.5), "flat": _telemetry("flat", 0.0, breadth=0.0)}
    feats, excluded = arm_features(tels)
    assert list(feats) == ["ok"]
    assert "flat" in excluded and "breadth" in excluded["flat"]


def test_arm_scores_are_relative_within_the_arm():
    feats, _ = arm_features({m: _telemetry(m, 0.1 * i) for i, m in enumerate(MODELS[:4])})
    scores = arm_scores(feats)
    assert list(scores) == sorted(feats)
    assert abs(sum(scores.values())) < 1e-9


def test_identical_orderings_are_invariant(tmp_path):
    _populate(tmp_path, graph_order=MODELS)
    stats = transfer_stats(tmp_path, MODELS, n_boot=500)
    assert stats["n_complete"] == 10
    assert stats["transfer"]["rho"] == pytest.approx(1.0)
    assert stats["retest"]["rho"] == pytest.approx(1.0)
    assert stats["decision"] == "INVARIANT"
    assert set(stats["per_feature"]) == {"conversion_rate", "retention_factor", "zero_advantage_rate"}
    assert len(stats["loo_transfer"]) == 10


def test_reversed_graph_ordering_is_domain_specific(tmp_path):
    _populate(tmp_path, graph_order=MODELS[::-1])
    stats = transfer_stats(tmp_path, MODELS, n_boot=500)
    assert stats["transfer"]["rho"] == pytest.approx(-1.0)
    assert stats["decision"] == "DOMAIN_SPECIFIC"


def test_noisy_replicate_is_inconclusive_even_if_transfer_looks_perfect(tmp_path):
    _populate(tmp_path, graph_order=MODELS, retest_order=MODELS[::-1])
    stats = transfer_stats(tmp_path, MODELS, n_boot=500)
    assert stats["transfer"]["rho"] == pytest.approx(1.0)
    assert stats["decision"] == "INCONCLUSIVE_RELIABILITY"


def test_three_missing_graph_jobs_are_inconclusive_and_named(tmp_path):
    _populate(tmp_path, graph_order=MODELS[:7])
    stats = transfer_stats(tmp_path, MODELS, n_boot=500)
    assert stats["n_complete"] == 7
    assert stats["decision"] == "INCONCLUSIVE_INCOMPLETE"
    assert set(stats["arms"]["graphpath_s0"]["failures"]) == {"m7", "m8", "m9"}
    report = render_report(stats, [])
    assert "m7" in report and "INCONCLUSIVE_INCOMPLETE" in report


def test_no_breadth_exclusion_reduces_n_and_is_named(tmp_path):
    _populate(tmp_path, graph_order=MODELS)
    _write(tmp_path, "m3", "graphpath", 0, conv=0.0, breadth=0.0)
    stats = transfer_stats(tmp_path, MODELS, n_boot=500)
    assert stats["n_complete"] == 9
    assert "m3" in stats["arms"]["graphpath_s0"]["excluded"]
    assert "m3" in render_report(stats, [])


def test_decide_order():
    from amenability.eval.stats import CorrelationResult as C
    good = C(rho=0.9, ci_low=0.5, ci_high=1.0, p_value=0.001, n=10)
    mid = C(rho=0.5, ci_low=0.0, ci_high=0.9, p_value=0.2, n=10)
    low = C(rho=0.1, ci_low=-0.5, ci_high=0.6, p_value=0.8, n=10)
    assert decide(good, good, 7) == "INCONCLUSIVE_INCOMPLETE"
    assert decide(good, low, 10) == "INCONCLUSIVE_RELIABILITY"
    assert decide(good, None, 10) == "INCONCLUSIVE_RELIABILITY"
    assert decide(good, good, 10) == "INVARIANT"
    assert decide(mid, good, 10) == "PARTIAL"
    assert decide(low, good, 10) == "DOMAIN_SPECIFIC"
    assert decide(C(rho=0.75, ci_low=0, ci_high=1, p_value=0.2, n=10), good, 10) == "PARTIAL"
    assert MIN_COMPLETE == 8 and len(DECISIONS) == 5


def test_calibration_table_flags_floored_and_saturated(tmp_path):
    def write_pre(model, suite, by_bucket):
        p = preeval_path(tmp_path, model, suite)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model_key": model, "suite_key": suite, "item_seed": 0, "n_items": 300,
                                 "pre": {"ks": {"1": 0.1, "32": 0.3}, "n_samples": 64, "n_items": 300, "per_item_correct": {}},
                                 "by_bucket": by_bucket}))
    write_pre("m0", "graphpath", {"8": {"1": 0.0, "32": 0.01}, "10": {"1": 0.0, "32": 0.0}, "12": {"1": 0.0, "32": 0.0}})
    write_pre("m0", "countdown", {"3": {"1": 0.97, "32": 1.0}, "4": {"1": 0.96, "32": 1.0}, "5": {"1": 0.99, "32": 1.0}})
    rows = {(r["model"], r["suite"]): r for r in calibration_table(tmp_path, ["m0", "m1"])}
    assert rows[("m0", "graphpath")]["floored"] is True and rows[("m0", "graphpath")]["saturated"] is False
    assert rows[("m0", "countdown")]["saturated"] is True and rows[("m0", "countdown")]["floored"] is False
    assert rows[("m1", "countdown")]["pass@1"] is None   # missing pre-eval is a row, not a crash


def test_main_writes_the_three_outputs(tmp_path, monkeypatch):
    _populate(tmp_path, graph_order=MODELS)
    import scripts.analyze_transfer as at
    monkeypatch.setattr(at, "load_registry", lambda: {m: None for m in MODELS})
    main(["--work-dir", str(tmp_path), "--n-boot", "200"])
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "report.json").read_text())["decision"] == "INVARIANT"
    lb = (tmp_path / "leaderboard_preview.md").read_text()
    assert "| model |" in lb and "m9" in lb
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analyze_transfer.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement the analysis**

```python
# scripts/analyze_transfer.py
"""Probe transfer analysis: does a model's probe score survive a change of probe suite?

Implements prereg/transfer.md. Every statistic prints its N; failed jobs and
no-breadth exclusions are named, never dropped silently; the transfer rho is never
shown without the test-retest rho beside it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from amenability.eval.stats import CorrelationResult, loo_spearman, spearman_with_ci
from amenability.probe.run import preeval_path, telemetry_path
from amenability.probe.telemetry import FeatureVector, ProbeTelemetry, extract_features
from amenability.registry.loader import load_registry
from amenability.scoring.score import amenability_score

ARMS: dict[str, tuple[str, int]] = {
    "countdown_s0": ("countdown", 0),
    "graphpath_s0": ("graphpath", 0),
    "countdown_s1": ("countdown", 1),
}
SCORED_FEATURES = ("conversion_rate", "retention_factor", "zero_advantage_rate")

INVARIANT_RHO = 0.7
PARTIAL_RHO = 0.4
P_THRESHOLD = 0.05
RETEST_MIN = 0.5
MIN_COMPLETE = 8
DECISIONS = ("INVARIANT", "PARTIAL", "DOMAIN_SPECIFIC", "INCONCLUSIVE_RELIABILITY", "INCONCLUSIVE_INCOMPLETE")

FLOOR_PASS32 = 0.02
SATURATION_PASS1 = 0.95


def load_arm(
    work_dir: Path, model_keys: list[str], suite_key: str, run_seed: int
) -> tuple[dict[str, ProbeTelemetry], dict[str, str]]:
    loaded: dict[str, ProbeTelemetry] = {}
    failures: dict[str, str] = {}
    for m in model_keys:
        p = telemetry_path(work_dir, m, suite_key, run_seed)
        if not p.exists():
            failures[m] = "missing"
            continue
        try:
            loaded[m] = ProbeTelemetry.from_json(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001 - a corrupt file is a named failure, not a crash
            failures[m] = f"unreadable: {type(e).__name__}: {e}"
    return loaded, failures


def arm_features(telemetries: dict[str, ProbeTelemetry]) -> tuple[dict[str, FeatureVector], dict[str, str]]:
    feats: dict[str, FeatureVector] = {}
    excluded: dict[str, str] = {}
    for m, t in telemetries.items():
        try:
            feats[m] = extract_features(t)
        except ValueError as e:
            excluded[m] = str(e)
    return feats, excluded


def arm_scores(features: dict[str, FeatureVector]) -> dict[str, float]:
    keys = sorted(features)
    if not keys:
        return {}
    return dict(zip(keys, amenability_score([features[k] for k in keys])))


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], list[float], list[float]]:
    keys = sorted(set(a) & set(b))
    return keys, [a[k] for k in keys], [b[k] for k in keys]


def correlate(a: dict[str, float], b: dict[str, float], n_boot: int) -> CorrelationResult | None:
    keys, x, y = paired(a, b)
    if len(keys) < 3:
        return None
    return spearman_with_ci(x, y, n_boot=n_boot)


def decide(transfer: CorrelationResult | None, retest: CorrelationResult | None, n_complete: int) -> str:
    if n_complete < MIN_COMPLETE or transfer is None:
        return "INCONCLUSIVE_INCOMPLETE"
    if retest is None or retest.rho < RETEST_MIN:
        return "INCONCLUSIVE_RELIABILITY"
    if transfer.rho >= INVARIANT_RHO and transfer.p_value < P_THRESHOLD:
        return "INVARIANT"
    if transfer.rho >= PARTIAL_RHO:
        return "PARTIAL"
    return "DOMAIN_SPECIFIC"


def _opt(r: CorrelationResult | None) -> dict | None:
    return None if r is None else asdict(r)


def transfer_stats(work_dir: Path, model_keys: list[str], n_boot: int = 10000) -> dict:
    arms: dict[str, dict] = {}
    scores: dict[str, dict[str, float]] = {}
    feats: dict[str, dict[str, FeatureVector]] = {}
    tels: dict[str, dict[str, ProbeTelemetry]] = {}
    for arm, (suite, seed) in ARMS.items():
        loaded, failures = load_arm(work_dir, model_keys, suite, seed)
        f, excluded = arm_features(loaded)
        s = arm_scores(f)
        tels[arm], feats[arm], scores[arm] = loaded, f, s
        arms[arm] = {
            "suite": suite, "run_seed": seed, "loaded": sorted(loaded),
            "failures": failures, "excluded": excluded, "scores": s,
            "features": {m: asdict(v) for m, v in f.items()},
        }

    cd0, gp0, cd1 = scores["countdown_s0"], scores["graphpath_s0"], scores["countdown_s1"]
    common, x, y = paired(cd0, gp0)
    transfer = correlate(cd0, gp0, n_boot)
    retest = correlate(cd0, cd1, n_boot)
    per_feature = {
        name: _opt(correlate(
            {m: getattr(v, name) for m, v in feats["countdown_s0"].items()},
            {m: getattr(v, name) for m, v in feats["graphpath_s0"].items()}, n_boot))
        for name in SCORED_FEATURES
    }
    static = _opt(correlate(
        {m: t.pre.ks[32] for m, t in tels["countdown_s0"].items()},
        {m: t.pre.ks[32] for m, t in tels["graphpath_s0"].items()}, n_boot))
    loo = loo_spearman(x, y) if len(common) >= 4 else None
    return {
        "n_models": len(model_keys), "arms": arms, "n_complete": len(common),
        "complete_models": common,
        "transfer": _opt(transfer), "retest": _opt(retest), "per_feature": per_feature,
        "static_pass32": static, "loo_transfer": loo,
        "decision": decide(transfer, retest, len(common)),
        "thresholds": {"invariant_rho": INVARIANT_RHO, "partial_rho": PARTIAL_RHO,
                       "p": P_THRESHOLD, "retest_min": RETEST_MIN, "min_complete": MIN_COMPLETE},
    }


def calibration_table(work_dir: Path, model_keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    suites = sorted({suite for suite, _ in ARMS.values()})
    for m in model_keys:
        for suite in suites:
            p = preeval_path(work_dir, m, suite)
            row = {"model": m, "suite": suite, "pass@1": None, "pass@32": None,
                   "by_bucket": None, "floored": None, "saturated": None}
            if p.exists():
                d = json.loads(p.read_text())
                by_bucket = {int(b): {int(k): v for k, v in ks.items()} for b, ks in d["by_bucket"].items()}
                row.update({
                    "pass@1": d["pre"]["ks"]["1"], "pass@32": d["pre"]["ks"]["32"],
                    "by_bucket": by_bucket,
                    "floored": all(ks[32] < FLOOR_PASS32 for ks in by_bucket.values()),
                    "saturated": all(ks[1] > SATURATION_PASS1 for ks in by_bucket.values()),
                })
            rows.append(row)
    return rows


def _fmt(r: dict | None) -> str:
    if r is None:
        return "n/a"
    return f"rho={r['rho']:.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] p={r['p_value']:.3f} n={r['n']}"


def render_report(stats: dict, calibration: list[dict]) -> str:
    lines = ["# Probe Transfer Report", "", f"**Decision: {stats['decision']}** "
             f"(n_complete={stats['n_complete']} of {stats['n_models']})", ""]
    lines += ["| statistic | value |", "|---|---|",
              f"| transfer rho (Countdown s0 vs Graph s0) | {_fmt(stats['transfer'])} |",
              f"| test-retest rho (Countdown s0 vs s1) | {_fmt(stats['retest'])} |",
              f"| static comparator (pre pass@32 across suites) | {_fmt(stats['static_pass32'])} |"]
    for name, r in stats["per_feature"].items():
        lines.append(f"| per-feature: {name} | {_fmt(r)} |")
    if stats["loo_transfer"]:
        lo, hi = min(stats["loo_transfer"]), max(stats["loo_transfer"])
        lines.append(f"| leave-one-out transfer rho range | {lo:.3f} to {hi:.3f} |")
    lines += ["", "## Arms", ""]
    for arm, a in stats["arms"].items():
        lines.append(f"### {arm} ({a['suite']}, seed {a['run_seed']}): {len(a['loaded'])} loaded")
        for m, why in sorted(a["failures"].items()):
            lines.append(f"- FAILED {m}: {why}")
        for m, why in sorted(a["excluded"].items()):
            lines.append(f"- EXCLUDED {m}: {why}")
        lines.append("")
    if calibration:
        lines += ["## Calibration (pre-probe pass@k by bucket)", "",
                  "| model | suite | pass@1 | pass@32 | floored | saturated | by bucket |", "|---|---|---|---|---|---|---|"]
        for r in calibration:
            if r["pass@1"] is None:
                lines.append(f"| {r['model']} | {r['suite']} | missing | | | | |")
                continue
            buckets = "; ".join(f"{b}: {ks[1]:.2f}/{ks[32]:.2f}" for b, ks in sorted(r["by_bucket"].items()))
            lines.append(f"| {r['model']} | {r['suite']} | {r['pass@1']:.3f} | {r['pass@32']:.3f} | "
                         f"{r['floored']} | {r['saturated']} | {buckets} |")
    lines += ["", "Thresholds: " + json.dumps(stats["thresholds"])]
    return "\n".join(lines) + "\n"


def render_leaderboard(stats: dict) -> str:
    cd0, gp0 = stats["arms"]["countdown_s0"], stats["arms"]["graphpath_s0"]
    models = sorted(set(cd0["scores"]) | set(gp0["scores"]),
                    key=lambda m: -(cd0["scores"].get(m, float("-inf"))))
    lines = ["# Leaderboard preview (probe-only, no ground truth, scores relative within arm)", "",
             "| model | score (Countdown) | score (Graph) | conversion (C) | conversion (G) | zero-adv (C) | zero-adv (G) |",
             "|---|---|---|---|---|---|---|"]
    for m in models:
        fc, fg = cd0["features"].get(m), gp0["features"].get(m)
        lines.append(f"| {m} | {_score_cell(cd0['scores'], m)} | {_score_cell(gp0['scores'], m)} "
                     f"| {_feature_cell(fc, 'conversion_rate')} | {_feature_cell(fg, 'conversion_rate')} "
                     f"| {_feature_cell(fc, 'zero_advantage_rate')} | {_feature_cell(fg, 'zero_advantage_rate')} |")
    return "\n".join(lines) + "\n"


def _score_cell(scores: dict[str, float], m: str) -> str:
    return f"{scores[m]:.3f}" if m in scores else "n/a"


def _feature_cell(features: dict | None, name: str) -> str:
    return "n/a" if features is None else f"{features[name]:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("results/transfer"))
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    model_keys = sorted(load_registry())
    stats = transfer_stats(args.work_dir, model_keys, n_boot=args.n_boot)
    calibration = calibration_table(args.work_dir, model_keys)
    (args.work_dir / "report.json").write_text(json.dumps({**stats, "calibration": calibration}, indent=2, sort_keys=True))
    (args.work_dir / "report.md").write_text(render_report(stats, calibration))
    (args.work_dir / "leaderboard_preview.md").write_text(render_leaderboard(stats))
    print(render_report(stats, calibration))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analyze_transfer.py -q -v`
Expected: 11 PASS. Bootstrap with `n_boot=500` on 10 points takes well under a second per call.

- [ ] **Step 5: Commit**

```bash
uv run pytest -q -m "not gpu"
git add scripts/analyze_transfer.py tests/test_analyze_transfer.py
git commit -m "feat: probe transfer analysis with pre-registered decision rule"
```

---

### Task 10: Run it

This task is a procedure, not code. Each step's output goes into the tree or into the tracking issue. The executing session stops at every "STOP" line and asks the owner.

- [ ] **Step 1: Sync and verify on CARC**

```bash
git pull
uv sync --extra dev --extra gpu
uv run pytest -q -m "not gpu"       # 227 passed expected
```

- [ ] **Step 2: Prefetch (resolves #20)**

Follow `scripts/slurm/README.md` step 1. Requires Checkpoint H2. Expected: `prefetch.json` has 10 rows with `"ok": true`. Commit `results/transfer/prefetch.json`. Close #20 with a comment naming the ten resolved `hf_id`s.

If any row fails with 401/403: STOP, tell the owner which licence to accept.
If any row fails with an unsupported-architecture error: STOP, report the exact error. This is a transformers 5.17 compatibility problem and needs a ruling.

- [ ] **Step 3: Calibration pilot (resolves #21)**

Follow README step 2. When all 20 `preeval/*.json` exist, run:

```bash
uv run python -m scripts.analyze_transfer --work-dir results/transfer
```

Read the calibration table. Acceptance, per (model, suite): `floored` is False and `saturated` is False, and Graph is not `breadth_saturated` for 3 or more models.

- If Countdown is floored for any model: #21's concern is real. STOP and report; the fix is a bucket change in `countdown.py` and a ruling, and both suites' pilot must be rerun for that suite.
- If Graph is floored for 3 or more models: apply the bound contingency from spec section 4, buckets `(7, 9, 11)`, as the next free T number in `docs/decisions/transfer-rulings.md`, rerun `test_graphpath.py` (the guessability test must still pass), and rerun the Graph pilot with `--force`.
- If Graph is floored for 1 or 2 models: proceed; those models will be excluded by name if they have no breadth, and the decision rule already accounts for that.
- If Graph is breadth-saturated (pre-probe pass@32 > 0.95 in every bucket) for 3 or more models: STOP and rule (transfer-rulings T4); candidate remedy is larger buckets, rerun the Graph pilot with `--force` after the change.

Commit `results/transfer/preeval/` and the report. Comment on #21 with the calibration table and close it if nothing was floored.

- [ ] **Step 4: Smoke probes (#33 and spec section 8)**

Follow README step 3. Then read `results/transfer/meta/smollm2-1.7b-countdown-grpo-s0.json` and `results/transfer/meta/qwen2.5-0.5b-countdown-grpo-s0.json`.

Checks:
1. Both jobs completed with 60 step records and no `Traceback` in their logs. If the SmolLM2 job died at post-eval with a CUDA OOM: #33's fix is insufficient in practice. STOP and report; the contingency is to run pre-eval, training and post-eval as three subprocesses inside `run_probe`, recorded as a ruling.
2. `peak_memory_bytes` for SmolLM2-1.7B. If above 36e9: the batch needs 80 GB nodes; add `--constraint=<feature from H1>` to the batch launch. Record which in the tracking issue.
3. `wall_seconds` total per job. Multiply by 30 for the batch estimate. If a job took more than 2 hours, STOP and report before launching the batch.
4. Qwen2.5-0.5B: `post.ks[1] > pre.ks[1]` in its telemetry. If not, spec section 8 applies: run
   ```bash
   uv run python -m scripts.run_probe --model qwen2.5-0.5b --suite countdown --work-dir results/transfer/lr_pilot --lr 3e-6
   uv run python -m scripts.run_probe --model qwen2.5-0.5b --suite countdown --work-dir results/transfer/lr_pilot --lr 5e-6
   ```
   (as two array tasks or two `srun`s), pin the smallest lr showing gain by changing `ProbeConfig.learning_rate`'s default AND `RealRunner`'s expectations AND `prereg/stage0.md`'s "Fixed quantities" line, record a ruling as the next free T number in `docs/decisions/transfer-rulings.md` with the three measured gains, rerun the seed-0 Countdown smoke for both models with `--force`, and only then continue.

Close #33 with the SmolLM2 meta file's numbers as evidence.

- [ ] **Step 5: STOP: batch approval**

Present to the owner: pilot table summary, both smoke metas (peak memory, wall), the GPU-hour estimate (30 x measured wall), whether 80 GB nodes are needed, and confirmation that `prereg/transfer.md` is committed (`git log --oneline -- prereg/transfer.md`). Wait for an explicit yes.

- [ ] **Step 6: Launch the batch**

Follow README step 4. Monitor with README's commands. Rerun failed task IDs. A task that fails twice with the same traceback is a defect: STOP and report it rather than rerunning a third time.

- [ ] **Step 7: Analyse and commit results**

```bash
uv run python -m scripts.analyze_transfer --work-dir results/transfer
git add results/transfer/telemetry results/transfer/meta results/transfer/manifests results/transfer/preeval results/transfer/report.md results/transfer/report.json results/transfer/leaderboard_preview.md
git commit -m "results: probe transfer batch, 10 models x 3 arms"
```

The decision line of `report.md` is the deliverable. Post the whole report as a comment on the tracking issue, and comment on #34 with the decision and a link.

- [ ] **Step 8: Record what the decision means for the board**

Per `prereg/transfer.md`:
- INVARIANT or PARTIAL: #28 unblocks unchanged; comment on #24 that Stage 1 should include Q2 (a second target task).
- DOMAIN_SPECIFIC: label #28 `blocked` with the reason; open an issue "Re-plan Stage 0 per domain" referencing the report.
- INCONCLUSIVE_*: open an issue named for the cause with the report attached; do not spend further GPU until it is resolved.

---

## Self-review notes

**Spec coverage.** Section 3 arms: Task 6 `ARMS` and Task 9 `ARMS` match. Section 4 suite: Task 2, with the corrected buckets. Section 5 statistics and rule: Tasks 8 and 9, and `decide` matches the prereg table row for row. Section 6 architecture: every listed path has a task; the Modal fan-out is explicitly not built. Section 7 compute and memory check: Task 10 steps 3 and 4. Section 8 lr rule: Task 10 step 4 and `--lr` in Task 5. Section 9 checkpoints: H1, H2, and Task 10 step 5. #33: Task 1. #20: Task 6 and Task 10 step 2. #21: Task 10 step 3.

**Type consistency.** `run_probe` keyword names in Task 4 match the CLI call in Task 5 and the fakes in both test files. `preeval_path`/`telemetry_path` signatures match between Tasks 4 and 9. `CorrelationResult` fields (`rho, ci_low, ci_high, p_value, n`) match `stats.py`. `ProbeSuite.generate` takes `(n, seed)` positionally, which both `generate_countdown(n, seed, buckets=...)` and `generate_graphpath(n, seed, buckets=...)` satisfy.

**Review Focus coverage.** 1 -> `test_verifier_takes_the_last_answer_block_and_tolerates_whitespace`, `test_verifier_rejects_unparseable_and_lowercase`. 2 -> `test_no_breadth_exclusion_reduces_n_and_is_named`. 3 -> `test_load_arm_reports_missing_and_unreadable_files`, `test_three_missing_graph_jobs_are_inconclusive_and_named`. 4 -> `test_cache_is_keyed_by_suite_and_seed`. 5 -> `test_run_seed_changes_the_trainer_seed_but_not_the_items`. 6 -> `test_unknown_model_fails_before_any_gpu_work`.

**Test count.** 177 + 2 (T1) + 10 (T2) + 3 (T3) + 11 (T4) + 5 (T5) + 8 (T6) + 11 (T9) = 227 expected at the end of Task 9. Update the README's test count then.

**Execution model, per the owner.** Planning is done here; every line of code in this plan is written by Opus 5.5 subagents under `superpowers:subagent-driven-development`, one implementer and one reviewer per task, then a whole-branch review. The implementer for each task reads only that task plus the Global Constraints and the spec.
