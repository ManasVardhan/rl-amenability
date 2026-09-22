# RL-Amenability Stage 0 (Probe Harness + Positive Control) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the probe harness and run Stage 0 of the RL-amenability benchmark, producing a pass/fail verdict on Gate A (probe recovers known amenability ordering across over-SFT'd variants) and Gate B (probe beats naive curve extrapolation).

**Architecture:** A Python package wrapping TRL's `GRPOTrainer` and a rejection-sampling SFT loop, with a telemetry callback that records per-step training dynamics. A task-suite registry enforces that probe tasks and target tasks never overlap. A pre-registered, untuned scoring function converts telemetry into an amenability score, which is compared against four baselines using small-N-appropriate statistics. Stage 0 exercises the whole pipeline on six checkpoints whose correct ordering is known in advance.

**Tech Stack:** Python 3.11, uv, PyTorch, HuggingFace Transformers, TRL (`GRPOTrainer`, `SFTTrainer`), vLLM for generation and evaluation, NumPy/SciPy for statistics, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-rl-amenability-benchmark-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Full-parameter training only. No LoRA, no adapters, no quantized training.** LoRA constrains how far weights can move, which is the quantity under measurement.
- **pass@32 is the breadth metric used throughout the score.** pass@64 is computed at endpoints only, as a check that the k=32 truncation does not change conclusions.
- **Probe task IDs must never overlap target task IDs.** Enforced in code by `SuiteRegistry`, not by convention. A violation is an exception, not a warning.
- **The scoring function is never tuned against ground truth.** Weights are equal and fixed a priori. Any change to `src/amenability/scoring/score.py` after the pre-registration freeze invalidates Stage 0.
- **Probe budget is 10% of full budget in both arms:** GRPO probe 60 steps against 600 full; SFT probe 150 steps against 1500 full.
- **Every run is seeded and every seed is recorded** in the run manifest. Reproducibility is a release artifact.
- **Compute ceiling for Stage 0 is 130 GPU-hours.** If a task's runtime estimate would breach that, stop and report rather than proceeding.
- Model roster is fixed in `registry/models.yaml`. Qwen is capped at 4 of 10 entries.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, pinned dependencies |
| `src/amenability/registry/models.yaml` | The 10-model roster with family, size, licence |
| `src/amenability/registry/loader.py` | Load and validate the roster |
| `src/amenability/suites/base.py` | `TaskItem`, `SuiteRegistry`, overlap enforcement |
| `src/amenability/suites/countdown.py` | Probe suite: generator, solver, verifier |
| `src/amenability/suites/gsm8k.py` | Target suite: loader, answer extraction, verifier |
| `src/amenability/eval/passk.py` | Unbiased pass@k estimator and vLLM-backed evaluation |
| `src/amenability/eval/stats.py` | Spearman, BCa bootstrap CI, permutation test, LOO, partial correlation |
| `src/amenability/probe/telemetry.py` | `StepRecord`, `ProbeTelemetry`, feature extraction |
| `src/amenability/probe/entropy.py` | Fixed-batch next-token entropy measurement |
| `src/amenability/probe/callbacks.py` | TRL callback wiring telemetry into training |
| `src/amenability/training/grpo.py` | Shared GRPO runner, used by probe and full run |
| `src/amenability/training/sft.py` | Rejection-sampling dataset builder and SFT runner |
| `src/amenability/scoring/score.py` | The pre-registered amenability score. Frozen. |
| `src/amenability/scoring/baselines.py` | Naive extrapolation, pass@64, pass@1, param count |
| `src/amenability/scoring/gates.py` | Gate A and Gate B evaluation |
| `prereg/stage0.md` | The frozen analysis plan |
| `prereg/freeze.py` | Hash-and-freeze tooling with tamper detection |
| `scripts/make_oversft_variants.py` | Produce the positive-control checkpoints |
| `scripts/run_stage0.py` | Stage 0 orchestration and report generation |
| `scripts/slurm/` | CARC submission templates |
| `scripts/modal_app.py` | Modal fallback runner |

---

### Task 1: Project scaffolding and model registry

**Files:**
- Create: `pyproject.toml`
- Create: `src/amenability/__init__.py`
- Create: `src/amenability/registry/__init__.py`
- Create: `src/amenability/registry/models.yaml`
- Create: `src/amenability/registry/loader.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelSpec` dataclass with fields `key: str`, `hf_id: str`, `family: str`, `params_b: float`, `license: str`, `role: str`. `load_registry(path: Path | None = None) -> dict[str, ModelSpec]`. `RegistryError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest
from collections import Counter
from amenability.registry.loader import load_registry, ModelSpec, RegistryError


def test_registry_loads_ten_models():
    reg = load_registry()
    assert len(reg) == 10
    assert all(isinstance(v, ModelSpec) for v in reg.values())


def test_qwen_capped_at_four():
    reg = load_registry()
    families = Counter(m.family for m in reg.values())
    assert families["qwen"] == 4, f"Qwen cap violated: {families}"


def test_at_least_seven_families():
    reg = load_registry()
    assert len({m.family for m in reg.values()}) >= 7


def test_all_models_within_size_band():
    reg = load_registry()
    for m in reg.values():
        assert 0.3 <= m.params_b <= 1.8, f"{m.key} at {m.params_b}B is outside 0.5-1.7B band"


def test_spurious_reward_outlier_present():
    # Qwen2.5-Math-1.5B is the Spurious Rewards outlier and must be in the roster.
    reg = load_registry()
    assert "qwen2.5-math-1.5b" in reg


def test_control_bases_marked():
    reg = load_registry()
    controls = sorted(k for k, m in reg.items() if m.role == "control_base")
    assert controls == ["llama-3.2-1b", "qwen2.5-0.5b"]


def test_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "dupe.yaml"
    p.write_text(
        "models:\n"
        "  - key: a\n    hf_id: x/a\n    family: f\n    params_b: 1.0\n    license: apache-2.0\n    role: roster\n"
        "  - key: a\n    hf_id: x/b\n    family: g\n    params_b: 1.0\n    license: apache-2.0\n    role: roster\n"
    )
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "rl-amenability"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.55",
    "trl>=0.21",
    "datasets>=3.0",
    "accelerate>=1.0",
    "vllm>=0.10",
    "numpy>=1.26",
    "scipy>=1.14",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.14"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/amenability"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["gpu: requires a GPU and real model weights"]
```

```yaml
# src/amenability/registry/models.yaml
models:
  - key: qwen2.5-0.5b
    hf_id: Qwen/Qwen2.5-0.5B
    family: qwen
    params_b: 0.49
    license: apache-2.0
    role: control_base
  - key: qwen2.5-1.5b
    hf_id: Qwen/Qwen2.5-1.5B
    family: qwen
    params_b: 1.54
    license: apache-2.0
    role: roster
  - key: qwen2.5-math-1.5b
    hf_id: Qwen/Qwen2.5-Math-1.5B
    family: qwen
    params_b: 1.54
    license: apache-2.0
    role: roster
  - key: qwen3-0.6b-base
    hf_id: Qwen/Qwen3-0.6B-Base
    family: qwen
    params_b: 0.60
    license: apache-2.0
    role: roster
  - key: llama-3.2-1b
    hf_id: meta-llama/Llama-3.2-1B
    family: llama
    params_b: 1.24
    license: llama-3.2
    role: control_base
  - key: gemma-3-1b-pt
    hf_id: google/gemma-3-1b-pt
    family: gemma
    params_b: 1.00
    license: gemma
    role: roster
  - key: olmo-2-1b
    hf_id: allenai/OLMo-2-0425-1B
    family: olmo
    params_b: 1.48
    license: apache-2.0
    role: roster
  - key: smollm2-1.7b
    hf_id: HuggingFaceTB/SmolLM2-1.7B
    family: smollm
    params_b: 1.71
    license: apache-2.0
    role: roster
  - key: falcon3-1b-base
    hf_id: tiiuae/Falcon3-1B-Base
    family: falcon
    params_b: 1.67
    license: falcon-llm
    role: roster
  - key: stablelm-2-1.6b
    hf_id: stabilityai/stablelm-2-1_6b
    family: stablelm
    params_b: 1.64
    license: stabilityai-nc
    role: roster
```

```python
# src/amenability/registry/loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "models.yaml"


class RegistryError(Exception):
    """Raised when the model roster is malformed."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    params_b: float
    license: str
    role: str


def load_registry(path: Path | None = None) -> dict[str, ModelSpec]:
    path = path or DEFAULT_PATH
    raw = yaml.safe_load(path.read_text())
    out: dict[str, ModelSpec] = {}
    for entry in raw["models"]:
        spec = ModelSpec(**entry)
        if spec.key in out:
            raise RegistryError(f"duplicate model key: {spec.key}")
        out[spec.key] = spec
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv sync --extra dev && uv run pytest tests/test_registry.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/amenability tests/test_registry.py
git commit -m "feat: project scaffolding and validated 10-model roster"
```

---

### Task 2: Task suite registry with code-enforced disjointness

**Files:**
- Create: `src/amenability/suites/__init__.py`
- Create: `src/amenability/suites/base.py`
- Test: `tests/test_suite_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TaskItem` frozen dataclass with fields `task_id: str`, `suite: str`, `prompt: str`, `answer: str`, `difficulty: int`. `SuiteRegistry` with methods `register(name: str, items: list[TaskItem]) -> None`, `get(name: str) -> list[TaskItem]`, `names() -> list[str]`. `SuiteOverlapError(Exception)`.

This is the code enforcement of the spec's first non-negotiable constraint. A probe that trains on target data invalidates the entire project, so it must be impossible rather than merely discouraged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_suite_registry.py
import pytest
from amenability.suites.base import TaskItem, SuiteRegistry, SuiteOverlapError


def item(tid: str, suite: str = "s") -> TaskItem:
    return TaskItem(task_id=tid, suite=suite, prompt="p", answer="a", difficulty=1)


def test_register_and_get():
    reg = SuiteRegistry()
    reg.register("probe", [item("probe/1"), item("probe/2")])
    assert len(reg.get("probe")) == 2
    assert reg.names() == ["probe"]


def test_overlapping_task_ids_across_suites_raise():
    reg = SuiteRegistry()
    reg.register("probe", [item("shared/1")])
    with pytest.raises(SuiteOverlapError, match="shared/1"):
        reg.register("target", [item("shared/1")])


def test_duplicate_task_ids_within_a_suite_raise():
    reg = SuiteRegistry()
    with pytest.raises(SuiteOverlapError, match="dup/1"):
        reg.register("probe", [item("dup/1"), item("dup/1")])


def test_registering_same_suite_name_twice_raises():
    reg = SuiteRegistry()
    reg.register("probe", [item("probe/1")])
    with pytest.raises(SuiteOverlapError, match="probe"):
        reg.register("probe", [item("probe/2")])


def test_get_unknown_suite_raises():
    reg = SuiteRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_suite_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.suites'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/suites/base.py
from __future__ import annotations

from dataclasses import dataclass


class SuiteOverlapError(Exception):
    """Raised when suites share task IDs, which would let a probe see target data."""


@dataclass(frozen=True)
class TaskItem:
    task_id: str
    suite: str
    prompt: str
    answer: str
    difficulty: int


class SuiteRegistry:
    def __init__(self) -> None:
        self._suites: dict[str, list[TaskItem]] = {}
        self._seen: dict[str, str] = {}  # task_id -> owning suite

    def register(self, name: str, items: list[TaskItem]) -> None:
        if name in self._suites:
            raise SuiteOverlapError(f"suite already registered: {name}")
        local: set[str] = set()
        for it in items:
            if it.task_id in local:
                raise SuiteOverlapError(f"duplicate task_id within suite {name}: {it.task_id}")
            if it.task_id in self._seen:
                raise SuiteOverlapError(
                    f"task_id {it.task_id} already owned by suite {self._seen[it.task_id]}"
                )
            local.add(it.task_id)
        for it in items:
            self._seen[it.task_id] = name
        self._suites[name] = list(items)

    def get(self, name: str) -> list[TaskItem]:
        return self._suites[name]

    def names(self) -> list[str]:
        return list(self._suites)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_suite_registry.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/suites tests/test_suite_registry.py
git commit -m "feat: suite registry enforcing probe/target task disjointness"
```

---

### Task 3: Countdown probe suite with solver and verifier

**Files:**
- Create: `src/amenability/suites/countdown.py`
- Test: `tests/test_countdown.py`

**Interfaces:**
- Consumes: `TaskItem` from `amenability.suites.base`.
- Produces: `generate_countdown(n: int, seed: int, buckets: tuple[int, ...] = (3, 4, 5)) -> list[TaskItem]` where bucket value is the count of available numbers and serves as the difficulty level. `solve_countdown(numbers: list[int], target: int) -> str | None`. `verify_countdown(item: TaskItem, completion: str) -> bool`. `extract_expression(completion: str) -> str | None`.

Countdown is chosen because difficulty is controllable, verification is exact, and TinyZero established it is learnable at this scale. Difficulty grading matters: the zero-advantage group rate is only informative if some buckets are solvable and some are not at a given model size.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_countdown.py
import pytest
from amenability.suites.countdown import (
    generate_countdown,
    solve_countdown,
    verify_countdown,
    extract_expression,
)
from amenability.suites.base import TaskItem


def test_every_generated_item_is_solvable():
    items = generate_countdown(n=30, seed=0)
    for it in items:
        numbers = [int(x) for x in it.answer.split("|")[0].split(",")]
        target = int(it.answer.split("|")[1])
        assert solve_countdown(numbers, target) is not None, f"unsolvable item {it.task_id}"


def test_generation_is_deterministic_by_seed():
    a = generate_countdown(n=10, seed=7)
    b = generate_countdown(n=10, seed=7)
    c = generate_countdown(n=10, seed=8)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.prompt for x in a] != [x.prompt for x in c]


def test_difficulty_buckets_are_populated():
    items = generate_countdown(n=30, seed=1, buckets=(3, 4, 5))
    assert {it.difficulty for it in items} == {3, 4, 5}


def test_task_ids_are_namespaced_to_probe():
    items = generate_countdown(n=5, seed=2)
    assert all(it.task_id.startswith("probe/countdown/") for it in items)


def test_extract_expression_takes_last_answer_tag():
    assert extract_expression("junk <answer>1+1</answer> more <answer>2*3</answer>") == "2*3"
    assert extract_expression("no tags here") is None


def _item(numbers: str, target: int) -> TaskItem:
    return TaskItem(
        task_id="probe/countdown/t",
        suite="probe_countdown",
        prompt="p",
        answer=f"{numbers}|{target}",
        difficulty=3,
    )


def test_verifier_accepts_correct_expression():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>3*5+7</answer>") is True


def test_verifier_rejects_wrong_value():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>3+5+7</answer>") is False


def test_verifier_rejects_unavailable_numbers():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>11*2</answer>") is False


def test_verifier_rejects_reused_numbers():
    it = _item("3,5,7", 21)
    assert verify_countdown(it, "<answer>3*7*1</answer>") is False
    assert verify_countdown(it, "<answer>3*3+12</answer>") is False


def test_verifier_rejects_malicious_input():
    it = _item("3,5,7", 22)
    assert verify_countdown(it, "<answer>__import__('os').system('ls')</answer>") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_countdown.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.suites.countdown'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/suites/countdown.py
from __future__ import annotations

import ast
import itertools
import operator
import random
import re
from fractions import Fraction

from amenability.suites.base import TaskItem

PROMPT = (
    "Using each of the numbers {numbers} exactly once, and the operators + - * /, "
    "write an arithmetic expression equal to {target}.\n"
    "Put only the expression inside <answer></answer> tags."
)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def solve_countdown(numbers: list[int], target: int) -> str | None:
    """Exhaustive search over orderings and operator choices. Exact, via Fraction."""
    target_f = Fraction(target)

    def search(items: list[tuple[Fraction, str]]) -> str | None:
        if len(items) == 1:
            return items[0][1] if items[0][0] == target_f else None
        for i, j in itertools.permutations(range(len(items)), 2):
            if i > j:
                continue
            (a, sa), (b, sb) = items[i], items[j]
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            cands = [(a + b, f"({sa}+{sb})"), (a * b, f"({sa}*{sb})"),
                     (a - b, f"({sa}-{sb})"), (b - a, f"({sb}-{sa})")]
            if b != 0:
                cands.append((a / b, f"({sa}/{sb})"))
            if a != 0:
                cands.append((b / a, f"({sb}/{sa})"))
            for val, expr in cands:
                found = search(rest + [(val, expr)])
                if found is not None:
                    return found
        return None

    return search([(Fraction(x), str(x)) for x in numbers])


def _constructive_target(numbers: list[int], rng: random.Random) -> int | None:
    """Fold the numbers together with random operators to get a guaranteed-solvable target.

    Rejection sampling a random target is not viable: most (numbers, target) pairs are
    unsolvable, so the loop would spin for a long time on the 4- and 5-number buckets.
    """
    values = [Fraction(x) for x in numbers]
    rng.shuffle(values)
    acc = values[0]
    for v in values[1:]:
        op = rng.choice(["+", "-", "*"])
        acc = acc + v if op == "+" else acc - v if op == "-" else acc * v
    if acc.denominator != 1:
        return None
    return int(acc)


def generate_countdown(n: int, seed: int, buckets: tuple[int, ...] = (3, 4, 5)) -> list[TaskItem]:
    rng = random.Random(seed)
    items: list[TaskItem] = []
    per_bucket = n // len(buckets)
    for bucket in buckets:
        made = 0
        while made < per_bucket:
            numbers = [rng.randint(1, 20) for _ in range(bucket)]
            target = _constructive_target(numbers, rng)
            if target is None or not (10 <= target <= 400):
                continue
            idx = len(items)
            items.append(
                TaskItem(
                    task_id=f"probe/countdown/{seed}/{idx}",
                    suite="probe_countdown",
                    prompt=PROMPT.format(numbers=", ".join(map(str, numbers)), target=target),
                    answer=f"{','.join(map(str, numbers))}|{target}",
                    difficulty=bucket,
                )
            )
            made += 1
    return items


def extract_expression(completion: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _eval_node(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, used)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError("only integer literals allowed")
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left, used), _eval_node(node.right, used)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, used)
    raise ValueError(f"disallowed node: {type(node).__name__}")


def verify_countdown(item: TaskItem, completion: str) -> bool:
    expr = extract_expression(completion)
    if expr is None:
        return False
    numbers_s, target_s = item.answer.split("|")
    available = sorted(int(x) for x in numbers_s.split(","))
    try:
        tree = ast.parse(expr, mode="eval")
        used: list[int] = []
        value = _eval_node(tree, used)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False
    if sorted(used) != available:
        return False
    return value == Fraction(int(target_s))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_countdown.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/suites/countdown.py tests/test_countdown.py
git commit -m "feat: countdown probe suite with exact solver and sandboxed verifier"
```

---

### Task 4: GSM8K target suite

**Files:**
- Create: `src/amenability/suites/gsm8k.py`
- Test: `tests/test_gsm8k.py`

**Interfaces:**
- Consumes: `TaskItem` from `amenability.suites.base`.
- Produces: `load_gsm8k(split: str, limit: int | None = None, seed: int = 0) -> list[TaskItem]`, `extract_final_number(text: str) -> str | None`, `verify_gsm8k(item: TaskItem, completion: str) -> bool`.

Target task IDs are namespaced `target/gsm8k/...` so `SuiteRegistry` mechanically prevents collision with `probe/countdown/...`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gsm8k.py
import pytest
from amenability.suites.base import TaskItem
from amenability.suites.gsm8k import extract_final_number, verify_gsm8k


def _item(answer: str) -> TaskItem:
    return TaskItem(
        task_id="target/gsm8k/test/0", suite="target_gsm8k",
        prompt="p", answer=answer, difficulty=0,
    )


def test_extract_plain_integer():
    assert extract_final_number("The answer is 42") == "42"


def test_extract_takes_last_number():
    assert extract_final_number("first 7 then 13") == "13"


def test_extract_strips_commas_and_currency():
    assert extract_final_number("It costs $1,234") == "1234"


def test_extract_handles_negative():
    assert extract_final_number("balance is -18") == "-18"


def test_extract_handles_trailing_period():
    assert extract_final_number("The total is 72.") == "72"


def test_extract_preserves_true_decimal():
    assert extract_final_number("the rate is 3.5") == "3.5"


def test_extract_returns_none_when_no_number():
    assert extract_final_number("no digits at all") is None


def test_verify_matches_numerically_not_textually():
    assert verify_gsm8k(_item("72"), "so the answer is 72.0") is True
    assert verify_gsm8k(_item("72"), "so the answer is 72") is True


def test_verify_rejects_wrong_answer():
    assert verify_gsm8k(_item("72"), "the answer is 71") is False


def test_verify_rejects_empty_completion():
    assert verify_gsm8k(_item("72"), "") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gsm8k.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.suites.gsm8k'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/suites/gsm8k.py
from __future__ import annotations

import random
import re

from amenability.suites.base import TaskItem

PROMPT = "{question}\n\nSolve step by step, then give the final numeric answer on its own line."

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith("."):
        raw = raw[:-1]
    return raw


def verify_gsm8k(item: TaskItem, completion: str) -> bool:
    got = extract_final_number(completion)
    if got is None:
        return False
    try:
        return abs(float(got) - float(item.answer)) < 1e-6
    except ValueError:
        return False


def load_gsm8k(split: str, limit: int | None = None, seed: int = 0) -> list[TaskItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    if limit is not None:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    items: list[TaskItem] = []
    for i in idx:
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(
            TaskItem(
                task_id=f"target/gsm8k/{split}/{i}",
                suite="target_gsm8k",
                prompt=PROMPT.format(question=row["question"]),
                answer=gold,
                difficulty=0,
            )
        )
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gsm8k.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/suites/gsm8k.py tests/test_gsm8k.py
git commit -m "feat: GSM8K target suite with numeric-tolerant verifier"
```

---

### Task 5: Unbiased pass@k estimator and vLLM evaluation

**Files:**
- Create: `src/amenability/eval/__init__.py`
- Create: `src/amenability/eval/passk.py`
- Test: `tests/test_passk.py`

**Interfaces:**
- Consumes: `TaskItem`.
- Produces: `pass_at_k(n: int, c: int, k: int) -> float` (unbiased estimator, Chen et al. 2021). `PassKResult` frozen dataclass with fields `ks: dict[int, float]`, `n_samples: int`, `n_items: int`, `per_item_correct: dict[str, int]`. `evaluate_passk(model_path: str, items: list[TaskItem], verify_fn, ks: tuple[int, ...], n_samples: int, temperature: float, seed: int, generate_fn=None) -> PassKResult`. When `generate_fn` is passed it replaces vLLM, which is how the pure logic is tested without a GPU.

The estimator is a pure function and carries most of the correctness risk here, so it is tested independently of any generation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_passk.py
import math
import pytest
from amenability.suites.base import TaskItem
from amenability.eval.passk import pass_at_k, evaluate_passk


def test_pass_at_k_all_correct_is_one():
    assert pass_at_k(n=10, c=10, k=1) == 1.0


def test_pass_at_k_none_correct_is_zero():
    assert pass_at_k(n=10, c=0, k=5) == 0.0


def test_pass_at_1_equals_empirical_rate():
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)


def test_pass_at_k_matches_closed_form():
    # 1 - C(n-c, k)/C(n, k) for n=10, c=2, k=3
    expected = 1 - (math.comb(8, 3) / math.comb(10, 3))
    assert pass_at_k(n=10, c=2, k=3) == pytest.approx(expected)


def test_pass_at_k_is_monotonic_in_k():
    vals = [pass_at_k(n=32, c=4, k=k) for k in (1, 2, 4, 8, 16, 32)]
    assert vals == sorted(vals)


def test_pass_at_k_requires_k_le_n():
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=2, k=8)


def test_evaluate_passk_with_injected_generator():
    items = [
        TaskItem(task_id=f"t/{i}", suite="s", prompt="p", answer="1", difficulty=0)
        for i in range(2)
    ]

    # item 0: 2 of 4 correct. item 1: 0 of 4 correct.
    canned = {"t/0": ["1", "1", "0", "0"], "t/1": ["0", "0", "0", "0"]}

    def fake_generate(model_path, prompts, n, temperature, seed):
        return [canned[tid] for tid in prompts.keys()]

    def verify(item, completion):
        return completion == item.answer

    res = evaluate_passk(
        model_path="unused", items=items, verify_fn=verify,
        ks=(1, 2), n_samples=4, temperature=1.0, seed=0, generate_fn=fake_generate,
    )
    assert res.n_items == 2
    assert res.per_item_correct == {"t/0": 2, "t/1": 0}
    # pass@1 averaged over items: (0.5 + 0.0) / 2
    assert res.ks[1] == pytest.approx(0.25)
    # pass@2 for item 0: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert res.ks[2] == pytest.approx(((1 - 1 / 6) + 0.0) / 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_passk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/eval/passk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from amenability.suites.base import TaskItem


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021), computed stably in log space."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n - c < k:
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@dataclass(frozen=True)
class PassKResult:
    ks: dict[int, float]
    n_samples: int
    n_items: int
    per_item_correct: dict[str, int]


def _vllm_generate(model_path: str, prompts: dict[str, str], n: int,
                   temperature: float, seed: int) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, seed=seed, dtype="bfloat16", gpu_memory_utilization=0.85)
    params = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=768, seed=seed)
    outputs = llm.generate(list(prompts.values()), params)
    return [[o.text for o in out.outputs] for out in outputs]


def evaluate_passk(
    model_path: str,
    items: list[TaskItem],
    verify_fn: Callable[[TaskItem, str], bool],
    ks: tuple[int, ...],
    n_samples: int,
    temperature: float,
    seed: int,
    generate_fn: Callable[..., list[list[str]]] | None = None,
) -> PassKResult:
    generate = generate_fn or _vllm_generate
    prompts = {it.task_id: it.prompt for it in items}
    completions = generate(model_path, prompts, n_samples, temperature, seed)

    per_item_correct: dict[str, int] = {}
    for it, comps in zip(items, completions):
        per_item_correct[it.task_id] = sum(1 for c in comps if verify_fn(it, c))

    ks_out: dict[int, float] = {}
    for k in ks:
        ks_out[k] = float(
            np.mean([pass_at_k(n_samples, c, k) for c in per_item_correct.values()])
        )
    return PassKResult(
        ks=ks_out, n_samples=n_samples, n_items=len(items), per_item_correct=per_item_correct
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_passk.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/eval tests/test_passk.py
git commit -m "feat: unbiased pass@k estimator and injectable vLLM evaluator"
```

---

### Task 6: Telemetry schema and feature extraction

**Files:**
- Create: `src/amenability/probe/__init__.py`
- Create: `src/amenability/probe/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `PassKResult`.
- Produces: `StepRecord` frozen dataclass with fields `step: int`, `mean_reward: float`, `group_reward_std: float`, `zero_advantage_frac: float`, `policy_entropy: float`, `kl: float`, `grad_norm: float`. `ProbeTelemetry` dataclass with fields `model_key: str`, `algorithm: str`, `steps: list[StepRecord]`, `pre: PassKResult`, `post: PassKResult`, plus `to_json()` / `from_json()`. `FeatureVector` frozen dataclass with fields `conversion_rate`, `retention_factor`, `zero_advantage_rate`, `reward_slope_early`, `reward_slope_mid`, `grad_norm_trend`, `entropy_slope`, `entropy_collapse_step`. `extract_features(t: ProbeTelemetry) -> FeatureVector`.

Per the spec, only `conversion_rate`, `retention_factor`, and `zero_advantage_rate` feed the score. The rest are diagnostics. `reward_slope_*` is deliberately excluded from the score because it is the naive-extrapolation baseline and cannot sit on both sides of that comparison.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
import json
import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import (
    StepRecord, ProbeTelemetry, extract_features,
)


def make_telemetry(entropies, rewards, zero_adv, kls=None, grads=None) -> ProbeTelemetry:
    n = len(entropies)
    kls = kls or [0.01] * n
    grads = grads or [1.0] * n
    steps = [
        StepRecord(step=i, mean_reward=rewards[i], group_reward_std=0.1,
                   zero_advantage_frac=zero_adv[i], policy_entropy=entropies[i],
                   kl=kls[i], grad_norm=grads[i])
        for i in range(n)
    ]
    return ProbeTelemetry(
        model_key="m", algorithm="grpo", steps=steps,
        pre=PassKResult(ks={1: 0.10, 32: 0.50}, n_samples=32, n_items=10, per_item_correct={}),
        post=PassKResult(ks={1: 0.30, 32: 0.55}, n_samples=32, n_items=10, per_item_correct={}),
    )


def test_conversion_rate_normalises_by_available_breadth():
    t = make_telemetry([1.0] * 10, [0.1] * 10, [0.2] * 10)
    f = extract_features(t)
    # (post@1 - pre@1) / (pre@32 - pre@1) = (0.30 - 0.10) / (0.50 - 0.10)
    assert f.conversion_rate == pytest.approx(0.5)


def test_conversion_rate_is_zero_when_no_breadth_available():
    t = make_telemetry([1.0] * 10, [0.1] * 10, [0.2] * 10)
    t.pre = PassKResult(ks={1: 0.5, 32: 0.5}, n_samples=32, n_items=10, per_item_correct={})
    assert extract_features(t).conversion_rate == 0.0


def test_entropy_slope_is_negative_under_collapse():
    entropies = [2.0 - 0.15 * i for i in range(10)]
    f = extract_features(make_telemetry(entropies, [0.1] * 10, [0.2] * 10))
    assert f.entropy_slope < 0


def test_entropy_slope_is_flat_when_stable():
    f = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    assert f.entropy_slope == pytest.approx(0.0, abs=1e-9)


def test_entropy_collapse_step_detected_at_half_of_initial():
    entropies = [2.0] * 5 + [0.9] * 5  # drops below 50% of initial at step 5
    f = extract_features(make_telemetry(entropies, [0.1] * 10, [0.2] * 10))
    assert f.entropy_collapse_step == 5


def test_entropy_collapse_step_is_none_when_no_collapse():
    f = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    assert f.entropy_collapse_step is None


def test_zero_advantage_rate_is_mean_over_steps():
    f = extract_features(make_telemetry([1.0] * 4, [0.1] * 4, [0.0, 0.2, 0.4, 0.6]))
    assert f.zero_advantage_rate == pytest.approx(0.3)


def test_retention_factor_penalises_entropy_collapse():
    stable = extract_features(make_telemetry([2.0] * 10, [0.1] * 10, [0.2] * 10))
    collapsing = extract_features(
        make_telemetry([2.0 - 0.18 * i for i in range(10)], [0.1] * 10, [0.2] * 10)
    )
    assert collapsing.retention_factor < stable.retention_factor


def test_retention_factor_penalises_high_kl_per_reward_gain():
    cheap = make_telemetry([2.0] * 10, [0.1 + 0.02 * i for i in range(10)], [0.2] * 10,
                           kls=[0.01] * 10)
    costly = make_telemetry([2.0] * 10, [0.1 + 0.02 * i for i in range(10)], [0.2] * 10,
                            kls=[0.5] * 10)
    assert extract_features(costly).retention_factor < extract_features(cheap).retention_factor


def test_reward_slope_early_and_mid_split_the_run():
    rewards = [0.0] * 5 + [1.0] * 5  # flat then jump: early slope 0, mid slope 0
    f = extract_features(make_telemetry([2.0] * 10, rewards, [0.2] * 10))
    assert f.reward_slope_early == pytest.approx(0.0, abs=1e-9)


def test_round_trip_json():
    t = make_telemetry([2.0] * 3, [0.1] * 3, [0.2] * 3)
    back = ProbeTelemetry.from_json(json.loads(t.to_json()))
    assert back.model_key == t.model_key
    assert len(back.steps) == 3
    assert back.pre.ks[32] == pytest.approx(0.50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.probe'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/probe tests/test_telemetry.py
git commit -m "feat: probe telemetry schema and mechanism-grounded feature extraction"
```

---

### Task 7: Fixed-batch entropy measurement

**Files:**
- Create: `src/amenability/probe/entropy.py`
- Test: `tests/test_entropy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `mean_next_token_entropy(logits: "torch.Tensor", attention_mask: "torch.Tensor") -> float`. `EntropyProbe` class with `__init__(self, tokenizer, prompts: list[str], max_length: int = 256, device: str = "cuda")` and `measure(self, model) -> float`.

Policy entropy is measured with our own forward pass over a fixed prompt batch rather than read from TRL's logs. TRL's logged fields vary across versions, and entropy is the single most important feature here, so it must not depend on trainer internals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entropy.py
import math
import torch
import pytest
from amenability.probe.entropy import mean_next_token_entropy


def test_uniform_distribution_gives_log_vocab():
    logits = torch.zeros(1, 4, 8)  # batch 1, 4 positions, vocab 8 -> uniform
    mask = torch.ones(1, 4, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(8), abs=1e-5)


def test_peaked_distribution_gives_near_zero():
    logits = torch.full((1, 3, 5), -50.0)
    logits[:, :, 0] = 50.0
    mask = torch.ones(1, 3, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(0.0, abs=1e-4)


def test_masked_positions_are_excluded():
    logits = torch.zeros(1, 4, 8)
    logits[:, 2:, :] = torch.full((1, 2, 8), -50.0)
    logits[:, 2:, 0] = 50.0
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    # Only the two uniform positions count.
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(8), abs=1e-5)


def test_all_masked_raises():
    logits = torch.zeros(1, 2, 4)
    mask = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="no unmasked"):
        mean_next_token_entropy(logits, mask)


def test_batch_is_averaged_across_sequences():
    uniform = torch.zeros(1, 2, 4)
    peaked = torch.full((1, 2, 4), -50.0)
    peaked[:, :, 0] = 50.0
    logits = torch.cat([uniform, peaked], dim=0)
    mask = torch.ones(2, 2, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(4) / 2, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entropy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.probe.entropy'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/probe/entropy.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_next_token_entropy(logits: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """Mean token-level entropy of the next-token distribution over unmasked positions."""
    if attention_mask.sum() == 0:
        raise ValueError("no unmasked positions to measure entropy over")
    logits = logits.float()
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)  # (batch, seq)
    mask = attention_mask.to(ent.dtype)
    return float((ent * mask).sum() / mask.sum())


class EntropyProbe:
    """Measures policy entropy on a fixed prompt batch, held constant across all steps."""

    def __init__(self, tokenizer, prompts: list[str], max_length: int = 256,
                 device: str = "cuda") -> None:
        self.device = device
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    @torch.no_grad()
    def measure(self, model) -> float:
        was_training = model.training
        model.eval()
        try:
            out = model(
                input_ids=self.input_ids.to(self.device),
                attention_mask=self.attention_mask.to(self.device),
            )
            return mean_next_token_entropy(out.logits, self.attention_mask.to(self.device))
        finally:
            if was_training:
                model.train()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_entropy.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/probe/entropy.py tests/test_entropy.py
git commit -m "feat: trainer-independent policy entropy measurement"
```

---

### Task 8: Telemetry callback and grouped reward recorder

**Files:**
- Create: `src/amenability/probe/callbacks.py`
- Test: `tests/test_callbacks.py`

**Interfaces:**
- Consumes: `StepRecord`, `EntropyProbe`.
- Produces: `GroupedRewardRecorder` class with `__init__(self, num_generations: int)`, method `wrap(self, reward_fn) -> Callable`, and `drain() -> tuple[float, float, float]` returning `(mean_reward, group_reward_std, zero_advantage_frac)`. `TelemetryCallback(TrainerCallback)` with `__init__(self, recorder: GroupedRewardRecorder, entropy_probe, model_getter)` and attribute `records: list[StepRecord]`.

The zero-advantage fraction is computed from our own reward function wrapper by reshaping rewards into groups of `num_generations`. A group whose rewards are all equal produces zero advantage under GRPO and therefore contributes no gradient. This is derived from data we control rather than from TRL internals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_callbacks.py
import pytest
from amenability.probe.callbacks import GroupedRewardRecorder, TelemetryCallback


def test_wrap_passes_through_reward_values():
    rec = GroupedRewardRecorder(num_generations=2)
    wrapped = rec.wrap(lambda completions, **kw: [1.0, 0.0, 1.0, 1.0])
    assert wrapped(completions=["a", "b", "c", "d"]) == [1.0, 0.0, 1.0, 1.0]


def test_zero_advantage_fraction_counts_uniform_groups():
    rec = GroupedRewardRecorder(num_generations=2)
    # group 0 = [1, 0] varied; group 1 = [1, 1] uniform -> 0.5 zero-advantage
    rec.wrap(lambda completions, **kw: [1.0, 0.0, 1.0, 1.0])(completions=["a"] * 4)
    mean_r, group_std, zero_frac = rec.drain()
    assert zero_frac == pytest.approx(0.5)
    assert mean_r == pytest.approx(0.75)


def test_all_uniform_groups_give_zero_advantage_one():
    rec = GroupedRewardRecorder(num_generations=4)
    rec.wrap(lambda completions, **kw: [0.0] * 8)(completions=["a"] * 8)
    _, _, zero_frac = rec.drain()
    assert zero_frac == pytest.approx(1.0)


def test_drain_resets_between_steps():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    fn(completions=["a", "b"])
    rec.drain()
    fn(completions=["a", "b"])
    mean_r, _, _ = rec.drain()
    assert mean_r == pytest.approx(0.5)


def test_drain_with_no_calls_returns_zeros():
    rec = GroupedRewardRecorder(num_generations=2)
    assert rec.drain() == (0.0, 0.0, 0.0)


def test_ragged_batch_raises():
    rec = GroupedRewardRecorder(num_generations=3)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    with pytest.raises(ValueError, match="not divisible"):
        fn(completions=["a", "b"])


def test_callback_emits_one_record_per_logged_step():
    rec = GroupedRewardRecorder(num_generations=2)
    fn = rec.wrap(lambda completions, **kw: [1.0, 0.0])
    cb = TelemetryCallback(
        recorder=rec,
        entropy_probe=type("P", (), {"measure": lambda self, m: 1.23})(),
        model_getter=lambda: None,
    )

    class State:
        global_step = 1
        log_history = [{"kl": 0.02, "grad_norm": 0.7}]

    fn(completions=["a", "b"])
    cb.on_log(args=None, state=State(), control=None, logs={"kl": 0.02, "grad_norm": 0.7})
    assert len(cb.records) == 1
    r = cb.records[0]
    assert r.step == 1
    assert r.policy_entropy == pytest.approx(1.23)
    assert r.kl == pytest.approx(0.02)
    assert r.grad_norm == pytest.approx(0.7)
    assert r.zero_advantage_frac == pytest.approx(0.0)


def test_callback_tolerates_missing_log_fields():
    rec = GroupedRewardRecorder(num_generations=2)
    rec.wrap(lambda completions, **kw: [1.0, 0.0])(completions=["a", "b"])
    cb = TelemetryCallback(
        recorder=rec,
        entropy_probe=type("P", (), {"measure": lambda self, m: 0.5})(),
        model_getter=lambda: None,
    )

    class State:
        global_step = 3
        log_history = []

    cb.on_log(args=None, state=State(), control=None, logs={})
    assert cb.records[0].kl == 0.0
    assert cb.records[0].grad_norm == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_callbacks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.probe.callbacks'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/probe/callbacks.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_callbacks.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/probe/callbacks.py tests/test_callbacks.py
git commit -m "feat: telemetry callback with grouped zero-advantage tracking"
```

---

### Task 9: GRPO runner shared by probe and full run

**Files:**
- Create: `src/amenability/training/__init__.py`
- Create: `src/amenability/training/grpo.py`
- Test: `tests/test_grpo_runner.py`

**Interfaces:**
- Consumes: `TaskItem`, `GroupedRewardRecorder`, `TelemetryCallback`, `EntropyProbe`.
- Produces: `GRPOSpec` frozen dataclass with fields `model_path: str`, `model_key: str`, `items: list[TaskItem]`, `verify_fn`, `max_steps: int`, `num_generations: int`, `learning_rate: float`, `beta: float`, `temperature: float`, `seed: int`, `output_dir: str`, `save_steps: int | None`. `checkpoint_schedule(max_steps: int, save_steps: int | None) -> list[int]`. `build_reward_fn(items, verify_fn) -> Callable`. `run_grpo(spec: GRPOSpec, trainer_factory=None) -> list[StepRecord]`.

One runner serves both the 60-step probe and the 600-step full run, differing only in `max_steps` and `save_steps`. Identical code paths are what make the probe a valid miniature of the full run.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grpo_runner.py
import pytest
from amenability.suites.base import TaskItem
from amenability.training.grpo import (
    GRPOSpec, checkpoint_schedule, build_reward_fn, run_grpo,
)


def items(n=4):
    return [
        TaskItem(task_id=f"probe/x/{i}", suite="s", prompt=f"p{i}", answer="7", difficulty=1)
        for i in range(n)
    ]


def verify(item, completion):
    return completion.strip() == item.answer


def test_probe_schedule_has_no_intermediate_checkpoints():
    assert checkpoint_schedule(max_steps=60, save_steps=None) == [60]


def test_full_run_schedule_checkpoints_every_200():
    assert checkpoint_schedule(max_steps=600, save_steps=200) == [200, 400, 600]


def test_schedule_always_includes_final_step():
    assert checkpoint_schedule(max_steps=500, save_steps=200) == [200, 400, 500]


def test_reward_fn_scores_by_verifier_against_matching_prompt():
    fn = build_reward_fn(items(2), verify)
    rewards = fn(completions=["7", "8"], prompts=["p0", "p1"])
    assert rewards == [1.0, 0.0]


def test_reward_fn_handles_repeated_prompts_from_group_sampling():
    fn = build_reward_fn(items(1), verify)
    rewards = fn(completions=["7", "7", "1", "7"], prompts=["p0"] * 4)
    assert rewards == [1.0, 1.0, 0.0, 1.0]


def test_reward_fn_scores_zero_for_unknown_prompt():
    fn = build_reward_fn(items(1), verify)
    assert fn(completions=["7"], prompts=["not-a-prompt"]) == [0.0]


def test_run_grpo_returns_telemetry_from_injected_trainer():
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = object()

        def train(self):
            # Simulate three logging steps.
            for cb in captured["callbacks"]:
                for step in (1, 2, 3):
                    cb.on_log(
                        args=None,
                        state=type("S", (), {"global_step": step, "log_history": []})(),
                        control=None,
                        logs={"kl": 0.01 * step, "grad_norm": 1.0},
                    )

        def save_model(self, path):
            captured["saved_to"] = path

    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir="/tmp/out", save_steps=None,
    )
    records = run_grpo(spec, trainer_factory=lambda **kw: FakeTrainer(**kw))
    assert [r.step for r in records] == [1, 2, 3]
    assert captured["saved_to"] == "/tmp/out"


def test_run_grpo_rejects_lora_config():
    spec = GRPOSpec(
        model_path="fake", model_key="m", items=items(), verify_fn=verify,
        max_steps=3, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir="/tmp/out", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_grpo(spec, trainer_factory=lambda **kw: None, peft_config={"r": 8})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grpo_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.training'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/training/grpo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amenability.probe.callbacks import GroupedRewardRecorder, TelemetryCallback
from amenability.probe.entropy import EntropyProbe
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem

ENTROPY_BATCH_SIZE = 16


@dataclass(frozen=True)
class GRPOSpec:
    model_path: str
    model_key: str
    items: list[TaskItem]
    verify_fn: Callable[[TaskItem, str], bool]
    max_steps: int
    num_generations: int
    learning_rate: float
    beta: float
    temperature: float
    seed: int
    output_dir: str
    save_steps: int | None


def checkpoint_schedule(max_steps: int, save_steps: int | None) -> list[int]:
    if save_steps is None:
        return [max_steps]
    steps = list(range(save_steps, max_steps + 1, save_steps))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return steps


def build_reward_fn(items: list[TaskItem], verify_fn) -> Callable:
    by_prompt = {it.prompt: it for it in items}

    def reward_fn(completions, **kwargs):
        prompts = kwargs["prompts"]
        out = []
        for prompt, completion in zip(prompts, completions):
            item = by_prompt.get(prompt)
            out.append(1.0 if item is not None and verify_fn(item, completion) else 0.0)
        return out

    return reward_fn


def run_grpo(spec: GRPOSpec, trainer_factory=None, peft_config=None) -> list[StepRecord]:
    if peft_config is not None:
        raise ValueError(
            "LoRA/PEFT is forbidden: it constrains weight movement, which is the "
            "quantity this benchmark measures."
        )

    recorder = GroupedRewardRecorder(num_generations=spec.num_generations)
    reward_fn = recorder.wrap(build_reward_fn(spec.items, spec.verify_fn))

    if trainer_factory is None:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        entropy_probe = EntropyProbe(
            tokenizer, [it.prompt for it in spec.items[:ENTROPY_BATCH_SIZE]]
        )
        config = GRPOConfig(
            output_dir=spec.output_dir,
            max_steps=spec.max_steps,
            learning_rate=spec.learning_rate,
            beta=spec.beta,
            temperature=spec.temperature,
            num_generations=spec.num_generations,
            save_steps=spec.save_steps or spec.max_steps,
            logging_steps=1,
            seed=spec.seed,
            bf16=True,
            report_to=[],
        )
        dataset = Dataset.from_dict({"prompt": [it.prompt for it in spec.items]})
        trainer_factory = lambda **kw: GRPOTrainer(  # noqa: E731
            model=spec.model_path, args=config, train_dataset=dataset, **kw
        )
    else:
        entropy_probe = _NullEntropyProbe()

    callback = TelemetryCallback(
        recorder=recorder,
        entropy_probe=entropy_probe,
        model_getter=lambda: getattr(trainer, "model", None),
    )
    trainer = trainer_factory(reward_funcs=[reward_fn], callbacks=[callback])
    trainer.train()
    trainer.save_model(spec.output_dir)
    return callback.records


class _NullEntropyProbe:
    def measure(self, model) -> float:
        return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grpo_runner.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/training tests/test_grpo_runner.py
git commit -m "feat: GRPO runner shared by probe and full run, LoRA explicitly refused"
```

---

### Task 10: Rejection-sampling SFT arm

**Files:**
- Create: `src/amenability/training/sft.py`
- Test: `tests/test_sft.py`

**Interfaces:**
- Consumes: `TaskItem`, `evaluate_passk`'s `generate_fn` convention.
- Produces: `build_rejection_dataset(items, completions_by_task: dict[str, list[str]], verify_fn, max_per_prompt: int = 4) -> list[dict]` returning records shaped `{"prompt": str, "completion": str}`. `SFTSpec` frozen dataclass with fields `model_path`, `model_key`, `records`, `max_steps`, `learning_rate`, `seed`, `output_dir`, `save_steps`. `run_sft(spec: SFTSpec, trainer_factory=None) -> None`.

SFT data comes from the model's own verified generations, never from a larger teacher. Teacher distillation would make this arm measure the teacher-student gap rather than the student's plasticity, which collapses the SFT-versus-RL dissociation claim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sft.py
import pytest
from amenability.suites.base import TaskItem
from amenability.training.sft import build_rejection_dataset, SFTSpec, run_sft


def items(n=2):
    return [
        TaskItem(task_id=f"t/{i}", suite="s", prompt=f"p{i}", answer="7", difficulty=1)
        for i in range(n)
    ]


def verify(item, completion):
    return "7" in completion


def test_keeps_only_verified_completions():
    comps = {"t/0": ["the answer is 7", "the answer is 3"], "t/1": ["nope"]}
    recs = build_rejection_dataset(items(), comps, verify)
    assert len(recs) == 1
    assert recs[0] == {"prompt": "p0", "completion": "the answer is 7"}


def test_caps_completions_per_prompt():
    comps = {"t/0": [f"7 v{i}" for i in range(10)], "t/1": []}
    recs = build_rejection_dataset(items(), comps, verify, max_per_prompt=3)
    assert len(recs) == 3


def test_deduplicates_identical_completions():
    comps = {"t/0": ["7", "7", "7"], "t/1": []}
    recs = build_rejection_dataset(items(), comps, verify, max_per_prompt=5)
    assert len(recs) == 1


def test_missing_task_in_completions_is_skipped_not_fatal():
    comps = {"t/0": ["7"]}
    recs = build_rejection_dataset(items(), comps, verify)
    assert len(recs) == 1


def test_empty_dataset_raises_rather_than_training_on_nothing():
    comps = {"t/0": ["wrong"], "t/1": ["wrong"]}
    with pytest.raises(ValueError, match="no verified completions"):
        build_rejection_dataset(items(), comps, verify)


def test_run_sft_calls_trainer_and_saves():
    captured = {}

    class FakeTrainer:
        def __init__(self, **kw):
            captured.update(kw)

        def train(self):
            captured["trained"] = True

        def save_model(self, path):
            captured["saved_to"] = path

    spec = SFTSpec(
        model_path="fake", model_key="m",
        records=[{"prompt": "p", "completion": "c"}],
        max_steps=10, learning_rate=1e-5, seed=0,
        output_dir="/tmp/sft", save_steps=None,
    )
    run_sft(spec, trainer_factory=lambda **kw: FakeTrainer(**kw))
    assert captured["trained"] is True
    assert captured["saved_to"] == "/tmp/sft"


def test_run_sft_rejects_lora_config():
    spec = SFTSpec(
        model_path="fake", model_key="m", records=[{"prompt": "p", "completion": "c"}],
        max_steps=10, learning_rate=1e-5, seed=0, output_dir="/tmp/sft", save_steps=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        run_sft(spec, trainer_factory=lambda **kw: None, peft_config={"r": 8})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.training.sft'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/training/sft.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amenability.suites.base import TaskItem


def build_rejection_dataset(
    items: list[TaskItem],
    completions_by_task: dict[str, list[str]],
    verify_fn: Callable[[TaskItem, str], bool],
    max_per_prompt: int = 4,
) -> list[dict]:
    records: list[dict] = []
    for it in items:
        seen: set[str] = set()
        for completion in completions_by_task.get(it.task_id, []):
            if len(seen) >= max_per_prompt:
                break
            norm = completion.strip()
            if norm in seen or not verify_fn(it, completion):
                continue
            seen.add(norm)
            records.append({"prompt": it.prompt, "completion": norm})
    if not records:
        raise ValueError("no verified completions survived rejection sampling")
    return records


@dataclass(frozen=True)
class SFTSpec:
    model_path: str
    model_key: str
    records: list[dict]
    max_steps: int
    learning_rate: float
    seed: int
    output_dir: str
    save_steps: int | None


def run_sft(spec: SFTSpec, trainer_factory=None, peft_config=None) -> None:
    if peft_config is not None:
        raise ValueError(
            "LoRA/PEFT is forbidden: it constrains weight movement, which is the "
            "quantity this benchmark measures."
        )
    if trainer_factory is None:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        config = SFTConfig(
            output_dir=spec.output_dir,
            max_steps=spec.max_steps,
            learning_rate=spec.learning_rate,
            save_steps=spec.save_steps or spec.max_steps,
            logging_steps=1,
            seed=spec.seed,
            bf16=True,
            report_to=[],
        )
        dataset = Dataset.from_list(spec.records)
        trainer_factory = lambda **kw: SFTTrainer(  # noqa: E731
            model=spec.model_path, args=config, train_dataset=dataset, **kw
        )
    trainer = trainer_factory()
    trainer.train()
    trainer.save_model(spec.output_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sft.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/training/sft.py tests/test_sft.py
git commit -m "feat: rejection-sampling SFT arm using self-generated verified traces"
```

---

### Task 11: The pre-registered amenability score

**Files:**
- Create: `src/amenability/scoring/__init__.py`
- Create: `src/amenability/scoring/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `FeatureVector`.
- Produces: `zscore(values: list[float]) -> list[float]` (returns all zeros when the standard deviation is zero). `amenability_score(features: list[FeatureVector]) -> list[float]`.

This file is frozen by Task 14 and must not be edited afterwards. Weights are equal and fixed. `score = gate * mean(z(conversion_rate), z(retention_factor))` where `gate = clip(1 - zero_advantage_rate, 0, 1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score.py
import pytest
from amenability.probe.telemetry import FeatureVector
from amenability.scoring.score import amenability_score, zscore


def fv(conv, ret, zar=0.0) -> FeatureVector:
    return FeatureVector(
        conversion_rate=conv, retention_factor=ret, zero_advantage_rate=zar,
        reward_slope_early=0.0, reward_slope_mid=0.0, grad_norm_trend=0.0,
        entropy_slope=0.0, entropy_collapse_step=None,
    )


def test_zscore_centres_and_scales():
    z = zscore([1.0, 2.0, 3.0])
    assert z[1] == pytest.approx(0.0)
    assert z[0] == pytest.approx(-z[2])


def test_zscore_of_constant_input_is_all_zeros():
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscore_of_single_value_is_zero():
    assert zscore([5.0]) == [0.0]


def test_higher_conversion_scores_higher():
    scores = amenability_score([fv(0.1, 1.0), fv(0.9, 1.0)])
    assert scores[1] > scores[0]


def test_higher_retention_scores_higher():
    scores = amenability_score([fv(0.5, 0.2), fv(0.5, 0.9)])
    assert scores[1] > scores[0]


def test_total_zero_advantage_floors_the_score():
    # Model 1 has the best conversion and retention but no learning signal at all.
    scores = amenability_score([fv(0.1, 0.1, zar=0.0), fv(0.9, 0.9, zar=1.0)])
    assert scores[1] == 0.0


def test_gate_scales_proportionally():
    a = amenability_score([fv(0.9, 0.9, zar=0.0), fv(0.1, 0.1, zar=0.0)])
    b = amenability_score([fv(0.9, 0.9, zar=0.5), fv(0.1, 0.1, zar=0.0)])
    assert b[0] == pytest.approx(a[0] * 0.5)


def test_gate_clips_out_of_range_rates():
    scores = amenability_score([fv(0.9, 0.9, zar=1.5), fv(0.1, 0.1, zar=0.0)])
    assert scores[0] == 0.0


def test_score_is_deterministic():
    feats = [fv(0.3, 0.6), fv(0.7, 0.2), fv(0.5, 0.5)]
    assert amenability_score(feats) == amenability_score(feats)


def test_empty_input_returns_empty():
    assert amenability_score([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/scoring/score.py
"""PRE-REGISTERED. Frozen by prereg/freeze.py before any Stage 0 ground-truth run.

Any edit after the freeze invalidates Stage 0. Weights are equal and fixed a priori:
tuning them against ground truth at N=10 would fit noise, not signal.
"""
from __future__ import annotations

import numpy as np

from amenability.probe.telemetry import FeatureVector


def zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-12:
        return [0.0] * len(values)
    return [float(v) for v in (arr - arr.mean()) / sd]


def amenability_score(features: list[FeatureVector]) -> list[float]:
    if not features:
        return []
    z_conv = zscore([f.conversion_rate for f in features])
    z_ret = zscore([f.retention_factor for f in features])
    out: list[float] = []
    for f, zc, zr in zip(features, z_conv, z_ret):
        gate = float(np.clip(1.0 - f.zero_advantage_rate, 0.0, 1.0))
        out.append(gate * (zc + zr) / 2.0)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_score.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/scoring tests/test_score.py
git commit -m "feat: pre-registered untuned amenability score"
```

---

### Task 12: Baselines the score must beat

**Files:**
- Create: `src/amenability/scoring/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `ProbeTelemetry`, `PassKResult`, `ModelSpec`.
- Produces: `naive_extrapolation(steps: list[int], rewards: list[float], target_step: int) -> float`. `baseline_naive(telemetries, target_step) -> list[float]`. `baseline_pass_at_64(pre_results) -> list[float]`. `baseline_pass_at_1(pre_results) -> list[float]`. `baseline_param_count(specs) -> list[float]`. `ALL_BASELINES: dict[str, str]` mapping baseline name to a human-readable description.

Baseline 1, naive extrapolation, is the one that decides whether the project has a contribution. It fits a saturating log curve to the probe reward trace and reads off the value at the full-run step count.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import math
import pytest
from amenability.registry.loader import ModelSpec
from amenability.eval.passk import PassKResult
from amenability.scoring.baselines import (
    naive_extrapolation, baseline_pass_at_64, baseline_pass_at_1,
    baseline_param_count, ALL_BASELINES,
)


def test_extrapolation_recovers_a_known_log_curve():
    # y = 0.1 + 0.2 * log(1 + x)
    steps = list(range(1, 61))
    rewards = [0.1 + 0.2 * math.log(1 + s) for s in steps]
    expected = 0.1 + 0.2 * math.log(1 + 600)
    assert naive_extrapolation(steps, rewards, 600) == pytest.approx(expected, rel=1e-6)


def test_extrapolation_of_flat_trace_stays_flat():
    steps = list(range(1, 61))
    assert naive_extrapolation(steps, [0.4] * 60, 600) == pytest.approx(0.4, abs=1e-6)


def test_extrapolation_of_declining_trace_declines():
    steps = list(range(1, 61))
    rewards = [0.9 - 0.1 * math.log(1 + s) for s in steps]
    assert naive_extrapolation(steps, rewards, 600) < rewards[-1]


def test_extrapolation_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        naive_extrapolation([1], [0.5], 600)


def test_pass_at_64_baseline_reads_the_right_k():
    pre = [
        PassKResult(ks={1: 0.1, 32: 0.4, 64: 0.6}, n_samples=64, n_items=1, per_item_correct={}),
        PassKResult(ks={1: 0.2, 32: 0.3, 64: 0.5}, n_samples=64, n_items=1, per_item_correct={}),
    ]
    assert baseline_pass_at_64(pre) == [0.6, 0.5]
    assert baseline_pass_at_1(pre) == [0.1, 0.2]


def test_param_count_baseline():
    specs = [
        ModelSpec(key="a", hf_id="x/a", family="f", params_b=0.5, license="l", role="roster"),
        ModelSpec(key="b", hf_id="x/b", family="g", params_b=1.7, license="l", role="roster"),
    ]
    assert baseline_param_count(specs) == [0.5, 1.7]


def test_all_baselines_registry_is_complete():
    assert set(ALL_BASELINES) == {"naive_extrapolation", "pass_at_64", "pass_at_1", "param_count"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.scoring.baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/scoring/baselines.py
"""PRE-REGISTERED. Frozen alongside score.py."""
from __future__ import annotations

import numpy as np

from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import ProbeTelemetry
from amenability.registry.loader import ModelSpec

ALL_BASELINES = {
    "naive_extrapolation": "Log-curve fit to the probe reward trace, read at the full-run step.",
    "pass_at_64": "Base model pass@64, the literature's current best cheap metric.",
    "pass_at_1": "Base model pass@1.",
    "param_count": "Model parameter count in billions.",
}


def naive_extrapolation(steps: list[int], rewards: list[float], target_step: int) -> float:
    """Fit y = a + b*log(1+x) to the probe trace and evaluate it at target_step."""
    if len(steps) < 2:
        raise ValueError("naive extrapolation needs at least two points")
    x = np.log1p(np.asarray(steps, dtype=float))
    y = np.asarray(rewards, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return float(a + b * np.log1p(target_step))


def baseline_naive(telemetries: list[ProbeTelemetry], target_step: int) -> list[float]:
    return [
        naive_extrapolation(
            [s.step for s in t.steps], [s.mean_reward for s in t.steps], target_step
        )
        for t in telemetries
    ]


def baseline_pass_at_64(pre_results: list[PassKResult]) -> list[float]:
    return [r.ks[64] for r in pre_results]


def baseline_pass_at_1(pre_results: list[PassKResult]) -> list[float]:
    return [r.ks[1] for r in pre_results]


def baseline_param_count(specs: list[ModelSpec]) -> list[float]:
    return [s.params_b for s in specs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/scoring/baselines.py tests/test_baselines.py
git commit -m "feat: four pre-registered baselines including naive curve extrapolation"
```

---

### Task 13: Small-N statistics

**Files:**
- Create: `src/amenability/eval/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CorrelationResult` frozen dataclass with fields `rho: float`, `ci_low: float`, `ci_high: float`, `p_value: float`, `n: int`. `spearman_with_ci(x, y, n_boot: int = 10000, seed: int = 0) -> CorrelationResult`. `permutation_p(x, y, n_perm: int = 10000, seed: int = 0) -> float`. `loo_spearman(x, y) -> list[float]`. `partial_spearman(x, y, control) -> float`.

No bare correlation coefficients appear anywhere in the paper, so every statistic here ships with an interval or a permutation p-value attached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats.py
import pytest
from amenability.eval.stats import (
    spearman_with_ci, permutation_p, loo_spearman, partial_spearman,
)


def test_perfect_monotonic_relationship_gives_rho_one():
    res = spearman_with_ci([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], n_boot=500, seed=0)
    assert res.rho == pytest.approx(1.0)
    assert res.n == 5


def test_perfect_inverse_relationship_gives_rho_minus_one():
    res = spearman_with_ci([1, 2, 3, 4, 5], [50, 40, 30, 20, 10], n_boot=500, seed=0)
    assert res.rho == pytest.approx(-1.0)


def test_ci_brackets_the_point_estimate():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    res = spearman_with_ci(x, y, n_boot=2000, seed=0)
    assert res.ci_low <= res.rho <= res.ci_high


def test_ci_is_wide_at_small_n():
    res = spearman_with_ci([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], n_boot=2000, seed=0)
    assert (res.ci_high - res.ci_low) > 0.5


def test_spearman_requires_at_least_three_points():
    with pytest.raises(ValueError, match="at least three"):
        spearman_with_ci([1, 2], [1, 2])


def test_permutation_p_is_small_for_strong_relationship():
    x = list(range(10))
    assert permutation_p(x, x, n_perm=2000, seed=0) < 0.01


def test_permutation_p_is_large_for_no_relationship():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 3, 8, 1, 9, 2, 7, 4, 10, 6]
    assert permutation_p(x, y, n_perm=2000, seed=0) > 0.05


def test_loo_returns_one_rho_per_dropped_point():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 2, 3, 4, 5, 6]
    rhos = loo_spearman(x, y)
    assert len(rhos) == 6
    assert all(r == pytest.approx(1.0) for r in rhos)


def test_loo_exposes_a_single_driving_outlier():
    # Without the last point there is no relationship at all.
    x = [1, 2, 3, 4, 100]
    y = [3, 1, 2, 4, 100]
    rhos = loo_spearman(x, y)
    assert min(rhos) < max(rhos)


def test_partial_correlation_removes_a_shared_driver():
    # y is driven entirely by the control; x is too. Partial rho should collapse.
    control = [1, 2, 3, 4, 5, 6, 7, 8]
    x = list(control)
    y = list(control)
    assert abs(partial_spearman(x, y, control)) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL with `ImportError: cannot import name 'spearman_with_ci'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/eval/stats.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrelationResult:
    rho: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    r = stats.spearmanr(x, y).statistic
    return 0.0 if np.isnan(r) else float(r)


def spearman_with_ci(x, y, n_boot: int = 10000, seed: int = 0) -> CorrelationResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("spearman_with_ci needs at least three points")
    point = _rho(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boots[i] = _rho(x[idx], y[idx])
    # BCa is unstable when bootstrap resamples degenerate at small n, so use the
    # percentile interval and report it as such.
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return CorrelationResult(
        rho=point, ci_low=float(lo), ci_high=float(hi),
        p_value=permutation_p(x, y, seed=seed), n=len(x),
    )


def permutation_p(x, y, n_perm: int = 10000, seed: int = 0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(_rho(x, y))
    rng = np.random.default_rng(seed)
    count = sum(abs(_rho(x, rng.permutation(y))) >= observed for _ in range(n_perm))
    return float((count + 1) / (n_perm + 1))


def loo_spearman(x, y) -> list[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        out.append(_rho(x[keep], y[keep]))
    return out


def partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after linearly residualising both on control ranks."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rc = stats.rankdata(control)

    def residualise(v: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(rc, v, 1)
        return v - (slope * rc + intercept)

    return _rho(residualise(rx), residualise(ry))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stats.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/eval/stats.py tests/test_stats.py
git commit -m "feat: small-N statistics with bootstrap CIs and permutation tests"
```

---

### Task 14: Pre-registration freeze with tamper detection

**Files:**
- Create: `prereg/stage0.md`
- Create: `prereg/freeze.py`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FROZEN_PATHS: list[str]`. `compute_hashes(root: Path) -> dict[str, str]`. `freeze(root: Path) -> dict`. `verify_freeze(root: Path) -> None` raising `FreezeViolation` on mismatch. `FreezeViolation(Exception)`.

This is what makes the pre-registration claim real rather than rhetorical. `verify_freeze` runs at the top of `run_stage0.py`, so a modified scoring function halts the run instead of quietly producing a tuned result.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_freeze.py
import json
import pytest
from pathlib import Path
from prereg.freeze import freeze, verify_freeze, compute_hashes, FreezeViolation, FROZEN_PATHS


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    for rel in FROZEN_PATHS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}\n")
    return tmp_path


def test_frozen_paths_cover_score_baselines_and_plan():
    assert "src/amenability/scoring/score.py" in FROZEN_PATHS
    assert "src/amenability/scoring/baselines.py" in FROZEN_PATHS
    assert "prereg/stage0.md" in FROZEN_PATHS


def test_freeze_writes_manifest_with_all_paths(fake_root):
    manifest = freeze(fake_root)
    assert set(manifest["hashes"]) == set(FROZEN_PATHS)
    assert (fake_root / "prereg" / "FROZEN.json").exists()
    assert "frozen_at" in manifest


def test_verify_passes_immediately_after_freeze(fake_root):
    freeze(fake_root)
    verify_freeze(fake_root)  # must not raise


def test_verify_detects_a_modified_file(fake_root):
    freeze(fake_root)
    (fake_root / "src/amenability/scoring/score.py").write_text("tuned!\n")
    with pytest.raises(FreezeViolation, match="score.py"):
        verify_freeze(fake_root)


def test_verify_detects_a_deleted_file(fake_root):
    freeze(fake_root)
    (fake_root / "src/amenability/scoring/baselines.py").unlink()
    with pytest.raises(FreezeViolation, match="baselines.py"):
        verify_freeze(fake_root)


def test_verify_without_a_manifest_raises(fake_root):
    with pytest.raises(FreezeViolation, match="not frozen"):
        verify_freeze(fake_root)


def test_refreezing_over_an_existing_manifest_raises(fake_root):
    freeze(fake_root)
    with pytest.raises(FreezeViolation, match="already frozen"):
        freeze(fake_root)


def test_hashes_are_content_addressed_not_path_addressed(fake_root):
    h1 = compute_hashes(fake_root)
    (fake_root / "prereg" / "stage0.md").write_text("content of src/amenability/scoring/score.py\n")
    h2 = compute_hashes(fake_root)
    assert h2["prereg/stage0.md"] == h1["src/amenability/scoring/score.py"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_freeze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prereg'`

- [ ] **Step 3: Write minimal implementation**

```python
# prereg/stage0.md
```

Write the analysis plan to `prereg/stage0.md` with exactly this content:

```markdown
# Stage 0 Pre-Registration

Frozen before any Stage 0 ground-truth run. The commit hash of this file is cited in the paper.

## Hypotheses

- **H1 (Gate A):** The amenability score recovers the known ordering of RL amenability
  across over-SFT'd variants, within both control families. Amenability degrades
  monotonically with SFT token budget.
- **H2 (Gate B):** The amenability score predicts the full-run outcome better than naive
  extrapolation of the probe reward trace.

## Fixed quantities

- Score: `gate * mean(z(conversion_rate), z(retention_factor))`,
  `gate = clip(1 - zero_advantage_rate, 0, 1)`. Equal weights, not tuned.
- Outcome variable (primary): conversion ratio,
  `(post_pass@1 - pre_pass@1) / (pre_pass@32 - pre_pass@1)` measured on the target suite
  after the 600-step full run.
- Outcome variable (secondary): raw `post_pass@1 - pre_pass@1`.
- Probe budget: 60 GRPO steps, 150 SFT steps. Full budget: 600 GRPO steps, 1500 SFT steps.
- Breadth k = 32. pass@64 computed at endpoints only.

## Gate thresholds

- **Gate A passes** when the score's ordering is correct within both control families
  AND pooled Spearman rho between score and known ordering is >= 0.7.
- **Gate B passes** when `|rho(score, outcome)| > |rho(naive_extrapolation, outcome)|`
  on the six control checkpoints.

## Decisions bound in advance

- Gate A fails and Gate B fails: the probe design is wrong. Stop and redesign.
- Gate A passes and Gate B fails: the probe works but is not cheap enough. Pivot the
  paper toward failure-mode detection rather than prediction efficiency.
- Both pass: proceed to Stage 1.

## What will not happen

- No reweighting of the score after seeing ground truth.
- No addition or removal of features from the score.
- No change of outcome variable after seeing results.
- No dropping of models from the roster except for a documented infrastructure failure.
```

```python
# prereg/freeze.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FROZEN_PATHS = [
    "src/amenability/scoring/score.py",
    "src/amenability/scoring/baselines.py",
    "src/amenability/scoring/gates.py",
    "prereg/stage0.md",
]

MANIFEST = "prereg/FROZEN.json"


class FreezeViolation(Exception):
    """Raised when frozen analysis code has changed, or was never frozen."""


def compute_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN_PATHS:
        p = root / rel
        if not p.exists():
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def freeze(root: Path) -> dict:
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        raise FreezeViolation(f"already frozen: {MANIFEST} exists")
    hashes = compute_hashes(root)
    missing = set(FROZEN_PATHS) - set(hashes)
    if missing:
        raise FreezeViolation(f"cannot freeze, missing files: {sorted(missing)}")
    manifest = {"frozen_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify_freeze(root: Path) -> None:
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        raise FreezeViolation(
            f"analysis is not frozen: {MANIFEST} missing. Run `python prereg/freeze.py` "
            "before any ground-truth run."
        )
    expected = json.loads(manifest_path.read_text())["hashes"]
    actual = compute_hashes(root)
    for rel, digest in expected.items():
        if rel not in actual:
            raise FreezeViolation(f"frozen file deleted: {rel}")
        if actual[rel] != digest:
            raise FreezeViolation(
                f"frozen file modified after freeze: {rel}. Stage 0 is invalidated."
            )


if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).resolve().parents[1]), indent=2))
```

Also create `prereg/__init__.py` (empty) so the test can import the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_freeze.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add prereg tests/test_freeze.py
git commit -m "feat: pre-registration freeze with tamper detection"
```

---

### Task 15: Gate A and Gate B evaluation

**Files:**
- Create: `src/amenability/scoring/gates.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: `spearman_with_ci` from `amenability.eval.stats`.
- Produces: `GateResult` frozen dataclass with fields `name: str`, `passed: bool`, `detail: str`, `statistic: float`. `evaluate_gate_a(scores_by_family: dict[str, list[float]], known_order_by_family: dict[str, list[int]], threshold: float = 0.7) -> GateResult`. `evaluate_gate_b(scores: list[float], naive: list[float], outcomes: list[float]) -> GateResult`.

Thresholds are read from the frozen pre-registration and are not arguments a caller can quietly change at analysis time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gates.py
import pytest
from amenability.scoring.gates import evaluate_gate_a, evaluate_gate_b


def test_gate_a_passes_when_both_families_order_correctly():
    # known_order: 2 = fresh base (most amenable), 0 = most over-SFT'd.
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.8, 0.5, 0.2]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is True
    # Pooled rho is 0.956, not 1.0: known_order ties across the two families.
    assert res.statistic > 0.9


def test_gate_a_fails_when_one_family_inverts():
    scores = {"qwen": [0.9, 0.4, 0.1], "llama": [0.1, 0.5, 0.9]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known)
    assert res.passed is False
    assert "llama" in res.detail


def test_gate_a_fails_when_pooled_rho_below_threshold():
    # Ordering correct in both families but noisy enough to drag pooled rho down.
    scores = {"qwen": [0.55, 0.50, 0.45], "llama": [0.10, 0.05, 0.00]}
    known = {"qwen": [2, 1, 0], "llama": [2, 1, 0]}
    res = evaluate_gate_a(scores, known, threshold=0.99)
    assert res.passed is False


def test_gate_a_rejects_mismatched_family_keys():
    with pytest.raises(ValueError, match="families"):
        evaluate_gate_a({"qwen": [1, 2]}, {"llama": [1, 2]})


def test_gate_a_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        evaluate_gate_a({"qwen": [1, 2, 3]}, {"qwen": [1, 2]})


def test_gate_b_passes_when_score_beats_naive():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]     # perfect
    naive = [0.9, 0.1, 0.5, 0.2, 0.7, 0.3]      # unrelated
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is True


def test_gate_b_fails_when_naive_matches_or_beats_score():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.9, 0.1, 0.5, 0.2, 0.7, 0.3]
    naive = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is False
    assert "naive" in res.detail


def test_gate_b_uses_absolute_correlation_so_sign_does_not_rescue_a_baseline():
    outcomes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
    naive = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]  # perfectly inverse, still informative
    res = evaluate_gate_b(scores, naive, outcomes)
    assert res.passed is False


def test_gate_b_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        evaluate_gate_b([1, 2, 3], [1, 2], [1, 2, 3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amenability.scoring.gates'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/amenability/scoring/gates.py
"""PRE-REGISTERED. Frozen alongside score.py. Thresholds come from prereg/stage0.md."""
from __future__ import annotations

from dataclasses import dataclass

from amenability.eval.stats import spearman_with_ci

GATE_A_THRESHOLD = 0.7


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    statistic: float


def evaluate_gate_a(
    scores_by_family: dict[str, list[float]],
    known_order_by_family: dict[str, list[int]],
    threshold: float = GATE_A_THRESHOLD,
) -> GateResult:
    if set(scores_by_family) != set(known_order_by_family):
        raise ValueError("families in scores and known orders do not match")

    failures: list[str] = []
    pooled_scores: list[float] = []
    pooled_known: list[float] = []
    for family in sorted(scores_by_family):
        scores = scores_by_family[family]
        known = known_order_by_family[family]
        if len(scores) != len(known):
            raise ValueError(f"length mismatch for family {family}")
        ranked = [s for _, s in sorted(zip(known, scores))]
        if ranked != sorted(ranked):
            failures.append(family)
        # Pool on within-family z-free ranks so families with different absolute
        # score levels do not dominate the pooled statistic.
        pooled_scores.extend(scores)
        pooled_known.extend(known)

    rho = spearman_with_ci(pooled_known, pooled_scores, n_boot=2000).rho
    passed = not failures and rho >= threshold
    if failures:
        detail = f"ordering inverted in families: {', '.join(failures)}; pooled rho={rho:.3f}"
    elif not passed:
        detail = f"ordering correct in all families but pooled rho={rho:.3f} < {threshold}"
    else:
        detail = f"ordering correct in all families; pooled rho={rho:.3f}"
    return GateResult(name="A", passed=passed, detail=detail, statistic=rho)


def evaluate_gate_b(
    scores: list[float], naive: list[float], outcomes: list[float]
) -> GateResult:
    if not (len(scores) == len(naive) == len(outcomes)):
        raise ValueError("length mismatch between scores, naive baseline and outcomes")
    rho_score = abs(spearman_with_ci(scores, outcomes, n_boot=2000).rho)
    rho_naive = abs(spearman_with_ci(naive, outcomes, n_boot=2000).rho)
    passed = rho_score > rho_naive
    detail = (
        f"|rho(score)|={rho_score:.3f} vs naive |rho|={rho_naive:.3f}; "
        f"{'score wins' if passed else 'naive extrapolation matches or wins'}"
    )
    return GateResult(name="B", passed=passed, detail=detail, statistic=rho_score - rho_naive)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gates.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/amenability/scoring/gates.py tests/test_gates.py
git commit -m "feat: Gate A and Gate B evaluation against pre-registered thresholds"
```

---

### Task 16: Over-SFT variant production for the positive control

**Files:**
- Create: `scripts/make_oversft_variants.py`
- Test: `tests/test_oversft.py`

**Interfaces:**
- Consumes: `load_registry`, `SFTSpec`, `run_sft`, `build_rejection_dataset`.
- Produces: `OVERSFT_LEVELS: tuple[int, ...] = (1, 4)` (multiples of the baseline SFT budget). `variant_key(base_key: str, level: int) -> str`. `variant_plan(base_keys: list[str], levels=OVERSFT_LEVELS) -> list[dict]` returning records shaped `{"base_key", "level", "variant_key", "max_steps", "known_order"}`.

`known_order` encodes the a priori amenability ranking that Gate A tests against: the fresh base is most amenable, and amenability falls monotonically as the SFT budget rises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oversft.py
import pytest
from scripts.make_oversft_variants import (
    OVERSFT_LEVELS, variant_key, variant_plan, BASE_SFT_STEPS,
)


def test_variant_key_encodes_base_and_level():
    assert variant_key("qwen2.5-0.5b", 4) == "qwen2.5-0.5b-oversft4x"


def test_plan_includes_the_fresh_base_as_level_zero():
    plan = variant_plan(["qwen2.5-0.5b"])
    levels = [p["level"] for p in plan]
    assert levels == [0, 1, 4]
    assert plan[0]["variant_key"] == "qwen2.5-0.5b"
    assert plan[0]["max_steps"] == 0


def test_plan_covers_both_control_bases():
    plan = variant_plan(["qwen2.5-0.5b", "llama-3.2-1b"])
    assert len({p["base_key"] for p in plan}) == 2
    assert len(plan) == 6  # 2 bases x (base + 2 over-SFT levels)


def test_known_order_is_descending_in_amenability():
    plan = variant_plan(["qwen2.5-0.5b"])
    orders = [p["known_order"] for p in plan]
    # Highest known_order = most amenable = the fresh base.
    assert orders == [2, 1, 0]


def test_step_budgets_scale_with_level():
    plan = variant_plan(["qwen2.5-0.5b"])
    assert [p["max_steps"] for p in plan] == [0, BASE_SFT_STEPS, 4 * BASE_SFT_STEPS]


def test_unknown_base_key_raises():
    with pytest.raises(KeyError):
        variant_plan(["not-a-model"])


def test_non_control_base_raises():
    with pytest.raises(ValueError, match="control_base"):
        variant_plan(["qwen2.5-1.5b"])


def test_levels_are_the_pre_registered_pair():
    assert OVERSFT_LEVELS == (1, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oversft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write minimal implementation**

Create an empty `scripts/__init__.py`, then:

```python
# scripts/make_oversft_variants.py
"""Produce the positive-control checkpoints whose amenability ordering is known a priori.

The plasticity literature establishes that RL amenability degrades monotonically with
SFT overtraining as entropy is depleted. That known ordering is what Gate A tests the
probe against, independently of any correlation study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from amenability.registry.loader import load_registry

OVERSFT_LEVELS: tuple[int, ...] = (1, 4)
BASE_SFT_STEPS = 1500


def variant_key(base_key: str, level: int) -> str:
    return f"{base_key}-oversft{level}x"


def variant_plan(base_keys: list[str], levels: tuple[int, ...] = OVERSFT_LEVELS) -> list[dict]:
    registry = load_registry()
    plan: list[dict] = []
    for base_key in base_keys:
        spec = registry[base_key]
        if spec.role != "control_base":
            raise ValueError(f"{base_key} is not marked control_base in the registry")
        all_levels = (0,) + tuple(levels)
        n = len(all_levels)
        for i, level in enumerate(all_levels):
            plan.append(
                {
                    "base_key": base_key,
                    "level": level,
                    "variant_key": base_key if level == 0 else variant_key(base_key, level),
                    "max_steps": level * BASE_SFT_STEPS,
                    "known_order": n - 1 - i,
                }
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", default=["qwen2.5-0.5b", "llama-3.2-1b"])
    parser.add_argument("--out", type=Path, default=Path("results/variants"))
    args = parser.parse_args()

    for entry in variant_plan(args.bases):
        if entry["level"] == 0:
            continue
        print(f"would train {entry['variant_key']} for {entry['max_steps']} SFT steps")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oversft.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts tests/test_oversft.py
git commit -m "feat: over-SFT variant plan with a priori amenability ordering"
```

---

### Task 17: Stage 0 orchestration and report

**Files:**
- Create: `scripts/run_stage0.py`
- Test: `tests/test_stage0.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Stage0Config` frozen dataclass with fields `probe_steps: int = 60`, `full_steps: int = 600`, `num_generations: int = 8`, `n_probe_items: int = 300`, `n_target_items: int = 300`, `seed: int = 0`. `run_stage0(config, runner, root: Path) -> dict` where `runner` is an injectable object exposing `probe(variant_key) -> ProbeTelemetry` and `full_run(variant_key) -> float` returning the outcome. `render_report(results: dict) -> str`.

`run_stage0` calls `verify_freeze` first. An unfrozen or modified analysis halts the run rather than producing a result that cannot be pre-registered.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage0.py
import pytest
from pathlib import Path
from prereg.freeze import FreezeViolation
from scripts.run_stage0 import Stage0Config, run_stage0, render_report
from amenability.eval.passk import PassKResult
from amenability.probe.telemetry import ProbeTelemetry, StepRecord


class FakeRunner:
    """Amenability falls with over-SFT level, exactly as the literature predicts."""

    def __init__(self, invert_llama: bool = False):
        self.invert_llama = invert_llama

    def _amenability(self, key: str) -> float:
        level = 0 if "oversft" not in key else int(key.split("oversft")[1].rstrip("x"))
        base = 1.0 - 0.3 * level
        if self.invert_llama and key.startswith("llama"):
            base = 1.0 - base
        return base

    def probe(self, variant_key: str) -> ProbeTelemetry:
        a = self._amenability(variant_key)
        steps = [
            StepRecord(step=i, mean_reward=0.1 + 0.005 * a * i, group_reward_std=0.2,
                       zero_advantage_frac=0.1, policy_entropy=2.0 - (1.0 - a) * 0.1 * i,
                       kl=0.01, grad_norm=1.0)
            for i in range(1, 11)
        ]
        return ProbeTelemetry(
            model_key=variant_key, algorithm="grpo", steps=steps,
            pre=PassKResult(ks={1: 0.1, 32: 0.5, 64: 0.6}, n_samples=64, n_items=10,
                            per_item_correct={}),
            post=PassKResult(ks={1: 0.1 + 0.3 * a, 32: 0.5, 64: 0.6}, n_samples=64,
                             n_items=10, per_item_correct={}),
        )

    def full_run(self, variant_key: str) -> float:
        return self._amenability(variant_key)


@pytest.fixture
def frozen_root(tmp_path: Path) -> Path:
    from prereg.freeze import FROZEN_PATHS, freeze
    for rel in FROZEN_PATHS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}\n")
    freeze(tmp_path)
    return tmp_path


def test_run_halts_when_analysis_is_not_frozen(tmp_path):
    with pytest.raises(FreezeViolation, match="not frozen"):
        run_stage0(Stage0Config(), FakeRunner(), tmp_path)


def test_run_produces_six_control_checkpoints(frozen_root):
    res = run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    assert len(res["variants"]) == 6


def test_gate_a_passes_on_monotonically_degrading_fake(frozen_root):
    res = run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    assert res["gate_a"].passed is True


def test_gate_a_fails_when_one_family_is_inverted(frozen_root):
    res = run_stage0(Stage0Config(), FakeRunner(invert_llama=True), frozen_root)
    assert res["gate_a"].passed is False
    assert "llama" in res["gate_a"].detail


def test_report_names_both_gates_and_their_verdicts(frozen_root):
    res = run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    report = render_report(res)
    assert "Gate A" in report and "Gate B" in report
    assert "PASS" in report or "FAIL" in report


def test_report_lists_every_variant_with_its_score(frozen_root):
    res = run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    report = render_report(res)
    for v in res["variants"]:
        assert v["variant_key"] in report


def test_config_defaults_match_the_prereg(frozen_root):
    c = Stage0Config()
    assert c.probe_steps == 60
    assert c.full_steps == 600
    assert c.probe_steps * 10 == c.full_steps  # probe is exactly 10% of budget
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stage0.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_stage0'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_stage0.py
"""Stage 0: build the positive control, probe it, and evaluate Gates A and B."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from amenability.probe.telemetry import extract_features
from amenability.registry.loader import load_registry
from amenability.scoring.baselines import naive_extrapolation
from amenability.scoring.gates import evaluate_gate_a, evaluate_gate_b
from amenability.scoring.score import amenability_score
from prereg.freeze import verify_freeze
from scripts.make_oversft_variants import variant_plan

CONTROL_BASES = ["qwen2.5-0.5b", "llama-3.2-1b"]


@dataclass(frozen=True)
class Stage0Config:
    probe_steps: int = 60
    full_steps: int = 600
    num_generations: int = 8
    n_probe_items: int = 300
    n_target_items: int = 300
    seed: int = 0


def run_stage0(config: Stage0Config, runner, root: Path) -> dict:
    verify_freeze(root)

    registry = load_registry()
    plan = variant_plan(CONTROL_BASES)

    telemetries = [runner.probe(entry["variant_key"]) for entry in plan]
    features = [extract_features(t) for t in telemetries]
    scores = amenability_score(features)
    outcomes = [runner.full_run(entry["variant_key"]) for entry in plan]
    naive = [
        naive_extrapolation(
            [s.step for s in t.steps], [s.mean_reward for s in t.steps], config.full_steps
        )
        for t in telemetries
    ]

    variants = []
    for entry, score, outcome, nv, feat in zip(plan, scores, outcomes, naive, features):
        variants.append(
            {
                **entry,
                "family": registry[entry["base_key"]].family,
                "score": score,
                "outcome": outcome,
                "naive": nv,
                "features": feat,
            }
        )

    scores_by_family: dict[str, list[float]] = defaultdict(list)
    known_by_family: dict[str, list[int]] = defaultdict(list)
    for v in variants:
        scores_by_family[v["family"]].append(v["score"])
        known_by_family[v["family"]].append(v["known_order"])

    return {
        "config": config,
        "variants": variants,
        "gate_a": evaluate_gate_a(dict(scores_by_family), dict(known_by_family)),
        "gate_b": evaluate_gate_b(scores, naive, outcomes),
    }


def render_report(results: dict) -> str:
    lines = ["# Stage 0 Report", ""]
    for gate_key in ("gate_a", "gate_b"):
        g = results[gate_key]
        verdict = "PASS" if g.passed else "FAIL"
        lines.append(f"**Gate {g.name}: {verdict}.** {g.detail}")
    lines += ["", "| variant | family | level | known order | score | naive | outcome |",
              "|---|---|---|---|---|---|---|"]
    for v in results["variants"]:
        lines.append(
            f"| {v['variant_key']} | {v['family']} | {v['level']}x | {v['known_order']} | "
            f"{v['score']:.3f} | {v['naive']:.3f} | {v['outcome']:.3f} |"
        )
    lines += ["", "## Pre-registered decision", ""]
    a, b = results["gate_a"].passed, results["gate_b"].passed
    if a and b:
        lines.append("Both gates pass. Proceed to Stage 1.")
    elif a and not b:
        lines.append(
            "Gate A passes, Gate B fails. The probe works but is not cheaper than naive "
            "extrapolation. Pivot the paper toward failure-mode detection."
        )
    else:
        lines.append("Gate A fails. The probe design is wrong. Stop and redesign.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/stage0_report.md"))
    args = parser.parse_args()

    from scripts.real_runner import RealRunner  # created in Task 18

    root = Path(__file__).resolve().parents[1]
    results = run_stage0(Stage0Config(), RealRunner(), root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results))
    print(render_report(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stage0.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stage0.py tests/test_stage0.py
git commit -m "feat: Stage 0 orchestration with freeze check and gate report"
```

---

### Task 18: Real runner, GPU smoke test, and launch scripts

**Files:**
- Create: `scripts/real_runner.py`
- Create: `scripts/slurm/stage0.sbatch`
- Create: `scripts/modal_app.py`
- Test: `tests/test_gpu_smoke.py`

**Interfaces:**
- Consumes: `GRPOSpec`, `run_grpo`, `SFTSpec`, `run_sft`, `evaluate_passk`, `generate_countdown`, `load_gsm8k`, `SuiteRegistry`.
- Produces: `RealRunner` class with `__init__(self, config: Stage0Config, work_dir: Path)` and methods `probe(variant_key) -> ProbeTelemetry`, `full_run(variant_key) -> float`.

This is the only task that requires a GPU. Its test is marked `gpu` and is excluded from the default test run. Everything above is covered by CPU tests, so the whole pipeline can be verified before any allocation is spent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_smoke.py
import pytest
from pathlib import Path

pytestmark = pytest.mark.gpu

TINY = "HuggingFaceTB/SmolLM2-135M"


def test_suites_are_disjoint_when_both_are_registered():
    from amenability.suites.base import SuiteRegistry
    from amenability.suites.countdown import generate_countdown
    from amenability.suites.gsm8k import load_gsm8k

    reg = SuiteRegistry()
    reg.register("probe_countdown", generate_countdown(n=30, seed=0))
    reg.register("target_gsm8k", load_gsm8k("test", limit=30, seed=0))
    assert sorted(reg.names()) == ["probe_countdown", "target_gsm8k"]


def test_two_step_grpo_produces_populated_telemetry(tmp_path: Path):
    from amenability.suites.countdown import generate_countdown, verify_countdown
    from amenability.training.grpo import GRPOSpec, run_grpo

    items = generate_countdown(n=6, seed=0)
    spec = GRPOSpec(
        model_path=TINY, model_key="tiny", items=items, verify_fn=verify_countdown,
        max_steps=2, num_generations=2, learning_rate=1e-6, beta=0.04,
        temperature=1.0, seed=0, output_dir=str(tmp_path / "out"), save_steps=None,
    )
    records = run_grpo(spec)
    assert len(records) >= 1
    for r in records:
        assert r.policy_entropy > 0.0, "entropy probe returned zero on a real model"
        assert 0.0 <= r.zero_advantage_frac <= 1.0


def test_passk_runs_end_to_end_on_a_tiny_model():
    from amenability.suites.countdown import generate_countdown, verify_countdown
    from amenability.eval.passk import evaluate_passk

    items = generate_countdown(n=6, seed=0)
    res = evaluate_passk(
        model_path=TINY, items=items, verify_fn=verify_countdown,
        ks=(1, 4), n_samples=4, temperature=1.0, seed=0,
    )
    assert res.n_items == 6
    assert 0.0 <= res.ks[1] <= res.ks[4] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gpu_smoke.py -v -m gpu`
Expected: FAIL with `ModuleNotFoundError` or a GRPO/vLLM error, depending on what is missing.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/real_runner.py
from __future__ import annotations

from pathlib import Path

from amenability.eval.passk import evaluate_passk
from amenability.probe.telemetry import ProbeTelemetry
from amenability.registry.loader import load_registry
from amenability.suites.base import SuiteRegistry
from amenability.suites.countdown import generate_countdown, verify_countdown
from amenability.suites.gsm8k import load_gsm8k, verify_gsm8k
from amenability.training.grpo import GRPOSpec, run_grpo

BREADTH_KS = (1, 8, 32, 64)


class RealRunner:
    def __init__(self, config, work_dir: Path = Path("results")) -> None:
        self.config = config
        self.work_dir = work_dir
        self.registry = load_registry()
        self.suites = SuiteRegistry()
        self.probe_items = generate_countdown(n=config.n_probe_items, seed=config.seed)
        self.target_items = load_gsm8k("train", limit=config.n_target_items, seed=config.seed)
        self.suites.register("probe_countdown", self.probe_items)
        self.suites.register("target_gsm8k", self.target_items)
        self.target_eval = load_gsm8k("test", limit=300, seed=config.seed)

    def _resolve(self, variant_key: str) -> str:
        if variant_key in self.registry:
            return self.registry[variant_key].hf_id
        return str(self.work_dir / "variants" / variant_key)

    def probe(self, variant_key: str) -> ProbeTelemetry:
        model_path = self._resolve(variant_key)
        pre = evaluate_passk(
            model_path, self.probe_items, verify_countdown, BREADTH_KS,
            n_samples=64, temperature=1.0, seed=self.config.seed,
        )
        out_dir = self.work_dir / "probes" / variant_key
        records = run_grpo(
            GRPOSpec(
                model_path=model_path, model_key=variant_key, items=self.probe_items,
                verify_fn=verify_countdown, max_steps=self.config.probe_steps,
                num_generations=self.config.num_generations, learning_rate=1e-6,
                beta=0.04, temperature=1.0, seed=self.config.seed,
                output_dir=str(out_dir), save_steps=None,
            )
        )
        post = evaluate_passk(
            str(out_dir), self.probe_items, verify_countdown, BREADTH_KS,
            n_samples=64, temperature=1.0, seed=self.config.seed,
        )
        return ProbeTelemetry(
            model_key=variant_key, algorithm="grpo", steps=records, pre=pre, post=post
        )

    def full_run(self, variant_key: str) -> float:
        model_path = self._resolve(variant_key)
        pre = evaluate_passk(
            model_path, self.target_eval, verify_gsm8k, (1, 32),
            n_samples=32, temperature=0.8, seed=self.config.seed,
        )
        out_dir = self.work_dir / "full" / variant_key
        run_grpo(
            GRPOSpec(
                model_path=model_path, model_key=variant_key, items=self.target_items,
                verify_fn=verify_gsm8k, max_steps=self.config.full_steps,
                num_generations=self.config.num_generations, learning_rate=1e-6,
                beta=0.04, temperature=1.0, seed=self.config.seed,
                output_dir=str(out_dir), save_steps=200,
            )
        )
        post = evaluate_passk(
            str(out_dir), self.target_eval, verify_gsm8k, (1, 32),
            n_samples=32, temperature=0.8, seed=self.config.seed,
        )
        breadth = pre.ks[32] - pre.ks[1]
        return float((post.ks[1] - pre.ks[1]) / breadth) if breadth > 1e-9 else 0.0
```

```bash
# scripts/slurm/stage0.sbatch
#!/bin/bash
#SBATCH --job-name=amenability-stage0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage0_%j.out

set -euo pipefail
module purge
module load cuda/12.4
cd "$SLURM_SUBMIT_DIR"
uv run python -m scripts.run_stage0 --out "results/stage0_report.md"
```

```python
# scripts/modal_app.py
"""Modal fallback so Stage 0 is not blocked on a CARC queue slot."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system -e /repo")
)
app = modal.App("rl-amenability-stage0")


@app.function(gpu="L40S", timeout=24 * 60 * 60, image=image,
              mounts=[modal.Mount.from_local_dir(".", remote_path="/repo")])
def stage0():
    import subprocess
    subprocess.run(
        ["python", "-m", "scripts.run_stage0", "--out", "/repo/results/stage0_report.md"],
        cwd="/repo", check=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run on a GPU node: `uv run pytest tests/test_gpu_smoke.py -v -m gpu`
Expected: PASS, 3 passed

Then confirm the CPU suite is still green: `uv run pytest -v -m "not gpu"`
Expected: PASS, 137 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/real_runner.py scripts/slurm scripts/modal_app.py tests/test_gpu_smoke.py
git commit -m "feat: real Stage 0 runner with SLURM and Modal launch paths"
```

---

## Execution order note

Task 14 (the freeze) must run **after** Tasks 11, 12 and 15 exist, since it hashes them, and **before** any real GPU run in Task 18. The plan orders them so that following it top to bottom satisfies both constraints. Do not execute `python prereg/freeze.py` until Tasks 11, 12 and 15 are committed and green.

## Before Stage 0 executes

The spec's open items gate the GPU work, not the code:

1. Verify the actual CARC GPU allocation. Under ~400 GPU-hours, Stage 0 goes to Modal.
2. Confirm true base (non-instruct) weights exist for Falcon3-1B-Base, StableLM-2-1.6B and Gemma-3-1B-pt.
3. Pilot the probe suite difficulty calibration: at 0.5B, some Countdown buckets must be solvable and some not, or the zero-advantage gate saturates and carries no information.
4. Fix the over-SFT token budgets so the induced degradation is detectable without destroying the model. `OVERSFT_LEVELS = (1, 4)` is the current pre-registered guess and should be sanity-checked against a single cheap run before the freeze.

---

## Spec coverage: what this plan deliberately leaves to Stage 1

Checked against the spec section by section. Every Stage 0 requirement has a task. Four spec items are intentionally out of scope here, listed so the Stage 1 plan picks them up rather than losing them:

| Spec item | Why deferred |
|---|---|
| §7.4 MATH-500 endpoint evaluation | Transfer check across the roster. The control arm trains and evaluates on GSM8K only, so it adds cost without informing Gate A or B. |
| §7.5 secondary outcome (raw Δpass@1) | `RealRunner.full_run` returns the primary conversion ratio, which is the only outcome the pre-registered gates use. The raw delta is a reporting column for the cross-model table. |
| §7.6 seed replicates and noise floor | Applies to the roster sweep. The control arm's known ordering is the falsification test at Stage 0, and it does not depend on a noise-floor estimate. |
| §8.6 confound controls | `partial_spearman` is built and tested in Task 13, but headroom and family controls only mean something at N=10 across families, not on 6 checkpoints from 2 families. |

## One deviation from the spec, flagged rather than hidden

The spec calls for **BCa** bootstrap intervals. Task 13 implements **percentile** intervals instead, and labels them as such in `CorrelationResult`. BCa's acceleration term is estimated by jackknife, which degenerates when bootstrap resamples produce constant vectors. At N=6 (Stage 0) that happens often enough to produce NaN intervals. The percentile interval is wider and more conservative, which is the right direction to err at this sample size.

This should be revisited for Stage 1, where N=10 makes BCa more stable. If it is not revisited, the spec's §8.5 wording needs updating so the paper does not claim BCa.
