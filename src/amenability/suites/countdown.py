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
