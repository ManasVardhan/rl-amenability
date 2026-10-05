"""Difficulty calibration of the probe training pools (transfer-rulings T22).

GRPO with a group of 8 gets a gradient from a prompt only if at least one of
its 8 rollouts is correct. The calibration samples 8 completions per item with
the GRPO rollout sampling, for every difficulty candidate
(`amenability.suites.difficulty`), and reports per bucket and over the pool:

- pass@1 (mean fraction correct) and pass@8 (the unbiased estimator from the 8
  samples; with n = k = 8 it is exactly the fraction of items with at least one
  correct sample);
- the group-signal rate, the fraction of items with at least one correct of 8,
  i.e. the expected share of GRPO groups with a nonzero advantage. Numerically
  equal to pass@8 here; reported under its own name because it is the quantity
  the selection rule reads;
- a failure taxonomy over all samples (CATEGORIES), which separates answer-format
  failures from search failures;
- the shaped group-signal rate (transfer-rulings T23): the fraction of items
  whose 8 shaped TRAINING rewards (1.0 correct, 0.1 answer block, 0.0 none;
  CATEGORY_REWARD) are not all equal, i.e. the share of GRPO groups with a
  nonzero advantage under the shaped reward. Computed from the per-item
  category counts, which are saved as `per_item_categories`.

`select_candidate` is the pre-registered selection rule, applied mechanically:
on the strict group-signal rate (T22) or on the shaped one (T23).
"""
from __future__ import annotations

from amenability.eval.passk import pass_at_k
from amenability.suites.base import ANSWER_STOP, TaskItem, first_answer_block, truncate_at_stop
from amenability.suites.countdown import (
    _strip_trailing_target,
    extract_expression,
    parse_expression,
    strip_any_trailing_equals,
    verify_countdown,
)
from amenability.suites.graphpath import extract_path, verify_graphpath
from amenability.training.grpo import (
    MAX_COMPLETION_LENGTH,
    ROLLOUT_MIN_P,
    ROLLOUT_REPETITION_PENALTY,
    ROLLOUT_TOP_K,
    ROLLOUT_TOP_P,
    generation_kwargs_for,
)
from amenability.training.reward import CORRECT_SCORE, FORMAT_SCORE, NO_ANSWER_SCORE

N_SAMPLES = 8  # the GRPO group size, ProbeConfig.num_generations

# no_answer_tag: no closed <answer>...</answer> block (including a completion cut
#   at the token cap before </answer>).
# unparseable_answer: a block is present but is not a well-formed answer: not an
#   arithmetic expression in the task's grammar (Countdown; one trailing
#   "= <integer>" is allowed), or not node letters joined by -> (Graph path).
#   Prose inside the tag lands here.
# parseable_wrong: a well-formed answer that the verifier rejects: wrong numbers
#   or value, a wrong right-hand side, a division by zero (Countdown); wrong
#   endpoints, a non-edge, a repeat or a single node (Graph path).
# correct: the suite's verifier accepts it.
CATEGORIES = ("no_answer_tag", "unparseable_answer", "parseable_wrong", "correct")
FORMAT_FAILURES = ("no_answer_tag", "unparseable_answer")

# The shaped training reward (transfer-rulings T23) of each category: an answer
# block the verifier rejects, parseable or not, earns the format score. A test
# checks this equals amenability.training.reward.shaped_reward per completion.
CATEGORY_REWARD = {
    "no_answer_tag": NO_ANSWER_SCORE,
    "unparseable_answer": FORMAT_SCORE,
    "parseable_wrong": FORMAT_SCORE,
    "correct": CORRECT_SCORE,
}

# Which group-signal rate the selection rule gates on.
STRICT_SIGNAL = "group_signal_rate"            # T22: >= 1 correct of 8
SHAPED_SIGNAL = "shaped_group_signal_rate"     # T23: 8 shaped rewards not all equal

# The selection rule (T22). Thresholds are inclusive.
MIN_GROUP_SIGNAL = 0.15
MAX_BEST_PASS1 = 0.80
FORMAT_SHARE_THRESHOLD = 0.5

# Floored on the suite in the calibration pilot (pass@32 < 0.02 in every bucket;
# issue #21, job 12630546): only StableLM-2-1.6B, on both suites. Pinned here
# because the rule was registered against it.
PILOT_FLOORED = {
    "countdown": frozenset({"stablelm-2-1.6b"}),
    "graphpath": frozenset({"stablelm-2-1.6b"}),
}

REMEDIES = (
    "partial format reward as in TinyZero (0.1 for a well-formed wrong answer)",
    "a larger GRPO group size",
    "accepting zero-signal models (they tie near score 0)",
)
# After T23 the format reward is in place, so it is not offered again.
REMEDIES_SHAPED = (
    "a larger GRPO group size",
    "accepting zero-signal models (they tie near score 0)",
)


def rollout_sampling_kwargs(n: int, seed: int, stop: str | None = ANSWER_STOP) -> dict:
    """vLLM SamplingParams for the calibration: the GRPO rollout sampling
    (temperature 1.0, no top-p/top-k/min-p filter, no repetition penalty, the
    768-token cap, the suite stop kept in the output). The one difference is
    mechanical: TRL submits each prompt 8 times with n=1 and no seed, the
    calibration submits it once with n=8 and a seed, for reproducibility. The
    distribution sampled from is the same."""
    from amenability.probe.run import ProbeConfig

    kw = {
        "n": n,
        "temperature": ProbeConfig.temperature,
        "top_p": ROLLOUT_TOP_P,
        "top_k": ROLLOUT_TOP_K,
        "min_p": ROLLOUT_MIN_P,
        "repetition_penalty": ROLLOUT_REPETITION_PENALTY,
        "max_tokens": MAX_COMPLETION_LENGTH,
        "seed": seed,
    }
    kw.update(generation_kwargs_for(stop) or {})
    return kw


def _classify_countdown(item: TaskItem, completion: str) -> str:
    expr = extract_expression(completion)
    if expr is None:
        return "no_answer_tag"
    if verify_countdown(item, completion):
        return "correct"
    target = int(item.answer.split("|")[1])
    lhs = _strip_trailing_target(expr, target)
    if lhs is None:
        lhs = strip_any_trailing_equals(expr)
    if lhs is None or parse_expression(lhs) is None:
        return "unparseable_answer"
    return "parseable_wrong"


def _classify_graphpath(item: TaskItem, completion: str) -> str:
    if first_answer_block(completion) is None:
        return "no_answer_tag"
    if verify_graphpath(item, completion):
        return "correct"
    if extract_path(completion) is None:
        return "unparseable_answer"
    return "parseable_wrong"


_CLASSIFIERS = {"countdown": _classify_countdown, "graphpath": _classify_graphpath}


def classify_completion(
    suite_key: str, item: TaskItem, completion: str, stop: str | None = ANSWER_STOP
) -> str:
    """One of CATEGORIES. The completion is truncated at the stop first, as every
    reward and pass@k is (T20). "correct" if and only if the suite's own verifier
    accepts the truncated completion."""
    return _CLASSIFIERS[suite_key](item, truncate_at_stop(completion, stop))


def _block_stats(cats_per_item: list[list[str]]) -> dict:
    n_items = len(cats_per_item)
    n_samples = sum(len(c) for c in cats_per_item)
    correct = [sum(1 for x in cats if x == "correct") for cats in cats_per_item]
    counts = {cat: sum(cats.count(cat) for cats in cats_per_item) for cat in CATEGORIES}
    taxonomy = {cat: counts[cat] / n_samples for cat in CATEGORIES}
    return {
        "n_items": n_items,
        "n_samples": n_samples,
        "pass@1": sum(c / len(cats) for c, cats in zip(correct, cats_per_item)) / n_items,
        "pass@8": sum(pass_at_k(len(cats), c, N_SAMPLES)
                      for c, cats in zip(correct, cats_per_item)) / n_items,
        "group_signal_rate": sum(1 for c in correct if c > 0) / n_items,
        SHAPED_SIGNAL: sum(1 for cats in cats_per_item
                           if len({CATEGORY_REWARD[x] for x in cats}) > 1) / n_items,
        "taxonomy_counts": counts,
        "taxonomy": taxonomy,
        "format_failure_share": sum(taxonomy[c] for c in FORMAT_FAILURES),
    }


def _spread(xs: list, k: int) -> list:
    """Up to k elements spread evenly over xs (deterministic)."""
    if len(xs) <= k:
        return list(xs)
    return [xs[(i * len(xs)) // k] for i in range(k)]


def summarize_candidate(
    suite_key: str,
    items: list[TaskItem],
    completions: list[list[str]],
    n_examples: int = 5,
    stop: str | None = ANSWER_STOP,
) -> dict:
    """Per-bucket and pool statistics for one candidate's items, from N_SAMPLES
    completions per item, plus up to n_examples raw completions per category."""
    if len(completions) != len(items):
        raise ValueError(f"{len(completions)} completion lists for {len(items)} items")
    cats_per_item: list[list[str]] = []
    examples: dict[str, list] = {cat: [] for cat in CATEGORIES}
    by_bucket: dict[int, list[list[str]]] = {}
    per_item_correct: dict[str, int] = {}
    per_item_categories: dict[str, dict[str, int]] = {}
    for it, comps in zip(items, completions):
        if len(comps) != N_SAMPLES:
            raise ValueError(f"{len(comps)} samples for {it.task_id}, expected {N_SAMPLES}")
        cats = [classify_completion(suite_key, it, c, stop) for c in comps]
        cats_per_item.append(cats)
        by_bucket.setdefault(it.difficulty, []).append(cats)
        per_item_correct[it.task_id] = cats.count("correct")
        per_item_categories[it.task_id] = {cat: cats.count(cat) for cat in CATEGORIES}
        for c, cat in zip(comps, cats):
            examples[cat].append(
                {"task_id": it.task_id, "bucket": it.difficulty,
                 "completion": truncate_at_stop(c, stop)}
            )
    return {
        "by_bucket": {str(b): _block_stats(v) for b, v in sorted(by_bucket.items())},
        "pool": _block_stats(cats_per_item),
        "per_item_correct": per_item_correct,
        # Per item, the count of each category over its 8 samples (T23): enough
        # to recompute either reward's group signal without the completions.
        "per_item_categories": per_item_categories,
        "examples": {cat: _spread(v, n_examples) for cat, v in examples.items()},
    }


def select_candidate(
    results: dict[str, dict[str, dict]],
    order: list[tuple[str, bool]],
    floored: set[str] | frozenset[str],
    signal_key: str = STRICT_SIGNAL,
) -> dict:
    """The T22 selection rule for one suite, or with signal_key=SHAPED_SIGNAL the
    T23 rule: identical except that (a) reads the shaped group-signal rate. The
    strict rate is reported per model and candidate (correct_signal), not gated.

    results: model -> candidate name -> summary (only "pool" is read).
    order: (candidate, guesser-eligible) from hardest to easiest.
    floored: models floored on this suite in the pilot; exempt from (a).

    Walk the eligible candidates from hardest to easiest and select the first
    with (a) every non-floored model's pool signal rate >= 0.15 and (b) the best
    model's pool pass@1 <= 0.80. A model with no result for a candidate, or a
    result without the signal_key field (a calibration written before T23),
    fails (a) for it. If none is selected the batch does not launch; when no
    candidate meets (a), every model failing (a) on the easiest eligible
    candidate is diagnosed as format (no_answer_tag + unparseable_answer >= 50%
    of its samples) or search.
    """
    models = sorted(results)
    required = [m for m in models if m not in floored]
    rows = []
    selected = None
    for name, eligible in order:
        failing, missing, pass1s, correct_signal = [], [], [], {}
        for m in models:
            pool = results[m].get(name, {}).get("pool")
            if pool is None or signal_key not in pool:
                if m in required:
                    missing.append(m)
                if pool is None:
                    continue
            pass1s.append(pool["pass@1"])
            correct_signal[m] = pool.get(STRICT_SIGNAL)
            if signal_key not in pool:
                continue
            if m in required and pool[signal_key] < MIN_GROUP_SIGNAL:
                failing.append(m)
        best = max(pass1s) if pass1s else None
        meets_a = not failing and not missing
        meets_b = best is not None and best <= MAX_BEST_PASS1
        rows.append({
            "candidate": name, "eligible": eligible, "failing_models": failing,
            "missing_models": missing, "best_pass1": best,
            "correct_signal": correct_signal,
            "meets_signal": meets_a, "meets_headroom": meets_b,
        })
        if selected is None and eligible and meets_a and meets_b:
            selected = name

    out = {"selected": selected, "launch": selected is not None, "candidates": rows,
           "signal_key": signal_key,
           "models": models,
           "floored_exempt": sorted(floored), "reason": None,
           "diagnosis_candidate": None, "diagnosis": [], "remedies": []}
    if selected is not None:
        return out
    eligible_rows = [r for r in rows if r["eligible"]]
    if any(r["meets_signal"] for r in eligible_rows):
        out["reason"] = (
            f"candidates meeting the group-signal condition all have best pass@1 > "
            f"{MAX_BEST_PASS1}"
        )
        return out
    what = "shaped group signal" if signal_key == SHAPED_SIGNAL else "group signal"
    out["reason"] = (
        f"no candidate gives every non-floored model {what} >= {MIN_GROUP_SIGNAL}"
    )
    out["remedies"] = list(REMEDIES_SHAPED if signal_key == SHAPED_SIGNAL else REMEDIES)
    if eligible_rows:
        easiest = eligible_rows[-1]
        out["diagnosis_candidate"] = easiest["candidate"]
        for m in easiest["failing_models"]:
            pool = results[m][easiest["candidate"]]["pool"]
            share = pool["format_failure_share"]
            out["diagnosis"].append({
                "model": m, "signal_rate": pool[signal_key],
                "correct_signal_rate": pool.get(STRICT_SIGNAL),
                "format_failure_share": share,
                "failure_mode": "format" if share >= FORMAT_SHARE_THRESHOLD else "search",
            })
    return out
