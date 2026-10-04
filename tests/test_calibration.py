"""Difficulty calibration (transfer-rulings T22): the failure taxonomy, the
per-candidate summary, the rollout sampling it uses and the selection rule."""
import pytest

from amenability.eval.calibration import (
    CATEGORIES, classify_completion, rollout_sampling_kwargs, select_candidate,
    summarize_candidate,
)
from amenability.suites.base import ANSWER_STOP, TaskItem
from amenability.suites.countdown import verify_countdown
from amenability.suites.graphpath import generate_graphpath, verify_graphpath


def _cd(numbers="3,5,2", target=13, difficulty=3, tid="probe/countdown/t"):
    return TaskItem(task_id=tid, suite="probe_countdown", prompt="p",
                    answer=f"{numbers}|{target}", difficulty=difficulty)


@pytest.mark.parametrize("completion,expected", [
    ("I think the answer is 13", "no_answer_tag"),
    ("<answer>3*5-2", "no_answer_tag"),                               # cut at the cap
    ("<answer>The equation 3 * 5 - 2 = 13</answer>", "unparseable_answer"),  # prose
    ("<answer></answer>", "unparseable_answer"),
    ("<answer>3 x 5 - 2</answer>", "unparseable_answer"),
    ("<answer>3 ** 2</answer>", "unparseable_answer"),                # outside the grammar
    ("<answer>3*5-2</answer>", "correct"),
    ("<answer> (3 * 5) - 2 = 13 </answer>", "correct"),
    ("<answer>3+5+2</answer>", "parseable_wrong"),                    # wrong value
    ("<answer>3*5</answer>", "parseable_wrong"),                      # wrong numbers
    ("<answer>3 + 5 = 8</answer>", "parseable_wrong"),                # equation, wrong target
    ("<answer>3 / (5 - 5) + 2</answer>", "parseable_wrong"),          # valid, divides by zero
    ("<answer>3*5-2</answer> then <answer>prose</answer>", "correct"),  # first block counts
    ("<answer>prose</answer> then <answer>3*5-2</answer>", "unparseable_answer"),
])
def test_countdown_taxonomy(completion, expected):
    assert classify_completion("countdown", _cd(), completion) == expected


def _graph_item_and_path():
    it = generate_graphpath(n=3, seed=3)[0]
    edges_s, source, target = it.answer.split("|")
    adj = {}
    for e in edges_s.split(","):
        a, b = e.split("-")
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    prev, frontier = {source: None}, [source]
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
    return it, path[::-1]


def test_graphpath_taxonomy():
    it, path = _graph_item_and_path()
    ans = lambda p: "<answer>" + " -> ".join(p) + "</answer>"
    assert classify_completion("graphpath", it, "no tag " + " -> ".join(path)) == "no_answer_tag"
    assert classify_completion("graphpath", it, ans(path)) == "correct"
    assert classify_completion("graphpath", it, "<answer>The path is A -> B</answer>") == "unparseable_answer"
    assert classify_completion("graphpath", it, "<answer>" + ", ".join(path) + "</answer>") == "unparseable_answer"
    assert classify_completion("graphpath", it, "<answer></answer>") == "unparseable_answer"
    assert classify_completion("graphpath", it, ans(path[1:])) == "parseable_wrong"   # wrong start
    assert classify_completion("graphpath", it, ans([path[0], path[-1]])) == "parseable_wrong"  # non-edge
    assert classify_completion("graphpath", it, ans(path[:1])) == "parseable_wrong"   # one node


def test_the_stop_is_applied_before_classifying():
    # Anything after the first </answer> is the base model's loop (T20).
    assert classify_completion("countdown", _cd(), "<answer>3*5-2</answer>junk") == "correct"


@pytest.mark.parametrize("suite_key,verify", [("countdown", verify_countdown),
                                              ("graphpath", verify_graphpath)])
def test_correct_iff_the_suite_verifier_accepts(suite_key, verify):
    it = _cd() if suite_key == "countdown" else _graph_item_and_path()[0]
    path = _graph_item_and_path()[1]
    pool = ["", "<answer>", "<answer>3*5-2</answer>", "<answer>13</answer>", "<answer>2-3*5</answer>",
            "<answer>" + "->".join(path) + "</answer>", "<answer>A->B->C</answer>",
            "<answer>5*3-2=13</answer>", "<answer>(3*5)-2 = 12</answer>"]
    for c in pool:
        assert (classify_completion(suite_key, it, c) == "correct") == verify(it, c), c


def test_summary_per_bucket_and_pool():
    items = [_cd(tid=f"a{i}", difficulty=2) for i in range(2)] + [_cd(tid="b0", difficulty=3)]
    good, wrong, prose, none = ("<answer>3*5-2</answer>", "<answer>3+5+2</answer>",
                                "<answer>it is 13</answer>", "thinking...")
    completions = [
        [good] * 2 + [wrong] * 6,          # 2 of 8 correct
        [none] * 4 + [prose] * 4,          # 0 of 8
        [good] * 8,                        # 8 of 8
    ]
    s = summarize_candidate("countdown", items, completions, n_examples=1)
    b2, b3, pool = s["by_bucket"]["2"], s["by_bucket"]["3"], s["pool"]
    assert b2["n_items"] == 2 and b2["n_samples"] == 16
    assert b2["pass@1"] == pytest.approx((2 / 8 + 0) / 2)
    assert b2["pass@8"] == pytest.approx(0.5)
    assert b2["group_signal_rate"] == pytest.approx(0.5)
    assert b2["taxonomy"] == pytest.approx({
        "no_answer_tag": 4 / 16, "unparseable_answer": 4 / 16,
        "parseable_wrong": 6 / 16, "correct": 2 / 16})
    assert b3["pass@1"] == 1.0 and b3["group_signal_rate"] == 1.0
    assert pool["n_items"] == 3
    assert pool["group_signal_rate"] == pytest.approx(2 / 3)
    assert pool["pass@1"] == pytest.approx((0.25 + 0 + 1) / 3)
    assert pool["format_failure_share"] == pytest.approx(8 / 24)
    assert s["per_item_correct"] == {"a0": 2, "a1": 0, "b0": 8}
    assert set(s["examples"]) == set(CATEGORIES)
    assert all(len(v) <= 1 for v in s["examples"].values())
    assert s["examples"]["unparseable_answer"][0]["completion"] == prose


def test_summary_refuses_a_wrong_number_of_samples():
    with pytest.raises(ValueError):
        summarize_candidate("countdown", [_cd()], [["<answer>1</answer>"] * 7])
    with pytest.raises(ValueError):
        summarize_candidate("countdown", [_cd(), _cd(tid="x")], [["a"] * 8])


def test_rollout_sampling_matches_the_grpo_rollout_config():
    """The calibration must sample the way GRPO rollouts do (T22), not the way
    pass@k does (top_p 0.95)."""
    from trl import GRPOConfig

    from amenability.probe.run import ProbeConfig
    from amenability.training.grpo import GRPOSpec, build_grpo_config_kwargs

    spec = GRPOSpec(model_path="m", model_key="k", items=[_cd()], verify_fn=verify_countdown,
                    max_steps=60, num_generations=8, learning_rate=1e-6, beta=0.04,
                    temperature=ProbeConfig.temperature, seed=0, output_dir="/tmp/x",
                    save_steps=None, stop=ANSWER_STOP)
    cfg = GRPOConfig(**build_grpo_config_kwargs(spec))
    kw = rollout_sampling_kwargs(n=8, seed=0, stop=ANSWER_STOP)
    assert kw["n"] == cfg.num_generations == 8
    assert kw["temperature"] == cfg.temperature
    assert kw["top_p"] == cfg.top_p
    assert kw["top_k"] == cfg.top_k
    assert kw["min_p"] == (0.0 if cfg.min_p is None else cfg.min_p)
    assert kw["repetition_penalty"] == cfg.repetition_penalty
    assert kw["max_tokens"] == cfg.max_completion_length == 768
    assert kw["stop"] == cfg.generation_kwargs["stop"] == ["</answer>"]
    assert kw["include_stop_str_in_output"] is True
    assert kw["seed"] == 0


# ---- the selection rule ----

def _summ(gsr, p1, fmt=0.0):
    return {"pool": {"group_signal_rate": gsr, "pass@1": p1, "format_failure_share": fmt}}


ORDER = [("x-current", True), ("x-ineligible", False), ("x-easy", True), ("x-easier", True)]


def test_selects_the_hardest_candidate_meeting_both_conditions():
    results = {
        "strong": {"x-current": _summ(0.9, 0.6), "x-ineligible": _summ(1, 0.7),
                   "x-easy": _summ(1, 0.75), "x-easier": _summ(1, 0.9)},
        "weak": {"x-current": _summ(0.05, 0.01), "x-ineligible": _summ(0.5, 0.2),
                 "x-easy": _summ(0.2, 0.05), "x-easier": _summ(0.4, 0.1)},
        "floored": {"x-current": _summ(0, 0), "x-ineligible": _summ(0, 0),
                    "x-easy": _summ(0.0, 0.0), "x-easier": _summ(0.01, 0.0)},
    }
    out = select_candidate(results, ORDER, floored={"floored"})
    assert out["selected"] == "x-easy"           # ineligible skipped despite passing
    assert out["launch"] is True
    rows = {r["candidate"]: r for r in out["candidates"]}
    assert rows["x-current"]["failing_models"] == ["weak"]
    assert rows["x-ineligible"]["eligible"] is False
    assert rows["x-easier"]["best_pass1"] == 0.9


def test_threshold_edges_are_inclusive():
    results = {"a": {"x-current": _summ(0.15, 0.80)}}
    assert select_candidate(results, [("x-current", True)], floored=set())["selected"] == "x-current"


def test_no_candidate_meets_signal_then_reports_format_or_search():
    results = {
        "fmt": {"x-current": _summ(0.0, 0.0, fmt=0.8), "x-easy": _summ(0.1, 0.02, fmt=0.6)},
        "srch": {"x-current": _summ(0.1, 0.05, fmt=0.1), "x-easy": _summ(0.12, 0.05, fmt=0.2)},
    }
    out = select_candidate(results, [("x-current", True), ("x-easy", True)], floored=set())
    assert out["selected"] is None and out["launch"] is False
    assert out["reason"] == "no candidate gives every non-floored model group signal >= 0.15"
    diag = {d["model"]: d["failure_mode"] for d in out["diagnosis"]}
    assert diag == {"fmt": "format", "srch": "search"}
    assert out["diagnosis_candidate"] == "x-easy"   # the easiest eligible candidate


def test_signal_met_but_no_headroom_does_not_launch():
    results = {"a": {"x-current": _summ(0.1, 0.5), "x-easy": _summ(0.5, 0.95)}}
    out = select_candidate(results, [("x-current", True), ("x-easy", True)], floored=set())
    assert out["selected"] is None and out["launch"] is False
    assert "pass@1" in out["reason"]


def test_a_model_missing_from_a_candidate_fails_it():
    results = {"a": {"x-current": _summ(0.9, 0.5)}, "b": {}}
    out = select_candidate(results, [("x-current", True)], floored=set())
    assert out["selected"] is None
    assert out["candidates"][0]["missing_models"] == ["b"]
