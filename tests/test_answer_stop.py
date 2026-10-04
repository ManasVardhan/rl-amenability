"""Generation stops at the first </answer> on every path that samples probe-suite
completions (transfer-rulings T20), and only there: gsm8k has no answer tags, so it
gets no stop string."""
import json
from pathlib import Path

import pytest

from amenability.eval.passk import evaluate_passk, sampling_kwargs
from amenability.suites.base import ANSWER_STOP, TaskItem, truncate_at_stop
from amenability.suites.catalog import PROBE_SUITES
from amenability.suites.countdown import generate_countdown, solve_countdown, verify_countdown
from amenability.training.grpo import GRPOSpec, build_reward_fn, run_grpo
from amenability.training.sft import build_rejection_dataset


def _item():
    return TaskItem(task_id="t/0", suite="s", prompt="p", answer="1", difficulty=0)


def test_answer_stop_is_the_closing_tag():
    assert ANSWER_STOP == "</answer>"


def test_truncate_keeps_the_stop_string_and_drops_the_rest():
    assert truncate_at_stop("a<answer>x</answer>loop<answer>y</answer>", ANSWER_STOP) == "a<answer>x</answer>"
    assert truncate_at_stop("no tag here", ANSWER_STOP) == "no tag here"
    assert truncate_at_stop("a</answer>b", None) == "a</answer>b"


def test_every_probe_suite_stops_at_the_closing_answer_tag():
    for suite in PROBE_SUITES.values():
        assert suite.stop == ANSWER_STOP


def test_sampling_kwargs_keep_the_stop_string_in_the_output():
    # vLLM strips stop strings from the returned text by default, which would make
    # every <answer>(.*?)</answer> match fail and score the whole suite 0.
    kw = sampling_kwargs(n=4, temperature=1.0, seed=3, stop=ANSWER_STOP)
    assert kw["stop"] == ["</answer>"]
    assert kw["include_stop_str_in_output"] is True
    assert kw["max_tokens"] == 768 and kw["top_p"] == 0.95
    assert kw["n"] == 4 and kw["temperature"] == 1.0 and kw["seed"] == 3


def test_sampling_kwargs_without_a_stop_are_unchanged():
    kw = sampling_kwargs(n=4, temperature=0.8, seed=0, stop=None)
    assert kw == {"n": 4, "temperature": 0.8, "top_p": 0.95, "max_tokens": 768, "seed": 0}


def test_evaluate_passk_forwards_the_stop_and_truncates_completions():
    seen = {}

    def gen(model_path, prompts, n, temperature, seed, stop=None):
        seen["stop"] = stop
        return [["<answer>1</answer><answer>0</answer>"]]

    verified = []

    def verify(item, completion):
        verified.append(completion)
        return True

    evaluate_passk("m", [_item()], verify, (1,), n_samples=1, temperature=1.0, seed=0,
                   generate_fn=gen, stop=ANSWER_STOP)
    assert seen["stop"] == ANSWER_STOP
    assert verified == ["<answer>1</answer>"]


def test_evaluate_passk_defaults_to_no_stop():
    seen = {}

    def gen(model_path, prompts, n, temperature, seed, stop=None):
        seen["stop"] = stop
        return [["x</answer>y"]]

    verified = []
    evaluate_passk("m", [_item()], lambda it, c: verified.append(c) or True, (1,),
                   n_samples=1, temperature=1.0, seed=0, generate_fn=gen)
    assert seen["stop"] is None
    assert verified == ["x</answer>y"]


def test_grpo_reward_scores_only_up_to_the_first_stop():
    seen = []
    fn = build_reward_fn([_item()], lambda it, c: seen.append(c) or True, stop=ANSWER_STOP)
    fn(completions=["<answer>1</answer> loop <answer>2</answer>"], prompts=["p"])
    assert seen == ["<answer>1</answer>"]


def test_grpo_reward_without_a_stop_sees_the_whole_completion():
    seen = []
    fn = build_reward_fn([_item()], lambda it, c: seen.append(c) or True)
    fn(completions=["a</answer>b"], prompts=["p"])
    assert seen == ["a</answer>b"]


def test_run_grpo_records_the_stop_in_the_manifest(tmp_path):
    class T:
        model = None
        def train(self): pass
        def save_model(self, d): Path(d).mkdir(parents=True, exist_ok=True)

    spec = GRPOSpec(model_path="m", model_key="k", items=[_item()], verify_fn=lambda i, c: True,
                    max_steps=1, num_generations=2, learning_rate=1e-6, beta=0.0,
                    temperature=1.0, seed=0, output_dir=str(tmp_path), save_steps=None,
                    stop=ANSWER_STOP)
    run_grpo(spec, trainer_factory=lambda **kw: T())
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["completion_stop"] == "</answer>"
    assert manifest["generation_runs_to_cap"] is True


def test_sft_targets_are_cut_at_the_first_stop():
    it = generate_countdown(n=3, seed=0)[0]
    numbers_s, target_s = it.answer.split("|")
    expr = solve_countdown([int(x) for x in numbers_s.split(",")], int(target_s))
    head = f"t\n</think>\n<answer>{expr}</answer>"
    looped = head + "\n<think>\nagain</think>\n<answer>1 + 1</answer>"
    recs = build_rejection_dataset([it], {it.task_id: [looped]}, verify_countdown, stop=ANSWER_STOP)
    assert recs == [{"prompt": it.prompt, "completion": head}]


def test_sft_without_a_stop_keeps_the_completion():
    it = _item()
    recs = build_rejection_dataset([it], {"t/0": ["a</answer>b"]}, lambda i, c: True)
    assert recs[0]["completion"] == "a</answer>b"
