"""The probe prompts use the TinyZero base-model completion scaffold (transfer-rulings T19).

A base model given a bare instruction treats it as the start of a web document and
never emits an answer tag. The scaffold frames a User/Assistant exchange and ends
the prompt inside an opened <think> block, so the model's continuation is the
reasoning, then the answer. These tests pin the scaffold and prove that the format
example inside the prompt can never be scored as the model's answer: every consumer
(pass@k, GRPO reward, rejection-sampled SFT) verifies the completion alone.
"""
import pytest

from amenability.suites.countdown import (
    generate_countdown, solve_countdown, verify_countdown,
)
from amenability.suites.graphpath import _labels, extract_path, generate_graphpath, verify_graphpath
from amenability.training.grpo import build_reward_fn
from amenability.training.sft import build_rejection_dataset

FRAMING = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer.\nUser: "
)
OPENER = "\nAssistant: Let me solve this step by step.\n<think>\n"


def _countdown_items():
    return generate_countdown(n=30, seed=0)


def _graph_items():
    return generate_graphpath(n=30, seed=0)


def _numbers_target(item):
    numbers_s, target_s = item.answer.split("|")
    return [int(x) for x in numbers_s.split(",")], int(target_s)


def _bfs_path(item):
    edges_s, source, target = item.answer.split("|")
    adj = {}
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


@pytest.mark.parametrize("items", [_countdown_items, _graph_items], ids=["countdown", "graphpath"])
def test_prompt_uses_the_conversation_scaffold_and_ends_inside_think(items):
    for it in items():
        assert it.prompt.startswith(FRAMING)
        assert it.prompt.endswith(OPENER)
        assert "<think> </think>" in it.prompt
        assert "<answer>" in it.prompt and "</answer>" in it.prompt


def test_countdown_prompt_names_numbers_target_and_exactly_once():
    for it in _countdown_items():
        numbers, target = _numbers_target(it)
        assert f"Using the numbers {', '.join(map(str, numbers))}," in it.prompt
        assert f"create an equation that equals {target}." in it.prompt
        assert "each number must be used exactly once" in it.prompt
        assert "<answer> (1 + 2) / 3 </answer>" in it.prompt


def test_countdown_prompt_example_never_verifies_as_an_answer():
    # The example evaluates to 1, and every target is at least 10. Feeding the whole
    # prompt to the verifier, as if a consumer had scored prompt + completion, must
    # still score 0 for every item.
    for it in generate_countdown(n=300, seed=0):
        assert verify_countdown(it, it.prompt) is False


def test_countdown_well_formed_completion_verifies_for_real_items():
    for it in _countdown_items():
        numbers, target = _numbers_target(it)
        expr = solve_countdown(numbers, target)
        completion = (
            f"I need {target} from {numbers}. Trying combinations.\n</think>\n"
            f"<answer> {expr} </answer>"
        )
        assert verify_countdown(it, completion) is True


def test_countdown_completion_without_answer_scores_zero_even_though_prompt_has_one():
    it = _countdown_items()[0]
    assert verify_countdown(it, "Let me think about forum rules.\n</think>") is False


def test_countdown_tolerates_a_trailing_equals_target_only():
    it = _countdown_items()[0]
    numbers, target = _numbers_target(it)
    expr = solve_countdown(numbers, target)
    assert verify_countdown(it, f"<answer>{expr} = {target}</answer>") is True
    assert verify_countdown(it, f"<answer> {expr}={target} </answer>") is True
    assert verify_countdown(it, f"<answer>{expr} = {target + 1}</answer>") is False
    assert verify_countdown(it, f"<answer>{expr} = {target} = {target}</answer>") is False
    assert verify_countdown(it, f"<answer>{target} = {expr}</answer>") is False
    # The equality does not rescue a wrong expression.
    wrong = "+".join(map(str, numbers))
    if sum(numbers) != target:
        assert verify_countdown(it, f"<answer>{wrong} = {target}</answer>") is False


def test_graph_prompt_example_uses_nodes_no_item_can_contain():
    all_labels = set(_labels(12))
    it = _graph_items()[0]
    example = extract_path(it.prompt)
    assert example is not None and len(example) >= 2
    assert not set(example) & all_labels


def test_graph_prompt_example_never_verifies_as_an_answer():
    for it in generate_graphpath(n=300, seed=0):
        assert verify_graphpath(it, it.prompt) is False


def test_graph_well_formed_completion_verifies_for_real_items():
    for it in _graph_items():
        path = _bfs_path(it)
        completion = (
            "Start at the source and follow edges.\n</think>\n"
            "<answer> " + " -> ".join(path) + " </answer>"
        )
        assert verify_graphpath(it, completion) is True


@pytest.mark.parametrize(
    "items,verify",
    [(_countdown_items, verify_countdown), (_graph_items, verify_graphpath)],
    ids=["countdown", "graphpath"],
)
def test_grpo_reward_scores_the_completion_not_the_prompt_example(items, verify):
    its = items()
    fn = build_reward_fn(its, verify)
    prompts = [it.prompt for it in its[:4]]
    # A completion that never answers must earn 0, even though every prompt
    # contains a well-formed example answer.
    rewards = fn(completions=["I am not sure.\n</think>"] * 4, prompts=prompts)
    assert rewards == [0.0] * 4


def test_grpo_reward_scores_a_correct_completion_on_real_prompts():
    its = _countdown_items()[:3]
    fn = build_reward_fn(its, verify_countdown)
    comps = []
    for it in its:
        numbers, target = _numbers_target(it)
        comps.append(f"reasoning\n</think>\n<answer>{solve_countdown(numbers, target)}</answer>")
    assert fn(completions=comps, prompts=[it.prompt for it in its]) == [1.0] * 3


def test_sft_records_pair_the_scaffold_prompt_with_the_text_that_follows_it():
    it = _countdown_items()[0]
    numbers, target = _numbers_target(it)
    completion = f"Try it.\n</think>\n<answer>{solve_countdown(numbers, target)}</answer>"
    recs = build_rejection_dataset([it], {it.task_id: [completion]}, verify_countdown)
    assert recs == [{"prompt": it.prompt, "completion": completion}]
    assert recs[0]["prompt"].endswith("<think>\n")
    assert not recs[0]["completion"].startswith("<think>")


def test_scaffold_keeps_prompts_unique_for_reward_attribution():
    for its in (generate_countdown(n=300, seed=0), generate_graphpath(n=300, seed=0)):
        assert len({it.prompt for it in its}) == len(its)


def test_graph_generator_refuses_buckets_that_could_contain_the_example_nodes():
    generate_graphpath(n=3, seed=0, buckets=(23,))  # labels A..W, still disjoint
    with pytest.raises(ValueError, match="example"):
        generate_graphpath(n=3, seed=0, buckets=(24,))
