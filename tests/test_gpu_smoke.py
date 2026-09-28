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
    # EXACTLY two, not ">= 1". transformers' Trainer._finalize_training logs once
    # more after training ends; a third record here means that phantom log is
    # being recorded as a training step with mean_reward=0.0.
    assert len(records) == 2, f"expected one record per training step, got {len(records)}"
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
