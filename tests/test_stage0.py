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
                       kl=0.01, grad_norm=1.0, mean_correct=0.05 + 0.004 * a * i,
                       zero_correct_group_frac=0.5)
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


def test_run_refuses_when_control_bases_are_reduced_to_one_family(monkeypatch, frozen_root):
    # Ruling R32: the primary claim rests on BOTH control families. A single-family
    # run must not be silently treated as primary-claim evidence.
    import scripts.run_stage0 as run_stage0_module

    monkeypatch.setattr(run_stage0_module, "CONTROL_BASES", ["qwen2.5-0.5b"])
    with pytest.raises(ValueError, match="qwen2.5-0.5b") as excinfo:
        run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    assert "llama-3.2-1b" in str(excinfo.value)


def test_naive_baseline_fits_the_correctness_trace(frozen_root):
    # Transfer-rulings T23: Gate B's naive baseline reads mean_correct, not the
    # shaped training reward.
    from amenability.scoring.baselines import naive_extrapolation
    res = run_stage0(Stage0Config(), FakeRunner(), frozen_root)
    runner = FakeRunner()
    for v in res["variants"]:
        t = runner.probe(v["variant_key"])
        want = naive_extrapolation([s.step for s in t.steps],
                                   [s.mean_correct for s in t.steps], 600)
        assert v["naive"] == pytest.approx(want)
