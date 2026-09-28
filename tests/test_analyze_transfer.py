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


def test_calibration_table_flags_breadth_saturated(tmp_path):
    """Transfer-rulings T4: pass@32 > 0.95 in every bucket is breadth-saturated, whatever pass@1 is."""
    def write_pre(model, suite, by_bucket):
        p = preeval_path(tmp_path, model, suite)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model_key": model, "suite_key": suite, "item_seed": 0, "n_items": 300,
                                 "pre": {"ks": {"1": 0.3, "32": 1.0}, "n_samples": 64, "n_items": 300, "per_item_correct": {}},
                                 "by_bucket": by_bucket}))
    write_pre("m0", "graphpath", {"8": {"1": 0.3, "32": 1.0}, "10": {"1": 0.3, "32": 1.0}, "12": {"1": 0.3, "32": 1.0}})
    write_pre("m0", "countdown", {"3": {"1": 0.3, "32": 1.0}, "4": {"1": 0.3, "32": 0.95}, "5": {"1": 0.3, "32": 1.0}})
    rows = {(r["model"], r["suite"]): r for r in calibration_table(tmp_path, ["m0", "m1"])}
    g = rows[("m0", "graphpath")]
    assert g["breadth_saturated"] is True and g["saturated"] is False and g["floored"] is False
    assert rows[("m0", "countdown")]["breadth_saturated"] is False   # 0.95 in one bucket is not > 0.95
    assert rows[("m1", "graphpath")]["breadth_saturated"] is None    # missing pre-eval: unknown, not False
    report = render_report(
        {"decision": "INCONCLUSIVE_INCOMPLETE", "n_complete": 0, "n_models": 2, "transfer": None, "retest": None,
         "static_pass32": None, "per_feature": {}, "loo_transfer": None, "arms": {}, "thresholds": {}},
        calibration_table(tmp_path, ["m0", "m1"]))
    header = next(line for line in report.splitlines() if line.startswith("| model | suite |"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols.index("breadth_saturated") == cols.index("saturated") + 1
    g_line = next(line for line in report.splitlines() if line.startswith("| m0 | graphpath |"))
    assert [c.strip() for c in g_line.strip("|").split("|")][cols.index("breadth_saturated")] == "True"


def test_main_writes_the_three_outputs(tmp_path, monkeypatch):
    _populate(tmp_path, graph_order=MODELS)
    import scripts.analyze_transfer as at
    monkeypatch.setattr(at, "load_registry", lambda: {m: None for m in MODELS})
    main(["--work-dir", str(tmp_path), "--n-boot", "200"])
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "report.json").read_text())["decision"] == "INVARIANT"
    lb = (tmp_path / "leaderboard_preview.md").read_text()
    assert "| model |" in lb and "m9" in lb
