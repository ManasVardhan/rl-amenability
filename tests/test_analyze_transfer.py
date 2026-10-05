"""Synthetic telemetry with controlled conversion rates, so the expected rho is known."""
import json
from pathlib import Path

import pytest
from amenability.eval.passk import PassKResult
from amenability.probe.run import ProbeConfig, meta_path, preeval_path, telemetry_path
from amenability.probe.telemetry import ProbeTelemetry, StepRecord
from scripts.analyze_transfer import (
    DECISIONS, MIN_COMPLETE, Undefined, arm_features, arm_scores, calibration_table, decide,
    load_arm, main, render_leaderboard, render_report, transfer_stats,
)

MODELS = [f"m{i}" for i in range(10)]


def _telemetry(model: str, conv: float, breadth: float = 0.4, zero_adv: float = 0.1,
               kl: float = 0.01) -> ProbeTelemetry:
    """conversion_rate == conv exactly: pre pass@1 0.1, pass@32 0.1 + breadth, post pass@1 = 0.1 + conv * breadth."""
    steps = [
        StepRecord(step=i, mean_reward=0.1 + 0.01 * i, group_reward_std=0.2,
                   zero_advantage_frac=zero_adv, policy_entropy=2.0, kl=kl, grad_norm=1.0)
        for i in range(1, 6)
    ]
    pre = PassKResult(ks={1: 0.1, 8: 0.2, 32: 0.1 + breadth, 64: 0.1 + breadth}, n_samples=64,
                      n_items=3, per_item_correct={"a": 1, "b": 0, "c": 0})
    post = PassKResult(ks={1: 0.1 + conv * breadth, 8: 0.2, 32: 0.1 + breadth, 64: 0.1 + breadth},
                       n_samples=64, n_items=3, per_item_correct={"a": 1, "b": 0, "c": 0})
    return ProbeTelemetry(model_key=model, algorithm="grpo", steps=steps, pre=pre, post=post)


def _meta(model: str, suite: str, seed: int, **override) -> dict:
    cfg = ProbeConfig()
    meta = {"probe_key": f"{model}-{suite}-grpo-s{seed}", "model_key": model, "suite_key": suite,
            "item_seed": 0, "run_seed": seed, "learning_rate": cfg.learning_rate,
            "probe_steps": cfg.probe_steps, "n_step_records": cfg.probe_steps,
            "train_reward": "shaped"}
    meta.update(override)
    return meta


def _write(work_dir: Path, model: str, suite: str, seed: int, conv: float, breadth: float = 0.4,
           meta: dict | None = None, **telemetry_kw):
    """Telemetry plus a protocol-valid meta file, as run_probe writes them (ruling P10)."""
    p = telemetry_path(work_dir, model, suite, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_telemetry(model, conv, breadth, **telemetry_kw).to_json())
    mp = meta_path(work_dir, model, suite, seed)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta if meta is not None else _meta(model, suite, seed)))


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
    assert stats["n_retest"] == 10
    report = render_report(stats, [])
    assert "(n=9 per fit, 10 fits)" in report
    assert ("static comparator (pre pass@32 across suites; descriptive, not in the decision; "
            "includes breadth-excluded models)") in report


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
    flat = Undefined(reason="constant input", n=10)
    assert decide(good, good, 7, 10) == "INCONCLUSIVE_INCOMPLETE"
    assert decide(good, good, 10, 7) == "INCONCLUSIVE_INCOMPLETE"      # ruling P7: retest floor
    assert decide(flat, flat, 7, 10) == "INCONCLUSIVE_INCOMPLETE"      # completeness before degeneracy
    assert decide(flat, good, 10, 10) == "INCONCLUSIVE_DEGENERATE"     # ruling P9
    assert decide(good, flat, 10, 10) == "INCONCLUSIVE_DEGENERATE"
    assert decide(good, low, 10, 10) == "INCONCLUSIVE_RELIABILITY"
    assert decide(good, None, 10, 10) == "INCONCLUSIVE_RELIABILITY"
    assert decide(good, good, 10, 10) == "INVARIANT"
    assert decide(mid, good, 10, 10) == "PARTIAL"
    assert decide(low, good, 10, 10) == "DOMAIN_SPECIFIC"
    assert decide(C(rho=0.75, ci_low=0, ci_high=1, p_value=0.2, n=10), good, 10, 10) == "PARTIAL"
    assert MIN_COMPLETE == 8 and len(DECISIONS) == 6 and "INCONCLUSIVE_DEGENERATE" in DECISIONS


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
        {"decision": "INCONCLUSIVE_INCOMPLETE", "n_complete": 0, "n_retest": 0, "n_models": 2, "transfer": None, "retest": None,
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


# At n=3 many bootstrap resamples repeat one model, so the resampled vectors are
# constant by construction; that warning is expected here and says nothing.
@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_missing_retest_replicates_are_inconclusive_and_named(tmp_path):
    """Ruling P7: the retest pair has the same completeness floor as the transfer pair."""
    _populate(tmp_path, graph_order=MODELS, retest_order=MODELS[:3])
    stats = transfer_stats(tmp_path, MODELS, n_boot=200)
    assert stats["n_complete"] == 10 and stats["n_retest"] == 3
    assert stats["decision"] == "INCONCLUSIVE_INCOMPLETE"
    assert set(stats["arms"]["countdown_s1"]["failures"]) == set(MODELS[3:])
    report = render_report(stats, [])
    for m in MODELS[3:]:
        assert f"FAILED {m}" in report


def _spread_features(work_dir: Path, models: list[str], suite: str, seed: int, rng_seed: int):
    """Features that vary on every axis, so the roster a z-score is taken over changes the ranks."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    for m in models:
        conv = 5.0 if m == "m5" else float(rng.uniform(0.0, 1.0))   # m5 is an outlier that shifts the mean
        _write(work_dir, m, suite, seed, conv=conv, zero_adv=float(rng.uniform(0.0, 0.9)),
               kl=float(rng.uniform(0.001, 0.2)))


def _copy_arms(src: Path, dst: Path, models: list[str]):
    """The same files for `models`, and nothing else: the model removed from every arm."""
    for suite, seed in (("countdown", 0), ("countdown", 1), ("graphpath", 0)):
        for m in models:
            for fn in (telemetry_path, meta_path):
                if fn(src, m, suite, seed).exists():
                    out = fn(dst, m, suite, seed)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(fn(src, m, suite, seed).read_text())


def test_pairwise_statistics_renormalise_over_the_common_set(tmp_path):
    """Ruling P8: a model missing from one arm must not shift the other arm's normalisation."""
    partial, trimmed = tmp_path / "partial", tmp_path / "trimmed"
    rest = [m for m in MODELS if m != "m5"]
    _spread_features(partial, MODELS, "countdown", 0, 1)
    _spread_features(partial, MODELS, "countdown", 1, 2)
    _spread_features(partial, rest, "graphpath", 0, 3)
    _copy_arms(partial, trimmed, rest)
    a = transfer_stats(partial, MODELS, n_boot=200)
    b = transfer_stats(trimmed, MODELS, n_boot=200)
    assert a["n_complete"] == b["n_complete"] == 9
    assert a["transfer"] == b["transfer"]
    assert a["per_feature"] == b["per_feature"]
    # the per-arm leaderboard scores stay over each arm's own loaded set
    assert "m5" in a["arms"]["countdown_s0"]["scores"]
    assert a["arms"]["countdown_s0"]["scores"] != b["arms"]["countdown_s0"]["scores"]


def test_retest_renormalises_over_the_common_set(tmp_path):
    partial, trimmed = tmp_path / "partial", tmp_path / "trimmed"
    rest = [m for m in MODELS if m != "m5"]
    _spread_features(partial, MODELS, "countdown", 0, 1)
    _spread_features(partial, rest, "countdown", 1, 2)
    _spread_features(partial, MODELS, "graphpath", 0, 3)
    _copy_arms(partial, trimmed, rest)
    a = transfer_stats(partial, MODELS, n_boot=200)
    b = transfer_stats(trimmed, MODELS, n_boot=200)
    assert a["n_retest"] == b["n_retest"] == 9
    assert a["retest"] == b["retest"]


def test_leaderboard_labels_normalisation_and_shows_breadth(tmp_path):
    _populate(tmp_path, graph_order=MODELS)
    lb = render_leaderboard(transfer_stats(tmp_path, MODELS, n_boot=100))
    assert ("scores normalised within each arm over that arm's scored models; "
            "the statistics re-normalise over each pair's common models") in lb
    header = next(line for line in lb.splitlines() if line.startswith("| model |"))
    assert "breadth (C)" in header and "breadth (G)" in header
    row = next(line for line in lb.splitlines() if line.startswith("| m0 |"))
    assert "0.400" in row   # pre pass@32 0.5 minus pre pass@1 0.1


def test_load_arm_missing_meta_is_a_named_failure(tmp_path):
    _write(tmp_path, "m0", "countdown", 0, 0.5)
    meta_path(tmp_path, "m0", "countdown", 0).unlink()
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {} and failures["m0"] == "missing meta"


def test_load_arm_unreadable_meta_is_a_named_failure(tmp_path):
    _write(tmp_path, "m0", "countdown", 0, 0.5)
    meta_path(tmp_path, "m0", "countdown", 0).write_text("{nope")
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {} and failures["m0"].startswith("unreadable meta")


def test_load_arm_non_protocol_learning_rate_is_a_named_failure(tmp_path):
    _write(tmp_path, "m0", "countdown", 0, 0.5, meta=_meta("m0", "countdown", 0, learning_rate=3e-6))
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {}
    assert "learning_rate" in failures["m0"] and "3e-06" in failures["m0"] and "1e-06" in failures["m0"]


@pytest.mark.parametrize("value", [None, "binary"])
def test_load_arm_telemetry_not_trained_on_the_shaped_reward_is_a_named_failure(tmp_path, value):
    # Transfer-rulings T23: telemetry from before the shaped reward (no field) or
    # trained on the binary reward is not the protocol run.
    meta = _meta("m0", "countdown", 0, train_reward=value)
    if value is None:
        del meta["train_reward"]
    _write(tmp_path, "m0", "countdown", 0, 0.5, meta=meta)
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {} and "train_reward" in failures["m0"]


@pytest.mark.parametrize("field,value", [("probe_steps", 2), ("item_seed", 1)])
def test_load_arm_non_protocol_steps_or_item_seed_is_a_named_failure(tmp_path, field, value):
    _write(tmp_path, "m0", "countdown", 0, 0.5, meta=_meta("m0", "countdown", 0, **{field: value}))
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {} and field in failures["m0"]


def test_load_arm_short_run_is_a_named_failure(tmp_path):
    _write(tmp_path, "m0", "countdown", 0, 0.5, meta=_meta("m0", "countdown", 0, n_step_records=59))
    loaded, failures = load_arm(tmp_path, ["m0"], "countdown", 0)
    assert loaded == {}
    assert "n_step_records" in failures["m0"] and "59" in failures["m0"] and "60" in failures["m0"]


def _populate_flat(work_dir: Path, arms: list[tuple[str, int]]):
    """Every model identical on the named arms: the score vector is constant there."""
    for suite, seed in (("countdown", 0), ("graphpath", 0), ("countdown", 1)):
        for i, m in enumerate(MODELS):
            conv = 0.5 if (suite, seed) in arms else 0.1 + 0.08 * i
            _write(work_dir, m, suite, seed, conv=conv)


def test_constant_graph_scores_make_transfer_undefined_and_degenerate(tmp_path):
    _populate_flat(tmp_path, [("graphpath", 0)])
    stats = transfer_stats(tmp_path, MODELS, n_boot=100)
    assert stats["transfer"] == {"undefined": "constant input", "n": 10}
    assert stats["retest"]["rho"] == pytest.approx(1.0)
    assert stats["decision"] == "INCONCLUSIVE_DEGENERATE"
    assert stats["loo_transfer"] is None
    report = render_report(stats, [])
    assert "undefined (constant input), n=10" in report
    assert "rho=0.000" not in report


def test_constant_replicate_scores_make_retest_undefined_and_degenerate(tmp_path):
    _populate_flat(tmp_path, [("countdown", 1)])
    stats = transfer_stats(tmp_path, MODELS, n_boot=100)
    assert stats["retest"] == {"undefined": "constant input", "n": 10}
    assert stats["decision"] == "INCONCLUSIVE_DEGENERATE"


def test_constant_features_and_static_render_undefined_not_zero(tmp_path):
    _populate(tmp_path, graph_order=MODELS)   # retention, zero-advantage and pre pass@32 are all constant
    stats = transfer_stats(tmp_path, MODELS, n_boot=100)
    assert stats["per_feature"]["zero_advantage_rate"] == {"undefined": "constant input", "n": 10}
    assert stats["static_pass32"] == {"undefined": "constant input", "n": 10}
    report = render_report(stats, [])
    zline = next(line for line in report.splitlines() if "zero_advantage_rate" in line)
    assert "undefined (constant input), n=10" in zline
