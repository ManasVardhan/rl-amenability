"""Probe transfer analysis: does a model's probe score survive a change of probe suite?

Implements prereg/transfer.md. Every statistic prints its N; failed jobs and
no-breadth exclusions are named, never dropped silently; the transfer rho is never
shown without the test-retest rho beside it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from amenability.eval.stats import CorrelationResult, loo_spearman, spearman_with_ci
from amenability.probe.run import preeval_path, telemetry_path
from amenability.probe.telemetry import FeatureVector, ProbeTelemetry, extract_features
from amenability.registry.loader import load_registry
from amenability.scoring.score import amenability_score

ARMS: dict[str, tuple[str, int]] = {
    "countdown_s0": ("countdown", 0),
    "graphpath_s0": ("graphpath", 0),
    "countdown_s1": ("countdown", 1),
}
SCORED_FEATURES = ("conversion_rate", "retention_factor", "zero_advantage_rate")

INVARIANT_RHO = 0.7
PARTIAL_RHO = 0.4
P_THRESHOLD = 0.05
RETEST_MIN = 0.5
MIN_COMPLETE = 8
DECISIONS = ("INVARIANT", "PARTIAL", "DOMAIN_SPECIFIC", "INCONCLUSIVE_RELIABILITY", "INCONCLUSIVE_INCOMPLETE")

FLOOR_PASS32 = 0.02
SATURATION_PASS1 = 0.95
BREADTH_SATURATION_PASS32 = 0.95  # transfer-rulings T4


def load_arm(
    work_dir: Path, model_keys: list[str], suite_key: str, run_seed: int
) -> tuple[dict[str, ProbeTelemetry], dict[str, str]]:
    loaded: dict[str, ProbeTelemetry] = {}
    failures: dict[str, str] = {}
    for m in model_keys:
        p = telemetry_path(work_dir, m, suite_key, run_seed)
        if not p.exists():
            failures[m] = "missing"
            continue
        try:
            loaded[m] = ProbeTelemetry.from_json(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001 - a corrupt file is a named failure, not a crash
            failures[m] = f"unreadable: {type(e).__name__}: {e}"
    return loaded, failures


def arm_features(telemetries: dict[str, ProbeTelemetry]) -> tuple[dict[str, FeatureVector], dict[str, str]]:
    feats: dict[str, FeatureVector] = {}
    excluded: dict[str, str] = {}
    for m, t in telemetries.items():
        try:
            feats[m] = extract_features(t)
        except ValueError as e:
            excluded[m] = str(e)
    return feats, excluded


def arm_scores(features: dict[str, FeatureVector]) -> dict[str, float]:
    keys = sorted(features)
    if not keys:
        return {}
    return dict(zip(keys, amenability_score([features[k] for k in keys])))


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], list[float], list[float]]:
    keys = sorted(set(a) & set(b))
    return keys, [a[k] for k in keys], [b[k] for k in keys]


def correlate(a: dict[str, float], b: dict[str, float], n_boot: int) -> CorrelationResult | None:
    keys, x, y = paired(a, b)
    if len(keys) < 3:
        return None
    return spearman_with_ci(x, y, n_boot=n_boot)


def decide(transfer: CorrelationResult | None, retest: CorrelationResult | None, n_complete: int) -> str:
    if n_complete < MIN_COMPLETE or transfer is None:
        return "INCONCLUSIVE_INCOMPLETE"
    if retest is None or retest.rho < RETEST_MIN:
        return "INCONCLUSIVE_RELIABILITY"
    if transfer.rho >= INVARIANT_RHO and transfer.p_value < P_THRESHOLD:
        return "INVARIANT"
    if transfer.rho >= PARTIAL_RHO:
        return "PARTIAL"
    return "DOMAIN_SPECIFIC"


def _opt(r: CorrelationResult | None) -> dict | None:
    return None if r is None else asdict(r)


def transfer_stats(work_dir: Path, model_keys: list[str], n_boot: int = 10000) -> dict:
    arms: dict[str, dict] = {}
    scores: dict[str, dict[str, float]] = {}
    feats: dict[str, dict[str, FeatureVector]] = {}
    tels: dict[str, dict[str, ProbeTelemetry]] = {}
    for arm, (suite, seed) in ARMS.items():
        loaded, failures = load_arm(work_dir, model_keys, suite, seed)
        f, excluded = arm_features(loaded)
        s = arm_scores(f)
        tels[arm], feats[arm], scores[arm] = loaded, f, s
        arms[arm] = {
            "suite": suite, "run_seed": seed, "loaded": sorted(loaded),
            "failures": failures, "excluded": excluded, "scores": s,
            "features": {m: asdict(v) for m, v in f.items()},
        }

    cd0, gp0, cd1 = scores["countdown_s0"], scores["graphpath_s0"], scores["countdown_s1"]
    common, x, y = paired(cd0, gp0)
    transfer = correlate(cd0, gp0, n_boot)
    retest = correlate(cd0, cd1, n_boot)
    per_feature = {
        name: _opt(correlate(
            {m: getattr(v, name) for m, v in feats["countdown_s0"].items()},
            {m: getattr(v, name) for m, v in feats["graphpath_s0"].items()}, n_boot))
        for name in SCORED_FEATURES
    }
    static = _opt(correlate(
        {m: t.pre.ks[32] for m, t in tels["countdown_s0"].items()},
        {m: t.pre.ks[32] for m, t in tels["graphpath_s0"].items()}, n_boot))
    loo = loo_spearman(x, y) if len(common) >= 4 else None
    return {
        "n_models": len(model_keys), "arms": arms, "n_complete": len(common),
        "complete_models": common,
        "transfer": _opt(transfer), "retest": _opt(retest), "per_feature": per_feature,
        "static_pass32": static, "loo_transfer": loo,
        "decision": decide(transfer, retest, len(common)),
        "thresholds": {"invariant_rho": INVARIANT_RHO, "partial_rho": PARTIAL_RHO,
                       "p": P_THRESHOLD, "retest_min": RETEST_MIN, "min_complete": MIN_COMPLETE},
    }


def calibration_table(work_dir: Path, model_keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    suites = sorted({suite for suite, _ in ARMS.values()})
    for m in model_keys:
        for suite in suites:
            p = preeval_path(work_dir, m, suite)
            row = {"model": m, "suite": suite, "pass@1": None, "pass@32": None,
                   "by_bucket": None, "floored": None, "saturated": None,
                   "breadth_saturated": None}
            if p.exists():
                d = json.loads(p.read_text())
                by_bucket = {int(b): {int(k): v for k, v in ks.items()} for b, ks in d["by_bucket"].items()}
                row.update({
                    "pass@1": d["pre"]["ks"]["1"], "pass@32": d["pre"]["ks"]["32"],
                    "by_bucket": by_bucket,
                    "floored": all(ks[32] < FLOOR_PASS32 for ks in by_bucket.values()),
                    "saturated": all(ks[1] > SATURATION_PASS1 for ks in by_bucket.values()),
                    "breadth_saturated": all(ks[32] > BREADTH_SATURATION_PASS32 for ks in by_bucket.values()),
                })
            rows.append(row)
    return rows


def _fmt(r: dict | None) -> str:
    if r is None:
        return "n/a"
    return f"rho={r['rho']:.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] p={r['p_value']:.3f} n={r['n']}"


def render_report(stats: dict, calibration: list[dict]) -> str:
    lines = ["# Probe Transfer Report", "", f"**Decision: {stats['decision']}** "
             f"(n_complete={stats['n_complete']} of {stats['n_models']})", ""]
    lines += ["| statistic | value |", "|---|---|",
              f"| transfer rho (Countdown s0 vs Graph s0) | {_fmt(stats['transfer'])} |",
              f"| test-retest rho (Countdown s0 vs s1) | {_fmt(stats['retest'])} |",
              f"| static comparator (pre pass@32 across suites) | {_fmt(stats['static_pass32'])} |"]
    for name, r in stats["per_feature"].items():
        lines.append(f"| per-feature: {name} | {_fmt(r)} |")
    if stats["loo_transfer"]:
        lo, hi = min(stats["loo_transfer"]), max(stats["loo_transfer"])
        lines.append(f"| leave-one-out transfer rho range | {lo:.3f} to {hi:.3f} |")
    lines += ["", "## Arms", ""]
    for arm, a in stats["arms"].items():
        lines.append(f"### {arm} ({a['suite']}, seed {a['run_seed']}): {len(a['loaded'])} loaded")
        for m, why in sorted(a["failures"].items()):
            lines.append(f"- FAILED {m}: {why}")
        for m, why in sorted(a["excluded"].items()):
            lines.append(f"- EXCLUDED {m}: {why}")
        lines.append("")
    if calibration:
        lines += ["## Calibration (pre-probe pass@k by bucket)", "",
                  "| model | suite | pass@1 | pass@32 | floored | saturated | breadth_saturated | by bucket |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in calibration:
            if r["pass@1"] is None:
                lines.append(f"| {r['model']} | {r['suite']} | missing | | | | | |")
                continue
            buckets = "; ".join(f"{b}: {ks[1]:.2f}/{ks[32]:.2f}" for b, ks in sorted(r["by_bucket"].items()))
            lines.append(f"| {r['model']} | {r['suite']} | {r['pass@1']:.3f} | {r['pass@32']:.3f} | "
                         f"{r['floored']} | {r['saturated']} | {r['breadth_saturated']} | {buckets} |")
    lines += ["", "Thresholds: " + json.dumps(stats["thresholds"])]
    return "\n".join(lines) + "\n"


def render_leaderboard(stats: dict) -> str:
    cd0, gp0 = stats["arms"]["countdown_s0"], stats["arms"]["graphpath_s0"]
    models = sorted(set(cd0["scores"]) | set(gp0["scores"]),
                    key=lambda m: -(cd0["scores"].get(m, float("-inf"))))
    lines = ["# Leaderboard preview (probe-only, no ground truth, scores relative within arm)", "",
             "| model | score (Countdown) | score (Graph) | conversion (C) | conversion (G) | zero-adv (C) | zero-adv (G) |",
             "|---|---|---|---|---|---|---|"]
    for m in models:
        fc, fg = cd0["features"].get(m), gp0["features"].get(m)
        lines.append(f"| {m} | {_score_cell(cd0['scores'], m)} | {_score_cell(gp0['scores'], m)} "
                     f"| {_feature_cell(fc, 'conversion_rate')} | {_feature_cell(fg, 'conversion_rate')} "
                     f"| {_feature_cell(fc, 'zero_advantage_rate')} | {_feature_cell(fg, 'zero_advantage_rate')} |")
    return "\n".join(lines) + "\n"


def _score_cell(scores: dict[str, float], m: str) -> str:
    return f"{scores[m]:.3f}" if m in scores else "n/a"


def _feature_cell(features: dict | None, name: str) -> str:
    return "n/a" if features is None else f"{features[name]:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("results/transfer"))
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    model_keys = sorted(load_registry())
    stats = transfer_stats(args.work_dir, model_keys, n_boot=args.n_boot)
    calibration = calibration_table(args.work_dir, model_keys)
    (args.work_dir / "report.json").write_text(json.dumps({**stats, "calibration": calibration}, indent=2, sort_keys=True))
    (args.work_dir / "report.md").write_text(render_report(stats, calibration))
    (args.work_dir / "leaderboard_preview.md").write_text(render_leaderboard(stats))
    print(render_report(stats, calibration))


if __name__ == "__main__":
    main()
