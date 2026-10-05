"""Stage 0: build the positive control, probe it, and evaluate Gates A and B."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from amenability.probe.telemetry import extract_features
from amenability.registry.loader import load_registry
from amenability.scoring.naive_correct import baseline_naive_correct
from amenability.scoring.gates import evaluate_gate_a, evaluate_gate_b
from amenability.scoring.score import amenability_score
from prereg.freeze import verify_freeze
from scripts.make_oversft_variants import variant_plan

CONTROL_BASES = ["qwen2.5-0.5b", "llama-3.2-1b"]


@dataclass(frozen=True)
class Stage0Config:
    probe_steps: int = 60
    full_steps: int = 600
    num_generations: int = 8
    n_probe_items: int = 300
    n_target_items: int = 300
    seed: int = 0


def run_stage0(config: Stage0Config, runner, root: Path) -> dict:
    verify_freeze(root)

    # The primary claim rests on BOTH control families. A single-family run cannot
    # rule out the objection that entropy collapse is a Qwen-specific artifact, so
    # refuse to produce a Stage 0 report from a reduced control arm.
    expected_bases = {"qwen2.5-0.5b", "llama-3.2-1b"}
    if set(CONTROL_BASES) != expected_bases:
        raise ValueError(
            f"Stage 0 requires both pre-registered control bases {sorted(expected_bases)}, "
            f"got {sorted(CONTROL_BASES)}. A single-family run is not primary-claim evidence."
        )

    registry = load_registry()
    plan = variant_plan(CONTROL_BASES)

    telemetries = [runner.probe(entry["variant_key"]) for entry in plan]
    features = [extract_features(t) for t in telemetries]
    scores = amenability_score(features)
    outcomes = [runner.full_run(entry["variant_key"]) for entry in plan]
    # The probe's correctness trace, not its shaped training reward (T23).
    naive = baseline_naive_correct(telemetries, config.full_steps)

    variants = []
    for entry, score, outcome, nv, feat in zip(plan, scores, outcomes, naive, features):
        variants.append(
            {
                **entry,
                "family": registry[entry["base_key"]].family,
                "score": score,
                "outcome": outcome,
                "naive": nv,
                "features": feat,
            }
        )

    scores_by_family: dict[str, list[float]] = defaultdict(list)
    known_by_family: dict[str, list[int]] = defaultdict(list)
    for v in variants:
        scores_by_family[v["family"]].append(v["score"])
        known_by_family[v["family"]].append(v["known_order"])

    return {
        "config": config,
        "variants": variants,
        "gate_a": evaluate_gate_a(dict(scores_by_family), dict(known_by_family)),
        "gate_b": evaluate_gate_b(scores, naive, outcomes),
    }


def render_report(results: dict) -> str:
    lines = ["# Stage 0 Report", ""]
    for gate_key in ("gate_a", "gate_b"):
        g = results[gate_key]
        verdict = "PASS" if g.passed else "FAIL"
        lines.append(f"**Gate {g.name}: {verdict}.** {g.detail}")
    lines += ["", "| variant | family | level | known order | score | naive | outcome |",
              "|---|---|---|---|---|---|---|"]
    for v in results["variants"]:
        lines.append(
            f"| {v['variant_key']} | {v['family']} | {v['level']}x | {v['known_order']} | "
            f"{v['score']:.3f} | {v['naive']:.3f} | {v['outcome']:.3f} |"
        )
    lines += ["", "## Pre-registered decision", ""]
    a, b = results["gate_a"].passed, results["gate_b"].passed
    if a and b:
        lines.append("Both gates pass. Proceed to Stage 1.")
    elif a and not b:
        lines.append(
            "Gate A passes, Gate B fails. The probe works but is not cheaper than naive "
            "extrapolation. Pivot the paper toward failure-mode detection."
        )
    else:
        lines.append("Gate A fails. The probe design is wrong. Stop and redesign.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/stage0_report.md"))
    args = parser.parse_args()

    from scripts.real_runner import RealRunner  # created in Task 18

    root = Path(__file__).resolve().parents[1]
    config = Stage0Config()
    results = run_stage0(config, RealRunner(config), root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results))
    print(render_report(results))


if __name__ == "__main__":
    main()
