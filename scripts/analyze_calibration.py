# scripts/analyze_calibration.py
"""Tables and the pre-registered selection for the difficulty calibration
(transfer-rulings T22, revised by T23). CPU, seconds.

Reads results/calibration/<model>.json (from scripts/calibrate_difficulty.py)
and prints, per suite, model x candidate tables of the gated signal rate,
the strict correct-signal rate, pass@1 and the failure taxonomy, then applies
the selection rule (amenability.eval.calibration.select_candidate).

--reward shaped (default, T23): condition (a) reads the shaped group-signal
rate (items whose 8 shaped training rewards, 1.0 / 0.1 / 0.0, are not all
equal); the strict correct-signal rate (items with >= 1 correct of 8) is
reported, not gated. Needs calibration files with per-item category counts
(written since T23); a model whose file predates them is named as needing a
calibration rerun and blocks every candidate. Writes selection.json, report.md.

--reward strict (T22, superseded): (a) reads the strict rate. Writes
selection_t22.json and report_t22.md.

    uv run python -m scripts.analyze_calibration --dir results/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from amenability.eval.calibration import (
    FORMAT_SHARE_THRESHOLD, MAX_BEST_PASS1, MIN_GROUP_SIGNAL, PILOT_FLOORED, SHAPED_SIGNAL,
    STRICT_SIGNAL, select_candidate,
)
from amenability.suites.difficulty import candidates_for, guesser_eligibility
from scripts.calibrate_difficulty import ROSTER

SUITES = ("countdown", "graphpath")
REWARDS = {
    # reward mode -> (signal key gated by (a), output file names, ruling)
    "shaped": (SHAPED_SIGNAL, ("selection.json", "report.md"), "T23"),
    "strict": (STRICT_SIGNAL, ("selection_t22.json", "report_t22.md"), "T22"),
}
OUTPUT_NAMES = {names[0] for _, names, _ in REWARDS.values()}
RERUN_COMMAND = "sbatch --array=0-8 scripts/slurm/calibrate.sbatch"


def load_results(directory: Path) -> dict[str, dict[str, dict]]:
    """model -> candidate name -> candidate record (as written by the calibration)."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.name in OUTPUT_NAMES:
            continue
        data = json.loads(path.read_text())
        if "candidates" not in data:
            continue
        out[data.get("model_key", path.stem)] = data["candidates"]
    return out


def _suite_results(results: dict, suite_key: str) -> dict:
    return {
        m: {n: c for n, c in cands.items() if c.get("suite_key") == suite_key}
        for m, cands in results.items()
    }


def analyze(results: dict, suite_key: str, expected_models: tuple[str, ...] = (),
            reward: str = "shaped") -> dict:
    """The selection for one suite. A model in expected_models with no result file
    at all counts as missing on every candidate, so a failed calibration job can
    never let a candidate pass by leaving its model out. In shaped mode a model
    with any candidate record lacking the shaped signal (a pre-T23 file) is
    listed in needs_rerun."""
    signal_key = REWARDS[reward][0]
    order = [(c.name, guesser_eligibility(c)["eligible"]) for c in candidates_for(suite_key)]
    sr = _suite_results(results, suite_key)
    for m in expected_models:
        sr.setdefault(m, {})
    out = select_candidate(sr, order, floored=PILOT_FLOORED[suite_key], signal_key=signal_key)
    out["needs_rerun"] = sorted(
        m for m, cands in sr.items()
        if any(signal_key not in c.get("pool", {}) for c in cands.values())
    )
    blocking = sorted({m for r in out["candidates"] for m in r["missing_models"]}
                      & set(out["needs_rerun"]))
    if not out["selected"] and blocking:
        # The rule cannot be applied, which is not the same as no candidate passing.
        out["reason"] = (
            f"{', '.join(blocking)} have calibration results without per-item categories "
            "(written before T23), so the shaped signal is unknown; rerun the calibration"
        )
        out["remedies"] = []
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _table(title: str, models: list[str], labels: dict[str, str], names: list[str],
           header: list[str], cell) -> list[str]:
    lines = [f"#### {title}", "", "| model | " + " | ".join(header) + " |",
             "|---|" + "---|" * len(names)]
    for m in models:
        lines.append(f"| {labels[m]} | " + " | ".join(cell(m, n) for n in names) + " |")
    return lines + [""]


def render_suite(results: dict, suite_key: str, selection: dict) -> str:
    sr = {m: {} for m in selection["models"]}
    sr.update(_suite_results(results, suite_key))
    cands = candidates_for(suite_key)
    names = [c.name for c in cands]
    eligible = {r["candidate"]: r["eligible"] for r in selection["candidates"]}
    header = [n if eligible.get(n, True) else f"{n} (ineligible)" for n in names]
    floored = set(selection["floored_exempt"])
    models = sorted(sr)
    label = {m: f"{m} (floored, exempt)" if m in floored else m for m in models}

    def pool(m, n):
        return sr[m].get(n, {}).get("pool")

    def metric(key):
        return lambda m, n: (_fmt(pool(m, n)[key]) if pool(m, n) and key in pool(m, n)
                             else "missing")

    def taxonomy(m, n):
        p = pool(m, n)
        if not p:
            return "missing"
        t = p["taxonomy"]
        return " / ".join(f"{100 * t[k]:.0f}" for k in
                          ("no_answer_tag", "unparseable_answer", "parseable_wrong", "correct"))

    shaped = selection.get("signal_key") == SHAPED_SIGNAL
    lines = [f"### {suite_key}", ""]
    if selection.get("needs_rerun"):
        lines += [f"Results without per-item categories (written before T23), so no shaped "
                  f"signal; rerun the calibration (`{RERUN_COMMAND}`): "
                  + ", ".join(selection["needs_rerun"]), ""]
    signal_tables = (
        [(f"Shaped group-signal rate (items whose 8 shaped rewards 1.0 / 0.1 / 0.0 are "
          f"not all equal; need >= {MIN_GROUP_SIGNAL})", metric(SHAPED_SIGNAL)),
         ("Correct-signal rate (items with >= 1 correct of 8; strict, reported, not gated)",
          metric(STRICT_SIGNAL))]
        if shaped else
        [(f"Group-signal rate (items with >= 1 correct of 8; need >= {MIN_GROUP_SIGNAL})",
          metric(STRICT_SIGNAL))]
    )
    for title, cell in (
        *signal_tables,
        (f"pass@1 (best model needs <= {MAX_BEST_PASS1})", metric("pass@1")),
        ("Failure taxonomy, % of samples: no tag / unparseable / parseable wrong / correct",
         taxonomy),
    ):
        lines += _table(title, models, label, names, header, cell)

    lines += ["#### Selection", ""]
    for r in selection["candidates"]:
        notes = []
        if not r["eligible"]:
            notes.append("ineligible (guesser bound)")
        if r["failing_models"]:
            notes.append("signal < threshold: " + ", ".join(r["failing_models"]))
        if r["missing_models"]:
            notes.append("missing: " + ", ".join(r["missing_models"]))
        if not r["meets_headroom"]:
            notes.append(f"best pass@1 {_fmt(r['best_pass1'])} > {MAX_BEST_PASS1}")
        lines.append(f"- {r['candidate']}: " + ("; ".join(notes) if notes else "meets both conditions"))
    lines.append("")
    if selection["selected"]:
        lines.append(f"Selected: {selection['selected']}")
    else:
        lines.append(f"Selected: none. {selection['reason']}. The batch does NOT launch; "
                     "the owner decides.")
        if selection["diagnosis"]:
            lines.append("")
            lines.append(f"Failing models on {selection['diagnosis_candidate']} (easiest eligible), "
                         f"format if no tag + unparseable >= {FORMAT_SHARE_THRESHOLD:.0%} of samples:")
            for d in selection["diagnosis"]:
                lines.append(f"- {d['model']}: {d['failure_mode']} (gated signal "
                             f"{_fmt(d['signal_rate'])}, correct signal "
                             f"{_fmt(d['correct_signal_rate'])}, format failures "
                             f"{_fmt(d['format_failure_share'])})")
        if selection["remedies"]:
            lines.append("")
            lines.append("Candidate remedies (not implemented): " + "; ".join(selection["remedies"]) + ".")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("results/calibration"))
    parser.add_argument("--models", nargs="+", default=list(ROSTER),
                        help="models the rule requires (default: the nine-model roster)")
    parser.add_argument("--reward", choices=sorted(REWARDS), default="shaped",
                        help="shaped: the T23 rule (default); strict: the superseded T22 rule")
    args = parser.parse_args(argv)
    results = load_results(args.dir)
    if not results:
        raise SystemExit(f"no calibration results in {args.dir}")
    _, (sel_name, report_name), ruling = REWARDS[args.reward]
    selection = {s: analyze(results, s, tuple(args.models), args.reward) for s in SUITES}
    selection["reward"] = args.reward
    selection["batch_may_launch"] = all(selection[s]["launch"] for s in SUITES)
    report = (f"# Difficulty calibration (transfer-rulings {ruling}, {args.reward} "
              "training reward)\n\n") + "\n".join(
        render_suite(results, s, selection[s]) for s in SUITES
    )
    report += ("\nBatch: may launch with the selected candidates.\n"
               if selection["batch_may_launch"]
               else "\nBatch: does NOT launch; see the suites above.\n")
    (args.dir / sel_name).write_text(json.dumps(selection, indent=2, sort_keys=True))
    (args.dir / report_name).write_text(report)
    print(report)
    return selection


if __name__ == "__main__":
    main()
