# scripts/analyze_calibration.py
"""Tables and the pre-registered selection for the difficulty calibration
(transfer-rulings T22). CPU, seconds.

Reads results/calibration/<model>.json (from scripts/calibrate_difficulty.py)
and prints, per suite, model x candidate tables of the group-signal rate
(items with at least one correct of 8, over the pool), pass@1 and the failure
taxonomy, then applies the selection rule (amenability.eval.calibration
.select_candidate). Writes selection.json and report.md next to the inputs.

    uv run python -m scripts.analyze_calibration --dir results/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from amenability.eval.calibration import (
    FORMAT_SHARE_THRESHOLD, MAX_BEST_PASS1, MIN_GROUP_SIGNAL, PILOT_FLOORED, select_candidate,
)
from amenability.suites.difficulty import candidates_for, guesser_eligibility
from scripts.calibrate_difficulty import ROSTER

SUITES = ("countdown", "graphpath")
OUTPUT_NAMES = {"selection.json"}


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


def analyze(results: dict, suite_key: str, expected_models: tuple[str, ...] = ()) -> dict:
    """The selection for one suite. A model in expected_models with no result file
    at all counts as missing on every candidate, so a failed calibration job can
    never let a candidate pass by leaving its model out."""
    order = [(c.name, guesser_eligibility(c)["eligible"]) for c in candidates_for(suite_key)]
    sr = _suite_results(results, suite_key)
    for m in expected_models:
        sr.setdefault(m, {})
    return select_candidate(sr, order, floored=PILOT_FLOORED[suite_key])


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
        return lambda m, n: _fmt(pool(m, n)[key]) if pool(m, n) else "missing"

    def taxonomy(m, n):
        p = pool(m, n)
        if not p:
            return "missing"
        t = p["taxonomy"]
        return " / ".join(f"{100 * t[k]:.0f}" for k in
                          ("no_answer_tag", "unparseable_answer", "parseable_wrong", "correct"))

    lines = [f"### {suite_key}", ""]
    for title, cell in (
        (f"Group-signal rate (items with >= 1 correct of 8; need >= {MIN_GROUP_SIGNAL})",
         metric("group_signal_rate")),
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
                lines.append(f"- {d['model']}: {d['failure_mode']} (group signal "
                             f"{_fmt(d['group_signal_rate'])}, format failures "
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
    args = parser.parse_args(argv)
    results = load_results(args.dir)
    if not results:
        raise SystemExit(f"no calibration results in {args.dir}")
    selection = {s: analyze(results, s, tuple(args.models)) for s in SUITES}
    selection["batch_may_launch"] = all(selection[s]["launch"] for s in SUITES)
    report = "# Difficulty calibration (transfer-rulings T22)\n\n" + "\n".join(
        render_suite(results, s, selection[s]) for s in SUITES
    )
    report += ("\nBatch: may launch with the selected candidates.\n"
               if selection["batch_may_launch"]
               else "\nBatch: does NOT launch; see the suites above.\n")
    (args.dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True))
    (args.dir / "report.md").write_text(report)
    print(report)
    return selection


if __name__ == "__main__":
    main()
