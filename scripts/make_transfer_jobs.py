"""Write the transfer batch's job manifest: one line per (model, suite, run seed).

Arm-major order so the SLURM array works through one arm at a time and a partial
batch is still an analysable arm rather than a random third of every arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from amenability.registry.loader import load_registry

ARMS: list[tuple[str, int]] = [("countdown", 0), ("graphpath", 0), ("countdown", 1)]
PILOT_SUITES = ("countdown", "graphpath")


def transfer_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]:
    return [(m, suite, seed) for suite, seed in ARMS for m in model_keys]


def pilot_jobs(model_keys: list[str]) -> list[tuple[str, str, int]]:
    return [(m, suite, 0) for suite in PILOT_SUITES for m in model_keys]


def write_jobs(jobs: list[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{m} {s} {seed}\n" for m, s, seed in jobs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="pre-eval calibration jobs instead of the batch")
    parser.add_argument("--only-model", nargs="+", default=None, help="restrict to these registry keys")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    keys = sorted(load_registry())
    if args.only_model:
        unknown = sorted(set(args.only_model) - set(keys))
        if unknown:
            parser.error(f"unknown model keys: {unknown}; valid: {keys}")
        keys = [k for k in keys if k in set(args.only_model)]
    jobs = pilot_jobs(keys) if args.pilot else transfer_jobs(keys)
    out = args.out or Path("results/transfer") / ("jobs_pilot.txt" if args.pilot else "jobs.txt")
    write_jobs(jobs, out)
    print(f"wrote {len(jobs)} jobs to {out}")


if __name__ == "__main__":
    main()
