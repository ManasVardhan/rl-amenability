# scripts/run_probe.py
"""One probe job: a (model, suite, run seed) triple. Launcher-agnostic.

A SLURM array task, a Modal function or a shell loop each call this once per job.
Everything about the protocol is fixed in ProbeConfig; --steps and --lr exist only
for the learning-rate recalibration bound in the spec (section 8) and for GPU smoke
tests, and the batch never passes them. Because the telemetry cache key ignores
steps, learning rate and item seed, any of --steps, --lr or --item-seed away from
the protocol value is refused unless --work-dir points somewhere other than the
batch directory, so a pilot or smoke run can never poison the batch cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amenability.eval.passk import PassKResult
from amenability.probe.run import ProbeConfig, run_probe
from amenability.registry.loader import load_registry
from amenability.suites.catalog import PROBE_SUITES

DEFAULT_WORK_DIR = Path("results/transfer")


def resolve_model(model: str, registry: dict) -> tuple[str, str]:
    if model in registry:
        return model, registry[model].hf_id
    p = Path(model)
    if p.exists():
        return p.name, str(p)
    raise KeyError(model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registry key or local checkpoint path")
    parser.add_argument("--suite", required=True, choices=sorted(PROBE_SUITES))
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--item-seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--pre-only", action="store_true", help="pre-eval only (calibration pilot)")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    parser.add_argument("--steps", type=int, default=ProbeConfig.probe_steps)
    parser.add_argument("--lr", type=float, default=ProbeConfig.learning_rate)
    return parser.parse_args(argv)


def non_protocol_flags(args: argparse.Namespace) -> list[str]:
    flags = []
    if args.steps != ProbeConfig.probe_steps:
        flags.append(f"--steps {args.steps}")
    if args.lr != ProbeConfig.learning_rate:
        flags.append(f"--lr {args.lr:g}")
    if args.item_seed != 0:
        flags.append(f"--item-seed {args.item_seed}")
    return flags


def main(argv: list[str] | None = None, run_probe_fn=run_probe) -> None:
    args = parse_args(argv)
    flags = non_protocol_flags(args)
    if flags and args.work_dir.resolve() == DEFAULT_WORK_DIR.resolve():
        print(
            f"non-protocol flags ({', '.join(flags)}) would write into the batch cache "
            f"at {DEFAULT_WORK_DIR}, where the real protocol run would later reuse them.\n"
            f"pass a separate --work-dir, e.g. --work-dir {DEFAULT_WORK_DIR / 'lr_pilot'}",
            file=sys.stderr,
        )
        sys.exit(2)
    registry = load_registry()
    try:
        model_key, model_path = resolve_model(args.model, registry)
    except KeyError:
        print(
            f"unknown model {args.model!r}: not a registry key and not an existing path.\n"
            f"registry keys: {', '.join(sorted(registry))}",
            file=sys.stderr,
        )
        sys.exit(2)

    result = run_probe_fn(
        model_path=model_path, model_key=model_key, suite_key=args.suite,
        work_dir=args.work_dir,
        config=ProbeConfig(probe_steps=args.steps, learning_rate=args.lr),
        item_seed=args.item_seed, run_seed=args.run_seed, force=args.force,
        keep_checkpoint=args.keep_checkpoint, pre_only=args.pre_only,
    )
    if isinstance(result, PassKResult):
        print(f"{model_key}/{args.suite} pre: " + " ".join(f"pass@{k}={v:.3f}" for k, v in sorted(result.ks.items())))
    else:
        print(
            f"{model_key}/{args.suite}/s{args.run_seed}: "
            f"pre pass@1={result.pre.ks[1]:.3f} pass@32={result.pre.ks[32]:.3f} | "
            f"post pass@1={result.post.ks[1]:.3f} pass@32={result.post.ks[32]:.3f} | "
            f"{len(result.steps)} steps recorded"
        )


if __name__ == "__main__":
    main()
