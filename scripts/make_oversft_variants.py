"""Produce the positive-control checkpoints whose amenability ordering is known a priori.

The plasticity literature establishes that RL amenability degrades monotonically with
SFT overtraining as entropy is depleted. That known ordering is what Gate A tests the
probe against, independently of any correlation study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from amenability.registry.loader import load_registry

OVERSFT_LEVELS: tuple[int, ...] = (1, 4)
BASE_SFT_STEPS = 1500


def variant_key(base_key: str, level: int) -> str:
    return f"{base_key}-oversft{level}x"


def variant_plan(base_keys: list[str], levels: tuple[int, ...] = OVERSFT_LEVELS) -> list[dict]:
    registry = load_registry()
    plan: list[dict] = []
    for base_key in base_keys:
        spec = registry[base_key]
        if spec.role != "control_base":
            raise ValueError(f"{base_key} is not marked control_base in the registry")
        all_levels = (0,) + tuple(levels)
        n = len(all_levels)
        for i, level in enumerate(all_levels):
            plan.append(
                {
                    "base_key": base_key,
                    "level": level,
                    "variant_key": base_key if level == 0 else variant_key(base_key, level),
                    "max_steps": level * BASE_SFT_STEPS,
                    "known_order": n - 1 - i,
                }
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", default=["qwen2.5-0.5b", "llama-3.2-1b"])
    parser.add_argument("--out", type=Path, default=Path("results/variants"))
    args = parser.parse_args()

    for entry in variant_plan(args.bases):
        if entry["level"] == 0:
            continue
        print(f"would train {entry['variant_key']} for {entry['max_steps']} SFT steps")


if __name__ == "__main__":
    main()
