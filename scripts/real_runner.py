from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from amenability.eval.passk import evaluate_passk
from amenability.probe.run import ProbeConfig, run_probe
from amenability.probe.telemetry import ProbeTelemetry
from amenability.registry.loader import load_registry
from amenability.suites.base import SuiteRegistry
from amenability.suites.countdown import generate_countdown
from amenability.suites.gsm8k import load_gsm8k, verify_gsm8k
from amenability.training.grpo import GRPOSpec, run_grpo

BREADTH_KS = (1, 8, 32, 64)


class RealRunner:
    def __init__(self, config, work_dir: Path = Path("results")) -> None:
        self.config = config
        self.work_dir = work_dir
        self.registry = load_registry()

        # Load-bearing: registering both suites into one SuiteRegistry is the
        # mechanism that prevents the probe from training on target data. If a
        # probe (countdown) task_id ever collided with a target (gsm8k) task_id,
        # SuiteRegistry.register would raise SuiteOverlapError here, before any
        # GPU time is spent, rather than silently letting the probe see target
        # data and invalidating the result.
        self.suites = SuiteRegistry()
        self.probe_items = generate_countdown(n=config.n_probe_items, seed=config.seed)
        self.target_items = load_gsm8k("train", limit=config.n_target_items, seed=config.seed)
        self.suites.register("probe_countdown", self.probe_items)
        self.suites.register("target_gsm8k", self.target_items)

        # self.target_eval is deliberately NOT registered with self.suites. It is
        # a separate held-out GSM8K split (test, not train) used only to measure
        # pre/post pass@k in full_run, never as training data, so it does not
        # participate in the probe/target overlap check above.
        self.target_eval = load_gsm8k("test", limit=300, seed=config.seed)

    def _resolve(self, variant_key: str) -> str:
        if variant_key in self.registry:
            return self.registry[variant_key].hf_id
        return str(self.work_dir / "variants" / variant_key)

    def probe(self, variant_key: str, force: bool = False) -> ProbeTelemetry:
        # Stage 0 is roughly 130 GPU-hours of sequential work in one process under
        # a 24-hour SLURM wall clock. run_probe persists each probe's telemetry and
        # reuses it on a later invocation, which is what stops a timeout destroying
        # every number measured before it. Stage 0 uses one seed for items and runs.
        return run_probe(
            model_path=self._resolve(variant_key), model_key=variant_key,
            suite_key="countdown", work_dir=self.work_dir,
            config=ProbeConfig(
                probe_steps=self.config.probe_steps,
                num_generations=self.config.num_generations,
                n_probe_items=self.config.n_probe_items,
            ),
            item_seed=self.config.seed, run_seed=self.config.seed, force=force,
        )

    def full_run(self, variant_key: str, force: bool = False) -> float:
        outcome_path = self.work_dir / "outcomes" / f"{variant_key}.json"
        if not force and outcome_path.exists():
            print(f"reusing cached full-run outcome for {variant_key} from {outcome_path}")
            return float(json.loads(outcome_path.read_text())["outcome"])

        model_path = self._resolve(variant_key)
        pre = evaluate_passk(
            model_path, self.target_eval, verify_gsm8k, (1, 32),
            n_samples=32, temperature=0.8, seed=self.config.seed,
        )
        out_dir = self.work_dir / "full" / variant_key
        run_grpo(
            GRPOSpec(
                model_path=model_path, model_key=variant_key, items=self.target_items,
                verify_fn=verify_gsm8k, max_steps=self.config.full_steps,
                num_generations=self.config.num_generations, learning_rate=1e-6,
                beta=0.04, temperature=1.0, seed=self.config.seed,
                output_dir=str(out_dir), save_steps=200,
            )
        )
        post = evaluate_passk(
            str(out_dir), self.target_eval, verify_gsm8k, (1, 32),
            n_samples=32, temperature=0.8, seed=self.config.seed,
        )
        breadth = pre.ks[32] - pre.ks[1]
        # Same reasoning as extract_features: with no measurable breadth the
        # conversion ratio is undefined, and returning 0.0 would look like a
        # genuine measurement of no conversion while biasing the result in this
        # project's favour. Over-SFT'd variants are where breadth may collapse.
        if breadth <= 1e-9:
            raise ValueError(
                f"no measurable breadth for {variant_key} on the target suite: "
                f"pre pass@1={pre.ks[1]!r}, pre pass@32={pre.ks[32]!r}, "
                f"breadth={breadth!r}; a conversion ratio is undefined without "
                "headroom and returning 0.0 would silently favour the hypothesis"
            )
        outcome = float((post.ks[1] - pre.ks[1]) / breadth)

        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        outcome_path.write_text(
            json.dumps(
                {
                    "variant_key": variant_key,
                    "outcome": outcome,
                    "outcome_secondary": float(post.ks[1] - pre.ks[1]),
                    "breadth": breadth,
                    "pre": asdict(pre),
                    "post": asdict(post),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return outcome
