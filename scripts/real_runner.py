from __future__ import annotations

from pathlib import Path

from amenability.eval.passk import evaluate_passk
from amenability.probe.telemetry import ProbeTelemetry
from amenability.registry.loader import load_registry
from amenability.suites.base import SuiteRegistry
from amenability.suites.countdown import generate_countdown, verify_countdown
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

    def probe(self, variant_key: str) -> ProbeTelemetry:
        model_path = self._resolve(variant_key)
        pre = evaluate_passk(
            model_path, self.probe_items, verify_countdown, BREADTH_KS,
            n_samples=64, temperature=1.0, seed=self.config.seed,
        )
        out_dir = self.work_dir / "probes" / variant_key
        records = run_grpo(
            GRPOSpec(
                model_path=model_path, model_key=variant_key, items=self.probe_items,
                verify_fn=verify_countdown, max_steps=self.config.probe_steps,
                num_generations=self.config.num_generations, learning_rate=1e-6,
                beta=0.04, temperature=1.0, seed=self.config.seed,
                output_dir=str(out_dir), save_steps=None,
            )
        )
        post = evaluate_passk(
            str(out_dir), self.probe_items, verify_countdown, BREADTH_KS,
            n_samples=64, temperature=1.0, seed=self.config.seed,
        )
        return ProbeTelemetry(
            model_key=variant_key, algorithm="grpo", steps=records, pre=pre, post=post
        )

    def full_run(self, variant_key: str) -> float:
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
        return float((post.ks[1] - pre.ks[1]) / breadth) if breadth > 1e-9 else 0.0
