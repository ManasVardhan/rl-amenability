# scripts/calibrate_difficulty.py
"""Difficulty calibration of the probe training pools (transfer-rulings T22). GPU.

For each model and each difficulty candidate (amenability.suites.difficulty),
generate the candidate's pool (100 items per bucket from item seed 0 by default,
which is exactly the 300-item pool a probe job trains on), sample 8 completions
per item with the GRPO rollout sampling (temperature 1.0, top_p 1.0, top_k 0,
768 tokens, stop at the first </answer> kept in the output), and write per
bucket and over the pool: pass@1, pass@8, the group-signal rate (items with at
least one correct of 8), the shaped group-signal rate (items whose 8 shaped
training rewards are not all equal, transfer-rulings T23), the failure taxonomy
and its per-item counts, plus example completions.

One vLLM engine per model, reused across candidates and shut down before the
next model loads. Output: <out-dir>/<model>.json, rewritten after each
candidate, so a job killed by the walltime resumes where it stopped. A cached
candidate is reused only if it was made with the same settings and has per-item
category counts (written since T23).

    uv run python -m scripts.calibrate_difficulty --model qwen2.5-0.5b
    uv run python -m scripts.calibrate_difficulty --model-index $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from amenability.eval.calibration import N_SAMPLES, rollout_sampling_kwargs, summarize_candidate
from amenability.probe.run import _write_json
from amenability.registry.loader import load_registry
from amenability.suites.base import ANSWER_STOP
from amenability.suites.difficulty import CANDIDATES, get_candidate, guesser_eligibility

DEFAULT_OUT_DIR = Path("results/calibration")
# The nine models of the calibration pilot (llama-3.2-1b is excluded pending its
# licence, issue #46). --model-index i picks ROSTER[i], for a SLURM array.
ROSTER = (
    "falcon3-1b-base", "gemma-3-1b-pt", "olmo-2-1b", "qwen2.5-0.5b", "qwen2.5-1.5b",
    "qwen2.5-math-1.5b", "qwen3-0.6b-base", "smollm2-1.7b", "stablelm-2-1.6b",
)


class VLLMEngine:
    """One vLLM engine for one model. Prompts go in as text, which vLLM tokenizes
    with the tokenizer's special tokens, as TRL's GRPO trainer does before handing
    token IDs to its colocated engine. max_model_len is the GRPO engine's, and the
    same context check refuses a pool whose longest prompt plus the cap would not
    fit."""

    def __init__(self, model_path: str, seed: int) -> None:
        from vllm import LLM

        from amenability.training.grpo import VLLM_MAX_MODEL_LENGTH

        self.llm = LLM(model=model_path, seed=seed, dtype="bfloat16",
                       gpu_memory_utilization=0.85, max_model_len=VLLM_MAX_MODEL_LENGTH)

    def generate(self, prompts: list[str], sampling: dict) -> list[list[str]]:
        from vllm import SamplingParams

        from amenability.training.grpo import check_fits_vllm_context

        tok = self.llm.get_tokenizer()
        check_fits_vllm_context(max(len(tok(p).input_ids) for p in prompts),
                                sampling["max_tokens"])
        outputs = self.llm.generate(prompts, SamplingParams(**sampling))
        return [[o.text for o in out.outputs] for out in outputs]

    def release(self) -> None:
        import gc

        from amenability.eval.passk import shutdown_vllm_engine

        try:
            shutdown_vllm_engine(self.llm)
        finally:
            self.llm = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _jsonable(params: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def calibrate_model(
    model_key: str,
    model_path: str,
    candidate_names: list[str],
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    make_engine=VLLMEngine,
    n_per_bucket: int = 100,
    item_seed: int = 0,
    sample_seed: int = 0,
    n_examples: int = 5,
    force: bool = False,
) -> Path:
    out = Path(out_dir) / f"{model_key}.json"
    sampling = rollout_sampling_kwargs(n=N_SAMPLES, seed=sample_seed, stop=ANSWER_STOP)
    settings = {"n_per_bucket": n_per_bucket, "item_seed": item_seed,
                "sample_seed": sample_seed, "sampling": sampling}
    data = {"model_key": model_key, "model_path": model_path, **settings,
            "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            "candidates": {}}
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if all(cached.get(k) == v for k, v in settings.items()):
            # A candidate written before transfer-rulings T23 has no per-item
            # category counts, so no shaped group signal: recompute it. Same
            # seeds, so its strict numbers reproduce.
            data["candidates"] = {
                k: v for k, v in cached.get("candidates", {}).items()
                if "per_item_categories" in v
            }
            stale = sorted(set(cached.get("candidates", {})) - set(data["candidates"]))
            if stale:
                print(f"{out}: {', '.join(stale)} predate per-item categories (T23); recomputing")
        else:
            print(f"{out} was made with other settings; recomputing every candidate")

    todo = [n for n in candidate_names if n not in data["candidates"]]
    for name in candidate_names:
        if name not in todo:
            print(f"{model_key}/{name}: cached in {out}")
    if not todo:
        return out

    engine = make_engine(model_path, sample_seed)
    try:
        for name in todo:
            cand = get_candidate(name)
            n_buckets = len(cand.params["buckets"])
            items = cand.generate(n_per_bucket * n_buckets, item_seed)
            t0 = time.monotonic()
            completions = engine.generate([it.prompt for it in items], sampling)
            wall = time.monotonic() - t0
            summary = summarize_candidate(cand.suite_key, items, completions,
                                          n_examples=n_examples, stop=ANSWER_STOP)
            guesser = guesser_eligibility(cand)
            data["candidates"][name] = {
                "suite_key": cand.suite_key, "params": _jsonable(cand.params),
                "description": cand.description,
                "guesser": {
                    "rates": None if guesser["rates"] is None
                    else {str(b): r for b, r in guesser["rates"].items()},
                    "mean": guesser["mean"],
                },
                "eligible": guesser["eligible"], "n_items": len(items),
                "generation_seconds": wall, **summary,
            }
            # Keep the requested order in the file, cached entries included.
            data["candidates"] = {
                k: data["candidates"][k] for k in
                [n for n in candidate_names if n in data["candidates"]]
                + [k for k in data["candidates"] if k not in candidate_names]
            }
            _write_json(out, data)
            pool = summary["pool"]
            print(f"{model_key}/{name}: group signal {pool['group_signal_rate']:.3f} "
                  f"pass@1 {pool['pass@1']:.3f} format failures "
                  f"{pool['format_failure_share']:.3f} ({wall:.0f} s)", flush=True)
    finally:
        engine.release()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", nargs="+", help="registry keys, run in order")
    who.add_argument("--model-index", type=int, help=f"index into ROSTER (0..{len(ROSTER) - 1})")
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES),
                        choices=list(CANDIDATES), metavar="NAME",
                        help="difficulty candidates (default: all, hardest first per suite)")
    parser.add_argument("--n-per-bucket", type=int, default=100)
    parser.add_argument("--item-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, make_engine=VLLMEngine) -> None:
    args = parse_args(argv)
    if args.model_index is not None:
        if not 0 <= args.model_index < len(ROSTER):
            print(f"--model-index must be in 0..{len(ROSTER) - 1}", file=sys.stderr)
            sys.exit(2)
        models = [ROSTER[args.model_index]]
    else:
        models = args.model
    registry = load_registry()
    unknown = [m for m in models if m not in registry]
    if unknown:
        print(f"unknown model(s) {unknown}; registry keys: {', '.join(sorted(registry))}",
              file=sys.stderr)
        sys.exit(2)
    for m in models:
        out = calibrate_model(
            m, registry[m].hf_id, args.candidates, out_dir=args.out_dir,
            make_engine=make_engine, n_per_bucket=args.n_per_bucket,
            item_seed=args.item_seed, sample_seed=args.sample_seed,
            n_examples=args.n_examples, force=args.force,
        )
        print(f"{m}: {out}")


if __name__ == "__main__":
    main()
