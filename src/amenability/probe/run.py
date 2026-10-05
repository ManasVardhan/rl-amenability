"""The probe: pre-eval, a short fixed-budget GRPO run, post-eval, telemetry to disk.

This is the ONE implementation of the probe. RealRunner (Stage 0) and the per-job
CLI (the transfer batch) both call it, so the protocol cannot drift between the
two. Everything that varies between jobs is an argument; everything that must not
vary lives in ProbeConfig's defaults.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from amenability.eval.passk import PassKResult, evaluate_passk, pass_at_k
from amenability.probe.telemetry import ProbeTelemetry
from amenability.suites.base import TaskItem
from amenability.suites.catalog import get_probe_suite
from amenability.training.grpo import GRPOSpec, run_grpo

# os.umask can only be read by setting it. Read it once at import, not per write,
# so no write opens a process-wide umask-0 window for other threads.
_UMASK = os.umask(0)
os.umask(_UMASK)


@dataclass(frozen=True)
class ProbeConfig:
    probe_steps: int = 60
    num_generations: int = 8
    n_probe_items: int = 300
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0
    n_samples: int = 64
    ks: tuple[int, ...] = (1, 8, 32, 64)


def probe_key(model_key: str, suite_key: str, run_seed: int) -> str:
    return f"{model_key}-{suite_key}-grpo-s{run_seed}"


def telemetry_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path:
    return work_dir / "telemetry" / f"{probe_key(model_key, suite_key, run_seed)}.json"


def meta_path(work_dir: Path, model_key: str, suite_key: str, run_seed: int) -> Path:
    return work_dir / "meta" / f"{probe_key(model_key, suite_key, run_seed)}.json"


def preeval_path(work_dir: Path, model_key: str, suite_key: str) -> Path:
    return work_dir / "preeval" / f"{model_key}-{suite_key}.json"


def passk_by_bucket(
    result: PassKResult, items: list[TaskItem], ks: tuple[int, ...]
) -> dict[int, dict[int, float]]:
    by_bucket: dict[int, list[int]] = {}
    for it in items:
        by_bucket.setdefault(it.difficulty, []).append(result.per_item_correct[it.task_id])
    return {
        bucket: {
            k: float(sum(pass_at_k(result.n_samples, c, k) for c in correct) / len(correct))
            for k in ks
        }
        for bucket, correct in sorted(by_bucket.items())
    }


def _passk_from_dict(d: dict) -> PassKResult:
    return PassKResult(
        ks={int(k): v for k, v in d["ks"].items()}, n_samples=d["n_samples"],
        n_items=d["n_items"], per_item_correct=d["per_item_correct"],
    )


def _write_json(path: Path, payload: dict) -> None:
    """Write to a temp file in the same directory, then rename over the target.

    A job killed mid-write (walltime, OOM, quota) must not leave a truncated file
    at the cache path, where a later run would treat it as a finished result.
    os.replace is atomic within one filesystem, hence the same directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        # mkstemp creates the file 0600 and os.replace keeps that mode, which
        # would make every result owner-only. Give it the mode a plain open()
        # would have had.
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _cuda():
    """torch.cuda when a GPU is present, else None; every memory field is then None."""
    import torch

    return torch.cuda if torch.cuda.is_available() else None


def _sleep_pool_bytes(manifest: Path) -> int | None:
    try:
        value = json.loads(manifest.read_text()).get("vllm_sleep_pool_bytes")
    except (OSError, ValueError, AttributeError):
        return None
    return int(value) if value is not None else None


def run_probe(
    *,
    model_path: str,
    model_key: str,
    suite_key: str,
    work_dir: Path,
    config: ProbeConfig = ProbeConfig(),
    item_seed: int = 0,
    run_seed: int = 0,
    force: bool = False,
    keep_checkpoint: bool = False,
    pre_only: bool = False,
    evaluate_fn=None,
    train_fn=None,
) -> ProbeTelemetry | PassKResult:
    evaluate = evaluate_fn or evaluate_passk
    train = train_fn or run_grpo
    suite = get_probe_suite(suite_key)
    # Items come from item_seed, NEVER from run_seed: a seed replicate must see
    # byte-identical prompts or it measures item variance, not run variance.
    items = suite.generate(config.n_probe_items, item_seed)

    def evaluate_at(path: str) -> PassKResult:
        return evaluate(
            path, items, suite.verify, config.ks,
            n_samples=config.n_samples, temperature=config.temperature, seed=run_seed,
            stop=suite.stop,
        )

    if pre_only:
        out = preeval_path(work_dir, model_key, suite_key)
        if out.exists() and not force:
            print(f"reusing cached pre-eval for {model_key}/{suite_key} from {out}")
            return _passk_from_dict(json.loads(out.read_text())["pre"])
        pre = evaluate_at(model_path)
        _write_json(out, {
            "model_key": model_key, "suite_key": suite_key, "item_seed": item_seed,
            "n_items": len(items), "pre": asdict(pre),
            "by_bucket": passk_by_bucket(pre, items, config.ks),
            # Which vLLM sampler produced the samples (transfer-rulings T17).
            "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        })
        return pre

    key = probe_key(model_key, suite_key, run_seed)
    t_path = telemetry_path(work_dir, model_key, suite_key, run_seed)
    if t_path.exists() and not force:
        print(f"reusing cached probe telemetry for {key} from {t_path}")
        return ProbeTelemetry.from_json(json.loads(t_path.read_text()))

    wall: dict[str, float] = {}
    t0 = time.monotonic()
    pre = evaluate_at(model_path)
    wall["pre"] = time.monotonic() - t0

    out_dir = work_dir / "probes" / key
    cuda = _cuda()
    # Reset the peak so peak_train_bytes measures training alone; the pre-eval
    # peak is kept so peak_memory_bytes stays the whole job's figure.
    pre_peak = None
    if cuda is not None:
        pre_peak = int(cuda.max_memory_allocated())
        cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    records = train(
        GRPOSpec(
            model_path=model_path, model_key=model_key, items=items,
            verify_fn=suite.verify, max_steps=config.probe_steps,
            num_generations=config.num_generations, learning_rate=config.learning_rate,
            beta=config.beta, temperature=config.temperature, seed=run_seed,
            output_dir=str(out_dir), save_steps=None, stop=suite.stop,
            # Shaped training reward (transfer-rulings T23); verify_fn stays the
            # binary correctness that pre/post pass@k and mean_correct use.
            train_reward_fn=suite.train_reward,
        )
    )
    wall["train"] = time.monotonic() - t0
    peak_train = None if cuda is None else int(cuda.max_memory_allocated())
    allocated_after_train = None if cuda is None else int(cuda.memory_allocated())
    # torch counts the colocated engine's sleep-mode pool as allocated even while
    # its memory is released, so peak_train over-reports by the pool's size
    # (transfer-rulings T22). run_grpo records the pool in its manifest.
    sleep_pool = _sleep_pool_bytes(out_dir / "run_manifest.json") if cuda is not None else None
    peak_train_excl_pool = (
        peak_train - sleep_pool if peak_train is not None and sleep_pool is not None else None
    )

    t0 = time.monotonic()
    post = evaluate_at(str(out_dir))
    wall["post"] = time.monotonic() - t0

    telemetry = ProbeTelemetry(
        model_key=model_key, algorithm="grpo", steps=records, pre=pre, post=post
    )
    # Meta first, telemetry last: the telemetry file is the cache marker, so it is
    # the commit. A job killed between the two writes leaves meta without
    # telemetry, and the rerun retrains; the reverse order would leave a cache
    # hit that never writes meta, a "missing meta" failure until --force.
    _write_json(meta_path(work_dir, model_key, suite_key, run_seed), {
        "probe_key": key, "model_key": model_key, "suite_key": suite_key,
        "item_seed": item_seed, "run_seed": run_seed,
        "learning_rate": config.learning_rate, "probe_steps": config.probe_steps,
        "n_step_records": len(records), "wall_seconds": wall,
        # The analysis accepts only telemetry trained on this reward (T23).
        "train_reward": "shaped",
        "peak_memory_bytes": None if cuda is None else max(pre_peak, int(cuda.max_memory_allocated())),
        "peak_train_bytes": peak_train,
        "allocated_after_train_bytes": allocated_after_train,
        "vllm_sleep_pool_bytes": sleep_pool,
        "peak_train_bytes_excl_vllm_pool": peak_train_excl_pool,
        # Which vLLM sampler produced the samples (transfer-rulings T17). Recorded
        # only; the analysis does not gate on it.
        "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    })
    # Round-trip through to_json so ProbeTelemetry stays the single definition of
    # the on-disk shape that from_json reads back.
    _write_json(t_path, json.loads(telemetry.to_json()))
    manifest = out_dir / "run_manifest.json"
    if manifest.exists():
        dest = work_dir / "manifests" / f"{key}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, dest)
    if not keep_checkpoint:
        # A probe checkpoint is ~3.5 GB and nothing downstream reads it; thirty of
        # them would exhaust a home quota. The manifest and telemetry survive.
        shutil.rmtree(out_dir, ignore_errors=True)
    return telemetry
