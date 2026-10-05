from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from amenability.eval.passk import PASSK_MAX_TOKENS, shutdown_vllm_engine
from amenability.probe.callbacks import GroupedRewardRecorder, TelemetryCallback
from amenability.probe.entropy import EntropyProbe
from amenability.probe.telemetry import StepRecord
from amenability.suites.base import TaskItem, truncate_at_stop
from amenability.training.reward import CORRECT_SCORE, FORMAT_SCORE, NO_ANSWER_SCORE

ENTROPY_BATCH_SIZE = 16

# Spec section 6.2 asks for batches "sized by tokens, not sequences, for tokenizer
# comparability". TRL's GRPOConfig offers no token-based batching, and building it
# is out of scope here, so the batch shape is PINNED as sequences instead: one
# device batch holds exactly one prompt's group of num_generations completions,
# and PROMPTS_PER_STEP groups are accumulated per optimiser step. Leaving these
# unset means TRL's defaults apply and the effective protocol silently depends on
# the number of processes, so this is at least reproducible and machine-independent
# across models. The deviation from token-based batching is a known limitation
# recorded in the pre-registration, not a silent choice.
PROMPTS_PER_STEP = 4

# Rollout generation (transfer-rulings T21). TRL generates GRPO completions with a
# vLLM engine colocated in the training process on the job's one GPU, so the
# suite's stop string ends generation instead of only truncating the reward. These
# are engine settings, not protocol quantities: the sampling distribution, the
# loss and the batch shape are as on the transformers path.
GENERATION_BACKEND = "vllm_colocate"
# vLLM's own budget, as a fraction of total device memory, while it generates. With
# sleep mode on it holds this only during generation: at level 2 it discards its
# weights and KV cache before the trainer's forward and backward passes, and TRL
# re-pushes the current weights on wake-up. So the trainer's activation peak and
# vLLM's budget are never resident together. On a 40 GB A100 (42.4e9 bytes) 0.3 is
# 12.7e9 bytes: a 1.7B model's bf16 weights (3.4e9), vLLM's profiling and CUDA
# graph overhead (about 1.5e9) and a KV cache of about 7.8e9 bytes, which holds all
# 32 rollouts of a step at once even for SmolLM2-1.7B's 196 KB per token. During
# generation the trainer keeps only its weights, the reference model and Adam
# state (about 13.6e9 bytes at 1.7B in bf16), so generation peaks near 26e9. The
# step's peak stays the trainer's own forward/backward peak.
VLLM_GPU_MEMORY_UTILIZATION = 0.3
VLLM_ENABLE_SLEEP_MODE = True
# Rollout token cap (transfer-rulings T22). Pinned to the pass@k budget instead of
# TRL's default of 512: at 512, 70% of Qwen2.5-0.5B's smoke rollouts were cut off
# before </answer> and scored 0, though the same model had 768 tokens at pre-eval.
MAX_COMPLETION_LENGTH = PASSK_MAX_TOKENS
# Rollout sampling filters, pinned to TRL's defaults so the GRPO config and the
# difficulty calibration (which must sample exactly as rollouts do, T22) read the
# same values: no top-p, top-k or min-p filter and no repetition penalty, the
# distribution GRPO has sampled from since Stage 0.
ROLLOUT_TOP_P = 1.0
ROLLOUT_TOP_K = 0
ROLLOUT_MIN_P = 0.0
ROLLOUT_REPETITION_PENALTY = 1.0
# vLLM otherwise sizes its context from the model config (131072 tokens for
# Llama-3.2-1B and Qwen2.5-1.5B) and refuses to start unless one sequence of that
# length fits in the KV cache. Prompts are at most about 250 tokens, so 2048 holds
# prompt plus the 768-token completion cap with room; check_fits_vllm_context
# refuses any item set for which a completion could be cut short by the context.
VLLM_MAX_MODEL_LENGTH = 2048
# TRL defaults this to True whenever use_vllm is set: it reweights each sequence's
# loss by the trainer-vs-vLLM likelihood ratio and masks sequences whose ratio
# exceeds 3. The transformers path has no such term, so it is off: the loss is
# unchanged and only the sampler moved.
VLLM_IMPORTANCE_SAMPLING_CORRECTION = False

# Environment variables that TRL's colocate init (RANK, LOCAL_RANK, WORLD_SIZE,
# MASTER_ADDR, MASTER_PORT) and vLLM's external_launcher backend
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) write process-wide. run_grpo restores them,
# so a later pass@k engine in the same process runs as a subprocess, exactly as the
# pre-eval did.
VLLM_ENV_KEYS = (
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@dataclass(frozen=True)
class GRPOSpec:
    model_path: str
    model_key: str
    items: list[TaskItem]
    verify_fn: Callable[[TaskItem, str], bool]
    max_steps: int
    num_generations: int
    learning_rate: float
    beta: float
    temperature: float
    seed: int
    output_dir: str
    save_steps: int | None
    # End each completion at the first occurrence of this string, inclusive.
    # None (gsm8k) means no stop. Generation stops there (transfer-rulings T21:
    # vLLM's stop with include_stop_str_in_output), so tokens after it are never
    # sampled and never enter the loss. The reward is also computed on the text
    # truncated there (T20), as defence in depth: the token that completes the
    # stop string can carry a few trailing characters.
    stop: str | None = None
    # The training reward (transfer-rulings T23). None (gsm8k) means the binary
    # verify_fn is the reward. The probe suites pass their shaped reward
    # (ProbeSuite.train_reward: 1.0 correct, 0.1 answer block, 0.0 none).
    # verify_fn still defines correctness, recorded beside it in the telemetry.
    train_reward_fn: Callable[[TaskItem, str], float] | None = None


def checkpoint_schedule(max_steps: int, save_steps: int | None) -> list[int]:
    if save_steps is None:
        return [max_steps]
    steps = list(range(save_steps, max_steps + 1, save_steps))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return steps


def generation_kwargs_for(stop: str | None) -> dict | None:
    """Extra vLLM SamplingParams for GRPO rollouts.

    include_stop_str_in_output is set for parity with pass@k's sampling_kwargs.
    TRL decodes the returned token ids rather than vLLM's text, and the token that
    completes the stop string is always among them, so the tag reaches the reward
    either way. Never add a seed here: TRL's colocate mode submits each prompt
    num_generations times with n=1, and a per-request seed would make every
    completion in a group identical.
    """
    if not stop:
        return None
    return {"stop": [stop], "include_stop_str_in_output": True}


def build_grpo_config_kwargs(spec: GRPOSpec) -> dict:
    """Keyword arguments for trl.GRPOConfig. Pure, so the config is testable on CPU."""
    return dict(
        output_dir=spec.output_dir,
        max_steps=spec.max_steps,
        learning_rate=spec.learning_rate,
        beta=spec.beta,
        temperature=spec.temperature,
        num_generations=spec.num_generations,
        max_completion_length=MAX_COMPLETION_LENGTH,
        top_p=ROLLOUT_TOP_P,
        top_k=ROLLOUT_TOP_K,
        min_p=None if ROLLOUT_MIN_P == 0.0 else ROLLOUT_MIN_P,
        repetition_penalty=ROLLOUT_REPETITION_PENALTY,
        # See PROMPTS_PER_STEP above: pins the batch shape so the protocol is
        # identical across models and machines.
        per_device_train_batch_size=spec.num_generations,
        gradient_accumulation_steps=PROMPTS_PER_STEP,
        save_steps=spec.save_steps or spec.max_steps,
        logging_steps=1,
        seed=spec.seed,
        bf16=True,
        report_to=[],
        # Rollout generation (transfer-rulings T21); see the constants above.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        vllm_enable_sleep_mode=VLLM_ENABLE_SLEEP_MODE,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        vllm_importance_sampling_correction=VLLM_IMPORTANCE_SAMPLING_CORRECTION,
        generation_kwargs=generation_kwargs_for(spec.stop),
    )


def check_fits_vllm_context(max_prompt_tokens: int, max_completion_length: int) -> None:
    """vLLM ends a sequence at max_model_len, so a prompt long enough to push prompt
    plus completion cap past it would silently shorten completions, a protocol
    change. Refuse instead."""
    if max_prompt_tokens + max_completion_length > VLLM_MAX_MODEL_LENGTH:
        raise ValueError(
            f"longest prompt ({max_prompt_tokens} tokens) plus max_completion_length "
            f"({max_completion_length}) exceeds vllm_max_model_length "
            f"({VLLM_MAX_MODEL_LENGTH}); completions would be cut short"
        )


class CompletionStats:
    """Counts what the reward function sees, for the run manifest. With a stop
    string, max_chars_after_first_stop shows whether generation actually stopped:
    a few characters at most if it did, hundreds if the model ran on to the cap."""

    def __init__(self, stop: str | None) -> None:
        self.stop = stop
        self.n = 0
        self.chars = 0
        self.n_with_stop = 0
        self.max_after = 0

    def wrap(self, reward_fn: Callable) -> Callable:
        def wrapped(completions, **kwargs):
            for c in completions:
                self.n += 1
                self.chars += len(c)
                if self.stop:
                    i = c.find(self.stop)
                    if i >= 0:
                        self.n_with_stop += 1
                        self.max_after = max(self.max_after, len(c) - i - len(self.stop))
            return reward_fn(completions=completions, **kwargs)

        return wrapped

    def as_dict(self) -> dict:
        return {
            "n_completions": self.n,
            "mean_chars": self.chars / self.n if self.n else None,
            "n_with_stop": self.n_with_stop if self.stop else None,
            "max_chars_after_first_stop": self.max_after if self.stop else None,
        }


def build_reward_fn(
    items: list[TaskItem],
    verify_fn,
    stop: str | None = None,
    train_reward_fn: Callable[[TaskItem, str], float] | None = None,
    correctness_sink: Callable[[list[float]], None] | None = None,
) -> Callable:
    """The TRL reward function. Returns train_reward_fn's reward per completion,
    or the binary verify_fn reward when it is None, on the completion truncated
    at stop. correctness_sink, if given, receives the strict 0/1 correctness of
    the same completions in the same order (transfer-rulings T23). A completion
    whose prompt matches no item scores 0.0 on both."""
    by_prompt = {it.prompt: it for it in items}
    if len(by_prompt) != len(items):
        raise ValueError(
            f"{len(items) - len(by_prompt)} item(s) share prompt text with another "
            "item; reward attribution would be ambiguous"
        )

    def reward_fn(completions, **kwargs):
        prompts = kwargs["prompts"]
        if len(prompts) > 1 and not any(p in by_prompt for p in prompts):
            raise ValueError(
                f"no prompt in a batch of {len(prompts)} matched any known item; "
                "prompt text is not reaching the reward function as expected, so every "
                "reward would be 0.0 and the run would silently produce a null result"
            )
        out, correct = [], []
        for prompt, completion in zip(prompts, completions):
            item = by_prompt.get(prompt)
            if item is None:
                out.append(0.0)
                correct.append(0.0)
                continue
            text = truncate_at_stop(completion, stop)
            ok = 1.0 if verify_fn(item, text) else 0.0
            correct.append(ok)
            out.append(ok if train_reward_fn is None else float(train_reward_fn(item, text)))
        if correctness_sink is not None:
            correctness_sink(correct)
        return out

    return reward_fn


def run_grpo(spec: GRPOSpec, trainer_factory=None, peft_config=None) -> list[StepRecord]:
    if peft_config is not None:
        raise ValueError(
            "LoRA/PEFT is forbidden: it constrains weight movement, which is the "
            "quantity this benchmark measures."
        )

    recorder = GroupedRewardRecorder(num_generations=spec.num_generations)
    stats = CompletionStats(spec.stop)
    reward_fn = recorder.wrap(
        stats.wrap(build_reward_fn(
            spec.items, spec.verify_fn, stop=spec.stop,
            train_reward_fn=spec.train_reward_fn, correctness_sink=recorder.record_correct,
        ))
    )

    if trainer_factory is None:
        from datasets import Dataset
        from transformers import AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        entropy_probe = EntropyProbe(
            tokenizer, [it.prompt for it in spec.items[:ENTROPY_BATCH_SIZE]]
        )
        config = GRPOConfig(**build_grpo_config_kwargs(spec))
        check_fits_vllm_context(
            max(len(tokenizer(it.prompt).input_ids) for it in spec.items),
            config.max_completion_length,
        )
        dataset = Dataset.from_dict({"prompt": [it.prompt for it in spec.items]})

        def trainer_factory(**kw):
            trainer = GRPOTrainer(
                model=spec.model_path, args=config, train_dataset=dataset, **kw
            )
            # The colocated vLLM engine is built in-process after Trainer.__init__
            # has seeded torch with spec.seed, and vLLM's worker re-seeds torch's
            # global generators with its engine seed, which TRL fixes at 0. Unseeded
            # vLLM requests sample from that global CUDA generator, so without this
            # every run seed would draw the same random stream (transfer-rulings T21).
            set_seed(spec.seed)
            return trainer
    else:
        entropy_probe = _NullEntropyProbe()

    callback = TelemetryCallback(
        recorder=recorder,
        entropy_probe=entropy_probe,
        model_getter=lambda: getattr(trainer, "model", None),
    )
    saved_env = {key: os.environ.get(key) for key in VLLM_ENV_KEYS}
    trainer = None
    sleep_pool_bytes = None
    try:
        trainer = trainer_factory(reward_funcs=[reward_fn], callbacks=[callback])
        model_obj = getattr(trainer, "model", None)
        if model_obj is not None and hasattr(model_obj, "peft_config"):
            raise ValueError(
                "trainer was constructed with a PEFT/LoRA model, which this benchmark "
                "forbids: LoRA constrains weight movement, the quantity under measurement."
            )
        trainer.train()
        trainer.save_model(spec.output_dir)
    finally:
        # The colocated engine runs in this process. Shut it down explicitly
        # before anything else touches the GPU: dropping the trainer alone leaves
        # its weights, KV cache and distributed state alive (vLLM gc.freeze()s
        # them at startup), and the post-eval's engine would then fail its
        # free-memory check (issue #33, transfer-rulings T21).
        # Read the sleep pool's size first: shutdown releases it (T22).
        sleep_pool_bytes = _current_sleep_pool_bytes()
        _shutdown_colocated_vllm(trainer)
        _restore_env(saved_env)

    Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_key": spec.model_key,
        "model_path": spec.model_path,
        "algorithm": "grpo",
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "num_generations": spec.num_generations,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "prompts_per_step": PROMPTS_PER_STEP,
        "per_device_train_batch_size": spec.num_generations,
        "gradient_accumulation_steps": PROMPTS_PER_STEP,
        "learning_rate": spec.learning_rate,
        "beta": spec.beta,
        "temperature": spec.temperature,
        "seed": spec.seed,
        "n_items": len(spec.items),
        # Rollouts come from a colocated vLLM engine that stops at
        # generation_stop (transfer-rulings T21); the reward is also scored only
        # up to completion_stop (T20). Both are the suite's stop, None for gsm8k.
        "generation_backend": GENERATION_BACKEND,
        "generation_stop": spec.stop,
        "include_stop_str_in_output": spec.stop is not None,
        "completion_stop": spec.stop,
        # Training reward (transfer-rulings T23): "shaped" on the probe suites,
        # "binary" (the verifier) on gsm8k. The step telemetry's mean_reward is
        # over this reward; mean_correct is over the verifier.
        "train_reward": "binary" if spec.train_reward_fn is None else "shaped",
        "train_reward_scores": None if spec.train_reward_fn is None else {
            "correct": CORRECT_SCORE, "format": FORMAT_SCORE, "no_answer": NO_ANSWER_SCORE,
        },
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "vllm_enable_sleep_mode": VLLM_ENABLE_SLEEP_MODE,
        "vllm_max_model_length": VLLM_MAX_MODEL_LENGTH,
        "vllm_importance_sampling_correction": VLLM_IMPORTANCE_SAMPLING_CORRECTION,
        # Bytes the colocated engine holds in vLLM's sleep-mode pool; torch counts
        # them as allocated even while asleep (transfer-rulings T22).
        "vllm_sleep_pool_bytes": sleep_pool_bytes,
        "completion_stats": stats.as_dict(),
        "expected_checkpoints": checkpoint_schedule(spec.max_steps, spec.save_steps),
        "n_step_records": len(callback.records),
    }
    (Path(spec.output_dir) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    records = list(callback.records)
    # Break the closure before dropping the trainer. model_getter closes over the
    # `trainer` variable and the trainer holds the callback, so after return the
    # trainer (model, reference model, optimiser state) would survive in a
    # reference cycle, which reference counting cannot free. It would stay on the
    # GPU until the cyclic GC happened to run, i.e. not reliably before the probe
    # initialises a vLLM engine at 0.85 memory utilisation in the same process,
    # which is the OOM issue #33 describes. Breaking the closure and collecting
    # explicitly frees it deterministically.
    callback.model_getter = lambda: None
    del callback, model_obj, trainer
    _release_gpu()
    return records


def vllm_sleep_pool_bytes(allocator) -> int | None:
    """Total bytes allocated in vLLM's sleep-mode pool (CuMemAllocator), awake or not.

    Why peak_train_bytes over-reports (transfer-rulings T22): with sleep mode on,
    vLLM allocates its weights and KV cache through a torch MemPool backed by its
    own cuMem allocator. Sleeping unmaps and releases the physical memory behind
    those allocations but keeps the tensors, so torch's caching allocator still
    counts every byte as allocated. torch.cuda.max_memory_allocated therefore
    reports the trainer's peak PLUS the whole pool, whichever phase the peak
    falls in, which is how Gemma-3-1B reported 45.6e9 on a 42.4e9-byte card.
    Subtracting this figure gives the trainer-phase peak; the generation-phase
    peak (trainer state plus the awake pool) is bounded by the raw figure.
    None when there is no allocator (CPU, or no colocated engine was built).
    """
    if allocator is None:
        return None
    return int(sum(data.handle[1] for data in allocator.pointer_to_data.values()))


def _current_sleep_pool_bytes() -> int | None:
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception:
        return None
    try:
        return vllm_sleep_pool_bytes(CuMemAllocator.instance)
    except Exception:
        return None


def _shutdown_colocated_vllm(trainer) -> None:
    generation = getattr(trainer, "vllm_generation", None)
    llm = getattr(generation, "llm", None)
    if llm is None:
        return
    shutdown_vllm_engine(llm)
    generation.llm = None


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _NullEntropyProbe:
    def measure(self, model) -> float:
        return 0.0


def _release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
