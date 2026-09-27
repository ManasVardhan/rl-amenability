# rl-amenability

A benchmark that scores a base language model by **how much it will gain from RL post-training**, before you spend the compute to find out.

Selecting a base model to post-train is currently guesswork. Practitioners pick by pretraining benchmark scores, but those measure what a model can already do, not how much it will improve. The two dissociate badly: Qwen2.5-Math-7B gains 21 to 27% on MATH-500 from *random or incorrect* rewards, while Llama3.1/3.2 and OLMo2 gain nothing from identical training.

## The idea

RLVR does not add capability. It converts latent breadth into realized reliability: RL-trained models win at low sampling budgets, base models catch up at high k. So **amenability is defined as conversion efficiency**: how effectively a fixed training budget turns pass@k breadth the model already has into pass@1 reliability, and how much breadth remains unconverted.

It is measured with a **short, real RL run** (a probe) rather than a training-free proxy, which is what lets it observe entropy collapse and plasticity loss directly instead of inferring them.

## Why this is not already solved

[RuDE](https://arxiv.org/abs/2605.11978) (May 2026) is the closest prior work and reports Pearson > 0.90 across 16 models. It is also unreplicable (no code release), emits a single non-algorithm-conditional scalar, and states outright that it cannot model training dynamics such as entropy collapse or reward hacking. That last limitation is the gap this targets.

## Two constraints that keep it honest

A short RL run trivially predicts a long RL run, because learning curves are smooth. "Reward at step 60 predicts reward at step 600" is not a finding. So:

1. **Probe tasks are disjoint from target tasks**, enforced in code by a registry that raises on overlapping task IDs, not by convention.
2. **The score must beat naive curve extrapolation.** Fit a curve to the probe trace and extrapolate: that is the baseline. The interesting cases are where it fails (entropy collapse, late plateaus, rank inversion).

The analysis is **pre-registered and hash-frozen** before any ground-truth run. `prereg/freeze.py` hashes the score, baselines and gates; the orchestrator refuses to run if any changed. At N=10 you cannot fit a scoring function honestly after seeing results, so it is specified before.

## Status

Stage 0 builds the harness and runs the positive control through two gates. See the [Stage 0 milestone](../../milestone/1) and [open issues](../../issues).

| | Task | Tests |
|---|---|---|
| ✅ | [1. Scaffolding and model registry](../../issues/1) | 7 |
| ✅ | [2. Suite registry with enforced disjointness](../../issues/2) | 6 |
| ✅ | [3. Countdown probe suite and sandboxed verifier](../../issues/3) | 10 |
| ✅ | [4. GSM8K target suite](../../issues/4) | 10 |
| ✅ | [5. Unbiased pass@k estimator](../../issues/5) | 10 |
| ✅ | [6. Probe telemetry and features](../../issues/6) | 12 |
| ✅ | [7. Policy entropy measurement](../../issues/7) | 9 |
| ✅ | [8. Telemetry callback and reward recorder](../../issues/8) | 8 |
| ✅ | [9. GRPO runner](../../issues/9) | 12 |
| 🔄 | [10. Rejection-sampling SFT arm](../../issues/10) | 7 |
| ⬜ | [11. Pre-registered amenability score](../../issues/11) | 10 |
| ⬜ | [12. Baselines the score must beat](../../issues/12) | 7 |
| ⬜ | [13. Small-N statistics](../../issues/13) | 10 |
| ⬜ | [14. Pre-registration freeze](../../issues/14) | 8 |
| ⬜ | [15. Gate A and Gate B evaluation](../../issues/15) | 9 |
| ⬜ | [16. Over-SFT variant plan](../../issues/16) | 8 |
| ⬜ | [17. Stage 0 orchestration and report](../../issues/17) | 7 |
| ⬜ | [18. Real runner and launch scripts](../../issues/18) | 3 (GPU) |

**84 tests green.** Only task 18 needs a GPU, so the entire pipeline is verifiable before any allocation is spent.

## The gates

Stage 0 costs ~130 GPU-hours and answers both kill questions before the remaining ~230 are committed.

- **Gate A:** does the probe recover the *known* amenability ordering across over-SFT'd variants, in both control families? Six checkpoints whose correct ordering is fixed a priori by the plasticity literature. This is a falsification test that does not depend on any correlation study.
- **Gate B:** does the score beat naive extrapolation?

Failing both means the probe design is wrong, at a cost of 130 GPU-hours rather than 355. Failing only B means the paper pivots toward failure-mode detection. The decisions are bound in advance in `prereg/stage0.md`.

## Scope, stated up front

Validated on the 0.5B to 1.7B nominal size class only. Single training target (GSM8K). Two algorithms (GRPO, rejection-sampling SFT). N=10 observational plus N=6 causal, with the observational arm preliminary by design. Verifiable-reward capability gain only: no claim about behavioral alignment, safety durability, or inference-time steerability.

## Layout

```
docs/superpowers/specs/     design spec (the authority)
docs/superpowers/plans/     Stage 0 implementation plan, 18 TDD tasks
docs/decisions/             execution rulings, with cost-if-wrong for each
src/amenability/
  suites/    probe (Countdown) and target (GSM8K) task suites
  eval/      pass@k estimator, small-N statistics
  probe/     telemetry, entropy, trainer callbacks
  training/  GRPO and SFT runners
  scoring/   score, baselines, gates  [frozen before Stage 0]
prereg/                     frozen analysis plan and hash manifest
```

## Running the tests

```bash
uv sync --extra dev
uv run pytest -v -m "not gpu"     # full CPU suite
uv run pytest -v -m gpu           # needs a GPU; add --extra gpu to sync
```

`vllm` lives in the `gpu` extra so the CPU suite installs on any platform.
