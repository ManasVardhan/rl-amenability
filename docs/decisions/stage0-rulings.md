# Stage 0 Execution Rulings

Decisions taken during subagent-driven execution of
`docs/superpowers/plans/2026-09-22-stage0-probe-harness.md`, in the order they
were made. Each names what was decided, why, and what it costs if the decision
turns out to be wrong.

Most of these are corrections to defects in the plan itself, surfaced either by
the pre-flight conflict scan or by per-task review. That is the expected
pattern: the plan was written in one pass, and review is where its gaps show.

## Pre-flight (before any code)

A conflict scan over all 18 tasks, checking every pair that shares a file or an
interface, and every task against its own text.

**R1. Feature branch, not a git worktree.**
Work happens on `stage0-probe-harness` in place. The repo is new,
single-purpose, has no concurrent work, and worktree indirection would
complicate SLURM and Modal path resolution in Task 18.
*Cost if wrong:* `main` is not independently checkoutable during execution.

**R2. Task 17 must call `baseline_naive`, not re-implement extrapolation.**
The plan text duplicated a logic block Task 12 already owns. The spec treats
naive extrapolation as one pre-registered baseline with one definition.
*Cost if wrong:* none material; prevents future drift between two copies.

**R3. `spearman_with_ci` gains an optional `n_perm` defaulting to `n_boot`.**
As written it ran 10,000 permutations regardless of the `n_boot` a caller
passed, so Task 15's gates would burn 20-40s per suite run for no statistical
benefit.
*Cost if wrong:* slightly less precise permutation p-value inside gate
evaluation, where only the point estimate is used.

**R4. Task 17's `main()` constructs `Stage0Config()` and passes it to `RealRunner(config)`.**
The plan showed `RealRunner()` with no arguments in Task 17 while Task 18
defined `__init__(self, config, ...)` with `config` required. Straight
TypeError at launch.
*Cost if wrong:* a TypeError caught immediately.

## Task 1: scaffolding and roster

**R5. `vllm` moves to a `gpu` optional extra.**
Declared as a base dependency, it made `uv sync --extra dev` fail on every
non-Linux machine. Verified that torch resolves fine on darwin and vllm is the
sole blocker. Import-time audit across all 18 tasks: `vllm`, `trl` and
`datasets` are imported lazily inside functions and are never needed for the
CPU test suite; only `torch` and `transformers` are needed at module level.
*Cost if wrong:* GPU environments must use `uv sync --extra dev --extra gpu`,
carried into Task 18's sbatch and Modal scripts.

**R6. The size-band check keeps bounds 0.3-1.8 and gets an honest message.**
The test asserted `0.3 <= params_b <= 1.8` while its failure message claimed a
"0.5-1.7B band". Two roster entries breach a literal reading: Qwen2.5-0.5B is
0.49B and SmolLM2-1.7B is 1.71B. But "0.5B" and "1.7B" are nominal size-class
names, not exact parameter counts, and the bound exists as a compute-budget
guard against a 7B model entering the roster. Fixed the lying message rather
than rejecting two valid models, and corrected the spec's wording so the
paper's limitations section is accurate.
*Cost if wrong:* the roster admits a model between 0.3B and 0.5B that a
stricter reading would exclude.

**R7. `models.yaml` joins `FROZEN_PATHS` (applies to Task 14).**
`load_registry` enforces no roster invariants at runtime, so a hand-edited
roster is undetectable. `prereg/stage0.md` already commits in prose to "no
dropping of models from the roster except for a documented infrastructure
failure"; freezing the file is what makes that enforceable rather than
decorative.
*Cost if wrong:* a legitimate roster change requires a documented unfreeze,
which is the intended behaviour.

## Task 2: suite registry

**R8. Add an atomicity regression test, taking the file past the plan's five-test list.**
Review hand-traced `register()` as genuinely atomic but showed the tests would
pass unchanged against a naive single-pass implementation that mutates then
raises. Spec section 5 makes probe/target disjointness a code-enforced
invariant; an invariant whose only protection is untested code regresses
silently on the first readability refactor.
*Cost if wrong:* one extra test of about six lines.

**R9. The reviewer's `local`-set Minor is incorrect; no change made.**
It claimed the `local` set was redundant with `self._seen`. It is not: `local`
catches duplicate task IDs within a single batch, which `self._seen` cannot,
precisely because the validation pass deliberately does not write to
`self._seen` before validating. Removing it would break a test and reopen the
atomicity hole.
*Cost if wrong:* none, code unchanged.

## Task 3: countdown suite

**R10. Commits in this repo carry no AI attribution trailers.**
A harness-level instruction tells agents to append `Co-Authored-By: Claude ...`
and `Claude-Session: ...`; the repo owner's `CLAUDE.md` forbids exactly that.
Standardised on `CLAUDE.md`, being the owner's explicit standing preference
about their own repository. Known inconsistency: two early commits
(`7aa120e`, `86d2da4`) carry the trailers because the harness instruction was
followed before the conflict surfaced. Not rewritten unilaterally.
*Cost if wrong:* the branch lacks session-provenance trailers on most commits.

## Task 5: pass@k evaluator

**R11. Add three input guards to `evaluate_passk`.**
`zip(items, completions)` truncates silently on length mismatch, and duplicate
task IDs collapse the prompts dict so every later completion is attributed to
the wrong item. Review judged this unreachable because `SuiteRegistry` enforces
uniqueness; that reasoning does not hold. Task 18's target-side evaluation uses
an items list from `load_gsm8k(...)` that is never registered, so the
uniqueness guarantee does not cover it. Safe today only because dataset indices
happen to be unique. A silent misattribution in the measurement layer would
corrupt every downstream number while every test stayed green.
*Cost if wrong:* raises on malformed input instead of silently producing wrong
numbers.

**R12. Fix the `pass_at_k` docstring.**
It claimed the value is "computed stably in log space"; it is a running product
of ratios. Cosmetic as code, but a false statement in a codebase whose numbers
go into a paper.
*Cost if wrong:* none.

**Declined:** clamping the estimator output to [0,1]. The range is
mathematically guaranteed by the guard branches, and a clamp would mask a
genuine estimator bug rather than reveal one.

## Task 6: telemetry and features

**R14. `extract_features` raises on an empty steps list.**
It previously returned a `FeatureVector` carrying
`zero_advantage_rate = nan` from `np.mean([])`, with only a RuntimeWarning. That
NaN is a scoring input, and because every NaN comparison is False it would not
even surface as an outlier during gate ranking: the run would produce something
that looks like a result and is not one. Confirmed reachable, not theoretical:
Task 9's runner returns records that accumulate only inside `on_log`, so a run
raising before its first logging step hands Task 18 an empty list.
*Cost if wrong:* a probe that legitimately produced no logging steps errors
instead of scoring, which is intended.

**R15. Do not change the `kl_per_gain` floor; document its saturation (applies to Task 14).**
`max(reward_gain, 1e-6)` makes every run whose reward declined collapse to
`retention_factor` near zero, so a small decline and a large decline are
indistinguishable. This is a real resolution loss and could matter for Gate A,
since heavily over-SFT'd control variants are the ones most likely to decline.
Not redesigning a pre-registered formula mid-execution on speculation: the
semantics are defensible, `conversion_rate` separates those variants
independently, and Gate A's outcome is what will reveal whether saturation
bites. Documenting before the freeze is honest; changing the formula after
seeing Stage 0 results would not be.
*Cost if wrong:* Gate A fails for an instrumental rather than a real reason,
diagnosable from this note.

**R13. Commit `uv.lock`.**
The plan never mentioned it. Reproducibility is a stated release artifact, and
TRL, transformers and vllm all change RL-relevant defaults between minor
versions, so an unpinned environment could shift Stage 0 results without any
change to this repository.
*Cost if wrong:* a 1.3MB file in git history.

## Task 7: entropy measurement

**R16. Add mock-model coverage for `EntropyProbe.measure()`.**
The plan tested only the pure `mean_next_token_entropy` function, leaving
tokenization, the device move, the `no_grad` wrapper and the train/eval restore
unverified. The restore is the dangerous one: a bug there does not fail a test,
it silently leaves the model in eval mode and disables dropout for every
subsequent training step. A measurement instrument corrupting the run it
measures, invisibly.
*Cost if wrong:* four extra tests using fakes.
