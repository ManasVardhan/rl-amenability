# Git history rewrite: AI attribution trailers removed

On 2026-09-27 the repository owner directed that AI attribution trailers be stripped
from commit history, per the standing rule in their `CLAUDE.md`:

> NEVER add `Co-Authored-By: Claude` (or any Claude/Anthropic co-author) trailer.
> NEVER add "Generated with Claude Code" or similar attribution lines.

Two commits carried them, made before a conflict between that rule and a harness-level
instruction surfaced. Ruling R10 standardised on `CLAUDE.md` for all later commits, so
only these two were affected:

- `7aa120e` Add design spec for RL-amenability benchmark
- `86d2da4` Add Stage 0 implementation plan

## Consequence

Those are the first two commits in the repository, so rewriting them changed the SHA of
every descendant. All 33 commits on `main` and `stage0-probe-harness` have new hashes.
File contents are byte-identical: `git diff` between the pre-rewrite and post-rewrite
branch tips is empty.

Commit ranges cited in GitHub issue comments predate the rewrite and name commits that no
longer exist. The mapping below resolves them.

## Recovery

The pre-rewrite history is preserved locally at:

- tag `backup-before-trailer-strip` (the branch tip)
- branch `backup-main`

These are intentionally NOT pushed. Delete them once the rewrite is confirmed good.

## Old to new SHA mapping

```
2f0a0be -> f36ec99  fix: pool Gate A on within-family ranks, not raw z-scored values
ea2d026 -> 8141a0f  feat: Gate A and Gate B evaluation against pre-registered thresholds
6540d7b -> a79040e  fix: record git commit provenance in freeze manifest (R29, R30)
2913151 -> fcbac2c  feat: pre-registration freeze with tamper detection
151ea89 -> c2c0e36  feat: small-N statistics with bootstrap CIs and permutation tests
1269ba7 -> a74de8a  feat: four pre-registered baselines including naive curve extrapolation
9067bb6 -> c38f216  docs: update status table through task 11
227f7be -> ab25465  fix: add guards against non-finite values in scoring and telemetry
3bc7622 -> a59f799  feat: pre-registered untuned amenability score
b96087d -> b7d8fc8  fix: guard against PEFT/LoRA model injection via trainer factory in sft.py
d05b76b -> 9e43e1d  feat: rejection-sampling SFT arm using self-generated verified traces
cacfa0e -> a51ea3c  docs: update status table through task 9
9eab884 -> 83ea445  fix: GRPO runner writes run manifest, closes LoRA and prompt-matching gaps
66d5962 -> a7c2344  feat: GRPO runner shared by probe and full run, LoRA explicitly refused
762d94f -> 6809571  feat: telemetry callback with grouped zero-advantage tracking
301a239 -> 6997a37  docs: add README and Stage 0 execution decision log
27d411d -> e232316  test: add EntropyProbe mock-based coverage
16a9de0 -> 6d89485  feat: trainer-independent policy entropy measurement
defc5cb -> 5168518  fix: raise on empty telemetry steps instead of producing nan feature
9d0114c -> bcd71e8  chore: commit uv.lock to pin the dependency graph
acabc25 -> 16d2b17  feat: probe telemetry schema and mechanism-grounded feature extraction
65eb8a5 -> 1272247  fix: add guards against silent misattribution and correct pass_at_k docstring
ce9449c -> 86ab842  feat: unbiased pass@k estimator and injectable vLLM evaluator
912277d -> bf7d28b  feat: GSM8K target suite with numeric-tolerant verifier
784576e -> ff3ef8e  feat: countdown probe suite with exact solver and sandboxed verifier
6ce3971 -> ae4cd9e  test: add atomicity test for failed registration
bfa808d -> 78847d5  feat: suite registry enforcing probe/target task disjointness
7f710c8 -> f51b3b6  fix: correct size-band test message and move vllm to optional gpu dependencies
6e8bedd -> 98fa010  docs: state the roster's size band as a nominal class with actual bounds
f33885a -> 9f24545  feat: project scaffolding and validated 10-model roster
3c9b85f -> 53dd37f  chore: ignore SDD workspace
86d2da4 -> cc70c0d  Add Stage 0 implementation plan
7aa120e -> 2d823f4  Add design spec for RL-amenability benchmark```
