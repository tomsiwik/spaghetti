# LEARNINGS — P11.F0 exp_p11_s1k_reasoning_train_eval

**Experiment:** P11.F0 — Train math-s1k-reasoning-v0 + Register + Eval
**Verdict:** KILLED (2026-04-17)
**Analyst pass:** 2026-04-17

## Core Finding
`mlx_lm.lora` subprocess crashed at 1854.3 s (~31 min) before the first `save-every=200` checkpoint. No safetensors landed, so K1508/K1509 were never measured and K1510 fails by construction. Theorem 1 (epoch-count drift bound) is **not** falsified — the training run produced no data to test it against.

## Why
Three compounding process failures, not a theorem failure:

1. **Stderr discarded.** `run_experiment.py` invoked `mlx_lm.lora` with `capture_output=False`. When the subprocess exited non-zero the actionable error went to pueue stdout, and pueue had already rotated the task history by reviewer time (`pueue log 0` → "no finished tasks"). Root cause cannot be attributed with certainty.
2. **Long-sequence OOM (most likely hypothesis, unverified).** `MAX_TOTAL_CHARS=32000` → ≈8000-token sequences against `MAX_SEQ_LEN=8192` on a 40 GB MLX memory budget. A long s1K trace at the tail of the token-length distribution plausibly saturated the cache/weight working set on a single forward. Consistent with the crash landing well past the warm-up tier.
3. **`save-every=200` with crash before step 200.** No partial adapter survived for postmortem. With `save-every=50` an early checkpoint would have been available to smoke-eval.

Together these turn a recoverable runtime failure into an unrecoverable black-box KILL.

## Implications for Next Experiment

- **P11.F0.v2 (successor, recommended).** Same MATH.md and K1508/K1509/K1510 thresholds. Code-only changes:
  - Redirect subprocess stderr to `logs/mlx_lm_lora.stderr.log` (e.g. `stderr=open(path, "w")`); keep stdout streaming.
  - Lower `MAX_SEQ_LEN` to 4096 **or** filter training examples by token count (not char count — chars/tokens ratio varies across s1K traces).
  - `save-every=50` so a crash leaves a checkpoint.
  - Reconcile DB KC#1508 description (still "≥65%") with MATH.md/code (59%) before the run — REVIEW finding-g, non-blocking but cheap.
- **P11.G0 blast radius.** `exp_p11_grpo_improve` is `blocked-by` F0 and expects `adapters/math-s1k-reasoning-v0/*.safetensors`. Three options:
  1. Wait for F0.v2 (preferred — tightest causal chain with the MATH proof G0 cites).
  2. Re-scope G0 onto `math-gsm8k-knowledge-v0` (already persisted). Weakens "improve best reasoning adapter" framing — hypothesis changes.
  3. Re-scope G0 as base-model-only. Weakest option; drops the adapter-improvement story entirely.
- **Antipattern captured.** `capture_output=False` on long-running training subprocesses is now an explicit fix memory. Pueue log rotation is a hostile environment for postmortem; disk logs are the only reliable channel.

No literature reference needed — the failure mode is process-engineering. The OOM hypothesis is covered by standard memory-budget reasoning already noted in prior mem-antipattern entries.
