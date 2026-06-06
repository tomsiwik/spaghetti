# LEARNINGS: exp_followup_budget_forcing_baseline_fix

## Core finding
F#530's 62.1 % MMLU-Pro base+thinking baseline on Gemma 4 E4B 4-bit **reproduces** at
63.2 % (n=280, seed=42) under identical config. The 15.4 pp gap observed in
`exp_p10_budget_forcing` (46.7 %) is **config drift**, not model/mlx-lm drift.

## Why
Three independent config differences between F#530's pipeline and
`exp_p10_budget_forcing`'s pipeline compounded:
1. **Data pool**: 20 %-held-out eval_df vs full test.parquet sampling.
2. **Per-category N**: seed=42, n=15 ≠ seed=42, n=20 (different pandas draws).
3. **Prompt text**: "Think step by step..." vs "Do not explain." — the former
   conflicts with `enable_thinking=True` by inviting reasoning in the answer
   channel, reducing thinking-channel usage.

Each of these can shift binomial accuracy by 5–10 pp. Together they straightforwardly
produce the observed gap.

## Implications for next experiment
- **Pin F#530 at 62.1 %** for any experiment that references base+thinking Gemma 4 E4B.
  `exp_p11_meta_r1_metacognition`'s `BASE_ACCURACY_REFERENCE=0.621` was correct
  (stale only if the in-run config diverged from F#530).
- **Config-drift triple** (data pool, per-cat N, prompt text) is the diff to check
  first whenever a replication attempt shifts >2 σ from the anchor. Apply this
  check proactively when citing a cross-experiment baseline number.
- **Budget-forcing "cliff effect"** (`exp_p10_budget_forcing`) is unaffected — that
  experiment's KILL verdict on B=128–512 (10–12 % catastrophic) stands; this
  experiment only explains the B=2048 "46.7 % vs 62.1 %" anchoring gap.

## Antipattern candidate (non-blocking, for analyst synthesis)
`cross-experiment-baseline-cited-without-config-match`: citing a prior experiment's
headline number without verifying the eval config (data pool + per-cat N + prompt
text + thinking knobs) matches. If not guaranteed, Wilson-95 % CI is the correct
expectation, not the point estimate. Anchor: this experiment + `exp_p11_meta_r1`
stale-reference case. Not yet recurring-worthy; register as candidate.

## Reusable building block (positive)
`load_and_split_data() → evaluate(per_cat=N) → Wilson-CI KC` is a clean replication
harness. Any future base-model drift check can reuse it by varying the thinking /
prompt / max_tokens arguments only. First repo instance of pre-registering a
Wilson-95 % binomial CI as a target-direct KC threshold.

## Platform notes
- mlx-lm 0.31.2 + Gemma 4 E4B 4-bit + MMLU-Pro at n=280, max_tokens=2048,
  enable_thinking=True → ~57 min wall-clock on M5 Pro 48 GB unified memory.
- Peak memory 5.13 GB active (well under the `total − 8 GB` soft cap).
- No OOMs, no retries, no stability issues across the 280-question run.

## Confidence
**High.** Direct target-metric measurement with pre-registered Wilson-95 % CI. All
three KCs PASS; math-category PASS is an *exact* (0.85 ≡ 0.85) match with F#530;
biology and engineering categories also match exactly. The +1.1 pp overall shift
from F#530's 0.621 is inside 1 σ of the binomial CI.
