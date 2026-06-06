# LEARNINGS.md — P11.H1: thinking-metar1-metacognitive-v0

**Date**: 2026-04-18 (post-kill — supersedes 2026-04-14 design-review draft)
**Status**: KILLED (preemptive)

## Core Finding

H1 preemptively killed. Sixth consecutive `mlx_lm.lora` Gemma-4 reasoning-adapter kill
today (F0 → H0 → B0 → C0 → D0 → H1). All three KCs fail: K1520 (thinking reduction) bug-passable,
K1521 (vs H0) vacuous after H0 regressed to 47.6%, K1522 (PLAN structure) entangled with
protocol pathology. No new mechanism demonstrated — H1 replicates B0's training harness bug.

## Why

1. **Protocol bug replicated verbatim.** `run_experiment.py:260` emits
   `f"<|channel>thought\n...<channel|>..."` as `assistant.content` for `mlx_lm.lora` SFT.
   Identical pattern in B0:267 (measured −15.3pp MMLU-Pro, −71% thinking) and D0:267 (preemptive).
   `mlx_lm.lora` has no channel-protocol awareness — it tokenises the delimiters as text.
2. **Relative-to-upstream KC became vacuous.** K1521 was designed when H0 was predicted ≥65.1%
   MMLU-Pro. H0 regressed to 47.6% in-run. A trivial no-op H1 satisfies "≥ H0". The KC no
   longer measures "metacognitive structure preserves quality" — it measures "H1 fails ≤ as
   badly as broken H0".
3. **Theorem 2 premise weakened.** H0 catastrophically forgets humanities (eng 13.3%, phil
   20.0%) — Phase 1 stratifies by category, so the training-data-quality assumption breaks
   per-category even if Q_H0 > Q_base holds in aggregate.

## Implications for Next Experiment

1. **P11.HARNESS (unblocks C0/D0/H1/M0/H0-v2):** one canonical `mlx_lm.lora`-compatible
   serialization for thinking-channel SFT on Gemma 4. Acceptance: MMLU-Pro+thinking ≥ base−2pp
   on a 50-trace pilot. Three candidate paths: (i) strip channel tokens from training targets;
   (ii) switch to `<think>...</think>` text format (H0 used this — K1519 PASS); (iii) custom
   MLX SFT loop respecting the chat template.
2. **KC design rule:** never phrase KCs relative to an unmeasured upstream (`≥ H0_predicted`).
   Always `≥ base − ε` against an in-run locally-measured baseline. Makes KC falsifiable even
   when upstreams regress.
3. **F#530 baseline-reconciliation thread still open** (62.1% cited vs 40.7% measured); should
   close by the P11.HARNESS re-measurement. H1-v2 must NOT pre-register on F#530.
4. **Theorem 2 recompute required:** Q_H0 and Q_base are measured quantities post-harness-fix,
   not cited — re-derive before re-claiming H1.
