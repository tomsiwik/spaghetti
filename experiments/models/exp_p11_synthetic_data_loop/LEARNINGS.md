# LEARNINGS.md — P11.I0: Synthetic Reasoning Data Generation Loop (STAR)

**Date**: 2026-04-17 (supersedes 2026-04-14 design-phase draft)
**Status**: KILLED (preemptive, 7th consecutive mlx_lm.lora Gemma-4 reasoning-adapter kill)
**DB KCs**: K1523/K1524/K1525 all FAIL

---

## Core Finding

I0 is the 4th confirmed instance of **antipattern-018 (CHANNEL-TOKENS-AS-SFT-TEXT)** at `run_experiment.py:278-281`, byte-equivalent to B0:267 / D0:267 / H1:260. Running I0 without a shared-harness fix would reproduce B0's measured regression (−15.3pp MMLU-Pro, −71% thinking suppression) with zero research value. Two additional independent drivers (dead upstream G0, structurally unreachable K1545) make the kill over-determined. A new pre-registration integrity failure mode (DB KCs K1523/1524/1525 vs MATH.md KCs K1544/1545/1546) was also surfaced but is tracked as "flag, pending recurrence" per Reviewer guidance.

## Why

Four-for-four Gemma-4 reasoning-adapter experiments today (B0, D0, H1, I0) share the same mlx_lm.lora SFT harness that treats Gemma's `<|channel>thought...<channel|>` as plain UTF-8 text. The trainer has no protocol awareness; the eval-time chat template invokes thinking as a protocol. Train/eval format mismatch collapses thinking generation and regresses base accuracy. STAR's self-improvement premise (yield ~ρ, ρ = base accuracy) also fails because the measured base (baseline_eval: 40.7%, F#560) is 21pp below the cited 62.1% (F#530, now stale) — K1545 R1 ≥ 59% requires +18pp from ~30 SFT examples, unsupported by STAR's published +2–5pp gains from hundreds of examples. Running I0 spends ~2–3h verifying a harness bug instead of fixing it.

## Implications for Next Experiment

1. **P11.HARNESS is the unblocking atomic unit.** Fixing antipattern-018 once unblocks B0-v2, C0, D0, H1, I0, J0, M0. Preferred fix: switch training targets to `<think>...</think>` (H0 K1519 existence proof) or strip channel tokens and keep inner body only. Skills `/mlx-dev` and `/fast-mlx` MUST be invoked before writing the harness code (§1011 blocking).
2. **Pre-register under measured base, not cited base.** All future reasoning-adapter KCs must use in-run baseline_eval numbers; K1545-style absolute thresholds (≥ 59%) should be replaced with `≥ base + ε` (STAR-compatible floor). F#560 baseline-reconciliation thread carries forward until a harness-fixed re-measurement closes it.
3. **DB ↔ MATH.md KC sync is now a pre-flight item.** Single occurrence so far; if it recurs outside I0, promote to mem-antipattern-019. Immediate mitigation: before `experiment complete`, `diff <(experiment get ... | awk KC) <(grep ^K MATH.md)` and halt on mismatch.
4. **I0-v2 scope narrows once harness lands.** STAR self-improvement is a cleaner claim against a harness-fixed v0; the dead upstream G0 dependency can be replaced by "base model + fixed thinking format" as Phase 1 seed.
