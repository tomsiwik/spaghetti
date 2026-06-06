# LEARNINGS.md — T5.1: User Local Training

**Status:** SUPPORTED (Finding #436, V2 strengthened)
**Date:** 2026-04-18 (V2 audit-rerun; V1 2026-04-10)

---

## Core Finding

Personal stylistic adapters are viable on consumer hardware. Rank-4 LoRA on
Gemma 4 E4B 4-bit trained for 1.3 min on 40 synthetic examples achieves
**60%** suffix-phrase compliance vs 0% base — a 60pp behavioral lift at
2.44 MB and < 2 min wall time on M5 Pro 48 GB.

## Why It Works

Low-rank sufficiency (Hu 2021 + Aghajanyan 2020): a single-phrase suffix
lives in an effectively rank-1 task direction, so rank-4 LoRA on the last
16 layers (q/k/v/o_proj) has ample capacity. The adapter injects a
direction the base distribution does not produce — confirmed by
`base avg_thinking = 2687 chars, 25/25 closed` (base genuinely reasoned
and answered without emitting the target phrase; V2 falsifies the V1
truncation-as-cause story).

## V1 → V2 Correction (mem-antipattern-008)

V1 measured 76pp with `MAX_TOKENS=120` — conflated style injection with
thinking-mode truncation. V2 surgical eval-only fix (`MAX_TOKENS` 120→4096,
`split_thinking()` instrumentation, K1097 gated on
`base_thinking_chars > 0`) recovered the pure style-injection effect:
**60pp (16pp was the confound)**. Theorems unchanged. Headline lift
drops but now reflects the real mechanism.

## Retained Caveat for T5.2 / Composition

Adapter `avg_thinking_chars = 0` — the adapter suppresses Gemma 4's
native `<|channel>` thinking because the training corpus contains no
thinking tokens. Not a confound for T5.1 (marker-presence KC is
orthogonal), but **any downstream routing / composition experiment
inherits a latent "kill-thinking" side-effect** unless user adapters are
trained with thinking-aware data or evaluated only on the post-thinking
slice.

## Implications for Next Experiment

T5.2 (user adapter validation / multi-adapter coexistence) must: (a)
train with thinking-aware conversation data *or* deterministically keep
`<|channel>` traces in gold answers so the adapter learns to preserve
them; (b) measure `adapter_thinking_chars` and treat a drop to 0 as a
blocking side-effect, not a benign artifact; (c) confirm T3.6/T3.7
hot-add/remove invariance with the corrected adapter and re-test under
safety-composition (exp_prod_safety_under_composition). Size/time
budget ($0, < 2 min, < 3 MB) is locked in; remaining open question is
behavioral coexistence without collateral thinking suppression.
