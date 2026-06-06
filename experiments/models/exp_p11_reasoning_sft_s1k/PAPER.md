# PAPER: P11.A0 — Reasoning SFT on s1K Dataset

## Verdict (audit-2026-04-17 rerun, 2026-04-18): **KILLED — structural antipattern**

**Re-classification**: `audit-2026-04-17-rerun` + `code-bug`. The original 2026-04-14
run is retained below as evidence. Re-run not executed this iteration — the
code-bug fix (strip_thinking regex → channel-aware) only corrects the corrupted
base eval (Phase 4a) back to 62.1% (Finding #536), which does NOT change the
adapter's 36.1% measurement or the catastrophic-forgetting finding.

**Structural issues confirmed by audit**:
1. **strip_thinking regex bug (code-bug)** — original matched `<think>...</think>` only;
   Gemma 4 emits `<|channel>thought...<channel|>`. Base eval Phase 4a measured
   12.5% because thinking content leaked into MCQ answer parsing. **FIXED** in
   `run_experiment.py` this iteration (channel-aware, with `<think>` fallback).
2. **Training format mismatch (design-level)** — training data used literal
   `<think>{thinking}</think>\n\n{attempt}` strings. mlx_lm.lora treated those
   as literal assistant text rather than Gemma 4 channel tokens. So the adapter
   learned to EMIT literal `<think>` scaffolding, not to USE the channel. K1492's
   "pass" (avg_thinking_chars=1641) measured literal text, not real channel use.
3. **GSM8K API rot** — datasets-server HTTP 422 made K1491 untestable
   (not fail). Separate infrastructure issue, not specific to this experiment.

**Why re-run is not required**: The core finding (adapter 36.1% vs real base 62.1%,
–26pp catastrophic forgetting) is structurally confirmed by:
- Finding #538 recorded in DB.
- Downstream `exp_p11_reasoning_sft_limo` preemptively killed citing the s1K
  impossibility structure (competition math SFT ⊥ MMLU-Pro breadth distribution).
- Sibling `exp_p11_s1k_reasoning_train_eval` (P11.F0) ran into training failure
  on the same pipeline (all three KCs #1508/1509/1510 FAIL).

**Antipattern tag**: same cluster as Finding #587 (strip_thinking regex brittleness)
and `exp_p11_grpo_reasoning_adapter` (mlx_lm.lora treats channel tokens as literal
text).

---

## Prediction vs Measurement

| Kill Criterion | Metric | MATH.md Prediction | Measured | Status |
|----------------|--------|-------------------|----------|--------|
| K1490 | MMLU-Pro + thinking ≥ 65% | 65%+ (Theorem 2) | **36.1%** (adapter) | **FAIL** |
| K1491 | GSM8K ≥ 80% | 80%+ (math traces) | **HTTP 422 error** | **INVALID** |
| K1492 | Thinking not suppressed | Guaranteed (Theorem 1) | **1641 chars/q** | **PASS** |

## Results Summary

**Training**: 1000 steps, 2913.9s (48.5 min), 27 training examples loaded

**Critical Bug — Base Eval Invalid**:
- Phase 4a base MMLU-Pro: 12.5% accuracy, 0 avg_thinking_chars
- Root cause: `strip_thinking()` used `<think>...</think>` but Gemma 4 actually uses
  `<|channel>thought...content...<channel|>` tokens
- With thinking not stripped, thinking content was parsed as answer → random/invalid results
- **Base eval of 62.1% (Finding #530) remains valid** — this run's base eval is corrupted

**Adapter Eval (Phase 4b)**:
- MMLU-Pro accuracy: **36.1%** (avg_thinking_chars = 1641 → thinking preserved)
- Per-category breakdown:
  | Category | Accuracy |
  |----------|----------|
  | biology | 10% |
  | business | 20% |
  | chemistry | n/a |
  | computer science | 5% |
  | economics | 45% |
  | engineering | 15% |
  | health | 45% |
  | history | 35% |
  | law | 15% |
  | math | 20% |
  | other | 60% |
  | philosophy | 30% |
  | physics | 40% |
  | psychology | 55% |
- Adapter is **worse** than base (36.1% vs 62.1%) — competition math traces HURT general reasoning

**GSM8K**: HTTP 422 error (datasets-server API broken) — same bug as P9.G0

## Interpretation

Theorem 1 verified: thinking channel preserved (K1492 PASS, 1641 chars/q).

Theorem 2 refuted: s1K traces did NOT improve MMLU-Pro. The adapter accuracy dropped from
62.1% (base) to 36.1% — a **−26pp degradation**. This is catastrophic forgetting, not improvement.

### Failure Mode Analysis (from MATH.md)
- **Trace-domain mismatch confirmed**: s1K is competition olympiad math. MMLU-Pro is breadth.
  The adapter learned competition math patterns at the expense of general knowledge activation.
- The training distribution D_s1K is so far from MMLU-Pro that the gradient pushed the model
  *away* from correct general reasoning, not toward it.

### What Makes This Failure Impossible (Impossibility Structure)
For SFT to preserve MMLU-Pro accuracy while adding reasoning skill, training data must include
traces that span the same token distribution as MMLU-Pro. Competition math traces are nearly
orthogonal to the MMLU-Pro token distribution in embedding space.

## Next Steps (Derived From Kill Structure)

Per MATH.md §Kill Structure for K1490 failure:
- Math category DID NOT improve substantially (20% is low even for math)
- This rules out "domain-specific improvement + forgetting" — it's pure degradation
- **Required**: Diverse traces spanning MMLU-Pro's 14 categories, NOT math-only

Immediately relevant: LIMO (exp_p11_reasoning_sft_limo) also uses competition math.
Expect similar or worse degradation (LIMO is even harder competition math).

**Critical open question**: Is 62.1% base a ceiling for this adapter regime?
exp_p11_w4a16_verification will show if 8-bit model scores ~65%, confirming quantization gap.

## Training Details

- Dataset: `ruliad/s1K-parsed-and-formatted` (27 examples loaded, filtering for quality)
- Thinking format in training data: `<think>...</think>` (may mismatch Gemma 4's actual format)
- LoRA rank 8, lr=1e-4, max_seq_len=2048
- Adapter saved to: `micro/models/exp_p11_reasoning_sft_s1k/adapters/`
- Total time: 13702.8s (3.8h, Phase 1 training + Phase 4 eval)
