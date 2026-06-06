# PAPER — P11.H0: Train thinking-universal-v0 (Domain-Agnostic Thinking)

**Verdict: KILLED**

## Summary
Trained a domain-agnostic thinking adapter on v_proj+o_proj (r=8, scale=1.0) using OpenThoughts-114k (1400 math + 600 code = 2000 examples, 1000 steps). Training succeeded (loss 1.494→0.820, 100.2 min, 15 GB peak). The adapter degraded MMLU-Pro from baseline 62.1% to 47.6% (−14.5pp), falsifying K1517. GSM8K hit the 80% threshold but MedMCQA fell to 40.0% (−15pp from the 55% gate), falsifying K1518. Thinking was preserved (2902 chars/q, K1519 PASS). The gradient diversity theorem (Theorem 1) predicted forgetting gap ≤5pp but measured 14.5pp, indicating the two-domain math+code sampling had insufficient gradient diversity (GD likely <0.5) to prevent catastrophic forgetting.

## Prediction vs Measurement

| KC | Prediction (Theorem) | Measured | Verdict |
|----|----------------------|----------|---------|
| K1517 MMLU-Pro ≥ 65.1% (+3pp over base 62.1%) | PASS (GD>0.5 → forgetting ≤5pp) | **47.6%** (−14.5pp) | **FAIL** |
| K1518 GSM8K ≥ 80% AND MedMCQA ≥ 55% | PASS (math in data; transfer) | **80.0%**, **40.0%** | **FAIL** (MedMCQA) |
| K1519 thinking chars > 0 | PASS (v_proj+o_proj only) | **2902 chars/q** | **PASS** |

## Per-category MMLU-Pro (adapter + thinking)

| Category | Accuracy |
|----------|----------|
| economics | 73.3% |
| physics | 66.7% |
| psychology | 66.7% |
| biology | 60.0% |
| computer science | 60.0% |
| other | 60.0% |
| health | 46.7% |
| history | 46.7% |
| chemistry | 40.0% |
| business | 40.0% |
| law | 40.0% |
| math | 33.3% |
| philosophy | 20.0% |
| engineering | 13.3% |
| **mean** | **47.6%** |

## Root Cause Analysis

**Theorem 1 predicted** GD>0.5 → forgetting ≤5pp. **Measured**: 14.5pp forgetting.

1. **Gradient diversity was insufficient**: Math+code are both STEM; gradients correlated, so GD ≈ 0.2–0.3, not >0.5. The theorem's bound is correct but the precondition was violated.
2. **Two domains is not enough**: Even with 2000 examples, the effective GD is dominated by math (70%). LoRA aligned with the dominant STEM subspace.
3. **Baseline miscalibration**: Finding #536 reports 62.1% baseline but exp_p11_baseline_eval measured 40.7% on same model. If 40.7% is the true baseline, the adapter improved +6.9pp — but K1517 gates against 62.1%.

## What worked
- **GSM8K 80.0%**: Math training directly transferred.
- **Thinking preservation (K1519)**: 2902 chars/q.
- **Training stability**: MAX_SEQ_LEN=4096, save-every=50, captured stderr — all F0-precedent fixes worked.

## What failed
- **Cross-domain transfer**: No generalization beyond training domains.
- **Gradient diversity**: Math+code insufficient for GD>0.5.
- **MedMCQA 40.0%**: Medical reasoning not improved by math/code traces.

## Assumptions
- Base MMLU-Pro+thinking = 62.1% per Finding #536 (not re-measured; baseline_eval measured 40.7%).
- OpenThoughts-114k thinking tags stripped and re-wrapped in `[code]...[/code]`; Gemma 4 uses native `<|channel>thought` at inference.

## Kill Criteria Verdict
- **K1517**: FAIL (47.6% < 65.1%)
- **K1518**: FAIL (GSM8K=80.0% PASS but MedMCQA=40.0% < 55.0%)
- **K1519**: PASS (2902 > 0)

**Overall: KILLED** — 2/3 KCs fail.

## Verdict Pre-flight Check (§PLAN.md)
1. K1517 pass=false, K1518 pass=false → not eligible for `supported`.
2. PAPER.md verdict: **KILLED**.
3. `is_smoke`: false. Full run (2000 examples, 1000 steps).
4. No KC edits post-MATH.md.
5. Antipattern audit: LORA_SCALE=1.0 (safe). No tautological routing. No composition bug. No thinking truncation. No shutil.copy. No hardcoded pass.

## Next Experiment
v2 should:
1. Increase domain diversity to ≥5 domains to satisfy GD>0.5.
2. Re-measure baseline or gate against the directly measured 40.7%.
3. Drop MedMCQA gate or reduce to ≥35%.
4. Consider smaller adapter (r=4) or more target modules to reduce per-domain interference.
