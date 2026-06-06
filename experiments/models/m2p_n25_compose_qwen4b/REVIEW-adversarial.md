# REVIEW-adversarial.md — N=25 Domain Grassmannian Composition at 4B

**Experiment:** exp_m2p_n25_compose_qwen4b
**Reviewer:** Adversarial Reviewer
**Date:** 2026-04-08
**Verdict: PROCEED** (with caveats)

---

## Summary

All 3 kill criteria pass. MATH.md has proper Theorem/Proof/QED structure. PAPER.md has the required prediction-vs-measurement table. Evidence in results.json matches reported numbers. No fabricated results detected.

---

## What Passes

- **K981 PASS:** max|A_i^T A_j|_F = 1.38e-05 across all 300 pairs — verified in results.json. Matches bf16 quantization floor from Findings #404, #405.
- **K982 PASS:** 99.96% overall routing accuracy (2499/2500), 99.0% minimum per-domain. Exceeds both threshold (80%) and prediction (95%).
- **K983 PASS:** quality_ratio = 1.3125, exactly as predicted. Verified in results.json with accuracy=0.755.
- **Theorem 4 is logically airtight:** exclusive routing → N doesn't affect quality. Prediction of qr≈1.31 is confirmed.
- **Prediction-vs-measurement table is complete and accurate.**

---

## Issues (Non-Blocking)

### 1. The "scaling law" is partially circular (non-blocking caveat)

The quality_ratio is IDENTICAL across N=2, N=5, N=25 because **the same math M2P weights are loaded in all three experiments** (from m2p_qwen4b_sft_residual). This is correct and expected per Theorem 4, but the "scaling law confirmed" headline overstates the finding. We're not observing that quality is *invariant to N* — we're confirming that the same model gives the same output when correctly routed.

**Impact:** Theorem 4 is still correct and verified. But the "production-ready for 25+ domains" claim goes beyond what's shown. We've proven structural isolation and routing scale. We have NOT proven that 25 independently trained real-domain M2P adapters at 4B maintain qr≈1.31 each.

**Suggested caveat in finding:** "N=25 Grassmannian isolation and routing verified; quality invariance shown for math domain with pre-trained M2P; multi-domain behavioral quality at N=25 remains for future work."

### 2. 24/25 domains use B=0 (synthetic-only)

The A-matrix isolation is real. The routing is real. But the behavioral test (K983) only exercises the math domain — the other 24 domains are structurally present but behaviorally empty (B=0). The experiment proves the *infrastructure* scales to 25 domains, not that 25 independently useful domains compose without interference.

**Impact:** Finding status "supported" is correct (not "conclusive"). The isolation math is verified; the multi-domain behavioral quality claim is not yet tested.

### 3. Theorem 2 "monotone" claim is misleading for legacy pairs

Theorem 2 claims isolation improves monotonically. However, the worst pair (math×code, 1.38e-05) is IDENTICAL across N=2, N=5, N=25 because those matrices were NOT re-projected — they're inherited from prior experiments. The "monotone improvement" only holds for *newly added* pairs relative to existing ones. This is fine structurally (Gram-Schmidt still works) but the theorem statement should say "newly added domains achieve monotonically better isolation relative to existing ones."

### 4. No confidence interval on K983

quality_ratio=1.3125 at n=200. SE(accuracy) ≈ 0.030. No CI reported. For a supported finding this is acceptable but worth noting: the true quality_ratio 95% CI is approximately [1.03, 1.59] (error propagation through the ratio formula with base_acc=0.65, sft_acc=0.73).

---

## Verdict

**PROCEED.** The math is correct, predictions match measurements precisely, and the paper is complete. The issues are clarifications/caveats, not errors.

**Finding status: supported** (frontier extension: proven framework, N scaled from 5 to 25; B=0 for 24/25 domains means full behavioral verification is future work)

**Key result to record:** N=25 Grassmannian isolation (1.38e-05), TF-IDF routing (99.96%), and math quality (qr=1.3125) all verified. N_max=640 confirmed with 96.1% capacity remaining. Structural proof that composition scales to 25 domains at 4B.
