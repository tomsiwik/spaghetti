# REVIEW-adversarial.md: SFT-Residual M2P at 4B

**Verdict: PROCEED**

Experiment type: frontier-extension. Finding status: supported. Both appropriate.

---

## Core Validity

**Prediction-vs-measurement table:** Present and complete. 5/5 metrics match predictions.

**Kill criteria:** All 3 PASS with correct evidence in results.json.
- K972: init_quality_ratio=1.00 ≥ 0.80 ✓
- K973: quality_ratio=1.175 ≥ 0.60 ✓ (passes by large margin)
- K974: grad_norm=1.804 > 0 ✓

**Theorem 1 (Quality Floor):** VERIFIED EXACTLY. Zero-init heads guarantee B_applied = B_sft at step 0. init_accuracy=0.7300 = sft_accuracy=0.7300 to 4 decimal places. Structural guarantee holds.

**MATH.md inconsistency (non-blocking):** The proof uses small-variance σ init, but the implementation used zero-init. Zero-init is strictly stronger — makes Theorem 1 exact rather than approximate. The theorem holds either way; the paper correctly describes the actual implementation.

---

## Non-Blocking Issues

**1. Statistical significance of M2P > SFT overstated**

The paper claims "exceeding SFT quality" and "VERIFIED: quality_ratio=1.175 > 1.0". However:
- 74.4% vs 73.0% at n=500 = 372 vs 365 correct answers
- Two-proportion z-test: z ≈ 0.50, p ≈ 0.62 (not significant)
- This mirrors the 0.6B v4 finding where the same framing was corrected

The kill criterion K973 requires quality_ratio ≥ 0.60 — this passes with large margin regardless. The KEY structural result is that quality_ratio > 0 (not -0.187 like v5). The "exceeds SFT" framing in Theorem 2 verification is not supported at n=500.

Appropriate framing: "M2P matches SFT quality (74.4% vs 73.0%, difference not significant at p<0.05) while the previous architecture degraded to negative quality_ratio."

**2. peak_memory_gb=0.0 in results.json**

Measurement artifact — scratchpad records 17.91 GB. Minor instrumentation bug, does not affect finding.

**3. m2p_params=808M**

808M parameter M2P network for a 4B model. Not flagged as an issue in experiment scope (this is v6 architecture, not the VeRA reduction experiment), but the parameter overhead is relevant context for the research program.

---

## Structural Significance

This finding resolves the 4B scaling failure (third consecutive failure). The mechanism is proven: a residual connection in weight space makes failure structurally impossible at init. This is a genuine architectural fix, not a hyperparameter tweak.

The finding enables: (1) scaling the composition experiments to 4B, (2) the SFT-residual pattern as a baseline for future M2P variants at larger scales.

**Status: supported** is correct for frontier-extension with verified theorem + empirical support.
