# REVIEW-adversarial.md — T6.2: Crystallize Domain Adapters

**Verdict: PROCEED** (with caveats)

---

## What Passes

1. **Mathematical foundation solid**: Theorem 1 (LLN noise reduction) is correctly derived — E[||ε̄||²] = σ²/N is standard. QED is valid. Corollary (cosine improvement) follows correctly.
2. **PAPER.md has prediction-vs-measurement table**: All 4 kill criteria with predictions vs measurements. Finding status SUPPORTED is appropriate.
3. **Kill criteria match evidence**: results.json confirms K1120-K1123 all PASS. No fabrication.
4. **Literature grounding**: Cites Model Soup (2203.05482), Task Arithmetic (2212.04089), FedAvg — all directly applicable.

---

## Adversarial Concerns (non-blocking)

### 1. Synthetic-only evaluation — no behavioral test
All adapters are canonical B_D* + Gaussian noise(σ_frac=0.5). The math trivially holds by construction — crystallization is **guaranteed** to improve cosine to the canonical when averaging i.i.d. noise. There's no empirical content beyond "averaging reduces variance," which is definitionally true.

**Non-blocking because**: This is a Type 1 verification experiment. The theorem must be verified in controlled conditions before real-user evaluation. T6.3 (base promotion) is explicitly flagged as the stage for behavioral evaluation.

### 2. Suspicious uniformity — identical results across all 5 domains
Every domain shows exactly cos_crystal=0.9806, delta=6.5pp, norm_ratio=1.020. This is expected given the identical σ_frac=0.5 applied symmetrically, but it signals that domain-specific properties (adapter magnitude, std) are not differentiating the result. Real users would have heterogeneous noise levels.

**Non-blocking because**: T6.1 and T6.2 use the same synthetic infrastructure. The framework is consistent. Heterogeneous σ testing belongs in T6.3+.

### 3. Prediction magnitude off (+8.2pp predicted vs +6.5pp measured, -1.7pp gap)
The first-order cosine expansion underestimates denominator shrinkage. The paper acknowledges this. Direction correct, magnitude 21% off.

**Non-blocking because**: The prediction still correctly identifies the sign and rough scale of the improvement. The 1.7pp gap is within the approximation error of the first-order expansion (MATH.md derives exact formula, approximates norm ratio).

### 4. canonical included in mean_cos_user (inflated baseline)
mean_cos_user=0.9156 includes the canonical adapter (cos=1.0) as user 0. This makes the "baseline" artificially high (mean pulled toward 1.0) and the improvement claim (+6.5pp) conservative but non-standard. Real deployment would not include the canonical as a user adapter.

**Non-blocking because**: The comparison is still valid for the defined experiment. The crystallized adapter still outperforms the average including the perfect baseline — a stronger claim than necessary.

---

## Verdict

**PROCEED**. Theorem verified in controlled conditions. PAPER.md complete. All kill criteria pass. Caveats documented above should be addressed in T6.3 (base promotion with real generation quality test).

**T6.3 must include behavioral evaluation** (generation quality or MMLU) to earn finding status beyond PROVISIONAL.
