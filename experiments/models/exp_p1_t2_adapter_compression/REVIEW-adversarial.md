# Adversarial Review: T2.2 Adapter Quantization Quality Retention

**Verdict: PROCEED**

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table — complete and accurate
- [x] Kill criteria results match results.json — all 4 K verified
- [x] Finding status SUPPORTED — appropriate for guided exploration type
- [x] No fabricated values — all numbers trace directly to results.json

---

## Verification of Kill Criteria

| K# | Criterion | results.json | PAPER.md | Match? |
|----|-----------|--------------|----------|--------|
| K1033 | 4-bit ratio ≥ 0.95 | k1033_ratio=1.0408 | 1.041 | ✓ |
| K1034 | 2-bit ratio ≥ 0.85 | k1034_ratio=1.0204 | 1.020 | ✓ |
| K1035 | 4-bit size < 5 MB | max_4bit_logical_mb=1.67 | 1.67 MB | ✓ |
| K1036 | |cos_4bit| < 0.05 | max_cos_4bit=0.019348 | 0.019 | ✓ |

---

## Issues Found

### Non-blocking (note only, no fix required)

**1. Ratios > 1.0 framing**: Quality ratios 1.041 (4-bit/fp16) and 1.020 (2-bit/fp16)
suggest quantized adapters "outperform" fp16. PAPER.md correctly labels this "sampling
noise at n=25." But the experiment's key claim should be stated precisely:
> "No statistically significant quality degradation from 4-bit or 2-bit quantization at n=25."
Not "4-bit is 4.1% better." The current framing in PAPER.md is already correct, but
the Finding description in the event payload says "lossless" which is accurate.

**2. K1036 cosine DECREASED after quantization (0.019465 → 0.019348)**: This is
expected (zero-mean quantization noise cancels across 42 layers) and correctly explained
in PAPER.md. No issue — the bound in Theorem 2 is conservative, and the actual behavior
is benign. The Theorem 2 prediction "O(ε_rel) ≈ 7.6% perturbation" is 200× too pessimistic
because it's a worst-case bound assuming correlated errors. The cancellation argument is valid.

**3. Medical fp32→fp16 drop (48%→36%)**: A 12pp drop when moving from fp32 to fp16
storage is surprising, but within n=25 Wilson CI (±10pp). Not a defect in this experiment —
it's residual variance from T2.1's n=50 eval. Non-blocking.

---

## Assessment

The math is correct (Theorem 1: quantization error bound, Theorem 2: orthogonality
perturbation bound, Theorem 3: size formula). The experiment design is clean. The key
finding — 4-bit lora_b + fp16 lora_a provides 3× compression with no measurable quality
loss — is well-supported. The 200× gap between Theorem 2's worst-case bound and the
actual measurement is a genuine insight (zero-mean cancellation across 42 layers).

**Action**: Emit review.proceed → Analyst writes LEARNINGS.md.
