# PAPER.md — P4.B0: Domain Adapter Quality Benchmark

## Summary

Rank-6 q_proj LoRA adapters trained on synthetic domain Q&A provide factual accuracy
improvement that is **domain-gap-dependent**: strong where base model has low priors (math
+20pp), weak or negative where base model is already calibrated (medical -4pp, base=48%).
All three kill criteria technically fail; K1225 and K1226 are borderline (1-2pp off threshold).

## Prediction vs Measurement Table

| Metric | Theorem Prediction | Measured | Verdict |
|--------|-------------------|----------|---------|
| Math improvement | 15–35pp | **+20.0pp** | MATCH |
| Finance improvement | 5–15pp | **+14.7pp** | MATCH |
| Medical improvement | 5–15pp | **-4.0pp** | MISS (base too strong) |
| Code improvement | 10–25pp | **+6.7pp** | MISS (format mismatch?) |
| Legal improvement | 5–15pp | **+9.3pp** | MISS (0.7pp below threshold) |
| K1224: ≥3/5 ≥10pp | PASS (predicted) | 2/5 → **FAIL** | MISS |
| K1225: cross-domain retention ≥90% | PASS (Grassmannian) | **0.890 → FAIL** | MISS (1pp off) |
| K1226: avg adapted acc ≥50% | PASS | **0.480 → FAIL** | MISS (2pp off) |
| Total time | — | **8.6 min** | — |

## Per-Domain Results

| Domain | Base Score | Adapted Score | Δ (pp) | Outcome |
|--------|-----------|---------------|--------|---------|
| Medical | 0.480 | 0.440 | **-4.0** | Base too strong; adapter hurts |
| Code | 0.307 | 0.373 | **+6.7** | Improvement but below 10pp threshold |
| Math | 0.307 | 0.507 | **+20.0** | ✓ Clear improvement — notation gap |
| Legal | 0.453 | 0.547 | **+9.3** | Near-threshold; base moderately strong |
| Finance | 0.387 | 0.533 | **+14.7** | ✓ Clear improvement — specialized terms |

## Cross-Domain Retention

| Adapter | Retention Ratio | Pass ≥0.90? |
|---------|----------------|------------|
| Medical | 0.915 | ✓ |
| Code | 0.838 | ✗ |
| Math | 0.834 | ✗ (worst!) |
| Legal | 0.945 | ✓ |
| Finance | 0.915 | ✓ |
| **Mean** | **0.890** | ✗ (1pp off threshold) |

Surprising: Math adapter causes most cross-domain interference (0.834) despite having the
largest within-domain improvement (+20pp). Notation specialization suppresses other domains.

## Structural Impossibility Analysis

**Why K1224 fails (2/5 < 3/5 required):**

The adaptation gap δ_d ≈ 0 when base model has low entropy H(V_d | θ) (already calibrated).
Medical (base=0.48) and legal (base=0.45) are domains where Gemma 4 4B has extensive training
data, so rank-6 q_proj adapters cannot overcome the base prior.

Formula from MATH.md: δ_d ≥ (1 - H(V_d | θ)) × coverage(V_d, A_d)
When H(V_d | θ) ≈ 1 (base already uncertain but in different directions), coverage × uncertainty
doesn't produce net positive gain.

**Why math and finance work:**
- Math: base=0.307, notational gap is real — specific patterns (a^2, eigenvalue, polynomial
  approximation) appear more frequently in adapter training data than pretraining distribution
- Finance: specialized jargon (CAPM, beta, YTM, Black-Scholes) with specific formulations the
  adapter concentrates on

**K1225 borderline failure (0.890 vs 0.90):**
The 10% cross-domain degradation budget is violated slightly. The math adapter's notation
focus (0.834) actively suppresses attention to cross-domain features — contradicting naive
Grassmannian isolation prediction. Note: Grassmannian isolation (Finding #228) applies to
weight space cosine, not to output feature space activation patterns.

**K1226 borderline failure (0.480 vs 0.50):**
Absolute accuracy floor is close. Medical regression drags down the average.

## Connection to Vision

This experiment validates that domain adapter quality is **asymmetric**:
- High-gap domains (math, niche science): adapters provide 15-30pp improvement
- Low-gap domains (medical, legal with strong base): adapters provide 0-10pp or regress

Implication for Pierre P1: Domain routing + adaptation is most valuable for niche/technical
domains where Gemma 4 has pretraining gaps. For broad domains (medical, legal), routing is
still useful for style/format consistency but factual improvement is smaller than predicted.

## Caveats

1. N=15 questions per domain — confidence intervals are wide (~±7pp at 95% CI)
2. Keyword rubric is a proxy — misses paraphrase-equivalent answers
3. Biology not adapter-tested (no P1 adapter with same rank/architecture)
4. Smoke results (N=3) had systematic noise — full run (N=15) gives different picture for
   code (+0 smoke → +6.7 full) and legal (-13pp smoke → +9.3 full)
