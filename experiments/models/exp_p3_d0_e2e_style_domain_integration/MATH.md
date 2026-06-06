# MATH.md — P3.D0: E2E Style + Domain Routing Integration

## Experiment Type: Verification

Verify that the Coverage Lemma (rank ≥ n_categories → full style coverage) holds
through the full production pipeline, not just in isolation.

---

## Problem

P3.C5 (Finding #472): rank-16 personal adapter achieves **93.3%** style compliance in isolation.
P3.C0 (Finding #467): full pipeline with rank-4 adapter achieves **60%** style compliance.

Question: does rank-16 adapter maintain ≥80% compliance through the full pipeline
(ridge routing → domain-conditional composition → personal style layer)?

---

## Theorem 1: E2E Style Preservation Under Routing

**Theorem:** Let α_R be the fraction of style queries correctly passed to the personal
adapter (routing pass-through rate). Let ρ_C be the style compliance of the personal
adapter in isolation. Then E2E style compliance satisfies:

    E[style_e2e] ≥ α_R × ρ_C

**Proof:**
Style compliance requires two conditions to hold simultaneously:
1. The router correctly identifies the query as a style (non-math) query (probability α_R).
2. Given the correct routing, the personal adapter applies the learned style (probability ρ_C).

Since (2) is conditionally independent of (1) given correct routing:
    P(style_applied) = P(routed correctly) × P(style_applied | routed correctly)
                     = α_R × ρ_C

E2E style compliance ≥ P(style_applied) by definition (styling is applied iff both hold). □

**Quantitative Predictions:**
- P3.C0 measured α_R = 1.0 (100% routing accuracy, 0 false positives)
- P3.C5 measured ρ_C = 0.933 (93.3% in isolation)
- Therefore: E[style_e2e] ≥ 1.0 × 0.933 = **93.3%** (theoretical floor)
- Conservative prediction allowing for distribution shift: **80–93%**

---

## Theorem 2: Composition Degradation Bound

**Theorem:** If the personal adapter is trained on the domain-fused base (P3.B5
architecture), then E2E composition degradation is 0pp — i.e., style compliance
through composition equals isolation compliance.

**Proof (from P3.B5, Finding #466):**
Let ΔW_P be the personal adapter trained on the domain-fused base H_D = H_0 + ΔH_D.
At inference, the input to the personal adapter is exactly H_D (since domain adapter is
already fused into the base weights). The training distribution P_train = P_infer (both
use H_D), so the Hellinger distance d_H(P_train, P_infer) = 0.

P3.B5 verified: personal_alone = composed = 92% (0pp degradation).
Therefore composition degradation ≤ 0pp for domain-conditional adapters. □

**Prediction:** Style compliance through full pipeline = style compliance in isolation
(within measurement noise ±5pp from question sampling variation).

---

## Failure Mode Analysis

**If K_D0_1 (style ≥ 80%) is KILLED:**
- Either α_R < 0.86 (router routes style queries to math → style adapter not applied)
- Or ρ_C degrades through the pipeline (composition breaks coverage)
- Distinguish: measure routing false-positive rate separately (K_D0_2)

**If K_D0_2 (routing_fp ≤ 10%) is KILLED:**
- Ridge router degrades on the rank-16 test set (different question vocabulary)
- Fix: re-train router on the same diverse 10-category vocabulary
- Impossibility: router failure is geometrically impossible in TF-IDF space when
  math vocabulary ("solve", "calculate", "total") is linearly separable from style
  vocabulary ("great question", "hope that helps") — confirmed by P3.C0

**If K_D0_3 (degradation ≤ 13.3pp) is KILLED:**
- Composition corrupts personal adapter directions
- Investigate: are rank-16 ΔW directions still orthogonal to domain ΔW_D?
- Check: P3.B5 covariate shift theorem assumes same base architecture

---

## Kill Criteria and Predictions

| Kill | Condition | Prediction | Basis |
|------|-----------|------------|-------|
| K1211 (K_D0_1) | E2E style ≥ 80% | 80–93% | Theorem 1: α_R=1.0 × ρ_C=0.933 |
| K1212 (K_D0_2) | routing_fp ≤ 10% | ~0% (same router as P3.C0) | P3.C0 Finding #467 |
| K1213 (K_D0_3) | degradation ≤ 13.3pp | ~0pp (Theorem 2, P3.B5) | P3.B5 Finding #466 |

---

## Method

1. **Reuse artifacts**: domain_fused_base (P3.B5), rank-16 personal adapter (P3.C5)
2. **Build ridge router**: same TF-IDF + ridge classifier as P3.C0/C5
3. **E2E eval**: route 15 diverse style questions → compose → generate → check style compliance
4. **Routing eval**: 20 math + 20 general → check accuracy and false-positive rate
5. **No new training**: all artifacts already exist from P3.B5 and P3.C5

---

## Citations

- Finding #472 (P3.C5): rank-16 adapter, 93.3% isolation style compliance
- Finding #467 (P3.C0): full pipeline with rank-4, 60% style, 100% routing accuracy
- Finding #466 (P3.B5): domain-conditional composition, 0pp degradation
- Finding #458 (ridge routing N=25): 98.8% accuracy
- arxiv 2106.09685 (LoRA): rank-r adaptation, rank-nullity theorem grounds Coverage Lemma
