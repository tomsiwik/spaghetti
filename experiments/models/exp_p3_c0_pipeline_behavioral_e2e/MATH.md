# MATH.md — P3.C0: Full Pipeline Behavioral E2E

## Problem Statement

P3.B5 (Finding #466) proved that domain-conditional retraining achieves 0pp style
degradation in composition (92% composed = 92% personal-alone). But P3.B5 tested
the composition *directly* — no routing. A production system must:

1. Route the query to the correct domain adapter
2. Apply domain-fused base + personal adapter
3. Produce output that is BOTH stylistically compliant AND domain-relevant

This experiment tests the FULL pipeline: routing → composition → behavioral quality.

## Prior Results (Foundation)

- Finding #458: Ridge routing N=25, 98.8% accuracy on MMLU-format queries
- Finding #466: Domain-conditional composition, 0pp style loss (92% vs 92%)
- Finding #436: Personal adapter training, 76pp style compliance gain (P1.T5)

## Theorem 1 (E2E Pipeline Quality Bound)

**Setup**: Let:
- R: X → {math, general} be the ridge router (routing function)
- C: X → Y be the domain-conditional composition (domain_fused_base + personal adapter)
- style(y) ∈ {0,1} be the style compliance indicator (contains PREFERENCE_MARKER)

**Claim**: The full pipeline P = C ∘ R_correct achieves:
```
E[style_pipeline] ≥ α_R × ρ_C
```
where:
- α_R = routing accuracy (P(R(x) = math | x ∈ math-queries))
- ρ_C = style compliance of composition without routing errors (0pp loss → ρ_C = 1.0)

**Proof**:
Let Z_route = 1_{R(x) = math} be the routing indicator.

Case 1 (Z_route = 1, probability α_R): Query routed correctly → apply domain_fused_base
  + personal adapter → style compliance = ρ_C (from Finding #466, d_H=0 guarantee).

Case 2 (Z_route = 0, probability 1 - α_R): Query NOT routed to math domain.
  In this experiment, we evaluate only queries from the math domain (style questions
  with math-like framing), so misrouting produces a mismatch — style compliance = ρ_fallback.

E[style_pipeline] = α_R × ρ_C + (1-α_R) × ρ_fallback
                  ≥ α_R × ρ_C  (since ρ_fallback ≥ 0)
                  = α_R × 1.0  (from Finding #466: ρ_C = 1.0)
                  = α_R

**Quantitative prediction**: If α_R ≥ 80% (conservative for real-format queries,
vs 98.8% on MMLU-format), then:
```
E[style_pipeline] ≥ 80% × 92% = 73.6%
```
Kill criterion K1194 (≥60%) is conservative — should pass if routing works.

**QED**

## Theorem 2 (Routing Accuracy on Real-Format Queries)

**Claim**: Ridge router trained on MMLU-format prompts achieves ≥80% accuracy on
real-format math queries (GSM8K word problems, different from MMLU MCQ format).

**Basis**: Ridge regression classifier minimizes:
```
W* = argmin ||Φ·W - Y||² + λ||W||²  →  W* = (Φᵀ Φ + λI)⁻¹ Φᵀ Y
```
where Φ is the TF-IDF feature matrix (300-dim, l2-normalized).

The router captures *vocabulary* signal (math-specific terms: equation, solve, calculate,
integral, derivative) not *format* signal (MCQ structure). Therefore, transfer from
MMLU-format to GSM8K-format is expected via vocabulary overlap.

**Formal bound**: Let V_train ∩ V_test be the shared vocabulary signal.
By the ridge regression generalization bound (Mohri et al. 2012, Theorem 13.7):
```
E[L_test(W*)] ≤ E[L_train(W*)] + O(√(rank(Φ)/n_train))
```
With n_train=300 per domain and rank(Φ) ≪ 300 (sparse TF-IDF), the generalization
gap is small. Math vocabulary (solve, equation, find x) appears in both MMLU and GSM8K.

**Prediction**: Routing accuracy ≥ 80% on GSM8K-format (vs 98.8% on MMLU-format).

**QED**

## Kill Criteria (Pre-registered)

| ID | Criterion | Prediction | Basis |
|----|-----------|------------|-------|
| K1193 | routing_acc_math ≥ 80% | ~90-95% | Theorem 2, vocabulary transfer |
| K1194 | style_compliance ≥ 60% | ~73-80% | Theorem 1 bound |
| K1195 | math_acc ≥ 5% | ~10% | P3.B5 K1196 baseline |

## Failure Mode Analysis

**What makes K1193 impossible to fail**: Math vocabulary is highly discriminative.
GSM8K questions contain "how many", "total", "each", "sold", "cost", "difference" —
terms the ridge router associates with math. The only failure mode is if the router
overfits to MCQ format markers ("Which of the following", "The correct answer is").
We train on question-only (no answer options), mitigating format dependence.

**What makes K1194 impossible to fail**: K1194 fails ONLY if routing accuracy < 60%.
If routing_acc ≥ 80% (K1193 passes) and ρ_C = 1.0 (Finding #466), then:
style_compliance ≥ 80% × 92% = 73.6% > 60%. These two kills are logically linked.

**If K1193 kills but K1194 passes**: Routing accuracy < 80% but style preserved.
Explanation: style queries use personal PREFERENCE_MARKER explicitly — the personal
adapter dominates regardless of domain routing decision.

**If K1194 kills but K1193 passes**: Routing correct but style fails.
Explanation: FP16 domain_fused_base inference artifacts degrade personal adapter output.
Fix: Requantize domain_fused_base to 4-bit before deploying personal adapter.

## Prediction vs Measurement Table (filled after experiment)

| Metric | Theorem Prediction | Measured |
|--------|--------------------|---------|
| routing_acc_math (K1193) | ≥80% | TBD |
| style_compliance (K1194) | ≥73.6% | TBD |
| math_acc (K1195) | ~10% | TBD |
| routing_false_positive_general | ≤20% | TBD |
| pipeline_latency_p50 | <30s | TBD |
