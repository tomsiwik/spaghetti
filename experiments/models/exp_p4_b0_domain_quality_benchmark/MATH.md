# MATH.md — P4.B0: Domain Adapter Quality Benchmark

## Problem

Experiments P4.A0–A2 verified routing accuracy (97–100%) and "new domain in <10 min" (Finding #475).
But the behavioral quality metric used was a vocabulary rubric (≥8 bio terms), which P4.A1 LEARNINGS
flagged as noisy — 3/20 apparent regressions were format shifts, not knowledge loss.

**Right question (SIGReg):** What is the true factual accuracy gain from domain adapters over the
already-strong Gemma 4 base? And does applying a domain adapter degrade other-domain performance?

## Theorem 1 (Domain Adapter Factual Gap)

**Setup:**
- Base model M with parameters θ, trained on general corpus D_gen
- Domain adapter A_d trained on domain corpus D_d (LoRA rank=6, q_proj, 1000 steps)
- Adapted model M_d = M ⊕ A_d (M with adapter A_d applied)
- Evaluation: binary keyword scoring f(r, K) = #{k ∈ K : k ∈ r} / |K|
  where K is the set of key facts for a question and r is the generated response

**Theorem:** For domain-specific questions Q_d, if D_d concentrates probability mass on domain
vocabulary V_d that is underrepresented in D_gen:

  E[f(M_d(q), K)] ≥ E[f(M(q), K)] + δ_d

where δ_d is the domain adaptation gap, lower bounded by:

  δ_d ≥ (1 - H(V_d | θ)) × coverage(V_d, A_d)

with H(V_d | θ) = normalized entropy of domain vocab under base model, and
coverage(V_d, A_d) = fraction of domain terms appearing in adapter training data.

**Proof sketch:**
The LoRA adapter A_d modifies q_proj attention weights, shifting token-level probability mass
toward domain-specific patterns. For domain questions where base model entropy H(V_d | θ) is high
(uncertain), the adapter provides a low-entropy channel that reliably produces domain vocabulary.
For questions where base already has low entropy (already knows the answer), δ_d ≈ 0.

**QED** (informal: adapter shifts uncertain outputs toward domain-specific outputs)

## Theorem 2 (Adapter Isolation — No Cross-Domain Regression)

**Setup:** Domain adapters A_1, ..., A_5 trained independently on orthogonal domains
(proven in Finding #228: max pairwise cosine = 2.25e-8 for Grassmannian separation)

**Theorem:** Applying adapter A_d (domain d) does not significantly degrade performance on domain d':

  E[f(M_{d}(q_{d'}), K_{d'})] ≥ η × E[f(M(q_{d'}), K_{d'})]

where η ≥ 0.90 (10% degradation budget) for orthogonal domain pairs.

**Proof sketch:**
Since A_d is trained on D_d with domain-specific patterns, for queries q_{d'} outside domain d,
the adapter weight ΔW_d projects to a subspace orthogonal to the features activated by q_{d'}.
The q_proj modification does not suppress domain d' features because they are in complementary
subspaces (Finding #228 Grassmannian separation). Therefore cross-domain interference is bounded.

**QED** (conditional on Grassmannian separation holding for factual queries)

## Quantitative Predictions

Based on T2.1 LEARNINGS (Finding #421) and P3.B5 domain-conditional retrain (Finding #466):

| Domain | Base HF Score (prior MCQ) | Expected δ_d (keyword) | Prediction: PASS/FAIL K1224 |
|--------|--------------------------|------------------------|------------------------------|
| Math   | 82% MCQ (with adapter)  | 15–35pp keyword        | PASS                         |
| Code   | +46pp MCQ               | 10–25pp keyword        | PASS                         |
| Medical| +22pp MCQ               | 5–15pp keyword         | MARGINAL (may fail)          |
| Legal  | unknown                 | 5–15pp keyword         | UNCERTAIN                    |
| Finance| unknown                 | 5–15pp keyword         | UNCERTAIN                    |
| Biology| +20pp vocab (P4.A1)     | 10–20pp keyword        | PASS                         |

**Aggregate prediction:** ≥3 of 5 domains exceed 10pp improvement (K1224 PASS)

**Cross-domain prediction (K1225):**
η ≥ 0.90 for all adapter × non-target-domain pairs (Theorem 2 predicts near-zero interference)

**Absolute accuracy prediction (K1226):**
Gemma 4 base is strong: E[f(M(q), K)] ≈ 40–60% for specific domain questions.
Adapted: E[f(M_d(q_d), K)] ≈ 55–75%. Average across all adapted ≥ 50%.

## Kill Criteria Derivation

K1224 derives from Theorem 1: the adapter must improve factual accuracy on domain questions.
If fewer than 3 domains improve by ≥10pp, the adapter vocabulary shift is insufficient for
factual recall — possible if base model already mastered the domain (δ_d ≈ 0).

K1225 derives from Theorem 2: cross-domain interference should be bounded by Grassmannian isolation.
If η < 0.90, the adapters are NOT truly isolated — suggests the rank-6 q_proj update is large
enough to corrupt cross-domain attention patterns.

K1226 is an absolute floor: adapted accuracy < 50% would suggest the questions are too hard or
the keyword rubric is miscalibrated.

## Connection to Architecture

Domain adapters in Pierre P1 serve as the "expert activation" in the Room Model:
W_combined = W_base + Σ α_d × ΔW_d

This experiment validates that individual ΔW_d terms provide useful factual signal when activated
alone (α_d = 1, all others = 0). If individual adapters don't improve quality, the composition
claim is weaker than we thought.
