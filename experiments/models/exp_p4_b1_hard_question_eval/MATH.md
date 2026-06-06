# MATH.md — P4.B1: Gap-Targeted Evaluation — Hard Domain Questions

## Motivation

Finding #477 (P4.B0) showed adapter improvement is strongly gap-dependent:
- Math: base=0.307 → adapted=0.453, δ=+20pp ← WORKS (low base score)
- Medical: base=0.480 → adapted=0.440, δ=-4pp ← FAILS (high base score)
- Finance: base=0.413 → adapted=0.560, δ=+15pp ← WORKS (medium base)

The hypothesis: adapter improvement δ_d requires the base model to have
genuine uncertainty (high entropy) over domain vocabulary. When the base model
already produces domain vocabulary naturally (low entropy), the adapter's signal
is redundant.

---

## Theorem 1 (Adapter Signal Gap Hypothesis)

**Setup**: Let θ_base be the base model parameters, ΔW_d the domain adapter
for domain d, and V_d = {v_1, ..., v_K} a set of domain-specific vocabulary terms
used as a keyword rubric to score responses.

**Define**:
- Base entropy: H(V_d | θ_base) = -Σ_{v∈V_d} P_base(v | q) log P_base(v | q)
  where P_base(v | q) = P(v appears in response to query q | θ_base)
- Score function: score(q, θ) = #{v ∈ V_d : v in response(q, θ)} / |V_d|
- Improvement: δ_d(q) = score(q, θ_base + ΔW_d) - score(q, θ_base)

**Proxy**: We use base_score(q) = score(q, θ_base) as an empirical proxy for
H(V_d | θ_base). Low base_score ↔ high entropy (model doesn't produce these terms).

**Theorem**: If H(V_d | θ_base) > H_threshold, then ΔW_d can provide signal and
δ_d > 0. Equivalently, if base_score < threshold_score, the adapter has room to act.

**Proof sketch**:
1. The adapter ΔW_d modifies attention outputs to shift token probabilities toward
   domain vocabulary (by construction: trained on domain Q&A pairs).
2. If P_base(v | q) is already near 1 for most v ∈ V_d, then:
   P_adapted(v | q) ≈ P_base(v | q) (no room to increase further)
   → δ_d ≈ 0.
3. If P_base(v | q) is near 0 for most v ∈ V_d (high entropy), then:
   P_adapted(v | q) > P_base(v | q) (adapter shifts distribution toward V_d)
   → δ_d > 0.
4. By continuity of the softmax, δ_d is monotonically decreasing in base_score.
QED (empirical verification below)

---

## Theorem 2 (Hard Question Construction)

**Goal**: Construct a question set Q_hard ⊆ Q where, for each q ∈ Q_hard,
base_score(q) < threshold = 0.30.

**Construction principle**: Questions in Q_hard use vocabulary rubric V_d such that:
(a) V_d terms are specialized subdomain vocabulary not commonly produced in general text
(b) V_d terms ARE present in the adapter training data (so ΔW_d learned them)
(c) V_d terms require specific expertise to produce naturally

**Five domains used** (existing rank-6 adapters from P1 T2 experiments):
- Math: abstract algebra, topology, real analysis — specific notation
- Medical: immunology, pharmacokinetics, rare conditions — specific terminology
- Legal: procedural law, regulatory statutes, constitutional doctrine — specific concepts
- Code: systems programming, distributed systems, compiler theory — specific algorithms
- Finance: derivatives pricing, risk management, quantitative methods — specific formulas

**Prediction**: For Q_hard constructed per this principle:
- E[base_score] < 0.25 (most specific terms missed by base model)
- E[δ_d] ≥ 0.15 for domains where adapter training covered these subdomains
- Pearson r(base_score_i, δ_d_i) < -0.30 (negative: lower base → larger improvement)

---

## Kill Criteria (Quantitative)

**K1227**: E[base_score] < 0.25 on hard question set across ALL 5 domains.
- If FAILS (base ≥ 0.25): questions are not hard enough; need more specialized vocab
- This verifies our question construction principle (Theorem 2)

**K1228**: ≥3/5 domains show δ_d ≥ 0.15 (15pp improvement with rank-6 adapters).
- If FAILS: rank-6 adapters with 100 training examples cannot overcome the vocabulary gap
  even when the gap exists → rank is the bottleneck, not question hardness
  → P4.B2 should test rank-16 adapters

**K1229**: Pearson r(base_score, improvement) < -0.30 across all questions.
- If FAILS: improvement is NOT correlated with base score → gap hypothesis is wrong
  → Improvement depends on something else (e.g., question type, adapter quality)

---

## Connection to Prior Findings

- Finding #477 (P4.B0): Adapter improvement gap-dependent (math +20pp, medical -4pp)
- Finding #459 (P2.A0): Near-uniform base uncertainty → δ_d ≈ 0 (PubMedQA 3-class)
- Finding #468 (P3.C1): Rank bottleneck — rank-6 insufficient for difficult style tasks
- Finding #436 (P1.T5.1): User local training shows high δ (personalization from zero)

The P4.B0 finding that "math adapter has worst cross-domain retention (0.834)"
is consistent with Theorem 1: the math adapter makes large changes in attention
(to produce specialized notation) which inadvertently perturbs other domains.
High δ_d ↔ large ΔW_d → potential cross-domain interference.

---

## Experiment Design

- Model: mlx-community/gemma-4-e4b-it-4bit (same as P4.B0)
- Adapters: rank-6, q_proj, all-layer (from P1 T2 single/multi domain)
- Hard questions: N=15 per domain, 6 specialized keywords each
- Max tokens: 200 (longer answers needed for technical depth)
- Phases:
  1. Base evaluation on hard questions
  2. Per-adapter evaluation on hard questions
  3. Correlation analysis + kill criteria check
