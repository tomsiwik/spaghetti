# SHINE S4: Document QA Without Context — Mathematical Framework

## Grounding

- arXiv:2602.06358 (SHINE) — 63.6% F1 on SQuAD with 1.3B hypernetwork
- Finding #484 — S2: CE ratio 0.134 (86.6% reduction), centroid trap cos=0.998
- S3 PAPER.md — Multi-projection (q+v+o) 7.7x over q-only, meta LoRA killed
- Finding #480 — v_proj + o_proj unlock format priors

## The Question

S2 achieves 86.6% CE reduction. S3 shows q+v+o gives 7.7x further improvement.
But the centroid trap persists (cos=0.988). **Is the CE reduction genuine
document encoding, or a universal language model improvement?**

QA is the behavioral test: if generated LoRA encodes document-specific knowledge,
the model should answer extractive questions WITHOUT the document in prompt.

## Theorem 1: Information Lower Bound for Extractive QA

**Setup.** Let D be a document, Q a question about D, A the answer span.
Let θ_base be the frozen base model, ΔW = M2P(mem(D)) the generated LoRA.

**Claim.** For extractive QA with F1 > 0, the generated LoRA must satisfy:

    I(ΔW; D) > 0

where I is mutual information. If all documents produce the same ΔW
(centroid trap: cos(ΔW_i, ΔW_j) → 1), then I(ΔW; D) → 0 and QA
degenerates to the base model's prior.

**Proof.**
Let p(A|Q, θ_base + ΔW) be the answer distribution with LoRA applied.
For the model to answer correctly about D, it needs:

    p(A_correct | Q, θ_base + ΔW(D)) > p(A_correct | Q, θ_base)

If ΔW(D_1) = ΔW(D_2) for all D_1, D_2 (centroid), then:

    p(A | Q, θ_base + ΔW(D)) = p(A | Q, θ_base + ΔW_centroid)

This is a fixed model independent of D. It can only answer questions the
base model already knows or that ΔW_centroid biases toward. For document-
specific facts NOT in the base model's training data, F1 → 0.  ∎

## Theorem 2: CE Ratio vs QA — Orthogonal Metrics

**Claim.** CE ratio < 1 does NOT imply QA F1 > 0.

**Proof.** CE measures next-token prediction on the SAME sequence used for
memory extraction. A LoRA that shifts the model's distribution toward "more
probable English text" will reduce CE on ANY passage without encoding
passage-specific facts.

Formally: let ΔW_centroid minimize E_D[CE(D | θ_base + ΔW)]. This is
the maximum-likelihood LoRA over the training distribution. It reduces CE
on average but encodes no document-specific information.

The centroid trap IS this minimum: the optimization converges to the
single ΔW that best predicts "typical English" rather than each specific
document. This is why cos=0.998 despite CE ratio=0.134.  ∎

## Predictions

Given centroid cos=0.988 (S3) and the theorems above:

| ID | Prediction | Reasoning |
|----|-----------|-----------|
| P1 | QA F1 < 10% without document | Centroid LoRA ≈ universal LM improvement, not document-specific |
| P2 | QA F1 ≈ ICL F1 for base knowledge questions | Base model knows these facts regardless of LoRA |
| P3 | QA F1 << ICL F1 for novel facts | Novel facts require I(ΔW;D) > 0, which centroid blocks |
| P4 | Adapter generation < 5s | M2P is 8.2M params, one forward pass, should be fast |
| P5 | CE ratio with q+v+o < 0.20 | S3 measured 0.151 with meta LoRA overhead; without it, should be ≤ 0.15 |

## Kill Criteria Mapping

| Kill | Prediction | Expected |
|------|-----------|----------|
| K1261 (F1 > 30%) | P1: F1 < 10% | **EXPECTED FAIL** |
| K1262 (F1 ≥ 50% of ICL) | P3: F1 << ICL | **EXPECTED FAIL** |
| K1263 (gen < 5s) | P4: fast gen | EXPECTED PASS |

## What Each Outcome Means

**If K1261+K1262 FAIL (predicted):** Confirms centroid trap kills behavioral
utility. CE ratio is misleading. Next step: solve centroid trap with more
diverse training data (1000+ passages) or contrastive loss before attempting
QA again.

**If K1261+K1262 PASS (surprising):** The residual ε in cos=0.988 encodes
enough document-specific information for extractive QA. This would mean the
centroid trap is less severe than expected — the 1.2% angular difference
between LoRA vectors IS informationally sufficient. This would be a major
positive finding.

## Experiment Design

**Architecture: S4 = S2 + multi-projection (no meta LoRA)**
1. Pre-extract memory states (S2 approach, no trainable meta LoRA)
2. M2P with q+v+o output projections (validated 7.7x in S3)
3. Train on reconstruction loss (1000 steps)
4. Evaluate on QA task

**QA Protocol:**
1. 10 documents with 3 questions each (30 QA pairs)
2. Questions range from factual recall to inference
3. Three conditions:
   - **No-adapter**: question only (base model baseline)
   - **SHINE**: question only + generated LoRA from document
   - **ICL**: document + question in prompt (upper bound)
4. F1 = token-level overlap between generated answer and gold answer

**Generation:** Greedy decode, max 50 tokens, stop at newline.
