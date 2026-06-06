# REVIEW-adversarial.md — P4.A0: 5-Domain TF-IDF Ridge Routing

**Verdict: PROCEED**
**Round:** 2 (blocking fix applied — PAPER.md written)

---

## Fix Applied (Round 2)

PAPER.md was written with complete prediction-vs-measurement table and explanations for both
prediction misses. The blocking issue is resolved.

---

## Non-Blocking Observations

**N1: Finance sample imbalance** — N_TEST for finance was 90, not 100 (dataset size limit).
This is fine but should be noted: 13 finance queries were absorbed into the 90 denominator,
making the weighted_accuracy denominator 490 not 500. Results remain valid.

**N2: Medical lowest precision (93.2%)** — Medical queries bleed into legal (1) and math (3)
from incoming confusions. The primary failure mode F1 (legal/finance overlap) predicted in
MATH.md was partially observed but as medical confusion, not legal/finance pair confusion.
Legal and finance are well-separated (F1=0.970, 0.972). The actual hard pair is math/medical
(3 math queries routed to medical) which Theorem 2 estimated at cos=0.123 — directionally
correct but underestimated.

**N3: Claim "≥ 98.8% predicted" is overconfident** — Finding #458 achieved 98.8% with N=25
*synthetic* MMLU categories. The claim in MATH.md that N=5 real data would be ≥ 98.8% was
plausible but optimistic. Real domain corpora contain more vocabulary diversity within domains
(medical includes oncology, cardiology, neurology), which increases within-class variance.
The kill threshold (95%) was the right conservative boundary; the point prediction (98%) was
aspirational.

---

## Math Quality

Theorems 1 and 2 are structurally correct. The classification error bound (Step 3) is a
standard result. Theorem 2's vocabulary divergence formula is sound in principle but the
parameter estimates (|V_shared|=200, |V_medical|=8000) were not verified empirically before
the run. The experimental results vindicate the framework even if the specific estimates were
off. This is acceptable for a verification experiment.

---

## Final Status

**SUPPORTED** — Finding #474 added.
- Title: "5-Domain TF-IDF Ridge Routing: 97.3% Weighted Accuracy, 0.247ms Latency"
- Status: supported
- Result: All 3 kill criteria pass. Router scales to 5 real domains with train_time=76ms.
  Math_vs_legal is the hardest pair (cos=0.237, precision=97%/97%). No adapter needed.
- Scale: micro
