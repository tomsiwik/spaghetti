# Adversarial Review: exp_p0_discriminative_diagnosis

**Verdict: PROCEED**

## Summary

Clean A/B experiment isolating compression vs training objective as the cause of
MedMCQA discriminative collapse. Core finding ("compression is the disease") is
strongly supported by a 34pp gap between Standard LoRA (52.5%) and TT-LoRA (18.5%).
All REVISE fixes from round 1 addressed: crash fixed, base discrepancy resolved,
PAPER.md written with prediction-vs-measurement table.

## Checklist

| Check | Result |
|---|---|
| Prediction-vs-measurement table | YES — 4 rows, 2/3 in range, 1 above upper bound |
| Kill criteria match evidence | YES — results.json confirms K1430 PASS, K1431 FAIL, K1432 FAIL |
| Finding status appropriate | YES — SUPPORTED (Theorem 2 confirmed, Theorem 1 partially wrong) |
| Math errors | Theorem 1 partially refuted (see below) — honestly reported |
| Data integrity | results.json matches all PAPER.md claims exactly |

## Strengths

1. **Honest prediction failure.** Theorem 1 predicted NTP ⊥ discrimination, but LoRA+NTP
   improved MCQ by +22pp. The paper acknowledges this openly and provides a mechanistic
   explanation (NTP teaches medical knowledge that transfers to MCQ).

2. **Training loss paradox is compelling.** TT-LoRA loss 0.169 < LoRA loss 0.179, yet
   MCQ accuracy 18.5% << 52.5%. This directly demonstrates that NTP loss is a poor proxy
   for discriminative capacity — a concrete instance of the project's "metrics ≠ behavior"
   principle.

3. **Clean experimental design.** Same data, same hyperparameters, same eval set (200
   questions, seed=42). Only the adapter architecture differs.

## Non-Blocking Notes

1. **Prediction miss for LoRA.** Predicted 35-45%, measured 52.5%. The paper calls this
   "conservative" but doesn't explain why the upper bound was 7.5pp off. Likely because
   Finding #508 used 1000 steps (50% MCQ) and the prediction assumed linear interpolation
   to 500 steps, but learning curves are not linear. Minor — doesn't affect the diagnosis.

2. **No confidence intervals.** 200-question eval has ~±3.5pp standard error for binomial
   proportions at 50%. The 34pp compression gap far exceeds this, so the finding is robust,
   but future experiments should report error bars.

3. **TT-LoRA training time anomaly.** TT-LoRA (422s, 135K params) took 1.6x longer than
   LoRA (263s, 2.7M params). The TT forward pass has overhead from core contractions.
   Not a concern for the finding, but worth noting for future benchmarking.

4. **Theorem 1 needs formal correction.** The post-hoc explanation ("NTP teaches knowledge
   that enables MCQ") is intuitive but not formalized. A future experiment could make this
   rigorous by measuring knowledge transfer specifically (e.g., probing intermediate
   representations for medical concept encoding).

## Verdict: PROCEED

The core finding is well-supported. Compression (20x, rank-6) discards discriminative
features that standard LoRA (rank-8) preserves. Status SUPPORTED is appropriate: Theorem 2
confirmed, Theorem 1 partially refuted but honestly reported.
