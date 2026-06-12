# LEARNINGS — exp_bet_jury_r1_verifier_gain

**Core finding.** The math adapter re-prompted as a judge is a genuinely good pooled ranker
(AUC 0.821 vs 0.685 for likelihood), yet verifier best-of-8 LOSES to plain self-consistency
on GSM8K at equal generation budget (0.785 vs 0.820, −3.5pp; gate K2316 → KILLED).

**Why.** Pooled AUC is the wrong sufficient statistic for BoN: selection needs per-question
top-1 correctness, and the verifier's scores are not calibrated across questions — it misranks
within exactly the questions majority vote already gets right. With pass@8 = 0.935 there was
~11.5pp of headroom above SC; the verifier converted none and paid 447k prefill tokens of rent.
(Caveat: −3.5pp is not individually significant, sign test p≈0.19, but the +3–6pp prediction
is decisively refuted.)

**Implication for the next experiment.** Raw logP(Yes)−logP(No) argmax is dead as a selector;
any jury-decode v2 must fix per-question calibration first (e.g., within-question rank
aggregation or verifier-weighted voting that degrades gracefully to SC) — not add more jurors,
since R2's decorrelation premise sits on top of a selector that loses to majority vote. The
salvage is real: the judge signal exists (AUC 0.821 ≫ likelihood), the aggregation is broken.

**PIERRE-IMPACT:** shelved — R1 killed (K2316); no decode-time verifier lands on
bet/jury-decode; bet ladder blocked at R1 until a calibration-aware selector beats SC.
