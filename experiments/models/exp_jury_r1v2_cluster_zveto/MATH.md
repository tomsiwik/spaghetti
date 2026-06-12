# MATH — Jury R1v2: per-question z-scored verifier as cluster veto+reweight

## Disease (not symptom)
R1 (F#877) measured pooled verifier AUC 0.821 yet BoN top-1 (0.785) *lost* to self-consistency
(0.820). The disease is **difficulty-dependent score bias**: vscores are comparable *within* a
question but not *across* questions, and top-1 selection forces the verifier to act as a global
ranker — the one role miscalibration destroys. Negative evidence ("this candidate is clearly worse
than its siblings") survives miscalibration that destroys positive top-1 ranking.

## Theorem (prediction sketch)
Let each question q have 8 candidates with vscores v₁..v₈, μ_q, σ_q, and z_i = (v_i−μ_q)/σ_q
(GRPO-style group normalization: the candidate set is its own reference distribution).
Cluster candidates by final answer; for cluster c: vote(c)=|c|/8, maxz(c)=max_{i∈c} z_i.

Decision rule (alpha, τ tuned on the EVEN half only):
1. **Guess gate:** if σ_q < τ the verifier is uninformative → output the SC answer unchanged.
2. **Veto:** drop clusters with maxz(c) < −1 (unless all clusters are vetoed → restore all).
3. **Reweight:** answer = argmax_c [ alpha·vote(c) + (1−alpha)·maxz(c) ].

Claim: because AUC 0.821 means correct candidates carry higher *within-question relative* scores,
veto + reweight over the SC cluster structure converts the unconverted pass@8 headroom
(0.935 vs SC 0.820) into accuracy, while the SC fallback (gate, alpha→1 limit) bounds the downside.

## Predicted number
Held-out (ODD indices, 100 questions) accuracy of the tuned rule ≥ **held-out SC + 3pp**,
achieved with ZERO new tokens (pure reanalysis of cached R1 data), with the gain carried by
**> 5 win-flips** (questions SC got wrong that the jury gets right).

## Refutation threshold (pre-registered, per kill #2332)
KILLED if, with (alpha, τ) tuned on the even half:
- held-out accuracy < held-out SC + 3pp (the pre-registered gate; the ≥1pp clause is subsumed), OR
- win-flips ≤ 5 on the held-out half (binomial noise).

Controls reported on the same held-out half: SC (cached sc_ok), raw-vscore BoN (cached bon_ok),
z-only top-1 (argmax z, no clustering).

Data: `experiments/models/exp_bet_jury_r1_verifier_gain/results.json` (200 q × 8 cands;
pred/ok/vscore per candidate; sc/bon baselines per question). No model calls.
