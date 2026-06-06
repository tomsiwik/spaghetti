# LEARNINGS — exp_followup_answer_conditioned_ppl

## Core Finding
K1567 is refuted. Both answer-only PPL (0.978) and full-seq PPL (0.984) route
5-domain mixed queries to their correct expert at >97% top-1 — far above the
0.85 threshold. The "full-seq fails" clause of the conjunction does not hold
on disjoint-alphabet synthetic domains. Answer-only PPL offers no marginal
routing benefit in this regime.

## Why
Two distinct quantities were conflated in the prediction:
- Predecessor: r_full = −0.31 measures **relative change** (ΔPPL_full vs
  Δaccuracy, expert-minus-base per domain). Own-domain prompt-PPL sometimes
  degrades (reverse, sort).
- This experiment: argmin_j PPL_full(θ_j, q) measures **absolute cross-expert
  ranking**. Expert-k (k≠j) has never seen domain-j data, so its cross-domain
  prompt-PPL penalty dwarfs expert-j's mild own-domain prompt degradation.

The two are independent: relative-change correlation near zero/negative is
compatible with near-perfect absolute ranking, because different experts score
different domains under wildly different token distributions. On disjoint
alphabets, any metric monotone in cross-entropy routes trivially.

## Implications for Next Experiment
1. **The answer-vs-full routing question is vacuous on disjoint alphabets.**
   A v2 (`*_shared_alphabet`) must use prose-style domains with overlapping
   subword vocabulary and T_p/T_a ≥ 3, making full-seq PPL dominated by
   prompt modeling. Only then is K1567 falsifiable at the distinguishing
   regime.
2. **Stop using relative-change correlations to predict absolute-ranking
   routing.** They answer different questions; the KC prediction chain
   `r_full = −0.31 ⇒ Top1_full < 0.85` was invalid a priori. Future routing
   experiments should predict top-1 directly from cross-domain token-distribution
   mismatch, not from ΔPPL-vs-Δaccuracy slopes.
3. **Routing in the Pierre P1 context** (Gemma 4, prose/code/math): shared
   tokenizer, realistic T_p/T_a, so the distinction may matter. But don't
   rebuild a synthetic proxy — test answer-only PPL routing directly on the
   Gemma 4 adapter bank the polar-adapter stack will ship, skipping the
   microverse.
4. **sort↔reverse confusion (5-6.5%)** is the only signal in this data:
   shared alphabet + shared `>` delimiter. That pair is the only micro-regime
   that would be worth scaling up if one wanted to study routing edge cases
   in this synthetic setup.

## No antipattern captured
REVIEW-adversarial confirmed all process checks pass; pre-registration was
honored, KC was not relaxed, and the prediction-gap reconciliation is math-sound
(not post-hoc). No process bug to store as a memory.
