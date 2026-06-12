# LEARNINGS — exp_jury_r1v2_cluster_zveto

**Core finding.** Within-question z-normalization (GRPO-style) of the R1 verifier scores repairs
the raw-BoN pathology — jury 0.84 vs raw-BoN 0.80 on held-out GSM8K — but only back to
self-consistency parity (SC 0.83, +1pp, 3 win-flips vs 2 loss-flips); gate K2332 → KILLED.

**Why.** The calibration fix was real but the underlying signal is exhausted: after z-norm the
verifier adds at most ~1 correct question per 100 over pure voting (alpha grid plateaus at 0.84;
the tune-half +4pp was grid overfitting). The verifier never converts the pass@8 = 0.935 headroom;
it only stops hurting. n=100 held-out puts +1pp squarely inside binomial noise.

**Implication for the next experiment.** Single-adapter verifier BoN reranking on GSM8K is a dead
class: raw scores lose to SC (K2316), calibrated scores tie SC (K2332). This is the bet's second
consecutive dead rung on R1, and the v2 *was* the proposed fix — there is no v3 for the selector.
The bet survives only if R2 decorrelation (multiple adapter-verifiers with provably non-overlapping
errors) is reframed as a first rung with its own mechanism check; otherwise this is jury-decode's
obituary.

**PIERRE-IMPACT:** shelved — killed (K2332); no decode-time jury lands on bet/jury-decode; the
single-verifier selector class is closed, and the branch stays at solo-adapter greedy baseline.
