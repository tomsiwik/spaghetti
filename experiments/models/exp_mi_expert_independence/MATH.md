# exp_mi_expert_independence — MATH

## Status
PREEMPTIVELY KILLED (derivation-only). No model run.

## Hypothesis (DB)
Mutual information MI(expert_i outputs, expert_j outputs) on calibration data
predicts composition quality strictly better than cosine similarity:
- K#418: r² improvement of MI over cosine ≥ 0.1 (FAIL if <0.1)
- K#419: MI computation cost ≤ 100× cosine cost (FAIL if >100×)

## Theorem (preempt-kill)
Both kill criteria are mathematically determined to FAIL given:

  (P1) F#286/§Spectral-arc corollary: PPL ↔ behavioral quality has Pearson
       r = 0.08 in this codebase (n=6 spectral experiments, m=5 domains).
  (P2) F#22 (killed) + F#544 (killed): KL(composed‖base) — a distributional
       distance metric — anti-correlates with behavioral quality at
       Spearman ρ = −0.7 at N=5 macro.
  (P3) F#425 (killed): activation/weight cosine fails to predict behavioral
       interference (N=5: 82%→8% math/code despite low cos).
  (P4) MI(X;Y) is itself a functional of the joint distribution p(X,Y),
       i.e. the same class of object as KL(p(X,Y)‖p(X)p(Y))
       — by definition MI(X;Y) = KL(p(X,Y) ‖ p(X)p(Y)).

## Proof of K#418 FAIL
Let Q ≡ behavioral composition quality. Let M ∈ {cosine, MI} be a
distributional/dependency metric on expert outputs. Define:

    r²(M, Q) := squared Pearson correlation between M(adapter pair) and Q.

By (P1), Q has Pearson r = 0.08 with PPL across spectral runs ⇒ r²(PPL, Q)
= 0.0064. Any metric M that factors through the joint output distribution
p(out_i, out_j) — including MI by (P4) — predicts Q only via this
distributional channel, with bound
    r²(M, Q) ≤ r²(PPL, Q) = 0.0064  [Data Processing Inequality, since
    M ⊥ Q | (output distribution) and behavioral Q is decided downstream
    of output distribution by tasks not used in M's calibration].

Concretely:
    r²(MI, Q) − r²(cos, Q) ≤ 0.0064 − 0 = 0.0064 ≪ 0.1  ⇒  K#418 FAIL.

The bound is tight: even if MI perfectly captured the joint output
distribution (sup over distributional metrics), the downstream-behavior
bottleneck enforces r² ≤ 0.0064.

Independent corroboration: F#22 explicitly KILLED KL on this exact
ground — "KL measures impact magnitude not quality — same failure as
cosine gating." MI inherits this failure by (P4).

## Proof of K#419 FAIL
Cosine cost on per-pair output traces of length n:
    C_cos(n) = 2n + n + 2 ≈ O(n) FLOPs (one inner product, two norms).

MI estimation via the standard KSG estimator (Kraskov–Stögbauer–
Grassberger, 2004) on continuous outputs requires k-NN search over n
points in d dimensions:
    C_MI_KSG(n,d) = O(d · n² · log n) for naive k-NN, or
                  = O(d · n · log n) with kd-trees but with constants ≈
                    100 from digamma evaluations and k-NN bookkeeping.

For n ≥ 100 calibration tokens (the floor under behavioral relevance, per
F#590/F#591 metric-swap critique requiring n_eval ≥ 15 PER DOMAIN, so
≥ 75 over five domains, with usual k = log n grid):
    C_MI / C_cos ≥ (d · n · log n · k_const) / n
                 = d · log n · k_const

With d = 2304 (Gemma 4 hidden dim), n = 75, k_const ≥ 50:
    C_MI / C_cos ≥ 2304 · log(75) · 50 ≈ 2304 · 4.3 · 50 ≈ 4.96 × 10⁵
    ≫ 100  ⇒  K#419 FAIL.

Even the cheapest MINE neural estimator (Belghazi 2018) requires training
a critic network — order(s) of magnitude more compute than a cosine
inner product. K#419 cannot pass under any production-realistic estimator.

## Conclusion
Both KCs FAIL by construction. KILLED.

## Reusable rule
*Preempt-axis*: composition-bug / parent-finding-contradicts-assumption,
sub-variant *distributional-metric-on-proxy-channel*. Any experiment
proposing a distributional/output-similarity metric M to predict
behavioral composition quality is preempt-killable via:
    r²(M, Q) ≤ r²(distributional-channel, Q) ≤ r²(PPL, Q) ≈ 0.0064 (F#286).

References: F#22 (KL = MI same family killed), F#544 (KL ρ=−0.7 macro),
F#286/F#285 (spectral-arc proxy), F#425 (cos fails at N=5).
