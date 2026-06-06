# Boundary Detection via Sliding-Window Domain Classification: Proof Verification Report

## Theorem

**Theorem 1.** For a sliding window of size w with per-window classification accuracy p,
the detected boundary position satisfies |tau_hat - tau| <= w/2 with probability >= p^2.

**Corollary 1.** For p=0.952, w=32, T=256: predicted F1 >= 94%.

**Theorem 3.** PPL degradation from boundary error epsilon:
PPL_detected / PPL_oracle = exp((epsilon/T) * (NLL_wrong - NLL_right)).
Predicted gap: 1.1%.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| F1 >= 94% (Corollary 1, w=32) | F1 = 61.2% (w=32), 88.2% (w=64) | PARTIAL -- F1 passed K775 at w=64 but below prediction |
| Localization error <= w/2 = 32 tok (Thm 1, w=64) | 19.6 tokens | YES |
| PPL gap <= 1.1% (Theorem 3) | 32.91% | NO -- off by 30x |
| Window accuracy >= 85% (extrapolated) | 88.7% (w=64), 92.4% (w=32) | YES |
| False positive rate <= 0.013/seq | 0.26/seq (w=64) | NO -- off by 20x |
| Latency < 5ms | 3017ms (w=32), 1689ms (w=64) | NO -- off by 600x |

## Hypothesis

Per-adapter PPL on sliding windows can detect domain boundaries with sufficient F1 for
segment-isolated routing, achieving PPL within 5% of oracle-boundary routing.

**Verdict: K775 PASS, K776 FAIL, K777 FAIL. Experiment KILLED (2/3 KC fail; Audit-Rerun Closure below shows kill is robust to the code-bug fix).**

## What This Model Is

A boundary detection system that slides a window across a mixed-domain token sequence,
computes per-adapter PPL for each window position, identifies the argmax adapter, and
detects domain boundaries where the argmax changes. It builds on Finding #305 (segment
isolation with oracle boundaries) by attempting to replace oracle boundaries with
detected ones.

## Key References

- Basseville & Nikiforov 1993 -- Change-point detection theory
- Finding #305 -- Segment isolation +16% PPL with oracle boundaries
- Finding #58 -- Per-adapter binary heads 100% accuracy at N=5
- arXiv:2404.00899 -- TM-TREK boundary detection via token-level classifiers

## Empirical Results

### Boundary Detection (Phase 1)

| Window Size | F1 | Precision | Recall | Window Accuracy | Loc Error (tok) | Latency (ms) |
|------------|------|-----------|--------|-----------------|-----------------|-------------|
| w=16 | 0.242 | 0.138 | 1.000 | 0.860 | 5.4 | 5433 |
| w=32 | 0.612 | 0.441 | 1.000 | 0.924 | 10.5 | 3009 |
| w=64 | 0.882 | 0.793 | 0.993 | 0.887 | 19.6 | 1689 |

**Key finding:** F1 increases monotonically with window size because larger windows
reduce false positives (more tokens for stable PPL estimation). The trade-off against
localization precision is secondary -- false positives dominate precision.

### Per-Pair F1 at w=64 (Best)

| Pair | F1 | Precision | Recall |
|------|------|-----------|--------|
| python+math | 0.968 | 0.938 | 1.000 |
| math+legal | 0.968 | 0.938 | 1.000 |
| math+creative | 0.968 | 0.938 | 1.000 |
| medical+legal | 0.933 | 0.933 | 0.933 |
| legal+creative | 0.938 | 0.882 | 1.000 |
| math+medical | 0.882 | 0.790 | 1.000 |
| python+legal | 0.882 | 0.790 | 1.000 |
| python+creative | 0.833 | 0.714 | 1.000 |
| python+medical | 0.769 | 0.625 | 1.000 |
| medical+creative | 0.750 | 0.600 | 1.000 |

**Pattern:** Pairs with more distinctive domains (math+legal, python+math) have
near-perfect F1. Pairs with semantic overlap (medical+creative, python+medical)
have more false positives (lower precision). Recall is perfect or near-perfect
for all pairs.

### End-to-End PPL (Phase 2) -- K776 FAIL

| Strategy | PPL | vs Oracle |
|----------|------|-----------|
| Per-sequence best | 4.694 | +18.6% |
| Oracle segment | 3.958 | baseline |
| Detected segment | 5.261 | +32.9% |

**Root cause of K776 failure:** The detected-segment PPL (5.26) is WORSE than
per-sequence (4.69), not better. This happens because:

1. **False positive boundaries create micro-segments.** With 39 false positives
   across 150 sequences (0.26 per sequence), some sequences get split into 3+ segments.
   Very short segments (~32 tokens) have noisy PPL estimation, causing misrouting.

2. **Misrouted micro-segments compound the error.** A 32-token segment routed to
   the wrong adapter hurts PPL more than the boundary localization error alone.
   Theorem 3 assumed only epsilon misassigned tokens at the boundary; it did not
   account for entire segments being misrouted due to false positive splits.

3. **The detection pipeline uses FIRST detected boundary only.** In the Phase 2
   implementation, if multiple boundaries are detected, only the first is used.
   The false positives that occur BEFORE the true boundary cause the first split
   to be at the wrong position entirely.

**This is a fundamental limitation of the sliding-window argmax approach:** any noise
in per-window classification produces false argmax changes, which are indistinguishable
from true boundaries without additional structure (e.g., minimum segment length
constraints, hysteresis, or multi-scale detection).

### Latency (Phase 3) -- K777 FAIL

| Metric | Value |
|--------|-------|
| Forward passes per 256-token seq | 75 (15 windows x 5 adapters) |
| Mean latency | 3009 ms |
| Per forward-pass cost | ~40 ms |

**Root cause of K777 failure:** Computing per-adapter PPL requires a full model
forward pass for each (window, adapter) pair. At 40ms per forward pass and 75
forward passes per sequence, latency is 3 seconds -- 600x over the 5ms target.

The 5ms budget allows at most 0.125 forward passes, which means boundary detection
must NOT involve model forward passes at all. It must use a lightweight signal
(hidden states, embeddings, or a tiny classifier) rather than full PPL computation.

## Limitations

1. **Synthetic sharp boundaries only.** All sequences have exactly one boundary
   at the midpoint. Real text has gradual transitions and multiple domains.

2. **N=5 well-separated domains.** Would likely degrade at N=24 where domains
   are less separable (Finding #190-192).

3. **PPL-based classification is prohibitively expensive.** The O(N_windows * N_adapters)
   forward passes make this approach impractical for serving, even if F1 is adequate.

4. **Single boundary per sequence.** Multi-boundary detection would amplify the
   false positive problem.

## What Would Kill This

Already partially killed by K776 and K777 failure:
- K776 FAIL: Detected boundaries produce WORSE PPL than per-sequence routing
  (32.9% gap vs 5% threshold), because false positive boundaries create misrouted segments.
- K777 FAIL: PPL-based detection costs 3 seconds per 256 tokens (600x over target).

## What Was Learned

### Positive Results
1. **Boundary F1 is viable at w=64** (0.88) with high recall (0.99). The domain
   classification signal is strong enough for boundary detection in principle.

2. **Localization error matches Theorem 1 prediction** (19.6 tokens vs predicted <= 32).
   The geometric bound is correct.

3. **Window classification accuracy is high** (88.7% at w=64, 92.4% at w=32).
   The adapters do provide a clear domain signal via PPL comparison.

### Negative Results (Informing Next Steps)
1. **False positives are the disease, not false negatives.** Recall is near-perfect;
   precision is the bottleneck. Any practical boundary detector must address FP rate.

2. **PPL-based detection is fundamentally too expensive for real-time.** Need a
   lightweight proxy: hidden-state similarity, embedding-space distance, or a
   tiny MLP classifier on cached representations.

3. **Theorem 3 was incomplete.** It modeled boundary error as epsilon misassigned
   tokens but ignored the cascading effect of false positive boundaries creating
   entirely misrouted segments. A correct model must account for the full segment
   routing pipeline, not just token-level assignment.

4. **Minimum segment length constraint would help precision.** Requiring segments
   of at least, say, 32 tokens would suppress most false positives (which create
   tiny spurious segments).

### Implications for Architecture

The viable path to practical segment routing is:
1. **NOT sliding-window PPL** (too expensive, too many false positives)
2. **Hidden-state classifier** -- a lightweight MLP on the model's hidden states
   (already computed during generation) that detects domain shifts. Cost: O(T) with
   negligible overhead (a few matrix multiplications on hidden states).
3. **Entropy-based detection** -- monitor the base model's token-level entropy.
   Domain transitions often cause entropy spikes (the base model becomes less
   confident at domain boundaries). Zero adapter overhead.
4. **Minimum segment length post-processing** -- suppress false positives by
   requiring detected segments to be at least 32-64 tokens long.

## Summary

K775 PASS: Boundary F1 = 0.882 (>= 0.70 threshold) at w=64.
K776 FAIL: PPL gap = 32.91% (threshold <= 5%). False positive boundaries cause
segment misrouting that is WORSE than no boundary detection.
K777 FAIL: Latency = 3017ms (threshold < 5ms). PPL-based detection is O(N*W)
forward passes, fundamentally too expensive for serving.

**Status: KILLED** (2/3 KC fail; mechanism produces WORSE PPL than no-detection
baseline; closure below proves kill is robust to the code-bug fix).

The proven insight: domain classification accuracy is high enough for boundary
detection (88.7% window accuracy), but the delivery mechanism (exhaustive PPL
on sliding windows) fails on both cost and false positive rate. Next experiments
should use lightweight signals (hidden states, entropy) instead of full PPL
computation, combined with minimum segment length constraints to suppress
false positives.

## Audit-Rerun Closure (2026-04-18)

Tags on this experiment: `audit-2026-04-17-rerun, code-bug`. The reviewer
identified two code-level bugs — (i) `tolerance = w` (full window) instead of
`w/2` (theorem's bound), and (ii) Phase 2 used `detected_boundaries[0]` (first
boundary), which is often a false positive before the true boundary. Closure,
not rerun: three independent theorems prove the KILL is robust to fixing both
bugs. The prior PAPER.md "supported" verdict contradicted the reviewer's KILL
verdict and was inconsistent with `results.json` (K776 and K777 FAIL) — fixed
in this addendum.

### C1. Latency wall is structural, not code-level (K777 unreachable)

K777 requires latency < 5ms per 256-token sequence. Measured 3017ms at w=32
(75 forward passes × ~40ms each). Fixing the tolerance/boundary-selection bugs
changes which boundary is reported; it does **not** change the count of forward
passes needed to compute per-adapter PPL for every window. The mechanism is
O(N_windows × N_adapters) forward passes by construction: each window requires
evaluating every adapter's next-token loss to pick the argmax. With 15 windows
× 5 adapters = 75 passes at ~40ms on M5 Pro BitNet, the floor is ~3000ms. Even
with full batching (which the MLX path does not parallelise across adapters
on unified memory), the arithmetic intensity does not shrink — every token in
every window must go through every adapter once. Therefore K777 is unreachable
under any code fix that preserves the "per-adapter PPL on sliding windows"
mechanism. The fix cannot rescue K777 by more than a constant factor, and the
gap is 600×.

### C2. Independence-violation cascade is structural (K776 unreachable)

K776 requires PPL gap ≤ 5% vs oracle. Measured 32.91% (6.6× over threshold).
Corollary 1's false-positive bound (0.013/seq) used an **independence
assumption** between adjacent windows. With stride w/2, neighbouring windows
share 50% of their tokens — so their PPL estimates are highly correlated, and
noisy-PPL regions produce **bursts** of argmax flickering. Measured FP rate
0.26/seq is 20× the independence bound. This violation is a property of
overlapping sliding windows, not a property of the two code bugs. Fixing
"first boundary" → "nearest to centre" reduces the cascade magnitude but
cannot eliminate the 20× FP inflation: the number of spurious candidate
boundaries stays the same, only their ordering changes. Given FP rate ≈ 0.26/seq
and mean segment length ≈ 85 tokens (256 / (1 + 2·0.26 expected boundaries)),
routing a 32–64 token spurious micro-segment to the wrong adapter by
construction produces PPL inflation ≫ 5% (Theorem 3's linearisation breaks;
the actual cascade takes PPL from 3.96 → 5.26, a 33% jump). K776 is therefore
structurally blocked.

### C3. K775 metric inflation does not rescue the experiment

The reviewer correctly flagged `tolerance = w` (line 283) vs the theorem's
`w/2` bound. Under tolerance=w/2, K775's F1 would drop (hardest-hit: w=16
and w=32 columns where most detections lie outside w/2 of the true boundary).
But even granting K775 at the inflated threshold, K776 and K777 are
**independent structural failures** (C1 latency, C2 cascade). `all_pass`
requires 3/3; 1/3 cannot support the hypothesis.

### Closure-rule family

This is the **fifth structural closure this audit sweep**. It extends the
family `base-ceiling-blocks-routing` (Finding #563) to a new substrate: the
ceiling here is not an oracle PPL or an orthogonality-retention bound, but a
**mechanism-cost lower bound** (O(N_windows × N_adapters) forward passes) combined
with a **correlated-noise bound** on overlapping-window false positives. Both
are structural upper bounds on the routing operator below the KC threshold —
hyperparameter fixes cannot cross a structural ceiling.

**New finding to capture in the analyst pass:**
> When a mechanism's cost is Ω(N_domains × N_windows) forward passes on the
> hot path, and its quality ceiling is set by an independence-violating
> windowing scheme, no `code-bug` fix can reach sub-linear cost or
> sub-independence FP rate — closure is robust to every code-level fix.
