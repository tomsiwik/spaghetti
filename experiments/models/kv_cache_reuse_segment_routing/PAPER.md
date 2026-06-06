# KV-Cache Reuse Across Adapter Switches: Proof Verification Report

## Theorem

From MATH.md, restated:

**Theorem 3:** KV-cache reuse across adapter switches should not degrade quality
vs isolated evaluation, because additional context (even from a different adapter)
provides information that reduces prediction uncertainty (data processing inequality).

**Theorem 2:** The cross-adapter attention interaction is O(alpha^2 * r^2 / d^2) ~ 1.6%
of the base attention pattern, making it negligible.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: KV-reuse PPL <= isolated PPL (Thm 3) | KV-reuse 5.704 > isolated 5.367 (+6.27%) | **NO** |
| P2: KV-reuse within 3% of full-recompute (Thm 2) | KV-reuse 13.26% better than full-recompute | **NO** (direction correct, magnitude wrong) |
| P3: Latency speedup > 1.2x | Speedup 0.69x (KV-reuse is SLOWER) | **NO** |
| P4: Math+medical shows least improvement | Math+medical is the ONLY pair showing improvement (+1.72%) | **PARTIALLY** (prediction direction inverted) |

**Verdict: All three kill criteria FAIL. Hypothesis killed.**

## Hypothesis

KV-cache from prior segments can be reused across adapter switches with < 3%
PPL degradation vs full recompute, because LoRA perturbations modify attention
patterns minimally (bounded by Grassmannian orthogonality).

**Status: KILLED.** Cross-adapter KV-cache reuse hurts quality by 6.27% vs
isolated evaluation. The theoretical perturbation bound (~1.6%) drastically
underestimates the actual effect.

## What This Experiment Tested

Three strategies for evaluating segment B in a mixed-domain sequence [A|B]:

1. **Isolated:** Evaluate segment B independently with adapter B. No context
   from segment A. (Finding #305 baseline: PPL 4.042 combined, 5.367 seg B only.)

2. **KV-reuse:** Process segment A with adapter A (filling KV-cache), switch to
   adapter B, process segment B using cached KV from segment A.

3. **Full-recompute:** Process full [A+B] sequence with adapter B. Compute PPL
   on segment B only. Segment A tokens are processed by the WRONG adapter.

## Key References

- Finding #305: Segment isolation +16% over per-sequence (SUPPORTED)
- arXiv:2512.17910: Cross-model KV-cache reuse, up to 58x latency reduction
- Grassmannian skeleton: ||Delta W_i^T Delta W_j|| bound (VISION.md)

## Empirical Results

### Global Summary

| Strategy | Seg B PPL | vs Isolated | vs Full-Recompute |
|----------|-----------|-------------|-------------------|
| Isolated | 5.367 | -- | -18.37% (better) |
| KV-reuse | 5.704 | +6.27% (worse) | -13.26% (better) |
| Full-recompute | 6.575 | +22.49% (worse) | -- |

### Ordering

Isolated < KV-reuse < Full-recompute (lower is better)

This ordering is the OPPOSITE of what Theorem 3 predicted. The expected ordering
was Full-recompute < KV-reuse < Isolated. Instead:

- **Isolated is best**: No cross-segment context = no cross-adapter confusion.
  Segment B performs best when it doesn't try to attend to segment A at all.

- **KV-reuse is middle**: Cross-segment context from adapter A's cache provides
  some useful information but also introduces cross-adapter confusion.

- **Full-recompute is worst**: Processing segment A with the WRONG adapter (B)
  produces actively misleading context. Segment B attending to incorrectly-processed
  segment A tokens is worse than no context at all.

### Per-Pair Analysis

| Pair | Isolated | KV-reuse | Full-recomp | KV context effect |
|------|----------|----------|-------------|-------------------|
| python+math | 2.989 | 3.268 | 3.813 | -9.35% (hurts) |
| python+medical | 2.720 | 2.785 | 3.271 | -2.39% (hurts) |
| python+legal | 11.154 | 11.680 | 13.271 | -4.71% (hurts) |
| python+creative | 5.416 | 5.802 | 6.642 | -7.13% (hurts) |
| math+medical | 2.866 | 2.817 | 3.369 | **+1.72% (helps)** |
| math+legal | 11.983 | 12.395 | 13.826 | -3.44% (hurts) |
| math+creative | 5.281 | 5.730 | 6.704 | -8.49% (hurts) |
| medical+legal | 8.058 | 8.643 | 9.931 | -7.26% (hurts) |
| medical+creative | 5.498 | 6.146 | 7.073 | -11.79% (hurts) |
| legal+creative | 5.026 | 5.559 | 6.269 | -10.60% (hurts) |

Only math+medical shows KV-reuse helping (+1.72%). This is the most semantically
similar pair (shared technical vocabulary, overlapping reasoning patterns).

### Latency Analysis

| Strategy | Mean latency (s) |
|----------|-----------------|
| Isolated (2 segments) | 0.164 |
| KV-reuse (prefill + continue) | 0.150 |
| Full-recompute (1 pass, 256 tok) | 0.104 |

Full-recompute is fastest because a single 256-token forward pass is more
efficient than two separate passes (GPU batching, reduced Python overhead).
KV-reuse requires two passes (prefill seg A + process seg B), which is 1.45x
slower than single-pass.

## Why the Proof Failed

### Error 1: Data Processing Inequality Misapplied (Theorem 3)

The data processing inequality states that conditioning on additional observations
cannot increase entropy. But this assumes the observations are CORRECT. KV-cache
entries from adapter A are not "observations of segment A" -- they are "adapter A's
internal representation of segment A." When adapter B's queries attend to adapter A's
keys/values, the representations are subtly incompatible. The additional information
is noise-contaminated rather than pure signal.

### Error 2: Perturbation Bound Too Loose (Theorem 2)

The theoretical bound of ~1.6% cross-adapter interaction assumed the dominant term
was the base-base attention (h^T W_Q^T W_K h'). But with LoRA scale alpha=20.0, the
adapter terms are NOT small perturbations -- they are significant modifications:

  alpha * r / d = 20 * 16 / 2560 = 0.125

This is 12.5% per projection. The cross-adapter term involves products of these
perturbations PLUS the interaction through hidden states that themselves have been
modified by the different adapters at every preceding layer. The layer-wise
accumulation (28 layers) amplifies the mismatch far beyond the single-layer bound.

### Error 3: Context Is Not Always Helpful for Language Models

The assumption that "more context = lower PPL" holds for CONSISTENT context
(same model, same adapter). For INCONSISTENT context (different adapter's
representation), the model's attention mechanism must reconcile two different
"dialects" of the same information. This reconciliation is harder than starting
from scratch.

## What Was Learned

1. **Isolated segment evaluation is the correct baseline.** Segment B's PPL is
   best when it has NO cross-segment context. The 128-token segment provides
   sufficient local context for the adapter to work effectively.

2. **Cross-adapter KV-cache reuse HURTS quality.** The cross-adapter attention
   incompatibility is not a 1.6% perturbation -- it's a 6.27% degradation.
   The Grassmannian orthogonality that protects composition (||cos|| < 0.05) is
   exactly what makes KV-cache reuse fail: orthogonal adapters produce maximally
   different key/value representations.

3. **The wrong-adapter penalty is large.** Full-recompute (adapter B on segment A)
   is 22.5% worse than isolated. Using the WRONG adapter to process preceding
   context is actively harmful.

4. **Domain similarity predicts compatibility.** Math+medical is the only pair
   where KV-reuse helps. This suggests that KV-cache reuse might work for adapters
   within the same semantic cluster, but not across clusters.

5. **Latency argument is invalid.** The assumed speedup from KV-cache reuse
   (avoiding recomputation) is negated by the overhead of two separate forward
   passes vs one combined pass.

## Implications for the Architecture

Finding #305's segment isolation is NOT leaving value on the table by discarding
cross-segment context. The context from a different domain segment processed by
a different adapter is noise, not signal. Segment isolation is the correct
strategy for mixed-domain sequences.

The one exception (math+medical, +1.72%) suggests that for WITHIN-CLUSTER
domain switches, some form of context preservation might help. But this is a
narrow case that does not justify the complexity of KV-cache management.

## Limitations

1. Synthetic mixed sequences with sharp boundaries (real text has gradual transitions)
2. Only 5 domains at micro scale with toy adapters
3. Ternary base model (BitNet-2B-4T) -- results may differ on FP16 models
4. LoRA scale alpha=20.0 is high; smaller perturbations might show different results
5. 128-token segments may be long enough for self-contained context; shorter segments
   would make cross-segment context more valuable

## What Would Kill This (Already Killed)

- K781 FAIL: 13.26% gap (threshold 3%)
- K782 FAIL: 0.69x speedup (threshold 1.2x)
- K783 FAIL: -6.27% context improvement (threshold > 0%)

All three criteria decisively killed. The hypothesis is wrong: cross-adapter
KV-cache reuse on mixed-domain sequences with Grassmannian-orthogonal adapters
does NOT preserve quality.
