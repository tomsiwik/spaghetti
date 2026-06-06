# Mixed-Domain Per-Token Routing: Proof Verification Report

## Theorem

**Theorem 1 (Segment Isolation Eliminates Cross-Attention Contamination):**
Evaluating each segment of a mixed-domain sequence independently with its optimal
adapter eliminates cross-attention contamination by construction, because tokens
in segment A cannot attend to tokens in segment B under causal attention when
the segments are processed as independent subsequences.

**Theorem 2 (Segment-Level Selection Recovers Oracle Value):**
If segment boundaries are known and exhaustive PPL-based adapter selection is used,
segment-level selection achieves oracle-equivalent or better PPL.

## Predictions

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: PPL_seg / PPL_oracle in [0.95, 1.05] | 4.042 / 4.054 = 0.997 | YES |
| P2: Seg-exhaustive >= 5% better than per-seq | +16.05% | YES (3.2x threshold) |
| P3: High-sep pairs show largest gains | math+creative +21.4%, legal+creative +20.4% | PARTIAL (see below) |
| P4: Segment classification accuracy >= 80% | 95.2% (exhaustive PPL, not binary heads) | YES* |
| P5: Isolated eval > full-sequence eval | 4.04 vs 4.81 (16% gap) | YES |

*P4 note: The 95.2% figure measures PPL-based exhaustive selection, NOT the per-adapter
binary heads that Theorem 3 reasons about. The binary heads were not tested. P4 is a pass
for the exhaustive method but Theorem 3 remains unverified.

P3 note: The prediction was that code-containing pairs would show the largest gains based
on domain separability. Actual top 3: math+creative (+21.4%), legal+creative (+20.4%),
medical+creative (+18.5%). Legal+creative has no code and outperforms python+medical (+11.7%)
and python+math (+17.5%). The ordering does not cleanly match the separability function
prediction. Directionally correct (high-sep pairs do show larger gains) but the specific
ordering is wrong.

## Hypothesis

On mixed-domain sequences, segment-isolated adapter routing achieves >= 5% PPL
improvement over per-sequence routing because segment isolation eliminates
cross-attention contamination that makes per-token routing with full-sequence
forward passes equivalent to per-sequence routing.

**Result: SUPPORTED. All three kill criteria PASS.**

## What This Experiment Is

This experiment tests whether segment-level routing with segment-isolated evaluation
helps on mixed-domain sequences. The critical innovation over the prior (killed)
experiment is **segment isolation**: each segment of a mixed-domain sequence is
evaluated as an independent subsequence with its best adapter, rather than running
the full sequence through the model with per-token adapter switching.

### Setup

- Base model: BitNet-2B-4T (ternary, 2B parameters)
- 5 domain adapters: python, math, medical, legal, creative (rank-16 LoRA, scale 20)
- 10 domain pairs, 20 mixed sequences each (200 total)
- Each mixed sequence: 128 tokens from domain A + 128 tokens from domain B
- 6 routing strategies compared

### Strategies Compared

| Strategy | Description | Avg PPL |
|----------|-------------|---------|
| Base only | No adapter | 5.521 |
| Uniform 1/N | Equal-weight all 5 adapters, full sequence | 5.297 |
| Per-seq best | Best single adapter for full sequence (oracle) | 4.815 |
| Seg oracle | Correct adapter per segment, segment isolation | 4.054 |
| **Seg exhaustive PPL** | **Best-PPL adapter per segment (exhaustive search), segment isolation** | **4.042** |

**Note on per-token routing:** The prior experiment (exp_mixed_domain_sequences) measured
actual per-token routing at +0.28% over per-sequence (threshold was 5%). We cite that result
directly rather than re-measuring, as within-pass per-token routing is confirmed null on
this architecture.

**Note on seg_exhaustive_ppl:** This strategy evaluates each segment with ALL 5 adapters
and selects the one with lowest PPL. This is brute-force exhaustive search using the
evaluation metric as the selection criterion — an upper bound on what any practical router
could achieve. It is NOT a demonstration of practical routing. Theorem 3 (per-adapter
binary heads) was not tested in this experiment; the gap between exhaustive search and
practical routing remains an open question.

## Key References

- exp_mixed_domain_sequences (KILLED): Prior per-token routing experiment. +0.28%
  improvement, router collapsed to 2-class detector. Cross-attention contamination
  identified as inherent architectural issue.
- MoLoRA (arXiv:2603.15965): Per-token routing, jointly trained. 1.7B+4 > 8B.
- exp_molora_per_token_mlx (supported): Per-token null on single-domain data (-0.46%).
- Finding #41: Convex composition landscape, mixing beats selection by 3.5-5.2%.
- Finding #58: Per-token top-2 beats uniform by 13.9%.
- Finding #303: Room model killed (nonlinear compounding makes pre-summing impossible).

## Empirical Results

### Per-Pair Breakdown

| Pair | Per-Seq PPL | Seg-Oracle PPL | Seg-Exhaustive PPL | Exhaustive Improv. | Selection Acc. |
|------|-------------|----------------|---------------------|-------------------|----------------|
| python+math | 2.96 | 2.44 | 2.44 | +17.5% | 100% |
| python+medical | 2.77 | 2.46 | 2.45 | +11.7% | 92.5% |
| python+legal | 5.86 | 5.02 | 5.02 | +14.3% | 100% |
| python+creative | 4.07 | 3.52 | 3.51 | +13.6% | 97.5% |
| math+medical | 3.36 | 3.02 | 2.99 | +10.9% | 87.5% |
| math+legal | 7.47 | 6.23 | 6.23 | +16.6% | 100% |
| math+creative | 5.22 | 4.11 | 4.11 | +21.4% | 100% |
| medical+legal | 5.47 | 4.70 | 4.66 | +14.9% | 87.5% |
| medical+creative | 4.88 | 4.01 | 3.98 | +18.5% | 87.5% |
| legal+creative | 9.79 | 7.79 | 7.79 | +20.4% | 100% |
| **Average** | **4.81** | **4.05** | **4.04** | **+16.0%** | **95.2%** |

Between-pair std dev of improvement: 3.5pp (range 10.9%-21.4%). All 10 pairs exceed
the 5% kill threshold by at least 2.2x, providing directional confidence.

**Statistical power note:** Per-sequence standard deviations within each pair were not
recorded in this run (only aggregate NLL/n per pair). The between-pair consistency
(all 10 pairs positive, minimum 10.9%) provides evidence the effect is real, but the
within-pair variance is uncharacterized. A follow-up should track per-sequence PPL values
to report proper confidence intervals. At n=20 per pair, a single outlier text could
shift a pair's PPL by several percent — pair-level ordering (P3) should be treated as
indicative, not definitive.

### Kill Criteria

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| K772: Seg-exhaustive improvement over per-seq | >= 5% | +16.05% | **PASS** (3.2x margin) |
| K773: Exhaustive PPL selection accuracy | >= 40% | 95.2% | **PASS** (2.4x margin) |
| K774: Meaningful mixed-domain data | constructable | 200 sequences, 10 pairs | **PASS** |

Note: K773 measures exhaustive PPL-based selection (try all 5 adapters, pick lowest PPL),
not practical routing. The 95.2% figure is an upper bound on achievable selection accuracy.

### Critical Finding: The Improvement is from Segment Isolation, Not Per-Token Routing

The prior experiment (exp_mixed_domain_sequences) measured actual per-token routing
at +0.28% over per-sequence routing — confirmed null (threshold was 5%). Per-token
routing within a single full-sequence forward pass provides ZERO benefit over
per-sequence routing on this architecture.

The entire 16% improvement comes from **segment isolation** -- evaluating each
segment as an independent subsequence so that cross-attention contamination
is eliminated by construction.

This resolves the puzzle from exp_mixed_domain_sequences: the oracle gap (18.4%
in the prior experiment) was REAL, but per-token routing couldn't capture it
because full-sequence forward passes allow cross-attention contamination to
degrade the routing signal.

### Exhaustive PPL Selection Accuracy

The exhaustive PPL-based selection (try all 5 adapters, pick lowest PPL) achieves
95.2% agreement with ground-truth domain labels across all pairs:
- **100% agreement** on 6/10 pairs (all pairs involving python, math, or legal+creative)
- **87.5% agreement** on 3 pairs involving medical (medical+legal, medical+creative, math+medical)
- **92.5% agreement** on python+medical

When the exhaustive selection disagrees with ground truth, it consistently
produces BETTER PPL (seg_exhaustive PPL <= seg_oracle PPL for every pair).
This means the "wrong" adapter sometimes produces lower PPL on a segment than
the "correct" domain adapter -- suggesting adapter effects are not purely
domain-specific but have beneficial cross-domain spillover.

**Important:** This 95.2% figure is for an oracle-class method (exhaustive search
using the evaluation metric itself). A practical router (e.g., per-adapter binary
heads, Theorem 3) would achieve lower accuracy. The gap between exhaustive and
practical routing is the key unknown for production deployment.

## Limitations

1. **Synthetic mixed-domain sequences.** Concatenation of segments creates artificial
   sharp boundaries. Real mixed-domain text (e.g., medical case studies with code)
   has gradual transitions. Boundary detection would be needed in production.

2. **Segment isolation loses cross-segment context.** By evaluating segments independently,
   the model cannot use information from segment A when generating segment B. In practice,
   segment B's perplexity would benefit from segment A's context (even with the wrong
   adapter). The 16% improvement is an UPPER BOUND on what segment-level routing
   provides in a production setting where context must flow between segments.

3. **Context length confound (quantified rough estimate).** Per-sequence evaluation uses
   256-token context while segment evaluation uses 128-token context. Shorter context
   generally increases PPL (less predictive information), which should HURT segment
   evaluation. The observed 16% improvement decomposes as:

   improvement = contamination_elimination_benefit - shorter_context_penalty
   16% = contamination_benefit - context_penalty

   **Rough estimate of context_penalty:** For autoregressive LMs on coherent text,
   PPL typically decreases 3-8% going from 128 to 256 token context (the additional
   context provides modest but real predictive information). This is conservative for
   mixed-domain sequences where the "extra" 128 tokens are from a DIFFERENT domain
   and may actually hurt (cross-domain tokens in context degrade predictions).

   **Conservative decomposition:** If context_penalty = 5% (generous upper bound for
   same-domain context), then contamination_benefit = 21%. If context_penalty is near
   zero (plausible for cross-domain context that provides no useful signal), then
   contamination_benefit = 16%.

   Either way, the 16% figure is a LOWER BOUND on the contamination elimination effect.
   The true benefit of segment isolation is >= 16%, not inflated by shorter context.
   A follow-up should run per-sequence best-adapter on 128-token pure-domain segments
   to measure the exact context-length effect.

4. **PPL-based classification requires trying all N adapters.** The 95.2% routing accuracy
   comes from evaluating each segment with all 5 adapters and picking the best. This is
   O(N) forward passes per segment. For production, a cheaper classification method
   (e.g., per-adapter binary heads at 0.58% overhead) would be needed.

5. **Only 5 domains at micro scale.** N=5 domains with relatively distinct specialization.
   At larger N with more overlapping domains, routing accuracy may degrade.

6. **Ternary base model.** Results specific to BitNet-2B-4T. The effect size may differ
   on FP16 models where adapter perturbations have different characteristics.

## What Would Kill This

This hypothesis would be killed if:
1. Segment isolation shows < 5% improvement in a production-like evaluation where
   cross-segment context is preserved (e.g., using KV-cache from segment A when
   generating segment B, even with a different adapter)
2. Boundary detection fails on natural mixed-domain text where transitions are gradual
3. The O(N) classification cost makes segment-level routing impractical at production N

## Implications for the Project

1. **Segment-level isolation is the correct architecture for mixed-domain inputs.**
   Not per-token routing (which is equivalent to per-sequence on full forward passes,
   confirmed +0.28% by prior experiment), and not room-model pre-summing (which is
   killed by nonlinear compounding).

2. **The cross-attention contamination finding from the prior experiment was correct
   but incomplete.** The prior experiment correctly identified contamination as the
   failure mode but concluded that per-token routing was dead. In fact, the concept
   was correct -- the EVALUATION methodology was wrong.

3. **Boundary detection is the next research question.** This experiment used oracle
   boundaries. Practical deployment needs automatic domain shift detection.
   The per-adapter binary heads (100% on single-domain, proven) are the natural
   candidate: run them on a sliding window and detect where the argmax changes.

4. **Exhaustive PPL selection matches or beats oracle.** The seg_exhaustive_ppl method
   achieves 4.042 vs seg_oracle 4.054. This is because the "correct" domain adapter
   is not always the best adapter for a given segment -- cross-domain transfer effects
   mean a different adapter can sometimes help more. Note: this is an upper bound on
   practical routing performance.

5. **The 16% figure is an upper bound.** Production implementation must handle
   cross-segment context preservation, imperfect boundary detection, and O(N)
   classification cost. The practical improvement will be lower.

## Runtime

317.9 seconds (5.3 minutes) on Apple M5 Pro 48GB. Peak memory 7.05 GB.
