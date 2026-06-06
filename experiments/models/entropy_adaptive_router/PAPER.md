# Entropy-Adaptive Router: Research Digest (Revised)

## Hypothesis

Adapting per-token expert count (k) based on routing entropy reduces average
compute while matching fixed k=2 quality under composition.

## What This Model Is

The entropy-adaptive router extends the standard softmax capsule MoE router by
computing the Shannon entropy of the routing probability distribution for each
token. Tokens with low entropy (confident routing, one expert dominates) use
k=1. Tokens with high entropy (uncertain routing, multiple experts compete) use
k=2. The entropy threshold is a learned parameter calibrated alongside the router.

This is inspired by arithmetic coding, where variable bits are allocated per
symbol based on probability -- here, variable compute (number of experts) is
allocated per token based on routing confidence.

## Lineage in the Arena

```
gpt
 `-- capsule_moe (fixed top-k routing)
      `-- entropy_adaptive_router (variable-k based on routing entropy)
```

## Key References

- **Switch Transformers** (Fedus et al. 2022): k=1 routing at scale, capacity-bound
- **ReMoE** (ICLR 2025): ReLU routing gives variable expert count naturally
- **Mixture-of-Depths** (Raposo et al. 2024): adaptive compute at layer level
  (skip layers for easy tokens), complementary to our within-layer approach
- **DeepSeek-V3**: 256 fine-grained experts, fixed k=8, bias-based balancing
- **Mixtral 8x7B**: production top-2 MoE

No prior work uses entropy of routing scores to dynamically select k within an
MoE layer. ReMoE achieves variable k via ReLU (binary on/off per expert), but
without explicit entropy-based selection. Mixture-of-Depths adapts compute at
the depth level (skip entire layers), not width level (fewer experts per layer).

## Revision History

**V2 (current):** Addresses 5 required fixes from adversarial review:
1. Implemented conditional expert execution; measured actual wall-clock savings (zero)
2. Fixed raw_tau unfreezing during calibration
3. Added random-k baseline control
4. Reported per-layer entropy profile as 3-seed mean with standard deviations
5. Measured soft-to-hard quality gap

## Empirical Results

### Experiment 2 (Revised): Composition (base 300 + domain 300 + cal 200 steps, 3 seeds)

| Config | Val Loss | Avg k | vs k=2 | vs Joint |
|--------|----------|-------|--------|----------|
| Joint ref | 0.5096 | - | - | baseline |
| Fixed k=1 | 1.1133 | 1.0 | +120.4% | +118.5% |
| Fixed k=2 | 0.5052 | 2.0 | baseline | -0.87% |
| EA sc=0.0 | 0.5074 | 1.94 | +0.44% | -0.44% |
| EA sc=0.1 | 0.5085 | 1.90 | +0.66% | -0.22% |
| EA sc=0.3 | 0.5084 | 1.83 | +0.65% | -0.23% |

### Random-k Baselines (Fix 3)

For each EA config, a control where tokens randomly get k=1 with probability
matched to the observed fraction:

| Config | Val Loss | vs matched EA |
|--------|----------|---------------|
| Random-k (matched to ea_sc0.0) | 0.5059 | -0.28% |
| Random-k (matched to ea_sc0.1) | 0.5061 | -0.47% |
| Random-k (matched to ea_sc0.3) | 0.5082 | -0.04% |

**The random-k baseline matches or beats entropy-adaptive routing in all cases.**
This means the entropy criterion provides no value over random budget allocation
at this scale. The quality is maintained not because the router correctly identifies
which tokens need fewer experts, but because the composition task is robust to
randomly reducing a small fraction of tokens to k=1.

### Soft-to-Hard Quality Gap (Fix 5)

| Config | Soft Loss | Hard Loss | Gap |
|--------|-----------|-----------|-----|
| ea_sc0.0 | 0.5074 | 0.5074 | +0.01% |
| ea_sc0.1 | 0.5085 | 0.5086 | +0.02% |
| ea_sc0.3 | 0.5084 | 0.5087 | +0.05% |

The soft-to-hard gap is negligible (<0.1%). Hard thresholding is deployable
without retraining. The sigmoid temperature of 0.1 is sharp enough that the
soft approximation and the hard decision are functionally equivalent.

### Per-Layer Entropy Profile (3-seed mean +/- std, Fix 4)

**ea_sc0.0:**

| Layer | Mean H | tau | Avg k | Frac k=1 |
|-------|--------|-----|-------|----------|
| 0 | 1.611 +/- 0.175 | 0.711 +/- 0.197 | 1.918 | 8.2% +/- 11.2% |
| 1 | 1.633 +/- 0.149 | 0.653 +/- 0.051 | 1.952 | 4.8% +/- 2.3% |
| 2 | 1.637 +/- 0.043 | 0.723 +/- 0.153 | 1.939 | 6.1% +/- 5.7% |
| 3 | 1.678 +/- 0.116 | 0.671 +/- 0.106 | 1.959 | 4.1% +/- 2.1% |

**ea_sc0.3:**

| Layer | Mean H | tau | Avg k | Frac k=1 |
|-------|--------|-----|-------|----------|
| 0 | 1.565 +/- 0.145 | 0.830 +/- 0.238 | 1.831 | 16.9% +/- 7.7% |
| 1 | 1.458 +/- 0.245 | 0.988 +/- 0.194 | 1.800 | 20.0% +/- 14.2% |
| 2 | 1.552 +/- 0.051 | 1.112 +/- 0.143 | 1.814 | 18.6% +/- 4.5% |
| 3 | 1.629 +/- 0.093 | 0.911 +/- 0.022 | 1.884 | 11.6% +/- 3.1% |

**V1 finding invalidated:** The original PAPER.md reported that "early layers are
more confident (lower entropy), consistent with Layer 0 showing most redundancy."
With proper raw_tau unfreezing (Fix 2) and 3-seed averaging (Fix 4), this pattern
does NOT hold robustly. The frac_k1 standard deviations are large (up to 14.2%),
and the layer ordering varies across seeds. The per-layer entropy profile is
seed-dependent, not a stable architectural property.

### Wall-Clock Timing (Fix 1)

| Config | Eval Time (s) | vs k=2 |
|--------|---------------|--------|
| fixed_k2 | 0.090 | baseline |
| ea_sc0.0 | 0.408 | 4.5x slower |
| ea_sc0.1 | 0.319 | 3.5x slower |
| ea_sc0.3 | 0.335 | 3.7x slower |

**FLOP savings are zero.** Conditional expert execution was implemented (skip
groups where max weight < epsilon=0.001). In practice, 0 out of 8 experts are
skipped in nearly all forward passes, because the soft mask interpolation
distributes nonzero weight across all groups. The entropy computation, topk
operations, mask interpolation, and per-expert max-weight checks add substantial
overhead, making EA 3-4.5x slower than fixed k=2 in wall-clock time.

The original PAPER.md claimed "8.5% FLOP savings." This was a theoretical
projection that assumed conditional execution of experts with near-zero weight.
In practice:
- The softmax distribution is never sparse enough at G=8 to zero out any expert
- The overhead of entropy computation exceeds any hypothetical savings
- All experts are computed for all tokens regardless of adaptive k

## Kill Criteria Evaluation (Revised)

| Criterion | Result | Verdict |
|-----------|--------|---------|
| KC1: Variable-k worse than fixed k=2 | +0.44% to +0.66% | **MARGINAL** (noise range) |
| KC2: Avg k doesn't drop below 1.8 | 1.83-1.94 | **FAIL** |
| Random-k control | -0.04% to -0.47% vs EA | **ENTROPY ADDS NO VALUE** |

**KC1 weakened from V1:** The original +0.01% gap widened to +0.44-0.66% after
fixing raw_tau unfreezing. Still within noise, but the margin is narrower.

**KC2 still fails:** Average k ranges from 1.83 (sc=0.3) to 1.94 (sc=0.0),
further above the 1.8 threshold than the original 1.82-1.85 range.

**New finding (random-k):** The entropy criterion provides no value over random
budget allocation. A random k=1/k=2 assignment with matched probability produces
the same or better quality. This undermines the core hypothesis that routing
entropy identifies which tokens can safely use fewer experts.

## Interpretation

The revision reveals that the entropy-adaptive router's quality preservation
is not due to intelligent token-level compute allocation, but rather due to the
composition task's inherent robustness to randomly reducing a small fraction of
tokens to k=1. The entropy criterion is a costly no-op.

**What the revision confirmed:**
1. Hard thresholding works perfectly (gap < 0.1%) -- if the mechanism were useful,
   deployment would be straightforward
2. The k=1 catastrophe under composition is real but requires MOST tokens to
   use k=1 -- a small random fraction at k=1 is fine

**What the revision invalidated:**
1. FLOP savings are zero (4x slower, not 8.5% faster)
2. Per-layer entropy profiles are not robust (high variance across seeds)
3. Entropy-based k-selection adds no value over random (the core contribution)
4. The "learned per-layer threshold" finding was partially an artifact of
   frozen raw_tau (V1 tau range 0.33-0.43, V2 tau range 0.55-1.27)

## Micro-Scale Limitations

1. **Only G=8 experts**: At larger G, softmax distributions may be sparser,
   making entropy a more discriminative signal
2. **Only k=1 vs k=2**: The binary choice may be too coarse -- k=1..8 at G=256
   could show more differentiation
3. **Character-level names data**: Low diversity means the router cannot develop
   strong specialization. With diverse domains, entropy variation may be larger
4. **Composition of 2 domains only**: More domains increase the composition challenge
5. **Random-k baseline used hard assignment**: The random baseline uses hard
   k=1/k=2 decisions while EA uses soft interpolation -- this is actually a
   disadvantage for random-k, making EA's failure more stark

## What Would Kill This

**Already demonstrated at micro scale:**
- Entropy criterion adds no value over random k assignment (KILLED)
- No wall-clock savings (KILLED)
- Per-layer patterns not robust (KILLED)

**Conditions under which entropy-adaptive might revive at macro scale:**
- If softmax distributions become bimodal at large G (some tokens very peaked,
  others very flat), entropy discrimination becomes meaningful
- If conditional execution is hardware-efficient at large G (skip 200 of 256
  experts vs skip 6 of 8), savings dominate overhead
- If the task demands genuine per-token compute allocation (different tokens
  requiring genuinely different numbers of experts, not just noise tolerance)

## Novel Contributions (Retained)

1. **Disambiguating the k=1 catastrophe**: k=1 is catastrophic ONLY under
   composition, not single-domain training. Furthermore, the catastrophe requires
   ALL tokens to use k=1 -- a random fraction at k=1 is harmless.
2. **Hard threshold is deployable**: Soft sigmoid training with hard inference
   produces negligible quality gap (<0.1%).
3. **Negative result: entropy adds no value over random at G=8**: This is
   informative for the field -- routing entropy may not be a useful signal for
   adaptive compute allocation at small expert counts.
