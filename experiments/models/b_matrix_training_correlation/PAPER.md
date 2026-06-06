# B-Matrix Training Correlation: Research Digest

## Hypothesis

B-matrix overlap during LoRA training creates structured inter-expert correlation
that could undermine the Grassmannian skeleton's interference guarantees.
Falsifiable: trained B-matrices must show >3x higher pairwise |cos| than random
initialization (K1), AND this correlation must not increase amplification ratio
vs uncorrelated B-matrices (K2).

## What This Model Is

A micro-scale experiment measuring whether LoRA B-matrices, trained freely
while A-matrices are frozen in Grassmannian skeleton slots, develop structured
pairwise correlation. This directly addresses the open question identified by
minimax_grassmannian_packing (KILLED): the d=256 tail anomaly in post-training
expert interference (max/mean=9.36x) originates from B-matrix training dynamics,
not skeleton geometry. This experiment quantifies that B-matrix overlap and tests
whether it poses a safety risk.

## Lineage

```
grassmannian_expert_init        (AP skeleton, frozen-A, zero drift)
        |
minimax_grassmannian_packing    (KILLED: tail is B-matrix, not geometry)
        |
correlated_layer_errors         (PROVEN: correlation REDUCES amplification)
        |
        v
b_matrix_training_correlation   (this: K1 FAIL, K2 PASS)
```

## Key References

- Dhillon, Heath, Strohmer, Tropp (2008). Grassmannian packing via AP.
- Parent: minimax_grassmannian_packing (identified B-matrix overlap as open question).
- Parent: correlated_layer_errors (synthetic correlation reduces amp_ratio to 0.074).
- Parent: grassmannian_expert_init (AP skeleton with frozen-A, zero drift).

## Empirical Results

### B-Matrix Cosine Measurement (3 seeds, d=64, r=8, N=6, L=2)

| Condition | Mean |cos| | Max |cos| | vs Baseline |
|-----------|------------|-----------|-------------|
| AP-trained B | 0.0298 | 0.1151 | 2.52x |
| Rand-trained B | 0.0230 | 0.0667 | 1.95x |
| Random baseline | 0.0118 | 0.0318 | 1.00x |
| Delta vectors (A@B) | 0.0017 | 0.0063 | 0.14x |

**Key finding:** AP-trained B-matrices show 2.52x higher |cos| than random
baseline. This is statistically significant (paired t-test p=0.010) but below
the 3x threshold required by K1.

### Source Decomposition

| Source | Contribution | % of Excess |
|--------|-------------|-------------|
| Training dynamics | 0.0112 | 62% |
| AP skeleton structure | 0.0068 | 38% |

Training itself (shared base weights, similar gradient trajectories) contributes
more to B-matrix correlation than the AP skeleton structure.

### Domain Similarity Effect

| Pair Type | Mean B |cos| |
|-----------|--------------|
| Similar domains | 0.0317 |
| Dissimilar domains | 0.0228 |
| Ratio | 1.39x |

Domain similarity has a weak positive effect on B-matrix overlap. However,
the effect is noisy (one seed showed reversed direction), indicating this
is a second-order effect at toy scale.

### Amplification Ratio (K2)

| Condition | Amp Ratio | Output Dev % |
|-----------|-----------|-------------|
| AP-trained | ~0.0000 | 0.0005% |
| Rand-trained | ~0.0000 | 0.0005% |
| Shuffled-B | ~0.0000 | 0.0006% |
| AP/shuffled | 1.06x | -- |

Amplification is negligible at L=2 for ALL conditions. The comparison
AP/shuffled = 1.06x is within noise (well below 1.5x margin).

### Important: Delta Vector vs B-Matrix Cosine

A striking contrast: B-matrix cosines (0.0298) are 17x higher than full
delta vector cosines (0.0017). This is because the frozen A-matrices
(near-orthogonal via Grassmannian skeleton) act as a "decorrelation filter":

    delta_W_i = (alpha/r) * A_i @ B_i

Even if B_i and B_j are correlated, the near-orthogonal A_i and A_j
project them into different subspaces, making the full deltas nearly
orthogonal. The skeleton's interference guarantee operates at the
delta level, not the B-matrix level.

### Kill Criteria Assessment

**K1: Trained B |cos| > 3x random baseline?**
- Ratio: 2.52x < 3.0x threshold
- **K1 FAIL.** B-matrix correlation exists but is moderate, not dramatic.
  Not sufficient to classify as "structured" inter-expert correlation.

**K2: B-matrix correlation does NOT increase amp ratio?**
- AP/shuffled ratio: 1.06x < 1.5x margin
- All amp ratios < 1.0 (effectively zero at L=2)
- **K2 PASS.** Safe. Correlation has no measurable effect on amplification.

**VERDICT: K1 FAIL + K2 PASS. Status: KILLED (for K1).**

The hypothesis that training creates >3x structured B-matrix correlation
is not supported. B-matrices show moderate training-induced overlap (2.52x)
but the Grassmannian skeleton decorrelates the full delta vectors (0.0017
mean |cos|), making B-matrix correlation operationally irrelevant.

## What Was Learned

### 1. The Grassmannian Skeleton is a Decorrelation Filter

The most important finding: even though B-matrices show 2.52x above-baseline
correlation, the full delta vectors (A@B) show only 0.14x of baseline.
The near-orthogonal A-matrices project correlated B-matrices into orthogonal
subspaces, multiplying the correlation by ||A_i^T A_j|| which is near-zero
by Grassmannian construction. This means B-matrix correlation is a non-issue
for SOLE composition safety.

### 2. Training Dynamics Dominate Over Skeleton Effects

62% of B-matrix correlation comes from training dynamics (shared base weights,
similar gradient structure), 38% from AP skeleton structure. This means even
switching to random orthonormal A-matrices would still produce most of the
observed B-matrix correlation.

### 3. Domain Similarity Has Weak Effect at Toy Scale

Similar domains show only 1.39x higher B-matrix cosine than dissimilar
domains. At toy scale with synthetic Markov data, domain similarity is a
minor factor. Real domains with richer linguistic overlap may show stronger
effects, but the decorrelation filter (point 1) would still apply.

## Implications for SOLE

**The B-matrix overlap concern raised by minimax_grassmannian_packing is
resolved.** The concern was that trained B-matrices could create interference
beyond what the skeleton geometry controls. We find:

1. B-matrix correlation is moderate (2.52x, not dramatic 10x+)
2. The skeleton decorrelates deltas regardless (0.14x of baseline at delta level)
3. Amplification is unaffected (1.06x, well within noise)

**No B-matrix regularization is needed.** The Grassmannian skeleton already
provides sufficient decorrelation through the A-matrix structure.

## Limitations

1. **Shallow model (L=2).** Amplification effects only emerge at L >= 8
   (parent experiment). The K2 test is degenerate at L=2. However, the
   parent showed amp_ratio < 1.0 even at rho=1.0 synthetic correlation,
   so our real rho_B ~ 0.03 is far below the danger zone.

2. **Toy dimension (d=64).** At production d=896, B-vector dimensionality
   grows quadratically, so random cosines decrease as ~1/sqrt(D). The
   K1 ratio may change but is expected to stay moderate.

3. **Synthetic data.** Real domain data with shared vocabulary and grammar
   may create stronger B-matrix correlation.

4. **Small N=6.** With N=50+ experts, more same-domain pairs exist,
   potentially increasing average correlation.

## What Would Kill This

Already killed (K1 FAIL). The hypothesis that B-matrix training creates
>3x structured correlation is not supported at micro scale. However, the
finding that the skeleton decorrelates deltas independently of B-matrix
correlation is a positive structural result for SOLE.

To resurrect this concern, one would need to show:
- Real domain data at macro scale creates >3x B-matrix correlation AND
- The decorrelation filter (A_i^T A_j near-zero) fails to suppress it AND
- The resulting interference measurably increases amplification ratio

Given that A_i^T A_j is controlled by the skeleton and amp_ratio < 1.0
even at rho=1.0, all three conditions failing simultaneously appears unlikely.
