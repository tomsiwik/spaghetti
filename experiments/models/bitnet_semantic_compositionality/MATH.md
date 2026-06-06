# Mathematical Foundations: Semantic Compositionality + OSRM Diagnostic

## Notation

| Symbol | Meaning | Shape |
|--------|---------|-------|
| d | Model hidden dimension | scalar (2560) |
| r | LoRA rank | scalar (16) |
| N | Number of adapters | scalar (5) |
| A_i | LoRA down-projection for adapter i | (d, r) |
| B_i | LoRA up-projection for adapter i | (r, d_out) |
| h_i | Hidden state from domain i data | (d,) |
| alpha | LoRA scaling factor | scalar (20.0) |

## Weight-Space vs Data-Space Orthogonality

### Weight-Space (What We Had)

Our Grassmannian skeleton guarantees:

```
cos(vec(DW_i), vec(DW_j)) ~ 0.001
```

where DW_i = alpha * B_i @ A_i is the effective LoRA delta weight.

### Data-Space (What OSRM Measures)

OSRM (arXiv:2505.22934) defines interference as how adapter j
transforms domain i's data:

```
interference(j, i) = ||B_j @ A_j @ h_i|| / ||B_i @ A_i @ h_i||
```

The key insight: weight orthogonality (A_i perp A_j) constrains
A_i^T A_j ~ 0, but data-space orthogonality requires A_j^T h_i ~ 0,
which is a DIFFERENT condition.

### When Do They Coincide?

If hidden states h_i are isotropic (uniformly distributed in direction),
then by the JL-lemma, random projections A_j should satisfy:

```
E[||A_j @ h_i||^2] = ||h_i||^2 * ||A_j||_F^2 / d
```

This is the SAME for all j regardless of i. So:

```
E[||A_j @ h_i|| / ||A_i @ h_i||] ~ ||A_j||_F / ||A_i||_F ~ 1.0
```

The ratio should be approximately 1.0 for RANDOM A matrices on
isotropic data, regardless of weight-space orthogonality.

This is exactly what we observe empirically (mean ratio = 0.86).

### Why This Does NOT Mean Interference Is Bad

The ratio ||A_j @ h_i|| / ||A_i @ h_i|| ~ 1.0 means adapter j
PROJECTS domain i's data into a comparable subspace. But this
projection alone is not interference -- interference requires:

```
true_interference = ||B_j @ A_j @ h_i||
```

If B_j has learned to respond only to domain j features (which
are absent in h_i), then B_j maps the projection to near-zero
even though A_j does not.

### The Full OSRM Measure

```
full_ratio = ||B_j @ A_j @ h_i|| / ||B_i @ A_i @ h_i||
```

Empirically: mean full_ratio = 0.88 (still high). This means B
does NOT provide sufficient filtering. At d=2560 with rank-16,
the A matrices project into 16-dimensional subspaces that substantially
overlap on actual data distributions.

### Dimensional Analysis

For random A matrices in R^{d x r}:

- Each A_i defines an r-dimensional subspace of R^d
- Two random r-dimensional subspaces in R^d have expected
  principal angle cosines of approximately r/d
- At r=16, d=2560: expected overlap ~ 16/2560 = 0.00625
- But hidden states are NOT isotropic -- they concentrate on a
  lower-dimensional manifold

The effective dimensionality of h_i determines the ratio.
If hidden states span only d_eff << d dimensions, then:

```
expected_ratio ~ sqrt(r / d_eff)
```

For our observed ratio of ~0.86, this implies d_eff ~ r / 0.86^2 ~ 22.
This is consistent with the well-known observation that transformer
hidden states have low intrinsic dimensionality.

## Cross-Domain Composition Quality

For composed adapter (i+j) with 1/2 scaling on domain i data:

```
output = base(h) + 0.5 * B_i @ A_i @ h + 0.5 * B_j @ A_j @ h
                    ^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^
                    domain i signal         domain j "noise"
```

The signal-to-noise ratio is:

```
SNR = ||B_i @ A_i @ h_i|| / ||B_j @ A_j @ h_i||
```

From our measurements: SNR ~ 1/0.88 ~ 1.14. This is barely
above 1.0 -- the signal from the correct adapter is only 14%
stronger than the noise from the wrong adapter.

Yet composition WORKS (K1 PASS: 4/5 pairs benefit). Why?

### Resolution: The Interference Is Constructive

The "noise" from adapter j on domain i data is not random noise --
it is a coherent transformation that the base model can partially
use. At 1/2 scaling, each adapter contributes half its normal
effect, and the cross-domain "interference" acts as mild regularization
rather than destructive interference. This is consistent with our
prior finding that composed PPL is often BETTER than 1/N prediction
(constructive cross-domain transfer).

## Computational Complexity

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Hidden state extraction (per sample) | O(d^2 * L) | O(d * L) |
| A-only OSRM diagnostic (all pairs) | O(N^2 * n * d * r) | O(N * d * r) |
| Full BA OSRM diagnostic (all pairs) | O(N^2 * n * d * r + N^2 * n * r * d_out) | O(N * (d*r + r*d_out)) |
| Cross-domain generation eval | O(N^2 * q * T * d^2 * L) | O(d * L) |

Where L = layers (30), n = samples (30), q = queries (10), T = gen tokens (150).

## Worked Example (d=2560, r=16, N=5)

Given: 5 adapters with A_i in R^{2560 x 16}, 30 hidden states per domain.

Self-activation (medical on medical data):
```
||A_med @ h_med|| = 44805 (averaged over 30 samples, 210 layers)
```

Cross-activation (math on medical data):
```
||A_math @ h_med|| = 34554
```

Ratio: 34554 / 44805 = 0.77

This is 7.7x above the 0.1 threshold. Random A matrices on
concentrated hidden states produce high cross-activation by
construction. OSRM-style data-aware initialization would need
to constrain A_j perp Cov(h_i), which requires seeing domain
i's data before training adapter j.

## Assumptions

1. Hidden states extracted from base model (no adapter), consistent with
   routing literature (MoLoRA, X-LoRA)
2. Mean-pooled over sequence positions (loses positional specificity)
3. Last transformer layer used (layer -1 = layer 29)
4. 1/2 scaling for 2-adapter composition (standard 1/N)
5. Instruction-tuned adapters (not NTP), matching prior task eval findings
