# Shared Layer 0 at N=5: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| d | scalar | Embedding dimension (d=64 at micro) |
| P | scalar | Capsules per domain per layer (P=128) |
| D | scalar | Number of domains (D=5 for this experiment) |
| L | scalar | Number of layers (L=4) |
| A_l^k | (P, d) | Layer l detector matrix for domain k |
| B_l^k | (d, P) | Layer l expansion matrix for domain k |
| x | (B, T, d) | Input hidden state |

## Double Counting Magnitude at N=D

At Layer 0, full concatenation produces:

```
y_0^concat = sum_{k=1}^{D} B_0^k @ ReLU(A_0^k @ x)
```

When all D pools produce similar outputs (high co-activation Jaccard J),
this approximately multiplies the Layer 0 contribution by D relative to
what each single-domain model was trained to produce:

```
||y_0^concat|| ~ D * ||y_0^single||     (when J is high)
```

At N=2 (parent experiment): ~2x distortion. Sharing fixes this,
producing ~1x contribution. Net effect: improvement.

At N=5: ~5x distortion. Sharing fixes this, producing ~1x contribution.
But the question is whether the higher layers, which now receive input
from 5 concatenated pools (5*P = 640 capsules) at Layers 1+, are
calibrated for a Layer 0 contribution at 1x vs 5x.

## The Calibration Mismatch Problem

The key insight from the N=5 failure: the deeper layers' concatenated
pools are trained in single-domain context where Layer 0 produces a
single-domain output. When composed at Layers 1+, each domain's deeper
pools still expect Layer 0 to produce that single-domain magnitude.

At N=2:
- Full concat Layer 0: 2x magnitude (double counting)
- Shared Layer 0: 1x magnitude (correct)
- Layers 1+: 2 pools concatenated, each expects 1x from Layer 0
- The 2x distortion from double counting hurts more than the 1x undercount
- Net: sharing wins

At N=5:
- Full concat Layer 0: 5x magnitude (quintuple counting)
- Shared Layer 0: 1x magnitude
- Layers 1+: 5 pools concatenated, each still expects 1x from Layer 0
- BUT: in full concat, the 5x from Layer 0 partially compensates for
  the 5x accumulation from Layers 1+ concatenation. There is a balance
  between the double counting at Layer 0 and the concatenation expansion
  at deeper layers.
- Shared Layer 0 breaks this balance: Layer 0 contributes 1x while
  Layers 1+ contribute 5x. The residual stream is now Layer 0-starved.

## Formal Analysis: Residual Stream Balance

In a residual network, each layer adds to the stream:

```
x_{l+1} = x_l + y_l
```

After L=4 layers in the single-domain model:

```
x_4 = x_0 + y_0 + y_1 + y_2 + y_3
```

The balance between y_0 and sum(y_1..y_3) is learned during training.

After full concatenation at N=5:

```
x_4^concat = x_0 + D*y_0 + D*y_1 + D*y_2 + D*y_3
           = x_0 + D*(y_0 + y_1 + y_2 + y_3)
```

This is "uniformly loud" -- all layers are amplified by D. The layer
ratios are preserved (y_0/y_1 ratio is unchanged).

After shared Layer 0 at N=5:

```
x_4^shared = x_0 + 1*y_0 + D*y_1 + D*y_2 + D*y_3
           = x_0 + y_0 + D*(y_1 + y_2 + y_3)
```

Now y_0 is 1/(D) of what it was in full concat relative to deeper layers.
At D=2: y_0 is 1/2 relative, which is better than 1x (training ratio).
At D=5: y_0 is 1/5 relative, which is much worse than 1x.

## Crossover Point

The crossover occurs when the benefit of eliminating double counting
equals the cost of Layer 0 starvation. Let alpha = y_0 / sum(y_1..y_{L-1}).

At D=1 (single domain): alpha_train = baseline
At D=N (full concat): alpha_concat = alpha_train (ratios preserved)
At D=N (shared L0): alpha_shared = alpha_train / N

The degradation from shared Layer 0 scales as:
```
delta_quality ~ |alpha_shared - alpha_train| = alpha_train * (1 - 1/N)
```

At N=2: delta ~ alpha_train * 0.5 (moderate, offset by double counting fix)
At N=5: delta ~ alpha_train * 0.8 (severe, not offset)

## Parameter Savings

```
Full concat capsule params: L * D * 2*P*d
Shared L0 capsule params:   2*P*d + (L-1)*D*2*P*d

Savings: (D-1)*2*P*d
Savings fraction: (D-1) / (L*D)

At D=5, L=4:  4/20 = 20% of capsule params
At D=5 total: 65,536 / 398,720 = 16.4% of all params
At D=2 total: 16,384 / 202,112 = 8.1% of all params
Ratio: 4.0x more savings at N=5
```

## Layer 0 Cross-Domain Jaccard at N=5

The pairwise Jaccard measures how similar Layer 0 capsule fire patterns
are across domain models when run on the same data.

```
J(A_0^i, A_0^j) = |fire(A_0^i, X) AND fire(A_0^j, X)| / |fire(A_0^i, X) OR fire(A_0^j, X)|
```

At N=2 (parent): J = 0.544 (cross-pool in composed model)
At N=5: Mean pairwise J = 0.853 (domain model Layer 0s on same data)
        Co-activation J = 0.482 (cross-pool in composed model)

The pairwise Jaccard remains very high (0.853), confirming Layer 0
domains-invariance persists. The kill criterion (J < 0.40) is not
triggered. Layer 0 features ARE similar across 5 domains -- the
problem is not that sharing is a bad representation, but that
sharing breaks the magnitude balance.

## Worked Example (d=64, P=4, D=5)

```
Single-domain residual stream (training):
  x = x_0 + y_0 + y_1 + y_2 + y_3
  Let ||y_l|| ~ 1 for each layer
  Total: ||delta|| ~ 4

Full concat (D=5):
  x = x_0 + 5*y_0 + 5*y_1 + 5*y_2 + 5*y_3
  Total: ||delta|| ~ 20
  Layer ratios: y_0/(y_1+y_2+y_3) = 5/15 = 1/3 (same as training)

Shared L0 (D=5):
  x = x_0 + 1*y_0 + 5*y_1 + 5*y_2 + 5*y_3
  Total: ||delta|| ~ 16
  Layer ratios: y_0/(y_1+y_2+y_3) = 1/15 = 1/15 (5x smaller than training)

  Layer 0 is starved: it contributes 6.25% of the total delta
  instead of the expected 25%.
```

## Assumptions

1. **Layer 0 remains domain-invariant at N=5**: CONFIRMED (J=0.853).
   The problem is not representation quality but magnitude balance.

2. **Double counting is the dominant effect at N=2**: CONFIRMED (parent
   experiment improvement). The 2x distortion is harmful.

3. **Layer 0 starvation becomes dominant at N=5**: CONFIRMED (this
   experiment). The 1/5 relative contribution is more harmful than
   the 5x double counting.

4. **Residual stream balance matters for quality**: CONFIRMED.
   Uniformly-scaled full concat (all layers at Dx) preserves ratios
   and works better than selectively-scaled shared L0 (Layer 0 at 1x,
   rest at Dx).
