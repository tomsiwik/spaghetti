# Gap Causal Mechanism: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| W_base | (d, d') | Frozen base model weight matrix |
| Delta_i | (d, d') | LoRA delta for expert i |
| x | (B, T, d) | Input hidden states |
| w_i(x) | scalar | Router weight for expert i on token x |
| f_i(x) | (V,) | Expert i's output logits for token x |
| f_comp(x) | (V,) | Composed model output (uniform average of experts) |
| f_joint(x) | (V,) | Jointly-trained model output |
| G_CE | scalar | CE gap: |CE(f_comp) - CE(f_joint)| |
| D_ij(x) | scalar | Expert discriminability: |f_i(x) - f_j(x)| |
| g_router | (d, N) | Router weight gradient (one column per expert) |
| cos(i,j) | scalar | Cosine similarity of flattened Delta_i, Delta_j |

## The Hypothesis (Original)

**Claim:** Gap magnitude G drives router gradient magnitude ||g_router||.

The parent experiment (gap_as_signal) showed r^2=0.74 correlation between G_CE
and calibration quality. The claim was: larger gap -> stronger gradient signal
-> faster/better calibration.

## The Actual Mechanism

### Router Gradient Derivation

The router produces softmax weights over N experts:

```
w(x) = softmax(R @ h(x))     R in R^{N x d}, h(x) in R^d
```

The loss is:

```
L = CE(sum_i w_i(x) * f_i(x), target)
```

The gradient of L w.r.t. router parameters R is:

```
dL/dR_j = sum_t dL/dw_j(x_t) * dw_j/dR_j

dL/dw_j(x_t) = f_j(x_t) @ d(CE)/d(logits)
```

The key term is `dL/dw_j(x_t)`: the "usefulness" of routing to expert j for
token t. This depends on how expert j's output differs from the current
weighted combination.

### Discriminability vs Gap

There are two distinct quantities:

1. **Function-space gap** G = ||f_comp - f_joint||: measures how far the
   composed model is from joint. INCREASES with cosine (correlated experts
   produce poor composition).

2. **Expert discriminability** D = ||f_A(x) - f_B(x)||: measures how
   different the individual expert outputs are. DECREASES with cosine
   (correlated experts produce similar outputs).

The router gradient magnitude is proportional to D, NOT G:

```
||g_router|| ~ E_x[||f_A(x) - f_B(x)||]    (expert discriminability)
```

This is because the router must distinguish between experts. When
f_A(x) approx f_B(x), the gradient w.r.t. routing weights vanishes
regardless of how far the composition is from the joint model.

### Why the Correlation is Negative

At cos=0.0 (orthogonal experts):
- Delta_A and Delta_B modify orthogonal subspaces
- f_A(x) != f_B(x) for most tokens -> D is LARGE
- f_comp = avg(f_A, f_B) is close to f_joint -> G is SMALL
- Router gradient is LARGE (strong discriminability signal)

At cos=0.9 (correlated experts):
- Delta_A and Delta_B modify nearly the same subspace
- f_A(x) approx f_B(x) for all tokens -> D is SMALL
- f_comp diverges from f_joint -> G is LARGE
- Router gradient is SMALL (no discriminability signal)

Therefore: G and ||g_router|| are INVERSELY related. The parent
experiment's correlation was correct (orthogonal -> better quality)
but the mechanism is through discriminability, not through gap magnitude.

## Formal Prediction (Corrected)

```
cos(i,j) -> D(i,j): monotonically DECREASING
cos(i,j) -> ||g_router||: monotonically DECREASING
cos(i,j) -> quality gap: monotonically INCREASING
```

The causal chain:
```
cos_low -> high_D -> large_grad -> fast_learning -> good_quality
cos_high -> low_D -> small_grad -> slow_learning -> poor_quality
```

## Empirical Results (Mean Curve, 3 Seeds)

| Cosine | ||g_router|| | Gradient/0.9 ratio |
|--------|--------------|--------------------|
| 0.0    | 0.1679       | 15.5x              |
| 0.1    | 0.1530       | 14.2x              |
| 0.2    | 0.1790       | 16.6x              |
| 0.3    | 0.1751       | 16.2x              |
| 0.5    | 0.1957       | 18.1x              |
| 0.7    | 0.0524       | 4.8x               |
| 0.9    | 0.0108       | 1.0x               |

Pearson correlations on mean curve (7 points):
- r^2(cos, ||g_router||) = 0.63
- r^2(||g_router||, quality) = 0.75

Phase transition between cos=0.5 and cos=0.7:
- Regime A (cos <= 0.5): mean ||g_router|| = 0.174, CoV = 0.090
- Regime B (cos >= 0.7): mean ||g_router|| = 0.032, CoV = 0.931
- Ratio: 5.5x

## Computational Cost

Same as parent (gap_as_signal) plus gradient norm extraction:
- O(1) additional cost per step (gradient norms computed from existing gradients)
- No additional forward/backward passes
- Total experiment: ~150s for 3 seeds x 7 cosine levels

## Assumptions

1. **Top-k=2 with N=2 collapses to weight-learning.** The router always
   uses both experts; only the mixing ratio changes. The gradient measures
   the sensitivity of the loss to the mixing ratio. At N>2, the gradient
   would also reflect expert SELECTION, a harder problem.

2. **Gram-Schmidt projection creates synthetic experts.** Real correlated
   experts (trained on overlapping data) may produce different gradient
   patterns than geometrically projected ones.

3. **Per-seed variance is high.** Different seeds concentrate gradient
   norms in different layers (seed 42: Layer 3 dominates; seed 123:
   Layer 0 dominates; seed 7: Layer 1 dominates). The total gradient
   norm is stable, but per-layer analysis requires more seeds.
