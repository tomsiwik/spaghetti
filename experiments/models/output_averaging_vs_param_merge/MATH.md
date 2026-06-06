# Output-Averaging vs Parameter-Merging: Mathematical Framework

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| W_base | (d_out, d_in) | Base model weight matrix (frozen) |
| A_i | (r, d_in) | LoRA down-projection for adapter i |
| B_i | (d_out, r) | LoRA up-projection for adapter i |
| Delta_i | (d_out, d_in) | = B_i @ A_i, adapter i's weight perturbation |
| x | (1, T, d_in) | Input activation tensor |
| N | scalar | Total number of adapters in the composition |
| k | scalar | Number of adapters selected for composition (k <= N) |
| S | set | Selected adapter indices, |S| = k |
| d | scalar | d_model = 2560 (BitNet-2B-4T) |
| r | scalar | LoRA rank = 16 |
| L | scalar | Number of transformer layers = 30 |
| P | scalar | Number of projection matrices per layer = 7 (q,k,v,o,gate,up,down) |

## Two Composition Strategies

### Strategy 1: Parameter-Merging (Pre-Merge)

Merge adapter weights into the base model, then run a single forward pass:

```
W_merged = W_base + (1/k) * sum_{i in S} (B_i @ A_i)
logits_merge = f(x; W_merged)
```

The merged forward pass is:
```
h_merge = W_merged @ x = W_base @ x + (1/k) * sum_i (B_i @ (A_i @ x))
```

**Cost:** One forward pass. Merge is O(k * d * r) per projection, done once.

**The cross-term problem:** Expanding the full model computation through
nonlinearities, the merged model computes:

```
f(x; W_base + (1/k) * sum_i Delta_i) != (1/k) * sum_i f(x; W_base + Delta_i)
```

The inequality arises because nonlinear activation functions (SiLU, softmax)
do not distribute over addition. The difference is the "cross-term" error:

```
epsilon_cross = f(x; W_base + (1/k)*sum Delta_i) - (1/k)*sum f(x; W_base + Delta_i)
```

Under 1/k scaling, each Delta_i contributes O(1/k) perturbation to W_base.
For small perturbations, Taylor expansion gives:
```
f(W + sum eps_i) approx f(W) + sum f'(W)*eps_i + (1/2) sum_{i,j} f''(W)*eps_i*eps_j + ...
```

The cross-terms are the i != j second-order terms, which scale as O(1/k^2).
At large k, each adapter's signal is O(1/k) and cross-terms are O(1/k^2),
so cross-terms become negligible relative to each adapter's contribution.

### Strategy 2: Output-Averaging (Ensemble)

Run each adapter separately and average the output logits:

```
logits_i = f(x; W_base + Delta_i)     for each i in S
logits_avg = (1/k) * sum_{i in S} logits_i
```

**Cost:** k forward passes through the full model.

**No cross-terms:** Each adapter operates independently on the full base model
(not diluted by 1/k). The averaging happens in output space (logits) where
linear averaging is valid before softmax.

**Key difference from pre-merge:** Each adapter sees W_base + Delta_i (full
adapter contribution), not W_base + (1/k)*Delta_i (diluted contribution).

## When Does Output-Averaging Win?

Output-averaging wins when:
1. Cross-term interference in parameter space is significant
2. 1/k dilution weakens each adapter's signal enough to matter
3. The quality gain outweighs the k-fold compute cost

Output-averaging loses when:
1. Adapters are nearly orthogonal (cross-terms are small)
2. k is large (each adapter's signal is already diluted to O(1/k))
3. The compute budget doesn't allow k forward passes

### Theoretical prediction for our architecture

Our Grassmannian skeleton ensures A_i^T @ A_j approx 0 for i != j.
This means the weight-space cross-terms are already small:

```
||Delta_i^T @ Delta_j|| <= ||B_i|| * ||A_i^T A_j|| * ||B_j|| approx 0
```

If cross-terms are already near zero, parameter-merging should be nearly
equivalent to output-averaging, and the extra compute of output-averaging
is wasted. This experiment tests this prediction.

## Compute Cost Analysis

### Pre-merge
- Merge cost: k * L * P * (d * r + r * d) = k * L * P * 2dr FLOPs (one-time)
- Inference cost: 1 forward pass = C_fwd FLOPs per token
- Total per token: C_fwd (merge amortized over many tokens)

### Output-averaging
- Inference cost: k * C_fwd FLOPs per token
- No merge cost, but k-fold inference cost

### Latency at micro scale (d=128, L=4)
From adapter_inference_speed_mlx results:
- Base forward: 0.71 ms
- Pre-merge (any N): ~0.70 ms (within noise)
- Runtime k=1: 0.94 ms, k=4: 1.86 ms, k=8: 2.33 ms

For output-averaging at full model scale (d=2560, L=30):
- Base forward: ~X ms (to be measured)
- Output-averaging k adapters: ~k * X ms

Kill criterion K2: 200 ms/token threshold for interactive serving.

## Worked Example (d=2560, r=16, k=5)

Pre-merge:
- Merge: 5 * 30 * 7 * 2 * 2560 * 16 = 86M FLOPs (one-time, ~0.1ms)
- Inference: 1 forward pass through 2.4B param model
- Latency: ~X ms (same as base)

Output-averaging k=5:
- No merge needed
- Inference: 5 forward passes
- Latency: ~5X ms

At k=50:
- Output-averaging: 50X ms (likely exceeds 200ms/token threshold)
- Pre-merge: still ~X ms

## PPL Relationship

For the ensemble (output-averaging), the effective probability distribution is:

```
p_ensemble(y|x) = softmax((1/k) * sum_i logits_i(x))
```

Note: averaging logits before softmax != averaging probabilities after softmax.
Logit averaging is a geometric mean of probabilities (up to normalization),
which is sharper than arithmetic mean. This is the standard ensemble method
and is what arxiv 2603.03535 uses.

For pre-merge, the distribution is:
```
p_merge(y|x) = softmax(f(x; W_base + (1/k)*sum Delta_i))
```

The gap is:
```
PPL_merge - PPL_ensemble = f(cross_terms, dilution_effect)
```

If cross-terms are small and dilution is negligible, PPL_merge approx PPL_ensemble.
