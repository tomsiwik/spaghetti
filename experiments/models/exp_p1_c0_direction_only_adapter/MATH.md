# C0.2: Direction-Only Adapter on Gemma 4 (QKV-Norm Thesis)

## Motivation

Gemma 4 applies RMSNorm to all three attention projections:
- Q: RMSNorm with learned scale (q_norm)
- K: RMSNorm with learned scale (k_norm)
- V: RMSNorm WITHOUT learned scale (v_norm)

RMSNorm has re-scaling invariance: for any scalar alpha > 0,
  RMSNorm(alpha * x) = RMSNorm(x)

This means when LoRA modifies q_proj:
  Q = q_norm((W_q + scale * B @ A) @ x)

The magnitude of the LoRA perturbation (controlled by scale, ||B||, ||A||) is
DISCARDED by q_norm. Only the DIRECTION of (W_q + scale * B @ A) @ x survives.

**Hypothesis:** If magnitude is discarded, then a direction-only adapter (B columns
projected to unit norm) should match standard LoRA quality on Gemma 4.

This has profound implications:
1. LoRA's scale hyperparameter is irrelevant on normalized architectures
2. PoLAR's Stiefel constraint (direction-preserving) is geometrically natural
3. Standard LoRA wastes capacity on magnitude that gets killed

## Type: Guided Exploration

The normalization-kills-magnitude claim is proven (RMSNorm algebra). The unknown
is whether direction-only training CONVERGES as well as standard LoRA.

## Theorem 1: RMSNorm Discards LoRA Magnitude

**Statement.** Let W_q be a frozen weight matrix, delta_W = s * B @ A a LoRA
perturbation with scale s. For any input x:

  RMSNorm((W_q + delta_W) @ x) = RMSNorm((W_q + delta_W/s) @ x * s)
                                = RMSNorm((W_q + delta_W/s) @ x)  (re-scaling invariance)

More precisely, RMSNorm(v) = v / RMS(v) where RMS(v) = sqrt(mean(v^2)).
Therefore ||RMSNorm(v)|| = sqrt(d) regardless of ||v||.

**Proof.** For v in R^d:
  RMS(v) = sqrt(sum(v_i^2) / d)
  ||RMSNorm(v)|| = ||v / RMS(v)|| = ||v|| / (||v|| / sqrt(d)) = sqrt(d)

The output norm is always sqrt(d), independent of the input norm. QED.

**Consequence for LoRA:** The post-normalization representation depends only on
the DIRECTION of (W_q + delta_W) @ x, not its magnitude. Two LoRA adapters
with B_1 = alpha * B_2 (parallel B-matrices with different norms) produce
identical outputs after normalization.

## Theorem 2: Direction-Only Training Has Non-Zero Gradients

**Statement.** Let B in R^{r x d_out} with unit-norm rows (||B_i|| = 1 for all i).
The gradient of the loss L with respect to B, projected onto the tangent space
of the unit sphere, is non-zero whenever the Euclidean gradient is not parallel
to B_i.

**Proof.** The tangent space of S^{d-1} at point b is {v : v^T b = 0}.
The projected gradient is:

  grad_tangent = grad_euclidean - (grad_euclidean^T b) * b

This is zero iff grad_euclidean is parallel to b, i.e., grad_euclidean = lambda * b
for some scalar lambda. For a non-trivial loss on a non-trivial task, this
alignment occurs with probability zero under continuous distributions. QED.

**Prediction:** Training loss will decrease monotonically after warmup (first ~50 steps),
though convergence may be slightly slower than standard LoRA due to the constraint.

## Theorem 3: Unit-Norm B Prevents Stable Rank Collapse

**Statement.** If all rows of B have unit norm, then:
  sr(B) = ||B||_F^2 / ||B||_2^2 = r / ||B||_2^2

Since ||B||_2 <= 1 (each row unit norm, so spectral norm <= sqrt(r)),
and ||B||_F^2 = r (r unit-norm rows):
  sr(B) >= r / r = 1

But more importantly, if the rows of B are NOT all parallel (which training
enforces for a non-trivial multi-token task), then ||B||_2 < sqrt(r) and sr(B) > 1.

For standard LoRA, rows can have wildly different norms, allowing a single large-norm
row to dominate (sr -> 1). Unit-norm constraint prevents this.

**Prediction:** sr(delta_W) for direction-only adapter > sr(delta_W) for standard LoRA.

## Kill Criteria

| K | Criterion | Prediction | Grounding |
|---|-----------|------------|-----------|
| KC05 | Direction-only GSM8K >= 90% of standard LoRA | >= 95% (magnitude irrelevant post-norm) | Theorem 1 |
| KC06 | Training loss decreases monotonically after step 50 | Yes (non-zero tangent gradient) | Theorem 2 |

## Measurements (beyond kill criteria)

- B column norms during standard LoRA training (do they grow unboundedly?)
- Stable rank comparison: sr(delta_W) for direction-only vs standard
- Per-step loss curves overlaid for both methods
- Scale sweep: if direction-only, scale={1,5,10,20} should give IDENTICAL accuracy

## Implementation

**Phase 1:** Standard LoRA baseline — already done (T2.1: 82% GSM8K).

**Phase 2:** Direction-constrained LoRA on Gemma 4 E4B.
Custom training loop (cannot use mlx_lm.lora subprocess because we need to
project B to unit norm after each optimizer step):

```
for step in range(1000):
    loss = forward(model, batch)
    loss.backward()
    optimizer.update()
    # Unit-norm projection:
    for layer in adapted_layers:
        B = layer.lora_b  # shape (r, d_out) or (d_out, r)
        norms = norm(B, axis=appropriate_axis, keepdims=True)
        layer.lora_b = B / (norms + 1e-8)
```

**Phase 3:** Evaluate GSM8K, measure stable rank, run scale sweep.
