# Shared Layer 0 Calibrated: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| d | scalar | Embedding dimension (d=64 at micro) |
| P | scalar | Capsules per domain per layer (P=128) |
| D | scalar | Number of domains (D=2 at micro) |
| L | scalar | Number of layers (L=4) |
| S_cal | scalar | Calibration steps (S_cal=200) |
| A_l^k | (P, d) | Layer l detector matrix for domain k |
| B_l^k | (d, P) | Layer l expansion matrix for domain k |
| x | (B, T, d) | Input hidden state |
| theta_cal | subset | Trainable parameters during calibration (capsule pool weights only) |

## Setup

Two composition protocols are compared, each followed by calibration:

### Full Concatenation + Calibration

All L layers concatenate domain capsule pools, then calibrate:

```
Before calibration (layer l):
  h_l = ReLU([A_l^1; A_l^2] @ x_l)          shape: (B, T, 2P)
  y_l = [B_l^1, B_l^2] @ h_l                shape: (B, T, d)

Calibration objective:
  theta* = argmin_{theta_cal} E_{x ~ D_mixed} [ L_NTP(x; theta) ]

where D_mixed alternates batches from each domain.
```

Total capsule parameters: L * D * 2 * P * d = 4 * 2 * 2 * 128 * 64 = 131,072

### Shared Layer 0 + Calibration

Layer 0 uses a single shared pool, layers 1+ concatenate, then calibrate:

```
Before calibration:
  Layer 0 (shared):     y_0 = B_0^avg @ ReLU(A_0^avg @ x_0)
  Layer l > 0 (concat): y_l = [B_l^1, B_l^2] @ ReLU([A_l^1; A_l^2] @ x_l)

Same calibration objective applied.
```

Total capsule parameters: 2*P*d + (L-1)*D*2*P*d = 16,384 + 98,304 = 114,688

## The Calibration Hypothesis

The parent experiment (shared_layer0_pool) showed shared L0 improves
quality by 1.7-3.0% over full concat in zero-shot composition. The
explanation was "double counting": concatenating redundant Layer 0
pools approximately doubles Layer 0's contribution magnitude.

The calibration hypothesis: 200 steps of MLP weight updates can absorb
this per-layer magnitude distortion. Calibration has access to the full
gradient signal and can learn to:

1. Downscale Layer 0 B weights in the full-concat model to compensate
   for double counting
2. Redistribute activation patterns across layers to correct the
   residual stream balance

If calibration absorbs the distortion, the shared L0 advantage
disappears and sharing becomes a parameter-saving convenience rather
than a quality improvement.

## Calibration Protocol

Calibration trains only capsule pool weights (A and B matrices in all
layers) on mixed-domain data:

```
freeze: embeddings, attention, layer norms, lm_head
unfreeze: capsule_pool.A, capsule_pool.B for all layers

for step in 1..S_cal:
  domain = step % D
  batch ~ D_domain
  loss = L_NTP(batch; model)
  theta_cal -= lr * grad(loss, theta_cal)
```

This allows the MLP weights to adapt to the composed configuration
while keeping the shared attention structure fixed.

## Convergence Analysis

The loss curves show that both models converge to similar quality by
~step 100 of calibration. The key dynamics:

```
Step 1:   shared_L0 slightly better (double counting absent)
Step ~50: curves cross (full concat has corrected Layer 0 distortion)
Step 100+: both models within noise (~0.1% of each other)
Step 200: final gap is +0.09% (within noise, not significant)
```

This pattern is consistent with the calibration hypothesis: the
initial advantage from avoiding double counting is absorbed as the
full-concat model's gradients correct the Layer 0 magnitude.

## Degrees of Freedom

Full concat has MORE trainable parameters during calibration:
- Full concat capsule params: 131,072 (all layers concatenated)
- Shared L0 capsule params: 114,688 (Layer 0 has P, not 2P)

Full concat can represent strictly more functions during calibration
because its Layer 0 has 2P capsules vs P for shared L0. This extra
capacity may explain why it catches up: calibration unlocks the
expressiveness of having 2P capsules at Layer 0 while learning to
avoid the double-counting distortion.

## Worked Example (d=64, P=4, D=2, S_cal=200)

```
Full concat Layer 0 at step 0:
  A_concat = [A^1; A^2]     shape (8, 64)
  h = ReLU(A_concat @ x)    ~4 capsules fire (50% sparsity)
  y = B_concat @ h           magnitude ~2x single-domain training

  Gradient signal: loss is high because magnitude is distorted
  -> gradients push B_concat Layer 0 weights toward smaller values
  -> after ~100 steps, B_concat has adapted to produce ~1x magnitude

Shared L0 at step 0:
  A_avg = (A^1 + A^2)/2     shape (4, 64)
  h = ReLU(A_avg @ x)       ~2 capsules fire (50% sparsity)
  y = B_avg @ h              magnitude ~1x (correct from start)

  Gradient signal: loss is already low, small corrections needed
  -> B_avg weights change little during calibration
```

Result: both models converge to the same region after calibration,
because calibration can correct the double-counting distortion that
sharing avoids structurally.

## Computational Cost

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Full concat forward (Layer 0) | O(B*T*2P*d) | O(B*T*2P) |
| Shared L0 forward (Layer 0) | O(B*T*P*d) | O(B*T*P) |
| Calibration (S_cal steps) | O(S_cal * B * T * sum_l(P_l * d)) | O(optimizer state) |

Calibration cost is the dominant factor in this experiment: 200 steps
of forward+backward through the full model. This dwarfs the forward-
pass savings from Layer 0 sharing at d=64.

## Assumptions

1. **Calibration can correct per-layer magnitude distortion**: CONFIRMED.
   200 steps suffices to absorb the double-counting effect. Convergence
   occurs by ~step 100.

2. **Mixed-domain calibration provides balanced gradient signal**:
   Alternating batches ensures both domains contribute equally. With
   imbalanced domains, calibration might favor one domain.

3. **MLP-only calibration is sufficient**: Attention weights are frozen.
   If attention also needed adaptation, the comparison might differ.

4. **200 steps is sufficient calibration budget**: The loss curves show
   convergence, supporting this. Longer calibration is unlikely to
   change the conclusion.
