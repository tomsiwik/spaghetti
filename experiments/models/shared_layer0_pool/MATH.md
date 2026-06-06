# Shared Layer 0 Capsule Pool: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| d | scalar | Embedding dimension (d=64 at micro) |
| P | scalar | Capsules per domain per layer (P=128) |
| D | scalar | Number of domains (D=2 at micro) |
| L | scalar | Number of layers (L=4) |
| A_l^k | (P, d) | Layer l detector matrix for domain k |
| B_l^k | (d, P) | Layer l expansion matrix for domain k |
| x | (B, T, d) | Input hidden state |

## Composition by Concatenation (Baseline)

Standard composition concatenates all domain pools at every layer:

```
Layer l composed output:
  h_l = ReLU([A_l^1; A_l^2; ...; A_l^D] @ x_l)     shape: (B, T, P*D)
  y_l = [B_l^1, B_l^2, ..., B_l^D] @ h_l            shape: (B, T, d)
      = sum_{k=1}^{D} B_l^k @ ReLU(A_l^k @ x_l)
```

This produces the exact sum of individual pool outputs due to ReLU's
independence property: each neuron's activation depends only on its own
detector vector, not on other neurons in the pool.

Total parameters in capsule pools: L * D * 2 * P * d

## Shared Layer 0 Composition

**Key observation** (from Exp behavioral_dedup): Layer 0 capsules show
massive cross-domain co-activation (Jaccard J=0.527 mean, confirmed here
at J=0.544), while deeper layers specialize (J<0.05). This means Layer 0
domain-specific pools learn nearly identical functions.

**Shared Layer 0 protocol**: Use a single capsule pool at Layer 0, with
per-domain concatenation at Layers 1+.

```
Layer 0 (shared):
  h_0 = ReLU(A_0^shared @ x_0)              shape: (B, T, P)
  y_0 = B_0^shared @ h_0                    shape: (B, T, d)

Layer l > 0 (concatenated):
  h_l = ReLU([A_l^1; ...; A_l^D] @ x_l)    shape: (B, T, P*D)
  y_l = [B_l^1, ..., B_l^D] @ h_l          shape: (B, T, d)
```

Total parameters: 2*P*d + (L-1)*D*2*P*d = 2*P*d * (1 + (L-1)*D)

## Parameter Savings

```
Full concat:  L * D * 2*P*d = 4 * 2 * 2*128*64 = 131,072  (capsule params only)
Shared L0:    2*P*d + (L-1)*D*2*P*d = 16,384 + 98,304 = 114,688  (capsule params only)
Savings:      D * 2*P*d - 2*P*d = (D-1)*2*P*d = 16,384  (one full layer's worth)
Savings %:    (D-1) / (L*D) = 1/8 = 12.5% of capsule params
              16,384 / 202,112 = 8.1% of total model params (includes attention, embeddings)
```

At macro scale with D=5 domains and L=24 layers:
- Savings: (D-1)*2*P*d / (L*D*2*P*d) = 4/120 = 3.3% of capsule params
- Absolute savings: 4 * 2 * P * d (four full layers' worth)
- Practical value: more from eliminating redundant fine-tuning than from
  param count reduction

## Sharing Strategies

Three strategies for constructing A_0^shared, B_0^shared:

### 1. Base (pretrained)
```
A_0^shared = A_0^base      (pretrained, before domain fine-tuning)
B_0^shared = B_0^base
```
Uses the generic features learned during pretraining. No domain-specific
adaptation at Layer 0.

### 2. Average
```
A_0^shared = (1/D) * sum_{k=1}^{D} A_0^k
B_0^shared = (1/D) * sum_{k=1}^{D} B_0^k
```
Weight averaging of domain-specific Layer 0 pools. Preserves any
domain-specific learning that is consistent across domains.

Note: this is NOT equivalent to the average of outputs:
  avg_output = (1/D) * sum_k B_0^k @ ReLU(A_0^k @ x)
  shared_output = B_0^avg @ ReLU(A_0^avg @ x)
These differ because ReLU is nonlinear. But at Layer 0 where co-activation
Jaccard is ~0.54, the difference is small because domains fire the same
neurons on the same inputs.

### 3. First (arbitrary domain)
```
A_0^shared = A_0^1
B_0^shared = B_0^1
```
Uses one domain's Layer 0 unchanged. If Layer 0 is truly domain-invariant,
the choice of domain should not matter. Tests the extreme hypothesis.

## Why Sharing Improves Quality

The surprising finding: all shared strategies IMPROVE quality vs full
concatenation. This is explained by the "double counting" problem.

Full concatenation at Layer 0 produces:
```
y_0^concat = B_0^1 @ ReLU(A_0^1 @ x) + B_0^2 @ ReLU(A_0^2 @ x)
```

When A_0^1 and A_0^2 produce similar activations (J=0.54), this
approximately doubles the Layer 0 contribution relative to what each
domain was trained with. This "loudness" distortion propagates to all
subsequent layers.

Shared Layer 0 avoids double counting:
```
y_0^shared = B_0^shared @ ReLU(A_0^shared @ x)
```

This produces a single contribution at the expected magnitude, preserving
the balance between layers that the model was trained with.

**Reconciliation with loudness-falsification (relu_router):** The
relu_router experiment tested whether a *global* learned scalar could close
the composition gap. It could not (learned scales converged to ~0.99),
falsifying the hypothesis that the overall composition gap is a loudness
(global magnitude) problem. The double counting described here is a
*per-layer* magnitude imbalance: Layer 0 contributes at ~2x its trained
magnitude while Layers 1-3 remain at 1x. A single global scalar cannot
correct a per-layer imbalance — it would need to simultaneously halve
Layer 0's contribution while leaving deeper layers unchanged. These are
distinct mechanisms: global loudness (killed) vs layer-specific magnitude
distortion from redundant pools (what we observe here).

## Worked Example (d=64, P=4, D=2)

```
Detector vectors (Layer 0):
  Domain 1: A_0^1 = [[a1, a2, ..., a64],    shape (4, 64)
                      [a65, ...],
                      [a129, ...],
                      [a193, ...]]
  Domain 2: A_0^2 = [[b1, b2, ..., b64],    (similar to A_0^1)
                      [b65, ...],
                      [b129, ...],
                      [b193, ...]]

Full concat: A_concat = [[a1..]; [a65..]; [a129..]; [a193..];
                          [b1..]; [b65..]; [b129..]; [b193..]]
             shape (8, 64) -- 8 capsules

Shared (average): A_avg = [[(a1+b1)/2, ...]; [(a65+b65)/2, ...];
                           [(a129+b129)/2, ...]; [(a193+b193)/2, ...]]
                  shape (4, 64) -- 4 capsules

For input x with ||x|| = 1:
  Full concat fires ~4 capsules (50% ReLU sparsity) from EACH pool
  = ~4 from pool 1 + ~4 from pool 2 = ~8 active neurons
  = approximately 2x the contribution each domain was calibrated for

  Shared fires ~2 capsules (50% sparsity) from single pool
  = contribution magnitude matches single-domain training
```

## Assumptions

1. **Layer 0 learns domain-invariant features**: Supported by J=0.54 from
   behavioral_dedup and confirmed here. This is the feature hierarchy
   principle (Yosinski et al. 2014).

2. **Deeper layers specialize**: Supported by J<0.05 for Layers 1-3.
   Domain-specific knowledge concentrates in later layers.

3. **Double counting harms quality**: Supported by the improvement from
   shared Layer 0 (-1.7% to -3.0% vs full concat). The excess magnitude
   from redundant pools distorts the residual stream.

4. **Weight averaging is a reasonable approximation for Layer 0**: Supported
   by high co-activation Jaccard. When neurons fire on the same inputs,
   their averaged weights produce similar outputs.
