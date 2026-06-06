# Combined Parallel + Pure-Linear: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| x_l | (B, T, d) | Input to layer l |
| d | scalar | Model dimension (64 at micro) |
| h | scalar | Number of attention heads (4) |
| d_h | scalar | Head dimension = d/h (16) |
| L | scalar | Number of layers (4) |
| N | scalar | Number of capsule groups per layer |
| k | scalar | Top-k groups selected per token |
| C | scalar | Capsules per group (64) |

## Architecture: Two Orthogonal Modifications

### Modification 1: Parallel Blocks

Standard sequential transformer block:
```
h_l = x_l + Attn(Norm1(x_l))
x_{l+1} = h_l + CapsulePool(Norm2(h_l))
```

Parallel transformer block (Tiny Aya):
```
x_{l+1} = x_l + Attn(Norm(x_l)) + CapsulePool(Norm(x_l))
```

Key difference: In sequential blocks, the capsule pool operates on
post-attention features Norm2(x_l + Attn(Norm1(x_l))). In parallel blocks,
both branches operate on the same Norm(x_l).

**Composition implication:** In sequential blocks, domain-specific capsule
adapters see features already processed by (shared, potentially interfered)
attention. In parallel blocks, capsule adapters see the raw normalized input,
decoupling them from the attention interference chain.

### Modification 2: Pure-Linear Attention (GatedDeltaNet)

Replace all causal self-attention layers with full GatedDeltaNet linear
attention. The hybrid 3:1 baseline uses GatedDeltaNet for layers 0-2 and
full attention for layer 3.

Per-timestep recurrence (all 6 GatedDeltaNet components):

1. **Conv1d preprocessing:** q, k, v = SiLU(CausalConv1d(W_q x)), etc.
2. **L2 normalization:** q, k = q/||q||, k/||k||
3. **Decay:** g_t = exp(-A * softplus(W_a x_t + b_dt))
4. **Per-dim beta:** beta_t = sigmoid(W_beta x_t), shape (B, h, d_h)
5. **Delta rule:**
   - Retrieve: kv_mem = S^T k_t
   - Correct: delta = (v_t - kv_mem) * beta_t
   - Update: S = g_t * S + k_t * delta^T
6. **Output:** o_t = S^T q_t, gated by RMSNorm(o_t) * SiLU(z_t)

**Composition implication:** GatedDeltaNet's exponential decay gate causes
older key-value associations to fade, naturally limiting cross-domain
interference propagation through the recurrent state.

### Combined Architecture

Each layer l computes:
```
n_l = Norm(x_l)
x_{l+1} = x_l + GDN(n_l) + CapsulePool(n_l)
```

where GDN is the full GatedDeltaNet attention mechanism (no full attention
layers anywhere in the stack).

## Factorial Design

The experiment uses a 2x2 factorial design to isolate and measure interactions:

| | Hybrid 3:1 | Pure-Linear 4:0 |
|---|---|---|
| **Sequential** | A (baseline) | B (linear effect) |
| **Parallel** | C (parallel effect) | D (combined, test) |

If effects are additive (no interaction):
```
effect(D - A) = effect(B - A) + effect(C - A)
```

Or equivalently, the interaction term:
```
I = (D - A) - (B - A) - (C - A) = D - B - C + A
```

should be approximately zero.

## Computational Complexity

### Parameter Count

Sequential hybrid (baseline):
- Per layer: 2 RMSNorm + Attn + CapsulePool
- Layer 0-2 (GDN): d^2 * 6 (Q,K,V,O,A,Beta,Z) + conv1d + norms
- Layer 3 (Full): d^2 * 4 (Q,K,V,O) + norms

Parallel pure-linear (test):
- Per layer: 1 RMSNorm + GDN_Attn + CapsulePool
- All layers (GDN): d^2 * 6 + conv1d
- Saves: one norm per layer, but all layers have GDN (slightly more attn params)

Net difference: ~negligible at micro scale. Parallel saves L norms but
has one more GDN layer than hybrid. At d=64, this is < 5% param difference.

### FLOPs

Sequential per layer: Norm1(x) -> Attn -> Add -> Norm2 -> MLP -> Add
  = 2 * norm_cost + attn_cost + mlp_cost (sequential dependency chain)

Parallel per layer: Norm(x) -> [Attn, MLP] -> Add
  = 1 * norm_cost + max(attn_cost, mlp_cost) (parallel branches)

At micro scale with sequential execution on MLX, the "parallel" benefit
is primarily from simpler graph (1 norm instead of 2), not true hardware
parallelism. At macro scale with proper hardware parallelism, the
speedup could approach:

  speedup = (norm + attn + norm + mlp) / (norm + max(attn, mlp))

For typical transformer where mlp ~= 2/3 FLOPs and attn ~= 1/3 FLOPs:
  speedup ~= (1 + 0.33 + 1 + 0.67) / (1 + 0.67) = 3.0 / 1.67 = 1.8x

In practice, ~30% speedup was measured at micro scale (from the parallel
block experiment).

## Worked Example (d=64, h=4, L=4, N=4, k=2)

1. Input: x of shape (B=32, T=32, 64)
2. Norm: n = RMSNorm(x), shape (32, 32, 64)
3. GDN attention branch:
   - Q, K, V projections: (32, 32, 64) each
   - Conv1d: (32, 32, 64) each
   - Reshape to heads: (32, 32, 4, 16)
   - L2 norm on Q, K
   - Recurrence: S state (32, 4, 16, 16) per timestep
   - Output: (32, 32, 64) after head merge + W_o
4. CapsulePool branch:
   - Router: (32, 32, 64) @ (4, 64)^T -> scores (32, 32, 4)
   - Top-2 selection
   - Per-group: A @ ReLU(B @ n), shapes: B=(64, 64), A=(64, 64)
   - Weighted sum of top-2 group outputs
   - Output: (32, 32, 64)
5. Residual: x + attn_out + mlp_out = (32, 32, 64)
6. Repeat for all 4 layers

## Kill Criterion Formalization

Let L_composed(condition) be the mean composed validation loss across seeds.

Kill: (L_composed(par_pure_linear) - L_composed(seq_hybrid)) / L_composed(seq_hybrid) > 5%

This measures whether the maximally-simplified architecture degrades
composed model quality by more than 5% compared to the validated
sequential+hybrid baseline.

## Assumptions

1. **Independence of modifications:** Parallel block execution and linear
   attention type are orthogonal modifications that do not interact
   destructively. Justified by: they modify different aspects of the
   transformer (norm/residual topology vs attention mechanism).

2. **Composition protocol invariance:** The same compose-and-calibrate
   protocol works for all block/attention combinations. Justified by:
   the protocol operates on capsule pools, which have identical structure
   in all conditions.

3. **Micro-scale validity:** Results at d=64, L=4, T=32 are directionally
   indicative of behavior at macro scale. Assumption, not proven.
