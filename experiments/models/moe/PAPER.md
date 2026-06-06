# MoE GPT: Research Digest

## What This Model Is

`MoEGPT` is a sparse Mixture-of-Experts transformer. It has the same architecture as the parent `GPT` model in this arena, with one change: every `Block`'s MLP is replaced by a `MoELayer` containing `N` independent expert MLPs and a learned router. Each token selects `top_k` experts per layer; only those experts contribute to the output. The rest of the model — token and position embeddings, causal self-attention, RMSNorm, and the language model head — is unchanged.

The net effect: total parameters scale with `N`, but floating-point operations per token scale with `top_k`. At `N=4, top_k=2`, this model stores 4x the expert weights of the dense baseline but executes only 2x the MLP compute per token.

## Key References

**Shazeer et al. 2017 — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"**
The paper that introduced the modern MoE formulation for deep learning. Proposed top-k gating with a learned router, noted the load imbalance problem, and introduced an auxiliary loss to encourage uniform expert usage. This implementation follows that architecture directly.

**Fedus et al. 2022 — "Switch Transformers: Scaling to Trillion Parameter Models"**
Simplified Shazeer's formulation to `top_k=1` (Switch routing) and showed that `top_k=1` with capacity buffers still works well. Introduced the "Switch" balance loss formulation `L = N * sum(f_i * p_i)` (product of dispatch fraction and mean probability). This implementation uses a simpler quadratic form `N * sum(p_i^2)` which is convex and differentiable without requiring a discrete dispatch fraction.

**Zoph et al. 2022 — "ST-MoE: Designing Stable and Transferable Sparse Expert Models"**
Characterized instability pathways in large MoE models: router z-loss, expert collapse, and the interaction between balance loss coefficient and model capacity. Recommended `top_k=2` over `top_k=1` for quality; this implementation uses `top_k=2` by default.

## Design Choices

### top_k = 2

`top_k=1` (Switch routing) is computationally minimal but brittle: a single misrouted token gets zero contribution from other experts. `top_k=2` provides a redundancy hedge — if the primary expert is a poor fit, the secondary expert provides a correction signal. Empirically, `top_k=2` outperforms `top_k=1` at matched compute budgets (Zoph et al. 2022). Beyond `top_k=2` the gains diminish relative to the increased compute cost.

### Softmax Gating

The router uses standard softmax over `N` expert scores. The alternatives are:
- Sigmoid + normalization: allows multi-modal distributions but adds a normalization step
- Noisy top-k (Shazeer 2017): adds Gaussian noise before top-k to encourage exploration during training

Softmax without noise is the simplest stable choice for small `N`. With only 4 experts the load imbalance problem is less severe than at `N=128+`, so noise-based exploration is unnecessary.

### Balance Loss Coefficient = 0.01

The total loss is `L_CE + 0.01 * sum_layers(L_bal)`. The coefficient controls the trade-off between task performance and routing uniformity:

- Too small: router collapses; one or two experts get all tokens; others are undertrained
- Too large: router is forced toward uniform selection regardless of token content; expert specialization is suppressed and task loss increases

`0.01` is the standard order-of-magnitude choice from Switch Transformers. At micro scale (`d=64, N=4, n_layer=4`) the balance loss values are small in absolute terms, so a larger coefficient might be needed in practice — but `0.01` is the safe default.

### Threshold on Scores, Not Probabilities

Top-k selection uses raw logit scores to identify which experts to activate, then reweights using the corresponding softmax probabilities. If top-k were applied directly to probabilities, the result is the same (softmax is monotone), but applying the threshold to scores and then renormalizing probabilities is numerically cleaner: the probabilities of selected experts always form a valid conditional distribution regardless of the magnitude of `exp(s_i)` for non-selected experts.

## The "Run All Experts" Shortcut

In large MoE systems (`N=64` to `N=2048`), sparse dispatch is essential: only `top_k` of `N` experts are evaluated per token, using scatter/gather operations. At `N=4, top_k=2`, running all 4 experts and then zeroing out 2 via `g_i=0` is cheaper than the gather overhead. This is what the implementation does:

```python
for i, expert in enumerate(self.experts):
    w = masked_probs[..., i:i+1]   # zero for non-selected experts
    out = out + w * expert(x)
```

Non-selected experts compute `expert(x)` but their output is multiplied by `w=0`, so it contributes nothing. The compute cost is `N` expert forward passes regardless of `top_k`, but at `N=4` this is acceptable and avoids the need for sparse indexing operations that are expensive on accelerators with regular memory access patterns (Apple Silicon via MLX in this case).

The crossover point where sparse dispatch wins over dense evaluation depends on hardware and `N`; for MLX on Apple Silicon it is likely above `N=8` to `N=16`.

## Relationship to the Parent GPT Model

`MoEGPT` is registered as a child of `gpt` in the model registry (`@register("moe", parent="gpt")`). It reuses three components directly from `gpt.py`:

- `RMSNorm` — applied before attention and before MoE in each block
- `CausalSelfAttention` — unchanged; attends over the full context with causal mask
- `Block` — imported but not used in `MoEGPT` directly; `MoEBlock` is its replacement

`MoEBlock` mirrors `Block` exactly except `self.mlp` is replaced by `self.moe`. The residual stream structure is identical:

```
x = x + attn(norm1(x))
x = x + moe(norm2(x))       # was: x = x + mlp(norm2(x))
```

The `on_domain_switch` hook is a no-op, consistent with the base GPT. The `aux_loss` method is overridden to accumulate balance losses across all MoE layers and return them scaled by `0.01`.

## Trade-offs

| Property | Dense GPT | MoE GPT (N=4, k=2) |
|---|---|---|
| Parameters (MLP portion) | `8d^2` per layer | `N * 8d^2` per layer (+N*d router) |
| MLP FLOPs per token | `2 * 8d^2` | `top_k * 2 * 8d^2` |
| Active param fraction | 100% | ~56% (at d=64) |
| Expert specialization | None | Possible but not guaranteed |
| Load imbalance risk | None | Requires aux loss |
| Routing overhead | None | `2*d*N` per token per layer |
| Implementation complexity | Low | Moderate |

**More parameters, same-or-less active compute.** This is the fundamental MoE promise: capacity (total knowledge stored) can grow without proportional increase in inference cost.

**Routing is a learned bottleneck.** The router must learn to assign tokens to specialists that can handle them. At small scale (this micro model), the 4 experts may not develop strong specialization — there is too little training data and too few parameters to diverge meaningfully. The balance loss helps prevent collapse but does not guarantee useful specialization.

**Load imbalance can degrade training.** If routing collapses early in training, some experts receive no gradient signal and remain untrained. The balance loss coefficient must be tuned: too weak and collapse happens, too strong and specialization is suppressed. At `N=4` this is a mild concern; at `N=64+` it becomes a primary engineering challenge.
