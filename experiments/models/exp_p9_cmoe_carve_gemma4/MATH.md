# CMoE Carving: Dense FFN → Shared + Routed Experts

## Prior Work
CMoE (arXiv:2502.04416): training-free expert extraction from dense FFN via
activation statistics. Key insight: in GeGLU networks, neuron activation
patterns on calibration data reveal natural expert groupings.

## Setup

Gemma 4 E4B dense MLP per layer:
- gate_proj: R^{2560} → R^{10240}, up_proj: R^{2560} → R^{10240}, down_proj: R^{10240} → R^{2560}
- Activation: GeGLU: h_i = gelu(gate_i(x)) · up_i(x), output = down(h)

## Theorem 1 (Exact Decomposition)

**Statement.** Let h = [h_1, ..., h_D] ∈ R^D (D=10240) be the intermediate
activation vector. Let {G_0, G_1, ..., G_K} be any partition of {1,...,D}
into disjoint groups. Define:

  y_g = W_down[:, G_g] · h[G_g]   (slice of down_proj acting on group g)

Then:

  y_dense = Σ_{g=0}^{K} y_g     (exact, no approximation)

**Proof.** Linear algebra: down_proj(h) = W_down · h = Σ_g W_down[:, G_g] · h[G_g]
since the groups partition the column indices. QED.

**Corollary.** Carving introduces zero error when all experts are active.
Error arises only from the routing approximation (activating k < K groups).

## Theorem 2 (Routing Error Bound)

**Statement.** Let S ⊂ {1,...,K} be the set of activated routed expert groups
(|S| = k), plus group G_0 (shared, always active). The approximation error is:

  ||y_dense - y_approx|| = ||Σ_{g ∉ S, g≠0} y_g||

**Bound.** If neurons are clustered such that high-activation-rate neurons
are in G_0 (shared) and remaining neurons are grouped by activation pattern
similarity (k-means on binary activation markers), then for a calibration
input x:

  ||error|| ≤ Σ_{g ∉ S} ||W_down[:, G_g]||_F · ||h[G_g]||

The k-means clustering minimizes co-activation mismatch: neurons that
activate together are in the same group, so inactive groups contain
neurons with near-zero activations for the current input.

**Prediction:** With 8 experts (1 shared + 7 routed), activating 1+3=4 groups
(50% activation), perplexity degradation ≤ 5% on calibration distribution.

## Theorem 3 (Speedup from Sparsity)

**Statement.** Dense FFN FLOPs per token: 3 × 2 × d × D = 6dD (three projections).
With N experts of size D/N each, activating k_total = n_shared + n_activated:

  MoE FLOPs = k_total × 3 × 2 × d × (D/N) + routing overhead
            = (k_total/N) × 6dD + O(d × N_routed)

**Prediction:** At 50% activation (k_total/N = 4/8):
- Theoretical speedup: 2.0x
- After routing overhead + multiple kernel launches: ≥ 1.3x on MLX
- Routing overhead: one d×(N-1) matmul ≈ 2560×7 = negligible vs 6×2560×10240

## Kill Criteria Predictions

| Kill | Prediction | Reasoning |
|------|-----------|-----------|
| K1342: PPL ≤ 5% degradation | 1-3% | Shared experts capture high-frequency neurons; k-means groups co-activating neurons |
| K1343: Carving < 10 min | 3-5 min | 42 layers × (100 samples forward + k-means on 10240 neurons). M5 Pro GPU handles this |
| K1344: ≥ 1.3x speedup at 50% | 1.4-1.8x | 50% compute reduction, minus routing + kernel launch overhead |

## Design

- N_experts = 8, N_shared = 1, N_activated = 3 (50% activation)
- Calibration: 100 samples from wikitext (1024 tokens each)
- K_act = 128 (top-128 activation markers per sample, ~1.25% of 10240)
- Balanced k-means via linear assignment (scipy)
- Router: initialized from representative neuron weights per cluster
