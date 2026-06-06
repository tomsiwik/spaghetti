# TTT-Style Embedded Adapter Update — Mathematical Analysis

## Type: Guided Exploration

## Failure Mode
The TTT-style self-supervised update fails to encode factual content in adapter
weights because: (a) the backward pass through attention cannot be eliminated for
LoRA gradient computation, making "zero cost" impossible, or (b) local
self-supervised objectives cannot encode input→output mappings (facts).

## Prior Math

**Test-Time Training (Sun et al., arXiv:2407.04620):**
TTT replaces attention with a learned linear model whose weights are updated
during inference via self-supervised gradient descent. The inner model's weights
serve as "context state" analogous to the KV cache. Key property: for a linear
inner model f(x) = Wx, the gradient ∇_W L has a closed form, enabling updates
without autograd.

**Online Convex Optimization (Zinkevich, 2003):**
Per-token gradient descent achieves regret R_T ≤ O(√T). With T tokens per turn
and 20 turns (~2000 total tokens), the regret per token is O(1/√2000) ≈ 0.022.

**LoRA (Hu et al., arXiv:2106.09685):**
ΔW = BA where B ∈ R^{d×r}, A ∈ R^{r×d}. The LoRA output y = scale · x @ A^T @ B^T
is a LINEAR function of A, B for fixed input x. Gradient has closed form IF the
loss is defined directly on y.

**P6.A0 Baseline (Finding: exp_p6_lingering_adapter_online):**
Rank-4 LoRA, AdamW lr=1e-3, 20 per-turn updates achieves 60% project QA accuracy,
110ms/turn training latency, zero general knowledge degradation.

## Theorem 1 (Gradient Path Necessity)

**Claim.** For a LoRA adapter at layer l in a transformer model, computing the
gradient of a loss L(logits, targets) w.r.t. LoRA parameters (A_l, B_l) requires
backpropagation through layers l+1, ..., L and the language head. No closed-form
shortcut exists.

**Proof.** The model output is logits = g(h_L) where h_L = f_L ∘ ... ∘ f_{l+1}(h_l)
and h_l depends on (A_l, B_l) through the LoRA-modified attention at layer l. By
the chain rule:

  ∂L/∂A_l = (∂L/∂logits) × (∂logits/∂h_L) × ∏_{k=l+1}^{L} (∂h_k/∂h_{k-1}) × (∂h_l/∂A_l)

Each factor ∂h_k/∂h_{k-1} involves the Jacobian of layer k's transformer block,
which includes softmax attention (non-linear, input-dependent). This product cannot
be simplified to a closed form because:

1. Softmax attention: ∂softmax(QK^T/√d)/∂Q depends on the specific Q, K values
2. Non-linear activations: GELU in MLP creates input-dependent Jacobians
3. Layer normalization: RMSNorm Jacobian depends on the input magnitude

Therefore, computing ∂L/∂A_l from a global loss requires sequential
backpropagation through L-l transformer blocks. QED.

**Consequence.** TTT's "zero additional cost" claim relies on the inner model being
a SEPARATE linear layer with a self-supervised loss defined DIRECTLY on its output.
For LoRA adapters embedded in attention (q_proj, o_proj), the gradient path goes
through the attention mechanism, preventing closed-form computation.

## Theorem 2 (Local Loss Information Bound)

**Claim.** A self-supervised loss at layer l that depends only on hidden states at
layer l cannot encode the mapping from input tokens to output tokens (factual
associations).

**Proof (by information flow).** Consider a factual association: "question Q maps
to answer A." The information about this mapping exists in:

1. The input tokens (encoding Q)
2. The output logits (encoding P(A|Q))

At intermediate layer l, the hidden state h_l encodes a compressed representation
of Q. The next-token prediction mapping Q→A is distributed across ALL layers and
the language head. No single layer's hidden state contains the complete Q→A mapping.

Formally: Let I(X; Y) denote mutual information. The data processing inequality
gives: I(h_l; A | Q) ≤ I(logits; A | Q). The hidden state at layer l contains
LESS information about the answer than the output logits. A loss defined on h_l
can recover at most I(h_l; A | Q) bits about the Q→A mapping.

For intermediate layers, much of I(h_l; A | Q) is in the representation of Q
(which tokens are present), not in the specific Q→A association. The association
is created by the attention mechanism across layers, not stored at any single layer.

Therefore, a local loss at layer l cannot fully recover factual associations. QED.

**Caveat.** If the TTT layer REPLACES attention (as in the TTT paper), then the
TTT loss IS the mechanism for creating associations — this theorem doesn't apply.
It only applies to TTT-as-addon where attention already handles associations.

## Theorem 3 (Online Update Convergence with Approximate Cache)

**Claim.** TTT-style per-token updates with stale KV cache (computed with old
weights) converge if the per-step weight change is bounded.

**Proof sketch.** Let W_t be the LoRA weights after processing token t. The KV
cache at position t was computed with weights W_{t-k} for some k ≥ 1. The
approximation error in the attention output is:

  ||attn(Q_t, K_cached, V_cached) - attn(Q_t, K_fresh, V_fresh)|| 
    ≤ ||K_cached - K_fresh|| × ||V||  (Lipschitz bound)
    ≤ ||W_t - W_{t-k}|| × ||x|| × ||V||
    ≤ k × η × G × ||x|| × ||V||

where η is learning rate and G is gradient bound. For small η (≤ 1e-4) and
modest k (≤ 100), this error is O(10⁻²), negligible compared to the attention
output magnitude.

Standard online GD convergence (Zinkevich 2003) still applies with the additional
error term, giving regret:

  R_T ≤ O(√T) + T × O(kηG)

For kηG << 1/√T, the stale cache error is dominated by the standard regret. QED.

## Quantitative Predictions

| Prediction | Value | Basis |
|-----------|-------|-------|
| Self-supervised (all-token loss) project accuracy | ~50-60% | Similar signal to P6.A0 |
| Per-token update improvement over per-turn | 0-10pp | More updates but noisier |
| Per-token training latency overhead | ~10-15ms/token | One fwd+bwd per token |
| Total training time (per-turn, 20 turns) | ~6-7s | Same as P6.A0 |
| K1289 (zero latency) | FAIL | Backward pass required (Theorem 1) |
| General knowledge degradation | <2pp | Same LoRA capacity bound as P6.A0 |

## Behavioral Predictions

1. **Self-supervised loss (all tokens) should match supervised (response-only)**
   because the question tokens provide context that reinforces the answer tokens'
   gradient signal. Both losses train toward the same next-token prediction objective.

2. **Per-token updates will NOT help** because each update is based on a single
   token's loss, which is very noisy. The noise-to-signal ratio scales as O(1)
   per token vs O(1/√T) per turn (T tokens). AdamW's momentum partially
   compensates, but 2000 noisy steps may not outperform 20 clean steps.

3. **K1289 will fail** because LoRA gradients require backprop through attention
   (Theorem 1). The minimum additional cost per training step is one backward
   pass through the model (~5-10ms per token, ~100ms per turn).

## Kill Criteria Mapping

- K1288 (>= 50% recall): Self-supervised with full backprop should achieve
  similar results to P6.A0 (Claim 1 + empirical). Predicted PASS at ~50-60%.
  
- K1289 (zero additional latency): FAIL per Theorem 1. Backward pass adds
  ~100ms per turn minimum. TTT's closed-form gradient trick does not apply
  to LoRA in attention (non-linear gradient path).

- K1290 (matches P6.A0 = 60%): Marginal. Self-supervised might slightly
  underperform due to including prompt tokens in loss. Predicted 50-60%.

## What Would Kill This

1. **Self-supervised loss significantly worse than supervised**: If all-token loss
   produces <30% accuracy while response-only produces 60%, the supervised signal
   from QA formatting is essential. This would mean TTT-style passive learning
   fundamentally cannot match explicit training.

2. **Per-token updates diverge or oscillate**: If noisy per-token gradients cause
   the LoRA to oscillate instead of converge, the online approach fails. Manifests
   as flat or increasing loss.

3. **Stale KV cache causes quality degradation**: If approximate TTT (reusing old
   cache) produces garbage outputs, the per-token approach is infeasible without
   full recomputation (O(T²) cost).
