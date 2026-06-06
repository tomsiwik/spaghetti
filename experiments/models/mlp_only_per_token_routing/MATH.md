# MLP-Only Per-Token Routing: Eliminating Cross-Attention Contamination

## Type: Guided Exploration (Type 2)

The proven framework is the MoE architecture principle (shared attention + routed FFN).
The unknown is whether MLP-only LoRA per-token adaptation provides sufficient domain
signal on our specific ternary base model with these specific adapters.

**EXPERIMENT-PROOF GAP:** This proof addresses single-pass mixed-adapter MLP-only
routing. The experiment implements multi-pass oracle selection (5 separate forward
passes, per-token NLL compositing). The contamination mechanism is circumvented by
the multi-pass methodology, not tested. See PAPER.md for full discussion.

## A. Failure Mode Identification

**Disease:** Cross-attention contamination in per-token routed adapter composition.

When LoRA adapters modify attention layers (Q, K, V, O projections) per-token,
each token's K/V projections are computed with its domain's adapter. Under causal
self-attention, token t attends to ALL preceding tokens' K/V. If tokens 0..127 used
adapter A (domain medical) and tokens 128..255 used adapter B (domain code), then
token 255 attends to K/V projections computed via adapter A -- receiving contaminated
representations from the wrong domain.

**Finding #305 showed** per-token full-module routing produces IDENTICAL PPL to
per-sequence routing (both 4.815). This was originally attributed to cross-attention
contamination. However, this experiment's multi-pass methodology (separate forward
passes per adapter, per-token NLL selection) shows per_token_full = 4.500 < per_seq
= 4.815, revealing Finding #305's null was a **methodological artifact**: its
single-adapter-per-pass approach could not differentiate per-token from per-sequence.
The contamination hypothesis remains **untested** — it was circumvented, not confirmed.

**Root cause:** The self-attention operation mixes information across tokens.
If different tokens contribute adapted K/V from different domains, the attention
output for each token is a contaminated mixture. This is a fundamental property
of the attention mechanism, not a training or optimization issue.

## B. The Right Question (Reframe)

Wrong: "How do we prevent cross-attention contamination?"
Right: "What is the architecture where contamination is structurally impossible
while preserving both full causal context and per-token domain adaptation?"

Answer: The Mixture-of-Experts (MoE) architecture.

In MoE models (Mixtral, Switch Transformer, DeepSeek-V3), ALL tokens share the
same attention layers. Routing happens ONLY in the FFN/MLP layers. This is not
an accident -- it is a structural necessity:

1. Attention must see coherent representations to compute meaningful similarity
2. FFN/MLP operates token-independently (no cross-token interaction)
3. Therefore routing in MLP cannot contaminate other tokens

## C. Prior Mathematical Foundations

### Theorem (Attention linearity in K/V, Vaswani et al. 2017)

For causal self-attention at position t:

  Attn(Q_t, K_{0:t}, V_{0:t}) = softmax(Q_t K_{0:t}^T / sqrt(d_k)) V_{0:t}

The output is a weighted sum of V vectors with weights determined by Q*K similarities.
If K_i and V_i are computed by different adapters for different tokens i, the
attention output for token t mixes V vectors from multiple adapter domains.

### Property (MLP token-independence)

In standard transformer architectures, the MLP/FFN block operates independently
per token:

  MLP(x_t) = down_proj(SiLU(gate_proj(x_t)) * up_proj(x_t))

Each token t's MLP output depends ONLY on that token's input x_t. There is no
cross-token interaction in the MLP. Therefore, applying different LoRA adapters
to different tokens' MLP computations cannot contaminate other tokens.

### Proposition (MLP-only isolation guarantee)

**Claim:** If LoRA adapters modify only MLP layers (gate_proj, up_proj, down_proj),
and attention layers use base model weights for all tokens, then per-token routing
cannot produce cross-attention contamination.

**Proof sketch:**

Let f_base denote the base model forward pass for one transformer layer:

  h_t' = h_t + Attn_base(LayerNorm(h_t), {LayerNorm(h_i)}_{i<=t})     ... (attention)
  h_t'' = h_t' + MLP_adapted(LayerNorm(h_t'), domain(t))                ... (MLP)

where domain(t) selects which adapter to use for token t's MLP.

Step 1: The attention block uses base weights for ALL tokens. Therefore K_i and V_i
are computed identically for all tokens regardless of domain assignment. The attention
output is a function of base representations only. No contamination.

Step 2: The MLP block receives h_t' (which is clean base + base-attention). It applies
the domain-specific adapter to produce h_t''. Since MLP is token-independent, token
t's MLP output does not affect token s's MLP output (for s != t).

Step 3: The residual stream at token t after this layer is:
  h_t'' = h_t + Attn_base(h_t, ...) + MLP_{domain(t)}(h_t + Attn_base(h_t, ...))

The only domain-specific component is the MLP term, which depends only on token t's
own input.  QED (for one layer).

**Multi-layer propagation:** After layer l, the residual stream for token t contains
MLP-adapted information from layer l. In layer l+1, this enters the base attention's
Q/K/V projections. Token s in layer l+1 can attend to token t's residual, which
includes token t's domain-specific MLP output from layer l.

This is NOT contamination in the same sense as attention-adapter contamination. Here:
- Token t's residual truthfully represents what the model computed for token t
- The base attention in layer l+1 computes similarity using base Q/K weights
- There is no "wrong adapter" in the K/V computation -- base weights are used

This matches how MoE models work: expert-routed FFN outputs flow through shared
attention in subsequent layers. The MoE literature does not consider this contamination.

### Reference: Mixtral architecture (Jiang et al., 2024, arxiv 2401.04088)

"We use the same SlidingWindowAttention in every layer... The expert layer (MLP)
is where routing happens." Mixtral shares attention across all tokens and routes
only in the MLP. This is the production-validated architecture for per-token routing.

### Reference: Switch Transformer (Fedus et al., 2022, arxiv 2101.03961)

"We replace the feed-forward network (FFN) with a Switch FFN layer... The attention
layers are shared." Same principle at Google scale.

## D. Predictions

### Behavioral Predictions

1. **MLP-only per-token != per-sequence.** Unlike Finding #305's null result where
   per-token-full == per-sequence (both 4.815), MLP-only per-token routing should
   produce DIFFERENT PPL from per-sequence, because the contamination mechanism is
   eliminated. (K792 criterion)

2. **MLP-only per-token < per-sequence best.** With contamination eliminated, per-token
   routing can exploit the correct adapter per token within full causal context.
   (K790 criterion)

### Quantitative Predictions

The proof guarantees structural isolation, but the MAGNITUDE of improvement is the
Type 2 unknown. What we can bound:

- Finding #305 segment-isolated: 4.042 PPL (16% below per-sequence 4.815)
- Segment isolation ALSO loses cross-segment context (128 tokens only)
- MLP-only preserves full 256-token context while eliminating contamination
- Therefore MLP-only should be BETWEEN per-sequence and segment-isolated,
  or potentially BETTER than segment-isolated (if full context helps more than
  MLP-only adaptation loses vs full-module)

**K790 prediction:** MLP-only per-token PPL < 4.815 (per-sequence best)
**K791 prediction:** MLP-only per-token PPL < 4.042 (segment-isolated) -- this is
  the ambitious target. If full causal context compensates for MLP-only (vs full-module),
  this should hold.
**K792 prediction:** |MLP-only per-token PPL - per-sequence PPL| > 0.01 (non-null).
  The contamination elimination should produce a measurably different result.

### Uncertainty (the Type 2 unknown)

Finding #304 showed that for medical/math, attn-only adapters are BETTER than
full-module. This implies MLP adapters may be net-harmful for those domains.
Conversely, for code, full-module >> attn-only.

If we route ONLY MLP, we lose the attention adaptation that Finding #304 showed is
valuable. The question is whether the contamination elimination outweighs this loss.

This is the guided exploration: we know the isolation mechanism works (MoE proves it),
but we do not know whether MLP-only provides enough domain signal for our specific
adapters which were trained full-module.

## E. Assumptions & Breaking Conditions

1. **Adapters trained full-module, applied MLP-only.** The adapters' MLP parameters
   were trained jointly with attention parameters. Post-hoc ablation (dropping attention
   LoRA) may degrade MLP effectiveness. Finding #304 confirmed post-hoc ablation
   outperforms purpose-trained (Finding #308), so this is likely safe.

2. **Base attention is sufficient for causal context.** We assume the base model's
   attention provides adequate cross-token context without domain-specific adaptation.
   If domain-specific attention patterns are critical, MLP-only will underperform.

3. **Router accuracy transfers to MLP-only setting.** The ridge router (Finding #310,
   98.3%) was trained on base model hidden states. MLP-only adaptation does not change
   the hidden states input to the router (which operates before any adaptation).
   Therefore router accuracy should be preserved.

**If Assumption 1 fails:** MLP-only PPL will be worse than per-sequence, and K790 FAIL.
**If Assumption 2 fails:** MLP-only will show improvement over per-sequence but less
than segment-isolated, and K791 FAIL.

## F. Worked Example

Consider a 2-token sequence [token_0: medical, token_1: code] with 1 transformer layer.

**Full-module per-token (Finding #305 null result):**
- token_0 attention: Q=Q_base+Q_med, K=K_base+K_med, V=V_base+V_med
- token_1 attention: Q=Q_base+Q_code, K=K_base+K_code, V=V_base+V_code
- token_1 attends to token_0: uses (Q_base+Q_code) . (K_base+K_med)^T
  -- cross-adapter K contamination! token_1's query computed with code adapter
  attends to token_0's key computed with medical adapter. Similarity scores are
  computed in a mixed adapter space.

**MLP-only per-token (this experiment):**
- token_0 attention: Q=Q_base, K=K_base, V=V_base
- token_1 attention: Q=Q_base, K=K_base, V=V_base
- token_1 attends to token_0: uses Q_base . K_base^T -- clean base similarity!
- token_0 MLP: gate_base+gate_med, up_base+up_med, down_base+down_med
- token_1 MLP: gate_base+gate_code, up_base+up_code, down_base+down_code
- No cross-token contamination. Each token's MLP adaptation is independent.

## G. Complexity & Architecture Connection

**Memory:** Same as full-module -- all 5 adapter parameter sets loaded.
Only the APPLICATION changes (attention layers use base weights, MLP layers use
per-token adapter).

**FLOPs per token:** Slightly LESS than full-module per-token, because attention
forward passes use only base weights (no LoRA addmm for Q,K,V,O).

**Implementation:** The key challenge is per-token adapter application in the MLP.
Unlike standard forward passes where one adapter is applied globally, we need to:
1. Run attention with base weights (one forward pass for all tokens)
2. For each MLP layer, apply different adapters to different tokens

The simplest approach: for each domain d, compute MLP output for ALL tokens using
adapter d, then select per-token which output to use based on routing decisions.
This is O(N_domains * n_tokens) MLP forward passes but avoids custom kernels.

**Connection to production:** This is exactly how Mixtral/DeepSeek-V3 work, except
they use separate expert FFN weights instead of base + LoRA delta. The architecture
principle is identical: shared attention + routed MLP.

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   MLP token-independence: MLP(x_t) depends only on x_t, so per-token MLP routing
   cannot contaminate other tokens' representations.

2. Which existing theorem(s) does the proof build on?
   Attention mechanism definition (Vaswani et al., 2017). MoE architecture principle
   validated by Mixtral (Jiang et al., 2024, arxiv 2401.04088) and Switch Transformer
   (Fedus et al., 2022, arxiv 2101.03961).

3. What specific numbers does the proof predict?
   K790: MLP-only per-token PPL < 4.815 (per-sequence best from Finding #305)
   K791: MLP-only per-token PPL < 4.042 (segment-isolated from Finding #305)
   K792: |MLP-only per-token - per-sequence| > 0.01 (non-null)

4. What would FALSIFY the proof (not just the experiment)?
   The proof is wrong if MLP operations have cross-token dependencies in this
   architecture. This would require a non-standard MLP that mixes tokens (e.g.,
   cross-token attention within MLP). BitNet-2B-4T uses standard SiLU MLP,
   so this cannot happen.

5. How many hyperparameters does this approach add?
   Count: 0. The approach is parameter-free -- it uses existing adapters, existing
   router, and the structural decision of which modules to route is fixed (MLP only).

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is not a fix -- it is an architectural reframe. Instead of fixing
   contamination (segment isolation, scale sweep, etc.), we eliminate the
   contamination pathway entirely by restricting routing to MLP layers.
