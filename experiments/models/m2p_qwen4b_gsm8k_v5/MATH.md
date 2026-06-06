# MATH.md: M2P v5 — SHINE-Aligned Base-as-Encoder on Qwen3-0.6B

## TYPE: frontier-extension
## PROVEN FRAMEWORK: M2P adapter generation (#376/#378), functional LoRA (#376)
## FRONTIER: Replace standalone MLP encoder with base-model-as-encoder (SHINE architecture)

---

## Why v5

v3/v4 used a standalone 2-layer MLP encoder (357M params at 0.6B, 763M at 4B).
At 4B, this produced degenerate B-matrices (quality_ratio = -0.125, Finding #400).

Root cause: mean-pooling (L, d_model) → single d_model vector destroys per-layer
variation. All 36 layers receive B-matrices generated from the SAME bottleneck vector.

SHINE (arXiv:2602.06358) solves this by using the frozen base model as encoder:
1. Prepend M learnable memory tokens to input
2. Run full sequence through frozen base → extract memory hidden states per layer
3. Small M2P transformer with alternating row/column attention contextualizes across layers
4. Per-layer heads generate B-matrices from layer-specific representations

## Theorem 1: Memory Token Information Capacity

For M memory tokens at d_model width, each layer produces M·d_model values.
To generate rank-r B-matrices for q_proj (d_out=q_proj_out) and v_proj (d_out=v_proj_out):

    required_per_layer = r · (q_proj_out + v_proj_out)
    capacity_per_layer = M · d_model

The capacity condition: M ≥ r·(q_proj_out + v_proj_out) / d_model

For Qwen3-0.6B: M ≥ 4·(2048+1024)/1024 = 12. We use M=16 (33% overcomplete).
For Qwen3-4B:   M ≥ 4·(4096+1024)/2560 = 8. We use M=16 (100% overcomplete).

## Theorem 2: Cross-Layer Attention Preserves Per-Layer Variation

v3/v4's mean-pool maps (L, d_model) → (d_model), losing all layer identity.
The M2P transformer with column attention maps (L, M, d_model) → (L, M, d_model),
preserving the layer dimension. Each layer's B-head receives a DIFFERENT input
(the L-th row of the transformer output), not the same global average.

This is the key architectural fix for the 4B failure: the M2P transformer can
learn that early layers need different B-matrices than late layers.

## Theorem 3: Parameter Scaling (SHINE vs Standalone)

Standalone encoder (v3/v4):
    params = encoder_mlp + Σ_l (head_q[l] + head_v[l])
           = O(d_model²) + L·d_m2p·r·(q_out + v_out)

SHINE-aligned M2P (v5):
    params = memory_tokens + proj + m2p_transformer + Σ_l (head_q[l] + head_v[l])
           = M·d + d·d_m2p + 4·12·d_m2p² + L·d_m2p·r·(q_out + v_out)

The per-layer heads dominate both. The key difference: the M2P transformer
replaces the mean-pool with cross-layer attention at cost 4·12·d_m2p² ≈ 48M.
This is 48M extra params for dramatically better per-layer differentiation.

At 0.6B: v5 total ≈ 401M (vs v3's 357M). Similar, but architecturally superior.
At 200B: the M2P transformer is tiny relative to the per-layer heads.

## Kill Criteria

**K1:** grad_norm > 0 at step 0 (gradient flows through M2P transformer + heads)
**K2:** quality_ratio ≥ 0.60 on GSM8K (matches v3 performance)
**K3:** M2P transformer + positional params < 100M

## Predictions

| Metric | Predicted | Reasoning |
|--------|-----------|-----------|
| quality_ratio | 0.60-0.85 | Should match v3 (0.75) or improve via cross-layer attention |
| M2P transformer params | ~48M | 4 blocks × 12 × 1024² |
| Total M2P params | ~401M | Transformer + per-layer heads |
| Generation time | ~15ms | Base forward + 4 M2P layers on 16 tokens |

## Self-Test

1. **What makes failure impossible?** The M2P transformer's column attention ensures each
   layer's B-head receives a layer-specific representation, not a mean-pooled global average.
   This is structural (alternating attention pattern), not tuned.

2. **Cited theorems:** SHINE (2602.06358) validated on Qwen3-8B. Finding #376 proved
   functional LoRA gradient flow. Finding #363/#365 proved M2P quality at L=36.

3. **Predicted numbers:** See table above. quality_ratio 0.60-0.85 based on v3 at 0.75.

4. **Falsification:** If quality_ratio < 0.30, the M2P transformer fails to learn useful
   cross-layer representations. This would mean the architectural fix is insufficient.

5. **Hyperparameters:** M=16 (derived from capacity bound), output_scale=0.032 (SHINE),
   4 M2P blocks (SHINE default), d_m2p=d_model (SHINE requirement for 0.6B).
