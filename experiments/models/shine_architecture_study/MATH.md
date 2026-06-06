# MATH: SHINE Piece C — M2P Transformer Architecture for Gemma 4 E4B

**Type:** Verification (Type 1)
**Status:** Pre-experiment

---

## Motivation

exp_shine_port (Finding #336) proved M2P is portable to MLX at toy scale (L=4, H=64, 197K params).
exp_p1_c1_polar_gemma4_correct (Finding #1146) proved PoLAR achieves sr=r=16 exactly on Gemma 4 E4B.

C2 integration requires M2P configured for Gemma 4 E4B production dimensions:
- L = 42 layers, d_model = 2560, r = 16 (PoLAR rank)

This experiment verifies (1) M2P has no Qwen-specific components at production scale,
and (2) parameter count stays << 1B for practical M2P configs targeting E4B.

---

## Architecture Reference

**SHINE** (arXiv:2602.06358, §3.4): Memory-to-Parameter (M2P) Transformer.

Input memory state: **Z ∈ ℝ^{L×M×H}**
- L = number of model layers (Gemma 4 E4B: L=42)
- M = number of memory tokens per layer (hyperparameter, typically 8-64)
- H = M2P hidden dimension (hyperparameter, independent of d_model)

Architecture:
```
for i in range(n_m2p_layers):
    Z = Z + RowAttn(LayerNorm(Z))     # attend over M dimension within each layer
    Z = Z + ColAttn(LayerNorm(Z))     # attend over L dimension across layers
    Z = Z + FFN(LayerNorm(Z))
Z_out = mean(Z, dim=M)                # (L, H) — aggregate per-layer representation
W_A   = proj_A(Z_out)                 # (L, r, d_in)  — LoRA A matrices
W_B   = proj_B(Z_out)                 # (L, d_out, r) — LoRA B matrices
```

---

## Theorem 1: Architecture Agnosticism

**Claim:** M2P Transformer contains no components specific to any LLM architecture.

**Proof:**
The M2P forward pass depends only on:
- Input shape (L, M, H) — fully parameterized, not architecture-specific
- Output projections proj_A, proj_B — linear maps to (r, d_in) and (d_out, r)
  where d_in, d_out are adapter dimensions (runtime parameters, not baked in)

No Qwen-specific modules are used:
- No RotaryEmbedding or NTK scaling
- No RMSNorm with group-norm (Qwen3 uses standard RMSNorm)
- No GQA implementation internal to M2P
- No tokenizer/vocabulary artifacts

The row/column attention pattern is standard bidirectional self-attention applied to
a 2D grid — purely geometric, parameterized only by (L, M, H). ∎

**K806 prediction:** PASS. Architecture instantiates cleanly for any (L, M, H) triple.

---

## Theorem 2: Parameter Count Bound

**Claim:** M2P with n_layers=4, M=32, H=256, targeting Gemma 4 E4B (L=42, d_model=2560, r=16)
has < 30M parameters.

**Proof:**
Components:
1. Positional embeddings: P_layer ∈ ℝ^{L×1×H}, P_token ∈ ℝ^{1×M×H}
   → (L + M) × H = (42 + 32) × 256 = 18,944

2. n_m2p layers of row/column attention + FFN:
   Each attention: 4 weight matrices of shape (H, H) = 4 × H² = 4 × 65,536 = 262,144
   × 2 (row + column) × n_m2p = 2 × 4 × 262,144 = 2,097,152
   FFN (H → 4H → H): 2 × 4H² = 2 × 4 × 65,536 = 524,288
   Layer norms: 3 × 2H = 1,536 (negligible)
   Total attention+FFN: 4 × (2,097,152 + 524,288) ≈ 10.5M

3. Output projections (applied per-layer as shared linear):
   proj_A: H → r × d_in = 256 → 16 × 2560 = 256 × 40,960 = 10,485,760
   proj_B: H → d_out × r = 256 → 2560 × 16 = same = 10,485,760
   Total: ~21M

Grand total: ≈ 18,944 + 10.5M + 21M ≈ **31.5M** < 1B ∎

Note: If we use rank-factored output projection (H → r → adapter) instead of direct H → d×r,
parameter count drops to ~2M for the output head. The 31.5M bound is the worst case.

**K807 prediction:** PASS. 31.5M << 1B = 1000M.

---

## Quantitative Predictions

| Metric | Prediction | Source |
|--------|-----------|--------|
| K806: Qwen-specific features | None found (PASS) | Theorem 1 |
| K807: Parameter count | < 35M (PASS) | Theorem 2 |
| Output shape lora_A | (42, 16, 2560) | Gemma 4 E4B dims |
| Output shape lora_B | (42, 2560, 16) | Gemma 4 E4B dims |
| Forward latency (L=42) | < 100ms | shine_port: 4.1ms at L=4 |
| Architecture portable to L=10,20,42 | Yes (shape-agnostic) | Theorem 1 |

---

## Kill Criteria Mapping

- **K806 KILL:** If any M2P component requires Qwen-specific imports or config fields
  that don't map to generic (L, M, H) parameters
- **K807 KILL:** If minimum viable M2P for E4B exceeds 1B parameters

Both should PASS given Theorem 1 and Theorem 2. Finding the exact parameter count
at E4B production dimensions is the primary measurement.

---

## Connection to C2 Integration

This experiment directly enables C2: PoLAR + M2P joint architecture.

C2 design (future):
- M2P generates initial LoRA weights: W_A (42, r, 2560), W_B (42, 2560, r)
- PoLAR polar retraction applied to W_A (→ Stiefel manifold), W_B (→ Stiefel)
- Grassmannian composition: adapters are orthogonal by construction
- Session context → M2P → PoLAR adapters → composed base behavior

This experiment establishes that M2P can generate the right shapes for this pipeline.

---

## References

- SHINE (arXiv:2602.06358): M2P architecture, §3.4
- exp_shine_port (Finding #336): M2P portable at toy scale, 197K params
- exp_p1_c1_polar_gemma4_correct (Finding #1146): PoLAR sr=r=16 on Gemma 4 E4B
- Gemma 4 E4B: hidden_size=2560, num_hidden_layers=42, head_dim=256
