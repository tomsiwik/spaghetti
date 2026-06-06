# MATH.md — T4.4: Multi-Tenant KV Cache Sharing Under Q-Only Adapters

**Type:** Verification  
**Status:** Theorem proven, experiment verifies predictions  

---

## Background

Gemma 4 E4B has 42 attention layers:
- **Local layers (35):** Sliding-window attention, window=4096
- **Global layers (7):** Full-sequence attention, at indices {6, 13, 19, 25, 31, 37, 41} (every 6th)

Finding #T0.4 established: On global layers with K=V (GQA), Q-only adapters produce
adapter-independent K and V. This is algebraically guaranteed.

Finding #309 (killed) found KV reuse harmful — but that experiment mixed adapters on K/V
with reuse. Our adapters only touch q_proj.

---

## Theorem 1: KV Cache Independence Under Q-Only Adapters

**Setup:**
- Base weight matrix W_K ∈ ℝ^{d_k × d} (no LoRA adapter on k_proj in global layers)
- User i's adapter: ΔW_Q^i = B_i @ A_i where A_i ∈ ℝ^{r × d}, B_i ∈ ℝ^{d × r}
- Input sequence x ∈ ℝ^{T × d}

**Claim:** For any two users i, j and any input x:
```
K_i = K_j  (bit-exact, not floating-point approximation)
```

**Proof:**
```
K_i = W_K @ x     (k_proj has no adapter for user i)
K_j = W_K @ x     (k_proj has no adapter for user j)
∴ K_i - K_j = 0   □ (algebraic identity, same computation)
```

Note: In Gemma 4 E4B with GQA (4 query heads, 1 KV head), V = v_proj(x) is also
untouched by adapters. So V_i = V_j = V for all users. QED.

---

## Theorem 2: Multi-Tenant KV Memory Reduction

**Setup:** N users, each with distinct q_proj adapter. Serving batch of T tokens.

**Claim:** Shared KV serving requires exactly (1/N) of the KV memory of per-user serving.

**Proof:**
```
Per-user serving: allocate KV_1, KV_2, ..., KV_N  →  N copies of shape [T, d_k]
Shared serving: allocate KV once                  →  1 copy  of shape [T, d_k]
Ratio: 1/N  □
```

For N=8, d_k=256, T=1024, dtype=float16:
```
Per-user KV memory:  8 × 1024 × 256 × 2B = 4.194 MB
Shared KV memory:    1 × 1024 × 256 × 2B = 0.524 MB
Savings: 87.5% (exactly 8×)
```

**Note:** This is per global layer. For 7 global layers:
```
Savings per sequence: 7 × (N-1) × T × d_k × dtype_bytes
At N=8, T=1024, d_k=256, float16: 7 × 7 × 0.524 MB = 25.7 MB
```

---

## Theorem 3: Attention Output Correctness Under KV Sharing

**Claim:** Per-user attention output is preserved exactly under shared KV.

**Proof:**
```
Per-user: Attn_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
Shared:   Attn_i = softmax(Q_i @ K^T   / sqrt(d_k)) @ V

Since K_i = K (Theorem 1) and V_i = V (same argument):
  softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
= softmax(Q_i @ K^T  / sqrt(d_k)) @ V   □
```

The attention output for user i is identical whether computed with per-user or shared KV.
This is bit-exact: same floating-point operations, same inputs.

---

## Kill Criteria Predictions

**K1085: 8 users share global-layer KV cache**
- Prediction: K_user_i = K_base for all i ∈ {1..8}, max|diff| = 0.0 (exact)
- Ground: Theorem 1. Algebraic identity.

**K1086: KV memory: shared < 8x individual**  
- Prediction: exactly 8x savings on global-layer KV (N=8 users)
- Ground: Theorem 2. Memory = 1 allocation vs 8.
- Measured: ratio = bytes_shared / bytes_individual = 1/8 = 0.125

**K1087: Quality identical (bit-exact on global layers)**
- Prediction: max|Attn_shared_i - Attn_individual_i| = 0.0 (exact)
- Ground: Theorem 3. Same operations, same inputs.

---

## Connection to T4.3 and T4.6

T4.3 showed swap latency ~5ms per adapter. With KV sharing:
- K, V computed once for global layers → no per-user KV recomputation
- Q computed N times (one per user per global layer)
- Net: global attention saves (N-1)/N of KV bandwidth per global layer

At 8 users, 7 global layers, each saving 87.5% of KV computation:
- KV compute savings: 7 layers × 87.5% = 6.125 layer-equivalents saved
- Q extra compute: 7 layers × 8 Q projections (vs 1 base) = negligible LoRA overhead

**Behavioral prediction:** A shared-KV serving system for 8 users produces
bit-exact identical outputs vs per-user serving while using 8x less KV memory
on global layers. This is not approximate — it's an algebraic identity.
