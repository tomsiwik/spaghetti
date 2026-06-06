# T0.4: Q-Only Adapters Work on K=V Global Layers

**Experiment:** exp_p1_t0_kv_shared_qonly  
**Type:** verification  
**Date:** 2026-04-09

---

## Theorem 1 (KV Cache Invariance with Q-Only Adapters)

**Statement:** In Gemma 4 global attention layers where V = K.clone() (attention_k_eq_v=True,
v_proj=None), a Q-only adapter leaves K and V unchanged for any input. Therefore, all
Q-only adapter variants share an identical KV cache for the same input sequence.

**Proof:**

Gemma 4 global attention layer (from mlx_vlm/models/gemma4/language.py):
```
Q = q_proj(x) + adapter_q(x)      ← adapter modifies Q only
K = k_proj(x)                      ← no adapter, pure base model
V = K.clone()                       ← V IS K (no v_proj)
attn = softmax(Q @ K^T / sqrt(d)) @ V
```

For any two Q-only adapters A₁, A₂ on the same input x:
```
K_A1 = k_proj(x) = K_A2           ← K is identical (no adapter on K)
V_A1 = K_A1.clone() = K_A2.clone() = V_A2   ← V is identical
```

Therefore: KV cache = {K, V} is IDENTICAL for all Q-only adapters on same input.

**Corollary (Multi-Tenant KV Sharing):** N users with different Q-only domain adapters
can share a single KV cache. Memory cost is O(1) KV cache regardless of N.

**Proof:** Follows directly from Theorem 1 — K and V are base-model-only, so any
number of Q-only adapters produce the same K, V for the same input. **QED**

---

## Theorem 2 (Q-Projection Sufficiency for Domain Adaptation)

**Statement:** In the K=V attention architecture, the attention output is:
```
Output = softmax(Q_adapted @ K^T / sqrt(d)) @ V
```
where K and V are fixed. The adapter can fully control the attention distribution
via Q alone, which is sufficient for domain adaptation.

**Proof sketch:**

The attention score A = softmax(Q @ K^T / sqrt(d)) is a function of Q given fixed K.
Domain adaptation requires changing which K-positions are attended to.

By modifying Q via LoRA: Q_adapted = Q_base + ΔQ = Q_base + x @ A_q @ B_q

The space of achievable attention distributions via Q-only LoRA is:
```
{softmax((Q_base + x @ A_q @ B_q) @ K^T / sqrt(d)) : A_q ∈ R^{d×r}, B_q ∈ R^{r×d_q}}
```

For rank-r LoRA, this spans a neighborhood of the base distribution rich enough to:
1. Attend to task-relevant tokens (math problem tokens vs medical tokens)
2. Ignore irrelevant tokens (domain-specific skip patterns)

Reference: The attention output is controlled by the query-key similarity. Domain-
specific queries (what to ask) are independent of domain-specific values (what to say),
supporting Q-only as the correct hook for query-side domain routing.

**Prediction (K1000):** Q-only quality_ratio >= 0.85 vs Q+K adapter.
For tasks where query adaptation is sufficient (retrieval, classification), ratio → 1.0.

---

## Quantitative Predictions

| Kill Criterion | Prediction | Confidence |
|---|---|---|
| K1000: Q-only quality_ratio >= 0.85 vs Q+K | 0.90-1.05 (query-centric tasks) | 80% |
| K1001: K output identical with/without Q adapter | Exactly equal (algebraic) | 100% |
| K1002: 2 users with diff Q adapters → same K | Exactly equal (algebraic) | 100% |

---

## Architecture Reference

Gemma 4 global layer config (26B-A4B):
```python
num_attention_heads: 16      # Q heads
num_global_key_value_heads: 2  # K and V heads (shared, K=V)
global_head_dim: 512           # head dimension
attention_k_eq_v: True         # V = K.clone(), no v_proj
```

Implication: In a request batch of N users:
- Q: N different adapter outputs (user-specific)
- K: 1 shared base output (same for all users)
- V: 1 shared = K (same for all users)
- Memory: 1 KV cache shared across N users

Reference: Gemma 3 Technical Report (arxiv 2503.19786) — GQA + sliding window foundation.
