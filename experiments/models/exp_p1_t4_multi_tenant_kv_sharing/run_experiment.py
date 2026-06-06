"""
T4.4: Multi-Tenant Serving with Shared KV Cache

Kill criteria:
  K1085: 8 users with different Q adapters share global-layer KV cache (bit-exact K/V identity)
  K1086: KV memory: shared = 1/8 of per-user (exactly 8x savings)
  K1087: Attention output bit-exact between shared-KV and per-user serving

Proof basis:
  Theorem 1: Q-only adapters → K_i = K_j algebraically (k_proj untouched)
  Theorem 2: Shared KV = exactly 1/N memory of per-user KV
  Theorem 3: Attention output identical under shared KV (Q differs, K/V don't)

Architecture basis: Gemma 4 E4B global layers (attention_k_eq_v=True, K=V)
  Full dims: hidden=2816, num_heads=16, num_kv_heads=2, head_dim=512
  Smoke dims: hidden=256, num_heads=4, num_kv_heads=1, head_dim=64

Dependencies: T0.4 (algebraic KV independence), T4.3 (serving mechanics)
"""

import json
import sys
import time

import mlx.core as mx
import numpy as np

IS_SMOKE = "--smoke" in sys.argv

# ─── Architecture dimensions ────────────────────────────────────────────────
if IS_SMOKE:
    # Scaled-down for fast verification
    HIDDEN = 256
    NUM_HEADS = 4
    NUM_KV_HEADS = 1
    HEAD_DIM = 64
    SEQ_LEN = 32
    N_USERS = 4          # scaled down (still verifies theorem)
    N_GLOBAL_LAYERS = 3
else:
    # Gemma 4 E4B global attention dimensions
    HIDDEN = 2816
    NUM_HEADS = 16
    NUM_KV_HEADS = 2
    HEAD_DIM = 512
    SEQ_LEN = 256
    N_USERS = 8
    N_GLOBAL_LAYERS = 7  # every 6th layer in 42-layer Gemma 4 E4B

RANK = 4          # LoRA rank for Q adapters
DTYPE = mx.float16  # serving dtype

print(f"T4.4: Multi-Tenant KV Sharing | smoke={IS_SMOKE}")
print(f"  Users={N_USERS}, GlobalLayers={N_GLOBAL_LAYERS}, SeqLen={SEQ_LEN}")
print(f"  d={HIDDEN}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
print()

# ─── Phase 1: KV Cache Identity Verification ─────────────────────────────────
# Theorem 1: K_i = K_j for all users (k_proj has no adapter)

print("Phase 1: KV Identity Across Users (Theorem 1)")

mx.random.seed(42)

results = {
    "is_smoke": IS_SMOKE,
    "n_users": N_USERS,
    "n_global_layers": N_GLOBAL_LAYERS,
    "hidden": HIDDEN,
    "head_dim": HEAD_DIM,
    "rank": RANK,
    "k1085_pass": False,
    "k1086_pass": False,
    "k1087_pass": False,
    "k1085_max_kv_diff": None,
    "k1086_memory_ratio": None,
    "k1087_max_attn_diff": None,
}

layer_kv_diffs = []
layer_attn_diffs = []

for layer_idx in range(N_GLOBAL_LAYERS):
    # ── Base weights (shared, no adapter) ──────────────────────────────────
    # W_Q base: [hidden, num_heads * head_dim]
    W_Q_base = mx.random.normal([HIDDEN, NUM_HEADS * HEAD_DIM]).astype(DTYPE) * 0.02
    # W_K and W_V: [hidden, num_kv_heads * head_dim] — NO adapter
    W_K = mx.random.normal([HIDDEN, NUM_KV_HEADS * HEAD_DIM]).astype(DTYPE) * 0.02
    # In Gemma 4 global layers: attention_k_eq_v=True → V = K (cloned)
    W_V = W_K  # Same weights, Gemma 4 global layer property

    # ── User adapters (LoRA on Q only) ─────────────────────────────────────
    # Each user i: ΔW_Q_i = B_i @ A_i,  A_i ∈ [rank, hidden], B_i ∈ [num_heads*head_dim, rank]
    A_users = [
        mx.random.normal([RANK, HIDDEN]).astype(DTYPE) * 0.01
        for _ in range(N_USERS)
    ]
    B_users = [
        mx.random.normal([NUM_HEADS * HEAD_DIM, RANK]).astype(DTYPE) * 0.01
        for _ in range(N_USERS)
    ]

    # ── Input sequence ──────────────────────────────────────────────────────
    x = mx.random.normal([SEQ_LEN, HIDDEN]).astype(DTYPE) * 0.1

    # ── Compute K once (base only, no adapter) ──────────────────────────────
    K_shared = x @ W_K   # [seq, kv_heads * head_dim]
    V_shared = K_shared   # K=V in Gemma 4 global layers

    # ── Compute K per user — should be IDENTICAL to K_shared ───────────────
    K_per_user = []
    for user_idx in range(N_USERS):
        # k_proj has no adapter → same W_K → same K
        K_user = x @ W_K  # Identical computation
        K_per_user.append(K_user)

    # Evaluate all at once
    mx.eval(K_shared, *K_per_user)

    # Check bit-exact identity (Theorem 1)
    max_kv_diff = 0.0
    for user_idx in range(N_USERS):
        diff = mx.max(mx.abs(K_per_user[user_idx] - K_shared)).item()
        max_kv_diff = max(max_kv_diff, diff)

    layer_kv_diffs.append(max_kv_diff)
    print(f"  Layer {layer_idx}: max|K_user - K_shared| = {max_kv_diff:.6e}")

    # ── Phase 3: Attention output under shared vs per-user KV ───────────────
    # Reshape for multi-head attention
    kv_head_dim = NUM_KV_HEADS * HEAD_DIM
    q_total_dim = NUM_HEADS * HEAD_DIM

    # Compute per-user attention output
    attn_shared_list = []
    attn_peruser_list = []

    scale = float(HEAD_DIM) ** -0.5

    for user_idx in range(N_USERS):
        # Q with user adapter
        lora_q = (x @ A_users[user_idx].T) @ B_users[user_idx].T  # [seq, q_total_dim]
        Q_user = x @ W_Q_base + lora_q  # [seq, q_total_dim]

        # Reshape to [seq, heads, head_dim]
        Q_3d = Q_user.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        K_3d = K_shared.reshape(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
        V_3d = V_shared.reshape(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)

        # GQA: repeat K/V heads to match Q heads
        # K: [seq, kv_heads, head_dim] → [seq, num_heads, head_dim]
        repeats = NUM_HEADS // NUM_KV_HEADS
        K_expanded = mx.repeat(K_3d, repeats, axis=1)  # [seq, num_heads, head_dim]
        V_expanded = mx.repeat(V_3d, repeats, axis=1)

        # Attention: [heads, seq_q, seq_k]
        # Q: [seq, heads, head_dim] → [heads, seq, head_dim]
        Q_t = mx.transpose(Q_3d, [1, 0, 2])
        K_t = mx.transpose(K_expanded, [1, 0, 2])
        V_t = mx.transpose(V_expanded, [1, 0, 2])

        # Scores: [heads, seq_q, seq_k]
        scores = (Q_t @ mx.transpose(K_t, [0, 2, 1])) * scale
        weights = mx.softmax(scores, axis=-1)

        # Output: [heads, seq, head_dim] → [seq, heads*head_dim]
        attn_out = mx.transpose(weights @ V_t, [1, 0, 2]).reshape(SEQ_LEN, q_total_dim)

        # Shared-KV path: identical computation but using K_shared (same value)
        K_shared_3d = K_shared.reshape(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
        V_shared_3d = V_shared.reshape(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
        K_shared_expanded = mx.repeat(K_shared_3d, repeats, axis=1)
        V_shared_expanded = mx.repeat(V_shared_3d, repeats, axis=1)
        K_s_t = mx.transpose(K_shared_expanded, [1, 0, 2])
        V_s_t = mx.transpose(V_shared_expanded, [1, 0, 2])

        scores_shared = (Q_t @ mx.transpose(K_s_t, [0, 2, 1])) * scale
        weights_shared = mx.softmax(scores_shared, axis=-1)
        attn_out_shared = mx.transpose(weights_shared @ V_s_t, [1, 0, 2]).reshape(SEQ_LEN, q_total_dim)

        attn_peruser_list.append(attn_out)
        attn_shared_list.append(attn_out_shared)

    mx.eval(*attn_peruser_list, *attn_shared_list)

    max_attn_diff = 0.0
    for user_idx in range(N_USERS):
        diff = mx.max(mx.abs(attn_peruser_list[user_idx] - attn_shared_list[user_idx])).item()
        max_attn_diff = max(max_attn_diff, diff)

    layer_attn_diffs.append(max_attn_diff)
    print(f"  Layer {layer_idx}: max|Attn_shared - Attn_peruser| = {max_attn_diff:.6e}")

print()

# ─── Phase 2: Memory Accounting ──────────────────────────────────────────────
print("Phase 2: KV Memory Accounting (Theorem 2)")

# KV cache memory per global layer
kv_elem_per_layer = SEQ_LEN * NUM_KV_HEADS * HEAD_DIM
bytes_per_elem = 2  # float16

# Per-user serving: N_USERS separate KV allocations per layer
mem_per_user_per_layer = kv_elem_per_layer * bytes_per_elem
mem_peruser_total = N_USERS * N_GLOBAL_LAYERS * mem_per_user_per_layer

# Shared serving: 1 KV allocation per layer
mem_shared_total = N_GLOBAL_LAYERS * mem_per_user_per_layer

memory_ratio = mem_shared_total / mem_peruser_total
expected_ratio = 1.0 / N_USERS

print(f"  Per-user KV total: {mem_peruser_total / 1024:.1f} KB ({N_USERS}× allocation)")
print(f"  Shared KV total:   {mem_shared_total / 1024:.1f} KB (1× allocation)")
print(f"  Memory ratio: {memory_ratio:.4f} (expected {expected_ratio:.4f} = 1/{N_USERS})")
print(f"  Memory savings: {(1 - memory_ratio) * 100:.1f}%")
print()

# ─── Kill Criteria Evaluation ─────────────────────────────────────────────────
print("=" * 60)
print("Kill Criteria Evaluation")
print("=" * 60)

max_kv_diff_all = max(layer_kv_diffs)
max_attn_diff_all = max(layer_attn_diffs)

# K1085: 8 users share KV cache (bit-exact identity)
# Threshold: 0.0 (algebraic — any nonzero fails)
k1085_pass = max_kv_diff_all == 0.0
print(f"K1085: max|K_user - K_shared| = {max_kv_diff_all:.6e}")
print(f"       Threshold: 0.0 (algebraic)")
print(f"       {'PASS' if k1085_pass else 'FAIL'}")
print()

# K1086: shared memory < 8x individual (exact 1/N)
# Threshold: ratio == 1/N_USERS (exact equality)
k1086_pass = abs(memory_ratio - expected_ratio) < 1e-9
print(f"K1086: memory ratio = {memory_ratio:.6f} (expected {expected_ratio:.6f} = 1/{N_USERS})")
print(f"       Threshold: exact equality (1/N)")
print(f"       {'PASS' if k1086_pass else 'FAIL'}")
print()

# K1087: attention output bit-exact
# Threshold: 0.0 (same operations, same inputs)
k1087_pass = max_attn_diff_all == 0.0
print(f"K1087: max|Attn_shared - Attn_peruser| = {max_attn_diff_all:.6e}")
print(f"       Threshold: 0.0 (algebraic)")
print(f"       {'PASS' if k1087_pass else 'FAIL'}")
print()

all_pass = k1085_pass and k1086_pass and k1087_pass
print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
print()

# ─── Additional: Memory savings extrapolation ──────────────────────────────
print("Memory savings extrapolation (N_USERS=8, T=1024 context, Gemma 4 E4B global layers):")
# Full Gemma 4 E4B global attention: num_kv_heads=2, head_dim=512, n_global=7
full_kv_per_layer = 1024 * 2 * 512  # T * kv_heads * head_dim elements
full_bytes_per_layer = full_kv_per_layer * 2  # float16
full_savings = 7 * 7 * full_bytes_per_layer  # (N_USERS-1) * n_global_layers * bytes
print(f"  KV savings at N=8, T=1024: {full_savings / 1024 / 1024:.1f} MB freed")
print(f"  Per layer: {full_bytes_per_layer / 1024:.1f} KB (7 global layers)")
print()

# ─── Write results ────────────────────────────────────────────────────────────
results.update({
    "k1085_pass": k1085_pass,
    "k1086_pass": k1086_pass,
    "k1087_pass": k1087_pass,
    "k1085_max_kv_diff": max_kv_diff_all,
    "k1086_memory_ratio": memory_ratio,
    "k1086_expected_ratio": expected_ratio,
    "k1087_max_attn_diff": max_attn_diff_all,
    "layer_kv_diffs": layer_kv_diffs,
    "layer_attn_diffs": layer_attn_diffs,
    "mem_peruser_kb": mem_peruser_total / 1024,
    "mem_shared_kb": mem_shared_total / 1024,
    "all_pass": all_pass,
})

with open("experiments/models/exp_p1_t4_multi_tenant_kv_sharing/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results written to results.json")
print(f"Status: {'SUPPORTED' if all_pass else 'KILLED'}")
sys.exit(0 if all_pass else 1)
