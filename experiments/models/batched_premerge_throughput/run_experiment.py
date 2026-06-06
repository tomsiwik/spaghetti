#!/usr/bin/env python3
"""Batched Pre-Merge Throughput: amortize adapter merging across token groups.

Measures throughput of three composition strategies under per-token routing:
  1. Naive: merge adapters separately for each token
  2. Batched: group tokens by expert set, merge once per unique set
  3. Runtime LoRA: apply adapters as separate forward passes (no merge)

Uses realistic tensor shapes from BitNet-2B-4T (d=2560, r=16) with synthetic
weights to isolate the merge/matmul pipeline from model loading overhead.

Kill criteria:
  K1 (#530): Batched merge throughput not faster than naive per-token merge
  K2 (#531): Token grouping overhead exceeds merge savings

Success criteria:
  S1 (#53): Batched pre-merge achieves >= 2x throughput over naive at N>=4

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import time
from pathlib import Path
from itertools import combinations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Configuration
# ============================================================================

# Realistic dimensions from BitNet-2B-4T
D_IN = 2560      # input dimension
D_OUT = 2560     # output dimension (square for self-attn projections)
RANK = 16        # LoRA rank (proven)
LORA_SCALE = 20.0

# Test configurations
N_EXPERTS_LIST = [4, 5, 8, 12, 16]  # number of available experts
K_LIST = [1, 2]                      # top-k routing
T_LIST = [32, 64, 128, 256, 512]    # batch sizes (tokens)
N_LAYERS = 7                         # number of target layers per transformer layer
# (q, k, v, o, gate, up, down)

N_WARMUP = 3     # warmup iterations
N_MEASURE = 10   # measurement iterations


# ============================================================================
# Synthetic weight generation
# ============================================================================

def create_synthetic_weights(n_experts):
    """Create synthetic base weight and adapter matrices at realistic scale.

    Returns:
        W_base: (D_OUT, D_IN) base weight matrix
        A_matrices: list of n_experts A matrices, each (D_IN, RANK)
        B_matrices: list of n_experts B matrices, each (RANK, D_OUT)
    """
    # Base weight (ternary-like: values in {-1, 0, 1} * scale)
    W_base = mx.random.normal((D_OUT, D_IN)).astype(mx.bfloat16) * 0.01

    # Grassmannian A matrices (orthonormal, frozen)
    A_matrices = []
    for i in range(n_experts):
        A = mx.random.normal((D_IN, RANK)).astype(mx.bfloat16)
        # QR orthogonalization (approximate -- exact QR not needed for throughput test)
        A = A / mx.sqrt(mx.sum(A * A, axis=0, keepdims=True) + 1e-6)
        A_matrices.append(A)

    # B matrices (trained, not orthogonal)
    B_matrices = []
    for i in range(n_experts):
        B = mx.random.normal((RANK, D_OUT)).astype(mx.bfloat16) * 0.01
        B_matrices.append(B)

    mx.eval(W_base, *A_matrices, *B_matrices)
    return W_base, A_matrices, B_matrices


def generate_routing_decisions(T, n_experts, k):
    """Generate per-token routing decisions.

    Returns:
        assignments: list of T tuples, each a sorted tuple of k expert indices
    """
    rng = np.random.default_rng(42)
    assignments = []
    for _ in range(T):
        experts = tuple(sorted(rng.choice(n_experts, size=k, replace=False).tolist()))
        assignments.append(experts)
    return assignments


def group_tokens_by_expert_set(assignments):
    """Group token indices by their expert set assignment.

    Returns:
        groups: dict mapping expert_set_tuple -> list of token indices
    """
    groups = {}
    for t, expert_set in enumerate(assignments):
        if expert_set not in groups:
            groups[expert_set] = []
        groups[expert_set].append(t)
    return groups


# ============================================================================
# Strategy 1: Naive per-token merge
# ============================================================================

def naive_per_token_merge(X, W_base, A_matrices, B_matrices, assignments):
    """Merge adapters separately for each token, then compute output.

    For each token t with expert set S_t:
        W_t = W_base + sum_{i in S_t} scale * B_i^T @ A_i^T
        y_t = x_t @ W_t^T

    Args:
        X: (T, D_IN) input tokens
        W_base: (D_OUT, D_IN) base weights
        A_matrices: list of A matrices
        B_matrices: list of B matrices
        assignments: list of T expert set tuples

    Returns:
        Y: (T, D_OUT) output
    """
    T = X.shape[0]
    outputs = []
    for t in range(T):
        expert_set = assignments[t]
        # Merge for this token
        W_t = W_base
        for i in expert_set:
            # delta = scale * B_i^T @ A_i^T, shape (D_OUT, D_IN)
            delta = LORA_SCALE * (B_matrices[i].T @ A_matrices[i].T)
            W_t = W_t + delta
        # Forward pass: y = x @ W^T
        y_t = X[t:t+1] @ W_t.T
        outputs.append(y_t)
    Y = mx.concatenate(outputs, axis=0)
    return Y


# ============================================================================
# Strategy 2: Batched pre-merge
# ============================================================================

def batched_premerge(X, W_base, A_matrices, B_matrices, assignments):
    """Group tokens by expert set, merge once per group, batch matmul.

    1. Group tokens by expert set
    2. For each unique expert set, compute merged weight once
    3. Gather tokens for that group, batch matmul

    Args: same as naive_per_token_merge
    Returns: Y: (T, D_OUT) output (in original token order)
    """
    T = X.shape[0]
    groups = group_tokens_by_expert_set(assignments)

    # Pre-allocate output
    # We'll collect (indices, values) and scatter back
    all_indices = []
    all_outputs = []

    for expert_set, token_indices in groups.items():
        # Merge once for this expert set
        W_merged = W_base
        for i in expert_set:
            delta = LORA_SCALE * (B_matrices[i].T @ A_matrices[i].T)
            W_merged = W_merged + delta

        # Gather tokens for this group
        idx = mx.array(token_indices)
        X_group = X[idx]  # (T_group, D_IN)

        # Batch matmul
        Y_group = X_group @ W_merged.T  # (T_group, D_OUT)

        all_indices.append(idx)
        all_outputs.append(Y_group)

    # Scatter outputs back to original order
    # Create output tensor and fill in order
    Y = mx.zeros((T, D_OUT), dtype=mx.bfloat16)
    for idx, out in zip(all_indices, all_outputs):
        # Use scatter to place outputs at correct positions
        Y = Y.at[idx].add(out)

    return Y


# ============================================================================
# Strategy 3: Runtime LoRA (no merge, apply adapters in forward pass)
# ============================================================================

def runtime_lora(X, W_base, A_matrices, B_matrices, assignments):
    """Apply adapters as separate matmuls without merging into base.

    y_t = x_t @ W_base^T + sum_{i in S_t} scale * (x_t @ A_i) @ B_i

    More efficient per token because it uses the factored form
    (cost O(d*r) instead of O(d*r*d) for merge).
    """
    T = X.shape[0]
    # Base output for all tokens
    Y_base = X @ W_base.T  # (T, D_OUT)

    groups = group_tokens_by_expert_set(assignments)

    # Accumulate adapter contributions per group
    Y_adapter = mx.zeros((T, D_OUT), dtype=mx.bfloat16)

    for expert_set, token_indices in groups.items():
        idx = mx.array(token_indices)
        X_group = X[idx]  # (T_group, D_IN)

        adapter_out = mx.zeros((len(token_indices), D_OUT), dtype=mx.bfloat16)
        for i in expert_set:
            # Factored: (X @ A) @ B, cost O(T_g * d * r + T_g * r * d_out)
            hidden = X_group @ A_matrices[i]  # (T_g, RANK)
            lora_out = hidden @ B_matrices[i]  # (T_g, D_OUT)
            adapter_out = adapter_out + LORA_SCALE * lora_out

        Y_adapter = Y_adapter.at[idx].add(adapter_out)

    return Y_base + Y_adapter


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_strategy(strategy_fn, X, W_base, A_matrices, B_matrices,
                       assignments, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Benchmark a composition strategy.

    Returns:
        mean_time_ms: mean execution time in milliseconds
        std_time_ms: standard deviation
        tokens_per_sec: throughput
    """
    T = X.shape[0]

    # Warmup
    for _ in range(n_warmup):
        Y = strategy_fn(X, W_base, A_matrices, B_matrices, assignments)
        mx.eval(Y)
        del Y

    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        Y = strategy_fn(X, W_base, A_matrices, B_matrices, assignments)
        mx.eval(Y)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del Y

    times_ms = [t * 1000 for t in times]
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    tokens_per_sec = T / (mean_ms / 1000)

    return mean_ms, std_ms, tokens_per_sec


def benchmark_multi_layer(strategy_fn, n_layers, X, W_bases, A_all, B_all,
                          assignments, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Benchmark strategy across multiple layers (simulating full transformer block).

    W_bases: list of n_layers weight matrices
    A_all: list of n_layers lists of A matrices
    B_all: list of n_layers lists of B matrices
    """
    T = X.shape[0]

    def run_all_layers():
        out = X
        for layer_idx in range(n_layers):
            out = strategy_fn(out, W_bases[layer_idx], A_all[layer_idx],
                              B_all[layer_idx], assignments)
        return out

    # Warmup
    for _ in range(n_warmup):
        Y = run_all_layers()
        mx.eval(Y)
        del Y

    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        Y = run_all_layers()
        mx.eval(Y)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del Y

    times_ms = [t * 1000 for t in times]
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    tokens_per_sec = T / (mean_ms / 1000)

    return mean_ms, std_ms, tokens_per_sec


# ============================================================================
# Phase 1: Single-layer benchmark across configurations
# ============================================================================

def phase_single_layer_benchmark():
    """Benchmark all strategies on a single layer across configurations."""
    log("\n" + "=" * 70)
    log("[Phase 1] Single-layer throughput benchmark")
    log("=" * 70)

    results = []

    for n_experts in N_EXPERTS_LIST:
        for k in K_LIST:
            if k > n_experts:
                continue

            log(f"\n--- N={n_experts}, k={k} ---")
            W_base, A_matrices, B_matrices = create_synthetic_weights(n_experts)

            for T in T_LIST:
                X = mx.random.normal((T, D_IN)).astype(mx.bfloat16)
                mx.eval(X)

                assignments = generate_routing_decisions(T, n_experts, k)
                groups = group_tokens_by_expert_set(assignments)
                M = len(groups)

                log(f"  T={T}, M={M} unique expert sets (M/T={M/T:.3f})")

                # Benchmark each strategy
                strategies = {
                    "naive": naive_per_token_merge,
                    "batched": batched_premerge,
                    "runtime_lora": runtime_lora,
                }

                row = {
                    "n_experts": n_experts, "k": k, "T": T,
                    "M_unique_sets": M, "M_over_T": round(M / T, 4),
                }

                for name, fn in strategies.items():
                    mean_ms, std_ms, tps = benchmark_strategy(
                        fn, X, W_base, A_matrices, B_matrices, assignments
                    )
                    row[f"{name}_mean_ms"] = round(mean_ms, 3)
                    row[f"{name}_std_ms"] = round(std_ms, 3)
                    row[f"{name}_tokens_per_sec"] = round(tps, 1)
                    log(f"    {name:15s}: {mean_ms:8.2f} ms +/- {std_ms:5.2f}  "
                        f"({tps:10.0f} tok/s)")

                # Compute speedups
                if row["naive_mean_ms"] > 0:
                    row["batched_vs_naive_speedup"] = round(
                        row["naive_mean_ms"] / row["batched_mean_ms"], 3
                    )
                    row["runtime_vs_naive_speedup"] = round(
                        row["naive_mean_ms"] / row["runtime_lora_mean_ms"], 3
                    )
                    row["batched_vs_runtime_speedup"] = round(
                        row["runtime_lora_mean_ms"] / row["batched_mean_ms"], 3
                    )

                log(f"    Speedup batched/naive: {row.get('batched_vs_naive_speedup', 0):.2f}x")
                log(f"    Speedup runtime/naive: {row.get('runtime_vs_naive_speedup', 0):.2f}x")

                results.append(row)
                del X

            cleanup(W_base, *A_matrices, *B_matrices)

    log_memory("after-phase-1")
    return results


# ============================================================================
# Phase 2: Multi-layer benchmark (realistic transformer block)
# ============================================================================

def phase_multi_layer_benchmark():
    """Benchmark across N_LAYERS layers to simulate a full transformer block.

    This tests whether the grouping overhead is amortized across layers,
    since the grouping is done once and reused for all layers.
    """
    log("\n" + "=" * 70)
    log("[Phase 2] Multi-layer throughput benchmark (7 projections per block)")
    log("=" * 70)

    # Fixed config: N=5, k=2, varying T
    n_experts = 5
    k = 2
    results = []

    for T in T_LIST:
        log(f"\n--- T={T}, N={n_experts}, k={k}, layers={N_LAYERS} ---")

        # Create weights for all layers
        W_bases = []
        A_all = []
        B_all = []
        for _ in range(N_LAYERS):
            W, As, Bs = create_synthetic_weights(n_experts)
            W_bases.append(W)
            A_all.append(As)
            B_all.append(Bs)

        X = mx.random.normal((T, D_IN)).astype(mx.bfloat16)
        mx.eval(X)

        assignments = generate_routing_decisions(T, n_experts, k)
        groups = group_tokens_by_expert_set(assignments)
        M = len(groups)

        row = {
            "T": T, "n_experts": n_experts, "k": k,
            "n_layers": N_LAYERS, "M_unique_sets": M,
        }

        for name, fn in [("naive", naive_per_token_merge),
                          ("batched", batched_premerge),
                          ("runtime_lora", runtime_lora)]:
            mean_ms, std_ms, tps = benchmark_multi_layer(
                fn, N_LAYERS, X, W_bases, A_all, B_all, assignments
            )
            row[f"{name}_mean_ms"] = round(mean_ms, 3)
            row[f"{name}_std_ms"] = round(std_ms, 3)
            row[f"{name}_tokens_per_sec"] = round(tps, 1)
            log(f"  {name:15s}: {mean_ms:8.2f} ms +/- {std_ms:5.2f}  "
                f"({tps:10.0f} tok/s)")

        if row["naive_mean_ms"] > 0:
            row["batched_vs_naive_speedup"] = round(
                row["naive_mean_ms"] / row["batched_mean_ms"], 3
            )
        log(f"  Multi-layer speedup batched/naive: "
            f"{row.get('batched_vs_naive_speedup', 0):.2f}x")

        results.append(row)
        del X
        cleanup(*W_bases, *[a for As in A_all for a in As],
                *[b for Bs in B_all for b in Bs])

    log_memory("after-phase-2")
    return results


# ============================================================================
# Phase 3: Grouping overhead isolation
# ============================================================================

def phase_grouping_overhead():
    """Measure the pure grouping/scatter overhead without merge or matmul.

    This isolates K2: does grouping overhead exceed merge savings?
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Grouping overhead isolation")
    log("=" * 70)

    n_experts = 5
    k = 2
    results = []

    for T in T_LIST:
        X = mx.random.normal((T, D_IN)).astype(mx.bfloat16)
        mx.eval(X)

        assignments = generate_routing_decisions(T, n_experts, k)

        # Measure grouping + gather + scatter only (no merge, no matmul)
        times = []
        for _ in range(N_WARMUP):
            groups = group_tokens_by_expert_set(assignments)
            for expert_set, token_indices in groups.items():
                idx = mx.array(token_indices)
                X_group = X[idx]
                mx.eval(X_group)
                del X_group, idx

        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            groups = group_tokens_by_expert_set(assignments)
            all_gathered = []
            for expert_set, token_indices in groups.items():
                idx = mx.array(token_indices)
                X_group = X[idx]
                all_gathered.append(X_group)
            # Force eval of all gathers
            mx.eval(*all_gathered)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            del all_gathered

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        M = len(groups)

        log(f"  T={T}, M={M}: grouping+gather = {mean_ms:.3f} ms +/- {std_ms:.3f}")
        results.append({
            "T": T, "M_unique_sets": M,
            "grouping_mean_ms": round(mean_ms, 4),
            "grouping_std_ms": round(std_ms, 4),
        })

        del X

    cleanup()
    log_memory("after-phase-3")
    return results


# ============================================================================
# Phase 4: Compiled merge benchmark
# ============================================================================

def phase_compiled_merge():
    """Test mx.compile on the merge operation to see if compilation helps."""
    log("\n" + "=" * 70)
    log("[Phase 4] mx.compile on merge operation")
    log("=" * 70)

    n_experts = 5
    k = 2
    T = 256

    W_base, A_matrices, B_matrices = create_synthetic_weights(n_experts)
    X = mx.random.normal((T, D_IN)).astype(mx.bfloat16)
    mx.eval(X)

    assignments = generate_routing_decisions(T, n_experts, k)
    groups = group_tokens_by_expert_set(assignments)

    # Create a compilable merge function for a fixed expert set
    def merge_and_matmul(X_group, W_base, *adapter_params):
        """Merge adapters and compute output. adapter_params alternates A, B."""
        n_adapters = len(adapter_params) // 2
        W = W_base
        for i in range(n_adapters):
            A = adapter_params[2 * i]
            B = adapter_params[2 * i + 1]
            W = W + LORA_SCALE * (B.T @ A.T)
        return X_group @ W.T

    compiled_merge_and_matmul = mx.compile(merge_and_matmul)

    results = {}

    # Uncompiled
    times = []
    for _ in range(N_WARMUP):
        for expert_set, token_indices in groups.items():
            idx = mx.array(token_indices)
            X_group = X[idx]
            params = []
            for i in expert_set:
                params.extend([A_matrices[i], B_matrices[i]])
            Y = merge_and_matmul(X_group, W_base, *params)
            mx.eval(Y)
            del Y

    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        outputs = []
        for expert_set, token_indices in groups.items():
            idx = mx.array(token_indices)
            X_group = X[idx]
            params = []
            for i in expert_set:
                params.extend([A_matrices[i], B_matrices[i]])
            Y = merge_and_matmul(X_group, W_base, *params)
            outputs.append(Y)
        mx.eval(*outputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        del outputs

    results["uncompiled_mean_ms"] = round(float(np.mean(times)), 3)
    results["uncompiled_std_ms"] = round(float(np.std(times)), 3)
    log(f"  Uncompiled: {results['uncompiled_mean_ms']:.2f} ms")

    # Compiled
    times = []
    for _ in range(N_WARMUP + 2):  # extra warmup for compilation
        for expert_set, token_indices in groups.items():
            idx = mx.array(token_indices)
            X_group = X[idx]
            params = []
            for i in expert_set:
                params.extend([A_matrices[i], B_matrices[i]])
            Y = compiled_merge_and_matmul(X_group, W_base, *params)
            mx.eval(Y)
            del Y

    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        outputs = []
        for expert_set, token_indices in groups.items():
            idx = mx.array(token_indices)
            X_group = X[idx]
            params = []
            for i in expert_set:
                params.extend([A_matrices[i], B_matrices[i]])
            Y = compiled_merge_and_matmul(X_group, W_base, *params)
            outputs.append(Y)
        mx.eval(*outputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        del outputs

    results["compiled_mean_ms"] = round(float(np.mean(times)), 3)
    results["compiled_std_ms"] = round(float(np.std(times)), 3)
    results["compile_speedup"] = round(
        results["uncompiled_mean_ms"] / max(results["compiled_mean_ms"], 0.001), 3
    )
    log(f"  Compiled:   {results['compiled_mean_ms']:.2f} ms")
    log(f"  Compile speedup: {results['compile_speedup']:.2f}x")

    results["T"] = T
    results["n_experts"] = n_experts
    results["k"] = k
    results["M_unique_sets"] = len(groups)

    cleanup(W_base, X, *A_matrices, *B_matrices)
    log_memory("after-phase-4")
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Batched Pre-Merge Throughput Experiment")
    log(f"d_in={D_IN}, d_out={D_OUT}, rank={RANK}, scale={LORA_SCALE}")
    log(f"N_experts: {N_EXPERTS_LIST}, k: {K_LIST}, T: {T_LIST}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Single-layer benchmarks
    single_layer_results = phase_single_layer_benchmark()

    # Phase 2: Multi-layer benchmarks
    multi_layer_results = phase_multi_layer_benchmark()

    # Phase 3: Grouping overhead
    grouping_results = phase_grouping_overhead()

    # Phase 4: Compiled merge
    compiled_results = phase_compiled_merge()

    # ====================================================================
    # Kill criteria assessment
    # ====================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Batched merge throughput must be faster than naive
    # Check across all configs with N>=4
    k1_results = []
    for r in single_layer_results:
        if r["n_experts"] >= 4 and "batched_vs_naive_speedup" in r:
            k1_results.append(r["batched_vs_naive_speedup"])

    k1_all_faster = all(s > 1.0 for s in k1_results) if k1_results else False
    k1_mean_speedup = float(np.mean(k1_results)) if k1_results else 0.0
    k1_min_speedup = float(np.min(k1_results)) if k1_results else 0.0

    log(f"\nK1: Batched vs naive throughput (N>=4)")
    log(f"  All configs faster: {k1_all_faster}")
    log(f"  Mean speedup: {k1_mean_speedup:.2f}x")
    log(f"  Min speedup:  {k1_min_speedup:.2f}x")
    log(f"  K1 {'PASS' if k1_all_faster else 'FAIL'}")

    # K2: Grouping overhead must not exceed merge savings
    # Compare grouping time to merge time savings
    k2_pass = True
    for gr in grouping_results:
        # Find matching single-layer result
        matching = [r for r in single_layer_results
                    if r["T"] == gr["T"] and r["n_experts"] == 5 and r["k"] == 2]
        if matching:
            merge_savings_ms = matching[0]["naive_mean_ms"] - matching[0]["batched_mean_ms"]
            grouping_ms = gr["grouping_mean_ms"]
            if merge_savings_ms < 0:
                k2_pass = False
                log(f"\n  K2 FAIL at T={gr['T']}: no merge savings "
                    f"(naive={matching[0]['naive_mean_ms']:.2f} vs "
                    f"batched={matching[0]['batched_mean_ms']:.2f})")
            elif grouping_ms > merge_savings_ms:
                k2_pass = False
                log(f"\n  K2 FAIL at T={gr['T']}: grouping overhead "
                    f"({grouping_ms:.3f}ms) > merge savings ({merge_savings_ms:.2f}ms)")
            else:
                log(f"\n  K2 T={gr['T']}: grouping {grouping_ms:.3f}ms < "
                    f"savings {merge_savings_ms:.2f}ms -- OK")

    log(f"\nK2: Grouping overhead < merge savings: {'PASS' if k2_pass else 'FAIL'}")

    # S1: >= 2x throughput at N>=4
    s1_results = [r["batched_vs_naive_speedup"] for r in single_layer_results
                  if r["n_experts"] >= 4 and "batched_vs_naive_speedup" in r]
    s1_pass = all(s >= 2.0 for s in s1_results) if s1_results else False
    s1_mean = float(np.mean(s1_results)) if s1_results else 0.0
    log(f"\nS1: >= 2x throughput at N>=4")
    log(f"  All configs >= 2x: {s1_pass}")
    log(f"  Mean speedup: {s1_mean:.2f}x")
    log(f"  S1 {'PASS' if s1_pass else 'FAIL'}")

    # Summary
    overall = "SUPPORTED" if (k1_all_faster and k2_pass) else "KILLED"
    log(f"\nOverall: {overall}")

    # ====================================================================
    # Save results
    # ====================================================================
    results = {
        "experiment": "batched_premerge_throughput",
        "config": {
            "d_in": D_IN, "d_out": D_OUT, "rank": RANK,
            "lora_scale": LORA_SCALE, "n_layers_per_block": N_LAYERS,
            "n_experts_list": N_EXPERTS_LIST, "k_list": K_LIST,
            "t_list": T_LIST, "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
        },
        "single_layer_results": single_layer_results,
        "multi_layer_results": multi_layer_results,
        "grouping_overhead": grouping_results,
        "compiled_merge": compiled_results,
        "kill_criteria": {
            "K1_batched_faster_than_naive": k1_all_faster,
            "K1_mean_speedup": round(k1_mean_speedup, 3),
            "K1_min_speedup": round(k1_min_speedup, 3),
            "K2_grouping_overhead_within_savings": k2_pass,
        },
        "success_criteria": {
            "S1_2x_throughput_at_N_gte_4": s1_pass,
            "S1_mean_speedup": round(s1_mean, 3),
        },
        "verdict": overall,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
