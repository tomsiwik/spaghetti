#!/usr/bin/env python3
"""Benchmark: composition latency vs N adapters (1-100) with pre-merge on MLX.

Measures:
  - Pre-merge latency as function of N adapters (1, 2, 5, 10, 25, 50, 75, 100)
  - Forward pass latency (should be constant regardless of N)
  - Memory usage at each N
  - Scaling law fit (power law T = a * N^alpha)
  - mx.compile optimization effect
  - Bottleneck analysis (merge vs forward vs dispatch)

Kill criteria:
  K1 (#255): Pre-merge latency grows superlinearly with N (alpha > 1.05)

Success criteria:
  S1 (#48): Latency grows sub-linearly with N, interactive at N=25 (<50ms overhead)

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Configuration: BitNet-2B-4T dimensions
D_IN = 2560
D_OUT = 2560
RANK = 16
LORA_SCALE = 20.0

# Sweep over N adapters
N_SWEEP = [1, 2, 5, 10, 25, 50, 75, 100]

# Benchmark parameters
N_WARMUP = 5
N_MEASURE = 20  # more measurements for stable timing
N_LAYERS = 7    # projections per transformer block: q, k, v, o, gate, up, down


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
# Synthetic weight generation
# ============================================================================

def create_adapters(n_adapters):
    """Create synthetic Grassmannian A matrices and trained B matrices.

    Returns:
        A_list: list of n_adapters A matrices, each (D_IN, RANK) bf16
        B_list: list of n_adapters B matrices, each (RANK, D_OUT) bf16
    """
    A_list = []
    B_list = []
    for i in range(n_adapters):
        # Grassmannian-like orthonormal A (column-normalized)
        A = mx.random.normal((D_IN, RANK)).astype(mx.bfloat16)
        A = A / mx.sqrt(mx.sum(A * A, axis=0, keepdims=True) + 1e-6)
        # Trained B
        B = mx.random.normal((RANK, D_OUT)).astype(mx.bfloat16) * 0.01
        A_list.append(A)
        B_list.append(B)

    mx.eval(*A_list, *B_list)
    return A_list, B_list


def create_base_weight():
    """Create synthetic base weight matrix."""
    W = mx.random.normal((D_OUT, D_IN)).astype(mx.bfloat16) * 0.01
    mx.eval(W)
    return W


# ============================================================================
# Pre-merge: uncompiled
# ============================================================================

def premerge_uncompiled(W_base, A_list, B_list, alphas):
    """Merge N adapters into base weight.

    W_merged = W_base + sum_{i} alpha_i * (B_i^T @ A_i^T)
    """
    W = W_base
    for i in range(len(A_list)):
        # delta_i = B_i^T @ A_i^T, shape (D_OUT, D_IN)
        delta = B_list[i].T @ A_list[i].T
        W = W + alphas[i] * delta
    return W


# ============================================================================
# Pre-merge: compiled (for fixed N, adapters passed as explicit args)
# ============================================================================

def make_compiled_merge(n_adapters):
    """Create a compiled merge function for a fixed number of adapters.

    We pass all adapter params as a flat list to avoid closure capture issues.
    """
    def merge_fn(W_base, alphas, *adapter_params):
        # adapter_params alternates: A_0, B_0, A_1, B_1, ...
        W = W_base
        for i in range(n_adapters):
            A = adapter_params[2 * i]
            B = adapter_params[2 * i + 1]
            delta = B.T @ A.T
            W = W + alphas[i] * delta
        return W

    return mx.compile(merge_fn)


# ============================================================================
# Phase 1: Pre-merge latency sweep (uncompiled)
# ============================================================================

def phase_merge_latency_sweep():
    """Measure pre-merge latency as f(N) for N in N_SWEEP."""
    log("\n" + "=" * 70)
    log("[Phase 1] Pre-merge latency sweep (uncompiled)")
    log("=" * 70)

    results = []

    for N in N_SWEEP:
        log(f"\n--- N={N} adapters ---")

        W_base = create_base_weight()
        A_list, B_list = create_adapters(N)
        alphas = mx.ones((N,), dtype=mx.bfloat16) / N  # uniform 1/N
        mx.eval(alphas)

        # Warmup
        for _ in range(N_WARMUP):
            W_merged = premerge_uncompiled(W_base, A_list, B_list, alphas)
            mx.eval(W_merged)
            del W_merged

        # Measure merge time
        merge_times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            W_merged = premerge_uncompiled(W_base, A_list, B_list, alphas)
            mx.eval(W_merged)
            elapsed = time.perf_counter() - t0
            merge_times.append(elapsed * 1000)  # ms
            del W_merged

        # Measure forward pass time (should be constant)
        x = mx.random.normal((1, D_IN)).astype(mx.bfloat16)
        mx.eval(x)

        W_merged = premerge_uncompiled(W_base, A_list, B_list, alphas)
        mx.eval(W_merged)

        fwd_times = []
        for _ in range(N_WARMUP):
            y = x @ W_merged.T
            mx.eval(y)
            del y

        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            y = x @ W_merged.T
            mx.eval(y)
            elapsed = time.perf_counter() - t0
            fwd_times.append(elapsed * 1000)
            del y

        del W_merged

        # Measure combined merge+forward (the full latency hit)
        combined_times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            W_merged = premerge_uncompiled(W_base, A_list, B_list, alphas)
            y = x @ W_merged.T
            mx.eval(y)
            elapsed = time.perf_counter() - t0
            combined_times.append(elapsed * 1000)
            del W_merged, y

        # Memory snapshot
        mx.eval(mx.zeros(1))  # force eval to get accurate memory
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9

        merge_mean = float(np.mean(merge_times))
        merge_std = float(np.std(merge_times))
        fwd_mean = float(np.mean(fwd_times))
        fwd_std = float(np.std(fwd_times))
        combined_mean = float(np.mean(combined_times))

        row = {
            "N": N,
            "merge_mean_ms": round(merge_mean, 4),
            "merge_std_ms": round(merge_std, 4),
            "merge_p50_ms": round(float(np.percentile(merge_times, 50)), 4),
            "merge_p95_ms": round(float(np.percentile(merge_times, 95)), 4),
            "forward_mean_ms": round(fwd_mean, 4),
            "forward_std_ms": round(fwd_std, 4),
            "combined_mean_ms": round(combined_mean, 4),
            "active_memory_gb": round(active_gb, 3),
            "peak_memory_gb": round(peak_gb, 3),
        }
        results.append(row)

        log(f"  Merge:   {merge_mean:.3f} +/- {merge_std:.3f} ms "
            f"(p50={row['merge_p50_ms']:.3f}, p95={row['merge_p95_ms']:.3f})")
        log(f"  Forward: {fwd_mean:.3f} +/- {fwd_std:.3f} ms")
        log(f"  Combined: {combined_mean:.3f} ms")
        log(f"  Memory: active={active_gb:.3f}GB peak={peak_gb:.3f}GB")

        cleanup(W_base, x, alphas, *A_list, *B_list)

    log_memory("after-phase-1")
    return results


# ============================================================================
# Phase 2: Compiled pre-merge latency sweep
# ============================================================================

def phase_compiled_merge_sweep():
    """Measure compiled pre-merge latency as f(N)."""
    log("\n" + "=" * 70)
    log("[Phase 2] Pre-merge latency sweep (mx.compile)")
    log("=" * 70)

    results = []

    for N in N_SWEEP:
        log(f"\n--- N={N} adapters (compiled) ---")

        W_base = create_base_weight()
        A_list, B_list = create_adapters(N)
        alphas = mx.ones((N,), dtype=mx.bfloat16) / N
        mx.eval(alphas)

        compiled_merge = make_compiled_merge(N)

        # Flatten adapter params for compiled fn
        adapter_params = []
        for i in range(N):
            adapter_params.extend([A_list[i], B_list[i]])

        # Extra warmup for compilation
        for _ in range(N_WARMUP + 3):
            W_merged = compiled_merge(W_base, alphas, *adapter_params)
            mx.eval(W_merged)
            del W_merged

        # Measure
        merge_times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            W_merged = compiled_merge(W_base, alphas, *adapter_params)
            mx.eval(W_merged)
            elapsed = time.perf_counter() - t0
            merge_times.append(elapsed * 1000)
            del W_merged

        merge_mean = float(np.mean(merge_times))
        merge_std = float(np.std(merge_times))

        row = {
            "N": N,
            "compiled_merge_mean_ms": round(merge_mean, 4),
            "compiled_merge_std_ms": round(merge_std, 4),
            "compiled_merge_p50_ms": round(float(np.percentile(merge_times, 50)), 4),
            "compiled_merge_p95_ms": round(float(np.percentile(merge_times, 95)), 4),
        }
        results.append(row)

        log(f"  Compiled merge: {merge_mean:.3f} +/- {merge_std:.3f} ms "
            f"(p50={row['compiled_merge_p50_ms']:.3f}, p95={row['compiled_merge_p95_ms']:.3f})")

        cleanup(W_base, alphas, *A_list, *B_list, *adapter_params)
        del compiled_merge

    log_memory("after-phase-2")
    return results


# ============================================================================
# Phase 3: Multi-layer merge (realistic transformer block)
# ============================================================================

def phase_multi_layer_merge():
    """Measure merge latency across 7 projections (one transformer block)."""
    log("\n" + "=" * 70)
    log("[Phase 3] Multi-layer merge (7 projections per block)")
    log("=" * 70)

    results = []

    for N in N_SWEEP:
        log(f"\n--- N={N} adapters, {N_LAYERS} layers ---")

        # Create weights for all layers
        all_W = []
        all_A = []
        all_B = []
        for _ in range(N_LAYERS):
            W = create_base_weight()
            A_list, B_list = create_adapters(N)
            all_W.append(W)
            all_A.append(A_list)
            all_B.append(B_list)

        alphas = mx.ones((N,), dtype=mx.bfloat16) / N
        mx.eval(alphas)

        # Warmup
        for _ in range(N_WARMUP):
            for layer_idx in range(N_LAYERS):
                W_merged = premerge_uncompiled(
                    all_W[layer_idx], all_A[layer_idx], all_B[layer_idx], alphas
                )
                mx.eval(W_merged)
                del W_merged

        # Measure: merge all 7 layers
        times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            merged = []
            for layer_idx in range(N_LAYERS):
                W_merged = premerge_uncompiled(
                    all_W[layer_idx], all_A[layer_idx], all_B[layer_idx], alphas
                )
                merged.append(W_merged)
            mx.eval(*merged)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            del merged

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))

        row = {
            "N": N,
            "n_layers": N_LAYERS,
            "multilayer_merge_mean_ms": round(mean_ms, 4),
            "multilayer_merge_std_ms": round(std_ms, 4),
            "per_layer_merge_ms": round(mean_ms / N_LAYERS, 4),
        }
        results.append(row)

        log(f"  {N_LAYERS}-layer merge: {mean_ms:.3f} +/- {std_ms:.3f} ms "
            f"({mean_ms/N_LAYERS:.3f} ms/layer)")

        cleanup(alphas,
                *all_W,
                *[a for As in all_A for a in As],
                *[b for Bs in all_B for b in Bs])

    log_memory("after-phase-3")
    return results


# ============================================================================
# Phase 4: Bottleneck analysis (merge vs dispatch vs memory)
# ============================================================================

def phase_bottleneck_analysis():
    """Isolate bottleneck: pure matmul vs addition vs Metal dispatch."""
    log("\n" + "=" * 70)
    log("[Phase 4] Bottleneck analysis")
    log("=" * 70)

    N = 25  # target N for interactive use
    W_base = create_base_weight()
    A_list, B_list = create_adapters(N)
    alphas = mx.ones((N,), dtype=mx.bfloat16) / N
    mx.eval(alphas)

    results = {}

    # 4a: Time just the matmuls (B^T @ A^T) without accumulation
    log("\n  [4a] Pure matmul time (25 rank-16 outer products)")
    for _ in range(N_WARMUP):
        deltas = [B_list[i].T @ A_list[i].T for i in range(N)]
        mx.eval(*deltas)
        del deltas

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        deltas = [B_list[i].T @ A_list[i].T for i in range(N)]
        mx.eval(*deltas)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        del deltas

    results["matmul_only_ms"] = round(float(np.mean(times)), 4)
    log(f"    Matmul only: {results['matmul_only_ms']:.3f} ms")

    # 4b: Time the accumulation (sum of pre-computed deltas)
    log("  [4b] Pure accumulation time (sum 25 deltas into W)")
    precomputed = [B_list[i].T @ A_list[i].T for i in range(N)]
    mx.eval(*precomputed)

    for _ in range(N_WARMUP):
        W = W_base
        for i in range(N):
            W = W + alphas[i] * precomputed[i]
        mx.eval(W)
        del W

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        W = W_base
        for i in range(N):
            W = W + alphas[i] * precomputed[i]
        mx.eval(W)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        del W

    results["accumulation_only_ms"] = round(float(np.mean(times)), 4)
    log(f"    Accumulation only: {results['accumulation_only_ms']:.3f} ms")

    # 4c: Time a vectorized merge (stack all deltas, sum along axis)
    log("  [4c] Vectorized merge (stack + weighted sum)")
    stacked_deltas = mx.stack(precomputed, axis=0)  # (N, D_OUT, D_IN)
    mx.eval(stacked_deltas)

    for _ in range(N_WARMUP):
        weighted = alphas[:, None, None] * stacked_deltas  # broadcast
        total_delta = mx.sum(weighted, axis=0)
        W = W_base + total_delta
        mx.eval(W)
        del weighted, total_delta, W

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        weighted = alphas[:, None, None] * stacked_deltas
        total_delta = mx.sum(weighted, axis=0)
        W = W_base + total_delta
        mx.eval(W)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        del weighted, total_delta, W

    results["vectorized_merge_ms"] = round(float(np.mean(times)), 4)
    log(f"    Vectorized merge: {results['vectorized_merge_ms']:.3f} ms")

    # 4d: Memory cost of stacked deltas
    stacked_bytes = N * D_OUT * D_IN * 2  # bf16
    results["stacked_deltas_mb"] = round(stacked_bytes / 1e6, 2)
    log(f"    Stacked deltas memory: {results['stacked_deltas_mb']:.1f} MB")

    cleanup(W_base, alphas, stacked_deltas, *A_list, *B_list, *precomputed)
    log_memory("after-phase-4")
    return results


# ============================================================================
# Phase 5: Precomputed delta cache strategy
# ============================================================================

def phase_precomputed_delta_cache():
    """Measure latency when deltas are precomputed and cached.

    In production, adapter deltas can be precomputed at load time:
      delta_i = B_i^T @ A_i^T  (computed once, cached)
    Then merge = W_base + sum(alpha_i * delta_i) is pure addition.
    """
    log("\n" + "=" * 70)
    log("[Phase 5] Precomputed delta cache strategy")
    log("=" * 70)

    results = []

    for N in N_SWEEP:
        log(f"\n--- N={N} (precomputed deltas) ---")

        W_base = create_base_weight()
        A_list, B_list = create_adapters(N)
        alphas = mx.ones((N,), dtype=mx.bfloat16) / N
        mx.eval(alphas)

        # Precompute and cache deltas (one-time cost at adapter load)
        t0 = time.perf_counter()
        cached_deltas = [B_list[i].T @ A_list[i].T for i in range(N)]
        mx.eval(*cached_deltas)
        precompute_ms = (time.perf_counter() - t0) * 1000

        # Measure merge from cached deltas (the runtime cost)
        for _ in range(N_WARMUP):
            W = W_base
            for i in range(N):
                W = W + alphas[i] * cached_deltas[i]
            mx.eval(W)
            del W

        times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            W = W_base
            for i in range(N):
                W = W + alphas[i] * cached_deltas[i]
            mx.eval(W)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            del W

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))

        # Memory for cached deltas
        cache_mb = N * D_OUT * D_IN * 2 / 1e6  # bf16

        row = {
            "N": N,
            "precompute_time_ms": round(precompute_ms, 3),
            "cached_merge_mean_ms": round(mean_ms, 4),
            "cached_merge_std_ms": round(std_ms, 4),
            "delta_cache_mb": round(cache_mb, 2),
        }
        results.append(row)

        log(f"  Precompute: {precompute_ms:.3f} ms (one-time)")
        log(f"  Cached merge: {mean_ms:.3f} +/- {std_ms:.3f} ms")
        log(f"  Delta cache: {cache_mb:.1f} MB")

        cleanup(W_base, alphas, *A_list, *B_list, *cached_deltas)

    log_memory("after-phase-5")
    return results


# ============================================================================
# Analysis: fit scaling law
# ============================================================================

def fit_scaling_law(N_values, latency_values):
    """Fit T = a * N^alpha via log-linear regression.

    Returns: (a, alpha, r_squared)
    """
    log_N = np.log(np.array(N_values, dtype=np.float64))
    log_T = np.log(np.array(latency_values, dtype=np.float64))

    # Linear regression in log-log space
    coeffs = np.polyfit(log_N, log_T, 1)
    alpha = coeffs[0]
    a = np.exp(coeffs[1])

    # R-squared
    predicted = coeffs[0] * log_N + coeffs[1]
    ss_res = np.sum((log_T - predicted) ** 2)
    ss_tot = np.sum((log_T - np.mean(log_T)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(a), float(alpha), float(r_squared)


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Benchmark: Composition Latency vs N Adapters")
    log(f"d_in={D_IN}, d_out={D_OUT}, rank={RANK}, scale={LORA_SCALE}")
    log(f"N sweep: {N_SWEEP}")
    log(f"Warmup={N_WARMUP}, Measure={N_MEASURE}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Uncompiled merge sweep
    phase1 = phase_merge_latency_sweep()

    # Phase 2: Compiled merge sweep
    phase2 = phase_compiled_merge_sweep()

    # Phase 3: Multi-layer merge
    phase3 = phase_multi_layer_merge()

    # Phase 4: Bottleneck analysis
    phase4 = phase_bottleneck_analysis()

    # Phase 5: Precomputed delta cache
    phase5 = phase_precomputed_delta_cache()

    # ====================================================================
    # Analysis: Scaling law
    # ====================================================================
    log("\n" + "=" * 70)
    log("SCALING LAW ANALYSIS")
    log("=" * 70)

    N_vals = [r["N"] for r in phase1]
    merge_vals = [r["merge_mean_ms"] for r in phase1]

    a_unc, alpha_unc, r2_unc = fit_scaling_law(N_vals, merge_vals)
    log(f"\nUncompiled merge: T = {a_unc:.4f} * N^{alpha_unc:.4f}  (R2={r2_unc:.4f})")

    compiled_vals = [r["compiled_merge_mean_ms"] for r in phase2]
    a_comp, alpha_comp, r2_comp = fit_scaling_law(N_vals, compiled_vals)
    log(f"Compiled merge:   T = {a_comp:.4f} * N^{alpha_comp:.4f}  (R2={r2_comp:.4f})")

    cached_vals = [r["cached_merge_mean_ms"] for r in phase5]
    a_cache, alpha_cache, r2_cache = fit_scaling_law(N_vals, cached_vals)
    log(f"Cached merge:     T = {a_cache:.4f} * N^{alpha_cache:.4f}  (R2={r2_cache:.4f})")

    # Compile speedup at each N
    compile_speedups = []
    for p1, p2 in zip(phase1, phase2):
        speedup = p1["merge_mean_ms"] / max(p2["compiled_merge_mean_ms"], 0.001)
        compile_speedups.append(round(speedup, 3))
    log(f"\nCompile speedup by N: {dict(zip(N_vals, compile_speedups))}")

    # Cache speedup at each N
    cache_speedups = []
    for p1, p5 in zip(phase1, phase5):
        speedup = p1["merge_mean_ms"] / max(p5["cached_merge_mean_ms"], 0.001)
        cache_speedups.append(round(speedup, 3))
    log(f"Cache speedup by N:   {dict(zip(N_vals, cache_speedups))}")

    # ====================================================================
    # Kill criteria assessment
    # ====================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Pre-merge latency grows superlinearly with N
    # alpha > 1.05 => superlinear => KILL
    k1_pass = alpha_unc <= 1.05  # allow 5% tolerance
    log(f"\nK1: Pre-merge latency scaling exponent")
    log(f"  Uncompiled alpha = {alpha_unc:.4f} (threshold: <= 1.05)")
    log(f"  K1: {'PASS (linear/sublinear)' if k1_pass else 'FAIL (superlinear)'}")

    # S1: Interactive at N=25 (<50ms overhead)
    n25_result = next((r for r in phase1 if r["N"] == 25), None)
    n25_compiled = next((r for r in phase2 if r["N"] == 25), None)
    n25_cached = next((r for r in phase5 if r["N"] == 25), None)

    s1_uncompiled = n25_result["merge_mean_ms"] < 50.0 if n25_result else False
    s1_compiled = n25_compiled["compiled_merge_mean_ms"] < 50.0 if n25_compiled else False
    s1_cached = n25_cached["cached_merge_mean_ms"] < 50.0 if n25_cached else False
    s1_any = s1_uncompiled or s1_compiled or s1_cached

    log(f"\nS1: Interactive at N=25 (<50ms merge overhead)")
    if n25_result:
        log(f"  Uncompiled: {n25_result['merge_mean_ms']:.3f} ms "
            f"({'PASS' if s1_uncompiled else 'FAIL'})")
    if n25_compiled:
        log(f"  Compiled:   {n25_compiled['compiled_merge_mean_ms']:.3f} ms "
            f"({'PASS' if s1_compiled else 'FAIL'})")
    if n25_cached:
        log(f"  Cached:     {n25_cached['cached_merge_mean_ms']:.3f} ms "
            f"({'PASS' if s1_cached else 'FAIL'})")
    log(f"  S1: {'PASS' if s1_any else 'FAIL'} (any strategy)")

    # Sublinearity check
    s1_sublinear = alpha_unc < 1.0
    log(f"\n  Sublinear scaling: {s1_sublinear} (alpha={alpha_unc:.4f})")

    verdict = "SUPPORTED" if k1_pass else "KILLED"
    log(f"\nOverall verdict: {verdict}")

    # ====================================================================
    # Summary table
    # ====================================================================
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    log(f"{'N':>4} | {'Uncompiled':>10} | {'Compiled':>10} | {'Cached':>10} | "
        f"{'7-Layer':>10} | {'Compile':>8} | {'Cache':>8}")
    log(f"{'':>4} | {'merge(ms)':>10} | {'merge(ms)':>10} | {'merge(ms)':>10} | "
        f"{'merge(ms)':>10} | {'speedup':>8} | {'speedup':>8}")
    log("-" * 80)
    for i, N in enumerate(N_vals):
        unc = phase1[i]["merge_mean_ms"]
        comp = phase2[i]["compiled_merge_mean_ms"]
        cach = phase5[i]["cached_merge_mean_ms"]
        ml = phase3[i]["multilayer_merge_mean_ms"]
        log(f"{N:4d} | {unc:10.3f} | {comp:10.3f} | {cach:10.3f} | "
            f"{ml:10.3f} | {compile_speedups[i]:8.2f}x | {cache_speedups[i]:8.2f}x")

    # ====================================================================
    # Save results
    # ====================================================================
    results = {
        "experiment": "benchmark_composition_latency_sweep",
        "config": {
            "d_in": D_IN, "d_out": D_OUT, "rank": RANK,
            "lora_scale": LORA_SCALE, "n_layers": N_LAYERS,
            "n_sweep": N_SWEEP, "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
        },
        "phase1_uncompiled_sweep": phase1,
        "phase2_compiled_sweep": phase2,
        "phase3_multilayer_merge": phase3,
        "phase4_bottleneck": phase4,
        "phase5_precomputed_cache": phase5,
        "scaling_law": {
            "uncompiled": {
                "a": round(a_unc, 6), "alpha": round(alpha_unc, 4),
                "r_squared": round(r2_unc, 4),
                "formula": f"T = {a_unc:.4f} * N^{alpha_unc:.4f}",
            },
            "compiled": {
                "a": round(a_comp, 6), "alpha": round(alpha_comp, 4),
                "r_squared": round(r2_comp, 4),
                "formula": f"T = {a_comp:.4f} * N^{alpha_comp:.4f}",
            },
            "cached": {
                "a": round(a_cache, 6), "alpha": round(alpha_cache, 4),
                "r_squared": round(r2_cache, 4),
                "formula": f"T = {a_cache:.4f} * N^{alpha_cache:.4f}",
            },
        },
        "compile_speedups": dict(zip([str(n) for n in N_vals], compile_speedups)),
        "cache_speedups": dict(zip([str(n) for n in N_vals], cache_speedups)),
        "kill_criteria": {
            "K1_superlinear": not k1_pass,
            "K1_alpha": round(alpha_unc, 4),
            "K1_pass": k1_pass,
        },
        "success_criteria": {
            "S1_interactive_n25": s1_any,
            "S1_uncompiled_n25_ms": n25_result["merge_mean_ms"] if n25_result else None,
            "S1_compiled_n25_ms": n25_compiled["compiled_merge_mean_ms"] if n25_compiled else None,
            "S1_cached_n25_ms": n25_cached["cached_merge_mean_ms"] if n25_cached else None,
            "S1_sublinear": s1_sublinear,
            "S1_alpha": round(alpha_unc, 4),
        },
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
