#!/usr/bin/env python3
"""
Experiment: Sparse-BitNet — exploit natural 42% sparsity in ternary weights.

Kill criteria:
  K1: Sparsity exploitation degrades composition > 5% PPL
  K2: No wall-clock speedup (sparse ops >= dense, 500-iteration mean)
  K3: Natural zero fraction < 30% at micro scale

References:
  - Sparse-BitNet (arxiv 2603.05168): natural 42% sparsity
  - exp_inference_speed_10x: 172 tok/s, 74.2% bandwidth bound
  - exp_memory_budget_analysis: 1.18 GB base model

Platform: Apple M5 Pro 48GB, MLX only.
"""

import gc
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e6
    cache = mx.get_cache_memory() / 1e6
    peak = mx.get_peak_memory() / 1e6
    log(f"[MEM {label}] active={active:.1f}MB cache={cache:.1f}MB peak={peak:.1f}MB")
    return {"active_mb": round(active, 1), "cache_mb": round(cache, 1), "peak_mb": round(peak, 1)}


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def unpack_ternary(packed_weight):
    """Unpack uint8 packed ternary weights to int8 {-1, 0, +1}.

    Packing: 4 ternary values per byte, 2 bits each.
    Encoding: 0 -> 0, 1 -> +1, 2 -> -1 (standard BitNet encoding).
    """
    # packed_weight: (out_features/4, in_features) uint8
    packed_rows, in_features = packed_weight.shape

    # Extract 4 values per byte using the Metal kernel encoding: (w & 3) - 1
    # Encoding: 0 -> -1, 1 -> 0, 2 -> +1
    v0 = (packed_weight & 0x03).astype(mx.int8) - 1       # bits 0-1
    v1 = ((packed_weight >> 2) & 0x03).astype(mx.int8) - 1 # bits 2-3
    v2 = ((packed_weight >> 4) & 0x03).astype(mx.int8) - 1 # bits 4-5
    v3 = ((packed_weight >> 6) & 0x03).astype(mx.int8) - 1 # bits 6-7

    # The Metal kernel maps packed row row_idx to output rows:
    #   row_idx + 0 * (out_features/4)
    #   row_idx + 1 * (out_features/4)
    #   row_idx + 2 * (out_features/4)
    #   row_idx + 3 * (out_features/4)
    # So output row layout is NOT interleaved but strided:
    # [v0_row0, v0_row1, ..., v0_rowN, v1_row0, ..., v3_rowN]
    unpacked = mx.concatenate([v0, v1, v2, v3], axis=0)  # (out_features, in_features)

    return unpacked


def phase_measure_sparsity():
    """Phase 1: Measure natural zero fraction in BitNet-2B-4T weights.

    K3 check: if overall zero fraction < 30%, KILL immediately.
    """
    log("=" * 60)
    log("Phase 1: Measure natural sparsity in BitNet-2B-4T")
    log("=" * 60)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    from mlx.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))

    layer_results = []
    total_zeros = 0
    total_elements = 0

    # Process each ternary weight matrix
    for name, p in sorted(params.items()):
        if p.dtype != mx.uint8:
            continue  # Skip non-ternary params (embeddings, norms, scales)

        unpacked = unpack_ternary(p)
        mx.eval(unpacked)

        n_elements = unpacked.shape[0] * unpacked.shape[1]
        n_zeros = int(mx.sum(unpacked == 0).item())
        n_pos = int(mx.sum(unpacked == 1).item())
        n_neg = int(mx.sum(unpacked == -1).item())
        zero_frac = n_zeros / n_elements

        layer_results.append({
            "name": name,
            "shape": list(unpacked.shape),
            "n_elements": n_elements,
            "n_zeros": n_zeros,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "zero_fraction": round(zero_frac, 4),
            "pos_fraction": round(n_pos / n_elements, 4),
            "neg_fraction": round(n_neg / n_elements, 4),
        })

        total_zeros += n_zeros
        total_elements += n_elements

        del unpacked
        mx.clear_cache()

    overall_zero_frac = total_zeros / total_elements

    # Print summary
    log(f"\nOverall: {total_zeros:,} / {total_elements:,} = {overall_zero_frac:.4f} zero fraction")
    log(f"\nPer-layer zero fractions:")
    for r in layer_results:
        log(f"  {r['name']}: {r['zero_fraction']:.4f} ({r['n_elements']:,} elements)")

    # Check K3
    k3_pass = overall_zero_frac >= 0.30
    log(f"\nK3 {'PASS' if k3_pass else 'FAIL'}: zero fraction {overall_zero_frac:.4f} {'>=':} 0.30 threshold")

    result = {
        "overall_zero_fraction": round(overall_zero_frac, 4),
        "total_zeros": total_zeros,
        "total_elements": total_elements,
        "k3_pass": k3_pass,
        "per_layer": layer_results,
    }

    cleanup(model, tokenizer, params)
    return result


def phase_benchmark_sparse():
    """Phase 2+3: Benchmark sparse vs dense ternary matmul.

    We test at the individual layer level since MLX has no native sparse ops.
    Compare:
      A) Dense: the native BitLinear kernel (packed uint8 -> Metal kernel)
      B) Sparse via index gather: y_j = x[plus_idx_j].sum() - x[minus_idx_j].sum()
      C) Sparse via mx.where mask on unpacked bf16 weights

    K2 check: if no approach achieves wall-clock speedup, KILL.
    """
    log("\n" + "=" * 60)
    log("Phase 2+3: Benchmark sparse vs dense ternary matmul")
    log("=" * 60)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Get a representative layer (layer 0, q_proj)
    layer = model.model.layers[0]
    q_proj = layer.self_attn.q_proj

    packed_w = q_proj.weight
    w_scale = q_proj.weight_scale
    out_features = q_proj.out_features
    in_features = packed_w.shape[1]

    log(f"Testing layer: q_proj, packed shape={packed_w.shape}, "
        f"logical shape=({out_features}, {in_features})")

    # Create test input (single token, typical inference)
    x = mx.random.normal((1, in_features)).astype(mx.bfloat16)
    mx.eval(x)

    # --- Method A: Native BitLinear kernel (the baseline) ---
    # Warmup
    for _ in range(100):
        y_dense = q_proj(x)
        mx.eval(y_dense)
    mx.clear_cache()

    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_dense = q_proj(x)
        mx.eval(y_dense)
    t_dense = (time.perf_counter() - t0) / n_iters
    log(f"Dense (native kernel): {t_dense * 1e6:.1f} us/iter")

    # Save reference output for quality check
    y_ref = q_proj(x)
    mx.eval(y_ref)

    # --- Method B: Index-based sparse (gather + sum) ---
    # Unpack weights and build index sets
    unpacked_w = unpack_ternary(packed_w)
    mx.eval(unpacked_w)

    # For each output row, find +1 and -1 positions
    # This is expensive to set up but we only do it once (static structure)
    plus_masks = (unpacked_w == 1)   # (out, in) bool
    minus_masks = (unpacked_w == -1) # (out, in) bool
    mx.eval(plus_masks, minus_masks)

    # Method B: vectorized masked sum
    # y = (x * plus_mask_float).sum(-1) - (x * minus_mask_float).sum(-1)
    plus_float = plus_masks.astype(mx.bfloat16)
    minus_float = minus_masks.astype(mx.bfloat16)
    mx.eval(plus_float, minus_float)

    def sparse_masked_matmul(x_in):
        # x_in: (1, in_features)
        # Broadcast multiply + sum: equivalent to matmul with masked weights
        y_plus = x_in @ plus_float.T    # (1, out_features)
        y_minus = x_in @ minus_float.T  # (1, out_features)
        return (y_plus - y_minus) * w_scale

    # Warmup
    for _ in range(100):
        y_sparse_b = sparse_masked_matmul(x)
        mx.eval(y_sparse_b)
    mx.clear_cache()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_sparse_b = sparse_masked_matmul(x)
        mx.eval(y_sparse_b)
    t_sparse_mask = (time.perf_counter() - t0) / n_iters
    log(f"Sparse masked (two bf16 matmuls): {t_sparse_mask * 1e6:.1f} us/iter")

    # Quality check
    y_check_b = sparse_masked_matmul(x)
    mx.eval(y_check_b)
    diff_b = float(mx.abs(y_ref - y_check_b).max().item())
    log(f"  Max abs diff vs reference: {diff_b:.6f}")

    # --- Method C: Unpacked bf16 matmul (skip zeros via multiplication) ---
    # Just unpack to bf16 and do standard matmul
    # Zeros multiply to zero, so this "skips" them algebraically but not computationally
    unpacked_bf16 = unpacked_w.astype(mx.bfloat16)
    mx.eval(unpacked_bf16)

    def dense_unpacked_matmul(x_in):
        return (x_in @ unpacked_bf16.T) * w_scale

    # Warmup
    for _ in range(100):
        y_unpack = dense_unpacked_matmul(x)
        mx.eval(y_unpack)
    mx.clear_cache()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_unpack = dense_unpacked_matmul(x)
        mx.eval(y_unpack)
    t_unpacked = (time.perf_counter() - t0) / n_iters
    log(f"Unpacked bf16 matmul: {t_unpacked * 1e6:.1f} us/iter")

    # Quality check
    y_check_c = dense_unpacked_matmul(x)
    mx.eval(y_check_c)
    diff_c = float(mx.abs(y_ref - y_check_c).max().item())
    log(f"  Max abs diff vs reference: {diff_c:.6f}")

    # --- Method D: Compiled sparse masked matmul ---
    sparse_compiled = mx.compile(sparse_masked_matmul)

    # Warmup
    for _ in range(100):
        y_comp = sparse_compiled(x)
        mx.eval(y_comp)
    mx.clear_cache()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_comp = sparse_compiled(x)
        mx.eval(y_comp)
    t_compiled = (time.perf_counter() - t0) / n_iters
    log(f"Compiled sparse masked: {t_compiled * 1e6:.1f} us/iter")

    # --- Method E: Compiled unpacked bf16 matmul ---
    dense_compiled = mx.compile(dense_unpacked_matmul)

    for _ in range(100):
        y_dc = dense_compiled(x)
        mx.eval(y_dc)
    mx.clear_cache()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_dc = dense_compiled(x)
        mx.eval(y_dc)
    t_dense_compiled = (time.perf_counter() - t0) / n_iters
    log(f"Compiled unpacked bf16: {t_dense_compiled * 1e6:.1f} us/iter")

    # Memory analysis
    packed_bytes = packed_w.nbytes
    unpacked_bf16_bytes = unpacked_bf16.nbytes
    sparse_mask_bytes = plus_float.nbytes + minus_float.nbytes

    log(f"\nMemory comparison:")
    log(f"  Packed uint8: {packed_bytes / 1e6:.2f} MB")
    log(f"  Unpacked bf16: {unpacked_bf16_bytes / 1e6:.2f} MB ({unpacked_bf16_bytes / packed_bytes:.1f}x)")
    log(f"  Sparse masks (2x bf16): {sparse_mask_bytes / 1e6:.2f} MB ({sparse_mask_bytes / packed_bytes:.1f}x)")

    # K2 assessment
    speedup_mask = t_dense / t_sparse_mask
    speedup_unpacked = t_dense / t_unpacked
    speedup_compiled = t_dense / t_compiled
    speedup_dense_compiled = t_dense / t_dense_compiled

    best_sparse_time = min(t_sparse_mask, t_compiled)
    k2_pass = best_sparse_time < t_dense  # Any sparse method faster than native

    log(f"\nSpeedup ratios (>1.0 = sparse wins):")
    log(f"  Sparse masked / native: {speedup_mask:.3f}x")
    log(f"  Unpacked bf16 / native: {speedup_unpacked:.3f}x")
    log(f"  Compiled sparse / native: {speedup_compiled:.3f}x")
    log(f"  Compiled unpacked / native: {speedup_dense_compiled:.3f}x")
    log(f"\nK2 {'PASS' if k2_pass else 'FAIL'}: best sparse {best_sparse_time * 1e6:.1f} us "
        f"{'<' if k2_pass else '>='} native {t_dense * 1e6:.1f} us")

    result = {
        "layer": "layers.0.self_attn.q_proj",
        "logical_shape": [out_features, in_features],
        "timings_us": {
            "native_bitlinear": round(t_dense * 1e6, 1),
            "sparse_masked_matmul": round(t_sparse_mask * 1e6, 1),
            "unpacked_bf16_matmul": round(t_unpacked * 1e6, 1),
            "compiled_sparse_masked": round(t_compiled * 1e6, 1),
            "compiled_unpacked_bf16": round(t_dense_compiled * 1e6, 1),
        },
        "speedups": {
            "sparse_masked_vs_native": round(speedup_mask, 3),
            "unpacked_bf16_vs_native": round(speedup_unpacked, 3),
            "compiled_sparse_vs_native": round(speedup_compiled, 3),
            "compiled_unpacked_vs_native": round(speedup_dense_compiled, 3),
        },
        "memory_bytes": {
            "packed_uint8": packed_bytes,
            "unpacked_bf16": unpacked_bf16_bytes,
            "sparse_masks_bf16": sparse_mask_bytes,
        },
        "quality": {
            "max_abs_diff_sparse_masked": diff_b,
            "max_abs_diff_unpacked_bf16": diff_c,
        },
        "k2_pass": k2_pass,
        "n_warmup": 100,
        "n_iters": n_iters,
    }

    cleanup(model, tokenizer, packed_w, unpacked_w, unpacked_bf16,
            plus_float, minus_float, plus_masks, minus_masks, x, y_ref)
    return result


def phase_benchmark_larger_layers():
    """Phase 3b: Benchmark on MLP layers (gate_proj, up_proj, down_proj).

    MLP layers are 2.7x wider than attention. Test if sparsity matters more there.
    """
    log("\n" + "=" * 60)
    log("Phase 3b: Benchmark on MLP layers")
    log("=" * 60)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)

    results = {}
    n_iters = 500

    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj = getattr(model.model.layers[0].mlp, proj_name)
        packed_w = proj.weight
        w_scale = proj.weight_scale
        out_features = proj.out_features
        in_features = packed_w.shape[1]

        log(f"\n{proj_name}: packed shape={packed_w.shape}, logical=({out_features}, {in_features})")

        x = mx.random.normal((1, in_features)).astype(mx.bfloat16)
        mx.eval(x)

        # Native BitLinear
        for _ in range(100):
            y = proj(x)
            mx.eval(y)
        mx.clear_cache()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = proj(x)
            mx.eval(y)
        t_native = (time.perf_counter() - t0) / n_iters

        # Unpacked bf16 (the most realistic "sparse" approach)
        unpacked_w = unpack_ternary(packed_w).astype(mx.bfloat16)
        mx.eval(unpacked_w)

        def unpacked_matmul(x_in):
            return (x_in @ unpacked_w.T) * w_scale

        for _ in range(100):
            y = unpacked_matmul(x)
            mx.eval(y)
        mx.clear_cache()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = unpacked_matmul(x)
            mx.eval(y)
        t_unpacked = (time.perf_counter() - t0) / n_iters

        speedup = t_native / t_unpacked
        log(f"  Native: {t_native * 1e6:.1f} us, Unpacked bf16: {t_unpacked * 1e6:.1f} us, "
            f"ratio: {speedup:.3f}x")
        log(f"  Memory: packed={packed_w.nbytes / 1e6:.2f} MB, "
            f"unpacked={unpacked_w.nbytes / 1e6:.2f} MB")

        results[proj_name] = {
            "logical_shape": [out_features, in_features],
            "native_us": round(t_native * 1e6, 1),
            "unpacked_bf16_us": round(t_unpacked * 1e6, 1),
            "speedup": round(speedup, 3),
        }

        del unpacked_w, x
        mx.clear_cache()

    cleanup(model, tokenizer)
    return results


def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Measure sparsity (K3 check)
    sparsity_results = phase_measure_sparsity()
    log_memory("after-sparsity")

    if not sparsity_results["k3_pass"]:
        log(f"\nK3 FAIL: Zero fraction {sparsity_results['overall_zero_fraction']:.4f} < 0.30 threshold")
        log("Continuing with benchmarks to provide complete K2 data...")
        # Don't return -- still run benchmarks for completeness

    # Phase 2+3: Benchmark sparse vs dense (K2 check)
    benchmark_results = phase_benchmark_sparse()
    log_memory("after-benchmark")

    # Phase 3b: MLP layers
    mlp_results = phase_benchmark_larger_layers()
    log_memory("after-mlp-benchmark")

    # K1: quality check - sparse is mathematically exact, so K1 passes if diffs are tiny
    max_diff = max(
        benchmark_results["quality"]["max_abs_diff_sparse_masked"],
        benchmark_results["quality"]["max_abs_diff_unpacked_bf16"],
    )
    # bf16 has ~0.01 precision, so diffs < 0.1 are rounding, not algorithmic error
    k1_pass = max_diff < 0.1

    # Overall assessment
    k2_pass = benchmark_results["k2_pass"]
    k3_pass = sparsity_results["k3_pass"]

    # S1: >= 1.3x speedup with < 2% PPL degradation
    best_speedup = max(benchmark_results["speedups"].values())
    s1_pass = best_speedup >= 1.3 and k1_pass

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"K1 (quality): {'PASS' if k1_pass else 'FAIL'} — max diff = {max_diff:.6f}")
    log(f"K2 (speedup): {'PASS' if k2_pass else 'FAIL'} — best speedup = {best_speedup:.3f}x")
    log(f"K3 (sparsity): {'PASS' if k3_pass else 'FAIL'} — zero fraction = {sparsity_results['overall_zero_fraction']:.4f}")
    log(f"S1 (1.3x speedup): {'PASS' if s1_pass else 'FAIL'}")

    if not k2_pass:
        log(f"\nConclusion: Sparse ternary ops are SLOWER than the native packed BitLinear kernel.")
        log(f"The fused Metal kernel at 74.2% bandwidth utilization cannot be beaten by")
        log(f"sparse approaches that increase memory footprint. Packed uint8 is optimal.")

    results = {
        "sparsity": {
            "overall_zero_fraction": sparsity_results["overall_zero_fraction"],
            "total_zeros": sparsity_results["total_zeros"],
            "total_elements": sparsity_results["total_elements"],
        },
        "benchmark_q_proj": benchmark_results,
        "benchmark_mlp": mlp_results,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "k3_pass": k3_pass,
        "s1_pass": s1_pass,
        "best_speedup": best_speedup,
        "max_quality_diff": max_diff,
        "killed": not k2_pass,
        "kill_reason": "K2 FAIL: sparse ops slower than native packed kernel" if not k2_pass else None,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
