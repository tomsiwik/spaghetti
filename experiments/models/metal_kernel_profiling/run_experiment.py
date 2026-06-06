#!/usr/bin/env python3
"""
Metal Kernel Profiling: Identify dispatch bottlenecks in MLX inference.

Profile BitNet-2B-4T forward pass to find where the 3.5x gap between
theoretical and measured throughput comes from.

Components:
1. Per-component timing (embed, attention, FFN, norms, LM head)
2. mx.compile impact (fused vs unfused)
3. Eval boundary analysis (sync points)
4. Ternary unpacking overhead isolation

Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

WARMUP_TOKENS = 50
MEASURE_TOKENS = 200
PROMPT = "The quick brown fox jumps over the lazy dog. In the field of"
SEQ_LENGTHS = [1, 64, 256]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


# ===========================================================================
# Phase 1: Baseline throughput measurement
# ===========================================================================
def measure_baseline_throughput(model, tokenizer):
    """Measure tokens/sec with KV cache (mlx_lm.generate) and without."""
    log("\n" + "=" * 70)
    log("[Phase 1] Baseline Throughput")
    log("=" * 70)

    from mlx_lm.generate import generate_step

    tokens = tokenizer.encode(PROMPT)

    # Method 1: With KV cache via generate_step
    log("  Measuring with KV cache...")
    # Warmup
    prompt = mx.array(tokens)
    for i, (token, logprobs) in enumerate(generate_step(prompt, model, max_tokens=WARMUP_TOKENS)):
        pass

    # Measure
    prompt = mx.array(tokens)
    generated = 0
    t0 = time.time()
    for i, (token, logprobs) in enumerate(generate_step(prompt, model, max_tokens=MEASURE_TOKENS)):
        mx.eval(token)
        generated += 1
    elapsed = time.time() - t0
    tok_per_sec_cached = generated / elapsed
    log(f"  KV cache: {tok_per_sec_cached:.1f} tok/s ({generated} tokens in {elapsed:.2f}s)")

    # Method 2: Single forward pass timing (no generation loop)
    log("  Measuring single forward pass (no cache)...")
    x = mx.array([tokens])
    mx.eval(x)

    # Warmup
    for _ in range(5):
        logits = model(x)
        mx.eval(logits)

    # Measure single forward
    times_single = []
    for _ in range(50):
        t0 = time.time()
        logits = model(x)
        mx.eval(logits)
        times_single.append(time.time() - t0)

    mean_single_ms = sum(times_single) / len(times_single) * 1000
    tok_per_sec_naive = 1000 / mean_single_ms  # one token per forward
    log(f"  Single forward: {mean_single_ms:.1f}ms (effective {tok_per_sec_naive:.1f} tok/s if recomputing)")
    log(f"  KV cache speedup vs recompute: {tok_per_sec_cached / tok_per_sec_naive:.1f}x")

    return {
        "tokens_per_sec_cached": round(tok_per_sec_cached, 2),
        "tokens_per_sec_no_cache": round(tok_per_sec_naive, 2),
        "ms_per_token_cached": round(elapsed / generated * 1000, 2),
        "ms_per_token_no_cache": round(mean_single_ms, 2),
        "kv_cache_speedup": round(tok_per_sec_cached / tok_per_sec_naive, 2),
        "tokens_per_sec": round(tok_per_sec_cached, 2),
        "ms_per_token": round(elapsed / generated * 1000, 2),
    }


# ===========================================================================
# Phase 2: Per-component timing
# ===========================================================================
def time_component(fn, x, n_runs=100, label=""):
    """Time a single component over n_runs, return mean time in ms."""
    # Warmup
    for _ in range(10):
        out = fn(x)
        mx.eval(out)

    times = []
    for _ in range(n_runs):
        mx.eval(x)  # ensure input is ready
        t0 = time.time()
        out = fn(x)
        mx.eval(out)
        t1 = time.time()
        times.append(t1 - t0)

    mean_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    return {
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "min_ms": round(min(times) * 1000, 4),
        "max_ms": round(max(times) * 1000, 4),
    }


def profile_components(model):
    """Profile each component of the forward pass."""
    log("\n" + "=" * 70)
    log("[Phase 2] Per-Component Timing")
    log("=" * 70)

    results = {}

    for seq_len in SEQ_LENGTHS:
        log(f"\n  Sequence length: {seq_len}")
        x = mx.random.randint(0, 1000, shape=(1, seq_len))
        mx.eval(x)

        # 1. Token embedding
        result = time_component(model.model.embed_tokens, x, label="embed")
        log(f"    Embedding: {result['mean_ms']:.4f}ms")
        results[f"embed_seq{seq_len}"] = result

        # Get embedded representation
        h = model.model.embed_tokens(x)
        mx.eval(h)

        # 2. Single layer forward
        layer = model.model.layers[0]
        result = time_component(lambda h: layer(h), h, label="layer")
        log(f"    Single layer: {result['mean_ms']:.4f}ms")
        results[f"layer_seq{seq_len}"] = result

        # 3. All layers sequential
        def all_layers(h):
            for layer in model.model.layers:
                h = layer(h)
            return h

        result = time_component(all_layers, h, n_runs=20, label="all_layers")
        log(f"    All 30 layers: {result['mean_ms']:.4f}ms")
        results[f"all_layers_seq{seq_len}"] = result

        # 4. Final norm
        h_out = all_layers(h)
        mx.eval(h_out)
        result = time_component(model.model.norm, h_out, label="final_norm")
        log(f"    Final norm: {result['mean_ms']:.4f}ms")
        results[f"final_norm_seq{seq_len}"] = result

        # 5. LM head
        h_normed = model.model.norm(h_out)
        mx.eval(h_normed)

        def lm_head_fn(h):
            if hasattr(model, 'lm_head'):
                return model.lm_head(h)
            return model.model.embed_tokens.as_linear(h)

        result = time_component(lm_head_fn, h_normed, label="lm_head")
        log(f"    LM head: {result['mean_ms']:.4f}ms")
        results[f"lm_head_seq{seq_len}"] = result

        # 6. Full forward pass
        result = time_component(lambda x: model(x), x, n_runs=20, label="full_forward")
        log(f"    Full forward: {result['mean_ms']:.4f}ms")
        results[f"full_forward_seq{seq_len}"] = result

        # Component breakdown
        layer_total = results[f"layer_seq{seq_len}"]["mean_ms"] * 30
        embed_total = results[f"embed_seq{seq_len}"]["mean_ms"]
        norm_total = results[f"final_norm_seq{seq_len}"]["mean_ms"]
        lm_head_total = results[f"lm_head_seq{seq_len}"]["mean_ms"]
        full_total = results[f"full_forward_seq{seq_len}"]["mean_ms"]
        accounted = embed_total + layer_total + norm_total + lm_head_total
        overhead = full_total - accounted

        log(f"    --- Breakdown ---")
        log(f"    Embed:      {embed_total:.3f}ms ({embed_total/full_total*100:.1f}%)")
        log(f"    30 Layers:  {layer_total:.3f}ms ({layer_total/full_total*100:.1f}%)")
        log(f"    Final norm: {norm_total:.3f}ms ({norm_total/full_total*100:.1f}%)")
        log(f"    LM head:    {lm_head_total:.3f}ms ({lm_head_total/full_total*100:.1f}%)")
        log(f"    Overhead:   {overhead:.3f}ms ({overhead/full_total*100:.1f}%)")
        log(f"    Total:      {full_total:.3f}ms")

        results[f"breakdown_seq{seq_len}"] = {
            "embed_ms": round(embed_total, 4),
            "layers_ms": round(layer_total, 4),
            "final_norm_ms": round(norm_total, 4),
            "lm_head_ms": round(lm_head_total, 4),
            "overhead_ms": round(overhead, 4),
            "full_ms": round(full_total, 4),
            "layers_pct": round(layer_total / full_total * 100, 1),
            "overhead_pct": round(overhead / full_total * 100, 1),
        }

        del h, h_out, h_normed, x

    return results


# ===========================================================================
# Phase 3: mx.compile impact
# ===========================================================================
def profile_compilation(model):
    """Test mx.compile impact on inference speed."""
    log("\n" + "=" * 70)
    log("[Phase 3] mx.compile Impact")
    log("=" * 70)

    results = {}

    for seq_len in SEQ_LENGTHS:
        log(f"\n  Sequence length: {seq_len}")
        x = mx.random.randint(0, 1000, shape=(1, seq_len))
        mx.eval(x)

        # Uncompiled
        def forward(x):
            return model(x)

        result_uncompiled = time_component(forward, x, n_runs=50, label="uncompiled")
        log(f"    Uncompiled: {result_uncompiled['mean_ms']:.3f}ms")

        # Compiled
        compiled_forward = mx.compile(forward)
        result_compiled = time_component(compiled_forward, x, n_runs=50, label="compiled")
        log(f"    Compiled:   {result_compiled['mean_ms']:.3f}ms")

        speedup = result_uncompiled["mean_ms"] / max(result_compiled["mean_ms"], 0.001)
        log(f"    Speedup:    {speedup:.2f}x")

        results[f"seq{seq_len}"] = {
            "uncompiled_ms": result_uncompiled["mean_ms"],
            "compiled_ms": result_compiled["mean_ms"],
            "speedup": round(speedup, 3),
        }

        del x

    return results


# ===========================================================================
# Phase 4: Eval boundary analysis
# ===========================================================================
def profile_eval_boundaries(model):
    """Count how many eval sync points occur during generation."""
    log("\n" + "=" * 70)
    log("[Phase 4] Eval Boundary Analysis")
    log("=" * 70)

    # Standard generation: one eval per token (for next_token)
    # But internally, how many ops get dispatched?

    x = mx.random.randint(0, 1000, shape=(1, 64))
    mx.eval(x)

    # Test: single eval at end vs eval-per-op
    results = {}

    # Method 1: eval only final logits (lazy graph)
    def single_eval_forward(x):
        logits = model(x)
        return logits[:, -1, :]

    t0 = time.time()
    for _ in range(100):
        out = single_eval_forward(x)
        mx.eval(out)
    single_eval_time = (time.time() - t0) / 100 * 1000
    log(f"  Single eval (lazy): {single_eval_time:.3f}ms")

    # Method 2: eval after every layer (eager)
    def eager_forward(x):
        h = model.model.embed_tokens(x)
        mx.eval(h)
        for layer in model.model.layers:
            h = layer(h)
            mx.eval(h)
        h = model.model.norm(h)
        mx.eval(h)
        logits = model.model.embed_tokens.as_linear(h)
        return logits[:, -1, :]

    t0 = time.time()
    for _ in range(100):
        out = eager_forward(x)
        mx.eval(out)
    eager_eval_time = (time.time() - t0) / 100 * 1000
    log(f"  Eager eval (per layer): {eager_eval_time:.3f}ms")

    overhead_pct = (eager_eval_time / single_eval_time - 1) * 100
    log(f"  Eval boundary overhead: {overhead_pct:.1f}%")

    results["single_eval_ms"] = round(single_eval_time, 3)
    results["eager_eval_ms"] = round(eager_eval_time, 3)
    results["eval_overhead_pct"] = round(overhead_pct, 1)
    results["n_eval_calls_eager"] = 32  # 30 layers + embed + norm

    return results


# ===========================================================================
# Phase 5: Ternary unpacking overhead
# ===========================================================================
def profile_ternary_overhead(model):
    """Measure overhead of ternary weight handling."""
    log("\n" + "=" * 70)
    log("[Phase 5] Weight Format Analysis")
    log("=" * 70)

    # Check if model uses BitLinear (packed ternary) or Linear (unpacked)
    results = {}

    layer = model.model.layers[0]
    q_proj = layer.self_attn.q_proj

    # Check type
    from mlx_lm.models.bitlinear_layers import BitLinear
    is_bitlinear = isinstance(q_proj, BitLinear)
    log(f"  Weight format: {'BitLinear (packed ternary)' if is_bitlinear else 'Linear (unpacked)'}")

    if is_bitlinear:
        # Measure unpack time
        w = q_proj.weight
        out_features = q_proj.out_features
        weight_scale = q_proj.weight_scale
        invert = q_proj.invert_weight_scales

        def unpack():
            w0 = (w & 3).astype(mx.bfloat16) - 1
            w1 = ((w >> 2) & 3).astype(mx.bfloat16) - 1
            w2 = ((w >> 4) & 3).astype(mx.bfloat16) - 1
            w3 = ((w >> 6) & 3).astype(mx.bfloat16) - 1
            unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
            scale = weight_scale.astype(mx.bfloat16)
            if invert:
                return unpacked / scale
            else:
                return unpacked * scale

        # Time unpacking
        for _ in range(10):
            out = unpack()
            mx.eval(out)

        times = []
        for _ in range(100):
            t0 = time.time()
            out = unpack()
            mx.eval(out)
            times.append(time.time() - t0)

        mean_ms = sum(times) / len(times) * 1000
        log(f"  Unpack time (single q_proj): {mean_ms:.4f}ms")
        log(f"  Estimated per-layer (7 projections): {mean_ms * 7:.4f}ms")
        log(f"  Estimated total (30 layers): {mean_ms * 7 * 30:.4f}ms")

        results["format"] = "bitlinear_packed"
        results["unpack_single_ms"] = round(mean_ms, 4)
        results["unpack_per_layer_ms"] = round(mean_ms * 7, 4)
        results["unpack_total_ms"] = round(mean_ms * 7 * 30, 4)
    else:
        results["format"] = "linear_unpacked"
        results["note"] = "Model was loaded with unpacked weights, no unpacking overhead"

        # Measure matmul time for comparison
        w = q_proj.weight  # (d, d) or similar
        h = mx.random.normal(shape=(1, 1, q_proj.weight.shape[1]))
        mx.eval(h)

        def matmul():
            return h @ w.T

        for _ in range(10):
            out = matmul()
            mx.eval(out)

        times = []
        for _ in range(100):
            t0 = time.time()
            out = matmul()
            mx.eval(out)
            times.append(time.time() - t0)

        mean_ms = sum(times) / len(times) * 1000
        log(f"  Single matmul (q_proj, seq=1): {mean_ms:.4f}ms")
        log(f"  Estimated per-layer (7 projections): {mean_ms * 7:.4f}ms")
        log(f"  Estimated total (30 layers): {mean_ms * 7 * 30:.4f}ms")

        results["matmul_single_ms"] = round(mean_ms, 4)
        results["matmul_per_layer_ms"] = round(mean_ms * 7, 4)
        results["matmul_total_ms"] = round(mean_ms * 7 * 30, 4)

    return results


# ===========================================================================
# Phase 6: Memory bandwidth analysis
# ===========================================================================
def profile_memory_bandwidth():
    """Estimate achieved memory bandwidth vs theoretical."""
    log("\n" + "=" * 70)
    log("[Phase 6] Memory Bandwidth Analysis")
    log("=" * 70)

    results = {}

    # Test with different sized tensors to find bandwidth curve
    sizes_mb = [1, 10, 50, 100, 500, 1000]

    for size_mb in sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 4  # float32
        a = mx.random.normal(shape=(n_elements,))
        b = mx.random.normal(shape=(n_elements,))
        mx.eval(a, b)

        # Measure: c = a + b (reads 2 arrays, writes 1 = 3x data movement)
        for _ in range(5):
            c = a + b
            mx.eval(c)

        times = []
        for _ in range(50):
            t0 = time.time()
            c = a + b
            mx.eval(c)
            times.append(time.time() - t0)

        mean_s = sum(times) / len(times)
        bytes_moved = 3 * n_elements * 4  # read a, read b, write c
        bandwidth_gbps = bytes_moved / mean_s / 1e9

        log(f"  {size_mb}MB: {bandwidth_gbps:.1f} GB/s ({mean_s*1000:.3f}ms)")
        results[f"{size_mb}mb"] = {
            "bandwidth_gbps": round(bandwidth_gbps, 2),
            "time_ms": round(mean_s * 1000, 3),
        }

        del a, b, c

    gc.collect()
    mx.clear_cache()

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    log("=" * 70)
    log("Metal Kernel Profiling Experiment")
    log(f"Platform: {mx.default_device()}")
    log(f"Device: {device.get('device_name', 'unknown')}")
    log("=" * 70)

    t_start = time.time()
    results = {}

    # Phase 6 first (no model needed)
    results["memory_bandwidth"] = profile_memory_bandwidth()

    # Load model
    log("\n  Loading model...")
    from mlx_lm import load
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    log_memory("after-load")

    # Check model architecture
    n_layers = len(model.model.layers)
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[-1]
    log(f"  Architecture: {n_layers} layers, d_model inferred from weight shape")

    # Phase 5: Ternary overhead (before unpacking)
    results["ternary_overhead"] = profile_ternary_overhead(model)

    # Phase 1: Baseline throughput
    results["baseline"] = measure_baseline_throughput(model, tokenizer)

    # Phase 2: Component timing
    results["components"] = profile_components(model)

    # Phase 3: Compilation
    results["compilation"] = profile_compilation(model)

    # Phase 4: Eval boundaries
    results["eval_boundaries"] = profile_eval_boundaries(model)

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    baseline_ms = results["baseline"]["ms_per_token"]
    log(f"  Baseline: {results['baseline']['tokens_per_sec']:.1f} tok/s ({baseline_ms:.1f}ms/tok)")

    # Theoretical
    bw_1gb = results["memory_bandwidth"].get("1000mb", results["memory_bandwidth"].get("500mb", {}))
    peak_bw = bw_1gb.get("bandwidth_gbps", 273)
    model_size_gb = 1.7  # packed ternary BitNet-2B-4T
    theoretical_ms = model_size_gb / peak_bw * 1000
    theoretical_tok_s = 1000 / theoretical_ms
    gap = baseline_ms / theoretical_ms
    log(f"  Theoretical: {theoretical_tok_s:.0f} tok/s ({theoretical_ms:.1f}ms/tok, {peak_bw:.0f} GB/s bandwidth)")
    log(f"  Gap: {gap:.1f}x")

    results["summary"] = {
        "baseline_tok_s": results["baseline"]["tokens_per_sec"],
        "baseline_ms_per_tok": baseline_ms,
        "measured_bandwidth_gbps": peak_bw,
        "theoretical_ms_per_tok": round(theoretical_ms, 2),
        "theoretical_tok_s": round(theoretical_tok_s, 1),
        "gap": round(gap, 2),
    }

    # Compile speedup at seq=1
    if "seq1" in results["compilation"]:
        compile_speedup = results["compilation"]["seq1"]["speedup"]
        log(f"  mx.compile speedup (seq=1): {compile_speedup:.2f}x")
        results["summary"]["compile_speedup_seq1"] = compile_speedup

    # Eval boundary overhead
    eval_overhead = results["eval_boundaries"]["eval_overhead_pct"]
    log(f"  Eval boundary overhead: {eval_overhead:.1f}%")
    results["summary"]["eval_boundary_overhead_pct"] = eval_overhead

    results["total_time_s"] = round(time.time() - t_start, 1)

    # Kill criteria
    k1_pass = True  # We successfully profiled
    results["kill_criteria"] = {"K1_can_profile": k1_pass}

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
