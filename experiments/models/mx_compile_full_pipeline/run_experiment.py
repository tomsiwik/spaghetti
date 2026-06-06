#!/usr/bin/env python3
"""
Experiment: mx.compile Full Pipeline Optimization

Tests mx.compile on individual components and the composed pipeline:
  Phase 1: Baseline uncompiled tok/s (single adapter)
  Phase 2: Compile routing head inference
  Phase 3: Compile LoRA delta computation
  Phase 4: Compile multi-adapter LoRA with fixed N
  Phase 5: Dynamic adapter selection (varying N) - recompilation cost
  Phase 6: Full pipeline: compiled routing + compiled LoRA

Kill criteria:
  K1 (#258): Compilation fails on dynamic adapter selection
  K2 (#259): No speedup (< 5% improvement)

Success criteria:
  S1 (#26): >20% throughput improvement from compilation

Platform: Apple M5 Pro 48GB, MLX 0.31.1
"""

import gc
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0

DOMAINS = ["medical", "code", "math", "legal", "finance"]

TEST_PROMPT = "What are the symptoms of diabetes?"

# Timing parameters
N_WARMUP = 3
N_MEASURE = 10
N_TOKENS = 100


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


# ============================================================================
# Timing utilities
# ============================================================================

def time_function(fn, *args, n_warmup=N_WARMUP, n_measure=N_MEASURE, label=""):
    """Time a function with warmup and multiple measurements.
    Returns dict with mean, std, min, max in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
        mx.eval(result)
        del result

    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        result = fn(*args)
        mx.eval(result)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        del result

    times_arr = np.array(times)
    stats = {
        "mean_ms": round(float(np.mean(times_arr)), 3),
        "std_ms": round(float(np.std(times_arr)), 3),
        "min_ms": round(float(np.min(times_arr)), 3),
        "max_ms": round(float(np.max(times_arr)), 3),
        "n_samples": n_measure,
    }
    if label:
        log(f"  {label}: {stats['mean_ms']:.3f} +/- {stats['std_ms']:.3f} ms")
    return stats


def measure_tps(model, tokenizer, prompt_text, n_tokens=N_TOKENS, n_warmup=2, **kwargs):
    """Measure tok/s using stream_generate and wall-clock."""
    from mlx_lm import stream_generate, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    # Warmup
    for _ in range(n_warmup):
        for resp in stream_generate(model, tokenizer, prompt_text,
                                     max_tokens=5, sampler=sampler, **kwargs):
            pass
        mx.clear_cache()

    # Internal timing
    gen_tps = prompt_tps = 0
    for resp in stream_generate(model, tokenizer, prompt_text,
                                 max_tokens=n_tokens, sampler=sampler, **kwargs):
        gen_tps = resp.generation_tps
        prompt_tps = resp.prompt_tps
    mx.clear_cache()

    # Wall-clock
    _ = mlx_generate(model, tokenizer, prompt_text, max_tokens=5,
                     sampler=sampler, verbose=False)
    mx.clear_cache()

    t0 = time.perf_counter()
    _ = mlx_generate(model, tokenizer, prompt_text, max_tokens=n_tokens,
                     sampler=sampler, verbose=False)
    wallclock = time.perf_counter() - t0
    mx.clear_cache()

    return {
        "generation_tps": round(gen_tps, 1),
        "prompt_tps": round(prompt_tps, 1),
        "wallclock_tps": round(n_tokens / wallclock, 1) if wallclock > 0 else 0,
    }


# ============================================================================
# Phase 1: Baseline (uncompiled) model with single adapter
# ============================================================================

def phase_baseline():
    """Baseline: load model, wrap with runtime LoRA, measure tok/s."""
    log("\n" + "=" * 60)
    log("[Phase 1] Baseline (uncompiled) single-adapter tok/s")
    log("=" * 60)
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Base speed
    base_tps = measure_tps(model, tokenizer, TEST_PROMPT, n_warmup=2)
    log(f"  Base (no adapter): {base_tps['generation_tps']} tok/s (internal), "
        f"{base_tps['wallclock_tps']} tok/s (wallclock)")

    # Load and apply single adapter (medical)
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))
    adapter_bf16 = {k: v.astype(mx.bfloat16) for k, v in adapter.items()}
    del adapter
    mx.eval(list(adapter_bf16.values()))

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    a_matrices = {}
    for k in skeleton:
        if k.endswith("_domain_0"):
            a_matrices[k] = mx.array(skeleton[k]).astype(mx.bfloat16)
    del skeleton
    mx.eval(list(a_matrices.values()))
    gc.collect()

    TARGET_KEYS = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    class BitLinearWithLoRA(nn.Module):
        """Wraps BitLinear to add runtime LoRA: y = BitLinear(x) + x @ A @ B * scale"""
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            lora_out = (x @ self.lora_a) @ self.lora_b * self.lora_scale
            return y + lora_out

    n_wrapped = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, BitLinear):
                continue

            skey = f"layer_{li}_{key}_domain_0"
            b_key = f"model.layers.{li}.{key}.lora_b"
            if skey not in a_matrices or b_key not in adapter_bf16:
                continue

            wrapped = BitLinearWithLoRA(module, a_matrices[skey], adapter_bf16[b_key], LORA_SCALE)
            updates.append((key, wrapped))
            n_wrapped += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    log(f"  Wrapped {n_wrapped} layers with runtime LoRA (uncompiled)")
    mem = log_memory("after-wrap")

    adapted_tps = measure_tps(model, tokenizer, TEST_PROMPT, n_warmup=2)
    log(f"  Adapted (uncompiled): {adapted_tps['generation_tps']} tok/s (internal), "
        f"{adapted_tps['wallclock_tps']} tok/s (wallclock)")

    results = {
        "base_tps": base_tps,
        "adapted_tps": adapted_tps,
        "n_wrapped": n_wrapped,
        "memory": mem,
    }

    cleanup(model, tokenizer, adapter_bf16, a_matrices)
    return results


# ============================================================================
# Phase 2: Compiled routing head
# ============================================================================

def phase_compiled_routing():
    """Test mx.compile on a routing head (small linear + sigmoid)."""
    log("\n" + "=" * 60)
    log("[Phase 2] Compiled Routing Head")
    log("=" * 60)
    cleanup()

    D = 2560
    N_EXPERTS = 5

    # Simulate routing head: x @ W + b -> sigmoid
    W_route = mx.random.normal((D, N_EXPERTS)).astype(mx.bfloat16) * 0.01
    b_route = mx.zeros((N_EXPERTS,), dtype=mx.bfloat16)
    mx.eval(W_route, b_route)

    # Input: single token embedding
    x = mx.random.normal((1, D)).astype(mx.bfloat16)
    mx.eval(x)

    # Uncompiled routing
    def route_uncompiled(x, W, b):
        return mx.sigmoid(x @ W + b)

    # Compiled routing
    route_compiled = mx.compile(route_uncompiled)

    stats_uncompiled = time_function(route_uncompiled, x, W_route, b_route,
                                      label="Routing (uncompiled)")
    stats_compiled = time_function(route_compiled, x, W_route, b_route,
                                    label="Routing (compiled)")

    speedup = stats_uncompiled["mean_ms"] / stats_compiled["mean_ms"] if stats_compiled["mean_ms"] > 0 else 0

    log(f"  Speedup: {speedup:.2f}x")

    # Test with varying batch sizes (fixed shapes vs dynamic)
    dynamic_results = []
    for batch in [1, 4, 16]:
        x_batch = mx.random.normal((batch, D)).astype(mx.bfloat16)
        mx.eval(x_batch)

        # Compiled version will recompile for new shape
        t0 = time.perf_counter()
        result = route_compiled(x_batch, W_route, b_route)
        mx.eval(result)
        first_call_ms = (time.perf_counter() - t0) * 1000

        # Second call (cached)
        t0 = time.perf_counter()
        result = route_compiled(x_batch, W_route, b_route)
        mx.eval(result)
        cached_call_ms = (time.perf_counter() - t0) * 1000

        dynamic_results.append({
            "batch_size": batch,
            "first_call_ms": round(first_call_ms, 3),
            "cached_call_ms": round(cached_call_ms, 3),
            "recompile_cost_ms": round(first_call_ms - cached_call_ms, 3),
        })
        log(f"  Batch={batch}: first={first_call_ms:.3f}ms, cached={cached_call_ms:.3f}ms, "
            f"recompile_cost={first_call_ms - cached_call_ms:.3f}ms")
        del x_batch, result

    results = {
        "uncompiled": stats_uncompiled,
        "compiled": stats_compiled,
        "speedup": round(speedup, 3),
        "dynamic_shapes": dynamic_results,
    }

    cleanup(W_route, b_route, x)
    return results


# ============================================================================
# Phase 3: Compiled LoRA delta computation
# ============================================================================

def phase_compiled_lora_delta():
    """Test mx.compile on LoRA delta: y = base_y + (x @ A) @ B * scale."""
    log("\n" + "=" * 60)
    log("[Phase 3] Compiled LoRA Delta Computation")
    log("=" * 60)
    cleanup()

    D_IN = 2560
    D_OUT = 2560
    R = 16

    # Create synthetic LoRA weights
    A = mx.random.normal((D_IN, R)).astype(mx.bfloat16) * 0.01
    B = mx.random.normal((R, D_OUT)).astype(mx.bfloat16) * 0.01
    x = mx.random.normal((1, D_IN)).astype(mx.bfloat16)
    base_y = mx.random.normal((1, D_OUT)).astype(mx.bfloat16)
    mx.eval(A, B, x, base_y)

    # Note: mx.addmm requires alpha as Python float, not mx.array.
    # Inside mx.compile, Python float constants are captured — if they change,
    # recompilation triggers. Our LORA_SCALE is fixed so this is fine.
    scale_f = float(LORA_SCALE)

    # Uncompiled: naive path
    def lora_naive(x, base_y, A, B):
        h = x @ A
        lora_out = h @ B
        return base_y + lora_out * scale_f

    # Uncompiled: addmm path
    def lora_addmm(x, base_y, A, B):
        h = x @ A
        return mx.addmm(base_y, h, B, alpha=scale_f)

    # Compiled versions
    lora_naive_compiled = mx.compile(lora_naive)
    lora_addmm_compiled = mx.compile(lora_addmm)

    stats_naive = time_function(lora_naive, x, base_y, A, B,
                                 label="LoRA naive (uncompiled)")
    stats_addmm = time_function(lora_addmm, x, base_y, A, B,
                                 label="LoRA addmm (uncompiled)")
    stats_naive_c = time_function(lora_naive_compiled, x, base_y, A, B,
                                   label="LoRA naive (compiled)")
    stats_addmm_c = time_function(lora_addmm_compiled, x, base_y, A, B,
                                   label="LoRA addmm (compiled)")

    speedup_naive = stats_naive["mean_ms"] / stats_naive_c["mean_ms"] if stats_naive_c["mean_ms"] > 0 else 0
    speedup_addmm = stats_addmm["mean_ms"] / stats_addmm_c["mean_ms"] if stats_addmm_c["mean_ms"] > 0 else 0

    log(f"  Speedup naive: {speedup_naive:.2f}x")
    log(f"  Speedup addmm: {speedup_addmm:.2f}x")

    # Test multi-layer chain: simulate 7 projections (one transformer block)
    As = [mx.random.normal((D_IN, R)).astype(mx.bfloat16) * 0.01 for _ in range(7)]
    Bs = [mx.random.normal((R, D_OUT)).astype(mx.bfloat16) * 0.01 for _ in range(7)]
    base_ys = [mx.random.normal((1, D_OUT)).astype(mx.bfloat16) for _ in range(7)]
    mx.eval(*As, *Bs, *base_ys)

    def block_lora_uncompiled(x, base_ys_flat, As_flat, Bs_flat):
        """Apply 7 LoRA projections (one transformer block)."""
        results = []
        for i in range(7):
            h = x @ As_flat[i]
            y = mx.addmm(base_ys_flat[i], h, Bs_flat[i], alpha=scale_f)
            results.append(y)
        return mx.stack(results)

    def block_lora_compiled_fn(x, *params):
        """Compiled version: params = [base_y0, A0, B0, base_y1, A1, B1, ...]"""
        results = []
        for i in range(7):
            base_y_i = params[3 * i]
            A_i = params[3 * i + 1]
            B_i = params[3 * i + 2]
            h = x @ A_i
            y = mx.addmm(base_y_i, h, B_i, alpha=scale_f)
            results.append(y)
        return mx.stack(results)

    block_compiled = mx.compile(block_lora_compiled_fn)

    # Flatten params for compiled version
    flat_params = []
    for i in range(7):
        flat_params.extend([base_ys[i], As[i], Bs[i]])

    stats_block_unc = time_function(
        lambda: block_lora_uncompiled(x, base_ys, As, Bs),
        label="Block 7-proj (uncompiled)"
    )
    stats_block_c = time_function(
        lambda: block_compiled(x, *flat_params),
        label="Block 7-proj (compiled)"
    )

    speedup_block = stats_block_unc["mean_ms"] / stats_block_c["mean_ms"] if stats_block_c["mean_ms"] > 0 else 0
    log(f"  Speedup block (7 projections): {speedup_block:.2f}x")

    results = {
        "single_projection": {
            "naive_uncompiled": stats_naive,
            "addmm_uncompiled": stats_addmm,
            "naive_compiled": stats_naive_c,
            "addmm_compiled": stats_addmm_c,
            "speedup_naive": round(speedup_naive, 3),
            "speedup_addmm": round(speedup_addmm, 3),
        },
        "block_7_projections": {
            "uncompiled": stats_block_unc,
            "compiled": stats_block_c,
            "speedup": round(speedup_block, 3),
        },
    }

    cleanup(A, B, x, base_y, *As, *Bs, *base_ys, *flat_params)
    return results


# ============================================================================
# Phase 4: Compiled multi-adapter LoRA with fixed N
# ============================================================================

def phase_compiled_multi_adapter():
    """Compile multi-adapter LoRA for fixed N=2 and N=5."""
    log("\n" + "=" * 60)
    log("[Phase 4] Compiled Multi-Adapter LoRA (fixed N)")
    log("=" * 60)
    cleanup()

    D_IN = 2560
    D_OUT = 2560
    R = 16

    x = mx.random.normal((1, D_IN)).astype(mx.bfloat16)
    base_y = mx.random.normal((1, D_OUT)).astype(mx.bfloat16)
    mx.eval(x, base_y)

    results = {}

    scale_f = float(LORA_SCALE)

    for N in [2, 5]:
        log(f"\n  --- N={N} adapters ---")

        As = [mx.random.normal((D_IN, R)).astype(mx.bfloat16) * 0.01 for _ in range(N)]
        Bs = [mx.random.normal((R, D_OUT)).astype(mx.bfloat16) * 0.01 for _ in range(N)]
        gates = mx.ones((N,), dtype=mx.bfloat16) / N
        mx.eval(*As, *Bs, gates)

        # Uncompiled: use manual scaling since addmm alpha must be Python float
        def multi_lora_uncompiled(x, base_y, As, Bs, gates):
            y = base_y
            for i in range(len(As)):
                h = x @ As[i]
                lora = (h @ Bs[i]) * (scale_f * gates[i])
                y = y + lora
            return y

        # Compiled: flatten adapter params, gates as array input
        def make_compiled_multi_lora(n):
            def fn(x, base_y, gates, *adapter_params):
                y = base_y
                for i in range(n):
                    A_i = adapter_params[2 * i]
                    B_i = adapter_params[2 * i + 1]
                    h = x @ A_i
                    lora = (h @ B_i) * (scale_f * gates[i])
                    y = y + lora
                return y
            return mx.compile(fn)

        compiled_fn = make_compiled_multi_lora(N)
        flat_adapters = []
        for i in range(N):
            flat_adapters.extend([As[i], Bs[i]])

        stats_unc = time_function(
            lambda: multi_lora_uncompiled(x, base_y, As, Bs, gates),
            label=f"N={N} multi-LoRA (uncompiled)"
        )
        stats_c = time_function(
            lambda: compiled_fn(x, base_y, gates, *flat_adapters),
            label=f"N={N} multi-LoRA (compiled)"
        )

        speedup = stats_unc["mean_ms"] / stats_c["mean_ms"] if stats_c["mean_ms"] > 0 else 0
        log(f"  N={N} speedup: {speedup:.2f}x")

        results[f"n_{N}"] = {
            "uncompiled": stats_unc,
            "compiled": stats_c,
            "speedup": round(speedup, 3),
        }

        cleanup(*As, *Bs, gates, *flat_adapters)

    cleanup(x, base_y)
    return results


# ============================================================================
# Phase 5: Dynamic adapter selection (recompilation cost)
# ============================================================================

def phase_dynamic_adapter():
    """Test dynamic N: compile for N=2, then switch to N=5. Measure recompile cost."""
    log("\n" + "=" * 60)
    log("[Phase 5] Dynamic Adapter Selection (recompilation cost)")
    log("=" * 60)
    cleanup()

    D_IN = 2560
    D_OUT = 2560
    R = 16

    x = mx.random.normal((1, D_IN)).astype(mx.bfloat16)
    base_y = mx.random.normal((1, D_OUT)).astype(mx.bfloat16)
    mx.eval(x, base_y)

    # Strategy A: Pre-compile for each possible N
    log("\n  Strategy A: Pre-compile per N")
    precompiled = {}
    compile_times = {}

    scale_f = float(LORA_SCALE)

    for N in [1, 2, 3, 4, 5]:
        As = [mx.random.normal((D_IN, R)).astype(mx.bfloat16) * 0.01 for _ in range(N)]
        Bs = [mx.random.normal((R, D_OUT)).astype(mx.bfloat16) * 0.01 for _ in range(N)]
        gates = mx.ones((N,), dtype=mx.bfloat16) / N
        mx.eval(*As, *Bs, gates)

        def make_fn(n):
            def fn(x, base_y, gates, *adapter_params):
                y = base_y
                for i in range(n):
                    A_i = adapter_params[2 * i]
                    B_i = adapter_params[2 * i + 1]
                    h = x @ A_i
                    lora = (h @ B_i) * (scale_f * gates[i])
                    y = y + lora
                return y
            return fn

        compiled_fn = mx.compile(make_fn(N))

        flat = []
        for i in range(N):
            flat.extend([As[i], Bs[i]])

        # First call (triggers compilation)
        t0 = time.perf_counter()
        result = compiled_fn(x, base_y, gates, *flat)
        mx.eval(result)
        first_ms = (time.perf_counter() - t0) * 1000

        # Second call (cached)
        t0 = time.perf_counter()
        result = compiled_fn(x, base_y, gates, *flat)
        mx.eval(result)
        cached_ms = (time.perf_counter() - t0) * 1000

        compile_times[N] = {
            "first_call_ms": round(first_ms, 3),
            "cached_call_ms": round(cached_ms, 3),
            "compile_overhead_ms": round(first_ms - cached_ms, 3),
        }
        log(f"  N={N}: first={first_ms:.3f}ms, cached={cached_ms:.3f}ms, "
            f"overhead={first_ms - cached_ms:.3f}ms")
        precompiled[N] = compiled_fn

        del As, Bs, gates, flat, result

    # Strategy B: Pad to max N with zero gates
    log("\n  Strategy B: Pad to max N=5 with zero gates")
    N_MAX = 5

    As_all = [mx.random.normal((D_IN, R)).astype(mx.bfloat16) * 0.01 for _ in range(N_MAX)]
    Bs_all = [mx.random.normal((R, D_OUT)).astype(mx.bfloat16) * 0.01 for _ in range(N_MAX)]
    mx.eval(*As_all, *Bs_all)

    def padded_multi_lora(x, base_y, gates, *adapter_params):
        """Always process N_MAX adapters, zero gates for inactive ones."""
        y = base_y
        for i in range(5):  # hardcoded N_MAX
            A_i = adapter_params[2 * i]
            B_i = adapter_params[2 * i + 1]
            h = x @ A_i
            lora = (h @ B_i) * (scale_f * gates[i])
            y = y + lora
        return y

    padded_compiled = mx.compile(padded_multi_lora)

    flat_all = []
    for i in range(N_MAX):
        flat_all.extend([As_all[i], Bs_all[i]])

    padded_results = {}
    for active_n in [1, 2, 3, 5]:
        # Create gate vector with zeros for inactive adapters
        active_gates = mx.ones((active_n,), dtype=mx.bfloat16) / active_n
        gates = mx.concatenate([active_gates, mx.zeros((N_MAX - active_n,), dtype=mx.bfloat16)])
        mx.eval(gates)

        stats = time_function(
            lambda: padded_compiled(x, base_y, gates, *flat_all),
            label=f"Padded N=5, active={active_n}"
        )
        padded_results[f"active_{active_n}"] = stats

    k1_result = "PASS"
    k1_evidence = "Compilation succeeds for all N via both strategies (pre-compile per N and padding)"

    # Check if any strategy failed
    try:
        # Quick test: call each precompiled function
        for N, fn in precompiled.items():
            As_test = [mx.random.normal((D_IN, R)).astype(mx.bfloat16) for _ in range(N)]
            Bs_test = [mx.random.normal((R, D_OUT)).astype(mx.bfloat16) for _ in range(N)]
            gates_test = mx.ones((N,), dtype=mx.bfloat16) / N
            mx.eval(*As_test, *Bs_test, gates_test)
            flat_test = []
            for i in range(N):
                flat_test.extend([As_test[i], Bs_test[i]])
            result = fn(x, base_y, gates_test, *flat_test)
            mx.eval(result)
            del As_test, Bs_test, gates_test, flat_test, result
    except Exception as e:
        k1_result = "FAIL"
        k1_evidence = f"Compilation failed: {e}"

    results = {
        "strategy_a_precompile": compile_times,
        "strategy_b_padded": padded_results,
        "k1_dynamic_selection": k1_result,
        "k1_evidence": k1_evidence,
    }

    cleanup(x, base_y, *As_all, *Bs_all, *flat_all)
    return results


# ============================================================================
# Phase 6: End-to-end compiled adapter serving (the real test)
# ============================================================================

def phase_e2e_compiled_serving():
    """Full end-to-end: compiled LoRA wrapper vs uncompiled, real model tok/s."""
    log("\n" + "=" * 60)
    log("[Phase 6] End-to-End Compiled Adapter Serving")
    log("=" * 60)
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Load adapter
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))
    adapter_bf16 = {k: v.astype(mx.bfloat16) for k, v in adapter.items()}
    del adapter
    mx.eval(list(adapter_bf16.values()))

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    a_matrices = {}
    for k in skeleton:
        if k.endswith("_domain_0"):
            a_matrices[k] = mx.array(skeleton[k]).astype(mx.bfloat16)
    del skeleton
    mx.eval(list(a_matrices.values()))
    gc.collect()

    TARGET_KEYS = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    # --- Approach A: Uncompiled LoRA wrapper (baseline from Phase 1) ---
    class BitLinearWithLoRA(nn.Module):
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            lora_out = (x @ self.lora_a) @ self.lora_b * self.lora_scale
            return y + lora_out

    # --- Approach B: Compiled LoRA wrapper ---
    # Share ONE compiled function across all layers (same shapes, same scale)
    _shared_scale = float(LORA_SCALE)

    @mx.compile
    def _shared_compiled_lora(x, base_y, A, B):
        h = x @ A
        return mx.addmm(base_y, h, B, alpha=_shared_scale)

    class BitLinearWithCompiledLoRA(nn.Module):
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix

        def __call__(self, x):
            y = self.base(x)
            return _shared_compiled_lora(x, y, self.lora_a, self.lora_b)

    # --- Approach C: addmm without compile (to isolate compile benefit vs addmm benefit) ---
    class BitLinearWithAddmm(nn.Module):
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix
            self._scale_f = float(lora_scale)

        def __call__(self, x):
            y = self.base(x)
            h = x @ self.lora_a
            return mx.addmm(y, h, self.lora_b, alpha=self._scale_f)

    results = {}

    for approach_name, WrapperClass in [
        ("A_naive_uncompiled", BitLinearWithLoRA),
        ("B_addmm_uncompiled", BitLinearWithAddmm),
        ("C_addmm_compiled", BitLinearWithCompiledLoRA),
    ]:
        log(f"\n  --- Approach: {approach_name} ---")

        # Reload model fresh each time
        cleanup(model, tokenizer)
        model, tokenizer = load(MODEL_ID)
        mx.eval(model.parameters())

        n_wrapped = 0
        n_layers = len(model.model.layers)
        for li in range(n_layers):
            updates = []
            for key in TARGET_KEYS:
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is None or not isinstance(module, BitLinear):
                    continue

                skey = f"layer_{li}_{key}_domain_0"
                b_key = f"model.layers.{li}.{key}.lora_b"
                if skey not in a_matrices or b_key not in adapter_bf16:
                    continue

                wrapped = WrapperClass(module, a_matrices[skey], adapter_bf16[b_key], LORA_SCALE)
                updates.append((key, wrapped))
                n_wrapped += 1

            if updates:
                model.model.layers[li].update_modules(tree_unflatten(updates))

        log(f"  Wrapped {n_wrapped} layers ({approach_name})")
        mem = log_memory(f"{approach_name}")

        tps = measure_tps(model, tokenizer, TEST_PROMPT, n_warmup=3)
        log(f"  {approach_name}: {tps['generation_tps']} tok/s (internal), "
            f"{tps['wallclock_tps']} tok/s (wallclock)")

        results[approach_name] = {
            "tps": tps,
            "n_wrapped": n_wrapped,
            "memory": mem,
        }

    # Compute speedups
    naive_tps = results["A_naive_uncompiled"]["tps"]["generation_tps"]
    addmm_tps = results["B_addmm_uncompiled"]["tps"]["generation_tps"]
    compiled_tps = results["C_addmm_compiled"]["tps"]["generation_tps"]

    results["speedups"] = {
        "addmm_vs_naive": round(addmm_tps / naive_tps, 3) if naive_tps > 0 else 0,
        "compiled_vs_naive": round(compiled_tps / naive_tps, 3) if naive_tps > 0 else 0,
        "compiled_vs_addmm": round(compiled_tps / addmm_tps, 3) if addmm_tps > 0 else 0,
    }
    log(f"\n  Speedups vs naive:")
    log(f"    addmm (no compile): {results['speedups']['addmm_vs_naive']:.3f}x")
    log(f"    addmm + compile:    {results['speedups']['compiled_vs_naive']:.3f}x")
    log(f"    compile over addmm: {results['speedups']['compiled_vs_addmm']:.3f}x")

    cleanup(model, tokenizer, adapter_bf16, a_matrices)
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")
    log(f"MLX version: {mx.__version__}")
    log(f"Device: {mx.device_info().get('device_name', 'unknown')}")

    all_results = {
        "experiment": "mx_compile_full_pipeline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "Apple M5 Pro 48GB",
        "model": MODEL_ID,
        "mlx_version": mx.__version__,
    }

    # Phase 1: Baseline
    all_results["phase1_baseline"] = phase_baseline()

    # Phase 2: Compiled routing
    all_results["phase2_routing"] = phase_compiled_routing()

    # Phase 3: Compiled LoRA delta
    all_results["phase3_lora_delta"] = phase_compiled_lora_delta()

    # Phase 4: Multi-adapter
    all_results["phase4_multi_adapter"] = phase_compiled_multi_adapter()

    # Phase 5: Dynamic selection
    all_results["phase5_dynamic"] = phase_dynamic_adapter()

    # Phase 6: E2E compiled serving
    all_results["phase6_e2e"] = phase_e2e_compiled_serving()

    # ================================================================
    # Summary and verdicts
    # ================================================================
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    # K1: Dynamic adapter selection
    k1 = all_results["phase5_dynamic"]["k1_dynamic_selection"]
    log(f"  K1 (compilation fails on dynamic selection): {k1}")

    # K2: < 5% improvement
    e2e = all_results["phase6_e2e"]
    compiled_over_addmm = e2e["speedups"]["compiled_vs_addmm"]
    compiled_over_naive = e2e["speedups"]["compiled_vs_naive"]
    k2_pct_vs_addmm = (compiled_over_addmm - 1) * 100
    k2_pct_vs_naive = (compiled_over_naive - 1) * 100
    k2_pass = k2_pct_vs_naive >= 5
    log(f"  K2 (>5% improvement): {'PASS' if k2_pass else 'FAIL'}")
    log(f"    vs naive: {k2_pct_vs_naive:+.1f}%")
    log(f"    vs addmm: {k2_pct_vs_addmm:+.1f}%")

    # S1: >20% improvement
    s1_pass = k2_pct_vs_naive >= 20
    log(f"  S1 (>20% improvement): {'PASS' if s1_pass else 'FAIL'} ({k2_pct_vs_naive:+.1f}%)")

    # Component speedups
    routing_speedup = all_results["phase2_routing"]["speedup"]
    lora_single_speedup = all_results["phase3_lora_delta"]["single_projection"]["speedup_addmm"]
    lora_block_speedup = all_results["phase3_lora_delta"]["block_7_projections"]["speedup"]

    log(f"\n  Component speedups (compiled vs uncompiled):")
    log(f"    Routing head: {routing_speedup:.2f}x")
    log(f"    LoRA single proj (addmm): {lora_single_speedup:.2f}x")
    log(f"    LoRA block (7 proj): {lora_block_speedup:.2f}x")

    # Tok/s comparison
    naive_tps = e2e["A_naive_uncompiled"]["tps"]["generation_tps"]
    addmm_tps = e2e["B_addmm_uncompiled"]["tps"]["generation_tps"]
    compiled_tps = e2e["C_addmm_compiled"]["tps"]["generation_tps"]
    base_tps = all_results["phase1_baseline"]["base_tps"]["generation_tps"]

    log(f"\n  End-to-end tok/s:")
    log(f"    Base (no adapter):     {base_tps}")
    log(f"    Naive LoRA:            {naive_tps}")
    log(f"    addmm LoRA:            {addmm_tps}")
    log(f"    addmm + compile LoRA:  {compiled_tps}")

    all_results["summary"] = {
        "k1_dynamic_selection": k1,
        "k2_speedup_vs_naive_pct": round(k2_pct_vs_naive, 1),
        "k2_speedup_vs_addmm_pct": round(k2_pct_vs_addmm, 1),
        "k2_pass": k2_pass,
        "s1_pass": s1_pass,
        "s1_improvement_pct": round(k2_pct_vs_naive, 1),
        "component_speedups": {
            "routing": routing_speedup,
            "lora_single_addmm": lora_single_speedup,
            "lora_block_7proj": lora_block_speedup,
        },
        "e2e_tps": {
            "base": base_tps,
            "naive_lora": naive_tps,
            "addmm_lora": addmm_tps,
            "compiled_addmm_lora": compiled_tps,
        },
    }

    all_results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {all_results['total_time_s']}s")


if __name__ == "__main__":
    main()
