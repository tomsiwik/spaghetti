#!/usr/bin/env python3
"""
Memory Budget Analysis: How many adapters fit in 48GB alongside BitNet-2B?

Kill criteria:
  K1 (261): < 100 adapters fit in 48GB (too few for vision) -> KILL

Success criteria:
  S1 (27): >500 adapters fit in 48GB with base model and routing

Approach:
  1. Load BitNet-2B-4T base model, measure actual memory
  2. Synthesize N adapters (bf16 A+B, rank 16) and measure marginal cost
  3. Synthesize routing heads and measure marginal cost
  4. Profile peak memory during forward pass with routing + composition
  5. Find practical maximum N on 48 GB (40 GB usable)
  6. Test N = 10, 50, 100, 500, 1000, 5000

Platform: Apple M5 Pro 48GB, MLX
References:
  - Prior experiment: experiments/models/memory_optimized_serving/
  - S-LoRA (2311.03285): concurrent LoRA serving architecture
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0

# BitNet-2B-4T architecture constants
N_LAYERS = 30
D_MODEL = 2560
D_FFN = 6912
N_KV_HEADS = 8
D_HEAD = 80

# Target shapes for LoRA layers (7 targets per layer)
# q_proj: (2560, 2560), k_proj: (640, 2560), v_proj: (640, 2560)
# o_proj: (2560, 2560), gate_proj: (6912, 2560), up_proj: (6912, 2560)
# down_proj: (2560, 6912)
LORA_TARGETS = [
    ("q_proj", D_MODEL, D_MODEL),
    ("k_proj", N_KV_HEADS * D_HEAD, D_MODEL),
    ("v_proj", N_KV_HEADS * D_HEAD, D_MODEL),
    ("o_proj", D_MODEL, D_MODEL),
    ("gate_proj", D_FFN, D_MODEL),
    ("up_proj", D_FFN, D_MODEL),
    ("down_proj", D_MODEL, D_FFN),
]

# Test scales
# Note: 1000 adapters at 43.3 MB = 43.3 GB, exceeds 40 GB usable budget
# We measure up to 500 directly, then extrapolate for 1000+
TEST_NS = [10, 50, 100, 500]


def log(msg):
    print(msg, flush=True)


def get_memory_mb():
    """Return (active_mb, cache_mb, peak_mb)."""
    return (
        mx.get_active_memory() / 1e6,
        mx.get_cache_memory() / 1e6,
        mx.get_peak_memory() / 1e6,
    )


def log_memory(label=""):
    active, cache, peak = get_memory_mb()
    log(f"[MEM {label}] active={active:.1f}MB cache={cache:.1f}MB peak={peak:.1f}MB")
    return {"active_mb": round(active, 1), "cache_mb": round(cache, 1), "peak_mb": round(peak, 1)}


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Theoretical calculations
# ============================================================================

def compute_theoretical_budget():
    """Compute theoretical memory budget analytically."""
    log("\n=== Theoretical Memory Budget ===")

    # Base model
    # Packed ternary: each weight is 2 bits, packed 4 per byte
    packed_bytes = 0
    bf16_bytes = 0
    for layer_idx in range(N_LAYERS):
        for name, out_dim, in_dim in LORA_TARGETS:
            packed_bytes += math.ceil(out_dim / 4) * in_dim  # BitLinear packed
        # Also has layer norms etc (small)

    # Non-ternary params (embedding, lm_head, norms)
    embed_bytes = 151936 * D_MODEL * 2  # bf16
    lm_head_bytes = 151936 * D_MODEL * 2  # bf16 (tied with embed usually)
    norm_bytes = N_LAYERS * 2 * D_MODEL * 2 + D_MODEL * 2  # layer norms + final norm

    # Per adapter (bf16 A+B)
    adapter_bytes_per = 0
    a_bytes_per = 0
    b_bytes_per = 0
    for name, out_dim, in_dim in LORA_TARGETS:
        a_bytes_per += in_dim * LORA_RANK * 2  # A: in x r, bf16
        b_bytes_per += LORA_RANK * out_dim * 2  # B: r x out, bf16
    a_bytes_per *= N_LAYERS
    b_bytes_per *= N_LAYERS
    adapter_bytes_per = a_bytes_per + b_bytes_per

    # Per routing head (2-layer MLP: Linear(2560, 16) + Linear(16, 1))
    head_params = D_MODEL * 16 + 16 + 16 * 1 + 1  # weights + biases
    head_bytes = head_params * 2  # bf16

    # Router (shared, 659K params)
    router_bytes = 659_000 * 2

    # KV cache
    kv_per_token = 2 * N_LAYERS * N_KV_HEADS * D_HEAD * 2  # K+V, bf16

    log(f"  Packed ternary weights (210 layers): {packed_bytes / 1e6:.1f} MB")
    log(f"  Embedding + LM head: {(embed_bytes + lm_head_bytes) / 1e6:.1f} MB")
    log(f"  Layer norms: {norm_bytes / 1e6:.3f} MB")
    log(f"  Per adapter A (bf16): {a_bytes_per / 1e6:.1f} MB")
    log(f"  Per adapter B (bf16): {b_bytes_per / 1e6:.1f} MB")
    log(f"  Per adapter total (bf16): {adapter_bytes_per / 1e6:.1f} MB")
    log(f"  Per adapter B (int8): {b_bytes_per / 2 / 1e6:.1f} MB")
    log(f"  Per routing head: {head_bytes / 1e3:.1f} KB")
    log(f"  Shared router: {router_bytes / 1e6:.2f} MB")
    log(f"  KV cache per token: {kv_per_token / 1e3:.1f} KB")

    # Budget analysis
    usable_gb = 40.0
    usable_bytes = usable_gb * 1e9
    base_measured = 1_178.6e6  # From prior experiment

    scenarios = {}
    for format_name, per_adapter_bytes in [
        ("bf16_A_B", adapter_bytes_per),
        ("int8_B_bf16_A", b_bytes_per // 2 + a_bytes_per),
        ("bf16_B_only_shared_A", b_bytes_per),  # If A could be shared
    ]:
        for seq_len in [256, 2048, 8192]:
            kv_bytes = kv_per_token * seq_len
            available = usable_bytes - base_measured - router_bytes - kv_bytes - 10e6  # 10MB activations
            n_max = int(available / (per_adapter_bytes + head_bytes))
            key = f"{format_name}_seq{seq_len}"
            scenarios[key] = {
                "format": format_name,
                "seq_len": seq_len,
                "per_adapter_mb": round(per_adapter_bytes / 1e6, 2),
                "kv_cache_mb": round(kv_bytes / 1e6, 1),
                "n_max_theoretical": n_max,
            }
            log(f"  {format_name} seq={seq_len}: N_max = {n_max} (adapter={per_adapter_bytes/1e6:.1f}MB, KV={kv_bytes/1e6:.1f}MB)")

    return {
        "packed_ternary_mb": round(packed_bytes / 1e6, 1),
        "per_adapter_a_bf16_mb": round(a_bytes_per / 1e6, 2),
        "per_adapter_b_bf16_mb": round(b_bytes_per / 1e6, 2),
        "per_adapter_total_bf16_mb": round(adapter_bytes_per / 1e6, 2),
        "per_adapter_b_int8_mb": round(b_bytes_per / 2 / 1e6, 2),
        "per_head_kb": round(head_bytes / 1e3, 1),
        "router_mb": round(router_bytes / 1e6, 2),
        "kv_per_token_kb": round(kv_per_token / 1e3, 1),
        "scenarios": scenarios,
    }


# ============================================================================
# Phase 1: Measure base model memory (actual)
# ============================================================================

def phase_measure_base():
    """Load BitNet-2B-4T and measure actual base memory."""
    log("\n=== Phase 1: Measure Base Model Memory ===")
    cleanup()
    mx.reset_peak_memory()

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem_base = log_memory("base-model")

    # Quick forward pass to warm up
    tokens = tokenizer.encode("Hello world")
    x = mx.array(tokens)[None, :]
    logits = model(x)
    mx.eval(logits)
    mem_after_fwd = log_memory("after-forward")

    result = {
        "base_active_mb": mem_base["active_mb"],
        "base_peak_mb": mem_base["peak_mb"],
        "after_fwd_active_mb": mem_after_fwd["active_mb"],
        "after_fwd_peak_mb": mem_after_fwd["peak_mb"],
    }

    cleanup(model, tokenizer, logits, x)
    return result


# ============================================================================
# Phase 2: Measure per-adapter marginal cost with synthetic adapters
# ============================================================================

def phase_measure_adapter_scaling():
    """Create synthetic adapters at various N and measure actual memory."""
    log("\n=== Phase 2: Adapter Memory Scaling ===")

    results = {}

    for N in TEST_NS:
        # Check if this N is feasible given available memory
        estimated_mb = N * 43.3  # worst case bf16
        if estimated_mb > 38_000:  # > 38 GB of adapters
            log(f"\n  N={N}: SKIP (estimated {estimated_mb/1e3:.1f} GB adapters > 38 GB budget)")
            results[str(N)] = {"status": "skipped", "reason": f"estimated {estimated_mb/1e3:.1f} GB > budget"}
            continue

        log(f"\n  --- N={N} adapters ---")
        # Full cleanup and cache clear to get a clean baseline
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        # Force a sync to ensure cleanup is complete
        mx.eval(mx.zeros(1))
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()

        mem_before = log_memory(f"N={N}-before")

        # Allocate N adapters (B matrices + A matrices, bf16)
        # Use mx.random.normal to ensure actual memory allocation (not zero-page dedup)
        adapters = []
        t0 = time.time()

        for i in range(N):
            adapter = {}
            for layer_idx in range(N_LAYERS):
                for name, out_dim, in_dim in LORA_TARGETS:
                    key_a = f"layers.{layer_idx}.{name}.lora_a"
                    key_b = f"layers.{layer_idx}.{name}.lora_b"
                    # Use random data to prevent zero-page optimization
                    adapter[key_a] = mx.random.normal((in_dim, LORA_RANK)).astype(mx.bfloat16)
                    adapter[key_b] = mx.random.normal((LORA_RANK, out_dim)).astype(mx.bfloat16)
            adapters.append(adapter)

            # Evaluate periodically to force allocation
            if (i + 1) % max(1, N // 10) == 0 or i == N - 1:
                mx.eval([v for a in adapters for v in a.values()])

        alloc_time = time.time() - t0
        mem_after = log_memory(f"N={N}-after-alloc")

        # Also allocate N routing heads
        heads = []
        for i in range(N):
            # 2-layer MLP: Linear(2560, 16) + Linear(16, 1)
            w1 = mx.zeros((D_MODEL, 16), dtype=mx.bfloat16)
            b1 = mx.zeros((16,), dtype=mx.bfloat16)
            w2 = mx.zeros((16, 1), dtype=mx.bfloat16)
            b2 = mx.zeros((1,), dtype=mx.bfloat16)
            heads.append((w1, b1, w2, b2))
        mx.eval([t for h in heads for t in h])
        mem_with_heads = log_memory(f"N={N}-with-heads")

        # Compute per-adapter marginal cost
        adapter_delta = mem_after["active_mb"] - mem_before["active_mb"]
        per_adapter_mb = adapter_delta / N if N > 0 else 0
        head_delta = mem_with_heads["active_mb"] - mem_after["active_mb"]
        per_head_mb = head_delta / N if N > 0 else 0

        log(f"  N={N}: total={mem_with_heads['active_mb']:.1f}MB, "
            f"per_adapter={per_adapter_mb:.2f}MB, per_head={per_head_mb:.3f}MB, "
            f"alloc_time={alloc_time:.1f}s")

        results[str(N)] = {
            "n_adapters": N,
            "mem_before_mb": mem_before["active_mb"],
            "mem_adapters_mb": mem_after["active_mb"],
            "mem_with_heads_mb": mem_with_heads["active_mb"],
            "total_adapter_delta_mb": round(adapter_delta, 1),
            "per_adapter_mb": round(per_adapter_mb, 2),
            "total_head_delta_mb": round(head_delta, 1),
            "per_head_mb": round(per_head_mb, 3),
            "alloc_time_s": round(alloc_time, 1),
            "peak_mb": mem_with_heads["peak_mb"],
        }

        # Explicit cleanup of all adapter data
        del adapters
        del heads
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()

    return results


# ============================================================================
# Phase 3: Forward pass peak memory with routing + composition
# ============================================================================

def phase_forward_pass_peak():
    """Measure peak memory during forward pass with adapter composition.

    We load the actual base model + a few synthetic adapters and measure
    peak memory during a forward pass with runtime LoRA.
    """
    log("\n=== Phase 3: Forward Pass Peak Memory ===")
    cleanup()
    mx.reset_peak_memory()

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem_base = log_memory("base-loaded")

    # Create a small set of synthetic adapters to test forward pass overhead
    # We test with k=2 (typical top-k routing) applied to one layer
    k_values = [1, 2, 5]
    results = {}

    for k in k_values:
        log(f"\n  --- Forward pass with k={k} active adapters ---")
        mx.reset_peak_memory()

        # Create k adapter pairs (A, B) for all layers
        active_adapters = []
        for i in range(k):
            adapter = {}
            for layer_idx in range(N_LAYERS):
                for name, out_dim, in_dim in LORA_TARGETS:
                    key_a = f"layers.{layer_idx}.{name}.lora_a"
                    key_b = f"layers.{layer_idx}.{name}.lora_b"
                    adapter[key_a] = mx.random.normal((in_dim, LORA_RANK)) * 0.01
                    adapter[key_b] = mx.random.normal((LORA_RANK, out_dim)) * 0.01
                    adapter[key_a] = adapter[key_a].astype(mx.bfloat16)
                    adapter[key_b] = adapter[key_b].astype(mx.bfloat16)
            active_adapters.append(adapter)
        mx.eval([v for a in active_adapters for v in a.values()])
        mem_adapters_loaded = log_memory(f"k={k}-adapters-loaded")

        # Forward pass: base model output
        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
        x = mx.array(tokens)[None, :]
        logits = model(x)
        mx.eval(logits)
        mem_after_base_fwd = log_memory(f"k={k}-after-base-forward")

        # Simulate runtime LoRA addition (the actual computation pattern)
        # For a simplified test, compute LoRA deltas for first layer only
        # In practice this happens inside the model forward pass
        hidden = mx.random.normal((1, len(tokens), D_MODEL)).astype(mx.bfloat16)
        mx.eval(hidden)

        lora_output = mx.zeros_like(hidden)
        for adapter in active_adapters:
            a = adapter["layers.0.q_proj.lora_a"]  # (2560, 16)
            b = adapter["layers.0.q_proj.lora_b"]  # (16, 2560)
            delta = (hidden @ a) @ b * LORA_SCALE
            lora_output = lora_output + delta
        mx.eval(lora_output)
        mem_after_lora = log_memory(f"k={k}-after-lora-compute")

        results[f"k={k}"] = {
            "k_active": k,
            "mem_base_mb": mem_base["active_mb"],
            "mem_adapters_loaded_mb": mem_adapters_loaded["active_mb"],
            "mem_after_base_fwd_mb": mem_after_base_fwd["active_mb"],
            "mem_after_lora_mb": mem_after_lora["active_mb"],
            "peak_mb": mx.get_peak_memory() / 1e6,
            "lora_overhead_mb": round(mem_after_lora["active_mb"] - mem_after_base_fwd["active_mb"], 1),
        }

        del active_adapters, lora_output, hidden, delta, logits, x
        gc.collect()
        mx.clear_cache()

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 4: Compute practical maximum N
# ============================================================================

def phase_practical_max(theoretical, adapter_scaling, base_mem, forward_peak):
    """Compute practical maximum N from measured data."""
    log("\n=== Phase 4: Practical Maximum N ===")

    # Use measured per-adapter cost from the scaling experiment
    # Take the most reliable measurement (largest N that ran)
    measured_costs = {}
    for n_str, data in adapter_scaling.items():
        if isinstance(data, dict) and "per_adapter_mb" in data:
            measured_costs[int(n_str)] = data["per_adapter_mb"]

    if not measured_costs:
        log("  ERROR: No measured adapter costs available")
        return {"error": "no measured data"}

    # Use the per-adapter cost from the largest N (most accurate due to amortization)
    best_n = max(measured_costs.keys())
    per_adapter_measured = measured_costs[best_n]
    per_head_measured = adapter_scaling[str(best_n)].get("per_head_mb", 0.082)

    # Base memory from actual measurement
    base_mb = base_mem["base_active_mb"]

    # Available memory
    total_available_mb = 40_000  # 40 GB usable

    # Forward pass overhead (peak - active during forward)
    fwd_overhead_mb = 0
    if "k=2" in forward_peak:
        fwd_overhead_mb = forward_peak["k=2"].get("lora_overhead_mb", 50)

    router_mb = 1.32  # 659K params bf16

    results = {}
    for seq_len in [256, 2048, 8192]:
        kv_mb = N_LAYERS * N_KV_HEADS * D_HEAD * seq_len * 2 * 2 / 1e6  # K+V, bf16
        available_for_adapters = (total_available_mb - base_mb - router_mb -
                                   kv_mb - fwd_overhead_mb - 100)  # 100MB safety margin
        n_max = int(available_for_adapters / (per_adapter_measured + per_head_measured))

        # Memory at key milestones
        milestones = {}
        for n_check in [100, 500, 1000, n_max]:
            mem = base_mb + router_mb + kv_mb + n_check * (per_adapter_measured + per_head_measured)
            milestones[f"N={n_check}"] = round(mem / 1000, 2)  # GB

        log(f"  seq={seq_len}: N_max={n_max} "
            f"(per_adapter={per_adapter_measured:.2f}MB, "
            f"per_head={per_head_measured:.3f}MB, "
            f"KV={kv_mb:.0f}MB)")
        for k, v in milestones.items():
            log(f"    {k}: {v:.2f} GB")

        results[f"seq_{seq_len}"] = {
            "seq_len": seq_len,
            "kv_cache_mb": round(kv_mb, 1),
            "available_for_adapters_mb": round(available_for_adapters, 0),
            "per_adapter_mb": per_adapter_measured,
            "per_head_mb": per_head_measured,
            "n_max_practical": n_max,
            "milestones_gb": milestones,
        }

    # Kill criteria assessment
    n_max_default = results["seq_256"]["n_max_practical"]
    k1_pass = n_max_default >= 100
    s1_pass = n_max_default >= 500

    results["kill_criteria"] = {
        "K1_261": {
            "criterion": "< 100 adapters fit in 48GB",
            "n_max": n_max_default,
            "result": "PASS" if k1_pass else "FAIL",
            "margin": f"{n_max_default / 100:.1f}x above threshold",
        },
    }
    results["success_criteria"] = {
        "S1_27": {
            "criterion": ">500 adapters fit in 48GB",
            "n_max": n_max_default,
            "result": "PASS" if s1_pass else "FAIL",
            "margin": f"{n_max_default / 500:.1f}x above threshold",
        },
    }

    log(f"\n  K1 (261): {'PASS' if k1_pass else 'FAIL'} — N_max={n_max_default} "
        f"({'>>100' if k1_pass else '<100'})")
    log(f"  S1 (27): {'PASS' if s1_pass else 'FAIL'} — N_max={n_max_default} "
        f"({'>>500' if s1_pass else '<500'})")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Memory Budget Analysis: How Many Adapters Fit in 48GB?")
    log(f"Platform: {mx.device_info().get('device_type', 'unknown')} "
        f"with {total_mem / 1e9:.0f} GB unified memory")
    log("=" * 70)

    # Step 1: Theoretical analysis
    theoretical = compute_theoretical_budget()

    # Step 2: Measure base model
    base_mem = phase_measure_base()

    # Step 3: Measure adapter scaling
    adapter_scaling = phase_measure_adapter_scaling()

    # Step 4: Forward pass peak memory
    forward_peak = phase_forward_pass_peak()

    # Step 5: Compute practical maximum
    practical_max = phase_practical_max(theoretical, adapter_scaling, base_mem, forward_peak)

    # Compile results
    total_time = time.time() - t0
    results = {
        "experiment": "memory_budget_analysis",
        "platform": {
            "device": mx.device_info().get("device_type", "unknown"),
            "total_memory_gb": round(total_mem / 1e9, 1),
            "usable_memory_gb": round((total_mem - 8 * 1024**3) / 1e9, 1),
        },
        "theoretical": theoretical,
        "base_model": base_mem,
        "adapter_scaling": adapter_scaling,
        "forward_pass_peak": forward_peak,
        "practical_maximum": practical_max,
        "total_time_s": round(total_time, 1),
        "verdict": {
            "K1_261": practical_max.get("kill_criteria", {}).get("K1_261", {}).get("result", "UNTESTED"),
            "S1_27": practical_max.get("success_criteria", {}).get("S1_27", {}).get("result", "UNTESTED"),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Base model: {base_mem.get('base_active_mb', 'N/A')} MB")
    if "seq_256" in practical_max:
        pm = practical_max["seq_256"]
        log(f"Per adapter (measured): {pm['per_adapter_mb']:.2f} MB")
        log(f"Per routing head (measured): {pm['per_head_mb']:.3f} MB")
        log(f"Practical N_max (seq=256): {pm['n_max_practical']}")
        log(f"Practical N_max (seq=2048): {practical_max['seq_2048']['n_max_practical']}")
        log(f"Practical N_max (seq=8192): {practical_max['seq_8192']['n_max_practical']}")
    log(f"K1 (261): {results['verdict']['K1_261']}")
    log(f"S1 (27): {results['verdict']['S1_27']}")


if __name__ == "__main__":
    main()
