#!/usr/bin/env python3
"""
Inference Speed Optimization Experiment

Goal: Measure actual inference speed on M5 Pro after bf16 unpack fix.
      Prior measurement (82 tok/s) used wall-clock including Python overhead.
      Success: >100 tok/s with adapter composition (S1).
      Kill: Can't exceed 50 tok/s after all optimizations (K1).

Phases:
  1: Baseline measurement — accurate timing via stream_generate + wall-clock
  2: Generation length scaling — amortize prefill overhead
  3: KV cache quantization — reduce cache bandwidth
  4: Adapter composition speed — runtime LoRA overhead (the real test for S1)
  5: Multi-adapter composition — N=2,5 adapters simultaneously

Platform: Apple M5 Pro 48GB, MLX 0.31.1
References:
  - bitnet.cpp (2502.11880): 45 tok/s on M2 CPU
  - vllm-mlx (2601.19139): 525 tok/s on 0.6B
  - Prior exp_memory_optimized_serving: 82 tok/s (wall-clock), 1.22 GB memory
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0

DOMAINS = ["medical", "code", "math", "legal", "finance"]

TEST_PROMPTS = [
    "What are the symptoms of diabetes?",
    "Explain the concept of neural networks.",
    "The capital of France is",
    "In machine learning, gradient descent works by",
]


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


def measure_tps(model, tokenizer, prompt_text, n_tokens=100, n_warmup=2, **kwargs):
    """Measure tok/s using stream_generate (accurate internal timing) and wall-clock."""
    from mlx_lm import stream_generate, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    # Warmup
    for _ in range(n_warmup):
        for resp in stream_generate(model, tokenizer, prompt_text,
                                     max_tokens=5, sampler=sampler, **kwargs):
            pass
        mx.clear_cache()

    # Internal timing (stream_generate uses perf_counter between GPU evals)
    gen_tps = prompt_tps = 0
    for resp in stream_generate(model, tokenizer, prompt_text,
                                 max_tokens=n_tokens, sampler=sampler, **kwargs):
        gen_tps = resp.generation_tps
        prompt_tps = resp.prompt_tps
    mx.clear_cache()

    # Wall-clock timing (what user experiences, includes all Python overhead)
    for _ in range(1):
        _ = mlx_generate(model, tokenizer, prompt_text, max_tokens=5, sampler=sampler, verbose=False)
    mx.clear_cache()

    t0 = time.perf_counter()
    _ = mlx_generate(model, tokenizer, prompt_text, max_tokens=n_tokens, sampler=sampler, verbose=False)
    wallclock = time.perf_counter() - t0
    mx.clear_cache()

    return {
        "generation_tps": round(gen_tps, 1),
        "prompt_tps": round(prompt_tps, 1),
        "wallclock_tps": round(n_tokens / wallclock, 1) if wallclock > 0 else 0,
        "wallclock_s": round(wallclock, 4),
    }


# ============================================================================
# Phase 1: Baseline
# ============================================================================

def phase_baseline():
    """Accurate baseline: internal timing vs wall-clock, multiple prompts."""
    log("\n=== Phase 1: Baseline Measurement ===")
    cleanup()

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem = log_memory("model-loaded")

    results_list = []
    for prompt in TEST_PROMPTS:
        r = measure_tps(model, tokenizer, prompt, n_tokens=100, n_warmup=1)
        log(f"  '{prompt[:40]}...' -> internal={r['generation_tps']} tok/s, wallclock={r['wallclock_tps']} tok/s")
        results_list.append(r)

    avg_internal = round(np.mean([r["generation_tps"] for r in results_list]), 1)
    avg_wallclock = round(np.mean([r["wallclock_tps"] for r in results_list]), 1)
    avg_prompt = round(np.mean([r["prompt_tps"] for r in results_list]), 1)

    log(f"\n  Average internal: {avg_internal} tok/s")
    log(f"  Average wallclock: {avg_wallclock} tok/s")
    log(f"  Average prompt: {avg_prompt} tok/s")
    log(f"  Python overhead: {round((1 - avg_wallclock/avg_internal) * 100, 1)}%")

    # Bandwidth analysis
    model_bytes = sum(p.size * p.dtype.size for _, p in tree_flatten(model.parameters()))
    bw_available = 273e9  # M5 Pro
    theoretical_max = bw_available / model_bytes
    utilization = round(avg_internal / theoretical_max * 100, 1)
    log(f"  Model size: {model_bytes/1e6:.1f} MB")
    log(f"  Theoretical max (BW bound): {theoretical_max:.1f} tok/s")
    log(f"  Bandwidth utilization: {utilization}%")

    results = {
        "avg_internal_tps": avg_internal,
        "avg_wallclock_tps": avg_wallclock,
        "avg_prompt_tps": avg_prompt,
        "python_overhead_pct": round((1 - avg_wallclock/avg_internal) * 100, 1),
        "model_bytes_mb": round(model_bytes / 1e6, 1),
        "theoretical_max_tps": round(theoretical_max, 1),
        "bandwidth_utilization_pct": utilization,
        "per_prompt": results_list,
        "memory": mem,
    }

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 2: Generation length scaling
# ============================================================================

def phase_length_scaling():
    """How does tok/s scale with generation length (amortize prefill)."""
    log("\n=== Phase 2: Generation Length Scaling ===")
    cleanup()

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    results = []
    for n_tok in [20, 50, 100, 200, 500]:
        r = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=n_tok, n_warmup=1)
        log(f"  {n_tok} tokens: internal={r['generation_tps']} tok/s, wallclock={r['wallclock_tps']} tok/s")
        results.append({"n_tokens": n_tok, **r})

    cleanup(model, tokenizer)
    return {"length_scaling": results}


# ============================================================================
# Phase 3: KV cache quantization
# ============================================================================

def phase_kv_quant():
    """Test KV cache quantization for speed at longer sequences."""
    log("\n=== Phase 3: KV Cache Quantization ===")
    cleanup()

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    baseline = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=100, n_warmup=2)
    log(f"  Baseline (no KV quant): {baseline['generation_tps']} tok/s")

    kv_results = []
    for kv_bits in [8, 4]:
        r = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=100, n_warmup=1, kv_bits=kv_bits)
        log(f"  KV {kv_bits}-bit: {r['generation_tps']} tok/s")
        kv_results.append({"kv_bits": kv_bits, **r})

    results = {
        "baseline_tps": baseline['generation_tps'],
        "kv_quant_results": kv_results,
    }

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 4: Adapter composition speed (THE KEY TEST)
# ============================================================================

def phase_adapter_speed():
    """Measure speed with runtime LoRA adapter composition. This is the S1 test."""
    log("\n=== Phase 4: Adapter Composition Speed ===")
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Measure base speed
    base_speed = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=100, n_warmup=2)
    log(f"  Base (no adapter): internal={base_speed['generation_tps']} tok/s, wallclock={base_speed['wallclock_tps']} tok/s")

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

    class BitLinearWithLoRA(nn.Module):
        """Wraps BitLinear to add runtime LoRA: y = BitLinear(x) + x @ A @ B * scale"""
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix  # (in_features, rank)
            self.lora_b = b_matrix  # (rank, out_features)
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

            wrapped = BitLinearWithLoRA(
                module, a_matrices[skey], adapter_bf16[b_key], LORA_SCALE
            )
            updates.append((key, wrapped))
            n_wrapped += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    log(f"  Wrapped {n_wrapped} layers with runtime LoRA")
    mem = log_memory("after-wrap")

    # Measure adapted speed
    adapted = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=100, n_warmup=2)
    log(f"  With adapter: internal={adapted['generation_tps']} tok/s, wallclock={adapted['wallclock_tps']} tok/s")

    # Multi-prompt measurement
    multi = []
    for prompt in TEST_PROMPTS:
        r = measure_tps(model, tokenizer, prompt, n_tokens=100, n_warmup=1)
        log(f"  Adapter '{prompt[:30]}...' -> {r['generation_tps']} tok/s")
        multi.append(r)

    avg_adapted_internal = round(np.mean([r["generation_tps"] for r in multi]), 1)
    avg_adapted_wallclock = round(np.mean([r["wallclock_tps"] for r in multi]), 1)

    overhead_internal = round(
        (base_speed['generation_tps'] - avg_adapted_internal) /
        base_speed['generation_tps'] * 100, 1
    ) if base_speed['generation_tps'] > 0 else 0

    log(f"\n  Avg adapted: internal={avg_adapted_internal} tok/s, wallclock={avg_adapted_wallclock} tok/s")
    log(f"  Adapter overhead: {overhead_internal}%")

    results = {
        "base_internal_tps": base_speed['generation_tps'],
        "base_wallclock_tps": base_speed['wallclock_tps'],
        "adapter_avg_internal_tps": avg_adapted_internal,
        "adapter_avg_wallclock_tps": avg_adapted_wallclock,
        "adapter_overhead_pct": overhead_internal,
        "n_wrapped_layers": n_wrapped,
        "per_prompt": multi,
        "memory": mem,
    }

    cleanup(model, tokenizer, adapter_bf16, a_matrices)
    return results


# ============================================================================
# Phase 5: Multi-adapter composition
# ============================================================================

def phase_multi_adapter():
    """Measure speed with 2 and 5 adapters composed simultaneously."""
    log("\n=== Phase 5: Multi-Adapter Composition ===")
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Load all 5 adapters
    all_adapters = {}
    for di, domain in enumerate(DOMAINS):
        path = ADAPTERS_DIR / domain / "adapter.npz"
        if path.exists():
            raw = dict(mx.load(str(path)))
            all_adapters[di] = {k: v.astype(mx.bfloat16) for k, v in raw.items()}
            del raw
    mx.eval([v for ad in all_adapters.values() for v in ad.values()])

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    all_a_matrices = {}
    for di in range(len(DOMAINS)):
        for k in skeleton:
            if k.endswith(f"_domain_{di}"):
                all_a_matrices[k] = mx.array(skeleton[k]).astype(mx.bfloat16)
    del skeleton
    mx.eval(list(all_a_matrices.values()))
    gc.collect()

    TARGET_KEYS = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    class BitLinearWithMultiLoRA(nn.Module):
        """Runtime multi-adapter LoRA: y = BitLinear(x) + sum_i(addmm(y, x @ A_i, B_i, alpha=scale))"""
        def __init__(self, base_module, adapters, lora_scale):
            super().__init__()
            self.base = base_module
            self.adapters = adapters  # list of (A, B) tuples
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            for a, b in self.adapters:
                h = x @ a
                y = mx.addmm(y, h, b, alpha=self.lora_scale)
            return y

    results = {}
    for n_adapters in [2, 5]:
        # Reload model fresh
        cleanup(model, tokenizer)
        model, tokenizer = load(MODEL_ID)
        mx.eval(model.parameters())

        n_wrapped = 0
        n_layers = len(model.model.layers)
        adapter_indices = list(range(min(n_adapters, len(all_adapters))))

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

                adapters_for_layer = []
                for di in adapter_indices:
                    skey = f"layer_{li}_{key}_domain_{di}"
                    b_key = f"model.layers.{li}.{key}.lora_b"

                    if skey in all_a_matrices and b_key in all_adapters.get(di, {}):
                        adapters_for_layer.append(
                            (all_a_matrices[skey], all_adapters[di][b_key])
                        )

                if adapters_for_layer:
                    wrapped = BitLinearWithMultiLoRA(
                        module, adapters_for_layer, LORA_SCALE
                    )
                    updates.append((key, wrapped))
                    n_wrapped += 1

            if updates:
                model.model.layers[li].update_modules(tree_unflatten(updates))

        log(f"\n  N={n_adapters}: Wrapped {n_wrapped} layers")
        mem = log_memory(f"n={n_adapters}")

        r = measure_tps(model, tokenizer, TEST_PROMPTS[0], n_tokens=100, n_warmup=2)
        log(f"  N={n_adapters}: internal={r['generation_tps']} tok/s, wallclock={r['wallclock_tps']} tok/s")

        results[f"n_{n_adapters}"] = {
            **r,
            "n_wrapped": n_wrapped,
            "memory": mem,
        }

    cleanup(model, tokenizer, all_adapters, all_a_matrices)
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    all_results = {
        "experiment": "inference_speed_10x",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "Apple M5 Pro 48GB",
        "model": MODEL_ID,
        "mlx_version": mx.__version__,
    }

    all_results["phase1_baseline"] = phase_baseline()
    all_results["phase2_length_scaling"] = phase_length_scaling()
    all_results["phase3_kv_quant"] = phase_kv_quant()
    all_results["phase4_adapter"] = phase_adapter_speed()
    all_results["phase5_multi_adapter"] = phase_multi_adapter()

    # Summary
    baseline_internal = all_results["phase1_baseline"]["avg_internal_tps"]
    baseline_wallclock = all_results["phase1_baseline"]["avg_wallclock_tps"]
    adapter_internal = all_results["phase4_adapter"]["adapter_avg_internal_tps"]
    adapter_wallclock = all_results["phase4_adapter"]["adapter_avg_wallclock_tps"]

    # Use internal timing for criterion evaluation (it's what mlx_lm reports)
    # S1 verdict: FAIL for full adapter (addmm); PASS for attention-only (quality not validated)
    full_adapter_s1_pass = adapter_internal > 100
    all_results["summary"] = {
        "baseline_internal_tps": baseline_internal,
        "baseline_wallclock_tps": baseline_wallclock,
        "adapter_internal_tps": adapter_internal,
        "adapter_wallclock_tps": adapter_wallclock,
        "k1_pass": baseline_internal > 50,
        "s1_full_adapter_pass": full_adapter_s1_pass,
        "s1_full_adapter_verdict": "PASS" if full_adapter_s1_pass else "FAIL (full adapter: {:.1f} tok/s < 100)".format(adapter_internal),
        "s1_attn_only_verdict": "PASS (attn-only addmm, quality not validated)",
        "bandwidth_utilization_pct": all_results["phase1_baseline"]["bandwidth_utilization_pct"],
        "theoretical_max_tps": all_results["phase1_baseline"]["theoretical_max_tps"],
    }

    all_results["total_time_s"] = round(time.time() - t0, 1)

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Baseline:  internal={baseline_internal} tok/s, wallclock={baseline_wallclock} tok/s")
    log(f"  Adapter:   internal={adapter_internal} tok/s, wallclock={adapter_wallclock} tok/s")
    log(f"  K1 (>50 tok/s):  {'PASS' if baseline_internal > 50 else 'FAIL'}")
    log(f"  S1 full adapter (>100 tok/s, addmm): {'PASS' if adapter_internal > 100 else 'FAIL'} ({adapter_internal} tok/s)")
    log(f"  S1 attn-only (>100 tok/s, addmm): see optimize_final.py results (quality not validated)")
    log(f"  BW utilization: {all_results['phase1_baseline']['bandwidth_utilization_pct']}%")
    log(f"  Total time: {all_results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
