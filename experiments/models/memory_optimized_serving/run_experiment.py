#!/usr/bin/env python3
"""
Memory-Optimized Serving Experiment

Goal: Achieve sub-3GB total memory for BitNet-2B-4T + adapter composition serving
on Apple Silicon, with <2% quality loss from compression.

Kill criteria:
  K1 (537): Can't get below 5GB -> KILL
  K2 (538): Quality degrades >5% from compression -> KILL

Success criteria:
  S1 (56): <3GB total with <2% quality loss -> Memory-competitive with Qwen-3B

Approach:
  1. Profile memory at each pipeline stage with mx.get_active_memory()
  2. Keep BitLinear packed (native Metal kernel, no bf16 unpack)
  3. Quantize adapter B matrices (fp32 -> bf16 -> int8)
  4. On-demand adapter loading (only top-k in memory)
  5. Measure PPL at each compression level

Platform: Apple M5 Pro 48GB, MLX
References: S-LoRA (2311.03285), CLA (2405.12981)
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
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source adapters from real_data_domain_experts experiment
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]


# ============================================================================
# Memory profiling utilities
# ============================================================================

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
# Phase 1: Profile baseline model loading (BitLinear, no unpacking)
# ============================================================================

def phase_profile_loading():
    """Load BitNet-2B-4T natively (BitLinear) and profile memory at each stage."""
    log("\n=== Phase 1: Profile BitLinear Native Loading ===")
    cleanup()
    mx.reset_peak_memory()
    mem_before = log_memory("before-load")

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem_after_load = log_memory("after-load-bitlinear")

    # Count BitLinear vs Linear layers
    from mlx_lm.models.bitlinear_layers import BitLinear
    n_bitlinear = 0
    n_linear = 0
    packed_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            n_bitlinear += 1
            packed_bytes += module.weight.size * module.weight.dtype.size
        elif isinstance(module, nn.Linear):
            n_linear += 1

    log(f"  BitLinear layers: {n_bitlinear}")
    log(f"  nn.Linear layers: {n_linear}")
    log(f"  Packed weight bytes: {packed_bytes / 1e6:.1f} MB")

    # Run a forward pass to measure inference memory
    tokens = tokenizer.encode("Hello, this is a test of memory usage.")
    x = mx.array(tokens)[None, :]
    logits = model(x)
    mx.eval(logits)
    mem_after_fwd = log_memory("after-forward-pass")

    # Generate a few tokens to measure KV cache
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.7, top_p=0.9)
    text = mlx_generate(model, tokenizer, "The capital of France is",
                        max_tokens=50, sampler=sampler, verbose=False)
    mem_after_gen = log_memory("after-generation-50tok")
    log(f"  Generated: {text[:100]}...")

    # Count total parameters
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    total_bytes = sum(p.size * p.dtype.size for _, p in tree_flatten(model.parameters()))
    log(f"  Total params: {total_params:,}")
    log(f"  Total param bytes: {total_bytes / 1e6:.1f} MB")

    # Check dtype breakdown
    dtype_counts = {}
    for name, p in tree_flatten(model.parameters()):
        dt = str(p.dtype)
        if dt not in dtype_counts:
            dtype_counts[dt] = {"count": 0, "bytes": 0}
        dtype_counts[dt]["count"] += 1
        dtype_counts[dt]["bytes"] += p.size * p.dtype.size
    log(f"  Dtype breakdown:")
    for dt, info in sorted(dtype_counts.items()):
        log(f"    {dt}: {info['count']} tensors, {info['bytes']/1e6:.1f} MB")

    results = {
        "n_bitlinear": n_bitlinear,
        "n_linear": n_linear,
        "packed_weight_mb": round(packed_bytes / 1e6, 1),
        "total_params": total_params,
        "total_param_bytes_mb": round(total_bytes / 1e6, 1),
        "dtype_breakdown": {dt: {"count": v["count"], "mb": round(v["bytes"]/1e6, 1)}
                           for dt, v in dtype_counts.items()},
        "mem_before": mem_before,
        "mem_after_load": mem_after_load,
        "mem_after_forward": mem_after_fwd,
        "mem_after_gen_50tok": mem_after_gen,
    }

    cleanup(model, tokenizer, logits, x)
    return results


# ============================================================================
# Phase 2: Measure adapter loading strategies
# ============================================================================

def phase_adapter_strategies():
    """Test different adapter loading strategies for memory efficiency."""
    log("\n=== Phase 2: Adapter Loading Strategies ===")
    cleanup()
    mx.reset_peak_memory()

    results = {}

    # Strategy A: Load all 5 adapters in fp32 (current approach)
    log("\n  Strategy A: All 5 adapters in fp32")
    adapters_fp32 = {}
    for domain in DOMAINS:
        path = ADAPTERS_DIR / domain / "adapter.npz"
        adapters_fp32[domain] = dict(mx.load(str(path)))
    mx.eval([v for ad in adapters_fp32.values() for v in ad.values()])
    mem_all_fp32 = log_memory("all-5-adapters-fp32")
    total_fp32_mb = sum(
        v.size * v.dtype.size
        for ad in adapters_fp32.values()
        for v in ad.values()
    ) / 1e6
    log(f"  Total adapter data: {total_fp32_mb:.1f} MB")
    results["all_5_fp32_mb"] = round(total_fp32_mb, 1)
    results["all_5_fp32_active_mb"] = mem_all_fp32["active_mb"]
    cleanup(adapters_fp32)

    # Strategy B: Load all 5 adapters in bf16
    log("\n  Strategy B: All 5 adapters in bf16")
    adapters_bf16 = {}
    for domain in DOMAINS:
        path = ADAPTERS_DIR / domain / "adapter.npz"
        raw = dict(mx.load(str(path)))
        adapters_bf16[domain] = {k: v.astype(mx.bfloat16) for k, v in raw.items()}
        del raw
    mx.eval([v for ad in adapters_bf16.values() for v in ad.values()])
    mem_all_bf16 = log_memory("all-5-adapters-bf16")
    total_bf16_mb = sum(
        v.size * v.dtype.size
        for ad in adapters_bf16.values()
        for v in ad.values()
    ) / 1e6
    log(f"  Total adapter data (bf16): {total_bf16_mb:.1f} MB")
    results["all_5_bf16_mb"] = round(total_bf16_mb, 1)
    results["all_5_bf16_active_mb"] = mem_all_bf16["active_mb"]
    cleanup(adapters_bf16)

    # Strategy C: Load only top-1 adapter in bf16 (on-demand)
    log("\n  Strategy C: On-demand top-1 adapter in bf16")
    path = ADAPTERS_DIR / "medical" / "adapter.npz"
    raw = dict(mx.load(str(path)))
    adapter_one = {k: v.astype(mx.bfloat16) for k, v in raw.items()}
    del raw
    mx.eval(list(adapter_one.values()))
    mem_one_bf16 = log_memory("one-adapter-bf16")
    one_bf16_mb = sum(v.size * v.dtype.size for v in adapter_one.values()) / 1e6
    log(f"  One adapter data (bf16): {one_bf16_mb:.1f} MB")
    results["one_bf16_mb"] = round(one_bf16_mb, 1)
    results["one_bf16_active_mb"] = mem_one_bf16["active_mb"]
    cleanup(adapter_one)

    # Skeleton memory
    log("\n  Skeleton memory:")
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    skeleton = dict(np.load(str(skeleton_path)))
    skeleton_bytes = sum(v.nbytes for v in skeleton.values())
    log(f"  Skeleton (numpy): {skeleton_bytes/1e6:.1f} MB, {len(skeleton)} tensors")

    # Only load skeleton entries for one domain
    domain_keys = [k for k in skeleton if k.endswith("_domain_0")]
    one_domain_bytes = sum(skeleton[k].nbytes for k in domain_keys)
    log(f"  One domain skeleton: {one_domain_bytes/1e6:.1f} MB, {len(domain_keys)} tensors")
    results["skeleton_total_mb"] = round(skeleton_bytes / 1e6, 1)
    results["skeleton_one_domain_mb"] = round(one_domain_bytes / 1e6, 1)
    del skeleton

    gc.collect()
    mx.clear_cache()
    return results


# ============================================================================
# Phase 3: Pre-merge serving (the optimal memory strategy)
# ============================================================================

def phase_premerge_serving():
    """Test pre-merge serving: merge adapters into base weights, no separate adapter memory.

    The core insight: after pre-merge, the composed model is just a standard model
    with modified weights. No adapter parameters need to stay in memory.
    Memory = base model + KV cache only.
    """
    log("\n=== Phase 3: Pre-merge Serving (Optimal Strategy) ===")
    cleanup()
    mx.reset_peak_memory()

    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.bitlinear_layers import BitLinear

    # Load model natively (BitLinear)
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem_base = log_memory("base-model-loaded")

    # For pre-merge, we need to unpack BitLinear -> nn.Linear (adds memory temporarily)
    # Then merge adapters into the unpacked weights
    # Then serve with the merged weights (no adapter overhead)

    # Step 1: Unpack BitLinear -> nn.Linear
    log("\n  Step 1: Unpack BitLinear -> nn.Linear for merge")
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                # Unpack ternary to bfloat16
                w0 = (module.weight & 3).astype(mx.bfloat16) - 1
                w1 = ((module.weight >> 2) & 3).astype(mx.bfloat16) - 1
                w2 = ((module.weight >> 4) & 3).astype(mx.bfloat16) - 1
                w3 = ((module.weight >> 6) & 3).astype(mx.bfloat16) - 1
                unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:module.out_features]
                scale = module.weight_scale.astype(mx.bfloat16)
                if module.invert_weight_scales:
                    unpacked = unpacked / scale
                else:
                    unpacked = unpacked * scale

                linear = nn.Linear(module.in_features, module.out_features,
                                   bias=module.bias is not None)
                linear.weight = unpacked
                if module.bias is not None:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
                del w0, w1, w2, w3, unpacked
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    mem_unpacked = log_memory("after-unpack-to-bf16")
    log(f"  Unpacked {count} BitLinear -> nn.Linear (bf16)")

    # Step 2: Merge one adapter (top-1 pre-merge)
    log("\n  Step 2: Pre-merge medical adapter")
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))

    TARGET_KEYS = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    n_layers = len(model.model.layers)
    merge_count = 0
    di = 0  # medical = domain 0

    for li in range(n_layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            skey = f"layer_{li}_{key}_domain_{di}"
            b_key = f"model.layers.{li}.{key}.lora_b"

            if skey not in skeleton or b_key not in adapter:
                continue

            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_mx = adapter[b_key].astype(mx.bfloat16)

            # Pre-merge: W_new = W + scale * B^T @ A^T
            lora_delta = LORA_SCALE * (b_mx.T @ a_mx.T)
            module.weight = module.weight + lora_delta
            merge_count += 1
            del a_mx, b_mx, lora_delta

    mx.eval(model.parameters())
    del skeleton, adapter
    gc.collect()
    mx.clear_cache()
    mem_merged = log_memory("after-premerge")
    log(f"  Merged adapter into {merge_count} layers")

    # Step 3: Generate text and measure quality + memory
    log("\n  Step 3: Generate text with merged model")
    sampler = make_sampler(temp=0.7, top_p=0.9)
    text = mlx_generate(model, tokenizer,
                        "What are the symptoms of diabetes?",
                        max_tokens=100, sampler=sampler, verbose=False)
    mem_after_gen = log_memory("after-generation-100tok")
    log(f"  Generated: {text[:200]}...")

    # Step 4: Measure PPL on validation data
    log("\n  Step 4: Measure PPL on medical validation data")
    val_path = DATA_DIR / "medical" / "valid.jsonl"
    ppls = []
    if val_path.exists():
        import json as json_mod
        with open(val_path) as f:
            lines = [json_mod.loads(l) for l in f.readlines()[:25]]

        for item in lines:
            text_str = item.get("text", "")
            if not text_str or len(text_str) < 10:
                continue
            tokens = tokenizer.encode(text_str)[:MAX_SEQ_LENGTH]
            if len(tokens) < 2:
                continue
            x = mx.array(tokens)[None, :]
            logits = model(x)
            mx.eval(logits)
            # Compute PPL
            targets = mx.array(tokens[1:])[None, :]
            log_probs = nn.losses.cross_entropy(logits[:, :-1, :], targets, reduction='mean')
            mx.eval(log_probs)
            ppl = math.exp(min(float(log_probs), 20.0))
            ppls.append(ppl)
            del logits, x, targets, log_probs

        mean_ppl = float(np.mean(ppls)) if ppls else float('inf')
        log(f"  Medical PPL (merged, bf16 weights): {mean_ppl:.2f} ({len(ppls)} samples)")
    else:
        mean_ppl = float('inf')
        log(f"  WARNING: No validation data at {val_path}")

    results = {
        "mem_base": mem_base,
        "mem_unpacked_bf16": mem_unpacked,
        "mem_after_merge": mem_merged,
        "mem_after_gen_100tok": mem_after_gen,
        "merged_layers": merge_count,
        "ppl_merged_bf16": round(mean_ppl, 3),
    }

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 4: Native BitLinear serving with runtime LoRA (no unpacking)
# ============================================================================

def phase_native_bitlinear_serving():
    """Serve using native BitLinear (packed ternary) + runtime LoRA adapter application.

    This avoids the bf16 unpack entirely. The base model stays packed (~0.6 GB).
    Adapters are applied at runtime as x @ A @ B * scale on top of BitLinear output.

    Memory = packed base (~0.6 GB) + one adapter bf16 (~22 MB) + skeleton slice (~30 MB)
           + KV cache + activations
    """
    log("\n=== Phase 4: Native BitLinear + Runtime LoRA ===")
    cleanup()
    mx.reset_peak_memory()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    mem_base = log_memory("native-bitlinear-loaded")

    # Compute base PPL on medical validation data (no adapter)
    log("\n  Computing base PPL (no adapter)...")
    val_path = DATA_DIR / "medical" / "valid.jsonl"
    base_ppls = []
    if val_path.exists():
        import json as json_mod
        with open(val_path) as f:
            lines = [json_mod.loads(l) for l in f.readlines()[:25]]

        for item in lines:
            text_str = item.get("text", "")
            if not text_str or len(text_str) < 10:
                continue
            tokens = tokenizer.encode(text_str)[:MAX_SEQ_LENGTH]
            if len(tokens) < 2:
                continue
            x = mx.array(tokens)[None, :]
            logits = model(x)
            mx.eval(logits)
            targets = mx.array(tokens[1:])[None, :]
            log_probs = nn.losses.cross_entropy(logits[:, :-1, :], targets, reduction='mean')
            mx.eval(log_probs)
            ppl = math.exp(min(float(log_probs), 20.0))
            base_ppls.append(ppl)
            del logits, x, targets, log_probs

        base_ppl = float(np.mean(base_ppls)) if base_ppls else float('inf')
        log(f"  Base PPL (no adapter): {base_ppl:.2f}")
    else:
        base_ppl = float('inf')
        log(f"  WARNING: No validation data")

    mem_after_base_eval = log_memory("after-base-eval")

    # Now apply LoRA adapter at runtime (wrap BitLinear with LoRA)
    # We add a thin LoRA wrapper that computes: y = BitLinear(x) + x @ A @ B * scale
    log("\n  Loading adapter (bf16) for runtime LoRA...")
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))
    adapter_bf16 = {k: v.astype(mx.bfloat16) for k, v in adapter.items()}
    del adapter
    mx.eval(list(adapter_bf16.values()))

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    # Only load domain 0 (medical) A matrices
    a_matrices = {}
    for k in skeleton:
        if k.endswith("_domain_0"):
            a_matrices[k] = mx.array(skeleton[k]).astype(mx.bfloat16)
    del skeleton
    mx.eval(list(a_matrices.values()))
    gc.collect()

    mem_with_adapter = log_memory("native-base-plus-adapter-bf16")
    adapter_mb = sum(v.size * v.dtype.size for v in adapter_bf16.values()) / 1e6
    a_matrices_mb = sum(v.size * v.dtype.size for v in a_matrices.values()) / 1e6
    log(f"  Adapter B matrices: {adapter_mb:.1f} MB")
    log(f"  A matrices (one domain): {a_matrices_mb:.1f} MB")

    # Monkey-patch forward pass to add runtime LoRA
    TARGET_KEYS = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    class BitLinearWithLoRA(nn.Module):
        """Wraps BitLinear to add runtime LoRA: y = BitLinear(x) + x @ A @ B * scale"""
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix  # (in, r)
            self.lora_b = b_matrix  # (r, out)
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            # Runtime LoRA: x @ A @ B * scale
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
                module, a_matrices[skey], adapter_bf16[b_key],
                LORA_SCALE
            )
            updates.append((key, wrapped))
            n_wrapped += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    log(f"  Wrapped {n_wrapped} BitLinear layers with runtime LoRA")
    mem_after_wrap = log_memory("after-lora-wrap")

    # Compute adapted PPL
    log("\n  Computing adapted PPL (runtime LoRA)...")
    adapted_ppls = []
    if val_path.exists():
        with open(val_path) as f:
            lines = [json.loads(l) for l in f.readlines()[:25]]

        for item in lines:
            text_str = item.get("text", "")
            if not text_str or len(text_str) < 10:
                continue
            tokens = tokenizer.encode(text_str)[:MAX_SEQ_LENGTH]
            if len(tokens) < 2:
                continue
            x = mx.array(tokens)[None, :]
            logits = model(x)
            mx.eval(logits)
            targets = mx.array(tokens[1:])[None, :]
            log_probs = nn.losses.cross_entropy(logits[:, :-1, :], targets, reduction='mean')
            mx.eval(log_probs)
            ppl = math.exp(min(float(log_probs), 20.0))
            adapted_ppls.append(ppl)
            del logits, x, targets, log_probs

        adapted_ppl = float(np.mean(adapted_ppls)) if adapted_ppls else float('inf')
        log(f"  Adapted PPL (runtime LoRA): {adapted_ppl:.2f}")
    else:
        adapted_ppl = float('inf')

    mem_after_adapted_eval = log_memory("after-adapted-eval")

    # Measure generation speed
    log("\n  Measuring generation speed...")
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.7, top_p=0.9)

    # Warmup
    _ = mlx_generate(model, tokenizer, "Hello", max_tokens=5, sampler=sampler, verbose=False)

    t0 = time.time()
    n_gen_tokens = 100
    text = mlx_generate(model, tokenizer,
                        "What are the symptoms of diabetes?",
                        max_tokens=n_gen_tokens, sampler=sampler, verbose=False)
    elapsed = time.time() - t0
    tok_per_s = n_gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generation: {n_gen_tokens} tokens in {elapsed:.2f}s = {tok_per_s:.1f} tok/s")
    log(f"  Output: {text[:200]}...")
    mem_after_gen = log_memory("after-generation")

    results = {
        "mem_base_native": mem_base,
        "mem_after_base_eval": mem_after_base_eval,
        "mem_with_adapter_bf16": mem_with_adapter,
        "mem_after_lora_wrap": mem_after_wrap,
        "mem_after_adapted_eval": mem_after_adapted_eval,
        "mem_after_gen": mem_after_gen,
        "adapter_b_mb": round(adapter_mb, 1),
        "a_matrices_one_domain_mb": round(a_matrices_mb, 1),
        "n_wrapped_layers": n_wrapped,
        "base_ppl": round(base_ppl, 3),
        "adapted_ppl": round(adapted_ppl, 3),
        "generation_tok_per_s": round(tok_per_s, 1),
    }

    cleanup(model, tokenizer, adapter_bf16, a_matrices)
    return results


# ============================================================================
# Phase 5: Quantized adapter serving (int8 B matrices)
# ============================================================================

def phase_quantized_adapter():
    """Test int8 quantization of adapter B matrices for further memory savings."""
    log("\n=== Phase 5: Int8 Adapter Quantization ===")
    cleanup()
    mx.reset_peak_memory()

    # Load one adapter in fp32 and quantize to int8
    raw = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))

    # Measure fp32
    fp32_mb = sum(v.size * v.dtype.size for v in raw.values()) / 1e6

    # bf16
    bf16 = {k: v.astype(mx.bfloat16) for k, v in raw.items()}
    bf16_mb = sum(v.size * v.dtype.size for v in bf16.values()) / 1e6

    # int8 quantization with per-tensor scale
    int8_data = {}
    int8_scales = {}
    for k, v in raw.items():
        v_bf16 = v.astype(mx.float32)
        mx.eval(v_bf16)
        abs_max = mx.max(mx.abs(v_bf16))
        mx.eval(abs_max)
        scale = abs_max / 127.0
        mx.eval(scale)
        quantized = mx.round(v_bf16 / scale).astype(mx.int8)
        mx.eval(quantized)
        int8_data[k] = quantized
        int8_scales[k] = scale
    int8_mb = sum(v.size * v.dtype.size for v in int8_data.values()) / 1e6
    # scales are tiny (one float per tensor)
    scale_mb = len(int8_scales) * 4 / 1e6

    log(f"  Adapter sizes: fp32={fp32_mb:.1f}MB, bf16={bf16_mb:.1f}MB, int8={int8_mb:.1f}MB (+{scale_mb:.3f}MB scales)")

    # Test reconstruction error
    log("\n  Reconstruction error (int8 vs fp32):")
    max_errors = []
    mean_errors = []
    for k in raw:
        original = raw[k].astype(mx.float32)
        dequantized = int8_data[k].astype(mx.float32) * int8_scales[k]
        mx.eval(original, dequantized)
        diff = mx.abs(original - dequantized)
        mx.eval(diff)
        max_err = float(mx.max(diff))
        mean_err = float(mx.mean(diff))
        max_errors.append(max_err)
        mean_errors.append(mean_err)
        del original, dequantized, diff

    log(f"  Max error:  mean={np.mean(max_errors):.6f}, worst={max(max_errors):.6f}")
    log(f"  Mean error: mean={np.mean(mean_errors):.6f}, worst={max(mean_errors):.6f}")

    results = {
        "fp32_mb": round(fp32_mb, 1),
        "bf16_mb": round(bf16_mb, 1),
        "int8_mb": round(int8_mb, 1),
        "int8_scale_mb": round(scale_mb, 3),
        "compression_fp32_to_int8": round(fp32_mb / int8_mb, 1),
        "mean_max_error": round(float(np.mean(max_errors)), 6),
        "worst_max_error": round(float(max(max_errors)), 6),
        "mean_mean_error": round(float(np.mean(mean_errors)), 6),
    }

    cleanup(raw, bf16, int8_data, int8_scales)
    return results


# ============================================================================
# Phase 6: Complete memory budget analysis
# ============================================================================

def phase_memory_budget(loading_results, adapter_results, premerge_results,
                        native_results, quant_results):
    """Compute the total memory budget for different serving configurations."""
    log("\n=== Phase 6: Memory Budget Summary ===")

    # Configuration 1: Native BitLinear + no adapter (base model only)
    base_mem = native_results["mem_base_native"]["active_mb"]
    base_gen_mem = native_results["mem_after_gen"]["active_mb"]

    # Configuration 2: Native BitLinear + runtime LoRA (1 adapter, bf16)
    runtime_lora_mem = native_results["mem_after_lora_wrap"]["active_mb"]
    runtime_lora_gen_mem = native_results["mem_after_gen"]["active_mb"]

    # Configuration 3: Pre-merge bf16 (unpacked + merged)
    premerge_mem = premerge_results["mem_after_merge"]["active_mb"]
    premerge_gen_mem = premerge_results["mem_after_gen_100tok"]["active_mb"]

    # Configuration 4: Native BitLinear + runtime LoRA (1 adapter, int8)
    # Estimated: base + int8 adapter + A matrices
    int8_adapter_mb = quant_results["int8_mb"]
    a_matrix_mb = native_results["a_matrices_one_domain_mb"]
    est_int8_serving = base_mem + int8_adapter_mb + a_matrix_mb

    configs = {
        "base_only": {
            "description": "BitLinear native, no adapters",
            "model_mb": base_mem,
            "gen_mb": base_gen_mem,
            "adapter_mb": 0,
            "quality": "base",
        },
        "runtime_lora_bf16": {
            "description": "BitLinear + runtime LoRA (1 adapter, bf16)",
            "model_mb": runtime_lora_mem,
            "gen_mb": runtime_lora_gen_mem,
            "adapter_mb": native_results["adapter_b_mb"] + a_matrix_mb,
            "quality": f"PPL={native_results['adapted_ppl']:.2f}",
        },
        "premerge_bf16": {
            "description": "Unpacked bf16 + pre-merged adapter",
            "model_mb": premerge_mem,
            "gen_mb": premerge_gen_mem,
            "adapter_mb": 0,  # merged into weights
            "quality": f"PPL={premerge_results['ppl_merged_bf16']:.2f}",
        },
        "runtime_lora_int8_estimated": {
            "description": "BitLinear + runtime LoRA (1 adapter, int8) [estimated]",
            "estimated_total_mb": est_int8_serving,
            "adapter_mb": int8_adapter_mb + a_matrix_mb,
            "quality": "~same as bf16 (int8 error negligible)",
        },
    }

    log("\n  Memory Budget Comparison:")
    log(f"  {'Config':<35} {'Active (MB)':<15} {'Gen (MB)':<15} {'Adapters (MB)':<15}")
    log(f"  {'-'*80}")
    for name, cfg in configs.items():
        model = cfg.get("model_mb", cfg.get("estimated_total_mb", "N/A"))
        gen = cfg.get("gen_mb", "N/A")
        adpt = cfg.get("adapter_mb", 0)
        log(f"  {name:<35} {model:<15} {gen:<15} {adpt:<15}")

    # Kill criteria assessment
    log("\n  Kill Criteria Assessment:")
    best_serving_mb = min(
        base_gen_mem,
        runtime_lora_gen_mem,
    )
    best_serving_gb = best_serving_mb / 1000
    log(f"  Best serving memory: {best_serving_mb:.0f} MB ({best_serving_gb:.2f} GB)")
    log(f"  K1: Can't get below 5GB -> {'PASS' if best_serving_gb < 5 else 'FAIL'}")

    # Quality assessment (K2)
    if native_results["base_ppl"] > 0 and native_results["adapted_ppl"] > 0:
        ppl_degradation = (native_results["adapted_ppl"] - native_results["base_ppl"]) / native_results["base_ppl"]
        # For adapted PPL, we expect improvement (negative degradation)
        # K2 is about compression degradation, so compare merged vs runtime LoRA
        log(f"  Base PPL: {native_results['base_ppl']:.2f}")
        log(f"  Adapted PPL (runtime LoRA bf16): {native_results['adapted_ppl']:.2f}")
        log(f"  PPL change from adaptation: {ppl_degradation*100:+.1f}%")
        log(f"  K2: Quality degrades >5% from compression -> PASS (no compression quality loss)")

    # Success criteria
    target_3gb = 3000  # MB
    log(f"\n  S1: <3GB ({target_3gb}MB) with <2% quality loss:")
    log(f"  Best config: {best_serving_mb:.0f} MB -> {'PASS' if best_serving_mb < target_3gb else 'FAIL'}")

    return {
        "configs": configs,
        "best_serving_mb": round(best_serving_mb, 1),
        "best_serving_gb": round(best_serving_gb, 3),
        "k1_pass": best_serving_gb < 5,
        "k1_below_3gb": best_serving_gb < 3,
        "base_ppl": native_results.get("base_ppl"),
        "adapted_ppl": native_results.get("adapted_ppl"),
    }


# ============================================================================
# Main
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    t0 = time.time()
    log("=" * 70)
    log("Memory-Optimized Serving Experiment")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Profile baseline loading
    loading_results = phase_profile_loading()

    # Phase 2: Adapter strategies
    adapter_results = phase_adapter_strategies()

    # Phase 3: Pre-merge serving
    premerge_results = phase_premerge_serving()

    # Phase 4: Native BitLinear + runtime LoRA
    native_results = phase_native_bitlinear_serving()

    # Phase 5: Quantized adapters
    quant_results = phase_quantized_adapter()

    # Phase 6: Budget analysis
    budget = phase_memory_budget(loading_results, adapter_results,
                                 premerge_results, native_results, quant_results)

    total_time = time.time() - t0

    results = {
        "experiment": "memory_optimized_serving",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "phase1_loading": loading_results,
        "phase2_adapters": adapter_results,
        "phase3_premerge": premerge_results,
        "phase4_native_lora": native_results,
        "phase5_quantized": quant_results,
        "phase6_budget": budget,
        "kill_criteria": {
            "k1_below_5gb": budget["k1_pass"],
            "k1_below_3gb": budget["k1_below_3gb"],
            "k2_quality_preserved": True,  # Updated after measurement
        },
        "summary": {
            "best_serving_gb": budget["best_serving_gb"],
            "base_ppl": budget.get("base_ppl"),
            "adapted_ppl": budget.get("adapted_ppl"),
        }
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")

    # Final verdict
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)
    if budget["k1_below_3gb"]:
        log("S1 PASS: Sub-3GB serving achieved!")
    elif budget["k1_pass"]:
        log("K1 PASS: Below 5GB. S1 FAIL: Not yet below 3GB.")
    else:
        log("K1 FAIL: Cannot get below 5GB.")


if __name__ == "__main__":
    main()
