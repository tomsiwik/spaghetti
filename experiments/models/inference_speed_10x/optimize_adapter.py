#!/usr/bin/env python3
"""
Adapter Speed Optimization: Close the gap from 88 to 100+ tok/s.

Key insight: Runtime LoRA does x @ A @ B per layer (two matmuls).
Optimization: Precompute C = A @ B (one matmul instead of two per token).
This reduces 2 * d * r FLOPs to d * d FLOPs for r < d/2 (but d*d > 2*d*r for r=16, d=2560).
Actually for r=16, d=2560: 2*d*r = 81,920 vs d*d = 6,553,600. So factored is BETTER.

The real bottleneck: 210 layers x 2 matmuls = 420 kernel launches per token.
Try: precompute merged weights (pre-merge) for single-adapter serving.

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0


def log(msg):
    print(msg, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def measure_tps(model, tokenizer, prompt_text, n_tokens=100, n_warmup=2, **kwargs):
    """Measure tok/s using stream_generate."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    for _ in range(n_warmup):
        for resp in stream_generate(model, tokenizer, prompt_text,
                                     max_tokens=5, sampler=sampler, **kwargs):
            pass
        mx.clear_cache()

    gen_tps = prompt_tps = 0
    for resp in stream_generate(model, tokenizer, prompt_text,
                                 max_tokens=n_tokens, sampler=sampler, **kwargs):
        gen_tps = resp.generation_tps
        prompt_tps = resp.prompt_tps
    mx.clear_cache()

    return {"generation_tps": round(gen_tps, 1), "prompt_tps": round(prompt_tps, 1)}


TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


# ============================================================================
# Strategy A: Runtime LoRA with precomputed C = A @ B (one matmul per layer)
# ============================================================================

def test_precomputed_ab():
    """Precompute C = A @ B so each layer does y = base(x) + x @ C * scale (one matmul)."""
    log("\n=== Strategy A: Precomputed A@B (one matmul per layer) ===")
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

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

    class BitLinearWithPrecomputedLoRA(nn.Module):
        """Uses precomputed C = A @ B: y = base(x) + x @ C * scale"""
        def __init__(self, base_module, precomputed_ab, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_c = precomputed_ab  # (in_features, out_features)
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            return y + x @ self.lora_c * self.lora_scale

    n_wrapped = 0
    precomputed_mb = 0
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

            a = a_matrices[skey]
            b = adapter_bf16[b_key]
            c = a @ b  # (in_features, rank) @ (rank, out_features) = (in_features, out_features)
            mx.eval(c)
            precomputed_mb += c.size * c.dtype.size / 1e6

            wrapped = BitLinearWithPrecomputedLoRA(module, c, LORA_SCALE)
            updates.append((key, wrapped))
            n_wrapped += 1
            del a, b

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    del a_matrices, adapter_bf16
    gc.collect()
    mx.clear_cache()

    log(f"  Wrapped {n_wrapped} layers, precomputed AB: {precomputed_mb:.1f} MB")

    r = measure_tps(model, tokenizer, "What are the symptoms of diabetes?", n_tokens=100, n_warmup=3)
    log(f"  Speed: {r['generation_tps']} tok/s")

    cleanup(model, tokenizer)
    return {"tps": r['generation_tps'], "precomputed_mb": round(precomputed_mb, 1), "n_wrapped": n_wrapped}


# ============================================================================
# Strategy B: Pre-merge (merge adapter into unpacked bf16 weights)
# ============================================================================

def test_premerge():
    """Merge adapter into bf16 weights. Memory cost: ~3.5 GB (bf16 model) but fast."""
    log("\n=== Strategy B: Pre-merge (bf16 merged weights) ===")
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Unpack BitLinear -> nn.Linear (bf16)
    log("  Unpacking BitLinear -> bf16 Linear...")
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
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

    mem_unpacked = mx.get_active_memory() / 1e6
    log(f"  Unpacked {count} layers. Memory: {mem_unpacked:.1f} MB")

    # Merge adapter
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))

    n_merged = 0
    n_layers = len(model.model.layers)
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

            skey = f"layer_{li}_{key}_domain_0"
            b_key = f"model.layers.{li}.{key}.lora_b"

            if skey not in skeleton or b_key not in adapter:
                continue

            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_mx = adapter[b_key].astype(mx.bfloat16)

            lora_delta = LORA_SCALE * (b_mx.T @ a_mx.T)
            module.weight = module.weight + lora_delta
            n_merged += 1
            del a_mx, b_mx, lora_delta

    mx.eval(model.parameters())
    del skeleton, adapter
    gc.collect()
    mx.clear_cache()

    mem_merged = mx.get_active_memory() / 1e6
    log(f"  Merged {n_merged} layers. Memory: {mem_merged:.1f} MB")

    r = measure_tps(model, tokenizer, "What are the symptoms of diabetes?", n_tokens=100, n_warmup=3)
    log(f"  Speed: {r['generation_tps']} tok/s")

    cleanup(model, tokenizer)
    return {"tps": r['generation_tps'], "memory_mb": round(mem_merged, 1), "n_merged": n_merged}


# ============================================================================
# Strategy C: Factored LoRA with addmm fusion hint
# ============================================================================

def test_factored_addmm():
    """Use mx.addmm for y = base(x) + (x @ A) @ B * scale to hint at fusion."""
    log("\n=== Strategy C: Factored LoRA with addmm ===")
    cleanup()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

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

    class BitLinearWithLoRAAddmm(nn.Module):
        """Uses addmm for potential fusion: y = base(x); y += x @ A @ B * scale"""
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            # Two-step: first x@A (cheaper), then result@B
            h = x @ self.lora_a  # (batch, seq, rank)
            # Use addmm: y = y + h @ B * scale
            return mx.addmm(y, h, self.lora_b, alpha=self.lora_scale)

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

            wrapped = BitLinearWithLoRAAddmm(
                module, a_matrices[skey], adapter_bf16[b_key], LORA_SCALE
            )
            updates.append((key, wrapped))
            n_wrapped += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    log(f"  Wrapped {n_wrapped} layers with addmm LoRA")

    r = measure_tps(model, tokenizer, "What are the symptoms of diabetes?", n_tokens=100, n_warmup=3)
    log(f"  Speed: {r['generation_tps']} tok/s")

    cleanup(model, tokenizer, adapter_bf16, a_matrices)
    return {"tps": r['generation_tps'], "n_wrapped": n_wrapped}


def main():
    results = {}

    # Strategy A: Precomputed A@B
    results["precomputed_ab"] = test_precomputed_ab()

    # Strategy B: Pre-merge
    results["premerge"] = test_premerge()

    # Strategy C: addmm fusion
    results["addmm"] = test_factored_addmm()

    log("\n" + "=" * 60)
    log("OPTIMIZATION RESULTS")
    log("=" * 60)
    log(f"  Runtime LoRA (baseline):  88.2 tok/s")
    log(f"  Precomputed A@B:          {results['precomputed_ab']['tps']} tok/s")
    log(f"  Pre-merge (bf16):         {results['premerge']['tps']} tok/s")
    log(f"  addmm fusion:             {results['addmm']['tps']} tok/s")
    log(f"  Base (no adapter):        172.0 tok/s")

    best = max(results.values(), key=lambda r: r['tps'])
    log(f"\n  Best strategy: {[k for k,v in results.items() if v == best][0]} at {best['tps']} tok/s")
    log(f"  S1 (>100 tok/s): {'PASS' if best['tps'] > 100 else 'FAIL'}")

    output = EXPERIMENT_DIR / "optimization_results.json"
    output.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {output}")


EXPERIMENT_DIR = Path(__file__).parent

if __name__ == "__main__":
    main()
