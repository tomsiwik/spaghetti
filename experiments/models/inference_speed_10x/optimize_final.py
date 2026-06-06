#!/usr/bin/env python3
"""
Final optimization push: get adapter composition above 100 tok/s.

Current best: addmm at 97.3 tok/s. Need ~3% more.

Strategies:
A) addmm with float16 (avoid bfloat16 conversion overhead)
B) addmm with precomputed scale*B (avoid multiplication per token)
C) Reduced adapter: only wrap attention layers, skip MLP layers
D) Profile which layers are bottleneck

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

TEST_PROMPTS = [
    "What are the symptoms of diabetes?",
    "Explain the concept of neural networks.",
    "The capital of France is",
    "In machine learning, gradient descent works by",
]

TARGET_KEYS_ALL = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

TARGET_KEYS_ATTN_ONLY = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj",
]


def log(msg):
    print(msg, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def measure_tps(model, tokenizer, prompt_text="What are the symptoms of diabetes?",
                n_tokens=100, n_warmup=3, **kwargs):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    for _ in range(n_warmup):
        for resp in stream_generate(model, tokenizer, prompt_text,
                                     max_tokens=5, sampler=sampler, **kwargs):
            pass
        mx.clear_cache()

    gen_tps = 0
    for resp in stream_generate(model, tokenizer, prompt_text,
                                 max_tokens=n_tokens, sampler=sampler, **kwargs):
        gen_tps = resp.generation_tps
    mx.clear_cache()
    return round(gen_tps, 1)


def measure_tps_all_prompts(model, tokenizer, n_tokens=100, n_warmup=2, **kwargs):
    """Measure tok/s across all 4 TEST_PROMPTS. Returns (mean, std, values)."""
    values = []
    for prompt in TEST_PROMPTS:
        tps = measure_tps(model, tokenizer, prompt_text=prompt,
                          n_tokens=n_tokens, n_warmup=n_warmup, **kwargs)
        values.append(tps)
    mean = round(float(np.mean(values)), 1)
    std = round(float(np.std(values)), 1)
    return mean, std, values


def load_adapter_data():
    """Load medical adapter and skeleton A-matrices."""
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

    return adapter_bf16, a_matrices


def wrap_model_addmm(model, adapter_bf16, a_matrices, target_keys, prescale=False):
    """Wrap model layers with addmm LoRA. Returns number of wrapped layers."""
    from mlx_lm.models.bitlinear_layers import BitLinear

    class BitLinearWithLoRAAddmm(nn.Module):
        def __init__(self, base_module, a_matrix, b_matrix, lora_scale):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b = b_matrix
            self.lora_scale = lora_scale

        def __call__(self, x):
            y = self.base(x)
            h = x @ self.lora_a
            return mx.addmm(y, h, self.lora_b, alpha=self.lora_scale)

    class BitLinearWithLoRAAddmmPrescaled(nn.Module):
        """B is pre-scaled: B_scaled = B * scale, so we just do addmm(y, h, B_scaled)."""
        def __init__(self, base_module, a_matrix, b_scaled):
            super().__init__()
            self.base = base_module
            self.lora_a = a_matrix
            self.lora_b_scaled = b_scaled

        def __call__(self, x):
            y = self.base(x)
            h = x @ self.lora_a
            return mx.addmm(y, h, self.lora_b_scaled)

    n_wrapped = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        updates = []
        for key in target_keys:
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

            if prescale:
                b_scaled = adapter_bf16[b_key] * LORA_SCALE
                mx.eval(b_scaled)
                wrapped = BitLinearWithLoRAAddmmPrescaled(
                    module, a_matrices[skey], b_scaled
                )
            else:
                wrapped = BitLinearWithLoRAAddmm(
                    module, a_matrices[skey], adapter_bf16[b_key], LORA_SCALE
                )
            updates.append((key, wrapped))
            n_wrapped += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    return n_wrapped


def main():
    from mlx_lm import load

    results = {}

    # Test A: addmm baseline — measured across all 4 prompts for variance
    log("=== A: addmm baseline (all 4 prompts) ===")
    cleanup()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    adapter_bf16, a_matrices = load_adapter_data()
    n = wrap_model_addmm(model, adapter_bf16, a_matrices, TARGET_KEYS_ALL)
    log(f"  Wrapped {n} layers")
    mean_tps, std_tps, per_prompt = measure_tps_all_prompts(model, tokenizer)
    log(f"  Speed: {mean_tps} ± {std_tps} tok/s (per-prompt: {per_prompt})")
    results["addmm_baseline"] = mean_tps
    results["addmm_baseline_std"] = std_tps
    results["addmm_baseline_per_prompt"] = per_prompt
    cleanup(model, tokenizer, adapter_bf16, a_matrices)

    # Test B: addmm with prescaled B
    log("\n=== B: addmm with prescaled B ===")
    cleanup()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    adapter_bf16, a_matrices = load_adapter_data()
    n = wrap_model_addmm(model, adapter_bf16, a_matrices, TARGET_KEYS_ALL, prescale=True)
    log(f"  Wrapped {n} layers")
    tps = measure_tps(model, tokenizer)
    log(f"  Speed: {tps} tok/s")
    results["addmm_prescaled"] = tps
    cleanup(model, tokenizer, adapter_bf16, a_matrices)

    # Test C: addmm attention-only — measured across all 4 prompts for variance
    log("\n=== C: addmm attention-only (all 4 prompts) ===")
    cleanup()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    adapter_bf16, a_matrices = load_adapter_data()
    n = wrap_model_addmm(model, adapter_bf16, a_matrices, TARGET_KEYS_ATTN_ONLY)
    log(f"  Wrapped {n} layers (attn only)")
    mean_tps, std_tps, per_prompt = measure_tps_all_prompts(model, tokenizer)
    log(f"  Speed: {mean_tps} ± {std_tps} tok/s (per-prompt: {per_prompt})")
    results["addmm_attn_only"] = mean_tps
    results["addmm_attn_only_std"] = std_tps
    results["addmm_attn_only_per_prompt"] = per_prompt
    cleanup(model, tokenizer, adapter_bf16, a_matrices)

    # Test D: addmm prescaled + longer generation (amortize)
    log("\n=== D: addmm prescaled, 200 tokens ===")
    cleanup()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    adapter_bf16, a_matrices = load_adapter_data()
    n = wrap_model_addmm(model, adapter_bf16, a_matrices, TARGET_KEYS_ALL, prescale=True)
    log(f"  Wrapped {n} layers")
    tps = measure_tps(model, tokenizer, n_tokens=200)
    log(f"  Speed (200 tok): {tps} tok/s")
    results["addmm_prescaled_200tok"] = tps

    # Also measure wall-clock for 200 tokens
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    _ = mlx_generate(model, tokenizer, "Hello", max_tokens=5, sampler=sampler, verbose=False)
    t0 = time.perf_counter()
    _ = mlx_generate(model, tokenizer, "What are the symptoms of diabetes?",
                     max_tokens=200, sampler=sampler, verbose=False)
    wallclock = time.perf_counter() - t0
    wallclock_tps = round(200 / wallclock, 1)
    log(f"  Wall-clock (200 tok): {wallclock_tps} tok/s")
    results["addmm_prescaled_200tok_wallclock"] = wallclock_tps

    cleanup(model, tokenizer, adapter_bf16, a_matrices)

    # Test E: addmm prescaled + 500 tokens
    log("\n=== E: addmm prescaled, 500 tokens ===")
    cleanup()
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    adapter_bf16, a_matrices = load_adapter_data()
    n = wrap_model_addmm(model, adapter_bf16, a_matrices, TARGET_KEYS_ALL, prescale=True)
    tps = measure_tps(model, tokenizer, n_tokens=500)
    log(f"  Speed (500 tok): {tps} tok/s")
    results["addmm_prescaled_500tok"] = tps
    cleanup(model, tokenizer, adapter_bf16, a_matrices)

    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    for k, v in results.items():
        log(f"  {k}: {v} tok/s")
    scalar_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    best = max(scalar_results.values())
    best_key = [k for k, v in scalar_results.items() if v == best][0]
    log(f"\n  Best: {best_key} = {best} tok/s")
    log(f"  S1 full adapter (>100 tok/s): {'PASS' if results['addmm_baseline'] > 100 else 'FAIL'} ({results['addmm_baseline']} ± {results['addmm_baseline_std']} tok/s)")
    log(f"  S1 attn-only (>100 tok/s): {'PASS' if results['addmm_attn_only'] > 100 else 'FAIL'} ({results['addmm_attn_only']} ± {results['addmm_attn_only_std']} tok/s, quality not validated)")

    output = EXPERIMENT_DIR / "final_results.json"
    output.write_text(json.dumps(results, indent=2))
    log(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
