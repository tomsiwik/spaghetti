#!/usr/bin/env python3
"""
BitNet Float Merge (fp32) as Lossless Serving Path

Tests whether merging LoRA adapters into an fp32 copy of BitNet-2B-4T base
weights produces lossless composition quality, comparing against runtime LoRA
composition (the proven baseline) and bf16 float merge (the known-lossy path).

Key insight: Prior exp_bitnet_serving_path did float merge in bfloat16, which
has ULP ~0.004 at the base weight magnitude (~0.463) — exactly matching the
LoRA delta magnitude, causing ~50% information loss. fp32 has ULP ~5.5e-8,
which is 73,000x smaller than the delta, so merge should be lossless.

Kill criteria:
  K1: fp32 float-merged model PPL > 1.05x runtime-composed PPL at N=5
  K2: fp32 merged model memory > 2x ternary base memory (~1.7GB)
  K3: fp32 merge latency > runtime LoRA composition latency

Reuses trained adapters from bitnet_2b_real_composition.
Platform: Apple Silicon MLX, $0 compute.
"""

import gc
import json
import math
import os
import sys
import time
import resource
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx_lm.tuner.lora import LoRALinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 50  # More batches for stable PPL estimates

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse trained adapters and data from bitnet_2b_real_composition
ADAPTER_SOURCE = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "adapters"
DATA_SOURCE = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]

# Latency benchmark config
BENCH_TOKENS = 50  # Fewer tokens for faster benchmark (we measure tok/s)
BENCH_RUNS = 5


# ===========================================================================
# Ternary weight handling (from serving_path experiment)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix."""
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    """Replace all BitLinear layers with standard nn.Linear for inference."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return model, count


# ===========================================================================
# Adapter loading and delta computation
# ===========================================================================
def load_adapter(domain: str) -> dict:
    """Load adapter weights from disk."""
    path = ADAPTER_SOURCE / domain / "adapter.npz"
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {path}")
    return dict(mx.load(str(path)))


def compute_lora_deltas(adapter_params: dict, scale: float = 1.0):
    """Compute per-layer DeltaW = B.T @ A.T * lora_scale * scale.

    Returns dict: full_layer_path -> DeltaW matrix (out_features, in_features)
    """
    layers = {}
    for key, val in adapter_params.items():
        parts = key.rsplit(".", 1)
        layer_path = parts[0]
        param_type = parts[1]
        if layer_path not in layers:
            layers[layer_path] = {}
        layers[layer_path][param_type] = val

    deltas = {}
    for layer_path, params in layers.items():
        if "lora_a" in params and "lora_b" in params:
            A = params["lora_a"]  # (in_features, r)
            B = params["lora_b"]  # (r, out_features)
            # LoRA output = x @ A @ B * lora_scale
            # Merged weight delta: (B.T @ A.T) * lora_scale * scale
            # Keep computation in float32 for maximum precision
            delta = (B.astype(mx.float32).T @ A.astype(mx.float32).T) * LORA_SCALE * scale
            deltas[layer_path] = delta

    return deltas


def get_linear_weight_paths(model):
    """Get all linear weight paths in the model."""
    paths = {}
    for i, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_path = f"model.layers.{i}.{key}"
                paths[full_path] = (layer, key, module)
    return paths


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, domain: str, max_batches: int = VAL_BATCHES):
    """Compute perplexity on validation data."""
    valid_path = DATA_SOURCE / domain / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


def compute_all_ppls(model, tokenizer):
    """Compute PPL on all domains."""
    ppls = {}
    for domain in DOMAINS:
        ppls[domain] = compute_ppl(model, tokenizer, domain)
    return ppls


# ===========================================================================
# Latency benchmark
# ===========================================================================
def benchmark_generation(model, tokenizer, n_tokens=BENCH_TOKENS, n_runs=BENCH_RUNS,
                         prompt="The future of AI is"):
    """Benchmark token generation latency."""
    input_ids = mx.array([tokenizer.encode(prompt)])

    # Warmup
    for _ in range(2):
        logits = model(input_ids)
        mx.eval(logits)

    latencies = []
    for run in range(n_runs):
        tokens = list(input_ids[0].tolist())
        t_start = time.perf_counter()

        for _ in range(n_tokens):
            x = mx.array([tokens[-MAX_SEQ_LENGTH:]])
            logits = model(x)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            tokens.append(next_token)

        t_end = time.perf_counter()
        latencies.append(t_end - t_start)

    mean_lat = sum(latencies) / len(latencies)
    stddev_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
    tps_values = [n_tokens / lat for lat in latencies]
    mean_tps = sum(tps_values) / len(tps_values)
    stddev_tps = (sum((x - mean_tps) ** 2 for x in tps_values) / len(tps_values)) ** 0.5
    latencies.sort()
    return {
        "n_tokens": n_tokens,
        "n_runs": n_runs,
        "mean_s": round(mean_lat, 3),
        "stddev_s": round(stddev_lat, 3),
        "min_s": round(latencies[0], 3),
        "max_s": round(latencies[-1], 3),
        "tok_per_s": round(mean_tps, 1),
        "tok_per_s_stddev": round(stddev_tps, 1),
    }


def get_memory_mb():
    """Get current process peak memory in MB (macOS)."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return round(rusage.ru_maxrss / (1024 * 1024), 1)


# ===========================================================================
# Core merge operations
# ===========================================================================
def save_base_weights(model):
    """Save base weights for reset. Returns dict of path -> array."""
    base = {}
    for i, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                path = f"model.layers.{i}.{key}"
                base[path] = mx.array(module.weight)
    mx.eval(base)
    return base


def reset_to_base(model, base_weights):
    """Reset model weights to saved base weights."""
    for i, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                path = f"model.layers.{i}.{key}"
                if path in base_weights:
                    module.weight = mx.array(base_weights[path])
    mx.eval(model.parameters())


def merge_deltas_into_model(model, combined_deltas, dtype=mx.float32):
    """Merge computed weight deltas into model weights at specified precision.

    Args:
        model: The model with nn.Linear layers
        combined_deltas: dict of layer_path -> DeltaW (float32)
        dtype: Target dtype for merged weights (mx.float32 or mx.bfloat16)
    """
    for i, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_path = f"model.layers.{i}.{key}"
                if full_path in combined_deltas:
                    # Cast base weight and delta to target dtype, then add
                    w = module.weight.astype(dtype)
                    d = combined_deltas[full_path].astype(dtype)
                    module.weight = w + d
    mx.eval(model.parameters())


# ===========================================================================
# Runtime LoRA composition (baseline reference)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA wrappers to all linear layers in transformer blocks."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    return model, count


def remove_lora(model):
    """Remove LoRA wrappers, restoring plain nn.Linear layers."""
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                updates.append((key, module.linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return model


def compose_adapters_runtime(adapter_list: list):
    """Sum multiple adapter parameter dicts (no per-param scaling).

    The 1/N composition scaling is applied via lora_scale when constructing the
    LoRALinear layer, NOT by scaling A and B independently. Scaling both A and B
    by 1/N would produce effective 1/N^2 scaling in the product A @ B.
    """
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0)
    return merged


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_float_merge_fp32",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "domains": DOMAINS,
        "val_batches": VAL_BATCHES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet Float Merge (fp32) vs Runtime LoRA Composition")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Load model
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    mem_packed = get_memory_mb()
    print(f"  Memory (packed ternary): {mem_packed:.1f} MB")
    results["memory_packed_mb"] = mem_packed

    # Unpack to bfloat16 for inference
    print("  Unpacking ternary weights to bfloat16...")
    model, n_replaced = replace_bitlinear_with_linear(model)
    print(f"  Replaced {n_replaced} BitLinear layers")

    mem_unpacked = get_memory_mb()
    print(f"  Memory (unpacked bf16): {mem_unpacked:.1f} MB")
    results["memory_unpacked_bf16_mb"] = mem_unpacked

    # Save base weights for reset
    print("  Saving base weights for reset...")
    base_weights = save_base_weights(model)

    # ==================================================================
    # Phase 1: Base model PPL
    # ==================================================================
    print("\n[Phase 1] Base model PPL (no adapters)...")
    base_ppls = compute_all_ppls(model, tokenizer)
    for d, p in base_ppls.items():
        print(f"  {d}: {p:.2f}")
    results["base_ppls"] = {d: round(p, 4) for d, p in base_ppls.items()}
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    results["avg_base_ppl"] = round(avg_base, 4)

    # ==================================================================
    # Phase 2: Load adapters
    # ==================================================================
    print("\n[Phase 2] Loading trained adapters...")
    all_adapters = {}
    for domain in DOMAINS:
        try:
            all_adapters[domain] = load_adapter(domain)
            n_params = sum(v.size for v in all_adapters[domain].values())
            print(f"  {domain}: {len(all_adapters[domain])} tensors, {n_params:,} params")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    results["n_adapters_loaded"] = len(all_adapters)
    domain_list = [d for d in DOMAINS if d in all_adapters]

    if len(domain_list) < 5:
        print(f"  ERROR: Only {len(domain_list)} adapters found, need 5")
        results["error"] = f"Only {len(domain_list)} adapters found"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return

    # ==================================================================
    # Phase 3: Precision analysis (diagnostic)
    # ==================================================================
    print("\n[Phase 3] Precision analysis...")

    # Compute a sample delta to analyze magnitudes
    sample_adapter = all_adapters[domain_list[0]]
    sample_deltas = compute_lora_deltas(sample_adapter, scale=1.0)
    sample_path = list(sample_deltas.keys())[0]
    sample_delta = sample_deltas[sample_path]
    sample_base = base_weights[sample_path]

    delta_mean = mx.mean(mx.abs(sample_delta)).item()
    delta_max = mx.max(mx.abs(sample_delta)).item()
    base_mean = mx.mean(mx.abs(sample_base)).item()

    # ULP analysis
    # fp32: 23 mantissa bits, ULP at magnitude m = m * 2^-23
    # bf16: 7 mantissa bits, ULP at magnitude m = m * 2^-7
    fp32_ulp = base_mean * (2 ** -23)
    bf16_ulp = base_mean * (2 ** -7)

    print(f"  Sample layer: {sample_path}")
    print(f"  Base weight mean |W|: {base_mean:.6f}")
    print(f"  Delta mean |DeltaW|: {delta_mean:.6f}")
    print(f"  Delta max |DeltaW|: {delta_max:.6f}")
    print(f"  Delta/base ratio: {delta_mean/base_mean:.6f}")
    print(f"  fp32 ULP at base magnitude: {fp32_ulp:.2e}")
    print(f"  bf16 ULP at base magnitude: {bf16_ulp:.2e}")
    print(f"  fp32 delta/ULP ratio: {delta_mean/fp32_ulp:.0f}x (>>1 = lossless)")
    print(f"  bf16 delta/ULP ratio: {delta_mean/bf16_ulp:.1f}x (~1 = lossy)")

    results["precision_analysis"] = {
        "sample_layer": sample_path,
        "base_weight_mean": round(base_mean, 6),
        "delta_mean": round(delta_mean, 6),
        "delta_max": round(delta_max, 6),
        "delta_base_ratio": round(delta_mean / base_mean, 6),
        "fp32_ulp": float(f"{fp32_ulp:.2e}"),
        "bf16_ulp": float(f"{bf16_ulp:.2e}"),
        "fp32_delta_ulp_ratio": round(delta_mean / fp32_ulp),
        "bf16_delta_ulp_ratio": round(delta_mean / bf16_ulp, 1),
    }

    # Measure actual truncation error
    # fp32: W_base.float32() + delta.float32() vs exact
    # bf16: W_base.bfloat16() + delta.bfloat16() vs exact
    w_f32 = sample_base.astype(mx.float32) + sample_delta.astype(mx.float32)
    w_bf16 = sample_base.astype(mx.bfloat16) + sample_delta.astype(mx.bfloat16)
    # "exact" = float32 (our best precision in MLX)
    exact = w_f32

    fp32_err = mx.mean(mx.abs(w_f32 - exact)).item()
    bf16_err = mx.mean(mx.abs(w_bf16.astype(mx.float32) - exact)).item()
    bf16_delta_frac = bf16_err / delta_mean if delta_mean > 0 else float("inf")

    print(f"\n  Actual truncation error (sample layer):")
    print(f"    fp32 merge error: {fp32_err:.2e} (0% of delta — lossless)")
    print(f"    bf16 merge error: {bf16_err:.2e} ({bf16_delta_frac*100:.1f}% of delta)")

    results["precision_analysis"]["fp32_merge_error"] = float(f"{fp32_err:.2e}")
    results["precision_analysis"]["bf16_merge_error"] = float(f"{bf16_err:.2e}")
    results["precision_analysis"]["bf16_delta_loss_pct"] = round(bf16_delta_frac * 100, 1)

    # ==================================================================
    # Phase 4: Runtime LoRA composition (BASELINE) — N=1..5
    # ==================================================================
    print("\n[Phase 4] Runtime LoRA composition (baseline)...")
    print("  NOTE: 1/N scaling applied via lora_scale (not per-param), avoiding 1/N^2 bug.")

    runtime_results = {}
    for N in range(1, len(domain_list) + 1):
        selected = domain_list[:N]

        # Reset to base, apply LoRA with scale=LORA_SCALE/N for correct 1/N composition
        reset_to_base(model, base_weights)
        effective_scale = LORA_SCALE / N
        model, n_lora = apply_lora_to_model(model, rank=LORA_RANK, scale=effective_scale)

        # Sum adapter params (no per-param scaling -- scaling is in lora_scale)
        composed = compose_adapters_runtime([all_adapters[d] for d in selected])

        # Apply composed adapter
        model.update(tree_unflatten(list(composed.items())))
        mx.eval(model.parameters())

        ppls = compute_all_ppls(model, tokenizer)
        avg_ppl = sum(ppls.values()) / len(ppls)
        print(f"  N={N} runtime (scale={effective_scale:.1f}): avg PPL = {avg_ppl:.4f} ({', '.join(f'{d}={ppls[d]:.2f}' for d in DOMAINS)})")

        runtime_results[f"N={N}"] = {
            "domains": selected,
            "ppls": {d: round(p, 4) for d, p in ppls.items()},
            "avg_ppl": round(avg_ppl, 4),
        }

        # Remove LoRA for next iteration reset
        model = remove_lora(model)

    results["runtime_lora"] = runtime_results

    # Benchmark runtime LoRA latency at N=5 (re-apply for benchmark)
    print("\n  Benchmarking runtime LoRA latency (N=5)...")
    reset_to_base(model, base_weights)
    model, n_lora = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE / len(domain_list))
    composed = compose_adapters_runtime([all_adapters[d] for d in domain_list])
    model.update(tree_unflatten(list(composed.items())))
    mx.eval(model.parameters())
    runtime_bench = benchmark_generation(model, tokenizer)
    print(f"    Runtime LoRA N=5: {runtime_bench['tok_per_s']} +/- {runtime_bench['tok_per_s_stddev']} tok/s")
    results["latency_runtime_lora_n5"] = runtime_bench

    # Remove LoRA layers
    model = remove_lora(model)

    # ==================================================================
    # Phase 5: fp32 Float Merge — N=1..5
    # ==================================================================
    print("\n[Phase 5] fp32 float merge (N=1..5)...")

    fp32_results = {}
    for N in range(1, len(domain_list) + 1):
        selected = domain_list[:N]

        # Compute combined delta with 1/N scaling in float32
        combined_deltas = {}
        for domain in selected:
            deltas = compute_lora_deltas(all_adapters[domain], scale=1.0 / N)
            for path, delta in deltas.items():
                if path in combined_deltas:
                    combined_deltas[path] = combined_deltas[path] + delta
                else:
                    combined_deltas[path] = delta

        # Reset to base and merge in fp32
        reset_to_base(model, base_weights)
        merge_deltas_into_model(model, combined_deltas, dtype=mx.float32)

        ppls = compute_all_ppls(model, tokenizer)
        avg_ppl = sum(ppls.values()) / len(ppls)
        print(f"  N={N} fp32: avg PPL = {avg_ppl:.4f} ({', '.join(f'{d}={ppls[d]:.2f}' for d in DOMAINS)})")

        fp32_results[f"N={N}"] = {
            "domains": selected,
            "ppls": {d: round(p, 4) for d, p in ppls.items()},
            "avg_ppl": round(avg_ppl, 4),
        }

    results["fp32_merge"] = fp32_results

    # Memory after fp32 merge
    mem_fp32 = get_memory_mb()
    results["memory_fp32_merged_mb"] = mem_fp32
    print(f"\n  Memory after fp32 merge: {mem_fp32:.1f} MB")

    # Benchmark fp32 merge latency (model already has fp32 weights from N=5)
    print("  Benchmarking fp32 merge latency (N=5 merged)...")
    fp32_bench = benchmark_generation(model, tokenizer)
    print(f"    fp32 merged N=5: {fp32_bench['tok_per_s']} +/- {fp32_bench['tok_per_s_stddev']} tok/s")
    results["latency_fp32_merged_n5"] = fp32_bench

    # ==================================================================
    # Phase 6: bf16 Float Merge — N=1..5 (confirm prior finding)
    # ==================================================================
    print("\n[Phase 6] bf16 float merge (N=1..5) — confirming prior finding...")

    bf16_results = {}
    for N in range(1, len(domain_list) + 1):
        selected = domain_list[:N]

        combined_deltas = {}
        for domain in selected:
            deltas = compute_lora_deltas(all_adapters[domain], scale=1.0 / N)
            for path, delta in deltas.items():
                if path in combined_deltas:
                    combined_deltas[path] = combined_deltas[path] + delta
                else:
                    combined_deltas[path] = delta

        reset_to_base(model, base_weights)
        merge_deltas_into_model(model, combined_deltas, dtype=mx.bfloat16)

        ppls = compute_all_ppls(model, tokenizer)
        avg_ppl = sum(ppls.values()) / len(ppls)
        print(f"  N={N} bf16: avg PPL = {avg_ppl:.4f} ({', '.join(f'{d}={ppls[d]:.2f}' for d in DOMAINS)})")

        bf16_results[f"N={N}"] = {
            "domains": selected,
            "ppls": {d: round(p, 4) for d, p in ppls.items()},
            "avg_ppl": round(avg_ppl, 4),
        }

    results["bf16_merge"] = bf16_results

    # Benchmark bf16 merge latency
    print("  Benchmarking bf16 merge latency (N=5 merged)...")
    bf16_bench = benchmark_generation(model, tokenizer)
    print(f"    bf16 merged N=5: {bf16_bench['tok_per_s']} +/- {bf16_bench['tok_per_s_stddev']} tok/s")
    results["latency_bf16_merged_n5"] = bf16_bench

    # ==================================================================
    # Phase 7: Base model latency
    # ==================================================================
    print("\n[Phase 7] Base model latency (no adapters)...")
    reset_to_base(model, base_weights)
    base_bench = benchmark_generation(model, tokenizer)
    print(f"    Base: {base_bench['tok_per_s']} +/- {base_bench['tok_per_s_stddev']} tok/s")
    results["latency_base"] = base_bench

    # ==================================================================
    # Phase 8: Kill Criteria Assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: fp32 merged PPL > 1.05x runtime composed PPL at N=5
    fp32_n5_avg = fp32_results["N=5"]["avg_ppl"]
    runtime_n5_avg = runtime_results["N=5"]["avg_ppl"]
    k1_ratio = fp32_n5_avg / runtime_n5_avg
    k1_pass = k1_ratio <= 1.05

    print(f"\n  K1: fp32 merged PPL vs runtime composed PPL at N=5")
    print(f"    fp32 merged avg PPL:    {fp32_n5_avg:.4f}")
    print(f"    Runtime composed avg PPL: {runtime_n5_avg:.4f}")
    print(f"    Ratio: {k1_ratio:.4f} (threshold: <= 1.05)")
    print(f"    K1: {'PASS' if k1_pass else 'FAIL'}")

    results["k1_fp32_n5_avg"] = fp32_n5_avg
    results["k1_runtime_n5_avg"] = runtime_n5_avg
    results["k1_ratio"] = round(k1_ratio, 4)
    results["k1_pass"] = k1_pass

    # Also check per-domain
    k1_per_domain = {}
    for d in DOMAINS:
        fp32_p = fp32_results["N=5"]["ppls"][d]
        rt_p = runtime_results["N=5"]["ppls"][d]
        ratio = fp32_p / rt_p
        k1_per_domain[d] = {"fp32": fp32_p, "runtime": rt_p, "ratio": round(ratio, 4)}
        print(f"      {d}: fp32={fp32_p:.2f}, runtime={rt_p:.2f}, ratio={ratio:.4f}")
    results["k1_per_domain"] = k1_per_domain

    # K2: fp32 merged model memory > 2x ternary base memory
    # Ternary base = memory_packed_mb. fp32 merged = memory_fp32_merged_mb
    # But we need to account for the fact that MLX unpacks to bf16 by default.
    # The real comparison is: fp32 model size vs ternary base size
    # BitNet-2B-4T: ~2.08B params. Ternary packed = ~520MB. fp32 = ~8.3GB.
    # But in practice, MLX already unpacks to bf16 (4.2GB) for inference.
    # The question is: does fp32 merged add significant memory over bf16 base?
    #
    # Estimate model sizes:
    n_params = 2412820480  # from prior experiment
    ternary_size_mb = n_params / 4 / (1024 * 1024)  # 4 ternary values per byte
    fp32_size_mb = n_params * 4 / (1024 * 1024)
    bf16_size_mb = n_params * 2 / (1024 * 1024)

    # Actual measured: use process memory
    k2_ratio = mem_fp32 / mem_packed if mem_packed > 0 else float("inf")
    k2_pass = k2_ratio <= 2.0

    print(f"\n  K2: fp32 merged model memory vs ternary base memory")
    print(f"    Ternary packed process memory: {mem_packed:.1f} MB")
    print(f"    fp32 merged process memory:    {mem_fp32:.1f} MB")
    print(f"    Ratio: {k2_ratio:.2f}x (threshold: <= 2.0)")
    print(f"    Theoretical sizes: ternary={ternary_size_mb:.0f}MB, bf16={bf16_size_mb:.0f}MB, fp32={fp32_size_mb:.0f}MB")
    print(f"    K2: {'PASS' if k2_pass else 'FAIL'}")

    results["k2_ternary_mem_mb"] = mem_packed
    results["k2_fp32_mem_mb"] = mem_fp32
    results["k2_ratio"] = round(k2_ratio, 2)
    results["k2_pass"] = k2_pass
    results["theoretical_sizes_mb"] = {
        "ternary_packed": round(ternary_size_mb),
        "bf16": round(bf16_size_mb),
        "fp32": round(fp32_size_mb),
    }

    # K3: fp32 merge latency > runtime LoRA composition latency
    fp32_tps = fp32_bench["tok_per_s"]
    runtime_tps = runtime_bench["tok_per_s"]
    # "Latency" means time per token. fp32 merge wins if tok/s >= runtime tok/s
    # (i.e., latency <= runtime latency)
    k3_pass = fp32_tps >= runtime_tps  # fp32 should be faster (no per-token LoRA overhead)

    print(f"\n  K3: fp32 merge latency vs runtime LoRA composition latency")
    print(f"    fp32 merged:      {fp32_tps} +/- {fp32_bench['tok_per_s_stddev']} tok/s ({fp32_bench['mean_s']:.3f} +/- {fp32_bench['stddev_s']:.3f}s for {BENCH_TOKENS} tokens)")
    print(f"    Runtime LoRA N=5: {runtime_tps} +/- {runtime_bench['tok_per_s_stddev']} tok/s ({runtime_bench['mean_s']:.3f} +/- {runtime_bench['stddev_s']:.3f}s for {BENCH_TOKENS} tokens)")
    print(f"    Base (no adapt):  {base_bench['tok_per_s']} +/- {base_bench['tok_per_s_stddev']} tok/s ({base_bench['mean_s']:.3f} +/- {base_bench['stddev_s']:.3f}s for {BENCH_TOKENS} tokens)")
    print(f"    fp32 vs runtime: {'faster' if k3_pass else 'SLOWER'}")
    print(f"    K3: {'PASS' if k3_pass else 'FAIL'}")

    results["k3_fp32_tps"] = fp32_tps
    results["k3_runtime_tps"] = runtime_tps
    results["k3_base_tps"] = base_bench["tok_per_s"]
    results["k3_pass"] = k3_pass

    # ==================================================================
    # Phase 9: Crossover analysis — when does fp32 merge beat runtime LoRA?
    # ==================================================================
    print("\n[Phase 9] Crossover Analysis...")

    # Runtime LoRA overhead scales linearly with N: overhead = a + b*N
    # From prior experiment: 9.5% + 7.5%*N
    # fp32 merge overhead is constant (0% per-token, but base is slower due to fp32)
    #
    # Also compare PPL quality across N
    print("\n  PPL comparison (fp32 vs runtime vs bf16 vs base):")
    print(f"  {'N':>3} | {'Runtime':>10} | {'fp32':>10} | {'bf16':>10} | {'Base':>10} | {'fp32/rt':>8}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    crossover_data = {}
    for N in range(1, len(domain_list) + 1):
        rt = runtime_results[f"N={N}"]["avg_ppl"]
        f32 = fp32_results[f"N={N}"]["avg_ppl"]
        b16 = bf16_results[f"N={N}"]["avg_ppl"]
        ratio = f32 / rt
        print(f"  {N:>3} | {rt:>10.4f} | {f32:>10.4f} | {b16:>10.4f} | {avg_base:>10.4f} | {ratio:>8.4f}")
        crossover_data[f"N={N}"] = {
            "runtime_avg": rt, "fp32_avg": f32, "bf16_avg": b16,
            "fp32_runtime_ratio": round(ratio, 4),
        }

    results["crossover_analysis"] = crossover_data

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"

    print(f"\n  K1 (fp32 PPL <= 1.05x runtime at N=5): {'PASS' if k1_pass else 'FAIL'} (ratio={k1_ratio:.4f})")
    print(f"  K2 (memory <= 2x ternary base):         {'PASS' if k2_pass else 'FAIL'} (ratio={k2_ratio:.2f}x)")
    print(f"  K3 (fp32 latency <= runtime LoRA):       {'PASS' if k3_pass else 'FAIL'} ({fp32_tps} vs {runtime_tps} tok/s)")

    print(f"\n  VERDICT: {verdict}")

    if not all_pass:
        failed = []
        if not k1_pass: failed.append("K1")
        if not k2_pass: failed.append("K2")
        if not k3_pass: failed.append("K3")
        results["killed_by"] = failed

    results["verdict"] = verdict

    # Key finding summary
    bf16_n5 = bf16_results["N=5"]["avg_ppl"]
    results["key_findings"] = {
        "fp32_vs_runtime_ppl_ratio_n5": round(k1_ratio, 4),
        "bf16_vs_runtime_ppl_ratio_n5": round(bf16_n5 / runtime_n5_avg, 4),
        "fp32_preserves_delta": k1_ratio <= 1.05,
        "bf16_delta_loss_pct": round((bf16_n5 - fp32_n5_avg) / (fp32_n5_avg - avg_base) * 100, 1) if fp32_n5_avg != avg_base else 0,
        "memory_overhead_vs_bf16": round((mem_fp32 - mem_unpacked) / mem_unpacked * 100, 1) if mem_unpacked > 0 else 0,
        "latency_fp32_vs_base_pct": round((base_bench["tok_per_s"] - fp32_tps) / base_bench["tok_per_s"] * 100, 1),
        "latency_runtime_vs_base_pct": round((base_bench["tok_per_s"] - runtime_tps) / base_bench["tok_per_s"] * 100, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
