#!/usr/bin/env python3
"""
BitNet Serving Path: LoTA-QAF Lossless Merge + Apple Silicon Benchmarks

Tests whether trained ternary LoRA adapters can be losslessly merged into
BitNet-2B-4T base weights via the LoTA-QAF principle, enabling pre-merged
serving without runtime adapter composition.

LoTA-QAF principle (arxiv 2505.18724):
  Given ternary base W_int in {-1, 0, 1} with scale alpha:
    W_float = W_int * alpha
  And LoRA delta: DeltaW = (B @ A) * lora_scale / N
  Merge: W'_float = W_float + DeltaW
  Requantize: W'_int = round(W'_float / alpha'), alpha' = mean(|W'_float|)

  If DeltaW is small relative to alpha, the merge is nearly lossless because
  W'_int stays close to the integer grid.

Kill criteria:
  K1: lossless ternary merge loses >5% of adapter PPL benefit at N>1
  K2: p95 latency >10s for 100-token generation on Apple Silicon
  K3: memory exceeds 8GB with 15 pre-merged adapters (simulated by repeated merge)

Reuses trained ternary adapters from bitnet_sole_vs_monolithic experiment.
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

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 128
VAL_BATCHES = 50

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse trained adapters and data from sole_vs_monolithic
ADAPTER_SOURCE = Path(__file__).parent.parent / "bitnet_sole_vs_monolithic" / "adapters" / "sole"
DATA_SOURCE = Path(__file__).parent.parent / "bitnet_sole_vs_monolithic" / "data"

DOMAINS = ["medical", "code", "math", "legal", "creative"]

# ===========================================================================
# Ternary weight handling
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


def unpack_ternary_raw(packed_weights, out_features):
    """Unpack to raw integer ternary {-1, 0, 1} without scaling."""
    w0 = (packed_weights & 3).astype(mx.int8) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.int8) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.int8) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.int8) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
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
# Adapter loading and LoRA delta computation
# ===========================================================================
def load_adapter(domain: str) -> dict:
    """Load adapter weights from disk."""
    path = ADAPTER_SOURCE / domain / "adapter.npz"
    return dict(mx.load(str(path)))


def ste_ternary(W):
    """Ternary quantization: round to {-1, 0, 1} * alpha."""
    alpha = mx.mean(mx.abs(W)) + 1e-10
    W_scaled = W / alpha
    W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
    return W_q


def compute_lora_deltas(adapter_params: dict, scale: float = 1.0, apply_ste: bool = False):
    """Compute per-layer DeltaW = B @ A * lora_scale * scale.

    Groups lora_a and lora_b by their parent layer path, then computes
    the full-rank delta matrix.

    If apply_ste=True, quantizes A and B to ternary before computing delta
    (matching the STE forward pass used during training).

    Returns dict: layer_path -> DeltaW matrix
    """
    # Group by layer path
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

            if apply_ste:
                A = ste_ternary(A)
                B = ste_ternary(B)

            # LoRA output = x @ A @ B * lora_scale
            # Standard linear: y = x @ W.T
            # Merged: W' = W + (A @ B).T * lora_scale = W + B.T @ A.T * lora_scale
            delta = (B.T @ A.T).astype(mx.bfloat16) * LORA_SCALE * scale
            deltas[layer_path] = delta

    return deltas


def merge_deltas_into_model(model, all_deltas: dict):
    """Merge computed weight deltas directly into model weights (float merge).

    all_deltas: dict of layer_path -> DeltaW (out_features, in_features)
    """
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Construct the full path
                layer_idx = None
                for i, l in enumerate(model.model.layers):
                    if l is layer:
                        layer_idx = i
                        break

                full_path = f"model.layers.{layer_idx}.{key}"
                if full_path in all_deltas:
                    delta = all_deltas[full_path]
                    module.weight = module.weight + delta

    mx.eval(model.parameters())


def lota_qaf_merge(model, all_deltas: dict):
    """LoTA-QAF merge: add delta then requantize to ternary grid.

    For each linear layer with a delta:
      W_merged = W + DeltaW
      alpha' = mean(|W_merged|)
      W_int' = clip(round(W_merged / alpha'), -1, 1)
      W_final = W_int' * alpha'

    This keeps the weights on the ternary grid {-alpha', 0, alpha'}.
    """
    merge_stats = []

    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                layer_idx = None
                for i, l in enumerate(model.model.layers):
                    if l is layer:
                        layer_idx = i
                        break

                full_path = f"model.layers.{layer_idx}.{key}"
                if full_path in all_deltas:
                    delta = all_deltas[full_path]
                    W_orig = module.weight
                    W_merged = W_orig + delta

                    # Requantize to ternary
                    alpha_new = mx.mean(mx.abs(W_merged)) + 1e-10
                    W_int_new = mx.clip(mx.round(W_merged / alpha_new), -1.0, 1.0)
                    W_final = W_int_new * alpha_new

                    # Track how much was lost
                    quant_error = mx.mean(mx.abs(W_merged - W_final)).item()
                    delta_norm = mx.mean(mx.abs(delta)).item()
                    orig_norm = mx.mean(mx.abs(W_orig)).item()

                    # Count how many weights changed ternary state
                    alpha_orig = mx.mean(mx.abs(W_orig)) + 1e-10
                    W_int_orig = mx.clip(mx.round(W_orig / alpha_orig), -1.0, 1.0)
                    changed = mx.sum(W_int_new != W_int_orig).item()
                    total = W_int_orig.size

                    merge_stats.append({
                        "layer": full_path,
                        "quant_error": round(quant_error, 6),
                        "delta_norm": round(delta_norm, 6),
                        "orig_norm": round(orig_norm, 6),
                        "delta_to_orig_ratio": round(delta_norm / (orig_norm + 1e-10), 6),
                        "ternary_changed_pct": round(100.0 * changed / total, 2),
                    })

                    module.weight = W_final

    mx.eval(model.parameters())
    return merge_stats


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
def benchmark_generation(model, tokenizer, n_tokens=100, n_runs=5, prompt="The future of AI is"):
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

    latencies.sort()
    return {
        "n_tokens": n_tokens,
        "n_runs": n_runs,
        "mean_s": round(sum(latencies) / len(latencies), 3),
        "min_s": round(latencies[0], 3),
        "max_s": round(latencies[-1], 3),
        "p95_s": round(latencies[int(0.95 * len(latencies))], 3),
        "tok_per_s": round(n_tokens / (sum(latencies) / len(latencies)), 1),
    }


def get_memory_usage_mb():
    """Get current process memory in MB."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return round(rusage.ru_maxrss / (1024 * 1024), 1)  # macOS reports in bytes


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_serving_path",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "adapter_source": str(ADAPTER_SOURCE),
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet Serving Path: LoTA-QAF Merge + Apple Silicon Benchmarks")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Load model
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    mem_packed = get_memory_usage_mb()
    print(f"  Memory (packed): {mem_packed:.1f} MB")

    # Unpack for inference
    print("  Unpacking ternary weights...")
    model, n_replaced = replace_bitlinear_with_linear(model)
    print(f"  Replaced {n_replaced} BitLinear layers")

    mem_unpacked = get_memory_usage_mb()
    print(f"  Memory (unpacked bfloat16): {mem_unpacked:.1f} MB")
    results["memory_packed_mb"] = mem_packed
    results["memory_unpacked_mb"] = mem_unpacked

    # ==================================================================
    # Phase 1: Base model PPL (reference)
    # ==================================================================
    print("\n[Phase 1] Base model PPL (no adapters)...")
    # Save a copy of base weights for reset
    base_weights = {}
    for i, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                path = f"model.layers.{i}.{key}"
                base_weights[path] = mx.array(module.weight)
    mx.eval(base_weights)

    base_ppls = compute_all_ppls(model, tokenizer)
    for d, p in base_ppls.items():
        print(f"  {d}: PPL = {p:.2f}")
    results["base_ppls"] = {d: round(p, 4) for d, p in base_ppls.items()}

    # ==================================================================
    # Phase 2: Reference PPL from bitnet_sole_vs_monolithic
    # ==================================================================
    print("\n[Phase 2] Loading reference PPLs from sole_vs_monolithic...")
    ref_results_path = ADAPTER_SOURCE.parent.parent / "results.json"
    ref_ppls = {}
    if ref_results_path.exists():
        with open(ref_results_path) as f:
            ref = json.load(f)
        ref_ppls = ref.get("sole_routed_ppls", {})
        ref_composed = ref.get("sole_composed_ppls", {})
        print(f"  Routed PPLs: {ref_ppls}")
        print(f"  Composed PPLs: {ref_composed}")
    results["reference_routed_ppls"] = ref_ppls
    results["reference_composed_ppls"] = ref_composed if ref_results_path.exists() else {}

    # ==================================================================
    # Phase 3: Load all adapters
    # ==================================================================
    print("\n[Phase 3] Loading trained adapters...")
    all_adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTER_SOURCE / domain / "adapter.npz"
        if not adapter_path.exists():
            print(f"  WARNING: {domain} adapter not found at {adapter_path}")
            continue
        all_adapters[domain] = load_adapter(domain)
        n_params = sum(v.size for v in all_adapters[domain].values())
        print(f"  {domain}: {len(all_adapters[domain])} tensors, {n_params:,} params")

    results["n_adapters_loaded"] = len(all_adapters)

    # ==================================================================
    # Phase 4: Single-adapter merge comparison (N=1)
    # ==================================================================
    print("\n[Phase 4] Single-adapter merge comparison (N=1)...")
    single_merge_results = {}

    for domain in DOMAINS:
        if domain not in all_adapters:
            continue

        print(f"\n  --- {domain} ---")
        adapter = all_adapters[domain]
        deltas = compute_lora_deltas(adapter, scale=1.0)

        # (a) Float merge: W' = W + DeltaW (no requantization)
        # Reset to base
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    idx = None
                    for i, l in enumerate(model.model.layers):
                        if l is layer:
                            idx = i
                            break
                    path = f"model.layers.{idx}.{key}"
                    if path in base_weights:
                        module.weight = mx.array(base_weights[path])
        mx.eval(model.parameters())

        merge_deltas_into_model(model, deltas)
        ppl_float = compute_ppl(model, tokenizer, domain)
        print(f"    Float merge PPL: {ppl_float:.4f}")

        # (b) LoTA-QAF merge: W' = requantize(W + DeltaW)
        # Reset to base again
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    idx = None
                    for i, l in enumerate(model.model.layers):
                        if l is layer:
                            idx = i
                            break
                    path = f"model.layers.{idx}.{key}"
                    if path in base_weights:
                        module.weight = mx.array(base_weights[path])
        mx.eval(model.parameters())

        merge_stats = lota_qaf_merge(model, deltas)
        ppl_qaf = compute_ppl(model, tokenizer, domain)

        # Summarize merge stats
        avg_delta_ratio = sum(s["delta_to_orig_ratio"] for s in merge_stats) / len(merge_stats)
        avg_quant_err = sum(s["quant_error"] for s in merge_stats) / len(merge_stats)
        avg_changed = sum(s["ternary_changed_pct"] for s in merge_stats) / len(merge_stats)

        print(f"    LoTA-QAF merge PPL: {ppl_qaf:.4f}")
        print(f"    Avg delta/orig ratio: {avg_delta_ratio:.6f}")
        print(f"    Avg quant error: {avg_quant_err:.6f}")
        print(f"    Avg ternary changed: {avg_changed:.2f}%")

        base_p = base_ppls[domain]
        # PPL benefit = base - individual; check how much merge preserves
        float_benefit = (base_p - ppl_float) / (base_p + 1e-10) * 100
        qaf_benefit = (base_p - ppl_qaf) / (base_p + 1e-10) * 100
        ref_routed = ref_ppls.get(domain, base_p)
        ref_benefit = (base_p - ref_routed) / (base_p + 1e-10) * 100

        print(f"    Base PPL: {base_p:.4f}")
        print(f"    Reference routed PPL: {ref_routed:.4f} ({ref_benefit:+.1f}%)")
        print(f"    Float merge benefit: {float_benefit:+.1f}%")
        print(f"    QAF merge benefit: {qaf_benefit:+.1f}%")

        single_merge_results[domain] = {
            "base_ppl": round(base_p, 4),
            "ref_routed_ppl": round(ref_routed, 4),
            "float_merge_ppl": round(ppl_float, 4),
            "qaf_merge_ppl": round(ppl_qaf, 4),
            "ref_benefit_pct": round(ref_benefit, 2),
            "float_benefit_pct": round(float_benefit, 2),
            "qaf_benefit_pct": round(qaf_benefit, 2),
            "qaf_vs_float_loss_pct": round(float_benefit - qaf_benefit, 2) if float_benefit != 0 else 0,
            "avg_delta_ratio": round(avg_delta_ratio, 6),
            "avg_ternary_changed_pct": round(avg_changed, 2),
        }

    results["single_merge_n1"] = single_merge_results

    # ==================================================================
    # Phase 5: Multi-adapter merge scaling (N=1..5)
    # ==================================================================
    print("\n[Phase 5] Multi-adapter merge scaling (N=1..5)...")
    domain_list = [d for d in DOMAINS if d in all_adapters]
    multi_merge_results = {}

    for N in range(1, len(domain_list) + 1):
        print(f"\n  --- N={N} adapters ---")
        selected = domain_list[:N]
        print(f"    Domains: {selected}")

        # Compute combined delta with 1/N scaling
        combined_deltas = {}
        for domain in selected:
            adapter = all_adapters[domain]
            deltas = compute_lora_deltas(adapter, scale=1.0 / N)
            for path, delta in deltas.items():
                if path in combined_deltas:
                    combined_deltas[path] = combined_deltas[path] + delta
                else:
                    combined_deltas[path] = delta

        # (a) Float merge
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    idx = None
                    for i, l in enumerate(model.model.layers):
                        if l is layer:
                            idx = i
                            break
                    path = f"model.layers.{idx}.{key}"
                    if path in base_weights:
                        module.weight = mx.array(base_weights[path])
        mx.eval(model.parameters())

        merge_deltas_into_model(model, combined_deltas)
        float_ppls = compute_all_ppls(model, tokenizer)

        # (b) QAF merge
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    idx = None
                    for i, l in enumerate(model.model.layers):
                        if l is layer:
                            idx = i
                            break
                    path = f"model.layers.{idx}.{key}"
                    if path in base_weights:
                        module.weight = mx.array(base_weights[path])
        mx.eval(model.parameters())

        merge_stats = lota_qaf_merge(model, combined_deltas)
        qaf_ppls = compute_all_ppls(model, tokenizer)

        avg_delta_ratio = sum(s["delta_to_orig_ratio"] for s in merge_stats) / len(merge_stats)
        avg_changed = sum(s["ternary_changed_pct"] for s in merge_stats) / len(merge_stats)

        print(f"    Float PPLs: {', '.join(f'{d}={float_ppls[d]:.2f}' for d in DOMAINS)}")
        print(f"    QAF PPLs:   {', '.join(f'{d}={qaf_ppls[d]:.2f}' for d in DOMAINS)}")
        print(f"    Avg delta/orig: {avg_delta_ratio:.6f}, ternary changed: {avg_changed:.2f}%")

        # Compare with reference composed PPLs at this N
        multi_merge_results[f"N={N}"] = {
            "domains": selected,
            "float_ppls": {d: round(p, 4) for d, p in float_ppls.items()},
            "qaf_ppls": {d: round(p, 4) for d, p in qaf_ppls.items()},
            "avg_float_ppl": round(sum(float_ppls.values()) / len(float_ppls), 4),
            "avg_qaf_ppl": round(sum(qaf_ppls.values()) / len(qaf_ppls), 4),
            "avg_delta_ratio": round(avg_delta_ratio, 6),
            "avg_ternary_changed_pct": round(avg_changed, 2),
        }

    results["multi_merge_scaling"] = multi_merge_results

    # ==================================================================
    # Phase 6: K1 assessment -- QAF merge vs float merge PPL loss
    # ==================================================================
    print("\n[Phase 6] K1 Assessment: QAF merge fidelity...")

    # At N=5 (the hardest case), compare QAF vs float merge
    n5 = multi_merge_results.get("N=5", {})
    if n5:
        float_avg = n5["avg_float_ppl"]
        qaf_avg = n5["avg_qaf_ppl"]
        # K1: does QAF lose >5% of the adapter PPL benefit?
        # Benefit = base_avg - float_avg
        base_avg = sum(base_ppls.values()) / len(base_ppls)
        float_benefit = base_avg - float_avg
        qaf_benefit = base_avg - qaf_avg

        if float_benefit > 0:
            benefit_loss_pct = (float_benefit - qaf_benefit) / float_benefit * 100
        else:
            benefit_loss_pct = 0  # No benefit to lose

        k1_pass = benefit_loss_pct <= 5.0
        print(f"  Base avg PPL: {base_avg:.4f}")
        print(f"  Float merge avg PPL (N=5): {float_avg:.4f} (benefit: {float_benefit:.4f})")
        print(f"  QAF merge avg PPL (N=5):   {qaf_avg:.4f} (benefit: {qaf_benefit:.4f})")
        print(f"  Benefit loss: {benefit_loss_pct:.1f}%")
        print(f"  K1: {'PASS' if k1_pass else 'FAIL'} (threshold: 5%)")

        results["k1_base_avg_ppl"] = round(base_avg, 4)
        results["k1_float_benefit"] = round(float_benefit, 4)
        results["k1_qaf_benefit"] = round(qaf_benefit, 4)
        results["k1_benefit_loss_pct"] = round(benefit_loss_pct, 2)
        results["k1_pass"] = k1_pass
    else:
        results["k1_pass"] = False
        results["k1_error"] = "N=5 merge not computed"

    # ==================================================================
    # Phase 7: Latency benchmark (K2)
    # ==================================================================
    print("\n[Phase 7] Latency benchmark (100-token generation)...")

    # Reset to base for clean benchmark
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                idx = None
                for i, l in enumerate(model.model.layers):
                    if l is layer:
                        idx = i
                        break
                path = f"model.layers.{idx}.{key}"
                if path in base_weights:
                    module.weight = mx.array(base_weights[path])
    mx.eval(model.parameters())

    # Benchmark base model
    print("  Benchmarking base model...")
    base_bench = benchmark_generation(model, tokenizer, n_tokens=100, n_runs=5)
    print(f"    Base: {base_bench['tok_per_s']} tok/s, p95={base_bench['p95_s']}s")
    results["latency_base"] = base_bench

    # Benchmark with N=5 QAF merged
    print("  Benchmarking N=5 QAF merged model...")
    combined_deltas = {}
    N = len(domain_list)
    for domain in domain_list:
        adapter = all_adapters[domain]
        deltas = compute_lora_deltas(adapter, scale=1.0 / N)
        for path, delta in deltas.items():
            if path in combined_deltas:
                combined_deltas[path] = combined_deltas[path] + delta
            else:
                combined_deltas[path] = delta

    lota_qaf_merge(model, combined_deltas)
    merged_bench = benchmark_generation(model, tokenizer, n_tokens=100, n_runs=5)
    print(f"    Merged: {merged_bench['tok_per_s']} tok/s, p95={merged_bench['p95_s']}s")
    results["latency_merged_n5"] = merged_bench

    k2_pass = merged_bench["p95_s"] <= 10.0
    print(f"  K2: {'PASS' if k2_pass else 'FAIL'} (p95={merged_bench['p95_s']}s, threshold: 10s)")
    results["k2_pass"] = k2_pass

    # ==================================================================
    # Phase 8: Memory assessment (K3)
    # ==================================================================
    print("\n[Phase 8] Memory assessment (K3)...")

    # The key insight: pre-merged model has SAME memory as base model.
    # No adapter storage needed at all. So K3 is trivially satisfied.
    # But let's verify by simulating 15 adapters merged (using repeated merge).

    # Reset to base
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                idx = None
                for i, l in enumerate(model.model.layers):
                    if l is layer:
                        idx = i
                        break
                path = f"model.layers.{idx}.{key}"
                if path in base_weights:
                    module.weight = mx.array(base_weights[path])
    mx.eval(model.parameters())

    mem_before = get_memory_usage_mb()

    # Merge 15 "adapters" (reuse 5 adapters 3x with different scaling)
    print("  Merging 15 simulated adapters (5 real x 3 replicas)...")
    combined_deltas_15 = {}
    N_sim = 15
    for rep in range(3):
        for domain in domain_list:
            adapter = all_adapters[domain]
            deltas = compute_lora_deltas(adapter, scale=1.0 / N_sim)
            for path, delta in deltas.items():
                if path in combined_deltas_15:
                    combined_deltas_15[path] = combined_deltas_15[path] + delta
                else:
                    combined_deltas_15[path] = delta

    lota_qaf_merge(model, combined_deltas_15)

    mem_after = get_memory_usage_mb()
    print(f"  Memory before merge: {mem_before:.1f} MB")
    print(f"  Memory after N=15 merge: {mem_after:.1f} MB")
    print(f"  Delta: {mem_after - mem_before:.1f} MB")

    # Also compute PPL to see degradation
    ppls_n15 = compute_all_ppls(model, tokenizer)
    avg_n15 = sum(ppls_n15.values()) / len(ppls_n15)
    print(f"  N=15 QAF PPLs: {', '.join(f'{d}={ppls_n15[d]:.2f}' for d in DOMAINS)}")
    print(f"  Avg N=15 PPL: {avg_n15:.2f}")

    # K3: model memory after merge < 8GB
    # On macOS, we measure process RSS which includes Python overhead
    # The model itself is ~4GB unpacked, and merge doesn't add memory
    model_mem_estimate_gb = mem_after / 1024.0
    k3_pass = model_mem_estimate_gb < 8.0
    print(f"  K3: {'PASS' if k3_pass else 'FAIL'} (memory={model_mem_estimate_gb:.2f}GB, threshold: 8GB)")

    results["memory_before_merge_mb"] = round(mem_before, 1)
    results["memory_after_n15_merge_mb"] = round(mem_after, 1)
    results["memory_delta_mb"] = round(mem_after - mem_before, 1)
    results["n15_ppls"] = {d: round(p, 4) for d, p in ppls_n15.items()}
    results["n15_avg_ppl"] = round(avg_n15, 4)
    results["k3_pass"] = k3_pass

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  K1 (QAF benefit loss <=5%): {'PASS' if results.get('k1_pass') else 'FAIL'}")
    if 'k1_benefit_loss_pct' in results:
        print(f"      Benefit loss: {results['k1_benefit_loss_pct']:.1f}%")

    print(f"  K2 (p95 latency <=10s):     {'PASS' if results.get('k2_pass') else 'FAIL'}")
    print(f"      p95: {merged_bench['p95_s']}s, throughput: {merged_bench['tok_per_s']} tok/s")

    print(f"  K3 (memory <8GB at N=15):   {'PASS' if results.get('k3_pass') else 'FAIL'}")
    print(f"      Process memory: {model_mem_estimate_gb:.2f}GB")

    all_pass = results.get("k1_pass", False) and results.get("k2_pass", False) and results.get("k3_pass", False)
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

    if not all_pass:
        failed = []
        if not results.get("k1_pass"): failed.append("K1")
        if not results.get("k2_pass"): failed.append("K2")
        if not results.get("k3_pass"): failed.append("K3")
        results["killed_by"] = failed

    print(f"\n  VERDICT: {results['verdict']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
