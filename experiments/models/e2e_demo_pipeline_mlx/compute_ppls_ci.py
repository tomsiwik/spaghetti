#!/usr/bin/env python3
"""Compute per-sample PPLs with confidence intervals for the E2E pipeline.

This is a lightweight post-hoc analysis script that:
1. Loads BitNet-2B-4T + adapters
2. For each domain: computes PPL on 25 samples for base AND composed
3. Reports mean +/- SE, 95% CI
4. Saves results to results_ppls_ci.json

Addresses review revision R3: PPL confidence intervals (N=25).
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from scipy import stats

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_SAMPLES = 25

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def compute_ppl_on_text(model, tokenizer, text):
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")
    tokens = tokens[:MAX_SEQ_LENGTH + 1]
    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="sum")
    mx.eval(loss)
    n_tokens = y.size
    ppl = math.exp(min(loss.item() / n_tokens, 100))
    del logits, loss, x, y
    return ppl


def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_all_adapters():
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        adapters[domain] = dict(mx.load(str(adapter_path)))
    return adapters


def premerge_adapters_into_model(model, skeleton, adapters, domain_weights):
    n_layers = len(model.model.layers)
    merge_count = 0
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
            delta = None
            for domain, w in domain_weights.items():
                if w < 1e-6:
                    continue
                di = DOMAINS.index(domain)
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key not in adapters[domain]:
                    continue
                b_mx = adapters[domain][b_key]
                lora_delta = w * LORA_SCALE * (b_mx.T @ a_mx.T)
                if delta is None:
                    delta = lora_delta
                else:
                    delta = delta + lora_delta
            if delta is not None:
                module.weight = module.weight + delta
                merge_count += 1
    mx.eval(model.parameters())
    return model


def save_base_weights(model):
    base_weights = {}
    for li, layer in enumerate(model.model.layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                base_weights[(li, key)] = mx.array(module.weight)
    mx.eval(base_weights)
    return base_weights


def restore_base_weights(model, base_weights):
    for (li, key), w in base_weights.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = w
    mx.eval(model.parameters())


def oracle_route_top1(domain):
    weights = {d: 0.0 for d in DOMAINS}
    weights[domain] = 1.0
    return weights


def compute_ci(ppls):
    """Compute mean, SE, and 95% CI for a list of PPL values."""
    arr = np.array(ppls)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    return {
        "mean": round(mean, 4),
        "se": round(se, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "n": n,
        "per_sample": [round(float(p), 4) for p in ppls],
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("PPL Confidence Interval Computation")
    log("=" * 70)

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load")

    # Save base weights for restoration
    base_weights = save_base_weights(model)

    # Load adapters
    skeleton = load_skeleton()
    all_adapters = load_all_adapters()

    results = {}

    for domain in DOMAINS:
        log(f"\n--- Domain: {domain} ---")
        val_path = DATA_DIR / domain / "valid.jsonl"

        # Load validation texts
        val_texts = []
        with open(val_path) as f:
            for j, line in enumerate(f):
                if j >= N_SAMPLES:
                    break
                val_texts.append(json.loads(line)["text"])

        # Base PPL (per-sample)
        log(f"  Computing base PPL on {len(val_texts)} samples...")
        base_ppls = []
        for text in val_texts:
            ppl = compute_ppl_on_text(model, tokenizer, text)
            base_ppls.append(ppl)

        base_stats = compute_ci(base_ppls)
        log(f"  Base PPL: {base_stats['mean']:.4f} +/- {base_stats['se']:.4f} "
            f"(95% CI: [{base_stats['ci_95_low']:.4f}, {base_stats['ci_95_high']:.4f}])")

        # Composed PPL (top-1, per-sample)
        domain_weights = oracle_route_top1(domain)
        premerge_adapters_into_model(model, skeleton, all_adapters, domain_weights)

        log(f"  Computing composed PPL on {len(val_texts)} samples...")
        composed_ppls = []
        for text in val_texts:
            ppl = compute_ppl_on_text(model, tokenizer, text)
            composed_ppls.append(ppl)

        composed_stats = compute_ci(composed_ppls)
        log(f"  Composed PPL: {composed_stats['mean']:.4f} +/- {composed_stats['se']:.4f} "
            f"(95% CI: [{composed_stats['ci_95_low']:.4f}, {composed_stats['ci_95_high']:.4f}])")

        # Check CI overlap
        ci_overlaps = composed_stats["ci_95_high"] >= base_stats["ci_95_low"]
        improvement_pct = round((base_stats["mean"] - composed_stats["mean"]) / base_stats["mean"] * 100, 2)

        results[domain] = {
            "base": base_stats,
            "composed_top1": composed_stats,
            "improvement_pct": improvement_pct,
            "composed_ci_overlaps_base_mean": bool(composed_stats["ci_95_high"] >= base_stats["mean"]),
            "cis_overlap": bool(ci_overlaps),
        }

        log(f"  Improvement: {improvement_pct:+.1f}%")
        log(f"  Composed 95% CI overlaps base mean: {composed_stats['ci_95_high'] >= base_stats['mean']}")

        # Restore for next domain
        restore_base_weights(model, base_weights)

    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log_memory("final")

    # Summary
    log(f"\n{'=' * 70}")
    log("SUMMARY: PPL with 95% Confidence Intervals")
    log("=" * 70)
    any_overlap = False
    for domain in DOMAINS:
        r = results[domain]
        b = r["base"]
        c = r["composed_top1"]
        overlap_flag = " [CI OVERLAPS BASE MEAN]" if r["composed_ci_overlaps_base_mean"] else ""
        log(f"  {domain:10s}: base={b['mean']:.2f}+/-{b['se']:.2f}  "
            f"composed={c['mean']:.2f}+/-{c['se']:.2f}  "
            f"improvement={r['improvement_pct']:+.1f}%{overlap_flag}")
        if r["composed_ci_overlaps_base_mean"]:
            any_overlap = True

    if not any_overlap:
        log("\nAll domains: composed 95% CI is entirely below base mean PPL.")
        log("Quality improvement is statistically significant for all domains.")
    else:
        log("\nWARNING: Some domains have composed CI overlapping base mean.")

    output = {
        "experiment": "e2e_demo_pipeline_mlx_ppls_ci",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": N_SAMPLES,
        "runtime_s": round(elapsed, 1),
        "domains": results,
        "all_significant": bool(not any_overlap),
    }
    out_path = EXPERIMENT_DIR / "results_ppls_ci.json"
    out_path.write_text(json.dumps(output, indent=2))
    log(f"\nResults saved to {out_path}")

    cleanup(model, tokenizer, base_weights, skeleton, all_adapters)


if __name__ == "__main__":
    main()
