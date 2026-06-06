#!/usr/bin/env python3
"""Fisher-weighted adapter composition: principled scale balancing via per-parameter importance.

Kill criteria:
  K706: Fisher-weighted mixed PPL not better than partial equalization (50% log-compression) baseline
  K707: Fisher diagonal computation exceeds 10 minutes for 5 adapters on M5 Pro
  K708: Fisher weights rank-correlated >0.9 with raw Frobenius norms (Fisher adds no info)

Type: Guided Exploration (Type 2)
Papers: Fisher Merging (arXiv:2111.09832), EWC (Kirkpatrick et al. 2017)
Prior: Finding #279 (Frobenius equalization), Finding #277 (scale root cause)

Approach:
  1. Load 5 trained domain adapters (real_data_domain_experts)
  2. For each domain: compute diagonal Fisher on domain validation data
  3. Derive Fisher importance per adapter: w_i = sum(F_i * Delta_i^2)
  4. Normalize to composition weights: alpha_i = w_i / sum(w_j)
  5. Compare: raw sum, full Frobenius eq, partial eq (50%), Fisher-weighted
  6. Evaluate: per-domain PPL, mixed PPL, Gini, generation quality
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source adapters
NTP_SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = NTP_SOURCE_DIR / "adapters"
NTP_DATA_DIR = NTP_SOURCE_DIR / "data"
SKELETON_PATH = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEED = 42
N_EVAL_SAMPLES = 20  # per domain for PPL evaluation
N_FISHER_SAMPLES = 10  # per domain for Fisher estimation (10 sufficient for diagonal Fisher)
FISHER_SEQ_LENGTH = 128  # shorter sequences for Fisher to save memory

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0, "code": 20.0, "math": 20.0,
    "legal": 4.0, "finance": 1.0,
}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

NUM_LAYERS = 30


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


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


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


def gini_coefficient(values):
    """Compute Gini coefficient of an array of non-negative values."""
    vals = np.sort(np.asarray(values, dtype=np.float64))
    n = len(vals)
    if n <= 1 or np.sum(vals) < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * vals) / (n * np.sum(vals)))


# ============================================================================
# BitNet unpacking (reused from Frobenius experiment)
# ============================================================================

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
    from mlx_lm.models.bitlinear_layers import BitLinear
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


# ============================================================================
# Adapter loading and delta computation (from Frobenius experiment)
# ============================================================================

def compute_per_domain_frobenius_norms(skeleton_np, all_adapters_np):
    """Compute per-domain Frobenius norms of Delta_i = s_i * B_i^T @ A_i^T."""
    domain_norms = {}
    for di, domain in enumerate(DOMAINS):
        total_frobenius_sq = 0.0
        scale = OPTIMAL_SCALES[domain]
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                B = all_adapters_np[di][bkey].astype(np.float64)
                b_frob_sq = float(np.sum(B ** 2))
                total_frobenius_sq += scale ** 2 * b_frob_sq
        domain_norms[domain] = math.sqrt(total_frobenius_sq)
    return domain_norms


def compute_equalization_scales(domain_norms, method="partial"):
    """Compute per-domain scaling factors for Frobenius equalization."""
    norms = np.array([domain_norms[d] for d in DOMAINS])
    log_norms = np.log(norms + 1e-30)
    if method == "full":
        geo_mean = np.exp(np.mean(log_norms))
        scales = geo_mean / norms
    elif method == "partial":
        mean_log = np.mean(log_norms)
        new_log = mean_log + 0.5 * (log_norms - mean_log)
        new_norms = np.exp(new_log)
        scales = new_norms / norms
    else:
        raise ValueError(f"Unknown method: {method}")
    return {d: float(scales[i]) for i, d in enumerate(DOMAINS)}


def compose_deltas(skeleton_np, all_adapters_np, eq_scales=None):
    """Compose N=5 adapter deltas with optional per-domain scaling."""
    delta_dict = {}
    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            composed = None
            for di, domain in enumerate(DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                A = skeleton_np[skey].astype(np.float64)
                B = all_adapters_np[di][bkey].astype(np.float64)
                scale = OPTIMAL_SCALES[domain]
                delta = scale * (B.T @ A.T)
                if eq_scales is not None:
                    delta = delta * eq_scales[domain]
                if composed is None:
                    composed = delta
                else:
                    composed += delta
            if composed is not None:
                delta_dict[(li, key)] = composed
    return delta_dict


def compute_composed_gini(delta_dict, sample_layers=None, sample_keys=None):
    """Compute Gini coefficient of composed deltas."""
    if sample_layers is None:
        sample_layers = [0, 5, 10, 15, 20, 25, 29]
    if sample_keys is None:
        sample_keys = ["self_attn.q_proj", "mlp.gate_proj"]
    ginis = []
    for li in sample_layers:
        for key in sample_keys:
            if (li, key) not in delta_dict:
                continue
            composed = delta_dict[(li, key)]
            _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
            S_nz = S_c[S_c > 1e-6]
            if len(S_nz) > 1:
                ginis.append(gini_coefficient(S_nz))
    return {
        "mean_gini": float(np.mean(ginis)) if ginis else 0.0,
        "std_gini": float(np.std(ginis)) if ginis else 0.0,
        "n_samples": len(ginis),
    }


# ============================================================================
# Model weight manipulation
# ============================================================================

def apply_delta_dict_to_model(model, delta_dict_np):
    """Apply dict of {(li, key): np_delta} to model weights."""
    count = 0
    for (li, key), delta_np in delta_dict_np.items():
        delta_mx = mx.array(delta_np.astype(np.float32))
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta_mx.astype(module.weight.dtype)
            count += 1
        if count % 30 == 0:
            mx.eval(model.parameters())
    mx.eval(model.parameters())
    return count


def save_base_weights(model):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                layer_w[key] = module.weight
        base_weights.append(layer_w)
    return base_weights


def restore_base_weights(model, base_weights):
    for li, layer_weights in enumerate(base_weights):
        for key, weight in layer_weights.items():
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                module.weight = weight
    mx.eval(model.parameters())


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_val_loss(model, tokenizer, texts):
    """Compute mean cross-entropy val loss on texts."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
        if len(ids) < 2:
            continue
        x = mx.array([ids[:-1]])
        y = mx.array([ids[1:]])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += len(ids) - 1
        del logits, loss
    return total_loss / max(total_tokens, 1)


# ============================================================================
# Generation quality evaluation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=80):
    """Simple greedy generation for quality evaluation."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    ids_mx = mx.array([ids])
    for _ in range(max_tokens):
        logits = model(ids_mx)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        ids.append(token_id)
        ids_mx = mx.array([ids])
        del logits
    return tokenizer.decode(ids, skip_special_tokens=True)


# ============================================================================
# Phase 1: Diagonal Fisher Estimation
# ============================================================================

def phase_fisher_estimation(skeleton_np, all_adapters_np):
    """Compute diagonal Fisher Information for each adapter's parameter positions.

    Memory-efficient approach: instead of storing full Fisher diagonal matrices
    (which would be ~11GB for 210 weight matrices), we compute the Fisher
    importance w_i = sum(F_i[j] * Delta_i[j]^2) incrementally. For each sample,
    we compute the gradient, square it, multiply by delta_sq, and sum immediately.

    For diagnostic purposes, we also track a few aggregate statistics (mean Fisher,
    CV) computed per-(layer,key) on a SAMPLED subset of positions.

    For each domain i:
      1. Load base model + apply adapter i's delta
      2. For N_FISHER_SAMPLES from domain i, compute grad of log p(x|theta)
      3. Fisher importance w_i = sum_j E[(d loss / d theta_j)^2] * Delta_i[j]^2
    """
    log("\n" + "=" * 70)
    log("PHASE 1: Diagonal Fisher Estimation")
    log("=" * 70)
    t0 = time.time()

    # Load model
    log("  Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("model-loaded")

    base_weights = save_base_weights(model)

    # For each domain: compute deltas on the fly, apply, compute Fisher
    fisher_importances = {}
    fisher_stats = {}

    for di, domain in enumerate(DOMAINS):
        log(f"\n  Computing Fisher for {domain} (domain {di})...")
        t_domain = time.time()

        # Compute this domain's deltas (on the fly to save memory)
        scale = OPTIMAL_SCALES[domain]
        delta_single_np = {}   # (li, key) -> np delta
        delta_sq_mx = {}       # (li, key) -> mx.array of delta^2
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                A = skeleton_np[skey].astype(np.float64)
                B = all_adapters_np[di][bkey].astype(np.float64)
                delta_np = scale * (B.T @ A.T)
                delta_single_np[(li, key)] = delta_np
                delta_sq_mx[(li, key)] = mx.array((delta_np ** 2).astype(np.float32))

        # Evaluate all delta_sq arrays
        if delta_sq_mx:
            mx.eval(*list(delta_sq_mx.values()))

        apply_delta_dict_to_model(model, delta_single_np)

        # Freeze everything first, then unfreeze only target layers
        model.freeze()
        for li in range(NUM_LAYERS):
            layer = model.model.layers[li]
            for key in TARGET_KEYS:
                parts = key.split(".")
                module = layer
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None and isinstance(module, nn.Linear):
                    module.unfreeze()

        # Load domain validation data
        val_path = NTP_DATA_DIR / domain / "valid.jsonl"
        texts = []
        with open(val_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if len(texts) >= N_FISHER_SAMPLES:
                    break

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Incremental Fisher importance: accumulate sum(grad^2 * delta^2) per sample
        # Also track per-key Fisher importance and a few diagnostic stats
        per_key_fisher_accum = {}   # (li, key) -> scalar accumulator
        per_key_fisher_mean_accum = {}  # (li, key) -> running sum of mean(grad^2) for diagnostics
        n_computed = 0

        for si, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=True)[:FISHER_SEQ_LENGTH]
            if len(ids) < 2:
                continue

            x = mx.array([ids[:-1]])
            y = mx.array([ids[1:]])

            loss_val, grads = loss_and_grad(model, x, y)

            # Extract gradients and compute Fisher*delta^2 on GPU
            layer_grads = grads.get("model", {}).get("layers", [])
            sample_contributions = []

            for li in range(NUM_LAYERS):
                if li >= len(layer_grads):
                    continue
                for key in TARGET_KEYS:
                    if (li, key) not in delta_sq_mx:
                        continue

                    # Navigate to gradient
                    parts = key.split(".")
                    g = layer_grads[li]
                    found = True
                    for part in parts:
                        if isinstance(g, dict) and part in g:
                            g = g[part]
                        else:
                            found = False
                            break

                    if not found or not isinstance(g, dict) or "weight" not in g:
                        continue

                    grad_w = g["weight"]
                    dsq = delta_sq_mx[(li, key)]

                    # Fisher importance contribution: sum(grad^2 * delta^2)
                    # Computed entirely on GPU
                    fi_contrib = mx.sum(grad_w * grad_w * dsq)
                    fisher_mean = mx.mean(grad_w * grad_w)
                    sample_contributions.append((li, key, fi_contrib, fisher_mean))

            # Evaluate all contributions at once
            if sample_contributions:
                to_eval = [c for _, _, c, m in sample_contributions] + [m for _, _, c, m in sample_contributions]
                to_eval.append(loss_val)
                mx.eval(*to_eval)

                for li, key, fi_contrib, fisher_mean in sample_contributions:
                    fkey = (li, key)
                    val = fi_contrib.item()
                    mean_val = fisher_mean.item()
                    if fkey not in per_key_fisher_accum:
                        per_key_fisher_accum[fkey] = val
                        per_key_fisher_mean_accum[fkey] = mean_val
                    else:
                        per_key_fisher_accum[fkey] += val
                        per_key_fisher_mean_accum[fkey] += mean_val

            n_computed += 1
            del loss_val, grads, x, y, sample_contributions

        # Average over samples
        for fkey in per_key_fisher_accum:
            per_key_fisher_accum[fkey] /= max(n_computed, 1)
            per_key_fisher_mean_accum[fkey] /= max(n_computed, 1)

        # Total Fisher importance for this domain
        total_fisher_importance = sum(per_key_fisher_accum.values())

        # Frobenius norm squared (for comparison)
        total_frobenius_sq = 0.0
        for (li, key), delta_np in delta_single_np.items():
            total_frobenius_sq += float(np.sum(delta_np ** 2))

        fisher_importances[domain] = total_fisher_importance
        elapsed_domain = time.time() - t_domain

        # Collect per-key diagnostics
        fisher_means_all = list(per_key_fisher_mean_accum.values())
        fisher_per_key_importance = list(per_key_fisher_accum.values())

        fisher_stats[domain] = {
            "fisher_importance": total_fisher_importance,
            "frobenius_sq": total_frobenius_sq,
            "fisher_to_frob_ratio": total_fisher_importance / max(total_frobenius_sq, 1e-30),
            "n_samples_used": n_computed,
            "n_keys_with_fisher": len(per_key_fisher_accum),
            "fisher_per_key_mean": float(np.mean(fisher_means_all)) if fisher_means_all else 0.0,
            "fisher_per_key_std": float(np.std(fisher_means_all)) if fisher_means_all else 0.0,
            "fisher_per_key_cv": float(np.std(fisher_means_all) / max(np.mean(fisher_means_all), 1e-30)) if fisher_means_all else 0.0,
            "fisher_importance_cv": float(np.std(fisher_per_key_importance) / max(np.mean(fisher_per_key_importance), 1e-30)) if fisher_per_key_importance else 0.0,
            "elapsed_s": elapsed_domain,
        }

        log(f"    Fisher importance: {total_fisher_importance:.6e}")
        log(f"    Frobenius sq: {total_frobenius_sq:.4f}")
        log(f"    Fisher/Frob ratio: {total_fisher_importance / max(total_frobenius_sq, 1e-30):.6e}")
        log(f"    Per-key Fisher CV: {fisher_stats[domain]['fisher_per_key_cv']:.4f}")
        log(f"    Time: {elapsed_domain:.1f}s")

        # Restore base weights for next domain
        restore_base_weights(model, base_weights)
        del delta_single_np, delta_sq_mx
        gc.collect()
        mx.clear_cache()

    # Compute normalized Fisher weights
    total_w = sum(fisher_importances.values())
    fisher_weights = {d: fisher_importances[d] / max(total_w, 1e-30) for d in DOMAINS}

    # Compute Frobenius-based weights for comparison
    domain_norms = compute_per_domain_frobenius_norms(skeleton_np, all_adapters_np)
    frob_energy = {d: domain_norms[d] ** 2 for d in DOMAINS}
    total_frob = sum(frob_energy.values())
    frob_weights = {d: frob_energy[d] / max(total_frob, 1e-30) for d in DOMAINS}

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    fisher_vals = [fisher_weights[d] for d in DOMAINS]
    frob_vals = [frob_weights[d] for d in DOMAINS]
    rho, pval = spearmanr(fisher_vals, frob_vals)

    elapsed = time.time() - t0
    log(f"\n  Total Fisher computation time: {elapsed:.1f}s")
    log(f"  K707: Fisher computation < 10 min? {elapsed:.0f}s -> {'PASS' if elapsed < 600 else 'FAIL'}")

    log(f"\n  Per-domain weights comparison:")
    log(f"  {'Domain':10s} {'Fisher_w':>10s} {'Frob_w':>10s} {'F/Frob ratio':>14s} {'Scale':>6s}")
    log("  " + "-" * 56)
    for domain in DOMAINS:
        ratio = fisher_weights[domain] / max(frob_weights[domain], 1e-30)
        log(f"  {domain:10s} {fisher_weights[domain]:>10.6f} {frob_weights[domain]:>10.6f} "
            f"{ratio:>14.4f} {OPTIMAL_SCALES[domain]:>6.0f}")

    log(f"\n  Spearman rho(Fisher, Frobenius): {rho:.4f} (p={pval:.4f})")
    log(f"  K708: rho > 0.9? -> {'FAIL (Fisher adds no info)' if rho > 0.9 else 'PASS (Fisher decorrelates)'}")

    # Compute Fisher-based composition scales
    # alpha_i = w_i / sum(w_j), then scale each adapter by alpha_i * N
    # so that the average weight is 1, preserving total magnitude
    fisher_eq_scales = {}
    mean_weight = sum(fisher_weights.values()) / len(DOMAINS)
    for d in DOMAINS:
        fisher_eq_scales[d] = fisher_weights[d] / mean_weight
    log(f"\n  Fisher equalization scales (relative to mean=1):")
    for d in DOMAINS:
        log(f"    {d}: {fisher_eq_scales[d]:.4f}")

    cleanup(model, tokenizer, base_weights)

    return {
        "fisher_importances": {d: float(v) for d, v in fisher_importances.items()},
        "fisher_weights": {d: float(v) for d, v in fisher_weights.items()},
        "frob_weights": {d: float(v) for d, v in frob_weights.items()},
        "fisher_eq_scales": {d: float(v) for d, v in fisher_eq_scales.items()},
        "domain_norms": {d: float(v) for d, v in domain_norms.items()},
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "fisher_stats": fisher_stats,
        "k707_pass": elapsed < 600,
        "k707_value_s": elapsed,
        "k708_pass": abs(rho) <= 0.9,
        "k708_value_rho": float(rho),
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 2: PPL Comparison (raw sum, full eq, partial eq, Fisher-weighted)
# ============================================================================

def phase_ppl_comparison(skeleton_np, all_adapters_np, fisher_eq_scales, frob_eq_scales_full, frob_eq_scales_partial):
    """Compare PPL across 4 composition strategies."""
    log("\n" + "=" * 70)
    log("PHASE 2: PPL Comparison Across Composition Strategies")
    log("=" * 70)
    t0 = time.time()

    # Load model
    log("  Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("model-loaded")

    # Load validation data
    log("  Loading validation data...")
    domain_texts = {}
    for domain in DOMAINS:
        val_path = NTP_DATA_DIR / domain / "valid.jsonl"
        texts = []
        with open(val_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if len(texts) >= N_EVAL_SAMPLES:
                    break
        domain_texts[domain] = texts
    log(f"  Loaded {sum(len(v) for v in domain_texts.values())} total validation texts")

    base_weights = save_base_weights(model)

    # Base model PPL
    log("\n  Base model PPL:")
    base_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        base_ppl[domain] = math.exp(loss)
        log(f"    {domain}: {base_ppl[domain]:.4f}")

    strategies = {
        "raw_sum": None,
        "full_equalization": frob_eq_scales_full,
        "partial_equalization": frob_eq_scales_partial,
        "fisher_weighted": fisher_eq_scales,
    }

    ppl_results = {}
    gini_results = {}

    for name, eq_scales in strategies.items():
        log(f"\n  {name} composition:")
        delta_dict_np = compose_deltas(skeleton_np, all_adapters_np, eq_scales)

        # Gini
        gini_result = compute_composed_gini(delta_dict_np)
        gini_results[name] = gini_result
        log(f"    Composed Gini = {gini_result['mean_gini']:.4f} +/- {gini_result['std_gini']:.4f}")

        # PPL
        apply_delta_dict_to_model(model, delta_dict_np)

        domain_ppl = {}
        for domain in DOMAINS:
            loss = compute_val_loss(model, tokenizer, domain_texts[domain])
            domain_ppl[domain] = math.exp(loss)
            log(f"    {domain}: {domain_ppl[domain]:.4f}")

        # Mixed PPL
        all_texts = []
        for d in DOMAINS:
            all_texts.extend(domain_texts[d])
        mixed_loss = compute_val_loss(model, tokenizer, all_texts)
        mixed_ppl = math.exp(mixed_loss)
        log(f"    Mixed: {mixed_ppl:.4f}")

        ppl_results[name] = {
            "per_domain_ppl": domain_ppl,
            "mixed_ppl": mixed_ppl,
        }

        restore_base_weights(model, base_weights)
        del delta_dict_np
        gc.collect()
        mx.clear_cache()

    # K706 assessment: Fisher vs partial equalization mixed PPL
    fisher_mixed = ppl_results["fisher_weighted"]["mixed_ppl"]
    partial_mixed = ppl_results["partial_equalization"]["mixed_ppl"]
    k706_pass = fisher_mixed < partial_mixed

    log(f"\n  K706: Fisher mixed PPL ({fisher_mixed:.4f}) < Partial eq ({partial_mixed:.4f})? -> {'PASS' if k706_pass else 'FAIL'}")

    # Per-domain changes relative to raw sum
    raw_ppl = ppl_results["raw_sum"]["per_domain_ppl"]
    ppl_changes = {}
    for strat in strategies:
        changes = {}
        for domain in DOMAINS:
            change = (ppl_results[strat]["per_domain_ppl"][domain] - raw_ppl[domain]) / raw_ppl[domain]
            changes[domain] = change
        ppl_changes[strat] = changes

    log(f"\n  Per-domain PPL changes vs raw sum:")
    log(f"  {'Domain':10s} {'Raw':>8s} {'Full eq':>10s} {'Partial':>10s} {'Fisher':>10s}")
    log("  " + "-" * 52)
    for domain in DOMAINS:
        log(f"  {domain:10s} {'---':>8s} "
            f"{ppl_changes['full_equalization'][domain]:>+9.1%} "
            f"{ppl_changes['partial_equalization'][domain]:>+9.1%} "
            f"{ppl_changes['fisher_weighted'][domain]:>+9.1%}")

    # Generation quality check
    log("\n  --- Generation Quality (Fisher-weighted) ---")
    delta_dict_fisher = compose_deltas(skeleton_np, all_adapters_np, fisher_eq_scales)
    apply_delta_dict_to_model(model, delta_dict_fisher)

    gen_prompts = {
        "medical": "The patient presents with acute chest pain and elevated troponin levels. The differential diagnosis includes",
        "code": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number efficiently.\"\"\"\n",
        "math": "To solve the integral of x^2 * e^x dx, we apply integration by parts. Let u = x^2 and dv = e^x dx. Then",
        "legal": "The court held that the defendant's Fourth Amendment rights were violated because the warrantless search of",
        "finance": "The company reported Q3 earnings of $2.50 per share, exceeding analyst estimates by 15%. The revenue growth was driven by",
    }

    gen_results = {}
    for domain, prompt in gen_prompts.items():
        output = generate_text(model, tokenizer, prompt, max_tokens=80)
        gen_results[domain] = {"prompt": prompt, "output": output}
        log(f"\n    [{domain}] {output[:150]}...")

    restore_base_weights(model, base_weights)

    elapsed = time.time() - t0
    log(f"\n  Phase 2 time: {elapsed:.1f}s")
    log_memory("phase2")

    cleanup(model, tokenizer, base_weights)

    return {
        "base_ppl": base_ppl,
        "strategies": ppl_results,
        "gini_results": gini_results,
        "ppl_changes_vs_raw": ppl_changes,
        "generation_fisher": gen_results,
        "k706_pass": k706_pass,
        "k706_fisher_mixed": fisher_mixed,
        "k706_partial_mixed": partial_mixed,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 3: Fisher Weight Analysis (deeper diagnostics)
# ============================================================================

def phase_fisher_analysis(fisher_result):
    """Analyze Fisher weight properties: per-layer, per-key distributions."""
    log("\n" + "=" * 70)
    log("PHASE 3: Fisher Weight Diagnostic Analysis")
    log("=" * 70)

    # Fisher-to-Frobenius ratio per domain (P5 prediction)
    log("\n  Fisher-to-Frobenius ratio per domain (P5):")
    log(f"  {'Domain':10s} {'F/Frob ratio':>14s} {'Scale':>6s}")
    log("  " + "-" * 36)
    for domain in DOMAINS:
        stats = fisher_result["fisher_stats"][domain]
        log(f"  {domain:10s} {stats['fisher_to_frob_ratio']:>14.8f} {OPTIMAL_SCALES[domain]:>6.0f}")

    # Fisher CV within adapter (tests Assumption 1: non-trivial variance)
    log(f"\n  Fisher per-key importance CV per domain:")
    for domain in DOMAINS:
        stats = fisher_result["fisher_stats"][domain]
        log(f"    {domain}: per-key-CV={stats['fisher_per_key_cv']:.4f}, "
            f"importance-CV={stats['fisher_importance_cv']:.4f}")

    # Check if Fisher weights are just rescaled Frobenius
    fisher_w = fisher_result["fisher_weights"]
    frob_w = fisher_result["frob_weights"]

    # Compute weight ratios
    ratios = {}
    for d in DOMAINS:
        ratios[d] = fisher_w[d] / max(frob_w[d], 1e-30)
    ratio_vals = list(ratios.values())
    ratio_cv = np.std(ratio_vals) / max(np.mean(ratio_vals), 1e-30)

    log(f"\n  Fisher/Frobenius weight ratios per domain:")
    for d in DOMAINS:
        log(f"    {d}: {ratios[d]:.4f}")
    log(f"  Ratio CV: {ratio_cv:.4f} (0 = identical ranking, high = different)")

    return {
        "fisher_frob_ratios": {d: float(v) for d, v in ratios.items()},
        "ratio_cv": float(ratio_cv),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    results = {"experiment": "fisher_weighted_composition", "phases": {}}

    log("=" * 70)
    log("Fisher-Weighted Adapter Composition")
    log("Principled Scale Balancing via Per-Parameter Importance")
    log("=" * 70)

    # Load shared resources
    log("\nLoading skeleton and adapters...")
    skeleton_np = dict(np.load(str(SKELETON_PATH)))
    log(f"  Skeleton keys: {len(skeleton_np)}")

    all_adapters_np = []
    for di, domain in enumerate(DOMAINS):
        adapter_path = NTP_ADAPTERS_DIR / domain / "adapter.npz"
        adapter = dict(np.load(str(adapter_path)))
        all_adapters_np.append(adapter)
        log(f"  {domain}: {len(adapter)} keys")

    # Phase 1: Fisher estimation
    phase1 = phase_fisher_estimation(skeleton_np, all_adapters_np)
    results["phases"]["fisher_estimation"] = phase1

    # Compute Frobenius equalization scales for comparison
    domain_norms = phase1["domain_norms"]
    frob_eq_full = compute_equalization_scales(domain_norms, "full")
    frob_eq_partial = compute_equalization_scales(domain_norms, "partial")

    gc.collect()
    mx.clear_cache()

    # Phase 2: PPL comparison
    phase2 = phase_ppl_comparison(
        skeleton_np, all_adapters_np,
        phase1["fisher_eq_scales"],
        frob_eq_full, frob_eq_partial,
    )
    results["phases"]["ppl_comparison"] = phase2

    # Phase 3: Fisher analysis
    phase3 = phase_fisher_analysis(phase1)
    results["phases"]["fisher_analysis"] = phase3

    # Summary
    elapsed = time.time() - t_start
    k706 = phase2["k706_pass"]
    k707 = phase1["k707_pass"]
    k708 = phase1["k708_pass"]

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  K706 (Fisher PPL < partial eq): {'PASS' if k706 else 'FAIL'} "
        f"(Fisher={phase2['k706_fisher_mixed']:.4f} vs Partial={phase2['k706_partial_mixed']:.4f})")
    log(f"  K707 (Fisher compute < 10 min): {'PASS' if k707 else 'FAIL'} ({phase1['k707_value_s']:.0f}s)")
    log(f"  K708 (rho <= 0.9):              {'PASS' if k708 else 'FAIL'} (rho={phase1['k708_value_rho']:.4f})")

    log(f"\n  Fisher weights: {', '.join(f'{d}={phase1['fisher_weights'][d]:.4f}' for d in DOMAINS)}")
    log(f"  Frob weights:   {', '.join(f'{d}={phase1['frob_weights'][d]:.4f}' for d in DOMAINS)}")
    log(f"  Fisher/Frob ratio CV: {phase3['ratio_cv']:.4f}")

    log(f"\n  Mixed PPL comparison:")
    for strat in ["raw_sum", "full_equalization", "partial_equalization", "fisher_weighted"]:
        log(f"    {strat}: {phase2['strategies'][strat]['mixed_ppl']:.4f}")

    log(f"\n  Gini comparison:")
    for strat in ["raw_sum", "full_equalization", "partial_equalization", "fisher_weighted"]:
        log(f"    {strat}: {phase2['gini_results'][strat]['mean_gini']:.4f}")

    log(f"\n  Total time: {elapsed:.0f}s")

    results["summary"] = {
        "k706_pass": k706,
        "k706_fisher_mixed": phase2["k706_fisher_mixed"],
        "k706_partial_mixed": phase2["k706_partial_mixed"],
        "k707_pass": k707,
        "k707_value_s": phase1["k707_value_s"],
        "k708_pass": k708,
        "k708_value_rho": phase1["k708_value_rho"],
        "fisher_weights": phase1["fisher_weights"],
        "frob_weights": phase1["frob_weights"],
        "fisher_eq_scales": phase1["fisher_eq_scales"],
        "fisher_frob_ratio_cv": phase3["ratio_cv"],
        "mixed_ppls": {
            strat: phase2["strategies"][strat]["mixed_ppl"]
            for strat in ["raw_sum", "full_equalization", "partial_equalization", "fisher_weighted"]
        },
        "gini_values": {
            strat: phase2["gini_results"][strat]["mean_gini"]
            for strat in ["raw_sum", "full_equalization", "partial_equalization", "fisher_weighted"]
        },
        "elapsed_seconds": elapsed,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
