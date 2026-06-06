#!/usr/bin/env python3
"""Frobenius-norm equalized composition for cross-domain scale balancing.

Kill criteria:
  K703: Composed Gini coefficient drops >40% from ~0.49 to <0.30
  K704: At least 3/5 domains PPL within 5% of raw-sum baseline
  K705: Generation quality preserved: coherent, domain-relevant text on >=2 domains

Type: Guided Exploration (Type 2)
Papers: FroM (arXiv:2506.02478), DO-Merging (arXiv:2505.15875)
Prior: Finding #277 (scale imbalance = root cause), Finding #278 (spectral surgery killed)

Approach:
  1. Load 5 trained domain adapters (real_data_domain_experts)
  2. Measure per-domain Frobenius norms and energy fractions
  3. Test 4 composition strategies:
     (a) Raw sum (baseline -- per-domain optimal scales, no normalization)
     (b) Full Frobenius equalization (all domains equal energy)
     (c) Partial equalization (geometric mean compression)
     (d) DO-Merging inspired: equalize directions, preserve relative magnitudes
  4. Evaluate: composed Gini, per-domain PPL, generation quality samples
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
# BitNet unpacking (reused from DC-Merge experiment)
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    from mlx_lm.models.bitlinear_layers import BitLinear
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
# Frobenius norm computation and equalization
# ============================================================================

def compute_per_domain_frobenius_norms(skeleton_np, all_adapters_np):
    """Compute per-domain Frobenius norms of Delta_i = s_i * B_i^T @ A_i^T.

    Since A has orthonormal rows, ||B^T @ A^T||_F = ||B||_F.
    So ||Delta_i||_F = s_i * ||B_i||_F (sum over all layers/keys).
    """
    domain_norms = {}
    domain_b_norms = {}
    per_layer_norms = {d: {} for d in DOMAINS}

    for di, domain in enumerate(DOMAINS):
        total_frobenius_sq = 0.0
        total_b_frobenius_sq = 0.0
        scale = OPTIMAL_SCALES[domain]

        for li in range(NUM_LAYERS):
            layer_frob_sq = 0.0
            for key in TARGET_KEYS:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                B = all_adapters_np[di][bkey].astype(np.float64)
                b_frob_sq = float(np.sum(B ** 2))
                total_b_frobenius_sq += b_frob_sq
                # Delta_i = s_i * B^T A^T, ||Delta_i||_F = s_i * ||B||_F (orthonormal A rows)
                delta_frob_sq = scale ** 2 * b_frob_sq
                total_frobenius_sq += delta_frob_sq
                layer_frob_sq += delta_frob_sq

            per_layer_norms[domain][li] = math.sqrt(layer_frob_sq)

        domain_norms[domain] = math.sqrt(total_frobenius_sq)
        domain_b_norms[domain] = math.sqrt(total_b_frobenius_sq)

    return domain_norms, domain_b_norms, per_layer_norms


def compute_equalization_scales(domain_norms, method="full"):
    """Compute per-domain scaling factors for Frobenius equalization.

    Methods:
      full -- all domains get equal Frobenius norm (geometric mean target)
      partial -- compress ratio via sqrt (geometric mean of log norms)
      unit -- all domains get unit Frobenius norm per layer-key
    """
    norms = np.array([domain_norms[d] for d in DOMAINS])
    log_norms = np.log(norms + 1e-30)
    geo_mean = np.exp(np.mean(log_norms))

    if method == "full":
        # Full equalization: target = geometric mean
        scales = geo_mean / norms
    elif method == "partial":
        # Partial: compress in log space by 50%
        # new_log = mean(log) + 0.5 * (log_i - mean(log))
        mean_log = np.mean(log_norms)
        new_log = mean_log + 0.5 * (log_norms - mean_log)
        new_norms = np.exp(new_log)
        scales = new_norms / norms
    elif method == "unit":
        # Equal weight per domain (each domain's total delta has norm = 1)
        scales = 1.0 / norms
    else:
        raise ValueError(f"Unknown method: {method}")

    return {d: float(scales[i]) for i, d in enumerate(DOMAINS)}


# ============================================================================
# Delta composition with optional Frobenius equalization
# ============================================================================

def compose_deltas(skeleton_np, all_adapters_np, eq_scales=None):
    """Compose N=5 adapter deltas with optional per-domain scaling.

    eq_scales: dict {domain: multiplier} or None for raw sum.
    Returns: dict (li, key) -> np.array delta
    """
    delta_dict = {}

    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            composed = None
            for di, domain in enumerate(DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue

                A = skeleton_np[skey].astype(np.float64)      # (d_in, r)
                B = all_adapters_np[di][bkey].astype(np.float64)  # (r, d_out)
                scale = OPTIMAL_SCALES[domain]

                delta = scale * (B.T @ A.T)  # (d_out, d_in)

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
    """Compute Gini coefficient of composed deltas on sampled (layer, key) pairs."""
    if sample_layers is None:
        sample_layers = [0, 5, 10, 15, 20, 25, 29]
    if sample_keys is None:
        sample_keys = ["self_attn.q_proj", "mlp.gate_proj"]

    ginis = []
    ratios = []
    top1_fracs = []

    for li in sample_layers:
        for key in sample_keys:
            if (li, key) not in delta_dict:
                continue
            composed = delta_dict[(li, key)]
            _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
            S_nz = S_c[S_c > 1e-6]
            if len(S_nz) > 1:
                ginis.append(gini_coefficient(S_nz))
                ratios.append(float(S_nz[0] / S_nz[-1]))
                top1_fracs.append(float(S_nz[0] ** 2 / np.sum(S_nz ** 2)))

    return {
        "mean_gini": float(np.mean(ginis)),
        "std_gini": float(np.std(ginis)),
        "mean_ratio": float(np.mean(ratios)),
        "mean_top1_frac": float(np.mean(top1_fracs)),
        "n_samples": len(ginis),
        "per_sample_ginis": [float(g) for g in ginis],
    }


# ============================================================================
# Model weight manipulation (reused from DC-Merge)
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

        # Evaluate periodically to prevent graph explosion
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
# Phase 1: Frobenius Norm Analysis (numpy only)
# ============================================================================

def phase_frobenius_analysis(skeleton_np, all_adapters_np):
    """Measure per-domain Frobenius norms and energy fractions."""
    log("\n" + "=" * 70)
    log("PHASE 1: Frobenius Norm Analysis")
    log("=" * 70)
    t0 = time.time()

    domain_norms, domain_b_norms, per_layer_norms = compute_per_domain_frobenius_norms(
        skeleton_np, all_adapters_np
    )

    log("\n  Per-domain Frobenius norms:")
    log(f"  {'Domain':10s} {'Scale':>6s} {'||B||_F':>10s} {'||Delta||_F':>12s} {'Energy %':>10s}")
    log("  " + "-" * 54)

    total_energy = sum(n ** 2 for n in domain_norms.values())
    for domain in DOMAINS:
        energy_frac = domain_norms[domain] ** 2 / total_energy * 100
        log(f"  {domain:10s} {OPTIMAL_SCALES[domain]:>6.0f} "
            f"{domain_b_norms[domain]:>10.4f} "
            f"{domain_norms[domain]:>12.4f} "
            f"{energy_frac:>9.1f}%")

    # Compute scale ratios
    max_norm = max(domain_norms.values())
    min_norm = min(domain_norms.values())
    norm_ratio = max_norm / min_norm

    log(f"\n  Max/min Frobenius norm ratio: {norm_ratio:.1f}:1")
    log(f"  Top-3 energy share: {sum(domain_norms[d] ** 2 for d in ['medical', 'code', 'math']) / total_energy * 100:.1f}%")

    # Equalization scale factors
    eq_methods = ["full", "partial"]
    eq_scales_all = {}
    for method in eq_methods:
        eq_scales = compute_equalization_scales(domain_norms, method)
        eq_scales_all[method] = eq_scales
        log(f"\n  Equalization scales ({method}):")
        for domain in DOMAINS:
            log(f"    {domain}: {eq_scales[domain]:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Phase 1 time: {elapsed:.1f}s")

    return {
        "domain_norms": {d: float(v) for d, v in domain_norms.items()},
        "domain_b_norms": {d: float(v) for d, v in domain_b_norms.items()},
        "energy_fractions": {d: float(domain_norms[d] ** 2 / total_energy) for d in DOMAINS},
        "norm_ratio": float(norm_ratio),
        "top3_energy_share": float(sum(domain_norms[d] ** 2 for d in ["medical", "code", "math"]) / total_energy),
        "equalization_scales": eq_scales_all,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 2: Gini Reduction Under Equalization (numpy only)
# ============================================================================

def phase_gini_analysis(skeleton_np, all_adapters_np, eq_scales_all):
    """Measure composed Gini for raw sum vs equalized compositions."""
    log("\n" + "=" * 70)
    log("PHASE 2: Gini Analysis Under Equalization")
    log("=" * 70)
    t0 = time.time()

    strategies = {
        "raw_sum": None,
        "full_equalization": eq_scales_all["full"],
        "partial_equalization": eq_scales_all["partial"],
    }

    results = {}
    for name, eq_scales in strategies.items():
        delta_dict = compose_deltas(skeleton_np, all_adapters_np, eq_scales)
        gini_result = compute_composed_gini(delta_dict)
        results[name] = gini_result
        log(f"\n  {name}:")
        log(f"    Composed Gini = {gini_result['mean_gini']:.4f} +/- {gini_result['std_gini']:.4f}")
        log(f"    Max/min SV ratio = {gini_result['mean_ratio']:.1f}")
        log(f"    Top-1 energy fraction = {gini_result['mean_top1_frac']:.4f}")
        del delta_dict

    # Gini reduction assessment
    raw_gini = results["raw_sum"]["mean_gini"]
    full_gini = results["full_equalization"]["mean_gini"]
    partial_gini = results["partial_equalization"]["mean_gini"]

    full_reduction = (raw_gini - full_gini) / raw_gini
    partial_reduction = (raw_gini - partial_gini) / raw_gini

    log(f"\n  Gini reduction (full):    {full_reduction:.1%} ({raw_gini:.4f} -> {full_gini:.4f})")
    log(f"  Gini reduction (partial): {partial_reduction:.1%} ({raw_gini:.4f} -> {partial_gini:.4f})")

    k703_pass = full_gini < 0.30
    log(f"\n  K703: Gini < 0.30 after full equalization? {full_gini:.4f} -> {'PASS' if k703_pass else 'FAIL'}")

    elapsed = time.time() - t0
    log(f"  Phase 2 time: {elapsed:.1f}s")

    return {
        "strategies": results,
        "gini_reduction_full": float(full_reduction),
        "gini_reduction_partial": float(partial_reduction),
        "k703_pass": k703_pass,
        "k703_value": float(full_gini),
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 3: Perplexity Evaluation (needs model)
# ============================================================================

def phase_perplexity(skeleton_np, all_adapters_np, eq_scales_all):
    """Compare PPL: base, raw sum, full equalization, partial equalization."""
    log("\n" + "=" * 70)
    log("PHASE 3: Perplexity Evaluation")
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

    # 3a. Base model PPL
    log("\n  Base model PPL:")
    base_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        base_ppl[domain] = math.exp(loss)
        log(f"    {domain}: {base_ppl[domain]:.4f}")

    strategies = {
        "raw_sum": None,
        "full_equalization": eq_scales_all["full"],
        "partial_equalization": eq_scales_all["partial"],
    }

    ppl_results = {}
    for name, eq_scales in strategies.items():
        log(f"\n  {name} composition PPL:")
        delta_dict_np = compose_deltas(skeleton_np, all_adapters_np, eq_scales)

        apply_delta_dict_to_model(model, delta_dict_np)

        domain_ppl = {}
        for domain in DOMAINS:
            loss = compute_val_loss(model, tokenizer, domain_texts[domain])
            domain_ppl[domain] = math.exp(loss)
            log(f"    {domain}: {domain_ppl[domain]:.4f}")

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

    # K704 assessment: per-domain PPL comparison
    raw_ppl = ppl_results["raw_sum"]["per_domain_ppl"]
    full_ppl = ppl_results["full_equalization"]["per_domain_ppl"]

    within_5pct = 0
    ppl_changes = {}
    for domain in DOMAINS:
        change = (full_ppl[domain] - raw_ppl[domain]) / raw_ppl[domain]
        ppl_changes[domain] = change
        if abs(change) <= 0.05:
            within_5pct += 1

    k704_pass = within_5pct >= 3
    log(f"\n  K704: PPL within 5% for full equalization:")
    for domain in DOMAINS:
        status = "OK" if abs(ppl_changes[domain]) <= 0.05 else "EXCEED"
        log(f"    {domain}: {ppl_changes[domain]:+.2%} [{status}]")
    log(f"  K704: {within_5pct}/5 within 5% -> {'PASS' if k704_pass else 'FAIL'}")

    # Also check partial
    partial_ppl = ppl_results["partial_equalization"]["per_domain_ppl"]
    partial_changes = {}
    partial_within_5 = 0
    for domain in DOMAINS:
        change = (partial_ppl[domain] - raw_ppl[domain]) / raw_ppl[domain]
        partial_changes[domain] = change
        if abs(change) <= 0.05:
            partial_within_5 += 1
    log(f"\n  Partial equalization: {partial_within_5}/5 within 5%")
    for domain in DOMAINS:
        log(f"    {domain}: {partial_changes[domain]:+.2%}")

    elapsed = time.time() - t0
    log(f"\n  Phase 3 time: {elapsed:.1f}s")
    log_memory("phase3")

    result = {
        "base_ppl": base_ppl,
        "strategies": ppl_results,
        "full_eq_ppl_changes": {d: float(v) for d, v in ppl_changes.items()},
        "full_eq_within_5pct": within_5pct,
        "partial_eq_ppl_changes": {d: float(v) for d, v in partial_changes.items()},
        "partial_eq_within_5pct": partial_within_5,
        "k704_pass": k704_pass,
        "elapsed_s": elapsed,
    }

    # Phase 3b: Generation quality (done within same model load to save time)
    log("\n  --- Generation Quality Evaluation ---")

    gen_prompts = {
        "medical": "The patient presents with acute chest pain and elevated troponin levels. The differential diagnosis includes",
        "code": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number efficiently.\"\"\"\n",
        "math": "To solve the integral of x^2 * e^x dx, we apply integration by parts. Let u = x^2 and dv = e^x dx. Then",
        "legal": "The court held that the defendant's Fourth Amendment rights were violated because the warrantless search of",
        "finance": "The company reported Q3 earnings of $2.50 per share, exceeding analyst estimates by 15%. The revenue growth was driven by",
    }

    gen_results = {}
    best_strategy_for_gen = "full_equalization"  # test the most aggressive equalization
    delta_dict_np = compose_deltas(skeleton_np, all_adapters_np, eq_scales_all["full"])
    apply_delta_dict_to_model(model, delta_dict_np)

    for domain, prompt in gen_prompts.items():
        output = generate_text(model, tokenizer, prompt, max_tokens=80)
        gen_results[domain] = {
            "prompt": prompt,
            "output": output,
        }
        log(f"\n    [{domain}] {output[:150]}...")

    restore_base_weights(model, base_weights)

    # Also generate with raw sum for comparison
    log("\n  --- Raw sum generation for comparison ---")
    raw_gen_results = {}
    raw_delta_dict_np = compose_deltas(skeleton_np, all_adapters_np, None)
    apply_delta_dict_to_model(model, raw_delta_dict_np)

    for domain, prompt in gen_prompts.items():
        output = generate_text(model, tokenizer, prompt, max_tokens=80)
        raw_gen_results[domain] = {
            "prompt": prompt,
            "output": output,
        }
        log(f"\n    [{domain}] {output[:150]}...")

    restore_base_weights(model, base_weights)

    result["generation_full_eq"] = gen_results
    result["generation_raw_sum"] = raw_gen_results

    cleanup(model, tokenizer, base_weights)

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    results = {"experiment": "frobenius_norm_equalization", "phases": {}}

    log("=" * 70)
    log("Frobenius-Norm Equalized Composition for Cross-Domain Scale Balancing")
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

    # Phase 1: Frobenius norm analysis
    phase1 = phase_frobenius_analysis(skeleton_np, all_adapters_np)
    results["phases"]["frobenius_analysis"] = phase1

    # Phase 2: Gini analysis
    phase2 = phase_gini_analysis(
        skeleton_np, all_adapters_np, phase1["equalization_scales"]
    )
    results["phases"]["gini_analysis"] = phase2

    gc.collect()

    # Phase 3: Perplexity + generation (needs model)
    phase3 = phase_perplexity(
        skeleton_np, all_adapters_np, phase1["equalization_scales"]
    )
    results["phases"]["perplexity"] = phase3

    # Summary
    elapsed = time.time() - t_start
    k703 = phase2["k703_pass"]
    k704 = phase3["k704_pass"]

    # K705: assess generation quality (manual, but we provide data)
    # Count domains with coherent output (non-repetitive, domain-relevant)
    gen_data = phase3.get("generation_full_eq", {})
    k705_note = "Requires manual assessment of generation samples"

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  K703 (Gini < 0.30 after equalization):  {'PASS' if k703 else 'FAIL'} (Gini={phase2['k703_value']:.4f})")
    log(f"  K704 (>=3/5 domains within 5% PPL):     {'PASS' if k704 else 'FAIL'} ({phase3['full_eq_within_5pct']}/5)")
    log(f"  K705 (generation quality):              {k705_note}")
    log(f"\n  Norm ratio (max/min): {phase1['norm_ratio']:.1f}:1")
    log(f"  Top-3 energy share: {phase1['top3_energy_share']:.1%}")
    log(f"  Gini reduction (full): {phase2['gini_reduction_full']:.1%}")
    log(f"  Gini reduction (partial): {phase2['gini_reduction_partial']:.1%}")
    log(f"\n  PPL changes (full eq vs raw sum):")
    for domain in DOMAINS:
        log(f"    {domain}: {phase3['full_eq_ppl_changes'][domain]:+.2%}")
    log(f"\n  Total time: {elapsed:.0f}s")

    results["summary"] = {
        "k703_pass": k703,
        "k703_value": phase2["k703_value"],
        "k704_pass": k704,
        "k704_count": phase3["full_eq_within_5pct"],
        "k705_note": k705_note,
        "norm_ratio": phase1["norm_ratio"],
        "top3_energy_share": phase1["top3_energy_share"],
        "gini_reduction_full": phase2["gini_reduction_full"],
        "gini_reduction_partial": phase2["gini_reduction_partial"],
        "full_eq_ppl_changes": phase3["full_eq_ppl_changes"],
        "partial_eq_ppl_changes": phase3["partial_eq_ppl_changes"],
        "elapsed_seconds": elapsed,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
