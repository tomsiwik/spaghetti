#!/usr/bin/env python3
"""Fisher-Rao Manifold Composition: Stable Adapter Merging Beyond N=5.

Kill criteria:
  K690: Activation variance preserved within 10% at N=10 vs N=1
  K691: Effective rank degradation < 5% at N=10
  K692: Fisher-Rao outperforms linear averaging on perplexity at N>5

Type: Verification (Type 1)
Paper: arXiv:2603.04972 — Fisher-Rao Manifold Merging

Approach:
  1. Load 5 trained domain adapters (real_data_domain_experts)
  2. Apply Fisher-Rao Karcher mean at B-matrix level (rank-16 vectors)
  3. Create synthetic adapter sets at N=5,10,15 via B-matrix perturbation
  4. Compare Fisher-Rao Karcher mean vs Euclidean averaging
  5. Measure: norm shrinkage, activation variance, effective rank, perplexity

Key design choice: operate on B-matrix vectors (rank 16) rather than materialized
full deltas (d_out x d_in) to stay within memory budget. The delta is scale*B^T@A^T,
with A frozen on the Grassmannian. Since A is shared per adapter slot and frozen,
the meaningful degree of freedom is B. We apply Fisher-Rao merging to B matrices
and then reconstruct delta = scale * B_merged^T @ A^T.
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
from mlx_lm.models.bitlinear_layers import BitLinear

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

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0, "code": 20.0, "math": 20.0,
    "legal": 4.0, "finance": 1.0,
}

# Fisher-Rao imports from existing implementation
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent.parent))
from pierre.merge.fisher_rao_merging.src.model import normalize, karcher_mean_spherical, slerp

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# N values to test
N_VALUES = [1, 3, 5, 10, 15]
N_VAL_SAMPLES = 50
N_ACT_SAMPLES = 30
NOISE_SCALE = 0.1


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


# ============================================================================
# BitNet unpacking
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
# B-matrix level operations (memory-efficient)
# ============================================================================

def load_all_b_matrices():
    """Load B matrices from all 5 adapters. Returns {domain: {b_key: mx.array}}."""
    all_b = {}
    for domain in DOMAINS:
        adapter_path = NTP_ADAPTERS_DIR / domain / "adapter.npz"
        adapter = dict(mx.load(str(adapter_path)))
        b_matrices = {}
        for k, v in adapter.items():
            if ".lora_b" in k:
                b_matrices[k] = v.astype(mx.float32)
        all_b[domain] = b_matrices
        del adapter
    gc.collect()
    mx.clear_cache()
    return all_b


def create_b_matrix_variants(all_b, n_total, rng):
    """Create N B-matrix sets by cycling real adapters + noise for N>5.

    Each variant is a dict {b_key: mx.array} like the originals.
    """
    real_domains = list(all_b.keys())
    n_real = len(real_domains)
    variants = []

    for i in range(n_total):
        base_domain = real_domains[i % n_real]
        base_b = all_b[base_domain]

        if i < n_real:
            variants.append(base_b)
        else:
            noisy_b = {}
            for k, b in base_b.items():
                noise = mx.array(rng.normal(0, 1, b.shape).astype(np.float32))
                b_norm = mx.sqrt(mx.sum(b * b) + 1e-8)
                noisy_b[k] = b + NOISE_SCALE * b_norm * noise / mx.sqrt(mx.array(float(b.size)))
                mx.eval(noisy_b[k])
            variants.append(noisy_b)

    return variants


def compose_b_euclidean(b_list):
    """Euclidean averaging of B matrices: B_merged = (1/N) * sum(B_i)."""
    n = len(b_list)
    all_keys = set()
    for b in b_list:
        all_keys.update(b.keys())

    composed = {}
    for key in all_keys:
        total = None
        count = 0
        for b in b_list:
            if key in b:
                if total is None:
                    total = b[key].astype(mx.float32)
                else:
                    total = total + b[key].astype(mx.float32)
                count += 1
        if total is not None:
            composed[key] = total / float(count)

    mx.eval(*composed.values())
    return composed


def compose_b_norm_rescaled_euclidean(b_list):
    """Euclidean mean with norm rescaling: compute Euclidean mean, then rescale
    to have mean source norm. This isolates the directional benefit of Karcher
    mean from the trivial norm preservation that a one-line rescaling achieves.

    result = euclidean_mean * (mean_source_norm / euclidean_mean_norm)
    """
    all_keys = set()
    for b in b_list:
        all_keys.update(b.keys())

    composed = {}
    for key in all_keys:
        available = [b[key].astype(mx.float32) for b in b_list if key in b]
        if not available:
            continue

        # Compute Euclidean mean
        euc_mean = available[0]
        for v in available[1:]:
            euc_mean = euc_mean + v
        euc_mean = euc_mean / float(len(available))

        # Measure mean source norm (flattened)
        source_norms = []
        for v in available:
            flat = v.reshape(-1)
            source_norms.append(mx.sqrt(mx.sum(flat * flat)))
        mean_source_norm = mx.mean(mx.stack(source_norms))

        # Measure Euclidean mean norm
        euc_flat = euc_mean.reshape(-1)
        euc_norm = mx.sqrt(mx.sum(euc_flat * euc_flat))

        # Rescale
        mx.eval(mean_source_norm, euc_norm)
        if euc_norm.item() > 1e-8:
            scale_factor = mean_source_norm / euc_norm
            composed[key] = euc_mean * scale_factor
        else:
            composed[key] = euc_mean

    mx.eval(*composed.values())
    return composed


def compose_b_fisher_rao(b_list):
    """Fisher-Rao Karcher mean of B matrices on S^(d-1) per block.

    For each B matrix (shape [rank, d_out]):
      1. Flatten to vector
      2. Normalize to unit sphere
      3. Compute Karcher mean direction
      4. Rescale by mean source norm
    """
    n = len(b_list)
    all_keys = set()
    for b in b_list:
        all_keys.update(b.keys())

    composed = {}
    norm_ratios = []

    for key in all_keys:
        available = [b[key] for b in b_list if key in b]
        if not available:
            continue

        original_shape = available[0].shape
        w = [1.0 / len(available)] * len(available)

        # Flatten each B matrix to a vector
        flat = [b.reshape(-1).astype(mx.float32) for b in available]

        # Normalize to unit sphere, store norms
        unit_vectors = []
        norms = []
        for v in flat:
            uv, norm = normalize(v)
            unit_vectors.append(uv)
            norms.append(norm)

        mean_norm = mx.mean(mx.stack(norms))
        mx.eval(mean_norm)

        if mean_norm.item() < 1e-8:
            composed[key] = available[0]
            continue

        # Compute Karcher mean
        if len(unit_vectors) == 1:
            direction = unit_vectors[0]
        elif len(unit_vectors) == 2:
            direction = slerp(unit_vectors[0], unit_vectors[1], t=w[1])
            mx.eval(direction)
        else:
            direction = karcher_mean_spherical(
                points=unit_vectors,
                weights=w,
                max_iter=50,
                step_size=1.0,
                tol=1e-6,
            )

        # Rescale and reshape
        merged_flat = mean_norm * direction
        composed[key] = merged_flat.reshape(original_shape)
        mx.eval(composed[key])

        # Track norm ratio
        composed_norm = mx.sqrt(mx.sum(merged_flat * merged_flat)).item()
        nr = composed_norm / max(mean_norm.item(), 1e-8)
        norm_ratios.append(nr)

        del flat, unit_vectors, norms

    return composed, norm_ratios


def measure_b_norm_shrinkage(composed_b, b_list):
    """Measure norm shrinkage at B-matrix level."""
    ratios = []
    for key in composed_b.keys():
        c_flat = composed_b[key].reshape(-1)
        c_norm = mx.sqrt(mx.sum(c_flat * c_flat)).item()

        source_norms = []
        for b in b_list:
            if key in b:
                s_flat = b[key].reshape(-1)
                s_norm = mx.sqrt(mx.sum(s_flat * s_flat)).item()
                source_norms.append(s_norm)

        if source_norms and np.mean(source_norms) > 1e-8:
            ratios.append(c_norm / np.mean(source_norms))

    return float(np.mean(ratios)) if ratios else 0.0


# ============================================================================
# Apply composed B matrices as deltas to model
# ============================================================================

def apply_composed_b_to_model(model, composed_b, skeleton, scale_override=None):
    """Apply composed B matrices to model via delta = scale * B^T @ A^T.

    Uses per-domain-averaged scale since adapters from multiple domains are mixed.
    For N=1 (single domain), uses that domain's scale.
    """
    if scale_override is None:
        # Use mean scale across all domains for multi-domain composition
        scale = np.mean(list(OPTIMAL_SCALES.values()))
    else:
        scale = scale_override

    merge_count = 0
    for li in range(30):
        for key in TARGET_KEYS:
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in composed_b:
                continue

            # Find matching A matrix — use domain 0's slot (all domains share skeleton layout)
            # Try all domain slots until we find one
            a_mx = None
            for di in range(5):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_mx = mx.array(skeleton[skey]).astype(mx.float32)
                    break

            if a_mx is None:
                continue

            b_mx = composed_b[b_key].astype(mx.float32)
            delta = scale * (b_mx.T @ a_mx.T)  # (d_out, d_in)

            # Apply to model weight
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                w_dtype = module.weight.dtype
                module.weight = module.weight.astype(mx.float32) + delta
                module.weight = module.weight.astype(w_dtype)
                merge_count += 1

            del delta

    mx.eval(model.parameters())
    return merge_count


# ============================================================================
# Model weight save/restore
# ============================================================================

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
# Metrics
# ============================================================================

def compute_activation_variance(model, tokenizer, texts, n_samples=N_ACT_SAMPLES):
    """Mean activation variance across hidden dimensions."""
    hiddens = []
    for text in texts[:n_samples]:
        ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
        if len(ids) < 2:
            continue
        x = mx.array([ids])
        h = model.model(x)
        h_mean = mx.mean(h, axis=1).squeeze(0)
        hiddens.append(h_mean)
        mx.eval(h_mean)
        del h

    if not hiddens:
        return 0.0

    H = mx.stack(hiddens, axis=0)
    var_per_feature = mx.var(H, axis=0)
    mean_var = mx.mean(var_per_feature)
    mx.eval(mean_var)
    result = mean_var.item()
    del H, hiddens
    return result


def compute_effective_rank(model, tokenizer, texts, n_samples=N_ACT_SAMPLES):
    """Effective rank = exp(spectral entropy) of activation matrix."""
    hiddens = []
    for text in texts[:n_samples]:
        ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
        if len(ids) < 2:
            continue
        x = mx.array([ids])
        h = model.model(x)
        h_mean = mx.mean(h, axis=1).squeeze(0)
        hiddens.append(h_mean)
        mx.eval(h_mean)
        del h

    if len(hiddens) < 2:
        return 0.0

    H = mx.stack(hiddens, axis=0).astype(mx.float32)
    mx.eval(H)

    n, d = H.shape
    cov = (H.T @ H) / float(n)
    eigvals = mx.linalg.eigvalsh(cov, stream=mx.cpu)
    mx.eval(eigvals)

    eigvals = mx.maximum(eigvals, 1e-10)
    p = eigvals / mx.sum(eigvals)
    entropy = -mx.sum(p * mx.log(p + 1e-10))
    eff_rank = mx.exp(entropy)
    mx.eval(eff_rank)
    result = eff_rank.item()
    del H, cov, eigvals
    return result


def compute_perplexity(model, tokenizer, texts, n_samples=N_VAL_SAMPLES):
    """Compute perplexity on validation texts."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts[:n_samples]:
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
        del logits

    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ============================================================================
# Data loading
# ============================================================================

def load_domain_val_data(domain, n=N_VAL_SAMPLES):
    val_path = NTP_DATA_DIR / domain / "valid.jsonl"
    texts = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            texts.append(text)
            if len(texts) >= n:
                break
    return texts


def load_all_val_texts():
    all_texts = []
    for domain in DOMAINS:
        texts = load_domain_val_data(domain, N_ACT_SAMPLES)
        all_texts.extend(texts)
    return all_texts


# ============================================================================
# Phase: Evaluate one composition
# ============================================================================

def phase_evaluate(model, tokenizer, base_weights, composed_b, skeleton,
                   val_texts, label="", scale_override=None):
    """Apply composed B via factored delta, measure all metrics, restore base."""
    log(f"\n  Evaluating {label}...")

    merge_count = apply_composed_b_to_model(model, composed_b, skeleton, scale_override)
    log(f"    Applied {merge_count} delta blocks")

    # Activation variance
    act_var = compute_activation_variance(model, tokenizer, val_texts)
    log(f"    Activation variance: {act_var:.6f}")

    # Effective rank
    eff_rank = compute_effective_rank(model, tokenizer, val_texts)
    log(f"    Effective rank: {eff_rank:.2f}")

    # Per-domain perplexity
    ppls = {}
    for domain in DOMAINS:
        domain_texts = load_domain_val_data(domain, N_VAL_SAMPLES)
        ppl = compute_perplexity(model, tokenizer, domain_texts, N_VAL_SAMPLES)
        ppls[domain] = ppl
        log(f"    PPL ({domain}): {ppl:.2f}")
        del domain_texts

    mean_ppl = float(np.mean(list(ppls.values())))
    log(f"    Mean PPL: {mean_ppl:.2f}")

    restore_base_weights(model, base_weights)
    gc.collect()
    mx.clear_cache()

    return {
        "activation_variance": act_var,
        "effective_rank": eff_rank,
        "per_domain_ppl": ppls,
        "mean_ppl": mean_ppl,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    results = {
        "experiment": "fisher_rao_composition_scaling",
        "paper": "arXiv:2603.04972",
        "kill_criteria": {
            "K690": "Activation variance preserved within 10% at N=10 vs N=1",
            "K691": "Effective rank degradation < 5% at N=10",
            "K692": "Fisher-Rao outperforms linear averaging on perplexity at N>5",
        },
    }

    log("=" * 70)
    log("Fisher-Rao Manifold Composition: Scaling Experiment (v2 — B-matrix level)")
    log("=" * 70)

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("model-loaded")

    # Load skeleton
    log("\nLoading Grassmannian skeleton...")
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"  Skeleton keys: {len(skeleton)}")

    # Load all B matrices (much smaller than full deltas)
    log("\nLoading B matrices from all 5 adapters...")
    all_b = load_all_b_matrices()
    for domain, b in all_b.items():
        n_keys = len(b)
        total_params = sum(v.size for v in b.values())
        log(f"  {domain}: {n_keys} B matrices, {total_params:,} params")

    base_weights = save_base_weights(model)
    val_texts = load_all_val_texts()
    log(f"\nLoaded {len(val_texts)} validation texts across all domains")

    # Baseline
    log(f"\n{'=' * 70}")
    log("BASELINE: Base model (no adapters)")
    log(f"{'=' * 70}")
    base_act_var = compute_activation_variance(model, tokenizer, val_texts)
    base_eff_rank = compute_effective_rank(model, tokenizer, val_texts)
    log(f"  Base activation variance: {base_act_var:.6f}")
    log(f"  Base effective rank: {base_eff_rank:.2f}")
    results["baseline"] = {
        "activation_variance": base_act_var,
        "effective_rank": base_eff_rank,
    }
    log_memory("baseline")

    # Main experiment loop
    rng = np.random.RandomState(SEED)
    scaling_results = {}

    for N in N_VALUES:
        log(f"\n{'=' * 70}")
        log(f"N = {N} adapters")
        log(f"{'=' * 70}")

        # Create B-matrix variants
        b_list = create_b_matrix_variants(all_b, N, rng)
        log(f"  Created {len(b_list)} B-matrix sets")

        n_results = {"N": N}

        # Use consistent mean scale for ALL N values (Fix #4: eliminates
        # scale confound between N=1 and N>1 that contaminated act. var. comparisons).
        # N=1 still uses only the medical adapter's B-matrix, but at scale=12.8
        # for fair cross-N comparison.
        scale = np.mean(list(OPTIMAL_SCALES.values()))

        # --- Euclidean ---
        log(f"\n  --- Euclidean averaging (N={N}) ---")
        t0 = time.time()
        euc_b = compose_b_euclidean(b_list)
        euc_time = time.time() - t0
        euc_shrinkage = measure_b_norm_shrinkage(euc_b, b_list)
        theory_shrinkage = 1.0 / math.sqrt(N) if N > 1 else 1.0
        log(f"    Composition time: {euc_time:.2f}s")
        log(f"    B-matrix norm shrinkage: {euc_shrinkage:.4f} (theory: {theory_shrinkage:.4f})")

        euc_metrics = phase_evaluate(
            model, tokenizer, base_weights, euc_b, skeleton,
            val_texts, label=f"Euclidean N={N}", scale_override=scale,
        )
        euc_metrics["norm_shrinkage"] = euc_shrinkage
        euc_metrics["time_s"] = euc_time
        n_results["euclidean"] = euc_metrics

        del euc_b
        gc.collect()
        mx.clear_cache()

        # --- Norm-Rescaled Euclidean ---
        log(f"\n  --- Norm-Rescaled Euclidean (N={N}) ---")
        t0 = time.time()
        nre_b = compose_b_norm_rescaled_euclidean(b_list)
        nre_time = time.time() - t0
        nre_shrinkage = measure_b_norm_shrinkage(nre_b, b_list)
        log(f"    Composition time: {nre_time:.2f}s")
        log(f"    B-matrix norm shrinkage: {nre_shrinkage:.4f} (target: 1.0)")

        nre_metrics = phase_evaluate(
            model, tokenizer, base_weights, nre_b, skeleton,
            val_texts, label=f"Norm-Rescaled Euclidean N={N}", scale_override=scale,
        )
        nre_metrics["norm_shrinkage"] = nre_shrinkage
        nre_metrics["time_s"] = nre_time
        n_results["norm_rescaled_euclidean"] = nre_metrics

        del nre_b
        gc.collect()
        mx.clear_cache()

        # --- Fisher-Rao ---
        log(f"\n  --- Fisher-Rao Karcher mean (N={N}) ---")
        t0 = time.time()
        fr_b, fr_norm_ratios = compose_b_fisher_rao(b_list)
        fr_time = time.time() - t0
        fr_shrinkage = measure_b_norm_shrinkage(fr_b, b_list)
        log(f"    Composition time: {fr_time:.2f}s")
        log(f"    B-matrix norm shrinkage: {fr_shrinkage:.4f} (theory: 1.0)")

        fr_metrics = phase_evaluate(
            model, tokenizer, base_weights, fr_b, skeleton,
            val_texts, label=f"Fisher-Rao N={N}", scale_override=scale,
        )
        fr_metrics["norm_shrinkage"] = fr_shrinkage
        fr_metrics["time_s"] = fr_time
        n_results["fisher_rao"] = fr_metrics

        del fr_b, b_list
        gc.collect()
        mx.clear_cache()

        scaling_results[str(N)] = n_results
        log_memory(f"after-N={N}")

    results["scaling"] = scaling_results

    # ========================================================================
    # Kill criteria assessment
    # ========================================================================
    log(f"\n{'=' * 70}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'=' * 70}")

    ref_act_var = scaling_results["1"]["fisher_rao"]["activation_variance"]
    ref_eff_rank = scaling_results["1"]["fisher_rao"]["effective_rank"]

    # K690
    if "10" in scaling_results:
        fr_av10 = scaling_results["10"]["fisher_rao"]["activation_variance"]
        av_ratio = fr_av10 / max(ref_act_var, 1e-10)
        k690_pass = abs(1.0 - av_ratio) < 0.10
        euc_av10 = scaling_results["10"]["euclidean"]["activation_variance"]
        euc_av_ratio = euc_av10 / max(ref_act_var, 1e-10)
        log(f"\nK690: Act. var. ratio (FR N=10/N=1): {av_ratio:.4f} | within 10%: {'PASS' if k690_pass else 'FAIL'}")
        log(f"  Euclidean ratio: {euc_av_ratio:.4f}")
    else:
        k690_pass = False
        av_ratio = None
        euc_av_ratio = None

    # K691
    if "10" in scaling_results:
        fr_er10 = scaling_results["10"]["fisher_rao"]["effective_rank"]
        rank_deg = 1.0 - (fr_er10 / max(ref_eff_rank, 1e-10))
        k691_pass = rank_deg < 0.05
        euc_er10 = scaling_results["10"]["euclidean"]["effective_rank"]
        euc_rank_deg = 1.0 - (euc_er10 / max(ref_eff_rank, 1e-10))
        log(f"\nK691: Eff. rank degradation (FR N=10): {rank_deg:.4f} ({rank_deg*100:.1f}%) | <5%: {'PASS' if k691_pass else 'FAIL'}")
        log(f"  Euclidean degradation: {euc_rank_deg:.4f} ({euc_rank_deg*100:.1f}%)")
    else:
        k691_pass = False
        rank_deg = None
        euc_rank_deg = None

    # K692 (now also compares against norm-rescaled Euclidean)
    k692_pass = False
    k692_vs_nre = False
    ppl_comparisons = {}
    for n_str in ["10", "15"]:
        if n_str in scaling_results:
            fr_ppl = scaling_results[n_str]["fisher_rao"]["mean_ppl"]
            euc_ppl = scaling_results[n_str]["euclidean"]["mean_ppl"]
            nre_ppl = scaling_results[n_str]["norm_rescaled_euclidean"]["mean_ppl"]
            better_vs_euc = fr_ppl < euc_ppl
            better_vs_nre = fr_ppl < nre_ppl
            ppl_comparisons[n_str] = {
                "fr_ppl": fr_ppl, "euc_ppl": euc_ppl, "nre_ppl": nre_ppl,
                "fr_better_vs_euc": better_vs_euc, "fr_better_vs_nre": better_vs_nre,
                "ratio_vs_euc": fr_ppl / max(euc_ppl, 1e-10),
                "ratio_vs_nre": fr_ppl / max(nre_ppl, 1e-10),
            }
            log(f"\nK692 (N={n_str}):")
            log(f"  FR PPL={fr_ppl:.2f} vs Euc PPL={euc_ppl:.2f} | FR better: {better_vs_euc}")
            log(f"  FR PPL={fr_ppl:.2f} vs NRE PPL={nre_ppl:.2f} | FR better: {better_vs_nre}")
            if better_vs_euc:
                k692_pass = True
            if better_vs_nre:
                k692_vs_nre = True

    log(f"\nK692 vs Euclidean: {'PASS' if k692_pass else 'FAIL'}")
    log(f"K692 vs Norm-Rescaled Euclidean: {'PASS' if k692_vs_nre else 'FAIL'}")

    results["kill_results"] = {
        "K690": {"pass": k690_pass, "fr_ratio": av_ratio, "euc_ratio": euc_av_ratio},
        "K691": {"pass": k691_pass, "fr_degradation": rank_deg, "euc_degradation": euc_rank_deg},
        "K692": {"pass_vs_euclidean": k692_pass, "pass_vs_norm_rescaled": k692_vs_nre,
                 "comparisons": ppl_comparisons},
    }

    # Norm shrinkage table
    log(f"\n{'=' * 70}")
    log("NORM SHRINKAGE SUMMARY (Theorem 1)")
    log(f"{'=' * 70}")
    log(f"{'N':>4} | {'Euclidean':>10} | {'NR-Euc':>10} | {'FR':>10} | {'Theory(Euc)':>12}")
    log("-" * 58)
    for n_str, nr in scaling_results.items():
        n = int(n_str)
        euc_s = nr["euclidean"]["norm_shrinkage"]
        nre_s = nr["norm_rescaled_euclidean"]["norm_shrinkage"]
        fr_s = nr["fisher_rao"]["norm_shrinkage"]
        theory = 1.0 / math.sqrt(n) if n > 1 else 1.0
        log(f"{n:4d} | {euc_s:10.4f} | {nre_s:10.4f} | {fr_s:10.4f} | {theory:12.4f}")

    elapsed = time.time() - t_start
    results["total_time_s"] = round(elapsed, 1)
    log(f"\nTotal time: {elapsed:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    log(f"\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")
    log(f"K690 (activation variance FR N=10 vs N=1):     {'PASS' if k690_pass else 'FAIL'}")
    log(f"K691 (effective rank degradation < 5%):         {'PASS' if k691_pass else 'FAIL'}")
    log(f"K692 (FR outperforms Euclidean at N>5):         {'PASS' if k692_pass else 'FAIL'}")
    log(f"K692b (FR outperforms Norm-Rescaled Euc at N>5): {'PASS' if k692_vs_nre else 'FAIL'}")

    cleanup(model, tokenizer)


if __name__ == "__main__":
    main()
