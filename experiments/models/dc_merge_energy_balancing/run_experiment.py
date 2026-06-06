#!/usr/bin/env python3
"""DC-Merge SVD energy smoothing for balanced adapter composition.

Kill criteria:
  K699: Energy-balanced composition improves over raw sum on perplexity
  K700: Singular value Gini coefficient reduced by >30% after smoothing

Type: Guided Exploration (Type 2)
Paper: DC-Merge (arXiv:2603.06242, Zhang et al., CVPR 2026)
Prior: Finding #270 (flat ternary spectra), Finding #225 (N=5 composition)
Dependency: exp_brainstacks_null_space_validation (SUPPORTED)

Approach:
  1. Load 5 trained domain adapters (real_data_domain_experts)
  2. For each layer+key: compute task vectors (delta = scale * B^T @ A^T)
  3. Apply DC-Merge energy smoothing (average + linear)
  4. For PPL: materialize only the merged delta (not per-task), apply to model
  5. Measure Gini coefficients before/after, perplexity, DirSim

OPTIMIZATION: Since delta = scale * B^T @ A^T has rank=16 (the LoRA rank),
we compute SVD of the small B matrix (16 x d_out) instead of the full delta
(d_out x d_in). This avoids O(d^3) SVD on 2560x2560 matrices.

For B = Ub Sb Vb^T (sizes r x d_out, so Ub is r x r, Sb is r, Vb is d_out x r):
  delta = scale * B^T @ A^T = scale * Vb Sb Ub^T A^T
  SVD(delta) has singular values = scale * Sb (since A^T columns are orthonormal
  within their Grassmannian subspace).

Energy smoothing on delta's SVs is equivalent to smoothing Sb and rescaling.
The reconstructed smoothed delta = scale * Vb Sb_smooth Ub^T A^T
                                 = (scale * (Ub diag(Sb_smooth/Sb) Ub^T) B)^T @ A^T
But simpler: just reconstruct B_smooth = Ub diag(Sb_smooth) Vb^T, then
delta_smooth = scale * B_smooth^T @ A^T.
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

# DC-Merge library (for energy_smoothing and dir_sim)
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent.parent))
from pierre.merge.dc_merge.src.model import energy_smoothing

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEED = 42
N_EVAL_SAMPLES = 20  # per domain for PPL evaluation (fast)

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
# Efficient B-matrix SVD and energy smoothing
# ============================================================================

def smooth_b_matrix(B_np, strategy, rho=5.0):
    """Apply energy smoothing to a B matrix (r x d_out) via its SVD.

    B = Ub @ diag(Sb) @ Vb^T
    B_smooth = Ub @ diag(Sb_smooth) @ Vb^T

    Returns B_smooth as numpy array.
    """
    Ub, Sb, Vbt = np.linalg.svd(B_np, full_matrices=False)
    # Sb is the r singular values of B
    # Apply MLX energy_smoothing
    Sb_mx = mx.array(Sb.astype(np.float32))
    Sb_smooth = energy_smoothing(Sb_mx, strategy=strategy, rho=rho)
    mx.eval(Sb_smooth)
    Sb_smooth_np = np.array(Sb_smooth)
    # Reconstruct
    B_smooth = (Ub * Sb_smooth_np) @ Vbt
    return B_smooth, Sb, Sb_smooth_np


def compute_composed_delta_svd(skeleton_np, all_adapters_np, li, key, strategy="none", rho=5.0):
    """Compute composed delta for one (layer, key) using optional energy smoothing.

    Returns: composed_delta as mx.array, and singular values for Gini measurement.
    """
    composed = None
    for di, domain in enumerate(DOMAINS):
        skey = f"layer_{li}_{key}_domain_{di}"
        bkey = f"model.layers.{li}.{key}.lora_b"
        if skey not in skeleton_np or bkey not in all_adapters_np[di]:
            continue

        A = skeleton_np[skey].astype(np.float64)     # (d_in, r)
        B = all_adapters_np[di][bkey].astype(np.float64)  # (r, d_out)
        scale = OPTIMAL_SCALES[domain]

        if strategy != "none":
            B, _, _ = smooth_b_matrix(B, strategy, rho)

        delta = scale * (B.T @ A.T)  # (d_out, d_in)
        if composed is None:
            composed = delta
        else:
            composed = composed + delta

    return composed


# ============================================================================
# Model weight manipulation
# ============================================================================

def apply_delta_dict_to_model(model, delta_dict):
    """Apply dict of {(li, key): delta} to model weights."""
    count = 0
    for (li, key), delta_mx in delta_dict.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta_mx.astype(module.weight.dtype)
            count += 1
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
# Phase 1: Spectral Analysis (numpy only, no model needed)
# ============================================================================

def phase_spectral_analysis(skeleton_np, all_adapters_np):
    """Measure singular value distribution of individual B-matrices and composed deltas."""
    log("\n" + "=" * 70)
    log("PHASE 1: Spectral Analysis")
    log("=" * 70)
    t0 = time.time()

    individual_ginis = {d: [] for d in DOMAINS}
    individual_ratios = {d: [] for d in DOMAINS}

    # Measure individual B-matrix SVD properties
    for di, domain in enumerate(DOMAINS):
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                bkey = f"model.layers.{li}.{key}.lora_b"
                if bkey in all_adapters_np[di]:
                    B = all_adapters_np[di][bkey].astype(np.float64)
                    _, S, _ = np.linalg.svd(B, full_matrices=False)
                    S_nz = S[S > 1e-8]
                    if len(S_nz) > 1:
                        individual_ginis[domain].append(gini_coefficient(S_nz))
                        individual_ratios[domain].append(float(S_nz[0] / S_nz[-1]))

    for domain in DOMAINS:
        mg = np.mean(individual_ginis[domain])
        mr = np.mean(individual_ratios[domain])
        log(f"  {domain}: mean B-matrix Gini={mg:.4f}, mean max/min ratio={mr:.2f} (N={len(individual_ginis[domain])})")

    # Measure composed delta SVD on sample layers
    sample_layers = [0, 5, 10, 15, 20, 25, 29]
    sample_keys = ["self_attn.q_proj", "mlp.gate_proj"]
    composed_ginis = []
    composed_ratios = []
    composed_top1_fracs = []

    for li in sample_layers:
        for key in sample_keys:
            composed = compute_composed_delta_svd(skeleton_np, all_adapters_np, li, key, "none")
            if composed is not None:
                _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
                S_nz = S_c[S_c > 1e-6]
                if len(S_nz) > 1:
                    composed_ginis.append(gini_coefficient(S_nz))
                    composed_ratios.append(float(S_nz[0] / S_nz[-1]))
                    composed_top1_fracs.append(float(S_nz[0] ** 2 / np.sum(S_nz ** 2)))

    mean_cg = float(np.mean(composed_ginis))
    mean_cr = float(np.mean(composed_ratios))
    log(f"\n  Composed raw sum: Gini={mean_cg:.4f}, ratio={mean_cr:.1f}, "
        f"top-1 frac={np.mean(composed_top1_fracs):.4f}")

    elapsed = time.time() - t0
    log(f"  Phase 1 time: {elapsed:.1f}s")

    return {
        "individual_ginis": {d: float(np.mean(individual_ginis[d])) for d in DOMAINS},
        "individual_ratios": {d: float(np.mean(individual_ratios[d])) for d in DOMAINS},
        "composed_gini_raw": mean_cg,
        "composed_ratio_raw": mean_cr,
        "composed_top1_frac_raw": float(np.mean(composed_top1_fracs)),
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 2: Energy Smoothing Gini Reduction
# ============================================================================

def phase_energy_smoothing(skeleton_np, all_adapters_np):
    """Apply energy smoothing and measure Gini reduction on composed deltas."""
    log("\n" + "=" * 70)
    log("PHASE 2: Energy Smoothing — Gini Reduction")
    log("=" * 70)
    t0 = time.time()

    sample_layers = [0, 5, 10, 15, 20, 25, 29]
    sample_keys = ["self_attn.q_proj", "mlp.gate_proj"]
    strategies = ["none", "average", "linear"]
    results_by_strategy = {}

    for strategy in strategies:
        composed_ginis = []
        composed_ratios = []
        composed_top1 = []

        for li in sample_layers:
            for key in sample_keys:
                composed = compute_composed_delta_svd(
                    skeleton_np, all_adapters_np, li, key, strategy, rho=5.0
                )
                if composed is not None:
                    _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
                    S_nz = S_c[S_c > 1e-6]
                    if len(S_nz) > 1:
                        composed_ginis.append(gini_coefficient(S_nz))
                        composed_ratios.append(float(S_nz[0] / S_nz[-1]))
                        composed_top1.append(float(S_nz[0] ** 2 / np.sum(S_nz ** 2)))

        mean_gini = float(np.mean(composed_ginis))
        mean_ratio = float(np.mean(composed_ratios))
        mean_top1 = float(np.mean(composed_top1))

        log(f"  Strategy '{strategy}': Gini={mean_gini:.4f}, ratio={mean_ratio:.1f}, "
            f"top-1 frac={mean_top1:.4f}")

        results_by_strategy[strategy] = {
            "mean_gini": mean_gini,
            "mean_ratio": mean_ratio,
            "mean_top1_frac": mean_top1,
            "per_sample_ginis": composed_ginis,
        }

    # Gini reduction
    raw_gini = results_by_strategy["none"]["mean_gini"]
    avg_gini = results_by_strategy["average"]["mean_gini"]
    lin_gini = results_by_strategy["linear"]["mean_gini"]

    avg_reduction = 1.0 - avg_gini / max(raw_gini, 1e-10) if raw_gini > 0 else 0.0
    lin_reduction = 1.0 - lin_gini / max(raw_gini, 1e-10) if raw_gini > 0 else 0.0

    log(f"\n  Gini reduction (average): {avg_reduction:.1%} ({raw_gini:.4f} -> {avg_gini:.4f})")
    log(f"  Gini reduction (linear):  {lin_reduction:.1%} ({raw_gini:.4f} -> {lin_gini:.4f})")

    k700_pass = avg_reduction > 0.30
    log(f"\n  K700: Gini reduced >30%? {avg_reduction:.1%} -> {'PASS' if k700_pass else 'FAIL'}")

    elapsed = time.time() - t0
    log(f"  Phase 2 time: {elapsed:.1f}s")

    return {
        "strategies": results_by_strategy,
        "gini_reduction_average": float(avg_reduction),
        "gini_reduction_linear": float(lin_reduction),
        "k700_pass": k700_pass,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 3: DirSim Measurement
# ============================================================================

def phase_dirsim(skeleton_np, all_adapters_np):
    """Measure pairwise directional similarity before/after smoothing."""
    log("\n" + "=" * 70)
    log("PHASE 3: Directional Similarity (DirSim)")
    log("=" * 70)
    t0 = time.time()

    li = 15
    key = "self_attn.q_proj"

    # Compute per-domain B matrices
    B_mats = []
    A_mats = []
    for di, domain in enumerate(DOMAINS):
        skey = f"layer_{li}_{key}_domain_{di}"
        bkey = f"model.layers.{li}.{key}.lora_b"
        A = skeleton_np[skey].astype(np.float64)
        B = all_adapters_np[di][bkey].astype(np.float64)
        scale = OPTIMAL_SCALES[domain]
        B_mats.append(scale * B)  # scale into B for simplicity
        A_mats.append(A)

    # DirSim via B-matrices: since delta_i = B_i^T @ A_i^T, and DirSim
    # normalizes out energy, we can compute it from SVD of each delta.
    # But since deltas are small rank, use rank-16 SVD of the small B.
    # DirSim(delta_a, delta_b) = (1/sqrt(r_a * r_b)) * sum |R_ij|
    # where R_ij = (u_a_i^T u_b_j) * (v_b_j^T v_a_i)

    def compute_dirsim(B_a, A_a, B_b, A_b, r=LORA_RANK):
        """Compute DirSim between two LoRA deltas from their B,A factors."""
        # delta = B^T @ A^T, shape (d_out, d_in)
        # SVD of delta: Since rank = r, compute via B's SVD
        # B = Ub Sb Vb^T (r x d_out), so B^T = Vb Sb Ub^T (d_out x r)
        # delta = B^T @ A^T = Vb Sb Ub^T A^T (d_out x d_in)
        # This is rank-r. Left singular vectors ~ Vb, right ~ A Ub
        Ub_a, Sb_a, Vbt_a = np.linalg.svd(B_a, full_matrices=False)
        Ub_b, Sb_b, Vbt_b = np.linalg.svd(B_b, full_matrices=False)

        # Left singular vectors of delta = Vb (columns of Vb^T transposed)
        # Right singular vectors of delta = A @ Ub
        # V_a = Vbt_a.T  # (d_out, r)
        # V_b = Vbt_b.T
        # U_a = A_a @ Ub_a  # (d_in, r) -- right SV of delta
        # U_b = A_b @ Ub_b

        # R_ij = (left_a_i^T left_b_j) * (right_b_j^T right_a_i)
        left_cos = Vbt_a @ Vbt_b.T       # (r, r): Vbt_a @ Vbt_b^T = V_a^T @ V_b
        right_cos = (A_a @ Ub_a).T @ (A_b @ Ub_b)  # (r, r)

        R = left_cos * right_cos
        n = Ub_a.shape[1]
        m = Ub_b.shape[1]
        return float(np.sum(np.abs(R)) / (np.sqrt(n) * np.sqrt(m)))

    # Raw DirSim
    dirsim_raw = {}
    for i in range(len(DOMAINS)):
        for j in range(i + 1, len(DOMAINS)):
            ds = compute_dirsim(B_mats[i], A_mats[i], B_mats[j], A_mats[j])
            dirsim_raw[f"{DOMAINS[i]}_vs_{DOMAINS[j]}"] = ds

    log("  DirSim (raw):")
    for pair, val in dirsim_raw.items():
        log(f"    {pair}: {val:.4f}")
    log(f"  Mean DirSim (raw): {np.mean(list(dirsim_raw.values())):.4f}")

    # Smoothed DirSim (average)
    B_smooth = []
    for B in B_mats:
        Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)
        Sb_mx = mx.array(Sb.astype(np.float32))
        Sb_s = energy_smoothing(Sb_mx, strategy="average")
        mx.eval(Sb_s)
        Sb_s_np = np.array(Sb_s).astype(np.float64)
        B_smooth.append((Ub * Sb_s_np) @ Vbt)

    dirsim_smooth = {}
    for i in range(len(DOMAINS)):
        for j in range(i + 1, len(DOMAINS)):
            ds = compute_dirsim(B_smooth[i], A_mats[i], B_smooth[j], A_mats[j])
            dirsim_smooth[f"{DOMAINS[i]}_vs_{DOMAINS[j]}"] = ds

    log("\n  DirSim (average smoothed):")
    for pair, val in dirsim_smooth.items():
        log(f"    {pair}: {val:.4f}")
    log(f"  Mean DirSim (smoothed): {np.mean(list(dirsim_smooth.values())):.4f}")

    elapsed = time.time() - t0
    log(f"  Phase 3 time: {elapsed:.1f}s")

    return {
        "dirsim_raw": dirsim_raw,
        "dirsim_smoothed": dirsim_smooth,
        "mean_dirsim_raw": float(np.mean(list(dirsim_raw.values()))),
        "mean_dirsim_smoothed": float(np.mean(list(dirsim_smooth.values()))),
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 4: Perplexity Evaluation
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


def compose_deltas_with_smoothing(skeleton_np, all_adapters_np, strategy, rho=5.0):
    """Build composed delta dict for all layers/keys using energy smoothing.

    Returns dict: (li, key) -> mx.array delta.
    """
    delta_dict = {}
    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            composed_np = compute_composed_delta_svd(
                skeleton_np, all_adapters_np, li, key, strategy, rho
            )
            if composed_np is not None:
                delta_dict[(li, key)] = mx.array(composed_np.astype(np.float32))

        # Evaluate every 5 layers to avoid graph explosion
        if li % 5 == 4:
            mx.eval(*[v for v in delta_dict.values()])

    return delta_dict


def phase_perplexity(skeleton_np, all_adapters_np):
    """Compare PPL: base, raw sum, DC-Merge average, DC-Merge linear."""
    log("\n" + "=" * 70)
    log("PHASE 4: Perplexity Evaluation")
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
    all_val_texts = []
    for domain in DOMAINS:
        val_path = NTP_DATA_DIR / domain / "valid.jsonl"
        texts = []
        with open(val_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if len(texts) >= N_EVAL_SAMPLES:
                    break
        domain_texts[domain] = texts
        all_val_texts.extend(texts)
    log(f"  Loaded {len(all_val_texts)} total validation texts")

    base_weights = save_base_weights(model)

    # 4a. Base model PPL
    log("\n  Base model PPL:")
    base_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        base_ppl[domain] = math.exp(loss)
        log(f"    {domain}: {base_ppl[domain]:.4f}")

    base_mixed_loss = compute_val_loss(model, tokenizer, all_val_texts)
    base_mixed_ppl = math.exp(base_mixed_loss)
    log(f"    Mixed: {base_mixed_ppl:.4f}")

    # 4b. Raw sum
    log("\n  Raw sum composition:")
    raw_deltas = compose_deltas_with_smoothing(skeleton_np, all_adapters_np, "none")
    apply_delta_dict_to_model(model, raw_deltas)

    raw_sum_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        raw_sum_ppl[domain] = math.exp(loss)
        log(f"    {domain}: {raw_sum_ppl[domain]:.4f}")

    raw_mixed_loss = compute_val_loss(model, tokenizer, all_val_texts)
    raw_mixed_ppl = math.exp(raw_mixed_loss)
    log(f"    Mixed: {raw_mixed_ppl:.4f}")

    restore_base_weights(model, base_weights)
    del raw_deltas
    gc.collect()
    mx.clear_cache()

    # 4c. DC-Merge compositions
    dc_results = {}
    for strategy in ["average", "linear"]:
        log(f"\n  DC-Merge ({strategy}):")
        merged_deltas = compose_deltas_with_smoothing(
            skeleton_np, all_adapters_np, strategy, rho=5.0
        )
        apply_delta_dict_to_model(model, merged_deltas)

        ppl = {}
        for domain in DOMAINS:
            loss = compute_val_loss(model, tokenizer, domain_texts[domain])
            ppl[domain] = math.exp(loss)
            log(f"    {domain}: {ppl[domain]:.4f}")

        mixed_loss = compute_val_loss(model, tokenizer, all_val_texts)
        mixed_ppl = math.exp(mixed_loss)
        log(f"    Mixed: {mixed_ppl:.4f}")

        dc_results[strategy] = {
            "per_domain_ppl": ppl,
            "mixed_ppl": mixed_ppl,
        }

        restore_base_weights(model, base_weights)
        del merged_deltas
        gc.collect()
        mx.clear_cache()

    # K699 assessment
    best_dc_mixed = min(dc_results["average"]["mixed_ppl"], dc_results["linear"]["mixed_ppl"])
    k699_pass = best_dc_mixed < raw_mixed_ppl
    ppl_improvement = (raw_mixed_ppl - best_dc_mixed) / raw_mixed_ppl

    log(f"\n  K699: Best DC-Merge = {best_dc_mixed:.4f} vs raw sum = {raw_mixed_ppl:.4f}")
    log(f"  K699: PPL improvement = {ppl_improvement:.2%}")
    log(f"  K699: {'PASS' if k699_pass else 'FAIL'}")

    elapsed = time.time() - t0
    log(f"  Phase 4 time: {elapsed:.1f}s")
    log_memory("phase4")

    cleanup(model, tokenizer)

    return {
        "base_ppl": base_ppl,
        "base_mixed_ppl": base_mixed_ppl,
        "raw_sum_ppl": raw_sum_ppl,
        "raw_sum_mixed_ppl": raw_mixed_ppl,
        "dc_merge_results": dc_results,
        "best_dc_mixed_ppl": best_dc_mixed,
        "ppl_improvement": float(ppl_improvement),
        "k699_pass": k699_pass,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    results = {"experiment": "dc_merge_energy_balancing", "phases": {}}

    log("=" * 70)
    log("DC-Merge SVD Energy Smoothing for Balanced Adapter Composition")
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

    # Phase 1: Spectral analysis (numpy only)
    results["phases"]["spectral_analysis"] = phase_spectral_analysis(skeleton_np, all_adapters_np)

    # Phase 2: Energy smoothing Gini reduction
    results["phases"]["energy_smoothing"] = phase_energy_smoothing(skeleton_np, all_adapters_np)

    # Phase 3: DirSim measurement
    results["phases"]["dirsim"] = phase_dirsim(skeleton_np, all_adapters_np)

    gc.collect()

    # Phase 4: Perplexity (needs model)
    results["phases"]["perplexity"] = phase_perplexity(skeleton_np, all_adapters_np)

    # Summary
    elapsed = time.time() - t_start
    k699 = results["phases"]["perplexity"]["k699_pass"]
    k700 = results["phases"]["energy_smoothing"]["k700_pass"]

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  K699 (DC-Merge improves PPL):    {'PASS' if k699 else 'FAIL'}")
    log(f"    Raw sum mixed PPL:  {results['phases']['perplexity']['raw_sum_mixed_ppl']:.4f}")
    log(f"    Best DC-Merge PPL:  {results['phases']['perplexity']['best_dc_mixed_ppl']:.4f}")
    log(f"    Improvement:        {results['phases']['perplexity']['ppl_improvement']:.2%}")
    log(f"  K700 (Gini reduced >30%):        {'PASS' if k700 else 'FAIL'}")
    log(f"    Average reduction:  {results['phases']['energy_smoothing']['gini_reduction_average']:.1%}")
    log(f"    Linear reduction:   {results['phases']['energy_smoothing']['gini_reduction_linear']:.1%}")
    log(f"  Total time: {elapsed:.0f}s")

    results["summary"] = {
        "k699_pass": k699,
        "k700_pass": k700,
        "raw_sum_mixed_ppl": results["phases"]["perplexity"]["raw_sum_mixed_ppl"],
        "best_dc_mixed_ppl": results["phases"]["perplexity"]["best_dc_mixed_ppl"],
        "ppl_improvement": results["phases"]["perplexity"]["ppl_improvement"],
        "gini_reduction_average": results["phases"]["energy_smoothing"]["gini_reduction_average"],
        "gini_reduction_linear": results["phases"]["energy_smoothing"]["gini_reduction_linear"],
        "elapsed_seconds": elapsed,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
