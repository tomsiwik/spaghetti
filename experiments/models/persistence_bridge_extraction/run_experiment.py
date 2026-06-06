#!/usr/bin/env python3
"""Persistence-Guided Bridge Extraction: Sparse Low-Rank Correction.

Kill criteria:
  K628: Bridge matrix B reduces d_B (H1) by >= 50%
  K629: Bridge matrix B has rank < r=16 (cheaper than another adapter)
  K630: PPL on overlap inputs improves >= 5% with bridge vs without

Type: Guided exploration (Type 2)
Platform: Apple M5 Pro 48GB

Method:
  1. Load BitNet-2B-4T base weights + adapters (reuse from dependency experiment)
  2. For each module, compute H1 PH on base, composed, and bridge-corrected
  3. Bridge = targeted SVD correction of perturbation at various ranks
  4. Measure H1 feature loss count, bottleneck distance reduction, and PPL
  5. Focus on modules where vulnerability bound > median H1 persistence

Key insight from dependency experiment:
  - H0 features are safe (persistence >> vulnerability bound)
  - H1 features are vulnerable: 91% of H1 features in modules where
    vuln_bound > median H1 persistence
  - This experiment targets H1 restoration specifically
"""

import gc
import json
import math
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

ADAPTER_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters"
SKELETON_PATH = ADAPTER_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
DOMAIN_NAMES = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = 5
LORA_RANK = 16
LORA_SCALE = 20.0

# Subsample rows for feasibility (Rips is O(n^3))
N_SUBSAMPLE = 300  # Reduced from 500 for speed; still sufficient for H1 detection
# Focus on modules where H1 is most vulnerable (from dependency results)
# Use a wider set of layers to get sufficient signal
TARGET_LAYERS = [0, 7, 15, 22, 29]
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
MAX_DIM = 1  # H0 and H1
BRIDGE_RANKS = [2, 8, 16]  # Key ranks: very sparse, moderate, full adapter cost

# Modules to SKIP detailed bridge analysis (low H1 feature count from dependency data)
# Only do full bridge sweep on modules with substantial H1 features (>50)
MIN_H1_FOR_BRIDGE = 10  # Only bridge-analyze modules with >= 10 H1 base features

# PPL eval parameters
MAX_SEQ_LENGTH = 512
PPL_BATCHES = 30  # samples per domain for PPL


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase_load_weights():
    """Load base model weights and adapter data as numpy arrays."""
    log("Loading model weights...")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_unflatten

    model, _ = load(MODEL_ID)

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

    for li in TARGET_LAYERS:
        layer = model.model.layers[li]
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                mx.eval(unpacked_w)
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log("  Unpacked BitLinear weights for target layers")

    weights = {}
    for li in TARGET_LAYERS:
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:
                continue
            W = module.weight
            mx.eval(W)
            w_np = np.array(W.astype(mx.float32)).T  # (in_features, out_features)
            weights[(li, key)] = w_np
            log(f"  layer {li} {key}: W^T shape={w_np.shape}")

    del model
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    log("Loading adapters and skeleton...")
    skeleton = dict(np.load(str(SKELETON_PATH)))
    adapters = {}
    for domain in DOMAIN_NAMES:
        path = ADAPTER_DIR / domain / "adapter.npz"
        adapters[domain] = dict(np.load(str(path)))

    return weights, skeleton, adapters


def compute_perturbation(li, key, skeleton, adapters):
    """Compute composed perturbation Delta = (scale/N) * sum_i(A_i @ B_i)."""
    param_name = f"model.layers.{li}.{key}.lora_b"
    delta = None

    for di, domain in enumerate(DOMAIN_NAMES):
        skey = f"layer_{li}_{key}_domain_{di}"
        if skey not in skeleton:
            continue
        if param_name not in adapters[domain]:
            continue

        A_i = np.nan_to_num(skeleton[skey].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        B_i = np.nan_to_num(adapters[domain][param_name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        contribution = A_i @ B_i
        if delta is None:
            delta = contribution
        else:
            delta += contribution

    if delta is None:
        return None

    delta = (LORA_SCALE / N_DOMAINS) * delta
    return delta


def compute_persistence(points, max_dim=1):
    """Compute Rips persistence diagram on a point cloud."""
    import ripser
    result = ripser.ripser(points, maxdim=max_dim)
    return result['dgms']


def finite_diagram(dgm):
    """Filter to finite persistence features only."""
    return dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm


def persistence_values(dgm):
    """Get persistence (death - birth) for finite features."""
    fd = finite_diagram(dgm)
    if len(fd) == 0:
        return np.array([])
    return fd[:, 1] - fd[:, 0]


def bottleneck_distance(dgm1, dgm2):
    """Compute bottleneck distance between two persistence diagrams."""
    import persim
    d1 = finite_diagram(dgm1)
    d2 = finite_diagram(dgm2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        pv = persistence_values(d2)
        return float(np.max(pv)) / 2.0 if len(pv) > 0 else 0.0
    if len(d2) == 0:
        pv = persistence_values(d1)
        return float(np.max(pv)) / 2.0 if len(pv) > 0 else 0.0
    return persim.bottleneck(d1, d2)


def compute_bridge(delta_sub, rank_k):
    """Compute rank-k bridge matrix via truncated SVD of the perturbation.

    Bridge = -Delta_k (the best rank-k approximation of the negative perturbation).
    Adding this to the composed weights gives: W + Delta - Delta_k = W + (Delta - Delta_k)
    which is closer to W than W + Delta.
    """
    if rank_k <= 0 or delta_sub.shape[0] == 0:
        return np.zeros_like(delta_sub)

    effective_rank = min(rank_k, min(delta_sub.shape) - 1)
    if effective_rank <= 0:
        return np.zeros_like(delta_sub)

    # Truncated SVD
    U, S, Vt = np.linalg.svd(delta_sub, full_matrices=False)
    # Bridge = negative of rank-k approximation of delta
    bridge = -(U[:, :effective_rank] * S[:effective_rank]) @ Vt[:effective_rank, :]
    return bridge


def count_lost_features(dgm_base, dgm_composed, dim=1):
    """Count features present in base but absent from composed diagram.

    A feature is considered 'lost' if the bottleneck matching maps it to the
    diagonal (i.e., its matched persistence is 0 or near-0).

    Simple heuristic: count features in base that are NOT within tolerance of
    any feature in composed.
    """
    base_pers = persistence_values(dgm_base)
    comp_pers = persistence_values(dgm_composed)

    if len(base_pers) == 0:
        return 0, 0, 0

    # Features present in base but gone from composed
    # Sort both and compare counts as a first approximation
    n_base = len(base_pers)
    n_comp = len(comp_pers)

    # Net loss: more features in base than composed
    net_lost = max(0, n_base - n_comp)

    # Count features above median persistence that were lost
    if len(base_pers) > 0:
        median_pers = np.median(base_pers)
        n_above_median_base = np.sum(base_pers > median_pers)
        n_above_median_comp = np.sum(comp_pers > median_pers) if len(comp_pers) > 0 else 0
        lost_above_median = max(0, int(n_above_median_base - n_above_median_comp))
    else:
        lost_above_median = 0

    return net_lost, lost_above_median, n_base


def phase_bridge_extraction(weights, skeleton, adapters):
    """Main experiment: compute PH with and without bridge correction."""
    log("\nPhase: Bridge extraction and topological restoration")

    np.random.seed(42)
    results = {}

    modules_analyzed = 0
    modules_with_h1_loss = 0

    for li in TARGET_LAYERS:
        for key in TARGET_KEYS:
            if (li, key) not in weights:
                continue

            W = weights[(li, key)]
            n_rows, d = W.shape
            label = f"layer_{li}_{key}"

            # Compute perturbation
            delta = compute_perturbation(li, key, skeleton, adapters)
            if delta is None:
                continue

            W_composed = W + delta

            # Subsample rows deterministically
            if n_rows > N_SUBSAMPLE:
                indices = np.linspace(0, n_rows - 1, N_SUBSAMPLE, dtype=int)
            else:
                indices = np.arange(n_rows)

            W_sub = W[indices].astype(np.float32)
            W_comp_sub = W_composed[indices].astype(np.float32)
            delta_sub = delta[indices].astype(np.float32)

            # Row perturbation norms
            row_norms = np.linalg.norm(delta_sub, axis=1)
            max_delta_norm = float(np.max(row_norms))
            vuln_bound = 2 * max_delta_norm

            log(f"\n  {label}: shape={W.shape}, max_delta={max_delta_norm:.4f}")

            # Compute base and composed PH
            t0 = time.time()
            dgms_base = compute_persistence(W_sub, MAX_DIM)
            dgms_composed = compute_persistence(W_comp_sub, MAX_DIM)
            t_ph = time.time() - t0

            # H1 analysis
            base_h1_pers = persistence_values(dgms_base[1])
            comp_h1_pers = persistence_values(dgms_composed[1])

            median_h1 = float(np.median(base_h1_pers)) if len(base_h1_pers) > 0 else 0.0
            bn_h1_original = bottleneck_distance(dgms_base[1], dgms_composed[1])
            bn_h0_original = bottleneck_distance(dgms_base[0], dgms_composed[0])

            net_lost_h1, lost_above_med_h1, n_base_h1 = count_lost_features(
                dgms_base[1], dgms_composed[1], dim=1
            )
            n_comp_h1 = len(comp_h1_pers)

            log(f"    H1: base={n_base_h1}, composed={n_comp_h1}, net_lost={net_lost_h1}")
            log(f"    H1 bottleneck (original): {bn_h1_original:.6f}")
            log(f"    H1 median persistence: {median_h1:.4f}, vuln_bound: {vuln_bound:.4f}")
            log(f"    PH time: {t_ph:.2f}s")

            modules_analyzed += 1
            if net_lost_h1 > 0 or bn_h1_original > 0.001:
                modules_with_h1_loss += 1

            # Skip detailed bridge analysis for modules with few H1 features
            if n_base_h1 < MIN_H1_FOR_BRIDGE:
                log(f"    Skipping bridge analysis (only {n_base_h1} H1 features)")
                results[label] = {
                    "shape": list(W.shape),
                    "n_subsample": len(indices),
                    "max_delta_norm": max_delta_norm,
                    "vuln_bound": vuln_bound,
                    "median_h1_persistence": median_h1,
                    "h1_vulnerable": vuln_bound > median_h1 if median_h1 > 0 else False,
                    "bn_h1_original": float(bn_h1_original),
                    "bn_h0_original": float(bn_h0_original),
                    "n_base_h1": n_base_h1,
                    "n_composed_h1": n_comp_h1,
                    "net_lost_h1": net_lost_h1,
                    "lost_above_median_h1": lost_above_med_h1,
                    "sv_decay": [],
                    "total_perturbation_energy": 0,
                    "bridge_results": {},
                    "ph_time_s": t_ph,
                    "skipped_bridge": True,
                }
                del dgms_base, dgms_composed
                gc.collect()
                continue

            # Compute SVD of perturbation for rank analysis
            U, S, Vt = np.linalg.svd(delta_sub, full_matrices=False)
            # Energy captured at each rank
            total_energy = float(np.sum(S ** 2))
            sv_decay = [float(s) for s in S[:20]]  # Top 20 singular values

            # --- Bridge correction at various ranks ---
            bridge_results = {}  # str keys for JSON compat
            for rank_k in BRIDGE_RANKS:
                bridge = compute_bridge(delta_sub, rank_k)
                W_bridge_sub = W_comp_sub + bridge  # W + Delta + B = W + (Delta - Delta_k)

                # Compute PH on bridge-corrected
                dgms_bridge = compute_persistence(W_bridge_sub, MAX_DIM)

                bn_h1_bridge = bottleneck_distance(dgms_base[1], dgms_bridge[1])
                bn_h0_bridge = bottleneck_distance(dgms_base[0], dgms_bridge[0])

                # H1 feature restoration
                bridge_h1_pers = persistence_values(dgms_bridge[1])
                n_bridge_h1 = len(bridge_h1_pers)
                net_lost_h1_bridge, _, _ = count_lost_features(
                    dgms_base[1], dgms_bridge[1], dim=1
                )

                # Residual max row norm (the new vulnerability bound)
                residual = delta_sub + bridge  # Delta - Delta_k
                residual_max_norm = float(np.max(np.linalg.norm(residual, axis=1)))

                # Energy captured by this rank
                if total_energy > 0:
                    energy_captured = float(np.sum(S[:rank_k] ** 2) / total_energy)
                else:
                    energy_captured = 1.0

                # d_B reduction
                if bn_h1_original > 1e-10:
                    db_reduction = 1.0 - (bn_h1_bridge / bn_h1_original)
                else:
                    db_reduction = 0.0

                bridge_results[str(rank_k)] = {
                    "rank": rank_k,
                    "bn_h1_bridge": float(bn_h1_bridge),
                    "bn_h0_bridge": float(bn_h0_bridge),
                    "n_bridge_h1": n_bridge_h1,
                    "net_lost_h1_bridge": net_lost_h1_bridge,
                    "residual_max_row_norm": residual_max_norm,
                    "energy_captured": energy_captured,
                    "db_h1_reduction_pct": float(db_reduction * 100),
                }

                log(f"    rank-{rank_k}: bn_h1={bn_h1_bridge:.6f} "
                    f"(reduction={db_reduction*100:.1f}%), "
                    f"h1_features={n_bridge_h1}, "
                    f"energy={energy_captured*100:.1f}%")

                del dgms_bridge, bridge, W_bridge_sub
                gc.collect()

            results[label] = {
                "shape": list(W.shape),
                "n_subsample": len(indices),
                "max_delta_norm": max_delta_norm,
                "vuln_bound": vuln_bound,
                "median_h1_persistence": median_h1,
                "h1_vulnerable": vuln_bound > median_h1 if median_h1 > 0 else False,
                "bn_h1_original": float(bn_h1_original),
                "bn_h0_original": float(bn_h0_original),
                "n_base_h1": n_base_h1,
                "n_composed_h1": n_comp_h1,
                "net_lost_h1": net_lost_h1,
                "lost_above_median_h1": lost_above_med_h1,
                "sv_decay": sv_decay,
                "total_perturbation_energy": total_energy,
                "bridge_results": bridge_results,
                "ph_time_s": t_ph,
            }

            del dgms_base, dgms_composed, U, S, Vt
            gc.collect()

    log(f"\nModules analyzed: {modules_analyzed}")
    log(f"Modules with H1 change: {modules_with_h1_loss}")

    return results


def _load_model_with_premerge(bridge_rank=None):
    """Load model and apply pre-merged adapters, optionally with bridge correction.

    Args:
        bridge_rank: If None, apply full adapter perturbation.
                     If int, apply perturbation minus rank-k bridge correction
                     on TARGET_LAYERS.
    Returns:
        model, tokenizer
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_unflatten

    model, tokenizer = load(MODEL_ID)

    skel = dict(np.load(str(SKELETON_PATH)))
    adapter_data = {}
    for domain in DOMAIN_NAMES:
        path = ADAPTER_DIR / domain / "adapter.npz"
        adapter_data[domain] = dict(np.load(str(path)))

    for li in range(30):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            param_name = f"model.layers.{li}.{key}.lora_b"
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, BitLinear):
                continue

            delta_np = None
            for di, domain in enumerate(DOMAIN_NAMES):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skel or param_name not in adapter_data[domain]:
                    continue
                A_i = np.nan_to_num(skel[skey].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                B_i = np.nan_to_num(adapter_data[domain][param_name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                contribution = A_i @ B_i
                if delta_np is None:
                    delta_np = contribution
                else:
                    delta_np += contribution

            if delta_np is None:
                continue

            delta_np = (LORA_SCALE / N_DOMAINS) * delta_np

            # Apply bridge correction on target layers
            if bridge_rank is not None and li in TARGET_LAYERS:
                U, S, Vt = np.linalg.svd(delta_np, full_matrices=False)
                k = min(bridge_rank, min(delta_np.shape) - 1)
                if k > 0:
                    bridge_np = -(U[:, :k] * S[:k]) @ Vt[:k, :]
                    delta_np = delta_np + bridge_np

            # delta_np is (in_features, out_features) -- W^T space
            # Weight is (out_features, in_features), so delta for weight is delta^T
            delta_w = mx.array(delta_np.T.astype(np.float32))

            packed = module.weight
            out_f = module.out_features
            w0 = (packed & 3).astype(mx.bfloat16) - 1
            w1 = ((packed >> 2) & 3).astype(mx.bfloat16) - 1
            w2 = ((packed >> 4) & 3).astype(mx.bfloat16) - 1
            w3 = ((packed >> 6) & 3).astype(mx.bfloat16) - 1
            unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_f]
            ws = module.weight_scale.astype(mx.bfloat16)
            if module.invert_weight_scales:
                unpacked = unpacked / ws
            else:
                unpacked = unpacked * ws

            new_w = unpacked + delta_w.astype(mx.bfloat16)
            mx.eval(new_w)

            has_bias = module.bias is not None
            linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
            linear.weight = new_w
            if has_bias:
                linear.bias = module.bias

            layer.update_modules(tree_unflatten([(key, linear)]))

    mx.eval(model.parameters())
    return model, tokenizer


def phase_ppl_evaluation():
    """Evaluate PPL with and without bridge correction.

    Loads model twice: once with full composition, once with bridge-corrected.
    Base PPL is from cached results (Finding #44).
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    log("\nPhase: PPL evaluation with bridge correction")

    # Load validation data
    data_dir = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"
    if not data_dir.exists():
        data_dir = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

    val_texts = {}
    for domain in DOMAIN_NAMES:
        val_path = data_dir / domain / "valid.jsonl"
        if not val_path.exists():
            log(f"  WARNING: No validation data for {domain}")
            continue
        texts = []
        with open(val_path) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if len(text) > 20:
                    texts.append(text)
                if len(texts) >= PPL_BATCHES:
                    break
        val_texts[domain] = texts
        log(f"  Loaded {len(texts)} validation samples for {domain}")

    if not val_texts:
        log("  No validation data found, skipping PPL evaluation")
        return {}

    def compute_ppl(model, tokenizer, texts):
        total_loss = 0.0
        total_tokens = 0
        for text in texts:
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
            del logits, loss, x, y
        if total_tokens == 0:
            return float("inf")
        return math.exp(min(total_loss / total_tokens, 100))

    # Composed PPL (full adapter perturbation, no bridge)
    log("  Loading model with full composition...")
    model, tokenizer = _load_model_with_premerge(bridge_rank=None)
    composed_ppls = {}
    for domain, texts in val_texts.items():
        ppl = compute_ppl(model, tokenizer, texts)
        composed_ppls[domain] = ppl
        log(f"    Composed PPL ({domain}): {ppl:.2f}")
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    # Bridge-corrected PPL (rank-8 bridge on target layers)
    best_bridge_rank = 8
    log(f"  Loading model with rank-{best_bridge_rank} bridge correction...")
    model, tokenizer = _load_model_with_premerge(bridge_rank=best_bridge_rank)
    bridge_ppls = {}
    for domain, texts in val_texts.items():
        ppl = compute_ppl(model, tokenizer, texts)
        bridge_ppls[domain] = ppl
        log(f"    Bridge PPL ({domain}): {ppl:.2f}")
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    ppl_results = {
        "composed_ppls": composed_ppls,
        "bridge_ppls": bridge_ppls,
        "bridge_rank_used": best_bridge_rank,
    }

    # Compute improvement (positive = bridge is better = lower PPL)
    for domain in val_texts:
        if domain in composed_ppls and domain in bridge_ppls:
            comp = composed_ppls[domain]
            brdg = bridge_ppls[domain]
            if comp > 0:
                improvement = (comp - brdg) / comp * 100
                ppl_results[f"improvement_{domain}_pct"] = float(improvement)
                log(f"    {domain} PPL improvement: {improvement:.1f}%")

    return ppl_results


def phase_analyze(bridge_data, ppl_data):
    """Analyze results and assess kill criteria."""
    log("\nPhase: Analysis")

    # Aggregate across modules
    all_modules = list(bridge_data.keys())
    n_modules = len(all_modules)
    n_vulnerable = sum(1 for m in all_modules if bridge_data[m].get("h1_vulnerable", False))
    n_with_h1_loss = sum(1 for m in all_modules if bridge_data[m]["net_lost_h1"] > 0)
    total_h1_lost = sum(bridge_data[m]["net_lost_h1"] for m in all_modules)
    total_h1_base = sum(bridge_data[m]["n_base_h1"] for m in all_modules)

    log(f"\n  Modules analyzed: {n_modules}")
    log(f"  Modules with H1 vulnerability: {n_vulnerable}/{n_modules}")
    log(f"  Modules with net H1 loss: {n_with_h1_loss}/{n_modules}")
    log(f"  Total H1 features: base={total_h1_base}, lost={total_h1_lost}")

    # Per-rank analysis
    rank_summary = {}
    for rank_k in BRIDGE_RANKS:
        reductions = []
        for m in all_modules:
            br = bridge_data[m]["bridge_results"].get(str(rank_k), {})
            if br:
                reductions.append(br["db_h1_reduction_pct"])

        if reductions:
            mean_reduction = float(np.mean(reductions))
            median_reduction = float(np.median(reductions))
            max_reduction = float(np.max(reductions))
            n_pass = sum(1 for r in reductions if r >= 50)
        else:
            mean_reduction = 0
            median_reduction = 0
            max_reduction = 0
            n_pass = 0

        rank_summary[rank_k] = {
            "mean_db_reduction_pct": mean_reduction,
            "median_db_reduction_pct": median_reduction,
            "max_db_reduction_pct": max_reduction,
            "n_modules_pass_50pct": n_pass,
            "n_modules_total": len(reductions),
        }
        log(f"  rank-{rank_k}: mean_reduction={mean_reduction:.1f}%, "
            f"median={median_reduction:.1f}%, pass_50%={n_pass}/{len(reductions)}")

    # K628: Bridge reduces d_B by >= 50%
    # Use rank-16 (same cost as one adapter) as primary test
    k628_data = rank_summary.get(16, {})
    k628_mean = k628_data.get("mean_db_reduction_pct", 0)
    k628_pass = k628_mean >= 50

    # Find minimum rank that achieves >= 50% mean reduction
    min_rank_50 = None
    for rank_k in BRIDGE_RANKS:
        if rank_summary[rank_k]["mean_db_reduction_pct"] >= 50:
            min_rank_50 = rank_k
            break

    # K629: Bridge rank < r=16
    k629_pass = min_rank_50 is not None and min_rank_50 < LORA_RANK
    k629_detail = f"min_rank_for_50pct_reduction={min_rank_50}"

    # K630: PPL improvement >= 5%
    if ppl_data and "composed_ppls" in ppl_data and "bridge_ppls" in ppl_data:
        improvements = []
        for domain in DOMAIN_NAMES:
            key = f"improvement_{domain}_pct"
            if key in ppl_data:
                improvements.append(ppl_data[key])

        mean_ppl_improvement = float(np.mean(improvements)) if improvements else 0
        k630_pass = mean_ppl_improvement >= 5
        k630_detail = f"mean_PPL_improvement={mean_ppl_improvement:.1f}%"
    else:
        mean_ppl_improvement = 0
        k630_pass = False
        k630_detail = "PPL evaluation not completed"

    # SVD energy analysis
    sv_90_ranks = []
    for m in all_modules:
        sv = bridge_data[m]["sv_decay"]
        total_e = bridge_data[m]["total_perturbation_energy"]
        if total_e > 0 and len(sv) > 0:
            cumulative = np.cumsum(np.array(sv) ** 2) / total_e
            rank_90 = int(np.searchsorted(cumulative, 0.9) + 1)
            sv_90_ranks.append(rank_90)

    mean_rank_90 = float(np.mean(sv_90_ranks)) if sv_90_ranks else 0

    summary = {
        "n_modules": n_modules,
        "n_vulnerable_h1": n_vulnerable,
        "n_with_h1_loss": n_with_h1_loss,
        "total_h1_base": total_h1_base,
        "total_h1_lost": total_h1_lost,
        "rank_summary": rank_summary,
        "min_rank_for_50pct_reduction": min_rank_50,
        "mean_rank_for_90pct_energy": mean_rank_90,
        "k628_pass": k628_pass,
        "k628_detail": f"mean_db_reduction={k628_mean:.1f}% at rank 16",
        "k629_pass": k629_pass,
        "k629_detail": k629_detail,
        "k630_pass": k630_pass,
        "k630_detail": k630_detail,
        "ppl_data": ppl_data,
    }

    log("\n" + "=" * 60)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 60)
    log(f"  K628 (d_B reduction >= 50%): {'PASS' if k628_pass else 'FAIL'} - {summary['k628_detail']}")
    log(f"  K629 (bridge rank < r=16):   {'PASS' if k629_pass else 'FAIL'} - {summary['k629_detail']}")
    log(f"  K630 (PPL improvement >= 5%): {'PASS' if k630_pass else 'FAIL'} - {summary['k630_detail']}")

    return summary


def main():
    t0 = time.time()
    log("=" * 70)
    log("Persistence-Guided Bridge Extraction")
    log("=" * 70)

    # Phase 1: Load weights
    weights, skeleton, adapters = phase_load_weights()

    # Phase 2: Bridge extraction and topological analysis
    bridge_data = phase_bridge_extraction(weights, skeleton, adapters)

    # Free weight data before PPL evaluation
    del weights, skeleton, adapters
    gc.collect()

    # Phase 3: PPL evaluation
    ppl_data = phase_ppl_evaluation()

    # Phase 4: Analysis
    summary = phase_analyze(bridge_data, ppl_data)

    # Save results
    results = {
        "experiment": "persistence_bridge_extraction",
        "total_time_s": round(time.time() - t0, 1),
        "config": {
            "model": MODEL_ID,
            "n_subsample": N_SUBSAMPLE,
            "target_layers": TARGET_LAYERS,
            "n_domains": N_DOMAINS,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "bridge_ranks_tested": BRIDGE_RANKS,
        },
        "summary": summary,
        "per_module": bridge_data,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
