#!/usr/bin/env python3
"""Room Model Piece B: Geometric Routing via Adapter Output Norms.

Tests whether token hidden states naturally project onto adapter "walls"
(DeltaW_i = A_i @ B_i) to produce a domain-discriminative routing signal
WITHOUT any trained router.

Grounded by:
  - Room Model MATH.md Theorem 2 (projection geometry)
  - FlyLoRA (arXiv 2510.08396) -- frozen random A as implicit router
  - Finding #302: A-only routing gave 14% (near random)
  - Finding #310: Ridge router gave 98.3% token accuracy (upper bound)

Type: Guided Exploration (Type 2) -- proven decomposition framework,
unknown is whether B-matrix amplification makes geometric signal
domain-discriminative.

Kill criteria:
  K804: Geometric routing accuracy < 60% -> KILL
  K805: Agreement with ridge router < 50% -> KILL

Platform: Apple M5 Pro 48GB, MLX + numpy.
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

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Sources (reuse from prior experiments)
HIDDEN_STATES_PATH = EXPERIMENT_DIR.parent / "hidden_state_probe_router" / "hidden_states.npz"
RIDGE_STATS_PATH = EXPERIMENT_DIR.parent / "hidden_state_probe_router" / "ridge_stats.npz"
SKELETON_PATH = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters" / "grassmannian_skeleton.npz"
SFT_ADAPTER_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)
LORA_RANK = 16
LORA_SCALE = 20.0
D_MODEL = 2560

# Modules to use for geometric routing
# Focus on modules with d_in = d_model = 2560 (matching hidden state dim)
# This excludes down_proj (d_in = 6912) which needs MLP hidden states
ROUTING_MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
]
# down_proj has d_in=6912, cannot directly use d=2560 hidden states

ADAPTER_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

N_LAYERS = 30
SEED = 42


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
# Phase 1: Load pre-extracted hidden states and adapter geometry
# ============================================================================

def phase_load_data():
    """Load hidden states from Finding #310 and adapter A/B matrices."""
    log("\n" + "=" * 70)
    log("PHASE 1: LOAD PRE-EXTRACTED DATA")
    log("=" * 70)
    t0 = time.time()

    # Load hidden states (from hidden_state_probe_router experiment)
    log(f"  Loading hidden states from {HIDDEN_STATES_PATH}...")
    hs = np.load(str(HIDDEN_STATES_PATH))
    test_h = hs["test_token_h"]  # (N_test_tokens, 2560)
    test_l = hs["test_token_l"]  # (N_test_tokens,)
    seq_h = hs["test_seq_h"]     # (N_test_seqs, 2560) -- mean-pooled
    seq_l = hs["test_seq_l"]     # (N_test_seqs,)
    log(f"  Test tokens: {test_h.shape}, Test sequences: {seq_h.shape}")

    # Token distribution
    for di, domain in enumerate(DOMAINS):
        n = np.sum(test_l == di)
        log(f"  {domain}: {n} test tokens")

    # Load ridge router weights for comparison
    log(f"  Loading ridge stats from {RIDGE_STATS_PATH}...")
    ridge = np.load(str(RIDGE_STATS_PATH))
    W_ridge = ridge["W_ridge"]  # (2560, 5)
    mean_h = ridge["mean_h"]    # (1, 2560)
    std_h = ridge["std_h"]      # (1, 2560)
    log(f"  Ridge weights: {W_ridge.shape}")

    # Load Grassmannian A-matrices
    log(f"  Loading Grassmannian skeleton from {SKELETON_PATH}...")
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"  Skeleton keys: {len(skeleton)}")

    # Load adapter B-matrices
    log("  Loading adapter B-matrices...")
    adapter_Bs = {}
    for di, domain in enumerate(DOMAINS):
        path = SFT_ADAPTER_DIR / domain / "adapter.npz"
        adapter_Bs[di] = dict(mx.load(str(path)))
        # Convert to numpy for consistency
        adapter_Bs[di] = {k: np.array(v) for k, v in adapter_Bs[di].items()}
        log(f"    {domain}: {len(adapter_Bs[di])} B-matrices loaded")

    elapsed = time.time() - t0
    log(f"  Data loading complete in {elapsed:.1f}s")

    return {
        "test_h": test_h, "test_l": test_l,
        "seq_h": seq_h, "seq_l": seq_l,
        "W_ridge": W_ridge, "mean_h": mean_h, "std_h": std_h,
        "skeleton": skeleton, "adapter_Bs": adapter_Bs,
        "load_time_s": elapsed,
    }


# ============================================================================
# Phase 2: Compute geometric routing scores
# ============================================================================

def phase_geometric_routing(data):
    """Compute geometric routing using adapter output norms.

    Methods:
      1. A-only: ||h @ A_i|| (baseline from Finding #302, expect ~14%)
      2. DeltaW single-module: ||h @ (A_i @ B_i)|| for one module at one layer
      3. DeltaW per-layer: aggregate ||h @ (A_i @ B_i)|| across all d=2560 modules at one layer
      4. DeltaW multi-layer: aggregate across representative layers
      5. Ridge router (oracle comparison from Finding #310)
    """
    log("\n" + "=" * 70)
    log("PHASE 2: GEOMETRIC ROUTING SCORES")
    log("=" * 70)
    t0 = time.time()

    test_h = data["test_h"]  # (N, 2560)
    test_l = data["test_l"]  # (N,)
    seq_h = data["seq_h"]    # (M, 2560)
    seq_l = data["seq_l"]    # (M,)
    skeleton = data["skeleton"]
    adapter_Bs = data["adapter_Bs"]
    W_ridge = data["W_ridge"]
    mean_h = data["mean_h"]
    std_h = data["std_h"]

    N_tokens = test_h.shape[0]
    N_seqs = seq_h.shape[0]
    results = {}

    # ---- Method 1: A-only routing (baseline) ----
    log("\n  Method 1: A-only routing (||h @ A_i||, baseline from Finding #302)")
    # Use layer 14 (middle of model, same as room_model_poc)
    layer = 14
    a_only_scores = np.zeros((N_tokens, N_DOMAINS))

    for di in range(N_DOMAINS):
        norm_sq_sum = np.zeros(N_tokens)
        n_mods = 0
        for key in ROUTING_MODULES:
            ak = f"layer_{layer}_{key}_domain_{di}"
            if ak not in skeleton:
                continue
            A = skeleton[ak].astype(np.float32)  # (d_in, rank)
            if A.shape[0] != D_MODEL:
                continue
            # h @ A: (N, rank)
            proj = test_h @ A  # (N, rank)
            norm_sq_sum += np.sum(proj ** 2, axis=1)
            n_mods += 1
        if n_mods > 0:
            a_only_scores[:, di] = np.sqrt(norm_sq_sum / n_mods)

    a_only_preds = np.argmax(a_only_scores, axis=1)
    a_only_acc = np.mean(a_only_preds == test_l)
    log(f"    A-only token accuracy: {a_only_acc:.1%} ({np.sum(a_only_preds == test_l)}/{N_tokens})")
    results["a_only"] = {
        "accuracy": float(a_only_acc),
        "layer": layer,
        "n_modules": n_mods,
        "per_domain": {},
    }
    for di, domain in enumerate(DOMAINS):
        mask = test_l == di
        if mask.sum() > 0:
            acc = np.mean(a_only_preds[mask] == di)
            results["a_only"]["per_domain"][domain] = float(acc)
            log(f"      {domain}: {acc:.1%}")

    # ---- Method 2: DeltaW single-module routing ----
    log("\n  Method 2: DeltaW single-module routing (||h @ A_i @ B_i||)")
    # Use q_proj at layer 14 as representative
    best_single_acc = 0
    best_single_key = None
    single_module_results = {}

    for key in ROUTING_MODULES:
        dw_scores = np.zeros((N_tokens, N_DOMAINS))
        valid = True
        for di in range(N_DOMAINS):
            ak = f"layer_{layer}_{key}_domain_{di}"
            bk = f"model.layers.{layer}.{key}.lora_b"
            if ak not in skeleton or bk not in adapter_Bs[di]:
                valid = False
                break
            A = skeleton[ak].astype(np.float32)  # (d_in, rank)
            B = adapter_Bs[di][bk].astype(np.float32)  # (rank, d_out)
            if A.shape[0] != D_MODEL:
                valid = False
                break
            # h @ A @ B: (N, d_out)
            proj = test_h @ A  # (N, rank)
            out = proj @ B     # (N, d_out)
            dw_scores[:, di] = np.linalg.norm(out, axis=1)

        if not valid:
            continue

        preds = np.argmax(dw_scores, axis=1)
        acc = np.mean(preds == test_l)
        single_module_results[key] = float(acc)
        log(f"    {key}: {acc:.1%}")
        if acc > best_single_acc:
            best_single_acc = acc
            best_single_key = key

    results["dw_single_module"] = {
        "per_module_accuracy": single_module_results,
        "best_module": best_single_key,
        "best_accuracy": float(best_single_acc),
    }
    log(f"    Best single module: {best_single_key} at {best_single_acc:.1%}")

    # ---- Method 3: DeltaW per-layer aggregated routing ----
    log("\n  Method 3: DeltaW per-layer aggregated (sum of ||h @ A_i @ B_i|| across modules)")
    # Try multiple layers
    layer_results = {}
    best_layer_acc = 0
    best_layer = None

    for test_layer in [0, 7, 14, 21, 29]:
        agg_scores = np.zeros((N_tokens, N_DOMAINS))
        n_valid = 0

        for di in range(N_DOMAINS):
            total_norm = np.zeros(N_tokens)
            for key in ROUTING_MODULES:
                ak = f"layer_{test_layer}_{key}_domain_{di}"
                bk = f"model.layers.{test_layer}.{key}.lora_b"
                if ak not in skeleton or bk not in adapter_Bs[di]:
                    continue
                A = skeleton[ak].astype(np.float32)
                B = adapter_Bs[di][bk].astype(np.float32)
                if A.shape[0] != D_MODEL:
                    continue
                proj = test_h @ A
                out = proj @ B
                total_norm += np.linalg.norm(out, axis=1)
                if di == 0:
                    n_valid += 1
            agg_scores[:, di] = total_norm

        preds = np.argmax(agg_scores, axis=1)
        acc = np.mean(preds == test_l)
        layer_results[test_layer] = float(acc)
        log(f"    Layer {test_layer}: {acc:.1%} ({n_valid} modules)")
        if acc > best_layer_acc:
            best_layer_acc = acc
            best_layer = test_layer

    results["dw_per_layer"] = {
        "per_layer_accuracy": layer_results,
        "best_layer": best_layer,
        "best_accuracy": float(best_layer_acc),
    }
    log(f"    Best layer: {best_layer} at {best_layer_acc:.1%}")

    # ---- Method 4: DeltaW multi-layer aggregated routing ----
    log("\n  Method 4: DeltaW multi-layer aggregated (all layers)")
    multi_scores = np.zeros((N_tokens, N_DOMAINS))

    for li in range(N_LAYERS):
        for di in range(N_DOMAINS):
            for key in ROUTING_MODULES:
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"
                if ak not in skeleton or bk not in adapter_Bs[di]:
                    continue
                A = skeleton[ak].astype(np.float32)
                B = adapter_Bs[di][bk].astype(np.float32)
                if A.shape[0] != D_MODEL:
                    continue
                proj = test_h @ A
                out = proj @ B
                multi_scores[:, di] += np.linalg.norm(out, axis=1)

    multi_preds = np.argmax(multi_scores, axis=1)
    multi_acc = np.mean(multi_preds == test_l)
    log(f"    Multi-layer token accuracy: {multi_acc:.1%}")
    results["dw_multi_layer"] = {
        "accuracy": float(multi_acc),
        "per_domain": {},
    }
    for di, domain in enumerate(DOMAINS):
        mask = test_l == di
        if mask.sum() > 0:
            acc = np.mean(multi_preds[mask] == di)
            results["dw_multi_layer"]["per_domain"][domain] = float(acc)
            log(f"      {domain}: {acc:.1%}")

    # ---- Method 5: Ridge router (oracle comparison) ----
    log("\n  Method 5: Ridge router (from Finding #310, 98.3% accuracy)")
    # Normalize hidden states same as training
    std_safe = std_h.copy()
    std_safe[std_safe == 0] = 1.0
    test_h_norm = (test_h - mean_h) / std_safe
    ridge_logits = test_h_norm @ W_ridge  # (N, 5)
    ridge_preds = np.argmax(ridge_logits, axis=1)
    ridge_acc = np.mean(ridge_preds == test_l)
    log(f"    Ridge token accuracy: {ridge_acc:.1%}")
    results["ridge"] = {
        "accuracy": float(ridge_acc),
        "per_domain": {},
    }
    for di, domain in enumerate(DOMAINS):
        mask = test_l == di
        if mask.sum() > 0:
            acc = np.mean(ridge_preds[mask] == di)
            results["ridge"]["per_domain"][domain] = float(acc)
            log(f"      {domain}: {acc:.1%}")

    # ---- Method 6: Sequence-level geometric routing ----
    log("\n  Method 6: Sequence-level geometric routing (mean-pooled h)")

    # A-only sequence-level
    a_only_seq_scores = np.zeros((N_seqs, N_DOMAINS))
    for di in range(N_DOMAINS):
        norm_sq_sum = np.zeros(N_seqs)
        n_mods = 0
        for key in ROUTING_MODULES:
            ak = f"layer_{layer}_{key}_domain_{di}"
            if ak not in skeleton:
                continue
            A = skeleton[ak].astype(np.float32)
            if A.shape[0] != D_MODEL:
                continue
            proj = seq_h @ A
            norm_sq_sum += np.sum(proj ** 2, axis=1)
            n_mods += 1
        if n_mods > 0:
            a_only_seq_scores[:, di] = np.sqrt(norm_sq_sum / n_mods)
    a_only_seq_preds = np.argmax(a_only_seq_scores, axis=1)
    a_only_seq_acc = np.mean(a_only_seq_preds == seq_l)
    log(f"    A-only sequence accuracy: {a_only_seq_acc:.1%}")

    # DeltaW multi-layer sequence-level
    dw_seq_scores = np.zeros((N_seqs, N_DOMAINS))
    for li in range(N_LAYERS):
        for di in range(N_DOMAINS):
            for key in ROUTING_MODULES:
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"
                if ak not in skeleton or bk not in adapter_Bs[di]:
                    continue
                A = skeleton[ak].astype(np.float32)
                B = adapter_Bs[di][bk].astype(np.float32)
                if A.shape[0] != D_MODEL:
                    continue
                proj = seq_h @ A
                out = proj @ B
                dw_seq_scores[:, di] += np.linalg.norm(out, axis=1)
    dw_seq_preds = np.argmax(dw_seq_scores, axis=1)
    dw_seq_acc = np.mean(dw_seq_preds == seq_l)
    log(f"    DeltaW multi-layer sequence accuracy: {dw_seq_acc:.1%}")

    # Ridge sequence-level
    seq_h_norm = (seq_h - mean_h) / std_safe
    ridge_seq_logits = seq_h_norm @ W_ridge
    ridge_seq_preds = np.argmax(ridge_seq_logits, axis=1)
    ridge_seq_acc = np.mean(ridge_seq_preds == seq_l)
    log(f"    Ridge sequence accuracy: {ridge_seq_acc:.1%}")

    results["sequence_level"] = {
        "a_only_accuracy": float(a_only_seq_acc),
        "dw_multi_layer_accuracy": float(dw_seq_acc),
        "ridge_accuracy": float(ridge_seq_acc),
    }

    elapsed = time.time() - t0
    results["routing_time_s"] = elapsed
    log(f"\n  Routing computation complete in {elapsed:.1f}s")

    return results, multi_preds, ridge_preds


# ============================================================================
# Phase 3: Agreement analysis between geometric and ridge router
# ============================================================================

def phase_agreement_analysis(data, geo_preds, ridge_preds):
    """Analyze agreement between geometric routing and ridge router."""
    log("\n" + "=" * 70)
    log("PHASE 3: AGREEMENT ANALYSIS")
    log("=" * 70)

    test_l = data["test_l"]
    N = len(test_l)

    # Overall agreement
    agree_mask = geo_preds == ridge_preds
    agreement = np.mean(agree_mask)
    log(f"  Overall agreement (geo vs ridge): {agreement:.1%}")

    # Per-domain agreement
    per_domain_agreement = {}
    for di, domain in enumerate(DOMAINS):
        mask = test_l == di
        if mask.sum() > 0:
            domain_agree = np.mean(agree_mask[mask])
            per_domain_agreement[domain] = float(domain_agree)
            log(f"    {domain}: {domain_agree:.1%}")

    # Confusion analysis: when they disagree, who is right?
    disagree_mask = ~agree_mask
    n_disagree = disagree_mask.sum()
    if n_disagree > 0:
        geo_right = np.mean(geo_preds[disagree_mask] == test_l[disagree_mask])
        ridge_right = np.mean(ridge_preds[disagree_mask] == test_l[disagree_mask])
        neither_right = np.mean(
            (geo_preds[disagree_mask] != test_l[disagree_mask]) &
            (ridge_preds[disagree_mask] != test_l[disagree_mask])
        )
        log(f"\n  Disagreements: {n_disagree}/{N} ({n_disagree/N:.1%})")
        log(f"    Geo right, ridge wrong: {geo_right:.1%}")
        log(f"    Ridge right, geo wrong: {ridge_right:.1%}")
        log(f"    Both wrong: {neither_right:.1%}")
    else:
        geo_right = 0.0
        ridge_right = 0.0
        neither_right = 0.0
        log("  No disagreements!")

    # Confusion matrix for geometric router
    log("\n  Confusion matrix (geometric router):")
    confusion = np.zeros((N_DOMAINS, N_DOMAINS), dtype=int)
    for true_d, pred_d in zip(test_l, geo_preds):
        confusion[true_d, pred_d] += 1
    log(f"    {'':>10s} " + " ".join(f"{d:>8s}" for d in DOMAINS))
    for di, domain in enumerate(DOMAINS):
        row = " ".join(f"{confusion[di, dj]:>8d}" for dj in range(N_DOMAINS))
        log(f"    {domain:>10s} {row}")

    return {
        "overall_agreement": float(agreement),
        "per_domain_agreement": per_domain_agreement,
        "n_disagreements": int(n_disagree),
        "disagree_geo_right_pct": float(geo_right) if n_disagree > 0 else None,
        "disagree_ridge_right_pct": float(ridge_right) if n_disagree > 0 else None,
        "disagree_neither_right_pct": float(neither_right) if n_disagree > 0 else None,
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================================
# Phase 4: Score distribution analysis
# ============================================================================

def phase_score_analysis(data):
    """Analyze the discriminability of geometric scores."""
    log("\n" + "=" * 70)
    log("PHASE 4: SCORE DISTRIBUTION ANALYSIS")
    log("=" * 70)

    test_h = data["test_h"]
    test_l = data["test_l"]
    skeleton = data["skeleton"]
    adapter_Bs = data["adapter_Bs"]
    N_tokens = test_h.shape[0]

    # Compute multi-layer scores for analysis
    scores = np.zeros((N_tokens, N_DOMAINS))
    for li in range(N_LAYERS):
        for di in range(N_DOMAINS):
            for key in ROUTING_MODULES:
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"
                if ak not in skeleton or bk not in adapter_Bs[di]:
                    continue
                A = skeleton[ak].astype(np.float32)
                B = adapter_Bs[di][bk].astype(np.float32)
                if A.shape[0] != D_MODEL:
                    continue
                proj = test_h @ A
                out = proj @ B
                scores[:, di] += np.linalg.norm(out, axis=1)

    # Normalize per token to get soft routing weights
    score_sum = scores.sum(axis=1, keepdims=True)
    score_sum[score_sum == 0] = 1.0
    weights = scores / score_sum

    # Mean weight on correct vs other domains
    correct_weights = []
    other_weights = []
    margin_ratios = []

    for ti in range(N_tokens):
        true_d = test_l[ti]
        w_true = weights[ti, true_d]
        w_others = np.delete(weights[ti], true_d)
        correct_weights.append(w_true)
        other_weights.append(np.mean(w_others))

        # Margin: correct weight / max other weight
        max_other = np.max(w_others)
        if max_other > 0:
            margin_ratios.append(w_true / max_other)
        else:
            margin_ratios.append(float('inf'))

    mean_correct = np.mean(correct_weights)
    mean_other = np.mean(other_weights)
    mean_margin = np.mean([r for r in margin_ratios if r != float('inf')])
    pct_margin_gt1 = np.mean([r > 1 for r in margin_ratios])

    log(f"  Mean weight on correct domain: {mean_correct:.4f}")
    log(f"  Mean weight on other domains:  {mean_other:.4f}")
    log(f"  Mean margin ratio (correct/max_other): {mean_margin:.3f}")
    log(f"  Pct tokens with correct > max_other: {pct_margin_gt1:.1%}")

    # Per-domain analysis
    per_domain_stats = {}
    for di, domain in enumerate(DOMAINS):
        mask = test_l == di
        if mask.sum() == 0:
            continue
        domain_scores = scores[mask]
        domain_weights = weights[mask]
        mean_w_correct = np.mean(domain_weights[:, di])
        mean_w_others = np.mean(np.delete(domain_weights, di, axis=1))

        # Which domain does geometric routing most confuse this with?
        other_scores = np.delete(domain_scores, di, axis=1)
        other_domains = [d for j, d in enumerate(DOMAINS) if j != di]
        top_confuser_idx = np.argmax(np.mean(other_scores, axis=0))
        top_confuser = other_domains[top_confuser_idx]

        per_domain_stats[domain] = {
            "mean_weight_correct": float(mean_w_correct),
            "mean_weight_others": float(mean_w_others),
            "top_confuser": top_confuser,
        }
        log(f"  {domain}: correct_w={mean_w_correct:.4f}, other_w={mean_w_others:.4f}, confuses_with={top_confuser}")

    # B-matrix norm analysis -- check if some adapters have larger B norms
    log("\n  B-matrix norm analysis (per domain, averaged across modules):")
    b_norms = {}
    for di, domain in enumerate(DOMAINS):
        total_norm = 0
        n_mods = 0
        for key in adapter_Bs[di]:
            B = adapter_Bs[di][key].astype(np.float32)
            total_norm += np.linalg.norm(B)
            n_mods += 1
        mean_norm = total_norm / max(n_mods, 1)
        b_norms[domain] = float(mean_norm)
        log(f"    {domain}: mean ||B|| = {mean_norm:.4f} ({n_mods} modules)")

    return {
        "mean_correct_weight": float(mean_correct),
        "mean_other_weight": float(mean_other),
        "mean_margin_ratio": float(mean_margin),
        "pct_margin_gt1": float(pct_margin_gt1),
        "per_domain": per_domain_stats,
        "b_norms": b_norms,
    }


# ============================================================================
# Phase 5: Latency benchmark
# ============================================================================

def phase_latency(data):
    """Benchmark geometric routing latency."""
    log("\n" + "=" * 70)
    log("PHASE 5: LATENCY BENCHMARK")
    log("=" * 70)

    # For latency, we simulate routing a single token through all layers
    # using MLX for fair comparison with the ridge router latency (0.17ms)
    skeleton = data["skeleton"]
    adapter_Bs = data["adapter_Bs"]

    # Pick a single test hidden state
    h = mx.array(data["test_h"][:1])  # (1, 2560)
    mx.eval(h)

    # Precompute DeltaW for one representative layer (layer 14)
    # In production, we would precompute DeltaW = A @ B once at adapter load time
    layer = 14
    deltaWs = []  # list of (N_DOMAINS,) tuples of DeltaW per module
    for key in ROUTING_MODULES:
        module_dws = []
        for di in range(N_DOMAINS):
            ak = f"layer_{layer}_{key}_domain_{di}"
            bk = f"model.layers.{layer}.{key}.lora_b"
            if ak not in skeleton or bk not in adapter_Bs[di]:
                module_dws.append(None)
                continue
            A = skeleton[ak].astype(np.float32)
            B = adapter_Bs[di][bk].astype(np.float32)
            if A.shape[0] != D_MODEL:
                module_dws.append(None)
                continue
            DW = mx.array(A @ B)  # (d_in, d_out) -- precomputed
            module_dws.append(DW)
        if all(dw is not None for dw in module_dws):
            deltaWs.append(module_dws)
    mx.eval([dw for module in deltaWs for dw in module])

    # Warm up
    for _ in range(10):
        scores = mx.zeros(N_DOMAINS)
        for module_dws in deltaWs:
            for di, dw in enumerate(module_dws):
                out = h @ dw
                scores = scores.at[di].add(mx.sqrt(mx.sum(out * out)))
        mx.eval(scores)

    # Benchmark single token
    N_iters = 100
    t0 = time.time()
    for _ in range(N_iters):
        scores = mx.zeros(N_DOMAINS)
        for module_dws in deltaWs:
            for di, dw in enumerate(module_dws):
                out = h @ dw
                scores = scores.at[di].add(mx.sqrt(mx.sum(out * out)))
        mx.eval(scores)
    single_ms = (time.time() - t0) / N_iters * 1000

    # Benchmark batch of 128 tokens
    h_batch = mx.array(data["test_h"][:128])
    mx.eval(h_batch)

    for _ in range(5):
        scores = mx.zeros((128, N_DOMAINS))
        for module_dws in deltaWs:
            for di, dw in enumerate(module_dws):
                out = h_batch @ dw
                norms = mx.sqrt(mx.sum(out * out, axis=-1))
                scores = scores.at[:, di].add(norms)
        mx.eval(scores)

    N_batch_iters = 50
    t0 = time.time()
    for _ in range(N_batch_iters):
        scores = mx.zeros((128, N_DOMAINS))
        for module_dws in deltaWs:
            for di, dw in enumerate(module_dws):
                out = h_batch @ dw
                norms = mx.sqrt(mx.sum(out * out, axis=-1))
                scores = scores.at[:, di].add(norms)
        mx.eval(scores)
    batch_total_ms = (time.time() - t0) / N_batch_iters * 1000
    batch_per_token_ms = batch_total_ms / 128

    log(f"  Single token: {single_ms:.3f}ms")
    log(f"  Batch 128 total: {batch_total_ms:.3f}ms")
    log(f"  Batch 128 per token: {batch_per_token_ms:.4f}ms")
    log(f"  Ridge router comparison: 0.170ms single, 0.0014ms batch/token")

    return {
        "single_token_ms": float(single_ms),
        "batch_128_total_ms": float(batch_total_ms),
        "batch_128_per_token_ms": float(batch_per_token_ms),
        "n_modules_used": len(deltaWs),
        "layer_used": layer,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")
    np.random.seed(SEED)

    # Phase 1: Load data
    data = phase_load_data()
    log_memory("after-load")

    # Phase 2: Geometric routing
    routing_results, geo_preds, ridge_preds = phase_geometric_routing(data)
    log_memory("after-routing")

    # Phase 3: Agreement analysis
    agreement_results = phase_agreement_analysis(data, geo_preds, ridge_preds)

    # Phase 4: Score distribution analysis
    score_results = phase_score_analysis(data)
    log_memory("after-scores")

    # Phase 5: Latency
    latency_results = phase_latency(data)
    log_memory("after-latency")

    # ---- Kill criteria assessment ----
    # K804: Best geometric routing accuracy >= 60%
    best_geo_acc = max(
        routing_results.get("dw_multi_layer", {}).get("accuracy", 0),
        routing_results.get("dw_per_layer", {}).get("best_accuracy", 0),
        routing_results.get("dw_single_module", {}).get("best_accuracy", 0),
    )

    # K805: Agreement with ridge router >= 50%
    agreement_pct = agreement_results["overall_agreement"]

    k804_pass = best_geo_acc >= 0.60
    k805_pass = agreement_pct >= 0.50

    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)
    log(f"  K804: Geometric routing accuracy >= 60%: {'PASS' if k804_pass else 'FAIL'} ({best_geo_acc:.1%})")
    log(f"  K805: Agreement with ridge router >= 50%: {'PASS' if k805_pass else 'FAIL'} ({agreement_pct:.1%})")
    log(f"  Verdict: {'SUPPORTED' if k804_pass and k805_pass else 'KILLED'}")

    # Compare with Finding #302 (A-only was 14%)
    a_only_acc = routing_results["a_only"]["accuracy"]
    improvement_over_a = best_geo_acc / max(a_only_acc, 0.01)
    log(f"\n  A-only routing (Finding #302 baseline): {a_only_acc:.1%}")
    log(f"  Best DeltaW routing: {best_geo_acc:.1%}")
    log(f"  Improvement over A-only: {improvement_over_a:.1f}x")
    log(f"  Ridge router (Finding #310): {routing_results['ridge']['accuracy']:.1%}")

    total_time = round(time.time() - t0, 1)

    # ---- Save results ----
    results = {
        "experiment": "exp_room_model_geometric_routing",
        "model": "microsoft/BitNet-b1.58-2B-4T",
        "domains": DOMAINS,
        "n_domains": N_DOMAINS,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "routing_modules": ROUTING_MODULES,
        "routing": routing_results,
        "agreement": agreement_results,
        "score_analysis": score_results,
        "latency": latency_results,
        "kill_criteria": {
            "K804": {
                "pass": k804_pass,
                "value": float(best_geo_acc),
                "threshold": 0.60,
                "detail": f"Best geometric routing accuracy = {best_geo_acc:.1%} (threshold 60%)",
            },
            "K805": {
                "pass": k805_pass,
                "value": float(agreement_pct),
                "threshold": 0.50,
                "detail": f"Agreement with ridge router = {agreement_pct:.1%} (threshold 50%)",
            },
        },
        "comparisons": {
            "a_only_accuracy": float(a_only_acc),
            "best_dw_accuracy": float(best_geo_acc),
            "improvement_over_a_only": float(improvement_over_a),
            "ridge_accuracy": float(routing_results["ridge"]["accuracy"]),
        },
        "verdict": "SUPPORTED" if k804_pass and k805_pass else "KILLED",
        "total_time_s": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total time: {total_time}s")


if __name__ == "__main__":
    main()
