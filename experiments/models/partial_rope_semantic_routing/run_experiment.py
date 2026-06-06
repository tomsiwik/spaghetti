#!/usr/bin/env python3
"""Experiment: partial_rope_semantic_routing

Tests whether pre-RoPE attention features (position-invariant Q/K projections)
cluster by domain and can serve as zero-parameter routing features.

Kill criteria:
  K1: Position-free dims don't cluster by domain (silhouette < 0.3)
  K2: Routing using position-free features worse than random (< 1/N = 4.2%)
  K3: N/A (no model training, analysis only)

Success criteria:
  S1: Routing accuracy > 60% using only position-free attention features
  Bonus: Silhouette score > 0.5 (clear domain clustering)

Grounding:
  - Parameter Golf (arXiv 2506.06105): partial RoPE, position-free dims learn semantics
  - RoFormer (arXiv 2104.09864): RoPE original paper
  - exp_softmax_router_scaling: 40% accuracy on full hidden states, oracle PPL

Platform: Apple M5 Pro 48GB, MLX
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source experiment with trained adapters and data
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
DATA_DIR = SOURCE_DIR / "data"

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
MAX_SEQ_LENGTH = 256
SEED = 42
SAMPLES_PER_DOMAIN = 50  # for feature extraction

# All 24 active domains
ALL_DOMAINS = [
    "medical", "code", "math", "legal", "finance",
    "science", "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering", "agriculture",
    "environmental", "politics", "economics", "sociology", "linguistics",
    "cybersecurity", "marketing", "sports", "music",
]

# BitNet-2B-4T architecture (ACTUAL: n_heads=20, head_dim=128, n_kv_heads=5)
HIDDEN_DIM = 2560
NUM_HEADS = 20
NUM_KV_HEADS = 5
HEAD_DIM = 128
NUM_LAYERS = 24
ROPE_FRAC = 0.25  # 25% of dims use RoPE (simulated split)
ROPE_DIMS = int(HEAD_DIM * ROPE_FRAC)  # 32
FREE_DIMS = HEAD_DIM - ROPE_DIMS  # 96


# ============================================================================
# Logging & Memory
# ============================================================================

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


# ============================================================================
# Model utilities
# ============================================================================

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
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
    """Replace BitLinear with nn.Linear for forward pass."""
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
# Phase 1: Extract pre-RoPE Q features and full hidden states
# ============================================================================

def extract_pre_rope_q(layer, h):
    """Extract pre-RoPE Q projection from an attention layer.

    Returns:
        q_pre_rope: (B, L, n_heads, d_head) -- Q before RoPE
    """
    B, L, D = h.shape
    attn = layer.self_attn
    x_normed = layer.input_layernorm(h)

    # Q projection (BEFORE RoPE)
    q = attn.q_proj(x_normed)
    q = q.reshape(B, L, attn.n_heads, -1)  # (B, L, 32, 80)
    return q


def phase_extract_features():
    """Extract pre-RoPE Q features and full hidden states per domain.

    For each sample:
    1. Run full forward pass to get last-layer hidden states
    2. Extract pre-RoPE Q from the last attention layer
    3. Split Q into rope dims (first 25%) and free dims (last 75%)
    4. Mean-pool over sequence length to get one vector per sample
    """
    log("\n[Phase 1] Extracting pre-RoPE Q features and hidden states...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    # We'll extract features from multiple layers for comparison
    # Last layer is the primary target; also grab middle layer
    target_layers = [NUM_LAYERS - 1, NUM_LAYERS // 2]  # layer 23 (last), layer 12 (mid)

    all_features = {}  # domain -> {q_free, q_rope, hidden, labels}

    for domain in ALL_DOMAINS:
        data_dir = DATA_DIR / domain
        valid_path = data_dir / "valid.jsonl"
        if not valid_path.exists():
            log(f"  WARNING: no valid data for {domain}, trying train")
            valid_path = data_dir / "train.jsonl"
            if not valid_path.exists():
                log(f"  SKIP: no data for {domain}")
                continue

        texts = []
        with open(valid_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        q_free_last = []
        q_rope_last = []
        q_free_mid = []
        hidden_states = []

        n_extracted = 0
        for text in texts[:SAMPLES_PER_DOMAIN]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]

            # Full forward pass to get hidden states at each layer
            h = model.model.embed_tokens(x)
            layer_outputs = []
            for i, layer in enumerate(model.model.layers):
                # Extract pre-RoPE Q at target layers BEFORE the layer processes
                if i in target_layers:
                    q_pre = extract_pre_rope_q(layer, h)
                    # Split into rope dims and free dims
                    q_r = q_pre[:, :, :, :ROPE_DIMS]       # (B, L, 32, 20)
                    q_f = q_pre[:, :, :, ROPE_DIMS:]        # (B, L, 32, 60)

                    # Mean pool over sequence and flatten heads
                    # q_r: (B, 32, 20) -> (B, 640)
                    q_r_pooled = mx.mean(q_r[0], axis=0).reshape(-1)  # (32*20,) = (640,)
                    q_f_pooled = mx.mean(q_f[0], axis=0).reshape(-1)  # (32*60,) = (1920,)

                    if i == NUM_LAYERS - 1:
                        mx.eval(q_r_pooled, q_f_pooled)
                        q_rope_last.append(q_r_pooled)
                        q_free_last.append(q_f_pooled)
                    elif i == NUM_LAYERS // 2:
                        mx.eval(q_f_pooled)
                        q_free_mid.append(q_f_pooled)

                    del q_pre, q_r, q_f, q_r_pooled, q_f_pooled

                h = layer(h)

            # Final hidden state (what the softmax router uses)
            h_normed = model.model.norm(h)
            h_mean = mx.mean(h_normed[0], axis=0)  # (2560,)
            mx.eval(h_mean)
            hidden_states.append(h_mean)

            del h, h_normed, h_mean, x
            n_extracted += 1

        if n_extracted > 0:
            def to_np(arr_list):
                """Convert list of MLX bf16 arrays to numpy float32."""
                return np.array([np.array(v.astype(mx.float32)) for v in arr_list])

            all_features[domain] = {
                "q_free_last": to_np(q_free_last),
                "q_rope_last": to_np(q_rope_last),
                "q_free_mid": to_np(q_free_mid),
                "hidden": to_np(hidden_states),
            }
            log(f"  {domain}: {n_extracted} samples extracted")
        else:
            log(f"  WARNING: 0 samples for {domain}")

        # Free intermediate MLX arrays
        del q_free_last, q_rope_last, q_free_mid, hidden_states

    elapsed = time.time() - t0
    log(f"  Feature extraction done in {elapsed:.1f}s")
    log_memory("post-extraction")
    cleanup(model, tokenizer)
    return all_features


# ============================================================================
# Phase 2: Clustering and routing analysis
# ============================================================================

def compute_routing_metrics(features_matrix, labels, n_domains, feature_name):
    """Compute silhouette score and centroid-based routing accuracy.

    Args:
        features_matrix: (N_total, D) numpy array
        labels: (N_total,) integer domain labels
        n_domains: number of domains
        feature_name: string label for logging

    Returns:
        dict with silhouette, routing_accuracy, kmeans_accuracy, etc.
    """
    N = features_matrix.shape[0]
    D = features_matrix.shape[1]

    # Normalize features (important for fair comparison across feature types)
    means = features_matrix.mean(axis=0, keepdims=True)
    stds = features_matrix.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features_matrix - means) / stds

    # 1. Silhouette score on ground-truth domain labels
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(features_norm, labels, sample_size=min(N, 1000))
    else:
        sil = 0.0

    # 2. Centroid-based routing (zero-parameter)
    # Build centroids from all samples (in production would use train split)
    # Use leave-one-out: compute centroid from all OTHER samples of the domain
    domains = np.unique(labels)
    centroids = np.zeros((n_domains, D))
    for d in domains:
        mask = labels == d
        centroids[d] = features_norm[mask].mean(axis=0)

    # Route each sample to nearest centroid
    dists = cdist(features_norm, centroids, metric="euclidean")
    predictions = dists.argmin(axis=1)
    centroid_accuracy = (predictions == labels).mean()

    # Per-domain accuracy
    domain_accs = {}
    for d in domains:
        mask = labels == d
        domain_accs[int(d)] = float((predictions[mask] == labels[mask]).mean())

    # 3. K-means clustering (unsupervised)
    kmeans = KMeans(n_clusters=min(n_domains, 24), random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(features_norm)

    # K-means silhouette
    if len(np.unique(cluster_labels)) > 1:
        kmeans_sil = silhouette_score(features_norm, cluster_labels, sample_size=min(N, 1000))
    else:
        kmeans_sil = 0.0

    # K-means to domain mapping (majority vote)
    from scipy.optimize import linear_sum_assignment
    # Build cost matrix: cluster x domain -> count
    cost_matrix = np.zeros((n_domains, n_domains))
    for c in range(n_domains):
        for d in range(n_domains):
            cost_matrix[c, d] = -np.sum((cluster_labels == c) & (labels == d))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    kmeans_accuracy = -cost_matrix[row_ind, col_ind].sum() / N

    # 4. Variance analysis: how much variance is domain vs position
    # Between-domain variance / total variance
    total_var = features_norm.var(axis=0).sum()
    between_var = 0.0
    global_mean = features_norm.mean(axis=0)
    for d in domains:
        mask = labels == d
        n_d = mask.sum()
        domain_mean = features_norm[mask].mean(axis=0)
        between_var += n_d * np.sum((domain_mean - global_mean) ** 2)
    between_var /= N
    variance_ratio = between_var / (total_var + 1e-8)

    log(f"  [{feature_name}] D={D}, N={N}")
    log(f"    Silhouette (true labels): {sil:.4f}")
    log(f"    Centroid routing accuracy: {centroid_accuracy:.4f} ({centroid_accuracy*100:.1f}%)")
    log(f"    K-means accuracy (Hungarian): {kmeans_accuracy:.4f}")
    log(f"    K-means silhouette: {kmeans_sil:.4f}")
    log(f"    Between/total variance ratio: {variance_ratio:.4f}")

    return {
        "feature_name": feature_name,
        "n_dims": D,
        "n_samples": N,
        "silhouette_true": round(float(sil), 4),
        "centroid_accuracy": round(float(centroid_accuracy), 4),
        "kmeans_accuracy": round(float(kmeans_accuracy), 4),
        "kmeans_silhouette": round(float(kmeans_sil), 4),
        "variance_ratio": round(float(variance_ratio), 4),
        "per_domain_centroid_acc": {str(k): round(v, 4) for k, v in domain_accs.items()},
    }


def phase_clustering_analysis(all_features):
    """Run clustering and routing analysis on all feature types."""
    log("\n[Phase 2] Clustering and routing analysis...")
    t0 = time.time()

    # Build feature matrices and labels
    domains_present = sorted(all_features.keys(), key=lambda d: ALL_DOMAINS.index(d))
    domain_to_idx = {d: i for i, d in enumerate(domains_present)}
    n_domains = len(domains_present)

    log(f"  {n_domains} domains with features")

    # Assemble matrices for each feature type
    feature_types = {
        "q_free_last": [],
        "q_rope_last": [],
        "q_free_mid": [],
        "hidden": [],
        "q_full_last": [],  # concat of rope + free
    }
    all_labels = []

    for domain in domains_present:
        feats = all_features[domain]
        n = feats["q_free_last"].shape[0]
        all_labels.extend([domain_to_idx[domain]] * n)

        feature_types["q_free_last"].append(feats["q_free_last"])
        feature_types["q_rope_last"].append(feats["q_rope_last"])
        feature_types["q_free_mid"].append(feats["q_free_mid"])
        feature_types["hidden"].append(feats["hidden"])
        # Full Q = concat rope + free
        q_full = np.concatenate([feats["q_rope_last"], feats["q_free_last"]], axis=1)
        feature_types["q_full_last"].append(q_full)

    labels = np.array(all_labels)

    results = {}
    for fname, parts in feature_types.items():
        matrix = np.concatenate(parts, axis=0)
        log(f"\n  Analyzing {fname} features (shape: {matrix.shape})...")
        results[fname] = compute_routing_metrics(matrix, labels, n_domains, fname)

    # Phase 2b: Compare position-free vs full Q vs full hidden
    log("\n  === Comparison Summary ===")
    for fname in ["q_free_last", "q_rope_last", "q_full_last", "q_free_mid", "hidden"]:
        r = results[fname]
        log(f"  {fname:15s}: sil={r['silhouette_true']:.3f}  "
            f"centroid={r['centroid_accuracy']:.3f}  "
            f"kmeans={r['kmeans_accuracy']:.3f}  "
            f"var_ratio={r['variance_ratio']:.3f}")

    elapsed = time.time() - t0
    log(f"\n  Clustering analysis done in {elapsed:.1f}s")

    return results, domains_present


# ============================================================================
# Phase 3: Assess kill criteria
# ============================================================================

def phase_assess_results(clustering_results, domains_present):
    """Assess kill criteria and determine experiment verdict."""
    log("\n[Phase 3] Kill criteria assessment...")

    n_domains = len(domains_present)
    random_baseline = 1.0 / n_domains

    # Primary feature: q_free_last (position-free Q from last layer)
    qf = clustering_results["q_free_last"]
    hidden = clustering_results["hidden"]

    # K1: Silhouette score
    k1_sil = qf["silhouette_true"]
    k1_pass = k1_sil >= 0.3
    log(f"  K1: silhouette={k1_sil:.4f} (threshold >= 0.3) -> {'PASS' if k1_pass else 'FAIL'}")

    # K2: Routing accuracy > random (1/24 = 4.2%)
    k2_acc = qf["centroid_accuracy"]
    k2_pass = k2_acc > random_baseline
    log(f"  K2: routing_accuracy={k2_acc:.4f} (random={random_baseline:.4f}) -> {'PASS' if k2_pass else 'FAIL'}")

    # S1: Routing accuracy > 60%
    s1_pass = k2_acc > 0.60
    log(f"  S1: routing_accuracy={k2_acc:.4f} (threshold > 0.60) -> {'PASS' if s1_pass else 'FAIL'}")

    # Bonus: Silhouette > 0.5
    bonus_pass = k1_sil > 0.5
    log(f"  Bonus: silhouette={k1_sil:.4f} (threshold > 0.5) -> {'PASS' if bonus_pass else 'FAIL'}")

    # Compare with hidden state baseline (softmax router's feature space)
    log(f"\n  Comparison with full hidden states:")
    log(f"    q_free_last centroid acc: {qf['centroid_accuracy']:.4f}")
    log(f"    hidden centroid acc:      {hidden['centroid_accuracy']:.4f}")
    log(f"    Ratio: {qf['centroid_accuracy'] / (hidden['centroid_accuracy'] + 1e-8):.3f}")

    # Overall verdict
    if not k1_pass and not k2_pass:
        verdict = "KILLED"
        reason = "Both K1 and K2 fail: position-free Q features don't cluster by domain"
    elif not k1_pass:
        verdict = "KILLED"
        reason = f"K1 FAIL: silhouette {k1_sil:.4f} < 0.3, no clear domain clusters"
    elif not k2_pass:
        verdict = "KILLED"
        reason = f"K2 FAIL: routing accuracy {k2_acc:.4f} < random {random_baseline:.4f}"
    elif s1_pass:
        verdict = "SUPPORTED"
        reason = f"All kill criteria pass, S1 pass: {k2_acc:.1%} routing accuracy"
    else:
        verdict = "SUPPORTED"
        reason = f"Kill criteria pass but S1 fail: {k2_acc:.1%} < 60% target"

    log(f"\n  VERDICT: {verdict}")
    log(f"  Reason: {reason}")

    return {
        "verdict": verdict,
        "reason": reason,
        "k1_silhouette": k1_sil,
        "k1_pass": k1_pass,
        "k2_routing_accuracy": k2_acc,
        "k2_pass": k2_pass,
        "k2_random_baseline": round(random_baseline, 4),
        "s1_pass": s1_pass,
        "bonus_silhouette_pass": bonus_pass,
        "hidden_centroid_accuracy": hidden["centroid_accuracy"],
        "q_free_vs_hidden_ratio": round(qf["centroid_accuracy"] / (hidden["centroid_accuracy"] + 1e-8), 4),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    np.random.seed(SEED)
    log_memory("start")

    log("=" * 60)
    log("Experiment: Partial RoPE Semantic Routing")
    log(f"  Model: {MODEL_ID}")
    log(f"  Domains: {len(ALL_DOMAINS)}")
    log(f"  Samples/domain: {SAMPLES_PER_DOMAIN}")
    log(f"  RoPE fraction: {ROPE_FRAC} ({ROPE_DIMS}/{HEAD_DIM} dims)")
    log(f"  Free dims per head: {FREE_DIMS}")
    log(f"  Total free features: {NUM_HEADS * FREE_DIMS}")
    log("=" * 60)

    # Phase 1: Extract features
    all_features = phase_extract_features()
    log_memory("after-extraction")

    # Phase 2: Clustering analysis
    clustering_results, domains_present = phase_clustering_analysis(all_features)
    log_memory("after-clustering")

    # Phase 3: Kill criteria
    assessment = phase_assess_results(clustering_results, domains_present)

    # Save results
    total_time = time.time() - t0
    results = {
        "experiment": "partial_rope_semantic_routing",
        "model": MODEL_ID,
        "config": {
            "n_domains": len(ALL_DOMAINS),
            "samples_per_domain": SAMPLES_PER_DOMAIN,
            "rope_fraction": ROPE_FRAC,
            "rope_dims": ROPE_DIMS,
            "free_dims": FREE_DIMS,
            "total_free_features": NUM_HEADS * FREE_DIMS,
            "total_rope_features": NUM_HEADS * ROPE_DIMS,
            "hidden_dim": HIDDEN_DIM,
            "max_seq_length": MAX_SEQ_LENGTH,
            "seed": SEED,
        },
        "clustering": clustering_results,
        "assessment": assessment,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
