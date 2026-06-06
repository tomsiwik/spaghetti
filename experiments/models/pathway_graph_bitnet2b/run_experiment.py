#!/usr/bin/env python3
"""Pathway Graph Construction on BitNet-2B: Co-Activation Topology.

Kill criteria:
  K623: Persistence diagram has >=10 features with persistence > 0.1
  K624: High-persistence features NOT simply top-k singular vectors (rank corr < 0.5)

Type: verification
Platform: Apple M5 Pro 48GB, MLX

Method:
  1. Sample 10K inputs from 5 domain validation sets
  2. Collect FFN activations at target layer
  3. SVD of activation matrix -> top-k singular directions
  4. Build co-activation graph (edge = fraction of inputs where both directions activate)
  5. 0-dim persistent homology via ripser
  6. Compare persistence rank vs singular value rank
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
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx.utils import tree_unflatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
TARGET_LAYER = 15  # Middle layer (30 layers total)
TOP_K_DIRECTIONS = 100  # Number of singular directions to use
INPUTS_PER_DOMAIN = 450  # Use all available (400 train + 50 valid)
MAX_SEQ_LEN = 128  # Truncate for memory efficiency
BATCH_SIZE = 32  # Forward pass batch size
EPSILON_FRACTION = 0.5  # Activation threshold = 0.5 * max per direction (sparser graph)
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]


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
# Model utilities
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
# Data loading
# ============================================================================

def load_inputs(tokenizer, n_per_domain=450, max_seq_len=128):
    """Load and tokenize inputs from all 5 domain train+valid sets."""
    all_tokens = []
    domain_labels = []

    for domain in DOMAINS:
        count = 0
        for split in ["train", "valid"]:
            data_path = DATA_DIR / domain / f"{split}.jsonl"
            if not data_path.exists():
                continue
            with open(data_path) as f:
                for line in f:
                    text = json.loads(line)["text"]
                    tokens = tokenizer.encode(text)
                    if len(tokens) > max_seq_len:
                        tokens = tokens[:max_seq_len]
                    if len(tokens) < 8:
                        continue
                    all_tokens.append(tokens)
                    domain_labels.append(domain)
                    count += 1
                    if count >= n_per_domain:
                        break
            if count >= n_per_domain:
                break
        log(f"  {domain}: {count} inputs loaded")

    return all_tokens, domain_labels


# ============================================================================
# Activation collection with hook
# ============================================================================

def collect_activations(model, tokenizer, all_tokens, target_layer, batch_size=64):
    """Collect FFN output activations at target_layer for all inputs.

    Returns activation matrix H of shape (n_inputs, d_model) where each row
    is the mean-pooled FFN output activation for one input.
    """
    log(f"\n  Collecting activations at layer {target_layer}...")

    # Get the FFN output (after mlp.down_proj) via a simple hook
    activations = []

    # We process one input at a time due to variable lengths
    # but batch the mx.eval calls
    n_total = len(all_tokens)
    batch_acts = []

    for i, tokens in enumerate(all_tokens):
        input_ids = mx.array([tokens])

        # Forward through embedding + layers up to target
        h = model.model.embed_tokens(input_ids)
        for li, layer in enumerate(model.model.layers):
            if li < target_layer:
                h = layer(h, mask=None)
            elif li == target_layer:
                # Get the FFN (MLP) output specifically
                residual = h
                h_norm = layer.post_attention_layernorm(h)
                mlp_out = layer.mlp(h_norm)
                # Mean pool over sequence length
                act = mx.mean(mlp_out, axis=1)  # (1, d_model)
                batch_acts.append(act)
                break

        # Eval in batches to balance graph size and throughput
        if len(batch_acts) >= batch_size or i == n_total - 1:
            mx.eval(*batch_acts)
            for a in batch_acts:
                arr = np.array(a[0].astype(mx.float32))
                # Replace nan/inf with 0
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                activations.append(arr)
            batch_acts = []

            if (i + 1) % 500 == 0 or i == n_total - 1:
                log(f"    {i+1}/{n_total} inputs processed")

    H = np.stack(activations, axis=0)  # (n_inputs, d_model)
    # Aggressive cleanup: replace any inf/nan and clip extreme values
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    # Check for bad rows
    row_norms = np.linalg.norm(H, axis=1)
    bad_rows = ~np.isfinite(row_norms) | (row_norms == 0)
    if bad_rows.sum() > 0:
        log(f"  WARNING: {bad_rows.sum()} bad rows (nan/inf/zero), replacing with mean")
        good_mean = H[~bad_rows].mean(axis=0) if (~bad_rows).sum() > 0 else np.zeros(H.shape[1])
        H[bad_rows] = good_mean
    log(f"  Activation matrix shape: {H.shape}, range: [{H.min():.2f}, {H.max():.2f}]")
    return H


# ============================================================================
# Co-activation graph construction
# ============================================================================

def build_coactivation_graph(H, top_k=100, epsilon_frac=0.1):
    """Build co-activation graph from activation matrix.

    Args:
        H: (n_inputs, d_model) activation matrix
        top_k: number of singular directions to use as vertices
        epsilon_frac: activation threshold as fraction of max per direction

    Returns:
        adj_matrix: (top_k, top_k) edge weight matrix
        singular_values: top_k singular values
        V: (d_model, top_k) right singular vectors
    """
    log(f"\n  Computing SVD of activation matrix...")
    n, d = H.shape

    # Replace any nan/inf in H
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # Center activations
    H_centered = H - H.mean(axis=0, keepdims=True)

    # SVD (truncated to top_k) -- use float64 for numerical stability
    H_centered_f64 = H_centered.astype(np.float64)
    U, S, Vt = np.linalg.svd(H_centered_f64, full_matrices=False)
    V = Vt[:top_k].T.astype(np.float64)  # (d_model, top_k)
    singular_values = S[:top_k].astype(np.float64)

    log(f"  Top-{top_k} singular values: [{singular_values[0]:.2f}, ..., {singular_values[-1]:.2f}]")
    energy_captured = np.sum(singular_values**2) / np.sum(S**2)
    log(f"  Energy captured by top-{top_k}: {energy_captured:.4f}")

    # Project activations onto singular directions (float64 for stability)
    projections = H_centered_f64 @ V  # (n_inputs, top_k)
    projections = np.nan_to_num(projections, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute per-direction thresholds
    max_abs = np.max(np.abs(projections), axis=0)  # (top_k,)
    thresholds = epsilon_frac * max_abs  # (top_k,)

    # Binary activation matrix: is direction i active for input x?
    active = np.abs(projections) > thresholds[np.newaxis, :]  # (n_inputs, top_k)

    log(f"  Average activation rate per direction: {active.mean():.3f}")

    # Build co-activation matrix
    # w(i,j) = fraction of inputs where both i and j are active
    log(f"  Building co-activation matrix...")
    # Efficient: active^T @ active gives co-activation counts
    active_f64 = active.astype(np.float64)
    coact_counts = active_f64.T @ active_f64  # (top_k, top_k)
    adj_matrix = coact_counts / n  # Normalize to fractions
    adj_matrix = np.nan_to_num(adj_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out diagonal (self-loops not meaningful)
    np.fill_diagonal(adj_matrix, 0.0)

    n_nonzero = np.sum(adj_matrix > 0)
    log(f"  Raw edge weight range: [{adj_matrix[adj_matrix > 0].min():.4f}, {adj_matrix.max():.4f}]")
    log(f"  Raw non-zero edges: {n_nonzero} / {top_k * (top_k - 1)}")

    # Sparsify: keep only edges above median co-activation
    # This creates structure that PH can detect
    nonzero_weights = adj_matrix[adj_matrix > 0]
    if len(nonzero_weights) > 0:
        sparsify_threshold = np.percentile(nonzero_weights, 50)  # Keep top 50% of edges
        adj_matrix[adj_matrix < sparsify_threshold] = 0.0
        n_kept = np.sum(adj_matrix > 0)
        log(f"  After sparsification (threshold={sparsify_threshold:.4f}): {n_kept} edges kept")

    return adj_matrix, singular_values, V, energy_captured


# ============================================================================
# Persistent homology
# ============================================================================

def compute_persistence(adj_matrix):
    """Compute 0-dimensional persistent homology via sublevel filtration.

    We use a distance matrix: d(i,j) = 1 - w(i,j), so high co-activation = low distance.
    Ripser computes PH on the Rips complex built from this distance matrix.
    """
    from ripser import ripser

    log(f"\n  Computing persistent homology...")

    # Convert edge weights to distances: high weight = low distance
    max_weight = adj_matrix.max()
    if max_weight == 0:
        log("  WARNING: All edge weights are zero!")
        return np.array([]), np.array([])

    # Distance matrix: d = 1 - w/max_w (so strongest edges = shortest distances)
    # Zero-weight edges get distance = inf (not connected at finite threshold)
    dist_matrix = np.where(
        adj_matrix > 0,
        1.0 - adj_matrix / max_weight,
        np.inf
    )
    np.fill_diagonal(dist_matrix, 0.0)

    log(f"  Distance matrix: finite range [{dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)].min():.4f}, "
        f"{dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)].max():.4f}]")
    log(f"  Infinite edges (disconnected): {np.sum(np.isinf(dist_matrix) & ~np.eye(len(dist_matrix), dtype=bool))}")

    # Run ripser (0-dim PH = connected components)
    # Use thresh to limit computation to finite distances
    finite_max = dist_matrix[np.isfinite(dist_matrix)].max()
    result = ripser(dist_matrix, maxdim=0, distance_matrix=True, thresh=finite_max + 0.01)

    dgm = result['dgms'][0]  # 0-dimensional persistence diagram
    # dgm is (n_features, 2) with (birth, death) pairs
    # Infinite death = component never merges (the final surviving component)

    # Filter out the infinite-death feature (the single connected component at the end)
    finite_mask = np.isfinite(dgm[:, 1])
    finite_dgm = dgm[finite_mask]

    if len(finite_dgm) == 0:
        log("  WARNING: No finite persistence features found")
        return np.array([]), np.array([])

    persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
    log(f"  Total features (finite): {len(persistence)}")
    log(f"  Persistence range: [{persistence.min():.4f}, {persistence.max():.4f}]")
    log(f"  Features with persistence > 0.1: {np.sum(persistence > 0.1)}")
    log(f"  Features with persistence > 0.05: {np.sum(persistence > 0.05)}")

    return finite_dgm, persistence


# ============================================================================
# Analysis
# ============================================================================

def compute_random_baseline(adj_matrix, n_trials=5):
    """Random baseline: same density and weight distribution, shuffled edges.

    If the random baseline produces similar persistence, the real result is
    a sparsification artifact, not meaningful structure.
    """
    from ripser import ripser

    log(f"\n  Computing random baseline ({n_trials} trials)...")

    # Extract non-zero edge weights from the real graph
    mask = adj_matrix > 0
    real_weights = adj_matrix[mask]
    n_edges = len(real_weights)
    n_vertices = adj_matrix.shape[0]

    baseline_stats = []
    for trial in range(n_trials):
        # Create random symmetric matrix with same edge density and weight distribution
        rand_adj = np.zeros_like(adj_matrix)
        # Randomly place edges (upper triangle)
        triu_indices = np.triu_indices(n_vertices, k=1)
        n_possible = len(triu_indices[0])
        # Sample which edges to include (same count as real)
        n_edges_half = n_edges // 2  # Symmetric, so half the edges in upper triangle
        chosen = np.random.choice(n_possible, size=min(n_edges_half, n_possible), replace=False)
        # Assign shuffled weights
        shuffled_weights = np.random.choice(real_weights, size=len(chosen), replace=True)
        rand_adj[triu_indices[0][chosen], triu_indices[1][chosen]] = shuffled_weights
        rand_adj = rand_adj + rand_adj.T  # Symmetrize

        # Same distance transform
        max_w = rand_adj.max()
        if max_w == 0:
            continue
        dist = np.where(rand_adj > 0, 1.0 - rand_adj / max_w, np.inf)
        np.fill_diagonal(dist, 0.0)

        finite_max = dist[np.isfinite(dist)].max() if np.any(np.isfinite(dist) & (dist > 0)) else 1.0
        result = ripser(dist, maxdim=0, distance_matrix=True, thresh=finite_max + 0.01)
        dgm = result['dgms'][0]
        finite_mask = np.isfinite(dgm[:, 1])
        if finite_mask.sum() > 0:
            pers = dgm[finite_mask, 1] - dgm[finite_mask, 0]
            baseline_stats.append({
                "n_features": len(pers),
                "n_above_0.1": int(np.sum(pers > 0.1)),
                "max_persistence": float(pers.max()),
                "mean_persistence": float(pers.mean()),
            })
        else:
            baseline_stats.append({"n_features": 0, "n_above_0.1": 0, "max_persistence": 0, "mean_persistence": 0})

    # Summarize
    avg_features = np.mean([s["n_features"] for s in baseline_stats])
    avg_above_01 = np.mean([s["n_above_0.1"] for s in baseline_stats])
    avg_max = np.mean([s["max_persistence"] for s in baseline_stats])
    avg_mean = np.mean([s["mean_persistence"] for s in baseline_stats])

    log(f"  Random baseline (avg over {n_trials} trials):")
    log(f"    Features: {avg_features:.1f} (real: {np.sum(adj_matrix > 0) // 2})")
    log(f"    Above 0.1: {avg_above_01:.1f}")
    log(f"    Max persistence: {avg_max:.4f}")
    log(f"    Mean persistence: {avg_mean:.4f}")

    return {
        "trials": baseline_stats,
        "avg_n_features": float(avg_features),
        "avg_n_above_0.1": float(avg_above_01),
        "avg_max_persistence": float(avg_max),
        "avg_mean_persistence": float(avg_mean),
    }


def compute_vertex_persistence(adj_matrix, top_k):
    """Compute per-vertex persistence contribution using union-find.

    For each vertex, find the maximum persistence of any merge event it participates in.
    This gives a true topological importance measure, not just degree.
    """
    from ripser import ripser

    log(f"\n  Computing per-vertex persistence (topological importance)...")

    max_weight = adj_matrix.max()
    if max_weight == 0:
        return np.zeros(top_k)

    # Get all edges sorted by weight (descending = shortest distance first)
    edges = []
    for i in range(top_k):
        for j in range(i + 1, top_k):
            if adj_matrix[i, j] > 0:
                dist = 1.0 - adj_matrix[i, j] / max_weight
                edges.append((dist, i, j))
    edges.sort()  # Ascending distance

    # Union-find to track merge events
    parent = list(range(top_k))
    rank_uf = [0] * top_k
    birth = [0.0] * top_k  # All components born at t=0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    vertex_max_persistence = np.zeros(top_k)

    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Merge event: persistence = dist - 0 (birth is always 0 for 0-dim)
            persistence = dist
            # Both vertices participate in this merge
            vertex_max_persistence[i] = max(vertex_max_persistence[i], persistence)
            vertex_max_persistence[j] = max(vertex_max_persistence[j], persistence)
            # Union
            if rank_uf[ri] < rank_uf[rj]:
                parent[ri] = rj
            elif rank_uf[ri] > rank_uf[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank_uf[ri] += 1

    return vertex_max_persistence


def analyze_persistence_vs_sv(persistence, singular_values, adj_matrix, top_k):
    """Check if topologically important vertices correlate with top singular values.

    Uses per-vertex persistence contribution (max persistence of any merge event
    the vertex participates in) rather than degree.
    """
    from scipy.stats import spearmanr

    log(f"\n  Analyzing persistence vs singular value rank...")

    # Compute per-vertex topological importance
    vertex_persistence = compute_vertex_persistence(adj_matrix, top_k)

    # Also compute degree for comparison
    vertex_degree = adj_matrix.sum(axis=1)

    # Rank correlation: persistence importance vs SV rank
    sv_rank_order = np.argsort(-singular_values)
    pers_rank_order = np.argsort(-vertex_persistence)
    degree_rank_order = np.argsort(-vertex_degree)

    sv_ranks = np.empty(top_k, dtype=int)
    pers_ranks = np.empty(top_k, dtype=int)
    degree_ranks = np.empty(top_k, dtype=int)
    sv_ranks[sv_rank_order] = np.arange(top_k)
    pers_ranks[pers_rank_order] = np.arange(top_k)
    degree_ranks[degree_rank_order] = np.arange(top_k)

    rho_pers, p_pers = spearmanr(sv_ranks, pers_ranks)
    rho_degree, p_degree = spearmanr(sv_ranks, degree_ranks)

    log(f"  Spearman (SV rank vs PERSISTENCE rank): rho={rho_pers:.4f}, p={p_pers:.4e}")
    log(f"  Spearman (SV rank vs DEGREE rank): rho={rho_degree:.4f}, p={p_degree:.4e}")

    # Top-10 overlap
    top10_sv = set(sv_rank_order[:10])
    top10_pers = set(pers_rank_order[:10])
    top10_degree = set(degree_rank_order[:10])
    overlap_pers = len(top10_sv & top10_pers)
    overlap_degree = len(top10_sv & top10_degree)
    log(f"  Top-10 overlap (SV vs persistence): {overlap_pers}/10")
    log(f"  Top-10 overlap (SV vs degree): {overlap_degree}/10")

    # Power-law check
    if len(persistence) > 1:
        sorted_pers = np.sort(persistence)[::-1]
        log_rank = np.log(np.arange(1, len(sorted_pers) + 1))
        log_pers = np.log(sorted_pers + 1e-10)
        if len(log_rank) > 2:
            coeffs = np.polyfit(log_rank, log_pers, 1)
            log(f"  Power-law exponent (log-log slope): {coeffs[0]:.2f}")
            log(f"  R^2 of power-law fit: {np.corrcoef(log_rank, log_pers)[0,1]**2:.4f}")

    return {
        "spearman_rho_persistence": float(rho_pers),
        "spearman_p_persistence": float(p_pers),
        "spearman_rho_degree": float(rho_degree),
        "spearman_p_degree": float(p_degree),
        "top10_overlap_persistence": overlap_pers,
        "top10_overlap_degree": overlap_degree,
        "vertex_persistence_top10": sorted(vertex_persistence.tolist(), reverse=True)[:10],
    }


def analyze_domain_bridges(H, V, domain_labels, persistence, adj_matrix):
    """Check if cross-domain inputs create longer-persisting bridges."""
    log(f"\n  Analyzing domain structure...")

    # Project activations onto singular directions
    H_centered = (H - H.mean(axis=0, keepdims=True)).astype(np.float64)
    projections = np.nan_to_num(H_centered @ V, nan=0.0, posinf=0.0, neginf=0.0)  # (n_inputs, top_k)

    # For each domain, compute mean activation pattern
    domain_means = {}
    for domain in DOMAINS:
        mask = np.array([d == domain for d in domain_labels])
        if mask.sum() > 0:
            domain_means[domain] = projections[mask].mean(axis=0)  # (top_k,)

    # Compute inter-domain similarity (cosine)
    domain_list = list(domain_means.keys())
    n_domains = len(domain_list)
    domain_sim = np.zeros((n_domains, n_domains))
    for i in range(n_domains):
        for j in range(n_domains):
            a = domain_means[domain_list[i]]
            b = domain_means[domain_list[j]]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                domain_sim[i, j] = np.dot(a, b) / (norm_a * norm_b)

    log(f"  Inter-domain cosine similarities:")
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            log(f"    {domain_list[i]} - {domain_list[j]}: {domain_sim[i,j]:.4f}")

    # Identify which directions are "bridge" directions (activated by multiple domains)
    max_abs = np.max(np.abs(projections), axis=0)
    thresholds = 0.1 * max_abs
    active = np.abs(projections) > thresholds[np.newaxis, :]

    # For each direction, count how many domains activate it above threshold
    domain_activation = {}
    for domain in DOMAINS:
        mask = np.array([d == domain for d in domain_labels])
        domain_activation[domain] = active[mask].mean(axis=0)  # fraction of domain inputs activating each direction

    # Bridge directions: activated by >= 3 domains at > 20% rate
    bridge_count = np.zeros(V.shape[1])
    for domain in DOMAINS:
        bridge_count += (domain_activation[domain] > 0.2).astype(float)

    n_bridges = np.sum(bridge_count >= 3)
    n_specialist = np.sum(bridge_count == 1)
    log(f"  Bridge directions (>= 3 domains): {n_bridges}")
    log(f"  Specialist directions (1 domain): {n_specialist}")

    return {
        "domain_similarities": {f"{domain_list[i]}-{domain_list[j]}": float(domain_sim[i, j])
                                 for i in range(n_domains) for j in range(i+1, n_domains)},
        "n_bridge_directions": int(n_bridges),
        "n_specialist_directions": int(n_specialist),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("PATHWAY GRAPH CONSTRUCTION ON BitNet-2B")
    log("=" * 70)
    log(f"Target layer: {TARGET_LAYER}")
    log(f"Top-k directions: {TOP_K_DIRECTIONS}")
    log(f"Inputs per domain: {INPUTS_PER_DOMAIN}")
    log(f"Total inputs: {INPUTS_PER_DOMAIN * len(DOMAINS)}")
    log_memory("start")

    np.random.seed(SEED)
    mx.random.seed(SEED)

    # Load model
    log("\n  Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load")

    # Load inputs
    log("\n  Loading inputs...")
    all_tokens, domain_labels = load_inputs(
        tokenizer, n_per_domain=INPUTS_PER_DOMAIN, max_seq_len=MAX_SEQ_LEN)
    log(f"  Total inputs: {len(all_tokens)}")

    # Collect activations
    H = collect_activations(model, tokenizer, all_tokens, TARGET_LAYER, batch_size=BATCH_SIZE)
    log_memory("post-activations")

    # Free model to save memory
    del model, tokenizer
    cleanup()
    log_memory("post-model-cleanup")

    # Build co-activation graph
    adj_matrix, singular_values, V, energy_captured = build_coactivation_graph(
        H, top_k=TOP_K_DIRECTIONS, epsilon_frac=EPSILON_FRACTION)

    # Compute persistent homology
    finite_dgm, persistence = compute_persistence(adj_matrix)

    if len(persistence) == 0:
        log("\nFATAL: No persistence features found. Cannot evaluate kill criteria.")
        results = {
            "experiment": "pathway_graph_bitnet2b",
            "status": "KILLED",
            "error": "No finite persistence features",
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
        return

    # Kill criteria evaluation
    log("\n" + "=" * 70)
    log("KILL CRITERIA EVALUATION")
    log("=" * 70)

    # Random baseline control (Fix #1 from review)
    baseline = compute_random_baseline(adj_matrix, n_trials=5)

    # K623: >= 10 features with persistence > 0.1 AND more than random baseline
    n_high_persistence = int(np.sum(persistence > 0.1))
    baseline_above_01 = baseline["avg_n_above_0.1"]
    # Real topology must have MORE persistent features than random
    k623_pass = n_high_persistence >= 10 and n_high_persistence > baseline_above_01 * 1.5
    log(f"  K623 (>=10 features persistence > 0.1, > 1.5x random): {'PASS' if k623_pass else 'FAIL -> KILL'}")
    log(f"    Real: {n_high_persistence}, Random baseline: {baseline_above_01:.1f}")

    # K624: Persistence rank vs SV rank correlation < 0.5 (Fix #3 from review)
    rank_analysis = analyze_persistence_vs_sv(persistence, singular_values, adj_matrix, TOP_K_DIRECTIONS)
    k624_rho = rank_analysis["spearman_rho_persistence"]
    k624_pass = abs(k624_rho) < 0.5
    log(f"  K624 (persistence rank vs SV rank corr < 0.5): {'PASS' if k624_pass else 'FAIL -> KILL'}")
    log(f"    Spearman rho (persistence): {k624_rho:.4f}")
    log(f"    Spearman rho (degree, for comparison): {rank_analysis['spearman_rho_degree']:.4f}")

    # Domain analysis (diagnostic)
    domain_analysis = analyze_domain_bridges(H, V, domain_labels, persistence, adj_matrix)

    # Compile results
    total_time = time.time() - t0

    # Persistence statistics
    pers_stats = {
        "total_features": len(persistence),
        "above_0.1": n_high_persistence,
        "above_0.05": int(np.sum(persistence > 0.05)),
        "above_0.01": int(np.sum(persistence > 0.01)),
        "max": float(persistence.max()),
        "median": float(np.median(persistence)),
        "mean": float(persistence.mean()),
        "top_10_persistence": sorted(persistence.tolist(), reverse=True)[:10],
    }

    results = {
        "experiment": "pathway_graph_bitnet2b",
        "model": MODEL_ID,
        "target_layer": TARGET_LAYER,
        "top_k_directions": TOP_K_DIRECTIONS,
        "n_inputs": len(all_tokens),
        "epsilon_fraction": EPSILON_FRACTION,
        "energy_captured": float(energy_captured),
        "singular_values_top10": singular_values[:10].tolist(),
        "persistence_stats": pers_stats,
        "persistence_diagram": finite_dgm.tolist(),
        "rank_analysis": rank_analysis,
        "domain_analysis": domain_analysis,
        "random_baseline": baseline,
        "kill_criteria": {
            "K623": {"pass": k623_pass, "n_features_above_0.1": n_high_persistence,
                     "random_baseline_above_0.1": baseline_above_01},
            "K624": {"pass": k624_pass,
                     "spearman_rho_persistence": k624_rho,
                     "spearman_rho_degree": rank_analysis["spearman_rho_degree"]},
        },
        "timing_s": total_time,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  K623 (non-trivial topology): {'PASS' if k623_pass else 'FAIL'} ({n_high_persistence} features, random baseline: {baseline_above_01:.1f})")
    log(f"  K624 (not just top-k SVs): {'PASS' if k624_pass else 'FAIL'} (persistence rho={k624_rho:.4f})")
    log(f"  Bridge directions: {domain_analysis['n_bridge_directions']}")
    log(f"  Specialist directions: {domain_analysis['n_specialist_directions']}")
    all_pass = k623_pass and k624_pass
    log(f"  Overall: {'ALL PASS -> SUPPORTED' if all_pass else 'KILL TRIGGERED'}")


if __name__ == "__main__":
    main()
