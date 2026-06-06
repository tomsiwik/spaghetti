#!/usr/bin/env python3
"""Sheaf Cohomology Dimension Estimation: Bridge Adapter Rank Budget.

Computes dim(H^1) of a sheaf on the Cech nerve of the adapter specialization cover.
Uses corrected inputs from exp_knowledge_region_overlap (Finding #240):
  - Specialization sets (top-k PPL) instead of improvement sets
  - L2 relative difference instead of cosine similarity
  - Multi-layer extraction (layers 5, 10, 15, 20, 25)

Kill criteria:
  K1 (#648): Cover NOT degenerate (specialization top-k must not assign all samples to all adapters)
  K2 (#649): H^1 non-trivial at >= 1 layer (dim(H^1) > 0)

Type: Guided exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx_lm.models.bitnet import create_attention_mask
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
# Layers to extract hidden states from (early, mid-early, mid, mid-late, late)
EXTRACT_LAYERS = [5, 10, 15, 20, 25]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Scale per adapter (Finding #217)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Top-k for specialization cover
TOP_K_VALUES = [2, 3]  # Test k=2 (non-trivial) and k=3 (for robustness)


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
# Model utilities (reused from knowledge_region_overlap)
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


def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    return dict(mx.load(str(adapter_path)))


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    n_layers = len(model.model.layers)
    merge_count = 0
    di = DOMAINS.index(domain)
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
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]
            delta = scale * (b_mx.T @ a_mx.T)
            module.weight = module.weight + delta
            merge_count += 1
    mx.eval(model.parameters())
    return merge_count


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
                base_weights[(li, key)] = module.weight
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


# ============================================================================
# Data loading
# ============================================================================

def load_all_samples():
    samples = []
    for domain in DOMAINS:
        fpath = DATA_DIR / domain / "valid.jsonl"
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                samples.append({"text": obj["text"], "domain": domain})
    log(f"  Loaded {len(samples)} samples from {len(DOMAINS)} domains")
    return samples


# ============================================================================
# Per-sample PPL and multi-layer hidden state extraction
# ============================================================================

def compute_ppl_and_multilayer_hidden(model, tokenizer, samples):
    """Compute PPL and hidden states at EXTRACT_LAYERS for each sample.

    Returns:
        ppls: list of float
        hiddens: dict of {layer_idx: list of np.ndarray}
    """
    ppls = []
    hiddens = {l: [] for l in EXTRACT_LAYERS}

    for i, sample in enumerate(samples):
        tokens = tokenizer.encode(sample["text"])
        if len(tokens) < 2:
            ppls.append(float("inf"))
            for l in EXTRACT_LAYERS:
                hiddens[l].append(None)
            continue

        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        # Forward with hidden state capture at multiple layers
        h = model.model.embed_tokens(x)
        mask = create_attention_mask(h, cache=None)

        captured = {}
        for li, layer_module in enumerate(model.model.layers):
            h = layer_module(h, mask=mask, cache=None)
            if li in EXTRACT_LAYERS:
                # Mean pool over sequence
                hidden_vec = mx.mean(h, axis=1)  # [1, d]
                mx.eval(hidden_vec)
                captured[li] = np.array(hidden_vec[0].astype(mx.float32))

        # Get logits for PPL
        h = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)

        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        n_tokens = y.size
        ppl = math.exp(min(loss.item() / n_tokens, 100))
        ppls.append(ppl)

        for l in EXTRACT_LAYERS:
            hiddens[l].append(captured.get(l))

        del logits, loss, x, y, h
        for v in captured.values():
            del v

        if (i + 1) % 50 == 0:
            log(f"    Processed {i+1}/{len(samples)} samples")

    return ppls, hiddens


# ============================================================================
# Phase 1: Base model PPL (no hidden states needed)
# ============================================================================

def phase_base_ppl(samples):
    log("\n" + "=" * 70)
    log("[Phase 1] Base model PPL")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.eval(model.parameters())
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    log_memory("post-load")

    # Compute base PPL only (no hidden states)
    ppls = []
    for i, sample in enumerate(samples):
        tokens = tokenizer.encode(sample["text"])
        if len(tokens) < 2:
            ppls.append(float("inf"))
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        n_tokens = y.size
        ppl = math.exp(min(loss.item() / n_tokens, 100))
        ppls.append(ppl)
        del logits, loss, x, y
        if (i + 1) % 50 == 0:
            log(f"    Processed {i+1}/{len(samples)} samples")

    elapsed = time.time() - t0
    log(f"  Phase 1 complete in {elapsed:.1f}s")
    log_memory("post-base-ppl")

    base_weights = save_base_weights(model)
    return ppls, model, tokenizer, base_weights


# ============================================================================
# Phase 2: Per-adapter PPL + multi-layer hidden states
# ============================================================================

def phase_adapter_eval(model, tokenizer, base_weights, samples, domain):
    log(f"\n  --- Adapter: {domain} (scale={DOMAIN_SCALES[domain]}) ---")
    t0 = time.time()
    skeleton = load_skeleton()
    adapter = load_adapter(domain)

    restore_base_weights(model, base_weights)
    merge_count = premerge_single_adapter(
        model, skeleton, adapter, domain, DOMAIN_SCALES[domain]
    )
    log(f"    Merged {merge_count} layers")
    del skeleton, adapter
    gc.collect()
    mx.clear_cache()

    ppls, hiddens = compute_ppl_and_multilayer_hidden(model, tokenizer, samples)

    elapsed = time.time() - t0
    log(f"    {domain} done in {elapsed:.1f}s")
    return ppls, hiddens


# ============================================================================
# Phase 3: Sheaf cohomology analysis
# ============================================================================

def build_specialization_cover(adapter_ppls, n_samples, k):
    """Build top-k specialization cover: U_i = {x : adapter i in top-k lowest PPL}."""
    cover = {d: set() for d in DOMAINS}
    for idx in range(n_samples):
        domain_ppls = [(d, adapter_ppls[d][idx]) for d in DOMAINS]
        domain_ppls.sort(key=lambda t: t[1])
        for d, _ in domain_ppls[:k]:
            cover[d].add(idx)
    return cover


def compute_cech_nerve(cover, domains):
    """Compute Cech nerve from cover: vertices, edges (non-empty pairwise overlaps),
    triangles (non-empty triple overlaps)."""
    vertices = list(domains)
    edges = []
    edge_overlaps = {}
    for d1, d2 in combinations(domains, 2):
        overlap = cover[d1] & cover[d2]
        if len(overlap) > 0:
            edges.append((d1, d2))
            edge_overlaps[(d1, d2)] = sorted(overlap)

    triangles = []
    triangle_overlaps = {}
    for d1, d2, d3 in combinations(domains, 3):
        overlap = cover[d1] & cover[d2] & cover[d3]
        if len(overlap) > 0:
            triangles.append((d1, d2, d3))
            triangle_overlaps[(d1, d2, d3)] = sorted(overlap)

    return {
        "vertices": vertices,
        "edges": edges,
        "edge_overlaps": edge_overlaps,
        "triangles": triangles,
        "triangle_overlaps": triangle_overlaps,
    }


def compute_sheaf_cohomology_at_layer(nerve, adapter_hiddens, layer_idx):
    """Compute dim(H^1) for a sheaf on the Cech nerve at a specific layer.

    Uses the representation difference approach:
    - For each edge (i,j), compute D_{ij} = matrix of h_i(x) - h_j(x) for overlap samples
    - Construct coboundary matrix delta_0
    - dim(H^1) = dim(C^1) - rank(delta_0) - rank(delta_1)

    For the VALUE sheaf (vector-valued stalks), we work with:
    - The coboundary delta_0: assigns oriented differences to edges
    - The coboundary delta_1: checks cocycle condition on triangles

    Returns dict with H^1 dimension and diagnostic info.
    """
    vertices = nerve["vertices"]
    edges = nerve["edges"]
    triangles = nerve["triangles"]
    edge_overlaps = nerve["edge_overlaps"]
    triangle_overlaps = nerve["triangle_overlaps"]

    n_vertices = len(vertices)
    n_edges = len(edges)
    n_triangles = len(triangles)

    if n_edges == 0:
        return {
            "dim_H0": n_vertices,
            "dim_H1": 0,
            "dim_C0": n_vertices,
            "dim_C1": 0,
            "dim_C2": 0,
            "rank_delta0": 0,
            "rank_delta1": 0,
            "edge_diff_stats": {},
            "note": "No edges in nerve — trivial cohomology",
        }

    vertex_idx = {v: i for i, v in enumerate(vertices)}
    edge_idx = {e: i for i, e in enumerate(edges)}

    # --- Approach 1: Topological (scalar) Betti numbers ---
    # This gives the "structural" H^1 from the nerve topology alone

    # Construct delta_0 (incidence matrix): n_edges x n_vertices
    # For oriented edge (i -> j): delta_0[e, i] = -1, delta_0[e, j] = +1
    delta_0_scalar = np.zeros((n_edges, n_vertices))
    for ei, (d1, d2) in enumerate(edges):
        delta_0_scalar[ei, vertex_idx[d1]] = -1.0
        delta_0_scalar[ei, vertex_idx[d2]] = +1.0

    # Construct delta_1: n_triangles x n_edges
    # For oriented triangle (i,j,k): delta_1[t, (i,j)] = +1, delta_1[t, (j,k)] = +1, delta_1[t, (i,k)] = -1
    delta_1_scalar = np.zeros((n_triangles, n_edges)) if n_triangles > 0 else np.zeros((0, n_edges))
    for ti, (d1, d2, d3) in enumerate(triangles):
        # Edges: (d1,d2), (d1,d3), (d2,d3) — with signs from orientation
        for pair, sign in [((d1, d2), +1), ((d1, d3), -1), ((d2, d3), +1)]:
            if pair in edge_idx:
                delta_1_scalar[ti, edge_idx[pair]] = sign

    rank_d0 = np.linalg.matrix_rank(delta_0_scalar)
    rank_d1 = np.linalg.matrix_rank(delta_1_scalar) if n_triangles > 0 else 0

    # Betti numbers
    beta_0 = n_vertices - rank_d0  # connected components
    beta_1 = n_edges - rank_d0 - rank_d1  # independent cycles
    beta_2 = n_triangles - rank_d1 if n_triangles > 0 else 0

    # --- Approach 2: Data-informed sheaf H^1 via restriction map analysis ---
    # For each edge, compute the representation difference matrix and its rank
    # This tells us the "effective" dimension of incompatibility

    edge_diff_stats = {}
    edge_diff_ranks = {}
    all_diffs = []

    for ei, (d1, d2) in enumerate(edges):
        overlap_samples = edge_overlaps[(d1, d2)]
        diffs = []
        for idx in overlap_samples:
            h1 = adapter_hiddens[d1][layer_idx][idx]
            h2 = adapter_hiddens[d2][layer_idx][idx]
            if h1 is None or h2 is None:
                continue
            diff = h1 - h2
            diffs.append(diff)

        if len(diffs) < 2:
            edge_diff_stats[f"{d1}_{d2}"] = {
                "n_samples": len(diffs),
                "rank": 0,
                "mean_l2_rel": 0.0,
            }
            edge_diff_ranks[(d1, d2)] = 0
            continue

        D = np.stack(diffs)  # (m, d)
        # Compute rank of difference matrix (with tolerance)
        # Use SVD to find number of significant singular values
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        # Threshold: singular values > 1% of max
        threshold = 0.01 * s[0] if s[0] > 1e-8 else 1e-8
        rank = int(np.sum(s > threshold))

        # Also compute L2 relative differences
        base_norms = []
        for idx in overlap_samples:
            h1 = adapter_hiddens[d1][layer_idx][idx]
            if h1 is not None:
                base_norms.append(np.linalg.norm(h1))
        mean_base_norm = np.mean(base_norms) if base_norms else 1.0
        mean_diff_norm = np.mean([np.linalg.norm(d) for d in diffs])
        mean_l2_rel = mean_diff_norm / mean_base_norm if mean_base_norm > 1e-8 else 0.0

        edge_diff_stats[f"{d1}_{d2}"] = {
            "n_samples": len(diffs),
            "rank": rank,
            "singular_values_top5": s[:5].tolist(),
            "mean_l2_rel": float(mean_l2_rel),
            "mean_diff_norm": float(mean_diff_norm),
        }
        edge_diff_ranks[(d1, d2)] = rank
        all_diffs.extend(diffs)

    # --- Approach 3: Global difference space analysis ---
    # Stack ALL edge differences and compute the rank of the combined system
    # This gives the total "incompatibility dimension" across all edges
    global_rank = 0
    global_sv_top10 = []
    if all_diffs:
        D_global = np.stack(all_diffs)
        U_g, s_g, Vt_g = np.linalg.svd(D_global, full_matrices=False)
        threshold_g = 0.01 * s_g[0] if s_g[0] > 1e-8 else 1e-8
        global_rank = int(np.sum(s_g > threshold_g))
        global_sv_top10 = s_g[:10].tolist()

    # --- Approach 4: Sheaf Laplacian H^1 ---
    # Construct the weighted coboundary using actual representation differences
    # For each edge, use the mean difference vector as the "weight"
    # This is a simplified sheaf where stalks are 1-dimensional (scalar sheaf
    # weighted by incompatibility magnitude)

    # Weighted incidence matrix: multiply each edge row by mean L2 rel diff
    delta_0_weighted = delta_0_scalar.copy()
    for ei, (d1, d2) in enumerate(edges):
        stat = edge_diff_stats.get(f"{d1}_{d2}", {})
        weight = stat.get("mean_l2_rel", 0.0)
        delta_0_weighted[ei, :] *= weight

    # Hodge Laplacian L_1 = delta_0^T @ delta_0 + delta_1 @ delta_1^T
    # For scalar sheaf: L_1 is n_edges x n_edges
    L_up = delta_0_scalar.T @ delta_0_scalar  # n_vertices x n_vertices (L_0)
    L_1 = delta_0_scalar @ delta_0_scalar.T  # n_edges x n_edges
    if n_triangles > 0:
        L_1 += delta_1_scalar.T @ delta_1_scalar

    # Nullity of L_1 = dim(H^1) for scalar sheaf
    eigenvalues = np.linalg.eigvalsh(L_1)
    # Count near-zero eigenvalues (< 1e-8)
    hodge_H1 = int(np.sum(np.abs(eigenvalues) < 1e-8))

    return {
        "dim_H0": int(beta_0),
        "dim_H1_betti": int(beta_1),  # Topological (from Betti numbers)
        "dim_H1_hodge": hodge_H1,  # From Hodge Laplacian (should equal beta_1)
        "dim_C0": n_vertices,
        "dim_C1": n_edges,
        "dim_C2": n_triangles,
        "rank_delta0": int(rank_d0),
        "rank_delta1": int(rank_d1),
        "euler_characteristic": int(n_vertices - n_edges + n_triangles),
        "edge_diff_stats": edge_diff_stats,
        "edge_diff_ranks": {f"{d1}_{d2}": r for (d1, d2), r in edge_diff_ranks.items()},
        "global_diff_rank": global_rank,
        "global_sv_top10": global_sv_top10,
        "L1_eigenvalues": eigenvalues.tolist(),
    }


def phase_analysis(samples, base_ppls, adapter_ppls, adapter_hiddens):
    """Full sheaf cohomology analysis across multiple layers and k values."""
    log("\n" + "=" * 70)
    log("[Phase 3] Sheaf Cohomology Analysis")
    log("=" * 70)

    n_samples = len(samples)
    results = {}

    for k in TOP_K_VALUES:
        log(f"\n  === Top-{k} Specialization Cover ===")

        # Build cover
        cover = build_specialization_cover(adapter_ppls, n_samples, k)

        # Report cover structure
        cover_sizes = {}
        for d in DOMAINS:
            size = len(cover[d])
            cover_sizes[d] = size
            by_source = {}
            for idx in cover[d]:
                src = samples[idx]["domain"]
                by_source[src] = by_source.get(src, 0) + 1
            log(f"    |U_{d}| = {size} (breakdown: {by_source})")

        # Check K1: is cover degenerate?
        max_cover_size = max(cover_sizes.values())
        min_cover_size = min(cover_sizes.values())
        all_in_all = all(size == n_samples for size in cover_sizes.values())
        log(f"    Cover sizes: min={min_cover_size}, max={max_cover_size}, "
            f"degenerate={all_in_all}")

        # Build Cech nerve
        nerve = compute_cech_nerve(cover, DOMAINS)
        log(f"    Cech nerve: {len(nerve['vertices'])} vertices, "
            f"{len(nerve['edges'])} edges, {len(nerve['triangles'])} triangles")
        for d1, d2 in nerve["edges"]:
            log(f"      Edge ({d1}, {d2}): {len(nerve['edge_overlaps'][(d1, d2)])} samples")
        for d1, d2, d3 in nerve["triangles"]:
            log(f"      Triangle ({d1}, {d2}, {d3}): "
                f"{len(nerve['triangle_overlaps'][(d1, d2, d3)])} samples")

        # Compute cohomology at each layer
        layer_results = {}
        for layer_idx in EXTRACT_LAYERS:
            log(f"\n    --- Layer {layer_idx} ---")
            cohom = compute_sheaf_cohomology_at_layer(
                nerve, adapter_hiddens, layer_idx
            )
            layer_results[layer_idx] = cohom

            log(f"      H^0 = {cohom['dim_H0']} (connected components)")
            log(f"      H^1 (Betti) = {cohom['dim_H1_betti']} (independent cycles)")
            log(f"      H^1 (Hodge) = {cohom['dim_H1_hodge']}")
            log(f"      Global diff rank = {cohom['global_diff_rank']}")
            log(f"      Euler char = {cohom['euler_characteristic']}")

            # Edge difference details
            for edge_key, stats in cohom["edge_diff_stats"].items():
                log(f"      Edge {edge_key}: rank={stats['rank']}, "
                    f"L2_rel={stats['mean_l2_rel']:.4f}, "
                    f"n={stats['n_samples']}")

        results[f"k{k}"] = {
            "cover_sizes": cover_sizes,
            "cover_degenerate": all_in_all,
            "nerve_summary": {
                "n_vertices": len(nerve["vertices"]),
                "n_edges": len(nerve["edges"]),
                "n_triangles": len(nerve["triangles"]),
                "edges": [list(e) for e in nerve["edges"]],
                "triangles": [list(t) for t in nerve["triangles"]],
                "edge_overlap_sizes": {
                    f"{d1}_{d2}": len(ov)
                    for (d1, d2), ov in nerve["edge_overlaps"].items()
                },
                "triangle_overlap_sizes": {
                    f"{d1}_{d2}_{d3}": len(ov)
                    for (d1, d2, d3), ov in nerve["triangle_overlaps"].items()
                },
            },
            "layer_results": {
                str(l): {
                    k2: v for k2, v in lr.items()
                    if k2 != "L1_eigenvalues"  # save space
                }
                for l, lr in layer_results.items()
            },
            "layer_L1_eigenvalues": {
                str(l): lr["L1_eigenvalues"]
                for l, lr in layer_results.items()
            },
        }

    # --- Kill criteria ---
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Cover not degenerate for k=2
    k2_degenerate = results["k2"]["cover_degenerate"]
    k1_pass = not k2_degenerate
    log(f"  K1 (#648): Cover degenerate (k=2)? {k2_degenerate} -> "
        f"{'PASS' if k1_pass else 'FAIL'}")

    # K2: H^1 > 0 at any layer (for k=2)
    any_h1_nonzero = False
    for l in EXTRACT_LAYERS:
        h1_betti = results["k2"]["layer_results"][str(l)]["dim_H1_betti"]
        h1_hodge = results["k2"]["layer_results"][str(l)]["dim_H1_hodge"]
        if h1_betti > 0 or h1_hodge > 0:
            any_h1_nonzero = True
            log(f"  K2 check: Layer {l}: H^1_betti={h1_betti}, H^1_hodge={h1_hodge} -> NON-TRIVIAL")
        else:
            log(f"  K2 check: Layer {l}: H^1_betti={h1_betti}, H^1_hodge={h1_hodge}")

    k2_pass = any_h1_nonzero
    log(f"  K2 (#649): H^1 > 0 at any layer? {any_h1_nonzero} -> "
        f"{'PASS' if k2_pass else 'FAIL'}")

    results["kill_criteria"] = {
        "k1_cover_not_degenerate": k1_pass,
        "k2_h1_nonzero_any_layer": k2_pass,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("Sheaf Cohomology Dimension Estimation")
    log("Bridge Adapter Rank Budget")
    log("=" * 70)
    log_memory("start")

    # Load samples
    samples = load_all_samples()

    # Phase 1: Base PPL
    base_ppls, model, tokenizer, base_weights = phase_base_ppl(samples)

    # Phase 2: Per-adapter evaluation with multi-layer hidden states
    log("\n" + "=" * 70)
    log("[Phase 2] Per-adapter PPL + multi-layer hidden states")
    log("=" * 70)

    adapter_ppls = {}
    adapter_hiddens = {}  # {domain: {layer: [hidden_vecs]}}
    for domain in DOMAINS:
        ppls, hiddens = phase_adapter_eval(
            model, tokenizer, base_weights, samples, domain
        )
        adapter_ppls[domain] = ppls
        adapter_hiddens[domain] = hiddens
        log_memory(f"post-{domain}")

    # Restore base weights
    restore_base_weights(model, base_weights)

    # Phase 3: Sheaf cohomology analysis
    analysis = phase_analysis(samples, base_ppls, adapter_ppls, adapter_hiddens)

    # Free model
    cleanup(model, tokenizer)
    del base_weights, adapter_hiddens
    gc.collect()
    mx.clear_cache()
    log_memory("post-cleanup")

    # Save results
    total_time = time.time() - t_start
    final_results = {
        "experiment": "sheaf_cohomology_dim",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_samples": len(samples),
        "extract_layers": EXTRACT_LAYERS,
        "domain_scales": DOMAIN_SCALES,
        "top_k_values": TOP_K_VALUES,
        **analysis,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(final_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
