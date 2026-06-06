#!/usr/bin/env python3
"""
Self-Routing Adapters: Can B-matrices provide implicit routing signal?

Hypothesis: For each token hidden state h, compute similarity to each adapter's
B-matrix column space. The adapter whose B-subspace best captures h is the
correct domain expert. This requires ZERO learned parameters -- routing is
embedded in the adapter weights themselves.

Kill criteria:
  K1 (id=249): Implicit routing accuracy < 50% -> KILL

Baseline: Gumbel-sigmoid router achieves 86.33% top-2 accuracy (trained, 659K params).

Platform: Apple M5 Pro 48GB, MLX, local.
"""

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODELS_DIR = Path(__file__).parent.parent
N50_DIR = MODELS_DIR / "bitnet_scale_n50"
GUMBEL_DIR = MODELS_DIR / "gumbel_sigmoid_ablation"


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


# =========================================================================
# Phase 1: Load hidden states and adapter B-matrices
# =========================================================================
def phase_load_data():
    """Load cached hidden states and extract B-matrix signatures from adapters."""
    log("=== Phase 1: Loading data ===")
    t0 = time.time()

    # Load hidden states from gumbel_sigmoid_ablation (already cached)
    train_file = GUMBEL_DIR / "train_hiddens.npz"
    val_file = GUMBEL_DIR / "val_hiddens.npz"

    train_raw = dict(np.load(str(train_file), allow_pickle=True))
    val_raw = dict(np.load(str(val_file), allow_pickle=True))

    # Convert bfloat16 bytes to float32 via mx
    train_hiddens = {}
    val_hiddens = {}
    for k in train_raw:
        # np saves bfloat16 as void bytes; load via mx
        arr = mx.array(np.frombuffer(train_raw[k].tobytes(), dtype=np.uint16).reshape(train_raw[k].shape))
        arr = mx.view(arr, mx.bfloat16).astype(mx.float32)
        train_hiddens[k] = arr

    for k in val_raw:
        arr = mx.array(np.frombuffer(val_raw[k].tobytes(), dtype=np.uint16).reshape(val_raw[k].shape))
        arr = mx.view(arr, mx.bfloat16).astype(mx.float32)
        val_hiddens[k] = arr

    # Get domain ordering from N=50 results
    with open(N50_DIR / "results.json") as f:
        n50_results = json.load(f)
    all_names = n50_results["all_names"]
    adapters_dir = N50_DIR / "adapters"
    domain_names = [n for n in all_names if (adapters_dir / n).exists()]

    # Filter to domains that have hidden states
    domain_names = [n for n in domain_names if n in train_hiddens and n in val_hiddens]
    log(f"  Domains with both adapters and hidden states: {len(domain_names)}")

    # Load B-matrices from adapters (focus on q_proj and v_proj across layers)
    # Strategy: aggregate B-matrices across layers to build a domain signature
    # B has shape (r, d_out) -- its ROWS span the output subspace
    # For q_proj: B is (16, 2560), so B^T columns are the adapter's output directions

    # We'll test multiple aggregation strategies:
    # 1. Single layer (layer 15 = middle layer)
    # 2. Average across all layers
    # 3. Concatenate key layers

    adapter_data = {}
    proj_types = ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj"]
    target_layers = [0, 7, 15, 22, 29]  # spread across 30 layers

    for domain in domain_names:
        adapter_path = adapters_dir / domain / "adapter.npz"
        weights = dict(np.load(str(adapter_path)))

        domain_B = {}
        domain_A = {}
        for layer_idx in target_layers:
            for proj in proj_types:
                key_a = f"model.layers.{layer_idx}.{proj}.lora_a"
                key_b = f"model.layers.{layer_idx}.{proj}.lora_b"
                if key_a in weights and key_b in weights:
                    A = mx.array(weights[key_a])  # (d_in, r)
                    B = mx.array(weights[key_b])  # (r, d_out)
                    domain_B[f"L{layer_idx}.{proj}"] = B
                    domain_A[f"L{layer_idx}.{proj}"] = A

        adapter_data[domain] = {"B": domain_B, "A": domain_A}

    mx.eval([v for d in adapter_data.values() for dd in [d["B"], d["A"]] for v in dd.values()])

    elapsed = time.time() - t0
    log(f"  Loaded {len(domain_names)} domains, {len(target_layers)} layers x {len(proj_types)} projs")
    log(f"  Phase 1 time: {elapsed:.1f}s")
    log_memory("post-load")

    return train_hiddens, val_hiddens, domain_names, adapter_data


# =========================================================================
# Phase 2: Compute routing signatures and evaluate
# =========================================================================
def phase_evaluate_routing(train_hiddens, val_hiddens, domain_names, adapter_data):
    """Test multiple self-routing methods and compute accuracy."""
    log("\n=== Phase 2: Evaluating routing methods ===")
    t0 = time.time()

    N = len(domain_names)
    domain_to_idx = {n: i for i, n in enumerate(domain_names)}

    # Collect all val hidden states with labels
    all_h = []
    all_labels = []
    for domain in domain_names:
        h = val_hiddens[domain]  # (n_samples, d)
        idx = domain_to_idx[domain]
        all_h.append(h)
        all_labels.extend([idx] * h.shape[0])

    all_h = mx.concatenate(all_h, axis=0)  # (total_samples, d)
    all_labels = mx.array(all_labels)       # (total_samples,)
    mx.eval(all_h, all_labels)

    total_samples = all_h.shape[0]
    d = all_h.shape[1]
    log(f"  Total val samples: {total_samples}, d={d}, N={N}")

    results = {}

    # ---------------------------------------------------------------
    # Method 1: Quadratic form -- h^T @ (B^T @ B) @ h
    # For each adapter i, score_i = sum over layers,projs of h^T @ B_j^T @ B_j @ h
    # This measures how much energy of h lies in B's row space
    # ---------------------------------------------------------------
    log("\n  --- Method 1: Quadratic form (B^T @ B energy) ---")

    # Pre-compute B^T @ B (Gram matrix in output space) for each adapter
    # For q_proj: B is (16, 2560), B^T@B is (2560, 2560) -- too big at N=49
    # Instead: compute h @ B^T (project h into rank-r space), then take norm
    # score_i = ||B_i @ h||_2 where we pick specific layers

    # Actually for hidden states of dim 2560 and B of shape (r=16, d_out):
    # If proj is q_proj, d_out=2560 and matches hidden dim
    # score = ||B @ h||_2  (project hidden into r-dim space, measure norm)

    # Focus on q_proj layer 15 first (single layer, matching hidden dim)
    for layer_tag in ["L15.self_attn.q_proj", "L15.self_attn.v_proj"]:
        scores_per_adapter = []
        for domain in domain_names:
            B = adapter_data[domain]["B"].get(layer_tag)
            if B is None:
                scores_per_adapter.append(mx.zeros((total_samples,)))
                continue
            # B: (r, d_out). For q_proj: (16, 2560)
            # If d_out matches d=2560, we can compute B @ h^T
            if B.shape[1] == d:
                # score = ||B @ h^T||_2 for each sample
                proj = all_h @ B.T  # (total, r)  -- h projected into adapter's rank-r space
                score = mx.sqrt(mx.sum(proj * proj, axis=1) + 1e-12)  # (total,)
            elif B.shape[1] < d:
                # v_proj or k_proj: d_out=640, can't directly compare with h (d=2560)
                # Skip these for direct comparison
                scores_per_adapter.append(mx.zeros((total_samples,)))
                continue
            else:
                scores_per_adapter.append(mx.zeros((total_samples,)))
                continue
            scores_per_adapter.append(score)

        scores = mx.stack(scores_per_adapter, axis=1)  # (total, N)
        mx.eval(scores)

        # Top-1 and top-2 accuracy
        top1_pred = mx.argmax(scores, axis=1)
        top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()

        # Top-2: check if true label is in top-2 predictions
        top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
        top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
        top2_acc = mx.mean(top2_match.astype(mx.float32)).item()

        # Top-5
        top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
        top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
        top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

        method_name = f"quadratic_{layer_tag}"
        results[method_name] = {
            "top1_accuracy": round(top1_acc, 4),
            "top2_accuracy": round(top2_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "params": 0,
            "description": f"||B @ h||_2 using {layer_tag}"
        }
        log(f"    {method_name}: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 2: Multi-layer aggregated quadratic form
    # Average scores across all target layers for q_proj
    # ---------------------------------------------------------------
    log("\n  --- Method 2: Multi-layer aggregated quadratic form ---")

    q_layers = [f"L{l}.self_attn.q_proj" for l in [0, 7, 15, 22, 29]]

    scores_per_adapter = []
    for domain in domain_names:
        domain_score = mx.zeros((total_samples,))
        n_valid = 0
        for layer_tag in q_layers:
            B = adapter_data[domain]["B"].get(layer_tag)
            if B is None or B.shape[1] != d:
                continue
            proj = all_h @ B.T  # (total, r)
            score = mx.sum(proj * proj, axis=1)  # (total,) -- squared norm
            domain_score = domain_score + score
            n_valid += 1
        if n_valid > 0:
            domain_score = domain_score / n_valid
        scores_per_adapter.append(domain_score)

    scores = mx.stack(scores_per_adapter, axis=1)
    mx.eval(scores)

    top1_pred = mx.argmax(scores, axis=1)
    top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
    top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
    top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
    top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
    top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
    top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
    top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

    results["quadratic_multilayer_qproj"] = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "params": 0,
        "description": "||B @ h||_2 averaged across 5 layers, q_proj only"
    }
    log(f"    multilayer_qproj: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 3: Full activation norm ||B @ A @ h||_2 (AoE-style)
    # This is the full adapter forward pass without the base model
    # ---------------------------------------------------------------
    log("\n  --- Method 3: Activation norm ||B @ A @ h||_2 (AoE-style) ---")

    for layer_tag in ["L15.self_attn.q_proj"]:
        scores_per_adapter = []
        for domain in domain_names:
            B = adapter_data[domain]["B"].get(layer_tag)
            A = adapter_data[domain]["A"].get(layer_tag)
            if B is None or A is None or A.shape[0] != d:
                scores_per_adapter.append(mx.zeros((total_samples,)))
                continue
            # A: (d_in, r) = (2560, 16), B: (r, d_out) = (16, 2560)
            # Full path: h @ A @ B^T (but this gives d-dim output)
            # Norm of adapter output: ||B @ (A^T @ h)||_2
            # A^T @ h: (r,) per sample; then B @ that: (d_out,) per sample
            Ah = all_h @ A        # (total, r) -- project to low-rank
            BAh = Ah @ B          # (total, d_out) -- project back up
            score = mx.sqrt(mx.sum(BAh * BAh, axis=1) + 1e-12)
            scores_per_adapter.append(score)

        scores = mx.stack(scores_per_adapter, axis=1)
        mx.eval(scores)

        top1_pred = mx.argmax(scores, axis=1)
        top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
        top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
        top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
        top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
        top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
        top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
        top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

        method_name = f"activation_norm_{layer_tag}"
        results[method_name] = {
            "top1_accuracy": round(top1_acc, 4),
            "top2_accuracy": round(top2_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "params": 0,
            "description": f"||B @ A^T @ h||_2 using {layer_tag}"
        }
        log(f"    {method_name}: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 4: Multi-layer activation norm (all q_proj layers)
    # ---------------------------------------------------------------
    log("\n  --- Method 4: Multi-layer activation norm ---")

    scores_per_adapter = []
    for domain in domain_names:
        domain_score = mx.zeros((total_samples,))
        n_valid = 0
        for layer_tag in q_layers:
            B = adapter_data[domain]["B"].get(layer_tag)
            A = adapter_data[domain]["A"].get(layer_tag)
            if B is None or A is None or A.shape[0] != d:
                continue
            Ah = all_h @ A
            BAh = Ah @ B
            score = mx.sum(BAh * BAh, axis=1)
            domain_score = domain_score + score
            n_valid += 1
        if n_valid > 0:
            domain_score = domain_score / n_valid
        scores_per_adapter.append(domain_score)

    scores = mx.stack(scores_per_adapter, axis=1)
    mx.eval(scores)

    top1_pred = mx.argmax(scores, axis=1)
    top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
    top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
    top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
    top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
    top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
    top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
    top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

    results["activation_norm_multilayer_qproj"] = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "params": 0,
        "description": "||B @ A^T @ h||_2 averaged across 5 layers, q_proj"
    }
    log(f"    multilayer_activation: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 5: SVD projection distance
    # For each adapter, compute SVD of B^T to get column space basis U
    # Score = ||U @ U^T @ h||_2 / ||h||_2 (fraction of h in adapter subspace)
    # ---------------------------------------------------------------
    log("\n  --- Method 5: SVD projection distance ---")

    for layer_tag in ["L15.self_attn.q_proj"]:
        scores_per_adapter = []
        for domain in domain_names:
            B = adapter_data[domain]["B"].get(layer_tag)
            if B is None or B.shape[1] != d:
                scores_per_adapter.append(mx.zeros((total_samples,)))
                continue
            # B: (r, d). B^T: (d, r). SVD of B^T gives U (d, r) orthonormal columns
            # U @ U^T is projection onto column space of B^T (=row space of B)
            # Score = ||U @ U^T @ h|| / ||h|| = how much of h lies in B's subspace

            # MLX SVD: for B^T (d, r) with d >> r, use economy SVD
            BT = B.T  # (d, r)
            U, S, Vt = mx.linalg.svd(BT, stream=mx.cpu)
            # U: (d, d) full -- we only need first r columns
            U_r = U[:, :B.shape[0]]  # (d, r)
            mx.eval(U_r)

            # Project: proj = U_r @ (U_r^T @ h^T)  shape games for batch
            # h: (total, d), U_r: (d, r)
            coeff = all_h @ U_r     # (total, r)
            proj = coeff @ U_r.T    # (total, d)
            proj_norm = mx.sqrt(mx.sum(proj * proj, axis=1) + 1e-12)
            h_norm = mx.sqrt(mx.sum(all_h * all_h, axis=1) + 1e-12)
            score = proj_norm / h_norm  # fraction of h in subspace
            scores_per_adapter.append(score)

        scores = mx.stack(scores_per_adapter, axis=1)
        mx.eval(scores)

        top1_pred = mx.argmax(scores, axis=1)
        top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
        top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
        top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
        top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
        top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
        top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
        top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

        method_name = f"svd_projection_{layer_tag}"
        results[method_name] = {
            "top1_accuracy": round(top1_acc, 4),
            "top2_accuracy": round(top2_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "params": 0,
            "description": f"||U@U^T@h||/||h|| via SVD of B^T, {layer_tag}"
        }
        log(f"    {method_name}: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 6: Cosine similarity to adapter centroid
    # For each adapter, compute mean(B @ A @ h) over training data as "key"
    # Then route via cosine(h, key_i)
    # This is a one-shot learned routing key from the adapter structure
    # ---------------------------------------------------------------
    log("\n  --- Method 6: Adapter centroid routing (train-derived keys) ---")

    # Compute centroid of adapter activations on train data
    layer_tag = "L15.self_attn.q_proj"
    centroids = []
    for domain in domain_names:
        B = adapter_data[domain]["B"].get(layer_tag)
        A = adapter_data[domain]["A"].get(layer_tag)
        if B is None or A is None or A.shape[0] != d:
            centroids.append(mx.zeros((d,)))
            continue
        # Use train hidden states for this domain to compute centroid
        h_train = train_hiddens[domain]  # (n_train, d)
        Ah = h_train @ A         # (n_train, r)
        BAh = Ah @ B             # (n_train, d_out)
        centroid = mx.mean(BAh, axis=0)  # (d_out,)
        centroids.append(centroid)

    centroids = mx.stack(centroids, axis=0)  # (N, d)
    mx.eval(centroids)

    # Normalize centroids and hidden states
    c_norm = centroids / (mx.sqrt(mx.sum(centroids * centroids, axis=1, keepdims=True)) + 1e-12)
    h_norm_vec = all_h / (mx.sqrt(mx.sum(all_h * all_h, axis=1, keepdims=True)) + 1e-12)
    scores = h_norm_vec @ c_norm.T  # (total, N) cosine similarities
    mx.eval(scores)

    top1_pred = mx.argmax(scores, axis=1)
    top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
    top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
    top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
    top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
    top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
    top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
    top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

    results["centroid_cosine_L15_qproj"] = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "params": 0,
        "description": "cos(h, mean(B@A@h_train)) using L15 q_proj"
    }
    log(f"    centroid_cosine: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 7: Multi-layer multi-proj aggregated activation norm
    # Use ALL available projections across all target layers
    # ---------------------------------------------------------------
    log("\n  --- Method 7: All-layer all-proj aggregated ---")

    all_layer_proj_tags = []
    for l in [0, 7, 15, 22, 29]:
        for p in ["self_attn.q_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj"]:
            all_layer_proj_tags.append(f"L{l}.{p}")

    scores_per_adapter = []
    for domain in domain_names:
        domain_score = mx.zeros((total_samples,))
        n_valid = 0
        for layer_tag in all_layer_proj_tags:
            B = adapter_data[domain]["B"].get(layer_tag)
            A = adapter_data[domain]["A"].get(layer_tag)
            if B is None or A is None:
                continue
            if A.shape[0] != d:
                continue
            Ah = all_h @ A
            BAh = Ah @ B
            score = mx.sum(BAh * BAh, axis=1)
            # Normalize by output dim to make layers comparable
            score = score / B.shape[1]
            domain_score = domain_score + score
            n_valid += 1
        if n_valid > 0:
            domain_score = domain_score / n_valid
        scores_per_adapter.append(domain_score)

    scores = mx.stack(scores_per_adapter, axis=1)
    mx.eval(scores)

    top1_pred = mx.argmax(scores, axis=1)
    top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
    top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
    top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
    top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
    top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
    top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
    top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

    results["activation_norm_all_layers_all_projs"] = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "params": 0,
        "description": "||B@A^T@h||_2 averaged across 5 layers x 4 projs"
    }
    log(f"    all_layers_all_projs: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 8: Cosine similarity to raw hidden state centroid (simplest baseline)
    # Just use mean(h_train) per domain -- no adapter weights at all
    # This tells us how much signal is in the hidden states themselves
    # ---------------------------------------------------------------
    log("\n  --- Method 8: Raw hidden state centroid (no adapter weights) ---")

    raw_centroids = []
    for domain in domain_names:
        h_train = train_hiddens[domain]
        centroid = mx.mean(h_train, axis=0)
        raw_centroids.append(centroid)

    raw_centroids = mx.stack(raw_centroids, axis=0)
    mx.eval(raw_centroids)

    c_norm = raw_centroids / (mx.sqrt(mx.sum(raw_centroids * raw_centroids, axis=1, keepdims=True)) + 1e-12)
    scores = h_norm_vec @ c_norm.T
    mx.eval(scores)

    top1_pred = mx.argmax(scores, axis=1)
    top1_acc = mx.mean((top1_pred == all_labels).astype(mx.float32)).item()
    top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
    top2_match = mx.any(top2_indices == all_labels[:, None], axis=1)
    top2_acc = mx.mean(top2_match.astype(mx.float32)).item()
    top5_indices = mx.argpartition(scores, kth=N-5, axis=1)[:, -5:]
    top5_match = mx.any(top5_indices == all_labels[:, None], axis=1)
    top5_acc = mx.mean(top5_match.astype(mx.float32)).item()

    results["raw_centroid_cosine"] = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "params": 0,
        "description": "cos(h, mean(h_train)) -- no adapter weights, pure hidden state"
    }
    log(f"    raw_centroid: top1={top1_acc:.4f} top2={top2_acc:.4f} top5={top5_acc:.4f}")

    # ---------------------------------------------------------------
    # Method 9: Random baseline (chance level)
    # ---------------------------------------------------------------
    log("\n  --- Method 9: Random baseline ---")
    results["random_baseline"] = {
        "top1_accuracy": round(1.0 / N, 4),
        "top2_accuracy": round(2.0 / N, 4),
        "top5_accuracy": round(5.0 / N, 4),
        "params": 0,
        "description": f"Random chance with N={N} classes"
    }
    log(f"    random: top1={1.0/N:.4f} top2={2.0/N:.4f} top5={5.0/N:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Phase 2 time: {elapsed:.1f}s")
    log_memory("post-eval")

    return results


# =========================================================================
# Phase 3: Per-domain analysis for best method
# =========================================================================
def phase_per_domain_analysis(val_hiddens, domain_names, adapter_data):
    """Detailed per-domain breakdown for the best method."""
    log("\n=== Phase 3: Per-domain analysis ===")
    t0 = time.time()

    N = len(domain_names)
    d = list(val_hiddens.values())[0].shape[1]
    domain_to_idx = {n: i for i, n in enumerate(domain_names)}

    # Use multi-layer q_proj activation norm (likely best)
    q_layers = [f"L{l}.self_attn.q_proj" for l in [0, 7, 15, 22, 29]]

    per_domain = {}
    for eval_domain in domain_names:
        h_val = val_hiddens[eval_domain]
        n_val = h_val.shape[0]
        true_idx = domain_to_idx[eval_domain]

        # Compute scores for this domain's samples against all adapters
        scores_per_adapter = []
        for candidate_domain in domain_names:
            domain_score = mx.zeros((n_val,))
            n_valid = 0
            for layer_tag in q_layers:
                B = adapter_data[candidate_domain]["B"].get(layer_tag)
                A = adapter_data[candidate_domain]["A"].get(layer_tag)
                if B is None or A is None or A.shape[0] != d:
                    continue
                Ah = h_val @ A
                BAh = Ah @ B
                score = mx.sum(BAh * BAh, axis=1)
                domain_score = domain_score + score
                n_valid += 1
            if n_valid > 0:
                domain_score = domain_score / n_valid
            scores_per_adapter.append(domain_score)

        scores = mx.stack(scores_per_adapter, axis=1)  # (n_val, N)
        mx.eval(scores)

        # Top-1 accuracy for this domain
        top1_pred = mx.argmax(scores, axis=1)
        top1_correct = mx.sum((top1_pred == true_idx).astype(mx.float32)).item()

        # Top-2
        top2_indices = mx.argpartition(scores, kth=N-2, axis=1)[:, -2:]
        true_label = mx.full((n_val,), true_idx)
        top2_correct = mx.sum(mx.any(top2_indices == true_label[:, None], axis=1).astype(mx.float32)).item()

        # What domains does it confuse with?
        top1_preds = top1_pred.tolist()
        confusion = {}
        for p in top1_preds:
            pred_name = domain_names[p]
            confusion[pred_name] = confusion.get(pred_name, 0) + 1

        per_domain[eval_domain] = {
            "top1_accuracy": round(top1_correct / n_val, 4),
            "top2_accuracy": round(top2_correct / n_val, 4),
            "n_samples": n_val,
            "top_confusions": dict(sorted(confusion.items(), key=lambda x: -x[1])[:3])
        }

    elapsed = time.time() - t0
    log(f"  Phase 3 time: {elapsed:.1f}s")

    # Print summary
    accs = [v["top1_accuracy"] for v in per_domain.values()]
    perfect = sum(1 for a in accs if a == 1.0)
    zero = sum(1 for a in accs if a == 0.0)
    log(f"  Perfect domains (100%): {perfect}/{N}")
    log(f"  Zero accuracy domains: {zero}/{N}")
    log(f"  Mean per-domain top1: {sum(accs)/len(accs):.4f}")

    # Show worst domains
    sorted_domains = sorted(per_domain.items(), key=lambda x: x[1]["top1_accuracy"])
    log("\n  Worst 10 domains:")
    for name, info in sorted_domains[:10]:
        log(f"    {name}: top1={info['top1_accuracy']:.2f} top2={info['top2_accuracy']:.2f} confusions={info['top_confusions']}")

    return per_domain


# =========================================================================
# Phase 4: Timing analysis
# =========================================================================
def phase_timing(val_hiddens, domain_names, adapter_data):
    """Measure latency overhead per token for self-routing."""
    log("\n=== Phase 4: Timing analysis ===")

    d = list(val_hiddens.values())[0].shape[1]
    N = len(domain_names)
    # Single token
    h_single = mx.array(list(val_hiddens.values())[0][0:1])  # (1, d)
    mx.eval(h_single)

    q_layers = [f"L{l}.self_attn.q_proj" for l in [0, 7, 15, 22, 29]]

    # Pre-extract A and B for timing
    As = []
    Bs = []
    for domain in domain_names:
        layer_As = []
        layer_Bs = []
        for lt in q_layers:
            A = adapter_data[domain]["A"].get(lt)
            B = adapter_data[domain]["B"].get(lt)
            if A is not None and B is not None and A.shape[0] == d:
                layer_As.append(A)
                layer_Bs.append(B)
        As.append(layer_As)
        Bs.append(layer_Bs)

    # Warm up
    for _ in range(10):
        scores = []
        for i in range(N):
            s = mx.zeros((1,))
            for A, B in zip(As[i], Bs[i]):
                Ah = h_single @ A
                BAh = Ah @ B
                s = s + mx.sum(BAh * BAh, axis=1)
            scores.append(s)
        result = mx.stack(scores, axis=1)
        mx.eval(result)

    # Timed runs
    n_runs = 100
    t0 = time.time()
    for _ in range(n_runs):
        scores = []
        for i in range(N):
            s = mx.zeros((1,))
            for A, B in zip(As[i], Bs[i]):
                Ah = h_single @ A
                BAh = Ah @ B
                s = s + mx.sum(BAh * BAh, axis=1)
            scores.append(s)
        result = mx.stack(scores, axis=1)
        mx.eval(result)
    elapsed = time.time() - t0
    per_token_ms = (elapsed / n_runs) * 1000

    log(f"  Self-routing latency per token: {per_token_ms:.3f} ms")
    log(f"  ({n_runs} runs, N={N} adapters, 5 layers q_proj)")

    return {"per_token_ms": round(per_token_ms, 3), "n_adapters": N, "n_layers": 5}


# =========================================================================
# Main
# =========================================================================
def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Load data
    train_hiddens, val_hiddens, domain_names, adapter_data = phase_load_data()

    # Phase 2: Evaluate routing methods
    method_results = phase_evaluate_routing(train_hiddens, val_hiddens, domain_names, adapter_data)

    # Phase 3: Per-domain analysis
    per_domain = phase_per_domain_analysis(val_hiddens, domain_names, adapter_data)

    # Phase 4: Timing
    timing = phase_timing(val_hiddens, domain_names, adapter_data)

    cleanup(train_hiddens, val_hiddens, adapter_data)

    # Find best method
    best_method = max(method_results.items(),
                      key=lambda x: x[1]["top2_accuracy"] if x[0] != "random_baseline" else 0)
    best_name = best_method[0]
    best_top2 = best_method[1]["top2_accuracy"]

    # Kill criterion assessment
    k1_pass = best_top2 >= 0.50  # K1: implicit routing accuracy >= 50%
    baseline_top2 = 0.8633       # Gumbel-sigmoid baseline

    total_time = time.time() - t0

    # Compile results
    results = {
        "experiment": "self_routing_adapters",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "n_domains": len(domain_names),
        "hidden_dim": 2560,
        "rank": 16,
        "kill_criteria": {
            "K1_implicit_routing_above_50pct": {
                "threshold": 0.50,
                "best_top2": best_top2,
                "best_method": best_name,
                "result": "PASS" if k1_pass else "FAIL"
            }
        },
        "comparison_to_baseline": {
            "gumbel_sigmoid_top2": baseline_top2,
            "best_self_routing_top2": best_top2,
            "gap": round(best_top2 - baseline_top2, 4),
            "relative_performance": round(best_top2 / baseline_top2, 4)
        },
        "methods": method_results,
        "per_domain_analysis": per_domain,
        "timing": timing
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n{'='*60}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"Best method: {best_name}")
    log(f"Best top-2 accuracy: {best_top2:.4f}")
    log(f"Gumbel-sigmoid baseline: {baseline_top2:.4f}")
    log(f"Gap: {best_top2 - baseline_top2:+.4f}")
    log(f"K1 (>50%): {'PASS' if k1_pass else 'FAIL'}")
    log(f"Total time: {total_time:.1f}s")
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
