#!/usr/bin/env python3
"""
Hierarchical Two-Stage Routing at N=24.

Two-stage: cluster 24 domains into K~5 groups by confusion similarity,
train cluster-level router (stage 1) and within-cluster routers (stage 2).
Each stage faces N<=5, exploiting the proven N=5 routing mechanism (Finding #179).

Kill criteria:
  K593: Top-1 routing accuracy >=60% at N=24 (vs 39.6% best flat)
  K594: Routed PPL < uniform PPL at N=24
  K595: Total two-stage routing overhead <15%

Type: frontier-extension
Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 20
SEED = 42

# Routing head config
ROUTER_HIDDEN_DIM = 64
ROUTER_LR = 3e-4
ROUTER_TRAIN_STEPS = 2000
ROUTER_BATCH_SIZE = 16

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Pre-trained adapters and data from real_data_25_domain_adapters
REAL_DATA_DIR = Path(__file__).parent.parent / "real_data_25_domain_adapters"
ADAPTERS_SOURCE_DIR = REAL_DATA_DIR / "adapters"
DATA_DIR = REAL_DATA_DIR / "data"

# 24 domains that have both adapters and data
DOMAINS = sorted([
    d.name for d in ADAPTERS_SOURCE_DIR.iterdir()
    if d.is_dir() and (DATA_DIR / d.name).exists()
])
N_DOMAINS = len(DOMAINS)
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}

# Hidden state cache config
HIDDEN_CACHE_TRAIN_PER_DOMAIN = 40
HIDDEN_CACHE_VAL_PER_DOMAIN = 20


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


# ===========================================================================
# Model utilities (reused from prior experiments)
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


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


def apply_lora_to_model(model, rank=16, scale=1.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_texts(domain_name, split="valid"):
    fpath = DATA_DIR / domain_name / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


# ===========================================================================
# PPL computation
# ===========================================================================
def compute_ppl(model, tokenizer, texts, max_batches=20):
    total_loss = 0.0
    total_tokens = 0
    for text in texts[:max_batches]:
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


def get_hidden_states(model, x):
    """Extract hidden states from the last layer (mean-pooled)."""
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h)
    h = model.model.norm(h)
    return h


# ===========================================================================
# Routing head (same architecture for stage-1 and stage-2)
# ===========================================================================
class MultiClassRouter(nn.Module):
    """Multi-class routing head: h_pool -> K logits -> softmax."""

    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ===========================================================================
# Phase 1: Cache hidden states for all 24 domains
# ===========================================================================
def phase_cache_hidden_states(model_id):
    """Load base model, extract and cache mean-pooled hidden states."""
    log("\n" + "=" * 70)
    log("[Phase 1] Caching hidden states for 24 domains")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")
    log(f"  N_domains = {N_DOMAINS}: {', '.join(DOMAINS)}")

    train_data = []  # list of (mx.array shape (1, d), int label)
    val_data = []

    for i, domain in enumerate(DOMAINS):
        label = DOMAIN_TO_IDX[domain]
        for split, data_list, max_s in [
            ("train", train_data, HIDDEN_CACHE_TRAIN_PER_DOMAIN),
            ("valid", val_data, HIDDEN_CACHE_VAL_PER_DOMAIN),
        ]:
            texts = load_domain_texts(domain, split=split)
            for text in texts[:max_s]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = get_hidden_states(model, x)
                h_pool = mx.mean(h, axis=1)  # (1, d)
                mx.eval(h_pool)
                data_list.append((h_pool, label))
                del h, x

        if (i + 1) % 8 == 0:
            log(f"  Cached {i+1}/{N_DOMAINS} domains...")
            log_memory(f"after-{i+1}-domains")

    log(f"  Caching done in {time.time() - t0:.1f}s")
    log(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    log_memory("after-caching")

    cleanup(model, tokenizer)
    return train_data, val_data, d_model


# ===========================================================================
# Phase 2: Build confusion matrix + spectral clustering
# ===========================================================================
def phase_build_clusters(train_data, val_data, d_model):
    """Train flat router, extract confusion matrix, cluster domains."""
    log("\n" + "=" * 70)
    log("[Phase 2] Building confusion matrix and clustering domains")
    log("=" * 70)

    t0 = time.time()

    # Step 1: Train flat 24-class router (same as centralized_multiclass experiment)
    log("  Training flat 24-class router for confusion matrix...")
    flat_router = MultiClassRouter(d_model, N_DOMAINS, ROUTER_HIDDEN_DIM)
    mx.eval(flat_router.parameters())
    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    def router_loss_fn(router, h_batch, labels_batch):
        logits = router(h_batch)
        return nn.losses.cross_entropy(logits, labels_batch, reduction="mean")

    router_loss_and_grad = nn.value_and_grad(flat_router, router_loss_fn)

    gc.disable()
    for step in range(ROUTER_TRAIN_STEPS):
        batch_indices = rng.sample(range(len(train_data)), min(ROUTER_BATCH_SIZE, len(train_data)))
        h_list = [train_data[idx][0] for idx in batch_indices]
        label_list = [train_data[idx][1] for idx in batch_indices]
        h_batch = mx.concatenate(h_list, axis=0)
        labels_batch = mx.array(label_list)
        loss, grads = router_loss_and_grad(flat_router, h_batch, labels_batch)
        optimizer.update(flat_router, grads)
        mx.eval(flat_router.parameters(), optimizer.state, loss)
    gc.enable()
    gc.collect()

    # Step 2: Extract confusion matrix from validation set
    log("  Extracting confusion matrix from validation set...")
    confusion = np.zeros((N_DOMAINS, N_DOMAINS), dtype=np.float64)
    flat_correct = 0
    flat_total = 0

    for h_pool, label in val_data:
        logits = flat_router(h_pool)
        mx.eval(logits)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        probs_np = np.array(probs[0].tolist())
        pred = int(np.argmax(probs_np))
        confusion[label, pred] += 1
        if pred == label:
            flat_correct += 1
        flat_total += 1

    flat_accuracy = flat_correct / flat_total
    log(f"  Flat router accuracy: {flat_accuracy:.1%} ({flat_correct}/{flat_total})")

    # Step 3: Build confusion distance matrix (symmetric)
    # Confusion similarity: how often i gets confused WITH j (bidirectional)
    # Higher confusion = more similar = should be in same cluster
    confusion_sym = confusion + confusion.T
    np.fill_diagonal(confusion_sym, 0)

    # Also compute centroid distances for a complementary view
    # Gather per-domain centroids
    domain_centroids = {}
    for h_pool, label in train_data:
        domain = DOMAINS[label]
        if domain not in domain_centroids:
            domain_centroids[domain] = []
        domain_centroids[domain].append(np.array(h_pool[0].tolist()))

    centroid_matrix = np.zeros((N_DOMAINS, d_model))
    for i, domain in enumerate(DOMAINS):
        vecs = np.array(domain_centroids[domain])
        centroid_matrix[i] = vecs.mean(axis=0)

    # Cosine distance between centroids
    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_normed = centroid_matrix / (norms + 1e-8)
    cos_sim = centroid_normed @ centroid_normed.T
    cos_dist = 1 - cos_sim  # distance: 0=identical, 2=opposite

    # Step 4: Hierarchical clustering using centroid cosine distance
    # Domains close in representation space should be in the same cluster
    # (these are the ones that confuse, and we proved misrouting among them is benign)
    condensed_dist = squareform(cos_dist, checks=False)
    Z = linkage(condensed_dist, method='ward')

    # Try K=4,5,6 and pick the one with best eigengap / balance
    best_K = None
    best_clusters = None
    best_score = -1

    for K in [4, 5, 6]:
        labels = fcluster(Z, t=K, criterion='maxclust')
        # Score: product of (cluster_size >= 2) * (cluster_size <= 8) * balance
        sizes = [np.sum(labels == c) for c in range(1, K + 1)]
        min_size = min(sizes)
        max_size = max(sizes)
        if min_size < 2:
            continue
        balance = min_size / max_size
        score = balance * K  # prefer more clusters with good balance
        log(f"  K={K}: sizes={sizes}, balance={balance:.2f}, score={score:.2f}")
        if score > best_score:
            best_score = score
            best_K = K
            best_clusters = labels

    if best_clusters is None:
        # Fallback: K=5
        best_K = 5
        best_clusters = fcluster(Z, t=5, criterion='maxclust')

    log(f"  Selected K={best_K}")

    # Build cluster mapping
    clusters = {}  # cluster_id -> list of domain names
    domain_to_cluster = {}  # domain -> cluster_id
    for i, domain in enumerate(DOMAINS):
        c = int(best_clusters[i])
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(domain)
        domain_to_cluster[domain] = c

    for c_id in sorted(clusters.keys()):
        members = clusters[c_id]
        log(f"  Cluster {c_id} ({len(members)}): {', '.join(members)}")

    # Step 5: Compute within-cluster and cross-cluster confusion
    intra_confusion = 0
    inter_confusion = 0
    total_misrouted = 0
    for i in range(N_DOMAINS):
        for j in range(N_DOMAINS):
            if i == j:
                continue
            count = confusion[i, j]
            if count == 0:
                continue
            total_misrouted += count
            if domain_to_cluster[DOMAINS[i]] == domain_to_cluster[DOMAINS[j]]:
                intra_confusion += count
            else:
                inter_confusion += count

    if total_misrouted > 0:
        log(f"  Intra-cluster confusion: {intra_confusion:.0f} ({intra_confusion/total_misrouted:.1%})")
        log(f"  Inter-cluster confusion: {inter_confusion:.0f} ({inter_confusion/total_misrouted:.1%})")

    cleanup(flat_router, optimizer, router_loss_and_grad)

    cluster_info = {
        "K": best_K,
        "clusters": {str(c): members for c, members in clusters.items()},
        "domain_to_cluster": domain_to_cluster,
        "flat_accuracy": round(flat_accuracy, 4),
        "intra_confusion_frac": round(intra_confusion / max(total_misrouted, 1), 4),
        "cos_dist_stats": {
            "mean": round(float(np.mean(cos_dist[np.triu_indices(N_DOMAINS, k=1)])), 4),
            "min": round(float(np.min(cos_dist[np.triu_indices(N_DOMAINS, k=1)])), 4),
            "max": round(float(np.max(cos_dist[np.triu_indices(N_DOMAINS, k=1)])), 4),
        },
    }

    log(f"  Clustering done in {time.time() - t0:.1f}s")
    return clusters, domain_to_cluster, cluster_info


# ===========================================================================
# Phase 3: Train hierarchical routers (stage-1 + K stage-2)
# ===========================================================================
def phase_train_hierarchical_routers(train_data, val_data, d_model, clusters, domain_to_cluster):
    """Train stage-1 cluster router and K stage-2 within-cluster routers."""
    log("\n" + "=" * 70)
    log("[Phase 3] Training hierarchical routers")
    log("=" * 70)

    t0 = time.time()
    K = len(clusters)
    cluster_ids = sorted(clusters.keys())
    cluster_to_idx = {c: i for i, c in enumerate(cluster_ids)}

    # -----------------------------------------------------------------------
    # Stage 1: Cluster-level router
    # -----------------------------------------------------------------------
    log(f"\n  --- Stage 1: {K}-class cluster router ---")

    # Prepare cluster-level training data
    cluster_train = []
    cluster_val = []
    for h_pool, label in train_data:
        domain = DOMAINS[label]
        c_idx = cluster_to_idx[domain_to_cluster[domain]]
        cluster_train.append((h_pool, c_idx))
    for h_pool, label in val_data:
        domain = DOMAINS[label]
        c_idx = cluster_to_idx[domain_to_cluster[domain]]
        cluster_val.append((h_pool, c_idx))

    stage1_router = MultiClassRouter(d_model, K, ROUTER_HIDDEN_DIM)
    mx.eval(stage1_router.parameters())
    stage1_params = sum(p.size for _, p in tree_flatten(stage1_router.parameters()))
    log(f"  Stage-1 router params: {stage1_params:,}")

    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    def loss_fn(router, h_batch, labels_batch):
        logits = router(h_batch)
        return nn.losses.cross_entropy(logits, labels_batch, reduction="mean")

    loss_and_grad = nn.value_and_grad(stage1_router, loss_fn)

    gc.disable()
    for step in range(ROUTER_TRAIN_STEPS):
        batch_indices = rng.sample(range(len(cluster_train)), min(ROUTER_BATCH_SIZE, len(cluster_train)))
        h_list = [cluster_train[idx][0] for idx in batch_indices]
        label_list = [cluster_train[idx][1] for idx in batch_indices]
        h_batch = mx.concatenate(h_list, axis=0)
        labels_batch = mx.array(label_list)
        loss_val, grads = loss_and_grad(stage1_router, h_batch, labels_batch)
        optimizer.update(stage1_router, grads)
        mx.eval(stage1_router.parameters(), optimizer.state, loss_val)
    gc.enable()
    gc.collect()

    # Evaluate stage-1
    s1_correct = 0
    s1_total = 0
    for h_pool, c_label in cluster_val:
        logits = stage1_router(h_pool)
        mx.eval(logits)
        pred = int(mx.argmax(logits, axis=-1).item())
        if pred == c_label:
            s1_correct += 1
        s1_total += 1

    s1_accuracy = s1_correct / s1_total
    log(f"  Stage-1 cluster accuracy: {s1_accuracy:.1%} ({s1_correct}/{s1_total})")

    del optimizer, loss_and_grad
    gc.collect()
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # Stage 2: Within-cluster routers (one per cluster)
    # -----------------------------------------------------------------------
    log(f"\n  --- Stage 2: {K} within-cluster routers ---")

    stage2_routers = {}
    stage2_results = {}

    for c_id in cluster_ids:
        members = clusters[c_id]
        n_members = len(members)
        member_to_local = {d: i for i, d in enumerate(members)}

        log(f"\n  Cluster {c_id} ({n_members} members): {', '.join(members)}")

        if n_members == 1:
            log(f"    Singleton cluster, no router needed")
            stage2_results[c_id] = {"n_members": 1, "accuracy": 1.0}
            continue

        # Prepare within-cluster data
        c_train = []
        c_val = []
        for h_pool, label in train_data:
            domain = DOMAINS[label]
            if domain in member_to_local:
                c_train.append((h_pool, member_to_local[domain]))
        for h_pool, label in val_data:
            domain = DOMAINS[label]
            if domain in member_to_local:
                c_val.append((h_pool, member_to_local[domain]))

        router = MultiClassRouter(d_model, n_members, ROUTER_HIDDEN_DIM)
        mx.eval(router.parameters())
        r_params = sum(p.size for _, p in tree_flatten(router.parameters()))

        optimizer = opt.Adam(learning_rate=ROUTER_LR)
        rng2 = random.Random(SEED + c_id)

        loss_and_grad2 = nn.value_and_grad(router, loss_fn)

        # More steps for smaller clusters (better convergence)
        steps = max(ROUTER_TRAIN_STEPS, 2000)

        gc.disable()
        for step in range(steps):
            batch_indices = rng2.sample(range(len(c_train)), min(ROUTER_BATCH_SIZE, len(c_train)))
            h_list = [c_train[idx][0] for idx in batch_indices]
            label_list = [c_train[idx][1] for idx in batch_indices]
            h_batch = mx.concatenate(h_list, axis=0)
            labels_batch = mx.array(label_list)
            loss_val, grads = loss_and_grad2(router, h_batch, labels_batch)
            optimizer.update(router, grads)
            mx.eval(router.parameters(), optimizer.state, loss_val)
        gc.enable()
        gc.collect()

        # Evaluate within-cluster
        s2_correct = 0
        s2_total = 0
        for h_pool, local_label in c_val:
            logits = router(h_pool)
            mx.eval(logits)
            pred = int(mx.argmax(logits, axis=-1).item())
            if pred == local_label:
                s2_correct += 1
            s2_total += 1

        s2_acc = s2_correct / s2_total if s2_total > 0 else 0
        log(f"    Within-cluster accuracy: {s2_acc:.1%} ({s2_correct}/{s2_total})")
        log(f"    Router params: {r_params:,}")

        stage2_routers[c_id] = (router, members, member_to_local)
        stage2_results[c_id] = {
            "n_members": n_members,
            "accuracy": round(s2_acc, 4),
            "n_params": r_params,
        }

        del optimizer, loss_and_grad2
        gc.collect()
        mx.clear_cache()

    total_params = stage1_params + sum(
        r.get("n_params", 0) for r in stage2_results.values()
    )
    log(f"\n  Total hierarchical router params: {total_params:,}")
    log(f"  Training done in {time.time() - t0:.1f}s")

    router_info = {
        "stage1_accuracy": round(s1_accuracy, 4),
        "stage1_params": stage1_params,
        "stage2_results": {str(k): v for k, v in stage2_results.items()},
        "total_params": total_params,
        "train_time_s": round(time.time() - t0, 1),
    }

    return stage1_router, stage2_routers, router_info


# ===========================================================================
# Phase 4: Evaluate hierarchical routing + PPL
# ===========================================================================
def phase_evaluate(model_id, stage1_router, stage2_routers, clusters, domain_to_cluster, val_data, d_model):
    """Evaluate hierarchical routing accuracy and composition PPL."""
    log("\n" + "=" * 70)
    log("[Phase 4] Evaluating hierarchical routing + PPL at N=24")
    log("=" * 70)

    t0 = time.time()
    cluster_ids = sorted(clusters.keys())
    cluster_to_idx = {c: i for i, c in enumerate(cluster_ids)}

    # -----------------------------------------------------------------------
    # 4A: Hierarchical routing accuracy
    # -----------------------------------------------------------------------
    log("\n  --- 4A: Hierarchical routing accuracy ---")

    correct_top1 = 0
    correct_cluster = 0
    correct_within_given_correct_cluster = 0
    total_correct_cluster = 0
    total = 0
    per_domain_correct = {d: 0 for d in DOMAINS}
    per_domain_total = {d: 0 for d in DOMAINS}

    for h_pool, label in val_data:
        true_domain = DOMAINS[label]
        true_cluster = domain_to_cluster[true_domain]
        true_c_idx = cluster_to_idx[true_cluster]

        # Stage 1: predict cluster
        s1_logits = stage1_router(h_pool)
        mx.eval(s1_logits)
        pred_c_idx = int(mx.argmax(s1_logits, axis=-1).item())
        pred_cluster = cluster_ids[pred_c_idx]

        cluster_correct = (pred_c_idx == true_c_idx)
        if cluster_correct:
            correct_cluster += 1

        # Stage 2: predict domain within predicted cluster
        if pred_cluster in stage2_routers:
            router, members, member_to_local = stage2_routers[pred_cluster]
            s2_logits = router(h_pool)
            mx.eval(s2_logits)
            pred_local = int(mx.argmax(s2_logits, axis=-1).item())
            pred_domain = members[pred_local]
        else:
            # Singleton cluster
            pred_domain = clusters[pred_cluster][0]

        if pred_domain == true_domain:
            correct_top1 += 1
            per_domain_correct[true_domain] += 1

        if cluster_correct:
            total_correct_cluster += 1
            if pred_domain == true_domain:
                correct_within_given_correct_cluster += 1

        per_domain_total[true_domain] += 1
        total += 1

    hierarchical_acc = correct_top1 / total
    cluster_acc = correct_cluster / total
    within_acc = correct_within_given_correct_cluster / max(total_correct_cluster, 1)

    log(f"  Hierarchical top-1 accuracy: {hierarchical_acc:.1%} ({correct_top1}/{total})")
    log(f"  Stage-1 cluster accuracy: {cluster_acc:.1%} ({correct_cluster}/{total})")
    log(f"  Stage-2 within-cluster accuracy (given correct cluster): {within_acc:.1%}")

    log("  Per-domain accuracy:")
    for domain in DOMAINS:
        if per_domain_total[domain] > 0:
            acc = per_domain_correct[domain] / per_domain_total[domain]
            cluster = domain_to_cluster[domain]
            log(f"    {domain:20s}: {acc:.0%} (cluster {cluster})")

    # -----------------------------------------------------------------------
    # 4B: Routing overhead
    # -----------------------------------------------------------------------
    log("\n  --- 4B: Routing overhead ---")

    # Measure base forward pass time
    model, tokenizer = load(model_id)
    model = replace_bitlinear_with_linear(model)
    apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    zero_adapter_in_model(model)

    sample_text = load_domain_texts(DOMAINS[0], "valid")[0]
    sample_tokens = tokenizer.encode(sample_text)[:MAX_SEQ_LENGTH]
    x_sample = mx.array(sample_tokens)[None, :]

    # Warm up
    for _ in range(3):
        _ = model(x_sample)
        mx.eval(_)

    # Base forward time
    n_timing = 20
    base_times = []
    for _ in range(n_timing):
        t_start = time.perf_counter()
        out = model(x_sample)
        mx.eval(out)
        base_times.append(time.perf_counter() - t_start)
        del out
    base_ms = np.median(base_times) * 1000

    # Routing time (stage-1 + stage-2)
    h = get_hidden_states(model, x_sample)
    h_pool = mx.mean(h, axis=1)
    mx.eval(h_pool)

    route_times = []
    for _ in range(n_timing):
        t_start = time.perf_counter()
        s1_logits = stage1_router(h_pool)
        mx.eval(s1_logits)
        pred_c = int(mx.argmax(s1_logits, axis=-1).item())
        pred_cluster = cluster_ids[pred_c]
        if pred_cluster in stage2_routers:
            router, _, _ = stage2_routers[pred_cluster]
            s2_logits = router(h_pool)
            mx.eval(s2_logits)
        route_times.append(time.perf_counter() - t_start)
    route_ms = np.median(route_times) * 1000

    overhead_pct = (route_ms / base_ms) * 100
    log(f"  Base forward: {base_ms:.2f} ms")
    log(f"  Routing (stage-1 + stage-2): {route_ms:.2f} ms")
    log(f"  Routing overhead: {overhead_pct:.2f}%")

    # -----------------------------------------------------------------------
    # 4C: Composition PPL (hierarchical vs uniform vs oracle)
    # -----------------------------------------------------------------------
    log("\n  --- 4C: Composition PPL ---")

    uniform_ppls = {}
    routed_ppls = {}
    oracle_ppls = {}
    base_ppls = {}

    # Load all adapter params
    adapter_params = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_SOURCE_DIR / domain
        if (adapter_path / "adapter.npz").exists():
            adapter_params[domain] = load_adapter(adapter_path)

    for eval_domain in DOMAINS:
        eval_texts = load_domain_texts(eval_domain, "valid")[:VAL_BATCHES]
        if not eval_texts:
            continue

        # Base PPL (no adapter)
        zero_adapter_in_model(model)
        base_ppl = compute_ppl(model, tokenizer, eval_texts, max_batches=VAL_BATCHES)
        base_ppls[eval_domain] = base_ppl

        # Oracle PPL (correct adapter)
        if eval_domain in adapter_params:
            apply_adapter_to_model(model, adapter_params[eval_domain])
            oracle_ppl = compute_ppl(model, tokenizer, eval_texts, max_batches=VAL_BATCHES)
            oracle_ppls[eval_domain] = oracle_ppl
            zero_adapter_in_model(model)

        # Uniform composition (all 24 weighted equally)
        uniform_merged = {}
        n_adapters = len(adapter_params)
        for name in list(adapter_params.values())[0].keys():
            stacked = [adapter_params[d][name] for d in adapter_params if name in adapter_params[d]]
            uniform_merged[name] = sum(stacked) * (1.0 / n_adapters)
        apply_adapter_to_model(model, uniform_merged)
        uniform_ppl = compute_ppl(model, tokenizer, eval_texts, max_batches=VAL_BATCHES)
        uniform_ppls[eval_domain] = uniform_ppl
        zero_adapter_in_model(model)

        # Hierarchical routing: use cached hidden states to pick adapter
        # For each validation sample, route and aggregate
        # Simplified: use the most common predicted domain across val samples
        # (since we're evaluating per-domain PPL, we use the routed adapter for this domain)
        domain_votes = {}
        for text in eval_texts[:10]:  # Quick voting on first 10
            tokens = tokenizer.encode(text)[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            h = get_hidden_states(model, x)
            h_p = mx.mean(h, axis=1)
            mx.eval(h_p)

            s1_logits = stage1_router(h_p)
            mx.eval(s1_logits)
            pred_c = int(mx.argmax(s1_logits, axis=-1).item())
            pred_cluster = cluster_ids[pred_c]

            if pred_cluster in stage2_routers:
                router_s2, members, _ = stage2_routers[pred_cluster]
                s2_logits = router_s2(h_p)
                mx.eval(s2_logits)
                pred_local = int(mx.argmax(s2_logits, axis=-1).item())
                pred_domain = members[pred_local]
            else:
                pred_domain = clusters[pred_cluster][0]

            domain_votes[pred_domain] = domain_votes.get(pred_domain, 0) + 1
            del h, h_p, x

        # Use the most-voted adapter
        routed_domain = max(domain_votes, key=domain_votes.get)
        if routed_domain in adapter_params:
            apply_adapter_to_model(model, adapter_params[routed_domain])
            routed_ppl = compute_ppl(model, tokenizer, eval_texts, max_batches=VAL_BATCHES)
        else:
            routed_ppl = base_ppl
        routed_ppls[eval_domain] = routed_ppl
        zero_adapter_in_model(model)

        log(f"  {eval_domain:20s}: base={base_ppl:.2f} oracle={oracle_ppls.get(eval_domain, 0):.2f} "
            f"uniform={uniform_ppl:.2f} routed={routed_ppl:.2f} "
            f"(voted={routed_domain})")

    # Compute aggregates
    domains_with_all = [d for d in DOMAINS if d in base_ppls and d in oracle_ppls
                        and d in uniform_ppls and d in routed_ppls]

    avg_base = np.mean([base_ppls[d] for d in domains_with_all])
    avg_oracle = np.mean([oracle_ppls[d] for d in domains_with_all])
    avg_uniform = np.mean([uniform_ppls[d] for d in domains_with_all])
    avg_routed = np.mean([routed_ppls[d] for d in domains_with_all])

    gamma_uniform = avg_uniform / avg_base
    gamma_routed = avg_routed / avg_base
    gamma_oracle = avg_oracle / avg_base

    log(f"\n  Average PPL: base={avg_base:.2f} oracle={avg_oracle:.2f} "
        f"uniform={avg_uniform:.2f} routed={avg_routed:.2f}")
    log(f"  Gamma (PPL/base): oracle={gamma_oracle:.4f} uniform={gamma_uniform:.4f} "
        f"routed={gamma_routed:.4f}")
    log(f"  Routed improvement over uniform: {(1 - avg_routed/avg_uniform)*100:.1f}%")

    cleanup(model, tokenizer)
    del adapter_params
    gc.collect()
    mx.clear_cache()

    eval_results = {
        "hierarchical_accuracy": round(hierarchical_acc, 4),
        "cluster_accuracy": round(cluster_acc, 4),
        "within_cluster_accuracy": round(within_acc, 4),
        "base_forward_ms": round(base_ms, 2),
        "routing_ms": round(route_ms, 2),
        "overhead_pct": round(overhead_pct, 2),
        "avg_base_ppl": round(float(avg_base), 4),
        "avg_oracle_ppl": round(float(avg_oracle), 4),
        "avg_uniform_ppl": round(float(avg_uniform), 4),
        "avg_routed_ppl": round(float(avg_routed), 4),
        "gamma_uniform": round(float(gamma_uniform), 4),
        "gamma_routed": round(float(gamma_routed), 4),
        "gamma_oracle": round(float(gamma_oracle), 4),
        "routed_improvement_over_uniform_pct": round(float((1 - avg_routed/avg_uniform)*100), 2),
        "per_domain_accuracy": {d: round(per_domain_correct[d] / max(per_domain_total[d], 1), 4)
                                for d in DOMAINS},
        "per_domain_ppls": {
            "base": {d: round(base_ppls.get(d, 0), 4) for d in DOMAINS},
            "oracle": {d: round(oracle_ppls.get(d, 0), 4) for d in DOMAINS},
            "uniform": {d: round(uniform_ppls.get(d, 0), 4) for d in DOMAINS},
            "routed": {d: round(routed_ppls.get(d, 0), 4) for d in DOMAINS},
        },
    }

    return eval_results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log_memory("start")

    # Phase 1: Cache hidden states
    train_data, val_data, d_model = phase_cache_hidden_states(MODEL_ID)
    log_memory("after-phase1")

    # Phase 2: Build clusters
    clusters, domain_to_cluster, cluster_info = phase_build_clusters(
        train_data, val_data, d_model
    )
    log_memory("after-phase2")

    # Phase 3: Train hierarchical routers
    stage1_router, stage2_routers, router_info = phase_train_hierarchical_routers(
        train_data, val_data, d_model, clusters, domain_to_cluster
    )
    log_memory("after-phase3")

    # Phase 4: Evaluate
    eval_results = phase_evaluate(
        MODEL_ID, stage1_router, stage2_routers, clusters,
        domain_to_cluster, val_data, d_model
    )
    log_memory("after-phase4")

    # -----------------------------------------------------------------------
    # Assemble results and kill criteria
    # -----------------------------------------------------------------------
    total_time = time.time() - t_start

    # Kill criteria assessment
    K593_pass = eval_results["hierarchical_accuracy"] >= 0.60
    K594_pass = eval_results["avg_routed_ppl"] < eval_results["avg_uniform_ppl"]
    K595_pass = eval_results["overhead_pct"] < 15.0

    results = {
        "experiment": "hierarchical_routing_n24",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cluster_info": cluster_info,
        "router_info": router_info,
        "eval_results": eval_results,
        "K593_hierarchical_accuracy": eval_results["hierarchical_accuracy"],
        "K593_threshold": 0.60,
        "K593_pass": bool(K593_pass),
        "K594_routed_ppl": eval_results["avg_routed_ppl"],
        "K594_uniform_ppl": eval_results["avg_uniform_ppl"],
        "K594_pass": bool(K594_pass),
        "K595_overhead_pct": eval_results["overhead_pct"],
        "K595_threshold": 15.0,
        "K595_pass": bool(K595_pass),
        "verdict": "SUPPORTED" if (K593_pass and K594_pass and K595_pass) else "KILLED",
        "kill_reasons": [],
        "total_time_s": round(total_time, 1),
    }

    if not K593_pass:
        results["kill_reasons"].append(
            f"K593 (accuracy {eval_results['hierarchical_accuracy']:.1%} < 60%)")
    if not K594_pass:
        results["kill_reasons"].append(
            f"K594 (routed PPL {eval_results['avg_routed_ppl']:.2f} >= uniform {eval_results['avg_uniform_ppl']:.2f})")
    if not K595_pass:
        results["kill_reasons"].append(
            f"K595 (overhead {eval_results['overhead_pct']:.1f}% >= 15%)")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n{'=' * 70}")
    log(f"VERDICT: {results['verdict']}")
    for reason in results["kill_reasons"]:
        log(f"  {reason}")
    log(f"Total time: {total_time:.0f}s")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
