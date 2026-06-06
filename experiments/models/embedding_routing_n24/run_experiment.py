#!/usr/bin/env python3
"""
Embedding-Based Routing at N=24.

Tests whether the base model's embedding layer (BEFORE transformer blocks)
preserves enough domain signal for routing, bypassing the mean-pooled hidden
state bottleneck that killed 5 routing architectures at ~40% accuracy.

Three routing methods compared:
  1. Embedding centroid routing (argmax cosine similarity, zero parameters)
  2. TF-IDF baseline (bag-of-words, no neural network)
  3. Hidden-state centroid routing (same mechanism but on post-transformer features)
  4. Trained softmax router baseline (39.4% from Finding #192)

Kill criteria:
  K590: Top-1 routing accuracy <60% at N=24 -> KILL
  K591: Embedding routing not significantly better than 39.4% baseline -> KILL
  K592: Embedding computation overhead >50ms per query -> KILL

Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
MAX_SEQ_LENGTH = 256
SEED = 42

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

# Samples per domain
TRAIN_PER_DOMAIN = 40
VAL_PER_DOMAIN = 20  # remaining 10 held out within valid split


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
# Model utilities (reused from prior experiment)
# ===========================================================================
from mlx_lm import load
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
# Phase 1: Compute embedding centroids and hidden-state centroids
# ===========================================================================
def phase_compute_representations(model_id):
    """Load model, compute both embedding-layer and hidden-state centroids."""
    log("\n" + "=" * 70)
    log("[Phase 1] Computing embedding and hidden-state representations")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    d_model = model.model.embed_tokens.weight.shape[1]
    log(f"  d_model = {d_model}")
    log(f"  N_domains = {N_DOMAINS}: {', '.join(DOMAINS)}")

    rng = random.Random(SEED)

    # Storage: per-domain lists of mean embeddings
    train_embs = defaultdict(list)   # domain -> list of (d,) arrays (embedding layer)
    val_embs = defaultdict(list)
    train_hidden = defaultdict(list)  # domain -> list of (d,) arrays (hidden states)
    val_hidden = defaultdict(list)

    # Also collect token IDs for TF-IDF
    train_token_ids = defaultdict(list)  # domain -> list of lists of token_ids
    val_token_ids = defaultdict(list)

    # Timing for overhead measurement
    embed_times = []
    hidden_times = []

    for i, domain in enumerate(DOMAINS):
        # Load texts for train and val splits
        train_texts = load_domain_texts(domain, split="train")
        val_texts = load_domain_texts(domain, split="valid")

        # Shuffle and split val: first TRAIN_PER_DOMAIN for centroid, rest for eval
        rng.shuffle(train_texts)
        rng.shuffle(val_texts)

        for split_name, texts, max_n, emb_store, hid_store, tok_store in [
            ("train", train_texts, TRAIN_PER_DOMAIN, train_embs, train_hidden, train_token_ids),
            ("valid", val_texts, VAL_PER_DOMAIN, val_embs, val_hidden, val_token_ids),
        ]:
            for text in texts[:max_n]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]

                # === Embedding layer only ===
                t_emb = time.time()
                emb = model.model.embed_tokens(x)  # (1, T, d)
                emb_mean = mx.mean(emb, axis=1).squeeze(0)  # (d,)
                mx.eval(emb_mean)
                embed_times.append(time.time() - t_emb)
                emb_store[domain].append(emb_mean)

                # === Full hidden states (for comparison baseline) ===
                t_hid = time.time()
                h = emb
                for layer in model.model.layers:
                    h = layer(h)
                h = model.model.norm(h)
                h_mean = mx.mean(h, axis=1).squeeze(0)  # (d,)
                mx.eval(h_mean)
                hidden_times.append(time.time() - t_hid)
                hid_store[domain].append(h_mean)

                # === Token IDs for TF-IDF ===
                tok_store[domain].append(tokens)

                del emb, h, x, emb_mean, h_mean

        if (i + 1) % 8 == 0:
            log(f"  Processed {i+1}/{N_DOMAINS} domains...")
            log_memory(f"after-{i+1}-domains")

    elapsed = time.time() - t0
    log(f"  Representation computation done in {elapsed:.1f}s")
    log(f"  Avg embed time: {sum(embed_times)/len(embed_times)*1000:.2f}ms")
    log(f"  Avg hidden time: {sum(hidden_times)/len(hidden_times)*1000:.2f}ms")
    log_memory("after-phase1")

    cleanup(model, tokenizer)

    return {
        "train_embs": dict(train_embs),
        "val_embs": dict(val_embs),
        "train_hidden": dict(train_hidden),
        "val_hidden": dict(val_hidden),
        "train_token_ids": dict(train_token_ids),
        "val_token_ids": dict(val_token_ids),
        "d_model": d_model,
        "embed_times": embed_times,
        "hidden_times": hidden_times,
    }


# ===========================================================================
# Phase 2: Centroid-based routing (embedding and hidden state)
# ===========================================================================
def compute_centroids(domain_embs):
    """Compute mean centroid for each domain from a dict of domain->list of arrays."""
    centroids = {}
    for domain, embs in domain_embs.items():
        stacked = mx.stack(embs, axis=0)  # (N, d)
        centroid = mx.mean(stacked, axis=0)  # (d,)
        # L2 normalize for cosine similarity
        norm = mx.linalg.norm(centroid)
        centroid = centroid / (norm + 1e-8)
        mx.eval(centroid)
        centroids[domain] = centroid
        del stacked
    return centroids


def route_by_cosine(query_emb, centroids, domains):
    """Route a single query by argmax cosine similarity to centroids.

    Args:
        query_emb: (d,) array, NOT normalized
        centroids: dict domain -> (d,) normalized arrays
        domains: list of domain names
    Returns:
        predicted domain name, similarity scores dict
    """
    # Normalize query
    q_norm = query_emb / (mx.linalg.norm(query_emb) + 1e-8)

    # Stack centroids in domain order
    centroid_matrix = mx.stack([centroids[d] for d in domains], axis=0)  # (K, d)

    # Cosine similarities
    sims = centroid_matrix @ q_norm  # (K,)
    mx.eval(sims)

    best_idx = mx.argmax(sims).item()
    sim_dict = {d: sims[j].item() for j, d in enumerate(domains)}
    return domains[best_idx], sim_dict


def evaluate_centroid_routing(centroids, val_embs, domains):
    """Evaluate centroid routing accuracy across all validation samples."""
    correct = 0
    total = 0
    per_domain_correct = defaultdict(int)
    per_domain_total = defaultdict(int)
    confusion = defaultdict(int)

    for true_domain in domains:
        for emb in val_embs.get(true_domain, []):
            pred_domain, _ = route_by_cosine(emb, centroids, domains)
            if pred_domain == true_domain:
                correct += 1
                per_domain_correct[true_domain] += 1
            else:
                confusion[f"{true_domain} -> {pred_domain}"] += 1
            total += 1
            per_domain_total[true_domain] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_domain_acc = {d: per_domain_correct[d] / per_domain_total[d]
                      if per_domain_total[d] > 0 else 0.0
                      for d in domains}

    # Top confusions
    top_confusion = dict(sorted(confusion.items(), key=lambda x: -x[1])[:15])

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_domain": per_domain_acc,
        "confusion_top15": top_confusion,
    }


def phase_centroid_routing(data):
    """Evaluate centroid-based routing for both embedding and hidden-state features."""
    log("\n" + "=" * 70)
    log("[Phase 2] Centroid-based routing evaluation")
    log("=" * 70)

    results = {}

    # --- Embedding centroids ---
    log("\n  [2a] Embedding-layer centroid routing")
    emb_centroids = compute_centroids(data["train_embs"])
    emb_results = evaluate_centroid_routing(emb_centroids, data["val_embs"], DOMAINS)
    results["embedding"] = emb_results
    log(f"  Embedding routing accuracy: {emb_results['accuracy']:.4f} ({emb_results['correct']}/{emb_results['total']})")

    # --- Hidden-state centroids ---
    log("\n  [2b] Hidden-state centroid routing (post-transformer)")
    hid_centroids = compute_centroids(data["train_hidden"])
    hid_results = evaluate_centroid_routing(hid_centroids, data["val_hidden"], DOMAINS)
    results["hidden_state"] = hid_results
    log(f"  Hidden-state routing accuracy: {hid_results['accuracy']:.4f} ({hid_results['correct']}/{hid_results['total']})")

    # --- Inter-centroid similarity analysis ---
    log("\n  [2c] Centroid separation analysis")
    emb_centroid_mat = mx.stack([emb_centroids[d] for d in DOMAINS], axis=0)  # (K, d)
    hid_centroid_mat = mx.stack([hid_centroids[d] for d in DOMAINS], axis=0)  # (K, d)

    emb_sim = emb_centroid_mat @ emb_centroid_mat.T  # (K, K) cosine sim
    hid_sim = hid_centroid_mat @ hid_centroid_mat.T
    mx.eval(emb_sim, hid_sim)

    # Off-diagonal statistics
    K = len(DOMAINS)
    emb_off_diag = []
    hid_off_diag = []
    for a in range(K):
        for b in range(a + 1, K):
            emb_off_diag.append(emb_sim[a, b].item())
            hid_off_diag.append(hid_sim[a, b].item())

    results["emb_centroid_mean_cos"] = sum(emb_off_diag) / len(emb_off_diag)
    results["emb_centroid_max_cos"] = max(emb_off_diag)
    results["emb_centroid_min_cos"] = min(emb_off_diag)
    results["hid_centroid_mean_cos"] = sum(hid_off_diag) / len(hid_off_diag)
    results["hid_centroid_max_cos"] = max(hid_off_diag)
    results["hid_centroid_min_cos"] = min(hid_off_diag)

    log(f"  Embedding centroid cos: mean={results['emb_centroid_mean_cos']:.4f}, "
        f"max={results['emb_centroid_max_cos']:.4f}, min={results['emb_centroid_min_cos']:.4f}")
    log(f"  Hidden centroid cos:    mean={results['hid_centroid_mean_cos']:.4f}, "
        f"max={results['hid_centroid_max_cos']:.4f}, min={results['hid_centroid_min_cos']:.4f}")

    del emb_centroids, hid_centroids, emb_centroid_mat, hid_centroid_mat, emb_sim, hid_sim
    return results


# ===========================================================================
# Phase 3: TF-IDF baseline
# ===========================================================================
def phase_tfidf_routing(data):
    """TF-IDF bag-of-words baseline routing."""
    log("\n" + "=" * 70)
    log("[Phase 3] TF-IDF baseline routing")
    log("=" * 70)

    import numpy as np

    # Build vocabulary and document frequency from training data
    doc_freq = Counter()  # token_id -> number of domains containing it
    domain_tf = {}  # domain -> Counter of token_id frequencies

    for domain in DOMAINS:
        all_tokens = []
        for token_list in data["train_token_ids"].get(domain, []):
            all_tokens.extend(token_list)
        domain_tf[domain] = Counter(all_tokens)
        unique_tokens = set(all_tokens)
        for t in unique_tokens:
            doc_freq[t] += 1

    # Build TF-IDF centroid for each domain
    vocab = sorted(doc_freq.keys())
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    log(f"  Vocabulary size: {V}")

    # IDF
    idf = np.zeros(V)
    for t, idx in vocab_idx.items():
        idf[idx] = math.log(N_DOMAINS / (1 + doc_freq[t]))

    # Domain TF-IDF vectors
    domain_vecs = {}
    for domain in DOMAINS:
        vec = np.zeros(V)
        tf = domain_tf[domain]
        total = sum(tf.values())
        if total == 0:
            domain_vecs[domain] = vec
            continue
        for t, count in tf.items():
            if t in vocab_idx:
                vec[vocab_idx[t]] = (count / total) * idf[vocab_idx[t]]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        domain_vecs[domain] = vec

    # Evaluate on validation data
    correct = 0
    total = 0
    per_domain_correct = defaultdict(int)
    per_domain_total = defaultdict(int)
    confusion = defaultdict(int)

    for true_domain in DOMAINS:
        for token_list in data["val_token_ids"].get(true_domain, []):
            # Build query TF-IDF vector
            q_tf = Counter(token_list)
            q_total = sum(q_tf.values())
            q_vec = np.zeros(V)
            for t, count in q_tf.items():
                if t in vocab_idx:
                    q_vec[vocab_idx[t]] = (count / q_total) * idf[vocab_idx[t]]
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec /= q_norm

            # Cosine similarity with all domain centroids
            best_domain = None
            best_sim = -1.0
            for d in DOMAINS:
                sim = float(np.dot(q_vec, domain_vecs[d]))
                if sim > best_sim:
                    best_sim = sim
                    best_domain = d

            if best_domain == true_domain:
                correct += 1
                per_domain_correct[true_domain] += 1
            else:
                confusion[f"{true_domain} -> {best_domain}"] += 1
            total += 1
            per_domain_total[true_domain] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_domain_acc = {d: per_domain_correct[d] / per_domain_total[d]
                      if per_domain_total[d] > 0 else 0.0
                      for d in DOMAINS}

    top_confusion = dict(sorted(confusion.items(), key=lambda x: -x[1])[:15])

    log(f"  TF-IDF routing accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_domain": per_domain_acc,
        "confusion_top15": top_confusion,
        "vocab_size": V,
    }


# ===========================================================================
# Phase 4: Timing overhead measurement
# ===========================================================================
def phase_overhead_measurement(data):
    """Measure embedding computation overhead."""
    log("\n" + "=" * 70)
    log("[Phase 4] Overhead measurement")
    log("=" * 70)

    embed_times_ms = [t * 1000 for t in data["embed_times"]]
    hidden_times_ms = [t * 1000 for t in data["hidden_times"]]

    avg_embed = sum(embed_times_ms) / len(embed_times_ms)
    avg_hidden = sum(hidden_times_ms) / len(hidden_times_ms)
    max_embed = max(embed_times_ms)
    p95_embed = sorted(embed_times_ms)[int(0.95 * len(embed_times_ms))]

    log(f"  Embedding lookup: avg={avg_embed:.2f}ms, max={max_embed:.2f}ms, p95={p95_embed:.2f}ms")
    log(f"  Hidden state:     avg={avg_hidden:.2f}ms")
    log(f"  Speedup:          {avg_hidden/avg_embed:.1f}x")

    return {
        "avg_embed_ms": round(avg_embed, 3),
        "max_embed_ms": round(max_embed, 3),
        "p95_embed_ms": round(p95_embed, 3),
        "avg_hidden_ms": round(avg_hidden, 3),
        "speedup_factor": round(avg_hidden / avg_embed, 1) if avg_embed > 0 else 0,
    }


# ===========================================================================
# Phase 5: Instruction-only embedding routing
# ===========================================================================
def phase_instruction_only_routing(data):
    """Test routing using only instruction tokens (before ### Response)."""
    log("\n" + "=" * 70)
    log("[Phase 5] Instruction-only embedding routing")
    log("=" * 70)

    # Re-load model just for embedding layer
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")

    # We only need the embedding layer - no need to unpack BitLinear
    embed_layer = model.model.embed_tokens

    # Extract instruction prefix from texts (before "### Response:")
    def get_instruction_tokens(text, tokenizer):
        """Extract tokens for instruction portion only."""
        resp_marker = "### Response:"
        idx = text.find(resp_marker)
        if idx > 0:
            instruction = text[:idx]
        else:
            instruction = text[:200]  # fallback: first 200 chars
        tokens = tokenizer.encode(instruction)
        return tokens[:MAX_SEQ_LENGTH]

    # Compute instruction-only centroids from training data
    train_centroids = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="train")
        embs = []
        for text in texts[:TRAIN_PER_DOMAIN]:
            tokens = get_instruction_tokens(text, tokenizer)
            if len(tokens) < 2:
                continue
            x = mx.array(tokens)[None, :]
            emb = embed_layer(x)
            emb_mean = mx.mean(emb, axis=1).squeeze(0)
            mx.eval(emb_mean)
            embs.append(emb_mean)
            del emb, x

        if embs:
            stacked = mx.stack(embs, axis=0)
            centroid = mx.mean(stacked, axis=0)
            norm = mx.linalg.norm(centroid)
            centroid = centroid / (norm + 1e-8)
            mx.eval(centroid)
            train_centroids[domain] = centroid
            del stacked

    # Evaluate on validation data (instruction-only)
    correct = 0
    total = 0
    per_domain_correct = defaultdict(int)
    per_domain_total = defaultdict(int)
    confusion = defaultdict(int)

    for true_domain in DOMAINS:
        val_texts = load_domain_texts(true_domain, split="valid")
        for text in val_texts[:VAL_PER_DOMAIN]:
            tokens = get_instruction_tokens(text, tokenizer)
            if len(tokens) < 2:
                continue
            x = mx.array(tokens)[None, :]
            emb = embed_layer(x)
            emb_mean = mx.mean(emb, axis=1).squeeze(0)
            mx.eval(emb_mean)

            pred_domain, _ = route_by_cosine(emb_mean, train_centroids, DOMAINS)
            if pred_domain == true_domain:
                correct += 1
                per_domain_correct[true_domain] += 1
            else:
                confusion[f"{true_domain} -> {pred_domain}"] += 1
            total += 1
            per_domain_total[true_domain] += 1

            del emb, x, emb_mean

    accuracy = correct / total if total > 0 else 0.0
    per_domain_acc = {d: per_domain_correct[d] / per_domain_total[d]
                      if per_domain_total[d] > 0 else 0.0
                      for d in DOMAINS}
    top_confusion = dict(sorted(confusion.items(), key=lambda x: -x[1])[:15])

    log(f"  Instruction-only embedding routing accuracy: {accuracy:.4f} ({correct}/{total})")

    cleanup(model, tokenizer)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_domain": per_domain_acc,
        "confusion_top15": top_confusion,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log_memory("start")

    log(f"Experiment: Embedding-Based Routing at N=24")
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {N_DOMAINS}")
    log(f"Train/Val per domain: {TRAIN_PER_DOMAIN}/{VAL_PER_DOMAIN}")

    # Phase 1: Compute representations
    data = phase_compute_representations(MODEL_ID)
    log_memory("after-phase1-cleanup")

    # Phase 2: Centroid routing (embedding vs hidden state)
    centroid_results = phase_centroid_routing(data)
    log_memory("after-phase2")

    # Phase 3: TF-IDF baseline
    tfidf_results = phase_tfidf_routing(data)
    log_memory("after-phase3")

    # Phase 4: Overhead measurement
    overhead = phase_overhead_measurement(data)

    # Free representation data before Phase 5
    cleanup(data)

    # Phase 5: Instruction-only embedding routing
    instruction_results = phase_instruction_only_routing({})
    log_memory("after-phase5")

    # ===========================================================================
    # Kill criteria assessment
    # ===========================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # Best embedding accuracy (full text vs instruction-only)
    emb_acc = centroid_results["embedding"]["accuracy"]
    instr_acc = instruction_results["accuracy"]
    best_emb_acc = max(emb_acc, instr_acc)
    best_emb_method = "instruction-only" if instr_acc > emb_acc else "full-text"

    baseline_acc = 0.394  # Finding #192 softmax baseline

    # K590: Top-1 accuracy >= 60%
    K590_pass = best_emb_acc >= 0.60
    log(f"\n  K590: Top-1 accuracy >= 60%")
    log(f"    Best embedding accuracy: {best_emb_acc:.4f} ({best_emb_method})")
    log(f"    Result: {'PASS' if K590_pass else 'FAIL'}")

    # K591: Significantly better than 39.4% baseline
    improvement = best_emb_acc - baseline_acc
    K591_pass = improvement > 0.05  # >5 percentage points improvement
    log(f"\n  K591: Significantly better than 39.4% baseline")
    log(f"    Improvement: {improvement*100:.1f} percentage points")
    log(f"    Result: {'PASS' if K591_pass else 'FAIL'}")

    # K592: Embedding overhead < 50ms
    K592_pass = overhead["p95_embed_ms"] < 50.0
    log(f"\n  K592: Embedding overhead < 50ms")
    log(f"    P95 embed time: {overhead['p95_embed_ms']:.2f}ms")
    log(f"    Result: {'PASS' if K592_pass else 'FAIL'}")

    all_pass = K590_pass and K591_pass and K592_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    kill_reasons = []
    if not K590_pass:
        kill_reasons.append(f"K590 (best accuracy {best_emb_acc:.4f} < 0.60)")
    if not K591_pass:
        kill_reasons.append(f"K591 (improvement {improvement*100:.1f}pp not significant)")
    if not K592_pass:
        kill_reasons.append(f"K592 (P95 overhead {overhead['p95_embed_ms']:.2f}ms >= 50ms)")

    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        log(f"  Kill reasons: {'; '.join(kill_reasons)}")

    # ===========================================================================
    # Summary comparison
    # ===========================================================================
    log("\n" + "=" * 70)
    log("ROUTING ACCURACY COMPARISON")
    log("=" * 70)

    hid_centroid_acc = centroid_results["hidden_state"]["accuracy"]

    log(f"  TF-IDF bag-of-words:          {tfidf_results['accuracy']:.4f}")
    log(f"  Embedding centroid (full):     {emb_acc:.4f}")
    log(f"  Embedding centroid (instr):    {instr_acc:.4f}")
    log(f"  Hidden-state centroid:         {hid_centroid_acc:.4f}")
    log(f"  Trained softmax (baseline):    {baseline_acc:.4f}")

    # Per-domain comparison table
    log(f"\n  Per-domain accuracy breakdown:")
    log(f"  {'Domain':<20} {'TF-IDF':>8} {'Emb-Full':>10} {'Emb-Instr':>10} {'Hidden':>8} {'Softmax':>8}")
    log(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    # Load softmax per-domain from prior results
    softmax_per_domain = {}
    prior_results_path = EXPERIMENT_DIR.parent / "centralized_multiclass_routing_n24" / "results.json"
    if prior_results_path.exists():
        with open(prior_results_path) as f:
            prior = json.load(f)
            softmax_per_domain = prior.get("per_domain_top1", {})

    for domain in DOMAINS:
        tf = tfidf_results["per_domain"].get(domain, 0.0)
        ef = centroid_results["embedding"]["per_domain"].get(domain, 0.0)
        ei = instruction_results["per_domain"].get(domain, 0.0)
        hd = centroid_results["hidden_state"]["per_domain"].get(domain, 0.0)
        sm = softmax_per_domain.get(domain, 0.0)
        log(f"  {domain:<20} {tf:>8.2f} {ef:>10.2f} {ei:>10.2f} {hd:>8.2f} {sm:>8.2f}")

    # ===========================================================================
    # Save results
    # ===========================================================================
    results = {
        "experiment": "embedding_routing_n24",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "seed": SEED,
        "d_model": data["d_model"] if isinstance(data, dict) else centroid_results.get("d_model", 2560),
        "train_per_domain": TRAIN_PER_DOMAIN,
        "val_per_domain": VAL_PER_DOMAIN,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

        # Embedding centroid routing (full text)
        "embedding_full_accuracy": emb_acc,
        "embedding_full_per_domain": centroid_results["embedding"]["per_domain"],
        "embedding_full_confusion": centroid_results["embedding"]["confusion_top15"],

        # Embedding centroid routing (instruction only)
        "embedding_instr_accuracy": instr_acc,
        "embedding_instr_per_domain": instruction_results["per_domain"],
        "embedding_instr_confusion": instruction_results["confusion_top15"],

        # Hidden-state centroid routing
        "hidden_state_accuracy": hid_centroid_acc,
        "hidden_state_per_domain": centroid_results["hidden_state"]["per_domain"],
        "hidden_state_confusion": centroid_results["hidden_state"]["confusion_top15"],

        # TF-IDF baseline
        "tfidf_accuracy": tfidf_results["accuracy"],
        "tfidf_per_domain": tfidf_results["per_domain"],
        "tfidf_confusion": tfidf_results["confusion_top15"],
        "tfidf_vocab_size": tfidf_results["vocab_size"],

        # Centroid separation analysis
        "emb_centroid_mean_cos": centroid_results["emb_centroid_mean_cos"],
        "emb_centroid_max_cos": centroid_results["emb_centroid_max_cos"],
        "emb_centroid_min_cos": centroid_results["emb_centroid_min_cos"],
        "hid_centroid_mean_cos": centroid_results["hid_centroid_mean_cos"],
        "hid_centroid_max_cos": centroid_results["hid_centroid_max_cos"],
        "hid_centroid_min_cos": centroid_results["hid_centroid_min_cos"],

        # Overhead
        "overhead": overhead,
        "baseline_softmax_accuracy": baseline_acc,

        # Kill criteria
        "K590_best_accuracy": best_emb_acc,
        "K590_best_method": best_emb_method,
        "K590_threshold": 0.60,
        "K590_pass": K590_pass,

        "K591_improvement_pp": round(improvement * 100, 1),
        "K591_threshold_pp": 5.0,
        "K591_pass": K591_pass,

        "K592_p95_embed_ms": overhead["p95_embed_ms"],
        "K592_threshold_ms": 50.0,
        "K592_pass": K592_pass,

        "verdict": verdict,
        "kill_reasons": kill_reasons,
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
