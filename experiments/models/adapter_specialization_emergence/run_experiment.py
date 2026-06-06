#!/usr/bin/env python3
"""Experiment: adapter_specialization_emergence

Do adapters self-specialize? Train N=10 adapters on mixed-domain data with
different Grassmannian A-matrices, measure emergent domain specialization.

Hypothesis: Orthogonal A matrices force each adapter to capture different
directions in weight space, leading to emergent specialization by domain.

Kill criteria:
  K1: No specialization emerges (silhouette < 0.2)
  K2 (implicit): Mixed-trained adapters worse than domain-trained on best domain

Grounding:
  FlyLoRA (arxiv 2510.08396) — frozen random A as implicit feature selector
  MoE self-specialization (Shazeer 2017) — experts specialize on mixed data
  exp_softmax_router_scaling LEARNINGS — semantic clustering of domains

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source experiment with domain-trained adapters and data
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
SOURCE_ADAPTERS_DIR = SOURCE_DIR / "adapters"
SOURCE_DATA_DIR = SOURCE_DIR / "data"

# Local adapter storage for mixed-trained adapters
MIXED_ADAPTERS_DIR = EXPERIMENT_DIR / "mixed_adapters"

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42
N_ADAPTERS = 10  # How many adapters to train on mixed data

# Select 10 diverse domains (avoid the cluster of 8 similar ones from softmax_router)
# Pick from different semantic clusters identified in softmax_router_scaling:
SELECTED_DOMAINS = [
    "code",           # singleton
    "math",           # singleton
    "medical",        # cluster: medical/health_fitness
    "legal",          # cluster: legal/finance
    "cooking",        # singleton
    "psychology",     # singleton
    "cybersecurity",  # singleton
    "philosophy",     # from large cluster
    "economics",      # from large cluster
    "music",          # from education/engineering/sports adjacent
]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


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
# Model utilities (reused from real_data_25_domain_adapters)
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
    """Replace BitLinear with nn.Linear for differentiable training."""
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
    return model


class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank, scale, a_matrices_per_layer):
    """Apply TernaryLoRALinear to all target projection layers."""
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            a_key = (li, key)
            if a_key in a_matrices_per_layer:
                a_np = a_matrices_per_layer[a_key]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
            else:
                a_mx = None
            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    return model


def get_trainable_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_trainable_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_b_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def _set_lora_a(model, skeleton, domain_idx, n_layers):
    """Set A matrices in TernaryLoRALinear modules from skeleton for a given domain."""
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_np = skeleton[skey]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, TernaryLoRALinear):
                    module.lora_a = a_mx


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")
    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
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
        loss_val = loss.item()
        n_tok = y.size
        total_loss += loss_val
        total_tokens += n_tok
        del logits, loss, x, y
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))
    return ppl


# ============================================================================
# Phase 1: Prepare mixed training data
# ============================================================================

def phase_prepare_mixed_data():
    """Create mixed training data by combining samples from all selected domains."""
    log("\n[Phase 1] Preparing mixed training data...")

    mixed_train_tokens_per_domain = {}  # domain -> list of token sequences
    domain_val_paths = {}  # domain -> Path to val data

    for domain in SELECTED_DOMAINS:
        data_dir = SOURCE_DATA_DIR / domain
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "valid.jsonl"

        if not train_path.exists():
            log(f"  WARNING: {domain} train data not found at {train_path}")
            continue

        # Load train texts
        texts = []
        with open(train_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        # Take 50 samples for mixed training (total: 50*10 = 500)
        rng = random.Random(SEED + hash(domain) % 10000)
        rng.shuffle(texts)
        selected = texts[:50]

        mixed_train_tokens_per_domain[domain] = selected
        domain_val_paths[domain] = data_dir
        log(f"  {domain}: {len(selected)} train samples, val at {data_dir}")

    # Combine all train samples into one mixed pool
    all_train_texts = []
    for domain in SELECTED_DOMAINS:
        if domain in mixed_train_tokens_per_domain:
            for text in mixed_train_tokens_per_domain[domain]:
                all_train_texts.append(text)

    # Shuffle the mixed pool
    rng = random.Random(SEED)
    rng.shuffle(all_train_texts)
    log(f"  Total mixed training samples: {len(all_train_texts)}")

    return all_train_texts, domain_val_paths


# ============================================================================
# Phase 2: Compute base PPL on selected domains
# ============================================================================

def phase_base_ppl(domain_val_paths):
    """Compute base model PPL on all selected domains."""
    log(f"\n[Phase 2] Computing base PPL on {len(domain_val_paths)} domains...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("post-unpack")

    base_ppls = {}
    for domain, data_dir in domain_val_paths.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain] = ppl
        log(f"  {domain}: base PPL = {ppl:.2f}")

    cleanup(model, tokenizer)
    log(f"  Base PPL done in {time.time()-t0:.1f}s")
    return base_ppls


# ============================================================================
# Phase 3: Train N adapters on mixed data
# ============================================================================

def phase_train_mixed_adapter(adapter_idx, all_train_texts, skeleton):
    """Train a single adapter on mixed data with Grassmannian A matrix for adapter_idx."""
    log(f"\n[Phase 3.{adapter_idx}] Training mixed adapter {adapter_idx}...")
    t0 = time.time()

    adapter_path = MIXED_ADAPTERS_DIR / f"adapter_{adapter_idx}" / "adapter.npz"
    if adapter_path.exists():
        log(f"  Already exists, skipping")
        return {"train_time_s": 0, "skipped": True}

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Build A matrix mapping for this adapter index
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{adapter_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)

    # Freeze everything except lora_b
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")

    # Tokenize training data
    train_tokens = []
    for text in all_train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    log(f"  {len(train_tokens)} training sequences (mixed from {len(SELECTED_DOMAINS)} domains)")

    # Training loop
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50

    log(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f}")

    save_adapter(model, MIXED_ADAPTERS_DIR / f"adapter_{adapter_idx}")

    result = {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "trainable_params": trainable,
        "skipped": False,
    }

    log_memory(f"post-train-mixed-{adapter_idx}")
    cleanup(model, tokenizer, optimizer)
    return result


# ============================================================================
# Phase 4: Evaluate mixed adapters on all domains (build N x K PPL matrix)
# ============================================================================

def phase_evaluate_mixed(domain_val_paths, skeleton):
    """Evaluate each mixed adapter on each domain. Returns N x K PPL matrix."""
    log(f"\n[Phase 4] Evaluating {N_ADAPTERS} mixed adapters on {len(domain_val_paths)} domains...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA structure with adapter_0's A matrices initially
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    domains_list = list(domain_val_paths.keys())
    K = len(domains_list)

    # ppl_matrix[i][k] = PPL of adapter i on domain k
    ppl_matrix = np.zeros((N_ADAPTERS, K))

    for ai in range(N_ADAPTERS):
        adapter_path = MIXED_ADAPTERS_DIR / f"adapter_{ai}"
        if not (adapter_path / "adapter.npz").exists():
            log(f"  Adapter {ai}: not found, filling with inf")
            ppl_matrix[ai, :] = float("inf")
            continue

        # Set correct A matrices for this adapter
        _set_lora_a(model, skeleton, ai, n_layers)

        # Load B weights
        params = load_adapter(adapter_path)
        zero_b_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        for ki, domain in enumerate(domains_list):
            data_dir = domain_val_paths[domain]
            ppl = compute_ppl(model, tokenizer, data_dir)
            ppl_matrix[ai, ki] = ppl

        best_domain_idx = np.argmin(ppl_matrix[ai])
        log(f"  Adapter {ai}: best={domains_list[best_domain_idx]} "
            f"(PPL={ppl_matrix[ai, best_domain_idx]:.2f}), "
            f"worst PPL={np.max(ppl_matrix[ai]):.2f}")

    eval_time = time.time() - t0
    log(f"  Mixed evaluation done in {eval_time:.1f}s")

    cleanup(model, tokenizer)
    del skeleton
    return ppl_matrix, domains_list


# ============================================================================
# Phase 5: Evaluate domain-trained adapters for comparison
# ============================================================================

def phase_evaluate_domain_trained(domain_val_paths):
    """Evaluate domain-specific adapters on their own and cross domains."""
    log(f"\n[Phase 5] Evaluating domain-trained adapters for comparison...")
    t0 = time.time()

    # Load the N=24 skeleton from original experiment
    skeleton_path = SOURCE_ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"
    skeleton = dict(np.load(str(skeleton_path)))

    # Map domain names to their indices in the original 24-domain experiment
    all_24_domains = [
        "medical", "code", "math", "legal", "finance",
        "science", "history", "philosophy", "creative_writing", "cooking",
        "health_fitness", "psychology", "education", "engineering", "agriculture",
        "environmental", "politics", "economics", "sociology", "linguistics",
        "cybersecurity", "marketing", "sports", "music",
    ]
    domain_to_idx = {d: i for i, d in enumerate(all_24_domains)}

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA structure
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    domains_list = list(domain_val_paths.keys())
    K = len(domains_list)

    # domain_trained_ppls[domain_name] = dict of {eval_domain: ppl}
    domain_trained_ppls = {}

    for domain in domains_list:
        adapter_path = SOURCE_ADAPTERS_DIR / domain
        if not (adapter_path / "adapter.npz").exists():
            log(f"  {domain}: domain-trained adapter not found")
            continue

        di = domain_to_idx.get(domain)
        if di is None:
            log(f"  {domain}: not in original 24 domains")
            continue

        _set_lora_a(model, skeleton, di, n_layers)
        params = load_adapter(adapter_path)
        zero_b_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        ppls = {}
        for eval_domain in domains_list:
            ppl = compute_ppl(model, tokenizer, domain_val_paths[eval_domain])
            ppls[eval_domain] = ppl

        domain_trained_ppls[domain] = ppls
        own_ppl = ppls[domain]
        log(f"  {domain}: own-domain PPL = {own_ppl:.2f}")

    eval_time = time.time() - t0
    log(f"  Domain-trained evaluation done in {eval_time:.1f}s")

    cleanup(model, tokenizer)
    return domain_trained_ppls


# ============================================================================
# Phase 6: Analyze specialization
# ============================================================================

def phase_analyze(ppl_matrix, domains_list, base_ppls, domain_trained_ppls):
    """Analyze specialization patterns in the mixed-trained adapters."""
    log("\n[Phase 6] Analyzing specialization...")

    N, K = ppl_matrix.shape

    # 1. Per-adapter best domain
    best_domains = []
    for i in range(N):
        best_k = np.argmin(ppl_matrix[i])
        best_domains.append(domains_list[best_k])
    log(f"  Best domains: {best_domains}")

    unique_best = len(set(best_domains))
    log(f"  Unique best domains: {unique_best}/{K}")

    # 2. PPL improvement over base for each adapter's best domain
    improvements = []
    for i in range(N):
        best_k = np.argmin(ppl_matrix[i])
        best_d = domains_list[best_k]
        base = base_ppls.get(best_d, float("inf"))
        if base != float("inf"):
            imp = (base - ppl_matrix[i, best_k]) / base * 100
            improvements.append(imp)
    log(f"  Mean improvement on best domain: {np.mean(improvements):.1f}%")

    # 3. Compute silhouette score
    # Normalize PPL matrix: for each adapter, compute relative PPL profile
    # (subtract row mean, divide by row std)
    row_means = np.mean(ppl_matrix, axis=1, keepdims=True)
    row_stds = np.std(ppl_matrix, axis=1, keepdims=True)
    row_stds = np.maximum(row_stds, 1e-6)
    normalized = (ppl_matrix - row_means) / row_stds

    # Assign each adapter to its best domain as cluster label
    cluster_labels = [np.argmin(ppl_matrix[i]) for i in range(N)]

    # Compute silhouette manually (avoid sklearn dependency)
    # silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
    # a(i) = mean distance to same-cluster points
    # b(i) = min mean distance to other-cluster points
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(normalized, normalized, metric='euclidean')

    silhouette_scores = []
    for i in range(N):
        ci = cluster_labels[i]
        same_cluster = [j for j in range(N) if j != i and cluster_labels[j] == ci]
        if len(same_cluster) == 0:
            # Singleton cluster
            a_i = 0.0
        else:
            a_i = np.mean([dist_matrix[i, j] for j in same_cluster])

        # Find nearest other cluster
        other_clusters = set(cluster_labels) - {ci}
        if len(other_clusters) == 0:
            silhouette_scores.append(0.0)
            continue

        b_i = float("inf")
        for oc in other_clusters:
            oc_members = [j for j in range(N) if cluster_labels[j] == oc]
            mean_dist = np.mean([dist_matrix[i, j] for j in oc_members])
            b_i = min(b_i, mean_dist)

        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        silhouette_scores.append(s_i)

    mean_silhouette = np.mean(silhouette_scores)
    log(f"  Mean silhouette score: {mean_silhouette:.4f}")

    # 4. Entropy of domain preferences
    from collections import Counter
    counts = Counter(best_domains)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(min(N, K))
    log(f"  Domain preference entropy: {entropy:.3f} / {max_entropy:.3f} (max)")

    # 5. PPL variance across adapters for each domain
    domain_variance = {}
    for ki, domain in enumerate(domains_list):
        ppls = ppl_matrix[:, ki]
        cv = np.std(ppls) / np.mean(ppls) * 100
        domain_variance[domain] = {
            "mean": round(float(np.mean(ppls)), 2),
            "std": round(float(np.std(ppls)), 2),
            "cv_pct": round(cv, 2),
            "min": round(float(np.min(ppls)), 2),
            "max": round(float(np.max(ppls)), 2),
        }
    log(f"  Per-domain PPL CV: {', '.join(f'{d}={v['cv_pct']:.1f}%' for d, v in domain_variance.items())}")

    # 6. Compare with domain-trained
    comparison = {}
    for domain in domains_list:
        dt_ppl = domain_trained_ppls.get(domain, {}).get(domain, float("inf"))
        # Best mixed adapter for this domain
        ki = domains_list.index(domain)
        best_mixed = float(np.min(ppl_matrix[:, ki]))
        base = base_ppls.get(domain, float("inf"))
        comparison[domain] = {
            "base_ppl": round(base, 2),
            "domain_trained_ppl": round(dt_ppl, 2),
            "best_mixed_ppl": round(best_mixed, 2),
            "mixed_vs_domain_trained_pct": round((best_mixed - dt_ppl) / dt_ppl * 100, 2) if dt_ppl > 0 else None,
        }
        log(f"  {domain}: base={base:.2f}, domain-trained={dt_ppl:.2f}, "
            f"best-mixed={best_mixed:.2f} ({comparison[domain]['mixed_vs_domain_trained_pct']:+.1f}%)")

    # 7. Routing test: for each domain, pick the mixed adapter with lowest PPL
    routing_ppls = {}
    for ki, domain in enumerate(domains_list):
        best_ai = np.argmin(ppl_matrix[:, ki])
        routing_ppls[domain] = {
            "selected_adapter": int(best_ai),
            "routed_ppl": round(float(ppl_matrix[best_ai, ki]), 2),
        }

    # K1 check
    k1_pass = mean_silhouette >= 0.2
    log(f"\n  K1 (silhouette >= 0.2): {'PASS' if k1_pass else 'FAIL'} "
        f"(silhouette = {mean_silhouette:.4f})")

    # K2 check: mixed-trained worse than domain-trained?
    worse_count = sum(1 for d in domains_list
                      if comparison[d]["mixed_vs_domain_trained_pct"] is not None
                      and comparison[d]["mixed_vs_domain_trained_pct"] > 0)
    k2_pass = worse_count < len(domains_list)  # Not ALL worse
    log(f"  K2 (mixed not all worse than domain-trained): {'PASS' if k2_pass else 'FAIL'} "
        f"({worse_count}/{len(domains_list)} worse)")

    return {
        "best_domains": best_domains,
        "unique_best_domains": unique_best,
        "mean_silhouette": round(mean_silhouette, 4),
        "silhouette_scores": [round(s, 4) for s in silhouette_scores],
        "domain_preference_entropy": round(entropy, 3),
        "max_entropy": round(max_entropy, 3),
        "domain_variance": domain_variance,
        "comparison": comparison,
        "routing": routing_ppls,
        "improvements_on_best": [round(v, 2) for v in improvements],
        "k1_pass": k1_pass,
        "k2_worse_count": worse_count,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Experiment: adapter_specialization_emergence")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Prepare data
    all_train_texts, domain_val_paths = phase_prepare_mixed_data()

    # Phase 2: Base PPL
    base_ppls = phase_base_ppl(domain_val_paths)

    # Load the N=24 Grassmannian skeleton (reuse first 10 for our 10 adapters)
    skeleton_path = SOURCE_ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"
    log(f"\nLoading Grassmannian skeleton from {skeleton_path}...")
    skeleton = dict(np.load(str(skeleton_path)))

    # Phase 3: Train 10 mixed adapters
    MIXED_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    train_results = []
    for ai in range(N_ADAPTERS):
        result = phase_train_mixed_adapter(ai, all_train_texts, skeleton)
        train_results.append(result)

    # Phase 4: Evaluate mixed adapters on all domains
    ppl_matrix, domains_list = phase_evaluate_mixed(domain_val_paths, skeleton)

    # Phase 5: Evaluate domain-trained baselines
    domain_trained_ppls = phase_evaluate_domain_trained(domain_val_paths)

    # Phase 6: Analyze
    analysis = phase_analyze(ppl_matrix, domains_list, base_ppls, domain_trained_ppls)

    # Save results
    total_time = time.time() - t0

    results = {
        "experiment": "adapter_specialization_emergence",
        "n_adapters": N_ADAPTERS,
        "n_domains": len(SELECTED_DOMAINS),
        "selected_domains": SELECTED_DOMAINS,
        "train_iters": TRAIN_ITERS,
        "lora_rank": LORA_RANK,
        "total_time_s": round(total_time, 1),
        "base_ppls": {k: round(v, 2) for k, v in base_ppls.items()},
        "train_results": train_results,
        "ppl_matrix": ppl_matrix.tolist(),
        "domains_list": domains_list,
        "analysis": analysis,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    RESULTS_FILE.write_text(json.dumps(convert(results), indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Silhouette score: {analysis['mean_silhouette']:.4f} (threshold: 0.2)")
    log(f"  Unique best domains: {analysis['unique_best_domains']}/{len(SELECTED_DOMAINS)}")
    log(f"  Domain preference entropy: {analysis['domain_preference_entropy']:.3f}/{analysis['max_entropy']:.3f}")
    log(f"  K1 (specialization): {'PASS' if analysis['k1_pass'] else 'FAIL'}")
    log(f"  Domains where mixed worse than domain-trained: {analysis['k2_worse_count']}/{len(SELECTED_DOMAINS)}")

    verdict = "SUPPORTED" if analysis["k1_pass"] else "KILLED"
    log(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
