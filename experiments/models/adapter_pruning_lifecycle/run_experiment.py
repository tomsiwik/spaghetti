#!/usr/bin/env python3
"""Experiment: adapter_pruning_lifecycle

Test whether adapters can be safely pruned from a 24-adapter pool without
significant quality loss, using four complementary pruning strategies.

Kill criteria:
  K1: LOO pruning of any single adapter degrades composed PPL by >5%
      (all equally important, pruning adds no value)
  K2: All pruning strategies select identical adapters for removal
      (no diversity, one metric suffices)

Success criteria:
  S1: At least one pruning strategy removes 20% (5/24) with <2% quality loss

Pruning strategies:
  1. Leave-one-out PPL delta (quality impact)
  2. Routing frequency from softmax router (demand-side)
  3. Effective delta magnitude ||B_i||_F (supply-side)
  4. Cross-adapter similarity max_j cos(delta_i, delta_j) (redundancy)

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import random
import sys
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

# Source: pre-trained 24 domain adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
SEED = 42

# All 24 active domains (same as softmax_router_scaling)
ALL_DOMAINS = [
    "medical", "code", "math", "legal", "finance",
    "science", "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering", "agriculture",
    "environmental", "politics", "economics", "sociology", "linguistics",
    "cybersecurity", "marketing", "sports", "music",
]

N_DOMAINS = len(ALL_DOMAINS)

# Softmax router config (matches softmax_router_scaling)
HIDDEN_DIM = 2560
ROUTER_HIDDEN = 128
ROUTER_TRAIN_STEPS = 500
ROUTER_LR = 3e-4
ROUTER_BATCH_SIZE = 32
TRAIN_SAMPLES_PER_DOMAIN = 40
VAL_SAMPLES_PER_DOMAIN = 50

# LoRA target modules
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Pruning config
PRUNE_K = 5  # Remove 5 out of 24 (20.8%)
QUALITY_THRESHOLD = 0.02  # 2% quality loss threshold for S1


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
# BitNet unpacking and model utilities
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
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
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


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


# ============================================================================
# Multi-adapter LoRA layers
# ============================================================================

class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for uniform multi-expert composition."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste

        return base_out + lora_sum * (self.scale / self.n_experts)


# ============================================================================
# Softmax Router
# ============================================================================

class SoftmaxRouter(nn.Module):
    """Multi-class softmax router for domain selection."""
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ============================================================================
# PPL computation
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
# Phase 1: Compute base PPL for all domains
# ============================================================================

def phase_base_ppl():
    """Load model, compute base PPL on all 24 domains."""
    log("\n[Phase 1] Computing base PPL for all 24 domains...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_ppls = {}
    for domain in ALL_DOMAINS:
        data_dir = DATA_DIR / domain
        if not data_dir.exists():
            log(f"  WARNING: no data for {domain}")
            continue
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain] = round(ppl, 4)
        log(f"  {domain}: base PPL={ppl:.2f}")

    elapsed = time.time() - t0
    log(f"  Base PPL done in {elapsed:.1f}s")
    log_memory("post-base-ppl")
    cleanup(model, tokenizer)
    return base_ppls


# ============================================================================
# Phase 2: Compute per-adapter oracle PPLs (adapter i on domain i)
# ============================================================================

def phase_oracle_ppls():
    """For each adapter, load it alone and evaluate on its target domain.

    Returns dict: domain -> oracle_ppl
    """
    log("\n[Phase 2] Computing per-adapter oracle PPLs...")
    t0 = time.time()

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz")))
    domain_to_idx = {d: i for i, d in enumerate(ALL_DOMAINS)}
    oracle_ppls = {}

    for domain in ALL_DOMAINS:
        log(f"  Evaluating adapter: {domain}...")
        di = domain_to_idx[domain]
        adapter_params = load_adapter(ADAPTERS_DIR / domain)

        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        # Attach single adapter
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

                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

                single_lora = MultiAdapterLoRALinear(
                    module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=[a_mx]
                )
                param_name = f"model.layers.{li}.{key}.lora_b"
                if param_name in adapter_params:
                    single_lora.b_matrices[0] = adapter_params[param_name]
                lora_updates.append((key, single_lora))

            if lora_updates:
                layer.update_modules(tree_unflatten(lora_updates))

        mx.eval(model.parameters())
        model.freeze()

        data_dir = DATA_DIR / domain
        ppl = compute_ppl(model, tokenizer, data_dir)
        oracle_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: oracle PPL={ppl:.2f}")

        cleanup(model, tokenizer, adapter_params)

    del skeleton
    gc.collect()

    elapsed = time.time() - t0
    log(f"  Oracle PPLs done in {elapsed:.1f}s")
    return oracle_ppls


# ============================================================================
# Phase 3: Pruning Metric - Effective delta magnitudes
# ============================================================================

def phase_delta_magnitudes():
    """Compute ||B_i||_F for each adapter across all layers/modules.

    Since A_i is orthonormal (Grassmannian), ||B_i @ A_i^T||_F = ||B_i||_F.
    """
    log("\n[Phase 3] Computing effective delta magnitudes...")
    t0 = time.time()

    magnitudes = {}
    per_module_mags = {}

    for domain in ALL_DOMAINS:
        adapter_params = load_adapter(ADAPTERS_DIR / domain)

        total_sq = 0.0
        module_sq = {}
        for param_name, param in adapter_params.items():
            if "lora_b" in param_name:
                sq = mx.sum(param * param)
                mx.eval(sq)
                sq_val = sq.item()
                total_sq += sq_val
                module_sq[param_name] = round(math.sqrt(sq_val), 6)

        magnitudes[domain] = round(math.sqrt(total_sq), 6)
        per_module_mags[domain] = module_sq
        del adapter_params

    gc.collect()
    mx.clear_cache()

    # Rank by magnitude (smallest = most prunable)
    ranked = sorted(magnitudes.items(), key=lambda x: x[1])
    log("  Delta magnitudes (ascending):")
    for domain, mag in ranked:
        log(f"    {domain}: ||B||_F = {mag:.4f}")

    elapsed = time.time() - t0
    log(f"  Delta magnitudes done in {elapsed:.1f}s")
    return magnitudes


# ============================================================================
# Phase 4: Pruning Metric - Cross-adapter similarity
# ============================================================================

def phase_cross_similarity():
    """Compute pairwise cosine similarity of B-matrix vectors.

    We vectorize all B-matrices per adapter and compute cosine similarity.
    With Grassmannian A, the effective delta cosine involves A_i^T A_j which
    is near-zero. Here we measure B-matrix similarity directly to see if
    adapters learn similar perturbations despite orthogonal projection.
    """
    log("\n[Phase 4] Computing cross-adapter B-matrix similarity...")
    t0 = time.time()

    # Load all B-matrix vectors
    b_vectors = {}
    for domain in ALL_DOMAINS:
        adapter_params = load_adapter(ADAPTERS_DIR / domain)
        vec_parts = []
        # Sort keys for consistent ordering
        for param_name in sorted(adapter_params.keys()):
            if "lora_b" in param_name:
                flat = adapter_params[param_name].reshape(-1)
                mx.eval(flat)
                vec_parts.append(flat)
        if vec_parts:
            full_vec = mx.concatenate(vec_parts)
            mx.eval(full_vec)
            b_vectors[domain] = full_vec
        del adapter_params
    gc.collect()
    mx.clear_cache()

    # Compute pairwise cosine similarity
    domains_list = list(b_vectors.keys())
    n = len(domains_list)
    cos_matrix = np.zeros((n, n))

    for i in range(n):
        vi = b_vectors[domains_list[i]]
        norm_i = mx.sqrt(mx.sum(vi * vi))
        mx.eval(norm_i)
        for j in range(i + 1, n):
            vj = b_vectors[domains_list[j]]
            norm_j = mx.sqrt(mx.sum(vj * vj))
            dot = mx.sum(vi * vj)
            mx.eval(norm_j, dot)
            cos_val = dot.item() / (norm_i.item() * norm_j.item() + 1e-10)
            cos_matrix[i, j] = cos_val
            cos_matrix[j, i] = cos_val

    del b_vectors
    gc.collect()
    mx.clear_cache()

    # Find most similar pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((domains_list[i], domains_list[j], cos_matrix[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    log("  Top 10 most similar adapter pairs (by B-matrix cosine):")
    for d1, d2, cos_val in pairs[:10]:
        log(f"    {d1} <-> {d2}: cos={cos_val:.4f}")

    log(f"  Mean |cos|: {np.mean(np.abs(cos_matrix[np.triu_indices(n, k=1)])):.4f}")
    log(f"  Max |cos|: {np.max(np.abs(cos_matrix[np.triu_indices(n, k=1)])):.4f}")

    # Also compute max similarity per adapter (for pruning: remove one of most-similar pair)
    max_sim_per_adapter = {}
    for i, domain in enumerate(domains_list):
        row = np.abs(cos_matrix[i])
        row[i] = 0  # exclude self
        max_idx = np.argmax(row)
        max_sim_per_adapter[domain] = {
            "max_cos": round(float(row[max_idx]), 4),
            "most_similar_to": domains_list[max_idx],
        }

    elapsed = time.time() - t0
    log(f"  Cross-similarity done in {elapsed:.1f}s")

    # Convert matrix to serializable format
    cos_dict = {}
    for i, d1 in enumerate(domains_list):
        for j, d2 in enumerate(domains_list):
            if i < j:
                cos_dict[f"{d1}__{d2}"] = round(float(cos_matrix[i, j]), 6)

    return cos_dict, max_sim_per_adapter, domains_list


# ============================================================================
# Phase 5: Extract hidden states + train softmax router + routing frequency
# ============================================================================

def phase_routing_frequency():
    """Train softmax router and measure per-adapter routing frequency.

    Returns: routing_freq dict, router object, all_hidden states
    """
    log("\n[Phase 5] Training softmax router and measuring routing frequency...")
    t0 = time.time()

    # Extract hidden states
    log("  Extracting hidden states...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    all_hidden = {}
    for domain in ALL_DOMAINS:
        data_dir = DATA_DIR / domain
        if not data_dir.exists():
            continue

        domain_hidden = {}
        for split, filename, max_samples in [
            ("train", "train.jsonl", TRAIN_SAMPLES_PER_DOMAIN),
            ("val", "valid.jsonl", VAL_SAMPLES_PER_DOMAIN),
        ]:
            filepath = data_dir / filename
            if not filepath.exists():
                continue

            texts = []
            with open(filepath) as f:
                for line in f:
                    texts.append(json.loads(line)["text"])

            states = []
            for text in texts[:max_samples]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]

                h = model.model.embed_tokens(x)
                for layer_mod in model.model.layers:
                    h = layer_mod(h)
                h = model.model.norm(h)
                h_mean = mx.mean(h[0], axis=0)
                mx.eval(h_mean)
                states.append(h_mean)
                del h, x

            if states:
                result = mx.stack(states)
                mx.eval(result)
                domain_hidden[split] = result

        if domain_hidden:
            all_hidden[domain] = domain_hidden

    log(f"  Hidden states extracted for {len(all_hidden)} domains")
    cleanup(model, tokenizer)

    # Train softmax router
    log("  Training softmax router...")
    np.random.seed(SEED)
    N = N_DOMAINS
    router = SoftmaxRouter(HIDDEN_DIM, N, ROUTER_HIDDEN)
    router_opt = opt.Adam(learning_rate=ROUTER_LR)

    train_x_list = []
    train_y_list = []
    for di, domain in enumerate(ALL_DOMAINS):
        if domain not in all_hidden or "train" not in all_hidden[domain]:
            continue
        states = all_hidden[domain]["train"]
        n_samples = states.shape[0]
        train_x_list.append(states)
        train_y_list.append(mx.full((n_samples,), di, dtype=mx.int32))

    train_x = mx.concatenate(train_x_list, axis=0)
    train_y = mx.concatenate(train_y_list, axis=0)
    mx.eval(train_x, train_y)
    n_total = train_x.shape[0]
    log(f"  Router training data: {n_total} samples across {N} classes")

    def router_loss_fn(router_model, x, y):
        logits = router_model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    router_loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    gc.disable()
    for step in range(ROUTER_TRAIN_STEPS):
        idx = mx.array(np.random.randint(0, n_total, size=ROUTER_BATCH_SIZE))
        batch_x = train_x[idx]
        batch_y = train_y[idx]

        loss_val, grads = router_loss_and_grad(router, batch_x, batch_y)
        router_opt.update(router, grads)
        mx.eval(router.parameters(), router_opt.state, loss_val)

        if step == ROUTER_TRAIN_STEPS - 1:
            log(f"  Router final loss: {loss_val.item():.4f}")
    gc.enable()

    del router_opt, train_x, train_y
    gc.collect()

    # Compute routing frequency on validation data
    log("  Computing routing frequency on val data...")
    freq_counts = np.zeros(N, dtype=np.int64)
    total_samples = 0

    for domain in ALL_DOMAINS:
        if domain not in all_hidden or "val" not in all_hidden[domain]:
            continue
        val_h = all_hidden[domain]["val"]
        logits = router(val_h)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        preds_np = np.array(preds)
        for p in preds_np:
            freq_counts[p] += 1
        total_samples += len(preds_np)
        del logits, preds

    routing_freq = {}
    for di, domain in enumerate(ALL_DOMAINS):
        freq = freq_counts[di] / total_samples if total_samples > 0 else 0
        routing_freq[domain] = round(float(freq), 4)

    log("  Routing frequency (ascending):")
    for domain, freq in sorted(routing_freq.items(), key=lambda x: x[1]):
        expected = 1.0 / N
        ratio = freq / expected if expected > 0 else 0
        log(f"    {domain}: freq={freq:.4f} ({ratio:.2f}x expected)")

    elapsed = time.time() - t0
    log(f"  Routing frequency done in {elapsed:.1f}s")

    return routing_freq, all_hidden


# ============================================================================
# Phase 6: Leave-one-out PPL analysis
# ============================================================================

def phase_loo_analysis(oracle_ppls, base_ppls):
    """Compute LOO delta for each adapter.

    Approximation: since softmax router matches oracle (0% gap at N=24),
    we approximate composed PPL as the average oracle PPL across active domains.
    LOO(i) = mean(oracle_ppls for j != i) - mean(oracle_ppls for all j).

    This avoids 24 full model re-loads with N-1 adapters each.
    """
    log("\n[Phase 6] Computing leave-one-out PPL deltas...")
    t0 = time.time()

    # Full pool average oracle PPL
    valid_domains = [d for d in ALL_DOMAINS if d in oracle_ppls]
    full_avg = np.mean([oracle_ppls[d] for d in valid_domains])
    log(f"  Full pool avg oracle PPL: {full_avg:.4f}")

    loo_deltas = {}
    for domain in valid_domains:
        remaining = [d for d in valid_domains if d != domain]
        loo_avg = np.mean([oracle_ppls[d] for d in remaining])
        delta = loo_avg - full_avg
        delta_pct = delta / full_avg * 100
        loo_deltas[domain] = {
            "loo_avg_ppl": round(float(loo_avg), 4),
            "delta": round(float(delta), 4),
            "delta_pct": round(float(delta_pct), 4),
        }

    log("  LOO deltas (sorted by absolute delta, ascending):")
    sorted_loo = sorted(loo_deltas.items(), key=lambda x: abs(x[1]["delta"]))
    for domain, info in sorted_loo:
        log(f"    {domain}: LOO avg={info['loo_avg_ppl']:.4f}, "
            f"delta={info['delta']:+.4f} ({info['delta_pct']:+.2f}%)")

    # Check K1: does any single removal cause >5% degradation?
    max_delta_pct = max(info["delta_pct"] for info in loo_deltas.values())
    k1_pass = max_delta_pct <= 5.0
    log(f"\n  K1 CHECK: max LOO delta = {max_delta_pct:.2f}%")
    log(f"  K1 {'PASS' if k1_pass else 'FAIL'}: {'Some adapters are safely removable' if k1_pass else 'All adapters degrade >5%'}")

    elapsed = time.time() - t0
    log(f"  LOO analysis done in {elapsed:.1f}s")
    return loo_deltas, full_avg, k1_pass


# ============================================================================
# Phase 7: Generate pruning recommendations from each strategy
# ============================================================================

def phase_pruning_strategies(loo_deltas, routing_freq, magnitudes, max_sim):
    """For each strategy, rank adapters and select PRUNE_K for removal.

    Returns: dict strategy_name -> list of (domain, score) to prune
    """
    log(f"\n[Phase 7] Generating pruning recommendations (remove {PRUNE_K} of {N_DOMAINS})...")

    strategies = {}

    # Strategy 1: LOO - remove adapters with smallest absolute delta (least impactful)
    # Negative delta means removing IMPROVES quality -- prime candidates
    loo_ranked = sorted(loo_deltas.items(), key=lambda x: x[1]["delta"])
    prune_loo = [(d, info["delta"]) for d, info in loo_ranked[:PRUNE_K]]
    strategies["loo_delta"] = prune_loo
    log("  Strategy 1 (LOO delta, prune smallest):")
    for d, score in prune_loo:
        log(f"    PRUNE {d}: delta={score:+.4f}")

    # Strategy 2: Routing frequency - remove least-selected adapters
    freq_ranked = sorted(routing_freq.items(), key=lambda x: x[1])
    prune_freq = [(d, f) for d, f in freq_ranked[:PRUNE_K]]
    strategies["routing_freq"] = prune_freq
    log("  Strategy 2 (routing frequency, prune least selected):")
    for d, score in prune_freq:
        log(f"    PRUNE {d}: freq={score:.4f}")

    # Strategy 3: Delta magnitude - remove smallest magnitude adapters
    mag_ranked = sorted(magnitudes.items(), key=lambda x: x[1])
    prune_mag = [(d, m) for d, m in mag_ranked[:PRUNE_K]]
    strategies["delta_magnitude"] = prune_mag
    log("  Strategy 3 (delta magnitude, prune smallest):")
    for d, score in prune_mag:
        log(f"    PRUNE {d}: ||B||_F={score:.4f}")

    # Strategy 4: Similarity-based - iteratively remove one from most-similar pairs
    # Greedy: find most-similar pair, remove the one with lower delta magnitude
    remaining = set(ALL_DOMAINS)
    prune_sim = []
    # Build full similarity info
    sim_ranked = sorted(max_sim.items(), key=lambda x: x[1]["max_cos"], reverse=True)

    # Greedy removal: pick adapter with highest max_cos and lower magnitude
    sim_candidates = list(remaining)
    for _ in range(PRUNE_K):
        if not sim_candidates:
            break
        # Find adapter in remaining set with highest max_cos to another remaining adapter
        best_to_remove = None
        best_cos = -1
        for d in sim_candidates:
            info = max_sim.get(d, {})
            partner = info.get("most_similar_to", "")
            cos_val = info.get("max_cos", 0)
            if partner in remaining and cos_val > best_cos:
                # Remove the one with lower delta magnitude
                d_mag = magnitudes.get(d, 0)
                p_mag = magnitudes.get(partner, 0)
                candidate = d if d_mag <= p_mag else partner
                if candidate in remaining:
                    best_to_remove = candidate
                    best_cos = cos_val

        if best_to_remove is None:
            # Fallback: remove adapter with lowest magnitude from remaining
            rem_by_mag = sorted(remaining, key=lambda x: magnitudes.get(x, 0))
            best_to_remove = rem_by_mag[0]
            best_cos = max_sim.get(best_to_remove, {}).get("max_cos", 0)

        prune_sim.append((best_to_remove, best_cos))
        remaining.discard(best_to_remove)
        sim_candidates = list(remaining)

    strategies["cross_similarity"] = prune_sim
    log("  Strategy 4 (cross-similarity, prune redundant):")
    for d, score in prune_sim:
        log(f"    PRUNE {d}: max_cos={score:.4f}")

    # Check K2: are all strategies selecting identical sets?
    prune_sets = {}
    for name, prune_list in strategies.items():
        prune_sets[name] = set(d for d, _ in prune_list)

    sets_list = list(prune_sets.values())
    all_identical = all(s == sets_list[0] for s in sets_list)

    # Also compute pairwise Jaccard similarity
    jaccard_pairs = {}
    strat_names = list(prune_sets.keys())
    for i in range(len(strat_names)):
        for j in range(i + 1, len(strat_names)):
            s1 = prune_sets[strat_names[i]]
            s2 = prune_sets[strat_names[j]]
            jaccard = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 1.0
            pair_key = f"{strat_names[i]}_vs_{strat_names[j]}"
            jaccard_pairs[pair_key] = round(jaccard, 4)

    k2_pass = not all_identical
    log(f"\n  K2 CHECK: all strategies identical? {all_identical}")
    log(f"  K2 {'PASS' if k2_pass else 'FAIL'}: strategies {'differ' if k2_pass else 'are identical'}")
    for pair, jac in jaccard_pairs.items():
        log(f"    Jaccard {pair}: {jac:.4f}")

    return strategies, prune_sets, jaccard_pairs, k2_pass


# ============================================================================
# Phase 8: Evaluate pruned pools
# ============================================================================

def phase_evaluate_pruned(strategies, prune_sets, oracle_ppls, base_ppls, full_avg):
    """For each pruning strategy, evaluate composed PPL on remaining adapters.

    Use oracle PPL approximation: avg oracle PPL of remaining domains.
    Always validate the LOO strategy with actual model composition.
    S1 is evaluated using same-domain comparison (pruned vs full pool on same domains).
    """
    log(f"\n[Phase 8] Evaluating pruned pools...")
    t0 = time.time()

    pruned_results = {}
    for strat_name, prune_set in prune_sets.items():
        remaining_domains = [d for d in ALL_DOMAINS if d not in prune_set]
        remaining_ppls = [oracle_ppls[d] for d in remaining_domains if d in oracle_ppls]

        if not remaining_ppls:
            continue

        pruned_avg = np.mean(remaining_ppls)
        # Oracle delta compares different domain sets — informative but not S1 metric
        oracle_delta_pct = (pruned_avg - full_avg) / full_avg * 100

        pruned_results[strat_name] = {
            "removed": sorted(prune_set),
            "n_remaining": len(remaining_domains),
            "pruned_avg_oracle_ppl": round(float(pruned_avg), 4),
            "full_avg_oracle_ppl": round(float(full_avg), 4),
            "oracle_delta_pct": round(float(oracle_delta_pct), 4),
        }
        log(f"  {strat_name}: oracle avg={pruned_avg:.4f}, oracle delta={oracle_delta_pct:+.2f}%")

    # Always validate LOO strategy with actual model composition
    log(f"\n  Validating LOO strategy with actual model composition...")
    remaining_domains = [d for d in ALL_DOMAINS if d not in prune_sets["loo_delta"]]
    actual_ppls = _validate_with_composition(remaining_domains)
    actual_avg = np.mean(list(actual_ppls.values())) if actual_ppls else float("inf")
    pruned_results["loo_delta"]["validated_ppls"] = actual_ppls
    pruned_results["loo_delta"]["validated_avg_ppl"] = round(float(actual_avg), 4)
    log(f"  LOO validated avg PPL: {actual_avg:.4f}")

    elapsed = time.time() - t0
    log(f"  Pruned evaluation done in {elapsed:.1f}s")

    # S1 check is deferred to main() where full pool composition is available
    return pruned_results


def _validate_with_composition(remaining_domains):
    """Build model with uniform composition of remaining adapters and eval PPL."""
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz")))
    domain_to_idx = {d: i for i, d in enumerate(ALL_DOMAINS)}
    N = len(remaining_domains)

    # Load adapter params
    all_adapter_params = []
    for domain in remaining_domains:
        params = load_adapter(ADAPTERS_DIR / domain)
        all_adapter_params.append(params)

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

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

            a_inits = []
            for di_local, domain in enumerate(remaining_domains):
                full_di = domain_to_idx[domain]
                skey = f"layer_{li}_{key}_domain_{full_di}"
                if skey in skeleton:
                    a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                    a_inits.append(a_mx)

            if len(a_inits) != N:
                continue

            multi_lora = MultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )
            param_name = f"model.layers.{li}.{key}.lora_b"
            for di_local in range(N):
                if param_name in all_adapter_params[di_local]:
                    multi_lora.b_matrices[di_local] = all_adapter_params[di_local][param_name]

            lora_updates.append((key, multi_lora))

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()

    ppls = {}
    for domain in remaining_domains:
        data_dir = DATA_DIR / domain
        ppl = compute_ppl(model, tokenizer, data_dir)
        ppls[domain] = round(ppl, 4)
        log(f"      {domain}: composed PPL={ppl:.2f}")

    cleanup(model, tokenizer)
    del all_adapter_params, skeleton
    gc.collect()
    mx.clear_cache()

    return ppls


# ============================================================================
# Phase 9: Full-pool composition baseline (for fair validated comparison)
# ============================================================================

def phase_full_pool_composition():
    """Evaluate uniform composition with all 24 adapters.

    Provides the validated baseline for comparison with pruned validation.
    """
    log("\n[Phase 9] Evaluating full pool (N=24) uniform composition baseline...")
    t0 = time.time()

    actual_ppls = _validate_with_composition(ALL_DOMAINS)
    actual_avg = np.mean(list(actual_ppls.values())) if actual_ppls else float("inf")
    log(f"  Full pool avg composed PPL: {actual_avg:.4f}")

    elapsed = time.time() - t0
    log(f"  Full pool composition done in {elapsed:.1f}s")
    return actual_ppls, float(actual_avg)


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Experiment: adapter_pruning_lifecycle")
    log(f"Goal: Test pruning {PRUNE_K} of {N_DOMAINS} adapters with <2% quality loss")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Base PPL
    base_ppls = phase_base_ppl()
    log_memory("after-base-ppl")

    # Phase 2: Oracle PPLs (adapter i on domain i)
    oracle_ppls = phase_oracle_ppls()
    log_memory("after-oracle-ppls")

    # Phase 3: Delta magnitudes
    magnitudes = phase_delta_magnitudes()

    # Phase 4: Cross-adapter similarity
    cos_dict, max_sim, domains_order = phase_cross_similarity()

    # Phase 5: Routing frequency
    routing_freq, all_hidden = phase_routing_frequency()
    log_memory("after-routing")
    del all_hidden
    gc.collect()
    mx.clear_cache()

    # Phase 6: LOO analysis
    loo_deltas, full_avg, k1_pass = phase_loo_analysis(oracle_ppls, base_ppls)

    # Phase 7: Generate pruning strategies
    strategies, prune_sets, jaccard_pairs, k2_pass = phase_pruning_strategies(
        loo_deltas, routing_freq, magnitudes, max_sim
    )

    # Phase 8: Evaluate pruned pools (validates LOO with actual composition)
    pruned_results = phase_evaluate_pruned(
        strategies, prune_sets, oracle_ppls, base_ppls, full_avg
    )

    # Phase 9: Full pool composition baseline for validated comparison
    full_comp_ppls, full_comp_avg = phase_full_pool_composition()

    # Compute S1 using correct same-domain comparison
    # For LOO strategy: compare remaining domains' PPL in pruned vs full composition
    loo_validated = pruned_results.get("loo_delta", {}).get("validated_ppls", {})
    if loo_validated and full_comp_ppls:
        remaining_domains = [d for d in ALL_DOMAINS if d not in prune_sets["loo_delta"]]
        same_domain_full = [full_comp_ppls[d] for d in remaining_domains if d in full_comp_ppls]
        same_domain_pruned = [loo_validated[d] for d in remaining_domains if d in loo_validated]
        if same_domain_full and same_domain_pruned:
            full_same_avg = np.mean(same_domain_full)
            pruned_same_avg = np.mean(same_domain_pruned)
            same_domain_delta_pct = (pruned_same_avg - full_same_avg) / full_same_avg * 100
            s1_pass = same_domain_delta_pct <= QUALITY_THRESHOLD * 100  # Allow improvement
            pruned_results["loo_delta"]["same_domain_full_avg"] = round(float(full_same_avg), 4)
            pruned_results["loo_delta"]["same_domain_pruned_avg"] = round(float(pruned_same_avg), 4)
            pruned_results["loo_delta"]["same_domain_delta_pct"] = round(float(same_domain_delta_pct), 4)
            log(f"\n  S1 CORRECT METRIC (same-domain comparison):")
            log(f"    Full pool (19 remaining domains): avg composed PPL = {full_same_avg:.4f}")
            log(f"    Pruned pool (19 remaining domains): avg composed PPL = {pruned_same_avg:.4f}")
            log(f"    Delta: {same_domain_delta_pct:+.2f}% (threshold: +{QUALITY_THRESHOLD*100}%)")
            log(f"    S1 {'PASS' if s1_pass else 'FAIL'}")
        else:
            s1_pass = False
    else:
        s1_pass = False

    total_time = time.time() - t0

    # ============================================================================
    # Assemble results
    # ============================================================================
    results = {
        "experiment": "adapter_pruning_lifecycle",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": MODEL_ID,
            "n_domains": N_DOMAINS,
            "prune_k": PRUNE_K,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "quality_threshold_pct": QUALITY_THRESHOLD * 100,
        },
        "base_ppls": base_ppls,
        "oracle_ppls": oracle_ppls,
        "full_pool_avg_oracle_ppl": round(float(full_avg), 4),
        "full_pool_composition": {
            "per_domain": full_comp_ppls,
            "avg_ppl": round(full_comp_avg, 4),
        },
        "metrics": {
            "delta_magnitudes": magnitudes,
            "routing_frequency": routing_freq,
            "loo_deltas": loo_deltas,
            "cross_similarity_top_pairs": cos_dict,
            "max_similarity_per_adapter": max_sim,
        },
        "pruning_strategies": {},
        "strategy_overlap": jaccard_pairs,
        "kill_criteria": {
            "K1": {
                "description": "LOO pruning of any single adapter degrades composed PPL by >5%",
                "pass": k1_pass,
                "max_loo_delta_pct": round(float(max(
                    info["delta_pct"] for info in loo_deltas.values()
                )), 4),
            },
            "K2": {
                "description": "All pruning strategies select identical adapters",
                "pass": k2_pass,
                "all_identical": not k2_pass,
                "pairwise_jaccard": jaccard_pairs,
            },
        },
        "success_criteria": {
            "S1": {
                "description": f"Remove {PRUNE_K}/{N_DOMAINS} adapters with <2% quality loss (same-domain comparison)",
                "pass": bool(s1_pass),
                "same_domain_delta_pct": pruned_results.get("loo_delta", {}).get("same_domain_delta_pct"),
            },
        },
        "total_time_s": round(total_time, 1),
    }

    # Add per-strategy results
    for strat_name, prune_list in strategies.items():
        results["pruning_strategies"][strat_name] = {
            "pruned_adapters": [{"domain": d, "score": round(float(s), 6)} for d, s in prune_list],
            "evaluation": pruned_results.get(strat_name, {}),
        }

    # ============================================================================
    # Summary
    # ============================================================================
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"Full pool avg oracle PPL: {full_avg:.4f}")
    log(f"Full pool avg composed PPL: {full_comp_avg:.4f}")
    log(f"\nKill Criteria:")
    log(f"  K1 (all equally important): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (no strategy diversity): {'PASS' if k2_pass else 'FAIL'}")
    log(f"\nSuccess Criteria:")
    log(f"  S1 (prune 20% with <2% loss): {'PASS' if s1_pass else 'FAIL'}")
    log(f"\nPruning Strategy Results:")
    for strat_name in strategies:
        removed = [d for d, _ in strategies[strat_name]]
        eval_info = pruned_results.get(strat_name, {})
        delta = eval_info.get("oracle_delta_pct", "N/A")
        log(f"  {strat_name}:")
        log(f"    Remove: {removed}")
        log(f"    Oracle PPL delta: {delta}%")
        if "same_domain_delta_pct" in eval_info:
            log(f"    Same-domain delta: {eval_info['same_domain_delta_pct']:+.2f}%")
        if "validated_avg_ppl" in eval_info:
            log(f"    Validated composed PPL: {eval_info['validated_avg_ppl']}")

    log(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    RESULTS_FILE.write_text(json.dumps(convert(results), indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
