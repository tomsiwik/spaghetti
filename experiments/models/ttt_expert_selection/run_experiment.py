#!/usr/bin/env python3
"""
TTT Expert Selection: Test-Time Training for Runtime Expert Selection on MLX

Tests whether loss-probing and projection-based scoring can select the correct
top-2 experts at runtime WITHOUT a pre-trained router, matching or beating the
learned Gumbel-sigmoid router (avg routed PPL 15.07 on 49 domains).

Strategies:
  1. Exhaustive loss-probe: O(N) forward passes -- oracle upper bound
  2. Arrow-style projection: 1 forward pass + matrix ops -- fastest
  3. Hybrid Arrow+probe: Arrow top-m, then loss-probe m candidates

Kill criteria:
  K1 (id=198): TTT selection overhead > 50% of base generation time for 512 tokens -> KILL
  K2 (id=199): TTT-selected top-2 PPL worse than learned router top-2 (15.07 avg) -> KILL
  K3 (id=200): TTT requires > 10 forward passes to identify correct experts -> KILL

Platform: Apple M5 Pro 48GB, MLX, $0.
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
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse N=50 experiment infrastructure
N50_DIR = EXPERIMENT_DIR.parent / "bitnet_scale_n50"
ADAPTERS_DIR = N50_DIR / "adapters"
DATA_DIR_N50 = N50_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 128
SEED = 42
PREFIX_LEN = 32  # tokens used for probing (shorter = faster TTT)
TOP_K = 2        # adapters to select

# Reference: learned router avg routed PPL from bitnet_scale_n50
REFERENCE_ROUTED_PPL = 15.07


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
# BitLinear unpacking (from N=50 experiment)
# ===========================================================================
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
# TernaryLoRALinear with STE (from N=50)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0, a_init=None):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model, seed=None):
    if seed is not None:
        mx.random.seed(seed)
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def compute_loss_on_tokens(model, token_ids):
    """Compute cross-entropy loss on a sequence of token IDs. Returns avg loss per token."""
    if len(token_ids) < 2:
        return float("inf")
    x = mx.array(token_ids[:-1])[None, :]
    y = mx.array(token_ids[1:])[None, :]
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    mx.eval(loss)
    val = loss.item()
    del logits, loss, x, y
    return val


def compute_ppl(model, tokenizer, data_path, max_batches=20, split="valid"):
    """Compute PPL on validation data."""
    fpath = data_path / f"{split}.jsonl"
    if not fpath.exists():
        return float("inf")

    texts = []
    with open(fpath) as f:
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
        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# Strategy 1: Exhaustive Loss-Probe
# ===========================================================================
def exhaustive_loss_probe(model, tokenizer, token_ids, adapter_names, adapters_dir,
                          prefix_len=32, top_k=2):
    """
    Try every adapter on prefix tokens, select top-k by loss reduction.
    Returns: (selected_names, n_forward_passes, probe_time_s)
    """
    t0 = time.time()
    prefix = token_ids[:prefix_len + 1]  # +1 for target
    if len(prefix) < 3:
        return adapter_names[:top_k], len(adapter_names) + 1, 0.0

    # Base loss (no adapter)
    zero_lora_params(model)
    mx.eval(model.parameters())
    base_loss = compute_loss_on_tokens(model, prefix)

    # Per-adapter loss
    adapter_scores = []
    for name in adapter_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            adapter_scores.append((name, 0.0))
            continue
        params = dict(mx.load(str(adapter_path)))
        zero_lora_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        adapter_loss = compute_loss_on_tokens(model, prefix)
        reduction = base_loss - adapter_loss
        adapter_scores.append((name, reduction))
        del params

    mx.clear_cache()

    # Select top-k by loss reduction
    adapter_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in adapter_scores[:top_k]]

    n_passes = len(adapter_names) + 1  # 1 base + N adapter passes
    elapsed = time.time() - t0
    return selected, n_passes, elapsed


# ===========================================================================
# Strategy 2: Arrow-Style Projection Scoring
# ===========================================================================
def extract_a_matrices(adapters_dir, adapter_names):
    """
    Extract and concatenate all A-matrices from each adapter.
    Returns dict: name -> flat A-matrix concatenation (for projection scoring).
    """
    a_matrices = {}
    for name in adapter_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            continue
        params = dict(mx.load(str(adapter_path)))
        # Collect all lora_a matrices
        a_list = []
        for k, v in sorted(params.items()):
            if "lora_a" in k:
                a_list.append(v)  # shape [in_features, r]
        if a_list:
            # Each A is [d_layer, r]. Concatenate across layers.
            a_concat = mx.concatenate([a.reshape(-1) for a in a_list])
            mx.eval(a_concat)
            a_matrices[name] = a_concat
        del params
    mx.clear_cache()
    return a_matrices


def arrow_projection_scoring(model, tokenizer, token_ids, adapter_names,
                              adapters_dir, a_matrices, prefix_len=32, top_k=2):
    """
    Score adapters by how much the input hidden states project onto adapter A subspaces.
    Uses only self_attn A-matrices (d_model input dim) that match hidden state shape.
    Returns: (selected_names, n_forward_passes, probe_time_s)
    """
    t0 = time.time()
    prefix = token_ids[:prefix_len]
    if len(prefix) < 2:
        return adapter_names[:top_k], 1, 0.0

    # Get mean-pooled hidden state from base model (no adapter)
    zero_lora_params(model)
    mx.eval(model.parameters())
    x = mx.array(prefix)[None, :]
    h = model.model(x)  # [1, seq_len, d]
    h_mean = mx.mean(h, axis=1)  # [1, d]
    mx.eval(h_mean)
    h_flat = h_mean.reshape(-1)  # [d_model]
    d_model = h_flat.shape[0]
    del h, x

    # Score each adapter by projection energy through attn A-matrices only
    scores = {}
    for name in adapter_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            scores[name] = 0.0
            continue
        params = dict(mx.load(str(adapter_path)))
        total_proj = 0.0
        for k, v in sorted(params.items()):
            if "lora_a" in k and v.shape[0] == d_model:
                # v is [d_model, r] -- self_attn projection matrices
                proj = h_flat @ v  # [r]
                proj_energy = mx.sum(proj * proj)
                mx.eval(proj_energy)
                total_proj += proj_energy.item()
        del params
        scores[name] = total_proj

    mx.clear_cache()

    # Select top-k by projection score
    sorted_names = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    selected = sorted_names[:top_k]

    elapsed = time.time() - t0
    return selected, 1, elapsed  # 1 forward pass


# ===========================================================================
# Strategy 2b: Hidden-State Cosine Scoring (like the learned router)
# ===========================================================================
def hidden_state_cosine_scoring(model, tokenizer, token_ids, adapter_names,
                                 domain_centroids, prefix_len=32, top_k=2):
    """
    Compute cosine similarity between input hidden state and per-domain centroids
    (computed offline from training data). This is the same signal the learned
    router uses, but without training a router network.
    Returns: (selected_names, n_forward_passes, probe_time_s)
    """
    t0 = time.time()
    prefix = token_ids[:prefix_len]
    if len(prefix) < 2:
        return adapter_names[:top_k], 1, 0.0

    # Get mean-pooled hidden state
    zero_lora_params(model)
    mx.eval(model.parameters())
    x = mx.array(prefix)[None, :]
    h = model.model(x)
    h_mean = mx.mean(h, axis=1).reshape(-1)  # [d]
    mx.eval(h_mean)
    del h, x

    h_norm = mx.sqrt(mx.sum(h_mean * h_mean))
    mx.eval(h_norm)

    scores = {}
    for name in adapter_names:
        if name not in domain_centroids:
            scores[name] = 0.0
            continue
        centroid = domain_centroids[name]
        c_norm = mx.sqrt(mx.sum(centroid * centroid))
        cos = mx.sum(h_mean * centroid) / (h_norm * c_norm + 1e-10)
        mx.eval(cos)
        scores[name] = cos.item()

    mx.clear_cache()

    sorted_names = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    selected = sorted_names[:top_k]

    elapsed = time.time() - t0
    return selected, 1, elapsed


# ===========================================================================
# Strategy 3: Hybrid Arrow + Selective Probe
# ===========================================================================
def hybrid_arrow_probe(model, tokenizer, token_ids, adapter_names, adapters_dir,
                       a_matrices, prefix_len=32, top_k=2, arrow_top_m=5):
    """
    Arrow-style scoring to get top-m candidates, then loss-probe the top-m.
    Returns: (selected_names, n_forward_passes, probe_time_s)
    """
    t0 = time.time()

    # Step 1: Arrow scoring to get top-m candidates
    arrow_selected, _, _ = arrow_projection_scoring(
        model, tokenizer, token_ids, adapter_names,
        adapters_dir, a_matrices, prefix_len=prefix_len, top_k=arrow_top_m
    )

    # Step 2: Loss-probe only the top-m candidates
    prefix = token_ids[:prefix_len + 1]
    if len(prefix) < 3:
        return arrow_selected[:top_k], arrow_top_m + 1, 0.0

    zero_lora_params(model)
    mx.eval(model.parameters())
    base_loss = compute_loss_on_tokens(model, prefix)

    candidate_scores = []
    for name in arrow_selected:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            candidate_scores.append((name, 0.0))
            continue
        params = dict(mx.load(str(adapter_path)))
        zero_lora_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())
        adapter_loss = compute_loss_on_tokens(model, prefix)
        reduction = base_loss - adapter_loss
        candidate_scores.append((name, reduction))
        del params

    mx.clear_cache()

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in candidate_scores[:top_k]]

    n_passes = 1 + 1 + arrow_top_m  # 1 hidden-state pass + 1 base loss + m probe passes
    elapsed = time.time() - t0
    return selected, n_passes, elapsed


# ===========================================================================
# Strategy 3b: Hybrid Cosine + Selective Probe
# ===========================================================================
def hybrid_cosine_probe(model, tokenizer, token_ids, adapter_names, adapters_dir,
                        domain_centroids, prefix_len=32, top_k=2, cosine_top_m=5):
    """
    Cosine similarity scoring to get top-m candidates, then loss-probe the top-m.
    """
    t0 = time.time()

    # Step 1: Cosine scoring for top-m
    cosine_selected, _, _ = hidden_state_cosine_scoring(
        model, tokenizer, token_ids, adapter_names,
        domain_centroids, prefix_len=prefix_len, top_k=cosine_top_m
    )

    # Step 2: Loss-probe the top-m
    prefix = token_ids[:prefix_len + 1]
    if len(prefix) < 3:
        return cosine_selected[:top_k], cosine_top_m + 1, 0.0

    zero_lora_params(model)
    mx.eval(model.parameters())
    base_loss = compute_loss_on_tokens(model, prefix)

    candidate_scores = []
    for name in cosine_selected:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            candidate_scores.append((name, 0.0))
            continue
        params = dict(mx.load(str(adapter_path)))
        zero_lora_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())
        adapter_loss = compute_loss_on_tokens(model, prefix)
        reduction = base_loss - adapter_loss
        candidate_scores.append((name, reduction))
        del params

    mx.clear_cache()

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in candidate_scores[:top_k]]

    n_passes = 1 + 1 + cosine_top_m  # 1 hidden-state + 1 base + m probes
    elapsed = time.time() - t0
    return selected, n_passes, elapsed


# ===========================================================================
# Phase functions
# ===========================================================================
def phase_load_model():
    """Phase 0: Load and prepare model."""
    log("\n[Phase 0] Loading model...")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()  # freeze base
    # Unfreeze LoRA params
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_a = module.lora_a
                module.lora_b = module.lora_b
    log_memory("model-loaded")
    return model, tokenizer


def phase_collect_data_dirs():
    """Phase 1: Collect data directories for all domains."""
    log("\n[Phase 1] Collecting data directories...")

    # Load N=50 results to get domain list
    n50_results = json.load(open(N50_DIR / "results.json"))
    all_names = [n for n in n50_results["all_names"] if n != "medical"]  # medical has no data

    # Map data directories
    data_dirs = {}
    # Existing data dirs from N=50 experiment
    existing_data_dirs = {
        "code": EXPERIMENT_DIR.parent / "bitnet_ternary_convergence" / "data" / "code",
        "math": EXPERIMENT_DIR.parent / "bitnet_ternary_convergence" / "data" / "math",
        "legal": EXPERIMENT_DIR.parent / "bitnet_ternary_convergence" / "data" / "legal",
        "creative": EXPERIMENT_DIR.parent / "bitnet_ternary_convergence" / "data" / "creative",
        "sql": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "sql",
        "javascript": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "javascript",
        "physics": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "physics",
        "chemistry": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "chemistry",
        "science": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "science",
        "wikitext": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "wikitext",
        "finance": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "finance",
        "cooking": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "cooking",
        "health": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "health",
        "dialogue": EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data" / "dialogue",
        "reasoning": EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data" / "reasoning",
        "instruction": EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data" / "instruction",
        "conciseness": EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data" / "conciseness",
        "safety": EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data" / "safety",
        "multilingual": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "multilingual",
        "coding_style": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "coding_style",
        "summarization": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "summarization",
        "debate": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "debate",
        "translation": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "translation",
        "formal_writing": EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data" / "formal_writing",
    }
    # New domains from N=50 data dir
    for name in all_names:
        if name in existing_data_dirs:
            path = existing_data_dirs[name]
            if path.exists() and (path / "valid.jsonl").exists():
                data_dirs[name] = path
        else:
            path = DATA_DIR_N50 / name
            if path.exists() and (path / "valid.jsonl").exists():
                data_dirs[name] = path

    active_names = [n for n in all_names if n in data_dirs and (ADAPTERS_DIR / n / "adapter.npz").exists()]
    log(f"  Active domains (data + adapter): {len(active_names)}")
    return active_names, data_dirs


def phase_compute_domain_centroids(model, tokenizer, active_names, data_dirs):
    """Phase 2: Compute mean hidden-state centroids per domain (offline)."""
    log("\n[Phase 2] Computing domain centroids...")
    zero_lora_params(model)
    mx.eval(model.parameters())

    centroids = {}
    for name in active_names:
        fpath = data_dirs[name] / "train.jsonl"
        if not fpath.exists():
            continue
        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        hiddens = []
        for text in texts[:20]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            h = model.model(x)
            h_mean = mx.mean(h, axis=1)  # [1, d]
            mx.eval(h_mean)
            hiddens.append(h_mean)
            del h, x

        if hiddens:
            stacked = mx.concatenate(hiddens, axis=0)  # [n, d]
            centroid = mx.mean(stacked, axis=0)  # [d]
            mx.eval(centroid)
            centroids[name] = centroid
            del stacked, hiddens
        mx.clear_cache()

    log(f"  Computed centroids for {len(centroids)} domains")
    return centroids


def phase_extract_a_matrices(active_names):
    """Phase 3: Extract A-matrices for Arrow-style scoring."""
    log("\n[Phase 3] Extracting A-matrices...")
    a_matrices = extract_a_matrices(ADAPTERS_DIR, active_names)
    log(f"  Extracted A-matrices for {len(a_matrices)} adapters")
    return a_matrices


def phase_evaluate_strategies(model, tokenizer, active_names, data_dirs,
                               domain_centroids, a_matrices):
    """Phase 4: Evaluate all TTT strategies on validation data."""
    log("\n[Phase 4] Evaluating TTT selection strategies...")

    # Load reference data: learned router selections and PPLs
    n50_results = json.load(open(N50_DIR / "results.json"))
    router_selections = n50_results.get("routed_composition", {}).get("router_selections", {})
    routed_ppls = n50_results.get("routed_composition", {}).get("routed_ppls", {})
    individual_ppls = n50_results.get("individual_ppls", {})
    base_ppls = n50_results.get("base_ppls", {})

    strategies = {
        "exhaustive_probe": {},
        "arrow_projection": {},
        "cosine_centroid": {},
        "hybrid_arrow_m5": {},
        "hybrid_arrow_m3": {},
        "hybrid_cosine_m5": {},
        "hybrid_cosine_m3": {},
    }

    # Per-domain evaluation
    n_domains_done = 0
    total_domains = len(active_names)

    for name in active_names:
        fpath = data_dirs[name] / "valid.jsonl"
        if not fpath.exists():
            continue

        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        if not texts:
            continue

        # Use first validation text as probe input
        probe_text = texts[0]
        probe_tokens = tokenizer.encode(probe_text)
        if len(probe_tokens) < 4:
            continue

        probe_tokens = probe_tokens[:MAX_SEQ_LENGTH + 1]

        # Strategy 1: Exhaustive loss-probe (oracle)
        selected, n_passes, elapsed = exhaustive_loss_probe(
            model, tokenizer, probe_tokens, active_names, ADAPTERS_DIR,
            prefix_len=PREFIX_LEN, top_k=TOP_K
        )
        strategies["exhaustive_probe"][name] = {
            "selected": selected,
            "n_passes": n_passes,
            "time_s": round(elapsed, 3),
            "correct": name in selected,
        }

        # Strategy 2: Arrow projection
        selected_arrow, n_passes_a, elapsed_a = arrow_projection_scoring(
            model, tokenizer, probe_tokens, active_names,
            ADAPTERS_DIR, a_matrices, prefix_len=PREFIX_LEN, top_k=TOP_K
        )
        strategies["arrow_projection"][name] = {
            "selected": selected_arrow,
            "n_passes": n_passes_a,
            "time_s": round(elapsed_a, 3),
            "correct": name in selected_arrow,
        }

        # Strategy 2b: Cosine centroid
        selected_cos, n_passes_c, elapsed_c = hidden_state_cosine_scoring(
            model, tokenizer, probe_tokens, active_names,
            domain_centroids, prefix_len=PREFIX_LEN, top_k=TOP_K
        )
        strategies["cosine_centroid"][name] = {
            "selected": selected_cos,
            "n_passes": n_passes_c,
            "time_s": round(elapsed_c, 3),
            "correct": name in selected_cos,
        }

        # Strategy 3a: Hybrid Arrow m=5
        selected_h5, n_passes_h5, elapsed_h5 = hybrid_arrow_probe(
            model, tokenizer, probe_tokens, active_names, ADAPTERS_DIR,
            a_matrices, prefix_len=PREFIX_LEN, top_k=TOP_K, arrow_top_m=5
        )
        strategies["hybrid_arrow_m5"][name] = {
            "selected": selected_h5,
            "n_passes": n_passes_h5,
            "time_s": round(elapsed_h5, 3),
            "correct": name in selected_h5,
        }

        # Strategy 3b: Hybrid Arrow m=3
        selected_h3, n_passes_h3, elapsed_h3 = hybrid_arrow_probe(
            model, tokenizer, probe_tokens, active_names, ADAPTERS_DIR,
            a_matrices, prefix_len=PREFIX_LEN, top_k=TOP_K, arrow_top_m=3
        )
        strategies["hybrid_arrow_m3"][name] = {
            "selected": selected_h3,
            "n_passes": n_passes_h3,
            "time_s": round(elapsed_h3, 3),
            "correct": name in selected_h3,
        }

        # Strategy 3c: Hybrid Cosine m=5
        selected_hc5, n_passes_hc5, elapsed_hc5 = hybrid_cosine_probe(
            model, tokenizer, probe_tokens, active_names, ADAPTERS_DIR,
            domain_centroids, prefix_len=PREFIX_LEN, top_k=TOP_K, cosine_top_m=5
        )
        strategies["hybrid_cosine_m5"][name] = {
            "selected": selected_hc5,
            "n_passes": n_passes_hc5,
            "time_s": round(elapsed_hc5, 3),
            "correct": name in selected_hc5,
        }

        # Strategy 3d: Hybrid Cosine m=3
        selected_hc3, n_passes_hc3, elapsed_hc3 = hybrid_cosine_probe(
            model, tokenizer, probe_tokens, active_names, ADAPTERS_DIR,
            domain_centroids, prefix_len=PREFIX_LEN, top_k=TOP_K, cosine_top_m=3
        )
        strategies["hybrid_cosine_m3"][name] = {
            "selected": selected_hc3,
            "n_passes": n_passes_hc3,
            "time_s": round(elapsed_hc3, 3),
            "correct": name in selected_hc3,
        }

        n_domains_done += 1
        if n_domains_done % 10 == 0:
            log(f"  Progress: {n_domains_done}/{total_domains} domains")
            mx.clear_cache()

    log(f"  Completed all {n_domains_done} domains")
    return strategies


def phase_compute_ppl_for_selections(model, tokenizer, active_names, data_dirs,
                                      strategies):
    """Phase 5: Compute PPL for each strategy's selections."""
    log("\n[Phase 5] Computing PPL for TTT-selected compositions...")

    strategy_ppls = {}

    for strat_name, strat_data in strategies.items():
        log(f"\n  Strategy: {strat_name}")
        ppls = {}
        for domain_name in active_names:
            if domain_name not in strat_data:
                continue
            selected = strat_data[domain_name]["selected"]

            # Load and compose selected adapters
            adapter_list = []
            for sel_name in selected:
                adapter_path = ADAPTERS_DIR / sel_name / "adapter.npz"
                if adapter_path.exists():
                    adapter_list.append(dict(mx.load(str(adapter_path))))

            if not adapter_list:
                ppls[domain_name] = float("inf")
                continue

            # Compose top-k with 1/k scaling
            merged = {}
            scale = 1.0 / len(adapter_list)
            for key in adapter_list[0].keys():
                stacked = mx.stack([a[key] for a in adapter_list])
                merged[key] = mx.sum(stacked, axis=0) * scale
            del adapter_list

            # Apply and evaluate
            zero_lora_params(model)
            apply_adapter_weights(model, merged)
            mx.eval(model.parameters())

            ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
            ppls[domain_name] = round(ppl, 4)
            del merged
            mx.clear_cache()

        strategy_ppls[strat_name] = ppls
        ppl_vals = [v for v in ppls.values() if v < float("inf")]
        avg_ppl = sum(ppl_vals) / len(ppl_vals) if ppl_vals else float("inf")
        log(f"    Avg PPL: {avg_ppl:.2f} (over {len(ppl_vals)} domains)")

    return strategy_ppls


def phase_timing_benchmark(model, tokenizer, active_names, data_dirs,
                            domain_centroids, a_matrices):
    """Phase 6: Time the selection overhead for K1 assessment."""
    log("\n[Phase 6] Timing benchmark for K1...")

    # Use a representative domain
    test_domain = "code"
    if test_domain not in data_dirs:
        test_domain = active_names[0]

    fpath = data_dirs[test_domain] / "valid.jsonl"
    texts = []
    with open(fpath) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    test_tokens = tokenizer.encode(texts[0])[:MAX_SEQ_LENGTH + 1]

    # Time base generation (512 tokens simulated as forward passes)
    log("  Timing base model forward passes...")
    zero_lora_params(model)
    mx.eval(model.parameters())

    # Time a single forward pass on prefix
    t0 = time.time()
    for _ in range(10):
        x = mx.array(test_tokens[:PREFIX_LEN])[None, :]
        logits = model(x)
        mx.eval(logits)
        del logits, x
    single_pass_time = (time.time() - t0) / 10

    # Estimated base generation time for 512 tokens
    # Each token needs ~1 forward pass (autoregressive)
    base_gen_time_512 = single_pass_time * 512

    timing = {
        "single_pass_time_s": round(single_pass_time, 4),
        "estimated_base_gen_512_s": round(base_gen_time_512, 4),
    }

    # Time each strategy
    for strat_name, func_and_args in [
        ("arrow_projection", lambda: arrow_projection_scoring(
            model, tokenizer, test_tokens, active_names,
            ADAPTERS_DIR, a_matrices, prefix_len=PREFIX_LEN, top_k=TOP_K
        )),
        ("cosine_centroid", lambda: hidden_state_cosine_scoring(
            model, tokenizer, test_tokens, active_names,
            domain_centroids, prefix_len=PREFIX_LEN, top_k=TOP_K
        )),
        ("hybrid_arrow_m3", lambda: hybrid_arrow_probe(
            model, tokenizer, test_tokens, active_names, ADAPTERS_DIR,
            a_matrices, prefix_len=PREFIX_LEN, top_k=TOP_K, arrow_top_m=3
        )),
        ("hybrid_cosine_m3", lambda: hybrid_cosine_probe(
            model, tokenizer, test_tokens, active_names, ADAPTERS_DIR,
            domain_centroids, prefix_len=PREFIX_LEN, top_k=TOP_K, cosine_top_m=3
        )),
    ]:
        _, n_passes, elapsed = func_and_args()
        overhead_ratio = elapsed / base_gen_time_512 if base_gen_time_512 > 0 else float("inf")
        timing[strat_name] = {
            "time_s": round(elapsed, 4),
            "n_passes": n_passes,
            "overhead_ratio": round(overhead_ratio, 4),
            "k1_pass": overhead_ratio <= 0.5,
        }
        log(f"  {strat_name}: {elapsed:.3f}s ({n_passes} passes, overhead={overhead_ratio:.1%})")

    return timing


def main():
    t0 = time.time()
    log("=" * 70)
    log("TTT Expert Selection: Test-Time Training for Runtime Expert Selection")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load model
    model, tokenizer = phase_load_model()
    log_memory("after-model-load")

    # Phase 1: Collect data directories
    active_names, data_dirs = phase_collect_data_dirs()

    # Phase 2: Compute domain centroids (offline, one-time cost)
    domain_centroids = phase_compute_domain_centroids(model, tokenizer, active_names, data_dirs)
    log_memory("after-centroids")

    # Phase 3: Extract A-matrices
    a_matrices = phase_extract_a_matrices(active_names)
    log_memory("after-a-matrices")

    # Phase 4: Evaluate selection strategies
    strategies = phase_evaluate_strategies(
        model, tokenizer, active_names, data_dirs, domain_centroids, a_matrices
    )
    log_memory("after-strategies")

    # Phase 5: Compute PPL for each strategy's selections
    strategy_ppls = phase_compute_ppl_for_selections(
        model, tokenizer, active_names, data_dirs, strategies
    )
    log_memory("after-ppl")

    # Phase 6: Timing benchmark
    timing = phase_timing_benchmark(
        model, tokenizer, active_names, data_dirs, domain_centroids, a_matrices
    )
    log_memory("after-timing")

    # Compile results
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    # Load reference data
    n50_results = json.load(open(N50_DIR / "results.json"))
    routed_ppls = n50_results.get("routed_composition", {}).get("routed_ppls", {})

    results = {
        "experiment": "ttt_expert_selection",
        "model": MODEL_ID,
        "n_adapters": len(active_names),
        "top_k": TOP_K,
        "prefix_len": PREFIX_LEN,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_router_avg_ppl": REFERENCE_ROUTED_PPL,
        "strategies": {},
        "timing": timing,
        "kill_criteria": {},
    }

    for strat_name in strategies:
        strat_data = strategies[strat_name]
        ppls = strategy_ppls.get(strat_name, {})
        ppl_vals = [v for v in ppls.values() if v < float("inf")]
        avg_ppl = sum(ppl_vals) / len(ppl_vals) if ppl_vals else float("inf")

        # Selection accuracy (does correct adapter appear in top-2?)
        correct = sum(1 for d in strat_data.values() if d.get("correct", False))
        total = len(strat_data)
        accuracy = correct / total if total > 0 else 0

        # Average forward passes
        avg_passes = sum(d["n_passes"] for d in strat_data.values()) / len(strat_data) if strat_data else 0
        avg_time = sum(d["time_s"] for d in strat_data.values()) / len(strat_data) if strat_data else 0

        results["strategies"][strat_name] = {
            "avg_ppl": round(avg_ppl, 2),
            "selection_accuracy": round(accuracy, 4),
            "avg_forward_passes": round(avg_passes, 1),
            "avg_selection_time_s": round(avg_time, 3),
            "per_domain_ppls": ppls,
            "per_domain_selections": {
                name: strat_data[name]["selected"]
                for name in strat_data
            },
        }
        log(f"\n  {strat_name}:")
        log(f"    Avg PPL: {avg_ppl:.2f} (reference: {REFERENCE_ROUTED_PPL:.2f})")
        log(f"    Selection accuracy: {accuracy:.1%}")
        log(f"    Avg forward passes: {avg_passes:.1f}")
        log(f"    Avg selection time: {avg_time:.3f}s")

    # Kill criteria assessment
    # K1: overhead > 50%
    # Check all strategies that satisfy K3
    k3_valid_strategies = []
    for strat_name, strat_info in results["strategies"].items():
        if strat_info["avg_forward_passes"] <= 10:
            k3_valid_strategies.append(strat_name)

    best_valid_ppl = float("inf")
    best_valid_strat = None
    for strat_name in k3_valid_strategies:
        ppl = results["strategies"][strat_name]["avg_ppl"]
        if ppl < best_valid_ppl:
            best_valid_ppl = ppl
            best_valid_strat = strat_name

    # K1
    if best_valid_strat and best_valid_strat in timing:
        k1_overhead = timing[best_valid_strat].get("overhead_ratio", float("inf"))
    else:
        # Use cosine_centroid as default (always 1 pass)
        k1_overhead = timing.get("cosine_centroid", {}).get("overhead_ratio", float("inf"))

    k1_pass = k1_overhead <= 0.5
    results["kill_criteria"]["K1"] = {
        "threshold": "overhead <= 50%",
        "measured": round(k1_overhead, 4) if k1_overhead != float("inf") else "inf",
        "strategy": best_valid_strat or "none",
        "result": "PASS" if k1_pass else "FAIL",
    }

    # K2: PPL worse than reference
    k2_pass = best_valid_ppl <= REFERENCE_ROUTED_PPL if best_valid_ppl < float("inf") else False
    results["kill_criteria"]["K2"] = {
        "threshold": f"avg PPL <= {REFERENCE_ROUTED_PPL}",
        "measured": round(best_valid_ppl, 2) if best_valid_ppl < float("inf") else "inf",
        "strategy": best_valid_strat or "none",
        "result": "PASS" if k2_pass else "FAIL",
    }

    # K3: > 10 forward passes
    if best_valid_strat:
        k3_passes = results["strategies"][best_valid_strat]["avg_forward_passes"]
    else:
        k3_passes = float("inf")
    k3_pass = k3_passes <= 10
    results["kill_criteria"]["K3"] = {
        "threshold": "<= 10 forward passes",
        "measured": round(k3_passes, 1) if k3_passes < float("inf") else "inf",
        "strategy": best_valid_strat or "none",
        "result": "PASS" if k3_pass else "FAIL",
    }

    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {v['result']} (measured={v['measured']}, threshold={v['threshold']}, strategy={v['strategy']})")

    overall = "SUPPORTED" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    results["verdict"] = overall
    log(f"\n  VERDICT: {overall}")

    results["total_time_s"] = round(time.time() - t0, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")

    cleanup(model, tokenizer, domain_centroids, a_matrices)


if __name__ == "__main__":
    main()
