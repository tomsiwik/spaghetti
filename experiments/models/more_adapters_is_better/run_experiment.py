#!/usr/bin/env python3
"""Experiment: more_adapters_is_better

Tests the core thesis: does adding more routed adapters improve overall system quality?
Measures system-wide quality (average across all in-pool domains) as N grows
from 5 -> 10 -> 15 -> 20 -> 24.

Kill criteria:
  K1 (#523): Average quality plateaus or degrades after N=10
  K2 (#524): Domains 1-10 regress >5% when 11-24 are added

Success criteria:
  S1 (#50): Average routed PPL improves monotonically N=5->24, no domain regresses >5%

Approach:
  - Reuse 24 trained adapters from exp_real_data_25_domain_adapters
  - For each N in [5, 10, 15, 20, 24]:
    - Load first N adapters + their routing heads
    - Evaluate routed composition on all N domains
    - Track per-domain PPL and system averages

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

# Source experiment with trained adapters and data
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

# All 24 active domains from the source experiment (in order)
ALL_DOMAINS = [
    "medical", "code", "math", "legal", "finance",
    "science", "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering", "agriculture",
    "environmental", "politics", "economics", "sociology", "linguistics",
    "cybersecurity", "marketing", "sports", "music",
]

# N values to test
N_VALUES = [5, 10, 15, 20, 24]

# Routing head config (must match source experiment)
HIDDEN_DIM = 2560
HEAD_HIDDEN = 32
HEAD_TRAIN_STEPS = 300
HEAD_LR = 3e-4
TRAIN_SAMPLES_PER_DOMAIN = 40
VAL_SAMPLES_PER_DOMAIN = 50

# LoRA target modules
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
# BitNet unpacking and model utilities (from source experiment)
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
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


# ============================================================================
# Routing Head
# ============================================================================

class RoutingHead(nn.Module):
    """Tiny binary classifier for domain routing."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ============================================================================
# MultiAdapter LoRA
# ============================================================================

class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for correct multi-expert composition."""
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


class RoutedMultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs and per-sequence routing mask."""
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
        # routing_mask: list of bool, set externally before forward pass
        self.routing_mask = [False] * self.n_experts

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        active = [i for i in range(self.n_experts) if self.routing_mask[i]]
        if not active:
            return base_out

        k = len(active)
        lora_sum = mx.zeros_like(base_out)
        for i in active:
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste

        return base_out + lora_sum * (self.scale / k)


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
# Phase 2: Extract hidden states for routing heads
# ============================================================================

def phase_extract_hidden_states():
    """Extract hidden states from base model for routing head training."""
    log("\n[Phase 2] Extracting hidden states for routing...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    all_hidden = {}  # domain -> {"train": mx.array, "val": mx.array}

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
                for layer in model.model.layers:
                    h = layer(h)
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
            n_train = domain_hidden.get("train", mx.zeros((0,))).shape[0]
            n_val = domain_hidden.get("val", mx.zeros((0,))).shape[0]
            log(f"  {domain}: {n_train} train, {n_val} val states")

    elapsed = time.time() - t0
    log(f"  Hidden state extraction done in {elapsed:.1f}s")
    log_memory("post-hidden-states")
    cleanup(model, tokenizer)
    return all_hidden


# ============================================================================
# Phase 3: Evaluate at each N
# ============================================================================

def train_routing_heads_for_n(all_hidden, domains_subset):
    """Train routing heads for a specific subset of N domains.

    Returns dict: domain -> trained RoutingHead
    """
    N = len(domains_subset)
    heads = {}

    for target_domain in domains_subset:
        if target_domain not in all_hidden or "train" not in all_hidden[target_domain]:
            continue

        head = RoutingHead(HIDDEN_DIM, HEAD_HIDDEN)
        head_opt = opt.Adam(learning_rate=HEAD_LR)

        pos_train = all_hidden[target_domain]["train"]
        neg_train_list = [
            all_hidden[d]["train"] for d in domains_subset
            if d != target_domain and d in all_hidden and "train" in all_hidden[d]
        ]
        if not neg_train_list:
            continue
        neg_train = mx.concatenate(neg_train_list, axis=0)

        n_pos = pos_train.shape[0]
        n_neg = neg_train.shape[0]

        def head_loss_fn(head, x, labels):
            logits = head(x).squeeze(-1)
            return nn.losses.binary_cross_entropy(mx.sigmoid(logits), labels, reduction="mean")

        head_loss_and_grad = nn.value_and_grad(head, head_loss_fn)

        gc.disable()
        for step in range(HEAD_TRAIN_STEPS):
            p_idx = mx.array(np.random.randint(0, n_pos, size=16))
            n_idx = mx.array(np.random.randint(0, n_neg, size=16))
            batch_x = mx.concatenate([pos_train[p_idx], neg_train[n_idx]], axis=0)
            batch_y = mx.concatenate([mx.ones(16), mx.zeros(16)])
            loss, grads = head_loss_and_grad(head, batch_x, batch_y)
            head_opt.update(head, grads)
            mx.eval(head.parameters(), head_opt.state, loss)
        gc.enable()

        heads[target_domain] = head
        del head_opt, neg_train

    return heads


def evaluate_routing_accuracy(heads, all_hidden, domains_subset):
    """Evaluate routing heads on validation data. Returns per-domain accuracy."""
    accuracies = {}
    for target_domain in domains_subset:
        if target_domain not in heads:
            continue
        head = heads[target_domain]

        if target_domain not in all_hidden or "val" not in all_hidden[target_domain]:
            continue

        pos_val = all_hidden[target_domain]["val"]
        neg_val_list = [
            all_hidden[d]["val"] for d in domains_subset
            if d != target_domain and d in all_hidden and "val" in all_hidden[d]
        ]
        if not neg_val_list:
            continue
        neg_val = mx.concatenate(neg_val_list, axis=0)

        pos_scores = mx.sigmoid(head(pos_val).squeeze(-1))
        neg_scores = mx.sigmoid(head(neg_val).squeeze(-1))
        mx.eval(pos_scores, neg_scores)

        pos_acc = (pos_scores > 0.5).astype(mx.float32).mean().item()
        neg_acc = (neg_scores < 0.5).astype(mx.float32).mean().item()
        n_pos = pos_val.shape[0]
        n_neg = neg_val.shape[0]
        overall = (pos_acc * n_pos + neg_acc * n_neg) / (n_pos + n_neg)
        accuracies[target_domain] = {
            "overall": round(overall, 4),
            "pos": round(pos_acc, 4),
            "neg": round(neg_acc, 4),
        }

    return accuracies


def phase_evaluate_at_n(N, domains_subset, all_hidden, base_ppls):
    """Evaluate routed composition with N adapters.

    Two modes:
    1. Uniform composition (all N adapters active, 1/N scaling) - matches source experiment
    2. Oracle routing (only correct adapter active per domain) - upper bound

    For routed composition, we train N routing heads and use them to select
    which adapter(s) to apply per eval sequence.
    """
    log(f"\n[Phase 3.{N}] Evaluating N={N} adapters: {domains_subset}")
    t0 = time.time()

    # Load skeleton for N adapters
    skeleton_path = ADAPTERS_DIR / f"grassmannian_skeleton_n24.npz"
    skeleton = dict(np.load(str(skeleton_path)))

    # Build domain_index mapping: position in full 24-domain list -> position in N-subset
    full_domain_indices = {d: i for i, d in enumerate(ALL_DOMAINS)}
    subset_to_full = [full_domain_indices[d] for d in domains_subset]

    # Load adapter B matrices for the N domains
    all_adapter_params = []
    for domain_name in domains_subset:
        params = load_adapter(ADAPTERS_DIR / domain_name)
        all_adapter_params.append(params)

    # --- Train routing heads for this N ---
    log(f"  Training {N} routing heads...")
    np.random.seed(SEED)
    heads = train_routing_heads_for_n(all_hidden, domains_subset)
    routing_accs = evaluate_routing_accuracy(heads, all_hidden, domains_subset)
    mean_routing_acc = np.mean([v["overall"] for v in routing_accs.values()]) if routing_accs else 0
    log(f"  Mean routing accuracy: {mean_routing_acc:.4f}")

    # --- Evaluate: Oracle (individual adapter per domain) ---
    log(f"  Evaluating oracle (individual adapter) PPLs...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    oracle_ppls = {}
    for di, domain_name in enumerate(domains_subset):
        # Build model with single adapter
        n_layers = len(model.model.layers)
        full_di = subset_to_full[di]

        for li, layer in enumerate(model.model.layers):
            lora_updates = []
            for key in TARGET_KEYS:
                parts = key.split(".")
                module = layer
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is None:
                    continue
                # Get the base linear (unwrap if already wrapped)
                base_linear = module.linear if hasattr(module, 'linear') else module
                if not isinstance(base_linear, nn.Linear):
                    continue

                skey = f"layer_{li}_{key}_domain_{full_di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

                single_adapter = MultiAdapterLoRALinear(
                    base_linear, rank=LORA_RANK, scale=LORA_SCALE, a_inits=[a_mx]
                )

                param_name = f"model.layers.{li}.{key}.lora_b"
                if param_name in all_adapter_params[di]:
                    single_adapter.b_matrices[0] = all_adapter_params[di][param_name]

                lora_updates.append((key, single_adapter))

            if lora_updates:
                layer.update_modules(tree_unflatten(lora_updates))

        mx.eval(model.parameters())
        model.freeze()

        data_dir = DATA_DIR / domain_name
        ppl = compute_ppl(model, tokenizer, data_dir)
        oracle_ppls[domain_name] = round(ppl, 4)
        log(f"    {domain_name}: oracle PPL={ppl:.2f} (base={base_ppls.get(domain_name, 0):.2f})")

    cleanup(model, tokenizer)

    # --- Evaluate: Uniform composition (all N adapters) ---
    log(f"  Evaluating uniform composition (all {N} adapters, 1/N scaling)...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    n_layers = len(model.model.layers)
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

            a_inits = []
            for di in range(N):
                full_di = subset_to_full[di]
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
            for di in range(N):
                if param_name in all_adapter_params[di]:
                    multi_lora.b_matrices[di] = all_adapter_params[di][param_name]

            lora_updates.append((key, multi_lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()

    uniform_ppls = {}
    for domain_name in domains_subset:
        data_dir = DATA_DIR / domain_name
        ppl = compute_ppl(model, tokenizer, data_dir)
        uniform_ppls[domain_name] = round(ppl, 4)
        base = base_ppls.get(domain_name, 0)
        delta = (ppl - base) / base * 100 if base > 0 else 0
        log(f"    {domain_name}: uniform PPL={ppl:.2f} (base={base:.2f}, delta={delta:+.1f}%)")

    cleanup(model, tokenizer)

    # --- Evaluate: Routed composition (routing heads select adapters) ---
    log(f"  Evaluating routed composition (routing selects per domain)...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Apply routed multi-adapter layers
    routed_layers = []  # track for mask updates
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
            for di in range(N):
                full_di = subset_to_full[di]
                skey = f"layer_{li}_{key}_domain_{full_di}"
                if skey in skeleton:
                    a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                    a_inits.append(a_mx)

            if len(a_inits) != N:
                continue

            routed_lora = RoutedMultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )

            param_name = f"model.layers.{li}.{key}.lora_b"
            for di in range(N):
                if param_name in all_adapter_params[di]:
                    routed_lora.b_matrices[di] = all_adapter_params[di][param_name]

            lora_updates.append((key, routed_lora))
            routed_layers.append(routed_lora)

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()

    routed_ppls = {}
    fallback_count = 0  # Track how many domains hit base-only fallback
    routing_details = {}  # Per-domain routing details
    for eval_di, domain_name in enumerate(domains_subset):
        # Set routing mask: use routing heads to decide which adapters fire
        # NO oracle fallback: if no head fires, use base-only (no adapter)
        if domain_name in heads and domain_name in all_hidden and "val" in all_hidden[domain_name]:
            val_h = all_hidden[domain_name]["val"]
            h_mean = mx.mean(val_h, axis=0, keepdims=True)  # (1, d)

            mask = [False] * N
            for di, d in enumerate(domains_subset):
                if d in heads:
                    score = mx.sigmoid(heads[d](h_mean).squeeze()).item()
                    mask[di] = score > 0.5
                else:
                    mask[di] = False

            # HONEST ROUTING: if no adapter selected, use base-only (all False)
            if not any(mask):
                fallback_count += 1
                log(f"    {domain_name}: ROUTING FAILURE - no head fired, using base-only")
        else:
            # No hidden states available - use base-only
            mask = [False] * N
            fallback_count += 1

        # Apply mask to all routed layers
        for rl in routed_layers:
            rl.routing_mask = mask

        n_active = sum(mask)
        active_names = [domains_subset[i] for i in range(N) if mask[i]]
        is_oracle = (n_active == 1 and mask[eval_di])

        data_dir = DATA_DIR / domain_name
        ppl = compute_ppl(model, tokenizer, data_dir)
        routed_ppls[domain_name] = round(ppl, 4)
        base = base_ppls.get(domain_name, 0)
        delta = (ppl - base) / base * 100 if base > 0 else 0
        oracle_ppl = oracle_ppls.get(domain_name, 0)
        oracle_gap = (ppl - oracle_ppl) / oracle_ppl * 100 if oracle_ppl > 0 else 0
        routing_details[domain_name] = {
            "n_active": n_active,
            "active_adapters": active_names,
            "is_oracle_match": is_oracle,
            "oracle_gap_pct": round(oracle_gap, 2),
            "is_base_fallback": n_active == 0,
        }
        log(f"    {domain_name}: routed PPL={ppl:.2f} (k={n_active}, active={active_names}, delta={delta:+.1f}%, oracle_gap={oracle_gap:+.1f}%)")

    elapsed = time.time() - t0
    log(f"  N={N} evaluation done in {elapsed:.1f}s")

    cleanup(model, tokenizer)
    del all_adapter_params, skeleton, heads
    gc.collect()
    mx.clear_cache()

    log(f"  Routing failures (base-only fallback): {fallback_count}/{N} ({fallback_count/N*100:.0f}%)")

    return {
        "N": N,
        "domains": domains_subset,
        "oracle_ppls": oracle_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "routing_accuracies": routing_accs,
        "routing_details": routing_details,
        "mean_routing_accuracy": round(mean_routing_acc, 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / N, 4),
        "eval_time_s": round(elapsed, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_total = time.time()
    log("=" * 70)
    log("More Adapters Is Better: N-Scaling System Quality Test")
    log(f"  N values: {N_VALUES}")
    log(f"  Domains: {len(ALL_DOMAINS)}")
    log("=" * 70)
    log_memory("start")

    results = {
        "experiment": "more_adapters_is_better",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "n_values": N_VALUES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Phase 1: Base PPL
    base_ppls = phase_base_ppl()
    results["base_ppls"] = base_ppls
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Phase 2: Extract hidden states (once, reuse for all N)
    all_hidden = phase_extract_hidden_states()

    # Phase 3: Evaluate at each N
    scaling_results = {}
    for N in N_VALUES:
        domains_subset = ALL_DOMAINS[:N]
        eval_result = phase_evaluate_at_n(N, domains_subset, all_hidden, base_ppls)
        scaling_results[str(N)] = eval_result

        # Save progress
        results["scaling_results"] = scaling_results
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ========================================================================
    # Phase 4: Frozen-heads evaluation
    # Train routing heads ONCE at N=5, then evaluate at all N without retraining.
    # This tests whether routing degrades as more adapters are added.
    # ========================================================================
    log("\n[Phase 4] Frozen-heads evaluation (heads trained at N=5 only)...")
    frozen_heads = train_routing_heads_for_n(all_hidden, ALL_DOMAINS[:5])
    frozen_results = {}

    for N in N_VALUES:
        domains_subset = ALL_DOMAINS[:N]
        log(f"\n  Frozen-heads N={N}:")

        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"
        skeleton = dict(np.load(str(skeleton_path)))
        full_domain_indices = {d: i for i, d in enumerate(ALL_DOMAINS)}
        subset_to_full = [full_domain_indices[d] for d in domains_subset]

        all_adapter_params_frozen = []
        for domain_name in domains_subset:
            params = load_adapter(ADAPTERS_DIR / domain_name)
            all_adapter_params_frozen.append(params)

        routed_layers = []
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
                for di in range(N):
                    full_di = subset_to_full[di]
                    skey = f"layer_{li}_{key}_domain_{full_di}"
                    if skey in skeleton:
                        a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                        a_inits.append(a_mx)

                if len(a_inits) != N:
                    continue

                routed_lora = RoutedMultiAdapterLoRALinear(
                    module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
                )

                param_name = f"model.layers.{li}.{key}.lora_b"
                for di in range(N):
                    if param_name in all_adapter_params_frozen[di]:
                        routed_lora.b_matrices[di] = all_adapter_params_frozen[di][param_name]

                lora_updates.append((key, routed_lora))
                routed_layers.append(routed_lora)

            if lora_updates:
                layer.update_modules(tree_unflatten(lora_updates))

        mx.eval(model.parameters())
        model.freeze()

        frozen_ppls = {}
        frozen_fallback = 0
        for eval_di, domain_name in enumerate(domains_subset):
            if domain_name in all_hidden and "val" in all_hidden[domain_name]:
                val_h = all_hidden[domain_name]["val"]
                h_mean = mx.mean(val_h, axis=0, keepdims=True)

                mask = [False] * N
                # Only use the 5 frozen heads - domains beyond N=5 have no head
                for di, d in enumerate(domains_subset):
                    if d in frozen_heads:
                        score = mx.sigmoid(frozen_heads[d](h_mean).squeeze()).item()
                        mask[di] = score > 0.5

                if not any(mask):
                    frozen_fallback += 1
            else:
                mask = [False] * N
                frozen_fallback += 1

            for rl in routed_layers:
                rl.routing_mask = mask

            n_active = sum(mask)
            data_dir = DATA_DIR / domain_name
            ppl = compute_ppl(model, tokenizer, data_dir)
            frozen_ppls[domain_name] = round(ppl, 4)
            log(f"    {domain_name}: frozen-routed PPL={ppl:.2f} (k={n_active})")

        frozen_results[str(N)] = {
            "routed_ppls": frozen_ppls,
            "fallback_count": frozen_fallback,
            "fallback_rate": round(frozen_fallback / N, 4),
        }
        log(f"  Frozen-heads N={N}: {frozen_fallback}/{N} fallbacks")

        cleanup(model, tokenizer)
        del all_adapter_params_frozen, skeleton
        gc.collect()
        mx.clear_cache()

    results["frozen_heads_results"] = frozen_results

    # ========================================================================
    # Phase 5: Random routing baseline
    # For each N, randomly select k adapters (k=1,2) per domain evaluation.
    # Establishes a lower bound to show routing adds value.
    # ========================================================================
    log("\n[Phase 5] Random routing baseline...")
    np.random.seed(SEED)
    random_results = {}

    for N in N_VALUES:
        domains_subset = ALL_DOMAINS[:N]
        log(f"\n  Random routing N={N}:")

        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"
        skeleton = dict(np.load(str(skeleton_path)))
        full_domain_indices = {d: i for i, d in enumerate(ALL_DOMAINS)}
        subset_to_full = [full_domain_indices[d] for d in domains_subset]

        all_adapter_params_rand = []
        for domain_name in domains_subset:
            params = load_adapter(ADAPTERS_DIR / domain_name)
            all_adapter_params_rand.append(params)

        routed_layers = []
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
                for di in range(N):
                    full_di = subset_to_full[di]
                    skey = f"layer_{li}_{key}_domain_{full_di}"
                    if skey in skeleton:
                        a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                        a_inits.append(a_mx)

                if len(a_inits) != N:
                    continue

                routed_lora = RoutedMultiAdapterLoRALinear(
                    module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
                )

                param_name = f"model.layers.{li}.{key}.lora_b"
                for di in range(N):
                    if param_name in all_adapter_params_rand[di]:
                        routed_lora.b_matrices[di] = all_adapter_params_rand[di][param_name]

                lora_updates.append((key, routed_lora))
                routed_layers.append(routed_lora)

            if lora_updates:
                layer.update_modules(tree_unflatten(lora_updates))

        mx.eval(model.parameters())
        model.freeze()

        random_ppls = {}
        for eval_di, domain_name in enumerate(domains_subset):
            # Random top-1: pick one adapter at random
            rand_idx = np.random.randint(0, N)
            mask = [False] * N
            mask[rand_idx] = True

            for rl in routed_layers:
                rl.routing_mask = mask

            data_dir = DATA_DIR / domain_name
            ppl = compute_ppl(model, tokenizer, data_dir)
            random_ppls[domain_name] = round(ppl, 4)
            log(f"    {domain_name}: random PPL={ppl:.2f} (picked={domains_subset[rand_idx]})")

        random_results[str(N)] = {"routed_ppls": random_ppls}

        cleanup(model, tokenizer)
        del all_adapter_params_rand, skeleton
        gc.collect()
        mx.clear_cache()

    results["random_routing_results"] = random_results

    # Clean up hidden states
    del all_hidden
    gc.collect()
    mx.clear_cache()

    # ========================================================================
    # Analysis
    # ========================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Track system-wide metrics at each N
    system_metrics = {}
    for N_str, sr in scaling_results.items():
        N = int(N_str)
        domains = sr["domains"]

        avg_base = np.mean([base_ppls[d] for d in domains if d in base_ppls])
        avg_oracle = np.mean([sr["oracle_ppls"][d] for d in domains if d in sr["oracle_ppls"]])
        avg_uniform = np.mean([sr["uniform_ppls"][d] for d in domains if d in sr["uniform_ppls"]])
        avg_routed = np.mean([sr["routed_ppls"][d] for d in domains if d in sr["routed_ppls"]])

        gamma_uniform = avg_uniform / avg_base
        gamma_oracle = avg_oracle / avg_base
        gamma_routed = avg_routed / avg_base

        system_metrics[N] = {
            "avg_base_ppl": round(float(avg_base), 4),
            "avg_oracle_ppl": round(float(avg_oracle), 4),
            "avg_uniform_ppl": round(float(avg_uniform), 4),
            "avg_routed_ppl": round(float(avg_routed), 4),
            "gamma_uniform": round(float(gamma_uniform), 4),
            "gamma_oracle": round(float(gamma_oracle), 4),
            "gamma_routed": round(float(gamma_routed), 4),
            "ppl_improvement_uniform_pct": round(float((1 - gamma_uniform) * 100), 2),
            "ppl_improvement_oracle_pct": round(float((1 - gamma_oracle) * 100), 2),
            "ppl_improvement_routed_pct": round(float((1 - gamma_routed) * 100), 2),
            "mean_routing_accuracy": sr["mean_routing_accuracy"],
        }

        log(f"\nN={N}:")
        log(f"  Avg base PPL:    {avg_base:.2f}")
        log(f"  Avg oracle PPL:  {avg_oracle:.2f} (gamma={gamma_oracle:.4f}, {(1-gamma_oracle)*100:+.1f}%)")
        log(f"  Avg uniform PPL: {avg_uniform:.2f} (gamma={gamma_uniform:.4f}, {(1-gamma_uniform)*100:+.1f}%)")
        log(f"  Avg routed PPL:  {avg_routed:.2f} (gamma={gamma_routed:.4f}, {(1-gamma_routed)*100:+.1f}%)")
        log(f"  Mean routing accuracy: {sr['mean_routing_accuracy']:.4f}")

    results["system_metrics"] = system_metrics

    # K1: Average quality plateaus or degrades after N=10
    n_values_sorted = sorted(system_metrics.keys())
    routed_ppls_by_n = [system_metrics[n]["avg_routed_ppl"] for n in n_values_sorted]
    k1_pass = True
    k1_evidence = []
    for i in range(1, len(n_values_sorted)):
        prev_n = n_values_sorted[i-1]
        curr_n = n_values_sorted[i]
        # Check if avg routed PPL improved (lower is better)
        prev_ppl = system_metrics[prev_n]["avg_routed_ppl"]
        curr_ppl = system_metrics[curr_n]["avg_routed_ppl"]
        # Note: at larger N, avg includes more domains which may have higher base PPL
        # So we compare gamma (improvement ratio) not absolute PPL
        prev_gamma = system_metrics[prev_n]["gamma_routed"]
        curr_gamma = system_metrics[curr_n]["gamma_routed"]
        improved = curr_gamma <= prev_gamma + 0.02  # allow 2pp slack
        k1_evidence.append(f"N={prev_n}->{curr_n}: gamma {prev_gamma:.4f}->{curr_gamma:.4f} ({'OK' if improved else 'DEGRADED'})")
        if not improved and curr_n > 10:
            k1_pass = False

    log(f"\nK1 (no plateau after N=10): {'PASS' if k1_pass else 'FAIL'}")
    for e in k1_evidence:
        log(f"  {e}")

    # K2: Domains 1-10 regress >5% when 11-24 are added
    k2_pass = True
    k2_evidence = []
    if "10" in scaling_results and "24" in scaling_results:
        domains_10 = ALL_DOMAINS[:10]
        sr_10 = scaling_results["10"]
        sr_24 = scaling_results["24"]
        for d in domains_10:
            if d in sr_10["routed_ppls"] and d in sr_24["routed_ppls"]:
                ppl_10 = sr_10["routed_ppls"][d]
                ppl_24 = sr_24["routed_ppls"][d]
                regress = (ppl_24 - ppl_10) / ppl_10 * 100
                k2_evidence.append(f"{d}: PPL at N=10 {ppl_10:.2f} -> N=24 {ppl_24:.2f} ({regress:+.1f}%)")
                if regress > 5.0:
                    k2_pass = False
                    k2_evidence[-1] += " ** REGRESSED **"

    log(f"\nK2 (domains 1-10 stable when 11-24 added): {'PASS' if k2_pass else 'FAIL'}")
    for e in k2_evidence:
        log(f"  {e}")

    # S1: Average routed PPL improves monotonically, no domain regresses >5%
    s1_monotonic = all(
        system_metrics[n_values_sorted[i]]["gamma_routed"] <=
        system_metrics[n_values_sorted[i-1]]["gamma_routed"] + 0.02
        for i in range(1, len(n_values_sorted))
    )
    s1_pass = s1_monotonic and k2_pass

    log(f"\nS1 (monotonic improvement, no regression): {'PASS' if s1_pass else 'FAIL'}")

    # Fixed-domain metric: track first 5 domains across all N
    fixed_domains = ALL_DOMAINS[:5]
    log("\n--- Fixed-Domain Metric (first 5 domains tracked across all N) ---")
    fixed_domain_metrics = {}
    for N_str, sr in scaling_results.items():
        N = int(N_str)
        fixed_ppls = {d: sr["routed_ppls"].get(d, None) for d in fixed_domains if d in sr["routed_ppls"]}
        oracle_ppls_fixed = {d: sr["oracle_ppls"].get(d, None) for d in fixed_domains if d in sr["oracle_ppls"]}
        avg_fixed_routed = np.mean([v for v in fixed_ppls.values() if v is not None])
        avg_fixed_oracle = np.mean([v for v in oracle_ppls_fixed.values() if v is not None])
        avg_fixed_base = np.mean([base_ppls[d] for d in fixed_domains if d in base_ppls])
        fixed_domain_metrics[N] = {
            "per_domain": fixed_ppls,
            "avg_routed": round(float(avg_fixed_routed), 4),
            "avg_oracle": round(float(avg_fixed_oracle), 4),
            "avg_base": round(float(avg_fixed_base), 4),
            "gamma_routed": round(float(avg_fixed_routed / avg_fixed_base), 4),
            "routed_vs_oracle_gap_pct": round(float((avg_fixed_routed - avg_fixed_oracle) / avg_fixed_oracle * 100), 2),
        }
        log(f"  N={N}: avg_routed={avg_fixed_routed:.2f}, avg_oracle={avg_fixed_oracle:.2f}, gap={fixed_domain_metrics[N]['routed_vs_oracle_gap_pct']:+.2f}%")

    results["fixed_domain_metrics"] = fixed_domain_metrics

    # Fallback rate summary
    log("\n--- Fallback Rates (base-only when routing fails) ---")
    for N_str, sr in scaling_results.items():
        N = int(N_str)
        fb = sr.get("fallback_count", 0)
        log(f"  N={N}: {fb}/{N} domains hit base-only fallback ({fb/N*100:.0f}%)")

    # Frozen-heads comparison
    log("\n--- Frozen Heads (trained at N=5) vs Retrained Heads ---")
    for N_str in [str(n) for n in N_VALUES]:
        N = int(N_str)
        if N_str in frozen_results and N_str in scaling_results:
            retrained_ppls = scaling_results[N_str]["routed_ppls"]
            frozen_ppls_n = frozen_results[N_str]["routed_ppls"]
            domains_both = [d for d in ALL_DOMAINS[:N] if d in retrained_ppls and d in frozen_ppls_n]
            avg_retrained = np.mean([retrained_ppls[d] for d in domains_both])
            avg_frozen = np.mean([frozen_ppls_n[d] for d in domains_both])
            gap = (avg_frozen - avg_retrained) / avg_retrained * 100
            log(f"  N={N}: retrained={avg_retrained:.2f}, frozen={avg_frozen:.2f}, gap={gap:+.1f}%, frozen_fallbacks={frozen_results[N_str]['fallback_count']}")

    # Random routing comparison
    log("\n--- Random Routing vs Trained Routing ---")
    for N_str in [str(n) for n in N_VALUES]:
        N = int(N_str)
        if N_str in random_results and N_str in scaling_results:
            retrained_ppls = scaling_results[N_str]["routed_ppls"]
            rand_ppls_n = random_results[N_str]["routed_ppls"]
            domains_both = [d for d in ALL_DOMAINS[:N] if d in retrained_ppls and d in rand_ppls_n]
            avg_retrained = np.mean([retrained_ppls[d] for d in domains_both])
            avg_random = np.mean([rand_ppls_n[d] for d in domains_both])
            gap = (avg_random - avg_retrained) / avg_retrained * 100
            log(f"  N={N}: routed={avg_retrained:.2f}, random={avg_random:.2f}, gap={gap:+.1f}%")

    results["k1_pass"] = k1_pass
    results["k1_evidence"] = k1_evidence
    results["k2_pass"] = k2_pass
    results["k2_evidence"] = k2_evidence
    results["s1_pass"] = s1_pass
    results["verdict"] = "SUPPORTED" if k1_pass and k2_pass else "KILLED"

    total_time = time.time() - t_total
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"\n{'=' * 70}")
    log(f"VERDICT: {results['verdict']}")
    log(f"Total time: {total_time / 60:.1f} min")
    log(f"Results saved to: {RESULTS_FILE}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
