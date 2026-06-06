#!/usr/bin/env python3
"""
Centralized Multi-Class Routing at N=24.

Replaces 24 independent binary routing heads with a single multi-class softmax
head. Tests the hypothesis that softmax normalization eliminates the FPR cascade
and loudest-voice failure modes that killed binary heads at N=24 (39.6% accuracy).

Kill criteria:
  K587: Top-1 routing accuracy <60% at N=24 -> KILL
  K588: Routed composition PPL worse than uniform 1/N at N=24 -> KILL
  K589: Training + inference overhead >15% of base forward pass -> KILL

Platform: Apple M5 Pro 48GB, MLX.
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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
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
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 20
SEED = 42

# Multi-class routing head config
ROUTER_HIDDEN_DIM = 64
ROUTER_LR = 3e-4
ROUTER_TRAIN_STEPS = 2000  # more steps for multi-class (was 500 per binary head)
ROUTER_BATCH_SIZE = 16

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ROUTER_DIR = EXPERIMENT_DIR / "router"

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
# Model utilities (reused from binary heads experiment)
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
# Multi-Class Routing Head: single softmax classifier over K domains
# ===========================================================================
class MultiClassRouter(nn.Module):
    """Single multi-class routing head: h_pool -> K logits -> softmax.

    Architecture: d -> h -> K (2-layer MLP with ReLU).
    Key difference from binary heads: ONE network outputs all K scores,
    softmax normalization forces competition between classes.
    """

    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: (batch, d) mean-pooled hidden states
        Returns:
            (batch, K) raw logits (apply softmax externally for loss/probs)
        """
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

    # Collect all hidden states as (hidden_vector, domain_label) pairs
    train_data = []  # list of (mx.array of shape (1, d), int label)
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

    log(f"  Hidden state caching done in {time.time() - t0:.1f}s")
    log(f"  Train samples: {len(train_data)} ({HIDDEN_CACHE_TRAIN_PER_DOMAIN}/domain)")
    log(f"  Val samples: {len(val_data)} ({HIDDEN_CACHE_VAL_PER_DOMAIN}/domain)")
    log_memory("after-caching")

    cleanup(model, tokenizer)
    return train_data, val_data, d_model


# ===========================================================================
# Phase 2: Train single multi-class routing head
# ===========================================================================
def phase_train_router(train_data, val_data, d_model):
    """Train one multi-class softmax routing head."""
    log("\n" + "=" * 70)
    log(f"[Phase 2] Training multi-class router (K={N_DOMAINS}, h={ROUTER_HIDDEN_DIM})")
    log("=" * 70)

    t0 = time.time()
    router = MultiClassRouter(d_model, N_DOMAINS, ROUTER_HIDDEN_DIM)
    mx.eval(router.parameters())

    n_params = sum(p.size for _, p in tree_flatten(router.parameters()))
    log(f"  Router params: {n_params:,}")

    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    def router_loss_fn(router, h_batch, labels_batch):
        """Multi-class cross-entropy loss."""
        logits = router(h_batch)  # (B, K)
        return nn.losses.cross_entropy(logits, labels_batch, reduction="mean")

    router_loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    # Training loop with mini-batches
    gc.disable()
    losses = []
    for step in range(ROUTER_TRAIN_STEPS):
        # Sample a mini-batch
        batch_indices = rng.sample(range(len(train_data)), min(ROUTER_BATCH_SIZE, len(train_data)))
        h_list = [train_data[idx][0] for idx in batch_indices]
        label_list = [train_data[idx][1] for idx in batch_indices]

        h_batch = mx.concatenate(h_list, axis=0)  # (B, d)
        labels_batch = mx.array(label_list)  # (B,)

        loss, grads = router_loss_and_grad(router, h_batch, labels_batch)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            recent_loss = sum(losses[-100:]) / len(losses[-100:])
            log(f"  Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={recent_loss:.4f}")

    gc.enable()
    gc.collect()

    final_loss = sum(losses[-100:]) / len(losses[-100:])
    log(f"  Final loss (last 100): {final_loss:.4f}")

    # --- Evaluate on validation data ---
    log("  Evaluating on validation set...")
    correct_top1 = 0
    correct_top2 = 0
    total = 0
    per_domain_correct = {d: 0 for d in DOMAINS}
    per_domain_total = {d: 0 for d in DOMAINS}
    confusion_matrix = {}  # (true, pred) -> count

    for h_pool, label in val_data:
        logits = router(h_pool)  # (1, K)
        mx.eval(logits)

        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        probs_list = probs[0].tolist()

        sorted_indices = sorted(range(N_DOMAINS), key=lambda k: probs_list[k], reverse=True)
        pred_top1 = sorted_indices[0]
        pred_top2 = sorted_indices[:2]

        true_domain = DOMAINS[label]
        pred_domain = DOMAINS[pred_top1]

        if pred_top1 == label:
            correct_top1 += 1
            per_domain_correct[true_domain] += 1
        else:
            key = f"{true_domain} -> {pred_domain}"
            confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

        if label in pred_top2:
            correct_top2 += 1

        per_domain_total[true_domain] += 1
        total += 1

    top1_acc = correct_top1 / total if total > 0 else 0
    top2_acc = correct_top2 / total if total > 0 else 0

    log(f"  Val top-1 accuracy: {top1_acc:.1%} ({correct_top1}/{total})")
    log(f"  Val top-2 accuracy: {top2_acc:.1%} ({correct_top2}/{total})")

    # Per-domain breakdown
    per_domain_acc = {}
    log("  Per-domain validation accuracy:")
    for domain in DOMAINS:
        if per_domain_total[domain] > 0:
            acc = per_domain_correct[domain] / per_domain_total[domain]
        else:
            acc = 0
        per_domain_acc[domain] = round(acc, 4)
        log(f"    {domain:20s}: {acc:.1%} ({per_domain_correct[domain]}/{per_domain_total[domain]})")

    # Top confusion pairs
    log("  Top confusion pairs:")
    for pair, count in sorted(confusion_matrix.items(), key=lambda x: -x[1])[:15]:
        log(f"    {pair}: {count}")

    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")

    # Save router weights
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    router_params = dict(tree_flatten(router.parameters()))
    mx.savez(str(ROUTER_DIR / "router.npz"), **router_params)
    log(f"  Router saved to {ROUTER_DIR}")

    # Clean up optimizer
    del optimizer, router_loss_and_grad
    gc.collect()
    mx.clear_cache()

    router_info = {
        "n_params": n_params,
        "hidden_dim": ROUTER_HIDDEN_DIM,
        "train_steps": ROUTER_TRAIN_STEPS,
        "batch_size": ROUTER_BATCH_SIZE,
        "lr": ROUTER_LR,
        "final_loss": round(final_loss, 4),
        "train_time_s": round(train_time, 1),
        "val_top1_accuracy": round(top1_acc, 4),
        "val_top2_accuracy": round(top2_acc, 4),
        "per_domain_accuracy": per_domain_acc,
        "confusion_top15": dict(sorted(confusion_matrix.items(), key=lambda x: -x[1])[:15]),
    }

    log_memory("after-router-training")
    return router, router_info


# ===========================================================================
# Phase 3: Evaluate routing + composition PPL
# ===========================================================================
def phase_evaluate(model_id, router, router_info, d_model):
    """Evaluate routing accuracy on held-out data, composition PPL, and overhead."""
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating routing and composition at N=24")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load all adapters
    adapters = {}
    for domain in DOMAINS:
        path = ADAPTERS_SOURCE_DIR / domain
        adapters[domain] = load_adapter(path)
    log(f"  Loaded {len(adapters)} adapters")
    log_memory("after-adapter-load")

    # Load validation texts
    val_texts = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        val_texts[domain] = texts

    # --- Base PPL ---
    log("\n  Computing base PPL (no adapters)...")
    zero_adapter_in_model(model)
    base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        base_ppls[domain] = round(ppl, 4)
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    log(f"    Average base PPL: {avg_base:.4f}")

    # --- Individual adapter PPL (oracle) ---
    log("\n  Computing individual adapter PPL (oracle)...")
    individual_ppls = {}
    for domain in DOMAINS:
        apply_adapter_to_model(model, adapters[domain])
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        individual_ppls[domain] = round(ppl, 4)
        zero_adapter_in_model(model)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    log(f"    Average individual PPL: {avg_individual:.4f}")

    # --- Uniform 1/N composition ---
    log("\n  Computing uniform 1/N composition PPL...")
    composed_1n = {}
    for key in adapters[DOMAINS[0]].keys():
        stacked = mx.stack([adapters[d][key] for d in DOMAINS])
        composed_1n[key] = mx.mean(stacked, axis=0)
    mx.eval(composed_1n)

    apply_adapter_to_model(model, composed_1n)
    uniform_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        uniform_ppls[domain] = round(ppl, 4)
    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    log(f"    Average uniform PPL: {avg_uniform:.4f}")
    zero_adapter_in_model(model)
    del composed_1n

    # --- Router-guided top-2 composition PPL ---
    log("\n  Computing router-guided top-2 composition PPL...")
    routed_ppls = {}
    top1_correct = 0
    top2_correct = 0
    total_routing = 0
    per_domain_top1 = {d: {"correct": 0, "total": 0} for d in DOMAINS}
    confusion_matrix = {}

    for eval_domain in DOMAINS:
        domain_losses = 0.0
        domain_tokens = 0

        for text in val_texts[eval_domain][:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x_tokens = tokens[:-1]
            y_tokens = tokens[1:]

            x = mx.array(x_tokens)[None, :]

            # Get hidden states for routing
            h = get_hidden_states(model, x)
            h_pool = mx.mean(h, axis=1)  # (1, d)
            mx.eval(h_pool)
            del h

            # Single router forward pass -> K logits -> softmax
            logits = router(h_pool)  # (1, K)
            probs = mx.softmax(logits, axis=-1)
            mx.eval(probs)
            probs_list = probs[0].tolist()
            del logits

            # Top-2 selection
            sorted_indices = sorted(range(N_DOMAINS), key=lambda k: probs_list[k], reverse=True)
            top2_indices = sorted_indices[:2]
            top2_domains = [DOMAINS[i] for i in top2_indices]
            top2_probs = [probs_list[i] for i in top2_indices]

            # Track routing accuracy
            true_label = DOMAIN_TO_IDX[eval_domain]
            if sorted_indices[0] == true_label:
                top1_correct += 1
                per_domain_top1[eval_domain]["correct"] += 1
            else:
                pred_domain = DOMAINS[sorted_indices[0]]
                key = f"{eval_domain} -> {pred_domain}"
                confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

            if true_label in top2_indices:
                top2_correct += 1
            total_routing += 1
            per_domain_top1[eval_domain]["total"] += 1

            # Pre-merge top-2 with probability-weighted composition
            total_prob = sum(top2_probs)
            composed = {}
            if total_prob > 1e-8:
                for sel_domain, sel_prob in zip(top2_domains, top2_probs):
                    w = sel_prob / total_prob
                    for key, val in adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * w
                        else:
                            composed[key] = composed[key] + val * w
            else:
                for sel_domain in top2_domains:
                    for key, val in adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * 0.5
                        else:
                            composed[key] = composed[key] + val * 0.5

            apply_adapter_to_model(model, composed)
            y = mx.array(y_tokens)[None, :]
            model_logits = model(x)
            loss = nn.losses.cross_entropy(model_logits, y, reduction="sum")
            mx.eval(loss)

            domain_losses += loss.item()
            domain_tokens += y.size

            zero_adapter_in_model(model)
            del composed, model_logits, loss, x, y, h_pool, probs

        if domain_tokens > 0:
            avg_loss = domain_losses / domain_tokens
            ppl = math.exp(min(avg_loss, 100))
        else:
            ppl = float("inf")

        routed_ppls[eval_domain] = round(ppl, 4)

    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)
    top1_acc = top1_correct / total_routing if total_routing > 0 else 0
    top2_acc = top2_correct / total_routing if total_routing > 0 else 0
    log(f"    Average routed PPL: {avg_routed:.4f}")
    log(f"    Top-1 routing accuracy: {top1_acc:.1%} ({top1_correct}/{total_routing})")
    log(f"    Top-2 routing accuracy: {top2_acc:.1%} ({top2_correct}/{total_routing})")

    # Per-domain routing accuracy
    per_domain_top1_pct = {}
    for domain in DOMAINS:
        info = per_domain_top1[domain]
        if info["total"] > 0:
            per_domain_top1_pct[domain] = round(info["correct"] / info["total"], 4)
        else:
            per_domain_top1_pct[domain] = 0

    # --- Router inference overhead (K589) ---
    log("\n  Measuring router inference overhead...")
    test_text = val_texts[DOMAINS[0]][0]
    test_tokens = tokenizer.encode(test_text)[:MAX_SEQ_LENGTH]
    test_x = mx.array(test_tokens)[None, :]

    # Warm up
    for _ in range(5):
        out = model(test_x)
        mx.eval(out)
    del out

    n_timing = 30
    t_base_start = time.time()
    for _ in range(n_timing):
        out = model(test_x)
        mx.eval(out)
    t_base = (time.time() - t_base_start) / n_timing
    del out

    # Time the single router
    h = get_hidden_states(model, test_x)
    h_pool = mx.mean(h, axis=1)
    mx.eval(h_pool)

    # Warm up router
    for _ in range(5):
        out = router(h_pool)
        mx.eval(out)
    del out

    t_router_start = time.time()
    for _ in range(n_timing):
        out = router(h_pool)
        mx.eval(out)
    t_router = (time.time() - t_router_start) / n_timing

    overhead_pct = (t_router / t_base) * 100
    log(f"    Base forward: {t_base*1000:.2f}ms")
    log(f"    Router forward: {t_router*1000:.2f}ms")
    log(f"    Overhead: {overhead_pct:.2f}%")

    del test_x, h, h_pool
    log_memory("after-eval")
    cleanup(model, tokenizer)

    return {
        "base_ppls": base_ppls,
        "individual_ppls": individual_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "per_domain_top1": per_domain_top1_pct,
        "confusion_top15": dict(sorted(confusion_matrix.items(), key=lambda x: -x[1])[:15]),
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "total_routing_decisions": total_routing,
        "timing": {
            "base_forward_ms": round(t_base * 1000, 2),
            "router_forward_ms": round(t_router * 1000, 2),
            "overhead_pct": round(overhead_pct, 2),
        },
    }


# ===========================================================================
# Phase 4: Kill criteria assessment
# ===========================================================================
def phase_analysis(eval_results, router_info, d_model):
    """Assess kill criteria and produce final results."""
    log("\n" + "=" * 70)
    log("[Phase 4] Kill criteria assessment")
    log("=" * 70)

    base_ppls = eval_results["base_ppls"]
    individual_ppls = eval_results["individual_ppls"]
    uniform_ppls = eval_results["uniform_ppls"]
    routed_ppls = eval_results["routed_ppls"]
    timing = eval_results["timing"]

    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)

    # --- K587: Top-1 routing accuracy >= 60% ---
    top1_acc = eval_results["top1_accuracy"]
    k587_pass = top1_acc >= 0.60
    log(f"\n  K587: Top-1 routing accuracy >= 60%")
    log(f"    Measured: {top1_acc:.1%} [{'PASS' if k587_pass else 'FAIL'}]")

    # --- K588: Routed PPL better than uniform 1/N ---
    k588_pass = avg_routed < avg_uniform
    improvement = ((avg_uniform - avg_routed) / avg_uniform) * 100
    log(f"\n  K588: Routed PPL < uniform 1/N")
    log(f"    Uniform: {avg_uniform:.4f}, Routed: {avg_routed:.4f}")
    log(f"    Improvement: {improvement:+.2f}% [{'PASS' if k588_pass else 'FAIL'}]")

    # --- K589: Overhead < 15% ---
    k589_pass = timing["overhead_pct"] < 15.0
    log(f"\n  K589: Training + inference overhead < 15%")
    log(f"    Router overhead: {timing['overhead_pct']:.2f}% [{'PASS' if k589_pass else 'FAIL'}]")

    # Per-domain routing breakdown
    log("\n  Per-domain routing accuracy:")
    for domain in DOMAINS:
        acc = eval_results["per_domain_top1"].get(domain, 0)
        log(f"    {domain:20s}: top-1={acc:.1%} "
            f"(base={base_ppls[domain]:.2f} indiv={individual_ppls[domain]:.2f} "
            f"unif={uniform_ppls[domain]:.2f} routed={routed_ppls[domain]:.2f})")

    # Confusion
    log("\n  Top confusion pairs (evaluation-time):")
    for pair, count in sorted(eval_results["confusion_top15"].items(), key=lambda x: -x[1])[:10]:
        log(f"    {pair}: {count}")

    # Comparison with binary heads
    log("\n  Comparison with binary heads experiment:")
    log(f"    Binary heads top-1:     39.6%")
    log(f"    Multi-class router top-1: {top1_acc:.1%}")
    log(f"    Binary heads top-2:     54.2%")
    log(f"    Multi-class router top-2: {eval_results['top2_accuracy']:.1%}")
    log(f"    Binary heads params:    1,967,640")
    log(f"    Multi-class router params: {router_info['n_params']:,}")
    log(f"    Binary heads overhead:  6.8%")
    log(f"    Multi-class router overhead: {timing['overhead_pct']:.2f}%")

    # Verdict
    all_pass = k587_pass and k588_pass and k589_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    kill_reasons = []
    if not k587_pass:
        kill_reasons.append(f"K587 (top-1 accuracy {top1_acc:.1%} < 60%)")
    if not k588_pass:
        kill_reasons.append(f"K588 (routed PPL {avg_routed:.4f} >= uniform {avg_uniform:.4f})")
    if not k589_pass:
        kill_reasons.append(f"K589 (overhead {timing['overhead_pct']:.2f}% >= 15%)")

    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        log(f"  Kill reasons: {', '.join(kill_reasons)}")

    # Parameter counts
    first_adapter = load_adapter(ADAPTERS_SOURCE_DIR / DOMAINS[0])
    total_adapter_params = sum(v.size for v in first_adapter.values()) * N_DOMAINS
    router_ratio = router_info["n_params"] / total_adapter_params
    del first_adapter

    results = {
        "experiment": "centralized_multiclass_routing_n24",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "seed": SEED,
        "d_model": d_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Router architecture
        "router_hidden_dim": ROUTER_HIDDEN_DIM,
        "router_train_steps": ROUTER_TRAIN_STEPS,
        "router_batch_size": ROUTER_BATCH_SIZE,
        "router_lr": ROUTER_LR,
        "router_n_params": router_info["n_params"],
        "router_final_loss": router_info["final_loss"],
        "router_train_time_s": router_info["train_time_s"],
        # PPLs
        "base_ppls": base_ppls,
        "individual_ppls": individual_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "avg_base_ppl": round(avg_base, 4),
        "avg_individual_ppl": round(avg_individual, 4),
        "avg_uniform_ppl": round(avg_uniform, 4),
        "avg_routed_ppl": round(avg_routed, 4),
        # Routing accuracy
        "val_top1_accuracy": router_info["val_top1_accuracy"],
        "val_top2_accuracy": router_info["val_top2_accuracy"],
        "eval_top1_accuracy": eval_results["top1_accuracy"],
        "eval_top2_accuracy": eval_results["top2_accuracy"],
        "eval_top1_correct": eval_results["top1_correct"],
        "eval_top2_correct": eval_results["top2_correct"],
        "eval_total_routing_decisions": eval_results["total_routing_decisions"],
        "per_domain_top1": eval_results["per_domain_top1"],
        "confusion_top15": eval_results["confusion_top15"],
        # Validation-time router performance
        "router_val_per_domain": router_info["per_domain_accuracy"],
        "router_val_confusion": router_info["confusion_top15"],
        # Comparison with binary heads
        "binary_heads_top1": 0.3958,
        "binary_heads_top2": 0.5417,
        "binary_heads_params": 1967640,
        "binary_heads_overhead_pct": 6.8,
        # Timing
        "timing": timing,
        # Parameters
        "total_adapter_params": total_adapter_params,
        "router_to_adapter_ratio": round(router_ratio, 6),
        # Kill criteria
        "K587_top1_accuracy": eval_results["top1_accuracy"],
        "K587_threshold": 0.60,
        "K587_pass": k587_pass,
        "K588_avg_uniform_ppl": round(avg_uniform, 4),
        "K588_avg_routed_ppl": round(avg_routed, 4),
        "K588_improvement_pct": round(improvement, 2),
        "K588_pass": k588_pass,
        "K589_overhead_pct": timing["overhead_pct"],
        "K589_threshold": 15.0,
        "K589_pass": k589_pass,
        # Verdict
        "verdict": verdict,
        "kill_reasons": kill_reasons if kill_reasons else None,
    }

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    log("=" * 70)
    log("Centralized Multi-Class Routing N=24")
    log(f"  {N_DOMAINS} domains: {', '.join(DOMAINS)}")
    log(f"  Router: 2-layer MLP, hidden={ROUTER_HIDDEN_DIM}, {ROUTER_TRAIN_STEPS} steps")
    log(f"  Pre-trained adapters from: {ADAPTERS_SOURCE_DIR}")
    log("=" * 70)

    # Validate setup
    missing_adapters = []
    missing_data = []
    for domain in DOMAINS:
        if not (ADAPTERS_SOURCE_DIR / domain / "adapter.npz").exists():
            missing_adapters.append(domain)
        if not (DATA_DIR / domain / "valid.jsonl").exists():
            missing_data.append(domain)

    if missing_adapters:
        log(f"FATAL: Missing adapters for: {missing_adapters}")
        sys.exit(1)
    if missing_data:
        log(f"FATAL: Missing validation data for: {missing_data}")
        sys.exit(1)

    # Phase 1: Cache hidden states
    train_data, val_data, d_model = phase_cache_hidden_states(MODEL_ID)
    log_memory("after-phase1")

    # Phase 2: Train multi-class router
    router, router_info = phase_train_router(train_data, val_data, d_model)
    log_memory("after-phase2")

    # Free cached hidden states before loading model again
    del train_data, val_data
    gc.collect()
    mx.clear_cache()

    # Phase 3: Evaluate routing + composition
    eval_results = phase_evaluate(MODEL_ID, router, router_info, d_model)
    log_memory("after-phase3")

    # Phase 4: Analysis
    results = phase_analysis(eval_results, router_info, d_model)
    results["total_time_s"] = round(time.time() - t_global, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
