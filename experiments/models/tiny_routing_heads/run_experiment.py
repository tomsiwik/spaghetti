#!/usr/bin/env python3
"""
Tiny Routing Heads: Per-adapter binary routing for expert composition.

Each adapter carries its own tiny routing head (~82K params, 2-layer MLP) that
predicts 'am I useful for this input?' (binary classifier on adapter's own domain
vs random data). At inference, heads vote, top-2 adapters are selected and pre-merged.

Kill criteria:
  K1: Per-adapter head accuracy < 75% on own domain -> KILL
  K2: Head inference overhead > 5% of base forward pass -> KILL
  K3: Head-routed composition PPL worse than uniform 1/N -> KILL

Success criteria:
  S1: Head-routed top-2 PPL beats uniform 1/N by > 5%
  S2: Total head params < 1% of adapter params
  S3: Adding a new adapter requires only training its head

Platform: Apple M5 Pro 48GB, MLX, $0.
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
TRAIN_ITERS = 200
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42

# Routing head config
HEAD_HIDDEN_DIM = 32       # 2-layer MLP hidden size
HEAD_LR = 3e-4
HEAD_TRAIN_STEPS = 500     # Per-head training steps
HEAD_NEG_RATIO = 1.0       # Ratio of negative to positive samples

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]
N_DOMAINS = len(DOMAINS)


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
# Ternary unpacking (from existing experiments)
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


# ===========================================================================
# LoRA helpers
# ===========================================================================
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


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


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
def compute_ppl(model, tokenizer, texts, max_batches=25):
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
# Routing Head: per-adapter binary classifier
# ===========================================================================
class RoutingHead(nn.Module):
    """Tiny binary classifier: h_pool -> sigmoid score.

    Architecture: Linear(d, h) -> ReLU -> Linear(h, 1) -> sigmoid
    Input: mean-pooled hidden state (d,)
    Output: scalar probability in [0, 1]
    """

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)  # raw logit, apply sigmoid in loss


# ===========================================================================
# Phase 1: Train domain adapters
# ===========================================================================
def phase_train_adapters(model_id):
    """Train 5 domain LoRA adapters. Returns adapter save paths."""
    log("\n" + "=" * 70)
    log("[Phase 1] Training 5 domain adapters")
    log("=" * 70)

    # Check for pre-existing adapters
    existing = []
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if adapter_path.exists():
            existing.append(domain)
    if len(existing) == len(DOMAINS):
        log("  All adapters already trained, skipping.")
        return {d: ADAPTERS_DIR / d for d in DOMAINS}

    # Load model
    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze only LoRA params
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {n_trainable:,}")

    adapter_paths = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain
        if (adapter_path / "adapter.npz").exists():
            log(f"\n  [{domain}] Already trained, skipping.")
            adapter_paths[domain] = adapter_path
            continue

        log(f"\n  [{domain}] Training adapter...")
        t_domain = time.time()

        # Reset LoRA params
        zero_lora_params(model)

        # Load training data
        texts = load_domain_texts(domain, split="train")
        if not texts:
            log(f"  WARNING: No training data for {domain}")
            continue

        # Training loop
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        gc.disable()
        losses = []
        for step in range(TRAIN_ITERS):
            text = texts[step % len(texts)]
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                log(f"    Step {step+1}/{TRAIN_ITERS}: loss={avg:.4f}")

        gc.enable()
        gc.collect()

        # Save adapter
        save_adapter(model, adapter_path)
        adapter_paths[domain] = adapter_path
        log(f"    Trained in {time.time() - t_domain:.1f}s, "
            f"final_loss={sum(losses[-20:])/len(losses[-20:]):.4f}")

    log_memory("after-adapter-training")
    cleanup(model, tokenizer)
    return adapter_paths


# ===========================================================================
# Phase 2: Train per-adapter routing heads
# ===========================================================================
def phase_train_routing_heads(model_id, adapter_paths):
    """Train per-adapter routing heads. Each head is a binary classifier."""
    log("\n" + "=" * 70)
    log("[Phase 2] Training per-adapter routing heads")
    log("=" * 70)

    # Load model for hidden state extraction only
    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    # Get model dimension
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")

    # Pre-compute hidden states for all domains (train + val)
    log("  Pre-computing hidden states for all domain data...")
    t_cache = time.time()

    domain_hidden_train = {}  # domain -> list of mean-pooled hidden states
    domain_hidden_val = {}

    for domain in DOMAINS:
        for split, cache in [("train", domain_hidden_train), ("valid", domain_hidden_val)]:
            texts = load_domain_texts(domain, split=split)
            max_samples = 80 if split == "train" else 25
            hiddens = []
            for text in texts[:max_samples]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = get_hidden_states(model, x)  # (1, seq_len, d)
                h_pool = mx.mean(h, axis=1)  # (1, d)
                mx.eval(h_pool)
                hiddens.append(h_pool)
                del h, x
            cache[domain] = hiddens
            log(f"    {domain}/{split}: {len(hiddens)} samples cached")

    log(f"  Hidden state caching: {time.time() - t_cache:.1f}s")
    log_memory("after-hidden-caching")

    # Free model (we only need cached hidden states now)
    cleanup(model, tokenizer)

    # Train one head per adapter
    heads = {}
    head_results = {}
    rng = random.Random(SEED)

    for domain in DOMAINS:
        log(f"\n  Training head for [{domain}]...")
        t_head = time.time()

        head = RoutingHead(d_model, HEAD_HIDDEN_DIM)
        mx.eval(head.parameters())

        n_params = sum(p.size for _, p in tree_flatten(head.parameters()))
        log(f"    Head params: {n_params:,}")

        optimizer = opt.Adam(learning_rate=HEAD_LR)

        # Build training data: positive = own domain, negative = other domains
        positives = domain_hidden_train[domain]
        negatives = []
        for other in DOMAINS:
            if other != domain:
                negatives.extend(domain_hidden_train[other])

        def head_loss_fn(head, h_pool, label):
            logit = head(h_pool)  # (1, 1)
            target = mx.array([[label]], dtype=mx.float32)
            return nn.losses.binary_cross_entropy(mx.sigmoid(logit), target, reduction="mean")

        head_loss_and_grad = nn.value_and_grad(head, head_loss_fn)

        gc.disable()
        losses = []
        for step in range(HEAD_TRAIN_STEPS):
            # Alternate positive and negative samples
            if step % 2 == 0:
                # Positive sample (own domain)
                h = positives[rng.randint(0, len(positives) - 1)]
                label = 1.0
            else:
                # Negative sample (other domain)
                h = negatives[rng.randint(0, len(negatives) - 1)]
                label = 0.0

            loss, grads = head_loss_and_grad(head, h, label)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            losses.append(loss.item())

        gc.enable()
        gc.collect()

        # Evaluate head on validation data
        correct = 0
        total = 0

        # Positive eval (own domain)
        own_correct = 0
        own_total = 0
        for h in domain_hidden_val[domain]:
            logit = head(h)
            mx.eval(logit)
            pred = logit.item() > 0  # Threshold at 0 (sigmoid(0) = 0.5)
            if pred:
                correct += 1
                own_correct += 1
            total += 1
            own_total += 1

        # Negative eval (other domains)
        neg_correct = 0
        neg_total = 0
        for other in DOMAINS:
            if other == domain:
                continue
            for h in domain_hidden_val[other][:10]:  # Limit per-domain negatives
                logit = head(h)
                mx.eval(logit)
                pred = logit.item() > 0
                if not pred:  # Correct = predicting negative
                    correct += 1
                    neg_correct += 1
                total += 1
                neg_total += 1

        accuracy = correct / total if total > 0 else 0
        own_acc = own_correct / own_total if own_total > 0 else 0
        neg_acc = neg_correct / neg_total if neg_total > 0 else 0

        final_loss = sum(losses[-50:]) / len(losses[-50:])
        log(f"    Accuracy: {accuracy:.1%} (own={own_acc:.1%}, neg={neg_acc:.1%})")
        log(f"    Final loss: {final_loss:.4f}, Time: {time.time() - t_head:.1f}s")

        heads[domain] = head
        head_results[domain] = {
            "accuracy": round(accuracy, 4),
            "own_domain_accuracy": round(own_acc, 4),
            "negative_accuracy": round(neg_acc, 4),
            "n_params": n_params,
            "final_loss": round(final_loss, 4),
            "train_time_s": round(time.time() - t_head, 1),
        }

        # Save head weights
        head_path = EXPERIMENT_DIR / "heads" / domain
        head_path.mkdir(parents=True, exist_ok=True)
        head_params = dict(tree_flatten(head.parameters()))
        mx.savez(str(head_path / "head.npz"), **{k: v for k, v in head_params.items()})

    return heads, head_results, d_model


# ===========================================================================
# Phase 3: Evaluate routing + composition PPL
# ===========================================================================
def phase_evaluate(model_id, adapter_paths, heads, head_results, d_model):
    """Evaluate: base PPL, individual, uniform 1/N, head-routed top-2."""
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating composition strategies")
    log("=" * 70)

    # Load model
    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load adapters from disk
    adapters = {}
    for domain in DOMAINS:
        path = adapter_paths[domain]
        adapters[domain] = load_adapter(path)
        log(f"  Loaded adapter: {domain}")

    # Load validation texts
    val_texts = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        val_texts[domain] = texts
        log(f"  Val data: {domain} = {len(texts)} texts")

    # --- Base PPL ---
    log("\n  Computing base PPL (no adapters)...")
    zero_adapter_in_model(model)
    base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        base_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f}")

    # --- Individual adapter PPL (oracle) ---
    log("\n  Computing individual adapter PPL (oracle routing)...")
    individual_ppls = {}
    for domain in DOMAINS:
        apply_adapter_to_model(model, adapters[domain])
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        individual_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f} (base: {base_ppls[domain]})")
        zero_adapter_in_model(model)

    # --- Uniform 1/N composition PPL ---
    log("\n  Computing uniform 1/N composition PPL...")
    composed_1n = {}
    for key in adapters[DOMAINS[0]].keys():
        stacked = mx.stack([adapters[d][key] for d in DOMAINS])
        composed_1n[key] = mx.mean(stacked, axis=0)

    apply_adapter_to_model(model, composed_1n)
    uniform_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        uniform_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f}")
    zero_adapter_in_model(model)
    del composed_1n

    # --- Head-routed top-2 composition PPL ---
    log("\n  Computing head-routed top-2 composition PPL...")
    routed_ppls = {}
    routing_decisions = {}

    for eval_domain in DOMAINS:
        domain_losses = 0.0
        domain_tokens = 0
        domain_routing = []

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

            # All heads score in parallel
            scores = {}
            for head_domain, head in heads.items():
                logit = head(h_pool)
                mx.eval(logit)
                scores[head_domain] = mx.sigmoid(logit).item()

            # Top-2 selection
            sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_domains[:2]
            domain_routing.append({d: round(s, 4) for d, s in sorted_domains})

            # Pre-merge top-2 with score-weighted composition
            total_score = sum(s for _, s in top2)
            composed = {}
            for sel_domain, sel_score in top2:
                w = sel_score / total_score
                for key, val in adapters[sel_domain].items():
                    if key not in composed:
                        composed[key] = val * w
                    else:
                        composed[key] = composed[key] + val * w

            # Forward pass with composed adapter
            apply_adapter_to_model(model, composed)
            y = mx.array(y_tokens)[None, :]
            logits = model(x)
            loss = nn.losses.cross_entropy(logits, y, reduction="sum")
            mx.eval(loss)

            domain_losses += loss.item()
            domain_tokens += y.size

            zero_adapter_in_model(model)
            del composed, logits, loss, x, y, h_pool

        if domain_tokens > 0:
            avg_loss = domain_losses / domain_tokens
            ppl = math.exp(min(avg_loss, 100))
        else:
            ppl = float("inf")

        routed_ppls[eval_domain] = round(ppl, 4)
        routing_decisions[eval_domain] = domain_routing
        log(f"    {eval_domain}: {ppl:.4f} (uniform: {uniform_ppls[eval_domain]}, "
            f"oracle: {individual_ppls[eval_domain]})")

    # --- Head inference overhead measurement (K2) ---
    log("\n  Measuring head inference overhead (K2)...")

    # Time base forward pass
    test_text = val_texts[DOMAINS[0]][0]
    test_tokens = tokenizer.encode(test_text)[:MAX_SEQ_LENGTH]
    test_x = mx.array(test_tokens)[None, :]

    # Warm up
    for _ in range(3):
        _ = model(test_x)
        mx.eval(_)

    # Time base forward pass
    n_timing = 20
    t_base_start = time.time()
    for _ in range(n_timing):
        out = model(test_x)
        mx.eval(out)
    t_base = (time.time() - t_base_start) / n_timing

    # Time head inference only (all 5 heads on pooled hidden state)
    h = get_hidden_states(model, test_x)
    h_pool = mx.mean(h, axis=1)
    mx.eval(h_pool)

    # Warm up heads
    for _ in range(3):
        for head in heads.values():
            out = head(h_pool)
            mx.eval(out)

    t_head_start = time.time()
    for _ in range(n_timing):
        for head in heads.values():
            out = head(h_pool)
            mx.eval(out)
    t_heads = (time.time() - t_head_start) / n_timing

    overhead_pct = (t_heads / t_base) * 100
    log(f"    Base forward: {t_base*1000:.2f}ms")
    log(f"    All heads: {t_heads*1000:.2f}ms")
    log(f"    Overhead: {overhead_pct:.2f}%")

    del test_x, h, h_pool

    log_memory("after-eval")
    cleanup(model, tokenizer)

    return {
        "base_ppls": base_ppls,
        "individual_ppls": individual_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "routing_decisions": routing_decisions,
        "head_results": head_results,
        "timing": {
            "base_forward_ms": round(t_base * 1000, 2),
            "all_heads_ms": round(t_heads * 1000, 2),
            "overhead_pct": round(overhead_pct, 2),
        },
    }


# ===========================================================================
# Phase 4: Analysis and kill criteria
# ===========================================================================
def phase_analysis(eval_results, d_model):
    """Assess kill criteria and write results."""
    log("\n" + "=" * 70)
    log("[Phase 4] Kill criteria assessment")
    log("=" * 70)

    base_ppls = eval_results["base_ppls"]
    individual_ppls = eval_results["individual_ppls"]
    uniform_ppls = eval_results["uniform_ppls"]
    routed_ppls = eval_results["routed_ppls"]
    head_results = eval_results["head_results"]
    timing = eval_results["timing"]

    # --- K1: Per-adapter head accuracy >= 75% ---
    log("\n  K1: Per-adapter head accuracy >= 75%")
    k1_pass = True
    for domain, hr in head_results.items():
        acc = hr["accuracy"]
        status = "PASS" if acc >= 0.75 else "FAIL"
        if acc < 0.75:
            k1_pass = False
        log(f"    {domain}: {acc:.1%} [{status}]")
    log(f"  K1 overall: {'PASS' if k1_pass else 'FAIL'}")

    # --- K2: Head inference overhead < 5% ---
    log(f"\n  K2: Head inference overhead < 5%")
    k2_pass = timing["overhead_pct"] < 5.0
    log(f"    Overhead: {timing['overhead_pct']:.2f}% [{'PASS' if k2_pass else 'FAIL'}]")

    # --- K3: Head-routed PPL better than uniform 1/N ---
    log(f"\n  K3: Head-routed PPL better than uniform 1/N")
    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    avg_base = sum(base_ppls.values()) / len(base_ppls)

    k3_pass = avg_routed < avg_uniform
    improvement = ((avg_uniform - avg_routed) / avg_uniform) * 100

    log(f"    Avg base PPL:       {avg_base:.4f}")
    log(f"    Avg individual PPL: {avg_individual:.4f}")
    log(f"    Avg uniform PPL:    {avg_uniform:.4f}")
    log(f"    Avg routed PPL:     {avg_routed:.4f}")
    log(f"    Improvement:        {improvement:+.1f}%")
    log(f"  K3 overall: {'PASS' if k3_pass else 'FAIL'}")

    # Per-domain breakdown
    log("\n  Per-domain comparison:")
    for domain in DOMAINS:
        log(f"    {domain}: base={base_ppls[domain]:.2f} individual={individual_ppls[domain]:.2f} "
            f"uniform={uniform_ppls[domain]:.2f} routed={routed_ppls[domain]:.2f}")

    # --- S1: Beats uniform by > 5% ---
    s1_pass = improvement > 5.0
    log(f"\n  S1 (>5% improvement over uniform): {improvement:.1f}% [{'PASS' if s1_pass else 'FAIL'}]")

    # --- S2: Head params < 1% of adapter params ---
    total_head_params = sum(hr["n_params"] for hr in head_results.values())
    # Estimate adapter params from first adapter
    first_adapter = ADAPTERS_DIR / DOMAINS[0] / "adapter.npz"
    adapter_data = dict(mx.load(str(first_adapter)))
    total_adapter_params = sum(v.size for v in adapter_data.values()) * N_DOMAINS
    head_ratio = total_head_params / total_adapter_params
    s2_pass = head_ratio < 0.01
    log(f"  S2 (head params < 1% of adapter params): {total_head_params:,} / {total_adapter_params:,} "
        f"= {head_ratio:.4%} [{'PASS' if s2_pass else 'FAIL'}]")
    del adapter_data

    # --- S3: Adding new adapter = training its head only ---
    s3_pass = True  # By construction: each head trains independently
    log(f"  S3 (independent head training): PASS (by construction)")

    # Routing accuracy analysis
    log("\n  Routing decision analysis:")
    for eval_domain in DOMAINS:
        decisions = eval_results["routing_decisions"][eval_domain]
        if not decisions:
            continue
        # How often is the correct domain in top-2?
        correct_in_top2 = 0
        for routing in decisions:
            sorted_r = sorted(routing.items(), key=lambda x: x[1], reverse=True)
            top2_domains = [d for d, _ in sorted_r[:2]]
            if eval_domain in top2_domains:
                correct_in_top2 += 1
        recall = correct_in_top2 / len(decisions)
        log(f"    {eval_domain}: correct adapter in top-2 = {correct_in_top2}/{len(decisions)} ({recall:.1%})")

    # Overall verdict
    all_kill_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if all_kill_pass else "KILLED"
    kill_reasons = []
    if not k1_pass:
        kill_reasons.append("K1 (head accuracy < 75%)")
    if not k2_pass:
        kill_reasons.append("K2 (head overhead > 5%)")
    if not k3_pass:
        kill_reasons.append("K3 (routed PPL worse than uniform)")

    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        log(f"  Kill reasons: {', '.join(kill_reasons)}")

    results = {
        "experiment": "tiny_routing_heads",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "head_train_steps": HEAD_TRAIN_STEPS,
        "seed": SEED,
        "d_model": d_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # PPLs
        "base_ppls": base_ppls,
        "individual_ppls": individual_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "avg_base_ppl": round(avg_base, 4),
        "avg_individual_ppl": round(avg_individual, 4),
        "avg_uniform_ppl": round(avg_uniform, 4),
        "avg_routed_ppl": round(avg_routed, 4),
        # Head results
        "head_results": head_results,
        "total_head_params": total_head_params,
        "total_adapter_params": total_adapter_params,
        "head_to_adapter_ratio": round(head_ratio, 6),
        # Timing
        "timing": timing,
        # Kill criteria
        "K1_per_head_accuracy": {d: hr["accuracy"] for d, hr in head_results.items()},
        "K1_threshold": 0.75,
        "K1_pass": k1_pass,
        "K2_overhead_pct": timing["overhead_pct"],
        "K2_threshold": 5.0,
        "K2_pass": k2_pass,
        "K3_avg_uniform_ppl": round(avg_uniform, 4),
        "K3_avg_routed_ppl": round(avg_routed, 4),
        "K3_improvement_pct": round(improvement, 2),
        "K3_pass": k3_pass,
        # Success criteria
        "S1_improvement_gt_5pct": s1_pass,
        "S2_head_ratio_lt_1pct": s2_pass,
        "S3_independent_training": s3_pass,
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
    log("Tiny Routing Heads: Per-adapter binary routing for composition")
    log(f"  {N_DOMAINS} domains: {', '.join(DOMAINS)}")
    log(f"  Head: 2-layer MLP, hidden={HEAD_HIDDEN_DIM}, {HEAD_TRAIN_STEPS} steps")
    log(f"  LoRA: rank={LORA_RANK}, scale={LORA_SCALE}, {TRAIN_ITERS} train iters")
    log("=" * 70)

    # Check data exists
    for domain in DOMAINS:
        if not (DATA_DIR / domain / "train.jsonl").exists():
            log(f"FATAL: Missing data for {domain} at {DATA_DIR / domain}")
            sys.exit(1)

    # Phase 1: Train adapters
    adapter_paths = phase_train_adapters(MODEL_ID)
    log_memory("after-phase1")

    # Phase 2: Train routing heads
    heads, head_results, d_model = phase_train_routing_heads(MODEL_ID, adapter_paths)
    log_memory("after-phase2")

    # Phase 3: Evaluate
    eval_results = phase_evaluate(MODEL_ID, adapter_paths, heads, head_results, d_model)
    log_memory("after-phase3")

    # Phase 4: Analysis
    results = phase_analysis(eval_results, d_model)
    results["total_time_s"] = round(time.time() - t_global, 1)

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
