#!/usr/bin/env python3
"""
Dynamic Adapter Addition (K3 Test): Does N+1 composition degrade N composition?

K1 PASS (95% head accuracy), K2 PASS (0.3% degradation) from prior run.
This script tests ONLY K3: composition with N+1 adapters worse than N adapters.

Protocol:
  1. Train 5 domain adapters on BitNet-2B-4T (python, math, medical, legal, creative)
  2. Train routing heads for all 5
  3. Measure N=5 routed top-2 composition PPL on all 5 domains
  4. Add 6th adapter (science) + its routing head, NO retraining of existing heads
  5. Measure N=6 routed top-2 composition PPL on original 5 domains
  6. K3 PASS if N=6 PPL <= N=5 PPL (+ small tolerance for noise)

Kill criteria:
  K3: avg(routed PPL at N=6) > avg(routed PPL at N=5) on original 5 domains

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
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42

# Routing head config
HEAD_HIDDEN_DIM = 32
HEAD_LR = 3e-4
HEAD_TRAIN_STEPS = 500

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
HEADS_DIR = EXPERIMENT_DIR / "heads"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition for original 5 domains
REFERENCE_DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

ORIGINAL_DOMAINS = ["python", "math", "medical", "legal", "creative"]
NEW_DOMAIN = "science"
ALL_DOMAINS = ORIGINAL_DOMAINS + [NEW_DOMAIN]

# Science domain config (new domain added at runtime)
SCIENCE_CONFIG = {
    "hf_dataset": "sciq",
    "text_key": "support",
    "max_samples_train": 500,
    "max_samples_val": 50,
}


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
# BitLinear unpacking (from proven pipeline)
# ===========================================================================
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


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


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


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_texts(domain, split="valid"):
    """Load text data for a domain."""
    if domain == NEW_DOMAIN:
        data_dir = EXPERIMENT_DIR / "data" / NEW_DOMAIN
    else:
        data_dir = REFERENCE_DATA_DIR / domain
    fpath = data_dir / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def prepare_science_data():
    """Download and prepare science domain data."""
    data_dir = EXPERIMENT_DIR / "data" / NEW_DOMAIN
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        log("  Science data already prepared")
        return data_dir

    from datasets import load_dataset as hf_load

    data_dir.mkdir(parents=True, exist_ok=True)
    log("  Downloading SciQ dataset...")
    ds = hf_load(SCIENCE_CONFIG["hf_dataset"])

    text_key = SCIENCE_CONFIG["text_key"]
    split_data = ds["train"]

    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= SCIENCE_CONFIG["max_samples_train"] + SCIENCE_CONFIG["max_samples_val"]:
            break

    train_texts = texts[:SCIENCE_CONFIG["max_samples_train"]]
    val_texts = texts[SCIENCE_CONFIG["max_samples_train"]:
                      SCIENCE_CONFIG["max_samples_train"] + SCIENCE_CONFIG["max_samples_val"]]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    log(f"  Science: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


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
# Routing Head
# ===========================================================================
class RoutingHead(nn.Module):
    """Tiny binary classifier: h_pool -> sigmoid score."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ===========================================================================
# Phase 1: Train all adapters (5 original + 1 science)
# ===========================================================================
def phase_train_adapters():
    """Train LoRA adapters for all 6 domains."""
    log("\n" + "=" * 70)
    log("[Phase 1] Training domain adapters")
    log("=" * 70)

    # Check if all adapters exist
    all_exist = all(
        (ADAPTERS_DIR / d / "adapter.npz").exists() for d in ALL_DOMAINS
    )
    if all_exist:
        log("  All 6 adapters already trained, skipping.")
        return

    # Load model
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {n_trainable:,}")

    for domain in ALL_DOMAINS:
        adapter_path = ADAPTERS_DIR / domain
        if (adapter_path / "adapter.npz").exists():
            log(f"\n  [{domain}] Already trained, skipping.")
            continue

        log(f"\n  [{domain}] Training adapter...")
        t_domain = time.time()

        zero_lora_params(model)

        texts = load_domain_texts(domain, split="train")
        if not texts:
            log(f"  WARNING: No training data for {domain}, skipping")
            continue

        # Tokenize
        train_tokens = []
        for text in texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

        log(f"    {len(train_tokens)} training sequences")

        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        gc.disable()
        losses = []
        for step in range(TRAIN_ITERS):
            idx = step % len(train_tokens)
            tokens = train_tokens[idx]
            x = tokens[:-1][None, :]
            y = tokens[1:][None, :]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                log(f"    Step {step+1}/{TRAIN_ITERS}: avg_loss={avg:.4f}")

        gc.enable()
        gc.collect()

        save_adapter(model, adapter_path)
        final_loss = sum(losses[-20:]) / len(losses[-20:])
        log(f"    Done in {time.time() - t_domain:.1f}s, final_loss={final_loss:.4f}")

    log_memory("after-adapter-training")
    cleanup(model, tokenizer)


# ===========================================================================
# Phase 2: Train routing heads for all domains
# ===========================================================================
def phase_train_routing_heads():
    """Train per-adapter routing heads using base model hidden states."""
    log("\n" + "=" * 70)
    log("[Phase 2] Training routing heads for all 6 domains")
    log("=" * 70)

    # Check if all heads exist
    all_exist = all(
        (HEADS_DIR / d / "head.npz").exists() for d in ALL_DOMAINS
    )
    if all_exist:
        log("  All 6 heads already trained, skipping.")
        return

    # Load model for hidden state extraction
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")

    # Cache hidden states for all domains
    log("  Pre-computing hidden states...")
    t_cache = time.time()

    domain_hidden_train = {}
    domain_hidden_val = {}

    for domain in ALL_DOMAINS:
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
                h = get_hidden_states(model, x)
                h_pool = mx.mean(h, axis=1)
                mx.eval(h_pool)
                hiddens.append(h_pool)
                del h, x
            cache[domain] = hiddens
            log(f"    {domain}/{split}: {len(hiddens)} samples")

    log(f"  Hidden states cached in {time.time() - t_cache:.1f}s")
    log_memory("after-hidden-caching")

    cleanup(model, tokenizer)

    # Train heads
    rng = random.Random(SEED)
    head_results = {}

    for domain in ALL_DOMAINS:
        head_path = HEADS_DIR / domain
        if (head_path / "head.npz").exists():
            log(f"\n  [{domain}] Head already trained, skipping.")
            continue

        log(f"\n  [{domain}] Training routing head...")
        t_head = time.time()

        head = RoutingHead(d_model, HEAD_HIDDEN_DIM)
        mx.eval(head.parameters())

        n_params = sum(p.size for _, p in tree_flatten(head.parameters()))
        log(f"    Head params: {n_params:,}")

        optimizer = opt.Adam(learning_rate=HEAD_LR)

        positives = domain_hidden_train[domain]
        negatives = []
        for other in ALL_DOMAINS:
            if other != domain:
                negatives.extend(domain_hidden_train[other])

        def head_loss_fn(head, h_pool, label):
            logit = head(h_pool)
            target = mx.array([[label]], dtype=mx.float32)
            return nn.losses.binary_cross_entropy(
                mx.sigmoid(logit), target, reduction="mean"
            )

        head_loss_and_grad = nn.value_and_grad(head, head_loss_fn)

        gc.disable()
        losses = []
        for step in range(HEAD_TRAIN_STEPS):
            if step % 2 == 0:
                h = positives[rng.randint(0, len(positives) - 1)]
                label = 1.0
            else:
                h = negatives[rng.randint(0, len(negatives) - 1)]
                label = 0.0

            loss, grads = head_loss_and_grad(head, h, label)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            losses.append(loss.item())
        gc.enable()
        gc.collect()

        # Evaluate head
        correct = 0
        total = 0
        for h in domain_hidden_val[domain]:
            logit = head(h)
            mx.eval(logit)
            if logit.item() > 0:
                correct += 1
            total += 1

        neg_correct = 0
        neg_total = 0
        for other in ALL_DOMAINS:
            if other == domain:
                continue
            for h in domain_hidden_val[other][:10]:
                logit = head(h)
                mx.eval(logit)
                if logit.item() <= 0:
                    neg_correct += 1
                neg_total += 1
                total += 1

        accuracy = (correct + neg_correct) / total if total > 0 else 0
        own_acc = correct / len(domain_hidden_val[domain]) if domain_hidden_val[domain] else 0

        log(f"    Accuracy: {accuracy:.1%} (own_domain: {own_acc:.1%})")
        log(f"    Time: {time.time() - t_head:.1f}s")

        head_results[domain] = {
            "accuracy": round(accuracy, 4),
            "own_domain_accuracy": round(own_acc, 4),
            "n_params": n_params,
        }

        # Save head
        head_path.mkdir(parents=True, exist_ok=True)
        head_params = dict(tree_flatten(head.parameters()))
        mx.savez(str(head_path / "head.npz"), **{k: v for k, v in head_params.items()})

    return head_results


# ===========================================================================
# Phase 3: K3 Test - N=5 vs N=6 routed composition
# ===========================================================================
def phase_k3_test():
    """Compare N=5 vs N=6 routed top-2 composition on original 5 domains."""
    log("\n" + "=" * 70)
    log("[Phase 3] K3 Test: N=5 vs N=6 Routed Composition")
    log("=" * 70)

    # Load model
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Get d_model from the LoRA-wrapped layer
    q_proj = model.model.layers[0].self_attn.q_proj
    if isinstance(q_proj, LoRALinear):
        d_model = q_proj.linear.weight.shape[1]
    else:
        d_model = q_proj.weight.shape[1]

    # Load all adapters
    adapters = {}
    for domain in ALL_DOMAINS:
        adapters[domain] = load_adapter(ADAPTERS_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Load all routing heads
    heads = {}
    for domain in ALL_DOMAINS:
        head = RoutingHead(d_model, HEAD_HIDDEN_DIM)
        head_data = dict(mx.load(str(HEADS_DIR / domain / "head.npz")))
        head.update(tree_unflatten(list(head_data.items())))
        mx.eval(head.parameters())
        heads[domain] = head

    # Load validation texts for original 5 domains
    val_texts = {}
    for domain in ORIGINAL_DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        val_texts[domain] = texts
        log(f"  Val data: {domain} = {len(texts)} texts")

    # Also load science val for completeness
    val_texts[NEW_DOMAIN] = load_domain_texts(NEW_DOMAIN, split="valid")
    log(f"  Val data: {NEW_DOMAIN} = {len(val_texts[NEW_DOMAIN])} texts")

    # --- Base PPL ---
    log("\n  Computing base PPL (no adapters)...")
    zero_adapter_in_model(model)
    base_ppls = {}
    for domain in ORIGINAL_DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain])
        base_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f}")

    # --- Individual adapter PPL (oracle) ---
    log("\n  Computing individual adapter PPL...")
    individual_ppls = {}
    for domain in ALL_DOMAINS:
        zero_adapter_in_model(model)
        apply_adapter_to_model(model, adapters[domain])
        mx.eval(model.parameters())
        texts = val_texts.get(domain, load_domain_texts(domain, split="valid"))
        ppl = compute_ppl(model, tokenizer, texts)
        individual_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f}")

    # --- Helper: routed top-2 composition PPL ---
    def compute_routed_ppl(active_domains, eval_domains):
        """Compute routed top-2 PPL using only heads/adapters from active_domains."""
        active_heads = {d: heads[d] for d in active_domains}
        active_adapters = {d: adapters[d] for d in active_domains}

        results = {}
        routing_log = {}

        for eval_domain in eval_domains:
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

                # Get hidden states for routing (from base model, no adapter)
                zero_adapter_in_model(model)
                h = get_hidden_states(model, x)
                h_pool = mx.mean(h, axis=1)
                mx.eval(h_pool)
                del h

                # Score all active heads
                scores = {}
                for head_domain, head in active_heads.items():
                    logit = head(h_pool)
                    mx.eval(logit)
                    scores[head_domain] = mx.sigmoid(logit).item()

                # Top-2 selection
                sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top2 = sorted_scores[:2]
                domain_routing.append({d: round(s, 4) for d, s in sorted_scores})

                # Pre-merge top-2 with score-weighted composition
                total_score = sum(s for _, s in top2)
                if total_score < 1e-8:
                    total_score = 1.0
                composed = {}
                for sel_domain, sel_score in top2:
                    w = sel_score / total_score
                    for key, val in active_adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * w
                        else:
                            composed[key] = composed[key] + val * w

                # Forward pass with composed adapter
                apply_adapter_to_model(model, composed)
                mx.eval(model.parameters())
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

            results[eval_domain] = round(ppl, 4)
            routing_log[eval_domain] = domain_routing

        return results, routing_log

    # --- N=5 routed composition (original 5 domains only) ---
    log("\n  Computing N=5 routed top-2 composition PPL...")
    n5_ppls, n5_routing = compute_routed_ppl(ORIGINAL_DOMAINS, ORIGINAL_DOMAINS)
    for domain, ppl in n5_ppls.items():
        log(f"    {domain}: {ppl}")

    avg_n5 = sum(n5_ppls.values()) / len(n5_ppls)
    log(f"    Avg N=5: {avg_n5:.4f}")

    # --- N=6 routed composition (all 6 domains, eval on original 5) ---
    log("\n  Computing N=6 routed top-2 composition PPL (eval on original 5)...")
    n6_ppls, n6_routing = compute_routed_ppl(ALL_DOMAINS, ORIGINAL_DOMAINS)
    for domain, ppl in n6_ppls.items():
        log(f"    {domain}: {ppl}")

    avg_n6 = sum(n6_ppls.values()) / len(n6_ppls)
    log(f"    Avg N=6: {avg_n6:.4f}")

    # --- N=6 on science domain (new domain quality) ---
    log("\n  Computing N=6 routed PPL on NEW domain (science)...")
    n6_science_ppls, n6_science_routing = compute_routed_ppl(ALL_DOMAINS, [NEW_DOMAIN])
    log(f"    science: {n6_science_ppls.get(NEW_DOMAIN, 'N/A')}")

    # --- Uniform 1/N composition for comparison ---
    log("\n  Computing uniform 1/N composition PPL (N=5 and N=6)...")

    def compose_uniform(domain_list):
        adapter_list = [adapters[d] for d in domain_list]
        N = len(adapter_list)
        merged = {}
        for key in adapter_list[0].keys():
            stacked = mx.stack([a[key] for a in adapter_list])
            merged[key] = mx.mean(stacked, axis=0)
        return merged

    # N=5 uniform
    zero_adapter_in_model(model)
    merged_5 = compose_uniform(ORIGINAL_DOMAINS)
    apply_adapter_to_model(model, merged_5)
    mx.eval(model.parameters())
    uniform_n5_ppls = {}
    for domain in ORIGINAL_DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain])
        uniform_n5_ppls[domain] = round(ppl, 4)
    avg_uniform_n5 = sum(uniform_n5_ppls.values()) / len(uniform_n5_ppls)
    del merged_5

    # N=6 uniform
    zero_adapter_in_model(model)
    merged_6 = compose_uniform(ALL_DOMAINS)
    apply_adapter_to_model(model, merged_6)
    mx.eval(model.parameters())
    uniform_n6_ppls = {}
    for domain in ORIGINAL_DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain])
        uniform_n6_ppls[domain] = round(ppl, 4)
    avg_uniform_n6 = sum(uniform_n6_ppls.values()) / len(uniform_n6_ppls)
    del merged_6

    # --- Orthogonality check ---
    log("\n  Computing adapter orthogonality...")
    cosines = {}
    for i, d1 in enumerate(ALL_DOMAINS):
        for d2 in ALL_DOMAINS[i+1:]:
            v1 = mx.concatenate([v.reshape(-1) for v in adapters[d1].values()])
            v2 = mx.concatenate([v.reshape(-1) for v in adapters[d2].values()])
            cos = mx.abs(mx.sum(v1 * v2) / (mx.sqrt(mx.sum(v1**2)) * mx.sqrt(mx.sum(v2**2))))
            mx.eval(cos)
            pair = f"{d1}-{d2}"
            cosines[pair] = round(cos.item(), 6)
    mean_cos = sum(cosines.values()) / len(cosines)
    log(f"    Mean |cos|: {mean_cos:.6f}")

    # --- Routing accuracy of science head on all domains ---
    log("\n  Science head routing accuracy on all domains...")
    science_head = heads[NEW_DOMAIN]
    science_scores_by_domain = {}
    for domain in ALL_DOMAINS:
        texts = val_texts.get(domain, load_domain_texts(domain, split="valid"))
        scores = []
        for text in texts[:15]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            zero_adapter_in_model(model)
            h = get_hidden_states(model, x)
            h_pool = mx.mean(h, axis=1)
            mx.eval(h_pool)
            logit = science_head(h_pool)
            mx.eval(logit)
            scores.append(mx.sigmoid(logit).item())
            del h, x, h_pool
        avg_score = sum(scores) / len(scores) if scores else 0
        science_scores_by_domain[domain] = round(avg_score, 4)
        log(f"    {domain}: avg_sigmoid={avg_score:.4f}")

    # --- K3 verdict ---
    delta = avg_n6 - avg_n5
    delta_pct = delta / avg_n5 * 100
    k3_pass = avg_n6 <= avg_n5 * 1.02  # 2% tolerance for noise

    log("\n" + "=" * 70)
    log("K3 RESULTS")
    log("=" * 70)
    log(f"  N=5 routed avg PPL: {avg_n5:.4f}")
    log(f"  N=6 routed avg PPL: {avg_n6:.4f}")
    log(f"  Delta: {delta:+.4f} ({delta_pct:+.2f}%)")
    log(f"  K3: {'PASS' if k3_pass else 'FAIL'} (threshold: N=6 <= N=5 * 1.02)")
    log(f"")
    log(f"  Uniform 1/N N=5: {avg_uniform_n5:.4f}")
    log(f"  Uniform 1/N N=6: {avg_uniform_n6:.4f}")
    log(f"  Routed advantage over uniform N=5: {(avg_uniform_n5 - avg_n5)/avg_uniform_n5*100:.1f}%")
    log(f"  Routed advantage over uniform N=6: {(avg_uniform_n6 - avg_n6)/avg_uniform_n6*100:.1f}%")

    # Per-domain breakdown
    log("\n  Per-domain breakdown:")
    log(f"  {'Domain':<12} {'Base':>8} {'Indiv':>8} {'N=5 rout':>10} {'N=6 rout':>10} {'Delta':>8}")
    for domain in ORIGINAL_DOMAINS:
        d = n6_ppls[domain] - n5_ppls[domain]
        log(f"  {domain:<12} {base_ppls[domain]:>8.2f} {individual_ppls[domain]:>8.2f} "
            f"{n5_ppls[domain]:>10.2f} {n6_ppls[domain]:>10.2f} {d:>+8.2f}")

    # Analyze routing changes
    log("\n  Routing changes (how often science head enters top-2):")
    for domain in ORIGINAL_DOMAINS:
        science_in_top2 = 0
        total = len(n6_routing.get(domain, []))
        for decision in n6_routing.get(domain, []):
            sorted_d = sorted(decision.items(), key=lambda kv: kv[1], reverse=True)
            top2_domains = [d for d, _ in sorted_d[:2]]
            if NEW_DOMAIN in top2_domains:
                science_in_top2 += 1
        pct = science_in_top2 / total * 100 if total > 0 else 0
        log(f"    {domain}: science in top-2 {science_in_top2}/{total} ({pct:.0f}%)")

    cleanup(model, tokenizer)

    return {
        "base_ppls": base_ppls,
        "individual_ppls": individual_ppls,
        "n5_routed_ppls": n5_ppls,
        "n6_routed_ppls": n6_ppls,
        "n6_science_ppl": n6_science_ppls,
        "uniform_n5_ppls": uniform_n5_ppls,
        "uniform_n6_ppls": uniform_n6_ppls,
        "avg_n5_routed": round(avg_n5, 4),
        "avg_n6_routed": round(avg_n6, 4),
        "avg_uniform_n5": round(avg_uniform_n5, 4),
        "avg_uniform_n6": round(avg_uniform_n6, 4),
        "delta_ppl": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
        "k3_pass": k3_pass,
        "cosines": cosines,
        "mean_cos": round(mean_cos, 6),
        "science_scores_by_domain": science_scores_by_domain,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("Dynamic Adapter Addition: K3 Test (N=5 vs N=6)")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Prepare science data
    log("\n[Phase 0] Preparing science domain data...")
    prepare_science_data()

    # Phase 1: Train all adapters
    phase_train_adapters()
    log_memory("after-phase1")

    # Phase 2: Train routing heads
    head_results = phase_train_routing_heads()
    log_memory("after-phase2")

    # Phase 3: K3 test
    k3_results = phase_k3_test()
    log_memory("after-phase3")

    # Save results
    results = {
        "experiment": "dynamic_adapter_addition_k3",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "original_domains": ORIGINAL_DOMAINS,
        "new_domain": NEW_DOMAIN,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(time.time() - t_start, 1),
        "head_results": head_results,
        **k3_results,
        "verdict": "K3 PASS" if k3_results["k3_pass"] else "K3 FAIL",
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
