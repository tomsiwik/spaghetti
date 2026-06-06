#!/usr/bin/env python3
"""
Tiny Routing Heads at N=24: Frontier extension from N=5 to 24 domains.

Adapters are PRE-TRAINED (from real_data_25_domain_adapters). This experiment
only trains routing heads and evaluates routing + composition quality.

Kill criteria:
  K584: Top-1 routing accuracy <60% at N=24 -> KILL
  K585: Routed composition PPL worse than uniform 1/N at N=24 -> KILL
  K586: Per-query overhead >10% of base forward pass at N=24 -> KILL

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

# Routing head config -- same as N=5
HEAD_HIDDEN_DIM = 32
HEAD_LR = 3e-4
HEAD_TRAIN_STEPS = 500
HEAD_NEG_RATIO = 1.0

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
HEADS_DIR = EXPERIMENT_DIR / "heads"

# Pre-trained adapters and data from real_data_25_domain_adapters
REAL_DATA_DIR = Path(__file__).parent.parent / "real_data_25_domain_adapters"
ADAPTERS_SOURCE_DIR = REAL_DATA_DIR / "adapters"
DATA_DIR = REAL_DATA_DIR / "data"

# 24 domains that have both adapters and data (real_estate has data but no adapter)
DOMAINS = sorted([
    d.name for d in ADAPTERS_SOURCE_DIR.iterdir()
    if d.is_dir() and (DATA_DIR / d.name).exists()
])
N_DOMAINS = len(DOMAINS)

# Hidden state cache config -- process in batches to manage memory
HIDDEN_CACHE_TRAIN_PER_DOMAIN = 40  # fewer samples since N is large
HIDDEN_CACHE_VAL_PER_DOMAIN = 15


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
# Model utilities (from N=5 experiment)
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
# Routing Head: per-adapter binary classifier
# ===========================================================================
class RoutingHead(nn.Module):
    """Tiny binary classifier: h_pool -> sigmoid score."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)  # raw logit


# ===========================================================================
# Phase 1: Cache hidden states for all 24 domains
# ===========================================================================
def phase_cache_hidden_states(model_id):
    """Load base model, extract and cache mean-pooled hidden states for all domains."""
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

    domain_hidden_train = {}
    domain_hidden_val = {}

    for i, domain in enumerate(DOMAINS):
        for split, cache, max_s in [
            ("train", domain_hidden_train, HIDDEN_CACHE_TRAIN_PER_DOMAIN),
            ("valid", domain_hidden_val, HIDDEN_CACHE_VAL_PER_DOMAIN),
        ]:
            texts = load_domain_texts(domain, split=split)
            hiddens = []
            for text in texts[:max_s]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = get_hidden_states(model, x)
                h_pool = mx.mean(h, axis=1)  # (1, d)
                mx.eval(h_pool)
                hiddens.append(h_pool)
                del h, x
            cache[domain] = hiddens

        if (i + 1) % 8 == 0:
            log(f"  Cached {i+1}/{N_DOMAINS} domains...")
            log_memory(f"after-{i+1}-domains")

    log(f"  Hidden state caching done in {time.time() - t0:.1f}s")
    log(f"  Train samples per domain: {HIDDEN_CACHE_TRAIN_PER_DOMAIN}")
    log(f"  Val samples per domain: {HIDDEN_CACHE_VAL_PER_DOMAIN}")
    log_memory("after-caching")

    cleanup(model, tokenizer)
    return domain_hidden_train, domain_hidden_val, d_model


# ===========================================================================
# Phase 2: Train per-adapter routing heads (24 heads)
# ===========================================================================
def phase_train_routing_heads(domain_hidden_train, domain_hidden_val, d_model):
    """Train 24 binary routing heads, one per adapter."""
    log("\n" + "=" * 70)
    log(f"[Phase 2] Training {N_DOMAINS} per-adapter routing heads")
    log("=" * 70)

    heads = {}
    head_results = {}
    rng = random.Random(SEED)

    for domain in DOMAINS:
        t_head = time.time()

        head = RoutingHead(d_model, HEAD_HIDDEN_DIM)
        mx.eval(head.parameters())

        n_params = sum(p.size for _, p in tree_flatten(head.parameters()))

        optimizer = opt.Adam(learning_rate=HEAD_LR)

        # Positive = own domain, negative = all other domains
        positives = domain_hidden_train[domain]
        negatives = []
        for other in DOMAINS:
            if other != domain:
                negatives.extend(domain_hidden_train[other])

        def head_loss_fn(head, h_pool, label):
            logit = head(h_pool)
            target = mx.array([[label]], dtype=mx.float32)
            return nn.losses.binary_cross_entropy(mx.sigmoid(logit), target, reduction="mean")

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

        # Evaluate on validation data
        correct = 0
        total = 0
        own_correct = 0
        own_total = 0

        for h in domain_hidden_val[domain]:
            logit = head(h)
            mx.eval(logit)
            pred = logit.item() > 0
            if pred:
                correct += 1
                own_correct += 1
            total += 1
            own_total += 1

        neg_correct = 0
        neg_total = 0
        for other in DOMAINS:
            if other == domain:
                continue
            for h in domain_hidden_val[other][:5]:  # 5 per other domain -> 115 negatives
                logit = head(h)
                mx.eval(logit)
                pred = logit.item() > 0
                if not pred:
                    correct += 1
                    neg_correct += 1
                total += 1
                neg_total += 1

        accuracy = correct / total if total > 0 else 0
        own_acc = own_correct / own_total if own_total > 0 else 0
        neg_acc = neg_correct / neg_total if neg_total > 0 else 0
        final_loss = sum(losses[-50:]) / len(losses[-50:])

        log(f"  [{domain:20s}] acc={accuracy:.1%} own={own_acc:.1%} neg={neg_acc:.1%} "
            f"loss={final_loss:.4f} time={time.time() - t_head:.1f}s")

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
        head_path = HEADS_DIR / domain
        head_path.mkdir(parents=True, exist_ok=True)
        head_params = dict(tree_flatten(head.parameters()))
        mx.savez(str(head_path / "head.npz"), **{k: v for k, v in head_params.items()})

        # Clean up optimizer to free memory
        del optimizer, head_loss_and_grad, positives, negatives
        gc.collect()
        mx.clear_cache()

    log_memory("after-head-training")
    return heads, head_results


# ===========================================================================
# Phase 3: Evaluate routing + composition PPL
# ===========================================================================
def phase_evaluate(model_id, heads, head_results, d_model):
    """Evaluate routing accuracy, composition PPL, and overhead."""
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating routing and composition at N=24")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load adapters from disk one at a time for composition, keep in memory
    # 24 adapters x ~23.5M params each = ~564M params. At bfloat16 ~1.1GB -- fits.
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

    # --- Base PPL (sample of domains to save time) ---
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

    # --- Head-routed top-2 composition PPL ---
    log("\n  Computing head-routed top-2 composition PPL...")
    routed_ppls = {}
    routing_decisions = {}
    top1_correct = 0
    top2_correct = 0
    total_routing = 0

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
            h_pool = mx.mean(h, axis=1)
            mx.eval(h_pool)
            del h

            # All heads score
            scores = {}
            for head_domain, head in heads.items():
                logit = head(h_pool)
                mx.eval(logit)
                scores[head_domain] = mx.sigmoid(logit).item()

            # Top-2 selection
            sorted_domains = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top2 = sorted_domains[:2]

            # Track routing accuracy
            if sorted_domains[0][0] == eval_domain:
                top1_correct += 1
            if eval_domain in [d for d, _ in top2]:
                top2_correct += 1
            total_routing += 1

            domain_routing.append({d: round(s, 4) for d, s in sorted_domains[:5]})

            # Pre-merge top-2 with score-weighted composition
            total_score = sum(s for _, s in top2)
            composed = {}
            if total_score > 1e-8:
                for sel_domain, sel_score in top2:
                    w = sel_score / total_score
                    for key, val in adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * w
                        else:
                            composed[key] = composed[key] + val * w
            else:
                # Fallback to uniform top-2
                for sel_domain, _ in top2:
                    for key, val in adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * 0.5
                        else:
                            composed[key] = composed[key] + val * 0.5

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

    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)
    top1_acc = top1_correct / total_routing if total_routing > 0 else 0
    top2_acc = top2_correct / total_routing if total_routing > 0 else 0
    log(f"    Average routed PPL: {avg_routed:.4f}")
    log(f"    Top-1 routing accuracy: {top1_acc:.1%} ({top1_correct}/{total_routing})")
    log(f"    Top-2 routing accuracy: {top2_acc:.1%} ({top2_correct}/{total_routing})")

    # --- Head inference overhead (K586) ---
    log("\n  Measuring head inference overhead...")
    test_text = val_texts[DOMAINS[0]][0]
    test_tokens = tokenizer.encode(test_text)[:MAX_SEQ_LENGTH]
    test_x = mx.array(test_tokens)[None, :]

    # Warm up
    for _ in range(3):
        out = model(test_x)
        mx.eval(out)

    n_timing = 20
    t_base_start = time.time()
    for _ in range(n_timing):
        out = model(test_x)
        mx.eval(out)
    t_base = (time.time() - t_base_start) / n_timing

    # Time all 24 heads
    h = get_hidden_states(model, test_x)
    h_pool = mx.mean(h, axis=1)
    mx.eval(h_pool)

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
    log(f"    All {N_DOMAINS} heads: {t_heads*1000:.2f}ms")
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
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "total_routing_decisions": total_routing,
        "timing": {
            "base_forward_ms": round(t_base * 1000, 2),
            "all_heads_ms": round(t_heads * 1000, 2),
            "overhead_pct": round(overhead_pct, 2),
        },
    }


# ===========================================================================
# Phase 4: Kill criteria assessment
# ===========================================================================
def phase_analysis(eval_results, d_model):
    """Assess kill criteria and produce final results."""
    log("\n" + "=" * 70)
    log("[Phase 4] Kill criteria assessment")
    log("=" * 70)

    base_ppls = eval_results["base_ppls"]
    individual_ppls = eval_results["individual_ppls"]
    uniform_ppls = eval_results["uniform_ppls"]
    routed_ppls = eval_results["routed_ppls"]
    head_results = eval_results["head_results"]
    timing = eval_results["timing"]

    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)

    # --- K584: Top-1 routing accuracy >= 60% ---
    top1_acc = eval_results["top1_accuracy"]
    k584_pass = top1_acc >= 0.60
    log(f"\n  K584: Top-1 routing accuracy >= 60%")
    log(f"    Measured: {top1_acc:.1%} [{'PASS' if k584_pass else 'FAIL'}]")

    # --- K585: Routed PPL better than uniform 1/N ---
    k585_pass = avg_routed < avg_uniform
    improvement = ((avg_uniform - avg_routed) / avg_uniform) * 100
    log(f"\n  K585: Routed PPL < uniform 1/N")
    log(f"    Uniform: {avg_uniform:.4f}, Routed: {avg_routed:.4f}")
    log(f"    Improvement: {improvement:+.1f}% [{'PASS' if k585_pass else 'FAIL'}]")

    # --- K586: Per-query overhead < 10% ---
    k586_pass = timing["overhead_pct"] < 10.0
    log(f"\n  K586: Per-query overhead < 10%")
    log(f"    Overhead: {timing['overhead_pct']:.2f}% [{'PASS' if k586_pass else 'FAIL'}]")

    # Per-head accuracy breakdown
    log("\n  Per-head accuracy breakdown:")
    accs = []
    for domain in DOMAINS:
        hr = head_results[domain]
        accs.append(hr["accuracy"])
        log(f"    {domain:20s}: acc={hr['accuracy']:.1%} "
            f"own={hr['own_domain_accuracy']:.1%} neg={hr['negative_accuracy']:.1%}")
    avg_acc = sum(accs) / len(accs)
    min_acc = min(accs)
    log(f"  Average head accuracy: {avg_acc:.1%}")
    log(f"  Min head accuracy: {min_acc:.1%}")

    # Per-domain routing analysis
    log("\n  Per-domain routing analysis:")
    per_domain_top1 = {}
    for eval_domain in DOMAINS:
        decisions = eval_results["routing_decisions"][eval_domain]
        if not decisions:
            per_domain_top1[eval_domain] = 0
            continue
        correct = 0
        for routing in decisions:
            sorted_r = sorted(routing.items(), key=lambda kv: kv[1], reverse=True)
            if sorted_r[0][0] == eval_domain:
                correct += 1
        recall = correct / len(decisions)
        per_domain_top1[eval_domain] = round(recall, 4)
        log(f"    {eval_domain:20s}: top-1={recall:.1%} "
            f"(base={base_ppls[eval_domain]:.2f} indiv={individual_ppls[eval_domain]:.2f} "
            f"unif={uniform_ppls[eval_domain]:.2f} routed={routed_ppls[eval_domain]:.2f})")

    # Confusion analysis: which domains get confused?
    log("\n  Confusion matrix (top-1 misroutes):")
    confusion = {}
    for eval_domain in DOMAINS:
        decisions = eval_results["routing_decisions"][eval_domain]
        for routing in decisions:
            sorted_r = sorted(routing.items(), key=lambda kv: kv[1], reverse=True)
            predicted = sorted_r[0][0]
            if predicted != eval_domain:
                key = f"{eval_domain} -> {predicted}"
                confusion[key] = confusion.get(key, 0) + 1
    for pair, count in sorted(confusion.items(), key=lambda x: -x[1])[:15]:
        log(f"    {pair}: {count} misroutes")

    # Verdict
    all_pass = k584_pass and k585_pass and k586_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    kill_reasons = []
    if not k584_pass:
        kill_reasons.append(f"K584 (top-1 accuracy {top1_acc:.1%} < 60%)")
    if not k585_pass:
        kill_reasons.append(f"K585 (routed PPL {avg_routed:.4f} >= uniform {avg_uniform:.4f})")
    if not k586_pass:
        kill_reasons.append(f"K586 (overhead {timing['overhead_pct']:.2f}% >= 10%)")

    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        log(f"  Kill reasons: {', '.join(kill_reasons)}")

    # Parameter efficiency
    total_head_params = sum(hr["n_params"] for hr in head_results.values())
    first_adapter = load_adapter(ADAPTERS_SOURCE_DIR / DOMAINS[0])
    total_adapter_params = sum(v.size for v in first_adapter.values()) * N_DOMAINS
    head_ratio = total_head_params / total_adapter_params
    del first_adapter

    results = {
        "experiment": "tiny_routing_heads_n24",
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
        # Routing
        "top1_accuracy": eval_results["top1_accuracy"],
        "top2_accuracy": eval_results["top2_accuracy"],
        "top1_correct": eval_results["top1_correct"],
        "top2_correct": eval_results["top2_correct"],
        "total_routing_decisions": eval_results["total_routing_decisions"],
        "per_domain_top1": per_domain_top1,
        "confusion_top15": dict(sorted(confusion.items(), key=lambda x: -x[1])[:15]),
        # Head results
        "head_results": head_results,
        "avg_head_accuracy": round(avg_acc, 4),
        "min_head_accuracy": round(min_acc, 4),
        "total_head_params": total_head_params,
        "total_adapter_params": total_adapter_params,
        "head_to_adapter_ratio": round(head_ratio, 6),
        # Timing
        "timing": timing,
        # Kill criteria
        "K584_top1_accuracy": eval_results["top1_accuracy"],
        "K584_threshold": 0.60,
        "K584_pass": k584_pass,
        "K585_avg_uniform_ppl": round(avg_uniform, 4),
        "K585_avg_routed_ppl": round(avg_routed, 4),
        "K585_improvement_pct": round(improvement, 2),
        "K585_pass": k585_pass,
        "K586_overhead_pct": timing["overhead_pct"],
        "K586_threshold": 10.0,
        "K586_pass": k586_pass,
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
    log("Tiny Routing Heads N=24: Frontier extension from N=5")
    log(f"  {N_DOMAINS} domains: {', '.join(DOMAINS)}")
    log(f"  Head: 2-layer MLP, hidden={HEAD_HIDDEN_DIM}, {HEAD_TRAIN_STEPS} steps")
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
    domain_hidden_train, domain_hidden_val, d_model = phase_cache_hidden_states(MODEL_ID)
    log_memory("after-phase1")

    # Phase 2: Train routing heads
    heads, head_results = phase_train_routing_heads(domain_hidden_train, domain_hidden_val, d_model)
    log_memory("after-phase2")

    # Free cached hidden states before loading model again
    del domain_hidden_train, domain_hidden_val
    gc.collect()
    mx.clear_cache()

    # Phase 3: Evaluate routing + composition
    eval_results = phase_evaluate(MODEL_ID, heads, head_results, d_model)
    log_memory("after-phase3")

    # Phase 4: Analysis
    results = phase_analysis(eval_results, d_model)
    results["total_time_s"] = round(time.time() - t_global, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
