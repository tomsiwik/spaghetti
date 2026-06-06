#!/usr/bin/env python3
"""Fix Grassmannian A-matrix loading bug and re-test routing at N=24.

All prior routing experiments loaded adapters with RANDOM A matrices instead of
the trained Grassmannian A matrices from the skeleton. This experiment:
1. Loads adapters with CORRECT A matrices from grassmannian_skeleton_n24.npz
2. Measures oracle PPL per domain (correct adapter on its own domain)
3. Trains a centralized softmax router on base hidden states
4. Measures routing accuracy and routed PPL

Kill criteria:
  K596: Oracle PPL with correct A shows <5% improvement -> bug hypothesis wrong
  K597: Routing accuracy still <50% -> representation bottleneck real
  K598: Memory exceeds 40GB

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
ROUTER_DIR = EXPERIMENT_DIR / "router"

# Source data from the 25-domain training experiment
REAL_DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = REAL_DATA_DIR / "adapters"
DATA_DIR = REAL_DATA_DIR / "data"
SKELETON_PATH = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 20
SEED = 42

# Router config
ROUTER_HIDDEN_DIM = 64
ROUTER_LR = 3e-4
ROUTER_TRAIN_STEPS = 2000
ROUTER_BATCH_SIZE = 16
HIDDEN_CACHE_TRAIN_PER_DOMAIN = 40
HIDDEN_CACHE_VAL_PER_DOMAIN = 20

# CRITICAL: Domain ordering must match the training experiment's ordering,
# because the skeleton indices (domain_0, domain_1, ...) correspond to the
# training-time domain order, NOT alphabetical order.
# The training experiment used ALL_DOMAINS order with failed domains removed.
TRAINING_DOMAIN_ORDER = [
    "medical", "code", "math", "legal", "finance",
    "science", "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering", "agriculture",
    "environmental", "politics", "economics", "sociology", "linguistics",
    "cybersecurity", "marketing", "sports", "music",
]

# Active domains that have both adapters and data, in TRAINING order
DOMAINS = [
    d for d in TRAINING_DOMAIN_ORDER
    if (ADAPTERS_DIR / d / "adapter.npz").exists() and (DATA_DIR / d).exists()
]
N_DOMAINS = len(DOMAINS)
# SKELETON_IDX maps domain name to its index in the skeleton file
# This is the position in TRAINING_DOMAIN_ORDER (which was used to generate the skeleton)
SKELETON_IDX = {d: i for i, d in enumerate(DOMAINS)}
# DOMAIN_TO_IDX for router (can be any consistent mapping)
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


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
# Model utilities
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
    """Replace BitLinear with nn.Linear for differentiable LoRA."""
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
# TernaryLoRALinear -- the CORRECT LoRA layer matching training
# ===========================================================================
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
            self.lora_a = mx.random.uniform(
                low=-s, high=s, shape=(in_features, rank)
            )

        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank

        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        # STE ternary quantization of B (same as training)
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank, scale, a_matrices):
    """Apply TernaryLoRALinear to all target projections with Grassmannian A."""
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
            a_mx = None
            if a_key in a_matrices:
                a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16)

            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    log(f"  Applied TernaryLoRA (r={rank}) to {count} layers")
    return model


def set_lora_a(model, skeleton, domain_idx, n_layers):
    """Set A matrices from skeleton for a given domain index."""
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, TernaryLoRALinear):
                    module.lora_a = a_mx


def zero_b_params(model):
    """Zero all lora_b matrices."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict):
    model.update(tree_unflatten(list(adapter_params.items())))


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
# Multi-Class Router
# ===========================================================================
class MultiClassRouter(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ===========================================================================
# Phase 1: Measure oracle PPL with CORRECT Grassmannian A matrices
# ===========================================================================
def phase_oracle_ppl():
    """Load each adapter with correct A + B, measure domain PPL."""
    log("\n" + "=" * 70)
    log("[Phase 1] Oracle PPL with CORRECT Grassmannian A matrices")
    log("=" * 70)

    t0 = time.time()

    # Load skeleton
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"  Loaded skeleton: {len(skeleton)} A matrices")

    # Build initial A matrices (domain 0)
    n_layers = 30  # BitNet-2B-4T
    a_matrices = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    n_layers_actual = len(model.model.layers)
    d_model = model.model.layers[0].self_attn.q_proj.linear.weight.shape[1]
    log(f"  d_model={d_model}, n_layers={n_layers_actual}, N_domains={N_DOMAINS}")

    # Measure base PPL (zero B weights, any A -- doesn't matter when B=0)
    log("\n  Computing base PPL (zero B)...")
    zero_b_params(model)
    base_ppls = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        ppl = compute_ppl(model, tokenizer, texts, max_batches=VAL_BATCHES)
        base_ppls[domain] = round(ppl, 4)
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    log(f"  Average base PPL: {avg_base:.4f}")

    # Measure individual oracle PPL (correct A + trained B per domain)
    log("\n  Computing oracle PPL (correct A + trained B per domain)...")
    oracle_ppls = {}
    improvements = {}
    for di, domain in enumerate(DOMAINS):
        # Set correct A matrices for this domain
        set_lora_a(model, skeleton, di, n_layers_actual)
        # Load trained B weights
        adapter_params = load_adapter(ADAPTERS_DIR / domain)
        zero_b_params(model)
        apply_adapter_weights(model, adapter_params)
        mx.eval(model.parameters())

        texts = load_domain_texts(domain, split="valid")
        ppl = compute_ppl(model, tokenizer, texts, max_batches=VAL_BATCHES)
        oracle_ppls[domain] = round(ppl, 4)

        base = base_ppls[domain]
        imp = (base - ppl) / base * 100 if base > 0 else 0
        improvements[domain] = round(imp, 2)
        log(f"  {domain:20s}: PPL={ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    avg_oracle = sum(oracle_ppls.values()) / len(oracle_ppls)
    avg_imp = sum(improvements.values()) / len(improvements)
    log(f"\n  Average oracle PPL: {avg_oracle:.4f}")
    log(f"  Average improvement: {avg_imp:+.1f}%")
    log(f"  Phase 1 time: {time.time() - t0:.1f}s")
    log_memory("post-oracle")

    cleanup(model, tokenizer)
    del skeleton

    return base_ppls, oracle_ppls, improvements, d_model


# ===========================================================================
# Phase 2: Cache hidden states for router training
# ===========================================================================
def phase_cache_hidden_states(d_model):
    """Cache mean-pooled hidden states from base model."""
    log("\n" + "=" * 70)
    log("[Phase 2] Caching hidden states for router training")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    train_data = []
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
    log_memory("post-caching")

    cleanup(model, tokenizer)
    return train_data, val_data


# ===========================================================================
# Phase 3: Train multi-class router
# ===========================================================================
def phase_train_router(train_data, val_data, d_model):
    """Train centralized softmax router."""
    log("\n" + "=" * 70)
    log(f"[Phase 3] Training multi-class router (K={N_DOMAINS})")
    log("=" * 70)

    t0 = time.time()
    router = MultiClassRouter(d_model, N_DOMAINS, ROUTER_HIDDEN_DIM)
    mx.eval(router.parameters())

    n_params = sum(p.size for _, p in tree_flatten(router.parameters()))
    log(f"  Router params: {n_params:,}")

    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    def router_loss_fn(router, h_batch, labels_batch):
        logits = router(h_batch)
        return nn.losses.cross_entropy(logits, labels_batch, reduction="mean")

    router_loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    gc.disable()
    losses = []
    for step in range(ROUTER_TRAIN_STEPS):
        batch_indices = rng.sample(range(len(train_data)), min(ROUTER_BATCH_SIZE, len(train_data)))
        h_list = [train_data[idx][0] for idx in batch_indices]
        label_list = [train_data[idx][1] for idx in batch_indices]

        h_batch = mx.concatenate(h_list, axis=0)
        labels_batch = mx.array(label_list)

        loss, grads = router_loss_and_grad(router, h_batch, labels_batch)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            recent = sum(losses[-100:]) / len(losses[-100:])
            log(f"  Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={recent:.4f}")

    gc.enable()
    gc.collect()

    final_loss = sum(losses[-100:]) / len(losses[-100:])
    log(f"  Final loss: {final_loss:.4f}")

    # Validate
    correct_top1 = 0
    correct_top2 = 0
    total = 0
    per_domain_correct = {d: 0 for d in DOMAINS}
    per_domain_total = {d: 0 for d in DOMAINS}

    for h_pool, label in val_data:
        logits = router(h_pool)
        mx.eval(logits)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        probs_list = probs[0].tolist()

        sorted_indices = sorted(range(N_DOMAINS), key=lambda k: probs_list[k], reverse=True)
        true_domain = DOMAINS[label]

        if sorted_indices[0] == label:
            correct_top1 += 1
            per_domain_correct[true_domain] += 1
        if label in sorted_indices[:2]:
            correct_top2 += 1
        per_domain_total[true_domain] += 1
        total += 1

    top1_acc = correct_top1 / total if total > 0 else 0
    top2_acc = correct_top2 / total if total > 0 else 0
    log(f"  Val top-1: {top1_acc:.1%}, top-2: {top2_acc:.1%}")

    per_domain_acc = {}
    for d in DOMAINS:
        if per_domain_total[d] > 0:
            per_domain_acc[d] = round(per_domain_correct[d] / per_domain_total[d], 4)
        else:
            per_domain_acc[d] = 0.0

    # Save router
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    router_params = dict(tree_flatten(router.parameters()))
    mx.savez(str(ROUTER_DIR / "router.npz"), **router_params)

    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")

    del optimizer, router_loss_and_grad
    gc.collect()
    mx.clear_cache()

    router_info = {
        "n_params": n_params,
        "final_loss": round(final_loss, 4),
        "train_time_s": round(train_time, 1),
        "val_top1_accuracy": round(top1_acc, 4),
        "val_top2_accuracy": round(top2_acc, 4),
        "per_domain_accuracy": per_domain_acc,
    }
    return router, router_info


# ===========================================================================
# Phase 4: Evaluate routed composition with CORRECT A matrices
# ===========================================================================
def phase_evaluate_routed(router, base_ppls, oracle_ppls, d_model):
    """Evaluate router-guided composition with correct Grassmannian A loading."""
    log("\n" + "=" * 70)
    log("[Phase 4] Evaluating routed composition with CORRECT A matrices")
    log("=" * 70)

    t0 = time.time()

    # Load skeleton for A matrices
    skeleton = dict(np.load(str(SKELETON_PATH)))

    # Build initial A matrices (domain 0)
    a_matrices = {}
    for li in range(30):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    n_layers = len(model.model.layers)

    # Pre-load all adapter B weights
    all_adapter_params = {}
    for domain in DOMAINS:
        all_adapter_params[domain] = load_adapter(ADAPTERS_DIR / domain)
    log(f"  Loaded {len(all_adapter_params)} adapter B-weight sets")

    # Evaluate per-sample routing + composition
    routed_ppls = {}
    top1_correct = 0
    top2_correct = 0
    total_routing = 0
    per_domain_top1 = {d: {"correct": 0, "total": 0} for d in DOMAINS}
    confusion_matrix = {}

    for eval_domain in DOMAINS:
        texts = load_domain_texts(eval_domain, split="valid")
        domain_losses = 0.0
        domain_tokens = 0

        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x_tokens = tokens[:-1]
            y_tokens = tokens[1:]

            x = mx.array(x_tokens)[None, :]

            # Get hidden states for routing (base model, B=0)
            zero_b_params(model)
            h = get_hidden_states(model, x)
            h_pool = mx.mean(h, axis=1)
            mx.eval(h_pool)
            del h

            # Route
            logits = router(h_pool)
            probs = mx.softmax(logits, axis=-1)
            mx.eval(probs)
            probs_list = probs[0].tolist()
            del logits

            sorted_indices = sorted(range(N_DOMAINS), key=lambda k: probs_list[k], reverse=True)
            top1_idx = sorted_indices[0]
            top2_indices = sorted_indices[:2]
            top2_domains = [DOMAINS[i] for i in top2_indices]
            top2_probs = [probs_list[i] for i in top2_indices]

            # Track accuracy
            true_label = DOMAIN_TO_IDX[eval_domain]
            if top1_idx == true_label:
                top1_correct += 1
                per_domain_top1[eval_domain]["correct"] += 1
            else:
                pred_name = DOMAINS[top1_idx]
                ck = f"{eval_domain} -> {pred_name}"
                confusion_matrix[ck] = confusion_matrix.get(ck, 0) + 1

            if true_label in top2_indices:
                top2_correct += 1
            total_routing += 1
            per_domain_top1[eval_domain]["total"] += 1

            # Apply top-1 adapter with CORRECT A matrix
            winner_domain = DOMAINS[top1_idx]
            winner_idx = DOMAIN_TO_IDX[winner_domain]

            # Set correct A for winner domain
            set_lora_a(model, skeleton, winner_idx, n_layers)
            # Load winner's B weights
            zero_b_params(model)
            apply_adapter_weights(model, all_adapter_params[winner_domain])
            mx.eval(model.parameters())

            # Forward pass with correct adapter
            y = mx.array(y_tokens)[None, :]
            model_logits = model(x)
            loss = nn.losses.cross_entropy(model_logits, y, reduction="sum")
            mx.eval(loss)

            domain_losses += loss.item()
            domain_tokens += y.size

            del model_logits, loss, x, y, h_pool, probs

        if domain_tokens > 0:
            avg_loss = domain_losses / domain_tokens
            ppl = math.exp(min(avg_loss, 100))
        else:
            ppl = float("inf")

        routed_ppls[eval_domain] = round(ppl, 4)
        log(f"  {eval_domain:20s}: routed_ppl={ppl:.2f} oracle={oracle_ppls.get(eval_domain, 'N/A')} base={base_ppls.get(eval_domain, 'N/A')}")

    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)
    top1_acc = top1_correct / total_routing if total_routing > 0 else 0
    top2_acc = top2_correct / total_routing if total_routing > 0 else 0

    log(f"\n  Average routed PPL: {avg_routed:.4f}")
    log(f"  Eval top-1: {top1_acc:.1%}, top-2: {top2_acc:.1%}")
    log(f"  Phase 4 time: {time.time() - t0:.1f}s")
    log_memory("post-eval")

    per_domain_top1_pct = {}
    for d in DOMAINS:
        info = per_domain_top1[d]
        per_domain_top1_pct[d] = round(info["correct"] / info["total"], 4) if info["total"] > 0 else 0

    cleanup(model, tokenizer)
    del skeleton, all_adapter_params

    return {
        "routed_ppls": routed_ppls,
        "avg_routed_ppl": round(avg_routed, 4),
        "eval_top1_accuracy": round(top1_acc, 4),
        "eval_top2_accuracy": round(top2_acc, 4),
        "eval_top1_correct": top1_correct,
        "eval_top2_correct": top2_correct,
        "eval_total": total_routing,
        "per_domain_top1": per_domain_top1_pct,
        "confusion_top15": dict(sorted(confusion_matrix.items(), key=lambda x: -x[1])[:15]),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log(f"Fix Grassmannian Loading Retest Routing")
    log(f"Domains: {N_DOMAINS} -- {', '.join(DOMAINS)}")
    log_memory("start")

    # Phase 1: Oracle PPL with correct A
    base_ppls, oracle_ppls, improvements, d_model = phase_oracle_ppl()

    # Phase 2: Cache hidden states
    train_data, val_data = phase_cache_hidden_states(d_model)

    # Phase 3: Train router
    router, router_info = phase_train_router(train_data, val_data, d_model)
    del train_data, val_data
    gc.collect()
    mx.clear_cache()

    # Phase 4: Evaluate routed composition
    eval_results = phase_evaluate_routed(router, base_ppls, oracle_ppls, d_model)
    del router

    # Compute summary stats
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_oracle = sum(oracle_ppls.values()) / len(oracle_ppls)
    avg_imp = sum(improvements.values()) / len(improvements)
    n_specialized = sum(1 for v in improvements.values() if v >= 5.0)

    # Kill criteria assessment
    k596_pass = avg_imp >= 5.0  # Oracle PPL >= 5% improvement
    k597_pass = eval_results["eval_top1_accuracy"] >= 0.50  # Routing accuracy >= 50%
    peak_gb = mx.get_peak_memory() / 1e9
    k598_pass = peak_gb < 40.0

    results = {
        "experiment": "fix_grassmannian_loading_retest",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "seed": SEED,
        "d_model": d_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_ppls": base_ppls,
        "oracle_ppls": oracle_ppls,
        "improvements_pct": improvements,
        "avg_base_ppl": round(avg_base, 4),
        "avg_oracle_ppl": round(avg_oracle, 4),
        "avg_improvement_pct": round(avg_imp, 2),
        "n_specialized_5pct": n_specialized,
        "router": router_info,
        **eval_results,
        "kill_criteria": {
            "K596_oracle_improvement_ge_5pct": "PASS" if k596_pass else "FAIL",
            "K597_routing_accuracy_ge_50pct": "PASS" if k597_pass else "FAIL",
            "K598_memory_lt_40GB": "PASS" if k598_pass else "FAIL",
            "peak_memory_gb": round(peak_gb, 2),
        },
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Avg base PPL:    {avg_base:.4f}")
    log(f"  Avg oracle PPL:  {avg_oracle:.4f} ({avg_imp:+.1f}%)")
    log(f"  Domains >= 5%:   {n_specialized}/{N_DOMAINS}")
    log(f"  Router top-1:    {eval_results['eval_top1_accuracy']:.1%}")
    log(f"  Router top-2:    {eval_results['eval_top2_accuracy']:.1%}")
    log(f"  Avg routed PPL:  {eval_results['avg_routed_ppl']:.4f}")
    log(f"  Peak memory:     {peak_gb:.1f} GB")
    log(f"  Total time:      {results['total_time_s']:.0f}s")
    log(f"\n  K596 (oracle >= 5% imp):   {'PASS' if k596_pass else 'FAIL'}")
    log(f"  K597 (routing >= 50%):     {'PASS' if k597_pass else 'FAIL'}")
    log(f"  K598 (memory < 40GB):      {'PASS' if k598_pass else 'FAIL'}")

    # Comparison with buggy experiment
    log(f"\n  --- Comparison with buggy centralized_multiclass_routing_n24 ---")
    log(f"  Buggy avg oracle PPL:  10.1192 (-0.6% vs base)")
    log(f"  Fixed avg oracle PPL:  {avg_oracle:.4f} ({avg_imp:+.1f}% vs base)")
    log(f"  Buggy routing top-1:   39.2%")
    log(f"  Fixed routing top-1:   {eval_results['eval_top1_accuracy']:.1%}")


if __name__ == "__main__":
    main()
