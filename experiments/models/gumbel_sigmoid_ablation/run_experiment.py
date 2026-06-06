#!/usr/bin/env python3
"""
Gumbel-Sigmoid Routing Ablation Study

Systematic ablation of router configuration for N=50 ternary LoRA composition.
Reuses trained adapters and data from bitnet_scale_n50.

Kill criteria:
  K1 (id=264): No config beats current default (86.33% top-2 accuracy) by >5% -> KILL

Ablation axes:
  1. Temperature: fixed 0.1, 0.5, 1.0, 2.0; anneal 2.0->0.5 (baseline), 1.0->0.1
  2. Top-k: 1, 2 (baseline), 3, 4
  3. Gate type: sigmoid (baseline) vs softmax (competing)
  4. Load-balancing loss: none (baseline), alpha=0.01, alpha=0.1
  5. Straight-through Gumbel: soft (baseline) vs hard ST

Platform: Apple M5 Pro 48GB, MLX, local.
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
N50_DIR = Path(__file__).parent.parent / "bitnet_scale_n50"
HIDDEN_STATES_FILE = EXPERIMENT_DIR / "hidden_states.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
MAX_SEQ_LENGTH = 128
SEED = 42

# Data directory mapping (copied from N=50 experiment)
N50_DATA_DIR = N50_DIR / "data"
MODELS_DIR = Path(__file__).parent.parent


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
# Data path resolution (mirrors N=50 experiment)
# ===========================================================================
EXISTING_DATA_DIRS = {
    "code": MODELS_DIR / "bitnet_ternary_convergence" / "data" / "code",
    "math": MODELS_DIR / "bitnet_ternary_convergence" / "data" / "math",
    "legal": MODELS_DIR / "bitnet_ternary_convergence" / "data" / "legal",
    "creative": MODELS_DIR / "bitnet_ternary_convergence" / "data" / "creative",
    "sql": MODELS_DIR / "bitnet_scale_n15" / "data" / "sql",
    "javascript": MODELS_DIR / "bitnet_scale_n15" / "data" / "javascript",
    "physics": MODELS_DIR / "bitnet_scale_n15" / "data" / "physics",
    "chemistry": MODELS_DIR / "bitnet_scale_n15" / "data" / "chemistry",
    "science": MODELS_DIR / "bitnet_scale_n15" / "data" / "science",
    "wikitext": MODELS_DIR / "bitnet_scale_n15" / "data" / "wikitext",
    "finance": MODELS_DIR / "bitnet_scale_n15" / "data" / "finance",
    "cooking": MODELS_DIR / "bitnet_scale_n15" / "data" / "cooking",
    "health": MODELS_DIR / "bitnet_scale_n15" / "data" / "health",
    "dialogue": MODELS_DIR / "bitnet_scale_n15" / "data" / "dialogue",
    "reasoning": MODELS_DIR / "capability_expert_taxonomy" / "data" / "reasoning",
    "instruction": MODELS_DIR / "capability_expert_taxonomy" / "data" / "instruction",
    "conciseness": MODELS_DIR / "capability_expert_taxonomy" / "data" / "conciseness",
    "safety": MODELS_DIR / "capability_expert_taxonomy" / "data" / "safety",
    "multilingual": MODELS_DIR / "bitnet_scale_n25" / "data" / "multilingual",
    "coding_style": MODELS_DIR / "bitnet_scale_n25" / "data" / "coding_style",
    "summarization": MODELS_DIR / "bitnet_scale_n25" / "data" / "summarization",
    "debate": MODELS_DIR / "bitnet_scale_n25" / "data" / "debate",
    "translation": MODELS_DIR / "bitnet_scale_n25" / "data" / "translation",
    "formal_writing": MODELS_DIR / "bitnet_scale_n25" / "data" / "formal_writing",
}

# Group B domains: data in N50_DATA_DIR
NEW_DOMAINS = [
    "history", "philosophy", "sports", "poetry", "news", "reviews",
    "qa_pairs", "stories", "science_qa", "recipes", "trivia", "eli5",
    "movie_plots", "tweets", "abstracts", "contracts", "emails",
    "bash_code", "math_proofs", "dialogues_2", "product_desc", "bio_text",
    "travel", "tech_docs", "music_text",
]


def get_data_dir(domain_name):
    """Resolve data directory for a domain."""
    if domain_name in EXISTING_DATA_DIRS:
        return EXISTING_DATA_DIRS[domain_name]
    if domain_name in NEW_DOMAINS:
        return N50_DATA_DIR / domain_name
    return None


def get_all_domain_names():
    """Get ordered list of domains that have adapters (matching N=50 ordering)."""
    # Load from N=50 results to ensure consistent ordering
    with open(N50_DIR / "results.json") as f:
        n50_results = json.load(f)
    all_names = n50_results["all_names"]
    # Filter to those with adapters (medical was excluded in N=50)
    adapters_dir = N50_DIR / "adapters"
    active = [n for n in all_names if (adapters_dir / n).exists()]
    return active


# ===========================================================================
# BitLinear unpacking (from N=50)
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
            from mlx.utils import tree_unflatten
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# Router variants
# ===========================================================================
class SigmoidRouter(nn.Module):
    """Gumbel-sigmoid router (non-competing, independent Bernoulli gates)."""
    def __init__(self, input_dim, n_adapters, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, n_adapters)
        self.n_adapters = n_adapters

    def __call__(self, h, temperature=1.0, hard=False, straight_through=False):
        z = nn.gelu(self.proj(h))
        logits = self.gate(z)

        if hard:
            gates = (logits > 0).astype(mx.float32)
        else:
            u = mx.random.uniform(shape=logits.shape)
            u = mx.clip(u, 1e-6, 1.0 - 1e-6)
            gumbel_noise = -mx.log(-mx.log(u))
            gates = mx.sigmoid((logits + gumbel_noise) / temperature)

            if straight_through:
                # Straight-through: hard forward, soft backward
                hard_gates = (gates > 0.5).astype(mx.float32)
                gates = gates + mx.stop_gradient(hard_gates - gates)

        return logits, gates


class SoftmaxRouter(nn.Module):
    """Competing softmax router (zero-sum gates)."""
    def __init__(self, input_dim, n_adapters, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, n_adapters)
        self.n_adapters = n_adapters

    def __call__(self, h, temperature=1.0, hard=False, straight_through=False):
        z = nn.gelu(self.proj(h))
        logits = self.gate(z)

        if hard:
            gates = mx.softmax(logits / 0.01, axis=-1)  # near-hard
        else:
            u = mx.random.uniform(shape=logits.shape)
            u = mx.clip(u, 1e-6, 1.0 - 1e-6)
            gumbel_noise = -mx.log(-mx.log(u))
            gates = mx.softmax((logits + gumbel_noise) / temperature, axis=-1)

            if straight_through:
                hard_gates = mx.zeros_like(gates)
                top_idx = mx.argmax(gates, axis=-1, keepdims=True)
                hard_gates = hard_gates.at[mx.arange(gates.shape[0])[:, None], top_idx].add(1.0)
                gates = gates + mx.stop_gradient(hard_gates - gates)

        return logits, gates


# ===========================================================================
# Router training with configurable loss
# ===========================================================================
@dataclass
class RouterConfig:
    name: str
    gate_type: str = "sigmoid"  # "sigmoid" or "softmax"
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    fixed_temperature: Optional[float] = None  # if set, overrides start/end
    top_k: int = 2
    straight_through: bool = False
    load_balance_alpha: float = 0.0  # 0 = no load balancing
    n_steps: int = 3000  # matches N=50 baseline
    lr: float = 1e-3     # matches N=50 baseline
    hidden_dim: int = 256  # matches N=50 baseline
    seed: int = 42


def train_and_evaluate_router(config: RouterConfig, train_hiddens, val_hiddens,
                               domain_names, n_adapters):
    """Train a router with given config and return metrics."""
    mx.random.seed(config.seed)
    rng = random.Random(config.seed)

    d = train_hiddens[domain_names[0]].shape[1]
    N = n_adapters

    # Create router
    if config.gate_type == "sigmoid":
        router = SigmoidRouter(d, N, hidden_dim=config.hidden_dim)
    else:
        router = SoftmaxRouter(d, N, hidden_dim=config.hidden_dim)

    mx.eval(router.parameters())
    optimizer = opt.Adam(learning_rate=config.lr)

    # Build domain -> index mapping
    domain_to_idx = {name: i for i, name in enumerate(domain_names)}
    domains_with_data = [n for n in domain_names if n in train_hiddens]

    # Loss function
    def router_loss_fn(router, h_batch, target_idx, temperature):
        logits, gates = router(h_batch, temperature=temperature,
                                straight_through=config.straight_through)

        if config.gate_type == "sigmoid":
            # Binary cross-entropy per adapter gate
            target = mx.zeros((h_batch.shape[0], N))
            target = target.at[:, target_idx].add(1.0)
            bce = -(target * mx.log(gates + 1e-8) + (1 - target) * mx.log(1 - gates + 1e-8))
            loss = mx.mean(bce)
        else:
            # Cross-entropy for softmax
            # target_idx is an int -- use negative log of the correct gate
            loss = -mx.mean(mx.log(gates[:, target_idx] + 1e-8))

        # L1 gate activation penalty (pushes total gate mass down)
        if config.load_balance_alpha > 0:
            # gate_means[i] = average activation of gate i across the batch
            gate_means = mx.mean(gates, axis=0)  # [N]
            # L1 penalty: sum of mean activations = total gate mass / N
            # This penalizes non-target gates being active (combined with BCE
            # which raises the target gate, the net effect is: raise target,
            # suppress everything else). Prevents expert collapse where a few
            # dominant experts absorb all routing probability.
            lb_loss = mx.sum(gate_means)  # L1: penalize total activation
            loss = loss + config.load_balance_alpha * lb_loss

        return loss

    loss_and_grad = nn.value_and_grad(router, router_loss_fn)
    losses = []

    gc.disable()
    for step in range(config.n_steps):
        # Sample domain
        name = rng.choice(domains_with_data)
        h_all = train_hiddens[name]
        target_idx = domain_to_idx[name]

        # Sample batch
        n_samples = h_all.shape[0]
        batch_idx = rng.randint(0, n_samples - 1)
        h_batch = h_all[batch_idx:batch_idx + 1]

        # Temperature
        if config.fixed_temperature is not None:
            temperature = config.fixed_temperature
        else:
            progress = step / max(config.n_steps - 1, 1)
            temperature = config.temperature_start + (config.temperature_end - config.temperature_start) * progress

        loss, grads = loss_and_grad(router, h_batch, target_idx, temperature)
        optimizer.update(router, grads)
        del grads
        mx.eval(router.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)
    gc.enable()
    gc.collect()

    # Evaluate on validation set
    eval_results = evaluate_router(router, val_hiddens, domain_names, config.top_k)

    return {
        "config": config.name,
        "gate_type": config.gate_type,
        "top_k": config.top_k,
        "final_loss": round(losses[-1], 4) if losses else None,
        "last_100_avg_loss": round(sum(losses[-100:]) / len(losses[-100:]), 4) if len(losses) >= 100 else None,
        **eval_results,
    }


def evaluate_router(router, val_hiddens, domain_names, top_k=2):
    """Evaluate router accuracy on validation hidden states."""
    domain_to_idx = {name: i for i, name in enumerate(domain_names)}
    correct_top1 = 0
    correct_topk = 0
    total = 0
    per_domain = {}

    for name in domain_names:
        if name not in val_hiddens:
            continue
        idx = domain_to_idx[name]
        h_val = val_hiddens[name]

        logits, _ = router(h_val, hard=True)
        mx.eval(logits)

        n_samples = h_val.shape[0]
        d_top1 = 0
        d_topk = 0

        for s in range(n_samples):
            sample_logits = logits[s:s+1]

            top1 = mx.argmax(sample_logits, axis=-1).item()
            if top1 == idx:
                d_top1 += 1
                correct_top1 += 1

            topk_indices = mx.argsort(sample_logits, axis=-1)[:, -top_k:]
            mx.eval(topk_indices)
            topk_list = topk_indices[0].tolist()
            if idx in topk_list:
                d_topk += 1
                correct_topk += 1

            total += 1

        per_domain[name] = {
            "top1": round(d_top1 / n_samples, 4) if n_samples > 0 else 0,
            "topk": round(d_topk / n_samples, 4) if n_samples > 0 else 0,
            "n": n_samples,
        }

        del logits

    top1_acc = correct_top1 / total if total > 0 else 0
    topk_acc = correct_topk / total if total > 0 else 0

    # Count zero-accuracy domains
    zero_acc_domains = [n for n, v in per_domain.items() if v["topk"] == 0]

    return {
        "top1_accuracy": round(top1_acc, 4),
        "topk_accuracy": round(topk_acc, 4),
        "n_zero_accuracy_domains": len(zero_acc_domains),
        "zero_accuracy_domains": zero_acc_domains,
        "total_samples": total,
        "per_domain": per_domain,
    }


# ===========================================================================
# Phase 1: Extract hidden states
# ===========================================================================
def phase_extract_hidden_states():
    """Extract mean-pooled hidden states from base model for all domains.
    Returns train/val hidden state dicts saved to disk for reuse."""
    log("\n[Phase 1] Extracting hidden states from base model...")
    t0 = time.time()

    # Check if cached
    train_cache = EXPERIMENT_DIR / "train_hiddens.npz"
    val_cache = EXPERIMENT_DIR / "val_hiddens.npz"
    if train_cache.exists() and val_cache.exists():
        log("  Loading cached hidden states...")
        train_data = dict(mx.load(str(train_cache)))
        val_data = dict(mx.load(str(val_cache)))
        log(f"  Loaded {len(train_data)} train, {len(val_data)} val domains from cache")
        log(f"  Phase 1 time: {time.time() - t0:.1f}s")
        return train_data, val_data

    from mlx_lm import load

    log("  Loading model and tokenizer...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.eval(model.parameters())
    log_memory("post-load")

    domain_names = get_all_domain_names()
    log(f"  Processing {len(domain_names)} domains...")

    train_hiddens = {}
    val_hiddens = {}

    for name in domain_names:
        data_dir = get_data_dir(name)
        if data_dir is None:
            continue

        for split, target_dict, max_samples in [
            ("train", train_hiddens, 20),
            ("valid", val_hiddens, 10),
        ]:
            fpath = data_dir / f"{split}.jsonl"
            if not fpath.exists():
                continue

            texts = []
            with open(fpath) as f:
                for line in f:
                    texts.append(json.loads(line)["text"])

            hiddens = []
            for text in texts[:max_samples]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = model.model(x)  # [1, seq_len, d]
                h_mean = mx.mean(h, axis=1)  # [1, d]
                mx.eval(h_mean)
                hiddens.append(h_mean)
                del h, x

            if hiddens:
                target_dict[name] = mx.concatenate(hiddens, axis=0)
                mx.eval(target_dict[name])
            del hiddens

        if name in train_hiddens:
            log(f"    {name}: train={train_hiddens[name].shape[0]}, val={val_hiddens.get(name, mx.zeros((0,))).shape[0]}")

    # Save to disk
    log("  Saving hidden states to disk...")
    mx.savez(str(train_cache), **train_hiddens)
    mx.savez(str(val_cache), **val_hiddens)

    elapsed = time.time() - t0
    log(f"  Phase 1 complete: {len(train_hiddens)} domains, {elapsed:.1f}s")
    log_memory("post-extract")

    # Cleanup model
    cleanup(model, tokenizer)
    return train_hiddens, val_hiddens


# ===========================================================================
# Phase 2: Ablation sweep
# ===========================================================================
def phase_ablation_sweep(train_hiddens, val_hiddens):
    """Run all ablation configurations."""
    log("\n[Phase 2] Running ablation sweep...")
    t0 = time.time()

    domain_names = get_all_domain_names()
    N = len(domain_names)
    log(f"  {N} adapters, {len(train_hiddens)} domains with data")

    # Define configurations
    configs = []

    # === Baseline ===
    configs.append(RouterConfig(
        name="baseline_sigmoid_anneal2to0.5_k2",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, straight_through=False, load_balance_alpha=0.0,
    ))

    # === Temperature sweep (fixed) ===
    for tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
        configs.append(RouterConfig(
            name=f"sigmoid_fixed_tau{tau}_k2",
            gate_type="sigmoid", fixed_temperature=tau,
            top_k=2,
        ))

    # === Temperature anneal variants ===
    configs.append(RouterConfig(
        name="sigmoid_anneal1to0.1_k2",
        gate_type="sigmoid", temperature_start=1.0, temperature_end=0.1,
        top_k=2,
    ))
    configs.append(RouterConfig(
        name="sigmoid_anneal5to0.5_k2",
        gate_type="sigmoid", temperature_start=5.0, temperature_end=0.5,
        top_k=2,
    ))

    # === Top-k sweep ===
    for k in [1, 3, 4]:
        configs.append(RouterConfig(
            name=f"sigmoid_anneal2to0.5_k{k}",
            gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
            top_k=k,
        ))

    # === Softmax (competing) ===
    configs.append(RouterConfig(
        name="softmax_anneal2to0.5_k2",
        gate_type="softmax", temperature_start=2.0, temperature_end=0.5,
        top_k=2,
    ))
    configs.append(RouterConfig(
        name="softmax_anneal1to0.1_k2",
        gate_type="softmax", temperature_start=1.0, temperature_end=0.1,
        top_k=2,
    ))

    # === Load-balancing loss ===
    for alpha in [0.01, 0.1, 0.5]:
        configs.append(RouterConfig(
            name=f"sigmoid_anneal2to0.5_k2_lb{alpha}",
            gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
            top_k=2, load_balance_alpha=alpha,
        ))

    # === Straight-through Gumbel ===
    configs.append(RouterConfig(
        name="sigmoid_anneal2to0.5_k2_st",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, straight_through=True,
    ))

    # === Combined: load-balance + anneal 1->0.1 ===
    configs.append(RouterConfig(
        name="sigmoid_anneal1to0.1_k2_lb0.1",
        gate_type="sigmoid", temperature_start=1.0, temperature_end=0.1,
        top_k=2, load_balance_alpha=0.1,
    ))

    # === More training steps for load-balanced configs ===
    configs.append(RouterConfig(
        name="sigmoid_anneal2to0.5_k2_lb0.1_6000steps",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, load_balance_alpha=0.1, n_steps=6000,
    ))

    # === Control: 6000 steps WITHOUT load-balancing (isolate training length effect) ===
    configs.append(RouterConfig(
        name="baseline_6000steps_no_lb",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, load_balance_alpha=0.0, n_steps=6000,
    ))

    # === Narrower hidden dim (128 vs 256 baseline) ===
    configs.append(RouterConfig(
        name="sigmoid_anneal2to0.5_k2_h128",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, hidden_dim=128,
    ))

    # === Higher learning rate ===
    configs.append(RouterConfig(
        name="sigmoid_anneal2to0.5_k2_lr3e-3",
        gate_type="sigmoid", temperature_start=2.0, temperature_end=0.5,
        top_k=2, lr=3e-3,
    ))

    all_results = []
    for i, config in enumerate(configs):
        log(f"\n  [{i+1}/{len(configs)}] Config: {config.name}")
        result = train_and_evaluate_router(
            config, train_hiddens, val_hiddens, domain_names, N
        )
        log(f"    top1={result['top1_accuracy']:.4f} topk={result['topk_accuracy']:.4f} "
            f"zero_acc={result['n_zero_accuracy_domains']} loss={result.get('last_100_avg_loss', '?')}")

        # Show zero-accuracy domains if any
        if result['zero_accuracy_domains']:
            log(f"    Zero-acc domains: {result['zero_accuracy_domains']}")

        all_results.append(result)

        # Cleanup between configs
        gc.collect()
        mx.clear_cache()

    elapsed = time.time() - t0
    log(f"\n  Phase 2 complete: {len(configs)} configs, {elapsed:.1f}s")
    return all_results


# ===========================================================================
# Phase 3: Analyze zero-accuracy domains
# ===========================================================================
def phase_analyze_zero_acc(train_hiddens, val_hiddens):
    """Analyze WHY chemistry, wikitext, dialogue, debate get 0% accuracy.
    Compute pairwise cosine similarity of domain centroids."""
    log("\n[Phase 3] Analyzing zero-accuracy domains...")
    t0 = time.time()

    domain_names = get_all_domain_names()
    zero_acc = ["chemistry", "wikitext", "dialogue", "debate"]

    # Compute centroids
    centroids = {}
    for name in domain_names:
        if name in train_hiddens:
            centroids[name] = mx.mean(train_hiddens[name], axis=0)  # [d]
            mx.eval(centroids[name])

    # For each zero-acc domain, find its nearest neighbors
    analysis = {}
    for name in zero_acc:
        if name not in centroids:
            continue

        c = centroids[name]
        c_norm = c / (mx.sqrt(mx.sum(c * c)) + 1e-8)
        mx.eval(c_norm)

        similarities = {}
        for other_name, other_c in centroids.items():
            if other_name == name:
                continue
            other_norm = other_c / (mx.sqrt(mx.sum(other_c * other_c)) + 1e-8)
            sim = mx.sum(c_norm * other_norm).item()
            similarities[other_name] = round(sim, 4)

        # Sort by similarity descending
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_sims[:5]

        analysis[name] = {
            "top5_similar": top5,
            "n_train_samples": train_hiddens[name].shape[0] if name in train_hiddens else 0,
            "n_val_samples": val_hiddens[name].shape[0] if name in val_hiddens else 0,
        }
        log(f"  {name}: nearest = {top5[:3]}")

    # Also compute intra-domain variance for zero-acc vs high-acc domains
    high_acc = ["sql", "javascript", "cooking", "health", "reasoning"]
    variance_comparison = {}
    for name in zero_acc + high_acc:
        if name not in train_hiddens:
            continue
        h = train_hiddens[name]
        mean = mx.mean(h, axis=0, keepdims=True)
        diffs = h - mean
        var = mx.mean(mx.sum(diffs * diffs, axis=1)).item()
        variance_comparison[name] = round(var, 4)

    analysis["variance_comparison"] = variance_comparison

    elapsed = time.time() - t0
    log(f"  Phase 3 complete: {elapsed:.1f}s")
    return analysis


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log("=" * 60)
    log("Gumbel-Sigmoid Routing Ablation Study")
    log("=" * 60)
    log_memory("start")

    # Phase 1: Extract hidden states (expensive, cached)
    train_hiddens, val_hiddens = phase_extract_hidden_states()
    log_memory("after-extract")

    # Phase 2: Ablation sweep (cheap -- just router training)
    ablation_results = phase_ablation_sweep(train_hiddens, val_hiddens)
    log_memory("after-ablation")

    # Phase 3: Analyze zero-accuracy domains
    zero_acc_analysis = phase_analyze_zero_acc(train_hiddens, val_hiddens)
    log_memory("after-analysis")

    # Compile results
    # Sort ablation by topk accuracy descending
    ablation_sorted = sorted(ablation_results, key=lambda x: x["topk_accuracy"], reverse=True)

    # Baseline comparison: use the N=50 original result (86.33%) as the K1 reference,
    # NOT the in-experiment baseline, which may differ due to hidden state caching/seed.
    baseline = next((r for r in ablation_results if r["config"] == "baseline_sigmoid_anneal2to0.5_k2"), None)
    in_experiment_baseline = baseline["topk_accuracy"] if baseline else 0.8633
    baseline_topk = 0.8633  # N=50 original result -- canonical K1 reference

    # Summary
    best = ablation_sorted[0]
    improvement = best["topk_accuracy"] - baseline_topk

    summary = {
        "baseline_topk_accuracy": baseline_topk,
        "in_experiment_baseline_topk": in_experiment_baseline,
        "best_config": best["config"],
        "best_topk_accuracy": best["topk_accuracy"],
        "best_top1_accuracy": best["top1_accuracy"],
        "best_zero_acc_domains": best["n_zero_accuracy_domains"],
        "improvement_over_baseline": round(improvement, 4),
        "k1_threshold": 0.05,  # >5% improvement needed
        "k1_result": "PASS" if improvement > 0.05 else "FAIL",
    }

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Baseline: topk={baseline_topk:.4f}")
    log(f"  Best: {best['config']} topk={best['topk_accuracy']:.4f}")
    log(f"  Improvement: {improvement:.4f} ({'PASS' if improvement > 0.05 else 'FAIL'} K1 threshold 0.05)")
    log(f"  Best zero-acc domains: {best['n_zero_accuracy_domains']}")

    # Show top 5 configs
    log("\n  Top 5 configs by topk accuracy:")
    for r in ablation_sorted[:5]:
        log(f"    {r['config']}: topk={r['topk_accuracy']:.4f} top1={r['top1_accuracy']:.4f} zero={r['n_zero_accuracy_domains']}")

    # Show zero-acc domain analysis
    log("\n  Zero-accuracy domain analysis:")
    for name in ["chemistry", "wikitext", "dialogue", "debate"]:
        if name in zero_acc_analysis:
            a = zero_acc_analysis[name]
            log(f"    {name}: nearest={a['top5_similar'][:2]}, n_train={a['n_train_samples']}")

    # Build leaderboard (compact)
    leaderboard = []
    for r in ablation_sorted:
        leaderboard.append({
            "config": r["config"],
            "topk_accuracy": r["topk_accuracy"],
            "top1_accuracy": r["top1_accuracy"],
            "n_zero_acc": r["n_zero_accuracy_domains"],
            "zero_acc_domains": r["zero_accuracy_domains"],
            "final_loss": r.get("final_loss"),
        })

    results = {
        "experiment": "gumbel_sigmoid_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(time.time() - t_start, 1),
        "n_configs": len(ablation_results),
        "n_domains": len(get_all_domain_names()),
        "summary": summary,
        "leaderboard": leaderboard,
        "zero_acc_analysis": zero_acc_analysis,
        "full_results": ablation_results,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
