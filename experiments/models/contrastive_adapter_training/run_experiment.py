#!/usr/bin/env python3
"""Contrastive Adapter Training: LoRACLR orthogonal specialization for 5 domains.

Kill criteria:
  K617: Contrastive-trained adapters still show code adapter as universal best
        (alpha > 0.9 on 4+ domains) -> KILL
  K618: Training diverges or adapters fail to converge (loss > 2x baseline) -> KILL
  K619: Contrastive adapters degrade ALL domains vs base (worse than current SFT) -> KILL

Type: Guided exploration (Type 2)
Platform: Apple M5 Pro 48GB, MLX

Approach:
  1. Train 5 domain adapters WITH contrastive orthogonality loss (lambda=1.0)
  2. Train 5 domain adapters WITHOUT contrastive loss (baseline, lambda=0)
  3. Compare inter-adapter cosine similarity, domain-specific performance, and
     whether code adapter remains universal best
  4. Evaluate via PPL on domain data + standardized benchmarks (MMLU subset, GSM8K)
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"

# Source data from real_data_domain_experts (instruction-formatted)
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 2.0  # LOW scale per Finding #212 (20.0 destroys capability)
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
CONTRASTIVE_LAMBDA = 1.0  # Weight for contrastive orthogonality loss
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Target layers for LoRA
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

RESPONSE_MARKER = "### Response:\n"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


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
# Model utilities
# ============================================================================

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx_lm.tuner.lora import LoRALinear


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


def apply_lora_to_model(model, rank=16, scale=2.0):
    """Apply LoRA wrappers to all target linear layers."""
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in set(TARGET_KEYS) and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(
                    module, r=rank, scale=scale, dropout=0.0
                )
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}, scale={scale}) to {count} layers")
    return model


def get_lora_params(model):
    """Extract LoRA parameters as a flat dict."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora_params(model):
    """Reset all LoRA params."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    """Apply adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def save_adapter(params, path):
    """Save LoRA adapter weights."""
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path):
    """Load adapter weights from disk."""
    return dict(mx.load(str(path / "adapter.npz")))


# ============================================================================
# SFT data loading
# ============================================================================

def load_sft_data(domain):
    """Load instruction-formatted data for domain."""
    train_path = DATA_DIR / domain / "train.jsonl"
    val_path = DATA_DIR / domain / "valid.jsonl"

    def load_jsonl(path, max_n=400):
        texts = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_n:
                    break
                texts.append(json.loads(line)["text"])
        return texts

    train_texts = load_jsonl(train_path, 400)
    val_texts = load_jsonl(val_path, 50)
    return train_texts, val_texts


def tokenize_with_sft_mask(text, tokenizer, max_len=256):
    """Tokenize and return (tokens, loss_mask). mask=1 for response tokens only."""
    response_idx = text.find(RESPONSE_MARKER)
    if response_idx < 0:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        return tokens, [1] * len(tokens)

    instruction_part = text[:response_idx + len(RESPONSE_MARKER)]
    instruction_tokens = tokenizer.encode(instruction_part, add_special_tokens=True)
    instruction_len = len(instruction_tokens)

    full_tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(full_tokens) > max_len:
        full_tokens = full_tokens[:max_len]

    mask = [0] * min(instruction_len, len(full_tokens))
    mask += [1] * (len(full_tokens) - len(mask))
    return full_tokens, mask


def prepare_batches(texts, tokenizer, max_len=256):
    """Prepare tokenized batches with SFT masking."""
    batches = []
    for text in texts:
        tokens, mask = tokenize_with_sft_mask(text, tokenizer, max_len)
        if len(tokens) >= 4:
            batches.append((tokens, mask))
    return batches


# ============================================================================
# Loss functions
# ============================================================================

def sft_loss_fn(model, tokens, mask):
    """Cross-entropy loss ONLY on response tokens."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    response_mask = mask[:, 1:]
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    return masked_loss.sum() / n_response


# ============================================================================
# Contrastive Orthogonality Loss
# ============================================================================

def compute_adapter_cosine_matrix(adapter_params_list):
    """Compute NxN cosine similarity matrix between adapter weight deltas.

    Each adapter's parameters are flattened into a single vector.
    Returns NxN matrix of absolute cosine similarities.
    """
    N = len(adapter_params_list)
    # Flatten each adapter into a single vector
    flat_vecs = []
    for params in adapter_params_list:
        # Only use lora_b params (lora_a is shared/random, lora_b is trained)
        vec_parts = []
        for key in sorted(params.keys()):
            if "lora_b" in key:
                vec_parts.append(params[key].reshape(-1))
        flat_vec = mx.concatenate(vec_parts)
        flat_vecs.append(flat_vec)

    # Compute cosine similarity matrix
    cos_matrix = []
    for i in range(N):
        row = []
        norm_i = mx.sqrt(mx.sum(flat_vecs[i] ** 2) + 1e-8)
        for j in range(N):
            if i == j:
                row.append(1.0)
            else:
                norm_j = mx.sqrt(mx.sum(flat_vecs[j] ** 2) + 1e-8)
                cos = mx.sum(flat_vecs[i] * flat_vecs[j]) / (norm_i * norm_j)
                row.append(mx.abs(cos).item())
        cos_matrix.append(row)
    return cos_matrix


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_ppl(model, tokenizer, val_texts, max_batches=25):
    """Compute perplexity on validation texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in val_texts[:max_batches]:
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


# ============================================================================
# Phase 1: Train adapters (with or without contrastive loss)
# ============================================================================

def phase_train_adapters(contrastive_lambda, condition_name):
    """Train 5 domain adapters with optional contrastive loss.

    Training approach: Round-robin through domains. Each step:
    1. Pick a domain
    2. Load that domain's adapter params into model
    3. Forward pass + SFT loss on that domain's data
    4. Add contrastive penalty (if lambda > 0) using stored params from other adapters
    5. Update
    6. Save updated params back

    This is simpler than true joint training but captures the contrastive signal.
    """
    log(f"\n{'='*70}")
    log(f"Training condition: {condition_name} (lambda={contrastive_lambda})")
    log(f"{'='*70}")

    # Load model
    log("Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("model-loaded")

    # Apply LoRA
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")

    # Prepare data for all domains
    all_batches = {}
    all_val_texts = {}
    for domain in DOMAINS:
        train_texts, val_texts = load_sft_data(domain)
        batches = prepare_batches(train_texts, tokenizer)
        all_batches[domain] = batches
        all_val_texts[domain] = val_texts
        log(f"  {domain}: {len(batches)} train batches, {len(val_texts)} val texts")

    # Initialize separate adapter params for each domain (all start from zero)
    domain_adapter_params = {}
    for domain in DOMAINS:
        zero_lora_params(model)
        domain_adapter_params[domain] = get_lora_params(model)

    # Training loop: round-robin through domains
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, sft_loss_fn)

    domain_losses = {d: [] for d in DOMAINS}
    t_start = time.time()

    log(f"\n  Training {TRAIN_ITERS} steps per domain...")

    gc.disable()  # Per CODING_GUIDELINES: disable GC in tight loops

    for step in range(TRAIN_ITERS * N_DOMAINS):
        domain_idx = step % N_DOMAINS
        domain = DOMAINS[domain_idx]
        batch_idx = step // N_DOMAINS

        # Load this domain's adapter params into model
        apply_adapter_weights(model, domain_adapter_params[domain])

        # Get batch
        batches = all_batches[domain]
        tokens, mask = batches[batch_idx % len(batches)]
        tokens_mx = mx.array([tokens])
        mask_mx = mx.array([mask])

        # Forward pass + gradient
        loss, grads = loss_and_grad(model, tokens_mx, mask_mx)

        # If contrastive loss is active, add penalty to gradients
        # We approximate the contrastive gradient by adding a penalty that
        # pushes this adapter's B-matrices away from other adapters' B-matrices
        if contrastive_lambda > 0 and batch_idx > 10:
            # Every 5 steps, compute contrastive penalty
            if batch_idx % 5 == 0:
                current_params = get_lora_params(model)
                # Compute contrastive gradient: for each lora_b param,
                # add gradient pushing away from other domains' lora_b
                contrastive_updates = {}
                for key in current_params:
                    if "lora_b" not in key:
                        continue
                    current_b = current_params[key]
                    current_norm = mx.sqrt(mx.sum(current_b ** 2) + 1e-8)

                    # Sum of gradients pushing away from other domains
                    push_grad = mx.zeros_like(current_b)
                    for other_domain in DOMAINS:
                        if other_domain == domain:
                            continue
                        other_b = domain_adapter_params[other_domain][key]
                        other_norm = mx.sqrt(mx.sum(other_b ** 2) + 1e-8)

                        # Gradient of |cos(a,b)|^2 w.r.t. a
                        # = 2 * cos(a,b) * (b/||b|| - cos(a,b) * a/||a||) / ||a||
                        cos_val = mx.sum(current_b * other_b) / (current_norm * other_norm + 1e-8)
                        grad_cos = (other_b / (other_norm + 1e-8) - cos_val * current_b / (current_norm + 1e-8)) / (current_norm + 1e-8)
                        push_grad = push_grad + 2.0 * cos_val * grad_cos

                    # Scale by lambda and N-1 pairs
                    contrastive_updates[key] = push_grad * contrastive_lambda / (N_DOMAINS - 1)

                # Add contrastive gradient to existing grads
                # grads is a nested dict matching model structure
                # We need to manually add the contrastive updates
                for key, contra_grad in contrastive_updates.items():
                    # Walk the grad tree to find the matching parameter
                    parts = key.split(".")
                    node = grads
                    for part in parts[:-1]:
                        if isinstance(node, dict):
                            node = node.get(part, None)
                        elif isinstance(node, list):
                            node = node[int(part)]
                        else:
                            node = getattr(node, part, None)
                        if node is None:
                            break
                    if node is not None and isinstance(node, dict):
                        last_key = parts[-1]
                        if last_key in node:
                            node[last_key] = node[last_key] + contra_grad

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        domain_losses[domain].append(loss_val)

        # Save updated params back
        domain_adapter_params[domain] = get_lora_params(model)

        if (step + 1) % (N_DOMAINS * 50) == 0:
            avg_losses = {d: sum(domain_losses[d][-50:]) / max(len(domain_losses[d][-50:]), 1)
                         for d in DOMAINS}
            log(f"    Step {step+1}/{TRAIN_ITERS*N_DOMAINS}: " +
                " | ".join(f"{d[:3]}={avg_losses[d]:.3f}" for d in DOMAINS))

    gc.enable()
    gc.collect()

    train_time = time.time() - t_start
    log(f"  Training done in {train_time:.0f}s")

    # Save adapters
    save_dir = ADAPTERS_DIR / condition_name
    train_results = {}
    for domain in DOMAINS:
        save_adapter(domain_adapter_params[domain], save_dir / domain)
        losses = domain_losses[domain]
        first_50 = sum(losses[:50]) / max(len(losses[:50]), 1)
        last_50 = sum(losses[-50:]) / max(len(losses[-50:]), 1)
        train_results[domain] = {
            "first_50_avg_loss": round(first_50, 4),
            "last_50_avg_loss": round(last_50, 4),
            "converged": last_50 < first_50 * 0.95,
            "n_steps": len(losses),
        }
        log(f"  {domain}: loss {first_50:.4f} -> {last_50:.4f} "
            f"({'ok' if train_results[domain]['converged'] else 'NOT converged'})")

    # Compute inter-adapter cosine similarities
    cos_matrix = compute_adapter_cosine_matrix(
        [domain_adapter_params[d] for d in DOMAINS]
    )

    # Evaluate PPL per domain
    log("\n  Evaluating PPL...")
    base_ppls = {}
    adapter_ppls = {}

    # Base PPL
    zero_lora_params(model)
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, all_val_texts[domain])
        base_ppls[domain] = ppl

    # Per-adapter PPL on own domain and code domain
    domain_vs_code = {}
    for domain in DOMAINS:
        apply_adapter_weights(model, domain_adapter_params[domain])
        mx.eval(model.parameters())

        own_ppl = compute_ppl(model, tokenizer, all_val_texts[domain])
        adapter_ppls[domain] = {"own_domain": own_ppl}

        # Also measure code adapter on this domain
        if domain != "code":
            apply_adapter_weights(model, domain_adapter_params["code"])
            mx.eval(model.parameters())
            code_ppl = compute_ppl(model, tokenizer, all_val_texts[domain])
            domain_vs_code[domain] = {
                "domain_adapter_ppl": own_ppl,
                "code_adapter_ppl": code_ppl,
                "domain_wins": own_ppl < code_ppl,
                "improvement_pct": round((code_ppl - own_ppl) / code_ppl * 100, 1),
            }
            log(f"  {domain}: own={own_ppl:.2f} code={code_ppl:.2f} "
                f"{'DOMAIN WINS' if own_ppl < code_ppl else 'CODE WINS'} "
                f"({(code_ppl - own_ppl) / code_ppl * 100:+.1f}%)")

    # Cleanup
    log_memory("pre-cleanup")
    cleanup(model, tokenizer, optimizer)
    log_memory("post-cleanup")

    return {
        "train_results": train_results,
        "train_time_s": round(train_time, 1),
        "cosine_matrix": cos_matrix,
        "base_ppls": base_ppls,
        "adapter_ppls": adapter_ppls,
        "domain_vs_code": domain_vs_code,
        "domain_adapter_params": domain_adapter_params,  # Keep for cross-eval
    }


# ============================================================================
# Phase 2: Cross-evaluation (which adapter is best on each domain?)
# ============================================================================

def phase_cross_eval(condition_name):
    """Load saved adapters and evaluate each adapter on ALL domains."""
    log(f"\n{'='*70}")
    log(f"Cross-evaluation: {condition_name}")
    log(f"{'='*70}")

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    # Load val texts
    all_val_texts = {}
    for domain in DOMAINS:
        _, val_texts = load_sft_data(domain)
        all_val_texts[domain] = val_texts

    # Load adapters
    adapter_dir = ADAPTERS_DIR / condition_name
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(adapter_dir / domain)

    # Base PPL
    zero_lora_params(model)
    base_ppls = {}
    for domain in DOMAINS:
        base_ppls[domain] = compute_ppl(model, tokenizer, all_val_texts[domain])

    # NxN PPL matrix: adapter_i evaluated on domain_j
    ppl_matrix = {}
    for adapter_name in DOMAINS:
        apply_adapter_weights(model, adapters[adapter_name])
        mx.eval(model.parameters())

        ppl_matrix[adapter_name] = {}
        for eval_domain in DOMAINS:
            ppl = compute_ppl(model, tokenizer, all_val_texts[eval_domain])
            ppl_matrix[adapter_name][eval_domain] = ppl

        zero_lora_params(model)

    # Determine best adapter per domain
    best_per_domain = {}
    code_is_best_count = 0
    for eval_domain in DOMAINS:
        best_adapter = min(DOMAINS, key=lambda a: ppl_matrix[a][eval_domain])
        best_ppl = ppl_matrix[best_adapter][eval_domain]
        code_ppl = ppl_matrix["code"][eval_domain]

        # Alpha = how close code is to best (1.0 = code IS best)
        if best_ppl > 0:
            alpha = best_ppl / code_ppl  # < 1 means code is worse than best
        else:
            alpha = 1.0

        is_code_best = best_adapter == "code"
        if is_code_best:
            code_is_best_count += 1

        best_per_domain[eval_domain] = {
            "best_adapter": best_adapter,
            "best_ppl": round(best_ppl, 2),
            "code_ppl": round(code_ppl, 2),
            "alpha": round(alpha, 3),
            "code_is_best": is_code_best,
        }
        log(f"  {eval_domain}: best={best_adapter} (PPL={best_ppl:.2f}), "
            f"code={code_ppl:.2f}, alpha={alpha:.3f}")

    # Compute domain adapter improvement vs base
    adapter_vs_base = {}
    for domain in DOMAINS:
        own_ppl = ppl_matrix[domain][domain]
        base_ppl = base_ppls[domain]
        adapter_vs_base[domain] = {
            "adapter_ppl": round(own_ppl, 2),
            "base_ppl": round(base_ppl, 2),
            "improvement_pct": round((base_ppl - own_ppl) / base_ppl * 100, 1),
            "adapter_better": own_ppl < base_ppl,
        }

    cleanup(model, tokenizer)

    return {
        "ppl_matrix": {a: {d: round(v, 2) for d, v in row.items()}
                      for a, row in ppl_matrix.items()},
        "base_ppls": {d: round(v, 2) for d, v in base_ppls.items()},
        "best_per_domain": best_per_domain,
        "adapter_vs_base": adapter_vs_base,
        "code_is_best_count": code_is_best_count,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    mx.random.seed(SEED)
    t0 = time.time()
    log_memory("start")

    results = {
        "experiment": "contrastive_adapter_training",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "contrastive_lambda": CONTRASTIVE_LAMBDA,
        "train_iters": TRAIN_ITERS,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---------------------------------------------------------------
    # Condition A: Baseline (no contrastive loss)
    # ---------------------------------------------------------------
    baseline_results = phase_train_adapters(
        contrastive_lambda=0.0, condition_name="baseline"
    )
    results["baseline"] = {
        "train_results": baseline_results["train_results"],
        "train_time_s": baseline_results["train_time_s"],
        "cosine_matrix": baseline_results["cosine_matrix"],
        "base_ppls": {d: round(v, 2) for d, v in baseline_results["base_ppls"].items()},
        "adapter_ppls": baseline_results["adapter_ppls"],
        "domain_vs_code": baseline_results["domain_vs_code"],
    }

    # ---------------------------------------------------------------
    # Condition B: Contrastive (lambda = 1.0)
    # ---------------------------------------------------------------
    contrastive_results = phase_train_adapters(
        contrastive_lambda=CONTRASTIVE_LAMBDA, condition_name="contrastive"
    )
    results["contrastive"] = {
        "train_results": contrastive_results["train_results"],
        "train_time_s": contrastive_results["train_time_s"],
        "cosine_matrix": contrastive_results["cosine_matrix"],
        "base_ppls": {d: round(v, 2) for d, v in contrastive_results["base_ppls"].items()},
        "adapter_ppls": contrastive_results["adapter_ppls"],
        "domain_vs_code": contrastive_results["domain_vs_code"],
    }

    # ---------------------------------------------------------------
    # Cross-evaluation for both conditions
    # ---------------------------------------------------------------
    results["baseline_cross_eval"] = phase_cross_eval("baseline")
    results["contrastive_cross_eval"] = phase_cross_eval("contrastive")

    # ---------------------------------------------------------------
    # Kill criteria assessment
    # ---------------------------------------------------------------
    log(f"\n{'='*70}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'='*70}")

    # K617: Code adapter still universal best?
    contrastive_cross = results["contrastive_cross_eval"]
    code_best_count = contrastive_cross["code_is_best_count"]
    high_alpha_count = sum(
        1 for d, info in contrastive_cross["best_per_domain"].items()
        if info["alpha"] > 0.9
    )
    k617_pass = not (high_alpha_count >= 4)  # KILL if alpha > 0.9 on 4+ domains
    results["k617_pass"] = k617_pass
    results["k617_detail"] = {
        "code_is_best_on_n_domains": code_best_count,
        "high_alpha_count": high_alpha_count,
    }
    log(f"\n  K617: Code best on {code_best_count}/5 domains, "
        f"high-alpha on {high_alpha_count}/5 -> {'PASS' if k617_pass else 'KILL'}")

    # K618: Training diverges?
    contrastive_train = results["contrastive"]["train_results"]
    baseline_train = results["baseline"]["train_results"]
    diverged = False
    for domain in DOMAINS:
        c_loss = contrastive_train[domain]["last_50_avg_loss"]
        b_loss = baseline_train[domain]["last_50_avg_loss"]
        if c_loss > 2 * b_loss:
            diverged = True
            log(f"  K618: {domain} contrastive loss {c_loss:.4f} > 2x baseline {b_loss:.4f}")
    k618_pass = not diverged
    results["k618_pass"] = k618_pass
    log(f"  K618: Training convergence -> {'PASS' if k618_pass else 'KILL'}")

    # K619: ALL domains worse than base?
    contrastive_adapter_vs_base = contrastive_cross["adapter_vs_base"]
    domains_better = sum(1 for d in DOMAINS if contrastive_adapter_vs_base[d]["adapter_better"])
    k619_pass = domains_better > 0  # At least one domain improves
    results["k619_pass"] = k619_pass
    results["k619_detail"] = {
        "domains_better_than_base": domains_better,
        "per_domain": contrastive_adapter_vs_base,
    }
    log(f"  K619: {domains_better}/5 domains improve vs base -> "
        f"{'PASS' if k619_pass else 'KILL'}")

    # Hypothesis check: domain adapter beats code by >= 15%
    domain_beats_code_count = 0
    for domain in DOMAINS:
        if domain == "code":
            continue
        info = contrastive_cross["best_per_domain"][domain]
        domain_ppl = contrastive_cross["ppl_matrix"][domain][domain]
        code_ppl = info["code_ppl"]
        improvement = (code_ppl - domain_ppl) / code_ppl * 100
        if improvement >= 15:
            domain_beats_code_count += 1
        log(f"  Hypothesis check {domain}: domain PPL={domain_ppl:.2f}, "
            f"code PPL={code_ppl:.2f}, improvement={improvement:+.1f}%")

    results["hypothesis_domain_beats_code_15pct"] = domain_beats_code_count
    log(f"\n  Hypothesis: {domain_beats_code_count}/4 domains beat code by >=15%")

    # Comparison: contrastive vs baseline cosine
    baseline_cos = results["baseline"]["cosine_matrix"]
    contrastive_cos = results["contrastive"]["cosine_matrix"]

    def avg_off_diagonal(matrix):
        n = len(matrix)
        total = 0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += abs(matrix[i][j])
                    count += 1
        return total / max(count, 1)

    baseline_avg_cos = avg_off_diagonal(baseline_cos)
    contrastive_avg_cos = avg_off_diagonal(contrastive_cos)
    results["cosine_comparison"] = {
        "baseline_avg_cos": round(baseline_avg_cos, 4),
        "contrastive_avg_cos": round(contrastive_avg_cos, 4),
        "reduction_pct": round((baseline_avg_cos - contrastive_avg_cos) / max(baseline_avg_cos, 1e-8) * 100, 1),
    }
    log(f"\n  Cosine similarity: baseline={baseline_avg_cos:.4f}, "
        f"contrastive={contrastive_avg_cos:.4f} "
        f"({(baseline_avg_cos - contrastive_avg_cos) / max(baseline_avg_cos, 1e-8) * 100:+.1f}% reduction)")

    # Summary
    results["total_time_s"] = round(time.time() - t0, 1)
    results["overall_verdict"] = (
        "PASS" if (k617_pass and k618_pass and k619_pass) else "KILL"
    )

    log(f"\n  OVERALL: {results['overall_verdict']}")
    log(f"  Total time: {results['total_time_s']:.0f}s")

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
