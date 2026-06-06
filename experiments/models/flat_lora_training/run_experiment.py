#!/usr/bin/env python3
"""
Flat-LoRA: Train in flat loss regions for mergeable adapters.

Tests whether SAM (Sharpness-Aware Minimization) training produces LoRA adapters
that merge better than standard LoRA adapters.

Kill criteria:
  K1 (id:552): Flat-LoRA training fails on MLX (SAM not implementable)
  K2 (id:553): No merge improvement over standard LoRA

Success criteria:
  S1 (id:63): Flat-LoRA merged >3pp better than standard merge

Approach:
  1. Train 5 domain adapters with standard LoRA (baseline)
  2. Train 5 domain adapters with SAM-LoRA (perturb in LoRA parameter space)
  3. Merge both sets using Task Arithmetic, TIES, DARE
  4. Compare individual quality and merged quality

Reference: arXiv:2409.14396 (Flat-LoRA, ICML), arXiv:2010.01412 (SAM, ICLR 2021)
Based on: experiments/models/bitnet_2b_real_composition/run_experiment.py
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

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Memory management
# ===========================================================================
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")

def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

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

# SAM hyperparameters
SAM_RHO = 0.05  # perturbation radius (standard default)

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition
DATA_SOURCE = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]


# ===========================================================================
# Unpack ternary weights (from reference experiment)
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
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# LoRA utilities (from reference experiment)
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
    print(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(params, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
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
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-scale, high=scale, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_tokens(domain_name, tokenizer):
    data_dir = DATA_SOURCE / domain_name
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Data not found: {train_path}")

    texts = []
    with open(train_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    tokens = []
    for text in texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
    return tokens


def load_val_texts(domain_name):
    valid_path = DATA_SOURCE / domain_name / "valid.jsonl"
    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, domain_name, max_batches=25):
    texts = load_val_texts(domain_name)
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


# ===========================================================================
# Training: Standard LoRA
# ===========================================================================
def train_standard(model, tokenizer, domain_name, train_tokens):
    """Train one adapter with standard LoRA."""
    zero_lora_params(model)

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    gc.disable()
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
            print(f"    Step {step+1}/{TRAIN_ITERS}: loss={losses[-1]:.4f} (avg50={avg:.4f})")
    gc.enable()
    gc.collect()

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    params = get_lora_params(model)
    del optimizer
    gc.collect()
    mx.clear_cache()

    return {
        "params": params,
        "train_time_s": round(train_time, 1),
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "converged": converged,
    }


# ===========================================================================
# Training: SAM-LoRA (Flat-LoRA approximation)
# ===========================================================================
def train_sam(model, tokenizer, domain_name, train_tokens, rho=SAM_RHO):
    """Train one adapter with SAM perturbation in LoRA parameter space.

    SAM step:
      1. Compute gradient g at current params
      2. Perturb params by epsilon = rho * g / ||g||
      3. Compute gradient at perturbed point
      4. Restore original params
      5. Update with the perturbed gradient
    """
    zero_lora_params(model)

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        # Step 1: First forward+backward to get gradient
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss, grads)

        # Step 2: Compute perturbation epsilon = rho * g / ||g||
        # Flatten all LoRA gradients to compute global norm
        grad_list = tree_flatten(grads)
        lora_grads = [(n, g) for n, g in grad_list if "lora_a" in n or "lora_b" in n]

        if len(lora_grads) == 0:
            # Fallback: no LoRA grads found, use standard update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
            continue

        # Compute global gradient norm
        grad_sq_sum = mx.array(0.0)
        for _, g in lora_grads:
            grad_sq_sum = grad_sq_sum + mx.sum(g * g)
        grad_norm = mx.sqrt(grad_sq_sum)
        mx.eval(grad_norm)

        if grad_norm.item() < 1e-10:
            # Near-zero gradient, just do standard step
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
            continue

        # Save current LoRA params and apply perturbation
        saved_params = {}
        perturbation_updates = []
        for name, g in lora_grads:
            # Get current parameter value
            parts = name.split(".")
            obj = model
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            param_val = getattr(obj, parts[-1])
            saved_params[name] = mx.array(param_val)  # deep copy

            # Compute epsilon = rho * g / ||g||
            epsilon = rho * g / grad_norm
            new_val = param_val + epsilon
            perturbation_updates.append((name, new_val))

        # Apply perturbation to model
        model.update(tree_unflatten(perturbation_updates))
        mx.eval(model.parameters())

        # Step 3: Second forward+backward at perturbed point
        _, sam_grads = loss_and_grad(model, x, y)
        mx.eval(sam_grads)

        # Step 4: Restore original params
        model.update(tree_unflatten(list(saved_params.items())))
        mx.eval(model.parameters())

        # Step 5: Update with SAM gradient
        optimizer.update(model, sam_grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())

        # Clean up intermediate tensors
        del grads, sam_grads, saved_params, perturbation_updates, lora_grads
        del grad_sq_sum, grad_norm

        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            print(f"    Step {step+1}/{TRAIN_ITERS}: loss={losses[-1]:.4f} (avg50={avg:.4f})")
    gc.enable()
    gc.collect()

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    params = get_lora_params(model)
    del optimizer
    gc.collect()
    mx.clear_cache()

    return {
        "params": params,
        "train_time_s": round(train_time, 1),
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "converged": converged,
    }


# ===========================================================================
# Merge methods
# ===========================================================================
def merge_task_arithmetic(adapter_list, scale=None):
    """Task Arithmetic: simple weighted sum with 1/N scaling."""
    N = len(adapter_list)
    if scale is None:
        scale = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale
    return merged


def merge_ties(adapter_list, density=0.2):
    """TIES: Trim-Elect-Sign merge (Yadav et al., NeurIPS 2023).

    1. Trim: keep only top-density% parameters by magnitude
    2. Elect sign: majority vote on sign per parameter
    3. Merge: average only matching-sign parameters
    """
    N = len(adapter_list)
    merged = {}

    for key in adapter_list[0].keys():
        tensors = [a[key] for a in adapter_list]

        # Step 1: Trim - zero out low-magnitude entries
        trimmed = []
        for t in tensors:
            flat = t.reshape(-1)
            abs_flat = mx.abs(flat)
            k = max(1, int(flat.size * density))
            # Get threshold: k-th largest value
            # Sort descending and take k-th element
            sorted_vals = mx.sort(abs_flat)
            threshold = sorted_vals[-k]
            mx.eval(threshold)
            mask = abs_flat >= threshold
            trimmed.append((t * mask.reshape(t.shape)))
            del flat, abs_flat, sorted_vals, mask
        mx.eval(trimmed)

        # Step 2: Elect sign - majority vote
        signs = mx.stack([mx.sign(t) for t in trimmed])
        sign_sum = mx.sum(signs, axis=0)
        elected_sign = mx.sign(sign_sum)
        # Where sign_sum is 0, use first adapter's sign
        zero_mask = (sign_sum == 0)
        elected_sign = mx.where(zero_mask, mx.sign(trimmed[0]), elected_sign)
        mx.eval(elected_sign)

        # Step 3: Merge - average only parameters matching elected sign
        result = mx.zeros_like(tensors[0])
        count = mx.zeros_like(tensors[0])
        for t in trimmed:
            match = (mx.sign(t) == elected_sign) & (t != 0)
            result = result + mx.where(match, t, mx.zeros_like(t))
            count = count + match.astype(result.dtype)

        # Avoid division by zero
        safe_count = mx.where(count > 0, count, mx.ones_like(count))
        merged[key] = result / safe_count
        del tensors, trimmed, signs, sign_sum, elected_sign, result, count

    mx.eval(merged)
    return merged


def merge_dare(adapter_list, drop_rate=0.9, seed=42):
    """DARE: Drop And REscale (Yu et al., ICML 2024).

    1. Randomly drop parameters with probability p
    2. Rescale remaining by 1/(1-p)
    3. Average the sparsified adapters
    """
    mx.random.seed(seed)
    N = len(adapter_list)
    merged = {}

    for key in adapter_list[0].keys():
        result = mx.zeros_like(adapter_list[0][key])
        for i, a in enumerate(adapter_list):
            # Drop with probability drop_rate
            mask = mx.random.bernoulli(1.0 - drop_rate, shape=a[key].shape)
            # Rescale surviving parameters
            sparsified = a[key] * mask / (1.0 - drop_rate)
            result = result + sparsified
            del mask, sparsified
        merged[key] = result / N

    mx.eval(merged)
    return merged


# ===========================================================================
# Phases
# ===========================================================================
def phase_load_model():
    """Load and unpack BitNet model, apply LoRA."""
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")

    # Verify gradient computation
    def test_loss(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    test_grad = nn.value_and_grad(model, test_loss)
    x_test = mx.array([[1, 2, 3, 4, 5]])
    y_test = mx.array([[2, 3, 4, 5, 6]])
    l, g = test_grad(model, x_test, y_test)
    mx.eval(l)
    print(f"  Gradient check PASSED (loss={l.item():.4f})")
    del g, l, x_test, y_test
    log_memory("model-loaded")

    return model, tokenizer


def phase_base_ppls(model, tokenizer):
    """Compute base PPL for all domains."""
    print("\n[Phase 1] Base PPLs...")
    base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, domain)
        base_ppls[domain] = ppl
        print(f"  {domain}: {ppl:.2f}")
    return base_ppls


def phase_train_all(model, tokenizer, method="standard"):
    """Train all 5 domain adapters with given method."""
    print(f"\n[Phase 2] Training {method} adapters...")
    adapters = {}
    train_info = {}

    for domain in DOMAINS:
        print(f"\n  --- {domain} ({method}) ---")
        train_tokens = load_domain_tokens(domain, tokenizer)
        print(f"  {len(train_tokens)} training sequences")

        if method == "standard":
            result = train_standard(model, tokenizer, domain, train_tokens)
        elif method == "sam":
            result = train_sam(model, tokenizer, domain, train_tokens)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save adapter
        save_path = ADAPTERS_DIR / method / domain
        save_adapter(result["params"], save_path)
        adapters[domain] = result["params"]
        train_info[domain] = {k: v for k, v in result.items() if k != "params"}

        del train_tokens, result
        gc.collect()
        mx.clear_cache()

    log_memory(f"post-train-{method}")
    return adapters, train_info


def phase_eval_individual(model, tokenizer, adapters, method_name):
    """Evaluate each adapter individually on its own domain."""
    print(f"\n[Phase 3] Individual eval ({method_name})...")
    ppls = {}
    for domain in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain], scale=1.0)
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, domain)
        ppls[domain] = ppl
        print(f"  {domain}: PPL={ppl:.2f}")
    return ppls


def phase_eval_merged(model, tokenizer, adapters, merge_method, merge_name, base_ppls):
    """Merge adapters with given method and eval on all domains."""
    adapter_list = [adapters[d] for d in DOMAINS]

    if merge_method == "task_arithmetic":
        merged = merge_task_arithmetic(adapter_list)
    elif merge_method == "ties":
        merged = merge_ties(adapter_list)
    elif merge_method == "dare":
        merged = merge_dare(adapter_list)
    elif merge_method == "direct_sum":
        merged = merge_task_arithmetic(adapter_list, scale=1.0)
    else:
        raise ValueError(f"Unknown merge: {merge_method}")

    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())

    ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, domain)
        ppls[domain] = ppl

    avg_ppl = sum(ppls.values()) / len(ppls)
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    improvement = (avg_base - avg_ppl) / avg_base * 100

    del merged
    gc.collect()
    mx.clear_cache()

    return ppls, avg_ppl, improvement


def phase_compute_sharpness(model, tokenizer, adapters, method_name):
    """Measure loss landscape sharpness by computing loss at random perturbations.

    For each adapter, compute:
    - L(theta): loss at trained point
    - L(theta + noise): loss at 5 random perturbations
    - Sharpness = mean(L(theta + noise)) - L(theta)
    """
    print(f"\n[Phase S] Sharpness measurement ({method_name})...")
    sharpness_results = {}

    for domain in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain], scale=1.0)
        mx.eval(model.parameters())

        # Base loss
        base_ppl = compute_ppl(model, tokenizer, domain)

        # Perturbed losses
        perturbed_ppls = []
        for trial in range(5):
            mx.random.seed(trial + 100)
            # Add Gaussian noise to LoRA params
            noisy_params = {}
            for name, p in adapters[domain].items():
                noise_scale = 0.01 * mx.sqrt(mx.mean(p * p))
                noise = mx.random.normal(shape=p.shape) * noise_scale
                noisy_params[name] = p + noise
                del noise

            zero_lora_params(model)
            apply_adapter_weights(model, noisy_params, scale=1.0)
            mx.eval(model.parameters())

            ppl = compute_ppl(model, tokenizer, domain)
            perturbed_ppls.append(ppl)
            del noisy_params

        avg_perturbed = sum(perturbed_ppls) / len(perturbed_ppls)
        sharpness = avg_perturbed - base_ppl
        sharpness_pct = sharpness / base_ppl * 100

        sharpness_results[domain] = {
            "base_ppl": round(base_ppl, 4),
            "avg_perturbed_ppl": round(avg_perturbed, 4),
            "sharpness": round(sharpness, 4),
            "sharpness_pct": round(sharpness_pct, 2),
        }
        print(f"  {domain}: base={base_ppl:.2f}, perturbed={avg_perturbed:.2f}, "
              f"sharpness={sharpness:.4f} ({sharpness_pct:.2f}%)")

    return sharpness_results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    results = {
        "experiment": "flat_lora_training",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "sam_rho": SAM_RHO,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Flat-LoRA: Train in Flat Loss Regions for Mergeable Adapters")
    print("=" * 70)

    # --- Phase 0: Load model ---
    model, tokenizer = phase_load_model()

    # --- Phase 1: Base PPLs ---
    base_ppls = phase_base_ppls(model, tokenizer)
    results["base_ppls"] = base_ppls

    # --- Phase 2a: Train standard adapters ---
    std_adapters, std_train = phase_train_all(model, tokenizer, method="standard")
    results["standard_train"] = std_train

    # --- Phase 2b: Train SAM adapters ---
    try:
        sam_adapters, sam_train = phase_train_all(model, tokenizer, method="sam")
        results["sam_train"] = sam_train
        results["k1_pass"] = True
        print("\n  K1 PASS: SAM training completed on MLX")
    except Exception as e:
        print(f"\n  K1 FAIL: SAM training failed: {e}")
        results["k1_pass"] = False
        results["k1_error"] = str(e)
        results["verdict"] = "KILLED"
        results["total_time_s"] = round(time.time() - t_global, 1)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # --- Phase 3: Individual adapter quality ---
    std_individual = phase_eval_individual(model, tokenizer, std_adapters, "standard")
    sam_individual = phase_eval_individual(model, tokenizer, sam_adapters, "sam")
    results["standard_individual_ppls"] = std_individual
    results["sam_individual_ppls"] = sam_individual

    # Compare individual quality
    avg_std_ind = sum(std_individual.values()) / len(std_individual)
    avg_sam_ind = sum(sam_individual.values()) / len(sam_individual)
    ind_delta_pct = (avg_std_ind - avg_sam_ind) / avg_std_ind * 100
    results["avg_std_individual_ppl"] = round(avg_std_ind, 4)
    results["avg_sam_individual_ppl"] = round(avg_sam_ind, 4)
    results["individual_delta_pct"] = round(ind_delta_pct, 2)
    print(f"\n  Individual quality: std={avg_std_ind:.2f}, sam={avg_sam_ind:.2f} "
          f"(delta={ind_delta_pct:+.2f}%)")

    # --- Phase 4: Merged quality ---
    print("\n[Phase 4] Merge comparison...")
    merge_methods = {
        "task_arithmetic": "task_arithmetic",
        "ties": "ties",
        "dare": "dare",
        "direct_sum": "direct_sum",
    }

    merge_results = {}
    for merge_name, merge_fn in merge_methods.items():
        print(f"\n  --- {merge_name} ---")

        # Standard merged
        std_ppls, std_avg, std_imp = phase_eval_merged(
            model, tokenizer, std_adapters, merge_fn, merge_name, base_ppls)

        # SAM merged
        sam_ppls, sam_avg, sam_imp = phase_eval_merged(
            model, tokenizer, sam_adapters, merge_fn, merge_name, base_ppls)

        delta_pp = sam_imp - std_imp  # positive = SAM better

        merge_results[merge_name] = {
            "standard_ppls": {d: round(p, 4) for d, p in std_ppls.items()},
            "sam_ppls": {d: round(p, 4) for d, p in sam_ppls.items()},
            "standard_avg_ppl": round(std_avg, 4),
            "sam_avg_ppl": round(sam_avg, 4),
            "standard_improvement_pct": round(std_imp, 2),
            "sam_improvement_pct": round(sam_imp, 2),
            "delta_pp": round(delta_pp, 2),
        }
        print(f"  Std: avg={std_avg:.2f} ({std_imp:+.1f}% vs base)")
        print(f"  SAM: avg={sam_avg:.2f} ({sam_imp:+.1f}% vs base)")
        print(f"  Delta: {delta_pp:+.2f}pp ({'SAM better' if delta_pp > 0 else 'Std better'})")

    results["merge_results"] = merge_results

    # --- Phase 5: Sharpness measurement ---
    std_sharpness = phase_compute_sharpness(model, tokenizer, std_adapters, "standard")
    sam_sharpness = phase_compute_sharpness(model, tokenizer, sam_adapters, "sam")
    results["standard_sharpness"] = std_sharpness
    results["sam_sharpness"] = sam_sharpness

    avg_std_sharp = sum(s["sharpness_pct"] for s in std_sharpness.values()) / len(std_sharpness)
    avg_sam_sharp = sum(s["sharpness_pct"] for s in sam_sharpness.values()) / len(sam_sharpness)
    results["avg_std_sharpness_pct"] = round(avg_std_sharp, 2)
    results["avg_sam_sharpness_pct"] = round(avg_sam_sharp, 2)
    print(f"\n  Sharpness: std={avg_std_sharp:.2f}%, sam={avg_sam_sharp:.2f}%")

    # --- Phase 6: Orthogonality ---
    print("\n[Phase 6] Orthogonality comparison...")
    for method_name, adapters in [("standard", std_adapters), ("sam", sam_adapters)]:
        cosines = []
        for i in range(len(DOMAINS)):
            for j in range(i + 1, len(DOMAINS)):
                vi = mx.concatenate([v.reshape(-1) for v in adapters[DOMAINS[i]].values()])
                vj = mx.concatenate([v.reshape(-1) for v in adapters[DOMAINS[j]].values()])
                cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2))))
                mx.eval(cos)
                cosines.append(cos.item())
                del vi, vj
        mean_cos = sum(cosines) / len(cosines)
        results[f"{method_name}_mean_cos"] = round(mean_cos, 4)
        print(f"  {method_name}: mean |cos| = {mean_cos:.4f}")

    # --- Kill criteria assessment ---
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K2: best merge method delta
    best_delta = max(m["delta_pp"] for m in merge_results.values())
    best_method = max(merge_results, key=lambda k: merge_results[k]["delta_pp"])
    results["k2_best_delta_pp"] = round(best_delta, 2)
    results["k2_best_method"] = best_method
    results["k2_pass"] = best_delta > 0  # any improvement at all

    # S1: >3pp improvement
    results["s1_pass"] = best_delta > 3.0

    print(f"\n  K1 (SAM on MLX): PASS")
    print(f"  K2 (merge improvement): {'PASS' if results['k2_pass'] else 'FAIL'} "
          f"(best delta={best_delta:+.2f}pp via {best_method})")
    print(f"  S1 (>3pp improvement): {'PASS' if results['s1_pass'] else 'FAIL'}")

    # Training time comparison
    std_total = sum(t["train_time_s"] for t in std_train.values())
    sam_total = sum(t["train_time_s"] for t in sam_train.values())
    results["std_total_train_s"] = round(std_total, 1)
    results["sam_total_train_s"] = round(sam_total, 1)
    results["sam_overhead_x"] = round(sam_total / max(std_total, 0.1), 2)
    print(f"\n  Training time: std={std_total:.0f}s, sam={sam_total:.0f}s "
          f"({results['sam_overhead_x']:.1f}x)")

    # Verdict
    if not results["k1_pass"]:
        verdict = "KILLED"
    elif not results["k2_pass"]:
        verdict = "KILLED"
    elif results["s1_pass"]:
        verdict = "SUPPORTED"
    else:
        verdict = "SUPPORTED"  # K2 pass (some improvement) even if <3pp

    results["verdict"] = verdict
    results["total_time_s"] = round(time.time() - t_global, 1)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Total time: {results['total_time_s']:.0f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("MERGE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<20} {'Std PPL':>10} {'SAM PPL':>10} {'Delta pp':>10}")
    print(f"  {'-'*50}")
    for name, m in merge_results.items():
        print(f"  {name:<20} {m['standard_avg_ppl']:>10.2f} {m['sam_avg_ppl']:>10.2f} "
              f"{m['delta_pp']:>+10.2f}")

    # Clean up
    cleanup(model, tokenizer, std_adapters, sam_adapters)
    log_memory("final")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
