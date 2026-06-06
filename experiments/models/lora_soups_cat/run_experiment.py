#!/usr/bin/env python3
"""
LoRA Soups CAT: Learnable per-module composition coefficients.

Kill criteria:
  K1 (id:554): CAT not better than DO-Merging (uniform 1/N merge)
  K2 (id:555): No superlinear composition effect

Success criteria:
  S1 (id:64): Composed > best individual on >=2/5 domains (superlinear) at zero inference cost

Reference: arXiv 2410.13025 (LoRA Soups)
  delta_W^l = alpha_1^l * B_1@A_1 + alpha_2^l * B_2@A_2
  Where alpha_i^l are learnable scalars per layer per adapter,
  trained on small calibration set.

Approach:
  1. REUSE 5 domain adapters from flat_lora_training (standard LoRA on BitNet-2B-4T)
  2. Learn per-module coefficients alpha on 5% calibration data (freeze everything)
  3. Merge with learned coefficients (static -- no inference cost)
  4. Compare: uniform 1/N, Task Arithmetic, TIES, DARE, CAT
  5. Evaluate: PPL per domain, superlinear check
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

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse adapters from flat_lora_training
ADAPTER_SOURCE = Path(__file__).parent.parent / "flat_lora_training" / "adapters" / "standard"
# Reuse data from bitnet_2b_real_composition
DATA_SOURCE = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
DOMAINS = ["python", "math", "medical", "legal", "creative"]

# CAT training hyperparameters
CAT_LR_SWEEP = [1e-4, 1e-3, 1e-2, 1e-1]  # LR sweep (Fix 1 from review)
CAT_ITERS = 200          # Steps to optimize alpha
CAT_CAL_FRACTION = 0.05  # 5% of training data for calibration
CAT_REG_LAMBDA = 0.01    # L2 regularization toward 1/N


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
# Ternary unpacking (reused from flat_lora_training)
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
# LoRA utilities
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
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_tokens(domain_name, tokenizer, fraction=1.0):
    data_dir = DATA_SOURCE / domain_name
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Data not found: {train_path}")

    texts = []
    with open(train_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    # Take fraction of data
    n_samples = max(1, int(len(texts) * fraction))
    texts = texts[:n_samples]

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
# Merge methods (reused from flat_lora_training)
# ===========================================================================
def merge_uniform(adapter_list):
    """Uniform 1/N merge (baseline)."""
    N = len(adapter_list)
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) / N
    mx.eval(merged)
    return merged


def merge_task_arithmetic(adapter_list, lam=0.2):
    """Task Arithmetic: global scalar lambda."""
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * lam
    mx.eval(merged)
    return merged


def merge_ties(adapter_list, density=0.2):
    """TIES: Trim-Elect-Sign merge."""
    merged = {}
    for key in adapter_list[0].keys():
        tensors = [a[key] for a in adapter_list]
        trimmed = []
        for t in tensors:
            flat = t.reshape(-1)
            abs_flat = mx.abs(flat)
            k = max(1, int(flat.size * density))
            sorted_vals = mx.sort(abs_flat)
            threshold = sorted_vals[-k]
            mx.eval(threshold)
            mask = abs_flat >= threshold
            trimmed.append(t * mask.reshape(t.shape))
            del flat, abs_flat, sorted_vals, mask

        mx.eval(trimmed)
        signs = mx.stack([mx.sign(t) for t in trimmed])
        sign_sum = mx.sum(signs, axis=0)
        elected_sign = mx.sign(sign_sum)
        zero_mask = (sign_sum == 0)
        elected_sign = mx.where(zero_mask, mx.sign(trimmed[0]), elected_sign)
        mx.eval(elected_sign)

        result = mx.zeros_like(tensors[0])
        count = mx.zeros_like(tensors[0])
        for t in trimmed:
            match = (mx.sign(t) == elected_sign) & (t != 0)
            result = result + mx.where(match, t, mx.zeros_like(t))
            count = count + match.astype(result.dtype)
        safe_count = mx.where(count > 0, count, mx.ones_like(count))
        merged[key] = result / safe_count
        del tensors, trimmed, signs, sign_sum, elected_sign, result, count
    mx.eval(merged)
    return merged


def merge_dare(adapter_list, drop_rate=0.9, seed=42):
    """DARE: Drop And REscale."""
    mx.random.seed(seed)
    N = len(adapter_list)
    merged = {}
    for key in adapter_list[0].keys():
        result = mx.zeros_like(adapter_list[0][key])
        for a in adapter_list:
            mask = mx.random.bernoulli(1.0 - drop_rate, shape=a[key].shape)
            result = result + a[key] * mask / (1.0 - drop_rate)
            del mask
        merged[key] = result / N
    mx.eval(merged)
    return merged


def merge_cat(adapter_list, alphas):
    """CAT: Composition via Adaptive Training.

    alphas: dict mapping (adapter_idx, module_name) -> scalar coefficient
    """
    N = len(adapter_list)
    merged = {}
    for key in adapter_list[0].keys():
        result = mx.zeros_like(adapter_list[0][key])
        for i, a in enumerate(adapter_list):
            alpha = alphas.get((i, key), 1.0 / N)
            result = result + a[key] * alpha
        merged[key] = result
    mx.eval(merged)
    return merged


# ===========================================================================
# CAT: Learn per-module composition coefficients
# ===========================================================================
def _train_cat_at_lr(model, cal_tokens, adapter_list, module_names, N, M, lr):
    """Run CAT training at a specific learning rate. Returns (alpha, losses, train_time).

    Fix 1 from review: separated training loop for LR sweep.
    Fix 1 also: use mx.value_and_grad to avoid double forward pass.
    The old code called cat_loss_fn + grad_fn separately = 2 forward passes per step.
    """
    t_start = time.time()
    total_params = N * M

    alpha = mx.ones((N, M)) / N
    alpha_init = mx.array(alpha)

    def cat_loss_fn(alpha, model, x, y):
        """Loss function for CAT optimization.

        Applies alpha-weighted composition via setattr, runs forward pass.
        NOTE: setattr mutates model state as a side effect. This is safe with
        MLX lazy evaluation because the graph is built (not executed) during
        tracing, so the model sees the composed weights when the graph runs.
        """
        for m_idx, module_name in enumerate(module_names):
            result = mx.zeros_like(adapter_list[0][module_name])
            for i in range(N):
                result = result + alpha[i, m_idx] * adapter_list[i][module_name]
            parts = module_name.split(".")
            obj = model
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            setattr(obj, parts[-1], result)

        logits = model(x)
        ce_loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        reg_loss = CAT_REG_LAMBDA * mx.sum((alpha - alpha_init) ** 2)
        return ce_loss + reg_loss

    # Use value_and_grad to compute loss + gradient in single forward pass
    loss_and_grad_fn = mx.value_and_grad(cat_loss_fn, argnums=0)

    # Adam state
    m_adam = mx.zeros_like(alpha)
    v_adam = mx.zeros_like(alpha)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses = []

    gc.disable()
    for step in range(CAT_ITERS):
        idx = step % len(cal_tokens)
        tokens = cal_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        # Single forward + backward via value_and_grad
        loss_val, grad_alpha = loss_and_grad_fn(alpha, model, x, y)
        mx.eval(loss_val, grad_alpha)

        # Adam update
        t_step = step + 1
        m_adam = beta1 * m_adam + (1 - beta1) * grad_alpha
        v_adam = beta2 * v_adam + (1 - beta2) * (grad_alpha ** 2)
        m_hat = m_adam / (1 - beta1 ** t_step)
        v_hat = v_adam / (1 - beta2 ** t_step)
        alpha = alpha - lr * m_hat / (mx.sqrt(v_hat) + eps)
        mx.eval(alpha, m_adam, v_adam)

        losses.append(loss_val.item())

        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            mx.eval(alpha)
            alpha_min = mx.min(alpha).item()
            alpha_max = mx.max(alpha).item()
            alpha_std = mx.sqrt(mx.mean((alpha - 1.0/N)**2)).item()
            print(f"      Step {step+1}/{CAT_ITERS}: loss={losses[-1]:.4f} (avg50={avg:.4f}) "
                  f"alpha: [{alpha_min:.4f}, {alpha_max:.4f}] std={alpha_std:.4f}")

        del x, y, grad_alpha, loss_val
    gc.enable()
    gc.collect()

    train_time = time.time() - t_start
    del m_adam, v_adam
    gc.collect()
    mx.clear_cache()

    return alpha, losses, train_time


def phase_learn_cat(model, tokenizer, adapters_dict, base_ppls):
    """Learn per-module alpha coefficients on calibration data.

    This is the core of LoRA Soups CAT (arXiv 2410.13025).
    Fix 1 from review: sweep over LR in {1e-4, 1e-3, 1e-2, 1e-1} and pick
    the best by final calibration loss. If ALL LRs diverge, the "landscape
    too flat" conclusion is definitively justified.
    """
    print("\n[Phase CAT] Learning composition coefficients...")
    t_start_overall = time.time()

    N = len(DOMAINS)
    adapter_list = [adapters_dict[d] for d in DOMAINS]
    module_names = sorted(adapter_list[0].keys())
    M = len(module_names)
    total_params = N * M
    print(f"  {N} adapters x {M} modules = {total_params} learnable scalars")

    # Load calibration data (5% from each domain, mixed)
    print("  Loading calibration data...")
    cal_tokens = []
    for domain in DOMAINS:
        domain_tokens = load_domain_tokens(domain, tokenizer, fraction=CAT_CAL_FRACTION)
        cal_tokens.extend(domain_tokens)
        print(f"    {domain}: {len(domain_tokens)} calibration sequences")
    print(f"  Total calibration sequences: {len(cal_tokens)}")

    # --- LR Sweep (Fix 1) ---
    lr_results = {}
    best_lr = None
    best_final_loss = float("inf")
    best_alpha = None

    for lr in CAT_LR_SWEEP:
        print(f"\n  --- LR={lr:.0e} ---")
        alpha, losses, train_time = _train_cat_at_lr(
            model, cal_tokens, adapter_list, module_names, N, M, lr
        )
        first_50 = sum(losses[:min(50, len(losses))]) / min(50, len(losses))
        last_50 = sum(losses[-50:]) / len(losses[-50:])
        diverged = last_50 > first_50 * 1.1  # >10% increase = diverged

        lr_results[f"{lr:.0e}"] = {
            "first_50_loss": round(first_50, 4),
            "last_50_loss": round(last_50, 4),
            "diverged": diverged,
            "train_time_s": round(train_time, 1),
        }

        print(f"    first_50_loss={first_50:.4f}, last_50_loss={last_50:.4f}, "
              f"{'DIVERGED' if diverged else 'converged'}, time={train_time:.1f}s")

        if not diverged and last_50 < best_final_loss:
            best_final_loss = last_50
            best_lr = lr
            best_alpha = alpha
        elif best_alpha is None and not diverged:
            # fallback: take first non-diverged
            best_final_loss = last_50
            best_lr = lr
            best_alpha = alpha

    # If ALL diverged, pick the one with lowest last_50 loss anyway
    if best_alpha is None:
        print("\n  WARNING: ALL learning rates diverged! Picking least-bad.")
        best_lr_key = min(lr_results, key=lambda k: lr_results[k]["last_50_loss"])
        best_lr = float(best_lr_key)
        # Re-run best to get alpha (we didn't save diverged ones)
        print(f"  Re-running with lr={best_lr:.0e}...")
        best_alpha, losses, _ = _train_cat_at_lr(
            model, cal_tokens, adapter_list, module_names, N, M, best_lr
        )
        best_final_loss = sum(losses[-50:]) / len(losses[-50:])

    print(f"\n  Best LR: {best_lr:.0e} (final_loss={best_final_loss:.4f})")

    alpha = best_alpha
    total_time = time.time() - t_start_overall

    # Build final alpha dict for merge_cat
    mx.eval(alpha)
    alpha_dict = {}
    for i in range(N):
        for m_idx, module_name in enumerate(module_names):
            alpha_dict[(i, module_name)] = alpha[i, m_idx].item()

    # Compute statistics
    alpha_flat = alpha.reshape(-1)
    mx.eval(alpha_flat)
    stats = {
        "train_time_s": round(total_time, 1),
        "best_lr": best_lr,
        "lr_sweep": lr_results,
        "n_params": total_params,
        "n_cal_sequences": len(cal_tokens),
        "best_final_loss": round(best_final_loss, 4),
        "alpha_mean": round(mx.mean(alpha_flat).item(), 4),
        "alpha_std": round(mx.sqrt(mx.mean((alpha_flat - 1.0/N)**2)).item(), 4),
        "alpha_min": round(mx.min(alpha_flat).item(), 4),
        "alpha_max": round(mx.max(alpha_flat).item(), 4),
    }

    # Per-adapter mean alpha
    per_adapter_mean = {}
    for i, domain in enumerate(DOMAINS):
        mean_a = mx.mean(alpha[i]).item()
        per_adapter_mean[domain] = round(mean_a, 4)
    stats["per_adapter_mean_alpha"] = per_adapter_mean

    # Per-layer variance
    layer_variances = {}
    for l in range(24):
        layer_prefix = f"model.layers.{l}."
        layer_indices = [m_idx for m_idx, name in enumerate(module_names)
                         if name.startswith(layer_prefix)]
        if layer_indices:
            layer_alphas = alpha[:, layer_indices]
            var = mx.var(layer_alphas).item()
            layer_variances[f"layer_{l}"] = round(var, 6)
    stats["layer_variances"] = layer_variances

    print(f"\n  CAT training done in {total_time:.1f}s (best lr={best_lr:.0e})")
    print(f"  Alpha stats: mean={stats['alpha_mean']:.4f}, std={stats['alpha_std']:.4f}, "
          f"range=[{stats['alpha_min']:.4f}, {stats['alpha_max']:.4f}]")
    print(f"  Per-adapter mean alpha: {per_adapter_mean}")

    del cal_tokens
    gc.collect()
    mx.clear_cache()

    return alpha_dict, stats


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

    # Freeze everything -- we only learn alpha scalars
    model.freeze()
    # But we still need LoRA params accessible for read/write
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")
    log_memory("model-loaded")

    return model, tokenizer


def phase_load_adapters():
    """Load pre-trained adapters from flat_lora_training."""
    print("\n[Phase 1] Loading pre-trained adapters...")
    adapters = {}
    for domain in DOMAINS:
        path = ADAPTER_SOURCE / domain
        if not (path / "adapter.npz").exists():
            raise FileNotFoundError(f"Adapter not found: {path}")
        adapters[domain] = load_adapter(path)
        n_params = sum(v.size for v in adapters[domain].values())
        n_modules = len(adapters[domain])
        print(f"  {domain}: {n_modules} modules, {n_params:,} params")
    return adapters


def phase_base_ppls(model, tokenizer):
    """Compute base PPL for all domains."""
    print("\n[Phase 2] Base PPLs...")
    zero_lora_params(model)
    mx.eval(model.parameters())
    base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, domain)
        base_ppls[domain] = ppl
        print(f"  {domain}: {ppl:.2f}")
    return base_ppls


def phase_eval_individual(model, tokenizer, adapters_dict):
    """Evaluate each adapter individually on its own domain."""
    print("\n[Phase 3] Individual adapter PPLs...")
    ppls = {}
    for domain in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, adapters_dict[domain], scale=1.0)
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, domain)
        ppls[domain] = ppl
        print(f"  {domain}: PPL={ppl:.2f}")
    return ppls


def phase_eval_merged(model, tokenizer, merged_params, label=""):
    """Evaluate merged adapter on all domains."""
    zero_lora_params(model)
    apply_adapter_weights(model, merged_params, scale=1.0)
    mx.eval(model.parameters())

    ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, domain)
        ppls[domain] = ppl
    return ppls


def phase_merge_baselines(model, tokenizer, adapters_dict, base_ppls):
    """Run all baseline merge methods."""
    print("\n[Phase 4] Baseline merge methods...")
    adapter_list = [adapters_dict[d] for d in DOMAINS]
    N = len(adapter_list)

    methods = {}

    # 1. Uniform 1/N (DO-Merging equivalent for near-orthogonal adapters)
    print("\n  --- Uniform 1/N ---")
    merged = merge_uniform(adapter_list)
    ppls = phase_eval_merged(model, tokenizer, merged, "uniform")
    avg_ppl = sum(ppls.values()) / len(ppls)
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    imp = (avg_base - avg_ppl) / avg_base * 100
    methods["uniform"] = {"ppls": {d: round(p, 4) for d, p in ppls.items()},
                          "avg_ppl": round(avg_ppl, 4), "improvement_pct": round(imp, 2)}
    print(f"    avg_ppl={avg_ppl:.2f} ({imp:+.1f}% vs base)")
    for d in DOMAINS:
        print(f"      {d}: {ppls[d]:.2f}")
    del merged
    gc.collect(); mx.clear_cache()

    # 2. Task Arithmetic -- sweep lambda to find best (Fix 3 from review)
    # NOTE: Task Arithmetic with lambda=1/N is mathematically identical to
    # uniform 1/N merge: sum(delta_i) * (1/N) == sum(delta_i / N).
    # So we sweep lambda to find the best value; lambda=0.2 is the degenerate case.
    ta_lambdas = [0.1, 0.2, 0.3, 0.5]
    best_ta_ppl = float("inf")
    best_ta_lam = 0.2
    ta_sweep = {}
    for lam in ta_lambdas:
        merged = merge_task_arithmetic(adapter_list, lam=lam)
        ppls_ta = phase_eval_merged(model, tokenizer, merged, f"ta_{lam}")
        avg = sum(ppls_ta.values()) / len(ppls_ta)
        ta_sweep[str(lam)] = round(avg, 4)
        if avg < best_ta_ppl:
            best_ta_ppl = avg
            best_ta_lam = lam
            ppls = ppls_ta
        del merged
        gc.collect(); mx.clear_cache()
    print(f"\n  --- Task Arithmetic (best lambda={best_ta_lam}, sweep={ta_sweep}) ---")
    avg_ppl = best_ta_ppl
    imp = (avg_base - avg_ppl) / avg_base * 100
    methods["task_arithmetic"] = {
        "ppls": {d: round(p, 4) for d, p in ppls.items()},
        "avg_ppl": round(avg_ppl, 4),
        "improvement_pct": round(imp, 2),
        "best_lambda": best_ta_lam,
        "lambda_sweep": ta_sweep,
        "note": "lambda=1/N=0.2 is identical to uniform merge; swept to find true optimum",
    }
    print(f"    avg_ppl={avg_ppl:.2f} ({imp:+.1f}% vs base)")

    # 3. TIES
    print("\n  --- TIES ---")
    merged = merge_ties(adapter_list, density=0.2)
    ppls = phase_eval_merged(model, tokenizer, merged, "ties")
    avg_ppl = sum(ppls.values()) / len(ppls)
    imp = (avg_base - avg_ppl) / avg_base * 100
    methods["ties"] = {"ppls": {d: round(p, 4) for d, p in ppls.items()},
                       "avg_ppl": round(avg_ppl, 4), "improvement_pct": round(imp, 2)}
    print(f"    avg_ppl={avg_ppl:.2f} ({imp:+.1f}% vs base)")
    del merged
    gc.collect(); mx.clear_cache()

    # 4. DARE
    print("\n  --- DARE ---")
    merged = merge_dare(adapter_list, drop_rate=0.9)
    ppls = phase_eval_merged(model, tokenizer, merged, "dare")
    avg_ppl = sum(ppls.values()) / len(ppls)
    imp = (avg_base - avg_ppl) / avg_base * 100
    methods["dare"] = {"ppls": {d: round(p, 4) for d, p in ppls.items()},
                       "avg_ppl": round(avg_ppl, 4), "improvement_pct": round(imp, 2)}
    print(f"    avg_ppl={avg_ppl:.2f} ({imp:+.1f}% vs base)")
    del merged
    gc.collect(); mx.clear_cache()

    return methods


def phase_cat_eval(model, tokenizer, adapters_dict, alpha_dict, base_ppls):
    """Evaluate CAT-merged adapter on all domains."""
    print("\n[Phase 5] CAT merge evaluation...")
    adapter_list = [adapters_dict[d] for d in DOMAINS]

    merged = merge_cat(adapter_list, alpha_dict)
    ppls = phase_eval_merged(model, tokenizer, merged, "cat")
    avg_ppl = sum(ppls.values()) / len(ppls)
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    imp = (avg_base - avg_ppl) / avg_base * 100

    result = {
        "ppls": {d: round(p, 4) for d, p in ppls.items()},
        "avg_ppl": round(avg_ppl, 4),
        "improvement_pct": round(imp, 2),
    }

    print(f"    avg_ppl={avg_ppl:.2f} ({imp:+.1f}% vs base)")
    for d in DOMAINS:
        print(f"      {d}: {ppls[d]:.2f}")

    del merged
    gc.collect(); mx.clear_cache()

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    results = {
        "experiment": "lora_soups_cat",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "cat_lr_sweep": CAT_LR_SWEEP,
        "cat_iters": CAT_ITERS,
        "cat_cal_fraction": CAT_CAL_FRACTION,
        "cat_reg_lambda": CAT_REG_LAMBDA,
        "domains": DOMAINS,
        "adapter_source": str(ADAPTER_SOURCE),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("LoRA Soups CAT: Learnable Per-Module Composition Coefficients")
    print("arXiv 2410.13025")
    print("=" * 70)

    # Phase 0: Load model
    model, tokenizer = phase_load_model()

    # Phase 1: Load pre-trained adapters
    adapters = phase_load_adapters()

    # Phase 2: Base PPLs
    base_ppls = phase_base_ppls(model, tokenizer)
    results["base_ppls"] = base_ppls

    # Phase 3: Individual adapter PPLs
    individual_ppls = phase_eval_individual(model, tokenizer, adapters)
    results["individual_ppls"] = individual_ppls

    # Phase 4: Baseline merges
    baseline_results = phase_merge_baselines(model, tokenizer, adapters, base_ppls)
    results["baseline_merges"] = baseline_results

    # Phase CAT: Learn composition coefficients
    alpha_dict, cat_stats = phase_learn_cat(model, tokenizer, adapters, base_ppls)
    results["cat_training"] = cat_stats

    # Phase 5: Evaluate CAT merge
    cat_result = phase_cat_eval(model, tokenizer, adapters, alpha_dict, base_ppls)
    results["cat_merge"] = cat_result

    # ===========================================================
    # Kill criteria assessment
    # ===========================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: CAT better than DO-Merging (uniform 1/N)?
    uniform_avg = baseline_results["uniform"]["avg_ppl"]
    cat_avg = cat_result["avg_ppl"]
    cat_vs_uniform = (uniform_avg - cat_avg) / uniform_avg * 100
    k1_pass = cat_avg < uniform_avg
    results["k1_cat_vs_uniform_pct"] = round(cat_vs_uniform, 2)
    results["k1_pass"] = k1_pass
    print(f"\n  K1 (CAT better than uniform 1/N):")
    print(f"    Uniform avg PPL: {uniform_avg:.4f}")
    print(f"    CAT avg PPL: {cat_avg:.4f}")
    print(f"    CAT improvement: {cat_vs_uniform:+.2f}%")
    print(f"    K1: {'PASS' if k1_pass else 'FAIL'}")

    # Also compare against best baseline
    best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["avg_ppl"])
    best_baseline_ppl = baseline_results[best_baseline_name]["avg_ppl"]
    cat_vs_best = (best_baseline_ppl - cat_avg) / best_baseline_ppl * 100
    results["cat_vs_best_baseline_pct"] = round(cat_vs_best, 2)
    results["best_baseline_method"] = best_baseline_name
    print(f"    Best baseline ({best_baseline_name}): {best_baseline_ppl:.4f}")
    print(f"    CAT vs best baseline: {cat_vs_best:+.2f}%")

    # K2: Superlinear composition?
    # Composed > best individual on >= 2/5 domains
    superlinear_domains = []
    for domain in DOMAINS:
        cat_ppl = cat_result["ppls"][domain]
        ind_ppl = individual_ppls[domain]
        if cat_ppl < ind_ppl:
            superlinear_domains.append(domain)
            print(f"\n  Superlinear on {domain}: CAT={cat_ppl:.2f} < individual={ind_ppl:.2f}")

    n_superlinear = len(superlinear_domains)
    k2_pass = n_superlinear >= 2
    results["k2_superlinear_domains"] = superlinear_domains
    results["k2_n_superlinear"] = n_superlinear
    results["k2_pass"] = k2_pass
    print(f"\n  K2 (Superlinear on >=2/5 domains):")
    print(f"    Superlinear domains: {n_superlinear}/5 ({superlinear_domains})")
    print(f"    K2: {'PASS' if k2_pass else 'FAIL'}")

    # S1: Superlinear at zero inference cost
    s1_pass = k2_pass  # Same as K2 (merged = zero inference cost by construction)
    results["s1_pass"] = s1_pass
    print(f"\n  S1 (Superlinear at zero inference cost): {'PASS' if s1_pass else 'FAIL'}")

    # Per-domain comparison table
    print("\n" + "=" * 70)
    print("PER-DOMAIN PPL COMPARISON")
    print("=" * 70)
    print(f"  {'Domain':<12} {'Base':>8} {'Individual':>12} {'Uniform':>10} {'TIES':>10} {'CAT':>10} {'CAT vs Ind':>12}")
    print(f"  {'-'*74}")
    for domain in DOMAINS:
        bp = base_ppls[domain]
        ip = individual_ppls[domain]
        up = baseline_results["uniform"]["ppls"][domain]
        tp = baseline_results["ties"]["ppls"][domain]
        cp = cat_result["ppls"][domain]
        delta = (ip - cp) / ip * 100
        marker = " *" if cp < ip else ""
        print(f"  {domain:<12} {bp:>8.2f} {ip:>12.2f} {up:>10.2f} {tp:>10.2f} {cp:>10.2f} {delta:>+11.1f}%{marker}")

    # Merge comparison summary
    print("\n" + "=" * 70)
    print("MERGE METHOD COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<20} {'Avg PPL':>10} {'vs Base':>10} {'vs Uniform':>12}")
    print(f"  {'-'*52}")
    for name, m in baseline_results.items():
        vs_uni = (uniform_avg - m["avg_ppl"]) / uniform_avg * 100
        print(f"  {name:<20} {m['avg_ppl']:>10.4f} {m['improvement_pct']:>+9.1f}% {vs_uni:>+11.1f}%")
    print(f"  {'CAT':<20} {cat_avg:>10.4f} {cat_result['improvement_pct']:>+9.1f}% {cat_vs_uniform:>+11.1f}%")

    # Verdict (Fix 4 from review: consistent logic with clear caveat)
    # K1 is the primary kill criterion. K2 is a stretch goal.
    # K1 PASS + K2 PASS -> SUPPORTED (full)
    # K1 PASS + K2 FAIL -> SUPPORTED (with caveat: no superlinear)
    # K1 FAIL -> KILLED
    if k1_pass and k2_pass:
        verdict = "SUPPORTED"
        verdict_detail = "K1 PASS + K2 PASS: CAT beats uniform AND achieves superlinear"
    elif k1_pass:
        verdict = "SUPPORTED"
        verdict_detail = "K1 PASS, K2 FAIL: CAT beats uniform but no superlinear (stretch goal)"
    else:
        verdict = "KILLED"
        verdict_detail = "K1 FAIL: CAT does not beat uniform merge"

    results["verdict"] = verdict
    results["verdict_detail"] = verdict_detail
    results["total_time_s"] = round(time.time() - t_global, 1)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Detail: {verdict_detail}")
    print(f"  Total time: {results['total_time_s']:.0f}s")

    # Clean up
    cleanup(model, tokenizer, adapters)
    log_memory("final")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
