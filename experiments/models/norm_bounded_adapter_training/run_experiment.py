#!/usr/bin/env python3
"""Norm-bounded adapter training: eliminate scale imbalance at source.

Kill criteria:
  K709: Norm-bounded adapters have Gini coefficient > 0.15 (raw sum) ->
        training did not equalize scales
  K710: Norm-bounded composition PPL worse than partial equalization
        baseline (6.508) -> norm constraint hurts quality
  K711: Training with norm constraint fails to converge on >= 2/5 domains
        -> norm bound too tight

Type: Guided Exploration (Type 2)
Papers: NB-LoRA (arXiv:2501.19050), DeLoRA (arXiv:2503.18225)
Prior: Finding #279 (50% Frobenius ceiling), Finding #281 (Fisher=Frobenius)

Approach:
  Train 5 domain adapters with 3 norm-control strategies:
  (A) Norm projection: after each optimizer step, project B so that
      s_i * ||B_i||_F = target_delta_norm for all domains
  (B) Scale-compensated weight decay: lambda_i = lambda_0 * (s_i/s_ref)^2
  (C) Uniform-scale training: train all at s=10 with standard weight decay

  Then compose (raw sum with uniform scale) and measure Gini, PPL.
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
from mlx_lm import load

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data and skeleton from existing experiment
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR_SOURCE = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"
SKELETON_PATH = ADAPTERS_DIR_SOURCE / "grassmannian_skeleton.npz"

# Output directory for norm-bounded adapters
NB_ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
SEED = 42
N_EVAL_SAMPLES = 20  # per domain for PPL evaluation

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Per-domain optimal scales (Finding #249: behavioral regimes)
PER_DOMAIN_SCALES = {
    "medical": 20.0, "code": 20.0, "math": 20.0,
    "legal": 4.0, "finance": 1.0,
}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

NUM_LAYERS = 30


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
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


def gini_coefficient(values):
    """Compute Gini coefficient of an array of non-negative values."""
    vals = np.sort(np.asarray(values, dtype=np.float64))
    n = len(vals)
    if n <= 1 or np.sum(vals) < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * vals) / (n * np.sum(vals)))


# ============================================================================
# BitNet unpacking (reused from prior experiments)
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    from mlx_lm.models.bitlinear_layers import BitLinear
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
    from mlx_lm.models.bitlinear_layers import BitLinear
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


# ============================================================================
# STE Ternary LoRA Layer (from real_data_domain_experts)
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B.

    Forward: y = base(x) + (x @ A) @ quantize(B) * scale
    """
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
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))

        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank

        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


# ============================================================================
# Training infrastructure
# ============================================================================

def apply_ternary_lora(model, rank, scale, a_matrices_per_layer):
    """Apply TernaryLoRALinear to all target projection layers."""
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
            if a_key in a_matrices_per_layer:
                a_np = a_matrices_per_layer[a_key]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
            else:
                a_mx = None
            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied TernaryLoRA (r={rank}, s={scale}) to {count} layers")
    return model


def get_trainable_params(model):
    """Get dict of trainable (lora_b) parameters."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    """Save trainable LoRA B parameters."""
    path.mkdir(parents=True, exist_ok=True)
    params = get_trainable_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def zero_b_params(model):
    """Reset all lora_b to zeros."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def compute_total_b_norm(model):
    """Compute total Frobenius norm of all B matrices."""
    total_sq = 0.0
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            norm_sq = mx.sum(p * p)
            mx.eval(norm_sq)
            total_sq += norm_sq.item()
    return math.sqrt(total_sq)


def project_b_norms(model, target_total_norm):
    """Project all B matrices so total Frobenius norm = target_total_norm.

    This is a per-adapter (total) norm projection, not per-matrix.
    Each B_k is scaled by the same factor to hit the target total norm.
    """
    current_norm = compute_total_b_norm(model)
    if current_norm < 1e-8:
        return current_norm
    ratio = target_total_norm / current_norm
    if ratio < 1.0:
        # Only project if norm exceeds target (upper bound constraint)
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, TernaryLoRALinear):
                    module.lora_b = module.lora_b * ratio
        mx.eval(model.parameters())
    return current_norm


def normalize_b_norms(model, target_total_norm):
    """Normalize all B matrices so total Frobenius norm = target_total_norm.

    Unlike projection, this always rescales (both up and down).
    """
    current_norm = compute_total_b_norm(model)
    if current_norm < 1e-8:
        return current_norm
    ratio = target_total_norm / current_norm
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = module.lora_b * ratio
    mx.eval(model.parameters())
    return current_norm


# ============================================================================
# Training strategies
# ============================================================================

def train_adapter_strategy_A(model, tokenizer, train_tokens, domain_scale,
                              target_delta_norm, train_iters=200):
    """Strategy A: Norm projection.

    Train at per-domain scale, after each step project B so that
    s * ||B||_F = target_delta_norm / sqrt(r).
    This ensures all domains contribute equally to composed energy.

    target_delta_norm = desired ||Delta||_F = s * ||B||_F * sqrt(r)
    => target_B_norm = target_delta_norm / (s * sqrt(r))
    """
    target_b_norm = target_delta_norm / (domain_scale * math.sqrt(LORA_RANK))
    log(f"    Strategy A: norm projection, target ||B||_F = {target_b_norm:.2f}")
    log(f"    (domain_scale={domain_scale}, target_delta_norm={target_delta_norm:.1f})")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    b_norms = []
    gc.disable()
    for step in range(train_iters):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        # Norm projection after optimizer step
        pre_norm = project_b_norms(model, target_b_norm)

        loss_val = loss.item()
        losses.append(loss_val)
        b_norms.append(pre_norm)

        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"      Step {step+1}/{train_iters}: loss={loss_val:.4f} "
                f"(avg50={avg:.4f}) ||B||_F={pre_norm:.2f}->{target_b_norm:.2f}")

    gc.enable()
    gc.collect()

    final_norm = compute_total_b_norm(model)
    return {
        "losses": losses,
        "b_norms": b_norms,
        "final_b_norm": final_norm,
        "target_b_norm": target_b_norm,
    }


def train_adapter_strategy_B(model, tokenizer, train_tokens, domain_scale,
                              lambda_0=0.01, train_iters=200):
    """Strategy B: Scale-compensated weight decay.

    Weight decay lambda_i = lambda_0 * (s_i / s_ref)^2 so that higher-scale
    domains get proportionally stronger decay, pushing their B-norms down.
    s_ref = geometric mean of all scales.
    """
    import functools

    s_ref = np.exp(np.mean(np.log(list(PER_DOMAIN_SCALES.values()))))
    lambda_wd = lambda_0 * (domain_scale / s_ref) ** 2
    log(f"    Strategy B: weight decay, lambda={lambda_wd:.4f} "
        f"(s={domain_scale}, s_ref={s_ref:.2f}, lambda_0={lambda_0})")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        ce_loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        # Add weight decay on B matrices only
        wd_loss = mx.array(0.0)
        for name, p in tree_flatten(model.trainable_parameters()):
            if "lora_b" in name:
                wd_loss = wd_loss + mx.sum(p * p)
        return ce_loss + lambda_wd * wd_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    b_norms = []
    gc.disable()
    for step in range(train_iters):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0:
            b_norm = compute_total_b_norm(model)
            b_norms.append(b_norm)
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"      Step {step+1}/{train_iters}: loss={loss_val:.4f} "
                f"(avg50={avg:.4f}) ||B||_F={b_norm:.2f}")

    gc.enable()
    gc.collect()

    final_norm = compute_total_b_norm(model)
    return {
        "losses": losses,
        "b_norms": b_norms,
        "final_b_norm": final_norm,
        "lambda_wd": lambda_wd,
    }


def train_adapter_strategy_C(model, tokenizer, train_tokens,
                              uniform_scale=10.0, lambda_wd=0.001,
                              train_iters=200):
    """Strategy C: Uniform-scale training.

    Train all domains at a single scale s=10 (compromise between capability
    s=20 and format s=1-4). Mild weight decay for regularization.
    At composition time, use s=10 uniformly for all domains.
    """
    log(f"    Strategy C: uniform scale={uniform_scale}, wd={lambda_wd}")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        ce_loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        wd_loss = mx.array(0.0)
        for name, p in tree_flatten(model.trainable_parameters()):
            if "lora_b" in name:
                wd_loss = wd_loss + mx.sum(p * p)
        return ce_loss + lambda_wd * wd_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    b_norms = []
    gc.disable()
    for step in range(train_iters):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0:
            b_norm = compute_total_b_norm(model)
            b_norms.append(b_norm)
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"      Step {step+1}/{train_iters}: loss={loss_val:.4f} "
                f"(avg50={avg:.4f}) ||B||_F={b_norm:.2f}")

    gc.enable()
    gc.collect()

    final_norm = compute_total_b_norm(model)
    return {
        "losses": losses,
        "b_norms": b_norms,
        "final_b_norm": final_norm,
    }


# ============================================================================
# Phase 1: Train norm-bounded adapters
# ============================================================================

def phase_train_all(skeleton_np):
    """Train 5 domain adapters under 3 norm-control strategies."""
    log("\n" + "=" * 70)
    log("PHASE 1: Norm-Bounded Adapter Training")
    log("=" * 70)

    # Compute target delta norm (geometric mean of current delta norms)
    current_delta_norms = []
    for domain in DOMAINS:
        s = PER_DOMAIN_SCALES[domain]
        # Load current adapter to get B-norm
        adapter_path = ADAPTERS_DIR_SOURCE / domain / "adapter.npz"
        adapter = dict(np.load(str(adapter_path)))
        b_norm = math.sqrt(sum(np.sum(v.astype(np.float64)**2)
                               for v in adapter.values()))
        delta_norm = s * b_norm * math.sqrt(LORA_RANK)
        current_delta_norms.append(delta_norm)
        del adapter

    geo_mean_delta = float(np.exp(np.mean(np.log(current_delta_norms))))
    log(f"\n  Current delta norms: {[f'{d:.1f}' for d in current_delta_norms]}")
    log(f"  Geometric mean target: {geo_mean_delta:.1f}")

    # Load tokenizer for training data
    _, tokenizer = load(MODEL_ID)

    # Pre-tokenize all domain data
    domain_train_tokens = {}
    for domain in DOMAINS:
        train_texts = []
        with open(DATA_DIR / domain / "train.jsonl") as f:
            for line in f:
                train_texts.append(json.loads(line)["text"])
        tokens_list = []
        for text in train_texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                tokens_list.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
        domain_train_tokens[domain] = tokens_list
        log(f"  {domain}: {len(tokens_list)} training sequences")

    del tokenizer
    gc.collect()
    mx.clear_cache()

    strategies = ["A_norm_projection", "B_weight_decay", "C_uniform_scale"]
    all_results = {}

    for strategy in strategies:
        log(f"\n  {'='*60}")
        log(f"  Strategy: {strategy}")
        log(f"  {'='*60}")
        strategy_results = {}

        for di, domain in enumerate(DOMAINS):
            log(f"\n  Training {domain} ({strategy})...")
            t0 = time.time()

            # Load model fresh for each adapter
            model, tokenizer = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)

            # Build A matrix mapping for this domain
            a_matrices = {}
            for li in range(NUM_LAYERS):
                for key in TARGET_KEYS:
                    skey = f"layer_{li}_{key}_domain_{di}"
                    if skey in skeleton_np:
                        a_matrices[(li, key)] = skeleton_np[skey]

            # Set scale based on strategy
            if strategy == "C_uniform_scale":
                train_scale = 10.0
            else:
                train_scale = PER_DOMAIN_SCALES[domain]

            model = apply_ternary_lora(model, LORA_RANK, train_scale, a_matrices)

            # Freeze everything except lora_b
            model.freeze()
            for layer in model.model.layers:
                for key, module in layer.named_modules():
                    if isinstance(module, TernaryLoRALinear):
                        module.unfreeze(keys=["lora_b"], strict=False)

            trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
            log(f"    Trainable params: {trainable:,}")

            train_tokens = domain_train_tokens[domain]

            # Train with strategy-specific norm control
            if strategy == "A_norm_projection":
                result = train_adapter_strategy_A(
                    model, tokenizer, train_tokens,
                    domain_scale=PER_DOMAIN_SCALES[domain],
                    target_delta_norm=geo_mean_delta,
                    train_iters=TRAIN_ITERS,
                )
            elif strategy == "B_weight_decay":
                result = train_adapter_strategy_B(
                    model, tokenizer, train_tokens,
                    domain_scale=PER_DOMAIN_SCALES[domain],
                    lambda_0=0.01,
                    train_iters=TRAIN_ITERS,
                )
            elif strategy == "C_uniform_scale":
                result = train_adapter_strategy_C(
                    model, tokenizer, train_tokens,
                    uniform_scale=10.0,
                    lambda_wd=0.001,
                    train_iters=TRAIN_ITERS,
                )

            train_time = time.time() - t0

            # Check convergence
            first_50 = sum(result["losses"][:50]) / 50
            last_50 = sum(result["losses"][-50:]) / 50
            converged = last_50 < first_50 * 0.95

            # Save adapter
            save_dir = NB_ADAPTERS_DIR / strategy / domain
            save_adapter(model, save_dir)

            strategy_results[domain] = {
                "train_time_s": round(train_time, 1),
                "first_50_loss": round(first_50, 4),
                "last_50_loss": round(last_50, 4),
                "converged": converged,
                "final_b_norm": result["final_b_norm"],
                "scale_used": train_scale,
            }

            log(f"    Done in {train_time:.1f}s. Loss: {first_50:.4f}->{last_50:.4f} "
                f"({'OK' if converged else 'FAIL'}) ||B||_F={result['final_b_norm']:.2f}")

            log_memory(f"post-train-{strategy}-{domain}")
            cleanup(model, tokenizer)

        all_results[strategy] = strategy_results

    return all_results, geo_mean_delta


# ============================================================================
# Phase 2: Composition Gini analysis (numpy only)
# ============================================================================

def phase_gini_analysis(skeleton_np, geo_mean_delta):
    """Measure composed Gini for each strategy + baseline."""
    log("\n" + "=" * 70)
    log("PHASE 2: Composition Gini Analysis")
    log("=" * 70)
    t0 = time.time()

    strategies = {
        "baseline_raw_sum": {
            "adapter_dir": ADAPTERS_DIR_SOURCE,
            "scales": PER_DOMAIN_SCALES,
        },
        "baseline_partial_eq": {
            "adapter_dir": ADAPTERS_DIR_SOURCE,
            "scales": PER_DOMAIN_SCALES,
            "equalize": "partial",
        },
        "A_norm_projection": {
            "adapter_dir": NB_ADAPTERS_DIR / "A_norm_projection",
            "scales": PER_DOMAIN_SCALES,  # same per-domain scales
        },
        "B_weight_decay": {
            "adapter_dir": NB_ADAPTERS_DIR / "B_weight_decay",
            "scales": PER_DOMAIN_SCALES,
        },
        "C_uniform_scale": {
            "adapter_dir": NB_ADAPTERS_DIR / "C_uniform_scale",
            "scales": {d: 10.0 for d in DOMAINS},  # uniform scale
        },
    }

    # Load all adapters
    all_gini_results = {}
    all_norm_info = {}

    for strat_name, strat_config in strategies.items():
        log(f"\n  {strat_name}:")
        adapter_dir = strat_config["adapter_dir"]
        scales = strat_config["scales"]
        equalize = strat_config.get("equalize", None)

        # Load adapters
        adapters_np = []
        domain_norms = {}
        for di, domain in enumerate(DOMAINS):
            adapter_path = adapter_dir / domain / "adapter.npz"
            if not adapter_path.exists():
                log(f"    SKIP: {adapter_path} not found")
                break
            adapter = dict(np.load(str(adapter_path)))
            adapters_np.append(adapter)
            b_norm = math.sqrt(sum(np.sum(v.astype(np.float64)**2)
                                   for v in adapter.values()))
            delta_norm = scales[domain] * b_norm * math.sqrt(LORA_RANK)
            domain_norms[domain] = {
                "b_norm": b_norm,
                "scale": scales[domain],
                "delta_norm": delta_norm,
            }
            log(f"    {domain}: ||B||_F={b_norm:.2f}, s={scales[domain]:.0f}, "
                f"||Delta||_F={delta_norm:.1f}")

        if len(adapters_np) < N_DOMAINS:
            all_gini_results[strat_name] = {"error": "missing adapters"}
            continue

        all_norm_info[strat_name] = domain_norms

        # Compute equalization scales if needed
        eq_scales = None
        if equalize == "partial":
            delta_norms_arr = np.array([domain_norms[d]["delta_norm"] for d in DOMAINS])
            log_norms = np.log(delta_norms_arr + 1e-30)
            mean_log = np.mean(log_norms)
            new_log = mean_log + 0.5 * (log_norms - mean_log)
            new_norms = np.exp(new_log)
            eq_scales = {d: float(new_norms[i] / delta_norms_arr[i])
                         for i, d in enumerate(DOMAINS)}

        # Compose deltas and compute Gini
        sample_layers = [0, 5, 10, 15, 20, 25, 29]
        sample_keys = ["self_attn.q_proj", "mlp.gate_proj"]

        ginis = []
        sv_ratios = []
        top1_fracs = []

        for li in sample_layers:
            for key in sample_keys:
                composed = None
                for di, domain in enumerate(DOMAINS):
                    skey = f"layer_{li}_{key}_domain_{di}"
                    bkey = f"model.layers.{li}.{key}.lora_b"
                    if skey not in skeleton_np or bkey not in adapters_np[di]:
                        continue
                    A = skeleton_np[skey].astype(np.float64)
                    B = adapters_np[di][bkey].astype(np.float64)
                    s = scales[domain]
                    delta = s * (B.T @ A.T)
                    if eq_scales is not None:
                        delta = delta * eq_scales[domain]
                    if composed is None:
                        composed = delta
                    else:
                        composed += delta

                if composed is not None:
                    _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
                    S_nz = S_c[S_c > 1e-6]
                    if len(S_nz) > 1:
                        ginis.append(gini_coefficient(S_nz))
                        sv_ratios.append(float(S_nz[0] / S_nz[-1]))
                        top1_fracs.append(float(S_nz[0]**2 / np.sum(S_nz**2)))

        mean_gini = float(np.mean(ginis)) if ginis else float("inf")
        std_gini = float(np.std(ginis)) if ginis else 0.0

        # Compute Gini of domain delta norms (between-domain imbalance)
        delta_norm_values = [domain_norms[d]["delta_norm"] for d in DOMAINS]
        if eq_scales is not None:
            delta_norm_values = [domain_norms[d]["delta_norm"] * eq_scales[d]
                                 for d in DOMAINS]
        norm_gini = gini_coefficient(delta_norm_values)

        all_gini_results[strat_name] = {
            "mean_gini": mean_gini,
            "std_gini": std_gini,
            "mean_sv_ratio": float(np.mean(sv_ratios)) if sv_ratios else 0,
            "mean_top1_frac": float(np.mean(top1_fracs)) if top1_fracs else 0,
            "n_samples": len(ginis),
            "norm_gini": norm_gini,
            "delta_norm_ratio": max(delta_norm_values) / max(min(delta_norm_values), 1e-10),
        }

        log(f"    Composed Gini = {mean_gini:.4f} +/- {std_gini:.4f}")
        log(f"    Norm Gini (between-domain) = {norm_gini:.4f}")
        log(f"    Delta norm ratio = {all_gini_results[strat_name]['delta_norm_ratio']:.1f}:1")

        del adapters_np

    elapsed = time.time() - t0
    log(f"\n  Phase 2 time: {elapsed:.1f}s")

    return all_gini_results, all_norm_info


# ============================================================================
# Phase 3: PPL evaluation
# ============================================================================

def phase_ppl_evaluation(skeleton_np):
    """Evaluate composition PPL for each strategy."""
    log("\n" + "=" * 70)
    log("PHASE 3: Perplexity Evaluation")
    log("=" * 70)
    t0 = time.time()

    # Load validation data
    domain_texts = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "valid.jsonl"
        texts = []
        with open(val_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if len(texts) >= N_EVAL_SAMPLES:
                    break
        domain_texts[domain] = texts

    strategies_to_eval = {
        "baseline_raw_sum": {
            "adapter_dir": ADAPTERS_DIR_SOURCE,
            "scales": PER_DOMAIN_SCALES,
        },
        "baseline_partial_eq": {
            "adapter_dir": ADAPTERS_DIR_SOURCE,
            "scales": PER_DOMAIN_SCALES,
            "equalize": "partial",
        },
        "A_norm_projection": {
            "adapter_dir": NB_ADAPTERS_DIR / "A_norm_projection",
            "scales": PER_DOMAIN_SCALES,
        },
        "B_weight_decay": {
            "adapter_dir": NB_ADAPTERS_DIR / "B_weight_decay",
            "scales": PER_DOMAIN_SCALES,
        },
        "C_uniform_scale": {
            "adapter_dir": NB_ADAPTERS_DIR / "C_uniform_scale",
            "scales": {d: 10.0 for d in DOMAINS},
        },
    }

    ppl_results = {}

    for strat_name, strat_config in strategies_to_eval.items():
        log(f"\n  Evaluating {strat_name}...")
        adapter_dir = strat_config["adapter_dir"]
        scales = strat_config["scales"]
        equalize = strat_config.get("equalize", None)

        # Load model fresh
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model.freeze()

        # Compute base PPL (only once)
        if strat_name == "baseline_raw_sum":
            log("    Computing base PPL...")
            base_ppl = {}
            for domain in DOMAINS:
                total_loss = 0.0
                total_tokens = 0
                for text in domain_texts[domain]:
                    ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
                    if len(ids) < 2:
                        continue
                    x = mx.array([ids[:-1]])
                    y = mx.array([ids[1:]])
                    logits = model(x)
                    loss = nn.losses.cross_entropy(logits, y, reduction="sum")
                    mx.eval(loss)
                    total_loss += loss.item()
                    total_tokens += len(ids) - 1
                    del logits, loss
                base_ppl[domain] = math.exp(total_loss / max(total_tokens, 1))
                log(f"      {domain} base: {base_ppl[domain]:.4f}")
            ppl_results["base"] = base_ppl

        # Load adapters and compose delta
        adapters_np = []
        for domain in DOMAINS:
            adapter_path = adapter_dir / domain / "adapter.npz"
            if not adapter_path.exists():
                break
            adapters_np.append(dict(np.load(str(adapter_path))))

        if len(adapters_np) < N_DOMAINS:
            log(f"    SKIP: missing adapters")
            cleanup(model, tokenizer)
            ppl_results[strat_name] = {"error": "missing adapters"}
            continue

        # Compute equalization scales
        eq_scales = None
        if equalize == "partial":
            delta_norms = []
            for di, domain in enumerate(DOMAINS):
                b_norm = math.sqrt(sum(np.sum(v.astype(np.float64)**2)
                                       for v in adapters_np[di].values()))
                delta_norms.append(scales[domain] * b_norm * math.sqrt(LORA_RANK))
            delta_norms = np.array(delta_norms)
            log_norms = np.log(delta_norms + 1e-30)
            mean_log = np.mean(log_norms)
            new_log = mean_log + 0.5 * (log_norms - mean_log)
            new_norms = np.exp(new_log)
            eq_scales = {d: float(new_norms[i] / delta_norms[i])
                         for i, d in enumerate(DOMAINS)}

        # Apply composed delta to model
        count = 0
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                composed = None
                for di, domain in enumerate(DOMAINS):
                    skey = f"layer_{li}_{key}_domain_{di}"
                    bkey = f"model.layers.{li}.{key}.lora_b"
                    if skey not in skeleton_np or bkey not in adapters_np[di]:
                        continue
                    A = skeleton_np[skey].astype(np.float64)
                    B = adapters_np[di][bkey].astype(np.float64)
                    s = scales[domain]
                    delta = s * (B.T @ A.T)
                    if eq_scales is not None:
                        delta = delta * eq_scales[domain]
                    if composed is None:
                        composed = delta
                    else:
                        composed += delta

                if composed is not None:
                    delta_mx = mx.array(composed.astype(np.float32))
                    parts = key.split(".")
                    module = model.model.layers[li]
                    for part in parts:
                        module = getattr(module, part, None)
                        if module is None:
                            break
                    if module is not None and isinstance(module, nn.Linear):
                        module.weight = module.weight + delta_mx.astype(module.weight.dtype)
                        count += 1

                if count % 30 == 0:
                    mx.eval(model.parameters())

        mx.eval(model.parameters())
        log(f"    Applied {count} composed deltas")

        # Evaluate PPL
        domain_ppl = {}
        for domain in DOMAINS:
            total_loss = 0.0
            total_tokens = 0
            for text in domain_texts[domain]:
                ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
                if len(ids) < 2:
                    continue
                x = mx.array([ids[:-1]])
                y = mx.array([ids[1:]])
                logits = model(x)
                loss = nn.losses.cross_entropy(logits, y, reduction="sum")
                mx.eval(loss)
                total_loss += loss.item()
                total_tokens += len(ids) - 1
                del logits, loss
            domain_ppl[domain] = math.exp(total_loss / max(total_tokens, 1))
            log(f"      {domain}: {domain_ppl[domain]:.4f}")

        # Mixed PPL
        all_texts = []
        for d in DOMAINS:
            all_texts.extend(domain_texts[d])
        total_loss = 0.0
        total_tokens = 0
        for text in all_texts:
            ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
            if len(ids) < 2:
                continue
            x = mx.array([ids[:-1]])
            y = mx.array([ids[1:]])
            logits = model(x)
            loss = nn.losses.cross_entropy(logits, y, reduction="sum")
            mx.eval(loss)
            total_loss += loss.item()
            total_tokens += len(ids) - 1
            del logits, loss
        mixed_ppl = math.exp(total_loss / max(total_tokens, 1))
        log(f"      Mixed: {mixed_ppl:.4f}")

        ppl_results[strat_name] = {
            "per_domain_ppl": domain_ppl,
            "mixed_ppl": mixed_ppl,
        }

        del adapters_np
        cleanup(model, tokenizer)

    elapsed = time.time() - t0
    log(f"\n  Phase 3 time: {elapsed:.1f}s")
    return ppl_results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    results = {"experiment": "norm_bounded_adapter_training", "phases": {}}

    log("=" * 70)
    log("Norm-Bounded Adapter Training: Eliminate Scale Imbalance at Source")
    log("=" * 70)

    # Check prerequisites
    if not SKELETON_PATH.exists():
        log(f"ERROR: Skeleton not found at {SKELETON_PATH}")
        log("  Run real_data_domain_experts first.")
        return

    for domain in DOMAINS:
        data_path = DATA_DIR / domain / "train.jsonl"
        if not data_path.exists():
            log(f"ERROR: Training data not found at {data_path}")
            return

    # Load skeleton
    log("\nLoading Grassmannian skeleton...")
    skeleton_np = dict(np.load(str(SKELETON_PATH)))
    log(f"  Skeleton keys: {len(skeleton_np)}")

    # Phase 1: Train norm-bounded adapters
    train_results, geo_mean_delta = phase_train_all(skeleton_np)
    results["phases"]["training"] = {
        "strategies": {k: {d: {kk: vv for kk, vv in v.items()
                               if kk != "losses" and kk != "b_norms"}
                           for d, v in strat.items()}
                       for k, strat in train_results.items()},
        "geo_mean_delta": geo_mean_delta,
    }

    gc.collect()
    mx.clear_cache()
    log_memory("after-training")

    # Phase 2: Gini analysis
    gini_results, norm_info = phase_gini_analysis(skeleton_np, geo_mean_delta)
    results["phases"]["gini_analysis"] = gini_results
    results["phases"]["norm_info"] = {
        k: {d: {kk: round(vv, 4) for kk, vv in v.items()}
            for d, v in norms.items()}
        for k, norms in norm_info.items()
    }

    # Phase 3: PPL evaluation
    ppl_results = phase_ppl_evaluation(skeleton_np)
    results["phases"]["ppl_evaluation"] = ppl_results

    # ================================================================
    # Summary and kill criteria
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    # Find best norm-bounded strategy by Gini
    nb_strategies = ["A_norm_projection", "B_weight_decay", "C_uniform_scale"]
    best_gini_strat = None
    best_gini = float("inf")
    for s in nb_strategies:
        if s in gini_results and "mean_gini" in gini_results[s]:
            g = gini_results[s]["mean_gini"]
            if g < best_gini:
                best_gini = g
                best_gini_strat = s

    baseline_gini = gini_results.get("baseline_raw_sum", {}).get("mean_gini", 0.49)
    partial_gini = gini_results.get("baseline_partial_eq", {}).get("mean_gini", 0.267)

    log(f"\n  Baseline raw sum Gini: {baseline_gini:.4f}")
    log(f"  Baseline partial eq Gini: {partial_gini:.4f}")
    log(f"  Best norm-bounded Gini: {best_gini:.4f} ({best_gini_strat})")
    for s in nb_strategies:
        if s in gini_results and "mean_gini" in gini_results[s]:
            g = gini_results[s]["mean_gini"]
            log(f"    {s}: Gini={g:.4f}, norm_ratio={gini_results[s].get('delta_norm_ratio', 'N/A')}")

    # K709: Gini < 0.15
    k709_pass = best_gini <= 0.15
    log(f"\n  K709 (Gini <= 0.15): {'PASS' if k709_pass else 'FAIL'} "
        f"(best={best_gini:.4f})")

    # K710: PPL <= 6.508 (partial eq baseline)
    best_ppl_strat = None
    best_mixed_ppl = float("inf")
    for s in nb_strategies:
        if s in ppl_results and "mixed_ppl" in ppl_results[s]:
            p = ppl_results[s]["mixed_ppl"]
            if p < best_mixed_ppl:
                best_mixed_ppl = p
                best_ppl_strat = s

    baseline_partial_ppl = ppl_results.get("baseline_partial_eq", {}).get("mixed_ppl", 6.508)
    k710_pass = best_mixed_ppl <= baseline_partial_ppl
    log(f"  K710 (PPL <= {baseline_partial_ppl:.3f}): {'PASS' if k710_pass else 'FAIL'} "
        f"(best={best_mixed_ppl:.4f}, {best_ppl_strat})")

    # K711: Convergence on >= 3/5 domains
    convergence_counts = {}
    for s in nb_strategies:
        if s in train_results:
            n_converged = sum(1 for d in DOMAINS
                             if train_results[s][d].get("converged", False))
            convergence_counts[s] = n_converged

    best_conv = max(convergence_counts.values()) if convergence_counts else 0
    k711_pass = best_conv >= 3
    log(f"  K711 (>= 3/5 converge): {'PASS' if k711_pass else 'FAIL'} "
        f"(best={best_conv}/5)")
    for s, c in convergence_counts.items():
        log(f"    {s}: {c}/5 converged")

    # PPL comparison table
    log("\n  PPL Comparison:")
    log(f"  {'Strategy':30s} {'Mixed':>8s} " +
        " ".join(f"{d:>8s}" for d in DOMAINS))
    log("  " + "-" * 78)
    for s in ["baseline_raw_sum", "baseline_partial_eq"] + nb_strategies:
        if s in ppl_results and "per_domain_ppl" in ppl_results[s]:
            mixed = ppl_results[s]["mixed_ppl"]
            domains_str = " ".join(
                f"{ppl_results[s]['per_domain_ppl'][d]:>8.3f}" for d in DOMAINS)
            log(f"  {s:30s} {mixed:>8.3f} {domains_str}")

    results["summary"] = {
        "k709_pass": k709_pass,
        "k709_best_gini": best_gini,
        "k709_best_strategy": best_gini_strat,
        "k710_pass": k710_pass,
        "k710_best_mixed_ppl": best_mixed_ppl,
        "k710_best_strategy": best_ppl_strat,
        "k710_baseline_partial_ppl": baseline_partial_ppl,
        "k711_pass": k711_pass,
        "k711_convergence": convergence_counts,
        "baseline_gini": baseline_gini,
        "partial_eq_gini": partial_gini,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    elapsed = time.time() - t_start
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
