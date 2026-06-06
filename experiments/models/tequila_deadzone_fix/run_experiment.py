#!/usr/bin/env python3
"""Tequila Minima Reactivation: fix 31.3% deadzone trapping in ternary base.

Kill criteria:
  K1 (id=239): Reactivation doesn't reduce zero fraction below 20% -> KILL
  K2 (id=240): Reactivated model PPL worse than without -> KILL

Conditions (3 total, ~10 min each):
  1. BitLinear baseline (cold-start ternary with Extra RMSNorm, no reactivation)
  2. TequilaBitLinear lambda=1e-3 (paper default)
  3. TequilaBitLinear lambda=1e-2 (aggressive)

FP32 baseline PPL (344.09) from prior warmstart experiment (same arch, data).
Architecture: d=512, 4 layers, 8 heads.
Data: FineWeb-Edu (2M train / 200K val tokens, GPT-2 BPE).
"""

import gc
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data cache from prior d=512 experiment
DATA_CACHE = EXPERIMENT_DIR.parent / "ternary_base_scale_d512" / "data_cache"

# Architecture (matched to warmstart experiment)
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
BLOCK_SIZE = 128
MLP_DIM = 4 * D_MODEL  # 2048
VOCAB_SIZE = 50257  # GPT-2 BPE

# Training hyperparams
TOTAL_STEPS = 2000
TERNARY_LR = 1e-3  # Higher LR for STE (proven in prior experiments)
BATCH_SIZE = 32
WARMUP_STEPS = 100

# Prior FP32 baseline (from warmstart experiment, identical architecture + data)
FP32_PPL_PRIOR = 344.09

# Lambda values to test
LAMBDA_VALUES = [1e-3, 1e-2]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Model components
# ============================================================================

class BitLinear(nn.Module):
    """Standard ternary linear with Extra RMSNorm (baseline, no reactivation)."""
    def __init__(self, in_features: int, out_features: int, **_kwargs):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        self.pre_quant_norm = nn.RMSNorm(in_features)

    def __call__(self, x):
        x = self.pre_quant_norm(x)
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


class TequilaBitLinear(nn.Module):
    """Ternary linear with Tequila Minima Reactivation.

    Dead weights (those quantizing to 0) are reactivated as a dynamic adaptive
    bias: C_j = lambda * sum_{i in D_j} w_{j,i} for each output unit j.

    The bias is differentiable w.r.t. both lambda and the dead shadow weights,
    providing a clean gradient path that bypasses STE noise.
    """
    def __init__(self, in_features: int, out_features: int, lam_init: float = 1e-3):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        self.pre_quant_norm = nn.RMSNorm(in_features)
        # Learnable reactivation parameter (one scalar per layer)
        self.lam = mx.array([lam_init])

    def __call__(self, x):
        x = self.pre_quant_norm(x)
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1)

        # STE for live weights
        w_ste = w + mx.stop_gradient(w_q * alpha - w)

        # Standard ternary output
        y = x @ w_ste.T

        # Reactivation bias for dead weights
        # dead_mask: 1 where w_q == 0 (in deadzone), 0 elsewhere
        dead_mask = mx.stop_gradient((w_q == 0).astype(w.dtype))
        # Sum dead shadow weights along input dim for each output unit
        dead_sum = mx.sum(w * dead_mask, axis=1)  # [out_features]
        # Scale by learnable lambda
        bias = self.lam * dead_sum  # [out_features]

        return y + bias


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls, **linear_kwargs):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = linear_cls(n_embd, n_embd, **linear_kwargs)
        self.wk = linear_cls(n_embd, n_embd, **linear_kwargs)
        self.wv = linear_cls(n_embd, n_embd, **linear_kwargs)
        self.wo = linear_cls(n_embd, n_embd, **linear_kwargs)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int, linear_cls, **linear_kwargs):
        super().__init__()
        self.fc1 = linear_cls(n_embd, 4 * n_embd, **linear_kwargs)
        self.fc2 = linear_cls(4 * n_embd, n_embd, **linear_kwargs)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls, **linear_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, linear_cls, **linear_kwargs)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd, linear_cls, **linear_kwargs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """GPT model parameterized by linear layer class."""
    def __init__(self, vocab_size: int, block_size: int, n_embd: int,
                 n_head: int, n_layer: int, linear_cls, **linear_kwargs):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [Block(n_embd, n_head, linear_cls, **linear_kwargs)
                       for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ============================================================================
# Data loading
# ============================================================================

def phase_load_data():
    """Load cached tokenized FineWeb-Edu data from prior d=512 experiment."""
    print("\n" + "=" * 60, flush=True)
    print("DATA LOADING: Reusing cached FineWeb-Edu BPE tokens", flush=True)
    print("=" * 60, flush=True)

    train_path = DATA_CACHE / "train_tokens.bin"
    val_path = DATA_CACHE / "val_tokens.bin"

    if not train_path.exists():
        print("Cache not found, downloading fresh data...", flush=True)
        return _download_data()

    train_tokens = np.fromfile(str(train_path), dtype=np.int32)
    val_tokens = np.fromfile(str(val_path), dtype=np.int32)
    print(f"Train tokens: {len(train_tokens):,}", flush=True)
    print(f"Val tokens: {len(val_tokens):,}", flush=True)
    return train_tokens, val_tokens


def _download_data():
    """Download and tokenize FineWeb-Edu if cache is missing."""
    import tiktoken
    from datasets import load_dataset

    local_cache = EXPERIMENT_DIR / "data_cache"
    local_cache.mkdir(exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    all_tokens = []
    total_tokens = 0
    target = 2_200_000

    for doc in ds:
        tokens = enc.encode(doc["text"])
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        if total_tokens >= target:
            break

    all_tokens = np.array(all_tokens, dtype=np.int32)
    train_tokens = all_tokens[:2_000_000]
    val_tokens = all_tokens[2_000_000:2_200_000]

    train_tokens.tofile(str(local_cache / "train_tokens.bin"))
    val_tokens.tofile(str(local_cache / "val_tokens.bin"))
    print(f"Train: {len(train_tokens):,}, Val: {len(val_tokens):,}", flush=True)
    return train_tokens, val_tokens


def get_batch(tokens, batch_size, block_size, rng):
    max_start = len(tokens) - block_size - 1
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    inputs = np.stack([tokens[s:s + block_size] for s in starts])
    targets = np.stack([tokens[s + 1:s + block_size + 1] for s in starts])
    return mx.array(inputs), mx.array(targets)


# ============================================================================
# Metrics
# ============================================================================

def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def compute_ppl(model, val_tokens, n_batches=50, batch_size=32, block_size=128):
    """Compute perplexity on validation set."""
    rng = random.Random(0)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = get_batch(val_tokens, batch_size, block_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    avg_loss = total_loss / n_batches
    return math.exp(avg_loss)


def compute_zero_fraction(model):
    """Compute fraction of weights that quantize to zero."""
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    total_ternary = 0
    zero_count = 0

    for name, param in flat_params.items():
        if "wte" in name or "wpe" in name or "norm" in name or "lm_head" in name:
            continue
        if "lam" in name:
            continue
        if param.ndim < 2:
            continue

        alpha = mx.mean(mx.abs(param))
        w_scaled = param / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1)
        mx.eval(w_q)
        n = w_q.size
        n_zero = int(mx.sum(w_q == 0).item())
        total_ternary += n
        zero_count += n_zero

    return zero_count / total_ternary if total_ternary > 0 else 0


def compute_per_layer_zeros(model):
    """Compute zero fraction per layer for detailed analysis."""
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    layer_zeros = {}

    for name, param in flat_params.items():
        if "wte" in name or "wpe" in name or "norm" in name or "lm_head" in name:
            continue
        if "lam" in name:
            continue
        if param.ndim < 2:
            continue

        alpha = mx.mean(mx.abs(param))
        w_scaled = param / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1)
        mx.eval(w_q)
        n = w_q.size
        n_zero = int(mx.sum(w_q == 0).item())
        layer_zeros[name] = round(n_zero / n, 4)

    return layer_zeros


def get_lambda_values(model):
    """Extract learned lambda values from TequilaBitLinear layers."""
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    lam_vals = {}
    for name, param in flat_params.items():
        if "lam" in name:
            mx.eval(param)
            lam_vals[name] = round(float(param.item()), 6)
    return lam_vals


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_tokens, val_tokens, label):
    """Train model and return metrics. Zero fraction tracked at midpoint and end only."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"Training: {label}", flush=True)
    print(f"{'=' * 60}", flush=True)

    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}", flush=True)
    log_memory(f"{label}-init")

    schedule = opt.cosine_decay(TERNARY_LR, TOTAL_STEPS - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, TERNARY_LR, WARMUP_STEPS)
    lr_schedule = opt.join_schedules([warmup, schedule], [WARMUP_STEPS])
    optimizer = opt.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    gc.disable()
    losses = []
    zero_frac_mid = None
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step == TOTAL_STEPS // 2:
            zero_frac_mid = compute_zero_fraction(model)

        if step % 500 == 0 or step == TOTAL_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss_val:.4f} | "
                  f"{step / elapsed:.1f} steps/s | {elapsed:.0f}s", flush=True)

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens)
    zero_frac = compute_zero_fraction(model)
    per_layer = compute_per_layer_zeros(model)
    lam_vals = get_lambda_values(model)

    print(f"\n{label} PPL: {ppl:.2f} | zeros: {zero_frac:.3f} | time: {train_time:.1f}s",
          flush=True)
    if lam_vals:
        vals = list(lam_vals.values())
        print(f"  lambda avg: {sum(vals)/len(vals):.6f} | "
              f"min: {min(vals):.6f} | max: {max(vals):.6f}", flush=True)
    log_memory(f"{label}-done")

    result = {
        "label": label,
        "ppl": round(ppl, 2),
        "final_loss": round(losses[-1], 4),
        "train_time_s": round(train_time, 1),
        "params": n_params,
        "zero_fraction": round(zero_frac, 4),
        "zero_frac_midpoint": round(zero_frac_mid, 4) if zero_frac_mid is not None else None,
        "per_layer_zeros": per_layer,
        "lambda_values": lam_vals,
        "loss_curve": [round(losses[i - 1], 4) for i in range(100, TOTAL_STEPS + 1, 100)],
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase functions
# ============================================================================

def phase_bitlinear_baseline(train_tokens, val_tokens):
    """BitLinear baseline (standard ternary with Extra RMSNorm, no reactivation)."""
    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, BitLinear)
    return train_model(model, train_tokens, val_tokens, "BitLinear (no reactivation)")


def phase_tequila(train_tokens, val_tokens, lam_init):
    """TequilaBitLinear with specified lambda initialization."""
    label = f"Tequila lambda={lam_init}"
    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                     TequilaBitLinear, lam_init=lam_init)
    return train_model(model, train_tokens, val_tokens, label)


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    # Phase 0: Data loading
    train_tokens, val_tokens = phase_load_data()
    log_memory("after-data")

    # Phase 1: BitLinear baseline (no reactivation)
    bitlinear_results = phase_bitlinear_baseline(train_tokens, val_tokens)
    bitlinear_ppl = bitlinear_results["ppl"]
    bitlinear_zeros = bitlinear_results["zero_fraction"]
    log_memory("after-bitlinear")

    # Phase 2: Tequila conditions
    tequila_results = {}
    for lam in LAMBDA_VALUES:
        key = f"lambda_{lam}"
        tequila_results[key] = phase_tequila(train_tokens, val_tokens, lam)
        log_memory(f"after-tequila-{key}")

    # ========================================================================
    # Kill criteria assessment
    # ========================================================================
    print("\n" + "=" * 60, flush=True)
    print("KILL CRITERIA ASSESSMENT", flush=True)
    print("=" * 60, flush=True)

    # Find best Tequila result by PPL
    best_key = min(tequila_results, key=lambda k: tequila_results[k]["ppl"])
    best_tequila = tequila_results[best_key]
    best_ppl = best_tequila["ppl"]
    best_zeros = best_tequila["zero_fraction"]

    # Find best by zero fraction
    best_zero_key = min(tequila_results, key=lambda k: tequila_results[k]["zero_fraction"])
    lowest_zeros = tequila_results[best_zero_key]["zero_fraction"]

    # K1: Reactivation doesn't reduce zero fraction below 20% -> KILL
    k1_pass = lowest_zeros < 0.20
    print(f"\n[K1] Zero fraction reduction", flush=True)
    print(f"     BitLinear baseline zeros: {bitlinear_zeros:.4f} ({bitlinear_zeros*100:.1f}%)",
          flush=True)
    for k, v in tequila_results.items():
        print(f"     Tequila {k} zeros: {v['zero_fraction']:.4f} ({v['zero_fraction']*100:.1f}%)",
              flush=True)
    print(f"     Lowest Tequila zeros: {lowest_zeros:.4f} ({lowest_zeros*100:.1f}%)", flush=True)
    print(f"     Threshold: < 20%", flush=True)
    print(f"     Result: {'PASS' if k1_pass else 'FAIL'}", flush=True)

    # K2: Reactivated model PPL worse than without -> KILL
    k2_pass = best_ppl <= bitlinear_ppl
    print(f"\n[K2] PPL comparison", flush=True)
    print(f"     BitLinear baseline PPL: {bitlinear_ppl:.2f}", flush=True)
    for k, v in tequila_results.items():
        delta = (v['ppl'] - bitlinear_ppl) / bitlinear_ppl * 100
        print(f"     Tequila {k} PPL: {v['ppl']:.2f} ({delta:+.1f}%)", flush=True)
    print(f"     Best Tequila PPL: {best_ppl:.2f} ({best_key})", flush=True)
    print(f"     Result: {'PASS' if k2_pass else 'FAIL'} "
          f"({'better' if k2_pass else 'worse'} than baseline)", flush=True)

    # Summary
    fp32_ppl = FP32_PPL_PRIOR
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  FP32 baseline (prior):  PPL {fp32_ppl:.2f} (1.000x)", flush=True)
    print(f"  BitLinear baseline:     PPL {bitlinear_ppl:.2f} ({bitlinear_ppl/fp32_ppl:.3f}x) | "
          f"zeros {bitlinear_zeros:.4f}", flush=True)
    for k, v in tequila_results.items():
        ratio = v['ppl'] / fp32_ppl
        ppl_delta = (v['ppl'] - bitlinear_ppl) / bitlinear_ppl * 100
        zero_delta = (v['zero_fraction'] - bitlinear_zeros) / bitlinear_zeros * 100
        print(f"  Tequila {k}:  PPL {v['ppl']:.2f} ({ratio:.3f}x) | "
              f"zeros {v['zero_fraction']:.4f} | "
              f"PPL {ppl_delta:+.1f}% | zeros {zero_delta:+.1f}%", flush=True)

    print(f"\n  K1 (zeros < 20%): {'PASS' if k1_pass else 'FAIL'}", flush=True)
    print(f"  K2 (PPL <= baseline): {'PASS' if k2_pass else 'FAIL'}", flush=True)

    # ========================================================================
    # Save results
    # ========================================================================
    total_time = time.time() - t0_total
    results = {
        "experiment": "tequila_deadzone_fix",
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "mlp_dim": MLP_DIM,
            "vocab_size": VOCAB_SIZE,
            "extra_rmsnorm": True,
            "lm_head_fp32": True,
        },
        "training": {
            "total_steps": TOTAL_STEPS,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "ternary_lr": TERNARY_LR,
            "warmup_steps": WARMUP_STEPS,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
        },
        "data": {
            "source": "HuggingFaceFW/fineweb-edu (sample-10BT)",
            "tokenizer": "GPT-2 BPE",
        },
        "fp32_ppl_prior": fp32_ppl,
        "bitlinear_baseline": bitlinear_results,
        "tequila_results": tequila_results,
        "best_tequila": {
            "by_ppl": {"condition": best_key, "ppl": best_ppl, "zeros": best_zeros},
            "by_zeros": {"condition": best_zero_key, "zeros": lowest_zeros},
        },
        "kill_criteria": {
            "K1_zeros_below_20pct": k1_pass,
            "K1_lowest_zeros": lowest_zeros,
            "K2_ppl_not_worse": k2_pass,
            "K2_best_ppl": best_ppl,
            "K2_baseline_ppl": bitlinear_ppl,
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
