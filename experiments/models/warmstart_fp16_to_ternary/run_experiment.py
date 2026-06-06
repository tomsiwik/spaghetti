#!/usr/bin/env python3
"""Warm-start FP16 -> Ternary QAT at d=512 scale.

Kill criteria:
  K1 (id=266): Warm-start ternary PPL > 1.5x FP32 baseline -> KILL
  K2 (id=267): FP16->ternary switch causes non-recoverable loss spike -> KILL

Success criteria:
  S1 (id=31): Warm-start ternary within 1.2x FP32 PPL at d=512

Prior: Cold-start ternary at d=512 gave 2.78x FP32 PPL (KILLED).
       This experiment tests the standard production recipe: pretrain FP16,
       switch to ternary QAT at 10-20% of training steps.

Conditions:
  1. FP32 baseline (full training, 3000 steps)
  2. Cold-start ternary (full training, 3000 steps, weight_decay=0.01)
  3. Cold-start ternary no-WD (full training, 3000 steps, weight_decay=0.0)
  4. Warm-start 10%: FP16 for 300 steps, ternary QAT for 2700 steps
  5. Warm-start 20%: FP16 for 600 steps, ternary QAT for 2400 steps

Key recipe elements (from literature):
  - Extra RMSNorm before every quantized linear (arxiv 2505.08823)
  - Retain AdamW optimizer state during FP16->ternary transition
  - LR bump post-switch (restart cosine from higher LR)
  - Weight decay = 0 in final ternary phase
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

# Architecture: d=512 (key dimension for ternary), 4 layers for speed
# The d=512 dimension is what matters for ternary weight behavior;
# 4 layers keeps runtime manageable (~25 min per condition)
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
BLOCK_SIZE = 128
MLP_DIM = 4 * D_MODEL  # 2048
VOCAB_SIZE = 50257  # GPT-2 BPE

# Training hyperparams
TOTAL_STEPS = 3000
FP16_LR = 3e-4
TERNARY_LR = 1e-3  # Higher LR for STE (proven in prior experiments)
POST_SWITCH_LR = 5e-4  # LR bump after switch (between FP16 and ternary-from-scratch)
BATCH_SIZE = 32
WARMUP_STEPS = 150

# Warm-start switch points
SWITCH_FRACTIONS = [0.10, 0.20]  # 10% and 20% of total steps


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


# ============================================================================
# Model components
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE.
    Includes Extra RMSNorm before quantization (arxiv 2505.08823).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        # Extra RMSNorm: the single most impactful architectural modification
        # for ternary training (arxiv 2505.08823)
        self.pre_quant_norm = nn.RMSNorm(in_features)

    def __call__(self, x):
        # Apply extra RMSNorm to input before quantized matmul
        x = self.pre_quant_norm(x)
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


class FP32Linear(nn.Module):
    """FP32 linear layer with same interface as BitLinear."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, x):
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight


class WarmStartLinear(nn.Module):
    """Linear layer that starts as FP32 and can switch to ternary QAT.
    Includes Extra RMSNorm (always present, even in FP16 phase, so norm
    parameters are pretrained by the time we switch).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        self.pre_quant_norm = nn.RMSNorm(in_features)
        self._ternary_mode = False

    def __call__(self, x):
        x = self.pre_quant_norm(x)
        if self._ternary_mode:
            w = self.weight
            alpha = mx.mean(mx.abs(w))
            w_scaled = w / (alpha + 1e-7)
            w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
            w_ste = w + mx.stop_gradient(w_q - w)
            return x @ w_ste.T
        else:
            return x @ self.weight.T


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = linear_cls(n_embd, n_embd)
        self.wk = linear_cls(n_embd, n_embd)
        self.wv = linear_cls(n_embd, n_embd)
        self.wo = linear_cls(n_embd, n_embd)

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
    def __init__(self, n_embd: int, linear_cls):
        super().__init__()
        self.fc1 = linear_cls(n_embd, 4 * n_embd)
        self.fc2 = linear_cls(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, linear_cls)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd, linear_cls)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """GPT model parameterized by linear layer class."""
    def __init__(self, vocab_size: int, block_size: int, n_embd: int,
                 n_head: int, n_layer: int, linear_cls):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [Block(n_embd, n_head, linear_cls) for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        # LM head: always FP32 for stability (standard practice)
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
# Data loading (reuse cached data from prior experiment)
# ============================================================================

def phase_load_data():
    """Load cached tokenized FineWeb-Edu data from prior d=512 experiment."""
    print("\n" + "=" * 60)
    print("DATA LOADING: Reusing cached FineWeb-Edu BPE tokens")
    print("=" * 60)

    train_path = DATA_CACHE / "train_tokens.bin"
    val_path = DATA_CACHE / "val_tokens.bin"

    if not train_path.exists():
        # Fall back to downloading if cache not available
        print("Cache not found, downloading fresh data...")
        return _download_data()

    train_tokens = np.fromfile(str(train_path), dtype=np.int32)
    val_tokens = np.fromfile(str(val_path), dtype=np.int32)
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
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
    print(f"Train: {len(train_tokens):,}, Val: {len(val_tokens):,}")
    return train_tokens, val_tokens


def get_batch(tokens, batch_size, block_size, rng):
    """Sample a random batch of sequences from token array."""
    max_start = len(tokens) - block_size - 1
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    inputs = np.stack([tokens[s:s + block_size] for s in starts])
    targets = np.stack([tokens[s + 1:s + block_size + 1] for s in starts])
    return mx.array(inputs), mx.array(targets)


# ============================================================================
# Training utilities
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


def set_ternary_mode(model, enabled):
    """Switch all WarmStartLinear layers to ternary mode."""
    for layer in model.layers:
        # Attention projections
        for proj in [layer.attn.wq, layer.attn.wk, layer.attn.wv, layer.attn.wo]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled
        # MLP projections
        for proj in [layer.mlp.fc1, layer.mlp.fc2]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled


# ============================================================================
# Phase 1: FP32 Baseline
# ============================================================================

def phase_fp32_baseline(train_tokens, val_tokens):
    """Train FP32 baseline."""
    print("\n" + "=" * 60)
    print(f"PHASE 1: FP32 Baseline (d={D_MODEL}, {N_LAYERS}L, {TOTAL_STEPS} steps)")
    print("=" * 60)

    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, FP32Linear)
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}")
    log_memory("fp32-init")

    schedule = opt.cosine_decay(FP16_LR, TOTAL_STEPS - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, FP16_LR, WARMUP_STEPS)
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
    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == TOTAL_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss_val:.4f} | "
                  f"{step / elapsed:.1f} steps/s | {elapsed:.0f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens)
    print(f"FP32 baseline PPL: {ppl:.2f} | time: {train_time:.1f}s")
    log_memory("fp32-done")

    result = {
        "ppl": ppl,
        "final_loss": losses[-1],
        "train_time_s": round(train_time, 1),
        "params": n_params,
        "loss_curve": [losses[i - 1] for i in range(100, TOTAL_STEPS + 1, 100)],
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Cold-Start Ternary (with Extra RMSNorm)
# ============================================================================

def phase_cold_start_ternary(train_tokens, val_tokens):
    """Cold-start ternary with Extra RMSNorm (improved over prior experiment)."""
    print("\n" + "=" * 60)
    print(f"PHASE 2: Cold-Start Ternary (Extra RMSNorm, {TOTAL_STEPS} steps)")
    print("=" * 60)

    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, BitLinear)
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}")
    log_memory("cold-ternary-init")

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
    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == TOTAL_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss_val:.4f} | "
                  f"{step / elapsed:.1f} steps/s | {elapsed:.0f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens)
    zero_frac = compute_zero_fraction(model)
    print(f"Cold-start ternary PPL: {ppl:.2f} | zeros: {zero_frac:.3f} | time: {train_time:.1f}s")
    log_memory("cold-ternary-done")

    result = {
        "ppl": ppl,
        "final_loss": losses[-1],
        "train_time_s": round(train_time, 1),
        "params": n_params,
        "zero_fraction": zero_frac,
        "loss_curve": [losses[i - 1] for i in range(100, TOTAL_STEPS + 1, 100)],
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2b: Cold-Start Ternary with NO Weight Decay (ablation control)
# ============================================================================

def phase_cold_start_ternary_no_wd(train_tokens, val_tokens):
    """Cold-start ternary with Extra RMSNorm and weight_decay=0.0.

    This isolates the weight decay confound: the warm-start uses weight_decay=0.0
    in the ternary phase, but the original cold-start uses weight_decay=0.01.
    By running cold-start with weight_decay=0.0, we can determine whether the
    warm-start advantage comes from initialization/optimizer state, or merely
    from removing weight decay.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2b: Cold-Start Ternary NO-WD (weight_decay=0.0, {TOTAL_STEPS} steps)")
    print("=" * 60)

    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, BitLinear)
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}")
    log_memory("cold-ternary-nowd-init")

    schedule = opt.cosine_decay(TERNARY_LR, TOTAL_STEPS - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, TERNARY_LR, WARMUP_STEPS)
    lr_schedule = opt.join_schedules([warmup, schedule], [WARMUP_STEPS])
    optimizer = opt.AdamW(learning_rate=lr_schedule, weight_decay=0.0)

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
    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == TOTAL_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss_val:.4f} | "
                  f"{step / elapsed:.1f} steps/s | {elapsed:.0f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens)
    zero_frac = compute_zero_fraction(model)
    print(f"Cold-start ternary NO-WD PPL: {ppl:.2f} | zeros: {zero_frac:.3f} | time: {train_time:.1f}s")
    log_memory("cold-ternary-nowd-done")

    result = {
        "ppl": ppl,
        "final_loss": losses[-1],
        "train_time_s": round(train_time, 1),
        "params": n_params,
        "zero_fraction": zero_frac,
        "weight_decay": 0.0,
        "loss_curve": [losses[i - 1] for i in range(100, TOTAL_STEPS + 1, 100)],
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 3: Warm-Start FP16 -> Ternary QAT
# ============================================================================

def phase_warm_start(train_tokens, val_tokens, switch_fraction):
    """Warm-start: FP16 pretrain then switch to ternary QAT.

    Key recipe elements:
    1. Train in FP32 mode for switch_fraction * TOTAL_STEPS
    2. At switch point: enable ternary STE, retain optimizer state
    3. Bump learning rate, restart cosine schedule for remaining steps
    4. Zero weight decay in final ternary phase
    """
    switch_step = int(TOTAL_STEPS * switch_fraction)
    ternary_steps = TOTAL_STEPS - switch_step
    label = f"{int(switch_fraction * 100)}%"

    print("\n" + "=" * 60)
    print(f"PHASE 3-{label}: Warm-Start (FP16 {switch_step} steps -> Ternary {ternary_steps} steps)")
    print("=" * 60)

    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}")
    log_memory(f"warm-{label}-init")

    # Phase 1 optimizer: FP16 pretraining with standard LR and weight decay
    schedule = opt.cosine_decay(FP16_LR, switch_step - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, FP16_LR, WARMUP_STEPS)
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
    t0 = time.time()

    # === FP16 pretraining phase ===
    pre_switch_loss = None
    for step in range(1, switch_step + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == switch_step:
            elapsed = time.time() - t0
            print(f"  [FP16] step {step:5d}/{switch_step} | loss {loss_val:.4f} | "
                  f"{step / elapsed:.1f} steps/s")

    pre_switch_loss = losses[-1]
    pre_switch_ppl = compute_ppl(model, val_tokens)
    print(f"\n  [SWITCH] Pre-switch loss: {pre_switch_loss:.4f}, PPL: {pre_switch_ppl:.2f}")

    # === TRANSITION: Enable ternary mode ===
    # Key: retain optimizer state (momentum/variance from FP16 pretraining)
    set_ternary_mode(model, True)
    print(f"  [SWITCH] Ternary mode ENABLED. Optimizer state RETAINED.")

    # Create new optimizer for ternary phase with new LR schedule
    # KEY: transfer Adam state (momentum, variance) from FP16 optimizer
    # This is the critical difference vs cold-start: warm momentum/variance
    ternary_warmup_steps = min(100, ternary_steps // 10)
    ternary_schedule = opt.cosine_decay(POST_SWITCH_LR, ternary_steps - ternary_warmup_steps)
    ternary_warmup_sched = opt.linear_schedule(FP16_LR * 0.1, POST_SWITCH_LR, ternary_warmup_steps)
    new_lr_schedule = opt.join_schedules(
        [ternary_warmup_sched, ternary_schedule], [ternary_warmup_steps]
    )

    # Save old optimizer state before creating new optimizer
    saved_state = optimizer.state
    optimizer = opt.AdamW(learning_rate=new_lr_schedule, weight_decay=0.0)
    # Restore the Adam momentum/variance from FP16 phase
    optimizer.state = saved_state

    # === Ternary QAT phase ===
    post_switch_losses = []
    max_spike = 0.0
    spike_recovered = False

    for step in range(1, ternary_steps + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)
        post_switch_losses.append(loss_val)

        # Track loss spike relative to pre-switch
        spike = loss_val - pre_switch_loss
        if spike > max_spike:
            max_spike = spike

        # Check recovery: loss returns to within 10% of pre-switch
        if not spike_recovered and len(post_switch_losses) > 50:
            recent_avg = sum(post_switch_losses[-50:]) / 50
            if recent_avg <= pre_switch_loss * 1.10:
                spike_recovered = True
                print(f"  [RECOVERY] Loss recovered within at most {step} steps "
                      f"(first measurement point; avg {recent_avg:.4f} <= {pre_switch_loss * 1.10:.4f})")

        global_step = switch_step + step
        if step % 500 == 0 or step == ternary_steps:
            elapsed = time.time() - t0
            print(f"  [QAT] step {step:5d}/{ternary_steps} (global {global_step}) | "
                  f"loss {loss_val:.4f} | spike {spike:+.4f} | "
                  f"{global_step / elapsed:.1f} steps/s")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens)
    zero_frac = compute_zero_fraction(model)

    # K2 assessment: non-recoverable spike
    # A spike is "non-recoverable" if:
    # 1. Final loss is still > 20% above pre-switch loss, AND
    # 2. Loss never came back to within 10% of pre-switch
    final_loss = losses[-1]
    k2_non_recoverable = (final_loss > pre_switch_loss * 1.20) and (not spike_recovered)

    print(f"\nWarm-start {label} PPL: {ppl:.2f} | zeros: {zero_frac:.3f} | time: {train_time:.1f}s")
    print(f"  Pre-switch PPL: {pre_switch_ppl:.2f}")
    print(f"  Max loss spike: {max_spike:+.4f}")
    print(f"  Spike recovered: {spike_recovered}")
    print(f"  K2 non-recoverable: {k2_non_recoverable}")
    log_memory(f"warm-{label}-done")

    result = {
        "switch_fraction": switch_fraction,
        "switch_step": switch_step,
        "ppl": ppl,
        "final_loss": final_loss,
        "pre_switch_loss": pre_switch_loss,
        "pre_switch_ppl": pre_switch_ppl,
        "max_loss_spike": max_spike,
        "spike_recovered": spike_recovered,
        "k2_non_recoverable": k2_non_recoverable,
        "train_time_s": round(train_time, 1),
        "params": n_params,
        "zero_fraction": zero_frac,
        "loss_curve": [losses[i - 1] for i in range(100, TOTAL_STEPS + 1, 100)],
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    # Phase 0: Data loading
    train_tokens, val_tokens = phase_load_data()
    log_memory("after-data")

    # Phase 1: FP32 baseline
    fp32_results = phase_fp32_baseline(train_tokens, val_tokens)
    fp32_ppl = fp32_results["ppl"]
    log_memory("after-fp32")

    # Phase 2: Cold-start ternary (with Extra RMSNorm, weight_decay=0.01)
    cold_results = phase_cold_start_ternary(train_tokens, val_tokens)
    log_memory("after-cold-ternary")

    # Phase 2b: Cold-start ternary with NO weight decay (ablation control)
    cold_nowd_results = phase_cold_start_ternary_no_wd(train_tokens, val_tokens)
    log_memory("after-cold-ternary-nowd")

    # Phase 3: Warm-start conditions
    warm_results = {}
    for frac in SWITCH_FRACTIONS:
        label = f"{int(frac * 100)}pct"
        warm_results[label] = phase_warm_start(train_tokens, val_tokens, frac)
        log_memory(f"after-warm-{label}")

    # ========================================================================
    # Kill criteria assessment
    # ========================================================================
    print("\n" + "=" * 60)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 60)

    # Find best warm-start result
    best_warm_label = min(warm_results, key=lambda k: warm_results[k]["ppl"])
    best_warm = warm_results[best_warm_label]
    best_warm_ppl = best_warm["ppl"]
    best_warm_ratio = best_warm_ppl / fp32_ppl

    # K1: Warm-start ternary PPL > 1.5x FP32 -> KILL
    k1_pass = best_warm_ppl <= 1.5 * fp32_ppl
    print(f"\n[K1] Best warm-start PPL: {best_warm_ppl:.2f} "
          f"({best_warm_ratio:.3f}x FP32={fp32_ppl:.2f})")
    print(f"     Threshold: <= {1.5 * fp32_ppl:.2f} (1.5x)")
    print(f"     Result: {'PASS' if k1_pass else 'FAIL'}")

    # K2: Non-recoverable loss spike -> KILL
    k2_any_non_recoverable = any(
        warm_results[k]["k2_non_recoverable"] for k in warm_results
    )
    print(f"\n[K2] Non-recoverable spike in any warm-start: {k2_any_non_recoverable}")
    for k, v in warm_results.items():
        print(f"     {k}: spike={v['max_loss_spike']:+.4f}, "
              f"recovered={v['spike_recovered']}, "
              f"non_recoverable={v['k2_non_recoverable']}")
    k2_pass = not k2_any_non_recoverable
    print(f"     Result: {'PASS' if k2_pass else 'FAIL'}")

    # S1: Warm-start within 1.2x FP32
    s1_pass = best_warm_ppl <= 1.2 * fp32_ppl
    print(f"\n[S1] Best warm-start PPL within 1.2x FP32: "
          f"{best_warm_ppl:.2f} <= {1.2 * fp32_ppl:.2f}")
    print(f"     Result: {'PASS' if s1_pass else 'FAIL'}")

    # Cold-start comparison
    cold_ppl = cold_results["ppl"]
    cold_ratio = cold_ppl / fp32_ppl
    cold_nowd_ppl = cold_nowd_results["ppl"]
    cold_nowd_ratio = cold_nowd_ppl / fp32_ppl
    improvement_vs_cold = (cold_ppl - best_warm_ppl) / cold_ppl * 100
    improvement_vs_cold_nowd = (cold_nowd_ppl - best_warm_ppl) / cold_nowd_ppl * 100
    print(f"\n[COMPARISON]")
    print(f"  FP32 baseline:              {fp32_ppl:.2f} (1.000x)")
    print(f"  Cold-start ternary (wd=0.01): {cold_ppl:.2f} ({cold_ratio:.3f}x)")
    print(f"  Cold-start ternary (wd=0.0):  {cold_nowd_ppl:.2f} ({cold_nowd_ratio:.3f}x)")
    for k, v in warm_results.items():
        ratio = v["ppl"] / fp32_ppl
        print(f"  Warm-start {k}:          {v['ppl']:.2f} ({ratio:.3f}x)")
    print(f"  Warm-start improvement over cold (wd=0.01): {improvement_vs_cold:.1f}%")
    print(f"  Warm-start improvement over cold (wd=0.0):  {improvement_vs_cold_nowd:.1f}%")
    wd_effect = (cold_ppl - cold_nowd_ppl) / cold_ppl * 100
    print(f"  Weight decay effect on cold-start: {wd_effect:+.1f}% PPL change")

    # ========================================================================
    # Save results
    # ========================================================================
    total_time = time.time() - t0_total
    results = {
        "experiment": "warmstart_fp16_to_ternary",
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
            "fp16_lr": FP16_LR,
            "ternary_lr": TERNARY_LR,
            "post_switch_lr": POST_SWITCH_LR,
            "warmup_steps": WARMUP_STEPS,
            "optimizer": "AdamW",
        },
        "data": {
            "source": "HuggingFaceFW/fineweb-edu (sample-10BT)",
            "tokenizer": "GPT-2 BPE",
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
        },
        "fp32_baseline": fp32_results,
        "cold_start_ternary": cold_results,
        "cold_start_ternary_no_wd": cold_nowd_results,
        "warm_start_results": warm_results,
        "best_warm_start": {
            "condition": best_warm_label,
            "ppl": best_warm_ppl,
            "ratio_vs_fp32": best_warm_ratio,
        },
        "kill_criteria": {
            "K1_warmstart_ppl_within_1_5x": k1_pass,
            "K2_no_non_recoverable_spike": k2_pass,
        },
        "success_criteria": {
            "S1_warmstart_within_1_2x": s1_pass,
        },
        "weight_decay_ablation": {
            "cold_wd001_ppl": cold_ppl,
            "cold_wd000_ppl": cold_nowd_ppl,
            "warm_best_ppl": best_warm_ppl,
            "wd_effect_on_cold_pct": round(wd_effect, 2),
            "warm_improvement_vs_cold_wd001_pct": round(improvement_vs_cold, 2),
            "warm_improvement_vs_cold_wd000_pct": round(improvement_vs_cold_nowd, 2),
        },
        "cold_vs_warm_improvement_pct": improvement_vs_cold,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
