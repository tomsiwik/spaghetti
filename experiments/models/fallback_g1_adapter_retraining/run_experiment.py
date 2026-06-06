#!/usr/bin/env python3
"""Fallback G1: Retrain adapters with longer training + better composition.

Root cause from warmstart_scale_validation:
  - rank-16 on d=1024 = 34.6M trainable params (17% of base!) -> catastrophic overfitting
  - PPL went from 84 -> 1415 (17x WORSE) on domain data
  - 1/N averaging of broken deltas = even worse

Fix 1: rank=4, ATTENTION ONLY (no MLP), ~49K trainable (<0.4% of base)
Fix 2: cosine LR schedule with proper warmup, early stopping on val loss
Fix 3: Task Arithmetic + TIES-Merging instead of 1/N averaging
Fix 4: Domain-specific PPL as primary metric (not keyword density)

Kill criteria:
  K1 (id=515): Retrained adapters still don't improve PPL >= 10% on domain vs base -> FAIL

Platform: Apple M5 Pro 48GB, MLX
Scale: micro (d=256, 4 layers, ~13M params)
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

# Memory limits (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_CACHE = EXPERIMENT_DIR / "data_cache"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters"

# Architecture -- small for fast iteration
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 32
BLOCK_SIZE = 256
MLP_DIM = 4 * D_MODEL  # 1024
VOCAB_SIZE = 50257  # GPT-2 BPE

# Base training
BASE_TRAIN_STEPS = 4000
BASE_LR = 3e-4
BATCH_SIZE = 32
WARMUP_STEPS = 200

# Adapter training -- CONSERVATIVE to avoid overfitting (prior rank-16 caused 17x PPL blowup)
LORA_RANK = 4
LORA_ALPHA = 8.0  # alpha/rank = 2.0
ADAPTER_TRAIN_STEPS = 3000
ADAPTER_LR = 3e-4
ADAPTER_BATCH_SIZE = 32
ADAPTER_ONLY_ATTN = True  # MLP LoRA at small scale causes overfitting

# Data budgets
TARGET_TRAIN_TOKENS = 5_000_000
TARGET_VAL_TOKENS = 500_000
DOMAIN_TOKEN_TARGET = 500_000

# Domains
DOMAIN_KEYWORDS = {
    "science": ["biology", "chemistry", "physics", "molecule", "atom", "cell",
                "organism", "experiment", "hypothesis", "scientific"],
    "history": ["century", "empire", "war", "ancient", "civilization", "dynasty",
                "revolution", "colonial", "medieval", "historical"],
    "technology": ["software", "algorithm", "computer", "programming", "data",
                   "network", "digital", "system", "code", "technology"],
}

# Composition parameters
TASK_ARITH_LAMBDAS = [0.3, 0.5, 0.7, 1.0]
TIES_TRIM_FRACTION = 0.80  # trim bottom 80%


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
# Model components (from warmstart_scale_validation, scaled down to d=256)
# ============================================================================

class WarmStartLinear(nn.Module):
    """Linear that starts FP and can switch to ternary QAT."""
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
# LoRA adapter
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen base weight."""
    def __init__(self, base_weight, rank: int, alpha: float,
                 pre_norm_weight=None):
        super().__init__()
        out_features, in_features = base_weight.shape
        self.base_weight = base_weight  # frozen
        self.lora_A = mx.random.normal(shape=(rank, in_features)) * (1.0 / math.sqrt(in_features))
        self.lora_B = mx.zeros((out_features, rank))
        self.scale = alpha / rank
        self.has_pre_norm = pre_norm_weight is not None
        if self.has_pre_norm:
            self.pre_norm_weight = pre_norm_weight  # frozen
        self._ternary_mode = False

    def __call__(self, x):
        if self.has_pre_norm:
            norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            x = x * norm * self.pre_norm_weight

        w = self.base_weight
        if self._ternary_mode:
            alpha_q = mx.mean(mx.abs(w))
            w_scaled = w / (alpha_q + 1e-7)
            w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha_q
            w = w + mx.stop_gradient(w_q - w)

        base_out = x @ w.T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base_out + lora_out


# ============================================================================
# Data loading
# ============================================================================

def phase_load_data():
    """Download and tokenize FineWeb-Edu data."""
    print("\n" + "=" * 60)
    print("DATA LOADING: FineWeb-Edu BPE tokens")
    print("=" * 60)

    DATA_CACHE.mkdir(exist_ok=True)
    train_path = DATA_CACHE / "train_tokens.bin"
    val_path = DATA_CACHE / "val_tokens.bin"

    if train_path.exists() and val_path.exists():
        train_tokens = np.fromfile(str(train_path), dtype=np.int32)
        val_tokens = np.fromfile(str(val_path), dtype=np.int32)
        if len(train_tokens) >= TARGET_TRAIN_TOKENS:
            print(f"Using cached data: train={len(train_tokens):,}, val={len(val_tokens):,}")
            return train_tokens, val_tokens

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    total_target = TARGET_TRAIN_TOKENS + TARGET_VAL_TOKENS + 200_000
    all_tokens = []
    total = 0

    print(f"Downloading {total_target:,} tokens...")
    for i, doc in enumerate(ds):
        tokens = enc.encode(doc["text"])
        all_tokens.extend(tokens)
        total += len(tokens)
        if i % 1000 == 0:
            print(f"  {total:,} / {total_target:,} tokens ({100*total/total_target:.0f}%)")
        if total >= total_target:
            break

    all_tokens = np.array(all_tokens[:total_target], dtype=np.int32)
    train_tokens = all_tokens[:TARGET_TRAIN_TOKENS]
    val_tokens = all_tokens[TARGET_TRAIN_TOKENS:TARGET_TRAIN_TOKENS + TARGET_VAL_TOKENS]

    train_tokens.tofile(str(train_path))
    val_tokens.tofile(str(val_path))
    print(f"Saved: train={len(train_tokens):,}, val={len(val_tokens):,}")
    return train_tokens, val_tokens


def phase_load_domain_data():
    """Download domain-specific data for adapter training."""
    print("\n" + "=" * 60)
    print("DOMAIN DATA LOADING: Filtered FineWeb-Edu for 3 domains")
    print("=" * 60)

    domain_data_dir = DATA_CACHE / "domains"
    domain_data_dir.mkdir(exist_ok=True)

    all_cached = True
    for domain in DOMAIN_KEYWORDS:
        path = domain_data_dir / f"{domain}_tokens.bin"
        if not path.exists():
            all_cached = False
            break
        tokens = np.fromfile(str(path), dtype=np.int32)
        if len(tokens) < DOMAIN_TOKEN_TARGET:
            all_cached = False
            break

    if all_cached:
        domain_tokens = {}
        for domain in DOMAIN_KEYWORDS:
            path = domain_data_dir / f"{domain}_tokens.bin"
            domain_tokens[domain] = np.fromfile(str(path), dtype=np.int32)
            print(f"  {domain}: {len(domain_tokens[domain]):,} tokens (cached)")
        return domain_tokens

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    domain_buffers = {d: [] for d in DOMAIN_KEYWORDS}
    domain_counts = {d: 0 for d in DOMAIN_KEYWORDS}
    docs_scanned = 0

    for doc in ds:
        docs_scanned += 1
        text_lower = doc["text"].lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain_counts[domain] >= DOMAIN_TOKEN_TARGET:
                continue
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 3:
                tokens = enc.encode(doc["text"])
                domain_buffers[domain].extend(tokens)
                domain_counts[domain] += len(tokens)

        if docs_scanned % 5000 == 0:
            print(f"  Scanned {docs_scanned} docs: " +
                  ", ".join(f"{d}={c:,}" for d, c in domain_counts.items()))

        if all(c >= DOMAIN_TOKEN_TARGET for c in domain_counts.values()):
            break

    domain_tokens = {}
    for domain in DOMAIN_KEYWORDS:
        arr = np.array(domain_buffers[domain][:DOMAIN_TOKEN_TARGET], dtype=np.int32)
        path = domain_data_dir / f"{domain}_tokens.bin"
        arr.tofile(str(path))
        domain_tokens[domain] = arr
        print(f"  {domain}: {len(arr):,} tokens saved")

    return domain_tokens


def get_batch(tokens, batch_size, block_size, rng):
    max_start = len(tokens) - block_size - 1
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    inputs = np.stack([tokens[s:s + block_size] for s in starts])
    targets = np.stack([tokens[s + 1:s + block_size + 1] for s in starts])
    return mx.array(inputs), mx.array(targets)


# ============================================================================
# Utilities
# ============================================================================

def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def compute_ppl(model, tokens, n_batches=50, batch_size=16):
    """Compute perplexity on tokens."""
    rng = random.Random(0)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = get_batch(tokens, batch_size, BLOCK_SIZE, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    return math.exp(total_loss / n_batches)


def set_ternary_mode(model, enabled):
    for layer in model.layers:
        for proj in [layer.attn.wq, layer.attn.wk, layer.attn.wv, layer.attn.wo]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled
        for proj in [layer.mlp.fc1, layer.mlp.fc2]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled


# ============================================================================
# Phase 1: Train small base model with warm-start ternary
# ============================================================================

def phase_train_base(train_tokens, val_tokens):
    """Train d=256 warm-start ternary base model."""
    print("\n" + "=" * 60)
    print(f"PHASE 1: Warm-Start Base (d={D_MODEL}, {N_LAYERS}L, {BASE_TRAIN_STEPS} steps)")
    print("=" * 60)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "base.npz"

    if ckpt_path.exists():
        print(f"Loading cached base checkpoint: {ckpt_path}")
        model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
        weights = dict(mx.load(str(ckpt_path)))
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        set_ternary_mode(model, True)
        ppl = compute_ppl(model, val_tokens, n_batches=30)
        n_params = count_params(model)
        print(f"  Loaded base: PPL={ppl:.2f}, params={n_params:,}")
        result = {"ppl": ppl, "params": n_params, "cached": True}
        cleanup(model)
        return result

    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Params: {n_params:,}")
    log_memory("base-init")

    # 10% FP then 90% ternary (proven recipe)
    switch_step = int(BASE_TRAIN_STEPS * 0.10)
    ternary_steps = BASE_TRAIN_STEPS - switch_step

    # FP phase optimizer
    schedule = opt.cosine_decay(BASE_LR, switch_step - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, BASE_LR, WARMUP_STEPS)
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
    t0 = time.time()

    # FP phase
    for step in range(1, switch_step + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 100 == 0 or step == switch_step:
            print(f"  [FP] step {step}/{switch_step} | loss {loss.item():.4f} | "
                  f"{step/(time.time()-t0):.1f} steps/s")

    # Switch to ternary
    set_ternary_mode(model, True)
    print(f"  [SWITCH] Ternary mode ON at step {switch_step}")

    # Ternary phase optimizer
    del optimizer
    gc.collect()
    ternary_warmup = min(100, ternary_steps // 10)
    t_schedule = opt.cosine_decay(BASE_LR * 1.5, ternary_steps - ternary_warmup)
    t_warmup = opt.linear_schedule(BASE_LR * 0.1, BASE_LR * 1.5, ternary_warmup)
    t_lr = opt.join_schedules([t_warmup, t_schedule], [ternary_warmup])
    optimizer = opt.AdamW(learning_rate=t_lr, weight_decay=0.01)

    for step in range(1, ternary_steps + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        global_step = switch_step + step
        if step % 500 == 0 or step == ternary_steps:
            elapsed = time.time() - t0
            print(f"  [TERN] step {global_step}/{BASE_TRAIN_STEPS} | "
                  f"loss {loss.item():.4f} | {global_step/elapsed:.1f} steps/s")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, val_tokens, n_batches=30)
    print(f"Base model PPL: {ppl:.2f} | time: {train_time:.1f}s")

    # Save checkpoint
    mx.savez(str(ckpt_path), **dict(nn.utils.tree_flatten(model.parameters())))
    print(f"  Saved checkpoint: {ckpt_path}")

    result = {"ppl": ppl, "params": n_params, "train_time_s": round(train_time, 1), "cached": False}
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Train domain adapters with AGGRESSIVE hyperparams
# ============================================================================

def phase_train_adapter(domain, domain_tokens, val_tokens):
    """Train a single domain adapter with strong hyperparams."""
    print(f"\n--- Training {domain} adapter (rank={LORA_RANK}, lr={ADAPTER_LR}, "
          f"steps={ADAPTER_TRAIN_STEPS}) ---")

    ADAPTER_DIR.mkdir(exist_ok=True)
    adapter_path = ADAPTER_DIR / f"{domain}_lora.npz"

    # Load base model
    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
    weights = dict(mx.load(str(CHECKPOINT_DIR / "base.npz")))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    set_ternary_mode(model, True)

    # Measure base PPL on domain data BEFORE adapter
    base_domain_ppl = compute_ppl(model, domain_tokens, n_batches=30, batch_size=16)
    base_val_ppl = compute_ppl(model, val_tokens, n_batches=20, batch_size=16)
    print(f"  Base domain PPL: {base_domain_ppl:.2f}, base val PPL: {base_val_ppl:.2f}")

    # LoRA targets (attention only to avoid overfitting at small scale)
    lora_targets = [('attn', ['wq', 'wk', 'wv', 'wo'])]
    if not ADAPTER_ONLY_ATTN:
        lora_targets.append(('mlp', ['fc1', 'fc2']))
    for layer in model.layers:
        for attr_group in lora_targets:
            parent_name, attrs = attr_group
            parent = getattr(layer, parent_name)
            for attr in attrs:
                old = getattr(parent, attr)
                lora = LoRALinear(
                    base_weight=old.weight,
                    rank=LORA_RANK,
                    alpha=LORA_ALPHA,
                    pre_norm_weight=old.pre_quant_norm.weight if hasattr(old, 'pre_quant_norm') else None,
                )
                lora._ternary_mode = True
                setattr(parent, attr, lora)

    mx.eval(model.parameters())

    # Freeze everything, then unfreeze only LoRA params
    model.freeze()
    for layer in model.layers:
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                lora = getattr(parent, attr)
                # Unfreeze LoRA A and B, keep base_weight frozen
                lora.unfreeze(keys=["lora_A", "lora_B"])

    n_trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = count_params(model)
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    if n_trainable == 0:
        print("  WARNING: No trainable parameters! Emergency unfreeze all LoRA params")
        for layer in model.layers:
            for attr_group in lora_targets:
                parent = getattr(layer, attr_group[0])
                for attr in attr_group[1]:
                    lora = getattr(parent, attr)
                    lora.unfreeze()
                    lora.freeze(keys=["base_weight", "pre_norm_weight"])
        n_trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
        print(f"  After emergency unfreeze: {n_trainable:,} trainable")

    # Optimizer with cosine schedule
    schedule = opt.cosine_decay(ADAPTER_LR, ADAPTER_TRAIN_STEPS - 200)
    warmup = opt.linear_schedule(1e-6, ADAPTER_LR, 200)
    lr_schedule = opt.join_schedules([warmup, schedule], [200])
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
    t0 = time.time()
    losses = []
    grad_norms = []
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 5  # stop if val loss doesn't improve for 5 checks (every 500 steps)
    actual_steps = 0

    for step in range(1, ADAPTER_TRAIN_STEPS + 1):
        inputs, targets = get_batch(domain_tokens, ADAPTER_BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)
        actual_steps = step

        if step % 500 == 0 or step == ADAPTER_TRAIN_STEPS:
            # Check validation loss for early stopping
            val_ppl_check = compute_ppl(model, val_tokens, n_batches=10, batch_size=16)
            val_loss_check = math.log(val_ppl_check)

            elapsed = time.time() - t0
            improving = val_loss_check < best_val_loss
            if improving:
                best_val_loss = val_loss_check
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  step {step}/{ADAPTER_TRAIN_STEPS} | loss {loss_val:.4f} | "
                  f"val_ppl {val_ppl_check:.1f} | {step/elapsed:.1f} steps/s | "
                  f"patience {patience_counter}/{PATIENCE}")
            grad_norms.append({"step": step, "loss": loss_val, "val_ppl": val_ppl_check})

            if patience_counter >= PATIENCE and step >= 1000:
                print(f"  EARLY STOPPING at step {step} (val loss not improving)")
                break

    gc.enable()
    gc.collect()

    train_time = time.time() - t0

    # Measure adapted PPL
    adapted_domain_ppl = compute_ppl(model, domain_tokens, n_batches=30, batch_size=16)
    adapted_val_ppl = compute_ppl(model, val_tokens, n_batches=20, batch_size=16)

    improvement_pct = 100 * (base_domain_ppl - adapted_domain_ppl) / base_domain_ppl
    val_degradation_pct = 100 * (adapted_val_ppl - base_val_ppl) / base_val_ppl

    print(f"  Domain PPL: {base_domain_ppl:.2f} -> {adapted_domain_ppl:.2f} "
          f"({improvement_pct:+.1f}%)")
    print(f"  Val PPL: {base_val_ppl:.2f} -> {adapted_val_ppl:.2f} "
          f"({val_degradation_pct:+.1f}%)")

    # Compute delta ratio ||B*A||/||W|| for each layer
    delta_ratios = []
    for layer in model.layers:
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                lora = getattr(parent, attr)
                delta = lora.lora_B @ lora.lora_A * lora.scale
                mx.eval(delta)
                delta_norm = mx.sqrt(mx.sum(delta * delta)).item()
                base_norm = mx.sqrt(mx.sum(lora.base_weight * lora.base_weight)).item()
                ratio = delta_norm / (base_norm + 1e-10)
                delta_ratios.append(ratio)
                del delta

    mean_delta_ratio = np.mean(delta_ratios)
    min_delta_ratio = np.min(delta_ratios)
    max_delta_ratio = np.max(delta_ratios)
    print(f"  Delta ratio ||B*A*s||/||W||: mean={mean_delta_ratio:.4f}, "
          f"min={min_delta_ratio:.4f}, max={max_delta_ratio:.4f}")

    # Save adapter weights (LoRA A and B only)
    adapter_weights = {}
    for li, layer in enumerate(model.layers):
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                lora = getattr(parent, attr)
                key = f"layers.{li}.{attr_group[0]}.{attr}"
                adapter_weights[f"{key}.lora_A"] = lora.lora_A
                adapter_weights[f"{key}.lora_B"] = lora.lora_B

    mx.savez(str(adapter_path), **adapter_weights)
    print(f"  Saved adapter: {adapter_path}")

    result = {
        "domain": domain,
        "base_domain_ppl": base_domain_ppl,
        "base_val_ppl": base_val_ppl,
        "adapted_domain_ppl": adapted_domain_ppl,
        "adapted_val_ppl": adapted_val_ppl,
        "improvement_pct": round(improvement_pct, 2),
        "val_degradation_pct": round(val_degradation_pct, 2),
        "trainable_params": n_trainable,
        "actual_steps": actual_steps,
        "train_time_s": round(train_time, 1),
        "final_loss": losses[-1],
        "delta_ratio_mean": round(mean_delta_ratio, 6),
        "delta_ratio_min": round(min_delta_ratio, 6),
        "delta_ratio_max": round(max_delta_ratio, 6),
        "grad_norms": grad_norms,
        "loss_trajectory": [losses[i] for i in range(499, len(losses), 500)],
    }

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 3: Composition methods
# ============================================================================

def phase_composition(val_tokens, domain_tokens_dict):
    """Test three composition methods: 1/N averaging, Task Arithmetic, TIES."""
    print("\n" + "=" * 60)
    print("PHASE 3: Composition Methods")
    print("=" * 60)

    domains = list(DOMAIN_KEYWORDS.keys())

    # Load all adapter deltas from disk
    adapter_deltas = {}  # domain -> {key: B@A * scale}
    for domain in domains:
        adapter_path = ADAPTER_DIR / f"{domain}_lora.npz"
        if not adapter_path.exists():
            print(f"  WARNING: {domain} adapter not found, skipping")
            continue
        raw = dict(mx.load(str(adapter_path)))
        deltas = {}
        keys_A = sorted([k for k in raw if k.endswith(".lora_A")])
        for k_a in keys_A:
            base_key = k_a.replace(".lora_A", "")
            k_b = base_key + ".lora_B"
            A = raw[k_a]
            B = raw[k_b]
            delta = (B @ A) * (LORA_ALPHA / LORA_RANK)
            mx.eval(delta)
            deltas[base_key] = delta
        adapter_deltas[domain] = deltas
        print(f"  Loaded {domain}: {len(deltas)} delta matrices")

    # Load base model
    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
    weights = dict(mx.load(str(CHECKPOINT_DIR / "base.npz")))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    set_ternary_mode(model, True)

    # Measure base PPL on each domain
    base_ppls = {}
    for domain in domains:
        ppl = compute_ppl(model, domain_tokens_dict[domain], n_batches=30, batch_size=16)
        base_ppls[domain] = ppl
        print(f"  Base PPL on {domain}: {ppl:.2f}")
    base_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=16)
    print(f"  Base val PPL: {base_val_ppl:.2f}")

    # Get base weight dict for composition (only layers that have adapters)
    lora_targets = [('attn', ['wq', 'wk', 'wv', 'wo'])]
    if not ADAPTER_ONLY_ATTN:
        lora_targets.append(('mlp', ['fc1', 'fc2']))

    base_weight_map = {}
    for li, layer in enumerate(model.layers):
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                module = getattr(parent, attr)
                key = f"layers.{li}.{attr_group[0]}.{attr}"
                base_weight_map[key] = module.weight

    N = len(adapter_deltas)
    composition_results = {}

    # --- Method A: 1/N Averaging ---
    print("\n  --- Method A: 1/N Averaging ---")
    for li, layer in enumerate(model.layers):
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                module = getattr(parent, attr)
                key = f"layers.{li}.{attr_group[0]}.{attr}"
                avg_delta = sum(adapter_deltas[d][key] for d in adapter_deltas) / N
                module.weight = base_weight_map[key] + avg_delta
    mx.eval(model.parameters())

    avg_ppls = {}
    for domain in domains:
        ppl = compute_ppl(model, domain_tokens_dict[domain], n_batches=30, batch_size=16)
        avg_ppls[domain] = ppl
        improvement = 100 * (base_ppls[domain] - ppl) / base_ppls[domain]
        print(f"    {domain}: {ppl:.2f} (vs base {base_ppls[domain]:.2f}, {improvement:+.1f}%)")
    avg_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=16)
    composition_results["avg_1_N"] = {
        "domain_ppls": {d: round(v, 2) for d, v in avg_ppls.items()},
        "val_ppl": round(avg_val_ppl, 2),
        "improvements": {d: round(100*(base_ppls[d]-avg_ppls[d])/base_ppls[d], 2) for d in domains},
    }

    # --- Method B: Task Arithmetic (various lambda) ---
    for lam in TASK_ARITH_LAMBDAS:
        print(f"\n  --- Method B: Task Arithmetic (lambda={lam}) ---")
        for li, layer in enumerate(model.layers):
            for attr_group in lora_targets:
                parent = getattr(layer, attr_group[0])
                for attr in attr_group[1]:
                    module = getattr(parent, attr)
                    key = f"layers.{li}.{attr_group[0]}.{attr}"
                    total_delta = sum(adapter_deltas[d][key] for d in adapter_deltas)
                    module.weight = base_weight_map[key] + lam * total_delta
        mx.eval(model.parameters())

        ta_ppls = {}
        for domain in domains:
            ppl = compute_ppl(model, domain_tokens_dict[domain], n_batches=30, batch_size=16)
            ta_ppls[domain] = ppl
            improvement = 100 * (base_ppls[domain] - ppl) / base_ppls[domain]
            print(f"    {domain}: {ppl:.2f} (vs base {base_ppls[domain]:.2f}, {improvement:+.1f}%)")
        ta_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=16)
        composition_results[f"task_arith_lambda_{lam}"] = {
            "domain_ppls": {d: round(v, 2) for d, v in ta_ppls.items()},
            "val_ppl": round(ta_val_ppl, 2),
            "improvements": {d: round(100*(base_ppls[d]-ta_ppls[d])/base_ppls[d], 2) for d in domains},
        }

    # --- Method C: TIES-Merging ---
    print(f"\n  --- Method C: TIES-Merging (trim={TIES_TRIM_FRACTION}) ---")
    for li, layer in enumerate(model.layers):
        for attr_group in lora_targets:
            parent = getattr(layer, attr_group[0])
            for attr in attr_group[1]:
                module = getattr(parent, attr)
                key = f"layers.{li}.{attr_group[0]}.{attr}"

                # Stack all deltas for this weight
                deltas = [adapter_deltas[d][key] for d in adapter_deltas]

                # Step 1: Trim bottom p% of each delta
                trimmed = []
                for d in deltas:
                    flat = mx.abs(d).reshape(-1)
                    mx.eval(flat)
                    threshold_idx = int(TIES_TRIM_FRACTION * flat.size)
                    sorted_vals = mx.sort(flat)
                    mx.eval(sorted_vals)
                    threshold = sorted_vals[threshold_idx].item()
                    mask = mx.abs(d) >= threshold
                    trimmed.append(d * mask)
                    del flat, sorted_vals

                # Step 2: Elect majority sign
                sign_sum = sum(mx.sign(t) for t in trimmed)
                elected_sign = mx.sign(sign_sum)

                # Step 3: Disjoint merge -- keep only values matching elected sign
                merged = mx.zeros_like(trimmed[0])
                count = mx.zeros_like(trimmed[0])
                for t in trimmed:
                    match = mx.sign(t) == elected_sign
                    merged = merged + mx.abs(t) * match
                    count = count + match.astype(mx.float32)
                mx.eval(merged, count)
                avg_merged = elected_sign * merged / mx.maximum(count, mx.array(1.0))

                module.weight = base_weight_map[key] + avg_merged
                del trimmed, sign_sum, elected_sign, merged, count, avg_merged

    mx.eval(model.parameters())

    ties_ppls = {}
    for domain in domains:
        ppl = compute_ppl(model, domain_tokens_dict[domain], n_batches=30, batch_size=16)
        ties_ppls[domain] = ppl
        improvement = 100 * (base_ppls[domain] - ppl) / base_ppls[domain]
        print(f"    {domain}: {ppl:.2f} (vs base {base_ppls[domain]:.2f}, {improvement:+.1f}%)")
    ties_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=16)
    composition_results["ties_merge"] = {
        "domain_ppls": {d: round(v, 2) for d, v in ties_ppls.items()},
        "val_ppl": round(ties_val_ppl, 2),
        "improvements": {d: round(100*(base_ppls[d]-ties_ppls[d])/base_ppls[d], 2) for d in domains},
    }

    result = {
        "base_ppls": {d: round(v, 2) for d, v in base_ppls.items()},
        "base_val_ppl": round(base_val_ppl, 2),
        "methods": composition_results,
    }

    cleanup(model)
    # Clean adapter deltas
    del adapter_deltas
    gc.collect()
    mx.clear_cache()

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Load data
    train_tokens, val_tokens = phase_load_data()
    domain_tokens = phase_load_domain_data()
    log_memory("after-data")

    # Phase 1: Train base model
    base_result = phase_train_base(train_tokens, val_tokens)
    log_memory("after-base")

    # Phase 2: Train domain adapters
    adapter_results = {}
    for domain in DOMAIN_KEYWORDS:
        result = phase_train_adapter(domain, domain_tokens[domain], val_tokens)
        adapter_results[domain] = result
        log_memory(f"after-{domain}-adapter")

    # Phase 3: Composition
    composition_result = phase_composition(val_tokens, domain_tokens)
    log_memory("after-composition")

    # Assess kill criteria
    improvements = [adapter_results[d]["improvement_pct"] for d in DOMAIN_KEYWORDS]
    k1_pass = all(imp >= 10.0 for imp in improvements)
    k1_any_pass = any(imp >= 10.0 for imp in improvements)

    # Check delta ratios
    delta_ratios = [adapter_results[d]["delta_ratio_mean"] for d in DOMAIN_KEYWORDS]
    delta_ratio_pass = all(r > 0.01 for r in delta_ratios)

    # Find best composition method
    best_method = None
    best_avg_improvement = -999
    for method_name, method_data in composition_result["methods"].items():
        avg_imp = np.mean(list(method_data["improvements"].values()))
        if avg_imp > best_avg_improvement:
            best_avg_improvement = avg_imp
            best_method = method_name

    total_time = time.time() - t0

    results = {
        "experiment": "fallback_g1_adapter_retraining",
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "adapter_lr": ADAPTER_LR,
            "adapter_steps": ADAPTER_TRAIN_STEPS,
        },
        "base": base_result,
        "adapters": adapter_results,
        "composition": composition_result,
        "kill_criteria": {
            "K1_all_adapters_10pct_improvement": k1_pass,
            "K1_any_adapter_10pct_improvement": k1_any_pass,
            "improvements_pct": {d: adapter_results[d]["improvement_pct"] for d in DOMAIN_KEYWORDS},
            "delta_ratio_target_met": delta_ratio_pass,
            "delta_ratios": {d: adapter_results[d]["delta_ratio_mean"] for d in DOMAIN_KEYWORDS},
        },
        "best_composition_method": best_method,
        "best_composition_avg_improvement": round(best_avg_improvement, 2),
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED: {RESULTS_FILE}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\nK1 (all adapters >= 10% improvement): {'PASS' if k1_pass else 'FAIL'}")
    imp_strs = [f"{d}={adapter_results[d]['improvement_pct']:.1f}%" for d in DOMAIN_KEYWORDS]
    print(f"  Improvements: {', '.join(imp_strs)}")
    print(f"Delta ratio target (>0.01): {'PASS' if delta_ratio_pass else 'FAIL'}")
    ratio_strs = [f"{d}={adapter_results[d]['delta_ratio_mean']:.4f}" for d in DOMAIN_KEYWORDS]
    print(f"  Ratios: {', '.join(ratio_strs)}")
    print(f"Best composition: {best_method} (avg improvement: {best_avg_improvement:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
