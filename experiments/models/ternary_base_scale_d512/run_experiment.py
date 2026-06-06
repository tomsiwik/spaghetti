#!/usr/bin/env python3
"""Ternary Base Scale d=512: STE ternary training with BPE tokenizer on real text.

Kill criteria:
  K1 (id=223): Doesn't converge within 10K steps -> KILL
  K2 (id=224): PPL > 2x FP32 at same d=512 scale -> KILL
  K3 (id=225): Deadzone > 40% after training -> KILL

Success criteria:
  S1 (id=14): Ternary d=512 within 1.5x FP32 PPL with BPE tokenizer

Prior: exp_ternary_base_from_scratch_mlx achieved 1.003x PPL at d=256/vocab-27
       (overcapacity artifact). This experiment tests on real English text.
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

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_CACHE = EXPERIMENT_DIR / "data_cache"

# Architecture hyperparams
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
BLOCK_SIZE = 128
MLP_DIM = 4 * D_MODEL  # 2048
VOCAB_SIZE = 50257  # GPT-2 BPE

# Training hyperparams
FP32_STEPS = 5000
FP32_LR = 3e-4
TERNARY_STEPS = 10000
TERNARY_LR = 1e-3  # 3.3x FP32 (per LEARNINGS: STE needs larger LR)
BATCH_SIZE = 32
WARMUP_STEPS = 500

# Data size: ~2M tokens for training, ~200K for validation
TARGET_TRAIN_TOKENS = 2_000_000
TARGET_VAL_TOKENS = 200_000


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
# BitLinear: Ternary weights with STE (same as d=256 experiment)
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE in forward pass."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale

    def __call__(self, x):
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


# ============================================================================
# Transformer Architecture
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls=BitLinear):
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
    def __init__(self, n_embd: int, linear_cls=BitLinear):
        super().__init__()
        self.fc1 = linear_cls(n_embd, 4 * n_embd)
        self.fc2 = linear_cls(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))  # GELU instead of ReLU for real LM


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls=BitLinear):
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
    """GPT model parameterized by linear layer class (BitLinear or nn.Linear)."""
    def __init__(self, vocab_size: int, block_size: int, n_embd: int,
                 n_head: int, n_layer: int, linear_cls=BitLinear):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [Block(n_embd, n_head, linear_cls) for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        # For lm_head, use the same linear class
        if linear_cls == BitLinear:
            self.lm_head = BitLinear(n_embd, vocab_size)
        else:
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class FP32Linear(nn.Module):
    """Wrapper to make nn.Linear work as a drop-in with same constructor."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, x):
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight


# ============================================================================
# Data loading: FineWeb-Edu with GPT-2 BPE tokenizer
# ============================================================================

def phase_load_data():
    """Download and tokenize FineWeb-Edu subset using GPT-2 BPE."""
    import tiktoken

    print("\n" + "=" * 60)
    print("DATA LOADING: FineWeb-Edu with GPT-2 BPE")
    print("=" * 60)

    DATA_CACHE.mkdir(exist_ok=True)
    train_path = DATA_CACHE / "train_tokens.bin"
    val_path = DATA_CACHE / "val_tokens.bin"

    if train_path.exists() and val_path.exists():
        print("Loading cached tokenized data...")
        import numpy as np
        train_tokens = np.fromfile(str(train_path), dtype=np.int32)
        val_tokens = np.fromfile(str(val_path), dtype=np.int32)
        print(f"Train tokens: {len(train_tokens):,}")
        print(f"Val tokens: {len(val_tokens):,}")
        return train_tokens, val_tokens

    print("Downloading and tokenizing FineWeb-Edu (streaming)...")
    enc = tiktoken.get_encoding("gpt2")

    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    all_tokens = []
    total_tokens = 0
    target = TARGET_TRAIN_TOKENS + TARGET_VAL_TOKENS
    doc_count = 0

    for doc in ds:
        tokens = enc.encode(doc["text"])
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        doc_count += 1
        if doc_count % 100 == 0:
            print(f"  Processed {doc_count} docs, {total_tokens:,} tokens...")
        if total_tokens >= target:
            break

    print(f"Total: {doc_count} docs, {total_tokens:,} tokens")

    import numpy as np
    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Split: first TARGET_TRAIN_TOKENS for train, rest for val
    split_idx = min(TARGET_TRAIN_TOKENS, len(all_tokens) - TARGET_VAL_TOKENS)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:split_idx + TARGET_VAL_TOKENS]

    # Cache to disk
    train_tokens.tofile(str(train_path))
    val_tokens.tofile(str(val_path))
    print(f"Cached train ({len(train_tokens):,}) and val ({len(val_tokens):,}) tokens")

    return train_tokens, val_tokens


def get_batch(tokens, batch_size, block_size, rng):
    """Sample a random batch of sequences from token array."""
    import numpy as np
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
    """Compute fraction of weights that quantize to zero (deadzone metric)."""
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    total_ternary = 0
    zero_count = 0
    layer_zeros = {}

    for name, param in flat_params.items():
        # Skip embeddings, norms, and non-weight params
        if "wte" in name or "wpe" in name or "norm" in name:
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

        # Track per-layer stats
        layer_key = name.split(".")[0] + "." + name.split(".")[1] if "layers" in name else name
        if layer_key not in layer_zeros:
            layer_zeros[layer_key] = {"total": 0, "zeros": 0}
        layer_zeros[layer_key]["total"] += n
        layer_zeros[layer_key]["zeros"] += n_zero

    zero_frac = zero_count / total_ternary if total_ternary > 0 else 0
    return zero_frac, layer_zeros


def cosine_lr_schedule(step, total_steps, warmup_steps, base_lr):
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Phase 1: Train FP32 baseline
# ============================================================================

def phase_fp32_baseline(train_tokens, val_tokens):
    """Train FP32 baseline to establish PPL target."""
    print("\n" + "=" * 60)
    print("PHASE 1: FP32 Baseline Training (d=512, 8 layers)")
    print("=" * 60)

    model = GPTModel(
        vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
        linear_cls=FP32Linear,
    )
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"FP32 model params: {n_params:,}")
    log_memory("fp32-init")

    # Cosine LR with warmup
    schedule = opt.cosine_decay(FP32_LR, FP32_STEPS - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, FP32_LR, WARMUP_STEPS)
    lr_schedule = opt.join_schedules([warmup, schedule], [WARMUP_STEPS])
    optimizer = opt.Adam(learning_rate=lr_schedule)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    gc.disable()
    losses = []
    t0 = time.time()
    for step in range(1, FP32_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == FP32_STEPS:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"  step {step:5d}/{FP32_STEPS} | loss {loss_val:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | {elapsed:.0f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    fp32_ppl = compute_ppl(model, val_tokens)
    print(f"\nFP32 baseline PPL: {fp32_ppl:.2f}")
    print(f"Training time: {train_time:.1f}s")
    log_memory("fp32-done")

    result = {
        "fp32_ppl": fp32_ppl,
        "fp32_final_loss": losses[-1],
        "fp32_train_time_s": round(train_time, 1),
        "fp32_params": n_params,
        "fp32_loss_history_500": [losses[i - 1] for i in range(500, FP32_STEPS + 1, 500)],
    }

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Train ternary base model
# ============================================================================

def phase_ternary_base(train_tokens, val_tokens):
    """Train ternary base model with STE."""
    print("\n" + "=" * 60)
    print("PHASE 2: Ternary Base Training (STE, d=512, 8 layers)")
    print("=" * 60)

    model = GPTModel(
        vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
        linear_cls=BitLinear,
    )
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"Ternary model params: {n_params:,}")
    log_memory("ternary-init")

    # Higher LR for STE (per LEARNINGS)
    schedule = opt.cosine_decay(TERNARY_LR, TERNARY_STEPS - WARMUP_STEPS)
    warmup = opt.linear_schedule(1e-7, TERNARY_LR, WARMUP_STEPS)
    lr_schedule = opt.join_schedules([warmup, schedule], [WARMUP_STEPS])
    optimizer = opt.Adam(learning_rate=lr_schedule)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    random_baseline_loss = math.log(VOCAB_SIZE)  # ln(50257) = 10.825
    print(f"Random baseline loss (ln({VOCAB_SIZE})): {random_baseline_loss:.4f}")

    gc.disable()
    losses = []
    loss_below_random = False
    convergence_step = None
    zero_fraction_history = []
    t0 = time.time()

    for step in range(1, TERNARY_STEPS + 1):
        inputs, targets = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        # K1 check: convergence = loss goes below random baseline
        if not loss_below_random and loss_val < random_baseline_loss:
            convergence_step = step
            loss_below_random = True
            print(f"  [K1] Loss below random at step {step}: "
                  f"{loss_val:.4f} < {random_baseline_loss:.4f}")

        # Track zero fraction every 1K steps (deadzone evolution)
        if step % 1000 == 0:
            gc.enable()
            zero_frac, layer_zeros = compute_zero_fraction(model)
            zero_fraction_history.append({"step": step, "zero_frac": zero_frac})
            print(f"  [DEADZONE] step {step}: {zero_frac:.4f} "
                  f"({zero_frac * 100:.1f}% zeros)")
            gc.disable()

        if step % 500 == 0 or step == TERNARY_STEPS:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"  step {step:5d}/{TERNARY_STEPS} | loss {loss_val:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | {elapsed:.0f}s")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0

    # K1 evaluation
    k1_pass = loss_below_random
    print(f"\n[K1] Converged within {TERNARY_STEPS} steps: "
          f"{'PASS' if k1_pass else 'FAIL'}")
    if convergence_step:
        print(f"     First convergence at step {convergence_step}")

    # Compute PPL
    ternary_ppl = compute_ppl(model, val_tokens)
    print(f"Ternary base PPL: {ternary_ppl:.2f}")
    print(f"Training time: {train_time:.1f}s")

    # K3: Final deadzone check
    final_zero_frac, final_layer_zeros = compute_zero_fraction(model)
    k3_pass = final_zero_frac <= 0.40
    print(f"\n[K3] Final zero fraction: {final_zero_frac:.4f} "
          f"({final_zero_frac * 100:.1f}% zeros)")
    print(f"[K3] <= 40% zeros: {'PASS' if k3_pass else 'FAIL'}")

    # Per-layer deadzone report
    print("\nPer-layer zero fractions:")
    for layer_key in sorted(final_layer_zeros.keys()):
        info = final_layer_zeros[layer_key]
        frac = info["zeros"] / info["total"] if info["total"] > 0 else 0
        print(f"  {layer_key}: {frac:.4f} ({frac * 100:.1f}%)")

    log_memory("ternary-done")

    result = {
        "ternary_ppl": ternary_ppl,
        "ternary_final_loss": losses[-1],
        "ternary_train_time_s": round(train_time, 1),
        "ternary_params": n_params,
        "k1_pass": k1_pass,
        "k1_convergence_step": convergence_step,
        "k3_zero_fraction": final_zero_frac,
        "k3_pass": k3_pass,
        "k3_per_layer_zeros": {
            k: v["zeros"] / v["total"] if v["total"] > 0 else 0
            for k, v in final_layer_zeros.items()
        },
        "zero_fraction_history": zero_fraction_history,
        "loss_history_500": [losses[i - 1] for i in range(500, TERNARY_STEPS + 1, 500)],
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
    log_memory("after-fp32")

    # Phase 2: Ternary base
    ternary_results = phase_ternary_base(train_tokens, val_tokens)
    log_memory("after-ternary")

    # K2 check: PPL ratio
    fp32_ppl = fp32_results["fp32_ppl"]
    ternary_ppl = ternary_results["ternary_ppl"]
    ppl_ratio = ternary_ppl / fp32_ppl
    k2_pass = ternary_ppl < 2.0 * fp32_ppl
    print(f"\n[K2] Ternary PPL / FP32 PPL = {ppl_ratio:.3f}x (threshold: 2.0x)")
    print(f"[K2] {ternary_ppl:.2f} < {2.0 * fp32_ppl:.2f}: "
          f"{'PASS' if k2_pass else 'FAIL'}")

    # S1 check
    s1_pass = ternary_ppl < 1.5 * fp32_ppl
    print(f"[S1] Ternary PPL < 1.5x FP32: {ternary_ppl:.2f} < {1.5 * fp32_ppl:.2f}: "
          f"{'PASS' if s1_pass else 'FAIL'}")

    # Aggregate results
    total_time = time.time() - t0_total
    results = {
        "experiment": "ternary_base_scale_d512",
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "mlp_dim": MLP_DIM,
            "vocab_size": VOCAB_SIZE,
        },
        "data": {
            "source": "HuggingFaceFW/fineweb-edu (sample-10BT)",
            "tokenizer": "GPT-2 BPE (tiktoken)",
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
        },
        "fp32_baseline": fp32_results,
        "ternary_base": ternary_results,
        "ppl_ratio": ppl_ratio,
        "kill_criteria": {
            "K1_converges_within_10K": ternary_results["k1_pass"],
            "K2_ppl_within_2x": k2_pass,
            "K3_deadzone_below_40pct": ternary_results["k3_pass"],
        },
        "success_criteria": {
            "S1_ppl_within_1_5x": s1_pass,
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"FP32 baseline PPL:    {fp32_ppl:.2f}")
    print(f"Ternary base PPL:     {ternary_ppl:.2f} ({ppl_ratio:.3f}x FP32)")
    print(f"Deadzone fraction:    {ternary_results['k3_zero_fraction']:.4f} "
          f"({ternary_results['k3_zero_fraction'] * 100:.1f}%)")
    print()
    for k, v in results["kill_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    for k, v in results["success_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
