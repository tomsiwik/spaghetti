#!/usr/bin/env python3
"""
Experiment: Training Speed Optimization for LoRA on BitNet-2B-4T (MLX)

Profiles the existing training pipeline and applies optimizations:
  Phase 1: Baseline profiling (per-step timing breakdown)
  Phase 2: Apply optimizations incrementally:
    O1: Disable Python GC during training loop
    O2: Pre-tokenize all data (avoid per-step tokenizer calls)
    O3: Wrap step in function (release grads before eval)
    O4: mx.compile the training step function
    O5: Increase batch size (1 -> 4 -> 8) with proper batching
    O6: All optimizations combined

Kill criteria:
  K1 (#260): No bottleneck found (already near-optimal)
  K2 (implicit): Optimizations break convergence

Platform: Apple M5 Pro 48GB, MLX 0.31.1
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

# Use existing data from bitnet_2b_real_composition
SOURCE_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4

# Use a single domain for fair comparison
TEST_DOMAIN = "medical"
TRAIN_STEPS = 100  # Enough to measure but not waste time
WARMUP_STEPS = 5  # Warmup before timing


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e6
    cache = mx.get_cache_memory() / 1e6
    peak = mx.get_peak_memory() / 1e6
    log(f"[MEM {label}] active={active:.1f}MB cache={cache:.1f}MB peak={peak:.1f}MB")
    return {"active_mb": round(active, 1), "cache_mb": round(cache, 1), "peak_mb": round(peak, 1)}


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Model setup (reuse from bitnet_2b_real_composition)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix."""
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
    """Replace BitLinear with nn.Linear for differentiable training."""
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


def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA wrappers to all linear layers."""
    from mlx_lm.tuner.lora import LoRALinear
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
    log(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model):
    """Reset LoRA params."""
    from mlx_lm.tuner.lora import LoRALinear
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-scale, high=scale, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def setup_model():
    """Load model, unpack, apply LoRA, freeze base."""
    from mlx_lm import load
    log("Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    from mlx_lm.tuner.lora import LoRALinear
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")
    return model, tokenizer


def load_training_data(tokenizer):
    """Load and pre-tokenize training data for the test domain."""
    data_dir = SOURCE_DIR / "data" / TEST_DOMAIN
    train_path = data_dir / "train.jsonl"

    if not train_path.exists():
        log(f"  ERROR: {train_path} does not exist. Run bitnet_2b_real_composition first.")
        sys.exit(1)

    texts = []
    with open(train_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    # Pre-tokenize all data
    pre_tokenized = []
    for text in texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            toks = toks[:MAX_SEQ_LENGTH + 1]
            pre_tokenized.append(mx.array(toks))

    log(f"  Pre-tokenized {len(pre_tokenized)} sequences")
    return texts, pre_tokenized


# ===========================================================================
# Training variants
# ===========================================================================

def train_baseline(model, tokenizer, texts, n_steps, label="baseline"):
    """Original training loop from bitnet_2b_real_composition."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Per-step timing
    step_times = []
    losses = []

    for step in range(n_steps + WARMUP_STEPS):
        idx = step % len(texts)
        # Tokenize on-the-fly (baseline behavior)
        tokens = tokenizer.encode(texts[idx])
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if step >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    return step_times, losses[WARMUP_STEPS:]


def train_gc_disabled(model, tokenizer, texts, n_steps, label="gc_disabled"):
    """O1: Disable Python GC during training."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    step_times = []
    losses = []

    gc.disable()  # <-- KEY OPTIMIZATION
    for step in range(n_steps + WARMUP_STEPS):
        idx = step % len(texts)
        tokens = tokenizer.encode(texts[idx])
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if step >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    gc.enable()
    gc.collect()

    return step_times, losses[WARMUP_STEPS:]


def train_pretokenized(model, pre_tokenized, n_steps, label="pretokenized"):
    """O2: Use pre-tokenized data."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    step_times = []
    losses = []

    for step in range(n_steps + WARMUP_STEPS):
        idx = step % len(pre_tokenized)
        tokens = pre_tokenized[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if step >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    return step_times, losses[WARMUP_STEPS:]


def train_step_wrapped(model, pre_tokenized, n_steps, label="step_wrapped"):
    """O3: Wrap loss+grad+update in a function to release grads before eval.
    From fast-mlx guide: this releases grad references so memory can be reused."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    step_times = []
    losses = []

    for s in range(n_steps + WARMUP_STEPS):
        idx = s % len(pre_tokenized)
        tokens = pre_tokenized[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        t0 = time.perf_counter()
        loss = step(x, y)
        mx.eval(model.parameters(), optimizer.state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if s >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    return step_times, losses[WARMUP_STEPS:]


def train_compiled(model, pre_tokenized, n_steps, label="compiled"):
    """O4: mx.compile the step function.
    Compiles the entire forward+backward+optimizer step to eliminate
    Python dispatch overhead between ops."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Wrap in a function that releases grads
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Compile with model.state and optimizer.state as mutable inputs/outputs
    # This tells mx.compile these are state that changes between calls
    state = [model.state, optimizer.state]
    compiled_step = mx.compile(step, inputs=state, outputs=state)

    step_times = []
    losses = []

    for s in range(n_steps + WARMUP_STEPS):
        idx = s % len(pre_tokenized)
        tokens = pre_tokenized[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        t0 = time.perf_counter()
        loss = compiled_step(x, y)
        mx.eval(state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if s >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    return step_times, losses[WARMUP_STEPS:]


def train_batched(model, pre_tokenized, n_steps, batch_size=4, label="batched"):
    """O5: Larger batch sizes for better GPU utilization.
    Pads sequences to uniform length for batching."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    # Pad all sequences to the same length for batching
    max_len = MAX_SEQ_LENGTH
    padded_x = []
    padded_y = []
    pad_id = 0  # Use 0 as pad token

    for tokens in pre_tokenized:
        toks = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        x_toks = toks[:-1]
        y_toks = toks[1:]
        # Pad to max_len
        pad_len = max_len - len(x_toks)
        if pad_len > 0:
            x_toks = x_toks + [pad_id] * pad_len
            y_toks = y_toks + [-100] * pad_len  # -100 = ignore in loss
        else:
            x_toks = x_toks[:max_len]
            y_toks = y_toks[:max_len]
        padded_x.append(x_toks)
        padded_y.append(y_toks)

    # Pre-create batch arrays
    all_x = mx.array(padded_x)  # (N, max_len)
    all_y = mx.array(padded_y)  # (N, max_len)
    n_samples = all_x.shape[0]

    def loss_fn(model, x, y):
        logits = model(x)
        # Mask padding tokens (y == -100)
        mask = (y != -100).astype(mx.bfloat16)
        ce = nn.losses.cross_entropy(logits, mx.maximum(y, 0), reduction="none")
        # ce shape: (batch, seq_len)
        loss = mx.sum(ce * mask) / mx.maximum(mx.sum(mask), 1)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    step_times = []
    losses = []

    for s in range(n_steps + WARMUP_STEPS):
        # Sample a batch
        start_idx = (s * batch_size) % n_samples
        end_idx = start_idx + batch_size
        if end_idx > n_samples:
            # Wrap around
            indices = list(range(start_idx, n_samples)) + list(range(0, end_idx - n_samples))
            x = all_x[mx.array(indices)]
            y = all_y[mx.array(indices)]
        else:
            x = all_x[start_idx:end_idx]
            y = all_y[start_idx:end_idx]

        t0 = time.perf_counter()
        loss = step(x, y)
        mx.eval(model.parameters(), optimizer.state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if s >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    return step_times, losses[WARMUP_STEPS:]


def train_all_optimized(model, pre_tokenized, n_steps, batch_size=4, label="all_optimized"):
    """O6: All optimizations combined: GC off + pre-tokenized + step wrapped + compiled + batched."""
    zero_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    # Pad sequences for batching
    max_len = MAX_SEQ_LENGTH
    padded_x = []
    padded_y = []
    pad_id = 0

    for tokens in pre_tokenized:
        toks = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        x_toks = toks[:-1]
        y_toks = toks[1:]
        pad_len = max_len - len(x_toks)
        if pad_len > 0:
            x_toks = x_toks + [pad_id] * pad_len
            y_toks = y_toks + [-100] * pad_len
        else:
            x_toks = x_toks[:max_len]
            y_toks = y_toks[:max_len]
        padded_x.append(x_toks)
        padded_y.append(y_toks)

    all_x = mx.array(padded_x)
    all_y = mx.array(padded_y)
    n_samples = all_x.shape[0]

    def loss_fn(model, x, y):
        logits = model(x)
        mask = (y != -100).astype(mx.bfloat16)
        ce = nn.losses.cross_entropy(logits, mx.maximum(y, 0), reduction="none")
        loss = mx.sum(ce * mask) / mx.maximum(mx.sum(mask), 1)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state]
    compiled_step = mx.compile(step, inputs=state, outputs=state)

    step_times = []
    losses = []

    gc.disable()
    for s in range(n_steps + WARMUP_STEPS):
        start_idx = (s * batch_size) % n_samples
        end_idx = start_idx + batch_size
        if end_idx > n_samples:
            indices = list(range(start_idx, n_samples)) + list(range(0, end_idx - n_samples))
            x = all_x[mx.array(indices)]
            y = all_y[mx.array(indices)]
        else:
            x = all_x[start_idx:end_idx]
            y = all_y[start_idx:end_idx]

        t0 = time.perf_counter()
        loss = compiled_step(x, y)
        mx.eval(state, loss)
        t1 = time.perf_counter()

        loss_val = loss.item()
        losses.append(loss_val)

        if s >= WARMUP_STEPS:
            step_times.append(t1 - t0)

    gc.enable()
    gc.collect()

    return step_times, losses[WARMUP_STEPS:]


# ===========================================================================
# Analysis helpers
# ===========================================================================
def analyze_times(step_times, label):
    """Compute statistics for step times."""
    import numpy as np
    arr = np.array(step_times)
    stats = {
        "mean_ms": round(float(np.mean(arr)) * 1000, 2),
        "std_ms": round(float(np.std(arr)) * 1000, 2),
        "min_ms": round(float(np.min(arr)) * 1000, 2),
        "max_ms": round(float(np.max(arr)) * 1000, 2),
        "p50_ms": round(float(np.median(arr)) * 1000, 2),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 2),
        "total_s": round(float(np.sum(arr)), 3),
        "n_steps": len(step_times),
    }
    log(f"  {label}: mean={stats['mean_ms']:.1f}ms p50={stats['p50_ms']:.1f}ms "
        f"p95={stats['p95_ms']:.1f}ms total={stats['total_s']:.1f}s")
    return stats


def check_convergence(losses, label):
    """Check that training converged (last 20 losses < first 20)."""
    if len(losses) < 40:
        return True  # Too few steps to judge
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    converged = last < first * 0.98  # 2% improvement threshold
    log(f"  {label}: loss {first:.4f} -> {last:.4f} ({'converged' if converged else 'flat/diverged'})")
    return converged


# ===========================================================================
# Main experiment
# ===========================================================================
def phase_setup():
    """Load model and data."""
    model, tokenizer = setup_model()
    texts, pre_tokenized = load_training_data(tokenizer)
    log_memory("after-setup")
    return model, tokenizer, texts, pre_tokenized


def phase_profile(model, tokenizer, texts, pre_tokenized):
    """Run all training variants and collect timings."""
    results = {}

    # O0: Baseline
    log("\n--- O0: Baseline (original training loop) ---")
    times, losses = train_baseline(model, tokenizer, texts, TRAIN_STEPS)
    results["O0_baseline"] = analyze_times(times, "baseline")
    results["O0_baseline"]["converged"] = check_convergence(losses, "baseline")
    results["O0_baseline"]["final_loss"] = round(losses[-1], 4)
    mx.reset_peak_memory()
    log_memory("after-baseline")

    # O1: GC disabled
    log("\n--- O1: GC disabled ---")
    times, losses = train_gc_disabled(model, tokenizer, texts, TRAIN_STEPS)
    results["O1_gc_disabled"] = analyze_times(times, "gc_disabled")
    results["O1_gc_disabled"]["converged"] = check_convergence(losses, "gc_disabled")
    results["O1_gc_disabled"]["final_loss"] = round(losses[-1], 4)
    mx.reset_peak_memory()

    # O2: Pre-tokenized
    log("\n--- O2: Pre-tokenized ---")
    times, losses = train_pretokenized(model, pre_tokenized, TRAIN_STEPS)
    results["O2_pretokenized"] = analyze_times(times, "pretokenized")
    results["O2_pretokenized"]["converged"] = check_convergence(losses, "pretokenized")
    results["O2_pretokenized"]["final_loss"] = round(losses[-1], 4)
    mx.reset_peak_memory()

    # O3: Step wrapped (grad release)
    log("\n--- O3: Step wrapped (grad release) ---")
    times, losses = train_step_wrapped(model, pre_tokenized, TRAIN_STEPS)
    results["O3_step_wrapped"] = analyze_times(times, "step_wrapped")
    results["O3_step_wrapped"]["converged"] = check_convergence(losses, "step_wrapped")
    results["O3_step_wrapped"]["final_loss"] = round(losses[-1], 4)
    mx.reset_peak_memory()

    # O4: Compiled step
    log("\n--- O4: Compiled step (mx.compile) ---")
    try:
        times, losses = train_compiled(model, pre_tokenized, TRAIN_STEPS)
        results["O4_compiled"] = analyze_times(times, "compiled")
        results["O4_compiled"]["converged"] = check_convergence(losses, "compiled")
        results["O4_compiled"]["final_loss"] = round(losses[-1], 4)
    except Exception as e:
        log(f"  FAILED: {e}")
        results["O4_compiled"] = {"error": str(e)}
    mx.reset_peak_memory()

    # O5a: Batch size 4
    log("\n--- O5a: Batch size 4 ---")
    times, losses = train_batched(model, pre_tokenized, TRAIN_STEPS, batch_size=4)
    results["O5a_batch4"] = analyze_times(times, "batch4")
    results["O5a_batch4"]["converged"] = check_convergence(losses, "batch4")
    results["O5a_batch4"]["final_loss"] = round(losses[-1], 4)
    # Calculate samples/sec for fair comparison
    results["O5a_batch4"]["samples_per_sec"] = round(4000 / results["O5a_batch4"]["total_s"], 1)
    mx.reset_peak_memory()
    log_memory("after-batch4")

    # O5b: Batch size 8
    log("\n--- O5b: Batch size 8 ---")
    times, losses = train_batched(model, pre_tokenized, TRAIN_STEPS, batch_size=8)
    results["O5b_batch8"] = analyze_times(times, "batch8")
    results["O5b_batch8"]["converged"] = check_convergence(losses, "batch8")
    results["O5b_batch8"]["final_loss"] = round(losses[-1], 4)
    results["O5b_batch8"]["samples_per_sec"] = round(8000 / results["O5b_batch8"]["total_s"], 1)
    mx.reset_peak_memory()
    log_memory("after-batch8")

    # O6: All optimizations combined (batch=4)
    log("\n--- O6: All optimizations combined (batch=4) ---")
    try:
        times, losses = train_all_optimized(model, pre_tokenized, TRAIN_STEPS, batch_size=4)
        results["O6_all_optimized_b4"] = analyze_times(times, "all_optimized_b4")
        results["O6_all_optimized_b4"]["converged"] = check_convergence(losses, "all_optimized_b4")
        results["O6_all_optimized_b4"]["final_loss"] = round(losses[-1], 4)
        results["O6_all_optimized_b4"]["samples_per_sec"] = round(
            4000 / results["O6_all_optimized_b4"]["total_s"], 1)
    except Exception as e:
        log(f"  FAILED: {e}")
        results["O6_all_optimized_b4"] = {"error": str(e)}
    mx.reset_peak_memory()

    # O6b: All optimizations combined (batch=8)
    log("\n--- O6b: All optimizations combined (batch=8) ---")
    try:
        times, losses = train_all_optimized(model, pre_tokenized, TRAIN_STEPS, batch_size=8)
        results["O6b_all_optimized_b8"] = analyze_times(times, "all_optimized_b8")
        results["O6b_all_optimized_b8"]["converged"] = check_convergence(losses, "all_optimized_b8")
        results["O6b_all_optimized_b8"]["final_loss"] = round(losses[-1], 4)
        results["O6b_all_optimized_b8"]["samples_per_sec"] = round(
            8000 / results["O6b_all_optimized_b8"]["total_s"], 1)
    except Exception as e:
        log(f"  FAILED: {e}")
        results["O6b_all_optimized_b8"] = {"error": str(e)}
    mx.reset_peak_memory()

    return results


def main():
    t_start = time.time()
    log("=" * 70)
    log("Training Speed Optimization Experiment")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domain: {TEST_DOMAIN}")
    log(f"Steps: {TRAIN_STEPS} (+ {WARMUP_STEPS} warmup)")
    log(f"Max seq length: {MAX_SEQ_LENGTH}")
    log(f"LoRA rank: {LORA_RANK}")

    # Phase 1: Setup
    log("\n[Phase 1] Setup")
    model, tokenizer, texts, pre_tokenized = phase_setup()

    # Add baseline samples/sec
    baseline_key = "O0_baseline"

    # Phase 2: Profile all variants
    log("\n[Phase 2] Profiling training variants")
    profile_results = phase_profile(model, tokenizer, texts, pre_tokenized)

    # Add samples/sec to single-batch results
    for key in ["O0_baseline", "O1_gc_disabled", "O2_pretokenized", "O3_step_wrapped", "O4_compiled"]:
        if key in profile_results and "total_s" in profile_results[key]:
            # batch_size=1, so steps = samples
            profile_results[key]["samples_per_sec"] = round(
                TRAIN_STEPS / profile_results[key]["total_s"], 1)

    # Analysis
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    baseline_mean = profile_results["O0_baseline"]["mean_ms"]
    log(f"\nBaseline: {baseline_mean:.1f} ms/step")
    log(f"\nSpeedup vs baseline (ms/step):")

    speedups = {}
    for key, data in sorted(profile_results.items()):
        if "mean_ms" in data:
            speedup = baseline_mean / data["mean_ms"]
            speedups[key] = speedup
            conv = "OK" if data.get("converged", True) else "DIVERGED"
            log(f"  {key}: {data['mean_ms']:.1f} ms/step -> {speedup:.2f}x ({conv})")

    log(f"\nSamples/sec comparison:")
    for key, data in sorted(profile_results.items()):
        if "samples_per_sec" in data:
            log(f"  {key}: {data['samples_per_sec']:.1f} samples/sec")

    # Kill criteria assessment
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Is there a bottleneck? If best speedup > 1.2x, there was a bottleneck
    best_speedup = max(speedups.values()) if speedups else 1.0
    best_key = max(speedups, key=speedups.get) if speedups else "none"

    # For throughput comparison, use samples/sec
    baseline_sps = profile_results["O0_baseline"].get("samples_per_sec", 0)
    best_sps = 0
    best_sps_key = "none"
    for key, data in profile_results.items():
        sps = data.get("samples_per_sec", 0)
        if sps > best_sps:
            best_sps = sps
            best_sps_key = key

    throughput_speedup = best_sps / baseline_sps if baseline_sps > 0 else 1.0

    k1_pass = throughput_speedup > 1.2  # >20% improvement means bottleneck existed
    log(f"\nK1: Bottleneck found? Best throughput speedup = {throughput_speedup:.2f}x ({best_sps_key})")
    log(f"    {'PASS (bottleneck found, optimizations help)' if k1_pass else 'FAIL (already near-optimal)'}")

    # K2: Do optimizations break convergence?
    all_converged = all(
        data.get("converged", True)
        for data in profile_results.values()
        if "error" not in data
    )
    log(f"\nK2: Convergence preserved? {'PASS (all converged)' if all_converged else 'FAIL (some diverged)'}")

    total_time = time.time() - t_start

    # Save results
    final_results = {
        "experiment": "training_speed_optimization",
        "model": MODEL_ID,
        "domain": TEST_DOMAIN,
        "train_steps": TRAIN_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_rank": LORA_RANK,
        "profile_results": profile_results,
        "speedups": {k: round(v, 3) for k, v in speedups.items()},
        "best_speedup_ms_per_step": round(best_speedup, 3),
        "best_speedup_key": best_key,
        "best_throughput_speedup": round(throughput_speedup, 3),
        "best_throughput_key": best_sps_key,
        "baseline_samples_per_sec": baseline_sps,
        "best_samples_per_sec": best_sps,
        "k1_bottleneck_found": k1_pass,
        "k2_convergence_preserved": all_converged,
        "verdict": "SUPPORTED" if k1_pass and all_converged else "KILLED",
        "total_experiment_time_s": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(final_results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.0f}s")
    log(f"\nVERDICT: {final_results['verdict']}")

    # Cleanup
    cleanup(model, tokenizer)


if __name__ == "__main__":
    main()
