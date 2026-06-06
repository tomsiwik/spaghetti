#!/usr/bin/env python3
"""
P5.A0: LoRI Sparse-Mask Composition

Implements LoRI (arXiv:2504.07448): frozen random A + disjoint-masked B.
Tests zero parameter interference and quality vs standard LoRA.

Kill criteria (DB IDs):
  K1264: max|cos| < 1e-4 between any adapter pair (should be exactly 0)
  K1265: Quality >= 90% of standard LoRA on math domain at matched param budget
  K1266: 5-adapter composition < 5pp PPL degradation vs solo

Grounded by:
  - arXiv:2504.07448 (LoRI): frozen A + sparse B, 95% fewer params
  - Finding #59: Prior LoRI test on BitNet-2B was null (A not frozen, ternary)
  - Finding #440: Grassmannian composition works at r=4, N=100
"""

import gc
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DATA_DIR = EXPERIMENT_DIR.parent / "bitnet_lori_sparse_b" / "data"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
TOTAL_RANK = 30
N_ADAPTERS = 5
RANK_PER_ADAPTER = TOTAL_RANK // N_ADAPTERS  # 6
LORA_SCALE = 20.0
LORA_KEYS = ["self_attn.q_proj"]
DOMAINS = ["medical", "math", "legal", "python", "creative"]
SEQ_LEN = 256
BATCH_SIZE = 4
TRAIN_ITERS = 20 if IS_SMOKE else 200
EVAL_SAMPLES = 10 if IS_SMOKE else 50
LR = 1e-4
SEED = 42

# Baseline LoRA rank: same effective dims per adapter
BASELINE_RANK = RANK_PER_ADAPTER  # 6

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


def cleanup():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_texts(domain: str, split: str = "train") -> list[str]:
    path = DATA_DIR / domain / f"{split}.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def make_batches(tokenizer, texts: list[str], batch_size: int, max_len: int):
    """Pre-tokenize and batch all texts."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 2:
            all_tokens.append(tokens[:max_len])

    # Shuffle deterministically
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(all_tokens))
    all_tokens = [all_tokens[i] for i in indices]

    batches = []
    for start in range(0, len(all_tokens) - batch_size + 1, batch_size):
        batch_tokens = all_tokens[start : start + batch_size]
        max_in_batch = max(len(t) for t in batch_tokens)
        # Pad to 32-aligned length
        padded_len = 32 * ((max_in_batch + 31) // 32)
        arr = np.zeros((batch_size, padded_len), dtype=np.int32)
        lengths = []
        for i, tokens in enumerate(batch_tokens):
            arr[i, : len(tokens)] = tokens
            lengths.append(len(tokens))
        batches.append((mx.array(arr), lengths))
    return batches


# ══════════════════════════════════════════════════════════════════════════════
# LoRI mask utilities
# ══════════════════════════════════════════════════════════════════════════════

def make_mask(adapter_idx: int) -> mx.array:
    """Binary mask for adapter i: active rows [i*s : (i+1)*s] of rank-r B."""
    mask_np = np.zeros((TOTAL_RANK, 1), dtype=np.float32)
    start = adapter_idx * RANK_PER_ADAPTER
    end = start + RANK_PER_ADAPTER
    mask_np[start:end, 0] = 1.0
    return mx.array(mask_np)


def apply_b_mask(model, mask: mx.array):
    """Zero out non-masked rows of lora_b across all LoRA layers."""
    updates = []
    for layer in model.layers:
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_b = module.lora_b * mask
                updates.append(module.lora_b)
    if updates:
        mx.eval(updates)


def reset_b(model):
    """Reset all lora_b to zeros."""
    updates = []
    for layer in model.layers:
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
                updates.append(module.lora_b)
    if updates:
        mx.eval(updates)


def freeze_lora_a(model):
    """Freeze lora_a in all LoRA layers (only lora_b trainable)."""
    for layer in model.layers:
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.freeze(keys=["lora_a"])


def get_shared_a(model) -> dict[str, mx.array]:
    """Extract A matrices from all LoRA layers."""
    a_dict = {}
    for idx, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                a_dict[f"layers.{idx}.{name}"] = mx.array(module.lora_a)
    return a_dict


def set_shared_a(model, a_dict: dict[str, mx.array]):
    """Restore shared A matrices (important: same A for all adapters)."""
    updates = []
    for idx, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                key = f"layers.{idx}.{name}"
                module.lora_a = a_dict[key]
                updates.append(module.lora_a)
    if updates:
        mx.eval(updates)


def get_b_weights(model) -> dict[str, mx.array]:
    """Extract all lora_b weights."""
    b_dict = {}
    for idx, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                b_dict[f"layers.{idx}.{name}"] = mx.array(module.lora_b)
    return b_dict


def set_b_weights(model, b_dict: dict[str, mx.array]):
    """Set lora_b weights from dict."""
    updates = []
    for idx, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                key = f"layers.{idx}.{name}"
                module.lora_b = b_dict[key]
                updates.append(module.lora_b)
    if updates:
        mx.eval(updates)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def ntp_loss(model, batch: mx.array, lengths: list[int]):
    """Next-token prediction loss with padding mask."""
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]

    # Build padding mask
    mask_np = np.zeros(targets.shape, dtype=np.float32)
    for i, l in enumerate(lengths):
        mask_np[i, : l - 1] = 1.0
    mask = mx.array(mask_np)

    loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum()


def train_lori_adapter(
    model, tokenizer, domain: str, adapter_idx: int, n_iters: int
) -> tuple[float, dict[str, mx.array]]:
    """Train one LoRI adapter (frozen A, masked B)."""
    log(f"\n  Training LoRI adapter: {domain} (idx={adapter_idx}, "
        f"rows [{adapter_idx*RANK_PER_ADAPTER}:{(adapter_idx+1)*RANK_PER_ADAPTER}])")

    mask = make_mask(adapter_idx)

    # Reset B to zeros
    reset_b(model)

    # Freeze lora_a (ensure only lora_b is trainable)
    freeze_lora_a(model)

    # Verify trainable params
    trainable = tree_flatten(model.trainable_parameters())
    n_lora_b = sum(1 for k, _ in trainable if "lora_b" in k)
    n_lora_a = sum(1 for k, _ in trainable if "lora_a" in k)
    log(f"    Trainable: {n_lora_b} lora_b, {n_lora_a} lora_a (should be 0)")

    # Load and batch data
    texts = load_texts(domain, "train")
    batches = make_batches(tokenizer, texts, BATCH_SIZE, SEQ_LEN)
    log(f"    Data: {len(texts)} texts, {len(batches)} batches")

    optimizer = optim.Adam(learning_rate=LR)
    loss_fn = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(n_iters):
        batch, lengths = batches[step % len(batches)]

        loss, grads = loss_fn(model, batch, lengths)
        optimizer.update(model, grads)

        # Project: zero out non-masked rows of B
        apply_b_mask(model, mask)

        mx.eval(loss, model.parameters())
        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step == 0:
            avg = np.mean(losses[-min(50, len(losses)):])
            log(f"    Step {step+1}/{n_iters}: loss={avg:.4f}")

    elapsed = time.time() - t0
    final_loss = float(np.mean(losses[-20:]))
    log(f"    {domain} done: final_loss={final_loss:.4f}, time={elapsed:.1f}s")

    # Save B weights
    b_weights = get_b_weights(model)
    return final_loss, b_weights


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_ppl(model, tokenizer, domain: str, n_samples: int) -> float:
    """Evaluate PPL on validation data."""
    texts = load_texts(domain, "valid")[:n_samples]
    batches = make_batches(tokenizer, texts, min(BATCH_SIZE, len(texts)), SEQ_LEN)

    total_loss = 0.0
    total_tokens = 0

    for batch, lengths in batches:
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]

        mask_np = np.zeros(targets.shape, dtype=np.float32)
        for i, l in enumerate(lengths):
            mask_np[i, : l - 1] = 1.0
        mask = mx.array(mask_np)

        loss = nn.losses.cross_entropy(logits, targets, reduction="none")
        total_loss += (loss * mask).sum().item()
        total_tokens += mask.sum().item()
        mx.eval(model.parameters())

    avg_loss = total_loss / max(total_tokens, 1)
    return float(math.exp(avg_loss))


# ══════════════════════════════════════════════════════════════════════════════
# Interference measurement
# ══════════════════════════════════════════════════════════════════════════════

def compute_delta_w(a_dict: dict, b_dict: dict, mask: mx.array) -> dict[str, mx.array]:
    """Compute ΔW = B * diag(m) * A^T for one adapter across all layers.
    ΔW applied as: y = x @ A @ diag(m) @ B (in mlx_lm convention: lora_a then lora_b).
    So effective weight change in output space: scale * lora_b^T @ lora_a^T."""
    dw = {}
    for key in a_dict:
        a = a_dict[key]       # (d_in, r)
        b = b_dict[key]       # (r, d_out)
        b_masked = b * mask   # (r, d_out), zero non-active rows
        # ΔW_eff = scale * (x @ a) @ b_masked → weight perturbation = a @ b_masked
        delta = a @ b_masked  # (d_in, d_out)
        dw[key] = delta
    return dw


def flatten_delta(dw: dict[str, mx.array]) -> mx.array:
    """Flatten all layer ΔW into single vector for cosine similarity."""
    parts = []
    for key in sorted(dw.keys()):
        parts.append(dw[key].reshape(-1))
    return mx.concatenate(parts)


def cosine_sim(v1: mx.array, v2: mx.array) -> float:
    """Cosine similarity between two vectors."""
    dot = mx.sum(v1 * v2).item()
    n1 = mx.sqrt(mx.sum(v1 * v1)).item()
    n2 = mx.sqrt(mx.sum(v2 * v2)).item()
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return dot / (n1 * n2)


def train_baseline_lora_inscript(
    model, tokenizer, domain: str, n_iters: int
) -> tuple[float, float]:
    """Train standard LoRA (rank 6, both A+B trainable) and return PPL."""
    log(f"\n  Training baseline LoRA (rank {BASELINE_RANK}, A+B trainable) on {domain}...")

    texts = load_texts(domain, "train")
    batches = make_batches(tokenizer, texts, BATCH_SIZE, SEQ_LEN)
    log(f"    Data: {len(texts)} texts, {len(batches)} batches")

    optimizer = optim.Adam(learning_rate=LR)
    loss_fn = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(n_iters):
        batch, lengths = batches[step % len(batches)]
        loss, grads = loss_fn(model, batch, lengths)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step == 0:
            avg = np.mean(losses[-min(50, len(losses)):])
            log(f"    Step {step+1}/{n_iters}: loss={avg:.4f}")

    elapsed = time.time() - t0
    final_loss = float(np.mean(losses[-20:]))
    log(f"    Baseline done: final_loss={final_loss:.4f}, time={elapsed:.1f}s")

    # Evaluate PPL
    ppl = eval_ppl(model, tokenizer, domain, EVAL_SAMPLES)
    log(f"    Baseline PPL ({domain}): {ppl:.2f}")
    return final_loss, ppl


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P5.A0: LoRI Sparse-Mask Composition")
    log(f"  IS_SMOKE={IS_SMOKE}, TOTAL_RANK={TOTAL_RANK}, "
        f"RANK_PER_ADAPTER={RANK_PER_ADAPTER}")
    log(f"  TRAIN_ITERS={TRAIN_ITERS}, LR={LR}, BATCH_SIZE={BATCH_SIZE}")
    log(f"  DOMAINS={DOMAINS}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # ── Phase 1: Load model and apply LoRI ────────────────────────────────
    log("\n=== Phase 1: Load Model + Apply LoRA (rank {}) ===".format(TOTAL_RANK))
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Freeze base model, then add LoRA
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=-1,  # all layers
        config={
            "rank": TOTAL_RANK,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
    )

    # Save the shared A matrices (frozen across all adapters)
    shared_a = get_shared_a(model)
    log(f"  Saved {len(shared_a)} shared A matrices")

    # Count LoRA layers
    n_lora = sum(
        1 for layer in model.layers
        for _, m in layer.named_modules() if isinstance(m, LoRALinear)
    )
    log(f"  LoRA layers: {n_lora}")

    # Show dimensions
    sample_layer = None
    for layer in model.layers:
        for _, m in layer.named_modules():
            if isinstance(m, LoRALinear):
                sample_layer = m
                break
        if sample_layer:
            break
    log(f"  lora_a shape: {sample_layer.lora_a.shape}")  # (d_in, r)
    log(f"  lora_b shape: {sample_layer.lora_b.shape}")  # (r, d_out)

    trainable_per_adapter = RANK_PER_ADAPTER * sample_layer.lora_b.shape[1] * n_lora
    baseline_per_adapter = BASELINE_RANK * (
        sample_layer.lora_a.shape[0] + sample_layer.lora_b.shape[1]
    ) * n_lora
    log(f"  LoRI trainable params/adapter: {trainable_per_adapter:,} "
        f"(B only, {RANK_PER_ADAPTER} active rows)")
    log(f"  Baseline LoRA params/adapter: {baseline_per_adapter:,} "
        f"(A+B, rank {BASELINE_RANK})")

    log_memory("lora-applied")

    # ── Phase 2: Train 5 LoRI adapters ────────────────────────────────────
    log("\n=== Phase 2: Train LoRI Adapters ===")
    adapter_b_weights = {}  # domain -> b_weights dict
    train_losses = {}

    for i, domain in enumerate(DOMAINS):
        # Restore shared A (important: same A for every adapter)
        set_shared_a(model, shared_a)
        loss, b_weights = train_lori_adapter(
            model, tokenizer, domain, adapter_idx=i, n_iters=TRAIN_ITERS
        )
        adapter_b_weights[domain] = b_weights
        train_losses[domain] = loss
        cleanup()

    log(f"\n  Training losses: {train_losses}")
    log_memory("after-training")

    # ── Phase 3: Measure interference (K1264) ─────────────────────────────
    log("\n=== Phase 3: Parameter Interference (K1264) ===")
    masks = {domain: make_mask(i) for i, domain in enumerate(DOMAINS)}

    # --- B-space (parameter space) interference ---
    # This is where the disjoint mask guarantee holds: <B_i, B_j>_F = 0
    log("\n  B-space (parameter space) interference:")
    b_cos_sims = {}
    b_max_cos = 0.0
    for d1, d2 in combinations(DOMAINS, 2):
        # Flatten all B weights for each adapter
        b1_parts = []
        b2_parts = []
        for key in sorted(adapter_b_weights[d1].keys()):
            b1_parts.append(adapter_b_weights[d1][key].reshape(-1))
            b2_parts.append(adapter_b_weights[d2][key].reshape(-1))
        b1_flat = mx.concatenate(b1_parts)
        b2_flat = mx.concatenate(b2_parts)
        cs = cosine_sim(b1_flat, b2_flat)
        b_cos_sims[f"{d1}_vs_{d2}"] = cs
        b_max_cos = max(b_max_cos, abs(cs))
        log(f"    B-cos({d1}, {d2}) = {cs:.2e}")
    log(f"  B-space max|cos| = {b_max_cos:.2e} (should be ~0 by mask disjointness)")

    # --- Weight-space (ΔW) interference ---
    # ΔW_i = A @ B_i_masked. NOT guaranteed zero because A^T A ≠ I.
    log("\n  Weight-space (ΔW = A @ B_masked) interference:")
    delta_ws = {}
    for domain in DOMAINS:
        dw = compute_delta_w(shared_a, adapter_b_weights[domain], masks[domain])
        delta_ws[domain] = flatten_delta(dw)
        mx.eval(delta_ws[domain])

    cos_sims = {}
    max_cos = 0.0
    for d1, d2 in combinations(DOMAINS, 2):
        cs = cosine_sim(delta_ws[d1], delta_ws[d2])
        cos_sims[f"{d1}_vs_{d2}"] = cs
        max_cos = max(max_cos, abs(cs))
        log(f"    W-cos({d1}, {d2}) = {cs:.2e}")

    # K1264 tests weight-space interference
    k1264_pass = max_cos < 1e-4
    log(f"\n  K1264 (weight-space): max|cos| = {max_cos:.2e} < 1e-4 → "
        f"{'PASS' if k1264_pass else 'FAIL'}")
    if not k1264_pass:
        log(f"  NOTE: B-space interference IS zero ({b_max_cos:.2e}). "
            f"Weight-space leak comes from A^T A off-diagonal blocks.")

    # Adapter norms
    norms = {}
    for domain in DOMAINS:
        n = mx.sqrt(mx.sum(delta_ws[domain] ** 2)).item()
        norms[domain] = n
        log(f"  ||ΔW_{domain}|| = {n:.4f}")

    cleanup()

    # ── Phase 4: Evaluate LoRI quality (PPL per domain) ───────────────────
    log("\n=== Phase 4: LoRI Per-Domain PPL ===")
    base_ppls = {}
    lori_ppls = {}

    # Base model PPL (no adapter)
    reset_b(model)
    set_shared_a(model, shared_a)
    mx.eval(model.parameters())

    for domain in DOMAINS:
        ppl = eval_ppl(model, tokenizer, domain, EVAL_SAMPLES)
        base_ppls[domain] = ppl
        log(f"  Base PPL ({domain}): {ppl:.2f}")
    cleanup()

    # LoRI adapter PPL (solo)
    for i, domain in enumerate(DOMAINS):
        set_shared_a(model, shared_a)
        set_b_weights(model, adapter_b_weights[domain])
        mx.eval(model.parameters())
        ppl = eval_ppl(model, tokenizer, domain, EVAL_SAMPLES)
        lori_ppls[domain] = ppl
        log(f"  LoRI PPL ({domain}): {ppl:.2f} (base: {base_ppls[domain]:.2f}, "
            f"reduction: {(1 - ppl/base_ppls[domain])*100:.1f}%)")
    cleanup()

    # ── Phase 5: Baseline LoRA comparison (K1265 — math domain) ──────────
    log("\n=== Phase 5: Baseline LoRA on Math (K1265) ===")

    # Reload model with standard LoRA rank 6 (both A+B trainable)
    del model
    cleanup()

    model, tokenizer = load(MODEL_ID)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=-1,
        config={
            "rank": BASELINE_RANK,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
    )
    # Both A and B trainable (standard LoRA)
    baseline_trainable = tree_flatten(model.trainable_parameters())
    log(f"  Baseline trainable params: {len(baseline_trainable)} "
        f"({sum(v.size for _, v in baseline_trainable):,} total)")

    _, baseline_math_ppl = train_baseline_lora_inscript(
        model, tokenizer, "math", TRAIN_ITERS
    )

    # Quality comparison: PPL improvement ratio
    base_math = base_ppls["math"]
    lori_math = lori_ppls["math"]
    baseline_improvement = (base_math - baseline_math_ppl) / base_math
    lori_improvement = (base_math - lori_math) / base_math
    quality_ratio = lori_improvement / max(baseline_improvement, 1e-8)

    log(f"  Base PPL: {base_math:.2f}")
    log(f"  Baseline improvement: {baseline_improvement*100:.1f}%")
    log(f"  LoRI improvement: {lori_improvement*100:.1f}%")
    log(f"  Quality ratio: {quality_ratio:.3f} (need >= 0.90)")

    k1265_pass = quality_ratio >= 0.90
    log(f"  K1265: quality_ratio = {quality_ratio:.3f} >= 0.90 → "
        f"{'PASS' if k1265_pass else 'FAIL'}")

    del model
    cleanup()

    # ── Phase 6: Composition test (K1266) ─────────────────────────────────
    log("\n=== Phase 6: 5-Adapter Composition (K1266) ===")

    # Reload model with LoRI rank
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=-1,
        config={
            "rank": TOTAL_RANK,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
    )

    # Compose: sum all masked B weights (disjoint → no overlap)
    composed_b = {}
    first_domain = DOMAINS[0]
    for key in adapter_b_weights[first_domain]:
        composed = mx.zeros_like(adapter_b_weights[first_domain][key])
        for i, domain in enumerate(DOMAINS):
            mask = make_mask(i)
            composed = composed + adapter_b_weights[domain][key] * mask
        composed_b[key] = composed

    # Set composed weights
    set_shared_a(model, shared_a)
    set_b_weights(model, composed_b)
    mx.eval(model.parameters())

    # Evaluate composed PPL per domain
    composed_ppls = {}
    max_degradation = 0.0
    for domain in DOMAINS:
        ppl = eval_ppl(model, tokenizer, domain, EVAL_SAMPLES)
        composed_ppls[domain] = ppl
        degradation = ppl - lori_ppls[domain]
        degradation_pct = (degradation / lori_ppls[domain]) * 100
        max_degradation = max(max_degradation, degradation_pct)
        log(f"  Composed PPL ({domain}): {ppl:.2f} "
            f"(solo: {lori_ppls[domain]:.2f}, degradation: {degradation_pct:.1f}%)")

    k1266_pass = max_degradation < 5.0
    log(f"\n  K1266: max_degradation = {max_degradation:.1f}% < 5% → "
        f"{'PASS' if k1266_pass else 'FAIL'}")

    del model
    cleanup()

    # ── Summary ───────────────────────────────────────────────────────────
    total_min = (time.time() - total_start) / 60.0

    log("\n" + "=" * 70)
    log("KILL CRITERIA RESULTS")
    log("=" * 70)
    log(f"  K1264 (interference < 1e-4): max|cos| = {max_cos:.2e} → "
        f"{'PASS' if k1264_pass else 'FAIL'}")
    log(f"  K1265 (quality >= 90% std LoRA): ratio = {quality_ratio:.3f} → "
        f"{'PASS' if k1265_pass else 'FAIL'}")
    log(f"  K1266 (composition < 5pp degrad): max = {max_degradation:.1f}% → "
        f"{'PASS' if k1266_pass else 'FAIL'}")
    log(f"  ALL_PASS = {k1264_pass and k1265_pass and k1266_pass}")
    log(f"  Total time: {total_min:.1f} min")
    log("=" * 70)

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "total_rank": TOTAL_RANK,
            "rank_per_adapter": RANK_PER_ADAPTER,
            "n_adapters": N_ADAPTERS,
            "lora_scale": LORA_SCALE,
            "lora_keys": LORA_KEYS,
            "train_iters": TRAIN_ITERS,
            "lr": LR,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "baseline_rank": BASELINE_RANK,
        },
        "train_losses": train_losses,
        "interference": {
            "weight_space_cosine_sims": cos_sims,
            "weight_space_max_abs_cos": max_cos,
            "b_space_cosine_sims": b_cos_sims,
            "b_space_max_abs_cos": b_max_cos,
            "adapter_norms": norms,
        },
        "quality": {
            "base_ppls": base_ppls,
            "lori_ppls": lori_ppls,
            "baseline_math_ppl": baseline_math_ppl,
            "baseline_improvement_pct": baseline_improvement * 100,
            "lori_improvement_pct": lori_improvement * 100,
            "quality_ratio": quality_ratio,
        },
        "composition": {
            "composed_ppls": composed_ppls,
            "solo_ppls": lori_ppls,
            "max_degradation_pct": max_degradation,
            "per_domain_degradation": {
                domain: (composed_ppls[domain] - lori_ppls[domain]) / lori_ppls[domain] * 100
                for domain in DOMAINS
            },
        },
        "kill_criteria": {
            "k1264_interference": {"pass": k1264_pass, "max_cos": max_cos},
            "k1265_quality": {"pass": k1265_pass, "quality_ratio": quality_ratio},
            "k1266_composition": {"pass": k1266_pass, "max_degradation_pct": max_degradation},
        },
        "all_pass": k1264_pass and k1265_pass and k1266_pass,
        "total_time_min": round(total_min, 2),
        "trainable_params_per_adapter": int(
            RANK_PER_ADAPTER * 2048 * n_lora  # d_out * n_layers
        ),
        "baseline_params_per_adapter": int(
            BASELINE_RANK * (2560 + 2048) * n_lora
        ),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
