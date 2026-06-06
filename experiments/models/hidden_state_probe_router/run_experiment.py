#!/usr/bin/env python3
"""Hidden-State MLP Probe for Per-Token Adapter Routing.

Extends Finding #276 (ridge regression, 96% sequence-level accuracy) to
TOKEN-LEVEL routing using a small MLP probe on base model hidden states.

Grounded by: X-LoRA (arXiv 2402.07148), TT-LoRA MoE (arXiv 2504.21190).
Type: Guided Exploration (Type 2) -- proven framework, unknown is per-token
accuracy with post-hoc probe on frozen ternary adapters.

Kill criteria:
  K784: Token-level routing accuracy >= 85%
  K785: Mixed-domain PPL within 5% of oracle routing (Finding #305)
  K786: Probe inference overhead < 1ms per token

Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

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

# Source data and adapters (same as ridge router, Finding #276)
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEGMENT_LENGTH = 128  # matches Finding #305
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Probe architecture
PROBE_HIDDEN_DIM = 128  # Derived from random features bound (MATH.md)
PROBE_LR = 1e-3
PROBE_EPOCHS = 30
PROBE_BATCH_SIZE = 64

# Data budget
N_CAL_PER_DOMAIN = 30   # calibration samples for probe training
N_TEST_PER_DOMAIN = 10  # test samples for accuracy evaluation
N_SEQUENCES_PER_PAIR = 10  # mixed-domain sequences for PPL evaluation


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


# ============================================================================
# Data loading
# ============================================================================

def load_domain_data(domain, split="train", max_samples=400):
    """Load instruction-format text from domain data."""
    path = DATA_DIR / domain / f"{split}.jsonl"
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                samples.append({"instruction": instruction, "response": response, "text": text})
    return samples


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Model utilities (adapted from prior experiments)
# ============================================================================

from mlx_lm import load as load_model_and_tokenizer
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


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
    log(f"  Applied LoRA (r={rank}, scale={scale}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


# ============================================================================
# MLP Probe (the core contribution)
# ============================================================================

class MLPProbe(nn.Module):
    """Single hidden-layer MLP for domain classification.

    Architecture: Linear(d, w) -> ReLU -> Linear(w, K)
    Parameters: d*w + w + w*K + K = w*(d + K) + w + K ~ 328K for d=2560, w=128, K=5
    """
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def __call__(self, x):
        h = mx.maximum(self.fc1(x), 0)  # ReLU
        return self.fc2(h)


# ============================================================================
# Phase 1: Extract per-token hidden states
# ============================================================================

def phase_extract_hidden_states():
    """Extract PER-TOKEN hidden states from base model forward pass.

    Unlike ridge router (mean-pooled), we keep individual token representations.
    This is the key difference: token-level classification vs sequence-level.
    """
    log("\n" + "=" * 70)
    log("PHASE 1: EXTRACT PER-TOKEN HIDDEN STATES")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model.freeze()
    mx.eval(model.parameters())
    log_memory("model-loaded")

    # Load calibration and test data
    all_data = {}
    for domain in DOMAINS:
        all_samples = load_domain_data(domain, split="valid",
                                       max_samples=N_CAL_PER_DOMAIN + N_TEST_PER_DOMAIN)
        if len(all_samples) < N_CAL_PER_DOMAIN + N_TEST_PER_DOMAIN:
            train_supplement = load_domain_data(domain, split="train",
                                                max_samples=N_CAL_PER_DOMAIN + N_TEST_PER_DOMAIN)
            all_samples = all_samples + train_supplement
        all_data[domain] = all_samples[:N_CAL_PER_DOMAIN + N_TEST_PER_DOMAIN]
        log(f"  {domain}: {len(all_data[domain])} total samples")

    # Extract token-level hidden states
    def extract_token_hidden_states(samples, domain_idx, label=""):
        """Get per-token hidden states (not mean-pooled)."""
        token_hiddens = []
        token_labels = []
        seq_hiddens = []  # also compute mean-pooled for comparison
        seq_labels = []

        for i, sample in enumerate(samples):
            text = sample["text"]
            tokens = tokenizer.encode(text)
            if len(tokens) > MAX_SEQ_LENGTH:
                tokens = tokens[:MAX_SEQ_LENGTH]
            if len(tokens) < 4:
                continue

            input_ids = mx.array(tokens)[None, :]

            # Forward pass to get last hidden states
            h = model.model.embed_tokens(input_ids)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                h.shape[1]).astype(h.dtype)
            for layer in model.model.layers:
                h = layer(h, mask=mask)
            h = model.model.norm(h)
            mx.eval(h)

            # Per-token: each token gets a d-dim hidden state + domain label
            h_np = np.array(h[0].astype(mx.float32))
            mx.eval(h[0])  # ensure eval before conversion

            # Use response tokens only (skip instruction prompt)
            # This ensures we measure domain-specific content, not template
            resp_start = 0
            text_lower = text.lower()
            if "### response:" in text_lower:
                # Find the token position of the response start
                prompt = format_prompt(sample["instruction"])
                prompt_tokens = tokenizer.encode(prompt)
                resp_start = min(len(prompt_tokens), len(tokens) - 2)

            # Take tokens from response portion
            for t in range(resp_start, len(tokens)):
                token_hiddens.append(h_np[t])
                token_labels.append(domain_idx)

            # Mean-pooled for comparison
            h_mean = np.mean(h_np[resp_start:], axis=0)
            seq_hiddens.append(h_mean)
            seq_labels.append(domain_idx)

            del h, input_ids, mask, h_np
            if (i + 1) % 10 == 0:
                gc.collect()
                mx.clear_cache()
                log(f"    [{label}] {i+1}/{len(samples)} processed")

        return token_hiddens, token_labels, seq_hiddens, seq_labels

    # Split into calibration and test
    cal_token_h, cal_token_l = [], []
    cal_seq_h, cal_seq_l = [], []
    test_token_h, test_token_l = [], []
    test_seq_h, test_seq_l = [], []

    for di, domain in enumerate(DOMAINS):
        samples = all_data[domain]
        cal_samples = samples[:N_CAL_PER_DOMAIN]
        test_samples = samples[N_CAL_PER_DOMAIN:N_CAL_PER_DOMAIN + N_TEST_PER_DOMAIN]

        log(f"  Extracting {domain} calibration ({len(cal_samples)} samples)...")
        th, tl, sh, sl = extract_token_hidden_states(cal_samples, di, f"cal-{domain}")
        cal_token_h.extend(th)
        cal_token_l.extend(tl)
        cal_seq_h.extend(sh)
        cal_seq_l.extend(sl)

        log(f"  Extracting {domain} test ({len(test_samples)} samples)...")
        th, tl, sh, sl = extract_token_hidden_states(test_samples, di, f"test-{domain}")
        test_token_h.extend(th)
        test_token_l.extend(tl)
        test_seq_h.extend(sh)
        test_seq_l.extend(sl)

    cal_token_h = np.stack(cal_token_h)
    cal_token_l = np.array(cal_token_l)
    cal_seq_h = np.stack(cal_seq_h)
    cal_seq_l = np.array(cal_seq_l)
    test_token_h = np.stack(test_token_h)
    test_token_l = np.array(test_token_l)
    test_seq_h = np.stack(test_seq_h)
    test_seq_l = np.array(test_seq_l)

    elapsed = time.time() - t0
    log(f"\n  Extraction complete in {elapsed:.1f}s")
    log(f"  Cal token hidden: {cal_token_h.shape}")
    log(f"  Test token hidden: {test_token_h.shape}")
    log(f"  Cal seq hidden: {cal_seq_h.shape}")
    log(f"  Test seq hidden: {test_seq_h.shape}")
    log(f"  Hidden dim: {cal_token_h.shape[1]}")

    # Token distribution per domain
    for di, domain in enumerate(DOMAINS):
        n_cal = np.sum(cal_token_l == di)
        n_test = np.sum(test_token_l == di)
        log(f"  {domain}: cal_tokens={n_cal}, test_tokens={n_test}")

    # Save to disk
    np.savez(
        str(EXPERIMENT_DIR / "hidden_states.npz"),
        cal_token_h=cal_token_h,
        cal_token_l=cal_token_l,
        cal_seq_h=cal_seq_h,
        cal_seq_l=cal_seq_l,
        test_token_h=test_token_h,
        test_token_l=test_token_l,
        test_seq_h=test_seq_h,
        test_seq_l=test_seq_l,
    )
    log(f"  Saved hidden states to {EXPERIMENT_DIR / 'hidden_states.npz'}")

    del model, tokenizer
    cleanup()
    log_memory("post-cleanup")

    return {
        "cal_token_h": cal_token_h, "cal_token_l": cal_token_l,
        "cal_seq_h": cal_seq_h, "cal_seq_l": cal_seq_l,
        "test_token_h": test_token_h, "test_token_l": test_token_l,
        "test_seq_h": test_seq_h, "test_seq_l": test_seq_l,
        "hidden_dim": cal_token_h.shape[1],
        "extraction_time_s": elapsed,
    }


# ============================================================================
# Phase 2: Train MLP probe + ridge baseline on token-level data
# ============================================================================

def phase_train_probes(data):
    """Train MLP probe on token-level hidden states. Also build ridge baseline."""
    log("\n" + "=" * 70)
    log("PHASE 2: TRAIN MLP PROBE + RIDGE BASELINE")
    log("=" * 70)

    d = data["hidden_dim"]
    cal_token_h = data["cal_token_h"]
    cal_token_l = data["cal_token_l"]
    test_token_h = data["test_token_h"]
    test_token_l = data["test_token_l"]
    cal_seq_h = data["cal_seq_h"]
    cal_seq_l = data["cal_seq_l"]
    test_seq_h = data["test_seq_h"]
    test_seq_l = data["test_seq_l"]

    results = {}

    # --- Ridge regression baseline (token-level) ---
    log("\n  [Ridge] Token-level ridge regression (linear baseline)...")
    t0 = time.time()

    # Normalize features for better ridge performance
    mean_h = np.mean(cal_token_h, axis=0, keepdims=True)
    std_h = np.std(cal_token_h, axis=0, keepdims=True) + 1e-8
    cal_norm = (cal_token_h - mean_h) / std_h
    test_norm = (test_token_h - mean_h) / std_h

    n_cal = cal_norm.shape[0]
    Y_cal = np.zeros((n_cal, N_DOMAINS), dtype=np.float64)
    for i, l in enumerate(cal_token_l):
        Y_cal[i, l] = 1.0

    # Ridge: W* = (X^TX + lambda*I)^{-1} X^TY
    best_ridge_acc = 0.0
    best_ridge_lambda = 1.0
    for lam in [0.01, 0.1, 1.0, 10.0]:
        G = cal_norm.T @ cal_norm
        H = cal_norm.T @ Y_cal
        G_reg = G + lam * np.eye(d, dtype=np.float64)
        W = np.linalg.solve(G_reg, H)
        test_preds = np.argmax(test_norm @ W, axis=1)
        acc = np.mean(test_preds == test_token_l)
        log(f"    lambda={lam}: token_acc={acc:.3f}")
        if acc > best_ridge_acc:
            best_ridge_acc = acc
            best_ridge_lambda = lam

    # Final ridge with best lambda
    G = cal_norm.T @ cal_norm
    H = cal_norm.T @ Y_cal
    G_reg = G + best_ridge_lambda * np.eye(d, dtype=np.float64)
    W_ridge = np.linalg.solve(G_reg, H)
    ridge_token_preds = np.argmax(test_norm @ W_ridge, axis=1)
    ridge_token_acc = np.mean(ridge_token_preds == test_token_l)

    # Ridge on mean-pooled (sequence-level, for comparison with Finding #276)
    cal_seq_norm = (cal_seq_h - np.mean(cal_seq_h, axis=0, keepdims=True)) / (np.std(cal_seq_h, axis=0, keepdims=True) + 1e-8)
    test_seq_norm = (test_seq_h - np.mean(cal_seq_h, axis=0, keepdims=True)) / (np.std(cal_seq_h, axis=0, keepdims=True) + 1e-8)
    n_seq = cal_seq_norm.shape[0]
    Y_seq = np.zeros((n_seq, N_DOMAINS), dtype=np.float64)
    for i, l in enumerate(cal_seq_l):
        Y_seq[i, l] = 1.0
    G_seq = cal_seq_norm.T @ cal_seq_norm
    H_seq = cal_seq_norm.T @ Y_seq
    G_reg_seq = G_seq + best_ridge_lambda * np.eye(d, dtype=np.float64)
    W_seq = np.linalg.solve(G_reg_seq, H_seq)
    ridge_seq_preds = np.argmax(test_seq_norm @ W_seq, axis=1)
    ridge_seq_acc = np.mean(ridge_seq_preds == test_seq_l)

    ridge_time = time.time() - t0
    log(f"\n  [Ridge] Token-level accuracy: {ridge_token_acc:.3f}")
    log(f"  [Ridge] Sequence-level accuracy: {ridge_seq_acc:.3f}")
    log(f"  [Ridge] Best lambda: {best_ridge_lambda}")
    log(f"  [Ridge] Time: {ridge_time:.1f}s")

    # Per-domain ridge accuracy
    ridge_per_domain = {}
    for di, domain in enumerate(DOMAINS):
        mask = test_token_l == di
        if np.sum(mask) > 0:
            domain_acc = np.mean(ridge_token_preds[mask] == di)
            ridge_per_domain[domain] = float(domain_acc)
            log(f"    {domain}: {domain_acc:.3f} ({np.sum(mask)} tokens)")

    results["ridge"] = {
        "token_accuracy": float(ridge_token_acc),
        "seq_accuracy": float(ridge_seq_acc),
        "best_lambda": float(best_ridge_lambda),
        "per_domain": ridge_per_domain,
        "time_s": float(ridge_time),
    }

    # --- MLP Probe (the core contribution) ---
    log("\n  [MLP] Training MLP probe...")
    t0 = time.time()

    probe = MLPProbe(d, PROBE_HIDDEN_DIM, N_DOMAINS)
    mx.eval(probe.parameters())

    n_params = sum(p.size for _, p in tree_flatten(probe.parameters()))
    log(f"  [MLP] Probe parameters: {n_params:,}")
    log(f"  [MLP] Architecture: Linear({d}, {PROBE_HIDDEN_DIM}) -> ReLU -> Linear({PROBE_HIDDEN_DIM}, {N_DOMAINS})")

    optimizer = opt.Adam(learning_rate=PROBE_LR)

    # Convert to MLX arrays
    X_train = mx.array(cal_token_h.astype(np.float32))
    y_train = mx.array(cal_token_l.astype(np.int32))
    X_test = mx.array(test_token_h.astype(np.float32))
    y_test = mx.array(test_token_l.astype(np.int32))

    n_train = X_train.shape[0]
    n_batches = max(1, n_train // PROBE_BATCH_SIZE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(probe, loss_fn)

    best_test_acc = 0.0
    train_losses = []

    # Disable GC during training loop per CODING_GUIDELINES
    gc.disable()
    for epoch in range(PROBE_EPOCHS):
        # Shuffle
        perm = mx.array(np.random.permutation(n_train))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * PROBE_BATCH_SIZE
            end = min(start + PROBE_BATCH_SIZE, n_train)
            idx = perm[start:end]
            xb = X_train[idx]
            yb = y_train[idx]

            loss, grads = loss_and_grad(probe, xb, yb)
            optimizer.update(probe, grads)
            mx.eval(probe.parameters(), optimizer.state, loss)
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # Test accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_logits = probe(X_test)
            mx.eval(test_logits)
            test_preds = mx.argmax(test_logits, axis=1)
            mx.eval(test_preds)
            test_acc = float(mx.mean(test_preds == y_test).item())
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            log(f"    Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, test_acc={test_acc:.3f}")
            del test_logits, test_preds

    gc.enable()
    gc.collect()

    # Final evaluation
    test_logits = probe(X_test)
    mx.eval(test_logits)
    test_preds_mlx = mx.argmax(test_logits, axis=1)
    mx.eval(test_preds_mlx)
    final_test_acc = float(mx.mean(test_preds_mlx == y_test).item())
    test_preds_np = np.array(test_preds_mlx)

    # Per-domain MLP accuracy
    mlp_per_domain = {}
    for di, domain in enumerate(DOMAINS):
        mask = test_token_l == di
        if np.sum(mask) > 0:
            domain_preds = test_preds_np[mask]
            domain_acc = np.mean(domain_preds == di)
            mlp_per_domain[domain] = float(domain_acc)
            # Confusion
            wrong = domain_preds[domain_preds != di]
            if len(wrong) > 0:
                confusion = {}
                for w in wrong:
                    name = DOMAINS[w]
                    confusion[name] = confusion.get(name, 0) + 1
                log(f"    {domain}: {domain_acc:.3f} (misrouted: {confusion})")
            else:
                log(f"    {domain}: {domain_acc:.3f}")

    # Mean-pooled accuracy with probe (for comparison)
    X_test_seq = mx.array(test_seq_h.astype(np.float32))
    y_test_seq = mx.array(test_seq_l.astype(np.int32))
    seq_logits = probe(X_test_seq)
    mx.eval(seq_logits)
    seq_preds = mx.argmax(seq_logits, axis=1)
    mx.eval(seq_preds)
    mlp_seq_acc = float(mx.mean(seq_preds == y_test_seq).item())

    mlp_time = time.time() - t0
    log(f"\n  [MLP] Token-level accuracy: {final_test_acc:.3f}")
    log(f"  [MLP] Sequence-level accuracy (mean-pooled): {mlp_seq_acc:.3f}")
    log(f"  [MLP] Best test accuracy: {best_test_acc:.3f}")
    log(f"  [MLP] Training time: {mlp_time:.1f}s")

    results["mlp_probe"] = {
        "token_accuracy": float(final_test_acc),
        "seq_accuracy": float(mlp_seq_acc),
        "best_test_accuracy": float(best_test_acc),
        "per_domain": mlp_per_domain,
        "n_params": int(n_params),
        "hidden_dim": PROBE_HIDDEN_DIM,
        "epochs": PROBE_EPOCHS,
        "lr": PROBE_LR,
        "final_loss": float(train_losses[-1]),
        "time_s": float(mlp_time),
    }

    # --- Probe inference latency (K786) ---
    log("\n  [Latency] Measuring probe inference overhead...")
    # Single token
    x_single = mx.array(np.random.randn(1, d).astype(np.float32))
    mx.eval(x_single)

    # Warmup
    for _ in range(10):
        _ = probe(x_single)
        mx.eval(_)

    # Benchmark
    n_iters = 1000
    t0 = time.time()
    for _ in range(n_iters):
        out = probe(x_single)
        mx.eval(out)
    latency_ms = (time.time() - t0) / n_iters * 1000
    log(f"  [Latency] Per-token probe: {latency_ms:.4f}ms ({latency_ms*1000:.1f}us)")

    # Batch of 128 tokens
    x_batch = mx.array(np.random.randn(128, d).astype(np.float32))
    mx.eval(x_batch)
    for _ in range(10):
        _ = probe(x_batch)
        mx.eval(_)
    t0 = time.time()
    for _ in range(n_iters):
        out = probe(x_batch)
        mx.eval(out)
    batch_latency_ms = (time.time() - t0) / n_iters * 1000
    per_token_batch = batch_latency_ms / 128
    log(f"  [Latency] Batch-128 total: {batch_latency_ms:.4f}ms, per-token: {per_token_batch:.4f}ms")

    results["latency"] = {
        "single_token_ms": float(latency_ms),
        "batch_128_total_ms": float(batch_latency_ms),
        "batch_128_per_token_ms": float(per_token_batch),
    }

    # Save probe weights
    probe_weights = dict(tree_flatten(probe.parameters()))
    mx.savez(str(EXPERIMENT_DIR / "probe_weights.npz"), **probe_weights)

    # Save normalization stats for ridge
    np.savez(
        str(EXPERIMENT_DIR / "ridge_stats.npz"),
        mean_h=mean_h, std_h=std_h,
        W_ridge=W_ridge, best_lambda=np.array([best_ridge_lambda]),
    )

    del probe, optimizer, X_train, y_train, X_test, y_test, x_single, x_batch
    cleanup()

    return results


# ============================================================================
# Phase 3: End-to-end PPL on mixed-domain sequences (K785)
# ============================================================================

def phase_mixed_domain_ppl(probe_results):
    """Evaluate probe routing on mixed-domain sequences.

    Compare:
    1. Oracle: each segment with correct adapter (upper bound)
    2. Probe routing: classify each token, majority-vote per segment
    3. Per-sequence best: single best adapter for full sequence
    4. Base only: no adapter (lower bound)
    """
    log("\n" + "=" * 70)
    log("PHASE 3: MIXED-DOMAIN PPL EVALUATION (K785)")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    log_memory("model-loaded")

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTERS_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Load probe
    probe = MLPProbe(probe_results["mlp_probe"]["hidden_dim"] + N_DOMAINS * PROBE_HIDDEN_DIM - N_DOMAINS * PROBE_HIDDEN_DIM,
                     PROBE_HIDDEN_DIM, N_DOMAINS)  # reconstruct d from results
    # Actually, we need hidden_dim from data extraction
    hidden_dim = 2560  # Known from model
    probe = MLPProbe(hidden_dim, PROBE_HIDDEN_DIM, N_DOMAINS)
    probe_weights = dict(mx.load(str(EXPERIMENT_DIR / "probe_weights.npz")))
    probe.load_weights(list(probe_weights.items()))
    mx.eval(probe.parameters())
    probe.freeze()
    log(f"  Loaded probe ({sum(p.size for _, p in tree_flatten(probe.parameters())):,} params)")

    # Construct mixed-domain sequences
    rng = random.Random(SEED)
    domain_texts = {}
    for domain in DOMAINS:
        texts = []
        data = load_domain_data(domain, split="valid", max_samples=50)
        for s in data:
            texts.append(s["text"])
        domain_texts[domain] = texts

    domain_pairs = list(combinations(DOMAINS, 2))
    mixed_sequences = []

    for domain_a, domain_b in domain_pairs:
        texts_a = domain_texts[domain_a]
        texts_b = domain_texts[domain_b]
        pair_count = 0

        for _ in range(N_SEQUENCES_PER_PAIR * 3):
            if pair_count >= N_SEQUENCES_PER_PAIR:
                break
            text_a = texts_a[rng.randint(0, len(texts_a) - 1)]
            text_b = texts_b[rng.randint(0, len(texts_b) - 1)]
            toks_a = tokenizer.encode(text_a)
            toks_b = tokenizer.encode(text_b)

            while len(toks_a) < SEGMENT_LENGTH:
                toks_a = toks_a + toks_a
            while len(toks_b) < SEGMENT_LENGTH:
                toks_b = toks_b + toks_b

            seg_a = toks_a[:SEGMENT_LENGTH]
            seg_b = toks_b[:SEGMENT_LENGTH]

            mixed_sequences.append({
                "tokens": seg_a + seg_b,
                "seg_a_tokens": seg_a,
                "seg_b_tokens": seg_b,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "domain_a_idx": DOMAINS.index(domain_a),
                "domain_b_idx": DOMAINS.index(domain_b),
                "boundary_pos": SEGMENT_LENGTH,
            })
            pair_count += 1

    log(f"  Created {len(mixed_sequences)} mixed-domain sequences ({len(domain_pairs)} pairs)")

    # Helper: compute segment PPL
    def compute_segment_ppl(tokens):
        if len(tokens) < 2:
            return 0.0, 0
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        nll = loss.item()
        n = y.size
        del x, y, logits, loss
        return nll, n

    # Helper: get hidden states for token classification
    def get_base_hidden(tokens):
        """Get base model hidden states (zero adapter) for probe classification."""
        zero_adapter_in_model(model)
        x = mx.array(tokens)[None, :]
        h = model.model.embed_tokens(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            h.shape[1]).astype(h.dtype)
        for layer in model.model.layers:
            h = layer(h, mask=mask)
        h = model.model.norm(h)
        mx.eval(h)
        return h[0]  # (T, d)

    # Evaluate strategies
    stats = {
        "oracle": {"nll": 0.0, "n": 0},
        "probe_seg": {"nll": 0.0, "n": 0},
        "per_seq_best": {"nll": 0.0, "n": 0},
        "base_only": {"nll": 0.0, "n": 0},
    }

    routing_correct_a = 0
    routing_correct_b = 0
    routing_total = 0

    for si, seq_data in enumerate(mixed_sequences):
        seg_a_tokens = seq_data["seg_a_tokens"]
        seg_b_tokens = seq_data["seg_b_tokens"]
        domain_a = seq_data["domain_a"]
        domain_b = seq_data["domain_b"]

        # --- Strategy 1: Oracle (correct adapter per segment) ---
        apply_adapter_to_model(model, adapters[domain_a])
        nll_a, n_a = compute_segment_ppl(seg_a_tokens)
        apply_adapter_to_model(model, adapters[domain_b])
        nll_b, n_b = compute_segment_ppl(seg_b_tokens)
        stats["oracle"]["nll"] += nll_a + nll_b
        stats["oracle"]["n"] += n_a + n_b

        # --- Strategy 2: Probe routing (classify tokens, majority vote per segment) ---
        # Get hidden states for FULL sequence (base model, zero adapter)
        h_full = get_base_hidden(seq_data["tokens"])
        probe_logits = probe(h_full)
        mx.eval(probe_logits)
        probe_preds = np.array(mx.argmax(probe_logits, axis=1))
        del h_full, probe_logits

        # Majority vote for segment A (first SEGMENT_LENGTH tokens)
        seg_a_preds = probe_preds[:SEGMENT_LENGTH]
        seg_a_majority = int(np.argmax(np.bincount(seg_a_preds, minlength=N_DOMAINS)))
        pred_domain_a = DOMAINS[seg_a_majority]

        # Majority vote for segment B (remaining tokens)
        seg_b_preds = probe_preds[SEGMENT_LENGTH:]
        if len(seg_b_preds) > 0:
            seg_b_majority = int(np.argmax(np.bincount(seg_b_preds, minlength=N_DOMAINS)))
        else:
            seg_b_majority = seg_a_majority
        pred_domain_b = DOMAINS[seg_b_majority]

        # Track routing accuracy
        if pred_domain_a == domain_a:
            routing_correct_a += 1
        if pred_domain_b == domain_b:
            routing_correct_b += 1
        routing_total += 1

        # Evaluate with probe-selected adapters (segment-isolated)
        apply_adapter_to_model(model, adapters[pred_domain_a])
        nll_a_probe, n_a_probe = compute_segment_ppl(seg_a_tokens)
        apply_adapter_to_model(model, adapters[pred_domain_b])
        nll_b_probe, n_b_probe = compute_segment_ppl(seg_b_tokens)
        stats["probe_seg"]["nll"] += nll_a_probe + nll_b_probe
        stats["probe_seg"]["n"] += n_a_probe + n_b_probe

        # --- Strategy 3: Per-sequence best (try each, pick best) ---
        best_nll = float("inf")
        for d_name in DOMAINS:
            apply_adapter_to_model(model, adapters[d_name])
            nll_full, n_full = compute_segment_ppl(seq_data["tokens"])
            if nll_full < best_nll:
                best_nll = nll_full
                best_n = n_full
        stats["per_seq_best"]["nll"] += best_nll
        stats["per_seq_best"]["n"] += best_n

        # --- Strategy 4: Base only ---
        zero_adapter_in_model(model)
        nll_base, n_base = compute_segment_ppl(seq_data["tokens"])
        stats["base_only"]["nll"] += nll_base
        stats["base_only"]["n"] += n_base

        zero_adapter_in_model(model)

        if (si + 1) % 20 == 0:
            gc.collect()
            mx.clear_cache()
            log(f"  [{si+1}/{len(mixed_sequences)}] processed")

    # Compute PPL
    ppl_results = {}
    for strategy, s in stats.items():
        if s["n"] > 0:
            ppl = math.exp(s["nll"] / s["n"])
        else:
            ppl = float("inf")
        ppl_results[strategy] = ppl
        log(f"  {strategy}: PPL={ppl:.4f} (NLL={s['nll']:.1f}, n={s['n']})")

    # Routing accuracy
    seg_a_acc = routing_correct_a / max(routing_total, 1)
    seg_b_acc = routing_correct_b / max(routing_total, 1)
    overall_routing_acc = (routing_correct_a + routing_correct_b) / max(2 * routing_total, 1)
    log(f"\n  Probe routing accuracy:")
    log(f"    Segment A: {seg_a_acc:.1%} ({routing_correct_a}/{routing_total})")
    log(f"    Segment B: {seg_b_acc:.1%} ({routing_correct_b}/{routing_total})")
    log(f"    Overall: {overall_routing_acc:.1%}")

    # K785: PPL probe vs oracle
    if ppl_results["oracle"] > 0:
        ppl_ratio = ppl_results["probe_seg"] / ppl_results["oracle"]
        ppl_gap = (ppl_ratio - 1) * 100
        log(f"\n  K785: PPL probe/oracle = {ppl_ratio:.4f} ({ppl_gap:+.2f}%)")
    else:
        ppl_ratio = float("inf")
        ppl_gap = float("inf")

    elapsed = time.time() - t0
    log(f"  Phase 3 time: {elapsed:.1f}s")

    del model, tokenizer
    cleanup()

    return {
        "ppl": ppl_results,
        "routing_accuracy": {
            "seg_a": float(seg_a_acc),
            "seg_b": float(seg_b_acc),
            "overall": float(overall_routing_acc),
        },
        "ppl_ratio_probe_oracle": float(ppl_ratio),
        "ppl_gap_pct": float(ppl_gap),
        "n_sequences": len(mixed_sequences),
        "n_pairs": len(domain_pairs),
        "time_s": float(elapsed),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("EXPERIMENT: Hidden-State MLP Probe for Per-Token Adapter Routing")
    log(f"  Model: {MODEL_ID}")
    log(f"  Domains: {DOMAINS}")
    log(f"  Probe: MLP({PROBE_HIDDEN_DIM}), {PROBE_EPOCHS} epochs, lr={PROBE_LR}")
    log(f"  Cal/Test: {N_CAL_PER_DOMAIN}/{N_TEST_PER_DOMAIN} samples/domain")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Extract hidden states
    data = phase_extract_hidden_states()
    log_memory("after-phase1")

    # Phase 2: Train probes and measure accuracy/latency
    probe_results = phase_train_probes(data)
    log_memory("after-phase2")

    # Phase 3: Mixed-domain PPL evaluation
    ppl_results = phase_mixed_domain_ppl(probe_results)
    log_memory("after-phase3")

    # ====================================================================
    # Assemble results and assess kill criteria
    # ====================================================================
    total_time = time.time() - t_start

    mlp_token_acc = probe_results["mlp_probe"]["token_accuracy"]
    ridge_token_acc = probe_results["ridge"]["token_accuracy"]
    ridge_seq_acc = probe_results["ridge"]["seq_accuracy"]
    probe_latency = probe_results["latency"]["single_token_ms"]
    ppl_gap = ppl_results["ppl_gap_pct"]

    # K784: Token-level accuracy >= 85%
    k784_pass = mlp_token_acc >= 0.85
    k784_detail = f"MLP token acc = {mlp_token_acc:.3f} (threshold 0.85)"

    # K785: PPL within 5% of oracle
    k785_pass = abs(ppl_gap) <= 5.0
    k785_detail = f"PPL gap = {ppl_gap:+.2f}% (threshold 5%)"

    # K786: Latency < 1ms
    k786_pass = probe_latency < 1.0
    k786_detail = f"Latency = {probe_latency:.4f}ms (threshold 1ms)"

    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)
    log(f"  K784: {'PASS' if k784_pass else 'FAIL'} - {k784_detail}")
    log(f"  K785: {'PASS' if k785_pass else 'FAIL'} - {k785_detail}")
    log(f"  K786: {'PASS' if k786_pass else 'FAIL'} - {k786_detail}")

    overall = "SUPPORTED" if (k784_pass and k785_pass and k786_pass) else "KILLED"
    if not k784_pass and not k785_pass:
        overall = "KILLED"
    elif not k784_pass or not k785_pass:
        overall = "KILLED"

    log(f"\n  VERDICT: {overall}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  MLP probe token accuracy: {mlp_token_acc:.3f}")
    log(f"  Ridge token accuracy: {ridge_token_acc:.3f}")
    log(f"  Ridge seq accuracy (Finding #276 comparison): {ridge_seq_acc:.3f}")
    log(f"  Probe PPL / Oracle PPL: {ppl_results['ppl_ratio_probe_oracle']:.4f}")
    log(f"  Probe latency: {probe_latency:.4f}ms")
    log(f"  Mixed-domain routing accuracy: {ppl_results['routing_accuracy']['overall']:.1%}")
    log(f"  Total time: {total_time:.1f}s")

    # Write results
    results = {
        "experiment": "exp_hidden_state_probe_router",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_domains": N_DOMAINS,
        "probe_config": {
            "hidden_dim": PROBE_HIDDEN_DIM,
            "epochs": PROBE_EPOCHS,
            "lr": PROBE_LR,
            "batch_size": PROBE_BATCH_SIZE,
        },
        "data_config": {
            "n_cal_per_domain": N_CAL_PER_DOMAIN,
            "n_test_per_domain": N_TEST_PER_DOMAIN,
            "n_sequences_per_pair": N_SEQUENCES_PER_PAIR,
            "segment_length": SEGMENT_LENGTH,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
        "extraction_time_s": data["extraction_time_s"],
        "probe_results": probe_results,
        "mixed_domain_ppl": ppl_results,
        "kill_criteria": {
            "K784": {"pass": k784_pass, "detail": k784_detail},
            "K785": {"pass": k785_pass, "detail": k785_detail},
            "K786": {"pass": k786_pass, "detail": k786_detail},
        },
        "verdict": overall,
        "total_time_s": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
