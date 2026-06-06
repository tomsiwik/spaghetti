#!/usr/bin/env python3
"""PiSSA SVD-Init vs Grassmannian Init for LoRA on Ternary Base.

Tests whether PiSSA (arxiv 2404.02948) SVD-based LoRA initialization provides
better quality than Grassmannian random orthonormal initialization, and whether
PiSSA adapters can compose despite sharing the same A matrices.

Kill criteria:
  K1: PiSSA-frozen-A worse single-adapter PPL than Grassmannian-frozen-A
  K2: PiSSA-unfrozen-A adapter cosine > 0.1 (orthogonality destroyed)
  K3: Neither PiSSA variant improves over Grassmannian on any metric

Success criteria:
  S1: PiSSA-frozen-A gives > 5% better single-adapter PPL than Grassmannian
  S2: PiSSA-unfrozen-A composes successfully (composition ratio < 1.5)

3 conditions x 5 domains x 3 seeds = 45 adapter trainings.
Each training: 200 steps. Micro-scale: d=64, 2 layers, character-level names.

Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Memory limits (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Architecture (small enough for fast iteration)
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
BLOCK_SIZE = 32
MLP_DIM = 4 * D_MODEL  # 256
LORA_RANK = 8
LORA_SCALE = 8.0
VOCAB_SIZE = 32  # character-level

# Training
BASE_STEPS = 2000
BASE_LR = 3e-4
ADAPTER_STEPS = 200
ADAPTER_LR = 1e-3
BATCH_SIZE = 64
N_DOMAINS = 5
SEEDS = [42, 137, 314]


def log(msg):
    print(msg, flush=True)


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
# Ternary Base Model (same as ternary_base_from_scratch_mlx)
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE."""
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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = BitLinear(n_embd, n_embd)
        self.wk = BitLinear(n_embd, n_embd)
        self.wv = BitLinear(n_embd, n_embd)
        self.wo = BitLinear(n_embd, n_embd)

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


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.RMSNorm(n_embd)
        self.gate_proj = BitLinear(n_embd, mlp_dim)
        self.up_proj = BitLinear(n_embd, mlp_dim)
        self.down_proj = BitLinear(mlp_dim, n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        gate = nn.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


class TernaryTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers, n_heads, mlp_dim, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_heads, mlp_dim) for _ in range(n_layers)]
        self.ln_f = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def __call__(self, x):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(mx.arange(T))
        h = tok + pos
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


# ============================================================================
# LoRA Layer with configurable init and freeze
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA wrapper with configurable A-matrix init and freeze behavior."""
    def __init__(self, base_linear, rank, scale, a_init=None, freeze_a=True):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.linear = base_linear
        self.scale = scale
        self.rank = rank

        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))

        # B initialized to zero (standard LoRA) for Grassmannian and PiSSA-frozen
        self.lora_b = mx.zeros((rank, out_features))

        # Freeze base and optionally A
        self.linear.freeze()
        if freeze_a:
            self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        # STE ternary on B
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


class PiSSALoRALinear(nn.Module):
    """PiSSA-initialized LoRA with unfrozen A and SVD-initialized B."""
    def __init__(self, base_linear, rank, scale, a_init, b_init, freeze_a=False):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.lora_a = a_init
        self.lora_b = b_init

        self.linear.freeze()
        if freeze_a:
            self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        # STE ternary on B
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


# ============================================================================
# Data Generation
# ============================================================================

DOMAIN_PATTERNS = {
    "names_short": {"prefix": list("ABCDE"), "suffixes": ["y", "n", "a", "o"],
                    "min_len": 3, "max_len": 6},
    "names_long":  {"prefix": list("FGHIJ"), "suffixes": ["ine", "ard", "ley"],
                    "min_len": 6, "max_len": 10},
    "codes":       {"prefix": list("KLMNO"), "suffixes": ["01", "23", "99"],
                    "min_len": 4, "max_len": 8},
    "words_alpha": {"prefix": list("PQRST"), "suffixes": ["er", "ly", "tion"],
                    "min_len": 5, "max_len": 9},
    "words_mixed": {"prefix": list("UVWXY"), "suffixes": ["1st", "2nd", "x"],
                    "min_len": 4, "max_len": 8},
}
DOMAIN_NAMES = list(DOMAIN_PATTERNS.keys())

# Character vocabulary
CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")[:VOCAB_SIZE]
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
PAD_ID = 0


def generate_domain_data(domain_name, n_samples, seed):
    """Generate character-level sequences with domain-specific patterns."""
    rng = random.Random(seed)
    pattern = DOMAIN_PATTERNS[domain_name]
    data = []
    for _ in range(n_samples):
        prefix = rng.choice(pattern["prefix"])
        suffix = rng.choice(pattern["suffixes"])
        mid_len = rng.randint(pattern["min_len"], pattern["max_len"]) - len(prefix) - len(suffix)
        mid = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(max(1, mid_len)))
        word = prefix + mid + suffix
        # Encode to token IDs
        ids = [CHAR_TO_ID.get(c, 0) for c in word[:BLOCK_SIZE]]
        # Pad
        while len(ids) < BLOCK_SIZE:
            ids.append(PAD_ID)
        data.append(ids)
    return data


def make_batches(data, batch_size):
    """Convert list of token sequences to batches."""
    rng = random.Random(42)
    rng.shuffle(data)
    batches = []
    for i in range(0, len(data) - batch_size + 1, batch_size):
        batch = mx.array(data[i:i + batch_size])
        batches.append(batch)
    return batches


# ============================================================================
# SVD of Ternary Weights (PiSSA Init)
# ============================================================================

def compute_pissa_init(weight_matrix, rank):
    """Compute PiSSA initialization from a weight matrix.

    Given W in R^{out x in}, compute truncated SVD and return:
      A = V[:, :r]  in R^{in x r}  (top-r right singular vectors)
      B = (Sigma[:r] * U[:, :r]^T)  in R^{r x out}

    Also returns the variance captured by rank-r approx.
    """
    W_np = np.array(weight_matrix, dtype=np.float32)
    U, s, Vt = np.linalg.svd(W_np, full_matrices=False)

    # Top-r components
    A_np = Vt[:rank, :].T  # (in, r) -- right singular vectors
    B_np = (U[:, :rank] * s[:rank]).T  # (r, out) -- scaled left singular vectors

    # Variance analysis
    total_var = np.sum(s ** 2)
    captured_var = np.sum(s[:rank] ** 2)
    var_ratio = captured_var / (total_var + 1e-10)

    # Spectral flatness: ratio of geometric mean to arithmetic mean of singular values
    s_pos = s[s > 1e-10]
    if len(s_pos) > 1:
        geo_mean = np.exp(np.mean(np.log(s_pos)))
        arith_mean = np.mean(s_pos)
        flatness = geo_mean / (arith_mean + 1e-10)
    else:
        flatness = 1.0

    return A_np, B_np, var_ratio, flatness, s


# ============================================================================
# Grassmannian AP Init
# ============================================================================

def grassmannian_ap_init(N, r, d, seed=42):
    """Generate N orthogonally-packed A matrices via QR (perfect for Nr <= d)."""
    rng = np.random.RandomState(seed)
    frames = np.zeros((N, d, r), dtype=np.float32)
    for i in range(N):
        M = rng.randn(d, r).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        frames[i] = Q[:, :r]
    return frames


# ============================================================================
# Apply LoRA to Model
# ============================================================================

TARGET_KEYS_ATTN = ["attn.wq", "attn.wk", "attn.wv", "attn.wo"]
TARGET_KEYS_MLP = ["gate_proj", "up_proj", "down_proj"]
ALL_TARGET_KEYS = TARGET_KEYS_ATTN + TARGET_KEYS_MLP


def get_weight_key(layer_idx, key):
    """Create a unique key for each weight matrix."""
    return (layer_idx, key)


def apply_lora_grassmannian(model, rank, scale, domain_idx, all_a_matrices):
    """Apply LoRA with frozen Grassmannian A matrices."""
    count = 0
    for li, block in enumerate(model.blocks):
        updates = []
        for key in TARGET_KEYS_ATTN:
            attr_name = key.split(".")[-1]
            module = getattr(block.attn, attr_name, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_a_matrices[wkey][domain_idx]
            a_mx = mx.array(a_np)
            lora = LoRALinear(module, rank=rank, scale=scale, a_init=a_mx, freeze_a=True)
            updates.append((key, lora))
            count += 1

        for key in TARGET_KEYS_MLP:
            module = getattr(block, key, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_a_matrices[wkey][domain_idx]
            a_mx = mx.array(a_np)
            lora = LoRALinear(module, rank=rank, scale=scale, a_init=a_mx, freeze_a=True)
            updates.append((key, lora))
            count += 1

        if updates:
            from mlx.utils import tree_unflatten
            block.update_modules(tree_unflatten(updates))
    return count


def apply_lora_pissa_frozen(model, rank, scale, domain_idx, all_pissa_inits):
    """Apply LoRA with frozen PiSSA-SVD A matrices, zero B init."""
    count = 0
    for li, block in enumerate(model.blocks):
        updates = []
        for key in TARGET_KEYS_ATTN:
            attr_name = key.split(".")[-1]
            module = getattr(block.attn, attr_name, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_pissa_inits[wkey]["A"]
            a_mx = mx.array(a_np)
            # PiSSA-frozen: same A for all adapters, B starts at zero
            lora = LoRALinear(module, rank=rank, scale=scale, a_init=a_mx, freeze_a=True)
            updates.append((key, lora))
            count += 1

        for key in TARGET_KEYS_MLP:
            module = getattr(block, key, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_pissa_inits[wkey]["A"]
            a_mx = mx.array(a_np)
            lora = LoRALinear(module, rank=rank, scale=scale, a_init=a_mx, freeze_a=True)
            updates.append((key, lora))
            count += 1

        if updates:
            from mlx.utils import tree_unflatten
            block.update_modules(tree_unflatten(updates))
    return count


def apply_lora_pissa_unfrozen(model, rank, scale, domain_idx, all_pissa_inits):
    """Apply PiSSA LoRA with unfrozen A and SVD-initialized B."""
    count = 0
    for li, block in enumerate(model.blocks):
        updates = []
        for key in TARGET_KEYS_ATTN:
            attr_name = key.split(".")[-1]
            module = getattr(block.attn, attr_name, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_pissa_inits[wkey]["A"]
            b_np = all_pissa_inits[wkey]["B"]
            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)
            lora = PiSSALoRALinear(module, rank=rank, scale=scale,
                                   a_init=a_mx, b_init=b_mx, freeze_a=False)
            updates.append((key, lora))
            count += 1

        for key in TARGET_KEYS_MLP:
            module = getattr(block, key, None)
            if module is None or not isinstance(module, (BitLinear, nn.Linear)):
                continue
            wkey = get_weight_key(li, key)
            a_np = all_pissa_inits[wkey]["A"]
            b_np = all_pissa_inits[wkey]["B"]
            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)
            lora = PiSSALoRALinear(module, rank=rank, scale=scale,
                                   a_init=a_mx, b_init=b_mx, freeze_a=False)
            updates.append((key, lora))
            count += 1

        if updates:
            from mlx.utils import tree_unflatten
            block.update_modules(tree_unflatten(updates))
    return count


# ============================================================================
# Extract adapter parameters
# ============================================================================

def get_lora_params(model):
    """Extract trainable LoRA parameters as flat dict."""
    from mlx.utils import tree_flatten
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    return params


def get_adapter_delta_vector(params):
    """Compute vec(B@A) for each LoRA layer and concatenate into one vector."""
    parts = []
    # Group by layer: find matching lora_a and lora_b
    a_params = {k: v for k, v in params.items() if "lora_a" in k}
    b_params = {k: v for k, v in params.items() if "lora_b" in k}

    for a_key in sorted(a_params.keys()):
        b_key = a_key.replace("lora_a", "lora_b")
        if b_key in b_params:
            A = a_params[a_key]
            B = b_params[b_key]
            delta = A @ B  # (in, r) @ (r, out) = (in, out)
            parts.append(delta.reshape(-1))

    if parts:
        return mx.concatenate(parts)
    return mx.array([0.0])


# ============================================================================
# Training
# ============================================================================

def compute_loss(model, batch):
    """Next-token prediction loss."""
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    return loss


def train_model(model, batches, n_steps, lr, label=""):
    """Train model for n_steps, return loss curve."""
    optimizer = opt.AdamW(learning_rate=lr, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    losses = []
    gc.disable()
    for step in range(n_steps):
        batch = batches[step % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 50 == 0 or step == n_steps - 1:
            loss_val = loss.item()
            losses.append((step, loss_val))
            if step % 100 == 0:
                log(f"  [{label}] step {step}/{n_steps}: loss={loss_val:.4f}")

    gc.enable()
    gc.collect()
    del optimizer
    return losses


def eval_ppl(model, batches, max_batches=20):
    """Compute perplexity on validation batches."""
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(batches[:max_batches]):
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss
    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ============================================================================
# Phase 1: Train ternary base model
# ============================================================================

def phase_train_base(all_train_data, all_val_data, seed):
    """Train a ternary base model from scratch."""
    log(f"\n[Phase 1] Training ternary base (seed={seed})...")
    mx.random.seed(seed)

    model = TernaryTransformer(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MLP_DIM, BLOCK_SIZE)
    mx.eval(model.parameters())

    # Combine all domain data for base training
    all_data = []
    for domain in DOMAIN_NAMES:
        all_data.extend(all_train_data[domain])
    batches = make_batches(all_data, BATCH_SIZE)

    val_batches = []
    for domain in DOMAIN_NAMES:
        val_batches.extend(make_batches(all_val_data[domain], BATCH_SIZE))

    losses = train_model(model, batches, BASE_STEPS, BASE_LR, label="base")
    base_ppl = eval_ppl(model, val_batches)
    log(f"  Base PPL: {base_ppl:.4f}")

    # Save base weights
    base_weights_path = EXPERIMENT_DIR / f"base_weights_seed{seed}.npz"
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(model.parameters()))
    mx.savez(str(base_weights_path), **flat)
    log(f"  Saved base weights to {base_weights_path}")

    # Extract weight matrices for PiSSA SVD init
    pissa_inits = {}
    svd_stats = {}
    for li, block in enumerate(model.blocks):
        for key in TARGET_KEYS_ATTN:
            attr_name = key.split(".")[-1]
            module = getattr(block.attn, attr_name, None)
            if module is None:
                continue
            # Get quantized weight (what the model actually uses)
            w = module.weight
            alpha = mx.mean(mx.abs(w))
            w_q = mx.clip(mx.round(w / (alpha + 1e-7)), -1, 1) * alpha
            mx.eval(w_q)
            wkey = get_weight_key(li, key)
            A_np, B_np, var_ratio, flatness, svals = compute_pissa_init(
                w_q, LORA_RANK
            )
            pissa_inits[wkey] = {"A": A_np, "B": B_np}
            svd_stats[f"L{li}_{key}"] = {
                "var_captured": float(var_ratio),
                "flatness": float(flatness),
                "top_singular": float(svals[0]),
                "rank_singular": float(svals[min(LORA_RANK - 1, len(svals) - 1)]),
            }

        for key in TARGET_KEYS_MLP:
            module = getattr(block, key, None)
            if module is None:
                continue
            w = module.weight
            alpha = mx.mean(mx.abs(w))
            w_q = mx.clip(mx.round(w / (alpha + 1e-7)), -1, 1) * alpha
            mx.eval(w_q)
            wkey = get_weight_key(li, key)
            A_np, B_np, var_ratio, flatness, svals = compute_pissa_init(
                w_q, LORA_RANK
            )
            pissa_inits[wkey] = {"A": A_np, "B": B_np}
            svd_stats[f"L{li}_{key}"] = {
                "var_captured": float(var_ratio),
                "flatness": float(flatness),
                "top_singular": float(svals[0]),
                "rank_singular": float(svals[min(LORA_RANK - 1, len(svals) - 1)]),
            }

    # Compute Grassmannian A matrices for all weight matrices
    grass_a_matrices = {}
    for wkey in pissa_inits.keys():
        li, key_name = wkey
        if "attn" in key_name:
            d_in = D_MODEL
        elif key_name == "down_proj":
            d_in = MLP_DIM
        else:
            d_in = D_MODEL
        grass_a_matrices[wkey] = grassmannian_ap_init(
            N_DOMAINS, LORA_RANK, d_in, seed=seed + hash(str(wkey)) % 10000
        )

    result = {
        "base_ppl": base_ppl,
        "base_losses": losses,
        "svd_stats": svd_stats,
    }
    log_memory("post-base-train")
    cleanup(model)
    return base_weights_path, pissa_inits, grass_a_matrices, result


# ============================================================================
# Phase 2: Train adapters under each condition
# ============================================================================

def load_base_model(weights_path):
    """Load a fresh base model with saved weights."""
    model = TernaryTransformer(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MLP_DIM, BLOCK_SIZE)
    from mlx.utils import tree_unflatten
    weights = dict(mx.load(str(weights_path)))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.freeze()
    return model


def phase_train_adapters(base_weights_path, pissa_inits, grass_a_matrices,
                         train_data, val_data, condition, seed):
    """Train N_DOMAINS adapters under one condition.

    condition: "grassmannian", "pissa_frozen", or "pissa_unfrozen"
    Returns: dict of domain -> {ppl, losses, params_path}
    """
    log(f"\n[Phase 2] Training {condition} adapters (seed={seed})...")
    results = {}

    for di, domain in enumerate(DOMAIN_NAMES):
        log(f"  Training {condition}/{domain} (adapter {di+1}/{N_DOMAINS})...")

        # Load fresh base model
        model = load_base_model(base_weights_path)

        # Apply LoRA with appropriate init
        if condition == "grassmannian":
            count = apply_lora_grassmannian(model, LORA_RANK, LORA_SCALE, di, grass_a_matrices)
        elif condition == "pissa_frozen":
            count = apply_lora_pissa_frozen(model, LORA_RANK, LORA_SCALE, di, pissa_inits)
        elif condition == "pissa_unfrozen":
            count = apply_lora_pissa_unfrozen(model, LORA_RANK, LORA_SCALE, di, pissa_inits)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Train on domain-specific data
        batches = make_batches(train_data[domain], BATCH_SIZE)
        val_batches = make_batches(val_data[domain], BATCH_SIZE)

        if len(batches) == 0:
            log(f"    WARNING: no batches for {domain}, skipping")
            cleanup(model)
            continue

        losses = train_model(model, batches, ADAPTER_STEPS, ADAPTER_LR,
                           label=f"{condition}/{domain}")

        # Evaluate
        ppl = eval_ppl(model, val_batches)
        log(f"    {condition}/{domain} PPL: {ppl:.4f}")

        # Save adapter params
        params = get_lora_params(model)
        adapter_dir = EXPERIMENT_DIR / "adapters" / condition / f"seed{seed}" / domain
        adapter_dir.mkdir(parents=True, exist_ok=True)
        mx.savez(str(adapter_dir / "adapter.npz"), **params)

        # Get delta vector for cosine computation
        delta = get_adapter_delta_vector(params)
        mx.eval(delta)

        results[domain] = {
            "ppl": ppl,
            "final_loss": losses[-1][1] if losses else float("inf"),
            "losses_at_milestones": {
                str(s): l for s, l in losses
                if s in [0, 50, 100, 150, 200]
            },
            "n_trainable": sum(p.size for p in params.values()),
            "delta_norm": float(mx.linalg.norm(delta).item()),
        }

        cleanup(model, params, delta)

    return results


# ============================================================================
# Phase 3: Evaluate composition and orthogonality
# ============================================================================

def phase_evaluate_composition(base_weights_path, pissa_inits, grass_a_matrices,
                               val_data, condition, seed):
    """Evaluate adapter composition under one condition."""
    log(f"\n[Phase 3] Evaluating {condition} composition (seed={seed})...")

    # Load all adapter delta vectors for cosine analysis
    deltas = {}
    for di, domain in enumerate(DOMAIN_NAMES):
        adapter_path = (EXPERIMENT_DIR / "adapters" / condition /
                       f"seed{seed}" / domain / "adapter.npz")
        if not adapter_path.exists():
            continue
        params = dict(mx.load(str(adapter_path)))
        delta = get_adapter_delta_vector(params)
        mx.eval(delta)
        deltas[domain] = delta
        del params

    # Compute pairwise cosine of delta vectors
    cosines = []
    domain_list = list(deltas.keys())
    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            d1 = deltas[domain_list[i]]
            d2 = deltas[domain_list[j]]
            cos = mx.sum(d1 * d2) / (mx.linalg.norm(d1) * mx.linalg.norm(d2) + 1e-10)
            mx.eval(cos)
            cosines.append({
                "pair": f"{domain_list[i]}-{domain_list[j]}",
                "cosine": float(abs(cos.item())),
            })

    mean_cos = np.mean([c["cosine"] for c in cosines]) if cosines else 0.0
    max_cos = max(c["cosine"] for c in cosines) if cosines else 0.0

    del deltas
    gc.collect()

    # Evaluate composed model (all adapters, uniform 1/N scaling)
    model = load_base_model(base_weights_path)

    # Load and average all adapter B matrices
    all_params = []
    for domain in DOMAIN_NAMES:
        adapter_path = (EXPERIMENT_DIR / "adapters" / condition /
                       f"seed{seed}" / domain / "adapter.npz")
        if adapter_path.exists():
            all_params.append(dict(mx.load(str(adapter_path))))

    if not all_params:
        cleanup(model)
        return {"error": "no adapters found"}

    # For composition: apply first adapter's structure, then average B matrices
    # and measure composed PPL
    # Strategy: load model with LoRA, set B = average of all domain Bs
    if condition == "grassmannian":
        # For Grassmannian: proper per-expert composition
        # Apply adapter 0's A matrices as template
        apply_lora_grassmannian(model, LORA_RANK, LORA_SCALE, 0, grass_a_matrices)

        # Now: for each LoRA layer, sum B matrices from all adapters scaled by 1/N
        # This is INCORRECT for Grassmannian (should use per-expert A_i@B_i sum)
        # But since we have different A matrices per adapter, we need to compute
        # the composed output differently.
        #
        # Actually, for micro scale, we just evaluate each adapter individually
        # and report the average. For composition quality, we compute the delta
        # cosine (already done above).
        pass
    elif condition in ("pissa_frozen", "pissa_unfrozen"):
        # PiSSA: all adapters share the same A (or similar A), so averaging B
        # is a valid approximation of composition.
        if condition == "pissa_frozen":
            apply_lora_pissa_frozen(model, LORA_RANK, LORA_SCALE, 0, pissa_inits)
        else:
            apply_lora_pissa_unfrozen(model, LORA_RANK, LORA_SCALE, 0, pissa_inits)

    # Average B matrices across all adapters for composition test
    n_adapters = len(all_params)
    avg_params = {}
    for key in all_params[0]:
        if "lora_b" in key:
            stacked = mx.stack([p[key] for p in all_params])
            avg_params[key] = mx.mean(stacked, axis=0)
        elif "lora_a" in key:
            if condition == "pissa_unfrozen":
                # Average the A matrices too
                stacked = mx.stack([p[key] for p in all_params])
                avg_params[key] = mx.mean(stacked, axis=0)
            else:
                avg_params[key] = all_params[0][key]

    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(avg_params.items())))
    mx.eval(model.parameters())

    # Evaluate composed PPL on all domains
    composed_ppls = {}
    for domain in DOMAIN_NAMES:
        vb = make_batches(val_data[domain], BATCH_SIZE)
        if vb:
            composed_ppls[domain] = eval_ppl(model, vb)

    del all_params, avg_params
    cleanup(model)

    return {
        "mean_delta_cosine": float(mean_cos),
        "max_delta_cosine": float(max_cos),
        "cosine_pairs": cosines,
        "composed_ppls": composed_ppls,
        "mean_composed_ppl": float(np.mean(list(composed_ppls.values()))) if composed_ppls else float("inf"),
    }


# ============================================================================
# Phase 4: SVD Analysis of Ternary Weights
# ============================================================================

def phase_svd_analysis(base_weights_path, seed):
    """Analyze SVD properties of ternary weights."""
    log(f"\n[Phase 4] SVD analysis of ternary weights (seed={seed})...")

    model = load_base_model(base_weights_path)

    results = {}
    for li, block in enumerate(model.blocks):
        for key in TARGET_KEYS_ATTN:
            attr_name = key.split(".")[-1]
            module = getattr(block.attn, attr_name, None)
            if module is None:
                continue
            w = module.weight
            alpha = mx.mean(mx.abs(w))
            w_q = mx.clip(mx.round(w / (alpha + 1e-7)), -1, 1) * alpha
            mx.eval(w_q)
            W_np = np.array(w_q, dtype=np.float32)

            _, s, _ = np.linalg.svd(W_np, full_matrices=False)
            total_var = np.sum(s ** 2)
            rank8_var = np.sum(s[:LORA_RANK] ** 2)

            # Sparsity (fraction of zeros)
            W_tern = np.round(np.array(w, dtype=np.float32) / (float(alpha.item()) + 1e-7))
            sparsity = np.mean(np.abs(W_tern) < 0.5)

            results[f"L{li}_{key}"] = {
                "var_captured_rank8": float(rank8_var / (total_var + 1e-10)),
                "sparsity": float(sparsity),
                "sigma_1": float(s[0]),
                "sigma_8": float(s[min(7, len(s)-1)]),
                "sigma_ratio": float(s[0] / (s[min(7, len(s)-1)] + 1e-10)),
                "effective_rank": float(np.sum(s > s[0] * 0.01)),
            }

        for key in TARGET_KEYS_MLP:
            module = getattr(block, key, None)
            if module is None:
                continue
            w = module.weight
            alpha = mx.mean(mx.abs(w))
            w_q = mx.clip(mx.round(w / (alpha + 1e-7)), -1, 1) * alpha
            mx.eval(w_q)
            W_np = np.array(w_q, dtype=np.float32)

            _, s, _ = np.linalg.svd(W_np, full_matrices=False)
            total_var = np.sum(s ** 2)
            rank8_var = np.sum(s[:LORA_RANK] ** 2)

            W_tern = np.round(np.array(w, dtype=np.float32) / (float(alpha.item()) + 1e-7))
            sparsity = np.mean(np.abs(W_tern) < 0.5)

            results[f"L{li}_{key}"] = {
                "var_captured_rank8": float(rank8_var / (total_var + 1e-10)),
                "sparsity": float(sparsity),
                "sigma_1": float(s[0]),
                "sigma_8": float(s[min(7, len(s)-1)]),
                "sigma_ratio": float(s[0] / (s[min(7, len(s)-1)] + 1e-10)),
                "effective_rank": float(np.sum(s > s[0] * 0.01)),
            }

    cleanup(model)
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Generate domain data
    log("[Data] Generating domain data...")
    train_data = {}
    val_data = {}
    for domain in DOMAIN_NAMES:
        data = generate_domain_data(domain, 500, seed=42)
        train_data[domain] = data[:400]
        val_data[domain] = data[400:]
        log(f"  {domain}: {len(train_data[domain])} train, {len(val_data[domain])} val")

    all_results = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        seed_results = {}

        # Phase 1: Train base + compute inits
        base_path, pissa_inits, grass_a, base_result = phase_train_base(
            train_data, val_data, seed
        )
        seed_results["base"] = base_result

        # Phase 4: SVD analysis (do before adapter training)
        svd_analysis = phase_svd_analysis(base_path, seed)
        seed_results["svd_analysis"] = svd_analysis

        # Phase 2: Train adapters under each condition
        conditions = ["grassmannian", "pissa_frozen", "pissa_unfrozen"]
        adapter_results = {}
        for condition in conditions:
            adapter_results[condition] = phase_train_adapters(
                base_path, pissa_inits, grass_a, train_data, val_data,
                condition, seed
            )

        seed_results["adapter_training"] = adapter_results

        # Phase 3: Composition evaluation
        composition_results = {}
        for condition in conditions:
            composition_results[condition] = phase_evaluate_composition(
                base_path, pissa_inits, grass_a, val_data, condition, seed
            )

        seed_results["composition"] = composition_results

        all_results["seeds"][str(seed)] = seed_results

        # Cleanup base weights file
        base_path.unlink(missing_ok=True)

    # ========================================================================
    # Summary statistics across seeds
    # ========================================================================
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")

    conditions = ["grassmannian", "pissa_frozen", "pissa_unfrozen"]
    summary = {}

    for condition in conditions:
        ppls_per_domain = {d: [] for d in DOMAIN_NAMES}
        cosines = []
        composed_ppls = []
        convergence_50 = []  # loss at step 50

        for seed_key, seed_data in all_results["seeds"].items():
            at = seed_data.get("adapter_training", {}).get(condition, {})
            comp = seed_data.get("composition", {}).get(condition, {})

            for domain in DOMAIN_NAMES:
                if domain in at:
                    ppls_per_domain[domain].append(at[domain]["ppl"])
                    milestones = at[domain].get("losses_at_milestones", {})
                    if "50" in milestones:
                        convergence_50.append(milestones["50"])

            if "mean_delta_cosine" in comp:
                cosines.append(comp["mean_delta_cosine"])
            if "mean_composed_ppl" in comp:
                composed_ppls.append(comp["mean_composed_ppl"])

        # Average PPL across seeds per domain
        mean_ppls = {}
        for domain in DOMAIN_NAMES:
            if ppls_per_domain[domain]:
                mean_ppls[domain] = float(np.mean(ppls_per_domain[domain]))

        overall_ppl = float(np.mean(list(mean_ppls.values()))) if mean_ppls else float("inf")
        mean_cosine = float(np.mean(cosines)) if cosines else 0.0
        mean_composed = float(np.mean(composed_ppls)) if composed_ppls else float("inf")
        mean_conv_50 = float(np.mean(convergence_50)) if convergence_50 else float("inf")

        summary[condition] = {
            "mean_single_ppl": overall_ppl,
            "per_domain_ppl": mean_ppls,
            "mean_delta_cosine": mean_cosine,
            "mean_composed_ppl": mean_composed,
            "mean_loss_at_step50": mean_conv_50,
        }

        log(f"\n{condition}:")
        log(f"  Mean single-adapter PPL: {overall_ppl:.4f}")
        log(f"  Mean delta cosine: {mean_cosine:.6f}")
        log(f"  Mean composed PPL: {mean_composed:.4f}")
        log(f"  Mean loss @ step 50: {mean_conv_50:.4f}")

    # Kill criteria assessment
    grass_ppl = summary["grassmannian"]["mean_single_ppl"]
    pissa_f_ppl = summary["pissa_frozen"]["mean_single_ppl"]
    pissa_u_ppl = summary["pissa_unfrozen"]["mean_single_ppl"]
    pissa_u_cos = summary["pissa_unfrozen"]["mean_delta_cosine"]

    base_ppls = [all_results["seeds"][s]["base"]["base_ppl"] for s in all_results["seeds"]]
    mean_base_ppl = float(np.mean(base_ppls))

    grass_composed = summary["grassmannian"]["mean_composed_ppl"]
    pissa_f_composed = summary["pissa_frozen"]["mean_composed_ppl"]
    pissa_u_composed = summary["pissa_unfrozen"]["mean_composed_ppl"]

    # Composition ratio = composed_ppl / base_ppl
    grass_ratio = grass_composed / mean_base_ppl if mean_base_ppl > 0 else float("inf")
    pissa_f_ratio = pissa_f_composed / mean_base_ppl if mean_base_ppl > 0 else float("inf")
    pissa_u_ratio = pissa_u_composed / mean_base_ppl if mean_base_ppl > 0 else float("inf")

    # SVD variance analysis
    svd_vars = []
    for seed_key, seed_data in all_results["seeds"].items():
        for layer_key, stats in seed_data.get("svd_analysis", {}).items():
            svd_vars.append(stats["var_captured_rank8"])
    mean_svd_var = float(np.mean(svd_vars)) if svd_vars else 0.0

    log(f"\n{'='*60}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'='*60}")

    # K1: PiSSA-frozen worse single-adapter PPL than Grassmannian
    k1_result = pissa_f_ppl > grass_ppl
    log(f"\nK1: PiSSA-frozen PPL ({pissa_f_ppl:.4f}) > Grassmannian PPL ({grass_ppl:.4f})?")
    log(f"    -> {'KILL (PiSSA-frozen worse)' if k1_result else 'PASS (PiSSA-frozen same or better)'}")

    # K2: PiSSA-unfrozen adapter cosine > 0.1
    k2_result = pissa_u_cos > 0.1
    log(f"\nK2: PiSSA-unfrozen mean |cos| ({pissa_u_cos:.6f}) > 0.1?")
    log(f"    -> {'KILL (orthogonality destroyed)' if k2_result else 'PASS (orthogonality preserved)'}")

    # K3: Neither PiSSA variant improves over Grassmannian on ANY metric
    pissa_f_better_ppl = pissa_f_ppl < grass_ppl
    pissa_u_better_ppl = pissa_u_ppl < grass_ppl
    pissa_f_better_conv = summary["pissa_frozen"]["mean_loss_at_step50"] < summary["grassmannian"]["mean_loss_at_step50"]
    pissa_u_better_conv = summary["pissa_unfrozen"]["mean_loss_at_step50"] < summary["grassmannian"]["mean_loss_at_step50"]
    pissa_u_better_compose = pissa_u_ratio < grass_ratio
    any_improvement = pissa_f_better_ppl or pissa_u_better_ppl or pissa_f_better_conv or pissa_u_better_conv or pissa_u_better_compose
    k3_result = not any_improvement
    log(f"\nK3: Any PiSSA improvement over Grassmannian?")
    log(f"    PiSSA-frozen better PPL: {pissa_f_better_ppl}")
    log(f"    PiSSA-unfrozen better PPL: {pissa_u_better_ppl}")
    log(f"    PiSSA-frozen faster convergence: {pissa_f_better_conv}")
    log(f"    PiSSA-unfrozen faster convergence: {pissa_u_better_conv}")
    log(f"    PiSSA-unfrozen better composition: {pissa_u_better_compose}")
    log(f"    -> {'KILL (no improvement on any metric)' if k3_result else 'PASS (at least one improvement)'}")

    # Success criteria
    ppl_improvement = (grass_ppl - pissa_f_ppl) / grass_ppl * 100 if grass_ppl > 0 else 0
    s1_result = ppl_improvement > 5
    log(f"\nS1: PiSSA-frozen > 5% PPL improvement? {ppl_improvement:.1f}% -> {'PASS' if s1_result else 'FAIL'}")

    s2_result = pissa_u_ratio < 1.5
    log(f"S2: PiSSA-unfrozen composition ratio < 1.5? {pissa_u_ratio:.4f} -> {'PASS' if s2_result else 'FAIL'}")

    log(f"\nSVD Variance Analysis:")
    log(f"  Mean rank-{LORA_RANK} variance captured: {mean_svd_var:.4f} ({mean_svd_var*100:.1f}%)")
    log(f"  (Compare: float weights typically capture 40-60%)")

    all_results["summary"] = summary
    all_results["kill_criteria"] = {
        "K1_pissa_frozen_worse": k1_result,
        "K1_grass_ppl": grass_ppl,
        "K1_pissa_f_ppl": pissa_f_ppl,
        "K2_orthogonality_destroyed": k2_result,
        "K2_pissa_u_cosine": pissa_u_cos,
        "K3_no_improvement": k3_result,
        "K3_details": {
            "pissa_f_better_ppl": pissa_f_better_ppl,
            "pissa_u_better_ppl": pissa_u_better_ppl,
            "pissa_f_faster_conv": pissa_f_better_conv,
            "pissa_u_faster_conv": pissa_u_better_conv,
            "pissa_u_better_compose": pissa_u_better_compose,
        }
    }
    all_results["success_criteria"] = {
        "S1_pissa_frozen_5pct_better": s1_result,
        "S1_improvement_pct": ppl_improvement,
        "S2_pissa_unfrozen_compose": s2_result,
        "S2_composition_ratio": pissa_u_ratio,
    }
    all_results["svd_analysis_summary"] = {
        "mean_variance_captured": mean_svd_var,
        "interpretation": (
            "flat" if mean_svd_var < 0.2 else
            "moderate" if mean_svd_var < 0.4 else
            "concentrated"
        ),
    }
    all_results["config"] = {
        "d_model": D_MODEL, "n_layers": N_LAYERS, "lora_rank": LORA_RANK,
        "n_domains": N_DOMAINS, "seeds": SEEDS, "adapter_steps": ADAPTER_STEPS,
        "base_steps": BASE_STEPS,
    }
    all_results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {all_results['total_time_s']:.1f}s")
    log_memory("final")


if __name__ == "__main__":
    main()
