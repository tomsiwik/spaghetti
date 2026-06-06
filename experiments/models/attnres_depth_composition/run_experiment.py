#!/usr/bin/env python3
"""
AttnRes Depth-Wise Attention for LoRA Adapter Composition.

Tests whether replacing standard residual connections with depth-wise softmax
attention (AttnRes, arXiv 2603.15031) improves how LoRA adapters compose
across layers.

Kill criteria:
  K1 (525): AttnRes model >10% worse than standard on base quality
  K2 (526): No composition improvement (AttnRes ratio >= standard, 3-seed)
  K3 (527): Depth attention weights uniform (doesn't specialize for adapters)

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# ============================================================================
# Hyperparameters
# ============================================================================
VOCAB_SIZE = 256  # character-level
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
N_LAYERS = 4
FFN_DIM = D_MODEL * 4  # 512
MAX_SEQ_LEN = 128
DROPOUT = 0.0

# Training
TRAIN_STEPS = 600
LEARNING_RATE = 3e-4
BATCH_SIZE = 16

# LoRA
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 2.0

# Domains (character-level synthetic — 5 separable pattern generators)
N_DOMAINS = 5
DOMAIN_NAMES = ["alpha", "numeric", "mixed", "upper", "symbol"]
ADAPTER_TRAIN_STEPS = 400
ADAPTER_LR = 5e-4

# Multi-seed
SEEDS = [42, 137, 314]

# AttnRes block config (2 blocks of 2 layers each)
ATTNRES_N_BLOCKS = 2


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
# Data generation (character-level, separable domains)
# ============================================================================
def generate_domain_data(domain, n_samples, seq_len, seed=42):
    """Generate character-level sequences with domain-specific patterns."""
    import random as pyrandom
    rng = pyrandom.Random(seed)

    if domain == "alpha":
        chars = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz"]
        data = []
        for i in range(n_samples):
            start = rng.randint(0, 25)
            seq = [chars[(start + j) % 26] for j in range(seq_len)]
            data.append(seq)
    elif domain == "numeric":
        chars = [ord(c) for c in "0123456789"]
        data = []
        for i in range(n_samples):
            start = rng.randint(0, 9)
            seq = [chars[(start + j) % 10] for j in range(seq_len)]
            data.append(seq)
    elif domain == "mixed":
        data = []
        for i in range(n_samples):
            start = rng.randint(0, 25)
            seq = []
            for j in range(seq_len):
                if j % 2 == 0:
                    seq.append(ord('a') + (start + j // 2) % 26)
                else:
                    seq.append(ord('0') + (j // 2) % 10)
            data.append(seq)
    elif domain == "upper":
        chars = [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        data = []
        for i in range(n_samples):
            start = rng.randint(0, 25)
            seq = [chars[(start + j) % 26] for j in range(seq_len)]
            data.append(seq)
    elif domain == "symbol":
        symbols = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
        chars = [ord(c) for c in symbols]
        n_sym = len(symbols)
        data = []
        for i in range(n_samples):
            start = rng.randint(0, n_sym - 1)
            seq = [chars[(start + j) % n_sym] for j in range(seq_len)]
            data.append(seq)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return mx.array(data, dtype=mx.int32)


def generate_base_data(n_samples, seq_len, seed=42):
    """Generate mixed data from all domains for base training."""
    per_domain = n_samples // N_DOMAINS
    all_data = []
    for i, domain in enumerate(DOMAIN_NAMES):
        data = generate_domain_data(domain, per_domain, seq_len, seed=seed + i)
        all_data.append(data)
    return mx.concatenate(all_data, axis=0)


# ============================================================================
# Model: Micro Transformer with optional AttnRes
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = 1e-6

    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(self.d_head)

    def __call__(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FFN(d_model, ffn_dim)

    def __call__(self, x, mask=None):
        """Returns the layer OUTPUT (not residual-added), for AttnRes compatibility."""
        attn_out = self.attn(self.attn_norm(x), mask=mask)
        h = x + attn_out  # internal residual for attention sublayer
        ffn_out = self.ffn(self.ffn_norm(h))
        return ffn_out  # Return the FFN output (the "value" for depth attention)

    def forward_with_residual(self, x, mask=None):
        """Standard forward with residual connection."""
        v = self(x, mask=mask)
        return x + v


class DepthAttention(nn.Module):
    """AttnRes: depth-wise softmax attention over layer outputs.

    For each target layer, a learned pseudo-query w ∈ R^d computes attention
    weights over all preceding layer outputs via softmax(w^T · RMSNorm(v_i)).
    """

    def __init__(self, d_model, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # One pseudo-query per layer, zero-initialized for uniform start
        self.pseudo_queries = mx.zeros((n_layers, d_model))
        self.depth_norms = [RMSNorm(d_model) for _ in range(n_layers + 1)]

    def __call__(self, values, target_layer_idx):
        """Compute AttnRes attention for target_layer_idx over values[0:target_layer_idx].

        Args:
            values: list of layer outputs [v_0, v_1, ..., v_{l-1}], each ∈ R^{B×T×d}
            target_layer_idx: which layer's pseudo-query to use (0-indexed)

        Returns:
            h_l = Σ_i α_{i→l} · v_i, shape R^{B×T×d}
        """
        n_sources = len(values)
        if n_sources == 0:
            raise ValueError("No values to attend over")
        if n_sources == 1:
            return values[0]

        w = self.pseudo_queries[target_layer_idx]  # (d,)

        # Compute scores for each source
        scores = []
        for i, v in enumerate(values):
            v_normed = self.depth_norms[i](v)  # (B, T, d)
            s = mx.sum(v_normed * w, axis=-1, keepdims=True)  # (B, T, 1)
            scores.append(s)

        # Stack scores: (B, T, n_sources)
        scores_stacked = mx.concatenate(scores, axis=-1)  # (B, T, n_sources)
        alpha = mx.softmax(scores_stacked, axis=-1)  # (B, T, n_sources)

        # Weighted sum over depth
        # values_stacked: (B, T, n_sources, d)
        values_stacked = mx.stack(values, axis=2)  # (B, T, n_sources, d)
        alpha_expanded = alpha[:, :, :, None]  # (B, T, n_sources, 1)
        h = mx.sum(values_stacked * alpha_expanded, axis=2)  # (B, T, d)

        return h, alpha  # Return attention weights for analysis


class MicroTransformer(nn.Module):
    """Micro transformer with optional AttnRes residual connections."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim, use_attnres=False):
        super().__init__()
        self.use_attnres = use_attnres
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if use_attnres:
            self.depth_attn = DepthAttention(d_model, n_layers)

    def __call__(self, x, return_depth_weights=False):
        B, T = x.shape
        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)

        h = self.embed(x)
        depth_weights = []

        if self.use_attnres:
            # AttnRes: collect layer outputs, attend over depth
            values = [h]  # v_0 = embedding output

            for i, layer in enumerate(self.layers):
                v = layer(h, mask=mask)  # layer output (not residual-added)
                values.append(v)
                # Compute h for next layer using depth attention over all values so far
                h, alpha = self.depth_attn(values, i)
                if return_depth_weights:
                    depth_weights.append(alpha)
        else:
            # Standard residual connections
            for layer in self.layers:
                h = layer.forward_with_residual(h, mask=mask)

        h = self.norm(h)
        logits = self.head(h)

        if return_depth_weights:
            return logits, depth_weights
        return logits


# ============================================================================
# LoRA layer for micro model
# ============================================================================
class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear."""

    def __init__(self, base_linear, rank, scale):
        super().__init__()
        in_dim = base_linear.weight.shape[1]
        out_dim = base_linear.weight.shape[0]
        self.base = base_linear
        self.lora_a = mx.random.normal(shape=(in_dim, rank)) * (1.0 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((rank, out_dim))
        self.scale = scale

    def __call__(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scale
        return base_out + lora_out


def apply_lora(model, rank, scale):
    """Apply LoRA to all attention and FFN projections."""
    count = 0
    for layer in model.layers:
        # Attention projections
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(layer.attn, proj_name)
            lora = LoRALinear(base, rank, scale)
            setattr(layer.attn, proj_name, lora)
            count += 1
        # FFN projections
        for proj_name in ["gate", "up", "down"]:
            base = getattr(layer.ffn, proj_name)
            lora = LoRALinear(base, rank, scale)
            setattr(layer.ffn, proj_name, lora)
            count += 1
    return count


def get_lora_params(model):
    """Extract LoRA parameters."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora(model):
    """Reset LoRA B matrices to zero, reinit A."""
    for layer in model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(layer.attn, proj_name)
            if isinstance(mod, LoRALinear):
                in_dim = mod.lora_a.shape[0]
                mod.lora_a = mx.random.normal(shape=mod.lora_a.shape) * (1.0 / math.sqrt(in_dim))
                mod.lora_b = mx.zeros(mod.lora_b.shape)
        for proj_name in ["gate", "up", "down"]:
            mod = getattr(layer.ffn, proj_name)
            if isinstance(mod, LoRALinear):
                in_dim = mod.lora_a.shape[0]
                mod.lora_a = mx.random.normal(shape=mod.lora_a.shape) * (1.0 / math.sqrt(in_dim))
                mod.lora_b = mx.zeros(mod.lora_b.shape)
    mx.eval(model.parameters())


def save_lora_params(params, path):
    """Save LoRA parameters to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path), **params)


def load_lora_params(path):
    """Load LoRA parameters from disk."""
    return dict(mx.load(str(path)))


def apply_lora_params(model, params):
    """Apply saved LoRA parameters to model."""
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(params.items())))


# ============================================================================
# Training utilities
# ============================================================================
def compute_loss(model, x):
    """Cross-entropy loss for next-token prediction."""
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    return loss


def compute_ppl(model, data, batch_size=16):
    """Compute perplexity on data."""
    total_loss = 0.0
    total_tokens = 0
    n_batches = len(data) // batch_size
    if n_batches == 0:
        n_batches = 1

    for i in range(min(n_batches, 20)):  # Cap at 20 batches
        batch = data[i * batch_size:(i + 1) * batch_size]
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.size
        del logits, loss

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


def get_depth_weights(model, data, batch_size=16):
    """Extract depth attention weights from AttnRes model."""
    if not model.use_attnres:
        return None

    all_weights = []
    n_batches = min(len(data) // batch_size, 10)
    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        _, depth_weights = model(batch[:, :-1], return_depth_weights=True)
        # depth_weights is list of (B, T, n_sources) tensors, one per layer
        layer_means = []
        for alpha in depth_weights:
            mx.eval(alpha)
            # Mean over batch and sequence positions
            mean_alpha = mx.mean(alpha, axis=(0, 1))  # (n_sources,)
            layer_means.append(mean_alpha.tolist())
        all_weights.append(layer_means)
        del depth_weights

    # Average across batches
    avg_weights = []
    for layer_idx in range(len(all_weights[0])):
        layer_avg = [0.0] * len(all_weights[0][layer_idx])
        for batch_weights in all_weights:
            for j, w in enumerate(batch_weights[layer_idx]):
                layer_avg[j] += w / len(all_weights)
        avg_weights.append(layer_avg)

    return avg_weights


# ============================================================================
# Phase 1: Train base model (standard or AttnRes)
# ============================================================================
def phase_train_base(use_attnres, seed):
    """Train a micro transformer base model."""
    mx.random.seed(seed)
    log(f"\n{'='*60}")
    log(f"  Training base model (attnres={use_attnres}, seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, use_attnres=use_attnres)
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Total params: {n_params:,}")

    # Generate training and validation data
    train_data = generate_base_data(500, MAX_SEQ_LEN, seed=seed)
    val_data = generate_base_data(100, MAX_SEQ_LEN, seed=seed + 1000)
    mx.eval(train_data, val_data)

    optimizer = opt.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    gc.disable()
    for step in range(TRAIN_STEPS):
        idx = (step * BATCH_SIZE) % len(train_data)
        batch = train_data[idx:idx + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = train_data[:BATCH_SIZE]

        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if (step + 1) % 100 == 0:
            log(f"    Step {step+1}/{TRAIN_STEPS}: loss={loss.item():.4f}")
    gc.enable()
    gc.collect()

    # Evaluate
    base_ppl = compute_ppl(model, val_data)
    elapsed = time.time() - t0
    log(f"  Base PPL: {base_ppl:.4f}, Time: {elapsed:.1f}s")

    result = {
        "base_ppl": round(base_ppl, 4),
        "n_params": n_params,
        "train_time_s": round(elapsed, 1),
    }

    # Get depth weights if AttnRes
    if use_attnres:
        depth_weights = get_depth_weights(model, val_data)
        result["base_depth_weights"] = depth_weights
        log(f"  Depth weights (base): {depth_weights}")

    # Save model weights
    tag = "attnres" if use_attnres else "standard"
    model_path = EXPERIMENT_DIR / f"models/{tag}_seed{seed}.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    params = dict(tree_flatten(model.parameters()))
    mx.savez(str(model_path), **params)

    cleanup(model, optimizer, train_data, val_data)
    return result


# ============================================================================
# Phase 2: Train domain adapters
# ============================================================================
def phase_train_adapters(use_attnres, seed):
    """Train LoRA adapters for each domain."""
    mx.random.seed(seed + 500)
    tag = "attnres" if use_attnres else "standard"
    log(f"\n{'='*60}")
    log(f"  Training adapters ({tag}, seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    # Load base model
    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, use_attnres=use_attnres)
    model_path = EXPERIMENT_DIR / f"models/{tag}_seed{seed}.npz"
    from mlx.utils import tree_unflatten
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved

    # Freeze base, apply LoRA
    model.freeze()
    n_lora = apply_lora(model, LORA_RANK, LORA_SCALE)
    log(f"  Applied LoRA to {n_lora} layers")

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {n_trainable:,}")

    adapter_results = {}
    for domain in DOMAIN_NAMES:
        log(f"\n  [{domain}] Training adapter...")
        t_d = time.time()

        zero_lora(model)

        # Generate domain-specific data
        train_data = generate_domain_data(domain, 200, MAX_SEQ_LEN, seed=seed + hash(domain) % 10000)
        val_data = generate_domain_data(domain, 50, MAX_SEQ_LEN, seed=seed + hash(domain) % 10000 + 5000)
        mx.eval(train_data, val_data)

        optimizer = opt.Adam(learning_rate=ADAPTER_LR)
        loss_and_grad = nn.value_and_grad(model, compute_loss)

        gc.disable()
        for step in range(ADAPTER_TRAIN_STEPS):
            idx = (step * BATCH_SIZE) % len(train_data)
            batch = train_data[idx:idx + BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                batch = train_data[:BATCH_SIZE]

            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            if (step + 1) % 100 == 0:
                log(f"    Step {step+1}/{ADAPTER_TRAIN_STEPS}: loss={loss.item():.4f}")
        gc.enable()
        gc.collect()

        # Evaluate adapter individually
        individual_ppl = compute_ppl(model, val_data)

        # Save adapter
        lora_params = get_lora_params(model)
        adapter_path = EXPERIMENT_DIR / f"adapters/{tag}_seed{seed}/{domain}.npz"
        save_lora_params(lora_params, adapter_path)

        adapter_results[domain] = {
            "individual_ppl": round(individual_ppl, 4),
            "train_time_s": round(time.time() - t_d, 1),
        }
        log(f"    {domain}: PPL={individual_ppl:.4f}, time={time.time()-t_d:.1f}s")

        del train_data, val_data, optimizer, lora_params

    elapsed = time.time() - t0
    log(f"  Total adapter training: {elapsed:.1f}s")

    cleanup(model)
    return adapter_results


# ============================================================================
# Phase 3: Evaluate composition
# ============================================================================
def phase_evaluate_composition(use_attnres, seed):
    """Evaluate composition quality: base PPL, individual PPL, composed PPL."""
    mx.random.seed(seed + 1000)
    tag = "attnres" if use_attnres else "standard"
    log(f"\n{'='*60}")
    log(f"  Evaluating composition ({tag}, seed={seed})")
    log(f"{'='*60}")

    # Load base model
    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, use_attnres=use_attnres)
    model_path = EXPERIMENT_DIR / f"models/{tag}_seed{seed}.npz"
    from mlx.utils import tree_unflatten
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved

    # Generate mixed validation data
    val_data = generate_base_data(200, MAX_SEQ_LEN, seed=seed + 2000)
    mx.eval(val_data)

    # Domain-specific val data
    domain_val = {}
    for domain in DOMAIN_NAMES:
        dv = generate_domain_data(domain, 50, MAX_SEQ_LEN, seed=seed + hash(domain) % 10000 + 5000)
        mx.eval(dv)
        domain_val[domain] = dv

    # Base PPL (no adapters)
    base_ppl = compute_ppl(model, val_data)
    log(f"  Base PPL (mixed): {base_ppl:.4f}")

    base_domain_ppls = {}
    for domain in DOMAIN_NAMES:
        ppl = compute_ppl(model, domain_val[domain])
        base_domain_ppls[domain] = round(ppl, 4)
    log(f"  Base domain PPLs: {base_domain_ppls}")

    # Apply LoRA
    model.freeze()
    apply_lora(model, LORA_RANK, LORA_SCALE)

    # Evaluate individual adapters
    individual_ppls = {}
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/{tag}_seed{seed}/{domain}.npz"
        params = load_lora_params(adapter_path)
        apply_lora_params(model, params)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, domain_val[domain])
        individual_ppls[domain] = round(ppl, 4)
        del params

    log(f"  Individual PPLs: {individual_ppls}")

    # Composed PPL (1/N scaling)
    # Load all adapters and average with 1/N
    all_adapter_params = {}
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/{tag}_seed{seed}/{domain}.npz"
        params = load_lora_params(adapter_path)
        for k, v in params.items():
            if k not in all_adapter_params:
                all_adapter_params[k] = v / N_DOMAINS
            else:
                all_adapter_params[k] = all_adapter_params[k] + v / N_DOMAINS

    apply_lora_params(model, all_adapter_params)
    mx.eval(model.parameters())
    del all_adapter_params

    composed_ppl = compute_ppl(model, val_data)
    log(f"  Composed PPL (mixed, 1/N): {composed_ppl:.4f}")

    composed_domain_ppls = {}
    for domain in DOMAIN_NAMES:
        ppl = compute_ppl(model, domain_val[domain])
        composed_domain_ppls[domain] = round(ppl, 4)
    log(f"  Composed domain PPLs: {composed_domain_ppls}")

    # Composition ratio = composed_ppl / base_ppl
    # Lower is better (closer to 1 = less degradation from composition)
    # A ratio < 1 means composition IMPROVES over base (expected for domain data)
    composition_ratio = composed_ppl / base_ppl if base_ppl > 0 else float("inf")
    log(f"  Composition ratio: {composition_ratio:.4f}")

    # Per-domain composition ratio
    domain_ratios = {}
    for domain in DOMAIN_NAMES:
        if base_domain_ppls[domain] > 0:
            ratio = composed_domain_ppls[domain] / base_domain_ppls[domain]
            domain_ratios[domain] = round(ratio, 4)
    log(f"  Domain ratios: {domain_ratios}")

    # Depth attention weights with composed adapters (AttnRes only)
    composed_depth_weights = None
    if use_attnres:
        composed_depth_weights = get_depth_weights(model, val_data)
        log(f"  Composed depth weights: {composed_depth_weights}")

    cleanup(model, val_data)
    for v in domain_val.values():
        del v

    return {
        "base_ppl_mixed": round(base_ppl, 4),
        "base_domain_ppls": base_domain_ppls,
        "individual_ppls": individual_ppls,
        "composed_ppl_mixed": round(composed_ppl, 4),
        "composed_domain_ppls": composed_domain_ppls,
        "composition_ratio_mixed": round(composition_ratio, 4),
        "domain_ratios": domain_ratios,
        "composed_depth_weights": composed_depth_weights,
    }


# ============================================================================
# Phase 4: Analyze adapter per-layer contributions
# ============================================================================
def phase_adapter_layer_analysis(use_attnres, seed):
    """Measure per-layer adapter contribution norms."""
    tag = "attnres" if use_attnres else "standard"
    log(f"\n{'='*60}")
    log(f"  Adapter layer analysis ({tag}, seed={seed})")
    log(f"{'='*60}")

    results = {}
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/{tag}_seed{seed}/{domain}.npz"
        params = load_lora_params(adapter_path)

        # Compute per-layer norms of ΔW = B @ A
        layer_norms = []
        for layer_idx in range(N_LAYERS):
            layer_norm_sum = 0.0
            count = 0
            for k, v in params.items():
                if f"layers.{layer_idx}." in k and "lora_b" in k:
                    # Find corresponding lora_a
                    a_key = k.replace("lora_b", "lora_a")
                    if a_key in params:
                        # lora_a: (in, r), lora_b: (r, out)
                        # ΔW = A @ B: (in, out)
                        a = params[a_key]  # (in, r)
                        b = v  # lora_b: (r, out)
                        delta_w = a @ b
                        mx.eval(delta_w)
                        norm = mx.sqrt(mx.sum(delta_w * delta_w)).item()
                        layer_norm_sum += norm
                        count += 1
                        del delta_w
            layer_norms.append(round(layer_norm_sum / max(count, 1), 6))

        results[domain] = layer_norms
        log(f"  {domain} layer norms: {layer_norms}")
        del params

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("AttnRes Depth-Wise Attention for LoRA Composition")
    log("=" * 70)
    log_memory("start")

    all_results = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        seed_results = {}
        log(f"\n{'#'*70}")
        log(f"# SEED {seed}")
        log(f"{'#'*70}")

        for use_attnres in [False, True]:
            tag = "attnres" if use_attnres else "standard"
            log(f"\n>>> Architecture: {tag}")

            # Phase 1: Train base
            base_result = phase_train_base(use_attnres, seed)
            log_memory(f"after-base-{tag}")

            # Phase 2: Train adapters
            adapter_result = phase_train_adapters(use_attnres, seed)
            log_memory(f"after-adapters-{tag}")

            # Phase 3: Evaluate composition
            comp_result = phase_evaluate_composition(use_attnres, seed)
            log_memory(f"after-comp-{tag}")

            # Phase 4: Layer analysis
            layer_result = phase_adapter_layer_analysis(use_attnres, seed)
            log_memory(f"after-analysis-{tag}")

            seed_results[tag] = {
                "base": base_result,
                "adapters": adapter_result,
                "composition": comp_result,
                "layer_norms": layer_result,
            }

        all_results["seeds"][str(seed)] = seed_results

    # ========================================================================
    # Summary statistics across seeds
    # ========================================================================
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    std_ratios = []
    attn_ratios = []
    std_base_ppls = []
    attn_base_ppls = []

    for seed_str, seed_data in all_results["seeds"].items():
        std_ratio = seed_data["standard"]["composition"]["composition_ratio_mixed"]
        attn_ratio = seed_data["attnres"]["composition"]["composition_ratio_mixed"]
        std_ratios.append(std_ratio)
        attn_ratios.append(attn_ratio)

        std_base = seed_data["standard"]["base"]["base_ppl"]
        attn_base = seed_data["attnres"]["base"]["base_ppl"]
        std_base_ppls.append(std_base)
        attn_base_ppls.append(attn_base)

    mean_std_ratio = sum(std_ratios) / len(std_ratios)
    mean_attn_ratio = sum(attn_ratios) / len(attn_ratios)
    mean_std_base = sum(std_base_ppls) / len(std_base_ppls)
    mean_attn_base = sum(attn_base_ppls) / len(attn_base_ppls)

    # K1: AttnRes base quality
    base_quality_ratio = mean_attn_base / mean_std_base
    k1_pass = base_quality_ratio < 1.10
    log(f"\nK1 (base quality): AttnRes/Standard PPL ratio = {base_quality_ratio:.4f}")
    log(f"  Standard mean PPL: {mean_std_base:.4f}")
    log(f"  AttnRes mean PPL: {mean_attn_base:.4f}")
    log(f"  K1 {'PASS' if k1_pass else 'FAIL'} (threshold: <1.10)")

    # K2: Composition improvement
    # Lower ratio = better composition. AttnRes should have LOWER ratio.
    composition_improvement = (mean_std_ratio - mean_attn_ratio) / mean_std_ratio
    k2_pass = mean_attn_ratio < mean_std_ratio  # AttnRes ratio must be LOWER
    k2_strong = composition_improvement > 0.05  # >5% improvement
    log(f"\nK2 (composition improvement): improvement = {composition_improvement:.4f}")
    log(f"  Standard mean ratio: {mean_std_ratio:.4f}")
    log(f"  AttnRes mean ratio: {mean_attn_ratio:.4f}")
    log(f"  Improvement: {composition_improvement*100:.2f}%")
    log(f"  K2 {'PASS' if k2_pass else 'FAIL'} (AttnRes ratio < Standard)")
    log(f"  S1 {'PASS' if k2_strong else 'FAIL'} (>5% improvement)")

    # K3: Depth attention specialization
    # Check if depth weights are non-uniform across seeds
    all_depth_weights = []
    for seed_str, seed_data in all_results["seeds"].items():
        dw = seed_data["attnres"]["composition"].get("composed_depth_weights")
        if dw:
            all_depth_weights.append(dw)

    depth_entropy_ratios = []
    if all_depth_weights:
        for dw in all_depth_weights:
            for layer_idx, weights in enumerate(dw):
                n = len(weights)
                if n <= 1:
                    continue
                # Compute entropy
                entropy = -sum(w * math.log(w + 1e-10) for w in weights)
                max_entropy = math.log(n)
                entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
                depth_entropy_ratios.append(entropy_ratio)

    if depth_entropy_ratios:
        mean_entropy_ratio = sum(depth_entropy_ratios) / len(depth_entropy_ratios)
        k3_pass = mean_entropy_ratio < 0.95  # Non-uniform (entropy < 95% of max)
    else:
        mean_entropy_ratio = 1.0
        k3_pass = False

    log(f"\nK3 (depth weight specialization):")
    log(f"  Mean entropy ratio: {mean_entropy_ratio:.4f} (1.0 = perfectly uniform)")
    log(f"  K3 {'PASS' if k3_pass else 'FAIL'} (threshold: <0.95)")

    # Overall verdict
    all_pass = k1_pass and k2_pass and k3_pass
    log(f"\n{'='*70}")
    log(f"VERDICT: {'SUPPORTED' if all_pass else 'KILLED'}")
    log(f"  K1 (base quality): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (composition):  {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K3 (specialization): {'PASS' if k3_pass else 'FAIL'}")
    log(f"{'='*70}")

    # Also check per-seed consistency
    per_seed_k2 = []
    for s, a in zip(std_ratios, attn_ratios):
        per_seed_k2.append(a < s)
    log(f"\nPer-seed K2: {per_seed_k2} ({sum(per_seed_k2)}/{len(per_seed_k2)} seeds)")

    # Store depth weights analysis
    base_depth_weights = []
    composed_depth_weights = []
    for seed_str, seed_data in all_results["seeds"].items():
        bdw = seed_data["attnres"]["base"].get("base_depth_weights")
        cdw = seed_data["attnres"]["composition"].get("composed_depth_weights")
        if bdw:
            base_depth_weights.append(bdw)
        if cdw:
            composed_depth_weights.append(cdw)

    all_results["summary"] = {
        "k1_base_quality_ratio": round(base_quality_ratio, 4),
        "k1_pass": k1_pass,
        "k2_composition_improvement_pct": round(composition_improvement * 100, 2),
        "k2_pass": k2_pass,
        "k2_strong_pass": k2_strong,
        "k3_mean_entropy_ratio": round(mean_entropy_ratio, 4),
        "k3_pass": k3_pass,
        "mean_std_ratio": round(mean_std_ratio, 4),
        "mean_attn_ratio": round(mean_attn_ratio, 4),
        "mean_std_base_ppl": round(mean_std_base, 4),
        "mean_attn_base_ppl": round(mean_attn_base, 4),
        "std_ratios": [round(r, 4) for r in std_ratios],
        "attn_ratios": [round(r, 4) for r in attn_ratios],
        "per_seed_k2": per_seed_k2,
        "verdict": "supported" if all_pass else "killed",
        "base_depth_weights_per_seed": base_depth_weights,
        "composed_depth_weights_per_seed": composed_depth_weights,
    }

    total_time = time.time() - t_start
    all_results["total_time_s"] = round(total_time, 1)
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    log(f"Results saved to {RESULTS_FILE}")
    log_memory("end")


if __name__ == "__main__":
    main()
