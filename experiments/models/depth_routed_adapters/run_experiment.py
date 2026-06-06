#!/usr/bin/env python3
"""
Depth-Routed Adapters: Per-Layer Adapter Selection via AttnRes Pseudo-Queries.

Combines two orthogonal routing axes:
  1. Token-level: softmax router selects WHICH adapters (proven in softmax_router_scaling)
  2. Layer-level: learned depth weights select HOW MUCH each adapter contributes per layer

Kill criteria:
  K1 (#528): Depth routing weights uniform (no layer specialization) -> KILL
  K2 (#529): Layer-routed composition < 2% better than token-only routing -> KILL

Success criteria:
  S1 (#52): Token+layer routing beats token-only by >5% with clear layer specialization

Literature:
  - AttnRes (arXiv 2603.15031): pseudo-query mechanism for depth attention
  - MoLoRA (arXiv 2603.15965): per-token softmax routing
  - Our exp_attnres_depth_composition: AttnRes learns non-uniform weights (entropy 0.775)
  - Our exp_softmax_router_scaling: softmax router matches oracle at N=24

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
from mlx.utils import tree_flatten, tree_unflatten

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

# Depth routing
DEPTH_EMBED_DIM = 32  # expert embedding dimension (compact)
ROUTER_HIDDEN = 64    # softmax router hidden dim
ROUTER_TRAIN_STEPS = 300
ROUTER_LR = 3e-3


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
        for _ in range(n_samples):
            start = rng.randint(0, 25)
            seq = [chars[(start + j) % 26] for j in range(seq_len)]
            data.append(seq)
    elif domain == "numeric":
        chars = [ord(c) for c in "0123456789"]
        data = []
        for _ in range(n_samples):
            start = rng.randint(0, 9)
            seq = [chars[(start + j) % 10] for j in range(seq_len)]
            data.append(seq)
    elif domain == "mixed":
        data = []
        for _ in range(n_samples):
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
        for _ in range(n_samples):
            start = rng.randint(0, 25)
            seq = [chars[(start + j) % 26] for j in range(seq_len)]
            data.append(seq)
    elif domain == "symbol":
        symbols = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
        chars = [ord(c) for c in symbols]
        n_sym = len(symbols)
        data = []
        for _ in range(n_samples):
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
# Model components
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
        attn_out = self.attn(self.attn_norm(x), mask=mask)
        h = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(h))
        return h + ffn_out


class MicroTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.norm(h)
        return self.head(h)


# ============================================================================
# LoRA
# ============================================================================
class LoRALinear(nn.Module):
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
    count = 0
    for layer in model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(layer.attn, proj_name)
            setattr(layer.attn, proj_name, LoRALinear(base, rank, scale))
            count += 1
        for proj_name in ["gate", "up", "down"]:
            base = getattr(layer.ffn, proj_name)
            setattr(layer.ffn, proj_name, LoRALinear(base, rank, scale))
            count += 1
    return count


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora(model):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path), **params)


def load_lora_params(path):
    return dict(mx.load(str(path)))


def apply_lora_params(model, params):
    model.update(tree_unflatten(list(params.items())))


# ============================================================================
# Routing: Token-Level Softmax Router
# ============================================================================
class TokenRouter(nn.Module):
    """Softmax router: maps sequence embedding to adapter probabilities.

    Input: mean-pooled hidden states (B, d)
    Output: probabilities over N adapters (B, N)
    """

    def __init__(self, d_model, n_adapters, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, n_adapters, bias=False)

    def __call__(self, x):
        # x: (B, d) mean-pooled hidden
        h = nn.gelu(self.proj(x))
        return mx.softmax(self.out(h), axis=-1)  # (B, N)


# ============================================================================
# Routing: Layer-Level Depth Router
# ============================================================================
class DepthRouter(nn.Module):
    """Per-layer adapter weighting via learned pseudo-queries and expert embeddings.

    For each layer l, a pseudo-query w_l ∈ R^d_e and each adapter has an embedding
    r_i ∈ R^d_e. The depth weight for adapter i at layer l is:

        α_{i,l} = softmax_i(w_l^T · r_i / √d_e)

    This gives per-layer, per-adapter weights that allow different layers to
    emphasize different adapters.

    Parameters:
        pseudo_queries: (n_layers, d_embed) — one query per layer
        expert_embeds:  (n_adapters, d_embed) — one embedding per adapter
    """

    def __init__(self, n_layers, n_adapters, d_embed):
        super().__init__()
        self.n_layers = n_layers
        self.n_adapters = n_adapters
        self.d_embed = d_embed
        self.scale = math.sqrt(d_embed)

        # Zero-init pseudo-queries: starts as uniform weighting (softmax(0) = 1/N)
        self.pseudo_queries = mx.zeros((n_layers, d_embed))
        # Random-init expert embeddings (need to be distinguishable)
        self.expert_embeds = mx.random.normal(shape=(n_adapters, d_embed)) * 0.1

    def __call__(self):
        """Compute depth routing weights for all layers and adapters.

        Returns:
            alpha: (n_layers, n_adapters) — depth weight for each adapter at each layer
        """
        # (n_layers, d_embed) @ (d_embed, n_adapters) -> (n_layers, n_adapters)
        scores = (self.pseudo_queries @ self.expert_embeds.T) / self.scale
        alpha = mx.softmax(scores, axis=-1)  # softmax over adapters per layer
        return alpha

    def get_layer_weights(self):
        """Get depth weights as numpy-friendly list of lists."""
        alpha = self()
        mx.eval(alpha)
        return alpha.tolist()


# ============================================================================
# Composed inference with routing
# ============================================================================
def compose_with_token_routing(model, adapter_params_list, token_router, x, domain_val_data=None):
    """Run inference with token-level routing only (baseline).

    For each input, the router selects top-1 adapter. Each adapter is applied
    with full weight (no 1/N dilution — the router selects, not averages).
    """
    B, T = x.shape
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)

    # Get routing decisions from base model hidden states
    h = model.embed(x)
    for layer in model.layers:
        h = layer(h, mask=mask)
    h = model.norm(h)

    # Mean-pool for routing decision
    h_pool = mx.mean(h, axis=1)  # (B, d)
    probs = token_router(h_pool)  # (B, N)
    top1 = mx.argmax(probs, axis=-1)  # (B,)
    mx.eval(top1)
    top1_list = top1.tolist()

    # For each sample, apply the selected adapter and compute loss
    total_loss = 0.0
    total_tokens = 0
    for b in range(B):
        adapter_idx = top1_list[b]
        # Apply this adapter's params
        apply_lora_params(model, adapter_params_list[adapter_idx])
        mx.eval(model.parameters())

        single = x[b:b+1]  # (1, T)
        logits = model(single[:, :-1])
        targets = single[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.size
        del logits, loss

    return total_loss, total_tokens, top1_list


def compose_with_depth_routing(model, adapter_params_list, token_router, depth_router, x):
    """Run inference with token+layer routing (the novel approach).

    Token router selects top-1 adapter per input (WHICH adapter).
    Depth router modulates per-layer weights (HOW MUCH at each layer).

    Instead of applying adapter uniformly across all layers, each layer l gets
    a depth-modulated scale: α_{adapter_idx, l} for adapter_idx selected by token router.
    """
    B, T = x.shape
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)

    # Get routing decisions
    h = model.embed(x)
    for layer in model.layers:
        h = layer(h, mask=mask)
    h = model.norm(h)

    h_pool = mx.mean(h, axis=1)
    probs = token_router(h_pool)
    top1 = mx.argmax(probs, axis=-1)
    mx.eval(top1)
    top1_list = top1.tolist()

    # Get depth weights: (n_layers, n_adapters)
    depth_alpha = depth_router()
    mx.eval(depth_alpha)

    total_loss = 0.0
    total_tokens = 0
    for b in range(B):
        adapter_idx = top1_list[b]
        adapter_params = adapter_params_list[adapter_idx]

        # Apply adapter with per-layer depth modulation
        # For each LoRA param, scale by depth weight at that layer
        modulated_params = {}
        for k, v in adapter_params.items():
            # Extract layer index from param name like "layers.2.attn.q_proj.lora_b"
            parts = k.split(".")
            layer_idx = None
            for pi, part in enumerate(parts):
                if part == "layers" and pi + 1 < len(parts):
                    try:
                        layer_idx = int(parts[pi + 1])
                    except ValueError:
                        pass
                    break

            if layer_idx is not None and "lora_b" in k:
                # Scale B matrix by depth weight (scales the full ΔW = A @ B)
                # α_{adapter_idx, layer_idx} is the depth weight
                depth_weight = depth_alpha[layer_idx, adapter_idx]
                # Normalize: multiply by n_layers so mean depth weight = 1.0
                # This way uniform routing = standard routing
                modulated_params[k] = v * (depth_weight * N_LAYERS)
            else:
                modulated_params[k] = v

        apply_lora_params(model, modulated_params)
        mx.eval(model.parameters())

        single = x[b:b+1]
        logits = model(single[:, :-1])
        targets = single[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.size
        del logits, loss, modulated_params

    return total_loss, total_tokens, top1_list


def compute_ppl_routed(model, adapter_params_list, token_router, data,
                       depth_router=None, batch_size=16, max_batches=20):
    """Compute PPL with routing (token-only or token+depth)."""
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(len(data) // batch_size, max_batches)
    if n_batches == 0:
        n_batches = 1

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        if depth_router is not None:
            loss, tokens, _ = compose_with_depth_routing(
                model, adapter_params_list, token_router, depth_router, batch)
        else:
            loss, tokens, _ = compose_with_token_routing(
                model, adapter_params_list, token_router, batch)
        total_loss += loss
        total_tokens += tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ============================================================================
# Training utilities
# ============================================================================
def compute_loss(model, x):
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def compute_ppl(model, data, batch_size=16):
    total_loss = 0.0
    total_tokens = 0
    n_batches = len(data) // batch_size
    if n_batches == 0:
        n_batches = 1

    for i in range(min(n_batches, 20)):
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


def compute_oracle_ppl(model, adapter_params_list, domain_names, domain_val_data):
    """Oracle PPL: each domain evaluated with its own adapter (best possible)."""
    total_loss = 0.0
    total_tokens = 0
    for i, domain in enumerate(domain_names):
        apply_lora_params(model, adapter_params_list[i])
        mx.eval(model.parameters())
        data = domain_val_data[domain]
        logits = model(data[:, :-1])
        targets = data[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.size
        del logits, loss

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ============================================================================
# Phase 1: Train base model
# ============================================================================
def phase_train_base(seed):
    """Train a micro transformer base model."""
    mx.random.seed(seed)
    log(f"\n{'='*60}")
    log(f"  Phase 1: Train base model (seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Total params: {n_params:,}")

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
        if (step + 1) % 200 == 0:
            log(f"    Step {step+1}/{TRAIN_STEPS}: loss={loss.item():.4f}")
    gc.enable()
    gc.collect()

    base_ppl = compute_ppl(model, val_data)
    elapsed = time.time() - t0
    log(f"  Base PPL: {base_ppl:.4f}, Time: {elapsed:.1f}s")

    # Save model
    model_path = EXPERIMENT_DIR / f"models/base_seed{seed}.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    params = dict(tree_flatten(model.parameters()))
    mx.savez(str(model_path), **params)

    result = {"base_ppl": round(base_ppl, 4), "n_params": n_params, "train_time_s": round(elapsed, 1)}
    cleanup(model, optimizer, train_data, val_data)
    return result


# ============================================================================
# Phase 2: Train domain adapters
# ============================================================================
def phase_train_adapters(seed):
    """Train LoRA adapters for each domain."""
    mx.random.seed(seed + 500)
    log(f"\n{'='*60}")
    log(f"  Phase 2: Train adapters (seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM)
    model_path = EXPERIMENT_DIR / f"models/base_seed{seed}.npz"
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved

    model.freeze()
    n_lora = apply_lora(model, LORA_RANK, LORA_SCALE)
    log(f"  Applied LoRA to {n_lora} projections")

    adapter_results = {}
    for domain in DOMAIN_NAMES:
        log(f"  [{domain}] Training adapter...")
        t_d = time.time()
        zero_lora(model)

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
        gc.enable()
        gc.collect()

        individual_ppl = compute_ppl(model, val_data)
        lora_params = get_lora_params(model)
        adapter_path = EXPERIMENT_DIR / f"adapters/seed{seed}/{domain}.npz"
        save_lora_params(lora_params, adapter_path)

        adapter_results[domain] = {
            "individual_ppl": round(individual_ppl, 4),
            "train_time_s": round(time.time() - t_d, 1),
        }
        log(f"    {domain}: PPL={individual_ppl:.4f}")
        del train_data, val_data, optimizer, lora_params

    cleanup(model)
    log(f"  Total adapter training: {time.time()-t0:.1f}s")
    return adapter_results


# ============================================================================
# Phase 3: Train token-level router
# ============================================================================
def phase_train_router(seed):
    """Train softmax token-level router on domain classification."""
    mx.random.seed(seed + 2000)
    log(f"\n{'='*60}")
    log(f"  Phase 3: Train token router (seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    # Load base model for embedding extraction
    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM)
    model_path = EXPERIMENT_DIR / f"models/base_seed{seed}.npz"
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved
    model.freeze()

    # Generate labeled training data
    all_embeddings = []
    all_labels = []
    for di, domain in enumerate(DOMAIN_NAMES):
        data = generate_domain_data(domain, 100, MAX_SEQ_LEN, seed=seed + di * 100)
        mx.eval(data)
        # Get hidden states
        mask = nn.MultiHeadAttention.create_additive_causal_mask(MAX_SEQ_LEN).astype(mx.bfloat16)
        h = model.embed(data)
        for layer in model.layers:
            h = layer(h, mask=mask)
        h = model.norm(h)
        h_pool = mx.mean(h, axis=1)  # (n_samples, d)
        mx.eval(h_pool)
        all_embeddings.append(h_pool)
        all_labels.append(mx.full((len(data),), di, dtype=mx.int32))
        del data, h

    embeddings = mx.concatenate(all_embeddings, axis=0)  # (500, d)
    labels = mx.concatenate(all_labels, axis=0)  # (500,)
    mx.eval(embeddings, labels)

    # Train router
    router = TokenRouter(D_MODEL, N_DOMAINS, ROUTER_HIDDEN)
    mx.eval(router.parameters())

    def router_loss(router, emb, lab):
        logits = router.proj(emb)
        logits = nn.gelu(logits)
        logits = router.out(logits)  # (B, N) raw logits
        return nn.losses.cross_entropy(logits, lab, reduction="mean")

    router_opt = opt.Adam(learning_rate=ROUTER_LR)
    loss_and_grad = nn.value_and_grad(router, router_loss)

    gc.disable()
    for step in range(ROUTER_TRAIN_STEPS):
        # Random batch
        idx = mx.random.randint(0, len(embeddings), shape=(BATCH_SIZE,))
        batch_emb = embeddings[idx]
        batch_lab = labels[idx]
        loss, grads = loss_and_grad(router, batch_emb, batch_lab)
        router_opt.update(router, grads)
        mx.eval(router.parameters(), router_opt.state, loss)
        if (step + 1) % 100 == 0:
            log(f"    Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={loss.item():.4f}")
    gc.enable()
    gc.collect()

    # Evaluate router accuracy
    probs = router(embeddings)
    preds = mx.argmax(probs, axis=-1)
    mx.eval(preds)
    accuracy = mx.mean(preds == labels).item()
    log(f"  Router accuracy: {accuracy:.3f}")

    # Save router
    router_path = EXPERIMENT_DIR / f"models/router_seed{seed}.npz"
    router_params = dict(tree_flatten(router.parameters()))
    mx.savez(str(router_path), **router_params)

    result = {"accuracy": round(accuracy, 4), "train_time_s": round(time.time() - t0, 1)}
    cleanup(model, embeddings, labels, router_opt)
    # Don't cleanup router — return it for later use
    return result, router


# ============================================================================
# Phase 4: Train depth router
# ============================================================================
def phase_train_depth_router(seed, token_router):
    """Train depth router via PPL minimization on validation data.

    The depth router learns per-layer weights that modulate adapter contribution.
    We optimize depth router params to minimize routed PPL on held-out data.
    """
    mx.random.seed(seed + 3000)
    log(f"\n{'='*60}")
    log(f"  Phase 4: Train depth router (seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    # Load base model with LoRA applied
    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM)
    model_path = EXPERIMENT_DIR / f"models/base_seed{seed}.npz"
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved
    model.freeze()
    apply_lora(model, LORA_RANK, LORA_SCALE)

    # Load all adapter params
    adapter_params_list = []
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/seed{seed}/{domain}.npz"
        adapter_params_list.append(load_lora_params(adapter_path))

    # Generate training data (mixed domains)
    train_data = generate_base_data(200, MAX_SEQ_LEN, seed=seed + 4000)
    mx.eval(train_data)

    # Initialize depth router
    depth_router = DepthRouter(N_LAYERS, N_DOMAINS, DEPTH_EMBED_DIM)
    mx.eval(depth_router.pseudo_queries, depth_router.expert_embeds)

    # Simple gradient-free optimization: perturb and evaluate
    # (Gradient-based would require differentiating through the per-sample routing loop,
    # which is awkward with the current per-sample adapter application)
    best_depth_router_params = {
        "pseudo_queries": mx.array(depth_router.pseudo_queries),
        "expert_embeds": mx.array(depth_router.expert_embeds),
    }
    mx.eval(best_depth_router_params)

    # Evaluate baseline (uniform depth = token-only routing)
    baseline_ppl = compute_ppl_routed(
        model, adapter_params_list, token_router, train_data, depth_router=None, max_batches=5)
    log(f"  Baseline PPL (token-only): {baseline_ppl:.4f}")

    best_ppl = compute_ppl_routed(
        model, adapter_params_list, token_router, train_data, depth_router=depth_router, max_batches=5)
    log(f"  Initial depth-routed PPL: {best_ppl:.4f}")

    # CMA-ES-like search: perturb pseudo-queries and expert embeddings
    n_iterations = 40
    sigma = 0.3
    for iteration in range(n_iterations):
        # Perturb
        pq_noise = mx.random.normal(shape=depth_router.pseudo_queries.shape) * sigma
        ee_noise = mx.random.normal(shape=depth_router.expert_embeds.shape) * sigma

        # Try perturbation
        depth_router.pseudo_queries = best_depth_router_params["pseudo_queries"] + pq_noise
        depth_router.expert_embeds = best_depth_router_params["expert_embeds"] + ee_noise
        mx.eval(depth_router.pseudo_queries, depth_router.expert_embeds)

        ppl = compute_ppl_routed(
            model, adapter_params_list, token_router, train_data,
            depth_router=depth_router, max_batches=3)

        if ppl < best_ppl:
            best_ppl = ppl
            best_depth_router_params = {
                "pseudo_queries": mx.array(depth_router.pseudo_queries),
                "expert_embeds": mx.array(depth_router.expert_embeds),
            }
            mx.eval(best_depth_router_params)
            log(f"    Iter {iteration+1}: new best PPL={best_ppl:.4f}")

        # Decay sigma
        if (iteration + 1) % 15 == 0:
            sigma *= 0.7

    # Apply best params
    depth_router.pseudo_queries = best_depth_router_params["pseudo_queries"]
    depth_router.expert_embeds = best_depth_router_params["expert_embeds"]
    mx.eval(depth_router.pseudo_queries, depth_router.expert_embeds)

    # Get final depth weights
    depth_weights = depth_router.get_layer_weights()
    log(f"  Final depth weights:")
    for l in range(N_LAYERS):
        log(f"    Layer {l}: {[f'{w:.3f}' for w in depth_weights[l]]}")

    # Measure specialization
    entropy_ratios = []
    for l in range(N_LAYERS):
        weights = depth_weights[l]
        entropy = -sum(w * math.log(w + 1e-10) for w in weights)
        max_ent = math.log(N_DOMAINS)
        entropy_ratios.append(entropy / max_ent if max_ent > 0 else 1.0)
    mean_entropy = sum(entropy_ratios) / len(entropy_ratios)
    log(f"  Mean entropy ratio: {mean_entropy:.4f} (1.0=uniform)")

    # Save depth router
    dr_path = EXPERIMENT_DIR / f"models/depth_router_seed{seed}.npz"
    mx.savez(str(dr_path), **best_depth_router_params)

    result = {
        "baseline_ppl": round(baseline_ppl, 4),
        "best_ppl": round(best_ppl, 4),
        "improvement_pct": round((baseline_ppl - best_ppl) / baseline_ppl * 100, 2),
        "depth_weights": depth_weights,
        "entropy_ratios": [round(e, 4) for e in entropy_ratios],
        "mean_entropy_ratio": round(mean_entropy, 4),
        "train_time_s": round(time.time() - t0, 1),
    }

    cleanup(model, train_data)
    for p in adapter_params_list:
        del p

    return result, depth_router


# ============================================================================
# Phase 5: Evaluate all routing modes
# ============================================================================
def phase_evaluate(seed, token_router, depth_router):
    """Compare: base, oracle, token-only, token+depth, random routing."""
    mx.random.seed(seed + 5000)
    log(f"\n{'='*60}")
    log(f"  Phase 5: Evaluate routing modes (seed={seed})")
    log(f"{'='*60}")
    t0 = time.time()

    # Load model
    model = MicroTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM)
    model_path = EXPERIMENT_DIR / f"models/base_seed{seed}.npz"
    saved = dict(mx.load(str(model_path)))
    model.update(tree_unflatten(list(saved.items())))
    mx.eval(model.parameters())
    del saved
    model.freeze()
    apply_lora(model, LORA_RANK, LORA_SCALE)

    # Load adapters
    adapter_params_list = []
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/seed{seed}/{domain}.npz"
        adapter_params_list.append(load_lora_params(adapter_path))

    # Validation data: mixed and per-domain
    val_mixed = generate_base_data(200, MAX_SEQ_LEN, seed=seed + 6000)
    mx.eval(val_mixed)

    domain_val = {}
    for domain in DOMAIN_NAMES:
        dv = generate_domain_data(domain, 50, MAX_SEQ_LEN, seed=seed + hash(domain) % 10000 + 8000)
        mx.eval(dv)
        domain_val[domain] = dv

    # 1. Base PPL (no adapter)
    zero_lora(model)
    base_ppl = compute_ppl(model, val_mixed)
    log(f"  Base PPL: {base_ppl:.4f}")

    # 2. Oracle PPL (each domain gets its own adapter)
    oracle_ppl = compute_oracle_ppl(model, adapter_params_list, DOMAIN_NAMES, domain_val)
    log(f"  Oracle PPL: {oracle_ppl:.4f}")

    # 3. Token-only routing PPL
    token_ppl = compute_ppl_routed(
        model, adapter_params_list, token_router, val_mixed, depth_router=None)
    log(f"  Token-only PPL: {token_ppl:.4f}")

    # 4. Token+depth routing PPL
    depth_ppl = compute_ppl_routed(
        model, adapter_params_list, token_router, val_mixed, depth_router=depth_router)
    log(f"  Token+depth PPL: {depth_ppl:.4f}")

    # 5. Random routing (baseline)
    class RandomRouter(nn.Module):
        def __init__(self, n_adapters):
            super().__init__()
            self.n = n_adapters
        def __call__(self, x):
            B = x.shape[0]
            return mx.softmax(mx.random.normal(shape=(B, self.n)), axis=-1)

    random_router = RandomRouter(N_DOMAINS)
    random_ppl = compute_ppl_routed(
        model, adapter_params_list, random_router, val_mixed, depth_router=None)
    log(f"  Random routing PPL: {random_ppl:.4f}")

    # 6. 1/N uniform composition PPL
    avg_params = {}
    for params in adapter_params_list:
        for k, v in params.items():
            if k not in avg_params:
                avg_params[k] = v / N_DOMAINS
            else:
                avg_params[k] = avg_params[k] + v / N_DOMAINS
    apply_lora_params(model, avg_params)
    mx.eval(model.parameters())
    uniform_ppl = compute_ppl(model, val_mixed)
    log(f"  Uniform 1/N PPL: {uniform_ppl:.4f}")
    del avg_params

    # Per-domain analysis for token+depth routing
    domain_depth_ppls = {}
    for di, domain in enumerate(DOMAIN_NAMES):
        # Apply oracle adapter
        apply_lora_params(model, adapter_params_list[di])
        mx.eval(model.parameters())
        oracle_d_ppl = compute_ppl(model, domain_val[domain])

        # Token-only on this domain
        token_d_ppl = compute_ppl_routed(
            model, adapter_params_list, token_router,
            domain_val[domain], depth_router=None, max_batches=5)

        # Token+depth on this domain
        depth_d_ppl = compute_ppl_routed(
            model, adapter_params_list, token_router,
            domain_val[domain], depth_router=depth_router, max_batches=5)

        domain_depth_ppls[domain] = {
            "oracle": round(oracle_d_ppl, 4),
            "token_only": round(token_d_ppl, 4),
            "token_depth": round(depth_d_ppl, 4),
        }
        log(f"  {domain}: oracle={oracle_d_ppl:.4f} token={token_d_ppl:.4f} depth={depth_d_ppl:.4f}")

    # Compute gamma (geometric mean of domain PPLs)
    def gamma(ppls):
        log_sum = sum(math.log(p) for p in ppls if p > 0)
        return math.exp(log_sum / len(ppls)) if ppls else float("inf")

    gamma_oracle = gamma([domain_depth_ppls[d]["oracle"] for d in DOMAIN_NAMES])
    gamma_token = gamma([domain_depth_ppls[d]["token_only"] for d in DOMAIN_NAMES])
    gamma_depth = gamma([domain_depth_ppls[d]["token_depth"] for d in DOMAIN_NAMES])
    log(f"\n  Gamma (geom mean PPL):")
    log(f"    Oracle: {gamma_oracle:.4f}")
    log(f"    Token-only: {gamma_token:.4f}")
    log(f"    Token+depth: {gamma_depth:.4f}")

    elapsed = time.time() - t0
    result = {
        "base_ppl": round(base_ppl, 4),
        "oracle_ppl": round(oracle_ppl, 4),
        "token_only_ppl": round(token_ppl, 4),
        "token_depth_ppl": round(depth_ppl, 4),
        "random_ppl": round(random_ppl, 4),
        "uniform_1n_ppl": round(uniform_ppl, 4),
        "gamma_oracle": round(gamma_oracle, 4),
        "gamma_token_only": round(gamma_token, 4),
        "gamma_token_depth": round(gamma_depth, 4),
        "domain_ppls": domain_depth_ppls,
        "eval_time_s": round(elapsed, 1),
    }

    cleanup(model, val_mixed)
    for v in domain_val.values():
        del v
    for p in adapter_params_list:
        del p

    return result


# ============================================================================
# Phase 6: Adapter per-layer norms (for understanding depth routing basis)
# ============================================================================
def phase_layer_norms(seed):
    """Measure per-layer ΔW norms to understand what depth routing can exploit."""
    log(f"\n{'='*60}")
    log(f"  Phase 6: Adapter layer norms (seed={seed})")
    log(f"{'='*60}")

    results = {}
    for domain in DOMAIN_NAMES:
        adapter_path = EXPERIMENT_DIR / f"adapters/seed{seed}/{domain}.npz"
        params = load_lora_params(adapter_path)

        layer_norms = []
        for layer_idx in range(N_LAYERS):
            norm_sum = 0.0
            count = 0
            for k, v in params.items():
                if f"layers.{layer_idx}." in k and "lora_b" in k:
                    a_key = k.replace("lora_b", "lora_a")
                    if a_key in params:
                        delta_w = params[a_key] @ v
                        mx.eval(delta_w)
                        norm = mx.sqrt(mx.sum(delta_w * delta_w)).item()
                        norm_sum += norm
                        count += 1
                        del delta_w
            layer_norms.append(round(norm_sum / max(count, 1), 6))

        results[domain] = layer_norms
        log(f"  {domain}: {layer_norms}")
        del params

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("Depth-Routed Adapters: Token + Layer Routing")
    log("=" * 70)
    log_memory("start")

    all_results = {"seeds": {}, "summary": {}}

    for seed in SEEDS:
        log(f"\n{'#'*70}")
        log(f"# SEED {seed}")
        log(f"{'#'*70}")

        seed_result = {}

        # Phase 1: Train base
        seed_result["base"] = phase_train_base(seed)
        log_memory("after-base")

        # Phase 2: Train adapters
        seed_result["adapters"] = phase_train_adapters(seed)
        log_memory("after-adapters")

        # Phase 3: Train token router
        router_result, token_router = phase_train_router(seed)
        seed_result["token_router"] = router_result
        log_memory("after-token-router")

        # Phase 4: Train depth router
        depth_result, depth_router = phase_train_depth_router(seed, token_router)
        seed_result["depth_router"] = depth_result
        log_memory("after-depth-router")

        # Phase 5: Evaluate all modes
        seed_result["eval"] = phase_evaluate(seed, token_router, depth_router)
        log_memory("after-eval")

        # Phase 6: Layer norms
        seed_result["layer_norms"] = phase_layer_norms(seed)
        log_memory("after-layer-norms")

        all_results["seeds"][str(seed)] = seed_result
        cleanup(token_router, depth_router)

    # ========================================================================
    # Summary
    # ========================================================================
    log(f"\n{'='*70}")
    log("SUMMARY ACROSS SEEDS")
    log(f"{'='*70}")

    gamma_oracles = []
    gamma_tokens = []
    gamma_depths = []
    entropy_ratios_all = []

    for seed_str, sd in all_results["seeds"].items():
        gamma_oracles.append(sd["eval"]["gamma_oracle"])
        gamma_tokens.append(sd["eval"]["gamma_token_only"])
        gamma_depths.append(sd["eval"]["gamma_token_depth"])
        entropy_ratios_all.append(sd["depth_router"]["mean_entropy_ratio"])

    mean_gamma_oracle = sum(gamma_oracles) / len(gamma_oracles)
    mean_gamma_token = sum(gamma_tokens) / len(gamma_tokens)
    mean_gamma_depth = sum(gamma_depths) / len(gamma_depths)
    mean_entropy = sum(entropy_ratios_all) / len(entropy_ratios_all)

    # K1: Depth weight specialization
    k1_pass = mean_entropy < 0.95
    log(f"\nK1 (depth specialization):")
    log(f"  Mean entropy ratio: {mean_entropy:.4f} (1.0=uniform, <0.95=specialized)")
    log(f"  Per-seed: {[round(e, 4) for e in entropy_ratios_all]}")
    log(f"  K1: {'PASS' if k1_pass else 'FAIL'}")

    # K2: Layer-routed > token-only by ≥2%
    depth_improvement = (mean_gamma_token - mean_gamma_depth) / mean_gamma_token * 100
    k2_pass = depth_improvement >= 2.0
    log(f"\nK2 (depth routing improves over token-only):")
    log(f"  Mean gamma token-only: {mean_gamma_token:.4f}")
    log(f"  Mean gamma token+depth: {mean_gamma_depth:.4f}")
    log(f"  Improvement: {depth_improvement:.2f}%")
    log(f"  K2: {'PASS' if k2_pass else 'FAIL'} (threshold: ≥2%)")

    # Per-seed consistency
    per_seed_k2 = []
    for gt, gd in zip(gamma_tokens, gamma_depths):
        per_seed_k2.append(gd < gt)
    log(f"  Per-seed K2: {per_seed_k2} ({sum(per_seed_k2)}/{len(per_seed_k2)})")

    # S1: >5% improvement with clear specialization
    s1_pass = depth_improvement >= 5.0 and k1_pass
    log(f"\nS1 (>5% improvement + layer specialization):")
    log(f"  S1: {'PASS' if s1_pass else 'FAIL'}")

    # Oracle gap
    oracle_gap_token = (mean_gamma_token - mean_gamma_oracle) / mean_gamma_oracle * 100
    oracle_gap_depth = (mean_gamma_depth - mean_gamma_oracle) / mean_gamma_oracle * 100
    log(f"\nOracle gaps:")
    log(f"  Token-only vs oracle: +{oracle_gap_token:.2f}%")
    log(f"  Token+depth vs oracle: +{oracle_gap_depth:.2f}%")
    log(f"  Mean oracle gamma: {mean_gamma_oracle:.4f}")

    # Verdict
    if not k1_pass and not k2_pass:
        verdict = "killed"
    elif k1_pass and k2_pass:
        verdict = "supported"
    else:
        verdict = "killed"  # Both K criteria must pass

    log(f"\n{'='*70}")
    log(f"VERDICT: {verdict.upper()}")
    log(f"  K1 (specialization): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (≥2% improvement): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  S1 (≥5% + special.): {'PASS' if s1_pass else 'FAIL'}")
    log(f"{'='*70}")

    all_results["summary"] = {
        "k1_mean_entropy_ratio": round(mean_entropy, 4),
        "k1_per_seed": [round(e, 4) for e in entropy_ratios_all],
        "k1_pass": k1_pass,
        "k2_depth_improvement_pct": round(depth_improvement, 2),
        "k2_pass": k2_pass,
        "k2_per_seed": per_seed_k2,
        "s1_pass": s1_pass,
        "mean_gamma_oracle": round(mean_gamma_oracle, 4),
        "mean_gamma_token_only": round(mean_gamma_token, 4),
        "mean_gamma_token_depth": round(mean_gamma_depth, 4),
        "oracle_gap_token_pct": round(oracle_gap_token, 2),
        "oracle_gap_depth_pct": round(oracle_gap_depth, 2),
        "gamma_oracles": [round(g, 4) for g in gamma_oracles],
        "gamma_tokens": [round(g, 4) for g in gamma_tokens],
        "gamma_depths": [round(g, 4) for g in gamma_depths],
        "verdict": verdict,
    }

    total_time = time.time() - t_start
    all_results["total_time_s"] = round(total_time, 1)
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    log(f"Results saved to {RESULTS_FILE}")
    log_memory("end")


if __name__ == "__main__":
    main()
