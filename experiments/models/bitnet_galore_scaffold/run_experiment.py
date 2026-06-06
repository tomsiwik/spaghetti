#!/usr/bin/env python3
"""
GaLore-Grown Ternary Scaffold for Adapter Composition

Tests whether a model trained from random init using GaLore (gradient low-rank
projection) can serve as a scaffold for LoRA adapter composition, matching
a conventionally-trained baseline.

Key insight: Prior base-free experiments (exp_bitnet_basefree_exploration) failed
because adapters trained on pretrained base don't transfer to random scaffold.
This experiment is DIFFERENT: we grow a scaffold via GaLore from scratch, then
train adapters ON that scaffold. GaLore produces genuine language model weights.

Design:
  - Tiny GPT model (~5M params, d=256, 6 layers, 4 heads) for fast iteration
  - Two training paths from SAME random init:
    (A) Standard full-rank Adam training = "baseline"
    (B) GaLore training (low-rank gradient projection, rank-64) = "scaffold"
  - Same data, same steps, same hyperparams (except GaLore projection)
  - After pretraining: optionally quantize to ternary via STE
  - Train 5 domain LoRA adapters on EACH pretrained model
  - Compare: base PPL, adapter convergence, composition quality, cosines

Kill criteria:
  K1: GaLore scaffold PPL > 2x baseline PPL after equivalent compute budget
  K2: Adapters on GaLore scaffold compose worse (composition ratio > 2x baseline)

Platform: Apple Silicon MLX, $0.
"""

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

# ===========================================================================
# Configuration
# ===========================================================================
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
VOCAB_SIZE = 8192  # Character-level or small BPE
MAX_SEQ_LEN = 128
DROPOUT = 0.0

# Training
PRETRAIN_STEPS = 2000
PRETRAIN_LR = 3e-4
PRETRAIN_BATCH_SIZE = 4  # sequences per batch

# GaLore
GALORE_RANK = 64  # rank of gradient projection (d/4)
GALORE_UPDATE_FREQ = 200  # re-compute SVD every T steps
GALORE_SCALE = 1.0

# LoRA adapter training
LORA_RANK = 16
LORA_SCALE = 4.0
ADAPTER_TRAIN_STEPS = 400
ADAPTER_LR = 1e-3
ADAPTER_BATCH_SIZE = 1

# Eval
VAL_BATCHES = 25

DEFAULT_SEEDS = [42, 123, 456]
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition
DATA_ROOT = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"
DOMAINS = ["python", "math", "medical", "legal", "creative"]


# ===========================================================================
# Tiny GPT Model
# ===========================================================================
class TinyMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 4, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 4, bias=False)
        self.down_proj = nn.Linear(d_model * 4, d_model, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.o_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = TinyMLP(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.blocks = [TinyBlock(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(mx.arange(T))
        h = tok + pos

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(h.dtype)

        for block in self.blocks:
            h = block(h, mask=mask)

        h = self.ln_f(h)
        return self.lm_head(h)


# ===========================================================================
# GaLore Optimizer Wrapper
# ===========================================================================
class GaLoreProjection:
    """Maintains SVD-based gradient projections for a weight matrix."""

    def __init__(self, shape, rank, update_freq):
        self.rank = min(rank, min(shape))
        self.update_freq = update_freq
        self.step = 0
        # Initialize projection matrices as None -- computed on first gradient
        self.P = None  # left projection (m, rank)

    def project(self, grad):
        """Project gradient to low-rank space."""
        if self.step % self.update_freq == 0 or self.P is None:
            self._update_projection(grad)
        self.step += 1

        # Project: grad_proj = P^T @ grad  (rank, n)
        grad_proj = self.P.T @ grad
        return grad_proj

    def unproject(self, grad_proj):
        """Reconstruct full-rank update from low-rank projection."""
        # Unproject: delta = P @ grad_proj  (m, n)
        return self.P @ grad_proj

    def _update_projection(self, grad):
        """Compute SVD of gradient and update projection matrix P."""
        # Use SVD to get top-rank left singular vectors
        # For m x n gradient, P is m x rank
        U, S, Vt = mx.linalg.svd(grad, stream=mx.cpu)
        mx.eval(U, S, Vt)
        self.P = U[:, :self.rank]
        mx.eval(self.P)


class GaLoreAdamState:
    """Adam optimizer state in the projected (low-rank) space."""

    def __init__(self, shape_proj, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = mx.zeros(shape_proj)  # first moment (rank, n)
        self.v = mx.zeros(shape_proj)  # second moment (rank, n)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, grad_proj, lr):
        """Adam update in projected space, return projected update."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_proj
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_proj ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        update_proj = -lr * m_hat / (mx.sqrt(v_hat) + self.eps)
        mx.eval(self.m, self.v)
        return update_proj


class GaLoreOptimizer:
    """GaLore optimizer: applies low-rank gradient projection to large weight matrices.

    For small parameters (embeddings, norms), uses standard Adam.
    For large weight matrices (attention/MLP projections), uses GaLore projection.
    """

    def __init__(self, model, lr, galore_rank, galore_update_freq,
                 min_dim_for_galore=128):
        self.lr = lr
        self.galore_rank = galore_rank
        self.galore_update_freq = galore_update_freq
        self.min_dim = min_dim_for_galore

        # Separate parameters into GaLore and standard
        self.galore_projections = {}  # name -> GaLoreProjection
        self.galore_states = {}      # name -> GaLoreAdamState
        self.standard_adam = opt.Adam(learning_rate=lr)

        self.galore_param_names = set()
        for name, p in tree_flatten(model.parameters()):
            if len(p.shape) == 2 and min(p.shape) >= self.min_dim:
                self.galore_param_names.add(name)
                proj = GaLoreProjection(p.shape, galore_rank, galore_update_freq)
                self.galore_projections[name] = proj
                # State will be initialized on first gradient (need projected shape)

    def update(self, model, grads):
        """Update model parameters using GaLore for large matrices, Adam for rest."""
        flat_grads = tree_flatten(grads)

        galore_updates = {}
        standard_grads = {}

        for name, g in flat_grads:
            if name in self.galore_param_names:
                proj = self.galore_projections[name]
                # Project gradient
                g_proj = proj.project(g)

                # Initialize Adam state if needed
                if name not in self.galore_states:
                    self.galore_states[name] = GaLoreAdamState(g_proj.shape)

                # Adam step in projected space
                update_proj = self.galore_states[name].step(g_proj, self.lr)

                # Unproject to full-rank
                update_full = proj.unproject(update_proj)
                galore_updates[name] = update_full
            else:
                standard_grads[name] = g

        # Apply GaLore updates directly
        if galore_updates:
            current_params = dict(tree_flatten(model.parameters()))
            updated = {}
            for name, delta in galore_updates.items():
                updated[name] = current_params[name] + delta
            model.update(tree_unflatten(list(updated.items())))

        # Apply standard Adam updates
        if standard_grads:
            standard_grads_tree = tree_unflatten(list(standard_grads.items()))
            self.standard_adam.update(model, standard_grads_tree)

        mx.eval(model.parameters())
        if hasattr(self.standard_adam, 'state'):
            mx.eval(self.standard_adam.state)


# ===========================================================================
# LoRA for TinyGPT
# ===========================================================================
class LoRALinear(nn.Module):
    """Simple LoRA wrapper around nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        in_dim = base.weight.shape[1]
        out_dim = base.weight.shape[0]
        self.base = base
        self.lora_a = mx.random.normal(shape=(in_dim, rank)) * (1.0 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((rank, out_dim))
        self.scale = scale

    def __call__(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora(model, rank, scale):
    """Apply LoRA to all attention and MLP projections."""
    count = 0
    for block in model.blocks:
        for attr_path in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            base_linear = getattr(parent, parts[-1])
            if isinstance(base_linear, nn.Linear):
                lora = LoRALinear(base_linear, rank, scale)
                setattr(parent, parts[-1], lora)
                count += 1
    print(f"  Applied LoRA (r={rank}, scale={scale}) to {count} layers")
    return model


def get_lora_params(model):
    """Extract LoRA parameters."""
    params = {}
    for name, p in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora(model):
    """Reset LoRA params."""
    for block in model.blocks:
        for attr_path in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                in_dim = module.lora_a.shape[0]
                module.lora_a = mx.random.normal(shape=module.lora_a.shape) * (1.0 / math.sqrt(in_dim))
                module.lora_b = mx.zeros(module.lora_b.shape)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    """Load adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def remove_lora(model):
    """Remove LoRA, keep base weights."""
    for block in model.blocks:
        for attr_path in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                setattr(parent, parts[-1], module.base)


# ===========================================================================
# Ternary Quantization
# ===========================================================================
def ternary_quantize(model):
    """Quantize all weight matrices to ternary {-1, 0, 1} with per-tensor scale.

    Uses absmean quantization: threshold = mean(|W|), scale = mean(|W|>threshold).
    """
    count = 0
    for name, p in tree_flatten(model.parameters()):
        if len(p.shape) == 2 and p.size > 100:  # only large weight matrices
            abs_w = mx.abs(p)
            threshold = mx.mean(abs_w)
            # Ternary: -1 where w < -threshold, +1 where w > threshold, 0 otherwise
            ternary = mx.where(p > threshold, 1.0, mx.where(p < -threshold, -1.0, 0.0))
            # Scale: mean absolute value of non-zero entries
            nonzero_mask = ternary != 0
            n_nonzero = mx.sum(nonzero_mask)
            if n_nonzero.item() > 0:
                scale = mx.sum(abs_w * nonzero_mask) / n_nonzero
            else:
                scale = mx.array(1.0)
            quantized = ternary * scale
            count += 1
            # Set directly
            model.update(tree_unflatten([(name, quantized)]))
    mx.eval(model.parameters())
    print(f"  Quantized {count} weight matrices to ternary")


# ===========================================================================
# Data Loading (reuse from bitnet_2b_real_composition)
# ===========================================================================
def load_domain_data(domain_name):
    """Load train and val data from existing domain directories."""
    data_dir = DATA_ROOT / domain_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_texts = []
    val_texts = []

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "valid.jsonl"

    with open(train_path) as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    with open(val_path) as f:
        for line in f:
            val_texts.append(json.loads(line)["text"])

    return train_texts, val_texts


def build_tokenizer(texts, vocab_size=VOCAB_SIZE):
    """Build a simple character-level tokenizer from training data.

    Returns encode/decode functions and the vocabulary.
    """
    # Collect all characters
    all_chars = set()
    for t in texts:
        all_chars.update(t)
    chars = sorted(all_chars)

    # Build char-to-id mapping (reserve 0 for padding, 1 for UNK)
    char_to_id = {c: i + 2 for i, c in enumerate(chars)}
    id_to_char = {i + 2: c for i, c in enumerate(chars)}
    actual_vocab = len(chars) + 2

    def encode(text):
        return [char_to_id.get(c, 1) for c in text]

    def decode(ids):
        return "".join(id_to_char.get(i, "?") for i in ids)

    return encode, decode, min(actual_vocab, vocab_size)


def tokenize_texts(texts, encode_fn, max_len=MAX_SEQ_LEN):
    """Tokenize texts into fixed-length sequences."""
    all_tokens = []
    for text in texts:
        tokens = encode_fn(text)
        if len(tokens) > max_len + 1:
            tokens = tokens[:max_len + 1]
        if len(tokens) >= 4:
            all_tokens.append(mx.array(tokens))
    return all_tokens


# ===========================================================================
# Pretraining
# ===========================================================================
def pretrain_standard(model, train_tokens, val_tokens, steps, lr, batch_size, label="standard"):
    """Standard full-rank Adam pretraining."""
    optimizer = opt.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t_start = time.time()
    for step in range(steps):
        # Sample batch
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            idx = (step * batch_size + _) % len(train_tokens)
            tokens = train_tokens[idx]
            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])

        # Pad to same length
        max_len = max(len(b) for b in batch_x)
        x = mx.zeros((batch_size, max_len), dtype=mx.int32)
        y = mx.zeros((batch_size, max_len), dtype=mx.int32)
        for i, (bx, by) in enumerate(zip(batch_x, batch_y)):
            x[i, :len(bx)] = bx
            y[i, :len(by)] = by

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())
        if (step + 1) % 200 == 0 or step == 0:
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"  [{label}] Step {step+1}/{steps}: loss={loss.item():.4f} (avg200={avg:.4f})")

    train_time = time.time() - t_start
    final_avg = sum(losses[-100:]) / len(losses[-100:])
    print(f"  [{label}] Done in {train_time:.1f}s. Final avg loss: {final_avg:.4f}")
    return losses, train_time


def pretrain_galore(model, train_tokens, val_tokens, steps, lr, batch_size,
                    galore_rank, galore_update_freq, label="galore"):
    """GaLore pretraining with gradient low-rank projection."""
    galore_opt = GaLoreOptimizer(
        model, lr=lr,
        galore_rank=galore_rank,
        galore_update_freq=galore_update_freq
    )
    print(f"  [{label}] GaLore params: {len(galore_opt.galore_param_names)} matrices, "
          f"rank={galore_rank}, update_freq={galore_update_freq}")

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t_start = time.time()
    for step in range(steps):
        # Sample batch
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            idx = (step * batch_size + _) % len(train_tokens)
            tokens = train_tokens[idx]
            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])

        max_len = max(len(b) for b in batch_x)
        x = mx.zeros((batch_size, max_len), dtype=mx.int32)
        y = mx.zeros((batch_size, max_len), dtype=mx.int32)
        for i, (bx, by) in enumerate(zip(batch_x, batch_y)):
            x[i, :len(bx)] = bx
            y[i, :len(by)] = by

        loss, grads = loss_and_grad(model, x, y)
        galore_opt.update(model, grads)

        losses.append(loss.item())
        if (step + 1) % 200 == 0 or step == 0:
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"  [{label}] Step {step+1}/{steps}: loss={loss.item():.4f} (avg200={avg:.4f})")

    train_time = time.time() - t_start
    final_avg = sum(losses[-100:]) / len(losses[-100:])
    print(f"  [{label}] Done in {train_time:.1f}s. Final avg loss: {final_avg:.4f}")
    return losses, train_time


# ===========================================================================
# PPL Evaluation
# ===========================================================================
def compute_ppl(model, val_tokens, max_batches=VAL_BATCHES):
    """Compute perplexity on validation tokens."""
    total_loss = 0.0
    total_tokens = 0

    for i, tokens in enumerate(val_tokens[:max_batches]):
        if len(tokens) < 2:
            continue
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ===========================================================================
# Adapter Training
# ===========================================================================
def train_adapter(model, domain_train_tokens, steps, lr, label="adapter"):
    """Train a LoRA adapter on domain data."""
    # Freeze base, unfreeze LoRA
    model.freeze()
    for name, p in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            pass  # will unfreeze below

    # Unfreeze LoRA params specifically
    for block in model.blocks:
        for attr_path in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                module.unfreeze()

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))

    optimizer = opt.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    for step in range(steps):
        idx = step % len(domain_train_tokens)
        tokens = domain_train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

    first_50 = sum(losses[:50]) / max(len(losses[:50]), 1)
    last_50 = sum(losses[-50:]) / max(len(losses[-50:]), 1)
    converged = last_50 < first_50 * 0.95

    return {
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "converged": converged,
        "trainable_params": trainable,
    }


# ===========================================================================
# Composition
# ===========================================================================
def compose_adapters(adapter_list, scale=None):
    """Compose multiple adapter parameter dicts with 1/N scaling."""
    N = len(adapter_list)
    if scale is None:
        scale = 1.0 / N

    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale
    return merged


def compute_cosines(adapter_list):
    """Compute pairwise cosine similarities between adapters."""
    N = len(adapter_list)
    cosines = []

    for i in range(N):
        for j in range(i + 1, N):
            # Flatten all params into single vector
            vec_i = mx.concatenate([adapter_list[i][k].flatten() for k in sorted(adapter_list[i].keys())])
            vec_j = mx.concatenate([adapter_list[j][k].flatten() for k in sorted(adapter_list[j].keys())])

            cos = mx.sum(vec_i * vec_j) / (mx.sqrt(mx.sum(vec_i**2)) * mx.sqrt(mx.sum(vec_j**2)) + 1e-8)
            mx.eval(cos)
            cosines.append(abs(cos.item()))

    return cosines


# ===========================================================================
# Single-seed run
# ===========================================================================
def run_single_seed(seed, all_train_tokens, domain_tokens, actual_vocab):
    """Run the full experiment pipeline for a single seed. Returns a results dict."""
    mx.random.seed(seed)
    print(f"\n{'#' * 70}")
    print(f"# SEED = {seed}")
    print(f"{'#' * 70}")

    seed_results = {"seed": seed}

    # ------------------------------------------------------------------
    # Phase 1: Create two models from same init
    # ------------------------------------------------------------------
    print("\n[Phase 1] Creating models...")
    model_std = TinyGPT(actual_vocab, D_MODEL, N_HEADS, N_LAYERS)
    mx.eval(model_std.parameters())

    total_params = sum(p.size for _, p in tree_flatten(model_std.parameters()))
    print(f"  Model params: {total_params:,}")
    seed_results["total_params"] = total_params

    # Clone initial weights for GaLore model
    init_weights = {name: mx.array(p) for name, p in tree_flatten(model_std.parameters())}
    mx.eval(init_weights)

    model_gal = TinyGPT(actual_vocab, D_MODEL, N_HEADS, N_LAYERS)
    model_gal.update(tree_unflatten(list(init_weights.items())))
    mx.eval(model_gal.parameters())
    print("  Both models initialized with same weights")

    # ------------------------------------------------------------------
    # Phase 2: Pretrain both models
    # ------------------------------------------------------------------
    print("\n[Phase 2a] Standard pretraining...")
    std_losses, std_time = pretrain_standard(
        model_std, all_train_tokens, all_train_tokens,
        PRETRAIN_STEPS, PRETRAIN_LR, PRETRAIN_BATCH_SIZE, "standard"
    )

    print("\n[Phase 2b] GaLore pretraining...")
    gal_losses, gal_time = pretrain_galore(
        model_gal, all_train_tokens, all_train_tokens,
        PRETRAIN_STEPS, PRETRAIN_LR, PRETRAIN_BATCH_SIZE,
        GALORE_RANK, GALORE_UPDATE_FREQ, "galore"
    )

    seed_results["pretrain"] = {
        "standard": {
            "final_loss": round(sum(std_losses[-100:]) / len(std_losses[-100:]), 4),
            "train_time_s": round(std_time, 1),
        },
        "galore": {
            "final_loss": round(sum(gal_losses[-100:]) / len(gal_losses[-100:]), 4),
            "train_time_s": round(gal_time, 1),
        },
    }

    # ------------------------------------------------------------------
    # Phase 3: Evaluate base PPL (both models, pre-quantization)
    # ------------------------------------------------------------------
    print("\n[Phase 3] Evaluating base PPL...")
    base_ppls = {"standard": {}, "galore": {}}
    for domain in DOMAINS:
        ppl_std = compute_ppl(model_std, domain_tokens[domain]["val"])
        ppl_gal = compute_ppl(model_gal, domain_tokens[domain]["val"])
        base_ppls["standard"][domain] = round(ppl_std, 2)
        base_ppls["galore"][domain] = round(ppl_gal, 2)
        print(f"  {domain}: standard={ppl_std:.2f}, galore={ppl_gal:.2f}, "
              f"ratio={ppl_gal/ppl_std:.3f}")

    mean_std = sum(base_ppls["standard"].values()) / len(DOMAINS)
    mean_gal = sum(base_ppls["galore"].values()) / len(DOMAINS)
    ppl_ratio = mean_gal / mean_std
    print(f"\n  Mean: standard={mean_std:.2f}, galore={mean_gal:.2f}, ratio={ppl_ratio:.3f}")
    seed_results["base_ppls"] = base_ppls
    seed_results["ppl_ratio"] = round(ppl_ratio, 4)

    # ------------------------------------------------------------------
    # Phase 4: Ternary quantization (both models)
    # ------------------------------------------------------------------
    print("\n[Phase 4] Ternary quantization...")
    print("  Quantizing standard model...")
    ternary_quantize(model_std)
    print("  Quantizing GaLore model...")
    ternary_quantize(model_gal)

    # Re-evaluate PPL after quantization
    print("  Re-evaluating PPL after quantization...")
    ternary_ppls = {"standard": {}, "galore": {}}
    for domain in DOMAINS:
        ppl_std = compute_ppl(model_std, domain_tokens[domain]["val"])
        ppl_gal = compute_ppl(model_gal, domain_tokens[domain]["val"])
        ternary_ppls["standard"][domain] = round(ppl_std, 2)
        ternary_ppls["galore"][domain] = round(ppl_gal, 2)

    mean_std_t = sum(ternary_ppls["standard"].values()) / len(DOMAINS)
    mean_gal_t = sum(ternary_ppls["galore"].values()) / len(DOMAINS)
    t_ratio = mean_gal_t / mean_std_t
    print(f"  After ternary: standard={mean_std_t:.2f}, galore={mean_gal_t:.2f}, ratio={t_ratio:.3f}")
    seed_results["ternary_ppls"] = ternary_ppls
    seed_results["ternary_ppl_ratio"] = round(t_ratio, 4)

    # ------------------------------------------------------------------
    # Phase 5: Train domain adapters on BOTH models
    # ------------------------------------------------------------------
    print("\n[Phase 5] Training domain adapters...")

    adapter_results = {"standard": {}, "galore": {}}
    adapter_params_std = {}
    adapter_params_gal = {}

    for model_label, model, adapter_store in [
        ("standard", model_std, adapter_params_std),
        ("galore", model_gal, adapter_params_gal),
    ]:
        print(f"\n  --- {model_label} model ---")
        apply_lora(model, LORA_RANK, LORA_SCALE)

        for domain in DOMAINS:
            print(f"    Training {domain} adapter on {model_label}...")
            zero_lora(model)
            train_result = train_adapter(
                model, domain_tokens[domain]["train"],
                ADAPTER_TRAIN_STEPS, ADAPTER_LR,
                label=f"{model_label}-{domain}"
            )
            adapter_results[model_label][domain] = train_result

            params = get_lora_params(model)
            adapter_store[domain] = params
            print(f"    {domain}: loss {train_result['first_50_loss']:.4f} -> "
                  f"{train_result['last_50_loss']:.4f} "
                  f"({'OK' if train_result['converged'] else 'NOT CONV'})")

        remove_lora(model)

    seed_results["adapter_training"] = adapter_results

    # ------------------------------------------------------------------
    # Phase 6: Evaluate individual adapter PPL
    # ------------------------------------------------------------------
    print("\n[Phase 6] Individual adapter PPL...")
    individual_ppls = {"standard": {}, "galore": {}}

    for model_label, model, adapter_store in [
        ("standard", model_std, adapter_params_std),
        ("galore", model_gal, adapter_params_gal),
    ]:
        apply_lora(model, LORA_RANK, LORA_SCALE)

        for domain in DOMAINS:
            apply_adapter_weights(model, adapter_store[domain])
            ppl = compute_ppl(model, domain_tokens[domain]["val"])
            base_ppl = ternary_ppls[model_label][domain]
            improvement = (base_ppl - ppl) / base_ppl * 100
            individual_ppls[model_label][domain] = {
                "ppl": round(ppl, 2),
                "improvement_pct": round(improvement, 2),
            }
            print(f"  [{model_label}] {domain}: {ppl:.2f} (base {base_ppl:.2f}, "
                  f"improvement {improvement:+.1f}%)")

        remove_lora(model)

    seed_results["individual_ppls"] = individual_ppls

    # ------------------------------------------------------------------
    # Phase 7: Composition evaluation
    # ------------------------------------------------------------------
    print("\n[Phase 7] Composition evaluation...")
    composition_results = {"standard": {}, "galore": {}}

    for model_label, model, adapter_store in [
        ("standard", model_std, adapter_params_std),
        ("galore", model_gal, adapter_params_gal),
    ]:
        apply_lora(model, LORA_RANK, LORA_SCALE)
        all_adapters = [adapter_store[d] for d in DOMAINS]

        composed = compose_adapters(all_adapters)
        apply_adapter_weights(model, composed)

        composed_ppls = {}
        for domain in DOMAINS:
            ppl = compute_ppl(model, domain_tokens[domain]["val"])
            composed_ppls[domain] = round(ppl, 2)
            print(f"  [{model_label}] composed {domain}: {ppl:.2f}")

        mean_individual = sum(
            individual_ppls[model_label][d]["ppl"] for d in DOMAINS
        ) / len(DOMAINS)
        mean_composed = sum(composed_ppls.values()) / len(DOMAINS)
        comp_ratio = mean_composed / mean_individual

        composition_results[model_label] = {
            "composed_ppls": composed_ppls,
            "mean_composed": round(mean_composed, 2),
            "mean_individual": round(mean_individual, 2),
            "composition_ratio": round(comp_ratio, 4),
        }
        print(f"  [{model_label}] composition ratio: {comp_ratio:.4f}")

        remove_lora(model)

    seed_results["composition"] = composition_results

    # ------------------------------------------------------------------
    # Phase 8: Cosine analysis
    # ------------------------------------------------------------------
    print("\n[Phase 8] Adapter cosine similarity...")
    cosine_results = {}
    for model_label, adapter_store in [
        ("standard", adapter_params_std),
        ("galore", adapter_params_gal),
    ]:
        all_adapters = [adapter_store[d] for d in DOMAINS]
        cosines = compute_cosines(all_adapters)
        mean_cos = sum(cosines) / len(cosines)
        max_cos = max(cosines)
        cosine_results[model_label] = {
            "mean_abs_cosine": round(mean_cos, 6),
            "max_abs_cosine": round(max_cos, 6),
            "all_cosines": [round(c, 6) for c in cosines],
        }
        print(f"  [{model_label}] mean |cos|={mean_cos:.6f}, max |cos|={max_cos:.6f}")

    seed_results["cosines"] = cosine_results

    # ------------------------------------------------------------------
    # Phase 9: Kill criteria for this seed
    # ------------------------------------------------------------------
    k1_ratio = seed_results["ternary_ppl_ratio"]
    k1_pass = k1_ratio <= 2.0

    comp_ratio_std = composition_results["standard"]["composition_ratio"]
    comp_ratio_gal = composition_results["galore"]["composition_ratio"]
    k2_ratio = comp_ratio_gal / comp_ratio_std if comp_ratio_std > 0 else float("inf")
    k2_pass = k2_ratio <= 2.0

    overall = "PASS" if (k1_pass and k2_pass) else "KILL"
    print(f"\n  [Seed {seed}] K1={k1_ratio:.4f} ({'PASS' if k1_pass else 'KILL'}), "
          f"K2={k2_ratio:.4f} ({'PASS' if k2_pass else 'KILL'}), OVERALL={overall}")

    seed_results["kill_criteria"] = {
        "k1_ppl_ratio": round(k1_ratio, 4),
        "k1_pass": k1_pass,
        "k2_comp_ratio_standard": round(comp_ratio_std, 4),
        "k2_comp_ratio_galore": round(comp_ratio_gal, 4),
        "k2_ratio_of_ratios": round(k2_ratio, 4),
        "k2_pass": k2_pass,
        "overall": overall,
    }

    return seed_results


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GaLore scaffold experiment")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated list of seeds (default: 42,123,456)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print("=" * 70)
    print("GaLore-Grown Ternary Scaffold Experiment")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    t_experiment_start = time.time()

    # ------------------------------------------------------------------
    # Phase 0: Load data (shared across seeds)
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading domain data...")
    mx.random.seed(seeds[0])  # deterministic data loading
    all_train_texts = []
    domain_data = {}
    for domain in DOMAINS:
        train_texts, val_texts = load_domain_data(domain)
        all_train_texts.extend(train_texts)
        domain_data[domain] = (train_texts, val_texts)
        print(f"  {domain}: {len(train_texts)} train, {len(val_texts)} val")

    encode, decode, actual_vocab = build_tokenizer(all_train_texts, VOCAB_SIZE)
    print(f"  Vocabulary size: {actual_vocab}")

    print("  Tokenizing...")
    all_train_tokens = tokenize_texts(all_train_texts, encode)
    domain_tokens = {}
    for domain in DOMAINS:
        train_texts, val_texts = domain_data[domain]
        domain_tokens[domain] = {
            "train": tokenize_texts(train_texts, encode),
            "val": tokenize_texts(val_texts, encode),
        }
    print(f"  Total pretrain sequences: {len(all_train_tokens)}")

    # ------------------------------------------------------------------
    # Run each seed
    # ------------------------------------------------------------------
    per_seed_results = []
    for seed in seeds:
        seed_result = run_single_seed(seed, all_train_tokens, domain_tokens, actual_vocab)
        per_seed_results.append(seed_result)

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MULTI-SEED AGGREGATION")
    print("=" * 70)

    k1_ratios = [r["kill_criteria"]["k1_ppl_ratio"] for r in per_seed_results]
    k2_ratios = [r["kill_criteria"]["k2_ratio_of_ratios"] for r in per_seed_results]
    ppl_ratios_prequant = [r["ppl_ratio"] for r in per_seed_results]
    comp_ratios_std = [r["composition"]["standard"]["composition_ratio"] for r in per_seed_results]
    comp_ratios_gal = [r["composition"]["galore"]["composition_ratio"] for r in per_seed_results]
    mean_cos_std = [r["cosines"]["standard"]["mean_abs_cosine"] for r in per_seed_results]
    mean_cos_gal = [r["cosines"]["galore"]["mean_abs_cosine"] for r in per_seed_results]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    k1_mean = mean(k1_ratios)
    k1_std = std(k1_ratios)
    k2_mean = mean(k2_ratios)
    k2_std = std(k2_ratios)

    print(f"\n  K1 (ternary PPL ratio) per seed: {[f'{r:.4f}' for r in k1_ratios]}")
    print(f"  K1 mean: {k1_mean:.4f} +/- {k1_std:.4f}")
    print(f"  K2 (comp ratio ratio) per seed: {[f'{r:.4f}' for r in k2_ratios]}")
    print(f"  K2 mean: {k2_mean:.4f} +/- {k2_std:.4f}")

    k1_pass = k1_mean <= 2.0
    k2_pass = k2_mean <= 2.0
    overall = "PASS" if (k1_pass and k2_pass) else "KILL"

    print(f"\n  K1 (mean <= 2.0): {'PASS' if k1_pass else 'KILL'} ({k1_mean:.4f})")
    print(f"  K2 (mean <= 2.0): {'PASS' if k2_pass else 'KILL'} ({k2_mean:.4f})")
    print(f"  OVERALL: {overall}")

    # Compute quant degradation per seed
    quant_degrad_std = []
    quant_degrad_gal = []
    for r in per_seed_results:
        mean_pre_std = mean(list(r["base_ppls"]["standard"].values()))
        mean_post_std = mean(list(r["ternary_ppls"]["standard"].values()))
        mean_pre_gal = mean(list(r["base_ppls"]["galore"].values()))
        mean_post_gal = mean(list(r["ternary_ppls"]["galore"].values()))
        quant_degrad_std.append(mean_post_std / mean_pre_std)
        quant_degrad_gal.append(mean_post_gal / mean_pre_gal)

    total_time = time.time() - t_experiment_start

    # Build final results
    results = {
        "experiment": "bitnet_galore_scaffold",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "pretrain_steps": PRETRAIN_STEPS,
        "galore_rank": GALORE_RANK,
        "galore_update_freq": GALORE_UPDATE_FREQ,
        "lora_rank": LORA_RANK,
        "adapter_train_steps": ADAPTER_TRAIN_STEPS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "actual_vocab_size": actual_vocab,
        "total_pretrain_sequences": len(all_train_tokens),
        "total_params": per_seed_results[0]["total_params"],
        "per_seed": per_seed_results,
        "aggregate": {
            "k1_ppl_ratios": [round(r, 4) for r in k1_ratios],
            "k1_mean": round(k1_mean, 4),
            "k1_std": round(k1_std, 4),
            "k2_ratios": [round(r, 4) for r in k2_ratios],
            "k2_mean": round(k2_mean, 4),
            "k2_std": round(k2_std, 4),
            "prequant_ppl_ratios": [round(r, 4) for r in ppl_ratios_prequant],
            "prequant_ppl_ratio_mean": round(mean(ppl_ratios_prequant), 4),
            "quant_degradation_standard": [round(r, 4) for r in quant_degrad_std],
            "quant_degradation_standard_mean": round(mean(quant_degrad_std), 4),
            "quant_degradation_galore": [round(r, 4) for r in quant_degrad_gal],
            "quant_degradation_galore_mean": round(mean(quant_degrad_gal), 4),
            "comp_ratio_standard": [round(r, 4) for r in comp_ratios_std],
            "comp_ratio_standard_mean": round(mean(comp_ratios_std), 4),
            "comp_ratio_galore": [round(r, 4) for r in comp_ratios_gal],
            "comp_ratio_galore_mean": round(mean(comp_ratios_gal), 4),
            "mean_cos_standard": [round(r, 6) for r in mean_cos_std],
            "mean_cos_standard_mean": round(mean(mean_cos_std), 6),
            "mean_cos_galore": [round(r, 6) for r in mean_cos_gal],
            "mean_cos_galore_mean": round(mean(mean_cos_gal), 6),
        },
        "kill_criteria": {
            "k1_ppl_ratio_mean": round(k1_mean, 4),
            "k1_ppl_ratio_std": round(k1_std, 4),
            "k1_pass": k1_pass,
            "k2_ratio_of_ratios_mean": round(k2_mean, 4),
            "k2_ratio_of_ratios_std": round(k2_std, 4),
            "k2_pass": k2_pass,
            "overall": overall,
        },
        "total_time_s": round(total_time, 1),
        "total_time_min": round(total_time / 60, 1),
    }

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"  Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
