#!/usr/bin/env python3
"""M2P Layer Depth Scaling: Option A (single call) vs Option B (per-layer calls) at L=2,4,8,16.

TYPE: frontier-extension (Type 3)
MATH: experiments/models/m2p_layer_depth/MATH.md

PRIOR FINDINGS:
  Finding #359 (exp_m2p_data_scale):   d=256, L=2, quality_ratio=97.6%
  Finding #361 (exp_m2p_macro_quality): d=512, L=2, quality_ratio=101.0%
  Finding #362 (exp_m2p_qwen3_quality): d=1024, L=2, quality_ratio=99.6%
  Proven recipe: d_M2P=64, L_m2p=2, n=2000, T=1000, GL alpha=5.0

QUESTION:
  Does M2P maintain quality when target network depth scales from L=2 (proven)
  to L=4, 8, 16? Two strategies:
  Option A: Single M2P forward pass generates ALL L layers' B-matrices.
            Output head grows as O(n_layers). Motivated by Ha et al. (arXiv:1609.09106).
  Option B: One independent M2P call per layer. L calls total.
            Proven correct by induction from Finding #362. Provides the gold baseline.

MATHEMATICAL FRAMEWORK:
  Theorem 1 (MATH.md): n_train>=T guarantee is n_layers-independent (Ghadimi-Lan).
    At n=2000: T/n_train = 0.625 < 1 epoch. Holds for all L.
  Theorem 2 (MATH.md): Option B correct by induction from Finding #362.
    Each M2P call in Option B = proven recipe. Quality >= 85% guaranteed at all L.
  Theorem 3 (MATH.md): Option A works IFF effective rank([B_1*,...,B_L*]) <= d_M2P=64.
    Ha et al. (arXiv:1609.09106): hypernetworks achieve 90-95% of per-layer quality.
    Compression analogy: L=4 -> 256:1 (= Finding #361), L=8 -> 512:1 (= Finding #362).

KILL CRITERIA:
  K891: Option A quality_ratio >= 85% at L=16 (single M2P, output head scales O(n_layers))
  K892: Option B quality_ratio >= 85% at L=16 (L independent M2P calls, proven recipe x L)
  K893 (KILL): Option A quality_ratio < 50% at L=4 -> output head bottleneck, kill

ARCHITECTURE FIXED (same as proven recipe):
  d_model=256, n_heads=4, vocab=128, block_size=48
  lora_rank=4, lora_scale=2.0
  d_M2P=64, L_m2p=2 M2P layers, N_memory=32
  n=2000 (n_train=1600), T=1000, GL alpha=5.0, patience=5, interval=50
  Domains: sort + reverse (2 valid domains; arithmetic excluded by parity guard at d=256+)
  Note: arithmetic at d=256 may pass parity guard -- we include all 3 and let parity guard decide.

SWEEP: n_layers in {2, 4, 8, 16}
OUTPUT: results.json with option_a_quality_ratio_{L} and option_b_quality_ratio_{L} for each L.
"""

import gc
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

# Memory safety (CODING_GUIDELINES §2)
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ====================================================================
# ARCHITECTURE CONSTANTS (fixed at proven micro recipe)
# ====================================================================
D_MODEL = 256
N_HEADS = 4          # d_head = 256/4 = 64 (same as all prior experiments)
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0

# M2P architecture FIXED (proven recipe)
M2P_LAYERS = 2
D_M2P = 64
N_MEMORY = 32

# Domain configuration
N_DOMAINS = 3
DOMAIN_NAMES = ["arithmetic", "sort", "reverse"]

# Module names and output dims at D_MODEL=256
MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS_BASE = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
# [256, 256, 256, 256, 1024] at d=256

# Sweep variable: n_layers for the target transformer
LAYER_VALUES = [2, 4, 8, 16] if not SMOKE_TEST else [2, 4]

# Training constants
N_SAMPLES = 2000
T_FIXED = 1000
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3
BASE_STEPS = 1200

if SMOKE_TEST:
    N_SAMPLES = 80
    T_FIXED = 10
    BASE_STEPS = 30

# Early stopping (Prechelt 1998 GL criterion, proven recipe)
EARLY_STOP_INTERVAL = 50
GL_THRESHOLD = 5.0
PATIENCE = 5

# Parity guard threshold
PARITY_GUARD_THRESHOLD = 0.05


# ====================================================================
# UTILITIES
# ====================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def log(m): print(m, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


# ====================================================================
# DATA GENERATION (same as prior experiments — 3 domains)
# ====================================================================

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    """Generate synthetic task data. Same logic as all prior M2P experiments."""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        if domain_id == 0:  # arithmetic
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            data.append(f"{a}+{b}={a+b}")
        elif domain_id == 1:  # sort
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(sorted(s))}")
        elif domain_id == 2:  # reverse
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(reversed(s))}")
    return data


def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]


def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]


# ====================================================================
# TOY GPT (parameterized by n_layers — THE SWEEP VARIABLE)
# ====================================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scale = hd ** -0.5
        attn = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyGPT(nn.Module):
    """Toy GPT with configurable depth. d=256, 4 heads. d_head=64 (fixed)."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.wpe = nn.Embedding(BLOCK_SIZE + 1, D_MODEL)
        self.blocks = [Block(D_MODEL, N_HEADS) for _ in range(n_layers)]
        self.norm_f = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def get_hidden_states(self, tokens):
        """Return list of per-layer hidden states (for M2P context input)."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states


# ====================================================================
# GRASSMANNIAN A-MATRICES (parameterized by n_layers)
# ====================================================================

def generate_grassmannian_A(n_domains, n_layers, n_modules, d, rank, seed=42):
    """Orthogonal A-matrices for Grassmannian LoRA (same as all prior experiments)."""
    total_rank = n_domains * rank
    assert total_rank <= d, (
        f"total_rank={total_rank} > d={d}: "
        f"increase d_model or reduce n_domains * rank"
    )
    rng = np.random.RandomState(seed)
    A_matrices = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start + rank])
    return A_matrices


# ====================================================================
# LORA FORWARD PASS (parameterized by n_layers via B_matrices keys)
# ====================================================================

def lora_forward_with_B(base: ToyGPT, tokens, A_matrices, domain_id, B_matrices):
    """Forward pass with Grassmannian LoRA. Handles arbitrary n_layers."""
    n_modules = len(MODULE_NAMES)
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        def _apply_lora(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            A = A_matrices[(domain_id, li, mi)]
            B = B_matrices[(li, mi)]
            return base_out + LORA_SCALE * (x_in @ A) @ B

        q = _apply_lora(attn.wq, x_norm, li, 0)
        k = _apply_lora(attn.wk, x_norm, li, 1)
        v = _apply_lora(attn.wv, x_norm, li, 2)

        q = q.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)

        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        scale_factor = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale_factor + mask, axis=-1)
        attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(B_b, T_b, C)

        attn_out = _apply_lora(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        x_norm2 = block.norm2(x)
        fc1_base = block.mlp.fc1(x_norm2)
        A_fc1 = A_matrices[(domain_id, li, 4)]
        B_fc1 = B_matrices[(li, 4)]
        fc1_out = fc1_base + LORA_SCALE * (x_norm2 @ A_fc1) @ B_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    return base.lm_head(base.norm_f(x))


# ====================================================================
# SFT B-MATRICES CONTAINER (parameterized by n_layers)
# ====================================================================

def make_b_container_class(n_layers: int):
    """Dynamically create a BMatrices module for given n_layers."""
    class BMatrices(nn.Module):
        def __init__(self):
            super().__init__()
            for li in range(n_layers):
                for mi in range(len(MODULE_NAMES)):
                    d_out = MODULE_OUT_DIMS_BASE[mi]
                    setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

        def as_dict(self):
            return {
                (li, mi): getattr(self, f"B_{li}_{mi}")
                for li in range(n_layers) for mi in range(len(MODULE_NAMES))
            }
    return BMatrices


def sft_loss_fn(b_container, base, tokens, A_matrices, domain_id):
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ====================================================================
# M2P TRANSFORMER — Option A: Single call, generates ALL n_layers' B-matrices
# ====================================================================

class M2PAttention(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        scale = hd ** -0.5
        scores = q @ k.transpose(0, 1, 3, 2) * scale
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scores = scores + mask
        a = mx.softmax(scores, axis=-1)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class M2PMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class M2PBlock(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = M2PAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = M2PMLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class M2PTransformerOptionA(nn.Module):
    """Option A: Single M2P call generates ALL n_layers B-matrices at once.
    Output head = n_layers × LORA_RANK × d_out per module.
    Compression at L=2: 128:1 (= Finding #359)
    Compression at L=4: 256:1 (= Finding #361, proven)
    Compression at L=8: 512:1 (= Finding #362, proven)
    Compression at L=16: 1024:1 (frontier, the key test)
    """
    def __init__(self, n_layers: int, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.d_base = d_base
        self.d_m2p = d_m2p
        # Input projection: d_base -> d_m2p (compresses context to M2P space)
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(m2p_layers)]
        self.norm_f = RMSNorm(d_m2p)
        # Output heads: d_m2p -> n_layers × LORA_RANK × d_out_m (generates ALL layers)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_BASE)):
            total_out = n_layers * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list):
        """hidden_states_list: list of per-layer hidden states from base model.
        Returns dict: (layer_idx, module_idx) -> B_matrix.
        """
        # Pool each layer's hidden states to get context encodings
        layer_encodings = []
        for h in hidden_states_list:
            pooled = mx.mean(h[0], axis=0)  # (d_base,)
            enc = self.input_proj(pooled)    # (d_m2p,)
            layer_encodings.append(enc)

        # Build memory tokens with positional embeddings
        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)
        # Condition memory on mean context encoding
        context_enc = mx.mean(mx.stack(layer_encodings, axis=0), axis=0)
        memory = memory + context_enc[None, :]

        # M2P transformer forward
        x = memory[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        pooled_memory = mx.mean(x[0], axis=0)  # (d_m2p,)

        # Generate ALL layers' B-matrices from single pooled memory
        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_BASE)):
            out = self.out_heads[mname](pooled_memory)          # (n_layers * rank * d_out,)
            out = out.reshape(self.n_layers, LORA_RANK, d_out)  # (n_layers, rank, d_out)
            for li in range(self.n_layers):
                B_matrices[(li, mi)] = out[li]  # (rank, d_out)
        return B_matrices


class M2PTransformerOptionB(nn.Module):
    """Option B: One M2P call per layer. Each call generates ONE layer's B-matrices.
    This is exactly the proven recipe (Finding #362) applied L times independently.
    Output head = LORA_RANK × d_out per module (same as proven case, L=2).
    """
    def __init__(self, n_layers: int, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.d_base = d_base
        self.d_m2p = d_m2p
        # L independent M2P sub-networks, one per layer
        self.sub_m2ps = [
            _SingleLayerM2P(d_base, d_m2p, m2p_layers) for _ in range(n_layers)
        ]

    def __call__(self, hidden_states_list):
        """Call each sub-M2P for its corresponding layer.
        Returns dict: (layer_idx, module_idx) -> B_matrix.
        """
        B_matrices = {}
        for li in range(self.n_layers):
            b_for_layer = self.sub_m2ps[li](hidden_states_list)  # uses all context
            for mi in range(len(MODULE_NAMES)):
                B_matrices[(li, mi)] = b_for_layer[(0, mi)]  # reindex to (li, mi)
        return B_matrices


class _SingleLayerM2P(nn.Module):
    """Single-layer M2P: generates B-matrices for exactly ONE target layer.
    This is the proven architecture from Finding #362.
    """
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(m2p_layers)]
        self.norm_f = RMSNorm(d_m2p)
        # Output heads: d_m2p -> 1 × LORA_RANK × d_out (generates ONE layer)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_BASE)):
            total_out = 1 * LORA_RANK * d_out  # n_layers=1 for single-layer M2P
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list):
        """Returns dict: (0, module_idx) -> B_matrix for ONE layer."""
        layer_encodings = []
        for h in hidden_states_list:
            pooled = mx.mean(h[0], axis=0)
            enc = self.input_proj(pooled)
            layer_encodings.append(enc)

        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)
        context_enc = mx.mean(mx.stack(layer_encodings, axis=0), axis=0)
        memory = memory + context_enc[None, :]

        x = memory[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        pooled_memory = mx.mean(x[0], axis=0)

        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_BASE)):
            out = self.out_heads[mname](pooled_memory)   # (rank * d_out,)
            out = out.reshape(LORA_RANK, d_out)
            B_matrices[(0, mi)] = out
        return B_matrices


# ====================================================================
# LOSS FUNCTIONS
# ====================================================================

def m2p_ntp_loss(m2p, base, A_matrices, domain_id, tokens):
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ====================================================================
# EVALUATION HELPERS
# ====================================================================

def eval_ntp_loss(base, batches, A_matrices=None, domain_id=None, B_matrices=None):
    """Evaluate NTP loss on val set. Cleans up each forward pass."""
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        if A_matrices is not None and B_matrices is not None:
            logits = lora_forward_with_B(base, tokens_2d, A_matrices,
                                          domain_id, B_matrices)
        else:
            logits = base(tokens_2d)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def save_b_matrices(B_matrices, path, n_layers):
    """Save B-matrices dict to .npz file."""
    np_dict = {f"{li}_{mi}": np.array(B_matrices[(li, mi)])
               for li in range(n_layers) for mi in range(len(MODULE_NAMES))}
    np.savez(str(path), **np_dict)


def load_b_matrices(path, n_layers):
    """Load B-matrices from .npz file."""
    data = np.load(str(path))
    return {
        (li, mi): mx.array(data[f"{li}_{mi}"])
        for li in range(n_layers) for mi in range(len(MODULE_NAMES))
    }


# ====================================================================
# PHASE FUNCTIONS (each self-contained per CODING_GUIDELINES §1)
# ====================================================================

def phase_generate_data(rng: np.random.RandomState, n_per_domain: int) -> dict:
    """Generate train/val data for 3 domains."""
    log(f"\n=== Phase: Generate Data (n_per_domain={n_per_domain}) ===")
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        train_texts = texts[:split]
        val_texts = texts[split:]
        domain_data[name] = {
            "train": make_batches(train_texts),
            "val": make_batches(val_texts),
            "domain_id": di,
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, "
            f"{len(domain_data[name]['val'])} val")
    return domain_data


def phase_pretrain_base(n_layers: int, domain_data: dict):
    """Pre-train ToyGPT with n_layers on all domains. Returns (base, base_losses)."""
    log(f"\n=== Phase: Pre-train Base Model (L={n_layers}) ===")
    log(f"  ToyGPT: D_MODEL={D_MODEL}, N_LAYERS={n_layers}, N_HEADS={N_HEADS}, "
        f"d_head={D_MODEL//N_HEADS}")
    mx.random.seed(SEED)

    base = ToyGPT(n_layers=n_layers)
    mx.eval(base.parameters())

    param_count = sum(p.size for _, p in tree_flatten(base.parameters()))
    log(f"  ToyGPT params: {param_count:,}")

    all_train = []
    for name in DOMAIN_NAMES:
        all_train.extend(domain_data[name]["train"])

    optimizer = opt.Adam(learning_rate=LR)

    def loss_fn(model, tokens):
        tokens_2d = tokens[None, :]
        logits = model(tokens_2d)
        return nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(base, loss_fn)

    gc.disable()
    for step in range(BASE_STEPS):
        tokens = all_train[step % len(all_train)]
        loss, grads = loss_and_grad(base, tokens)
        optimizer.update(base, grads)
        mx.eval(base.parameters(), optimizer.state, loss)
        if (step + 1) % max(1, BASE_STEPS // 4) == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={loss.item():.4f}")
    gc.enable()
    cleanup(optimizer)

    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")
    return base, base_losses


def phase_grassmannian(n_layers: int, base) -> tuple:
    """Generate Grassmannian A-matrices for n_layers target transformer."""
    log(f"\n=== Phase: Grassmannian A-matrices (L={n_layers}) ===")
    log(f"  D_MODEL={D_MODEL}, N_DOMAINS={N_DOMAINS}, N_LAYERS={n_layers}, "
        f"LORA_RANK={LORA_RANK}")
    log(f"  total_rank = {N_DOMAINS * LORA_RANK} (must be << {D_MODEL})")
    A_matrices = generate_grassmannian_A(
        N_DOMAINS, n_layers, len(MODULE_NAMES), D_MODEL, LORA_RANK, seed=SEED
    )
    # Quick orthogonality check
    cos_values = []
    for li in range(min(n_layers, 2)):  # check first 2 layers only for speed
        for mi in range(len(MODULE_NAMES)):
            for di in range(N_DOMAINS):
                for dj in range(di + 1, N_DOMAINS):
                    ai = A_matrices[(di, li, mi)].reshape(-1)
                    aj = A_matrices[(dj, li, mi)].reshape(-1)
                    cos = mx.abs(
                        mx.sum(ai * aj) /
                        (mx.linalg.norm(ai) * mx.linalg.norm(aj) + 1e-12)
                    ).item()
                    cos_values.append(cos)
    max_cos = float(np.max(cos_values)) if cos_values else 0.0
    log(f"  Max|cos| between domain subspaces (first 2 layers): {max_cos:.6f}")
    assert max_cos < 1e-5, f"Grassmannian guarantee failed: max|cos|={max_cos}"
    return A_matrices


def phase_sft_domain(n_layers, domain_name, domain_id, domain_data, base,
                      A_matrices, base_loss, adapter_dir) -> dict:
    """Train SFT LoRA adapter for one domain and n_layers configuration.

    SFT defines the 100% quality reference. Uses T_FIXED steps (same as M2P).
    """
    local_path = adapter_dir / f"sft_{domain_name}.npz"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"  SFT {domain_name} (L={n_layers}, domain {domain_id})...")

    if local_path.exists():
        log(f"    Reusing cached SFT adapter")
        B_matrices = load_b_matrices(str(local_path), n_layers)
    else:
        BMatrices = make_b_container_class(n_layers)
        b_container = BMatrices()
        mx.eval(b_container.parameters())
        optimizer = opt.Adam(learning_rate=SFT_LR)

        def _loss(b_cont, tokens):
            return sft_loss_fn(b_cont, base, tokens[None, :], A_matrices, domain_id)

        grad_fn = nn.value_and_grad(b_container, _loss)
        train_batches = domain_data["train"]

        gc.disable()
        for step in range(T_FIXED):
            tokens = train_batches[step % len(train_batches)]
            loss, grads = grad_fn(b_container, tokens)
            optimizer.update(b_container, grads)
            mx.eval(b_container.parameters(), optimizer.state, loss)
        gc.enable()
        cleanup(optimizer)

        B_matrices = b_container.as_dict()
        save_b_matrices(B_matrices, str(local_path), n_layers)
        cleanup(b_container)

    sft_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)
    gap = base_loss - sft_loss
    log(f"    SFT loss={sft_loss:.4f} base={base_loss:.4f} gap={gap:.4f}")
    cleanup(B_matrices)
    return {"sft_loss": round(sft_loss, 4), "gap": round(gap, 4)}


def phase_sft_all_domains(n_layers, domain_data, base, A_matrices, base_losses,
                           adapter_dir) -> dict:
    log(f"\n=== Phase: SFT Baselines (L={n_layers}) ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            n_layers, name, di, domain_data[name], base, A_matrices,
            base_losses[name], adapter_dir
        )
        sft_results[name] = result
    return sft_results


def _train_m2p_with_gl(m2p, base, A_matrices, domain_id, domain_data, option_label,
                        n_layers):
    """Train M2P (any option) with GL early stopping. Returns (B_matrices, diagnostics)."""
    train_batches = domain_data["train"]
    val_batches = domain_data["val"]

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id, tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    optimizer = opt.Adam(learning_rate=M2P_LR)

    best_val_loss = float("inf")
    best_step = 0
    consecutive_gl_exceeded = 0
    early_stop_triggered = False
    stopping_step = T_FIXED
    final_train_loss = None
    val_loss_trajectory = []

    gc.disable()
    for step in range(T_FIXED):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        final_train_loss = loss.item()

        if (step + 1) % EARLY_STOP_INTERVAL == 0 and not early_stop_triggered:
            gc.enable()
            context_tokens = train_batches[0][None, :]
            hidden_states = base.get_hidden_states(context_tokens)
            B_now = m2p(hidden_states)
            mx.eval(*[B_now[(li, mi)] for li in range(n_layers)
                      for mi in range(len(MODULE_NAMES))])
            val_loss_now = eval_ntp_loss(base, val_batches, A_matrices,
                                          domain_id, B_now)
            del B_now
            gc.disable()

            val_loss_trajectory.append({"step": step + 1, "val_loss": round(val_loss_now, 4)})

            if val_loss_now < best_val_loss:
                best_val_loss = val_loss_now
                best_step = step + 1
                consecutive_gl_exceeded = 0
            else:
                gl = 100.0 * (val_loss_now / best_val_loss - 1.0)
                if gl > GL_THRESHOLD:
                    consecutive_gl_exceeded += 1
                    if consecutive_gl_exceeded >= PATIENCE:
                        early_stop_triggered = True
                        stopping_step = step + 1
                        log(f"      Early stop at step {stopping_step}: "
                            f"GL={gl:.2f} > {GL_THRESHOLD} for {PATIENCE} checks. "
                            f"best_val_loss={best_val_loss:.4f}")
                        break
                else:
                    consecutive_gl_exceeded = 0

    gc.enable()
    cleanup(optimizer)

    # Generate final B-matrices from first training context
    context_tokens = train_batches[0][None, :]
    hidden_states = base.get_hidden_states(context_tokens)
    B_matrices = m2p(hidden_states)
    mx.eval(*[B_matrices[(li, mi)] for li in range(n_layers)
              for mi in range(len(MODULE_NAMES))])

    diag = {
        "final_train_loss": round(final_train_loss, 4) if final_train_loss else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "best_step": best_step,
        "stopping_step": stopping_step,
        "early_stop_triggered": early_stop_triggered,
        "val_loss_trajectory": val_loss_trajectory,
    }
    return B_matrices, diag


def phase_train_m2p_option_a(
    n_layers: int,
    domain_name: str,
    domain_id: int,
    domain_data: dict,
    base,
    A_matrices: dict,
    base_loss: float,
    sft_loss: float,
    adapter_dir: Path,
) -> dict:
    """Train Option A M2P (single call, all layers) for one domain.

    Option A output head at this L:
      fc1: d_M2P=64 -> n_layers * LORA_RANK * 1024
           L=2: 8192 (128:1 compression = Finding #359)
           L=4: 16384 (256:1 = Finding #361, proven 101%)
           L=8: 32768 (512:1 = Finding #362, proven 99.6%)
           L=16: 65536 (1024:1, frontier)
    """
    save_path = adapter_dir / f"optA_{domain_name}.npz"

    log(f"  [Option A, L={n_layers}] Training M2P for {domain_name}...")
    fc1_out_dim = n_layers * LORA_RANK * MODULE_OUT_DIMS_BASE[4]
    fc1_compression = fc1_out_dim / D_M2P
    log(f"    Output head fc1: {D_M2P} -> {fc1_out_dim} (compression {fc1_compression:.0f}:1)")
    mx.random.seed(SEED)

    m2p = M2PTransformerOptionA(n_layers=n_layers)
    mx.eval(m2p.parameters())
    param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"    M2P-A params: {param_count:,}")

    B_matrices, diag = _train_m2p_with_gl(
        m2p, base, A_matrices, domain_id, domain_data, "A", n_layers
    )

    save_b_matrices(B_matrices, str(save_path), n_layers)
    cleanup(m2p)

    B_loaded = load_b_matrices(str(save_path), n_layers)
    m2p_val_loss = eval_ntp_loss(base, domain_data["val"], A_matrices,
                                  domain_id, B_loaded)
    cleanup(B_loaded)

    final_train_loss = diag["final_train_loss"]
    train_val_gap = abs(final_train_loss - m2p_val_loss) if final_train_loss else None

    quality_ratio = 0.0
    excluded_parity = False
    gap = base_loss - sft_loss
    if gap > PARITY_GUARD_THRESHOLD:
        quality_ratio = (base_loss - m2p_val_loss) / gap
    else:
        excluded_parity = True

    log(f"    [A, L={n_layers}] {domain_name}: m2p_val={m2p_val_loss:.4f} "
        f"sft={sft_loss:.4f} base={base_loss:.4f} "
        f"quality={quality_ratio:.1%} gap={gap:.4f} "
        f"{'EXCLUDED(parity)' if excluded_parity else ''}")

    return {
        "m2p_val_loss": round(m2p_val_loss, 4),
        "final_train_loss": diag["final_train_loss"],
        "train_val_gap": round(train_val_gap, 4) if train_val_gap else None,
        "quality_ratio": round(quality_ratio, 4),
        "excluded_parity": excluded_parity,
        "gap": round(gap, 4),
        "early_stop_triggered": diag["early_stop_triggered"],
        "stopping_step": diag["stopping_step"],
        "best_val_loss": diag["best_val_loss"],
        "fc1_compression_ratio": float(n_layers * LORA_RANK * MODULE_OUT_DIMS_BASE[4] / D_M2P),
        "val_loss_trajectory": diag["val_loss_trajectory"],
    }


def phase_train_m2p_option_b(
    n_layers: int,
    domain_name: str,
    domain_id: int,
    domain_data: dict,
    base,
    A_matrices: dict,
    base_loss: float,
    sft_loss: float,
    adapter_dir: Path,
) -> dict:
    """Train Option B M2P (L independent calls, one per layer) for one domain.

    Each sub-M2P is identical to the proven single-layer recipe (Finding #362).
    Output head per sub-M2P: d_M2P=64 -> LORA_RANK × d_out (constant, not scaled by L).
    Theorem 2 (MATH.md) guarantees quality >= 85% by induction from Finding #362.
    """
    save_path = adapter_dir / f"optB_{domain_name}.npz"

    log(f"  [Option B, L={n_layers}] Training M2P for {domain_name}...")
    log(f"    {n_layers} independent sub-M2Ps, each = proven recipe (Finding #362)")
    mx.random.seed(SEED)

    m2p = M2PTransformerOptionB(n_layers=n_layers)
    mx.eval(m2p.parameters())
    param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"    M2P-B params: {param_count:,} ({n_layers} × proven recipe)")

    B_matrices, diag = _train_m2p_with_gl(
        m2p, base, A_matrices, domain_id, domain_data, "B", n_layers
    )

    save_b_matrices(B_matrices, str(save_path), n_layers)
    cleanup(m2p)

    B_loaded = load_b_matrices(str(save_path), n_layers)
    m2p_val_loss = eval_ntp_loss(base, domain_data["val"], A_matrices,
                                  domain_id, B_loaded)
    cleanup(B_loaded)

    final_train_loss = diag["final_train_loss"]
    train_val_gap = abs(final_train_loss - m2p_val_loss) if final_train_loss else None

    quality_ratio = 0.0
    excluded_parity = False
    gap = base_loss - sft_loss
    if gap > PARITY_GUARD_THRESHOLD:
        quality_ratio = (base_loss - m2p_val_loss) / gap
    else:
        excluded_parity = True

    log(f"    [B, L={n_layers}] {domain_name}: m2p_val={m2p_val_loss:.4f} "
        f"sft={sft_loss:.4f} base={base_loss:.4f} "
        f"quality={quality_ratio:.1%} gap={gap:.4f} "
        f"{'EXCLUDED(parity)' if excluded_parity else ''}")

    return {
        "m2p_val_loss": round(m2p_val_loss, 4),
        "final_train_loss": diag["final_train_loss"],
        "train_val_gap": round(train_val_gap, 4) if train_val_gap else None,
        "quality_ratio": round(quality_ratio, 4),
        "excluded_parity": excluded_parity,
        "gap": round(gap, 4),
        "early_stop_triggered": diag["early_stop_triggered"],
        "stopping_step": diag["stopping_step"],
        "best_val_loss": diag["best_val_loss"],
        "val_loss_trajectory": diag["val_loss_trajectory"],
    }


def phase_depth_sweep(domain_data: dict) -> dict:
    """Sweep n_layers in LAYER_VALUES. For each L, train base + SFT + M2P-A + M2P-B.

    Each L is its own complete experiment (separate base model, separate SFT reference).
    Phase structure per L:
    1. Pre-train ToyGPT(L)
    2. Generate Grassmannian A-matrices for L layers
    3. Train SFT baselines (gold reference)
    4. Train M2P Option A (single call, output head = L × proven head)
    5. Train M2P Option B (L independent calls, proven recipe × L)
    """
    sweep_results = {}

    for n_layers in LAYER_VALUES:
        log(f"\n{'='*70}")
        log(f"DEPTH SWEEP: L={n_layers}")
        log(f"  Option A fc1 compression: "
            f"{n_layers * LORA_RANK * MODULE_OUT_DIMS_BASE[4] / D_M2P:.0f}:1")
        log(f"  Option B: {n_layers} independent M2P calls (proven recipe × L)")
        log(f"{'='*70}")

        adapter_dir = EXPERIMENT_DIR / "adapters" / f"L{n_layers}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Pre-train base
        base, base_losses = phase_pretrain_base(n_layers, domain_data)
        log_memory(f"after base L={n_layers}")

        # Phase 2: Grassmannian A-matrices
        A_matrices = phase_grassmannian(n_layers, base)
        log_memory(f"after grassmannian L={n_layers}")

        # Phase 3: SFT baselines
        sft_results = phase_sft_all_domains(
            n_layers, domain_data, base, A_matrices, base_losses, adapter_dir
        )
        log_memory(f"after SFT L={n_layers}")

        # Phase 4: Option A training (per domain)
        log(f"\n--- Option A (single M2P call, output head O(L)) ---")
        optA_by_domain = {}
        for di, name in enumerate(DOMAIN_NAMES):
            result = phase_train_m2p_option_a(
                n_layers, name, di, domain_data[name], base, A_matrices,
                base_losses[name], sft_results[name]["sft_loss"], adapter_dir
            )
            optA_by_domain[name] = result
        log_memory(f"after Option A L={n_layers}")

        # Phase 5: Option B training (per domain)
        log(f"\n--- Option B ({n_layers} independent M2P calls) ---")
        optB_by_domain = {}
        for di, name in enumerate(DOMAIN_NAMES):
            result = phase_train_m2p_option_b(
                n_layers, name, di, domain_data[name], base, A_matrices,
                base_losses[name], sft_results[name]["sft_loss"], adapter_dir
            )
            optB_by_domain[name] = result
        log_memory(f"after Option B L={n_layers}")

        # Aggregate quality ratios (parity-filtered)
        def aggregate_quality(domain_results):
            valid = [r["quality_ratio"] for r in domain_results.values()
                     if not r["excluded_parity"]]
            excluded = [name for name, r in domain_results.items()
                        if r["excluded_parity"]]
            median_q = float(np.median(valid)) if valid else 0.0
            mean_q = float(np.mean(valid)) if valid else 0.0
            max_gap = max(
                (r["train_val_gap"] for r in domain_results.values()
                 if not r["excluded_parity"] and r["train_val_gap"] is not None),
                default=None
            )
            return {
                "median_quality": round(median_q, 4),
                "mean_quality": round(mean_q, 4),
                "max_train_val_gap": round(max_gap, 4) if max_gap is not None else None,
                "excluded_domains": excluded,
                "n_valid": len(valid),
                "per_domain": {
                    name: round(r["quality_ratio"], 4)
                    for name, r in domain_results.items()
                    if not r["excluded_parity"]
                },
            }

        optA_agg = aggregate_quality(optA_by_domain)
        optB_agg = aggregate_quality(optB_by_domain)

        # Option A vs Option B comparison
        optA_q = optA_agg["median_quality"]
        optB_q = optB_agg["median_quality"]
        ratio_A_vs_B = optA_q / optB_q if optB_q > 0 else 0.0

        log(f"\n  SUMMARY L={n_layers}:")
        log(f"  Option A quality: {optA_q:.1%} (median valid domains)")
        log(f"  Option B quality: {optB_q:.1%} (median valid domains)")
        log(f"  Option A / Option B ratio: {ratio_A_vs_B:.3f} "
            f"(Ha et al. predicts 0.90-0.95)")
        log(f"  Option A train-val gap: {optA_agg['max_train_val_gap']} "
            f"(threshold: 0.7)")

        # Kill criteria for this L
        k891_pass = (optA_q >= 0.85) if n_layers == 16 else None
        k892_pass = (optB_q >= 0.85) if n_layers == 16 else None
        k893_kill = (optA_q < 0.50) if n_layers == 4 else None

        if k893_kill is not None and k893_kill:
            log(f"  *** K893 KILL TRIGGERED: Option A quality < 50% at L=4 ***")
            log(f"  *** Effective rank argument fails. Joint B-matrices > 64-dim. ***")
        if k891_pass is not None:
            log(f"  K891 (L=16 Option A >= 85%): {'PASS' if k891_pass else 'FAIL'}")
        if k892_pass is not None:
            log(f"  K892 (L=16 Option B >= 85%): {'PASS' if k892_pass else 'FAIL'}")

        sweep_results[f"L{n_layers}"] = {
            "n_layers": n_layers,
            "option_a": {
                "by_domain": {
                    name: {k: v for k, v in r.items() if k != "val_loss_trajectory"}
                    for name, r in optA_by_domain.items()
                },
                **optA_agg,
            },
            "option_b": {
                "by_domain": {
                    name: {k: v for k, v in r.items() if k != "val_loss_trajectory"}
                    for name, r in optB_by_domain.items()
                },
                **optB_agg,
            },
            "option_a_vs_b_ratio": round(ratio_A_vs_B, 4),
            "base_losses": base_losses,
            "sft_losses": {name: sft_results[name]["sft_loss"] for name in DOMAIN_NAMES},
            "sft_gaps": {name: sft_results[name]["gap"] for name in DOMAIN_NAMES},
            "k891_pass": k891_pass,
            "k892_pass": k892_pass,
            "k893_kill": k893_kill,
            # Top-level summaries for easy reading
            f"option_a_quality_ratio_L{n_layers}": round(optA_q, 4),
            f"option_b_quality_ratio_L{n_layers}": round(optB_q, 4),
            f"train_val_gap_option_a_L{n_layers}": optA_agg["max_train_val_gap"],
        }

        # EARLY EXIT if K893 triggered (Option A killed at L=4)
        if k893_kill:
            log(f"\n  K893 TRIGGERED at L={n_layers}. Stopping sweep early.")
            log(f"  Option A bottleneck confirmed. Skipping L>4.")
            break

        # Release base model (keeps memory bounded across L values)
        cleanup(base, A_matrices)

    return sweep_results


# ====================================================================
# KILL CRITERIA EVALUATION
# ====================================================================

def evaluate_kill_criteria(sweep_results: dict) -> dict:
    """Evaluate K891, K892, K893 from sweep results.

    K891: Option A quality_ratio >= 85% at L=16
    K892: Option B quality_ratio >= 85% at L=16
    K893 (KILL): Option A quality_ratio < 50% at L=4
    """
    log("\n=== Kill Criteria Evaluation ===")

    # Collect per-L quality ratios for the summary table
    summary = {}
    for lval in LAYER_VALUES:
        key = f"L{lval}"
        if key not in sweep_results:
            log(f"  L={lval}: NOT REACHED (early stop or smoke test)")
            summary[key] = None
            continue
        r = sweep_results[key]
        optA_q = r["option_a"]["median_quality"]
        optB_q = r["option_b"]["median_quality"]
        ratio = r["option_a_vs_b_ratio"]
        gap = r["option_a"].get("max_train_val_gap")
        log(f"  L={lval}: Option A={optA_q:.1%} Option B={optB_q:.1%} "
            f"A/B={ratio:.3f} gap={gap}")
        summary[key] = {
            "option_a_quality": optA_q,
            "option_b_quality": optB_q,
            "ratio_a_vs_b": ratio,
            "max_train_val_gap": gap,
        }

    # K893 at L=4
    k893_result = None
    if "L4" in sweep_results and sweep_results["L4"]["k893_kill"] is not None:
        k893_result = sweep_results["L4"]["k893_kill"]
        log(f"\n  K893 (KILL): Option A < 50% at L=4: {'KILL' if k893_result else 'NOT TRIGGERED'}")

    # K891, K892 at L=16
    k891_result = None
    k892_result = None
    if "L16" in sweep_results:
        k891_result = sweep_results["L16"]["k891_pass"]
        k892_result = sweep_results["L16"]["k892_pass"]
        if k891_result is not None:
            log(f"  K891: Option A >= 85% at L=16: {'PASS' if k891_result else 'FAIL'}")
        if k892_result is not None:
            log(f"  K892: Option B >= 85% at L=16: {'PASS' if k892_result else 'FAIL'}")

    # Overall interpretation
    if k893_result:
        outcome = "KILL_option_a_bottleneck_at_L4"
        interpretation = (
            "K893 TRIGGERED: Option A quality < 50% at L=4. "
            "Joint B-matrix effective rank exceeds d_M2P=64 for 4-layer adapters. "
            "Cross-layer structure argument (Ha et al.) fails at toy scale. "
            "Option A is not viable for multi-layer generation at this compression. "
            "Option B remains the correct strategy (proven recipe × L)."
        )
    elif k891_result and k892_result:
        outcome = "PASS_both_options_work_at_L16"
        interpretation = (
            "K891 PASS and K892 PASS: Both Option A and B achieve >= 85% at L=16. "
            "Option A is the preferred deployment: L× cheaper at inference, same quality. "
            "Cross-layer structure in toy transformer B-matrices confirmed (Ha et al. holds). "
            "Single M2P call scales to at least 16 layers. "
            "Next: test on real Qwen3-4B (36 layers)."
        )
    elif k891_result is None or k892_result is None:
        outcome = "INCOMPLETE_sweep_not_reached_L16"
        interpretation = (
            "Sweep did not reach L=16 (smoke test, early kill, or timeout). "
            "Check partial results at available L values."
        )
    elif not k891_result and k892_result:
        outcome = "PARTIAL_option_b_ok_option_a_bottleneck_at_L16"
        interpretation = (
            "K891 FAIL but K892 PASS: Option B works (proven recipe × 16), "
            "Option A fails at L=16. The 1024:1 compression ratio is too extreme. "
            "Joint B-matrix rank exceeds 64 for 16-layer toy transformer. "
            "Ha et al. finding holds for small L but breaks at L=16. "
            "Option B is the deployable strategy. Option A may work up to some L < 16."
        )
    elif k891_result and not k892_result:
        outcome = "UNEXPECTED_option_a_ok_option_b_fails"
        interpretation = (
            "Unexpected: Option A passes but Option B fails. "
            "This contradicts Theorem 2 (Option B proven by induction). "
            "Check for bugs in Option B implementation."
        )
    else:
        outcome = "BOTH_FAIL_AT_L16"
        interpretation = (
            "Both options fail at L=16. The toy transformer task may be too simple "
            "or too hard at L=16 for the current M2P configuration. "
            "Check per-L results for where quality starts degrading."
        )

    log(f"\n  Outcome: {outcome}")
    log(f"  Interpretation: {interpretation}")

    return {
        "k891_pass": k891_result,
        "k892_pass": k892_result,
        "k893_kill": k893_result,
        "outcome": outcome,
        "interpretation": interpretation,
        "per_L_summary": summary,
        # Flat keys for PAPER.md verification
        "option_a_quality_L2": (summary.get("L2") or {}).get("option_a_quality"),
        "option_a_quality_L4": (summary.get("L4") or {}).get("option_a_quality"),
        "option_a_quality_L8": (summary.get("L8") or {}).get("option_a_quality"),
        "option_a_quality_L16": (summary.get("L16") or {}).get("option_a_quality"),
        "option_b_quality_L2": (summary.get("L2") or {}).get("option_b_quality"),
        "option_b_quality_L4": (summary.get("L4") or {}).get("option_b_quality"),
        "option_b_quality_L8": (summary.get("L8") or {}).get("option_b_quality"),
        "option_b_quality_L16": (summary.get("L16") or {}).get("option_b_quality"),
        "train_val_gap_optA_L2": (summary.get("L2") or {}).get("max_train_val_gap"),
        "train_val_gap_optA_L4": (summary.get("L4") or {}).get("max_train_val_gap"),
        "train_val_gap_optA_L8": (summary.get("L8") or {}).get("max_train_val_gap"),
        "train_val_gap_optA_L16": (summary.get("L16") or {}).get("max_train_val_gap"),
    }


# ====================================================================
# MAIN ORCHESTRATOR
# ====================================================================

def main():
    t0 = time.time()
    log("M2P Layer Depth Scaling: Option A (single call) vs Option B (per-layer)")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log(f"")
    log(f"SWEEP: n_layers in {LAYER_VALUES}")
    log(f"Architecture: D_MODEL={D_MODEL}, D_M2P={D_M2P}, L_m2p={M2P_LAYERS}, "
        f"n={N_SAMPLES}, T={T_FIXED}")
    log(f"")
    log(f"MATH.md Theorem 1: n_train>=T is n_layers-independent (Ghadimi-Lan).")
    log(f"  At n={N_SAMPLES}: n_train={int(0.8*N_SAMPLES)}, T/n_train="
        f"{T_FIXED/int(0.8*N_SAMPLES):.3f} < 1 epoch. Holds for all L.")
    log(f"MATH.md Theorem 2: Option B correct by induction from Finding #362.")
    log(f"  Each M2P call = proven recipe. Quality >= 85% guaranteed.")
    log(f"MATH.md Theorem 3: Option A works IFF rank([B_1*,...,B_L*]) <= d_M2P=64.")
    log(f"  Ha et al. (arXiv:1609.09106): hypernetworks achieve 90-95% of per-layer.")
    log(f"")
    log(f"Option A compression analogy:")
    for lval in LAYER_VALUES:
        comp = lval * LORA_RANK * MODULE_OUT_DIMS_BASE[4] / D_M2P
        log(f"  L={lval}: fc1 compression {comp:.0f}:1 "
            f"{'(= Finding #359, 97.6%)' if lval == 2 else ''}"
            f"{'(= Finding #361, 101.0%)' if lval == 4 else ''}"
            f"{'(= Finding #362, 99.6%)' if lval == 8 else ''}"
            f"{'(FRONTIER)' if lval == 16 else ''}")
    log(f"")
    log(f"KILL CRITERIA:")
    log(f"  K891: Option A quality >= 85% at L=16 -> PASS")
    log(f"  K892: Option B quality >= 85% at L=16 -> PASS")
    log(f"  K893 (KILL): Option A quality < 50% at L=4 -> KILL (bottleneck)")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # -- Generate shared data (same data for all L values) --
    domain_data = phase_generate_data(rng, n_per_domain=N_SAMPLES)
    log_memory("after data")

    # -- Depth sweep: L=2, 4, 8, 16 --
    sweep_results = phase_depth_sweep(domain_data)
    log_memory("after sweep")

    # -- Kill criteria evaluation --
    kill_criteria = evaluate_kill_criteria(sweep_results)

    # -- Results assembly --
    total_time = round(time.time() - t0, 1)

    # Flat keys for easy access by PAPER.md / experiment complete
    flat_results = {}
    for lval in LAYER_VALUES:
        key = f"L{lval}"
        if key in sweep_results:
            flat_results[f"option_a_quality_ratio_L{lval}"] = sweep_results[key][f"option_a_quality_ratio_L{lval}"]
            flat_results[f"option_b_quality_ratio_L{lval}"] = sweep_results[key][f"option_b_quality_ratio_L{lval}"]
            flat_results[f"train_val_gap_option_a_L{lval}"] = sweep_results[key][f"train_val_gap_option_a_L{lval}"]

    results = {
        "experiment": "exp_m2p_layer_depth",
        "total_time_s": total_time,
        "smoke_test": SMOKE_TEST,
        # Architecture
        "d_model": D_MODEL,
        "d_m2p": D_M2P,
        "m2p_layers": M2P_LAYERS,
        "lora_rank": LORA_RANK,
        "n_samples": N_SAMPLES,
        "t_fixed": T_FIXED,
        "layer_values": LAYER_VALUES,
        "n_domains": N_DOMAINS,
        "domain_names": DOMAIN_NAMES,
        "parity_guard_threshold": PARITY_GUARD_THRESHOLD,
        "gl_threshold": GL_THRESHOLD,
        # Flat output metrics (required by experiment spec)
        **flat_results,
        # Full sweep data
        "sweep": sweep_results,
        # Kill criteria
        "kill_criteria": kill_criteria,
        # Prior baselines
        "prior_baselines": {
            "finding_359_d256_L2": 0.976,
            "finding_361_d512_L2": 1.010,
            "finding_362_d1024_L2": 0.996,
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to: {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")

    # Final summary
    log(f"\n{'='*70}")
    log(f"FINAL RESULTS SUMMARY")
    log(f"{'='*70}")
    for lval in LAYER_VALUES:
        key = f"L{lval}"
        if key in sweep_results:
            optA_q = flat_results.get(f"option_a_quality_ratio_L{lval}", "N/A")
            optB_q = flat_results.get(f"option_b_quality_ratio_L{lval}", "N/A")
            gap = flat_results.get(f"train_val_gap_option_a_L{lval}", "N/A")
            log(f"  L={lval}: Option A={optA_q:.1%} Option B={optB_q:.1%} gap={gap}"
                if isinstance(optA_q, float) else
                f"  L={lval}: NOT REACHED")
    log(f"")
    log(f"Kill criteria:")
    log(f"  K891 (Option A >= 85% at L=16): {kill_criteria.get('k891_pass')}")
    log(f"  K892 (Option B >= 85% at L=16): {kill_criteria.get('k892_pass')}")
    log(f"  K893 KILL (Option A < 50% at L=4): {kill_criteria.get('k893_kill')}")
    log(f"  Outcome: {kill_criteria.get('outcome')}")


if __name__ == "__main__":
    main()
