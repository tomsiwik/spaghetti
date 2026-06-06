#!/usr/bin/env python3
"""M2P Macro Quality: Validate n_train>=T Guarantee at 2x d_model (d=512).

TYPE: guided-exploration (Type 2)
MATH: experiments/models/m2p_macro_quality/MATH.md

PRIOR FINDING (Finding #359, exp_m2p_data_scale):
  At d_model=256, n=2000 + GL early stopping achieves 97.6% of SFT quality.
  n_train >= T structural guarantee (Ghadimi-Lan i.i.d. condition) + GL stopping
  is the validated recipe. Early stopping alone contributes +7.6pp.

QUESTION:
  Does the SAME recipe (L=2, d_M2P=64, n=2000, GL) achieve >= 85% of SFT when
  the ONLY change is d_model: 256 -> 512?

ROOT CAUSE (MATH.md Theorem 1):
  n_train >= T is d_model-independent. At n=2000 (n_train=1600), T=1000:
  T/n_train = 0.625 epochs. No sample visited twice. Ghadimi-Lan i.i.d. condition
  satisfied regardless of d_model.

MATHEMATICAL FRAMEWORK:
  Theorem 1 (MATH.md): n_train >= T structural guarantee is d_model-independent.
    At n=2000: T/n_train = 0.625 < 1 (no cycling). QED.
  Theorem 2 (MATH.md): GL early stopping bounds val_loss <= 1.05 * best_val_loss.
  Bartlett et al. (arXiv:1906.11300): at d=512, d_eff ~ 2048, n/d_eff ~ 1.
    Quality degradation expected but not catastrophic (85% floor vs 97.6% at micro).
  K883 threshold: 2 * 0.337 nats (micro measured gap * 2 for harder targets).

KILL CRITERIA:
  K882 (#885): quality_ratio(d=512, n=2000, T=1000) >= 85% of SFT (PASS = capacity ok)
  K883 (#886): train-val gap at n=2000 < 0.7 nats (PASS = overfitting controlled)
  K884 (#887, KILL): quality_ratio(d=512) < 60% (FAIL = M2P capacity insufficient)

ARCHITECTURE CHANGE (THE ONLY CHANGE from Finding #359):
  D_MODEL = 512 (was 256)
  N_HEADS = 8  (was 4; maintains d_head = D_MODEL/N_HEADS = 64, same as micro)
  MODULE_OUT_DIMS scaled with D_MODEL (wq/wk/wv/wo: 512, fc1: 2048)

EVERYTHING ELSE FIXED (same as exp_m2p_data_scale):
  N_LAYERS=2, VOCAB_SIZE=128, BLOCK_SIZE=48
  LORA_RANK=4, LORA_SCALE=2.0
  M2P_LAYERS=2, D_M2P=64, N_MEMORY=32 (proven sufficient at d=256 by Findings #355, #357)
  T_FIXED=1000, GL_THRESHOLD=5.0, PATIENCE=5, EARLY_STOP_INTERVAL=50

EXPERIMENT VARIABLE: N_SAMPLES sweep {1000, 2000} (skip 500 -- already proven bad)
  n=1000: n_train=800, T/n_train=1.25 epochs -- REFERENCE POINT (partial cycling)
  n=2000: n_train=1600, T/n_train=0.625 epoch -- PRIMARY TEST (structural guarantee)
  NOTE: n=500 already proven overfitting-inducing in Finding #358. Not retested.

N_DOMAINS = 3 (arithmetic, sort, reverse) -- reduced from 5 for runtime
DOMAINS: Parity excluded (parity guard), repeat dropped for 2-hour runtime budget.

BASELINE REFERENCE: micro quality = 97.6% at n=2000, d=256 (Finding #359)
FLOOR PREDICTION: >= 85% at n=2000, d=512 (K882; Bartlett d_eff scaling)
KILL PREDICTION: < 60% -> capacity bottleneck (K884)
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

# Memory safety (CODING_GUIDELINES 2)
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ====================================================================
# ARCHITECTURE CONSTANTS
# THE ONLY CHANGE from exp_m2p_data_scale: D_MODEL=512, N_HEADS=8
# ====================================================================
D_MODEL = 512   # <-- THE ONLY CHANGE (was 256 in micro, Finding #359)
N_LAYERS = 2
N_HEADS = 8     # scaled with D_MODEL: d_head = D_MODEL/N_HEADS = 512/8 = 64
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0

# M2P architecture FIXED (proven sufficient at d=256, Findings #355, #357)
M2P_LAYERS = 2   # Finding #357: L=2 saturates
D_M2P = 64       # Finding #355: width not bottleneck
N_MEMORY = 32

# DOMAINS: 3 (reduced from 5 for 2-hour runtime; parity excluded by guard, repeat dropped)
N_DOMAINS = 3
DOMAIN_NAMES = ["arithmetic", "sort", "reverse"]

# Module names and output dims (SCALED with D_MODEL=512)
MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
# At D_MODEL=512: [512, 512, 512, 512, 2048]
# At D_MODEL=256: [256, 256, 256, 256, 1024] (micro baseline)
N_MODULES = len(MODULE_NAMES)

# ====================================================================
# EXPERIMENT VARIABLE: N_SAMPLES sweep
# n=1000 (n_train=800): T/n_train=1.25 epochs -- reference (partial cycling)
# n=2000 (n_train=1600): T/n_train=0.625 epoch -- structural guarantee (K882/K883)
# n=500 SKIPPED: already proven overfitting-inducing at micro scale (Finding #358/359)
# ====================================================================
SAMPLE_VALUES = [1000, 2000]

# Fixed training steps (T <= n_train is satisfied for n=2000 at T=1000)
# Same as micro (exp_m2p_data_scale): T_FIXED=1000
T_FIXED = 1000

# Shared training constants (same as micro)
BASE_STEPS = 1200 if not SMOKE_TEST else 30
SFT_STEPS  = T_FIXED  # match M2P training budget (1000 steps)
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

# Smoke test overrides
if SMOKE_TEST:
    SAMPLE_VALUES = [80, 160]
    T_FIXED = 10
    SFT_STEPS = T_FIXED  # keep consistent

# Parity guard (unchanged from prior experiments)
PARITY_GUARD_THRESHOLD = 0.05

# Early stopping parameters (Prechelt 1998 GL criterion -- identical to micro)
EARLY_STOP_INTERVAL = 50   # check val loss every N steps
GL_THRESHOLD = 5.0          # stop when GL(t) > 5.0 for PATIENCE checks
PATIENCE = 5                # consecutive GL > threshold checks before stopping


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
# DATA GENERATION (same logic as micro, 3 domains only)
# ====================================================================

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    """Generate synthetic task data. Domain IDs match DOMAIN_NAMES order."""
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
# TOY GPT (d=512, 8 heads) -- scaled from micro's d=256, 4 heads
# d_head = 64 in both cases. Architecture structure identical.
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
    """Toy GPT: d=512, L=2, 8 heads, vocab=128. d_head=64 (same as micro)."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.wpe = nn.Embedding(BLOCK_SIZE + 1, D_MODEL)
        self.blocks = [Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]
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
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states


# ====================================================================
# GRASSMANNIAN A-MATRICES (identical to micro)
# ====================================================================

def generate_grassmannian_A(n_domains, n_layers, n_modules, d, rank, seed=42):
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


def verify_grassmannian_orthogonality(A_matrices, n_domains, n_layers, n_modules):
    cos_values = []
    for li in range(n_layers):
        for mi in range(n_modules):
            for di in range(n_domains):
                for dj in range(di + 1, n_domains):
                    ai = A_matrices[(di, li, mi)].reshape(-1)
                    aj = A_matrices[(dj, li, mi)].reshape(-1)
                    cos = mx.abs(
                        mx.sum(ai * aj) /
                        (mx.linalg.norm(ai) * mx.linalg.norm(aj) + 1e-12)
                    ).item()
                    cos_values.append(cos)
    return {
        "mean_cos": float(np.mean(cos_values)),
        "max_cos": float(np.max(cos_values)),
        "n_pairs": len(cos_values),
    }


# ====================================================================
# LORA FORWARD PASS (identical structure to micro; scales with D_MODEL)
# ====================================================================

def lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices):
    """Forward pass with Grassmannian LoRA applied for given domain."""
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
# SFT TRAINING INFRASTRUCTURE
# ====================================================================

class BMatrices(nn.Module):
    def __init__(self):
        super().__init__()
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                d_out = MODULE_OUT_DIMS[mi]
                setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

    def as_dict(self):
        return {
            (li, mi): getattr(self, f"B_{li}_{mi}")
            for li in range(N_LAYERS) for mi in range(N_MODULES)
        }


def sft_loss_fn(b_container, base, tokens, A_matrices, domain_id):
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ====================================================================
# M2P TRANSFORMER (d_M2P=64 -- identical to micro, d_model-independent)
# ====================================================================

class M2PAttention(nn.Module):
    """M2P attention with causal masking (same as micro -- d_model independent)."""
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


class M2PTransformer(nn.Module):
    """M2P transformer: L=2, d_M2P=64, causal attention.
    Architecture FIXED (Finding #355 width, Finding #357 depth).
    CRITICAL: This architecture is the SAME as micro -- we test whether it
    scales to d_model=512 without modification (the core question of this experiment).
    Input projection: D_MODEL (512) -> D_M2P (64) via input_proj.
    Output heads: D_M2P (64) -> N_LAYERS * LORA_RANK * d_out (now 2x larger for d=512).
    """
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.m2p_layers = m2p_layers
        # Input projection: d_model -> d_m2p (handles D_MODEL=512 -> D_M2P=64)
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(m2p_layers)]
        self.norm_f = RMSNorm(d_m2p)
        # Output heads: d_m2p -> N_LAYERS * LORA_RANK * d_out
        # At d=512: fc1 head is d_m2p=64 -> 2 * 4 * 2048 = 16,384 params
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            total_out = N_LAYERS * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list):
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
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pooled_memory)
            out = out.reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS):
                B_matrices[(li, mi)] = out[li]
        return B_matrices


def m2p_ntp_loss(m2p, base, A_matrices, domain_id, tokens):
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ====================================================================
# EVALUATION HELPERS
# ====================================================================

def eval_ntp_loss(base, batches, A_matrices=None, domain_id=None, B_matrices=None):
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


def load_B_matrices(path: str) -> dict:
    data = np.load(path)
    B_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(N_MODULES):
            key = f"{li}_{mi}"
            B_matrices[(li, mi)] = mx.array(data[key])
    return B_matrices


# ====================================================================
# PHASE FUNCTIONS (each self-contained per CODING_GUIDELINES 1)
# ====================================================================

def phase_generate_data(rng: np.random.RandomState, n_per_domain: int) -> dict:
    """Generate train/val data for N_DOMAINS=3 domains with specified sample count."""
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
            "n_train": len(make_batches(train_texts)),
            "n_val": len(make_batches(val_texts)),
        }
        log(f"  {name}: {domain_data[name]['n_train']} train, "
            f"{domain_data[name]['n_val']} val")
    return domain_data


def phase_pretrain_base(domain_data: dict) -> tuple:
    """Pre-train ToyGPT (d=512) on all domains.

    Architecture: d=512, 8 heads, L=2. d_head=64 (same as micro).
    Uses n_per_domain=max(SAMPLE_VALUES) data to avoid data-size confound.
    """
    log("\n=== Phase: Pre-train Base Model (d=512, 8 heads) ===")
    log(f"  ToyGPT: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, "
        f"d_head={D_MODEL//N_HEADS}, N_LAYERS={N_LAYERS}")
    mx.random.seed(SEED)

    base = ToyGPT()
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


def phase_grassmannian(base: ToyGPT) -> tuple:
    """Generate and verify Grassmannian A-matrices.

    At d=512: total_rank = N_DOMAINS * LORA_RANK = 3 * 4 = 12 << 512.
    Grassmannian guarantee: max|cos| between domain subspaces is machine-epsilon.
    """
    log("\n=== Phase: Grassmannian A-matrices ===")
    log(f"  D_MODEL={D_MODEL}, N_DOMAINS={N_DOMAINS}, LORA_RANK={LORA_RANK}")
    log(f"  total_rank = {N_DOMAINS * LORA_RANK} (must be << {D_MODEL})")
    A_matrices = generate_grassmannian_A(
        N_DOMAINS, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK, seed=SEED
    )
    ortho = verify_grassmannian_orthogonality(
        A_matrices, N_DOMAINS, N_LAYERS, N_MODULES
    )
    log(f"  Orthogonality: mean|cos|={ortho['mean_cos']:.6f}, "
        f"max|cos|={ortho['max_cos']:.6f} ({ortho['n_pairs']} pairs)")
    assert ortho["max_cos"] < 1e-5, \
        f"Grassmannian guarantee failed: max|cos|={ortho['max_cos']}"
    return A_matrices, ortho


def phase_sft_domain(domain_name, domain_id, domain_data, base,
                      A_matrices, base_loss) -> dict:
    """Train SFT LoRA adapter for one domain (always fresh).

    Uses domain_data["train"] (max-scale data) for best SFT quality reference.
    SFT quality defines the 100% reference for quality_ratio computation.
    """
    log(f"  SFT {domain_name} (domain {domain_id})...")

    local_path = EXPERIMENT_DIR / "adapters" / f"sft_{domain_name}.npz"
    local_path.parent.mkdir(exist_ok=True)

    if local_path.exists():
        log(f"    Reusing local SFT adapter (same experiment run)")
        b_container = BMatrices()
        data = np.load(str(local_path))
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                setattr(b_container, f"B_{li}_{mi}",
                        mx.array(data[f"{li}_{mi}"]))
        mx.eval(b_container.parameters())
    else:
        log(f"    Training SFT adapter from scratch")
        b_container = BMatrices()
        mx.eval(b_container.parameters())
        optimizer = opt.Adam(learning_rate=SFT_LR)

        def _loss(b_cont, tokens):
            return sft_loss_fn(b_cont, base, tokens[None, :], A_matrices, domain_id)

        grad_fn = nn.value_and_grad(b_container, _loss)
        train_batches = domain_data["train"]

        gc.disable()
        for step in range(SFT_STEPS):
            tokens = train_batches[step % len(train_batches)]
            loss, grads = grad_fn(b_container, tokens)
            optimizer.update(b_container, grads)
            mx.eval(b_container.parameters(), optimizer.state, loss)
        gc.enable()
        cleanup(optimizer)

        np_dict = {f"{li}_{mi}": np.array(getattr(b_container, f"B_{li}_{mi}"))
                   for li in range(N_LAYERS) for mi in range(N_MODULES)}
        np.savez(str(local_path), **np_dict)

    B_matrices = b_container.as_dict()
    sft_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)
    log(f"    SFT loss={sft_loss:.4f} base={base_loss:.4f}")

    cleanup(b_container)
    return {"sft_loss": round(sft_loss, 4), "save_path": str(local_path)}


def phase_sft_all_domains(domain_data, base, A_matrices, base_losses) -> dict:
    log("\n=== Phase: SFT Baselines (fresh training) ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            name, di, domain_data[name], base, A_matrices, base_losses[name]
        )
        sft_results[name] = result
    return sft_results


def phase_train_m2p_with_early_stopping(
    n_samples: int,
    domain_name: str,
    domain_id: int,
    domain_data: dict,
    base: ToyGPT,
    A_matrices: dict,
    base_loss: float,
    sft_loss: float,
    eval_val_batches: list = None,
) -> dict:
    """Train ONE M2P at fixed T=T_FIXED for ONE domain.

    Key structural guarantee (MATH.md Theorem 1):
    n_train >= T ensures i.i.d. gradient sampling (Ghadimi-Lan condition).
    At n=2000 (n_train=1600), T=1000: epochs = 0.625 < 1. No cycling.
    This is d_model-independent -- same guarantee applies at d=512.

    GL early stopping (Prechelt 1998, MATH.md Theorem 2):
    Stop when GL(t) = 100*(val_loss(t)/best_val_loss - 1) > GL_THRESHOLD=5.0
    for PATIENCE=5 consecutive checks (every EARLY_STOP_INTERVAL=50 steps).
    Identical to micro -- d_model-independent criterion.

    CAPACITY TEST: The M2P generates B-matrices at d=512 (2x larger than d=256).
    Output head fc1: D_M2P=64 -> N_LAYERS*LORA_RANK*4*D_MODEL = 2*4*2048 = 16384.
    Whether D_M2P=64 is sufficient for this task is the core question (K882/K884).

    Returns quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)
    """
    n_train = len(domain_data["train"])
    n_epochs_expected = T_FIXED / max(n_train, 1)

    adapter_dir = EXPERIMENT_DIR / f"adapters_n{n_samples}"
    adapter_dir.mkdir(exist_ok=True)
    save_path = adapter_dir / f"m2p_{domain_name}.npz"

    log(f"    [n={n_samples}, d={D_MODEL}] Training M2P for {domain_name} "
        f"(T={T_FIXED}, n_train={n_train}, "
        f"epochs={n_epochs_expected:.2f})...")
    mx.random.seed(SEED)

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS)
    mx.eval(m2p.parameters())

    param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"      M2P params: {param_count:,} "
        f"(L={M2P_LAYERS}, d_m2p={D_M2P}, d_base={D_MODEL})")

    # Verify output head dimensions
    fc1_out_dim = N_LAYERS * LORA_RANK * MODULE_OUT_DIMS[4]
    log(f"      Output head fc1: {D_M2P} -> {fc1_out_dim} "
        f"(micro was {N_LAYERS * LORA_RANK * (4 * 256)})")

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id,
                             tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]
    val_batches = domain_data["val"]

    # Early stopping state (identical to micro)
    best_val_loss = float("inf")
    best_step = 0
    consecutive_gl_exceeded = 0
    early_stop_triggered = False
    stopping_step = T_FIXED

    # Track trajectories
    loss_trajectory = []
    val_loss_trajectory = []
    final_train_loss = None
    final_val_loss = None

    gc.disable()
    for step in range(T_FIXED):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)

        train_loss_val = loss.item()
        final_train_loss = train_loss_val

        # Log at 25%, 50%, 75%, 100%
        if (step + 1) % max(1, T_FIXED // 4) == 0:
            loss_trajectory.append({
                "step": step + 1,
                "train_loss": round(train_loss_val, 4),
            })

        # Early stopping: evaluate val loss every EARLY_STOP_INTERVAL steps
        if (step + 1) % EARLY_STOP_INTERVAL == 0 and not early_stop_triggered:
            gc.enable()
            context_tokens = train_batches[0][None, :]
            hidden_states = base.get_hidden_states(context_tokens)
            B_now = m2p(hidden_states)
            mx.eval(*[B_now[(li, mi)] for li in range(N_LAYERS)
                      for mi in range(N_MODULES)])
            val_loss_now = eval_ntp_loss(base, val_batches, A_matrices,
                                          domain_id, B_now)
            del B_now
            gc.disable()

            val_loss_trajectory.append({
                "step": step + 1,
                "val_loss": round(val_loss_now, 4),
            })

            # GL criterion (Prechelt 1998, MATH.md Theorem 2)
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
                            f"GL={gl:.2f} > {GL_THRESHOLD} "
                            f"for {PATIENCE} consecutive checks. "
                            f"best_val_loss={best_val_loss:.4f} at step {best_step}")
                        break
                else:
                    consecutive_gl_exceeded = 0

    gc.enable()
    cleanup(optimizer)

    # Generate B-matrices from first training context and save
    context_tokens = train_batches[0][None, :]
    hidden_states = base.get_hidden_states(context_tokens)
    B_matrices = m2p(hidden_states)
    mx.eval(*[B_matrices[(li, mi)] for li in range(N_LAYERS)
              for mi in range(N_MODULES)])

    np_dict = {f"{li}_{mi}": np.array(B_matrices[(li, mi)])
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(save_path), **np_dict)

    cleanup(m2p)

    # Evaluate: load saved B-matrices and measure NTP loss.
    # Use eval_val_batches (domain_data_max val set, same as SFT) for the FINAL
    # quality evaluation so quality_ratio uses consistent validation sets.
    # val_batches (domain_data_by_n val set) is used only for early stopping above.
    final_eval_batches = eval_val_batches if eval_val_batches is not None else val_batches
    B_matrices_loaded = load_B_matrices(str(save_path))
    m2p_val_loss = eval_ntp_loss(base, final_eval_batches, A_matrices,
                                  domain_id, B_matrices_loaded)
    final_val_loss = m2p_val_loss

    # Train-val gap at stopping point (K883)
    train_val_gap = abs(final_train_loss - final_val_loss) if final_train_loss else None

    # Parity guard (same as micro)
    quality_ratio = 0.0
    excluded_parity = False
    gap = base_loss - sft_loss
    if gap > PARITY_GUARD_THRESHOLD:
        quality_ratio = (base_loss - m2p_val_loss) / gap
    else:
        excluded_parity = True
        log(f"    [n={n_samples}] {domain_name}: EXCLUDED "
            f"(gap={gap:.4f} < {PARITY_GUARD_THRESHOLD})")

    if not excluded_parity:
        log(f"    [n={n_samples}, d={D_MODEL}] {domain_name}: "
            f"val_loss={m2p_val_loss:.4f} "
            f"train_loss={final_train_loss:.4f} "
            f"train_val_gap={train_val_gap:.4f} "
            f"SFT={sft_loss:.4f} base={base_loss:.4f} "
            f"quality={quality_ratio:.1%} "
            f"epochs={n_epochs_expected:.2f} "
            f"stopped_at={stopping_step}")

    cleanup(B_matrices_loaded)
    return {
        "m2p_val_loss": round(m2p_val_loss, 4),
        "final_train_loss": round(final_train_loss, 4) if final_train_loss else None,
        "train_val_gap": round(train_val_gap, 4) if train_val_gap else None,
        "quality_ratio": round(quality_ratio, 4),
        "excluded_parity": excluded_parity,
        "gap": round(gap, 4),
        "n_train": n_train,
        "n_epochs_actual": round(T_FIXED / max(n_train, 1), 3),
        "early_stop_triggered": early_stop_triggered,
        "stopping_step": stopping_step,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "best_step": best_step,
        "loss_trajectory": loss_trajectory,
        "val_loss_trajectory": val_loss_trajectory,
    }


def phase_sweep_n_samples(domain_data_by_n, base, A_matrices,
                           base_losses, sft_results,
                           domain_data_max=None) -> dict:
    """Sweep N_SAMPLES in {1000, 2000} with fixed T=T_FIXED at d_model=512.

    MATH.md Theorem 1: n_train >= T is d_model-independent.
    n=1000 at T=1000: n_train=800, T/n_train=1.25 epochs -- PARTIAL CYCLING (reference)
    n=2000 at T=1000: n_train=1600, T/n_train=0.625 epoch -- STRUCTURAL GUARANTEE (primary)

    Micro baseline (Finding #359, d=256):
      quality(n=2000, d=256, T=1000) = 97.6%
    K882 floor (d=512): quality(n=2000, d=512, T=1000) >= 85%
    K883 gap (d=512): max train-val gap < 0.7 nats (2x micro measured 0.337)
    K884 kill (d=512): quality < 60% -> capacity bottleneck
    """
    log(f"\n=== Phase: N_SAMPLES Sweep at d={D_MODEL} ===")
    log(f"  Sweeping N_SAMPLES in {SAMPLE_VALUES}")
    log(f"  Fixed T={T_FIXED} steps, d_model={D_MODEL}")
    log(f"  n_train >= T condition (Theorem 1, d_model-independent):")
    for n in SAMPLE_VALUES:
        n_train = int(0.8 * n)
        epochs = T_FIXED / max(n_train, 1)
        status = "SATISFIES" if n_train >= T_FIXED else "REFERENCE (partial cycling)"
        log(f"    n={n}: n_train={n_train}, T/n_train={epochs:.2f} -- {status}")
    log(f"  Early stopping: GL threshold={GL_THRESHOLD}, "
        f"patience={PATIENCE} checks, interval={EARLY_STOP_INTERVAL} steps")
    log(f"  Micro baseline (Finding #359, d=256): quality=97.6% at n=2000")
    log(f"  K882 floor: >= 85% at n=2000, d=512")
    log(f"  K884 kill:  < 60% at n=2000, d=512")

    sweep_results = {}

    for n_samples in SAMPLE_VALUES:
        log(f"\n  --- n={n_samples} (T={T_FIXED}, d={D_MODEL}) ---")
        domain_data = domain_data_by_n[n_samples]
        domain_qualities = {}
        domain_val_losses = {}
        domain_train_losses = {}
        domain_train_val_gaps = {}
        domain_epochs = {}
        domain_early_stopped = {}
        excluded_domains = []

        for di, name in enumerate(DOMAIN_NAMES):
            # Use domain_data_max val set for final quality evaluation so
            # quality_ratio uses the same val set as SFT (blocking fix 1).
            eval_val = domain_data_max[name]["val"] if domain_data_max is not None else None
            result = phase_train_m2p_with_early_stopping(
                n_samples=n_samples,
                domain_name=name,
                domain_id=di,
                domain_data=domain_data[name],
                base=base,
                A_matrices=A_matrices,
                base_loss=base_losses[name],
                sft_loss=sft_results[name]["sft_loss"],
                eval_val_batches=eval_val,
            )
            domain_qualities[name] = result["quality_ratio"]
            domain_val_losses[name] = result["m2p_val_loss"]
            domain_train_losses[name] = result["final_train_loss"]
            domain_train_val_gaps[name] = result["train_val_gap"]
            domain_epochs[name] = result["n_epochs_actual"]
            domain_early_stopped[name] = result["early_stop_triggered"]
            if result["excluded_parity"]:
                excluded_domains.append(name)

        if excluded_domains:
            log(f"  Excluded domains (parity guard): {excluded_domains}")

        valid_qualities = [domain_qualities[n] for n in DOMAIN_NAMES
                           if n not in excluded_domains]
        valid_gaps = [domain_train_val_gaps[n] for n in DOMAIN_NAMES
                      if n not in excluded_domains
                      and domain_train_val_gaps[n] is not None]

        median_q = float(np.median(valid_qualities)) if valid_qualities else 0.0
        mean_q = float(np.mean(valid_qualities)) if valid_qualities else 0.0
        max_train_val_gap = float(np.max(valid_gaps)) if valid_gaps else None
        mean_train_val_gap = float(np.mean(valid_gaps)) if valid_gaps else None

        log(f"  n={n_samples}: median quality={median_q:.1%}, mean={mean_q:.1%}")
        log(f"  Per-domain: {dict((k, f'{v:.1%}') for k, v in domain_qualities.items() if k not in excluded_domains)}")
        log(f"  Train-val gaps: {dict((k, f'{v:.4f}') for k, v in domain_train_val_gaps.items() if k not in excluded_domains)}")
        if max_train_val_gap is not None:
            log(f"  Max train-val gap: {max_train_val_gap:.4f} (K883 threshold: 0.7)")
        n_early_stopped = sum(1 for n in DOMAIN_NAMES
                              if n not in excluded_domains and domain_early_stopped[n])
        log(f"  Early stops triggered: {n_early_stopped}/{len(valid_qualities)} valid domains")

        sweep_results[f"n{n_samples}"] = {
            "n_samples": n_samples,
            "t_fixed": T_FIXED,
            "d_model": D_MODEL,
            "m2p_layers": M2P_LAYERS,
            "d_m2p": D_M2P,
            "domain_quality": domain_qualities,
            "domain_val_loss": domain_val_losses,
            "domain_train_loss": domain_train_losses,
            "domain_train_val_gap": domain_train_val_gaps,
            "domain_epochs": domain_epochs,
            "domain_early_stopped": domain_early_stopped,
            "median_quality": round(median_q, 4),
            "mean_quality": round(mean_q, 4),
            "max_train_val_gap": round(max_train_val_gap, 4) if max_train_val_gap else None,
            "mean_train_val_gap": round(mean_train_val_gap, 4) if mean_train_val_gap else None,
            "n_early_stopped": n_early_stopped,
            "excluded_domains": excluded_domains,
            "n_valid_domains": len(valid_qualities),
        }

        log_memory(f"after n={n_samples} sweep")

    return sweep_results


# ====================================================================
# KILL CRITERIA EVALUATION
# ====================================================================

def evaluate_kill_criteria(sweep_results: dict) -> dict:
    """Evaluate K882, K883, K884 from sweep results.

    K882 (#885): quality_ratio(d=512, n=2000, T=1000) >= 85% of SFT
         Source: Conservative capacity floor. Micro (d=256) achieved 97.6%.
         2x d_model -> 2x d_eff (Bartlett) -> some degradation expected.
         85% allows ~12pp degradation for harder B-matrix targets.

    K883 (#886): train-val gap at n=2000 < 0.7 nats
         Source: MATH.md K883 derivation: 2 * 0.337 nats (micro measured).
         0.7 nats is conservative (allows 2x micro measured gap).

    K884 (#887, KILL): quality_ratio(d=512) < 60%
         Source: Definitive capacity bottleneck. d_M2P=64 cannot represent
         B-matrices for 512-dim modules. Architecture search must reopen.
    """
    n_high_key = f"n{SAMPLE_VALUES[-1]}"    # n2000 (primary)
    n_low_key = f"n{SAMPLE_VALUES[0]}"       # n1000 (reference)

    r_high = sweep_results[n_high_key]
    r_low = sweep_results[n_low_key]

    q_high = r_high["median_quality"]
    q_low = r_low["median_quality"]
    gap_high = r_high["max_train_val_gap"]
    gap_low = r_low["max_train_val_gap"]

    # K882: quality >= 85% at n=2000 (PASS = capacity adequate)
    K882_THRESHOLD = 0.85
    k882_pass = (q_high >= K882_THRESHOLD)

    # K883: train-val gap < 0.7 nats at n=2000 (PASS = overfitting controlled)
    K883_THRESHOLD = 0.7
    k883_pass = (gap_high is not None and gap_high < K883_THRESHOLD)

    # K884: quality < 60% at n=2000 (KILL = capacity bottleneck)
    K884_KILL_THRESHOLD = 0.60
    k884_kill = (q_high < K884_KILL_THRESHOLD)

    log("\n=== Kill Criteria Evaluation ===")
    log(f"  Micro baseline (Finding #359, d=256): 97.6% quality at n=2000")
    log(f"  This experiment (d={D_MODEL}):")
    log(f"    n={SAMPLE_VALUES[0]} (reference, partial cycling): "
        f"quality={q_low:.1%}, max_train_val_gap={gap_low}")
    log(f"    n={SAMPLE_VALUES[1]} (structural guarantee): "
        f"quality={q_high:.1%}, max_train_val_gap={gap_high}")
    log("")
    log(f"  K882 (#885): quality(n=2000, d=512) >= {K882_THRESHOLD:.0%}: "
        f"{q_high:.1%} -> {'PASS' if k882_pass else 'FAIL'}")
    log(f"  K883 (#886): max train-val gap(n=2000) < {K883_THRESHOLD}: "
        f"{gap_high} -> {'PASS' if k883_pass else 'FAIL'}")
    log(f"  K884 (#887, KILL): quality(n=2000, d=512) < {K884_KILL_THRESHOLD:.0%}: "
        f"{q_high:.1%} -> {'KILL' if k884_kill else 'PASS (not triggered)'}")

    # Interpret results
    if k884_kill:
        outcome = "KILL_capacity_bottleneck"
        interpretation = (
            f"K884 TRIGGERED: quality={q_high:.1%} < {K884_KILL_THRESHOLD:.0%}. "
            f"M2P at d_M2P=64, L=2 is capacity-insufficient for d_model=512 targets. "
            f"Architecture search must reopen: increase d_M2P or add output head layers."
        )
    elif k882_pass and k883_pass:
        outcome = "PASS_recipe_scales"
        interpretation = (
            f"K882 PASS ({q_high:.1%} >= {K882_THRESHOLD:.0%}), "
            f"K883 PASS (gap={gap_high:.3f} < {K883_THRESHOLD}). "
            f"The n_train>=T recipe transfers to d_model=512. "
            f"Quality degradation {0.976 - q_high:.1%} vs micro ({0.976:.1%}) "
            f"is within the Bartlett d_eff scaling prediction (12pp allowed). "
            f"Next: test at n=5000 or scale M2P architecture for full 97%+ quality."
        )
    elif k882_pass and not k883_pass:
        outcome = "PARTIAL_quality_ok_overfit_detected"
        interpretation = (
            f"K882 PASS ({q_high:.1%} >= {K882_THRESHOLD:.0%}) but "
            f"K883 FAIL (gap={gap_high:.3f} >= {K883_THRESHOLD}). "
            f"Quality is acceptable but overfitting is more severe at d=512. "
            f"Consider increasing n or reducing GL threshold."
        )
    elif not k882_pass and not k884_kill:
        outcome = "PARTIAL_quality_degraded"
        interpretation = (
            f"K882 FAIL ({q_high:.1%} < {K882_THRESHOLD:.0%}) but above K884 threshold. "
            f"Quality degradation ({0.976 - q_high:.1%}) exceeds the 12pp allowance. "
            f"Possible: n=2000 insufficient at d=512, or M2P capacity marginal. "
            f"Try n=4000 to test data scale, or increase d_M2P to 128."
        )
    else:
        outcome = "UNEXPECTED"
        interpretation = "Unexpected combination of kill criteria results."

    log(f"\n  Outcome: {outcome}")
    log(f"  Interpretation: {interpretation}")

    # N-scale comparison (n=1000 vs n=2000 at d=512)
    n_comparison_note = (
        f"n={SAMPLE_VALUES[0]} (epochs=1.25): quality={q_low:.1%}, gap={gap_low} | "
        f"n={SAMPLE_VALUES[1]} (epochs=0.625): quality={q_high:.1%}, gap={gap_high}"
    )
    log(f"\n  N-scale comparison: {n_comparison_note}")
    quality_improvement_n = q_high - q_low
    log(f"  Quality improvement n={SAMPLE_VALUES[0]}->n={SAMPLE_VALUES[1]}: "
        f"{quality_improvement_n:+.1%}")

    return {
        "k882_pass": bool(k882_pass),
        "k883_pass": bool(k883_pass),
        "k884_kill": bool(k884_kill),
        "outcome": outcome,
        "interpretation": interpretation,
        "quality_n_low": round(float(q_low), 4),
        "quality_n_high": round(float(q_high), 4),
        "micro_baseline_quality": 0.976,
        "delta_n_low_to_n_high": round(float(quality_improvement_n), 4),
        "delta_micro_to_macro_d": round(float(q_high - 0.976), 4),
        "max_train_val_gap_n_low": gap_low,
        "max_train_val_gap_n_high": gap_high,
        "k882_threshold": K882_THRESHOLD,
        "k883_threshold": K883_THRESHOLD,
        "k884_kill_threshold": K884_KILL_THRESHOLD,
        "n_comparison_note": n_comparison_note,
    }


# ====================================================================
# MAIN ORCHESTRATOR
# ====================================================================

def main():
    t0 = time.time()
    log("M2P Macro Quality: n_train>=T Guarantee at d_model=512")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log(f"THE ONLY CHANGE: D_MODEL={D_MODEL} (was 256 in Finding #359)")
    log(f"Architecture FIXED: L={M2P_LAYERS}, D_M2P={D_M2P}, N_HEADS={N_HEADS}")
    log(f"Sweep: N_SAMPLES in {SAMPLE_VALUES}, T FIXED at {T_FIXED}")
    log(f"Domains: {DOMAIN_NAMES} (N={N_DOMAINS}, reduced from 5 for runtime)")
    log(f"")
    log(f"MATH.md Theorem 1: n_train>=T is d_model-independent.")
    log(f"  At n=2000: n_train=1600, T/n_train=0.625 < 1 epoch. No cycling.")
    log(f"Ghadimi & Lan (arXiv:1309.5549): O(1/T) convergence under i.i.d. gradients")
    log(f"Bartlett et al. (arXiv:1906.11300): d_eff doubles at d=512 (n/d_eff ~ 1)")
    log(f"Prechelt (1998): GL criterion stops when val_loss/best_val_loss - 1 > {GL_THRESHOLD/100:.0%}")
    log(f"")
    log(f"KILL CRITERIA:")
    log(f"  K882 (#885): quality(n=2000, d=512) >= 85% -> PASS (capacity adequate)")
    log(f"  K883 (#886): max train-val gap(n=2000) < 0.7 nats -> PASS (no overfit)")
    log(f"  K884 (#887): quality(n=2000, d=512) < 60% -> KILL (capacity bottleneck)")
    log(f"")
    log(f"MICRO BASELINE: quality(n=2000, d=256, T=1000) = 97.6% (Finding #359)")
    log(f"FLOOR PREDICTION: >= 85% at d=512 (Bartlett d_eff scaling, 12pp allowance)")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # -- Generate data at maximum scale for base/SFT training --
    domain_data_max = phase_generate_data(rng, n_per_domain=max(SAMPLE_VALUES))
    log_memory("after max-scale data")

    # -- Base model (d=512, 8 heads -- THE SINGLE CHANGE) --
    base, base_losses = phase_pretrain_base(domain_data_max)
    log_memory("after base")

    # -- Grassmannian A-matrices (at D_MODEL=512) --
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # -- SFT baselines (fresh, max-scale data) --
    sft_results = phase_sft_all_domains(domain_data_max, base, A_matrices, base_losses)
    log_memory("after SFT")

    # -- Generate data at each sweep scale for M2P training --
    domain_data_by_n = {}
    for n_val in SAMPLE_VALUES:
        rng_n = np.random.RandomState(SEED + n_val)
        domain_data_by_n[n_val] = phase_generate_data(rng_n, n_per_domain=n_val)
        log_memory(f"after data n={n_val}")

    # -- N_SAMPLES sweep at d=512 --
    sweep_results = phase_sweep_n_samples(
        domain_data_by_n, base, A_matrices, base_losses, sft_results,
        domain_data_max=domain_data_max,
    )
    log_memory("after n_samples sweep")

    # -- Kill criteria evaluation --
    kill_criteria = evaluate_kill_criteria(sweep_results)

    # -- Results assembly --
    total_time = round(time.time() - t0, 1)
    results = {
        "experiment": "exp_m2p_macro_quality",
        "total_time_s": total_time,
        "smoke_test": SMOKE_TEST,
        # THE SINGLE CHANGE
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_head": D_MODEL // N_HEADS,
        # Architecture FIXED
        "m2p_layers_fixed": M2P_LAYERS,
        "d_m2p_fixed": D_M2P,
        "t_fixed": T_FIXED,
        "sample_values": SAMPLE_VALUES,
        "n_domains": N_DOMAINS,
        "domain_names": DOMAIN_NAMES,
        "parity_guard_threshold": PARITY_GUARD_THRESHOLD,
        "early_stopping_gl_threshold": GL_THRESHOLD,
        "early_stopping_patience": PATIENCE,
        "early_stopping_interval": EARLY_STOP_INTERVAL,
        # Module output dims (scaled with d_model)
        "module_out_dims": MODULE_OUT_DIMS,
        # Micro baseline reference (Finding #359)
        "micro_baseline": {
            "finding": 359,
            "d_model": 256,
            "n_samples": 2000,
            "t_fixed": 1000,
            "quality_ratio": 0.976,
            "max_train_val_gap": 0.337,
        },
        # Per-n-scale results
        f"n{SAMPLE_VALUES[0]}_T{T_FIXED}": sweep_results[f"n{SAMPLE_VALUES[0]}"],
        f"n{SAMPLE_VALUES[1]}_T{T_FIXED}": sweep_results[f"n{SAMPLE_VALUES[1]}"],
        # Convenience: median quality per condition
        f"median_n{SAMPLE_VALUES[0]}": sweep_results[f"n{SAMPLE_VALUES[0]}"]["median_quality"],
        f"median_n{SAMPLE_VALUES[1]}": sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"],
        # Kill criteria
        "kill_criteria": kill_criteria,
        # Reference losses
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        # Grassmannian verification
        "grassmannian_A_cos_max": ortho_result["max_cos"],
        # Prediction vs measurement table (for PAPER.md)
        "predictions_vs_measurements": {
            "theorem_1_n_train_gte_t": {
                "description": (
                    "MATH.md Theorem 1: n_train>=T is d_model-independent. "
                    "At n=2000: epochs=0.625. Structural guarantee satisfied."
                ),
                "predicted": "T/n_train = 0.625 < 1 (structural, no cycling)",
                "measured": round(T_FIXED / (0.8 * max(SAMPLE_VALUES)), 3),
                "match": "structural guarantee (cannot fail by construction)",
            },
            "quality_n2000_d512": {
                "description": f"quality_ratio(n=2000, d={D_MODEL}, T={T_FIXED}) >= 85%",
                "predicted": ">= 85% (K882, Bartlett d_eff scaling, 12pp allowance)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"],
                "k882_pass": kill_criteria["k882_pass"],
            },
            "train_val_gap_n2000": {
                "description": f"train-val gap at n=2000, d={D_MODEL} < 0.7 nats",
                "predicted": "< 0.7 nats (K883, 2x micro measured 0.337 nats)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[1]}"]["max_train_val_gap"],
                "k883_pass": kill_criteria["k883_pass"],
            },
            "k884_capacity_check": {
                "description": f"quality_ratio >= 60% (not capacity bottleneck)",
                "predicted": ">= 60% (K884; M2P d_M2P=64 expected sufficient)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"],
                "k884_not_triggered": not kill_criteria["k884_kill"],
            },
            "micro_to_macro_degradation": {
                "description": f"Degradation from d=256 to d=512",
                "predicted": "<= 12pp (Bartlett d_eff 2x scaling)",
                "measured": round(0.976 - sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"], 4),
            },
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults written to {RESULTS_FILE}")
    log(f"Total time: {total_time}s")

    # Final summary
    kc = kill_criteria
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"D_MODEL: {D_MODEL} (THE ONLY CHANGE from micro)")
    log(f"Micro baseline (Finding #359, d=256): 97.6%")
    log(f"This experiment (d=512):")
    log(f"  n={SAMPLE_VALUES[0]}: quality={kc['quality_n_low']:.1%}")
    log(f"  n={SAMPLE_VALUES[1]}: quality={kc['quality_n_high']:.1%}")
    log(f"  Quality delta (n={SAMPLE_VALUES[0]}->n={SAMPLE_VALUES[1]}): "
        f"{kc['delta_n_low_to_n_high']:+.1%}")
    log(f"  Degradation vs micro: {kc['delta_micro_to_macro_d']:+.1%}")
    log("")
    log(f"K882 (#885): quality >= 85%:           {'PASS' if kc['k882_pass'] else 'FAIL'} "
        f"(measured: {kc['quality_n_high']:.1%})")
    log(f"K883 (#886): train-val gap < 0.7 nats: {'PASS' if kc['k883_pass'] else 'FAIL'} "
        f"(measured: {kc['max_train_val_gap_n_high']})")
    log(f"K884 (#887): quality < 60% KILL:       {'KILL' if kc['k884_kill'] else 'NOT TRIGGERED'} "
        f"(measured: {kc['quality_n_high']:.1%})")
    log(f"\nOutcome: {kc['outcome']}")
    log(f"Interpretation: {kc['interpretation']}")


if __name__ == "__main__":
    main()
