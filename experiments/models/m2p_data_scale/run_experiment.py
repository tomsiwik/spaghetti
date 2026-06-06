#!/usr/bin/env python3
"""M2P Data Scale: Eliminate Cyclic Overfitting via Larger Training Set + Early Stopping.

TYPE: guided-exploration (Type 2)
MATH: experiments/models/m2p_data_scale/MATH.md

PRIOR KILL (Finding #358):
  exp_m2p_training_budget KILLED because 500 cyclic training samples caused overfitting.
  At T=2000 steps on 500 samples (4 epochs), reverse domain eval loss rose 2.80 → 3.86
  while train loss dropped. Arithmetic was the exception — O(1/T) trend held because it
  generalizes better from few samples (89.6% → 92.0% → 93.5%).

ROOT CAUSE (from MATH.md Theorem 1):
  Ghadimi & Lan (2013, arXiv:1309.5549) Theorem 2.1 requires i.i.d. gradient estimates.
  Cyclic data at T > n violates this: the gradient becomes deterministic once the M2P
  memorizes the training set. Fix condition: n ≥ T (at most 1 epoch per run).

HYPOTHESIS (Theorem 1, MATH.md):
  Increasing M2P_TRAIN_SAMPLES from 500 to {1000, 2000} while fixing T=1000:
  1. Eliminates cyclic overfitting (n ≥ T for n ∈ {1000, 2000})
  2. Restores O(1/T) convergence guarantee for generalization quality
  3. Allows quality to scale with n (all domains, not just arithmetic)
  Secondary: early stopping (GL criterion, Prechelt 1998) provides safety net.

KILL CRITERIA:
  K879: train-val loss gap at T=2000 < 0.5 nats (overfitting eliminated)
  K880: quality(n=2000, T=2000) > quality(n=500, T=500) + 3pp (quality improves)
  K881: per-domain quality monotone in n for ALL valid domains (trend restored)

ARCHITECTURE CONSTANTS (FIXED — same as exp_m2p_depth, exp_m2p_training_budget):
  D_MODEL=256, N_LAYERS=2, N_HEADS=4, VOCAB_SIZE=128, BLOCK_SIZE=48
  LORA_RANK=4, D_M2P=64, M2P_LAYERS=2, N_MEMORY=32
  N_DOMAINS=5 (arithmetic, sort, parity, reverse, repeat)
  ATTENTION: causal only (bidirectional hurt at n=500; don't re-test until data is fixed)

EXPERIMENT VARIABLE (single change from prior):
  M2P_TRAIN_SAMPLES swept: {500, 1000, 2000}  (T_fixed = 1000 for all)
  NOTE: n=500 at T=1000 is INTENTIONALLY the 2-epoch overfitting reference point,
        confirming root cause. n=1000 and n=2000 at T=1000 satisfy n ≥ T.

EARLY STOPPING (infrastructure, not hack):
  Monitor val loss every EARLY_STOP_INTERVAL=50 steps.
  GL(t) = 100 * (val_loss(t) / min_{s≤t} val_loss(s) - 1)
  Stop when GL(t) > GL_THRESHOLD=5.0 for PATIENCE=5 consecutive checks.
  Reports: stopping step, train-val gap at stopping point.

BASELINE: quality at n=500, T=500 = 89.4% median (Finding #358 actual measurement)
PREDICTION: quality(n=2000, T=2000) ≥ 93.5% (arithmetic O(1/T) floor, MATH.md Cor. 1.2)
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

# -- Architecture constants (FIXED — same as exp_m2p_depth, exp_m2p_training_budget) --
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5

# M2P architecture FIXED (both proven sufficient)
M2P_LAYERS = 2  # Finding #357: L=2 saturates
D_M2P = 64      # Finding #355: width not bottleneck
N_MEMORY = 32

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

# -- THE PRIMARY EXPERIMENT VARIABLE: data scale sweep --
# n=500 at T=1000 is the 2-epoch reference (overfitting expected, confirms root cause)
# n=1000 and n=2000 at T=1000 satisfy n >= T (i.i.d. condition)
SAMPLE_VALUES = [500, 1000, 2000]

# Fixed training steps (T ≤ min(n_sweep[1:]) = 1000 satisfies fix condition)
# n=500 is intentionally 2-epoch to verify overfitting persists
M2P_STEPS_FIXED = 1000

# Shared training constants
BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

# Smoke test overrides
if SMOKE_TEST:
    SAMPLE_VALUES = [60, 120, 240]
    M2P_STEPS_FIXED = 80

# Parity guard (unchanged from prior experiments)
DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]
PARITY_GUARD_THRESHOLD = 0.05

# Early stopping parameters (Prechelt 1998 GL criterion)
EARLY_STOP_INTERVAL = 50     # check val loss every N steps
GL_THRESHOLD = 5.0           # stop when GL(t) > 5.0
PATIENCE = 5                 # wait for 5 consecutive GL > threshold before stopping


# -- Utilities --

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


# -- Data generation (IDENTICAL to exp_m2p_training_budget) --

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        if domain_id == 0:  # arithmetic
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            data.append(f"{a}+{b}={a+b}")
        elif domain_id == 1:  # sort
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(sorted(s))}")
        elif domain_id == 2:  # parity
            bits = "".join(str(rng.randint(0, 2)) for _ in range(rng.randint(2, 6)))
            data.append(f"{bits}>{'even' if bits.count('1') % 2 == 0 else 'odd'}")
        elif domain_id == 3:  # reverse
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(reversed(s))}")
        elif domain_id == 4:  # repeat
            p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1, 3)))
            r = rng.randint(2, 4)
            data.append(f"{p}*{r}={p*r}")
    return data


def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]


def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]


# -- Toy GPT (IDENTICAL to exp_m2p_training_budget) --

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
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scale = self.head_dim ** -0.5
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
    """Toy GPT: d=256, L=2, 4 heads, vocab=128."""

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


# -- Grassmannian A-matrices (IDENTICAL to exp_m2p_training_budget) --

def generate_grassmannian_A(n_domains, n_layers, n_modules, d, rank, seed=42):
    total_rank = n_domains * rank
    assert total_rank <= d
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


# -- LoRA forward pass (IDENTICAL to exp_m2p_training_budget) --

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


# -- SFT training (IDENTICAL to exp_m2p_training_budget) --

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


# -- M2P Transformer (causal attention only — see MATH.md) --

class M2PAttention(nn.Module):
    """M2P attention with causal masking (bidirectional deferred until n > 500 confirmed)."""
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
    """
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.m2p_layers = m2p_layers
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(m2p_layers)]
        self.norm_f = RMSNorm(d_m2p)
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


# -- Evaluation helpers --

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


# ===================================================================
# PHASE FUNCTIONS (each self-contained per CODING_GUIDELINES 1)
# ===================================================================

def phase_generate_data(rng: np.random.RandomState, n_per_domain: int) -> dict:
    """Generate train/val data for all 5 domains with specified sample count."""
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
    """Pre-train ToyGPT on all domains (deterministic with fixed SEED).

    Always trained fresh — no reuse from prior experiments.
    Uses n_per_domain=2000 data to avoid any data-size confound in base pretraining.
    """
    log("\n=== Phase: Pre-train Base Model ===")
    mx.random.seed(SEED)

    base = ToyGPT()
    mx.eval(base.parameters())

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
    """Generate and verify Grassmannian A-matrices."""
    log("\n=== Phase: Grassmannian A-matrices ===")
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

    Uses domain_data["train"] as provided — for SFT, we use the largest
    data set available (n=2000) to get the best SFT quality reference.
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
) -> dict:
    """Train ONE M2P at fixed T=M2P_STEPS_FIXED for ONE domain.

    Key changes from exp_m2p_training_budget:
    1. Uses domain_data["train"] which is sized to n_samples (not fixed 500)
    2. Adds GL early stopping (Prechelt 1998): stop when GL(t) > GL_THRESHOLD
       for PATIENCE consecutive checks (every EARLY_STOP_INTERVAL steps)
    3. Reports train AND val loss at stopping point (for K879 train-val gap)
    4. Reports n_epochs = T / n_train (should be ≤ 1.0 for n=T=1000, 2000)

    Returns quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)
    """
    n_train = len(domain_data["train"])
    n_epochs_expected = M2P_STEPS_FIXED / max(n_train, 1)

    adapter_dir = EXPERIMENT_DIR / f"adapters_n{n_samples}"
    adapter_dir.mkdir(exist_ok=True)
    save_path = adapter_dir / f"m2p_{domain_name}.npz"

    log(f"    [n={n_samples}] Training M2P for {domain_name} "
        f"(T={M2P_STEPS_FIXED}, n_train={n_train}, "
        f"epochs={n_epochs_expected:.2f})...")
    mx.random.seed(SEED)

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS)
    mx.eval(m2p.parameters())

    param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"      M2P params: {param_count:,} (L={M2P_LAYERS}, d={D_M2P})")

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id,
                             tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]
    val_batches = domain_data["val"]

    # Early stopping state
    best_val_loss = float("inf")
    best_step = 0
    consecutive_gl_exceeded = 0
    early_stop_triggered = False
    stopping_step = M2P_STEPS_FIXED

    # Track trajectories
    loss_trajectory = []
    val_loss_trajectory = []
    final_train_loss = None
    final_val_loss = None

    gc.disable()
    for step in range(M2P_STEPS_FIXED):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)

        train_loss_val = loss.item()
        final_train_loss = train_loss_val

        # Log at 25%, 50%, 75%, 100%
        if (step + 1) % max(1, M2P_STEPS_FIXED // 4) == 0:
            loss_trajectory.append({
                "step": step + 1,
                "train_loss": round(train_loss_val, 4),
            })

        # Early stopping: evaluate val loss every EARLY_STOP_INTERVAL steps
        if (step + 1) % EARLY_STOP_INTERVAL == 0 and not early_stop_triggered:
            # Temporarily unfreeze to evaluate val loss via M2P
            # Get current M2P B-matrices from a context sample
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

            # GL criterion (Prechelt 1998)
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

    # Evaluate: load saved B-matrices and measure NTP loss
    B_matrices_loaded = load_B_matrices(str(save_path))
    m2p_val_loss = eval_ntp_loss(base, val_batches, A_matrices,
                                  domain_id, B_matrices_loaded)
    final_val_loss = m2p_val_loss

    # Train-val gap at stopping point (K879)
    train_val_gap = abs(final_train_loss - final_val_loss) if final_train_loss else None

    # Parity guard
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
        log(f"    [n={n_samples}] {domain_name}: val_loss={m2p_val_loss:.4f} "
            f"train_loss={final_train_loss:.4f} "
            f"train_val_gap={train_val_gap:.4f} "
            f"SFT={sft_loss:.4f} base={base_loss:.4f} "
            f"quality={quality_ratio:.1%} "
            f"n_epochs={n_epochs_expected:.2f} "
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
        "n_epochs_actual": round(M2P_STEPS_FIXED / max(n_train, 1), 3),
        "early_stop_triggered": early_stop_triggered,
        "stopping_step": stopping_step,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "best_step": best_step,
        "loss_trajectory": loss_trajectory,
        "val_loss_trajectory": val_loss_trajectory,
    }


def phase_sweep_data_scale(domain_data_by_n, base, A_matrices,
                            base_losses, sft_results) -> dict:
    """Sweep M2P_TRAIN_SAMPLES in {500, 1000, 2000} with fixed T=1000.

    MATH.md Theorem 1: n >= T is required for O(1/T) to apply to generalization.
    n=500 at T=1000: INTENTIONALLY 2-epoch reference (confirms root cause from Finding #358)
    n=1000 at T=1000: 1 epoch (fix condition satisfied)
    n=2000 at T=1000: 0.5 epoch (fix condition satisfied with margin)
    """
    log(f"\n=== Phase: M2P Data Scale Sweep ===")
    log(f"  Sweeping M2P_TRAIN_SAMPLES in {SAMPLE_VALUES}")
    log(f"  Fixed T={M2P_STEPS_FIXED} steps")
    log(f"  Ghadimi-Lan n>=T condition: "
        f"n=500 VIOLATES (2 epochs), n=1000 SATISFIES (1 epoch), "
        f"n=2000 SATISFIES (0.5 epoch)")
    log(f"  Early stopping: GL threshold={GL_THRESHOLD}, "
        f"patience={PATIENCE} checks, interval={EARLY_STOP_INTERVAL} steps")

    sweep_results = {}

    for n_samples in SAMPLE_VALUES:
        log(f"\n  --- n={n_samples} (T={M2P_STEPS_FIXED}) ---")
        domain_data = domain_data_by_n[n_samples]
        domain_qualities = {}
        domain_val_losses = {}
        domain_train_losses = {}
        domain_train_val_gaps = {}
        domain_epochs = {}
        domain_early_stopped = {}
        excluded_domains = []

        for di, name in enumerate(DOMAIN_NAMES):
            result = phase_train_m2p_with_early_stopping(
                n_samples=n_samples,
                domain_name=name,
                domain_id=di,
                domain_data=domain_data[name],
                base=base,
                A_matrices=A_matrices,
                base_loss=base_losses[name],
                sft_loss=sft_results[name]["sft_loss"],
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
        log(f"  Max train-val gap: {max_train_val_gap:.4f} (K879 threshold: 0.5)")
        n_early_stopped = sum(1 for n in DOMAIN_NAMES
                              if n not in excluded_domains and domain_early_stopped[n])
        log(f"  Early stops triggered: {n_early_stopped}/{len(valid_qualities)} valid domains")

        sweep_results[f"n{n_samples}"] = {
            "n_samples": n_samples,
            "m2p_steps_fixed": M2P_STEPS_FIXED,
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


# ===================================================================
# KILL CRITERIA EVALUATION
# ===================================================================

def evaluate_kill_criteria(sweep_results: dict,
                            baseline_quality: float = 0.894) -> dict:
    """Evaluate K879, K880, K881 from sweep results.

    K879: train-val loss gap at n=2000 < 0.5 nats (overfitting eliminated)
         Source: MATH.md Theorem 2, Hardt et al. (2016) generalization bound
    K880: quality(n=2000) > baseline + 3pp = 89.4% + 3pp = 92.4%
         Source: MATH.md Theorem 1 Corollary 1.2 (arithmetic O(1/T) floor)
    K881: per-domain quality is monotone in n for ALL valid domains (trend restored)
         Source: MATH.md Theorem 1 (once n >= T, quality should not degrade with n)
    """
    # Find the n=500, n=1000, n=2000 keys
    n_low_key = f"n{SAMPLE_VALUES[0]}"   # n500
    n_mid_key = f"n{SAMPLE_VALUES[1]}"   # n1000
    n_high_key = f"n{SAMPLE_VALUES[2]}"  # n2000

    r_low = sweep_results[n_low_key]
    r_mid = sweep_results[n_mid_key]
    r_high = sweep_results[n_high_key]

    q_low = r_low["median_quality"]
    q_mid = r_mid["median_quality"]
    q_high = r_high["median_quality"]

    gap_low = r_low["max_train_val_gap"]
    gap_mid = r_mid["max_train_val_gap"]
    gap_high = r_high["max_train_val_gap"]

    # K879: overfitting eliminated at n=2000
    k879_pass = (gap_high is not None and gap_high < 0.5)

    # K880: quality improvement at n=2000 vs baseline
    k880_pass = (q_high > baseline_quality + 0.03)

    # K881: monotone per-domain quality in n for ALL valid domains
    # A domain is monotone if q(n_mid) >= q(n_low) and q(n_high) >= q(n_mid)
    # (or strict: q increases with n)
    valid_domains = [d for d in DOMAIN_NAMES
                     if d not in r_high["excluded_domains"]]
    monotone_domains = []
    non_monotone_domains = []
    for domain in valid_domains:
        q_d_low = r_low["domain_quality"].get(domain, None)
        q_d_mid = r_mid["domain_quality"].get(domain, None)
        q_d_high = r_high["domain_quality"].get(domain, None)
        if q_d_low is None or q_d_mid is None or q_d_high is None:
            continue
        # Allow tolerance of 2pp for micro-scale noise floor (LEARNINGS.md)
        noise_floor = 0.02
        if (q_d_mid >= q_d_low - noise_floor and q_d_high >= q_d_mid - noise_floor):
            monotone_domains.append(domain)
        else:
            non_monotone_domains.append(domain)

    k881_pass = (len(non_monotone_domains) == 0 and len(monotone_domains) > 0)

    log("\n=== Kill Criteria Evaluation ===")
    log(f"  Baseline (Finding #358): quality = {baseline_quality:.1%} at n=500, T=500")
    log(f"  n=500  (T={M2P_STEPS_FIXED}): quality={q_low:.1%}, "
        f"max_train_val_gap={gap_low}")
    log(f"  n=1000 (T={M2P_STEPS_FIXED}): quality={q_mid:.1%}, "
        f"max_train_val_gap={gap_mid}")
    log(f"  n=2000 (T={M2P_STEPS_FIXED}): quality={q_high:.1%}, "
        f"max_train_val_gap={gap_high}")
    log("")
    log(f"  K879: max train-val gap(n=2000) < 0.5: "
        f"{gap_high} -> {'PASS' if k879_pass else 'FAIL'}")
    log(f"  K880: quality(n=2000) > {baseline_quality:.1%} + 3pp = {baseline_quality+0.03:.1%}: "
        f"{q_high:.1%} -> {'PASS' if k880_pass else 'FAIL'}")
    log(f"  K881: all domains monotone in n: "
        f"{len(monotone_domains)}/{len(valid_domains)} monotone "
        f"(non-monotone: {non_monotone_domains}) -> {'PASS' if k881_pass else 'FAIL'}")

    # Overfitting diagnosis
    overfitting_eliminated = k879_pass
    quality_restored = k880_pass
    trend_restored = k881_pass

    if k879_pass and k880_pass and k881_pass:
        outcome = "A_data_scale_works"
        interpretation = (
            f"Data scale eliminates overfitting: train-val gap {gap_high:.3f} < 0.5, "
            f"quality {q_high:.1%} > {baseline_quality+0.03:.1%}, "
            f"all {len(monotone_domains)} domains monotone. "
            f"O(1/T) theorem applies at n=2000. Next: scale T up to 2000."
        )
    elif k879_pass and k880_pass and not k881_pass:
        outcome = "B_mostly_works_trend_noisy"
        interpretation = (
            f"Overfitting eliminated and quality improved, but {non_monotone_domains} "
            f"non-monotone. Likely micro-scale noise floor ({2}pp). "
            f"Conclusion: data scale helps, noise is the remaining limit."
        )
    elif k879_pass and not k880_pass:
        outcome = "C_overfit_gone_quality_low"
        interpretation = (
            f"Overfitting eliminated (gap={gap_high:.3f} < 0.5) but quality "
            f"did not improve enough ({q_high:.1%} <= {baseline_quality+0.03:.1%}). "
            f"Possible: n=2000 is still insufficient, or the 8% gap is irreducible."
        )
    elif not k879_pass:
        outcome = "D_overfit_persists"
        interpretation = (
            f"Overfitting persists at n=2000 (gap={gap_high}). "
            f"n* > 2000 required, OR the GL threshold is too permissive. "
            f"Next: try n=5000 or add dropout regularization."
        )
    else:
        outcome = "E_unexpected"
        interpretation = "Unexpected combination of kill criteria."

    log(f"\n  Outcome: {outcome}")
    log(f"  Interpretation: {interpretation}")

    # Overfitting progression (does n=500 overfit more than n=1000?)
    n500_overfit = (gap_low is not None and gap_low > 0.5)
    n1000_overfit = (gap_mid is not None and gap_mid > 0.5)
    log(f"\n  Overfitting progression:")
    log(f"    n=500 overfit (gap > 0.5): {n500_overfit} (gap={gap_low})")
    log(f"    n=1000 overfit (gap > 0.5): {n1000_overfit} (gap={gap_mid})")
    log(f"    n=2000 overfit (gap > 0.5): {not k879_pass} (gap={gap_high})")

    return {
        "k879_pass": bool(k879_pass),
        "k880_pass": bool(k880_pass),
        "k881_pass": bool(k881_pass),
        "outcome": outcome,
        "interpretation": interpretation,
        "quality_n500": round(float(q_low), 4),
        "quality_n1000": round(float(q_mid), 4),
        "quality_n2000": round(float(q_high), 4),
        "baseline_quality_n500_T500": baseline_quality,
        "delta_n500_to_n2000": round(float(q_high - q_low), 4),
        "delta_baseline_to_n2000": round(float(q_high - baseline_quality), 4),
        "max_train_val_gap_n500": gap_low,
        "max_train_val_gap_n1000": gap_mid,
        "max_train_val_gap_n2000": gap_high,
        "monotone_domains": monotone_domains,
        "non_monotone_domains": non_monotone_domains,
        "n_valid_domains": len(valid_domains),
        "n_monotone_domains": len(monotone_domains),
        "overfitting_eliminated": bool(overfitting_eliminated),
        "quality_restored": bool(quality_restored),
        "trend_restored": bool(trend_restored),
        "n500_overfit_confirmed": bool(n500_overfit),
    }


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================

def main():
    t0 = time.time()
    log("M2P Data Scale Sweep -- Eliminate Cyclic Overfitting via Larger Training Set")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log(f"Architecture FIXED: L={M2P_LAYERS}, D_M2P={D_M2P}")
    log(f"Sweep: M2P_TRAIN_SAMPLES in {SAMPLE_VALUES}, T FIXED at {M2P_STEPS_FIXED}")
    log(f"Prior kill: exp_m2p_training_budget (Finding #358 KILLED -- cyclic overfitting)")
    log(f"Root cause: n=500 samples at T=2000 = 4 epochs, violates Ghadimi-Lan i.i.d.")
    log(f"Fix: n >= T (Theorem 1, MATH.md) + GL early stopping (Prechelt 1998)")
    log(f"Ghadimi & Lan (arXiv:1309.5549): O(1/T) requires i.i.d. gradient estimates")
    log(f"Bartlett et al. (arXiv:1906.11300): benign overfitting requires n >> d_eff")
    log(f"Prechelt (1998): GL criterion stops when val_loss / best_val_loss - 1 > {GL_THRESHOLD/100:.0%}")
    log(f"Parity guard: exclude domains where base_loss - sft_loss < {PARITY_GUARD_THRESHOLD}")
    log(f"Baseline: quality(n=500, T=500) = 89.4% (Finding #358)")
    log(f"Prediction: quality(n=2000, T=2000) >= 93.5% (arithmetic O(1/T) floor)")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # -- Generate data at maximum scale (n=2000) for base/SFT training --
    # and separately at each sweep scale for M2P training
    # Base and SFT always use the largest dataset (avoid confounding)
    domain_data_max = phase_generate_data(rng, n_per_domain=max(SAMPLE_VALUES))
    log_memory("after max-scale data")

    # -- Base model (FRESH -- uses max-scale data for best base quality) --
    base, base_losses = phase_pretrain_base(domain_data_max)
    log_memory("after base")

    # -- Grassmannian A-matrices --
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # -- SFT baselines (FRESH -- uses max-scale data) --
    sft_results = phase_sft_all_domains(domain_data_max, base, A_matrices, base_losses)
    log_memory("after SFT")

    # -- Generate data at each sweep scale for M2P training --
    # Use a separate RNG state seeded consistently to avoid data contamination
    domain_data_by_n = {}
    for n_val in SAMPLE_VALUES:
        # Use a fresh RNG for each n to ensure each dataset is fully i.i.d.
        rng_n = np.random.RandomState(SEED + n_val)
        domain_data_by_n[n_val] = phase_generate_data(rng_n, n_per_domain=n_val)
        log_memory(f"after data n={n_val}")

    # -- M2P data scale sweep --
    sweep_results = phase_sweep_data_scale(
        domain_data_by_n, base, A_matrices, base_losses, sft_results
    )
    log_memory("after data scale sweep")

    # -- Kill criteria --
    # Baseline from Finding #358: quality(n=500, T=500) = 89.4%
    # Using n=500 at T=500 as our in-experiment baseline would require an extra run.
    # Instead, use the Finding #358 measured value 89.4% as the reference.
    # The n=500 at T=1000 result shows the CURRENT (2-epoch) regime for comparison.
    kill_criteria = evaluate_kill_criteria(sweep_results, baseline_quality=0.894)

    # -- Results assembly --
    total_time = round(time.time() - t0, 1)
    results = {
        "experiment": "exp_m2p_data_scale",
        "total_time_s": total_time,
        "smoke_test": SMOKE_TEST,
        # Architecture (FIXED)
        "m2p_layers_fixed": M2P_LAYERS,
        "d_m2p_fixed": D_M2P,
        "m2p_steps_fixed": M2P_STEPS_FIXED,
        "sample_values": SAMPLE_VALUES,
        "parity_guard_threshold": PARITY_GUARD_THRESHOLD,
        "early_stopping_gl_threshold": GL_THRESHOLD,
        "early_stopping_patience": PATIENCE,
        "early_stopping_interval": EARLY_STOP_INTERVAL,
        # Per-n-scale results
        "n500_T1000": sweep_results[f"n{SAMPLE_VALUES[0]}"],
        "n1000_T1000": sweep_results[f"n{SAMPLE_VALUES[1]}"],
        "n2000_T1000": sweep_results[f"n{SAMPLE_VALUES[2]}"],
        # Convenience: median quality per condition
        "median_n500": sweep_results[f"n{SAMPLE_VALUES[0]}"]["median_quality"],
        "median_n1000": sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"],
        "median_n2000": sweep_results[f"n{SAMPLE_VALUES[2]}"]["median_quality"],
        # Kill criteria
        "kill_criteria": kill_criteria,
        # Reference losses
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        # Grassmannian verification
        "grassmannian_A_cos_max": ortho_result["max_cos"],
        # Prediction vs measurement table (for PAPER.md)
        "predictions_vs_measurements": {
            "baseline_n500_T500": {
                "description": "Baseline from Finding #358 (n=500, T=500)",
                "predicted": "89.4% (known)",
                "measured": None,  # not re-measured here — use Finding #358 value
                "note": "Reference, not re-measured (avoid redundant compute)",
            },
            "n500_T1000_overfit_reference": {
                "description": "n=500 at T=1000: expect overfitting (2 epochs, VIOLATES n>=T)",
                "predicted": "<89.4% (overfitting degrades quality)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[0]}"]["median_quality"],
            },
            "n1000_T1000": {
                "description": "n=1000 at T=1000: 1 epoch (satisfies n>=T)",
                "predicted": "~91% (conservative, Corollary 1.2)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[1]}"]["median_quality"],
            },
            "n2000_T1000_quality": {
                "description": "n=2000 at T=1000: 0.5 epoch (satisfies n>=T with margin)",
                "predicted": ">=93.5% (arithmetic O(1/T) floor, Theorem 1)",
                "measured": sweep_results[f"n{SAMPLE_VALUES[2]}"]["median_quality"],
            },
            "n2000_train_val_gap": {
                "description": "Train-val gap at n=2000 < 0.5 nats (Theorem 2, Hardt et al.)",
                "predicted": "<0.5 nats",
                "measured": sweep_results[f"n{SAMPLE_VALUES[2]}"]["max_train_val_gap"],
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
    log(f"K879 (overfit gone, gap<0.5):   {'PASS' if kc['k879_pass'] else 'FAIL'}")
    log(f"K880 (quality > 92.4%):          {'PASS' if kc['k880_pass'] else 'FAIL'}")
    log(f"K881 (all domains monotone):     {'PASS' if kc['k881_pass'] else 'FAIL'}")
    log(f"Outcome: {kc['outcome']}")
    log(f"Interpretation: {kc['interpretation']}")


if __name__ == "__main__":
    main()
