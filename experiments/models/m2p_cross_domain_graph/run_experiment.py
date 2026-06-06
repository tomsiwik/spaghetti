#!/usr/bin/env python3
"""M2P Cross-Domain Graph with Dissolve-Recrystallize Cycle.

TYPE: guided-exploration
MATH: experiments/models/m2p_cross_domain_graph/MATH.md

APPROACH:
  1. Pre-train toy GPT base, generate Grassmannian A-matrices
  2. Train per-domain SFT baselines (reference quality)
  3. Train per-domain M2P (proven: Finding #351, 93.3%)
  4. Train cross-domain M2P with 3 options:
     A: Cross-prediction (domain a context → help domain b)
     B: Residual transfer (domain a context → fix domain b's remaining errors)
     C: Combined (joint loss of A + B)
  5. Dissolve: merge cross-domain adapters into base (promotion at scale=5)
  6. Recrystallize: retrain per-domain M2P on enriched base
  7. Measure: does enriched base improve per-domain quality?

Kill criteria:
  K_A: ≥3/10 cross-domain pairs reduce target loss >5%
  K_B: Enriched base per-domain quality ≥ original for ≥3/5 domains
  K_C: Grassmannian |cos| < 1e-5 (structural)
  K_D: Best option achieves >50% median quality ratio on enriched base
"""

import gc
import json
import math
import os
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (CODING_GUIDELINES §2)
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── Architecture constants ─────────────────────────────────────────────────
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5
PROMOTE_SCALE = 5.0  # Finding #333: safe promotion scale

# M2P config
D_M2P = 64
N_MEMORY = 32
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

# Training config
BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
M2P_STEPS  = 500  if not SMOKE_TEST else 30
CROSS_STEPS = 300 if not SMOKE_TEST else 20  # cross-domain training
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]


# ── Utilities ──────────────────────────────────────────────────────────────

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


# ── Data generation ────────────────────────────────────────────────────────

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


# ── Toy GPT (identical to m2p_composition_n5) ─────────────────────────────

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


# ── Grassmannian A-matrices ────────────────────────────────────────────────

def generate_grassmannian_A(n_slots: int, n_layers: int, n_modules: int,
                             d: int, rank: int, seed: int = 42) -> dict:
    """Generate frozen orthogonal A-matrices via QR decomposition.
    n_slots = n_domains + n_cross_pairs (all get orthogonal slots).
    Returns: dict[(slot_idx, layer_idx, module_idx)] → mx.array(d, rank)
    """
    total_rank = n_slots * rank
    assert total_rank <= d, \
        f"Capacity violated: need {total_rank} orthogonal vectors but d={d}"

    rng = np.random.RandomState(seed)
    A_matrices = {}

    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)
            for si in range(n_slots):
                start = si * rank
                A_matrices[(si, li, mi)] = mx.array(Q[:, start:start + rank])

    return A_matrices


def verify_orthogonality(A_matrices: dict, n_slots: int,
                          n_layers: int, n_modules: int) -> dict:
    cos_values = []
    for li in range(n_layers):
        for mi in range(n_modules):
            for si in range(n_slots):
                for sj in range(si + 1, n_slots):
                    ai = A_matrices[(si, li, mi)].reshape(-1)
                    aj = A_matrices[(sj, li, mi)].reshape(-1)
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


# ── LoRA forward pass ─────────────────────────────────────────────────────

def lora_forward_with_B(base: ToyGPT, tokens: mx.array,
                         A_matrices: dict, slot_id: int,
                         B_matrices: dict) -> mx.array:
    """Forward pass with single adapter applied (slot_id selects A-matrices)."""
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
            A = A_matrices[(slot_id, li, mi)]
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
        A_fc1 = A_matrices[(slot_id, li, 4)]
        B_fc1 = B_matrices[(li, 4)]
        fc1_out = fc1_base + LORA_SCALE * (x_norm2 @ A_fc1) @ B_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    logits = base.lm_head(base.norm_f(x))
    return logits


def lora_forward_two_adapters(base: ToyGPT, tokens: mx.array,
                               A_matrices: dict,
                               slot_1: int, B_1: dict,
                               slot_2: int, B_2: dict) -> mx.array:
    """Forward pass with TWO adapters composed (for residual transfer: per-domain + cross-domain)."""
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        def _apply_two_lora(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            A1 = A_matrices[(slot_1, li, mi)]
            B1m = B_1[(li, mi)]
            A2 = A_matrices[(slot_2, li, mi)]
            B2m = B_2[(li, mi)]
            return base_out + LORA_SCALE * (x_in @ A1) @ B1m + LORA_SCALE * (x_in @ A2) @ B2m

        q = _apply_two_lora(attn.wq, x_norm, li, 0)
        k = _apply_two_lora(attn.wk, x_norm, li, 1)
        v = _apply_two_lora(attn.wv, x_norm, li, 2)

        q = q.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)

        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        scale_factor = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale_factor + mask, axis=-1)
        attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(B_b, T_b, C)

        attn_out = _apply_two_lora(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        x_norm2 = block.norm2(x)
        fc1_base = block.mlp.fc1(x_norm2)
        A1_fc1 = A_matrices[(slot_1, li, 4)]
        B1_fc1 = B_1[(li, 4)]
        A2_fc1 = A_matrices[(slot_2, li, 4)]
        B2_fc1 = B_2[(li, 4)]
        fc1_out = fc1_base + LORA_SCALE * (x_norm2 @ A1_fc1) @ B1_fc1 + LORA_SCALE * (x_norm2 @ A2_fc1) @ B2_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    logits = base.lm_head(base.norm_f(x))
    return logits


# ── B-matrix container ─────────────────────────────────────────────────────

class BMatrices(nn.Module):
    def __init__(self):
        super().__init__()
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                d_out = MODULE_OUT_DIMS[mi]
                setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

    def as_dict(self) -> dict:
        return {
            (li, mi): getattr(self, f"B_{li}_{mi}")
            for li in range(N_LAYERS) for mi in range(N_MODULES)
        }


# ── M2P Transformer ───────────────────────────────────────────────────────

class M2PAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
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
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class M2PMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
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
    """Memory-to-Parameter Transformer (single-domain, proven #351)."""

    def __init__(self, d_base: int = D_MODEL, d_m2p: int = D_M2P):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(M2P_LAYERS)]
        self.norm_f = RMSNorm(d_m2p)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            total_out = N_LAYERS * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list: list) -> dict:
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


# ── Loss functions ─────────────────────────────────────────────────────────

def m2p_ntp_loss(m2p: M2PTransformer, base: ToyGPT,
                 A_matrices: dict, slot_id: int,
                 tokens: mx.array) -> mx.array:
    """Standard M2P loss: generate B from context, evaluate NTP."""
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, slot_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


def cross_prediction_loss(m2p_cross: M2PTransformer, base: ToyGPT,
                           A_matrices: dict, cross_slot: int,
                           context_tokens: mx.array, target_tokens: mx.array) -> mx.array:
    """Option A: Show M2P context from domain a, evaluate loss on domain b tokens."""
    hidden_states = base.get_hidden_states(context_tokens)
    B_cross = m2p_cross(hidden_states)
    logits = lora_forward_with_B(base, target_tokens, A_matrices, cross_slot, B_cross)
    return nn.losses.cross_entropy(logits[:, :-1], target_tokens[:, 1:], reduction="mean")


def residual_transfer_loss(m2p_cross: M2PTransformer, base: ToyGPT,
                            A_matrices: dict, cross_slot: int,
                            perdomain_slot: int, perdomain_B: dict,
                            context_tokens: mx.array, target_tokens: mx.array) -> mx.array:
    """Option B: Generate cross-domain adapter, compose with per-domain, evaluate on target."""
    hidden_states = base.get_hidden_states(context_tokens)
    B_cross = m2p_cross(hidden_states)
    logits = lora_forward_two_adapters(
        base, target_tokens, A_matrices,
        perdomain_slot, perdomain_B,
        cross_slot, B_cross
    )
    return nn.losses.cross_entropy(logits[:, :-1], target_tokens[:, 1:], reduction="mean")


def combined_loss(m2p_cross: M2PTransformer, base: ToyGPT,
                   A_matrices: dict, cross_slot: int,
                   perdomain_slot: int, perdomain_B: dict,
                   context_tokens: mx.array, target_tokens: mx.array,
                   alpha: float = 0.5) -> mx.array:
    """Option C: Combined cross-prediction + residual transfer."""
    hidden_states = base.get_hidden_states(context_tokens)
    B_cross = m2p_cross(hidden_states)

    # Cross-prediction component
    logits_cross = lora_forward_with_B(base, target_tokens, A_matrices, cross_slot, B_cross)
    loss_cross = nn.losses.cross_entropy(logits_cross[:, :-1], target_tokens[:, 1:], reduction="mean")

    # Residual component
    logits_resid = lora_forward_two_adapters(
        base, target_tokens, A_matrices,
        perdomain_slot, perdomain_B,
        cross_slot, B_cross
    )
    loss_resid = nn.losses.cross_entropy(logits_resid[:, :-1], target_tokens[:, 1:], reduction="mean")

    return alpha * loss_cross + (1 - alpha) * loss_resid


# ── Evaluation ─────────────────────────────────────────────────────────────

def eval_ntp_loss(base: ToyGPT, batches: list,
                  A_matrices: dict = None, slot_id: int = None,
                  B_matrices: dict = None) -> float:
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        if A_matrices is not None and B_matrices is not None:
            logits = lora_forward_with_B(base, tokens_2d, A_matrices, slot_id, B_matrices)
        else:
            logits = base(tokens_2d)
        loss = nn.losses.cross_entropy(logits[:, :-1], tokens_2d[:, 1:], reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def eval_two_adapters(base: ToyGPT, batches: list, A_matrices: dict,
                       slot_1: int, B_1: dict, slot_2: int, B_2: dict) -> float:
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        logits = lora_forward_two_adapters(base, tokens_2d, A_matrices, slot_1, B_1, slot_2, B_2)
        loss = nn.losses.cross_entropy(logits[:, :-1], tokens_2d[:, 1:], reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


# ── LoRA merge into base weights (dissolve) ───────────────────────────────

def merge_adapter_into_base(base: ToyGPT, A_matrices: dict, slot_id: int,
                             B_matrices: dict, scale: float) -> None:
    """Promote adapter into base weights: W' = W + scale * A @ B^T.
    Modifies base IN PLACE. The Grassmannian slot is 'spent' after this.
    """
    module_map = {
        0: lambda block: block.attn.wq,
        1: lambda block: block.attn.wk,
        2: lambda block: block.attn.wv,
        3: lambda block: block.attn.wo,
        4: lambda block: block.mlp.fc1,
    }

    for li, block in enumerate(base.blocks):
        for mi in range(N_MODULES):
            A = A_matrices[(slot_id, li, mi)]   # (d_in, rank)
            B = B_matrices[(li, mi)]             # (rank, d_out)
            linear = module_map[mi](block)
            # W is (d_out, d_in), LoRA delta = (A @ B)^T = B^T @ A^T → (d_out, d_in)
            delta = (B.T @ A.T) if B.shape[0] == LORA_RANK else (A @ B).T
            # For nn.Linear: out = x @ W.T, so W is (d_out, d_in)
            # delta in parameter space: scale * B^T A^T has shape (d_out, d_in)
            delta_param = scale * B.T @ A.T  # (d_out, d_in) — matches W.weight
            new_weight = linear.weight + delta_param
            linear.weight = new_weight
            mx.eval(linear.weight)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    results = {"experiment": "m2p_cross_domain_graph", "smoke_test": SMOKE_TEST}
    rng = np.random.RandomState(SEED)

    # ── Phase 1: Generate data ─────────────────────────────────────────────
    log("=== Phase 1: Generate Data ===")
    n_per_domain = 500 if not SMOKE_TEST else 60
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        domain_data[name] = {
            "train": make_batches(texts[:split]),
            "val": make_batches(texts[split:]),
            "domain_id": di,
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, {len(domain_data[name]['val'])} val")

    # ── Phase 2: Pre-train base ────────────────────────────────────────────
    log("\n=== Phase 2: Pre-train Base Model ===")
    mx.random.seed(SEED)
    base = ToyGPT()
    mx.eval(base.parameters())

    all_train = []
    for name in DOMAIN_NAMES:
        all_train.extend(domain_data[name]["train"])

    optimizer = opt.Adam(learning_rate=LR)

    def base_loss_fn(model, tokens):
        tokens_2d = tokens[None, :]
        logits = model(tokens_2d)
        return nn.losses.cross_entropy(logits[:, :-1], tokens_2d[:, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(base, base_loss_fn)

    gc.disable()
    for step in range(BASE_STEPS):
        tokens = all_train[step % len(all_train)]
        loss, grads = loss_and_grad(base, tokens)
        optimizer.update(base, grads)
        mx.eval(base.parameters(), optimizer.state, loss)
        if (step + 1) % (BASE_STEPS // 4) == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={loss.item():.4f}")
    gc.enable()

    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")
    results["base_losses"] = base_losses

    # Save base weights
    base_weights_path = EXPERIMENT_DIR / "base_weights.npz"
    weights_dict = {}
    for k, v in tree_flatten(base.parameters()):
        weights_dict[k.replace(".", "_")] = np.array(v)
    np.savez(str(base_weights_path), **weights_dict)

    cleanup(optimizer)

    # ── Phase 3: Generate Grassmannian A-matrices ──────────────────────────
    log("\n=== Phase 3: Generate Grassmannian A-matrices ===")
    # Slot layout: 0-4 = per-domain, 5-14 = cross-domain pairs
    cross_pairs = list(combinations(range(N_DOMAINS), 2))  # 10 pairs
    n_cross = len(cross_pairs)
    n_total_slots = N_DOMAINS + n_cross  # 15 slots
    log(f"  Slots: {N_DOMAINS} per-domain + {n_cross} cross-domain = {n_total_slots} total")
    log(f"  Capacity: {n_total_slots * LORA_RANK} / {D_MODEL} = {n_total_slots * LORA_RANK / D_MODEL:.1%}")

    # Map cross-domain pairs to slot IDs
    cross_slot_map = {}  # (src_domain, tgt_domain) → slot_id
    for idx, (a, b) in enumerate(cross_pairs):
        cross_slot_map[(a, b)] = N_DOMAINS + idx
        cross_slot_map[(b, a)] = N_DOMAINS + idx  # symmetric slot, direction in training

    A_matrices = generate_grassmannian_A(n_total_slots, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK)
    orth_check = verify_orthogonality(A_matrices, n_total_slots, N_LAYERS, N_MODULES)
    log(f"  Orthogonality: mean_cos={orth_check['mean_cos']:.2e}, max_cos={orth_check['max_cos']:.2e}")
    results["grassmannian"] = orth_check
    results["cross_pairs"] = [(DOMAIN_NAMES[a], DOMAIN_NAMES[b]) for a, b in cross_pairs]

    # ── Phase 4: SFT baselines ─────────────────────────────────────────────
    log("\n=== Phase 4: SFT Baselines (per-domain) ===")
    sft_losses = {}
    sft_B = {}

    for di, name in enumerate(DOMAIN_NAMES):
        log(f"  Training SFT for {name}...")
        b_container = BMatrices()
        mx.eval(b_container.parameters())
        sft_opt = opt.Adam(learning_rate=SFT_LR)

        def sft_loss(bc, tokens):
            B_dict = bc.as_dict()
            logits = lora_forward_with_B(base, tokens[None, :], A_matrices, di, B_dict)
            return nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:], reduction="mean")

        sft_vg = nn.value_and_grad(b_container, sft_loss)
        train_data = domain_data[name]["train"]

        gc.disable()
        for step in range(SFT_STEPS):
            tokens = train_data[step % len(train_data)]
            loss, grads = sft_vg(b_container, tokens)
            sft_opt.update(b_container, grads)
            mx.eval(b_container.parameters(), sft_opt.state, loss)
        gc.enable()

        sl = eval_ntp_loss(base, domain_data[name]["val"], A_matrices, di, b_container.as_dict())
        sft_losses[name] = round(sl, 4)
        sft_B[di] = {k: mx.array(v) for k, v in b_container.as_dict().items()}
        mx.eval(list(sft_B[di].values()))
        log(f"    {name}: SFT loss={sl:.4f} (base={base_losses[name]:.4f})")
        cleanup(b_container, sft_opt)

    results["sft_losses"] = sft_losses

    # ── Phase 5: Per-domain M2P (proven, Finding #351) ─────────────────────
    log("\n=== Phase 5: Per-domain M2P ===")
    perdomain_m2p_losses = {}
    perdomain_B = {}

    for di, name in enumerate(DOMAIN_NAMES):
        log(f"  Training M2P for {name}...")
        mx.random.seed(SEED + di)
        m2p = M2PTransformer()
        mx.eval(m2p.parameters())
        m2p_opt = opt.Adam(learning_rate=M2P_LR)

        def m2p_loss(m2p_model, tokens):
            return m2p_ntp_loss(m2p_model, base, A_matrices, di, tokens[None, :])

        m2p_vg = nn.value_and_grad(m2p, m2p_loss)
        train_data = domain_data[name]["train"]

        gc.disable()
        for step in range(M2P_STEPS):
            tokens = train_data[step % len(train_data)]
            loss, grads = m2p_vg(m2p, tokens)
            m2p_opt.update(m2p, grads)
            mx.eval(m2p.parameters(), m2p_opt.state, loss)
        gc.enable()

        # Generate B-matrices from validation context
        val_tokens = domain_data[name]["val"][0][None, :]
        hidden = base.get_hidden_states(val_tokens)
        B_gen = m2p(hidden)
        mx.eval(list(B_gen.values()))
        perdomain_B[di] = {k: mx.array(v) for k, v in B_gen.items()}

        ml = eval_ntp_loss(base, domain_data[name]["val"], A_matrices, di, perdomain_B[di])
        perdomain_m2p_losses[name] = round(ml, 4)

        quality = (base_losses[name] - ml) / (base_losses[name] - sft_losses[name] + 1e-8)
        log(f"    {name}: M2P loss={ml:.4f}, quality={quality:.1%}")
        cleanup(m2p, m2p_opt)

    results["perdomain_m2p_losses"] = perdomain_m2p_losses

    # ── Phase 6: Cross-domain M2P (Options A, B, C) ────────────────────────
    log("\n=== Phase 6: Cross-Domain M2P ===")

    option_results = {}

    for option in ["A", "B", "C"]:
        log(f"\n--- Option {option}: {'Cross-Prediction' if option == 'A' else 'Residual Transfer' if option == 'B' else 'Combined'} ---")
        cross_B = {}  # (src, tgt) → B_matrices
        cross_quality = {}

        for (src, tgt) in cross_pairs:
            src_name = DOMAIN_NAMES[src]
            tgt_name = DOMAIN_NAMES[tgt]
            cross_slot = cross_slot_map[(src, tgt)]

            mx.random.seed(SEED + 100 * (ord(option) - ord('A')) + src * N_DOMAINS + tgt)
            m2p_cross = M2PTransformer()
            mx.eval(m2p_cross.parameters())
            cross_opt = opt.Adam(learning_rate=M2P_LR)

            src_train = domain_data[src_name]["train"]
            tgt_train = domain_data[tgt_name]["train"]

            if option == "A":
                def loss_fn_a(m2p_model, step_idx):
                    ctx = src_train[step_idx % len(src_train)][None, :]
                    tgt_tok = tgt_train[step_idx % len(tgt_train)][None, :]
                    return cross_prediction_loss(m2p_model, base, A_matrices, cross_slot, ctx, tgt_tok)

                m2p_vg = nn.value_and_grad(m2p_cross, loss_fn_a)
                gc.disable()
                for step in range(CROSS_STEPS):
                    loss, grads = m2p_vg(m2p_cross, step)
                    cross_opt.update(m2p_cross, grads)
                    mx.eval(m2p_cross.parameters(), cross_opt.state, loss)
                gc.enable()

            elif option == "B":
                tgt_perdomain_B = perdomain_B[tgt]

                def loss_fn_b(m2p_model, step_idx):
                    ctx = src_train[step_idx % len(src_train)][None, :]
                    tgt_tok = tgt_train[step_idx % len(tgt_train)][None, :]
                    return residual_transfer_loss(
                        m2p_model, base, A_matrices, cross_slot,
                        tgt, tgt_perdomain_B, ctx, tgt_tok
                    )

                m2p_vg = nn.value_and_grad(m2p_cross, loss_fn_b)
                gc.disable()
                for step in range(CROSS_STEPS):
                    loss, grads = m2p_vg(m2p_cross, step)
                    cross_opt.update(m2p_cross, grads)
                    mx.eval(m2p_cross.parameters(), cross_opt.state, loss)
                gc.enable()

            else:  # Option C
                tgt_perdomain_B = perdomain_B[tgt]

                def loss_fn_c(m2p_model, step_idx):
                    ctx = src_train[step_idx % len(src_train)][None, :]
                    tgt_tok = tgt_train[step_idx % len(tgt_train)][None, :]
                    return combined_loss(
                        m2p_model, base, A_matrices, cross_slot,
                        tgt, tgt_perdomain_B, ctx, tgt_tok, alpha=0.5
                    )

                m2p_vg = nn.value_and_grad(m2p_cross, loss_fn_c)
                gc.disable()
                for step in range(CROSS_STEPS):
                    loss, grads = m2p_vg(m2p_cross, step)
                    cross_opt.update(m2p_cross, grads)
                    mx.eval(m2p_cross.parameters(), cross_opt.state, loss)
                gc.enable()

            # Generate cross-domain B from source context
            src_val = domain_data[src_name]["val"][0][None, :]
            hidden = base.get_hidden_states(src_val)
            B_gen = m2p_cross(hidden)
            mx.eval(list(B_gen.values()))
            cross_B[(src, tgt)] = {k: mx.array(v) for k, v in B_gen.items()}

            # Evaluate: cross-domain adapter alone on target
            tgt_val = domain_data[tgt_name]["val"]
            cross_loss = eval_ntp_loss(base, tgt_val, A_matrices, cross_slot, cross_B[(src, tgt)])
            improvement = (base_losses[tgt_name] - cross_loss) / base_losses[tgt_name] * 100
            cross_quality[(src_name, tgt_name)] = {
                "cross_loss": round(cross_loss, 4),
                "base_loss": base_losses[tgt_name],
                "improvement_pct": round(improvement, 2),
                "useful": improvement > 5.0,
            }
            log(f"    {src_name}→{tgt_name}: loss={cross_loss:.4f}, improvement={improvement:+.1f}%")

            cleanup(m2p_cross, cross_opt)

        # Count useful pairs
        n_useful = sum(1 for v in cross_quality.values() if v["useful"])
        log(f"  Option {option}: {n_useful}/10 useful pairs (threshold: 3)")

        # ── Dissolve: merge cross-domain adapters into base ────────────────
        log(f"  Dissolving cross-domain adapters into base (scale={PROMOTE_SCALE})...")

        # Reload fresh base for this option
        base_fresh = ToyGPT()
        saved = np.load(str(base_weights_path))
        param_list = []
        for k, v in tree_flatten(base_fresh.parameters()):
            key = k.replace(".", "_")
            param_list.append((k, mx.array(saved[key])))
        base_fresh.load_weights(param_list)
        base_fresh.freeze()
        mx.eval(base_fresh.parameters())

        # Merge all cross-domain adapters
        for (src, tgt), B_cross in cross_B.items():
            slot = cross_slot_map[(src, tgt)]
            merge_adapter_into_base(base_fresh, A_matrices, slot, B_cross, PROMOTE_SCALE)

        # Verify enriched base losses
        enriched_base_losses = {}
        for name in DOMAIN_NAMES:
            el = eval_ntp_loss(base_fresh, domain_data[name]["val"])
            enriched_base_losses[name] = round(el, 4)
        log(f"  Enriched base losses: {enriched_base_losses}")

        # ── Recrystallize: retrain per-domain M2P on enriched base ─────────
        log(f"  Recrystallizing per-domain M2P on enriched base...")

        # Need NEW Grassmannian A for per-domain slots (old cross slots are spent)
        # But per-domain slots 0-4 are unchanged — they were never merged
        # We reuse the same A_matrices for per-domain slots
        recrystal_losses = {}
        recrystal_quality = {}

        for di, name in enumerate(DOMAIN_NAMES):
            mx.random.seed(SEED + 200 + di)
            m2p_new = M2PTransformer()
            mx.eval(m2p_new.parameters())
            m2p_opt = opt.Adam(learning_rate=M2P_LR)

            def m2p_loss_enriched(m2p_model, tokens):
                return m2p_ntp_loss(m2p_model, base_fresh, A_matrices, di, tokens[None, :])

            m2p_vg = nn.value_and_grad(m2p_new, m2p_loss_enriched)
            train_data = domain_data[name]["train"]

            gc.disable()
            for step in range(M2P_STEPS):
                tokens = train_data[step % len(train_data)]
                loss, grads = m2p_vg(m2p_new, tokens)
                m2p_opt.update(m2p_new, grads)
                mx.eval(m2p_new.parameters(), m2p_opt.state, loss)
            gc.enable()

            # Generate B on enriched base
            val_tokens = domain_data[name]["val"][0][None, :]
            hidden = base_fresh.get_hidden_states(val_tokens)
            B_gen = m2p_new(hidden)
            mx.eval(list(B_gen.values()))

            rl = eval_ntp_loss(base_fresh, domain_data[name]["val"], A_matrices, di,
                                {k: mx.array(v) for k, v in B_gen.items()})
            recrystal_losses[name] = round(rl, 4)

            # Quality vs SFT on ORIGINAL base (apples-to-apples)
            quality = (base_losses[name] - rl) / (base_losses[name] - sft_losses[name] + 1e-8)
            improvement_vs_original = (perdomain_m2p_losses[name] - rl) / perdomain_m2p_losses[name] * 100
            recrystal_quality[name] = {
                "enriched_loss": round(rl, 4),
                "original_m2p_loss": perdomain_m2p_losses[name],
                "quality_ratio": round(quality, 4),
                "improvement_vs_original_pct": round(improvement_vs_original, 2),
                "improved": rl < perdomain_m2p_losses[name],
            }
            log(f"    {name}: enriched M2P loss={rl:.4f} (was {perdomain_m2p_losses[name]:.4f}), "
                f"Δ={improvement_vs_original:+.1f}%, quality={quality:.1%}")
            cleanup(m2p_new, m2p_opt)

        n_improved = sum(1 for v in recrystal_quality.values() if v["improved"])
        quality_ratios = [v["quality_ratio"] for v in recrystal_quality.values()]
        median_quality = sorted(quality_ratios)[len(quality_ratios) // 2]

        option_results[option] = {
            "cross_quality": {f"{k[0]}→{k[1]}": v for k, v in cross_quality.items()},
            "n_useful_pairs": n_useful,
            "enriched_base_losses": enriched_base_losses,
            "recrystal_quality": recrystal_quality,
            "n_domains_improved": n_improved,
            "median_quality_ratio": round(median_quality, 4),
        }

        cleanup(base_fresh)

    results["options"] = option_results

    # ── Kill criteria ──────────────────────────────────────────────────────
    log("\n=== Kill Criteria ===")

    # K_A: ≥3/10 useful cross-domain pairs (best option)
    best_useful = max(r["n_useful_pairs"] for r in option_results.values())
    ka_pass = best_useful >= 3
    results["K_A"] = {"pass": ka_pass, "best_useful_pairs": best_useful, "threshold": 3}
    log(f"  K_A (≥3 useful pairs): {'PASS' if ka_pass else 'FAIL'} — {best_useful}/10")

    # K_B: ≥3/5 domains improved on enriched base (best option)
    best_improved = max(r["n_domains_improved"] for r in option_results.values())
    kb_pass = best_improved >= 3
    results["K_B"] = {"pass": kb_pass, "best_domains_improved": best_improved, "threshold": 3}
    log(f"  K_B (≥3 domains improved): {'PASS' if kb_pass else 'FAIL'} — {best_improved}/5")

    # K_C: Grassmannian preservation
    kc_pass = orth_check["max_cos"] < 1e-5
    results["K_C"] = {"pass": kc_pass, "max_cos": orth_check["max_cos"], "threshold": 1e-5}
    log(f"  K_C (Grassmannian |cos| < 1e-5): {'PASS' if kc_pass else 'FAIL'} — {orth_check['max_cos']:.2e}")

    # K_D: Best option median quality >50%
    best_median = max(r["median_quality_ratio"] for r in option_results.values())
    best_option = max(option_results.keys(), key=lambda k: option_results[k]["median_quality_ratio"])
    kd_pass = best_median > 0.5
    results["K_D"] = {"pass": kd_pass, "best_median_quality": best_median, "best_option": best_option, "threshold": 0.5}
    log(f"  K_D (median quality >50%): {'PASS' if kd_pass else 'FAIL'} — Option {best_option} at {best_median:.1%}")

    results["all_pass"] = ka_pass and kb_pass and kc_pass and kd_pass
    results["total_time_s"] = round(time.time() - t0, 1)

    # ── Summary ────────────────────────────────────────────────────────────
    log(f"\n=== Summary ===")
    for opt_name in ["A", "B", "C"]:
        r = option_results[opt_name]
        log(f"  Option {opt_name}: {r['n_useful_pairs']}/10 pairs, "
            f"{r['n_domains_improved']}/5 improved, median={r['median_quality_ratio']:.1%}")
    log(f"  Best: Option {best_option}")
    log(f"  Total time: {results['total_time_s']:.1f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
