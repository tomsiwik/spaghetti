#!/usr/bin/env python3
"""M2P Activation Scaling: How activation-space interference scales with N adapters.

TYPE: guided-exploration
MATH: experiments/models/m2p_activation_scaling/MATH.md

QUESTION: Does activation-space interference stay bounded as N grows?

Parameter-space orthogonality is proven (Grassmannian A). But:
    h_out = W·x + Σ_i B_i(A_i·x)
Even with A_i⊥A_j, B_i(A_i·x) and B_j(A_j·x) can interfere in output space.
Finding #353 measured max|cos| = 0.29 at N=5. We extend to N ∈ {2,3,5,8,10}.

Kill criteria:
  K_activation: max|cos| at N=10 < 0.5
  K_scaling: fitted alpha < 0.5 (sub-linear growth)
  K_quality_N10: composition quality at N=10 >= 80% of best-single

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

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
ADAPTER_DIR = EXPERIMENT_DIR / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
SEED = 42

# ── Architecture constants ─────────────────────────────────────────────────
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 10

D_M2P = 64
N_MEMORY = 32
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

# Training config — reduced in smoke test
BASE_STEPS = 800  if not IS_SMOKE else 30
SFT_STEPS  = 300  if not IS_SMOKE else 20
M2P_STEPS  = 400  if not IS_SMOKE else 20
LR     = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

N_VALUES = [2, 3, 5, 8, 10] if not IS_SMOKE else [2, 3]
N_EVAL_BATCHES = 30 if not IS_SMOKE else 5

DOMAIN_NAMES = [
    "arithmetic", "sort", "reverse", "repeat", "parity",
    "cipher", "counting", "dedup", "mapping", "interleave",
]


# ── Utilities ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):       return bool(o)
        if isinstance(o, (np.integer,)):     return int(o)
        if isinstance(o, (np.floating,)):    return float(o)
        if isinstance(o, np.ndarray):        return o.tolist()
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
    """Generate n text samples for a given domain (10 domains)."""
    chars = "abcdefghij"
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
        elif domain_id == 3:  # repeat
            p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1, 3)))
            r = rng.randint(2, 4)
            data.append(f"{p}*{r}={p*r}")
        elif domain_id == 4:  # parity
            bits = "".join(str(rng.randint(0, 2)) for _ in range(rng.randint(2, 6)))
            data.append(f"{bits}>{'even' if bits.count('1') % 2 == 0 else 'odd'}")
        elif domain_id == 5:  # cipher: caesar +3 shift on a-z
            s = "".join(rng.choice(list("abcdefghij")) for _ in range(rng.randint(2, 5)))
            shifted = "".join(chr((ord(c) - ord('a') + 3) % 26 + ord('a')) for c in s)
            data.append(f"{s}>{shifted}")
        elif domain_id == 6:  # counting: count occurrences of each char
            s = "".join(rng.choice(list("abcd")) for _ in range(rng.randint(3, 6)))
            from collections import Counter
            cnt = Counter(s)
            result = "".join(f"{c}{cnt[c]}" for c in sorted(cnt))
            data.append(f"{s}>{result}")
        elif domain_id == 7:  # dedup: remove consecutive duplicates
            s = "".join(rng.choice(list("abcd")) for _ in range(rng.randint(3, 7)))
            deduped = s[0]
            for c in s[1:]:
                if c != deduped[-1]:
                    deduped += c
            data.append(f"{s}>{deduped}")
        elif domain_id == 8:  # mapping: fixed char swap a<->z, b<->y, c<->x etc
            s = "".join(rng.choice(list("abcdefghij")) for _ in range(rng.randint(2, 5)))
            mapped = "".join(chr(ord('a') + 25 - (ord(c) - ord('a'))) for c in s)
            data.append(f"{s}>{mapped}")
        elif domain_id == 9:  # interleave: interleave with 1-indexed numbers
            s = "".join(rng.choice(list("abcde")) for _ in range(rng.randint(2, 4)))
            interleaved = "".join(f"{c}{i+1}" for i, c in enumerate(s))
            data.append(f"{s}>{interleaved}")
    return data


def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]


def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]


# ── Toy GPT ────────────────────────────────────────────────────────────────

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


# ── Grassmannian A-matrices ────────────────────────────────────────────────

def generate_grassmannian_A(n_domains: int, n_layers: int, n_modules: int,
                             d: int, rank: int, seed: int = 42) -> dict:
    """Generate frozen orthogonal A-matrices via QR decomposition.

    Capacity: n_domains * rank = 10 * 4 = 40 <= d=256 (6.4x margin).
    Returns: dict[(domain_idx, layer_idx, module_idx)] -> mx.array(d, rank)
    """
    total_rank = n_domains * rank
    assert total_rank <= d, f"Capacity violated: need {total_rank} but d={d}"

    rng = np.random.RandomState(seed)
    A_matrices = {}

    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)  # Q: (d, total_rank), Q^T Q = I
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start + rank])

    return A_matrices


# ── LoRA forward pass ─────────────────────────────────────────────────────

def lora_forward_with_B(base: ToyGPT, tokens: mx.array,
                         A_matrices: dict, domain_id: int,
                         B_matrices: dict) -> mx.array:
    """Forward pass with LoRA adapter for a single domain."""
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
        sf = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * sf + mask, axis=-1)
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


def sft_loss_fn(b_container: BMatrices, base: ToyGPT, tokens: mx.array,
                A_matrices: dict, domain_id: int) -> mx.array:
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ── M2P Transformer ───────────────────────────────────────────────────────

class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)
        # MLP
        x = x + self.fc2(nn.gelu(self.fc1(self.norm2(x))))
        return x


class M2PTransformer(nn.Module):
    """Memory-to-Parameter Transformer (single domain)."""

    def __init__(self, d_base: int = D_MODEL, d_m2p: int = D_M2P):
        super().__init__()
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
        # Encode base hidden states -> context vector
        layer_encs = []
        for h in hidden_states_list:
            pooled = mx.mean(h[0], axis=0)  # (D_BASE,)
            layer_encs.append(self.input_proj(pooled))  # (D_M2P,)

        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)  # (N_MEMORY, D_M2P)
        context = mx.mean(mx.stack(layer_encs, axis=0), axis=0)  # (D_M2P,)
        memory = memory + context[None, :]  # (N_MEMORY, D_M2P)

        x = memory[None, :, :]  # (1, N_MEMORY, D_M2P)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        pooled_mem = mx.mean(x[0], axis=0)  # (D_M2P,)

        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pooled_mem)  # (N_LAYERS * rank * d_out,)
            out = out.reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS):
                B_matrices[(li, mi)] = out[li]
        return B_matrices


def m2p_ntp_loss(m2p: M2PTransformer, base: ToyGPT,
                 A_matrices: dict, domain_id: int, tokens: mx.array) -> mx.array:
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ── Evaluation helpers ─────────────────────────────────────────────────────

def eval_ntp_loss(base: ToyGPT, batches: list,
                  A_matrices: dict = None, domain_id: int = None,
                  B_matrices: dict = None) -> float:
    total = 0.0
    n = 0
    for tokens in batches[:N_EVAL_BATCHES]:
        tokens_2d = tokens[None, :]
        if A_matrices is not None and B_matrices is not None:
            logits = lora_forward_with_B(base, tokens_2d, A_matrices, domain_id, B_matrices)
        else:
            logits = base(tokens_2d)
        loss = nn.losses.cross_entropy(logits[:, :-1], tokens_2d[:, 1:], reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def load_B_matrices(path: str) -> dict:
    """Load B-matrices from .npz file."""
    data = np.load(path)
    B_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(N_MODULES):
            B_matrices[(li, mi)] = mx.array(data[f"{li}_{mi}"])
    return B_matrices


# ── Activation-space cosine measurement ──────────────────────────────────

def _per_token_cosines_for_module(x_norm: mx.array, A_matrices: dict,
                                   B_list: list, slot_ids: list,
                                   li: int, mi: int) -> dict:
    """Compute per-token pairwise cosines for one (layer, module) pair.

    For each adapter pair (i, j) and each token position t, computes:
        |cos(B_i A_i x_t, B_j A_j x_t)|
    where x_t is the (d,) vector at position t.
    Reports the max and mean over all (pair, token) combinations.

    Args:
      x_norm: (1, T, D) normalized hidden states
      A_matrices: dict[(domain_id, li, mi)] -> (d, rank)
      B_list: list of B-matrix dicts, one per slot_id
      slot_ids: list of domain indices
      li: layer index
      mi: module index

    Returns: dict with max_cos, mean_cos, n_pairs
    """
    T = x_norm.shape[1]

    # Compute adapter output deltas: each is (1, T, d_out)
    activations = []
    for idx, slot_id in enumerate(slot_ids):
        A = A_matrices[(slot_id, li, mi)]   # (d, rank)
        B = B_list[idx][(li, mi)]            # (rank, d_out)
        delta = (x_norm @ A) @ B             # (1, T, d_out)
        activations.append(delta)

    # Force eval before computing cosines
    mx.eval(*activations)

    # Convert to numpy for efficient per-token iteration (eval already forced above)
    acts_np = [np.array(a[0])  for a in activations]  # list of (T, d_out) float32
    del activations

    cos_values = []
    for i in range(len(acts_np)):
        for j in range(i + 1, len(acts_np)):
            # Per-token cosine: iterate over each token position t
            for t in range(T):
                ai_t = acts_np[i][t]   # (d_out,)
                aj_t = acts_np[j][t]   # (d_out,)
                ni = float(np.linalg.norm(ai_t))
                nj = float(np.linalg.norm(aj_t))
                # Skip zero-norm tokens (padding or dead neurons)
                if ni < 1e-12 or nj < 1e-12:
                    continue
                cos = float(abs(np.dot(ai_t, aj_t) / (ni * nj)))
                cos_values.append(cos)

    del acts_np
    return {
        "max_cos": float(max(cos_values)) if cos_values else 0.0,
        "mean_cos": float(sum(cos_values) / len(cos_values)) if cos_values else 0.0,
        "n_pairs": len(cos_values),
    }


def measure_activation_cos(base: ToyGPT, tokens: mx.array,
                             A_matrices: dict, B_list: list,
                             slot_ids: list) -> dict:
    """Measure per-token pairwise activation-space cosine between adapters.

    Measures at layer 0 for two representative modules:
      - wq (mi=0): d_out = D_MODEL = 256
      - fc1 (mi=4): d_out = 4 * D_MODEL = 1024

    For each pair (i,j) and each token position t, computes:
        activation_cos(i,j,t) = |cos(B_i A_i x_t, B_j A_j x_t)|
    Reports max over all (pair, token) combinations per module.

    Args:
      base: frozen ToyGPT
      tokens: (1, T) int32
      A_matrices: dict[(domain_id, layer_idx, module_idx)] -> (d, rank)
      B_list: list of B-matrix dicts, one per slot_id
      slot_ids: list of domain indices to compare

    Returns: dict with max_cos, mean_cos, n_pairs (worst-case across modules),
             and per_module breakdown for wq and fc1.
    """
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    li = 0
    block = base.blocks[li]

    # wq uses norm1(x) as input
    x_norm1 = block.norm1(x)  # (1, T, D)
    mx.eval(x_norm1)

    # fc1 uses norm2(x + attn(x)) as input — run the base attention forward
    # (no LoRA, we only need the residual stream position)
    attn_out = block.attn(x_norm1)          # (1, T, D)
    x_post_attn = x + attn_out              # (1, T, D)
    x_norm2 = block.norm2(x_post_attn)     # (1, T, D)
    mx.eval(x_norm2)
    del attn_out, x_post_attn

    # Measure wq (mi=0) and fc1 (mi=4) separately with their correct inputs
    wq_result  = _per_token_cosines_for_module(x_norm1, A_matrices, B_list, slot_ids, li, mi=0)
    fc1_result = _per_token_cosines_for_module(x_norm2, A_matrices, B_list, slot_ids, li, mi=4)

    del x_norm1, x_norm2, x

    # Report the worst case across both modules as the top-level metric
    max_cos = max(wq_result["max_cos"], fc1_result["max_cos"])
    n_total = wq_result["n_pairs"] + fc1_result["n_pairs"]
    mean_cos = (
        (wq_result["mean_cos"] * wq_result["n_pairs"] +
         fc1_result["mean_cos"] * fc1_result["n_pairs"]) / max(n_total, 1)
    )

    return {
        "max_cos": float(max_cos),
        "mean_cos": float(mean_cos),
        "n_pairs": n_total,
        "per_module": {
            "wq":  wq_result,
            "fc1": fc1_result,
        },
    }


def composed_forward_no_router(base: ToyGPT, tokens: mx.array,
                                A_matrices: dict, all_B: list,
                                slot_ids: list) -> mx.array:
    """Equal-weight composed forward (1/N per adapter)."""
    N = len(slot_ids)
    w = 1.0 / N

    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        def _apply_composed(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            lora_sum = mx.zeros_like(base_out)
            for idx, slot_id in enumerate(slot_ids):
                A = A_matrices[(slot_id, li, mi)]
                B = all_B[idx][(li, mi)]
                lora_sum = lora_sum + w * LORA_SCALE * (x_in @ A) @ B
            return base_out + lora_sum

        q = _apply_composed(attn.wq, x_norm, li, 0)
        k = _apply_composed(attn.wk, x_norm, li, 1)
        v = _apply_composed(attn.wv, x_norm, li, 2)

        q = q.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)

        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        sf = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * sf + mask, axis=-1)
        attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(B_b, T_b, C)

        attn_out = _apply_composed(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        x_norm2 = block.norm2(x)
        fc1_base = block.mlp.fc1(x_norm2)
        lora_fc1 = mx.zeros_like(fc1_base)
        for idx, slot_id in enumerate(slot_ids):
            A = A_matrices[(slot_id, li, 4)]
            B = all_B[idx][(li, 4)]
            lora_fc1 = lora_fc1 + w * LORA_SCALE * (x_norm2 @ A) @ B
        fc1_out = fc1_base + lora_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    return base.lm_head(base.norm_f(x))


def fit_power_law(n_values: list, max_cos_values: list) -> dict:
    """Fit max_cos ~ c * N^alpha via log-log linear regression."""
    if len(n_values) < 2:
        return {"alpha": 0.0, "c": 0.0, "r_squared": 0.0}

    log_n = np.log(np.array(n_values, dtype=float))
    log_y = np.log(np.clip(np.array(max_cos_values, dtype=float), 1e-8, None))

    # Linear regression: log_y = alpha * log_n + log_c
    A_reg = np.column_stack([log_n, np.ones_like(log_n)])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_reg, log_y, rcond=None)
    alpha = float(coeffs[0])
    c = float(np.exp(coeffs[1]))

    # R^2
    y_pred = alpha * log_n + np.log(c)
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = float(1 - ss_res / (ss_tot + 1e-12))

    return {"alpha": round(alpha, 4), "c": round(c, 6), "r_squared": round(r_squared, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def phase_generate_data(rng: np.random.RandomState) -> dict:
    """Generate train/val data for all 10 domains."""
    log("=== Phase 0: Generate Data ===")
    n_per_domain = 400 if not IS_SMOKE else 40
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        domain_data[name] = {
            "train": make_batches(texts[:split]),
            "val": make_batches(texts[split:]),
            "domain_id": di,
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, "
            f"{len(domain_data[name]['val'])} val")
    return domain_data


def phase_pretrain_base(domain_data: dict) -> tuple:
    """Pre-train ToyGPT on all 10 domains. Returns (base, base_losses, weights_path)."""
    log("\n=== Phase 1: Pre-train Base Model ===")
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
        return nn.losses.cross_entropy(logits[:, :-1], tokens_2d[:, 1:], reduction="mean")

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

    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")

    # Save base weights
    weights_path = EXPERIMENT_DIR / "base_weights.npz"
    weights_dict = {}
    for k, v in tree_flatten(base.parameters()):
        weights_dict[k.replace(".", "_")] = np.array(v)
    np.savez(str(weights_path), **weights_dict)
    log(f"  Saved base weights -> {weights_path}")

    cleanup(optimizer)
    return base, base_losses, str(weights_path)


def phase_sft_domain(domain_name: str, domain_id: int,
                      domain_data: dict, base: ToyGPT,
                      A_matrices: dict, base_loss: float) -> dict:
    """Train SFT LoRA adapter for one domain."""
    log(f"  SFT {domain_name} (domain {domain_id})...")

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

    B_matrices = b_container.as_dict()
    sft_loss = eval_ntp_loss(base, domain_data["val"], A_matrices, domain_id, B_matrices)
    log(f"    loss={sft_loss:.4f} (base={base_loss:.4f})")

    save_path = ADAPTER_DIR / f"sft_{domain_name}.npz"
    np_dict = {f"{li}_{mi}": np.array(getattr(b_container, f"B_{li}_{mi}"))
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(save_path), **np_dict)

    cleanup(optimizer, b_container)
    return {"sft_loss": round(sft_loss, 4), "save_path": str(save_path)}


def phase_sft_all_domains(domain_data: dict, base: ToyGPT,
                           A_matrices: dict, base_losses: dict) -> dict:
    """Train SFT adapters for all 10 domains."""
    log("\n=== Phase 2: SFT Adapters (all 10 domains) ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(name, di, domain_data[name], base,
                                   A_matrices, base_losses[name])
        sft_results[name] = result
        log_memory(f"sft-{name}")
    return sft_results


def phase_m2p_domain(domain_name: str, domain_id: int,
                      domain_data: dict, base: ToyGPT,
                      A_matrices: dict, base_loss: float, sft_loss: float) -> dict:
    """Train M2P for ONE domain independently."""
    log(f"  M2P {domain_name} (domain {domain_id})...")

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P)
    mx.eval(m2p.parameters())
    m2p_param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"    params: {m2p_param_count:,}")

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id, tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]

    gc.disable()
    for step in range(M2P_STEPS):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        if (step + 1) % max(1, M2P_STEPS // 4) == 0:
            log(f"    Step {step+1}/{M2P_STEPS}: loss={loss.item():.4f}")
    gc.enable()

    # Evaluate on validation set
    total_loss = 0.0
    n_eval = 0
    for tokens in domain_data["val"][:N_EVAL_BATCHES]:
        l = m2p_ntp_loss(m2p, base, A_matrices, domain_id, tokens[None, :])
        mx.eval(l)
        total_loss += l.item()
        n_eval += 1
        del l
    m2p_val_loss = total_loss / max(n_eval, 1)

    quality_ratio = 0.0
    if (base_loss - sft_loss) > 0.01:
        quality_ratio = (base_loss - m2p_val_loss) / (base_loss - sft_loss)
    log(f"    val_loss={m2p_val_loss:.4f} SFT={sft_loss:.4f} quality={quality_ratio:.1%}")

    # Generate and save B-matrices from representative context
    context_tokens = domain_data["train"][0][None, :]
    hidden_states = base.get_hidden_states(context_tokens)
    B_matrices = m2p(hidden_states)
    mx.eval(*[B_matrices[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])

    save_path = ADAPTER_DIR / f"m2p_{domain_name}.npz"
    np_dict = {f"{li}_{mi}": np.array(B_matrices[(li, mi)])
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(save_path), **np_dict)

    cleanup(optimizer, m2p)
    return {
        "m2p_loss": round(m2p_val_loss, 4),
        "quality_ratio": round(quality_ratio, 3),
        "save_path": str(save_path),
    }


def phase_m2p_all_domains(domain_data: dict, base: ToyGPT,
                           A_matrices: dict, base_losses: dict,
                           sft_results: dict) -> dict:
    """Train M2P for all 10 domains."""
    log("\n=== Phase 3: M2P Training (all 10 domains) ===")
    m2p_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_m2p_domain(
            name, di, domain_data[name], base, A_matrices,
            base_losses[name], sft_results[name]["sft_loss"]
        )
        m2p_results[name] = result
        log_memory(f"m2p-{name}")
    return m2p_results


def phase_activation_scaling(domain_data: dict, base: ToyGPT,
                               A_matrices: dict, m2p_results: dict,
                               sft_results: dict, base_losses: dict) -> dict:
    """Measure activation-space interference scaling across N values.

    For each N in N_VALUES:
      1. Load first N adapter B-matrices from disk
      2. Measure pairwise activation cosines
      3. Measure per-domain composition quality
    Then fit power law.
    """
    log("\n=== Phase 4: Activation Scaling Measurement ===")

    per_n_results = {}

    for N in N_VALUES:
        log(f"\n  N={N}:")
        slot_ids = list(range(N))
        domain_subset = [DOMAIN_NAMES[i] for i in slot_ids]

        # Load B-matrices for this N
        B_list = []
        for name in domain_subset:
            path = m2p_results[name]["save_path"]
            B_mats = load_B_matrices(path)
            mx.eval(*[B_mats[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])
            B_list.append(B_mats)

        # --- Activation cosine measurement ---
        # Collect across multiple test tokens (batched: run per domain's val set)
        all_cos_results = []
        n_test_batches = min(N_EVAL_BATCHES, len(domain_data[domain_subset[0]]["val"]))
        for bi in range(n_test_batches):
            # Use first domain's validation tokens as representative input
            tokens = domain_data[domain_subset[0]]["val"][bi][None, :]
            cos_result = measure_activation_cos(base, tokens, A_matrices, B_list, slot_ids)
            all_cos_results.append(cos_result)
            del tokens

        max_cos_over_batches = max(r["max_cos"] for r in all_cos_results)
        mean_cos_over_batches = sum(r["mean_cos"] for r in all_cos_results) / len(all_cos_results)
        n_pairs = all_cos_results[0]["n_pairs"]

        # Per-module worst-case (wq vs fc1) — take max over batches per module
        wq_max  = max(r["per_module"]["wq"]["max_cos"]  for r in all_cos_results)
        fc1_max = max(r["per_module"]["fc1"]["max_cos"] for r in all_cos_results)
        wq_mean  = sum(r["per_module"]["wq"]["mean_cos"]  for r in all_cos_results) / len(all_cos_results)
        fc1_mean = sum(r["per_module"]["fc1"]["mean_cos"] for r in all_cos_results) / len(all_cos_results)

        log(f"    Activation cos (per-token max): max={max_cos_over_batches:.4f} "
            f"mean={mean_cos_over_batches:.4f} ({n_pairs} (pair,token) obs per batch)")
        log(f"      wq  (d_out=256):  max={wq_max:.4f}  mean={wq_mean:.4f}")
        log(f"      fc1 (d_out=1024): max={fc1_max:.4f}  mean={fc1_mean:.4f}")

        # --- Composition quality per domain ---
        domain_quality = {}
        for di, name in enumerate(domain_subset):
            sft_loss = sft_results[name]["sft_loss"]
            base_loss = base_losses[name]
            total_loss = 0.0
            n = 0
            for tokens in domain_data[name]["val"][:N_EVAL_BATCHES]:
                tokens_2d = tokens[None, :]
                logits = composed_forward_no_router(
                    base, tokens_2d, A_matrices, B_list, slot_ids
                )
                loss = nn.losses.cross_entropy(
                    logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
                )
                mx.eval(loss)
                total_loss += loss.item()
                n += 1
                del logits, loss

            comp_loss = total_loss / max(n, 1)
            # Quality: fraction of SFT improvement retained
            quality_frac = 0.0
            if (base_loss - sft_loss) > 0.01:
                quality_frac = (base_loss - comp_loss) / (base_loss - sft_loss)
            domain_quality[name] = {
                "comp_loss": round(comp_loss, 4),
                "sft_loss": sft_loss,
                "quality_frac": round(quality_frac, 4),
            }
            log(f"    {name}: comp={comp_loss:.4f} SFT={sft_loss:.4f} "
                f"quality={quality_frac:.1%}")

        # Best single quality (best adapter quality across the N domains)
        quality_fracs = [v["quality_frac"] for v in domain_quality.values()]
        mean_quality = float(np.mean(quality_fracs))
        min_quality = float(np.min(quality_fracs))

        # Clean up B_list for this N
        del B_list
        gc.collect()
        mx.clear_cache()

        per_n_results[N] = {
            "max_cos": round(max_cos_over_batches, 6),
            "mean_cos": round(mean_cos_over_batches, 6),
            "n_pairs": n_pairs,
            "per_module": {
                "wq":  {"max_cos": round(wq_max, 6),  "mean_cos": round(wq_mean, 6)},
                "fc1": {"max_cos": round(fc1_max, 6), "mean_cos": round(fc1_mean, 6)},
            },
            "mean_quality": round(mean_quality, 4),
            "min_quality": round(min_quality, 4),
            "domain_quality": domain_quality,
        }
        log(f"    Summary N={N}: max_cos={max_cos_over_batches:.4f} "
            f"mean_quality={mean_quality:.1%}")

    # Fit power law over N_VALUES x max_cos_values
    n_vals = list(per_n_results.keys())
    max_cos_vals = [per_n_results[N]["max_cos"] for N in n_vals]
    power_law = fit_power_law(n_vals, max_cos_vals)
    log(f"\n  Power law fit: max_cos ~ {power_law['c']:.4f} * N^{power_law['alpha']:.4f} "
        f"(R²={power_law['r_squared']:.4f})")

    return {
        "per_n": per_n_results,
        "power_law": power_law,
    }


def assess_kill_criteria(scaling_results: dict) -> dict:
    """Assess kill criteria from MATH.md."""
    per_n = scaling_results["per_n"]
    power_law = scaling_results["power_law"]

    # K_activation: max|cos| at N=10 < 0.5
    max_n = max(per_n.keys())
    max_cos_at_maxN = per_n[max_n]["max_cos"]
    k_activation_pass = max_cos_at_maxN < 0.5

    # K_scaling: fitted alpha < 0.5
    alpha = power_law["alpha"]
    k_scaling_pass = alpha < 0.5

    # K_quality_N10: composition quality at N=10 >= 80% of best-single
    min_quality_at_maxN = per_n[max_n]["min_quality"]
    k_quality_pass = min_quality_at_maxN >= 0.80

    log(f"\n=== Kill Criteria Assessment ===")
    log(f"  K_activation (max|cos| at N={max_n} < 0.5): "
        f"max_cos={max_cos_at_maxN:.4f} -> {'PASS' if k_activation_pass else 'FAIL'}")
    log(f"  K_scaling (alpha < 0.5): "
        f"alpha={alpha:.4f} -> {'PASS' if k_scaling_pass else 'FAIL'}")
    log(f"  K_quality_N{max_n} (min_quality >= 80%): "
        f"min_quality={min_quality_at_maxN:.1%} -> {'PASS' if k_quality_pass else 'FAIL'}")

    return {
        "k_activation": {
            "pass": k_activation_pass,
            "max_cos_at_n": {str(max_n): max_cos_at_maxN},
            "threshold": 0.5,
        },
        "k_scaling": {
            "pass": k_scaling_pass,
            "alpha": alpha,
            "threshold": 0.5,
        },
        "k_quality": {
            "pass": k_quality_pass,
            "min_quality_at_n": {str(max_n): min_quality_at_maxN},
            "threshold": 0.80,
        },
        "all_pass": k_activation_pass and k_scaling_pass and k_quality_pass,
    }


def main():
    t0 = time.time()
    log(f"=== M2P Activation Scaling ===")
    log(f"SMOKE_TEST={IS_SMOKE}, N_VALUES={N_VALUES}, N_DOMAINS={N_DOMAINS}")
    log_memory("start")

    rng = np.random.RandomState(SEED)

    # Phase 0: Data
    domain_data = phase_generate_data(rng)
    log_memory("after-data")

    # Phase 1: Pre-train base model
    base, base_losses, weights_path = phase_pretrain_base(domain_data)
    log_memory("after-pretrain")

    # Generate Grassmannian A-matrices
    log("\n=== Grassmannian A-matrices ===")
    A_matrices = generate_grassmannian_A(N_DOMAINS, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK, SEED)
    total_rank = N_DOMAINS * LORA_RANK
    log(f"  Capacity: {N_DOMAINS}×{LORA_RANK}={total_rank} / d={D_MODEL} "
        f"({D_MODEL/total_rank:.1f}x margin)")

    # Phase 2: SFT adapters for all 10 domains
    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after-sft")

    # Phase 3: M2P adapters for all 10 domains
    m2p_results = phase_m2p_all_domains(domain_data, base, A_matrices, base_losses, sft_results)
    log_memory("after-m2p")

    # Phase 4: Activation scaling measurement
    scaling_results = phase_activation_scaling(
        domain_data, base, A_matrices, m2p_results, sft_results, base_losses
    )
    log_memory("after-scaling")

    # Assess kill criteria
    kill_criteria = assess_kill_criteria(scaling_results)

    total_time = round(time.time() - t0, 1)
    log(f"\nTotal time: {total_time}s")

    results = {
        "experiment": "m2p_activation_scaling",
        "smoke_test": IS_SMOKE,
        "n_domains": N_DOMAINS,
        "n_values": N_VALUES,
        "d_model": D_MODEL,
        "lora_rank": LORA_RANK,
        "base_losses": base_losses,
        "sft_results": {k: {"sft_loss": v["sft_loss"]} for k, v in sft_results.items()},
        "m2p_results": {k: {
            "m2p_loss": v["m2p_loss"],
            "quality_ratio": v["quality_ratio"],
        } for k, v in m2p_results.items()},
        "scaling": scaling_results,
        "kill_criteria": kill_criteria,
        "total_time_s": total_time,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Print summary table
    log("\n=== Prediction vs Measurement (per-token cosine) ===")
    log(f"  {'N':>4} | {'pred max_cos':>14} | {'max_cos(wq)':>12} | {'max_cos(fc1)':>13} | {'min_quality':>12}")
    log(f"  {'-'*4}-+-{'-'*14}-+-{'-'*12}-+-{'-'*13}-+-{'-'*12}")
    predictions = {2: "0.10-0.20", 5: "0.20-0.35", 10: "0.30-0.50"}
    per_n = scaling_results["per_n"]
    for N in N_VALUES:
        pred = predictions.get(N, "—")
        meas = per_n[N]["max_cos"]
        wq_m  = per_n[N]["per_module"]["wq"]["max_cos"]
        fc1_m = per_n[N]["per_module"]["fc1"]["max_cos"]
        qual  = per_n[N]["min_quality"]
        log(f"  {N:>4} | {pred:>14} | {wq_m:>12.4f} | {fc1_m:>13.4f} | {qual:>12.1%}")
    pl = scaling_results["power_law"]
    log(f"\n  Power law: max_cos ~ {pl['c']:.4f} * N^{pl['alpha']:.4f} (R²={pl['r_squared']:.4f})")
    log(f"  Predicted alpha: 0.3-0.5")

    cleanup(base)
    log_memory("final")


if __name__ == "__main__":
    main()
