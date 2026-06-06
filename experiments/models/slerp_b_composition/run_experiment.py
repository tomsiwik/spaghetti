#!/usr/bin/env python3
"""SLERP B-matrix composition: geodesic blending prevents candy-wrapper activation collapse.

TYPE: verification
MATH: experiments/models/slerp_b_composition/MATH.md

THEOREM: For normalized B matrices with angle θ between them:
  ||LERP(B̂₁, B̂₂, 0.5)||_F = sqrt((1+cosθ)/2) ≤ 1   [collapses for diverse B]
  ||SLERP(B̂₁, B̂₂, 0.5)||_F = 1                       [always preserved]

Kill criteria:
  K931: SLERP norm ratio / LERP norm ratio > 1.30 at N=5 (>30% strength advantage)
  K932: Quality under SLERP composition >= quality under LERP composition

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
from pathlib import Path
from functools import partial

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
RESULTS_FILE   = EXPERIMENT_DIR / "results.json"

IS_SMOKE   = os.environ.get("SMOKE_TEST") == "1"
SEED       = 42

# ── Architecture ────────────────────────────────────────────────────────────
D_MODEL    = 256
N_LAYERS   = 2
N_HEADS    = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK  = 4
LORA_SCALE = 1.0
N_DOMAINS  = 5
DOMAIN_NAMES = ["arithmetic", "sort", "reverse", "repeat", "parity"]
FF_DIM     = 4 * D_MODEL

# Modules we apply LoRA to (name, in_dim, out_dim)
LORA_MODULES = [("wq", D_MODEL, D_MODEL), ("fc1", D_MODEL, FF_DIM)]
N_PARAMS     = len(LORA_MODULES) * N_LAYERS   # 4

# Training config
BASE_STEPS = 600 if not IS_SMOKE else 30
SFT_STEPS  = 300 if not IS_SMOKE else 20
LR         = 3e-4
SFT_LR     = 1e-3
N_EVAL     = 20  if not IS_SMOKE else 4
BATCH_SIZE = 16


# ── Utilities ────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):  return bool(o)
        if isinstance(o, np.integer):   return int(o)
        if isinstance(o, np.floating):  return float(o)
        if isinstance(o, np.ndarray):   return o.tolist()
        return super().default(o)


def log(m): print(m, flush=True)

def cleanup(*objs):
    for o in objs: del o
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ── Data generation ──────────────────────────────────────────────────────────
def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    chars = "abcdefghij"
    data  = []
    for _ in range(n):
        d = domain_id
        if d == 0:  # arithmetic
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            data.append(f"{a}+{b}={a+b}")
        elif d == 1:  # sort
            L   = rng.randint(3, 7)
            seq = list(rng.choice(list(chars), L, replace=False))
            data.append("".join(seq) + ">" + "".join(sorted(seq)))
        elif d == 2:  # reverse
            L   = rng.randint(3, 7)
            seq = list(rng.choice(list(chars), L, replace=False))
            data.append("".join(seq) + ">" + "".join(reversed(seq)))
        elif d == 3:  # repeat
            L   = rng.randint(3, 6)
            seq = "".join(rng.choice(list(chars), L, replace=True))
            data.append(seq + ">" + seq * 2)
        else:  # parity
            L   = rng.randint(4, 8)
            seq = "".join(rng.choice(list(chars), L, replace=True))
            p   = seq.count("a") % 2
            data.append(seq + ">" + str(p))
    return data


def tokenize(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text]


def make_batch(samples: list, rng: np.random.RandomState) -> tuple:
    xs, ys = [], []
    for s in samples:
        toks = tokenize(s)
        if len(toks) < 2:
            continue
        toks = toks[:BLOCK_SIZE + 1]
        while len(toks) < BLOCK_SIZE + 1:
            toks += toks
        toks = toks[:BLOCK_SIZE + 1]
        xs.append(toks[:-1])
        ys.append(toks[1:])
    return mx.array(np.array(xs, dtype=np.int32)), mx.array(np.array(ys, dtype=np.int32))


# ── Base Transformer ─────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def __call__(self, x, B_wq=None):
        """B_wq: optional LoRA B for wq (out, rank). A is stored externally."""
        B, T, _ = x.shape
        # wq with optional LoRA (A applied externally via functional pattern)
        q = self.wq(x)
        if B_wq is not None:
            # B_wq: (D_MODEL, rank) — we receive (B@A) delta directly
            q = q  # delta already incorporated via weight patching below
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.tril(mx.ones((T, T))).reshape(1, 1, T, T)
        att  = mx.where(mask == 0, mx.array(-1e9, dtype=mx.float32), att)
        att  = mx.softmax(att.astype(mx.float32), axis=-1)
        y    = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)
        return self.wo(y)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attn  = CausalSelfAttention()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.fc1   = nn.Linear(D_MODEL, FF_DIM, bias=False)
        self.fc2   = nn.Linear(FF_DIM, D_MODEL, bias=False)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        h = nn.gelu(self.fc1(self.norm2(x)))
        return x + self.fc2(h)


class MicroTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos    = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = [TransformerBlock() for _ in range(N_LAYERS)]
        self.norm_f = nn.LayerNorm(D_MODEL)
        self.head   = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def __call__(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(mx.arange(T))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_f(h))


def model_loss(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T), reduction="mean")


# ── LoRA B-matrix module ─────────────────────────────────────────────────────
class BModule(nn.Module):
    """Holds trainable B matrices for one domain (all layers × modules)."""
    def __init__(self, init_dict: dict):
        """init_dict: {(l, name): np.array(out, rank)}"""
        super().__init__()
        for (l, name), v in init_dict.items():
            setattr(self, f"b_{l}_{name}", mx.array(v))

    def as_dict(self) -> dict:
        """Return {(l, name): mx.array}"""
        return {(l, name): getattr(self, f"b_{l}_{name}")
                for l in range(N_LAYERS) for name, _, _ in LORA_MODULES}

    def to_numpy(self) -> dict:
        return {(l, name): np.array(getattr(self, f"b_{l}_{name}"))
                for l in range(N_LAYERS) for name, _, _ in LORA_MODULES}


# ── Functional LoRA forward ───────────────────────────────────────────────────
def lora_forward_loss(model: MicroTransformer,
                      A_np: dict,          # {(l, name): np.array (rank, in)}
                      b_mod: BModule,
                      x: mx.array, y: mx.array) -> mx.array:
    """Loss with LoRA injection.

    Uses functional pattern: computes ΔW = B @ A and adds to embedding
    BEFORE linear ops (weight patching via mx addition, not attribute mutation).

    This keeps B in the computation graph for autodiff.
    """
    Bdict = b_mod.as_dict()   # {(l,name): mx.array (out, rank)}

    # Pre-compute delta weights as MLX arrays
    # delta[(l, name)] shape: (out, in) — same as linear weight
    delta = {}
    for (l, name), B in Bdict.items():
        A = mx.array(A_np[(l, name)])   # (rank, in)
        delta[(l, name)] = B @ A        # (out, in)

    # Forward pass with patched weights
    B_batch, T = x.shape
    h = model.embed(x) + model.pos(mx.arange(T))

    for l, blk in enumerate(model.blocks):
        # Attention with patched wq
        normed1 = blk.norm1(h)
        dW_wq = delta.get((l, "wq"))
        if dW_wq is not None:
            # q = normed1 @ (W_wq + ΔW).T = base_q + normed1 @ ΔW.T
            q = blk.attn.wq(normed1) + normed1 @ dW_wq.T
        else:
            q = blk.attn.wq(normed1)
        nh, hd = blk.attn.n_heads, blk.attn.head_dim
        q = q.reshape(B_batch, T, nh, hd).transpose(0, 2, 1, 3)
        k = blk.attn.wk(normed1).reshape(B_batch, T, nh, hd).transpose(0, 2, 1, 3)
        v = blk.attn.wv(normed1).reshape(B_batch, T, nh, hd).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) * blk.attn.scale
        mask = mx.tril(mx.ones((T, T))).reshape(1, 1, T, T)
        att  = mx.where(mask == 0, mx.array(-1e9, dtype=mx.float32), att)
        att  = mx.softmax(att.astype(mx.float32), axis=-1)
        attn_out = (att @ v).transpose(0, 2, 1, 3).reshape(B_batch, T, D_MODEL)
        attn_out = blk.attn.wo(attn_out)
        h = h + attn_out

        # FFN with patched fc1
        normed2 = blk.norm2(h)
        dW_fc1 = delta.get((l, "fc1"))
        if dW_fc1 is not None:
            fc1_out = blk.fc1(normed2) + normed2 @ dW_fc1.T
        else:
            fc1_out = blk.fc1(normed2)
        h = h + blk.fc2(nn.gelu(fc1_out))

    logits = model.head(model.norm_f(h))
    B2, T2, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(B2*T2, V), y.reshape(B2*T2), reduction="mean")


# ── SLERP utilities ──────────────────────────────────────────────────────────
def slerp_vectors(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit vectors in R^n (operates on already-normalized inputs)."""
    cos_theta = float(np.clip(np.dot(v1.flatten(), v2.flatten()), -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-7:
        return (1.0 - t) * v1 + t * v2
    sin_t = math.sin(theta)
    return (math.sin((1-t)*theta) / sin_t) * v1 + (math.sin(t*theta) / sin_t) * v2


def compose_B_matrices(B_list: list, weights: list = None, method: str = "slerp") -> np.ndarray:
    """Compose N B-matrices using SLERP or LERP.

    Both methods: scale = weighted sum of Frobenius norms.
    LERP: direction = weighted sum of unit B-matrices (may collapse).
    SLERP: direction = iterative geodesic interpolation (preserves unit norm).
    """
    N = len(B_list)
    if N == 1:
        return B_list[0].copy()
    if weights is None:
        weights = [1.0 / N] * N

    norms = [np.linalg.norm(B, "fro") for B in B_list]
    scale = sum(w * n for w, n in zip(weights, norms))

    # Normalize to unit Frobenius sphere
    B_hats = [B / (n + 1e-10) for B, n in zip(B_list, norms)]
    shape  = B_list[0].shape

    if method == "lerp":
        # Weighted average of unit B-matrices, then scale.
        # Theorem 1: ||Σ w_i B̂_i||_F = sqrt(1 - 2t(1-t)(1-cosθ)) < 1 for diverse B.
        # The candy-wrapper: composed direction has SMALLER norm than any individual.
        composed = sum(w * Bh for w, Bh in zip(weights, B_hats))
        return scale * composed  # norm = scale × ||Σ w_i B̂_i||_F ≤ scale

    # SLERP: iterative pairwise interpolation with cumulative weight
    acc    = B_hats[0].flatten()   # unit vector
    cum_w  = weights[0]
    for i in range(1, N):
        v2    = B_hats[i].flatten()
        t     = weights[i] / (cum_w + weights[i])
        acc   = slerp_vectors(acc, v2, t)
        # Re-normalize numerically (stays on sphere by construction, but float drift)
        acc_n = np.linalg.norm(acc)
        if acc_n > 1e-10:
            acc = acc / acc_n
        cum_w += weights[i]

    return scale * acc.reshape(shape)


def measure_b_cosines(B_list: list) -> dict:
    flat = [B.flatten() / (np.linalg.norm(B.flatten()) + 1e-10) for B in B_list]
    cosines = [float(np.dot(flat[i], flat[j]))
               for i in range(len(flat)) for j in range(i+1, len(flat))]
    return {"values": cosines,
            "mean": float(np.mean(cosines)),
            "min":  float(np.min(cosines)),
            "max":  float(np.max(cosines))}


# ── Training ─────────────────────────────────────────────────────────────────
def init_A_matrices(rng: np.random.RandomState) -> dict:
    """Create shared frozen A matrices. {(l, name): np.array (rank, in)}"""
    A = {}
    for l in range(N_LAYERS):
        for name, in_f, out_f in LORA_MODULES:
            std = math.sqrt(2.0 / in_f)
            A[(l, name)] = rng.normal(0, std, (LORA_RANK, in_f)).astype(np.float32)
    return A


def init_B_matrices(rng: np.random.RandomState) -> dict:
    """Zero-initialized B matrices per domain. {dom: {(l,name): np.array (out, rank)}}"""
    B = {}
    for dom in range(N_DOMAINS):
        B[dom] = {}
        for l in range(N_LAYERS):
            for name, in_f, out_f in LORA_MODULES:
                B[dom][(l, name)] = rng.normal(0, 0.01, (out_f, LORA_RANK)).astype(np.float32)
    return B


def train_base(model: MicroTransformer, all_data: list,
               rng: np.random.RandomState) -> float:
    log("=== Training base model ===")
    optimizer = opt.AdamW(learning_rate=LR)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, model_loss)(model, x, y)
        optimizer.update(model, grads)
        return loss

    for i in range(BASE_STEPS):
        dom = i % N_DOMAINS
        idxs = rng.randint(0, len(all_data[dom]), BATCH_SIZE)
        samples = [all_data[dom][j] for j in idxs]
        x, y = make_batch(samples, rng)
        loss = step(x, y)
        mx.eval(state)
        if i % 150 == 0:
            log(f"  base step {i:4d}: loss={loss.item():.3f}")
    return loss.item()


def train_sft_adapter(model: MicroTransformer, A_np: dict, B_init: dict,
                      data: list, rng: np.random.RandomState) -> tuple:
    """Train one domain's B matrix. A is frozen numpy, B is trainable."""
    b_mod = BModule(B_init)
    optimizer = opt.AdamW(learning_rate=SFT_LR)

    def loss_fn(bm, x, y):
        return lora_forward_loss(model, A_np, bm, x, y)

    for i in range(SFT_STEPS):
        idxs = rng.randint(0, len(data), BATCH_SIZE)
        samples = [data[j] for j in idxs]
        x, y = make_batch(samples, rng)
        loss, grads = nn.value_and_grad(b_mod, loss_fn)(b_mod, x, y)
        optimizer.update(b_mod, grads)
        mx.eval(b_mod.parameters(), optimizer.state, loss)
        if i % 100 == 0:
            log(f"    step {i:3d}: loss={loss.item():.3f}")

    return loss.item(), b_mod.to_numpy()


def eval_with_B(model: MicroTransformer, A_np: dict, B_dict_np: dict,
                data: list, rng: np.random.RandomState) -> float:
    """Evaluate cross-entropy under a composed B (numpy dict {(l,name): array})."""
    b_mod = BModule(B_dict_np)
    total = 0.0
    for _ in range(N_EVAL):
        idxs = rng.randint(0, len(data), BATCH_SIZE)
        samples = [data[j] for j in idxs]
        x, y = make_batch(samples, rng)
        loss = lora_forward_loss(model, A_np, b_mod, x, y)
        mx.eval(loss)
        total += loss.item()
    cleanup(b_mod)
    return total / N_EVAL


def eval_base(model: MicroTransformer, data: list,
              rng: np.random.RandomState) -> float:
    total = 0.0
    for _ in range(N_EVAL):
        idxs = rng.randint(0, len(data), BATCH_SIZE)
        samples = [data[j] for j in idxs]
        x, y = make_batch(samples, rng)
        loss = model_loss(model, x, y)
        mx.eval(loss)
        total += loss.item()
    return total / N_EVAL


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    log("=== SLERP B-matrix Composition Experiment ===")
    log(f"SMOKE={IS_SMOKE}, D={D_MODEL}, L={N_LAYERS}, r={LORA_RANK}, N={N_DOMAINS}")

    # Generate data
    n_train = 300 if not IS_SMOKE else 50
    all_data = [gen_domain_data(d, n_train, rng) for d in range(N_DOMAINS)]
    mixed    = [s for dom in all_data for s in dom]
    log(f"Data: {N_DOMAINS} domains × {n_train} = {len(mixed)} samples")

    # Train base
    model = MicroTransformer()
    mx.eval(model.parameters())
    train_base(model, all_data, rng)
    base_weights_np = {k: np.array(v) for k, v in tree_flatten(model.parameters())}

    # Base perplexity per domain
    base_losses = {DOMAIN_NAMES[d]: eval_base(model, all_data[d], rng)
                   for d in range(N_DOMAINS)}
    log(f"Base losses: { {k: f'{v:.3f}' for k,v in base_losses.items()} }")

    # Shared frozen A, per-domain B
    A_np    = init_A_matrices(rng)
    B_all_np = init_B_matrices(rng)

    # Train SFT adapters
    log("\n=== Training SFT adapters ===")
    sft_losses = {}
    for dom in range(N_DOMAINS):
        log(f"Domain: {DOMAIN_NAMES[dom]}")
        # Reset model to base (SFT must start from same base)
        for k, v in tree_flatten(model.parameters()):
            ...  # we don't reset — model is shared base; only B changes
        final_loss, B_trained = train_sft_adapter(
            model, A_np, B_all_np[dom], all_data[dom], rng
        )
        B_all_np[dom] = B_trained
        sft_losses[DOMAIN_NAMES[dom]] = final_loss
    log(f"SFT final losses: { {k: f'{v:.3f}' for k,v in sft_losses.items()} }")

    # Eval single-adapter quality (before composition)
    log("\n=== Single-adapter eval ===")
    sft_eval = {}
    for dom in range(N_DOMAINS):
        sft_eval[DOMAIN_NAMES[dom]] = eval_with_B(model, A_np, B_all_np[dom], all_data[dom], rng)
    log(f"SFT eval losses: { {k: f'{v:.3f}' for k,v in sft_eval.items()} }")

    # ── K931 & K932: LERP vs SLERP composition ───────────────────────────────
    log("\n=== K931 / K932: LERP vs SLERP composition ===")

    # Collect B matrices per (layer, module) across all domains
    # B_by_module[(l, name)] = list of np.array (out, rank)
    B_by_module = {(l, name): [B_all_np[d][(l, name)] for d in range(N_DOMAINS)]
                   for l in range(N_LAYERS) for name, _, _ in LORA_MODULES}

    # Pairwise B-matrix cosines per module
    cos_stats = {}
    for (l, name), B_list in B_by_module.items():
        stats = measure_b_cosines(B_list)
        cos_stats[f"L{l}_{name}"] = stats
        log(f"  cos L{l}_{name}: mean={stats['mean']:.3f}  "
            f"min={stats['min']:.3f}  max={stats['max']:.3f}")

    # Build LERP and SLERP composed B dicts
    lerp_B = {(l, name): compose_B_matrices(B_by_module[(l, name)], method="lerp")
              for l in range(N_LAYERS) for name, _, _ in LORA_MODULES}
    slerp_B = {(l, name): compose_B_matrices(B_by_module[(l, name)], method="slerp")
               for l in range(N_LAYERS) for name, _, _ in LORA_MODULES}

    # K931: norm ratios
    norm_results = {}
    for (l, name), B_list in B_by_module.items():
        mean_n  = np.mean([np.linalg.norm(B, "fro") for B in B_list])
        lerp_n  = np.linalg.norm(lerp_B[(l, name)],  "fro")
        slerp_n = np.linalg.norm(slerp_B[(l, name)], "fro")
        ratio   = slerp_n / (lerp_n + 1e-10)
        norm_results[f"L{l}_{name}"] = {
            "mean_individual": float(mean_n),
            "lerp_norm":       float(lerp_n),
            "slerp_norm":      float(slerp_n),
            "lerp_ratio":      float(lerp_n / (mean_n + 1e-10)),
            "slerp_ratio":     float(slerp_n / (mean_n + 1e-10)),
            "slerp_over_lerp": float(ratio),
        }
        log(f"  norms L{l}_{name}: LERP={lerp_n:.3f}  SLERP={slerp_n:.3f}  ratio={ratio:.3f}")

    slerp_over_lerp_values = [v["slerp_over_lerp"] for v in norm_results.values()]
    mean_ratio = float(np.mean(slerp_over_lerp_values))
    k931_pass  = mean_ratio > 1.30
    log(f"\nK931 mean SLERP/LERP norm ratio: {mean_ratio:.3f}  "
        f"threshold=1.30  → {'PASS' if k931_pass else 'FAIL'}")

    # K932: quality on mixed eval
    lerp_loss  = eval_with_B(model, A_np, lerp_B,  mixed, rng)
    slerp_loss = eval_with_B(model, A_np, slerp_B, mixed, rng)
    k932_pass  = slerp_loss <= lerp_loss
    log(f"K932 mixed loss: LERP={lerp_loss:.4f}  SLERP={slerp_loss:.4f}  "
        f"→ {'PASS' if k932_pass else 'FAIL'} (SLERP≤LERP)")

    # Per-domain breakdown
    lerp_domain_losses  = {DOMAIN_NAMES[d]: eval_with_B(model, A_np, lerp_B,  all_data[d], rng)
                           for d in range(N_DOMAINS)}
    slerp_domain_losses = {DOMAIN_NAMES[d]: eval_with_B(model, A_np, slerp_B, all_data[d], rng)
                           for d in range(N_DOMAINS)}
    log(f"LERP domain:  { {k: f'{v:.3f}' for k,v in lerp_domain_losses.items()} }")
    log(f"SLERP domain: { {k: f'{v:.3f}' for k,v in slerp_domain_losses.items()} }")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment": "slerp_b_composition",
        "smoke_test": IS_SMOKE,
        "config": {"d_model": D_MODEL, "n_layers": N_LAYERS,
                   "lora_rank": LORA_RANK, "n_domains": N_DOMAINS},
        "base_losses":       base_losses,
        "sft_eval_losses":   sft_eval,
        "b_cosine_stats":    cos_stats,
        "norm_results":      norm_results,
        "k931": {
            "mean_slerp_over_lerp": mean_ratio,
            "per_module":           slerp_over_lerp_values,
            "threshold":            1.30,
            "pass":                 bool(k931_pass),
        },
        "k932": {
            "lerp_mixed_loss":  lerp_loss,
            "slerp_mixed_loss": slerp_loss,
            "pass":             bool(k932_pass),
        },
        "lerp_domain_losses":  lerp_domain_losses,
        "slerp_domain_losses": slerp_domain_losses,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved → {RESULTS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"K931 SLERP/LERP norm ratio: {mean_ratio:.3f} > 1.30 → {'PASS' if k931_pass else 'FAIL'}")
    log(f"K932 SLERP≤LERP quality: {slerp_loss:.4f} ≤ {lerp_loss:.4f} → {'PASS' if k932_pass else 'FAIL'}")

    if k931_pass and k932_pass:
        log("OUTCOME: SUPPORTED — SLERP prevents candy-wrapper AND preserves quality")
    elif k931_pass and not k932_pass:
        log("OUTCOME: K931 PASS / K932 FAIL — norm preserved but direction matters more than magnitude")
    else:
        log("OUTCOME: KILLED — B matrices not diverse enough; see cos_stats")


if __name__ == "__main__":
    main()
