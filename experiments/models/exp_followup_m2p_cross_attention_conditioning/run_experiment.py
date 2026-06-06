#!/usr/bin/env python3
"""M2P cross-attention context conditioning (followup to exp_m2p_scale_calibrated).

Isolates ONE architectural change: the context-to-memory injection path.

  killed sibling :  mem = mem_init + broadcast(mean(task_embed(c)))
  this experiment:  mem = mem_init + CrossAttn(Q=mem_init, K=task_embed(c), V=task_embed(c))

Everything else (base GPT pretrain, SFT reference, Grassmannian A-matrices,
L_task + lambda*L_preserve loss, evaluation protocol, 20-context CV measurement,
seed, hyperparameters) is carried over unchanged so the comparison is apples-to-apples.

KC #1556 pre-registered in MATH.md:
  K1556a: CV_cross_attn > 0.05 (headline)
  K1556b: CV_mean_pool_baseline <= 0.02 (control, reproduces Finding #343)
  K1556c: CV_cross_attn >= 3 * CV_mean_pool_baseline (relative, guards against noise drift)
All three must pass for supported.

Reference implementation of the mean-pool baseline path is retained from the
sibling's run_experiment.py so we have a true within-run control.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Memory safety: leave 8 GB for system, cap cache at 2 GB
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── Architecture constants (INHERITED VERBATIM from killed sibling) ──────
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
D_M2P = 64
N_MEMORY = 8
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]

BASE_STEPS = 800 if not SMOKE_TEST else 50
SFT_STEPS  = 400 if not SMOKE_TEST else 30
M2P_STEPS  = 600 if not SMOKE_TEST else 40
LAMBDA_PRESERVE = 0.1
N_CONTEXT_VARIANTS = 20 if not SMOKE_TEST else 5

LR = 3e-4
BATCH_SIZE = 8


# ── Utilities ────────────────────────────────────────────────────────────

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


# ── Data generation (inherited from sibling) ─────────────────────────────

def generate_arithmetic_tokens(n_samples: int, rng: np.random.RandomState,
                                difficulty: str = "mixed") -> np.ndarray:
    samples = []
    for _ in range(n_samples):
        if difficulty == "easy" or (difficulty == "mixed" and rng.rand() < 0.5):
            n_ops = rng.randint(2, 4)
            nums = rng.randint(1, 10, size=n_ops).tolist()
        else:
            n_ops = rng.randint(4, 7)
            nums = rng.randint(10, 200, size=n_ops).tolist()
        expr = "+".join(str(n) for n in nums) + "=" + str(sum(nums))
        text = (expr + "\n") * (BLOCK_SIZE // (len(expr) + 1) + 2)
        tokens = np.array([ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]],
                          dtype=np.int32)
        samples.append(tokens)
    return np.stack(samples, axis=0)


def generate_general_tokens(n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    samples = []
    for _ in range(n_samples):
        tokens = rng.randint(32, 127, size=BLOCK_SIZE + 1).astype(np.int32)
        samples.append(tokens)
    return np.stack(samples, axis=0)


# ── Base GPT (inherited) ─────────────────────────────────────────────────

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
        self.wpe = nn.Embedding(BLOCK_SIZE, D_MODEL)
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


# ── M2P internal blocks ──────────────────────────────────────────────────

class M2PSelfBlock(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)  # self-attn on memory tokens
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """mem_k = Σ_t softmax_t(Q_k · K_t / √d_h) · V_t

    Query from memory tokens, keys/values from context token embeddings.
    Lemma 2 of MATH.md: rank(∂output/∂context) >= min(N_MEMORY, T, d_k).
    """
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, mem, ctx):
        # mem: (B, N, d);  ctx: (B, T, d)
        B, N, C = mem.shape
        _, T, _ = ctx.shape
        q = self.wq(mem).reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(ctx).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(ctx).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        # no causal mask: memory attends to all context positions
        attn = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale, axis=-1)  # (B, H, N, T)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.wo(out)


# ── M2P transformer variants ─────────────────────────────────────────────

class M2PMeanPool(nn.Module):
    """Killed-sibling control path.

    mem = mem_init + broadcast(mean(task_embed(c), axis=1))
    """
    def __init__(self):
        super().__init__()
        self.task_embed = nn.Embedding(VOCAB_SIZE, D_M2P)
        scale_init = math.sqrt(2.0 / (1 + D_M2P))
        self.memory = mx.random.normal((1, N_MEMORY, D_M2P)) * scale_init
        self.blocks = [M2PSelfBlock(D_M2P, n_heads=4) for _ in range(M2P_LAYERS)]
        self.final_norm = RMSNorm(D_M2P)
        total_B_params = sum(LORA_RANK * d_out for d_out in MODULE_OUT_DIMS) * N_LAYERS
        self.B_proj = nn.Linear(D_M2P * N_MEMORY, total_B_params, bias=False)

    def _init_B_proj_small(self):
        self.B_proj.weight = self.B_proj.weight * 0.01

    def __call__(self, task_tokens):
        task_emb = self.task_embed(task_tokens)                    # (1, T, d)
        task_ctx = mx.mean(task_emb, axis=1, keepdims=True)        # (1, 1, d) ← BOTTLENECK
        mem = self.memory + mx.broadcast_to(task_ctx, (1, N_MEMORY, D_M2P))
        for block in self.blocks:
            mem = block(mem)
        mem = self.final_norm(mem)
        flat = mem.reshape(1, -1)
        B_flat = self.B_proj(flat)[0]
        B_matrices = {}
        offset = 0
        for li in range(N_LAYERS):
            for mi, d_out in enumerate(MODULE_OUT_DIMS):
                n = LORA_RANK * d_out
                B_ij = B_flat[offset:offset + n].reshape(LORA_RANK, d_out)
                B_matrices[(li, mi)] = B_ij
                offset += n
        return B_matrices


class M2PCrossAttn(nn.Module):
    """Cross-attention replacement of the mean-pool injection.

    mem = mem_init + CrossAttn(Q=mem_init, K=task_embed(c), V=task_embed(c))

    Lemma 2 rank-8 Jacobian → K1556a should pass; everything downstream is
    identical to the sibling.
    """
    def __init__(self):
        super().__init__()
        self.task_embed = nn.Embedding(VOCAB_SIZE, D_M2P)
        scale_init = math.sqrt(2.0 / (1 + D_M2P))
        self.memory = mx.random.normal((1, N_MEMORY, D_M2P)) * scale_init
        self.cross = CrossAttention(D_M2P, n_heads=4)
        self.cross_norm_q = RMSNorm(D_M2P)
        self.cross_norm_kv = RMSNorm(D_M2P)
        self.blocks = [M2PSelfBlock(D_M2P, n_heads=4) for _ in range(M2P_LAYERS)]
        self.final_norm = RMSNorm(D_M2P)
        total_B_params = sum(LORA_RANK * d_out for d_out in MODULE_OUT_DIMS) * N_LAYERS
        self.B_proj = nn.Linear(D_M2P * N_MEMORY, total_B_params, bias=False)

    def _init_B_proj_small(self):
        self.B_proj.weight = self.B_proj.weight * 0.01

    def __call__(self, task_tokens):
        task_emb = self.task_embed(task_tokens)                    # (1, T, d)
        q_norm = self.cross_norm_q(self.memory)                    # (1, N, d)
        kv_norm = self.cross_norm_kv(task_emb)                     # (1, T, d)
        mem = self.memory + self.cross(q_norm, kv_norm)            # (1, N, d) ← per-slot context mix
        for block in self.blocks:
            mem = block(mem)
        mem = self.final_norm(mem)
        flat = mem.reshape(1, -1)
        B_flat = self.B_proj(flat)[0]
        B_matrices = {}
        offset = 0
        for li in range(N_LAYERS):
            for mi, d_out in enumerate(MODULE_OUT_DIMS):
                n = LORA_RANK * d_out
                B_ij = B_flat[offset:offset + n].reshape(LORA_RANK, d_out)
                B_matrices[(li, mi)] = B_ij
                offset += n
        return B_matrices


# ── Grassmannian A-matrices ──────────────────────────────────────────────

def generate_grassmannian_A() -> dict:
    rng = np.random.RandomState(SEED + 100)
    A_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(len(MODULE_NAMES)):
            Q, _ = np.linalg.qr(rng.randn(D_MODEL, LORA_RANK).astype(np.float32))
            A_matrices[(li, mi)] = mx.array(Q[:, :LORA_RANK])
    return A_matrices


# ── Loss ─────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits, tokens):
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


# ── Base GPT pretrain ────────────────────────────────────────────────────

def phase_pretrain_base(rng: np.random.RandomState):
    log("\n=== Phase 1: Pre-train Base GPT ===")
    mx.random.seed(SEED)

    model = ToyGPT()
    mx.eval(model.parameters())

    optimizer = opt.AdamW(learning_rate=LR, weight_decay=0.01)

    data_np = generate_arithmetic_tokens(BATCH_SIZE * (BASE_STEPS + 50), rng, difficulty="mixed")

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        return cross_entropy_loss(logits, tokens)

    losses = []
    gc.disable()
    for step in range(BASE_STEPS):
        idx = (step * BATCH_SIZE) % (len(data_np) - BATCH_SIZE)
        batch = mx.array(data_np[idx:idx + BATCH_SIZE])
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={losses[-1]:.4f}")
    gc.enable()

    final_loss = float(np.mean(losses[-50:]))
    log(f"  Final base loss (mean last 50): {final_loss:.4f}")

    def get_block_weights(block):
        return {
            "wq": np.array(block.attn.wq.weight),
            "wk": np.array(block.attn.wk.weight),
            "wv": np.array(block.attn.wv.weight),
            "wo": np.array(block.attn.wo.weight),
            "fc1": np.array(block.mlp.fc1.weight),
            "fc2": np.array(block.mlp.fc2.weight),
            "norm1_weight": np.array(block.norm1.weight),
            "norm2_weight": np.array(block.norm2.weight),
        }

    base_weights_np = {
        "wte": np.array(model.wte.weight),
        "wpe": np.array(model.wpe.weight),
        "norm_f_weight": np.array(model.norm_f.weight),
        "lm_head": np.array(model.lm_head.weight),
        "blocks": [get_block_weights(b) for b in model.blocks],
    }

    results = {"base_final_loss": final_loss}

    cleanup(model, optimizer)
    return base_weights_np, results


def _make_mlx_base_weights(base_weights_np: dict) -> dict:
    return {
        "wte": mx.array(base_weights_np["wte"]),
        "wpe": mx.array(base_weights_np["wpe"]),
        "norm_f_weight": mx.array(base_weights_np["norm_f_weight"]),
        "lm_head": mx.array(base_weights_np["lm_head"]),
        "blocks": [
            {k: mx.array(v) for k, v in bw.items()}
            for bw in base_weights_np["blocks"]
        ],
    }


def _forward_with_b(tokens, base_mlx: dict, A_matrices: dict,
                    B_matrices: dict, scale: float):
    inp = tokens[:, :-1]
    B_batch, T = inp.shape
    pos = mx.arange(T)
    x = base_mlx["wte"][inp] + base_mlx["wpe"][pos]

    for li in range(N_LAYERS):
        bw = base_mlx["blocks"][li]
        rms1 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        n1 = x * rms1 * bw["norm1_weight"]

        def lora_proj(h, W, li_, mi_):
            out = h @ W.T
            A = A_matrices[(li_, mi_)]
            B_mat = B_matrices.get((li_, mi_))
            if B_mat is not None:
                out = out + scale * ((h @ A) @ B_mat)
            return out

        hd = D_MODEL // N_HEADS
        q = lora_proj(n1, bw["wq"], li, 0).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
        k = lora_proj(n1, bw["wk"], li, 1).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
        v = lora_proj(n1, bw["wv"], li, 2).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn_w = mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd ** -0.5) + mask, axis=-1)
        attn_out = (attn_w @ v).transpose(0, 2, 1, 3).reshape(B_batch, T, D_MODEL)
        attn_out = lora_proj(attn_out, bw["wo"], li, 3)
        x = x + attn_out

        rms2 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        n2 = x * rms2 * bw["norm2_weight"]
        fc1_out = lora_proj(n2, bw["fc1"], li, 4)
        fc1_act = nn.gelu(fc1_out)
        fc2_out = fc1_act @ bw["fc2"].T
        x = x + fc2_out

    rms_f = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    x = x * rms_f * base_mlx["norm_f_weight"]
    return x @ base_mlx["lm_head"].T


# ── M2P training (parametrized by architecture) ──────────────────────────

def phase_train_m2p(base_weights_np: dict, A_matrices: dict,
                    rng: np.random.RandomState,
                    arch: str,
                    label: str):
    """arch: "mean_pool" or "cross_attn".

    Matches killed-sibling training protocol exactly, modulo the M2P class.
    """
    assert arch in {"mean_pool", "cross_attn"}
    log(f"\n=== Phase: Train M2P [{arch}] — {label} ===")
    # Identical seed policy as sibling: +2 when use_preserve=True.
    # For cross-attn we use a fresh but deterministic offset so it doesn't collide.
    seed_offset = 2 if arch == "mean_pool" else 12
    mx.random.seed(SEED + seed_offset)

    base_mlx = _make_mlx_base_weights(base_weights_np)
    ADAPTER_SCALE = 1.0

    m2p = M2PMeanPool() if arch == "mean_pool" else M2PCrossAttn()
    m2p._init_B_proj_small()
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")

    optimizer = opt.AdamW(learning_rate=LR * 0.5, weight_decay=0.01)

    n_train = BATCH_SIZE * (M2P_STEPS + 50)
    arith_easy_data = generate_arithmetic_tokens(n_train // 2, rng, difficulty="easy")
    arith_hard_data = generate_arithmetic_tokens(n_train - n_train // 2, rng, difficulty="hard")
    gen_data = generate_general_tokens(n_train, rng)

    def m2p_loss_fn(m2p, ctx_tokens, arith_tokens, gen_tokens):
        task_inp = ctx_tokens[:, :BLOCK_SIZE]
        B_matrices = m2p(task_inp)
        logits_task = _forward_with_b(arith_tokens, base_mlx, A_matrices, B_matrices, ADAPTER_SCALE)
        L_task = cross_entropy_loss(logits_task, arith_tokens)
        logits_preserve = _forward_with_b(gen_tokens, base_mlx, A_matrices, B_matrices, ADAPTER_SCALE)
        L_preserve = cross_entropy_loss(logits_preserve, gen_tokens)
        L_total = L_task + LAMBDA_PRESERVE * L_preserve
        return L_total, L_task, L_preserve

    losses_total, losses_task, losses_preserve = [], [], []
    gc.disable()
    for step in range(M2P_STEPS):
        if step % 2 == 0:
            ctx_np = arith_easy_data[(step // 2) % len(arith_easy_data)][None, :]
        else:
            ctx_np = arith_hard_data[(step // 2) % len(arith_hard_data)][None, :]

        idx_a = step % (max(len(arith_easy_data), len(arith_hard_data)) - BATCH_SIZE)
        arith_batch_np = (arith_easy_data if step % 2 == 0 else arith_hard_data)
        arith_batch_np = arith_batch_np[idx_a:idx_a + BATCH_SIZE]
        if len(arith_batch_np) < BATCH_SIZE:
            arith_batch_np = arith_batch_np[:1].repeat(BATCH_SIZE, axis=0)

        idx_g = (step * BATCH_SIZE) % (len(gen_data) - BATCH_SIZE)
        ctx_batch = mx.array(ctx_np)
        arith_batch = mx.array(arith_batch_np)
        gen_batch = mx.array(gen_data[idx_g:idx_g + BATCH_SIZE])

        def loss_for_grad(m2p):
            return m2p_loss_fn(m2p, ctx_batch, arith_batch, gen_batch)

        (L_total, L_task, L_preserve), grads = nn.value_and_grad(m2p, loss_for_grad)(m2p)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, L_total, L_task, L_preserve)

        losses_total.append(L_total.item())
        losses_task.append(L_task.item())
        losses_preserve.append(L_preserve.item())

        if (step + 1) % 100 == 0:
            log(f"  Step {step+1}/{M2P_STEPS}: "
                f"L_total={losses_total[-1]:.4f} "
                f"L_task={losses_task[-1]:.4f} "
                f"L_pres={losses_preserve[-1]:.4f}")
    gc.enable()

    # ── K1556 evaluation: magnitude CV across 20 contexts ────────────────
    log(f"\n  === Evaluation: adapter magnitude CV across {N_CONTEXT_VARIANTS} easy + {N_CONTEXT_VARIANTS} hard contexts ===")
    # Use a FIXED eval rng seed so the mean-pool and cross-attn arches see
    # identical eval contexts — this is critical for an apples-to-apples CV comparison.
    eval_rng = np.random.RandomState(SEED + 999)
    easy_norms, hard_norms = [], []
    for i in range(N_CONTEXT_VARIANTS):
        ctx_easy_np = generate_arithmetic_tokens(1, eval_rng, difficulty="easy")
        ctx_easy = mx.array(ctx_easy_np[:, :BLOCK_SIZE])
        B_easy = m2p(ctx_easy)
        mx.eval(*list(B_easy.values()))
        n_easy = float(np.mean([mx.linalg.norm(B_easy[(li, mi)].reshape(-1)).item()
                                for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))]))
        easy_norms.append(n_easy)

        ctx_hard_np = generate_arithmetic_tokens(1, eval_rng, difficulty="hard")
        ctx_hard = mx.array(ctx_hard_np[:, :BLOCK_SIZE])
        B_hard = m2p(ctx_hard)
        mx.eval(*list(B_hard.values()))
        n_hard = float(np.mean([mx.linalg.norm(B_hard[(li, mi)].reshape(-1)).item()
                                for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))]))
        hard_norms.append(n_hard)

    mean_easy = float(np.mean(easy_norms))
    mean_hard = float(np.mean(hard_norms))
    all_norms = easy_norms + hard_norms
    magnitude_mean = float(np.mean(all_norms))
    magnitude_std = float(np.std(all_norms))
    magnitude_cv = magnitude_std / (magnitude_mean + 1e-8)
    hard_easy_ratio = mean_hard / (mean_easy + 1e-8)

    log(f"  Easy ||B||_F mean: {mean_easy:.4f}")
    log(f"  Hard ||B||_F mean: {mean_hard:.4f}  (ratio {hard_easy_ratio:.3f})")
    log(f"  Overall CV = {magnitude_cv:.4f}")

    # General quality sanity (K849 analog, informational)
    eval_gen = generate_general_tokens(BATCH_SIZE, eval_rng)
    gen_eval_batch = mx.array(eval_gen)
    ctx_eval_np = generate_arithmetic_tokens(1, eval_rng, difficulty="mixed")
    ctx_eval = mx.array(ctx_eval_np[:, :BLOCK_SIZE])
    B_final = m2p(ctx_eval)
    mx.eval(*list(B_final.values()))
    logits_g_adapted = _forward_with_b(gen_eval_batch, base_mlx, A_matrices, B_final, ADAPTER_SCALE)
    logits_g_base = _forward_with_b(gen_eval_batch, base_mlx, A_matrices, {}, ADAPTER_SCALE)
    mx.eval(logits_g_adapted, logits_g_base)
    gen_ce_adapted = cross_entropy_loss(logits_g_adapted, gen_eval_batch).item()
    gen_ce_base = cross_entropy_loss(logits_g_base, gen_eval_batch).item()
    gen_degradation_pct = (gen_ce_adapted - gen_ce_base) / gen_ce_base * 100.0

    results = {
        "arch": arch,
        "label": label,
        "final_loss_total": float(np.mean(losses_total[-30:])),
        "final_loss_task": float(np.mean(losses_task[-30:])),
        "final_loss_preserve": float(np.mean(losses_preserve[-30:])),
        "gen_degradation_pp": gen_degradation_pct,
        "adapter_magnitude_mean": magnitude_mean,
        "adapter_magnitude_std": magnitude_std,
        "adapter_magnitude_cv": magnitude_cv,
        "easy_norms": easy_norms,
        "hard_norms": hard_norms,
        "mean_easy_norm": mean_easy,
        "mean_hard_norm": mean_hard,
        "hard_easy_ratio": hard_easy_ratio,
        "n_m2p_params": n_params,
    }

    cleanup(m2p, optimizer, base_mlx, gen_eval_batch,
            logits_g_adapted, logits_g_base, ctx_eval, B_final)
    return results


# ── Main orchestration ───────────────────────────────────────────────────

def main():
    t0 = time.time()

    log("=" * 65)
    log("M2P cross-attention context conditioning (follow-up to exp_m2p_scale_calibrated)")
    log("=" * 65)
    log(f"SMOKE_TEST: {SMOKE_TEST}")
    log(f"D_MODEL={D_MODEL}, N_LAYERS={N_LAYERS}, LORA_RANK={LORA_RANK}, "
        f"N_MEMORY={N_MEMORY}, D_M2P={D_M2P}")
    log(f"LAMBDA_PRESERVE={LAMBDA_PRESERVE}, N_CONTEXT_VARIANTS={N_CONTEXT_VARIANTS}")
    log("KC #1556: K1556a CV_cross > 0.05, K1556b CV_mean_pool <= 0.02, K1556c ratio >= 3")
    log("")

    rng = np.random.RandomState(SEED)

    base_weights_np, base_results = phase_pretrain_base(rng)
    A_matrices = generate_grassmannian_A()

    # Control: mean-pool (reproduces Finding #343 CV~=0.0093)
    mean_pool_results = phase_train_m2p(
        base_weights_np, A_matrices, rng,
        arch="mean_pool",
        label="control (reproduce killed sibling)"
    )
    # Treatment: cross-attention (this experiment's architectural fix)
    cross_attn_results = phase_train_m2p(
        base_weights_np, A_matrices, rng,
        arch="cross_attn",
        label="treatment (Lemma 2 rank-increasing)"
    )

    # ── Kill criteria ────────────────────────────────────────────────────
    cv_mean = mean_pool_results["adapter_magnitude_cv"]
    cv_cross = cross_attn_results["adapter_magnitude_cv"]

    k1556a_pass = cv_cross > 0.05
    k1556b_pass = cv_mean <= 0.02
    # ratio-based KC: cross must be at least 3x baseline CV
    # guard against baseline collapse (cv_mean near zero) via absolute headline K1556a
    ratio = cv_cross / (cv_mean + 1e-8)
    k1556c_pass = ratio >= 3.0

    all_pass = k1556a_pass and k1556b_pass and k1556c_pass

    log("\n" + "=" * 65)
    log("KILL CRITERIA SUMMARY")
    log("=" * 65)
    log(f"K1556a (CV_cross > 0.05):           CV_cross={cv_cross:.4f}  "
        f"=> {'PASS' if k1556a_pass else 'FAIL'}")
    log(f"K1556b (CV_mean_pool <= 0.02):      CV_mean={cv_mean:.4f}  "
        f"=> {'PASS' if k1556b_pass else 'FAIL'}")
    log(f"K1556c (CV_cross >= 3 * CV_mean):   ratio={ratio:.2f}  "
        f"=> {'PASS' if k1556c_pass else 'FAIL'}")
    log(f"OVERALL: {'SUPPORTED' if all_pass else 'KILLED'}")

    # ── Prediction table ─────────────────────────────────────────────────
    predictions = {
        "P1_cv_cross_gt_005": {
            "predicted": "> 0.05", "measured": cv_cross, "pass": k1556a_pass,
        },
        "P2_cv_ratio_gt_3x": {
            "predicted": ">= 3x", "measured": ratio, "pass": k1556c_pass,
        },
        "P3_cv_mean_le_002": {
            "predicted": "<= 0.02", "measured": cv_mean, "pass": k1556b_pass,
        },
        "P4_hard_gt_easy_cross_attn": {
            "predicted": ">= 1.10",
            "measured": cross_attn_results["hard_easy_ratio"],
            "pass": cross_attn_results["hard_easy_ratio"] >= 1.10,
        },
        "P5_gen_deg_within_10pp_of_baseline": {
            "mean_pool_gen_deg_pp": mean_pool_results["gen_degradation_pp"],
            "cross_attn_gen_deg_pp": cross_attn_results["gen_degradation_pp"],
            "delta_abs": abs(cross_attn_results["gen_degradation_pp"]
                             - mean_pool_results["gen_degradation_pp"]),
            "pass": abs(cross_attn_results["gen_degradation_pp"]
                        - mean_pool_results["gen_degradation_pp"]) <= 10.0,
        },
    }

    total_time = time.time() - t0

    results = {
        "experiment": "exp_followup_m2p_cross_attention_conditioning",
        "verdict": "supported" if all_pass else "killed",
        "status": "supported" if all_pass else "killed",
        "is_smoke": SMOKE_TEST,
        "total_time_s": round(total_time, 1),
        "config": {
            "d_model": D_MODEL,
            "d_m2p": D_M2P,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "vocab_size": VOCAB_SIZE,
            "lora_rank": LORA_RANK,
            "n_memory": N_MEMORY,
            "m2p_layers": M2P_LAYERS,
            "lambda_preserve": LAMBDA_PRESERVE,
            "n_context_variants": N_CONTEXT_VARIANTS,
            "adapter_scale": 1.0,
            "seed": SEED,
            "smoke_test": SMOKE_TEST,
        },
        "base_model": base_results,
        "mean_pool_control": mean_pool_results,
        "cross_attn_treatment": cross_attn_results,
        # KC pass flags — primary outputs for `experiment complete`
        "k1556a_pass": k1556a_pass,
        "k1556b_pass": k1556b_pass,
        "k1556c_pass": k1556c_pass,
        "all_pass": all_pass,
        # Summary for easy parsing
        "cv_cross_attn": cv_cross,
        "cv_mean_pool": cv_mean,
        "cv_ratio_cross_over_mean": ratio,
        "predictions": predictions,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults written to: {RESULTS_FILE}")
    log(f"Total runtime: {total_time:.1f}s")

    return all_pass


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
