#!/usr/bin/env python3
"""M2P Teacher Distillation: Cross-model knowledge transfer via KL distillation.

TYPE: Guided Exploration (Type 2)
MATH: experiments/models/m2p_teacher_distillation/MATH.md

APPROACH:
  1. Pre-train BOTH teacher (d=512, L=4) and student (d=256, L=2) on 3 domains
  2. SFT teacher on each domain separately (reference quality ceiling)
  3. SFT student directly on each domain (SFT upper bound for comparison)
  4. For each domain: train M2P with KL(teacher || student+M2P_adapter) loss
     M2P reads teacher last-layer hidden states via a learned projection layer
     B-matrices generated for student's Grassmannian LoRA slots
  5. Eval: compare student+M2P vs direct SFT student on all domains

KEY THEOREM (MATH.md Theorem 1, Corollary 1):
  Gibbs' inequality (KL ≥ 0) guarantees student+M2P CANNOT regress below base.
  Quality gap closure ≥ 0.50 is the empirical hypothesis (K853).

Kill criteria:
  K853: quality_gap_closure ≥ 0.50 for ≥2/3 domains (M2P useful vs direct SFT)
  K854: no architecture crash from teacher/student dimension mismatch
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
# Teacher: 2× student in all model dimensions for a natural scaling relationship
# Student matches m2p_composition_n5: d=256, L=2, 4 heads
D_STUDENT = 256
N_LAYERS_S = 2
N_HEADS_S = 4

D_TEACHER = 512          # 2× student
N_LAYERS_T = 4           # 2× student layers
N_HEADS_T = 8            # 2× student heads

VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0

# 3 domains (keeps experiment tractable, matches delegation spec)
N_DOMAINS = 3
DOMAIN_NAMES = ["arithmetic", "sort", "reverse"]

# M2P architecture
D_M2P = 64              # M2P internal hidden dim
N_MEMORY = 32           # M2P memory tokens
M2P_LAYERS = 2
# Projection: teacher hidden states → M2P input space
# Linear(D_TEACHER → D_M2P) = 512×64 = 32K params

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
# Student dimensions (M2P generates adapters for student)
MODULE_OUT_DIMS_S = [D_STUDENT, D_STUDENT, D_STUDENT, D_STUDENT, 4 * D_STUDENT]
N_MODULES = len(MODULE_NAMES)

# Distillation hyperparameters
# T_temp=2.0: Hinton 2015 §3 recommends T=2-5 for most tasks
# ALPHA=0.7: 70% KL distillation, 30% NTP auxiliary (Hinton 2015 §3.4)
KL_TEMP = 2.0
KL_ALPHA = 0.7

# Training config
BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 600  if not SMOKE_TEST else 30
M2P_STEPS  = 600  if not SMOKE_TEST else 30
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3


# ── Utilities ─────────────────────────────────────────────────────────────

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
    """Generate n text samples for a given domain."""
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


# ── Toy GPT (parametric: works for both teacher and student) ───────────────

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
    """Parametric toy GPT: works for both teacher (d=512,L=4) and student (d=256,L=2)."""

    def __init__(self, d: int, n_layers: int, n_heads: int):
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.wte = nn.Embedding(VOCAB_SIZE, d)
        self.wpe = nn.Embedding(BLOCK_SIZE + 1, d)
        self.blocks = [Block(d, n_heads) for _ in range(n_layers)]
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, VOCAB_SIZE, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def get_last_hidden(self, tokens):
        """Return last-layer hidden states (for teacher → M2P input)."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)  # (B, T, d)

    def get_all_hidden(self, tokens):
        """Return all-layer hidden states list."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states


# ── Grassmannian A-matrices (for student only) ────────────────────────────

def generate_grassmannian_A(n_domains: int, n_layers: int, n_modules: int,
                             d: int, rank: int, seed: int = 42) -> dict:
    """Generate frozen orthogonal A-matrices for student LoRA slots.

    Theorem 1 (MATH.md m2p_composition_n5): A_i^T A_j = 0 by QR construction.
    Capacity: n_domains * rank ≤ d  →  3 * 4 = 12 ≤ 256. Margin: 21×.
    """
    total_rank = n_domains * rank
    assert total_rank <= d, \
        f"Capacity violated: need {total_rank} orthogonal vectors but d={d}"

    rng = np.random.RandomState(seed)
    A_matrices = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)   # Q: (d, total_rank), Q^T Q = I
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start + rank])
    return A_matrices


def verify_grassmannian_orthogonality(A_matrices: dict, n_domains: int,
                                       n_layers: int, n_modules: int) -> dict:
    """Verify A_i^T A_j = 0 numerically (should be float32 machine zero)."""
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


# ── LoRA forward pass for student ─────────────────────────────────────────

def lora_forward_student(base: ToyGPT, tokens: mx.array,
                          A_matrices: dict, domain_id: int,
                          B_matrices: dict) -> mx.array:
    """Student forward pass with Grassmannian LoRA adapters.

    output = base_output + scale * (x @ A) @ B
    A is frozen (Grassmannian), B is provided (from SFT or M2P).
    """
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
            A = A_matrices[(domain_id, li, mi)]  # (d, rank)
            B = B_matrices[(li, mi)]             # (rank, d_out)
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

    logits = base.lm_head(base.norm_f(x))
    return logits


# ── SFT B-matrix container ────────────────────────────────────────────────

class BMatrices(nn.Module):
    """Trainable B-matrices for student LoRA adapter."""

    def __init__(self, n_layers: int = N_LAYERS_S):
        super().__init__()
        self.n_layers = n_layers
        for li in range(n_layers):
            for mi, d_out in enumerate(MODULE_OUT_DIMS_S):
                setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

    def as_dict(self) -> dict:
        return {
            (li, mi): getattr(self, f"B_{li}_{mi}")
            for li in range(self.n_layers) for mi in range(N_MODULES)
        }


def sft_loss_fn(b_container: BMatrices, base: ToyGPT, tokens: mx.array,
                A_matrices: dict, domain_id: int) -> mx.array:
    B_matrices = b_container.as_dict()
    logits = lora_forward_student(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(
        logits[:, :-1], tokens[:, 1:], reduction="mean"
    )


# ── M2P Transformer ────────────────────────────────────────────────────────

class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = _M2PAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = _M2PMLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _M2PAttention(nn.Module):
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
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class _M2PMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class M2PTransformerDistill(nn.Module):
    """M2P Transformer for teacher distillation.

    Extends SHINE M2P (Finding #339) with:
    1. Projection layer: Linear(D_TEACHER → D_M2P) — handles dimension mismatch (K854)
    2. Reads teacher last-layer hidden states (richer signal than student hidden states)
    3. Generates B-matrices for STUDENT's Grassmannian LoRA slots

    Architecture reference: SHINE arXiv:2602.06358
    """

    def __init__(self, d_teacher: int = D_TEACHER, d_m2p: int = D_M2P,
                 n_layers_student: int = N_LAYERS_S):
        super().__init__()
        self.d_teacher = d_teacher
        self.d_m2p = d_m2p
        self.n_layers_student = n_layers_student

        # K854: learned projection handles teacher/student dimension mismatch
        self.teacher_proj = nn.Linear(d_teacher, d_m2p, bias=False)

        # M2P memory tokens (learnable)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02

        # Positional embedding for memory tokens
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)

        # M2P transformer blocks
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(M2P_LAYERS)]
        self.norm_f = RMSNorm(d_m2p)

        # Output projections: one per module type, for ALL student layers
        # d_m2p → n_layers_student * LORA_RANK * d_out
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_S)):
            total_out = n_layers_student * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, teacher_hidden: mx.array) -> dict:
        """Generate student B-matrices from teacher's last-layer hidden states.

        Args:
          teacher_hidden: (1, T, D_TEACHER) — teacher last-layer hidden states

        Returns:
          dict[(layer_idx, module_idx)] → mx.array(LORA_RANK, d_out_student)
        """
        # Step 1: Project teacher hidden states to M2P space
        # Mean-pool over tokens: (1, T, D_TEACHER) → (D_TEACHER,) → (D_M2P,)
        pooled_teacher = mx.mean(teacher_hidden[0], axis=0)    # (D_TEACHER,)
        context_enc = self.teacher_proj(pooled_teacher)        # (D_M2P,)

        # Step 2: Initialize memory tokens and inject teacher context
        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)  # (N_MEMORY, D_M2P)
        memory = memory + context_enc[None, :]                 # broadcast context

        # Step 3: Run M2P transformer blocks
        x = memory[None, :, :]          # (1, N_MEMORY, D_M2P)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)              # (1, N_MEMORY, D_M2P)

        # Step 4: Pool memory tokens → generate B-matrices
        pooled_memory = mx.mean(x[0], axis=0)  # (D_M2P,)

        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS_S)):
            out = self.out_heads[mname](pooled_memory)  # (n_layers * rank * d_out,)
            out = out.reshape(self.n_layers_student, LORA_RANK, d_out)
            for li in range(self.n_layers_student):
                B_matrices[(li, mi)] = out[li]  # (rank, d_out)

        return B_matrices


# ── KL Distillation Loss ───────────────────────────────────────────────────

def kl_distill_loss(m2p: M2PTransformerDistill,
                    teacher: ToyGPT, student: ToyGPT,
                    A_matrices: dict, domain_id: int,
                    tokens: mx.array,
                    temp: float = KL_TEMP, alpha: float = KL_ALPHA) -> mx.array:
    """M2P KL distillation loss.

    L = alpha * T^2 * KL(teacher_soft || student_soft) + (1-alpha) * NTP(student+M2P)

    where teacher_soft = softmax(logits_T / T), student_soft = softmax(logits_S / T).

    Proof (MATH.md Theorem 1): KL ≥ 0 (Gibbs), minimizing KL moves student → teacher.
    T^2 scaling: Hinton 2015 §3, preserves gradient magnitude when temperature scales logits.

    Args:
      teacher: frozen teacher GPT
      student: frozen student GPT base
      A_matrices: Grassmannian A for student
      domain_id: which domain adapter to generate
      tokens: (1, T) input sequence
      temp: distillation temperature
      alpha: KL weight (1-alpha = NTP weight)
    """
    # Teacher forward (frozen — no gradients through teacher)
    teacher_hidden = teacher.get_last_hidden(tokens)  # (1, T, D_TEACHER)
    teacher_logits = teacher.lm_head(teacher_hidden)  # (1, T, V)

    # Generate student B-matrices from teacher hidden states
    B_matrices = m2p(teacher_hidden)

    # Student forward with M2P adapter
    student_logits = lora_forward_student(student, tokens, A_matrices, domain_id, B_matrices)
    # student_logits: (1, T, V)

    # Align sequences: predict tokens[1:] from positions [:-1]
    T_seq = tokens.shape[1]
    t_logits = teacher_logits[:, :-1, :]   # (1, T-1, V)
    s_logits = student_logits[:, :-1, :]   # (1, T-1, V)
    target_tokens = tokens[:, 1:]           # (1, T-1)

    # Soft targets at temperature T
    t_soft = mx.softmax(t_logits / temp, axis=-1)  # (1, T-1, V)

    # KL(teacher_soft || student_soft):
    # = Σ p_T log(p_T / p_S) = Σ p_T (log p_T - log p_S)
    # Using log-softmax for numerical stability
    s_log_soft = nn.log_softmax(s_logits / temp, axis=-1)  # (1, T-1, V)
    kl_loss = -mx.mean(mx.sum(t_soft * s_log_soft, axis=-1))  # scalar
    # Entropy of teacher (constant w.r.t. student, but include for correct gradient scaling)
    # We use: KL(p_T || p_S) ≈ -Σ p_T log p_S + const, so using -Σ p_T log p_S
    # is sufficient (the const doesn't affect student gradients).

    # NTP loss (hard targets) for stability
    ntp_loss = nn.losses.cross_entropy(s_logits, target_tokens, reduction="mean")

    # Combined loss: Hinton 2015 §3.4 recommends T^2 to preserve gradient scale
    total_loss = alpha * (temp ** 2) * kl_loss + (1 - alpha) * ntp_loss

    return total_loss, kl_loss, ntp_loss


# ── Evaluation helpers ─────────────────────────────────────────────────────

def eval_ntp_loss_base(model: ToyGPT, batches: list) -> float:
    """Evaluate base model NTP loss (no adapters)."""
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        logits = model(tokens_2d)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def eval_ntp_loss_lora(base: ToyGPT, batches: list,
                        A_matrices: dict, domain_id: int,
                        B_matrices: dict) -> float:
    """Evaluate NTP loss with LoRA adapter."""
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        logits = lora_forward_student(base, tokens_2d, A_matrices, domain_id, B_matrices)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def eval_m2p_loss(m2p: M2PTransformerDistill, teacher: ToyGPT,
                   student: ToyGPT, batches: list,
                   A_matrices: dict, domain_id: int) -> float:
    """Evaluate student+M2P NTP loss using teacher hidden states as input to M2P."""
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        teacher_hidden = teacher.get_last_hidden(tokens_2d)
        B_matrices = m2p(teacher_hidden)
        logits = lora_forward_student(student, tokens_2d, A_matrices, domain_id, B_matrices)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss, B_matrices, teacher_hidden
    return total / max(n, 1)


def measure_projection_stats(m2p: M2PTransformerDistill, teacher: ToyGPT,
                              domain_name: str, val_batches: list) -> dict:
    """Measure projection layer statistics for a single domain.

    P1 (MATH.md): Projection should produce non-trivial (non-random) representations.
    We measure: mean norm of projected vectors, and intra-domain cosine consistency
    (how similar are projections of different samples from the same domain?).
    """
    projections = []
    for tokens in val_batches[:8]:
        tokens_2d = tokens[None, :]
        teacher_hidden = teacher.get_last_hidden(tokens_2d)
        pooled = mx.mean(teacher_hidden[0], axis=0)    # (D_TEACHER,)
        proj = m2p.teacher_proj(pooled)                # (D_M2P,)
        mx.eval(proj)
        projections.append(np.array(proj.tolist()))
        del teacher_hidden, pooled, proj

    if len(projections) < 2:
        return {"mean_norm": 0.0, "intra_domain_cos": 0.0, "n_samples": len(projections)}

    projections = np.array(projections)   # (n_samples, D_M2P)
    norms = np.linalg.norm(projections, axis=1)
    mean_norm = float(np.mean(norms))

    # Intra-domain cosine: all pairs from same domain
    cos_vals = []
    for i in range(len(projections)):
        for j in range(i+1, len(projections)):
            v1, v2 = projections[i], projections[j]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            cos_vals.append(cos)

    return {
        "mean_norm": mean_norm,
        "intra_domain_cos": float(np.mean(cos_vals)) if cos_vals else 0.0,
        "n_samples": len(projections),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS (CODING_GUIDELINES §1: each phase in own scope)
# ═══════════════════════════════════════════════════════════════════════════

def phase_generate_data(rng: np.random.RandomState) -> dict:
    """Generate train/val data for all 3 domains."""
    domain_data = {}
    n_per_domain = 400 if not SMOKE_TEST else 40
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


def phase_pretrain_base(model: ToyGPT, domain_data: dict, model_name: str,
                         n_steps: int = BASE_STEPS, lr: float = LR) -> dict:
    """Pre-train base model on mixed-domain data.

    Returns: {"final_loss": float}
    """
    log(f"\n=== Pre-training {model_name} ({n_steps} steps) ===")
    optimizer = opt.AdamW(learning_rate=lr)

    # Mixed-domain batches: interleave all domains
    all_train = []
    for di, name in enumerate(DOMAIN_NAMES):
        all_train.extend(domain_data[name]["train"])

    rng_local = np.random.RandomState(SEED + hash(model_name) % 1000)
    rng_local.shuffle(all_train)

    def loss_fn(model, tokens):
        logits = model(tokens[None, :])
        return nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    gc.disable()
    for step in range(n_steps):
        tokens = all_train[step % len(all_train)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        del loss, grads

        if (step + 1) % 200 == 0:
            avg = np.mean(losses[-100:])
            log(f"  step {step+1}/{n_steps}: loss={avg:.4f}")
    gc.enable()
    gc.collect()

    result = {"final_loss": float(np.mean(losses[-50:]))}
    cleanup(optimizer, losses)
    return result


def phase_save_base(model: ToyGPT, model_name: str) -> Path:
    """Save base model weights to disk."""
    params = dict(tree_flatten(model.parameters()))
    save_path = EXPERIMENT_DIR / f"{model_name}_base_weights.npz"
    np.savez(save_path, **{k: np.array(v.tolist()) for k, v in params.items()})
    log(f"  Saved {model_name} base weights to {save_path}")
    return save_path


def phase_load_base(model: ToyGPT, save_path: Path) -> ToyGPT:
    """Load base model weights from disk."""
    data = np.load(save_path)
    params = {k: mx.array(data[k]) for k in data.files}
    model.load_weights(list(params.items()))
    mx.eval(model.parameters())
    return model


def phase_sft_domain(domain_name: str, domain_id: int,
                      student_path: Path,
                      A_matrices: dict,
                      domain_batches_train: list,
                      n_steps: int = SFT_STEPS) -> dict:
    """SFT student on a single domain. Returns B-matrices saved to disk.

    Scope: student model loaded and freed within this function.
    """
    log(f"\n=== SFT student on domain: {domain_name} ===")
    student = ToyGPT(D_STUDENT, N_LAYERS_S, N_HEADS_S)
    student = phase_load_base(student, student_path)

    b_container = BMatrices(N_LAYERS_S)
    optimizer = opt.AdamW(learning_rate=SFT_LR)
    loss_and_grad = nn.value_and_grad(b_container, sft_loss_fn)

    rng_local = np.random.RandomState(SEED + domain_id)
    losses = []
    gc.disable()
    for step in range(n_steps):
        tokens = domain_batches_train[step % len(domain_batches_train)]
        loss, grads = loss_and_grad(b_container, student, tokens[None, :],
                                     A_matrices, domain_id)
        optimizer.update(b_container, grads)
        mx.eval(b_container.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        del loss, grads

        if (step + 1) % 100 == 0:
            log(f"  step {step+1}/{n_steps}: loss={np.mean(losses[-50:]):.4f}")
    gc.enable()
    gc.collect()

    # Save B-matrices
    B_dict = b_container.as_dict()
    save_path = ADAPTER_DIR / f"sft_{domain_name}.npz"
    np.savez(save_path, **{
        f"B_{li}_{mi}": np.array(B_dict[(li, mi)].tolist())
        for li in range(N_LAYERS_S) for mi in range(N_MODULES)
    })
    log(f"  Saved SFT adapter to {save_path}")

    result = {"final_loss": float(np.mean(losses[-50:]))}
    cleanup(student, b_container, optimizer)
    return result


def phase_sft_teacher_domain(domain_name: str, domain_id: int,
                               teacher_path: Path,
                               domain_batches_train: list,
                               n_steps: int = SFT_STEPS) -> dict:
    """SFT teacher on a single domain (no LoRA, full fine-tune).

    We save only the teacher SFT loss (quality reference).
    Teacher SFT is done to measure its domain quality ceiling.
    """
    log(f"\n=== SFT teacher on domain: {domain_name} (quality reference) ===")
    teacher = ToyGPT(D_TEACHER, N_LAYERS_T, N_HEADS_T)
    teacher = phase_load_base(teacher, teacher_path)

    optimizer = opt.AdamW(learning_rate=SFT_LR * 0.5)  # slower for larger model

    def loss_fn(model, tokens):
        logits = model(tokens[None, :])
        return nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(teacher, loss_fn)

    losses = []
    gc.disable()
    for step in range(n_steps):
        tokens = domain_batches_train[step % len(domain_batches_train)]
        loss, grads = loss_and_grad(teacher, tokens)
        optimizer.update(teacher, grads)
        mx.eval(teacher.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        del loss, grads

        if (step + 1) % 100 == 0:
            log(f"  step {step+1}/{n_steps}: loss={np.mean(losses[-50:]):.4f}")
    gc.enable()
    gc.collect()

    # Save fine-tuned teacher weights (needed for teacher hidden states during M2P training)
    params = dict(tree_flatten(teacher.parameters()))
    save_path = EXPERIMENT_DIR / f"teacher_sft_{domain_name}.npz"
    np.savez(save_path, **{k: np.array(v.tolist()) for k, v in params.items()})
    log(f"  Saved teacher SFT weights to {save_path}")

    result = {"final_loss": float(np.mean(losses[-50:])), "save_path": str(save_path)}
    cleanup(teacher, optimizer)
    return result


def phase_train_m2p_domain(domain_name: str, domain_id: int,
                             teacher_sft_path: Path,
                             student_path: Path,
                             A_matrices: dict,
                             domain_batches_train: list,
                             n_steps: int = M2P_STEPS) -> dict:
    """Train M2P via KL distillation for one domain.

    M2P reads domain-fine-tuned teacher hidden states.
    Generates B-matrices for student's Grassmannian LoRA slots.
    Loss: alpha * T^2 * KL(teacher || student+M2P) + (1-alpha) * NTP(student+M2P)
    """
    log(f"\n=== Training M2P via KL distillation: {domain_name} ===")

    # Load frozen teacher (domain fine-tuned)
    teacher = ToyGPT(D_TEACHER, N_LAYERS_T, N_HEADS_T)
    data = np.load(teacher_sft_path)
    teacher_params = {k: mx.array(data[k]) for k in data.files}
    teacher.load_weights(list(teacher_params.items()))
    mx.eval(teacher.parameters())

    # Load frozen student (pre-trained base only)
    student = ToyGPT(D_STUDENT, N_LAYERS_S, N_HEADS_S)
    student = phase_load_base(student, student_path)

    # M2P transformer (trainable)
    m2p = M2PTransformerDistill(D_TEACHER, D_M2P, N_LAYERS_S)
    optimizer = opt.AdamW(learning_rate=M2P_LR)

    def m2p_loss_wrapper(m2p, teacher, student, tokens):
        total, kl, ntp = kl_distill_loss(m2p, teacher, student, A_matrices,
                                           domain_id, tokens, KL_TEMP, KL_ALPHA)
        return total

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_wrapper)

    losses = []
    kl_losses = []
    gc.disable()
    for step in range(n_steps):
        tokens = domain_batches_train[step % len(domain_batches_train)]
        loss, grads = loss_and_grad(m2p, teacher, student, tokens[None, :])
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        del loss, grads

        if (step + 1) % 100 == 0:
            avg = np.mean(losses[-50:])
            log(f"  step {step+1}/{n_steps}: loss={avg:.4f}")
    gc.enable()
    gc.collect()

    # Save M2P weights
    m2p_params = dict(tree_flatten(m2p.parameters()))
    save_path = ADAPTER_DIR / f"m2p_{domain_name}.npz"
    np.savez(save_path, **{k: np.array(v.tolist()) for k, v in m2p_params.items()})
    log(f"  Saved M2P weights to {save_path}")

    result = {
        "final_loss": float(np.mean(losses[-50:])),
        "initial_loss": float(np.mean(losses[:10])),
        "loss_reduction_pct": float(
            (np.mean(losses[:10]) - np.mean(losses[-50:])) / (np.mean(losses[:10]) + 1e-8) * 100
        ),
    }
    cleanup(teacher, student, m2p, optimizer)
    return result


def phase_eval_all(domain_data: dict, A_matrices: dict,
                    teacher_base_path: Path, student_base_path: Path,
                    teacher_sft_paths: dict, sft_adapter_paths: dict,
                    m2p_weight_paths: dict) -> dict:
    """Evaluate all models on all domains.

    Computes:
    - teacher_domain_loss: per-domain loss of SFT teacher (quality ceiling)
    - student_base_loss: per-domain loss of base student (lower bound)
    - student_sft_loss: per-domain loss of SFT student (upper bound)
    - student_m2p_loss: per-domain loss of student + M2P adapter
    - quality_gap_closure: (base - m2p) / (base - sft) per domain
    """
    log("\n=== Evaluating all models ===")
    results = {
        "teacher_domain_loss": {},
        "student_base_loss": {},
        "student_sft_loss": {},
        "student_m2p_loss": {},
        "quality_gap_closure": {},
        "projection_analysis": {},
    }

    # Load student base (needed for all evals)
    student_base = ToyGPT(D_STUDENT, N_LAYERS_S, N_HEADS_S)
    student_base = phase_load_base(student_base, student_base_path)

    for di, name in enumerate(DOMAIN_NAMES):
        val_batches = domain_data[name]["val"]
        log(f"\n  Domain: {name}")

        # 1. Teacher domain loss (SFT teacher)
        teacher_sft = ToyGPT(D_TEACHER, N_LAYERS_T, N_HEADS_T)
        data = np.load(teacher_sft_paths[name])
        teacher_params = {k: mx.array(data[k]) for k in data.files}
        teacher_sft.load_weights(list(teacher_params.items()))
        mx.eval(teacher_sft.parameters())
        teacher_loss = eval_ntp_loss_base(teacher_sft, val_batches)
        log(f"    Teacher SFT loss: {teacher_loss:.4f}")

        # 2. Student base loss (no adapters)
        student_base_loss = eval_ntp_loss_base(student_base, val_batches)
        log(f"    Student base loss: {student_base_loss:.4f}")

        # 3. Student SFT loss (direct SFT on student)
        sft_data = np.load(sft_adapter_paths[name])
        B_sft = {
            (li, mi): mx.array(sft_data[f"B_{li}_{mi}"])
            for li in range(N_LAYERS_S) for mi in range(N_MODULES)
        }
        mx.eval(list(B_sft.values()))
        student_sft_loss = eval_ntp_loss_lora(student_base, val_batches,
                                               A_matrices, di, B_sft)
        log(f"    Student SFT loss: {student_sft_loss:.4f}")
        del sft_data, B_sft

        # 4. Student M2P loss (M2P generates adapter from teacher hidden states)
        m2p = M2PTransformerDistill(D_TEACHER, D_M2P, N_LAYERS_S)
        m2p_data = np.load(m2p_weight_paths[name])
        m2p_params_list = [(k, mx.array(m2p_data[k])) for k in m2p_data.files]
        m2p.load_weights(m2p_params_list)
        mx.eval(m2p.parameters())

        student_m2p_loss = eval_m2p_loss(m2p, teacher_sft, student_base,
                                          val_batches, A_matrices, di)
        log(f"    Student M2P loss: {student_m2p_loss:.4f}")

        # 5. Quality gap closure: (base - m2p) / (base - sft)
        gap_base = student_base_loss - student_sft_loss
        gap_m2p = student_base_loss - student_m2p_loss
        if abs(gap_base) < 1e-4:
            closure = 0.0  # degenerate: teacher not better than student
            log(f"    WARNING: small gap ({gap_base:.4f}), closure undefined")
        else:
            closure = float(gap_m2p / gap_base)
        log(f"    Quality gap closure: {closure:.3f} (K853 threshold: 0.50)")

        # 6. Projection statistics (P1): are projections non-trivial?
        proj_analysis = measure_projection_stats(m2p, teacher_sft, name, val_batches)
        log(f"    Projection: norm={proj_analysis['mean_norm']:.4f}, "
            f"intra_cos={proj_analysis['intra_domain_cos']:.4f}")

        results["teacher_domain_loss"][name] = round(float(teacher_loss), 4)
        results["student_base_loss"][name] = round(float(student_base_loss), 4)
        results["student_sft_loss"][name] = round(float(student_sft_loss), 4)
        results["student_m2p_loss"][name] = round(float(student_m2p_loss), 4)
        results["quality_gap_closure"][name] = round(closure, 4)
        results["projection_analysis"][name] = proj_analysis

        cleanup(teacher_sft, m2p, m2p_data)
        gc.collect()
        mx.clear_cache()

    cleanup(student_base)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("M2P Teacher Distillation Experiment")
    log(f"SMOKE_TEST: {SMOKE_TEST}")
    log(f"Domains: {DOMAIN_NAMES}")
    log(f"Teacher: d={D_TEACHER}, L={N_LAYERS_T}, H={N_HEADS_T}")
    log(f"Student: d={D_STUDENT}, L={N_LAYERS_S}, H={N_HEADS_S}")
    log(f"M2P:     d_m2p={D_M2P}, memory={N_MEMORY}, layers={M2P_LAYERS}")
    log(f"Distill: temp={KL_TEMP}, alpha={KL_ALPHA}")
    log("=" * 60)
    log_memory("start")

    rng = np.random.RandomState(SEED)

    # Phase 0: Generate data
    log("\n--- Phase 0: Generate Data ---")
    domain_data = phase_generate_data(rng)

    # Phase 1: Pre-train teacher
    log("\n--- Phase 1: Pre-train Teacher ---")
    teacher_model = ToyGPT(D_TEACHER, N_LAYERS_T, N_HEADS_T)
    phase_pretrain_base(teacher_model, domain_data, "teacher",
                         n_steps=BASE_STEPS, lr=LR * 0.5)
    teacher_path = phase_save_base(teacher_model, "teacher")
    cleanup(teacher_model)
    log_memory("after teacher pretrain")

    # Phase 2: Pre-train student
    log("\n--- Phase 2: Pre-train Student ---")
    student_model = ToyGPT(D_STUDENT, N_LAYERS_S, N_HEADS_S)
    phase_pretrain_base(student_model, domain_data, "student",
                         n_steps=BASE_STEPS, lr=LR)
    student_path = phase_save_base(student_model, "student")
    cleanup(student_model)
    log_memory("after student pretrain")

    # Phase 3: Generate Grassmannian A-matrices (for student)
    log("\n--- Phase 3: Generate Grassmannian A-matrices ---")
    total_rank = N_DOMAINS * LORA_RANK
    log(f"  Capacity: {N_DOMAINS} domains × rank {LORA_RANK} = {total_rank} ≤ d={D_STUDENT}")
    A_matrices = generate_grassmannian_A(N_DOMAINS, N_LAYERS_S, N_MODULES,
                                          D_STUDENT, LORA_RANK, SEED)
    orth_check = verify_grassmannian_orthogonality(A_matrices, N_DOMAINS,
                                                    N_LAYERS_S, N_MODULES)
    log(f"  Orthogonality check: max|cos|={orth_check['max_cos']:.2e}, "
        f"mean|cos|={orth_check['mean_cos']:.2e} (should be ~0)")
    log_memory("after A-matrix generation")

    # Phase 4: SFT teacher on each domain (quality reference)
    log("\n--- Phase 4: SFT Teacher (Domain Reference) ---")
    teacher_sft_results = {}
    teacher_sft_paths = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_teacher_domain(
            name, di, teacher_path,
            domain_data[name]["train"],
            n_steps=SFT_STEPS
        )
        teacher_sft_results[name] = result
        teacher_sft_paths[name] = Path(result["save_path"])
    log_memory("after teacher SFT")

    # Phase 5: SFT student on each domain (upper bound for comparison)
    log("\n--- Phase 5: SFT Student (Upper Bound) ---")
    student_sft_results = {}
    sft_adapter_paths = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            name, di, student_path, A_matrices,
            domain_data[name]["train"],
            n_steps=SFT_STEPS
        )
        student_sft_results[name] = result
        sft_adapter_paths[name] = ADAPTER_DIR / f"sft_{name}.npz"
    log_memory("after student SFT")

    # Phase 6: Train M2P via KL distillation (main experiment)
    log("\n--- Phase 6: Train M2P via KL Distillation ---")
    m2p_train_results = {}
    m2p_weight_paths = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_train_m2p_domain(
            name, di,
            teacher_sft_paths[name],
            student_path,
            A_matrices,
            domain_data[name]["train"],
            n_steps=M2P_STEPS
        )
        m2p_train_results[name] = result
        m2p_weight_paths[name] = ADAPTER_DIR / f"m2p_{name}.npz"
    log_memory("after M2P training")

    # Phase 7: Evaluate all models
    log("\n--- Phase 7: Evaluate All Models ---")
    eval_results = phase_eval_all(
        domain_data, A_matrices,
        teacher_path, student_path,
        teacher_sft_paths, sft_adapter_paths, m2p_weight_paths
    )
    log_memory("after evaluation")

    # Compute kill criteria
    closure_values = list(eval_results["quality_gap_closure"].values())
    n_domains = len(closure_values)
    n_pass_k853 = sum(1 for c in closure_values if c >= 0.50)
    majority = n_pass_k853 >= (2 * n_domains) // 3 + (1 if n_domains % 3 != 0 else 0)
    # Conservative majority: >= 2/3 domains must pass
    k853_pass = majority
    k854_pass = True  # If we got here, no architecture crash

    log("\n=== KILL CRITERIA ===")
    log(f"K853 (gap closure ≥0.50 for ≥2/3 domains): "
        f"{n_pass_k853}/{n_domains} domains pass → {'PASS' if k853_pass else 'FAIL'}")
    for name, closure in eval_results["quality_gap_closure"].items():
        log(f"  {name}: closure={closure:.3f} ({'PASS' if closure >= 0.50 else 'FAIL'})")
    log(f"K854 (no architecture crash): {'PASS' if k854_pass else 'FAIL'}")

    # Full results
    final_results = {
        "experiment": "exp_m2p_teacher_distillation",
        "smoke_test": SMOKE_TEST,
        "config": {
            "d_teacher": D_TEACHER, "n_layers_t": N_LAYERS_T,
            "d_student": D_STUDENT, "n_layers_s": N_LAYERS_S,
            "d_m2p": D_M2P, "n_memory": N_MEMORY,
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "kl_temp": KL_TEMP, "kl_alpha": KL_ALPHA,
            "base_steps": BASE_STEPS, "sft_steps": SFT_STEPS, "m2p_steps": M2P_STEPS,
        },
        "grassmannian_check": orth_check,
        "teacher_sft_train": teacher_sft_results,
        "student_sft_train": student_sft_results,
        "m2p_train": m2p_train_results,
        "teacher_domain_loss": eval_results["teacher_domain_loss"],
        "student_base_loss": eval_results["student_base_loss"],
        "student_sft_loss": eval_results["student_sft_loss"],
        "student_m2p_loss": eval_results["student_m2p_loss"],
        "quality_gap_closure": eval_results["quality_gap_closure"],
        "projection_analysis": eval_results["projection_analysis"],
        "k853_pass": k853_pass,
        "k854_pass": k854_pass,
        "k853_detail": {
            "n_domains_pass": n_pass_k853,
            "n_domains_total": n_domains,
            "per_domain": {
                name: float(c) >= 0.50
                for name, c in eval_results["quality_gap_closure"].items()
            },
        },
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(final_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {final_results['total_time_s']:.1f}s")

    return final_results


if __name__ == "__main__":
    main()
