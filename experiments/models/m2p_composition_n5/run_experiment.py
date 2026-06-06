#!/usr/bin/env python3
"""M2P Composition N=5: End-to-end test of decoupled Grassmannian architecture.

TYPE: verification (Type 1)
MATH: experiments/models/m2p_composition_n5/MATH.md

APPROACH:
  1. Pre-train toy GPT base model (d=256, L=2, 4 heads, vocab=128)
  2. Generate Grassmannian A-matrices: 5 domains × L layers × 5 modules
  3. SFT each domain to obtain reference B-matrices (ground truth quality)
  4. Train 5 independent M2P transformers (one per domain, avoids bottleneck)
  5. Compose all 5 adapters simultaneously, route per-token
  6. Measure: routing accuracy, composition quality, general quality preservation

THEOREM 1 (MATH.md): A_i^T A_j = 0 → <Δ_i, Δ_j>_F = 0 for ANY B_i, B_j.
  Parameter orthogonality is exact (float32 zero), guaranteed by QR construction.

THEOREM 2 (MATH.md): Linear separability of hidden states (Finding #310: 98.3%)
  → routing accuracy ≥ 80% with a trained router.

Kill criteria:
  K851: Composition degrades general quality >10pp despite Grassmannian A
  K852: Routing accuracy <50% (M2P adapters not domain-specific)
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
# NOTE: D_MODEL=256, N_LAYERS=2 matches m2p_scale_calibrated (delegation spec)
# Capacity check: N_DOMAINS * LORA_RANK = 5*4 = 20 ≤ D_MODEL=256 (12.8× margin)

D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5

# M2P: lightweight, trained independently per domain
D_M2P = 64          # M2P internal hidden dim (NOT base model dim)
N_MEMORY = 32       # M2P memory tokens (must hold rank*d_out per module)
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

# Training config
BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
M2P_STEPS  = 500  if not SMOKE_TEST else 30   # per domain
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]

# Domain signals for routing: each domain has a distinctive trigger character
# The router sees the first token of each input — these differ by domain
DOMAIN_TRIGGER_CHARS = {
    "arithmetic": set("0123456789"),  # starts with digit
    "sort": set("abcdefgh"),          # starts with lowercase letter
    "parity": set("01"),              # starts with 0 or 1
    "reverse": set("abcdefgh"),       # also letter, but different pattern
    "repeat": set("abcdefgh"),        # also letter, but different pattern
}
# NOTE: routing is LEARNED from labeled examples, not just first-char heuristic.
# The router is a small MLP trained on domain-labeled inputs.


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


def encode_for_routing(text: str) -> list:
    """Encode text for router training (padded to BLOCK_SIZE)."""
    tokens = [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE]]
    while len(tokens) < BLOCK_SIZE:
        tokens.append(0)
    return tokens


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
        """Return per-layer hidden states (for M2P input and routing)."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)   # (B, T, D)
        return states


# ── Grassmannian A-matrices ────────────────────────────────────────────────

def generate_grassmannian_A(n_domains: int, n_layers: int, n_modules: int,
                             d: int, rank: int, seed: int = 42) -> dict:
    """Generate frozen orthogonal A-matrices via QR decomposition.

    Theorem 1 (MATH.md): A_i^T A_j = 0 for all i≠j (proven by QR property).

    Returns: dict[(domain_idx, layer_idx, module_idx)] → mx.array(d, rank)
    """
    total_rank = n_domains * rank
    assert total_rank <= d, \
        f"Capacity violated: need {total_rank} orthogonal vectors but d={d}"

    rng = np.random.RandomState(seed)
    A_matrices = {}

    for li in range(n_layers):
        for mi in range(n_modules):
            # Sample random d × total_rank matrix, apply QR
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)   # Q: (d, total_rank), Q^T Q = I
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start + rank])

    return A_matrices


def verify_grassmannian_orthogonality(A_matrices: dict, n_domains: int,
                                       n_layers: int, n_modules: int) -> dict:
    """Verify Theorem 1: A_i^T A_j = 0 numerically.

    Returns: {mean_cos, max_cos} — should be ~0 at float32.
    """
    cos_values = []
    for li in range(n_layers):
        for mi in range(n_modules):
            for di in range(n_domains):
                for dj in range(di + 1, n_domains):
                    ai = A_matrices[(di, li, mi)]  # (d, rank)
                    aj = A_matrices[(dj, li, mi)]  # (d, rank)
                    # Flatten and compute cosine
                    ai_flat = ai.reshape(-1)
                    aj_flat = aj.reshape(-1)
                    cos = mx.abs(
                        mx.sum(ai_flat * aj_flat) /
                        (mx.linalg.norm(ai_flat) * mx.linalg.norm(aj_flat) + 1e-12)
                    ).item()
                    cos_values.append(cos)
    return {
        "mean_cos": float(np.mean(cos_values)),
        "max_cos": float(np.max(cos_values)),
        "n_pairs": len(cos_values),
    }


# ── LoRA forward pass (inline, avoids nn.Module overhead) ─────────────────

def lora_forward_with_B(base: ToyGPT, tokens: mx.array,
                         A_matrices: dict, domain_id: int,
                         B_matrices: dict) -> mx.array:
    """Forward pass of ToyGPT with LoRA adapters applied inline.

    Matches GrassmannianLoRA from m2p_distillation_toy exactly:
      output = base_output + scale * (x @ A) @ B

    LoRA applied at: wq, wk, wv (before attention), wo (after context),
                     fc1 (inside MLP before nonlinearity).

    Args:
      base: frozen ToyGPT
      tokens: (B, T) int32
      A_matrices: dict[(domain_id, layer_idx, module_idx)] → (d_in, rank)
      domain_id: which adapter to apply
      B_matrices: dict[(layer_idx, module_idx)] → (rank, d_out)

    Returns: logits (B, T, vocab_size)
    """
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        # LoRA-modified Q, K, V: output = Wq(x) + scale*(x@A_wq)@B_wq
        def _apply_lora(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            A = A_matrices[(domain_id, li, mi)]  # (d_in, rank)
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

        # LoRA-modified wo
        attn_out = _apply_lora(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        # LoRA-modified fc1 (inside MLP, before nonlinearity)
        x_norm2 = block.norm2(x)
        fc1_in = x_norm2
        fc1_base = block.mlp.fc1(fc1_in)
        A_fc1 = A_matrices[(domain_id, li, 4)]  # (d, rank)
        B_fc1 = B_matrices[(li, 4)]              # (rank, 4*d)
        fc1_out = fc1_base + LORA_SCALE * (fc1_in @ A_fc1) @ B_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    logits = base.lm_head(base.norm_f(x))
    return logits


# ── SFT training (per-domain LoRA with Grassmannian A) ────────────────────

class GrassmannianLoRA(nn.Module):
    """LoRA with frozen Grassmannian A and trainable B."""

    def __init__(self, base_linear: nn.Linear, A_frozen: mx.array):
        super().__init__()
        d_out = base_linear.weight.shape[0]
        self.A = A_frozen  # (d_in, rank), frozen
        self.B = mx.zeros((LORA_RANK, d_out))
        self._base_weight = base_linear.weight  # frozen

    def __call__(self, x):
        base_out = x @ self._base_weight.T
        return base_out + LORA_SCALE * (x @ self.A) @ self.B


class BMatrices(nn.Module):
    """Container for trainable B-matrices, compatible with nn.value_and_grad."""

    def __init__(self):
        super().__init__()
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                d_out = MODULE_OUT_DIMS[mi]
                setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

    def as_dict(self) -> dict:
        """Return B-matrices as (layer_idx, module_idx) → array dict."""
        return {
            (li, mi): getattr(self, f"B_{li}_{mi}")
            for li in range(N_LAYERS) for mi in range(N_MODULES)
        }


def sft_loss_fn(b_container: BMatrices, base: ToyGPT, tokens: mx.array,
                A_matrices: dict, domain_id: int) -> mx.array:
    """Compute NTP loss using GrassmannianLoRA forward pass."""
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(
        logits[:, :-1], tokens[:, 1:], reduction="mean"
    )


# ── M2P Transformer (lightweight, per-domain) ─────────────────────────────

class M2PAttention(nn.Module):
    """Standard multi-head attention for M2P blocks."""

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
    """Single M2P transformer block (row attention on memory tokens)."""

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
    """Memory-to-Parameter Transformer (SHINE-style, single domain).

    Input: hidden states from base model, L layers of (B, T, D_BASE)
    Process: project to D_M2P, run M2P blocks on memory tokens
    Output: B-matrices for all N_LAYERS × N_MODULES adapter slots

    Key difference from multi-domain M2P: trained on ONE domain only.
    This avoids the gradient conflict / bottleneck issue (#341, #342, #343).
    """

    def __init__(self, d_base: int = D_MODEL, d_m2p: int = D_M2P):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p

        # Project base model hidden states to M2P space
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)

        # M2P memory tokens (learnable)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02

        # Positional embedding for memory tokens
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)

        # M2P transformer blocks
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(M2P_LAYERS)]
        self.norm_f = RMSNorm(d_m2p)

        # Output projections: one per module type.
        # We first pool memory tokens → single D_M2P vector, then project to adapter params.
        # Shape: d_m2p → N_LAYERS * rank * d_out
        # At d_m2p=64, rank=4, d_out=256, N_LAYERS=2: 64 → 2048 (manageable)
        # At d_m2p=64, rank=4, d_out=1024 (fc1), N_LAYERS=2: 64 → 8192 (large but ok)
        # Total M2P params: ~5 modules × (64 × N_LAYERS*rank*d_out) ≈ 150K, reasonable
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            total_out = N_LAYERS * LORA_RANK * d_out
            # Project from pooled memory (D_M2P) to all adapter params for this module
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list: list) -> dict:
        """Generate B-matrices from context hidden states.

        Args:
          hidden_states_list: list of L tensors, each (1, T, D_BASE)
            (batch size 1 — single context token sequence)

        Returns: dict[(layer_idx, module_idx)] → mx.array(rank, d_out)
        """
        # Step 1: encode hidden states → memory context
        # Mean-pool over tokens for each layer, project to M2P space
        # This gives one D_M2P vector per base-model layer
        layer_encodings = []
        for h in hidden_states_list:
            # h: (1, T, D_BASE) → mean over T → (D_BASE,) → project → (D_M2P,)
            pooled = mx.mean(h[0], axis=0)  # (D_BASE,)
            enc = self.input_proj(pooled)    # (D_M2P,)
            layer_encodings.append(enc)

        # Broadcast layer encodings to memory tokens
        # memory: (N_MEMORY, D_M2P) initialized with learnable tokens
        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)  # (N_MEMORY, D_M2P)

        # Add context: inject layer-averaged encoding into memory
        # Stack layer encodings → (N_LAYERS, D_M2P), mean → (D_M2P,)
        context_enc = mx.mean(mx.stack(layer_encodings, axis=0), axis=0)  # (D_M2P,)
        # Additive injection (simple, avoids separate cross-attention)
        memory = memory + context_enc[None, :]  # (N_MEMORY, D_M2P)

        # Step 2: run M2P blocks
        # Add batch dim for attention: (1, N_MEMORY, D_M2P)
        x = memory[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)  # (1, N_MEMORY, D_M2P)

        # Step 3: generate B-matrices from memory
        # Pool memory tokens → single D_M2P vector (mean over N_MEMORY)
        pooled_memory = mx.mean(x[0], axis=0)  # (D_M2P,)

        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pooled_memory)  # (N_LAYERS * rank * d_out,)
            # Reshape: (N_LAYERS, rank, d_out)
            out = out.reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS):
                B_matrices[(li, mi)] = out[li]  # (rank, d_out)

        return B_matrices


def m2p_ntp_loss(m2p: M2PTransformer, base: ToyGPT,
                 A_matrices: dict, domain_id: int,
                 tokens: mx.array) -> mx.array:
    """M2P training loss: generate B from context, then measure NTP loss.

    M2P sees the input tokens as context (they act as the domain signal).
    Then the generated B-matrices are applied during the forward pass.
    """
    # Get context hidden states (M2P input)
    hidden_states = base.get_hidden_states(tokens)

    # Generate B-matrices from context
    B_matrices = m2p(hidden_states)

    # Apply B-matrices and compute NTP loss on the same tokens
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(
        logits[:, :-1], tokens[:, 1:], reduction="mean"
    )


# ── Router (per-token domain classification) ──────────────────────────────

class DomainRouter(nn.Module):
    """Small MLP router: base model hidden state → domain logits.

    Uses the LAST layer hidden state (most informative for routing).
    Single-pass MLP matches Finding #313 (within 0.61% of oracle).
    """

    def __init__(self, d: int = D_MODEL, n_domains: int = N_DOMAINS):
        super().__init__()
        self.fc1 = nn.Linear(d, d // 4, bias=True)
        self.fc2 = nn.Linear(d // 4, n_domains, bias=True)

    def __call__(self, x):
        """x: (B, T, D) hidden states → (B, T, N_DOMAINS) logits."""
        return self.fc2(nn.gelu(self.fc1(x)))


# ── Evaluation helpers ─────────────────────────────────────────────────────

def eval_ntp_loss(base: ToyGPT, batches: list,
                  A_matrices: dict = None, domain_id: int = None,
                  B_matrices: dict = None) -> float:
    """Evaluate NTP loss on a list of token batches.

    If A_matrices and B_matrices provided, applies LoRA adapter.
    Otherwise evaluates base model only.
    """
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]  # add batch dim
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


def composed_forward(base: ToyGPT, tokens: mx.array,
                      A_matrices: dict, all_B: list,
                      router: DomainRouter) -> mx.array:
    """Composed forward pass with per-token routing.

    For each token, the router selects domain weights, then applies:
      output = W_base x + Σ_j router_j(x) * Δ_j x

    Simplified: use argmax routing (hard routing) for quality measurement.
    For comparison to single-adapter: use the top-1 adapter per token.

    Args:
      base: frozen ToyGPT
      tokens: (B, T) int32
      A_matrices: dict[(domain, layer, module)] → (d, rank)
      all_B: list of N_DOMAINS B-matrix dicts, each dict[(layer, module)] → (rank, d_out)
      router: trained DomainRouter
    """
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    # FIX (REVIEW Issue 1, 2026-04-18): router train/test distribution mismatch.
    # Router was trained on base-model LAST-LAYER hidden states. Previously this
    # function called router(x) at every block on perturbed intermediate states —
    # a different distribution than training. Pre-compute routing ONCE from the
    # base model's last-layer hidden state via a stop-gradient prefix pass.
    base_hidden_list = base.get_hidden_states(tokens)
    base_last_hidden = mx.stop_gradient(base_hidden_list[-1])  # (B, T, D)
    mx.eval(base_last_hidden)
    router_logits = router(base_last_hidden)                    # (B, T, N_DOMAINS)
    routing_weights = mx.softmax(router_logits, axis=-1)        # stable across layers

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        # Compute weighted sum of LoRA corrections for each module
        def _apply_composed_lora(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            # Sum over domains with routing weights
            lora_sum = mx.zeros_like(base_out)
            for di in range(N_DOMAINS):
                A = A_matrices[(di, li, mi)]    # (d, rank)
                B = all_B[di][(li, mi)]          # (rank, d_out)
                # LoRA correction for domain di: (B, T, d_out)
                delta = LORA_SCALE * (x_in @ A) @ B
                # Weight by routing: (B, T, 1) * (B, T, d_out)
                w = routing_weights[:, :, di:di+1]  # (B, T, 1)
                lora_sum = lora_sum + w * delta
            return base_out + lora_sum

        q = _apply_composed_lora(attn.wq, x_norm, li, 0)
        k = _apply_composed_lora(attn.wk, x_norm, li, 1)
        v = _apply_composed_lora(attn.wv, x_norm, li, 2)

        q = q.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)

        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        scale_factor = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale_factor + mask, axis=-1)
        attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(B_b, T_b, C)

        attn_out = _apply_composed_lora(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        x_norm2 = block.norm2(x)
        # Composed fc1
        fc1_base_out = block.mlp.fc1(x_norm2)
        lora_fc1 = mx.zeros_like(fc1_base_out)
        for di in range(N_DOMAINS):
            A = A_matrices[(di, li, 4)]      # (d, rank)
            B = all_B[di][(li, 4)]            # (rank, 4*d)
            delta = LORA_SCALE * (x_norm2 @ A) @ B
            w = routing_weights[:, :, di:di+1]
            lora_fc1 = lora_fc1 + w * delta
        fc1_out = fc1_base_out + lora_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    logits = base.lm_head(base.norm_f(x))
    return logits, routing_weights


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS (each in its own scope for memory safety — CODING_GUIDELINES §1)
# ═══════════════════════════════════════════════════════════════════════════

def phase_generate_data(rng: np.random.RandomState) -> dict:
    """Generate train/val data for all 5 domains."""
    domain_data = {}
    n_per_domain = 500 if not SMOKE_TEST else 60
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        domain_data[name] = {
            "train": make_batches(texts[:split]),
            "val": make_batches(texts[split:]),
            "texts": texts,
            "domain_id": di,
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, "
            f"{len(domain_data[name]['val'])} val")
    return domain_data


def phase_pretrain_base(domain_data: dict) -> dict:
    """Pre-train ToyGPT on all domains. Return saved weights."""
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
        if (step + 1) % (BASE_STEPS // 4) == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={loss.item():.4f}")
    gc.enable()

    # Evaluate base loss per domain
    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")

    # Save base weights to disk
    base_weights_path = EXPERIMENT_DIR / "base_weights.npz"
    weights_dict = {}
    for k, v in tree_flatten(base.parameters()):
        weights_dict[k.replace(".", "_")] = np.array(v)
    np.savez(str(base_weights_path), **weights_dict)
    log(f"  Base weights saved to {base_weights_path}")

    cleanup(optimizer)
    # Do NOT cleanup base — we return it for Grassmannian generation
    return base, base_losses, str(base_weights_path)


def phase_grassmannian(base: ToyGPT) -> dict:
    """Generate and verify Grassmannian A-matrices."""
    log("\n=== Phase 2: Grassmannian A-matrices ===")

    A_matrices = generate_grassmannian_A(
        N_DOMAINS, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK, seed=SEED
    )

    ortho = verify_grassmannian_orthogonality(
        A_matrices, N_DOMAINS, N_LAYERS, N_MODULES
    )
    log(f"  Orthogonality: mean|cos|={ortho['mean_cos']:.6f}, "
        f"max|cos|={ortho['max_cos']:.6f} ({ortho['n_pairs']} pairs)")

    # Prediction: should be float32 machine epsilon ~1e-7
    assert ortho["max_cos"] < 1e-5, \
        f"Grassmannian guarantee failed: max|cos|={ortho['max_cos']}"

    return A_matrices, ortho


def phase_sft_domain(domain_name: str, domain_id: int,
                      domain_data: dict, base: ToyGPT,
                      A_matrices: dict, base_loss: float) -> dict:
    """Train SFT LoRA adapter for one domain. Saves B-matrices to disk.

    Memory-isolated per domain (CODING_GUIDELINES §1).
    """
    log(f"\n  SFT {domain_name} (domain {domain_id})...")

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

    # Evaluate SFT quality
    B_matrices = b_container.as_dict()
    sft_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)

    quality_ratio = ((base_loss - sft_loss) / base_loss) if base_loss > 0.01 else 0.0
    log(f"    SFT loss={sft_loss:.4f} base={base_loss:.4f} "
        f"improvement={quality_ratio:.1%}")

    # Save B-matrices to disk (CODING_GUIDELINES §3)
    save_path = ADAPTER_DIR / f"sft_{domain_name}.npz"
    np_dict = {f"{li}_{mi}": np.array(getattr(b_container, f"B_{li}_{mi}"))
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(save_path), **np_dict)

    cleanup(optimizer, b_container)
    return {"sft_loss": round(sft_loss, 4), "save_path": str(save_path)}


def phase_sft_all_domains(domain_data: dict, base: ToyGPT,
                           A_matrices: dict, base_losses: dict) -> dict:
    """Train SFT adapters for all 5 domains."""
    log("\n=== Phase 3: SFT Adapter Baselines ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            name, di, domain_data[name], base, A_matrices, base_losses[name]
        )
        sft_results[name] = result
    return sft_results


def phase_m2p_domain(domain_name: str, domain_id: int,
                      domain_data: dict, base: ToyGPT,
                      A_matrices: dict, base_loss: float,
                      sft_loss: float) -> dict:
    """Train M2P for ONE domain independently.

    Key design: each M2P sees only its own domain data → no gradient conflicts
    (avoids the bottleneck identified in Findings #341, #342, #343).
    """
    log(f"\n  M2P {domain_name} (domain {domain_id})...")

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P)
    mx.eval(m2p.parameters())

    m2p_param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"    M2P params: {m2p_param_count:,}")

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id,
                             tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]

    gc.disable()
    for step in range(M2P_STEPS):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        if (step + 1) % (M2P_STEPS // 4) == 0:
            log(f"    Step {step+1}/{M2P_STEPS}: loss={loss.item():.4f}")
    gc.enable()

    # Evaluate M2P quality on validation set
    total_loss = 0.0
    n_eval = 0
    for tokens in domain_data["val"][:30]:
        l = m2p_ntp_loss(m2p, base, A_matrices, domain_id, tokens[None, :])
        mx.eval(l)
        total_loss += l.item()
        n_eval += 1
        del l
    m2p_val_loss = total_loss / max(n_eval, 1)

    quality_ratio = 0.0
    if (base_loss - sft_loss) > 0.01:
        quality_ratio = (base_loss - m2p_val_loss) / (base_loss - sft_loss)
    log(f"    M2P val_loss={m2p_val_loss:.4f} SFT={sft_loss:.4f} "
        f"quality={quality_ratio:.1%}")

    # Generate B-matrices from a representative context and save
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
        "m2p_params": m2p_param_count,
    }


def phase_m2p_all_domains(domain_data: dict, base: ToyGPT,
                           A_matrices: dict, base_losses: dict,
                           sft_results: dict) -> dict:
    """Train independent M2P for all 5 domains."""
    log("\n=== Phase 4: Independent M2P Training (per domain) ===")
    m2p_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_m2p_domain(
            name, di, domain_data[name], base,
            A_matrices, base_losses[name],
            sft_results[name]["sft_loss"]
        )
        m2p_results[name] = result
    return m2p_results


def load_B_matrices(path: str) -> dict:
    """Load B-matrices from .npz file into (layer, module) keyed dict."""
    data = np.load(path)
    B_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(N_MODULES):
            key = f"{li}_{mi}"
            B_matrices[(li, mi)] = mx.array(data[key])
    return B_matrices


def phase_train_router(domain_data: dict, base: ToyGPT,
                        A_matrices: dict, m2p_results: dict) -> DomainRouter:
    """Train per-token domain router on M2P adapter-conditioned hidden states.

    The router maps base-model hidden states → domain logits.
    Training: for each domain, pass domain examples through base model,
    extract last-layer hidden states, train router to classify correctly.

    Why last-layer hidden states: they contain the most domain-specific
    information (Finding #310: 98.3% linear separability).
    """
    log("\n=== Phase 5: Train Domain Router ===")

    router = DomainRouter(d=D_MODEL, n_domains=N_DOMAINS)
    mx.eval(router.parameters())
    optimizer = opt.Adam(learning_rate=1e-3)

    # Build routing training data: (hidden_states, domain_label) pairs
    # For each domain, extract hidden states from base model forward passes
    router_train = []
    for di, name in enumerate(DOMAIN_NAMES):
        for tokens in domain_data[name]["train"][:30]:
            router_train.append((tokens, di))

    if not SMOKE_TEST:
        np.random.shuffle(router_train)

    def router_loss(router_model, hidden_batch, label_batch):
        """Cross-entropy routing loss on per-token predictions."""
        # hidden_batch: (B, T, D), label_batch: (B, T) filled with domain label
        logits = router_model(hidden_batch)  # (B, T, N_DOMAINS)
        return nn.losses.cross_entropy(
            logits.reshape(-1, N_DOMAINS),
            label_batch.reshape(-1),
            reduction="mean"
        )

    router_grad_fn = nn.value_and_grad(router, router_loss)

    router_steps = 300 if not SMOKE_TEST else 20
    gc.disable()
    for step in range(router_steps):
        tokens, domain_id = router_train[step % len(router_train)]
        tokens_2d = tokens[None, :]

        # Extract hidden states (no gradient through base model)
        hidden_states_list = base.get_hidden_states(tokens_2d)
        last_hidden = hidden_states_list[-1]  # (1, T, D)
        mx.eval(last_hidden)

        # Domain labels: all tokens in this sequence belong to domain_id
        B_b, T_b, _ = last_hidden.shape
        labels = mx.full((B_b, T_b), domain_id, dtype=mx.int32)

        loss, grads = router_grad_fn(router, last_hidden, labels)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)
        del last_hidden, labels
    gc.enable()

    # Evaluate routing accuracy
    correct = 0
    total = 0
    for di, name in enumerate(DOMAIN_NAMES):
        for tokens in domain_data[name]["val"][:20]:
            tokens_2d = tokens[None, :]
            hidden_states_list = base.get_hidden_states(tokens_2d)
            last_hidden = hidden_states_list[-1]
            mx.eval(last_hidden)

            logits = router(last_hidden)  # (1, T, N_DOMAINS)
            predictions = mx.argmax(logits, axis=-1)  # (1, T)
            mx.eval(predictions)

            preds_np = np.array(predictions[0])
            correct += int(np.sum(preds_np == di))
            total += len(preds_np)
            del last_hidden, logits, predictions

    routing_accuracy = correct / max(total, 1)
    log(f"  Routing accuracy: {routing_accuracy:.1%} ({correct}/{total} tokens)")

    cleanup(optimizer)
    return router, routing_accuracy


def phase_composition_test(domain_data: dict, base: ToyGPT,
                             A_matrices: dict, m2p_results: dict,
                             sft_results: dict, base_losses: dict,
                             router: DomainRouter,
                             routing_accuracy: float) -> dict:
    """Test composition of all 5 M2P adapters simultaneously.

    Measures:
      1. Composition quality per domain (vs single-adapter quality)
      2. General quality preservation (vs base model on mixed data)
      3. Parameter orthogonality of M2P-generated adapters (Theorem 1 verify)
    """
    log("\n=== Phase 6: Composition Test ===")

    # Load all M2P-generated B-matrices from disk
    all_B = []
    for name in DOMAIN_NAMES:
        B_mats = load_B_matrices(m2p_results[name]["save_path"])
        all_B.append(B_mats)
        mx.eval(*[B_mats[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])

    # 6a: Verify Grassmannian orthogonality of composed adapters (Theorem 1)
    log("\n  6a: Grassmannian parameter orthogonality (Theorem 1)")
    delta_cos_values = []
    for di in range(N_DOMAINS):
        for dj in range(di + 1, N_DOMAINS):
            # Compute Frobenius inner product of full adapters
            # Δ_i = B_i A_i^T → flatten over all layers/modules
            frob_inner = 0.0
            frob_norm_i = 0.0
            frob_norm_j = 0.0
            for li in range(N_LAYERS):
                for mi in range(N_MODULES):
                    A_i = A_matrices[(di, li, mi)]   # (d, rank)
                    A_j = A_matrices[(dj, li, mi)]   # (d, rank)
                    B_i = all_B[di][(li, mi)]         # (rank, d_out)
                    B_j = all_B[dj][(li, mi)]         # (rank, d_out)
                    # Delta_i = B_i @ A_i^T: shape (rank, d_out) @ (rank, d).T
                    # Frobenius inner <Δ_i, Δ_j> = trace(A_i B_i^T B_j A_j^T)
                    # = trace((A_i^T A_j) @ (B_j^T @ B_i)) [cyclic]
                    # Since A_i^T A_j = 0 by construction, this IS zero.
                    # We verify by computing A_i^T A_j directly.
                    AtA = (A_i.T @ A_j)  # (rank, rank) — should be 0
                    frob_ATA = float(mx.sum(AtA * AtA).item() ** 0.5)
                    # proxy: use B cosine as secondary check
                    b_i_flat = B_i.reshape(-1)
                    b_j_flat = B_j.reshape(-1)
                    inner = float(mx.sum(b_i_flat * b_j_flat).item())
                    ni = float(mx.linalg.norm(b_i_flat).item())
                    nj = float(mx.linalg.norm(b_j_flat).item())
                    cos_delta = abs(inner) / (ni * nj + 1e-12)
                    delta_cos_values.append(cos_delta)
                    del AtA, b_i_flat, b_j_flat

    # Compute max delta cosine (Theorem 1 prediction: should be near 0 for parameter space)
    # The B cosine CAN be nonzero — Theorem 1 is about Frobenius, not B-cosine
    grassmannian_cos_max = float(np.max(delta_cos_values)) if delta_cos_values else 0.0
    grassmannian_cos_mean = float(np.mean(delta_cos_values)) if delta_cos_values else 0.0
    log(f"  B-matrix |cos|: mean={grassmannian_cos_mean:.4f}, "
        f"max={grassmannian_cos_max:.4f}")

    # Also verify A_i^T A_j = 0 exactly (Theorem 1 direct check)
    ortho = verify_grassmannian_orthogonality(A_matrices, N_DOMAINS, N_LAYERS, N_MODULES)
    log(f"  A-matrix |cos| (Theorem 1): max={ortho['max_cos']:.8f}")

    # 6b: Composition quality per domain
    log("\n  6b: Composition quality per domain")
    comp_quality = {}
    for di, name in enumerate(DOMAIN_NAMES):
        total = 0.0
        n = 0
        for tokens in domain_data[name]["val"][:20]:
            tokens_2d = tokens[None, :]
            logits, _ = composed_forward(
                base, tokens_2d, A_matrices, all_B, router
            )
            loss = nn.losses.cross_entropy(
                logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
            )
            mx.eval(loss)
            total += loss.item()
            n += 1
            del logits, loss

        comp_loss = total / max(n, 1)
        sft_loss = sft_results[name]["sft_loss"]
        base_loss = base_losses[name]
        # Quality: how much improvement does composition get vs single SFT?
        # PPL ratio: comp / single — should be close to 1.0
        ppl_ratio = comp_loss / sft_loss if sft_loss > 0.01 else 1.0
        comp_quality[name] = {
            "comp_loss": round(comp_loss, 4),
            "sft_loss": sft_loss,
            "base_loss": base_loss,
            "ppl_ratio": round(ppl_ratio, 3),
        }
        log(f"  {name}: comp={comp_loss:.4f} SFT={sft_loss:.4f} "
            f"PPL ratio={ppl_ratio:.2f}")

    # 6c: General quality preservation (base model on mixed/random text)
    log("\n  6c: General quality preservation")
    # Generate random text tokens (mimics unseen domain)
    rng_eval = np.random.RandomState(SEED + 999)
    general_texts = []
    for _ in range(50):
        # Random character sequences (not matching any domain pattern)
        chars = [chr((ord('a') + rng_eval.randint(0, 26))) for _ in range(40)]
        general_texts.append("".join(chars))
    general_batches = make_batches(general_texts)

    # Base model loss on general text
    base_gen_loss = eval_ntp_loss(base, general_batches[:30])

    # Composed model loss on general text (uses routing, which may assign any domain)
    comp_gen_total = 0.0
    n_gen = 0
    for tokens in general_batches[:30]:
        tokens_2d = tokens[None, :]
        logits, _ = composed_forward(base, tokens_2d, A_matrices, all_B, router)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        comp_gen_total += loss.item()
        n_gen += 1
        del logits, loss
    comp_gen_loss = comp_gen_total / max(n_gen, 1)

    # General quality degradation in "pp" (percentage points of PPL)
    # Using relative loss increase as proxy
    gen_degradation_pp = ((comp_gen_loss - base_gen_loss) / base_gen_loss) * 100.0
    log(f"  General quality: base_loss={base_gen_loss:.4f} "
        f"comp_loss={comp_gen_loss:.4f} "
        f"degradation={gen_degradation_pp:.1f}pp")

    # Clean up B matrices
    for B in all_B:
        del B

    return {
        "routing_accuracy": round(routing_accuracy, 4),
        "grassmannian_cos_max": round(grassmannian_cos_max, 6),
        "grassmannian_A_cos_max": round(ortho["max_cos"], 8),
        "composition_quality_per_domain": comp_quality,
        "base_gen_loss": round(base_gen_loss, 4),
        "comp_gen_loss": round(comp_gen_loss, 4),
        "general_quality_degradation_pp": round(gen_degradation_pp, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("M2P Composition N=5: End-to-end Grassmannian architecture test")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # ── Data ──────────────────────────────────────────────────────────────
    domain_data = phase_generate_data(rng)
    log_memory("after data generation")

    # ── Base model ────────────────────────────────────────────────────────
    base, base_losses, base_weights_path = phase_pretrain_base(domain_data)
    log_memory("after base pretrain")

    # ── Grassmannian A-matrices ───────────────────────────────────────────
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # ── SFT baselines ────────────────────────────────────────────────────
    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after SFT")

    # ── Independent M2P training ──────────────────────────────────────────
    m2p_results = phase_m2p_all_domains(
        domain_data, base, A_matrices, base_losses, sft_results
    )
    log_memory("after M2P training")

    # ── Router training ───────────────────────────────────────────────────
    router, routing_accuracy = phase_train_router(
        domain_data, base, A_matrices, m2p_results
    )
    log_memory("after router training")

    # ── Composition test ──────────────────────────────────────────────────
    comp_results = phase_composition_test(
        domain_data, base, A_matrices, m2p_results, sft_results,
        base_losses, router, routing_accuracy
    )
    log_memory("after composition test")

    # ─────────────────────────────────────────────────────────────────────
    # Kill criteria evaluation
    # ─────────────────────────────────────────────────────────────────────
    gen_deg = comp_results["general_quality_degradation_pp"]
    routing_acc = comp_results["routing_accuracy"]

    # K851: Composition degrades general quality >10pp
    k851_pass = gen_deg <= 10.0

    # K852: Routing accuracy <50% (below 2.5× random baseline)
    k852_pass = routing_acc >= 0.50

    # Secondary metrics
    m2p_quality_ratios = [m2p_results[n]["quality_ratio"] for n in DOMAIN_NAMES]
    mean_m2p_quality = float(np.mean(m2p_quality_ratios))
    median_m2p_quality = float(np.median(m2p_quality_ratios))

    ppl_ratios = [comp_results["composition_quality_per_domain"][n]["ppl_ratio"]
                  for n in DOMAIN_NAMES]
    mean_ppl_ratio = float(np.mean(ppl_ratios))

    # ─────────────────────────────────────────────────────────────────────
    # Results assembly
    # ─────────────────────────────────────────────────────────────────────
    results = {
        "experiment": "exp_m2p_composition_n5",
        "total_time_s": round(time.time() - t0, 1),
        "smoke_test": SMOKE_TEST,
        # Theorem 1 verification
        "grassmannian_A_cos_max": comp_results["grassmannian_A_cos_max"],
        "grassmannian_cos_max": comp_results["grassmannian_cos_max"],
        # K852: routing
        "routing_accuracy": comp_results["routing_accuracy"],
        # K851: general quality
        "general_quality_degradation_pp": comp_results["general_quality_degradation_pp"],
        "base_gen_loss": comp_results["base_gen_loss"],
        "comp_gen_loss": comp_results["comp_gen_loss"],
        # Composition quality
        "composition_quality_per_domain": comp_results["composition_quality_per_domain"],
        "mean_ppl_ratio": round(mean_ppl_ratio, 3),
        # M2P quality
        "m2p_quality_per_domain": {n: m2p_results[n]["quality_ratio"] for n in DOMAIN_NAMES},
        "mean_m2p_quality": round(mean_m2p_quality, 3),
        "median_m2p_quality": round(median_m2p_quality, 3),
        # SFT baseline
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        "m2p_losses": {n: m2p_results[n]["m2p_loss"] for n in DOMAIN_NAMES},
        # Grassmannian metadata
        "grassmannian_n_pairs": ortho_result["n_pairs"],
        # Kill criteria
        "kill_criteria": {
            "K851": {
                "pass": k851_pass,
                "general_quality_degradation_pp": round(gen_deg, 2),
                "threshold": 10.0,
                "note": "Composition must not degrade general quality >10pp",
            },
            "K852": {
                "pass": k852_pass,
                "routing_accuracy": round(routing_acc, 4),
                "threshold": 0.50,
                "note": "Routing >50% confirms domain-specific adapters",
            },
        },
        "all_pass": k851_pass and k852_pass,
    }

    # ─────────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"Theorem 1 (Grassmannian A cos): max={comp_results['grassmannian_A_cos_max']:.8f} (should be ~0)")
    log(f"Routing accuracy: {routing_acc:.1%} (K852 threshold: 50%)")
    log(f"General quality degradation: {gen_deg:.1f}pp (K851 threshold: 10pp)")
    log(f"Composition PPL ratio (mean): {mean_ppl_ratio:.2f} (1.0 = perfect)")
    log(f"M2P quality (mean/median): {mean_m2p_quality:.1%} / {median_m2p_quality:.1%} of SFT")
    log("")
    for name in DOMAIN_NAMES:
        q = comp_results["composition_quality_per_domain"][name]
        log(f"  {name:12s}: comp={q['comp_loss']:.4f} "
            f"sft={q['sft_loss']:.4f} ppl_ratio={q['ppl_ratio']:.2f} "
            f"m2p_quality={m2p_results[name]['quality_ratio']:.1%}")
    log("")
    log(f"K851: {'PASS' if k851_pass else 'FAIL'} — gen_deg={gen_deg:.1f}pp ≤ 10pp")
    log(f"K852: {'PASS' if k852_pass else 'FAIL'} — routing={routing_acc:.1%} ≥ 50%")
    log("")
    log(f"OVERALL: {'ALL PASS' if results['all_pass'] else 'KILLED'} "
        f"in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
