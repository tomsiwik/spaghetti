#!/usr/bin/env python3
"""M2P Scale Calibrated: L_preserve teaches M2P the optimal adapter scale.

Hypothesis: L_total = L_task + λ·L_preserve converges to a fixed point α*
where the task gradient and preservation gradient are in equilibrium (KKT
conditions). M2P learns to output adapters at α* automatically, without
being explicitly told the scale.

Experiment type: Type 1 (Proof Verification) + Type 2 (guided exploration of λ).

Single domain only (arithmetic). Multi-domain joint training is structurally
impossible due to gradient conflicts (Finding #341, #342).

Kill criteria:
  K849: M2P generates adapters that degrade general quality >10pp  [FAIL = bad]
  K850: M2P fails to self-calibrate (all outputs same magnitude)   [FAIL = bad]

Comparison:
  (a) M2P + L_preserve (Theorem 1 prediction: self-calibrates to α* ∈ [3,15])
  (b) M2P without L_preserve (baseline: unconstrained scale, expected α → large)

Math reference: experiments/models/m2p_scale_calibrated/MATH.md
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

# ── Architecture constants ────────────────────────────────────────────────

D_MODEL = 256       # Toy GPT-2 scale (matching delegation spec)
N_LAYERS = 2        # Toy GPT layers
N_HEADS = 4         # Attention heads
VOCAB_SIZE = 128    # Character-level vocab
BLOCK_SIZE = 48     # Context window
LORA_RANK = 4       # LoRA rank
D_M2P = 64          # M2P internal hidden dimension (separate from base model D_MODEL=256)
N_MEMORY = 8        # M2P memory tokens
M2P_LAYERS = 2      # M2P transformer blocks (lightweight, single domain)

# LoRA target modules (per-layer): wq, wk, wv, wo, fc1
MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]  # output dims

# Training config (fast enough for micro scale)
BASE_STEPS = 800 if not SMOKE_TEST else 50        # Steps to train base GPT
SFT_STEPS  = 400 if not SMOKE_TEST else 30        # Steps to train SFT reference
M2P_STEPS  = 600 if not SMOKE_TEST else 40        # Steps to train M2P
LAMBDA_PRESERVE = 0.1                             # Regularization weight λ
N_CONTEXT_VARIANTS = 20 if not SMOKE_TEST else 5  # Distinct task inputs for K850

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
    """Release MLX memory between phases (CODING_GUIDELINES §2)."""
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


# ── Data generation ───────────────────────────────────────────────────────

def generate_arithmetic_tokens(n_samples: int, rng: np.random.RandomState,
                                difficulty: str = "mixed") -> np.ndarray:
    """Generate arithmetic sequences as token arrays.

    Format: "a+b+c+...=result\n" repeated to fill BLOCK_SIZE.
    Characters encoded as ASCII (within VOCAB_SIZE=128).

    difficulty:
      'easy':  2-3 operands, 1-digit each
      'hard':  4-6 operands, 2-3 digits each
      'mixed': random mix (default)
    """
    samples = []
    for _ in range(n_samples):
        if difficulty == "easy" or (difficulty == "mixed" and rng.rand() < 0.5):
            n_ops = rng.randint(2, 4)
            nums = rng.randint(1, 10, size=n_ops).tolist()
        else:
            n_ops = rng.randint(4, 7)
            nums = rng.randint(10, 200, size=n_ops).tolist()

        expr = "+".join(str(n) for n in nums) + "=" + str(sum(nums))
        # Tile to fill BLOCK_SIZE
        text = (expr + "\n") * (BLOCK_SIZE // (len(expr) + 1) + 2)
        tokens = np.array([ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]],
                          dtype=np.int32)
        samples.append(tokens)
    return np.stack(samples, axis=0)  # (n_samples, BLOCK_SIZE+1)


def generate_general_tokens(n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate random general-purpose token sequences.

    Simulates 'general text' — random character sequences that a well-functioning
    language model should be able to handle. Used for L_preserve.
    """
    samples = []
    for _ in range(n_samples):
        # Mix of alphabetic, numeric, punctuation patterns
        tokens = rng.randint(32, 127, size=BLOCK_SIZE + 1).astype(np.int32)
        samples.append(tokens)
    return np.stack(samples, axis=0)


def generate_context_tokens(rng: np.random.RandomState, difficulty: str,
                            n_samples: int = 4) -> np.ndarray:
    """Generate M2P context tokens from arithmetic examples of given difficulty.

    The M2P sees actual arithmetic example tokens (not text descriptions).
    This is cleaner than text descriptions: the arithmetic tokens naturally
    embed difficulty (short sequences = easy, long = hard).

    difficulty: 'easy' | 'hard'
    Returns: (n_samples, BLOCK_SIZE) context token array
    """
    return generate_arithmetic_tokens(n_samples, rng, difficulty=difficulty)


# Difficulty index for K850: easy=0, hard=1
DIFFICULTY_LABELS = ["easy", "hard"]

def get_task_context(rng: np.random.RandomState, difficulty: str) -> np.ndarray:
    """Get a single context token array for the given difficulty.

    M2P conditions on this context to generate B matrices.
    For K850: easy contexts should generate smaller B norms than hard ones
    (if self-calibration is working — Theorem 1, step 5).
    """
    tokens = generate_arithmetic_tokens(1, rng, difficulty=difficulty)
    return tokens[0]  # (BLOCK_SIZE+1,) — take first sample


# ── Toy GPT-2 scale ──────────────────────────────────────────────────────

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
    """Toy GPT-2 scale: 2 layers, d=256, 4 heads, vocab=128."""

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

    def get_hidden_states(self, tokens):
        """Return list of hidden states from each layer (for M2P input)."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)  # (B, T, D)
        return states


class ToyGPTWithLoRA(nn.Module):
    """ToyGPT with LoRA adapters applied in-line.

    For each layer and each target module, applies:
      output = base_output + scale * (x @ A.T) @ B.T
    where A is a frozen Grassmannian matrix and B is the M2P output.

    Scale is applied to the full ΔW = A @ B contribution (not absorbed into A or B).
    """

    def __init__(self, base_weights: dict, A_matrices: dict, scale: float = 5.0):
        """
        base_weights: saved weights from a ToyGPT instance (frozen)
        A_matrices: dict (layer_idx, module_name) → mx.array (d_in, rank)
        scale: adapter scale α (the parameter we're studying)
        """
        super().__init__()
        # Store base model parameters as frozen leaves
        # We store them directly so the optimizer can't touch them
        self._base_wte = base_weights["wte"]
        self._base_wpe = base_weights["wpe"]
        self._base_norm_f_weight = base_weights["norm_f_weight"]
        self._base_lm_head = base_weights["lm_head"]
        self._base_block_weights = base_weights["blocks"]

        self.A_matrices = A_matrices  # frozen, not in trainable params
        self.scale = scale

        # B-matrices: the ONLY trainable parameters
        # Shape: (rank, d_out) per (layer, module)
        self.B = {}
        for li in range(N_LAYERS):
            for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
                key = f"B_{li}_{mname}"
                self.B[key] = mx.zeros((LORA_RANK, d_out))

    def set_B_from_m2p(self, B_matrices: dict):
        """Set B-matrices from M2P output (no gradient through this assignment).
        Used for inference / measurement only.
        """
        for li in range(N_LAYERS):
            for mi, mname in enumerate(MODULE_NAMES):
                key = f"B_{li}_{mname}"
                if (li, mi) in B_matrices:
                    self.B[key] = B_matrices[(li, mi)]

    def forward_with_B(self, tokens, B_matrices: dict):
        """Forward pass with explicit B-matrices (used during M2P training)."""
        B_batch, T = tokens.shape
        pos = mx.arange(T)

        # Embedding (no adapter on embedding layers)
        x = self._base_wte[tokens] + self._base_wpe[mx.broadcast_to(pos[None, :], (B_batch, T))]

        for li in range(N_LAYERS):
            bw = self._base_block_weights[li]

            # Norm1
            rms1 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            n1 = x * rms1 * bw["norm1_weight"]

            # Attention with LoRA on wq, wk, wv, wo
            def lora_linear(h, W_base, module_idx):
                out = h @ W_base.T
                A = self.A_matrices[(li, module_idx)]   # (d_in, rank)
                B = B_matrices.get((li, module_idx), None)
                if B is not None:
                    # LoRA contribution: (B, T, d_in) @ (d_in, rank) @ (rank, d_out)
                    lora_out = (h @ A) @ B.T
                    out = out + self.scale * lora_out
                return out

            hd = D_MODEL // N_HEADS
            q = lora_linear(n1, bw["wq"], 0).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
            k = lora_linear(n1, bw["wk"], 1).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
            v = lora_linear(n1, bw["wv"], 2).reshape(B_batch, T, N_HEADS, hd).transpose(0, 2, 1, 3)
            mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
            scale_attn = hd ** -0.5
            attn_w = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale_attn + mask, axis=-1)
            attn_out = (attn_w @ v).transpose(0, 2, 1, 3).reshape(B_batch, T, D_MODEL)
            attn_out = lora_linear(attn_out, bw["wo"], 3)
            x = x + attn_out

            # Norm2
            rms2 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            n2 = x * rms2 * bw["norm2_weight"]

            # MLP with LoRA on fc1
            fc1_out = lora_linear(n2, bw["fc1"], 4)
            fc1_act = nn.gelu(fc1_out)
            fc2_out = fc1_act @ bw["fc2"].T
            x = x + fc2_out

        # Final norm + head (no adapter)
        rms_f = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        x = x * rms_f * self._base_norm_f_weight
        return x @ self._base_lm_head.T


# ── M2P Transformer (single domain) ─────────────────────────────────────

class M2PBlock(nn.Module):
    """One M2P block (row attention: attend across memory tokens)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)

    def __call__(self, x):
        """x: (1, N_MEMORY, d) — single 'layer' of memory tokens."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class M2PTransformer(nn.Module):
    """Lightweight M2P Transformer for single-domain scale calibration.

    Takes task description tokens → generates B-matrices for all LoRA slots.

    Input: task description tokens (1, T_desc) → (1, N_MEMORY, d_m2p)
    Output: dict of (layer_idx, module_idx) → B matrix (rank, d_out)

    Difference from prior experiments (m2p_distillation_toy, m2p_domain_conditioned):
    - Single domain only: no domain embedding, no mode collapse from gradient conflicts
    - Scale is NOT fixed externally: M2P learns optimal B magnitude via L_preserve
    - Smaller N_MEMORY (8 vs 32): sufficient for single-domain task variation
    """

    def __init__(self):
        super().__init__()
        # Task encoder: embed task description tokens into M2P hidden space
        # D_M2P=64 is the M2P's own dimension, separate from base model D_MODEL=256
        self.task_embed = nn.Embedding(VOCAB_SIZE, D_M2P)
        # Input projection from base model dimension to M2P hidden dimension
        # (not used in this simplified version; task desc is already in D_M2P space)
        # Memory tokens: learned starting memory in D_M2P space
        scale_init = math.sqrt(2.0 / (1 + D_M2P))
        self.memory = mx.random.normal((1, N_MEMORY, D_M2P)) * scale_init
        # M2P blocks operating in D_M2P space
        # Using 4 heads in D_M2P=64 space: head_dim = 16
        self.blocks = [M2PBlock(D_M2P, n_heads=4) for _ in range(M2P_LAYERS)]
        self.final_norm = RMSNorm(D_M2P)
        # Output projection: D_M2P → B-matrix parameters
        # Total B params: N_LAYERS * Σ(rank * d_out) per module
        # = 2 * (4*256 + 4*256 + 4*256 + 4*256 + 4*1024) = 2 * 8192 = 16384 params
        # B_proj: (N_MEMORY * D_M2P, total_B_params) = (512, 16384) = 8.4M params
        # This is much smaller than the prior 34M params at D_M2P=256
        total_B_params = sum(LORA_RANK * d_out for d_out in MODULE_OUT_DIMS) * N_LAYERS
        # Initialize B_proj with near-zero weights to start at small adapter scale
        # This encourages M2P to learn the right scale, not start enormous
        self.B_proj = nn.Linear(D_M2P * N_MEMORY, total_B_params, bias=False)
        # Near-zero init: scale down by 10x to start B norms near zero
        # M2P then learns to scale up as needed (via L_task gradient)
        # while L_preserve keeps it from growing too large

    def _init_B_proj_small(self):
        """Scale down B_proj weights to start with near-zero B norms.

        Called after construction. This is critical: with default init,
        B norms are O(sqrt(total_B_params / (N_MEMORY * D_M2P))) ≈ 4.5,
        which is already in the scale=5 safe range. We want to START
        near zero so the gradient dynamics determine the learned scale.
        """
        # Get current weight and scale down
        w = self.B_proj.weight   # (total_B_params, N_MEMORY * D_M2P)
        # Scale by 0.01 to start B outputs near zero
        self.B_proj.weight = w * 0.01

    def __call__(self, task_tokens):
        """
        task_tokens: (1, T_desc) integer token ids
        Returns: dict of (layer_idx, module_idx) → B matrix (rank, d_out)
        """
        # Encode task description into M2P hidden space
        task_emb = self.task_embed(task_tokens)   # (1, T_desc, D_M2P)
        task_ctx = mx.mean(task_emb, axis=1, keepdims=True)  # (1, 1, D_M2P)

        # Add task context to memory tokens
        mem = self.memory + mx.broadcast_to(task_ctx, (1, N_MEMORY, D_M2P))

        # Run M2P blocks (in D_M2P space)
        for block in self.blocks:
            mem = block(mem)
        mem = self.final_norm(mem)  # (1, N_MEMORY, D_M2P)

        # Project to B-matrix space
        flat = mem.reshape(1, -1)  # (1, N_MEMORY * D_M2P)
        B_flat = self.B_proj(flat)[0]  # (total_B_params,)

        # Unpack into per-(layer, module) B matrices
        B_matrices = {}
        offset = 0
        for li in range(N_LAYERS):
            for mi, d_out in enumerate(MODULE_OUT_DIMS):
                n = LORA_RANK * d_out
                B_ij = B_flat[offset:offset + n].reshape(LORA_RANK, d_out)
                B_matrices[(li, mi)] = B_ij
                offset += n
        return B_matrices


# ── Grassmannian A-matrices ───────────────────────────────────────────────

def generate_grassmannian_A() -> dict:
    """Generate frozen orthogonal A-matrices via QR decomposition.

    Returns dict of (layer_idx, module_idx) → mx.array (d_in, rank)

    Theorem 1 (m2p_distillation_toy MATH.md): A_i^T A_j = 0 for i ≠ j
    across domains → composition interference = 0 by construction.

    For this single-domain experiment: one A-matrix per (layer, module) slot.
    d_in is always D_MODEL (256) for all modules.
    """
    rng = np.random.RandomState(SEED + 100)  # separate seed from data generation
    A_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(len(MODULE_NAMES)):
            # All modules: input dimension is D_MODEL
            Q, _ = np.linalg.qr(rng.randn(D_MODEL, LORA_RANK).astype(np.float32))
            A_matrices[(li, mi)] = mx.array(Q[:, :LORA_RANK])  # (D_MODEL, LORA_RANK)
    return A_matrices


# ── Loss functions ────────────────────────────────────────────────────────

def causal_lm_loss(model, tokens):
    """Causal LM cross-entropy loss for nn.Module.

    tokens: (B, T+1) full sequence (input + target)
    model: nn.Module, called with tokens[:, :-1] → logits (B, T, vocab)
    Loss: predict tokens[:, 1:] from all T positions.
    """
    logits = model(tokens[:, :-1])   # (B, T, vocab), T = tokens.shape[1] - 1
    targets = tokens[:, 1:]          # (B, T)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def cross_entropy_loss(logits, tokens):
    """Cross-entropy loss given pre-computed logits.

    logits: (B, T, vocab) — already computed via _forward_with_b(tokens, ...)
    tokens: (B, T+1) — full token sequence

    Note: _forward_with_b internally uses tokens[:, :-1] as input,
    so logits already has T = tokens.shape[1] - 1 positions.
    Targets are tokens[:, 1:] which also has T positions. Shapes match.
    """
    targets = tokens[:, 1:]   # (B, T)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


# ── Phase functions ──────────────────────────────────────────────────────

def phase_pretrain_base(rng: np.random.RandomState) -> dict:
    """Phase 1: pre-train the base ToyGPT on arithmetic data.

    Returns saved base model weights (frozen for all subsequent phases).
    """
    log("\n=== Phase 1: Pre-train Base GPT ===")
    mx.random.seed(SEED)

    model = ToyGPT()
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    log(f"  ToyGPT params: {n_params:,}")

    optimizer = opt.AdamW(learning_rate=LR, weight_decay=0.01)

    # Generate training data: arithmetic sequences
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
        if (step + 1) % 100 == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={losses[-1]:.4f}")
    gc.enable()

    final_loss = float(np.mean(losses[-50:]))
    log(f"  Final base loss (mean last 50): {final_loss:.4f}")

    # Save base weights as numpy arrays for use in downstream phases
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

    # Compute base model PPL on arithmetic and general data
    arithmetic_test = generate_arithmetic_tokens(BATCH_SIZE, rng, difficulty="mixed")
    general_test = generate_general_tokens(BATCH_SIZE, rng)

    arith_batch = mx.array(arithmetic_test)
    gen_batch = mx.array(general_test)

    logits_arith = model(arith_batch[:, :-1])
    logits_gen = model(gen_batch[:, :-1])
    mx.eval(logits_arith, logits_gen)

    loss_arith = cross_entropy_loss(logits_arith, arith_batch).item()
    loss_gen = cross_entropy_loss(logits_gen, gen_batch).item()

    log(f"  Base model: arith_CE={loss_arith:.4f}, general_CE={loss_gen:.4f}")
    log(f"  Base model: arith_PPL={math.exp(loss_arith):.2f}, general_PPL={math.exp(loss_gen):.2f}")

    results = {
        "base_final_loss": final_loss,
        "base_arith_ce": loss_arith,
        "base_arith_ppl": math.exp(loss_arith),
        "base_general_ce": loss_gen,
        "base_general_ppl": math.exp(loss_gen),
        "n_params": n_params,
    }

    cleanup(model, optimizer, arith_batch, gen_batch)
    return base_weights_np, results


def _make_mlx_base_weights(base_weights_np: dict) -> dict:
    """Convert numpy base weights to MLX arrays for forward passes."""
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
    """Differentiable forward pass of adapted GPT.

    tokens: (B, T+1) — full token sequence including target
    B_matrices: dict (li, mi) → mx.array (rank, d_out)
    Returns: logits (B, T, vocab) using tokens[:, :-1] as input
    """
    inp = tokens[:, :-1]   # (B, T)
    B_batch, T = inp.shape
    pos = mx.arange(T)

    # wte[inp]: (B, T, D_MODEL) — token embeddings
    # wpe[pos]: (T, D_MODEL) — position embeddings, broadcast to (B, T, D_MODEL)
    x = base_mlx["wte"][inp] + base_mlx["wpe"][pos]

    for li in range(N_LAYERS):
        bw = base_mlx["blocks"][li]

        # Norm1
        rms1 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        n1 = x * rms1 * bw["norm1_weight"]

        def lora_proj(h, W, li_, mi_):
            out = h @ W.T
            A = A_matrices[(li_, mi_)]      # (d_in, rank)
            B_mat = B_matrices.get((li_, mi_))
            if B_mat is not None:
                # h: (B, T, d_in)
                # A: (d_in, rank) → h@A: (B, T, rank)
                # B_mat: (rank, d_out) → (h@A)@B_mat: (B, T, d_out)
                # ΔW = A @ B_mat, applied as h @ A @ B_mat
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

        # Norm2
        rms2 = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        n2 = x * rms2 * bw["norm2_weight"]

        # MLP with LoRA on fc1
        fc1_out = lora_proj(n2, bw["fc1"], li, 4)
        fc1_act = nn.gelu(fc1_out)
        fc2_out = fc1_act @ bw["fc2"].T
        x = x + fc2_out

    rms_f = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    x = x * rms_f * base_mlx["norm_f_weight"]
    return x @ base_mlx["lm_head"].T


def phase_train_sft_reference(base_weights_np: dict, A_matrices: dict,
                               rng: np.random.RandomState) -> dict:
    """Phase 2: train a reference SFT (adapter with fixed scale=5, direct gradient).

    This is the upper bound on quality — a directly-trained adapter at safe scale.
    Used to normalize M2P quality as a fraction of SFT quality.
    """
    log("\n=== Phase 2: Train SFT Reference Adapter (scale=5) ===")
    mx.random.seed(SEED + 1)

    base_mlx = _make_mlx_base_weights(base_weights_np)
    SCALE_SFT = 5.0

    # Trainable B-matrices as a plain dict of mx.arrays.
    # mx.grad works on pytrees (dicts), so this is the correct MLX idiom.
    B_params = {}
    for li in range(N_LAYERS):
        for mi, d_out in enumerate(MODULE_OUT_DIMS):
            B_params[f"B_{li}_{mi}"] = mx.zeros((LORA_RANK, d_out))

    data_np = generate_arithmetic_tokens(BATCH_SIZE * (SFT_STEPS + 20), rng, difficulty="mixed")

    def sft_loss(B_p, tokens):
        B_dict = {(li, mi): B_p[f"B_{li}_{mi}"]
                  for li in range(N_LAYERS)
                  for mi in range(len(MODULE_NAMES))}
        logits = _forward_with_b(tokens, base_mlx, A_matrices, B_dict, SCALE_SFT)
        return cross_entropy_loss(logits, tokens)

    # Adam state for B_params (manual Adam since B_params is not an nn.Module)
    m_state = {k: mx.zeros_like(v) for k, v in B_params.items()}
    v_state = {k: mx.zeros_like(v) for k, v in B_params.items()}
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    losses = []
    gc.disable()
    for step in range(SFT_STEPS):
        idx = (step * BATCH_SIZE) % (len(data_np) - BATCH_SIZE)
        batch = mx.array(data_np[idx:idx + BATCH_SIZE])

        loss_val, grads = mx.value_and_grad(sft_loss)(B_params, batch)

        # Adam update
        t = step + 1
        lr_t = LR * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        for key in B_params:
            g = grads[key]
            m_state[key] = beta1 * m_state[key] + (1 - beta1) * g
            v_state[key] = beta2 * v_state[key] + (1 - beta2) * g * g
            B_params[key] = B_params[key] - lr_t * m_state[key] / (mx.sqrt(v_state[key]) + eps_adam)

        mx.eval(*list(B_params.values()), loss_val)
        losses.append(loss_val.item())
        if (step + 1) % 100 == 0:
            log(f"  Step {step+1}/{SFT_STEPS}: loss={losses[-1]:.4f}")
    gc.enable()

    # Evaluate SFT adapter quality
    test_arith = generate_arithmetic_tokens(BATCH_SIZE, rng, difficulty="mixed")
    test_gen = generate_general_tokens(BATCH_SIZE, rng)
    B_dict = {(li, mi): B_params[f"B_{li}_{mi}"]
              for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))}

    arith_batch = mx.array(test_arith)
    gen_batch = mx.array(test_gen)

    logits_a = _forward_with_b(arith_batch, base_mlx, A_matrices, B_dict, SCALE_SFT)
    logits_g = _forward_with_b(gen_batch, base_mlx, A_matrices, {}, SCALE_SFT)  # no adapter for general
    logits_g_adapted = _forward_with_b(gen_batch, base_mlx, A_matrices, B_dict, SCALE_SFT)
    mx.eval(logits_a, logits_g, logits_g_adapted)

    sft_arith_ce = cross_entropy_loss(logits_a, arith_batch).item()
    base_gen_ce = cross_entropy_loss(logits_g, gen_batch).item()
    sft_gen_ce = cross_entropy_loss(logits_g_adapted, gen_batch).item()

    # Compute B-matrix Frobenius norms
    b_norms = [float(mx.linalg.norm(B_dict[(li, mi)].reshape(-1)).item())
               for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))]
    mean_norm = float(np.mean(b_norms))

    log(f"  SFT arith CE: {sft_arith_ce:.4f} (PPL={math.exp(sft_arith_ce):.2f})")
    log(f"  SFT general CE: {sft_gen_ce:.4f} (PPL={math.exp(sft_gen_ce):.2f})")
    log(f"  General CE without adapter: {base_gen_ce:.4f}")
    log(f"  General degradation: {(sft_gen_ce - base_gen_ce):.4f} nats "
        f"({((sft_gen_ce - base_gen_ce)/base_gen_ce*100):.1f}%)")
    log(f"  Mean B-matrix Frobenius norm (SFT scale=5): {mean_norm:.4f}")

    results = {
        "sft_arith_ce": sft_arith_ce,
        "sft_arith_ppl": math.exp(sft_arith_ce),
        "sft_gen_ce": sft_gen_ce,
        "sft_gen_ppl": math.exp(sft_gen_ce),
        "sft_base_gen_ce": base_gen_ce,
        "sft_gen_degradation_pct": (sft_gen_ce - base_gen_ce) / base_gen_ce * 100,
        "sft_b_norm_mean": mean_norm,
        "sft_scale": SCALE_SFT,
    }

    cleanup(base_mlx, arith_batch, gen_batch, logits_a, logits_g, logits_g_adapted)
    return B_dict, results


def phase_train_m2p(base_weights_np: dict, A_matrices: dict,
                    rng: np.random.RandomState,
                    use_preserve: bool) -> dict:
    """Phase 3/4: train M2P with or without L_preserve.

    This is the core of the experiment. M2P generates B-matrices from a
    task description. With L_preserve, Theorem 1 predicts self-calibration.
    Without L_preserve, scale is unconstrained (baseline).

    Returns: dict with training metrics, final B-matrix statistics.
    """
    label = "WITH L_preserve" if use_preserve else "WITHOUT L_preserve (baseline)"
    log(f"\n=== Phase: Train M2P {label} ===")
    mx.random.seed(SEED + (2 if use_preserve else 3))

    base_mlx = _make_mlx_base_weights(base_weights_np)

    # Fixed adapter scale (the LoRA scale multiplier α in ΔW = scale * A @ B)
    # We set scale=1.0 here and let M2P learn ||B||_F directly.
    # The "learned scale" is measured as ||B||_F at convergence.
    ADAPTER_SCALE = 1.0

    m2p = M2PTransformer()
    m2p._init_B_proj_small()   # Scale down B_proj to start with near-zero B norms
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")
    log(f"  M2P hidden dim: D_M2P={D_M2P} (separate from base D_MODEL={D_MODEL})")

    optimizer = opt.AdamW(learning_rate=LR * 0.5, weight_decay=0.01)

    # Training data: arithmetic (task domain) + general (preservation domain)
    # Pre-generate BOTH easy and hard contexts for training
    # This is critical for K850: M2P must see varying difficulty at training time
    n_train = BATCH_SIZE * (M2P_STEPS + 50)
    arith_easy_data = generate_arithmetic_tokens(n_train // 2, rng, difficulty="easy")
    arith_hard_data = generate_arithmetic_tokens(n_train - n_train // 2, rng, difficulty="hard")
    gen_data = generate_general_tokens(n_train, rng)

    def m2p_loss_fn(m2p, ctx_tokens, arith_tokens, gen_tokens):
        """M2P loss: generates B from context, applies to base model.

        ctx_tokens: (1, BLOCK_SIZE+1) — arithmetic context (easy or hard)
        arith_tokens: (B, BLOCK_SIZE+1) — arithmetic sequences for L_task
        gen_tokens: (B, BLOCK_SIZE+1) — general sequences for L_preserve

        The context (ctx_tokens) is what M2P conditions on — different contexts
        → potentially different B matrices → self-calibration mechanism.
        """
        # M2P generates B matrices from the context
        task_inp = ctx_tokens[:, :BLOCK_SIZE]  # (1, BLOCK_SIZE) — input tokens only
        B_matrices = m2p(task_inp)

        # L_task: cross-entropy on arithmetic sequences
        logits_task = _forward_with_b(arith_tokens, base_mlx, A_matrices,
                                      B_matrices, ADAPTER_SCALE)
        L_task = cross_entropy_loss(logits_task, arith_tokens)

        if use_preserve:
            # L_preserve: cross-entropy on general sequences (with adapter applied)
            logits_preserve = _forward_with_b(gen_tokens, base_mlx, A_matrices,
                                              B_matrices, ADAPTER_SCALE)
            L_preserve = cross_entropy_loss(logits_preserve, gen_tokens)
            L_total = L_task + LAMBDA_PRESERVE * L_preserve
        else:
            L_preserve = mx.array(0.0)
            L_total = L_task

        return L_total, L_task, L_preserve

    losses_total, losses_task, losses_preserve = [], [], []
    gc.disable()
    for step in range(M2P_STEPS):
        # Alternate between easy and hard contexts during training
        # This teaches M2P to vary its output based on context difficulty
        if step % 2 == 0:
            ctx_np = arith_easy_data[(step // 2) % len(arith_easy_data)][None, :]
            arith_np = arith_easy_data[(step // 2) % len(arith_easy_data)]
        else:
            ctx_np = arith_hard_data[(step // 2) % len(arith_hard_data)][None, :]
            arith_np = arith_hard_data[(step // 2) % len(arith_hard_data)]

        # Batch of arithmetic samples (same difficulty as context)
        idx_a = step % (max(len(arith_easy_data), len(arith_hard_data)) - BATCH_SIZE)
        arith_batch_np = (arith_easy_data if step % 2 == 0 else arith_hard_data)
        arith_batch_np = arith_batch_np[idx_a:idx_a + BATCH_SIZE]
        if len(arith_batch_np) < BATCH_SIZE:
            arith_batch_np = arith_batch_np[:1].repeat(BATCH_SIZE, axis=0)

        idx_g = (step * BATCH_SIZE) % (len(gen_data) - BATCH_SIZE)
        ctx_batch = mx.array(ctx_np)       # (1, BLOCK_SIZE+1)
        arith_batch = mx.array(arith_batch_np)   # (B, BLOCK_SIZE+1)
        gen_batch = mx.array(gen_data[idx_g:idx_g + BATCH_SIZE])

        def loss_for_grad(m2p):
            # MLX nn.value_and_grad uses the FIRST return value for gradients.
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

    # ── Evaluate K849: general quality degradation ────────────────────────
    log(f"\n  === Evaluation: K849 (general quality degradation) ===")
    # Use a "mixed" context for K849 evaluation
    ctx_eval_np = generate_arithmetic_tokens(1, rng, difficulty="mixed")
    ctx_eval = mx.array(ctx_eval_np[:, :BLOCK_SIZE])  # (1, BLOCK_SIZE)
    B_final = m2p(ctx_eval)
    mx.eval(*list(B_final.values()))

    eval_gen = generate_general_tokens(BATCH_SIZE, rng)
    gen_eval_batch = mx.array(eval_gen)

    # General quality with adapter
    logits_g_adapted = _forward_with_b(gen_eval_batch, base_mlx, A_matrices, B_final, ADAPTER_SCALE)
    mx.eval(logits_g_adapted)
    gen_ce_adapted = cross_entropy_loss(logits_g_adapted, gen_eval_batch).item()

    # General quality without adapter (base model, empty B_matrices)
    logits_g_base = _forward_with_b(gen_eval_batch, base_mlx, A_matrices, {}, ADAPTER_SCALE)
    mx.eval(logits_g_base)
    gen_ce_base = cross_entropy_loss(logits_g_base, gen_eval_batch).item()

    gen_degradation_pct = (gen_ce_adapted - gen_ce_base) / gen_ce_base * 100.0
    k849_pass = gen_degradation_pct < 10.0

    # Also eval arithmetic quality
    eval_arith = generate_arithmetic_tokens(BATCH_SIZE, rng, difficulty="mixed")
    arith_eval_batch = mx.array(eval_arith)
    logits_a = _forward_with_b(arith_eval_batch, base_mlx, A_matrices, B_final, ADAPTER_SCALE)
    mx.eval(logits_a)
    arith_ce_adapted = cross_entropy_loss(logits_a, arith_eval_batch).item()

    log(f"  Arithmetic CE with adapter: {arith_ce_adapted:.4f} (PPL={math.exp(arith_ce_adapted):.2f})")
    log(f"  General CE base:            {gen_ce_base:.4f} (PPL={math.exp(gen_ce_base):.2f})")
    log(f"  General CE with adapter:    {gen_ce_adapted:.4f} (PPL={math.exp(gen_ce_adapted):.2f})")
    log(f"  General degradation:        {gen_degradation_pct:.2f}%")
    log(f"  K849: {'PASS' if k849_pass else 'FAIL'} (threshold: <10%, measured: {gen_degradation_pct:.2f}%)")

    # ── Evaluate K850: adapter magnitude self-calibration ────────────────
    # Theorem 1, Step 5: harder tasks (larger task gradient) → larger B norm
    # Test: easy vs hard contexts should produce different B norms
    log(f"\n  === Evaluation: K850 (adapter magnitude self-calibration) ===")

    easy_norms, hard_norms = [], []
    for i in range(N_CONTEXT_VARIANTS):
        # Easy context
        ctx_easy_np = generate_arithmetic_tokens(1, rng, difficulty="easy")
        ctx_easy = mx.array(ctx_easy_np[:, :BLOCK_SIZE])  # (1, BLOCK_SIZE)
        B_easy = m2p(ctx_easy)
        mx.eval(*list(B_easy.values()))
        n_easy = float(np.mean([mx.linalg.norm(B_easy[(li, mi)].reshape(-1)).item()
                                for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))]))
        easy_norms.append(n_easy)

        # Hard context
        ctx_hard_np = generate_arithmetic_tokens(1, rng, difficulty="hard")
        ctx_hard = mx.array(ctx_hard_np[:, :BLOCK_SIZE])  # (1, BLOCK_SIZE)
        B_hard = m2p(ctx_hard)
        mx.eval(*list(B_hard.values()))
        n_hard = float(np.mean([mx.linalg.norm(B_hard[(li, mi)].reshape(-1)).item()
                                for li in range(N_LAYERS) for mi in range(len(MODULE_NAMES))]))
        hard_norms.append(n_hard)

        log(f"  Sample {i:2d}: easy ||B||_F = {n_easy:.4f}  hard ||B||_F = {n_hard:.4f}  "
            f"ratio = {n_hard / (n_easy + 1e-8):.3f}")

    mean_easy = float(np.mean(easy_norms))
    mean_hard = float(np.mean(hard_norms))
    all_norms = easy_norms + hard_norms
    magnitude_mean = float(np.mean(all_norms))
    magnitude_std = float(np.std(all_norms))
    magnitude_cv = magnitude_std / (magnitude_mean + 1e-8)

    # K850 primary criterion: CV > 0.05 across all contexts (easy + hard)
    # This is satisfied if easy and hard contexts produce different magnitudes
    k850_pass = magnitude_cv > 0.05

    # Secondary signal: hard > easy (Theorem 1, Step 5 prediction)
    hard_gt_easy = mean_hard > mean_easy
    hard_easy_ratio = mean_hard / (mean_easy + 1e-8)

    log(f"  Easy contexts:  mean ||B||_F = {mean_easy:.4f}")
    log(f"  Hard contexts:  mean ||B||_F = {mean_hard:.4f}")
    log(f"  Hard/Easy ratio: {hard_easy_ratio:.3f} (Theorem 1 predicts > 1.0 if self-calibrating)")
    log(f"  Overall CV = std/mean = {magnitude_cv:.4f}")
    log(f"  K850: {'PASS' if k850_pass else 'FAIL'} "
        f"(threshold: CV>0.05, measured CV={magnitude_cv:.4f})")

    # Inferred adapter scale
    scale_learned = magnitude_mean  # ||B||_F is the effective scale when ADAPTER_SCALE=1

    results = {
        "use_preserve": use_preserve,
        "label": label,
        "final_loss_total": float(np.mean(losses_total[-30:])),
        "final_loss_task": float(np.mean(losses_task[-30:])),
        "final_loss_preserve": float(np.mean(losses_preserve[-30:])) if use_preserve else None,
        "arith_ce_adapted": arith_ce_adapted,
        "arith_ppl_adapted": math.exp(arith_ce_adapted),
        "gen_ce_base": gen_ce_base,
        "gen_ce_adapted": gen_ce_adapted,
        "gen_ppl_adapted": math.exp(gen_ce_adapted),
        "general_degradation_pp": gen_degradation_pct,
        "k849_pass": k849_pass,
        "adapter_magnitude_mean": magnitude_mean,
        "adapter_magnitude_std": magnitude_std,
        "adapter_magnitude_cv": magnitude_cv,
        "easy_norms": easy_norms,
        "hard_norms": hard_norms,
        "mean_easy_norm": mean_easy,
        "mean_hard_norm": mean_hard,
        "hard_easy_ratio": hard_easy_ratio,
        "hard_gt_easy": hard_gt_easy,
        "k850_pass": k850_pass,
        "scale_learned": magnitude_mean,
        "n_m2p_params": n_params,
    }

    cleanup(m2p, optimizer, base_mlx, gen_eval_batch, arith_eval_batch,
            logits_a, logits_g_adapted, logits_g_base)
    return results


def phase_verify_grassmannian(A_matrices: dict) -> dict:
    """Verify Grassmannian A-matrix orthogonality (K848 analog).

    This is the baseline structural guarantee — unaffected by scale calibration.
    """
    log("\n=== Phase: Verify Grassmannian A-matrix Orthogonality ===")
    all_A = list(A_matrices.values())
    n = len(all_A)

    max_cos = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # cos(A_i, A_j) over vectorized A matrices
            a = all_A[i].reshape(-1).astype(mx.float32)
            b = all_A[j].reshape(-1).astype(mx.float32)
            dot = mx.sum(a * b).item()
            na = mx.linalg.norm(a).item()
            nb = mx.linalg.norm(b).item()
            cos = abs(dot / (na * nb + 1e-10))
            if cos > max_cos:
                max_cos = cos

    log(f"  Max pairwise |cos| across {n} A-matrices: {max_cos:.8f}")
    log(f"  Orthogonality: {'PERFECT' if max_cos < 1e-5 else 'PARTIAL'}")

    return {"max_pairwise_cos": max_cos, "n_A_matrices": n,
            "orthogonal": max_cos < 1e-5}


# ── Main orchestration ────────────────────────────────────────────────────

def main():
    t0 = time.time()

    log("=" * 65)
    log("M2P Scale Calibrated: L_preserve → scale self-calibration")
    log("=" * 65)
    log(f"SMOKE_TEST: {SMOKE_TEST}")
    log(f"D_MODEL={D_MODEL}, N_LAYERS={N_LAYERS}, LORA_RANK={LORA_RANK}")
    log(f"LAMBDA_PRESERVE={LAMBDA_PRESERVE}, N_CONTEXT_VARIANTS={N_CONTEXT_VARIANTS}")
    log(f"Theorem 1 (MATH.md): L_total = L_task + λ·L_preserve → α* via KKT")
    log("")

    rng = np.random.RandomState(SEED)
    log_memory("start")

    # Phase 1: pre-train base GPT
    base_weights_np, base_results = phase_pretrain_base(rng)
    log_memory("after-phase1")

    # Generate Grassmannian A-matrices (frozen for all phases)
    A_matrices = generate_grassmannian_A()
    grassmannian_results = phase_verify_grassmannian(A_matrices)
    log_memory("after-grassmannian")

    # Phase 2: train SFT reference (directly-trained adapter, scale=5)
    # B_sft is discarded (we only need the metrics, not the weights)
    _B_sft_unused, sft_results = phase_train_sft_reference(base_weights_np, A_matrices, rng)
    del _B_sft_unused
    gc.collect()
    mx.clear_cache()
    log_memory("after-phase2-sft")

    # Phase 3: train M2P WITH L_preserve (Theorem 1 prediction: self-calibrates)
    m2p_preserve_results = phase_train_m2p(
        base_weights_np, A_matrices, rng, use_preserve=True
    )
    log_memory("after-phase3-m2p-preserve")

    # Phase 4: train M2P WITHOUT L_preserve (baseline: unconstrained scale)
    m2p_baseline_results = phase_train_m2p(
        base_weights_np, A_matrices, rng, use_preserve=False
    )
    log_memory("after-phase4-m2p-baseline")

    # ── Kill criteria evaluation ─────────────────────────────────────────
    log("\n" + "=" * 65)
    log("KILL CRITERIA SUMMARY")
    log("=" * 65)

    k849_pass = m2p_preserve_results["k849_pass"]
    k850_pass = m2p_preserve_results["k850_pass"]
    gen_deg = m2p_preserve_results["general_degradation_pp"]
    mag_cv = m2p_preserve_results["adapter_magnitude_cv"]
    scale_learned = m2p_preserve_results["scale_learned"]

    # Comparison: baseline should have WORSE general degradation
    baseline_gen_deg = m2p_baseline_results["general_degradation_pp"]
    baseline_scale = m2p_baseline_results["scale_learned"]

    log(f"\nK849 (general degradation < 10pp):")
    log(f"  WITH L_preserve:    {gen_deg:.2f}pp — {'PASS' if k849_pass else 'FAIL'}")
    log(f"  WITHOUT L_preserve: {baseline_gen_deg:.2f}pp (baseline, for comparison)")
    log(f"  Theorem 1 predicts: L_preserve constrains degradation to < 10pp")

    log(f"\nK850 (magnitude CV > 0.05 = self-calibration):")
    log(f"  WITH L_preserve:    CV={mag_cv:.4f} — {'PASS' if k850_pass else 'FAIL'}")
    log(f"  WITHOUT L_preserve: CV={m2p_baseline_results['adapter_magnitude_cv']:.4f}")
    log(f"  Theorem 1 predicts: context-dependent scale → CV > 0.05")

    log(f"\nScale self-calibration:")
    log(f"  Learned scale WITH L_preserve:    {scale_learned:.4f}")
    log(f"  Learned scale WITHOUT L_preserve: {baseline_scale:.4f}")
    log(f"  Theorem 1 predicts: 3 ≤ scale ≤ 15")
    log(f"  Scale in range [3,15]: {3.0 <= scale_learned <= 15.0}")

    all_pass = k849_pass and k850_pass
    log(f"\nOVERALL: {'SUPPORTED' if all_pass else 'KILLED'}")

    # ── Prediction table (PAPER.md template) ────────────────────────────
    log("\n" + "=" * 65)
    log("PREDICTION TABLE (Theorem 1, MATH.md)")
    log("=" * 65)
    log(f"{'Prediction (from proof)':<45} {'Measured':<15} {'Match?'}")
    log("-" * 75)

    pred1 = f"gen_degradation < 10pp (KKT equilibrium)"
    meas1 = f"{gen_deg:.2f}pp"
    match1 = "YES" if k849_pass else "NO"
    log(f"{pred1:<45} {meas1:<15} {match1}")

    pred2 = f"adapter magnitude CV > 0.05 (self-cal.)"
    meas2 = f"{mag_cv:.4f}"
    match2 = "YES" if k850_pass else "NO"
    log(f"{pred2:<45} {meas2:<15} {match2}")

    pred3 = f"scale_learned in [3, 15]"
    meas3 = f"{scale_learned:.3f}"
    match3 = "YES" if 3.0 <= scale_learned <= 15.0 else "NO"
    log(f"{pred3:<45} {meas3:<15} {match3}")

    pred4 = f"L_preserve reduces gen degradation vs baseline"
    meas4 = f"{gen_deg:.2f} vs {baseline_gen_deg:.2f}pp"
    match4 = "YES" if gen_deg < baseline_gen_deg else "NO"
    log(f"{pred4:<45} {meas4:<15} {match4}")

    pred5 = f"Grassmannian |cos| = 0 (structural guarantee)"
    meas5 = f"{grassmannian_results['max_pairwise_cos']:.8f}"
    match5 = "YES" if grassmannian_results["orthogonal"] else "NO"
    log(f"{pred5:<45} {meas5:<15} {match5}")

    total_time = time.time() - t0

    # ── Write results.json ───────────────────────────────────────────────
    # Secondary predictions
    hard_easy_ratio_preserve = m2p_preserve_results.get("hard_easy_ratio", 1.0)
    hard_gt_easy_preserve = m2p_preserve_results.get("hard_gt_easy", False)

    results = {
        "experiment": "m2p_scale_calibrated",
        "status": "supported" if all_pass else "killed",
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
            "smoke_test": SMOKE_TEST,
        },
        "base_model": base_results,
        "sft_reference": sft_results,
        "grassmannian": grassmannian_results,
        "m2p_with_preserve": m2p_preserve_results,
        "m2p_baseline": m2p_baseline_results,
        # Kill criteria (K849, K850) — primary outputs for experiment complete
        "k849_pass": k849_pass,
        "k850_pass": k850_pass,
        "all_pass": all_pass,
        # Summary fields for easy parsing
        "general_degradation_pp": gen_deg,
        "adapter_magnitude_mean": m2p_preserve_results["adapter_magnitude_mean"],
        "adapter_magnitude_std": m2p_preserve_results["adapter_magnitude_std"],
        "scale_learned": scale_learned,
        "baseline_general_degradation_pp": baseline_gen_deg,
        "baseline_scale_learned": baseline_scale,
        # Theorem 1 prediction table
        "prediction_table": {
            "gen_deg_lt_10pp": {"predicted": 10.0, "measured": gen_deg, "pass": k849_pass},
            "magnitude_cv_gt_005": {"predicted": 0.05, "measured": mag_cv, "pass": k850_pass},
            "scale_in_range_3_15": {
                "predicted": [3.0, 15.0],
                "measured": scale_learned,
                "pass": 3.0 <= scale_learned <= 15.0,
            },
            "preserve_reduces_degradation": {
                "measured_with": gen_deg,
                "measured_without": baseline_gen_deg,
                "pass": gen_deg < baseline_gen_deg,
            },
            "hard_gt_easy": {
                "predicted": True,
                "measured": hard_gt_easy_preserve,
                "hard_easy_ratio": hard_easy_ratio_preserve,
                "pass": hard_gt_easy_preserve,
            },
            # Grassmannian: informational only (single domain, not cross-domain)
            "grassmannian_structural": {
                "note": "Single domain: A matrices need not be orthogonal to each other",
                "max_pairwise_cos_across_slots": grassmannian_results["max_pairwise_cos"],
            },
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults written to: {RESULTS_FILE}")
    log(f"Total runtime: {total_time:.1f}s")

    return all_pass


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
