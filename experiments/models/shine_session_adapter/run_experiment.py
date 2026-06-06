#!/usr/bin/env python3
"""SHINE-style session adapter: context → adapter in one forward pass.

Experiment type: Frontier Extension (Type 3)
Proven result: SHINE M2P architecture (arXiv:2602.06358) + Finding #336 (M2P ports to MLX).
Gap: Does the full SHINE training loop (NTP loss → M2P gradients → domain adapter)
     converge, and does the generated adapter capture >= 50% of SFT quality?

Kill criteria:
  K832: PPL_M2P_generated < PPL_base - 0.5 * (PPL_base - PPL_SFT)
        i.e. generated adapter quality >= 50% of SFT adapter improvement
        FAIL if generated adapter is below 50% of SFT improvement
  K833: Adapter generation (context encode + M2P forward) < 5s
        FAIL if > 5s

Approach: Toy language model (4 layers, d=256, vocab=65 char-level) with
synthetic domain data. Train M2P to generate LoRA adapters from context.
Compare to SFT-trained adapter baseline.
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

# Toy model hyperparams (micro budget: < 30 min)
TOY_VOCAB = 65          # printable ASCII chars
TOY_D = 128             # hidden dim (small, fast)
TOY_LAYERS = 4          # LM layers
TOY_HEADS = 4           # attention heads
TOY_SEQ = 64            # sequence length
LORA_RANK = 4           # adapter rank

# M2P hyperparams (from Finding #336)
M2P_MEM_TOKENS = 8      # M memory tokens sampled from sequence
M2P_LAYERS = 4          # M2P transformer depth
M2P_HEADS = 4           # M2P attention heads

# Training hyperparams
SFT_STEPS = 200         # steps to train SFT adapter baseline
M2P_STEPS = 300         # steps to train M2P (needs a bit more for cold start)
LR_SFT = 3e-4
LR_M2P = 1e-3
BATCH_SIZE = 4

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    SFT_STEPS = 20
    M2P_STEPS = 30
    print("[SMOKE] Reduced steps for quick test")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)

def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)

def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ── Synthetic Domain Data ────────────────────────────────────────────────────

def make_domain_data(domain_id: int, n_sequences: int, seq_len: int, seed: int):
    """Generate synthetic 'domain' data as structured character-level sequences.

    Each domain has distinctive repeating bigrams and vocabulary patterns.
    This gives the base LM meaningful domain-specific hidden states.
    """
    rng = np.random.RandomState(seed + domain_id * 1000)

    # Domain-specific character distributions (skewed toward certain ASCII chars)
    # Domain 0 = "medical": heavy on letters a-m (first half of alphabet)
    # Domain 1 = "code": heavy on letters n-z, digits, brackets
    # Domain 2 = "math": heavy on digits and operators
    char_weights = np.ones(TOY_VOCAB, dtype=np.float32)
    if domain_id == 0:
        char_weights[:20] = 10.0   # a-t range (lowercase letters a-t)
        char_weights[20:45] = 0.5  # reduced weight for other letters
    elif domain_id == 1:
        char_weights[20:52] = 8.0  # u-z, digits, upper range
        char_weights[:20] = 0.5
    elif domain_id == 2:
        char_weights[48:58] = 12.0  # heavy on 0-9 (ASCII 48-57), +/- (ASCII 43,45)
        char_weights[43] = 6.0
        char_weights[45] = 6.0
        char_weights[:43] = 0.3

    char_weights /= char_weights.sum()

    # Enforce domain-specific bigram transitions (Markov-1 structure)
    # This creates predictable structure the adapter can learn
    data = []
    for _ in range(n_sequences):
        seq = np.zeros(seq_len + 1, dtype=np.int32)
        seq[0] = rng.randint(1, TOY_VOCAB)
        for t in range(1, seq_len + 1):
            prev = seq[t - 1]
            # Domain-specific transition: tend to repeat or move to nearby chars
            transition = char_weights.copy()
            transition[prev] *= 3.0  # self-loop bias (predictable repetition)
            if domain_id == 0:
                # Medical: even-to-odd transitions dominant
                neighbor = (prev + 2) % TOY_VOCAB
                transition[neighbor] *= 5.0
            elif domain_id == 1:
                # Code: mod-4 skip transitions
                neighbor = (prev + 4) % TOY_VOCAB
                transition[neighbor] *= 5.0
            elif domain_id == 2:
                # Math: mod-7 transitions (prime-step)
                neighbor = (prev + 7) % TOY_VOCAB
                transition[neighbor] *= 5.0
            transition /= transition.sum()
            seq[t] = rng.choice(TOY_VOCAB, p=transition)
        data.append(seq)

    return np.array(data, dtype=np.int32)  # (n_sequences, seq_len+1)


def compute_ppl(model, data_batches, adapter=None):
    """Compute perplexity on data. Optionally apply a LoRA adapter."""
    total_loss = 0.0
    total_tokens = 0
    for batch in data_batches:
        tokens = mx.array(batch[:, :-1])    # (B, T) inputs
        targets = mx.array(batch[:, 1:])    # (B, T) targets
        if adapter is not None:
            logits = model(tokens, adapter=adapter)
        else:
            logits = model(tokens)
        mx.eval(logits)
        # Cross-entropy per token
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="sum"
        )
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += B * T
        del logits, loss
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ── Toy Language Model (with optional LoRA adapter) ──────────────────────────

class ToyAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, lora_q=None, lora_v=None):
        B, T, C = x.shape
        q = self.wq(x)
        if lora_q is not None:
            q = q + (x @ lora_q[0]) @ lora_q[1]
        v = self.wv(x)
        if lora_v is not None:
            v = v + (x @ lora_v[0]) @ lora_v[1]
        k = self.wk(x)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Causal mask
        mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.where(mask[None, None, :, :], attn, mx.array(-1e9))
        attn = mx.softmax(attn, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))


class ToyMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1 = nn.Linear(d, 4 * d, bias=False)
        self.w2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.w2(nn.gelu(self.w1(x)))


class ToyBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn = ToyAttention(d, n_heads)
        self.mlp = ToyMLP(d)
        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

    def __call__(self, x, lora_q=None, lora_v=None):
        x = x + self.attn(self.norm1(x), lora_q=lora_q, lora_v=lora_v)
        x = x + self.mlp(self.norm2(x))
        return x


class ToyLM(nn.Module):
    """Small toy language model with optional LoRA adapter support.

    LoRA adapters applied to Q and V projections in all layers.
    Adapter format: list of (A_Q, B_Q, A_V, B_V) per layer.
    A: (d, r), B: (r, d) — standard LoRA decomposition.
    """
    def __init__(self, vocab_size, d, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = [ToyBlock(d, n_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.alpha = LORA_RANK  # LoRA scaling factor = rank (so alpha/r = 1.0)

    def __call__(self, x, adapter=None):
        """x: (B, T) token indices. adapter: list of (A_Q, B_Q, A_V, B_V) or None."""
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            if adapter is not None and i < len(adapter):
                A_Q, B_Q, A_V, B_V = adapter[i]
                # Scale: alpha/r = 1.0 (standard LoRA scaling)
                lora_q = (A_Q, B_Q)
                lora_v = (A_V, B_V)
            else:
                lora_q = lora_v = None
            h = block(h, lora_q=lora_q, lora_v=lora_v)
        return self.head(self.norm(h))

    def get_hidden_states(self, x):
        """Extract hidden states from all layers. Returns list[Tensor(B,T,d)]."""
        h = self.embed(x)
        hiddens = []
        for block in self.blocks:
            h = block(h)
            hiddens.append(h)
        return hiddens   # list of L tensors, each (B, T, d)


class LoRAAdapter(nn.Module):
    """Trainable LoRA adapter (for SFT baseline training).

    For each layer: A_Q (d,r), B_Q (r,d), A_V (d,r), B_V (r,d).
    B initialized to zero (standard LoRA init: ΔW = B@A = 0 at start).
    A initialized with Kaiming normal.
    """
    def __init__(self, n_layers, d, rank):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        # Initialize A with Kaiming, B with zeros (standard LoRA)
        scale = math.sqrt(2.0 / d)
        self.A_Q = [mx.random.normal((d, rank)) * scale for _ in range(n_layers)]
        self.B_Q = [mx.zeros((rank, d)) for _ in range(n_layers)]
        self.A_V = [mx.random.normal((d, rank)) * scale for _ in range(n_layers)]
        self.B_V = [mx.zeros((rank, d)) for _ in range(n_layers)]

    def get_adapter_list(self):
        """Return list of (A_Q, B_Q, A_V, B_V) tuples, one per layer."""
        return [(self.A_Q[i], self.B_Q[i], self.A_V[i], self.B_V[i])
                for i in range(self.n_layers)]


# ── M2P Transformer (copied from shine_port, adapted for toy scale) ──────────

class M2PAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))


class M2PBlock(nn.Module):
    """One M2P block: alternating row/column attention (SHINE §3.4)."""
    def __init__(self, dim, n_heads=4, is_column=True):
        super().__init__()
        self.attn = M2PAttention(dim, n_heads)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )
        self.is_column = is_column

    def __call__(self, x):
        """x: (L, M, H)"""
        L, M, H = x.shape
        if self.is_column:
            x_t = x.transpose(1, 0, 2)  # (M, L, H)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.mlp(self.norm2(x_t))
            return x_t.transpose(1, 0, 2)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class M2PTransformer(nn.Module):
    """Memory-to-Parameter Transformer (SHINE §3.4).

    Input: (L, M, H) memory grid
    Output: (L, M, H) contextualized grid → extract adapter weights from this

    The M2P output has shape (L, M, H). For each layer i, we have M*H values.
    We carve out A_Q (d*r), B_Q (r*d), A_V (d*r), B_V (r*d) per layer.
    This requires M*H >= 4*d*r params per layer.

    At M=8, H=128 (same as toy LM d), r=4:
      M*H = 1024 >= 4*128*4 = 2048? No — need M=16 OR a projection head.

    Solution: use a linear projection head to map M2P output to adapter params.
    Per-layer: output (M, H) → flatten (M*H,) → project to (4*d*r,) → reshape
    """
    def __init__(self, n_lm_layers, n_mem_tokens, m2p_dim, lm_d, lora_rank,
                 n_layers_m2p=4, n_heads=4):
        super().__init__()
        self.n_lm_layers = n_lm_layers
        self.n_mem_tokens = n_mem_tokens
        self.m2p_dim = m2p_dim
        self.lm_d = lm_d
        self.lora_rank = lora_rank
        self.adapter_size_per_layer = 4 * lm_d * lora_rank  # A_Q, B_Q, A_V, B_V

        # Positional embeddings (SHINE §3.4 Eq. 5, Xavier normal init)
        scale = math.sqrt(2.0 / (1 + m2p_dim))
        self.p_layer = mx.random.normal((n_lm_layers, 1, m2p_dim)) * scale
        self.p_token = mx.random.normal((1, n_mem_tokens, m2p_dim)) * scale

        # Transformer blocks
        self.blocks = []
        for i in range(n_layers_m2p):
            self.blocks.append(M2PBlock(m2p_dim, n_heads, is_column=(i % 2 == 0)))
        self.final_norm = nn.RMSNorm(m2p_dim)

        # Per-layer projection head: maps M2P output per layer to adapter params
        in_dim = n_mem_tokens * m2p_dim
        out_dim = self.adapter_size_per_layer
        self.proj_heads = [nn.Linear(in_dim, out_dim, bias=False)
                           for _ in range(n_lm_layers)]

    def __call__(self, memory_states):
        """memory_states: (L, M, H) — one entry per LM layer, M memory tokens.

        Returns: list of (A_Q, B_Q, A_V, B_V) tuples, one per LM layer.
        Each matrix: A_Q=(lm_d, r), B_Q=(r, lm_d), A_V=(lm_d, r), B_V=(r, lm_d).
        """
        L, M, H = memory_states.shape

        # Apply positional embeddings
        x = memory_states + self.p_layer + self.p_token

        # M2P Transformer
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)  # (L, M, H)

        # Project each layer's memory to adapter params
        adapter = []
        d, r = self.lm_d, self.lora_rank
        scale = 1.0 / math.sqrt(r)  # scale B similar to Kaiming init norm

        for i in range(L):
            flat = x[i].reshape(-1)  # (M*H,)
            params = self.proj_heads[i](flat[None, :]).squeeze(0)  # (4*d*r,)

            # Split into four adapter matrices
            idx = 0
            A_Q = params[idx:idx + d * r].reshape(d, r);            idx += d * r
            B_Q = params[idx:idx + r * d].reshape(r, d) * scale;    idx += r * d
            A_V = params[idx:idx + d * r].reshape(d, r);            idx += d * r
            B_V = params[idx:idx + r * d].reshape(r, d) * scale;    # last chunk

            adapter.append((A_Q, B_Q, A_V, B_V))

        return adapter


# ── Phase Functions ───────────────────────────────────────────────────────────

def phase_build_toy_model():
    """Phase 1: Build and initialize the toy language model."""
    log("\n=== Phase 1: Build Toy LM ===")
    mx.random.seed(SEED)

    model = ToyLM(TOY_VOCAB, TOY_D, TOY_LAYERS, TOY_HEADS)
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    log(f"  Toy LM: {TOY_LAYERS}L x {TOY_D}d x {TOY_HEADS}H x vocab={TOY_VOCAB}")
    log(f"  Parameters: {n_params:,}")

    return model, n_params


def phase_generate_data():
    """Phase 2: Generate synthetic domain data."""
    log("\n=== Phase 2: Generate Domain Data ===")
    np.random.seed(SEED)

    n_train = 200 if not SMOKE_TEST else 20
    n_val = 50 if not SMOKE_TEST else 10

    domains = {}
    for domain_id in range(3):
        name = ["medical", "code", "math"][domain_id]
        train_data = make_domain_data(domain_id, n_train, TOY_SEQ, seed=SEED)
        val_data = make_domain_data(domain_id, n_val, TOY_SEQ, seed=SEED + 99)
        domains[name] = {"train": train_data, "val": val_data, "id": domain_id}
        log(f"  Domain '{name}': train={train_data.shape}, val={val_data.shape}")

    return domains


def make_batches(data, batch_size):
    """Split data into batches."""
    n = len(data)
    batches = []
    for start in range(0, n, batch_size):
        batch = data[start:start + batch_size]
        batches.append(batch)
    return batches


def phase_train_sft_adapter(model, domain_data, domain_name):
    """Phase 3: Train a SFT LoRA adapter on the target domain (baseline).

    This gives us the reference PPL improvement. The SFT adapter is the ceiling
    for the K832 quality criterion.

    Returns: (adapter_params, base_ppl, sft_ppl)
    """
    log(f"\n=== Phase 3: Train SFT Adapter for domain='{domain_name}' ===")

    # First measure base PPL (no adapter)
    val_batches = make_batches(domain_data["val"], BATCH_SIZE)
    model.freeze()  # freeze everything
    base_ppl = compute_ppl(model, val_batches, adapter=None)
    log(f"  Base PPL (no adapter): {base_ppl:.4f}")

    # Train LoRA adapter
    adapter = LoRAAdapter(TOY_LAYERS, TOY_D, LORA_RANK)
    mx.eval(adapter.parameters())
    model.freeze()  # keep base model frozen

    optimizer = opt.Adam(learning_rate=LR_SFT)
    train_data = domain_data["train"]

    def sft_loss(adapter, model, tokens, targets):
        adapter_list = adapter.get_adapter_list()
        logits = model(tokens, adapter=adapter_list)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(adapter, sft_loss)

    losses = []
    n_train = len(train_data)
    gc.disable()
    for step in range(SFT_STEPS):
        # Sample random batch
        idx = np.random.randint(0, n_train, size=BATCH_SIZE)
        batch = train_data[idx]
        tokens = mx.array(batch[:, :-1])
        targets = mx.array(batch[:, 1:])

        loss, grads = loss_and_grad(adapter, model, tokens, targets)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            log(f"  Step {step+1}/{SFT_STEPS}: loss={loss.item():.4f}")

        del loss, grads, tokens, targets, batch
    gc.enable()

    # Measure SFT PPL
    adapter_list = adapter.get_adapter_list()
    mx.eval(*[p for tpl in adapter_list for p in tpl])
    sft_ppl = compute_ppl(model, val_batches, adapter=adapter_list)
    log(f"  SFT PPL (with adapter): {sft_ppl:.4f}")
    log(f"  PPL improvement: {(base_ppl - sft_ppl):.4f} ({(base_ppl - sft_ppl)/base_ppl*100:.1f}%)")

    converged = losses[-1] < losses[0] * 0.5
    log(f"  Convergence: {'YES' if converged else 'WEAK'} (final/initial={losses[-1]/losses[0]:.3f})")

    # Save adapter params as numpy arrays for later comparison
    adapter_params_np = []
    for i in range(TOY_LAYERS):
        A_Q = np.array(adapter.A_Q[i].tolist())
        B_Q = np.array(adapter.B_Q[i].tolist())
        A_V = np.array(adapter.A_V[i].tolist())
        B_V = np.array(adapter.B_V[i].tolist())
        adapter_params_np.append((A_Q, B_Q, A_V, B_V))

    cleanup(adapter, optimizer)
    return adapter_params_np, base_ppl, sft_ppl, losses


def phase_train_m2p(model, domain_data, domain_name, m2p_dim):
    """Phase 4: Train M2P to generate domain adapters from context.

    SHINE training loop:
    1. Sample context batch from domain
    2. Extract hidden states from frozen base LM (no gradient)
    3. Sample M memory tokens from sequence
    4. M2P(memory) → adapter weights
    5. Apply adapter to base LM
    6. Compute NTP loss on task batch
    7. Backprop through M2P only

    Returns: (m2p, m2p_losses, generation_time_s)
    """
    log(f"\n=== Phase 4: Train M2P for domain='{domain_name}' ===")
    log(f"  M2P config: L={TOY_LAYERS}, M={M2P_MEM_TOKENS}, H={m2p_dim}")

    m2p = M2PTransformer(
        n_lm_layers=TOY_LAYERS,
        n_mem_tokens=M2P_MEM_TOKENS,
        m2p_dim=m2p_dim,
        lm_d=TOY_D,
        lora_rank=LORA_RANK,
        n_layers_m2p=M2P_LAYERS,
        n_heads=M2P_HEADS,
    )
    mx.eval(m2p.parameters())

    m2p_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P parameters: {m2p_params:,}")

    model.freeze()
    optimizer = opt.Adam(learning_rate=LR_M2P)
    train_data = domain_data["train"]
    n_train = len(train_data)

    def extract_memory(model, context_tokens):
        """Extract hidden states from all LM layers, sample M memory tokens.

        Returns: (L, M, H) memory grid as MLX array.
        context_tokens: (1, T) integer tokens.
        """
        # Get hidden states from all layers (no gradient through base LM)
        hiddens = model.get_hidden_states(context_tokens)  # list of L tensors (1, T, d)

        # Sample M_MEM_TOKENS evenly spaced from the sequence
        T = context_tokens.shape[1]
        # Use middle portion of the sequence for memory tokens (avoid start/end padding)
        step = max(1, T // M2P_MEM_TOKENS)
        indices = [min(step // 2 + i * step, T - 1) for i in range(M2P_MEM_TOKENS)]

        # Stack: for each layer, take M specific token positions
        memory_layers = []
        for h in hiddens:
            h_eval = h  # (1, T, d)
            mem_tokens = mx.stack([h_eval[0, idx, :] for idx in indices], axis=0)  # (M, d)
            memory_layers.append(mem_tokens)

        # Stack across layers: (L, M, d)
        memory_grid = mx.stack(memory_layers, axis=0)  # (L, M, d)
        mx.eval(memory_grid)
        return memory_grid

    def m2p_loss_fn(m2p, model, context_tokens, task_tokens, task_targets):
        """Full SHINE training loss.

        1. Extract memory from context (frozen base LM)
        2. M2P generates adapter
        3. Apply adapter to base LM for task tokens
        4. NTP loss on task targets
        """
        # Stop gradient: hidden state extraction does not backprop through base LM
        memory_grid = mx.stop_gradient(
            mx.stack(
                [mx.stack([h[0, min((TOY_SEQ // M2P_MEM_TOKENS // 2) + i * (TOY_SEQ // M2P_MEM_TOKENS), TOY_SEQ - 1), :]
                          for i in range(M2P_MEM_TOKENS)], axis=0)
                 for h in model.get_hidden_states(context_tokens)],
                axis=0
            )
        )  # (L, M, d)

        # M2P generates adapter from memory (gradient flows HERE)
        adapter = m2p(memory_grid)

        # Apply adapter to base LM for task prediction
        logits = model(task_tokens, adapter=adapter)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            task_targets.reshape(B * T),
            reduction="mean"
        )
        return loss

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    losses = []
    gc.disable()
    for step in range(M2P_STEPS):
        # Sample context and task batches (same domain, different sequences)
        ctx_idx = np.random.randint(0, n_train)
        task_idx_arr = np.random.randint(0, n_train, size=BATCH_SIZE)

        ctx_tokens = mx.array(train_data[ctx_idx:ctx_idx+1, :-1])   # (1, T)
        task_batch = train_data[task_idx_arr]
        task_tokens = mx.array(task_batch[:, :-1])                   # (B, T)
        task_targets = mx.array(task_batch[:, 1:])                   # (B, T)

        loss, grads = loss_and_grad(m2p, model, ctx_tokens, task_tokens, task_targets)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            log(f"  Step {step+1}/{M2P_STEPS}: loss={loss.item():.4f}")

        del loss, grads, ctx_tokens, task_tokens, task_targets, task_batch
    gc.enable()

    converged = losses[-1] < losses[0] * 0.5
    log(f"  M2P convergence: {'YES' if converged else 'WEAK'} (final/initial={losses[-1]/losses[0]:.3f})")

    cleanup(optimizer)
    return m2p, losses, converged


def phase_evaluate_m2p(model, m2p, domain_data, domain_name, base_ppl, sft_ppl):
    """Phase 5: Evaluate M2P-generated adapter quality and latency.

    Measures:
    - K832: PPL_M2P_generated vs base PPL and SFT PPL
    - K833: Time from context input to adapter ready (M2P forward only)
    """
    log(f"\n=== Phase 5: Evaluate M2P Adapter for domain='{domain_name}' ===")

    val_data = domain_data["val"]
    val_batches = make_batches(val_data, BATCH_SIZE)

    # Use a representative context from training data for the session
    train_data = domain_data["train"]
    ctx_tokens = mx.array(train_data[0:1, :-1])  # (1, T) context

    # Measure K833: generation latency (M2P forward only)
    # Warm up
    T = ctx_tokens.shape[1]
    hiddens = model.get_hidden_states(ctx_tokens)
    step = max(1, T // M2P_MEM_TOKENS)
    indices = [min(step // 2 + i * step, T - 1) for i in range(M2P_MEM_TOKENS)]
    memory_grid = mx.stack(
        [mx.stack([h[0, idx, :] for idx in indices], axis=0) for h in hiddens],
        axis=0
    )  # (L, M, d)
    mx.eval(memory_grid)

    # Time the actual generation
    n_timing_runs = 10
    t0 = time.time()
    for _ in range(n_timing_runs):
        adapter = m2p(memory_grid)
        # Force evaluation of all adapter tensors
        mx.eval(*[p for tpl in adapter for p in tpl])
    gen_time_s = (time.time() - t0) / n_timing_runs
    log(f"  K833: M2P adapter generation time: {gen_time_s*1000:.2f}ms ({gen_time_s:.4f}s)")

    # Time full session: context encoding + M2P forward
    t0 = time.time()
    for _ in range(n_timing_runs):
        hiddens_t = model.get_hidden_states(ctx_tokens)
        memory_t = mx.stack(
            [mx.stack([h[0, idx, :] for idx in indices], axis=0) for h in hiddens_t],
            axis=0
        )
        adapter_t = m2p(memory_t)
        mx.eval(*[p for tpl in adapter_t for p in tpl])
    full_session_time_s = (time.time() - t0) / n_timing_runs
    log(f"  K833: Full session time (encode+generate): {full_session_time_s*1000:.2f}ms ({full_session_time_s:.4f}s)")

    k833_pass = full_session_time_s < 5.0
    log(f"  K833: {'PASS' if k833_pass else 'FAIL'} (threshold: 5.0s)")

    # Measure K832: PPL with M2P-generated adapter
    # Generate adapter once from a fresh context
    adapter_eval = m2p(memory_grid)
    mx.eval(*[p for tpl in adapter_eval for p in tpl])

    m2p_ppl = compute_ppl(model, val_batches, adapter=adapter_eval)
    log(f"  Base PPL:     {base_ppl:.4f}")
    log(f"  SFT PPL:      {sft_ppl:.4f}")
    log(f"  M2P PPL:      {m2p_ppl:.4f}")

    delta_sft = base_ppl - sft_ppl
    k832_threshold = base_ppl - 0.5 * delta_sft
    k832_pass = m2p_ppl < k832_threshold

    if delta_sft > 0:
        m2p_fraction = (base_ppl - m2p_ppl) / delta_sft
    else:
        m2p_fraction = 0.0

    log(f"  delta_SFT: {delta_sft:.4f}")
    log(f"  K832 threshold (50% of SFT): {k832_threshold:.4f}")
    log(f"  M2P fraction of SFT improvement: {m2p_fraction*100:.1f}%")
    log(f"  K832: {'PASS' if k832_pass else 'FAIL'}")

    del adapter_eval, adapter_t, memory_t, hiddens_t

    return {
        "base_ppl": base_ppl,
        "sft_ppl": sft_ppl,
        "m2p_ppl": m2p_ppl,
        "delta_sft": delta_sft,
        "k832_threshold": k832_threshold,
        "m2p_fraction_of_sft": m2p_fraction,
        "k832_pass": k832_pass,
        "gen_time_ms": gen_time_s * 1000,
        "full_session_time_ms": full_session_time_s * 1000,
        "k833_pass": k833_pass,
    }


def phase_verify_hidden_state_separation(model, domains):
    """Phase 6: Verify that LM hidden states are domain-discriminative.

    Assumption E.1: the toy model produces informative hidden states.
    If hidden states are the same across domains, M2P cannot learn.

    Measure: inter-domain vs intra-domain cosine similarity of hidden states.
    """
    log("\n=== Phase 6: Verify Hidden State Domain Separation ===")

    domain_names = list(domains.keys())
    n_check = 5 if not SMOKE_TEST else 2

    # Get mean hidden state vector per domain
    domain_means = {}
    for name in domain_names:
        data = domains[name]["train"][:n_check]
        tokens = mx.array(data[:, :-1])
        hiddens = model.get_hidden_states(tokens)
        # Use last layer hidden state, average over tokens and batch
        last_h = hiddens[-1]  # (B, T, d)
        mean_h = mx.mean(last_h, axis=(0, 1))  # (d,)
        mx.eval(mean_h)
        domain_means[name] = np.array(mean_h.tolist())
        del hiddens, last_h, mean_h, tokens

    # Compute pairwise cosine similarities
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    inter_domain_cos = []
    for i, n1 in enumerate(domain_names):
        for n2 in domain_names[i+1:]:
            c = cosine(domain_means[n1], domain_means[n2])
            inter_domain_cos.append(c)
            log(f"  Inter-domain cos({n1}, {n2}): {c:.4f}")

    mean_inter = float(np.mean(inter_domain_cos)) if inter_domain_cos else 1.0
    log(f"  Mean inter-domain cosine: {mean_inter:.4f}")
    # We expect domain-separated hidden states (cosine << 1)
    assumption_ok = mean_inter < 0.99
    log(f"  Assumption E.1 (informative hidden states): {'OK' if assumption_ok else 'WEAK'}")

    return {
        "mean_inter_domain_cosine": mean_inter,
        "assumption_e1_ok": assumption_ok,
        "pairs": {f"{n1}_{n2}": cosine(domain_means[n1], domain_means[n2])
                  for i, n1 in enumerate(domain_names) for n2 in domain_names[i+1:]},
    }


def main():
    t0 = time.time()
    log("SHINE-Style Session Adapter Experiment")
    log("=" * 60)
    log("Type: Frontier Extension (Type 3)")
    log("Gap: Does SHINE training loop converge on toy LM?")
    log("     Does M2P-generated adapter capture >= 50% of SFT quality?")
    log(f"Toy LM: {TOY_LAYERS}L x {TOY_D}d x {TOY_VOCAB}vocab")
    log(f"LoRA rank: {LORA_RANK}")
    log(f"M2P: {M2P_LAYERS}L x {M2P_MEM_TOKENS}M")
    log(f"Training: SFT={SFT_STEPS}steps, M2P={M2P_STEPS}steps")
    mx.random.seed(SEED)

    # M2P hidden dim matches toy LM hidden dim
    M2P_DIM = TOY_D  # H = d

    # --- Phase 1: Build toy LM ---
    model, n_lm_params = phase_build_toy_model()
    log_memory("after-build")

    # --- Phase 2: Generate domain data ---
    domains = phase_generate_data()

    # --- Phase 6: Verify hidden state separation (before training) ---
    sep_results = phase_verify_hidden_state_separation(model, domains)
    log_memory("after-separation")

    # --- Main loop: one domain (medical) for micro budget ---
    # Focus on medical domain — same domain as Finding #333
    target_domain = "medical"
    domain_data = domains[target_domain]

    # --- Phase 3: Train SFT adapter (baseline) ---
    sft_params, base_ppl, sft_ppl, sft_losses = phase_train_sft_adapter(
        model, domain_data, target_domain
    )
    log_memory("after-sft")

    # Verify SFT adapter actually improves things (Assumption E.2)
    delta_sft = base_ppl - sft_ppl
    assumption_e2_ok = sft_ppl < 0.95 * base_ppl
    log(f"  Assumption E.2 (SFT shows measurable improvement): {'OK' if assumption_e2_ok else 'WEAK'}")
    log(f"  Base={base_ppl:.4f}, SFT={sft_ppl:.4f}, delta={delta_sft:.4f}")

    # --- Phase 4: Train M2P ---
    m2p, m2p_losses, m2p_converged = phase_train_m2p(
        model, domain_data, target_domain, M2P_DIM
    )
    log_memory("after-m2p-train")

    m2p_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P parameters: {m2p_params:,}")

    # --- Phase 5: Evaluate M2P ---
    eval_results = phase_evaluate_m2p(model, m2p, domain_data, target_domain, base_ppl, sft_ppl)
    log_memory("after-eval")

    # --- Final results ---
    total_time = time.time() - t0

    # Derive prediction D.1 check
    d1_pass = m2p_losses[-1] / m2p_losses[0] < 0.5 if len(m2p_losses) > 1 else False

    results = {
        "experiment": "shine_session_adapter",
        "type": "frontier_extension",
        "status": "supported" if (eval_results["k832_pass"] and eval_results["k833_pass"]) else "killed",
        "total_time_s": round(total_time, 1),
        "toy_lm": {
            "n_layers": TOY_LAYERS,
            "hidden_dim": TOY_D,
            "vocab_size": TOY_VOCAB,
            "n_heads": TOY_HEADS,
            "n_params": n_lm_params,
        },
        "m2p": {
            "n_params": m2p_params,
            "mem_tokens": M2P_MEM_TOKENS,
            "m2p_layers": M2P_LAYERS,
            "m2p_dim": M2P_DIM,
        },
        "hidden_state_separation": sep_results,
        "assumptions": {
            "e1_informative_hidden_states": sep_results["assumption_e1_ok"],
            "e2_sft_shows_improvement": assumption_e2_ok,
        },
        "sft_training": {
            "n_steps": SFT_STEPS,
            "initial_loss": round(sft_losses[0], 4),
            "final_loss": round(sft_losses[-1], 4),
            "loss_ratio": round(sft_losses[-1] / sft_losses[0], 4),
            "base_ppl": round(base_ppl, 4),
            "sft_ppl": round(sft_ppl, 4),
            "delta_ppl": round(delta_sft, 4),
            "improvement_pct": round(delta_sft / base_ppl * 100, 2),
        },
        "m2p_training": {
            "n_steps": M2P_STEPS,
            "initial_loss": round(m2p_losses[0], 4),
            "final_loss": round(m2p_losses[-1], 4),
            "loss_ratio": round(m2p_losses[-1] / m2p_losses[0], 4),
            "converged": m2p_converged,
        },
        "evaluation": {
            "base_ppl": round(eval_results["base_ppl"], 4),
            "sft_ppl": round(eval_results["sft_ppl"], 4),
            "m2p_ppl": round(eval_results["m2p_ppl"], 4),
            "delta_sft": round(eval_results["delta_sft"], 4),
            "k832_threshold": round(eval_results["k832_threshold"], 4),
            "m2p_fraction_of_sft_pct": round(eval_results["m2p_fraction_of_sft"] * 100, 1),
            "gen_time_ms": round(eval_results["gen_time_ms"], 2),
            "full_session_time_ms": round(eval_results["full_session_time_ms"], 2),
        },
        "predictions": {
            "D1_m2p_convergence_pass": d1_pass,
            "D1_note": f"final/initial={m2p_losses[-1]/m2p_losses[0]:.3f} (threshold <0.5)",
            "D3_generation_under_100ms": eval_results["gen_time_ms"] < 100,
        },
        "kill_criteria": {
            "K832": {
                "pass": eval_results["k832_pass"],
                "criterion": "M2P PPL < base_ppl - 0.5*(base_ppl - sft_ppl)",
                "threshold": round(eval_results["k832_threshold"], 4),
                "measured": round(eval_results["m2p_ppl"], 4),
                "m2p_fraction_of_sft_pct": round(eval_results["m2p_fraction_of_sft"] * 100, 1),
                "detail": (
                    f"threshold={eval_results['k832_threshold']:.4f}, "
                    f"m2p_ppl={eval_results['m2p_ppl']:.4f}, "
                    f"fraction={eval_results['m2p_fraction_of_sft']*100:.1f}% of SFT improvement"
                ),
            },
            "K833": {
                "pass": eval_results["k833_pass"],
                "criterion": "Full session adapter generation < 5.0s",
                "measured_s": round(eval_results["full_session_time_ms"] / 1000, 4),
                "measured_ms": round(eval_results["full_session_time_ms"], 2),
                "detail": (
                    f"Full session={eval_results['full_session_time_ms']:.2f}ms < 5000ms"
                ),
            },
        },
        "all_pass": eval_results["k832_pass"] and eval_results["k833_pass"],
    }

    log(f"\n{'='*60}")
    log(f"SUMMARY")
    log(f"{'='*60}")
    log(f"Toy LM: {n_lm_params:,} params, {TOY_LAYERS}L x {TOY_D}d")
    log(f"M2P: {m2p_params:,} params, {M2P_LAYERS}L x {M2P_MEM_TOKENS}M x {M2P_DIM}H")
    log(f"")
    log(f"SFT Training ({SFT_STEPS} steps):")
    log(f"  Base PPL: {base_ppl:.4f}")
    log(f"  SFT PPL:  {sft_ppl:.4f}")
    log(f"  Improvement: {delta_sft/base_ppl*100:.1f}%")
    log(f"")
    log(f"M2P Training ({M2P_STEPS} steps):")
    log(f"  Loss ratio: {m2p_losses[-1]/m2p_losses[0]:.3f}")
    log(f"  Converged: {m2p_converged}")
    log(f"")
    log(f"Kill Criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
    log(f"")
    log(f"Predictions:")
    log(f"  D.1 (M2P convergence): {'PASS' if d1_pass else 'FAIL'}")
    log(f"  D.3 (generation < 100ms): {'PASS' if eval_results['gen_time_ms'] < 100 else 'FAIL'} ({eval_results['gen_time_ms']:.1f}ms)")
    log(f"")
    log(f"Status: {results['status'].upper()}")
    log(f"Total time: {total_time:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to: {RESULTS_FILE}")

    cleanup(model, m2p)
    return results


if __name__ == "__main__":
    main()
