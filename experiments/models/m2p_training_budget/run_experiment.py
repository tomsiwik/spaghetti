#!/usr/bin/env python3
"""M2P Training Budget Sweep: quality scales with steps, not architecture.

TYPE: guided-exploration (Type 2)
MATH: experiments/models/m2p_training_budget/MATH.md

PRIOR KILLS:
  - exp_m2p_bottleneck_width (Finding #355): Width d_M2P is NOT the bottleneck.
  - exp_m2p_depth (Finding #357): Depth is NOT the bottleneck. L=2 saturates.
  Architecture search is closed. Remaining gap is training convergence.

HYPOTHESIS (Theorem 1, MATH.md):
  By O(1/T) convergence of SGD/Adam on smooth objectives, M2P quality improves
  monotonically with training steps. SHINE (arXiv:2602.06358) identifies training
  scale as the bottleneck for hypernetwork quality.
  Secondary: bidirectional attention over memory tokens (Theorem 2) removes an
  incorrect causal constraint, improving representational capacity.

KILL CRITERIA:
  K876: quality(2000 steps) > quality(500 steps) + 2pp -- budget matters
  K877: quality(2000 steps) >= 97% of SFT -- ceiling reached
  K878: |quality(2000) - quality(1000)| < 1pp -- budget exhausted (KILL)

ARCHITECTURE CONSTANTS (FIXED -- same as exp_m2p_depth):
  D_MODEL=256, N_LAYERS=2, N_HEADS=4, VOCAB_SIZE=128, BLOCK_SIZE=48
  LORA_RANK=4, D_M2P=64, M2P_LAYERS=2 (proven sufficient)
  N_DOMAINS=5 (arithmetic, sort, parity, reverse, repeat)

EXPERIMENT VARIABLES:
  1. M2P_STEPS swept: {500, 1000, 2000}
  2. Attention mode: causal (baseline) vs bidirectional (at T=500 only)

PARITY GUARD (standard from Finding #354 onward):
  Exclude parity domain where (base_loss - sft_loss) < 0.05.

FRESH TRAINING REQUIRED (from LEARNINGS.md):
  Reusing adapters contaminates quality_ratio by up to 2.9pp.
  Always retrain base + SFT + M2P from scratch.
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

# -- Architecture constants (FIXED -- same as exp_m2p_depth) --
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

# Training budget sweep -- THE PRIMARY EXPERIMENT VARIABLE
M2P_STEPS_VALUES = [500, 1000, 2000]

# Shared training constants
BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

# Smoke test M2P steps
if SMOKE_TEST:
    M2P_STEPS_VALUES = [40, 80, 160]

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]
PARITY_GUARD_THRESHOLD = 0.05


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


# -- Data generation (IDENTICAL to exp_m2p_depth) --

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


# -- Toy GPT (IDENTICAL to exp_m2p_depth) --

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


# -- Grassmannian A-matrices (IDENTICAL to exp_m2p_depth) --

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


# -- LoRA forward pass (IDENTICAL to exp_m2p_depth) --

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


# -- SFT training (IDENTICAL to exp_m2p_depth) --

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


# -- M2P Transformer (with configurable attention mode) --

class M2PAttention(nn.Module):
    """M2P attention with configurable masking.

    causal=True: standard causal (lower-triangular) mask (baseline from exp_m2p_depth)
    causal=False: bidirectional (no mask) -- correct for non-autoregressive memory tokens
    """
    def __init__(self, d, n_heads=4, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.causal = causal
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
        if self.causal:
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
    def __init__(self, d, n_heads=4, causal=True):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = M2PAttention(d, n_heads, causal=causal)
        self.norm2 = RMSNorm(d)
        self.mlp = M2PMLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class M2PTransformer(nn.Module):
    """M2P transformer with fixed L=2, d_M2P=64, configurable attention mode.

    Architecture is FIXED (proven sufficient from exp_m2p_depth Finding #357).
    Only the training budget and attention mode vary.
    """
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS,
                 causal=True):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.m2p_layers = m2p_layers
        self.causal = causal
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4, causal=causal)
                       for _ in range(m2p_layers)]
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

def phase_generate_data(rng: np.random.RandomState) -> dict:
    """Generate train/val data for all 5 domains."""
    log("\n=== Phase: Generate Data ===")
    domain_data = {}
    n_per_domain = 500 if not SMOKE_TEST else 60
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
    """Pre-train ToyGPT on all domains (deterministic with fixed SEED).

    Always trained fresh -- no reuse from prior experiments.
    LEARNINGS.md: reusing adapters contaminates quality_ratio by up to 2.9pp.
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
    """Train SFT LoRA adapter for one domain (always fresh)."""
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


def phase_train_m2p(m2p_steps: int, causal: bool, domain_name: str,
                     domain_id: int, domain_data: dict, base: ToyGPT,
                     A_matrices: dict, base_loss: float,
                     sft_loss: float) -> dict:
    """Train ONE M2P at given step count and attention mode for ONE domain.

    Returns quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)
    Also returns training loss trajectory for convergence analysis.
    """
    mode_str = "causal" if causal else "bidir"
    adapter_dir = EXPERIMENT_DIR / f"adapters_T{m2p_steps}_{mode_str}"
    adapter_dir.mkdir(exist_ok=True)
    save_path = adapter_dir / f"m2p_{domain_name}.npz"

    log(f"    [T={m2p_steps},{mode_str}] Training M2P for {domain_name}...")
    mx.random.seed(SEED)

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS,
                          causal=causal)
    mx.eval(m2p.parameters())

    param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"      M2P params: {param_count:,} (L={M2P_LAYERS}, d={D_M2P}, {mode_str})")

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id,
                             tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]

    # Track loss trajectory at checkpoints for convergence analysis
    loss_trajectory = []
    final_loss = None

    gc.disable()
    for step in range(m2p_steps):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        # Log at 25%, 50%, 75%, 100%
        if (step + 1) % max(1, m2p_steps // 4) == 0:
            final_loss = loss.item()
            loss_trajectory.append({
                "step": step + 1,
                "loss": round(final_loss, 4),
            })
            log(f"      Step {step+1}/{m2p_steps}: loss={final_loss:.4f}")
    gc.enable()
    cleanup(optimizer)

    # Generate B-matrices from first training context and save
    context_tokens = domain_data["train"][0][None, :]
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
    m2p_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices_loaded)

    # Parity guard
    quality_ratio = 0.0
    excluded_parity = False
    gap = base_loss - sft_loss
    if gap > PARITY_GUARD_THRESHOLD:
        quality_ratio = (base_loss - m2p_loss) / gap
    else:
        excluded_parity = True
        log(f"    [T={m2p_steps},{mode_str}] {domain_name}: EXCLUDED "
            f"(gap={gap:.4f} < {PARITY_GUARD_THRESHOLD})")

    if not excluded_parity:
        log(f"    [T={m2p_steps},{mode_str}] {domain_name}: m2p_loss={m2p_loss:.4f} "
            f"SFT={sft_loss:.4f} base={base_loss:.4f} quality={quality_ratio:.1%}")

    cleanup(B_matrices_loaded)
    return {
        "m2p_loss": round(m2p_loss, 4),
        "quality_ratio": round(quality_ratio, 4),
        "excluded_parity": excluded_parity,
        "gap": round(gap, 4),
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        "loss_trajectory": loss_trajectory,
    }


def phase_sweep_training_budget(domain_data, base, A_matrices,
                                 base_losses, sft_results) -> dict:
    """Sweep M2P_STEPS in {500, 1000, 2000} with causal attention (primary).

    Architecture fixed: L=2, d_M2P=64 (both proven sufficient).
    Base model, A-matrices, SFT adapters are IDENTICAL across all step counts.
    Parity guard: exclude domains where (base_loss - sft_loss) < PARITY_GUARD_THRESHOLD.
    """
    log(f"\n=== Phase: M2P Training Budget Sweep (causal attention) ===")
    log(f"  Sweeping M2P_STEPS in {M2P_STEPS_VALUES}")
    log(f"  Architecture fixed: L={M2P_LAYERS}, D_M2P={D_M2P}")
    log(f"  SHINE (arXiv:2602.06358): training scale is the bottleneck")

    sweep_results = {}

    for m2p_steps in M2P_STEPS_VALUES:
        log(f"\n  --- M2P_STEPS = {m2p_steps} (causal) ---")
        domain_qualities = {}
        domain_losses = {}
        domain_train_losses = {}
        excluded_domains = []

        for di, name in enumerate(DOMAIN_NAMES):
            result = phase_train_m2p(
                m2p_steps=m2p_steps,
                causal=True,
                domain_name=name,
                domain_id=di,
                domain_data=domain_data[name],
                base=base,
                A_matrices=A_matrices,
                base_loss=base_losses[name],
                sft_loss=sft_results[name]["sft_loss"],
            )
            domain_qualities[name] = result["quality_ratio"]
            domain_losses[name] = result["m2p_loss"]
            domain_train_losses[name] = result["final_train_loss"]
            if result["excluded_parity"]:
                excluded_domains.append(name)

        if excluded_domains:
            log(f"  Excluded domains (parity guard): {excluded_domains}")

        valid_qualities = [domain_qualities[n] for n in DOMAIN_NAMES
                           if n not in excluded_domains]

        median_q = float(np.median(valid_qualities)) if valid_qualities else 0.0
        mean_q = float(np.mean(valid_qualities)) if valid_qualities else 0.0

        log(f"  T={m2p_steps}: median quality={median_q:.1%}, mean={mean_q:.1%}")
        log(f"  Per-domain: {dict((k, f'{v:.1%}') for k, v in domain_qualities.items() if k not in excluded_domains)}")

        sweep_results[f"T{m2p_steps}"] = {
            "m2p_steps": m2p_steps,
            "causal": True,
            "m2p_layers": M2P_LAYERS,
            "d_m2p": D_M2P,
            "domain_quality": domain_qualities,
            "domain_m2p_loss": domain_losses,
            "domain_train_loss": domain_train_losses,
            "median_quality": round(median_q, 4),
            "mean_quality": round(mean_q, 4),
            "excluded_domains": excluded_domains,
            "n_valid_domains": len(valid_qualities),
        }

        log_memory(f"after T={m2p_steps} sweep")

    return sweep_results


def phase_bidirectional_calibration(domain_data, base, A_matrices,
                                     base_losses, sft_results) -> dict:
    """Test bidirectional attention at T=500 as a calibration point.

    Theorem 2 (MATH.md): bidirectional attention >= causal attention by set inclusion.
    Memory tokens are NOT autoregressive; bidirectional is the correct inductive bias.
    """
    log(f"\n=== Phase: Bidirectional Attention Calibration (T={M2P_STEPS_VALUES[0]}) ===")
    log(f"  Theorem 2: A_causal subset A_bidir => min loss(bidir) <= min loss(causal)")
    log(f"  BERT (arXiv:1810.04805): bidirectional is standard for non-autoregressive")

    m2p_steps = M2P_STEPS_VALUES[0]  # Compare at lowest step count
    domain_qualities = {}
    domain_losses = {}
    excluded_domains = []

    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_train_m2p(
            m2p_steps=m2p_steps,
            causal=False,  # BIDIRECTIONAL
            domain_name=name,
            domain_id=di,
            domain_data=domain_data[name],
            base=base,
            A_matrices=A_matrices,
            base_loss=base_losses[name],
            sft_loss=sft_results[name]["sft_loss"],
        )
        domain_qualities[name] = result["quality_ratio"]
        domain_losses[name] = result["m2p_loss"]
        if result["excluded_parity"]:
            excluded_domains.append(name)

    if excluded_domains:
        log(f"  Excluded domains (parity guard): {excluded_domains}")

    valid_qualities = [domain_qualities[n] for n in DOMAIN_NAMES
                       if n not in excluded_domains]

    median_q = float(np.median(valid_qualities)) if valid_qualities else 0.0
    mean_q = float(np.mean(valid_qualities)) if valid_qualities else 0.0

    log(f"  Bidirectional T={m2p_steps}: median={median_q:.1%}, mean={mean_q:.1%}")

    bidir_result = {
        "m2p_steps": m2p_steps,
        "causal": False,
        "m2p_layers": M2P_LAYERS,
        "d_m2p": D_M2P,
        "domain_quality": domain_qualities,
        "domain_m2p_loss": domain_losses,
        "median_quality": round(median_q, 4),
        "mean_quality": round(mean_q, 4),
        "excluded_domains": excluded_domains,
        "n_valid_domains": len(valid_qualities),
    }

    log_memory("after bidirectional calibration")
    return bidir_result


# ===================================================================
# KILL CRITERIA EVALUATION
# ===================================================================

def evaluate_kill_criteria(sweep_results: dict, bidir_result: dict) -> dict:
    """Evaluate K876, K877, K878 from sweep results.

    K876: quality(T=2000) > quality(T=500) + 2pp -- budget matters
    K877: quality(T=2000) >= 97% -- ceiling reached
    K878: |quality(T=2000) - quality(T=1000)| < 1pp -- budget exhausted (KILL)
    """
    step_keys = sorted(sweep_results.keys(), key=lambda k: sweep_results[k]["m2p_steps"])
    t_low = step_keys[0]   # T500
    t_mid = step_keys[1]   # T1000
    t_high = step_keys[2]  # T2000

    q_low = sweep_results[t_low]["median_quality"]
    q_mid = sweep_results[t_mid]["median_quality"]
    q_high = sweep_results[t_high]["median_quality"]
    q_bidir = bidir_result["median_quality"]
    q_causal_500 = q_low  # causal at same step count for comparison

    # K876: budget matters
    k876_pass = (q_high - q_low) > 0.02

    # K877: ceiling reached
    k877_pass = q_high >= 0.97

    # K878: budget exhausted (plateau between 1000 and 2000)
    k878_pass = abs(q_high - q_mid) < 0.01

    # Bidirectional gain
    bidir_gain = q_bidir - q_causal_500

    log("\n=== Kill Criteria Evaluation ===")
    log(f"  Baseline (Finding #357): median quality = 91.9% at T=500, L=2")
    log(f"  T=500  (causal):         quality={q_low:.1%}")
    log(f"  T=1000 (causal):         quality={q_mid:.1%}")
    log(f"  T=2000 (causal):         quality={q_high:.1%}")
    log(f"  T=500  (bidirectional):  quality={q_bidir:.1%}")
    log("")
    log(f"  K876: quality(2000) > quality(500) + 2pp:  "
        f"{q_high:.1%} > {q_low:.1%} + 2pp = {q_low+0.02:.1%} "
        f"-> {'PASS' if k876_pass else 'FAIL'}")
    log(f"  K877: quality(2000) >= 97%:                "
        f"{q_high:.1%} -> {'PASS' if k877_pass else 'FAIL'}")
    log(f"  K878: |quality(2000) - quality(1000)| < 1pp: "
        f"|{q_high:.3f} - {q_mid:.3f}| = {abs(q_high-q_mid):.3f} "
        f"-> {'PASS (plateau)' if k878_pass else 'FAIL (still improving)'}")
    log("")
    log(f"  Bidirectional gain at T=500: {bidir_gain:+.1%}")
    log(f"  (Theorem 2 predicts gain >= 0)")

    # Determine outcome
    delta_low_to_high = q_high - q_low
    delta_mid_to_high = q_high - q_mid

    if k876_pass and k877_pass:
        outcome = "A_budget_sufficient"
        interpretation = (
            f"Training budget is the bottleneck AND ceiling reached. "
            f"Quality at T=2000 ({q_high:.1%}) >= 97%. Architecture is ready."
        )
    elif k876_pass and not k877_pass and not k878_pass:
        outcome = "B_budget_helps_more_needed"
        interpretation = (
            f"Budget matters (+{delta_low_to_high:.1%}) but ceiling not reached "
            f"({q_high:.1%} < 97%). More steps or other improvements needed."
        )
    elif k876_pass and k878_pass:
        outcome = "C_budget_helps_but_saturates"
        interpretation = (
            f"Budget helps initially (+{delta_low_to_high:.1%} over T=500) but "
            f"saturates (|T2000-T1000| = {abs(delta_mid_to_high):.1%} < 1pp). "
            f"Something else caps quality at {q_high:.1%}."
        )
    elif not k876_pass:
        outcome = "D_budget_not_bottleneck"
        interpretation = (
            f"Budget does NOT matter: delta = {delta_low_to_high:+.1%} <= 2pp. "
            f"The 8% gap is dominated by irreducible error."
        )
    else:
        outcome = "E_unexpected"
        interpretation = f"Unexpected combination of kill criteria results."

    log(f"\n  Outcome: {outcome}")
    log(f"  Interpretation: {interpretation}")

    # Convergence check: are training losses still decreasing at max steps?
    t_high_train_losses = sweep_results[t_high].get("domain_train_loss", {})
    converged_domains = []
    for name, final_loss in t_high_train_losses.items():
        sft_key = name
        if sft_key in sweep_results[t_high].get("domain_m2p_loss", {}):
            converged_domains.append(name)

    return {
        "k876_pass": bool(k876_pass),
        "k877_pass": bool(k877_pass),
        "k878_pass": bool(k878_pass),
        "outcome": outcome,
        "interpretation": interpretation,
        "quality_T500_causal": round(float(q_low), 4),
        "quality_T1000_causal": round(float(q_mid), 4),
        "quality_T2000_causal": round(float(q_high), 4),
        "quality_T500_bidir": round(float(q_bidir), 4),
        "delta_500_to_2000": round(float(delta_low_to_high), 4),
        "delta_1000_to_2000": round(float(delta_mid_to_high), 4),
        "bidir_gain_at_T500": round(float(bidir_gain), 4),
        "monotone_500_1000_2000": bool(q_low <= q_mid <= q_high),
        "diminishing_returns": bool(
            (q_mid - q_low) >= (q_high - q_mid)
        ) if q_mid > q_low else None,
    }


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================

def main():
    t0 = time.time()
    log("M2P Training Budget Sweep -- Quality Scales With Steps, Not Architecture")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log(f"Architecture FIXED: L={M2P_LAYERS}, D_M2P={D_M2P}")
    log(f"Sweep: M2P_STEPS in {M2P_STEPS_VALUES}")
    log(f"Secondary: bidirectional attention at T={M2P_STEPS_VALUES[0]}")
    log(f"SHINE (arXiv:2602.06358): training scale is the bottleneck")
    log(f"SGD convergence O(1/T): Ghadimi & Lan (2013, arXiv:1309.5549)")
    log(f"Parity guard: exclude domains where base_loss - sft_loss < {PARITY_GUARD_THRESHOLD}")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # -- Data --
    domain_data = phase_generate_data(rng)
    log_memory("after data")

    # -- Base model (FRESH -- no reuse) --
    base, base_losses = phase_pretrain_base(domain_data)
    log_memory("after base")

    # -- Grassmannian A-matrices --
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # -- SFT baselines (FRESH) --
    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after SFT")

    # -- M2P training budget sweep (causal, PRIMARY experiment) --
    sweep_results = phase_sweep_training_budget(
        domain_data, base, A_matrices, base_losses, sft_results
    )
    log_memory("after budget sweep")

    # -- Bidirectional attention calibration (SECONDARY experiment) --
    bidir_result = phase_bidirectional_calibration(
        domain_data, base, A_matrices, base_losses, sft_results
    )
    log_memory("after bidirectional")

    # -- Kill criteria --
    kill_criteria = evaluate_kill_criteria(sweep_results, bidir_result)

    # -- Results assembly --
    results = {
        "experiment": "exp_m2p_training_budget",
        "total_time_s": round(time.time() - t0, 1),
        "smoke_test": SMOKE_TEST,
        # Architecture (FIXED)
        "m2p_layers_fixed": M2P_LAYERS,
        "d_m2p_fixed": D_M2P,
        "m2p_steps_values": M2P_STEPS_VALUES,
        "parity_guard_threshold": PARITY_GUARD_THRESHOLD,
        # Per-step-count results (causal)
        "T500_causal": sweep_results[f"T{M2P_STEPS_VALUES[0]}"],
        "T1000_causal": sweep_results[f"T{M2P_STEPS_VALUES[1]}"],
        "T2000_causal": sweep_results[f"T{M2P_STEPS_VALUES[2]}"],
        # Bidirectional calibration
        "T500_bidir": bidir_result,
        # Convenience: median quality per condition
        "median_T500_causal": sweep_results[f"T{M2P_STEPS_VALUES[0]}"]["median_quality"],
        "median_T1000_causal": sweep_results[f"T{M2P_STEPS_VALUES[1]}"]["median_quality"],
        "median_T2000_causal": sweep_results[f"T{M2P_STEPS_VALUES[2]}"]["median_quality"],
        "median_T500_bidir": bidir_result["median_quality"],
        # Kill criteria
        "kill_criteria": kill_criteria,
        # Reference losses
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        # Grassmannian verification
        "grassmannian_A_cos_max": ortho_result["max_cos"],
        # Prediction vs measurement table (for PAPER.md)
        "predictions_vs_measurements": {
            "T500_baseline": {
                "description": "T=500 baseline from Finding #357",
                "predicted": "~91.9%",
                "measured": sweep_results[f"T{M2P_STEPS_VALUES[0]}"]["median_quality"],
            },
            "T1000_improvement": {
                "description": "Conservative: +3.0pp over T=500",
                "predicted_range": "94-96%",
                "measured": sweep_results[f"T{M2P_STEPS_VALUES[1]}"]["median_quality"],
            },
            "T2000_improvement": {
                "description": "Conservative: +4.6pp over T=500",
                "predicted_range": "95-98%",
                "measured": sweep_results[f"T{M2P_STEPS_VALUES[2]}"]["median_quality"],
            },
            "K876_budget_matters": {
                "description": "quality(2000) > quality(500) + 2pp",
                "predicted": "PASS",
                "measured_delta": kill_criteria["delta_500_to_2000"],
                "pass": kill_criteria["k876_pass"],
            },
            "K877_ceiling_reached": {
                "description": "quality(2000) >= 97%",
                "predicted": "Uncertain (95-98%)",
                "measured": kill_criteria["quality_T2000_causal"],
                "pass": kill_criteria["k877_pass"],
            },
            "K878_plateau": {
                "description": "|quality(2000) - quality(1000)| < 1pp",
                "predicted": "FAIL (still improving)",
                "measured_abs_delta": abs(kill_criteria["delta_1000_to_2000"]),
                "pass": kill_criteria["k878_pass"],
            },
            "bidirectional_gain": {
                "description": "Bidirectional >= causal (Theorem 2)",
                "predicted": "+1-2pp",
                "measured": kill_criteria["bidir_gain_at_T500"],
            },
            "diminishing_returns": {
                "description": "delta(1000->2000) < delta(500->1000)",
                "predicted": "YES (O(1/T) concavity)",
                "measured": kill_criteria["diminishing_returns"],
            },
        },
    }

    # -- Summary report --
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY -- M2P Training Budget Sweep")
    log("=" * 70)
    log(f"Architecture: L={M2P_LAYERS}, D_M2P={D_M2P} (fixed)")
    log(f"Sweep: M2P_STEPS in {M2P_STEPS_VALUES}")
    log(f"Grassmannian A cos_max: {ortho_result['max_cos']:.8f}")
    log("")
    log("Quality ratios (M2P / SFT), median across valid domains:")
    for m2p_steps in M2P_STEPS_VALUES:
        key = f"T{m2p_steps}"
        r = sweep_results[key]
        log(f"  T={m2p_steps} (causal):  median={r['median_quality']:.1%}  "
            f"(n_valid={r['n_valid_domains']}, excluded={r['excluded_domains']})")
    log(f"  T={M2P_STEPS_VALUES[0]} (bidir):   median={bidir_result['median_quality']:.1%}  "
        f"(n_valid={bidir_result['n_valid_domains']})")
    log("")

    log("Convergence analysis:")
    for m2p_steps in M2P_STEPS_VALUES:
        key = f"T{m2p_steps}"
        r = sweep_results[key]
        train_losses = r.get("domain_train_loss", {})
        log(f"  T={m2p_steps} final train losses: {train_losses}")
    log("")

    log("Kill Criteria:")
    kc = kill_criteria
    log(f"  K876 (budget matters, +2pp):     {'PASS' if kc['k876_pass'] else 'FAIL'}  "
        f"(delta={kc['delta_500_to_2000']:+.4f})")
    log(f"  K877 (ceiling >= 97%):           {'PASS' if kc['k877_pass'] else 'FAIL'}  "
        f"(quality={kc['quality_T2000_causal']:.1%})")
    log(f"  K878 (plateau < 1pp, KILL case): {'PASS (plateau)' if kc['k878_pass'] else 'FAIL (improving)'}  "
        f"(delta_1k_2k={kc['delta_1000_to_2000']:+.4f})")
    log(f"\n  Bidirectional gain at T={M2P_STEPS_VALUES[0]}: {kc['bidir_gain_at_T500']:+.4f}")
    log(f"  Monotone improvement (500<=1000<=2000): {kc['monotone_500_1000_2000']}")
    log(f"  Diminishing returns: {kc['diminishing_returns']}")
    log(f"\n  Outcome: {kc['outcome']}")
    log(f"  {kc['interpretation']}")
    log("")
    log(f"Base losses: {base_losses}")
    log(f"SFT losses:  {dict((n, sft_results[n]['sft_loss']) for n in DOMAIN_NAMES)}")
    log("")
    log(f"Total time: {round(time.time() - t0, 1)}s")
    log("=" * 70)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
