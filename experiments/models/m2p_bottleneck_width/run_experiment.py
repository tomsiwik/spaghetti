#!/usr/bin/env python3
"""M2P Bottleneck Width Sweep: JL-bound fix at d_M2P in {64, 128, 256}.

TYPE: verification (Type 1)
MATH: experiments/models/m2p_bottleneck_width/MATH.md

ROOT CAUSE FIXED (exp_m2p_tfidf_routing_n5, Finding #354):
  TF-IDF routing achieved 95% accuracy AND oracle routing gave 92.2% of SFT.
  Oracle routing == TF-IDF routing quality → routing is NOT the bottleneck.
  The 7.8% gap is pure M2P generation quality, caused by d_M2P=64 being
  54% below the JL bound d_JL=138 (for N=5 adapters, epsilon=0.1).

HYPOTHESIS (Theorem 1, MATH.md):
  Increasing d_M2P from 64 to 128 (93% of d_JL=138) closes the 7.8% gap.
  d_M2P=256 (185% of d_JL) shows JL saturation: no significant gain over 128.

JL BOUND:
  d_JL(N=5, eps=0.1) = (4 ln 5) / (0.005 - 0.000333) = 6.4378 / 0.004667 = 138

KILL CRITERIA:
  K870: M2P quality >= 97% of SFT at d_M2P=128 (vs 92.2% at d_M2P=64)
  K871: quality(d_M2P=128) > quality(d_M2P=64)
  K872: |quality(d_M2P=256) - quality(d_M2P=128)| < 0.02 (JL saturation)

REUSE POLICY:
  - Load base weights from m2p_composition_n5/base_weights.npz if available.
  - Load SFT adapters from m2p_composition_n5/adapters/ if available.
  - Train one fresh M2P per d_m2p value — these are the experiment variable.

ARCHITECTURE (identical to m2p_tfidf_routing_n5, except D_M2P is swept):
  D_MODEL=256, N_LAYERS=2, N_HEADS=4, VOCAB_SIZE=128, BLOCK_SIZE=48
  LORA_RANK=4, N_DOMAINS=5, M2P_LAYERS=2
  d_M2P swept: {64, 128, 256}
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

# Memory safety (CODING_GUIDELINES §2)
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source adapters from m2p_composition_n5 (reuse if available)
SOURCE_DIR = EXPERIMENT_DIR.parent / "m2p_composition_n5"
SOURCE_ADAPTER_DIR = SOURCE_DIR / "adapters"

SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── Architecture constants (IDENTICAL to m2p_tfidf_routing_n5) ────────────
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5

# M2P sweep values — the experiment variable
D_M2P_VALUES = [64, 128, 256]

N_MEMORY = 32
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
M2P_STEPS  = 500  if not SMOKE_TEST else 40
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]

# JL bound constants (from MATH.md Section B.2)
D_JL = 138  # d_JL(N=5, eps=0.1) = (4*ln(5)) / (0.005 - 0.000333) = 138


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


# ── Data generation (IDENTICAL to m2p_tfidf_routing_n5) ──────────────────

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


# ── Toy GPT (IDENTICAL to m2p_tfidf_routing_n5) ───────────────────────────

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


# ── Grassmannian A-matrices (IDENTICAL to m2p_tfidf_routing_n5) ──────────

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


# ── LoRA forward pass (IDENTICAL to m2p_tfidf_routing_n5) ────────────────

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


# ── SFT training (IDENTICAL to m2p_tfidf_routing_n5) ─────────────────────

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


# ── M2P Transformer (d_m2p is parameterized — KEY sweep variable) ─────────

class M2PAttention(nn.Module):
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
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scale = hd ** -0.5
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
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
    """M2P transformer with variable bottleneck dimension d_m2p.

    n_heads=4 is compatible with all three d_m2p values:
      d=64:  head_dim = 64/4 = 16  (valid)
      d=128: head_dim = 128/4 = 32 (valid)
      d=256: head_dim = 256/4 = 64 (valid)
    """
    def __init__(self, d_base=D_MODEL, d_m2p=64):
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


# ── Evaluation helpers ─────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

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

    Training is always run fresh — it is fast (BASE_STEPS steps) and
    deterministic. This avoids fragile weight loading across experiments.
    The SFT adapters (which depend on base weights) are reused from
    m2p_composition_n5 only if available AND the base is retrained to the
    same state (same SEED, same data, same steps).
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
    """Train SFT LoRA adapter for one domain."""
    log(f"  SFT {domain_name} (domain {domain_id})...")

    local_path = EXPERIMENT_DIR / "adapters" / f"sft_{domain_name}.npz"
    source_path = SOURCE_ADAPTER_DIR / f"sft_{domain_name}.npz"

    local_path.parent.mkdir(exist_ok=True)

    if local_path.exists():
        log(f"    Reusing existing local SFT adapter")
        b_container = BMatrices()
        data = np.load(str(local_path))
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                setattr(b_container, f"B_{li}_{mi}",
                        mx.array(data[f"{li}_{mi}"]))
        mx.eval(b_container.parameters())
    elif source_path.exists():
        log(f"    Reusing SFT adapter from m2p_composition_n5")
        import shutil
        local_path.parent.mkdir(exist_ok=True)
        shutil.copy(str(source_path), str(local_path))
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
    log("\n=== Phase: SFT Baselines ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            name, di, domain_data[name], base, A_matrices, base_losses[name]
        )
        sft_results[name] = result
    return sft_results


def phase_train_m2p_for_width(d_m2p: int, domain_name: str, domain_id: int,
                               domain_data: dict, base: ToyGPT,
                               A_matrices: dict,
                               base_loss: float, sft_loss: float) -> dict:
    """Train ONE M2P at a given d_m2p for ONE domain.

    This is the KEY experiment function — the only variable is d_m2p.
    Every other parameter is identical across all three d_m2p values.

    Returns quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)
    """
    adapter_dir = EXPERIMENT_DIR / f"adapters_d{d_m2p}"
    adapter_dir.mkdir(exist_ok=True)
    save_path = adapter_dir / f"m2p_{domain_name}.npz"

    if save_path.exists():
        log(f"    [d={d_m2p}] Reusing cached M2P for {domain_name}")
    else:
        log(f"    [d={d_m2p}] Training M2P for {domain_name}...")
        mx.random.seed(SEED)

        m2p = M2PTransformer(d_base=D_MODEL, d_m2p=d_m2p)
        mx.eval(m2p.parameters())

        param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
        log(f"      M2P params (d={d_m2p}): {param_count:,}")

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
            if (step + 1) % max(1, M2P_STEPS // 4) == 0:
                log(f"      Step {step+1}/{M2P_STEPS}: loss={loss.item():.4f}")
        gc.enable()
        cleanup(optimizer)

        # Save representative B-matrices from the first training context
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
    B_matrices = load_B_matrices(str(save_path))
    m2p_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)

    quality_ratio = 0.0
    if (base_loss - sft_loss) > 0.01:
        quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)

    log(f"    [d={d_m2p}] {domain_name}: m2p_loss={m2p_loss:.4f} "
        f"SFT={sft_loss:.4f} base={base_loss:.4f} quality={quality_ratio:.1%}")

    cleanup(B_matrices)
    return {
        "m2p_loss": round(m2p_loss, 4),
        "quality_ratio": round(quality_ratio, 4),
    }


def phase_sweep_m2p_widths(domain_data, base, A_matrices,
                            base_losses, sft_results) -> dict:
    """Sweep d_M2P ∈ {64, 128, 256}, training fresh M2P per value.

    Each d_M2P gets fresh M2P transformers trained from scratch.
    Base model, A-matrices, SFT adapters are IDENTICAL across all three.
    """
    log("\n=== Phase: M2P Bottleneck Width Sweep ===")
    log(f"  Sweeping d_M2P ∈ {D_M2P_VALUES}")
    log(f"  JL bound: d_JL = {D_JL} (N=5, eps=0.1)")
    log(f"  d=64: {64/D_JL:.0%} of JL bound (below)")
    log(f"  d=128: {128/D_JL:.0%} of JL bound (near)")
    log(f"  d=256: {256/D_JL:.0%} of JL bound (above)")

    sweep_results = {}

    for d_m2p in D_M2P_VALUES:
        log(f"\n  --- d_M2P = {d_m2p} (d/d_JL = {d_m2p/D_JL:.2f}) ---")
        domain_qualities = {}
        domain_losses = {}

        for di, name in enumerate(DOMAIN_NAMES):
            result = phase_train_m2p_for_width(
                d_m2p=d_m2p,
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

        # Exclude "parity" domain if it's a measurement artifact
        # (base model near SFT for parity → quality_ratio undefined)
        # Per project convention: exclude domains where base_loss - sft_loss < 0.01
        valid_qualities = []
        excluded = []
        for name in DOMAIN_NAMES:
            bl = base_losses[name]
            sl = sft_results[name]["sft_loss"]
            if (bl - sl) > 0.01:
                valid_qualities.append(domain_qualities[name])
            else:
                excluded.append(name)

        if excluded:
            log(f"  Excluding {excluded} (measurement artifact: base ≈ SFT)")

        median_q = float(np.median(valid_qualities)) if valid_qualities else 0.0
        mean_q = float(np.mean(valid_qualities)) if valid_qualities else 0.0

        log(f"  d={d_m2p}: median quality={median_q:.1%}, mean={mean_q:.1%}")
        log(f"  Per-domain: {dict((k, f'{v:.1%}') for k, v in domain_qualities.items())}")

        sweep_results[f"d{d_m2p}"] = {
            "d_m2p": d_m2p,
            "d_over_jl": round(d_m2p / D_JL, 4),
            "domain_quality": domain_qualities,
            "domain_m2p_loss": domain_losses,
            "median_quality": round(median_q, 4),
            "mean_quality": round(mean_q, 4),
            "excluded_domains": excluded,
            "n_valid_domains": len(valid_qualities),
        }

        log_memory(f"after d={d_m2p} sweep")

    return sweep_results


# ═══════════════════════════════════════════════════════════════════════════
# KILL CRITERIA EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_kill_criteria(sweep_results: dict) -> dict:
    """Evaluate K870, K871, K872 from sweep results.

    K870: quality(d=128) >= 0.97
    K871: quality(d=128) > quality(d=64)
    K872: |quality(d=256) - quality(d=128)| < 0.02  (JL saturation)
    """
    q64  = sweep_results["d64"]["median_quality"]
    q128 = sweep_results["d128"]["median_quality"]
    q256 = sweep_results["d256"]["median_quality"]

    k870_pass = q128 >= 0.97
    k871_pass = q128 > q64
    k872_pass = abs(q256 - q128) < 0.02

    log("\n=== Kill Criteria Evaluation ===")
    log(f"  Baseline (Finding #354): 92.2% at d=64")
    log(f"  d=64:  quality={q64:.1%}  (reference)")
    log(f"  d=128: quality={q128:.1%}  (d/d_JL={128/D_JL:.2f})")
    log(f"  d=256: quality={q256:.1%}  (d/d_JL={256/D_JL:.2f})")
    log("")
    log(f"  K870: quality(d=128) >= 97%:    {q128:.1%} → {'PASS' if k870_pass else 'FAIL'}")
    log(f"  K871: quality(d=128) > quality(d=64): {q128:.3f} > {q64:.3f} → {'PASS' if k871_pass else 'FAIL'}")
    log(f"  K872: |quality(256)-quality(128)| < 2%: |{q256:.3f}-{q128:.3f}|={abs(q256-q128):.3f} → {'PASS' if k872_pass else 'FAIL'}")

    return {
        "k870_pass": k870_pass,
        "k871_pass": k871_pass,
        "k872_pass": k872_pass,
        "all_pass": k870_pass and k871_pass and k872_pass,
        "quality_d64": round(q64, 4),
        "quality_d128": round(q128, 4),
        "quality_d256": round(q256, 4),
        "improvement_64_to_128": round(q128 - q64, 4),
        "improvement_128_to_256": round(q256 - q128, 4),
        "jl_saturation_confirmed": k872_pass,
        "d_jl": D_JL,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("M2P Bottleneck Width Sweep — JL-Bound Fix")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log(f"d_JL(N=5, eps=0.1) = {D_JL}")
    log(f"Sweep: d_M2P ∈ {D_M2P_VALUES}")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # ── Data ──────────────────────────────────────────────────────────────
    domain_data = phase_generate_data(rng)
    log_memory("after data")

    # ── Base model ────────────────────────────────────────────────────────
    base, base_losses = phase_pretrain_base(domain_data)
    log_memory("after base")

    # ── Grassmannian A-matrices ───────────────────────────────────────────
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # ── SFT baselines ─────────────────────────────────────────────────────
    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after SFT")

    # ── M2P width sweep (KEY experiment) ──────────────────────────────────
    sweep_results = phase_sweep_m2p_widths(
        domain_data, base, A_matrices, base_losses, sft_results
    )
    log_memory("after sweep")

    # ── Kill criteria ─────────────────────────────────────────────────────
    kill_criteria = evaluate_kill_criteria(sweep_results)

    # ── Results assembly ──────────────────────────────────────────────────
    results = {
        "experiment": "exp_m2p_bottleneck_width",
        "total_time_s": round(time.time() - t0, 1),
        "smoke_test": SMOKE_TEST,
        # JL bound reference
        "d_jl": D_JL,
        "d_m2p_values": D_M2P_VALUES,
        # Per-d_M2P results (for K870-K872)
        "d64": sweep_results["d64"],
        "d128": sweep_results["d128"],
        "d256": sweep_results["d256"],
        # Convenience: median quality per width
        "median_d64": sweep_results["d64"]["median_quality"],
        "median_d128": sweep_results["d128"]["median_quality"],
        "median_d256": sweep_results["d256"]["median_quality"],
        # Kill criteria
        "kill_criteria": kill_criteria,
        # Reference losses
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        # Grassmannian verification
        "grassmannian_A_cos_max": ortho_result["max_cos"],
        # Prediction vs measurement table (for PAPER.md)
        "predictions_vs_measurements": {
            "d64_quality": {
                "prediction_from_finding_354": 0.922,
                "measured": sweep_results["d64"]["median_quality"],
                "match": abs(sweep_results["d64"]["median_quality"] - 0.922) < 0.05,
            },
            "d128_quality": {
                "predicted_gte": 0.97,
                "measured": sweep_results["d128"]["median_quality"],
                "pass": kill_criteria["k870_pass"],
            },
            "d256_vs_d128": {
                "predicted_abs_diff_lt": 0.02,
                "measured_abs_diff": abs(
                    sweep_results["d256"]["median_quality"] -
                    sweep_results["d128"]["median_quality"]
                ),
                "pass": kill_criteria["k872_pass"],
            },
            "monotonic_improvement": {
                "predicted_128_gt_64": True,
                "measured": kill_criteria["k871_pass"],
            },
        },
    }

    # ── Summary report ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY — M2P Bottleneck Width Sweep")
    log("=" * 70)
    log(f"JL bound: d_JL = {D_JL} (N=5, eps=0.1)")
    log(f"Grassmannian A cos_max: {ortho_result['max_cos']:.8f}")
    log("")
    log("Quality ratios (M2P / SFT), median across valid domains:")
    for d_m2p in D_M2P_VALUES:
        key = f"d{d_m2p}"
        q = sweep_results[key]["median_quality"]
        ratio = d_m2p / D_JL
        log(f"  d={d_m2p:3d} ({ratio:.2f}x JL): {q:.1%}")
    log("")
    log(f"K870 (d=128 >= 97%):    {'PASS' if kill_criteria['k870_pass'] else 'FAIL'}")
    log(f"K871 (128 > 64):        {'PASS' if kill_criteria['k871_pass'] else 'FAIL'}")
    log(f"K872 (|256-128| < 2%):  {'PASS' if kill_criteria['k872_pass'] else 'FAIL'}")
    log("")
    log(f"OVERALL: {'ALL PASS' if kill_criteria['all_pass'] else 'PARTIAL/FAIL'} "
        f"in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
