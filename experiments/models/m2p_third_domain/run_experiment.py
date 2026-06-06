#!/usr/bin/env python3
"""M2P Third Domain: Caesar Cipher — Structural Diversity Validation.

Kill criteria:
  K_3dom: sort/reverse/cipher quality >= 85%
  K_diversity: cross-domain sort->cipher transfer < 50%
  K_replication: sort/reverse within 5pp of Finding #361 (101%)

Supports SMOKE_TEST=1 for <60s validation.
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

# ARCHITECTURE CONSTANTS (same as m2p_macro_quality)
D_MODEL = 512
N_LAYERS = 2
N_HEADS = 8
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0

# M2P architecture FIXED (proven sufficient, Findings #355, #357)
M2P_LAYERS = 2
D_M2P = 64
N_MEMORY = 32

# N_DOMAINS = 4: arithmetic, sort, reverse, cipher
N_DOMAINS = 4
DOMAIN_NAMES = ["arithmetic", "sort", "reverse", "cipher"]

# Module names and output dims (scaled with D_MODEL=512)
MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

# Training (proven recipe: n=2000, T=1000, GL stopping)
N_SAMPLES = 200 if SMOKE_TEST else 2000
T_FIXED = 20 if SMOKE_TEST else 1000
BASE_STEPS = 60 if SMOKE_TEST else 1200
SFT_STEPS = T_FIXED
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

# Cross-domain pairs to evaluate: sort->cipher (1->3) and reverse->cipher (2->3)
# Cipher is domain index 3
CROSS_PAIRS = [(1, 3), (2, 3)]  # (src_domain_id, tgt_domain_id=cipher)
CROSS_STEPS = 10 if SMOKE_TEST else 300

# Early stopping (GL criterion, Prechelt 1998)
EARLY_STOP_INTERVAL = 10 if SMOKE_TEST else 50
GL_THRESHOLD = 5.0
PATIENCE = 5

# Parity guard: skip quality ratio if SFT doesn't actually learn
PARITY_GUARD_THRESHOLD = 0.05

# UTILITIES

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

# DATA GENERATION

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    """Generate synthetic task data for 4 domains (arithmetic/sort/reverse/cipher)."""
    chars = "abcdefgh"
    cchars = "abcdefghijklmnopqrstuvwxyz"
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
        elif domain_id == 3:  # cipher: Caesar shift, wraps around z->a
            shift = rng.randint(1, 26)
            length = rng.randint(2, 6)
            plain = "".join(rng.choice(list(cchars)) for _ in range(length))
            shifted = "".join(cchars[(cchars.index(c) + shift) % 26] for c in plain)
            data.append(f"{plain}>{shifted}")
    return data

def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]

def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]

# TOY GPT (d=512, 8 heads, L=2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, d, n_heads):
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
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd ** -0.5) + mask, axis=-1)
        return self.wo((a @ v).transpose(0, 2, 1, 3).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

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
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm_f(x))

    def get_hidden_states(self, tokens):
        B, T = tokens.shape
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states

# GRASSMANNIAN A-MATRICES

def generate_grassmannian_A(n_slots, n_layers, n_modules, d, rank, seed=42):
    total_rank = n_slots * rank
    assert total_rank <= d, f"total_rank={total_rank} > d={d}"
    rng = np.random.RandomState(seed)
    A = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            Q, _ = np.linalg.qr(rng.randn(d, total_rank).astype(np.float32))
            for si in range(n_slots):
                A[(si, li, mi)] = mx.array(Q[:, si * rank:(si + 1) * rank])
    return A

def verify_grassmannian_orthogonality(A_matrices, n_slots, n_layers, n_modules):
    cos_values = [
        mx.abs(mx.sum(A_matrices[(si, li, mi)].reshape(-1) * A_matrices[(sj, li, mi)].reshape(-1)) /
               (mx.linalg.norm(A_matrices[(si, li, mi)].reshape(-1)) *
                mx.linalg.norm(A_matrices[(sj, li, mi)].reshape(-1)) + 1e-12)).item()
        for li in range(n_layers) for mi in range(n_modules)
        for si in range(n_slots) for sj in range(si + 1, n_slots)
    ]
    return {"mean_cos": float(np.mean(cos_values)), "max_cos": float(np.max(cos_values))}

# LORA FORWARD PASS

def lora_forward_with_B(base, tokens, A_matrices, slot_id, B_matrices):
    _, T = tokens.shape
    x = base.wte(tokens) + base.wpe(mx.arange(T))
    def _lora(w, x_in, li, mi):
        return w(x_in) + LORA_SCALE * (x_in @ A_matrices[(slot_id, li, mi)]) @ B_matrices[(li, mi)]
    for li, block in enumerate(base.blocks):
        xn = block.norm1(x)
        Bb, Tb, C = xn.shape
        h, hd = block.attn.n_heads, block.attn.head_dim
        q = _lora(block.attn.wq, xn, li, 0).reshape(Bb, Tb, h, hd).transpose(0, 2, 1, 3)
        k = _lora(block.attn.wk, xn, li, 1).reshape(Bb, Tb, h, hd).transpose(0, 2, 1, 3)
        v = _lora(block.attn.wv, xn, li, 2).reshape(Bb, Tb, h, hd).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((Tb, Tb), float("-inf")), k=1)
        ctx = (mx.softmax(q @ k.transpose(0, 1, 3, 2) * hd**-0.5 + mask, axis=-1) @ v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(Bb, Tb, C)
        x = x + _lora(block.attn.wo, ctx, li, 3)
        xn2 = block.norm2(x)
        fc1 = _lora(block.mlp.fc1, xn2, li, 4)
        x = x + block.mlp.fc2(nn.gelu(fc1))
    return base.lm_head(base.norm_f(x))

# SFT / B-MATRIX INFRASTRUCTURE

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

def sft_loss_fn(b_container, base, tokens, A_matrices, slot_id):
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, slot_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")

def save_B(b_container, path):
    np_dict = {f"{li}_{mi}": np.array(getattr(b_container, f"B_{li}_{mi}"))
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(path), **np_dict)

def load_B(path):
    data = np.load(str(path))
    B_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(N_MODULES):
            B_matrices[(li, mi)] = mx.array(data[f"{li}_{mi}"])
    return B_matrices

# M2P TRANSFORMER

class M2PBlock(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class M2PTransformer(nn.Module):
    """M2P: d_M2P=64, L=2, N_MEMORY=32. Proven recipe (Finding #355, #357)."""

    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(m2p_layers)]
        self.norm_f = RMSNorm(d_m2p)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            total_out = N_LAYERS * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list):
        encs = [self.input_proj(mx.mean(h[0], axis=0)) for h in hidden_states_list]
        ctx = mx.mean(mx.stack(encs, axis=0), axis=0)
        mem = self.memory_tokens + self.pos_embed(mx.arange(N_MEMORY)) + ctx[None, :]
        x = mem[None, :, :]
        for blk in self.blocks: x = blk(x)
        pm = mx.mean(self.norm_f(x)[0], axis=0)
        B = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pm).reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS): B[(li, mi)] = out[li]
        return B

def m2p_ntp_loss(m2p, base, A_matrices, slot_id, tokens):
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, slot_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")

def cross_m2p_loss(m2p_cross, base, A_matrices, cross_slot_id, context_tokens, target_tokens):
    """Cross-domain M2P: show context from src domain, predict on target domain."""
    hidden_states = base.get_hidden_states(context_tokens)
    B_cross = m2p_cross(hidden_states)
    logits = lora_forward_with_B(base, target_tokens, A_matrices, cross_slot_id, B_cross)
    return nn.losses.cross_entropy(logits[:, :-1], target_tokens[:, 1:], reduction="mean")

# EVALUATION

def eval_ntp_loss(base, batches, A_matrices=None, slot_id=None, B_matrices=None):
    total, n = 0.0, 0
    for tok in batches[:50]:
        t2d = tok[None, :]
        logits = (lora_forward_with_B(base, t2d, A_matrices, slot_id, B_matrices)
                  if A_matrices is not None and B_matrices is not None else base(t2d))
        loss = nn.losses.cross_entropy(logits[:, :-1], t2d[:, 1:], reduction="mean")
        mx.eval(loss)
        total += loss.item(); n += 1
        del logits, loss
    return total / max(n, 1)

# PHASE FUNCTIONS

def phase_generate_data(rng, n_per_domain):
    log(f"\n=== Data: n={n_per_domain} x {N_DOMAINS} domains ===")
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        tr, vl = make_batches(texts[:split]), make_batches(texts[split:])
        domain_data[name] = {"train": tr, "val": vl, "domain_id": di}
        log(f"  {name}: {len(tr)} train, {len(vl)} val")
    return domain_data

def phase_pretrain_base(domain_data: dict) -> tuple:
    log("\n=== Phase: Pre-train Base (d=512, 4 domains) ===")
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
    cleanup(optimizer)
    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")
    return base, base_losses

def phase_grassmannian(n_total_slots: int) -> tuple:
    log(f"\n=== Phase: Grassmannian ({n_total_slots} slots, total_rank={n_total_slots * LORA_RANK}) ===")
    A_matrices = generate_grassmannian_A(
        n_total_slots, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK, seed=SEED
    )
    ortho = verify_grassmannian_orthogonality(A_matrices, n_total_slots, N_LAYERS, N_MODULES)
    log(f"  max|cos|={ortho['max_cos']:.2e}")
    assert ortho["max_cos"] < 1e-5, f"Grassmannian failed: max|cos|={ortho['max_cos']}"
    return A_matrices, ortho

def phase_sft_domain(domain_name, domain_id, domain_data, base,
                     A_matrices, base_loss) -> dict:
    log(f"  SFT {domain_name}...")
    local_path = ADAPTER_DIR / f"sft_{domain_name}.npz"
    if local_path.exists():
        b_container = BMatrices()
        data = np.load(str(local_path))
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                setattr(b_container, f"B_{li}_{mi}", mx.array(data[f"{li}_{mi}"]))
        mx.eval(b_container.parameters())
    else:
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
        save_B(b_container, local_path)

    B_matrices = b_container.as_dict()
    sft_loss = eval_ntp_loss(base, domain_data["val"], A_matrices, domain_id, B_matrices)
    log(f"    {domain_name}: sft_loss={sft_loss:.4f} base={base_loss:.4f}")
    cleanup(b_container)
    return {"sft_loss": round(sft_loss, 4), "save_path": str(local_path)}

def phase_sft_all_domains(domain_data, base, A_matrices, base_losses):
    log("\n=== Phase: SFT Baselines ===")
    return {name: phase_sft_domain(name, di, domain_data[name], base, A_matrices, base_losses[name])
            for di, name in enumerate(DOMAIN_NAMES)}

def phase_train_m2p(domain_name, domain_id, domain_data, base, A_matrices,
                    base_loss, sft_loss) -> dict:
    """Train M2P for one domain with GL early stopping (Prechelt 1998)."""
    n_train = len(domain_data["train"])
    save_path = ADAPTER_DIR / f"m2p_{domain_name}.npz"
    log(f"  M2P {domain_name} (slot {domain_id}, T={T_FIXED}, n_train={n_train})...")
    mx.random.seed(SEED)

    m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS)
    mx.eval(m2p.parameters())

    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m2p_model, tokens):
        return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id, tokens[None, :])

    grad_fn = nn.value_and_grad(m2p, _loss)
    train_batches = domain_data["train"]
    val_batches = domain_data["val"]

    best_val_loss = float("inf")
    consecutive_gl_exceeded = 0
    early_stop_triggered = False
    stopping_step = T_FIXED
    final_train_loss = None

    gc.disable()
    for step in range(T_FIXED):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = grad_fn(m2p, tokens)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        final_train_loss = loss.item()

        if (step + 1) % EARLY_STOP_INTERVAL == 0 and not early_stop_triggered:
            gc.enable()
            context_tokens = train_batches[0][None, :]
            hidden_states = base.get_hidden_states(context_tokens)
            B_now = m2p(hidden_states)
            mx.eval(*[B_now[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])
            val_loss_now = eval_ntp_loss(base, val_batches, A_matrices, domain_id, B_now)
            del B_now
            gc.disable()

            if val_loss_now < best_val_loss:
                best_val_loss = val_loss_now
                consecutive_gl_exceeded = 0
            else:
                gl = 100.0 * (val_loss_now / best_val_loss - 1.0)
                if gl > GL_THRESHOLD:
                    consecutive_gl_exceeded += 1
                    if consecutive_gl_exceeded >= PATIENCE:
                        early_stop_triggered = True
                        stopping_step = step + 1
                        log(f"    Early stop at step {stopping_step}: GL={gl:.2f}")
                        break
                else:
                    consecutive_gl_exceeded = 0

    gc.enable()
    cleanup(optimizer)

    # Generate B from context, save
    context_tokens = train_batches[0][None, :]
    hidden_states = base.get_hidden_states(context_tokens)
    B_matrices = m2p(hidden_states)
    mx.eval(*[B_matrices[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])
    np_dict = {f"{li}_{mi}": np.array(B_matrices[(li, mi)])
               for li in range(N_LAYERS) for mi in range(N_MODULES)}
    np.savez(str(save_path), **np_dict)

    cleanup(m2p)

    B_loaded = load_B(str(save_path))
    m2p_val_loss = eval_ntp_loss(base, val_batches, A_matrices, domain_id, B_loaded)
    cleanup(B_loaded)

    train_val_gap = abs(final_train_loss - m2p_val_loss) if final_train_loss else None

    # Parity guard
    quality_ratio = 0.0
    excluded = False
    gap = base_loss - sft_loss
    if gap > PARITY_GUARD_THRESHOLD:
        quality_ratio = (base_loss - m2p_val_loss) / gap
    else:
        excluded = True
        log(f"    {domain_name}: EXCLUDED (gap={gap:.4f} < {PARITY_GUARD_THRESHOLD})")

    if not excluded:
        log(f"    {domain_name}: val_loss={m2p_val_loss:.4f} "
            f"SFT={sft_loss:.4f} base={base_loss:.4f} "
            f"quality={quality_ratio:.1%} stopped_at={stopping_step}")

    return {
        "m2p_val_loss": round(m2p_val_loss, 4),
        "final_train_loss": round(final_train_loss, 4) if final_train_loss else None,
        "train_val_gap": round(train_val_gap, 4) if train_val_gap else None,
        "quality_ratio": round(quality_ratio, 4),
        "excluded": excluded,
        "early_stop_triggered": early_stop_triggered,
        "stopping_step": stopping_step,
    }

def phase_m2p_all_domains(domain_data, base, A_matrices, base_losses, sft_results):
    log("\n=== Phase: Per-Domain M2P ===")
    results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        results[name] = phase_train_m2p(
            name, di, domain_data[name], base, A_matrices,
            base_losses[name], sft_results[name]["sft_loss"]
        )
        log_memory(f"M2P {name}")
    return results

def phase_cross_domain_m2p(src_name, src_domain_id, tgt_name, tgt_domain_id,
                            cross_slot_id, domain_data, base, A_matrices,
                            base_losses, sft_results) -> dict:
    log(f"  Cross {src_name}->{tgt_name} (slot={cross_slot_id})...")
    mx.random.seed(SEED + cross_slot_id * 100)
    m2p_cross = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P, m2p_layers=M2P_LAYERS)
    mx.eval(m2p_cross.parameters())
    optimizer = opt.Adam(learning_rate=M2P_LR)

    def _loss(m, ctx, tgt):
        return cross_m2p_loss(m, base, A_matrices, cross_slot_id, ctx[None, :], tgt[None, :])

    grad_fn = nn.value_and_grad(m2p_cross, _loss)
    src_train = domain_data[src_name]["train"]
    tgt_train = domain_data[tgt_name]["train"]
    tgt_val = domain_data[tgt_name]["val"]
    gc.disable()
    for step in range(CROSS_STEPS):
        loss, grads = grad_fn(m2p_cross, src_train[step % len(src_train)], tgt_train[step % len(tgt_train)])
        optimizer.update(m2p_cross, grads)
        mx.eval(m2p_cross.parameters(), optimizer.state, loss)
    gc.enable()
    cleanup(optimizer)

    ctx_tokens = src_train[0][None, :]
    hidden_states = base.get_hidden_states(ctx_tokens)
    B_cross = m2p_cross(hidden_states)
    mx.eval(*[B_cross[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])

    cross_val_loss = eval_ntp_loss(base, tgt_val, A_matrices, cross_slot_id, B_cross)
    del B_cross
    cleanup(m2p_cross)

    # Quality ratio: how much of tgt domain gap does cross-domain M2P close?
    tgt_base_loss = base_losses[tgt_name]
    tgt_sft_loss = sft_results[tgt_name]["sft_loss"]
    gap = tgt_base_loss - tgt_sft_loss
    cross_quality = 0.0
    if gap > PARITY_GUARD_THRESHOLD:
        cross_quality = (tgt_base_loss - cross_val_loss) / gap

    log(f"    {src_name}->{tgt_name}: cross_val_loss={cross_val_loss:.4f} "
        f"tgt_base={tgt_base_loss:.4f} tgt_sft={tgt_sft_loss:.4f} "
        f"cross_quality={cross_quality:.1%}")

    return {
        "src": src_name, "tgt": tgt_name,
        "cross_val_loss": round(cross_val_loss, 4),
        "cross_quality_ratio": round(cross_quality, 4),
    }

def phase_cross_domain_all(domain_data, base, A_matrices, base_losses,
                            sft_results, cross_slot_offset):
    log("\n=== Phase: Cross-Domain M2P ===")
    cross_results = {}
    for idx, (src_id, tgt_id) in enumerate(CROSS_PAIRS):
        src_name, tgt_name = DOMAIN_NAMES[src_id], DOMAIN_NAMES[tgt_id]
        pair_key = f"{src_name}_to_{tgt_name}"
        cross_results[pair_key] = phase_cross_domain_m2p(
            src_name, src_id, tgt_name, tgt_id,
            cross_slot_offset + idx, domain_data, base, A_matrices, base_losses, sft_results
        )
        log_memory(pair_key)
    return cross_results

def evaluate_kill_criteria(m2p_results, cross_results) -> dict:
    log("\n=== Kill Criteria ===")
    K3DOM_THRESHOLD = 0.85
    K_DIV_THRESHOLD = 0.50
    REFERENCE_QUALITY = 1.01
    K_REP_PP = 0.05

    # K_3dom
    domain_qualities = {
        name: (m2p_results[name]["quality_ratio"]
               if not m2p_results[name].get("excluded") else None)
        for name in ["sort", "reverse", "cipher"]
    }
    k3dom_values = [v for v in domain_qualities.values() if v is not None]
    k3dom_pass = len(k3dom_values) == 3 and all(v >= K3DOM_THRESHOLD for v in k3dom_values)

    # K_diversity
    cross_qualities = {k: v["cross_quality_ratio"] for k, v in cross_results.items()}
    k_diversity_pass = all(v < K_DIV_THRESHOLD for v in cross_qualities.values())
    max_cross_quality = max(cross_qualities.values()) if cross_qualities else None

    # K_replication
    rep_domains = {name: m2p_results[name]["quality_ratio"]
                   for name in ["sort", "reverse"]
                   if not m2p_results[name].get("excluded")}
    k_replication_pass = bool(rep_domains) and all(
        abs(v - REFERENCE_QUALITY) <= K_REP_PP for v in rep_domains.values()
    )

    cipher_quality = domain_qualities.get("cipher")
    cipher_in_range = cipher_quality is not None and 0.85 <= cipher_quality <= 1.0

    cq = f"{cipher_quality:.1%}" if cipher_quality is not None else "N/A"
    mcq = f"{max_cross_quality:.1%}" if max_cross_quality is not None else "N/A"
    if k3dom_pass and k_diversity_pass and k_replication_pass:
        outcome, interpretation = "FULL_PASS", f"Cipher={cq}>=85%, cross={mcq}<50%, rep OK."
    elif k3dom_pass and not k_diversity_pass:
        outcome, interpretation = "PARTIAL_no_diversity", f"Cipher={cq} OK but cross={mcq}>=50%."
    elif not k3dom_pass and k_diversity_pass:
        outcome, interpretation = "PARTIAL_diversity_ok", f"{domain_qualities} but cross={mcq}<50%."
    else:
        outcome, interpretation = "MULTIPLE_FAIL", f"3dom={k3dom_pass} div={k_diversity_pass} rep={k_replication_pass}"

    log(f"  K_3dom: {domain_qualities} -> {'PASS' if k3dom_pass else 'FAIL'}")
    log(f"  K_diversity: {cross_qualities} -> {'PASS' if k_diversity_pass else 'FAIL'}")
    log(f"  K_replication: {rep_domains} -> {'PASS' if k_replication_pass else 'FAIL'}")
    log(f"  Outcome: {outcome} | {interpretation}")

    return {
        "k3dom_pass": bool(k3dom_pass),
        "k_diversity_pass": bool(k_diversity_pass),
        "k_replication_pass": bool(k_replication_pass),
        "outcome": outcome,
        "interpretation": interpretation,
        "domain_qualities": {k: round(float(v), 4) if v is not None else None
                             for k, v in domain_qualities.items()},
        "cross_qualities": {k: round(float(v), 4) for k, v in cross_qualities.items()},
        "cipher_quality": round(float(cipher_quality), 4) if cipher_quality is not None else None,
        "cipher_in_predicted_range": bool(cipher_in_range),
        "max_cross_quality": round(float(max_cross_quality), 4) if max_cross_quality is not None else None,
    }

# MAIN ORCHESTRATOR

def main():
    t0 = time.time()
    log(f"M2P Third Domain | SMOKE={SMOKE_TEST} | domains={DOMAIN_NAMES} | n={N_SAMPLES} T={T_FIXED}")
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    domain_data = phase_generate_data(rng, n_per_domain=N_SAMPLES)
    log_memory("after data")

    base, base_losses = phase_pretrain_base(domain_data)
    log_memory("after base")

    n_total_slots = N_DOMAINS + len(CROSS_PAIRS)
    A_matrices, ortho = phase_grassmannian(n_total_slots)
    log_memory("after grassmannian")

    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after SFT")

    m2p_results = phase_m2p_all_domains(domain_data, base, A_matrices, base_losses, sft_results)
    log_memory("after M2P")

    cross_results = phase_cross_domain_all(
        domain_data, base, A_matrices, base_losses, sft_results,
        cross_slot_offset=N_DOMAINS
    )
    log_memory("after cross-domain")

    kill_criteria = evaluate_kill_criteria(m2p_results, cross_results)

    total_time = round(time.time() - t0, 1)
    results = {
        "experiment": "m2p_third_domain",
        "total_time_s": total_time,
        "smoke_test": SMOKE_TEST,
        "config": {
            "d_model": D_MODEL, "n_domains": N_DOMAINS, "domain_names": DOMAIN_NAMES,
            "m2p_layers": M2P_LAYERS, "d_m2p": D_M2P, "n_memory": N_MEMORY,
            "n_samples": N_SAMPLES, "t_fixed": T_FIXED,
        },
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        "grassmannian_max_cos": ortho["max_cos"],
        "per_domain_m2p": {
            name: {
                "quality_ratio": m2p_results[name]["quality_ratio"],
                "m2p_val_loss": m2p_results[name]["m2p_val_loss"],
                "sft_loss": sft_results[name]["sft_loss"],
                "base_loss": base_losses[name],
                "early_stop_triggered": m2p_results[name]["early_stop_triggered"],
                "stopping_step": m2p_results[name]["stopping_step"],
                "train_val_gap": m2p_results[name]["train_val_gap"],
                "excluded": m2p_results[name]["excluded"],
            }
            for name in DOMAIN_NAMES
        },
        "cross_domain_m2p": cross_results,
        "kill_criteria": kill_criteria,
        "predictions_vs_measurements": {
            name: {
                "predicted": "85-100%" if name == "cipher" else "98-102%",
                "measured": f"{m2p_results[name]['quality_ratio']:.3f}",
            }
            for name in ["cipher", "sort", "reverse"]
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nDONE in {total_time}s | {RESULTS_FILE}")
    log(f"K_3dom={'PASS' if kill_criteria['k3dom_pass'] else 'FAIL'} "
        f"K_div={'PASS' if kill_criteria['k_diversity_pass'] else 'FAIL'} "
        f"K_rep={'PASS' if kill_criteria['k_replication_pass'] else 'FAIL'} "
        f"-> {kill_criteria['outcome']}")

if __name__ == "__main__":
    main()
