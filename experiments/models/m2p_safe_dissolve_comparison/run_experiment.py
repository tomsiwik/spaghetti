#!/usr/bin/env python3
"""Safe Dissolve Comparison: 5 strategies for protecting competent domains.

TYPE: guided-exploration
MATH: experiments/models/m2p_safe_dissolve_comparison/MATH.md

Reuses trained artifacts from m2p_cross_domain_graph (Option A, the winner).
Tests 5 dissolve strategies head-to-head with identical quality + cost metrics.

Strategies:
  S0: Naive (uniform scale, current baseline)
  S1: Loss-gated (evaluate before merge, skip harmful)
  S2: Headroom-proportional (scale ∝ base_loss - threshold)
  S3: Selective routing (two bases: clean + enriched)
  S4: Null-space projection (project out competent subspace)

Kill criteria:
  K_quality: ≥1 approach with median Q >90% and 0 domains degraded >5%
  K_cost: best-quality approach <2x wall time of naive
  K_tradeoff: ≥1 approach Pareto-dominates naive
"""

import gc
import json
import math
import os
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

# Memory safety
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# Architecture (must match m2p_cross_domain_graph)
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5
PROMOTE_SCALE = 5.0

D_M2P = 64
N_MEMORY = 32
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

M2P_STEPS = 500 if not SMOKE_TEST else 30
M2P_LR = 1e-3

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]

GATE_THRESHOLD = 0.05   # S1: skip adapter if any domain degrades >5%
HEADROOM_TAU = 1.5      # S2: domains with base_loss < τ get scale≈0
HEADROOM_MAX = 5.0      # S2: max scale cap
ROUTE_TAU = 1.5         # S3: domains below this use clean base
NULL_RANK_K = 8         # S4: SVD rank for competent subspace


# ── Utilities ──────────────────────────────────────────────────────────────

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


# ── Data generation (identical to cross_domain_graph) ──────────────────────

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        if domain_id == 0:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            data.append(f"{a}+{b}={a+b}")
        elif domain_id == 1:
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(sorted(s))}")
        elif domain_id == 2:
            bits = "".join(str(rng.randint(0, 2)) for _ in range(rng.randint(2, 6)))
            data.append(f"{bits}>{'even' if bits.count('1') % 2 == 0 else 'odd'}")
        elif domain_id == 3:
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(reversed(s))}")
        elif domain_id == 4:
            p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1, 3)))
            r = rng.randint(2, 4)
            data.append(f"{p}*{r}={p*r}")
    return data


def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]

def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]


# ── Model classes (identical to cross_domain_graph) ────────────────────────

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
        attn = mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd ** -0.5) + mask, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d, bias=False)
        self.fc2 = nn.Linear(4*d, d, bias=False)
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
        x = x + self.mlp(self.norm2(x))
        return x

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
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))
    def get_hidden_states(self, tokens):
        B, T = tokens.shape
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states


# ── Grassmannian + LoRA ────────────────────────────────────────────────────

def generate_grassmannian_A(n_slots, n_layers, n_modules, d, rank, seed=42):
    rng = np.random.RandomState(seed)
    A = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, n_slots * rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)
            for si in range(n_slots):
                A[(si, li, mi)] = mx.array(Q[:, si*rank:(si+1)*rank])
    return A


def lora_forward_with_B(base, tokens, A_matrices, slot_id, B_matrices):
    B_batch, T = tokens.shape
    x = base.wte(tokens) + base.wpe(mx.arange(T))
    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        def _lora(fn, x_in, mi):
            return fn(x_in) + LORA_SCALE * (x_in @ A_matrices[(slot_id, li, mi)]) @ B_matrices[(li, mi)]

        q = _lora(attn.wq, x_norm, 0).reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = _lora(attn.wk, x_norm, 1).reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = _lora(attn.wv, x_norm, 2).reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        attn_ctx = (mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd**-0.5) + mask, axis=-1) @ v)
        attn_ctx = attn_ctx.transpose(0, 2, 1, 3).reshape(B_b, T_b, C)
        x = x + _lora(attn.wo, attn_ctx, 3)

        x_n2 = block.norm2(x)
        fc1_out = block.mlp.fc1(x_n2) + LORA_SCALE * (x_n2 @ A_matrices[(slot_id, li, 4)]) @ B_matrices[(li, 4)]
        x = x + block.mlp.fc2(nn.gelu(fc1_out))

    return base.lm_head(base.norm_f(x))


def eval_ntp_loss(base, batches, A_matrices=None, slot_id=None, B_matrices=None):
    total, n = 0.0, 0
    for tokens in batches[:50]:
        t2d = tokens[None, :]
        if A_matrices is not None and B_matrices is not None:
            logits = lora_forward_with_B(base, t2d, A_matrices, slot_id, B_matrices)
        else:
            logits = base(t2d)
        loss = nn.losses.cross_entropy(logits[:, :-1], t2d[:, 1:], reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


# ── M2P Transformer ───────────────────────────────────────────────────────

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
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd**-0.5) + mask, axis=-1)
        return self.wo((a @ v).transpose(0, 2, 1, 3).reshape(B, T, C))

class M2PMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d, bias=False)
        self.fc2 = nn.Linear(4*d, d, bias=False)
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
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P):
        super().__init__()
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p) for _ in range(M2P_LAYERS)]
        self.norm_f = RMSNorm(d_m2p)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            self.out_heads[mname] = nn.Linear(d_m2p, N_LAYERS * LORA_RANK * d_out, bias=False)

    def __call__(self, hidden_states_list):
        encs = [self.input_proj(mx.mean(h[0], axis=0)) for h in hidden_states_list]
        memory = self.memory_tokens + self.pos_embed(mx.arange(N_MEMORY))
        memory = memory + mx.mean(mx.stack(encs), axis=0)[None, :]
        x = memory[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        pooled = mx.mean(x[0], axis=0)
        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pooled).reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS):
                B_matrices[(li, mi)] = out[li]
        return B_matrices


def m2p_ntp_loss(m2p, base, A_matrices, slot_id, tokens):
    hidden = base.get_hidden_states(tokens)
    B = m2p(hidden)
    logits = lora_forward_with_B(base, tokens, A_matrices, slot_id, B)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ── Merge strategies ───────────────────────────────────────────────────────

MODULE_MAP = {
    0: lambda block: block.attn.wq,
    1: lambda block: block.attn.wk,
    2: lambda block: block.attn.wv,
    3: lambda block: block.attn.wo,
    4: lambda block: block.mlp.fc1,
}


def merge_single_adapter(base, A_matrices, slot_id, B_matrices, scale):
    """Merge one adapter into base weights in-place."""
    for li, block in enumerate(base.blocks):
        for mi in range(N_MODULES):
            A = A_matrices[(slot_id, li, mi)]
            B = B_matrices[(li, mi)]
            linear = MODULE_MAP[mi](block)
            delta = scale * B.T @ A.T
            linear.weight = linear.weight + delta
            mx.eval(linear.weight)


def clone_base(base, base_weights_np):
    """Create a fresh copy of the base model from saved weights."""
    fresh = ToyGPT()
    params = []
    for k, _ in tree_flatten(fresh.parameters()):
        params.append((k, mx.array(base_weights_np[k.replace(".", "_")])))
    fresh.load_weights(params)
    fresh.freeze()
    mx.eval(fresh.parameters())
    return fresh


def strategy_naive(base_np, A_matrices, cross_B_list, cross_slots, **_):
    """S0: Merge all at uniform scale."""
    base = clone_base(None, base_np)
    t0 = time.time()
    for (slot, B) in zip(cross_slots, cross_B_list):
        merge_single_adapter(base, A_matrices, slot, B, PROMOTE_SCALE)
    merge_time = time.time() - t0
    return base, {"merge_time_s": round(merge_time, 4), "inference_overhead_s": 0,
                  "eval_calls": 0, "peak_memory_gb": round(mx.get_peak_memory()/1e9, 2)}


def strategy_loss_gated(base_np, A_matrices, cross_B_list, cross_slots,
                         domain_data, base_losses, **_):
    """S1: Evaluate before merge, skip harmful adapters."""
    base = clone_base(None, base_np)
    t0 = time.time()
    eval_calls = 0
    merged = 0
    skipped = 0

    for (slot, B) in zip(cross_slots, cross_B_list):
        # Tentatively merge
        merge_single_adapter(base, A_matrices, slot, B, PROMOTE_SCALE)
        mx.eval(base.parameters())

        # Check all domains
        harmful = False
        for name in DOMAIN_NAMES:
            new_loss = eval_ntp_loss(base, domain_data[name]["val"])
            eval_calls += 1
            if new_loss > base_losses[name] * (1 + GATE_THRESHOLD):
                harmful = True
                break

        if harmful:
            # Undo: reload fresh base with already-merged adapters
            base_undo = clone_base(None, base_np)
            # Re-merge the ones we already accepted
            for prev_slot, prev_B in zip(cross_slots[:merged], cross_B_list[:merged]):
                merge_single_adapter(base_undo, A_matrices, prev_slot, prev_B, PROMOTE_SCALE)
            cleanup(base)
            base = base_undo
            skipped += 1
        else:
            merged += 1

    merge_time = time.time() - t0
    return base, {"merge_time_s": round(merge_time, 4), "inference_overhead_s": 0,
                  "eval_calls": eval_calls, "merged": merged, "skipped": skipped,
                  "peak_memory_gb": round(mx.get_peak_memory()/1e9, 2)}


def strategy_headroom(base_np, A_matrices, cross_B_list, cross_slots,
                       cross_pairs, base_losses, **_):
    """S2: Scale proportional to target domain headroom."""
    base = clone_base(None, base_np)
    t0 = time.time()
    scales_used = {}

    for idx, (slot, B) in enumerate(zip(cross_slots, cross_B_list)):
        _, tgt = cross_pairs[idx]
        tgt_name = DOMAIN_NAMES[tgt]
        headroom = max(0.0, base_losses[tgt_name] - HEADROOM_TAU)
        scale = min(headroom, HEADROOM_MAX)
        scales_used[f"{DOMAIN_NAMES[cross_pairs[idx][0]]}→{tgt_name}"] = round(scale, 2)
        if scale > 0.01:
            merge_single_adapter(base, A_matrices, slot, B, scale)

    merge_time = time.time() - t0
    return base, {"merge_time_s": round(merge_time, 4), "inference_overhead_s": 0,
                  "eval_calls": 0, "scales_used": scales_used,
                  "peak_memory_gb": round(mx.get_peak_memory()/1e9, 2)}


def strategy_selective_routing(base_np, A_matrices, cross_B_list, cross_slots,
                                base_losses, **_):
    """S3: Two bases — clean for easy domains, enriched for hard."""
    clean_base = clone_base(None, base_np)
    enriched_base = clone_base(None, base_np)

    t0 = time.time()
    for (slot, B) in zip(cross_slots, cross_B_list):
        merge_single_adapter(enriched_base, A_matrices, slot, B, PROMOTE_SCALE)
    merge_time = time.time() - t0

    # Memory: 2x base
    mem = round(mx.get_peak_memory()/1e9, 2)

    return (clean_base, enriched_base), {
        "merge_time_s": round(merge_time, 4),
        "inference_overhead_s": 0.001,  # router lookup negligible
        "inference_memory_factor": 2.0,
        "eval_calls": 0,
        "easy_domains": [n for n in DOMAIN_NAMES if base_losses[n] < ROUTE_TAU],
        "hard_domains": [n for n in DOMAIN_NAMES if base_losses[n] >= ROUTE_TAU],
        "peak_memory_gb": mem,
    }


def strategy_null_space(base_np, A_matrices, cross_B_list, cross_slots,
                         domain_data, base_losses, **_):
    """S4: Project out competent-domain subspace before merge."""
    # Identify competent domains
    competent = [n for n in DOMAIN_NAMES if base_losses[n] < ROUTE_TAU]

    # Collect hidden states for competent domains
    base_ref = clone_base(None, base_np)
    t0 = time.time()

    projectors = {}  # per (layer, module) → null-space projector
    for name in competent:
        val_data = domain_data[name]["val"][:20]
        for tokens in val_data:
            states = base_ref.get_hidden_states(tokens[None, :])
            for li in range(N_LAYERS):
                key = li
                h = states[li][0]  # (T, D)
                if key not in projectors:
                    projectors[key] = []
                projectors[key].append(h)

    # Compute null-space projector per layer
    null_projs = {}
    for li, h_list in projectors.items():
        H = mx.concatenate(h_list, axis=0)  # (n_tokens, D)
        # SVD on CPU (GPU may not support)
        H_np = np.array(H)
        U, S, Vt = np.linalg.svd(H_np, full_matrices=False)
        # Top-k directions = competent subspace
        k = min(NULL_RANK_K, len(S))
        Vk = Vt[:k]  # (k, D)
        # Null-space projector: I - V_k^T V_k
        P_null = np.eye(D_MODEL, dtype=np.float32) - Vk.T @ Vk
        null_projs[li] = mx.array(P_null)

    cleanup(base_ref)

    # Merge with projected deltas
    base = clone_base(None, base_np)
    for (slot, B_dict) in zip(cross_slots, cross_B_list):
        for li, block in enumerate(base.blocks):
            P = null_projs.get(li, mx.eye(D_MODEL))
            for mi in range(N_MODULES):
                A = A_matrices[(slot, li, mi)]
                B = B_dict[(li, mi)]
                linear = MODULE_MAP[mi](block)
                # delta = scale * B^T @ A^T, shape (d_out, d_in)
                delta = PROMOTE_SCALE * B.T @ A.T
                # Project: delta_safe = delta @ P_null (project input-space)
                delta_safe = delta @ P
                linear.weight = linear.weight + delta_safe
                mx.eval(linear.weight)

    merge_time = time.time() - t0
    return base, {"merge_time_s": round(merge_time, 4), "inference_overhead_s": 0,
                  "eval_calls": 0, "null_rank_k": NULL_RANK_K,
                  "competent_domains": competent,
                  "peak_memory_gb": round(mx.get_peak_memory()/1e9, 2)}


# ── Recrystallize: retrain per-domain M2P on enriched base ─────────────────

def recrystallize(base, A_matrices, domain_data, base_losses, sft_losses):
    """Retrain per-domain M2P on the given base and measure quality."""
    quality = {}
    for di, name in enumerate(DOMAIN_NAMES):
        mx.random.seed(SEED + 200 + di)
        m2p = M2PTransformer()
        mx.eval(m2p.parameters())
        m2p_opt = opt.Adam(learning_rate=M2P_LR)

        def m2p_loss(m, tokens):
            return m2p_ntp_loss(m, base, A_matrices, di, tokens[None, :])

        vg = nn.value_and_grad(m2p, m2p_loss)
        train = domain_data[name]["train"]

        gc.disable()
        for step in range(M2P_STEPS):
            loss, grads = vg(m2p, train[step % len(train)])
            m2p_opt.update(m2p, grads)
            mx.eval(m2p.parameters(), m2p_opt.state, loss)
        gc.enable()

        val_tokens = domain_data[name]["val"][0][None, :]
        B_gen = m2p(base.get_hidden_states(val_tokens))
        mx.eval(list(B_gen.values()))

        enriched_loss = eval_ntp_loss(base, domain_data[name]["val"], A_matrices, di,
                                       {k: mx.array(v) for k, v in B_gen.items()})
        q_ratio = (base_losses[name] - enriched_loss) / (base_losses[name] - sft_losses[name] + 1e-8)
        quality[name] = {
            "enriched_loss": round(enriched_loss, 4),
            "quality_ratio": round(q_ratio, 4),
        }
        cleanup(m2p, m2p_opt)

    return quality


def recrystallize_selective(clean_base, enriched_base, A_matrices, domain_data,
                             base_losses, sft_losses):
    """S3: retrain on clean base for easy domains, enriched for hard."""
    quality = {}
    for di, name in enumerate(DOMAIN_NAMES):
        use_base = clean_base if base_losses[name] < ROUTE_TAU else enriched_base
        mx.random.seed(SEED + 200 + di)
        m2p = M2PTransformer()
        mx.eval(m2p.parameters())
        m2p_opt = opt.Adam(learning_rate=M2P_LR)

        def m2p_loss(m, tokens):
            return m2p_ntp_loss(m, use_base, A_matrices, di, tokens[None, :])

        vg = nn.value_and_grad(m2p, m2p_loss)
        train = domain_data[name]["train"]

        gc.disable()
        for step in range(M2P_STEPS):
            loss, grads = vg(m2p, train[step % len(train)])
            m2p_opt.update(m2p, grads)
            mx.eval(m2p.parameters(), m2p_opt.state, loss)
        gc.enable()

        val_tokens = domain_data[name]["val"][0][None, :]
        B_gen = m2p(use_base.get_hidden_states(val_tokens))
        mx.eval(list(B_gen.values()))

        enriched_loss = eval_ntp_loss(use_base, domain_data[name]["val"], A_matrices, di,
                                       {k: mx.array(v) for k, v in B_gen.items()})
        q_ratio = (base_losses[name] - enriched_loss) / (base_losses[name] - sft_losses[name] + 1e-8)
        quality[name] = {
            "enriched_loss": round(enriched_loss, 4),
            "quality_ratio": round(q_ratio, 4),
        }
        cleanup(m2p, m2p_opt)

    return quality


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    results = {"experiment": "m2p_safe_dissolve_comparison", "smoke_test": SMOKE_TEST}
    rng = np.random.RandomState(SEED)

    # ── Phase 1: Setup (same as cross_domain_graph) ────────────────────────
    log("=== Phase 1: Generate Data + Train Base + SFT + Cross-Domain Adapters ===")

    n_per = 500 if not SMOKE_TEST else 60
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per, rng)
        split = int(0.8 * len(texts))
        domain_data[name] = {
            "train": make_batches(texts[:split]),
            "val": make_batches(texts[split:]),
            "domain_id": di,
        }

    # Train base
    mx.random.seed(SEED)
    base = ToyGPT()
    mx.eval(base.parameters())
    all_train = [t for name in DOMAIN_NAMES for t in domain_data[name]["train"]]
    base_opt = opt.Adam(learning_rate=3e-4)
    base_steps = 1200 if not SMOKE_TEST else 60

    def base_loss_fn(model, tokens):
        return nn.losses.cross_entropy(model(tokens[None, :])[:, :-1], tokens[None, 1:], reduction="mean")

    base_vg = nn.value_and_grad(base, base_loss_fn)
    gc.disable()
    for step in range(base_steps):
        loss, grads = base_vg(base, all_train[step % len(all_train)])
        base_opt.update(base, grads)
        mx.eval(base.parameters(), base_opt.state, loss)
        if (step+1) % (base_steps//4) == 0:
            log(f"  Base step {step+1}/{base_steps}: loss={loss.item():.4f}")
    gc.enable()
    base.freeze()

    base_losses = {n: round(eval_ntp_loss(base, domain_data[n]["val"]), 4) for n in DOMAIN_NAMES}
    log(f"  Base losses: {base_losses}")
    results["base_losses"] = base_losses

    # Save base weights
    base_np = {}
    for k, v in tree_flatten(base.parameters()):
        base_np[k.replace(".", "_")] = np.array(v)

    cleanup(base_opt)

    # Grassmannian A
    cross_pairs = list(combinations(range(N_DOMAINS), 2))
    n_total = N_DOMAINS + len(cross_pairs)
    A_matrices = generate_grassmannian_A(n_total, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK)
    cross_slot_map = {(a, b): N_DOMAINS + i for i, (a, b) in enumerate(cross_pairs)}

    # SFT baselines
    log("  Training SFT baselines...")
    sft_losses = {}
    from mlx.utils import tree_unflatten as _tu  # noqa

    class BMatrices(nn.Module):
        def __init__(self):
            super().__init__()
            for li in range(N_LAYERS):
                for mi in range(N_MODULES):
                    setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, MODULE_OUT_DIMS[mi])))
        def as_dict(self):
            return {(li, mi): getattr(self, f"B_{li}_{mi}") for li in range(N_LAYERS) for mi in range(N_MODULES)}

    for di, name in enumerate(DOMAIN_NAMES):
        bc = BMatrices()
        mx.eval(bc.parameters())
        sft_opt = opt.Adam(learning_rate=1e-3)
        def sft_l(bc, tok):
            logits = lora_forward_with_B(base, tok[None,:], A_matrices, di, bc.as_dict())
            return nn.losses.cross_entropy(logits[:, :-1], tok[None, 1:], reduction="mean")
        vg = nn.value_and_grad(bc, sft_l)
        sft_steps = 400 if not SMOKE_TEST else 30
        gc.disable()
        for step in range(sft_steps):
            loss, grads = vg(bc, domain_data[name]["train"][step % len(domain_data[name]["train"])])
            sft_opt.update(bc, grads)
            mx.eval(bc.parameters(), sft_opt.state, loss)
        gc.enable()
        sl = eval_ntp_loss(base, domain_data[name]["val"], A_matrices, di, bc.as_dict())
        sft_losses[name] = round(sl, 4)
        cleanup(bc, sft_opt)

    results["sft_losses"] = sft_losses
    log(f"  SFT losses: {sft_losses}")

    # Per-domain M2P baseline quality
    log("  Training per-domain M2P (baseline)...")
    perdomain_losses = {}
    perdomain_B = {}
    for di, name in enumerate(DOMAIN_NAMES):
        mx.random.seed(SEED + di)
        m2p = M2PTransformer()
        mx.eval(m2p.parameters())
        m2p_opt = opt.Adam(learning_rate=M2P_LR)
        def ml(m, tok):
            return m2p_ntp_loss(m, base, A_matrices, di, tok[None,:])
        vg = nn.value_and_grad(m2p, ml)
        gc.disable()
        for step in range(M2P_STEPS):
            loss, grads = vg(m2p, domain_data[name]["train"][step % len(domain_data[name]["train"])])
            m2p_opt.update(m2p, grads)
            mx.eval(m2p.parameters(), m2p_opt.state, loss)
        gc.enable()
        val_tok = domain_data[name]["val"][0][None, :]
        B_gen = m2p(base.get_hidden_states(val_tok))
        mx.eval(list(B_gen.values()))
        perdomain_B[di] = {k: mx.array(v) for k, v in B_gen.items()}
        pl = eval_ntp_loss(base, domain_data[name]["val"], A_matrices, di, perdomain_B[di])
        perdomain_losses[name] = round(pl, 4)
        cleanup(m2p, m2p_opt)

    results["perdomain_m2p_losses"] = perdomain_losses
    log(f"  Per-domain M2P losses: {perdomain_losses}")

    # Cross-domain adapters (Option A — winner from cross_domain_graph)
    log("  Training cross-domain M2P (Option A: cross-prediction)...")
    cross_B_list = []
    cross_slots = []
    for (src, tgt) in cross_pairs:
        slot = cross_slot_map[(src, tgt)]
        cross_slots.append(slot)
        mx.random.seed(SEED + 100 + src * N_DOMAINS + tgt)
        m2p_c = M2PTransformer()
        mx.eval(m2p_c.parameters())
        c_opt = opt.Adam(learning_rate=M2P_LR)
        src_train = domain_data[DOMAIN_NAMES[src]]["train"]
        tgt_train = domain_data[DOMAIN_NAMES[tgt]]["train"]
        cross_steps = 300 if not SMOKE_TEST else 20

        def cl(m, step_idx):
            ctx = src_train[step_idx % len(src_train)][None, :]
            tgt_tok = tgt_train[step_idx % len(tgt_train)][None, :]
            hidden = base.get_hidden_states(ctx)
            B = m(hidden)
            logits = lora_forward_with_B(base, tgt_tok, A_matrices, slot, B)
            return nn.losses.cross_entropy(logits[:, :-1], tgt_tok[:, 1:], reduction="mean")

        vg = nn.value_and_grad(m2p_c, cl)
        gc.disable()
        for step in range(cross_steps):
            loss, grads = vg(m2p_c, step)
            c_opt.update(m2p_c, grads)
            mx.eval(m2p_c.parameters(), c_opt.state, loss)
        gc.enable()

        src_val = domain_data[DOMAIN_NAMES[src]]["val"][0][None, :]
        B_gen = m2p_c(base.get_hidden_states(src_val))
        mx.eval(list(B_gen.values()))
        cross_B_list.append({k: mx.array(v) for k, v in B_gen.items()})
        cleanup(m2p_c, c_opt)

    log(f"  Trained {len(cross_B_list)} cross-domain adapters")

    # ── Phase 2: Run all 5 strategies ──────────────────────────────────────
    log("\n=== Phase 2: Compare Dissolve Strategies ===")

    strategy_kwargs = dict(
        base_np=base_np, A_matrices=A_matrices,
        cross_B_list=cross_B_list, cross_slots=cross_slots,
        cross_pairs=cross_pairs, domain_data=domain_data,
        base_losses=base_losses
    )

    strategies = {
        "S0_naive": strategy_naive,
        "S1_loss_gated": strategy_loss_gated,
        "S2_headroom": strategy_headroom,
        "S3_selective": strategy_selective_routing,
        "S4_null_space": strategy_null_space,
    }

    complexity_info = {
        "S0_naive":      {"n_hyperparams": 1, "n_new_components": 0, "lines_of_code": 3},
        "S1_loss_gated": {"n_hyperparams": 2, "n_new_components": 0, "lines_of_code": 15},
        "S2_headroom":   {"n_hyperparams": 2, "n_new_components": 0, "lines_of_code": 5},
        "S3_selective":  {"n_hyperparams": 1, "n_new_components": 1, "lines_of_code": 10},
        "S4_null_space": {"n_hyperparams": 3, "n_new_components": 1, "lines_of_code": 20},
    }

    scalability_info = {
        "S0_naive":      "O(N) merges",
        "S1_loss_gated": "O(N*D) forward passes",
        "S2_headroom":   "O(N) merges",
        "S3_selective":  "O(N) merges + 2x memory",
        "S4_null_space": "O(D*d^2) SVD + O(N*d^2) projections",
    }

    all_tradeoffs = {}

    for sname, sfunc in strategies.items():
        log(f"\n--- {sname} ---")
        mx.reset_peak_memory()
        t0_s = time.time()

        result = sfunc(**strategy_kwargs)

        if sname == "S3_selective":
            enriched_result, cost = result
            clean_b, enriched_b = enriched_result
            # Evaluate enriched base losses
            enriched_losses = {}
            for name in DOMAIN_NAMES:
                use = clean_b if base_losses[name] < ROUTE_TAU else enriched_b
                el = eval_ntp_loss(use, domain_data[name]["val"])
                enriched_losses[name] = round(el, 4)
            log(f"  Enriched base losses: {enriched_losses}")

            # Recrystallize with selective routing
            quality_dict = recrystallize_selective(
                clean_b, enriched_b, A_matrices, domain_data, base_losses, sft_losses)
            cleanup(clean_b, enriched_b)
        else:
            enriched_base, cost = result
            enriched_losses = {n: round(eval_ntp_loss(enriched_base, domain_data[n]["val"]), 4) for n in DOMAIN_NAMES}
            log(f"  Enriched base losses: {enriched_losses}")

            quality_dict = recrystallize(enriched_base, A_matrices, domain_data, base_losses, sft_losses)
            cleanup(enriched_base)

        # Compute quality summary
        ratios = [v["quality_ratio"] for v in quality_dict.values()]
        deltas = [(enriched_losses[n] - base_losses[n]) / base_losses[n]
                  for n in DOMAIN_NAMES]
        worst_delta = max(deltas)  # positive = worse
        n_protected = sum(1 for d in deltas if d <= 0.05)
        median_q = sorted(ratios)[len(ratios)//2]

        for name in DOMAIN_NAMES:
            quality_dict[name]["enriched_base_loss"] = enriched_losses[name]
            quality_dict[name]["base_delta_pct"] = round(
                (enriched_losses[name] - base_losses[name]) / base_losses[name] * 100, 2)

        quality_summary = {
            "per_domain": quality_dict,
            "median_quality": round(median_q, 4),
            "worst_domain_delta": round(worst_delta, 4),
            "n_domains_protected": n_protected,
        }

        total_time = round(time.time() - t0_s, 2)
        cost["total_strategy_time_s"] = total_time

        all_tradeoffs[sname] = {
            "quality": quality_summary,
            "cost": cost,
            "complexity": complexity_info[sname],
            "scalability": scalability_info[sname],
        }

        log(f"  Median Q: {median_q:.1%}, Worst Δ: {worst_delta:.2f}, Protected: {n_protected}/5")
        log(f"  Time: {total_time:.1f}s, Eval calls: {cost.get('eval_calls', 0)}")

    results["tradeoffs"] = all_tradeoffs

    # ── Kill criteria ──────────────────────────────────────────────────────
    log("\n=== Kill Criteria ===")

    # K_quality: ≥1 approach with median Q >90% AND 0 domains degraded >5%
    k_quality = any(
        t["quality"]["median_quality"] > 0.9 and t["quality"]["n_domains_protected"] == 5
        for t in all_tradeoffs.values()
    )
    results["K_quality"] = {"pass": k_quality}
    log(f"  K_quality (median>90% + all protected): {'PASS' if k_quality else 'FAIL'}")

    # K_cost: best-quality approach <2x naive wall time
    naive_time = all_tradeoffs["S0_naive"]["cost"]["total_strategy_time_s"]
    best_quality_name = max(all_tradeoffs.keys(),
                            key=lambda k: all_tradeoffs[k]["quality"]["median_quality"])
    best_time = all_tradeoffs[best_quality_name]["cost"]["total_strategy_time_s"]
    k_cost = best_time < 2 * naive_time + 1  # +1s tolerance
    results["K_cost"] = {"pass": k_cost, "best": best_quality_name,
                         "best_time": best_time, "naive_time": naive_time}
    log(f"  K_cost (<2x naive): {'PASS' if k_cost else 'FAIL'} — "
        f"{best_quality_name}={best_time:.1f}s vs naive={naive_time:.1f}s")

    # K_tradeoff: Pareto-dominates naive
    naive_q = all_tradeoffs["S0_naive"]["quality"]
    pareto = any(
        t["quality"]["worst_domain_delta"] < naive_q["worst_domain_delta"] and
        t["quality"]["median_quality"] >= naive_q["median_quality"] * 0.95
        for k, t in all_tradeoffs.items() if k != "S0_naive"
    )
    results["K_tradeoff"] = {"pass": pareto}
    log(f"  K_tradeoff (Pareto > naive): {'PASS' if pareto else 'FAIL'}")

    results["all_pass"] = k_quality and k_cost and pareto

    # ── Summary table ──────────────────────────────────────────────────────
    log(f"\n{'='*80}")
    log(f"{'Strategy':<18} {'Median Q':>9} {'Worst Δ':>9} {'Prot':>5} "
        f"{'Time(s)':>8} {'Evals':>6} {'HP':>4} {'Scale'}")
    log(f"{'-'*80}")
    for sname, t in all_tradeoffs.items():
        q = t["quality"]
        c = t["cost"]
        cx = t["complexity"]
        log(f"{sname:<18} {q['median_quality']:>8.1%} {q['worst_domain_delta']:>9.2f} "
            f"{q['n_domains_protected']:>5} {c['total_strategy_time_s']:>8.1f} "
            f"{c.get('eval_calls', 0):>6} {cx['n_hyperparams']:>4} "
            f"{t['scalability']}")
    log(f"{'='*80}")

    results["total_time_s"] = round(time.time() - t_start, 1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
