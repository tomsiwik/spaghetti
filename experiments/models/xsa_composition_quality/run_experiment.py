#!/usr/bin/env python3
"""
XSA (Exclusive Self-Attention) Composition Quality Experiment

Tests whether XSA improves adapter composition quality by removing self-value
bias from attention outputs via orthogonal projection.

Kill criteria:
  K1 (id=201): XSA degrades single-adapter quality > 3% PPL on any domain -> KILL
  K2 (id=202): XSA composition ratio >= no-XSA ratio (3-seed mean) -> KILL
  K3 (id=203): XSA+composition worse than no-XSA on >= 3/5 domains -> KILL

Platform: Apple M5 Pro 48GB, MLX, local.

Architecture: Micro transformer (d=128, H=4, L=4, FFN=512) with character-level
tokenizer. Ternary-quantized base. LoRA adapters with Grassmannian A (frozen)
and trainable B (ternary STE). Two conditions: standard vs XSA in last 2 layers.
"""

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten
import numpy as np

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Architecture
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
N_LAYERS = 4
FFN_HIDDEN = 512
MAX_SEQ = 32
LORA_RANK = 8
N_XSA_LAYERS = 2

# Training
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
BASE_EPOCHS = 30
ADAPTER_EPOCHS = 25
LR_BASE = 3e-3
LR_ADAPTER = 1e-2
BATCH_SIZE = 64

SEEDS = [42, 123, 456]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Synthetic Data
# ===========================================================================

class CharTokenizer:
    def __init__(self):
        chars = sorted(set("0123456789abcdefghijklmnopqrstuvwxyz>+=* "))
        specials = ["<PAD>", "<EOS>"]
        self.vocab = specials + chars
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.pad_id = 0
        self.eos_id = 1
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        return [self.char2idx.get(c, self.pad_id) for c in s] + [self.eos_id]


def make_arithmetic_data(n, rng):
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        data.append(f"{a}+{b}={a+b}")
    return data


def make_reverse_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{s[::-1]}")
    return data


def make_repeat_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 4)
        pat = "".join(rng.choice(list(chars)) for _ in range(plen))
        rep = rng.randint(2, 4)
        data.append(f"{pat}*{rep}={pat * rep}")
    return data


def make_sort_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data


def make_parity_data(n, rng):
    data = []
    for _ in range(n):
        length = rng.randint(2, 7)
        bits = "".join(str(rng.randint(0, 2)) for _ in range(length))
        count = bits.count("1")
        parity = "even" if count % 2 == 0 else "odd"
        data.append(f"{bits}>{parity}")
    return data


DOMAIN_GENERATORS = {
    "arithmetic": make_arithmetic_data,
    "reverse": make_reverse_data,
    "repeat": make_repeat_data,
    "sort": make_sort_data,
    "parity": make_parity_data,
}
DOMAINS = list(DOMAIN_GENERATORS.keys())


def encode_dataset(texts, tokenizer, max_len=MAX_SEQ):
    encoded = []
    for t in texts:
        ids = tokenizer.encode(t)[:max_len]
        ids = ids + [tokenizer.pad_id] * (max_len - len(ids))
        encoded.append(ids)
    return mx.array(np.array(encoded, dtype=np.int32))


# ===========================================================================
# Ternary helpers
# ===========================================================================

def ternary_quantize(w):
    alpha = mx.mean(mx.abs(w))
    w_scaled = w / (alpha + 1e-10)
    return mx.clip(mx.round(w_scaled), -1, 1) * alpha


def ternary_ste(w):
    """Forward: ternary. Backward: identity (STE)."""
    w_q = ternary_quantize(w)
    return w + mx.stop_gradient(w_q - w)


# ===========================================================================
# Model with built-in LoRA support
# ===========================================================================

class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adapter (frozen A + trainable B)."""
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        # LoRA fields (set externally when adapter is applied)
        self._lora_A = None  # frozen, not a param
        self._lora_B = None  # will be set as param when active

    def __call__(self, x):
        out = self.linear(x)
        if self._lora_A is not None and hasattr(self, 'lora_B'):
            B = ternary_ste(self.lora_B)
            delta = x @ self._lora_A @ B  # (*, d_in) @ (d_in, r) @ (r, d_out)
            out = out + delta
        return out

    def set_lora(self, A, B_init):
        """Attach LoRA. A is frozen, B is trainable."""
        self._lora_A = A
        self.lora_B = B_init

    def clear_lora(self):
        self._lora_A = None
        if hasattr(self, 'lora_B'):
            del self.lora_B

    def get_delta_weight(self):
        """Return the LoRA delta as a weight matrix."""
        if self._lora_A is not None and hasattr(self, 'lora_B'):
            return self._lora_A @ ternary_quantize(self.lora_B)
        return None


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SwiGLUFFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = LoRALinear(d, hidden)
        self.w_up = LoRALinear(d, hidden)
        self.w_down = LoRALinear(hidden, d)

    def __call__(self, x):
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


class Attention(nn.Module):
    def __init__(self, d, n_heads, use_xsa=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.w_q = LoRALinear(d, d)
        self.w_k = LoRALinear(d, d)
        self.w_v = LoRALinear(d, d)
        self.w_o = LoRALinear(d, d)
        self._use_xsa = use_xsa

    def __call__(self, x, mask=None):
        B, T, d = x.shape
        H = self.n_heads
        dh = self.d_head

        q = self.w_q(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = self.w_k(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = self.w_v(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        scale = math.sqrt(dh)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        if self._use_xsa:
            dot_yv = mx.sum(y * v, axis=-1, keepdims=True)
            norm_v_sq = mx.sum(v * v, axis=-1, keepdims=True) + 1e-8
            y = y - (dot_yv / norm_v_sq) * v

        y = y.transpose(0, 2, 1, 3).reshape(B, T, d)
        return self.w_o(y)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, ffn_hidden, use_xsa=False):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, use_xsa=use_xsa)
        self.ln2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, ffn_hidden)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, vocab_size, d, n_heads, n_layers, ffn_hidden, max_seq,
                 xsa_layers=None):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq, d)
        xsa_set = xsa_layers or set()
        self.layers = [
            TransformerBlock(d, n_heads, ffn_hidden, use_xsa=(i in xsa_set))
            for i in range(n_layers)
        ]
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(mx.arange(T))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)

    def get_lora_layers(self):
        """Return all LoRALinear modules with their paths."""
        result = []
        for li, layer in enumerate(self.layers):
            for name in ["w_q", "w_k", "w_v", "w_o"]:
                ll = getattr(layer.attn, name)
                result.append((f"layers.{li}.attn.{name}", ll))
            for name in ["w_gate", "w_up", "w_down"]:
                ll = getattr(layer.ffn, name)
                result.append((f"layers.{li}.ffn.{name}", ll))
        return result

    def set_adapter(self, A_dict, B_dict):
        """Attach LoRA A (frozen) and B (trainable) to all adapted layers."""
        for path, ll in self.get_lora_layers():
            if path in A_dict:
                ll.set_lora(A_dict[path], B_dict[path])

    def clear_adapter(self):
        """Remove all LoRA adapters."""
        for _, ll in self.get_lora_layers():
            ll.clear_lora()

    def get_adapter_deltas(self):
        """Return dict of path -> delta_W for current adapter."""
        deltas = {}
        for path, ll in self.get_lora_layers():
            d = ll.get_delta_weight()
            if d is not None:
                deltas[path] = d
        return deltas

    def freeze_base(self):
        """Freeze all non-LoRA parameters."""
        self.tok_emb.freeze()
        self.pos_emb.freeze()
        self.ln_f.freeze()
        self.head.freeze()
        for layer in self.layers:
            layer.ln1.freeze()
            layer.ln2.freeze()
            for _, ll in self.get_lora_layers():
                ll.linear.freeze()

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        self.tok_emb.unfreeze()
        self.pos_emb.unfreeze()
        self.ln_f.unfreeze()
        self.head.unfreeze()
        for layer in self.layers:
            layer.ln1.unfreeze()
            layer.ln2.unfreeze()
            for _, ll in self.get_lora_layers():
                ll.linear.unfreeze()


def quantize_model_to_ternary(model):
    """Post-quantize weight matrices to ternary."""
    flat = tree_flatten(model.parameters())
    new_params = []
    for name, p in flat:
        if p.ndim >= 2 and "emb" not in name and "ln" not in name:
            new_params.append((name, ternary_quantize(p)))
        else:
            new_params.append((name, p))
    model.load_weights(new_params)
    mx.eval(model.parameters())


# ===========================================================================
# Grassmannian A matrices
# ===========================================================================

def make_grassmannian_As(d_in, rank, n, seed=42):
    """Generate n orthonormal A matrices via QR."""
    rng = np.random.RandomState(seed)
    As = []
    for _ in range(n):
        M = rng.randn(d_in, rank).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        As.append(mx.array(Q[:, :rank]))
    return As


def generate_all_A_matrices(model, n_adapters, seed):
    """Generate Grassmannian A matrices for all LoRA-able layers."""
    A_per_adapter = [{} for _ in range(n_adapters)]
    for path, ll in model.get_lora_layers():
        d_in = ll.linear.weight.shape[1]  # nn.Linear stores (d_out, d_in), but input dim is shape[1]
        # Actually MLX nn.Linear: weight is (out, in), x @ W.T
        # So d_in is weight.shape[1]
        As = make_grassmannian_As(d_in, LORA_RANK, n_adapters, seed=seed + hash(path) % 10000)
        for i in range(n_adapters):
            A_per_adapter[i][path] = As[i]
    return A_per_adapter


def init_B_matrices(model, A_dict, seed):
    """Initialize random B matrices for one adapter."""
    rng = np.random.RandomState(seed)
    B_dict = {}
    for path, ll in model.get_lora_layers():
        if path in A_dict:
            d_out = ll.linear.weight.shape[0]  # output dim
            B_dict[path] = mx.array(rng.randn(LORA_RANK, d_out).astype(np.float32) * 0.01)
    return B_dict


# ===========================================================================
# Loss and PPL
# ===========================================================================

def compute_loss(model, tokens, pad_id=0):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = model(inputs)
    mask = (targets != pad_id).astype(logits.dtype)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_lp = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    return -mx.sum(target_lp * mask) / (mx.sum(mask) + 1e-10)


def compute_ppl(model, data, pad_id=0, batch_size=64):
    total_loss = 0.0
    total_tokens = 0
    for s in range(0, data.shape[0], batch_size):
        batch = data[s:s + batch_size]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        mask = (targets != pad_id).astype(logits.dtype)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tlp = mx.take_along_axis(lp, targets[:, :, None], axis=-1).squeeze(-1)
        bl = -mx.sum(tlp * mask)
        bt = mx.sum(mask)
        mx.eval(bl, bt)
        total_loss += bl.item()
        total_tokens += bt.item()
        del logits, lp, tlp, bl, bt
    return math.exp(total_loss / (total_tokens + 1e-10))


# ===========================================================================
# Phase 1: Data generation
# ===========================================================================

def phase_generate_data(seed):
    log(f"Phase 1: Generate data (seed={seed})")
    tok = CharTokenizer()
    rng = np.random.RandomState(seed)
    dd = {}
    for d, fn in DOMAIN_GENERATORS.items():
        dd[d] = {
            "train": encode_dataset(fn(TRAIN_SAMPLES, rng), tok),
            "val": encode_dataset(fn(VAL_SAMPLES, rng), tok),
        }
    log(f"  {len(DOMAINS)} domains, V={tok.vocab_size}")
    return tok, dd


# ===========================================================================
# Phase 2: Train + quantize base
# ===========================================================================

def phase_train_base(tok, dd, seed, xsa_layers=None):
    cond = "XSA" if xsa_layers else "std"
    log(f"Phase 2: Train base ({cond}, seed={seed})")
    mx.random.seed(seed)

    model = MicroTransformer(
        tok.vocab_size, D_MODEL, N_HEADS, N_LAYERS, FFN_HIDDEN, MAX_SEQ,
        xsa_layers=xsa_layers
    )
    mx.eval(model.parameters())
    np_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Params: {np_total:,}")

    all_train = mx.concatenate([dd[d]["train"] for d in DOMAINS])
    optimizer = opt.Adam(learning_rate=LR_BASE)
    lg = nn.value_and_grad(model, compute_loss)

    gc.disable()
    for ep in range(BASE_EPOCHS):
        perm = mx.array(np.random.permutation(all_train.shape[0]))
        shuf = all_train[perm]
        el, nb = 0.0, 0
        for s in range(0, shuf.shape[0], BATCH_SIZE):
            loss, grads = lg(model, shuf[s:s+BATCH_SIZE])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            el += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == BASE_EPOCHS - 1:
            log(f"  Ep {ep}: loss={el/nb:.4f}")
    gc.enable()

    quantize_model_to_ternary(model)
    log(f"  Quantized to ternary")

    # Snapshot base weights
    base_w = [(n, mx.array(p)) for n, p in tree_flatten(model.parameters())]
    mx.eval([w for _, w in base_w])

    cleanup(optimizer, all_train)
    log_memory(f"post-base-{cond}")
    return model, base_w


# ===========================================================================
# Phase 3: Train domain adapters
# ===========================================================================

def phase_train_adapters(model, base_w, dd, tok, seed, xsa_layers=None):
    cond = "XSA" if xsa_layers else "std"
    log(f"Phase 3: Train adapters ({cond}, seed={seed})")

    # Generate all Grassmannian A matrices
    all_A = generate_all_A_matrices(model, len(DOMAINS), seed)

    trained_adapters = []  # list of (A_dict, B_dict) tuples
    single_ppls = {}

    for di, domain in enumerate(DOMAINS):
        log(f"  Adapter '{domain}' ({di+1}/{len(DOMAINS)})")

        # Restore base
        model.load_weights(base_w)
        mx.eval(model.parameters())

        # Init B
        A_dict = all_A[di]
        B_dict = init_B_matrices(model, A_dict, seed=seed + di * 1000)

        # Attach adapter
        model.set_adapter(A_dict, B_dict)
        mx.eval(model.parameters())

        # Freeze base, only train LoRA B
        model.freeze_base()

        optimizer = opt.Adam(learning_rate=LR_ADAPTER)
        lg = nn.value_and_grad(model, compute_loss)
        train_data = dd[domain]["train"]

        gc.disable()
        for ep in range(ADAPTER_EPOCHS):
            perm = mx.array(np.random.permutation(train_data.shape[0]))
            shuf = train_data[perm]
            el, nb = 0.0, 0
            for s in range(0, shuf.shape[0], BATCH_SIZE):
                loss, grads = lg(model, shuf[s:s+BATCH_SIZE])
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)
                el += loss.item()
                nb += 1
            if ep % 10 == 0 or ep == ADAPTER_EPOCHS - 1:
                log(f"    Ep {ep}: loss={el/nb:.4f}")
        gc.enable()

        # Extract trained B values
        final_B = {}
        for path, ll in model.get_lora_layers():
            if hasattr(ll, 'lora_B'):
                final_B[path] = mx.array(ll.lora_B)
        mx.eval(list(final_B.values()))

        # Compute delta weights (for composition later)
        deltas = model.get_adapter_deltas()
        mx.eval(list(deltas.values()))

        # Evaluate single-adapter PPL (adapter already applied)
        ppl = compute_ppl(model, dd[domain]["val"], tok.pad_id)
        single_ppls[domain] = ppl
        log(f"    PPL: {ppl:.4f}")

        trained_adapters.append((A_dict, final_B, deltas))

        # Clear adapter before next
        model.clear_adapter()
        model.unfreeze_all()
        del optimizer, lg
        gc.collect()

    # Restore base
    model.load_weights(base_w)
    mx.eval(model.parameters())
    log_memory(f"post-adapters-{cond}")
    return trained_adapters, single_ppls


# ===========================================================================
# Phase 4: Evaluate composition (pre-merge)
# ===========================================================================

def phase_eval_composition(model, base_w, trained_adapters, dd, tok):
    log("Phase 4: Composition eval")
    model.load_weights(base_w)
    mx.eval(model.parameters())

    # Average delta weights
    k = len(trained_adapters)
    all_deltas = [ta[2] for ta in trained_adapters]
    merged = {}
    for key in all_deltas[0]:
        merged[key] = mx.mean(mx.stack([d[key] for d in all_deltas]), axis=0)

    # Apply merged deltas to base linear weights
    flat = tree_flatten(model.parameters())
    new_w = []
    for name, p in flat:
        # Map from model param path to lora path
        # LoRA delta keys are like "layers.0.attn.w_q", model weight keys are
        # "layers.0.attn.w_q.linear.weight"
        matched = False
        for lora_path, delta in merged.items():
            expected_weight_key = lora_path + ".linear.weight"
            if name == expected_weight_key:
                # nn.Linear weight is (d_out, d_in), delta is (d_in, d_out)
                new_w.append((name, p + delta.T))
                matched = True
                break
        if not matched:
            new_w.append((name, p))
    model.load_weights(new_w)
    mx.eval(model.parameters())

    composed_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, dd[domain]["val"], tok.pad_id)
        composed_ppls[domain] = ppl
        log(f"    {domain}: {ppl:.4f}")

    model.load_weights(base_w)
    mx.eval(model.parameters())
    log_memory("post-compose")
    return composed_ppls


# ===========================================================================
# Phase 5: Adapter cosine diagnostics
# ===========================================================================

def phase_diagnostics(trained_adapters):
    log("Phase 5: Diagnostics")
    cosines = []
    n = len(trained_adapters)
    for i in range(n):
        d_i = trained_adapters[i][2]
        for j in range(i + 1, n):
            d_j = trained_adapters[j][2]
            pair_cos = []
            for key in d_i:
                vi = d_i[key].reshape(-1)
                vj = d_j[key].reshape(-1)
                cos = mx.sum(vi * vj) / (
                    mx.sqrt(mx.sum(vi * vi)) * mx.sqrt(mx.sum(vj * vj)) + 1e-10
                )
                mx.eval(cos)
                pair_cos.append(abs(cos.item()))
            cosines.append(float(np.mean(pair_cos)))
    mean_cos = float(np.mean(cosines)) if cosines else 0.0
    log(f"  Mean |cos|: {mean_cos:.6f}")
    return {"mean_adapter_cosine": mean_cos}


# ===========================================================================
# Full condition run
# ===========================================================================

def run_condition(seed, xsa_layers=None):
    cond = "XSA" if xsa_layers else "standard"
    log(f"\n=== {cond.upper()} seed={seed} ===")

    tok, dd = phase_generate_data(seed)
    model, base_w = phase_train_base(tok, dd, seed, xsa_layers)

    # Base PPLs
    base_ppls = {}
    for d in DOMAINS:
        base_ppls[d] = compute_ppl(model, dd[d]["val"], tok.pad_id)
    log(f"  Base: {', '.join(f'{d}={p:.2f}' for d, p in base_ppls.items())}")

    adapters, single_ppls = phase_train_adapters(model, base_w, dd, tok, seed, xsa_layers)
    composed_ppls = phase_eval_composition(model, base_w, adapters, dd, tok)
    diag = phase_diagnostics(adapters)

    result = {
        "base_ppls": base_ppls,
        "single_ppls": single_ppls,
        "composed_ppls": composed_ppls,
        "diagnostics": diag,
    }

    cleanup(model, base_w, dd, tok)
    # Adapters hold mx.arrays in their delta dicts - force cleanup
    del adapters
    gc.collect()
    mx.clear_cache()

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    log("=" * 60)
    log("XSA Composition Quality Experiment")
    log(f"d={D_MODEL} H={N_HEADS} L={N_LAYERS} r={LORA_RANK} XSA_layers={N_XSA_LAYERS}")
    log(f"Seeds: {SEEDS}")
    log("=" * 60)
    log_memory("start")

    xsa_set = set(range(N_LAYERS - N_XSA_LAYERS, N_LAYERS))

    results = {"seeds": {}, "config": {
        "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
        "ffn_hidden": FFN_HIDDEN, "lora_rank": LORA_RANK,
        "n_xsa_layers": N_XSA_LAYERS, "base_epochs": BASE_EPOCHS,
        "adapter_epochs": ADAPTER_EPOCHS, "seeds": SEEDS, "domains": DOMAINS,
    }}

    for seed in SEEDS:
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        std = run_condition(seed, xsa_layers=None)
        xsa = run_condition(seed, xsa_layers=xsa_set)
        results["seeds"][str(seed)] = {"standard": std, "xsa": xsa}

    # Aggregate
    log(f"\n{'='*60}")
    log("AGGREGATION")
    log(f"{'='*60}")

    summary = {}

    # K1
    k1_max = -999.0
    for d in DOMAINS:
        sp = [results["seeds"][str(s)]["standard"]["single_ppls"][d] for s in SEEDS]
        xp = [results["seeds"][str(s)]["xsa"]["single_ppls"][d] for s in SEEDS]
        sm, xm = float(np.mean(sp)), float(np.mean(xp))
        deg = (xm - sm) / sm * 100
        k1_max = max(k1_max, deg)
        log(f"  K1 {d}: std={sm:.4f} xsa={xm:.4f} delta={deg:+.2f}%")
    k1_pass = k1_max <= 3.0
    summary["k1_max_degradation_pct"] = round(k1_max, 2)
    summary["k1_pass"] = k1_pass
    log(f"  K1: {k1_max:.2f}% -> {'PASS' if k1_pass else 'FAIL'}")

    # K2
    sr, xr = [], []
    for s in SEEDS:
        sd = results["seeds"][str(s)]
        for d in DOMAINS:
            sr.append(sd["standard"]["composed_ppls"][d] / sd["standard"]["single_ppls"][d])
            xr.append(sd["xsa"]["composed_ppls"][d] / sd["xsa"]["single_ppls"][d])
    sm_r, xm_r = float(np.mean(sr)), float(np.mean(xr))
    k2_pass = xm_r < sm_r
    summary["std_ratio"] = round(sm_r, 4)
    summary["xsa_ratio"] = round(xm_r, 4)
    summary["k2_pass"] = k2_pass
    log(f"  K2: std_ratio={sm_r:.4f} xsa_ratio={xm_r:.4f} -> {'PASS' if k2_pass else 'FAIL'}")

    # K3
    xsa_wins = 0
    for d in DOMAINS:
        sc = float(np.mean([results["seeds"][str(s)]["standard"]["composed_ppls"][d] for s in SEEDS]))
        xc = float(np.mean([results["seeds"][str(s)]["xsa"]["composed_ppls"][d] for s in SEEDS]))
        won = xc < sc
        if won:
            xsa_wins += 1
        log(f"  K3 {d}: std={sc:.4f} xsa={xc:.4f} {'XSA' if won else 'std'}")
    k3_pass = (len(DOMAINS) - xsa_wins) < 3
    summary["xsa_wins"] = xsa_wins
    summary["k3_pass"] = k3_pass
    log(f"  K3: XSA wins {xsa_wins}/{len(DOMAINS)} -> {'PASS' if k3_pass else 'FAIL'}")

    # S1
    ri = (sm_r - xm_r) / sm_r * 100
    summary["ratio_improvement_pct"] = round(ri, 2)
    summary["s1_pass"] = ri > 2.0
    log(f"  S1: improvement={ri:.2f}% -> {'PASS' if summary['s1_pass'] else 'FAIL'}")

    # Cosines
    for c in ["standard", "xsa"]:
        mc = float(np.mean([results["seeds"][str(s)][c]["diagnostics"]["mean_adapter_cosine"] for s in SEEDS]))
        summary[f"{c}_cos"] = round(mc, 6)

    summary["verdict"] = "SUPPORTED" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    results["summary"] = summary
    results["total_time_s"] = round(time.time() - t0, 1)

    log(f"\nVERDICT: {summary['verdict']}")
    log(f"Time: {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
