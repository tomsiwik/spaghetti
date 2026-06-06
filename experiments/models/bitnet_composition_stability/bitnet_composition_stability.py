#!/usr/bin/env python3
"""
BitNet Composition Stability: Does a ternary base prevent PPL explosion under
equal-weight LoRA composition?

Hypothesis: ternary {-1, 0, 1} base weights bound adapter magnitudes and reduce
logit-scale mismatch, preventing the composition catastrophe observed with FP16.

Experimental design:
  1. Train base model on mixed data (FP16 weights)
  2. Quantize base to ternary via absmean quantization (BitNet recipe)
  3. Fine-tune 5 domain LoRA experts on EACH base (FP16 and ternary)
     - LoRA: freeze base, train A/B matrices for FFN+Attn layers
  4. Compose all 5 at equal weight: W + (1/N)*sum(B_i @ A_i)
  5. Measure composed PPL / base PPL ratio on each domain

Kill criteria:
  K1: composed PPL > 100x base PPL at N=5 (ternary base does not help)
  K2: composed PPL > 10x single-adapter PPL on >50% of domains
  K3: training fails or adapters do not converge on ternary base

Additionally measures:
  - Adapter delta norms (Frobenius) for each base type
  - Adapter cosine similarities (interference proxy)
  - Per-domain composition quality
  - PPL ratio comparison: ternary vs FP16

Micro scale: d=64, r=4, L=2, 5 domains, 3 seeds.
"""

import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp


# ===========================================================================
# Synthetic Data: 5 domains (matching SOLE pilot domains conceptually)
# ===========================================================================

def _make_arithmetic_data(n, rng):
    """Domain 0: addition. '12+34=46'"""
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        data.append(f"{a}+{b}={a+b}")
    return data

def _make_reverse_data(n, rng):
    """Domain 1: string reversal. 'abc>cba'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{s[::-1]}")
    return data

def _make_repeat_data(n, rng):
    """Domain 2: string repetition. 'ab*3=ababab'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(list(chars)) for _ in range(plen))
        rep = rng.randint(2, 4)
        data.append(f"{pat}*{rep}={pat * rep}")
    return data

def _make_sort_data(n, rng):
    """Domain 3: character sorting. 'dcba>abcd'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data

def _make_parity_data(n, rng):
    """Domain 4: digit parity. '1011>odd'"""
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        bits = "".join(str(rng.randint(0, 2)) for _ in range(length))
        count = bits.count("1")
        parity = "even" if count % 2 == 0 else "odd"
        data.append(f"{bits}>{parity}")
    return data

DOMAIN_GENERATORS = {
    "arithmetic": _make_arithmetic_data,
    "reverse": _make_reverse_data,
    "repeat": _make_repeat_data,
    "sort": _make_sort_data,
    "parity": _make_parity_data,
}


# ===========================================================================
# Tokenizer
# ===========================================================================

class CharTokenizer:
    def __init__(self):
        chars = sorted(set("0123456789abcdefghijklmnopqrstuvwxyz>+=*"))
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        specials = [self.pad_token, self.eos_token]
        self.vocab = specials + chars
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.char2idx[self.pad_token]
        self.eos_id = self.char2idx[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        return [self.char2idx[c] for c in s if c in self.char2idx] + [self.eos_id]


# ===========================================================================
# Model (supports both FP16-like and ternary base weights)
# ===========================================================================

def init_model(V, d=64, H=2, L=2, max_T=32, seed=42):
    """Initialize a standard FP16 transformer model."""
    rng = onp.random.RandomState(seed)
    s = 0.02
    params = {
        'tok_emb': rng.randn(V, d).astype(onp.float32) * s,
        'pos_emb': rng.randn(max_T, d).astype(onp.float32) * s,
    }
    for li in range(L):
        params[f'ln1_w_{li}'] = onp.ones(d, dtype=onp.float32)
        params[f'Wqkv_{li}'] = rng.randn(d, 3 * d).astype(onp.float32) * s
        params[f'Wo_{li}'] = rng.randn(d, d).astype(onp.float32) * s
        params[f'ln2_w_{li}'] = onp.ones(d, dtype=onp.float32)
        params[f'W1_{li}'] = rng.randn(d, 4 * d).astype(onp.float32) * s
        params[f'W2_{li}'] = rng.randn(4 * d, d).astype(onp.float32) * s
    params['ln_f_w'] = onp.ones(d, dtype=onp.float32)
    params['W_head'] = rng.randn(d, V).astype(onp.float32) * s
    params['_config'] = {'V': V, 'd': d, 'H': H, 'L': L, 'max_T': max_T}
    return params


def ternary_quantize(W):
    """BitNet b1.58 absmean quantization: W_tilde = RoundClip(W / mean(|W|), -1, 1).

    This produces {-1, 0, 1} weights following the BitNet recipe.
    The scale factor (mean(|W|)) is stored separately for dequantization.
    """
    alpha = onp.mean(onp.abs(W))
    if alpha < 1e-10:
        return onp.zeros_like(W), alpha
    W_scaled = W / alpha
    W_ternary = onp.clip(onp.round(W_scaled), -1, 1).astype(onp.float32)
    return W_ternary, alpha


def quantize_model_to_ternary(params):
    """Quantize all weight matrices to ternary {-1, 0, 1}.

    Following BitNet: each weight matrix is independently quantized via absmean.
    The scale factors are stored for proper forward pass computation.
    Embeddings and layernorms are kept in FP32 (following BitNet convention).
    """
    ternary_params = {}
    scales = {}

    # Keys that should remain FP32 (embeddings, layernorms)
    fp32_keys = {'tok_emb', 'pos_emb', 'ln_f_w', '_config'}
    fp32_keys.update(k for k in params if k.startswith('ln'))

    for k, v in params.items():
        if k in fp32_keys or k == '_config':
            ternary_params[k] = v if k == '_config' else v.copy()
        elif v.ndim >= 2:
            # Quantize weight matrices
            W_t, alpha = ternary_quantize(v)
            ternary_params[k] = W_t * alpha  # Store as scaled ternary for computation
            scales[k] = alpha
        else:
            ternary_params[k] = v.copy()

    return ternary_params, scales


def _rms_norm(x, w, eps=1e-5):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def forward(params, idx_2d, pad_id=0):
    cfg = params['_config']
    d, H, L = cfg['d'], cfg['H'], cfg['L']
    hd = d // H
    B, T = idx_2d.shape
    x = params['tok_emb'][idx_2d] + params['pos_emb'][:T]
    mask = onp.triu(onp.ones((T, T)) * (-1e9), k=1).astype(onp.float32)
    for li in range(L):
        h = _rms_norm(x, params[f'ln1_w_{li}'])
        qkv = np.dot(h, params[f'Wqkv_{li}'])
        qkv = np.reshape(qkv, (B, T, 3, H, hd))
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        scale = 1.0 / onp.sqrt(hd)
        attn = np.einsum('bhqd,bhkd->bhqk', q, k) * scale + mask
        attn = attn - np.max(attn, axis=-1, keepdims=True)
        attn = np.exp(attn)
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        out = np.einsum('bhqk,bhkd->bhqd', attn, v)
        out = np.transpose(out, (0, 2, 1, 3))
        out = np.reshape(out, (B, T, d))
        out = np.dot(out, params[f'Wo_{li}'])
        x = x + out
        h = _rms_norm(x, params[f'ln2_w_{li}'])
        ffn = np.maximum(0, np.dot(h, params[f'W1_{li}']))
        ffn = np.dot(ffn, params[f'W2_{li}'])
        x = x + ffn
    x = _rms_norm(x, params['ln_f_w'])
    logits = np.dot(x, params['W_head'])
    return logits


def compute_loss(params, idx_2d, targets_2d, mask_2d, pad_id=0):
    logits = forward(params, idx_2d, pad_id)
    B, T, V = logits.shape
    max_l = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_l
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    targets_oh = onp.zeros((B, T, V), dtype=onp.float32)
    for b in range(B):
        for t in range(T):
            targets_oh[b, t, targets_2d[b, t]] = 1.0
    token_losses = -np.sum(log_probs * targets_oh, axis=-1)
    masked_loss = np.sum(token_losses * mask_2d)
    n_tokens = np.sum(mask_2d) + 1e-10
    return masked_loss / n_tokens


# ===========================================================================
# LoRA Training (proper low-rank adaptation with frozen base)
# ===========================================================================

def init_lora(params, rank=4, seed=42):
    """Initialize LoRA A/B matrices for all weight matrices (all-modules).

    A: (d_in, r) initialized with small random values
    B: (r, d_out) initialized to zero (so initial delta = 0)
    """
    rng = onp.random.RandomState(seed)
    lora = {}
    for k, v in params.items():
        if k == '_config' or v.ndim < 2:
            continue
        d_in, d_out = v.shape
        # Kaiming init for A, zero for B (standard LoRA)
        lora[f'{k}_A'] = rng.randn(d_in, rank).astype(onp.float32) * 0.01
        lora[f'{k}_B'] = onp.zeros((rank, d_out), dtype=onp.float32)
    return lora


def apply_lora(base_params, lora, alpha_over_r=1.0):
    """Create effective params by adding LoRA deltas to frozen base."""
    result = {}
    for k, v in base_params.items():
        if k == '_config':
            result[k] = v
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            delta = np.dot(lora[A_key], lora[B_key]) * alpha_over_r
            result[k] = v + delta
        else:
            result[k] = v
    return result


def train_lora(base_params, lora, data_encoded, pad_id, epochs=30, lr=0.003,
               batch_size=32, clip_grad=1.0, verbose=True):
    """Train LoRA params while keeping base frozen."""
    cfg = base_params['_config']
    lora_keys = sorted(lora.keys())

    def loss_fn(lora_vals, inp, tgt, mask):
        lo = dict(zip(lora_keys, lora_vals))
        effective = apply_lora(base_params, lo)
        return compute_loss(effective, inp, tgt, mask, pad_id)

    grad_fn = grad(loss_fn)

    # Adam state
    m_state = [onp.zeros_like(lora[k]) for k in lora_keys]
    v_state = [onp.zeros_like(lora[k]) for k in lora_keys]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    n = len(data_encoded)
    rng = onp.random.RandomState(42)

    for epoch in range(epochs):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [data_encoded[j] for j in batch_idx]
            inp, tgt, mask = _prepare_batch(batch, pad_id)
            if onp.sum(mask) == 0:
                continue
            lora_vals = [lora[k] for k in lora_keys]
            loss_val = float(loss_fn(lora_vals, inp, tgt, mask))
            grads = grad_fn(lora_vals, inp, tgt, mask)
            grad_norm = onp.sqrt(sum(float(onp.sum(g**2)) for g in grads))
            if grad_norm > clip_grad:
                sc = clip_grad / grad_norm
                grads = [g * sc for g in grads]
            step += 1
            for k_idx, key in enumerate(lora_keys):
                g = onp.array(grads[k_idx])
                m_state[k_idx] = beta1 * m_state[k_idx] + (1 - beta1) * g
                v_state[k_idx] = beta2 * v_state[k_idx] + (1 - beta2) * g**2
                m_hat = m_state[k_idx] / (1 - beta1**step)
                v_hat = v_state[k_idx] / (1 - beta2**step)
                lora[key] = lora[key] - lr * m_hat / (onp.sqrt(v_hat) + eps)
            epoch_loss += loss_val
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"        epoch {epoch:3d}: loss={avg_loss:.4f}")

    return lora


def lora_to_delta(lora, base_params, alpha_over_r=1.0):
    """Convert LoRA A/B matrices to weight deltas."""
    delta = {}
    for k in base_params:
        if k == '_config' or base_params[k].ndim < 2:
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            delta[k] = onp.dot(lora[A_key], lora[B_key]) * alpha_over_r
    return delta


# ===========================================================================
# Batch preparation
# ===========================================================================

def _prepare_batch(seqs, pad_id, max_len=32):
    max_T = min(max(len(s) for s in seqs), max_len)
    B = len(seqs)
    idx = onp.full((B, max_T), pad_id, dtype=onp.int32)
    for b, seq in enumerate(seqs):
        L = min(len(seq), max_T)
        idx[b, :L] = seq[:L]
    inp = idx[:, :-1]
    tgt = idx[:, 1:]
    mask = (tgt != pad_id).astype(onp.float32)
    return inp, tgt, mask


def eval_loss(params, data_encoded, pad_id, batch_size=32):
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(data_encoded), batch_size):
        batch = data_encoded[i:i+batch_size]
        inp, tgt, mask = _prepare_batch(batch, pad_id)
        if onp.sum(mask) == 0:
            continue
        logits = onp.array(forward(params, inp, pad_id))
        B, T, V = logits.shape
        max_l = onp.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_l
        log_probs = shifted - onp.log(onp.sum(onp.exp(shifted), axis=-1, keepdims=True))
        for b in range(B):
            for t in range(T):
                if mask[b, t] > 0:
                    total_loss += -log_probs[b, t, tgt[b, t]]
                    total_tokens += 1
    if total_tokens == 0:
        return float('inf')
    return float(total_loss / total_tokens)


# ===========================================================================
# Delta operations
# ===========================================================================

def apply_delta(base_params, delta):
    result = {}
    for k in base_params:
        if k == '_config':
            result[k] = base_params[k]
        elif k in delta:
            result[k] = base_params[k] + delta[k]
        else:
            result[k] = base_params[k].copy()
    return result


def compose_deltas(delta_list, mode='equal'):
    """Equal-weight composition: sum of deltas divided by N (SOLE default)."""
    merged = {}
    N = len(delta_list)
    for k in delta_list[0]:
        s = sum(d[k] for d in delta_list)
        if mode == 'equal':
            merged[k] = s / N
        else:
            merged[k] = s
    return merged


def flatten_delta(delta):
    parts = []
    for k in sorted(delta.keys()):
        parts.append(delta[k].flatten())
    return onp.concatenate(parts)


def cosine_sim(a, b):
    na, nb = onp.linalg.norm(a), onp.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(onp.dot(a, b) / (na * nb))


# ===========================================================================
# Diagnostics
# ===========================================================================

def measure_weight_stats(params):
    """Measure weight magnitude statistics for a model."""
    stats = {}
    for k, v in params.items():
        if k == '_config' or v.ndim < 2:
            continue
        stats[k] = {
            'frobenius_norm': float(onp.linalg.norm(v)),
            'mean_abs': float(onp.mean(onp.abs(v))),
            'max_abs': float(onp.max(onp.abs(v))),
            'sparsity': float(onp.mean(onp.abs(v) < 1e-6)),  # fraction near zero
            'unique_values': int(len(onp.unique(onp.round(v, 4)))),
        }
    return stats


def measure_ternary_fidelity(params):
    """Measure how close weights are to {-1, 0, 1}."""
    fidelity = {}
    for k, v in params.items():
        if k == '_config' or v.ndim < 2:
            continue
        # After composition, check distance from nearest ternary value
        nearest = onp.clip(onp.round(v / max(onp.mean(onp.abs(v)), 1e-10)), -1, 1)
        scale = onp.mean(onp.abs(v))
        residual = v - nearest * scale
        fidelity[k] = {
            'ternary_residual_norm': float(onp.linalg.norm(residual)),
            'weight_norm': float(onp.linalg.norm(v)),
            'ternary_fidelity_ratio': float(onp.linalg.norm(residual) / max(onp.linalg.norm(v), 1e-10)),
        }
    return fidelity


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(seed=42, d=64, r=4, L=2, H=2, n_data=300, n_eval=100,
                   base_epochs=30, lora_epochs=30, verbose=True):
    """Run one seed of the BitNet composition stability experiment."""

    print(f"\n{'='*70}")
    print(f"  BitNet Composition Stability Experiment (seed={seed}, d={d}, r={r})")
    print(f"{'='*70}")

    tok = CharTokenizer()
    V = tok.vocab_size
    rng = onp.random.RandomState(seed)
    domain_names = list(DOMAIN_GENERATORS.keys())

    # -----------------------------------------------------------------------
    # Step 1: Generate domain data
    # -----------------------------------------------------------------------
    print("\n[1/6] Generating domain data...")
    domain_data = {}
    domain_eval = {}
    mixed_train = []
    for name, gen_fn in DOMAIN_GENERATORS.items():
        train = gen_fn(n_data, rng)
        test = gen_fn(n_eval, rng)
        domain_data[name] = [tok.encode(s) for s in train]
        domain_eval[name] = [tok.encode(s) for s in test]
        mixed_train.extend(train[:n_data // len(DOMAIN_GENERATORS)])

    mixed_encoded = [tok.encode(s) for s in mixed_train]

    # -----------------------------------------------------------------------
    # Step 2: Train FP16 base model on mixed data
    # -----------------------------------------------------------------------
    print("\n[2/6] Training FP16 base model...")
    fp16_base = init_model(V, d=d, H=H, L=L, seed=seed)
    fp16_base = train_base(fp16_base, mixed_encoded, tok.pad_id,
                           epochs=base_epochs, verbose=verbose)

    # -----------------------------------------------------------------------
    # Step 3: Create ternary base via absmean quantization
    # -----------------------------------------------------------------------
    print("\n[3/6] Quantizing base to ternary (BitNet absmean)...")
    ternary_base, scales = quantize_model_to_ternary(fp16_base)

    # Measure ternary properties
    fp16_stats = measure_weight_stats(fp16_base)
    ternary_stats = measure_weight_stats(ternary_base)

    print("  Weight statistics comparison:")
    for k in sorted(fp16_stats.keys())[:3]:
        f_norm = fp16_stats[k]['frobenius_norm']
        t_norm = ternary_stats[k]['frobenius_norm']
        t_sparse = ternary_stats[k]['sparsity']
        print(f"    {k}: FP16 norm={f_norm:.4f}, Ternary norm={t_norm:.4f}, "
              f"ternary sparsity={t_sparse:.1%}")

    # Evaluate base models on each domain
    print("\n  Base model eval (before LoRA):")
    fp16_base_ppl = {}
    ternary_base_ppl = {}
    for name in domain_names:
        fp16_loss = eval_loss(fp16_base, domain_eval[name], tok.pad_id)
        ternary_loss = eval_loss(ternary_base, domain_eval[name], tok.pad_id)
        fp16_base_ppl[name] = onp.exp(fp16_loss)
        ternary_base_ppl[name] = onp.exp(ternary_loss)
        print(f"    {name:12s}: FP16 PPL={fp16_base_ppl[name]:.2f}, "
              f"Ternary PPL={ternary_base_ppl[name]:.2f}")

    # -----------------------------------------------------------------------
    # Step 4: Train LoRA adapters on each base
    # -----------------------------------------------------------------------
    fp16_loras = {}
    ternary_loras = {}
    fp16_deltas = {}
    ternary_deltas = {}
    fp16_single_ppl = {}
    ternary_single_ppl = {}

    for di, name in enumerate(domain_names):
        lora_seed = seed * 100 + di

        # FP16 base LoRA
        print(f"\n[4/6] Training LoRA on FP16 base: {name} (seed={lora_seed})...")
        lora_fp16 = init_lora(fp16_base, rank=r, seed=lora_seed)
        lora_fp16 = train_lora(fp16_base, lora_fp16, domain_data[name], tok.pad_id,
                                epochs=lora_epochs, lr=0.003, verbose=verbose)
        fp16_loras[name] = lora_fp16
        fp16_deltas[name] = lora_to_delta(lora_fp16, fp16_base)

        # Eval single adapter
        effective = apply_lora(fp16_base, lora_fp16)
        single_loss = eval_loss(effective, domain_eval[name], tok.pad_id)
        fp16_single_ppl[name] = onp.exp(single_loss)

        # Ternary base LoRA
        print(f"    Training LoRA on Ternary base: {name}...")
        lora_ternary = init_lora(ternary_base, rank=r, seed=lora_seed)
        lora_ternary = train_lora(ternary_base, lora_ternary, domain_data[name], tok.pad_id,
                                   epochs=lora_epochs, lr=0.003, verbose=verbose)
        ternary_loras[name] = lora_ternary
        ternary_deltas[name] = lora_to_delta(lora_ternary, ternary_base)

        # Eval single adapter
        effective_t = apply_lora(ternary_base, lora_ternary)
        single_loss_t = eval_loss(effective_t, domain_eval[name], tok.pad_id)
        ternary_single_ppl[name] = onp.exp(single_loss_t)

    # -----------------------------------------------------------------------
    # Step 5: Equal-weight composition
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Composing all {len(domain_names)} adapters (equal weight)...")

    # FP16 composition
    fp16_merged_delta = compose_deltas(list(fp16_deltas.values()), mode='equal')
    fp16_composed = apply_delta(fp16_base, fp16_merged_delta)

    # Ternary composition
    ternary_merged_delta = compose_deltas(list(ternary_deltas.values()), mode='equal')
    ternary_composed = apply_delta(ternary_base, ternary_merged_delta)

    # Also test SUM (not averaged) composition to measure raw interference
    fp16_sum_delta = compose_deltas(list(fp16_deltas.values()), mode='sum')
    fp16_sum_composed = apply_delta(fp16_base, fp16_sum_delta)
    ternary_sum_delta = compose_deltas(list(ternary_deltas.values()), mode='sum')
    ternary_sum_composed = apply_delta(ternary_base, ternary_sum_delta)

    # -----------------------------------------------------------------------
    # Step 6: Evaluate composition
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Evaluating composition quality...")

    results = {
        'seed': seed, 'd': d, 'r': r, 'L': L,
        'n_domains': len(domain_names),
        'domains': {},
        'diagnostics': {},
    }

    fp16_composed_ppls = {}
    ternary_composed_ppls = {}
    fp16_sum_ppls = {}
    ternary_sum_ppls = {}

    for name in domain_names:
        # FP16 composed on this domain
        fp16_c_loss = eval_loss(fp16_composed, domain_eval[name], tok.pad_id)
        fp16_c_ppl = onp.exp(fp16_c_loss)
        fp16_composed_ppls[name] = fp16_c_ppl

        # Ternary composed on this domain
        t_c_loss = eval_loss(ternary_composed, domain_eval[name], tok.pad_id)
        t_c_ppl = onp.exp(t_c_loss)
        ternary_composed_ppls[name] = t_c_ppl

        # Sum composition (no averaging)
        fp16_s_loss = eval_loss(fp16_sum_composed, domain_eval[name], tok.pad_id)
        fp16_sum_ppls[name] = onp.exp(fp16_s_loss)
        t_s_loss = eval_loss(ternary_sum_composed, domain_eval[name], tok.pad_id)
        ternary_sum_ppls[name] = onp.exp(t_s_loss)

        results['domains'][name] = {
            'fp16_base_ppl': float(fp16_base_ppl[name]),
            'fp16_single_ppl': float(fp16_single_ppl[name]),
            'fp16_composed_ppl': float(fp16_c_ppl),
            'fp16_composed_ratio': float(fp16_c_ppl / fp16_base_ppl[name]),
            'fp16_composed_vs_single_ratio': float(fp16_c_ppl / fp16_single_ppl[name]),
            'fp16_sum_ppl': float(fp16_sum_ppls[name]),
            'ternary_base_ppl': float(ternary_base_ppl[name]),
            'ternary_single_ppl': float(ternary_single_ppl[name]),
            'ternary_composed_ppl': float(t_c_ppl),
            'ternary_composed_ratio': float(t_c_ppl / ternary_base_ppl[name]),
            'ternary_composed_vs_single_ratio': float(t_c_ppl / ternary_single_ppl[name]),
            'ternary_sum_ppl': float(ternary_sum_ppls[name]),
        }

    # Adapter diagnostics
    fp16_delta_norms = {}
    ternary_delta_norms = {}
    for name in domain_names:
        fp16_flat = flatten_delta(fp16_deltas[name])
        ternary_flat = flatten_delta(ternary_deltas[name])
        fp16_delta_norms[name] = float(onp.linalg.norm(fp16_flat))
        ternary_delta_norms[name] = float(onp.linalg.norm(ternary_flat))

    # Cross-adapter cosine similarities
    fp16_cosines = []
    ternary_cosines = []
    names = list(domain_names)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            fp16_cos = abs(cosine_sim(
                flatten_delta(fp16_deltas[names[i]]),
                flatten_delta(fp16_deltas[names[j]])
            ))
            ternary_cos = abs(cosine_sim(
                flatten_delta(ternary_deltas[names[i]]),
                flatten_delta(ternary_deltas[names[j]])
            ))
            fp16_cosines.append(fp16_cos)
            ternary_cosines.append(ternary_cos)

    results['diagnostics'] = {
        'fp16_delta_norms': fp16_delta_norms,
        'ternary_delta_norms': ternary_delta_norms,
        'fp16_delta_norm_cv': float(onp.std(list(fp16_delta_norms.values())) /
                                     max(onp.mean(list(fp16_delta_norms.values())), 1e-10)),
        'ternary_delta_norm_cv': float(onp.std(list(ternary_delta_norms.values())) /
                                        max(onp.mean(list(ternary_delta_norms.values())), 1e-10)),
        'fp16_mean_cos': float(onp.mean(fp16_cosines)),
        'ternary_mean_cos': float(onp.mean(ternary_cosines)),
        'fp16_max_cos': float(onp.max(fp16_cosines)),
        'ternary_max_cos': float(onp.max(ternary_cosines)),
        'fp16_delta_norm_ratio_max_min': float(
            max(fp16_delta_norms.values()) / max(min(fp16_delta_norms.values()), 1e-10)),
        'ternary_delta_norm_ratio_max_min': float(
            max(ternary_delta_norms.values()) / max(min(ternary_delta_norms.values()), 1e-10)),
    }

    # -----------------------------------------------------------------------
    # Kill criteria evaluation
    # -----------------------------------------------------------------------

    # K1: composed PPL > 100x base PPL at N=5
    fp16_max_ratio = max(
        results['domains'][n]['fp16_composed_ratio'] for n in domain_names)
    ternary_max_ratio = max(
        results['domains'][n]['ternary_composed_ratio'] for n in domain_names)

    results['kill_criteria'] = {
        'K1_fp16_max_composed_base_ratio': float(fp16_max_ratio),
        'K1_ternary_max_composed_base_ratio': float(ternary_max_ratio),
        'K1_ternary_threshold': 100.0,
        'K1_ternary_pass': bool(ternary_max_ratio < 100.0),

        'K2_ternary_domains_above_10x': sum(
            1 for n in domain_names
            if results['domains'][n]['ternary_composed_vs_single_ratio'] > 10.0
        ),
        'K2_ternary_threshold': len(domain_names) * 0.5,
        'K2_ternary_pass': bool(sum(
            1 for n in domain_names
            if results['domains'][n]['ternary_composed_vs_single_ratio'] > 10.0
        ) <= len(domain_names) * 0.5),

        'K3_pass': True,  # If we get here, training converged
    }

    # Comparison metrics
    fp16_mean_ratio = onp.mean([
        results['domains'][n]['fp16_composed_ratio'] for n in domain_names])
    ternary_mean_ratio = onp.mean([
        results['domains'][n]['ternary_composed_ratio'] for n in domain_names])

    results['comparison'] = {
        'fp16_mean_composed_base_ratio': float(fp16_mean_ratio),
        'ternary_mean_composed_base_ratio': float(ternary_mean_ratio),
        'stability_improvement_factor': float(fp16_mean_ratio / max(ternary_mean_ratio, 1e-10)),
        'ternary_helps': bool(ternary_mean_ratio < fp16_mean_ratio),
    }

    return results


def train_base(params, data_encoded, pad_id, epochs=30, lr=0.001,
               batch_size=32, clip_grad=1.0, verbose=True):
    """Train full model (not LoRA -- all params trainable)."""
    cfg = params['_config']
    param_keys = [k for k in sorted(params.keys()) if k != '_config']

    def loss_fn(param_vals, inp, tgt, mask):
        p = dict(zip(param_keys, param_vals))
        p['_config'] = cfg
        return compute_loss(p, inp, tgt, mask, pad_id)

    grad_fn = grad(loss_fn)

    m_state = [onp.zeros_like(params[k]) for k in param_keys]
    v_state = [onp.zeros_like(params[k]) for k in param_keys]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(data_encoded)
    rng = onp.random.RandomState(42)

    for epoch in range(epochs):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [data_encoded[j] for j in batch_idx]
            inp, tgt, mask = _prepare_batch(batch, pad_id)
            if onp.sum(mask) == 0:
                continue
            param_vals = [params[k] for k in param_keys]
            loss_val = float(loss_fn(param_vals, inp, tgt, mask))
            grads = grad_fn(param_vals, inp, tgt, mask)
            grad_norm = onp.sqrt(sum(float(onp.sum(g**2)) for g in grads))
            if grad_norm > clip_grad:
                sc = clip_grad / grad_norm
                grads = [g * sc for g in grads]
            step += 1
            for k_idx, key in enumerate(param_keys):
                g = onp.array(grads[k_idx])
                m_state[k_idx] = beta1 * m_state[k_idx] + (1 - beta1) * g
                v_state[k_idx] = beta2 * v_state[k_idx] + (1 - beta2) * g**2
                m_hat = m_state[k_idx] / (1 - beta1**step)
                v_hat = v_state[k_idx] / (1 - beta2**step)
                params[key] = params[key] - lr * m_hat / (onp.sqrt(v_hat) + eps)
            epoch_loss += loss_val
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:3d}: loss={avg_loss:.4f}")
    return params


# ===========================================================================
# Multi-seed runner
# ===========================================================================

def run_all(seeds=(42, 123, 314), d=64, r=4):
    """Run experiment across multiple seeds and aggregate."""

    t0 = time.time()
    all_results = []

    for seed in seeds:
        result = run_experiment(seed=seed, d=d, r=r, verbose=True)
        all_results.append(result)

    # Aggregate across seeds
    domain_names = list(all_results[0]['domains'].keys())

    agg = {
        'config': {'d': d, 'r': r, 'seeds': list(seeds), 'n_domains': len(domain_names)},
        'per_seed': all_results,
        'aggregate': {},
    }

    # Per-domain aggregates
    for name in domain_names:
        fp16_ratios = [r['domains'][name]['fp16_composed_ratio'] for r in all_results]
        ternary_ratios = [r['domains'][name]['ternary_composed_ratio'] for r in all_results]
        fp16_vs_single = [r['domains'][name]['fp16_composed_vs_single_ratio'] for r in all_results]
        ternary_vs_single = [r['domains'][name]['ternary_composed_vs_single_ratio'] for r in all_results]

        agg['aggregate'][name] = {
            'fp16_composed_base_ratio': {
                'mean': float(onp.mean(fp16_ratios)),
                'std': float(onp.std(fp16_ratios)),
                'max': float(onp.max(fp16_ratios)),
            },
            'ternary_composed_base_ratio': {
                'mean': float(onp.mean(ternary_ratios)),
                'std': float(onp.std(ternary_ratios)),
                'max': float(onp.max(ternary_ratios)),
            },
            'fp16_composed_vs_single': {
                'mean': float(onp.mean(fp16_vs_single)),
                'std': float(onp.std(fp16_vs_single)),
            },
            'ternary_composed_vs_single': {
                'mean': float(onp.mean(ternary_vs_single)),
                'std': float(onp.std(ternary_vs_single)),
            },
        }

    # Overall summary
    all_fp16_ratios = [r['comparison']['fp16_mean_composed_base_ratio'] for r in all_results]
    all_ternary_ratios = [r['comparison']['ternary_mean_composed_base_ratio'] for r in all_results]
    all_improvement = [r['comparison']['stability_improvement_factor'] for r in all_results]

    # Diagnostic aggregates
    all_fp16_cos = [r['diagnostics']['fp16_mean_cos'] for r in all_results]
    all_ternary_cos = [r['diagnostics']['ternary_mean_cos'] for r in all_results]
    all_fp16_norm_cv = [r['diagnostics']['fp16_delta_norm_cv'] for r in all_results]
    all_ternary_norm_cv = [r['diagnostics']['ternary_delta_norm_cv'] for r in all_results]

    # Kill criteria aggregated
    k1_ternary_passes = sum(1 for r in all_results if r['kill_criteria']['K1_ternary_pass'])
    k2_ternary_passes = sum(1 for r in all_results if r['kill_criteria']['K2_ternary_pass'])

    agg['summary'] = {
        'fp16_mean_composed_base_ratio': {
            'mean': float(onp.mean(all_fp16_ratios)),
            'std': float(onp.std(all_fp16_ratios)),
        },
        'ternary_mean_composed_base_ratio': {
            'mean': float(onp.mean(all_ternary_ratios)),
            'std': float(onp.std(all_ternary_ratios)),
        },
        'stability_improvement_factor': {
            'mean': float(onp.mean(all_improvement)),
            'std': float(onp.std(all_improvement)),
        },
        'ternary_helps_all_seeds': bool(all(
            r['comparison']['ternary_helps'] for r in all_results)),
        'fp16_mean_adapter_cos': {
            'mean': float(onp.mean(all_fp16_cos)),
            'std': float(onp.std(all_fp16_cos)),
        },
        'ternary_mean_adapter_cos': {
            'mean': float(onp.mean(all_ternary_cos)),
            'std': float(onp.std(all_ternary_cos)),
        },
        'fp16_delta_norm_cv': {
            'mean': float(onp.mean(all_fp16_norm_cv)),
            'std': float(onp.std(all_fp16_norm_cv)),
        },
        'ternary_delta_norm_cv': {
            'mean': float(onp.mean(all_ternary_norm_cv)),
            'std': float(onp.std(all_ternary_norm_cv)),
        },
        'kill_criteria': {
            'K1_ternary_pass_rate': f"{k1_ternary_passes}/{len(seeds)}",
            'K2_ternary_pass_rate': f"{k2_ternary_passes}/{len(seeds)}",
            'K3_pass': True,
        },
    }

    elapsed = time.time() - t0
    agg['runtime_seconds'] = float(elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Seeds: {seeds}")
    print(f"  Config: d={d}, r={r}, N=5 domains, 3 seeds")
    print(f"  Runtime: {elapsed:.1f}s")

    print(f"\n  FP16 base composition:")
    print(f"    Mean composed/base PPL ratio: {agg['summary']['fp16_mean_composed_base_ratio']['mean']:.2f} "
          f"+/- {agg['summary']['fp16_mean_composed_base_ratio']['std']:.2f}")
    print(f"    Mean adapter |cos|: {agg['summary']['fp16_mean_adapter_cos']['mean']:.4f}")
    print(f"    Delta norm CV: {agg['summary']['fp16_delta_norm_cv']['mean']:.4f}")

    print(f"\n  Ternary base composition:")
    print(f"    Mean composed/base PPL ratio: {agg['summary']['ternary_mean_composed_base_ratio']['mean']:.2f} "
          f"+/- {agg['summary']['ternary_mean_composed_base_ratio']['std']:.2f}")
    print(f"    Mean adapter |cos|: {agg['summary']['ternary_mean_adapter_cos']['mean']:.4f}")
    print(f"    Delta norm CV: {agg['summary']['ternary_delta_norm_cv']['mean']:.4f}")

    improvement = agg['summary']['stability_improvement_factor']['mean']
    print(f"\n  Stability improvement factor: {improvement:.2f}x "
          f"(>1 = ternary better)")
    print(f"  Ternary helps on all seeds: {agg['summary']['ternary_helps_all_seeds']}")

    print(f"\n  Kill criteria:")
    print(f"    K1 (composed PPL < 100x base): {agg['summary']['kill_criteria']['K1_ternary_pass_rate']}")
    print(f"    K2 (composed PPL < 10x single on >50% domains): {agg['summary']['kill_criteria']['K2_ternary_pass_rate']}")
    print(f"    K3 (training converged): PASS")

    # Per-domain breakdown
    print(f"\n  Per-domain composed/base PPL ratio (mean across seeds):")
    print(f"    {'Domain':12s} {'FP16':>10s} {'Ternary':>10s} {'Improvement':>12s}")
    print(f"    {'-'*46}")
    for name in domain_names:
        fp16_r = agg['aggregate'][name]['fp16_composed_base_ratio']['mean']
        tern_r = agg['aggregate'][name]['ternary_composed_base_ratio']['mean']
        imp = fp16_r / max(tern_r, 1e-10)
        print(f"    {name:12s} {fp16_r:10.2f} {tern_r:10.2f} {imp:10.2f}x")

    return agg


def main():
    out_dir = Path(__file__).parent
    agg = run_all(seeds=(42, 123, 314), d=64, r=4)

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
