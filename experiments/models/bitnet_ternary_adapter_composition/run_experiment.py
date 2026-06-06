#!/usr/bin/env python3
"""
BitLoRA Ternary Adapter Composition: Do ternary LoRA adapters compose better
than FP16 LoRA on a ternary (BitNet-style) base model?

Builds directly on bitnet_composition_stability.py. Reuses the same:
  - Micro transformer architecture (d=64, r=4, L=2)
  - 5 toy domain data generators
  - Training infrastructure (Adam, grad clipping, etc.)

NEW: Three adapter quantization conditions during training:
  (a) FP16 LoRA -- standard, continuous weights (baseline)
  (b) Ternary LoRA -- QAT with STE, forward pass quantized to {-1,0,1}*alpha
  (c) INT4 LoRA -- QAT with STE, forward pass quantized to 4-bit

All conditions use the SAME ternary base model (post-quantized from FP16).

Kill criteria:
  K1: ternary adapter individual PPL > 1.05x FP16 adapter individual PPL (>5% worse)
  K2: ternary composition PPL > FP16 composition PPL (ternary does not help)
  K3: ternary adapters fail to converge (loss > 2x FP16 loss)
"""

import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp


# ===========================================================================
# Synthetic Data: 5 domains (same as bitnet_composition_stability)
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
# Model
# ===========================================================================

def init_model(V, d=64, H=2, L=2, max_T=32, seed=42):
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


def ternary_quantize_weight(W):
    """BitNet b1.58 absmean quantization."""
    alpha = onp.mean(onp.abs(W))
    if alpha < 1e-10:
        return onp.zeros_like(W), alpha
    W_scaled = W / alpha
    W_ternary = onp.clip(onp.round(W_scaled), -1, 1).astype(onp.float32)
    return W_ternary, alpha


def quantize_model_to_ternary(params):
    """Quantize all weight matrices to ternary {-1, 0, 1} * alpha."""
    ternary_params = {}
    scales = {}
    fp32_keys = {'tok_emb', 'pos_emb', 'ln_f_w', '_config'}
    fp32_keys.update(k for k in params if k.startswith('ln'))
    for k, v in params.items():
        if k in fp32_keys or k == '_config':
            ternary_params[k] = v if k == '_config' else v.copy()
        elif v.ndim >= 2:
            W_t, alpha = ternary_quantize_weight(v)
            ternary_params[k] = W_t * alpha
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
# Quantization functions for adapter QAT (differentiable via STE)
#
# CRITICAL: autograd's np.round and np.clip have ZERO derivatives.
# We implement STE by computing quantized values with plain numpy (constants),
# then computing: W + (W_q_const - W_const) where _const values are treated
# as constants by autograd. This gives:
#   Forward: W + (W_q - W) = W_q  (correct quantized value)
#   Backward: dW/dW = 1  (STE, gradients pass through)
# ===========================================================================

def _get_value(W):
    """Extract raw numpy array from autograd ArrayBox or return as-is."""
    if hasattr(W, '_value'):
        return onp.array(W._value)
    return onp.array(W)


def ternary_quantize_ste(W):
    """Ternary quantization with proper STE for autograd.

    Forward: returns quantized value {-1, 0, 1} * alpha
    Backward: gradient passes through W unchanged (STE)
    """
    # Extract raw values (detach from autograd computation graph)
    W_np = _get_value(W)
    alpha = float(onp.mean(onp.abs(W_np))) + 1e-10
    W_scaled = W_np / alpha
    W_q_np = onp.clip(onp.round(W_scaled), -1, 1).astype(onp.float32) * alpha

    # STE trick: W + (W_q - W) where (W_q - W) is a constant
    # autograd sees: W + constant, so dL/dW flows through
    residual = W_q_np - W_np  # plain numpy constant
    return W + residual  # autograd differentiates W, residual is constant


def int4_quantize_ste(W):
    """INT4 quantization with proper STE for autograd."""
    W_np = _get_value(W)
    scale = float(onp.max(onp.abs(W_np))) / 7.0 + 1e-10
    W_scaled = W_np / scale
    W_q_np = onp.clip(onp.round(W_scaled), -8, 7).astype(onp.float32) * scale

    residual = W_q_np - W_np
    return W + residual


# ===========================================================================
# LoRA with optional quantization
# ===========================================================================

def init_lora(params, rank=4, seed=42):
    """Initialize LoRA A/B matrices for all weight matrices."""
    rng = onp.random.RandomState(seed)
    lora = {}
    for k, v in params.items():
        if k == '_config' or v.ndim < 2:
            continue
        d_in, d_out = v.shape
        lora[f'{k}_A'] = rng.randn(d_in, rank).astype(onp.float32) * 0.01
        lora[f'{k}_B'] = onp.zeros((rank, d_out), dtype=onp.float32)
    return lora


def apply_lora(base_params, lora, alpha_over_r=1.0, quant_mode='fp16'):
    """Create effective params by adding LoRA deltas to frozen base.

    quant_mode: 'fp16' (no quantization), 'ternary' (QAT), 'int4' (QAT)
    """
    result = {}
    for k, v in base_params.items():
        if k == '_config':
            result[k] = v
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            A = lora[A_key]
            B = lora[B_key]
            if quant_mode == 'ternary':
                A = ternary_quantize_ste(A)
                B = ternary_quantize_ste(B)
            elif quant_mode == 'int4':
                A = int4_quantize_ste(A)
                B = int4_quantize_ste(B)
            delta = np.dot(A, B) * alpha_over_r
            result[k] = v + delta
        else:
            result[k] = v
    return result


def train_lora(base_params, lora, data_encoded, pad_id, epochs=30, lr=0.003,
               batch_size=32, clip_grad=1.0, quant_mode='fp16', verbose=True):
    """Train LoRA params with optional quantization-aware training."""
    lora_keys = sorted(lora.keys())

    def loss_fn(lora_vals, inp, tgt, mask):
        lo = dict(zip(lora_keys, lora_vals))
        effective = apply_lora(base_params, lo, quant_mode=quant_mode)
        return compute_loss(effective, inp, tgt, mask, pad_id)

    grad_fn = grad(loss_fn)

    m_state = [onp.zeros_like(lora[k]) for k in lora_keys]
    v_state = [onp.zeros_like(lora[k]) for k in lora_keys]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(data_encoded)
    rng = onp.random.RandomState(42)
    final_loss = float('inf')

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
        final_loss = avg_loss
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"        epoch {epoch:3d}: loss={avg_loss:.4f} [{quant_mode}]")

    return lora, final_loss


def lora_to_delta(lora, base_params, alpha_over_r=1.0, quant_mode='fp16'):
    """Convert LoRA A/B matrices to weight deltas, applying quantization if needed."""
    delta = {}
    for k in base_params:
        if k == '_config' or base_params[k].ndim < 2:
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            A = onp.array(lora[A_key])
            B = onp.array(lora[B_key])
            if quant_mode == 'ternary':
                A = _np_ternary_quantize(A)
                B = _np_ternary_quantize(B)
            elif quant_mode == 'int4':
                A = _np_int4_quantize(A)
                B = _np_int4_quantize(B)
            delta[k] = onp.dot(A, B) * alpha_over_r
    return delta


def _np_ternary_quantize(W):
    """Numpy (non-autograd) ternary quantization."""
    alpha = onp.mean(onp.abs(W)) + 1e-10
    W_scaled = W / alpha
    W_q = onp.clip(onp.round(W_scaled), -1, 1) * alpha
    return W_q


def _np_int4_quantize(W):
    """Numpy (non-autograd) INT4 quantization."""
    scale = onp.max(onp.abs(W)) / 7.0 + 1e-10
    W_q = onp.clip(onp.round(W / scale), -8, 7) * scale
    return W_q


# ===========================================================================
# Batch preparation & eval
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


def train_base(params, data_encoded, pad_id, epochs=30, lr=0.001,
               batch_size=32, clip_grad=1.0, verbose=True):
    """Train full model (not LoRA)."""
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
# Adapter diagnostics
# ===========================================================================

def measure_adapter_stats(lora, base_params, quant_mode='fp16'):
    """Measure quantization-specific statistics of adapter weights."""
    stats = {}
    for k in base_params:
        if k == '_config' or base_params[k].ndim < 2:
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key not in lora:
            continue
        A = onp.array(lora[A_key])
        B = onp.array(lora[B_key])

        # Quantize for measurement
        if quant_mode == 'ternary':
            A_q = _np_ternary_quantize(A)
            B_q = _np_ternary_quantize(B)
        elif quant_mode == 'int4':
            A_q = _np_int4_quantize(A)
            B_q = _np_int4_quantize(B)
        else:
            A_q, B_q = A, B

        # Quantization error
        A_err = onp.linalg.norm(A_q - A) / max(onp.linalg.norm(A), 1e-10)
        B_err = onp.linalg.norm(B_q - B) / max(onp.linalg.norm(B), 1e-10)

        stats[k] = {
            'A_quant_error': float(A_err),
            'B_quant_error': float(B_err),
            'A_norm': float(onp.linalg.norm(A)),
            'B_norm': float(onp.linalg.norm(B)),
            'A_unique_vals': int(len(onp.unique(onp.round(A_q, 6)))),
            'B_unique_vals': int(len(onp.unique(onp.round(B_q, 6)))),
        }
    return stats


def count_adapter_bits(lora, quant_mode):
    """Estimate adapter storage in bits."""
    total_params = 0
    for k in lora:
        total_params += lora[k].size
    if quant_mode == 'fp16':
        return total_params * 16  # FP16
    elif quant_mode == 'ternary':
        # 1.58 bits per param + scale factors (32 bits each, 2 per matrix pair)
        n_matrices = len(lora) // 2
        return int(total_params * 1.58 + n_matrices * 2 * 32)
    elif quant_mode == 'int4':
        n_matrices = len(lora) // 2
        return total_params * 4 + n_matrices * 32  # 4 bits + scale
    return total_params * 32


# ===========================================================================
# Main experiment
# ===========================================================================

QUANT_MODES = ['fp16', 'ternary', 'int4']


def run_experiment(seed=42, d=64, r=4, L=2, H=2, n_data=300, n_eval=100,
                   base_epochs=30, lora_epochs=30, verbose=True):
    """Run one seed of the ternary adapter composition experiment."""

    print(f"\n{'='*70}")
    print(f"  BitLoRA Ternary Adapter Composition (seed={seed}, d={d}, r={r})")
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

    # Evaluate base models
    print("\n  Base model eval (before LoRA):")
    base_ppl = {}
    for name in domain_names:
        loss = eval_loss(ternary_base, domain_eval[name], tok.pad_id)
        base_ppl[name] = float(onp.exp(loss))
    print(f"    Ternary base mean PPL: {onp.mean(list(base_ppl.values())):.2f}")

    # -----------------------------------------------------------------------
    # Step 4: Train LoRA adapters for each quant mode
    # -----------------------------------------------------------------------
    all_loras = {mode: {} for mode in QUANT_MODES}
    all_deltas = {mode: {} for mode in QUANT_MODES}
    all_single_ppl = {mode: {} for mode in QUANT_MODES}
    all_final_loss = {mode: {} for mode in QUANT_MODES}

    for di, name in enumerate(domain_names):
        lora_seed = seed * 100 + di

        for mode in QUANT_MODES:
            print(f"\n[4/6] Training {mode} LoRA on ternary base: {name} "
                  f"(seed={lora_seed})...")
            lora = init_lora(ternary_base, rank=r, seed=lora_seed)
            lora, final_loss = train_lora(
                ternary_base, lora, domain_data[name], tok.pad_id,
                epochs=lora_epochs, lr=0.003, quant_mode=mode, verbose=verbose
            )
            all_loras[mode][name] = lora
            all_final_loss[mode][name] = final_loss

            # Convert to delta (with quantization applied)
            delta = lora_to_delta(lora, ternary_base, quant_mode=mode)
            all_deltas[mode][name] = delta

            # Eval single adapter
            effective = apply_delta(ternary_base, delta)
            single_loss = eval_loss(effective, domain_eval[name], tok.pad_id)
            all_single_ppl[mode][name] = float(onp.exp(single_loss))

    # -----------------------------------------------------------------------
    # Step 5: Equal-weight composition for each quant mode
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Composing all {len(domain_names)} adapters (equal weight)...")
    all_composed_ppl = {mode: {} for mode in QUANT_MODES}

    for mode in QUANT_MODES:
        merged_delta = compose_deltas(list(all_deltas[mode].values()), mode='equal')
        composed = apply_delta(ternary_base, merged_delta)

        for name in domain_names:
            c_loss = eval_loss(composed, domain_eval[name], tok.pad_id)
            all_composed_ppl[mode][name] = float(onp.exp(c_loss))

    # -----------------------------------------------------------------------
    # Step 6: Diagnostics and results
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Computing diagnostics...")

    results = {
        'seed': seed, 'd': d, 'r': r, 'L': L,
        'n_domains': len(domain_names),
        'base_ppl': base_ppl,
        'conditions': {},
    }

    for mode in QUANT_MODES:
        mode_results = {
            'domains': {},
            'diagnostics': {},
        }

        # Per-domain results
        for name in domain_names:
            mode_results['domains'][name] = {
                'base_ppl': base_ppl[name],
                'single_ppl': all_single_ppl[mode][name],
                'composed_ppl': all_composed_ppl[mode][name],
                'composed_base_ratio': all_composed_ppl[mode][name] / base_ppl[name],
                'composed_single_ratio': all_composed_ppl[mode][name] / all_single_ppl[mode][name],
                'final_train_loss': all_final_loss[mode][name],
            }

        # Cross-adapter cosine similarities
        delta_norms = {}
        cosines = []
        names = list(domain_names)
        for name in names:
            flat = flatten_delta(all_deltas[mode][name])
            delta_norms[name] = float(onp.linalg.norm(flat))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                cos = abs(cosine_sim(
                    flatten_delta(all_deltas[mode][names[i]]),
                    flatten_delta(all_deltas[mode][names[j]])
                ))
                cosines.append(cos)

        norm_vals = list(delta_norms.values())
        mode_results['diagnostics'] = {
            'delta_norms': delta_norms,
            'delta_norm_mean': float(onp.mean(norm_vals)),
            'delta_norm_cv': float(onp.std(norm_vals) / max(onp.mean(norm_vals), 1e-10)),
            'delta_norm_max_min_ratio': float(max(norm_vals) / max(min(norm_vals), 1e-10)),
            'mean_cos': float(onp.mean(cosines)),
            'max_cos': float(onp.max(cosines)),
            'adapter_bits': count_adapter_bits(all_loras[mode][names[0]], mode),
        }

        # Adapter quantization error stats (for ternary/int4)
        if mode in ('ternary', 'int4'):
            quant_errors = []
            for name in names:
                stats = measure_adapter_stats(all_loras[mode][name], ternary_base, mode)
                for k, v in stats.items():
                    quant_errors.append(v['A_quant_error'])
                    quant_errors.append(v['B_quant_error'])
            mode_results['diagnostics']['mean_quant_error'] = float(onp.mean(quant_errors))
            mode_results['diagnostics']['max_quant_error'] = float(onp.max(quant_errors))

        results['conditions'][mode] = mode_results

    # -----------------------------------------------------------------------
    # Kill criteria evaluation
    # -----------------------------------------------------------------------
    fp16_single_ppls = [all_single_ppl['fp16'][n] for n in domain_names]
    ternary_single_ppls = [all_single_ppl['ternary'][n] for n in domain_names]
    int4_single_ppls = [all_single_ppl['int4'][n] for n in domain_names]

    fp16_composed_ppls = [all_composed_ppl['fp16'][n] for n in domain_names]
    ternary_composed_ppls = [all_composed_ppl['ternary'][n] for n in domain_names]
    int4_composed_ppls = [all_composed_ppl['int4'][n] for n in domain_names]

    # K1: ternary individual quality < 1.05x FP16
    individual_ratio = onp.mean(ternary_single_ppls) / onp.mean(fp16_single_ppls)
    k1_pass = individual_ratio <= 1.05

    # K2: ternary composition PPL < FP16 composition PPL
    ternary_mean_composed = onp.mean(ternary_composed_ppls)
    fp16_mean_composed = onp.mean(fp16_composed_ppls)
    k2_pass = ternary_mean_composed <= fp16_mean_composed

    # K3: convergence (ternary final loss < 2x FP16 final loss)
    fp16_losses = [all_final_loss['fp16'][n] for n in domain_names]
    ternary_losses = [all_final_loss['ternary'][n] for n in domain_names]
    loss_ratio = onp.mean(ternary_losses) / onp.mean(fp16_losses)
    k3_pass = loss_ratio < 2.0

    results['kill_criteria'] = {
        'K1_individual_ratio': float(individual_ratio),
        'K1_threshold': 1.05,
        'K1_pass': bool(k1_pass),
        'K2_ternary_mean_composed': float(ternary_mean_composed),
        'K2_fp16_mean_composed': float(fp16_mean_composed),
        'K2_pass': bool(k2_pass),
        'K3_convergence_ratio': float(loss_ratio),
        'K3_threshold': 2.0,
        'K3_pass': bool(k3_pass),
    }

    # Summary comparisons
    results['comparison'] = {
        'fp16_mean_single_ppl': float(onp.mean(fp16_single_ppls)),
        'ternary_mean_single_ppl': float(onp.mean(ternary_single_ppls)),
        'int4_mean_single_ppl': float(onp.mean(int4_single_ppls)),
        'fp16_mean_composed_ppl': float(fp16_mean_composed),
        'ternary_mean_composed_ppl': float(ternary_mean_composed),
        'int4_mean_composed_ppl': float(onp.mean(int4_composed_ppls)),
        'fp16_mean_composed_base_ratio': float(onp.mean([
            all_composed_ppl['fp16'][n] / base_ppl[n] for n in domain_names])),
        'ternary_mean_composed_base_ratio': float(onp.mean([
            all_composed_ppl['ternary'][n] / base_ppl[n] for n in domain_names])),
        'int4_mean_composed_base_ratio': float(onp.mean([
            all_composed_ppl['int4'][n] / base_ppl[n] for n in domain_names])),
        'fp16_adapter_bits': results['conditions']['fp16']['diagnostics']['adapter_bits'],
        'ternary_adapter_bits': results['conditions']['ternary']['diagnostics']['adapter_bits'],
        'int4_adapter_bits': results['conditions']['int4']['diagnostics']['adapter_bits'],
        'ternary_compression_ratio': (
            results['conditions']['fp16']['diagnostics']['adapter_bits'] /
            max(results['conditions']['ternary']['diagnostics']['adapter_bits'], 1)
        ),
        'int4_compression_ratio': (
            results['conditions']['fp16']['diagnostics']['adapter_bits'] /
            max(results['conditions']['int4']['diagnostics']['adapter_bits'], 1)
        ),
    }

    return results


def run_all(seeds=(42, 123, 314), d=64, r=4):
    """Run experiment across multiple seeds and aggregate."""

    t0 = time.time()
    all_results = []

    for seed in seeds:
        result = run_experiment(seed=seed, d=d, r=r, verbose=True)
        all_results.append(result)

    domain_names = list(DOMAIN_GENERATORS.keys())

    # Aggregate
    agg = {
        'config': {'d': d, 'r': r, 'seeds': list(seeds), 'n_domains': len(domain_names)},
        'per_seed': all_results,
    }

    # Aggregate per condition
    agg['aggregate'] = {}
    for mode in QUANT_MODES:
        mode_agg = {}

        # Individual PPL across seeds
        single_ppls = []
        composed_ppls = []
        composed_base_ratios = []
        mean_cos_list = []
        delta_norm_cv_list = []

        for r_seed in all_results:
            cond = r_seed['conditions'][mode]
            single_ppls.append(onp.mean([
                cond['domains'][n]['single_ppl'] for n in domain_names]))
            composed_ppls.append(onp.mean([
                cond['domains'][n]['composed_ppl'] for n in domain_names]))
            composed_base_ratios.append(onp.mean([
                cond['domains'][n]['composed_base_ratio'] for n in domain_names]))
            mean_cos_list.append(cond['diagnostics']['mean_cos'])
            delta_norm_cv_list.append(cond['diagnostics']['delta_norm_cv'])

        mode_agg['mean_single_ppl'] = {
            'mean': float(onp.mean(single_ppls)),
            'std': float(onp.std(single_ppls)),
        }
        mode_agg['mean_composed_ppl'] = {
            'mean': float(onp.mean(composed_ppls)),
            'std': float(onp.std(composed_ppls)),
        }
        mode_agg['mean_composed_base_ratio'] = {
            'mean': float(onp.mean(composed_base_ratios)),
            'std': float(onp.std(composed_base_ratios)),
        }
        mode_agg['mean_cos'] = {
            'mean': float(onp.mean(mean_cos_list)),
            'std': float(onp.std(mean_cos_list)),
        }
        mode_agg['delta_norm_cv'] = {
            'mean': float(onp.mean(delta_norm_cv_list)),
            'std': float(onp.std(delta_norm_cv_list)),
        }
        mode_agg['adapter_bits'] = all_results[0]['conditions'][mode]['diagnostics']['adapter_bits']

        agg['aggregate'][mode] = mode_agg

    # Kill criteria across seeds
    k1_passes = sum(1 for r in all_results if r['kill_criteria']['K1_pass'])
    k2_passes = sum(1 for r in all_results if r['kill_criteria']['K2_pass'])
    k3_passes = sum(1 for r in all_results if r['kill_criteria']['K3_pass'])

    k1_ratios = [r['kill_criteria']['K1_individual_ratio'] for r in all_results]
    k2_ternary = [r['kill_criteria']['K2_ternary_mean_composed'] for r in all_results]
    k2_fp16 = [r['kill_criteria']['K2_fp16_mean_composed'] for r in all_results]
    k3_ratios = [r['kill_criteria']['K3_convergence_ratio'] for r in all_results]

    agg['kill_criteria'] = {
        'K1_individual_ratio': {
            'mean': float(onp.mean(k1_ratios)),
            'std': float(onp.std(k1_ratios)),
        },
        'K1_pass_rate': f"{k1_passes}/{len(seeds)}",
        'K2_ternary_mean_composed': {
            'mean': float(onp.mean(k2_ternary)),
            'std': float(onp.std(k2_ternary)),
        },
        'K2_fp16_mean_composed': {
            'mean': float(onp.mean(k2_fp16)),
            'std': float(onp.std(k2_fp16)),
        },
        'K2_pass_rate': f"{k2_passes}/{len(seeds)}",
        'K3_convergence_ratio': {
            'mean': float(onp.mean(k3_ratios)),
            'std': float(onp.std(k3_ratios)),
        },
        'K3_pass_rate': f"{k3_passes}/{len(seeds)}",
    }

    elapsed = time.time() - t0
    agg['runtime_seconds'] = float(elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Seeds: {seeds}")
    print(f"  Config: d={d}, r={r}, N=5 domains, {len(seeds)} seeds")
    print(f"  Runtime: {elapsed:.1f}s")

    print(f"\n  {'Condition':12s} {'Single PPL':>12s} {'Composed PPL':>14s} "
          f"{'Comp/Base':>10s} {'|cos|':>8s} {'Norm CV':>8s} {'Bits':>8s}")
    print(f"  {'-'*74}")
    for mode in QUANT_MODES:
        m = agg['aggregate'][mode]
        print(f"  {mode:12s} "
              f"{m['mean_single_ppl']['mean']:12.2f} "
              f"{m['mean_composed_ppl']['mean']:14.2f} "
              f"{m['mean_composed_base_ratio']['mean']:10.3f} "
              f"{m['mean_cos']['mean']:8.4f} "
              f"{m['delta_norm_cv']['mean']:8.4f} "
              f"{m['adapter_bits']:8d}")

    print(f"\n  Kill Criteria:")
    print(f"    K1 (individual quality): ratio={agg['kill_criteria']['K1_individual_ratio']['mean']:.4f} "
          f"(threshold 1.05, pass={agg['kill_criteria']['K1_pass_rate']})")
    print(f"    K2 (composition quality): ternary={agg['kill_criteria']['K2_ternary_mean_composed']['mean']:.3f} "
          f"vs fp16={agg['kill_criteria']['K2_fp16_mean_composed']['mean']:.3f} "
          f"(pass={agg['kill_criteria']['K2_pass_rate']})")
    print(f"    K3 (convergence): ratio={agg['kill_criteria']['K3_convergence_ratio']['mean']:.4f} "
          f"(threshold 2.0, pass={agg['kill_criteria']['K3_pass_rate']})")

    # Per-domain breakdown
    print(f"\n  Per-domain composed PPL (mean across seeds):")
    print(f"    {'Domain':12s} {'FP16':>10s} {'Ternary':>10s} {'INT4':>10s} {'Base':>10s}")
    print(f"    {'-'*54}")
    for name in domain_names:
        fp16_vals = [r['conditions']['fp16']['domains'][name]['composed_ppl'] for r in all_results]
        tern_vals = [r['conditions']['ternary']['domains'][name]['composed_ppl'] for r in all_results]
        int4_vals = [r['conditions']['int4']['domains'][name]['composed_ppl'] for r in all_results]
        base_vals = [r['base_ppl'][name] for r in all_results]
        print(f"    {name:12s} {onp.mean(fp16_vals):10.2f} {onp.mean(tern_vals):10.2f} "
              f"{onp.mean(int4_vals):10.2f} {onp.mean(base_vals):10.2f}")

    return agg


def main():
    out_dir = Path(__file__).parent
    agg = run_all(seeds=(42, 123, 314), d=64, r=4)

    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
