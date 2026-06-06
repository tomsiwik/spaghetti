#!/usr/bin/env python3
"""
MoTE-SOLE Architecture: FP16 shared base + ternary routed experts with top-k gating.

Compares three composition strategies at micro scale (d=64, r=4, L=2, 5 domains):
  (a) Equal-weight 1/N composition (baseline from prior experiments)
  (b) MoTE-style top-k routed composition with learned linear router
  (c) Learned-weight composition (router softmax weights, k=N -- all experts)

The MoTE architecture:
  - Frozen FP16 base model acts as "shared expert"
  - Ternary {-1,0,1} domain experts via QAT with STE
  - Linear router: h @ W_r -> softmax -> top-k selection
  - Load-balancing loss to prevent expert collapse

Kill criteria:
  K1: MoTE-SOLE quality < equal-weight on >50% of domains
  K2: Router training requires >1hr on CPU
  K3: Ternary experts lose >10% quality vs FP16 individually

References:
  - MoTE (arXiv 2506.14435): Mixture of Ternary Experts
  - Switch Transformer: load-balancing loss
  - Prior: exp_bitnet_ternary_adapter_composition (SUPPORTED)
  - Prior: exp_bitnet_composition_stability (SUPPORTED)
"""

import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp


# ===========================================================================
# Synthetic Data: 5 domains (reused from prior BitNet experiments)
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


def forward_with_hidden(params, idx_2d, pad_id=0):
    """Forward pass returning both logits and final hidden states (for router)."""
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
    h_final = _rms_norm(x, params['ln_f_w'])
    logits = np.dot(h_final, params['W_head'])
    return logits, h_final


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
# Quantization (STE for autograd)
# ===========================================================================

def _get_value(W):
    if hasattr(W, '_value'):
        return onp.array(W._value)
    return onp.array(W)


def ternary_quantize_ste(W):
    """Ternary quantization with STE for autograd."""
    W_np = _get_value(W)
    alpha = float(onp.mean(onp.abs(W_np))) + 1e-10
    W_scaled = W_np / alpha
    W_q_np = onp.clip(onp.round(W_scaled), -1, 1).astype(onp.float32) * alpha
    residual = W_q_np - W_np
    return W + residual


def _np_ternary_quantize(W):
    """Numpy (non-autograd) ternary quantization."""
    alpha = onp.mean(onp.abs(W)) + 1e-10
    W_scaled = W / alpha
    W_q = onp.clip(onp.round(W_scaled), -1, 1) * alpha
    return W_q


def ternary_quantize_base(W):
    """BitNet b1.58 absmean quantization for base model."""
    alpha = onp.mean(onp.abs(W))
    if alpha < 1e-10:
        return onp.zeros_like(W), alpha
    W_scaled = W / alpha
    W_ternary = onp.clip(onp.round(W_scaled), -1, 1).astype(onp.float32)
    return W_ternary, alpha


def quantize_model_to_ternary(params):
    """Quantize all weight matrices to ternary."""
    ternary_params = {}
    scales = {}
    fp32_keys = {'tok_emb', 'pos_emb', 'ln_f_w', '_config'}
    fp32_keys.update(k for k in params if k.startswith('ln'))
    for k, v in params.items():
        if k in fp32_keys or k == '_config':
            ternary_params[k] = v if k == '_config' else v.copy()
        elif v.ndim >= 2:
            W_t, alpha = ternary_quantize_base(v)
            ternary_params[k] = W_t * alpha
            scales[k] = alpha
        else:
            ternary_params[k] = v.copy()
    return ternary_params, scales


# ===========================================================================
# LoRA + Training
# ===========================================================================

def init_lora(params, rank=4, seed=42):
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
            delta = np.dot(A, B) * alpha_over_r
            result[k] = v + delta
        else:
            result[k] = v
    return result


def lora_to_delta(lora, base_params, alpha_over_r=1.0, quant_mode='fp16'):
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
            delta[k] = onp.dot(A, B) * alpha_over_r
    return delta


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


def train_base(params, data_encoded, pad_id, epochs=30, lr=0.001,
               batch_size=32, clip_grad=1.0, verbose=True):
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


def train_lora(base_params, lora, data_encoded, pad_id, epochs=30, lr=0.003,
               batch_size=32, clip_grad=1.0, quant_mode='fp16', verbose=True):
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


def compose_deltas_equal(delta_list):
    """Equal-weight 1/N composition."""
    merged = {}
    N = len(delta_list)
    for k in delta_list[0]:
        merged[k] = sum(d[k] for d in delta_list) / N
    return merged


def compose_deltas_weighted(delta_list, weights):
    """Weighted composition: sum(w_i * delta_i)."""
    merged = {}
    for k in delta_list[0]:
        merged[k] = sum(w * d[k] for w, d in zip(weights, delta_list))
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
# Router: MoTE-style top-k gating
# ===========================================================================

def init_router(d, N, seed=42):
    """Initialize a linear router: h -> expert scores."""
    rng = onp.random.RandomState(seed)
    return {
        'W_r': rng.randn(d, N).astype(onp.float32) * 0.01,
        'b_r': onp.zeros(N, dtype=onp.float32),
    }


def router_forward(router_params, hidden_states):
    """Compute router probabilities from hidden states.

    Args:
        router_params: dict with W_r (d, N) and b_r (N,)
        hidden_states: (B, T, d) hidden states from base model

    Returns:
        probs: (B, T, N) softmax probabilities over experts
    """
    logits = np.dot(hidden_states, router_params['W_r']) + router_params['b_r']
    # Stable softmax
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs


def top_k_mask(probs_np, k):
    """Create a binary mask for top-k selection (numpy, non-differentiable).

    Args:
        probs_np: (B, T, N) probabilities as numpy array
        k: number of experts to select

    Returns:
        mask: (B, T, N) binary mask, 1 for selected experts
    """
    B, T, N = probs_np.shape
    mask = onp.zeros_like(probs_np)
    for b in range(B):
        for t in range(T):
            top_idx = onp.argsort(probs_np[b, t])[-k:]
            mask[b, t, top_idx] = 1.0
    return mask


def compute_load_balance_loss(probs, top_k_mask_np, N):
    """Load-balancing loss from Switch Transformer.

    L_balance = N * sum_i(f_i * P_i)
    where f_i = fraction of tokens assigned to expert i (from hard top-k)
          P_i = mean probability for expert i (differentiable)
    """
    # f_i: fraction of tokens where expert i is in top-k (constant)
    B_T = probs.shape[0] * probs.shape[1]
    f = onp.sum(top_k_mask_np, axis=(0, 1)) / B_T  # (N,)

    # P_i: mean probability for expert i across all tokens (differentiable)
    P = np.mean(np.mean(probs, axis=0), axis=0)  # (N,)

    return N * np.sum(f * P)


def routed_compose_forward(base_params, deltas, router_params, idx_2d,
                           pad_id, k, return_routing_info=False):
    """Forward pass with MoTE-style routed composition.

    1. Run base model to get hidden states
    2. Router selects top-k experts per token
    3. Compose expert deltas weighted by router probabilities
    4. Re-run with composed params (or approximate via output-level mixing)

    For efficiency at micro scale, we use the "pre-merge" approach:
    compute router weights, then build a single composed delta, then
    do one forward pass with composed weights.

    Since router weights vary per token, we use SEQUENCE-LEVEL routing:
    average hidden states over sequence -> one routing decision per sequence.
    This matches the SOLE hash-ring routing granularity (per-query, not per-token).
    """
    N = len(deltas)
    domain_names = list(deltas.keys())

    # Step 1: Get hidden states from base model
    _, h_final = forward_with_hidden(base_params, idx_2d, pad_id)
    # h_final: (B, T, d)

    # Sequence-level routing: average over time
    # Use mean pooling (exclude padding)
    pad_mask = (idx_2d != pad_id).astype(onp.float32)  # (B, T)
    pad_mask_expanded = pad_mask[:, :, None]  # (B, T, 1)
    h_seq = np.sum(h_final * pad_mask_expanded, axis=1) / (np.sum(pad_mask_expanded, axis=1) + 1e-10)
    # h_seq: (B, d)

    # Step 2: Router
    logits = np.dot(h_seq, router_params['W_r']) + router_params['b_r']  # (B, N)
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (B, N)

    # Step 3: Top-k selection (STE: use hard mask in forward, soft in backward)
    probs_np = onp.array(_get_value(probs))
    B = probs_np.shape[0]
    mask_np = onp.zeros_like(probs_np)  # (B, N)
    for b in range(B):
        top_idx = onp.argsort(probs_np[b])[-k:]
        mask_np[b, top_idx] = 1.0

    # STE: probs * mask_const (mask is treated as constant by autograd)
    # Renormalize selected experts
    selected_probs = probs * mask_np  # (B, N) -- autograd differentiates probs
    renorm = np.sum(selected_probs, axis=-1, keepdims=True) + 1e-10
    weights = selected_probs / renorm  # (B, N) -- normalized weights

    # Step 4: For pre-merge, we need a single delta per batch element.
    # Since our batch typically has same-domain data, we average weights across B.
    avg_weights = np.mean(weights, axis=0)  # (N,)

    # Build composed delta
    delta_list = [deltas[name] for name in domain_names]
    merged = {}
    for key in delta_list[0]:
        weighted_sum = sum(
            avg_weights[i] * delta_list[i][key]
            for i in range(N)
        )
        merged[key] = weighted_sum

    # Step 5: Forward with composed params
    composed_params = {}
    for kk in base_params:
        if kk == '_config':
            composed_params[kk] = base_params[kk]
        elif kk in merged:
            composed_params[kk] = base_params[kk] + merged[kk]
        else:
            composed_params[kk] = base_params[kk]

    logits_out = forward(composed_params, idx_2d, pad_id)

    if return_routing_info:
        return logits_out, probs, mask_np, avg_weights
    return logits_out


def train_router(base_params, deltas, router_params, mixed_data,
                 domain_labels, pad_id, epochs=20, lr=0.01, k=2,
                 alpha_balance=0.01, batch_size=32, clip_grad=1.0,
                 verbose=True):
    """Train the router on mixed-domain data.

    The router learns to route each input to the best expert(s).
    Training signal: NTP loss + load-balancing loss.

    domain_labels: list of int, domain index for each example in mixed_data.
    """
    router_keys = sorted(router_params.keys())
    N = len(deltas)
    domain_names = list(deltas.keys())

    # Convert deltas to list (ordered)
    delta_list = [deltas[name] for name in domain_names]

    def loss_fn(router_vals, inp, tgt, mask):
        rp = dict(zip(router_keys, router_vals))

        # Get hidden states from base model
        _, h_final = forward_with_hidden(base_params, inp, pad_id)

        # Sequence-level routing
        pad_m = (inp != pad_id).astype(onp.float32)
        pad_m_exp = pad_m[:, :, None]
        h_seq = np.sum(h_final * pad_m_exp, axis=1) / (np.sum(pad_m_exp, axis=1) + 1e-10)

        # Router probabilities
        logits = np.dot(h_seq, rp['W_r']) + rp['b_r']
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (B, N)

        # Top-k selection with STE
        probs_np = onp.array(_get_value(probs))
        B_cur = probs_np.shape[0]
        mask_np = onp.zeros_like(probs_np)
        for b in range(B_cur):
            top_idx = onp.argsort(probs_np[b])[-k:]
            mask_np[b, top_idx] = 1.0

        selected_probs = probs * mask_np
        renorm = np.sum(selected_probs, axis=-1, keepdims=True) + 1e-10
        weights = selected_probs / renorm

        # Average weights across batch
        avg_w = np.mean(weights, axis=0)

        # Compose delta
        merged = {}
        for key in delta_list[0]:
            weighted_sum = sum(avg_w[i] * delta_list[i][key] for i in range(N))
            merged[key] = weighted_sum

        # Forward with composed params
        composed = {}
        for kk in base_params:
            if kk == '_config':
                composed[kk] = base_params[kk]
            elif kk in merged:
                composed[kk] = base_params[kk] + merged[kk]
            else:
                composed[kk] = base_params[kk]

        logits_out = forward(composed, inp, pad_id)

        # NTP loss
        B_s, T_s, V = logits_out.shape
        max_l = np.max(logits_out, axis=-1, keepdims=True)
        shifted = logits_out - max_l
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        targets_oh = onp.zeros((B_s, T_s, V), dtype=onp.float32)
        for b in range(B_s):
            for t in range(T_s):
                targets_oh[b, t, tgt[b, t]] = 1.0
        token_losses = -np.sum(log_probs * targets_oh, axis=-1)
        ntp_loss = np.sum(token_losses * mask) / (np.sum(mask) + 1e-10)

        # Load-balancing loss
        B_T = B_cur
        f = onp.sum(mask_np, axis=0) / B_T  # (N,) fraction routed
        P = np.mean(probs, axis=0)  # (N,) mean prob
        lb_loss = N * np.sum(f * P)

        return ntp_loss + alpha_balance * lb_loss

    grad_fn = grad(loss_fn)

    # Adam state
    m_state = [onp.zeros_like(router_params[k]) for k in router_keys]
    v_state = [onp.zeros_like(router_params[k]) for k in router_keys]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    step = 0
    n = len(mixed_data)
    rng = onp.random.RandomState(42)
    final_loss = float('inf')

    for epoch in range(epochs):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [mixed_data[j] for j in batch_idx]
            inp, tgt, mask = _prepare_batch(batch, pad_id)
            if onp.sum(mask) == 0:
                continue
            router_vals = [router_params[kk] for kk in router_keys]
            loss_val = float(loss_fn(router_vals, inp, tgt, mask))
            grads = grad_fn(router_vals, inp, tgt, mask)
            grad_norm = onp.sqrt(sum(float(onp.sum(g**2)) for g in grads))
            if grad_norm > clip_grad:
                sc = clip_grad / grad_norm
                grads = [g * sc for g in grads]
            step += 1
            for k_idx, key in enumerate(router_keys):
                g = onp.array(grads[k_idx])
                m_state[k_idx] = beta1 * m_state[k_idx] + (1 - beta1) * g
                v_state[k_idx] = beta2 * v_state[k_idx] + (1 - beta2) * g**2
                m_hat = m_state[k_idx] / (1 - beta1**step)
                v_hat = v_state[k_idx] / (1 - beta2**step)
                router_params[key] = router_params[key] - lr * m_hat / (onp.sqrt(v_hat) + eps_adam)
            epoch_loss += loss_val
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"      router epoch {epoch:3d}: loss={avg_loss:.4f}")

    return router_params, final_loss


# ===========================================================================
# Routing accuracy measurement
# ===========================================================================

def measure_routing_accuracy(base_params, router_params, domain_eval, tok, k=1):
    """Measure how often the router selects the correct domain expert."""
    domain_names = list(domain_eval.keys())
    N = len(domain_names)
    correct = 0
    total = 0

    for di, name in enumerate(domain_names):
        for seq in domain_eval[name]:
            inp_raw = onp.array(seq[:-1])[None, :]  # (1, T-1)
            if inp_raw.shape[1] == 0:
                continue
            _, h = forward_with_hidden(base_params, inp_raw, tok.pad_id)
            h_np = onp.array(h)  # (1, T, d)
            pad_m = (inp_raw != tok.pad_id).astype(onp.float32)[:, :, None]
            h_seq = onp.sum(h_np * pad_m, axis=1) / (onp.sum(pad_m, axis=1) + 1e-10)
            # h_seq: (1, d)

            logits = onp.dot(h_seq, onp.array(router_params['W_r'])) + onp.array(router_params['b_r'])
            probs = onp.exp(logits - onp.max(logits))
            probs = probs / (onp.sum(probs) + 1e-10)

            top_idx = onp.argsort(probs[0])[-k:]
            if di in top_idx:
                correct += 1
            total += 1

    return correct / max(total, 1)


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(seed=42, d=64, r=4, L=2, H=2, n_data=150, n_eval=50,
                   base_epochs=20, lora_epochs=20, router_epochs=15,
                   verbose=True):
    """Run one seed of the MoTE-SOLE experiment."""

    print(f"\n{'='*70}")
    print(f"  MoTE-SOLE Architecture (seed={seed}, d={d}, r={r})")
    print(f"{'='*70}")

    tok = CharTokenizer()
    V = tok.vocab_size
    rng = onp.random.RandomState(seed)
    domain_names = list(DOMAIN_GENERATORS.keys())
    N = len(domain_names)

    # -------------------------------------------------------------------
    # Step 1: Generate domain data
    # -------------------------------------------------------------------
    print("\n[1/8] Generating domain data...")
    domain_data = {}
    domain_eval = {}
    mixed_train_encoded = []
    mixed_labels = []
    for di, (name, gen_fn) in enumerate(DOMAIN_GENERATORS.items()):
        train = gen_fn(n_data, rng)
        test = gen_fn(n_eval, rng)
        domain_data[name] = [tok.encode(s) for s in train]
        domain_eval[name] = [tok.encode(s) for s in test]
        for s in train[:n_data // N]:
            mixed_train_encoded.append(tok.encode(s))
            mixed_labels.append(di)

    # -------------------------------------------------------------------
    # Step 2: Train FP16 base model
    # -------------------------------------------------------------------
    print("\n[2/8] Training FP16 base model on mixed data...")
    fp16_base = init_model(V, d=d, H=H, L=L, seed=seed)
    fp16_base = train_base(fp16_base, mixed_train_encoded, tok.pad_id,
                           epochs=base_epochs, verbose=verbose)

    # -------------------------------------------------------------------
    # Step 3: Create ternary base
    # -------------------------------------------------------------------
    print("\n[3/8] Quantizing base to ternary (BitNet absmean)...")
    ternary_base, scales = quantize_model_to_ternary(fp16_base)

    # Evaluate base models
    base_ppl = {}
    for name in domain_names:
        loss = eval_loss(ternary_base, domain_eval[name], tok.pad_id)
        base_ppl[name] = float(onp.exp(loss))
    print(f"    Ternary base mean PPL: {onp.mean(list(base_ppl.values())):.2f}")

    fp16_base_ppl = {}
    for name in domain_names:
        loss = eval_loss(fp16_base, domain_eval[name], tok.pad_id)
        fp16_base_ppl[name] = float(onp.exp(loss))
    print(f"    FP16 base mean PPL: {onp.mean(list(fp16_base_ppl.values())):.2f}")

    # -------------------------------------------------------------------
    # Step 4: Train domain experts (FP16 and ternary)
    # -------------------------------------------------------------------
    fp16_loras = {}
    ternary_loras = {}
    fp16_deltas = {}
    ternary_deltas = {}
    fp16_single_ppl = {}
    ternary_single_ppl = {}

    for di, name in enumerate(domain_names):
        lora_seed = seed * 100 + di

        # FP16 LoRA on ternary base
        print(f"\n[4/8] Training FP16 LoRA on ternary base: {name}...")
        lora_fp16 = init_lora(ternary_base, rank=r, seed=lora_seed)
        lora_fp16, _ = train_lora(ternary_base, lora_fp16, domain_data[name],
                                  tok.pad_id, epochs=lora_epochs, lr=0.003,
                                  quant_mode='fp16', verbose=verbose)
        fp16_loras[name] = lora_fp16
        fp16_deltas[name] = lora_to_delta(lora_fp16, ternary_base, quant_mode='fp16')

        effective = apply_delta(ternary_base, fp16_deltas[name])
        loss = eval_loss(effective, domain_eval[name], tok.pad_id)
        fp16_single_ppl[name] = float(onp.exp(loss))

        # Ternary LoRA on ternary base (QAT)
        print(f"    Training ternary LoRA on ternary base: {name}...")
        lora_tern = init_lora(ternary_base, rank=r, seed=lora_seed)
        lora_tern, _ = train_lora(ternary_base, lora_tern, domain_data[name],
                                  tok.pad_id, epochs=lora_epochs, lr=0.003,
                                  quant_mode='ternary', verbose=verbose)
        ternary_loras[name] = lora_tern
        ternary_deltas[name] = lora_to_delta(lora_tern, ternary_base, quant_mode='ternary')

        effective = apply_delta(ternary_base, ternary_deltas[name])
        loss = eval_loss(effective, domain_eval[name], tok.pad_id)
        ternary_single_ppl[name] = float(onp.exp(loss))

    # -------------------------------------------------------------------
    # Step 5: Equal-weight composition (baseline)
    # -------------------------------------------------------------------
    print(f"\n[5/8] Equal-weight composition (baseline)...")

    # FP16 equal-weight
    fp16_eq_delta = compose_deltas_equal(list(fp16_deltas.values()))
    fp16_eq_params = apply_delta(ternary_base, fp16_eq_delta)
    fp16_eq_ppl = {}
    for name in domain_names:
        loss = eval_loss(fp16_eq_params, domain_eval[name], tok.pad_id)
        fp16_eq_ppl[name] = float(onp.exp(loss))

    # Ternary equal-weight
    ternary_eq_delta = compose_deltas_equal(list(ternary_deltas.values()))
    ternary_eq_params = apply_delta(ternary_base, ternary_eq_delta)
    ternary_eq_ppl = {}
    for name in domain_names:
        loss = eval_loss(ternary_eq_params, domain_eval[name], tok.pad_id)
        ternary_eq_ppl[name] = float(onp.exp(loss))

    print(f"    FP16 equal-weight mean PPL: {onp.mean(list(fp16_eq_ppl.values())):.2f}")
    print(f"    Ternary equal-weight mean PPL: {onp.mean(list(ternary_eq_ppl.values())):.2f}")

    # -------------------------------------------------------------------
    # Step 6: Train router (MoTE-style)
    # -------------------------------------------------------------------
    print(f"\n[6/8] Training MoTE router...")
    t_router_start = time.time()

    # Train on mixed data with ternary deltas
    router = init_router(d, N, seed=seed)

    # Prepare mixed data with domain labels for routing
    router_mixed = []
    router_labels = []
    for di, name in enumerate(domain_names):
        for seq in domain_data[name][:n_data // N]:
            router_mixed.append(seq)
            router_labels.append(di)

    router, router_loss = train_router(
        ternary_base, ternary_deltas, router,
        router_mixed, router_labels, tok.pad_id,
        epochs=router_epochs, lr=0.01, k=2,
        alpha_balance=0.01, verbose=verbose
    )

    t_router_end = time.time()
    router_time = t_router_end - t_router_start
    print(f"    Router training time: {router_time:.1f}s")

    # -------------------------------------------------------------------
    # Step 7: Evaluate routed composition at k=1, k=2, k=3
    # -------------------------------------------------------------------
    print(f"\n[7/8] Evaluating routed composition...")

    routed_ppl = {}
    routing_accuracy = {}

    for k_val in [1, 2, 3]:
        routed_ppl[f'k{k_val}'] = {}

        # Measure routing accuracy
        acc = measure_routing_accuracy(ternary_base, router, domain_eval, tok, k=k_val)
        routing_accuracy[f'k{k_val}'] = acc
        print(f"    k={k_val} routing accuracy: {acc:.3f}")

        for name in domain_names:
            # For each domain eval set, compose using router
            eval_seqs = domain_eval[name]
            total_loss = 0.0
            total_tokens = 0
            for bi in range(0, len(eval_seqs), 32):
                batch = eval_seqs[bi:bi+32]
                inp, tgt, mask = _prepare_batch(batch, tok.pad_id)
                if onp.sum(mask) == 0:
                    continue

                # Use router to compose
                # Get hidden states
                _, h = forward_with_hidden(ternary_base, inp, tok.pad_id)
                h_np = onp.array(h)
                pad_m = (inp != tok.pad_id).astype(onp.float32)[:, :, None]
                h_seq = onp.sum(h_np * pad_m, axis=1) / (onp.sum(pad_m, axis=1) + 1e-10)

                # Router
                logits_r = onp.dot(h_seq, onp.array(router['W_r'])) + onp.array(router['b_r'])
                probs_r = onp.exp(logits_r - onp.max(logits_r, axis=-1, keepdims=True))
                probs_r = probs_r / (onp.sum(probs_r, axis=-1, keepdims=True) + 1e-10)

                # Top-k selection and renormalization
                B_cur = probs_r.shape[0]
                weights_batch = onp.zeros_like(probs_r)
                for b in range(B_cur):
                    top_idx = onp.argsort(probs_r[b])[-k_val:]
                    selected = probs_r[b, top_idx]
                    selected = selected / (onp.sum(selected) + 1e-10)
                    for si, idx in enumerate(top_idx):
                        weights_batch[b, idx] = selected[si]

                # Average weights across batch (for pre-merge)
                avg_w = onp.mean(weights_batch, axis=0)

                # Compose delta
                delta_list = [ternary_deltas[n] for n in domain_names]
                merged = {}
                for key in delta_list[0]:
                    merged[key] = sum(avg_w[i] * delta_list[i][key] for i in range(N))

                composed = apply_delta(ternary_base, merged)

                # Evaluate
                logits_out = onp.array(forward(composed, inp, tok.pad_id))
                B_s, T_s, V = logits_out.shape
                max_l = onp.max(logits_out, axis=-1, keepdims=True)
                shifted = logits_out - max_l
                log_probs = shifted - onp.log(onp.sum(onp.exp(shifted), axis=-1, keepdims=True))
                for b in range(B_s):
                    for t in range(T_s):
                        if mask[b, t] > 0:
                            total_loss += -log_probs[b, t, tgt[b, t]]
                            total_tokens += 1

            if total_tokens > 0:
                routed_ppl[f'k{k_val}'][name] = float(onp.exp(total_loss / total_tokens))
            else:
                routed_ppl[f'k{k_val}'][name] = float('inf')

    # -------------------------------------------------------------------
    # Step 8: Oracle routing (upper bound)
    # -------------------------------------------------------------------
    print(f"\n[8/8] Oracle routing (upper bound)...")
    oracle_ppl = {}
    for name in domain_names:
        # Oracle: use the single best expert for each domain
        oracle_ppl[name] = ternary_single_ppl[name]

    # -------------------------------------------------------------------
    # Results compilation
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS (seed={seed})")
    print(f"{'='*70}")

    # Per-domain table
    print(f"\n  {'Domain':12s} {'Base':>8s} {'Single':>8s} {'EqWt':>8s} "
          f"{'k=1':>8s} {'k=2':>8s} {'k=3':>8s} {'Oracle':>8s}")
    print(f"  {'-'*68}")
    for name in domain_names:
        print(f"  {name:12s} "
              f"{base_ppl[name]:8.2f} "
              f"{ternary_single_ppl[name]:8.2f} "
              f"{ternary_eq_ppl[name]:8.2f} "
              f"{routed_ppl['k1'][name]:8.2f} "
              f"{routed_ppl['k2'][name]:8.2f} "
              f"{routed_ppl['k3'][name]:8.2f} "
              f"{oracle_ppl[name]:8.2f}")

    # Kill criteria
    # K1: MoTE quality < equal-weight on >50% of domains
    # Compare best routed (k=2, the training k) vs equal-weight
    k1_domains_better = 0
    for name in domain_names:
        if routed_ppl['k2'][name] < ternary_eq_ppl[name]:
            k1_domains_better += 1
    k1_pass = k1_domains_better > N / 2  # routed beats equal-weight on >50%

    # K2: Router training < 1hr
    k2_pass = router_time < 3600

    # K3: Ternary experts < 10% worse than FP16 individually
    ternary_fp16_ratios = []
    for name in domain_names:
        ratio = ternary_single_ppl[name] / fp16_single_ppl[name]
        ternary_fp16_ratios.append(ratio)
    k3_ratio = onp.mean(ternary_fp16_ratios)
    k3_pass = k3_ratio <= 1.10

    print(f"\n  Kill Criteria:")
    print(f"    K1: Routed (k=2) beats equal-weight on {k1_domains_better}/{N} domains "
          f"(threshold >50%): {'PASS' if k1_pass else 'FAIL'}")
    print(f"    K2: Router training time = {router_time:.1f}s "
          f"(threshold <3600s): {'PASS' if k2_pass else 'FAIL'}")
    print(f"    K3: Ternary/FP16 individual ratio = {k3_ratio:.4f} "
          f"(threshold <1.10): {'PASS' if k3_pass else 'FAIL'}")

    # Routing accuracy
    print(f"\n  Routing Accuracy:")
    for k_val in [1, 2, 3]:
        print(f"    k={k_val}: {routing_accuracy[f'k{k_val}']:.3f}")

    results = {
        'seed': seed, 'd': d, 'r': r, 'L': L, 'N': N,
        'base_ppl': base_ppl,
        'fp16_base_ppl': fp16_base_ppl,
        'fp16_single_ppl': fp16_single_ppl,
        'ternary_single_ppl': ternary_single_ppl,
        'fp16_eq_ppl': fp16_eq_ppl,
        'ternary_eq_ppl': ternary_eq_ppl,
        'routed_ppl': routed_ppl,
        'oracle_ppl': oracle_ppl,
        'routing_accuracy': routing_accuracy,
        'router_training_time_s': router_time,
        'kill_criteria': {
            'K1_domains_routed_better': k1_domains_better,
            'K1_total_domains': N,
            'K1_pass': bool(k1_pass),
            'K2_router_time_s': router_time,
            'K2_pass': bool(k2_pass),
            'K3_ternary_fp16_ratio': float(k3_ratio),
            'K3_per_domain_ratios': {name: float(r) for name, r in zip(domain_names, ternary_fp16_ratios)},
            'K3_pass': bool(k3_pass),
        },
        # Summary metrics
        'summary': {
            'ternary_eq_mean_ppl': float(onp.mean(list(ternary_eq_ppl.values()))),
            'routed_k1_mean_ppl': float(onp.mean(list(routed_ppl['k1'].values()))),
            'routed_k2_mean_ppl': float(onp.mean(list(routed_ppl['k2'].values()))),
            'routed_k3_mean_ppl': float(onp.mean(list(routed_ppl['k3'].values()))),
            'oracle_mean_ppl': float(onp.mean(list(oracle_ppl.values()))),
            'base_mean_ppl': float(onp.mean(list(base_ppl.values()))),
            'fp16_eq_mean_ppl': float(onp.mean(list(fp16_eq_ppl.values()))),
        },
        # Cosine similarities between ternary deltas
        'cosine_sims': {},
    }

    # Compute pairwise cosines for ternary deltas
    for i in range(N):
        for j in range(i+1, N):
            n1, n2 = domain_names[i], domain_names[j]
            cos = abs(cosine_sim(
                flatten_delta(ternary_deltas[n1]),
                flatten_delta(ternary_deltas[n2])
            ))
            results['cosine_sims'][f'{n1}-{n2}'] = float(cos)

    return results


def run_all(seeds=(42, 123, 314), d=64, r=4):
    """Run experiment across multiple seeds and aggregate."""
    t0 = time.time()
    all_results = []

    for seed in seeds:
        result = run_experiment(seed=seed, d=d, r=r, verbose=True)
        all_results.append(result)

    domain_names = list(DOMAIN_GENERATORS.keys())
    N = len(domain_names)

    # Aggregate
    agg = {
        'config': {'d': d, 'r': r, 'seeds': list(seeds), 'n_domains': N},
        'per_seed': all_results,
    }

    # Aggregate means across seeds
    def mean_across_seeds(key_path):
        vals = []
        for r in all_results:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(obj)
        return float(onp.mean(vals)), float(onp.std(vals))

    agg['aggregate'] = {}

    for method in ['ternary_eq_mean_ppl', 'routed_k1_mean_ppl', 'routed_k2_mean_ppl',
                    'routed_k3_mean_ppl', 'oracle_mean_ppl', 'base_mean_ppl',
                    'fp16_eq_mean_ppl']:
        mu, sigma = mean_across_seeds(['summary', method])
        agg['aggregate'][method] = {'mean': mu, 'std': sigma}

    # Routing accuracy
    for k_val in [1, 2, 3]:
        mu, sigma = mean_across_seeds(['routing_accuracy', f'k{k_val}'])
        agg['aggregate'][f'routing_accuracy_k{k_val}'] = {'mean': mu, 'std': sigma}

    # Router training time
    times = [r['router_training_time_s'] for r in all_results]
    agg['aggregate']['router_training_time_s'] = {
        'mean': float(onp.mean(times)),
        'std': float(onp.std(times)),
    }

    # Kill criteria
    k1_passes = sum(1 for r in all_results if r['kill_criteria']['K1_pass'])
    k2_passes = sum(1 for r in all_results if r['kill_criteria']['K2_pass'])
    k3_passes = sum(1 for r in all_results if r['kill_criteria']['K3_pass'])

    k1_domains = [r['kill_criteria']['K1_domains_routed_better'] for r in all_results]
    k3_ratios = [r['kill_criteria']['K3_ternary_fp16_ratio'] for r in all_results]

    agg['kill_criteria'] = {
        'K1_domains_routed_better': {
            'mean': float(onp.mean(k1_domains)),
            'std': float(onp.std(k1_domains)),
        },
        'K1_pass_rate': f"{k1_passes}/{len(seeds)}",
        'K2_pass_rate': f"{k2_passes}/{len(seeds)}",
        'K2_max_time_s': float(max(times)),
        'K3_ternary_fp16_ratio': {
            'mean': float(onp.mean(k3_ratios)),
            'std': float(onp.std(k3_ratios)),
        },
        'K3_pass_rate': f"{k3_passes}/{len(seeds)}",
    }

    # Per-domain breakdown (mean across seeds)
    agg['per_domain'] = {}
    for name in domain_names:
        domain_row = {}
        for method_key, result_key in [
            ('base', 'base_ppl'),
            ('ternary_single', 'ternary_single_ppl'),
            ('fp16_single', 'fp16_single_ppl'),
            ('ternary_eq', 'ternary_eq_ppl'),
            ('fp16_eq', 'fp16_eq_ppl'),
        ]:
            vals = [r[result_key][name] for r in all_results]
            domain_row[method_key] = {'mean': float(onp.mean(vals)), 'std': float(onp.std(vals))}

        for k_val in [1, 2, 3]:
            vals = [r['routed_ppl'][f'k{k_val}'][name] for r in all_results]
            domain_row[f'routed_k{k_val}'] = {'mean': float(onp.mean(vals)), 'std': float(onp.std(vals))}

        vals = [r['oracle_ppl'][name] for r in all_results]
        domain_row['oracle'] = {'mean': float(onp.mean(vals)), 'std': float(onp.std(vals))}

        agg['per_domain'][name] = domain_row

    # Cosine similarities
    cos_keys = list(all_results[0]['cosine_sims'].keys())
    agg['cosine_sims'] = {}
    for ck in cos_keys:
        vals = [r['cosine_sims'][ck] for r in all_results]
        agg['cosine_sims'][ck] = {'mean': float(onp.mean(vals)), 'std': float(onp.std(vals))}

    elapsed = time.time() - t0
    agg['runtime_seconds'] = float(elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)
    print(f"\n  Seeds: {seeds}")
    print(f"  Config: d={d}, r={r}, N={N} domains, {len(seeds)} seeds")
    print(f"  Runtime: {elapsed:.1f}s")

    print(f"\n  Method Comparison (mean PPL across domains):")
    print(f"  {'Method':25s} {'Mean PPL':>10s} {'Std':>8s}")
    print(f"  {'-'*45}")
    for method in ['base_mean_ppl', 'fp16_eq_mean_ppl', 'ternary_eq_mean_ppl',
                    'routed_k1_mean_ppl', 'routed_k2_mean_ppl', 'routed_k3_mean_ppl',
                    'oracle_mean_ppl']:
        m = agg['aggregate'][method]
        label = method.replace('_mean_ppl', '').replace('_', ' ')
        print(f"  {label:25s} {m['mean']:10.2f} {m['std']:8.3f}")

    print(f"\n  Routing Accuracy:")
    for k_val in [1, 2, 3]:
        m = agg['aggregate'][f'routing_accuracy_k{k_val}']
        print(f"    k={k_val}: {m['mean']:.3f} +/- {m['std']:.3f}")

    print(f"\n  Per-Domain Comparison (mean PPL):")
    print(f"  {'Domain':12s} {'Base':>8s} {'EqWt':>8s} {'k=1':>8s} {'k=2':>8s} {'Oracle':>8s}")
    print(f"  {'-'*52}")
    for name in domain_names:
        d_row = agg['per_domain'][name]
        print(f"  {name:12s} "
              f"{d_row['base']['mean']:8.2f} "
              f"{d_row['ternary_eq']['mean']:8.2f} "
              f"{d_row['routed_k1']['mean']:8.2f} "
              f"{d_row['routed_k2']['mean']:8.2f} "
              f"{d_row['oracle']['mean']:8.2f}")

    print(f"\n  Kill Criteria (across {len(seeds)} seeds):")
    print(f"    K1: Routed beats equal-weight on {agg['kill_criteria']['K1_domains_routed_better']['mean']:.1f}/{N} domains "
          f"(pass rate: {agg['kill_criteria']['K1_pass_rate']})")
    print(f"    K2: Max router time = {agg['kill_criteria']['K2_max_time_s']:.1f}s "
          f"(pass rate: {agg['kill_criteria']['K2_pass_rate']})")
    print(f"    K3: Ternary/FP16 ratio = {agg['kill_criteria']['K3_ternary_fp16_ratio']['mean']:.4f} "
          f"(pass rate: {agg['kill_criteria']['K3_pass_rate']})")

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
