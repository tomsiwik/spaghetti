#!/usr/bin/env python3
"""
Cross-Domain Composition: Do merged experts handle queries spanning 2+ domains?

REVISION 2 (2026-03-14): Addresses 5 fixes from adversarial review:
  Fix 1: Report per-type gaps with stddev across seeds
  Fix 2: Add single-expert K1 metric (hash-ring scenario)
  Fix 3: Reframe claims -- oracle routing, not hash-ring
  Fix 4: Acknowledge cancellation artifact in aggregate K1
  Fix 5: All 10 domain pairs, 5 seeds for K2 statistical power

Experimental design:
  1. Train 5 domain experts on synthetic structured tasks
  2. Create CROSS-DOMAIN test queries for ALL 10 possible 2-domain pairs
  3. Evaluate: base, single expert (best), multi-expert (2 relevant, oracle)
  4. Report per-type gaps with stddev, max gap, 75th percentile gap
  5. Separate K1 metrics for single-expert and multi-expert

Kill criteria:
  K1_multi: Multi-expert (oracle 2) scores >20% worse than base (aggregate)
  K1_single: Best single expert scores >20% worse than base (aggregate)
  K2: Cross-domain queries route to wrong expert >20% (tightened from 50%)
"""

import json
import math
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp


# ===========================================================================
# Synthetic Data: Pure-domain + Cross-domain generators
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

# Cross-domain query generators for ALL 10 domain pairs
def _make_arith_reverse_data(n, rng):
    """Cross: arithmetic + reverse. '12+34=46>64'"""
    data = []
    for _ in range(n):
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        s = str(a + b)
        data.append(f"{a}+{b}={s}>{s[::-1]}")
    return data

def _make_arith_sort_data(n, rng):
    """Cross: arithmetic + sort. '31+42=73>37'"""
    data = []
    for _ in range(n):
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        s = str(a + b)
        data.append(f"{a}+{b}={s}>{''.join(sorted(s))}")
    return data

def _make_arith_repeat_data(n, rng):
    """Cross: arithmetic + repeat. '12+34=46*2=4646'"""
    data = []
    for _ in range(n):
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        s = str(a + b)
        rep = rng.randint(2, 3)
        data.append(f"{a}+{b}={s}*{rep}={s * rep}")
    return data

def _make_arith_parity_data(n, rng):
    """Cross: arithmetic + parity. '11+22=33>even' (parity of digit sum)"""
    data = []
    for _ in range(n):
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        s = str(a + b)
        digit_sum = sum(int(c) for c in s)
        parity = "even" if digit_sum % 2 == 0 else "odd"
        data.append(f"{a}+{b}={s}>{parity}")
    return data

def _make_reverse_repeat_data(n, rng):
    """Cross: reverse + repeat. 'abc>cba*2=cbacba'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 3)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        rev = s[::-1]
        rep = rng.randint(2, 3)
        data.append(f"{s}>{rev}*{rep}={rev * rep}")
    return data

def _make_reverse_sort_data(n, rng):
    """Cross: reverse + sort. 'dcba>abcd>abcd' (reverse then sort the reversed)"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        rev = s[::-1]
        data.append(f"{s}>{rev}>{''.join(sorted(rev))}")
    return data

def _make_reverse_parity_data(n, rng):
    """Cross: reverse + parity. 'abc>cba>odd' (parity of count of 'a')"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        rev = s[::-1]
        count_a = rev.count("a")
        parity = "even" if count_a % 2 == 0 else "odd"
        data.append(f"{s}>{rev}>{parity}")
    return data

def _make_repeat_sort_data(n, rng):
    """Cross: repeat + sort. 'ab*2=abab>aabb'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(list(chars)) for _ in range(plen))
        rep = rng.randint(2, 3)
        repeated = pat * rep
        data.append(f"{pat}*{rep}={repeated}>{''.join(sorted(repeated))}")
    return data

def _make_repeat_parity_data(n, rng):
    """Cross: repeat + parity. 'ab*2=abab>even' (parity of length)"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(list(chars)) for _ in range(plen))
        rep = rng.randint(2, 4)
        repeated = pat * rep
        parity = "even" if len(repeated) % 2 == 0 else "odd"
        data.append(f"{pat}*{rep}={repeated}>{parity}")
    return data

def _make_sort_parity_data(n, rng):
    """Cross: sort + parity. 'dcba>abcd>even' (parity of count of 'a')"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        sorted_s = "".join(sorted(s))
        count_a = sorted_s.count("a")
        parity = "even" if count_a % 2 == 0 else "odd"
        data.append(f"{s}>{''.join(sorted(s))}>{parity}")
    return data

CROSS_DOMAIN_GENERATORS = {
    "arith_reverse": (_make_arith_reverse_data, ["arithmetic", "reverse"]),
    "arith_sort": (_make_arith_sort_data, ["arithmetic", "sort"]),
    "arith_repeat": (_make_arith_repeat_data, ["arithmetic", "repeat"]),
    "arith_parity": (_make_arith_parity_data, ["arithmetic", "parity"]),
    "reverse_repeat": (_make_reverse_repeat_data, ["reverse", "repeat"]),
    "reverse_sort": (_make_reverse_sort_data, ["reverse", "sort"]),
    "reverse_parity": (_make_reverse_parity_data, ["reverse", "parity"]),
    "repeat_sort": (_make_repeat_sort_data, ["repeat", "sort"]),
    "repeat_parity": (_make_repeat_parity_data, ["repeat", "parity"]),
    "sort_parity": (_make_sort_parity_data, ["sort", "parity"]),
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

    def decode(self, ids):
        out = []
        for i in ids:
            c = self.idx2char.get(i, "")
            if c == self.eos_token: break
            if c == self.pad_token: continue
            out.append(c)
        return "".join(out)


# ===========================================================================
# Model
# ===========================================================================

def init_model(V, d=32, H=2, L=2, max_T=32, seed=42):
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
# Training
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


def train_model(params, data_encoded, pad_id, epochs=20, lr=0.001,
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
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:3d}: loss={avg_loss:.4f}")
    return params


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

def compute_delta(base_params, trained_params):
    delta = {}
    for k in base_params:
        if k == '_config':
            continue
        delta[k] = trained_params[k] - base_params[k]
    return delta


def svd_truncate_delta(delta, rank):
    truncated = {}
    signal_retained = {}
    for k, d in delta.items():
        if d.ndim == 1:
            truncated[k] = d.copy()
            signal_retained[k] = 1.0
        else:
            U, S, Vt = onp.linalg.svd(d, full_matrices=False)
            r = min(rank, len(S))
            truncated[k] = (U[:, :r] * S[:r]) @ Vt[:r, :]
            total_energy = float(onp.sum(S**2))
            kept_energy = float(onp.sum(S[:r]**2))
            signal_retained[k] = kept_energy / max(total_energy, 1e-10)
    return truncated, signal_retained


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


def merge_deltas(delta_list, mode='avg'):
    merged = {}
    N = len(delta_list)
    for k in delta_list[0]:
        s = sum(d[k] for d in delta_list)
        if mode == 'avg':
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
# Gram-Schmidt orthogonalization
# ===========================================================================

def gram_schmidt_deltas(delta_list):
    N = len(delta_list)
    flat_originals = [flatten_delta(d) for d in delta_list]

    pre_cosines = []
    for i in range(N):
        for j in range(i + 1, N):
            pre_cosines.append(abs(cosine_sim(flat_originals[i], flat_originals[j])))

    flat_ortho = []
    for k in range(N):
        v = flat_originals[k].copy()
        for i in range(len(flat_ortho)):
            e_i = flat_ortho[i]
            dot_ve = onp.dot(v, e_i)
            dot_ee = onp.dot(e_i, e_i)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e_i
        flat_ortho.append(v)

    post_cosines = []
    for i in range(N):
        for j in range(i + 1, N):
            post_cosines.append(abs(cosine_sim(flat_ortho[i], flat_ortho[j])))

    signal_retention = []
    for k in range(N):
        orig_norm = onp.linalg.norm(flat_originals[k])
        ortho_norm = onp.linalg.norm(flat_ortho[k])
        signal_retention.append(ortho_norm / max(orig_norm, 1e-12))

    ortho_deltas = []
    for k in range(N):
        offset = 0
        d = {}
        for key in sorted(delta_list[k].keys()):
            shape = delta_list[k][key].shape
            size = int(onp.prod(shape))
            d[key] = flat_ortho[k][offset:offset + size].reshape(shape)
            offset += size
        ortho_deltas.append(d)

    report = {
        'pre_cosines_mean': float(onp.mean(pre_cosines)) if pre_cosines else 0.0,
        'pre_cosines_max': float(onp.max(pre_cosines)) if pre_cosines else 0.0,
        'post_cosines_mean': float(onp.mean(post_cosines)) if post_cosines else 0.0,
        'post_cosines_max': float(onp.max(post_cosines)) if post_cosines else 0.0,
        'signal_retention': [float(s) for s in signal_retention],
        'signal_retention_min': float(min(signal_retention)),
    }
    return ortho_deltas, report


# ===========================================================================
# Subspace analysis
# ===========================================================================

def subspace_analysis(delta_list, domain_names):
    N = len(delta_list)
    flat_deltas = [flatten_delta(d) for d in delta_list]

    results = {}
    for i in range(N):
        for j in range(i + 1, N):
            cos = cosine_sim(flat_deltas[i], flat_deltas[j])
            proj_i_on_j = onp.dot(flat_deltas[i], flat_deltas[j]) / max(onp.dot(flat_deltas[j], flat_deltas[j]), 1e-12)
            shared_i = proj_i_on_j * flat_deltas[j]
            unique_i = flat_deltas[i] - shared_i
            shared_frac_i = onp.linalg.norm(shared_i) / max(onp.linalg.norm(flat_deltas[i]), 1e-12)

            results[f"{domain_names[i]}_vs_{domain_names[j]}"] = {
                'cosine': float(cos),
                'shared_fraction_i': float(shared_frac_i),
                'shared_norm_i': float(onp.linalg.norm(shared_i)),
                'unique_norm_i': float(onp.linalg.norm(unique_i)),
                'total_norm_i': float(onp.linalg.norm(flat_deltas[i])),
            }

    return results


# ===========================================================================
# Main Experiment
# ===========================================================================

def run_experiment(n_domains=5, rank_per_expert=4, epochs_per_expert=15,
                   d_model=32, n_heads=2, n_layers=2, n_train=200,
                   n_cross_test=50, n_pure_test=50, n_seeds=5,
                   lr=0.001, batch_size=16):
    """Run cross-domain composition experiment (Revision 2).

    Key changes from v1:
      - All 10 cross-domain pairs (was 5)
      - 5 seeds (was 3)
      - Separate K1 for single-expert and multi-expert
      - Per-type gaps with stddev reported
      - K2 threshold tightened to 20% (was 50%)
    """
    domains = list(DOMAIN_GENERATORS.keys())[:n_domains]
    cross_domains = list(CROSS_DOMAIN_GENERATORS.keys())

    results_dir = Path(__file__).parent
    tok = CharTokenizer()

    print("=" * 76)
    print("  CROSS-DOMAIN COMPOSITION EXPERIMENT (REVISION 2)")
    print("=" * 76)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}, V={tok.vocab_size}")
    print(f"  Pure domains: {domains}")
    print(f"  Cross domains: {len(cross_domains)} (all 10 pairs)")
    print(f"  Rank per expert: {rank_per_expert}")
    print(f"  Training: {epochs_per_expert} epochs/expert")
    print(f"  Data: {n_train} train/domain, {n_cross_test} cross-domain test")
    print(f"  Seeds: {n_seeds}")
    print("=" * 76)

    all_seed_results = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        rng = onp.random.RandomState(seed)
        t_seed_start = time.time()

        print(f"\n{'='*76}")
        print(f"  SEED {seed} ({seed_idx+1}/{n_seeds})")
        print(f"{'='*76}")

        # -- Generate data --
        domain_train_enc = {}
        domain_test_enc = {}
        for dom_name in domains:
            gen = DOMAIN_GENERATORS[dom_name]
            train_data = gen(n_train, onp.random.RandomState(seed + hash(dom_name) % 10000))
            test_data = gen(n_pure_test, onp.random.RandomState(seed + hash(dom_name) % 10000 + 1))
            domain_train_enc[dom_name] = [tok.encode(s) for s in train_data]
            domain_test_enc[dom_name] = [tok.encode(s) for s in test_data]

        combined_train = []
        for dom_name in domains:
            combined_train.extend(domain_train_enc[dom_name])
        rng.shuffle(combined_train)

        cross_test_enc = {}
        cross_test_involved = {}
        for cross_name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
            test_data = gen_fn(n_cross_test, onp.random.RandomState(seed + hash(cross_name) % 10000 + 2))
            cross_test_enc[cross_name] = [tok.encode(s) for s in test_data]
            cross_test_involved[cross_name] = involved

        # -- Train base model --
        print(f"\n  --- Training base model (all domains) ---")
        base_params = init_model(tok.vocab_size, d=d_model, H=n_heads, L=n_layers,
                                  max_T=32, seed=seed)
        base_trained = {}
        for k, v in base_params.items():
            if k == '_config':
                base_trained[k] = v
            else:
                base_trained[k] = v.copy()
        base_trained = train_model(base_trained, combined_train, tok.pad_id,
                                    epochs=epochs_per_expert, lr=lr,
                                    batch_size=batch_size, verbose=False)

        # -- Train domain experts --
        print(f"\n  --- Training {n_domains} domain experts ---")
        expert_deltas_trunc = {}

        for dom_name in domains:
            print(f"    Training expert '{dom_name}'...")
            expert_params = {}
            for k, v in base_params.items():
                if k == '_config':
                    expert_params[k] = v
                else:
                    expert_params[k] = v.copy()

            expert_params = train_model(
                expert_params, domain_train_enc[dom_name], tok.pad_id,
                epochs=epochs_per_expert, lr=lr, batch_size=batch_size,
                verbose=False
            )

            delta = compute_delta(base_params, expert_params)
            delta_trunc, _ = svd_truncate_delta(delta, rank_per_expert)
            expert_deltas_trunc[dom_name] = delta_trunc

        # -- Subspace analysis --
        print(f"\n  --- Subspace analysis ---")
        delta_list = [expert_deltas_trunc[d] for d in domains]
        subspace = subspace_analysis(delta_list, domains)

        # -- GS orthogonalization --
        gs_deltas, gs_report = gram_schmidt_deltas(delta_list)
        print(f"    Pre-GS mean|cos|: {gs_report['pre_cosines_mean']:.4f}")
        print(f"    Min retention: {gs_report['signal_retention_min']:.4f}")

        # -- Build composition models --
        naive_merge = merge_deltas(list(expert_deltas_trunc.values()), mode='avg')
        naive_params = apply_delta(base_params, naive_merge)

        gs_merge = merge_deltas(gs_deltas, mode='avg')
        gs_params = apply_delta(base_params, gs_merge)

        # -- Evaluate on pure-domain test sets --
        print(f"\n  --- Pure-domain evaluation ---")
        pure_results = {}
        for dom_name in domains:
            base_loss = eval_loss(base_trained, domain_test_enc[dom_name], tok.pad_id)
            expert_p = apply_delta(base_params, expert_deltas_trunc[dom_name])
            expert_loss = eval_loss(expert_p, domain_test_enc[dom_name], tok.pad_id)
            naive_loss = eval_loss(naive_params, domain_test_enc[dom_name], tok.pad_id)
            gs_loss = eval_loss(gs_params, domain_test_enc[dom_name], tok.pad_id)

            pure_results[dom_name] = {
                'base_trained': float(base_loss),
                'expert': float(expert_loss),
                'naive_merge': float(naive_loss),
                'gs_merge': float(gs_loss),
            }
            print(f"    {dom_name}: base={base_loss:.3f} expert={expert_loss:.3f} "
                  f"naive={naive_loss:.3f} gs={gs_loss:.3f}")

        # -- Evaluate on cross-domain test sets --
        print(f"\n  --- Cross-domain evaluation ---")
        cross_results = {}

        for cross_name in cross_domains:
            involved = cross_test_involved[cross_name]
            test_enc = cross_test_enc[cross_name]

            base_loss = eval_loss(base_trained, test_enc, tok.pad_id)

            # Single expert: both involved domains
            single1_params = apply_delta(base_params, expert_deltas_trunc[involved[0]])
            single1_loss = eval_loss(single1_params, test_enc, tok.pad_id)

            single2_params = apply_delta(base_params, expert_deltas_trunc[involved[1]])
            single2_loss = eval_loss(single2_params, test_enc, tok.pad_id)

            best_single_loss = min(single1_loss, single2_loss)

            # Naive merge
            naive_loss = eval_loss(naive_params, test_enc, tok.pad_id)

            # GS merge
            gs_loss = eval_loss(gs_params, test_enc, tok.pad_id)

            # Multi-expert (oracle 2-expert)
            multi_delta = merge_deltas(
                [expert_deltas_trunc[involved[0]], expert_deltas_trunc[involved[1]]],
                mode='avg'
            )
            multi_params = apply_delta(base_params, multi_delta)
            multi_loss = eval_loss(multi_params, test_enc, tok.pad_id)

            # All single expert losses for routing analysis
            all_single_losses = {}
            for dom in domains:
                single_p = apply_delta(base_params, expert_deltas_trunc[dom])
                all_single_losses[dom] = eval_loss(single_p, test_enc, tok.pad_id)

            best_dom = min(all_single_losses, key=all_single_losses.get)
            routing_correct = best_dom in involved

            cross_results[cross_name] = {
                'involved': involved,
                'base_trained': float(base_loss),
                'single_expert_1': float(single1_loss),
                'single_expert_2': float(single2_loss),
                'best_single_loss': float(best_single_loss),
                'naive_merge_all': float(naive_loss),
                'gs_merge_all': float(gs_loss),
                'multi_expert_2': float(multi_loss),
                'best_routed_domain': best_dom,
                'routing_correct': routing_correct,
                'all_single_losses': {k: float(v) for k, v in all_single_losses.items()},
            }

            print(f"    {cross_name}: base={base_loss:.3f} best_single={best_single_loss:.3f} "
                  f"multi={multi_loss:.3f} route={'OK' if routing_correct else 'WRONG'}")

        # -- Per-seed gap computation --
        k1_multi_gaps = []
        k1_single_gaps = []
        for cross_name, cr in cross_results.items():
            base = cr['base_trained']
            if base > 0:
                multi_gap = (cr['multi_expert_2'] - base) / base * 100
                single_gap = (cr['best_single_loss'] - base) / base * 100
            else:
                multi_gap = single_gap = 0.0
            k1_multi_gaps.append(multi_gap)
            k1_single_gaps.append(single_gap)

        k2_errors = sum(1 for cr in cross_results.values() if not cr['routing_correct'])
        k2_total = len(cross_results)

        seed_result = {
            'seed': seed,
            'pure_results': pure_results,
            'cross_results': cross_results,
            'subspace_analysis': subspace,
            'gs_report': gs_report,
            'k1_multi_gaps': k1_multi_gaps,
            'k1_single_gaps': k1_single_gaps,
            'k2_errors': k2_errors,
            'k2_total': k2_total,
        }
        all_seed_results.append(seed_result)

        elapsed = time.time() - t_seed_start
        print(f"\n  Seed {seed} time: {elapsed:.1f}s")

    # ==================================================================
    # Aggregate across seeds -- with per-type statistics (Fix 1, Fix 4)
    # ==================================================================
    print(f"\n{'='*76}")
    print("  AGGREGATE RESULTS (REVISION 2)")
    print(f"{'='*76}")

    # Per-type gap statistics across seeds
    print(f"\n  Per-type multi-expert gaps (% vs base, across {n_seeds} seeds):")
    print(f"  {'Cross-Type':<18s} | {'Mean':>7s} | {'Std':>7s} | {'Min':>7s} | {'Max':>7s} | {'P75':>7s}")
    print(f"  {'-'*65}")

    per_type_multi_stats = {}
    per_type_single_stats = {}

    for ci, cross_name in enumerate(cross_domains):
        # Multi-expert gaps across seeds for this type
        type_multi_gaps = [s['k1_multi_gaps'][ci] for s in all_seed_results]
        type_single_gaps = [s['k1_single_gaps'][ci] for s in all_seed_results]

        multi_mean = float(onp.mean(type_multi_gaps))
        multi_std = float(onp.std(type_multi_gaps))
        multi_min = float(onp.min(type_multi_gaps))
        multi_max = float(onp.max(type_multi_gaps))
        multi_p75 = float(onp.percentile(type_multi_gaps, 75))

        single_mean = float(onp.mean(type_single_gaps))
        single_std = float(onp.std(type_single_gaps))
        single_min = float(onp.min(type_single_gaps))
        single_max = float(onp.max(type_single_gaps))
        single_p75 = float(onp.percentile(type_single_gaps, 75))

        per_type_multi_stats[cross_name] = {
            'mean': multi_mean, 'std': multi_std,
            'min': multi_min, 'max': multi_max, 'p75': multi_p75,
            'per_seed': type_multi_gaps,
        }
        per_type_single_stats[cross_name] = {
            'mean': single_mean, 'std': single_std,
            'min': single_min, 'max': single_max, 'p75': single_p75,
            'per_seed': type_single_gaps,
        }

        print(f"  {cross_name:<18s} | {multi_mean:+7.1f} | {multi_std:7.1f} | "
              f"{multi_min:+7.1f} | {multi_max:+7.1f} | {multi_p75:+7.1f}")

    print(f"\n  Per-type single-expert gaps (% vs base, across {n_seeds} seeds):")
    print(f"  {'Cross-Type':<18s} | {'Mean':>7s} | {'Std':>7s} | {'Min':>7s} | {'Max':>7s} | {'P75':>7s}")
    print(f"  {'-'*65}")
    for cross_name in cross_domains:
        s = per_type_single_stats[cross_name]
        print(f"  {cross_name:<18s} | {s['mean']:+7.1f} | {s['std']:7.1f} | "
              f"{s['min']:+7.1f} | {s['max']:+7.1f} | {s['p75']:+7.1f}")

    # Aggregate statistics
    all_multi_gaps_flat = []
    all_single_gaps_flat = []
    for s in all_seed_results:
        all_multi_gaps_flat.extend(s['k1_multi_gaps'])
        all_single_gaps_flat.extend(s['k1_single_gaps'])

    agg_multi_mean = float(onp.mean(all_multi_gaps_flat))
    agg_multi_std = float(onp.std(all_multi_gaps_flat))
    agg_multi_max = float(onp.max(all_multi_gaps_flat))
    agg_multi_p75 = float(onp.percentile(all_multi_gaps_flat, 75))

    agg_single_mean = float(onp.mean(all_single_gaps_flat))
    agg_single_std = float(onp.std(all_single_gaps_flat))
    agg_single_max = float(onp.max(all_single_gaps_flat))
    agg_single_p75 = float(onp.percentile(all_single_gaps_flat, 75))

    print(f"\n  Aggregate multi-expert (oracle 2):")
    print(f"    Mean gap: {agg_multi_mean:+.2f}% +/- {agg_multi_std:.2f}%")
    print(f"    Max gap:  {agg_multi_max:+.2f}%")
    print(f"    P75 gap:  {agg_multi_p75:+.2f}%")
    n_multi_exceed_20 = sum(1 for g in all_multi_gaps_flat if g > 20.0)
    print(f"    Types exceeding 20%: {n_multi_exceed_20}/{len(all_multi_gaps_flat)}")

    print(f"\n  Aggregate single-expert (hash-ring scenario):")
    print(f"    Mean gap: {agg_single_mean:+.2f}% +/- {agg_single_std:.2f}%")
    print(f"    Max gap:  {agg_single_max:+.2f}%")
    print(f"    P75 gap:  {agg_single_p75:+.2f}%")
    n_single_exceed_20 = sum(1 for g in all_single_gaps_flat if g > 20.0)
    print(f"    Types exceeding 20%: {n_single_exceed_20}/{len(all_single_gaps_flat)}")

    # Cancellation artifact analysis (Fix 4)
    n_positive = sum(1 for g in all_multi_gaps_flat if g > 0)
    n_negative = sum(1 for g in all_multi_gaps_flat if g < 0)
    print(f"\n  Cancellation analysis (Fix 4):")
    print(f"    Positive gaps (multi worse than base): {n_positive}/{len(all_multi_gaps_flat)}")
    print(f"    Negative gaps (multi better than base): {n_negative}/{len(all_multi_gaps_flat)}")
    print(f"    Mean of POSITIVE gaps only: {onp.mean([g for g in all_multi_gaps_flat if g > 0]):+.2f}%")
    if n_negative > 0:
        print(f"    Mean of NEGATIVE gaps only: {onp.mean([g for g in all_multi_gaps_flat if g < 0]):+.2f}%")

    # Routing analysis (K2) with tightened threshold (Fix 5)
    all_k2_errors = sum(s['k2_errors'] for s in all_seed_results)
    all_k2_total = sum(s['k2_total'] for s in all_seed_results)
    k2_error_rate = all_k2_errors / max(all_k2_total, 1)

    print(f"\n  Routing (K2, tightened to 20%):")
    print(f"    Error rate: {k2_error_rate*100:.1f}% ({all_k2_errors}/{all_k2_total})")
    print(f"    N={all_k2_total} trials ({len(cross_domains)} types x {n_seeds} seeds)")

    # ==================================================================
    # Kill Criteria
    # ==================================================================
    print(f"\n{'='*76}")
    print("  KILL CRITERIA (REVISION 2)")
    print(f"{'='*76}")

    # K1_multi: oracle 2-expert aggregate
    k1_multi_kill = agg_multi_mean > 20.0
    print(f"\n  K1_multi: Oracle 2-expert mean gap >20%?")
    print(f"      Mean: {agg_multi_mean:+.2f}% +/- {agg_multi_std:.2f}%")
    print(f"      Max:  {agg_multi_max:+.2f}%")
    print(f"      STATUS: {'KILL' if k1_multi_kill else 'PASS'}")
    if agg_multi_max > 20.0:
        print(f"      WARNING: max gap {agg_multi_max:+.2f}% exceeds 20% on individual type/seed")

    # K1_single: hash-ring scenario (Fix 2)
    k1_single_kill = agg_single_mean > 20.0
    print(f"\n  K1_single: Best single expert mean gap >20%? (hash-ring scenario)")
    print(f"      Mean: {agg_single_mean:+.2f}% +/- {agg_single_std:.2f}%")
    print(f"      Max:  {agg_single_max:+.2f}%")
    print(f"      STATUS: {'KILL' if k1_single_kill else 'PASS'}")

    # K2: routing, tightened threshold
    k2_kill = k2_error_rate > 0.20
    print(f"\n  K2: Routing error rate >20%? (tightened from 50%)")
    print(f"      Error rate: {k2_error_rate*100:.1f}% ({all_k2_errors}/{all_k2_total})")
    print(f"      STATUS: {'KILL' if k2_kill else 'PASS'}")

    overall_kill = k1_multi_kill or k2_kill
    print(f"\n  OVERALL (K1_multi + K2): {'KILL' if overall_kill else 'PASS'}")
    print(f"  K1_single (informational): {'KILL' if k1_single_kill else 'PASS'}")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        'experiment': 'cross_domain_composition_v2',
        'revision': 2,
        'revision_fixes': [
            'Fix 1: per-type gaps with stddev',
            'Fix 2: single-expert K1 (hash-ring scenario)',
            'Fix 3: reframed claims (oracle routing, not hash-ring)',
            'Fix 4: cancellation artifact acknowledged',
            'Fix 5: 10 cross-domain types, 5 seeds, K2 threshold 20%',
        ],
        'config': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'vocab_size': tok.vocab_size,
            'n_domains': n_domains,
            'domains': domains,
            'n_cross_domains': len(cross_domains),
            'cross_domains': cross_domains,
            'rank_per_expert': rank_per_expert,
            'epochs_per_expert': epochs_per_expert,
            'n_train': n_train,
            'n_cross_test': n_cross_test,
            'n_pure_test': n_pure_test,
            'n_seeds': n_seeds,
        },
        'aggregate': {
            'k1_multi': {
                'mean_gap_pct': agg_multi_mean,
                'std_gap_pct': agg_multi_std,
                'max_gap_pct': agg_multi_max,
                'p75_gap_pct': agg_multi_p75,
                'n_exceed_20pct': n_multi_exceed_20,
                'n_total': len(all_multi_gaps_flat),
            },
            'k1_single': {
                'mean_gap_pct': agg_single_mean,
                'std_gap_pct': agg_single_std,
                'max_gap_pct': agg_single_max,
                'p75_gap_pct': agg_single_p75,
                'n_exceed_20pct': n_single_exceed_20,
                'n_total': len(all_single_gaps_flat),
            },
            'k2_error_rate': float(k2_error_rate),
            'k2_errors': all_k2_errors,
            'k2_total': all_k2_total,
            'cancellation': {
                'n_positive': n_positive,
                'n_negative': n_negative,
                'n_total': len(all_multi_gaps_flat),
                'mean_positive_only': float(onp.mean([g for g in all_multi_gaps_flat if g > 0])) if n_positive > 0 else 0.0,
                'mean_negative_only': float(onp.mean([g for g in all_multi_gaps_flat if g < 0])) if n_negative > 0 else 0.0,
            },
        },
        'per_type_multi_stats': per_type_multi_stats,
        'per_type_single_stats': per_type_single_stats,
        'kill_criteria': {
            'k1_multi_mean_pct': agg_multi_mean,
            'k1_multi_threshold': 20.0,
            'k1_multi_kill': bool(k1_multi_kill),
            'k1_single_mean_pct': agg_single_mean,
            'k1_single_threshold': 20.0,
            'k1_single_kill': bool(k1_single_kill),
            'k2_error_rate': float(k2_error_rate),
            'k2_threshold': 0.20,
            'k2_kill': bool(k2_kill),
            'overall_kill': bool(overall_kill),
        },
        'per_seed_results': all_seed_results,
    }

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n  Results saved to {output_file}")

    return output


if __name__ == '__main__':
    import sys
    if '--fast' in sys.argv:
        run_experiment(n_seeds=2, epochs_per_expert=8, n_train=100,
                       n_cross_test=30, n_pure_test=30)
    else:
        run_experiment()
