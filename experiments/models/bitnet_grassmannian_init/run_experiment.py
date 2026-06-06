#!/usr/bin/env python3
"""
Grassmannian Skeleton Init + Ternary QAT: Does AP init survive quantization?

Hypothesis: AP-initialized LoRA experts, after ternary QAT with STE, produce
more orthogonal expert deltas than random-initialized experts with the same QAT.

Two conditions (same training procedure, only initialization differs):
  (a) AP-init: LoRA A matrices from Grassmannian AP skeleton, B=0
  (b) Random-init: LoRA A matrices from Gaussian init, B=0

Both use ternary QAT with STE on a ternary base model.
Architecture: d=64, r=4, L=2, 5 domains, 3 seeds.

Kill criteria:
  K1: AP-init ternary |cos| > 0.7x random-init ternary (no improvement)
  K2: AP-init ternary experts >5% worse individual quality

Reuses model/data/training from bitnet_ternary_adapter_composition.
Reuses AP algorithm from grassmannian_expert_init.

Pure numpy + autograd, CPU only. Runtime target: < 10 minutes.
"""

import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp
from scipy.stats import wilcoxon


# ===========================================================================
# Synthetic Data: 5 domains (from bitnet_ternary_adapter_composition)
# ===========================================================================

def _make_arithmetic_data(n, rng):
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        data.append(f"{a}+{b}={a+b}")
    return data

def _make_reverse_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{s[::-1]}")
    return data

def _make_repeat_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(list(chars)) for _ in range(plen))
        rep = rng.randint(2, 4)
        data.append(f"{pat}*{rep}={pat * rep}")
    return data

def _make_sort_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        length = rng.randint(2, 5)
        s = "".join(rng.choice(list(chars)) for _ in range(length))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data

def _make_parity_data(n, rng):
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
# Model (identical to bitnet_ternary_adapter_composition)
# ===========================================================================

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 2
LORA_RANK = 4
MAX_T = 32
DTYPE = onp.float32


def init_model(V, d=D_MODEL, H=N_HEADS, L=N_LAYERS, max_T=MAX_T, seed=42):
    rng = onp.random.RandomState(seed)
    s = 0.02
    params = {
        'tok_emb': rng.randn(V, d).astype(DTYPE) * s,
        'pos_emb': rng.randn(max_T, d).astype(DTYPE) * s,
    }
    for li in range(L):
        params[f'ln1_w_{li}'] = onp.ones(d, dtype=DTYPE)
        params[f'Wqkv_{li}'] = rng.randn(d, 3 * d).astype(DTYPE) * s
        params[f'Wo_{li}'] = rng.randn(d, d).astype(DTYPE) * s
        params[f'ln2_w_{li}'] = onp.ones(d, dtype=DTYPE)
        params[f'W1_{li}'] = rng.randn(d, 4 * d).astype(DTYPE) * s
        params[f'W2_{li}'] = rng.randn(4 * d, d).astype(DTYPE) * s
    params['ln_f_w'] = onp.ones(d, dtype=DTYPE)
    params['W_head'] = rng.randn(d, V).astype(DTYPE) * s
    params['_config'] = {'V': V, 'd': d, 'H': H, 'L': L, 'max_T': max_T}
    return params


def ternary_quantize_weight(W):
    alpha = onp.mean(onp.abs(W))
    if alpha < 1e-10:
        return onp.zeros_like(W), alpha
    W_scaled = W / alpha
    W_ternary = onp.clip(onp.round(W_scaled), -1, 1).astype(DTYPE)
    return W_ternary, alpha


def quantize_model_to_ternary(params):
    ternary_params = {}
    fp32_keys = {'tok_emb', 'pos_emb', 'ln_f_w', '_config'}
    fp32_keys.update(k for k in params if k.startswith('ln'))
    for k, v in params.items():
        if k in fp32_keys or k == '_config':
            ternary_params[k] = v if k == '_config' else v.copy()
        elif v.ndim >= 2:
            W_t, alpha = ternary_quantize_weight(v)
            ternary_params[k] = W_t * alpha
        else:
            ternary_params[k] = v.copy()
    return ternary_params


def _rms_norm(x, w, eps=1e-5):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def forward(params, idx_2d, pad_id=0):
    cfg = params['_config']
    d, H, L = cfg['d'], cfg['H'], cfg['L']
    hd = d // H
    B, T = idx_2d.shape
    x = params['tok_emb'][idx_2d] + params['pos_emb'][:T]
    mask = onp.triu(onp.ones((T, T)) * (-1e9), k=1).astype(DTYPE)
    for li in range(L):
        h = _rms_norm(x, params[f'ln1_w_{li}'])
        qkv = np.dot(h, params[f'Wqkv_{li}'])
        qkv = np.reshape(qkv, (B, T, 3, H, hd))
        q, k, v = qkv[:,:,0,:,:], qkv[:,:,1,:,:], qkv[:,:,2,:,:]
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
    targets_oh = onp.zeros((B, T, V), dtype=DTYPE)
    for b in range(B):
        for t in range(T):
            targets_oh[b, t, targets_2d[b, t]] = 1.0
    token_losses = -np.sum(log_probs * targets_oh, axis=-1)
    masked_loss = np.sum(token_losses * mask_2d)
    n_tokens = np.sum(mask_2d) + 1e-10
    return masked_loss / n_tokens


# ===========================================================================
# Alternating Projection on Grassmannian (from grassmannian_expert_init)
# ===========================================================================

def welch_bound(N, r, d):
    Nr = N * r
    if Nr <= d:
        return 0.0
    return onp.sqrt(r * (Nr - d) / (d * (Nr - r)))


def random_grassmannian_points(N, r, d, rng):
    frames = onp.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        M = rng.randn(d, r).astype(DTYPE)
        Q, _ = onp.linalg.qr(M)
        frames[i] = Q[:, :r]
    return frames


def frames_to_gram(frames):
    N, d, r = frames.shape
    Nr = N * r
    G = onp.zeros((Nr, Nr), dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            G[i*r:(i+1)*r, j*r:(j+1)*r] = frames[i].T @ frames[j]
    return G


def block_norms(G, N, r):
    norms = onp.zeros((N, N), dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            block = G[i*r:(i+1)*r, j*r:(j+1)*r]
            norms[i, j] = onp.linalg.norm(block, 'fro')
    return norms


def structural_projection(G, N, r, mu_target):
    G_new = G.copy()
    for i in range(N):
        for j in range(N):
            if i == j:
                G_new[i*r:(i+1)*r, j*r:(j+1)*r] = onp.eye(r, dtype=DTYPE)
            else:
                block = G_new[i*r:(i+1)*r, j*r:(j+1)*r]
                norm = onp.linalg.norm(block, 'fro')
                if norm > mu_target:
                    G_new[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)
    return G_new


def spectral_projection(G, N, r, d):
    Nr = N * r
    G = (G + G.T) / 2
    eigvals, eigvecs = onp.linalg.eigh(G)
    idx = onp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals_proj = onp.zeros(Nr, dtype=DTYPE)
    k = min(d, Nr)
    eigvals_proj[:k] = onp.maximum(eigvals[:k], 0)
    current_trace = eigvals_proj.sum()
    if current_trace > 1e-10:
        eigvals_proj *= (N * r) / current_trace
    G_proj = (eigvecs * eigvals_proj[None, :]) @ eigvecs.T
    G_proj = (G_proj + G_proj.T) / 2
    return G_proj


def gram_to_frames(G, N, r, d):
    Nr = N * r
    G = (G + G.T) / 2
    eigvals, eigvecs = onp.linalg.eigh(G)
    idx = onp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(d, Nr)
    sqrt_eig = onp.sqrt(onp.maximum(eigvals[:k], 0)).astype(DTYPE)
    embedding = (eigvecs[:, :k] * sqrt_eig[None, :]).astype(DTYPE)
    frames = onp.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        block = embedding[i*r:(i+1)*r, :]
        if k < d:
            padded = onp.zeros((r, d), dtype=DTYPE)
            padded[:, :k] = block
            block = padded
        else:
            block = block[:, :d]
        Q, _ = onp.linalg.qr(block.T)
        frames[i] = Q[:, :r]
    return frames


def alternating_projection(N, r, d, n_iter=500, mu_factor=1.2, rng=None):
    if rng is None:
        rng = onp.random.RandomState(42)
    wb = welch_bound(N, r, d)
    mu_target = max(mu_factor * wb, 1e-6)
    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)
    for it in range(n_iter):
        G = structural_projection(G, N, r, mu_target)
        G = spectral_projection(G, N, r, d)
    frames = gram_to_frames(G, N, r, d)
    # Compute final coherence
    G_final = frames_to_gram(frames)
    norms = block_norms(G_final, N, r)
    onp.fill_diagonal(norms, 0)
    mask = onp.triu(onp.ones((N, N), dtype=bool), k=1)
    mean_coh = float(norms[mask].mean())
    max_coh = float(norms.max())
    return frames, {'welch_bound': float(wb), 'mean_coherence': mean_coh, 'max_coherence': max_coh}


# ===========================================================================
# STE Quantization (from bitnet_ternary_adapter_composition)
# ===========================================================================

def _get_value(W):
    if hasattr(W, '_value'):
        return onp.array(W._value)
    return onp.array(W)


def ternary_quantize_ste(W):
    W_np = _get_value(W)
    alpha = float(onp.mean(onp.abs(W_np))) + 1e-10
    W_scaled = W_np / alpha
    W_q_np = onp.clip(onp.round(W_scaled), -1, 1).astype(DTYPE) * alpha
    residual = W_q_np - W_np
    return W + residual


def _np_ternary_quantize(W):
    alpha = onp.mean(onp.abs(W)) + 1e-10
    W_scaled = W / alpha
    return onp.clip(onp.round(W_scaled), -1, 1) * alpha


# ===========================================================================
# LoRA Init: AP-based vs Random
# ===========================================================================

def init_lora_ap(base_params, frames, expert_idx, rank=LORA_RANK):
    """Initialize LoRA from AP Grassmannian frame.

    frame: (d, r) orthonormal matrix -- the assigned subspace slot.
    Uses the frame as A for all weight matrices in the model.
    For non-square weight matrices (e.g., d -> 4d), projects frame.
    B matrices start at zero (standard LoRA init).
    """
    frame = frames[expert_idx]  # (d, r)
    lora = {}
    for k, v in base_params.items():
        if k == '_config' or v.ndim < 2:
            continue
        d_in, d_out = v.shape
        if d_in == D_MODEL:
            # A is (d, r) -- use frame directly
            lora[f'{k}_A'] = frame.copy()
        else:
            # A is (d_ff, r) -- project frame into d_ff space via deterministic map
            # Use a random projection seeded by key hash for reproducibility
            proj_rng = onp.random.RandomState(hash(k) % (2**31))
            proj = proj_rng.randn(d_in, D_MODEL).astype(DTYPE)
            a_raw = proj @ frame  # (d_in, r)
            Q, _ = onp.linalg.qr(a_raw)
            lora[f'{k}_A'] = Q[:, :rank].astype(DTYPE)
        lora[f'{k}_B'] = onp.zeros((rank, d_out), dtype=DTYPE)
    return lora


def init_lora_random(base_params, rank=LORA_RANK, seed=42):
    """Standard random LoRA init (Gaussian A, zero B)."""
    rng = onp.random.RandomState(seed)
    lora = {}
    for k, v in base_params.items():
        if k == '_config' or v.ndim < 2:
            continue
        d_in, d_out = v.shape
        lora[f'{k}_A'] = rng.randn(d_in, rank).astype(DTYPE) * 0.01
        lora[f'{k}_B'] = onp.zeros((rank, d_out), dtype=DTYPE)
    return lora


# ===========================================================================
# Training with QAT
# ===========================================================================

def apply_lora_ternary(base_params, lora, alpha_over_r=1.0):
    """Apply LoRA with ternary QAT."""
    result = {}
    for k, v in base_params.items():
        if k == '_config':
            result[k] = v
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            A = ternary_quantize_ste(lora[A_key])
            B = ternary_quantize_ste(lora[B_key])
            delta = np.dot(A, B) * alpha_over_r
            result[k] = v + delta
        else:
            result[k] = v
    return result


def _prepare_batch(batch, pad_id, max_len=MAX_T):
    max_l = min(max(len(s) for s in batch), max_len)
    B = len(batch)
    inp = onp.full((B, max_l), pad_id, dtype=onp.int32)
    tgt = onp.full((B, max_l), pad_id, dtype=onp.int32)
    mask = onp.zeros((B, max_l), dtype=DTYPE)
    for b, seq in enumerate(batch):
        L = min(len(seq), max_l + 1)
        inp[b, :L-1] = seq[:L-1]
        tgt[b, :L-1] = seq[1:L]
        mask[b, :L-1] = 1.0
    return inp, tgt, mask


def train_lora_ternary(base_params, lora, data_encoded, pad_id,
                       epochs=30, lr=0.003, batch_size=32, clip_grad=1.0):
    """Train LoRA params with ternary QAT (STE)."""
    lora_keys = sorted(lora.keys())

    def loss_fn(lora_vals, inp, tgt, msk):
        lo = dict(zip(lora_keys, lora_vals))
        effective = apply_lora_ternary(base_params, lo)
        return compute_loss(effective, inp, tgt, msk, pad_id)

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
            inp, tgt, msk = _prepare_batch(batch, pad_id)
            if onp.sum(msk) == 0:
                continue
            lora_vals = [lora[k] for k in lora_keys]
            loss_val = float(loss_fn(lora_vals, inp, tgt, msk))
            grads = grad_fn(lora_vals, inp, tgt, msk)
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

    return lora, final_loss


def lora_to_delta_ternary(lora, base_params, alpha_over_r=1.0):
    """Convert LoRA to weight delta, applying ternary quantization."""
    delta = {}
    for k in base_params:
        if k == '_config' or base_params[k].ndim < 2:
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            A = _np_ternary_quantize(onp.array(lora[A_key]))
            B = _np_ternary_quantize(onp.array(lora[B_key]))
            delta[k] = onp.dot(A, B) * alpha_over_r
    return delta


# ===========================================================================
# Metrics
# ===========================================================================

def compute_pairwise_cosines(deltas_list):
    """Compute pairwise |cos| between flattened expert deltas."""
    N = len(deltas_list)
    flat = []
    for delta in deltas_list:
        vec = onp.concatenate([v.ravel() for k, v in sorted(delta.items())])
        flat.append(vec)
    flat = onp.array(flat)

    cosines = []
    for i in range(N):
        for j in range(i+1, N):
            dot = onp.dot(flat[i], flat[j])
            ni = onp.linalg.norm(flat[i])
            nj = onp.linalg.norm(flat[j])
            if ni > 1e-10 and nj > 1e-10:
                cosines.append(abs(dot / (ni * nj)))
            else:
                cosines.append(0.0)
    return cosines


def eval_ppl(base_params, delta, data_encoded, pad_id):
    """Evaluate PPL of base + delta on data."""
    effective = {}
    for k, v in base_params.items():
        if k == '_config':
            effective[k] = v
        elif k in delta:
            effective[k] = v + delta[k]
        else:
            effective[k] = v

    total_loss = 0.0
    total_tokens = 0.0
    batch_size = 32
    for i in range(0, len(data_encoded), batch_size):
        batch = data_encoded[i:i+batch_size]
        inp, tgt, msk = _prepare_batch(batch, pad_id)
        if onp.sum(msk) == 0:
            continue
        logits = forward(effective, inp, pad_id)
        B, T, V = logits.shape
        max_l = onp.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_l
        log_probs = shifted - onp.log(onp.sum(onp.exp(shifted), axis=-1, keepdims=True))
        for b in range(B):
            for t in range(T):
                if msk[b, t] > 0:
                    total_loss -= float(log_probs[b, t, tgt[b, t]])
                    total_tokens += 1.0
    avg_nll = total_loss / max(total_tokens, 1.0)
    return float(onp.exp(avg_nll))


def eval_composed_ppl(base_params, deltas_list, data_encoded_per_domain, pad_id, domains):
    """Evaluate PPL of composed model (1/N averaging) on each domain."""
    N = len(deltas_list)
    # Compose: sum deltas with 1/N weight
    composed_delta = {}
    for k in deltas_list[0]:
        composed_delta[k] = sum(d[k] for d in deltas_list) / N

    results = {}
    for dom_name, data in zip(domains, data_encoded_per_domain):
        results[dom_name] = eval_ppl(base_params, composed_delta, data, pad_id)
    results['mean'] = onp.mean(list(results.values()))
    return results


# ===========================================================================
# Main Experiment
# ===========================================================================

def run_experiment():
    print("=" * 70)
    print("Grassmannian AP Init + Ternary QAT Experiment")
    print("=" * 70)

    SEEDS = [42, 123, 314]
    DOMAINS = list(DOMAIN_GENERATORS.keys())
    N_EXPERTS = len(DOMAINS)
    N_TRAIN = 200
    N_EVAL = 50
    EPOCHS = 30
    LR = 0.003

    tok = CharTokenizer()
    all_results = {'seeds': {}, 'summary': {}}
    t_start = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # 1. Init base model and quantize to ternary
        base = init_model(tok.vocab_size, seed=seed)
        base_ternary = quantize_model_to_ternary(base)

        # 2. Generate data for all domains
        data_rng = onp.random.RandomState(seed + 1000)
        train_data = {}
        eval_data = {}
        for dom_name in DOMAINS:
            raw_train = DOMAIN_GENERATORS[dom_name](N_TRAIN, data_rng)
            raw_eval = DOMAIN_GENERATORS[dom_name](N_EVAL, data_rng)
            train_data[dom_name] = [tok.encode(s) for s in raw_train]
            eval_data[dom_name] = [tok.encode(s) for s in raw_eval]

        # 3. Compute AP skeleton for this seed
        print(f"\n  Computing AP skeleton (N={N_EXPERTS}, r={LORA_RANK}, d={D_MODEL})...")
        ap_rng = onp.random.RandomState(seed + 2000)
        ap_frames, ap_info = alternating_projection(
            N_EXPERTS, LORA_RANK, D_MODEL, n_iter=500, mu_factor=1.2, rng=ap_rng
        )
        print(f"    Welch bound: {ap_info['welch_bound']:.4f}")
        print(f"    AP mean coherence: {ap_info['mean_coherence']:.4f}")
        print(f"    AP max coherence: {ap_info['max_coherence']:.4f}")

        # 4. Train experts under both conditions
        ap_deltas = []
        rand_deltas = []
        ap_losses = []
        rand_losses = []
        ap_individual_ppls = []
        rand_individual_ppls = []

        for dom_idx, dom_name in enumerate(DOMAINS):
            print(f"\n  Domain: {dom_name}")

            # --- AP-init condition ---
            lora_ap = init_lora_ap(base_ternary, ap_frames, dom_idx, rank=LORA_RANK)
            print(f"    Training AP-init [{dom_name}]...", end=' ', flush=True)
            lora_ap, ap_loss = train_lora_ternary(
                base_ternary, lora_ap, train_data[dom_name], tok.pad_id,
                epochs=EPOCHS, lr=LR
            )
            ap_delta = lora_to_delta_ternary(lora_ap, base_ternary)
            ap_deltas.append(ap_delta)
            ap_losses.append(ap_loss)
            ap_ppl = eval_ppl(base_ternary, ap_delta, eval_data[dom_name], tok.pad_id)
            ap_individual_ppls.append(ap_ppl)
            print(f"loss={ap_loss:.4f}, PPL={ap_ppl:.3f}")

            # --- Random-init condition ---
            lora_rand = init_lora_random(
                base_ternary, rank=LORA_RANK, seed=seed * 1000 + dom_idx
            )
            print(f"    Training Random-init [{dom_name}]...", end=' ', flush=True)
            lora_rand, rand_loss = train_lora_ternary(
                base_ternary, lora_rand, train_data[dom_name], tok.pad_id,
                epochs=EPOCHS, lr=LR
            )
            rand_delta = lora_to_delta_ternary(lora_rand, base_ternary)
            rand_deltas.append(rand_delta)
            rand_losses.append(rand_loss)
            rand_ppl = eval_ppl(base_ternary, rand_delta, eval_data[dom_name], tok.pad_id)
            rand_individual_ppls.append(rand_ppl)
            print(f"loss={rand_loss:.4f}, PPL={rand_ppl:.3f}")

        # 5. Compute pairwise cosines
        ap_cosines = compute_pairwise_cosines(ap_deltas)
        rand_cosines = compute_pairwise_cosines(rand_deltas)

        ap_mean_cos = float(onp.mean(ap_cosines))
        rand_mean_cos = float(onp.mean(rand_cosines))
        ratio = ap_mean_cos / rand_mean_cos if rand_mean_cos > 1e-10 else float('inf')

        print(f"\n  --- Orthogonality Results (seed {seed}) ---")
        print(f"    AP-init   mean |cos|: {ap_mean_cos:.6f}")
        print(f"    Rand-init mean |cos|: {rand_mean_cos:.6f}")
        print(f"    Ratio (AP/Rand):      {ratio:.4f}")
        print(f"    Improvement:          {(1 - ratio)*100:.1f}%")

        # 6. Composition PPL
        eval_encoded_list = [eval_data[d] for d in DOMAINS]
        ap_composed = eval_composed_ppl(
            base_ternary, ap_deltas, eval_encoded_list, tok.pad_id, DOMAINS
        )
        rand_composed = eval_composed_ppl(
            base_ternary, rand_deltas, eval_encoded_list, tok.pad_id, DOMAINS
        )

        print(f"\n  --- Composition PPL (seed {seed}) ---")
        print(f"    AP-init   mean composed PPL: {ap_composed['mean']:.3f}")
        print(f"    Rand-init mean composed PPL: {rand_composed['mean']:.3f}")

        # 7. Individual quality comparison
        ap_mean_ppl = float(onp.mean(ap_individual_ppls))
        rand_mean_ppl = float(onp.mean(rand_individual_ppls))
        quality_ratio = ap_mean_ppl / rand_mean_ppl

        print(f"\n  --- Individual Quality (seed {seed}) ---")
        print(f"    AP-init   mean PPL: {ap_mean_ppl:.3f}")
        print(f"    Rand-init mean PPL: {rand_mean_ppl:.3f}")
        print(f"    Quality ratio (AP/Rand): {quality_ratio:.4f}")

        # Store
        all_results['seeds'][str(seed)] = {
            'ap_skeleton': ap_info,
            'ap_cosines': ap_cosines,
            'rand_cosines': rand_cosines,
            'ap_mean_cos': ap_mean_cos,
            'rand_mean_cos': rand_mean_cos,
            'cos_ratio': ratio,
            'ap_losses': ap_losses,
            'rand_losses': rand_losses,
            'ap_individual_ppls': ap_individual_ppls,
            'rand_individual_ppls': rand_individual_ppls,
            'ap_mean_ppl': ap_mean_ppl,
            'rand_mean_ppl': rand_mean_ppl,
            'quality_ratio': quality_ratio,
            'ap_composed': ap_composed,
            'rand_composed': rand_composed,
        }

    # ===========================================================
    # Aggregate across seeds
    # ===========================================================
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS (3 seeds)")
    print("=" * 70)

    all_ap_cos = []
    all_rand_cos = []
    all_ratios = []
    all_quality_ratios = []
    all_ap_composed_means = []
    all_rand_composed_means = []

    for s in SEEDS:
        r = all_results['seeds'][str(s)]
        all_ap_cos.extend(r['ap_cosines'])
        all_rand_cos.extend(r['rand_cosines'])
        all_ratios.append(r['cos_ratio'])
        all_quality_ratios.append(r['quality_ratio'])
        all_ap_composed_means.append(r['ap_composed']['mean'])
        all_rand_composed_means.append(r['rand_composed']['mean'])

    agg_ap_cos = float(onp.mean(all_ap_cos))
    agg_rand_cos = float(onp.mean(all_rand_cos))
    agg_ratio = agg_ap_cos / agg_rand_cos if agg_rand_cos > 1e-10 else float('inf')
    agg_quality = float(onp.mean(all_quality_ratios))
    agg_ap_composed = float(onp.mean(all_ap_composed_means))
    agg_rand_composed = float(onp.mean(all_rand_composed_means))

    # Wilcoxon signed-rank test on paired cosines
    try:
        stat, p_value = wilcoxon(all_ap_cos, all_rand_cos, alternative='less')
        p_val = float(p_value)
    except Exception as e:
        stat, p_val = 0.0, 1.0
        print(f"  Wilcoxon test failed: {e}")

    print(f"\n  Orthogonality:")
    print(f"    AP-init   aggregate mean |cos|: {agg_ap_cos:.6f}")
    print(f"    Rand-init aggregate mean |cos|: {agg_rand_cos:.6f}")
    print(f"    Aggregate ratio (AP/Rand):      {agg_ratio:.4f}")
    print(f"    Per-seed ratios: {[f'{r:.4f}' for r in all_ratios]}")
    print(f"    Wilcoxon p-value (AP < Rand):   {p_val:.6f}")
    print(f"    Improvement:                    {(1 - agg_ratio)*100:.1f}%")

    print(f"\n  Individual Quality:")
    print(f"    AP-init   mean PPL: {float(onp.mean([all_results['seeds'][str(s)]['ap_mean_ppl'] for s in SEEDS])):.3f}")
    print(f"    Rand-init mean PPL: {float(onp.mean([all_results['seeds'][str(s)]['rand_mean_ppl'] for s in SEEDS])):.3f}")
    print(f"    Quality ratio (AP/Rand): {agg_quality:.4f}")

    print(f"\n  Composition PPL:")
    print(f"    AP-init   mean composed: {agg_ap_composed:.3f}")
    print(f"    Rand-init mean composed: {agg_rand_composed:.3f}")
    print(f"    Composed ratio (AP/Rand): {agg_ap_composed/agg_rand_composed:.4f}")

    # ===========================================================
    # Kill criteria assessment
    # ===========================================================
    k1_pass = agg_ratio < 0.7  # AP shows >30% improvement
    k1_threshold = 0.7
    k2_pass = agg_quality <= 1.05  # AP not >5% worse
    k2_threshold = 1.05

    print(f"\n  Kill Criteria:")
    print(f"    K1: AP/Rand ratio = {agg_ratio:.4f} (threshold < {k1_threshold})")
    print(f"        {'PASS' if k1_pass else 'FAIL (KILLED)'}: AP {'does' if k1_pass else 'does NOT'} show >30% improvement")
    print(f"    K2: Quality ratio = {agg_quality:.4f} (threshold < {k2_threshold})")
    print(f"        {'PASS' if k2_pass else 'FAIL (KILLED)'}: AP {'does not' if k2_pass else 'DOES'} degrade quality >5%")

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    if not k1_pass and k2_pass:
        verdict = "KILLED (K1: insufficient orthogonality improvement)"
    elif k1_pass and not k2_pass:
        verdict = "KILLED (K2: quality degradation)"
    elif not k1_pass and not k2_pass:
        verdict = "KILLED (K1 + K2)"

    print(f"\n  VERDICT: {verdict}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    # Store summary
    all_results['summary'] = {
        'agg_ap_mean_cos': agg_ap_cos,
        'agg_rand_mean_cos': agg_rand_cos,
        'agg_ratio': agg_ratio,
        'per_seed_ratios': all_ratios,
        'wilcoxon_p': p_val,
        'improvement_pct': (1 - agg_ratio) * 100,
        'agg_quality_ratio': agg_quality,
        'agg_ap_composed_ppl': agg_ap_composed,
        'agg_rand_composed_ppl': agg_rand_composed,
        'composed_ratio': agg_ap_composed / agg_rand_composed,
        'k1_pass': k1_pass,
        'k1_ratio': agg_ratio,
        'k1_threshold': k1_threshold,
        'k2_pass': k2_pass,
        'k2_ratio': agg_quality,
        'k2_threshold': k2_threshold,
        'verdict': verdict,
        'elapsed_seconds': elapsed,
    }

    # Save
    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Results saved to {out_path}")

    return all_results


if __name__ == '__main__':
    run_experiment()
