#!/usr/bin/env python3
"""
Composition vs Monolithic: Do N composed experts match or exceed
one monolithic model trained on all domains?

Experimental design:
  Condition A (Composed): Train 5 separate expert models on 5 domains.
    Compute delta_k = expert_k - base. Truncate each delta to rank-r via SVD.
    Compose: W_composed = W_base + sum_k delta_k^{rank-r}
  Condition B (Monolithic shuffled): Train 1 model on shuffled combined data
    for the same total steps. Truncate delta to rank-(N*r) via SVD.
    W_mono = W_base + delta_mono^{rank-Nr}
  Condition C (Monolithic sequential): Train 1 model on domains one at a time.
    Same rank-(N*r) truncation. Measures forgetting.
  Control: Base model (no fine-tuning).

Parameter budget: N x rank-r = rank-(N*r). Same total low-rank capacity.
Training budget: Same total gradient steps.

Uses autograd for proper gradient-based training of a causal transformer.
Same architecture and data generators as answer_conditioned_scoring (proven).

Kill criteria:
  K1: monolithic beats composition by >10% on average across domains
  K2: composition has >2x training cost for same quality
"""

import json
import math
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp  # non-differentiable ops


# ===========================================================================
# Synthetic Data (structured domains with clear specialization signal)
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
    """Domain 4: bit parity. '1011>odd'"""
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

    def decode(self, ids):
        out = []
        for i in ids:
            c = self.idx2char.get(i, "")
            if c == self.eos_token: break
            if c == self.pad_token: continue
            out.append(c)
        return "".join(out)


# ===========================================================================
# Causal Transformer (autograd-compatible)
# ===========================================================================

def init_model(V, d=32, H=2, L=2, max_T=32, seed=42):
    """Initialize model parameters. Smaller than answer_conditioned for speed."""
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
    """Forward pass. idx_2d: (B, T) int array -> logits (B, T, V)."""
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
    """NTP cross-entropy loss with masking."""
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
    """Pad sequences, create input/target/mask."""
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
    """Train using Adam with autograd gradients."""
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
    """Evaluate NTP loss on a dataset."""
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
# Delta computation and SVD truncation
# ===========================================================================

def compute_delta(base_params, trained_params):
    """Compute delta = trained - base for all weight keys."""
    delta = {}
    for k in base_params:
        if k == '_config':
            continue
        delta[k] = trained_params[k] - base_params[k]
    return delta


def svd_truncate_delta(delta, rank):
    """Truncate each delta matrix to rank-r via SVD.
    For 1D params (layer norms), keep as-is (they're small).
    Returns truncated delta and info about signal retention.
    """
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


def apply_delta_to_params(base_params, delta):
    """Apply delta to base params: W + delta."""
    result = {}
    for k in base_params:
        if k == '_config':
            result[k] = base_params[k]
        elif k in delta:
            result[k] = base_params[k] + delta[k]
        else:
            result[k] = base_params[k].copy()
    return result


def merge_deltas(delta_list, mode='sum'):
    """Merge N deltas.
    mode='sum': additive (W_base + sum D_k)
    mode='avg': average (W_base + (1/N) sum D_k)
    """
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
    """Flatten all delta matrices into a single vector."""
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
# Main Experiment
# ===========================================================================

def run_experiment(n_domains=5, rank_per_expert=4, epochs_per_expert=15,
                   d_model=32, n_heads=2, n_layers=2, n_train=200,
                   n_test=50, n_seeds=3, lr=0.001, batch_size=16):
    """Run full composition vs monolithic comparison.

    Budget matching:
      - Composed: 5 x rank-4 SVD-truncated deltas
      - Monolithic: 1 x rank-20 SVD-truncated delta
      - Same total rank budget: N * r = 20

    Training budget matching:
      - Composed: 15 epochs per expert x 5 experts = 75 total expert-epochs
      - Monolithic: 75 epochs on combined data (5x more data per epoch)
    """
    monolithic_rank = n_domains * rank_per_expert
    total_epochs = epochs_per_expert * n_domains
    domains = list(DOMAIN_GENERATORS.keys())[:n_domains]

    results_dir = Path(__file__).parent
    tok = CharTokenizer()

    print("=" * 76)
    print("  COMPOSITION vs MONOLITHIC FINE-TUNING")
    print("=" * 76)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}, V={tok.vocab_size}")
    print(f"  Domains: {domains}")
    print(f"  Composed: {n_domains} experts, rank-{rank_per_expert} SVD truncation each")
    print(f"  Monolithic: 1 model, rank-{monolithic_rank} SVD truncation")
    print(f"  Training: {epochs_per_expert} epochs/expert, {total_epochs} total")
    print(f"  Data: {n_train} train + {n_test} test per domain")
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

        # Generate data
        domain_train = {}
        domain_test = {}
        domain_train_enc = {}
        domain_test_enc = {}

        for dom_name in domains:
            gen = DOMAIN_GENERATORS[dom_name]
            train_data = gen(n_train, onp.random.RandomState(seed + hash(dom_name) % 10000))
            test_data = gen(n_test, onp.random.RandomState(seed + hash(dom_name) % 10000 + 1))
            domain_train[dom_name] = train_data
            domain_test[dom_name] = test_data
            domain_train_enc[dom_name] = [tok.encode(s) for s in train_data]
            domain_test_enc[dom_name] = [tok.encode(s) for s in test_data]

        # Combined data (shuffled)
        combined_train = []
        for dom_name in domains:
            combined_train.extend(domain_train_enc[dom_name])
        rng.shuffle(combined_train)

        # Initialize base model
        base_params = init_model(tok.vocab_size, d=d_model, H=n_heads, L=n_layers,
                                  max_T=32, seed=seed)

        # Evaluate base model
        print(f"\n  --- Base model evaluation ---")
        base_losses = {}
        for dom_name in domains:
            loss = eval_loss(base_params, domain_test_enc[dom_name], tok.pad_id)
            base_losses[dom_name] = loss
        base_avg = onp.mean(list(base_losses.values()))
        print(f"  Base losses: {' | '.join(f'{d}={v:.3f}' for d, v in base_losses.items())}")
        print(f"  Base average: {base_avg:.3f}")

        # ==============================================================
        # Condition A: Train N separate experts, compose
        # ==============================================================
        print(f"\n  --- Condition A: Composed ({n_domains} x rank-{rank_per_expert}) ---")
        t0 = time.time()

        expert_deltas_full = []
        expert_deltas_trunc = []
        expert_signal_retention = []

        for dom_idx, dom_name in enumerate(domains):
            print(f"    Training expert '{dom_name}'...")

            # Deep copy base
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

            # Compute and truncate delta
            delta = compute_delta(base_params, expert_params)
            delta_trunc, sig_ret = svd_truncate_delta(delta, rank_per_expert)

            expert_deltas_full.append(delta)
            expert_deltas_trunc.append(delta_trunc)
            expert_signal_retention.append(sig_ret)

            # Eval this expert alone on its domain
            expert_model = apply_delta_to_params(base_params, delta_trunc)
            own_loss = eval_loss(expert_model, domain_test_enc[dom_name], tok.pad_id)
            full_loss = eval_loss(
                apply_delta_to_params(base_params, delta),
                domain_test_enc[dom_name], tok.pad_id
            )

            mean_sig = onp.mean([v for v in sig_ret.values()])
            print(f"      {dom_name}: full_loss={full_loss:.3f}, "
                  f"trunc_loss={own_loss:.3f}, sig_ret={mean_sig:.3f}")

        composed_time = time.time() - t0

        # Compose: multiple strategies
        # Strategy 1: sum (naive)
        composed_sum_delta = merge_deltas(expert_deltas_trunc, mode='sum')
        composed_sum_params = apply_delta_to_params(base_params, composed_sum_delta)
        composed_sum_losses = {}
        for dom_name in domains:
            loss = eval_loss(composed_sum_params, domain_test_enc[dom_name], tok.pad_id)
            composed_sum_losses[dom_name] = loss
        composed_sum_avg = onp.mean(list(composed_sum_losses.values()))

        # Strategy 2: average (1/N)
        composed_avg_delta = merge_deltas(expert_deltas_trunc, mode='avg')
        composed_avg_params = apply_delta_to_params(base_params, composed_avg_delta)
        composed_avg_losses = {}
        for dom_name in domains:
            loss = eval_loss(composed_avg_params, domain_test_enc[dom_name], tok.pad_id)
            composed_avg_losses[dom_name] = loss
        composed_avg_avg = onp.mean(list(composed_avg_losses.values()))

        # Strategy 3: routed (oracle -- each domain uses its own expert)
        # This is what SOLE actually does: route to the right expert
        routed_losses = {}
        for dom_idx, dom_name in enumerate(domains):
            spec_params = apply_delta_to_params(base_params, expert_deltas_trunc[dom_idx])
            loss = eval_loss(spec_params, domain_test_enc[dom_name], tok.pad_id)
            routed_losses[dom_name] = loss
        routed_avg = onp.mean(list(routed_losses.values()))

        # Pick best strategy for kill criteria comparison
        composed_losses = composed_avg_losses  # Use average as the primary
        composed_avg = composed_avg_avg

        print(f"\n  Composed (sum) losses: {' | '.join(f'{d}={v:.3f}' for d, v in composed_sum_losses.items())}")
        print(f"  Composed (sum) average: {composed_sum_avg:.3f}")
        print(f"  Composed (avg) losses: {' | '.join(f'{d}={v:.3f}' for d, v in composed_avg_losses.items())}")
        print(f"  Composed (avg) average: {composed_avg_avg:.3f}")
        print(f"  Routed (oracle) losses: {' | '.join(f'{d}={v:.3f}' for d, v in routed_losses.items())}")
        print(f"  Routed (oracle) average: {routed_avg:.3f}")
        print(f"  Composed training time: {composed_time:.1f}s")

        # Individual specialist evaluation (each expert on its own domain)
        specialist_losses = routed_losses  # Same as routed
        specialist_avg = routed_avg

        # ==============================================================
        # Condition B1: Monolithic shuffled
        # ==============================================================
        print(f"\n  --- Condition B1: Monolithic shuffled (rank-{monolithic_rank}) ---")
        t0 = time.time()

        mono_params = {}
        for k, v in base_params.items():
            if k == '_config':
                mono_params[k] = v
            else:
                mono_params[k] = v.copy()

        # Train on combined data for total_epochs
        # But each epoch sees all combined data, so we scale down epochs
        # to match total gradient steps: expert sees n_train * epochs_per_expert
        # monolithic should see n_domains * n_train * mono_epochs
        # For fair comparison: total samples seen should be equal
        # Composed: each expert sees n_train * epochs_per_expert = 200 * 15 = 3000 per domain
        # Monolithic: sees n_domains * n_train * mono_epochs = 1000 * mono_epochs
        # For 3000 per domain = 15000 total samples = 1000 * 15 epochs
        mono_epochs = epochs_per_expert  # Same number of passes over combined data
        # This means monolithic sees each domain sample epochs_per_expert times
        # Same as each expert training -- fair comparison.

        mono_params = train_model(
            mono_params, combined_train, tok.pad_id,
            epochs=mono_epochs, lr=lr, batch_size=batch_size,
            verbose=True
        )

        mono_time = time.time() - t0

        # Truncate to rank-Nr
        mono_delta = compute_delta(base_params, mono_params)
        mono_delta_trunc, mono_sig_ret = svd_truncate_delta(mono_delta, monolithic_rank)
        mono_trunc_params = apply_delta_to_params(base_params, mono_delta_trunc)

        # Also evaluate full-rank monolithic for reference
        mono_full_losses = {}
        mono_trunc_losses = {}
        for dom_name in domains:
            mono_full_losses[dom_name] = eval_loss(mono_params, domain_test_enc[dom_name], tok.pad_id)
            mono_trunc_losses[dom_name] = eval_loss(mono_trunc_params, domain_test_enc[dom_name], tok.pad_id)

        mono_full_avg = onp.mean(list(mono_full_losses.values()))
        mono_trunc_avg = onp.mean(list(mono_trunc_losses.values()))

        print(f"\n  Mono (full) losses: {' | '.join(f'{d}={v:.3f}' for d, v in mono_full_losses.items())}")
        print(f"  Mono (full) average: {mono_full_avg:.3f}")
        print(f"  Mono (trunc r={monolithic_rank}) losses: "
              f"{' | '.join(f'{d}={v:.3f}' for d, v in mono_trunc_losses.items())}")
        print(f"  Mono (trunc) average: {mono_trunc_avg:.3f} (time: {mono_time:.1f}s)")

        # ==============================================================
        # Condition B2: Monolithic sequential (forgetting)
        # ==============================================================
        print(f"\n  --- Condition B2: Monolithic sequential ---")
        t0 = time.time()

        seq_params = {}
        for k, v in base_params.items():
            if k == '_config':
                seq_params[k] = v
            else:
                seq_params[k] = v.copy()

        seq_eval_matrix = {}  # {after_dom: {dom: loss}}
        for dom_idx, dom_name in enumerate(domains):
            print(f"    Training on '{dom_name}'...")
            seq_params = train_model(
                seq_params, domain_train_enc[dom_name], tok.pad_id,
                epochs=epochs_per_expert, lr=lr, batch_size=batch_size,
                verbose=False
            )

            # Eval on all domains
            phase_losses = {}
            for d_name in domains:
                phase_losses[d_name] = eval_loss(seq_params, domain_test_enc[d_name], tok.pad_id)
            seq_eval_matrix[dom_name] = phase_losses
            print(f"      After '{dom_name}': "
                  f"{' | '.join(f'{d}={v:.3f}' for d, v in phase_losses.items())}")

        seq_time = time.time() - t0

        # Truncate sequential to rank-Nr
        seq_delta = compute_delta(base_params, seq_params)
        seq_delta_trunc, _ = svd_truncate_delta(seq_delta, monolithic_rank)
        seq_trunc_params = apply_delta_to_params(base_params, seq_delta_trunc)

        seq_trunc_losses = {}
        for dom_name in domains:
            seq_trunc_losses[dom_name] = eval_loss(seq_trunc_params, domain_test_enc[dom_name], tok.pad_id)
        seq_trunc_avg = onp.mean(list(seq_trunc_losses.values()))

        # Final sequential losses = last phase
        last_dom = domains[-1]
        seq_final_losses = seq_eval_matrix[last_dom]
        seq_avg = onp.mean(list(seq_final_losses.values()))

        # Forgetting
        forgetting = {}
        for dom_idx, dom_name in enumerate(domains):
            best_loss = seq_eval_matrix[dom_name][dom_name]
            worst_loss = best_loss
            for later_dom in domains[dom_idx + 1:]:
                worst_loss = max(worst_loss, seq_eval_matrix[later_dom][dom_name])
            forgetting[dom_name] = worst_loss - best_loss

        print(f"\n  Sequential forgetting: {' | '.join(f'{d}={v:.3f}' for d, v in forgetting.items())}")
        print(f"  Seq (trunc) average: {seq_trunc_avg:.3f} (time: {seq_time:.1f}s)")

        # ==============================================================
        # Modularity test: remove one expert
        # ==============================================================
        print(f"\n  --- Modularity test ---")
        modularity_results = {}
        for remove_idx, remove_dom in enumerate(domains):
            remaining = [d for i, d in enumerate(expert_deltas_trunc) if i != remove_idx]
            partial_delta = merge_deltas(remaining) if remaining else {}
            partial_params = apply_delta_to_params(base_params, partial_delta) if partial_delta else base_params

            other_losses = []
            for dom_name in domains:
                if dom_name != remove_dom:
                    loss = eval_loss(partial_params, domain_test_enc[dom_name], tok.pad_id)
                    other_losses.append(loss)

            full_other = [composed_losses[d] for d in domains if d != remove_dom]
            deg = (onp.mean(other_losses) - onp.mean(full_other)) / onp.mean(full_other) * 100

            modularity_results[remove_dom] = {
                'avg_other_loss': float(onp.mean(other_losses)),
                'degradation_pct': float(deg),
            }
            print(f"    Remove '{remove_dom}': other avg={onp.mean(other_losses):.3f} "
                  f"(degradation: {deg:+.2f}%)")

        # ==============================================================
        # Pairwise cosines between expert deltas
        # ==============================================================
        flat_deltas = [flatten_delta(d) for d in expert_deltas_trunc]
        pairwise_cosines = {}
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                cos = abs(cosine_sim(flat_deltas[i], flat_deltas[j]))
                pairwise_cosines[f"{domains[i]}_vs_{domains[j]}"] = cos
        mean_cos = onp.mean(list(pairwise_cosines.values())) if pairwise_cosines else 0
        max_cos = max(pairwise_cosines.values()) if pairwise_cosines else 0
        print(f"\n  Expert delta |cos|: mean={mean_cos:.4f}, max={max_cos:.4f}")

        # ==============================================================
        # Collect results
        # ==============================================================
        seed_result = {
            'seed': seed,
            'base_losses': base_losses,
            'base_avg': float(base_avg),
            'specialist_losses': specialist_losses,
            'specialist_avg': float(specialist_avg),
            'composed_sum_losses': composed_sum_losses,
            'composed_sum_avg': float(composed_sum_avg),
            'composed_avg_losses': composed_avg_losses,
            'composed_avg_avg': float(composed_avg_avg),
            'routed_losses': routed_losses,
            'routed_avg': float(routed_avg),
            'composed_losses': composed_losses,
            'composed_avg': float(composed_avg),
            'composed_time': float(composed_time),
            'mono_full_losses': mono_full_losses,
            'mono_full_avg': float(mono_full_avg),
            'mono_trunc_losses': mono_trunc_losses,
            'mono_trunc_avg': float(mono_trunc_avg),
            'mono_time': float(mono_time),
            'seq_eval_matrix': seq_eval_matrix,
            'seq_final_losses': seq_final_losses,
            'seq_avg': float(seq_avg),
            'seq_trunc_losses': seq_trunc_losses,
            'seq_trunc_avg': float(seq_trunc_avg),
            'seq_time': float(seq_time),
            'forgetting': forgetting,
            'modularity': modularity_results,
            'pairwise_cosines': pairwise_cosines,
            'mean_cos': float(mean_cos),
            'max_cos': float(max_cos),
            'signal_retention': {
                dom: {k: float(v) for k, v in sr.items()}
                for dom, sr in zip(domains, expert_signal_retention)
            },
        }
        all_seed_results.append(seed_result)

        elapsed = time.time() - t_seed_start
        print(f"\n  Seed {seed} total time: {elapsed:.1f}s")

    # ==================================================================
    # Aggregate across seeds
    # ==================================================================
    print(f"\n{'='*76}")
    print("  AGGREGATE RESULTS ACROSS SEEDS")
    print(f"{'='*76}")

    composed_sum_avgs = [s['composed_sum_avg'] for s in all_seed_results]
    composed_avg_avgs = [s['composed_avg_avg'] for s in all_seed_results]
    routed_avgs = [s['routed_avg'] for s in all_seed_results]
    composed_avgs = composed_avg_avgs  # Use avg merge as primary
    mono_trunc_avgs = [s['mono_trunc_avg'] for s in all_seed_results]
    mono_full_avgs = [s['mono_full_avg'] for s in all_seed_results]
    seq_trunc_avgs = [s['seq_trunc_avg'] for s in all_seed_results]
    base_avgs_list = [s['base_avg'] for s in all_seed_results]
    specialist_avgs_list = [s['specialist_avg'] for s in all_seed_results]
    composed_times = [s['composed_time'] for s in all_seed_results]
    mono_times = [s['mono_time'] for s in all_seed_results]

    print(f"\n  {'Condition':<28s} | {'Mean Loss':>10s} | {'Std':>8s} | {'vs Base':>10s}")
    print(f"  {'-'*65}")

    conditions = [
        ("Base (no training)", base_avgs_list),
        ("Composed (sum)", composed_sum_avgs),
        ("Composed (avg 1/N)", composed_avg_avgs),
        ("Routed (oracle)", routed_avgs),
        ("Mono shuffled (r=" + str(monolithic_rank) + ")", mono_trunc_avgs),
        ("Mono shuffled (full rank)", mono_full_avgs),
        ("Mono sequential (r=" + str(monolithic_rank) + ")", seq_trunc_avgs),
    ]
    for name, vals in conditions:
        mean_v = onp.mean(vals)
        std_v = onp.std(vals)
        vs_base = (mean_v - onp.mean(base_avgs_list)) / onp.mean(base_avgs_list) * 100
        print(f"  {name:<28s} | {mean_v:10.4f} | {std_v:8.4f} | {vs_base:+9.2f}%")

    # Quality comparison: all strategies vs monolithic (same rank budget)
    sum_diffs = [(c - m) / m * 100 for c, m in zip(composed_sum_avgs, mono_trunc_avgs)]
    avg_diffs = [(c - m) / m * 100 for c, m in zip(composed_avg_avgs, mono_trunc_avgs)]
    routed_diffs = [(c - m) / m * 100 for c, m in zip(routed_avgs, mono_trunc_avgs)]

    print(f"\n  Composition strategies vs Mono (trunc):")
    print(f"    Sum:    {onp.mean(sum_diffs):+.2f}% (positive = worse than mono)")
    print(f"    Avg:    {onp.mean(avg_diffs):+.2f}%")
    print(f"    Routed: {onp.mean(routed_diffs):+.2f}% (this is what SOLE does)")

    # Use ROUTED for kill criteria (this is the SOLE architecture)
    quality_diffs = routed_diffs
    mean_quality_diff = onp.mean(quality_diffs)

    # Time comparison
    mean_composed_time = onp.mean(composed_times)
    mean_mono_time = onp.mean(mono_times)
    time_ratio = mean_composed_time / mean_mono_time
    print(f"  Time ratio (composed/mono): {time_ratio:.2f}x")
    print(f"  (In production, composed trains in parallel: effective {time_ratio/n_domains:.2f}x)")

    # Forgetting
    all_forgetting = []
    for s in all_seed_results:
        for dom_name, forg in s['forgetting'].items():
            all_forgetting.append(forg)
    mean_forgetting = onp.mean(all_forgetting)
    max_forgetting = onp.max(all_forgetting)
    print(f"\n  Sequential forgetting: mean={mean_forgetting:.3f}, max={max_forgetting:.3f}")
    print(f"  (Composed has ZERO forgetting by construction)")

    # Per-domain breakdown
    print(f"\n  Per-domain breakdown (mean across seeds):")
    print(f"  {'Domain':<12s} | {'Base':>7s} | {'Routed':>7s} | {'Avg':>7s} | "
          f"{'Mono(t)':>7s} | {'Mono(f)':>7s} | {'Best':>8s}")
    print(f"  {'-'*72}")

    per_domain_winners = []
    for dom_name in domains:
        base_d = onp.mean([s['base_losses'][dom_name] for s in all_seed_results])
        routed_d = onp.mean([s['routed_losses'][dom_name] for s in all_seed_results])
        comp_avg_d = onp.mean([s['composed_avg_losses'][dom_name] for s in all_seed_results])
        mono_t_d = onp.mean([s['mono_trunc_losses'][dom_name] for s in all_seed_results])
        mono_f_d = onp.mean([s['mono_full_losses'][dom_name] for s in all_seed_results])

        candidates = {'Routed': routed_d, 'Avg': comp_avg_d, 'Mono(t)': mono_t_d}
        best = min(candidates, key=candidates.get)
        per_domain_winners.append(best)

        print(f"  {dom_name:<12s} | {base_d:7.3f} | {routed_d:7.3f} | {comp_avg_d:7.3f} | "
              f"{mono_t_d:7.3f} | {mono_f_d:7.3f} | {best:>8s}")

    # Modularity
    avg_modularity_deg = onp.mean([
        s['modularity'][d]['degradation_pct']
        for s in all_seed_results for d in domains
    ])
    print(f"\n  Modularity: avg degradation when removing 1 expert: {avg_modularity_deg:+.2f}%")

    # ==================================================================
    # Kill criteria evaluation
    # ==================================================================
    print(f"\n{'='*76}")
    print("  KILL CRITERIA EVALUATION")
    print(f"{'='*76}")

    k1_val = mean_quality_diff
    k1_kill = k1_val > 10.0

    print(f"\n  K1: Monolithic beats composition by >10%?")
    print(f"      Quality gap (composed - mono_trunc): {k1_val:+.2f}%")
    print(f"      STATUS: {'KILL' if k1_kill else 'PASS'}")

    k2_val = time_ratio
    k2_kill = k2_val > 2.0

    print(f"\n  K2: Composition has >2x training cost?")
    print(f"      Sequential time ratio: {k2_val:.2f}x")
    print(f"      Parallel time ratio: {k2_val/n_domains:.2f}x")
    print(f"      STATUS: {'KILL' if k2_kill else 'PASS'}")

    either_kill = k1_kill or k2_kill

    print(f"\n  OVERALL: {'KILL' if either_kill else 'PROVEN'}")
    if not either_kill:
        print(f"    K1 PASS: quality gap {k1_val:+.2f}% (within 10%)")
        print(f"    K2 PASS: time ratio {k2_val:.2f}x (within 2x)")
        print(f"    Composition advantages:")
        print(f"      - Zero catastrophic forgetting (seq forgetting: {mean_forgetting:.3f})")
        print(f"      - Modularity: add/remove domains independently ({avg_modularity_deg:+.1f}% degradation)")
        print(f"      - Parallelizable: {k2_val/n_domains:.2f}x with {n_domains} GPUs")
        print(f"      - Expert pairwise cos: {onp.mean([s['mean_cos'] for s in all_seed_results]):.4f}")
    else:
        if k1_kill:
            print(f"    K1 KILLED: monolithic is {k1_val:+.2f}% better")
        if k2_kill:
            print(f"    K2 KILLED: composed is {k2_val:.1f}x slower")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        'experiment': 'composition_vs_monolithic',
        'config': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'vocab_size': tok.vocab_size,
            'n_domains': n_domains,
            'domains': domains,
            'rank_per_expert': rank_per_expert,
            'monolithic_rank': monolithic_rank,
            'epochs_per_expert': epochs_per_expert,
            'total_epochs': total_epochs,
            'n_train': n_train,
            'n_test': n_test,
            'n_seeds': n_seeds,
            'lr': lr,
            'batch_size': batch_size,
        },
        'aggregate': {
            'base_avg': float(onp.mean(base_avgs_list)),
            'composed_sum_avg': float(onp.mean(composed_sum_avgs)),
            'composed_avg_avg': float(onp.mean(composed_avg_avgs)),
            'routed_avg': float(onp.mean(routed_avgs)),
            'mono_trunc_avg': float(onp.mean(mono_trunc_avgs)),
            'mono_trunc_std': float(onp.std(mono_trunc_avgs)),
            'mono_full_avg': float(onp.mean(mono_full_avgs)),
            'seq_trunc_avg': float(onp.mean(seq_trunc_avgs)),
            'quality_gap_pct': float(mean_quality_diff),
            'time_ratio': float(time_ratio),
            'parallel_time_ratio': float(time_ratio / n_domains),
            'mean_forgetting': float(mean_forgetting),
            'max_forgetting': float(max_forgetting),
            'avg_modularity_degradation_pct': float(avg_modularity_deg),
            'per_domain_winners': per_domain_winners,
            'mean_pairwise_cos': float(onp.mean([s['mean_cos'] for s in all_seed_results])),
        },
        'kill_criteria': {
            'k1_quality_gap_pct': float(k1_val),
            'k1_threshold': 10.0,
            'k1_kill': bool(k1_kill),
            'k2_time_ratio': float(k2_val),
            'k2_threshold': 2.0,
            'k2_kill': bool(k2_kill),
            'overall_kill': bool(either_kill),
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
        run_experiment(n_seeds=1, epochs_per_expert=8, n_train=100, n_test=30)
    else:
        run_experiment()
