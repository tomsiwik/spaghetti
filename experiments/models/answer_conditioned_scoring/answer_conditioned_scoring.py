#!/usr/bin/env python3
"""Answer-Conditioned Scoring: answer-only PPL correlates with task accuracy
where full-sequence PPL fails.

Direct follow-up to the KILLED ppl_vs_task_performance experiment, which showed
Pearson r=0.084 between full-sequence PPL improvement and task accuracy improvement.
Root cause: full-sequence PPL averages over prompt + answer tokens, diluting
the signal from answer tokens that determine task correctness.

Fix: compute PPL only over answer tokens (after the domain delimiter).
Same 5 domains, same setup, but with both full-sequence and answer-only PPL.

CPU-only, numpy + autograd (no PyTorch, no MLX, no GPU).
Autograd provides automatic differentiation for numpy code, enabling
proper gradient-based training of a transformer model.

Usage:
    uv run python -m micro.models.answer_conditioned_scoring.answer_conditioned_scoring
    uv run python -m micro.models.answer_conditioned_scoring.answer_conditioned_scoring --seeds 3
"""

import argparse
import json
import math
import random
import time
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            result = obj.tolist()
            if isinstance(result, (int, float)):
                return result
            return result
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

import autograd.numpy as np
from autograd import grad
import numpy as onp  # original numpy for non-differentiable ops


# ── Synthetic Data Generation ──────────────────────────────────────────────

def _make_arithmetic_data(n, rng):
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        data.append(f"{a}+{b}={a+b}")
    return data

def _make_reverse_data(n, rng):
    chars = "abcdefghijklmnopqrstuvwxyz"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(chars) for _ in range(length))
        data.append(f"{s}>{s[::-1]}")
    return data

def _make_repeat_data(n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(chars) for _ in range(plen))
        rep = rng.randint(2, 4)
        data.append(f"{pat}*{rep}={pat * rep}")
    return data

def _make_sort_data(n, rng):
    chars = "abcdefghijklmnop"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(chars) for _ in range(length))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data

def _make_parity_data(n, rng):
    data = []
    for _ in range(n):
        length = rng.randint(2, 8)
        bits = "".join(str(rng.randint(0, 1)) for _ in range(length))
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

DOMAIN_DELIMITERS = {
    "arithmetic": "=",
    "reverse": ">",
    "repeat": "=",
    "sort": ">",
    "parity": ">",
}


# ── Tokenizer ─────────────────────────────────────────────────────────────

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


# ── Causal Transformer (autograd-compatible) ───────────────────────────────
# All operations use autograd.numpy so gradients flow through.
# We represent the model as a single flat dict of arrays.

def init_model(V, d=64, H=4, L=4, max_T=48, seed=42):
    """Initialize model parameters as a dict of numpy arrays."""
    rng = onp.random.RandomState(seed)
    s = 0.02
    hd = d // H
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
    """Forward pass. idx_2d: (B, T) int array. Returns logits (B, T, V)."""
    cfg = params['_config']
    d, H, L = cfg['d'], cfg['H'], cfg['L']
    hd = d // H
    B, T = idx_2d.shape

    # Embedding
    x = params['tok_emb'][idx_2d] + params['pos_emb'][:T]  # (B, T, d)

    # Causal mask
    mask = onp.triu(onp.ones((T, T)) * (-1e9), k=1).astype(onp.float32)

    for li in range(L):
        # Pre-norm attention
        h = _rms_norm(x, params[f'ln1_w_{li}'])

        # QKV
        qkv = np.dot(h, params[f'Wqkv_{li}'])  # (B, T, 3*d)
        qkv = np.reshape(qkv, (B, T, 3, H, hd))
        q = qkv[:, :, 0, :, :]  # (B, T, H, hd)
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = np.transpose(q, (0, 2, 1, 3))  # (B, H, T, hd)
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = 1.0 / onp.sqrt(hd)
        attn = np.einsum('bhqd,bhkd->bhqk', q, k) * scale + mask
        attn = attn - np.max(attn, axis=-1, keepdims=True)
        attn = np.exp(attn)
        attn = attn / np.sum(attn, axis=-1, keepdims=True)

        out = np.einsum('bhqk,bhkd->bhqd', attn, v)  # (B, H, T, hd)
        out = np.transpose(out, (0, 2, 1, 3))  # (B, T, H, hd)
        out = np.reshape(out, (B, T, d))
        out = np.dot(out, params[f'Wo_{li}'])

        x = x + out

        # Pre-norm FFN
        h = _rms_norm(x, params[f'ln2_w_{li}'])
        ffn = np.maximum(0, np.dot(h, params[f'W1_{li}']))
        ffn = np.dot(ffn, params[f'W2_{li}'])
        x = x + ffn

    x = _rms_norm(x, params['ln_f_w'])
    logits = np.dot(x, params['W_head'])  # (B, T, V)
    return logits


def compute_loss(params, idx_2d, targets_2d, mask_2d, pad_id=0):
    """Compute mean cross-entropy loss.
    idx_2d: (B, T) input tokens
    targets_2d: (B, T) target tokens
    mask_2d: (B, T) float mask (1=valid, 0=pad)
    """
    logits = forward(params, idx_2d, pad_id)  # (B, T, V)
    B, T, V = logits.shape

    # Log-softmax
    max_l = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_l
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    # Gather target log-probs using one-hot
    # targets_one_hot: (B, T, V)
    targets_oh = onp.zeros((B, T, V), dtype=onp.float32)
    for b in range(B):
        for t in range(T):
            targets_oh[b, t, targets_2d[b, t]] = 1.0

    token_losses = -np.sum(log_probs * targets_oh, axis=-1)  # (B, T)
    masked_loss = np.sum(token_losses * mask_2d)
    n_tokens = np.sum(mask_2d) + 1e-10
    return masked_loss / n_tokens


def compute_per_token_loss_np(params, seq_ids, pad_id):
    """Compute per-token losses for a single sequence. Returns (losses, mask) arrays."""
    if len(seq_ids) <= 1:
        return onp.array([]), onp.array([])

    inp = onp.array([seq_ids[:-1]], dtype=onp.int32)
    tgt = onp.array(seq_ids[1:], dtype=onp.int32)
    mask = (tgt != pad_id).astype(onp.float32)

    logits = forward(params, inp, pad_id)  # (1, T, V)
    logits = onp.array(logits[0])  # (T, V) -- detach from autograd

    # Log-softmax
    max_l = onp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_l
    log_probs = shifted - onp.log(onp.sum(onp.exp(shifted), axis=-1, keepdims=True))

    T = len(tgt)
    losses = onp.zeros(T, dtype=onp.float32)
    for t in range(T):
        if mask[t] > 0:
            losses[t] = -log_probs[t, tgt[t]]

    return losses, mask


def compute_batched_per_token_losses(params, data_encoded, pad_id, batch_size=32):
    """Batched per-token loss computation for efficiency."""
    all_results = []
    for i in range(0, len(data_encoded), batch_size):
        batch = data_encoded[i:i+batch_size]
        max_T = max(len(s) for s in batch)
        if max_T <= 1:
            for s in batch:
                all_results.append((onp.array([]), onp.array([])))
            continue

        B = len(batch)
        idx = onp.full((B, max_T), pad_id, dtype=onp.int32)
        for b, seq in enumerate(batch):
            L = min(len(seq), max_T)
            idx[b, :L] = seq[:L]

        inp = idx[:, :-1]
        tgt = idx[:, 1:]
        mask = (tgt != pad_id).astype(onp.float32)

        logits = onp.array(forward(params, inp, pad_id))  # (B, T, V)

        # Log-softmax
        max_l = onp.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_l
        log_probs = shifted - onp.log(onp.sum(onp.exp(shifted), axis=-1, keepdims=True))

        for b in range(B):
            T = tgt.shape[1]
            losses = onp.zeros(T, dtype=onp.float32)
            for t in range(T):
                if mask[b, t] > 0:
                    losses[t] = -log_probs[b, t, tgt[b, t]]
            all_results.append((losses, mask[b]))

    return all_results


# ── Training ───────────────────────────────────────────────────────────────

def _prepare_batch(seqs, pad_id, max_len=48):
    """Prepare a batch: pad sequences and create input/target/mask arrays."""
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


def train_model(params, data_encoded, pad_id, epochs=30, lr=0.001,
                batch_size=32, verbose=True, clip_grad=1.0):
    """Train using Adam with autograd-computed gradients."""
    # We need to extract the differentiable params (exclude _config)
    cfg = params['_config']
    param_keys = [k for k in sorted(params.keys()) if k != '_config']

    def loss_fn(param_vals, inp, tgt, mask):
        p = dict(zip(param_keys, param_vals))
        p['_config'] = cfg
        return compute_loss(p, inp, tgt, mask, pad_id)

    grad_fn = grad(loss_fn)

    # Adam state
    m_state = [onp.zeros_like(params[k]) for k in param_keys]
    v_state = [onp.zeros_like(params[k]) for k in param_keys]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    n = len(data_encoded)
    rng = onp.random.RandomState(42)

    if verbose:
        total_p = sum(params[k].size for k in param_keys)
        print(f"    Training {total_p:,} params, {epochs} epochs, lr={lr}")

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

            # Compute loss and gradients
            loss_val = float(loss_fn(param_vals, inp, tgt, mask))
            grads = grad_fn(param_vals, inp, tgt, mask)

            # Gradient clipping
            grad_norm = onp.sqrt(sum(float(onp.sum(g**2)) for g in grads))
            if grad_norm > clip_grad:
                scale = clip_grad / grad_norm
                grads = [g * scale for g in grads]

            # Adam update
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
            ppl = math.exp(min(avg_loss, 20))
            print(f"      epoch {epoch:3d}: loss={avg_loss:.4f}  ppl={ppl:.2f}")

    return params


def train_expert(base_params, domain_data_encoded, pad_id,
                 epochs=40, lr=0.001, batch_size=32, verbose=True,
                 clip_grad=1.0):
    """Train expert by fine-tuning a copy of the base model on domain data.
    Returns the delta (expert_params - base_params) for each key.
    """
    import copy

    # Deep copy base params
    expert_params = {}
    for k, v in base_params.items():
        if k == '_config':
            expert_params[k] = v
        else:
            expert_params[k] = v.copy()

    expert_params = train_model(expert_params, domain_data_encoded, pad_id,
                                epochs=epochs, lr=lr, batch_size=batch_size,
                                verbose=verbose, clip_grad=clip_grad)

    # Compute delta
    delta = {}
    for k in expert_params:
        if k == '_config':
            continue
        delta[k] = expert_params[k] - base_params[k]

    return delta


# ── PPL Computation ────────────────────────────────────────────────────────

def compute_full_sequence_ppl(params, data_encoded, pad_id):
    """Full-sequence PPL: average over ALL tokens."""
    token_data = compute_batched_per_token_losses(params, data_encoded, pad_id)
    total_loss = sum(float(onp.sum(l)) for l, m in token_data)
    total_tokens = sum(float(onp.sum(m)) for l, m in token_data)
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def compute_answer_only_ppl(params, data_strings, data_encoded, delimiter, pad_id):
    """Answer-only PPL: average ONLY over answer tokens (after delimiter)."""
    total_loss = 0.0
    total_tokens = 0

    token_data = compute_batched_per_token_losses(params, data_encoded, pad_id)
    for i, (losses, mask) in enumerate(token_data):
        if len(losses) == 0:
            continue

        s = data_strings[i]
        delim_pos = s.rfind(delimiter)
        if delim_pos < 0:
            total_loss += float(onp.sum(losses * mask))
            total_tokens += float(onp.sum(mask))
            continue

        # Answer targets start at index delim_pos in the target array
        for t in range(delim_pos, len(losses)):
            if mask[t] > 0:
                total_loss += float(losses[t])
                total_tokens += 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def compute_prompt_only_ppl(params, data_strings, data_encoded, delimiter, pad_id):
    """Prompt-only PPL: average ONLY over prompt tokens (before delimiter)."""
    total_loss = 0.0
    total_tokens = 0

    token_data = compute_batched_per_token_losses(params, data_encoded, pad_id)
    for i, (losses, mask) in enumerate(token_data):
        if len(losses) == 0:
            continue

        s = data_strings[i]
        delim_pos = s.rfind(delimiter)
        if delim_pos < 0:
            continue

        for t in range(0, min(delim_pos, len(losses))):
            if mask[t] > 0:
                total_loss += float(losses[t])
                total_tokens += 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ── Generation + Accuracy ─────────────────────────────────────────────────

def generate_greedy(params, prompt_ids, max_new, tokenizer, pad_id):
    """Greedy generation using the transformer."""
    ids = list(prompt_ids)
    cfg = params['_config']

    for _ in range(max_new):
        inp = onp.array([ids[-cfg['max_T']:]], dtype=onp.int32)
        logits = forward(params, inp, pad_id)
        next_id = int(onp.argmax(onp.array(logits[0, -1])))
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)

    return tokenizer.decode(ids[len(prompt_ids):])


def evaluate_task_accuracy(params, domain, tokenizer, pad_id, n_eval=200):
    """Evaluate task-specific accuracy for a domain."""
    rng = random.Random(999)
    correct = 0
    total = 0

    for _ in range(n_eval):
        if domain == "arithmetic":
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            prompt = f"{a}+{b}="
            expected = str(a + b)
        elif domain == "reverse":
            chars = "abcdefghijklmnopqrstuvwxyz"
            length = rng.randint(2, 6)
            s = "".join(rng.choice(chars) for _ in range(length))
            prompt = f"{s}>"
            expected = s[::-1]
        elif domain == "repeat":
            chars = "abcdefgh"
            plen = rng.randint(1, 3)
            pat = "".join(rng.choice(chars) for _ in range(plen))
            rep = rng.randint(2, 4)
            prompt = f"{pat}*{rep}="
            expected = pat * rep
        elif domain == "sort":
            chars = "abcdefghijklmnop"
            length = rng.randint(2, 6)
            s = "".join(rng.choice(chars) for _ in range(length))
            prompt = f"{s}>"
            expected = "".join(sorted(s))
        elif domain == "parity":
            length = rng.randint(2, 8)
            bits = "".join(str(rng.randint(0, 1)) for _ in range(length))
            count = bits.count("1")
            parity = "even" if count % 2 == 0 else "odd"
            prompt = f"{bits}>"
            expected = parity
        else:
            raise ValueError(f"Unknown domain: {domain}")

        prompt_ids = [tokenizer.char2idx[c] for c in prompt if c in tokenizer.char2idx]
        generated = generate_greedy(params, prompt_ids, max_new=20, tokenizer=tokenizer,
                                    pad_id=tokenizer.pad_id)
        if generated.strip() == expected:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment(seed=42, d=64, H=4, L=4, max_T=48,
                   n_train=2000, n_eval_ppl=500, n_eval_task=200,
                   base_epochs=30, expert_epochs=40,
                   base_lr=0.001, expert_lr=0.001):
    """Run the answer-conditioned scoring experiment."""
    onp.random.seed(seed)
    random.seed(seed)

    tokenizer = CharTokenizer()
    V = tokenizer.vocab_size
    domains = list(DOMAIN_GENERATORS.keys())

    print(f"\n{'='*60}")
    print(f"  Answer-Conditioned Scoring | seed={seed}")
    print(f"  d={d}, H={H}, L={L}, V={V}")
    print(f"  numpy + autograd, CPU-only")
    print(f"{'='*60}")

    # 1. Generate data
    print("\n[1] Generating synthetic data...")
    all_train_enc = []
    domain_train_str = {}; domain_train_enc = {}
    domain_eval_str = {}; domain_eval_enc = {}

    for domain in domains:
        gen = DOMAIN_GENERATORS[domain]
        train_str = gen(n_train, random.Random(seed + hash(domain) % 1000))
        eval_str = gen(n_eval_ppl, random.Random(seed + 7777 + hash(domain) % 1000))
        train_enc = [tokenizer.encode(s) for s in train_str]
        eval_enc = [tokenizer.encode(s) for s in eval_str]
        domain_train_str[domain] = train_str
        domain_train_enc[domain] = train_enc
        domain_eval_str[domain] = eval_str
        domain_eval_enc[domain] = eval_enc
        all_train_enc.extend(train_enc)
        print(f"  {domain}: {len(train_str)} train, {len(eval_str)} eval")

    # 2. Train base model
    print("\n[2] Training base model on all domains...")
    params = init_model(V, d, H, L, max_T, seed)
    total_p = sum(params[k].size for k in params if k != '_config')
    print(f"  Base model params: {total_p:,}")

    t0 = time.time()
    params = train_model(params, all_train_enc, tokenizer.pad_id,
                         epochs=base_epochs, lr=base_lr, batch_size=32, verbose=True)
    print(f"  Base training: {time.time()-t0:.1f}s")

    # Save base params (deep copy)
    base_params = {k: (v.copy() if k != '_config' else v) for k, v in params.items()}

    # 3. Evaluate base model
    print("\n[3] Evaluating base model...")
    base_full_ppls = {}; base_answer_ppls = {}; base_prompt_ppls = {}; base_accs = {}

    for domain in domains:
        delim = DOMAIN_DELIMITERS[domain]
        full_ppl = compute_full_sequence_ppl(params, domain_eval_enc[domain], tokenizer.pad_id)
        ans_ppl = compute_answer_only_ppl(params, domain_eval_str[domain],
                                           domain_eval_enc[domain], delim, tokenizer.pad_id)
        prompt_ppl = compute_prompt_only_ppl(params, domain_eval_str[domain],
                                              domain_eval_enc[domain], delim, tokenizer.pad_id)
        acc = evaluate_task_accuracy(params, domain, tokenizer, tokenizer.pad_id, n_eval_task)

        base_full_ppls[domain] = float(full_ppl)
        base_answer_ppls[domain] = float(ans_ppl)
        base_prompt_ppls[domain] = float(prompt_ppl)
        base_accs[domain] = float(acc)
        print(f"  {domain:12s}: FullPPL={full_ppl:8.2f}  AnsPPL={ans_ppl:8.2f}  "
              f"PromptPPL={prompt_ppl:8.2f}  Acc={acc:.3f}")

    # 4. Train domain-specific experts
    print("\n[4] Training domain-specific experts...")
    expert_deltas = {}

    for domain in domains:
        print(f"  Training expert: {domain}...")
        t0 = time.time()
        delta = train_expert(base_params, domain_train_enc[domain], tokenizer.pad_id,
                             epochs=expert_epochs, lr=expert_lr, batch_size=32,
                             verbose=True)
        expert_deltas[domain] = delta
        print(f"    ({time.time()-t0:.1f}s)")

    # 5. Evaluate each expert
    print("\n[5] Evaluating experts...")
    expert_full_ppls = {}; expert_answer_ppls = {}; expert_prompt_ppls = {}; expert_accs = {}

    for domain in domains:
        delim = DOMAIN_DELIMITERS[domain]

        # Apply delta to base
        expert_params = {k: (base_params[k].copy() if k != '_config' else base_params[k])
                         for k in base_params}
        for k, d in expert_deltas[domain].items():
            expert_params[k] = expert_params[k] + d

        full_ppl = compute_full_sequence_ppl(expert_params, domain_eval_enc[domain], tokenizer.pad_id)
        ans_ppl = compute_answer_only_ppl(expert_params, domain_eval_str[domain],
                                           domain_eval_enc[domain], delim, tokenizer.pad_id)
        prompt_ppl = compute_prompt_only_ppl(expert_params, domain_eval_str[domain],
                                              domain_eval_enc[domain], delim, tokenizer.pad_id)
        acc = evaluate_task_accuracy(expert_params, domain, tokenizer, tokenizer.pad_id, n_eval_task)

        expert_full_ppls[domain] = float(full_ppl)
        expert_answer_ppls[domain] = float(ans_ppl)
        expert_prompt_ppls[domain] = float(prompt_ppl)
        expert_accs[domain] = float(acc)

        print(f"  {domain:12s}: FullPPL={full_ppl:8.2f} ({(1-full_ppl/base_full_ppls[domain])*100:+.1f}%)  "
              f"AnsPPL={ans_ppl:8.2f} ({(1-ans_ppl/base_answer_ppls[domain])*100:+.1f}%)  "
              f"Acc={acc:.3f} ({(acc-base_accs[domain])*100:+.1f}pp)")

    # 6. Correlation analysis
    print(f"\n{'='*60}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*60}")

    full_imps = []; ans_imps = []; prompt_imps = []; acc_imps = []
    for domain in domains:
        fi = (base_full_ppls[domain] - expert_full_ppls[domain]) / base_full_ppls[domain]
        ai = (base_answer_ppls[domain] - expert_answer_ppls[domain]) / base_answer_ppls[domain]
        pi = (base_prompt_ppls[domain] - expert_prompt_ppls[domain]) / base_prompt_ppls[domain]
        acci = expert_accs[domain] - base_accs[domain]
        full_imps.append(fi); ans_imps.append(ai); prompt_imps.append(pi); acc_imps.append(acci)
        print(f"  {domain:12s}: FullPPL_imp={fi:+.4f}  AnsPPL_imp={ai:+.4f}  "
              f"PromptPPL_imp={pi:+.4f}  Acc_imp={acci:+.4f}")

    full_arr = onp.array(full_imps)
    ans_arr = onp.array(ans_imps)
    prompt_arr = onp.array(prompt_imps)
    acc_arr = onp.array(acc_imps)

    def safe_pearson(x, y):
        if onp.std(x) < 1e-10 or onp.std(y) < 1e-10:
            return 0.0
        return float(onp.corrcoef(x, y)[0, 1])

    r_full = safe_pearson(full_arr, acc_arr)
    r_answer = safe_pearson(ans_arr, acc_arr)
    r_prompt = safe_pearson(prompt_arr, acc_arr)

    try:
        from scipy.stats import spearmanr
        rho_full, p_full = spearmanr(full_arr, acc_arr)
        rho_answer, p_answer = spearmanr(ans_arr, acc_arr)
        if onp.isnan(rho_full): rho_full = 0.0; p_full = 1.0
        if onp.isnan(rho_answer): rho_answer = 0.0; p_answer = 1.0
    except Exception:
        rho_full = rho_answer = 0.0; p_full = p_answer = 1.0

    print(f"\n  Pearson r(Full_PPL_imp, Acc_imp)   = {r_full:.4f}")
    print(f"  Pearson r(Answer_PPL_imp, Acc_imp) = {r_answer:.4f}")
    print(f"  Pearson r(Prompt_PPL_imp, Acc_imp) = {r_prompt:.4f}")
    print(f"  Spearman rho(Full, Acc)  = {rho_full:.4f}  (p={p_full:.4f})")
    print(f"  Spearman rho(Answer, Acc) = {rho_answer:.4f}  (p={p_answer:.4f})")

    full_ranking = sorted(domains, key=lambda d: full_imps[domains.index(d)], reverse=True)
    answer_ranking = sorted(domains, key=lambda d: ans_imps[domains.index(d)], reverse=True)
    acc_ranking = sorted(domains, key=lambda d: acc_imps[domains.index(d)], reverse=True)
    rankings_differ = full_ranking != answer_ranking

    print(f"\n  Full PPL ranking:   {full_ranking}")
    print(f"  Answer PPL ranking: {answer_ranking}")
    print(f"  Accuracy ranking:   {acc_ranking}")
    print(f"  Rankings differ:    {rankings_differ}")

    def rank_agreement(ranking, acc_ranking):
        n = len(ranking); agree = 0; total = 0
        for i in range(n):
            for j in range(i+1, n):
                ri = ranking.index(acc_ranking[i])
                rj = ranking.index(acc_ranking[j])
                if ri < rj: agree += 1
                total += 1
        return agree / total if total > 0 else 0.0

    full_agree = rank_agreement(full_ranking, acc_ranking)
    answer_agree = rank_agreement(answer_ranking, acc_ranking)
    print(f"  Full PPL rank agreement with Acc:   {full_agree:.2%}")
    print(f"  Answer PPL rank agreement with Acc:  {answer_agree:.2%}")

    # 7. Kill criteria
    print(f"\n{'='*60}")
    print("  KILL CRITERIA ASSESSMENT")
    print(f"{'='*60}")

    k1_pass = r_answer >= 0.5
    k2_pass = rankings_differ
    overall = k1_pass and k2_pass

    print(f"  K1: Answer-only Pearson r >= 0.5?  r={r_answer:.4f}  {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2: Rankings differ from full-seq?  {'PASS' if k2_pass else 'KILL'}")
    print(f"  Overall: {'SURVIVES' if overall else 'KILLED'}")
    print(f"  (Reference: full-seq r = {r_full:.4f}, predecessor full-seq r = 0.084)")
    print(f"{'='*60}\n")

    results = {
        "seed": seed,
        "config": {
            "d": d, "H": H, "L": L, "V": V,
            "n_train": n_train, "n_eval_ppl": n_eval_ppl,
            "base_epochs": base_epochs, "expert_epochs": expert_epochs,
            "pure_numpy_autograd": True, "model_type": "Transformer",
        },
        "base_full_ppls": base_full_ppls,
        "base_answer_ppls": base_answer_ppls,
        "base_prompt_ppls": base_prompt_ppls,
        "base_accs": base_accs,
        "expert_full_ppls": expert_full_ppls,
        "expert_answer_ppls": expert_answer_ppls,
        "expert_prompt_ppls": expert_prompt_ppls,
        "expert_accs": expert_accs,
        "full_ppl_improvements": dict(zip(domains, [float(x) for x in full_imps])),
        "answer_ppl_improvements": dict(zip(domains, [float(x) for x in ans_imps])),
        "prompt_ppl_improvements": dict(zip(domains, [float(x) for x in prompt_imps])),
        "acc_improvements": dict(zip(domains, [float(x) for x in acc_imps])),
        "pearson_r_full": float(r_full),
        "pearson_r_answer": float(r_answer),
        "pearson_r_prompt": float(r_prompt),
        "spearman_rho_full": float(rho_full),
        "spearman_rho_answer": float(rho_answer),
        "full_ppl_ranking": full_ranking,
        "answer_ppl_ranking": answer_ranking,
        "acc_ranking": acc_ranking,
        "rankings_differ": rankings_differ,
        "full_rank_agreement": full_agree,
        "answer_rank_agreement": answer_agree,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "overall": "SURVIVES" if overall else "KILLED",
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Answer-Conditioned Scoring")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--expert-epochs", type=int, default=40)
    parser.add_argument("--n-train", type=int, default=2000)
    args = parser.parse_args()

    results_dir = Path(__file__).parent
    all_results = []
    seeds = [42, 123, 7][:args.seeds]

    for seed in seeds:
        result = run_experiment(
            seed=seed, d=args.d, H=args.heads, L=args.layers,
            base_epochs=args.base_epochs, expert_epochs=args.expert_epochs,
            n_train=args.n_train,
        )
        all_results.append(result)
        with open(results_dir / f"results_seed_{seed}.json", "w") as f:
            json.dump(result, f, indent=2, cls=_NumpyEncoder)

    r_fulls = [r["pearson_r_full"] for r in all_results]
    r_answers = [r["pearson_r_answer"] for r in all_results]
    r_prompts = [r["pearson_r_prompt"] for r in all_results]
    k1_passes = [r["k1_pass"] for r in all_results]
    k2_passes = [r["k2_pass"] for r in all_results]

    aggregate = {
        "seeds": seeds,
        "pearson_r_full": {"values": r_fulls, "mean": float(onp.mean(r_fulls)), "std": float(onp.std(r_fulls))},
        "pearson_r_answer": {"values": r_answers, "mean": float(onp.mean(r_answers)), "std": float(onp.std(r_answers))},
        "pearson_r_prompt": {"values": r_prompts, "mean": float(onp.mean(r_prompts)), "std": float(onp.std(r_prompts))},
        "k1_pass_rate": sum(k1_passes) / len(k1_passes),
        "k2_pass_rate": sum(k2_passes) / len(k2_passes),
        "overall": "SURVIVES" if all(r["overall"] == "SURVIVES" for r in all_results) else "KILLED",
        "per_seed": all_results,
    }

    print(f"\n{'#'*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'#'*60}")
    print(f"  Full-seq PPL vs Acc:   r = {aggregate['pearson_r_full']['mean']:.4f} +/- {aggregate['pearson_r_full']['std']:.4f}")
    print(f"  Answer PPL vs Acc:     r = {aggregate['pearson_r_answer']['mean']:.4f} +/- {aggregate['pearson_r_answer']['std']:.4f}")
    print(f"  Prompt PPL vs Acc:     r = {aggregate['pearson_r_prompt']['mean']:.4f} +/- {aggregate['pearson_r_prompt']['std']:.4f}")
    print(f"  K1 pass rate: {aggregate['k1_pass_rate']:.0%}")
    print(f"  K2 pass rate: {aggregate['k2_pass_rate']:.0%}")
    print(f"  Overall: {aggregate['overall']}")
    print(f"{'#'*60}\n")

    with open(results_dir / "results_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2, cls=_NumpyEncoder)

    print(f"Results saved to {results_dir}/")
    return aggregate


if __name__ == "__main__":
    main()
