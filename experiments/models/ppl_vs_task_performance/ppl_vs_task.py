#!/usr/bin/env python3
"""PPL vs Task Performance: Does perplexity improvement predict downstream accuracy?

This experiment trains domain-specific LoRA experts on a toy character-level
transformer and measures both:
  (a) Held-out perplexity (what SOLE evolution optimizes)
  (b) Task-specific accuracy (what users actually care about)

If these don't correlate (Pearson r < 0.5), PPL is a misleading proxy and
the entire shadow-scoring evolution mechanism needs a different signal.

Five synthetic domains, each with a generative task (PPL) and a structured
task (accuracy):
  1. arithmetic:  "2+3=5" sequences / solve addition correctly
  2. reverse:     "abc>cba" sequences / reverse strings correctly
  3. repeat:      "ab*3=ababab" / repeat patterns correctly
  4. sort:        "bca>abc" / sort characters correctly
  5. parity:      "1011>even" / count 1-bits and determine parity

Usage:
    uv run python -m micro.models.ppl_vs_task_performance.ppl_vs_task
    uv run python -m micro.models.ppl_vs_task_performance.ppl_vs_task --seeds 3
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Lazy torch import ──────────────────────────────────────────────────────
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ── Synthetic Data Generation ──────────────────────────────────────────────

def _make_arithmetic_data(n: int, rng: random.Random) -> list[str]:
    """Generate 'A+B=C' strings where A,B in [0,99]."""
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        data.append(f"{a}+{b}={a+b}")
    return data

def _make_reverse_data(n: int, rng: random.Random) -> list[str]:
    """Generate 'abc>cba' strings."""
    chars = "abcdefghijklmnopqrstuvwxyz"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(chars) for _ in range(length))
        data.append(f"{s}>{s[::-1]}")
    return data

def _make_repeat_data(n: int, rng: random.Random) -> list[str]:
    """Generate 'ab*3=ababab' strings."""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        plen = rng.randint(1, 3)
        pat = "".join(rng.choice(chars) for _ in range(plen))
        rep = rng.randint(2, 4)
        data.append(f"{pat}*{rep}={pat * rep}")
    return data

def _make_sort_data(n: int, rng: random.Random) -> list[str]:
    """Generate 'bca>abc' strings."""
    chars = "abcdefghijklmnop"
    data = []
    for _ in range(n):
        length = rng.randint(2, 6)
        s = "".join(rng.choice(chars) for _ in range(length))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data

def _make_parity_data(n: int, rng: random.Random) -> list[str]:
    """Generate '1011>even' or '1011>odd' strings."""
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


# ── Tokenizer (character-level) ───────────────────────────────────────────

class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self):
        # All chars we might see
        chars = sorted(set(
            "0123456789abcdefghijklmnopqrstuvwxyz>+=*"
        ))
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        specials = [self.pad_token, self.eos_token]
        self.vocab = specials + chars
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.char2idx[self.pad_token]
        self.eos_id = self.char2idx[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, s: str) -> list[int]:
        return [self.char2idx[c] for c in s if c in self.char2idx] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            c = self.idx2char.get(i, "")
            if c == self.eos_token:
                break
            if c == self.pad_token:
                continue
            out.append(c)
        return "".join(out)


# ── Minimal Transformer (PyTorch) ─────────────────────────────────────────

def _build_model(vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 64, device: str = "cpu"):
    """Build a minimal causal transformer."""
    torch = _get_torch()
    import torch.nn as tnn

    class RMSNorm(tnn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.weight = tnn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            ms = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(ms + self.eps) * self.weight).to(x.dtype)

    class CausalSelfAttention(tnn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = tnn.Linear(d_model, 3 * d_model, bias=False)
            self.wo = tnn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))

    class MLP(tnn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.fc1 = tnn.Linear(d_model, 4 * d_model, bias=False)
            self.fc2 = tnn.Linear(4 * d_model, d_model, bias=False)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class Block(tnn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.ln1 = RMSNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads)
            self.ln2 = RMSNorm(d_model)
            self.mlp = MLP(d_model)
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class MiniGPT(tnn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = tnn.Embedding(vocab_size, d_model)
            self.pos_emb = tnn.Embedding(max_seq_len, d_model)
            self.blocks = tnn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
            self.ln_f = RMSNorm(d_model)
            self.head = tnn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    model = MiniGPT().to(device)
    return model


# ── LoRA Layer ─────────────────────────────────────────────────────────────

class LoRALinear:
    """Minimal LoRA wrapper that adds a low-rank delta via forward hook.

    The hook approach preserves the computational graph so gradients flow
    back to A and B during training, while the base weights stay frozen.
    """

    def __init__(self, linear, rank: int = 8):
        torch = _get_torch()
        device = linear.weight.device
        d_out, d_in = linear.weight.shape
        self.linear = linear
        self.rank = rank
        self.alpha = 1.0
        # A: (r, d_in), B: (d_out, r)
        self.A = torch.nn.Parameter(torch.randn(rank, d_in, device=device) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(d_out, rank, device=device))
        # Register hook: output += (alpha/r) * x @ A^T @ B^T
        self._hook = linear.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        # input is tuple; input[0] is (..., d_in)
        x = input[0]
        # LoRA path: x @ A^T -> (..., r), then @ B^T -> (..., d_out)
        lora_out = (self.alpha / self.rank) * (x @ self.A.T @ self.B.T)
        return output + lora_out

    def get_delta(self):
        return (self.alpha / self.rank) * (self.B @ self.A)

    def parameters(self):
        return [self.A, self.B]

    def remove(self):
        self._hook.remove()


def apply_lora_to_ffn(model, rank: int = 8):
    """Apply LoRA hooks to all FFN layers (fc1, fc2). Returns LoRA layer list."""
    lora_layers = []
    for block in model.blocks:
        lora_fc1 = LoRALinear(block.mlp.fc1, rank)
        lora_fc2 = LoRALinear(block.mlp.fc2, rank)
        lora_layers.append((block.mlp, "fc1", lora_fc1))
        lora_layers.append((block.mlp, "fc2", lora_fc2))
    return lora_layers


def remove_lora_hooks(lora_layers):
    """Remove all LoRA forward hooks."""
    for _, _, lora in lora_layers:
        lora.remove()


def get_lora_params(lora_layers):
    """Get all LoRA parameters for optimizer."""
    params = []
    for _, _, lora in lora_layers:
        params.extend(lora.parameters())
    return params


# ── Training Utilities ─────────────────────────────────────────────────────

def _prepare_batch(strings: list[str], tokenizer: CharTokenizer, max_len: int,
                   device: str):
    """Tokenize and pad a batch of strings."""
    torch = _get_torch()
    encoded = [tokenizer.encode(s) for s in strings]
    # Truncate
    encoded = [e[:max_len] for e in encoded]
    # Pad
    max_actual = max(len(e) for e in encoded)
    padded = [e + [tokenizer.pad_id] * (max_actual - len(e)) for e in encoded]
    return torch.tensor(padded, dtype=torch.long, device=device)


def compute_ppl(model, data: list[str], tokenizer: CharTokenizer,
                max_len: int, device: str, batch_size: int = 64,
                lora_layers=None) -> float:
    """Compute perplexity on a dataset."""
    torch = _get_torch()
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            ids = _prepare_batch(batch, tokenizer, max_len, device)
            inp = ids[:, :-1]
            tgt = ids[:, 1:]

            logits = model(inp)

            # Apply LoRA deltas if provided (for non-merged evaluation)
            if lora_layers is not None:
                # We need a forward pass that includes LoRA
                # For simplicity, merge temporarily, forward, unmerge
                pass  # We'll merge before eval instead

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=tokenizer.pad_id,
                reduction="sum"
            )
            mask = (tgt != tokenizer.pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += mask

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def train_model(model, data: list[str], tokenizer: CharTokenizer,
                max_len: int, device: str, epochs: int = 20,
                lr: float = 1e-3, batch_size: int = 64,
                lora_params=None, lora_layers=None):
    """Train model (or LoRA params) on data."""
    torch = _get_torch()
    if lora_params is not None:
        # Freeze base, train only LoRA
        for p in model.parameters():
            p.requires_grad = False
        for p in lora_params:
            p.requires_grad = True
        optimizer = torch.optim.Adam(lora_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    rng = random.Random(42)

    for epoch in range(epochs):
        indices = list(range(len(data)))
        rng.shuffle(indices)

        for i in range(0, len(data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [data[j] for j in batch_idx]
            ids = _prepare_batch(batch, tokenizer, max_len, device)
            inp = ids[:, :-1]
            tgt = ids[:, 1:]

            logits = model(inp)

            # If training LoRA, add deltas to logits via hook
            if lora_layers is not None:
                # Apply LoRA as additive delta to FFN outputs
                # For micro simplicity: just merge into weights for forward,
                # but keep LoRA params as the trainable ones
                pass  # Using the merged approach below

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_lora_expert(base_model, domain_data: list[str], tokenizer: CharTokenizer,
                      max_len: int, device: str, rank: int = 8,
                      epochs: int = 30, lr: float = 3e-3, batch_size: int = 64):
    """Train a LoRA expert on domain data using forward hooks.

    Hooks add the LoRA delta to each Linear output during forward pass,
    preserving the computational graph for gradient flow to A and B.
    Returns the trained LoRA deltas (as state dict entries).
    """
    torch = _get_torch()

    # Attach LoRA hooks to FFN layers
    lora_layers = apply_lora_to_ffn(base_model, rank)
    lora_params = get_lora_params(lora_layers)

    optimizer = torch.optim.Adam(lora_params, lr=lr)

    # Freeze base, train only LoRA
    for p in base_model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    rng = random.Random(42)
    base_model.train()

    for epoch in range(epochs):
        indices = list(range(len(domain_data)))
        rng.shuffle(indices)

        for i in range(0, len(domain_data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [domain_data[j] for j in batch_idx]
            ids = _prepare_batch(batch, tokenizer, max_len, device)
            inp = ids[:, :-1]
            tgt = ids[:, 1:]

            # Forward pass -- hooks add LoRA delta automatically
            logits = base_model(inp)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract final deltas and remove hooks
    deltas = {}
    with torch.no_grad():
        for parent, attr_name, lora in lora_layers:
            key = _find_key_for_linear(base_model, parent, attr_name)
            deltas[key] = lora.get_delta().clone()

    remove_lora_hooks(lora_layers)

    # Unfreeze base params for future use
    for p in base_model.parameters():
        p.requires_grad = True

    return deltas


def _get_weight_key(model, linear_module):
    """Find the state_dict key for a linear module's weight."""
    for name, module in model.named_modules():
        if module is linear_module:
            return f"{name}.weight"
    raise ValueError("Module not found in model")


def _find_key_for_linear(model, parent_module, attr_name):
    """Find state_dict key for parent.attr_name.weight."""
    for name, module in model.named_modules():
        if module is parent_module:
            return f"{name}.{attr_name}.weight"
    raise ValueError("Parent module not found")


def apply_expert_deltas(model, deltas: dict, base_state: dict):
    """Apply expert deltas to model weights (base + delta)."""
    torch = _get_torch()
    state = model.state_dict()
    with torch.no_grad():
        for key, delta in deltas.items():
            state[key] = base_state[key] + delta
    model.load_state_dict(state)


def restore_base(model, base_state: dict):
    """Restore model to base weights."""
    model.load_state_dict(base_state)


# ── Task-Specific Accuracy Evaluation ──────────────────────────────────────

def generate_greedy(model, prompt_ids: list[int], max_new: int,
                    tokenizer: CharTokenizer, device: str) -> str:
    """Greedy generation from a prompt."""
    torch = _get_torch()
    model.eval()
    ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids[-48:]], dtype=torch.long, device=device)
            logits = model(inp)
            next_id = logits[0, -1].argmax().item()
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)

    return tokenizer.decode(ids[len(prompt_ids):])


def evaluate_task_accuracy(model, domain: str, tokenizer: CharTokenizer,
                           device: str, n_eval: int = 200) -> float:
    """Evaluate task-specific accuracy for a domain.

    For each domain, generate a prompt (the input side) and check if the
    model's greedy completion matches the ground truth answer.
    """
    rng = random.Random(999)  # Fixed eval seed
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
        generated = generate_greedy(model, prompt_ids, max_new=20, tokenizer=tokenizer,
                                    device=device)
        # Strip trailing EOS and whitespace
        generated = generated.strip()

        if generated == expected:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment(seed: int = 42, device: str = "cpu",
                   d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                   lora_rank: int = 8, max_seq_len: int = 48,
                   n_train: int = 2000, n_eval_ppl: int = 500, n_eval_task: int = 200,
                   base_epochs: int = 30, expert_epochs: int = 40,
                   base_lr: float = 1e-3, expert_lr: float = 3e-3) -> dict:
    """Run the full PPL vs task accuracy experiment."""
    torch = _get_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = CharTokenizer()
    domains = list(DOMAIN_GENERATORS.keys())
    print(f"\n{'='*60}")
    print(f"  PPL vs Task Performance | seed={seed}")
    print(f"  d={d_model}, layers={n_layers}, rank={lora_rank}")
    print(f"{'='*60}")

    # 1. Generate all domain data
    print("\n[1] Generating synthetic data...")
    all_train_data = []
    domain_train = {}
    domain_eval = {}

    for domain in domains:
        gen = DOMAIN_GENERATORS[domain]
        train = gen(n_train, random.Random(seed + hash(domain) % 1000))
        eval_data = gen(n_eval_ppl, random.Random(seed + 7777 + hash(domain) % 1000))
        domain_train[domain] = train
        domain_eval[domain] = eval_data
        all_train_data.extend(train)
        print(f"  {domain}: {len(train)} train, {len(eval_data)} eval")

    # 2. Train base model on ALL domains
    print("\n[2] Training base model on all domains...")
    model = _build_model(tokenizer.vocab_size, d_model, n_heads, n_layers,
                         max_seq_len, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Base model params: {param_count:,}")

    t0 = time.time()
    train_model(model, all_train_data, tokenizer, max_seq_len, device,
                epochs=base_epochs, lr=base_lr)
    print(f"  Base training: {time.time()-t0:.1f}s")

    # Save base state
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 3. Evaluate base model on all domains (PPL + task accuracy)
    print("\n[3] Evaluating base model...")
    base_ppls = {}
    base_accs = {}
    for domain in domains:
        ppl = compute_ppl(model, domain_eval[domain], tokenizer, max_seq_len, device)
        acc = evaluate_task_accuracy(model, domain, tokenizer, device, n_eval_task)
        base_ppls[domain] = ppl
        base_accs[domain] = acc
        print(f"  {domain:12s}: PPL={ppl:8.2f}  Acc={acc:.3f}")

    # 4. Train domain-specific LoRA experts
    print("\n[4] Training domain-specific LoRA experts...")
    expert_deltas = {}
    for domain in domains:
        print(f"  Training expert: {domain}...")
        t0 = time.time()
        deltas = train_lora_expert(
            model, domain_train[domain], tokenizer, max_seq_len, device,
            rank=lora_rank, epochs=expert_epochs, lr=expert_lr
        )
        expert_deltas[domain] = deltas
        n_params = sum(d.numel() for d in deltas.values())
        print(f"    LoRA params: {n_params:,}  ({time.time()-t0:.1f}s)")

    # 5. Evaluate each expert (PPL + task accuracy)
    print("\n[5] Evaluating experts...")
    expert_ppls = {}
    expert_accs = {}
    for domain in domains:
        # Apply expert deltas
        apply_expert_deltas(model, expert_deltas[domain], base_state)

        ppl = compute_ppl(model, domain_eval[domain], tokenizer, max_seq_len, device)
        acc = evaluate_task_accuracy(model, domain, tokenizer, device, n_eval_task)
        expert_ppls[domain] = ppl
        expert_accs[domain] = acc
        print(f"  {domain:12s}: PPL={ppl:8.2f} (base: {base_ppls[domain]:8.2f}, "
              f"improv: {(1 - ppl/base_ppls[domain])*100:+.1f}%)  "
              f"Acc={acc:.3f} (base: {base_accs[domain]:.3f}, "
              f"improv: {(acc - base_accs[domain])*100:+.1f}pp)")

        # Restore base
        restore_base(model, base_state)

    # 6. Compute PPL improvement vs accuracy improvement correlation
    print("\n[6] Correlation analysis...")
    ppl_improvements = []
    acc_improvements = []
    for domain in domains:
        # PPL improvement: positive = better (lower PPL)
        ppl_imp = (base_ppls[domain] - expert_ppls[domain]) / base_ppls[domain]
        # Accuracy improvement: positive = better
        acc_imp = expert_accs[domain] - base_accs[domain]
        ppl_improvements.append(ppl_imp)
        acc_improvements.append(acc_imp)
        print(f"  {domain:12s}: PPL_imp={ppl_imp:.4f}  Acc_imp={acc_imp:.4f}")

    # Pearson correlation
    ppl_arr = np.array(ppl_improvements)
    acc_arr = np.array(acc_improvements)

    if np.std(ppl_arr) < 1e-10 or np.std(acc_arr) < 1e-10:
        pearson_r = 0.0
        print("  WARNING: Near-zero variance, correlation undefined")
    else:
        pearson_r = float(np.corrcoef(ppl_arr, acc_arr)[0, 1])

    print(f"\n  Pearson r(PPL_improvement, Acc_improvement) = {pearson_r:.4f}")

    # Also check: does best PPL expert have best accuracy?
    best_ppl_domain = min(expert_ppls, key=expert_ppls.get)
    best_acc_domain = max(expert_accs, key=expert_accs.get)
    best_ppl_imp_domain = max(domains, key=lambda d: ppl_improvements[domains.index(d)])
    best_acc_imp_domain = max(domains, key=lambda d: acc_improvements[domains.index(d)])

    print(f"  Best PPL improvement:  {best_ppl_imp_domain}")
    print(f"  Best Acc improvement:  {best_acc_imp_domain}")
    print(f"  Best absolute PPL:     {best_ppl_domain} ({expert_ppls[best_ppl_domain]:.2f})")
    print(f"  Best absolute Acc:     {best_acc_domain} ({expert_accs[best_acc_domain]:.3f})")

    # 7. Kill criteria assessment
    print(f"\n{'='*60}")
    print("  KILL CRITERIA ASSESSMENT")
    print(f"{'='*60}")

    k1_pass = pearson_r >= 0.5
    k2_pass = best_ppl_imp_domain == best_acc_imp_domain
    overall_pass = k1_pass  # K1 is the primary criterion

    print(f"  K1: Pearson r >= 0.5?  r={pearson_r:.4f}  {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2: Best PPL = Best Acc?  PPL:{best_ppl_imp_domain} Acc:{best_acc_imp_domain}  "
          f"{'PASS' if k2_pass else 'FAIL'}")
    print(f"  Overall: {'SURVIVES' if overall_pass else 'KILLED'}")
    print(f"{'='*60}\n")

    results = {
        "seed": seed,
        "config": {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "lora_rank": lora_rank, "n_train": n_train,
            "base_epochs": base_epochs, "expert_epochs": expert_epochs,
        },
        "base_ppls": base_ppls,
        "base_accs": base_accs,
        "expert_ppls": expert_ppls,
        "expert_accs": expert_accs,
        "ppl_improvements": dict(zip(domains, [float(x) for x in ppl_improvements])),
        "acc_improvements": dict(zip(domains, [float(x) for x in acc_improvements])),
        "pearson_r": float(pearson_r),
        "best_ppl_imp_domain": best_ppl_imp_domain,
        "best_acc_imp_domain": best_acc_imp_domain,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "overall": "SURVIVES" if overall_pass else "KILLED",
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="PPL vs Task Performance Correlation")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--expert-epochs", type=int, default=40)
    args = parser.parse_args()

    results_dir = Path(__file__).parent
    all_results = []
    seeds = [42, 123, 7][:args.seeds]

    for seed in seeds:
        result = run_experiment(
            seed=seed, device=args.device,
            d_model=args.d_model, n_layers=args.n_layers,
            lora_rank=args.lora_rank,
            base_epochs=args.base_epochs,
            expert_epochs=args.expert_epochs,
        )
        all_results.append(result)

        # Save per-seed
        with open(results_dir / f"results_seed_{seed}.json", "w") as f:
            json.dump(result, f, indent=2)

    # Aggregate across seeds
    pearson_rs = [r["pearson_r"] for r in all_results]
    k1_passes = [r["k1_pass"] for r in all_results]
    k2_passes = [r["k2_pass"] for r in all_results]

    aggregate = {
        "seeds": seeds,
        "pearson_rs": pearson_rs,
        "mean_pearson_r": float(np.mean(pearson_rs)),
        "std_pearson_r": float(np.std(pearson_rs)),
        "k1_pass_rate": sum(k1_passes) / len(k1_passes),
        "k2_pass_rate": sum(k2_passes) / len(k2_passes),
        "overall": "SURVIVES" if all(k1_passes) else "KILLED",
        "per_seed": all_results,
    }

    print(f"\n{'#'*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'#'*60}")
    print(f"  Mean Pearson r: {aggregate['mean_pearson_r']:.4f} +/- {aggregate['std_pearson_r']:.4f}")
    print(f"  K1 pass rate:   {aggregate['k1_pass_rate']:.0%}")
    print(f"  K2 pass rate:   {aggregate['k2_pass_rate']:.0%}")
    print(f"  Overall: {aggregate['overall']}")
    print(f"{'#'*60}\n")

    with open(results_dir / "results_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Results saved to {results_dir}/")
    return aggregate


if __name__ == "__main__":
    main()
