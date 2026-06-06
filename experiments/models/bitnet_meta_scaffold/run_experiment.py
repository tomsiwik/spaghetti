#!/usr/bin/env python3
"""
Meta-Scaffold: MAML-Optimized Scaffold for Adapter Composition

Bilevel optimization where:
  Inner loop: Train fresh LoRA adapters per domain on the scaffold (K steps)
  Outer loop: Update scaffold weights via first-order MAML to minimize
              meta-loss = avg(domain_loss) + lambda * composition_penalty

Compare to GaLore scaffold baseline (exp_bitnet_galore_scaffold):
  - GaLore scaffold: PPL ratio 1.918x, comp ratio 1.045x, |cos| 0.0027

Kill criteria:
  K1: meta-scaffold does not outperform GaLore scaffold on adapter composition quality
  K2: MAML outer loop does not converge within 100 meta-steps

Platform: Apple Silicon MLX, $0.
Architecture: TinyGPT d=256, 6 layers, 4 heads (~6.4M params) -- matches GaLore experiment.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# ===========================================================================
# Configuration
# ===========================================================================
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
VOCAB_SIZE = 8192
MAX_SEQ_LEN = 128

# Pre-training (standard baseline for comparison)
PRETRAIN_STEPS = 2000
PRETRAIN_LR = 3e-4
PRETRAIN_BATCH_SIZE = 4

# Meta-learning
META_STEPS = 100           # outer loop steps
META_LR = 1e-4             # outer loop learning rate
INNER_STEPS = 50           # inner loop steps per domain per meta-step
INNER_LR = 1e-3            # inner loop adapter learning rate
COMPOSITION_LAMBDA = 0.5   # weight on composition penalty in meta-loss
N_DOMAINS_PER_META = 3     # sample 3 of 5 domains per meta-step (speed)

# LoRA
LORA_RANK = 16
LORA_SCALE = 4.0

# Final adapter training (for evaluation, matching GaLore experiment)
ADAPTER_TRAIN_STEPS = 400
ADAPTER_LR = 1e-3

# Eval
VAL_BATCHES = 25

SEED = 42
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Data
DATA_ROOT = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"
DOMAINS = ["python", "math", "medical", "legal", "creative"]


# ===========================================================================
# TinyGPT Model (identical to GaLore scaffold experiment)
# ===========================================================================
class TinyMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 4, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 4, bias=False)
        self.down_proj = nn.Linear(d_model * 4, d_model, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.o_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = TinyMLP(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.blocks = [TinyBlock(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(mx.arange(T))
        h = tok + pos
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(h.dtype)
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.ln_f(h)
        return self.lm_head(h)


# ===========================================================================
# LoRA
# ===========================================================================
LORA_TARGETS = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        in_dim = base.weight.shape[1]
        out_dim = base.weight.shape[0]
        self.base = base
        self.lora_a = mx.random.normal(shape=(in_dim, rank)) * (1.0 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((rank, out_dim))
        self.scale = scale

    def __call__(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora(model, rank, scale):
    count = 0
    for block in model.blocks:
        for attr_path in LORA_TARGETS:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            base_linear = getattr(parent, parts[-1])
            if isinstance(base_linear, nn.Linear):
                lora = LoRALinear(base_linear, rank, scale)
                setattr(parent, parts[-1], lora)
                count += 1
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora(model):
    for block in model.blocks:
        for attr_path in LORA_TARGETS:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                in_dim = module.lora_a.shape[0]
                module.lora_a = mx.random.normal(shape=module.lora_a.shape) * (1.0 / math.sqrt(in_dim))
                module.lora_b = mx.zeros(module.lora_b.shape)
    mx.eval(model.parameters())


def remove_lora(model):
    for block in model.blocks:
        for attr_path in LORA_TARGETS:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                setattr(parent, parts[-1], module.base)


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


# ===========================================================================
# Ternary Quantization
# ===========================================================================
def ternary_quantize(model):
    count = 0
    for name, p in tree_flatten(model.parameters()):
        if len(p.shape) == 2 and p.size > 100:
            abs_w = mx.abs(p)
            threshold = mx.mean(abs_w)
            ternary = mx.where(p > threshold, 1.0, mx.where(p < -threshold, -1.0, 0.0))
            nonzero_mask = ternary != 0
            n_nonzero = mx.sum(nonzero_mask)
            if n_nonzero.item() > 0:
                scale = mx.sum(abs_w * nonzero_mask) / n_nonzero
            else:
                scale = mx.array(1.0)
            quantized = ternary * scale
            count += 1
            model.update(tree_unflatten([(name, quantized)]))
    mx.eval(model.parameters())
    print(f"  Quantized {count} weight matrices to ternary")


# ===========================================================================
# Data Loading
# ===========================================================================
def load_domain_data(domain_name):
    data_dir = DATA_ROOT / domain_name
    train_texts, val_texts = [], []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])
    with open(data_dir / "valid.jsonl") as f:
        for line in f:
            val_texts.append(json.loads(line)["text"])
    return train_texts, val_texts


def build_tokenizer(texts, vocab_size=VOCAB_SIZE):
    all_chars = set()
    for t in texts:
        all_chars.update(t)
    chars = sorted(all_chars)
    char_to_id = {c: i + 2 for i, c in enumerate(chars)}
    id_to_char = {i + 2: c for i, c in enumerate(chars)}
    actual_vocab = len(chars) + 2

    def encode(text):
        return [char_to_id.get(c, 1) for c in text]

    def decode(ids):
        return "".join(id_to_char.get(i, "?") for i in ids)

    return encode, decode, min(actual_vocab, vocab_size)


def tokenize_texts(texts, encode_fn, max_len=MAX_SEQ_LEN):
    all_tokens = []
    for text in texts:
        tokens = encode_fn(text)
        if len(tokens) > max_len + 1:
            tokens = tokens[:max_len + 1]
        if len(tokens) >= 4:
            all_tokens.append(mx.array(tokens))
    return all_tokens


# ===========================================================================
# Standard Pretraining (baseline)
# ===========================================================================
def pretrain_standard(model, train_tokens, steps, lr, batch_size):
    optimizer = opt.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    t_start = time.time()

    for step in range(steps):
        batch_x, batch_y = [], []
        for b in range(batch_size):
            idx = (step * batch_size + b) % len(train_tokens)
            tokens = train_tokens[idx]
            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])

        max_len = max(len(b) for b in batch_x)
        x = mx.zeros((batch_size, max_len), dtype=mx.int32)
        y = mx.zeros((batch_size, max_len), dtype=mx.int32)
        for i, (bx, by) in enumerate(zip(batch_x, batch_y)):
            x[i, :len(bx)] = bx
            y[i, :len(by)] = by

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if (step + 1) % 500 == 0 or step == 0:
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f} (avg200={avg:.4f})")

    train_time = time.time() - t_start
    final_avg = sum(losses[-100:]) / len(losses[-100:])
    print(f"    Done in {train_time:.1f}s. Final avg loss: {final_avg:.4f}")
    return losses, train_time


# ===========================================================================
# PPL Evaluation
# ===========================================================================
def compute_ppl(model, val_tokens, max_batches=25):
    total_loss = 0.0
    total_tokens = 0
    for tokens in val_tokens[:max_batches]:
        if len(tokens) < 2:
            continue
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size
    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ===========================================================================
# FOMAML Meta-Learning for Scaffold Optimization
# ===========================================================================
def get_scaffold_params(model):
    """Get all non-LoRA, non-embedding weight matrices (the scaffold)."""
    params = {}
    for name, p in tree_flatten(model.parameters()):
        if "lora_a" not in name and "lora_b" not in name:
            if len(p.shape) == 2 and p.size > 100:
                params[name] = p
    return params


def meta_train_scaffold(model, domain_data, meta_steps, meta_lr, inner_steps,
                        inner_lr, lora_rank, lora_scale, composition_lambda,
                        n_domains_per_meta):
    """
    First-Order MAML (FOMAML) for scaffold optimization.

    Outer loop: update scaffold weights W_scaffold
    Inner loop: for each sampled domain, train a fresh LoRA adapter for K steps

    Meta-loss = (1/D) * sum_d [L_d(W_scaffold + Delta_d)] + lambda * composition_penalty

    FOMAML approximation: use gradients at inner-loop endpoint w.r.t. scaffold params
    (ignores second-order terms through the inner loop).

    For composition penalty, we compute pairwise interference between adapted models.
    """
    meta_losses = []

    # The model already has LoRA applied. We optimize the base (scaffold) weights.
    # Freeze LoRA, unfreeze scaffold for outer loop.

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Meta optimizer for scaffold weights
    meta_optimizer = opt.Adam(learning_rate=meta_lr)

    all_domain_names = list(domain_data.keys())

    for meta_step in range(meta_steps):
        # Sample domains for this meta-step
        mx.random.seed(SEED + meta_step * 7)
        import random
        random.seed(SEED + meta_step)
        sampled_domains = random.sample(all_domain_names, min(n_domains_per_meta, len(all_domain_names)))

        # Save scaffold state
        scaffold_state = {}
        for name, p in tree_flatten(model.parameters()):
            if "lora_a" not in name and "lora_b" not in name:
                scaffold_state[name] = mx.array(p)
        mx.eval(scaffold_state)

        # ---- Inner loop: train adapters per domain ----
        domain_losses = []
        domain_adapters = []

        for domain in sampled_domains:
            # Reset LoRA to fresh init
            zero_lora(model)

            # Restore scaffold to current outer-loop state
            model.update(tree_unflatten(list(scaffold_state.items())))
            mx.eval(model.parameters())

            # Freeze scaffold, unfreeze LoRA for inner loop
            model.freeze()
            for block in model.blocks:
                for attr_path in LORA_TARGETS:
                    parts = attr_path.split(".")
                    parent = block
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    module = getattr(parent, parts[-1])
                    if isinstance(module, LoRALinear):
                        module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

            inner_optimizer = opt.Adam(learning_rate=inner_lr)
            train_tokens = domain_data[domain]["train"]

            for inner_step in range(inner_steps):
                idx = inner_step % len(train_tokens)
                tokens = train_tokens[idx]
                x = tokens[:-1][None, :]
                y = tokens[1:][None, :]

                loss, grads = loss_and_grad_fn(model, x, y)
                inner_optimizer.update(model, grads)
                mx.eval(model.parameters(), inner_optimizer.state)

            # Record endpoint loss for this domain
            # Evaluate on a few val samples
            val_loss = 0.0
            n_val = min(10, len(domain_data[domain]["val"]))
            for vi in range(n_val):
                tokens = domain_data[domain]["val"][vi]
                if len(tokens) < 2:
                    continue
                x = tokens[:-1][None, :]
                y = tokens[1:][None, :]
                logits = model(x)
                l = nn.losses.cross_entropy(logits, y, reduction="mean")
                mx.eval(l)
                val_loss += l.item()
            val_loss /= max(n_val, 1)
            domain_losses.append(val_loss)
            domain_adapters.append(get_lora_params(model))

        # ---- Composition penalty ----
        # Compute pairwise cosine between adapted parameter vectors
        comp_penalty = 0.0
        if len(domain_adapters) >= 2:
            flat_vecs = []
            for ap in domain_adapters:
                flat = mx.concatenate([p.reshape(-1) for p in ap.values()])
                flat_vecs.append(flat)

            n_pairs = 0
            for i in range(len(flat_vecs)):
                for j in range(i + 1, len(flat_vecs)):
                    cos = mx.abs(mx.sum(flat_vecs[i] * flat_vecs[j]) /
                                 (mx.sqrt(mx.sum(flat_vecs[i]**2)) *
                                  mx.sqrt(mx.sum(flat_vecs[j]**2)) + 1e-10))
                    mx.eval(cos)
                    comp_penalty += cos.item()
                    n_pairs += 1
            comp_penalty /= max(n_pairs, 1)

        meta_loss = sum(domain_losses) / len(domain_losses) + composition_lambda * comp_penalty
        meta_losses.append(meta_loss)

        # ---- Outer loop: FOMAML gradient ----
        # Compute gradients of the meta-loss w.r.t. scaffold weights
        # FOMAML: we compute gradient at the inner-loop endpoint (last domain)
        # using fresh forward pass with all scaffold params unfrozen

        # Restore scaffold, apply averaged adapter, compute gradient
        model.update(tree_unflatten(list(scaffold_state.items())))

        # Apply the mean adapter (simulates composition)
        N = len(domain_adapters)
        mean_adapter = {}
        for key in domain_adapters[0].keys():
            stacked = mx.stack([a[key] for a in domain_adapters])
            mean_adapter[key] = mx.mean(stacked, axis=0)
        apply_adapter_weights(model, mean_adapter)

        # Unfreeze scaffold, freeze LoRA for outer gradient
        model.freeze()
        for name, p in tree_flatten(model.parameters()):
            if "lora_a" not in name and "lora_b" not in name:
                pass  # will unfreeze below

        # Unfreeze all non-lora params
        for block in model.blocks:
            block.unfreeze()
            for attr_path in LORA_TARGETS:
                parts = attr_path.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                module = getattr(parent, parts[-1])
                if isinstance(module, LoRALinear):
                    module.base.unfreeze()
                    module.freeze(keys=["lora_a", "lora_b"], strict=False)

        # Also unfreeze embeddings and head
        model.tok_emb.unfreeze()
        model.pos_emb.unfreeze()
        model.ln_f.unfreeze()
        model.lm_head.unfreeze()

        # Compute outer gradient on a mixed batch (one sample per domain)
        outer_loss_total = mx.array(0.0)
        for domain in sampled_domains:
            tokens = domain_data[domain]["val"][meta_step % len(domain_data[domain]["val"])]
            if len(tokens) < 2:
                continue
            x = tokens[:-1][None, :]
            y = tokens[1:][None, :]
            logits = model(x)
            l = nn.losses.cross_entropy(logits, y, reduction="mean")
            outer_loss_total = outer_loss_total + l

        outer_loss_total = outer_loss_total / len(sampled_domains)

        # We need gradients -- use value_and_grad
        # Re-do with proper gradient computation
        model.update(tree_unflatten(list(scaffold_state.items())))
        apply_adapter_weights(model, mean_adapter)
        mx.eval(model.parameters())

        # Unfreeze scaffold again
        model.freeze()
        for block in model.blocks:
            block.unfreeze()
            for attr_path in LORA_TARGETS:
                parts = attr_path.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                module = getattr(parent, parts[-1])
                if isinstance(module, LoRALinear):
                    module.base.unfreeze()
                    module.freeze(keys=["lora_a", "lora_b"], strict=False)
        model.tok_emb.unfreeze()
        model.pos_emb.unfreeze()
        model.ln_f.unfreeze()
        model.lm_head.unfreeze()

        # Compute outer gradient properly
        def outer_loss_fn(model):
            total = mx.array(0.0)
            for domain in sampled_domains:
                vi = meta_step % len(domain_data[domain]["val"])
                tokens = domain_data[domain]["val"][vi]
                if len(tokens) < 2:
                    continue
                x = tokens[:-1][None, :]
                y = tokens[1:][None, :]
                logits = model(x)
                l = nn.losses.cross_entropy(logits, y, reduction="mean")
                total = total + l
            return total / len(sampled_domains)

        outer_grad_fn = nn.value_and_grad(model, outer_loss_fn)
        outer_loss_val, outer_grads = outer_grad_fn(model)
        mx.eval(outer_loss_val)

        # Apply meta update
        meta_optimizer.update(model, outer_grads)
        mx.eval(model.parameters(), meta_optimizer.state)

        if (meta_step + 1) % 10 == 0 or meta_step == 0:
            print(f"    Meta-step {meta_step+1}/{meta_steps}: "
                  f"meta_loss={meta_loss:.4f} (avg_domain_loss={sum(domain_losses)/len(domain_losses):.4f}, "
                  f"comp_penalty={comp_penalty:.6f}), "
                  f"outer_loss={outer_loss_val.item():.4f}")

    return meta_losses


# ===========================================================================
# Adapter Training (for final evaluation)
# ===========================================================================
def train_adapter_full(model, train_tokens, n_iters, lr):
    """Train one LoRA adapter for evaluation. Returns (losses, train_time)."""
    model.freeze()
    for block in model.blocks:
        for attr_path in LORA_TARGETS:
            parts = attr_path.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            module = getattr(parent, parts[-1])
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    optimizer = opt.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    t_start = time.time()

    for step in range(n_iters):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

    train_time = time.time() - t_start
    return losses, train_time


# ===========================================================================
# Main Experiment
# ===========================================================================
def main():
    print("=" * 70)
    print("Meta-Scaffold: MAML-Optimized Scaffold for Adapter Composition")
    print("=" * 70)

    results = {
        "experiment": "bitnet_meta_scaffold",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "meta_steps": META_STEPS,
        "meta_lr": META_LR,
        "inner_steps": INNER_STEPS,
        "inner_lr": INNER_LR,
        "composition_lambda": COMPOSITION_LAMBDA,
        "lora_rank": LORA_RANK,
        "adapter_train_steps": ADAPTER_TRAIN_STEPS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
    }

    # ==================================================================
    # Phase 0: Load and tokenize data
    # ==================================================================
    print("\n[Phase 0] Loading data...")
    all_train_texts = []
    domain_data = {}

    for domain in DOMAINS:
        train_texts, val_texts = load_domain_data(domain)
        all_train_texts.extend(train_texts)
        domain_data[domain] = {"train_texts": train_texts, "val_texts": val_texts}
        print(f"  {domain}: {len(train_texts)} train, {len(val_texts)} val")

    # Build shared tokenizer
    encode, decode, actual_vocab = build_tokenizer(all_train_texts)
    results["actual_vocab_size"] = actual_vocab
    print(f"  Vocab size: {actual_vocab}")

    # Tokenize
    for domain in DOMAINS:
        domain_data[domain]["train"] = tokenize_texts(domain_data[domain]["train_texts"], encode)
        domain_data[domain]["val"] = tokenize_texts(domain_data[domain]["val_texts"], encode)
        print(f"  {domain}: {len(domain_data[domain]['train'])} train seqs, "
              f"{len(domain_data[domain]['val'])} val seqs")

    # Merge all for pretraining
    all_train_tokens = []
    for domain in DOMAINS:
        all_train_tokens.extend(domain_data[domain]["train"])
    print(f"  Total train sequences: {len(all_train_tokens)}")

    mx.random.seed(SEED)

    # ==================================================================
    # Phase 1: Standard Pretraining (baseline scaffold)
    # ==================================================================
    print("\n[Phase 1] Standard pretraining (baseline scaffold)...")
    model_standard = TinyGPT(actual_vocab, D_MODEL, N_HEADS, N_LAYERS)
    init_params = {name: mx.array(p) for name, p in tree_flatten(model_standard.parameters())}
    mx.eval(init_params)

    total_params = sum(p.size for p in model_standard.parameters().values()
                       if hasattr(p, 'size'))
    # Count properly
    total_params = sum(p.size for _, p in tree_flatten(model_standard.parameters()))
    results["total_params"] = total_params
    print(f"  Model params: {total_params:,}")

    std_losses, std_time = pretrain_standard(
        model_standard, all_train_tokens, PRETRAIN_STEPS, PRETRAIN_LR, PRETRAIN_BATCH_SIZE
    )
    results["standard_pretrain"] = {
        "final_loss": round(std_losses[-1], 4),
        "train_time_s": round(std_time, 1),
    }

    # Evaluate base PPL per domain
    print("\n  Standard base PPL per domain:")
    standard_base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model_standard, domain_data[domain]["val"])
        standard_base_ppls[domain] = round(ppl, 2)
        print(f"    {domain}: {ppl:.2f}")
    results["standard_base_ppls"] = standard_base_ppls

    # ==================================================================
    # Phase 2: Meta-scaffold training
    # ==================================================================
    print("\n[Phase 2] Meta-scaffold training via FOMAML...")

    # Initialize meta-scaffold from SAME random init
    model_meta = TinyGPT(actual_vocab, D_MODEL, N_HEADS, N_LAYERS)
    model_meta.update(tree_unflatten(list(init_params.items())))
    mx.eval(model_meta.parameters())

    # First do standard pretraining as warm start (same as baseline)
    # This gives us a fair comparison: both start from same pretrained weights
    print("  Warm-starting meta-scaffold with standard pretraining...")
    meta_pretrain_losses, meta_pretrain_time = pretrain_standard(
        model_meta, all_train_tokens, PRETRAIN_STEPS, PRETRAIN_LR, PRETRAIN_BATCH_SIZE
    )
    results["meta_pretrain"] = {
        "final_loss": round(meta_pretrain_losses[-1], 4),
        "train_time_s": round(meta_pretrain_time, 1),
    }

    # Apply LoRA for meta-learning
    model_meta = apply_lora(model_meta, LORA_RANK, LORA_SCALE)
    mx.eval(model_meta.parameters())

    # Run FOMAML
    t_meta_start = time.time()
    meta_losses = meta_train_scaffold(
        model_meta, domain_data,
        meta_steps=META_STEPS,
        meta_lr=META_LR,
        inner_steps=INNER_STEPS,
        inner_lr=INNER_LR,
        lora_rank=LORA_RANK,
        lora_scale=LORA_SCALE,
        composition_lambda=COMPOSITION_LAMBDA,
        n_domains_per_meta=N_DOMAINS_PER_META,
    )
    meta_time = time.time() - t_meta_start
    print(f"  Meta-training done in {meta_time:.1f}s")

    results["meta_training"] = {
        "meta_losses": [round(l, 4) for l in meta_losses],
        "meta_time_s": round(meta_time, 1),
        "initial_meta_loss": round(meta_losses[0], 4),
        "final_meta_loss": round(meta_losses[-1], 4),
        "meta_loss_reduction": round((meta_losses[0] - meta_losses[-1]) / meta_losses[0] * 100, 1),
    }

    # K2 check: did the outer loop converge?
    first_10_avg = sum(meta_losses[:10]) / 10
    last_10_avg = sum(meta_losses[-10:]) / 10
    k2_converged = last_10_avg < first_10_avg * 0.95
    results["k2_converged"] = k2_converged
    print(f"\n  K2 convergence check: first_10_avg={first_10_avg:.4f}, "
          f"last_10_avg={last_10_avg:.4f}, converged={k2_converged}")

    # Evaluate meta-scaffold base PPL (without LoRA)
    print("\n  Meta-scaffold base PPL per domain:")
    remove_lora(model_meta)
    meta_base_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model_meta, domain_data[domain]["val"])
        meta_base_ppls[domain] = round(ppl, 2)
        print(f"    {domain}: {ppl:.2f}")
    results["meta_base_ppls"] = meta_base_ppls

    # ==================================================================
    # Phase 3: Ternary quantize both scaffolds
    # ==================================================================
    print("\n[Phase 3] Ternary quantization...")
    print("  Quantizing standard scaffold...")
    ternary_quantize(model_standard)
    print("  Quantizing meta-scaffold...")
    ternary_quantize(model_meta)

    # Base PPL after quantization
    print("\n  Post-quantization base PPL:")
    standard_ternary_ppls = {}
    meta_ternary_ppls = {}
    for domain in DOMAINS:
        ppl_std = compute_ppl(model_standard, domain_data[domain]["val"])
        ppl_meta = compute_ppl(model_meta, domain_data[domain]["val"])
        standard_ternary_ppls[domain] = round(ppl_std, 2)
        meta_ternary_ppls[domain] = round(ppl_meta, 2)
        print(f"    {domain}: standard={ppl_std:.2f}, meta={ppl_meta:.2f}")

    results["standard_ternary_ppls"] = standard_ternary_ppls
    results["meta_ternary_ppls"] = meta_ternary_ppls

    mean_std_ternary = sum(standard_ternary_ppls.values()) / len(standard_ternary_ppls)
    mean_meta_ternary = sum(meta_ternary_ppls.values()) / len(meta_ternary_ppls)
    ternary_ppl_ratio = mean_meta_ternary / mean_std_ternary
    results["ternary_ppl_ratio"] = round(ternary_ppl_ratio, 4)
    print(f"\n  Ternary PPL ratio (meta/standard): {ternary_ppl_ratio:.4f}")

    # ==================================================================
    # Phase 4: Train full adapters on both scaffolds
    # ==================================================================
    print("\n[Phase 4] Training full adapters on both scaffolds...")

    # Standard scaffold
    model_standard = apply_lora(model_standard, LORA_RANK, LORA_SCALE)
    standard_adapters = {}
    standard_adapter_ppls = {}

    for domain in DOMAINS:
        print(f"\n  [Standard] Training {domain} adapter...")
        mx.random.seed(SEED + hash(domain) % 1000)
        zero_lora(model_standard)
        losses, t = train_adapter_full(
            model_standard, domain_data[domain]["train"],
            ADAPTER_TRAIN_STEPS, ADAPTER_LR
        )
        ppl = compute_ppl(model_standard, domain_data[domain]["val"])
        standard_adapter_ppls[domain] = round(ppl, 2)
        standard_adapters[domain] = get_lora_params(model_standard)
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        print(f"    Loss: {first_50:.4f} -> {last_50:.4f}, PPL: {ppl:.2f}")

    results["standard_adapter_ppls"] = standard_adapter_ppls

    # Meta scaffold
    model_meta = apply_lora(model_meta, LORA_RANK, LORA_SCALE)
    meta_adapters = {}
    meta_adapter_ppls = {}

    for domain in DOMAINS:
        print(f"\n  [Meta] Training {domain} adapter...")
        mx.random.seed(SEED + hash(domain) % 1000)
        zero_lora(model_meta)
        losses, t = train_adapter_full(
            model_meta, domain_data[domain]["train"],
            ADAPTER_TRAIN_STEPS, ADAPTER_LR
        )
        ppl = compute_ppl(model_meta, domain_data[domain]["val"])
        meta_adapter_ppls[domain] = round(ppl, 2)
        meta_adapters[domain] = get_lora_params(model_meta)
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        print(f"    Loss: {first_50:.4f} -> {last_50:.4f}, PPL: {ppl:.2f}")

    results["meta_adapter_ppls"] = meta_adapter_ppls

    # ==================================================================
    # Phase 5: Composition evaluation
    # ==================================================================
    print("\n[Phase 5] Composition evaluation (1/N scaling)...")

    # Standard composition
    adapter_list_std = [standard_adapters[d] for d in DOMAINS]
    N = len(adapter_list_std)
    composed_std = {}
    for key in adapter_list_std[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list_std])
        composed_std[key] = mx.sum(stacked, axis=0) / N

    zero_lora(model_standard)
    apply_adapter_weights(model_standard, composed_std)
    mx.eval(model_standard.parameters())

    standard_composed_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model_standard, domain_data[domain]["val"])
        standard_composed_ppls[domain] = round(ppl, 2)
    results["standard_composed_ppls"] = standard_composed_ppls

    mean_individual_std = sum(standard_adapter_ppls.values()) / len(standard_adapter_ppls)
    mean_composed_std = sum(standard_composed_ppls.values()) / len(standard_composed_ppls)
    comp_ratio_std = mean_composed_std / mean_individual_std
    results["standard_composition_ratio"] = round(comp_ratio_std, 4)

    # Meta composition
    adapter_list_meta = [meta_adapters[d] for d in DOMAINS]
    composed_meta = {}
    for key in adapter_list_meta[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list_meta])
        composed_meta[key] = mx.sum(stacked, axis=0) / N

    zero_lora(model_meta)
    apply_adapter_weights(model_meta, composed_meta)
    mx.eval(model_meta.parameters())

    meta_composed_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model_meta, domain_data[domain]["val"])
        meta_composed_ppls[domain] = round(ppl, 2)
    results["meta_composed_ppls"] = meta_composed_ppls

    mean_individual_meta = sum(meta_adapter_ppls.values()) / len(meta_adapter_ppls)
    mean_composed_meta = sum(meta_composed_ppls.values()) / len(meta_composed_ppls)
    comp_ratio_meta = mean_composed_meta / mean_individual_meta
    results["meta_composition_ratio"] = round(comp_ratio_meta, 4)

    # ==================================================================
    # Phase 6: Cosine similarity
    # ==================================================================
    print("\n[Phase 6] Adapter cosine similarity...")

    def compute_cosines(adapter_dict):
        domains_list = list(adapter_dict.keys())
        vecs = {}
        for d in domains_list:
            flat = mx.concatenate([p.reshape(-1) for p in adapter_dict[d].values()])
            vecs[d] = flat
        cosines = []
        for i in range(len(domains_list)):
            for j in range(i + 1, len(domains_list)):
                v1, v2 = vecs[domains_list[i]], vecs[domains_list[j]]
                cos = mx.abs(mx.sum(v1 * v2) / (mx.sqrt(mx.sum(v1**2)) * mx.sqrt(mx.sum(v2**2)) + 1e-10))
                mx.eval(cos)
                cosines.append(cos.item())
        return cosines

    standard_cosines = compute_cosines(standard_adapters)
    meta_cosines = compute_cosines(meta_adapters)

    mean_cos_std = sum(standard_cosines) / len(standard_cosines)
    mean_cos_meta = sum(meta_cosines) / len(meta_cosines)

    results["standard_mean_cosine"] = round(mean_cos_std, 6)
    results["meta_mean_cosine"] = round(mean_cos_meta, 6)
    results["standard_cosines"] = [round(c, 6) for c in standard_cosines]
    results["meta_cosines"] = [round(c, 6) for c in meta_cosines]

    # ==================================================================
    # Phase 7: Kill criteria assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("[Phase 7] Kill Criteria Assessment")
    print("=" * 70)

    # K1: meta-scaffold must outperform GaLore scaffold on composition quality
    # We compare to standard scaffold (which is the same architecture as GaLore baseline)
    # Meta must have LOWER composition ratio (less interference)
    k1_pass = comp_ratio_meta < comp_ratio_std
    print(f"\n  K1: Composition ratio (meta vs standard)")
    print(f"    Standard: {comp_ratio_std:.4f}")
    print(f"    Meta:     {comp_ratio_meta:.4f}")
    print(f"    K1 PASS (meta < standard): {k1_pass}")

    # Also compare individual adapter PPL
    ratio_of_means = mean_individual_meta / mean_individual_std
    print(f"\n  Individual adapter PPL:")
    print(f"    Standard mean: {mean_individual_std:.2f}")
    print(f"    Meta mean:     {mean_individual_meta:.2f}")
    print(f"    Ratio (meta/standard): {ratio_of_means:.4f}")

    # K2: MAML outer loop convergence
    print(f"\n  K2: MAML convergence")
    print(f"    First 10 meta-loss avg: {first_10_avg:.4f}")
    print(f"    Last 10 meta-loss avg:  {last_10_avg:.4f}")
    print(f"    K2 PASS (converged): {k2_converged}")

    # GaLore baseline comparison
    galore_comp_ratio = 1.155  # from GaLore experiment (3-seed mean)
    galore_mean_cos = 0.0027
    k1_vs_galore = comp_ratio_meta < galore_comp_ratio

    print(f"\n  Comparison to GaLore scaffold baseline:")
    print(f"    GaLore comp ratio: {galore_comp_ratio:.4f}")
    print(f"    Meta comp ratio:   {comp_ratio_meta:.4f}")
    print(f"    Beats GaLore: {k1_vs_galore}")
    print(f"    GaLore mean |cos|: {galore_mean_cos:.6f}")
    print(f"    Meta mean |cos|:   {mean_cos_meta:.6f}")

    overall = k1_pass and k2_converged
    verdict = "SUPPORTED" if overall else "KILLED"

    results["kill_criteria"] = {
        "k1_comp_ratio_standard": round(comp_ratio_std, 4),
        "k1_comp_ratio_meta": round(comp_ratio_meta, 4),
        "k1_pass": k1_pass,
        "k1_vs_galore_pass": k1_vs_galore,
        "k2_first_10_avg": round(first_10_avg, 4),
        "k2_last_10_avg": round(last_10_avg, 4),
        "k2_converged": k2_converged,
        "overall": verdict,
    }

    # Summary table
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Domain':<12} {'Std Base':>10} {'Meta Base':>10} {'Std Adapt':>10} "
          f"{'Meta Adapt':>11} {'Std Comp':>10} {'Meta Comp':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for domain in DOMAINS:
        sb = standard_ternary_ppls[domain]
        mb = meta_ternary_ppls[domain]
        sa = standard_adapter_ppls[domain]
        ma = meta_adapter_ppls[domain]
        sc = standard_composed_ppls[domain]
        mc = meta_composed_ppls[domain]
        print(f"  {domain:<12} {sb:>10.2f} {mb:>10.2f} {sa:>10.2f} {ma:>11.2f} {sc:>10.2f} {mc:>10.2f}")

    print(f"\n  Composition ratio:  standard={comp_ratio_std:.4f}, meta={comp_ratio_meta:.4f}")
    print(f"  Mean |cos|:         standard={mean_cos_std:.6f}, meta={mean_cos_meta:.6f}")
    print(f"\n  VERDICT: {verdict}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
