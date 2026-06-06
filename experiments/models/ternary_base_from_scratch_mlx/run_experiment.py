#!/usr/bin/env python3
"""Ternary Base From Scratch on MLX: Train a ternary transformer using STE.

Kill criteria:
  K1 (id=183): Val loss doesn't decrease below random baseline within 2000 steps -> KILL
  K2 (id=184): Ternary base PPL > 3x FP32 baseline at same d=256 architecture -> KILL
  K3 (id=185): Adapter composition ratio > 2.0 on ternary base -> KILL

Success criteria:
  S1: Ternary base trains to within 2x PPL of FP32 baseline (~16, target < 32)
  S2: 5 domain adapters compose with ratio < 1.5
  S3: Adapter orthogonality holds (mean |cos| < 0.05)
"""

import gc
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"

# Architecture hyperparams (from MATH.md)
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 64
BLOCK_SIZE = 32
MLP_DIM = 4 * D_MODEL  # 1024
LORA_RANK = 8

# Training hyperparams
BASE_STEPS = 4000
BASE_LR = 3e-4
ADAPTER_STEPS = 1500
ADAPTER_LR = 1e-3
BATCH_SIZE = 64
FP32_STEPS = 4000
FP32_LR = 3e-4


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# BitLinear: Ternary weights with STE
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE in forward pass.

    Maintains FP32 latent weights. Forward pass quantizes to {-alpha, 0, +alpha}.
    Backward pass uses straight-through estimator.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier init scaled for ternary convergence
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale

    def __call__(self, x):
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        # STE: forward uses w_q, backward passes through w
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


# ============================================================================
# Transformer Architecture
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = BitLinear(n_embd, n_embd)
        self.wk = BitLinear(n_embd, n_embd)
        self.wv = BitLinear(n_embd, n_embd)
        self.wo = BitLinear(n_embd, n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        # Causal mask: additive, -inf on upper triangle (future tokens)
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = BitLinear(n_embd, 4 * n_embd)
        self.fc2 = BitLinear(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TernaryGPT(nn.Module):
    """Ternary GPT: all Linear layers replaced with BitLinear (STE quantization).
    Embeddings and norms remain FP32."""
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 256, n_head: int = 4, n_layer: int = 6):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = BitLinear(n_embd, vocab_size)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class FP32GPT(nn.Module):
    """FP32 baseline GPT: identical architecture, standard nn.Linear."""
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 256, n_head: int = 4, n_layer: int = 6):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [FP32Block(n_embd, n_head) for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class FP32Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        # Causal mask: additive, -inf on upper triangle (future tokens)
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class FP32MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class FP32Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = FP32Attention(n_embd, n_head)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = FP32MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Ternary LoRA Adapter with Grassmannian-init A matrices
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA adapter on top of a frozen BitLinear.

    A matrix: frozen, Grassmannian-initialized (orthogonal across adapters)
    B matrix: trainable, quantized to ternary via STE
    """
    def __init__(self, base_linear: BitLinear, rank: int, a_matrix: mx.array):
        super().__init__()
        self.base = base_linear
        self.base.freeze()
        out_features = base_linear.weight.shape[0]
        self.a_matrix = a_matrix  # (rank, in_features) -- frozen
        self.a_matrix = mx.stop_gradient(self.a_matrix)  # ensure no grads
        # B initialized to zero (standard LoRA init)
        self.b_matrix = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = self.base(x)
        # LoRA: x @ A^T @ B^T (with ternary B via STE)
        b = self.b_matrix
        alpha_b = mx.mean(mx.abs(b)) + 1e-7
        b_scaled = b / alpha_b
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha_b
        b_ste = b + mx.stop_gradient(b_q - b)

        # x @ A^T gives (B, T, rank), then @ B^T gives (B, T, out)
        lora_out = (x @ self.a_matrix.T) @ b_ste.T
        return base_out + lora_out


def generate_grassmannian_bases(n_adapters: int, rank: int, dim: int, seed: int = 42):
    """Generate n_adapters orthogonal A matrices on Grassmannian Gr(rank, dim).

    Uses QR decomposition of random matrices to get orthonormal bases,
    then assigns consecutive rank-dimensional subspaces to each adapter.
    """
    mx.random.seed(seed)
    # Generate a large random matrix and orthogonalize
    # We need n_adapters * rank orthogonal vectors in R^dim
    total_rank = n_adapters * rank
    assert total_rank <= dim, f"Need {total_rank} orthogonal vectors but dim={dim}"

    # Random matrix -> QR for orthogonal columns
    random_mat = mx.random.normal(shape=(dim, total_rank))
    mx.eval(random_mat)
    # QR decomposition -- Q columns are orthonormal
    # MLX doesn't have QR, so use numpy for this one-time computation
    import numpy as np
    Q, _ = np.linalg.qr(np.array(random_mat))
    Q = mx.array(Q[:, :total_rank])

    bases = []
    for i in range(n_adapters):
        start = i * rank
        end = start + rank
        A_i = Q[:, start:end].T  # (rank, dim)
        bases.append(A_i)
    return bases


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    """Load names dataset, return train/val CharDatasets."""
    from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split

    names = load_names()
    tok = CharTokenizer(names)
    train_names, val_names = train_val_split(names)
    train_ds = CharDataset(train_names, tok, block_size=BLOCK_SIZE)
    val_ds = CharDataset(val_names, tok, block_size=BLOCK_SIZE)

    # Also prepare domain splits for adapter training
    domains = domain_split(names, method="quintary")
    domain_datasets = {}
    for dname, dnames in domains.items():
        dtrain, dval = train_val_split(dnames)
        domain_datasets[dname] = {
            "train": CharDataset(dtrain, tok, block_size=BLOCK_SIZE),
            "val": CharDataset(dval, tok, block_size=BLOCK_SIZE),
        }

    return tok, train_ds, val_ds, domain_datasets


# ============================================================================
# Training utilities
# ============================================================================

def compute_ppl(model, dataset, n_batches=20, batch_size=64):
    """Compute perplexity on a dataset."""
    rng = random.Random(0)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    avg_loss = total_loss / n_batches
    return math.exp(avg_loss)


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def count_trainable(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


# ============================================================================
# Phase 1: Train FP32 baseline
# ============================================================================

def phase_fp32_baseline(train_ds, val_ds, vocab_size):
    """Train FP32 baseline to establish PPL target."""
    print("\n" + "="*60)
    print("PHASE 1: FP32 Baseline Training")
    print("="*60)

    model = FP32GPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                     n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"FP32 model params: {n_params:,}")
    log_memory("fp32-init")

    optimizer = opt.Adam(learning_rate=FP32_LR)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    gc.disable()
    losses = []
    t0 = time.time()
    for step in range(1, FP32_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == FP32_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{FP32_STEPS} | loss {loss_val:.4f} | "
                  f"time {elapsed:.1f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    fp32_ppl = compute_ppl(model, val_ds)
    print(f"\nFP32 baseline PPL: {fp32_ppl:.2f}")
    print(f"Training time: {train_time:.1f}s")
    log_memory("fp32-done")

    result = {
        "fp32_ppl": fp32_ppl,
        "fp32_final_loss": losses[-1],
        "fp32_train_time_s": round(train_time, 1),
        "fp32_params": n_params,
    }

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Train ternary base model
# ============================================================================

def phase_ternary_base(train_ds, val_ds, vocab_size):
    """Train ternary base model with STE."""
    print("\n" + "="*60)
    print("PHASE 2: Ternary Base Training (STE)")
    print("="*60)

    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"Ternary model params: {n_params:,}")
    log_memory("ternary-init")

    optimizer = opt.Adam(learning_rate=BASE_LR)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    random_baseline_loss = math.log(28)  # ln(V) for random baseline
    print(f"Random baseline loss (ln(28)): {random_baseline_loss:.4f}")

    gc.disable()
    losses = []
    loss_below_random = False
    t0 = time.time()
    for step in range(1, BASE_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)

        # K1 check: does loss go below random baseline within first 2000 steps?
        if step <= 2000 and loss_val < random_baseline_loss:
            if not loss_below_random:
                print(f"  [K1] Loss below random baseline at step {step}: {loss_val:.4f} < {random_baseline_loss:.4f}")
                loss_below_random = True

        if step % 500 == 0 or step == BASE_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{BASE_STEPS} | loss {loss_val:.4f} | "
                  f"time {elapsed:.1f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0

    # K1 evaluation
    k1_pass = loss_below_random
    print(f"\n[K1] Loss below random within 2000 steps: {'PASS' if k1_pass else 'FAIL'}")

    # Compute PPL
    ternary_ppl = compute_ppl(model, val_ds)
    print(f"Ternary base PPL: {ternary_ppl:.2f}")
    print(f"Training time: {train_time:.1f}s")
    log_memory("ternary-done")

    # K4: Check deadzone trapping (>20% of weights trapped at zero)
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    total_ternary = 0
    zero_count = 0
    for name, param in flat_params.items():
        if "wte" in name or "wpe" in name or "norm" in name:
            continue  # skip embeddings and norms
        alpha = mx.mean(mx.abs(param))
        w_scaled = param / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1)
        mx.eval(w_q)
        n = w_q.size
        n_zero = int(mx.sum(w_q == 0).item())
        total_ternary += n
        zero_count += n_zero
    zero_frac = zero_count / total_ternary if total_ternary > 0 else 0
    k4_pass = zero_frac <= 0.20
    print(f"\n[K4] Zero fraction: {zero_frac:.4f} ({zero_count}/{total_ternary})")
    print(f"[K4] <= 20% zeros: {'PASS' if k4_pass else 'FAIL'}")

    # Save model weights for adapter training
    weights_path = EXPERIMENT_DIR / "ternary_base_weights.npz"
    mx.savez(str(weights_path), **flat_params)
    print(f"Saved ternary base to {weights_path}")

    result = {
        "ternary_ppl": ternary_ppl,
        "ternary_final_loss": losses[-1],
        "ternary_train_time_s": round(train_time, 1),
        "ternary_params": n_params,
        "k1_pass": k1_pass,
        "k1_step_below_random": next((i+1 for i, l in enumerate(losses[:2000]) if l < random_baseline_loss), None),
        "loss_history_500": [losses[i-1] for i in range(500, BASE_STEPS+1, 500)],
        "k4_zero_fraction": zero_frac,
        "k4_pass": k4_pass,
    }

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 3: Train domain adapters
# ============================================================================

def phase_train_adapters(domain_datasets, vocab_size):
    """Train 5 domain-specific ternary LoRA adapters on ternary base."""
    print("\n" + "="*60)
    print("PHASE 3: Domain Adapter Training")
    print("="*60)

    ADAPTERS_DIR.mkdir(exist_ok=True)

    # Generate Grassmannian bases for all adapters
    # Need separate bases for D_MODEL (attn, fc1-in, fc2-out, lm_head) and MLP_DIM (fc2-in)
    n_domains = len(domain_datasets)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    # Verify orthogonality of A matrices (d_model bases)
    print("A-matrix orthogonality check (d_model):")
    import numpy as np
    for i in range(n_domains):
        for j in range(i+1, n_domains):
            cos = float(mx.abs(mx.sum(bases_d[i] * bases_d[j])).item()) / (
                float(mx.sqrt(mx.sum(bases_d[i]**2)).item()) *
                float(mx.sqrt(mx.sum(bases_d[j]**2)).item()) + 1e-10
            )
            print(f"  |cos(A_{i}, A_{j})| = {cos:.6f}")

    domain_results = {}
    domain_names = sorted(domain_datasets.keys())

    for idx, dname in enumerate(domain_names):
        print(f"\n--- Adapter {idx+1}/{n_domains}: {dname} ---")
        ddata = domain_datasets[dname]

        # Fresh model load from saved weights
        model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
        weights_path = EXPERIMENT_DIR / "ternary_base_weights.npz"
        loaded = mx.load(str(weights_path))
        model.load_weights(list(loaded.items()))
        mx.eval(model.parameters())

        # Freeze base, attach LoRA to all BitLinear layers
        model.freeze()
        a_d = bases_d[idx]    # (rank, D_MODEL) for layers with input dim D_MODEL
        a_mlp = bases_mlp[idx]  # (rank, MLP_DIM) for fc2 (input dim MLP_DIM)

        # Wrap every BitLinear with TernaryLoRALinear (matching input dims)
        for layer in model.layers:
            layer.attn.wq = TernaryLoRALinear(layer.attn.wq, LORA_RANK, a_d)
            layer.attn.wk = TernaryLoRALinear(layer.attn.wk, LORA_RANK, a_d)
            layer.attn.wv = TernaryLoRALinear(layer.attn.wv, LORA_RANK, a_d)
            layer.attn.wo = TernaryLoRALinear(layer.attn.wo, LORA_RANK, a_d)
            layer.mlp.fc1 = TernaryLoRALinear(layer.mlp.fc1, LORA_RANK, a_d)
            layer.mlp.fc2 = TernaryLoRALinear(layer.mlp.fc2, LORA_RANK, a_mlp)
        model.lm_head = TernaryLoRALinear(model.lm_head, LORA_RANK, a_d)

        n_trainable = count_trainable(model)
        print(f"  Trainable params: {n_trainable:,}")

        optimizer = opt.Adam(learning_rate=ADAPTER_LR)

        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            B, T, V = logits.shape
            return nn.losses.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        rng = random.Random(42 + idx)

        gc.disable()
        t0 = time.time()
        for step in range(1, ADAPTER_STEPS + 1):
            inputs, targets = ddata["train"].get_batch(BATCH_SIZE, rng)
            loss, grads = loss_and_grad(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            if step % 500 == 0 or step == ADAPTER_STEPS:
                print(f"  step {step:4d}/{ADAPTER_STEPS} | loss {loss.item():.4f}")
        gc.enable()
        gc.collect()

        adapter_time = time.time() - t0

        # Evaluate single-adapter PPL on domain val set
        domain_ppl = compute_ppl(model, ddata["val"])
        print(f"  {dname} PPL (with adapter): {domain_ppl:.2f}")

        # Save only the B matrices (A matrices are deterministic from seed)
        adapter_params = {}
        for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
            adapter_params[name] = param
        save_path = ADAPTERS_DIR / f"{dname}.npz"
        mx.savez(str(save_path), **adapter_params)

        domain_results[dname] = {
            "ppl": domain_ppl,
            "train_time_s": round(adapter_time, 1),
            "trainable_params": n_trainable,
        }

        cleanup(model, optimizer)

    print(f"\nDomain adapter PPLs: { {k: v['ppl'] for k, v in domain_results.items()} }")
    return domain_results


# ============================================================================
# Phase 4: Composition test
# ============================================================================

def phase_composition_test(domain_datasets, vocab_size, domain_results):
    """Test N-adapter composition and measure composition ratio."""
    print("\n" + "="*60)
    print("PHASE 4: Adapter Composition Test")
    print("="*60)

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    # Load base model
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    weights_path = EXPERIMENT_DIR / "ternary_base_weights.npz"
    loaded = mx.load(str(weights_path))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())

    # Collect all adapter B matrices (filter out A matrices which are also saved)
    all_adapter_b = {}
    for dname in domain_names:
        adapter_path = ADAPTERS_DIR / f"{dname}.npz"
        adapter_data = mx.load(str(adapter_path))
        all_adapter_b[dname] = {k: v for k, v in adapter_data.items() if "b_matrix" in k}

    # Compute composed delta: for each BitLinear, sum (1/N) * B_i @ A_i
    # Then add to base weights directly
    composed_deltas = {}
    for idx, dname in enumerate(domain_names):
        adapter_b = all_adapter_b[dname]
        for bname, b_val in adapter_b.items():
            # Use MLP-dim A for fc2 layers, D_MODEL-dim A for everything else
            if ".mlp.fc2." in bname:
                a_matrix = bases_mlp[idx]
            else:
                a_matrix = bases_d[idx]
            # b_val is the B matrix for this layer
            # delta = B @ A, shape (out, in)
            delta = (b_val @ a_matrix) / n_domains
            if bname in composed_deltas:
                composed_deltas[bname] = composed_deltas[bname] + delta
            else:
                composed_deltas[bname] = delta

    # Apply composed deltas to base model weights
    # The adapter B matrices are stored with paths like layers.0.attn.wq.b_matrix
    # We need to map them back to base weight paths (layers.0.attn.wq.weight)
    base_params = dict(nn.utils.tree_flatten(model.parameters()))
    for bname, delta in composed_deltas.items():
        # bname looks like "layers.0.attn.wq.b_matrix" -> base is "layers.0.attn.wq.weight"
        base_key = bname.replace(".b_matrix", ".weight")
        if base_key in base_params:
            base_params[base_key] = base_params[base_key] + delta
            mx.eval(base_params[base_key])

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    # Evaluate composed model on each domain
    composed_ppls = {}
    for dname in domain_names:
        ddata = domain_datasets[dname]
        ppl = compute_ppl(model, ddata["val"])
        composed_ppls[dname] = ppl
        print(f"  Composed PPL on {dname}: {ppl:.2f}")

    # Compute composition ratio
    single_ppls = [domain_results[d]["ppl"] for d in domain_names]
    composed_ppl_vals = [composed_ppls[d] for d in domain_names]
    mean_single = sum(single_ppls) / len(single_ppls)
    mean_composed = sum(composed_ppl_vals) / len(composed_ppl_vals)
    composition_ratio = mean_composed / mean_single

    print(f"\nMean single-adapter PPL: {mean_single:.2f}")
    print(f"Mean composed PPL: {mean_composed:.2f}")
    print(f"Composition ratio: {composition_ratio:.3f}")
    print(f"[K3] Composition ratio < 2.0: {'PASS' if composition_ratio < 2.0 else 'FAIL'}")

    result = {
        "composed_ppls": composed_ppls,
        "mean_single_ppl": mean_single,
        "mean_composed_ppl": mean_composed,
        "composition_ratio": composition_ratio,
        "k3_pass": composition_ratio < 2.0,
    }

    cleanup(model)

    # Orthogonality of adapter deltas
    print("\n--- Adapter Orthogonality ---")
    import numpy as np
    cos_sims = []
    for i in range(n_domains):
        for j in range(i+1, n_domains):
            # Compute full delta for adapter i and j
            delta_i_parts = []
            delta_j_parts = []
            adapter_i = all_adapter_b[domain_names[i]]
            adapter_j = all_adapter_b[domain_names[j]]
            for bname in adapter_i:
                bi = adapter_i[bname]
                bj = adapter_j[bname]
                if ".mlp.fc2." in bname:
                    di = bi @ bases_mlp[i]
                    dj = bj @ bases_mlp[j]
                else:
                    di = bi @ bases_d[i]
                    dj = bj @ bases_d[j]
                delta_i_parts.append(di.reshape(-1))
                delta_j_parts.append(dj.reshape(-1))

            vec_i = mx.concatenate(delta_i_parts)
            vec_j = mx.concatenate(delta_j_parts)
            mx.eval(vec_i, vec_j)
            dot = mx.sum(vec_i * vec_j).item()
            norm_i = mx.sqrt(mx.sum(vec_i * vec_i)).item()
            norm_j = mx.sqrt(mx.sum(vec_j * vec_j)).item()
            cos = abs(dot) / (norm_i * norm_j + 1e-10)
            cos_sims.append(cos)
            print(f"  |cos(delta_{domain_names[i]}, delta_{domain_names[j]})| = {cos:.6f}")
            del vec_i, vec_j

    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    print(f"\nMean |cos|: {mean_cos:.6f}")
    print(f"[S3] Mean |cos| < 0.05: {'PASS' if mean_cos < 0.05 else 'FAIL'}")

    result["cos_similarities"] = cos_sims
    result["mean_cos_similarity"] = mean_cos
    result["s3_pass"] = mean_cos < 0.05

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    print("Loading data...")
    tok, train_ds, val_ds, domain_datasets = load_data()
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size}, Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")
    print(f"Domain datasets: {sorted(domain_datasets.keys())}")
    for dname, ddata in sorted(domain_datasets.items()):
        print(f"  {dname}: train={len(ddata['train'])}, val={len(ddata['val'])}")

    # Phase 1: FP32 baseline
    fp32_results = phase_fp32_baseline(train_ds, val_ds, vocab_size)
    log_memory("after-fp32")

    # Phase 2: Ternary base
    ternary_results = phase_ternary_base(train_ds, val_ds, vocab_size)
    log_memory("after-ternary")

    # K2 check
    fp32_ppl = fp32_results["fp32_ppl"]
    ternary_ppl = ternary_results["ternary_ppl"]
    ppl_ratio = ternary_ppl / fp32_ppl
    k2_pass = ternary_ppl < 3.0 * fp32_ppl
    print(f"\n[K2] Ternary PPL / FP32 PPL = {ppl_ratio:.2f}x (threshold: 3.0x)")
    print(f"[K2] {ternary_ppl:.2f} < {3.0 * fp32_ppl:.2f}: {'PASS' if k2_pass else 'FAIL'}")

    # S1 check
    s1_pass = ternary_ppl < 2.0 * fp32_ppl
    print(f"[S1] Ternary PPL < 2x FP32: {ternary_ppl:.2f} < {2.0 * fp32_ppl:.2f}: {'PASS' if s1_pass else 'FAIL'}")

    # Phase 3: Domain adapters
    domain_results = phase_train_adapters(domain_datasets, vocab_size)
    log_memory("after-adapters")

    # Phase 4: Composition test
    composition_results = phase_composition_test(domain_datasets, vocab_size, domain_results)
    log_memory("after-composition")

    # S2 check
    s2_pass = composition_results["composition_ratio"] < 1.5
    print(f"\n[S2] Composition ratio < 1.5: {composition_results['composition_ratio']:.3f}: {'PASS' if s2_pass else 'FAIL'}")

    # Aggregate results
    total_time = time.time() - t0_total
    results = {
        "experiment": "ternary_base_from_scratch_mlx",
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "block_size": BLOCK_SIZE,
            "lora_rank": LORA_RANK,
            "vocab_size": vocab_size,
        },
        "fp32_baseline": fp32_results,
        "ternary_base": ternary_results,
        "ppl_ratio": ppl_ratio,
        "domain_adapters": domain_results,
        "composition": composition_results,
        "kill_criteria": {
            "K1_loss_below_random": ternary_results["k1_pass"],
            "K2_ppl_within_3x": k2_pass,
            "K3_composition_ratio_below_2": composition_results["k3_pass"],
            "K4_no_deadzone_trapping": ternary_results["k4_pass"],
        },
        "success_criteria": {
            "S1_ppl_within_2x": s1_pass,
            "S2_composition_ratio_below_1_5": s2_pass,
            "S3_orthogonality": composition_results["s3_pass"],
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"FP32 baseline PPL:    {fp32_ppl:.2f}")
    print(f"Ternary base PPL:     {ternary_ppl:.2f} ({ppl_ratio:.2f}x FP32)")
    print(f"Composition ratio:    {composition_results['composition_ratio']:.3f}")
    print(f"Mean |cos|:           {composition_results['mean_cos_similarity']:.6f}")
    print()
    for k, v in results["kill_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    for k, v in results["success_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
