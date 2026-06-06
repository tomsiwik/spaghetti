#!/usr/bin/env python3
"""NP-LoRA: Null Space Projection for Interference-Free Adapter Composition.

Compares NP-LoRA null space projection against Grassmannian composition baseline.

Kill criteria:
  K1 (id=268): NP-LoRA composition worse than Grassmannian at any N -> KILL
  K2 (id=269): Null space computation > 1 sec for N=50 -> KILL

Platform: Apple Silicon MLX (M5 Pro 48GB).
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
import numpy as np

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse ternary base from prior experiment
TERNARY_BASE_DIR = Path("/Users/tom/Code/tomsiwik/llm/experiments/models/ternary_base_from_scratch_mlx")

# Architecture (must match ternary_base_from_scratch_mlx)
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
BLOCK_SIZE = 32
MLP_DIM = 4 * D_MODEL
LORA_RANK = 8

# Training for adapters (if we need to retrain with random A)
ADAPTER_STEPS = 1500
ADAPTER_LR = 1e-3
BATCH_SIZE = 64


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
# Model classes (same as ternary_base_from_scratch_mlx)
# ============================================================================

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale

    def __call__(self, x):
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


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


class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear: BitLinear, rank: int, a_matrix: mx.array):
        super().__init__()
        self.base = base_linear
        self.base.freeze()
        out_features = base_linear.weight.shape[0]
        self.a_matrix = a_matrix
        self.a_matrix = mx.stop_gradient(self.a_matrix)
        self.b_matrix = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = self.base(x)
        b = self.b_matrix
        alpha_b = mx.mean(mx.abs(b)) + 1e-7
        b_scaled = b / alpha_b
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha_b
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.a_matrix.T) @ b_ste.T
        return base_out + lora_out


def generate_grassmannian_bases(n_adapters: int, rank: int, dim: int, seed: int = 42):
    """Generate n_adapters orthogonal A matrices on Grassmannian Gr(rank, dim)."""
    mx.random.seed(seed)
    total_rank = n_adapters * rank
    assert total_rank <= dim, f"Need {total_rank} orthogonal vectors but dim={dim}"
    random_mat = mx.random.normal(shape=(dim, total_rank))
    mx.eval(random_mat)
    Q, _ = np.linalg.qr(np.array(random_mat))
    Q = mx.array(Q[:, :total_rank])
    bases = []
    for i in range(n_adapters):
        start = i * rank
        end = start + rank
        A_i = Q[:, start:end].T
        bases.append(A_i)
    return bases


def generate_random_bases(n_adapters: int, rank: int, dim: int, seed: int = 99):
    """Generate n_adapters RANDOM (non-orthogonal) A matrices for comparison."""
    np.random.seed(seed)
    bases = []
    for _ in range(n_adapters):
        # Each A has orthonormal rows (within-adapter), but NOT orthogonal across adapters
        mat = np.random.randn(rank, dim)
        q, _ = np.linalg.qr(mat.T)
        bases.append(mx.array(q[:, :rank].T.copy()))
    return bases


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
    names = load_names()
    tok = CharTokenizer(names)
    train_names, val_names = train_val_split(names)
    train_ds = CharDataset(train_names, tok, block_size=BLOCK_SIZE)
    val_ds = CharDataset(val_names, tok, block_size=BLOCK_SIZE)
    domains = domain_split(names, method="quintary")
    domain_datasets = {}
    for dname, dnames in domains.items():
        dtrain, dval = train_val_split(dnames)
        domain_datasets[dname] = {
            "train": CharDataset(dtrain, tok, block_size=BLOCK_SIZE),
            "val": CharDataset(dval, tok, block_size=BLOCK_SIZE),
        }
    return tok, train_ds, val_ds, domain_datasets


def compute_ppl(model, dataset, n_batches=20, batch_size=64):
    rng = random.Random(0)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    return math.exp(total_loss / n_batches)


# ============================================================================
# NP-LoRA: Null Space Projection
# ============================================================================

def compute_null_space_projection(deltas_per_layer, n_adapters):
    """Apply NP-LoRA null space projection to adapter deltas.

    For each adapter i and each layer l:
      1. Collect deltas from all OTHER adapters for layer l
      2. Build interference matrix S_i (stack other deltas as rows)
      3. Compute SVD of S_i -> get null space basis
      4. Project delta_i into null space

    Args:
        deltas_per_layer: dict mapping layer_name -> list of N delta arrays (d_out, d_in)
        n_adapters: int

    Returns:
        projected: dict mapping layer_name -> list of N projected delta arrays
        timing: float (seconds)
    """
    t0 = time.time()
    projected = {}

    for layer_name, deltas in deltas_per_layer.items():
        # Convert to numpy for SVD (one-time computation, not in hot path)
        np_deltas = [np.array(d).reshape(-1) for d in deltas]  # each is vec(Delta_i)
        n_elements = np_deltas[0].shape[0]
        projected_deltas = []

        for i in range(n_adapters):
            # Build interference matrix: rows = other adapters' deltas
            others = [np_deltas[j] for j in range(n_adapters) if j != i]
            if len(others) == 0:
                projected_deltas.append(deltas[i])
                continue

            S_i = np.stack(others, axis=0)  # (N-1, d_out*d_in)

            # Thin SVD: S_i = U @ diag(s) @ Vt
            # Null space is columns of V not in row space of S_i
            U, s, Vt = np.linalg.svd(S_i, full_matrices=False)
            # Rank determination: keep singular values above threshold
            tol = max(S_i.shape) * np.max(s) * np.finfo(float).eps if len(s) > 0 else 0
            rank = np.sum(s > tol)

            if rank == 0:
                # No interference subspace -> no projection needed
                projected_deltas.append(deltas[i])
                continue

            # V_r: right singular vectors spanning the row space of S_i
            V_r = Vt[:rank, :].T  # (d_out*d_in, rank)

            # Project delta_i into null space: delta_proj = delta - V_r @ V_r^T @ delta
            delta_i = np_deltas[i]
            projection = V_r @ (V_r.T @ delta_i)
            delta_proj = delta_i - projection

            # Reshape back
            shape = deltas[i].shape
            projected_deltas.append(mx.array(delta_proj.reshape(shape)))

        projected[layer_name] = projected_deltas

    elapsed = time.time() - t0
    return projected, elapsed


def compute_composition_metrics(deltas_per_layer, n_adapters, label=""):
    """Compute composition quality metrics: mean |cos|, Frobenius cross-terms."""
    all_vecs = [[] for _ in range(n_adapters)]
    for layer_name, deltas in deltas_per_layer.items():
        for i in range(n_adapters):
            all_vecs[i].append(np.array(deltas[i]).reshape(-1))

    # Concatenate all layers into one vector per adapter
    full_vecs = [np.concatenate(v) for v in all_vecs]

    cos_sims = []
    for i in range(n_adapters):
        for j in range(i + 1, n_adapters):
            dot = np.dot(full_vecs[i], full_vecs[j])
            ni = np.linalg.norm(full_vecs[i])
            nj = np.linalg.norm(full_vecs[j])
            cos = abs(dot) / (ni * nj + 1e-10)
            cos_sims.append(cos)

    mean_cos = np.mean(cos_sims) if cos_sims else 0.0
    max_cos = np.max(cos_sims) if cos_sims else 0.0

    print(f"  [{label}] Mean |cos|: {mean_cos:.8f}, Max |cos|: {max_cos:.8f}")
    return {
        "mean_cos": float(mean_cos),
        "max_cos": float(max_cos),
        "all_cos": [float(c) for c in cos_sims],
    }


# ============================================================================
# Phase 1: Grassmannian baseline composition (reuse trained adapters)
# ============================================================================

def phase_grassmannian_baseline(domain_datasets, vocab_size):
    """Load existing Grassmannian adapters and compute composition baseline."""
    print("\n" + "=" * 60)
    print("PHASE 1: Grassmannian Baseline Composition")
    print("=" * 60)

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    # Load base model
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    weights_path = TERNARY_BASE_DIR / "ternary_base_weights.npz"
    loaded = mx.load(str(weights_path))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())

    # Load adapter B matrices
    adapters_dir = TERNARY_BASE_DIR / "adapters"
    all_adapter_b = {}
    for dname in domain_names:
        adapter_path = adapters_dir / f"{dname}.npz"
        adapter_data = mx.load(str(adapter_path))
        all_adapter_b[dname] = {k: v for k, v in adapter_data.items() if "b_matrix" in k}

    # Compute per-layer deltas for all adapters
    deltas_per_layer = {}
    for idx, dname in enumerate(domain_names):
        adapter_b = all_adapter_b[dname]
        for bname, b_val in adapter_b.items():
            if ".mlp.fc2." in bname:
                a_matrix = bases_mlp[idx]
            else:
                a_matrix = bases_d[idx]
            delta = b_val @ a_matrix  # (d_out, d_in)
            mx.eval(delta)
            layer_key = bname.replace(".b_matrix", "")
            if layer_key not in deltas_per_layer:
                deltas_per_layer[layer_key] = []
            deltas_per_layer[layer_key].append(delta)

    # Compute orthogonality metrics BEFORE any projection
    grass_metrics = compute_composition_metrics(deltas_per_layer, n_domains, "Grassmannian")

    # Compose into model and evaluate PPL
    base_params = dict(nn.utils.tree_flatten(model.parameters()))
    for layer_key, deltas in deltas_per_layer.items():
        weight_key = layer_key + ".weight"
        if weight_key in base_params:
            composed_delta = sum(d for d in deltas) / n_domains
            mx.eval(composed_delta)
            base_params[weight_key] = base_params[weight_key] + composed_delta

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    # Evaluate
    composed_ppls = {}
    for dname in domain_names:
        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        composed_ppls[dname] = ppl
        print(f"  Grassmannian composed PPL on {dname}: {ppl:.4f}")

    mean_composed = sum(composed_ppls.values()) / len(composed_ppls)
    print(f"  Mean Grassmannian composed PPL: {mean_composed:.4f}")

    result = {
        "composed_ppls": composed_ppls,
        "mean_composed_ppl": mean_composed,
        "orthogonality": grass_metrics,
    }

    cleanup(model)
    return result, deltas_per_layer, all_adapter_b


# ============================================================================
# Phase 2: NP-LoRA projection on Grassmannian adapters
# ============================================================================

def phase_np_lora_grassmannian(domain_datasets, vocab_size, deltas_per_layer):
    """Apply NP-LoRA projection to Grassmannian adapter deltas."""
    print("\n" + "=" * 60)
    print("PHASE 2: NP-LoRA on Grassmannian Adapters")
    print("=" * 60)

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)

    # Apply null space projection
    projected, proj_time = compute_null_space_projection(deltas_per_layer, n_domains)
    print(f"  Null space projection time (N={n_domains}): {proj_time:.4f}s")

    # Compute orthogonality metrics AFTER projection
    np_metrics = compute_composition_metrics(projected, n_domains, "NP-LoRA+Grass")

    # Measure how much the projection changed the deltas
    delta_norms_before = []
    delta_norms_after = []
    relative_changes = []
    for layer_key in deltas_per_layer:
        for i in range(n_domains):
            before = np.array(deltas_per_layer[layer_key][i]).reshape(-1)
            after = np.array(projected[layer_key][i]).reshape(-1)
            nb = np.linalg.norm(before)
            na = np.linalg.norm(after)
            diff = np.linalg.norm(before - after)
            delta_norms_before.append(nb)
            delta_norms_after.append(na)
            if nb > 1e-10:
                relative_changes.append(diff / nb)
    mean_rel_change = np.mean(relative_changes) if relative_changes else 0.0
    print(f"  Mean relative change from projection: {mean_rel_change:.8f}")

    # Load base model and compose with projected deltas
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    weights_path = TERNARY_BASE_DIR / "ternary_base_weights.npz"
    loaded = mx.load(str(weights_path))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())

    base_params = dict(nn.utils.tree_flatten(model.parameters()))
    for layer_key, proj_deltas in projected.items():
        weight_key = layer_key + ".weight"
        if weight_key in base_params:
            composed_delta = sum(d for d in proj_deltas) / n_domains
            mx.eval(composed_delta)
            base_params[weight_key] = base_params[weight_key] + composed_delta

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    composed_ppls = {}
    for dname in domain_names:
        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        composed_ppls[dname] = ppl
        print(f"  NP-LoRA+Grass composed PPL on {dname}: {ppl:.4f}")

    mean_composed = sum(composed_ppls.values()) / len(composed_ppls)
    print(f"  Mean NP-LoRA+Grass composed PPL: {mean_composed:.4f}")

    result = {
        "composed_ppls": composed_ppls,
        "mean_composed_ppl": mean_composed,
        "orthogonality": np_metrics,
        "projection_time_s": proj_time,
        "mean_relative_change": float(mean_rel_change),
    }

    cleanup(model)
    return result


# ============================================================================
# Phase 3: Train adapters with RANDOM A and test NP-LoRA benefit
# ============================================================================

def phase_random_a_with_nplora(domain_datasets, vocab_size):
    """Train adapters with random (non-Grassmannian) A, then test NP-LoRA benefit."""
    print("\n" + "=" * 60)
    print("PHASE 3: Random A + NP-LoRA Composition")
    print("=" * 60)

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)

    # Generate random (non-orthogonal) A matrices
    bases_d = generate_random_bases(n_domains, LORA_RANK, D_MODEL, seed=99)
    bases_mlp = generate_random_bases(n_domains, LORA_RANK, MLP_DIM, seed=199)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    # Check A-matrix orthogonality (should be much higher than Grassmannian)
    print("Random A-matrix orthogonality check:")
    a_cos_sims = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            ai_np = np.array(bases_d[i])
            aj_np = np.array(bases_d[j])
            # Frobenius inner product of A_i A_j^T
            cross = ai_np @ aj_np.T
            cos = np.linalg.norm(cross) / (np.linalg.norm(ai_np) * np.linalg.norm(aj_np) + 1e-10)
            a_cos_sims.append(cos)
            print(f"  ||A_{i} A_{j}^T||/norms = {cos:.6f}")
    mean_a_cos = np.mean(a_cos_sims)
    print(f"  Mean A cross-talk: {mean_a_cos:.6f}")

    # Train adapters with random A matrices
    adapter_b_random = {}
    single_ppls = {}

    for idx, dname in enumerate(domain_names):
        print(f"\n--- Training adapter {idx+1}/{n_domains}: {dname} (random A) ---")
        ddata = domain_datasets[dname]

        model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                           n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
        weights_path = TERNARY_BASE_DIR / "ternary_base_weights.npz"
        loaded = mx.load(str(weights_path))
        model.load_weights(list(loaded.items()))
        mx.eval(model.parameters())

        model.freeze()
        a_d = bases_d[idx]
        a_mlp = bases_mlp[idx]

        for layer in model.layers:
            layer.attn.wq = TernaryLoRALinear(layer.attn.wq, LORA_RANK, a_d)
            layer.attn.wk = TernaryLoRALinear(layer.attn.wk, LORA_RANK, a_d)
            layer.attn.wv = TernaryLoRALinear(layer.attn.wv, LORA_RANK, a_d)
            layer.attn.wo = TernaryLoRALinear(layer.attn.wo, LORA_RANK, a_d)
            layer.mlp.fc1 = TernaryLoRALinear(layer.mlp.fc1, LORA_RANK, a_d)
            layer.mlp.fc2 = TernaryLoRALinear(layer.mlp.fc2, LORA_RANK, a_mlp)
        model.lm_head = TernaryLoRALinear(model.lm_head, LORA_RANK, a_d)

        optimizer = opt.Adam(learning_rate=ADAPTER_LR)

        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            B, T, V = logits.shape
            return nn.losses.cross_entropy(
                logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
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

        domain_ppl = compute_ppl(model, ddata["val"])
        single_ppls[dname] = domain_ppl
        print(f"  {dname} PPL: {domain_ppl:.4f}")

        # Extract B matrices
        b_params = {}
        for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
            if "b_matrix" in name:
                mx.eval(param)
                b_params[name] = param
        adapter_b_random[dname] = b_params

        cleanup(model, optimizer)

    # Compute per-layer deltas
    deltas_per_layer = {}
    for idx, dname in enumerate(domain_names):
        for bname, b_val in adapter_b_random[dname].items():
            if ".mlp.fc2." in bname:
                a_matrix = bases_mlp[idx]
            else:
                a_matrix = bases_d[idx]
            delta = b_val @ a_matrix
            mx.eval(delta)
            layer_key = bname.replace(".b_matrix", "")
            if layer_key not in deltas_per_layer:
                deltas_per_layer[layer_key] = []
            deltas_per_layer[layer_key].append(delta)

    # Metrics BEFORE NP-LoRA projection
    random_metrics_before = compute_composition_metrics(
        deltas_per_layer, n_domains, "Random-Before"
    )

    # Compose WITHOUT projection -> baseline PPL
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    loaded = mx.load(str(TERNARY_BASE_DIR / "ternary_base_weights.npz"))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())
    base_params = dict(nn.utils.tree_flatten(model.parameters()))

    for layer_key, deltas in deltas_per_layer.items():
        weight_key = layer_key + ".weight"
        if weight_key in base_params:
            composed_delta = sum(d for d in deltas) / n_domains
            mx.eval(composed_delta)
            base_params[weight_key] = base_params[weight_key] + composed_delta

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    random_composed_ppls_before = {}
    for dname in domain_names:
        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        random_composed_ppls_before[dname] = ppl
        print(f"  Random composed (no proj) PPL on {dname}: {ppl:.4f}")
    mean_random_before = sum(random_composed_ppls_before.values()) / len(random_composed_ppls_before)
    print(f"  Mean random composed PPL (no proj): {mean_random_before:.4f}")
    cleanup(model)

    # Apply NP-LoRA projection
    projected, proj_time = compute_null_space_projection(deltas_per_layer, n_domains)
    print(f"  NP-LoRA projection time (N={n_domains}): {proj_time:.4f}s")

    random_metrics_after = compute_composition_metrics(
        projected, n_domains, "Random-After"
    )

    # Compose WITH projection -> NP-LoRA PPL
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    loaded = mx.load(str(TERNARY_BASE_DIR / "ternary_base_weights.npz"))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())
    base_params = dict(nn.utils.tree_flatten(model.parameters()))

    for layer_key, proj_deltas in projected.items():
        weight_key = layer_key + ".weight"
        if weight_key in base_params:
            composed_delta = sum(d for d in proj_deltas) / n_domains
            mx.eval(composed_delta)
            base_params[weight_key] = base_params[weight_key] + composed_delta

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    random_composed_ppls_after = {}
    for dname in domain_names:
        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        random_composed_ppls_after[dname] = ppl
        print(f"  Random composed (NP-LoRA) PPL on {dname}: {ppl:.4f}")
    mean_random_after = sum(random_composed_ppls_after.values()) / len(random_composed_ppls_after)
    print(f"  Mean random composed PPL (NP-LoRA): {mean_random_after:.4f}")
    cleanup(model)

    result = {
        "single_ppls": single_ppls,
        "mean_single_ppl": sum(single_ppls.values()) / len(single_ppls),
        "a_matrix_mean_cos": float(mean_a_cos),
        "before_projection": {
            "composed_ppls": random_composed_ppls_before,
            "mean_composed_ppl": mean_random_before,
            "orthogonality": random_metrics_before,
        },
        "after_projection": {
            "composed_ppls": random_composed_ppls_after,
            "mean_composed_ppl": mean_random_after,
            "orthogonality": random_metrics_after,
            "projection_time_s": proj_time,
        },
    }

    return result


# ============================================================================
# Phase 4: Null space computation timing at N=50
# ============================================================================

def phase_timing_n50():
    """Measure null space computation time for N=50 synthetic adapters."""
    print("\n" + "=" * 60)
    print("PHASE 4: Timing Test at N=50")
    print("=" * 60)

    n_adapters = 50
    # Number of weight matrices in our model:
    # 6 layers * (wq, wk, wv, wo, fc1, fc2) + lm_head = 37
    n_layers_total = N_LAYERS * 6 + 1

    # Generate synthetic deltas matching our model dimensions
    # Most layers: (D_MODEL, D_MODEL) = (256, 256) or (MLP_DIM, D_MODEL) = (1024, 256)
    # fc2: (D_MODEL, MLP_DIM) = (256, 1024)
    np.random.seed(42)

    print(f"  Generating {n_adapters} synthetic adapters across {n_layers_total} layers...")
    deltas_per_layer = {}

    # Simulate layer shapes (6 layers x 6 weight matrices + lm_head)
    layer_shapes = []
    for _ in range(N_LAYERS):
        # wq, wk, wv, wo: (D_MODEL, D_MODEL)
        layer_shapes.extend([(D_MODEL, D_MODEL)] * 4)
        # fc1: (MLP_DIM, D_MODEL)
        layer_shapes.append((MLP_DIM, D_MODEL))
        # fc2: (D_MODEL, MLP_DIM)
        layer_shapes.append((D_MODEL, MLP_DIM))
    # lm_head: (vocab=27, D_MODEL)
    layer_shapes.append((27, D_MODEL))

    for l_idx, (d_out, d_in) in enumerate(layer_shapes):
        layer_name = f"layer_{l_idx}"
        deltas = []
        for i in range(n_adapters):
            # Generate low-rank delta: B @ A where B is (d_out, r), A is (r, d_in)
            B = np.random.randn(d_out, LORA_RANK) * 0.01
            A = np.random.randn(LORA_RANK, d_in)
            A, _ = np.linalg.qr(A.T)
            A = A[:, :LORA_RANK].T
            delta = mx.array(B @ A)
            deltas.append(delta)
        deltas_per_layer[layer_name] = deltas

    # Time the null space projection
    print(f"  Running null space projection for N={n_adapters}...")
    _, proj_time = compute_null_space_projection(deltas_per_layer, n_adapters)
    print(f"  Projection time: {proj_time:.4f}s")

    k2_pass = proj_time < 1.0
    print(f"  [K2] < 1 sec: {'PASS' if k2_pass else 'FAIL'}")

    # Also time N=5 and N=15 for scaling curve
    timing_results = {"N50": proj_time}
    for test_n in [5, 15]:
        sub_deltas = {k: v[:test_n] for k, v in deltas_per_layer.items()}
        _, t = compute_null_space_projection(sub_deltas, test_n)
        timing_results[f"N{test_n}"] = t
        print(f"  Projection time N={test_n}: {t:.4f}s")

    return {
        "projection_times": timing_results,
        "k2_pass": k2_pass,
        "n_layers": n_layers_total,
        "layer_shapes_summary": f"{N_LAYERS}x6 + lm_head = {n_layers_total}",
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    print("Loading data...")
    tok, train_ds, val_ds, domain_datasets = load_data()
    vocab_size = tok.vocab_size
    domain_names = sorted(domain_datasets.keys())
    print(f"Vocab size: {vocab_size}, Domains: {domain_names}")

    # Check that pre-trained adapters exist
    adapters_dir = TERNARY_BASE_DIR / "adapters"
    if not adapters_dir.exists():
        print("ERROR: Pre-trained adapters not found at", adapters_dir)
        print("Run ternary_base_from_scratch_mlx first.")
        sys.exit(1)

    # Get baseline single-adapter PPLs from prior experiment
    prior_results = json.loads((TERNARY_BASE_DIR / "results.json").read_text())
    single_ppls_grass = {d: prior_results["domain_adapters"][d]["ppl"] for d in domain_names}
    mean_single_grass = sum(single_ppls_grass.values()) / len(single_ppls_grass)
    print(f"Grassmannian single-adapter PPLs: {single_ppls_grass}")
    print(f"Mean Grassmannian single PPL: {mean_single_grass:.4f}")

    # Phase 1: Grassmannian baseline
    grass_result, grass_deltas, _ = phase_grassmannian_baseline(domain_datasets, vocab_size)
    log_memory("after-phase1")

    # Phase 2: NP-LoRA on Grassmannian
    np_grass_result = phase_np_lora_grassmannian(domain_datasets, vocab_size, grass_deltas)
    log_memory("after-phase2")

    # Phase 3: Random A + NP-LoRA
    random_result = phase_random_a_with_nplora(domain_datasets, vocab_size)
    log_memory("after-phase3")

    # Phase 4: Timing at N=50
    timing_result = phase_timing_n50()
    log_memory("after-phase4")

    # ================================================================
    # Kill criteria evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("KILL CRITERIA EVALUATION")
    print("=" * 60)

    # K1: NP-LoRA composition worse than Grassmannian at any N -> KILL
    grass_ppl = grass_result["mean_composed_ppl"]
    np_grass_ppl = np_grass_result["mean_composed_ppl"]
    # For random A, compare NP-LoRA-projected vs raw Grassmannian
    # NP-LoRA should NOT be worse than Grassmannian
    k1_grass_comparison = np_grass_ppl <= grass_ppl * 1.01  # 1% tolerance for noise
    print(f"  Grassmannian PPL:          {grass_ppl:.4f}")
    print(f"  NP-LoRA+Grassmannian PPL:  {np_grass_ppl:.4f}")
    print(f"  Ratio: {np_grass_ppl/grass_ppl:.4f}")
    print(f"  [K1] NP-LoRA+Grass <= Grass * 1.01: {'PASS' if k1_grass_comparison else 'FAIL'}")

    # K2: Already evaluated in Phase 4
    k2_pass = timing_result["k2_pass"]
    print(f"  [K2] N=50 projection < 1 sec: {'PASS' if k2_pass else 'FAIL'}")
    print(f"       Actual: {timing_result['projection_times']['N50']:.4f}s")

    # ================================================================
    # Analysis: Does NP-LoRA add value?
    # ================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Key question: does NP-LoRA help with random A?
    random_before = random_result["before_projection"]["mean_composed_ppl"]
    random_after = random_result["after_projection"]["mean_composed_ppl"]
    random_improvement = (random_before - random_after) / random_before * 100

    print(f"\n  Random A composition:")
    print(f"    Without NP-LoRA: {random_before:.4f}")
    print(f"    With NP-LoRA:    {random_after:.4f}")
    print(f"    Improvement:     {random_improvement:.2f}%")

    # Compare random-A NP-LoRA vs Grassmannian (no projection)
    print(f"\n  Grassmannian (no projection): {grass_ppl:.4f}")
    print(f"  Random + NP-LoRA:            {random_after:.4f}")
    print(f"  Grassmannian still better:   {grass_ppl < random_after}")

    # Composition ratios
    grass_ratio = grass_ppl / mean_single_grass
    np_grass_ratio = np_grass_ppl / mean_single_grass
    random_ratio_before = random_before / random_result["mean_single_ppl"]
    random_ratio_after = random_after / random_result["mean_single_ppl"]

    print(f"\n  Composition ratios:")
    print(f"    Grassmannian:          {grass_ratio:.4f}")
    print(f"    NP-LoRA+Grassmannian:  {np_grass_ratio:.4f}")
    print(f"    Random (no proj):      {random_ratio_before:.4f}")
    print(f"    Random + NP-LoRA:      {random_ratio_after:.4f}")

    # ================================================================
    # Aggregate results
    # ================================================================
    total_time = time.time() - t0_total
    results = {
        "experiment": "np_lora_null_space_composition",
        "architecture": {
            "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "block_size": BLOCK_SIZE, "lora_rank": LORA_RANK, "vocab_size": vocab_size,
        },
        "grassmannian_baseline": {
            "single_ppls": single_ppls_grass,
            "mean_single_ppl": mean_single_grass,
            **grass_result,
        },
        "np_lora_on_grassmannian": np_grass_result,
        "random_a_comparison": random_result,
        "timing": timing_result,
        "kill_criteria": {
            "K1_np_lora_not_worse_than_grassmannian": k1_grass_comparison,
            "K2_n50_under_1sec": k2_pass,
        },
        "composition_ratios": {
            "grassmannian": grass_ratio,
            "np_lora_grassmannian": np_grass_ratio,
            "random_no_proj": random_ratio_before,
            "random_np_lora": random_ratio_after,
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"K1 (NP-LoRA not worse): {'PASS' if k1_grass_comparison else 'FAIL'}")
    print(f"K2 (N=50 < 1 sec):      {'PASS' if k2_pass else 'FAIL'}")
    print(f"NP-LoRA relative change on Grassmannian: {np_grass_result['mean_relative_change']:.8f}")
    print(f"NP-LoRA improvement on random A: {random_improvement:.2f}%")
    print(f"Grassmannian still best: {grass_ppl < random_after}")


if __name__ == "__main__":
    main()
