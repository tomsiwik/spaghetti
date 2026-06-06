#!/usr/bin/env python3
"""GaLore + STE Integration: Fix ternary quantization degradation.

Prior GaLore experiment showed 2.6x ternary PPL degradation when quantizing
post-hoc. This experiment integrates STE into the GaLore training loop so
ternary quantization happens DURING training.

Kill criteria:
  K1 (id=186): GaLore+STE ternary PPL > 1.5x standard-STE ternary PPL -> KILL
  K2 (id=187): Adapter composition ratio > 1.5x standard-STE composition ratio -> KILL
  K3 (id=188): Training time > 3x standard STE time -> KILL

Success criteria:
  S1: GaLore+STE ternary PPL within 1.2x of standard STE
  S2: Memory usage during training < 60% of standard STE

Baseline from exp_ternary_base_from_scratch_mlx:
  FP32 PPL: 1.5895, Ternary STE PPL: 1.5935, composition ratio: 1.022
  Training time: 53s (ternary base), 222s total

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
from mlx.utils import tree_flatten, tree_unflatten

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"

# Architecture hyperparams (match ternary_base_from_scratch_mlx exactly)
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 64
BLOCK_SIZE = 32
MLP_DIM = 4 * D_MODEL  # 1024
LORA_RANK = 8

# Training hyperparams (match baseline)
BASE_STEPS = 4000
BASE_LR = 3e-4
BATCH_SIZE = 64

# GaLore hyperparams
GALORE_RANK = 64
GALORE_UPDATE_FREQ = 200

# Adapter hyperparams (match baseline)
ADAPTER_STEPS = 1500
ADAPTER_LR = 1e-3

# Baseline results (from exp_ternary_base_from_scratch_mlx)
BASELINE_TERNARY_PPL = 1.5935
BASELINE_COMPOSITION_RATIO = 1.022
BASELINE_TRAIN_TIME = 53.0  # seconds for ternary base training


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
# BitLinear: Ternary weights with STE (same as ternary_base_from_scratch_mlx)
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE in forward pass."""
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


# ============================================================================
# Transformer Architecture (same as ternary_base_from_scratch_mlx)
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
    """Ternary GPT: all Linear layers replaced with BitLinear (STE quantization)."""
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


# ============================================================================
# GaLore Optimizer (adapted from bitnet_galore_scaffold)
# ============================================================================

class GaLoreProjection:
    """SVD-based gradient projection for a single weight matrix."""
    def __init__(self, shape, rank, update_freq):
        self.rank = min(rank, min(shape))
        self.update_freq = update_freq
        self.step = 0
        self.P = None

    def project(self, grad):
        if self.step % self.update_freq == 0 or self.P is None:
            self._update_projection(grad)
        self.step += 1
        return self.P.T @ grad

    def unproject(self, grad_proj):
        return self.P @ grad_proj

    def _update_projection(self, grad):
        U, S, Vt = mx.linalg.svd(grad, stream=mx.cpu)
        mx.eval(U, S, Vt)
        self.P = U[:, :self.rank]
        mx.eval(self.P)


class GaLoreAdamState:
    """Adam optimizer state in projected (low-rank) space."""
    def __init__(self, shape_proj, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = mx.zeros(shape_proj)
        self.v = mx.zeros(shape_proj)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, grad_proj, lr):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_proj
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_proj ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update_proj = -lr * m_hat / (mx.sqrt(v_hat) + self.eps)
        mx.eval(self.m, self.v)
        return update_proj


class GaLoreSTEOptimizer:
    """GaLore optimizer for BitLinear (STE) models.

    Applies low-rank gradient projection to large weight matrices (BitLinear.weight).
    Standard Adam for small params (embeddings, norms).

    The key difference from vanilla GaLore: gradients come through STE quantization,
    so they are quantization-aware. We project these STE gradients to low-rank space.
    """
    def __init__(self, model, lr, galore_rank, galore_update_freq,
                 min_dim_for_galore=128):
        self.lr = lr
        self.galore_rank = galore_rank
        self.galore_update_freq = galore_update_freq
        self.min_dim = min_dim_for_galore

        self.galore_projections = {}
        self.galore_states = {}
        self.standard_adam = opt.Adam(learning_rate=lr)

        self.galore_param_names = set()
        for name, p in tree_flatten(model.parameters()):
            if len(p.shape) == 2 and min(p.shape) >= self.min_dim:
                self.galore_param_names.add(name)
                proj = GaLoreProjection(p.shape, galore_rank, galore_update_freq)
                self.galore_projections[name] = proj

        n_galore = len(self.galore_param_names)
        n_total = len(list(tree_flatten(model.parameters())))
        print(f"  GaLore+STE optimizer: {n_galore}/{n_total} params use GaLore "
              f"(rank={galore_rank}, update_freq={galore_update_freq})")

    def update(self, model, grads):
        flat_grads = tree_flatten(grads)

        galore_updates = {}
        standard_grads = {}

        for name, g in flat_grads:
            if name in self.galore_param_names:
                proj = self.galore_projections[name]
                g_proj = proj.project(g)
                if name not in self.galore_states:
                    self.galore_states[name] = GaLoreAdamState(g_proj.shape)
                update_proj = self.galore_states[name].step(g_proj, self.lr)
                update_full = proj.unproject(update_proj)
                galore_updates[name] = update_full
            else:
                standard_grads[name] = g

        if galore_updates:
            current_params = dict(tree_flatten(model.parameters()))
            updated = {}
            for name, delta in galore_updates.items():
                updated[name] = current_params[name] + delta
            model.update(tree_unflatten(list(updated.items())))

        if standard_grads:
            standard_grads_tree = tree_unflatten(list(standard_grads.items()))
            self.standard_adam.update(model, standard_grads_tree)

        mx.eval(model.parameters())
        if hasattr(self.standard_adam, 'state'):
            mx.eval(self.standard_adam.state)

    def get_memory_stats(self):
        """Compute optimizer state memory usage."""
        galore_state_elements = 0
        projection_elements = 0
        for name in self.galore_param_names:
            if name in self.galore_states:
                state = self.galore_states[name]
                galore_state_elements += state.m.size + state.v.size
            if name in self.galore_projections:
                proj = self.galore_projections[name]
                if proj.P is not None:
                    projection_elements += proj.P.size
        return {
            "galore_state_elements": galore_state_elements,
            "projection_elements": projection_elements,
            "total_galore_elements": galore_state_elements + projection_elements,
        }


# ============================================================================
# Ternary LoRA (same as ternary_base_from_scratch_mlx)
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA adapter on frozen BitLinear with ternary B via STE."""
    def __init__(self, base_linear: BitLinear, rank: int, a_matrix: mx.array):
        super().__init__()
        self.base = base_linear
        self.base.freeze()
        out_features = base_linear.weight.shape[0]
        self.a_matrix = mx.stop_gradient(a_matrix)
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
    """Generate n_adapters orthogonal A matrices via QR decomposition."""
    mx.random.seed(seed)
    total_rank = n_adapters * rank
    assert total_rank <= dim, f"Need {total_rank} orthogonal vectors but dim={dim}"
    random_mat = mx.random.normal(shape=(dim, total_rank))
    mx.eval(random_mat)
    import numpy as np
    Q, _ = np.linalg.qr(np.array(random_mat))
    Q = mx.array(Q[:, :total_rank])
    bases = []
    for i in range(n_adapters):
        start = i * rank
        end = start + rank
        A_i = Q[:, start:end].T
        bases.append(A_i)
    return bases


# ============================================================================
# Data loading (reuse from micro/data.py)
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


# ============================================================================
# Training utilities
# ============================================================================

def compute_ppl(model, dataset, n_batches=20, batch_size=64):
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


def get_zero_fraction(model):
    """Compute fraction of weights that are zero in ternary quantization."""
    total = 0
    zero_count = 0
    for name, param in nn.utils.tree_flatten(model.parameters()):
        if "wte" in name or "wpe" in name or "norm" in name:
            continue
        alpha = mx.mean(mx.abs(param))
        w_scaled = param / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1)
        mx.eval(w_q)
        n = w_q.size
        n_zero = int(mx.sum(w_q == 0).item())
        total += n
        zero_count += n_zero
    return zero_count / total if total > 0 else 0


# ============================================================================
# Phase 1: GaLore+STE Base Training
# ============================================================================

def phase_galore_ste_base(train_ds, val_ds, vocab_size):
    """Train ternary base model with GaLore+STE."""
    print("\n" + "="*60)
    print("PHASE 1: GaLore+STE Base Training")
    print("="*60)

    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    log_memory("galore-ste-init")

    # GaLore+STE optimizer
    optimizer = GaLoreSTEOptimizer(
        model, lr=BASE_LR,
        galore_rank=GALORE_RANK,
        galore_update_freq=GALORE_UPDATE_FREQ
    )

    # Also train a standard STE baseline for direct comparison
    # (with standard Adam, matching the ternary_base_from_scratch_mlx experiment)
    model_std = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(model_std.parameters())
    optimizer_std = opt.Adam(learning_rate=BASE_LR)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss_and_grad_std = nn.value_and_grad(model_std, loss_fn)
    rng = random.Random(42)
    rng_std = random.Random(42)  # same seed for fair comparison

    # Track memory at peak
    mx.reset_peak_memory()

    # --- Train GaLore+STE ---
    print("\n--- GaLore+STE Training ---")
    gc.disable()
    losses_galore = []
    t0 = time.time()
    for step in range(1, BASE_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        # Note: optimizer.update already calls mx.eval on model params
        mx.eval(loss)
        loss_val = loss.item()
        losses_galore.append(loss_val)

        if step % 500 == 0 or step == BASE_STEPS:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{BASE_STEPS} | loss {loss_val:.4f} | "
                  f"time {elapsed:.1f}s")
    gc.enable()
    gc.collect()

    galore_train_time = time.time() - t0
    galore_peak_mem = mx.get_peak_memory() / 1e9
    galore_ppl = compute_ppl(model, val_ds)
    galore_zero_frac = get_zero_fraction(model)

    # Get GaLore optimizer memory stats
    galore_mem_stats = optimizer.get_memory_stats()

    print(f"\nGaLore+STE PPL: {galore_ppl:.4f}")
    print(f"GaLore+STE training time: {galore_train_time:.1f}s")
    print(f"GaLore+STE peak memory: {galore_peak_mem:.2f}GB")
    print(f"GaLore+STE zero fraction: {galore_zero_frac:.4f}")
    log_memory("galore-ste-done")

    # Save GaLore+STE model
    weights_path = EXPERIMENT_DIR / "galore_ste_weights.npz"
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(str(weights_path), **flat_params)

    # Cleanup GaLore model before training standard
    del optimizer  # free GaLore state
    cleanup(model)
    mx.reset_peak_memory()

    # --- Train Standard STE (reproduction of baseline) ---
    print("\n--- Standard STE Training (reproduction) ---")
    gc.disable()
    losses_std = []
    t0_std = time.time()
    for step in range(1, BASE_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng_std)
        loss, grads = loss_and_grad_std(model_std, inputs, targets)
        optimizer_std.update(model_std, grads)
        mx.eval(model_std.parameters(), optimizer_std.state, loss)
        loss_val = loss.item()
        losses_std.append(loss_val)

        if step % 500 == 0 or step == BASE_STEPS:
            elapsed = time.time() - t0_std
            print(f"  step {step:4d}/{BASE_STEPS} | loss {loss_val:.4f} | "
                  f"time {elapsed:.1f}s")
    gc.enable()
    gc.collect()

    std_train_time = time.time() - t0_std
    std_peak_mem = mx.get_peak_memory() / 1e9
    std_ppl = compute_ppl(model_std, val_ds)
    std_zero_frac = get_zero_fraction(model_std)

    # Compute standard optimizer state memory
    std_state_elements = 0
    if hasattr(optimizer_std, 'state'):
        for item in tree_flatten(optimizer_std.state):
            if len(item) == 2:
                std_state_elements += item[1].size

    print(f"\nStandard STE PPL: {std_ppl:.4f}")
    print(f"Standard STE training time: {std_train_time:.1f}s")
    print(f"Standard STE peak memory: {std_peak_mem:.2f}GB")
    print(f"Standard STE zero fraction: {std_zero_frac:.4f}")

    # Save standard model
    std_weights_path = EXPERIMENT_DIR / "std_ste_weights.npz"
    std_flat = dict(nn.utils.tree_flatten(model_std.parameters()))
    mx.savez(str(std_weights_path), **std_flat)

    # K1 check: GaLore+STE PPL vs standard STE PPL
    ppl_ratio = galore_ppl / std_ppl
    k1_pass = ppl_ratio <= 1.5
    print(f"\n[K1] GaLore+STE PPL / Std STE PPL = {ppl_ratio:.4f} (threshold: 1.5)")
    print(f"[K1] {'PASS' if k1_pass else 'FAIL'}")

    # K3 check: training time
    time_ratio = galore_train_time / std_train_time
    k3_pass = time_ratio <= 3.0
    print(f"[K3] Time ratio: {time_ratio:.2f}x (threshold: 3.0x)")
    print(f"[K3] {'PASS' if k3_pass else 'FAIL'}")

    # S1 check
    s1_pass = ppl_ratio <= 1.2
    print(f"[S1] PPL ratio <= 1.2: {ppl_ratio:.4f} {'PASS' if s1_pass else 'FAIL'}")

    # S2 check: memory comparison
    mem_ratio = galore_peak_mem / std_peak_mem if std_peak_mem > 0 else float('inf')
    s2_pass = mem_ratio < 0.6
    print(f"[S2] Memory ratio: {mem_ratio:.3f} (threshold: < 0.6)")
    print(f"[S2] {'PASS' if s2_pass else 'FAIL'}")

    result = {
        "galore_ste": {
            "ppl": galore_ppl,
            "final_loss": losses_galore[-1],
            "train_time_s": round(galore_train_time, 1),
            "peak_memory_gb": round(galore_peak_mem, 3),
            "zero_fraction": round(galore_zero_frac, 4),
            "loss_history_500": [losses_galore[i-1] for i in range(500, BASE_STEPS+1, 500)],
            "optimizer_state_elements": galore_mem_stats["total_galore_elements"],
        },
        "standard_ste": {
            "ppl": std_ppl,
            "final_loss": losses_std[-1],
            "train_time_s": round(std_train_time, 1),
            "peak_memory_gb": round(std_peak_mem, 3),
            "zero_fraction": round(std_zero_frac, 4),
            "loss_history_500": [losses_std[i-1] for i in range(500, BASE_STEPS+1, 500)],
            "optimizer_state_elements": std_state_elements,
        },
        "ppl_ratio": round(ppl_ratio, 4),
        "time_ratio": round(time_ratio, 2),
        "mem_ratio": round(mem_ratio, 3),
        "k1_pass": k1_pass,
        "k3_pass": k3_pass,
        "s1_pass": s1_pass,
        "s2_pass": s2_pass,
        "n_params": n_params,
    }

    cleanup(model_std, optimizer_std)
    return result


# ============================================================================
# Phase 2: Train domain adapters on GaLore+STE base
# ============================================================================

def phase_train_adapters(domain_datasets, vocab_size, base_type="galore_ste"):
    """Train 5 domain ternary LoRA adapters on a base model."""
    weights_file = "galore_ste_weights.npz" if base_type == "galore_ste" else "std_ste_weights.npz"
    print(f"\n{'='*60}")
    print(f"PHASE 2: Adapter Training on {base_type} base")
    print(f"{'='*60}")

    ADAPTERS_DIR.mkdir(exist_ok=True)
    subdir = ADAPTERS_DIR / base_type
    subdir.mkdir(exist_ok=True)

    n_domains = len(domain_datasets)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    domain_names = sorted(domain_datasets.keys())
    domain_results = {}

    for idx, dname in enumerate(domain_names):
        print(f"\n--- Adapter {idx+1}/{n_domains}: {dname} ---")
        ddata = domain_datasets[dname]

        model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
        weights_path = EXPERIMENT_DIR / weights_file
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
        domain_ppl = compute_ppl(model, ddata["val"])
        print(f"  {dname} PPL: {domain_ppl:.4f}")

        adapter_params = {}
        for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
            adapter_params[name] = param
        save_path = subdir / f"{dname}.npz"
        mx.savez(str(save_path), **adapter_params)

        domain_results[dname] = {
            "ppl": domain_ppl,
            "train_time_s": round(adapter_time, 1),
            "trainable_params": n_trainable,
        }

        cleanup(model, optimizer)

    return domain_results


# ============================================================================
# Phase 3: Composition test
# ============================================================================

def phase_composition_test(domain_datasets, vocab_size, domain_results,
                           base_type="galore_ste"):
    """Test composition on given base model."""
    weights_file = "galore_ste_weights.npz" if base_type == "galore_ste" else "std_ste_weights.npz"
    print(f"\n{'='*60}")
    print(f"PHASE 3: Composition Test ({base_type})")
    print(f"{'='*60}")

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    weights_path = EXPERIMENT_DIR / weights_file
    loaded = mx.load(str(weights_path))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())

    # Load adapter B matrices
    subdir = ADAPTERS_DIR / base_type
    all_adapter_b = {}
    for dname in domain_names:
        adapter_path = subdir / f"{dname}.npz"
        adapter_data = mx.load(str(adapter_path))
        all_adapter_b[dname] = {k: v for k, v in adapter_data.items() if "b_matrix" in k}

    # Compose: sum (1/N) * B_i @ A_i for each layer
    composed_deltas = {}
    for idx, dname in enumerate(domain_names):
        adapter_b = all_adapter_b[dname]
        for bname, b_val in adapter_b.items():
            if ".mlp.fc2." in bname:
                a_matrix = bases_mlp[idx]
            else:
                a_matrix = bases_d[idx]
            delta = (b_val @ a_matrix) / n_domains
            if bname in composed_deltas:
                composed_deltas[bname] = composed_deltas[bname] + delta
            else:
                composed_deltas[bname] = delta

    # Apply to base weights
    base_params = dict(nn.utils.tree_flatten(model.parameters()))
    for bname, delta in composed_deltas.items():
        base_key = bname.replace(".b_matrix", ".weight")
        if base_key in base_params:
            base_params[base_key] = base_params[base_key] + delta
            mx.eval(base_params[base_key])

    model.load_weights(list(base_params.items()))
    mx.eval(model.parameters())

    # Evaluate composed model
    composed_ppls = {}
    for dname in domain_names:
        ddata = domain_datasets[dname]
        ppl = compute_ppl(model, ddata["val"])
        composed_ppls[dname] = ppl
        print(f"  Composed PPL on {dname}: {ppl:.4f}")

    single_ppls = [domain_results[d]["ppl"] for d in domain_names]
    composed_ppl_vals = [composed_ppls[d] for d in domain_names]
    mean_single = sum(single_ppls) / len(single_ppls)
    mean_composed = sum(composed_ppl_vals) / len(composed_ppl_vals)
    composition_ratio = mean_composed / mean_single

    print(f"\nMean single-adapter PPL: {mean_single:.4f}")
    print(f"Mean composed PPL: {mean_composed:.4f}")
    print(f"Composition ratio: {composition_ratio:.4f}")

    # Orthogonality
    import numpy as np
    cos_sims = []
    for i in range(n_domains):
        for j in range(i+1, n_domains):
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
            del vec_i, vec_j

    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    print(f"Mean |cos|: {mean_cos:.8f}")

    result = {
        "composed_ppls": {k: round(v, 4) for k, v in composed_ppls.items()},
        "mean_single_ppl": round(mean_single, 4),
        "mean_composed_ppl": round(mean_composed, 4),
        "composition_ratio": round(composition_ratio, 4),
        "cos_similarities": cos_sims,
        "mean_cos_similarity": mean_cos,
    }

    cleanup(model)
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
    print(f"Vocab size: {vocab_size}, Train sequences: {len(train_ds)}, "
          f"Val sequences: {len(val_ds)}")
    for dname, ddata in sorted(domain_datasets.items()):
        print(f"  {dname}: train={len(ddata['train'])}, val={len(ddata['val'])}")

    # Phase 1: Train both GaLore+STE and Standard STE
    base_results = phase_galore_ste_base(train_ds, val_ds, vocab_size)
    log_memory("after-base")

    # Early kill check
    if not base_results["k1_pass"]:
        print("\n[EARLY KILL] K1 failed: GaLore+STE PPL too high vs standard STE")
        results = {
            "experiment": "galore_ste_ternary",
            "status": "KILLED",
            "kill_reason": "K1: PPL ratio too high",
            "base_results": base_results,
            "total_time_s": round(time.time() - t0_total, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
        return

    if not base_results["k3_pass"]:
        print("\n[EARLY KILL] K3 failed: Training too slow")
        results = {
            "experiment": "galore_ste_ternary",
            "status": "KILLED",
            "kill_reason": "K3: Training time > 3x baseline",
            "base_results": base_results,
            "total_time_s": round(time.time() - t0_total, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
        return

    # Phase 2: Train adapters on both bases
    galore_adapter_results = phase_train_adapters(domain_datasets, vocab_size, "galore_ste")
    log_memory("after-galore-adapters")

    std_adapter_results = phase_train_adapters(domain_datasets, vocab_size, "standard_ste")
    log_memory("after-std-adapters")

    # Phase 3: Composition on both bases
    galore_comp = phase_composition_test(domain_datasets, vocab_size,
                                          galore_adapter_results, "galore_ste")
    log_memory("after-galore-composition")

    std_comp = phase_composition_test(domain_datasets, vocab_size,
                                       std_adapter_results, "standard_ste")
    log_memory("after-std-composition")

    # K2 check
    comp_ratio_galore = galore_comp["composition_ratio"]
    comp_ratio_std = std_comp["composition_ratio"]
    comp_ratio_ratio = comp_ratio_galore / comp_ratio_std if comp_ratio_std > 0 else float('inf')
    k2_pass = comp_ratio_ratio <= 1.5
    print(f"\n[K2] GaLore composition ratio: {comp_ratio_galore:.4f}")
    print(f"[K2] Standard composition ratio: {comp_ratio_std:.4f}")
    print(f"[K2] Ratio of ratios: {comp_ratio_ratio:.4f} (threshold: 1.5)")
    print(f"[K2] {'PASS' if k2_pass else 'FAIL'}")

    total_time = time.time() - t0_total

    # Aggregate results
    results = {
        "experiment": "galore_ste_ternary",
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "block_size": BLOCK_SIZE,
            "mlp_dim": MLP_DIM,
            "lora_rank": LORA_RANK,
            "galore_rank": GALORE_RANK,
            "galore_update_freq": GALORE_UPDATE_FREQ,
            "vocab_size": vocab_size,
        },
        "base_training": base_results,
        "galore_ste_adapters": galore_adapter_results,
        "standard_ste_adapters": std_adapter_results,
        "galore_ste_composition": galore_comp,
        "standard_ste_composition": std_comp,
        "kill_criteria": {
            "K1_ppl_ratio": base_results["ppl_ratio"],
            "K1_threshold": 1.5,
            "K1_pass": base_results["k1_pass"],
            "K2_comp_ratio_galore": comp_ratio_galore,
            "K2_comp_ratio_std": comp_ratio_std,
            "K2_ratio_of_ratios": round(comp_ratio_ratio, 4),
            "K2_threshold": 1.5,
            "K2_pass": k2_pass,
            "K3_time_ratio": base_results["time_ratio"],
            "K3_threshold": 3.0,
            "K3_pass": base_results["k3_pass"],
        },
        "success_criteria": {
            "S1_ppl_within_1_2x": base_results["s1_pass"],
            "S2_memory_below_60pct": base_results["s2_pass"],
        },
        "comparison_with_baseline": {
            "baseline_ternary_ppl": BASELINE_TERNARY_PPL,
            "galore_ste_ppl": base_results["galore_ste"]["ppl"],
            "standard_ste_ppl_this_run": base_results["standard_ste"]["ppl"],
            "galore_ste_vs_baseline_ratio": round(
                base_results["galore_ste"]["ppl"] / BASELINE_TERNARY_PPL, 4),
        },
        "total_time_s": round(total_time, 1),
        "status": "SUPPORTED" if (base_results["k1_pass"] and k2_pass and base_results["k3_pass"]) else "KILLED",
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GaLore+STE PPL:       {base_results['galore_ste']['ppl']:.4f}")
    print(f"Standard STE PPL:     {base_results['standard_ste']['ppl']:.4f}")
    print(f"PPL ratio:            {base_results['ppl_ratio']}x")
    print(f"Time ratio:           {base_results['time_ratio']}x")
    print(f"Memory ratio:         {base_results['mem_ratio']}x")
    print(f"GaLore comp ratio:    {comp_ratio_galore:.4f}")
    print(f"Standard comp ratio:  {comp_ratio_std:.4f}")
    print(f"Zero fraction (GaLore+STE): {base_results['galore_ste']['zero_fraction']}")
    print(f"Zero fraction (Std STE):    {base_results['standard_ste']['zero_fraction']}")
    print()
    for k, v in results["kill_criteria"].items():
        if k.startswith("K") and k.endswith("pass"):
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
    for k, v in results["success_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nStatus: {results['status']}")


if __name__ == "__main__":
    main()
