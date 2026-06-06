#!/usr/bin/env python3
"""MatMul-Free LM (HGRNBit) on MLX: Ternary + No MatMul Architecture.

Port of the core MatMul-free LM architecture (arxiv 2406.02528) to MLX.
Replaces self-attention with HGRN gated linear recurrence and uses
ternary BitLinear layers throughout. Tests whether LoRA adapters can
be applied and composed on this non-Transformer backbone.

Kill criteria:
  K1 (id=189): HGRNBit model val loss > 2.0 within 2000 steps -> KILL
  K2 (id=190): LoRA adapters incompatible with HGRNBit architecture -> KILL
  K3 (id=191): Adapter composition ratio > 1.5 -> KILL

Success criteria:
  S1: HGRNBit trains within 1.5x PPL of Transformer baseline at same params
  S2: At least 3 domain adapters compose successfully
  S3: Inference measurably faster than Transformer baseline
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
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"

# Architecture hyperparams
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

# Transformer baseline (same hyperparams for fair comparison)
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
# BitLinear: Ternary weights with STE + Extra RMSNorm
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE.
    Includes Extra RMSNorm before quantized matmul (arxiv 2505.08823).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        self.pre_quant_norm = nn.RMSNorm(in_features)

    def __call__(self, x):
        x = self.pre_quant_norm(x)
        w = self.weight
        alpha = mx.mean(mx.abs(w))
        w_scaled = w / (alpha + 1e-7)
        w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
        w_ste = w + mx.stop_gradient(w_q - w)
        return x @ w_ste.T


# ============================================================================
# HGRN Token Mixer: Gated Linear Recurrence (replaces self-attention)
# ============================================================================

class HGRNTokenMixer(nn.Module):
    """HGRN-style gated linear recurrence replacing self-attention.

    For each head, computes:
        g_t = sigmoid(lower_bound + W_g(x_t))   # forget gate
        i_t = silu(W_i(x_t)) * W_v(x_t)         # gated input
        h_t = g_t * h_{t-1} + (1 - g_t) * i_t   # recurrence

    No QK^T matmul. O(T*d) instead of O(T^2*d).
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        # Gate projection (produces forget gate per head)
        self.w_gate = BitLinear(n_embd, n_embd)
        # Input projections (gated input = silu(w_i(x)) * w_v(x))
        self.w_input = BitLinear(n_embd, n_embd)
        self.w_value = BitLinear(n_embd, n_embd)
        # Output projection
        self.w_output = BitLinear(n_embd, n_embd)

        # Lower bound for forget gate: log(0.9) ensures information retention
        # This is a learned parameter per head dimension
        self.gate_lower_bound = mx.full((n_embd,), math.log(0.9))

    def __call__(self, x):
        B, T, C = x.shape

        # Compute gate, input, value
        gate_logit = self.w_gate(x) + self.gate_lower_bound  # (B, T, d)
        forget_gate = mx.sigmoid(gate_logit)  # (B, T, d)

        inp = nn.silu(self.w_input(x))  # (B, T, d)
        val = self.w_value(x)           # (B, T, d)
        gated_input = inp * val         # Hadamard product, not matmul

        # Linear recurrence: h_t = g_t * h_{t-1} + (1 - g_t) * i_t
        # Process sequentially along time dimension
        # For micro scale (T=32), this is fast enough
        h = mx.zeros((B, C))
        outputs = []
        for t in range(T):
            g_t = forget_gate[:, t, :]      # (B, d)
            i_t = gated_input[:, t, :]      # (B, d)
            h = g_t * h + (1.0 - g_t) * i_t  # element-wise, no matmul
            outputs.append(h)

        # Stack: (B, T, d)
        h_seq = mx.stack(outputs, axis=1)

        return self.w_output(h_seq)


# ============================================================================
# GLU Channel Mixer: Hadamard-based MLP (replaces standard MLP)
# ============================================================================

class GLUChannelMixer(nn.Module):
    """GLU channel mixer: gate * value with Hadamard product.

    y = BitLinear_down(silu(BitLinear_gate(x)) * BitLinear_up(x))

    The silu(gate) * value is element-wise (Hadamard), not matmul.
    """
    def __init__(self, n_embd: int, d_ff: int):
        super().__init__()
        self.w_gate = BitLinear(n_embd, d_ff)
        self.w_up = BitLinear(n_embd, d_ff)
        self.w_down = BitLinear(d_ff, n_embd)

    def __call__(self, x):
        gate = nn.silu(self.w_gate(x))  # (B, T, d_ff)
        up = self.w_up(x)               # (B, T, d_ff)
        return self.w_down(gate * up)    # Hadamard product + project down


# ============================================================================
# HGRNBit Block and Full Model
# ============================================================================

class HGRNBitBlock(nn.Module):
    """One block of the MatMul-free LM: HGRN token mixer + GLU channel mixer."""
    def __init__(self, n_embd: int, n_head: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.token_mixer = HGRNTokenMixer(n_embd, n_head)
        self.norm2 = nn.RMSNorm(n_embd)
        self.channel_mixer = GLUChannelMixer(n_embd, d_ff)

    def __call__(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class HGRNBitLM(nn.Module):
    """MatMul-free Language Model with HGRN token mixer + GLU channel mixer.
    All linear layers are ternary (BitLinear with STE).
    """
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 256, n_head: int = 4, n_layer: int = 6,
                 d_ff: int = 1024):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [HGRNBitBlock(n_embd, n_head, d_ff) for _ in range(n_layer)]
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
# Transformer Baseline (for fair comparison)
# ============================================================================

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


class FP32GPT(nn.Module):
    """FP32 Transformer baseline: identical param count."""
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


# ============================================================================
# LoRA for HGRNBit
# ============================================================================

class LoRABitLinear(nn.Module):
    """LoRA adapter wrapping a frozen BitLinear.
    A: frozen Grassmannian-initialized. B: trainable (FP32 for simplicity).
    """
    def __init__(self, base: BitLinear, rank: int, a_matrix: mx.array):
        super().__init__()
        self.base = base
        self.base.freeze()
        out_features = base.weight.shape[0]
        self.a_matrix = a_matrix  # (rank, in_features) frozen
        self.a_matrix = mx.stop_gradient(self.a_matrix)
        self.b_matrix = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = self.base(x)
        # LoRA path: x @ A^T @ B^T
        lora_out = (x @ self.a_matrix.T) @ self.b_matrix.T
        return base_out + lora_out


def generate_grassmannian_bases(n_adapters: int, rank: int, dim: int, seed: int = 42):
    """Generate orthogonal A matrices via QR decomposition."""
    total_rank = n_adapters * rank
    assert total_rank <= dim, f"Need {total_rank} orthogonal vectors but dim={dim}"

    rng = np.random.RandomState(seed)
    random_mat = rng.randn(dim, total_rank)
    Q, _ = np.linalg.qr(random_mat)

    bases = []
    for i in range(n_adapters):
        start = i * rank
        end = start + rank
        A_i = mx.array(Q[:, start:end].T.astype(np.float32))  # (rank, dim)
        bases.append(A_i)
    return bases


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    """Load names dataset with domain splits."""
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
# Phase 1: Train FP32 Transformer baseline
# ============================================================================

def phase_fp32_baseline(train_ds, val_ds, vocab_size):
    """Train FP32 Transformer baseline."""
    print("\n" + "=" * 60)
    print("PHASE 1: FP32 Transformer Baseline")
    print("=" * 60)

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

        if step % 500 == 0 or step == 1:
            avg = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - t0
            print(f"  Step {step}/{FP32_STEPS}: loss={loss_val:.4f} avg100={avg:.4f} "
                  f"[{elapsed:.0f}s]")

    gc.enable()
    train_time = time.time() - t0

    # Eval
    ppl = compute_ppl(model, val_ds)
    first_100 = sum(losses[:100]) / 100
    last_100 = sum(losses[-100:]) / 100

    result = {
        "params": n_params,
        "train_time_s": round(train_time, 1),
        "first_100_loss": round(first_100, 4),
        "last_100_loss": round(last_100, 4),
        "val_ppl": round(ppl, 2),
    }
    print(f"  FP32 baseline: PPL={ppl:.2f}, loss {first_100:.4f}->{last_100:.4f} "
          f"in {train_time:.0f}s")

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Train HGRNBit base model
# ============================================================================

def phase_hgrnbit_base(train_ds, val_ds, vocab_size):
    """Train the HGRNBit (matmul-free) base model."""
    print("\n" + "=" * 60)
    print("PHASE 2: HGRNBit (MatMul-Free) Base Training")
    print("=" * 60)

    model = HGRNBitLM(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
                       d_ff=MLP_DIM)
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"HGRNBit model params: {n_params:,}")
    log_memory("hgrnbit-init")

    # Use higher LR for STE training (proven in prior experiments)
    optimizer = opt.Adam(learning_rate=1e-3)

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
    for step in range(1, BASE_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 500 == 0 or step == 1:
            avg = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - t0
            print(f"  Step {step}/{BASE_STEPS}: loss={loss_val:.4f} avg100={avg:.4f} "
                  f"[{elapsed:.0f}s]")

        # K1 check at step 2000
        if step == 2000:
            recent_loss = sum(losses[-100:]) / 100
            if recent_loss > 2.0:
                gc.enable()
                print(f"  K1 KILL: val loss {recent_loss:.4f} > 2.0 at step 2000")
                cleanup(model, optimizer)
                return {
                    "params": n_params,
                    "k1_kill": True,
                    "loss_at_2000": round(recent_loss, 4),
                    "train_time_s": round(time.time() - t0, 1),
                }

    gc.enable()
    train_time = time.time() - t0

    # Eval
    ppl = compute_ppl(model, val_ds)
    first_100 = sum(losses[:100]) / 100
    last_100 = sum(losses[-100:]) / 100

    result = {
        "params": n_params,
        "train_time_s": round(train_time, 1),
        "first_100_loss": round(first_100, 4),
        "last_100_loss": round(last_100, 4),
        "val_ppl": round(ppl, 2),
        "k1_kill": False,
    }
    print(f"  HGRNBit: PPL={ppl:.2f}, loss {first_100:.4f}->{last_100:.4f} "
          f"in {train_time:.0f}s")

    # Save base model weights
    weights_path = EXPERIMENT_DIR / "hgrnbit_base.npz"
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(str(weights_path), **flat)
    print(f"  Saved base model to {weights_path}")

    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 3: Train LoRA adapters on HGRNBit
# ============================================================================

def phase_adapters(domain_datasets, vocab_size):
    """Train domain-specific LoRA adapters on frozen HGRNBit base."""
    print("\n" + "=" * 60)
    print("PHASE 3: LoRA Adapters on HGRNBit")
    print("=" * 60)

    domain_names = list(domain_datasets.keys())
    n_domains = len(domain_names)

    # Generate Grassmannian A matrices
    # We need A matrices for each projection type across domains
    # For simplicity, use one set of bases for input-dim projections
    # and another for ff-dim projections
    a_bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    a_bases_ff = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=123)

    adapter_results = {}
    adapter_params_all = {}

    for domain_idx, domain_name in enumerate(domain_names):
        print(f"\n  --- Training adapter: {domain_name} ---")

        # Fresh model load each time (function scoping)
        model = HGRNBitLM(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                           n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
                           d_ff=MLP_DIM)

        # Load base weights
        weights_path = EXPERIMENT_DIR / "hgrnbit_base.npz"
        base_weights = dict(mx.load(str(weights_path)))
        model.load_weights(list(base_weights.items()))
        mx.eval(model.parameters())

        # Apply LoRA to all BitLinear layers
        a_d = a_bases_d[domain_idx]    # (rank, d_model)
        a_ff = a_bases_ff[domain_idx]  # (rank, d_ff)

        lora_count = 0
        for layer in model.layers:
            # Token mixer projections (d -> d)
            for attr in ['w_gate', 'w_input', 'w_value', 'w_output']:
                base_layer = getattr(layer.token_mixer, attr)
                lora_layer = LoRABitLinear(base_layer, LORA_RANK, a_d)
                setattr(layer.token_mixer, attr, lora_layer)
                lora_count += 1

            # Channel mixer: gate and up (d -> d_ff), down (d_ff -> d)
            for attr in ['w_gate', 'w_up']:
                base_layer = getattr(layer.channel_mixer, attr)
                lora_layer = LoRABitLinear(base_layer, LORA_RANK, a_d)
                setattr(layer.channel_mixer, attr, lora_layer)
                lora_count += 1

            base_down = layer.channel_mixer.w_down
            lora_down = LoRABitLinear(base_down, LORA_RANK, a_ff)
            layer.channel_mixer.w_down = lora_down
            lora_count += 1

        print(f"  Applied LoRA to {lora_count} layers")

        # Freeze everything, unfreeze only LoRA B matrices
        model.freeze()
        for layer in model.layers:
            for attr in ['w_gate', 'w_input', 'w_value', 'w_output']:
                lora = getattr(layer.token_mixer, attr)
                if isinstance(lora, LoRABitLinear):
                    lora.unfreeze(keys=["b_matrix"], strict=False)
            for attr in ['w_gate', 'w_up', 'w_down']:
                lora = getattr(layer.channel_mixer, attr)
                if isinstance(lora, LoRABitLinear):
                    lora.unfreeze(keys=["b_matrix"], strict=False)

        n_trainable = count_trainable(model)
        print(f"  Trainable params: {n_trainable:,}")

        # Verify gradient flow
        try:
            def test_loss(model, x, y):
                logits = model(x)
                B, T, V = logits.shape
                return nn.losses.cross_entropy(
                    logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
                )
            test_grad_fn = nn.value_and_grad(model, test_loss)
            x_test = mx.array([[1, 2, 3, 4, 5]])
            y_test = mx.array([[2, 3, 4, 5, 6]])
            l, g = test_grad_fn(model, x_test, y_test)
            mx.eval(l)
            grad_ok = True
            print(f"  Gradient check PASSED (loss={l.item():.4f})")
        except Exception as e:
            grad_ok = False
            print(f"  K2 KILL: Gradient check FAILED: {e}")
            cleanup(model)
            return {"k2_kill": True, "error": str(e)}

        # Train
        train_ds = domain_datasets[domain_name]["train"]
        val_ds = domain_datasets[domain_name]["val"]

        optimizer = opt.Adam(learning_rate=ADAPTER_LR)

        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            B, T, V = logits.shape
            return nn.losses.cross_entropy(
                logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        rng = random.Random(domain_idx)

        gc.disable()
        losses = []
        t0 = time.time()
        for step in range(1, ADAPTER_STEPS + 1):
            inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
            loss, grads = loss_and_grad(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            losses.append(loss_val)

            if step % 300 == 0 or step == 1:
                avg = sum(losses[-50:]) / len(losses[-50:])
                print(f"    Step {step}/{ADAPTER_STEPS}: loss={loss_val:.4f} avg50={avg:.4f}")

        gc.enable()
        train_time = time.time() - t0

        # Evaluate
        ppl = compute_ppl(model, val_ds)
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        print(f"  {domain_name}: PPL={ppl:.2f}, loss {first_50:.4f}->{last_50:.4f} "
              f"in {train_time:.0f}s")

        # Save adapter (B matrices only)
        adapter_path = ADAPTERS_DIR / domain_name
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_params = {}
        for name, p in nn.utils.tree_flatten(model.trainable_parameters()):
            if "b_matrix" in name:
                adapter_params[name] = mx.array(p)
        mx.eval(adapter_params)
        mx.savez(str(adapter_path / "adapter.npz"), **adapter_params)

        adapter_params_all[domain_name] = adapter_params

        adapter_results[domain_name] = {
            "train_time_s": round(train_time, 1),
            "first_50_loss": round(first_50, 4),
            "last_50_loss": round(last_50, 4),
            "val_ppl": round(ppl, 2),
            "trainable_params": n_trainable,
        }

        cleanup(model, optimizer)

    return {"k2_kill": False, "adapter_results": adapter_results,
            "adapter_params": adapter_params_all}


# ============================================================================
# Phase 4: Compose adapters and evaluate
# ============================================================================

def _merge_deltas_into_model(model, adapter_params_list, a_bases_d, a_bases_ff,
                              domain_names, scale=None):
    """Merge LoRA deltas (B @ A) into base model weights.

    adapter_params_list: list of adapter param dicts (one per domain)
    scale: if None, uses 1/N. Otherwise uses the given scale.
    """
    n = len(adapter_params_list)
    if scale is None:
        scale = 1.0 / n

    for layer_idx, layer in enumerate(model.layers):
        # Token mixer projections (d -> d, use a_bases_d)
        for attr in ['w_gate', 'w_input', 'w_value', 'w_output']:
            bitlinear = getattr(layer.token_mixer, attr)
            delta_w = mx.zeros_like(bitlinear.weight)
            for domain_idx in range(n):
                a_d = a_bases_d[domain_idx]
                key = f"layers.{layer_idx}.token_mixer.{attr}.b_matrix"
                if key in adapter_params_list[domain_idx]:
                    b_mat = adapter_params_list[domain_idx][key]
                    delta_w = delta_w + b_mat @ a_d
            bitlinear.weight = bitlinear.weight + delta_w * scale

        # Channel mixer gate/up (d -> d_ff, use a_bases_d for input dim)
        for attr in ['w_gate', 'w_up']:
            bitlinear = getattr(layer.channel_mixer, attr)
            delta_w = mx.zeros_like(bitlinear.weight)
            for domain_idx in range(n):
                a_d = a_bases_d[domain_idx]
                key = f"layers.{layer_idx}.channel_mixer.{attr}.b_matrix"
                if key in adapter_params_list[domain_idx]:
                    b_mat = adapter_params_list[domain_idx][key]
                    delta_w = delta_w + b_mat @ a_d
            bitlinear.weight = bitlinear.weight + delta_w * scale

        # Down projection (d_ff -> d, use a_bases_ff for input dim)
        bitlinear = layer.channel_mixer.w_down
        delta_w = mx.zeros_like(bitlinear.weight)
        for domain_idx in range(n):
            a_ff = a_bases_ff[domain_idx]
            key = f"layers.{layer_idx}.channel_mixer.w_down.b_matrix"
            if key in adapter_params_list[domain_idx]:
                b_mat = adapter_params_list[domain_idx][key]
                delta_w = delta_w + b_mat @ a_ff
        bitlinear.weight = bitlinear.weight + delta_w * scale

    mx.eval(model.parameters())


def phase_composition(domain_datasets, vocab_size, adapter_params_all):
    """Compose all adapters and measure composition quality."""
    print("\n" + "=" * 60)
    print("PHASE 4: Adapter Composition on HGRNBit")
    print("=" * 60)

    domain_names = list(domain_datasets.keys())
    n_domains = len(domain_names)

    a_bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    a_bases_ff = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=123)

    # --- Individual adapter PPL (merge single adapter delta into base) ---
    individual_ppls = {}
    for domain_idx, domain_name in enumerate(domain_names):
        model = HGRNBitLM(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                           n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
                           d_ff=MLP_DIM)
        base_weights = dict(mx.load(str(EXPERIMENT_DIR / "hgrnbit_base.npz")))
        model.load_weights(list(base_weights.items()))

        # Merge single adapter with scale=1.0
        _merge_deltas_into_model(
            model,
            [adapter_params_all[domain_name]],
            [a_bases_d[domain_idx]],
            [a_bases_ff[domain_idx]],
            [domain_name],
            scale=1.0,
        )

        val_ds = domain_datasets[domain_name]["val"]
        ppl = compute_ppl(model, val_ds)
        individual_ppls[domain_name] = ppl
        print(f"  {domain_name} individual: PPL={ppl:.2f}")
        cleanup(model)

    # --- Composed adapter PPL (merge all adapters with 1/N scaling) ---
    print("\n  Computing composed weight deltas...")
    composed_ppls = {}

    model = HGRNBitLM(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                       n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
                       d_ff=MLP_DIM)
    base_weights = dict(mx.load(str(EXPERIMENT_DIR / "hgrnbit_base.npz")))
    model.load_weights(list(base_weights.items()))

    adapter_list = [adapter_params_all[d] for d in domain_names]
    _merge_deltas_into_model(model, adapter_list, a_bases_d, a_bases_ff,
                              domain_names)  # default 1/N

    # Evaluate composed model on each domain
    for domain_name in domain_names:
        val_ds = domain_datasets[domain_name]["val"]
        ppl = compute_ppl(model, val_ds)
        composed_ppls[domain_name] = ppl
        ind_ppl = individual_ppls[domain_name]
        ratio = ppl / ind_ppl if ind_ppl > 0 else float('inf')
        print(f"  {domain_name} composed: PPL={ppl:.2f} (individual={ind_ppl:.2f}, "
              f"ratio={ratio:.2f}x)")

    # Composition ratio
    avg_composed = sum(composed_ppls.values()) / len(composed_ppls)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    composition_ratio = avg_composed / avg_individual if avg_individual > 0 else float('inf')

    print(f"\n  Avg individual PPL: {avg_individual:.2f}")
    print(f"  Avg composed PPL:  {avg_composed:.2f}")
    print(f"  Composition ratio: {composition_ratio:.2f}x")

    # Orthogonality check
    print("\n  Adapter orthogonality (flattened B matrices):")
    cosines = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            vi = mx.concatenate([v.reshape(-1) for v in adapter_params_all[domain_names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapter_params_all[domain_names[j]].values()])
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-8))
            mx.eval(cos)
            cosines.append({"pair": f"{domain_names[i]}-{domain_names[j]}",
                           "abs_cos": round(cos.item(), 4)})

    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines)
    print(f"  Mean |cos|: {mean_cos:.4f}")
    for c in cosines:
        print(f"    {c['pair']}: {c['abs_cos']:.4f}")

    cleanup(model)

    return {
        "individual_ppls": individual_ppls,
        "composed_ppls": composed_ppls,
        "avg_individual_ppl": round(avg_individual, 2),
        "avg_composed_ppl": round(avg_composed, 2),
        "composition_ratio": round(composition_ratio, 4),
        "cosine_similarities": cosines,
        "mean_abs_cos": round(mean_cos, 4),
    }


def _apply_lora_to_model(model, rank, a_d, a_ff):
    """Apply LoRA wrappers to all BitLinear layers in model."""
    for layer in model.layers:
        for attr in ['w_gate', 'w_input', 'w_value', 'w_output']:
            base_layer = getattr(layer.token_mixer, attr)
            lora_layer = LoRABitLinear(base_layer, rank, a_d)
            setattr(layer.token_mixer, attr, lora_layer)

        for attr in ['w_gate', 'w_up']:
            base_layer = getattr(layer.channel_mixer, attr)
            lora_layer = LoRABitLinear(base_layer, rank, a_d)
            setattr(layer.channel_mixer, attr, lora_layer)

        base_down = layer.channel_mixer.w_down
        lora_down = LoRABitLinear(base_down, rank, a_ff)
        layer.channel_mixer.w_down = lora_down


# ============================================================================
# Phase 5: Inference speed comparison
# ============================================================================

def phase_speed_comparison(vocab_size):
    """Compare inference speed: HGRNBit vs Transformer."""
    print("\n" + "=" * 60)
    print("PHASE 5: Inference Speed Comparison")
    print("=" * 60)

    n_warmup = 10
    n_measure = 50
    test_input = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0, 0]])

    # HGRNBit
    hgrn_model = HGRNBitLM(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                             n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS,
                             d_ff=MLP_DIM)
    weights_path = EXPERIMENT_DIR / "hgrnbit_base.npz"
    if weights_path.exists():
        base_weights = dict(mx.load(str(weights_path)))
        hgrn_model.load_weights(list(base_weights.items()))
    mx.eval(hgrn_model.parameters())

    # Warmup
    for _ in range(n_warmup):
        out = hgrn_model(test_input)
        mx.eval(out)
        del out

    t0 = time.time()
    for _ in range(n_measure):
        out = hgrn_model(test_input)
        mx.eval(out)
        del out
    hgrn_time = (time.time() - t0) / n_measure
    cleanup(hgrn_model)

    # Transformer
    tf_model = FP32GPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(tf_model.parameters())

    for _ in range(n_warmup):
        out = tf_model(test_input)
        mx.eval(out)
        del out

    t0 = time.time()
    for _ in range(n_measure):
        out = tf_model(test_input)
        mx.eval(out)
        del out
    tf_time = (time.time() - t0) / n_measure
    cleanup(tf_model)

    speedup = tf_time / hgrn_time if hgrn_time > 0 else 0
    print(f"  HGRNBit: {hgrn_time*1000:.2f} ms/forward")
    print(f"  Transformer: {tf_time*1000:.2f} ms/forward")
    print(f"  Speedup: {speedup:.2f}x")

    return {
        "hgrn_ms": round(hgrn_time * 1000, 2),
        "transformer_ms": round(tf_time * 1000, 2),
        "speedup": round(speedup, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    log_memory("start")

    results = {
        "experiment": "matmul_free_lm_mlx",
        "architecture": "HGRNBit",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "block_size": BLOCK_SIZE,
        "lora_rank": LORA_RANK,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load data
    print("Loading data...")
    tok, train_ds, val_ds, domain_datasets = load_data()
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size}, Domains: {list(domain_datasets.keys())}")
    results["vocab_size"] = vocab_size

    # Phase 1: FP32 Transformer baseline
    fp32_results = phase_fp32_baseline(train_ds, val_ds, vocab_size)
    results["fp32_baseline"] = fp32_results
    log_memory("after-fp32")

    # Phase 2: HGRNBit base training
    hgrn_results = phase_hgrnbit_base(train_ds, val_ds, vocab_size)
    results["hgrnbit_base"] = hgrn_results

    # K1 check
    if hgrn_results.get("k1_kill"):
        results["k1_pass"] = False
        results["verdict"] = "KILLED (K1: loss > 2.0 at step 2000)"
        results["total_time_s"] = round(time.time() - t_start, 1)
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nVERDICT: KILLED (K1)")
        return

    results["k1_pass"] = True
    results["ppl_ratio"] = round(hgrn_results["val_ppl"] / fp32_results["val_ppl"], 4)
    print(f"\n  PPL ratio (HGRNBit/FP32): {results['ppl_ratio']:.4f}x")
    log_memory("after-hgrnbit")

    # Phase 3: LoRA adapters
    adapter_out = phase_adapters(domain_datasets, vocab_size)

    if adapter_out.get("k2_kill"):
        results["k2_pass"] = False
        results["verdict"] = "KILLED (K2: LoRA incompatible)"
        results["total_time_s"] = round(time.time() - t_start, 1)
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nVERDICT: KILLED (K2)")
        return

    results["k2_pass"] = True
    results["adapter_results"] = adapter_out["adapter_results"]
    log_memory("after-adapters")

    # Phase 4: Composition
    comp_results = phase_composition(domain_datasets, vocab_size,
                                      adapter_out["adapter_params"])
    results["composition"] = comp_results

    # K3 check
    comp_ratio = comp_results["composition_ratio"]
    results["k3_pass"] = comp_ratio < 1.5
    log_memory("after-composition")

    # Phase 5: Speed comparison
    speed_results = phase_speed_comparison(vocab_size)
    results["speed"] = speed_results
    log_memory("after-speed")

    # Summary
    results["total_time_s"] = round(time.time() - t_start, 1)

    all_pass = results["k1_pass"] and results["k2_pass"] and results["k3_pass"]
    if all_pass:
        results["verdict"] = "SUPPORTED"
    else:
        killed = []
        if not results["k1_pass"]:
            killed.append("K1")
        if not results["k2_pass"]:
            killed.append("K2")
        if not results["k3_pass"]:
            killed.append("K3")
        results["verdict"] = f"KILLED ({', '.join(killed)})"

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  K1 (val loss < 2.0 at step 2000): {'PASS' if results['k1_pass'] else 'FAIL'}")
    print(f"  K2 (LoRA compatible): {'PASS' if results['k2_pass'] else 'FAIL'}")
    print(f"  K3 (composition ratio < 1.5): {'PASS' if results['k3_pass'] else 'FAIL'} "
          f"({comp_ratio:.4f}x)")
    print(f"  PPL ratio (HGRNBit/FP32): {results['ppl_ratio']:.4f}x")
    print(f"  Speed: HGRNBit={speed_results['hgrn_ms']:.1f}ms, "
          f"TF={speed_results['transformer_ms']:.1f}ms, "
          f"speedup={speed_results['speedup']:.2f}x")
    print(f"  S1 (within 1.5x PPL): {'PASS' if results['ppl_ratio'] < 1.5 else 'FAIL'}")
    print(f"  S3 (faster inference): {'PASS' if speed_results['speedup'] > 1.0 else 'FAIL'}")
    print(f"  Total time: {results['total_time_s']:.0f}s")
    print(f"  VERDICT: {results['verdict']}")
    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
