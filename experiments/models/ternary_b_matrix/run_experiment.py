#!/usr/bin/env python3
"""Experiment: Ternary B-Matrix for Fully Ternary LoRA Adapters.

Can we make the B-matrix ternary too? Current adapters have fp32 B on ternary A.
Fully ternary adapter = pure addition composition, no matmul.

Kill criteria:
  K1 (id=256): Ternary B composition ratio > 2.0 -> KILL
  K2 (id=257): Per-domain PPL > 1.5x FP32 B baseline -> KILL

Success criteria:
  S1 (id=28): Fully ternary adapters compose with ratio < 1.5

Methods tested:
  1. Baseline: FP32 B on ternary base (Grassmannian A, existing approach)
  2. STE ternary B: Train B with STE quantization from the start
  3. PTQ ternary B: Train FP32 B, then post-training quantize
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

# Architecture (smaller than ternary_base_from_scratch for faster iteration)
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 32
BLOCK_SIZE = 32
MLP_DIM = 4 * D_MODEL  # 512
LORA_RANK = 8

# Training
BASE_STEPS = 3000
BASE_LR = 3e-4
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
# BitLinear: Ternary weights with STE
# ============================================================================

class BitLinear(nn.Module):
    """Linear layer with ternary quantization via STE."""
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
# Transformer
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
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 128, n_head: int = 4, n_layer: int = 4):
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
# LoRA Adapters: Three variants
# ============================================================================

class FP32LoRALinear(nn.Module):
    """Standard FP32 B-matrix LoRA on frozen BitLinear. Baseline."""
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
        lora_out = (x @ self.a_matrix.T) @ self.b_matrix.T
        return base_out + lora_out


class TernaryBLoRALinear(nn.Module):
    """Ternary B-matrix LoRA with STE quantization during training."""
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
        # STE ternary quantization on B
        b = self.b_matrix
        alpha_b = mx.mean(mx.abs(b)) + 1e-7
        b_scaled = b / alpha_b
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha_b
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.a_matrix.T) @ b_ste.T
        return base_out + lora_out


def generate_grassmannian_bases(n_adapters: int, rank: int, dim: int, seed: int = 42):
    """Generate n_adapters orthogonal A matrices via QR decomposition."""
    import numpy as np
    rng = np.random.RandomState(seed)
    total_rank = n_adapters * rank
    assert total_rank <= dim, f"Need {total_rank} orthogonal vectors but dim={dim}"
    random_mat = rng.randn(dim, total_rank).astype(np.float32)
    Q, _ = np.linalg.qr(random_mat)
    bases = []
    for i in range(n_adapters):
        start = i * rank
        end = start + rank
        A_i = mx.array(Q[:, start:end].T)  # (rank, dim)
        bases.append(A_i)
    return bases


# ============================================================================
# Data
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
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    return math.exp(total_loss / n_batches)


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def count_trainable(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


# ============================================================================
# Phase 1: Train ternary base
# ============================================================================

def phase_ternary_base(train_ds, val_ds, vocab_size):
    """Train ternary base model."""
    print("\n" + "=" * 60)
    print("PHASE 1: Ternary Base Training")
    print("=" * 60)

    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    mx.eval(model.parameters())
    print(f"Model params: {count_params(model):,}")
    log_memory("base-init")

    optimizer = opt.Adam(learning_rate=BASE_LR)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42)

    gc.disable()
    t0 = time.time()
    for step in range(1, BASE_STEPS + 1):
        inputs, targets = train_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 500 == 0 or step == BASE_STEPS:
            print(f"  step {step:4d}/{BASE_STEPS} | loss {loss.item():.4f} | "
                  f"time {time.time()-t0:.1f}s")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    base_ppl = compute_ppl(model, val_ds)
    print(f"Ternary base PPL: {base_ppl:.2f}")
    log_memory("base-done")

    # Save weights
    weights_path = EXPERIMENT_DIR / "ternary_base_weights.npz"
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(str(weights_path), **flat_params)

    result = {"base_ppl": round(base_ppl, 4), "base_train_time_s": round(train_time, 1)}
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase 2: Train adapters (3 conditions)
# ============================================================================

def _attach_adapters(model, adapter_idx, bases_d, bases_mlp, ternary_b: bool):
    """Attach LoRA adapters to all BitLinear layers.

    Args:
        ternary_b: if True, use TernaryBLoRALinear (STE on B); else FP32LoRALinear
    """
    model.freeze()
    a_d = bases_d[adapter_idx]
    a_mlp = bases_mlp[adapter_idx]
    LoRAClass = TernaryBLoRALinear if ternary_b else FP32LoRALinear

    for layer in model.layers:
        layer.attn.wq = LoRAClass(layer.attn.wq, LORA_RANK, a_d)
        layer.attn.wk = LoRAClass(layer.attn.wk, LORA_RANK, a_d)
        layer.attn.wv = LoRAClass(layer.attn.wv, LORA_RANK, a_d)
        layer.attn.wo = LoRAClass(layer.attn.wo, LORA_RANK, a_d)
        layer.mlp.fc1 = LoRAClass(layer.mlp.fc1, LORA_RANK, a_d)
        layer.mlp.fc2 = LoRAClass(layer.mlp.fc2, LORA_RANK, a_mlp)
    model.lm_head = LoRAClass(model.lm_head, LORA_RANK, a_d)


def _train_single_adapter(model, domain_data, adapter_idx, condition_name):
    """Train one adapter, return B-matrices and PPL."""
    optimizer = opt.Adam(learning_rate=ADAPTER_LR)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(42 + adapter_idx)

    gc.disable()
    t0 = time.time()
    for step in range(1, ADAPTER_STEPS + 1):
        inputs, targets = domain_data["train"].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 500 == 0 or step == ADAPTER_STEPS:
            print(f"    [{condition_name}] step {step:4d}/{ADAPTER_STEPS} | loss {loss.item():.4f}")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    ppl = compute_ppl(model, domain_data["val"])

    # Extract B-matrices
    b_matrices = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        if "b_matrix" in name:
            mx.eval(param)
            b_matrices[name] = param

    del optimizer
    return ppl, train_time, b_matrices


def phase_train_adapters(domain_datasets, vocab_size, condition_name, ternary_b):
    """Train 5 domain adapters under one condition (fp32_b or ste_ternary_b)."""
    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Adapter Training [{condition_name}]")
    print(f"{'=' * 60}")

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    save_dir = ADAPTERS_DIR / condition_name
    save_dir.mkdir(parents=True, exist_ok=True)

    domain_results = {}
    for idx, dname in enumerate(domain_names):
        print(f"\n--- {condition_name} adapter {idx+1}/{n_domains}: {dname} ---")

        # Fresh model from saved base
        model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
        loaded = mx.load(str(EXPERIMENT_DIR / "ternary_base_weights.npz"))
        model.load_weights(list(loaded.items()))
        mx.eval(model.parameters())

        _attach_adapters(model, idx, bases_d, bases_mlp, ternary_b=ternary_b)
        n_trainable = count_trainable(model)
        if idx == 0:
            print(f"  Trainable params per adapter: {n_trainable:,}")

        ppl, train_time, b_matrices = _train_single_adapter(
            model, domain_datasets[dname], idx, condition_name
        )
        print(f"  {dname} PPL: {ppl:.2f} ({train_time:.1f}s)")

        # Save B-matrices to disk
        mx.savez(str(save_dir / f"{dname}.npz"), **b_matrices)

        domain_results[dname] = {"ppl": round(ppl, 4), "time_s": round(train_time, 1)}
        cleanup(model)

    return domain_results


# ============================================================================
# Phase 2b: PTQ -- quantize fp32 B-matrices after training
# ============================================================================

def phase_ptq_adapters(domain_datasets, vocab_size):
    """Post-training quantize fp32 B-matrices to ternary and evaluate."""
    print(f"\n{'=' * 60}")
    print("PHASE 2b: Post-Training Quantization of FP32 B-matrices")
    print(f"{'=' * 60}")

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    fp32_dir = ADAPTERS_DIR / "fp32_b"
    ptq_dir = ADAPTERS_DIR / "ptq_ternary_b"
    ptq_dir.mkdir(parents=True, exist_ok=True)

    domain_results = {}
    for idx, dname in enumerate(domain_names):
        print(f"\n--- PTQ adapter {idx+1}/{n_domains}: {dname} ---")

        # Load fp32 B-matrices
        fp32_data = dict(mx.load(str(fp32_dir / f"{dname}.npz")))

        # Quantize each B-matrix
        ptq_data = {}
        quant_errors = []
        for bname, b_fp32 in fp32_data.items():
            alpha = mx.mean(mx.abs(b_fp32))
            mx.eval(alpha)
            if alpha.item() < 1e-10:
                ptq_data[bname] = mx.zeros_like(b_fp32)
                continue
            b_scaled = b_fp32 / alpha
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            mx.eval(b_q)
            ptq_data[bname] = b_q

            # Quantization error
            err = mx.sqrt(mx.sum((b_fp32 - b_q) ** 2)).item()
            norm = mx.sqrt(mx.sum(b_fp32 ** 2)).item()
            if norm > 1e-10:
                quant_errors.append(err / norm)

        mx.savez(str(ptq_dir / f"{dname}.npz"), **ptq_data)

        mean_qerr = sum(quant_errors) / len(quant_errors) if quant_errors else 0
        print(f"  {dname} mean quantization error: {mean_qerr:.4f}")

        # Evaluate PTQ adapter
        model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                            n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
        loaded = mx.load(str(EXPERIMENT_DIR / "ternary_base_weights.npz"))
        model.load_weights(list(loaded.items()))
        mx.eval(model.parameters())

        # Attach FP32 LoRA but load PTQ weights
        _attach_adapters(model, idx, bases_d, bases_mlp, ternary_b=False)

        # Overwrite B-matrices with PTQ versions
        ptq_loaded = dict(mx.load(str(ptq_dir / f"{dname}.npz")))
        for name, param in nn.utils.tree_flatten(model.parameters()):
            if "b_matrix" in name and name in ptq_loaded:
                # Find and set the parameter
                parts = name.split(".")
                obj = model
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                setattr(obj, parts[-1], ptq_loaded[name])
        mx.eval(model.parameters())

        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        print(f"  {dname} PTQ PPL: {ppl:.2f}")

        domain_results[dname] = {
            "ppl": round(ppl, 4),
            "mean_quant_error": round(mean_qerr, 4),
        }
        cleanup(model)

    return domain_results


# ============================================================================
# Phase 3: Composition test
# ============================================================================

def phase_composition_test(domain_datasets, vocab_size, condition_name,
                           domain_results_single):
    """Test N-adapter composition for a given condition."""
    print(f"\n{'=' * 60}")
    print(f"PHASE 3: Composition Test [{condition_name}]")
    print(f"{'=' * 60}")

    domain_names = sorted(domain_datasets.keys())
    n_domains = len(domain_names)
    bases_d = generate_grassmannian_bases(n_domains, LORA_RANK, D_MODEL, seed=42)
    bases_mlp = generate_grassmannian_bases(n_domains, LORA_RANK, MLP_DIM, seed=137)
    mx.eval(bases_d)
    mx.eval(bases_mlp)

    adapter_dir = ADAPTERS_DIR / condition_name

    # Load base model
    model = TernaryGPT(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                        n_embd=D_MODEL, n_head=N_HEADS, n_layer=N_LAYERS)
    loaded = mx.load(str(EXPERIMENT_DIR / "ternary_base_weights.npz"))
    model.load_weights(list(loaded.items()))
    mx.eval(model.parameters())

    # Collect all adapter B-matrices
    all_adapter_b = {}
    for dname in domain_names:
        data = dict(mx.load(str(adapter_dir / f"{dname}.npz")))
        all_adapter_b[dname] = {k: v for k, v in data.items() if "b_matrix" in k}

    # Compute composed deltas: sum (1/N) * B_i @ A_i
    composed_deltas = {}
    for idx, dname in enumerate(domain_names):
        adapter_b = all_adapter_b[dname]
        for bname, b_val in adapter_b.items():
            a_matrix = bases_mlp[idx] if ".mlp.fc2." in bname else bases_d[idx]
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

    # Evaluate on each domain
    composed_ppls = {}
    for dname in domain_names:
        ppl = compute_ppl(model, domain_datasets[dname]["val"])
        composed_ppls[dname] = round(ppl, 4)
        print(f"  Composed PPL on {dname}: {ppl:.2f}")

    # Composition ratio
    single_ppls = [domain_results_single[d]["ppl"] for d in domain_names]
    composed_vals = [composed_ppls[d] for d in domain_names]
    mean_single = sum(single_ppls) / len(single_ppls)
    mean_composed = sum(composed_vals) / len(composed_vals)
    ratio = mean_composed / mean_single

    print(f"\nMean single PPL: {mean_single:.2f}")
    print(f"Mean composed PPL: {mean_composed:.2f}")
    print(f"Composition ratio: {ratio:.3f}")

    # Adapter cosine similarities
    cos_sims = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            delta_i_parts, delta_j_parts = [], []
            for bname in all_adapter_b[domain_names[i]]:
                bi = all_adapter_b[domain_names[i]][bname]
                bj = all_adapter_b[domain_names[j]][bname]
                a_i = bases_mlp[i] if ".mlp.fc2." in bname else bases_d[i]
                a_j = bases_mlp[j] if ".mlp.fc2." in bname else bases_d[j]
                delta_i_parts.append((bi @ a_i).reshape(-1))
                delta_j_parts.append((bj @ a_j).reshape(-1))

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
    print(f"Mean |cos| between adapters: {mean_cos:.6f}")

    result = {
        "composed_ppls": composed_ppls,
        "mean_single_ppl": round(mean_single, 4),
        "mean_composed_ppl": round(mean_composed, 4),
        "composition_ratio": round(ratio, 4),
        "mean_cos_similarity": round(mean_cos, 6),
        "cos_similarities": [round(c, 6) for c in cos_sims],
    }

    cleanup(model)
    return result


# ============================================================================
# Phase 4: Adapter size analysis
# ============================================================================

def phase_size_analysis():
    """Compare adapter storage sizes across conditions."""
    print(f"\n{'=' * 60}")
    print("PHASE 4: Adapter Size Analysis")
    print(f"{'=' * 60}")

    results = {}
    for condition in ["fp32_b", "ste_ternary_b", "ptq_ternary_b"]:
        cond_dir = ADAPTERS_DIR / condition
        if not cond_dir.exists():
            continue
        total_bytes = 0
        total_params = 0
        for f in cond_dir.glob("*.npz"):
            data = mx.load(str(f))
            for k, v in data.items():
                total_params += v.size
                # FP32: 4 bytes per param
                total_bytes += v.size * 4
        # Ternary: 2 bits per param + scale overhead
        if "ternary" in condition:
            ternary_bytes = (total_params * 2) // 8  # 2 bits per element
            n_files = len(list(cond_dir.glob("*.npz")))
            ternary_bytes += n_files * 28 * 4  # ~28 layers * 4 bytes per scale
            results[condition] = {
                "total_params": total_params,
                "fp32_bytes": total_bytes,
                "ternary_bytes": ternary_bytes,
                "compression_ratio": round(total_bytes / max(ternary_bytes, 1), 1),
            }
        else:
            results[condition] = {
                "total_params": total_params,
                "fp32_bytes": total_bytes,
            }
        print(f"  {condition}: {total_params:,} params, {total_bytes/1024:.1f} KB (fp32)")
        if "ternary" in condition:
            print(f"    -> ternary: {ternary_bytes/1024:.1f} KB "
                  f"({results[condition]['compression_ratio']}x compression)")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    print("Loading data...")
    tok, train_ds, val_ds, domain_datasets = load_data()
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size}")
    print(f"Domains: {sorted(domain_datasets.keys())}")
    for dname, ddata in sorted(domain_datasets.items()):
        print(f"  {dname}: train={len(ddata['train'])}, val={len(ddata['val'])}")

    # Phase 1: Train ternary base
    base_results = phase_ternary_base(train_ds, val_ds, vocab_size)
    log_memory("after-base")

    # Phase 2a: Train FP32 B adapters (baseline)
    fp32_results = phase_train_adapters(domain_datasets, vocab_size, "fp32_b", ternary_b=False)
    log_memory("after-fp32-adapters")

    # Phase 2b: Train STE ternary B adapters
    ste_results = phase_train_adapters(domain_datasets, vocab_size, "ste_ternary_b", ternary_b=True)
    log_memory("after-ste-adapters")

    # Phase 2c: PTQ ternary B (quantize fp32 B after training)
    ptq_results = phase_ptq_adapters(domain_datasets, vocab_size)
    log_memory("after-ptq-adapters")

    # K2 check: per-domain PPL of ternary B vs fp32 B
    print(f"\n{'=' * 60}")
    print("KILL CRITERIA EVALUATION")
    print(f"{'=' * 60}")

    print("\n--- K2: Per-domain PPL comparison ---")
    k2_ste_ratios = []
    k2_ptq_ratios = []
    for dname in sorted(domain_datasets.keys()):
        fp32_ppl = fp32_results[dname]["ppl"]
        ste_ppl = ste_results[dname]["ppl"]
        ptq_ppl = ptq_results[dname]["ppl"]
        ste_ratio = ste_ppl / fp32_ppl
        ptq_ratio = ptq_ppl / fp32_ppl
        k2_ste_ratios.append(ste_ratio)
        k2_ptq_ratios.append(ptq_ratio)
        print(f"  {dname}: fp32={fp32_ppl:.2f}, STE={ste_ppl:.2f} ({ste_ratio:.3f}x), "
              f"PTQ={ptq_ppl:.2f} ({ptq_ratio:.3f}x)")

    max_ste_ratio = max(k2_ste_ratios)
    max_ptq_ratio = max(k2_ptq_ratios)
    k2_ste_pass = max_ste_ratio < 1.5
    k2_ptq_pass = max_ptq_ratio < 1.5
    print(f"\n  STE max ratio: {max_ste_ratio:.3f} -> {'PASS' if k2_ste_pass else 'FAIL'} (threshold 1.5)")
    print(f"  PTQ max ratio: {max_ptq_ratio:.3f} -> {'PASS' if k2_ptq_pass else 'FAIL'} (threshold 1.5)")

    # Phase 3: Composition tests
    fp32_comp = phase_composition_test(domain_datasets, vocab_size, "fp32_b", fp32_results)
    log_memory("after-fp32-composition")

    ste_comp = phase_composition_test(domain_datasets, vocab_size, "ste_ternary_b", ste_results)
    log_memory("after-ste-composition")

    ptq_comp = phase_composition_test(domain_datasets, vocab_size, "ptq_ternary_b", ptq_results)
    log_memory("after-ptq-composition")

    # K1 check: composition ratio
    print(f"\n--- K1: Composition ratio comparison ---")
    k1_fp32 = fp32_comp["composition_ratio"]
    k1_ste = ste_comp["composition_ratio"]
    k1_ptq = ptq_comp["composition_ratio"]
    print(f"  FP32 B: {k1_fp32:.3f}")
    print(f"  STE ternary B: {k1_ste:.3f} -> {'PASS' if k1_ste < 2.0 else 'FAIL'} (threshold 2.0)")
    print(f"  PTQ ternary B: {k1_ptq:.3f} -> {'PASS' if k1_ptq < 2.0 else 'FAIL'} (threshold 2.0)")

    # S1 check
    s1_ste = k1_ste < 1.5
    s1_ptq = k1_ptq < 1.5
    print(f"\n--- S1: Composition ratio < 1.5 ---")
    print(f"  STE: {k1_ste:.3f} -> {'PASS' if s1_ste else 'FAIL'}")
    print(f"  PTQ: {k1_ptq:.3f} -> {'PASS' if s1_ptq else 'FAIL'}")

    # Phase 4: Size analysis
    size_results = phase_size_analysis()

    # Aggregate results
    total_time = time.time() - t0_total
    results = {
        "experiment": "ternary_b_matrix",
        "architecture": {
            "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "lora_rank": LORA_RANK, "block_size": BLOCK_SIZE,
        },
        "base": base_results,
        "fp32_b": {
            "per_domain": fp32_results,
            "composition": fp32_comp,
        },
        "ste_ternary_b": {
            "per_domain": ste_results,
            "composition": ste_comp,
        },
        "ptq_ternary_b": {
            "per_domain": ptq_results,
            "composition": ptq_comp,
        },
        "size_analysis": size_results,
        "kill_criteria": {
            "k1_ste_composition_ratio": k1_ste,
            "k1_ste_pass": k1_ste < 2.0,
            "k1_ptq_composition_ratio": k1_ptq,
            "k1_ptq_pass": k1_ptq < 2.0,
            "k2_ste_max_ppl_ratio": round(max_ste_ratio, 4),
            "k2_ste_pass": k2_ste_pass,
            "k2_ptq_max_ppl_ratio": round(max_ptq_ratio, 4),
            "k2_ptq_pass": k2_ptq_pass,
        },
        "success_criteria": {
            "s1_ste_ratio_under_1_5": s1_ste,
            "s1_ptq_ratio_under_1_5": s1_ptq,
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log_memory("final")


if __name__ == "__main__":
    main()
