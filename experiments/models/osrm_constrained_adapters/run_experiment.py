#!/usr/bin/env python3
"""
OSRM-Constrained Adapter Init Experiment

Tests whether OSRM (arXiv:2505.22934) covariance-constrained A-matrix init
produces better merge quality than random or Grassmannian init.

Three conditions:
  1. Random init: A_i ~ N(0,1/d), QR orthogonalized
  2. Grassmannian init: A_i from AP-packed frames on Gr(r, d)
  3. OSRM init: A_i from minimal-variance eigenvectors of leave-one-out covariance

Kill criteria:
  K1: OSRM adapters >5% worse individually than random init
  K2: OSRM merged quality not better than random merge

Success criteria:
  S1: OSRM merged quality within 5% of best individual, +5pp over random merge

Platform: Apple M5 Pro, 48GB, MLX
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
HIDDEN_SAMPLES = 100  # samples per domain for covariance estimation

EXPERIMENT_DIR = Path(__file__).parent
# Reuse existing data from the 5-domain experiment
DATA_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition" / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

DOMAINS = ["python", "math", "medical", "legal", "creative"]

# Target projections for LoRA
TARGET_KEYS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


def cleanup(*objects):
    """Release MLX memory between phases."""
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def mem_report(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"  [{label}] Active: {active:.2f}GB, Cache: {cache:.2f}GB")


# ===========================================================================
# Ternary unpacking (from bitnet_2b_real_composition)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    return unpacked / scale if invert_scale else unpacked * scale


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# Hidden state extraction for covariance estimation
# ===========================================================================
def phase_extract_hidden_states(domain_name: str, max_samples: int = HIDDEN_SAMPLES):
    """Extract mean-pooled hidden states from frozen base model for one domain."""
    print(f"\n  Extracting hidden states for {domain_name}...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    data_path = DATA_DIR / domain_name / "train.jsonl"
    texts = []
    with open(data_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    states = []
    for i, text in enumerate(texts[:max_samples]):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH]
        x = mx.array(tokens)[None, :]

        # Forward through all layers, extract final hidden state
        h = model.model.embed_tokens(x)
        for layer_module in model.model.layers:
            h = layer_module(h)
        h = model.model.norm(h)

        # Mean pool over sequence
        h_mean = mx.mean(h[0], axis=0)  # (d,)
        mx.eval(h_mean)
        states.append(np.array(h_mean.astype(mx.float32)))

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{min(len(texts), max_samples)} samples")

    cleanup(model, tokenizer)
    H = np.stack(states)  # (k, d)
    print(f"  {domain_name}: extracted {H.shape[0]} hidden states, d={H.shape[1]}")
    return H


def phase_compute_covariances():
    """Extract hidden states for all domains, compute covariances."""
    print("\n[Phase 1] Extracting hidden states and computing covariances...")

    hidden_states = {}
    for domain in DOMAINS:
        H = phase_extract_hidden_states(domain)
        hidden_states[domain] = H

    # Compute per-domain covariance matrices (float64 for numerical stability)
    covariances = {}
    for domain, H in hidden_states.items():
        H64 = H.astype(np.float64)
        H_centered = H64 - H64.mean(axis=0, keepdims=True)
        S = H_centered.T @ H_centered / (H64.shape[0] - 1)  # (d, d)
        covariances[domain] = S
        print(f"  {domain}: covariance shape {S.shape}, trace={np.trace(S):.2f}")

    # Compute leave-one-out covariances for OSRM (already float64 from above)
    osrm_covariances = {}
    for target_domain in DOMAINS:
        others = [dom for dom in DOMAINS if dom != target_domain]
        S_combined = sum(covariances[dom] for dom in others) / len(others)
        osrm_covariances[target_domain] = S_combined

    # Eigenspectrum analysis
    eigen_info = {}
    for domain, S in osrm_covariances.items():
        eigenvalues = np.linalg.eigvalsh(S)  # ascending order
        # Bottom-r eigenvalues (these are what OSRM uses)
        bottom_r = eigenvalues[:LORA_RANK]
        top_r = eigenvalues[-LORA_RANK:]
        ratio = np.sum(bottom_r) / np.sum(top_r)
        eigen_info[domain] = {
            "bottom_r_sum": float(np.sum(bottom_r)),
            "top_r_sum": float(np.sum(top_r)),
            "ratio": float(ratio),
            "min_eigenval": float(eigenvalues[0]),
            "max_eigenval": float(eigenvalues[-1]),
        }
        print(f"  {domain}: bottom-{LORA_RANK} / top-{LORA_RANK} eigenvalue ratio = {ratio:.6f}")

    return hidden_states, covariances, osrm_covariances, eigen_info


# ===========================================================================
# A-matrix initialization methods
# ===========================================================================
def init_random_a(d_in: int, rank: int, seed: int = 42) -> np.ndarray:
    """Random A init with QR orthogonalization. Returns (rank, d_in) float32."""
    rng = np.random.RandomState(seed)
    M = rng.randn(d_in, rank)
    Q, _ = np.linalg.qr(M)
    return Q[:, :rank].T.astype(np.float32)  # (rank, d_in)


def init_grassmannian_a(N: int, d: int, rank: int, n_iters: int = 300, seed: int = 42):
    """Grassmannian AP-packed frames. Returns list of N arrays, each (rank, d)."""
    rng = np.random.RandomState(seed)
    frames = np.zeros((N, d, rank), dtype=np.float64)
    for i in range(N):
        M = rng.randn(d, rank)
        Q, _ = np.linalg.qr(M)
        frames[i] = Q[:, :rank]

    # Alternating projection to maximize minimum pairwise distance
    for it in range(n_iters):
        for i in range(N):
            G = np.zeros((d, rank), dtype=np.float64)
            for j in range(N):
                if j != i:
                    overlap = frames[i].T @ frames[j]
                    G += frames[j] @ overlap.T
            candidate = frames[i] - 0.01 * G
            Q, _ = np.linalg.qr(candidate)
            frames[i] = Q[:, :rank]

    return [frames[i].T.astype(np.float32) for i in range(N)]  # list of (rank, d)


def init_osrm_a(osrm_covariances: dict, domains: list, rank: int):
    """OSRM init: A_i = bottom-r eigenvectors of leave-one-out covariance.
    Returns dict mapping domain -> (rank, d) float32 array."""
    a_matrices = {}
    for domain in domains:
        S = osrm_covariances[domain]  # float64
        eigenvalues, eigenvectors = np.linalg.eigh(S)  # ascending order
        # Bottom-r eigenvectors = columns with smallest eigenvalues
        A = eigenvectors[:, :rank].T  # (rank, d)
        a_matrices[domain] = A.astype(np.float32)
        print(f"  OSRM A for {domain}: bottom-{rank} eigenvalues = "
              f"[{eigenvalues[0]:.6e}, ..., {eigenvalues[rank-1]:.6e}], "
              f"sum={eigenvalues[:rank].sum():.6e}")
    return a_matrices


# ===========================================================================
# Cross-activation measurement
# ===========================================================================
def compute_cross_activations(a_matrices: dict, hidden_states: dict, domains: list):
    """Compute ||A_i @ H_j^T||_F / (||H_j||_F) for all i != j pairs."""
    cross_acts = {}
    for i_domain in domains:
        A_i = a_matrices[i_domain].astype(np.float64)  # (rank, d)
        for j_domain in domains:
            if i_domain == j_domain:
                continue
            H_j = hidden_states[j_domain].astype(np.float64)  # (k, d)
            projection = H_j @ A_i.T  # (k, rank)
            cross_norm = float(np.linalg.norm(projection, 'fro'))
            h_norm = float(np.linalg.norm(H_j, 'fro'))
            ratio = cross_norm / (h_norm + 1e-10)
            cross_acts[f"{i_domain}->{j_domain}"] = ratio
    return cross_acts


# ===========================================================================
# LoRA with custom A init
# ===========================================================================
class CustomInitLoRALinear(nn.Module):
    """LoRA with frozen custom A, trainable STE-ternary B."""

    def __init__(self, base_linear: nn.Linear, a_init: mx.array, scale: float):
        super().__init__()
        self.linear = base_linear
        # a_init: (rank, d_in) - frozen
        self.lora_a = a_init
        d_out = base_linear.weight.shape[0]
        rank = a_init.shape[0]
        self.lora_b = mx.zeros((d_out, rank))
        self.scale = scale

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        # STE ternary quantization on B
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        # x: (..., d_in), lora_a: (rank, d_in), b_ste: (d_out, rank)
        lora_out = (x @ self.lora_a.T) @ b_ste.T * self.scale
        return base_out + lora_out


def apply_custom_lora(model, a_matrices_per_layer: dict, scale: float):
    """Apply LoRA with custom A matrices to all target layers.

    a_matrices_per_layer: dict mapping "layers.{i}.{key}" -> mx.array (rank, d_in)
    """
    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        updates = []
        for key, module in layer.named_modules():
            if key in TARGET_KEYS and isinstance(module, nn.Linear):
                full_key = f"layers.{layer_idx}.{key}"
                if full_key in a_matrices_per_layer:
                    a_init = a_matrices_per_layer[full_key]
                    custom_lora = CustomInitLoRALinear(module, a_init, scale)
                    updates.append((key, custom_lora))
                    count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    print(f"  Applied custom LoRA to {count} layers")
    return model


def remove_custom_lora(model):
    """Remove custom LoRA, restoring plain nn.Linear."""
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, (CustomInitLoRALinear, LoRALinear)):
                base = module.linear if hasattr(module, 'linear') else module
                updates.append((key, base))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return model


def get_custom_lora_params(model):
    """Extract LoRA B parameters (A is frozen)."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def get_all_lora_params(model):
    """Extract all LoRA params (A and B) for composition."""
    params = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, CustomInitLoRALinear):
                prefix = f"model.layers.{layer_idx}.{key}"
                params[f"{prefix}.lora_a"] = mx.array(module.lora_a)
                params[f"{prefix}.lora_b"] = mx.array(module.lora_b)
    mx.eval(params)
    return params


# ===========================================================================
# Training
# ===========================================================================
def phase_train_condition(condition_name: str, a_init_fn, domains: list):
    """Train adapters for one init condition. Returns adapter params + PPLs."""
    print(f"\n[Training] Condition: {condition_name}")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    mem_report("after unpack")

    adapter_params = {}
    train_metrics = {}
    individual_ppls = {}
    base_ppls = {}

    for domain_idx, domain in enumerate(domains):
        print(f"\n  --- {condition_name}/{domain} ({domain_idx+1}/{len(domains)}) ---")

        # Get A matrices for this domain
        a_matrices = a_init_fn(domain)

        # Remove old LoRA, apply fresh custom LoRA
        model = remove_custom_lora(model)
        model = apply_custom_lora(model, a_matrices, LORA_SCALE)

        # Freeze everything, unfreeze only lora_b
        model.freeze()
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, CustomInitLoRALinear):
                    module.unfreeze(keys=["lora_b"], strict=False)

        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        print(f"  Trainable B params: {trainable:,}")

        # Load training data
        data_path = DATA_DIR / domain / "train.jsonl"
        texts = []
        with open(data_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        train_tokens = []
        for text in texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

        # Training loop
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        t_start = time.time()
        losses = []
        for step in range(TRAIN_ITERS):
            idx = step % len(train_tokens)
            tokens = train_tokens[idx]
            x = tokens[:-1][None, :]
            y = tokens[1:][None, :]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                print(f"    Step {step+1}/{TRAIN_ITERS}: loss={losses[-1]:.4f} (avg50={avg:.4f})")

        train_time = time.time() - t_start
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        converged = last_50 < first_50 * 0.95

        print(f"  {train_time:.1f}s. Loss: {first_50:.4f}->{last_50:.4f} "
              f"({'OK' if converged else 'NO'})")

        train_metrics[domain] = {
            "time_s": round(train_time, 1),
            "first_50_loss": round(first_50, 4),
            "last_50_loss": round(last_50, 4),
            "converged": converged,
        }

        # Save adapter params
        params = get_all_lora_params(model)
        adapter_params[domain] = params

        # Compute base PPL (only once, first condition)
        if condition_name == "random":
            # Temporarily remove LoRA to get base PPL
            model_clean = remove_custom_lora(model)
            base_ppl = compute_ppl(model_clean, tokenizer, DATA_DIR / domain)
            base_ppls[domain] = base_ppl
            # Re-apply for individual eval
            model = apply_custom_lora(model_clean, a_matrices, LORA_SCALE)
            # Restore B params
            for layer_idx, layer in enumerate(model.model.layers):
                for key, module in layer.named_modules():
                    if isinstance(module, CustomInitLoRALinear):
                        pkey = f"model.layers.{layer_idx}.{key}.lora_b"
                        if pkey in params:
                            module.lora_b = params[pkey]
            mx.eval(model.parameters())

        # Compute individual PPL
        ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
        individual_ppls[domain] = ppl
        print(f"  {domain}: PPL={ppl:.2f}")

        # Cleanup optimizer
        del optimizer
        gc.collect()

    cleanup(model, tokenizer)
    return adapter_params, train_metrics, individual_ppls, base_ppls


def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = VAL_BATCHES):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ===========================================================================
# Composition evaluation
# ===========================================================================
def phase_evaluate_composition(condition_name: str, adapter_params: dict, domains: list):
    """Merge all adapters and evaluate composed PPL."""
    print(f"\n[Composition] Condition: {condition_name}")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Merge adapter params: average all A and B matrices
    all_keys = set()
    for params in adapter_params.values():
        all_keys.update(params.keys())

    merged = {}
    N = len(adapter_params)
    for key in all_keys:
        tensors = [adapter_params[d][key] for d in domains if key in adapter_params[d]]
        if tensors:
            stacked = mx.stack(tensors)
            merged[key] = mx.mean(stacked, axis=0)  # 1/N scaling via mean
    mx.eval(merged)

    # Apply merged params using standard LoRALinear
    from mlx_lm.tuner.lora import LoRALinear
    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        updates = []
        for key, module in layer.named_modules():
            if key in TARGET_KEYS and isinstance(module, nn.Linear):
                a_key = f"model.layers.{layer_idx}.{key}.lora_a"
                b_key = f"model.layers.{layer_idx}.{key}.lora_b"
                if a_key in merged and b_key in merged:
                    lora = LoRALinear.from_base(module, r=LORA_RANK, scale=LORA_SCALE, dropout=0.0)
                    lora.lora_a = merged[a_key].T  # LoRALinear expects (d_in, rank)
                    lora.lora_b = merged[b_key].T   # LoRALinear expects (rank, d_out)
                    updates.append((key, lora))
                    count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    print(f"  Applied merged adapter ({count} layers)")

    # Evaluate on all domains
    composed_ppls = {}
    for domain in domains:
        ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
        composed_ppls[domain] = ppl
        print(f"  {domain}: composed PPL = {ppl:.2f}")

    cleanup(model, tokenizer)
    return composed_ppls


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    t_total = time.time()
    results = {
        "experiment": "osrm_constrained_adapters",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("OSRM-Constrained Adapter Init Experiment")
    print("=" * 70)

    # Verify data exists
    for domain in DOMAINS:
        if not (DATA_DIR / domain / "train.jsonl").exists():
            print(f"FATAL: Missing data for {domain} at {DATA_DIR / domain}")
            print("Run bitnet_2b_real_composition first.")
            return

    # ------------------------------------------------------------------
    # Phase 1: Extract hidden states and compute covariances
    # ------------------------------------------------------------------
    hidden_states, covariances, osrm_covariances, eigen_info = phase_compute_covariances()
    results["eigenspectrum"] = eigen_info

    # Save hidden states to numpy for cross-activation analysis later
    hidden_states_np = hidden_states  # already numpy

    # ------------------------------------------------------------------
    # Phase 2: Compute A inits for all three conditions
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing A matrix initializations...")
    d = list(hidden_states.values())[0].shape[1]  # 2560

    # Random init
    random_a_matrices = {}
    for i, domain in enumerate(DOMAINS):
        A = init_random_a(d, LORA_RANK, seed=42 + i)
        random_a_matrices[domain] = A
    print(f"  Random: {len(random_a_matrices)} x ({LORA_RANK}, {d})")

    # Grassmannian init
    grassmannian_a_list = init_grassmannian_a(len(DOMAINS), d, LORA_RANK)
    grassmannian_a_matrices = {domain: grassmannian_a_list[i] for i, domain in enumerate(DOMAINS)}
    print(f"  Grassmannian: {len(grassmannian_a_matrices)} x ({LORA_RANK}, {d})")

    # OSRM init
    osrm_a_matrices = init_osrm_a(osrm_covariances, DOMAINS, LORA_RANK)
    print(f"  OSRM: {len(osrm_a_matrices)} x ({LORA_RANK}, {d})")

    # Cross-activation comparison BEFORE training
    print("\n  Cross-activation norms (pre-training):")
    for name, a_mats in [("random", random_a_matrices),
                          ("grassmannian", grassmannian_a_matrices),
                          ("osrm", osrm_a_matrices)]:
        cross = compute_cross_activations(a_mats, hidden_states_np, DOMAINS)
        mean_cross = np.mean(list(cross.values()))
        print(f"    {name}: mean cross-activation = {mean_cross:.6f}")
        results[f"cross_activation_pre_{name}"] = {k: round(v, 6) for k, v in cross.items()}
        results[f"cross_activation_pre_{name}_mean"] = round(float(mean_cross), 6)

    # ------------------------------------------------------------------
    # Phase 3: Build per-layer A matrices and train each condition
    # ------------------------------------------------------------------
    # For each condition, we need A matrices for every (layer, projection) pair.
    # We use the SAME A across all layers for simplicity (matches standard LoRA).

    # Detect projection input dimensions from the model
    # Most projections have d_in=2560, but down_proj has d_in=d_ffn
    # We only apply OSRM/Grassmannian to d_in=2560 projections;
    # down_proj gets random init in all conditions (OSRM covariance doesn't apply to FFN space)
    HIDDEN_DIM = d  # 2560

    def make_layer_a_dict(domain_a_matrices: dict, domain: str):
        """Create per-layer A matrix dict for a given domain.
        Uses the init method's A for d=2560 projections, random for others."""
        A_hidden = mx.array(domain_a_matrices[domain]).astype(mx.bfloat16)  # (rank, 2560)
        # For down_proj: d_in = d_ffn (6912 for BitNet-2B-4T)
        D_FFN = 6912
        seed = hash(domain) % (2**31)
        rng = np.random.RandomState(seed)
        M = rng.randn(D_FFN, LORA_RANK).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        A_ffn = mx.array(Q[:, :LORA_RANK].T.astype(np.float32)).astype(mx.bfloat16)  # (rank, d_ffn)

        result = {}
        for layer_idx in range(30):
            for key in TARGET_KEYS:
                if key == "mlp.down_proj":
                    result[f"layers.{layer_idx}.{key}"] = A_ffn
                else:
                    result[f"layers.{layer_idx}.{key}"] = A_hidden
        return result

    # Train random condition
    def random_a_fn(domain):
        return make_layer_a_dict(random_a_matrices, domain)

    random_params, random_train, random_ppls, base_ppls = phase_train_condition(
        "random", random_a_fn, DOMAINS
    )
    results["base_ppls"] = base_ppls
    results["random_train"] = random_train
    results["random_individual_ppls"] = random_ppls

    # Train Grassmannian condition
    def grassmannian_a_fn(domain):
        return make_layer_a_dict(grassmannian_a_matrices, domain)

    grass_params, grass_train, grass_ppls, _ = phase_train_condition(
        "grassmannian", grassmannian_a_fn, DOMAINS
    )
    results["grassmannian_train"] = grass_train
    results["grassmannian_individual_ppls"] = grass_ppls

    # Train OSRM condition
    def osrm_a_fn(domain):
        return make_layer_a_dict(osrm_a_matrices, domain)

    osrm_params, osrm_train, osrm_ppls, _ = phase_train_condition(
        "osrm", osrm_a_fn, DOMAINS
    )
    results["osrm_train"] = osrm_train
    results["osrm_individual_ppls"] = osrm_ppls

    # ------------------------------------------------------------------
    # Phase 4: Composition evaluation
    # ------------------------------------------------------------------
    random_composed = phase_evaluate_composition("random", random_params, DOMAINS)
    results["random_composed_ppls"] = random_composed

    grass_composed = phase_evaluate_composition("grassmannian", grass_params, DOMAINS)
    results["grassmannian_composed_ppls"] = grass_composed

    osrm_composed = phase_evaluate_composition("osrm", osrm_params, DOMAINS)
    results["osrm_composed_ppls"] = osrm_composed

    # ------------------------------------------------------------------
    # Phase 5: Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Individual PPL comparison
    print("\nIndividual PPL (lower = better):")
    print(f"  {'Domain':<12} {'Base':>8} {'Random':>8} {'Grass':>8} {'OSRM':>8}")
    for domain in DOMAINS:
        base = base_ppls.get(domain, float('nan'))
        r = random_ppls[domain]
        g = grass_ppls[domain]
        o = osrm_ppls[domain]
        print(f"  {domain:<12} {base:>8.2f} {r:>8.2f} {g:>8.2f} {o:>8.2f}")

    # K1: OSRM individual quality
    osrm_worse_count = 0
    for domain in DOMAINS:
        if osrm_ppls[domain] > random_ppls[domain] * 1.05:
            osrm_worse_count += 1
    k1_pass = osrm_worse_count == 0
    results["k1_pass"] = k1_pass
    results["k1_osrm_worse_count"] = osrm_worse_count
    print(f"\nK1 (OSRM not >5% worse individually): {'PASS' if k1_pass else 'FAIL'} "
          f"({osrm_worse_count}/5 domains worse)")

    # Composed PPL comparison
    print("\nComposed PPL (lower = better):")
    print(f"  {'Domain':<12} {'Random':>8} {'Grass':>8} {'OSRM':>8}")
    for domain in DOMAINS:
        r = random_composed[domain]
        g = grass_composed[domain]
        o = osrm_composed[domain]
        print(f"  {domain:<12} {r:>8.2f} {g:>8.2f} {o:>8.2f}")

    # K2: OSRM merge quality
    random_mean_composed = np.mean([random_composed[d] for d in DOMAINS])
    osrm_mean_composed = np.mean([osrm_composed[d] for d in DOMAINS])
    grass_mean_composed = np.mean([grass_composed[d] for d in DOMAINS])

    k2_pass = osrm_mean_composed < random_mean_composed
    results["k2_pass"] = k2_pass
    results["mean_composed_ppls"] = {
        "random": round(float(random_mean_composed), 2),
        "grassmannian": round(float(grass_mean_composed), 2),
        "osrm": round(float(osrm_mean_composed), 2),
    }
    improvement = (random_mean_composed - osrm_mean_composed) / random_mean_composed * 100
    results["osrm_vs_random_improvement_pct"] = round(float(improvement), 2)

    print(f"\nK2 (OSRM merge better than random): {'PASS' if k2_pass else 'FAIL'}")
    print(f"  Random mean composed PPL:  {random_mean_composed:.2f}")
    print(f"  Grassmannian mean:         {grass_mean_composed:.2f}")
    print(f"  OSRM mean:                 {osrm_mean_composed:.2f}")
    print(f"  OSRM improvement: {improvement:+.1f}%")

    # S1: success criterion
    best_individual = min(min(random_ppls[d], grass_ppls[d], osrm_ppls[d]) for d in DOMAINS)
    s1_within_5pct = osrm_mean_composed < best_individual * 1.05
    s1_5pp_better = improvement >= 5.0
    s1_pass = s1_within_5pct and s1_5pp_better
    results["s1_pass"] = s1_pass

    # Overall verdict
    if not k1_pass:
        verdict = "KILLED"
        reason = "OSRM hurts individual adapter quality"
    elif not k2_pass:
        verdict = "KILLED"
        reason = "OSRM merge not better than random"
    elif s1_pass:
        verdict = "SUPPORTED"
        reason = f"OSRM merge {improvement:+.1f}% better than random"
    else:
        verdict = "SUPPORTED"
        reason = f"K1+K2 pass but S1 stretch goal not met ({improvement:+.1f}%)"

    results["verdict"] = verdict
    results["reason"] = reason
    results["total_time_s"] = round(time.time() - t_total, 1)

    print(f"\nVERDICT: {verdict}")
    print(f"REASON: {reason}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
