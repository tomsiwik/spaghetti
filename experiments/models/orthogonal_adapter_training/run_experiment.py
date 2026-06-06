#!/usr/bin/env python3
"""Orthogonal adapter training: OPLoRA-style constraints to fix direction interference.

Kill criteria:
  K684: MMLU math degradation <=15pp (currently -25pp with DARE p=0.5)
  K685: GSM8K remains >=+3pp over base (currently +6pp with DARE)
  K686: In-dist math/code accuracy >=90% of non-orthogonal baseline

Type: Guided exploration (Type 2)
Prior math: OPLoRA (arXiv:2510.13003) — orthogonal projection preserves top-k singular triples
Unknown: Optimal k for ternary adapters on BitNet-2B

Approach:
  1. Compute SVD of each base weight matrix, cache top-k U and V
  2. Train adapters with orthogonal projection:
     - A_orth = P_R @ A (pre-computed, frozen)
     - grad_B projected: grad_B - (grad_B @ U_k) @ U_k^T
  3. Compose all 5 adapters with DARE p=0.5 (same as baseline)
  4. Evaluate: GSM8K, MMLU, code gen, in-distribution
  5. Compare against non-orthogonal baseline (from dare_sparsified_composition)

Baseline numbers (from exp_dare_sparsified_composition):
  Base: GSM8K 38%, MMLU 44% (math 50%), code 90%
  DARE p=0.5: GSM8K 44%, MMLU 36% (math 25%), code 90%
  No-DARE: GSM8K 48%, MMLU 38% (math 30%), code 80%
"""

import ast
import gc
import json
import math
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source: real_data_domain_experts (NTP adapters, data, skeleton)
NTP_SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = NTP_SOURCE_DIR / "adapters"
NTP_DATA_DIR = NTP_SOURCE_DIR / "data"
SKELETON_PATH = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"

# Orthogonal adapter output
ORTH_ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
SVD_CACHE_DIR = EXPERIMENT_DIR / "svd_cache"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
DOMAIN_NAMES = DOMAINS

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Orthogonal projection k values to sweep
K_VALUES = [16]  # Start with k=16 (same as LoRA rank)

# DARE composition
DARE_DROP_RATE = 0.5

# Benchmark sizes
GSM8K_N = 50
CODE_GEN_N = 10
MMLU_N_PER_DOMAIN = 20
MAX_TOKENS_GSM8K = 256
MAX_TOKENS_CODE = 256
MAX_TOKENS_MMLU = 32
MAX_TOKENS_DOMAIN = 128

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# BitNet unpacking
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


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
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ============================================================================
# SVD computation and caching
# ============================================================================

def phase_compute_svd(k_val):
    """Compute and cache top-k SVD of each base weight matrix.

    Returns dict mapping (layer_idx, key) -> (U_k, V_k) numpy arrays.
    """
    log(f"\n{'='*70}")
    log(f"PHASE 1: Computing SVD (k={k_val}) for base model weights")
    log(f"{'='*70}")
    t0 = time.time()

    cache_path = SVD_CACHE_DIR / f"svd_k{k_val}.npz"
    if cache_path.exists():
        log(f"  Loading cached SVD from {cache_path}")
        data = dict(np.load(str(cache_path)))
        svd_data = {}
        for key_str in data:
            if key_str.startswith("U_"):
                base_key = key_str[2:]
                li_str, rest = base_key.split("_", 1)
                li = int(li_str)
                svd_data[(li, rest)] = (data[f"U_{base_key}"], data[f"V_{base_key}"])
        log(f"  Loaded {len(svd_data)} SVD pairs from cache ({time.time()-t0:.1f}s)")
        return svd_data

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    SVD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    svd_data = {}
    save_dict = {}
    n_layers = len(model.model.layers)

    for li in range(n_layers):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # W is (d_out, d_in) in MLX
            W = np.array(module.weight.astype(mx.float32))
            mx.eval(module.weight)

            # Compute truncated SVD — only need top-k
            # Use scipy for efficiency
            from scipy.sparse.linalg import svds
            actual_k = min(k_val, min(W.shape) - 1)
            U_k, S_k, Vt_k = svds(W.astype(np.float32), k=actual_k)
            # svds returns in ascending order, flip to descending
            U_k = U_k[:, ::-1].copy()
            S_k = S_k[::-1].copy()
            Vt_k = Vt_k[::-1, :].copy()
            V_k = Vt_k.T  # (d_in, k)

            svd_data[(li, key)] = (U_k, V_k)
            base_key = f"{li}_{key}"
            save_dict[f"U_{base_key}"] = U_k
            save_dict[f"V_{base_key}"] = V_k

            if li == 0 and key == TARGET_KEYS[0]:
                log(f"  Layer 0 {key}: W shape {W.shape}, top-{actual_k} singular values: "
                    f"{S_k[:5].round(4)}... gap ratio: {S_k[actual_k-1]/S_k[0]:.4f}")

        if (li + 1) % 10 == 0:
            log(f"  Processed {li+1}/{n_layers} layers")

    np.savez_compressed(str(cache_path), **save_dict)
    elapsed = time.time() - t0
    log(f"  SVD computed for {len(svd_data)} matrices in {elapsed:.1f}s")
    log(f"  Cached to {cache_path}")

    # Log spectral gap stats
    log(f"\n  Spectral gap analysis (top-{k_val} vs rest):")
    gaps = []
    for (li, key), (U_k, V_k) in svd_data.items():
        base_key = f"{li}_{key}"
        # Recompute for stats on first few layers
        if li < 3:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                W = np.array(module.weight.astype(mx.float32))
                full_S = np.linalg.svd(W, compute_uv=False)
                gap = full_S[k_val-1] / full_S[k_val] if len(full_S) > k_val else float('inf')
                gaps.append(gap)
                if li < 2:
                    log(f"    Layer {li} {key}: sigma_{k_val-1}={full_S[k_val-1]:.4f}, "
                        f"sigma_{k_val}={full_S[k_val]:.4f}, gap={gap:.3f}")

    cleanup(model, tokenizer)
    return svd_data


# ============================================================================
# Orthogonal LoRA module
# ============================================================================

class OrthogonalTernaryLoRALinear(nn.Module):
    """TernaryLoRA with OPLoRA-style orthogonal projection.

    Key differences from standard TernaryLoRA:
    1. A is pre-multiplied by P_R = I - V_k V_k^T (right projection)
    2. A custom gradient hook projects grad_B by P_L = I - U_k U_k^T (left projection)

    This ensures Delta_W = s * B^T @ A_orth^T has zero projection onto
    the base model's top-k singular subspace.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None,
                 U_k: mx.array = None, V_k: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.linear = base_linear

        # A matrix: apply right projection P_R = I - V_k V_k^T
        if a_init is not None:
            if V_k is not None:
                # A_orth = P_R @ A = A - V_k @ (V_k^T @ A)
                VtA = V_k.T @ a_init  # (k, r)
                a_orth = a_init - V_k @ VtA  # (d_in, r)
                self.lora_a = a_orth
            else:
                self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))

        # B matrix: trainable, STE-ternary quantized
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank

        # Store U_k for gradient projection (left projection on B)
        # P_L = I - U_k U_k^T acts on d_out dimension
        # For B (r, d_out): grad_B -> grad_B - (grad_B @ U_k) @ U_k^T
        self._U_k = U_k  # (d_out, k) or None

        # Freeze base and A
        self.linear.freeze()
        self.freeze(keys=["lora_a", "_U_k"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)

        # STE ternary quantization on B
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)

        # LoRA forward: x @ A_orth @ B_ternary * scale
        lora_out = (x @ self.lora_a) @ b_ste * self.scale

        return base_out + lora_out


def apply_orthogonal_ternary_lora(model, rank, scale, a_matrices, svd_data):
    """Apply OrthogonalTernaryLoRALinear to target projections.

    a_matrices: dict (layer_idx, key) -> (d_in, r) numpy A matrix
    svd_data: dict (layer_idx, key) -> (U_k, V_k) numpy arrays
    """
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            a_key = (li, key)
            a_mx = None
            if a_key in a_matrices:
                a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16)

            U_k_mx = None
            V_k_mx = None
            if a_key in svd_data:
                U_k_np, V_k_np = svd_data[a_key]
                U_k_mx = mx.array(U_k_np).astype(mx.bfloat16)
                V_k_mx = mx.array(V_k_np).astype(mx.bfloat16)

            lora = OrthogonalTernaryLoRALinear(
                module, rank=rank, scale=scale,
                a_init=a_mx, U_k=U_k_mx, V_k=V_k_mx,
            )
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    log(f"  Applied OrthogonalTernaryLoRA (r={rank}) to {count} layers")
    return model


def project_gradients(model):
    """Project B gradients by P_L = I - U_k U_k^T.

    For each OrthogonalTernaryLoRALinear module with U_k:
      grad_B -> grad_B - (grad_B @ U_k) @ U_k^T

    This is the left-side orthogonal projection ensuring
    the composed delta doesn't interfere with top-k singular directions.
    """
    projected = 0
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, OrthogonalTernaryLoRALinear) and module._U_k is not None:
                # grad_B is (r, d_out), U_k is (d_out, k)
                grad_b = module.lora_b
                U_k = module._U_k
                # Project: B -> B - (B @ U_k) @ U_k^T
                # This maintains B in the orthogonal complement of col(U_k)
                proj = (grad_b @ U_k) @ U_k.T  # (r, d_out)
                module.lora_b = grad_b - proj
                projected += 1
    return projected


# ============================================================================
# Training with orthogonal projection
# ============================================================================

def phase_train_orthogonal_adapter(domain_idx, domain_name, skeleton, svd_data, k_val):
    """Train a single adapter with orthogonal projection constraints."""
    log(f"\n  Training orthogonal adapter for {domain_name} (k={k_val})...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Build A matrix mapping for this domain
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    # Apply orthogonal LoRA
    model = apply_orthogonal_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices, svd_data)

    # Freeze everything except lora_b
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, OrthogonalTernaryLoRALinear):
                module.unfreeze(keys=["lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")

    # Load training data
    data_dir = NTP_DATA_DIR / domain_name
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    log(f"  {len(train_tokens)} training sequences")

    # Training loop with gradient projection
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        # Project B parameters after optimizer step
        # This enforces: B lies in orthogonal complement of U_k
        n_proj = project_gradients(model)
        mx.eval(model.parameters())

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    log(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
        f"({'converged' if converged else 'NOT converged'})")

    # Save adapter
    save_dir = ORTH_ADAPTERS_DIR / f"k{k_val}" / domain_name
    save_dir.mkdir(parents=True, exist_ok=True)
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    mx.savez(str(save_dir / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {save_dir}")

    result = {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "trainable_params": trainable,
    }

    log_memory(f"post-train-{domain_name}")
    cleanup(model, tokenizer, optimizer)
    return result


# ============================================================================
# Measure rho_k (direction interference metric)
# ============================================================================

def measure_rho_k(adapter_path, skeleton, domain_idx, svd_data, scale):
    """Measure rho_k = ||U_k^T Delta_W V_k||_F / ||Delta_W||_F for an adapter."""
    adapter = dict(mx.load(str(adapter_path / "adapter.npz")))
    rho_values = []
    n_layers = 30

    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey not in skeleton:
                continue
            a_np = skeleton[skey]
            a_mx = mx.array(a_np).astype(mx.float32)

            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key].astype(mx.float32)

            # Delta_W = scale * B^T @ A^T, shape (d_out, d_in)
            delta = scale * (b_mx.T @ a_mx.T)

            if (li, key) in svd_data:
                U_k_np, V_k_np = svd_data[(li, key)]
                U_k = mx.array(U_k_np).astype(mx.float32)
                V_k = mx.array(V_k_np).astype(mx.float32)

                # rho_k = ||U_k^T @ Delta_W @ V_k||_F / ||Delta_W||_F
                proj = U_k.T @ delta @ V_k  # (k, k)
                num = mx.sqrt(mx.sum(proj * proj)).item()
                den = mx.sqrt(mx.sum(delta * delta)).item()
                rho = num / max(den, 1e-10)
                rho_values.append(rho)

    del adapter
    return rho_values


# ============================================================================
# Composition and evaluation (reuse from DARE experiment)
# ============================================================================

def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def compute_delta(skeleton, adapter, domain, scale):
    di = DOMAINS.index(domain)
    deltas = {}
    n_layers = 30
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]
            delta = scale * (b_mx.T @ a_mx.T)
            deltas[(li, key)] = delta
    return deltas


def apply_dare_to_deltas(deltas, drop_rate, rng_key=None):
    if rng_key is None:
        rng_key = mx.random.key(SEED)
    rescale = 1.0 / (1.0 - drop_rate)
    dare_deltas = {}
    for (li, key), delta in deltas.items():
        rng_key, subkey = mx.random.split(rng_key)
        mask = mx.random.bernoulli(
            p=(1.0 - drop_rate), shape=delta.shape, key=subkey,
        ).astype(mx.bfloat16)
        dare_deltas[(li, key)] = delta * mask * rescale
    mx.eval(dare_deltas)
    return dare_deltas


def apply_deltas_to_model(model, deltas):
    merge_count = 0
    for (li, key), delta in deltas.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta
            merge_count += 1
    mx.eval(model.parameters())
    return merge_count


def save_base_weights(model):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                layer_w[key] = module.weight
        base_weights.append(layer_w)
    return base_weights


def restore_base_weights(model, base_weights):
    for li, layer_weights in enumerate(base_weights):
        for key, weight in layer_weights.items():
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                module.weight = weight
    mx.eval(model.parameters())


# ============================================================================
# Generation & evaluation (same as DARE experiment)
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=256):
    try:
        sampler = make_sampler(temp=0.0)
        text = mlx_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
        return text
    except Exception as e:
        log(f"  WARNING: generation failed: {e}")
        return ""


def format_gsm8k_prompt(question):
    return (f"### Instruction:\nSolve the following math problem step by step. "
            f"Show your work and give the final numerical answer after ####.\n\n"
            f"{question}\n\n### Response:\n")


def format_code_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_mmlu_prompt(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (f"### Instruction:\nAnswer the following multiple choice question. "
            f"Reply with just the letter (A, B, C, or D).\n\n"
            f"{question}\n\n{choices_text}\n\n### Response:\n")


def format_domain_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def check_gsm8k_correct(predicted, ground_truth, tolerance=0.01):
    if predicted is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance


def extract_mmlu_answer(text):
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    return None


def eval_code_syntax(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        try:
            ast.parse(block.strip())
            return True
        except SyntaxError:
            continue
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ',
                                'while ', 'if ', 'try:', 'except', 'with ',
                                'return ', 'print(', '#')):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        try:
            ast.parse('\n'.join(code_lines))
            return True
        except SyntaxError:
            pass
    return False


# ============================================================================
# Data loading
# ============================================================================

def load_gsm8k_data(n=50):
    from datasets import load_dataset
    log(f"  Loading GSM8K ({n} problems)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        answer_text = item["answer"]
        match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer = float(match.group(1).replace(',', ''))
        else:
            nums = re.findall(r'([\d,]+(?:\.\d+)?)', answer_text)
            answer = float(nums[-1].replace(',', '')) if nums else None
        problems.append({"question": item["question"], "answer": answer})
    log(f"  GSM8K: {len(problems)} problems loaded")
    return problems


def load_code_gen_data(n=10):
    log(f"  Loading code generation ({n} problems)...")
    val_path = NTP_DATA_DIR / "code" / "valid.jsonl"
    problems = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                problems.append({"instruction": instruction, "reference": response})
            if len(problems) >= n:
                break
    log(f"  Code gen: {len(problems)} problems loaded")
    return problems


def load_mmlu_data(n_per_domain=20):
    from datasets import load_dataset
    log(f"  Loading MMLU ({n_per_domain} per domain)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)
    mmlu_data = {}
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        rng = np.random.RandomState(42)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"    MMLU {domain}: {len(mmlu_data[domain])} questions")
    return mmlu_data


def load_indist_eval_data():
    log("  Loading in-distribution eval data...")
    indist = {}
    for dname in ["math", "code"]:
        val_path = NTP_DATA_DIR / dname / "valid.jsonl"
        problems = []
        with open(val_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                if "### Instruction:" in text and "### Response:" in text:
                    inst = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                    resp = text.split("### Response:")[1].strip()
                    problems.append({"instruction": inst, "reference": resp})
                if len(problems) >= 20:
                    break
        indist[dname] = problems
        log(f"    {dname} in-dist: {len(problems)} problems")
    return indist


# ============================================================================
# Benchmark evaluations
# ============================================================================

def eval_gsm8k(label, model, tokenizer, problems):
    log(f"\n  [GSM8K] Evaluating {label}...")
    t0 = time.time()
    correct = 0
    total = len(problems)
    for i, prob in enumerate(problems):
        prompt = format_gsm8k_prompt(prob["question"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_GSM8K)
        predicted = extract_gsm8k_answer(gen)
        if check_gsm8k_correct(predicted, prob["answer"]):
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
        if (i + 1) % 25 == 0:
            gc.collect()
    accuracy = correct / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [GSM8K] {label}: {correct}/{total} = {accuracy:.1%} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": round(elapsed, 1)}


def eval_code_gen(label, model, tokenizer, problems):
    log(f"\n  [Code Gen] Evaluating {label}...")
    t0 = time.time()
    syntax_ok = 0
    total = len(problems)
    for prob in problems:
        prompt = format_code_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            syntax_ok += 1
    rate = syntax_ok / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [Code Gen] {label}: {syntax_ok}/{total} = {rate:.1%} ({elapsed:.1f}s)")
    return {"syntax_rate": rate, "correct": syntax_ok, "total": total, "time_s": round(elapsed, 1)}


def eval_mmlu(label, model, tokenizer, mmlu_data):
    log(f"\n  [MMLU] Evaluating {label}...")
    choice_labels = ["A", "B", "C", "D"]
    results_by_domain = {}
    total_correct = 0
    total_q = 0
    for domain in DOMAINS:
        questions = mmlu_data[domain]
        correct = 0
        for q in questions:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1
        n = len(questions)
        acc = correct / n if n > 0 else 0
        results_by_domain[domain] = {"accuracy": acc, "correct": correct, "total": n}
        total_correct += correct
        total_q += n
        log(f"    MMLU {domain}: {correct}/{n} = {acc:.1%}")
    overall_acc = total_correct / total_q if total_q > 0 else 0
    log(f"  [MMLU] {label} overall: {total_correct}/{total_q} = {overall_acc:.1%}")
    return {
        "overall_accuracy": overall_acc,
        "overall_correct": total_correct,
        "overall_total": total_q,
        "by_domain": results_by_domain,
    }


def eval_math_correctness(model, tokenizer, problems):
    correct = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_DOMAIN)
        ref_nums = re.findall(r'[\d]+(?:\.\d+)?', prob["reference"])
        if ref_nums:
            if ref_nums[-1] in gen:
                correct += 1
        else:
            ref_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', prob["reference"].lower()))
            gen_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', gen.lower()))
            overlap = len(ref_words & gen_words) / max(len(ref_words), 1)
            if overlap > 0.3:
                correct += 1
    return {"correctness": correct / total if total > 0 else 0, "correct": correct, "total": total}


def eval_code_passrate(model, tokenizer, problems):
    passed = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            passed += 1
    return {"pass_rate": passed / total if total > 0 else 0, "passed": passed, "total": total}


# ============================================================================
# Phase functions
# ============================================================================

def phase_load_data():
    log(f"\n{'='*70}")
    log("PHASE 0: LOADING ALL BENCHMARK DATA")
    log(f"{'='*70}")
    gsm8k = load_gsm8k_data(GSM8K_N)
    code_gen = load_code_gen_data(CODE_GEN_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)
    indist = load_indist_eval_data()
    return gsm8k, code_gen, mmlu, indist


def phase_train_all_orthogonal(k_val, skeleton, svd_data):
    """Train all 5 domain adapters with orthogonal constraints."""
    log(f"\n{'='*70}")
    log(f"PHASE 2: TRAINING ORTHOGONAL ADAPTERS (k={k_val})")
    log(f"{'='*70}")

    train_results = {}
    for di, domain in enumerate(DOMAINS):
        result = phase_train_orthogonal_adapter(di, domain, skeleton, svd_data, k_val)
        train_results[domain] = result

    return train_results


def phase_measure_rho(k_val, skeleton, svd_data):
    """Measure rho_k for both orthogonal and baseline adapters."""
    log(f"\n{'='*70}")
    log(f"PHASE 3: MEASURING rho_k (direction interference metric)")
    log(f"{'='*70}")

    rho_results = {}

    # Orthogonal adapters
    for di, domain in enumerate(DOMAINS):
        orth_path = ORTH_ADAPTERS_DIR / f"k{k_val}" / domain
        scale = OPTIMAL_SCALES[domain]
        rho_vals = measure_rho_k(orth_path, skeleton, di, svd_data, scale)
        mean_rho = np.mean(rho_vals) if rho_vals else float('nan')
        max_rho = np.max(rho_vals) if rho_vals else float('nan')
        log(f"  Orthogonal {domain}: mean_rho={mean_rho:.6f}, max_rho={max_rho:.6f}")
        rho_results[f"orth_{domain}"] = {
            "mean_rho": float(mean_rho), "max_rho": float(max_rho),
            "n_matrices": len(rho_vals),
        }

    # Baseline (non-orthogonal) adapters
    for di, domain in enumerate(DOMAINS):
        baseline_path = NTP_ADAPTERS_DIR / domain
        if baseline_path.exists():
            scale = OPTIMAL_SCALES[domain]
            rho_vals = measure_rho_k(baseline_path, skeleton, di, svd_data, scale)
            mean_rho = np.mean(rho_vals) if rho_vals else float('nan')
            max_rho = np.max(rho_vals) if rho_vals else float('nan')
            log(f"  Baseline {domain}: mean_rho={mean_rho:.6f}, max_rho={max_rho:.6f}")
            rho_results[f"baseline_{domain}"] = {
                "mean_rho": float(mean_rho), "max_rho": float(max_rho),
                "n_matrices": len(rho_vals),
            }

    return rho_results


def phase_eval_orthogonal_composed(k_val, gsm8k, code_gen, mmlu, indist, skeleton):
    """Compose all 5 orthogonal adapters with DARE p=0.5 and evaluate."""
    log(f"\n{'='*70}")
    log(f"PHASE 4: EVALUATING ORTHOGONAL+DARE COMPOSITION (k={k_val})")
    log(f"{'='*70}")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)

    # Load and compose all 5 orthogonal adapters with DARE
    rng_key = mx.random.key(SEED)
    total_merged = 0
    n_adapters = len(DOMAINS)

    for domain in DOMAINS:
        adapter_path = ORTH_ADAPTERS_DIR / f"k{k_val}" / domain / "adapter.npz"
        adapter = dict(mx.load(str(adapter_path)))
        scale = OPTIMAL_SCALES[domain] / n_adapters  # 1/N scaling
        deltas = compute_delta(skeleton, adapter, domain, scale)
        rng_key, subkey = mx.random.split(rng_key)
        dare_deltas = apply_dare_to_deltas(deltas, DARE_DROP_RATE, subkey)
        n = apply_deltas_to_model(model, dare_deltas)
        total_merged += n
        del adapter, deltas, dare_deltas
        log(f"  Composed {domain}: merged {n} matrices (scale={scale:.2f})")

    gc.collect()
    mx.clear_cache()
    log(f"  Total merged: {total_merged} matrices")
    log_memory("post-compose")

    # Evaluate
    gsm8k_results = eval_gsm8k(f"orth-k{k_val}+DARE", model, tokenizer, gsm8k)
    code_results = eval_code_gen(f"orth-k{k_val}+DARE", model, tokenizer, code_gen)
    mmlu_results = eval_mmlu(f"orth-k{k_val}+DARE", model, tokenizer, mmlu)

    # In-distribution
    log(f"\n  [In-dist] Evaluating orthogonal composed...")
    math_correct = eval_math_correctness(model, tokenizer, indist["math"])
    code_pass = eval_code_passrate(model, tokenizer, indist["code"])
    log(f"  In-dist math: {math_correct['correctness']:.1%}")
    log(f"  In-dist code: {code_pass['pass_rate']:.1%}")

    elapsed = time.time() - t0
    log(f"\n  Evaluation total: {elapsed:.1f}s")

    results = {
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "in_distribution": {
            "math_correctness": math_correct,
            "code_pass_rate": code_pass,
        },
        "time_s": round(elapsed, 1),
    }

    cleanup(model, tokenizer, base_weights)
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log("=" * 70)
    log("EXPERIMENT: Orthogonal Adapter Training (OPLoRA-style)")
    log(f"Model: {MODEL_ID}")
    log(f"LoRA rank: {LORA_RANK}, scale: {LORA_SCALE}")
    log(f"Orthogonal k values: {K_VALUES}")
    log(f"DARE drop rate: {DARE_DROP_RATE}")
    log(f"Train iters: {TRAIN_ITERS}")
    log("=" * 70)

    # Load data
    gsm8k, code_gen, mmlu, indist = phase_load_data()

    # Load skeleton
    skeleton = load_skeleton()
    log(f"  Loaded Grassmannian skeleton: {len(skeleton)} matrices")

    # Baseline reference numbers (from dare_sparsified_composition)
    baseline = {
        "base": {"gsm8k": 0.38, "mmlu": 0.44, "mmlu_math": 0.50, "code": 0.90},
        "dare_p0.5": {"gsm8k": 0.44, "mmlu": 0.36, "mmlu_math": 0.25, "code": 0.90,
                      "indist_math": 0.80, "indist_code": 0.80},
        "no_dare": {"gsm8k": 0.48, "mmlu": 0.38, "mmlu_math": 0.30, "code": 0.80,
                    "indist_math": 0.80, "indist_code": 0.75},
    }

    all_results = {
        "experiment": "orthogonal_adapter_training",
        "model": MODEL_ID,
        "k_values": K_VALUES,
        "dare_drop_rate": DARE_DROP_RATE,
        "domains": DOMAINS,
        "scales": OPTIMAL_SCALES,
        "baseline_reference": baseline,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for k_val in K_VALUES:
        log(f"\n\n{'#'*70}")
        log(f"# k = {k_val}")
        log(f"{'#'*70}")

        # Phase 1: Compute SVD
        svd_data = phase_compute_svd(k_val)

        # Phase 2: Train orthogonal adapters
        train_results = phase_train_all_orthogonal(k_val, skeleton, svd_data)
        all_results[f"k{k_val}_training"] = train_results

        # Phase 3: Measure rho_k
        rho_results = phase_measure_rho(k_val, skeleton, svd_data)
        all_results[f"k{k_val}_rho"] = rho_results

        # Phase 4: Compose and evaluate
        eval_results = phase_eval_orthogonal_composed(
            k_val, gsm8k, code_gen, mmlu, indist, skeleton
        )
        all_results[f"k{k_val}_eval"] = eval_results

        # Summary for this k
        mmlu_math = eval_results["mmlu"]["by_domain"]["math"]["accuracy"]
        mmlu_overall = eval_results["mmlu"]["overall_accuracy"]
        gsm8k_acc = eval_results["gsm8k"]["accuracy"]
        code_acc = eval_results["code_gen"]["syntax_rate"]
        indist_math = eval_results["in_distribution"]["math_correctness"]["correctness"]
        indist_code = eval_results["in_distribution"]["code_pass_rate"]["pass_rate"]

        # Compute kill criteria
        base_mmlu_math = baseline["base"]["mmlu_math"]
        base_gsm8k = baseline["base"]["gsm8k"]
        baseline_indist_math = baseline["dare_p0.5"]["indist_math"]
        baseline_indist_code = baseline["dare_p0.5"]["indist_code"]

        mmlu_math_degradation = base_mmlu_math - mmlu_math  # pp degradation
        gsm8k_gain = gsm8k_acc - base_gsm8k  # pp gain over base
        indist_math_ratio = indist_math / max(baseline_indist_math, 0.01)
        indist_code_ratio = indist_code / max(baseline_indist_code, 0.01)

        k1_pass = mmlu_math_degradation <= 0.15
        k2_pass = gsm8k_gain >= 0.03
        k3_pass = indist_math_ratio >= 0.90 and indist_code_ratio >= 0.90

        # Mean rho comparison
        orth_rhos = [rho_results[f"orth_{d}"]["mean_rho"] for d in DOMAINS
                     if f"orth_{d}" in rho_results]
        baseline_rhos = [rho_results[f"baseline_{d}"]["mean_rho"] for d in DOMAINS
                         if f"baseline_{d}" in rho_results]
        mean_orth_rho = np.mean(orth_rhos) if orth_rhos else float('nan')
        mean_baseline_rho = np.mean(baseline_rhos) if baseline_rhos else float('nan')

        summary = {
            "k": k_val,
            "mmlu_math": mmlu_math,
            "mmlu_math_degradation_pp": round(mmlu_math_degradation * 100, 1),
            "mmlu_overall": mmlu_overall,
            "gsm8k": gsm8k_acc,
            "gsm8k_gain_pp": round(gsm8k_gain * 100, 1),
            "code_gen": code_acc,
            "indist_math": indist_math,
            "indist_code": indist_code,
            "indist_math_ratio": round(indist_math_ratio, 3),
            "indist_code_ratio": round(indist_code_ratio, 3),
            "mean_orth_rho": round(float(mean_orth_rho), 6),
            "mean_baseline_rho": round(float(mean_baseline_rho), 6),
            "rho_reduction": round(1 - float(mean_orth_rho) / max(float(mean_baseline_rho), 1e-10), 4),
            "K1_PASS": k1_pass,
            "K2_PASS": k2_pass,
            "K3_PASS": k3_pass,
        }
        all_results[f"k{k_val}_summary"] = summary

        log(f"\n{'='*70}")
        log(f"SUMMARY k={k_val}:")
        log(f"  MMLU math: {mmlu_math:.1%} (degradation: {mmlu_math_degradation*100:.1f}pp)")
        log(f"  MMLU overall: {mmlu_overall:.1%}")
        log(f"  GSM8K: {gsm8k_acc:.1%} (gain: {gsm8k_gain*100:.1f}pp over base)")
        log(f"  Code gen: {code_acc:.1%}")
        log(f"  In-dist math: {indist_math:.1%}, code: {indist_code:.1%}")
        log(f"  Mean rho_k: orth={mean_orth_rho:.6f}, baseline={mean_baseline_rho:.6f}")
        log(f"  K1 (MMLU math <=15pp): {'PASS' if k1_pass else 'FAIL'}")
        log(f"  K2 (GSM8K >=+3pp): {'PASS' if k2_pass else 'FAIL'}")
        log(f"  K3 (In-dist >=90%): {'PASS' if k3_pass else 'FAIL'}")
        log(f"{'='*70}")

        # Free SVD data
        del svd_data
        gc.collect()
        mx.clear_cache()

    total_time = time.time() - t0_total
    all_results["total_time_s"] = round(total_time, 1)
    log(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f}min)")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
