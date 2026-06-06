#!/usr/bin/env python3
"""
Spectral Surgery: Training-Free LoRA Refinement as Evolve Quality Gate

Implements the Spectral Surgery algorithm (arXiv 2603.03995) on BitNet-2B-4T
LoRA adapters trained in experiments/models/bitnet_2b_real_composition/.

Algorithm:
  For each LoRA layer (B @ A):
    1. Compute delta_W = B @ A  (d_out x d_in)
    2. SVD: delta_W = U * diag(sigma) * V^T
    3. Forward pass on calibration set, compute gradient G = dL/d(delta_W)
    4. Sensitivity per singular value: g_k = u_k^T @ G @ v_k
    5. Signed update: sigma'_k = sigma_k * exp(-(eta_sup * g+_k + eta_amp * g-_k))
    6. Nuclear-norm renormalization: sigma' *= ||sigma||_1 / ||sigma'||_1
    7. Reconstruct: B_new = U * diag(sqrt(sigma')), A_new = diag(sqrt(sigma')) * V^T

Kill criteria:
  K1: spectral-refined adapter PPL not better than unrefined (surgery doesn't help)
  K2: spectral-refined adapter KR-Test not better than unrefined
  K3: SVD + reweighting takes >5 min per adapter (too slow for production gate)

Runtime: ~60-90 min on Apple Silicon (MLX)
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
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
MAX_SEQ_LENGTH = 256
N_CAL = 128          # calibration examples for gradient estimation
N_KR_PAIRS = 50      # contrastive pairs for KR-Test
ETA_SUP = 1.0        # suppress detrimental directions (positive gradient)
ETA_AMP = 0.5        # amplify beneficial directions (negative gradient)
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
COMPOSITION_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition"
ADAPTERS_DIR = COMPOSITION_DIR / "adapters"
DATA_DIR = COMPOSITION_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REFINED_DIR = EXPERIMENT_DIR / "adapters_refined"

DOMAINS = ["python", "math", "medical", "legal", "creative"]


def log(msg):
    print(msg, flush=True)


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


# ===========================================================================
# LoRA helpers (from bitnet_2b_real_composition)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(params: dict, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25):
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
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# KR-Test (contrastive evaluation)
# ===========================================================================
def generate_kr_pairs(data_path: Path, n_pairs: int = N_KR_PAIRS):
    """Generate cross-item contrastive pairs from validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return []
    texts = []
    with open(valid_path) as f:
        for line in f:
            t = json.loads(line)["text"]
            if len(t.strip()) > 30:
                texts.append(t.strip())
    if len(texts) < 2:
        return []
    random.seed(SEED)
    pairs = []
    n = min(len(texts), n_pairs)
    for i in range(n):
        j = (i + max(1, len(texts) // 3)) % len(texts)
        if texts[i] == texts[j]:
            j = (j + 1) % len(texts)
        # Context: first 50% of correct text; correct continuation: rest
        # Wrong continuation: corresponding portion of wrong text
        correct = texts[i]
        wrong = texts[j]
        split_point = len(correct) // 2
        if split_point < 20:
            continue
        context = correct[:split_point]
        correct_cont = correct[split_point:]
        wrong_cont = wrong[split_point:split_point + len(correct_cont)]
        if len(correct_cont) < 10 or len(wrong_cont) < 10:
            continue
        pairs.append((context, correct_cont, wrong_cont))
    return pairs


def compute_kr_score(model, tokenizer, pairs):
    """KR-Test: fraction of pairs where model prefers correct continuation."""
    if not pairs:
        return 0.5  # chance
    correct_count = 0
    total = 0
    for context, correct_cont, wrong_cont in pairs:
        # Tokenize
        ctx_tokens = tokenizer.encode(context)
        correct_tokens = tokenizer.encode(correct_cont)
        wrong_tokens = tokenizer.encode(wrong_cont)
        if len(correct_tokens) < 2 or len(wrong_tokens) < 2:
            continue
        # Truncate
        max_cont = min(len(correct_tokens), len(wrong_tokens), 64)
        correct_tokens = correct_tokens[:max_cont]
        wrong_tokens = wrong_tokens[:max_cont]
        # Compute log-prob for correct
        full_correct = ctx_tokens + correct_tokens
        full_correct = full_correct[:MAX_SEQ_LENGTH + 1]
        x = mx.array(full_correct[:-1])[None, :]
        y = mx.array(full_correct[1:])[None, :]
        logits = model(x)
        # Only score continuation tokens
        start = len(ctx_tokens) - 1
        if start >= logits.shape[1]:
            continue
        cont_logits = logits[:, start:, :]
        cont_y = y[:, start:]
        loss_correct = nn.losses.cross_entropy(cont_logits, cont_y, reduction="sum")
        mx.eval(loss_correct)
        lp_correct = -loss_correct.item()
        # Compute log-prob for wrong
        full_wrong = ctx_tokens + wrong_tokens
        full_wrong = full_wrong[:MAX_SEQ_LENGTH + 1]
        x = mx.array(full_wrong[:-1])[None, :]
        y = mx.array(full_wrong[1:])[None, :]
        logits = model(x)
        cont_logits = logits[:, start:, :]
        cont_y = y[:, start:]
        loss_wrong = nn.losses.cross_entropy(cont_logits, cont_y, reduction="sum")
        mx.eval(loss_wrong)
        lp_wrong = -loss_wrong.item()
        if lp_correct > lp_wrong:
            correct_count += 1
        total += 1
    return correct_count / max(total, 1)


# ===========================================================================
# Spectral Surgery Core
# ===========================================================================
def spectral_surgery_adapter(
    model, tokenizer, adapter_params, cal_texts,
    eta_sup=ETA_SUP, eta_amp=ETA_AMP,
):
    """Apply Spectral Surgery to a loaded adapter.

    For each LoRA layer pair (lora_a, lora_b):
      1. Compute delta_W = lora_b @ lora_a (factoring in the LoRA scale)
      2. SVD decomposition
      3. Compute per-singular-value gradient sensitivity on calibration set
      4. Reweight singular values
      5. Nuclear-norm renormalize
      6. Reconstruct B_new, A_new

    Returns: new adapter_params dict with refined weights.
    """
    t_start = time.time()

    # First, load the adapter into the model
    zero_lora_params(model)
    apply_adapter_weights(model, adapter_params)
    mx.eval(model.parameters())

    # Identify all LoRA layer pairs
    lora_layers = {}  # key_prefix -> (lora_a_key, lora_b_key)
    for key in adapter_params:
        if "lora_a" in key:
            prefix = key.replace("lora_a", "")
            b_key = key.replace("lora_a", "lora_b")
            if b_key in adapter_params:
                lora_layers[prefix] = (key, b_key)

    log(f"    Found {len(lora_layers)} LoRA layer pairs to refine")

    # Tokenize calibration data
    cal_tokens = []
    for text in cal_texts[:N_CAL]:
        tokens = tokenizer.encode(text)
        if len(tokens) > 2:
            cal_tokens.append(tokens[:MAX_SEQ_LENGTH + 1])

    log(f"    Using {len(cal_tokens)} calibration examples")

    # For each LoRA layer, compute sensitivity and reweight
    refined_params = dict(adapter_params)  # copy

    # We need gradient w.r.t. the LoRA parameters
    # Strategy: compute gradient of loss w.r.t. lora_a and lora_b for each layer,
    # then derive gradient w.r.t. delta_W = B @ A using chain rule:
    #   g_k = u_k^T @ (dL/d(delta_W)) @ v_k
    # where dL/d(delta_W) can be computed from dL/dB and dL/dA:
    #   dL/d(delta_W) is reconstructed from dL/dB @ A + B @ dL/dA... not quite.
    #
    # Actually, dL/d(delta_W) = dL/dY @ X^T (where Y = W@X + delta_W@X)
    # But we don't have direct access to delta_W as a parameter.
    #
    # Simpler approach: since delta_W = B @ A, and we have gradients w.r.t. B and A:
    #   dL/dB = dL/d(delta_W) @ A^T  =>  dL/d(delta_W) = dL/dB @ (A^T)^{-1}
    # But A is not square. Instead, use the pseudo-inverse or direct computation.
    #
    # BEST approach for MLX: compute loss, get gradients for lora_a and lora_b,
    # then reconstruct G = grad_B @ pinv(A) or equivalently compute G directly.
    #
    # Actually the cleanest: G = dL/d(delta_W). Since delta_W = B @ A and the
    # output is y = (W + B@A) @ x, we have dL/d(delta_W) = dL/dy @ x^T aggregated.
    # But this requires hooks which MLX doesn't support well.
    #
    # PRACTICAL APPROACH: Use the fact that for LoRA, the gradient relationship is:
    #   dL/dB = G @ A^T   (where G = dL/d(delta_W))
    #   dL/dA = B^T @ G
    # So: G = dL/dB @ pinv(A^T) = dL/dB @ A @ (A^T @ A)^{-1}
    # Since A is (r x d_in) with r << d_in, A @ A^T is (r x r), invertible.
    # G = dL/dB @ A @ inv(A @ A^T)... wait that gives wrong dims.
    #
    # Let me be careful:
    #   B is (d_out x r), A is (r x d_in)
    #   dL/dB is (d_out x r), dL/dA is (r x d_in)
    #   G = dL/d(delta_W) is (d_out x d_in)
    #   dL/dB = G @ A^T  =>  G = dL/dB @ pinv(A^T)
    #   pinv(A^T) = (A @ A^T)^{-1} @ A  (since A^T is (d_in x r), right pseudo-inverse)
    #   Actually pinv(A^T) = A @ (A^T @ A)^{-1}... no.
    #   A^T is (d_in x r). pinv(A^T) = (A @ A^T)^{-1} @ A  where A@A^T is (r x r).
    #   Wait: pinv(M) where M is (m x n) with m > n: pinv(M) = (M^T M)^{-1} M^T
    #   A^T is (d_in x r), so pinv(A^T) = (A @ A^T)^{-1} @ A  (r x d_in)
    #   G = dL/dB @ pinv(A^T) = (d_out x r) @ (r x d_in) = (d_out x d_in). Correct!
    #
    # EVEN SIMPLER: We don't need the full G matrix. We need:
    #   g_k = u_k^T @ G @ v_k
    # where U, sigma, V^T come from SVD(B @ A).
    # G = dL/dB @ (A @ A^T)^{-1} @ A
    # g_k = u_k^T @ dL/dB @ (A @ A^T)^{-1} @ A @ v_k
    #
    # But we can also use the other equation:
    #   dL/dA = B^T @ G  =>  G = pinv(B^T) @ dL/dA = B @ (B^T @ B)^{-1} @ dL/dA
    #   (since B^T is (r x d_out), pinv = B @ (B^T @ B)^{-1})
    #
    # For numerical stability, average both estimates.
    # OR: just use the SVD-based shortcut.
    #
    # CLEANEST APPROACH using SVD of delta_W = U @ S @ V^T:
    #   B_svd = U @ sqrt(S), A_svd = sqrt(S) @ V^T  (one possible factorization)
    #   Then dL/dB_svd = G @ A_svd^T = G @ V @ sqrt(S)
    #   And dL/dA_svd = B_svd^T @ G = sqrt(S) @ U^T @ G
    #   So g_k = u_k^T @ G @ v_k = (1/sqrt(s_k)) * (dL/dA_svd)[k, :] @ v_k
    #   Hmm, this gets circular.
    #
    # THE ACTUAL SIMPLEST: replace the LoRA layer with a parameterization
    # that has sigma as explicit parameters. But that requires model surgery.
    #
    # PRAGMATIC: Use finite differences or direct gradient computation.
    # For each singular value sigma_k, perturb it and measure loss change.
    # This is O(r * N_cal) forward passes per layer = 16 * 128 = 2048 per layer.
    # With 210 layers, that's 430K forward passes. Too expensive.
    #
    # REAL SOLUTION: Compute dL/dB and dL/dA via backprop (one pass per cal example),
    # then reconstruct g_k analytically.

    # Aggregate gradients over calibration set
    log(f"    Computing gradients on calibration set...")

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Accumulate gradients
    grad_accum = {}
    n_cal_used = 0
    for tokens in cal_tokens[:N_CAL]:
        if len(tokens) < 3:
            continue
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss)

        # Extract LoRA gradients
        flat_grads = dict(tree_flatten(grads))
        for key in adapter_params:
            if key in flat_grads:
                g = flat_grads[key]
                if key not in grad_accum:
                    grad_accum[key] = g
                else:
                    grad_accum[key] = grad_accum[key] + g
        mx.eval(grad_accum)
        n_cal_used += 1

        if n_cal_used % 20 == 0:
            log(f"      Processed {n_cal_used}/{len(cal_tokens)} calibration examples")

    # Average gradients
    for key in grad_accum:
        grad_accum[key] = grad_accum[key] / n_cal_used
    mx.eval(grad_accum)

    log(f"    Computed average gradients from {n_cal_used} examples")

    # Now apply spectral surgery per LoRA layer pair
    n_refined = 0
    total_sv_edits = 0

    for prefix, (a_key, b_key) in lora_layers.items():
        A = adapter_params[a_key]   # (r, d_in) or (d_in, r) depending on convention
        B = adapter_params[b_key]   # (d_out, r) or (r, d_out)

        grad_A = grad_accum.get(a_key)
        grad_B = grad_accum.get(b_key)

        if grad_A is None or grad_B is None:
            continue

        # In MLX LoRALinear: output = x @ lora_a @ lora_b * scale + x @ W
        # lora_a is (d_in, r), lora_b is (r, d_out)
        # delta_W_eff = lora_a @ lora_b * scale  (d_in, d_out)
        # For SVD we work with delta_W = A @ B * scale

        delta_W = (A @ B * LORA_SCALE).astype(mx.float32)
        mx.eval(delta_W)

        # SVD
        # delta_W is (d_in, d_out) -- we need U (d_in, r), S (r,), Vt (r, d_out)
        # Use full SVD then truncate to rank r
        r = A.shape[1]  # rank

        # For efficiency, since delta_W is already rank-r, SVD gives at most r
        # non-zero singular values
        try:
            U, S, Vt = mx.linalg.svd(delta_W, stream=mx.cpu)
            mx.eval(U, S, Vt)
        except Exception as e:
            log(f"    SVD failed for {prefix}: {e}, skipping")
            continue

        # Truncate to rank r
        U_r = U[:, :r]      # (d_in, r)
        S_r = S[:r]          # (r,)
        Vt_r = Vt[:r, :]    # (r, d_out)

        # Compute gradient of loss w.r.t delta_W:
        # dL/d(lora_a) = x^T @ dL/dy @ (lora_b * scale)^T  -- but we have the aggregated gradient
        # Actually in MLX, the gradient of loss w.r.t. lora_a and lora_b is computed by autograd.
        #
        # Reconstruct G = dL/d(delta_W) from dL/d(lora_a) and dL/d(lora_b):
        # Since delta_W = A @ B * scale:
        #   dL/dA = dL/d(delta_W) @ (B * scale)^T = G @ B^T * scale
        #   dL/dB = A^T @ dL/d(delta_W) * scale = A^T @ G * scale
        #
        # From dL/dA: G = dL/dA @ pinv(B^T * scale) = (1/scale) * dL/dA @ pinv(B^T)
        #   B^T is (d_out, r), pinv(B^T) = B @ (B^T @ B)^{-1}  ... (r, d_out) -> wrong
        #   Actually pinv(B^T) where B^T is (d_out, r): pinv = (B @ B^T)^{-1} @ B ... no
        #   For M (m x n), m > n: pinv(M) = (M^T M)^{-1} M^T
        #   B^T is (d_out, r), d_out > r: pinv(B^T) = (B @ B^T)^{-1} @ B   (r x r)^{-1} @ (r x d_out) = (r x d_out)
        #   Hmm wait: G = dL/dA @ pinv(B^T * scale)
        #   dL/dA is (d_in, r), pinv(B^T * scale) is (r, d_out)
        #   G = (d_in, r) @ (r, d_out) = (d_in, d_out). Correct!

        gA = grad_A.astype(mx.float32)
        gB = grad_B.astype(mx.float32)
        mx.eval(gA, gB)

        # Compute g_k = u_k^T @ G @ v_k without materializing full G.
        # G = (1/scale) * dL/dA @ pinv(B^T)
        # pinv(B^T) = (B @ B^T)^{-1} @ B where B is (r, d_out)
        # g_k = (1/scale) * u_k^T @ dL/dA @ (B@B^T)^{-1} @ B @ v_k
        #
        # Compute in steps:
        # step1: p = B @ v_k  (r,) for each k -- but vectorized: P = B @ Vt_r^T = B @ V_r  (r, r)
        # step2: q = (B@B^T)^{-1} @ P  (r, r)
        # step3: h = dL/dA @ q  (d_in, r)
        # step4: g = U_r^T @ h  (r, r) -- diagonal gives g_k
        #
        # Actually: g_k = (1/scale) * (U_r^T @ dL/dA @ (B@B^T)^{-1} @ B @ Vt_r^T)_{kk}
        # This is the diagonal of a (r x r) matrix. Very cheap!

        B_f32 = B.astype(mx.float32)
        BBt = B_f32 @ B_f32.T  # (r, r)
        BBt_reg = BBt + 1e-6 * mx.eye(r)
        BBt_inv = mx.linalg.inv(BBt_reg, stream=mx.cpu)  # (r, r) on CPU
        mx.eval(BBt_inv)

        # P = B @ Vt_r^T  (r, r)
        V_r = Vt_r.T  # (d_out, r)
        P = B_f32 @ V_r  # (r, r)
        # Q = BBt_inv @ P  (r, r)
        Q = BBt_inv @ P  # (r, r)
        # H = dL/dA @ Q  (d_in, r)
        H = gA @ Q  # (d_in, r)
        # sensitivities = diag(U_r^T @ H) / scale
        UtH = U_r.T @ H  # (r, r)
        mx.eval(UtH)
        sensitivities_arr = mx.diag(UtH) / LORA_SCALE  # (r,)
        mx.eval(sensitivities_arr)
        sensitivities = [sensitivities_arr[k].item() for k in range(r)]

        # Normalize sensitivities
        s_array = mx.array(sensitivities, dtype=mx.float32)
        s_norm = s_array / (mx.sqrt(mx.sum(s_array ** 2)) + 1e-10)
        mx.eval(s_norm)

        # Signed update
        g_pos = mx.maximum(s_norm, 0.0)   # detrimental (positive = loss increases when amplified)
        g_neg = mx.maximum(-s_norm, 0.0)  # beneficial (negative = loss decreases when amplified)
        g_eff = eta_sup * g_pos + eta_amp * g_neg
        mx.eval(g_eff)

        # Reweight: sigma'_k = sigma_k * exp(-g_eff_k)
        S_new = S_r * mx.exp(-g_eff)

        # Clamp to non-negative
        S_new = mx.maximum(S_new, 0.0)

        # Nuclear-norm renormalization: preserve ||sigma||_1
        l1_orig = mx.sum(mx.abs(S_r))
        l1_new = mx.sum(mx.abs(S_new))
        S_new = S_new * (l1_orig / (l1_new + 1e-10))
        mx.eval(S_new)

        # Reconstruct LoRA factors: B_new @ A_new = delta_W'
        # delta_W' = U_r @ diag(S_new) @ Vt_r
        # Split: A_new = diag(sqrt(S_new / scale)) @ Vt_r, B_new = U_r @ diag(sqrt(S_new / scale))
        # Wait, need to account for the LORA_SCALE factor.
        # Original: effective delta = A @ B * scale = delta_W
        # We want: A_new @ B_new * scale = U_r @ diag(S_new) @ Vt_r
        # So: A_new @ B_new = U_r @ diag(S_new/scale) @ Vt_r
        # Split evenly: A_new = U_r @ diag(sqrt(S_new/scale)), B_new = diag(sqrt(S_new/scale)) @ Vt_r

        sqrt_s = mx.sqrt(S_new / LORA_SCALE + 1e-10)
        A_new = U_r * sqrt_s[None, :]      # (d_in, r) * (1, r) = (d_in, r)
        B_new = sqrt_s[:, None] * Vt_r     # (r, 1) * (r, d_out) = (r, d_out)
        mx.eval(A_new, B_new)

        refined_params[a_key] = A_new.astype(mx.bfloat16)
        refined_params[b_key] = B_new.astype(mx.bfloat16)
        n_refined += 1
        total_sv_edits += r

    mx.eval(refined_params)
    surgery_time = time.time() - t_start
    log(f"    Refined {n_refined} layers, {total_sv_edits} singular value edits in {surgery_time:.1f}s")

    return refined_params, surgery_time


# ===========================================================================
# Composition
# ===========================================================================
def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_spectral_surgery",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "n_cal": N_CAL,
        "eta_sup": ETA_SUP,
        "eta_amp": ETA_AMP,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("Spectral Surgery: Training-Free LoRA Refinement")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    log("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    log("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable parameters: {trainable:,}")
    results["setup_time_s"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------------
    # Phase 1: Load adapters and compute baseline PPL
    # ------------------------------------------------------------------
    log("\n[Phase 1] Loading adapters and computing baseline PPL...")
    adapters = {}
    baseline_ppls = {}

    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain
        if not adapter_path.exists():
            log(f"  WARNING: {domain} adapter not found at {adapter_path}")
            continue

        params = load_adapter(adapter_path)
        adapters[domain] = params

        # Compute baseline PPL
        zero_lora_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
        baseline_ppls[domain] = ppl
        log(f"  {domain}: baseline PPL = {ppl:.2f}")

    results["baseline_ppls"] = baseline_ppls

    # ------------------------------------------------------------------
    # Phase 2: Compute baseline KR-Test
    # ------------------------------------------------------------------
    log("\n[Phase 2] Computing baseline KR-Test scores...")
    baseline_kr = {}
    kr_pairs_cache = {}

    for domain in adapters:
        pairs = generate_kr_pairs(DATA_DIR / domain)
        kr_pairs_cache[domain] = pairs

        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain])
        mx.eval(model.parameters())

        kr_score = compute_kr_score(model, tokenizer, pairs)
        baseline_kr[domain] = kr_score
        log(f"  {domain}: baseline KR-Test = {kr_score:.3f} ({len(pairs)} pairs)")

    results["baseline_kr"] = baseline_kr

    # Also compute base model KR (no adapter)
    log("  Computing base model KR-Test...")
    base_kr = {}
    zero_lora_params(model)
    # Reset to zero LoRA = base model behavior
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())

    for domain in adapters:
        kr_score = compute_kr_score(model, tokenizer, kr_pairs_cache[domain])
        base_kr[domain] = kr_score
        log(f"  {domain}: base model KR = {kr_score:.3f}")
    results["base_kr"] = base_kr

    # ------------------------------------------------------------------
    # Phase 3: Apply Spectral Surgery per adapter
    # ------------------------------------------------------------------
    log("\n[Phase 3] Applying Spectral Surgery...")
    refined_adapters = {}
    surgery_times = {}
    refined_ppls = {}
    refined_kr = {}

    for domain in adapters:
        log(f"\n  --- Spectral Surgery: {domain} ---")

        # Load calibration data (use training data from same domain)
        cal_path = DATA_DIR / domain / "train.jsonl"
        cal_texts = []
        if cal_path.exists():
            with open(cal_path) as f:
                for line in f:
                    cal_texts.append(json.loads(line)["text"])
        if len(cal_texts) < 10:
            log(f"  WARNING: only {len(cal_texts)} calibration texts for {domain}")
            continue

        # Apply surgery
        refined, stime = spectral_surgery_adapter(
            model, tokenizer, adapters[domain], cal_texts
        )
        refined_adapters[domain] = refined
        surgery_times[domain] = round(stime, 1)

        # Save refined adapter
        save_adapter(refined, REFINED_DIR / domain)

        # Compute refined PPL
        zero_lora_params(model)
        apply_adapter_weights(model, refined)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
        refined_ppls[domain] = ppl
        delta_ppl = (baseline_ppls[domain] - ppl) / baseline_ppls[domain] * 100
        log(f"  {domain}: refined PPL = {ppl:.2f} (baseline={baseline_ppls[domain]:.2f}, delta={delta_ppl:+.1f}%)")

        # Compute refined KR-Test
        kr_score = compute_kr_score(model, tokenizer, kr_pairs_cache[domain])
        refined_kr[domain] = kr_score
        delta_kr = kr_score - baseline_kr[domain]
        log(f"  {domain}: refined KR = {kr_score:.3f} (baseline={baseline_kr[domain]:.3f}, delta={delta_kr:+.3f})")

        log(f"  {domain}: surgery time = {stime:.1f}s")

    results["surgery_times"] = surgery_times
    results["refined_ppls"] = refined_ppls
    results["refined_kr"] = refined_kr

    # ------------------------------------------------------------------
    # Phase 4: Composed PPL comparison
    # ------------------------------------------------------------------
    log("\n[Phase 4] Composed adapter PPL comparison...")

    # Baseline composed
    baseline_list = [adapters[d] for d in DOMAINS if d in adapters]
    merged_baseline = compose_adapters(baseline_list)
    zero_lora_params(model)
    apply_adapter_weights(model, merged_baseline)
    mx.eval(model.parameters())

    composed_baseline_ppls = {}
    for domain in adapters:
        ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
        composed_baseline_ppls[domain] = ppl
    results["composed_baseline_ppls"] = composed_baseline_ppls

    # Refined composed
    refined_list = [refined_adapters[d] for d in DOMAINS if d in refined_adapters]
    if len(refined_list) == len(baseline_list):
        merged_refined = compose_adapters(refined_list)
        zero_lora_params(model)
        apply_adapter_weights(model, merged_refined)
        mx.eval(model.parameters())

        composed_refined_ppls = {}
        for domain in adapters:
            ppl = compute_ppl(model, tokenizer, DATA_DIR / domain)
            composed_refined_ppls[domain] = ppl
        results["composed_refined_ppls"] = composed_refined_ppls
    else:
        composed_refined_ppls = {}
        log("  WARNING: not all adapters refined, skipping composed comparison")

    # ------------------------------------------------------------------
    # Phase 5: Cosine similarity (composition safety check)
    # ------------------------------------------------------------------
    log("\n[Phase 5] Cosine similarity check...")
    cosines_before = []
    cosines_after = []
    names = [d for d in DOMAINS if d in adapters and d in refined_adapters]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Before
            vi = mx.concatenate([v.reshape(-1) for v in adapters[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapters[names[j]].values()])
            cos_b = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-10))
            mx.eval(cos_b)
            cosines_before.append({"pair": f"{names[i]}-{names[j]}", "cos": round(cos_b.item(), 4)})

            # After
            vi_r = mx.concatenate([v.reshape(-1) for v in refined_adapters[names[i]].values()])
            vj_r = mx.concatenate([v.reshape(-1) for v in refined_adapters[names[j]].values()])
            cos_a = mx.abs(mx.sum(vi_r * vj_r) / (mx.sqrt(mx.sum(vi_r**2)) * mx.sqrt(mx.sum(vj_r**2)) + 1e-10))
            mx.eval(cos_a)
            cosines_after.append({"pair": f"{names[i]}-{names[j]}", "cos": round(cos_a.item(), 4)})

    results["cosines_before"] = cosines_before
    results["cosines_after"] = cosines_after
    mean_cos_before = sum(c["cos"] for c in cosines_before) / max(len(cosines_before), 1)
    mean_cos_after = sum(c["cos"] for c in cosines_after) / max(len(cosines_after), 1)
    results["mean_cos_before"] = round(mean_cos_before, 4)
    results["mean_cos_after"] = round(mean_cos_after, 4)

    # ------------------------------------------------------------------
    # Phase 6: Kill criteria assessment
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: PPL improvement
    ppl_improved = 0
    ppl_total = 0
    for domain in DOMAINS:
        if domain in refined_ppls and domain in baseline_ppls:
            ppl_total += 1
            if refined_ppls[domain] < baseline_ppls[domain]:
                ppl_improved += 1
    k1_pass = ppl_improved > ppl_total / 2
    results["k1_ppl_improved"] = ppl_improved
    results["k1_ppl_total"] = ppl_total
    results["k1_pass"] = k1_pass
    log(f"  K1 (PPL improvement): {ppl_improved}/{ppl_total} domains improved -> {'PASS' if k1_pass else 'FAIL'}")

    for domain in DOMAINS:
        if domain in refined_ppls and domain in baseline_ppls:
            delta = (baseline_ppls[domain] - refined_ppls[domain]) / baseline_ppls[domain] * 100
            log(f"    {domain}: {baseline_ppls[domain]:.2f} -> {refined_ppls[domain]:.2f} ({delta:+.1f}%)")

    # K2: KR-Test improvement
    kr_improved = 0
    kr_total = 0
    for domain in DOMAINS:
        if domain in refined_kr and domain in baseline_kr:
            kr_total += 1
            if refined_kr[domain] > baseline_kr[domain]:
                kr_improved += 1
    k2_pass = kr_improved > kr_total / 2
    results["k2_kr_improved"] = kr_improved
    results["k2_kr_total"] = kr_total
    results["k2_pass"] = k2_pass
    log(f"  K2 (KR-Test improvement): {kr_improved}/{kr_total} domains improved -> {'PASS' if k2_pass else 'FAIL'}")

    for domain in DOMAINS:
        if domain in refined_kr and domain in baseline_kr:
            delta = refined_kr[domain] - baseline_kr[domain]
            log(f"    {domain}: {baseline_kr[domain]:.3f} -> {refined_kr[domain]:.3f} ({delta:+.3f})")

    # K3: Speed
    max_surgery_time = max(surgery_times.values()) if surgery_times else 0
    k3_pass = max_surgery_time < 300  # 5 min
    results["max_surgery_time_s"] = max_surgery_time
    results["k3_pass"] = k3_pass
    log(f"  K3 (Speed <5min): max={max_surgery_time:.1f}s -> {'PASS' if k3_pass else 'FAIL'}")

    # Composition comparison
    if composed_refined_ppls:
        avg_comp_before = sum(composed_baseline_ppls.values()) / len(composed_baseline_ppls)
        avg_comp_after = sum(composed_refined_ppls.values()) / len(composed_refined_ppls)
        comp_delta = (avg_comp_before - avg_comp_after) / avg_comp_before * 100
        results["avg_composed_ppl_before"] = round(avg_comp_before, 4)
        results["avg_composed_ppl_after"] = round(avg_comp_after, 4)
        results["composed_ppl_delta_pct"] = round(comp_delta, 2)
        log(f"\n  Composed PPL: {avg_comp_before:.2f} -> {avg_comp_after:.2f} ({comp_delta:+.1f}%)")

    log(f"\n  Cosine: {mean_cos_before:.4f} -> {mean_cos_after:.4f}")

    # Verdict
    all_pass = k1_pass and k2_pass and k3_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    # Also track partial pass
    results["partial_verdict"] = f"K1={'PASS' if k1_pass else 'FAIL'}, K2={'PASS' if k2_pass else 'FAIL'}, K3={'PASS' if k3_pass else 'FAIL'}"
    log(f"\n  VERDICT: {results['verdict']} ({results['partial_verdict']})")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
