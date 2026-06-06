#!/usr/bin/env python3
"""
P5.B0: Spectral Surgery on PoLAR Adapters

Tests gradient-based SVD reweighting (arXiv:2603.03995) on PoLAR adapters.

PREDICTIONS (from extended proof, Theorems 5-7):
  1. Retracted PoLAR has flat spectrum (all sv ≈ 1) → surgery ill-defined
  2. Different SVD bases give different surgery results → basis-dependent
  3. Any non-trivial surgery breaks Stiefel → destroys composition guarantee
  4. LoRA has non-flat spectrum → surgery CAN be defined (control)

Kill criteria:
  K1270: GSM8K improvement >= 2pp over raw PoLAR
  K1271: PPL preserved within 2pp
  K1272: Surgery < 60s
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_LAYERS = 42
LORA_RANK = 6
LORA_SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
RETRACT_EVERY = 20

N_TRAIN = 20 if IS_SMOKE else 500
N_STEPS = 10 if IS_SMOKE else 300
N_CAL = 5 if IS_SMOKE else 50
N_EVAL = 5 if IS_SMOKE else 30
SURGERY_EPS = 1e-3
SURGERY_LAYERS = [0, 10, 20, 30, 41]  # representative layers for sensitivity


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ── Data ────────────────────────────────────────────────────────

def prepare_gsm8k(tokenizer, n_samples: int) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_samples, len(ds))))
    samples = []
    for ex in ds:
        messages = [
            {"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
        if len(tokens) > 10:
            samples.append(tokens)
    log(f"  Prepared {len(samples)} samples")
    return samples


def get_batch(
    samples: list[list[int]], batch_size: int, step: int, pad_id: int
) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(SEED + step)
    indices = rng.choice(len(samples), size=batch_size, replace=True)
    batch_tokens = [samples[i] for i in indices]
    max_len = max(len(t) for t in batch_tokens)
    padded, masks = [], []
    for tokens in batch_tokens:
        pad_len = max_len - len(tokens)
        padded.append(tokens + [pad_id] * pad_len)
        masks.append([1.0] * len(tokens) + [0.0] * pad_len)
    return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)


# ── Adapter Modules ────────────────────────────────────────────

def _get_dims(base_linear: nn.Module) -> tuple[int, int]:
    if hasattr(base_linear, "group_size"):
        d_out = base_linear.weight.shape[0]
        d_in = base_linear.scales.shape[1] * base_linear.group_size
    else:
        d_in = base_linear.weight.shape[1]
        d_out = base_linear.weight.shape[0]
    return d_in, d_out


class PoLARLinear(nn.Module):
    """PoLAR: ΔW = A @ B with A^T A = I_r, B B^T = I_r (joint Stiefel)."""

    def __init__(self, base: nn.Module, rank: int, scale: float, layer_seed: int = 0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = scale
        d_in, d_out = _get_dims(base)
        self.d_in = d_in
        self.d_out = d_out

        rng = np.random.default_rng(SEED + layer_seed)
        rand_a = rng.standard_normal((d_in, rank)).astype(np.float32)
        Q, _ = np.linalg.qr(rand_a)
        self.lora_a = mx.array(Q[:, :rank])
        rand_b = rng.standard_normal((rank, d_out)).astype(np.float32) * 0.01
        self.lora_b = mx.array(rand_b)

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self.scale * ((x @ self.lora_a) @ self.lora_b)

    def retract_to_stiefel(self) -> tuple[float, float]:
        I_r = np.eye(self.rank, dtype=np.float64)

        # Use np.asarray(mx_array) for efficient conversion (avoids .tolist())
        A_np = np.array(self.lora_a, copy=False).astype(np.float64)
        if np.all(np.isfinite(A_np)) and np.sum(A_np**2) > 1e-12:
            W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
            A_ret = (W @ Vh).astype(np.float32)
            if np.all(np.isfinite(A_ret)):
                self.lora_a = mx.array(A_ret)
                dA = float(np.linalg.norm(
                    A_ret.astype(np.float64).T @ A_ret.astype(np.float64) - I_r, "fro"
                ))
            else:
                dA = float("inf")
        else:
            dA = float("inf")

        B_np = np.array(self.lora_b, copy=False).astype(np.float64)
        if np.all(np.isfinite(B_np)) and np.sum(B_np**2) > 1e-12:
            W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
            B_ret = (W2 @ Vh2).astype(np.float32)
            if np.all(np.isfinite(B_ret)):
                self.lora_b = mx.array(B_ret)
                dB = float(np.linalg.norm(
                    B_ret.astype(np.float64) @ B_ret.astype(np.float64).T - I_r, "fro"
                ))
            else:
                dB = float(np.sqrt(self.rank))
        else:
            dB = float(np.sqrt(self.rank))

        return dA, dB


class LoRALinear(nn.Module):
    """Standard LoRA baseline."""

    def __init__(self, base: nn.Module, rank: int, scale: float, layer_seed: int = 0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = scale
        d_in, d_out = _get_dims(base)
        self.d_in = d_in
        self.d_out = d_out

        rng = np.random.default_rng(SEED + layer_seed)
        A_init = rng.standard_normal((d_in, rank)).astype(np.float32) * (
            1.0 / math.sqrt(d_in)
        )
        self.lora_a = mx.array(A_init)
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self.scale * ((x @ self.lora_a) @ self.lora_b)


class DirectDeltaLinear(nn.Module):
    """Direct delta_W injection for surgical testing."""

    def __init__(self, base: nn.Module, delta_W: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.delta_W = delta_W
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self.scale * (x @ self.delta_W)


# ── Training ────────────────────────────────────────────────────


def inject_adapters(model, cls, rank: int, scale: float) -> list:
    modules = []
    for li in range(N_LAYERS):
        original = model.layers[li].self_attn.q_proj
        adapter = cls(original, rank, scale, layer_seed=li * 1000)
        model.layers[li].self_attn.q_proj = adapter
        modules.append(adapter)
    return modules


def train_adapter(
    model, tokenizer, samples, modules, n_steps: int,
    do_retract: bool = False, name: str = "",
) -> dict:
    model.freeze()
    for mod in modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])
    n_params = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters())
    )
    log(f"  [{name}] {n_params:,} trainable params, {n_steps} steps")
    pad_id = tokenizer.pad_token_id or 0
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens, mask):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss_mask = mask[:, 1:]
        B, L, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="none"
        ).reshape(B, L)
        return (ce * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    final_loss = float("nan")
    for step in range(n_steps):
        tokens, mask = get_batch(samples, BATCH_SIZE, step, pad_id)
        loss, grads = loss_and_grad(model, tokens, mask)
        optimizer.update(model, grads)
        if do_retract and (step + 1) % RETRACT_EVERY == 0:
            mx.eval(model.parameters())
            for mod in modules:
                mod.retract_to_stiefel()
            mx.eval(model.parameters())
        mx.eval(loss, model.parameters())
        final_loss = float(loss.item())
        if (step + 1) % 100 == 0 or step == 0:
            log(f"    step {step+1}/{n_steps}: loss={final_loss:.4f}")

    if do_retract:
        mx.eval(model.parameters())
        max_d = 0.0
        for mod in modules:
            dA, dB = mod.retract_to_stiefel()
            max_d = max(max_d, dA, dB)
        mx.eval(model.parameters())
        log(f"  Final retraction: max_dist={max_d:.2e}")

    elapsed = time.time() - t0
    log(f"  [{name}] Done in {elapsed:.0f}s")
    return {"final_loss": final_loss, "time_s": round(elapsed, 1)}


# ── Efficient SVD for Low-Rank Adapters ────────────────────────


def adapter_svd(mod) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute SVD of delta_W = A @ B via the small r×r intermediate.

    Returns (U_full, S, Vt_full) where:
      U_full: (d_in, r), S: (r,), Vt_full: (r, d_out)
      delta_W = U_full @ diag(S) @ Vt_full

    Returns None if weights contain NaN/inf.
    """
    A_np = np.array(mod.lora_a, copy=False).astype(np.float64)
    B_np = np.array(mod.lora_b, copy=False).astype(np.float64)

    if not (np.all(np.isfinite(A_np)) and np.all(np.isfinite(B_np))):
        return None

    # SVD of A: (d_in, r) → U_A (d_in, r), S_A (r,), Vt_A (r, r)
    U_A, S_A, Vt_A = np.linalg.svd(A_np, full_matrices=False)
    # SVD of B: (r, d_out) → U_B (r, r), S_B (r,), Vt_B (r, d_out)
    U_B, S_B, Vt_B = np.linalg.svd(B_np, full_matrices=False)

    # Intermediate r×r product: P = diag(S_A) @ Vt_A @ U_B @ diag(S_B)
    P = np.diag(S_A) @ Vt_A @ U_B @ np.diag(S_B)  # (r, r)
    Up, Sp, Vtp = np.linalg.svd(P, full_matrices=False)  # all (r, r)

    # Full SVD of delta_W: U_full = U_A @ Up, Vt_full = Vtp @ Vt_B
    U_full = U_A @ Up    # (d_in, r)
    Vt_full = Vtp @ Vt_B  # (r, d_out)

    return U_full, Sp, Vt_full


# ── Spectral Analysis ──────────────────────────────────────────


def analyze_spectrum(modules: list, name: str = "") -> dict:
    """Extract singular value spectra of all adapter delta_Ws."""
    spectra = []
    stable_ranks = []
    n_nan = 0
    for mod in modules:
        result = adapter_svd(mod)
        if result is None:
            n_nan += 1
            spectra.append(np.zeros(mod.rank).tolist())
            stable_ranks.append(0.0)
            continue
        _, S, _ = result
        spectra.append(S.tolist())
        frob_sq = float(np.sum(S**2))
        spec_sq = float(S[0] ** 2) if len(S) > 0 else 1e-12
        sr = frob_sq / spec_sq if spec_sq > 1e-12 else 0.0
        stable_ranks.append(sr)

    all_sv = np.array(spectra)  # (n_layers, r) — uniform shape now
    mean_sv = all_sv.mean(axis=0).tolist()
    std_sv = all_sv.std(axis=0).tolist()
    mean_arr = np.array(mean_sv)
    flatness = float(np.std(mean_arr) / (np.mean(mean_arr) + 1e-12))

    log(f"  [{name}] Mean SV: {[round(x, 4) for x in mean_sv]}")
    log(
        f"  [{name}] Flatness (CoV): {flatness:.6f} "
        f"({'FLAT' if flatness < 0.05 else 'NON-FLAT'})"
    )
    log(f"  [{name}] Stable rank: mean={np.mean(stable_ranks):.4f}")
    if n_nan > 0:
        log(f"  [{name}] WARNING: {n_nan} layers had NaN weights")

    return {
        "mean_sv": mean_sv,
        "std_sv": std_sv,
        "flatness_cov": flatness,
        "mean_stable_rank": float(np.mean(stable_ranks)),
        "spectra_sample": spectra[:3],
        "n_nan_layers": n_nan,
    }


# ── Calibration Loss ───────────────────────────────────────────


def compute_cal_loss(model, tokenizer, cal_samples: list[list[int]]) -> float:
    """Compute average cross-entropy loss on calibration samples."""
    pad_id = tokenizer.pad_token_id or 0
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(cal_samples), BATCH_SIZE):
        batch_tokens = cal_samples[i : i + BATCH_SIZE]
        if not batch_tokens:
            break
        max_len = max(len(t) for t in batch_tokens)
        padded = [t + [pad_id] * (max_len - len(t)) for t in batch_tokens]
        masks = [[1.0] * len(t) + [0.0] * (max_len - len(t)) for t in batch_tokens]

        tokens = mx.array(padded, dtype=mx.int32)
        mask = mx.array(masks, dtype=mx.float32)

        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss_mask = mask[:, 1:]
        B, L, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="none"
        ).reshape(B, L)
        loss = float(((ce * loss_mask).sum() / (loss_mask.sum() + 1e-8)).item())
        mx.eval()
        total_loss += loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Spectral Surgery ───────────────────────────────────────────


def reconstruct_delta_W(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> mx.array:
    """Reconstruct delta_W = U @ diag(S) @ Vt as a float32 MLX array."""
    dW = (U @ np.diag(S) @ Vt).astype(np.float32)
    return mx.array(dW)


def estimate_layer_sensitivity(
    model, layer_idx: int, U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
    base_module: nn.Module, scale: float, cal_samples: list[list[int]],
    tokenizer, orig_adapter: nn.Module,
) -> np.ndarray:
    """Estimate per-component sensitivity via finite differences.

    U: (d_in, r), S: (r,), Vt: (r, d_out) — compact SVD from adapter_svd().
    """
    r = len(S)
    sensitivities = np.zeros(r)
    cal_subset = cal_samples[:10]  # use 10 examples for speed

    for k in range(r):
        # Perturb sigma_k up
        S_up = S.copy()
        S_up[k] += SURGERY_EPS
        model.layers[layer_idx].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U, S_up, Vt), scale
        )
        model.layers[layer_idx].self_attn.q_proj.freeze()
        loss_up = compute_cal_loss(model, tokenizer, cal_subset)

        # Perturb sigma_k down
        S_down = S.copy()
        S_down[k] -= SURGERY_EPS
        model.layers[layer_idx].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U, S_down, Vt), scale
        )
        model.layers[layer_idx].self_attn.q_proj.freeze()
        loss_down = compute_cal_loss(model, tokenizer, cal_subset)

        sensitivities[k] = abs(loss_up - loss_down) / (2 * SURGERY_EPS)

    # Restore original adapter
    model.layers[layer_idx].self_attn.q_proj = orig_adapter
    return sensitivities


def reweight_singular_values(
    S: np.ndarray, sensitivities: np.ndarray
) -> np.ndarray:
    """Reweight SVs under Frobenius norm constraint (Theorem 3)."""
    mean_s = np.mean(sensitivities)
    if mean_s < 1e-12:
        return S.copy()

    # Boost high-sensitivity, shrink low-sensitivity
    weights = (sensitivities / (mean_s + 1e-12)) ** 0.5
    S_new = S * weights

    # Renormalize to preserve Frobenius norm
    norm_orig = np.sqrt(np.sum(S**2))
    norm_new = np.sqrt(np.sum(S_new**2))
    if norm_new > 1e-12:
        S_new = S_new * (norm_orig / norm_new)

    return S_new


def spectral_surgery(model, modules: list, cal_samples, tokenizer) -> dict:
    """Apply spectral surgery to all adapter layers."""
    t_surgery = time.time()

    baseline_loss = compute_cal_loss(model, tokenizer, cal_samples)
    log(f"  Baseline cal loss: {baseline_loss:.4f}")

    # SVD all layers (efficient small-matrix approach)
    all_svd = {}
    for li in range(N_LAYERS):
        result = adapter_svd(modules[li])
        if result is None:
            continue
        U, S, Vt = result
        all_svd[li] = (U, S, Vt)

    # Estimate sensitivity for representative layers
    all_sensitivities = {}
    for li in SURGERY_LAYERS:
        if li not in all_svd:
            log(f"    Layer {li}: SKIPPED (NaN weights)")
            continue
        U, S, Vt = all_svd[li]
        orig_adapter = model.layers[li].self_attn.q_proj
        base_module = orig_adapter.base
        scale = orig_adapter.scale
        sens = estimate_layer_sensitivity(
            model, li, U, S, Vt, base_module, scale, cal_samples, tokenizer,
            orig_adapter,
        )
        all_sensitivities[li] = sens
        log(
            f"    Layer {li}: S={np.round(S, 4).tolist()}, "
            f"sens={np.round(sens, 6).tolist()}"
        )

    # Reweight all layers
    mean_sens = np.mean(list(all_sensitivities.values()), axis=0)
    layer_results = {}

    for li in range(N_LAYERS):
        if li not in all_svd:
            continue
        U, S, Vt = all_svd[li]
        sens = all_sensitivities.get(li, mean_sens)
        S_new = reweight_singular_values(S, sens)

        # Inject surgical adapter
        base_module = modules[li].base
        scale = modules[li].scale
        model.layers[li].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U, S_new, Vt), scale
        )
        model.layers[li].self_attn.q_proj.freeze()

        layer_results[li] = {
            "S_before": S.tolist(),
            "S_after": S_new.tolist(),
        }

    post_loss = compute_cal_loss(model, tokenizer, cal_samples)
    surgery_time = time.time() - t_surgery

    log(f"  Surgery: {surgery_time:.1f}s")
    log(
        f"  Cal loss: {baseline_loss:.4f} → {post_loss:.4f} "
        f"(delta={post_loss - baseline_loss:+.4f})"
    )

    return {
        "baseline_loss": baseline_loss,
        "post_surgery_loss": post_loss,
        "delta_loss": post_loss - baseline_loss,
        "surgery_time_s": round(surgery_time, 1),
        "layer_results_sample": {
            str(k): layer_results[k] for k in SURGERY_LAYERS if k in layer_results
        },
        "sensitivities_sample": {
            str(k): all_sensitivities[k].tolist()
            for k in SURGERY_LAYERS
            if k in all_sensitivities
        },
    }


# ── Basis Rotation Test ────────────────────────────────────────


def test_basis_dependence(
    modules: list, cal_samples, tokenizer, model
) -> dict:
    """Test whether surgery results depend on SVD basis choice (Theorem 6)."""
    test_layer = 20
    mod = modules[test_layer]
    result = adapter_svd(mod)
    if result is None:
        log("  Basis test: SKIPPED (NaN weights)")
        return {"skipped": True, "basis_dependent": False, "sens_cosine": 1.0}
    U, S, Vt = result
    r = len(S)

    # Random orthogonal rotation
    rng = np.random.default_rng(SEED + 999)
    rand_mat = rng.standard_normal((r, r))
    R, _ = np.linalg.qr(rand_mat)

    # Rotated basis
    U_rot = U @ R
    Vt_rot = R.T @ Vt

    # Check reconstruction error (should be ~0 if spectrum is flat)
    recon_orig = U @ np.diag(S) @ Vt
    recon_rot = U_rot @ np.diag(S) @ Vt_rot
    recon_error = float(np.linalg.norm(recon_orig - recon_rot, "fro"))

    sv_variance = float(np.var(S))
    log(f"  Basis test layer {test_layer}:")
    log(f"    SV variance: {sv_variance:.8f}")
    log(f"    Reconstruction error: {recon_error:.8f}")

    orig_adapter = model.layers[test_layer].self_attn.q_proj
    base_module = orig_adapter.base
    scale = orig_adapter.scale
    cal_subset = cal_samples[:10]

    # Sensitivity in original basis
    sens_orig = np.zeros(r)
    for k in range(r):
        S_up = S.copy()
        S_up[k] += SURGERY_EPS
        model.layers[test_layer].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U, S_up, Vt), scale
        )
        model.layers[test_layer].self_attn.q_proj.freeze()
        l_up = compute_cal_loss(model, tokenizer, cal_subset)

        S_down = S.copy()
        S_down[k] -= SURGERY_EPS
        model.layers[test_layer].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U, S_down, Vt), scale
        )
        model.layers[test_layer].self_attn.q_proj.freeze()
        l_down = compute_cal_loss(model, tokenizer, cal_subset)

        sens_orig[k] = abs(l_up - l_down) / (2 * SURGERY_EPS)

    # Sensitivity in rotated basis
    sens_rot = np.zeros(r)
    for k in range(r):
        S_up = S.copy()
        S_up[k] += SURGERY_EPS
        model.layers[test_layer].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U_rot, S_up, Vt_rot), scale
        )
        model.layers[test_layer].self_attn.q_proj.freeze()
        l_up = compute_cal_loss(model, tokenizer, cal_subset)

        S_down = S.copy()
        S_down[k] -= SURGERY_EPS
        model.layers[test_layer].self_attn.q_proj = DirectDeltaLinear(
            base_module, reconstruct_delta_W(U_rot, S_down, Vt_rot), scale
        )
        model.layers[test_layer].self_attn.q_proj.freeze()
        l_down = compute_cal_loss(model, tokenizer, cal_subset)

        sens_rot[k] = abs(l_up - l_down) / (2 * SURGERY_EPS)

    # Restore
    model.layers[test_layer].self_attn.q_proj = orig_adapter

    # Compare
    norm_orig = np.linalg.norm(sens_orig)
    norm_rot = np.linalg.norm(sens_rot)
    sens_cos = float(
        np.dot(sens_orig, sens_rot) / (norm_orig * norm_rot + 1e-12)
    )

    log(f"    Sensitivity (original): {np.round(sens_orig, 6).tolist()}")
    log(f"    Sensitivity (rotated):  {np.round(sens_rot, 6).tolist()}")
    log(f"    Cosine similarity: {sens_cos:.4f}")
    log(
        f"    Basis-dependent: {'YES' if sens_cos < 0.9 else 'NO'} "
        f"(threshold: cos < 0.9)"
    )

    return {
        "sv_variance": sv_variance,
        "reconstruction_error": recon_error,
        "sens_original": sens_orig.tolist(),
        "sens_rotated": sens_rot.tolist(),
        "sens_cosine": sens_cos,
        "basis_dependent": sens_cos < 0.9,
    }


# ── GSM8K Evaluation ──────────────────────────────────────────


def eval_gsm8k(model, tokenizer, n_eval: int) -> dict:
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED + 99).select(range(min(n_eval, len(ds))))

    correct = 0
    total = 0
    for ex in ds:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Solve step by step.\n\n{ex['question']}\n\nAnswer:"
                ),
            }
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            response = mlx_generate(
                model, tokenizer, prompt=formatted, max_tokens=256, verbose=False,
            )
        except Exception:
            response = ""

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()
        total += 1

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match and pred_match.group(1).replace(",", "").strip() == gt:
            correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0.0
    log(f"  GSM8K: {correct}/{total} = {acc:.1f}%")
    return {"accuracy": acc, "correct": correct, "total": total}


# ── Main ───────────────────────────────────────────────────────


def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("P5.B0: Spectral Surgery on PoLAR Adapters")
    log(f"  SMOKE={IS_SMOKE}, steps={N_STEPS}, cal={N_CAL}, eval={N_EVAL}")
    log("=" * 70)

    results = {"smoke": IS_SMOKE, "model": MODEL_ID}
    t_start = time.time()

    # ── Phase 1: PoLAR ────────────────────────────────────────
    log("\n[Phase 1] Train PoLAR adapter")
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("loaded")

    train_samples = prepare_gsm8k(tokenizer, N_TRAIN)
    cal_samples = train_samples[:N_CAL]

    polar_modules = inject_adapters(model, PoLARLinear, LORA_RANK, LORA_SCALE)
    polar_train = train_adapter(
        model, tokenizer, train_samples, polar_modules,
        N_STEPS, do_retract=True, name="PoLAR",
    )
    results["polar_train"] = polar_train

    # Spectrum analysis
    log("\n[Phase 1b] PoLAR spectrum analysis")
    polar_spectrum = analyze_spectrum(polar_modules, "PoLAR")
    results["polar_spectrum"] = polar_spectrum

    # Pre-surgery GSM8K
    log("\n[Phase 1c] PoLAR pre-surgery GSM8K")
    polar_pre = eval_gsm8k(model, tokenizer, N_EVAL)
    results["polar_pre_gsm8k"] = polar_pre

    # Basis dependence test (Theorem 6)
    log("\n[Phase 1d] Basis dependence test")
    basis_test = test_basis_dependence(polar_modules, cal_samples, tokenizer, model)
    results["basis_dependence"] = basis_test

    # Spectral surgery
    log("\n[Phase 1e] Spectral surgery on PoLAR")
    polar_surgery = spectral_surgery(model, polar_modules, cal_samples, tokenizer)
    results["polar_surgery"] = polar_surgery

    # Post-surgery GSM8K
    log("\n[Phase 1f] PoLAR post-surgery GSM8K")
    polar_post = eval_gsm8k(model, tokenizer, N_EVAL)
    results["polar_post_gsm8k"] = polar_post

    cleanup(model, tokenizer)

    # ── Phase 2: LoRA control ─────────────────────────────────
    log("\n[Phase 2] Train LoRA adapter (control)")
    model2, tokenizer2 = mlx_load(MODEL_ID)
    log_memory("loaded-2")

    lora_modules = inject_adapters(model2, LoRALinear, LORA_RANK, LORA_SCALE)
    lora_train = train_adapter(
        model2, tokenizer2, train_samples, lora_modules,
        N_STEPS, do_retract=False, name="LoRA",
    )
    results["lora_train"] = lora_train

    # Spectrum analysis
    log("\n[Phase 2b] LoRA spectrum analysis")
    lora_spectrum = analyze_spectrum(lora_modules, "LoRA")
    results["lora_spectrum"] = lora_spectrum

    # Pre-surgery GSM8K
    log("\n[Phase 2c] LoRA pre-surgery GSM8K")
    lora_pre = eval_gsm8k(model2, tokenizer2, N_EVAL)
    results["lora_pre_gsm8k"] = lora_pre

    # Spectral surgery
    log("\n[Phase 2d] Spectral surgery on LoRA")
    lora_surgery = spectral_surgery(model2, lora_modules, cal_samples, tokenizer2)
    results["lora_surgery"] = lora_surgery

    # Post-surgery GSM8K
    log("\n[Phase 2e] LoRA post-surgery GSM8K")
    lora_post = eval_gsm8k(model2, tokenizer2, N_EVAL)
    results["lora_post_gsm8k"] = lora_post

    cleanup(model2, tokenizer2)

    # ── Summary ───────────────────────────────────────────────
    polar_delta = polar_post["accuracy"] - polar_pre["accuracy"]
    lora_delta = lora_post["accuracy"] - lora_pre["accuracy"]

    k1270_pass = polar_delta >= 2.0
    k1271_pass = abs(polar_surgery["delta_loss"]) < 0.02
    k1272_pass = polar_surgery["surgery_time_s"] < 60

    results["summary"] = {
        "polar_pre_acc": polar_pre["accuracy"],
        "polar_post_acc": polar_post["accuracy"],
        "polar_delta_acc": polar_delta,
        "lora_pre_acc": lora_pre["accuracy"],
        "lora_post_acc": lora_post["accuracy"],
        "lora_delta_acc": lora_delta,
        "polar_flatness": polar_spectrum["flatness_cov"],
        "lora_flatness": lora_spectrum["flatness_cov"],
        "polar_surgery_time_s": polar_surgery["surgery_time_s"],
        "basis_dependent": basis_test["basis_dependent"],
        "k1270_pass": k1270_pass,
        "k1271_pass": k1271_pass,
        "k1272_pass": k1272_pass,
        "all_pass": k1270_pass and k1271_pass and k1272_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(
        f"  PoLAR spectrum: CoV={polar_spectrum['flatness_cov']:.6f} "
        f"({'FLAT' if polar_spectrum['flatness_cov'] < 0.05 else 'NON-FLAT'})"
    )
    log(
        f"  LoRA  spectrum: CoV={lora_spectrum['flatness_cov']:.6f} "
        f"({'FLAT' if lora_spectrum['flatness_cov'] < 0.05 else 'NON-FLAT'})"
    )
    log(
        f"  PoLAR GSM8K: {polar_pre['accuracy']:.1f}% → "
        f"{polar_post['accuracy']:.1f}% (delta={polar_delta:+.1f}pp)"
    )
    log(
        f"  LoRA  GSM8K: {lora_pre['accuracy']:.1f}% → "
        f"{lora_post['accuracy']:.1f}% (delta={lora_delta:+.1f}pp)"
    )
    log(
        f"  Basis dependent: {'YES' if basis_test['basis_dependent'] else 'NO'} "
        f"(cos={basis_test['sens_cosine']:.4f})"
    )
    log(f"  Surgery time: {polar_surgery['surgery_time_s']:.1f}s")
    log("")
    log(
        f"  K1270 (GSM8K +2pp):   "
        f"{'PASS' if k1270_pass else 'FAIL'} — delta={polar_delta:+.1f}pp"
    )
    log(
        f"  K1271 (PPL stable):   "
        f"{'PASS' if k1271_pass else 'FAIL'} — delta_loss={polar_surgery['delta_loss']:+.4f}"
    )
    log(
        f"  K1272 (surgery <60s): "
        f"{'PASS' if k1272_pass else 'FAIL'} — {polar_surgery['surgery_time_s']:.1f}s"
    )

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
