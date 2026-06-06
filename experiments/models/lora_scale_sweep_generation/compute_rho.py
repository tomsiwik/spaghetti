#!/usr/bin/env python3
"""Compute rho(s) = s * ||B*A||_2 / ||W||_2 for the code adapter across layers.

rho(s) is the relative perturbation magnitude of the LoRA update at scale s.
A value near 1 means the adapter is perturbing W by ~100% its own magnitude.

Supports SMOKE_TEST=1 for fast validation (<60s).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

EXPERIMENT_DIR = Path(__file__).parent
# EXPERIMENT_DIR is .../llm/experiments/models/lora_scale_sweep_generation
# WORKSPACE is .../llm
WORKSPACE = EXPERIMENT_DIR.parent.parent.parent  # experiments/models/../../../llm

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
ADAPTER_PATH = WORKSPACE / "experiments/models/bitnet_sft_generation_v3/sft_adapters/code/adapter.npz"
SKELETON_PATH = WORKSPACE / "experiments/models/real_data_domain_experts/adapters/grassmannian_skeleton.npz"

CODE_DOMAIN_IDX = 1
LORA_RANK = 16
N_LAYERS = 30

TARGET_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Scales to evaluate rho at
SCALES = [1, 2, 4, 8, 20]


def log(msg):
    print(msg, flush=True)


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


def unpack_ternary_np(packed_weights_np, out_features, weight_scale_np, invert_scale):
    """Unpack 2-bit packed ternary weights to float32 numpy array.

    Returns W of shape (out_features, in_features).
    Packed layout: 4 values per byte, low bits first.
    """
    packed = packed_weights_np.astype(np.int32)
    w0 = (packed & 3).astype(np.float32) - 1
    w1 = ((packed >> 2) & 3).astype(np.float32) - 1
    w2 = ((packed >> 4) & 3).astype(np.float32) - 1
    w3 = ((packed >> 6) & 3).astype(np.float32) - 1
    unpacked = np.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale_np.astype(np.float32)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def spectral_norm_np(mat):
    """Compute largest singular value (spectral norm) of a 2D numpy matrix."""
    # Use svd with compute_uv=False for efficiency (just singular values)
    sv = np.linalg.svd(mat, compute_uv=False)
    return float(sv[0])


def phase_load_base_weights():
    """Load the BitNet model and extract ternary W matrices for all target layers.

    Returns dict: {(li, key) -> W_np} where W_np is float32 (out, in).
    """
    log("\n[Phase 1] Loading base model and unpacking ternary weights...")
    t0 = time.time()

    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    model, _tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    log_memory("after-model-load")

    layers_to_process = list(range(N_LAYERS))
    if IS_SMOKE:
        layers_to_process = layers_to_process[:3]
        log(f"  SMOKE_TEST: processing only layers {layers_to_process}")

    w_matrices = {}

    for li in layers_to_process:
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            # Navigate the module hierarchy
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break

            if module is None:
                log(f"  WARNING: layer {li} key {key} not found")
                continue

            if not isinstance(module, BitLinear):
                log(f"  WARNING: layer {li} key {key} is {type(module)}, expected BitLinear")
                continue

            # Extract packed weights and scale, force eval to numpy
            packed = module.weight
            scale = module.weight_scale
            invert = module.invert_weight_scales
            out_f = module.out_features

            mx.eval(packed, scale)
            # packed is uint8 — convert via float32 to avoid bfloat16 PEP3118 issues
            packed_np = np.array(packed)
            # weight_scale is bfloat16 — np.array() fails on bfloat16 via buffer protocol
            # cast to float32 in MLX first, then export
            scale_f32 = scale.astype(mx.float32)
            mx.eval(scale_f32)
            scale_np = np.array(scale_f32)

            W = unpack_ternary_np(packed_np, out_f, scale_np, invert)
            w_matrices[(li, key)] = W

            del packed, scale, packed_np, scale_np, W

    log(f"  Extracted {len(w_matrices)} W matrices")
    log(f"  Time: {time.time() - t0:.1f}s")
    log_memory("after-weight-extraction")

    cleanup(model, _tokenizer)
    return w_matrices


def phase_compute_rho(w_matrices):
    """Load skeleton A and adapter B, compute rho per layer/key.

    Returns per_layer_rho: list of dicts with layer, key, sigma_W, sigma_BA.
    """
    log("\n[Phase 2] Loading skeleton A and adapter B, computing spectral norms...")
    t0 = time.time()

    # Load skeleton (lazy-load in numpy)
    skeleton_npz = np.load(str(SKELETON_PATH))
    adapter_npz = np.load(str(ADAPTER_PATH))

    log(f"  Skeleton keys: {len(skeleton_npz.files)}")
    log(f"  Adapter keys: {len(adapter_npz.files)}")

    rows = []
    layers_present = sorted(set(li for (li, _) in w_matrices.keys()))

    for li in layers_present:
        for key in TARGET_KEYS:
            if (li, key) not in w_matrices:
                continue

            W = w_matrices[(li, key)]  # (out_features, in_features)

            # Skeleton key: layer_{li}_{key}_domain_{domain_idx}
            skel_key = f"layer_{li}_{key}_domain_{CODE_DOMAIN_IDX}"
            if skel_key not in skeleton_npz:
                log(f"  WARNING: skeleton missing {skel_key}")
                continue
            A = skeleton_npz[skel_key].astype(np.float32)  # (in_features, rank)

            # Adapter key: model.layers.{li}.{key}.lora_b
            ada_key = f"model.layers.{li}.{key}.lora_b"
            if ada_key not in adapter_npz:
                log(f"  WARNING: adapter missing {ada_key}")
                continue
            B = adapter_npz[ada_key].astype(np.float32)  # (rank, out_features)

            # B*A: the LoRA update direction
            # Output = W @ x + scale * B @ (A.T @ x)
            # = W @ x + scale * (B @ A.T) @ x
            # The perturbation matrix is B @ A.T  shape: (rank, out) @ (out_in, in).T
            # Wait -- let's be careful:
            # A: (in_features, rank)   -- projects input down to rank
            # B: (rank, out_features)  -- projects from rank to output
            # Update: x -> A.T @ x then B @ that = B @ A.T @ x
            # Perturbation matrix P = B @ A.T
            # W is (out_features, in_features) -- standard convention
            # P should also be (out_features, in_features):
            #   B: (rank, out_features).T = (out_features, rank)
            #   A: (in_features, rank).T = (rank, in_features)
            # Wait: the forward is x @ A = (batch, in) @ (in, rank) = (batch, rank)
            #       then (batch, rank) @ B = wait, B is (rank, out_features)
            #       result: (batch, out_features)
            # So the update adds: x @ A @ B  (in matrix terms: A @ B for the weight update)
            # as a weight this is A @ B: (in, rank) @ (rank, out) = (in, out)
            # transposed to weight form (out, in): (A @ B).T = B.T @ A.T
            # But for spectral norm we just need ||B @ A.T||_2 or equivalently
            # ||A @ B||_2 since sigma(M) = sigma(M.T)
            # Let's compute P = A @ B: (in, rank) @ (rank, out) = (in, out)
            P = A @ B  # (in_features, out_features)

            sigma_W = spectral_norm_np(W)
            sigma_BA = spectral_norm_np(P)

            rows.append({
                "layer": li,
                "key": key,
                "sigma_W": float(sigma_W),
                "sigma_BA": float(sigma_BA),
                "W_shape": list(W.shape),
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
            })

            del W, A, B, P

    log(f"  Computed {len(rows)} (layer, key) pairs in {time.time() - t0:.1f}s")
    return rows


def phase_summarize(rows):
    """Compute rho(s) for each scale and report summary statistics."""
    log("\n[Phase 3] Computing rho(s) = s * sigma_BA / sigma_W ...")

    rho_per_entry = []
    for row in rows:
        rho_base = row["sigma_BA"] / row["sigma_W"] if row["sigma_W"] > 0 else float("inf")
        row["rho_base"] = float(rho_base)
        for s in SCALES:
            row[f"rho_s{s}"] = float(s * rho_base)
        rho_per_entry.append(rho_base)

    mean_rho_base = float(np.mean(rho_per_entry))

    log(f"\n  Per-layer rho(base) = sigma_BA / sigma_W:")
    log(f"  {'Layer':>5}  {'Key':<22}  {'sigma_W':>10}  {'sigma_BA':>10}  {'rho_base':>10}")
    log(f"  {'-'*5}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for row in rows:
        log(f"  {row['layer']:>5}  {row['key']:<22}  {row['sigma_W']:>10.4f}  "
            f"{row['sigma_BA']:>10.6f}  {row['rho_base']:>10.6f}")

    log(f"\n  Mean rho_base = {mean_rho_base:.6f}")

    log(f"\n  rho(s) = s * mean_rho_base:")
    rho_at_scale = {}
    for s in SCALES:
        rho_s = s * mean_rho_base
        rho_at_scale[s] = rho_s
        log(f"    s={s:>3}: rho = {rho_s:.6f}")

    return mean_rho_base, rho_at_scale, rows


def main():
    t_start = time.time()
    log("=" * 70)
    log("compute_rho.py — rho(s) = s * ||B*A||_2 / ||W||_2 for code adapter")
    log(f"SMOKE_TEST: {IS_SMOKE}")
    log("=" * 70)
    log_memory("start")

    w_matrices = phase_load_base_weights()
    log_memory("after-phase1")

    rows = phase_compute_rho(w_matrices)
    del w_matrices
    gc.collect()
    mx.clear_cache()
    log_memory("after-phase2")

    mean_rho_base, rho_at_scale, rows_with_rho = phase_summarize(rows)

    total_time = time.time() - t_start

    results = {
        "experiment": "compute_rho",
        "model": MODEL_ID,
        "adapter": str(ADAPTER_PATH),
        "skeleton": str(SKELETON_PATH),
        "code_domain_idx": CODE_DOMAIN_IDX,
        "lora_rank": LORA_RANK,
        "smoke_test": IS_SMOKE,
        "scales": SCALES,
        "mean_rho_base": mean_rho_base,
        "rho_at_scale": {str(s): v for s, v in rho_at_scale.items()},
        "per_layer": rows_with_rho,
        "total_time_s": round(total_time, 1),
    }

    out_path = EXPERIMENT_DIR / "rho_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {total_time:.1f}s")

    log("\n" + "=" * 70)
    log("SUMMARY: rho(s) values")
    log("=" * 70)
    for s in SCALES:
        log(f"  rho(s={s:>3}) = {rho_at_scale[s]:.6f}")
    log(f"\n  Interpretation: at training scale s=20, each LoRA update perturbs")
    log(f"  the base weight by ~{rho_at_scale.get(20, 0)*100:.2f}% of its spectral norm")

    return results


if __name__ == "__main__":
    main()
