#!/usr/bin/env python3
"""Pierre Pro: Grassmannian LoRA-A initialization on Qwen3-4B.

Verify orthogonal A-matrices work on GQA architecture (not just MHA).
Generate and save skeleton for N=5 and N=24 domains.

Kill criteria:
  K810: Pairwise cos > 0.05 at N=5
  K811: Initialization takes > 60s
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Load model ID from pro_base_validation results
BASE_RESULTS = EXPERIMENT_DIR.parent / "pro_base_validation" / "results.json"

LORA_RANK = 16
SEED = 42

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

DOMAINS_5 = ["medical", "code", "math", "legal", "finance"]
DOMAINS_24 = [
    "medical", "code", "math", "legal", "finance", "science", "history",
    "philosophy", "creative_writing", "cooking", "health_fitness", "psychology",
    "education", "engineering", "agriculture", "environmental", "politics",
    "economics", "sociology", "linguistics", "cybersecurity", "marketing",
    "sports", "music",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


def generate_grassmannian_skeleton(n_layers, hidden_dims, target_keys, n_domains, rank, seed=42):
    """Generate orthogonal A-matrices via QR decomposition for all layers/modules.

    Returns dict of "layer_{l}_{key}_domain_{d}" -> np.array (in_features, rank).
    """
    rng = np.random.RandomState(seed)
    skeleton = {}

    for li in range(n_layers):
        for key in target_keys:
            in_features = hidden_dims.get(key, hidden_dims["default"])
            total_rank = n_domains * rank

            if total_rank > in_features:
                log(f"  WARNING: {key} at layer {li}: need {total_rank} orthogonal vectors but dim={in_features}")
                # Fall back to random init for overflow domains
                for di in range(n_domains):
                    if di * rank < in_features:
                        random_mat = rng.randn(in_features, rank).astype(np.float32)
                        Q, _ = np.linalg.qr(random_mat)
                        skeleton[f"layer_{li}_{key}_domain_{di}"] = Q[:, :rank]
                    else:
                        skeleton[f"layer_{li}_{key}_domain_{di}"] = rng.randn(in_features, rank).astype(np.float32) * 0.01
                continue

            # Generate orthogonal basis via QR decomposition
            random_mat = rng.randn(in_features, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(random_mat)

            # Partition into per-domain A-matrices
            for di in range(n_domains):
                start = di * rank
                end = start + rank
                skeleton[f"layer_{li}_{key}_domain_{di}"] = Q[:, start:end]

    return skeleton


def measure_pairwise_cosine(skeleton, n_layers, target_keys, n_domains):
    """Measure pairwise cosine similarity between all domain A-matrices."""
    cos_values = []
    for li in range(n_layers):
        for key in target_keys:
            for di in range(n_domains):
                for dj in range(di + 1, n_domains):
                    k_i = f"layer_{li}_{key}_domain_{di}"
                    k_j = f"layer_{li}_{key}_domain_{dj}"
                    if k_i in skeleton and k_j in skeleton:
                        A_i = skeleton[k_i]
                        A_j = skeleton[k_j]
                        # Cosine between flattened A-matrices
                        cos = np.abs(np.sum(A_i * A_j)) / (np.linalg.norm(A_i) * np.linalg.norm(A_j) + 1e-8)
                        cos_values.append(cos)
    return cos_values


def main():
    t0 = time.time()
    log("Pierre Pro: Grassmannian Initialization")
    log("=" * 60)

    # Get model ID from base validation
    if BASE_RESULTS.exists():
        base_data = json.loads(BASE_RESULTS.read_text())
        model_id = base_data.get("model_id", "mlx-community/Qwen3-4B-4bit")
        # Dimensions may be nested under "load" key
        load_data = base_data.get("load", base_data)
        n_layers = load_data.get("n_layers", base_data.get("n_layers", 36))
        hidden_dim = load_data.get("hidden_dim", base_data.get("hidden_dim", 2560))
    else:
        model_id = "mlx-community/Qwen3-4B-4bit"
        log("WARNING: No base validation results. Loading model to detect architecture.")
        model, _ = load(model_id)
        n_layers = len(model.model.layers)
        hidden_dim = model.model.embed_tokens.weight.shape[1]
        # Detect per-module dimensions
        layer0 = model.model.layers[0]
        cleanup(model)

    log(f"Model: {model_id}, layers={n_layers}, hidden_dim={hidden_dim}")

    # Detect module dimensions from model config (not weight shapes, which are
    # compressed for quantized models). For QuantizedLinear with 4-bit packing,
    # weight.shape[-1] = in_features * bits / 32, NOT the true in_features.
    # Use the model config directly to get correct logical dimensions.
    model, _ = load(model_id)
    layer0 = model.model.layers[0]

    # Read config for true dimensions
    config = getattr(model, 'config', None) or getattr(model, 'args', None)
    if config is not None:
        cfg_hidden = getattr(config, 'hidden_size', hidden_dim)
        cfg_heads = getattr(config, 'num_attention_heads', 32)
        cfg_kv_heads = getattr(config, 'num_key_value_heads', cfg_heads)
        cfg_head_dim = getattr(config, 'head_dim', cfg_hidden // cfg_heads)
        cfg_intermediate = getattr(config, 'intermediate_size', cfg_hidden * 4)
        log(f"  Config: hidden={cfg_hidden}, heads={cfg_heads}, kv_heads={cfg_kv_heads}, "
            f"head_dim={cfg_head_dim}, intermediate={cfg_intermediate}")
    else:
        cfg_hidden = hidden_dim
        cfg_heads = 32
        cfg_kv_heads = cfg_heads
        cfg_head_dim = cfg_hidden // cfg_heads
        cfg_intermediate = cfg_hidden * 4

    # Map each module to its TRUE in_features (input dimension for LoRA A-matrix)
    # All q/k/v projections take hidden_dim as input
    # o_proj takes num_heads * head_dim as input
    # gate_proj, up_proj take hidden_dim as input
    # down_proj takes intermediate_size as input
    hidden_dims = {
        "default": cfg_hidden,
        "self_attn.q_proj": cfg_hidden,
        "self_attn.k_proj": cfg_hidden,
        "self_attn.v_proj": cfg_hidden,
        "self_attn.o_proj": cfg_heads * cfg_head_dim,
        "mlp.gate_proj": cfg_hidden,
        "mlp.up_proj": cfg_hidden,
        "mlp.down_proj": cfg_intermediate,
    }

    # Verify by cross-checking with quantized weight shapes where possible
    for key in TARGET_KEYS:
        m = layer0
        for part in key.split("."):
            m = getattr(m, part, None)
            if m is None: break
        if m is not None and hasattr(m, 'weight'):
            # For QuantizedLinear, recover true in_features from packed weight
            bits = getattr(m, 'bits', None)
            if bits is not None:
                recovered = m.weight.shape[-1] * (32 // bits)
                expected = hidden_dims.get(key, cfg_hidden)
                match = "OK" if recovered == expected else f"MISMATCH (recovered={recovered})"
                log(f"  {key}: in_features={expected} [quantized verify: {match}]")
            else:
                log(f"  {key}: in_features={hidden_dims.get(key, cfg_hidden)} [unquantized]")
    cleanup(model)

    results = {}

    # Phase 1: Generate N=5 skeleton
    log(f"\n=== Phase 1: N=5 Grassmannian Skeleton ===")
    t1 = time.time()
    skeleton_5 = generate_grassmannian_skeleton(
        n_layers, hidden_dims, TARGET_KEYS, len(DOMAINS_5), LORA_RANK, SEED
    )
    dt_5 = time.time() - t1
    log(f"  Generated {len(skeleton_5)} keys in {dt_5:.2f}s")

    cos_5 = measure_pairwise_cosine(skeleton_5, n_layers, TARGET_KEYS, len(DOMAINS_5))
    mean_cos_5 = float(np.mean(cos_5)) if cos_5 else 0
    max_cos_5 = float(np.max(cos_5)) if cos_5 else 0
    log(f"  N=5 pairwise |cos|: mean={mean_cos_5:.6f}, max={max_cos_5:.6f}")

    # Save N=5 skeleton
    skeleton_5_path = EXPERIMENT_DIR / "grassmannian_skeleton_n5.npz"
    np.savez(str(skeleton_5_path), **skeleton_5)
    log(f"  Saved to {skeleton_5_path}")

    results["n5"] = {
        "n_keys": len(skeleton_5),
        "init_time_s": round(dt_5, 2),
        "mean_cos": round(mean_cos_5, 6),
        "max_cos": round(max_cos_5, 6),
        "n_pairs": len(cos_5),
    }

    # Phase 2: Generate N=24 skeleton
    log(f"\n=== Phase 2: N=24 Grassmannian Skeleton ===")
    t2 = time.time()
    skeleton_24 = generate_grassmannian_skeleton(
        n_layers, hidden_dims, TARGET_KEYS, len(DOMAINS_24), LORA_RANK, SEED + 1
    )
    dt_24 = time.time() - t2
    log(f"  Generated {len(skeleton_24)} keys in {dt_24:.2f}s")

    cos_24 = measure_pairwise_cosine(skeleton_24, n_layers, TARGET_KEYS, len(DOMAINS_24))
    mean_cos_24 = float(np.mean(cos_24)) if cos_24 else 0
    max_cos_24 = float(np.max(cos_24)) if cos_24 else 0
    log(f"  N=24 pairwise |cos|: mean={mean_cos_24:.6f}, max={max_cos_24:.6f}")

    skeleton_24_path = EXPERIMENT_DIR / "grassmannian_skeleton_n24.npz"
    np.savez(str(skeleton_24_path), **skeleton_24)
    log(f"  Saved to {skeleton_24_path}")

    results["n24"] = {
        "n_keys": len(skeleton_24),
        "init_time_s": round(dt_24, 2),
        "mean_cos": round(mean_cos_24, 6),
        "max_cos": round(max_cos_24, 6),
        "n_pairs": len(cos_24),
    }

    # Phase 3: Capacity check
    max_orthogonal = {k: v // LORA_RANK for k, v in hidden_dims.items()}
    log(f"\n=== Phase 3: Capacity ===")
    for k, cap in max_orthogonal.items():
        log(f"  {k}: max {cap} orthogonal domains at rank {LORA_RANK}")

    results["capacity"] = max_orthogonal
    results["model_id"] = model_id
    results["hidden_dims"] = hidden_dims
    results["total_time_s"] = round(time.time() - t0, 1)

    # Kill criteria
    k810 = max_cos_5 <= 0.05
    k811 = max(dt_5, dt_24) <= 60.0

    results["kill_criteria"] = {
        "K810": {"pass": k810, "value": round(max_cos_5, 6), "threshold": 0.05},
        "K811": {"pass": k811, "value": round(max(dt_5, dt_24), 2), "threshold": 60.0},
    }
    results["all_pass"] = k810 and k811

    log(f"\n{'='*60}")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
