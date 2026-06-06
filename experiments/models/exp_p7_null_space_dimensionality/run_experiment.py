"""P7.A0: Null-Space Dimensionality of Gemma 4 E4B Per Layer.

Measures the effective null space of weight matrices at every layer via SVD.
Determines how many rank-r adapter slots fit in each layer's null space.

Platform: Apple M5 Pro 48GB, MLX only.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_RANK = 6  # Target rank for adapter capacity calculation
THRESHOLDS = [1e-2, 1e-3, 1e-4]  # Relative singular-value thresholds
# Target projections — the ones relevant for LoRA composition
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
OUTPUT_DIR = Path(__file__).parent


def dequantize_weight(linear: nn.Module) -> mx.array:
    """Extract full-precision weight from a QuantizedLinear or Linear layer."""
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


def measure_null_space(W: mx.array, thresholds: list[float]) -> dict:
    """SVD of W, return null-space dims at each threshold.

    W shape: (output_dim, input_dim) — standard nn.Linear convention.
    Null space is in the input dimension: dim = input_dim - rank.
    """
    m, n = W.shape  # (output_dim, input_dim)

    # Compute singular values only (faster than full SVD)
    # mx.linalg.svd returns (U, S, Vt); we only need S
    _, S, _ = mx.linalg.svd(W, stream=mx.cpu)
    mx.eval(S)

    sigma_max = S[0].item()
    sigma_min = S[-1].item() if S.shape[0] > 0 else 0.0

    result = {
        "shape": [int(m), int(n)],
        "theoretical_null_dim": max(0, n - min(m, n)),
        "sigma_max": float(sigma_max),
        "sigma_min": float(sigma_min),
        "condition_number": float(sigma_max / sigma_min) if sigma_min > 0 else float("inf"),
        "num_singular_values": int(S.shape[0]),
    }

    for eps in thresholds:
        threshold_val = eps * sigma_max
        # Count singular values ABOVE threshold = effective rank
        above = mx.sum(S > threshold_val).item()
        null_dim = n - above
        result[f"null_dim_eps_{eps}"] = int(null_dim)
        result[f"effective_rank_eps_{eps}"] = int(above)
        result[f"adapter_slots_r{ADAPTER_RANK}_eps_{eps}"] = int(null_dim) // ADAPTER_RANK

    return result


def main():
    print(f"Loading model: {MODEL_ID}")
    t0 = time.time()

    from mlx_lm import load
    model, _ = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Discover layers — E4B wraps in language_model.model.layers
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find layers in model")
    n_layers = len(layers)
    print(f"Number of layers: {n_layers}")

    all_results = {
        "model": MODEL_ID,
        "adapter_rank": ADAPTER_RANK,
        "thresholds": THRESHOLDS,
        "load_time_s": round(load_time, 1),
        "n_layers": n_layers,
        "layers": {},
    }

    total_t = time.time()

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        attn = layer.self_attn
        layer_results = {}

        for module_name in TARGET_MODULES:
            linear = getattr(attn, module_name, None)
            if linear is None:
                continue

            t1 = time.time()
            W = dequantize_weight(linear)
            mx.eval(W)
            result = measure_null_space(W, THRESHOLDS)
            elapsed = time.time() - t1

            result["svd_time_s"] = round(elapsed, 2)
            layer_results[module_name] = result

            # Free memory
            del W

            print(
                f"  Layer {layer_idx:2d} {module_name:6s}: "
                f"shape={result['shape']}, "
                f"null@1e-3={result.get('null_dim_eps_0.001', '?'):4d}, "
                f"slots@r{ADAPTER_RANK}={result.get(f'adapter_slots_r{ADAPTER_RANK}_eps_0.001', '?'):3d}, "
                f"cond={result['condition_number']:.1f}, "
                f"t={elapsed:.2f}s"
            )

        all_results["layers"][str(layer_idx)] = layer_results

    total_elapsed = time.time() - total_t
    print(f"\nTotal SVD time: {total_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Aggregate statistics (for kill criteria evaluation)
    # ------------------------------------------------------------------
    eps_key = "null_dim_eps_0.001"  # Primary threshold for kill criteria

    # Classify layers by q_proj shape
    local_null_dims = []
    global_null_dims = []
    all_null_dims = []

    for layer_idx_str, layer_data in all_results["layers"].items():
        if "q_proj" not in layer_data:
            continue
        qp = layer_data["q_proj"]
        nd = qp.get(eps_key, 0)
        all_null_dims.append(nd)
        m, n = qp["shape"]
        if m <= n:
            # Local: output_dim <= input_dim → theoretical null space > 0
            local_null_dims.append(nd)
        else:
            # Global: output_dim > input_dim → theoretical null space = 0
            global_null_dims.append(nd)

    summary = {
        "total_svd_time_s": round(total_elapsed, 1),
    }

    if local_null_dims:
        import statistics
        mean_local = statistics.mean(local_null_dims)
        std_local = statistics.stdev(local_null_dims) if len(local_null_dims) > 1 else 0.0
        summary["local_layers"] = {
            "count": len(local_null_dims),
            "null_dim_mean": round(mean_local, 1),
            "null_dim_std": round(std_local, 1),
            "null_dim_min": min(local_null_dims),
            "null_dim_max": max(local_null_dims),
            "null_dim_cv": round(std_local / mean_local, 3) if mean_local > 0 else None,
            "adapter_slots_mean": int(mean_local) // ADAPTER_RANK,
        }

    if global_null_dims:
        import statistics
        mean_global = statistics.mean(global_null_dims)
        std_global = statistics.stdev(global_null_dims) if len(global_null_dims) > 1 else 0.0
        summary["global_layers"] = {
            "count": len(global_null_dims),
            "null_dim_mean": round(mean_global, 1),
            "null_dim_std": round(std_global, 1),
            "null_dim_min": min(global_null_dims),
            "null_dim_max": max(global_null_dims),
            "null_dim_cv": round(std_global / mean_global, 3) if mean_global > 0 else None,
            "adapter_slots_mean": int(mean_global) // ADAPTER_RANK,
        }

    if all_null_dims:
        import statistics
        mean_all = statistics.mean(all_null_dims)
        std_all = statistics.stdev(all_null_dims) if len(all_null_dims) > 1 else 0.0
        summary["all_layers"] = {
            "count": len(all_null_dims),
            "null_dim_mean": round(mean_all, 1),
            "null_dim_std": round(std_all, 1),
            "null_dim_min": min(all_null_dims),
            "null_dim_max": max(all_null_dims),
            "null_dim_cv": round(std_all / mean_all, 3) if mean_all > 0 else None,
        }

    all_results["summary"] = summary

    # ------------------------------------------------------------------
    # Kill criteria evaluation
    # ------------------------------------------------------------------
    k1294 = "PASS" if local_null_dims and min(local_null_dims) >= 100 else "FAIL"
    k1295 = "PASS" if global_null_dims and min(global_null_dims) >= 50 else (
        "PASS" if not global_null_dims else "FAIL"
    )
    # If no global layers exist in E4B, check all layers instead
    if not global_null_dims:
        k1295_note = "No global layers detected in E4B; all layers are local-type"
    else:
        k1295_note = f"min global null_dim = {min(global_null_dims)}"

    if all_null_dims and len(all_null_dims) > 1:
        import statistics
        cv = statistics.stdev(all_null_dims) / statistics.mean(all_null_dims)
        k1296 = "PASS" if cv < 0.20 else "FAIL"
        k1296_note = f"CV = {cv:.3f}"
    else:
        k1296 = "FAIL"
        k1296_note = "insufficient data"

    kill_results = {
        "K1294": {"result": k1294, "note": f"min local null_dim = {min(local_null_dims) if local_null_dims else 'N/A'}"},
        "K1295": {"result": k1295, "note": k1295_note},
        "K1296": {"result": k1296, "note": k1296_note},
    }
    all_results["kill_criteria"] = kill_results

    print("\n=== KILL CRITERIA ===")
    for k, v in kill_results.items():
        print(f"  {k}: {v['result']} — {v['note']}")

    print("\n=== SUMMARY ===")
    for section, data in summary.items():
        if isinstance(data, dict):
            print(f"  {section}:")
            for k2, v2 in data.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {section}: {data}")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
