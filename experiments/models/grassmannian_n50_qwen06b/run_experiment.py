"""
exp_grassmannian_n50_qwen06b: Grassmannian orthogonality at N=50 on Qwen3-0.6B.

Verifies Theorem 1 (QR construction → exact orthogonality) and Theorem 2 (memory bound)
at N=50 adapters, matching Qwen3-0.6B dimensions (d_in=1024, r=4).

K948: max|A_i^T A_j|_F < 1e-5 for all 1225 pairwise checks
K949: total adapter memory < 5GB
"""

import json
import sys
import time

import mlx.core as mx
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────

N = 50           # number of adapters
R = 4            # LoRA rank
D_IN = 1024      # Qwen3-0.6B hidden dim (q/k/v input dim)
N_LAYERS = 28    # Qwen3-0.6B layers
SEED = 42

# Qwen3-0.6B weight dimensions for memory calculation
WEIGHT_TYPES = {
    "q_proj":  {"d_in": 1024, "d_out": 2048},  # 16 heads × 128 head_dim = 2048
    "k_proj":  {"d_in": 1024, "d_out": 1024},
    "v_proj":  {"d_in": 1024, "d_out": 1024},
    "o_proj":  {"d_in": 2048, "d_out": 1024},  # output projection
    "gate":    {"d_in": 1024, "d_out": 3072},
    "up":      {"d_in": 1024, "d_out": 3072},
    "down":    {"d_in": 3072, "d_out": 1024},
}

K948_THRESHOLD = 1e-5
K949_THRESHOLD_GB = 5.0

# ── Grassmannian construction ───────────────────────────────────────────────

def build_grassmannian_slots(n: int, r: int, d_in: int, seed: int) -> np.ndarray:
    """
    Construct N Grassmannian A-matrices via global QR.

    X ~ N(0,1)^{d_in × (n*r)}, QR → Q with orthonormal columns.
    A_i = Q[:, i*r:(i+1)*r]  → A_i^T A_j = 0 exactly (float32 precision).

    Returns: (n, d_in, r) float32 array
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d_in, n * r)).astype(np.float32)
    Q, _ = np.linalg.qr(X, mode="reduced")  # Q: (d_in, n*r), columns orthonormal
    slots = Q.reshape(d_in, n, r).transpose(1, 0, 2)  # (n, d_in, r)
    return slots


def compute_pairwise_cosines(slots: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Compute max and mean |cos(A_i, A_j)| = ||A_i^T A_j||_F for all i < j.

    A_i^T A_j is r×r; ||.||_F measures total cross-subspace energy.
    For orthogonal subspaces: A_i^T A_j = 0 → ||A_i^T A_j||_F = 0.

    Returns: (max_cos, mean_cos, all_cross_norms)
    """
    n = slots.shape[0]
    cross_norms = []

    for i in range(n):
        for j in range(i + 1, n):
            # A_i^T A_j: (r, d_in) @ (d_in, r) = (r, r)
            cross = slots[i].T @ slots[j]  # (r, r)
            cross_norm = float(np.linalg.norm(cross, "fro"))
            cross_norms.append(cross_norm)

    cross_norms = np.array(cross_norms)
    return float(np.max(cross_norms)), float(np.mean(cross_norms)), cross_norms


# ── Memory calculation ──────────────────────────────────────────────────────

def compute_memory_bytes(
    n: int,
    r: int,
    n_layers: int,
    weight_types: dict,
    dtype_bytes: int = 2,  # float16
) -> dict:
    """
    Compute theoretical memory for N adapter sets.

    Returns breakdown by weight type + q_v_only total + all_types total.
    """
    results = {}
    total_all = 0
    total_qv = 0

    for name, dims in weight_types.items():
        d_in = dims["d_in"]
        d_out = dims["d_out"]
        # A: (r, d_in) per adapter per layer (we store transposed for matmul efficiency)
        # B: (d_out, r) per adapter per layer
        params_a = r * d_in
        params_b = d_out * r
        params_per_adapter_per_layer = params_a + params_b
        total_params = n * n_layers * params_per_adapter_per_layer
        total_bytes = total_params * dtype_bytes
        results[name] = {
            "params_per_adapter_per_layer": params_per_adapter_per_layer,
            "total_params": total_params,
            "total_mb": total_bytes / 1e6,
        }
        total_all += total_bytes
        if name in ("q_proj", "v_proj"):
            total_qv += total_bytes

    results["total_qv_only_mb"] = total_qv / 1e6
    results["total_all_types_mb"] = total_all / 1e6
    results["total_qv_only_gb"] = total_qv / 1e9
    results["total_all_types_gb"] = total_all / 1e9
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Building {N} Grassmannian A-matrices: d_in={D_IN}, r={R}")

    # 1. Construct slots
    slots = build_grassmannian_slots(N, R, D_IN, SEED)
    print(f"  Slots shape: {slots.shape}  dtype: {slots.dtype}")
    print(f"  N_max theoretical: {D_IN // R}")
    print(f"  N=50 uses {50 * R}/{D_IN} = {50*R/D_IN*100:.1f}% of subspace capacity")

    # 2. Verify self-orthonormality (A_i^T A_i should = I_r)
    self_checks = []
    for i in range(min(5, N)):
        diag_check = np.linalg.norm(slots[i].T @ slots[i] - np.eye(R), "fro")
        self_checks.append(diag_check)
    max_self_err = max(self_checks)
    print(f"\nSelf-orthonormality check (first 5): max ||A_i^T A_i - I||_F = {max_self_err:.2e}")

    # 3. Pairwise cosine computation (N*(N-1)/2 = 1225 pairs)
    n_pairs = N * (N - 1) // 2
    print(f"\nComputing {n_pairs} pairwise cross-subspace norms...")
    t_cross = time.time()
    max_cos, mean_cos, all_norms = compute_pairwise_cosines(slots)
    t_cross_end = time.time()
    print(f"  Done in {t_cross_end - t_cross:.2f}s")
    print(f"  max ||A_i^T A_j||_F = {max_cos:.2e}  (K948 threshold: {K948_THRESHOLD:.0e})")
    print(f"  mean ||A_i^T A_j||_F = {mean_cos:.2e}")

    k948_pass = max_cos < K948_THRESHOLD

    # 4. Memory calculation
    print("\nMemory breakdown (N=50, 28 layers, float16):")
    memory = compute_memory_bytes(N, R, N_LAYERS, WEIGHT_TYPES)
    for name in WEIGHT_TYPES:
        mb = memory[name]["total_mb"]
        print(f"  {name:12s}: {mb:6.1f} MB")
    print(f"\n  q+v only:    {memory['total_qv_only_mb']:.1f} MB")
    print(f"  All 7 types: {memory['total_all_types_mb']:.1f} MB")
    print(f"  All 7 types: {memory['total_all_types_gb']:.3f} GB  (K949 threshold: {K949_THRESHOLD_GB} GB)")

    k949_pass = memory["total_all_types_gb"] < K949_THRESHOLD_GB

    # 5. Percentile distribution of cross norms
    pct = np.percentile(all_norms, [50, 90, 99, 100])
    print(f"\nCross-norm distribution (N={N}, {n_pairs} pairs):")
    print(f"  p50={pct[0]:.2e}  p90={pct[1]:.2e}  p99={pct[2]:.2e}  max={pct[3]:.2e}")

    total_time = time.time() - t0

    # 6. Results
    results = {
        "experiment": "exp_grassmannian_n50_qwen06b",
        "config": {
            "N": N, "r": R, "d_in": D_IN, "n_layers": N_LAYERS,
            "n_max_theoretical": D_IN // R,
            "seed": SEED,
        },
        "orthogonality": {
            "n_pairs_checked": n_pairs,
            "max_cross_norm": max_cos,
            "mean_cross_norm": mean_cos,
            "max_self_error": max_self_err,
            "percentiles": {"p50": float(pct[0]), "p90": float(pct[1]),
                            "p99": float(pct[2]), "max": float(pct[3])},
        },
        "memory": {
            "q_v_only_mb": memory["total_qv_only_mb"],
            "all_7_types_mb": memory["total_all_types_mb"],
            "all_7_types_gb": memory["total_all_types_gb"],
        },
        "kill_criteria": {
            "K948_pass": k948_pass,
            "K948_detail": f"max_cos={max_cos:.2e} < {K948_THRESHOLD:.0e}",
            "K949_pass": k949_pass,
            "K949_detail": f"{memory['total_all_types_gb']:.3f}GB < {K949_THRESHOLD_GB}GB",
        },
        "total_time_s": total_time,
    }

    print(f"\n{'='*50}")
    print(f"K948 ({'PASS' if k948_pass else 'FAIL'}): max|A_i^T A_j|_F = {max_cos:.2e} (threshold: {K948_THRESHOLD:.0e})")
    print(f"K949 ({'PASS' if k949_pass else 'FAIL'}): total memory = {memory['total_all_types_gb']:.3f}GB < {K949_THRESHOLD_GB}GB")
    print(f"Total time: {total_time:.1f}s")

    with open("experiments/models/grassmannian_n50_qwen06b/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved.")
    return 0 if (k948_pass and k949_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
