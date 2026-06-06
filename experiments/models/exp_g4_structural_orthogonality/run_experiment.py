"""
exp_g4_structural_orthogonality — verify structural orthogonality at Gemma 4 dims.

Kill criterion:
  K1599: max|cos| <= 100*sqrt(r/d) at r=6 across d in {2816, 5376}, N=25, float32.

Pure algebra. No model load. No MLX kernel needed (NumPy QR is the standard path;
MLX float64 is CPU-only in 0.29.x and partition QR runs in ms for these sizes).
"""

import json
import sys
import time

import numpy as np


IS_SMOKE = "--smoke" in sys.argv

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
RANK = 6
N_ADAPTERS = 10 if IS_SMOKE else 25
D_VALUES = [2816, 5376]  # Gemma 4 26B-A4B hidden, Gemma 4 31B hidden


def construct_partition_qr(d: int, r: int, n: int, dtype=np.float32, seed: int = SEED) -> np.ndarray:
    """Partition QR: Q in R^{d x (n*r)}, Q^T Q = I. A_i = Q[:, i*r:(i+1)*r]."""
    assert n * r <= d, f"Capacity exceeded: n*r={n*r} > d={d}"
    rng = np.random.default_rng(seed)
    # Generate W in the target dtype so the QR runs in that precision
    W = rng.standard_normal((d, n * r)).astype(dtype)
    Q, _ = np.linalg.qr(W)
    return Q.astype(dtype)


def max_pairwise_cos(Q: np.ndarray, r: int, n: int) -> tuple[float, float]:
    """
    Return (max|cos| across all pairs (i!=j, k, l), mean|cos| across same).

    Since Q^T Q should be I, A_i has orthonormal columns so column-norms are 1
    and <A_i[:,k], A_j[:,l]> equals the cosine similarity directly.

    Verification is done in float64 regardless of Q's dtype: we want to measure
    the orthogonality of the float32-constructed Q without adding float32 matmul
    roundoff on top. This isolates the QR construction error from downstream use.
    """
    Q64 = Q.astype(np.float64)
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    for i in range(n):
        Ai = Q64[:, i * r:(i + 1) * r]
        # Verify column norms ~1 (sanity)
        for j in range(i + 1, n):
            Aj = Q64[:, j * r:(j + 1) * r]
            cross = Ai.T @ Aj  # (r, r)
            block_max = float(np.max(np.abs(cross)))
            block_mean = float(np.mean(np.abs(cross)))
            if block_max > max_abs:
                max_abs = block_max
            sum_abs += block_mean * r * r
            count += r * r
    mean_abs = sum_abs / count if count > 0 else 0.0
    return max_abs, mean_abs


def column_norm_deviation(Q: np.ndarray) -> float:
    """max |||col|| - 1| across all columns. Should be ~0 for orthonormal Q."""
    Q64 = Q.astype(np.float64)
    col_norms = np.linalg.norm(Q64, axis=0)
    return float(np.max(np.abs(col_norms - 1.0)))


def random_baseline_cos(d: int, r: int, n: int, dtype=np.float32, seed: int = SEED + 1) -> float:
    """
    Random i.i.d. unit-norm columns (no QR) — Grassmannian prediction ≈ sqrt(r/d).
    Each A_i is (d, r) with columns independently drawn and unit-normalized.
    """
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n):
        M = rng.standard_normal((d, r)).astype(dtype)
        M = M / np.linalg.norm(M, axis=0, keepdims=True)
        mats.append(M.astype(np.float64))
    max_abs = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cross = mats[i].T @ mats[j]
            block_max = float(np.max(np.abs(cross)))
            if block_max > max_abs:
                max_abs = block_max
    return max_abs


# ── Run ───────────────────────────────────────────────────────────────────────

results = {
    "is_smoke": IS_SMOKE,
    "rank": RANK,
    "n_adapters": N_ADAPTERS,
    "d_values": D_VALUES,
    "per_d": {},
}

all_pass = True

print(f"\n=== exp_g4_structural_orthogonality ===")
print(f"  r={RANK}, N={N_ADAPTERS}, seeds=42+1, smoke={IS_SMOKE}")

for d in D_VALUES:
    kill_threshold = 100.0 * (RANK / d) ** 0.5
    random_theory = (RANK / d) ** 0.5

    print(f"\n--- d={d} (kill threshold 100*sqrt(r/d)={kill_threshold:.4f}, random theory sqrt(r/d)={random_theory:.4f}) ---")

    # Partition QR (float32)
    t0 = time.perf_counter()
    Q_f32 = construct_partition_qr(d, RANK, N_ADAPTERS, dtype=np.float32)
    t_qr_f32 = time.perf_counter() - t0
    max_cos_f32, mean_cos_f32 = max_pairwise_cos(Q_f32, RANK, N_ADAPTERS)
    col_dev_f32 = column_norm_deviation(Q_f32)

    # Partition QR (float64) — reference
    Q_f64 = construct_partition_qr(d, RANK, N_ADAPTERS, dtype=np.float64)
    max_cos_f64, mean_cos_f64 = max_pairwise_cos(Q_f64, RANK, N_ADAPTERS)
    col_dev_f64 = column_norm_deviation(Q_f64)

    # Random subspace baseline (float32)
    random_max_cos = random_baseline_cos(d, RANK, N_ADAPTERS, dtype=np.float32)

    print(f"  QR float32:  max|cos|={max_cos_f32:.3e}  mean|cos|={mean_cos_f32:.3e}  col_dev={col_dev_f32:.3e}  ({t_qr_f32*1000:.1f} ms)")
    print(f"  QR float64:  max|cos|={max_cos_f64:.3e}  mean|cos|={mean_cos_f64:.3e}  col_dev={col_dev_f64:.3e}")
    print(f"  Random:      max|cos|={random_max_cos:.3e}  (predicted ≈ sqrt(r/d) = {random_theory:.3e})")

    passes = max_cos_f32 <= kill_threshold
    if not passes:
        all_pass = False
    print(f"  K1599 at d={d}: {'PASS' if passes else 'FAIL'} ({max_cos_f32:.3e} <= {kill_threshold:.3e})")

    results["per_d"][str(d)] = {
        "kill_threshold": kill_threshold,
        "random_theory": random_theory,
        "qr_f32_max_cos": float(max_cos_f32),
        "qr_f32_mean_cos": float(mean_cos_f32),
        "qr_f32_col_dev": float(col_dev_f32),
        "qr_f32_time_ms": float(t_qr_f32 * 1000),
        "qr_f64_max_cos": float(max_cos_f64),
        "qr_f64_mean_cos": float(mean_cos_f64),
        "qr_f64_col_dev": float(col_dev_f64),
        "random_max_cos": float(random_max_cos),
        "pass": bool(passes),
    }

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n=== Summary ===")
for d in D_VALUES:
    per = results["per_d"][str(d)]
    print(f"  d={d}: QR f32 max|cos|={per['qr_f32_max_cos']:.3e}  threshold={per['kill_threshold']:.3e}  "
          f"{'PASS' if per['pass'] else 'FAIL'}")

results["k1599_pass"] = bool(all_pass)
results["all_pass"] = bool(all_pass)
results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nVerdict: {results['verdict']}")
print("results.json written.")
sys.exit(0 if all_pass else 1)
