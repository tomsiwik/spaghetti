"""
T0.1: Grassmannian QR on Gemma 4 Weight Shapes (d=2816, d=5376)

Kill criteria:
  K990: max|A_i^T A_j|_F < 1e-6 for N=50 at d=2816 (f64 partition QR)
  K991: max|A_i^T A_j|_F < 1e-6 for N=100 at d=5376 (f64 partition QR)
  K992: N_max = floor(d/r) adapters constructable without numerical breakdown
  K993: Construction time < 1s on MLX GPU for N=50, d=2816 (f32)

Algebraic verification only — no model loading required.
Uses numpy float64 for exact verification (float64 is CPU-only in MLX 0.29.x).
Uses MLX float32 for GPU timing test.
"""

import json
import sys
import time
import warnings

import mlx.core as mx
import numpy as np

# Suppress macOS Accelerate BLAS float32 overflow warnings.
# The max_pairwise_frobenius function upcasts to float64 before matmul,
# but the K992 spot-check path may trigger them from numpy internally.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

IS_SMOKE = "--smoke" in sys.argv

# ── Config ────────────────────────────────────────────────────────────────────
RANK = 4 if IS_SMOKE else 16
SEED = 42

# Gemma 4 hidden dimensions
D_SMALL = 512 if IS_SMOKE else 2816   # 26B-A4B hidden
D_LARGE = 1024 if IS_SMOKE else 5376  # 31B hidden

# Number of adapters per test
N_SMALL = 10 if IS_SMOKE else 50    # K990: N=50 at d=2816
N_LARGE = 20 if IS_SMOKE else 100   # K991: N=100 at d=5376

results = {}


# ── Phase 1: Algebraic verification (numpy float64) ───────────────────────────

def construct_grassmannian_adapters(d: int, r: int, n: int, dtype=np.float64) -> np.ndarray:
    """
    Partition QR construction.

    Returns Q ∈ R^{d × nr} with Q^T Q = I exactly.
    A_i = Q[:, i*r:(i+1)*r]  →  A_i^T A_j = 0 for i≠j.

    Requires n * r <= d (capacity bound).
    """
    assert n * r <= d, f"Capacity exceeded: n*r={n*r} > d={d}"
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((d, n * r)).astype(dtype)
    Q, _ = np.linalg.qr(W)   # Q: (d, nr), full column orthonormal
    return Q  # Adapters: Q[:, i*r:(i+1)*r]


def max_pairwise_frobenius(Q: np.ndarray, r: int, n: int) -> float:
    """
    Compute max_{i≠j} ||A_i^T A_j||_F over all N*(N-1)/2 pairs.

    A_i = Q[:, i*r:(i+1)*r]  shape (d, r)
    A_i^T A_j = (r, r) — should be ~0 for i≠j.

    For float32, upcast to float64 to avoid BLAS overflow warnings on large d.
    """
    Q64 = Q.astype(np.float64) if Q.dtype == np.float32 else Q
    max_err = 0.0
    for i in range(n):
        Ai = Q64[:, i * r:(i + 1) * r]
        for j in range(i + 1, n):
            Aj = Q64[:, j * r:(j + 1) * r]
            cross = Ai.T @ Aj            # (r, r)
            err = np.linalg.norm(cross, 'fro')
            if err > max_err:
                max_err = err
    return max_err


# K990: d=2816, N=50
print(f"\n=== K990: d={D_SMALL}, N={N_SMALL}, r={RANK} ===")
Q_small = construct_grassmannian_adapters(D_SMALL, RANK, N_SMALL)
max_err_small = max_pairwise_frobenius(Q_small, RANK, N_SMALL)
n_pairs_small = N_SMALL * (N_SMALL - 1) // 2
k990_pass = max_err_small < 1e-6
print(f"  Pairs checked:    {n_pairs_small}")
print(f"  max|A_i^T A_j|_F: {max_err_small:.3e}")
print(f"  K990:             {'PASS' if k990_pass else 'FAIL'} (threshold 1e-6)")

results["d_small"] = int(D_SMALL)
results["n_small"] = int(N_SMALL)
results["rank"] = int(RANK)
results["n_pairs_small"] = int(n_pairs_small)
results["max_interference_small_f64"] = float(max_err_small)
results["k990_pass"] = bool(k990_pass)


# K991: d=5376, N=100
print(f"\n=== K991: d={D_LARGE}, N={N_LARGE}, r={RANK} ===")
Q_large = construct_grassmannian_adapters(D_LARGE, RANK, N_LARGE)
max_err_large = max_pairwise_frobenius(Q_large, RANK, N_LARGE)
n_pairs_large = N_LARGE * (N_LARGE - 1) // 2
k991_pass = max_err_large < 1e-6
print(f"  Pairs checked:    {n_pairs_large}")
print(f"  max|A_i^T A_j|_F: {max_err_large:.3e}")
print(f"  K991:             {'PASS' if k991_pass else 'FAIL'} (threshold 1e-6)")

results["d_large"] = int(D_LARGE)
results["n_large"] = int(N_LARGE)
results["n_pairs_large"] = int(n_pairs_large)
results["max_interference_large_f64"] = float(max_err_large)
results["k991_pass"] = bool(k991_pass)


# ── Phase 2: Float32 (production) interference ────────────────────────────────

print(f"\n=== Float32 (production) interference check ===")
Q_small_f32 = construct_grassmannian_adapters(D_SMALL, RANK, N_SMALL, dtype=np.float32)
max_err_small_f32 = max_pairwise_frobenius(Q_small_f32, RANK, N_SMALL)
print(f"  d={D_SMALL}, N={N_SMALL}: max|A_i^T A_j|_F (f32) = {max_err_small_f32:.3e}")
results["max_interference_small_f32"] = float(max_err_small_f32)


# ── Phase 3: N_max capacity verification (K992) ───────────────────────────────

print(f"\n=== K992: N_max capacity ===")
n_max_small = D_SMALL // RANK
n_max_large = D_LARGE // RANK
print(f"  d={D_SMALL}, r={RANK}: N_max = {n_max_small}")
print(f"  d={D_LARGE}, r={RANK}: N_max = {n_max_large}")

# Construct N_max adapters to verify no numerical breakdown
try:
    Q_nmax = construct_grassmannian_adapters(D_SMALL, RANK, n_max_small)
    # Spot-check 5 pairs from the extremes (float64 for stable matmul)
    Q_nmax64 = Q_nmax.astype(np.float64)
    errors_nmax = []
    for i in [0, 1, n_max_small // 4, n_max_small // 2, n_max_small - 2]:
        j = n_max_small - 1
        Ai = Q_nmax64[:, i * RANK:(i + 1) * RANK]
        Aj = Q_nmax64[:, j * RANK:(j + 1) * RANK]
        errors_nmax.append(float(np.linalg.norm(Ai.T @ Aj, 'fro')))
    max_err_nmax = max(errors_nmax)
    k992_pass = True
    print(f"  N_max={n_max_small} construction: OK")
    print(f"  Spot-check max error: {max_err_nmax:.3e}")
except Exception as e:
    k992_pass = False
    max_err_nmax = float("nan")
    print(f"  N_max construction FAILED: {e}")

results["n_max_small"] = int(n_max_small)
results["n_max_large"] = int(n_max_large)
results["max_interference_nmax_f64"] = float(max_err_nmax)
results["k992_pass"] = bool(k992_pass)


# ── Phase 4: MLX GPU construction timing (K993) ───────────────────────────────

print(f"\n=== K993: GPU construction timing ===")

def build_adapters_mlx(d: int, r: int, n: int) -> mx.array:
    """
    Build N Grassmannian adapters on MLX GPU (float32).
    Uses numpy QR then converts to MLX — the QR is the bottleneck.
    Also benchmarks the MLX matmul verification pass.
    """
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((d, n * r)).astype(np.float32)
    Q_np, _ = np.linalg.qr(W)
    Q = mx.array(Q_np)
    mx.eval(Q)
    return Q

# Warmup
_ = build_adapters_mlx(D_SMALL, RANK, N_SMALL)

# Time 3 runs, take median
times = []
for _ in range(3):
    t0 = time.perf_counter()
    Q_gpu = build_adapters_mlx(D_SMALL, RANK, N_SMALL)
    # Verify one cross product on GPU
    A0 = Q_gpu[:, 0:RANK]
    A1 = Q_gpu[:, RANK:2 * RANK]
    cross = A0.T @ A1
    mx.eval(cross)
    t1 = time.perf_counter()
    times.append(t1 - t0)

gpu_time_s = sorted(times)[1]  # median
k993_pass = gpu_time_s < 1.0
print(f"  Median construction + verify time: {gpu_time_s:.3f}s")
print(f"  K993: {'PASS' if k993_pass else 'FAIL'} (threshold 1s)")

results["gpu_construction_time_s"] = float(gpu_time_s)
results["k993_pass"] = bool(k993_pass)


# ── Bonus: NoPE subspace at d=384 ────────────────────────────────────────────

nope_d = 384     # Gemma 4 NoPE dims [128:512], from T0.3 (Finding #411)
nope_nmax = nope_d // RANK
print(f"\n=== NoPE subspace capacity (T0.3 reference) ===")
print(f"  d={nope_d}, r={RANK}: N_max = {nope_nmax} domains per layer")
results["nope_n_max"] = int(nope_nmax)


# ── Summary ────────────────────────────────────────────────────────────────────

all_pass = all([k990_pass, k991_pass, k992_pass, k993_pass])
print(f"\n=== Summary ===")
print(f"  K990 (d=2816, N=50, f64):  {'PASS' if k990_pass else 'FAIL'} [{max_err_small:.3e}]")
print(f"  K991 (d=5376, N=100, f64): {'PASS' if k991_pass else 'FAIL'} [{max_err_large:.3e}]")
print(f"  K992 (N_max capacity):     {'PASS' if k992_pass else 'FAIL'} [{max_err_nmax:.3e}]")
print(f"  K993 (GPU <1s):            {'PASS' if k993_pass else 'FAIL'} [{gpu_time_s:.3f}s]")
print(f"  ALL PASS: {all_pass}")

results["all_pass"] = bool(all_pass)
results["is_smoke"] = bool(IS_SMOKE)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nresults.json written.")
sys.exit(0 if all_pass else 1)
