"""Attention Layer Removal Safety: Expert removal in the high-cosine regime.

Parent experiment (expert_removal_graceful) showed:
  - cos<0.01 -> naive subtraction OK (<0.2% error)
  - cos>0.1  -> GS recompute required (but cheap: 1s at N=50)

Attention layers for related domains have cos=0.85 (documented in VISION.md,
measured in ffn_only_vs_all_modules). This experiment specifically tests:

  T1: Naive subtraction error at cos=0.85 (attention regime)
  T2: GS recompute cost for attention-only deltas (smaller D than full model)
  T3: Sweep cos from 0.01 to 0.95 to map the full error landscape
  T4: Partial removal -- remove attention part of expert, keep MLP
  T5: Production strategy simulation (mixed MLP+attention layers)

Kill criteria:
  K1: naive subtraction error >3% for attention layers at cos=0.85
      (EXPECTED TO TRIGGER -- this validates the parent finding)
  K2: GS recompute for attention layers takes >10s at N=50

Key insight being tested: K1 is expected to fail for naive subtraction at
cos=0.85 (confirming the parent's regime boundary), but K2 should pass
because attention-layer deltas are smaller than full-model deltas.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Reuse parent experiment utilities (adapted)
# ============================================================================

def generate_expert_set_controlled_cos(N: int, D: int,
                                        target_cos: float,
                                        seed: int = 42) -> list[np.ndarray]:
    """Generate N flattened delta vectors with controlled pairwise cosine.

    Uses the same alpha/beta mixing as parent but parameterized by target_cos
    directly. Each expert shares a common direction (alpha * shared) plus
    a unique direction (beta * unique).

    Pairwise cosine within cluster: cos(i,j) ~ alpha^2 = target_cos.

    Args:
        N: number of experts
        D: flattened delta dimension
        target_cos: target pairwise cosine similarity
        seed: random seed

    Returns:
        list of N flattened delta vectors, each shape (D,)
    """
    rng = np.random.RandomState(seed)

    # Shared direction
    shared = rng.randn(D)
    shared = shared / np.linalg.norm(shared)

    alpha = np.sqrt(target_cos)
    beta = np.sqrt(1.0 - target_cos)

    deltas = []
    for _ in range(N):
        # Random unique direction, orthogonalized against shared
        unique = rng.randn(D)
        proj = np.dot(unique, shared) * shared
        unique = unique - proj
        unique = unique / np.linalg.norm(unique)

        delta = alpha * shared + beta * unique
        # Scale to realistic magnitude
        delta = delta * rng.uniform(0.008, 0.012)
        deltas.append(delta)

    return deltas


def gram_schmidt(deltas: list[np.ndarray]) -> list[np.ndarray]:
    """Apply Gram-Schmidt orthogonalization to flattened delta vectors."""
    ortho = []
    for v in deltas:
        v = v.copy()
        for e in ortho:
            dot_ve = np.dot(v, e)
            dot_ee = np.dot(e, e)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e
        ortho.append(v)
    return ortho


def merge_with_gs(deltas: list[np.ndarray]) -> np.ndarray:
    """GS orthogonalize then sum all deltas."""
    ortho = gram_schmidt(deltas)
    return sum(ortho)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def reconstruction_error(w_method: np.ndarray, w_baseline: np.ndarray) -> float:
    """Relative reconstruction error: ||diff||_F / ||baseline||_F * 100."""
    diff_norm = np.linalg.norm(w_method - w_baseline)
    base_norm = np.linalg.norm(w_baseline)
    if base_norm < 1e-12:
        return 0.0
    return float(diff_norm / base_norm) * 100.0


def naive_removal(ortho_deltas: list[np.ndarray], merged: np.ndarray,
                  remove_idx: int) -> np.ndarray:
    """Naive subtraction: W_merged - delta_k'."""
    return merged - ortho_deltas[remove_idx]


def gs_recompute(deltas: list[np.ndarray], remove_idx: int) -> np.ndarray:
    """GS recomputation of remaining N-1 experts."""
    remaining = [d for i, d in enumerate(deltas) if i != remove_idx]
    return merge_with_gs(remaining)


# ============================================================================
# Attention-layer specific dimensions
# ============================================================================

# Qwen 0.5B: d=896, n_heads=14, head_dim=64
# Attention layer matrices: Q, K, V, O each (d, d) = (896, 896)
# But LoRA delta for attention is rank-r: shape d*d = 802816 per matrix
# For attention-only: 4 matrices, D_attn = 4 * d * d = 3,211,264
# For MLP: gate + up + down, D_mlp = 3 * d * d_ff
#   (Qwen 0.5B d_ff = 4864, so D_mlp ~ 3 * 896 * 4864 = 13,074,432)
# For all-modules: D_total = D_attn + D_mlp

D_SINGLE = 896 * 896          # 802,816 (one attention matrix)
D_ATTN = 4 * 896 * 896        # 3,211,264 (Q+K+V+O)
D_MLP = 3 * 896 * 4864        # 13,074,432 (gate+up+down, approximate)
D_ALL = D_ATTN + D_MLP        # 16,285,696 (all modules per layer)


# ============================================================================
# TEST 1: Naive subtraction at cos=0.85 (attention regime)
# ============================================================================

def test_attention_regime(seeds: list[int]) -> list[dict]:
    """Test naive subtraction specifically at cos=0.85 for attention-layer deltas."""
    print("\n" + "=" * 70)
    print("  TEST 1: Naive subtraction at cos=0.85 (attention layer regime)")
    print("  D_attn = {:,} (4 attention matrices at d=896)".format(D_ATTN))
    print("=" * 70)

    # Use D_ATTN for realistic attention-layer delta dimension
    # But D_ATTN=3.2M is large for numpy. Use D_SINGLE for speed,
    # then verify that dimension doesn't change the error (it shouldn't,
    # since error depends on cosine, not dimension).
    D_test = D_SINGLE  # 802,816 -- fast enough

    N_values = [10, 20, 50]
    results = []

    print(f"\n{'N':>4} {'Seed':>6} {'MeanCos':>9} {'Recon%':>9} "
          f"{'MaxPerExp%':>12} {'Naive(ms)':>10} {'GS_rec(ms)':>11}")
    print("-" * 75)

    for N in N_values:
        for seed in seeds:
            rng_remove = np.random.RandomState(seed + 1000)
            deltas = generate_expert_set_controlled_cos(N, D_test,
                                                         target_cos=0.85,
                                                         seed=seed)
            # Verify actual cosines
            cos_pairs = []
            for i in range(min(N, 20)):
                for j in range(i + 1, min(N, 20)):
                    cos_pairs.append(abs(cosine_sim(deltas[i], deltas[j])))
            mean_cos = np.mean(cos_pairs)

            # GS compose all N
            t0 = time.time()
            ortho_all = gram_schmidt(deltas)
            gs_time_all = time.time() - t0
            merged_all = sum(ortho_all)

            # Remove middle expert
            remove_idx = N // 2

            # Naive subtraction
            t0 = time.time()
            w_naive = naive_removal(ortho_all, merged_all, remove_idx)
            naive_time = time.time() - t0

            # GS recompute (ground truth)
            t0 = time.time()
            w_gt = gs_recompute(deltas, remove_idx)
            recompute_time = time.time() - t0

            recon_err = reconstruction_error(w_naive, w_gt)

            # Per-expert quality: projection score before/after
            active = [i for i in range(N) if i != remove_idx]
            max_per_exp = 0.0
            for i in active:
                q_naive = abs(cosine_sim(w_naive, deltas[i]))
                q_gt = abs(cosine_sim(w_gt, deltas[i]))
                if q_gt > 1e-12:
                    reg = abs(q_naive - q_gt) / q_gt * 100
                    max_per_exp = max(max_per_exp, reg)

            results.append({
                "test": "attn_cos085",
                "N": N,
                "seed": seed,
                "D": D_test,
                "target_cos": 0.85,
                "actual_mean_cos": mean_cos,
                "recon_error_pct": recon_err,
                "max_per_expert_pct": max_per_exp,
                "naive_time_ms": naive_time * 1000,
                "gs_recompute_time_ms": recompute_time * 1000,
            })

            print(f"{N:>4} {seed:>6} {mean_cos:>9.4f} {recon_err:>9.4f} "
                  f"{max_per_exp:>12.4f} {naive_time*1000:>10.2f} "
                  f"{recompute_time*1000:>11.2f}")

    return results


# ============================================================================
# TEST 2: GS recompute timing for attention-only deltas at scale
# ============================================================================

def test_gs_recompute_timing(seeds: list[int]) -> list[dict]:
    """Benchmark GS recompute cost for attention-only deltas."""
    print("\n" + "=" * 70)
    print("  TEST 2: GS recompute timing for attention-only deltas")
    print("  Using D_single = {:,} (one d*d matrix)".format(D_SINGLE))
    print("  Note: GS scales as O(N^2 * D), so D_attn = 4*D_single")
    print("=" * 70)

    # We test at D_SINGLE for speed, then extrapolate to D_ATTN and D_ALL
    D_test = D_SINGLE
    N_values = [10, 20, 50, 100]
    results = []

    print(f"\n{'N':>5} {'Cos':>6} {'GS_all(s)':>10} {'GS_rec(s)':>10} "
          f"{'Naive(ms)':>10} {'Extrap_attn(s)':>15} {'Extrap_all(s)':>14}")
    print("-" * 80)

    for N in N_values:
        seed = seeds[0]
        deltas = generate_expert_set_controlled_cos(N, D_test,
                                                     target_cos=0.85,
                                                     seed=seed)
        # Full GS
        t0 = time.time()
        ortho_all = gram_schmidt(deltas)
        gs_time_all = time.time() - t0
        merged_all = sum(ortho_all)

        remove_idx = N // 2

        # Naive
        t0 = time.time()
        w_naive = naive_removal(ortho_all, merged_all, remove_idx)
        naive_time = time.time() - t0

        # GS recompute
        t0 = time.time()
        w_gt = gs_recompute(deltas, remove_idx)
        recompute_time = time.time() - t0

        # Extrapolate: GS time scales linearly with D
        # D_attn = 4 * D_single, D_all = D_ALL / D_SINGLE * D_single
        extrap_attn = recompute_time * (D_ATTN / D_SINGLE)
        extrap_all = recompute_time * (D_ALL / D_SINGLE)

        results.append({
            "test": "timing",
            "N": N,
            "D_test": D_test,
            "cos": 0.85,
            "gs_all_s": gs_time_all,
            "gs_recompute_s": recompute_time,
            "naive_ms": naive_time * 1000,
            "extrap_attn_s": extrap_attn,
            "extrap_all_s": extrap_all,
        })

        print(f"{N:>5} {0.85:>6.2f} {gs_time_all:>10.3f} "
              f"{recompute_time:>10.3f} {naive_time*1000:>10.2f} "
              f"{extrap_attn:>15.3f} {extrap_all:>14.3f}")

    return results


# ============================================================================
# TEST 3: Full cosine sweep (map the error landscape)
# ============================================================================

def test_cosine_sweep(seeds: list[int]) -> list[dict]:
    """Sweep cosine from 0.01 to 0.95 to map naive subtraction error."""
    print("\n" + "=" * 70)
    print("  TEST 3: Cosine sweep (map error landscape)")
    print("=" * 70)

    D_test = D_SINGLE
    N = 20
    cos_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95]
    results = []

    print(f"\n{'Cos':>6} {'Seed':>6} {'ActCos':>8} {'Recon%':>10} "
          f"{'MaxPerExp%':>12} {'Regime':>15}")
    print("-" * 65)

    for target_cos in cos_values:
        for seed in seeds:
            deltas = generate_expert_set_controlled_cos(N, D_test,
                                                         target_cos=target_cos,
                                                         seed=seed)
            # Actual cosine check (sample)
            cos_sample = [abs(cosine_sim(deltas[i], deltas[j]))
                          for i in range(5) for j in range(i+1, 5)]
            actual_cos = np.mean(cos_sample)

            ortho_all = gram_schmidt(deltas)
            merged_all = sum(ortho_all)

            remove_idx = N // 2
            w_naive = naive_removal(ortho_all, merged_all, remove_idx)
            w_gt = gs_recompute(deltas, remove_idx)

            recon_err = reconstruction_error(w_naive, w_gt)

            # Per-expert max regression
            active = [i for i in range(N) if i != remove_idx]
            max_per_exp = 0.0
            for i in active:
                q_n = abs(cosine_sim(w_naive, deltas[i]))
                q_g = abs(cosine_sim(w_gt, deltas[i]))
                if q_g > 1e-12:
                    reg = abs(q_n - q_g) / q_g * 100
                    max_per_exp = max(max_per_exp, reg)

            if recon_err < 1.0:
                regime = "NAIVE OK"
            elif recon_err < 3.0:
                regime = "MARGINAL"
            else:
                regime = "RECOMPUTE"

            results.append({
                "test": "cosine_sweep",
                "target_cos": target_cos,
                "actual_cos": actual_cos,
                "seed": seed,
                "N": N,
                "recon_error_pct": recon_err,
                "max_per_expert_pct": max_per_exp,
                "regime": regime,
            })

            print(f"{target_cos:>6.2f} {seed:>6} {actual_cos:>8.4f} "
                  f"{recon_err:>10.4f} {max_per_exp:>12.4f} {regime:>15}")

    return results


# ============================================================================
# TEST 4: Partial removal (attention-only, keep MLP)
# ============================================================================

def test_partial_removal(seeds: list[int]) -> list[dict]:
    """Test removing only the attention component of an expert while keeping MLP.

    Each expert has two parts: attention delta (high cos=0.85) and MLP delta
    (low cos=0.001). We test:
    (a) Remove attention+MLP together (full expert removal)
    (b) Remove attention only (partial removal) -- MLP stays
    (c) Remove MLP only (partial removal) -- attention stays

    The key question: can partial removal exploit the different cosine regimes?
    Remove attention via GS recompute (mandatory at cos=0.85),
    keep MLP via naive subtraction (OK at cos=0.001).
    """
    print("\n" + "=" * 70)
    print("  TEST 4: Partial removal (attention-only vs MLP-only vs full)")
    print("=" * 70)

    # Use smaller D for computational feasibility
    D_attn = 50000   # attention portion (simulates Q+K+V+O)
    D_mlp = 150000   # MLP portion (simulates gate+up+down)
    D_full = D_attn + D_mlp

    N = 20
    results = []

    print(f"\n{'Mode':>15} {'Seed':>6} {'Recon%':>9} {'NaiveT(ms)':>11} "
          f"{'GSrecT(ms)':>11} {'Strategy':>20}")
    print("-" * 85)

    for seed in seeds:
        # Generate attention deltas (high cosine) and MLP deltas (low cosine)
        attn_deltas = generate_expert_set_controlled_cos(N, D_attn,
                                                          target_cos=0.85,
                                                          seed=seed)
        mlp_deltas = generate_expert_set_controlled_cos(N, D_mlp,
                                                         target_cos=0.001,
                                                         seed=seed + 5000)

        # Full expert deltas: concatenation of attention + MLP
        full_deltas = [np.concatenate([attn_deltas[i], mlp_deltas[i]])
                       for i in range(N)]

        remove_idx = N // 2

        # --- Mode A: Full expert removal ---
        ortho_full = gram_schmidt(full_deltas)
        merged_full = sum(ortho_full)

        t0 = time.time()
        w_naive_full = naive_removal(ortho_full, merged_full, remove_idx)
        naive_full_t = time.time() - t0

        t0 = time.time()
        w_gt_full = gs_recompute(full_deltas, remove_idx)
        gs_full_t = time.time() - t0

        recon_full = reconstruction_error(w_naive_full, w_gt_full)

        results.append({
            "test": "partial_full",
            "mode": "full_expert",
            "seed": seed,
            "recon_error_pct": recon_full,
            "naive_time_ms": naive_full_t * 1000,
            "gs_time_ms": gs_full_t * 1000,
        })
        print(f"{'full_expert':>15} {seed:>6} {recon_full:>9.4f} "
              f"{naive_full_t*1000:>11.2f} {gs_full_t*1000:>11.2f} "
              f"{'GS recompute':>20}")

        # --- Mode B: Remove attention only (keep MLP) ---
        # Attention portion: GS compose then recompute
        ortho_attn = gram_schmidt(attn_deltas)
        merged_attn = sum(ortho_attn)

        t0 = time.time()
        w_naive_attn = naive_removal(ortho_attn, merged_attn, remove_idx)
        naive_attn_t = time.time() - t0

        t0 = time.time()
        w_gt_attn = gs_recompute(attn_deltas, remove_idx)
        gs_attn_t = time.time() - t0

        recon_attn = reconstruction_error(w_naive_attn, w_gt_attn)

        results.append({
            "test": "partial_attn_only",
            "mode": "attn_only",
            "seed": seed,
            "recon_error_pct": recon_attn,
            "naive_time_ms": naive_attn_t * 1000,
            "gs_time_ms": gs_attn_t * 1000,
        })
        print(f"{'attn_only':>15} {seed:>6} {recon_attn:>9.4f} "
              f"{naive_attn_t*1000:>11.2f} {gs_attn_t*1000:>11.2f} "
              f"{'GS recompute':>20}")

        # --- Mode C: Remove MLP only (keep attention) ---
        ortho_mlp = gram_schmidt(mlp_deltas)
        merged_mlp = sum(ortho_mlp)

        t0 = time.time()
        w_naive_mlp = naive_removal(ortho_mlp, merged_mlp, remove_idx)
        naive_mlp_t = time.time() - t0

        t0 = time.time()
        w_gt_mlp = gs_recompute(mlp_deltas, remove_idx)
        gs_mlp_t = time.time() - t0

        recon_mlp = reconstruction_error(w_naive_mlp, w_gt_mlp)

        results.append({
            "test": "partial_mlp_only",
            "mode": "mlp_only",
            "seed": seed,
            "recon_error_pct": recon_mlp,
            "naive_time_ms": naive_mlp_t * 1000,
            "gs_time_ms": gs_mlp_t * 1000,
        })
        print(f"{'mlp_only':>15} {seed:>6} {recon_mlp:>9.4f} "
              f"{naive_mlp_t*1000:>11.2f} {gs_mlp_t*1000:>11.2f} "
              f"{'Naive OK':>20}")

        # --- Mode D: Hybrid strategy ---
        # Remove attention via GS recompute + remove MLP via naive subtraction
        # This is the "smart" approach: use the right tool per layer type
        t0 = time.time()
        w_hybrid_attn = gs_recompute(attn_deltas, remove_idx)
        w_hybrid_mlp = naive_removal(ortho_mlp, merged_mlp, remove_idx)
        w_hybrid = np.concatenate([w_hybrid_attn, w_hybrid_mlp])
        hybrid_time = time.time() - t0

        # Ground truth for full removal
        w_gt_combined = np.concatenate([w_gt_attn, w_gt_mlp])
        recon_hybrid = reconstruction_error(w_hybrid, w_gt_combined)

        results.append({
            "test": "partial_hybrid",
            "mode": "hybrid",
            "seed": seed,
            "recon_error_pct": recon_hybrid,
            "naive_time_ms": 0,
            "gs_time_ms": hybrid_time * 1000,
        })
        print(f"{'hybrid':>15} {seed:>6} {recon_hybrid:>9.4f} "
              f"{'N/A':>11} {hybrid_time*1000:>11.2f} "
              f"{'GS(attn)+naive(MLP)':>20}")

    return results


# ============================================================================
# TEST 5: Production strategy simulation
# ============================================================================

def test_production_strategy(seeds: list[int]) -> list[dict]:
    """Compare production strategies for expert removal.

    Strategy A: Naive subtraction everywhere (fast, inaccurate for attention)
    Strategy B: GS recompute everywhere (accurate, slower)
    Strategy C: Hybrid -- GS for attention, naive for MLP (best of both)
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Production strategy comparison")
    print("=" * 70)

    D_attn = 50000
    D_mlp = 150000

    N_values = [10, 20, 50]
    results = []

    print(f"\n{'N':>4} {'Strategy':>15} {'Recon%':>9} {'Time(ms)':>10} {'Speedup':>9}")
    print("-" * 55)

    for N in N_values:
        seed = seeds[0]

        attn_deltas = generate_expert_set_controlled_cos(N, D_attn,
                                                          target_cos=0.85,
                                                          seed=seed)
        mlp_deltas = generate_expert_set_controlled_cos(N, D_mlp,
                                                         target_cos=0.001,
                                                         seed=seed + 5000)
        full_deltas = [np.concatenate([attn_deltas[i], mlp_deltas[i]])
                       for i in range(N)]

        remove_idx = N // 2

        # Ground truth: separate GS recompute for attn and MLP, concatenate
        w_gt_attn = gs_recompute(attn_deltas, remove_idx)
        w_gt_mlp = gs_recompute(mlp_deltas, remove_idx)
        w_gt = np.concatenate([w_gt_attn, w_gt_mlp])

        # Strategy A: Naive everywhere
        ortho_full = gram_schmidt(full_deltas)
        merged_full = sum(ortho_full)
        t0 = time.time()
        w_a = naive_removal(ortho_full, merged_full, remove_idx)
        time_a = time.time() - t0
        recon_a = reconstruction_error(w_a, w_gt)

        # Strategy B: GS recompute everywhere
        t0 = time.time()
        remaining_full = [d for i, d in enumerate(full_deltas) if i != remove_idx]
        w_b = merge_with_gs(remaining_full)
        time_b = time.time() - t0
        recon_b = reconstruction_error(w_b, w_gt)

        # Strategy C: Hybrid (GS attn + naive MLP)
        ortho_mlp = gram_schmidt(mlp_deltas)
        merged_mlp = sum(ortho_mlp)
        t0 = time.time()
        w_c_attn = gs_recompute(attn_deltas, remove_idx)
        w_c_mlp = naive_removal(ortho_mlp, merged_mlp, remove_idx)
        w_c = np.concatenate([w_c_attn, w_c_mlp])
        time_c = time.time() - t0
        recon_c = reconstruction_error(w_c, w_gt)

        for strat, recon, t in [("naive_all", recon_a, time_a),
                                 ("gs_all", recon_b, time_b),
                                 ("hybrid", recon_c, time_c)]:
            speedup = time_b / max(t, 1e-9)
            results.append({
                "test": "production",
                "N": N,
                "strategy": strat,
                "recon_error_pct": recon,
                "time_ms": t * 1000,
                "speedup_vs_gs_all": speedup,
            })
            print(f"{N:>4} {strat:>15} {recon:>9.4f} {t*1000:>10.2f} "
                  f"{speedup:>9.1f}x")
        print()

    return results


# ============================================================================
# TEST 6: GS recompute at actual production dimensions
# ============================================================================

def test_actual_dimensions():
    """Time GS recompute at D_ATTN dimension for N=50 (K2 test)."""
    print("\n" + "=" * 70)
    print("  TEST 6: K2 test -- GS recompute at D_attn={:,} for N=50".format(D_ATTN))
    print("  WARNING: This test uses 3.2M-dimensional vectors. May take ~30s.")
    print("=" * 70)

    # D_ATTN = 3,211,264 -- this is the actual attention delta dimension
    # For N=50, this is 50 vectors of 3.2M floats = 1.2GB
    # GS requires O(N^2) dot products of O(D) each
    # Estimated: ~50^2 * 3.2M * 8 bytes ~ manageable

    N = 50
    seed = 42

    # Generate experts at actual dimension
    print("\n  Generating N=50 experts at D={:,}...".format(D_ATTN))
    t0 = time.time()
    deltas = generate_expert_set_controlled_cos(N, D_ATTN, target_cos=0.85,
                                                 seed=seed)
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    # GS compose all
    print("  Running GS on all N=50...")
    t0 = time.time()
    ortho_all = gram_schmidt(deltas)
    gs_all_time = time.time() - t0
    merged_all = sum(ortho_all)
    print(f"  GS compose time: {gs_all_time:.3f}s")

    # GS recompute (remove middle expert)
    remove_idx = N // 2
    print(f"  Running GS recompute (remove idx={remove_idx})...")
    t0 = time.time()
    w_gt = gs_recompute(deltas, remove_idx)
    recompute_time = time.time() - t0
    print(f"  GS recompute time: {recompute_time:.3f}s")

    # Naive subtraction
    t0 = time.time()
    w_naive = naive_removal(ortho_all, merged_all, remove_idx)
    naive_time = time.time() - t0

    recon_err = reconstruction_error(w_naive, w_gt)

    k2_pass = recompute_time < 10.0
    k1_fail = recon_err > 3.0

    print(f"\n  Results at actual D_attn dimension:")
    print(f"    GS recompute time: {recompute_time:.3f}s")
    print(f"    Naive time: {naive_time*1000:.2f}ms")
    print(f"    Recon error (naive): {recon_err:.4f}%")
    print(f"    K1 (naive error >3%): {'TRIGGERED (expected)' if k1_fail else 'PASS'}")
    print(f"    K2 (GS recompute <10s): {'PASS' if k2_pass else 'KILL'}")

    return {
        "test": "actual_dim_k2",
        "N": N,
        "D": D_ATTN,
        "gs_recompute_s": recompute_time,
        "naive_ms": naive_time * 1000,
        "recon_error_pct": recon_err,
        "k1_triggered": k1_fail,
        "k2_pass": k2_pass,
    }


# ============================================================================
# MAIN
# ============================================================================

def run_full_experiment():
    """Run the complete attention layer removal safety experiment."""
    print("=" * 70)
    print("  EXPERIMENT: Attention Layer Removal Safety")
    print("  Kill: K1: naive error >3% at cos=0.85 (expected)")
    print("        K2: GS recompute >10s at N=50")
    print("=" * 70)

    seeds = [42, 123, 777]
    all_results = {}

    # Run all tests
    all_results["t1_attn_regime"] = test_attention_regime(seeds)
    all_results["t2_timing"] = test_gs_recompute_timing(seeds)
    all_results["t3_cosine_sweep"] = test_cosine_sweep(seeds)
    all_results["t4_partial"] = test_partial_removal(seeds)
    all_results["t5_production"] = test_production_strategy(seeds)
    all_results["t6_actual_dim"] = test_actual_dimensions()

    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    # K1 assessment: naive subtraction error at cos=0.85
    t1 = all_results["t1_attn_regime"]
    recon_errors_085 = [r["recon_error_pct"] for r in t1]
    max_recon_085 = max(recon_errors_085)
    mean_recon_085 = np.mean(recon_errors_085)
    k1_triggered = max_recon_085 > 3.0

    print(f"\n  K1 (naive subtraction error >3% at cos=0.85):")
    print(f"    Mean recon error: {mean_recon_085:.4f}%")
    print(f"    Max recon error: {max_recon_085:.4f}%")
    print(f"    Threshold: 3.0%")
    if k1_triggered:
        print(f"    -> K1 TRIGGERED (as expected). Naive subtraction fails at cos=0.85.")
        print(f"       This CONFIRMS the parent finding: cos>0.1 requires GS recompute.")
    else:
        print(f"    -> K1 NOT triggered. Naive subtraction surprisingly OK at cos=0.85.")

    # K2 assessment: GS recompute time at N=50
    t6 = all_results["t6_actual_dim"]
    gs_time_n50 = t6["gs_recompute_s"]
    k2_pass = gs_time_n50 < 10.0

    print(f"\n  K2 (GS recompute <10s at N=50 for attention deltas):")
    print(f"    GS recompute time: {gs_time_n50:.3f}s")
    print(f"    Threshold: 10.0s")
    print(f"    -> K2 {'PASS' if k2_pass else 'KILL'}")

    # Hybrid strategy assessment
    t5 = all_results["t5_production"]
    hybrid_results = [r for r in t5 if r["strategy"] == "hybrid"]
    gs_all_results = [r for r in t5 if r["strategy"] == "gs_all"]
    naive_all_results = [r for r in t5 if r["strategy"] == "naive_all"]

    print(f"\n  Production strategy comparison (N=50):")
    for label, group in [("Naive all", naive_all_results),
                          ("GS all", gs_all_results),
                          ("Hybrid", hybrid_results)]:
        n50 = [r for r in group if r["N"] == 50]
        if n50:
            r = n50[0]
            print(f"    {label:>12}: error={r['recon_error_pct']:.4f}%, "
                  f"time={r['time_ms']:.2f}ms")

    # Cosine sweep regime boundaries
    t3 = all_results["t3_cosine_sweep"]
    print(f"\n  Cosine sweep regime boundaries (N=20):")
    for cos_val in [0.01, 0.1, 0.3, 0.5, 0.85, 0.95]:
        cos_group = [r for r in t3 if abs(r["target_cos"] - cos_val) < 0.005]
        if cos_group:
            mean_err = np.mean([r["recon_error_pct"] for r in cos_group])
            regime = cos_group[0]["regime"]
            print(f"    cos={cos_val:.2f}: mean recon error = {mean_err:.4f}% -> {regime}")

    # Partial removal assessment
    t4 = all_results["t4_partial"]
    print(f"\n  Partial removal viability:")
    for mode in ["full_expert", "attn_only", "mlp_only", "hybrid"]:
        mode_results = [r for r in t4 if r["mode"] == mode]
        if mode_results:
            mean_err = np.mean([r["recon_error_pct"] for r in mode_results])
            print(f"    {mode:>15}: mean recon error = {mean_err:.4f}%")

    # Final verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    if k1_triggered:
        print(f"\n  K1 TRIGGERED: Naive subtraction fails at cos=0.85 (error {max_recon_085:.2f}%).")
        print(f"  This is EXPECTED and CONFIRMS the parent finding.")
        print(f"  Attention layers REQUIRE GS recompute for expert removal.")
    else:
        print(f"\n  K1 NOT triggered: Naive subtraction works even at cos=0.85.")

    if k2_pass:
        print(f"\n  K2 PASS: GS recompute for attention layers = {gs_time_n50:.3f}s at N=50.")
        print(f"  Attention-only recompute is fast (well within 10s threshold).")
    else:
        print(f"\n  K2 KILL: GS recompute too slow ({gs_time_n50:.1f}s > 10s).")

    overall = "SUPPORTED" if k2_pass else "KILLED"
    print(f"\n  Overall hypothesis: {overall}")
    print(f"  Production recommendation: HYBRID strategy")
    print(f"    - MLP layers (cos~0.001): naive subtraction O(1)")
    print(f"    - Attention layers (cos~0.85): GS recompute per layer")
    print(f"    - Combined: accuracy of full recompute, speed of naive MLP")

    # Save results
    results_path = Path(__file__).parent / "results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results, k1_triggered, k2_pass


if __name__ == "__main__":
    t0 = time.time()
    results, k1, k2 = run_full_experiment()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
