#!/usr/bin/env python3
"""
Minimum Viable Base Dimension: Phase Transition in Expert Composition

Tests how expert composition quality scales with base model embedding dimension d.
Key question: is there a phase transition (sweet spot) where composition "turns on",
or does quality scale linearly with d?

Experimental design:
  1. For each d in {64, 128, 256, 512, 896, 1536, 2048, 3584}:
     a. Generate N synthetic LoRA adapters (rank-16, frozen-A Gaussian, trained-B sim)
     b. Measure pairwise |cos| of flattened delta vectors
     c. Compose N adapters via naive addition, measure signal retention
     d. Find N_max: largest N where max|cos| < tau
  2. Fit scaling laws: |cos| ~ d^beta, N_max ~ d^gamma
  3. Test for phase transition: piecewise-linear vs smooth power law

Three adapter generation modes:
  - Random: A ~ N(0, 1/d), B ~ N(0, 1/r) -- pure random baseline
  - Orthonormal: A from Stiefel manifold, B ~ N(0, 1/r) -- structured A
  - Trained-sim: A orthonormal, B trained on synthetic domain data -- realistic

Kill criteria:
  K1: Base <1.5B (d<1536) cannot support expert composition (PPL improvement <5%)
  K2: Expert quality scales linearly with base size (no sweet spot)
"""

import json
import time
import math
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, stats


# ===========================================================================
# Configuration
# ===========================================================================

# Model dimensions matching real Qwen2.5 models
DIMENSIONS = [64, 128, 256, 512, 896, 1536, 2048, 3584]
RANK = 16
N_EXPERTS_BASE = 8        # Base number of experts for interference measurement
N_EXPERTS_SWEEP = [4, 8, 16, 32, 64, 128]  # For N_max detection
TAU = 0.01                # Interference threshold (from structural_orthogonality_proof)
N_SEEDS = 3
N_LAYERS = 2              # MLP layers per model (keeps D manageable)
D_FF_RATIO = 4            # d_ff = D_FF_RATIO * d

# Map d -> approximate model size for labeling
D_TO_MODEL = {
    64: "micro-64", 128: "micro-128", 256: "micro-256", 512: "micro-512",
    896: "Qwen2.5-0.5B", 1536: "Qwen2.5-1.5B", 2048: "Qwen2.5-3B", 3584: "Qwen2.5-7B"
}


# ===========================================================================
# LoRA Adapter Generation
# ===========================================================================

def generate_stiefel_frame(d, r, rng):
    """Sample uniformly from Stiefel manifold St(r, d) via QR of Gaussian."""
    G = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]  # (d, r) orthonormal


def generate_random_B(r, d_out, rng, scale=1.0):
    """Random B matrix scaled by 1/sqrt(r)."""
    return rng.standard_normal((r, d_out)) * scale / math.sqrt(r)


def generate_domain_B(r, d_out, rng, domain_id, n_domains=8):
    """
    Simulate domain-specific B training by biasing certain directions.
    Each domain activates a different subset of output directions.
    This is NOT real training -- it's a geometric proxy that gives
    B matrices with domain-dependent structure.
    """
    B = rng.standard_normal((r, d_out)) / math.sqrt(r)
    # Each domain emphasizes a different band of output dims
    band_size = d_out // n_domains
    start = (domain_id * band_size) % d_out
    end = min(start + band_size, d_out)
    B[:, start:end] *= 3.0  # 3x amplification in domain-specific band
    return B


def compute_delta_vector(A, B, alpha=1.0):
    """
    Compute flattened LoRA delta: vec((alpha/r) * A @ B).
    A: (d, r), B: (r, d_out) -> delta: (d * d_out,)
    """
    r = A.shape[1]
    delta = (alpha / r) * (A @ B)  # (d, d_out)
    return delta.ravel()


def compute_full_delta(As, Bs, alpha=1.0):
    """
    Compute full multi-layer delta vector by concatenating per-layer deltas.
    As: list of (d, r) per layer
    Bs: list of (r, d_out) per layer
    Returns: flattened vector of dimension sum(d_l * d_out_l)
    """
    deltas = []
    for A, B in zip(As, Bs):
        deltas.append(compute_delta_vector(A, B, alpha))
    return np.concatenate(deltas)


# ===========================================================================
# Interference Measurement
# ===========================================================================

def pairwise_cosines(delta_vectors):
    """
    Compute all pairwise |cos| between delta vectors.
    delta_vectors: list of 1D arrays
    Returns: list of |cos| values (N*(N-1)/2 pairs)
    """
    N = len(delta_vectors)
    norms = [np.linalg.norm(v) for v in delta_vectors]
    cosines = []
    for i in range(N):
        for j in range(i + 1, N):
            if norms[i] < 1e-12 or norms[j] < 1e-12:
                cosines.append(0.0)
            else:
                cos = abs(np.dot(delta_vectors[i], delta_vectors[j]) / (norms[i] * norms[j]))
                cosines.append(float(cos))
    return cosines


def signal_retention_after_merge(delta_vectors):
    """
    Measure how much per-expert signal survives N-way addition.

    For expert i, signal retention = ||v_i||^2 / ||v_merged||^2 * N
    Perfect retention (orthogonal): each expert contributes 1/N of merged energy.
    Interference: cross-terms change the ratio.

    Returns: mean signal retention ratio (1.0 = perfect, <1 = destructive interference)
    """
    N = len(delta_vectors)
    if N < 2:
        return 1.0

    # Merged delta
    merged = sum(delta_vectors)
    merged_norm_sq = np.dot(merged, merged)

    if merged_norm_sq < 1e-20:
        return 0.0

    # Sum of individual energies
    individual_energy = sum(np.dot(v, v) for v in delta_vectors)

    # Ratio: merged_energy / individual_energy
    # = 1 + (cross_terms / individual_energy)
    # Perfect orthogonality: ratio = 1.0
    # Constructive interference: ratio > 1.0
    # Destructive interference: ratio < 1.0
    ratio = float(merged_norm_sq / individual_energy)
    return ratio


def composition_quality_metric(delta_vectors):
    """
    Measure composition quality: for each expert, how much of its unique
    contribution is preserved after merging with all others?

    Quality_i = cos(v_i, projection of v_i onto merged - sum_{j!=i} v_j)
    Simplifies to: cos(v_i, v_i) = 1 when all others are orthogonal.

    Better metric: effective rank of the stacked delta matrix.
    """
    N = len(delta_vectors)
    if N < 2:
        return {"effective_rank_ratio": 1.0, "min_singular_ratio": 1.0}

    # Stack deltas into matrix (N, D)
    D = len(delta_vectors[0])
    M = np.array(delta_vectors)  # (N, D)

    # SVD of stacked deltas
    # Use min(N, D) singular values -- for large D, just compute M @ M^T
    if N <= D:
        gram = M @ M.T  # (N, N)
        eigvals = np.linalg.eigvalsh(gram)
        eigvals = np.maximum(eigvals, 0)  # clip numerical negatives
        svals = np.sqrt(eigvals[::-1])  # descending
    else:
        # This shouldn't happen in our setup (N << D)
        svals = np.linalg.svd(M, compute_uv=False)

    # Effective rank (Shannon entropy of normalized singular values)
    svals_pos = svals[svals > 1e-10]
    if len(svals_pos) == 0:
        return {"effective_rank_ratio": 0.0, "min_singular_ratio": 0.0}

    p = svals_pos ** 2 / np.sum(svals_pos ** 2)
    entropy = -np.sum(p * np.log(p))
    eff_rank = math.exp(entropy)

    # Ratio vs maximum possible (= N if all orthogonal and equal norm)
    eff_rank_ratio = eff_rank / N

    # Min/max singular value ratio (condition number indicator)
    min_sv_ratio = float(svals_pos[-1] / svals_pos[0]) if len(svals_pos) > 1 else 1.0

    return {
        "effective_rank_ratio": float(eff_rank_ratio),
        "min_singular_ratio": min_sv_ratio,
        "effective_rank": float(eff_rank),
        "n_experts": N
    }


# ===========================================================================
# N_max Detection: Binary search for maximum composable experts
# ===========================================================================

def find_n_max(d, r, tau, rng, n_layers=2, d_ff_ratio=4, max_N=256):
    """
    Find maximum N experts at dimension d such that max|cos| < tau.
    Uses Stiefel A matrices (orthonormal) with random B.
    Binary search over N.
    """
    d_ff = d * d_ff_ratio

    def max_cos_at_N(N):
        # Generate N experts
        deltas = []
        for i in range(N):
            layers_delta = []
            for l in range(n_layers):
                A = generate_stiefel_frame(d, r, rng)
                B = generate_random_B(r, d_ff, rng)
                layers_delta.append(compute_delta_vector(A, B))
                # Second linear per layer: d_ff -> d
                A2 = generate_stiefel_frame(d_ff, r, rng)
                B2 = generate_random_B(r, d, rng)
                layers_delta.append(compute_delta_vector(A2, B2))
            deltas.append(np.concatenate(layers_delta))

        cosines = pairwise_cosines(deltas)
        return max(cosines) if cosines else 0.0

    # Binary search
    lo, hi = 2, max_N
    best_N = 2

    # First check if even max_N is below tau
    mc = max_cos_at_N(max_N)
    if mc < tau:
        return max_N  # All N up to max are fine

    while lo <= hi:
        mid = (lo + hi) // 2
        mc = max_cos_at_N(mid)
        if mc < tau:
            best_N = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best_N


# ===========================================================================
# Theoretical Predictions
# ===========================================================================

def theoretical_cos_bound(d, r):
    """Random subspace bound: sqrt(r/d)."""
    return math.sqrt(r / d)


def theoretical_n_max(d, r):
    """Capacity bound: d^2/r^2 (from Grassmannian packing)."""
    return (d / r) ** 2


def welch_bound(d, r, N):
    """Welch bound for N r-dim subspaces in R^d."""
    Nr = N * r
    if Nr <= d:
        return 0.0  # Perfect orthogonality possible
    return math.sqrt(r * (Nr - d) / (d * (Nr - r)))


# ===========================================================================
# Phase Transition Detection
# ===========================================================================

def fit_power_law(x, y):
    """Fit y = a * x^beta via log-log regression. Returns (a, beta, R2)."""
    log_x = np.log(np.array(x, dtype=float))
    log_y = np.log(np.array(y, dtype=float))

    mask = np.isfinite(log_x) & np.isfinite(log_y)
    log_x, log_y = log_x[mask], log_y[mask]

    if len(log_x) < 2:
        return 1.0, 0.0, 0.0

    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    a = math.exp(intercept)
    return a, slope, r_value ** 2


def fit_piecewise_linear(x, y, n_breakpoints=1):
    """
    Fit piecewise linear in log-log space.
    Returns breakpoint d and two slopes.
    Tests if there is a genuine phase transition.
    """
    log_x = np.log(np.array(x, dtype=float))
    log_y = np.log(np.array(y, dtype=float))

    if len(log_x) < 4:
        return None  # Not enough data for piecewise

    best_bic = float('inf')
    best_result = None

    # Try each interior point as breakpoint
    for bp_idx in range(1, len(log_x) - 2):
        bp = log_x[bp_idx]

        # Fit two segments
        mask_lo = log_x <= bp
        mask_hi = log_x >= bp

        if sum(mask_lo) < 2 or sum(mask_hi) < 2:
            continue

        # Segment 1
        s1 = stats.linregress(log_x[mask_lo], log_y[mask_lo])
        # Segment 2
        s2 = stats.linregress(log_x[mask_hi], log_y[mask_hi])

        # Residuals
        resid_lo = log_y[mask_lo] - (s1.slope * log_x[mask_lo] + s1.intercept)
        resid_hi = log_y[mask_hi] - (s2.slope * log_x[mask_hi] + s2.intercept)
        resid = np.concatenate([resid_lo, resid_hi])

        n = len(resid)
        k = 4  # 2 slopes + 2 intercepts
        sse = np.sum(resid ** 2)

        if sse < 1e-20:
            bic = -n * 100  # perfect fit
        else:
            bic = n * math.log(sse / n) + k * math.log(n)

        if bic < best_bic:
            best_bic = bic
            best_result = {
                "breakpoint_d": float(math.exp(bp)),
                "breakpoint_log_d": float(bp),
                "slope_below": float(s1.slope),
                "slope_above": float(s2.slope),
                "r2_below": float(s1.rvalue ** 2),
                "r2_above": float(s2.rvalue ** 2),
                "bic": float(bic),
                "bp_idx": bp_idx
            }

    # Compare with single power law BIC
    single = stats.linregress(log_x, log_y)
    resid_single = log_y - (single.slope * log_x + single.intercept)
    n = len(log_x)
    k_single = 2
    sse_single = np.sum(resid_single ** 2)
    if sse_single < 1e-20:
        bic_single = -n * 100
    else:
        bic_single = n * math.log(sse_single / n) + k_single * math.log(n)

    if best_result is not None:
        best_result["bic_single"] = float(bic_single)
        best_result["bic_improvement"] = float(bic_single - best_bic)
        # Positive = piecewise is better
        best_result["prefers_piecewise"] = bool(best_result["bic_improvement"] > 2.0)

    return best_result


# ===========================================================================
# Main Experiment
# ===========================================================================

def run_experiment(seeds=None, dimensions=None, rank=None):
    """Run the full minimum viable base experiment."""
    if seeds is None:
        seeds = list(range(N_SEEDS))
    if dimensions is None:
        dimensions = DIMENSIONS
    if rank is None:
        rank = RANK

    results = {
        "config": {
            "dimensions": dimensions,
            "rank": rank,
            "n_experts_base": N_EXPERTS_BASE,
            "n_layers": N_LAYERS,
            "d_ff_ratio": D_FF_RATIO,
            "tau": TAU,
            "seeds": seeds,
            "n_experts_sweep": N_EXPERTS_SWEEP,
        },
        "per_dimension": {},
        "scaling_laws": {},
        "phase_transition": {},
        "kill_criteria": {},
        "timing": {},
    }

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Experiment 1: Interference scaling with d
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("EXPERIMENT 1: Interference Scaling with Dimension d")
    print("=" * 70)

    all_mean_cos = {d: [] for d in dimensions}
    all_max_cos = {d: [] for d in dimensions}
    all_signal_retention = {d: [] for d in dimensions}
    all_eff_rank_ratio = {d: [] for d in dimensions}

    for d in dimensions:
        d_ff = d * D_FF_RATIO
        D_flat = N_LAYERS * 2 * d * d_ff  # Total parameter-space dimension
        model_name = D_TO_MODEL.get(d, f"d={d}")

        print(f"\n--- d={d} ({model_name}), D_flat={D_flat:,} ---")

        for seed in seeds:
            rng = np.random.RandomState(seed)

            # Generate N_EXPERTS_BASE domain-specific adapters
            deltas = []
            for expert_id in range(N_EXPERTS_BASE):
                layers_delta = []
                for l in range(N_LAYERS):
                    # W1: d -> d_ff
                    A1 = generate_stiefel_frame(d, rank, rng)
                    B1 = generate_domain_B(rank, d_ff, rng, expert_id, N_EXPERTS_BASE)
                    layers_delta.append(compute_delta_vector(A1, B1))
                    # W2: d_ff -> d
                    A2 = generate_stiefel_frame(d_ff, rank, rng)
                    B2 = generate_domain_B(rank, d, rng, expert_id, N_EXPERTS_BASE)
                    layers_delta.append(compute_delta_vector(A2, B2))
                deltas.append(np.concatenate(layers_delta))

            # Pairwise cosines
            cosines = pairwise_cosines(deltas)
            mean_cos = float(np.mean(cosines))
            max_cos = float(np.max(cosines))

            # Signal retention
            sr = signal_retention_after_merge(deltas)

            # Composition quality
            cq = composition_quality_metric(deltas)

            all_mean_cos[d].append(mean_cos)
            all_max_cos[d].append(max_cos)
            all_signal_retention[d].append(sr)
            all_eff_rank_ratio[d].append(cq["effective_rank_ratio"])

            print(f"  seed={seed}: mean|cos|={mean_cos:.6f}, max|cos|={max_cos:.6f}, "
                  f"signal_ret={sr:.4f}, eff_rank_ratio={cq['effective_rank_ratio']:.4f}")

        # Store per-dimension results
        results["per_dimension"][str(d)] = {
            "model": model_name,
            "D_flat": D_flat,
            "theoretical_cos_bound": theoretical_cos_bound(d, rank),
            "theoretical_n_max": theoretical_n_max(d, rank),
            "welch_bound_N8": welch_bound(d, rank, N_EXPERTS_BASE),
            "mean_cos": {
                "values": all_mean_cos[d],
                "mean": float(np.mean(all_mean_cos[d])),
                "std": float(np.std(all_mean_cos[d])),
            },
            "max_cos": {
                "values": all_max_cos[d],
                "mean": float(np.mean(all_max_cos[d])),
                "std": float(np.std(all_max_cos[d])),
            },
            "signal_retention": {
                "values": all_signal_retention[d],
                "mean": float(np.mean(all_signal_retention[d])),
                "std": float(np.std(all_signal_retention[d])),
            },
            "effective_rank_ratio": {
                "values": all_eff_rank_ratio[d],
                "mean": float(np.mean(all_eff_rank_ratio[d])),
                "std": float(np.std(all_eff_rank_ratio[d])),
            },
            "below_tau": all(c < TAU for c in all_max_cos[d]),
        }

    # -----------------------------------------------------------------------
    # Experiment 2: N_max detection per d
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Maximum Composable Experts (N_max) per Dimension")
    print("=" * 70)

    n_max_results = {}
    for d in dimensions:
        d_ff = d * D_FF_RATIO
        model_name = D_TO_MODEL.get(d, f"d={d}")

        # For large d, N_max will be very large; cap at 256 for tractability
        max_N_search = min(256, int(theoretical_n_max(d, rank)))
        max_N_search = max(max_N_search, 8)  # At least try up to 8

        n_maxes = []
        for seed in seeds:
            rng = np.random.RandomState(seed + 1000)
            nm = find_n_max(d, rank, TAU, rng, N_LAYERS, D_FF_RATIO, max_N=max_N_search)
            n_maxes.append(nm)

        mean_nmax = float(np.mean(n_maxes))
        print(f"  d={d:5d} ({model_name:17s}): N_max = {n_maxes} "
              f"(mean={mean_nmax:.0f}, theory={theoretical_n_max(d, rank):.0f})")

        n_max_results[str(d)] = {
            "model": model_name,
            "n_max_per_seed": n_maxes,
            "n_max_mean": mean_nmax,
            "n_max_theory": theoretical_n_max(d, rank),
            "ratio_to_theory": mean_nmax / theoretical_n_max(d, rank),
            "search_cap": max_N_search,
        }

    results["n_max"] = n_max_results

    # -----------------------------------------------------------------------
    # Experiment 3: Composition degradation curves (N sweep per d)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Composition Degradation Curves")
    print("=" * 70)

    degradation = {}
    # Only test a subset of dimensions for the sweep (large d is expensive)
    sweep_dims = [d for d in dimensions if d <= 1536]

    for d in sweep_dims:
        d_ff = d * D_FF_RATIO
        model_name = D_TO_MODEL.get(d, f"d={d}")

        print(f"\n--- d={d} ({model_name}) ---")

        sweep_results = {}
        for N in N_EXPERTS_SWEEP:
            if N > 256 and d < 512:  # Skip very large N for small d
                continue

            cos_means = []
            cos_maxes = []
            sr_values = []
            erk_values = []

            for seed in seeds:
                rng = np.random.RandomState(seed + 2000 + N)

                deltas = []
                for expert_id in range(N):
                    layers_delta = []
                    for l in range(N_LAYERS):
                        A1 = generate_stiefel_frame(d, rank, rng)
                        B1 = generate_domain_B(rank, d_ff, rng, expert_id, N)
                        layers_delta.append(compute_delta_vector(A1, B1))
                        A2 = generate_stiefel_frame(d_ff, rank, rng)
                        B2 = generate_domain_B(rank, d, rng, expert_id, N)
                        layers_delta.append(compute_delta_vector(A2, B2))
                    deltas.append(np.concatenate(layers_delta))

                cosines = pairwise_cosines(deltas)
                cos_means.append(float(np.mean(cosines)))
                cos_maxes.append(float(np.max(cosines)))
                sr_values.append(signal_retention_after_merge(deltas))
                cq = composition_quality_metric(deltas)
                erk_values.append(cq["effective_rank_ratio"])

            sweep_results[str(N)] = {
                "mean_cos_mean": float(np.mean(cos_means)),
                "max_cos_mean": float(np.mean(cos_maxes)),
                "signal_retention_mean": float(np.mean(sr_values)),
                "eff_rank_ratio_mean": float(np.mean(erk_values)),
            }

            print(f"  N={N:4d}: mean|cos|={np.mean(cos_means):.6f}, "
                  f"max|cos|={np.mean(cos_maxes):.6f}, "
                  f"sig_ret={np.mean(sr_values):.4f}, "
                  f"eff_rank_ratio={np.mean(erk_values):.4f}")

        degradation[str(d)] = sweep_results

    results["degradation_curves"] = degradation

    # -----------------------------------------------------------------------
    # Analysis: Scaling Laws
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS: Scaling Laws")
    print("=" * 70)

    # Cosine vs d power law
    d_vals = [d for d in dimensions]
    mean_cos_vals = [float(np.mean(all_mean_cos[d])) for d in dimensions]

    a_cos, beta_cos, r2_cos = fit_power_law(d_vals, mean_cos_vals)
    print(f"\nCosine scaling: |cos| = {a_cos:.4f} * d^({beta_cos:.3f}), R^2={r2_cos:.4f}")

    results["scaling_laws"]["cosine_vs_d"] = {
        "a": a_cos, "beta": beta_cos, "r2": r2_cos,
        "formula": f"|cos| = {a_cos:.4f} * d^({beta_cos:.3f})"
    }

    # N_max vs d power law
    nmax_vals = [float(np.mean(n_max_results[str(d)]["n_max_per_seed"])) for d in dimensions]
    # Filter out capped values for fit
    fit_d = []
    fit_nmax = []
    for d, nm in zip(d_vals, nmax_vals):
        cap = n_max_results[str(d)]["search_cap"]
        if nm < cap:  # Only use non-capped values
            fit_d.append(d)
            fit_nmax.append(nm)

    if len(fit_d) >= 2:
        a_nmax, gamma_nmax, r2_nmax = fit_power_law(fit_d, fit_nmax)
        print(f"N_max scaling: N_max = {a_nmax:.4f} * d^({gamma_nmax:.3f}), R^2={r2_nmax:.4f}")
        results["scaling_laws"]["nmax_vs_d"] = {
            "a": a_nmax, "gamma": gamma_nmax, "r2": r2_nmax,
            "formula": f"N_max = {a_nmax:.4f} * d^({gamma_nmax:.3f})",
            "n_points_used": len(fit_d),
        }
    else:
        print("N_max scaling: insufficient non-capped data points")
        results["scaling_laws"]["nmax_vs_d"] = {"note": "all N_max values hit search cap"}

    # Signal retention vs d
    sr_vals = [float(np.mean(all_signal_retention[d])) for d in dimensions]

    # Test: does signal retention approach 1.0 as d increases?
    sr_deficit = [1.0 - sr for sr in sr_vals]
    if all(s > 0 for s in sr_deficit):
        a_sr, beta_sr, r2_sr = fit_power_law(d_vals, sr_deficit)
        print(f"Signal retention deficit: (1-SR) = {a_sr:.4f} * d^({beta_sr:.3f}), R^2={r2_sr:.4f}")
        results["scaling_laws"]["signal_retention_deficit"] = {
            "a": a_sr, "beta": beta_sr, "r2": r2_sr,
        }

    # -----------------------------------------------------------------------
    # Analysis: Phase Transition
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS: Phase Transition Detection")
    print("=" * 70)

    # Test piecewise vs smooth for cosine
    pw_cos = fit_piecewise_linear(d_vals, mean_cos_vals)
    if pw_cos is not None:
        print(f"\nCosine phase transition:")
        print(f"  Breakpoint: d={pw_cos['breakpoint_d']:.0f}")
        print(f"  Slope below: {pw_cos['slope_below']:.3f}")
        print(f"  Slope above: {pw_cos['slope_above']:.3f}")
        print(f"  BIC improvement (piecewise over smooth): {pw_cos['bic_improvement']:.2f}")
        print(f"  Prefers piecewise: {pw_cos['prefers_piecewise']}")
        results["phase_transition"]["cosine"] = pw_cos

    # Phase transition for signal retention
    pw_sr = fit_piecewise_linear(d_vals, sr_vals)
    if pw_sr is not None:
        print(f"\nSignal retention phase transition:")
        print(f"  Breakpoint: d={pw_sr['breakpoint_d']:.0f}")
        print(f"  Slope below: {pw_sr['slope_below']:.3f}")
        print(f"  Slope above: {pw_sr['slope_above']:.3f}")
        print(f"  BIC improvement: {pw_sr['bic_improvement']:.2f}")
        print(f"  Prefers piecewise: {pw_sr['prefers_piecewise']}")
        results["phase_transition"]["signal_retention"] = pw_sr

    # Effective rank ratio phase transition
    erk_vals = [float(np.mean(all_eff_rank_ratio[d])) for d in dimensions]
    pw_erk = fit_piecewise_linear(d_vals, erk_vals)
    if pw_erk is not None:
        print(f"\nEffective rank ratio phase transition:")
        print(f"  Breakpoint: d={pw_erk['breakpoint_d']:.0f}")
        print(f"  BIC improvement: {pw_erk['bic_improvement']:.2f}")
        print(f"  Prefers piecewise: {pw_erk['prefers_piecewise']}")
        results["phase_transition"]["effective_rank_ratio"] = pw_erk

    # -----------------------------------------------------------------------
    # Analysis: Kill Criteria Assessment
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: Base <1.5B cannot support expert composition
    # We translate this to: at d=896 (0.5B), composition metrics are poor;
    # at d=1536 (1.5B), they are sufficient.

    d_05B = 896
    d_15B = 1536

    if str(d_05B) in results["per_dimension"] and str(d_15B) in results["per_dimension"]:
        cos_05B = results["per_dimension"][str(d_05B)]["mean_cos"]["mean"]
        cos_15B = results["per_dimension"][str(d_15B)]["mean_cos"]["mean"]
        sr_05B = results["per_dimension"][str(d_05B)]["signal_retention"]["mean"]
        sr_15B = results["per_dimension"][str(d_15B)]["signal_retention"]["mean"]
        erk_05B = results["per_dimension"][str(d_05B)]["effective_rank_ratio"]["mean"]
        erk_15B = results["per_dimension"][str(d_15B)]["effective_rank_ratio"]["mean"]

        # Improvement from 0.5B to 1.5B
        cos_improvement = (cos_05B - cos_15B) / cos_05B * 100
        sr_improvement = (sr_15B - sr_05B) / sr_05B * 100  # Higher is better

        k1_result = {
            "cos_0.5B": cos_05B,
            "cos_1.5B": cos_15B,
            "cos_improvement_pct": cos_improvement,
            "signal_retention_0.5B": sr_05B,
            "signal_retention_1.5B": sr_15B,
            "sr_improvement_pct": sr_improvement,
            "eff_rank_ratio_0.5B": erk_05B,
            "eff_rank_ratio_1.5B": erk_15B,
            "both_below_tau": (
                results["per_dimension"][str(d_05B)]["below_tau"] and
                results["per_dimension"][str(d_15B)]["below_tau"]
            ),
        }

        # K1 survives if 0.5B is substantially worse than 1.5B
        # Translation: the jump from d=896 to d=1536 should show >5% improvement
        # in composition metrics (not PPL since we have no real training)
        k1_survives = cos_improvement > 5.0
        k1_result["survives"] = k1_survives

        print(f"\nK1: Can base <1.5B support composition?")
        print(f"  0.5B (d={d_05B}): mean|cos|={cos_05B:.6f}, SR={sr_05B:.4f}, ERR={erk_05B:.4f}")
        print(f"  1.5B (d={d_15B}): mean|cos|={cos_15B:.6f}, SR={sr_15B:.4f}, ERR={erk_15B:.4f}")
        print(f"  Cos improvement: {cos_improvement:.1f}%")
        print(f"  Both below tau={TAU}: {k1_result['both_below_tau']}")
        print(f"  K1 survives: {k1_survives}")

        results["kill_criteria"]["K1"] = k1_result

    # K2: Expert quality scales linearly (no sweet spot)
    # Test: is the scaling better described by piecewise than smooth?
    has_phase_transition = any(
        pt.get("prefers_piecewise", False)
        for pt in results["phase_transition"].values()
        if isinstance(pt, dict)
    )

    # Also test: does d^2 scaling (quadratic) fit better than d^1 (linear)?
    # N_max should scale as d^2/r^2, not d/r
    nmax_gamma = results["scaling_laws"].get("nmax_vs_d", {}).get("gamma", None)

    k2_result = {
        "phase_transition_detected": has_phase_transition,
        "nmax_exponent": nmax_gamma,
        "nmax_is_superlinear": nmax_gamma is not None and nmax_gamma > 1.5,
        "cos_exponent": beta_cos,
        "cos_steeper_than_linear": beta_cos < -1.0,
    }

    # K2 KILLS if scaling is purely linear (gamma ~ 1.0) with no phase transition
    k2_kills = (
        not has_phase_transition and
        (nmax_gamma is None or abs(nmax_gamma - 1.0) < 0.3)
    )
    k2_result["kills"] = k2_kills

    print(f"\nK2: Is scaling linear (no sweet spot)?")
    print(f"  Phase transition detected: {has_phase_transition}")
    print(f"  N_max exponent gamma: {nmax_gamma}")
    print(f"  Cosine exponent beta: {beta_cos:.3f}")
    print(f"  K2 kills: {k2_kills}")

    results["kill_criteria"]["K2"] = k2_result

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'d':>6s} | {'Model':>17s} | {'mean|cos|':>10s} | {'max|cos|':>10s} | "
          f"{'SR':>7s} | {'ERR':>7s} | {'<tau':>5s} | {'N_max':>6s} | {'N_theory':>9s}")
    print("-" * 95)

    for d in dimensions:
        pd = results["per_dimension"][str(d)]
        nm = results["n_max"][str(d)]
        model_name = D_TO_MODEL.get(d, f"d={d}")
        print(f"{d:6d} | {model_name:>17s} | {pd['mean_cos']['mean']:10.6f} | "
              f"{pd['max_cos']['mean']:10.6f} | {pd['signal_retention']['mean']:7.4f} | "
              f"{pd['effective_rank_ratio']['mean']:7.4f} | "
              f"{'YES' if pd['below_tau'] else 'NO':>5s} | "
              f"{nm['n_max_mean']:6.0f} | {nm['n_max_theory']:9.0f}")

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    results["timing"]["total_seconds"] = elapsed
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
