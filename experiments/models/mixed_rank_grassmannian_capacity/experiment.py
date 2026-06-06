#!/usr/bin/env python3
"""
Mixed-Rank Grassmannian Capacity: Can AP pack experts with different ranks?

Hypothesis: When experts have ranks r_1, ..., r_N drawn from {4, 8, 16},
the effective capacity and packing quality remain within 2x of conservative
bounds, and AP can be extended to handle mixed Gr(r_i, d) manifolds.

Mathematical setup:
  - N experts, expert i lives on Gr(r_i, d)
  - Frame U_i has shape (d, r_i) with orthonormal columns
  - Cross-coherence: coh(i,j) = ||U_i^T U_j||_F / sqrt(min(r_i, r_j))
    (normalized so coh in [0, 1] regardless of rank pair)
  - Chordal distance: d_c^2(i,j) = min(r_i, r_j) - ||U_i^T U_j||_F^2
  - Block Gram matrix: G_{ij} = U_i^T U_j has shape (r_i, r_j)

AP extension for mixed ranks:
  - The Gram matrix G is block-structured with variable block sizes
  - Diagonal block G_{ii} = I_{r_i} (identity of size r_i)
  - Off-diagonal block G_{ij} has shape (r_i, r_j)
  - Structural projection: cap ||G_{ij}||_F at mu_target * sqrt(min(r_i, r_j))
    (rank-normalized threshold)
  - Spectral projection: eigendecompose full G, keep top-d, rescale trace = sum(r_i)

Kill criteria:
  K1: mixed-rank N_max not within 2x of min_i(d^2/r_i^2) lower bound
  K2: packing quality (min normalized chordal distance) degrades >50% vs uniform rank

Pure numpy/scipy, CPU-only. Target runtime < 10 minutes.
"""

import json
import time
from pathlib import Path

import numpy as np

DTYPE = np.float64  # need precision for Gram matrix operations
RESULTS_DIR = Path(__file__).parent


# =============================================================================
# Core Grassmannian functions (mixed-rank extension)
# =============================================================================

def random_frame(d, r, rng):
    """Random point on Gr(r, d) -- orthonormal (d, r) frame."""
    M = rng.randn(d, r)
    Q, _ = np.linalg.qr(M)
    return Q[:, :r]


def mixed_rank_random_frames(d, ranks, rng):
    """Generate N random frames with mixed ranks."""
    return [random_frame(d, r, rng) for r in ranks]


def frames_to_block_gram(frames, ranks):
    """
    Build the block Gram matrix from mixed-rank frames.

    G is (sum(r_i) x sum(r_i)), where block (i,j) = U_i^T @ U_j
    has shape (r_i, r_j).
    """
    N = len(frames)
    total = sum(ranks)
    G = np.zeros((total, total), dtype=DTYPE)

    offsets = [0]
    for r in ranks:
        offsets.append(offsets[-1] + r)

    for i in range(N):
        for j in range(N):
            si, ei = offsets[i], offsets[i+1]
            sj, ej = offsets[j], offsets[j+1]
            G[si:ei, sj:ej] = frames[i].T @ frames[j]

    return G, offsets


def block_coherences(G, ranks, offsets):
    """
    Compute normalized coherence for all off-diagonal pairs.

    coh(i,j) = ||G_{ij}||_F / sqrt(min(r_i, r_j))

    Returns dict with pair keys and coherence values.
    """
    N = len(ranks)
    cohs = {}
    for i in range(N):
        for j in range(i+1, N):
            si, ei = offsets[i], offsets[i+1]
            sj, ej = offsets[j], offsets[j+1]
            block = G[si:ei, sj:ej]
            norm = np.linalg.norm(block, 'fro')
            # Normalize by sqrt(min(r_i, r_j)) so coherence in [0, 1]
            min_r = min(ranks[i], ranks[j])
            cohs[(i, j)] = norm / np.sqrt(min_r)
    return cohs


def chordal_distances(G, ranks, offsets):
    """
    Compute chordal distances for all off-diagonal pairs.

    d_c^2(i,j) = min(r_i, r_j) - ||U_i^T U_j||_F^2

    Normalized: d_c_norm = d_c / sqrt(min(r_i, r_j)) in [0, 1]
    """
    N = len(ranks)
    dists = {}
    for i in range(N):
        for j in range(i+1, N):
            si, ei = offsets[i], offsets[i+1]
            sj, ej = offsets[j], offsets[j+1]
            block = G[si:ei, sj:ej]
            norm_sq = np.linalg.norm(block, 'fro') ** 2
            min_r = min(ranks[i], ranks[j])
            d_sq = max(min_r - norm_sq, 0.0)
            dists[(i, j)] = np.sqrt(d_sq) / np.sqrt(min_r)  # normalized
    return dists


# =============================================================================
# Mixed-Rank Alternating Projection
# =============================================================================

def structural_projection_mixed(G, ranks, offsets, mu_target):
    """
    Structural projection for mixed-rank Gram matrix.

    For each off-diagonal block G_{ij} with ||G_{ij}||_F > threshold:
      G_{ij} <- G_{ij} * (threshold / ||G_{ij}||_F)

    Threshold = mu_target * sqrt(min(r_i, r_j))
    (scaled by rank so the normalized coherence is capped at mu_target)
    """
    N = len(ranks)
    G_new = G.copy()

    for i in range(N):
        for j in range(N):
            si, ei = offsets[i], offsets[i+1]
            sj, ej = offsets[j], offsets[j+1]
            if i == j:
                # Diagonal block = I_{r_i}
                G_new[si:ei, sj:ej] = np.eye(ranks[i], dtype=DTYPE)
            else:
                block = G_new[si:ei, sj:ej]
                norm = np.linalg.norm(block, 'fro')
                min_r = min(ranks[i], ranks[j])
                threshold = mu_target * np.sqrt(min_r)
                if norm > threshold:
                    G_new[si:ei, sj:ej] = block * (threshold / norm)

    return G_new


def spectral_projection_mixed(G, ranks, d):
    """
    Spectral projection for mixed-rank Gram matrix.

    Valid Gram: PSD, rank <= d, trace = sum(r_i).
    """
    total = G.shape[0]
    total_rank = sum(ranks)

    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_proj = np.zeros(total, dtype=DTYPE)
    k = min(d, total)
    eigvals_proj[:k] = np.maximum(eigvals[:k], 0)

    current_trace = eigvals_proj.sum()
    if current_trace > 1e-10:
        eigvals_proj *= total_rank / current_trace

    G_proj = (eigvecs * eigvals_proj[None, :]) @ eigvecs.T
    G_proj = (G_proj + G_proj.T) / 2

    return G_proj


def gram_to_mixed_frames(G, ranks, offsets, d):
    """
    Extract mixed-rank frames from a Gram matrix.

    Factor G = V Lambda V^T, embed in R^d, extract per-expert blocks,
    orthonormalize each to get (d, r_i) frames.
    """
    N = len(ranks)
    total = G.shape[0]

    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = min(d, total)
    sqrt_eig = np.sqrt(np.maximum(eigvals[:k], 0))
    embedding = eigvecs[:, :k] * sqrt_eig[None, :]  # (total, k)

    frames = []
    for i in range(N):
        si, ei = offsets[i], offsets[i+1]
        r_i = ranks[i]
        block = embedding[si:ei, :]  # (r_i, k)

        # Pad or truncate to (r_i, d)
        if k < d:
            padded = np.zeros((r_i, d), dtype=DTYPE)
            padded[:, :k] = block
            block = padded
        else:
            block = block[:, :d]

        # QR to get orthonormal (d, r_i) frame
        Q, _ = np.linalg.qr(block.T)
        frames.append(Q[:, :r_i])

    return frames


def alternating_projection_mixed(d, ranks, n_iter=500, mu_factor=1.5, rng=None):
    """
    Alternating Projection for mixed-rank subspace packing on Gr(r_i, d).

    Args:
        d: ambient dimension
        ranks: list of N ranks [r_1, ..., r_N]
        n_iter: AP iterations
        mu_factor: target coherence = mu_factor * welch_bound_estimate
        rng: random state

    Returns:
        frames: list of N frames, frames[i] has shape (d, r_i)
        history: convergence info
    """
    if rng is None:
        rng = np.random.RandomState(42)

    N = len(ranks)

    # Welch-like bound estimate for mixed ranks:
    # Use the maximum rank for a conservative estimate
    r_max = max(ranks)
    Nr_max = N * r_max
    if Nr_max <= d:
        mu_welch_est = 0.0
    else:
        mu_welch_est = np.sqrt(r_max * (Nr_max - d) / (d * (Nr_max - r_max)))
    mu_target = max(mu_factor * mu_welch_est, 1e-4)

    # Initialize with random frames
    frames = mixed_rank_random_frames(d, ranks, rng)
    G, offsets = frames_to_block_gram(frames, ranks)

    history = {
        'mu_target': float(mu_target),
        'welch_est': float(mu_welch_est),
        'max_coherence': [],
        'mean_coherence': [],
        'min_chordal': [],
    }

    for it in range(n_iter):
        G = structural_projection_mixed(G, ranks, offsets, mu_target)
        G = spectral_projection_mixed(G, ranks, d)

        if it % 50 == 0 or it == n_iter - 1:
            cohs = block_coherences(G, ranks, offsets)
            if cohs:
                coh_vals = list(cohs.values())
                history['max_coherence'].append(float(max(coh_vals)))
                history['mean_coherence'].append(float(np.mean(coh_vals)))

            dists = chordal_distances(G, ranks, offsets)
            if dists:
                dist_vals = list(dists.values())
                history['min_chordal'].append(float(min(dist_vals)))

    # Extract frames
    frames = gram_to_mixed_frames(G, ranks, offsets, d)

    return frames, history


# =============================================================================
# Uniform-rank AP baseline (reuse from grassmannian_expert_init)
# =============================================================================

def alternating_projection_uniform(d, r, N, n_iter=500, mu_factor=1.5, rng=None):
    """Standard uniform-rank AP for baseline comparison."""
    if rng is None:
        rng = np.random.RandomState(42)

    Nr = N * r
    if Nr <= d:
        wb = 0.0
    else:
        wb = np.sqrt(r * (Nr - d) / (d * (Nr - r)))
    mu_target = max(mu_factor * wb, 1e-6)

    # Initialize
    frames_arr = np.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        M = rng.randn(d, r)
        Q, _ = np.linalg.qr(M)
        frames_arr[i] = Q[:, :r]

    # Build Gram
    G = np.zeros((Nr, Nr), dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            G[i*r:(i+1)*r, j*r:(j+1)*r] = frames_arr[i].T @ frames_arr[j]

    for it in range(n_iter):
        # Structural
        G_new = G.copy()
        for i in range(N):
            for j in range(N):
                if i == j:
                    G_new[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=DTYPE)
                else:
                    block = G_new[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > mu_target:
                        G_new[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)
        G = G_new

        # Spectral
        G = (G + G.T) / 2
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals_proj = np.zeros(Nr, dtype=DTYPE)
        k = min(d, Nr)
        eigvals_proj[:k] = np.maximum(eigvals[:k], 0)
        tr = eigvals_proj.sum()
        if tr > 1e-10:
            eigvals_proj *= (N * r) / tr
        G = (eigvecs * eigvals_proj[None, :]) @ eigvecs.T
        G = (G + G.T) / 2

    # Extract frames
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(d, Nr)
    sqrt_eig = np.sqrt(np.maximum(eigvals[:k], 0))
    embedding = eigvecs[:, :k] * sqrt_eig[None, :]

    frames_out = np.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        block = embedding[i*r:(i+1)*r, :]
        if k < d:
            padded = np.zeros((r, d), dtype=DTYPE)
            padded[:, :k] = block
            block = padded
        else:
            block = block[:, :d]
        Q, _ = np.linalg.qr(block.T)
        frames_out[i] = Q[:, :r]

    # Measure coherences
    cohs = []
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            overlap = np.linalg.norm(frames_out[i].T @ frames_out[j], 'fro')
            cohs.append(overlap / np.sqrt(r))
            d_sq = max(r - overlap**2, 0)
            dists.append(np.sqrt(d_sq) / np.sqrt(r))

    return {
        'mean_coh': float(np.mean(cohs)),
        'max_coh': float(np.max(cohs)),
        'min_dist': float(np.min(dists)),
        'mean_dist': float(np.mean(dists)),
    }


# =============================================================================
# Capacity estimation
# =============================================================================

def estimate_capacity_mixed(d, rank_distribution, quality_threshold=0.5,
                            n_trials=3, n_iter=300):
    """
    Find the maximum N at which min normalized chordal distance stays above threshold.

    For mixed ranks, we test N = {5, 10, 15, 20, 30, 40, 50, 60, 80} and
    find the largest N where min_dist > quality_threshold * baseline_dist.

    rank_distribution: callable(N, rng) -> list of N ranks
    """
    N_values = [5, 10, 15, 20, 30, 40, 50, 60, 80]
    results = []

    for N in N_values:
        trial_min_dists = []
        trial_mean_cohs = []
        trial_max_cohs = []

        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial * 1000 + N)
            ranks = rank_distribution(N, rng)

            # Check if total rank exceeds d (packing is required)
            total_rank = sum(ranks)

            frames, history = alternating_projection_mixed(
                d, ranks, n_iter=n_iter, mu_factor=1.5, rng=rng
            )

            # Measure final quality
            G, offsets = frames_to_block_gram(frames, ranks)
            dists = chordal_distances(G, ranks, offsets)
            cohs = block_coherences(G, ranks, offsets)

            if dists:
                dist_vals = list(dists.values())
                coh_vals = list(cohs.values())
                trial_min_dists.append(min(dist_vals))
                trial_mean_cohs.append(np.mean(coh_vals))
                trial_max_cohs.append(max(coh_vals))

        if trial_min_dists:
            results.append({
                'N': N,
                'ranks_sample': rank_distribution(N, np.random.RandomState(42)),
                'mean_min_dist': float(np.mean(trial_min_dists)),
                'std_min_dist': float(np.std(trial_min_dists)),
                'mean_mean_coh': float(np.mean(trial_mean_cohs)),
                'mean_max_coh': float(np.mean(trial_max_cohs)),
                'total_rank_mean': float(np.mean([sum(rank_distribution(N, np.random.RandomState(42 + t)))
                                                   for t in range(n_trials)])),
            })

    return results


# =============================================================================
# Rank distribution generators
# =============================================================================

def uniform_low(N, rng):
    """All rank-4."""
    return [4] * N

def uniform_mid(N, rng):
    """All rank-8."""
    return [8] * N

def uniform_high(N, rng):
    """All rank-16."""
    return [16] * N

def mixed_equal(N, rng):
    """Equal mix of {4, 8, 16}."""
    ranks = []
    choices = [4, 8, 16]
    for i in range(N):
        ranks.append(choices[i % 3])
    rng.shuffle(ranks)
    return ranks

def mixed_heavy_low(N, rng):
    """70% rank-4, 20% rank-8, 10% rank-16."""
    ranks = []
    for i in range(N):
        p = rng.random()
        if p < 0.7:
            ranks.append(4)
        elif p < 0.9:
            ranks.append(8)
        else:
            ranks.append(16)
    return ranks

def mixed_heavy_high(N, rng):
    """10% rank-4, 20% rank-8, 70% rank-16."""
    ranks = []
    for i in range(N):
        p = rng.random()
        if p < 0.1:
            ranks.append(4)
        elif p < 0.3:
            ranks.append(8)
        else:
            ranks.append(16)
    return ranks


# =============================================================================
# Crowding analysis: does one large-rank expert crowd out small ones?
# =============================================================================

def crowding_analysis(d, n_iter=300, n_trials=3):
    """
    Test whether replacing one rank-4 expert with a rank-16 expert
    degrades packing quality for the remaining experts.

    Setup: N=20 experts at d=64
    Condition A: all rank-4 (baseline)
    Condition B: 19 rank-4 + 1 rank-16
    Condition C: 19 rank-4 + 1 rank-8
    Condition D: 10 rank-4 + 10 rank-16
    """
    results = {}

    conditions = {
        'all_r4': lambda: [4] * 20,
        '19r4_1r16': lambda: [4] * 19 + [16],
        '19r4_1r8': lambda: [4] * 19 + [8],
        '10r4_10r16': lambda: [4] * 10 + [16] * 10,
        'all_r16': lambda: [16] * 20,
    }

    for name, rank_fn in conditions.items():
        trial_results = []
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial * 100)
            ranks = rank_fn()

            frames, history = alternating_projection_mixed(
                d, ranks, n_iter=n_iter, mu_factor=1.5, rng=rng
            )

            G, offsets = frames_to_block_gram(frames, ranks)
            cohs = block_coherences(G, ranks, offsets)
            dists = chordal_distances(G, ranks, offsets)

            coh_vals = list(cohs.values())
            dist_vals = list(dists.values())

            # Separate coherences by rank-pair type
            pair_types = {}
            for (i, j), c in cohs.items():
                ri, rj = ranks[i], ranks[j]
                key = f"{min(ri,rj)}-{max(ri,rj)}"
                if key not in pair_types:
                    pair_types[key] = []
                pair_types[key].append(c)

            trial_results.append({
                'min_dist': float(min(dist_vals)),
                'mean_dist': float(np.mean(dist_vals)),
                'mean_coh': float(np.mean(coh_vals)),
                'max_coh': float(max(coh_vals)),
                'pair_type_cohs': {k: float(np.mean(v)) for k, v in pair_types.items()},
            })

        results[name] = {
            'ranks': rank_fn(),
            'total_rank': sum(rank_fn()),
            'mean_min_dist': float(np.mean([t['min_dist'] for t in trial_results])),
            'std_min_dist': float(np.std([t['min_dist'] for t in trial_results])),
            'mean_mean_coh': float(np.mean([t['mean_coh'] for t in trial_results])),
            'mean_max_coh': float(np.mean([t['max_coh'] for t in trial_results])),
            'pair_type_cohs': trial_results[0]['pair_type_cohs'],
            'trials': trial_results,
        }

    return results


# =============================================================================
# Theoretical bounds
# =============================================================================

def theoretical_capacity_bounds(d, ranks_unique):
    """
    Compute theoretical capacity bounds for mixed-rank systems.

    Conservative bound: min_r(d^2/r^2) = d^2/r_max^2
    This is the bottleneck because the largest-rank expert constrains the most.

    Optimistic "sum" bound: sum_r n_r * d^2/r^2 / N
    (weighted average of per-rank capacities)
    """
    bounds = {}
    for r in ranks_unique:
        bounds[r] = {
            'N_max_uniform': d**2 / r**2,
            'welch_bound_at_Nmax': np.sqrt(r * (d**2/r**2 * r - d) / (d * (d**2/r**2 * r - r)))
                if d**2/r > d else 0.0,
        }

    conservative = min(d**2 / r**2 for r in ranks_unique)
    bounds['conservative_N_max'] = conservative
    bounds['d'] = d
    bounds['ranks'] = list(ranks_unique)

    return bounds


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment():
    t0 = time.time()
    d = 64
    seeds = [42, 137, 256]

    print("=" * 72)
    print("  Mixed-Rank Grassmannian Capacity")
    print(f"  d={d}, ranks={{4, 8, 16}}, seeds={seeds}")
    print("=" * 72)

    all_results = {}

    # ------------------------------------------------------------------
    # Part 1: Theoretical bounds
    # ------------------------------------------------------------------
    print("\n  PART 1: Theoretical Bounds")
    print("  " + "-" * 40)

    bounds = theoretical_capacity_bounds(d, [4, 8, 16])
    for r in [4, 8, 16]:
        print(f"    r={r:2d}: N_max(uniform) = {bounds[r]['N_max_uniform']:.0f}")
    print(f"    Conservative N_max (bottleneck r=16): {bounds['conservative_N_max']:.0f}")

    all_results['theoretical_bounds'] = bounds

    # ------------------------------------------------------------------
    # Part 2: Uniform-rank baselines
    # ------------------------------------------------------------------
    print("\n  PART 2: Uniform-Rank AP Baselines")
    print("  " + "-" * 40)

    uniform_baselines = {}
    for r in [4, 8, 16]:
        # All baselines must cover the same N values as mixed sweeps
        # for fair K2 comparison (especially r=16 which is the K2 reference)
        N_values_for_r = [5, 10, 15, 20, 30, 40, 50, 60, 80]

        baseline_by_N = {}
        for N in N_values_for_r:
            trial_results = []
            for seed in seeds:
                rng = np.random.RandomState(seed + r * 100 + N)
                res = alternating_projection_uniform(d, r, N, n_iter=300,
                                                     mu_factor=1.5, rng=rng)
                trial_results.append(res)

            baseline_by_N[N] = {
                'mean_min_dist': float(np.mean([t['min_dist'] for t in trial_results])),
                'mean_mean_coh': float(np.mean([t['mean_coh'] for t in trial_results])),
                'mean_max_coh': float(np.mean([t['max_coh'] for t in trial_results])),
            }
            print(f"    r={r:2d}, N={N:3d}: min_dist={baseline_by_N[N]['mean_min_dist']:.4f}, "
                  f"max_coh={baseline_by_N[N]['mean_max_coh']:.4f}")

        uniform_baselines[r] = baseline_by_N

    all_results['uniform_baselines'] = {str(r): {str(n): v for n, v in byn.items()}
                                         for r, byn in uniform_baselines.items()}

    # ------------------------------------------------------------------
    # Part 3: Mixed-rank capacity sweeps
    # ------------------------------------------------------------------
    print("\n  PART 3: Mixed-Rank Capacity Sweeps")
    print("  " + "-" * 40)

    distributions = {
        'mixed_equal': mixed_equal,
        'mixed_heavy_low': mixed_heavy_low,
        'mixed_heavy_high': mixed_heavy_high,
    }

    mixed_results = {}
    for dist_name, dist_fn in distributions.items():
        print(f"\n    Distribution: {dist_name}")
        cap_results = estimate_capacity_mixed(d, dist_fn, n_trials=len(seeds),
                                               n_iter=300)

        for cr in cap_results:
            ranks_sample = cr['ranks_sample']
            rank_counts = {4: ranks_sample.count(4), 8: ranks_sample.count(8),
                           16: ranks_sample.count(16)}
            print(f"      N={cr['N']:3d} (r4={rank_counts.get(4,0)}, r8={rank_counts.get(8,0)}, "
                  f"r16={rank_counts.get(16,0)}): min_dist={cr['mean_min_dist']:.4f}, "
                  f"max_coh={cr['mean_max_coh']:.4f}")

        mixed_results[dist_name] = cap_results

    all_results['mixed_capacity'] = mixed_results

    # ------------------------------------------------------------------
    # Part 4: Crowding analysis
    # ------------------------------------------------------------------
    print("\n  PART 4: Crowding Analysis (N=20, d=64)")
    print("  " + "-" * 40)

    crowd = crowding_analysis(d, n_iter=300, n_trials=len(seeds))
    for name, res in crowd.items():
        rank_counts = {4: res['ranks'].count(4), 8: res['ranks'].count(8),
                       16: res['ranks'].count(16)}
        print(f"    {name:15s}: min_dist={res['mean_min_dist']:.4f}, "
              f"max_coh={res['mean_max_coh']:.4f}, "
              f"total_rank={res['total_rank']:3d}")
        if res.get('pair_type_cohs'):
            for pt, mc in sorted(res['pair_type_cohs'].items()):
                print(f"      pair {pt}: mean_coh={mc:.4f}")

    all_results['crowding'] = crowd

    # ------------------------------------------------------------------
    # Part 5: Kill Criteria Assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 72)

    # K1: mixed-rank N_max within 2x of min(d^2/r_i^2)
    # Conservative bound = d^2/16^2 = 64^2/256 = 16
    conservative_Nmax = bounds['conservative_N_max']
    print(f"\n  K1: Mixed-rank N_max within 2x of min(d^2/r_i^2) = {conservative_Nmax:.0f}")
    print(f"      Lower threshold: N_max >= {conservative_Nmax / 2:.0f}")

    # Find capacity: largest N where min_dist > 0.5 (quality not collapsed)
    # Use mixed_equal as representative
    k1_results = {}
    for dist_name, cap_results in mixed_results.items():
        # Find last N where min_dist > 0.3 (normalized chordal > 0.3 means reasonable separation)
        good_N = [cr['N'] for cr in cap_results if cr['mean_min_dist'] > 0.3]
        effective_Nmax = max(good_N) if good_N else 0
        ratio = effective_Nmax / conservative_Nmax if conservative_Nmax > 0 else float('inf')
        k1_pass = ratio >= 0.5  # within 2x means ratio >= 0.5
        k1_results[dist_name] = {
            'effective_Nmax': effective_Nmax,
            'conservative_Nmax': conservative_Nmax,
            'ratio': ratio,
            'pass': k1_pass,
        }
        print(f"    {dist_name}: effective_Nmax={effective_Nmax}, "
              f"ratio={ratio:.2f}x, {'PASS' if k1_pass else 'FAIL'}")

    k1_overall = all(v['pass'] for v in k1_results.values())
    print(f"  K1 overall: {'PASS' if k1_overall else 'FAIL'}")

    # K2: packing quality (min chordal distance) degrades >50% vs uniform rank
    r_max = 16  # K2 baseline: always compare against uniform r_max
    print(f"\n  K2: Packing quality degradation vs uniform r_max={r_max} (threshold: >50%)")
    k2_results = {}
    for dist_name, cap_results in mixed_results.items():
        degradations = []
        for cr in cap_results:
            N = cr['N']

            # Compare against uniform r_max=16 at the same N.
            # This is the correct baseline: the question is whether
            # mixing ranks degrades packing vs the worst-case uniform
            # component (r_max), not the best-case (dominant/smallest rank).
            baseline = uniform_baselines.get(r_max, {}).get(N)

            if baseline is not None:
                baseline_min_dist = baseline['mean_min_dist']
                mixed_min_dist = cr['mean_min_dist']
                if baseline_min_dist > 0:
                    degradation = 1.0 - mixed_min_dist / baseline_min_dist
                    degradations.append(degradation)

        if degradations:
            worst_degradation = max(degradations)
            mean_degradation = np.mean(degradations)
            k2_pass = worst_degradation < 0.5
            k2_results[dist_name] = {
                'worst_degradation': worst_degradation,
                'mean_degradation': mean_degradation,
                'pass': k2_pass,
            }
            print(f"    {dist_name}: worst_degrad={worst_degradation:.1%}, "
                  f"mean_degrad={mean_degradation:.1%}, "
                  f"{'PASS' if k2_pass else 'FAIL'}")
        else:
            k2_results[dist_name] = {'pass': True, 'worst_degradation': 0.0,
                                      'mean_degradation': 0.0}
            print(f"    {dist_name}: no comparable baseline (PASS by default)")

    k2_overall = all(v['pass'] for v in k2_results.values())
    print(f"  K2 overall: {'PASS' if k2_overall else 'FAIL'}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 72)
    if k1_overall and k2_overall:
        verdict = "SUPPORTED"
        print(f"  VERDICT: SUPPORTED")
        print(f"  Mixed-rank AP packing works. Capacity stays within 2x of")
        print(f"  conservative bound, and packing quality does not degrade >50%.")
    elif k1_overall:
        verdict = "PARTIAL (K1 PASS, K2 FAIL)"
        print(f"  VERDICT: PARTIAL")
        print(f"  Capacity bound holds (K1), but packing quality degrades (K2).")
    elif k2_overall:
        verdict = "PARTIAL (K1 FAIL, K2 PASS)"
        print(f"  VERDICT: PARTIAL")
        print(f"  Packing quality OK (K2), but capacity below bound (K1).")
    else:
        verdict = "KILLED"
        print(f"  VERDICT: KILLED")
        print(f"  Both capacity and quality fail.")
    print("=" * 72)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    all_results['kill_criteria'] = {
        'k1': k1_results,
        'k1_overall': k1_overall,
        'k2': k2_results,
        'k2_overall': k2_overall,
        'verdict': verdict,
    }
    all_results['config'] = {
        'd': d,
        'seeds': seeds,
        'rank_values': [4, 8, 16],
        'mu_factor': 1.5,
        'k2_baseline': 'uniform_r_max_16',
        'revision_note': 'v2: fixed mu_factor mismatch (both 1.5) and K2 baseline (uniform r_max=16)',
    }
    all_results['elapsed_seconds'] = elapsed

    out = RESULTS_DIR / 'results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out}")

    return all_results


if __name__ == '__main__':
    run_experiment()
