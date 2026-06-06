#!/usr/bin/env python3
"""
TAAP Convergence: Does Truncated/Accelerated AP close the gap to Welch bound?

Hypothesis: Standard AP converges to 2.8-3x above the Welch bound after 500
iterations. TAAP variants (adaptive mu scheduling, momentum, selective
truncation) may converge closer to the bound.

Critical context from parent experiments:
  - grassmannian_expert_init: AP gives coherence 2.8-3x above Welch bound
  - minimax_grassmannian_packing: KILLED. AP already achieves perfect
    equidistribution (max/mean=1.00x). Stochastic refinement found 0%.
    The gap may be fundamental at small N.

Kill criteria:
  K1: TAAP coherence not closer to Welch bound than standard AP
  K2: TAAP runtime >3x standard AP for same N, d

Test dimensions: d=64 (N=12), d=128 (N=20), d=256 (N=40), rank r=8.

Five methods compared:
  1. Standard AP (500 iters, mu_factor=1.2) -- baseline from parent
  2. Standard AP (2000 iters) -- more iterations baseline
  3. TAAP-Schedule: adaptive mu schedule (start loose, tighten to Welch+epsilon)
  4. TAAP-Momentum: Nesterov-accelerated AP with momentum on Gram matrix
  5. TAAP-Selective: only truncate top-p% worst blocks per iteration

Also tests whether increasing iterations alone closes the gap, to distinguish
algorithmic improvement from insufficient convergence.

Pure numpy/scipy, CPU-only. No GPU.
"""

import json
import time
from pathlib import Path

import numpy as np

DTYPE = np.float32
RESULTS_DIR = Path(__file__).parent


# =============================================================================
# Welch bound and coherence metrics
# =============================================================================

def welch_bound(N, r, d):
    """Welch bound for N subspaces of dimension r in R^d."""
    Nr = N * r
    if Nr <= d:
        return 0.0
    return np.sqrt(r * (Nr - d) / (d * (Nr - r)))


def compute_coherence(frames):
    """
    Compute max and mean pairwise coherence (Frobenius norm of U_i^T @ U_j).
    Returns: (max_coherence, mean_coherence, all_coherences)
    """
    N = frames.shape[0]
    coherences = []
    for i in range(N):
        for j in range(i + 1, N):
            c = np.linalg.norm(frames[i].T @ frames[j], 'fro')
            coherences.append(float(c))
    coherences = np.array(coherences)
    return float(coherences.max()), float(coherences.mean()), coherences


def random_grassmannian_points(N, r, d, rng):
    """Generate N random orthonormal frames on Gr(r, d)."""
    frames = np.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        M = rng.randn(d, r).astype(DTYPE)
        Q, _ = np.linalg.qr(M)
        frames[i] = Q[:, :r]
    return frames


# =============================================================================
# Gram matrix operations (shared by all methods)
# =============================================================================

def frames_to_gram(frames):
    """Compute (Nr x Nr) block Gram matrix."""
    N, d, r = frames.shape
    Nr = N * r
    G = np.zeros((Nr, Nr), dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            G[i*r:(i+1)*r, j*r:(j+1)*r] = frames[i].T @ frames[j]
    return G


def block_norms(G, N, r):
    """Frobenius norms of off-diagonal blocks."""
    norms = np.zeros((N, N), dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            if i != j:
                block = G[i*r:(i+1)*r, j*r:(j+1)*r]
                norms[i, j] = np.linalg.norm(block, 'fro')
    return norms


def spectral_projection(G, N, r, d):
    """Project G to nearest valid Gram matrix (PSD, rank-d, trace=Nr)."""
    Nr = N * r
    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals_proj = np.zeros(Nr, dtype=DTYPE)
    eigvals_proj[:min(d, Nr)] = np.maximum(eigvals[:min(d, Nr)], 0)
    current_trace = eigvals_proj.sum()
    if current_trace > 1e-10:
        eigvals_proj *= (N * r) / current_trace
    G_proj = (eigvecs * eigvals_proj[None, :]) @ eigvecs.T
    return (G_proj + G_proj.T) / 2


def gram_to_frames(G, N, r, d):
    """Extract N orthonormal frames from Gram matrix."""
    Nr = N * r
    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(d, Nr)
    sqrt_eig = np.sqrt(np.maximum(eigvals[:k], 0)).astype(DTYPE)
    embedding = (eigvecs[:, :k] * sqrt_eig[None, :]).astype(DTYPE)
    frames = np.zeros((N, d, r), dtype=DTYPE)
    for i in range(N):
        block = embedding[i*r:(i+1)*r, :]
        if k < d:
            padded = np.zeros((r, d), dtype=DTYPE)
            padded[:, :k] = block
            block = padded
        else:
            block = block[:, :d]
        Q, _ = np.linalg.qr(block.T)
        frames[i] = Q[:, :r]
    return frames


# =============================================================================
# Method 1 & 2: Standard AP (baseline, reused from parent)
# =============================================================================

def standard_ap(N, r, d, n_iter, mu_factor=1.2, rng=None):
    """Standard Alternating Projection (Dhillon et al. 2008)."""
    if rng is None:
        rng = np.random.RandomState(42)

    wb = welch_bound(N, r, d)
    mu_target = max(mu_factor * wb, 1e-6)

    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)

    history = {'max_coh': [], 'mean_coh': [], 'iter': []}

    for it in range(n_iter):
        # Structural projection: cap all off-diagonal blocks at mu_target
        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=DTYPE)
                else:
                    block = G[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > mu_target:
                        G[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)

        # Spectral projection
        G = spectral_projection(G, N, r, d)

        # Track every 50 iterations
        if it % 50 == 0 or it == n_iter - 1:
            norms = block_norms(G, N, r)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            history['max_coh'].append(float(norms.max()))
            history['mean_coh'].append(float(norms[mask].mean()))
            history['iter'].append(it)

    frames = gram_to_frames(G, N, r, d)
    return frames, history


# =============================================================================
# Method 3: TAAP-Schedule (adaptive mu tightening)
# =============================================================================

def taap_schedule(N, r, d, n_iter, rng=None):
    """
    TAAP with adaptive mu schedule.

    Start with mu = 2 * Welch bound (loose), linearly tighten to
    1.01 * Welch bound over the iterations. This allows the algorithm
    to find a good global arrangement first, then refine.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    wb = welch_bound(N, r, d)
    mu_start = max(2.0 * wb, 1e-6)
    mu_end = max(1.01 * wb, 1e-6)

    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)

    history = {'max_coh': [], 'mean_coh': [], 'iter': [], 'mu': []}

    for it in range(n_iter):
        # Cosine annealing schedule for mu
        progress = it / max(n_iter - 1, 1)
        mu_target = mu_end + 0.5 * (mu_start - mu_end) * (1 + np.cos(np.pi * progress))

        # Structural projection with current mu
        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=DTYPE)
                else:
                    block = G[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > mu_target:
                        G[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)

        G = spectral_projection(G, N, r, d)

        if it % 50 == 0 or it == n_iter - 1:
            norms = block_norms(G, N, r)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            history['max_coh'].append(float(norms.max()))
            history['mean_coh'].append(float(norms[mask].mean()))
            history['iter'].append(it)
            history['mu'].append(float(mu_target))

    frames = gram_to_frames(G, N, r, d)
    return frames, history


# =============================================================================
# Method 4: TAAP-Momentum (Nesterov acceleration)
# =============================================================================

def taap_momentum(N, r, d, n_iter, beta=0.9, mu_factor=1.2, rng=None):
    """
    TAAP with Nesterov-style momentum on the Gram matrix.

    Instead of G_{k+1} = spectral(structural(G_k)), we apply:
      G_look = G_k + beta * (G_k - G_{k-1})   # momentum look-ahead
      G_{k+1} = spectral(structural(G_look))

    This accelerates convergence through the alternating projection
    fixed-point iteration.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    wb = welch_bound(N, r, d)
    mu_target = max(mu_factor * wb, 1e-6)

    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)
    G_prev = G.copy()

    history = {'max_coh': [], 'mean_coh': [], 'iter': []}

    for it in range(n_iter):
        # Momentum look-ahead (Nesterov)
        if it > 0:
            G_look = G + beta * (G - G_prev)
        else:
            G_look = G.copy()

        G_prev = G.copy()

        # Structural projection
        for i in range(N):
            for j in range(N):
                if i == j:
                    G_look[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=DTYPE)
                else:
                    block = G_look[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > mu_target:
                        G_look[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)

        # Spectral projection
        G = spectral_projection(G_look, N, r, d)

        if it % 50 == 0 or it == n_iter - 1:
            norms = block_norms(G, N, r)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            history['max_coh'].append(float(norms.max()))
            history['mean_coh'].append(float(norms[mask].mean()))
            history['iter'].append(it)

    frames = gram_to_frames(G, N, r, d)
    return frames, history


# =============================================================================
# Method 5: TAAP-Selective (only truncate worst blocks)
# =============================================================================

def taap_selective(N, r, d, n_iter, percentile=90, mu_factor=1.2, rng=None):
    """
    TAAP with selective truncation.

    Instead of capping ALL off-diagonal blocks at mu_target, only cap
    blocks above the p-th percentile of current norms. This preserves
    more of the Gram structure per iteration, allowing gentler convergence.

    The truncation threshold is max(mu_target, percentile_value), so
    we never truncate below mu_target but we skip blocks that are
    already reasonably small.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    wb = welch_bound(N, r, d)
    mu_target = max(mu_factor * wb, 1e-6)

    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)

    history = {'max_coh': [], 'mean_coh': [], 'iter': []}

    for it in range(n_iter):
        # Compute current block norms
        norms = block_norms(G, N, r)
        off_diag = norms[np.triu_indices(N, k=1)]

        # Only truncate blocks above percentile threshold
        threshold = max(np.percentile(off_diag, percentile), mu_target)

        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=DTYPE)
                else:
                    block = G[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > threshold:
                        G[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)

        G = spectral_projection(G, N, r, d)

        if it % 50 == 0 or it == n_iter - 1:
            norms = block_norms(G, N, r)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            history['max_coh'].append(float(norms.max()))
            history['mean_coh'].append(float(norms[mask].mean()))
            history['iter'].append(it)

    frames = gram_to_frames(G, N, r, d)
    return frames, history


# =============================================================================
# Additional diagnostic: Welch bound tightness analysis
# =============================================================================

def welch_bound_tightness(N, r, d):
    """
    Analyze how tight the Welch bound is for given parameters.

    The Welch bound is derived from:
      sum_{i<j} ||U_i^T U_j||_F^2 >= N(N-1)/2 * r^2(Nr-d) / (d(Nr-r)(N-1))

    This is an AVERAGING bound -- it says the mean squared coherence must
    exceed some floor. But it doesn't constrain the distribution. When N
    is small relative to d^2/r^2 (our regime), there may be configurations
    where all coherences are equal but above the Welch bound, because the
    bound is not achievable (no equiangular tight frame exists).

    An ETF (Equiangular Tight Frame) achieves the Welch bound exactly when
    it exists. ETFs require N <= d(d+1)/2 for r=1 (real case). For general
    r, the condition is more restrictive.

    Returns dict with analysis.
    """
    Nr = N * r
    wb = welch_bound(N, r, d)

    # Packing ratio: how full is the Grassmannian?
    packing_ratio = Nr / d

    # For r=1, ETFs exist when N <= d+1 (simplex bound) or specific N,d pairs
    # For general r, the Orthoplex/simplex analogs are more complex
    # Simple check: can we embed N equiangular subspaces?
    # Upper bound on N for equiangular subspaces: d(d+2r-1) / (r(2r-1)) approximately
    # (from Lemmens-Seidel for r=1, generalized)
    max_equiangular = d * (d + 2 * r - 1) / (r * (2 * r - 1))

    return {
        'welch_bound': float(wb),
        'packing_ratio': float(packing_ratio),
        'max_equiangular_estimate': float(max_equiangular),
        'N_over_max_equiangular': float(N / max_equiangular),
        'etf_possibly_exists': N <= max_equiangular,
    }


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment():
    t0_total = time.time()

    D_VALUES = [64, 128, 256]
    N_PER_D = {64: 12, 128: 20, 256: 40}
    R = 8
    SEEDS = [42, 137]
    N_ITER_BASE = 500
    N_ITER_LONG = 2000

    print("=" * 72)
    print("  TAAP Convergence: Does Truncated AP close the gap to Welch bound?")
    print(f"  d={D_VALUES}, r={R}, seeds={SEEDS}")
    print("=" * 72)

    all_results = {}

    for d in D_VALUES:
        N = N_PER_D[d]
        wb = welch_bound(N, R, d)
        tightness = welch_bound_tightness(N, R, d)

        print(f"\n{'='*72}")
        print(f"  d={d}, N={N}, Nr/d={N*R/d:.2f}, Welch bound={wb:.6f}")
        print(f"  Max equiangular N ~ {tightness['max_equiangular_estimate']:.0f}, "
              f"ETF possible: {tightness['etf_possibly_exists']}")
        print(f"{'='*72}")

        d_results = {
            'd': d, 'N': N, 'r': R, 'welch_bound': float(wb),
            'tightness': tightness, 'methods': {}
        }

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")

            # Method 1: Standard AP (500 iters) -- baseline
            rng = np.random.RandomState(seed)
            t0 = time.time()
            frames_std, hist_std = standard_ap(N, R, d, N_ITER_BASE, mu_factor=1.2, rng=rng)
            t_std = time.time() - t0
            max_c_std, mean_c_std, _ = compute_coherence(frames_std)

            # Method 2: Standard AP (2000 iters) -- more convergence
            rng = np.random.RandomState(seed)  # same init
            t0 = time.time()
            frames_long, hist_long = standard_ap(N, R, d, N_ITER_LONG, mu_factor=1.2, rng=rng)
            t_long = time.time() - t0
            max_c_long, mean_c_long, _ = compute_coherence(frames_long)

            # Method 3: TAAP-Schedule (adaptive mu, 500 iters)
            rng = np.random.RandomState(seed)
            t0 = time.time()
            frames_sched, hist_sched = taap_schedule(N, R, d, N_ITER_BASE, rng=rng)
            t_sched = time.time() - t0
            max_c_sched, mean_c_sched, _ = compute_coherence(frames_sched)

            # Method 4: TAAP-Momentum (Nesterov, 500 iters)
            rng = np.random.RandomState(seed)
            t0 = time.time()
            frames_mom, hist_mom = taap_momentum(N, R, d, N_ITER_BASE, beta=0.9,
                                                  mu_factor=1.2, rng=rng)
            t_mom = time.time() - t0
            max_c_mom, mean_c_mom, _ = compute_coherence(frames_mom)

            # Method 5: TAAP-Selective (top-10% truncation, 500 iters)
            rng = np.random.RandomState(seed)
            t0 = time.time()
            frames_sel, hist_sel = taap_selective(N, R, d, N_ITER_BASE,
                                                   percentile=90, mu_factor=1.2, rng=rng)
            t_sel = time.time() - t0
            max_c_sel, mean_c_sel, _ = compute_coherence(frames_sel)

            # Store results per seed
            seed_key = f"seed_{seed}"
            methods = {
                'std_ap_500': {
                    'max_coh': max_c_std, 'mean_coh': mean_c_std,
                    'time_s': t_std, 'n_iter': N_ITER_BASE,
                    'ratio_to_welch_max': max_c_std / wb if wb > 0 else float('inf'),
                    'ratio_to_welch_mean': mean_c_std / wb if wb > 0 else float('inf'),
                    'history_max': hist_std['max_coh'],
                    'history_mean': hist_std['mean_coh'],
                },
                'std_ap_2000': {
                    'max_coh': max_c_long, 'mean_coh': mean_c_long,
                    'time_s': t_long, 'n_iter': N_ITER_LONG,
                    'ratio_to_welch_max': max_c_long / wb if wb > 0 else float('inf'),
                    'ratio_to_welch_mean': mean_c_long / wb if wb > 0 else float('inf'),
                    'history_max': hist_long['max_coh'],
                    'history_mean': hist_long['mean_coh'],
                },
                'taap_schedule': {
                    'max_coh': max_c_sched, 'mean_coh': mean_c_sched,
                    'time_s': t_sched, 'n_iter': N_ITER_BASE,
                    'ratio_to_welch_max': max_c_sched / wb if wb > 0 else float('inf'),
                    'ratio_to_welch_mean': mean_c_sched / wb if wb > 0 else float('inf'),
                    'history_max': hist_sched['max_coh'],
                    'history_mean': hist_sched['mean_coh'],
                },
                'taap_momentum': {
                    'max_coh': max_c_mom, 'mean_coh': mean_c_mom,
                    'time_s': t_mom, 'n_iter': N_ITER_BASE,
                    'ratio_to_welch_max': max_c_mom / wb if wb > 0 else float('inf'),
                    'ratio_to_welch_mean': mean_c_mom / wb if wb > 0 else float('inf'),
                    'history_max': hist_mom['max_coh'],
                    'history_mean': hist_mom['mean_coh'],
                },
                'taap_selective': {
                    'max_coh': max_c_sel, 'mean_coh': mean_c_sel,
                    'time_s': t_sel, 'n_iter': N_ITER_BASE,
                    'ratio_to_welch_max': max_c_sel / wb if wb > 0 else float('inf'),
                    'ratio_to_welch_mean': mean_c_sel / wb if wb > 0 else float('inf'),
                    'history_max': hist_sel['max_coh'],
                    'history_mean': hist_sel['mean_coh'],
                },
            }

            if seed_key not in d_results['methods']:
                d_results['methods'][seed_key] = methods
            else:
                d_results['methods'][seed_key].update(methods)

            # Print comparison
            print(f"\n  Method comparison (d={d}, seed={seed}):")
            print(f"  {'Method':<20} {'Max coh':>10} {'Mean coh':>10} "
                  f"{'Max/WB':>8} {'Mean/WB':>8} {'Time':>8}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            for name, m in methods.items():
                print(f"  {name:<20} {m['max_coh']:10.6f} {m['mean_coh']:10.6f} "
                      f"{m['ratio_to_welch_max']:8.3f}x {m['ratio_to_welch_mean']:8.3f}x "
                      f"{m['time_s']:7.2f}s")

        all_results[d] = d_results

    # =========================================================================
    # Aggregate across seeds
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS (mean over {len(SEEDS)} seeds)")
    print(f"{'='*72}")

    method_names = ['std_ap_500', 'std_ap_2000', 'taap_schedule',
                    'taap_momentum', 'taap_selective']

    aggregate = {}
    for d in D_VALUES:
        N = N_PER_D[d]
        wb = welch_bound(N, R, d)
        agg = {'d': d, 'N': N, 'welch_bound': float(wb), 'methods': {}}

        print(f"\n  d={d}, N={N}, Welch bound={wb:.6f}")
        print(f"  {'Method':<20} {'Avg Max':>10} {'Avg Mean':>10} "
              f"{'Max/WB':>8} {'Mean/WB':>8} {'Avg Time':>8} {'Time/Std':>9}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")

        std_time = None
        for mname in method_names:
            max_vals = []
            mean_vals = []
            times = []
            for seed in SEEDS:
                sk = f"seed_{seed}"
                m = all_results[d]['methods'][sk][mname]
                max_vals.append(m['max_coh'])
                mean_vals.append(m['mean_coh'])
                times.append(m['time_s'])

            avg_max = float(np.mean(max_vals))
            avg_mean = float(np.mean(mean_vals))
            avg_time = float(np.mean(times))
            ratio_max = avg_max / wb if wb > 0 else float('inf')
            ratio_mean = avg_mean / wb if wb > 0 else float('inf')

            if mname == 'std_ap_500':
                std_time = avg_time

            time_ratio = avg_time / std_time if std_time and std_time > 0 else 1.0

            agg['methods'][mname] = {
                'avg_max_coh': avg_max,
                'avg_mean_coh': avg_mean,
                'max_over_welch': ratio_max,
                'mean_over_welch': ratio_mean,
                'avg_time_s': avg_time,
                'time_ratio_vs_std': time_ratio,
            }

            print(f"  {mname:<20} {avg_max:10.6f} {avg_mean:10.6f} "
                  f"{ratio_max:8.3f}x {ratio_mean:8.3f}x "
                  f"{avg_time:7.2f}s {time_ratio:8.2f}x")

        aggregate[d] = agg

    # =========================================================================
    # Convergence analysis: does more iteration help?
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  CONVERGENCE ANALYSIS: 500 vs 2000 iterations")
    print(f"{'='*72}")
    for d in D_VALUES:
        std500 = aggregate[d]['methods']['std_ap_500']
        std2000 = aggregate[d]['methods']['std_ap_2000']
        improvement = (std500['avg_mean_coh'] - std2000['avg_mean_coh']) / std500['avg_mean_coh'] * 100
        print(f"  d={d}: mean_coh 500it={std500['avg_mean_coh']:.6f}, "
              f"2000it={std2000['avg_mean_coh']:.6f}, "
              f"improvement={improvement:.2f}%")

    # =========================================================================
    # Kill criteria assessment
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA ASSESSMENT")
    print(f"{'='*72}")

    # K1: TAAP coherence must be closer to Welch bound than standard AP
    print(f"\n  K1: TAAP mean coherence closer to Welch bound than std AP (500 iter)?")
    k1_pass_any = False
    k1_results = {}
    for d in D_VALUES:
        std_ratio = aggregate[d]['methods']['std_ap_500']['mean_over_welch']
        best_taap_name = None
        best_taap_ratio = float('inf')
        for mname in ['taap_schedule', 'taap_momentum', 'taap_selective']:
            ratio = aggregate[d]['methods'][mname]['mean_over_welch']
            if ratio < best_taap_ratio:
                best_taap_ratio = ratio
                best_taap_name = mname

        improvement = (std_ratio - best_taap_ratio) / std_ratio * 100
        k1_d = best_taap_ratio < std_ratio
        if k1_d:
            k1_pass_any = True

        k1_results[d] = {
            'std_ratio': std_ratio,
            'best_taap': best_taap_name,
            'best_taap_ratio': best_taap_ratio,
            'improvement_pct': improvement,
            'pass': k1_d,
        }
        status = "PASS" if k1_d else "FAIL"
        print(f"    d={d}: std={std_ratio:.3f}x, best TAAP ({best_taap_name})="
              f"{best_taap_ratio:.3f}x, improvement={improvement:.2f}% -> {status}")

    # Also compare against std_ap_2000 (is TAAP better than just more iterations?)
    print(f"\n  K1b: TAAP (500 iter) better than std AP (2000 iter)?")
    for d in D_VALUES:
        long_ratio = aggregate[d]['methods']['std_ap_2000']['mean_over_welch']
        best_taap_ratio = k1_results[d]['best_taap_ratio']
        better = best_taap_ratio < long_ratio
        print(f"    d={d}: TAAP={best_taap_ratio:.3f}x vs std-2000={long_ratio:.3f}x "
              f"-> {'TAAP better' if better else 'More iterations better'}")

    # K2: TAAP runtime <= 3x standard AP
    print(f"\n  K2: TAAP runtime <= 3x standard AP?")
    k2_pass = True
    for d in D_VALUES:
        for mname in ['taap_schedule', 'taap_momentum', 'taap_selective']:
            ratio = aggregate[d]['methods'][mname]['time_ratio_vs_std']
            if ratio > 3.0:
                k2_pass = False
                print(f"    d={d}, {mname}: {ratio:.2f}x -> FAIL")
            else:
                print(f"    d={d}, {mname}: {ratio:.2f}x -> PASS")

    # Overall verdict
    print(f"\n{'='*72}")
    if k1_pass_any and k2_pass:
        print(f"  VERDICT: SUPPORTED")
        print(f"  At least one TAAP variant improves over standard AP.")
    elif not k1_pass_any:
        print(f"  VERDICT: KILLED (K1)")
        print(f"  No TAAP variant closes the gap to Welch bound beyond standard AP.")
        print(f"  The 2.8-3x gap is likely fundamental at these N/d ratios,")
        print(f"  confirming the minimax experiment's warning.")
    elif not k2_pass:
        print(f"  VERDICT: KILLED (K2)")
        print(f"  TAAP runtime exceeds 3x standard AP.")
    print(f"{'='*72}")

    # =========================================================================
    # Welch bound tightness diagnostic
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  WELCH BOUND TIGHTNESS DIAGNOSTIC")
    print(f"{'='*72}")
    for d in D_VALUES:
        N = N_PER_D[d]
        t = all_results[d]['tightness']
        best_achieved = min(
            aggregate[d]['methods'][m]['avg_mean_coh']
            for m in method_names
        )
        print(f"  d={d}, N={N}:")
        print(f"    Welch bound: {t['welch_bound']:.6f}")
        print(f"    Best achieved: {best_achieved:.6f} ({best_achieved/t['welch_bound']:.2f}x WB)")
        print(f"    Packing ratio Nr/d: {t['packing_ratio']:.2f}")
        print(f"    Max equiangular N: ~{t['max_equiangular_estimate']:.0f}")
        print(f"    N/max_equiangular: {t['N_over_max_equiangular']:.3f}")
        if t['N_over_max_equiangular'] < 0.1:
            print(f"    -> N << max: Welch bound likely LOOSE (ETF regime, low packing pressure)")
        elif t['N_over_max_equiangular'] < 0.5:
            print(f"    -> N moderate: Welch bound may be approachable")
        else:
            print(f"    -> N near max: Welch bound should be tight")

    elapsed = time.time() - t0_total
    print(f"\n  Total experiment time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'd_values': D_VALUES,
            'n_per_d': N_PER_D,
            'rank': R,
            'seeds': SEEDS,
            'n_iter_base': N_ITER_BASE,
            'n_iter_long': N_ITER_LONG,
        },
        'per_dimension': {
            str(d): {
                'd': all_results[d]['d'],
                'N': all_results[d]['N'],
                'welch_bound': all_results[d]['welch_bound'],
                'tightness': all_results[d]['tightness'],
                'per_seed': {
                    sk: {
                        mn: {k: v for k, v in mv.items() if k not in ('history_max', 'history_mean')}
                        for mn, mv in methods_d.items()
                    }
                    for sk, methods_d in all_results[d]['methods'].items()
                },
            }
            for d in D_VALUES
        },
        'aggregate': {str(d): a for d, a in aggregate.items()},
        'kill_criteria': {
            'k1_taap_closer_to_welch': k1_pass_any,
            'k1_details': {str(d): r for d, r in k1_results.items()},
            'k2_runtime_under_3x': k2_pass,
        },
        'elapsed_seconds': elapsed,
    }

    out = RESULTS_DIR / 'results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out}")

    return output


if __name__ == '__main__':
    run_experiment()
