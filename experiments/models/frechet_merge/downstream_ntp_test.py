#!/usr/bin/env python3
"""
Downstream NTP-like test: does Frechet merge produce better composed models?

This test addresses the gap identified in adversarial review: the main experiment
measures only geometric proxies (subspace preservation), not whether the merge
produces a model that better predicts held-out data.

Design:
  1. Create a simple linear "language model": y = W_base @ x + noise
  2. Train N LoRA experts (A_i, B_i) on N different "domains" (random projections
     of the input space, simulating domain-specific patterns)
  3. Compose all experts via (a) naive addition and (b) chordal Frechet merge
  4. Measure reconstruction MSE on held-out domain data for each method

This is a linear approximation to NTP loss. At micro scale with synthetic data,
MSE on a linear model is the cleanest proxy for "does the composed model
capture the expert knowledge?"

The key question: does the geometric advantage of Frechet merge (better subspace
preservation) translate to lower reconstruction error on held-out data?
"""

import json
import time
from pathlib import Path

import numpy as np
from frechet_merge import (
    chordal_frechet_mean, make_delta_from_subspace, naive_addition,
    random_subspace,
)

DTYPE = np.float64


def generate_domain_data(d_in, d_out, n_samples, W_true, noise_scale=0.1, rng=None):
    """Generate (X, Y) pairs from Y = W_true @ X + noise."""
    X = rng.randn(n_samples, d_in).astype(DTYPE)
    noise = rng.randn(n_samples, d_out).astype(DTYPE) * noise_scale
    Y = X @ W_true.T + noise
    return X, Y


def train_lora_expert(X, Y, W_base, rank, lr=0.01, n_steps=200, rng=None):
    """
    Train a LoRA expert (A, B) to minimize ||Y - (W_base + A @ B) @ X^T||^2.

    Uses alternating least squares:
      Fix A, solve for B: B = (A^T A)^{-1} A^T (W_residual)
      Fix B, solve for A: A = (W_residual) B^T (B B^T)^{-1}

    Returns trained (A, B) where A is (d_out, rank), B is (rank, d_in).
    """
    d_out, d_in = W_base.shape

    # Initialize
    A = rng.randn(d_out, rank).astype(DTYPE) * 0.01
    B = rng.randn(rank, d_in).astype(DTYPE) * 0.01

    # Compute residual target
    W_residual = (Y.T @ X) / X.shape[0] - W_base  # (d_out, d_in) least-squares target

    for step in range(n_steps):
        # Fix A, solve for B
        AtA = A.T @ A + 1e-6 * np.eye(rank, dtype=DTYPE)
        B = np.linalg.solve(AtA, A.T @ W_residual)

        # Fix B, solve for A
        BBt = B @ B.T + 1e-6 * np.eye(rank, dtype=DTYPE)
        A = W_residual @ B.T @ np.linalg.inv(BBt)

    return A, B


def evaluate_composed_model(W_composed, X_test, Y_test):
    """MSE of the composed model on held-out data."""
    Y_pred = X_test @ W_composed.T
    mse = np.mean((Y_test - Y_pred) ** 2)
    return float(mse)


def run_downstream_test(d=64, N=10, rank=8, n_train=500, n_test=200,
                         noise_scale=0.1, seed=42):
    """
    Full downstream comparison:
      1. Generate a ground-truth W_full that is the sum of N domain-specific patterns
      2. Train N LoRA experts on domain-specific data
      3. Compose via naive addition and chordal Frechet merge
      4. Measure MSE on held-out data from each domain and overall
    """
    rng = np.random.RandomState(seed)
    d_in = d
    d_out = d

    # Generate base model (random)
    W_base = rng.randn(d_out, d_in).astype(DTYPE) * 0.1

    # Generate N domain-specific "true" weight perturbations
    # Each domain has a low-rank (rank-r) perturbation
    domain_deltas = []
    for i in range(N):
        # True perturbation is rank-r
        A_true = rng.randn(d_out, rank).astype(DTYPE) * 0.3
        B_true = rng.randn(rank, d_in).astype(DTYPE) * 0.3
        domain_deltas.append(A_true @ B_true)

    # The "full expert" for each domain
    W_domains = [W_base + dd for dd in domain_deltas]

    # Generate training and test data for each domain
    train_data = []
    test_data = []
    for i in range(N):
        X_train, Y_train = generate_domain_data(
            d_in, d_out, n_train, W_domains[i], noise_scale, rng)
        X_test, Y_test = generate_domain_data(
            d_in, d_out, n_test, W_domains[i], noise_scale, rng)
        train_data.append((X_train, Y_train))
        test_data.append((X_test, Y_test))

    # Train N LoRA experts
    A_list = []
    B_list = []
    for i in range(N):
        X_tr, Y_tr = train_data[i]
        A, B = train_lora_expert(X_tr, Y_tr, W_base, rank, n_steps=100, rng=rng)
        A_list.append(A)
        B_list.append(B)

    # ---- Method 1: Naive addition ----
    delta_naive = naive_addition(A_list, B_list, alpha=1.0, rank=rank)
    W_naive = W_base + delta_naive

    # ---- Method 2: Chordal Frechet merge ----
    merged_chordal = chordal_frechet_mean(A_list, rank)
    delta_chordal = make_delta_from_subspace(merged_chordal, B_list, A_list,
                                              alpha=1.0, rank=rank)
    W_chordal = W_base + delta_chordal

    # ---- Method 3: Base model (no experts) ----
    # Baseline: how well does the base model do?

    # ---- Evaluate on all domains ----
    mse_base_per_domain = []
    mse_naive_per_domain = []
    mse_chordal_per_domain = []

    for i in range(N):
        X_test, Y_test = test_data[i]
        mse_base_per_domain.append(evaluate_composed_model(W_base, X_test, Y_test))
        mse_naive_per_domain.append(evaluate_composed_model(W_naive, X_test, Y_test))
        mse_chordal_per_domain.append(evaluate_composed_model(W_chordal, X_test, Y_test))

    # Also evaluate on pooled test data
    X_all = np.concatenate([td[0] for td in test_data])
    Y_all = np.concatenate([td[1] for td in test_data])
    mse_base_all = evaluate_composed_model(W_base, X_all, Y_all)
    mse_naive_all = evaluate_composed_model(W_naive, X_all, Y_all)
    mse_chordal_all = evaluate_composed_model(W_chordal, X_all, Y_all)

    # Per-domain advantage: fraction of domains where chordal wins
    chordal_wins = sum(1 for i in range(N)
                       if mse_chordal_per_domain[i] < mse_naive_per_domain[i])

    result = {
        'd': d, 'N': N, 'rank': rank, 'seed': seed,
        'n_train': n_train, 'n_test': n_test, 'noise_scale': noise_scale,
        'mse_base_overall': mse_base_all,
        'mse_naive_overall': mse_naive_all,
        'mse_chordal_overall': mse_chordal_all,
        'mse_base_per_domain': mse_base_per_domain,
        'mse_naive_per_domain': mse_naive_per_domain,
        'mse_chordal_per_domain': mse_chordal_per_domain,
        'chordal_wins_count': chordal_wins,
        'chordal_wins_frac': chordal_wins / N,
        'naive_improvement_over_base': (mse_base_all - mse_naive_all) / mse_base_all,
        'chordal_improvement_over_base': (mse_base_all - mse_chordal_all) / mse_base_all,
        'chordal_vs_naive_pct': (mse_naive_all - mse_chordal_all) / mse_naive_all * 100,
    }

    return result


def run_full_downstream_experiment():
    """Sweep across N values and dimensions, comparing naive vs chordal on MSE."""
    results_dir = Path(__file__).parent
    t_total = time.time()

    print("=" * 76)
    print("  Downstream NTP-like Test: Naive vs Chordal Frechet Merge")
    print("  Does geometric advantage translate to reconstruction quality?")
    print("=" * 76)

    all_results = []

    # Sweep N at fixed d=64 (fast)
    for seed in [42, 137, 271]:
        for d in [64, 128, 256]:
            for N in [2, 5, 10, 25]:
                # Skip large configs to keep runtime bounded
                if d >= 256 and N >= 25:
                    continue

                result = run_downstream_test(d=d, N=N, rank=8, n_train=500,
                                              n_test=200, seed=seed)
                all_results.append(result)

                print(f"  d={d:4d} N={N:2d} seed={seed:3d} | "
                      f"base={result['mse_base_overall']:.4f} "
                      f"naive={result['mse_naive_overall']:.4f} "
                      f"chordal={result['mse_chordal_overall']:.4f} | "
                      f"chordal vs naive: {result['chordal_vs_naive_pct']:+.2f}% | "
                      f"chordal wins {result['chordal_wins_count']}/{N} domains")

    elapsed = time.time() - t_total

    # Aggregate by (d, N)
    from collections import defaultdict
    by_d_n = defaultdict(list)
    for r in all_results:
        by_d_n[(r['d'], r['N'])].append(r)

    print(f"\n{'='*76}")
    print(f"  AGGREGATED RESULTS (mean over seeds)")
    print(f"{'='*76}")
    print(f"  {'d':>5} {'N':>3} | {'Base MSE':>10} {'Naive MSE':>10} {'Chordal MSE':>11} | {'Ch vs Na':>10} {'Ch wins':>8}")
    print(f"  {'-'*5} {'-'*3}-+-{'-'*10}-{'-'*10}-{'-'*11}-+-{'-'*10}-{'-'*8}")

    for d in [64, 128, 256]:
        for N in [2, 5, 10, 25]:
            key = (d, N)
            if key not in by_d_n:
                continue
            results = by_d_n[key]
            base = np.mean([r['mse_base_overall'] for r in results])
            naive = np.mean([r['mse_naive_overall'] for r in results])
            chordal = np.mean([r['mse_chordal_overall'] for r in results])
            ch_vs_na = np.mean([r['chordal_vs_naive_pct'] for r in results])
            ch_wins = np.mean([r['chordal_wins_frac'] for r in results])
            print(f"  {d:5d} {N:3d} | {base:10.4f} {naive:10.4f} {chordal:11.4f} | {ch_vs_na:+9.2f}% | {ch_wins:7.1%}")

    # Overall summary
    all_ch_vs_na = [r['chordal_vs_naive_pct'] for r in all_results]
    all_ch_wins = [r['chordal_wins_frac'] for r in all_results]

    print(f"\n  Overall chordal vs naive MSE: {np.mean(all_ch_vs_na):+.2f}% "
          f"(std: {np.std(all_ch_vs_na):.2f}%)")
    print(f"  Overall domain win rate: {np.mean(all_ch_wins):.1%}")
    print(f"  Total time: {elapsed:.1f}s")

    # Verdict
    mean_advantage = np.mean(all_ch_vs_na)
    if mean_advantage > 1.0:
        verdict = "SUPPORTED: chordal merge produces measurably lower reconstruction error"
    elif mean_advantage > 0.0:
        verdict = "WEAK: chordal advantage is positive but small (<1%)"
    else:
        verdict = "NOT SUPPORTED: naive merge produces equal or lower reconstruction error"

    print(f"\n  VERDICT: {verdict}")

    # Save
    output = {
        'description': 'Downstream reconstruction test: trained LoRA experts composed via naive vs chordal',
        'results': all_results,
        'summary': {
            'mean_chordal_vs_naive_pct': float(np.mean(all_ch_vs_na)),
            'std_chordal_vs_naive_pct': float(np.std(all_ch_vs_na)),
            'mean_domain_win_rate': float(np.mean(all_ch_wins)),
            'n_configs': len(all_results),
            'verdict': verdict,
        },
        'elapsed_seconds': elapsed,
    }

    out = results_dir / 'downstream_results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out}")

    return output


if __name__ == '__main__':
    run_full_downstream_experiment()
