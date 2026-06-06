#!/usr/bin/env python3
"""
Minimax Grassmannian Packing: worst-case interference guarantees.

Hypothesis: Post-AP stochastic minimax refinement reduces worst-case (max)
pairwise coherence while standard AP only controls mean coherence.

Mechanism: Standard AP converges to a max-equalized Gram matrix (max ~= mean
in the Gram space). But the frame extraction step (gram_to_frames) and
subsequent training introduce distributional tails. Minimax refinement
operates directly on extracted frames via stochastic local search on the
Grassmannian, specifically targeting and rotating the worst-case pair.

This is NOT a parameter change -- it's a fundamentally different optimization:
  AP:       Gram-space projection (global, all blocks simultaneously)
  Minimax:  Frame-space local search (one frame at a time, targeting worst pair)

Kill criteria:
  K1: Minimax max|cos| not lower than AP max|cos| (no worst-case improvement)
  K2: Minimax adds >2x compute vs standard AP

Parent: experiments/models/grassmannian_expert_init/
Pure numpy, CPU-only. Runtime target: < 5 minutes total.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Add parent to path for reuse
sys.path.insert(0, str(Path(__file__).parent.parent / 'grassmannian_expert_init'))
from grassmannian_expert_init import (
    DTYPE, VOCAB_SIZE, CONTEXT_LEN, LORA_RANK, LORA_ALPHA,
    D_VALUES, N_EXPERTS_PER_D, N_DOMAINS, D_CONFIG,
    welch_bound, random_grassmannian_points, frames_to_gram,
    block_norms, spectral_projection, gram_to_frames,
    structural_projection, alternating_projection as ap_standard,
    MicroMLP, generate_domain_data, init_lora_from_frame,
    init_lora_random_orthonormal, lora_delta_vec, train_lora,
    cosine_sim,
)

SEEDS = [42, 137]
AP_ITERATIONS = 500
REFINE_ITERATIONS = 500  # stochastic refinement steps


# =============================================================================
# Minimax refinement: stochastic local search on Grassmannian
# =============================================================================

def compute_pairwise_coherences(frames):
    """Compute upper-triangle pairwise coherences as (N, N) array."""
    N = frames.shape[0]
    cohs = np.zeros((N, N), dtype=DTYPE)
    for i in range(N):
        for j in range(i + 1, N):
            coh = np.linalg.norm(frames[i].T @ frames[j], 'fro')
            cohs[i, j] = coh
            cohs[j, i] = coh
    return cohs


def minimax_refine(frames, n_refine=500, rng=None):
    """
    Post-AP minimax refinement: stochastic local search on the Grassmannian
    that specifically targets and reduces the worst-case pairwise coherence.

    Algorithm:
    1. Compute pairwise coherence matrix
    2. Find worst pair (i*, j*) = argmax ||U_i^T U_j||_F
    3. Generate a random tangent vector at U_{i*} on the Stiefel manifold
    4. Retract to get a candidate new frame
    5. Accept if max coherence decreases (greedy descent on L_inf objective)
    6. Adaptive step size: grow on accept, shrink on reject

    This is fundamentally different from AP because:
    - AP operates on the Gram matrix (Nr x Nr) and modifies ALL blocks
    - Refinement operates on individual frames (d x r) one at a time
    - AP minimizes sum-of-squares; refinement minimizes max
    - AP uses convex projections; refinement uses stochastic descent
    """
    if rng is None:
        rng = np.random.RandomState(42)

    N, d, r = frames.shape
    frames = frames.copy()
    step_size = 0.1

    cohs = compute_pairwise_coherences(frames)
    best_max = float(cohs[np.triu_indices(N, k=1)].max())
    initial_max = best_max
    accepted = 0
    max_history = [best_max]

    for it in range(n_refine):
        # Find worst pair
        np.fill_diagonal(cohs, 0)
        flat_idx = cohs.argmax()
        worst_i, worst_j = divmod(flat_idx, N)

        # Perturb the frame involved in the worst pair
        target = worst_i if rng.rand() < 0.5 else worst_j
        old_frame = frames[target].copy()

        # Random tangent vector on Stiefel manifold at old_frame
        Z = rng.randn(d, r).astype(DTYPE) * step_size
        # Project to horizontal space: Z - U*(U^T*Z + Z^T*U)/2
        sym = old_frame.T @ Z
        Z = Z - old_frame @ (sym + sym.T) / 2
        candidate = old_frame + Z

        # Retract to Stiefel manifold via QR
        Q, R = np.linalg.qr(candidate)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        new_frame = (Q[:, :r] * signs[:r]).astype(DTYPE)

        # Compute new max coherence (only recompute row/col for target)
        frames[target] = new_frame
        new_cohs = cohs.copy()
        for j in range(N):
            if j != target:
                c = np.linalg.norm(frames[target].T @ frames[j], 'fro')
                new_cohs[target, j] = c
                new_cohs[j, target] = c

        np.fill_diagonal(new_cohs, 0)
        new_max = float(new_cohs[np.triu_indices(N, k=1)].max())

        if new_max < best_max:
            best_max = new_max
            cohs = new_cohs
            accepted += 1
            step_size = min(step_size * 1.05, 0.5)
        else:
            frames[target] = old_frame
            step_size = max(step_size * 0.97, 0.001)

        if it % 100 == 0:
            max_history.append(best_max)

    max_history.append(best_max)

    return frames, {
        'initial_max': initial_max,
        'final_max': best_max,
        'improvement': 1.0 - best_max / max(initial_max, 1e-12),
        'accepted': accepted,
        'acceptance_rate': accepted / max(n_refine, 1),
        'max_history': max_history,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(seeds=None, d_values=None):
    if seeds is None:
        seeds = SEEDS
    if d_values is None:
        d_values = D_VALUES

    results_dir = Path(__file__).parent
    t0 = time.time()

    print("=" * 72)
    print("  Minimax Grassmannian Packing: Worst-Case Interference Guarantees")
    print(f"  d={d_values}, seeds={seeds}, rank={LORA_RANK}")
    print(f"  AP iterations={AP_ITERATIONS}, Refine iterations={REFINE_ITERATIONS}")
    print("=" * 72)

    all_results = {}

    for seed in seeds:
        print(f"\n  === SEED {seed} ===")
        seed_results = {}

        for d in d_values:
            N = N_EXPERTS_PER_D[d]
            nl, d_ff_mult, steps, lr, n_seq, bs = D_CONFIG[d]
            d_ff = d_ff_mult * d
            r = LORA_RANK

            print(f"\n  d={d}, N={N}, layers={nl}, d_ff={d_ff}")

            wb = welch_bound(N, r, d)
            print(f"  Welch bound: {wb:.4f}")

            # ----------------------------------------------------------
            # Phase 1a: Standard AP skeleton
            # ----------------------------------------------------------
            rng_ap = np.random.RandomState(seed)
            t_ap_start = time.time()
            ap_frames, ap_hist = ap_standard(
                N, r, d, n_iter=AP_ITERATIONS, mu_factor=1.2, rng=rng_ap
            )
            t_ap = time.time() - t_ap_start

            # Measure AP skeleton coherences
            ap_cohs = compute_pairwise_coherences(ap_frames)
            ap_upper = ap_cohs[np.triu_indices(N, k=1)]
            ap_pre_max = float(ap_upper.max())
            ap_pre_mean = float(ap_upper.mean())
            ap_pre_p95 = float(np.percentile(ap_upper, 95))
            ap_pre_p99 = float(np.percentile(ap_upper, 99))

            print(f"\n  Standard AP ({AP_ITERATIONS} iters, {t_ap:.1f}s):")
            print(f"    Pre-train coherence: max={ap_pre_max:.4f}, mean={ap_pre_mean:.4f}, "
                  f"p95={ap_pre_p95:.4f}, max/mean={ap_pre_max/max(ap_pre_mean, 1e-12):.2f}x")

            # ----------------------------------------------------------
            # Phase 1b: Minimax refinement (starts from AP skeleton)
            # ----------------------------------------------------------
            rng_refine = np.random.RandomState(seed + 7777)
            t_mm_start = time.time()
            mm_frames, refine_info = minimax_refine(
                ap_frames, n_refine=REFINE_ITERATIONS, rng=rng_refine
            )
            t_mm = time.time() - t_mm_start
            t_total_mm = t_ap + t_mm  # total = AP + refinement

            mm_cohs = compute_pairwise_coherences(mm_frames)
            mm_upper = mm_cohs[np.triu_indices(N, k=1)]
            mm_pre_max = float(mm_upper.max())
            mm_pre_mean = float(mm_upper.mean())
            mm_pre_p95 = float(np.percentile(mm_upper, 95))
            mm_pre_p99 = float(np.percentile(mm_upper, 99))

            print(f"\n  Minimax refinement ({REFINE_ITERATIONS} iters, {t_mm:.1f}s):")
            print(f"    Pre-train coherence: max={mm_pre_max:.4f}, mean={mm_pre_mean:.4f}, "
                  f"p95={mm_pre_p95:.4f}, max/mean={mm_pre_max/max(mm_pre_mean, 1e-12):.2f}x")
            print(f"    Max coherence improvement: {refine_info['initial_max']:.4f} -> "
                  f"{refine_info['final_max']:.4f} ({100*refine_info['improvement']:.1f}%)")
            print(f"    Accepted/total: {refine_info['accepted']}/{REFINE_ITERATIONS} "
                  f"({100*refine_info['acceptance_rate']:.1f}%)")
            print(f"    Total time (AP+refine): {t_total_mm:.1f}s vs AP-only: {t_ap:.1f}s "
                  f"(ratio: {t_total_mm/max(t_ap, 0.01):.2f}x)")

            # ----------------------------------------------------------
            # Phase 2: Train experts with both skeletons + ortho baseline
            # ----------------------------------------------------------
            rng_model = np.random.RandomState(seed + d)
            model = MicroMLP(d, nl, d_ff_mult, rng_model)
            n_train = min(N, N_DOMAINS)

            # Train with AP skeleton
            ap_deltas, ap_losses = [], []
            for i in range(n_train):
                domain_id = i + seed * 100
                x, y = generate_domain_data(domain_id, n_seq)
                A1, B1, A2, B2 = init_lora_from_frame(ap_frames[i % N], d, d_ff, nl)
                A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2, steps, lr, bs)
                ap_deltas.append(lora_delta_vec(A1, B1, A2, B2))
                ap_losses.append(float(loss))

            # Train with minimax skeleton
            mm_deltas, mm_losses = [], []
            for i in range(n_train):
                domain_id = i + seed * 100
                x, y = generate_domain_data(domain_id, n_seq)
                A1, B1, A2, B2 = init_lora_from_frame(mm_frames[i % N], d, d_ff, nl)
                A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2, steps, lr, bs)
                mm_deltas.append(lora_delta_vec(A1, B1, A2, B2))
                mm_losses.append(float(loss))

            # Random-orthonormal baseline
            ortho_deltas, ortho_losses = [], []
            for i in range(n_train):
                domain_id = i + seed * 100
                x, y = generate_domain_data(domain_id, n_seq)
                rng_lora = np.random.RandomState(seed + d + i * 31 + 5000)
                A1, B1, A2, B2 = init_lora_random_orthonormal(d, d_ff, nl, rng_lora)
                A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2, steps, lr, bs)
                ortho_deltas.append(lora_delta_vec(A1, B1, A2, B2))
                ortho_losses.append(float(loss))

            # ----------------------------------------------------------
            # Phase 3: Measure pairwise cosine after training
            # ----------------------------------------------------------
            def compute_post_cos(deltas):
                post_cos = []
                for i in range(len(deltas)):
                    for j in range(i + 1, len(deltas)):
                        post_cos.append(abs(cosine_sim(deltas[i], deltas[j])))
                return post_cos

            ap_post = compute_post_cos(ap_deltas)
            mm_post = compute_post_cos(mm_deltas)
            ortho_post = compute_post_cos(ortho_deltas)

            ap_arr = np.array(ap_post)
            mm_arr = np.array(mm_post)
            ortho_arr = np.array(ortho_post)

            print(f"\n  Post-training |cos| (delta vectors):")
            for label, arr in [('AP-standard', ap_arr), ('AP+minimax', mm_arr), ('Random-ortho', ortho_arr)]:
                print(f"    {label:20s}: mean={arr.mean():.6f}, max={arr.max():.6f}, "
                      f"p95={np.percentile(arr, 95):.6f}, "
                      f"max/mean={arr.max()/max(arr.mean(), 1e-12):.2f}x")

            # Wilcoxon: minimax vs standard (key test)
            try:
                _, p_mm_std = wilcoxon(mm_arr, ap_arr, alternative='less')
            except ValueError:
                p_mm_std = 1.0
            try:
                _, p_mm_ortho = wilcoxon(mm_arr, ortho_arr, alternative='less')
            except ValueError:
                p_mm_ortho = 1.0
            try:
                _, p_ap_ortho = wilcoxon(ap_arr, ortho_arr, alternative='less')
            except ValueError:
                p_ap_ortho = 1.0

            print(f"\n  Wilcoxon (one-sided, 'less'):")
            print(f"    Minimax < Standard: p={p_mm_std:.4f} {'*' if p_mm_std < 0.05 else 'n.s.'}")
            print(f"    Minimax < Ortho:    p={p_mm_ortho:.4f} {'*' if p_mm_ortho < 0.05 else 'n.s.'}")
            print(f"    Standard < Ortho:   p={p_ap_ortho:.4f} {'*' if p_ap_ortho < 0.05 else 'n.s.'}")

            seed_results[d] = {
                'd': d, 'N': N, 'welch_bound': float(wb),
                'timing': {
                    'ap_seconds': t_ap,
                    'refine_seconds': t_mm,
                    'total_mm_seconds': t_total_mm,
                    'ratio': t_total_mm / max(t_ap, 0.01),
                },
                'pre_training': {
                    'ap': {'max': ap_pre_max, 'mean': ap_pre_mean, 'p95': ap_pre_p95, 'p99': ap_pre_p99},
                    'minimax': {'max': mm_pre_max, 'mean': mm_pre_mean, 'p95': mm_pre_p95, 'p99': mm_pre_p99},
                },
                'refine_info': refine_info,
                'post_training': {
                    'ap': {
                        'mean': float(ap_arr.mean()), 'max': float(ap_arr.max()),
                        'std': float(ap_arr.std()),
                        'p95': float(np.percentile(ap_arr, 95)),
                        'p99': float(np.percentile(ap_arr, 99)),
                        'max_over_mean': float(ap_arr.max() / max(ap_arr.mean(), 1e-12)),
                        'cosines': [float(x) for x in ap_arr],
                    },
                    'minimax': {
                        'mean': float(mm_arr.mean()), 'max': float(mm_arr.max()),
                        'std': float(mm_arr.std()),
                        'p95': float(np.percentile(mm_arr, 95)),
                        'p99': float(np.percentile(mm_arr, 99)),
                        'max_over_mean': float(mm_arr.max() / max(mm_arr.mean(), 1e-12)),
                        'cosines': [float(x) for x in mm_arr],
                    },
                    'ortho': {
                        'mean': float(ortho_arr.mean()), 'max': float(ortho_arr.max()),
                        'std': float(ortho_arr.std()),
                        'p95': float(np.percentile(ortho_arr, 95)),
                        'p99': float(np.percentile(ortho_arr, 99)),
                        'max_over_mean': float(ortho_arr.max() / max(ortho_arr.mean(), 1e-12)),
                        'cosines': [float(x) for x in ortho_arr],
                    },
                },
                'statistical_tests': {
                    'mm_vs_ap_p': float(p_mm_std),
                    'mm_vs_ortho_p': float(p_mm_ortho),
                    'ap_vs_ortho_p': float(p_ap_ortho),
                },
                'losses': {
                    'ap_mean': float(np.mean(ap_losses)),
                    'mm_mean': float(np.mean(mm_losses)),
                    'ortho_mean': float(np.mean(ortho_losses)),
                },
            }

        all_results[seed] = seed_results

    elapsed = time.time() - t0

    # =================================================================
    # Aggregate across seeds
    # =================================================================
    print(f"\n{'='*72}")
    print(f"  AGGREGATE ({len(seeds)} seeds)")
    print(f"{'='*72}")

    aggregate = {}
    for d in d_values:
        agg = {'d': d, 'N': N_EXPERTS_PER_D[d]}

        for label in ['ap', 'minimax', 'ortho']:
            cos_all = []
            for s in seeds:
                cos_all.extend(all_results[s][d]['post_training'][label]['cosines'])
            arr = np.array(cos_all)
            agg[f'{label}_mean'] = float(arr.mean())
            agg[f'{label}_max'] = float(arr.max())
            agg[f'{label}_std'] = float(arr.std())
            agg[f'{label}_p95'] = float(np.percentile(arr, 95))
            agg[f'{label}_p99'] = float(np.percentile(arr, 99))
            agg[f'{label}_max_over_mean'] = float(arr.max() / max(arr.mean(), 1e-12))

        # Aggregate Wilcoxon
        ap_all, mm_all, ort_all = [], [], []
        for s in seeds:
            ap_all.extend(all_results[s][d]['post_training']['ap']['cosines'])
            mm_all.extend(all_results[s][d]['post_training']['minimax']['cosines'])
            ort_all.extend(all_results[s][d]['post_training']['ortho']['cosines'])
        ap_a, mm_a, ort_a = np.array(ap_all), np.array(mm_all), np.array(ort_all)

        try:
            _, p_mm_ap = wilcoxon(mm_a, ap_a, alternative='less')
        except ValueError:
            p_mm_ap = 1.0
        try:
            _, p_mm_ort = wilcoxon(mm_a, ort_a, alternative='less')
        except ValueError:
            p_mm_ort = 1.0

        agg['wilcoxon_mm_vs_ap_p'] = float(p_mm_ap)
        agg['wilcoxon_mm_vs_ortho_p'] = float(p_mm_ort)

        # Timing
        t_ap_avg = np.mean([all_results[s][d]['timing']['ap_seconds'] for s in seeds])
        t_mm_avg = np.mean([all_results[s][d]['timing']['total_mm_seconds'] for s in seeds])
        agg['time_ap'] = float(t_ap_avg)
        agg['time_mm'] = float(t_mm_avg)
        agg['time_ratio'] = float(t_mm_avg / max(t_ap_avg, 0.01))

        # Pre-training coherence improvement
        pre_ap_max = np.mean([all_results[s][d]['pre_training']['ap']['max'] for s in seeds])
        pre_mm_max = np.mean([all_results[s][d]['pre_training']['minimax']['max'] for s in seeds])
        pre_improvement = 1.0 - pre_mm_max / max(pre_ap_max, 1e-12)
        agg['pre_ap_max'] = float(pre_ap_max)
        agg['pre_mm_max'] = float(pre_mm_max)
        agg['pre_improvement'] = float(pre_improvement)

        # Refine acceptance rate
        acc_rate = np.mean([all_results[s][d]['refine_info']['acceptance_rate'] for s in seeds])
        agg['refine_acceptance_rate'] = float(acc_rate)

        aggregate[d] = agg

        print(f"\n  d={d}, N={N_EXPERTS_PER_D[d]}:")
        print(f"    Pre-training max coherence improvement: "
              f"{pre_ap_max:.4f} -> {pre_mm_max:.4f} ({100*pre_improvement:.1f}%)")
        print(f"    Refine acceptance rate: {100*acc_rate:.1f}%")
        print(f"    Post-training |cos|:")
        for lbl in ['ap', 'minimax', 'ortho']:
            name = {'ap': 'AP-standard', 'minimax': 'AP+minimax', 'ortho': 'Random-ortho'}[lbl]
            print(f"      {name:20s}: mean={agg[f'{lbl}_mean']:.6f}, "
                  f"max={agg[f'{lbl}_max']:.6f}, "
                  f"p95={agg[f'{lbl}_p95']:.6f}, "
                  f"max/mean={agg[f'{lbl}_max_over_mean']:.2f}x")
        print(f"    Wilcoxon: mm<ap p={agg['wilcoxon_mm_vs_ap_p']:.4f}")
        print(f"    Timing: mm/ap = {agg['time_ratio']:.2f}x")

    # =================================================================
    # Kill Criteria
    # =================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA")
    print(f"{'='*72}")

    # K1: Minimax max|cos| < AP max|cos|
    print("\n  K1: Minimax post-training max|cos| < AP post-training max|cos|")
    k1_results = []
    for d in d_values:
        a = aggregate[d]
        mm_max = a['minimax_max']
        ap_max = a['ap_max']
        improvement = 1.0 - mm_max / max(ap_max, 1e-12)
        k1_pass = mm_max < ap_max
        k1_results.append(k1_pass)
        print(f"    d={d}: minimax={mm_max:.6f}, AP={ap_max:.6f}, "
              f"improvement={100*improvement:+.1f}% -> {'PASS' if k1_pass else 'FAIL'}")
    k1 = all(k1_results)
    print(f"  K1 overall: {'PASS' if k1 else 'FAIL'}")

    # K1b: Also check pre-training (skeleton-level) improvement
    print("\n  K1b: Pre-training skeleton max coherence improvement")
    k1b_results = []
    for d in d_values:
        a = aggregate[d]
        k1b_pass = a['pre_mm_max'] < a['pre_ap_max']
        k1b_results.append(k1b_pass)
        print(f"    d={d}: minimax={a['pre_mm_max']:.4f}, AP={a['pre_ap_max']:.4f}, "
              f"improvement={100*a['pre_improvement']:.1f}% -> {'PASS' if k1b_pass else 'FAIL'}")
    print(f"  K1b overall: {'PASS' if all(k1b_results) else 'FAIL'}")

    # K2: Compute within 2x
    print(f"\n  K2: Total compute (AP+refine) <= 2x AP-only")
    k2_results = []
    for d in d_values:
        a = aggregate[d]
        ratio = a['time_ratio']
        k2_pass = ratio <= 2.0
        k2_results.append(k2_pass)
        print(f"    d={d}: ratio={ratio:.2f}x -> {'PASS' if k2_pass else 'FAIL'}")
    k2 = all(k2_results)
    print(f"  K2 overall: {'PASS' if k2 else 'FAIL'}")

    # Tail compression
    print(f"\n  Tail compression (max/mean ratio, informational):")
    for d in d_values:
        a = aggregate[d]
        ap_ratio = a['ap_max_over_mean']
        mm_ratio = a['minimax_max_over_mean']
        print(f"    d={d}: AP={ap_ratio:.2f}x, minimax={mm_ratio:.2f}x, "
              f"improvement={100*(1-mm_ratio/max(ap_ratio, 1e-12)):+.1f}%")

    # Verdict
    print(f"\n{'='*72}")
    if k1 and k2:
        print(f"  VERDICT: PROVEN")
        print(f"  Minimax refinement provides lower worst-case coherence than")
        print(f"  standard AP with acceptable compute overhead.")
    elif not k1:
        print(f"  VERDICT: KILLED (K1)")
        print(f"  Minimax does NOT consistently reduce max|cos| below AP.")
    elif not k2:
        print(f"  VERDICT: KILLED (K2)")
        print(f"  Minimax adds >2x compute overhead.")
    print(f"{'='*72}")

    # Summary table
    print(f"\n  {'d':>4} | {'N':>3} | {'AP max':>10} | {'MM max':>10} | {'Ort max':>10} | "
          f"{'MM/AP':>7} | {'Pre %':>7} | {'Time x':>7}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for d in d_values:
        a = aggregate[d]
        mm_ap_ratio = a['minimax_max'] / max(a['ap_max'], 1e-12)
        print(f"  {d:4d} | {a['N']:3d} | {a['ap_max']:10.6f} | "
              f"{a['minimax_max']:10.6f} | {a['ortho_max']:10.6f} | "
              f"{mm_ap_ratio:7.4f} | {100*a['pre_improvement']:+6.1f}% | "
              f"{a['time_ratio']:7.2f}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'seeds': seeds, 'd_values': d_values,
            'n_experts_per_d': N_EXPERTS_PER_D,
            'rank': LORA_RANK,
            'ap_iterations': AP_ITERATIONS,
            'refine_iterations': REFINE_ITERATIONS,
        },
        'per_seed': {str(s): {str(d): r for d, r in sr.items()} for s, sr in all_results.items()},
        'aggregate': {str(d): a for d, a in aggregate.items()},
        'kill_criteria': {
            'k1_minimax_max_below_ap_max': k1,
            'k1b_pre_training_improvement': all(k1b_results),
            'k2_compute_within_2x': k2,
            'overall': k1 and k2,
        },
        'elapsed_seconds': elapsed,
    }

    out = results_dir / 'results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out}")

    return output


if __name__ == '__main__':
    run_experiment()
