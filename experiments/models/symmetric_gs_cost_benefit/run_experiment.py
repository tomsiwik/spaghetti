"""Symmetric GS Cost-Benefit Analysis.

Does symmetric Gram-Schmidt (averaging over P orderings) provide measurable
quality benefit over a single deterministic ordering, particularly at high
pairwise cosines (cos=0.85, matching attention-layer production measurements)?

Builds on:
  - merge_order_dependence (proven): variation ~ 80*cos, threshold at cos>0.06
  - layerwise_order_sensitivity (killed K2): attn/FFN scaling identical (1.01x)
  - gs_random_permutation_validation (proven): random permutation equalizes position

Design:
  Phase 1: Cost-benefit sweep across cosine values (0.01 to 0.90) and
           P values (1, 5, 10, 20, 50, 100). Measure quality of symmetric GS
           vs best/worst/canonical single ordering.

  Phase 2: High-overlap deep dive at cos=0.85. Full statistical characterization
           of the symmetric GS quality distribution vs single-ordering distribution.

  Phase 3: Convergence analysis. How quickly does symmetric GS converge?
           Is P=5 sufficient, or do you need P=100?

Kill Criteria:
  K1: symmetric GS quality within 0.1% of best single ordering (no benefit worth 50x cost)
  K2: symmetric GS at cos=0.85 shows <1% improvement over random ordering
"""

import json
import random
import statistics
import time
from itertools import permutations
from pathlib import Path

import numpy as np


# ── Config ──────────────────────────────────────────────────────────────────

N_EXPERTS = 5       # Matching parent experiment
DIM = 256           # Per parent experiment (merge_order_dependence Phase 2 used ~1280 flat)
SEEDS = [42, 7, 123]
N_SINGLE_ORDERINGS = 100  # Sample of single orderings for baseline distribution


# ── Core GS ─────────────────────────────────────────────────────────────────

def gs_orthogonalize(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Modified Gram-Schmidt on a list of numpy vectors."""
    ortho = []
    for k in range(len(vectors)):
        v = vectors[k].copy()
        for i in range(len(ortho)):
            e_i = ortho[i]
            dot_ee = np.dot(e_i, e_i)
            if dot_ee > 1e-12:
                v = v - (np.dot(v, e_i) / dot_ee) * e_i
        ortho.append(v)
    return ortho


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Synthetic Expert Creation ───────────────────────────────────────────────

def create_synthetic_experts(n: int, dim: int, target_cos: float,
                              seed: int = 42) -> list[np.ndarray]:
    """Create n experts with pairwise cosine ~ target_cos.

    Each expert = alpha * shared_direction + beta * unique_direction,
    where alpha^2 ~ target_cos and alpha^2 + beta^2 = 1 (unit norm).
    """
    rng = np.random.RandomState(seed)

    # Shared direction (the overlapping subspace)
    shared = rng.randn(dim)
    shared /= np.linalg.norm(shared)

    alpha = np.sqrt(max(target_cos, 0.0))
    beta = np.sqrt(max(1.0 - target_cos, 0.0))

    experts = []
    for _ in range(n):
        unique = rng.randn(dim)
        unique /= np.linalg.norm(unique)
        # Remove shared component to make truly orthogonal
        unique = unique - np.dot(unique, shared) * shared
        unique /= np.linalg.norm(unique)
        v = alpha * shared + beta * unique
        v /= np.linalg.norm(v)  # Normalize
        experts.append(v)

    return experts


def measure_actual_cosine(experts: list[np.ndarray]) -> float:
    """Mean pairwise |cosine| across all expert pairs."""
    N = len(experts)
    cos_vals = []
    for i in range(N):
        for j in range(i + 1, N):
            cos_vals.append(abs(cosine_sim(experts[i], experts[j])))
    return float(np.mean(cos_vals))


# ── Merge Quality Metrics ───────────────────────────────────────────────────

def gs_merge_with_ordering(experts: list[np.ndarray],
                            ordering: list[int]) -> np.ndarray:
    """Apply GS in given ordering, return mean of orthogonalized vectors."""
    ordered = [experts[i] for i in ordering]
    ortho = gs_orthogonalize(ordered)
    return np.mean(ortho, axis=0)


def gs_signal_retention(experts: list[np.ndarray],
                         ordering: list[int]) -> dict:
    """Compute per-expert signal retention and merged vector for one ordering."""
    ordered = [experts[i] for i in ordering]
    ortho = gs_orthogonalize(ordered)

    retentions = []
    for k in range(len(experts)):
        orig_norm = np.linalg.norm(ordered[k])
        ortho_norm = np.linalg.norm(ortho[k])
        retentions.append(ortho_norm / orig_norm if orig_norm > 1e-12 else 0.0)

    merged = np.mean(ortho, axis=0)
    merged_norm = np.linalg.norm(merged)

    return {
        'retentions': retentions,
        'min_retention': min(retentions),
        'mean_retention': float(np.mean(retentions)),
        'merged_norm': merged_norm,
        'merged': merged,
    }


def symmetric_gs(experts: list[np.ndarray], P: int,
                  seed: int = 42) -> dict:
    """Symmetric GS: average GS over P random orderings.

    Returns:
        merged: average merged vector across P orderings
        info: statistics about the averaging
    """
    N = len(experts)
    rng = random.Random(seed)

    merged_vectors = []
    all_retentions = []
    all_min_retentions = []

    for p in range(P):
        ordering = list(range(N))
        rng.shuffle(ordering)

        result = gs_signal_retention(experts, ordering)
        merged_vectors.append(result['merged'])
        all_retentions.extend(result['retentions'])
        all_min_retentions.append(result['min_retention'])

    # Average merged vector
    avg_merged = np.mean(merged_vectors, axis=0)
    avg_merged_norm = np.linalg.norm(avg_merged)

    # Quality metric: norm of averaged merged vector
    # Higher = more signal retained (orderings agreed on direction)
    # Lower = orderings disagreed (signal cancelled)
    individual_norms = [np.linalg.norm(v) for v in merged_vectors]

    return {
        'merged': avg_merged,
        'merged_norm': avg_merged_norm,
        'mean_individual_norm': float(np.mean(individual_norms)),
        'mean_retention': float(np.mean(all_retentions)),
        'mean_min_retention': float(np.mean(all_min_retentions)),
        'individual_norms': individual_norms,
        'P': P,
    }


# ── Phase 1: Cost-Benefit Sweep ────────────────────────────────────────────

def run_phase1():
    """Sweep across cosine values and P values.

    For each (cosine, P), measure:
    - Symmetric GS quality (norm of averaged merged vector)
    - Best single ordering quality
    - Worst single ordering quality
    - Canonical ordering (sorted: 0,1,2,...,N-1) quality
    - Improvement of symmetric over canonical/random
    """
    print(f"\n{'='*70}")
    print("PHASE 1: COST-BENEFIT SWEEP")
    print(f"{'='*70}")
    t0 = time.time()

    cosine_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.90]
    P_values = [1, 5, 10, 20, 50, 100]

    results = []

    for cos_target in cosine_values:
        for seed in SEEDS:
            experts = create_synthetic_experts(N_EXPERTS, DIM, cos_target, seed)
            actual_cos = measure_actual_cosine(experts)

            # Canonical ordering (deterministic, zero cost)
            canonical_ordering = list(range(N_EXPERTS))
            canonical_result = gs_signal_retention(experts, canonical_ordering)

            # Sample many single orderings to get best/worst/distribution
            rng = random.Random(seed + 1000)
            single_norms = []
            single_min_retentions = []
            for _ in range(N_SINGLE_ORDERINGS):
                ordering = list(range(N_EXPERTS))
                rng.shuffle(ordering)
                result = gs_signal_retention(experts, ordering)
                single_norms.append(result['merged_norm'])
                single_min_retentions.append(result['min_retention'])

            best_norm = max(single_norms)
            worst_norm = min(single_norms)
            mean_norm = float(np.mean(single_norms))
            std_norm = float(np.std(single_norms))

            # Symmetric GS at various P
            for P in P_values:
                sym = symmetric_gs(experts, P, seed=seed + 2000)

                # Improvement metrics
                # Quality = merged norm (higher is better = more retained signal)
                sym_norm = sym['merged_norm']
                improvement_vs_canonical = (sym_norm - canonical_result['merged_norm']) / canonical_result['merged_norm'] * 100 if canonical_result['merged_norm'] > 1e-12 else 0
                improvement_vs_mean_single = (sym_norm - mean_norm) / mean_norm * 100 if mean_norm > 1e-12 else 0
                improvement_vs_best = (sym_norm - best_norm) / best_norm * 100 if best_norm > 1e-12 else 0

                # Cost ratio: P forward passes of GS
                cost_ratio = P

                results.append({
                    'cos_target': cos_target,
                    'cos_actual': actual_cos,
                    'seed': seed,
                    'P': P,
                    'cost_ratio': cost_ratio,
                    'sym_norm': sym_norm,
                    'canonical_norm': canonical_result['merged_norm'],
                    'best_single_norm': best_norm,
                    'worst_single_norm': worst_norm,
                    'mean_single_norm': mean_norm,
                    'std_single_norm': std_norm,
                    'improvement_vs_canonical_pct': improvement_vs_canonical,
                    'improvement_vs_mean_single_pct': improvement_vs_mean_single,
                    'improvement_vs_best_pct': improvement_vs_best,
                    'sym_mean_retention': sym['mean_retention'],
                    'sym_mean_min_retention': sym['mean_min_retention'],
                })

    # Print summary table (aggregate across seeds)
    print(f"\n  {'cos':>6} {'P':>4} {'sym_norm':>10} {'canon':>10} "
          f"{'best':>10} {'mean':>10} {'vs_canon%':>10} {'vs_best%':>10}")
    print(f"  {'-'*72}")

    for cos_target in cosine_values:
        for P in P_values:
            matching = [r for r in results
                       if r['cos_target'] == cos_target and r['P'] == P]
            if not matching:
                continue
            avg = lambda key: float(np.mean([r[key] for r in matching]))
            print(f"  {cos_target:>6.2f} {P:>4d} {avg('sym_norm'):>10.6f} "
                  f"{avg('canonical_norm'):>10.6f} "
                  f"{avg('best_single_norm'):>10.6f} "
                  f"{avg('mean_single_norm'):>10.6f} "
                  f"{avg('improvement_vs_canonical_pct'):>10.4f} "
                  f"{avg('improvement_vs_best_pct'):>10.4f}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 elapsed: {elapsed:.1f}s")
    return results


# ── Phase 2: High-Overlap Deep Dive ────────────────────────────────────────

def run_phase2():
    """Deep statistical analysis at cos=0.85 (attention-layer production value).

    Measures the full distribution of single-ordering quality and symmetric GS
    quality to determine if symmetric GS is statistically distinguishable from
    a good single ordering.
    """
    print(f"\n{'='*70}")
    print("PHASE 2: HIGH-OVERLAP DEEP DIVE (cos=0.85)")
    print(f"{'='*70}")
    t0 = time.time()

    COS_TARGET = 0.85
    N_SINGLE_SAMPLES = 500  # Large sample for distribution

    all_seed_results = []

    for seed in SEEDS:
        experts = create_synthetic_experts(N_EXPERTS, DIM, COS_TARGET, seed)
        actual_cos = measure_actual_cosine(experts)

        # Full distribution of single orderings
        rng = random.Random(seed + 3000)
        single_merged_vectors = []
        single_norms = []
        single_min_retentions = []

        for _ in range(N_SINGLE_SAMPLES):
            ordering = list(range(N_EXPERTS))
            rng.shuffle(ordering)
            result = gs_signal_retention(experts, ordering)
            single_merged_vectors.append(result['merged'])
            single_norms.append(result['merged_norm'])
            single_min_retentions.append(result['min_retention'])

        # Canonical ordering
        canonical = gs_signal_retention(experts, list(range(N_EXPERTS)))

        # Symmetric GS at increasing P
        P_values = [1, 2, 3, 5, 10, 20, 50, 100, 200]
        sym_results = {}
        for P in P_values:
            sym = symmetric_gs(experts, P, seed=seed + 4000)
            sym_results[P] = sym

        # Reference: "oracle" best ordering (best from our sample)
        best_idx = np.argmax(single_norms)
        best_norm = single_norms[best_idx]
        worst_norm = min(single_norms)

        # Key metrics
        # 1. Where does symmetric GS sit relative to the single-ordering distribution?
        sym_100 = sym_results[100]
        percentile = float(np.mean([1 if n <= sym_100['merged_norm'] else 0
                                     for n in single_norms]) * 100)

        # 2. How much does symmetric GS improve over mean single ordering?
        mean_single = float(np.mean(single_norms))
        std_single = float(np.std(single_norms))
        improvement_pct = (sym_100['merged_norm'] - mean_single) / mean_single * 100

        # 3. How close is symmetric GS to best single ordering?
        gap_to_best_pct = (best_norm - sym_100['merged_norm']) / best_norm * 100

        # 4. K1 check: symmetric vs best single
        k1_gap = abs(sym_100['merged_norm'] - best_norm) / best_norm * 100

        # 5. K2 check: symmetric vs random (mean single) improvement
        k2_improvement = improvement_pct

        print(f"\n  Seed {seed} (actual cos={actual_cos:.4f}):")
        print(f"    Single ordering distribution (N={N_SINGLE_SAMPLES}):")
        print(f"      Mean norm:  {mean_single:.6f}")
        print(f"      Std norm:   {std_single:.6f}")
        print(f"      Best norm:  {best_norm:.6f}")
        print(f"      Worst norm: {worst_norm:.6f}")
        print(f"      CV%:        {std_single/mean_single*100:.4f}")
        print(f"      Canonical:  {canonical['merged_norm']:.6f}")

        print(f"\n    Symmetric GS convergence:")
        for P in P_values:
            sr = sym_results[P]
            vs_mean = (sr['merged_norm'] - mean_single) / mean_single * 100
            vs_best = (sr['merged_norm'] - best_norm) / best_norm * 100
            print(f"      P={P:>3d}: norm={sr['merged_norm']:.6f} "
                  f"vs_mean={vs_mean:>+7.3f}% vs_best={vs_best:>+7.3f}%")

        print(f"\n    Kill criteria:")
        print(f"      K1 (sym vs best < 0.1%): gap={k1_gap:.4f}% "
              f"-> {'KILLED' if k1_gap < 0.1 else 'SURVIVES'}")
        print(f"      K2 (sym vs random < 1%): improvement={k2_improvement:.4f}% "
              f"-> {'KILLED' if abs(k2_improvement) < 1.0 else 'SURVIVES'}")
        print(f"      Percentile of sym in single dist: {percentile:.1f}%")

        all_seed_results.append({
            'seed': seed,
            'actual_cos': actual_cos,
            'mean_single_norm': mean_single,
            'std_single_norm': std_single,
            'best_single_norm': best_norm,
            'worst_single_norm': worst_norm,
            'canonical_norm': canonical['merged_norm'],
            'sym_100_norm': sym_100['merged_norm'],
            'percentile': percentile,
            'k1_gap_pct': k1_gap,
            'k2_improvement_pct': k2_improvement,
            'sym_convergence': {P: sym_results[P]['merged_norm'] for P in P_values},
            'cv_pct': std_single / mean_single * 100,
        })

    # Aggregate across seeds
    print(f"\n  AGGREGATE ({len(SEEDS)} seeds):")
    mean_k1 = float(np.mean([r['k1_gap_pct'] for r in all_seed_results]))
    mean_k2 = float(np.mean([r['k2_improvement_pct'] for r in all_seed_results]))
    mean_cv = float(np.mean([r['cv_pct'] for r in all_seed_results]))
    mean_percentile = float(np.mean([r['percentile'] for r in all_seed_results]))

    print(f"    Mean K1 gap (sym vs best): {mean_k1:.4f}%")
    print(f"    Mean K2 improvement (sym vs random): {mean_k2:.4f}%")
    print(f"    Mean single-ordering CV: {mean_cv:.4f}%")
    print(f"    Mean percentile of sym in single dist: {mean_percentile:.1f}%")

    k1_killed = mean_k1 < 0.1
    k2_killed = abs(mean_k2) < 1.0

    print(f"\n    K1 (< 0.1% gap to best): {'KILLED' if k1_killed else 'SURVIVES'}")
    print(f"    K2 (< 1% improvement over random): {'KILLED' if k2_killed else 'SURVIVES'}")

    elapsed = time.time() - t0
    print(f"\n  Phase 2 elapsed: {elapsed:.1f}s")
    return all_seed_results, mean_k1, mean_k2


# ── Phase 3: Convergence and Practical Alternatives ────────────────────────

def run_phase3():
    """How quickly does symmetric GS converge, and are there cheaper alternatives?

    Tests:
    1. Convergence rate of symmetric GS: plot quality vs P
    2. Comparison against practical alternatives:
       - Canonical ordering (sorted by index)
       - Reverse canonical ordering
       - Sorted by norm (largest first)
       - Sorted by norm (smallest first)
       - Random with fixed seed
    """
    print(f"\n{'='*70}")
    print("PHASE 3: CONVERGENCE AND PRACTICAL ALTERNATIVES")
    print(f"{'='*70}")
    t0 = time.time()

    cosine_values = [0.30, 0.50, 0.70, 0.85]
    results = []

    for cos_target in cosine_values:
        for seed in SEEDS:
            experts = create_synthetic_experts(N_EXPERTS, DIM, cos_target, seed)
            actual_cos = measure_actual_cosine(experts)
            norms = [np.linalg.norm(e) for e in experts]

            # Deterministic orderings (zero cost beyond single GS)
            orderings = {
                'canonical': list(range(N_EXPERTS)),
                'reverse': list(range(N_EXPERTS - 1, -1, -1)),
                'norm_desc': sorted(range(N_EXPERTS), key=lambda i: -norms[i]),
                'norm_asc': sorted(range(N_EXPERTS), key=lambda i: norms[i]),
                'random_fixed': None,  # Will use seeded random
            }

            # Random with fixed seed
            rng = random.Random(seed + 5000)
            fixed_random = list(range(N_EXPERTS))
            rng.shuffle(fixed_random)
            orderings['random_fixed'] = fixed_random

            deterministic_results = {}
            for name, ordering in orderings.items():
                result = gs_signal_retention(experts, ordering)
                deterministic_results[name] = {
                    'norm': result['merged_norm'],
                    'min_retention': result['min_retention'],
                    'mean_retention': result['mean_retention'],
                }

            # Symmetric GS at fine-grained P values
            P_values = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]
            sym_norms = {}
            for P in P_values:
                sym = symmetric_gs(experts, P, seed=seed + 6000)
                sym_norms[P] = sym['merged_norm']

            # Reference: mean of 200 random orderings
            rng2 = random.Random(seed + 7000)
            ref_norms = []
            for _ in range(200):
                ordering = list(range(N_EXPERTS))
                rng2.shuffle(ordering)
                result = gs_signal_retention(experts, ordering)
                ref_norms.append(result['merged_norm'])
            mean_random = float(np.mean(ref_norms))

            # How close is each deterministic ordering to symmetric P=100?
            sym_100 = sym_norms[100]
            comparisons = {}
            for name, det_result in deterministic_results.items():
                gap = abs(det_result['norm'] - sym_100) / sym_100 * 100
                comparisons[name] = gap

            results.append({
                'cos_target': cos_target,
                'cos_actual': actual_cos,
                'seed': seed,
                'deterministic': deterministic_results,
                'sym_norms': sym_norms,
                'mean_random_norm': mean_random,
                'sym_100_norm': sym_100,
                'gaps_vs_sym100': comparisons,
            })

    # Print summary
    print(f"\n  Deterministic ordering quality vs symmetric GS (P=100):")
    print(f"  {'cos':>6} {'canonical':>12} {'reverse':>12} {'norm_desc':>12} "
          f"{'norm_asc':>12} {'rand_fixed':>12} {'sym_100':>12}")
    print(f"  {'-'*78}")

    for cos_target in cosine_values:
        matching = [r for r in results if r['cos_target'] == cos_target]
        def avg_norm(key):
            return float(np.mean([r['deterministic'][key]['norm'] for r in matching]))
        avg_sym = float(np.mean([r['sym_100_norm'] for r in matching]))
        print(f"  {cos_target:>6.2f} {avg_norm('canonical'):>12.6f} "
              f"{avg_norm('reverse'):>12.6f} {avg_norm('norm_desc'):>12.6f} "
              f"{avg_norm('norm_asc'):>12.6f} {avg_norm('random_fixed'):>12.6f} "
              f"{avg_sym:>12.6f}")

    print(f"\n  Gap of deterministic orderings vs symmetric GS (P=100), % of sym norm:")
    print(f"  {'cos':>6} {'canonical':>12} {'reverse':>12} {'norm_desc':>12} "
          f"{'norm_asc':>12} {'rand_fixed':>12}")
    print(f"  {'-'*66}")

    for cos_target in cosine_values:
        matching = [r for r in results if r['cos_target'] == cos_target]
        def avg_gap(key):
            return float(np.mean([r['gaps_vs_sym100'][key] for r in matching]))
        print(f"  {cos_target:>6.2f} {avg_gap('canonical'):>12.4f} "
              f"{avg_gap('reverse'):>12.4f} {avg_gap('norm_desc'):>12.4f} "
              f"{avg_gap('norm_asc'):>12.4f} {avg_gap('random_fixed'):>12.4f}")

    # Convergence: P at which sym is within 0.01% of P=100
    print(f"\n  Convergence: P at which symmetric GS is within 0.01% of P=100")
    for cos_target in cosine_values:
        matching = [r for r in results if r['cos_target'] == cos_target]
        P_values_sorted = sorted(matching[0]['sym_norms'].keys())
        converged_P = None
        for P in P_values_sorted:
            avg_norm_P = float(np.mean([r['sym_norms'][P] for r in matching]))
            avg_norm_100 = float(np.mean([r['sym_100_norm'] for r in matching]))
            gap = abs(avg_norm_P - avg_norm_100) / avg_norm_100 * 100
            if gap < 0.01 and converged_P is None:
                converged_P = P
        print(f"    cos={cos_target:.2f}: converges at P={converged_P}"
              f" (cost ratio: {converged_P}x)" if converged_P else
              f"    cos={cos_target:.2f}: does not converge within P=100")

    elapsed = time.time() - t0
    print(f"\n  Phase 3 elapsed: {elapsed:.1f}s")
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def run_full_experiment():
    """Run all three phases and aggregate results."""
    print("=" * 70)
    print("SYMMETRIC GS COST-BENEFIT ANALYSIS")
    print(f"N_EXPERTS={N_EXPERTS}, DIM={DIM}, SEEDS={SEEDS}")
    print("=" * 70)

    t0 = time.time()

    phase1_results = run_phase1()
    phase2_results, mean_k1, mean_k2 = run_phase2()
    phase3_results = run_phase3()

    # ── Final Verdict ──
    print(f"\n\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    # K1: symmetric GS quality within 0.1% of best single ordering
    k1_killed = mean_k1 < 0.1
    print(f"\n  K1 (sym within 0.1% of best single at cos=0.85):")
    print(f"    Gap to best: {mean_k1:.4f}%")
    print(f"    Threshold: 0.1%")
    print(f"    Verdict: {'KILLED -- no benefit worth 50x cost' if k1_killed else 'SURVIVES -- meaningful improvement'}")

    # K2: symmetric GS at cos=0.85 shows <1% improvement over random
    k2_killed = abs(mean_k2) < 1.0
    print(f"\n  K2 (sym vs random < 1% improvement at cos=0.85):")
    print(f"    Improvement over random: {mean_k2:.4f}%")
    print(f"    Threshold: 1%")
    print(f"    Verdict: {'KILLED -- negligible improvement' if k2_killed else 'SURVIVES -- meaningful improvement'}")

    overall = "KILLED" if (k1_killed and k2_killed) else ("KILLED" if k1_killed or k2_killed else "PROVEN")
    print(f"\n  OVERALL: {overall}")

    if k1_killed:
        print(f"\n  Symmetric GS is within {mean_k1:.4f}% of the best single ordering.")
        print(f"  The 50x-100x compute cost is not justified.")
        print(f"  Recommendation: use canonical (sorted) or random-fixed ordering.")

    total_elapsed = time.time() - t0
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    # Save results
    output = {
        'config': {
            'N_EXPERTS': N_EXPERTS,
            'DIM': DIM,
            'SEEDS': SEEDS,
            'N_SINGLE_ORDERINGS': N_SINGLE_ORDERINGS,
        },
        'phase2_aggregate': {
            'mean_k1_gap_pct': mean_k1,
            'mean_k2_improvement_pct': mean_k2,
        },
        'phase2_per_seed': phase2_results,
        'phase3_summary': [],
        'kill_criteria': {
            'K1_killed': k1_killed,
            'K1_value': mean_k1,
            'K1_threshold': 0.1,
            'K2_killed': k2_killed,
            'K2_value': mean_k2,
            'K2_threshold': 1.0,
        },
        'overall': overall,
        'elapsed_seconds': total_elapsed,
    }

    # Add phase 3 summary
    for r in phase3_results:
        output['phase3_summary'].append({
            'cos_target': r['cos_target'],
            'cos_actual': r['cos_actual'],
            'seed': r['seed'],
            'canonical_norm': r['deterministic']['canonical']['norm'],
            'sym_100_norm': r['sym_100_norm'],
            'mean_random_norm': r['mean_random_norm'],
            'gaps_vs_sym100': r['gaps_vs_sym100'],
        })

    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Results saved to {output_path}")

    return output


if __name__ == '__main__':
    run_full_experiment()
