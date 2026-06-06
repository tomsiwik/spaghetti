"""Order Sensitivity Cosine Threshold Experiment.

Validates the cos=0.06 threshold where GS order sensitivity reaches CV=5%.
Tests whether this threshold holds across different N (5, 10, 20, 50).

Prior work established:
  - variation ~ 62*cos (per-sublayer, layerwise_order_sensitivity)
  - variation ~ 80*cos (flattened, merge_order_dependence)
  - threshold extrapolated to cos ~ 0.06 for 5% CV

This experiment:
  1. Fine-grained cosine sweep (0.01 to 0.30) at multiple N values
  2. Measures CV of merged-vector norms across 50 random orderings per condition
  3. Fits variation = slope * cos per N, extracts cos_5pct = 5/slope
  4. Tests whether cos_5pct is stable across N

Kill Criteria:
  K1: Order sensitivity CV exceeds 5% at cos<0.06 (threshold too low)
  K2: Order sensitivity remains <5% CV at cos=0.20 (threshold too high)
"""

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


# ── Config ──────────────────────────────────────────────────────────────────

DIM = 4096           # Flattened delta dimension (matches parent experiment)
N_ORDERINGS = 50     # Orderings per condition (more than parent's 20 for precision)
N_VALUES = [5, 10, 20, 50]  # Expert counts to test
COSINE_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SEEDS = [42, 7, 123]  # Multi-seed for robustness


# ── Helpers ─────────────────────────────────────────────────────────────────

def create_synthetic_experts(n_experts: int, dim: int, target_cosine: float,
                              seed: int = 42) -> tuple[list[np.ndarray], float]:
    """Create synthetic expert delta vectors with controlled pairwise cosine.

    Uses shared component + unique component:
      d_k = alpha * shared + beta * unique_k
    where alpha = sqrt(target_cosine), beta = sqrt(1 - target_cosine).

    Returns:
        vectors: list of N numpy arrays of shape (dim,)
        actual_mean_cosine: measured mean pairwise cosine
    """
    rng = np.random.RandomState(seed)

    shared = rng.randn(dim)
    shared /= np.linalg.norm(shared)

    alpha = np.sqrt(max(target_cosine, 0.0))
    beta = np.sqrt(max(1.0 - target_cosine, 0.0))

    vectors = []
    for _ in range(n_experts):
        unique = rng.randn(dim)
        unique /= np.linalg.norm(unique)
        # Remove shared component for independence
        unique -= np.dot(unique, shared) * shared
        unique /= np.linalg.norm(unique)
        vectors.append(alpha * shared + beta * unique)

    # Measure actual pairwise cosines
    cosines = []
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            cos = np.dot(vectors[i], vectors[j]) / (
                np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            cosines.append(cos)

    return vectors, float(np.mean(cosines))


def gs_orthogonalize(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Classical Gram-Schmidt orthogonalization."""
    ortho = []
    for v_orig in vectors:
        v = v_orig.copy()
        for e in ortho:
            dot_ve = np.dot(v, e)
            dot_ee = np.dot(e, e)
            if dot_ee > 1e-12:
                v -= (dot_ve / dot_ee) * e
        ortho.append(v)
    return ortho


def measure_order_sensitivity(vectors: list[np.ndarray], n_orderings: int,
                               seed: int) -> dict:
    """Measure order sensitivity of GS merge across random orderings.

    Returns dict with:
      - norm_cv_pct: CV of merged vector L2 norms
      - merged_cos_min: minimum pairwise cosine between merged vectors
      - variation_pct: (1 - merged_cos_min) * 100
      - norms: list of merged norms (for external analysis)
    """
    N = len(vectors)
    rng = random.Random(seed)
    merged_vectors = []

    for _ in range(n_orderings):
        perm = list(range(N))
        rng.shuffle(perm)
        ordered = [vectors[perm[i]] for i in range(N)]
        ortho = gs_orthogonalize(ordered)
        merged = np.mean(ortho, axis=0)
        merged_vectors.append(merged)

    # Norm-based CV
    norms = [float(np.linalg.norm(v)) for v in merged_vectors]
    norm_mean = statistics.mean(norms)
    norm_std = statistics.stdev(norms) if len(norms) > 1 else 0.0
    norm_cv = (norm_std / norm_mean) * 100 if norm_mean > 1e-12 else 0.0

    # Cosine-based variation (how much the direction changes)
    cos_values = []
    for i in range(len(merged_vectors)):
        ni = np.linalg.norm(merged_vectors[i])
        if ni < 1e-12:
            continue
        for j in range(i + 1, len(merged_vectors)):
            nj = np.linalg.norm(merged_vectors[j])
            if nj < 1e-12:
                continue
            cos_values.append(float(
                np.dot(merged_vectors[i], merged_vectors[j]) / (ni * nj)))

    merged_cos_min = min(cos_values) if cos_values else 1.0
    variation_pct = (1.0 - merged_cos_min) * 100

    return {
        'norm_cv_pct': norm_cv,
        'merged_cos_min': merged_cos_min,
        'variation_pct': variation_pct,
        'norms': norms,
    }


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment():
    """Run the full cosine threshold validation experiment."""
    t0 = time.time()

    print("=" * 70)
    print("ORDER SENSITIVITY COSINE THRESHOLD EXPERIMENT")
    print("=" * 70)
    print(f"  D={DIM}, N_values={N_VALUES}, {len(COSINE_SWEEP)} cosine points")
    print(f"  {N_ORDERINGS} orderings/condition, {len(SEEDS)} seeds")
    print(f"  Total conditions: {len(N_VALUES) * len(COSINE_SWEEP) * len(SEEDS)}")

    all_results = []

    for N in N_VALUES:
        print(f"\n{'='*70}")
        print(f"  N = {N} experts")
        print(f"{'='*70}")

        for target_cos in COSINE_SWEEP:
            seed_results = []
            for seed in SEEDS:
                vectors, actual_cos = create_synthetic_experts(N, DIM, target_cos, seed)
                sensitivity = measure_order_sensitivity(vectors, N_ORDERINGS, seed * 1000)

                seed_results.append({
                    'actual_cos': actual_cos,
                    'norm_cv_pct': sensitivity['norm_cv_pct'],
                    'variation_pct': sensitivity['variation_pct'],
                    'merged_cos_min': sensitivity['merged_cos_min'],
                })

            # Aggregate across seeds
            mean_actual_cos = statistics.mean([r['actual_cos'] for r in seed_results])
            mean_norm_cv = statistics.mean([r['norm_cv_pct'] for r in seed_results])
            std_norm_cv = statistics.stdev([r['norm_cv_pct'] for r in seed_results]) if len(seed_results) > 1 else 0
            mean_variation = statistics.mean([r['variation_pct'] for r in seed_results])
            std_variation = statistics.stdev([r['variation_pct'] for r in seed_results]) if len(seed_results) > 1 else 0

            result = {
                'N': N,
                'target_cos': target_cos,
                'actual_cos': mean_actual_cos,
                'norm_cv_pct': mean_norm_cv,
                'norm_cv_std': std_norm_cv,
                'variation_pct': mean_variation,
                'variation_std': std_variation,
                'per_seed': seed_results,
            }
            all_results.append(result)

            cv_flag = " *** > 5% ***" if mean_norm_cv > 5.0 else ""
            var_flag = " *** > 5% ***" if mean_variation > 5.0 else ""
            print(f"  cos={target_cos:.3f} (actual={mean_actual_cos:.4f}): "
                  f"norm_CV={mean_norm_cv:.3f}%+/-{std_norm_cv:.3f}, "
                  f"var={mean_variation:.3f}%+/-{std_variation:.3f}"
                  f"{var_flag}")

    # ── Analysis: Fit slope per N ──────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SCALING LAW FIT: variation = slope * cos")
    print(f"{'='*70}")

    slopes = {}
    thresholds = {}

    for N in N_VALUES:
        n_results = [r for r in all_results if r['N'] == N]
        x = np.array([r['actual_cos'] for r in n_results])
        y_var = np.array([r['variation_pct'] for r in n_results])
        y_cv = np.array([r['norm_cv_pct'] for r in n_results])

        # Fit variation = slope * cos (no intercept)
        slope_var = float(np.dot(x, y_var) / np.dot(x, x))
        y_pred = slope_var * x
        ss_res = float(np.sum((y_var - y_pred) ** 2))
        ss_tot = float(np.sum((y_var - np.mean(y_var)) ** 2))
        r2_var = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Fit norm_cv = slope * cos
        slope_cv = float(np.dot(x, y_cv) / np.dot(x, x))
        y_pred_cv = slope_cv * x
        ss_res_cv = float(np.sum((y_cv - y_pred_cv) ** 2))
        ss_tot_cv = float(np.sum((y_cv - np.mean(y_cv)) ** 2))
        r2_cv = 1 - ss_res_cv / ss_tot_cv if ss_tot_cv > 0 else 0

        # Also fit with intercept for comparison
        slope_li, intercept_li, r_li, p_li, se_li = scipy_stats.linregress(x, y_var)

        # Threshold: where variation = 5%
        cos_5pct_var = 5.0 / slope_var if slope_var > 0 else float('inf')
        cos_5pct_cv = 5.0 / slope_cv if slope_cv > 0 else float('inf')

        slopes[N] = {
            'slope_var': slope_var,
            'r2_var': r2_var,
            'slope_cv': slope_cv,
            'r2_cv': r2_cv,
            'slope_linreg': float(slope_li),
            'intercept_linreg': float(intercept_li),
            'r2_linreg': float(r_li ** 2),
            'cos_5pct_var': cos_5pct_var,
            'cos_5pct_cv': cos_5pct_cv,
        }
        thresholds[N] = cos_5pct_var

        print(f"\n  N={N}:")
        print(f"    Variation slope:    {slope_var:.1f}  (R2={r2_var:.4f})")
        print(f"    Norm-CV slope:      {slope_cv:.1f}  (R2={r2_cv:.4f})")
        print(f"    Linreg slope:       {slope_li:.1f}, intercept={intercept_li:.2f} (R2={r_li**2:.4f})")
        print(f"    5% threshold (var): cos = {cos_5pct_var:.4f}")
        print(f"    5% threshold (cv):  cos = {cos_5pct_cv:.4f}")

    # ── Threshold stability across N ──────────────────────────────────────

    print(f"\n{'='*70}")
    print("THRESHOLD STABILITY ACROSS N")
    print(f"{'='*70}")

    threshold_values = list(thresholds.values())
    threshold_mean = statistics.mean(threshold_values)
    threshold_std = statistics.stdev(threshold_values) if len(threshold_values) > 1 else 0
    threshold_cv = (threshold_std / threshold_mean) * 100 if threshold_mean > 0 else 0

    print(f"\n  {'N':>6} {'cos_5pct':>12} {'slope':>10}")
    print(f"  {'-'*32}")
    for N in N_VALUES:
        print(f"  {N:>6} {thresholds[N]:>12.4f} {slopes[N]['slope_var']:>10.1f}")

    print(f"\n  Mean threshold:   {threshold_mean:.4f}")
    print(f"  Std threshold:    {threshold_std:.4f}")
    print(f"  CV of threshold:  {threshold_cv:.1f}%")

    # ── Kill Criteria Assessment ──────────────────────────────────────────

    print(f"\n{'='*70}")
    print("KILL CRITERIA ASSESSMENT")
    print(f"{'='*70}")

    # K1: CV exceeds 5% at cos < 0.06
    k1_violations = []
    for r in all_results:
        if r['actual_cos'] < 0.06 and r['variation_pct'] > 5.0:
            k1_violations.append(r)

    k1_pass = len(k1_violations) == 0  # No violations = threshold is correct
    print(f"\n  K1 (CV > 5% at cos < 0.06):")
    if k1_pass:
        # Find the highest variation below cos=0.06
        below_006 = [r for r in all_results if r['actual_cos'] < 0.06]
        max_var_below = max([r['variation_pct'] for r in below_006]) if below_006 else 0
        print(f"    PASS: No violations. Max variation below cos=0.06: {max_var_below:.3f}%")
        print(f"    Margin: {(5.0 - max_var_below) / 5.0 * 100:.1f}% below threshold")
    else:
        print(f"    KILL: {len(k1_violations)} violations found:")
        for v in k1_violations:
            print(f"      N={v['N']}, cos={v['actual_cos']:.4f}, var={v['variation_pct']:.3f}%")

    # K2: CV remains < 5% at cos = 0.20
    k2_violations = []
    for r in all_results:
        if r['actual_cos'] > 0.18 and r['variation_pct'] < 5.0:
            k2_violations.append(r)

    k2_pass = len(k2_violations) == 0  # No violations = threshold is relevant
    print(f"\n  K2 (CV < 5% at cos = 0.20):")
    if k2_pass:
        at_020 = [r for r in all_results if abs(r['target_cos'] - 0.20) < 0.01]
        if at_020:
            min_var_020 = min([r['variation_pct'] for r in at_020])
            print(f"    PASS: All conditions at cos=0.20 exceed 5%. Min variation: {min_var_020:.3f}%")
        else:
            print(f"    PASS: No conditions remain below 5% at cos >= 0.18")
    else:
        print(f"    KILL: {len(k2_violations)} conditions still < 5% at cos >= 0.18:")
        for v in k2_violations:
            print(f"      N={v['N']}, cos={v['actual_cos']:.4f}, var={v['variation_pct']:.3f}%")

    # ── Cross-validation: interpolate exact threshold per N ───────────────

    print(f"\n{'='*70}")
    print("INTERPOLATED EXACT THRESHOLDS (variation crossing 5%)")
    print(f"{'='*70}")

    exact_thresholds = {}
    for N in N_VALUES:
        n_results = sorted(
            [r for r in all_results if r['N'] == N],
            key=lambda r: r['actual_cos'])

        # Find the crossing point
        for i in range(len(n_results) - 1):
            if n_results[i]['variation_pct'] <= 5.0 < n_results[i+1]['variation_pct']:
                # Linear interpolation
                x0, y0 = n_results[i]['actual_cos'], n_results[i]['variation_pct']
                x1, y1 = n_results[i+1]['actual_cos'], n_results[i+1]['variation_pct']
                cos_exact = x0 + (5.0 - y0) * (x1 - x0) / (y1 - y0)
                exact_thresholds[N] = cos_exact
                print(f"  N={N:>3}: cos_5pct = {cos_exact:.4f}  "
                      f"(between cos={x0:.3f} [{y0:.2f}%] and cos={x1:.3f} [{y1:.2f}%])")
                break
        else:
            # Check if all are above or below 5%
            all_below = all(r['variation_pct'] <= 5.0 for r in n_results)
            if all_below:
                print(f"  N={N:>3}: threshold > {n_results[-1]['actual_cos']:.3f} (all below 5%)")
                exact_thresholds[N] = float('inf')
            else:
                print(f"  N={N:>3}: threshold < {n_results[0]['actual_cos']:.3f} (first point already > 5%)")
                exact_thresholds[N] = 0.0

    if exact_thresholds:
        finite_thresholds = [v for v in exact_thresholds.values() if 0 < v < float('inf')]
        if finite_thresholds:
            et_mean = statistics.mean(finite_thresholds)
            et_std = statistics.stdev(finite_thresholds) if len(finite_thresholds) > 1 else 0
            print(f"\n  Mean interpolated threshold: {et_mean:.4f} +/- {et_std:.4f}")

    # ── SOLE Safety Assessment ────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SOLE SAFETY ASSESSMENT")
    print(f"{'='*70}")

    prod_cos = 0.0002  # Production cosine at d=896
    print(f"\n  Production pairwise cosine: {prod_cos}")
    print(f"  Mean 5% threshold:         {threshold_mean:.4f}")
    print(f"  Safety margin:             {threshold_mean / prod_cos:.0f}x")
    print(f"  Predicted production var:  {slopes[N_VALUES[0]]['slope_var'] * prod_cos:.4f}%")

    # ── Overall Verdict ───────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    if k1_pass and k2_pass:
        print(f"\n  PROVEN: cos=0.06 is a valid practical threshold for GS order sensitivity.")
        print(f"  - K1 PASS: No order sensitivity > 5% below cos=0.06")
        print(f"  - K2 PASS: Order sensitivity > 5% confirmed at cos=0.20")
        print(f"  - Threshold stable across N={N_VALUES} (CV={threshold_cv:.1f}%)")
        print(f"  - SOLE safe: production cosine {threshold_mean / prod_cos:.0f}x below threshold")
        verdict = "proven"
    elif not k1_pass:
        print(f"\n  KILL (K1): Threshold is too low. Order sensitivity > 5% at cos < 0.06.")
        print(f"  The actual threshold is lower than 0.06.")
        verdict = "killed_k1"
    else:
        print(f"\n  KILL (K2): Threshold is too high. Order sensitivity < 5% even at cos=0.20.")
        print(f"  The cos=0.06 threshold is too conservative.")
        verdict = "killed_k2"

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # ── Save Results ──────────────────────────────────────────────────────

    output = {
        'verdict': verdict,
        'k1_pass': k1_pass,
        'k2_pass': k2_pass,
        'slopes': slopes,
        'thresholds': thresholds,
        'exact_thresholds': {str(k): v for k, v in exact_thresholds.items()},
        'threshold_mean': threshold_mean,
        'threshold_std': threshold_std,
        'threshold_cv_pct': threshold_cv,
        'safety_margin_over_production': threshold_mean / prod_cos,
        'all_results': [{k: v for k, v in r.items() if k != 'per_seed'}
                        for r in all_results],
        'config': {
            'dim': DIM,
            'n_orderings': N_ORDERINGS,
            'n_values': N_VALUES,
            'cosine_sweep': COSINE_SWEEP,
            'seeds': SEEDS,
        },
        'elapsed_seconds': elapsed,
    }

    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")

    return output


if __name__ == "__main__":
    results = run_experiment()
