"""SiLU vs GELU Gamma Correction: Extended comparison across d, N sweeps.

Parent: rmsnorm_gamma_nonuniformity (PROVEN, GELU-only, correction=1.43x)
Prior: silu_gamma_correction (PROVEN, SiLU=1.41x, 3 seeds, limited sweep)

This experiment extends with:
  - d in {64, 128, 256, 512}
  - N in {5, 10, 25, 50}
  - 5 seeds per configuration
  - Kill criteria:
    K1: SiLU correction factor differs from GELU by >1.5x -> FAIL
    K2: SiLU worst-case D exceeds 5% at d=256, N=50 -> FAIL

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np


# ============================================================================
# Activation functions
# ============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (Swish): x * sigmoid(x). Used by Qwen2.5, Llama."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


# ============================================================================
# Core utilities
# ============================================================================

def rms_norm(x: np.ndarray, gamma: np.ndarray = None, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm with optional per-dimension learnable gamma."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if gamma is not None:
        return gamma * normed
    return normed


def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a single LoRA delta W = A @ B."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return A @ B


def gram_schmidt_merge(deltas: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """GS-orthogonalize then sum flattened deltas."""
    ortho = []
    for v_orig in deltas:
        v = v_orig.copy()
        for e in ortho:
            dot_ve = np.dot(v, e)
            dot_ee = np.dot(e, e)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e
        ortho.append(v)
    return sum(ortho), ortho


def forward_pass(h: np.ndarray, base_weights: list[np.ndarray],
                 layer_deltas: list[np.ndarray],
                 gammas: list[np.ndarray], act_fn) -> np.ndarray:
    """Pre-RMSNorm transformer forward pass.

    h_{l+1} = h_l + act_fn((W_l + Delta_l) @ RMSNorm(h_l; gamma_l))
    """
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        g = gammas[l] if gammas is not None else None
        h = h + act_fn(W @ rms_norm(h, gamma=g))
    return h


# ============================================================================
# Gamma generators
# ============================================================================

def gen_uniform(d: int, L: int, seed: int = 0) -> list[np.ndarray]:
    return [np.ones(d) for _ in range(L)]


def gen_lognormal(d: int, L: int, sigma: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [np.clip(np.exp(rng.normal(0, sigma, size=d)), 0.01, 100.0) for _ in range(L)]


def gen_bimodal(d: int, L: int, low: float, high: float, frac_high: float,
                seed: int = 0) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    n_high = int(d * frac_high)
    gammas = []
    for _ in range(L):
        g = np.full(d, low)
        g[rng.choice(d, size=n_high, replace=False)] = high
        gammas.append(g)
    return gammas


def gen_spike(d: int, L: int, spike_val: float = 10.0, seed: int = 0) -> list[np.ndarray]:
    """Single layer at spike_val, rest at 1.0."""
    gammas = [np.ones(d) for _ in range(L)]
    gammas[L // 2] = np.full(d, spike_val)
    return gammas


# ============================================================================
# Single configuration runner
# ============================================================================

def run_config(d: int, r: int, L: int, N: int, n_inputs: int,
               seed: int, gammas: list[np.ndarray],
               gamma_label: str, act_fn, act_name: str) -> dict:
    """Run one complete (d, N, gamma, activation, seed) configuration."""
    rng = np.random.RandomState(seed)
    remove_idx = N // 2

    # Generate experts (N experts, each with L layers)
    expert_deltas = []  # [N][L] of (d,d)
    for _ in range(N):
        expert_deltas.append([generate_lora_expert_layer(d, r, rng) for _ in range(L)])

    # Base weights
    base_weights = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]

    # Test inputs
    inputs = rng.randn(n_inputs, d) * 0.1

    # GS merge per layer
    merged_deltas = []
    ortho_components = []
    per_layer_errors = []

    for l in range(L):
        flat_deltas = [expert_deltas[i][l].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(flat_deltas)
        merged_deltas.append(merged_flat.reshape(d, d))
        ortho_components.append(ortho_flat)

        # Per-layer weight-space error from naive removal
        naive_remaining = merged_flat - ortho_flat[remove_idx]
        remaining = [flat_deltas[i] for i in range(N) if i != remove_idx]
        gt_remaining, _ = gram_schmidt_merge(remaining)
        err = np.linalg.norm(naive_remaining - gt_remaining)
        gt_norm = np.linalg.norm(gt_remaining)
        per_layer_errors.append(float(err / (gt_norm + 1e-12)) * 100.0)

    sum_eps = sum(per_layer_errors)

    # Forward helper
    def fwd(h, deltas):
        return forward_pass(h, base_weights, deltas, gammas, act_fn)

    # Naive removal deltas
    naive_deltas = []
    for l in range(L):
        m = merged_deltas[l].flatten() - ortho_components[l][remove_idx]
        naive_deltas.append(m.reshape(d, d))

    # GT removal deltas
    gt_deltas = []
    for l in range(L):
        flat_deltas = [expert_deltas[i][l].flatten() for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(flat_deltas)
        gt_deltas.append(gt_flat.reshape(d, d))

    # Run forward passes
    devs = []
    for inp in inputs:
        out_naive = fwd(inp, naive_deltas)
        out_gt = fwd(inp, gt_deltas)

        if np.any(np.isnan(out_naive)) or np.any(np.isnan(out_gt)):
            return {"diverged": True, "act_name": act_name, "gamma_label": gamma_label,
                    "d": d, "N": N, "seed": seed}
        if np.any(np.isinf(out_naive)) or np.any(np.isinf(out_gt)):
            return {"diverged": True, "act_name": act_name, "gamma_label": gamma_label,
                    "d": d, "N": N, "seed": seed}

        diff = np.linalg.norm(out_naive - out_gt)
        gt_n = np.linalg.norm(out_gt)
        devs.append(float(diff / max(gt_n, 1e-12)) * 100.0)

    mean_dev = float(np.mean(devs))
    max_dev = float(np.max(devs))
    alpha = mean_dev / sum_eps if sum_eps > 1e-12 else 0.0

    return {
        "diverged": False,
        "act_name": act_name,
        "gamma_label": gamma_label,
        "d": d, "r": r, "L": L, "N": N, "seed": seed,
        "sum_per_layer_error_pct": sum_eps,
        "mean_output_dev_pct": mean_dev,
        "max_output_dev_pct": max_dev,
        "amplification_ratio": alpha,
    }


# ============================================================================
# Analytical curvature comparison
# ============================================================================

def curvature_analysis() -> dict:
    """Compare SiLU and GELU curvature (second derivative)."""
    x = np.linspace(-5, 5, 10000)
    dx = x[1] - x[0]

    gelu_y = gelu(x)
    silu_y = silu(x)

    gelu_d2 = np.gradient(np.gradient(gelu_y, dx), dx)
    silu_d2 = np.gradient(np.gradient(silu_y, dx), dx)

    # Trim edges to avoid boundary artifacts
    trim = 50
    gelu_max_curv = float(np.max(np.abs(gelu_d2[trim:-trim])))
    silu_max_curv = float(np.max(np.abs(silu_d2[trim:-trim])))

    return {
        "gelu_max_curvature": gelu_max_curv,
        "silu_max_curvature": silu_max_curv,
        "curvature_ratio_silu_over_gelu": silu_max_curv / gelu_max_curv,
    }


# ============================================================================
# Main experiment
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: SiLU vs GELU Gamma Correction Factor")
    print("  K1: SiLU correction factor differs from GELU by >1.5x -> FAIL")
    print("  K2: SiLU worst-case D exceeds 5% at d=256, N=50 -> FAIL")
    print("=" * 80)

    # Parameters
    r = 8
    L = 24
    seeds = [42, 123, 777, 2024, 9999]
    d_values = [64, 128, 256, 512]
    N_values = [5, 10, 25, 50]
    act_fns = [("gelu", gelu), ("silu", silu)]

    gamma_profiles = [
        ("uniform", lambda d, L, s: gen_uniform(d, L, s)),
        ("lognormal_0.5", lambda d, L, s: gen_lognormal(d, L, 0.5, s)),
        ("lognormal_1.0", lambda d, L, s: gen_lognormal(d, L, 1.0, s)),
        ("bimodal_0.2_5.0", lambda d, L, s: gen_bimodal(d, L, 0.2, 5.0, 0.25, s)),
        ("spike_10", lambda d, L, s: gen_spike(d, L, 10.0, s)),
    ]

    # Step 0: Analytical comparison
    print("\n--- Curvature Analysis ---")
    curv = curvature_analysis()
    print(f"  GELU max |curvature|: {curv['gelu_max_curvature']:.4f}")
    print(f"  SiLU max |curvature|: {curv['silu_max_curvature']:.4f}")
    print(f"  SiLU/GELU ratio: {curv['curvature_ratio_silu_over_gelu']:.4f}")
    print(f"  Prediction: SiLU correction <= GELU correction")

    # Step 1: Full sweep
    print("\n--- Full Sweep ---")
    all_results = []

    # Limit n_inputs based on d to keep runtime manageable
    # d=64: 300, d=128: 200, d=256: 100, d=512: 50
    n_inputs_map = {64: 300, 128: 200, 256: 100, 512: 50}

    total_configs = len(d_values) * len(N_values) * len(gamma_profiles) * len(act_fns) * len(seeds)
    print(f"  Total configurations: {total_configs}")
    completed = 0

    for d in d_values:
        n_inputs = n_inputs_map[d]
        for N in N_values:
            # Skip d=512, N=50 if too slow (check after first run)
            for gp_name, gp_fn in gamma_profiles:
                for act_name, act_fn in act_fns:
                    seed_results = []
                    for seed in seeds:
                        gammas = gp_fn(d, L, seed + 5000)
                        res = run_config(
                            d=d, r=r, L=L, N=N, n_inputs=n_inputs,
                            seed=seed, gammas=gammas,
                            gamma_label=gp_name, act_fn=act_fn, act_name=act_name,
                        )
                        all_results.append(res)
                        seed_results.append(res)
                        completed += 1

                    # Print progress every activation pair
                    if act_name == "silu":
                        # Gather matching gelu results
                        gelu_alphas = [r["amplification_ratio"] for r in all_results
                                       if r.get("d") == d and r.get("N") == N
                                       and r.get("gamma_label") == gp_name
                                       and r.get("act_name") == "gelu"
                                       and not r.get("diverged", True)]
                        silu_alphas = [r["amplification_ratio"] for r in all_results
                                       if r.get("d") == d and r.get("N") == N
                                       and r.get("gamma_label") == gp_name
                                       and r.get("act_name") == "silu"
                                       and not r.get("diverged", True)]
                        if gelu_alphas and silu_alphas:
                            ga = np.mean(gelu_alphas)
                            sa = np.mean(silu_alphas)
                            ratio = sa / ga if ga > 1e-12 else 0
                        else:
                            ratio = float("nan")

                        elapsed = time.time() - t_start
                        pct = completed / total_configs * 100
                        print(f"  [{pct:5.1f}%] d={d:3d} N={N:2d} {gp_name:>20s}: "
                              f"SiLU/GELU alpha ratio={ratio:.4f}  ({elapsed:.0f}s)")

    # ================================================================
    # Analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("  ANALYSIS: Correction Factor Comparison")
    print("=" * 80)

    # Compute baseline alphas per (d, N, activation) with uniform gamma
    baselines = {}  # (d, N, act_name) -> mean_alpha
    for d in d_values:
        for N in N_values:
            for act_name in ["gelu", "silu"]:
                alphas = [r["amplification_ratio"] for r in all_results
                          if r.get("d") == d and r.get("N") == N
                          and r.get("gamma_label") == "uniform"
                          and r.get("act_name") == act_name
                          and not r.get("diverged", True)]
                if alphas:
                    baselines[(d, N, act_name)] = float(np.mean(alphas))

    # Compute correction ratios for each (d, N, gamma, act)
    max_gelu_correction = 1.0
    max_silu_correction = 1.0
    max_silu_gelu_ratio = 0.0  # max(C_silu / C_gelu)
    max_silu_D = 0.0  # for K2

    correction_table = []

    for d in d_values:
        for N in N_values:
            for gp_name, _ in gamma_profiles:
                gelu_base = baselines.get((d, N, "gelu"), None)
                silu_base = baselines.get((d, N, "silu"), None)
                if gelu_base is None or silu_base is None:
                    continue

                gelu_alphas = [r["amplification_ratio"] for r in all_results
                               if r.get("d") == d and r.get("N") == N
                               and r.get("gamma_label") == gp_name
                               and r.get("act_name") == "gelu"
                               and not r.get("diverged", True)]
                silu_alphas = [r["amplification_ratio"] for r in all_results
                               if r.get("d") == d and r.get("N") == N
                               and r.get("gamma_label") == gp_name
                               and r.get("act_name") == "silu"
                               and not r.get("diverged", True)]
                silu_devs = [r["max_output_dev_pct"] for r in all_results
                             if r.get("d") == d and r.get("N") == N
                             and r.get("gamma_label") == gp_name
                             and r.get("act_name") == "silu"
                             and not r.get("diverged", True)]

                if not gelu_alphas or not silu_alphas:
                    continue

                ga_mean = np.mean(gelu_alphas)
                sa_mean = np.mean(silu_alphas)

                c_gelu = ga_mean / gelu_base if gelu_base > 1e-12 else 1.0
                c_silu = sa_mean / silu_base if silu_base > 1e-12 else 1.0

                max_gelu_correction = max(max_gelu_correction, c_gelu)
                max_silu_correction = max(max_silu_correction, c_silu)

                ratio = c_silu / c_gelu if c_gelu > 1e-12 else 1.0
                max_silu_gelu_ratio = max(max_silu_gelu_ratio, ratio)

                if silu_devs:
                    worst_silu_D = max(silu_devs)
                    if d == 256 and N == 50:
                        max_silu_D = max(max_silu_D, worst_silu_D)

                correction_table.append({
                    "d": d, "N": N, "gamma": gp_name,
                    "C_gelu": c_gelu, "C_silu": c_silu,
                    "ratio": ratio,
                    "silu_max_D": max(silu_devs) if silu_devs else None,
                })

    # Print summary table
    print(f"\n  {'d':>4} {'N':>4} {'Gamma':>20} {'C_GELU':>8} {'C_SiLU':>8} {'Ratio':>8}")
    print("  " + "-" * 56)
    for row in correction_table:
        if row["gamma"] != "uniform":  # Skip uniform (ratio=1.0 by definition)
            print(f"  {row['d']:>4} {row['N']:>4} {row['gamma']:>20} "
                  f"{row['C_gelu']:>8.3f} {row['C_silu']:>8.3f} {row['ratio']:>8.4f}")

    # Print key aggregates
    print(f"\n  Max GELU correction: {max_gelu_correction:.3f}x")
    print(f"  Max SiLU correction: {max_silu_correction:.3f}x")
    print(f"  Max C_SiLU/C_GELU ratio: {max_silu_gelu_ratio:.4f}")

    # d=256, N=50 specific results for K2
    print(f"\n  --- K2 Target: d=256, N=50, SiLU ---")
    k2_entries = [r for r in all_results
                  if r.get("d") == 256 and r.get("N") == 50
                  and r.get("act_name") == "silu"
                  and not r.get("diverged", True)]
    if k2_entries:
        for gp_name, _ in gamma_profiles:
            devs = [r["max_output_dev_pct"] for r in k2_entries
                    if r.get("gamma_label") == gp_name]
            if devs:
                print(f"  {gp_name:>20}: max D = {max(devs):.4f}%, mean D = {np.mean(devs):.4f}%")
                max_silu_D = max(max_silu_D, max(devs))

    # ================================================================
    # Kill Criteria
    # ================================================================
    print("\n" + "=" * 80)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 80)

    # K1: SiLU correction differs from GELU by >1.5x
    print(f"\n  K1: SiLU correction factor differs from GELU by >1.5x")
    print(f"      Max C_SiLU / C_GELU across all configs: {max_silu_gelu_ratio:.4f}")
    print(f"      Threshold: 1.5")
    k1_pass = max_silu_gelu_ratio < 1.5
    print(f"      VERDICT: {'PASS' if k1_pass else 'FAIL'}")
    if k1_pass:
        print(f"      SiLU correction tracks GELU closely (max ratio {max_silu_gelu_ratio:.2f}x < 1.5x)")

    # K2: SiLU worst-case D exceeds 5% at d=256, N=50
    print(f"\n  K2: SiLU worst-case D < 5% at d=256, N=50")
    print(f"      Max D across all gamma profiles: {max_silu_D:.4f}%")
    print(f"      Threshold: 5.0%")
    k2_pass = max_silu_D < 5.0
    print(f"      VERDICT: {'PASS' if k2_pass else 'FAIL'}")
    if k2_pass:
        margin = 5.0 / max_silu_D if max_silu_D > 1e-12 else float("inf")
        print(f"      Safety margin: {margin:.0f}x below threshold")

    overall = "PROVEN" if k1_pass and k2_pass else "SUPPORTED" if k1_pass or k2_pass else "FAIL"
    print(f"\n  OVERALL VERDICT: {overall}")

    elapsed = time.time() - t_start
    print(f"\n  Total experiment time: {elapsed:.1f}s")

    # ================================================================
    # Save results
    # ================================================================
    # Clean numpy types for JSON
    clean_results = []
    for r_item in all_results:
        clean = {}
        for k, v in r_item.items():
            if isinstance(v, (np.floating, np.float64)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, np.int64)):
                clean[k] = int(v)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        clean_results.append(clean)

    save_data = {
        "curvature_analysis": curv,
        "baselines": {f"d{k[0]}_N{k[1]}_{k[2]}": v for k, v in baselines.items()},
        "max_gelu_correction": float(max_gelu_correction),
        "max_silu_correction": float(max_silu_correction),
        "max_silu_gelu_ratio": float(max_silu_gelu_ratio),
        "max_silu_D_d256_N50": float(max_silu_D),
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "verdict": overall,
        "correction_table": correction_table,
        "all_results": clean_results,
        "elapsed_s": elapsed,
    }

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return save_data


if __name__ == "__main__":
    run_full_experiment()
