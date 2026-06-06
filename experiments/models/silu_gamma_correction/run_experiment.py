"""SiLU Gamma Correction: Does SiLU produce similar correction factor as GELU (~1.43x)?

Parent experiment (rmsnorm_gamma_nonuniformity) PROVED that with GELU activation:
  - Worst-case gamma correction factor = 1.43x (single layer gamma=10)
  - Realistic gamma (lognormal sigma=0.5) gives only 1.02x
  - K2: D=0.098% at d=256, N=50 (51x below 5% threshold)

Adversarial review flagged: Qwen2.5 and Llama use SiLU (x * sigmoid(x)), not GELU.
The 1.43x factor needs revalidation with the actual production activation.

Kill criteria:
  K1: SiLU worst-case correction factor exceeds 2.0x (vs GELU 1.43x)
  K2: SiLU correction diverges from GELU by >50% at matched gamma profiles

This experiment runs the SAME gamma sweep with BOTH activations side-by-side
and compares the correction factors directly.

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Core utilities (from parent experiment)
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single LoRA expert for one layer."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


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


# ============================================================================
# Activation functions
# ============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (aka Swish): x * sigmoid(x).

    Used by Qwen2.5, Llama, Mistral, and most modern LLMs.
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    # Numerically stable: avoid overflow in exp(-x) for large negative x
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


# ============================================================================
# RMSNorm with learnable gamma
# ============================================================================

def rms_norm(x: np.ndarray, gamma: np.ndarray = None, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm with optional per-dimension learnable gamma."""
    if x.ndim == 1:
        rms = np.sqrt(np.mean(x ** 2) + eps)
        normed = x / rms
    else:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        normed = x / rms
    if gamma is not None:
        return gamma * normed
    return normed


# ============================================================================
# Gamma distribution generators (identical to parent)
# ============================================================================

def generate_uniform_gamma(d: int, L: int) -> list[np.ndarray]:
    return [np.ones(d) for _ in range(L)]


def generate_lognormal_gamma(d: int, L: int, sigma: float = 0.5,
                              seed: int = 999) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    gammas = []
    for _ in range(L):
        g = np.exp(rng.normal(0, sigma, size=d))
        g = np.clip(g, 0.01, 100.0)
        gammas.append(g)
    return gammas


def generate_bimodal_gamma(d: int, L: int, low: float = 0.2, high: float = 5.0,
                           frac_high: float = 0.25, seed: int = 999) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    gammas = []
    n_high = int(d * frac_high)
    for _ in range(L):
        g = np.full(d, low)
        high_idx = rng.choice(d, size=n_high, replace=False)
        g[high_idx] = high
        gammas.append(g)
    return gammas


def generate_layerwise_gamma(d: int, L: int, layer_scales: np.ndarray,
                              seed: int = 999) -> list[np.ndarray]:
    gammas = []
    for l in range(L):
        gammas.append(np.full(d, layer_scales[l]))
    return gammas


def describe_gamma(gammas: list[np.ndarray]) -> dict:
    all_vals = np.concatenate(gammas)
    return {
        "global_mean": float(np.mean(all_vals)),
        "global_std": float(np.std(all_vals)),
        "global_min": float(np.min(all_vals)),
        "global_max": float(np.max(all_vals)),
        "max_ratio": float(np.max(all_vals) / (np.min(all_vals) + 1e-12)),
    }


# ============================================================================
# Forward pass with selectable activation
# ============================================================================

def forward_pre_rmsn_gamma(h: np.ndarray, base_weights: list[np.ndarray],
                            layer_deltas: list[np.ndarray],
                            gammas: list[np.ndarray] = None,
                            act_fn=gelu) -> np.ndarray:
    """Pre-RMSNorm transformer with per-layer learnable gamma.

    h_{l+1} = h_l + act_fn((W_l + Delta_l) @ RMSNorm(h_l; gamma_l))
    """
    L = len(base_weights)
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        g = gammas[l] if gammas is not None else None
        h = h + act_fn(W @ rms_norm(h, gamma=g))
    return h


# ============================================================================
# Core experiment: measure alpha for a given gamma profile and activation
# ============================================================================

def run_single_config(d: int, r: int, L: int, N: int,
                      n_inputs: int, remove_idx: int, seed: int,
                      gammas: list[np.ndarray] = None,
                      gamma_label: str = "uniform",
                      act_fn=gelu,
                      act_name: str = "gelu") -> dict:
    """Run the complete bound experiment for one configuration."""
    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Generate N experts at L layers
    experts = []
    for i in range(N):
        layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
        experts.append(layers)

    # Generate base weights
    base_weights = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]

    # Generate test inputs
    inputs = rng.randn(n_inputs, d) * 0.1

    # GS merge all N experts per layer
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Per-layer weight-space error
        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_errors.append(recon_err)

    sum_per_layer = sum(per_layer_errors)

    # Forward pass helper
    def fwd(h, deltas):
        return forward_pre_rmsn_gamma(h, base_weights, deltas,
                                       gammas=gammas, act_fn=act_fn)

    # Forward with all experts
    outputs_all = np.array([fwd(inp, all_merged_deltas) for inp in inputs])

    # Forward after naive removal
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = np.array([fwd(inp, naive_removed_deltas) for inp in inputs])

    # Forward with GS recompute (ground truth)
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = np.array([fwd(inp, gt_removed_deltas) for inp in inputs])

    # Metrics
    has_nan = bool(np.any(np.isnan(outputs_naive)) or np.any(np.isnan(outputs_gt)))
    has_inf = bool(np.any(np.isinf(outputs_naive)) or np.any(np.isinf(outputs_gt)))

    if has_nan or has_inf:
        return {
            "act_name": act_name,
            "gamma_label": gamma_label,
            "d": d, "r": r, "L": L, "N": N,
            "seed": seed,
            "sum_per_layer_error_pct": sum_per_layer,
            "mean_output_dev_pct": float("nan"),
            "max_output_dev_pct": float("nan"),
            "amplification_ratio": float("nan"),
            "has_nan": has_nan, "has_inf": has_inf,
            "diverged": True,
            "elapsed_s": time.time() - t0,
        }

    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_dev = float(np.mean(relative_devs))
    max_dev = float(np.max(relative_devs))
    amp_ratio = mean_dev / sum_per_layer if sum_per_layer > 1e-12 else 0.0

    return {
        "act_name": act_name,
        "gamma_label": gamma_label,
        "d": d, "r": r, "L": L, "N": N,
        "seed": seed,
        "sum_per_layer_error_pct": sum_per_layer,
        "mean_output_dev_pct": mean_dev,
        "max_output_dev_pct": max_dev,
        "amplification_ratio": amp_ratio,
        "has_nan": False, "has_inf": False,
        "diverged": False,
        "elapsed_s": time.time() - t0,
    }


# ============================================================================
# Analytical comparison: SiLU vs GELU nonlinearity response
# ============================================================================

def analyze_activation_response():
    """Compare SiLU and GELU response curves at key operating points.

    The gamma correction factor comes from the nonlinear activation response.
    At the linear regime (small |x|), both activations are ~x/2, so gamma cancels.
    The correction factor comes from how much the activation deviates from linearity
    at the operating points created by gamma-scaled inputs.
    """
    print("\n" + "=" * 80)
    print("  ANALYTICAL: SiLU vs GELU nonlinearity comparison")
    print("=" * 80)

    x = np.linspace(-5, 5, 1000)
    gelu_y = gelu(x)
    silu_y = silu(x)

    # Derivatives (numerical)
    dx = x[1] - x[0]
    gelu_dy = np.gradient(gelu_y, dx)
    silu_dy = np.gradient(silu_y, dx)

    # Second derivatives (curvature -- this drives the correction factor)
    gelu_d2y = np.gradient(gelu_dy, dx)
    silu_d2y = np.gradient(silu_dy, dx)

    # Key metrics at different operating points
    test_points = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"\n  {'x':>6} {'GELU(x)':>10} {'SiLU(x)':>10} {'GELU/SiLU':>10} {'GELU_d2':>10} {'SiLU_d2':>10}")
    print("  " + "-" * 62)
    for xp in test_points:
        gv = gelu(np.array([xp]))[0]
        sv = silu(np.array([xp]))[0]
        ratio = gv / sv if abs(sv) > 1e-12 else float("inf")
        # Second derivative at xp
        idx = int((xp + 5) / 10 * 999)
        idx = min(idx, 998)
        print(f"  {xp:>6.1f} {gv:>10.4f} {sv:>10.4f} {ratio:>10.4f} {gelu_d2y[idx]:>10.4f} {silu_d2y[idx]:>10.4f}")

    # The max absolute curvature determines how much gamma affects alpha
    gelu_max_curv = np.max(np.abs(gelu_d2y[10:-10]))
    silu_max_curv = np.max(np.abs(silu_d2y[10:-10]))
    print(f"\n  Max |curvature| in [-5,5]: GELU={gelu_max_curv:.4f}, SiLU={silu_max_curv:.4f}")
    print(f"  SiLU/GELU curvature ratio: {silu_max_curv/gelu_max_curv:.4f}")
    print(f"  Prediction: SiLU correction factor should be ~{silu_max_curv/gelu_max_curv:.2f}x of GELU's")

    return {
        "gelu_max_curvature": float(gelu_max_curv),
        "silu_max_curvature": float(silu_max_curv),
        "curvature_ratio": float(silu_max_curv / gelu_max_curv),
    }


# ============================================================================
# Main experiment
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: SiLU Gamma Correction Factor")
    print("  Does SiLU produce similar correction factor as GELU (~1.43x)?")
    print("  K1: SiLU worst-case correction factor < 2.0x")
    print("  K2: SiLU correction diverges from GELU by < 50%")
    print("=" * 80)

    seeds = [42, 123, 777]

    # ================================================================
    # Step 0: Analytical comparison
    # ================================================================
    analytical = analyze_activation_response()

    # ================================================================
    # TEST 1: Side-by-side GELU vs SiLU gamma variance sweep
    # d=64, N=8, L=24 (matches parent exactly)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: GELU vs SiLU gamma variance sweep (d=64, N=8, L=24)")
    print("=" * 80)

    d, r, L, N, n_inputs = 64, 8, 24, 8, 300
    sigma_values = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

    activations = [
        ("gelu", gelu),
        ("silu", silu),
    ]

    # Store baseline alphas per activation
    baseline_alphas = {"gelu": [], "silu": []}
    variance_results = []

    for sigma in sigma_values:
        label = f"lognormal_sigma={sigma:.1f}" if sigma > 0 else "uniform"
        row_data = {"sigma": sigma, "label": label}

        for act_name, act_fn in activations:
            seed_amps = []
            for seed in seeds:
                if sigma == 0:
                    gammas = generate_uniform_gamma(d, L)
                else:
                    gammas = generate_lognormal_gamma(d, L, sigma=sigma, seed=seed + 1000)

                res = run_single_config(
                    d=d, r=r, L=L, N=N, n_inputs=n_inputs,
                    remove_idx=N // 2, seed=seed,
                    gammas=gammas, gamma_label=label,
                    act_fn=act_fn, act_name=act_name,
                )
                variance_results.append(res)

                if not res["diverged"]:
                    seed_amps.append(res["amplification_ratio"])
                    if sigma == 0:
                        baseline_alphas[act_name].append(res["amplification_ratio"])

            if seed_amps:
                row_data[f"{act_name}_alpha"] = float(np.mean(seed_amps))
                row_data[f"{act_name}_std"] = float(np.std(seed_amps))

    # Print comparison table
    gelu_base = np.mean(baseline_alphas["gelu"]) if baseline_alphas["gelu"] else 0.022
    silu_base = np.mean(baseline_alphas["silu"]) if baseline_alphas["silu"] else 0.022

    print(f"\n  Baselines: GELU alpha={gelu_base:.4f}, SiLU alpha={silu_base:.4f}")
    print(f"\n  {'Sigma':>6} {'GELU_alpha':>12} {'GELU_ratio':>12} {'SiLU_alpha':>12} {'SiLU_ratio':>12} {'SiLU/GELU':>10}")
    print("  " + "-" * 70)

    # Re-aggregate for printing
    for sigma in sigma_values:
        label = f"lognormal_sigma={sigma:.1f}" if sigma > 0 else "uniform"
        gelu_amps = [r["amplification_ratio"] for r in variance_results
                     if r["gamma_label"] == label and r["act_name"] == "gelu"
                     and not r["diverged"]]
        silu_amps = [r["amplification_ratio"] for r in variance_results
                     if r["gamma_label"] == label and r["act_name"] == "silu"
                     and not r["diverged"]]

        if gelu_amps and silu_amps:
            ga = np.mean(gelu_amps)
            sa = np.mean(silu_amps)
            gr = ga / gelu_base if gelu_base > 1e-12 else 0
            sr = sa / silu_base if silu_base > 1e-12 else 0
            ratio = sa / ga if ga > 1e-12 else 0
            print(f"  {sigma:>6.1f} {ga:>12.4f} {gr:>12.2f}x {sa:>12.4f} {sr:>12.2f}x {ratio:>10.4f}")

    # ================================================================
    # TEST 2: Side-by-side worst-case layer-wise profiles
    # These produced the 1.43x max in the parent experiment
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: GELU vs SiLU layer-wise gamma profiles (d=64, N=8, L=24)")
    print("  These are the profiles that produced the 1.43x max in the parent")
    print("=" * 80)

    layerwise_configs = [
        {"name": "linear_ramp_1_to_3",
         "scales": np.linspace(1.0, 3.0, L)},
        {"name": "alternating_0.5_2.0",
         "scales": np.array([0.5 if l % 2 == 0 else 2.0 for l in range(L)])},
        {"name": "early_high_5.0",
         "scales": np.array([5.0 if l < L // 4 else 1.0 for l in range(L)])},
        {"name": "late_high_5.0",
         "scales": np.array([1.0 if l < 3 * L // 4 else 5.0 for l in range(L)])},
        {"name": "single_spike_10.0",
         "scales": np.array([10.0 if l == L // 2 else 1.0 for l in range(L)])},
    ]

    layerwise_results = []

    print(f"\n  {'Profile':>25} {'GELU_alpha':>12} {'GELU_ratio':>12} {'SiLU_alpha':>12} {'SiLU_ratio':>12} {'SiLU/GELU':>10}")
    print("  " + "-" * 85)

    max_gelu_ratio = 1.0
    max_silu_ratio = 1.0

    for cfg in layerwise_configs:
        gelu_amps = []
        silu_amps = []

        for act_name, act_fn in activations:
            seed_amps = []
            for seed in seeds:
                gammas = generate_layerwise_gamma(d, L, cfg["scales"], seed=seed + 3000)
                res = run_single_config(
                    d=d, r=r, L=L, N=N, n_inputs=n_inputs,
                    remove_idx=N // 2, seed=seed,
                    gammas=gammas, gamma_label=cfg["name"],
                    act_fn=act_fn, act_name=act_name,
                )
                layerwise_results.append(res)
                if not res["diverged"]:
                    seed_amps.append(res["amplification_ratio"])

            if act_name == "gelu":
                gelu_amps = seed_amps
            else:
                silu_amps = seed_amps

        if gelu_amps and silu_amps:
            ga = np.mean(gelu_amps)
            sa = np.mean(silu_amps)
            gr = ga / gelu_base
            sr = sa / silu_base
            max_gelu_ratio = max(max_gelu_ratio, gr)
            max_silu_ratio = max(max_silu_ratio, sr)
            ratio = sa / ga
            print(f"  {cfg['name']:>25} {ga:>12.4f} {gr:>12.2f}x {sa:>12.4f} {sr:>12.2f}x {ratio:>10.4f}")

    # ================================================================
    # TEST 3: Bimodal worst-case profiles
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 3: GELU vs SiLU bimodal gamma profiles (d=64, N=8, L=24)")
    print("=" * 80)

    bimodal_configs = [
        {"low": 0.5, "high": 2.0, "frac": 0.25, "label": "bimodal_0.5/2.0"},
        {"low": 0.2, "high": 5.0, "frac": 0.25, "label": "bimodal_0.2/5.0"},
        {"low": 0.1, "high": 10.0, "frac": 0.25, "label": "bimodal_0.1/10.0"},
    ]

    bimodal_results = []

    print(f"\n  {'Profile':>25} {'GELU_alpha':>12} {'GELU_ratio':>12} {'SiLU_alpha':>12} {'SiLU_ratio':>12} {'SiLU/GELU':>10}")
    print("  " + "-" * 85)

    for cfg in bimodal_configs:
        gelu_amps = []
        silu_amps = []

        for act_name, act_fn in activations:
            seed_amps = []
            for seed in seeds:
                gammas = generate_bimodal_gamma(
                    d, L, low=cfg["low"], high=cfg["high"],
                    frac_high=cfg["frac"], seed=seed + 2000
                )
                res = run_single_config(
                    d=d, r=r, L=L, N=N, n_inputs=n_inputs,
                    remove_idx=N // 2, seed=seed,
                    gammas=gammas, gamma_label=cfg["label"],
                    act_fn=act_fn, act_name=act_name,
                )
                bimodal_results.append(res)
                if not res["diverged"]:
                    seed_amps.append(res["amplification_ratio"])

            if act_name == "gelu":
                gelu_amps = seed_amps
            else:
                silu_amps = seed_amps

        if gelu_amps and silu_amps:
            ga = np.mean(gelu_amps)
            sa = np.mean(silu_amps)
            gr = ga / gelu_base
            sr = sa / silu_base
            max_gelu_ratio = max(max_gelu_ratio, gr)
            max_silu_ratio = max(max_silu_ratio, sr)
            ratio = sa / ga
            print(f"  {cfg['label']:>25} {ga:>12.4f} {gr:>12.2f}x {sa:>12.4f} {sr:>12.2f}x {ratio:>10.4f}")

    # ================================================================
    # TEST 4: Scale sweep at d=256, N=50 (K2 target)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 4: GELU vs SiLU at d=256, N=50 (K2 target scale)")
    print("=" * 80)

    scale_configs = [
        {"d": 64, "r": 8, "L": 24, "N": 50, "n_inputs": 200, "label": "d64_N50"},
        {"d": 128, "r": 8, "L": 24, "N": 50, "n_inputs": 200, "label": "d128_N50"},
        {"d": 256, "r": 8, "L": 24, "N": 50, "n_inputs": 100, "label": "d256_N50"},
    ]

    gamma_variants = [
        {"name": "uniform", "gen": lambda d, L, seed: generate_uniform_gamma(d, L)},
        {"name": "lognormal_0.5", "gen": lambda d, L, seed: generate_lognormal_gamma(d, L, sigma=0.5, seed=seed)},
        {"name": "bimodal_0.2/5.0", "gen": lambda d, L, seed: generate_bimodal_gamma(d, L, low=0.2, high=5.0, frac_high=0.25, seed=seed)},
    ]

    scale_results = []

    for cfg in scale_configs:
        print(f"\n  Config: {cfg['label']}")
        print(f"  {'Gamma':>20} {'GELU_dev%':>12} {'SiLU_dev%':>12} {'SiLU/GELU':>10}")
        print("  " + "-" * 58)

        for gv in gamma_variants:
            gelu_devs = []
            silu_devs = []

            for act_name, act_fn in activations:
                seed_devs = []
                for seed in seeds:
                    gammas = gv["gen"](cfg["d"], cfg["L"], seed + 4000)
                    res = run_single_config(
                        d=cfg["d"], r=cfg["r"], L=cfg["L"], N=cfg["N"],
                        n_inputs=cfg["n_inputs"],
                        remove_idx=cfg["N"] // 2, seed=seed,
                        gammas=gammas, gamma_label=gv["name"],
                        act_fn=act_fn, act_name=act_name,
                    )
                    res["config_label"] = cfg["label"]
                    scale_results.append(res)
                    if not res["diverged"]:
                        seed_devs.append(res["mean_output_dev_pct"])

                if act_name == "gelu":
                    gelu_devs = seed_devs
                else:
                    silu_devs = seed_devs

            if gelu_devs and silu_devs:
                gd = np.mean(gelu_devs)
                sd = np.mean(silu_devs)
                ratio = sd / gd if gd > 1e-12 else 0
                print(f"  {gv['name']:>20} {gd:>12.4f} {sd:>12.4f} {ratio:>10.4f}")

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("  AGGREGATE ANALYSIS")
    print("=" * 80)

    # Collect all ratios per activation across all tests (d=64 only for ratio analysis)
    all_d64_results = variance_results + layerwise_results + bimodal_results

    # Group by gamma_label and activation
    label_act_groups = {}
    for r_item in all_d64_results:
        if r_item["diverged"]:
            continue
        key = (r_item["gamma_label"], r_item["act_name"])
        if key not in label_act_groups:
            label_act_groups[key] = []
        label_act_groups[key].append(r_item["amplification_ratio"])

    # Compute ratios
    all_gelu_ratios = []
    all_silu_ratios = []
    all_divergence_pcts = []

    unique_labels = sorted(set(k[0] for k in label_act_groups.keys()))

    print(f"\n  {'Gamma Profile':>30} {'GELU_ratio':>12} {'SiLU_ratio':>12} {'Divergence%':>12}")
    print("  " + "-" * 70)

    for label in unique_labels:
        gelu_key = (label, "gelu")
        silu_key = (label, "silu")

        if gelu_key in label_act_groups and silu_key in label_act_groups:
            ga = np.mean(label_act_groups[gelu_key])
            sa = np.mean(label_act_groups[silu_key])
            gr = ga / gelu_base
            sr = sa / silu_base
            all_gelu_ratios.append(gr)
            all_silu_ratios.append(sr)

            # Divergence = |SiLU_ratio - GELU_ratio| / GELU_ratio * 100
            divergence = abs(sr - gr) / gr * 100.0 if gr > 1e-12 else 0
            all_divergence_pcts.append(divergence)

            print(f"  {label:>30} {gr:>12.2f}x {sr:>12.2f}x {divergence:>12.1f}%")

    # Overall max ratios
    if all_gelu_ratios:
        max_gelu = max(all_gelu_ratios)
    else:
        max_gelu = max_gelu_ratio

    if all_silu_ratios:
        max_silu = max(all_silu_ratios)
    else:
        max_silu = max_silu_ratio

    max_divergence = max(all_divergence_pcts) if all_divergence_pcts else 0

    print(f"\n  Max GELU correction ratio: {max_gelu:.2f}x")
    print(f"  Max SiLU correction ratio: {max_silu:.2f}x")
    print(f"  Max divergence between SiLU and GELU: {max_divergence:.1f}%")

    # ================================================================
    # KILL CRITERIA ASSESSMENT
    # ================================================================
    print("\n" + "=" * 80)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 80)

    # K1: SiLU worst-case correction factor < 2.0x
    print(f"\n  K1: SiLU worst-case correction factor < 2.0x")
    print(f"      SiLU baseline alpha: {silu_base:.4f}")
    print(f"      SiLU max correction ratio: {max_silu:.2f}x")
    k1_pass = max_silu < 2.0
    print(f"      VERDICT: {'PASS' if k1_pass else 'FAIL'} (threshold: 2.0x)")

    # K2: SiLU correction diverges from GELU by <50%
    print(f"\n  K2: SiLU correction diverges from GELU by <50%")
    print(f"      GELU max correction ratio: {max_gelu:.2f}x")
    print(f"      SiLU max correction ratio: {max_silu:.2f}x")
    print(f"      Max divergence across all profiles: {max_divergence:.1f}%")
    k2_pass = max_divergence < 50.0
    print(f"      VERDICT: {'PASS' if k2_pass else 'FAIL'} (threshold: 50%)")

    # D at d=256 with SiLU
    d256_silu_devs = [r_item["mean_output_dev_pct"] for r_item in scale_results
                      if r_item.get("config_label") == "d256_N50"
                      and r_item["act_name"] == "silu"
                      and not r_item["diverged"]]
    if d256_silu_devs:
        max_d256_silu = max(d256_silu_devs)
        mean_d256_silu = np.mean(d256_silu_devs)
        print(f"\n  Bonus: d=256, N=50 SiLU deviation")
        print(f"      Mean D: {mean_d256_silu:.4f}%")
        print(f"      Max D: {max_d256_silu:.4f}%")
        print(f"      Margin below 5%: {5.0 / max_d256_silu:.0f}x")

    # Overall
    overall = "PROVEN" if k1_pass and k2_pass else "SUPPORTED" if k1_pass or k2_pass else "FAIL"
    print(f"\n  OVERALL VERDICT: {overall}")

    if k1_pass and k2_pass:
        print(f"    SiLU produces a similar gamma correction factor as GELU.")
        print(f"    The 1.43x GELU result transfers to SiLU-based architectures.")
        print(f"    Qwen2.5 and Llama safety bounds are validated.")
    elif k1_pass:
        print(f"    SiLU correction factor is within 2.0x (safe), but diverges")
        print(f"    from GELU by >{max_divergence:.0f}%. Architecture-specific bounds recommended.")
    else:
        print(f"    SiLU correction factor exceeds 2.0x threshold!")
        print(f"    Safety bound must be recalculated for SiLU architectures.")

    # ================================================================
    # Save results
    # ================================================================
    all_results = variance_results + layerwise_results + bimodal_results + scale_results

    # Clean for JSON
    clean_results = []
    for r_item in all_results:
        clean = {}
        for k, v in r_item.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            elif isinstance(v, (np.floating, np.float64)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, np.int64)):
                clean[k] = int(v)
            else:
                clean[k] = v
        clean_results.append(clean)

    save_data = {
        "gelu_baseline_alpha": float(gelu_base),
        "silu_baseline_alpha": float(silu_base),
        "max_gelu_correction_ratio": float(max_gelu),
        "max_silu_correction_ratio": float(max_silu),
        "max_divergence_pct": float(max_divergence),
        "analytical": analytical,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "verdict": overall,
        "all_results": clean_results,
    }

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total experiment time: {elapsed:.1f}s")

    return save_data


if __name__ == "__main__":
    run_full_experiment()
