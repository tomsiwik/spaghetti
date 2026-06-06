"""Residual + LayerNorm Error Dynamics: How do transformer-like architectures
change multi-layer error propagation for expert removal?

Parent experiment (multilayer_removal_cascade) showed:
  - Feedforward-only: amp_ratio=0.25 at L=24 (75% error dampening)
  - Sub-additive error accumulation with 3 mechanisms:
    activation masking, direction randomization, spectral contraction

This experiment adds realistic transformer components:
  1. Residual connections: h_{l+1} = h_l + f(h_l) -- identity path
  2. LayerNorm: normalizes activations per layer
  3. RMSNorm: simpler norm (Qwen/Llama use this)
  4. Pre-LN transformer block: h_{l+1} = h_l + f(LN(h_l))
  5. Post-LN transformer block: h_{l+1} = LN(h_l + f(h_l))

Kill criteria:
  K1: residual connections change amplification ratio by >50% vs feedforward-only
      (dynamics fundamentally different)
  K2: LayerNorm renormalization makes error propagation dimension-independent
      (breaks 1/d scaling observed in parent)

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Reuse from parent: expert generation, GS merge
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single LoRA expert for one layer."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


def generate_multilayer_experts(N: int, L: int, d: int, r: int,
                                 cluster_cos: float | None = None,
                                 seed: int = 42) -> list[list[dict]]:
    """Generate N experts, each with L layer-wise LoRA deltas."""
    rng = np.random.RandomState(seed)

    if cluster_cos is None:
        experts = []
        for i in range(N):
            layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
            experts.append(layers)
        return experts

    alpha = np.sqrt(cluster_cos)
    beta = np.sqrt(1.0 - cluster_cos)

    shared_dWs = []
    for l in range(L):
        shared = rng.randn(d, d) / np.sqrt(d)
        shared = shared / np.linalg.norm(shared)
        shared_dWs.append(shared)

    experts = []
    for i in range(N):
        layers = []
        for l in range(L):
            unique = generate_lora_expert_layer(d, r, rng)
            unique_dW = unique["dW"]
            unique_dW_norm = np.linalg.norm(unique_dW)
            if unique_dW_norm > 1e-12:
                unique_dW = unique_dW / unique_dW_norm

            shared = shared_dWs[l]
            proj = np.sum(unique_dW * shared) * shared
            unique_dW = unique_dW - proj
            un = np.linalg.norm(unique_dW)
            if un > 1e-12:
                unique_dW = unique_dW / un

            dW = alpha * shared + beta * unique_dW
            dW = dW * rng.uniform(0.8, 1.2) * 0.01
            layers.append({"dW": dW})
        experts.append(layers)
    return experts


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


def activation(x: np.ndarray, kind: str = "gelu") -> np.ndarray:
    """Activation function."""
    if kind == "relu":
        return np.maximum(0, x)
    elif kind == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    elif kind == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {kind}")


# ============================================================================
# Normalization functions (numpy, no learnable params)
# ============================================================================

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm without learnable scale/shift (unit scale, zero shift).

    x: shape (d,) or (batch, d). Normalizes over last dimension.
    """
    if x.ndim == 1:
        mean = x.mean()
        var = x.var()
        return (x - mean) / np.sqrt(var + eps)
    else:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm without learnable scale (Qwen/Llama style).

    x: shape (d,) or (batch, d).
    """
    if x.ndim == 1:
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return x / rms
    else:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms


# ============================================================================
# Forward passes for different architectures
# ============================================================================

def forward_feedforward(h: np.ndarray, base_weights: list[np.ndarray],
                        layer_deltas: list[np.ndarray],
                        act: str = "gelu") -> np.ndarray:
    """Parent baseline: h_{l+1} = sigma((W_l + Delta_l) @ h_l)."""
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = activation(W @ h, kind=act)
    return h


def forward_residual(h: np.ndarray, base_weights: list[np.ndarray],
                     layer_deltas: list[np.ndarray],
                     act: str = "gelu") -> np.ndarray:
    """Residual: h_{l+1} = h_l + sigma((W_l + Delta_l) @ h_l).

    Scale the linear layer output by 1/sqrt(L) to prevent explosion,
    following standard init practice for deep residual networks.
    """
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ h, kind=act)
    return h


def forward_layernorm(h: np.ndarray, base_weights: list[np.ndarray],
                      layer_deltas: list[np.ndarray],
                      act: str = "gelu") -> np.ndarray:
    """LayerNorm only: h_{l+1} = LN(sigma((W_l + Delta_l) @ h_l))."""
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = layer_norm(activation(W @ h, kind=act))
    return h


def forward_rmsnorm(h: np.ndarray, base_weights: list[np.ndarray],
                    layer_deltas: list[np.ndarray],
                    act: str = "gelu") -> np.ndarray:
    """RMSNorm only: h_{l+1} = RMSNorm(sigma((W_l + Delta_l) @ h_l))."""
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = rms_norm(activation(W @ h, kind=act))
    return h


def forward_pre_ln(h: np.ndarray, base_weights: list[np.ndarray],
                   layer_deltas: list[np.ndarray],
                   act: str = "gelu") -> np.ndarray:
    """Pre-LN transformer: h_{l+1} = h_l + sigma((W_l + Delta_l) @ LN(h_l)).

    This is what GPT-2 and most modern transformers use.
    Scale by 1/sqrt(L) for stability.
    """
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ layer_norm(h), kind=act)
    return h


def forward_pre_rmsn(h: np.ndarray, base_weights: list[np.ndarray],
                     layer_deltas: list[np.ndarray],
                     act: str = "gelu") -> np.ndarray:
    """Pre-RMSNorm transformer: h_{l+1} = h_l + sigma((W_l + Delta_l) @ RMSNorm(h_l)).

    This is what Qwen/Llama use in production.
    Scale by 1/sqrt(L) for stability.
    """
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ rms_norm(h), kind=act)
    return h


def forward_post_ln(h: np.ndarray, base_weights: list[np.ndarray],
                    layer_deltas: list[np.ndarray],
                    act: str = "gelu") -> np.ndarray:
    """Post-LN transformer: h_{l+1} = LN(h_l + sigma((W_l + Delta_l) @ h_l)).

    Original transformer (Vaswani et al., 2017).
    """
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = layer_norm(h + scale * activation(W @ h, kind=act))
    return h


# Map of architecture names to forward functions
ARCHITECTURES = {
    "feedforward": forward_feedforward,
    "residual": forward_residual,
    "layernorm": forward_layernorm,
    "rmsnorm": forward_rmsnorm,
    "pre_ln": forward_pre_ln,
    "pre_rmsn": forward_pre_rmsn,
    "post_ln": forward_post_ln,
}


# ============================================================================
# Core Experiment: Removal Error per Architecture
# ============================================================================

def run_removal_cascade(N: int, L: int, d: int, r: int,
                         cluster_cos: float | None, act: str,
                         n_inputs: int, remove_idx: int,
                         seed: int, arch: str) -> dict:
    """Run one configuration of the removal cascade experiment.

    Same structure as parent but parameterized by architecture.
    """
    rng = np.random.RandomState(seed)
    forward_fn = ARCHITECTURES[arch]

    # Generate experts
    experts = generate_multilayer_experts(N, L, d, r, cluster_cos, seed=seed)

    # Generate base weights
    base_weights = []
    for l in range(L):
        W = rng.randn(d, d) / np.sqrt(d)
        base_weights.append(W)

    # Generate test inputs
    inputs = rng.randn(n_inputs, d) * 0.1

    # Step 1: Merge all N experts per layer via GS
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_recon_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_recon_errors.append(recon_err)

    # Step 2: Forward pass with ALL experts
    outputs_all = []
    for inp in inputs:
        out = forward_fn(inp, base_weights, all_merged_deltas, act=act)
        outputs_all.append(out)
    outputs_all = np.array(outputs_all)

    # Step 3: Forward pass after naive removal
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = []
    for inp in inputs:
        out = forward_fn(inp, base_weights, naive_removed_deltas, act=act)
        outputs_naive.append(out)
    outputs_naive = np.array(outputs_naive)

    # Step 4: Forward pass with GS recompute (ground truth)
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = []
    for inp in inputs:
        out = forward_fn(inp, base_weights, gt_removed_deltas, act=act)
        outputs_gt.append(out)
    outputs_gt = np.array(outputs_gt)

    # Metrics
    sum_per_layer = sum(per_layer_recon_errors)

    # Handle NaN/Inf outputs (can happen with deep feedforward without residual)
    if np.any(~np.isfinite(outputs_naive)) or np.any(~np.isfinite(outputs_gt)):
        return {
            "N": N, "L": L, "d": d, "r": r,
            "cluster_cos": cluster_cos, "activation": act, "arch": arch,
            "n_inputs": n_inputs, "remove_idx": remove_idx, "seed": seed,
            "sum_per_layer_error": sum_per_layer,
            "mean_output_dev_pct": float("nan"),
            "max_output_dev_pct": float("nan"),
            "median_output_dev_pct": float("nan"),
            "amplification_ratio": float("nan"),
            "expert_signal_pct": float("nan"),
            "output_scale": float("nan"),
            "diverged": True,
        }

    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_output_dev = float(np.mean(relative_devs))
    max_output_dev = float(np.max(relative_devs))
    median_output_dev = float(np.median(relative_devs))

    if sum_per_layer > 1e-12:
        amplification_ratio = mean_output_dev / sum_per_layer
    else:
        amplification_ratio = 0.0

    all_vs_gt_norms = np.linalg.norm(outputs_all - outputs_gt, axis=1)
    expert_signal = float(np.mean(all_vs_gt_norms / safe_gt * 100.0))
    output_scale = float(np.mean(gt_norms))

    return {
        "N": N, "L": L, "d": d, "r": r,
        "cluster_cos": cluster_cos, "activation": act, "arch": arch,
        "n_inputs": n_inputs, "remove_idx": remove_idx, "seed": seed,
        "sum_per_layer_error": sum_per_layer,
        "mean_output_dev_pct": mean_output_dev,
        "max_output_dev_pct": max_output_dev,
        "median_output_dev_pct": median_output_dev,
        "amplification_ratio": amplification_ratio,
        "expert_signal_pct": expert_signal,
        "output_scale": output_scale,
        "diverged": False,
    }


# ============================================================================
# Experiment Runner
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: Residual + LayerNorm Error Dynamics")
    print("  Parent: multilayer_removal_cascade (feedforward, amp_ratio=0.25 at L=24)")
    print("  K1: residual changes amp_ratio by >50%")
    print("  K2: LayerNorm breaks 1/d scaling")
    print("=" * 80)

    d = 64
    r = 8
    n_inputs = 200
    seeds = [42, 123, 777]
    N = 8

    all_results = []

    # ================================================================
    # TEST 1: Architecture comparison across depth (near-orthogonal)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: Architecture comparison across depth")
    print("  d=64, r=8, N=8, GELU, near-orthogonal")
    print("=" * 80)

    L_values = [1, 2, 4, 8, 12, 16, 24]
    archs = ["feedforward", "residual", "layernorm", "rmsnorm",
             "pre_ln", "pre_rmsn", "post_ln"]

    for arch in archs:
        print(f"\n  Architecture: {arch}")
        print(f"  {'L':>4} {'Seed':>6} {'SumLayErr%':>12} "
              f"{'MeanOutDev%':>13} {'AmpRatio':>10} {'OutScale':>10}")
        print("  " + "-" * 65)

        for L in L_values:
            for seed in seeds:
                res = run_removal_cascade(
                    N, L, d, r, cluster_cos=None, act="gelu",
                    n_inputs=n_inputs, remove_idx=N // 2,
                    seed=seed, arch=arch
                )
                res["test"] = "depth_arch"
                all_results.append(res)

                amp_str = f"{res['amplification_ratio']:>10.4f}" if not res["diverged"] else "  DIVERGED"
                dev_str = f"{res['mean_output_dev_pct']:>13.4f}" if not res["diverged"] else "     DIVERGED"
                scale_str = f"{res['output_scale']:>10.4f}" if not res["diverged"] else "  DIVERGED"
                print(f"  {L:>4} {seed:>6} {res['sum_per_layer_error']:>12.4f} "
                      f"{dev_str} {amp_str} {scale_str}")

    # ================================================================
    # TEST 2: Dimension scaling per architecture (K2 test)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: Dimension scaling per architecture (K2: 1/d scaling?)")
    print("  N=8, L=24, r=8, GELU, near-orthogonal")
    print("=" * 80)

    d_values = [32, 64, 128, 256]
    L = 24
    key_archs = ["feedforward", "residual", "pre_ln", "pre_rmsn", "post_ln"]

    for arch in key_archs:
        print(f"\n  Architecture: {arch}")
        print(f"  {'d':>5} {'Seed':>6} {'SumLayErr%':>12} "
              f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
        print("  " + "-" * 55)

        for d_val in d_values:
            for seed in seeds:
                res = run_removal_cascade(
                    N, L, d_val, r, cluster_cos=None, act="gelu",
                    n_inputs=n_inputs, remove_idx=N // 2,
                    seed=seed, arch=arch
                )
                res["test"] = "dim_scaling"
                all_results.append(res)

                amp_str = f"{res['amplification_ratio']:>10.4f}" if not res["diverged"] else "  DIVERGED"
                dev_str = f"{res['mean_output_dev_pct']:>13.4f}" if not res["diverged"] else "     DIVERGED"
                print(f"  {d_val:>5} {seed:>6} {res['sum_per_layer_error']:>12.4f} "
                      f"{dev_str} {amp_str}")

    # ================================================================
    # TEST 3: Clustered cosines (cos~0.3) across architectures
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 3: Clustered experts (cos~0.3) across architectures")
    print("  d=64, r=8, N=8, L=24, GELU")
    print("=" * 80)

    L = 24

    print(f"\n  {'Arch':>12} {'Seed':>6} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("  " + "-" * 60)

    for arch in key_archs:
        for seed in seeds:
            res = run_removal_cascade(
                N, L, 64, r, cluster_cos=0.3, act="gelu",
                n_inputs=n_inputs, remove_idx=N // 2,
                seed=seed, arch=arch
            )
            res["test"] = "clustered"
            all_results.append(res)

            amp_str = f"{res['amplification_ratio']:>10.4f}" if not res["diverged"] else "  DIVERGED"
            dev_str = f"{res['mean_output_dev_pct']:>13.4f}" if not res["diverged"] else "     DIVERGED"
            print(f"  {arch:>12} {seed:>6} {res['sum_per_layer_error']:>12.4f} "
                  f"{dev_str} {amp_str}")

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("  AGGREGATE ANALYSIS")
    print("=" * 80)

    # --- Depth scaling comparison ---
    depth_results = [r for r in all_results if r["test"] == "depth_arch" and not r["diverged"]]

    print("\n  Mean amplification ratio at L=24, by architecture (near-orthogonal):")
    print(f"  {'Architecture':>12} {'MeanAmpRatio':>14} {'StdAmpRatio':>13} "
          f"{'MeanOutDev%':>13}")
    print("  " + "-" * 58)

    parent_amp_L24 = None
    arch_amp_L24 = {}

    for arch in archs:
        L24 = [r for r in depth_results if r["arch"] == arch and r["L"] == 24]
        if not L24:
            continue
        amps = [r["amplification_ratio"] for r in L24]
        devs = [r["mean_output_dev_pct"] for r in L24]
        mean_amp = np.mean(amps)
        std_amp = np.std(amps)
        mean_dev = np.mean(devs)
        arch_amp_L24[arch] = (mean_amp, std_amp)

        if arch == "feedforward":
            parent_amp_L24 = mean_amp

        print(f"  {arch:>12} {mean_amp:>14.4f} {std_amp:>13.4f} {mean_dev:>13.4f}")

    # --- K1 Assessment: does residual change amp_ratio by >50%? ---
    print("\n" + "-" * 80)
    print("  K1 ASSESSMENT: Does residual change amp_ratio by >50%?")
    print("-" * 80)

    if parent_amp_L24 is not None:
        for arch in archs:
            if arch == "feedforward" or arch not in arch_amp_L24:
                continue
            mean_amp, std_amp = arch_amp_L24[arch]
            pct_change = abs(mean_amp - parent_amp_L24) / parent_amp_L24 * 100
            direction = "higher" if mean_amp > parent_amp_L24 else "lower"
            triggered = pct_change > 50
            verdict = "TRIGGERED (dynamics different)" if triggered else "NOT triggered (similar dynamics)"
            print(f"\n  {arch}: amp_ratio = {mean_amp:.4f} vs feedforward = {parent_amp_L24:.4f}")
            print(f"    Change: {pct_change:.1f}% {direction}")
            print(f"    K1 verdict: {verdict}")

    # --- K2 Assessment: does LayerNorm break 1/d scaling? ---
    print("\n" + "-" * 80)
    print("  K2 ASSESSMENT: Does LayerNorm break 1/d scaling?")
    print("-" * 80)

    dim_results = [r for r in all_results if r["test"] == "dim_scaling" and not r["diverged"]]

    for arch in key_archs:
        arch_dim = [r for r in dim_results if r["arch"] == arch]
        if len(arch_dim) < 6:
            continue

        # Group by d, compute mean output dev
        d_devs = {}
        for r_res in arch_dim:
            d_val = r_res["d"]
            if d_val not in d_devs:
                d_devs[d_val] = []
            d_devs[d_val].append(r_res["mean_output_dev_pct"])

        ds = sorted(d_devs.keys())
        mean_devs = [np.mean(d_devs[d_val]) for d_val in ds]

        # Fit power law: dev = C * d^alpha
        # log(dev) = log(C) + alpha * log(d)
        log_ds = np.log(np.array(ds, dtype=float))
        log_devs = np.log(np.array(mean_devs) + 1e-12)

        if len(log_ds) >= 3 and np.all(np.isfinite(log_devs)):
            slope, intercept, r_val, p_val, se = stats.linregress(log_ds, log_devs)
            C = np.exp(intercept)
            print(f"\n  {arch}: dev(d) = {C:.2f} * d^({slope:.3f})")
            print(f"    R^2 = {r_val**2:.4f}, p = {p_val:.6f}")
            print(f"    Measured devs: {dict(zip(ds, [f'{v:.3f}' for v in mean_devs]))}")

            if abs(slope) < 0.3:
                print(f"    -> WEAK d-dependence (exponent {slope:.3f}, near 0)")
                print(f"    -> K2 potentially TRIGGERED: error NOT scaling as 1/d")
            elif slope < -0.5:
                print(f"    -> STRONG 1/d scaling (exponent {slope:.3f})")
                print(f"    -> K2 NOT triggered: 1/d scaling preserved")
            else:
                print(f"    -> MODERATE d-dependence (exponent {slope:.3f})")
        else:
            print(f"\n  {arch}: insufficient data or NaN for regression")

    # --- Depth scaling regression per architecture ---
    print("\n" + "-" * 80)
    print("  DEPTH SCALING: amp_ratio vs L regression")
    print("-" * 80)

    for arch in archs:
        arch_depth = [r for r in depth_results if r["arch"] == arch]
        if len(arch_depth) < 6:
            continue

        Ls = [r["L"] for r in arch_depth]
        amps = [r["amplification_ratio"] for r in arch_depth]

        slope, intercept, r_val, p_val, se = stats.linregress(Ls, amps)
        print(f"\n  {arch}:")
        print(f"    amp_ratio = {slope:.6f} * L + {intercept:.4f}")
        print(f"    R^2 = {r_val**2:.4f}, p = {p_val:.6f}")
        if slope > 0 and p_val < 0.05:
            print(f"    -> Amplification INCREASES with depth")
        elif slope < 0 and p_val < 0.05:
            print(f"    -> Amplification DECREASES with depth (sub-additive)")
        else:
            print(f"    -> No significant trend")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE: All architectures at L=24, d=64")
    print("=" * 80)

    print(f"\n  {'Architecture':>12} {'AmpRatio':>10} {'MeanDev%':>10} {'MaxDev%':>10} "
          f"{'vs FF (%)':>10} {'K1':>20}")
    print("  " + "-" * 78)

    for arch in archs:
        L24 = [r for r in depth_results if r["arch"] == arch and r["L"] == 24]
        if not L24:
            continue
        mean_amp = np.mean([r["amplification_ratio"] for r in L24])
        mean_dev = np.mean([r["mean_output_dev_pct"] for r in L24])
        max_dev = max([r["max_output_dev_pct"] for r in L24])

        if parent_amp_L24 and arch != "feedforward":
            pct = abs(mean_amp - parent_amp_L24) / parent_amp_L24 * 100
            k1 = "TRIGGERED" if pct > 50 else "not triggered"
            pct_str = f"{pct:>10.1f}"
        else:
            pct_str = "  baseline"
            k1 = "baseline"

        print(f"  {arch:>12} {mean_amp:>10.4f} {mean_dev:>10.4f} {max_dev:>10.4f} "
              f"{pct_str} {k1:>20}")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    # Convert NaN to None for JSON
    clean_results = []
    for r_res in all_results:
        clean = {}
        for k, v in r_res.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        clean_results.append(clean)

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    run_full_experiment()
