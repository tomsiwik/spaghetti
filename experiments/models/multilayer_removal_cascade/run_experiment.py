"""Multi-layer Removal Cascade: Does per-layer error compound through depth?

Parent experiment (expert_removal_graceful) showed:
  - At SOLE cosines (cos~0.0002): naive subtraction error 0.18% per layer
  - At clustered cosines (cos~0.3): error 7-10% per layer

But real models have L=24+ layers. If per-layer error compounds multiplicatively
through the nonlinear forward pass, 0.18% * L could become significant.

This experiment builds a synthetic L-layer model where each layer is:
  h_{l+1} = activation(W_l @ h_l)

with LoRA experts applied at each layer:
  h_{l+1} = activation((W_l + sum delta_i_l') @ h_l)

We remove one expert across all layers and measure:
  1. Per-layer reconstruction error (weight space, inherited from parent)
  2. End-to-end output deviation (function space, NEW)
  3. Whether the ratio (end-to-end / sum-of-per-layer) is > 1 (multiplicative)
     or <= 1 (sub-additive)

Kill criteria:
  K1: cumulative removal error across L layers exceeds 3% PPL regression
      (measured as relative output deviation). If so, amplification is real.
  K2: per-layer error does NOT compound (stays additive, not multiplicative).
      If error is strictly additive, K2 is killed.

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Synthetic Multi-Layer Model
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single LoRA expert for one layer.

    Returns dict with A (d, r), B (r, d), dW = A @ B shape (d, d).
    """
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


def generate_multilayer_experts(N: int, L: int, d: int, r: int,
                                 cluster_cos: float | None = None,
                                 seed: int = 42) -> list[list[dict]]:
    """Generate N experts, each with L layer-wise LoRA deltas.

    Returns: experts[expert_idx][layer_idx] = {"A", "B", "dW"}

    If cluster_cos is not None, creates clustered structure within each layer.
    """
    rng = np.random.RandomState(seed)

    if cluster_cos is None:
        # Purely random experts -- near-orthogonal at high d
        experts = []
        for i in range(N):
            layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
            experts.append(layers)
        return experts

    # Clustered: generate a shared direction per layer, mix with per-expert unique
    alpha = np.sqrt(cluster_cos)
    beta = np.sqrt(1.0 - cluster_cos)

    # Generate shared directions per layer
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
            dW = dW * rng.uniform(0.8, 1.2) * 0.01  # scale to reasonable magnitude
            layers.append({"dW": dW})
        experts.append(layers)
    return experts


def gram_schmidt_merge(deltas: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """GS-orthogonalize then sum flattened deltas.

    Returns: (merged_sum, list_of_ortho_deltas)
    """
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


def activation(x: np.ndarray, kind: str = "relu") -> np.ndarray:
    """Activation function for synthetic model."""
    if kind == "relu":
        return np.maximum(0, x)
    elif kind == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    elif kind == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {kind}")


# ============================================================================
# Forward Pass Through Multi-Layer Model
# ============================================================================

def forward_multilayer(h: np.ndarray, base_weights: list[np.ndarray],
                       layer_deltas: list[np.ndarray],
                       act: str = "gelu") -> np.ndarray:
    """Forward pass: h_{l+1} = activation((W_l + delta_l) @ h_l).

    Args:
        h: input (d,) or (batch, d)
        base_weights: list of L weight matrices, each (d, d)
        layer_deltas: list of L merged delta matrices, each (d, d)
        act: activation function

    Returns: final hidden state
    """
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = activation(W @ h, kind=act)
    return h


def compute_per_layer_cosines(experts: list[list[dict]], L: int) -> np.ndarray:
    """Compute mean |cos| between all expert pairs, per layer.

    Returns: (L,) array of mean absolute cosine similarities.
    """
    N = len(experts)
    per_layer_cos = np.zeros(L)

    for l in range(L):
        cos_vals = []
        for i in range(N):
            for j in range(i + 1, N):
                di = experts[i][l]["dW"].flatten()
                dj = experts[j][l]["dW"].flatten()
                cos = abs(np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj) + 1e-12))
                cos_vals.append(cos)
        per_layer_cos[l] = np.mean(cos_vals)

    return per_layer_cos


# ============================================================================
# Core Experiment: Removal Error Amplification
# ============================================================================

def run_removal_cascade(N: int, L: int, d: int, r: int,
                         cluster_cos: float | None, act: str,
                         n_inputs: int, remove_idx: int,
                         seed: int) -> dict:
    """Run one configuration of the multi-layer removal cascade experiment.

    Measures:
    1. Per-layer weight-space reconstruction error (naive subtraction vs GS recompute)
    2. End-to-end output deviation with vs without the removed expert
    3. Amplification ratio = output deviation / sum(per-layer errors)
    """
    rng = np.random.RandomState(seed)

    # Generate experts
    experts = generate_multilayer_experts(N, L, d, r, cluster_cos, seed=seed)

    # Generate base weights (random, scaled to avoid vanishing/exploding)
    base_weights = []
    for l in range(L):
        W = rng.randn(d, d) / np.sqrt(d)
        base_weights.append(W)

    # Generate test inputs
    inputs = rng.randn(n_inputs, d) * 0.1  # small inputs to keep activations bounded

    # ================================================================
    # Step 1: Merge all N experts per layer via GS
    # ================================================================
    all_merged_deltas = []  # (L,) list of merged delta (d, d)
    all_ortho_deltas = []   # (L,) list of [N] ortho delta vectors
    per_layer_recon_errors = []

    for l in range(L):
        # Get deltas for this layer, flatten for GS
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]

        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Per-layer naive removal error
        # Naive: subtract ortho[remove_idx]
        naive_flat = merged_flat - ortho_flat[remove_idx]

        # Ground truth: GS recompute on N-1
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        # Relative reconstruction error
        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_recon_errors.append(recon_err)

    # ================================================================
    # Step 2: Forward pass with ALL experts
    # ================================================================
    outputs_all = []
    for inp in inputs:
        out = forward_multilayer(inp, base_weights, all_merged_deltas, act=act)
        outputs_all.append(out)
    outputs_all = np.array(outputs_all)

    # ================================================================
    # Step 3: Forward pass AFTER naive removal (subtract ortho delta at each layer)
    # ================================================================
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = []
    for inp in inputs:
        out = forward_multilayer(inp, base_weights, naive_removed_deltas, act=act)
        outputs_naive.append(out)
    outputs_naive = np.array(outputs_naive)

    # ================================================================
    # Step 4: Forward pass with GS recompute (ground truth removal)
    # ================================================================
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = []
    for inp in inputs:
        out = forward_multilayer(inp, base_weights, gt_removed_deltas, act=act)
        outputs_gt.append(out)
    outputs_gt = np.array(outputs_gt)

    # ================================================================
    # Step 5: Forward pass with base model only (no experts -- reference)
    # ================================================================
    zero_deltas = [np.zeros((d, d)) for _ in range(L)]
    outputs_base = []
    for inp in inputs:
        out = forward_multilayer(inp, base_weights, zero_deltas, act=act)
        outputs_base.append(out)
    outputs_base = np.array(outputs_base)

    # ================================================================
    # Metrics
    # ================================================================

    # A. Per-layer reconstruction errors (weight space)
    sum_per_layer = sum(per_layer_recon_errors)
    max_per_layer = max(per_layer_recon_errors)
    mean_per_layer = np.mean(per_layer_recon_errors)

    # B. End-to-end output deviation: naive vs GT removal
    # Relative output deviation per input, then aggregate
    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    # Avoid division by zero
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_output_dev = float(np.mean(relative_devs))
    max_output_dev = float(np.max(relative_devs))
    median_output_dev = float(np.median(relative_devs))

    # C. Amplification ratio: output deviation vs sum of per-layer weight errors
    # If this ratio > 1: errors amplify multiplicatively through layers
    # If <= 1: errors are sub-additive (layers dampen)
    if sum_per_layer > 1e-12:
        amplification_ratio = mean_output_dev / sum_per_layer
    else:
        amplification_ratio = 0.0

    # D. Also compare: naive vs GT removal output, normalized by expert signal
    # Expert signal = how much the removed expert changes output
    all_vs_gt_norms = np.linalg.norm(outputs_all - outputs_gt, axis=1)
    expert_signal = float(np.mean(all_vs_gt_norms / safe_gt * 100.0))

    # E. Naive vs GT relative to total output magnitude (absolute scale)
    output_scale = float(np.mean(gt_norms))

    # F. Per-layer cosine similarities
    per_layer_cos = compute_per_layer_cosines(experts, L)

    return {
        "N": N, "L": L, "d": d, "r": r,
        "cluster_cos": cluster_cos,
        "activation": act,
        "n_inputs": n_inputs,
        "remove_idx": remove_idx,
        "seed": seed,
        "mean_pairwise_cos": float(np.mean(per_layer_cos)),
        # Weight-space metrics
        "per_layer_recon_errors": [float(x) for x in per_layer_recon_errors],
        "sum_per_layer_error": sum_per_layer,
        "max_per_layer_error": max_per_layer,
        "mean_per_layer_error": mean_per_layer,
        # Output-space metrics
        "mean_output_dev_pct": mean_output_dev,
        "max_output_dev_pct": max_output_dev,
        "median_output_dev_pct": median_output_dev,
        # Key ratio
        "amplification_ratio": amplification_ratio,
        # Context
        "expert_signal_pct": expert_signal,
        "output_scale": output_scale,
    }


# ============================================================================
# Experiment Runner
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 75)
    print("  EXPERIMENT: Multi-layer Removal Cascade")
    print("  Kill: K1 cumulative error > 3% | K2 error stays additive")
    print("=" * 75)

    d = 64       # Smaller than parent (896) for speed with L layers
    r = 8        # LoRA rank
    n_inputs = 200
    seeds = [42, 123, 777]

    all_results = []

    # ================================================================
    # TEST 1: Depth scaling at SOLE-like cosines (random, near-orthogonal)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 1: Depth scaling — near-orthogonal experts (random)")
    print("  d=64, r=8, N=8 experts, GELU activation")
    print("=" * 75)

    N = 8
    L_values = [1, 2, 4, 8, 12, 16, 24]
    act = "gelu"

    print(f"\n{'L':>4} {'Seed':>6} {'MeanCos':>9} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10} {'ExpertSig%':>12}")
    print("-" * 80)

    for L in L_values:
        for seed in seeds:
            res = run_removal_cascade(N, L, d, r, cluster_cos=None,
                                       act=act, n_inputs=n_inputs,
                                       remove_idx=N // 2, seed=seed)
            res["test"] = "depth_ortho"
            all_results.append(res)
            print(f"{L:>4} {seed:>6} {res['mean_pairwise_cos']:>9.6f} "
                  f"{res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f} "
                  f"{res['expert_signal_pct']:>12.4f}")

    # ================================================================
    # TEST 2: Depth scaling at clustered cosines (cos~0.3)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 2: Depth scaling — clustered experts (cos~0.3)")
    print("  d=64, r=8, N=8 experts, GELU activation")
    print("=" * 75)

    print(f"\n{'L':>4} {'Seed':>6} {'MeanCos':>9} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10} {'ExpertSig%':>12}")
    print("-" * 80)

    for L in L_values:
        for seed in seeds:
            res = run_removal_cascade(N, L, d, r, cluster_cos=0.3,
                                       act=act, n_inputs=n_inputs,
                                       remove_idx=N // 2, seed=seed)
            res["test"] = "depth_clustered"
            all_results.append(res)
            print(f"{L:>4} {seed:>6} {res['mean_pairwise_cos']:>9.6f} "
                  f"{res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f} "
                  f"{res['expert_signal_pct']:>12.4f}")

    # ================================================================
    # TEST 3: Activation function comparison (linear vs GELU vs ReLU)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 3: Activation function comparison at L=24")
    print("  d=64, r=8, N=8, L=24, near-orthogonal")
    print("=" * 75)

    L = 24
    activations = ["linear", "relu", "gelu"]

    print(f"\n{'Act':>8} {'Seed':>6} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 55)

    for act in activations:
        for seed in seeds:
            res = run_removal_cascade(N, L, d, r, cluster_cos=None,
                                       act=act, n_inputs=n_inputs,
                                       remove_idx=N // 2, seed=seed)
            res["test"] = "activation"
            all_results.append(res)
            print(f"{act:>8} {seed:>6} {res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 4: Scaling d (check if higher d reduces amplification)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 4: Dimension scaling — does higher d reduce amplification?")
    print("  N=8, L=24, r=8, GELU, near-orthogonal")
    print("=" * 75)

    d_values = [32, 64, 128, 256]
    L = 24

    print(f"\n{'d':>5} {'Seed':>6} {'MeanCos':>9} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 60)

    for d_val in d_values:
        for seed in seeds:
            res = run_removal_cascade(N, L, d_val, r, cluster_cos=None,
                                       act="gelu", n_inputs=n_inputs,
                                       remove_idx=N // 2, seed=seed)
            res["test"] = "dimension_scaling"
            all_results.append(res)
            print(f"{d_val:>5} {seed:>6} {res['mean_pairwise_cos']:>9.6f} "
                  f"{res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 5: N scaling (more experts, same depth)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 5: Expert count scaling")
    print("  d=64, L=24, r=8, GELU, near-orthogonal")
    print("=" * 75)

    N_values = [4, 8, 16, 32]
    L = 24

    print(f"\n{'N':>4} {'Seed':>6} {'MeanCos':>9} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 60)

    for N_val in N_values:
        for seed in seeds:
            res = run_removal_cascade(N_val, L, 64, r, cluster_cos=None,
                                       act="gelu", n_inputs=n_inputs,
                                       remove_idx=N_val // 2, seed=seed)
            res["test"] = "N_scaling"
            all_results.append(res)
            print(f"{N_val:>4} {seed:>6} {res['mean_pairwise_cos']:>9.6f} "
                  f"{res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 6: Position sensitivity in multi-layer context
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 6: Removal position sensitivity (L=24, N=8)")
    print("  d=64, r=8, GELU, clustered cos=0.3")
    print("=" * 75)

    N = 8
    L = 24
    positions = [0, N // 4, N // 2, 3 * N // 4, N - 1]

    print(f"\n{'Pos':>4} {'Seed':>6} {'SumLayErr%':>12} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 50)

    for pos in positions:
        for seed in seeds:
            res = run_removal_cascade(N, L, 64, r, cluster_cos=0.3,
                                       act="gelu", n_inputs=n_inputs,
                                       remove_idx=pos, seed=seed)
            res["test"] = "position"
            all_results.append(res)
            print(f"{pos:>4} {seed:>6} {res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 75)
    print("  AGGREGATE ANALYSIS")
    print("=" * 75)

    # Analyze depth scaling for orthogonal experts
    depth_ortho = [r for r in all_results if r["test"] == "depth_ortho"]
    depth_clustered = [r for r in all_results if r["test"] == "depth_clustered"]

    for label, results in [("Near-orthogonal", depth_ortho),
                            ("Clustered cos~0.3", depth_clustered)]:
        print(f"\n  {label} — Depth scaling summary:")
        print(f"  {'L':>4} {'MeanSumErr%':>13} {'MeanOutDev%':>13} "
              f"{'MeanAmpRatio':>14} {'MaxOutDev%':>12}")
        print("  " + "-" * 60)

        for L in L_values:
            L_results = [r for r in results if r["L"] == L]
            if not L_results:
                continue
            mean_sum = np.mean([r["sum_per_layer_error"] for r in L_results])
            mean_out = np.mean([r["mean_output_dev_pct"] for r in L_results])
            mean_amp = np.mean([r["amplification_ratio"] for r in L_results])
            max_out = max([r["max_output_dev_pct"] for r in L_results])
            print(f"  {L:>4} {mean_sum:>13.4f} {mean_out:>13.4f} "
                  f"{mean_amp:>14.4f} {max_out:>12.4f}")

    # Key question: does amplification ratio increase with depth?
    print("\n  KEY QUESTION: Does amplification ratio increase with L?")
    for label, results in [("Near-orthogonal", depth_ortho),
                            ("Clustered cos~0.3", depth_clustered)]:
        L_list = []
        amp_list = []
        for r_res in results:
            L_list.append(r_res["L"])
            amp_list.append(r_res["amplification_ratio"])

        if len(L_list) > 2:
            slope, intercept, r_val, p_val, se = stats.linregress(L_list, amp_list)
            print(f"\n  {label}:")
            print(f"    Linear regression: amp_ratio = {slope:.6f} * L + {intercept:.4f}")
            print(f"    R^2 = {r_val**2:.4f}, p = {p_val:.4f}")
            if slope > 0 and p_val < 0.05:
                print(f"    -> Amplification INCREASES with depth (significant)")
            elif slope <= 0:
                print(f"    -> Amplification does NOT increase with depth")
            else:
                print(f"    -> Trend not statistically significant")

    # K1 assessment: max output deviation at L=24
    L24_ortho = [r for r in depth_ortho if r["L"] == 24]
    L24_clustered = [r for r in depth_clustered if r["L"] == 24]

    print("\n" + "-" * 75)
    print("  KILL CRITERIA ASSESSMENT")
    print("-" * 75)

    if L24_ortho:
        max_dev_ortho = max([r["max_output_dev_pct"] for r in L24_ortho])
        mean_dev_ortho = np.mean([r["mean_output_dev_pct"] for r in L24_ortho])
        print(f"\n  Near-orthogonal at L=24:")
        print(f"    Mean output deviation: {mean_dev_ortho:.4f}%")
        print(f"    Max output deviation:  {max_dev_ortho:.4f}%")

    if L24_clustered:
        max_dev_clust = max([r["max_output_dev_pct"] for r in L24_clustered])
        mean_dev_clust = np.mean([r["mean_output_dev_pct"] for r in L24_clustered])
        print(f"\n  Clustered cos~0.3 at L=24:")
        print(f"    Mean output deviation: {mean_dev_clust:.4f}%")
        print(f"    Max output deviation:  {max_dev_clust:.4f}%")

    # K1: cumulative error > 3%?
    if L24_ortho:
        k1_ortho = max_dev_ortho > 3.0
        print(f"\n  K1 (ortho, >3% cumulative): {'YES (amplification real)' if k1_ortho else 'NO (safe)'}")
        print(f"      Max output deviation at L=24: {max_dev_ortho:.4f}%")

    if L24_clustered:
        k1_clust = max_dev_clust > 3.0
        print(f"\n  K1 (clustered, >3% cumulative): {'YES (amplification real)' if k1_clust else 'NO (safe)'}")
        print(f"      Max output deviation at L=24: {max_dev_clust:.4f}%")

    # K2: error additive or multiplicative?
    # Compare amplification ratios across depths
    # If amp_ratio ~ 1 for all L: additive. If amp_ratio >> 1 at high L: multiplicative.
    all_amps_ortho = [r["amplification_ratio"] for r in depth_ortho]
    all_amps_clust = [r["amplification_ratio"] for r in depth_clustered]

    if all_amps_ortho:
        mean_amp_ortho = np.mean(all_amps_ortho)
        max_amp_ortho = max(all_amps_ortho)
        print(f"\n  K2 (ortho, additive vs multiplicative):")
        print(f"    Mean amplification ratio: {mean_amp_ortho:.4f}")
        print(f"    Max amplification ratio:  {max_amp_ortho:.4f}")
        if max_amp_ortho <= 2.0:
            print(f"    -> Error is SUB-ADDITIVE or WEAKLY SUPER-ADDITIVE (amp <= 2x)")
            print(f"    -> K2 verdict: additive regime (error does NOT compound dangerously)")
        elif max_amp_ortho <= 5.0:
            print(f"    -> Error shows MODERATE amplification (amp 2-5x)")
        else:
            print(f"    -> Error shows STRONG amplification (amp > 5x)")

    if all_amps_clust:
        mean_amp_clust = np.mean(all_amps_clust)
        max_amp_clust = max(all_amps_clust)
        print(f"\n  K2 (clustered, additive vs multiplicative):")
        print(f"    Mean amplification ratio: {mean_amp_clust:.4f}")
        print(f"    Max amplification ratio:  {max_amp_clust:.4f}")
        if max_amp_clust <= 2.0:
            print(f"    -> Error is SUB-ADDITIVE or WEAKLY SUPER-ADDITIVE (amp <= 2x)")
        elif max_amp_clust <= 5.0:
            print(f"    -> Error shows MODERATE amplification (amp 2-5x)")
        else:
            print(f"    -> Error shows STRONG amplification (amp > 5x)")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    run_full_experiment()
