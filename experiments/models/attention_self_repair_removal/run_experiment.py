"""Attention Self-Repair After Expert Removal

Does attention provide self-repair when a LoRA expert is removed?

Parent experiments:
  - residual_layernorm_error_dynamics: Pre-RMSNorm amp_ratio=0.022 (MLP-only)
  - attention_layer_removal_safety: GS(attn)+naive(MLP) hybrid strategy proven
  - correlated_layer_errors: correlation REDUCES amplification (amp_ratio=0.074)

Key mechanism to test: Attention's softmax creates data-dependent routing.
After expert removal, the perturbation delta propagates through attention.
Softmax could either:
  (a) Amplify: if perturbation aligns with dominant attention directions
  (b) Redistribute: attention re-routes around perturbed subspace (self-repair)

McGill et al. (2024) documented "self-repair" in transformers: when attention
heads are ablated, remaining heads compensate. This experiment tests whether
the same effect applies to LoRA expert removal.

Kill criteria:
  K1: transformer with attention shows >30% lower output deviation than
      MLP-only at same depth/dimension after expert removal
  K2: self-repair effect INCREASES with depth (attention redistributes
      around perturbation over layers)

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Core utilities (reused from parent experiments)
# ============================================================================

def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm without learnable scale (Qwen/Llama style)."""
    if x.ndim == 1:
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return x / rms
    else:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms


def activation(x: np.ndarray, kind: str = "gelu") -> np.ndarray:
    """Activation function."""
    if kind == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    elif kind == "relu":
        return np.maximum(0, x)
    else:
        raise ValueError(f"Unknown activation: {kind}")


def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a single LoRA expert delta (A @ B) for one layer."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return A @ B  # shape (d, d)


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
# Model architectures
# ============================================================================

class MLPOnlyModel:
    """Pre-RMSNorm residual MLP-only model.

    Each layer: h_{l+1} = h_l + (1/sqrt(L)) * gelu((W_l + Delta_l) @ RN(h_l))

    This is the baseline from residual_layernorm_error_dynamics.
    """

    def __init__(self, L: int, d: int, seed: int):
        self.L = L
        self.d = d
        rng = np.random.RandomState(seed)
        # Base MLP weights
        self.W = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        self.scale = 1.0 / np.sqrt(L) if L > 1 else 1.0

    def forward(self, h: np.ndarray, deltas: list[np.ndarray]) -> np.ndarray:
        """h: (batch, d) or (d,). deltas: list of L (d, d) matrices."""
        single = h.ndim == 1
        if single:
            h = h[np.newaxis, :]  # (1, d)

        for l in range(self.L):
            W_eff = self.W[l] + deltas[l]
            h_norm = rms_norm(h)  # (batch, d)
            branch = activation(h_norm @ W_eff.T)  # (batch, d)
            h = h + self.scale * branch

        return h[0] if single else h


class TransformerModel:
    """Pre-RMSNorm residual Transformer model with multi-head self-attention.

    Each layer has:
      1. Attention sub-layer:  h = h + (1/sqrt(2L)) * Attn(RN(h))
      2. MLP sub-layer:       h = h + (1/sqrt(2L)) * gelu((W_l + Delta_l) @ RN(h))

    Attention uses random (frozen) Q, K, V, O projections.
    Expert deltas apply ONLY to the MLP sub-layer (matching SOLE architecture
    where LoRA experts modify weight matrices, and attention is frozen base).
    """

    def __init__(self, L: int, d: int, n_heads: int, seed: int):
        self.L = L
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"

        rng = np.random.RandomState(seed)

        # Attention weights (frozen, random init matching base model)
        self.Wq = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        self.Wk = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        self.Wv = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        self.Wo = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]

        # MLP weights (base + expert deltas applied here)
        self.W = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]

        # Scale: 2L sub-layers total (attn + MLP per layer)
        self.scale = 1.0 / np.sqrt(2 * L) if L > 1 else 1.0

    def attention(self, h: np.ndarray, l: int) -> np.ndarray:
        """Multi-head self-attention.

        h: (seq, d) -- we treat batch dimension as sequence for attention.
        Returns: (seq, d)
        """
        seq_len = h.shape[0]
        d = self.d
        hd = self.head_dim
        nh = self.n_heads

        # Project Q, K, V
        Q = h @ self.Wq[l].T  # (seq, d)
        K = h @ self.Wk[l].T  # (seq, d)
        V = h @ self.Wv[l].T  # (seq, d)

        # Reshape for multi-head: (seq, nh, hd)
        Q = Q.reshape(seq_len, nh, hd)
        K = K.reshape(seq_len, nh, hd)
        V = V.reshape(seq_len, nh, hd)

        # Attention scores: (nh, seq, seq)
        # scores[h, i, j] = Q[i, h, :] . K[j, h, :] / sqrt(hd)
        scores = np.einsum('shr,thr->hst', Q, K) / np.sqrt(hd)

        # Softmax per head, per query position
        # Subtract max for numerical stability
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-12)

        # Weighted sum of values: (nh, seq, hd)
        context = np.einsum('hst,thr->shr', attn_weights, V)

        # Reshape back: (seq, d)
        context = context.reshape(seq_len, d)

        # Output projection
        output = context @ self.Wo[l].T  # (seq, d)
        return output

    def forward(self, h: np.ndarray, deltas: list[np.ndarray]) -> np.ndarray:
        """h: (batch, d). deltas: list of L (d, d) matrices for MLP sub-layer."""
        single = h.ndim == 1
        if single:
            h = h[np.newaxis, :]

        for l in range(self.L):
            # Attention sub-layer: h = h + scale * Attn(RN(h))
            h_norm = rms_norm(h)
            attn_out = self.attention(h_norm, l)
            h = h + self.scale * attn_out

            # MLP sub-layer: h = h + scale * gelu((W + Delta) @ RN(h))
            h_norm = rms_norm(h)
            W_eff = self.W[l] + deltas[l]
            mlp_out = activation(h_norm @ W_eff.T)
            h = h + self.scale * mlp_out

        return h[0] if single else h


# ============================================================================
# Expert removal experiment
# ============================================================================

def run_removal_comparison(
    N: int, L: int, d: int, r: int, n_heads: int,
    n_inputs: int, seed: int
) -> dict:
    """Run one configuration comparing MLP-only vs Transformer after expert removal.

    Returns metrics for both architectures.
    """
    rng = np.random.RandomState(seed)

    # Generate N experts, each with L layer-wise MLP deltas
    experts = []
    for i in range(N):
        layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
        experts.append(layers)

    # Generate random test inputs
    inputs = rng.randn(n_inputs, d) * 0.1  # (n_inputs, d)

    # Build both models (same seed for comparable base weights for MLP part)
    mlp_model = MLPOnlyModel(L, d, seed=seed + 10000)
    tf_model = TransformerModel(L, d, n_heads, seed=seed + 10000)

    # Copy MLP weights from mlp_model to transformer's MLP sub-layer
    # so the ONLY difference is the presence of attention
    for l in range(L):
        tf_model.W[l] = mlp_model.W[l].copy()

    # Remove expert index
    remove_idx = N // 2

    # Merge all N experts per layer via GS
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_recon_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Per-layer naive removal error
        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_recon_errors.append(recon_err)

    sum_per_layer_err = sum(per_layer_recon_errors)

    # Compute ground-truth removal deltas
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    # Naive removal deltas
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    # ---- MLP-only model ----
    out_mlp_all = mlp_model.forward(inputs, all_merged_deltas)
    out_mlp_gt = mlp_model.forward(inputs, gt_removed_deltas)
    out_mlp_naive = mlp_model.forward(inputs, naive_removed_deltas)

    # ---- Transformer model ----
    out_tf_all = tf_model.forward(inputs, all_merged_deltas)
    out_tf_gt = tf_model.forward(inputs, gt_removed_deltas)
    out_tf_naive = tf_model.forward(inputs, naive_removed_deltas)

    # ---- Metrics ----
    def compute_deviation(out_naive, out_gt):
        """Relative output deviation: ||naive - gt|| / ||gt|| * 100."""
        diff_norms = np.linalg.norm(out_naive - out_gt, axis=-1)
        gt_norms = np.maximum(np.linalg.norm(out_gt, axis=-1), 1e-12)
        relative = diff_norms / gt_norms * 100.0
        return {
            "mean_dev_pct": float(np.mean(relative)),
            "max_dev_pct": float(np.max(relative)),
            "median_dev_pct": float(np.median(relative)),
            "std_dev_pct": float(np.std(relative)),
        }

    def compute_signal(out_all, out_gt):
        """Expert signal: how much the removed expert matters."""
        diff_norms = np.linalg.norm(out_all - out_gt, axis=-1)
        gt_norms = np.maximum(np.linalg.norm(out_gt, axis=-1), 1e-12)
        return float(np.mean(diff_norms / gt_norms * 100.0))

    mlp_dev = compute_deviation(out_mlp_naive, out_mlp_gt)
    tf_dev = compute_deviation(out_tf_naive, out_tf_gt)

    mlp_signal = compute_signal(out_mlp_all, out_mlp_gt)
    tf_signal = compute_signal(out_tf_all, out_tf_gt)

    # Amplification ratios
    mlp_amp = mlp_dev["mean_dev_pct"] / sum_per_layer_err if sum_per_layer_err > 1e-12 else 0.0
    tf_amp = tf_dev["mean_dev_pct"] / sum_per_layer_err if sum_per_layer_err > 1e-12 else 0.0

    # Self-repair ratio: how much lower is transformer deviation vs MLP-only
    if mlp_dev["mean_dev_pct"] > 1e-12:
        repair_ratio = 1.0 - tf_dev["mean_dev_pct"] / mlp_dev["mean_dev_pct"]
    else:
        repair_ratio = 0.0

    # Check for divergence
    diverged_mlp = not np.all(np.isfinite(out_mlp_naive))
    diverged_tf = not np.all(np.isfinite(out_tf_naive))

    return {
        "N": N, "L": L, "d": d, "r": r, "n_heads": n_heads,
        "n_inputs": n_inputs, "seed": seed,
        "sum_per_layer_err": sum_per_layer_err,
        # MLP-only metrics
        "mlp_mean_dev_pct": mlp_dev["mean_dev_pct"],
        "mlp_max_dev_pct": mlp_dev["max_dev_pct"],
        "mlp_amp_ratio": mlp_amp,
        "mlp_signal_pct": mlp_signal,
        "mlp_diverged": diverged_mlp,
        # Transformer metrics
        "tf_mean_dev_pct": tf_dev["mean_dev_pct"],
        "tf_max_dev_pct": tf_dev["max_dev_pct"],
        "tf_amp_ratio": tf_amp,
        "tf_signal_pct": tf_signal,
        "tf_diverged": diverged_tf,
        # Self-repair
        "repair_ratio": repair_ratio,  # >0 means attention helps
    }


# ============================================================================
# Layer-by-layer self-repair analysis
# ============================================================================

def run_layerwise_repair(
    N: int, L: int, d: int, r: int, n_heads: int,
    n_inputs: int, seed: int
) -> list[dict]:
    """Measure output deviation after each layer for both architectures.

    This directly tests K2: does self-repair increase with depth?
    We feed inputs through the model and measure deviation at each layer.
    """
    rng = np.random.RandomState(seed)

    experts = []
    for i in range(N):
        layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
        experts.append(layers)

    inputs = rng.randn(n_inputs, d) * 0.1
    remove_idx = N // 2

    # Merge experts per layer
    all_merged_deltas = []
    all_ortho_deltas = []

    for l in range(L):
        layer_deltas = [experts[i][l].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

    # GT removal deltas
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    # Naive removal deltas
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    # Build models
    mlp_model = MLPOnlyModel(L, d, seed=seed + 10000)
    tf_model = TransformerModel(L, d, n_heads, seed=seed + 10000)
    for l in range(L):
        tf_model.W[l] = mlp_model.W[l].copy()

    # Run layer-by-layer and record deviation at each layer
    results = []

    # MLP-only: step through layers
    h_gt = inputs.copy()
    h_naive = inputs.copy()
    mlp_scale = 1.0 / np.sqrt(L) if L > 1 else 1.0

    for l in range(L):
        # GT path
        W_gt = mlp_model.W[l] + gt_removed_deltas[l]
        h_gt_norm = rms_norm(h_gt)
        h_gt = h_gt + mlp_scale * activation(h_gt_norm @ W_gt.T)

        # Naive path
        W_naive = mlp_model.W[l] + naive_removed_deltas[l]
        h_naive_norm = rms_norm(h_naive)
        h_naive = h_naive + mlp_scale * activation(h_naive_norm @ W_naive.T)

        # Deviation after this layer
        diff = np.linalg.norm(h_naive - h_gt, axis=-1)
        gt_norm = np.maximum(np.linalg.norm(h_gt, axis=-1), 1e-12)
        dev = float(np.mean(diff / gt_norm * 100.0))

        results.append({
            "arch": "mlp_only",
            "layer": l,
            "depth_fraction": (l + 1) / L,
            "mean_dev_pct": dev,
            "seed": seed,
        })

    # Transformer: step through layers
    h_gt = inputs.copy()
    h_naive = inputs.copy()
    tf_scale = 1.0 / np.sqrt(2 * L) if L > 1 else 1.0

    for l in range(L):
        # GT path: attention sub-layer
        h_gt_norm = rms_norm(h_gt)
        attn_out_gt = tf_model.attention(h_gt_norm, l)
        h_gt = h_gt + tf_scale * attn_out_gt

        # GT path: MLP sub-layer
        W_gt = tf_model.W[l] + gt_removed_deltas[l]
        h_gt_norm2 = rms_norm(h_gt)
        h_gt = h_gt + tf_scale * activation(h_gt_norm2 @ W_gt.T)

        # Naive path: attention sub-layer
        h_naive_norm = rms_norm(h_naive)
        attn_out_naive = tf_model.attention(h_naive_norm, l)
        h_naive = h_naive + tf_scale * attn_out_naive

        # Naive path: MLP sub-layer
        W_naive = tf_model.W[l] + naive_removed_deltas[l]
        h_naive_norm2 = rms_norm(h_naive)
        h_naive = h_naive + tf_scale * activation(h_naive_norm2 @ W_naive.T)

        # Deviation after this layer
        diff = np.linalg.norm(h_naive - h_gt, axis=-1)
        gt_norm = np.maximum(np.linalg.norm(h_gt, axis=-1), 1e-12)
        dev = float(np.mean(diff / gt_norm * 100.0))

        results.append({
            "arch": "transformer",
            "layer": l,
            "depth_fraction": (l + 1) / L,
            "mean_dev_pct": dev,
            "seed": seed,
        })

    return results


# ============================================================================
# Full experiment
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: Attention Self-Repair After Expert Removal")
    print("  K1: transformer shows >30% lower deviation than MLP-only")
    print("  K2: self-repair INCREASES with depth")
    print("=" * 80)

    d = 64
    r = 8
    n_heads = 4  # d=64 / n_heads=4 => head_dim=16
    N = 8
    n_inputs = 100  # Keep manageable for attention O(n^2)
    seeds = [42, 123, 777]

    all_results = []
    layerwise_results = []

    # ================================================================
    # TEST 1: Architecture comparison across depth
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: MLP-only vs Transformer across depth")
    print("  d=64, r=8, N=8, n_heads=4")
    print("=" * 80)

    L_values = [1, 2, 4, 8, 12, 16]

    print(f"\n{'L':>4} {'Seed':>6} "
          f"{'MLP_Dev%':>10} {'TF_Dev%':>10} {'Repair%':>9} "
          f"{'MLP_Amp':>9} {'TF_Amp':>9}")
    print("-" * 70)

    for L in L_values:
        for seed in seeds:
            res = run_removal_comparison(
                N, L, d, r, n_heads, n_inputs, seed
            )
            res["test"] = "depth_comparison"
            all_results.append(res)

            if res["mlp_diverged"] or res["tf_diverged"]:
                print(f"{L:>4} {seed:>6} {'DIVERGED':>10} {'DIVERGED':>10}")
            else:
                print(f"{L:>4} {seed:>6} "
                      f"{res['mlp_mean_dev_pct']:>10.4f} "
                      f"{res['tf_mean_dev_pct']:>10.4f} "
                      f"{res['repair_ratio']*100:>8.1f}% "
                      f"{res['mlp_amp_ratio']:>9.4f} "
                      f"{res['tf_amp_ratio']:>9.4f}")

    # ================================================================
    # TEST 2: Dimension scaling
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: Dimension scaling (L=12, N=8)")
    print("=" * 80)

    d_values = [32, 64, 128]
    L = 12

    print(f"\n{'d':>5} {'Seed':>6} "
          f"{'MLP_Dev%':>10} {'TF_Dev%':>10} {'Repair%':>9}")
    print("-" * 50)

    for d_val in d_values:
        nh = max(2, d_val // 16)  # head_dim=16
        for seed in seeds:
            res = run_removal_comparison(
                N, L, d_val, r, nh, n_inputs, seed
            )
            res["test"] = "dim_scaling"
            all_results.append(res)

            if res["mlp_diverged"] or res["tf_diverged"]:
                print(f"{d_val:>5} {seed:>6} DIVERGED")
            else:
                print(f"{d_val:>5} {seed:>6} "
                      f"{res['mlp_mean_dev_pct']:>10.4f} "
                      f"{res['tf_mean_dev_pct']:>10.4f} "
                      f"{res['repair_ratio']*100:>8.1f}%")

    # ================================================================
    # TEST 3: Layer-by-layer self-repair trajectory (K2)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 3: Layer-by-layer deviation trajectory (K2 test)")
    print("  Tracking deviation at each layer for L=16")
    print("=" * 80)

    L = 16
    for seed in seeds:
        lw = run_layerwise_repair(N, L, d, r, n_heads, n_inputs, seed)
        layerwise_results.extend(lw)

    # Print trajectory
    print(f"\n{'Layer':>6} {'MLP_Dev%':>10} {'TF_Dev%':>10} {'Repair%':>9}")
    print("-" * 42)

    for l in range(L):
        mlp_devs = [r["mean_dev_pct"] for r in layerwise_results
                     if r["arch"] == "mlp_only" and r["layer"] == l]
        tf_devs = [r["mean_dev_pct"] for r in layerwise_results
                    if r["arch"] == "transformer" and r["layer"] == l]
        if mlp_devs and tf_devs:
            m = np.mean(mlp_devs)
            t = np.mean(tf_devs)
            rep = (1.0 - t / m) * 100 if m > 1e-12 else 0.0
            print(f"{l:>6} {m:>10.4f} {t:>10.4f} {rep:>8.1f}%")

    # ================================================================
    # TEST 4: Number of experts sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 4: Expert count sweep (L=12, d=64)")
    print("=" * 80)

    N_values = [4, 8, 16]
    L = 12

    print(f"\n{'N':>4} {'Seed':>6} "
          f"{'MLP_Dev%':>10} {'TF_Dev%':>10} {'Repair%':>9}")
    print("-" * 50)

    for N_val in N_values:
        for seed in seeds:
            res = run_removal_comparison(
                N_val, L, d, r, n_heads, n_inputs, seed
            )
            res["test"] = "expert_count"
            all_results.append(res)

            if res["mlp_diverged"] or res["tf_diverged"]:
                print(f"{N_val:>4} {seed:>6} DIVERGED")
            else:
                print(f"{N_val:>4} {seed:>6} "
                      f"{res['mlp_mean_dev_pct']:>10.4f} "
                      f"{res['tf_mean_dev_pct']:>10.4f} "
                      f"{res['repair_ratio']*100:>8.1f}%")

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("  AGGREGATE ANALYSIS")
    print("=" * 80)

    # --- K1: Self-repair exists? ---
    print("\n  K1 Assessment: Does attention reduce deviation by >30%?")
    print("  (repair_ratio > 0.3)")
    print()

    depth_results = [r for r in all_results
                     if r["test"] == "depth_comparison"
                     and not r["mlp_diverged"] and not r["tf_diverged"]]

    print(f"  {'L':>4} {'MeanRepair%':>13} {'StdRepair':>11} "
          f"{'MLP_Dev%':>10} {'TF_Dev%':>10} {'K1':>10}")
    print("  " + "-" * 65)

    repair_by_L = {}
    for L in L_values:
        L_results = [r for r in depth_results if r["L"] == L]
        if not L_results:
            continue
        repairs = [r["repair_ratio"] for r in L_results]
        mlp_devs = [r["mlp_mean_dev_pct"] for r in L_results]
        tf_devs = [r["tf_mean_dev_pct"] for r in L_results]

        mean_repair = np.mean(repairs)
        std_repair = np.std(repairs)
        repair_by_L[L] = (mean_repair, std_repair)
        k1_pass = mean_repair > 0.30

        print(f"  {L:>4} {mean_repair*100:>12.1f}% {std_repair*100:>10.1f}% "
              f"{np.mean(mlp_devs):>10.4f} {np.mean(tf_devs):>10.4f} "
              f"{'PASS' if k1_pass else 'FAIL':>10}")

    # Overall K1
    all_repairs = [r["repair_ratio"] for r in depth_results]
    overall_repair = np.mean(all_repairs) if all_repairs else 0.0
    k1_pass_overall = overall_repair > 0.30

    print(f"\n  Overall mean repair ratio: {overall_repair*100:.1f}%")
    print(f"  K1 threshold: >30%")
    print(f"  K1 VERDICT: {'PASS (self-repair exists)' if k1_pass_overall else 'FAIL (no significant self-repair)'}")

    # --- K2: Self-repair increases with depth? ---
    print("\n  " + "-" * 70)
    print("  K2 Assessment: Does self-repair INCREASE with depth?")
    print()

    # Use layerwise data to measure repair ratio trend
    layers_list = []
    repair_list = []
    for l in range(16):  # L=16 from Test 3
        mlp_devs = [r["mean_dev_pct"] for r in layerwise_results
                     if r["arch"] == "mlp_only" and r["layer"] == l]
        tf_devs = [r["mean_dev_pct"] for r in layerwise_results
                    if r["arch"] == "transformer" and r["layer"] == l]
        if mlp_devs and tf_devs:
            m = np.mean(mlp_devs)
            t = np.mean(tf_devs)
            if m > 1e-12:
                rep = 1.0 - t / m
                layers_list.append(l)
                repair_list.append(rep)

    if len(layers_list) >= 3:
        slope, intercept, r_val, p_val, se = stats.linregress(
            layers_list, repair_list
        )
        print(f"  Regression: repair_ratio = {slope:.6f} * layer + {intercept:.4f}")
        print(f"  R^2 = {r_val**2:.4f}, p = {p_val:.6f}")
        print(f"  Slope sign: {'POSITIVE (increasing)' if slope > 0 else 'NEGATIVE (decreasing)'}")
        print(f"  Significance: {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05)'}")

        k2_pass = slope > 0 and p_val < 0.05
        print(f"\n  K2 VERDICT: {'PASS (self-repair increases with depth)' if k2_pass else 'FAIL'}")
    else:
        k2_pass = False
        slope = 0.0
        p_val = 1.0
        print("  Insufficient data for K2 regression")

    # Also check depth comparison data for K2
    print("\n  Cross-check: repair ratio by depth (from Test 1):")
    Ls_for_reg = []
    repairs_for_reg = []
    for L in sorted(repair_by_L.keys()):
        mean_rep, _ = repair_by_L[L]
        Ls_for_reg.append(L)
        repairs_for_reg.append(mean_rep)
        print(f"    L={L}: repair = {mean_rep*100:.1f}%")

    if len(Ls_for_reg) >= 3:
        slope2, _, r2, p2, _ = stats.linregress(Ls_for_reg, repairs_for_reg)
        print(f"  Cross-check regression: slope={slope2:.6f}, R^2={r2**2:.4f}, p={p2:.6f}")
        k2_cross = slope2 > 0 and p2 < 0.05
        print(f"  Cross-check verdict: {'PASS' if k2_cross else 'FAIL'}")

    # --- Amplification ratio comparison ---
    print("\n  " + "-" * 70)
    print("  Amplification Ratio Comparison:")
    print(f"  {'L':>4} {'MLP_AmpRatio':>14} {'TF_AmpRatio':>13} {'Reduction':>11}")
    print("  " + "-" * 50)

    for L in L_values:
        L_results = [r for r in depth_results if r["L"] == L]
        if not L_results:
            continue
        mlp_amps = [r["mlp_amp_ratio"] for r in L_results]
        tf_amps = [r["tf_amp_ratio"] for r in L_results]
        ma = np.mean(mlp_amps)
        ta = np.mean(tf_amps)
        red = (1.0 - ta / ma) * 100 if ma > 1e-12 else 0.0
        print(f"  {L:>4} {ma:>14.4f} {ta:>13.4f} {red:>10.1f}%")

    # --- Expert signal comparison ---
    print("\n  " + "-" * 70)
    print("  Expert Signal (how much the removed expert matters):")
    print(f"  {'L':>4} {'MLP_Signal%':>13} {'TF_Signal%':>12}")
    print("  " + "-" * 35)

    for L in L_values:
        L_results = [r for r in depth_results if r["L"] == L]
        if not L_results:
            continue
        ms = [r["mlp_signal_pct"] for r in L_results]
        ts = [r["tf_signal_pct"] for r in L_results]
        print(f"  {L:>4} {np.mean(ms):>13.4f} {np.mean(ts):>12.4f}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    print(f"\n  K1 (self-repair exists, >30% lower deviation):")
    print(f"    Overall repair ratio: {overall_repair*100:.1f}%")
    print(f"    Verdict: {'PASS' if k1_pass_overall else 'FAIL'}")

    print(f"\n  K2 (self-repair increases with depth):")
    print(f"    Layer-wise slope: {slope:.6f} (p={p_val:.4f})")
    print(f"    Verdict: {'PASS' if k2_pass else 'FAIL'}")

    if k1_pass_overall and k2_pass:
        overall = "PROVEN"
        print(f"\n  OVERALL: {overall}")
        print("  Attention provides self-repair after expert removal.")
        print("  The effect increases with depth -- deeper transformers are safer.")
        print("  Production implication: the residual_layernorm safety margin")
        print("  (amp_ratio=0.022) is CONSERVATIVE; real transformers with attention")
        print("  are even safer.")
    elif k1_pass_overall:
        overall = "SUPPORTED"
        print(f"\n  OVERALL: {overall}")
        print("  Attention provides self-repair (K1 pass) but the depth trend")
        print("  is not statistically significant (K2 fail).")
    else:
        overall = "KILLED"
        print(f"\n  OVERALL: {overall}")
        print("  Attention does NOT provide significant self-repair.")
        print("  The residual_layernorm safety margin is the correct bound.")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    output = {
        "summary": {
            "k1_pass": bool(k1_pass_overall),
            "k2_pass": bool(k2_pass),
            "overall_repair_ratio": float(overall_repair),
            "overall_verdict": overall,
        },
        "depth_comparison": [r for r in all_results if r["test"] == "depth_comparison"],
        "dim_scaling": [r for r in all_results if r["test"] == "dim_scaling"],
        "expert_count": [r for r in all_results if r["test"] == "expert_count"],
        "layerwise": layerwise_results,
    }

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(output), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    run_full_experiment()
