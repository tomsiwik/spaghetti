"""Profile activation FREQUENCY as a pruning signal for Qwen2.5-0.5B.

Parent experiment (swiglu_macro_pruning_transfer) found that mean gate-product
magnitude is an ANTI-signal at macro scale: 8.9x worse than random pruning.
Low mean magnitude selects specialist neurons that fire rarely but strongly.

This experiment tests whether activation FREQUENCY (fraction of positions where
a neuron fires above a threshold epsilon) is a better pruning signal.

Hypothesis: Neurons that fire on nearly ALL inputs (high frequency) are
"always-on" generalist neurons. Removing them distributes damage uniformly
across all inputs (graceful degradation). Neurons that fire rarely (low
frequency) are specialists -- removing them catastrophically hurts specific
inputs.

Kill criteria:
  1. Frequency-based pruning is not >2x better than random at matched neuron count
  2. Frequency signal correlates >0.8 (Spearman) with mean magnitude (redundant)

Data pipeline (reused from parent):
  - Calibration: WikiText-2-raw-v1 TEST split
  - Evaluation: WikiText-2-raw-v1 VALIDATION split (genuinely held-out)

Architecture: Qwen2.5-0.5B, 24 layers, 4864 SwiGLU neurons/layer = 116,736 total
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Reuse data loading and model loading from parent experiment
# ============================================================================

parent_dir = Path(__file__).parent.parent / "swiglu_macro_pruning_transfer"
sys.path.insert(0, str(parent_dir))
from profile_gate_products import (
    load_model,
    load_wikitext2_split,
    compute_perplexity,
)


# ============================================================================
# Frequency profiling (the NEW signal)
# ============================================================================

def profile_frequency(model, calibration_data, batch_size: int = 8,
                      epsilons: list[float] = None):
    """Profile per-neuron firing frequency across all layers.

    For each layer l and neuron j:
      freq_j^l(eps) = (1/M) * sum_m 1[|h_j^l(x_m)| > eps]

    where h_j^l(x) = SiLU(gate_proj_j(x)) * up_proj_j(x) is the gate product.

    Also collects mean magnitude for correlation analysis.

    Args:
        model: Qwen2.5-0.5B model
        calibration_data: (n_sequences, seq_len) token array
        batch_size: batch size for forward pass
        epsilons: list of firing thresholds to test

    Returns:
        dict with per-layer frequency arrays and mean magnitudes
    """
    if epsilons is None:
        # Range of epsilon values to test
        # The parent experiment showed median gate product ~0.078
        # So we test a range from very low (nearly all fire) to moderate
        epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    n_sequences, seq_len = calibration_data.shape

    print(f"\nProfiling activation frequency:")
    print(f"  Layers: {n_layers}, neurons/layer: {intermediate_dim}")
    print(f"  Sequences: {n_sequences}, seq_len: {seq_len}")
    print(f"  Epsilons: {epsilons}")

    # Accumulators: count of firings above each epsilon
    # freq_counts[eps_idx][layer] = array of shape (intermediate_dim,)
    freq_counts = {
        eps_idx: [mx.zeros(intermediate_dim) for _ in range(n_layers)]
        for eps_idx in range(len(epsilons))
    }
    # Also accumulate mean magnitude for correlation analysis
    sum_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    # And max magnitude for specialist identification
    max_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]

    total_positions = 0
    n_batches = math.ceil(n_sequences / batch_size)
    eps_array = mx.array(epsilons)  # for vectorized comparison

    t0 = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_sequences)
        batch = calibration_data[start:end]
        B, T = batch.shape
        total_positions += B * T

        # Forward through embedding
        x = model.model.embed_tokens(batch)

        # Forward through each layer, intercepting MLP
        for l_idx, layer in enumerate(model.model.layers):
            # Run attention
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, mask=None, cache=None)
            if isinstance(x, tuple):
                x = x[0]
            x = residual + x

            # MLP with gate product interception
            residual = x
            x_norm = layer.post_attention_layernorm(x)

            mlp = layer.mlp
            gate_out = nn.silu(mlp.gate_proj(x_norm))   # (B, T, d_ff)
            up_out = mlp.up_proj(x_norm)                  # (B, T, d_ff)
            gate_product = gate_out * up_out               # (B, T, d_ff)

            gp_abs = mx.abs(gate_product)  # (B, T, d_ff)

            # Accumulate mean magnitude
            sum_abs[l_idx] = sum_abs[l_idx] + mx.sum(gp_abs, axis=(0, 1))
            # Accumulate max magnitude
            max_abs[l_idx] = mx.maximum(max_abs[l_idx], mx.max(gp_abs, axis=(0, 1)))

            # Count firings above each epsilon
            for eps_idx, eps in enumerate(epsilons):
                fired = (gp_abs > eps).astype(mx.float32)  # (B, T, d_ff)
                freq_counts[eps_idx][l_idx] = (
                    freq_counts[eps_idx][l_idx] + mx.sum(fired, axis=(0, 1))
                )

            # Complete MLP forward pass
            mlp_out = mlp.down_proj(gate_product)
            x = residual + mlp_out

            # Evaluate to free memory
            eval_arrays = [sum_abs[l_idx], max_abs[l_idx]]
            for eps_idx in range(len(epsilons)):
                eval_arrays.append(freq_counts[eps_idx][l_idx])
            mx.eval(*eval_arrays)

        if (batch_idx + 1) % 4 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{n_batches} ({elapsed:.1f}s)")

    # Compute final results
    results = {
        "epsilons": epsilons,
        "total_positions": total_positions,
        "n_layers": n_layers,
        "intermediate_dim": intermediate_dim,
        "per_layer": [],
    }

    for l_idx in range(n_layers):
        mean_mag = sum_abs[l_idx] / total_positions
        mx.eval(mean_mag, max_abs[l_idx])

        layer_data = {
            "layer": l_idx,
            "mean_magnitude": mean_mag,          # (d_ff,) mean |gate_product|
            "max_magnitude": max_abs[l_idx],     # (d_ff,) max |gate_product|
            "frequencies": {},                    # eps -> (d_ff,) fraction
        }

        for eps_idx, eps in enumerate(epsilons):
            freq = freq_counts[eps_idx][l_idx] / total_positions
            mx.eval(freq)
            layer_data["frequencies"][eps] = freq

        results["per_layer"].append(layer_data)

    elapsed = time.time() - t0
    print(f"\nProfiling complete in {elapsed:.1f}s ({total_positions} positions)")
    return results


# ============================================================================
# Correlation analysis (kill criterion 2)
# ============================================================================

def compute_spearman_correlation(x_vals: list[float], y_vals: list[float]) -> float:
    """Compute Spearman rank correlation between two lists.

    Spearman rho = Pearson correlation of the ranks.
    """
    n = len(x_vals)
    assert n == len(y_vals)

    # Compute ranks
    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j+1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x_vals)
    ry = rank(y_vals)

    # Pearson on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n)) / n
    std_rx = (sum((rx[i] - mean_rx)**2 for i in range(n)) / n) ** 0.5
    std_ry = (sum((ry[i] - mean_ry)**2 for i in range(n)) / n) ** 0.5

    if std_rx == 0 or std_ry == 0:
        return 0.0
    return cov / (std_rx * std_ry)


def analyze_correlation(profiles: dict, eps_for_corr: float = 0.01):
    """Analyze correlation between firing frequency and mean magnitude.

    Kill criterion 2: if Spearman |rho| > 0.8, frequency is redundant
    with mean magnitude (no new information).
    """
    all_freq = []
    all_mean = []

    for layer_data in profiles["per_layer"]:
        freq = layer_data["frequencies"][eps_for_corr].tolist()
        mean_mag = layer_data["mean_magnitude"].tolist()
        all_freq.extend(freq)
        all_mean.extend(mean_mag)

    rho = compute_spearman_correlation(all_freq, all_mean)
    return {
        "epsilon": eps_for_corr,
        "spearman_rho": rho,
        "abs_rho": abs(rho),
        "is_redundant": abs(rho) > 0.8,
        "n_neurons": len(all_freq),
    }


# ============================================================================
# Frequency-based pruning
# ============================================================================

def prune_by_frequency(model, profiles: dict, eval_data,
                        eps: float = 0.01,
                        prune_fractions: list[float] = None,
                        direction: str = "high_first",
                        batch_size: int = 4):
    """Prune neurons by firing frequency and measure perplexity.

    Args:
        direction: "high_first" prunes highest-frequency neurons first
                   (always-on generalists -- our hypothesis says these are safe)
                   "low_first" prunes lowest-frequency neurons first
                   (specialists -- parent experiment showed these are dangerous)
    """
    if prune_fractions is None:
        prune_fractions = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    # Save original weights
    print(f"\nPruning by frequency (eps={eps}, direction={direction}):")
    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    # Baseline perplexity
    print("  Computing baseline perplexity...")
    base_ppl, base_loss = compute_perplexity(model, eval_data, batch_size)
    print(f"  Baseline: ppl={base_ppl:.2f}")

    # Collect all (layer, neuron, frequency) tuples
    all_neurons = []
    for layer_data in profiles["per_layer"]:
        l_idx = layer_data["layer"]
        freq = layer_data["frequencies"][eps].tolist()
        mean_mag = layer_data["mean_magnitude"].tolist()
        for j in range(intermediate_dim):
            all_neurons.append((l_idx, j, freq[j], mean_mag[j]))

    # Sort by frequency
    if direction == "high_first":
        # Prune highest frequency first (always-on neurons)
        all_neurons.sort(key=lambda x: -x[2])
    else:
        # Prune lowest frequency first (specialist neurons)
        all_neurons.sort(key=lambda x: x[2])

    results = {
        "baseline_ppl": base_ppl,
        "baseline_loss": base_loss,
        "epsilon": eps,
        "direction": direction,
        "pruning": [],
    }

    for frac in prune_fractions:
        n_to_prune = int(total_neurons * frac)

        # Restore original weights
        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight = mx.array(original_weights[l_idx]["gate_proj"])
            mlp.up_proj.weight = mx.array(original_weights[l_idx]["up_proj"])
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        # Build per-layer masks
        per_layer_prune = {l: set() for l in range(n_layers)}
        for i in range(min(n_to_prune, total_neurons)):
            l_idx, j, _, _ = all_neurons[i]
            per_layer_prune[l_idx].add(j)

        actual_pruned = sum(len(v) for v in per_layer_prune.values())

        for l_idx, layer in enumerate(model.model.layers):
            neurons_to_zero = per_layer_prune[l_idx]
            if not neurons_to_zero:
                continue
            mlp = layer.mlp
            mask_list = [True] * intermediate_dim
            for j in neurons_to_zero:
                mask_list[j] = False
            mask = mx.array(mask_list).reshape(-1, 1).astype(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight = mlp.gate_proj.weight * mask
            mlp.up_proj.weight = mlp.up_proj.weight * mask
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        pct_pruned = actual_pruned / total_neurons * 100
        ppl, loss = compute_perplexity(model, eval_data, batch_size)
        delta_ppl = (ppl - base_ppl) / base_ppl * 100

        # Get frequency stats of pruned neurons
        pruned_freqs = [all_neurons[i][2] for i in range(min(n_to_prune, total_neurons))]
        pruned_means = [all_neurons[i][3] for i in range(min(n_to_prune, total_neurons))]
        avg_pruned_freq = sum(pruned_freqs) / len(pruned_freqs) if pruned_freqs else 0
        avg_pruned_mean = sum(pruned_means) / len(pruned_means) if pruned_means else 0

        print(f"  {frac*100:.0f}% ({actual_pruned} neurons): ppl={ppl:.2f} "
              f"(delta={delta_ppl:+.1f}%), avg_freq={avg_pruned_freq:.3f}, "
              f"avg_mean_mag={avg_pruned_mean:.4f}")

        results["pruning"].append({
            "target_fraction": frac,
            "actual_pruned": actual_pruned,
            "pct_pruned": pct_pruned,
            "ppl": ppl,
            "loss": loss,
            "delta_ppl_pct": delta_ppl,
            "avg_pruned_freq": avg_pruned_freq,
            "avg_pruned_mean_mag": avg_pruned_mean,
        })

    # Restore weights
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    return results


# ============================================================================
# Random pruning baseline (reuse logic from parent)
# ============================================================================

def test_random_pruning(model, n_neurons_to_prune: int, eval_data,
                         n_seeds: int = 3, batch_size: int = 4):
    """Random pruning baseline: same as parent experiment."""
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    print(f"\n  Random baseline: {n_neurons_to_prune}/{total_neurons} neurons, {n_seeds} seeds")

    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    results = []
    for seed in range(n_seeds):
        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight = mx.array(original_weights[l_idx]["gate_proj"])
            mlp.up_proj.weight = mx.array(original_weights[l_idx]["up_proj"])
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        rng = random.Random(seed + 2000)
        all_ids = [(l, j) for l in range(n_layers) for j in range(intermediate_dim)]
        selected = rng.sample(all_ids, min(n_neurons_to_prune, total_neurons))

        per_layer = {l: [] for l in range(n_layers)}
        for l, j in selected:
            per_layer[l].append(j)

        for l_idx, layer in enumerate(model.model.layers):
            if not per_layer[l_idx]:
                continue
            mlp = layer.mlp
            mask_list = [True] * intermediate_dim
            for j in per_layer[l_idx]:
                mask_list[j] = False
            mask = mx.array(mask_list).reshape(-1, 1).astype(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight = mlp.gate_proj.weight * mask
            mlp.up_proj.weight = mlp.up_proj.weight * mask
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        ppl, loss = compute_perplexity(model, eval_data, batch_size)
        print(f"    Seed {seed}: ppl={ppl:.2f}")
        results.append({"seed": seed, "ppl": ppl, "loss": loss})

    # Restore
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    mean_ppl = sum(r["ppl"] for r in results) / len(results)
    std_ppl = (sum((r["ppl"] - mean_ppl)**2 for r in results) / len(results)) ** 0.5
    return {
        "n_pruned": n_neurons_to_prune,
        "n_seeds": n_seeds,
        "per_seed": results,
        "mean_ppl": mean_ppl,
        "std_ppl": std_ppl,
    }


# ============================================================================
# Mean magnitude pruning (parent experiment signal, for comparison)
# ============================================================================

def prune_by_mean_magnitude(model, profiles: dict, eval_data,
                             prune_fractions: list[float] = None,
                             batch_size: int = 4):
    """Prune lowest-mean-magnitude neurons (parent experiment's signal).

    This reproduces the parent finding for direct comparison at matched
    prune fractions.
    """
    if prune_fractions is None:
        prune_fractions = [0.05, 0.10, 0.15]

    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    print(f"\nPruning by mean magnitude (low-first, parent signal):")

    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    base_ppl, _ = compute_perplexity(model, eval_data, batch_size)
    print(f"  Baseline: ppl={base_ppl:.2f}")

    # Collect (layer, neuron, mean_mag)
    all_neurons = []
    for layer_data in profiles["per_layer"]:
        l_idx = layer_data["layer"]
        mean_mag = layer_data["mean_magnitude"].tolist()
        for j in range(intermediate_dim):
            all_neurons.append((l_idx, j, mean_mag[j]))

    # Sort ascending (lowest mean mag first = parent's pruning order)
    all_neurons.sort(key=lambda x: x[2])

    results = {"baseline_ppl": base_ppl, "pruning": []}

    for frac in prune_fractions:
        n_to_prune = int(total_neurons * frac)

        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight = mx.array(original_weights[l_idx]["gate_proj"])
            mlp.up_proj.weight = mx.array(original_weights[l_idx]["up_proj"])
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        per_layer_prune = {l: set() for l in range(n_layers)}
        for i in range(min(n_to_prune, total_neurons)):
            l_idx, j, _ = all_neurons[i]
            per_layer_prune[l_idx].add(j)

        actual_pruned = sum(len(v) for v in per_layer_prune.values())

        for l_idx, layer in enumerate(model.model.layers):
            neurons_to_zero = per_layer_prune[l_idx]
            if not neurons_to_zero:
                continue
            mlp = layer.mlp
            mask_list = [True] * intermediate_dim
            for j in neurons_to_zero:
                mask_list[j] = False
            mask = mx.array(mask_list).reshape(-1, 1).astype(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight = mlp.gate_proj.weight * mask
            mlp.up_proj.weight = mlp.up_proj.weight * mask
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        ppl, loss = compute_perplexity(model, eval_data, batch_size)
        delta = (ppl - base_ppl) / base_ppl * 100
        print(f"  {frac*100:.0f}% ({actual_pruned} neurons): ppl={ppl:.2f} (delta={delta:+.1f}%)")
        results["pruning"].append({
            "target_fraction": frac,
            "actual_pruned": actual_pruned,
            "ppl": ppl,
            "loss": loss,
            "delta_ppl_pct": delta,
        })

    # Restore
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    return results


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(model_name: str = "Qwen/Qwen2.5-0.5B",
                   n_cal_sequences: int = 128,
                   n_eval_sequences: int = 64,
                   seq_len: int = 128,
                   batch_size: int = 8):
    """Run the full activation frequency pruning experiment."""

    output_dir = Path(__file__).parent
    results_path = output_dir / "results.json"

    # Load model
    model, tokenizer = load_model(model_name)

    # Load data (same splits as parent)
    print("\nLoading calibration data (WikiText-2 test split)...")
    cal_data, cal_prov = load_wikitext2_split(
        tokenizer, split="test", n_sequences=n_cal_sequences, seq_len=seq_len
    )
    print("\nLoading evaluation data (WikiText-2 validation split)...")
    eval_data, eval_prov = load_wikitext2_split(
        tokenizer, split="validation", n_sequences=n_eval_sequences, seq_len=seq_len
    )

    # ---- Phase 1: Profile frequencies ----
    profiles = profile_frequency(model, cal_data, batch_size)

    # ---- Phase 2: Correlation analysis (kill criterion 2) ----
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS (Kill Criterion 2)")
    print("="*70)

    corr_results = {}
    for eps in profiles["epsilons"]:
        corr = analyze_correlation(profiles, eps_for_corr=eps)
        corr_results[str(eps)] = corr
        status = "KILL (redundant)" if corr["is_redundant"] else "PASS"
        print(f"  eps={eps:.3f}: Spearman rho = {corr['spearman_rho']:.4f} "
              f"|rho| = {corr['abs_rho']:.4f} [{status}]")

    # ---- Phase 3: Frequency distribution summary ----
    print("\n" + "="*70)
    print("FREQUENCY DISTRIBUTION SUMMARY")
    print("="*70)

    for eps in [0.01, 0.05]:
        all_freqs = []
        for ld in profiles["per_layer"]:
            all_freqs.extend(ld["frequencies"][eps].tolist())
        all_freqs.sort()
        n = len(all_freqs)
        p = lambda pct: all_freqs[int(n * pct / 100)]
        print(f"\n  eps={eps}: Firing frequency across {n} neurons")
        print(f"    Min={all_freqs[0]:.4f}  P5={p(5):.4f}  P25={p(25):.4f}  "
              f"Median={p(50):.4f}  P75={p(75):.4f}  P95={p(95):.4f}  "
              f"Max={all_freqs[-1]:.4f}")
        # Count always-on and never-fire
        always_on = sum(1 for f in all_freqs if f > 0.99)
        never_fire = sum(1 for f in all_freqs if f < 0.01)
        print(f"    Always-on (>99%): {always_on} ({always_on/n*100:.1f}%)")
        print(f"    Never-fire (<1%): {never_fire} ({never_fire/n*100:.1f}%)")

    # ---- Phase 4: Pruning experiments ----
    print("\n" + "="*70)
    print("PRUNING EXPERIMENTS")
    print("="*70)

    prune_fractions = [0.01, 0.02, 0.05, 0.10, 0.15]

    # 4a: Frequency-based, high-first (our hypothesis: safe to prune)
    freq_high_results = prune_by_frequency(
        model, profiles, eval_data, eps=0.01,
        prune_fractions=prune_fractions, direction="high_first",
        batch_size=batch_size
    )

    # 4b: Frequency-based, low-first (should be bad -- removing specialists)
    freq_low_results = prune_by_frequency(
        model, profiles, eval_data, eps=0.01,
        prune_fractions=prune_fractions, direction="low_first",
        batch_size=batch_size
    )

    # 4c: Mean magnitude, low-first (parent experiment signal, anti-signal)
    mean_mag_results = prune_by_mean_magnitude(
        model, profiles, eval_data,
        prune_fractions=prune_fractions, batch_size=batch_size
    )

    # 4d: Random baselines at key prune fractions
    print("\n" + "="*70)
    print("RANDOM PRUNING BASELINES")
    print("="*70)
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    random_results = {}
    for frac in [0.05, 0.10, 0.15]:
        n_prune = int(total_neurons * frac)
        rr = test_random_pruning(model, n_prune, eval_data, n_seeds=3,
                                  batch_size=batch_size)
        random_results[str(frac)] = rr

    # ---- Phase 5: Summary and kill criterion checks ----
    print("\n" + "="*70)
    print("KILL CRITERION CHECKS")
    print("="*70)

    # KC1: frequency pruning >2x better than random at 5%
    frac_05_freq = None
    for pr in freq_high_results["pruning"]:
        if abs(pr["target_fraction"] - 0.05) < 0.001:
            frac_05_freq = pr
            break
    random_05 = random_results.get("0.05")

    if frac_05_freq and random_05:
        freq_ppl = frac_05_freq["ppl"]
        rand_ppl = random_05["mean_ppl"]
        base_ppl = freq_high_results["baseline_ppl"]

        # "Better" means less perplexity increase
        freq_delta = freq_ppl - base_ppl
        rand_delta = rand_ppl - base_ppl
        ratio = rand_delta / freq_delta if freq_delta > 0 else float('inf')

        print(f"\nKC1: Frequency pruning vs random at 5%")
        print(f"  Frequency (high-first) ppl: {freq_ppl:.2f} (delta: +{freq_delta:.2f})")
        print(f"  Random mean ppl: {rand_ppl:.2f} (delta: +{rand_delta:.2f})")
        print(f"  Ratio (random_delta / freq_delta): {ratio:.2f}x")
        if ratio > 2.0:
            print(f"  --> PASS: frequency pruning is {ratio:.1f}x better than random")
        elif freq_delta <= 0:
            print(f"  --> PASS: frequency pruning IMPROVES perplexity!")
        else:
            print(f"  --> KILL: frequency pruning is only {ratio:.2f}x better (need >2x)")

    # KC2: correlation
    corr_01 = corr_results.get("0.01", {})
    print(f"\nKC2: Frequency-magnitude correlation")
    print(f"  Spearman |rho| at eps=0.01: {corr_01.get('abs_rho', 'N/A'):.4f}")
    if corr_01.get("is_redundant"):
        print(f"  --> KILL: |rho| > 0.8, frequency is redundant with mean magnitude")
    else:
        print(f"  --> PASS: |rho| < 0.8, frequency is an independent signal")

    # ---- Comparison table ----
    print("\n" + "="*70)
    print("COMPARISON TABLE (5% pruned)")
    print("="*70)

    frac_05_mean = None
    for pr in mean_mag_results["pruning"]:
        if abs(pr["target_fraction"] - 0.05) < 0.001:
            frac_05_mean = pr
            break
    frac_05_low = None
    for pr in freq_low_results["pruning"]:
        if abs(pr["target_fraction"] - 0.05) < 0.001:
            frac_05_low = pr
            break

    print(f"\n  {'Method':<30} {'PPL':>10} {'Delta%':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline':<30} {base_ppl:>10.2f} {'--':>10}")
    if frac_05_freq:
        print(f"  {'Freq high-first (ours)':<30} {frac_05_freq['ppl']:>10.2f} "
              f"{frac_05_freq['delta_ppl_pct']:>+10.1f}%")
    if frac_05_low:
        print(f"  {'Freq low-first (specialists)':<30} {frac_05_low['ppl']:>10.2f} "
              f"{frac_05_low['delta_ppl_pct']:>+10.1f}%")
    if frac_05_mean:
        print(f"  {'Mean magnitude (parent)':<30} {frac_05_mean['ppl']:>10.2f} "
              f"{frac_05_mean['delta_ppl_pct']:>+10.1f}%")
    if random_05:
        print(f"  {'Random (3-seed mean)':<30} {random_05['mean_ppl']:>10.2f} "
              f"{(random_05['mean_ppl'] - base_ppl) / base_ppl * 100:>+10.1f}%")

    # ---- Save results ----
    def make_serializable(obj):
        if isinstance(obj, mx.array):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, set):
            return list(obj)
        return obj

    # Don't save full per-neuron arrays (too large), save summary stats
    profiles_summary = []
    for ld in profiles["per_layer"]:
        layer_sum = {"layer": ld["layer"]}
        mm = ld["mean_magnitude"]
        layer_sum["mean_mag_stats"] = {
            "min": float(mx.min(mm).item()),
            "mean": float(mx.mean(mm).item()),
            "max": float(mx.max(mm).item()),
        }
        for eps, freq in ld["frequencies"].items():
            layer_sum[f"freq_eps_{eps}"] = {
                "min": float(mx.min(freq).item()),
                "mean": float(mx.mean(freq).item()),
                "max": float(mx.max(freq).item()),
                "pct_always_on": float(mx.mean((freq > 0.99).astype(mx.float32)).item()) * 100,
                "pct_never_fire": float(mx.mean((freq < 0.01).astype(mx.float32)).item()) * 100,
            }
        profiles_summary.append(layer_sum)

    output = {
        "model": model_name,
        "calibration": cal_prov,
        "evaluation": eval_prov,
        "profiles_summary": profiles_summary,
        "correlation": make_serializable(corr_results),
        "freq_high_first": make_serializable(freq_high_results),
        "freq_low_first": make_serializable(freq_low_results),
        "mean_magnitude": make_serializable(mean_mag_results),
        "random_baselines": make_serializable(random_results),
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    return output


if __name__ == "__main__":
    run_experiment()
