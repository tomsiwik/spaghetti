"""Wanda-style structured pruning at macro scale on Qwen2.5-0.5B.

Hypothesis: Wanda scoring (weight_norm * activation_magnitude) corrects the
specialist neuron problem that made pure activation-magnitude pruning 8.9x
WORSE than random. Specialist neurons have low mean activation but large
weights, so Wanda should score them highly (= keep them).

Adapted for STRUCTURED SwiGLU pruning (full neuron removal, not unstructured):
  For neuron j at layer l:
    W_j = [gate_proj_j ; up_proj_j]  (concatenated weight rows, shapes d each)
    X_j = mean_over_calibration |SiLU(gate_proj_j(x)) * up_proj_j(x)|
    wanda_score_j = ||W_j||_2 * X_j

  Low Wanda score = low weight AND low activation = truly unimportant.
  High weight + low activation (specialist) = high Wanda score = KEEP.

Kill criteria:
  1. Wanda scoring not >2x better than random at matching neuron count
  2. Requires >100 calibration samples (impractical)

Baselines from parent experiment (exp_swiglu_macro_pruning_transfer):
  - Baseline ppl: 21.31 (WikiText-2 validation)
  - Random pruning (18,420 neurons): mean ppl 61.97 +/- 8.52
  - Activation-only pruning (18,420 neurons): ppl 552.78 (8.9x worse than random)

Reuses: model loading, WikiText-2 loading, perplexity computation, random pruning
baseline from parent experiment profile_gate_products.py.

Reference: Sun et al., "A Simple and Effective Pruning Approach for Large
Language Models" (2023). https://arxiv.org/abs/2306.11695
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

# Reuse parent experiment utilities
parent_dir = Path(__file__).parent.parent / "swiglu_macro_pruning_transfer"
sys.path.insert(0, str(parent_dir))
from profile_gate_products import (
    load_model,
    load_wikitext2_split,
    compute_perplexity,
    test_random_pruning,
)


# ============================================================================
# Wanda-style structured profiling
# ============================================================================

def compute_weight_norms(model):
    """Compute L2 norm of [gate_proj_j; up_proj_j] for each neuron j.

    For SwiGLU neuron j at layer l:
      W_j = concat(gate_proj.weight[j, :], up_proj.weight[j, :])
      ||W_j||_2 = sqrt(||gate_proj_j||_2^2 + ||up_proj_j||_2^2)

    Returns:
      list of mx.array, one per layer, shape (d_ff,)
    """
    n_layers = len(model.model.layers)
    weight_norms = []

    for l_idx in range(n_layers):
        mlp = model.model.layers[l_idx].mlp
        # gate_proj.weight shape: (d_ff, d)
        # up_proj.weight shape: (d_ff, d)
        gate_w = mlp.gate_proj.weight  # (d_ff, d)
        up_w = mlp.up_proj.weight      # (d_ff, d)

        # L2 norm per row (neuron)
        gate_norm_sq = mx.sum(gate_w ** 2, axis=1)  # (d_ff,)
        up_norm_sq = mx.sum(up_w ** 2, axis=1)      # (d_ff,)

        # Combined norm: sqrt(||gate||^2 + ||up||^2)
        combined_norm = mx.sqrt(gate_norm_sq + up_norm_sq)
        mx.eval(combined_norm)
        weight_norms.append(combined_norm)

    return weight_norms


def profile_activations(model, calibration_data, batch_size: int = 8):
    """Profile mean absolute gate-product per neuron (activation component).

    Identical to parent experiment's profiling, but returns per-neuron
    mean absolute values for Wanda score computation.

    Returns:
      list of mx.array, one per layer, shape (d_ff,) -- mean |gate_product|
    """
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    n_sequences, seq_len = calibration_data.shape

    print(f"\nProfiling activations:")
    print(f"  Layers: {n_layers}, neurons/layer: {intermediate_dim}")
    print(f"  Sequences: {n_sequences}, seq_len: {seq_len}")

    sum_gp_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    total_positions = 0
    n_batches = math.ceil(n_sequences / batch_size)

    t0 = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_sequences)
        batch = calibration_data[start:end]
        B, T = batch.shape
        total_positions += B * T

        x = model.model.embed_tokens(batch)

        for l_idx, layer in enumerate(model.model.layers):
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, mask=None, cache=None)
            if isinstance(x, tuple):
                x = x[0]
            x = residual + x

            residual = x
            x_norm = layer.post_attention_layernorm(x)

            mlp = layer.mlp
            gate_out = nn.silu(mlp.gate_proj(x_norm))
            up_out = mlp.up_proj(x_norm)
            gate_product = gate_out * up_out

            gp_abs = mx.abs(gate_product)
            sum_gp_abs[l_idx] = sum_gp_abs[l_idx] + mx.sum(gp_abs, axis=(0, 1))

            mlp_out = mlp.down_proj(gate_product)
            x = residual + mlp_out

            mx.eval(sum_gp_abs[l_idx])

        if (batch_idx + 1) % 4 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{n_batches} ({elapsed:.1f}s)")

    activation_means = []
    for l_idx in range(n_layers):
        mean_abs = sum_gp_abs[l_idx] / total_positions
        mx.eval(mean_abs)
        activation_means.append(mean_abs)

    elapsed = time.time() - t0
    print(f"\nProfiling complete in {elapsed:.1f}s ({total_positions} positions)")
    return activation_means


def compute_wanda_scores(weight_norms, activation_means):
    """Compute Wanda scores: ||W_j||_2 * mean|X_j| for each neuron.

    Args:
        weight_norms: list of mx.array (d_ff,) per layer - weight L2 norms
        activation_means: list of mx.array (d_ff,) per layer - mean |gate_product|

    Returns:
        list of mx.array (d_ff,) per layer - Wanda scores
    """
    scores = []
    for w_norm, a_mean in zip(weight_norms, activation_means):
        s = w_norm * a_mean
        mx.eval(s)
        scores.append(s)
    return scores


# ============================================================================
# Pruning by score ranking (structured: full neuron removal)
# ============================================================================

def test_scored_pruning(model, scores, n_to_prune: int, eval_data,
                        score_name: str = "wanda", batch_size: int = 4):
    """Prune neurons with lowest scores, measure perplexity.

    Structured pruning: zeros out gate_proj and up_proj rows for pruned neurons.

    Args:
        model: the Qwen model
        scores: list of mx.array (d_ff,) per layer - importance scores
        n_to_prune: total number of neurons to prune across all layers
        eval_data: evaluation data
        score_name: label for logging
        batch_size: batch size for perplexity computation

    Returns:
        dict with pruning results
    """
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    # Collect all (score, layer_idx, neuron_idx) tuples
    all_scored = []
    for l_idx in range(n_layers):
        s_vals = scores[l_idx].tolist()
        for j, sv in enumerate(s_vals):
            all_scored.append((sv, l_idx, j))

    # Sort by score ascending (lowest = prune first)
    all_scored.sort(key=lambda x: x[0])

    # Select neurons to prune
    to_prune = all_scored[:n_to_prune]

    # Group by layer
    per_layer_prune = {l: [] for l in range(n_layers)}
    for _, l_idx, j in to_prune:
        per_layer_prune[l_idx].append(j)

    # Save original weights
    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    # Apply pruning
    for l_idx, layer in enumerate(model.model.layers):
        neurons = per_layer_prune[l_idx]
        if not neurons:
            continue
        mlp = layer.mlp
        mask_list = [True] * intermediate_dim
        for j in neurons:
            mask_list[j] = False
        mask = mx.array(mask_list).reshape(-1, 1).astype(mlp.gate_proj.weight.dtype)
        mlp.gate_proj.weight = mlp.gate_proj.weight * mask
        mlp.up_proj.weight = mlp.up_proj.weight * mask
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    # Measure perplexity
    ppl, loss = compute_perplexity(model, eval_data, batch_size)

    # Restore weights
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    # Score statistics for pruned neurons
    pruned_scores = [s for s, _, _ in to_prune]
    kept_scores = [s for s, _, _ in all_scored[n_to_prune:]]

    result = {
        "method": score_name,
        "n_pruned": n_to_prune,
        "total_neurons": total_neurons,
        "pct_pruned": n_to_prune / total_neurons * 100,
        "ppl": ppl,
        "loss": loss,
        "pruned_score_max": max(pruned_scores) if pruned_scores else 0,
        "pruned_score_min": min(pruned_scores) if pruned_scores else 0,
        "pruned_score_mean": sum(pruned_scores) / len(pruned_scores) if pruned_scores else 0,
        "kept_score_min": min(kept_scores) if kept_scores else 0,
        "per_layer_pruned": {l: len(per_layer_prune[l]) for l in range(n_layers)},
    }

    print(f"  {score_name}: {n_to_prune}/{total_neurons} pruned ({result['pct_pruned']:.1f}%), "
          f"ppl={ppl:.2f}, loss={loss:.4f}")

    return result


# ============================================================================
# Calibration sample sweep (kill criterion 2)
# ============================================================================

def calibration_sweep(model, tokenizer, eval_data, n_to_prune: int,
                      sample_counts=(8, 16, 32, 64, 128),
                      seq_len: int = 128, batch_size: int = 8):
    """Test Wanda scoring with varying calibration sample counts.

    Kill criterion 2: must work with <=100 calibration samples.
    """
    results = []

    for n_cal in sample_counts:
        print(f"\n--- Calibration samples: {n_cal} ---")
        cal_data, _ = load_wikitext2_split(
            tokenizer, split="test", n_sequences=n_cal, seq_len=seq_len
        )

        # Profile activations
        activation_means = profile_activations(model, cal_data, batch_size)

        # Compute weight norms (same regardless of calibration)
        weight_norms = compute_weight_norms(model)

        # Wanda scores
        wanda_scores = compute_wanda_scores(weight_norms, activation_means)

        # Test pruning
        r = test_scored_pruning(
            model, wanda_scores, n_to_prune, eval_data,
            score_name=f"wanda_cal{n_cal}", batch_size=4
        )
        r["n_calibration_samples"] = n_cal
        results.append(r)

    return results


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(model_name: str = "Qwen/Qwen2.5-0.5B",
                   n_cal_sequences: int = 128,
                   n_eval_sequences: int = 64,
                   seq_len: int = 128,
                   batch_size: int = 8):
    """Run the full Wanda structured pruning experiment.

    Compares:
    1. Wanda scoring (weight_norm * activation_mean) -- THIS EXPERIMENT
    2. Activation-only scoring (parent experiment baseline: anti-signal)
    3. Weight-only scoring (||W_j||_2 only, no activation)
    4. Random pruning baseline (3 seeds)

    All at the same neuron count as parent experiment tau=0.05 (18,420 neurons).
    """
    output_dir = Path(__file__).parent
    results_path = output_dir / "results.json"

    # Load model
    model, tokenizer = load_model(model_name)

    # Load data
    print("\nLoading calibration data (WikiText-2 test split)...")
    cal_data, cal_provenance = load_wikitext2_split(
        tokenizer, split="test", n_sequences=n_cal_sequences, seq_len=seq_len
    )

    print("\nLoading evaluation data (WikiText-2 validation split)...")
    eval_data, eval_provenance = load_wikitext2_split(
        tokenizer, split="validation", n_sequences=n_eval_sequences, seq_len=seq_len
    )

    # Target: prune same number as parent experiment at tau=0.05
    N_PRUNE = 18_420  # from parent experiment

    # ---- Step 1: Compute weight norms (data-independent) ----
    print("\n" + "=" * 70)
    print("STEP 1: Weight norms")
    print("=" * 70)
    weight_norms = compute_weight_norms(model)

    for l_idx in range(len(weight_norms)):
        wn = weight_norms[l_idx]
        print(f"  Layer {l_idx:>2}: min={mx.min(wn).item():.4f}, "
              f"median={mx.sort(wn)[wn.shape[0]//2].item():.4f}, "
              f"max={mx.max(wn).item():.4f}, "
              f"mean={mx.mean(wn).item():.4f}")

    # ---- Step 2: Profile activations ----
    print("\n" + "=" * 70)
    print("STEP 2: Activation profiling")
    print("=" * 70)
    activation_means = profile_activations(model, cal_data, batch_size)

    # ---- Step 3: Compute all scoring methods ----
    print("\n" + "=" * 70)
    print("STEP 3: Scoring methods")
    print("=" * 70)

    wanda_scores = compute_wanda_scores(weight_norms, activation_means)

    # Print Wanda score statistics
    all_wanda = []
    all_act = []
    all_weight = []
    for l_idx in range(len(wanda_scores)):
        ws = wanda_scores[l_idx].tolist()
        am = activation_means[l_idx].tolist()
        wn = weight_norms[l_idx].tolist()
        all_wanda.extend(ws)
        all_act.extend(am)
        all_weight.extend(wn)

    all_wanda_sorted = sorted(all_wanda)
    all_act_sorted = sorted(all_act)
    all_weight_sorted = sorted(all_weight)
    n_total = len(all_wanda_sorted)

    def pct(vals, p):
        k = int((len(vals) - 1) * p / 100)
        return vals[k]

    print(f"\nWanda score distribution ({n_total} neurons):")
    print(f"  P1={pct(all_wanda_sorted, 1):.6f}, P5={pct(all_wanda_sorted, 5):.6f}, "
          f"P25={pct(all_wanda_sorted, 25):.6f}, P50={pct(all_wanda_sorted, 50):.6f}")
    print(f"  P75={pct(all_wanda_sorted, 75):.6f}, P95={pct(all_wanda_sorted, 95):.6f}, "
          f"P99={pct(all_wanda_sorted, 99):.6f}")
    print(f"  Min={all_wanda_sorted[0]:.6f}, Max={all_wanda_sorted[-1]:.6f}")

    # Spearman rank correlation (manual implementation, no scipy needed)
    def _ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks

    def spearman_rho(a, b):
        ra, rb = _ranks(a), _ranks(b)
        n = len(ra)
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n)) / n
        std_a = (sum((r - mean_a)**2 for r in ra) / n) ** 0.5
        std_b = (sum((r - mean_b)**2 for r in rb) / n) ** 0.5
        if std_a == 0 or std_b == 0:
            return 0.0
        return cov / (std_a * std_b)

    rho_act_wanda = spearman_rho(all_act, all_wanda)
    rho_wt_wanda = spearman_rho(all_weight, all_wanda)
    rho_act_wt = spearman_rho(all_act, all_weight)

    print(f"\nSpearman correlations:")
    print(f"  activation vs wanda: rho={rho_act_wanda:.4f}")
    print(f"  weight vs wanda:     rho={rho_wt_wanda:.4f}")
    print(f"  activation vs weight: rho={rho_act_wt:.4f}")

    # ---- Step 4: Baseline perplexity ----
    print("\n" + "=" * 70)
    print("STEP 4: Baseline perplexity")
    print("=" * 70)
    base_ppl, base_loss = compute_perplexity(model, eval_data, batch_size=4)
    print(f"  Baseline: ppl={base_ppl:.2f}, loss={base_loss:.4f}")

    # ---- Step 5: Test all scoring methods ----
    print("\n" + "=" * 70)
    print(f"STEP 5: Pruning comparison ({N_PRUNE} neurons = {N_PRUNE/n_total*100:.1f}%)")
    print("=" * 70)

    # 5a. Wanda scoring
    print("\n[Wanda scoring: ||W_j||_2 * mean|X_j|]")
    wanda_result = test_scored_pruning(
        model, wanda_scores, N_PRUNE, eval_data,
        score_name="wanda", batch_size=4
    )

    # 5b. Activation-only scoring (parent experiment signal)
    print("\n[Activation-only scoring: mean|X_j| (parent baseline)]")
    act_result = test_scored_pruning(
        model, activation_means, N_PRUNE, eval_data,
        score_name="activation_only", batch_size=4
    )

    # 5c. Weight-only scoring
    print("\n[Weight-only scoring: ||W_j||_2]")
    wt_result = test_scored_pruning(
        model, weight_norms, N_PRUNE, eval_data,
        score_name="weight_only", batch_size=4
    )

    # 5d. Random pruning baseline
    print("\n[Random pruning baseline (3 seeds)]")
    random_result = test_random_pruning(
        model, N_PRUNE, eval_data,
        n_seeds=3, batch_size=4
    )

    # ---- Step 6: Additional pruning levels ----
    print("\n" + "=" * 70)
    print("STEP 6: Wanda at multiple pruning levels")
    print("=" * 70)

    prune_fractions = [0.05, 0.10, 0.15, 0.20, 0.30]
    multi_results = []
    for frac in prune_fractions:
        n_prune = int(n_total * frac)
        print(f"\n  --- {frac*100:.0f}% pruning ({n_prune} neurons) ---")
        r = test_scored_pruning(
            model, wanda_scores, n_prune, eval_data,
            score_name=f"wanda_{frac*100:.0f}pct", batch_size=4
        )
        r["prune_fraction"] = frac
        multi_results.append(r)

    # ---- Step 7: Calibration sample sweep ----
    print("\n" + "=" * 70)
    print("STEP 7: Calibration sample sweep (kill criterion 2)")
    print("=" * 70)
    cal_sweep = calibration_sweep(
        model, tokenizer, eval_data, N_PRUNE,
        sample_counts=(8, 16, 32, 64, 128),
        seq_len=seq_len, batch_size=batch_size
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline ppl: {base_ppl:.2f}")
    print(f"\nAt {N_PRUNE} neurons pruned ({N_PRUNE/n_total*100:.1f}%):")
    print(f"  {'Method':<25} {'PPL':>10} {'Delta':>10} {'vs Random':>10}")
    print(f"  {'-'*55}")

    methods = [
        ("Wanda (W*A)", wanda_result["ppl"]),
        ("Activation only", act_result["ppl"]),
        ("Weight only", wt_result["ppl"]),
        ("Random (mean)", random_result["mean_ppl"]),
    ]

    random_ppl = random_result["mean_ppl"]
    for name, ppl in methods:
        delta = (ppl - base_ppl) / base_ppl * 100
        vs_random = ppl / random_ppl
        print(f"  {name:<25} {ppl:>10.2f} {delta:>+9.1f}% {vs_random:>9.3f}x")

    # Kill criterion checks
    print(f"\n{'='*70}")
    print("KILL CRITERION CHECKS")
    print(f"{'='*70}")

    # KC1: Wanda >2x better than random
    wanda_ppl = wanda_result["ppl"]
    ratio = wanda_ppl / random_ppl
    kc1_pass = ratio < 0.5  # Wanda must be <50% of random ppl elevation
    # More precisely: is the ppl elevation from baseline 2x smaller for Wanda?
    wanda_elevation = wanda_ppl - base_ppl
    random_elevation = random_ppl - base_ppl
    elevation_ratio = wanda_elevation / random_elevation if random_elevation > 0 else float('inf')
    kc1_pass_strict = elevation_ratio < 0.5

    print(f"\nKC1: Wanda >2x better than random?")
    print(f"  Wanda ppl elevation:  {wanda_elevation:.2f} (ppl {wanda_ppl:.2f})")
    print(f"  Random ppl elevation: {random_elevation:.2f} (ppl {random_ppl:.2f})")
    print(f"  Elevation ratio: {elevation_ratio:.3f} (need <0.5 for 2x better)")
    print(f"  --> {'PASS' if kc1_pass_strict else 'KILL'}")

    # KC2: Works with <=100 calibration samples
    cal_ppls = {r["n_calibration_samples"]: r["ppl"] for r in cal_sweep}
    best_small_cal = min(r["ppl"] for r in cal_sweep if r["n_calibration_samples"] <= 64)
    full_cal_ppl = wanda_ppl  # 128 samples
    cal_stability = abs(best_small_cal - full_cal_ppl) / full_cal_ppl * 100

    print(f"\nKC2: Works with <=100 calibration samples?")
    for r in cal_sweep:
        print(f"  {r['n_calibration_samples']:>3} samples: ppl={r['ppl']:.2f}")
    print(f"  Best with <=64 samples: ppl={best_small_cal:.2f}")
    print(f"  Full 128 samples: ppl={full_cal_ppl:.2f}")
    print(f"  Stability (<=64 vs 128): {cal_stability:.1f}% difference")
    kc2_pass = cal_stability < 20  # Somewhat arbitrary but reasonable
    print(f"  --> {'PASS' if kc2_pass else 'KILL'}")

    # ---- Save results ----
    output = {
        "model": model_name,
        "calibration": cal_provenance,
        "evaluation": eval_provenance,
        "baseline_ppl": base_ppl,
        "baseline_loss": base_loss,
        "n_prune_target": N_PRUNE,
        "total_neurons": n_total,
        "correlations": {
            "activation_vs_wanda": {"rho": rho_act_wanda},
            "weight_vs_wanda": {"rho": rho_wt_wanda},
            "activation_vs_weight": {"rho": rho_act_wt},
        },
        "comparison": {
            "wanda": wanda_result,
            "activation_only": act_result,
            "weight_only": wt_result,
            "random": {
                "mean_ppl": random_result["mean_ppl"],
                "std_ppl": random_result["std_ppl"],
                "per_seed": random_result["per_seed"],
            },
        },
        "wanda_multi_level": multi_results,
        "calibration_sweep": cal_sweep,
        "kill_criteria": {
            "kc1_wanda_vs_random": {
                "elevation_ratio": elevation_ratio,
                "threshold": 0.5,
                "pass": kc1_pass_strict,
            },
            "kc2_calibration_efficiency": {
                "stability_pct": cal_stability,
                "threshold": 20.0,
                "pass": kc2_pass,
            },
        },
    }

    def make_serializable(obj):
        if isinstance(obj, mx.array):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(output), f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    return output


if __name__ == "__main__":
    run_experiment()
