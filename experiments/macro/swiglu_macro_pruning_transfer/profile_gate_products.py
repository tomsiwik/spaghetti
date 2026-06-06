"""Profile gate products on pretrained Qwen2.5-0.5B (no aux sparsity loss).

This is the critical experiment: does the bimodal gate-product distribution
observed in micro experiments (with L1 sparsity loss, target 50%) also exist
in production models trained with standard cross-entropy only?

Architecture reference (Qwen2.5-0.5B):
  - 24 transformer layers
  - SwiGLU MLP per layer: out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
  - gate_proj: (4864, 896), up_proj: (4864, 896), down_proj: (896, 4864)
  - 4864 neurons per layer to profile
  - Total: 24 * 4864 = 116,736 neurons

Kill criterion 1: gate-product distribution is NOT bimodal (no concentrated
low-magnitude region suitable for pruning).

Data pipeline (REVISION v2):
  - Calibration: WikiText-2-raw-v1 TEST split (loaded via HuggingFace datasets)
  - Evaluation: WikiText-2-raw-v1 VALIDATION split (genuinely held-out)
  - Previous version used 16 hardcoded prompts; this was corrected per review.
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
# Model loading via mlx-lm
# ============================================================================

def load_model(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Load a pretrained model and tokenizer via mlx-lm."""
    from mlx_lm import load
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    print(f"  Layers: {len(model.model.layers)}")
    mlp = model.model.layers[0].mlp
    gate_shape = mlp.gate_proj.weight.shape
    print(f"  MLP: gate_proj {gate_shape}, up_proj {mlp.up_proj.weight.shape}")
    print(f"  Intermediate dim (neurons per layer): {gate_shape[0]}")
    return model, tokenizer


# ============================================================================
# WikiText-2 data loading
# ============================================================================

def load_wikitext2_split(tokenizer, split: str = "test",
                          n_sequences: int = 128, seq_len: int = 128):
    """Load WikiText-2-raw-v1 split and tokenize into fixed-length sequences.

    Args:
        tokenizer: HuggingFace tokenizer (from mlx-lm)
        split: "test" or "validation"
        n_sequences: number of sequences to produce
        seq_len: tokens per sequence

    Returns:
        mx.array of shape (n_sequences, seq_len), dict with provenance info
    """
    from datasets import load_dataset

    print(f"Loading WikiText-2-raw-v1 ({split} split)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all non-empty lines into a single token stream
    all_text = "\n".join(line for line in ds["text"] if line.strip())
    print(f"  Raw text length: {len(all_text)} characters")

    # Tokenize the full text
    all_tokens = tokenizer.encode(all_text)
    n_total_tokens = len(all_tokens)
    print(f"  Total tokens: {n_total_tokens}")

    # We need n_sequences * seq_len tokens
    needed = n_sequences * seq_len
    if n_total_tokens < needed:
        print(f"  WARNING: only {n_total_tokens} tokens available, "
              f"need {needed}. Using {n_total_tokens // seq_len} sequences.")
        n_sequences = n_total_tokens // seq_len

    # Split into non-overlapping sequences
    sequences = []
    for i in range(n_sequences):
        start = i * seq_len
        seq = all_tokens[start:start + seq_len]
        sequences.append(seq)

    provenance = {
        "dataset": "wikitext-2-raw-v1",
        "split": split,
        "n_sequences": n_sequences,
        "seq_len": seq_len,
        "total_positions": n_sequences * seq_len,
        "total_tokens_in_split": n_total_tokens,
        "unique_tokens_used": len(set(t for seq in sequences for t in seq)),
        "raw_text_chars": len(all_text),
    }
    print(f"  Sequences: {n_sequences} x {seq_len} = {n_sequences * seq_len} positions")
    print(f"  Unique tokens: {provenance['unique_tokens_used']}")

    return mx.array(sequences), provenance


# ============================================================================
# Gate product profiling (the core experiment)
# ============================================================================

def profile_gate_products(model, calibration_data, batch_size: int = 8):
    """Profile per-neuron gate product magnitudes across all layers.

    For each layer l and neuron j (j in 0..4863):
      gate_product[l][j] = mean over all positions |SiLU(gate_proj_j(x)) * up_proj_j(x)|

    Also profiles individual components:
      gate_only[l][j] = mean |SiLU(gate_proj_j(x))|
      up_only[l][j] = mean |up_proj_j(x)|

    Returns dict with per-layer statistics.
    """
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    n_sequences, seq_len = calibration_data.shape

    print(f"\nProfiling gate products:")
    print(f"  Layers: {n_layers}, neurons/layer: {intermediate_dim}")
    print(f"  Sequences: {n_sequences}, seq_len: {seq_len}")
    print(f"  Total positions: {n_sequences * seq_len}")

    # Accumulators
    sum_gp_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    sum_gate_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    sum_up_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    sum_gp_sq = [mx.zeros(intermediate_dim) for _ in range(n_layers)]
    max_gp_abs = [mx.zeros(intermediate_dim) for _ in range(n_layers)]

    total_positions = 0
    n_batches = math.ceil(n_sequences / batch_size)

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
            # Run attention (we don't intercept this)
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, mask=None, cache=None)
            # Handle tuple return from attention
            if isinstance(x, tuple):
                x = x[0]
            x = residual + x

            # MLP with gate product interception
            residual = x
            x_norm = layer.post_attention_layernorm(x)

            mlp = layer.mlp
            gate_out = nn.silu(mlp.gate_proj(x_norm))   # (B, T, intermediate_dim)
            up_out = mlp.up_proj(x_norm)                  # (B, T, intermediate_dim)
            gate_product = gate_out * up_out               # (B, T, intermediate_dim)

            # Accumulate statistics
            gp_abs = mx.abs(gate_product)
            sum_gp_abs[l_idx] = sum_gp_abs[l_idx] + mx.sum(gp_abs, axis=(0, 1))
            sum_gate_abs[l_idx] = sum_gate_abs[l_idx] + mx.sum(mx.abs(gate_out), axis=(0, 1))
            sum_up_abs[l_idx] = sum_up_abs[l_idx] + mx.sum(mx.abs(up_out), axis=(0, 1))
            sum_gp_sq[l_idx] = sum_gp_sq[l_idx] + mx.sum(gp_abs ** 2, axis=(0, 1))
            max_gp_abs[l_idx] = mx.maximum(max_gp_abs[l_idx], mx.max(gp_abs, axis=(0, 1)))

            # Complete the MLP forward pass
            mlp_out = mlp.down_proj(gate_product)
            x = residual + mlp_out

            # Evaluate to free memory
            mx.eval(sum_gp_abs[l_idx], sum_gate_abs[l_idx], sum_up_abs[l_idx],
                    sum_gp_sq[l_idx], max_gp_abs[l_idx])

        if (batch_idx + 1) % 4 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{n_batches} ({elapsed:.1f}s)")

    # Compute final statistics
    results = []
    for l_idx in range(n_layers):
        gp_mean = sum_gp_abs[l_idx] / total_positions
        gate_mean = sum_gate_abs[l_idx] / total_positions
        up_mean = sum_up_abs[l_idx] / total_positions
        gp_var = (sum_gp_sq[l_idx] / total_positions) - (gp_mean ** 2)
        gp_std = mx.sqrt(mx.maximum(gp_var, mx.array(0.0)))

        mx.eval(gp_mean, gate_mean, up_mean, gp_std, max_gp_abs[l_idx])

        results.append({
            "layer": l_idx,
            "gate_product_mean_abs": gp_mean,
            "gate_only_mean_abs": gate_mean,
            "up_only_mean_abs": up_mean,
            "gate_product_std": gp_std,
            "gate_product_max_abs": max_gp_abs[l_idx],
            "n_neurons": intermediate_dim,
        })

    elapsed = time.time() - t0
    print(f"\nProfiling complete in {elapsed:.1f}s ({total_positions} positions)")
    return results


# ============================================================================
# Distribution analysis
# ============================================================================

def analyze_distributions(profiles: list[dict], thresholds=None):
    """Analyze gate product distributions for bimodality and pruning potential.

    Key metrics:
    1. Fraction of neurons below each threshold (pruning potential)
    2. Distribution statistics (min, p10, p25, median, p75, p90, max)
    3. Bimodality coefficient: BC = (skew^2 + 1) / kurtosis
       BC > 5/9 (~0.555) suggests bimodality (SAS criterion)
    4. Dip statistic (Hartigan's dip test approximation)
    """
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    analysis = {
        "per_layer": [],
        "aggregate": {},
        "thresholds": {},
    }

    all_means = []

    for prof in profiles:
        l_idx = prof["layer"]
        gp_mean = prof["gate_product_mean_abs"]
        gate_mean = prof["gate_only_mean_abs"]
        up_mean = prof["up_only_mean_abs"]
        n = prof["n_neurons"]

        # Convert to numpy-like for statistics
        gp_vals = gp_mean.tolist()
        gate_vals = gate_mean.tolist()
        up_vals = up_mean.tolist()

        all_means.extend(gp_vals)

        sorted_gp = sorted(gp_vals)
        n_vals = len(sorted_gp)

        def percentile(vals, p):
            k = (n_vals - 1) * p / 100.0
            f = int(k)
            c = f + 1 if f + 1 < n_vals else f
            d = k - f
            return vals[f] + d * (vals[c] - vals[f])

        # Basic statistics
        layer_stats = {
            "layer": l_idx,
            "n_neurons": n,
            "gp_min": sorted_gp[0],
            "gp_p10": percentile(sorted_gp, 10),
            "gp_p25": percentile(sorted_gp, 25),
            "gp_median": percentile(sorted_gp, 50),
            "gp_p75": percentile(sorted_gp, 75),
            "gp_p90": percentile(sorted_gp, 90),
            "gp_max": sorted_gp[-1],
            "gp_mean": sum(gp_vals) / n,
            "gate_only_min": min(gate_vals),
            "gate_only_median": sorted(gate_vals)[n // 2],
            "up_only_min": min(up_vals),
            "up_only_median": sorted(up_vals)[n // 2],
        }

        # Threshold analysis
        for tau in thresholds:
            below = sum(1 for v in gp_vals if v < tau)
            layer_stats[f"below_{tau}"] = below
            layer_stats[f"pct_below_{tau}"] = below / n * 100

        # Bimodality coefficient (Sarle's bimodality coefficient)
        # BC = (skewness^2 + 1) / kurtosis
        mean_gp = sum(gp_vals) / n
        m2 = sum((v - mean_gp)**2 for v in gp_vals) / n
        m3 = sum((v - mean_gp)**3 for v in gp_vals) / n
        m4 = sum((v - mean_gp)**4 for v in gp_vals) / n

        if m2 > 0:
            std = m2 ** 0.5
            skewness = m3 / (std ** 3) if std > 0 else 0
            kurtosis = m4 / (m2 ** 2) if m2 > 0 else 0
            # Excess kurtosis for normal = 3
            if kurtosis > 0:
                bc = (skewness**2 + 1) / kurtosis
            else:
                bc = 0
        else:
            skewness = 0
            kurtosis = 0
            bc = 0

        layer_stats["skewness"] = skewness
        layer_stats["kurtosis"] = kurtosis
        layer_stats["bimodality_coefficient"] = bc
        layer_stats["is_bimodal"] = bc > 0.555  # SAS criterion

        analysis["per_layer"].append(layer_stats)

    # Aggregate across all layers
    all_means_sorted = sorted(all_means)
    n_total = len(all_means_sorted)

    def agg_percentile(p):
        k = (n_total - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < n_total else f
        d = k - f
        return all_means_sorted[f] + d * (all_means_sorted[c] - all_means_sorted[f])

    analysis["aggregate"] = {
        "n_total_neurons": n_total,
        "gp_min": all_means_sorted[0],
        "gp_p1": agg_percentile(1),
        "gp_p5": agg_percentile(5),
        "gp_p10": agg_percentile(10),
        "gp_p25": agg_percentile(25),
        "gp_median": agg_percentile(50),
        "gp_p75": agg_percentile(75),
        "gp_p90": agg_percentile(90),
        "gp_p95": agg_percentile(95),
        "gp_p99": agg_percentile(99),
        "gp_max": all_means_sorted[-1],
        "gp_mean": sum(all_means) / n_total,
    }

    for tau in thresholds:
        below = sum(1 for v in all_means if v < tau)
        analysis["thresholds"][str(tau)] = {
            "n_below": below,
            "pct_below": below / n_total * 100,
        }

    # Aggregate bimodality
    mean_all = sum(all_means) / n_total
    m2 = sum((v - mean_all)**2 for v in all_means) / n_total
    m3 = sum((v - mean_all)**3 for v in all_means) / n_total
    m4 = sum((v - mean_all)**4 for v in all_means) / n_total
    if m2 > 0:
        std = m2 ** 0.5
        skew = m3 / (std ** 3)
        kurt = m4 / (m2 ** 2)
        bc = (skew**2 + 1) / kurt if kurt > 0 else 0
    else:
        skew = kurt = bc = 0
    analysis["aggregate"]["skewness"] = skew
    analysis["aggregate"]["kurtosis"] = kurt
    analysis["aggregate"]["bimodality_coefficient"] = bc
    analysis["aggregate"]["is_bimodal_sas"] = bc > 0.555

    return analysis


# ============================================================================
# Pruning test (zero-shot: just zero out neurons and measure perplexity)
# ============================================================================

def compute_perplexity(model, data, batch_size: int = 4):
    """Compute perplexity on held-out data."""
    n_sequences, seq_len = data.shape
    total_loss = 0.0
    total_tokens = 0
    n_batches = math.ceil(n_sequences / batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_sequences)
        batch = data[start:end]

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)

        # Cross-entropy loss
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='sum')
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


def test_pruning(model, profiles, eval_data, thresholds=None, batch_size: int = 4):
    """Test gate-product pruning at various thresholds.

    For each threshold, zeros out gate_proj and up_proj rows for neurons
    with mean |gate_product| < threshold, then measures perplexity.

    NOTE: This modifies model weights in-place. Save originals first.
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

    n_layers = len(model.model.layers)

    # Save original weights
    print("Saving original weights...")
    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    # Baseline perplexity
    print("Computing baseline perplexity...")
    base_ppl, base_loss = compute_perplexity(model, eval_data, batch_size)
    print(f"  Baseline: ppl={base_ppl:.2f}, loss={base_loss:.4f}")

    results = {"baseline_ppl": base_ppl, "baseline_loss": base_loss, "pruning": []}

    for tau in thresholds:
        # Restore original weights
        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight = mx.array(original_weights[l_idx]["gate_proj"])
            mlp.up_proj.weight = mx.array(original_weights[l_idx]["up_proj"])
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        # Apply pruning: zero out rows for neurons below threshold
        total_pruned = 0
        total_neurons = 0
        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            gp_mean = profiles[l_idx]["gate_product_mean_abs"]

            mask = gp_mean >= tau  # True = keep, False = prune
            mx.eval(mask)

            n_pruned = int(mx.sum(~mask).item())
            n_neurons = gp_mean.shape[0]
            total_pruned += n_pruned
            total_neurons += n_neurons

            if n_pruned > 0:
                # Zero out pruned neuron rows in gate_proj and up_proj
                mask_expanded = mask.reshape(-1, 1)  # (intermediate_dim, 1)
                new_gate = mlp.gate_proj.weight * mask_expanded
                new_up = mlp.up_proj.weight * mask_expanded
                mlp.gate_proj.weight = new_gate
                mlp.up_proj.weight = new_up
                mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        pct_pruned = total_pruned / total_neurons * 100
        print(f"\nThreshold tau={tau}: {total_pruned}/{total_neurons} pruned ({pct_pruned:.1f}%)")

        # Measure perplexity
        pruned_ppl, pruned_loss = compute_perplexity(model, eval_data, batch_size)
        delta_ppl = (pruned_ppl - base_ppl) / base_ppl * 100
        print(f"  Pruned: ppl={pruned_ppl:.2f} (delta={delta_ppl:+.2f}%), loss={pruned_loss:.4f}")

        results["pruning"].append({
            "threshold": tau,
            "total_pruned": total_pruned,
            "total_neurons": total_neurons,
            "pct_pruned": pct_pruned,
            "pruned_ppl": pruned_ppl,
            "pruned_loss": pruned_loss,
            "delta_ppl_pct": delta_ppl,
        })

    # Restore original weights
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    return results


# ============================================================================
# Random pruning baseline (control experiment)
# ============================================================================

def test_random_pruning(model, n_neurons_to_prune: int, eval_data,
                         n_seeds: int = 3, batch_size: int = 4):
    """Random pruning baseline: prune the same number of neurons randomly.

    For each random seed, randomly select n_neurons_to_prune neurons across
    all layers, zero their gate_proj and up_proj rows, and measure perplexity.

    This validates whether gate-product profiling provides signal above chance.
    """
    n_layers = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    total_neurons = n_layers * intermediate_dim

    # Save original weights
    print(f"\nRandom pruning baseline: {n_neurons_to_prune}/{total_neurons} neurons")
    print(f"  Running {n_seeds} random seeds...")

    original_weights = []
    for layer in model.model.layers:
        mlp = layer.mlp
        original_weights.append({
            "gate_proj": mx.array(mlp.gate_proj.weight),
            "up_proj": mx.array(mlp.up_proj.weight),
        })

    results = []

    for seed in range(n_seeds):
        # Restore original weights
        for l_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight = mx.array(original_weights[l_idx]["gate_proj"])
            mlp.up_proj.weight = mx.array(original_weights[l_idx]["up_proj"])
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        # Randomly select neurons to prune (layer_idx, neuron_idx)
        rng = random.Random(seed + 1000)  # offset to avoid collision with other seeds
        all_neuron_ids = [(l, j) for l in range(n_layers) for j in range(intermediate_dim)]
        selected = rng.sample(all_neuron_ids, min(n_neurons_to_prune, total_neurons))

        # Group by layer for efficient masking
        per_layer_prune = {l: [] for l in range(n_layers)}
        for l, j in selected:
            per_layer_prune[l].append(j)

        for l_idx, layer in enumerate(model.model.layers):
            neurons_to_zero = per_layer_prune[l_idx]
            if not neurons_to_zero:
                continue
            mlp = layer.mlp
            # Build boolean mask: True = keep, False = prune
            mask_list = [True] * intermediate_dim
            for j in neurons_to_zero:
                mask_list[j] = False
            mask = mx.array(mask_list)
            mask_expanded = mask.reshape(-1, 1).astype(mlp.gate_proj.weight.dtype)
            mlp.gate_proj.weight = mlp.gate_proj.weight * mask_expanded
            mlp.up_proj.weight = mlp.up_proj.weight * mask_expanded
            mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

        # Measure perplexity
        ppl, loss = compute_perplexity(model, eval_data, batch_size)
        print(f"  Seed {seed}: ppl={ppl:.2f}, loss={loss:.4f}")
        results.append({"seed": seed, "ppl": ppl, "loss": loss})

    # Restore original weights
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp.gate_proj.weight = original_weights[l_idx]["gate_proj"]
        mlp.up_proj.weight = original_weights[l_idx]["up_proj"]
        mx.eval(mlp.gate_proj.weight, mlp.up_proj.weight)

    mean_ppl = sum(r["ppl"] for r in results) / len(results)
    std_ppl = (sum((r["ppl"] - mean_ppl)**2 for r in results) / len(results)) ** 0.5

    summary = {
        "n_neurons_pruned": n_neurons_to_prune,
        "n_seeds": n_seeds,
        "per_seed": results,
        "mean_ppl": mean_ppl,
        "std_ppl": std_ppl,
    }
    print(f"  Random pruning: mean ppl={mean_ppl:.2f} +/- {std_ppl:.2f}")
    return summary


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiment(model_name: str = "Qwen/Qwen2.5-0.5B",
                   n_cal_sequences: int = 128,
                   n_eval_sequences: int = 64,
                   seq_len: int = 128,
                   batch_size: int = 8,
                   run_pruning_test: bool = True):
    """Run the full gate-product profiling experiment.

    Data pipeline:
      - Calibration: WikiText-2-raw-v1 TEST split
      - Evaluation: WikiText-2-raw-v1 VALIDATION split (genuinely held-out)
    """

    output_dir = Path(__file__).parent
    results_path = output_dir / "results.json"

    # Load model
    model, tokenizer = load_model(model_name)

    # Load WikiText-2 data (calibration from test, evaluation from validation)
    print("\nLoading calibration data (WikiText-2 test split)...")
    cal_data, cal_provenance = load_wikitext2_split(
        tokenizer, split="test", n_sequences=n_cal_sequences, seq_len=seq_len
    )

    print("\nLoading evaluation data (WikiText-2 validation split)...")
    eval_data, eval_provenance = load_wikitext2_split(
        tokenizer, split="validation", n_sequences=n_eval_sequences, seq_len=seq_len
    )

    # Profile gate products
    profiles = profile_gate_products(model, cal_data, batch_size)

    # Analyze distributions
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)
    analysis = analyze_distributions(profiles)

    # Print summary
    agg = analysis["aggregate"]
    print(f"\nAggregate statistics ({agg['n_total_neurons']} neurons across all layers):")
    print(f"  Min:    {agg['gp_min']:.6f}")
    print(f"  P1:     {agg['gp_p1']:.6f}")
    print(f"  P5:     {agg['gp_p5']:.6f}")
    print(f"  P10:    {agg['gp_p10']:.6f}")
    print(f"  P25:    {agg['gp_p25']:.6f}")
    print(f"  Median: {agg['gp_median']:.6f}")
    print(f"  P75:    {agg['gp_p75']:.6f}")
    print(f"  P90:    {agg['gp_p90']:.6f}")
    print(f"  P95:    {agg['gp_p95']:.6f}")
    print(f"  P99:    {agg['gp_p99']:.6f}")
    print(f"  Max:    {agg['gp_max']:.6f}")
    print(f"  Mean:   {agg['gp_mean']:.6f}")
    print(f"  Skewness: {agg['skewness']:.4f}")
    print(f"  Kurtosis: {agg['kurtosis']:.4f}")
    print(f"  Bimodality coeff (SAS): {agg['bimodality_coefficient']:.4f} "
          f"({'BIMODAL' if agg['is_bimodal_sas'] else 'NOT bimodal'})")

    print(f"\nThreshold analysis:")
    for tau_str, info in analysis["thresholds"].items():
        tau = float(tau_str)
        print(f"  tau={tau:<6.3f}: {info['n_below']:>5} neurons below ({info['pct_below']:>5.1f}%)")

    print(f"\nPer-layer summary (gate product mean abs):")
    print(f"  {'Layer':>5} {'Min':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'Max':>10} {'BC':>8} {'Bimodal?':>9}")
    for ls in analysis["per_layer"]:
        print(f"  {ls['layer']:>5} {ls['gp_min']:>10.6f} {ls['gp_p25']:>10.6f} "
              f"{ls['gp_median']:>10.6f} {ls['gp_p75']:>10.6f} {ls['gp_max']:>10.6f} "
              f"{ls['bimodality_coefficient']:>8.4f} {'YES' if ls['is_bimodal'] else 'no':>9}")

    # Gate vs up component comparison
    print(f"\nGate (SiLU) vs Up component floors:")
    print(f"  {'Layer':>5} {'GP_min':>10} {'Gate_min':>10} {'Up_min':>10} {'GP/Gate':>10}")
    for ls in analysis["per_layer"]:
        ratio = ls['gp_min'] / ls['gate_only_min'] if ls['gate_only_min'] > 0 else 0
        print(f"  {ls['layer']:>5} {ls['gp_min']:>10.6f} {ls['gate_only_min']:>10.6f} "
              f"{ls['up_only_min']:>10.6f} {ratio:>10.3f}x")

    # Pruning test
    pruning_results = None
    random_pruning_results = None
    if run_pruning_test:
        print("\n" + "="*70)
        print("PRUNING TEST (gate-product profiled)")
        print("="*70)
        pruning_results = test_pruning(
            model, profiles, eval_data,
            thresholds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            batch_size=batch_size,
        )

        # Random pruning baseline at tau=0.05 neuron count
        # Find how many neurons are pruned at tau=0.05
        print("\n" + "="*70)
        print("RANDOM PRUNING BASELINE")
        print("="*70)
        n_pruned_at_005 = None
        for pr in pruning_results["pruning"]:
            if abs(pr["threshold"] - 0.05) < 0.001:
                n_pruned_at_005 = pr["total_pruned"]
                break
        if n_pruned_at_005 is None:
            # Compute it from profiles
            n_pruned_at_005 = 0
            for prof in profiles:
                gp = prof["gate_product_mean_abs"]
                n_pruned_at_005 += int(mx.sum(gp < 0.05).item())

        random_pruning_results = test_random_pruning(
            model, n_pruned_at_005, eval_data,
            n_seeds=3, batch_size=batch_size,
        )

    # Save results
    def make_serializable(obj):
        if isinstance(obj, mx.array):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    output = {
        "model": model_name,
        "calibration": make_serializable(cal_provenance),
        "evaluation": make_serializable(eval_provenance),
        "analysis": make_serializable(analysis),
    }
    if pruning_results:
        output["pruning"] = make_serializable(pruning_results)
    if random_pruning_results:
        output["random_pruning_baseline"] = make_serializable(random_pruning_results)

    # Save profiles summary (not full per-neuron data, too large)
    output["profiles_summary"] = []
    for prof in profiles:
        gp = prof["gate_product_mean_abs"]
        output["profiles_summary"].append({
            "layer": prof["layer"],
            "gp_mean": float(mx.mean(gp).item()),
            "gp_min": float(mx.min(gp).item()),
            "gp_max": float(mx.max(gp).item()),
            "gp_std": float(mx.std(gp).item()),
        })

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    # Kill criterion check
    print("\n" + "="*70)
    print("KILL CRITERION CHECK")
    print("="*70)

    # KC1: Is the distribution bimodal?
    n_bimodal_layers = sum(1 for ls in analysis["per_layer"] if ls["is_bimodal"])
    pct_below_005 = analysis["thresholds"]["0.05"]["pct_below"]

    print(f"\nKC1: Bimodal distribution?")
    print(f"  Bimodal layers (SAS BC>0.555): {n_bimodal_layers}/{len(analysis['per_layer'])}")
    print(f"  Aggregate BC: {agg['bimodality_coefficient']:.4f}")
    print(f"  Neurons below tau=0.05: {pct_below_005:.1f}%")

    if pct_below_005 > 5:
        print(f"  --> PASS: {pct_below_005:.1f}% below tau=0.05 (>5% threshold)")
    else:
        print(f"  --> KILL: only {pct_below_005:.1f}% below tau=0.05 (<5%)")
        print(f"       No concentrated low-magnitude region for pruning")

    if pruning_results:
        print(f"\nKC2: Pruning quality?")
        for pr in pruning_results["pruning"]:
            status = "PASS" if abs(pr["delta_ppl_pct"]) < 5 else "KILL"
            print(f"  tau={pr['threshold']}: {pr['pct_pruned']:.1f}% pruned, "
                  f"ppl delta={pr['delta_ppl_pct']:+.2f}% [{status}]")

    if random_pruning_results:
        print(f"\nRandom pruning baseline ({random_pruning_results['n_neurons_pruned']} neurons):")
        print(f"  Random mean ppl: {random_pruning_results['mean_ppl']:.2f} "
              f"+/- {random_pruning_results['std_ppl']:.2f}")
        # Find matching gate-product pruning result
        for pr in (pruning_results or {}).get("pruning", []):
            if abs(pr["total_pruned"] - random_pruning_results["n_neurons_pruned"]) < 100:
                print(f"  Gate-product pruning ppl: {pr['pruned_ppl']:.2f}")
                if random_pruning_results["mean_ppl"] > 0:
                    ratio = pr["pruned_ppl"] / random_pruning_results["mean_ppl"]
                    signal = "BETTER" if ratio < 1.0 else "WORSE"
                    print(f"  Ratio (profiled/random): {ratio:.3f}x ({signal})")
                break

    return output


if __name__ == "__main__":
    run_experiment()
