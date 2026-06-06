"""Bloom Pre-Filter Experiment: Full evaluation pipeline.

Protocol:
1. Train BloomPrefilterGPT with standard softmax routing (Bloom inactive)
2. Train baseline CapsuleMoEGPT identically for comparison
3. Profile: run training data through trained model, build Bloom filters
4. Activate Bloom filters and measure:
   a. Expert elimination rate (kill criterion: must eliminate >=30%)
   b. False positive rate (kill criterion: must be <=20%)
   c. Quality impact: val loss with vs without Bloom filtering
5. Sweep Bloom filter parameters: m_bits in {64, 128, 256, 512, 1024}

Key insight: at micro scale (G=8), we need the Bloom filters to learn
which experts DON'T fire for given tokens. With character-level data,
the specialization may be weak (all groups fire for all tokens), making
elimination difficult. The hypothesis is that trained capsule groups
specialize enough for Bloom filters to capture the pattern.
"""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.models.bloom_prefilter.bloom_prefilter import (
    BloomPrefilterGPT,
    VectorizedBloomBank,
)
from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT


def train_model(model, train_ds, val_ds, steps=500, batch_size=32,
                lr=3e-3, seed=42, log_every=100):
    """Train a model, return metrics."""
    rng = random.Random(seed)

    def ntp_loss(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        return loss + model.aux_loss()

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    t0 = time.time()
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == steps:
            vl = evaluate(model, val_ds, batch_size)
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{steps} | loss {loss.item():.4f} | val {vl:.4f} | {elapsed:.1f}s")

    final_val = evaluate(model, val_ds, batch_size)
    elapsed = time.time() - t0
    return {"val_loss": final_val, "elapsed_s": elapsed}


def evaluate(model, dataset, batch_size=32, n_batches=10):
    """Evaluate average loss over n_batches."""
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        total += loss.item()
    return total / n_batches


def profile_model(model, train_ds, n_batches=20, batch_size=32, seed=0):
    """Profile model: run tokens through and build Bloom filters."""
    rng = random.Random(seed)
    for i in range(n_batches):
        inputs, _ = train_ds.get_batch(batch_size, rng)
        mx.eval(inputs)
        model.profile_batch(inputs)
    return model.get_bloom_diagnostics()


def measure_elimination(model, val_ds, n_batches=10, batch_size=32, seed=0):
    """Measure elimination rate on validation data."""
    rng = random.Random(seed)
    total_eliminated = 0.0
    total_pairs = 0
    per_layer_rates = {}

    for i in range(n_batches):
        inputs, _ = val_ds.get_batch(batch_size, rng)
        mx.eval(inputs)
        rates = model.get_elimination_rate(inputs)
        for layer_name, rate in rates.items():
            if layer_name not in per_layer_rates:
                per_layer_rates[layer_name] = []
            per_layer_rates[layer_name].append(rate)

    # Average across batches
    avg_rates = {k: sum(v) / len(v) for k, v in per_layer_rates.items()}
    overall = sum(avg_rates.values()) / len(avg_rates) if avg_rates else 0.0
    return {"per_layer": avg_rates, "overall": overall}


def measure_false_positive_rate(model, val_ds, n_batches=10, batch_size=32, seed=0):
    """Measure false positive rate of Bloom filtering.

    FPR = fraction of (token, group) pairs where:
    - Bloom says "maybe relevant" (survivor)
    - But group activation is below threshold (actually irrelevant)

    Among all pairs where the group is actually irrelevant.
    """
    rng = random.Random(seed)
    total_true_negatives = 0  # group truly irrelevant
    total_false_positives = 0  # Bloom says yes, group says no

    for i in range(n_batches):
        inputs, _ = val_ds.get_batch(batch_size, rng)
        mx.eval(inputs)

        B, T = inputs.shape
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for layer in model.layers:
            h = layer.norm2(x + layer.attn(layer.norm1(x)))
            mx.eval(h)

            pool = layer.capsule_pool
            # Get actual group activations
            activations = pool._get_group_activations(h)
            mx.eval(activations)
            threshold = pool.bloom_bank.activation_threshold

            # Get Bloom filter predictions
            bloom_mask = pool.bloom_bank.query_batch(h)
            mx.eval(bloom_mask)

            # For each (token, group): is group truly irrelevant?
            truly_irrelevant = (activations <= threshold)  # (B, T, G)
            bloom_says_yes = bloom_mask  # (B, T, G) bool

            mx.eval(truly_irrelevant, bloom_says_yes)

            n_truly_irrelevant = mx.sum(truly_irrelevant.astype(mx.float32)).item()
            n_false_pos = mx.sum(
                (bloom_says_yes.astype(mx.float32) * truly_irrelevant.astype(mx.float32))
            ).item()

            total_true_negatives += n_truly_irrelevant
            total_false_positives += n_false_pos

            x = layer(x)

    fpr = total_false_positives / max(total_true_negatives, 1)
    return {
        "fpr": fpr,
        "true_negatives": int(total_true_negatives),
        "false_positives": int(total_false_positives),
    }


def run_full_experiment(seed=42):
    """Run the complete Bloom pre-filter experiment."""
    print("=" * 70)
    print(f"BLOOM PRE-FILTER EXPERIMENT (seed={seed})")
    print("=" * 70)

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    results = {}

    # --- Train baseline CapsuleMoE ---
    print("\n--- Training Baseline CapsuleMoE ---")
    baseline = CapsuleMoEGPT(
        vocab_size=tokenizer.vocab_size, block_size=32,
        n_groups=8, n_capsules_per_group=32, top_k_groups=2,
    )
    mx.eval(baseline.parameters())
    n_params_baseline = sum(v.size for _, v in nn.utils.tree_flatten(baseline.trainable_parameters()))
    print(f"  Params: {n_params_baseline:,}")
    baseline_result = train_model(baseline, train_ds, val_ds, steps=500,
                                   seed=seed)
    results["baseline_val_loss"] = baseline_result["val_loss"]
    print(f"  Final val loss: {baseline_result['val_loss']:.4f}")

    # --- Train BloomPrefilterGPT (with Bloom inactive, same as baseline) ---
    print("\n--- Training BloomPrefilterGPT ---")
    bloom_model = BloomPrefilterGPT(
        vocab_size=tokenizer.vocab_size, block_size=32,
        n_groups=8, n_capsules_per_group=32, top_k_groups=2,
        m_bits=256, k_hash=4,
    )
    mx.eval(bloom_model.parameters())
    n_params_bloom = sum(v.size for _, v in nn.utils.tree_flatten(bloom_model.trainable_parameters()))
    print(f"  Params: {n_params_bloom:,}")
    bloom_result = train_model(bloom_model, train_ds, val_ds, steps=500,
                                seed=seed)
    results["bloom_trained_val_loss"] = bloom_result["val_loss"]
    print(f"  Final val loss (no Bloom): {bloom_result['val_loss']:.4f}")

    # --- Profile: build Bloom filters ---
    print("\n--- Profiling (building Bloom filters) ---")
    profile_diag = profile_model(bloom_model, train_ds, n_batches=30,
                                  batch_size=32, seed=seed)
    for layer_name, ld in profile_diag.items():
        print(f"  {layer_name}: fill={ld['mean_fill_ratio']:.3f}, "
              f"theo_fpr={ld['mean_theoretical_fpr']:.3f}, "
              f"profiled={ld['n_profiled']}")

    # --- Measure elimination rate ---
    print("\n--- Measuring Elimination Rate ---")
    elim = measure_elimination(bloom_model, val_ds, n_batches=20, batch_size=32)
    results["elimination"] = elim
    print(f"  Overall elimination rate: {elim['overall']*100:.1f}%")
    for layer_name, rate in elim["per_layer"].items():
        print(f"    {layer_name}: {rate*100:.1f}%")

    # --- Measure false positive rate ---
    print("\n--- Measuring False Positive Rate ---")
    fpr_result = measure_false_positive_rate(bloom_model, val_ds,
                                              n_batches=20, batch_size=32)
    results["fpr"] = fpr_result
    print(f"  FPR: {fpr_result['fpr']*100:.1f}%")
    print(f"  True negatives: {fpr_result['true_negatives']}")
    print(f"  False positives: {fpr_result['false_positives']}")

    # --- Quality with Bloom active ---
    print("\n--- Quality With Bloom Active ---")
    bloom_model.set_bloom_active(True)
    val_loss_bloom = evaluate(bloom_model, val_ds, batch_size=32, n_batches=20)
    results["bloom_active_val_loss"] = val_loss_bloom
    bloom_model.set_bloom_active(False)
    val_loss_no_bloom = evaluate(bloom_model, val_ds, batch_size=32, n_batches=20)
    results["no_bloom_val_loss"] = val_loss_no_bloom
    quality_delta = (val_loss_bloom - val_loss_no_bloom) / val_loss_no_bloom * 100
    results["quality_delta_pct"] = quality_delta
    print(f"  Without Bloom: {val_loss_no_bloom:.4f}")
    print(f"  With Bloom:    {val_loss_bloom:.4f}")
    print(f"  Quality delta: {quality_delta:+.2f}%")

    # --- m_bits sweep ---
    print("\n--- m_bits Sweep ---")
    m_sweep_results = {}
    for m_bits in [64, 128, 256, 512, 1024]:
        sweep_model = BloomPrefilterGPT(
            vocab_size=tokenizer.vocab_size, block_size=32,
            n_groups=8, n_capsules_per_group=32, top_k_groups=2,
            m_bits=m_bits, k_hash=4,
        )
        # Copy weights from trained model
        sweep_model.load_weights(list(zip(
            [k for k, _ in nn.utils.tree_flatten(bloom_model.trainable_parameters())],
            [v for _, v in nn.utils.tree_flatten(bloom_model.trainable_parameters())]
        )))
        mx.eval(sweep_model.parameters())

        # Profile
        profile_model(sweep_model, train_ds, n_batches=30, batch_size=32, seed=seed)

        # Measure elimination
        elim_sweep = measure_elimination(sweep_model, val_ds, n_batches=10, batch_size=32)

        # Measure FPR
        fpr_sweep = measure_false_positive_rate(sweep_model, val_ds,
                                                 n_batches=10, batch_size=32)

        # Measure quality
        sweep_model.set_bloom_active(True)
        vl_bloom = evaluate(sweep_model, val_ds, batch_size=32, n_batches=10)
        sweep_model.set_bloom_active(False)
        vl_no_bloom = evaluate(sweep_model, val_ds, batch_size=32, n_batches=10)
        q_delta = (vl_bloom - vl_no_bloom) / vl_no_bloom * 100

        m_sweep_results[m_bits] = {
            "elimination": elim_sweep["overall"],
            "fpr": fpr_sweep["fpr"],
            "quality_delta_pct": q_delta,
            "val_bloom": vl_bloom,
            "val_no_bloom": vl_no_bloom,
        }
        print(f"  m={m_bits:5d}: elim={elim_sweep['overall']*100:5.1f}%, "
              f"fpr={fpr_sweep['fpr']*100:5.1f}%, "
              f"quality={q_delta:+.2f}%")

    results["m_sweep"] = m_sweep_results

    # --- Kill criteria evaluation ---
    print("\n" + "=" * 70)
    print("KILL CRITERIA EVALUATION")
    print("=" * 70)

    kill_elim = elim["overall"] < 0.30
    kill_fpr = fpr_result["fpr"] > 0.20
    print(f"  Elimination rate: {elim['overall']*100:.1f}% (threshold: >=30%)")
    if kill_elim:
        print(f"    --> KILLED: insufficient filtering (<30%)")
    else:
        print(f"    --> PASSES: sufficient filtering (>=30%)")

    print(f"  False positive rate: {fpr_result['fpr']*100:.1f}% (threshold: <=20%)")
    if kill_fpr:
        print(f"    --> KILLED: too many false positives (>20%)")
    else:
        print(f"    --> PASSES: acceptable FPR (<=20%)")

    overall_verdict = "KILLED" if (kill_elim or kill_fpr) else "PASSES"
    print(f"\n  OVERALL VERDICT: {overall_verdict}")
    results["verdict"] = overall_verdict
    results["kill_elim"] = kill_elim
    results["kill_fpr"] = kill_fpr

    return results


def run_multiseed():
    """Run experiment across 3 seeds for statistical robustness."""
    all_results = {}
    for seed in [42, 123, 777]:
        all_results[seed] = run_full_experiment(seed)
        print()

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)
    print(f"{'Seed':>6} | {'Baseline':>10} | {'Bloom':>10} | {'Elim%':>8} | {'FPR%':>8} | {'Quality':>10} | {'Verdict'}")
    print("-" * 75)
    for seed, r in all_results.items():
        print(f"{seed:>6} | {r['baseline_val_loss']:>10.4f} | "
              f"{r.get('bloom_active_val_loss', 0):>10.4f} | "
              f"{r['elimination']['overall']*100:>7.1f}% | "
              f"{r['fpr']['fpr']*100:>7.1f}% | "
              f"{r.get('quality_delta_pct', 0):>+9.2f}% | "
              f"{r['verdict']}")

    # Averages
    avg_elim = sum(r["elimination"]["overall"] for r in all_results.values()) / 3
    avg_fpr = sum(r["fpr"]["fpr"] for r in all_results.values()) / 3
    avg_q = sum(r.get("quality_delta_pct", 0) for r in all_results.values()) / 3
    print("-" * 75)
    print(f"{'Mean':>6} | {'':>10} | {'':>10} | "
          f"{avg_elim*100:>7.1f}% | {avg_fpr*100:>7.1f}% | {avg_q:>+9.2f}%")

    return all_results


if __name__ == "__main__":
    results = run_multiseed()
