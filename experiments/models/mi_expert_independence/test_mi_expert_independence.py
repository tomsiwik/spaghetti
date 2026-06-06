"""Tests and experiment runner for MI Expert Independence diagnostic.

Tests:
1. KSG estimator validates on known distributions
2. Profile collection works on a trained model
3. Full experiment: MI vs cosine as predictors of composition quality
"""

import time
import random

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from experiments.models.mi_expert_independence.mi_expert_independence import (
    ksg_mi_1d_fast,
    ksg_mi,
    pairwise_cosine,
    pairwise_mi_activations,
    pairwise_mi_outputs_pca,
    profile_groups,
    compute_independence_metrics,
)


# ---------------------------------------------------------------------------
# Unit tests for KSG estimator
# ---------------------------------------------------------------------------

def test_ksg_independent():
    """MI of independent variables should be ~0."""
    print("=" * 60)
    print("test_ksg_independent")
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N)
    y = np.random.randn(N)
    mi = ksg_mi_1d_fast(x, y, k=5)
    print(f"  MI(independent) = {mi:.4f} (should be ~0)")
    assert mi < 0.15, f"MI of independent vars too high: {mi}"
    print("  PASSED\n")


def test_ksg_dependent():
    """MI of y = x + noise should be > 0."""
    print("=" * 60)
    print("test_ksg_dependent")
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N)
    y = x + 0.1 * np.random.randn(N)  # highly dependent
    mi = ksg_mi_1d_fast(x, y, k=5)
    print(f"  MI(y=x+noise) = {mi:.4f} (should be >>0)")
    assert mi > 0.5, f"MI of dependent vars too low: {mi}"
    print("  PASSED\n")


def test_ksg_nonlinear():
    """MI should detect nonlinear dependence that correlation misses."""
    print("=" * 60)
    print("test_ksg_nonlinear")
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N)
    y = x ** 2 + 0.1 * np.random.randn(N)  # nonlinear dependence
    # Pearson correlation should be near 0
    corr = np.corrcoef(x, y)[0, 1]
    mi = ksg_mi_1d_fast(x, y, k=5)
    print(f"  Pearson r(x, x^2+noise) = {corr:.4f} (should be ~0)")
    print(f"  MI(x, x^2+noise) = {mi:.4f} (should be >0)")
    assert abs(corr) < 0.15, f"Correlation too high for nonlinear test: {corr}"
    assert mi > 0.2, f"MI should detect nonlinear dependence: {mi}"
    print("  PASSED\n")


def test_ksg_multidim():
    """KSG on multi-dimensional inputs."""
    print("=" * 60)
    print("test_ksg_multidim")
    np.random.seed(42)
    N = 500
    d = 4
    x = np.random.randn(N, d)
    y = x @ np.random.randn(d, d) + 0.5 * np.random.randn(N, d)
    mi = ksg_mi(x, y, k=5)
    print(f"  MI(linear_dep, d={d}) = {mi:.4f} (should be >0)")
    assert mi > 0.1, f"MI too low for dependent multidim: {mi}"

    # Independent
    y_indep = np.random.randn(N, d)
    mi_indep = ksg_mi(x, y_indep, k=5)
    print(f"  MI(independent, d={d}) = {mi_indep:.4f} (should be ~0)")
    assert mi_indep < mi, "Independent MI should be less than dependent MI"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(seeds=(42, 123, 7)):
    """Full experiment: MI vs cosine as predictors of composition quality.

    Protocol:
    1. For each seed, train capsule_moe on joint data
    2. Profile group outputs and activations
    3. Compute cosine, MI-activation, MI-PCA metrics
    4. Vary routing (uniform vs learned, different top-k) to create
       varying composition quality
    5. Correlate metrics with quality (r-squared)
    """
    print("\n" + "=" * 70)
    print("MI EXPERT INDEPENDENCE EXPERIMENT")
    print("=" * 70)

    docs = load_names()
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED = {seed}")
        print(f"{'='*60}")

        tok = CharTokenizer(docs)
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tok, block_size=32)
        val_ds = CharDataset(docs_val, tok, block_size=32)

        # Train multiple model variants with different routing configs
        configs = [
            ("learned_k2", dict(top_k_groups=2, uniform_routing=False)),
            ("learned_k1", dict(top_k_groups=1, uniform_routing=False)),
            ("learned_k4", dict(top_k_groups=4, uniform_routing=False)),
            ("uniform", dict(top_k_groups=4, uniform_routing=True)),
        ]

        seed_data = []
        for config_name, config_kwargs in configs:
            print(f"\n--- Config: {config_name} (seed={seed}) ---")

            model = get_model(
                "capsule_moe",
                vocab_size=tok.vocab_size,
                block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=4, n_capsules_per_group=64,
                **config_kwargs,
            )
            mx.eval(model.parameters())

            # Train
            result = train(model, train_ds, val_ds, steps=500,
                          batch_size=32, lr=3e-3, seed=seed, log_every=250)
            val_loss = result["val_loss"]
            print(f"  val_loss = {val_loss:.4f}")

            # Profile
            t0 = time.time()
            profile = profile_groups(model, val_ds, n_batches=20,
                                    batch_size=32, seed=0)
            t_profile = time.time() - t0
            print(f"  profiling took {t_profile:.2f}s")

            # Compute metrics
            metrics = compute_independence_metrics(profile, k=3, d_pca=4)

            seed_data.append({
                "config": config_name,
                "seed": seed,
                "val_loss": val_loss,
                "metrics": metrics,
                "profile_time_s": t_profile,
            })

        all_results.extend(seed_data)

    # ---------------------------------------------------------------------------
    # Analysis: correlate metrics with quality
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # For each layer, compute correlation between metric and val_loss
    n_layers = 4
    for layer_idx in range(n_layers):
        layer_key = f"layer_{layer_idx}"
        print(f"\n--- {layer_key} ---")

        val_losses = []
        mean_cosines = []
        mean_mi_acts = []
        mean_mi_pcas = []
        t_cosines = []
        t_mi_acts = []
        t_mi_pcas = []

        for r in all_results:
            m = r["metrics"][layer_key]
            G = m["n_groups"]
            val_losses.append(r["val_loss"])

            # Extract upper triangle (pairwise, excluding diagonal)
            mask = np.triu(np.ones((G, G), dtype=bool), k=1)
            mean_cosines.append(np.mean(np.abs(m["cosine"][mask])))
            mean_mi_acts.append(np.mean(m["mi_activation"][mask]))
            mean_mi_pcas.append(np.mean(m["mi_pca"][mask]))
            t_cosines.append(m["time_cosine_s"])
            t_mi_acts.append(m["time_mi_activation_s"])
            t_mi_pcas.append(m["time_mi_pca_s"])

        val_losses = np.array(val_losses)
        mean_cosines = np.array(mean_cosines)
        mean_mi_acts = np.array(mean_mi_acts)
        mean_mi_pcas = np.array(mean_mi_pcas)

        # Compute r-squared for each metric vs val_loss
        def r_squared(x, y):
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            r = np.corrcoef(x, y)[0, 1]
            return r ** 2

        r2_cosine = r_squared(mean_cosines, val_losses)
        r2_mi_act = r_squared(mean_mi_acts, val_losses)
        r2_mi_pca = r_squared(mean_mi_pcas, val_losses)

        print(f"  Mean |cosine|: {np.mean(mean_cosines):.6f} (std={np.std(mean_cosines):.6f})")
        print(f"  Mean MI-act:   {np.mean(mean_mi_acts):.6f} (std={np.std(mean_mi_acts):.6f})")
        print(f"  Mean MI-PCA:   {np.mean(mean_mi_pcas):.6f} (std={np.std(mean_mi_pcas):.6f})")
        print(f"  r^2(|cosine|, val_loss) = {r2_cosine:.4f}")
        print(f"  r^2(MI-act, val_loss)   = {r2_mi_act:.4f}")
        print(f"  r^2(MI-PCA, val_loss)   = {r2_mi_pca:.4f}")
        print(f"  MI-act r^2 improvement over cosine: {r2_mi_act - r2_cosine:+.4f}")
        print(f"  MI-PCA r^2 improvement over cosine: {r2_mi_pca - r2_cosine:+.4f}")

    # Aggregate timing
    print(f"\n--- Computational Cost ---")
    total_t_cosine = np.mean([r["metrics"]["layer_0"]["time_cosine_s"] for r in all_results])
    total_t_mi_act = np.mean([r["metrics"]["layer_0"]["time_mi_activation_s"] for r in all_results])
    total_t_mi_pca = np.mean([r["metrics"]["layer_0"]["time_mi_pca_s"] for r in all_results])
    print(f"  Cosine:  {total_t_cosine:.4f}s")
    print(f"  MI-act:  {total_t_mi_act:.4f}s ({total_t_mi_act/max(total_t_cosine,1e-8):.1f}x cosine)")
    print(f"  MI-PCA:  {total_t_mi_pca:.4f}s ({total_t_mi_pca/max(total_t_cosine,1e-8):.1f}x cosine)")

    # Print full summary table
    print(f"\n--- Full Results Table ---")
    print(f"{'Config':<12} {'Seed':>4} {'Val Loss':>9} {'|Cos| L0':>10} {'MI-act L0':>10} {'MI-PCA L0':>10}")
    print("-" * 65)
    for r in all_results:
        m = r["metrics"]["layer_0"]
        G = m["n_groups"]
        mask = np.triu(np.ones((G, G), dtype=bool), k=1)
        print(f"{r['config']:<12} {r['seed']:>4} {r['val_loss']:>9.4f} "
              f"{np.mean(np.abs(m['cosine'][mask])):>10.6f} "
              f"{np.mean(m['mi_activation'][mask]):>10.6f} "
              f"{np.mean(m['mi_pca'][mask]):>10.6f}")

    # Kill criteria evaluation
    print(f"\n{'='*70}")
    print("KILL CRITERIA EVALUATION")
    print(f"{'='*70}")

    # Aggregate best r^2 improvement across layers
    best_mi_improvement = -999
    best_layer = None
    for layer_idx in range(n_layers):
        layer_key = f"layer_{layer_idx}"
        val_losses_arr = np.array([r["val_loss"] for r in all_results])
        mask = np.triu(np.ones((4, 4), dtype=bool), k=1)

        cos_vals = np.array([np.mean(np.abs(r["metrics"][layer_key]["cosine"][mask])) for r in all_results])
        mi_act_vals = np.array([np.mean(r["metrics"][layer_key]["mi_activation"][mask]) for r in all_results])
        mi_pca_vals = np.array([np.mean(r["metrics"][layer_key]["mi_pca"][mask]) for r in all_results])

        r2_cos = r_squared(cos_vals, val_losses_arr)
        r2_mi_act = r_squared(mi_act_vals, val_losses_arr)
        r2_mi_pca = r_squared(mi_pca_vals, val_losses_arr)

        best_mi = max(r2_mi_act, r2_mi_pca)
        improvement = best_mi - r2_cos
        if improvement > best_mi_improvement:
            best_mi_improvement = improvement
            best_layer = layer_key

    # Cost ratio
    cost_ratio_act = total_t_mi_act / max(total_t_cosine, 1e-8)
    cost_ratio_pca = total_t_mi_pca / max(total_t_cosine, 1e-8)
    max_cost_ratio = max(cost_ratio_act, cost_ratio_pca)

    print(f"\n  Kill criterion 1: r^2 improvement < 0.1")
    print(f"    Best MI r^2 improvement: {best_mi_improvement:+.4f} (layer: {best_layer})")
    if best_mi_improvement < 0.1:
        print(f"    VERDICT: KILL (improvement {best_mi_improvement:.4f} < 0.1)")
    else:
        print(f"    VERDICT: PASS")

    print(f"\n  Kill criterion 2: MI cost > 100x cosine")
    print(f"    MI-act cost ratio:  {cost_ratio_act:.1f}x")
    print(f"    MI-PCA cost ratio:  {cost_ratio_pca:.1f}x")
    if max_cost_ratio > 100:
        print(f"    VERDICT: KILL (cost {max_cost_ratio:.1f}x > 100x)")
    else:
        print(f"    VERDICT: PASS")

    return all_results


def r_squared(x, y):
    """Compute r-squared between two arrays."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return r ** 2


if __name__ == "__main__":
    # Unit tests
    test_ksg_independent()
    test_ksg_dependent()
    test_ksg_nonlinear()
    test_ksg_multidim()
    print("All KSG unit tests passed!\n")

    # Full experiment
    run_experiment(seeds=(42, 123, 7))
