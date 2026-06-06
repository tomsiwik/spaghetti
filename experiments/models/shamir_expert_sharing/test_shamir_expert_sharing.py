"""Tests and experiment for Shamir secret sharing of expert weights.

This file IS the experiment. It validates:
1. Shamir primitives (polynomial creation, evaluation, Lagrange interpolation)
2. Exact reconstruction from k-of-n shares (kill: >2% quality degradation)
3. Reconstruction overhead (kill: >10% of forward pass time)
4. Fault tolerance (drop shares, reconstruct from remainder)
5. Expert blending via polynomial interpolation at non-share points
"""

import time
import random
import itertools

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import train, evaluate
from experiments.models.shamir_expert_sharing.shamir_expert_sharing import (
    create_shares, reconstruct_from_shares, lagrange_interpolate_at_zero,
    create_polynomial, evaluate_polynomial, evaluate_at_point,
)


# ---- Config ----
CFG = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)
TRAIN_STEPS = 500
SEEDS = [42, 123, 7]


# ========================================================================== #
#  Unit tests: Shamir primitives                                              #
# ========================================================================== #

def test_polynomial_roundtrip():
    """Polynomial evaluation at x=0 returns the secret."""
    print("=" * 60)
    print("test_polynomial_roundtrip")

    rng = np.random.default_rng(42)
    secret = rng.standard_normal(100).astype(np.float64)

    for degree in [1, 2, 4, 8]:
        coeffs = create_polynomial(secret, degree, rng)
        recovered = evaluate_polynomial(coeffs, 0.0)
        max_err = np.max(np.abs(recovered - secret))
        assert max_err < 1e-14, f"degree={degree}: max_err={max_err}"
        print(f"  degree={degree}: max_err={max_err:.2e}  OK")

    print("  PASSED\n")


def test_lagrange_reconstruction():
    """k shares reconstruct the secret via Lagrange interpolation."""
    print("=" * 60)
    print("test_lagrange_reconstruction")

    rng = np.random.default_rng(42)
    secret = rng.standard_normal(50).astype(np.float64)

    for k in [2, 3, 5]:
        n = k + 2  # some extra shares
        shares = create_shares(secret, k, n, seed=42)
        reconstructed = lagrange_interpolate_at_zero(shares[:k])
        max_err = np.max(np.abs(reconstructed - secret.flatten()))
        assert max_err < 1e-10, f"k={k}: max_err={max_err}"
        print(f"  k={k}, n={n}: max_err={max_err:.2e}  OK")

    print("  PASSED\n")


def test_any_k_subset_works():
    """Any k of n shares reconstructs the secret, not just the first k."""
    print("=" * 60)
    print("test_any_k_subset_works")

    rng = np.random.default_rng(42)
    secret = rng.standard_normal(30).astype(np.float64)
    k, n = 3, 7
    shares = create_shares(secret, k, n, seed=42)

    # Try all C(7,3) = 35 subsets
    num_tested = 0
    max_err_overall = 0.0
    for combo in itertools.combinations(range(n), k):
        selected = [shares[i] for i in combo]
        reconstructed = lagrange_interpolate_at_zero(selected)
        err = np.max(np.abs(reconstructed - secret.flatten()))
        max_err_overall = max(max_err_overall, err)
        num_tested += 1

    assert max_err_overall < 1e-8, f"max_err across all subsets: {max_err_overall}"
    print(f"  Tested {num_tested} subsets of C({n},{k}), max_err={max_err_overall:.2e}  OK")
    print("  PASSED\n")


def test_fewer_than_k_fails():
    """k-1 shares do NOT reconstruct the secret."""
    print("=" * 60)
    print("test_fewer_than_k_fails")

    rng = np.random.default_rng(42)
    secret = rng.standard_normal(30).astype(np.float64)
    k, n = 3, 5
    shares = create_shares(secret, k, n, seed=42)

    # Use only k-1 = 2 shares
    insufficient = shares[:k - 1]
    bad_reconstruction = lagrange_interpolate_at_zero(insufficient)
    err = np.max(np.abs(bad_reconstruction - secret.flatten()))
    print(f"  With k-1={k-1} shares: max_err={err:.4f} (should be large)")
    assert err > 0.01, f"k-1 shares should NOT reconstruct: err={err}"
    print("  PASSED\n")


# ========================================================================== #
#  Experiment: Quality preservation after reconstruction                       #
# ========================================================================== #

def run_quality_experiment():
    """Main experiment: train GPT, Shamir-share MLP weights, reconstruct, compare.

    Kill criteria:
    - Reconstructed expert >2% worse than original
    - k-of-n reconstruction overhead >10% of forward pass
    """
    print("=" * 70)
    print("EXPERIMENT: Shamir Expert Sharing Quality Preservation")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tok, block_size=32)
        val_ds = CharDataset(docs_val, tok, block_size=32)

        # 1. Train a standard GPT
        model = get_model("shamir_expert", n_shares=5, k_threshold=3,
                          **{**CFG, "vocab_size": tok.vocab_size})
        mx.eval(model.parameters())
        train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
              lr=3e-3, seed=seed, log_every=250)

        # 2. Evaluate original model
        original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        print(f"  Original val_loss: {original_loss:.6f}")

        # 3. Save original weights and create shares
        model.save_original_weights()
        shares_dict = model.create_shares(seed=seed)

        # 4. Test reconstruction with different k-of-n configs
        configs = [
            ("3-of-5 (first 3)", [0, 1, 2]),
            ("3-of-5 (last 3)", [2, 3, 4]),
            ("3-of-5 (sparse)", [0, 2, 4]),
            ("4-of-5", [0, 1, 2, 3]),
            ("5-of-5 (all)", [0, 1, 2, 3, 4]),
        ]

        seed_results = {"seed": seed, "original_loss": original_loss, "configs": {}}

        for config_name, indices in configs:
            model.reconstruct_from_shares(shares_dict, share_indices=indices)
            mx.eval(model.parameters())
            recon_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
            pct_change = 100.0 * (recon_loss - original_loss) / original_loss
            print(f"  {config_name}: val_loss={recon_loss:.6f} ({pct_change:+.4f}%)")
            seed_results["configs"][config_name] = {
                "val_loss": recon_loss,
                "pct_change": pct_change,
            }

        # 5. Measure weight reconstruction error (numerical)
        model.reconstruct_from_shares(shares_dict, share_indices=[0, 1, 2])
        mx.eval(model.parameters())
        max_weight_err = 0.0
        for layer_idx, layer in enumerate(model.layers):
            for name in ["fc1.weight", "fc2.weight"]:
                parts = name.split(".")
                param = getattr(layer.mlp, parts[0])
                w_recon = np.array(getattr(param, parts[1]).tolist(), dtype=np.float64)
                w_orig = model._original_weights[layer_idx][name]
                err = np.max(np.abs(w_recon - w_orig))
                max_weight_err = max(max_weight_err, err)
        seed_results["max_weight_error"] = max_weight_err
        print(f"  Max weight reconstruction error: {max_weight_err:.2e}")

        results.append(seed_results)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_configs = list(results[0]["configs"].keys())
    print(f"\n{'Config':<25} {'Mean Loss':>10} {'Mean %':>10} {'Max %':>10}")
    print("-" * 55)
    for config in all_configs:
        losses = [r["configs"][config]["val_loss"] for r in results]
        pcts = [r["configs"][config]["pct_change"] for r in results]
        print(f"{config:<25} {np.mean(losses):>10.6f} {np.mean(pcts):>+10.6f} {np.max(pcts):>+10.6f}")

    max_pct = max(
        r["configs"][c]["pct_change"]
        for r in results
        for c in all_configs
    )
    max_weight_errs = [r["max_weight_error"] for r in results]
    print(f"\nMax quality degradation across all configs/seeds: {max_pct:+.6f}%")
    print(f"Max weight error across seeds: {max(max_weight_errs):.2e}")
    print(f"KILL threshold: >2.0%  -->  {'KILLED' if max_pct > 2.0 else 'PASSED'}")

    return results


# ========================================================================== #
#  Experiment: Reconstruction overhead                                         #
# ========================================================================== #

def run_overhead_experiment():
    """Measure Shamir reconstruction time vs forward pass time.

    Kill criterion: reconstruction overhead >10% of forward pass.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Reconstruction Overhead")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    model = get_model("shamir_expert", n_shares=7, k_threshold=3,
                      **{**CFG, "vocab_size": tok.vocab_size})
    mx.eval(model.parameters())

    # Train briefly
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tok, block_size=32)
    val_ds = CharDataset(docs_val, tok, block_size=32)
    train(model, train_ds, val_ds, steps=200, batch_size=32, lr=3e-3, seed=42, log_every=200)

    # Create shares
    shares_dict = model.create_shares(seed=42)

    # Measure forward pass time (average over many runs)
    tokens = mx.zeros((32, 32), dtype=mx.int32)
    # Warmup
    for _ in range(5):
        logits = model(tokens)
        mx.eval(logits)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        logits = model(tokens)
        mx.eval(logits)
    forward_time = (time.perf_counter() - t0) / n_runs

    # Measure reconstruction time for different k values
    print(f"\n  Forward pass time: {forward_time*1000:.3f} ms")
    print()

    results = {}
    for k in [2, 3, 5]:
        n = 7
        if k > n:
            continue
        # Override model's k_threshold for this test
        model.k_threshold = k
        # Recreate shares with this k
        shares_dict_k = model.create_shares(seed=42)

        # Time the reconstruction
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.reconstruct_from_shares(shares_dict_k, share_indices=list(range(k)))
            mx.eval(model.parameters())
        recon_time = (time.perf_counter() - t0) / n_runs

        overhead_pct = 100.0 * recon_time / forward_time
        print(f"  k={k}-of-{n} reconstruction: {recon_time*1000:.3f} ms ({overhead_pct:.1f}% of forward)")
        results[f"k={k}"] = {
            "recon_time_ms": recon_time * 1000,
            "forward_time_ms": forward_time * 1000,
            "overhead_pct": overhead_pct,
        }

    # Kill check
    min_overhead = min(r["overhead_pct"] for r in results.values())
    max_overhead = max(r["overhead_pct"] for r in results.values())
    print(f"\n  Overhead range: {min_overhead:.1f}% - {max_overhead:.1f}%")
    print(f"  KILL threshold: >10% for k-of-n  -->  {'KILLED' if min_overhead > 10 else 'see per-k results'}")

    return results


# ========================================================================== #
#  Experiment: Fault tolerance (drop shares)                                   #
# ========================================================================== #

def run_fault_tolerance_experiment():
    """Test that dropping shares still allows reconstruction.

    The point: if 2 of 5 shares are "corrupted" (unavailable), the remaining
    3 still reconstruct the expert exactly.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Fault Tolerance (Drop Shares)")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    model = get_model("shamir_expert", n_shares=5, k_threshold=3,
                      **{**CFG, "vocab_size": tok.vocab_size})
    mx.eval(model.parameters())

    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tok, block_size=32)
    val_ds = CharDataset(docs_val, tok, block_size=32)
    train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
          lr=3e-3, seed=42, log_every=250)

    original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
    print(f"  Original val_loss: {original_loss:.6f}")

    model.save_original_weights()
    shares_dict = model.create_shares(seed=42)

    # Test all C(5,3) = 10 subsets
    import itertools
    all_subsets = list(itertools.combinations(range(5), 3))
    print(f"\n  Testing all {len(all_subsets)} subsets of 3-of-5:")

    worst_pct = -999.0
    for subset in all_subsets:
        model.reconstruct_from_shares(shares_dict, share_indices=list(subset))
        mx.eval(model.parameters())
        loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        pct = 100.0 * (loss - original_loss) / original_loss
        worst_pct = max(worst_pct, pct)
        print(f"    shares {subset}: val_loss={loss:.6f} ({pct:+.6f}%)")

    print(f"\n  Worst degradation across all subsets: {worst_pct:+.6f}%")
    print(f"  KILL threshold: >2.0%  -->  {'KILLED' if worst_pct > 2.0 else 'PASSED'}")

    return worst_pct


# ========================================================================== #
#  Experiment: Expert blending via polynomial interpolation                    #
# ========================================================================== #

def run_blending_experiment():
    """Test polynomial interpolation at non-share points for expert blending.

    Novel idea: evaluating the sharing polynomial at intermediate points
    produces weight-space interpolations that may give meaningful "blended" experts.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Expert Blending via Polynomial Interpolation")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    model = get_model("shamir_expert", n_shares=5, k_threshold=3,
                      **{**CFG, "vocab_size": tok.vocab_size})
    mx.eval(model.parameters())

    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tok, block_size=32)
    val_ds = CharDataset(docs_val, tok, block_size=32)
    train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
          lr=3e-3, seed=42, log_every=250)

    original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
    print(f"  Original val_loss (x=0): {original_loss:.6f}")

    model.save_original_weights()
    shares_dict = model.create_shares(seed=42)

    # Evaluate at various points along the polynomial
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.5]
    print(f"\n  {'Point':>8} {'Val Loss':>10} {'vs Original':>12}")
    print("  " + "-" * 35)

    for x in test_points:
        if x == 0.0:
            # Reconstruct at 0 = original
            model.reconstruct_from_shares(shares_dict, share_indices=[0, 1, 2])
        else:
            model.blend_at_point(shares_dict, target_x=x)
        mx.eval(model.parameters())
        loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        pct = 100.0 * (loss - original_loss) / original_loss
        print(f"  {x:>8.2f} {loss:>10.6f} {pct:>+11.4f}%")


# ========================================================================== #
#  Run all                                                                     #
# ========================================================================== #

if __name__ == "__main__":
    # Unit tests
    test_polynomial_roundtrip()
    test_lagrange_reconstruction()
    test_any_k_subset_works()
    test_fewer_than_k_fails()

    # Experiments
    quality_results = run_quality_experiment()
    overhead_results = run_overhead_experiment()
    fault_tolerance_worst = run_fault_tolerance_experiment()
    run_blending_experiment()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    # Quality kill check
    max_quality_deg = max(
        r["configs"][c]["pct_change"]
        for r in quality_results
        for c in r["configs"]
    )
    print(f"  Max quality degradation: {max_quality_deg:+.6f}% (kill: >2.0%)")
    print(f"  Quality verdict: {'KILLED' if max_quality_deg > 2.0 else 'PASSED'}")

    # Overhead kill check (use k=3 as the practical config)
    if "k=3" in overhead_results:
        oh = overhead_results["k=3"]["overhead_pct"]
        print(f"  k=3 reconstruction overhead: {oh:.1f}% of forward (kill: >10%)")
        print(f"  Overhead verdict: {'KILLED' if oh > 10.0 else 'PASSED'}")

    print(f"  Fault tolerance (worst 3-of-5): {fault_tolerance_worst:+.6f}% (kill: >2.0%)")
    print(f"  Fault tolerance verdict: {'KILLED' if fault_tolerance_worst > 2.0 else 'PASSED'}")
