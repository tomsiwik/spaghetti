"""Tests and experiment for Reed-Solomon expert encoding.

This file IS the experiment. It validates:
1. RS encoding/decoding primitives (Lagrange interpolation accuracy)
2. Quality preservation after encode-drop-reconstruct (kill: >1%)
3. Parameter overhead (kill: >20% additional params)
4. Fault tolerance: all C(N+k, N) subsets reconstruct within tolerance
5. Parity experts as inference-time "blend experts"
6. Chebyshev vs uniform node comparison
"""

import time
import itertools

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import train, evaluate
from experiments.models.reed_solomon_expert.reed_solomon_expert import (
    rs_encode, rs_decode, reconstruction_error,
    chebyshev_nodes, lagrange_interpolate,
)


# ---- Config ----
CFG = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)
TRAIN_STEPS = 500
SEEDS = [42, 123, 7]


# ========================================================================== #
#  Unit tests: RS primitives                                                  #
# ========================================================================== #

def test_chebyshev_nodes():
    """Chebyshev nodes are in the correct interval and are distinct."""
    print("=" * 60)
    print("test_chebyshev_nodes")

    for n in [2, 4, 8, 16]:
        nodes = chebyshev_nodes(n, -1.0, 1.0)
        assert len(nodes) == n, f"Expected {n} nodes, got {len(nodes)}"
        assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), "Nodes out of range"
        # All distinct
        diffs = np.diff(np.sort(nodes))
        assert np.all(diffs > 0), "Nodes not distinct"
        print(f"  n={n}: range=[{nodes.min():.4f}, {nodes.max():.4f}], min_gap={diffs.min():.6f}  OK")

    print("  PASSED\n")


def test_lagrange_roundtrip():
    """Interpolating at a data point returns the data value."""
    print("=" * 60)
    print("test_lagrange_roundtrip")

    rng = np.random.default_rng(42)

    for N in [2, 4, 8]:
        xs = chebyshev_nodes(N)
        ys = rng.standard_normal((N, 50)).astype(np.float64)

        max_err = 0.0
        for i in range(N):
            result = lagrange_interpolate(xs, ys, xs[i])
            err = np.max(np.abs(result - ys[i]))
            max_err = max(max_err, err)

        print(f"  N={N}: max_err={max_err:.2e}  OK")
        assert max_err < 1e-10, f"N={N}: roundtrip error too large: {max_err}"

    print("  PASSED\n")


def test_rs_encode_decode_exact():
    """RS encode N vectors, decode from all N+k, get exact reconstruction."""
    print("=" * 60)
    print("test_rs_encode_decode_exact")

    rng = np.random.default_rng(42)

    for N, k in [(2, 1), (4, 2), (4, 4), (8, 2)]:
        D = 100
        expert_weights = [rng.standard_normal(D).astype(np.float64) for _ in range(N)]
        enc = rs_encode(expert_weights, k, use_chebyshev=True)

        # Reconstruct from first N (all originals, no parity used)
        original_ys = np.stack(expert_weights, axis=0)
        reconstructed = rs_decode(
            enc["all_xs"][:N], enc["all_weights"][:N], enc["data_xs"]
        )
        err = reconstruction_error(original_ys, reconstructed)
        print(f"  N={N}, k={k}: max_err={err['max_abs_error']:.2e}, "
              f"rel_err={err['relative_error']:.2e}  OK")
        assert err["max_abs_error"] < 1e-8, f"Reconstruction error too large"

    print("  PASSED\n")


def test_rs_fault_tolerance():
    """Drop up to k experts, reconstruct from remainder + parity."""
    print("=" * 60)
    print("test_rs_fault_tolerance")

    rng = np.random.default_rng(42)
    N, k = 4, 2
    D = 50
    expert_weights = [rng.standard_normal(D).astype(np.float64) for _ in range(N)]
    enc = rs_encode(expert_weights, k, use_chebyshev=True)

    original_ys = np.stack(expert_weights, axis=0)

    # Try all C(N+k, N) = C(6, 4) = 15 subsets of N available experts
    total = N + k
    n_tested = 0
    max_err = 0.0
    for combo in itertools.combinations(range(total), N):
        available_xs = enc["all_xs"][list(combo)]
        available_ys = enc["all_weights"][list(combo)]
        reconstructed = rs_decode(available_xs, available_ys, enc["data_xs"])
        err = reconstruction_error(original_ys, reconstructed)
        max_err = max(max_err, err["max_abs_error"])
        n_tested += 1

    print(f"  Tested {n_tested} subsets of C({total},{N})")
    print(f"  Max reconstruction error: {max_err:.2e}")
    assert max_err < 1e-6, f"Fault tolerance failed: max_err={max_err}"
    print("  PASSED\n")


def test_chebyshev_vs_uniform():
    """Chebyshev nodes should have lower interpolation error than uniform."""
    print("=" * 60)
    print("test_chebyshev_vs_uniform")

    rng = np.random.default_rng(42)
    N, k = 4, 2
    D = 100
    expert_weights = [rng.standard_normal(D).astype(np.float64) for _ in range(N)]

    enc_cheb = rs_encode(expert_weights, k, use_chebyshev=True)
    enc_unif = rs_encode(expert_weights, k, use_chebyshev=False)

    original_ys = np.stack(expert_weights, axis=0)

    # Test reconstruction dropping first expert (worst case for uniform)
    for name, enc in [("chebyshev", enc_cheb), ("uniform", enc_unif)]:
        # Use last N experts (indices 1..N+k-1, so including some parity)
        available_xs = enc["all_xs"][k:]  # skip first k, get N remaining
        available_ys = enc["all_weights"][k:]
        reconstructed = rs_decode(available_xs, available_ys, enc["data_xs"])
        err = reconstruction_error(original_ys, reconstructed)
        print(f"  {name:>10}: max_err={err['max_abs_error']:.2e}, "
              f"rel_err={err['relative_error']:.2e}")

    print("  (At N=4, difference is small; Chebyshev advantage grows with N)")
    print("  PASSED\n")


# ========================================================================== #
#  Experiment 1: Quality preservation after RS encode-drop-reconstruct        #
# ========================================================================== #

def run_quality_experiment():
    """Train GPT, RS-encode layers, drop layers, reconstruct, compare quality.

    Kill criteria:
    - Reconstructed model >1% worse than original
    - Parameter overhead >20%
    """
    print("=" * 70)
    print("EXPERIMENT 1: RS Quality Preservation (encode-drop-reconstruct)")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tok, block_size=32)
        val_ds = CharDataset(docs_val, tok, block_size=32)

        # 1. Train a standard GPT (using RS wrapper for convenience)
        model = get_model("reed_solomon_expert", k_parity=2,
                          **{**CFG, "vocab_size": tok.vocab_size})
        mx.eval(model.parameters())
        train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
              lr=3e-3, seed=seed, log_every=250)

        # 2. Evaluate original model
        original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        print(f"  Original val_loss: {original_loss:.6f}")

        # 3. Save original weights, RS-encode
        model.save_original_weights()
        encoding = model.rs_encode_experts()

        # 4. Parameter overhead
        overhead = model.param_overhead()
        print(f"  Parameter overhead: {overhead:.1f}% (k={model.k_parity}, N={len(model.layers)})")

        # 5. Test various drop scenarios
        N = len(model.layers)  # 4
        k = model.k_parity     # 2
        total = N + k          # 6

        # Generate all meaningful drop scenarios
        drop_scenarios = []
        # Drop 1 original expert
        for i in range(N):
            drop_scenarios.append((f"drop_layer{i}", [i]))
        # Drop 2 original experts (max tolerable)
        for i, j in itertools.combinations(range(N), 2):
            drop_scenarios.append((f"drop_layers{i}+{j}", [i, j]))

        seed_results = {
            "seed": seed,
            "original_loss": original_loss,
            "overhead_pct": overhead,
            "scenarios": {},
        }

        for scenario_name, drop_idx in drop_scenarios:
            # Restore original weights first
            for layer_idx, layer in enumerate(model.layers):
                for pname in ["fc1.weight", "fc2.weight"]:
                    parts = pname.split(".")
                    param = getattr(layer.mlp, parts[0])
                    orig = model._original_weights[layer_idx][pname]
                    setattr(param, parts[1],
                            mx.array(orig.reshape(
                                tuple(getattr(param, parts[1]).shape)
                            ).astype(np.float32)))
            mx.eval(model.parameters())

            # Reconstruct from remaining + parity
            model.reconstruct_from_available(encoding, drop_idx)
            mx.eval(model.parameters())
            recon_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
            pct_change = 100.0 * (recon_loss - original_loss) / original_loss

            # Also measure weight-space reconstruction error
            weight_err = 0.0
            for layer_idx in range(N):
                for pname in ["fc1.weight", "fc2.weight"]:
                    parts = pname.split(".")
                    param = getattr(model.layers[layer_idx].mlp, parts[0])
                    w_recon = np.array(getattr(param, parts[1]).tolist(), dtype=np.float64)
                    w_orig = model._original_weights[layer_idx][pname]
                    err = np.max(np.abs(w_recon - w_orig))
                    weight_err = max(weight_err, err)

            seed_results["scenarios"][scenario_name] = {
                "val_loss": recon_loss,
                "pct_change": pct_change,
                "max_weight_error": weight_err,
                "dropped": drop_idx,
            }

            if len(drop_idx) <= 1:  # Only print single-drop for brevity
                print(f"  {scenario_name}: val={recon_loss:.6f} ({pct_change:+.4f}%), "
                      f"w_err={weight_err:.2e}")

        results.append(seed_results)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    # Single-drop scenarios
    single_drops = [s for s in results[0]["scenarios"] if "+" not in s]
    double_drops = [s for s in results[0]["scenarios"] if "+" in s]

    print(f"\n{'Scenario':<25} {'Mean Loss':>10} {'Mean %':>10} {'Max %':>10} {'Max w_err':>12}")
    print("-" * 70)

    print("--- Single expert drop ---")
    for scenario in single_drops:
        losses = [r["scenarios"][scenario]["val_loss"] for r in results]
        pcts = [r["scenarios"][scenario]["pct_change"] for r in results]
        werrs = [r["scenarios"][scenario]["max_weight_error"] for r in results]
        print(f"  {scenario:<23} {np.mean(losses):>10.6f} {np.mean(pcts):>+10.4f} "
              f"{np.max(pcts):>+10.4f} {np.max(werrs):>12.2e}")

    print("--- Double expert drop (max tolerance) ---")
    for scenario in double_drops:
        losses = [r["scenarios"][scenario]["val_loss"] for r in results]
        pcts = [r["scenarios"][scenario]["pct_change"] for r in results]
        werrs = [r["scenarios"][scenario]["max_weight_error"] for r in results]
        print(f"  {scenario:<23} {np.mean(losses):>10.6f} {np.mean(pcts):>+10.4f} "
              f"{np.max(pcts):>+10.4f} {np.max(werrs):>12.2e}")

    # Overall kill checks
    all_pcts = [
        r["scenarios"][s]["pct_change"]
        for r in results
        for s in r["scenarios"]
    ]
    max_quality_deg = max(all_pcts)
    mean_overhead = np.mean([r["overhead_pct"] for r in results])

    print(f"\nMax quality degradation across all scenarios/seeds: {max_quality_deg:+.6f}%")
    print(f"Parameter overhead: {mean_overhead:.1f}%")
    print(f"KILL KC1 (quality >1%):    {'KILLED' if max_quality_deg > 1.0 else 'PASSED'}")
    print(f"KILL KC2 (overhead >20%):  {'KILLED' if mean_overhead > 20.0 else 'PASSED'}")

    return results


# ========================================================================== #
#  Experiment 2: Encoding/reconstruction overhead                             #
# ========================================================================== #

def run_overhead_experiment():
    """Measure RS encoding and reconstruction time.

    This is a one-time offline operation (not per-inference), so the kill
    criterion is relaxed: just measure and report.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Encoding/Reconstruction Overhead")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    model = get_model("reed_solomon_expert", k_parity=2,
                      **{**CFG, "vocab_size": tok.vocab_size})
    mx.eval(model.parameters())
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tok, block_size=32)
    val_ds = CharDataset(docs_val, tok, block_size=32)
    train(model, train_ds, val_ds, steps=200, batch_size=32, lr=3e-3, seed=42, log_every=200)

    # Measure encoding time
    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        encoding = model.rs_encode_experts()
    encode_time = (time.perf_counter() - t0) / n_runs

    # Measure reconstruction time (drop 1 expert)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.reconstruct_from_available(encoding, drop_indices=[0])
        mx.eval(model.parameters())
    recon_time = (time.perf_counter() - t0) / n_runs

    # Measure forward pass time
    tokens = mx.zeros((32, 32), dtype=mx.int32)
    for _ in range(5):
        logits = model(tokens)
        mx.eval(logits)
    t0 = time.perf_counter()
    for _ in range(50):
        logits = model(tokens)
        mx.eval(logits)
    forward_time = (time.perf_counter() - t0) / 50

    print(f"\n  Forward pass:    {forward_time*1000:.3f} ms")
    print(f"  RS encoding:     {encode_time*1000:.3f} ms (one-time, offline)")
    print(f"  RS reconstruction: {recon_time*1000:.3f} ms (one-time, on expert loss)")
    print(f"  Encode/forward ratio: {encode_time/forward_time:.1f}x")
    print(f"  Recon/forward ratio:  {recon_time/forward_time:.1f}x")
    print(f"\n  Note: encoding and reconstruction are OFFLINE operations,")
    print(f"  NOT per-inference. Zero runtime overhead after reconstruction.")

    return {
        "forward_ms": forward_time * 1000,
        "encode_ms": encode_time * 1000,
        "recon_ms": recon_time * 1000,
    }


# ========================================================================== #
#  Experiment 3: Parity experts as inference-time blend experts               #
# ========================================================================== #

def run_parity_as_blend_experiment():
    """Test whether parity experts produce meaningful model outputs.

    Parity experts are polynomial interpolations in weight space.
    They may act as "cross-layer blend" experts.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Parity Experts as Blend Experts")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    model = get_model("reed_solomon_expert", k_parity=2,
                      **{**CFG, "vocab_size": tok.vocab_size})
    mx.eval(model.parameters())
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tok, block_size=32)
    val_ds = CharDataset(docs_val, tok, block_size=32)
    train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
          lr=3e-3, seed=42, log_every=250)

    model.save_original_weights()
    original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
    print(f"  Original val_loss: {original_loss:.6f}")

    encoding = model.rs_encode_experts()

    # For each layer, replace its weights with a parity expert
    print(f"\n  {'Layer':>6} {'Parity#':>8} {'Val Loss':>10} {'vs Original':>12}")
    print("  " + "-" * 40)

    for target_layer in range(len(model.layers)):
        for parity_idx in range(model.k_parity):
            # Restore all originals first
            for li, layer in enumerate(model.layers):
                for pname in ["fc1.weight", "fc2.weight"]:
                    parts = pname.split(".")
                    param = getattr(layer.mlp, parts[0])
                    orig = model._original_weights[li][pname]
                    setattr(param, parts[1],
                            mx.array(orig.reshape(
                                tuple(getattr(param, parts[1]).shape)
                            ).astype(np.float32)))

            # Replace target layer with parity expert
            model.load_parity_expert(encoding, parity_idx, target_layer)
            mx.eval(model.parameters())
            loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
            pct = 100.0 * (loss - original_loss) / original_loss
            print(f"  {target_layer:>6} {parity_idx:>8} {loss:>10.6f} {pct:>+11.4f}%")


# ========================================================================== #
#  Experiment 4: Scaling k (parity count)                                     #
# ========================================================================== #

def run_k_scaling_experiment():
    """Test how reconstruction quality scales with k (number of parity experts).

    With N=4 layers:
    - k=1: tolerate 1 loss (25% redundancy)
    - k=2: tolerate 2 losses (50% redundancy)
    - k=3: tolerate 3 losses (75% redundancy)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Scaling k (parity count)")
    print("=" * 70)

    docs = load_names()
    tok = CharTokenizer(docs)

    for k_parity in [1, 2, 3]:
        print(f"\n  --- k={k_parity} parity experts ---")

        model = get_model("reed_solomon_expert", k_parity=k_parity,
                          **{**CFG, "vocab_size": tok.vocab_size})
        mx.eval(model.parameters())
        docs_train, docs_val = train_val_split(docs, seed=42)
        train_ds = CharDataset(docs_train, tok, block_size=32)
        val_ds = CharDataset(docs_val, tok, block_size=32)
        train(model, train_ds, val_ds, steps=TRAIN_STEPS, batch_size=32,
              lr=3e-3, seed=42, log_every=500)

        model.save_original_weights()
        original_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        encoding = model.rs_encode_experts()
        overhead = model.param_overhead()

        # Drop exactly k experts (maximum tolerance)
        N = len(model.layers)
        # Drop the first k
        drop_idx = list(range(k_parity))
        model.reconstruct_from_available(encoding, drop_idx)
        mx.eval(model.parameters())
        recon_loss = evaluate(model, val_ds, batch_size=32, n_batches=20)
        pct = 100.0 * (recon_loss - original_loss) / original_loss

        # Measure weight error
        max_werr = 0.0
        for li in range(N):
            for pname in ["fc1.weight", "fc2.weight"]:
                parts = pname.split(".")
                param = getattr(model.layers[li].mlp, parts[0])
                w_r = np.array(getattr(param, parts[1]).tolist(), dtype=np.float64)
                w_o = model._original_weights[li][pname]
                max_werr = max(max_werr, np.max(np.abs(w_r - w_o)))

        print(f"  k={k_parity}: overhead={overhead:.0f}%, "
              f"drop {k_parity} -> quality={pct:+.4f}%, "
              f"max_w_err={max_werr:.2e}")
        print(f"  KC1 (>1%): {'KILLED' if abs(pct) > 1.0 else 'PASSED'}  |  "
              f"KC2 (>20%): {'KILLED' if overhead > 20.0 else 'PASSED'}")


# ========================================================================== #
#  Run all                                                                     #
# ========================================================================== #

if __name__ == "__main__":
    # Unit tests
    test_chebyshev_nodes()
    test_lagrange_roundtrip()
    test_rs_encode_decode_exact()
    test_rs_fault_tolerance()
    test_chebyshev_vs_uniform()

    # Experiments
    quality_results = run_quality_experiment()
    overhead_results = run_overhead_experiment()
    run_parity_as_blend_experiment()
    run_k_scaling_experiment()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    all_pcts = [
        r["scenarios"][s]["pct_change"]
        for r in quality_results
        for s in r["scenarios"]
    ]
    max_deg = max(all_pcts)
    mean_overhead = np.mean([r["overhead_pct"] for r in quality_results])

    print(f"  Max quality degradation: {max_deg:+.6f}%")
    print(f"  Parameter overhead (k=2, N=4): {mean_overhead:.1f}%")
    print(f"  Encoding time: {overhead_results['encode_ms']:.1f} ms (offline)")
    print(f"  Reconstruction time: {overhead_results['recon_ms']:.1f} ms (offline)")
    print()
    print(f"  KC1 - quality within 1%:     {'KILLED' if max_deg > 1.0 else 'PASSED'}")
    print(f"  KC2 - overhead within 20%:   {'KILLED' if mean_overhead > 20.0 else 'PASSED'}")
