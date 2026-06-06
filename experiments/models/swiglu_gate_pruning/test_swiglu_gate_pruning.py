"""Test and experiment runner for SwiGLU gate-product pruning.

Experiment: exp_swiglu_gate_pruning
Kill criteria:
  - <10% of capsules prunable at gate-product threshold
  - Pruning by gate product >3% worse than no pruning

This experiment:
1. Trains SwiGLU models (matching Qwen's MLP architecture)
2. Profiles gate products: |SiLU(W_gate @ x) * (W_up @ x)| per capsule
3. Compares gate product sparsity vs SiLU-only sparsity
4. Prunes by gate product threshold and measures quality impact
5. Compares against SiLU pruning baseline (killed: 0% prunable)
"""

import copy
import random
import json

import mlx.core as mx
import mlx.nn as nn


def count_params(model) -> int:
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def _std(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = sum(vals) / n
    return (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5


def run_experiment(seeds=(42, 123, 7), steps=300, verbose=True):
    """Full experiment: SwiGLU gate-product pruning vs SiLU-only pruning.

    Returns dict with all results for PAPER.md.
    """
    from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.silu_capsule.silu_capsule import SiLUCapsuleGPT
    from experiments.models.silu_pruning.silu_pruning import (
        profile_silu_activations,
        identify_prunable_capsules as silu_identify,
        prune_silu_model,
    )
    from experiments.models.swiglu_gate_pruning.swiglu_gate_pruning import (
        SwiGLUGatePruningGPT,
        profile_gate_products,
        identify_prunable_by_gate_product,
        prune_swiglu_model,
    )

    # Gate product thresholds to sweep
    gp_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        docs = load_names()
        tokenizer = CharTokenizer(docs)
        splits = domain_split(docs, method="binary")
        vocab_size = tokenizer.vocab_size

        # Prepare per-domain datasets
        domain_data = {}
        for dname, ddocs in splits.items():
            dtrain, dval = train_val_split(ddocs, seed=seed)
            domain_data[dname] = {
                "train": CharDataset(dtrain, tokenizer, block_size=32),
                "val": CharDataset(dval, tokenizer, block_size=32),
            }

        # Joint dataset
        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train_docs, tokenizer, block_size=32)
        joint_val = CharDataset(all_val_docs, tokenizer, block_size=32)

        model_kwargs = dict(vocab_size=vocab_size, block_size=32, n_embd=64,
                            n_head=4, n_layer=4, n_capsules=128)

        seed_results = {}

        # ---- 1. Train SiLU baseline (for comparison) ----
        print(f"\n--- Training SiLU baseline ---")
        mx.random.seed(seed)
        silu_model = SiLUCapsuleGPT(**model_kwargs)
        mx.eval(silu_model.parameters())

        train(silu_model, domain_data["a_m"]["train"],
              domain_data["a_m"]["val"],
              steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        silu_unpruned_loss = evaluate(silu_model, joint_val, batch_size=32)
        print(f"  SiLU unpruned val loss: {silu_unpruned_loss:.4f}")

        # Profile SiLU activations
        silu_profiles = profile_silu_activations(silu_model, joint_val,
                                                  n_batches=20, batch_size=32, seed=seed)

        silu_dist = []
        for l_idx, prof in enumerate(silu_profiles):
            vals = prof["mean_abs"].tolist()
            silu_dist.append({
                "layer": l_idx,
                "mean_abs_min": min(vals),
                "mean_abs_max": max(vals),
                "mean_abs_median": sorted(vals)[len(vals)//2],
                "n_below_0.01": sum(1 for v in vals if v <= 0.01),
                "n_below_0.05": sum(1 for v in vals if v <= 0.05),
                "n_capsules": len(vals),
            })
            if verbose:
                print(f"    SiLU Layer {l_idx}: mean_abs range [{min(vals):.6f}, {max(vals):.6f}], "
                      f"below 0.01: {silu_dist[-1]['n_below_0.01']}/{len(vals)}")

        seed_results["silu_baseline"] = {
            "unpruned_loss": silu_unpruned_loss,
            "activation_dist": silu_dist,
            "n_params": count_params(silu_model),
        }

        # ---- 2. Train SwiGLU model ----
        print(f"\n--- Training SwiGLU model ---")
        mx.random.seed(seed)
        swiglu_model = SwiGLUGatePruningGPT(**model_kwargs)
        mx.eval(swiglu_model.parameters())

        train(swiglu_model, domain_data["a_m"]["train"],
              domain_data["a_m"]["val"],
              steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        swiglu_unpruned_loss = evaluate(swiglu_model, joint_val, batch_size=32)
        print(f"  SwiGLU unpruned val loss: {swiglu_unpruned_loss:.4f}")

        # ---- 3. Profile gate products ----
        print(f"\n--- Profiling SwiGLU gate products ---")
        gp_profiles = profile_gate_products(swiglu_model, joint_val,
                                             n_batches=20, batch_size=32, seed=seed)

        gp_dist = []
        for l_idx, prof in enumerate(gp_profiles):
            gp_vals = prof["gate_product_mean_abs"].tolist()
            gate_vals = prof["gate_only_mean_abs"].tolist()
            up_vals = prof["up_only_mean_abs"].tolist()

            gp_dist.append({
                "layer": l_idx,
                "gp_mean_abs_min": min(gp_vals),
                "gp_mean_abs_max": max(gp_vals),
                "gp_mean_abs_median": sorted(gp_vals)[len(gp_vals)//2],
                "gate_mean_abs_min": min(gate_vals),
                "gate_mean_abs_max": max(gate_vals),
                "up_mean_abs_min": min(up_vals),
                "up_mean_abs_max": max(up_vals),
                "n_below_0.001": sum(1 for v in gp_vals if v <= 0.001),
                "n_below_0.005": sum(1 for v in gp_vals if v <= 0.005),
                "n_below_0.01": sum(1 for v in gp_vals if v <= 0.01),
                "n_below_0.02": sum(1 for v in gp_vals if v <= 0.02),
                "n_below_0.05": sum(1 for v in gp_vals if v <= 0.05),
                "n_capsules": len(gp_vals),
            })

            if verbose:
                print(f"    Layer {l_idx}:")
                print(f"      Gate product: [{min(gp_vals):.6f}, {max(gp_vals):.6f}], "
                      f"median={sorted(gp_vals)[len(gp_vals)//2]:.6f}")
                print(f"      SiLU(gate):   [{min(gate_vals):.6f}, {max(gate_vals):.6f}]")
                print(f"      Up:           [{min(up_vals):.6f}, {max(up_vals):.6f}]")
                for tau in [0.001, 0.005, 0.01, 0.02, 0.05]:
                    n_below = sum(1 for v in gp_vals if v <= tau)
                    print(f"      tau={tau:.3f}: {n_below}/{len(gp_vals)} below "
                          f"({n_below/len(gp_vals)*100:.1f}%)")

        # ---- 4. Gate product pruning threshold sweep ----
        print(f"\n--- SwiGLU gate-product pruning sweep ---")
        gp_pruning_results = {}

        for tau in gp_thresholds:
            swiglu_pruned = copy.deepcopy(swiglu_model)
            masks = identify_prunable_by_gate_product(gp_profiles, threshold=tau,
                                                       method="gate_product_mean_abs")
            prune_stats = prune_swiglu_model(swiglu_pruned, masks, verbose=False)
            pruned_loss = evaluate(swiglu_pruned, joint_val, batch_size=32)
            pct_pruned = prune_stats["pct_pruned"]
            delta_pct = (pruned_loss - swiglu_unpruned_loss) / swiglu_unpruned_loss * 100

            print(f"  tau={tau:.4f}: {pct_pruned:.1f}% pruned, "
                  f"loss={pruned_loss:.4f} (delta: {delta_pct:+.2f}%)")

            gp_pruning_results[str(tau)] = {
                "threshold": tau,
                "pct_pruned": pct_pruned,
                "pruned_loss": pruned_loss,
                "delta_pct": delta_pct,
                "per_layer": prune_stats["per_layer"],
            }

        # Also try max_abs method
        print(f"\n--- SwiGLU gate-product pruning (max_abs method) ---")
        gp_maxabs_results = {}
        for tau in [0.01, 0.05, 0.1]:
            swiglu_pruned = copy.deepcopy(swiglu_model)
            masks = identify_prunable_by_gate_product(gp_profiles, threshold=tau,
                                                       method="product_max_abs")
            prune_stats = prune_swiglu_model(swiglu_pruned, masks, verbose=False)
            pruned_loss = evaluate(swiglu_pruned, joint_val, batch_size=32)
            pct_pruned = prune_stats["pct_pruned"]
            delta_pct = (pruned_loss - swiglu_unpruned_loss) / swiglu_unpruned_loss * 100

            print(f"  max_abs tau={tau:.4f}: {pct_pruned:.1f}% pruned, "
                  f"loss={pruned_loss:.4f} (delta: {delta_pct:+.2f}%)")

            gp_maxabs_results[str(tau)] = {
                "threshold": tau,
                "pct_pruned": pct_pruned,
                "pruned_loss": pruned_loss,
                "delta_pct": delta_pct,
            }

        seed_results["swiglu"] = {
            "unpruned_loss": swiglu_unpruned_loss,
            "gate_product_dist": gp_dist,
            "pruning_sweep": gp_pruning_results,
            "maxabs_sweep": gp_maxabs_results,
            "n_params": count_params(swiglu_model),
        }

        all_results[str(seed)] = seed_results

    # ---- Aggregate across seeds ----
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*60}")

    # SiLU baseline aggregate
    silu_losses = [all_results[str(s)]["silu_baseline"]["unpruned_loss"] for s in seeds]
    print(f"\nSiLU baseline:")
    print(f"  Unpruned: {sum(silu_losses)/len(silu_losses):.4f} (std={_std(silu_losses):.4f})")
    print(f"  Prunable at tau=0.01 (SiLU): 0% (floor ~0.046, from Exp 15)")

    # SwiGLU aggregate
    swiglu_losses = [all_results[str(s)]["swiglu"]["unpruned_loss"] for s in seeds]
    print(f"\nSwiGLU baseline:")
    print(f"  Unpruned: {sum(swiglu_losses)/len(swiglu_losses):.4f} (std={_std(swiglu_losses):.4f})")

    # Gate product distribution summary
    print(f"\nGate product distribution (aggregate):")
    for tau in gp_thresholds:
        tau_str = str(tau)
        pcts = [all_results[str(s)]["swiglu"]["pruning_sweep"][tau_str]["pct_pruned"] for s in seeds]
        deltas = [all_results[str(s)]["swiglu"]["pruning_sweep"][tau_str]["delta_pct"] for s in seeds]
        mean_pct = sum(pcts) / len(pcts)
        mean_delta = sum(deltas) / len(deltas)
        std_pct = _std(pcts)
        std_delta = _std(deltas)
        print(f"  tau={tau:.4f}: {mean_pct:.1f}% pruned (std={std_pct:.1f}%), "
              f"delta={mean_delta:+.2f}% (std={std_delta:.2f}%)")

    # Kill criteria check
    print(f"\n--- Kill Criteria Check ---")
    print(f"Kill 1: <10% prunable at gate-product threshold")
    print(f"Kill 2: pruning by gate product >3% worse than no pruning")

    best_safe_tau = None
    best_safe_pct = 0

    for tau in gp_thresholds:
        tau_str = str(tau)
        pcts = [all_results[str(s)]["swiglu"]["pruning_sweep"][tau_str]["pct_pruned"] for s in seeds]
        deltas = [all_results[str(s)]["swiglu"]["pruning_sweep"][tau_str]["delta_pct"] for s in seeds]
        mean_pct = sum(pcts) / len(pcts)
        mean_delta = sum(deltas) / len(deltas)

        quality_ok = mean_delta <= 3.0
        pruning_ok = mean_pct >= 10.0

        status = []
        if quality_ok:
            status.append("quality OK")
        else:
            status.append("QUALITY KILL")
        if pruning_ok:
            status.append("pruning OK")
        else:
            status.append("PRUNING KILL")

        print(f"  tau={tau:.4f}: {mean_pct:.1f}% pruned, delta={mean_delta:+.2f}% -> {', '.join(status)}")

        if quality_ok and mean_pct > best_safe_pct:
            best_safe_tau = tau
            best_safe_pct = mean_pct

    # Final verdict
    print(f"\n--- VERDICT ---")
    if best_safe_tau is not None and best_safe_pct >= 10.0:
        print(f"  PASS: {best_safe_pct:.1f}% prunable at tau={best_safe_tau} with acceptable quality")
        print(f"  SwiGLU gate-product pruning provides meaningful compression")
    elif best_safe_tau is not None:
        print(f"  PARTIAL: Best safe tau={best_safe_tau} prunes {best_safe_pct:.1f}% (<10% threshold)")
        print(f"  SwiGLU gate-product is better than SiLU-only but below 10% target")
    else:
        print(f"  KILL: No threshold achieves >10% pruning within 3% quality loss")

    # Comparison with SiLU-only pruning
    print(f"\n--- SwiGLU vs SiLU Pruning Comparison ---")
    print(f"  SiLU-only (Exp 15): 0% prunable at tau<=0.01 (floor ~0.046)")
    if best_safe_tau is not None:
        print(f"  SwiGLU gate-product: {best_safe_pct:.1f}% prunable at tau={best_safe_tau}")
        print(f"  Improvement: SwiGLU gate-product unlocks pruning that SiLU-only cannot")
    else:
        print(f"  SwiGLU gate-product: also limited pruning")

    return all_results


def test_swiglu_gate_pruning_basic():
    """Basic test: SwiGLU model trains and gate-product pruning runs."""
    from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.swiglu_gate_pruning.swiglu_gate_pruning import (
        SwiGLUGatePruningGPT,
        profile_gate_products,
        identify_prunable_by_gate_product,
        prune_swiglu_model,
    )

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    model = SwiGLUGatePruningGPT(vocab_size=tokenizer.vocab_size, block_size=32,
                                  n_embd=64, n_head=4, n_layer=4, n_capsules=64)
    mx.eval(model.parameters())

    # Verify SwiGLU has 3 weight matrices per capsule pool
    pool = model.layers[0].capsule_pool
    assert hasattr(pool, 'W_gate'), "Missing W_gate"
    assert hasattr(pool, 'W_up'), "Missing W_up"
    assert hasattr(pool, 'B'), "Missing B (down projection)"
    print(f"SwiGLU pool: W_gate {pool.W_gate.weight.shape}, "
          f"W_up {pool.W_up.weight.shape}, B {pool.B.weight.shape}")

    # Quick training
    train(model, train_ds, val_ds, steps=50, batch_size=16, lr=3e-3, seed=42, log_every=25)

    loss_before = evaluate(model, val_ds, batch_size=16)
    print(f"Loss before pruning: {loss_before:.4f}")

    # Profile gate products
    profiles = profile_gate_products(model, val_ds, n_batches=5, batch_size=16, seed=42)
    for l_idx, prof in enumerate(profiles):
        gp_vals = prof["gate_product_mean_abs"].tolist()
        gate_vals = prof["gate_only_mean_abs"].tolist()
        up_vals = prof["up_only_mean_abs"].tolist()
        print(f"Layer {l_idx}:")
        print(f"  Gate product: [{min(gp_vals):.6f}, {max(gp_vals):.6f}]")
        print(f"  SiLU(gate):   [{min(gate_vals):.6f}, {max(gate_vals):.6f}]")
        print(f"  Up:           [{min(up_vals):.6f}, {max(up_vals):.6f}]")
        assert len(gp_vals) == 64, f"Expected 64 capsules, got {len(gp_vals)}"

        # Key test: gate product min should be <= SiLU gate min
        # (the multiplicative interaction can push values closer to zero)
        gp_min = min(gp_vals)
        gate_min = min(gate_vals)
        print(f"  GP min ({gp_min:.6f}) vs Gate min ({gate_min:.6f}): "
              f"{'GP lower' if gp_min < gate_min else 'Gate lower'}")

    # Prune at tau=0.01
    masks = identify_prunable_by_gate_product(profiles, threshold=0.01)
    model_pruned = copy.deepcopy(model)
    stats = prune_swiglu_model(model_pruned, masks, verbose=True)

    loss_after = evaluate(model_pruned, val_ds, batch_size=16)
    delta_pct = (loss_after - loss_before) / loss_before * 100
    print(f"Loss after pruning: {loss_after:.4f} (delta: {delta_pct:+.2f}%)")
    print(f"Pruned: {stats['pct_pruned']:.1f}%")

    assert loss_after < 10.0, "Pruned model produces garbage output"
    print("\nBasic test PASSED")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test_swiglu_gate_pruning_basic()
    else:
        results = run_experiment()
        with open("/Users/tom/Code/tomsiwik/llm/experiments/models/swiglu_gate_pruning/results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nResults saved to experiments/models/swiglu_gate_pruning/results.json")
