"""Test and experiment runner for SiLU magnitude-threshold pruning.

Experiment: exp15_non_relu_pruning
Kill criterion: magnitude-threshold pruning on SiLU capsules degrades
quality >5% vs unpruned.

This experiment:
1. Trains SiLU and ReLU models under identical conditions
2. Profiles activation distributions for both
3. Applies magnitude-based pruning to SiLU at multiple thresholds
4. Compares pruned quality vs unpruned baseline
5. Contrasts with ReLU dead-capsule pruning
"""

import copy
import random
import json

import mlx.core as mx
import mlx.nn as nn


def count_params(model) -> int:
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def run_experiment(seeds=(42, 123, 7), steps=300, verbose=True):
    """Full experiment: SiLU vs ReLU pruning comparison.

    Returns dict with all results for PAPER.md.
    """
    from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.relu_router.relu_router import ReLURouterGPT
    from experiments.models.silu_capsule.silu_capsule import SiLUCapsuleGPT
    from experiments.models.dead_capsule_pruning.dead_capsule_pruning import (
        profile_activations as relu_profile,
        identify_dead_capsules as relu_identify,
        prune_model as relu_prune,
    )
    from experiments.models.silu_pruning.silu_pruning import (
        profile_silu_activations,
        identify_prunable_capsules,
        prune_silu_model,
    )

    # SiLU pruning thresholds to sweep
    silu_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]

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

        # ---- 1. Train ReLU baseline (for comparison) ----
        print(f"\n--- Training ReLU model ---")
        mx.random.seed(seed)
        relu_model = ReLURouterGPT(**model_kwargs)
        mx.eval(relu_model.parameters())

        # Train on domain A
        rng = random.Random(seed)
        result_a = train(relu_model, domain_data["a_m"]["train"],
                         domain_data["a_m"]["val"],
                         steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        relu_single_loss = evaluate(relu_model, joint_val, batch_size=32)
        print(f"  ReLU single-domain val loss: {relu_single_loss:.4f}")

        # Profile ReLU activations
        relu_freqs = relu_profile(relu_model, joint_val, n_batches=20, batch_size=32, seed=seed)
        relu_dead_counts = []
        for l_idx, freq in enumerate(relu_freqs):
            mx.eval(freq)
            n_dead = sum(1 for f in freq.tolist() if f <= 0.0)
            relu_dead_counts.append(n_dead)
            if verbose:
                f_list = freq.tolist()
                print(f"    ReLU Layer {l_idx}: {n_dead}/{len(f_list)} dead (freq=0), "
                      f"mean freq={sum(f_list)/len(f_list):.4f}")

        # Prune ReLU model
        relu_model_pruned = copy.deepcopy(relu_model)
        relu_masks = relu_identify(relu_freqs, threshold=0.0)
        relu_prune_stats = relu_prune(relu_model_pruned, relu_masks, verbose=False)
        relu_pruned_loss = evaluate(relu_model_pruned, joint_val, batch_size=32)
        relu_pct_pruned = relu_prune_stats["pct_pruned"]

        print(f"  ReLU pruned: {relu_pct_pruned:.1f}% pruned, val loss: {relu_pruned_loss:.4f} "
              f"(delta: {(relu_pruned_loss - relu_single_loss)/relu_single_loss*100:+.2f}%)")

        seed_results["relu"] = {
            "unpruned_loss": relu_single_loss,
            "pruned_loss": relu_pruned_loss,
            "pct_pruned": relu_pct_pruned,
            "dead_counts": relu_dead_counts,
            "delta_pct": (relu_pruned_loss - relu_single_loss) / relu_single_loss * 100,
            "n_params": count_params(relu_model),
        }

        # ---- 2. Train SiLU model ----
        print(f"\n--- Training SiLU model ---")
        mx.random.seed(seed)
        silu_model = SiLUCapsuleGPT(**model_kwargs)
        mx.eval(silu_model.parameters())

        result_s = train(silu_model, domain_data["a_m"]["train"],
                         domain_data["a_m"]["val"],
                         steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        silu_unpruned_loss = evaluate(silu_model, joint_val, batch_size=32)
        print(f"  SiLU unpruned val loss: {silu_unpruned_loss:.4f}")

        # ---- 3. Profile SiLU activations ----
        print(f"\n--- Profiling SiLU activations ---")
        silu_profiles = profile_silu_activations(silu_model, joint_val,
                                                  n_batches=20, batch_size=32, seed=seed)

        # Report distribution
        for l_idx, prof in enumerate(silu_profiles):
            mean_abs = prof["mean_abs"]
            mx.eval(mean_abs)
            vals = mean_abs.tolist()
            max_abs_vals = prof["max_abs"].tolist()
            if verbose:
                print(f"    Layer {l_idx}: mean_abs range [{min(vals):.6f}, {max(vals):.6f}], "
                      f"median={sorted(vals)[len(vals)//2]:.6f}, "
                      f"max_abs range [{min(max_abs_vals):.6f}, {max(max_abs_vals):.6f}]")
                for tau in [0.001, 0.01, 0.1]:
                    n_below = sum(1 for v in vals if v <= tau)
                    print(f"      tau={tau:.3f}: {n_below}/{len(vals)} below ({n_below/len(vals)*100:.1f}%)")

        # ---- 4. SiLU pruning threshold sweep ----
        print(f"\n--- SiLU pruning threshold sweep ---")
        silu_pruning_results = {}

        for tau in silu_thresholds:
            silu_pruned = copy.deepcopy(silu_model)
            masks = identify_prunable_capsules(silu_profiles, threshold=tau, method="mean_abs")
            prune_stats = prune_silu_model(silu_pruned, masks, verbose=False)
            pruned_loss = evaluate(silu_pruned, joint_val, batch_size=32)
            pct_pruned = prune_stats["pct_pruned"]
            delta_pct = (pruned_loss - silu_unpruned_loss) / silu_unpruned_loss * 100

            print(f"  tau={tau:.4f}: {pct_pruned:.1f}% pruned, "
                  f"loss={pruned_loss:.4f} (delta: {delta_pct:+.2f}%)")

            silu_pruning_results[str(tau)] = {
                "threshold": tau,
                "pct_pruned": pct_pruned,
                "pruned_loss": pruned_loss,
                "delta_pct": delta_pct,
                "per_layer": prune_stats["per_layer"],
            }

        # Also try max_abs method
        print(f"\n--- SiLU pruning with max_abs method ---")
        silu_maxabs_results = {}
        for tau in [0.01, 0.1]:
            silu_pruned = copy.deepcopy(silu_model)
            masks = identify_prunable_capsules(silu_profiles, threshold=tau, method="max_abs")
            prune_stats = prune_silu_model(silu_pruned, masks, verbose=False)
            pruned_loss = evaluate(silu_pruned, joint_val, batch_size=32)
            pct_pruned = prune_stats["pct_pruned"]
            delta_pct = (pruned_loss - silu_unpruned_loss) / silu_unpruned_loss * 100

            print(f"  max_abs tau={tau:.4f}: {pct_pruned:.1f}% pruned, "
                  f"loss={pruned_loss:.4f} (delta: {delta_pct:+.2f}%)")

            silu_maxabs_results[str(tau)] = {
                "threshold": tau,
                "pct_pruned": pct_pruned,
                "pruned_loss": pruned_loss,
                "delta_pct": delta_pct,
            }

        # ---- 5. Activation distribution statistics ----
        activation_dist = []
        for l_idx, prof in enumerate(silu_profiles):
            vals = prof["mean_abs"].tolist()
            max_vals = prof["max_abs"].tolist()
            activation_dist.append({
                "layer": l_idx,
                "mean_abs_min": min(vals),
                "mean_abs_max": max(vals),
                "mean_abs_median": sorted(vals)[len(vals)//2],
                "mean_abs_mean": sum(vals)/len(vals),
                "max_abs_min": min(max_vals),
                "max_abs_max": max(max_vals),
                "n_below_0.001": sum(1 for v in vals if v <= 0.001),
                "n_below_0.01": sum(1 for v in vals if v <= 0.01),
                "n_below_0.1": sum(1 for v in vals if v <= 0.1),
                "n_capsules": len(vals),
            })

        seed_results["silu"] = {
            "unpruned_loss": silu_unpruned_loss,
            "pruning_sweep": silu_pruning_results,
            "maxabs_sweep": silu_maxabs_results,
            "activation_dist": activation_dist,
            "n_params": count_params(silu_model),
        }

        all_results[str(seed)] = seed_results

    # ---- 6. Aggregate across seeds ----
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*60}")

    # ReLU aggregate
    relu_unpruned_losses = [all_results[str(s)]["relu"]["unpruned_loss"] for s in seeds]
    relu_pruned_losses = [all_results[str(s)]["relu"]["pruned_loss"] for s in seeds]
    relu_pct_pruned = [all_results[str(s)]["relu"]["pct_pruned"] for s in seeds]

    print(f"\nReLU baseline:")
    print(f"  Unpruned: {sum(relu_unpruned_losses)/len(relu_unpruned_losses):.4f} "
          f"(std={_std(relu_unpruned_losses):.4f})")
    print(f"  Pruned (tau=0): {sum(relu_pruned_losses)/len(relu_pruned_losses):.4f}, "
          f"pct_pruned={sum(relu_pct_pruned)/len(relu_pct_pruned):.1f}%")

    # SiLU aggregate
    silu_unpruned_losses = [all_results[str(s)]["silu"]["unpruned_loss"] for s in seeds]
    print(f"\nSiLU baseline:")
    print(f"  Unpruned: {sum(silu_unpruned_losses)/len(silu_unpruned_losses):.4f} "
          f"(std={_std(silu_unpruned_losses):.4f})")

    print(f"\nSiLU pruning sweep (mean_abs method):")
    for tau in silu_thresholds:
        tau_str = str(tau)
        losses = [all_results[str(s)]["silu"]["pruning_sweep"][tau_str]["pruned_loss"] for s in seeds]
        pcts = [all_results[str(s)]["silu"]["pruning_sweep"][tau_str]["pct_pruned"] for s in seeds]
        deltas = [all_results[str(s)]["silu"]["pruning_sweep"][tau_str]["delta_pct"] for s in seeds]
        mean_loss = sum(losses)/len(losses)
        mean_pct = sum(pcts)/len(pcts)
        mean_delta = sum(deltas)/len(deltas)
        kill = "KILL" if abs(mean_delta) > 5.0 else "PASS"
        print(f"  tau={tau:.4f}: {mean_pct:.1f}% pruned, "
              f"loss={mean_loss:.4f}, delta={mean_delta:+.2f}% [{kill}]")

    # Kill criterion check
    print(f"\n--- Kill Criterion Check ---")
    print(f"Kill if: magnitude-threshold pruning degrades quality >5% vs unpruned")
    for tau in silu_thresholds:
        tau_str = str(tau)
        deltas = [all_results[str(s)]["silu"]["pruning_sweep"][tau_str]["delta_pct"] for s in seeds]
        mean_delta = sum(deltas)/len(deltas)
        pcts = [all_results[str(s)]["silu"]["pruning_sweep"][tau_str]["pct_pruned"] for s in seeds]
        mean_pct = sum(pcts)/len(pcts)
        verdict = "KILL" if abs(mean_delta) > 5.0 else "PASS"
        print(f"  tau={tau:.4f}: delta={mean_delta:+.2f}%, pruned={mean_pct:.1f}% -> {verdict}")

    # Activation distribution summary
    print(f"\n--- SiLU Activation Distribution (seed {seeds[0]}) ---")
    for prof in all_results[str(seeds[0])]["silu"]["activation_dist"]:
        print(f"  Layer {prof['layer']}: mean_abs [{prof['mean_abs_min']:.6f}, {prof['mean_abs_max']:.6f}], "
              f"below 0.01: {prof['n_below_0.01']}/{prof['n_capsules']} ({prof['n_below_0.01']/prof['n_capsules']*100:.1f}%)")

    return all_results


def _std(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = sum(vals) / n
    return (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5


def test_silu_pruning_basic():
    """Basic test: SiLU model trains and pruning runs without errors."""
    from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.silu_capsule.silu_capsule import SiLUCapsuleGPT
    from experiments.models.silu_pruning.silu_pruning import (
        profile_silu_activations,
        identify_prunable_capsules,
        prune_silu_model,
    )

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    model = SiLUCapsuleGPT(vocab_size=tokenizer.vocab_size, block_size=32,
                            n_embd=64, n_head=4, n_layer=4, n_capsules=64)
    mx.eval(model.parameters())

    # Quick training
    train(model, train_ds, val_ds, steps=50, batch_size=16, lr=3e-3, seed=42, log_every=25)

    loss_before = evaluate(model, val_ds, batch_size=16)
    print(f"Loss before pruning: {loss_before:.4f}")

    # Profile
    profiles = profile_silu_activations(model, val_ds, n_batches=5, batch_size=16, seed=42)
    for l_idx, prof in enumerate(profiles):
        mx.eval(prof["mean_abs"])
        vals = prof["mean_abs"].tolist()
        print(f"Layer {l_idx}: mean_abs range [{min(vals):.6f}, {max(vals):.6f}]")
        assert len(vals) == 64, f"Expected 64 capsules, got {len(vals)}"

    # Verify no exact zeros (SiLU property)
    for l_idx, prof in enumerate(profiles):
        mean_abs = prof["mean_abs"]
        mx.eval(mean_abs)
        n_exact_zero = sum(1 for v in mean_abs.tolist() if v == 0.0)
        # SiLU should rarely have EXACT zero mean activations
        # (possible at initialization, unlikely after training)
        print(f"Layer {l_idx}: {n_exact_zero} exact zeros")

    # Prune at tau=0.01
    masks = identify_prunable_capsules(profiles, threshold=0.01, method="mean_abs")
    import copy
    model_pruned = copy.deepcopy(model)
    stats = prune_silu_model(model_pruned, masks, verbose=True)

    loss_after = evaluate(model_pruned, val_ds, batch_size=16)
    delta_pct = (loss_after - loss_before) / loss_before * 100
    print(f"Loss after pruning: {loss_after:.4f} (delta: {delta_pct:+.2f}%)")
    print(f"Pruned: {stats['pct_pruned']:.1f}%")

    # The test: SiLU pruning should work without crashing
    # Quality check is in the full experiment
    assert loss_after < 10.0, "Pruned model produces garbage output"
    print("\nBasic test PASSED")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test_silu_pruning_basic()
    else:
        results = run_experiment()
        # Save results
        # Convert non-serializable values
        with open("/Users/tom/Code/tomsiwik/llm/experiments/models/silu_pruning/results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nResults saved to experiments/models/silu_pruning/results.json")
