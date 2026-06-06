"""Test and experiment runner for combined dead capsule + gate-product pruning.

Experiment: exp_swiglu_combined_dead_capsule
Kill criteria:
  1. Combined pruning does not exceed either method alone by >5pp
  2. Quality degrades >3% vs no pruning

This experiment:
1. Trains SwiGLU models (with aux sparsity loss to match parent experiment)
2. Profiles BOTH dead capsules (fire frequency = 0) AND gate products (mean mag)
3. Measures set overlap: what fraction of gate-prunable params are also dead?
4. If complementary (low overlap), combines both pruning criteria
5. Tests combined pruning rate vs quality degradation
6. Target: >70% combined pruning with <3% quality loss
"""

import copy
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
    """Full experiment: combined dead capsule + gate-product pruning.

    For each seed:
    1. Train SwiGLU model (with aux sparsity loss)
    2. Profile dead capsules (fire frequency)
    3. Profile gate products (mean magnitude)
    4. Measure set overlap at various gate-product thresholds
    5. Apply combined pruning and measure quality
    6. Compare against dead-only and gate-only pruning
    """
    from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.swiglu_combined_dead_capsule.swiglu_combined_dead_capsule import (
        SwiGLUCombinedDeadCapsuleGPT,
        profile_dead_capsules_swiglu,
        identify_dead_capsules_swiglu,
        profile_gate_products,
        identify_prunable_by_gate_product,
        prune_swiglu_model,
        combined_pruning_masks,
        compute_set_overlap,
    )

    # Gate product thresholds to sweep
    gp_thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
    # Dead capsule frequency thresholds
    dead_thresholds = [0.0, 0.001, 0.005]

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        docs = load_names()
        tokenizer = CharTokenizer(docs)
        splits = domain_split(docs, method="binary")
        vocab_size = tokenizer.vocab_size

        # Per-domain datasets
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

        # ---- 1. Train SwiGLU model ----
        print(f"\n--- Training SwiGLU model (with aux sparsity loss) ---")
        mx.random.seed(seed)
        model = SwiGLUCombinedDeadCapsuleGPT(**model_kwargs)
        mx.eval(model.parameters())

        train(model, domain_data["a_m"]["train"],
              domain_data["a_m"]["val"],
              steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        unpruned_loss = evaluate(model, joint_val, batch_size=32)
        print(f"  Unpruned val loss: {unpruned_loss:.4f}")

        seed_results["unpruned_loss"] = unpruned_loss
        seed_results["n_params"] = count_params(model)

        # ---- 2. Profile dead capsules ----
        print(f"\n--- Profiling dead capsules (SwiGLU) ---")
        dead_profiles = profile_dead_capsules_swiglu(
            model, joint_val, n_batches=20, batch_size=32, seed=seed)

        dead_summary = []
        for l_idx, prof in enumerate(dead_profiles):
            freq_list = prof["fire_frequency"].tolist()
            mean_list = prof["mean_abs"].tolist()
            info = {
                "layer": l_idx,
                "n_capsules": prof["n_capsules"],
                "n_dead_exact": prof["n_dead"],
                "pct_dead_exact": prof["pct_dead"],
                "n_dead_001": sum(1 for f in freq_list if f <= 0.001),
                "n_dead_005": sum(1 for f in freq_list if f <= 0.005),
                "freq_min": min(freq_list),
                "freq_max": max(freq_list),
                "freq_median": sorted(freq_list)[len(freq_list)//2],
                "mean_abs_min": min(mean_list),
                "mean_abs_max": max(mean_list),
            }
            dead_summary.append(info)
            if verbose:
                print(f"  Layer {l_idx}: {info['n_dead_exact']}/{info['n_capsules']} "
                      f"dead ({info['pct_dead_exact']:.1f}%), "
                      f"freq range [{info['freq_min']:.6f}, {info['freq_max']:.6f}]")

        seed_results["dead_profiles"] = dead_summary

        # ---- 3. Profile gate products ----
        print(f"\n--- Profiling gate products ---")
        gp_profiles = profile_gate_products(
            model, joint_val, n_batches=20, batch_size=32, seed=seed)

        gp_summary = []
        for l_idx, prof in enumerate(gp_profiles):
            gp_vals = prof["gate_product_mean_abs"].tolist()
            info = {
                "layer": l_idx,
                "n_capsules": prof["n_capsules"],
                "gp_min": min(gp_vals),
                "gp_max": max(gp_vals),
                "gp_median": sorted(gp_vals)[len(gp_vals)//2],
            }
            for tau in gp_thresholds:
                info[f"n_below_{tau}"] = sum(1 for v in gp_vals if v <= tau)
            gp_summary.append(info)
            if verbose:
                print(f"  Layer {l_idx}: gate product range [{info['gp_min']:.6f}, "
                      f"{info['gp_max']:.6f}], median={info['gp_median']:.6f}")
                for tau in gp_thresholds:
                    n = info[f"n_below_{tau}"]
                    print(f"    tau={tau}: {n}/{prof['n_capsules']} "
                          f"({n/prof['n_capsules']*100:.1f}%) below")

        seed_results["gp_profiles"] = gp_summary

        # ---- 4. Overlap analysis ----
        print(f"\n--- Set Overlap Analysis ---")
        n_caps = [prof["n_capsules"] for prof in dead_profiles]

        overlap_results = {}
        for dead_tau in dead_thresholds:
            dead_masks = identify_dead_capsules_swiglu(dead_profiles, threshold=dead_tau)
            for gp_tau in gp_thresholds:
                gate_masks = identify_prunable_by_gate_product(
                    gp_profiles, threshold=gp_tau, method="gate_product_mean_abs")

                overlap = compute_set_overlap(dead_masks, gate_masks, n_caps)
                key = f"dead_{dead_tau}_gate_{gp_tau}"
                overlap_results[key] = overlap

                if verbose:
                    print(f"  dead_tau={dead_tau}, gate_tau={gp_tau}:")
                    print(f"    Dead: {overlap['total_dead']}/{overlap['total_capsules']} "
                          f"({overlap['total_dead']/overlap['total_capsules']*100:.1f}%)")
                    print(f"    Gate: {overlap['total_gate_prunable']}/{overlap['total_capsules']} "
                          f"({overlap['total_gate_prunable']/overlap['total_capsules']*100:.1f}%)")
                    print(f"    Both: {overlap['total_both']}, Either: {overlap['total_either']} "
                          f"({overlap['pct_combined']:.1f}%)")
                    print(f"    Jaccard: {overlap['jaccard']:.3f}, "
                          f"Gate-also-dead: {overlap['pct_gate_also_dead']:.1f}%")

        seed_results["overlap"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"}
                                    for k, v in overlap_results.items()}

        # ---- 5. Pruning comparison: dead-only vs gate-only vs combined ----
        print(f"\n--- Pruning Comparison ---")
        pruning_comparison = {}

        for dead_tau in [0.0]:  # Primary: exact dead
            for gp_tau in gp_thresholds:
                # Dead-only pruning
                dead_masks = identify_dead_capsules_swiglu(dead_profiles, threshold=dead_tau)
                model_dead = copy.deepcopy(model)
                dead_stats = prune_swiglu_model(model_dead, dead_masks, verbose=False)
                dead_loss = evaluate(model_dead, joint_val, batch_size=32)
                dead_delta = (dead_loss - unpruned_loss) / unpruned_loss * 100

                # Gate-only pruning
                gate_masks = identify_prunable_by_gate_product(
                    gp_profiles, threshold=gp_tau, method="gate_product_mean_abs")
                model_gate = copy.deepcopy(model)
                gate_stats = prune_swiglu_model(model_gate, gate_masks, verbose=False)
                gate_loss = evaluate(model_gate, joint_val, batch_size=32)
                gate_delta = (gate_loss - unpruned_loss) / unpruned_loss * 100

                # Combined pruning (union of pruning sets)
                dead_masks_fresh = identify_dead_capsules_swiglu(dead_profiles, threshold=dead_tau)
                gate_masks_fresh = identify_prunable_by_gate_product(
                    gp_profiles, threshold=gp_tau, method="gate_product_mean_abs")
                combo_masks = combined_pruning_masks(dead_masks_fresh, gate_masks_fresh)
                model_combo = copy.deepcopy(model)
                combo_stats = prune_swiglu_model(model_combo, combo_masks, verbose=False)
                combo_loss = evaluate(model_combo, joint_val, batch_size=32)
                combo_delta = (combo_loss - unpruned_loss) / unpruned_loss * 100

                key = f"dead_{dead_tau}_gate_{gp_tau}"
                pruning_comparison[key] = {
                    "dead_tau": dead_tau,
                    "gate_tau": gp_tau,
                    "dead_only": {
                        "pct_pruned": dead_stats["pct_pruned"],
                        "loss": dead_loss,
                        "delta_pct": dead_delta,
                    },
                    "gate_only": {
                        "pct_pruned": gate_stats["pct_pruned"],
                        "loss": gate_loss,
                        "delta_pct": gate_delta,
                    },
                    "combined": {
                        "pct_pruned": combo_stats["pct_pruned"],
                        "loss": combo_loss,
                        "delta_pct": combo_delta,
                    },
                    "combined_advantage_pp": (combo_stats["pct_pruned"]
                                               - max(dead_stats["pct_pruned"],
                                                     gate_stats["pct_pruned"])),
                }

                print(f"  dead_tau={dead_tau}, gate_tau={gp_tau}:")
                print(f"    Dead-only:  {dead_stats['pct_pruned']:5.1f}% pruned, "
                      f"delta={dead_delta:+.2f}%")
                print(f"    Gate-only:  {gate_stats['pct_pruned']:5.1f}% pruned, "
                      f"delta={gate_delta:+.2f}%")
                print(f"    Combined:   {combo_stats['pct_pruned']:5.1f}% pruned, "
                      f"delta={combo_delta:+.2f}%")
                print(f"    Advantage:  {pruning_comparison[key]['combined_advantage_pp']:+.1f}pp "
                      f"over best single method")

        seed_results["pruning_comparison"] = pruning_comparison

        # ---- 6. Also test with dead_tau > 0 for "nearly dead" ----
        print(f"\n--- Nearly-Dead + Gate Combined ---")
        nearly_dead_results = {}
        for dead_tau in [0.001, 0.005]:
            for gp_tau in [0.02, 0.05]:
                dead_masks = identify_dead_capsules_swiglu(dead_profiles, threshold=dead_tau)
                gate_masks = identify_prunable_by_gate_product(
                    gp_profiles, threshold=gp_tau, method="gate_product_mean_abs")
                combo_masks = combined_pruning_masks(dead_masks, gate_masks)

                model_combo = copy.deepcopy(model)
                combo_stats = prune_swiglu_model(model_combo, combo_masks, verbose=False)
                combo_loss = evaluate(model_combo, joint_val, batch_size=32)
                combo_delta = (combo_loss - unpruned_loss) / unpruned_loss * 100

                key = f"dead_{dead_tau}_gate_{gp_tau}"
                nearly_dead_results[key] = {
                    "pct_pruned": combo_stats["pct_pruned"],
                    "loss": combo_loss,
                    "delta_pct": combo_delta,
                }
                print(f"  dead_tau={dead_tau}, gate_tau={gp_tau}: "
                      f"{combo_stats['pct_pruned']:.1f}% pruned, "
                      f"delta={combo_delta:+.2f}%")

        seed_results["nearly_dead_combined"] = nearly_dead_results
        all_results[str(seed)] = seed_results

    # ---- Aggregate across seeds ----
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")

    # Aggregate unpruned
    unpruned_losses = [all_results[str(s)]["unpruned_loss"] for s in seeds]
    print(f"\nUnpruned: {sum(unpruned_losses)/len(unpruned_losses):.4f} "
          f"(std={_std(unpruned_losses):.4f})")

    # Dead capsule summary
    print(f"\nDead capsule rates (exact, tau=0):")
    for l_idx in range(4):
        rates = [all_results[str(s)]["dead_profiles"][l_idx]["pct_dead_exact"] for s in seeds]
        print(f"  Layer {l_idx}: {sum(rates)/len(rates):.1f}% dead (std={_std(rates):.1f}%)")

    # Aggregate pruning comparison
    print(f"\n--- Aggregate Pruning Comparison (dead_tau=0) ---")
    print(f"{'gate_tau':>10} | {'Dead%':>7} {'dD%':>7} | {'Gate%':>7} {'dG%':>7} | "
          f"{'Combo%':>7} {'dC%':>7} | {'Adv.pp':>7}")
    print("-" * 80)

    best_combo_tau = None
    best_combo_pct = 0
    best_combo_delta = 0

    for gp_tau in gp_thresholds:
        key = f"dead_0.0_gate_{gp_tau}"
        dead_pcts = [all_results[str(s)]["pruning_comparison"][key]["dead_only"]["pct_pruned"]
                     for s in seeds]
        gate_pcts = [all_results[str(s)]["pruning_comparison"][key]["gate_only"]["pct_pruned"]
                     for s in seeds]
        combo_pcts = [all_results[str(s)]["pruning_comparison"][key]["combined"]["pct_pruned"]
                      for s in seeds]
        dead_deltas = [all_results[str(s)]["pruning_comparison"][key]["dead_only"]["delta_pct"]
                       for s in seeds]
        gate_deltas = [all_results[str(s)]["pruning_comparison"][key]["gate_only"]["delta_pct"]
                       for s in seeds]
        combo_deltas = [all_results[str(s)]["pruning_comparison"][key]["combined"]["delta_pct"]
                        for s in seeds]
        advs = [all_results[str(s)]["pruning_comparison"][key]["combined_advantage_pp"]
                for s in seeds]

        md = sum(dead_pcts)/len(dead_pcts)
        mg = sum(gate_pcts)/len(gate_pcts)
        mc = sum(combo_pcts)/len(combo_pcts)
        dd = sum(dead_deltas)/len(dead_deltas)
        dg = sum(gate_deltas)/len(gate_deltas)
        dc = sum(combo_deltas)/len(combo_deltas)
        ma = sum(advs)/len(advs)

        print(f"{gp_tau:>10.3f} | {md:>6.1f}% {dd:>+6.2f}% | {mg:>6.1f}% {dg:>+6.2f}% | "
              f"{mc:>6.1f}% {dc:>+6.2f}% | {ma:>+6.1f}pp")

        if dc <= 3.0 and mc > best_combo_pct:
            best_combo_tau = gp_tau
            best_combo_pct = mc
            best_combo_delta = dc

    # ---- Kill Criteria Check ----
    print(f"\n--- Kill Criteria Check ---")
    print(f"KC1: Combined must exceed best single method by >5pp")
    print(f"KC2: Quality must not degrade >3% vs no pruning")

    kc1_pass = False
    kc2_pass = True

    for gp_tau in gp_thresholds:
        key = f"dead_0.0_gate_{gp_tau}"
        advs = [all_results[str(s)]["pruning_comparison"][key]["combined_advantage_pp"]
                for s in seeds]
        combo_deltas = [all_results[str(s)]["pruning_comparison"][key]["combined"]["delta_pct"]
                        for s in seeds]
        mean_adv = sum(advs) / len(advs)
        mean_delta = sum(combo_deltas) / len(combo_deltas)

        if mean_adv > 5.0 and mean_delta <= 3.0:
            kc1_pass = True
            print(f"  KC1 PASS at gate_tau={gp_tau}: advantage={mean_adv:+.1f}pp, "
                  f"delta={mean_delta:+.2f}%")

        if mean_delta > 3.0:
            print(f"  KC2 WARN at gate_tau={gp_tau}: delta={mean_delta:+.2f}% exceeds 3%")

    if not kc1_pass:
        print(f"  KC1 FAIL: No threshold achieves >5pp combined advantage with <3% quality loss")

    # Final verdict
    print(f"\n--- VERDICT ---")
    if kc1_pass and kc2_pass:
        print(f"  PASS: Combined pruning provides meaningful advantage over either method alone")
        if best_combo_tau is not None:
            print(f"  Best combined: {best_combo_pct:.1f}% pruned at gate_tau={best_combo_tau}, "
                  f"delta={best_combo_delta:+.2f}%")
    elif not kc1_pass:
        print(f"  KILL (KC1): Combined pruning does not exceed best single method by >5pp")
        print(f"  The pruning sets are too overlapping -- dead and gate-product identify "
              f"the same capsules")
    else:
        print(f"  KILL (KC2): Quality degradation exceeds 3% at all useful pruning rates")

    return all_results


def test_combined_pruning_basic():
    """Basic test: combined profiling and pruning runs without errors."""
    from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
    from experiments.train import train, evaluate
    from experiments.models.swiglu_combined_dead_capsule.swiglu_combined_dead_capsule import (
        SwiGLUCombinedDeadCapsuleGPT,
        profile_dead_capsules_swiglu,
        identify_dead_capsules_swiglu,
        profile_gate_products,
        identify_prunable_by_gate_product,
        prune_swiglu_model,
        combined_pruning_masks,
        compute_set_overlap,
    )

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    model = SwiGLUCombinedDeadCapsuleGPT(
        vocab_size=tokenizer.vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4, n_capsules=64)
    mx.eval(model.parameters())

    # Quick train
    train(model, train_ds, val_ds, steps=50, batch_size=16, lr=3e-3, seed=42, log_every=25)

    loss_before = evaluate(model, val_ds, batch_size=16)
    print(f"Loss before pruning: {loss_before:.4f}")

    # Profile both criteria
    dead_profiles = profile_dead_capsules_swiglu(
        model, val_ds, n_batches=5, batch_size=16, seed=42)
    gp_profiles = profile_gate_products(
        model, val_ds, n_batches=5, batch_size=16, seed=42)

    for l_idx in range(4):
        print(f"Layer {l_idx}:")
        print(f"  Dead: {dead_profiles[l_idx]['n_dead']}/{dead_profiles[l_idx]['n_capsules']} "
              f"({dead_profiles[l_idx]['pct_dead']:.1f}%)")
        gp_vals = gp_profiles[l_idx]["gate_product_mean_abs"].tolist()
        n_below_01 = sum(1 for v in gp_vals if v <= 0.01)
        print(f"  Gate product below 0.01: {n_below_01}/{len(gp_vals)}")

    # Test overlap computation
    n_caps = [prof["n_capsules"] for prof in dead_profiles]
    dead_masks = identify_dead_capsules_swiglu(dead_profiles, threshold=0.0)
    gate_masks = identify_prunable_by_gate_product(gp_profiles, threshold=0.01)
    overlap = compute_set_overlap(dead_masks, gate_masks, n_caps)

    print(f"\nOverlap analysis:")
    print(f"  Dead: {overlap['total_dead']}, Gate: {overlap['total_gate_prunable']}")
    print(f"  Both: {overlap['total_both']}, Either: {overlap['total_either']}")
    print(f"  Jaccard: {overlap['jaccard']:.3f}")
    print(f"  Combined pruning: {overlap['pct_combined']:.1f}%")

    # Test combined pruning
    import copy
    dead_masks2 = identify_dead_capsules_swiglu(dead_profiles, threshold=0.0)
    gate_masks2 = identify_prunable_by_gate_product(gp_profiles, threshold=0.01)
    combo_masks = combined_pruning_masks(dead_masks2, gate_masks2)

    model_pruned = copy.deepcopy(model)
    stats = prune_swiglu_model(model_pruned, combo_masks, verbose=True)
    loss_after = evaluate(model_pruned, val_ds, batch_size=16)
    delta = (loss_after - loss_before) / loss_before * 100

    print(f"\nCombined pruning: {stats['pct_pruned']:.1f}% pruned, "
          f"loss {loss_before:.4f} -> {loss_after:.4f} ({delta:+.2f}%)")

    assert loss_after < 10.0, "Pruned model produces garbage"
    print("\nBasic test PASSED")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test_combined_pruning_basic()
    else:
        results = run_experiment()
        out_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/swiglu_combined_dead_capsule/results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
