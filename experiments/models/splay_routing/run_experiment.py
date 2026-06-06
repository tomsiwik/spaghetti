"""Experiment: Splay-Tree Adaptive Routing vs Static Tree.

Tests two kill criteria:
1. Splay restructuring does NOT reduce routing cost on non-stationary data
2. Splay overhead (restructuring cost) exceeds routing savings

Experimental design:
- Train both splay and static tree models on domain A (a-m names)
- Switch to domain B (n-z names) mid-stream WITHOUT retraining router
- Measure adaptation speed: how quickly does val_loss on domain B improve?
- Compare: splay (automatic bias adjustment) vs static (no adaptation)

The splay tree should adapt faster because:
- Domain B's experts get selected more often -> their frequencies increase
- Gate biases shift toward those experts -> faster convergence on new domain
- Static tree must rely entirely on its frozen gate weights

Additionally, we measure:
- Routing entropy before/after switch (splay should show faster entropy reduction)
- Per-leaf frequency distribution (splay should concentrate on fewer leaves)
- Training quality (splay should not hurt baseline training)
"""

import sys
import time
import random
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_with_tracking(model, train_ds, val_ds, steps=500, batch_size=32,
                        lr=3e-3, seed=42, log_every=50, track_splay=False):
    """Train a model and track per-step metrics including splay state."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    metrics = {"losses": [], "val_losses": [], "splay_states": []}
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        metrics["losses"].append(loss_val)

        if step % log_every == 0 or step == steps:
            val_loss = evaluate(model, val_ds, batch_size, n_batches=5)
            metrics["val_losses"].append({"step": step, "val_loss": val_loss})
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss_val:.4f} | val {val_loss:.4f} | {elapsed:.1f}s")

            if track_splay and hasattr(model, 'get_splay_diagnostics'):
                splay_diag = model.get_splay_diagnostics()
                metrics["splay_states"].append({
                    "step": step,
                    "layer0_freq": splay_diag[0]["leaf_freq"],
                    "layer0_biases": splay_diag[0]["gate_biases"],
                })

    final_val = evaluate(model, val_ds, batch_size)
    metrics["final_val"] = final_val
    metrics["elapsed_s"] = time.time() - t0
    return metrics


def compute_routing_entropy(model, dataset, batch_size=32, n_batches=5):
    """Compute mean normalized routing entropy across layers."""
    import math
    rng = random.Random(0)
    entropies = []

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        _ = model(inputs)
        mx.eval(model.parameters())

        for layer in model.layers:
            tree = layer.tree
            if tree._leaf_probs is not None:
                lp = tree._leaf_probs  # (B, T, n_leaves)
                eps = 1e-8
                H = -mx.sum(lp * mx.log(lp + eps), axis=-1)  # (B, T)
                H_max = math.log(tree.n_leaves)
                norm_H = mx.mean(H).item() / H_max
                entropies.append(norm_H)

    return sum(entropies) / len(entropies) if entropies else 0.0


def run_domain_shift_experiment(seeds=(42, 123, 777),
                                 steps_domain_a=300,
                                 steps_domain_b=200):
    """Main experiment: train on domain A, switch to B, measure adaptation.

    Kill criterion 1: Splay routing should reduce routing cost (measured as
    time-to-convergence) on non-stationary data. If splay's domain-B val_loss
    is not better than static's at matched steps, KILL.

    Kill criterion 2: Splay overhead should not exceed savings. If splay is
    slower (wall-clock) or worse quality overall, KILL.
    """
    print("=" * 70)
    print("SPLAY ADAPTIVE ROUTING vs STATIC TREE")
    print("Domain shift experiment: A (a-m) -> B (n-z)")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)

    results = {
        "splay": {"domain_a_val": [], "domain_b_val": [], "domain_b_trajectory": [],
                   "entropy_before_switch": [], "entropy_after_switch": [],
                   "total_time": []},
        "static": {"domain_a_val": [], "domain_b_val": [], "domain_b_trajectory": [],
                    "entropy_before_switch": [], "entropy_after_switch": [],
                    "total_time": []},
    }

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        # Prepare domain datasets
        a_train, a_val = train_val_split(splits["a_m"], seed=seed)
        b_train, b_val = train_val_split(splits["n_z"], seed=seed)
        ds_a_train = CharDataset(a_train, tokenizer, 32)
        ds_a_val = CharDataset(a_val, tokenizer, 32)
        ds_b_train = CharDataset(b_train, tokenizer, 32)
        ds_b_val = CharDataset(b_val, tokenizer, 32)

        for model_type in ["static", "splay"]:
            print(f"\n  --- {model_type} tree ---")
            t0_total = time.time()

            mx.random.seed(seed)
            if model_type == "static":
                model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                                   tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
            else:
                model = get_model("splay_routing", vocab_size=vs, block_size=32,
                                   tree_depth=3, n_capsules_per_leaf=32, beam_width=2,
                                   splay_alpha=1.0, splay_decay=0.95)
            mx.eval(model.parameters())
            n_params = count_params(model)
            print(f"    params: {n_params:,}")

            # Phase 1: Train on domain A
            print(f"\n    Phase 1: Domain A (a-m), {steps_domain_a} steps")
            a_metrics = train_with_tracking(
                model, ds_a_train, ds_a_val,
                steps=steps_domain_a, seed=seed, log_every=100,
                track_splay=(model_type == "splay")
            )
            results[model_type]["domain_a_val"].append(a_metrics["final_val"])

            # Measure routing entropy before switch
            entropy_before = compute_routing_entropy(model, ds_b_val)
            results[model_type]["entropy_before_switch"].append(entropy_before)
            print(f"    Entropy on domain B (before switch): {entropy_before:.4f}")

            # Phase 2: Switch to domain B (no retraining, just continue training)
            print(f"\n    Phase 2: Domain B (n-z), {steps_domain_b} steps")
            if model_type == "splay":
                model.on_domain_switch("n_z")  # Reset splay biases
                print("    [splay biases reset]")

            b_metrics = train_with_tracking(
                model, ds_b_train, ds_b_val,
                steps=steps_domain_b, seed=seed + 100, log_every=50,
                track_splay=(model_type == "splay")
            )
            results[model_type]["domain_b_val"].append(b_metrics["final_val"])
            results[model_type]["domain_b_trajectory"].append(
                b_metrics["val_losses"]
            )

            # Measure routing entropy after switch
            entropy_after = compute_routing_entropy(model, ds_b_val)
            results[model_type]["entropy_after_switch"].append(entropy_after)
            print(f"    Entropy on domain B (after switch): {entropy_after:.4f}")

            total_time = time.time() - t0_total
            results[model_type]["total_time"].append(total_time)
            print(f"    Total time: {total_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    n_seeds = len(seeds)
    for mt in ["static", "splay"]:
        a_mean = sum(results[mt]["domain_a_val"]) / n_seeds
        b_mean = sum(results[mt]["domain_b_val"]) / n_seeds
        ent_before = sum(results[mt]["entropy_before_switch"]) / n_seeds
        ent_after = sum(results[mt]["entropy_after_switch"]) / n_seeds
        time_mean = sum(results[mt]["total_time"]) / n_seeds
        print(f"\n  {mt:6s}:")
        print(f"    Domain A val_loss (mean): {a_mean:.4f}  {results[mt]['domain_a_val']}")
        print(f"    Domain B val_loss (mean): {b_mean:.4f}  {results[mt]['domain_b_val']}")
        print(f"    Entropy before switch:    {ent_before:.4f}")
        print(f"    Entropy after switch:     {ent_after:.4f}")
        print(f"    Mean wall time:           {time_mean:.1f}s")

    # Domain B convergence comparison
    static_b_mean = sum(results["static"]["domain_b_val"]) / n_seeds
    splay_b_mean = sum(results["splay"]["domain_b_val"]) / n_seeds
    delta_pct = 100 * (splay_b_mean - static_b_mean) / static_b_mean

    print(f"\n  Domain B comparison:")
    print(f"    Static:  {static_b_mean:.4f}")
    print(f"    Splay:   {splay_b_mean:.4f}")
    print(f"    Delta:   {delta_pct:+.2f}% (negative = splay better)")

    # Early convergence comparison (val_loss at step 50 on domain B)
    print(f"\n  Early convergence (domain B, first checkpoint):")
    for mt in ["static", "splay"]:
        early_vals = []
        for traj in results[mt]["domain_b_trajectory"]:
            if traj:
                early_vals.append(traj[0]["val_loss"])
        if early_vals:
            early_mean = sum(early_vals) / len(early_vals)
            print(f"    {mt:6s}: {early_mean:.4f} at step {traj[0]['step']}")

    # Entropy comparison
    static_ent_before = sum(results["static"]["entropy_before_switch"]) / n_seeds
    static_ent_after = sum(results["static"]["entropy_after_switch"]) / n_seeds
    splay_ent_before = sum(results["splay"]["entropy_before_switch"]) / n_seeds
    splay_ent_after = sum(results["splay"]["entropy_after_switch"]) / n_seeds

    print(f"\n  Routing entropy (normalized, 1.0=uniform):")
    print(f"    Static: {static_ent_before:.4f} -> {static_ent_after:.4f} (delta: {static_ent_after - static_ent_before:+.4f})")
    print(f"    Splay:  {splay_ent_before:.4f} -> {splay_ent_after:.4f} (delta: {splay_ent_after - splay_ent_before:+.4f})")

    # Time overhead
    static_time = sum(results["static"]["total_time"]) / n_seeds
    splay_time = sum(results["splay"]["total_time"]) / n_seeds
    overhead_pct = 100 * (splay_time - static_time) / static_time

    print(f"\n  Time overhead:")
    print(f"    Static: {static_time:.1f}s")
    print(f"    Splay:  {splay_time:.1f}s")
    print(f"    Overhead: {overhead_pct:+.1f}%")

    # Kill criteria assessment
    print(f"\n{'='*70}")
    print("KILL CRITERIA ASSESSMENT")
    print(f"{'='*70}")

    # KC1: Splay should reduce routing cost on non-stationary data
    # Measured as: splay domain-B val_loss <= static domain-B val_loss
    kc1_pass = splay_b_mean <= static_b_mean * 1.005  # within 0.5% tolerance
    print(f"\n  KC1: Splay reduces routing cost on non-stationary data")
    print(f"       Static B val: {static_b_mean:.4f}, Splay B val: {splay_b_mean:.4f}")
    print(f"       Delta: {delta_pct:+.2f}%")
    print(f"       Verdict: {'PASS' if kc1_pass else 'KILL'}")

    # KC2: Splay overhead should not exceed savings
    # Measured as: splay wall-clock time <= 1.2x static (20% overhead budget)
    kc2_pass = splay_time <= static_time * 1.20
    print(f"\n  KC2: Splay overhead does not exceed routing savings")
    print(f"       Static time: {static_time:.1f}s, Splay time: {splay_time:.1f}s")
    print(f"       Overhead: {overhead_pct:+.1f}%")
    print(f"       Verdict: {'PASS' if kc2_pass else 'KILL'}")

    overall = "PASS" if (kc1_pass and kc2_pass) else "KILL"
    print(f"\n  OVERALL: {overall}")

    # Save results
    save_results = {
        "seeds": list(seeds),
        "steps_domain_a": steps_domain_a,
        "steps_domain_b": steps_domain_b,
        "static_domain_a_val": results["static"]["domain_a_val"],
        "static_domain_b_val": results["static"]["domain_b_val"],
        "splay_domain_a_val": results["splay"]["domain_a_val"],
        "splay_domain_b_val": results["splay"]["domain_b_val"],
        "static_entropy_before": results["static"]["entropy_before_switch"],
        "static_entropy_after": results["static"]["entropy_after_switch"],
        "splay_entropy_before": results["splay"]["entropy_before_switch"],
        "splay_entropy_after": results["splay"]["entropy_after_switch"],
        "static_time": results["static"]["total_time"],
        "splay_time": results["splay"]["total_time"],
        "delta_pct": delta_pct,
        "overhead_pct": overhead_pct,
        "kc1_pass": kc1_pass,
        "kc2_pass": kc2_pass,
        "overall": overall,
    }

    with open("/Users/tom/Code/tomsiwik/llm/experiments/models/splay_routing/results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to experiments/models/splay_routing/results.json")

    return results, save_results


def run_alpha_sweep(seed=42, steps_a=300, steps_b=200):
    """Sweep splay_alpha to find optimal strength."""
    print("\n" + "=" * 70)
    print("SPLAY ALPHA SWEEP")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)

    a_train, a_val = train_val_split(splits["a_m"], seed=seed)
    b_train, b_val = train_val_split(splits["n_z"], seed=seed)
    ds_a_train = CharDataset(a_train, tokenizer, 32)
    ds_a_val = CharDataset(a_val, tokenizer, 32)
    ds_b_train = CharDataset(b_train, tokenizer, 32)
    ds_b_val = CharDataset(b_val, tokenizer, 32)

    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    sweep_results = []

    for alpha in alphas:
        print(f"\n  alpha={alpha}")
        mx.random.seed(seed)
        model = get_model("splay_routing", vocab_size=vs, block_size=32,
                           tree_depth=3, n_capsules_per_leaf=32, beam_width=2,
                           splay_alpha=alpha, splay_decay=0.95)
        mx.eval(model.parameters())

        # Phase 1: domain A
        a_metrics = train_with_tracking(model, ds_a_train, ds_a_val,
                                         steps=steps_a, seed=seed, log_every=300)

        # Switch to domain B
        model.on_domain_switch("n_z")
        b_metrics = train_with_tracking(model, ds_b_train, ds_b_val,
                                         steps=steps_b, seed=seed + 100, log_every=200)

        sweep_results.append({
            "alpha": alpha,
            "domain_a_val": a_metrics["final_val"],
            "domain_b_val": b_metrics["final_val"],
        })
        print(f"    A val: {a_metrics['final_val']:.4f}, B val: {b_metrics['final_val']:.4f}")

    print(f"\n  Alpha sweep summary:")
    print(f"  {'alpha':>6s}  {'A val':>8s}  {'B val':>8s}")
    for r in sweep_results:
        print(f"  {r['alpha']:6.1f}  {r['domain_a_val']:8.4f}  {r['domain_b_val']:8.4f}")

    return sweep_results


def main():
    t0 = time.time()

    # Main experiment
    results, summary = run_domain_shift_experiment(
        seeds=(42, 123, 777),
        steps_domain_a=300,
        steps_domain_b=200,
    )

    # Alpha sweep (single seed, for direction)
    sweep = run_alpha_sweep(seed=42)

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
