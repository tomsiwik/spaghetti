"""Combined parallel+pure-linear composition at N=5 domains.

Scales the 2x2 factorial experiment from parallel_pure_linear_combined
to N=5 domains using the quintary character split (a-e, f-j, k-o, p-t, u-z).

Two conditions (not full 2x2 -- focus on the test and baseline):
  A. sequential + hybrid 3:1    (validated baseline)
  B. parallel   + pure-linear   (test: combined simplification at N=5)

Kill criterion:
  Combined parallel+pure-linear (B) composition gap >8% vs joint training.

Protocol (identical to parent, extended to 5 domains):
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps each, attention frozen)
  Phase 3: Compose by concatenating all 5 domain groups, scale top-k
  Phase 4: Calibrate router on mixed data (200 steps -- scaled from 100 for N=2)
  Phase 5: Evaluate on per-domain val sets

3 seeds per condition (sufficient given 8% kill threshold with >150x
margins at N=2).
"""

import random
import statistics
import time
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


# Standard micro config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PRETRAIN = 300
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 200  # scaled from 100 at N=2 to account for 5 domains
BATCH_SIZE = 32
LR = 3e-3
N_DOMAINS = 5


def compose_n_models(domain_models, model_name, vocab_size, block_size,
                     base_model, layer_types=None, is_parallel=False):
    """Create a composed model by concatenating capsule groups from N domains.

    Generalizes compose_models from the N=2 experiment to N domains.
    """
    n_groups_per_domain = domain_models[0].layers[0].capsule_pool.n_groups
    composed_groups = n_groups_per_domain * len(domain_models)
    composed_top_k = domain_models[0].layers[0].capsule_pool.top_k_groups * len(domain_models)

    kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_groups=composed_groups,
        n_capsules_per_group=CAP["n_capsules_per_group"],
        top_k_groups=composed_top_k,
        n_embd=BASE["n_embd"], n_head=BASE["n_head"], n_layer=BASE["n_layer"],
    )
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    composed = get_model(model_name, **kwargs)

    # Copy shared params from base
    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_src = base_model.layers[l_idx]

        # Copy attention weights from base
        for name in ["wq", "wk", "wv", "wo", "wg", "w_a", "w_beta", "w_z"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        # Copy non-Linear params
        for name in ["dt_bias", "A_log"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                setattr(layer_c.attn, name, getattr(layer_src.attn, name))

        # Copy conv1d weights
        for conv_name in ["conv_q", "conv_k", "conv_v"]:
            if hasattr(layer_src.attn, conv_name) and hasattr(layer_c.attn, conv_name):
                src_conv = getattr(layer_src.attn, conv_name)
                dst_conv = getattr(layer_c.attn, conv_name)
                dst_conv.weight = src_conv.weight

        # Copy norms
        if is_parallel:
            if hasattr(layer_src, "norm") and hasattr(layer_c, "norm"):
                if hasattr(layer_src.norm, "weight") and hasattr(layer_c.norm, "weight"):
                    layer_c.norm.weight = layer_src.norm.weight
        else:
            for norm_name in ["norm1", "norm2"]:
                if hasattr(layer_src, norm_name) and hasattr(layer_c, norm_name):
                    src_norm = getattr(layer_src, norm_name)
                    dst_norm = getattr(layer_c, norm_name)
                    if hasattr(src_norm, "weight") and hasattr(dst_norm, "weight"):
                        dst_norm.weight = src_norm.weight

        # Copy capsule groups from each domain model
        pool_c = layer_c.capsule_pool
        for d_idx, dm in enumerate(domain_models):
            pool_d = dm.layers[l_idx].capsule_pool
            offset = d_idx * n_groups_per_domain
            for g in range(n_groups_per_domain):
                pool_c.groups[offset + g].A.weight = pool_d.groups[g].A.weight
                pool_c.groups[offset + g].B.weight = pool_d.groups[g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    """Freeze all params except capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, domain_train_datasets, steps=200, lr=3e-3, seed=42):
    """Train only router weights on mixed-domain data (round-robin)."""
    freeze_except_router(model)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    ds_list = list(domain_train_datasets.values())
    n_ds = len(ds_list)

    for step in range(1, steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    model.unfreeze()


def run_single_seed(model_name, layer_types, docs, tokenizer, seed,
                    domain_datasets, is_parallel=False):
    """Run the full N=5 composition protocol for one seed, return metrics."""
    V = tokenizer.vocab_size

    domain_names = list(domain_datasets.keys())
    domain_trains = {d: domain_datasets[d][0] for d in domain_names}
    domain_vals = {d: domain_datasets[d][1] for d in domain_names}

    all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train_docs, tokenizer, BASE["block_size"])

    kwargs = dict(vocab_size=V, **CAP)
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    # --- Joint training baseline ---
    model_joint = get_model(model_name, **kwargs)
    mx.eval(model_joint.parameters())

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = N_DOMAINS * STEPS_PER_DOMAIN
    ds_list = list(domain_trains.values())
    n_ds = len(ds_list)
    for step in range(1, total_steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    joint_vals = [evaluate(model_joint, domain_vals[d], BATCH_SIZE) for d in domain_names]
    joint_avg = sum(joint_vals) / len(joint_vals)

    # --- Pretrain base ---
    base_model = get_model(model_name, **kwargs)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # --- Fine-tune per domain ---
    domain_models = []
    for d_name in domain_names:
        model_d = get_model(model_name, **kwargs)
        mx.eval(model_d.parameters())
        # Load base weights
        model_d.load_weights(list(zip(
            [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
            [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
        )))
        mx.eval(model_d.parameters())
        # Freeze everything except capsule groups
        model_d.freeze()
        for layer in model_d.layers:
            for group in layer.capsule_pool.groups:
                group.unfreeze()
        train(model_d, domain_trains[d_name], domain_vals[d_name],
              steps=STEPS_PER_DOMAIN, batch_size=BATCH_SIZE, lr=LR,
              seed=seed, log_every=9999)
        model_d.unfreeze()
        domain_models.append(model_d)

    # --- Compose + calibrate ---
    composed = compose_n_models(
        domain_models, model_name, V, BASE["block_size"],
        base_model=base_model, layer_types=layer_types,
        is_parallel=is_parallel,
    )
    calibrate_router(composed, domain_trains,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_vals = [evaluate(composed, domain_vals[d], BATCH_SIZE) for d in domain_names]
    comp_avg = sum(comp_vals) / len(comp_vals)

    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    # Single-domain specialists average
    spec_vals = []
    for i, d_name in enumerate(domain_names):
        spec_vals.append(evaluate(domain_models[i], domain_vals[d_name], BATCH_SIZE))
    single_avg = sum(spec_vals) / len(spec_vals)

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "single_avg": single_avg,
        "gap_pct": gap_pct,
        "joint_per_domain": {d: v for d, v in zip(domain_names, joint_vals)},
        "comp_per_domain": {d: v for d, v in zip(domain_names, comp_vals)},
    }


def run_experiment(n_seeds=3):
    """Run the N=5 composition experiment comparing seq_hybrid vs par_pure_linear."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="quintary")

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=42)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    seeds = list(range(n_seeds))

    # Two conditions: baseline (seq_hybrid) and test (par_pure_linear)
    conditions = {
        "seq_hybrid": {
            "model_name": "sequential_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "is_parallel": False,
        },
        "par_pure_linear": {
            "model_name": "combined_n5_scaling",
            "layer_types": ["linear", "linear", "linear", "linear"],
            "is_parallel": True,
        },
    }

    all_results = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"\n{'='*70}")
        print(f"CONDITION: {cond_name} (N={N_DOMAINS} domains)")
        print(f"{'='*70}")

        cond_results = []
        for i, seed in enumerate(seeds):
            print(f"  Seed {seed} ({i+1}/{n_seeds})...", end=" ", flush=True)
            try:
                result = run_single_seed(
                    model_name=cond_cfg["model_name"],
                    layer_types=cond_cfg["layer_types"],
                    docs=docs,
                    tokenizer=tokenizer,
                    seed=seed,
                    domain_datasets=domain_datasets,
                    is_parallel=cond_cfg["is_parallel"],
                )
                print(f"gap={result['gap_pct']:+.2f}%  "
                      f"joint={result['joint_avg']:.4f}  "
                      f"comp={result['composed_avg']:.4f}")
                cond_results.append(result)
            except RuntimeError as e:
                print(f"FAILED (RuntimeError: {e})")
                continue

        if len(cond_results) < 2:
            print(f"  WARNING: Only {len(cond_results)} seeds succeeded")
        all_results[cond_name] = cond_results

    for cond_name, results in all_results.items():
        if len(results) < 2:
            print(f"\nFATAL: {cond_name} has only {len(results)} results. Aborting.")
            return all_results

    # =========================================================================
    # Analysis
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS (N={N_DOMAINS} domains, {n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    # --- Per-condition summary ---
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        joints = [r["joint_avg"] for r in results]
        comps = [r["composed_avg"] for r in results]
        print(f"\n--- {cond_name} ---")
        print(f"  Joint:    mean={statistics.mean(joints):.4f}  "
              f"std={statistics.stdev(joints):.4f}" if len(joints) > 1 else "")
        print(f"  Composed: mean={statistics.mean(comps):.4f}  "
              f"std={statistics.stdev(comps):.4f}" if len(comps) > 1 else "")
        print(f"  Gap mean:   {statistics.mean(gaps):+.2f}%")
        print(f"  Gap median: {statistics.median(gaps):+.2f}%")
        if len(gaps) > 1:
            print(f"  Gap std:    {statistics.stdev(gaps):.2f}%")

    # =========================================================================
    # PRIMARY KILL CRITERION: par_pure_linear composition gap > 8%
    # =========================================================================
    print(f"\n{'='*70}")
    print("PRIMARY KILL CRITERION: par_pure_linear N=5 composition gap >8%")
    print(f"{'='*70}")

    test = all_results["par_pure_linear"]
    test_gap_mean = statistics.mean([r["gap_pct"] for r in test])
    test_comp_mean = statistics.mean([r["composed_avg"] for r in test])
    test_joint_mean = statistics.mean([r["joint_avg"] for r in test])

    baseline = all_results["seq_hybrid"]
    baseline_gap_mean = statistics.mean([r["gap_pct"] for r in baseline])
    baseline_comp_mean = statistics.mean([r["composed_avg"] for r in baseline])
    baseline_joint_mean = statistics.mean([r["joint_avg"] for r in baseline])

    # Gap is test composed vs test joint
    killed = test_gap_mean > 8.0

    print(f"\n  par_pure_linear:")
    print(f"    Joint mean:    {test_joint_mean:.4f}")
    print(f"    Composed mean: {test_comp_mean:.4f}")
    print(f"    Composition gap: {test_gap_mean:+.2f}%")
    print(f"\n  seq_hybrid (reference):")
    print(f"    Joint mean:    {baseline_joint_mean:.4f}")
    print(f"    Composed mean: {baseline_comp_mean:.4f}")
    print(f"    Composition gap: {baseline_gap_mean:+.2f}%")

    # Cross-architecture comparison
    cross_degradation = (test_comp_mean - baseline_comp_mean) / baseline_comp_mean * 100
    gap_diff = test_gap_mean - baseline_gap_mean

    print(f"\n  Cross-architecture comparison:")
    print(f"    Composed loss degradation (par_pure_linear vs seq_hybrid): {cross_degradation:+.2f}%")
    print(f"    Gap difference (par_pure_linear - seq_hybrid): {gap_diff:+.2f}pp")

    if killed:
        print(f"\n  ** KILL: par_pure_linear N=5 gap is {test_gap_mean:+.2f}% (threshold 8%) **")
    else:
        print(f"\n  PASS: par_pure_linear N=5 gap is {test_gap_mean:+.2f}% (threshold 8%)")

    # =========================================================================
    # N=2 vs N=5 comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("N=2 vs N=5 Scaling")
    print(f"{'='*70}")

    print(f"\n  N=2 results (from parallel_pure_linear_combined):")
    print(f"    par_pure_linear gap: +0.96% (5 seeds)")
    print(f"    seq_hybrid gap:      -0.50% (5 seeds)")
    print(f"    Degradation (par_pl vs seq_h): +1.48%")
    print(f"\n  N=5 results (this experiment):")
    print(f"    par_pure_linear gap: {test_gap_mean:+.2f}% ({n_seeds} seeds)")
    print(f"    seq_hybrid gap:      {baseline_gap_mean:+.2f}% ({n_seeds} seeds)")
    print(f"    Degradation (par_pl vs seq_h): {cross_degradation:+.2f}%")
    print(f"\n  Scaling trend:")
    print(f"    N=2 degradation: +1.48%")
    print(f"    N=5 degradation: {cross_degradation:+.2f}%")
    print(f"    Change: {cross_degradation - 1.48:+.2f}pp")

    # =========================================================================
    # Per-seed details
    # =========================================================================
    print(f"\n{'='*70}")
    print("Per-seed composition gaps")
    print(f"{'='*70}")
    cond_names = list(all_results.keys())
    header = f"  {'Seed':>4}"
    for cn in cond_names:
        header += f" {cn:>18}"
    print(header)
    print(f"  " + "-" * (4 + 19 * len(cond_names)))
    for i in range(n_seeds):
        row = f"  {i:>4}"
        for cn in cond_names:
            if i < len(all_results[cn]):
                gap = all_results[cn][i]["gap_pct"]
                row += f" {gap:>+17.2f}%"
            else:
                row += f" {'N/A':>18}"
        print(row)

    # =========================================================================
    # Per-domain analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("Per-domain composition gaps (par_pure_linear, seed-averaged)")
    print(f"{'='*70}")
    domain_names = list(test[0]["joint_per_domain"].keys())
    print(f"\n  {'Domain':<8} | {'Joint':>8} | {'Composed':>10} | {'Gap':>8}")
    print(f"  " + "-" * 44)
    for d_name in domain_names:
        j_vals = [r["joint_per_domain"][d_name] for r in test]
        c_vals = [r["comp_per_domain"][d_name] for r in test]
        j_mean = statistics.mean(j_vals)
        c_mean = statistics.mean(c_vals)
        d_gap = (c_mean - j_mean) / j_mean * 100
        print(f"  {d_name:<8} | {j_mean:>8.4f} | {c_mean:>10.4f} | {d_gap:>+7.2f}%")

    # =========================================================================
    # Catastrophic failure check
    # =========================================================================
    print(f"\n{'='*70}")
    print("Catastrophic failure check (gap > 20%)")
    print(f"{'='*70}")
    for cond_name, results in all_results.items():
        failures = [(i, r["gap_pct"]) for i, r in enumerate(results) if r["gap_pct"] > 20.0]
        if failures:
            print(f"  {cond_name}: {len(failures)}/{n_seeds} failures: {failures}")
        else:
            print(f"  {cond_name}: 0/{n_seeds} catastrophic failures")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "n_seeds": n_seeds,
        "n_domains": N_DOMAINS,
        "elapsed_seconds": elapsed,
        "conditions": {},
        "kill_criterion": {
            "description": "par_pure_linear N=5 composition gap >8%",
            "test_gap_mean_pct": test_gap_mean,
            "threshold": 8.0,
            "killed": killed,
            "test_comp_mean": test_comp_mean,
            "test_joint_mean": test_joint_mean,
        },
        "cross_architecture": {
            "degradation_pct": cross_degradation,
            "gap_diff_pp": gap_diff,
            "baseline_comp_mean": baseline_comp_mean,
            "test_comp_mean": test_comp_mean,
        },
        "n2_comparison": {
            "n2_degradation": 1.48,
            "n5_degradation": cross_degradation,
            "delta_pp": cross_degradation - 1.48,
        },
    }
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        output["conditions"][cond_name] = {
            "gap_mean": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0,
            "composed_mean": statistics.mean([r["composed_avg"] for r in results]),
            "joint_mean": statistics.mean([r["joint_avg"] for r in results]),
            "per_seed": [
                {
                    "seed": i,
                    "gap_pct": r["gap_pct"],
                    "joint_avg": r["joint_avg"],
                    "composed_avg": r["composed_avg"],
                    "single_avg": r["single_avg"],
                }
                for i, r in enumerate(results)
            ],
        }

    results_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/combined_n5_scaling/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=3)
