"""Combined parallel blocks + pure-linear attention composition experiment.

Tests whether two individually-proven architectural simplifications compose
well together:
  1. Parallel blocks: attention + capsule pool from same normalized input
  2. Pure-linear attention: all GatedDeltaNet layers, no full attention

Four conditions (2x2 factorial design):
  A. sequential + hybrid 3:1    (validated baseline -- both experiments' control)
  B. sequential + pure-linear   (pure_linear_composition finding: +1.02%)
  C. parallel   + hybrid 3:1    (parallel_block_capsules finding: -0.39pp)
  D. parallel   + pure-linear   (TEST: combined simplification)

Kill criterion:
  Combined parallel+pure-linear (D) degrades >5% vs sequential+hybrid (A).

Protocol (identical to parent experiments):
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets

5 seeds per condition.
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
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def compose_models(model_a, model_b, model_name, vocab_size, block_size,
                   base_model=None, layer_types=None, is_parallel=False):
    """Create a composed model by concatenating capsule groups from A and B.

    Handles both sequential (norm1/norm2) and parallel (single norm) blocks.
    """
    n_groups_a = model_a.layers[0].capsule_pool.n_groups
    n_groups_b = model_b.layers[0].capsule_pool.n_groups
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = model_a.layers[0].capsule_pool.top_k_groups + \
                     model_b.layers[0].capsule_pool.top_k_groups

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

    source = base_model if base_model is not None else model_a
    composed.wte.weight = source.wte.weight
    composed.wpe.weight = source.wpe.weight
    composed.lm_head.weight = source.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_src = source.layers[l_idx]
        layer_a = model_a.layers[l_idx]
        layer_b = model_b.layers[l_idx]

        # Copy attention weights from base (all Linear layers)
        for name in ["wq", "wk", "wv", "wo", "wg", "w_a", "w_beta", "w_z"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        # Copy non-Linear params
        for name in ["dt_bias", "A_log"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                setattr(layer_c.attn, name, getattr(layer_src.attn, name))

        # Copy conv1d weights if they exist
        for conv_name in ["conv_q", "conv_k", "conv_v"]:
            if hasattr(layer_src.attn, conv_name) and hasattr(layer_c.attn, conv_name):
                src_conv = getattr(layer_src.attn, conv_name)
                dst_conv = getattr(layer_c.attn, conv_name)
                dst_conv.weight = src_conv.weight

        # Copy norms -- handle both parallel (single norm) and sequential (norm1/norm2)
        if is_parallel:
            # Parallel block: single norm
            if hasattr(layer_src, "norm") and hasattr(layer_c, "norm"):
                if hasattr(layer_src.norm, "weight") and hasattr(layer_c.norm, "weight"):
                    layer_c.norm.weight = layer_src.norm.weight
        else:
            # Sequential block: norm1 and norm2
            for norm_name in ["norm1", "norm2"]:
                if hasattr(layer_src, norm_name) and hasattr(layer_c, norm_name):
                    src_norm = getattr(layer_src, norm_name)
                    dst_norm = getattr(layer_c, norm_name)
                    if hasattr(src_norm, "weight") and hasattr(dst_norm, "weight"):
                        dst_norm.weight = src_norm.weight

        # Copy capsule groups
        pool_c = layer_c.capsule_pool
        pool_a = layer_a.capsule_pool
        pool_b = layer_b.capsule_pool

        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = pool_a.groups[g].A.weight
            pool_c.groups[g].B.weight = pool_a.groups[g].B.weight

        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = pool_b.groups[g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = pool_b.groups[g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    """Freeze all params except capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_ds_a, train_ds_b, steps=100, lr=3e-3, seed=42):
    """Train only router weights on mixed-domain data."""
    freeze_except_router(model)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    model.unfreeze()


def run_single_seed(model_name, layer_types, docs, tokenizer, seed,
                    is_parallel=False):
    """Run the full composition protocol for one seed, return metrics."""
    V = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        domain_datasets[name] = (
            CharDataset(dtrain, tokenizer, BASE["block_size"]),
            CharDataset(dval, tokenizer, BASE["block_size"]),
        )
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    kwargs = dict(vocab_size=V, **CAP)
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    # --- Joint training baseline ---
    model_joint = get_model(model_name, **kwargs)
    mx.eval(model_joint.parameters())

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = 2 * STEPS_PER_DOMAIN
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2

    # --- Pretrain base ---
    base_model = get_model(model_name, **kwargs)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # --- Fine-tune domain A ---
    model_a = get_model(model_name, **kwargs)
    mx.eval(model_a.parameters())
    model_a.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_a.parameters())
    model_a.freeze()
    for layer in model_a.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_a, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_a.unfreeze()

    # --- Fine-tune domain B ---
    model_b = get_model(model_name, **kwargs)
    mx.eval(model_b.parameters())
    model_b.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_b.parameters())
    model_b.freeze()
    for layer in model_b.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_b, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_b.unfreeze()

    # --- Compose + calibrate ---
    composed = compose_models(model_a, model_b, model_name, V, BASE["block_size"],
                              base_model=base_model, layer_types=layer_types,
                              is_parallel=is_parallel)
    calibrate_router(composed, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2

    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    # Single-domain avg for reference
    spec_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_b = evaluate(model_b, val_b, BATCH_SIZE)
    single_avg = (spec_a + spec_b) / 2

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "single_avg": single_avg,
        "gap_pct": gap_pct,
    }


def run_experiment(n_seeds=5):
    """Run the 2x2 factorial composition experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    seeds = list(range(n_seeds))

    # 2x2 factorial: {sequential, parallel} x {hybrid_3_1, pure_linear}
    conditions = {
        "seq_hybrid": {
            "model_name": "sequential_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "is_parallel": False,
        },
        "seq_pure_linear": {
            "model_name": "sequential_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "linear"],
            "is_parallel": False,
        },
        "par_hybrid": {
            "model_name": "parallel_pure_linear_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "is_parallel": True,
        },
        "par_pure_linear": {
            "model_name": "parallel_pure_linear_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "linear"],
            "is_parallel": True,
        },
    }

    all_results = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"\n{'='*70}")
        print(f"CONDITION: {cond_name}")
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
                    is_parallel=cond_cfg["is_parallel"],
                )
                print(f"gap={result['gap_pct']:+.2f}%  "
                      f"joint={result['joint_avg']:.4f}  "
                      f"comp={result['composed_avg']:.4f}")
                cond_results.append(result)
            except RuntimeError as e:
                print(f"FAILED (RuntimeError: {e})")
                continue

        if len(cond_results) < 3:
            print(f"  WARNING: Only {len(cond_results)} seeds succeeded")
        all_results[cond_name] = cond_results

    for cond_name, results in all_results.items():
        if len(results) < 3:
            print(f"\nFATAL: {cond_name} has only {len(results)} results. Aborting.")
            return all_results

    # =========================================================================
    # Analysis
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    # --- Per-condition summary ---
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        joints = [r["joint_avg"] for r in results]
        comps = [r["composed_avg"] for r in results]
        print(f"\n--- {cond_name} ---")
        print(f"  Joint:    mean={statistics.mean(joints):.4f}  std={statistics.stdev(joints):.4f}")
        print(f"  Composed: mean={statistics.mean(comps):.4f}  std={statistics.stdev(comps):.4f}")
        print(f"  Gap mean:   {statistics.mean(gaps):+.2f}%")
        print(f"  Gap median: {statistics.median(gaps):+.2f}%")
        print(f"  Gap std:    {statistics.stdev(gaps):.2f}%")

    # =========================================================================
    # PRIMARY KILL CRITERION: par_pure_linear vs seq_hybrid
    # =========================================================================
    print(f"\n{'='*70}")
    print("PRIMARY KILL CRITERION: par_pure_linear vs seq_hybrid (>5% = KILL)")
    print(f"{'='*70}")

    baseline = all_results["seq_hybrid"]
    test = all_results["par_pure_linear"]

    baseline_comp_mean = statistics.mean([r["composed_avg"] for r in baseline])
    test_comp_mean = statistics.mean([r["composed_avg"] for r in test])

    degradation_pct = (test_comp_mean - baseline_comp_mean) / baseline_comp_mean * 100

    baseline_gap_mean = statistics.mean([r["gap_pct"] for r in baseline])
    test_gap_mean = statistics.mean([r["gap_pct"] for r in test])
    gap_diff = test_gap_mean - baseline_gap_mean

    print(f"\n  {'Condition':<20} {'Composed mean':>14} {'Gap mean':>10} {'Gap median':>12}")
    print(f"  {'-'*58}")
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        comp_mean = statistics.mean([r["composed_avg"] for r in results])
        print(f"  {cond_name:<20} {comp_mean:>14.4f} {statistics.mean(gaps):>+9.2f}% {statistics.median(gaps):>+11.2f}%")

    print(f"\n  Degradation (par_pure_linear vs seq_hybrid): {degradation_pct:+.2f}%")
    print(f"  Gap difference (par_pure_linear - seq_hybrid): {gap_diff:+.2f}pp")

    killed = degradation_pct > 5.0
    if killed:
        print(f"\n  ** KILL: Combined degrades {degradation_pct:+.2f}% vs baseline (threshold 5%) **")
    else:
        print(f"\n  PASS: Combined is within 5% of baseline ({degradation_pct:+.2f}%)")

    # =========================================================================
    # SECONDARY: Factorial analysis -- are effects additive?
    # =========================================================================
    print(f"\n{'='*70}")
    print("FACTORIAL ANALYSIS: Are parallel and pure-linear effects additive?")
    print(f"{'='*70}")

    # Effect of parallel (holding attention type constant)
    seq_h_comp = statistics.mean([r["composed_avg"] for r in all_results["seq_hybrid"]])
    par_h_comp = statistics.mean([r["composed_avg"] for r in all_results["par_hybrid"]])
    parallel_effect_hybrid = (par_h_comp - seq_h_comp) / seq_h_comp * 100

    seq_l_comp = statistics.mean([r["composed_avg"] for r in all_results["seq_pure_linear"]])
    par_l_comp = statistics.mean([r["composed_avg"] for r in all_results["par_pure_linear"]])
    parallel_effect_linear = (par_l_comp - seq_l_comp) / seq_l_comp * 100

    # Effect of pure-linear (holding block type constant)
    linear_effect_seq = (seq_l_comp - seq_h_comp) / seq_h_comp * 100
    linear_effect_par = (par_l_comp - par_h_comp) / par_h_comp * 100

    print(f"\n  Effect of parallel blocks (on composed loss):")
    print(f"    With hybrid attention:     {parallel_effect_hybrid:+.2f}%")
    print(f"    With pure-linear attention: {parallel_effect_linear:+.2f}%")
    print(f"\n  Effect of pure-linear attention (on composed loss):")
    print(f"    With sequential blocks:    {linear_effect_seq:+.2f}%")
    print(f"    With parallel blocks:      {linear_effect_par:+.2f}%")

    # Predicted additive = parallel_effect + linear_effect
    predicted_additive = parallel_effect_hybrid + linear_effect_seq
    actual_combined = (par_l_comp - seq_h_comp) / seq_h_comp * 100
    interaction = actual_combined - predicted_additive
    print(f"\n  Predicted additive effect: {predicted_additive:+.2f}%")
    print(f"  Actual combined effect:   {actual_combined:+.2f}%")
    print(f"  Interaction term:         {interaction:+.2f}%")
    if abs(interaction) < 1.0:
        print(f"  Effects are approximately ADDITIVE (|interaction| < 1%)")
    elif interaction > 0:
        print(f"  Negative interaction: combined is WORSE than sum of parts")
    else:
        print(f"  Positive interaction: combined is BETTER than sum of parts")

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
            gap = all_results[cn][i]["gap_pct"]
            row += f" {gap:>+17.2f}%"
        print(row)

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
        "elapsed_seconds": elapsed,
        "conditions": {},
        "kill_criterion": {
            "description": "combined parallel+pure-linear degrades >5% vs sequential+hybrid baseline",
            "degradation_pct": degradation_pct,
            "threshold": 5.0,
            "killed": killed,
            "baseline_comp_mean": baseline_comp_mean,
            "test_comp_mean": test_comp_mean,
            "gap_diff_pp": gap_diff,
        },
        "factorial_analysis": {
            "parallel_effect_with_hybrid": parallel_effect_hybrid,
            "parallel_effect_with_linear": parallel_effect_linear,
            "linear_effect_with_sequential": linear_effect_seq,
            "linear_effect_with_parallel": linear_effect_par,
            "predicted_additive": predicted_additive,
            "actual_combined": actual_combined,
            "interaction_term": interaction,
        },
    }
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        output["conditions"][cond_name] = {
            "gap_mean": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_std": statistics.stdev(gaps),
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

    results_path = "experiments/models/parallel_pure_linear_combined/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=5)
