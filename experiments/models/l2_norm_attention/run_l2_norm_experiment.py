"""L2 QK normalization stability experiment.

Tests whether L2 normalization on Q and K eliminates the ~20% catastrophic
failure rate found in the hybrid attention composition experiment.

Protocol: identical to hybrid_attention/run_composition_experiment.py
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets

Two conditions:
  1. hybrid_3_1 (unnormalized): same as hybrid_attention experiment
  2. l2_norm_3_1 (L2 normalized): L2 norm on Q and K in linear layers

25 seeds per condition for reliable catastrophic failure statistics.

Kill criteria:
  1. L2-normalized still shows >10% catastrophic failure rate across 20+ seeds
  2. L2 normalization degrades median composition gap by >3% vs unnormalized

Catastrophic failure defined as: composition gap > +20% (generous threshold,
original experiment's catastrophic seed showed +88.78%).
"""

import copy
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


# Standard micro config (matches hybrid_attention experiment exactly)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3

# Catastrophic failure threshold
CATASTROPHIC_GAP_THRESHOLD = 20.0  # +20%


def compose_models(model_a, model_b, model_name, vocab_size, block_size,
                   base_model=None, layer_types=None):
    """Create a composed model by concatenating capsule groups from A and B."""
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

        for name in ["wq", "wk", "wv", "wo"]:
            src_param = getattr(layer_src.attn, name).weight
            getattr(layer_c.attn, name).weight = src_param
        if hasattr(layer_src.attn, "wg") and hasattr(layer_c.attn, "wg"):
            layer_c.attn.wg.weight = layer_src.attn.wg.weight

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


def run_single_seed(model_name, layer_types, docs, tokenizer, seed):
    """Run the composition protocol for one seed, return gap_pct."""
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
                              base_model=base_model, layer_types=layer_types)
    calibrate_router(composed, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2

    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "gap_pct": gap_pct,
        "catastrophic": gap_pct > CATASTROPHIC_GAP_THRESHOLD,
    }


def run_experiment(n_seeds=25):
    """Run L2 normalization stability experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    # Generate seeds
    seeds = list(range(n_seeds))  # 0, 1, 2, ..., n_seeds-1

    conditions = {
        "hybrid_unnorm": {
            "model_name": "hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
        },
        "hybrid_l2norm": {
            "model_name": "l2_norm_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
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
            result = run_single_seed(
                model_name=cond_cfg["model_name"],
                layer_types=cond_cfg["layer_types"],
                docs=docs,
                tokenizer=tokenizer,
                seed=seed,
            )
            status = "CATASTROPHIC" if result["catastrophic"] else "ok"
            print(f"gap={result['gap_pct']:+.2f}% [{status}]")
            cond_results.append(result)

        all_results[cond_name] = cond_results

    # --- Analysis ---
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        n_catastrophic = sum(1 for r in results if r["catastrophic"])
        failure_rate = n_catastrophic / len(results) * 100

        print(f"\n--- {cond_name} ---")
        print(f"  Seeds: {n_seeds}")
        print(f"  Catastrophic failures (gap > {CATASTROPHIC_GAP_THRESHOLD}%): "
              f"{n_catastrophic}/{n_seeds} ({failure_rate:.1f}%)")
        print(f"  Gap mean:   {statistics.mean(gaps):+.2f}%")
        print(f"  Gap median: {statistics.median(gaps):+.2f}%")
        print(f"  Gap std:    {statistics.stdev(gaps):.2f}%")
        print(f"  Gap min:    {min(gaps):+.2f}%")
        print(f"  Gap max:    {max(gaps):+.2f}%")

        # List catastrophic seeds
        if n_catastrophic > 0:
            cat_seeds = [seeds[i] for i, r in enumerate(results) if r["catastrophic"]]
            print(f"  Catastrophic seeds: {cat_seeds}")

    # --- Kill Criterion 1: Catastrophic failure rate ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 1: Catastrophic failure rate (threshold: 10%)")
    print(f"{'='*70}")

    l2_results = all_results["hybrid_l2norm"]
    l2_catastrophic = sum(1 for r in l2_results if r["catastrophic"])
    l2_failure_rate = l2_catastrophic / len(l2_results) * 100

    unnorm_results = all_results["hybrid_unnorm"]
    unnorm_catastrophic = sum(1 for r in unnorm_results if r["catastrophic"])
    unnorm_failure_rate = unnorm_catastrophic / len(unnorm_results) * 100

    print(f"  Unnormalized: {unnorm_catastrophic}/{n_seeds} ({unnorm_failure_rate:.1f}%)")
    print(f"  L2 normalized: {l2_catastrophic}/{n_seeds} ({l2_failure_rate:.1f}%)")

    if l2_failure_rate > 10.0:
        print(f"  ** KILL: L2 normalized failure rate {l2_failure_rate:.1f}% > 10% threshold **")
    else:
        print(f"  PASS: L2 normalized failure rate {l2_failure_rate:.1f}% <= 10% threshold")

    # --- Kill Criterion 2: Median gap degradation ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 2: Median gap degradation (threshold: 3%)")
    print(f"{'='*70}")

    unnorm_gaps = [r["gap_pct"] for r in unnorm_results]
    l2_gaps = [r["gap_pct"] for r in l2_results]

    unnorm_median = statistics.median(unnorm_gaps)
    l2_median = statistics.median(l2_gaps)
    degradation = l2_median - unnorm_median

    print(f"  Unnormalized median gap: {unnorm_median:+.2f}%")
    print(f"  L2 normalized median gap: {l2_median:+.2f}%")
    print(f"  Degradation: {degradation:+.2f}pp")

    if degradation > 3.0:
        print(f"  ** KILL: median degradation {degradation:+.2f}pp > 3pp threshold **")
    else:
        print(f"  PASS: median degradation {degradation:+.2f}pp <= 3pp threshold")

    # --- Per-seed gap comparison ---
    print(f"\n{'='*70}")
    print("Per-seed gaps")
    print(f"{'='*70}")
    print(f"{'Seed':>6} {'Unnorm':>10} {'L2 Norm':>10} {'Status':>15}")
    print("-" * 45)
    for i, seed in enumerate(seeds):
        u_gap = unnorm_results[i]["gap_pct"]
        l_gap = l2_results[i]["gap_pct"]
        u_status = "CATASTROPHIC" if unnorm_results[i]["catastrophic"] else ""
        l_status = "CATASTROPHIC" if l2_results[i]["catastrophic"] else ""
        status = f"{u_status}/{l_status}" if u_status or l_status else ""
        print(f"{seed:>6} {u_gap:>+9.2f}% {l_gap:>+9.2f}% {status:>15}")

    # --- Summary verdict ---
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    k1_pass = l2_failure_rate <= 10.0
    k2_pass = degradation <= 3.0

    if k1_pass and k2_pass:
        print("  PASS: L2 normalization eliminates instability without degrading quality")
        print("  The 'conditional pass' from hybrid_attention becomes UNCONDITIONAL")
    elif not k1_pass:
        print("  KILL (criterion 1): L2 normalization does NOT eliminate catastrophic failures")
    elif not k2_pass:
        print("  KILL (criterion 2): L2 normalization degrades composition quality")

    # Save results to JSON
    output = {
        "n_seeds": n_seeds,
        "catastrophic_threshold": CATASTROPHIC_GAP_THRESHOLD,
        "elapsed_seconds": elapsed,
        "conditions": {},
    }
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        n_cat = sum(1 for r in results if r["catastrophic"])
        output["conditions"][cond_name] = {
            "n_catastrophic": n_cat,
            "failure_rate_pct": n_cat / len(results) * 100,
            "gap_mean": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_std": statistics.stdev(gaps),
            "gap_min": min(gaps),
            "gap_max": max(gaps),
            "per_seed": [
                {"seed": seeds[i], "gap_pct": r["gap_pct"],
                 "joint_avg": r["joint_avg"], "composed_avg": r["composed_avg"],
                 "catastrophic": r["catastrophic"]}
                for i, r in enumerate(results)
            ],
        }
    output["kill_criteria"] = {
        "k1_l2_failure_rate": l2_failure_rate,
        "k1_threshold": 10.0,
        "k1_pass": k1_pass,
        "k2_median_degradation": degradation,
        "k2_threshold": 3.0,
        "k2_pass": k2_pass,
        "overall_pass": k1_pass and k2_pass,
    }

    results_path = "experiments/models/l2_norm_attention/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=25)
