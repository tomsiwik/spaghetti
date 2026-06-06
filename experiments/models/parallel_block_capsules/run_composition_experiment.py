"""Composition experiment: parallel vs sequential block capsule injection.

Compares capsule composition quality between:
  1. sequential_capsule_moe: standard pre-norm (norm -> attn -> + -> norm -> mlp -> +)
  2. parallel_capsule_moe: Tiny Aya style (norm -> [attn, mlp] -> +)

Protocol (established in capsule_moe/test_composition.py):
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune only capsule groups per domain (freeze attention) -- 300 steps
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets

Kill criterion:
  Parallel block composition degrades >5% vs sequential block composition
  (measured as composition gap relative to joint training baseline)
"""

import copy
import random
import statistics
import time

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
                   base_model=None):
    """Create a composed model by concatenating capsule groups from A and B.

    Uses shared base model for attention weights.
    Capsule groups from model_a go to slots 0..G-1, from model_b to slots G..2G-1.
    Router is randomly initialized (will be calibrated).
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

    composed = get_model(model_name, **kwargs)

    # Use base model for shared params (attention, embeddings)
    source = base_model if base_model is not None else model_a
    composed.wte.weight = source.wte.weight
    composed.wpe.weight = source.wpe.weight
    composed.lm_head.weight = source.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_src = source.layers[l_idx]
        layer_a = model_a.layers[l_idx]
        layer_b = model_b.layers[l_idx]

        # Copy attention from base/source
        for name in ["wq", "wk", "wv", "wo"]:
            src_param = getattr(layer_src.attn, name).weight
            getattr(layer_c.attn, name).weight = src_param

        # Copy capsule groups: first G from A, next G from B
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

        if step % 50 == 0 or step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


def run_single_condition(model_name, condition_name, docs, tokenizer, seed=42):
    """Run the full composition protocol for one block type condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name} (seed={seed})")
    print(f"{'='*70}")

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
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # --- Joint training baseline ---
    print(f"\n--- Joint training ({condition_name}) ---")
    kwargs = dict(vocab_size=V, **CAP)
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
        if step % 200 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2
    results["joint"] = {"a_m": joint_val_a, "n_z": joint_val_b, "avg": joint_avg}
    print(f"  Joint: a_m={joint_val_a:.4f}, n_z={joint_val_b:.4f}, avg={joint_avg:.4f}")

    # --- Phase 1: Pretrain base ---
    print(f"\n--- Pretrain base ({condition_name}) ---")
    kwargs_base = dict(vocab_size=V, **CAP)
    base_model = get_model(model_name, **kwargs_base)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    # --- Phase 2a: Fine-tune capsules on domain A ---
    print(f"\n--- Fine-tune domain A ({condition_name}) ---")
    model_a = get_model(model_name, **kwargs_base)
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
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_a.unfreeze()

    # --- Phase 2b: Fine-tune capsules on domain B ---
    print(f"\n--- Fine-tune domain B ({condition_name}) ---")
    model_b = get_model(model_name, **kwargs_base)
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
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_b.unfreeze()

    # --- Phase 3: Compose + calibrate ---
    print(f"\n--- Compose + calibrate ({condition_name}) ---")
    composed = compose_models(model_a, model_b, model_name, V, BASE["block_size"],
                              base_model=base_model)
    calibrate_router(composed, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2
    results["composed"] = {"a_m": comp_val_a, "n_z": comp_val_b, "avg": comp_avg}
    print(f"  Composed: a_m={comp_val_a:.4f}, n_z={comp_val_b:.4f}, avg={comp_avg:.4f}")

    # --- Composition gap ---
    gap_pct = (comp_avg - joint_avg) / joint_avg * 100
    results["gap_pct"] = gap_pct
    print(f"  Gap vs joint: {gap_pct:+.2f}%")

    # Single-domain eval
    spec_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_b = evaluate(model_b, val_b, BATCH_SIZE)
    single_avg = (spec_a + spec_b) / 2
    results["single_avg"] = single_avg
    print(f"\n  Single-domain avg: {single_avg:.4f}")

    return results


def run_experiment(seeds=(42, 123, 777)):
    """Run the full experiment across conditions and seeds."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    conditions = {
        "sequential": {
            "model_name": "sequential_capsule_moe",
        },
        "parallel": {
            "model_name": "parallel_capsule_moe",
        },
    }

    all_results = {}
    for cond_name, cond_cfg in conditions.items():
        all_results[cond_name] = {}
        for seed in seeds:
            result = run_single_condition(
                model_name=cond_cfg["model_name"],
                condition_name=f"{cond_name} (seed={seed})",
                docs=docs,
                tokenizer=tokenizer,
                seed=seed,
            )
            all_results[cond_name][seed] = result

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT SUMMARY ({len(seeds)} seeds, {elapsed:.0f}s total)")
    print(f"{'='*70}")

    print(f"\n{'Condition':<16} {'Joint':>10} {'Composed':>10} {'Single':>10} {'Gap (%)':>10}")
    print("-" * 60)
    for cond_name in conditions:
        joints = [all_results[cond_name][s]["joint"]["avg"] for s in seeds]
        comps = [all_results[cond_name][s]["composed"]["avg"] for s in seeds]
        singles = [all_results[cond_name][s]["single_avg"] for s in seeds]
        gaps = [all_results[cond_name][s]["gap_pct"] for s in seeds]
        print(f"{cond_name:<16} {statistics.mean(joints):>10.4f} "
              f"{statistics.mean(comps):>10.4f} {statistics.mean(singles):>10.4f} "
              f"{statistics.mean(gaps):>+9.2f}%")
        if len(seeds) > 1:
            print(f"{'  (std)':<16} {statistics.stdev(joints):>10.4f} "
                  f"{statistics.stdev(comps):>10.4f} {statistics.stdev(singles):>10.4f} "
                  f"{statistics.stdev(gaps):>10.2f}")

    # --- Kill criterion: parallel composition gap vs sequential ---
    print(f"\n--- KILL CRITERION: Parallel vs sequential composition gap ---")
    seq_gaps = [all_results["sequential"][s]["gap_pct"] for s in seeds]
    par_gaps = [all_results["parallel"][s]["gap_pct"] for s in seeds]

    seq_mean_gap = statistics.mean(seq_gaps)
    par_mean_gap = statistics.mean(par_gaps)
    seq_median_gap = statistics.median(seq_gaps)
    par_median_gap = statistics.median(par_gaps)

    print(f"  Sequential:  mean gap = {seq_mean_gap:+.2f}%, median gap = {seq_median_gap:+.2f}%")
    print(f"  Parallel:    mean gap = {par_mean_gap:+.2f}%, median gap = {par_median_gap:+.2f}%")

    # "degrades >5%" means parallel gap is >5pp worse than sequential gap
    degradation_mean = par_mean_gap - seq_mean_gap
    degradation_median = par_median_gap - seq_median_gap
    print(f"  Degradation (mean):   {degradation_mean:+.2f}pp")
    print(f"  Degradation (median): {degradation_median:+.2f}pp")

    # Also compute absolute gap comparison
    seq_comp_means = statistics.mean([all_results["sequential"][s]["composed"]["avg"] for s in seeds])
    par_comp_means = statistics.mean([all_results["parallel"][s]["composed"]["avg"] for s in seeds])
    print(f"\n  Mean composed val loss: seq={seq_comp_means:.4f}, par={par_comp_means:.4f}")
    abs_diff_pct = (par_comp_means - seq_comp_means) / seq_comp_means * 100
    print(f"  Absolute composed loss difference: {abs_diff_pct:+.2f}%")

    if max(degradation_mean, degradation_median) > 5.0:
        print(f"\n  ** KILL: parallel degrades >5pp threshold **")
    else:
        print(f"\n  PASS: within 5pp threshold (both mean and median)")

    # Check if parallel is actually BETTER (potential positive finding)
    if par_mean_gap < seq_mean_gap and degradation_mean < -1.0:
        print(f"  NOTE: Parallel block appears BETTER for composition by {-degradation_mean:.2f}pp")

    # --- Per-seed details ---
    print(f"\n--- Per-seed composition gaps ---")
    print(f"{'Seed':>6} {'Seq gap':>10} {'Par gap':>10} {'Diff':>10}")
    for seed in seeds:
        sg = all_results["sequential"][seed]["gap_pct"]
        pg = all_results["parallel"][seed]["gap_pct"]
        print(f"{seed:>6} {sg:>+9.2f}% {pg:>+9.2f}% {pg-sg:>+9.2f}pp")

    # --- Joint training quality comparison ---
    print(f"\n--- Joint training quality (parallel vs sequential) ---")
    print(f"{'Seed':>6} {'Seq joint':>12} {'Par joint':>12} {'Diff':>10}")
    for seed in seeds:
        sj = all_results["sequential"][seed]["joint"]["avg"]
        pj = all_results["parallel"][seed]["joint"]["avg"]
        diff = (pj - sj) / sj * 100
        print(f"{seed:>6} {sj:>12.4f} {pj:>12.4f} {diff:>+9.2f}%")

    return all_results


if __name__ == "__main__":
    run_experiment()
