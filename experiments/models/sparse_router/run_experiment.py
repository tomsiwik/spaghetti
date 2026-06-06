"""Sparse routing experiment — top-k sweep in capsule MoE composition.

Tests whether top-1 group selection matches top-2 quality at half active compute.
Uses the validated shared-base composition protocol from capsule_moe.

Experimental design:
  1. Pretrain shared base on all data (300 steps)
  2. Fine-tune capsule groups per domain (300 steps each, attention frozen)
  3. Compose 8 groups (4+4)
  4. For each k in {1,2,4,8}: fresh router, calibrate 100 steps, evaluate
  5. Uniform baseline: no calibration (random router) at each k
  6. Analyze: router entropy, concentration, domain alignment
  7. Compare to joint training baseline

Kill thresholds (from MATH.md Section 10):
  - Top-1 degrades >10% vs top-2: KILL
  - Learned top-1 loses to uniform top-1: KILL
  - Router entropy at k=1 > 0.9 * H_max: KILL
  - Top-1 degrades >15% vs joint: KILL
"""

import math
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss


# Standard micro config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CAL_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3
TOP_K_VALUES = [1, 2, 4, 8]


def build_domain_datasets(tokenizer, docs, block_size=32, seed=42):
    splits = domain_split(docs)
    datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        datasets[name] = (
            CharDataset(dtrain, tokenizer, block_size),
            CharDataset(dval, tokenizer, block_size),
        )
    return datasets


def compose_sparse_model(base_model, groups_a, groups_b,
                         vocab_size, block_size, top_k=2):
    """Create composed SparseRouterGPT from shared base + domain groups."""
    n_groups_a = len(groups_a[0])
    n_groups_b = len(groups_b[0])
    composed_groups = n_groups_a + n_groups_b

    composed = get_model("sparse_router",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_groups,
                         n_capsules_per_group=CAP["n_capsules_per_group"],
                         top_k_groups=top_k,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_base = base_model.layers[l_idx]

        layer_c.attn.wq.weight = layer_base.attn.wq.weight
        layer_c.attn.wk.weight = layer_base.attn.wk.weight
        layer_c.attn.wv.weight = layer_base.attn.wv.weight
        layer_c.attn.wo.weight = layer_base.attn.wo.weight

        pool_c = layer_c.capsule_pool
        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = groups_a[l_idx][g].A.weight
            pool_c.groups[g].B.weight = groups_a[l_idx][g].B.weight
        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = groups_b[l_idx][g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = groups_b[l_idx][g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_ds_a, train_ds_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=42):
    """Train only router on mixed-domain data. Returns final loss."""
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
    return loss.item()


def compute_domain_alignment(model, val_a, val_b,
                             groups_per_domain=4, n_batches=5, seed=42):
    """Fraction of tokens where top-1 group belongs to correct domain."""
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    correct = 0
    total = 0

    for _ in range(n_batches):
        # Domain A tokens → top-1 should be in groups 0..gpd-1
        inputs_a, _ = val_a.get_batch(BATCH_SIZE, rng_a)
        model(inputs_a)
        for layer in model.layers:
            probs = layer.capsule_pool._gate_probs
            mx.eval(probs)
            top1 = mx.argmax(probs, axis=-1)
            mx.eval(top1)
            correct += mx.sum((top1 < groups_per_domain).astype(mx.float32)).item()
            total += top1.size

        # Domain B tokens → top-1 should be in groups gpd..2*gpd-1
        inputs_b, _ = val_b.get_batch(BATCH_SIZE, rng_b)
        model(inputs_b)
        for layer in model.layers:
            probs = layer.capsule_pool._gate_probs
            mx.eval(probs)
            top1 = mx.argmax(probs, axis=-1)
            mx.eval(top1)
            correct += mx.sum((top1 >= groups_per_domain).astype(mx.float32)).item()
            total += top1.size

    return correct / total if total > 0 else 0.0


def run_experiment(seed=42):
    """Run the full sparse routing experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"SPARSE ROUTING EXPERIMENT (seed={seed})")
    print(f"{'='*70}")
    t_start = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    domain_datasets = build_domain_datasets(tokenizer, docs, seed=seed)
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]
    all_train, all_val = train_val_split(docs, seed=seed)

    results = {}

    # --- Joint training baseline ---
    print("\n--- Joint training baseline ---")
    model_joint = get_model("capsule_moe", vocab_size=V, **CAP)
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

    # --- Shared base pretraining ---
    print("\n--- Shared base pretraining (300 steps, all data) ---")
    base_model = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(base_model.parameters())
    joint_ds = CharDataset(all_train, tokenizer, BASE["block_size"])
    train(base_model, joint_ds, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    # --- Domain A fine-tuning ---
    print("\n--- Fine-tuning capsule groups on domain A ---")
    model_a = get_model("capsule_moe", vocab_size=V, **CAP)
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

    # --- Domain B fine-tuning ---
    print("\n--- Fine-tuning capsule groups on domain B ---")
    model_b = get_model("capsule_moe", vocab_size=V, **CAP)
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

    # Extract domain groups
    groups_a = [layer.capsule_pool.groups for layer in model_a.layers]
    groups_b = [layer.capsule_pool.groups for layer in model_b.layers]

    # --- Top-k sweep with learned routing ---
    print("\n--- Top-k sweep (learned routing) ---")
    learned_results = {}
    router_analysis = {}

    for k in TOP_K_VALUES:
        print(f"\n  k={k}:")
        composed = compose_sparse_model(base_model, groups_a, groups_b,
                                        V, BASE["block_size"], top_k=k)
        calibrate_router(composed, train_a, train_b, seed=seed)

        val_a_loss = evaluate(composed, val_a, BATCH_SIZE)
        val_b_loss = evaluate(composed, val_b, BATCH_SIZE)
        avg_loss = (val_a_loss + val_b_loss) / 2
        learned_results[k] = {"a_m": val_a_loss, "n_z": val_b_loss, "avg": avg_loss}

        # Router analysis: run forward on val data to populate stats
        rng_eval = random.Random(0)
        inp_a, _ = val_a.get_batch(BATCH_SIZE, rng_eval)
        composed(inp_a)
        stats = composed.router_stats()

        alignment = compute_domain_alignment(composed, val_a, val_b)

        router_analysis[k] = {
            "entropy": statistics.mean(stats["entropy"]),
            "entropy_ratio": statistics.mean(stats["entropy_ratio"]),
            "concentration_1": statistics.mean(stats["concentration_1"]),
            "domain_alignment": alignment,
            "group_freqs": stats["group_freqs"],
        }

        print(f"    val: a_m={val_a_loss:.4f}, n_z={val_b_loss:.4f}, avg={avg_loss:.4f}")
        print(f"    H={router_analysis[k]['entropy']:.3f}, "
              f"H/H_max={router_analysis[k]['entropy_ratio']:.3f}, "
              f"C_1={router_analysis[k]['concentration_1']:.3f}, "
              f"domain_align={alignment:.1%}")

    results["learned"] = learned_results

    # --- Top-k sweep with random routing (no calibration) ---
    print("\n--- Top-k sweep (random routing, no calibration) ---")
    uniform_results = {}
    for k in TOP_K_VALUES:
        composed = compose_sparse_model(base_model, groups_a, groups_b,
                                        V, BASE["block_size"], top_k=k)
        # NO calibration — random router weights
        val_a_loss = evaluate(composed, val_a, BATCH_SIZE)
        val_b_loss = evaluate(composed, val_b, BATCH_SIZE)
        avg_loss = (val_a_loss + val_b_loss) / 2
        uniform_results[k] = {"a_m": val_a_loss, "n_z": val_b_loss, "avg": avg_loss}
        print(f"  k={k}: a_m={val_a_loss:.4f}, n_z={val_b_loss:.4f}, avg={avg_loss:.4f}")

    results["uniform"] = uniform_results

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Setting':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} "
          f"{'vs joint':>10} {'vs k=2':>10}")
    print("-" * 75)

    print(f"{'joint':<25} {results['joint']['a_m']:>8.4f} "
          f"{results['joint']['n_z']:>8.4f} {results['joint']['avg']:>8.4f} "
          f"{'baseline':>10}")

    for k in TOP_K_VALUES:
        lr_k = learned_results[k]
        vs_joint = (lr_k["avg"] - joint_avg) / joint_avg * 100
        vs_k2 = ((lr_k["avg"] - learned_results[2]["avg"])
                 / learned_results[2]["avg"] * 100) if k != 2 else 0
        label = f"learned k={k}"
        print(f"{label:<25} {lr_k['a_m']:>8.4f} {lr_k['n_z']:>8.4f} "
              f"{lr_k['avg']:>8.4f} {vs_joint:>+9.1f}% {vs_k2:>+9.1f}%")

    print("-" * 75)
    for k in TOP_K_VALUES:
        ur_k = uniform_results[k]
        vs_joint = (ur_k["avg"] - joint_avg) / joint_avg * 100
        label = f"uniform k={k}"
        print(f"{label:<25} {ur_k['a_m']:>8.4f} {ur_k['n_z']:>8.4f} "
              f"{ur_k['avg']:>8.4f} {vs_joint:>+9.1f}%")

    # Router analysis table
    h_max = math.log(8)
    print(f"\nRouter Analysis (G=8, H_max={h_max:.3f}):")
    print(f"{'k':>3} {'H(p)':>8} {'H/H_max':>8} {'C_1':>8} {'Domain%':>8}")
    print("-" * 38)
    for k in TOP_K_VALUES:
        ra = router_analysis[k]
        print(f"{k:>3} {ra['entropy']:>8.3f} {ra['entropy_ratio']:>8.3f} "
              f"{ra['concentration_1']:>8.3f} {ra['domain_alignment']:>7.1%}")

    # Kill threshold checks
    print(f"\n--- Kill threshold checks ---")
    k1_vs_k2 = ((learned_results[1]["avg"] - learned_results[2]["avg"])
                / learned_results[2]["avg"] * 100)
    k1_vs_joint = (learned_results[1]["avg"] - joint_avg) / joint_avg * 100
    k1_entropy_ratio = router_analysis[1]["entropy_ratio"]
    learned_k1 = learned_results[1]["avg"]
    uniform_k1 = uniform_results[1]["avg"]

    kills = []

    if k1_vs_k2 > 10:
        kills.append(f"top-1 vs top-2 = {k1_vs_k2:+.1f}% (>10% threshold)")
    elif k1_vs_k2 > 5:
        print(f"  MARGINAL: top-1 vs top-2 = {k1_vs_k2:+.1f}% (5-10% range)")
    else:
        print(f"  PASS: top-1 vs top-2 = {k1_vs_k2:+.1f}% (<5%)")

    if learned_k1 >= uniform_k1:
        kills.append(f"learned k=1 ({learned_k1:.4f}) >= "
                     f"uniform k=1 ({uniform_k1:.4f})")
    else:
        margin = (uniform_k1 - learned_k1) / uniform_k1 * 100
        print(f"  PASS: learned k=1 beats uniform by {margin:.1f}%")

    if k1_entropy_ratio > 0.9:
        kills.append(f"router entropy ratio = {k1_entropy_ratio:.3f} (>0.9)")
    else:
        print(f"  PASS: router entropy ratio = {k1_entropy_ratio:.3f} (<0.9)")

    if k1_vs_joint > 15:
        kills.append(f"top-1 vs joint = {k1_vs_joint:+.1f}% (>15% threshold)")
    elif k1_vs_joint > 10:
        print(f"  MARGINAL: top-1 vs joint = {k1_vs_joint:+.1f}% (10-15% range)")
    else:
        print(f"  PASS: top-1 vs joint = {k1_vs_joint:+.1f}% (<10%)")

    for kill in kills:
        print(f"  ** KILL: {kill} **")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "results": results,
        "router_analysis": router_analysis,
        "kill_checks": {
            "k1_vs_k2_pct": k1_vs_k2,
            "k1_vs_joint_pct": k1_vs_joint,
            "entropy_ratio_k1": k1_entropy_ratio,
            "learned_k1_loss": learned_k1,
            "uniform_k1_loss": uniform_k1,
            "n_kills": len(kills),
        },
    }


def run_multiseed(seeds=(42, 123, 7)):
    """Run across multiple seeds and aggregate."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_experiment(seed)

    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    # Aggregate
    joint_means = [all_results[s]["results"]["joint"]["avg"] for s in seeds]
    joint_mean = statistics.mean(joint_means)

    k2_means = [all_results[s]["results"]["learned"][2]["avg"] for s in seeds]
    k2_mean = statistics.mean(k2_means)

    print(f"\n{'Setting':<25} {'mean':>8} {'std':>8} "
          f"{'vs joint':>10} {'vs k=2':>10}")
    print("-" * 65)

    print(f"{'joint':<25} {joint_mean:>8.4f} "
          f"{statistics.stdev(joint_means):>8.4f} {'baseline':>10}")

    for k in TOP_K_VALUES:
        avgs = [all_results[s]["results"]["learned"][k]["avg"] for s in seeds]
        mean_v = statistics.mean(avgs)
        std_v = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = (mean_v - joint_mean) / joint_mean * 100
        vs_k2 = (mean_v - k2_mean) / k2_mean * 100 if k != 2 else 0
        label = f"learned k={k}"
        print(f"{label:<25} {mean_v:>8.4f} {std_v:>8.4f} "
              f"{vs_joint:>+9.1f}% {vs_k2:>+9.1f}%")

    print("-" * 65)
    for k in TOP_K_VALUES:
        avgs = [all_results[s]["results"]["uniform"][k]["avg"] for s in seeds]
        mean_v = statistics.mean(avgs)
        std_v = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = (mean_v - joint_mean) / joint_mean * 100
        label = f"uniform k={k}"
        print(f"{label:<25} {mean_v:>8.4f} {std_v:>8.4f} {vs_joint:>+9.1f}%")

    # Router analysis aggregate
    h_max = math.log(8)
    print(f"\nRouter Analysis (mean across seeds, H_max={h_max:.3f}):")
    print(f"{'k':>3} {'H(p)':>8} {'H/H_max':>8} {'C_1':>8} {'Domain%':>8}")
    print("-" * 38)
    for k in TOP_K_VALUES:
        h = statistics.mean(
            [all_results[s]["router_analysis"][k]["entropy"] for s in seeds])
        hr = statistics.mean(
            [all_results[s]["router_analysis"][k]["entropy_ratio"] for s in seeds])
        c1 = statistics.mean(
            [all_results[s]["router_analysis"][k]["concentration_1"] for s in seeds])
        da = statistics.mean(
            [all_results[s]["router_analysis"][k]["domain_alignment"] for s in seeds])
        print(f"{k:>3} {h:>8.3f} {hr:>8.3f} {c1:>8.3f} {da:>7.1%}")

    # Aggregate kill checks
    print(f"\n--- Aggregate kill threshold checks ---")
    k1_vs_k2_vals = [all_results[s]["kill_checks"]["k1_vs_k2_pct"] for s in seeds]
    k1_vs_joint_vals = [all_results[s]["kill_checks"]["k1_vs_joint_pct"]
                        for s in seeds]
    entropy_ratios = [all_results[s]["kill_checks"]["entropy_ratio_k1"]
                      for s in seeds]
    learned_k1_vals = [all_results[s]["kill_checks"]["learned_k1_loss"]
                       for s in seeds]
    uniform_k1_vals = [all_results[s]["kill_checks"]["uniform_k1_loss"]
                       for s in seeds]

    mean_k1_vs_k2 = statistics.mean(k1_vs_k2_vals)
    mean_k1_vs_joint = statistics.mean(k1_vs_joint_vals)
    mean_entropy_ratio = statistics.mean(entropy_ratios)
    mean_learned_k1 = statistics.mean(learned_k1_vals)
    mean_uniform_k1 = statistics.mean(uniform_k1_vals)

    kills = []

    if mean_k1_vs_k2 > 10:
        kills.append(f"top-1 vs top-2 = {mean_k1_vs_k2:+.1f}% (>10%)")
    else:
        status = "PASS" if mean_k1_vs_k2 <= 5 else "MARGINAL"
        print(f"  {status}: top-1 vs top-2 = {mean_k1_vs_k2:+.1f}% "
              f"+/- {statistics.stdev(k1_vs_k2_vals):.1f}%")

    if mean_learned_k1 >= mean_uniform_k1:
        kills.append(f"learned k=1 ({mean_learned_k1:.4f}) >= "
                     f"uniform k=1 ({mean_uniform_k1:.4f})")
    else:
        if len(seeds) > 1:
            diffs = [all_results[s]["kill_checks"]["uniform_k1_loss"] -
                     all_results[s]["kill_checks"]["learned_k1_loss"]
                     for s in seeds]
            se = statistics.stdev(diffs) / len(seeds)**0.5
            print(f"  PASS: learned k=1 beats uniform by "
                  f"{(mean_uniform_k1 - mean_learned_k1):.4f} +/- {se:.4f}")
        else:
            print(f"  PASS: learned k=1 beats uniform")

    if mean_entropy_ratio > 0.9:
        kills.append(f"entropy ratio = {mean_entropy_ratio:.3f} (>0.9)")
    else:
        print(f"  PASS: entropy ratio = {mean_entropy_ratio:.3f}")

    if mean_k1_vs_joint > 15:
        kills.append(f"top-1 vs joint = {mean_k1_vs_joint:+.1f}% (>15%)")
    else:
        status = "PASS" if mean_k1_vs_joint <= 10 else "MARGINAL"
        print(f"  {status}: top-1 vs joint = {mean_k1_vs_joint:+.1f}%")

    for kill in kills:
        print(f"  ** KILL: {kill} **")

    # Final verdict
    print(f"\n--- Final verdict ---")
    if len(kills) == 0 and mean_k1_vs_k2 <= 5:
        print("  PASS: Top-1 matches top-2 within 5%. Sparse routing validated.")
    elif len(kills) == 0:
        print("  MARGINAL: No kill thresholds exceeded, but "
              "top-1 degrades 5-10% vs top-2.")
    else:
        print(f"  KILL: {len(kills)} threshold(s) exceeded.")

    return all_results


if __name__ == "__main__":
    run_multiseed()
