"""Value norm dynamics experiment under L2-normalized QK composition.

Tests whether value norms stay bounded during composition training with
L2-normalized Q/K, and whether value norm growth correlates with quality
degradation.

Kill criteria:
  1. Value norms grow >10x during composition training with L2 QK norm
  2. Value norm growth correlates >0.5 with composition quality degradation

Protocol (adapted from l2_norm_attention/run_l2_norm_experiment.py):
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps) -- TRACK VALUE NORMS
  Phase 5: Evaluate on per-domain val sets

Value norm tracking points:
  - After pretraining (baseline norms)
  - After composition, before calibration
  - Every 10 steps during calibration
  - After calibration (final norms)

7 seeds for statistical reliability. Growth ratio = max_during_composition / baseline.
"""

import copy
import json
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
N_SEEDS = 7
NORM_TRACK_INTERVAL = 10  # record norms every N calibration steps


def compose_models(model_a, model_b, vocab_size, block_size, base_model=None,
                   layer_types=None):
    """Create composed model by concatenating capsule groups from A and B."""
    n_groups_a = model_a.layers[0].capsule_pool.n_groups
    n_groups_b = model_b.layers[0].capsule_pool.n_groups
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = (model_a.layers[0].capsule_pool.top_k_groups +
                      model_b.layers[0].capsule_pool.top_k_groups)

    kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_groups=composed_groups,
        n_capsules_per_group=CAP["n_capsules_per_group"],
        top_k_groups=composed_top_k,
        n_embd=BASE["n_embd"], n_head=BASE["n_head"], n_layer=BASE["n_layer"],
    )
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    composed = get_model("value_norm_tracking_moe", **kwargs)

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


def record_norms(model, dataset, label=""):
    """Do a forward pass with tracking and return norms dict."""
    model.enable_tracking()
    inputs, _ = dataset.get_batch(BATCH_SIZE)
    logits = model(inputs)
    mx.eval(logits)
    norms = model.get_value_norms()
    model.disable_tracking()
    return norms


def flatten_norms(norms_dict):
    """Flatten {layer: [h0, h1, ...]} to a single list of floats."""
    flat = []
    for layer_idx in sorted(norms_dict.keys()):
        flat.extend(norms_dict[layer_idx])
    return flat


def max_norm(norms_dict):
    """Get max value norm across all layers and heads."""
    flat = flatten_norms(norms_dict)
    return max(flat) if flat else 0.0


def mean_norm(norms_dict):
    """Get mean value norm across all layers and heads."""
    flat = flatten_norms(norms_dict)
    return statistics.mean(flat) if flat else 0.0


def run_single_seed(docs, tokenizer, seed):
    """Run the full composition protocol for one seed with value norm tracking.

    Returns dict with composition gap, value norm trajectories, and growth ratios.
    """
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

    layer_types = ["linear", "linear", "linear", "full"]
    kwargs = dict(vocab_size=V, **CAP, layer_types=layer_types)

    # --- Joint training baseline ---
    model_joint = get_model("value_norm_tracking_moe", **kwargs)
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

    # Record joint model value norms as reference
    joint_norms = record_norms(model_joint, joint_train, "joint_final")

    # --- Pretrain base ---
    base_model = get_model("value_norm_tracking_moe", **kwargs)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Record post-pretrain norms (this is our BASELINE for growth ratio)
    baseline_norms = record_norms(base_model, joint_train, "post_pretrain")

    # --- Fine-tune domain A ---
    model_a = get_model("value_norm_tracking_moe", **kwargs)
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

    # Record post-finetune-A norms
    finetuned_a_norms = record_norms(model_a, train_a, "finetuned_a")

    # --- Fine-tune domain B ---
    model_b = get_model("value_norm_tracking_moe", **kwargs)
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

    # Record post-finetune-B norms
    finetuned_b_norms = record_norms(model_b, train_b, "finetuned_b")

    # --- Compose ---
    composed = compose_models(model_a, model_b, V, BASE["block_size"],
                              base_model=base_model, layer_types=layer_types)

    # Record pre-calibration composed norms
    pre_calib_norms = record_norms(composed, joint_train, "pre_calibration")

    # --- Calibrate router with norm tracking ---
    freeze_except_router(composed)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(composed, ntp_loss)

    calibration_trajectory = []  # list of (step, norms_dict)

    for step in range(1, ROUTER_CALIBRATION_STEPS + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(composed, inputs, targets)
        optimizer.update(composed, grads)
        mx.eval(composed.parameters(), optimizer.state)

        # Track norms at intervals
        if step % NORM_TRACK_INTERVAL == 0 or step == 1:
            norms = record_norms(composed, joint_train, f"calib_step_{step}")
            calibration_trajectory.append({
                "step": step,
                "norms": norms,
                "max_norm": max_norm(norms),
                "mean_norm": mean_norm(norms),
            })

    composed.unfreeze()

    # Record final composed norms
    post_calib_norms = record_norms(composed, joint_train, "post_calibration")

    # --- Evaluate ---
    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2
    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    # --- Compute growth ratios ---
    baseline_max = max_norm(baseline_norms)
    baseline_mean = mean_norm(baseline_norms)

    # Growth during entire composition pipeline
    max_during_composition = max(
        max_norm(pre_calib_norms),
        max_norm(post_calib_norms),
        max(t["max_norm"] for t in calibration_trajectory) if calibration_trajectory else 0.0,
    )
    mean_during_composition = max(
        mean_norm(pre_calib_norms),
        mean_norm(post_calib_norms),
        max(t["mean_norm"] for t in calibration_trajectory) if calibration_trajectory else 0.0,
    )

    max_growth_ratio = max_during_composition / baseline_max if baseline_max > 0 else float("inf")
    mean_growth_ratio = mean_during_composition / baseline_mean if baseline_mean > 0 else float("inf")

    return {
        "seed": seed,
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "gap_pct": gap_pct,
        "baseline_norms": baseline_norms,
        "joint_norms": joint_norms,
        "finetuned_a_norms": finetuned_a_norms,
        "finetuned_b_norms": finetuned_b_norms,
        "pre_calib_norms": pre_calib_norms,
        "post_calib_norms": post_calib_norms,
        "calibration_trajectory": calibration_trajectory,
        "baseline_max": baseline_max,
        "baseline_mean": baseline_mean,
        "max_during_composition": max_during_composition,
        "mean_during_composition": mean_during_composition,
        "max_growth_ratio": max_growth_ratio,
        "mean_growth_ratio": mean_growth_ratio,
    }


def pearson_r(x, y):
    """Compute Pearson correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0
    mx_val = sum(x) / n
    my_val = sum(y) / n
    sx = sum((xi - mx_val) ** 2 for xi in x)
    sy = sum((yi - my_val) ** 2 for yi in y)
    if sx == 0 or sy == 0:
        return 0.0
    sxy = sum((xi - mx_val) * (yi - my_val) for xi, yi in zip(x, y))
    return sxy / (sx * sy) ** 0.5


def run_experiment():
    """Run the full value norm dynamics experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    seeds = list(range(N_SEEDS))
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({i+1}/{N_SEEDS})")
        print(f"{'='*70}")

        result = run_single_seed(docs, tokenizer, seed)
        all_results.append(result)

        print(f"  gap: {result['gap_pct']:+.2f}%")
        print(f"  baseline max norm:  {result['baseline_max']:.4f}")
        print(f"  baseline mean norm: {result['baseline_mean']:.4f}")
        print(f"  max during composition:  {result['max_during_composition']:.4f}")
        print(f"  mean during composition: {result['mean_during_composition']:.4f}")
        print(f"  max growth ratio:  {result['max_growth_ratio']:.2f}x")
        print(f"  mean growth ratio: {result['mean_growth_ratio']:.2f}x")

    elapsed = time.time() - t0

    # --- Analysis ---
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({N_SEEDS} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    gaps = [r["gap_pct"] for r in all_results]
    max_growths = [r["max_growth_ratio"] for r in all_results]
    mean_growths = [r["mean_growth_ratio"] for r in all_results]

    print(f"\nComposition gap:")
    print(f"  mean:   {statistics.mean(gaps):+.2f}%")
    print(f"  median: {statistics.median(gaps):+.2f}%")
    print(f"  std:    {statistics.stdev(gaps):.2f}%")
    print(f"  range:  [{min(gaps):+.2f}%, {max(gaps):+.2f}%]")

    print(f"\nMax growth ratio (max_during_composition / baseline_max):")
    print(f"  mean:   {statistics.mean(max_growths):.2f}x")
    print(f"  median: {statistics.median(max_growths):.2f}x")
    print(f"  max:    {max(max_growths):.2f}x")
    print(f"  range:  [{min(max_growths):.2f}x, {max(max_growths):.2f}x]")

    print(f"\nMean growth ratio (mean_during_composition / baseline_mean):")
    print(f"  mean:   {statistics.mean(mean_growths):.2f}x")
    print(f"  median: {statistics.median(mean_growths):.2f}x")
    print(f"  max:    {max(mean_growths):.2f}x")
    print(f"  range:  [{min(mean_growths):.2f}x, {max(mean_growths):.2f}x]")

    # --- Kill Criterion 1: Growth ratio ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 1: Value norm growth >10x")
    print(f"{'='*70}")

    any_over_10x = any(g > 10.0 for g in max_growths)
    worst_growth = max(max_growths)
    if any_over_10x:
        seeds_over = [r["seed"] for r in all_results if r["max_growth_ratio"] > 10.0]
        print(f"  KILL: {len(seeds_over)} seeds show >10x growth: {seeds_over}")
        print(f"  Worst growth: {worst_growth:.2f}x")
        k1_pass = False
    else:
        print(f"  PASS: No seed shows >10x growth (worst: {worst_growth:.2f}x)")
        k1_pass = True

    # --- Kill Criterion 2: Correlation with quality degradation ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 2: Growth-quality correlation >0.5")
    print(f"{'='*70}")

    r_max = pearson_r(max_growths, gaps)
    r_mean = pearson_r(mean_growths, gaps)

    print(f"  Pearson r(max_growth, gap):  {r_max:.3f}")
    print(f"  Pearson r(mean_growth, gap): {r_mean:.3f}")

    if abs(r_max) > 0.5 or abs(r_mean) > 0.5:
        print(f"  KILL: Correlation exceeds 0.5 threshold")
        k2_pass = False
    else:
        print(f"  PASS: No correlation exceeds 0.5 threshold")
        k2_pass = True

    # --- Per-layer analysis ---
    print(f"\n{'='*70}")
    print("PER-LAYER VALUE NORM ANALYSIS")
    print(f"{'='*70}")

    for layer_idx in range(3):  # 3 linear layers
        layer_baselines = []
        layer_peaks = []
        for r in all_results:
            if layer_idx in r["baseline_norms"]:
                layer_baselines.append(statistics.mean(r["baseline_norms"][layer_idx]))
            if layer_idx in r["post_calib_norms"]:
                layer_peaks.append(statistics.mean(r["post_calib_norms"][layer_idx]))

        if layer_baselines and layer_peaks:
            base_avg = statistics.mean(layer_baselines)
            peak_avg = statistics.mean(layer_peaks)
            ratio = peak_avg / base_avg if base_avg > 0 else float("inf")
            print(f"  Layer {layer_idx} (linear): "
                  f"baseline={base_avg:.4f}, post-calib={peak_avg:.4f}, "
                  f"ratio={ratio:.2f}x")

    # --- Norm trajectory (averaged over seeds) ---
    print(f"\n{'='*70}")
    print("VALUE NORM TRAJECTORY DURING CALIBRATION (mean across seeds)")
    print(f"{'='*70}")
    print(f"{'Step':>6} {'Max Norm':>10} {'Mean Norm':>10}")
    print("-" * 30)

    # Collect trajectories across seeds
    n_points = len(all_results[0]["calibration_trajectory"])
    for pt_idx in range(n_points):
        step = all_results[0]["calibration_trajectory"][pt_idx]["step"]
        max_norms_at_step = [r["calibration_trajectory"][pt_idx]["max_norm"]
                            for r in all_results]
        mean_norms_at_step = [r["calibration_trajectory"][pt_idx]["mean_norm"]
                             for r in all_results]
        print(f"{step:>6} {statistics.mean(max_norms_at_step):>10.4f} "
              f"{statistics.mean(mean_norms_at_step):>10.4f}")

    # --- Per-seed summary ---
    print(f"\n{'='*70}")
    print("PER-SEED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Seed':>6} {'Gap':>8} {'Max Growth':>12} {'Mean Growth':>12} "
          f"{'Base Max':>10} {'Comp Max':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['seed']:>6} {r['gap_pct']:>+7.2f}% "
              f"{r['max_growth_ratio']:>11.2f}x "
              f"{r['mean_growth_ratio']:>11.2f}x "
              f"{r['baseline_max']:>10.4f} "
              f"{r['max_during_composition']:>10.4f}")

    # --- Overall verdict ---
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    if k1_pass and k2_pass:
        print("  PASS: Value norms stay bounded during composition")
        print("  The L2 QK normalization state boundedness argument is empirically confirmed:")
        print(f"    - Max growth ratio across seeds: {worst_growth:.2f}x (threshold: 10x)")
        print(f"    - Growth-quality correlations: r_max={r_max:.3f}, r_mean={r_mean:.3f} (threshold: 0.5)")
    elif not k1_pass:
        print("  KILL (criterion 1): Value norms grow >10x during composition")
    elif not k2_pass:
        print("  KILL (criterion 2): Value norm growth correlates with quality degradation")

    # --- Save results ---
    # Convert norms dicts for JSON serialization (int keys -> str keys)
    def serialize_norms(nd):
        return {str(k): v for k, v in nd.items()}

    output = {
        "n_seeds": N_SEEDS,
        "elapsed_seconds": elapsed,
        "kill_criteria": {
            "k1_max_growth_threshold": 10.0,
            "k1_worst_growth": worst_growth,
            "k1_pass": k1_pass,
            "k2_correlation_threshold": 0.5,
            "k2_r_max_growth_gap": r_max,
            "k2_r_mean_growth_gap": r_mean,
            "k2_pass": k2_pass,
            "overall_pass": k1_pass and k2_pass,
        },
        "summary": {
            "gap_mean": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_std": statistics.stdev(gaps),
            "max_growth_mean": statistics.mean(max_growths),
            "max_growth_median": statistics.median(max_growths),
            "max_growth_max": max(max_growths),
            "mean_growth_mean": statistics.mean(mean_growths),
            "mean_growth_median": statistics.median(mean_growths),
            "mean_growth_max": max(mean_growths),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "gap_pct": r["gap_pct"],
                "joint_avg": r["joint_avg"],
                "composed_avg": r["composed_avg"],
                "baseline_max": r["baseline_max"],
                "baseline_mean": r["baseline_mean"],
                "max_during_composition": r["max_during_composition"],
                "mean_during_composition": r["mean_during_composition"],
                "max_growth_ratio": r["max_growth_ratio"],
                "mean_growth_ratio": r["mean_growth_ratio"],
                "baseline_norms": serialize_norms(r["baseline_norms"]),
                "joint_norms": serialize_norms(r["joint_norms"]),
                "pre_calib_norms": serialize_norms(r["pre_calib_norms"]),
                "post_calib_norms": serialize_norms(r["post_calib_norms"]),
                "calibration_trajectory": [
                    {
                        "step": t["step"],
                        "max_norm": t["max_norm"],
                        "mean_norm": t["mean_norm"],
                        "norms": serialize_norms(t["norms"]),
                    }
                    for t in r["calibration_trajectory"]
                ],
            }
            for r in all_results
        ],
    }

    results_path = "experiments/models/value_norm_dynamics/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()
