"""Dead Capsule Pruning experiment.

Tests whether pruning dead capsules from composed ReLU Router models
preserves quality while achieving significant parameter reduction.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose by concatenating A and B weight matrices from both domains
  4. Profile: run calibration data through composed model, measure activation freq
  5. Prune: remove capsules that fire less than threshold
  6. Evaluate: quality impact, parameter savings, interaction with calibration

Sweep: threshold in {0.0, 0.001, 0.005, 0.01, 0.05, 0.10}

Key questions:
  - Does pruning the ~60% dead capsules preserve quality?
  - Does pruning IMPROVE quality (by reducing interference)?
  - What is the optimal pruning threshold?
  - How does pruning interact with calibration?
    (prune-then-calibrate vs calibrate-then-prune)
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..relu_router.test_composition import (
    compose_relu_models, weight_average_relu_models,
    _make_relu_model, _freeze_attention, _eval_domains,
    full_capsule_calibrate,
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE,
    CALIBRATION_STEPS, BATCH_SIZE, LR,
)
from .dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
    prune_composed_model,
)


def count_params(model) -> int:
    """Count total parameters in a model."""
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def count_capsule_params(model) -> int:
    """Count only capsule pool parameters."""
    total = 0
    for layer in model.layers:
        pool = layer.capsule_pool
        total += pool.A.weight.size + pool.B.weight.size
    return total


def run_pruning_experiment(seed=42):
    """Run the full dead capsule pruning experiment.

    Returns:
        results: dict of method -> {domain: loss, "avg": avg_loss}
        prune_stats: dict of method -> pruning statistics
        param_counts: dict of method -> total params
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    results = {}
    prune_stats_all = {}
    param_counts = {}
    domain_names = list(domain_datasets.keys())

    # ============================================================
    # 1. Joint training baseline (upper bound)
    # ============================================================
    print("  [1/9] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)
    param_counts["joint"] = count_params(model_joint)

    # ============================================================
    # 2. Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/9] Pretrain base + fine-tune per domain...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        domain_models.append(model_d)

    # ============================================================
    # 3. Unmerged concatenation (zero-shot baseline)
    # ============================================================
    print("  [3/9] Unmerged concatenation (zero-shot)...")
    composed = compose_relu_models(base, domain_models)
    results["concat_zero_shot"] = _eval_domains(composed, domain_datasets)
    param_counts["concat"] = count_params(composed)

    # ============================================================
    # 4. Weight averaging baseline
    # ============================================================
    print("  [4/9] Weight averaging baseline...")
    averaged = weight_average_relu_models(base, domain_models)
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)
    param_counts["weight_avg"] = count_params(averaged)

    # ============================================================
    # 5. Pruning sweep: different thresholds
    # ============================================================
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10]

    for tau in thresholds:
        key = f"prune_t{tau:.3f}"
        print(f"  [5/9] Pruning threshold={tau:.3f}...")
        model_pruned = copy.deepcopy(composed)

        stats = prune_composed_model(
            model_pruned, joint_val, threshold=tau,
            n_batches=20, batch_size=32, seed=seed,
            verbose=True,
        )
        mx.eval(model_pruned.parameters())

        results[key] = _eval_domains(model_pruned, domain_datasets)
        prune_stats_all[key] = stats
        param_counts[key] = count_params(model_pruned)

    # ============================================================
    # 6. Prune-then-calibrate (prune dead, then fine-tune capsules)
    # ============================================================
    print("  [6/9] Prune (threshold=0.0) then calibrate...")
    model_prune_cal = copy.deepcopy(composed)
    prune_stats_pc = prune_composed_model(
        model_prune_cal, joint_val, threshold=0.0,
        n_batches=20, batch_size=32, seed=seed,
        verbose=True,
    )
    mx.eval(model_prune_cal.parameters())

    # Calibrate: train capsule weights on joint data
    _freeze_attention(model_prune_cal)
    train(model_prune_cal, joint_train, steps=CALIBRATION_STEPS,
          batch_size=BATCH_SIZE, lr=LR * 0.1, seed=seed, log_every=9999)
    model_prune_cal.unfreeze()
    results["prune_then_cal"] = _eval_domains(model_prune_cal, domain_datasets)
    prune_stats_all["prune_then_cal"] = prune_stats_pc
    param_counts["prune_then_cal"] = count_params(model_prune_cal)

    # ============================================================
    # 7. Calibrate-then-prune (calibrate first, then prune dead)
    # ============================================================
    print("  [7/9] Calibrate then prune (threshold=0.0)...")
    model_cal_prune = copy.deepcopy(composed)

    # First calibrate
    _freeze_attention(model_cal_prune)
    train(model_cal_prune, joint_train, steps=CALIBRATION_STEPS,
          batch_size=BATCH_SIZE, lr=LR * 0.1, seed=seed, log_every=9999)
    model_cal_prune.unfreeze()
    results["cal_no_prune"] = _eval_domains(model_cal_prune, domain_datasets)

    # Then prune the calibrated model
    model_cal_then_prune = copy.deepcopy(model_cal_prune)
    prune_stats_cp = prune_composed_model(
        model_cal_then_prune, joint_val, threshold=0.0,
        n_batches=20, batch_size=32, seed=seed,
        verbose=True,
    )
    mx.eval(model_cal_then_prune.parameters())
    results["cal_then_prune"] = _eval_domains(model_cal_then_prune, domain_datasets)
    prune_stats_all["cal_then_prune"] = prune_stats_cp
    param_counts["cal_then_prune"] = count_params(model_cal_then_prune)
    param_counts["cal_no_prune"] = count_params(model_cal_prune)

    # ============================================================
    # 8. Aggressive prune-then-calibrate (threshold=0.01)
    # ============================================================
    print("  [8/9] Aggressive prune (threshold=0.01) then calibrate...")
    model_agg = copy.deepcopy(composed)
    prune_stats_agg = prune_composed_model(
        model_agg, joint_val, threshold=0.01,
        n_batches=20, batch_size=32, seed=seed,
        verbose=True,
    )
    mx.eval(model_agg.parameters())

    _freeze_attention(model_agg)
    train(model_agg, joint_train, steps=CALIBRATION_STEPS,
          batch_size=BATCH_SIZE, lr=LR * 0.1, seed=seed, log_every=9999)
    model_agg.unfreeze()
    results["agg_prune_then_cal"] = _eval_domains(model_agg, domain_datasets)
    prune_stats_all["agg_prune_then_cal"] = prune_stats_agg
    param_counts["agg_prune_then_cal"] = count_params(model_agg)

    # ============================================================
    # 9. Per-domain profiling: prune using only each domain's data
    # ============================================================
    print("  [9/9] Per-domain profiling...")
    # Profile on domain A data
    model_domain_a = copy.deepcopy(composed)
    freqs_a = profile_activations(model_domain_a, domain_datasets[domain_names[0]][1],
                                  n_batches=10, batch_size=32, seed=seed)
    # Profile on domain B data
    model_domain_b = copy.deepcopy(composed)
    freqs_b = profile_activations(model_domain_b, domain_datasets[domain_names[1]][1],
                                  n_batches=10, batch_size=32, seed=seed)

    # Per-domain dead counts (for analysis)
    per_domain_stats = {"domain_a": {}, "domain_b": {}, "union": {}}
    for l_idx in range(len(composed.layers)):
        fa = freqs_a[l_idx].tolist()
        fb = freqs_b[l_idx].tolist()
        dead_a = sum(1 for f in fa if f == 0)
        dead_b = sum(1 for f in fb if f == 0)
        dead_both = sum(1 for a, b in zip(fa, fb) if a == 0 and b == 0)
        dead_either = sum(1 for a, b in zip(fa, fb) if a == 0 or b == 0)
        per_domain_stats["domain_a"][l_idx] = {"n_dead": dead_a, "total": len(fa)}
        per_domain_stats["domain_b"][l_idx] = {"n_dead": dead_b, "total": len(fb)}
        per_domain_stats["union"][l_idx] = {
            "dead_both": dead_both,
            "dead_either": dead_either,
            "total": len(fa),
        }
    prune_stats_all["per_domain"] = per_domain_stats

    return results, prune_stats_all, param_counts


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []
    all_stats = []
    all_params = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r, s, p = run_pruning_experiment(seed=seed)
        all_results.append(r)
        all_stats.append(s)
        all_params.append(p)

        # Per-seed summary
        for method, vals in r.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains)
            print(f"  {method:<25} avg={vals['avg']:.4f} ({detail})")

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])
    concat_mean = statistics.mean([r["concat_zero_shot"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs joint':>10} {'vs concat':>10}")
    print("  " + "-" * 70)
    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_concat:>+9.1f}%")

    # ============================================================
    # Pruning statistics
    # ============================================================
    print(f"\n  Pruning Statistics (3-seed mean)")
    print("  " + "-" * 70)

    for key in sorted(all_stats[0].keys()):
        if key == "per_domain":
            continue
        if "pct_pruned" in all_stats[0][key]:
            pct_pruned = [s[key]["pct_pruned"] for s in all_stats]
            mean_pct = statistics.mean(pct_pruned)
            std_pct = statistics.stdev(pct_pruned) if len(pct_pruned) > 1 else 0
            print(f"  {key:<25} pruned={mean_pct:.1f}% (std={std_pct:.1f}%)")

            # Per-layer breakdown
            for l in range(4):
                layer_pruned = [s[key]["per_layer"][l]["pct_pruned"] for s in all_stats]
                mean_l = statistics.mean(layer_pruned)
                print(f"    layer {l}: {mean_l:.1f}% pruned")

    # ============================================================
    # Per-domain profiling analysis
    # ============================================================
    print(f"\n  Per-Domain Dead Capsule Analysis (3-seed mean)")
    print("  " + "-" * 70)

    for l in range(4):
        dead_a = [s["per_domain"]["domain_a"][l]["n_dead"] for s in all_stats]
        dead_b = [s["per_domain"]["domain_b"][l]["n_dead"] for s in all_stats]
        dead_both = [s["per_domain"]["union"][l]["dead_both"] for s in all_stats]
        dead_either = [s["per_domain"]["union"][l]["dead_either"] for s in all_stats]
        total = all_stats[0]["per_domain"]["domain_a"][l]["total"]

        print(f"  Layer {l} (P={total}):")
        print(f"    Dead on domain A only:    {statistics.mean(dead_a):.0f} ({statistics.mean(dead_a)/total*100:.1f}%)")
        print(f"    Dead on domain B only:    {statistics.mean(dead_b):.0f} ({statistics.mean(dead_b)/total*100:.1f}%)")
        print(f"    Dead on BOTH domains:     {statistics.mean(dead_both):.0f} ({statistics.mean(dead_both)/total*100:.1f}%)")
        print(f"    Dead on EITHER domain:    {statistics.mean(dead_either):.0f} ({statistics.mean(dead_either)/total*100:.1f}%)")

    # ============================================================
    # Parameter counts
    # ============================================================
    print(f"\n  Parameter Counts (seed 42)")
    print("  " + "-" * 50)
    p = all_params[0]
    for key in sorted(p.keys()):
        savings = (1 - p[key] / p["concat"]) * 100
        print(f"  {key:<25} {p[key]:>8,} params ({savings:>+.1f}% vs concat)")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n  Kill Threshold Analysis")
    print("  " + "-" * 70)

    # Kill 1: Pruning >50% of capsules degrades quality by >2% vs unpruned
    prune_t0_avgs = [r["prune_t0.000"]["avg"] for r in all_results]
    prune_t0_mean = statistics.mean(prune_t0_avgs)
    vs_concat_pct = ((prune_t0_mean - concat_mean) / concat_mean) * 100
    kill1 = abs(vs_concat_pct) > 2.0
    print(f"  Kill 1: Prune(t=0) vs concat = {vs_concat_pct:+.2f}% "
          f"({'KILL' if kill1 else 'PASS'}, threshold=2%)")

    # Kill 2: Pruning doesn't achieve at least 30% parameter reduction
    prune_pcts = [s["prune_t0.000"]["pct_pruned"] for s in all_stats]
    mean_prune_pct = statistics.mean(prune_pcts)
    kill2 = mean_prune_pct < 30.0
    print(f"  Kill 2: Parameter reduction = {mean_prune_pct:.1f}% "
          f"({'KILL' if kill2 else 'PASS'}, threshold=30%)")

    # Kill 3: Prune-then-calibrate doesn't match calibrated quality within 3%
    ptc_avgs = [r["prune_then_cal"]["avg"] for r in all_results]
    cal_avgs = [r["cal_no_prune"]["avg"] for r in all_results]
    ptc_mean = statistics.mean(ptc_avgs)
    cal_mean = statistics.mean(cal_avgs)
    ptc_vs_cal = ((ptc_mean - cal_mean) / cal_mean) * 100
    kill3 = ptc_vs_cal > 3.0
    print(f"  Kill 3: Prune-then-cal vs cal = {ptc_vs_cal:+.2f}% "
          f"({'KILL' if kill3 else 'PASS'}, threshold=3%)")

    # Kill 4: Dead capsule ratio not consistent across seeds (std > 15%)
    dead_stds = [statistics.stdev([s["prune_t0.000"]["pct_pruned"] for s in all_stats])]
    mean_dead_std = dead_stds[0]
    kill4 = mean_dead_std > 15.0
    print(f"  Kill 4: Dead ratio std = {mean_dead_std:.1f}% "
          f"({'KILL' if kill4 else 'PASS'}, threshold=15%)")

    # ============================================================
    # Summary
    # ============================================================
    n_kills = sum([kill1, kill2, kill3, kill4])
    print(f"\n  VERDICT: {n_kills}/4 kill criteria triggered")
    if n_kills == 0:
        print("  STATUS: PASSED -- Dead capsule pruning is viable")
    elif n_kills <= 1:
        print("  STATUS: MARGINAL -- One criterion failed")
    else:
        print("  STATUS: KILLED -- Multiple criteria failed")


if __name__ == "__main__":
    main()
