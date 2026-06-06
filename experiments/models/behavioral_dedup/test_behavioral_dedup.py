"""Behavioral Deduplication Experiment.

Tests whether activation-based behavioral similarity finds functional
redundancy that weight-cosine misses.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose by concatenating A and B weight matrices from both domains
  4. Profile behavioral similarity (co-activation Jaccard, output correlation)
  5. Compare against weight-cosine baseline (Exp 8)
  6. Test merging quality for behaviorally-identified redundant pairs

Kill criterion: behavioral dedup finds <5% functional redundancy
above weight-cosine baseline.
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..relu_router.test_composition import (
    compose_relu_models, weight_average_relu_models,
    _make_relu_model, _freeze_attention, _eval_domains,
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE, BATCH_SIZE, LR,
)
from .behavioral_dedup import (
    profile_behavioral,
    compute_jaccard_matrix,
    compute_output_correlation_matrix,
    compute_conditioned_output_cosine,
    compare_behavioral_vs_weight,
    behavioral_deduplicate,
)
from ..capsule_dedup.capsule_dedup import (
    cosine_similarity_matrix,
    deduplicate_composed_model,
)


def run_experiment(seed=42):
    """Run the full behavioral dedup experiment for one seed.

    Returns dict with all results.
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
    domain_names = list(domain_datasets.keys())
    results = {}

    # ============================================================
    # 1. Joint training baseline
    # ============================================================
    print("  [1/6] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 2. Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/6] Pretrain base + fine-tune per domain...")
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
    # 3. Compose by concatenation (zero-shot baseline)
    # ============================================================
    print("  [3/6] Compose by concatenation...")
    composed = compose_relu_models(base, domain_models)
    results["concat_zero_shot"] = _eval_domains(composed, domain_datasets)

    # Weight averaging baseline
    averaged = weight_average_relu_models(base, domain_models)
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)

    # ============================================================
    # 4. Behavioral vs weight-cosine comparison (THE CORE TEST)
    # ============================================================
    print("  [4/6] Behavioral vs weight-cosine comparison...")
    pool_sizes = [N_CAPSULES, N_CAPSULES]

    # Test multiple Jaccard thresholds (at default tau_rho=0.3)
    jaccard_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    comparison_results = {}

    for jt in jaccard_thresholds:
        model_copy = copy.deepcopy(composed)
        comp = compare_behavioral_vs_weight(
            model_copy, joint_val,
            n_batches=20, batch_size=32, seed=seed,
            jaccard_threshold=jt,
            output_corr_threshold=0.3,  # relatively permissive
            weight_cos_threshold=0.95,
            pool_sizes=pool_sizes,
            verbose=(jt == 0.7),  # only verbose for primary threshold
        )
        comparison_results[f"jt_{jt:.1f}"] = comp

    # Sweep output correlation threshold at primary Jaccard=0.7
    output_corr_thresholds = [0.3, 0.5, 0.7]
    output_corr_results = {}

    for tau_rho in output_corr_thresholds:
        model_copy = copy.deepcopy(composed)
        comp = compare_behavioral_vs_weight(
            model_copy, joint_val,
            n_batches=20, batch_size=32, seed=seed,
            jaccard_threshold=0.7,
            output_corr_threshold=tau_rho,
            weight_cos_threshold=0.95,
            pool_sizes=pool_sizes,
            verbose=False,
        )
        output_corr_results[f"tau_rho_{tau_rho:.1f}"] = comp

    results["comparisons"] = comparison_results
    results["output_corr_sweep"] = output_corr_results

    # ============================================================
    # 5. Behavioral dedup + quality evaluation
    # ============================================================
    print("  [5/6] Behavioral dedup + quality evaluation...")
    dedup_results = {}

    for jt in [0.5, 0.7, 0.9]:
        key = f"behavioral_jt{jt:.1f}"
        model_dedup = copy.deepcopy(composed)
        merge_stats = behavioral_deduplicate(
            model_dedup, joint_val,
            jaccard_threshold=jt,
            output_corr_threshold=0.3,
            n_batches=20, batch_size=32, seed=seed,
            pool_sizes=pool_sizes,
            verbose=True,
        )
        mx.eval(model_dedup.parameters())
        quality = _eval_domains(model_dedup, domain_datasets)
        dedup_results[key] = {
            "merge_stats": merge_stats,
            "quality": quality,
        }
        results[key] = quality

    # Weight-cosine dedup baseline for comparison
    print("  [5b/6] Weight-cosine dedup baseline (tau=0.95)...")
    model_wcos = copy.deepcopy(composed)
    wcos_stats = deduplicate_composed_model(
        model_wcos, threshold=0.95,
        pool_sizes=pool_sizes, cross_pool_only=True, verbose=True,
    )
    mx.eval(model_wcos.parameters())
    results["weight_cos_dedup"] = _eval_domains(model_wcos, domain_datasets)

    results["dedup_details"] = dedup_results
    results["weight_cos_stats"] = wcos_stats

    # ============================================================
    # 6. Jaccard distribution analysis (even without merging)
    # ============================================================
    print("  [6/6] Jaccard distribution analysis...")
    dist_model = copy.deepcopy(composed)
    layer_stats = profile_behavioral(dist_model, joint_val, n_batches=20,
                                     batch_size=32, seed=seed)

    jaccard_distributions = []
    for l_idx, layer in enumerate(dist_model.layers):
        stats = layer_stats[l_idx]
        J = compute_jaccard_matrix(stats)
        mx.eval(J)
        J_np = J.tolist()
        P = len(J_np)

        # Collect cross-pool Jaccard values for alive capsules
        freq = stats["fire_count"] / max(stats["total_positions"], 1)
        mx.eval(freq)
        freq_list = freq.tolist()

        cross_jacs = []
        for i in range(N_CAPSULES):
            if freq_list[i] <= 0:
                continue
            for j in range(N_CAPSULES, P):
                if freq_list[j] <= 0:
                    continue
                cross_jacs.append(J_np[i][j])

        if cross_jacs:
            cross_jacs.sort()
            n = len(cross_jacs)
            percentiles = {
                "p50": cross_jacs[n // 2],
                "p75": cross_jacs[int(n * 0.75)],
                "p90": cross_jacs[int(n * 0.90)],
                "p95": cross_jacs[int(n * 0.95)],
                "p99": cross_jacs[int(n * 0.99)],
                "max": cross_jacs[-1],
                "mean": sum(cross_jacs) / n,
                "n_above_0.5": sum(1 for v in cross_jacs if v > 0.5),
                "n_above_0.7": sum(1 for v in cross_jacs if v > 0.7),
                "n_above_0.9": sum(1 for v in cross_jacs if v > 0.9),
                "n_total": n,
            }
        else:
            percentiles = {"n_total": 0}

        jaccard_distributions.append({
            "layer": l_idx,
            "percentiles": percentiles,
        })

    results["jaccard_distributions"] = jaccard_distributions

    return results


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r = run_experiment(seed=seed)
        all_results.append(r)

    # ============================================================
    # Aggregate quality results
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    quality_methods = ["joint", "concat_zero_shot", "weight_avg",
                       "weight_cos_dedup",
                       "behavioral_jt0.5", "behavioral_jt0.7", "behavioral_jt0.9"]
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])
    concat_mean = statistics.mean([r["concat_zero_shot"]["avg"] for r in all_results])

    print(f"\n  {'Method':<30} {'avg':>8} {'std':>8} {'vs joint':>10} {'vs concat':>10}")
    print("  " + "-" * 75)
    for method in quality_methods:
        if method not in all_results[0]:
            continue
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        print(f"  {method:<30} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_concat:>+9.1f}%")

    # ============================================================
    # Behavioral vs Weight comparison
    # ============================================================
    print(f"\n  Behavioral vs Weight-Cosine Redundancy (3-seed mean)")
    print("  " + "-" * 75)

    for jt_key in sorted(all_results[0]["comparisons"].keys()):
        beh_only = [r["comparisons"][jt_key]["total_behavioral_only"] for r in all_results]
        beh_total = [r["comparisons"][jt_key]["total_behavioral_redundant"] for r in all_results]
        wt_total = [r["comparisons"][jt_key]["total_weight_redundant"] for r in all_results]
        both = [r["comparisons"][jt_key]["total_both"] for r in all_results]
        beh_only_pct = [r["comparisons"][jt_key]["behavioral_only_capsule_pct"] for r in all_results]

        print(f"  {jt_key}: behavioral={statistics.mean(beh_total):.1f} pairs, "
              f"weight={statistics.mean(wt_total):.1f} pairs, "
              f"behavioral-only={statistics.mean(beh_only):.1f} pairs, "
              f"both={statistics.mean(both):.1f} pairs")
        print(f"    behavioral-only capsule %: {statistics.mean(beh_only_pct):.1f}%")

    # ============================================================
    # Output Correlation Threshold Sweep (at J>0.7)
    # ============================================================
    print(f"\n  Output Correlation Threshold Sweep (J>0.7, 3-seed mean)")
    print("  " + "-" * 75)

    for tau_key in sorted(all_results[0]["output_corr_sweep"].keys()):
        beh_total = [r["output_corr_sweep"][tau_key]["total_behavioral_redundant"] for r in all_results]
        beh_only = [r["output_corr_sweep"][tau_key]["total_behavioral_only"] for r in all_results]
        beh_only_pct = [r["output_corr_sweep"][tau_key]["behavioral_only_capsule_pct"] for r in all_results]

        print(f"  {tau_key}: behavioral={statistics.mean(beh_total):.1f} pairs, "
              f"behavioral-only={statistics.mean(beh_only):.1f} pairs, "
              f"capsule %: {statistics.mean(beh_only_pct):.1f}% "
              f"(per-seed: {', '.join(f'{v:.1f}%' for v in beh_only_pct)})")

    # ============================================================
    # Jaccard distribution
    # ============================================================
    print(f"\n  Cross-Pool Jaccard Distribution (3-seed mean)")
    print("  " + "-" * 75)
    for l_idx in range(4):
        p_means = {}
        for pkey in ["mean", "p50", "p75", "p90", "p95", "p99", "max",
                     "n_above_0.5", "n_above_0.7", "n_above_0.9", "n_total"]:
            vals = [r["jaccard_distributions"][l_idx]["percentiles"].get(pkey, 0)
                    for r in all_results]
            p_means[pkey] = statistics.mean(vals)

        print(f"  Layer {l_idx}: mean={p_means['mean']:.4f} "
              f"p50={p_means['p50']:.4f} p90={p_means['p90']:.4f} "
              f"p95={p_means['p95']:.4f} max={p_means['max']:.4f}")
        print(f"    n_above_0.5={p_means['n_above_0.5']:.0f} "
              f"n_above_0.7={p_means['n_above_0.7']:.0f} "
              f"n_above_0.9={p_means['n_above_0.9']:.0f} "
              f"(of {p_means['n_total']:.0f} alive cross-pool pairs)")

    # ============================================================
    # Dedup merge statistics
    # ============================================================
    print(f"\n  Behavioral Dedup Merge Statistics (3-seed mean)")
    print("  " + "-" * 75)
    for jt in [0.5, 0.7, 0.9]:
        key = f"behavioral_jt{jt:.1f}"
        pcts = [r["dedup_details"][key]["merge_stats"]["pct_merged"] for r in all_results]
        merged = [r["dedup_details"][key]["merge_stats"]["total_merged"] for r in all_results]
        print(f"  Jaccard>{jt}: {statistics.mean(merged):.1f} capsules merged "
              f"({statistics.mean(pcts):.1f}%)")

    # Weight-cosine for comparison
    wcos_pcts = [r["weight_cos_stats"]["pct_capsules_removed"] for r in all_results]
    wcos_removed = [r["weight_cos_stats"]["total_capsules_removed"] for r in all_results]
    print(f"  Weight-cos>0.95: {statistics.mean(wcos_removed):.1f} capsules merged "
          f"({statistics.mean(wcos_pcts):.1f}%)")

    # ============================================================
    # Kill criterion analysis
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  KILL CRITERION ANALYSIS")
    print(f"  {'='*70}")

    # Primary threshold: Jaccard > 0.7
    primary_key = "jt_0.7"
    beh_only_pcts = [r["comparisons"][primary_key]["behavioral_only_capsule_pct"]
                     for r in all_results]
    mean_beh_only_pct = statistics.mean(beh_only_pcts)

    std_beh_only_pct = statistics.stdev(beh_only_pcts) if len(beh_only_pcts) > 1 else 0

    print(f"\n  Kill criterion: co-activation dedup finds <5% functional redundancy")
    print(f"  above weight-cosine baseline")
    print(f"\n  Behavioral-only capsule % (Jaccard>0.7, cross-pool, alive only):")
    print(f"    Mean: {mean_beh_only_pct:.1f}% +/- {std_beh_only_pct:.1f}%")
    for i, seed in enumerate(seeds):
        print(f"    Seed {seed}: {beh_only_pcts[i]:.1f}%")

    # Also show at different output correlation thresholds
    print(f"\n  Kill criterion at different tau_rho (J>0.7):")
    for tau_key in sorted(all_results[0]["output_corr_sweep"].keys()):
        pcts = [r["output_corr_sweep"][tau_key]["behavioral_only_capsule_pct"] for r in all_results]
        mean_pct = statistics.mean(pcts)
        std_pct = statistics.stdev(pcts) if len(pcts) > 1 else 0
        status = "PASS" if mean_pct >= 5.0 else "KILL"
        print(f"    {tau_key}: {mean_pct:.1f}% +/- {std_pct:.1f}% [{status}]")

    if mean_beh_only_pct < 5.0:
        print(f"\n  VERDICT: KILL")
        print(f"  Behavioral analysis finds {mean_beh_only_pct:.1f}% additional")
        print(f"  functional redundancy, below 5% threshold.")
    else:
        print(f"\n  VERDICT: PASS")
        print(f"  Behavioral analysis finds {mean_beh_only_pct:.1f}% additional")
        print(f"  functional redundancy, above 5% threshold.")

    # Also check quality impact
    print(f"\n  Quality impact of behavioral dedup:")
    for jt in [0.5, 0.7, 0.9]:
        key = f"behavioral_jt{jt:.1f}"
        if key in all_results[0]:
            avgs = [r[key]["avg"] for r in all_results]
            mean = statistics.mean(avgs)
            vs_concat = ((mean - concat_mean) / concat_mean) * 100
            print(f"    Jaccard>{jt}: {mean:.4f} (vs concat: {vs_concat:+.1f}%)")


if __name__ == "__main__":
    main()
