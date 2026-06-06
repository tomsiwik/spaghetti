"""Pruning Controls experiment (Exp 10).

Tests two missing controls from the Exp 9 adversarial review:

Phase 1 -- Pre-composition death rate:
  Profile single-domain models BEFORE composition to distinguish
  training-induced death (general ReLU) from composition-induced
  distribution shift.

Phase 2 -- Random pruning baseline:
  Prune same fraction of capsules at RANDOM and compare to
  targeted dead-capsule pruning. Validates that profiling adds value.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Profile single-domain models on own-domain and cross-domain data
  4. Compose by concatenating A and B weight matrices
  5. Profile composed model on joint data
  6. Targeted pruning: prune dead capsules, evaluate
  7. Random pruning: prune same fraction at random (5 draws), evaluate
  8. Compare and decompose death rate

Kill criteria:
  1. If single-domain death > 45%: pruning is general ReLU, not composition-specific
  2. If random pruning quality within 2% of targeted: profiling unnecessary
  3. If composition-induced death < 10%: MATH.md Assumption 6 wrong
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
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
    prune_composed_model,
)
from .pruning_controls import (
    profile_single_domain,
    random_prune_model,
)


def count_params(model) -> int:
    """Count total parameters in a model."""
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def run_pruning_controls_experiment(seed=42):
    """Run the full pruning controls experiment.

    Returns:
        results: dict of method -> {domain: loss, "avg": avg_loss}
        precomp_stats: pre-composition profiling statistics
        random_stats: random pruning statistics
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
    precomp_stats = {}
    random_stats = {}
    domain_names = list(domain_datasets.keys())

    # ============================================================
    # 1. Joint training baseline (upper bound)
    # ============================================================
    print("  [1/7] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 2. Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/7] Pretrain base + fine-tune per domain...")
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
    # 3. PHASE 1: Pre-composition profiling (THE KEY NEW CONTROL)
    # ============================================================
    print("  [3/7] Phase 1: Pre-composition death rate profiling...")

    for d_idx, d_name in enumerate(domain_names):
        other_idx = 1 - d_idx  # other domain index
        other_name = domain_names[other_idx]

        stats = profile_single_domain(
            domain_models[d_idx],
            own_domain_dataset=domain_datasets[d_name][1],
            cross_domain_dataset=domain_datasets[other_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        precomp_stats[d_name] = stats

        print(f"    {d_name} model:")
        for ls in stats["per_layer"]:
            print(f"      Layer {ls['layer']}: dead_own={ls['pct_dead_own']:.1f}%, "
                  f"dead_cross={ls['pct_dead_cross']:.1f}%, "
                  f"dead_both={ls['pct_dead_both']:.1f}%, "
                  f"alive_own_dead_cross={ls['pct_alive_own_dead_cross']:.1f}%")
        agg = stats["aggregate"]
        print(f"      AGGREGATE: dead_own={agg['pct_dead_own']:.1f}%, "
              f"dead_cross={agg['pct_dead_cross']:.1f}%, "
              f"dead_both={agg['pct_dead_both']:.1f}%")

    # ============================================================
    # 4. Compose models + profile post-composition
    # ============================================================
    print("  [4/7] Compose and profile post-composition...")
    composed = compose_relu_models(base, domain_models)
    results["concat_zero_shot"] = _eval_domains(composed, domain_datasets)

    # Profile composed model
    composed_freqs = profile_activations(
        composed, joint_val,
        n_batches=20, batch_size=32, seed=seed,
    )

    # Post-composition death statistics
    post_stats = []
    for l_idx, freq in enumerate(composed_freqs):
        mx.eval(freq)
        freq_list = freq.tolist()
        n_dead = sum(1 for f in freq_list if f == 0.0)
        post_stats.append({
            "layer": l_idx,
            "n_capsules": len(freq_list),
            "n_dead": n_dead,
            "pct_dead": n_dead / len(freq_list) * 100,
        })
        print(f"    Composed Layer {l_idx}: {n_dead}/{len(freq_list)} dead "
              f"({post_stats[-1]['pct_dead']:.1f}%)")

    precomp_stats["composed"] = post_stats

    # ============================================================
    # 5. Targeted pruning (dead capsules, same as Exp 9)
    # ============================================================
    print("  [5/7] Targeted pruning (dead capsules, threshold=0.0)...")
    model_targeted = copy.deepcopy(composed)
    targeted_masks = identify_dead_capsules(composed_freqs, threshold=0.0)
    targeted_prune_stats = prune_model(model_targeted, targeted_masks, verbose=True)
    mx.eval(model_targeted.parameters())
    results["targeted_prune"] = _eval_domains(model_targeted, domain_datasets)

    # Also with calibration
    model_targeted_cal = copy.deepcopy(model_targeted)
    _freeze_attention(model_targeted_cal)
    train(model_targeted_cal, joint_train, steps=CALIBRATION_STEPS,
          batch_size=BATCH_SIZE, lr=LR * 0.1, seed=seed, log_every=9999)
    model_targeted_cal.unfreeze()
    results["targeted_prune_cal"] = _eval_domains(model_targeted_cal, domain_datasets)

    # ============================================================
    # 6. PHASE 2: Random pruning (THE KEY NEW CONTROL)
    # ============================================================
    target_prune_rate = targeted_prune_stats["pct_pruned"] / 100.0
    n_random_draws = 5

    print(f"  [6/7] Phase 2: Random pruning at {target_prune_rate:.1%} "
          f"({n_random_draws} draws)...")

    random_results_list = []
    random_cal_results_list = []

    for draw in range(n_random_draws):
        draw_seed = seed * 1000 + draw
        model_random = copy.deepcopy(composed)
        rstats = random_prune_model(
            model_random, target_prune_rate=target_prune_rate,
            seed=draw_seed, verbose=False,
        )
        mx.eval(model_random.parameters())

        r = _eval_domains(model_random, domain_datasets)
        random_results_list.append(r)

        # Also with calibration
        model_random_cal = copy.deepcopy(model_random)
        _freeze_attention(model_random_cal)
        train(model_random_cal, joint_train, steps=CALIBRATION_STEPS,
              batch_size=BATCH_SIZE, lr=LR * 0.1, seed=draw_seed, log_every=9999)
        model_random_cal.unfreeze()
        rc = _eval_domains(model_random_cal, domain_datasets)
        random_cal_results_list.append(rc)

        print(f"    Draw {draw}: random_prune avg={r['avg']:.4f}, "
              f"random_prune+cal avg={rc['avg']:.4f}")

    # Aggregate random pruning results
    random_avgs = [r["avg"] for r in random_results_list]
    random_cal_avgs = [r["avg"] for r in random_cal_results_list]

    results["random_prune"] = {
        "avg": statistics.mean(random_avgs),
        "std": statistics.stdev(random_avgs) if len(random_avgs) > 1 else 0,
        "all_avgs": random_avgs,
    }
    results["random_prune_cal"] = {
        "avg": statistics.mean(random_cal_avgs),
        "std": statistics.stdev(random_cal_avgs) if len(random_cal_avgs) > 1 else 0,
        "all_avgs": random_cal_avgs,
    }
    # Copy domain-level results from first draw for display
    for d_name in domain_names:
        results["random_prune"][d_name] = statistics.mean(
            [r[d_name] for r in random_results_list])
        results["random_prune_cal"][d_name] = statistics.mean(
            [r[d_name] for r in random_cal_results_list])

    random_stats = {
        "target_prune_rate": target_prune_rate,
        "n_draws": n_random_draws,
        "raw_avgs": random_avgs,
        "raw_cal_avgs": random_cal_avgs,
    }

    # ============================================================
    # 7. Weight averaging baseline (for comparison)
    # ============================================================
    print("  [7/7] Weight averaging baseline...")
    averaged = weight_average_relu_models(base, domain_models)
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)

    return results, precomp_stats, random_stats


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []
    all_precomp = []
    all_random = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r, pc, rs = run_pruning_controls_experiment(seed=seed)
        all_results.append(r)
        all_precomp.append(pc)
        all_random.append(rs)

        # Per-seed summary
        for method, vals in r.items():
            if isinstance(vals.get("avg"), (int, float)):
                domains = [k for k in vals if k not in ("avg", "std", "all_avgs")]
                detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains if isinstance(vals.get(d), (int, float)))
                extra = f" (std={vals['std']:.4f})" if "std" in vals else ""
                print(f"  {method:<25} avg={vals['avg']:.4f}{extra} ({detail})")

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    # Standard methods (single value per seed)
    simple_methods = ["joint", "concat_zero_shot", "targeted_prune",
                      "targeted_prune_cal", "weight_avg"]
    random_methods = ["random_prune", "random_prune_cal"]

    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])
    concat_mean = statistics.mean([r["concat_zero_shot"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs joint':>10} {'vs concat':>10}")
    print("  " + "-" * 70)

    for method in simple_methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_concat:>+9.1f}%")

    for method in random_methods:
        # Aggregate across all draws and seeds
        all_draw_avgs = []
        for r in all_results:
            all_draw_avgs.extend(r[method]["all_avgs"])
        mean = statistics.mean(all_draw_avgs)
        std = statistics.stdev(all_draw_avgs) if len(all_draw_avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        n = len(all_draw_avgs)
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_concat:>+9.1f}%  (n={n})")

    # ============================================================
    # Pre-composition death rate analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Pre-Composition Death Rate Analysis (3-seed mean)")
    print(f"{'='*70}")

    domain_names = list(all_precomp[0].keys())
    domain_names = [d for d in domain_names if d != "composed"]

    for d_name in domain_names:
        print(f"\n  {d_name} model (single-domain, P={N_CAPSULES}):")
        for l_idx in range(4):
            dead_own = [pc[d_name]["per_layer"][l_idx]["pct_dead_own"] for pc in all_precomp]
            dead_cross = [pc[d_name]["per_layer"][l_idx]["pct_dead_cross"] for pc in all_precomp]
            dead_both = [pc[d_name]["per_layer"][l_idx]["pct_dead_both"] for pc in all_precomp]
            alive_own_dead_cross = [pc[d_name]["per_layer"][l_idx]["pct_alive_own_dead_cross"] for pc in all_precomp]
            print(f"    Layer {l_idx}: dead_own={statistics.mean(dead_own):.1f}% "
                  f"dead_cross={statistics.mean(dead_cross):.1f}% "
                  f"dead_both={statistics.mean(dead_both):.1f}% "
                  f"alive_own_dead_cross={statistics.mean(alive_own_dead_cross):.1f}%")

        agg_dead_own = [pc[d_name]["aggregate"]["pct_dead_own"] for pc in all_precomp]
        agg_dead_cross = [pc[d_name]["aggregate"]["pct_dead_cross"] for pc in all_precomp]
        agg_dead_both = [pc[d_name]["aggregate"]["pct_dead_both"] for pc in all_precomp]
        print(f"    AGGREGATE: dead_own={statistics.mean(agg_dead_own):.1f}% "
              f"(std={statistics.stdev(agg_dead_own):.1f}%) "
              f"dead_cross={statistics.mean(agg_dead_cross):.1f}% "
              f"dead_both={statistics.mean(agg_dead_both):.1f}%")

    # Composed model death rate (from Exp 9 replication)
    print(f"\n  Composed model (P_total={N_CAPSULES*2}):")
    for l_idx in range(4):
        pct_dead = [pc["composed"][l_idx]["pct_dead"] for pc in all_precomp]
        print(f"    Layer {l_idx}: dead={statistics.mean(pct_dead):.1f}% "
              f"(std={statistics.stdev(pct_dead):.1f}%)")

    total_composed_dead = []
    for pc in all_precomp:
        total_dead = sum(pc["composed"][l]["n_dead"] for l in range(4))
        total_caps = sum(pc["composed"][l]["n_capsules"] for l in range(4))
        total_composed_dead.append(total_dead / total_caps * 100)
    print(f"    AGGREGATE: dead={statistics.mean(total_composed_dead):.1f}% "
          f"(std={statistics.stdev(total_composed_dead):.1f}%)")

    # ============================================================
    # Death rate decomposition
    # ============================================================
    print(f"\n{'='*70}")
    print("  Death Rate Decomposition (3-seed mean)")
    print(f"{'='*70}")

    # Average single-domain death across both domains
    all_single_death = []
    all_cross_death = []
    all_domain_specific_death = []
    for pc in all_precomp:
        for d_name in domain_names:
            all_single_death.append(pc[d_name]["aggregate"]["pct_dead_own"])
            all_cross_death.append(pc[d_name]["aggregate"]["pct_dead_cross"])
            # alive_own_dead_cross per domain
            total_P = pc[d_name]["aggregate"]["total_capsules"]
            alive_own_dead_cross_total = sum(
                pc[d_name]["per_layer"][l]["alive_own_dead_cross"]
                for l in range(4)
            )
            all_domain_specific_death.append(alive_own_dead_cross_total / total_P * 100)

    delta_training = statistics.mean(all_single_death)
    delta_cross = statistics.mean(all_cross_death)
    delta_composed = statistics.mean(total_composed_dead)
    delta_domain_specific = statistics.mean(all_domain_specific_death)

    # delta_shift = what's left after accounting for training death and domain death
    # In the composed model, each pool contributes delta_training + delta_domain
    # But the actual composed rate includes distribution shift too
    delta_shift = delta_composed - delta_training

    print(f"\n  delta_training (single-domain dead): {delta_training:.1f}%")
    print(f"  delta_composed (post-composition dead): {delta_composed:.1f}%")
    print(f"  delta_shift (composition-induced):     {delta_shift:.1f}%")
    print(f"  delta_domain_specific (alive on own, dead on cross): {delta_domain_specific:.1f}%")

    if delta_training > 45:
        print(f"\n  NOTE: Single-domain death rate ({delta_training:.1f}%) > 45%")
        print("  -> Pruning is largely a GENERAL ReLU technique, not composition-specific")
    elif delta_training < 20:
        print(f"\n  NOTE: Single-domain death rate ({delta_training:.1f}%) < 20%")
        print("  -> Death is predominantly COMPOSITION-INDUCED (strong finding)")
    else:
        print(f"\n  NOTE: Single-domain death rate ({delta_training:.1f}%) is moderate")
        print("  -> Both training and composition contribute to death")

    # ============================================================
    # Random pruning analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Random vs Targeted Pruning Analysis")
    print(f"{'='*70}")

    # Targeted pruning quality
    targeted_avgs = [r["targeted_prune"]["avg"] for r in all_results]
    targeted_mean = statistics.mean(targeted_avgs)

    # Random pruning quality (all draws across all seeds)
    random_all = []
    random_cal_all = []
    for r in all_results:
        random_all.extend(r["random_prune"]["all_avgs"])
        random_cal_all.extend(r["random_prune_cal"]["all_avgs"])
    random_mean = statistics.mean(random_all)
    random_std = statistics.stdev(random_all) if len(random_all) > 1 else 0
    random_cal_mean = statistics.mean(random_cal_all)
    random_cal_std = statistics.stdev(random_cal_all) if len(random_cal_all) > 1 else 0

    targeted_cal_avgs = [r["targeted_prune_cal"]["avg"] for r in all_results]
    targeted_cal_mean = statistics.mean(targeted_cal_avgs)

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs concat':>10} {'vs targeted':>12}")
    print("  " + "-" * 70)
    print(f"  {'concat (no prune)':<25} {concat_mean:>8.4f} {'':>8} {'0.0%':>10} {'':>12}")
    print(f"  {'targeted prune':<25} {targeted_mean:>8.4f} "
          f"{statistics.stdev(targeted_avgs):>8.4f} "
          f"{((targeted_mean-concat_mean)/concat_mean*100):>+9.1f}% {'baseline':>12}")
    print(f"  {'random prune':<25} {random_mean:>8.4f} {random_std:>8.4f} "
          f"{((random_mean-concat_mean)/concat_mean*100):>+9.1f}% "
          f"{((random_mean-targeted_mean)/targeted_mean*100):>+11.1f}%")
    print(f"  {'targeted+cal':<25} {targeted_cal_mean:>8.4f} "
          f"{statistics.stdev(targeted_cal_avgs):>8.4f} "
          f"{((targeted_cal_mean-concat_mean)/concat_mean*100):>+9.1f}% {'':>12}")
    print(f"  {'random+cal':<25} {random_cal_mean:>8.4f} {random_cal_std:>8.4f} "
          f"{((random_cal_mean-concat_mean)/concat_mean*100):>+9.1f}% "
          f"{((random_cal_mean-targeted_cal_mean)/targeted_cal_mean*100):>+11.1f}%")

    random_vs_targeted = ((random_mean - targeted_mean) / targeted_mean) * 100
    random_cal_vs_targeted_cal = ((random_cal_mean - targeted_cal_mean) / targeted_cal_mean) * 100

    print(f"\n  Random vs targeted (no cal): {random_vs_targeted:+.1f}%")
    print(f"  Random vs targeted (with cal): {random_cal_vs_targeted_cal:+.1f}%")

    if abs(random_vs_targeted) < 2.0:
        print("  -> Random pruning within 2% of targeted: profiling may be unnecessary")
    else:
        print(f"  -> Random pruning {random_vs_targeted:+.1f}% vs targeted: profiling MATTERS")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Kill 1: Single-domain death > 45%
    kill1 = delta_training > 45
    print(f"\n  Kill 1: Single-domain death = {delta_training:.1f}% "
          f"({'KILL (general ReLU)' if kill1 else 'PASS (composition-specific)'}, threshold=45%)")

    # Kill 2: Random pruning within 2% of targeted
    kill2 = abs(random_vs_targeted) < 2.0
    print(f"  Kill 2: Random vs targeted = {random_vs_targeted:+.1f}% "
          f"({'KILL (profiling unnecessary)' if kill2 else 'PASS (profiling matters)'}, threshold=2%)")

    # Kill 3: Composition-induced death < 10%
    kill3 = delta_shift < 10
    print(f"  Kill 3: Composition-induced death = {delta_shift:.1f}% "
          f"({'KILL (too little shift)' if kill3 else 'PASS (significant shift)'}, threshold=10%)")

    # ============================================================
    # Summary
    # ============================================================
    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    if kill1 and not kill2:
        print("  FINDING: Pruning is a general ReLU technique (not composition-specific),")
        print("  BUT targeted identification still matters (not random).")
    elif not kill1 and not kill2:
        print("  FINDING: Death is composition-specific AND targeted identification matters.")
        print("  The profiling step adds real value.")
    elif kill1 and kill2:
        print("  FINDING: Pruning is general AND random pruning works too.")
        print("  The composed model is overparameterized enough that any pruning helps.")
    elif not kill1 and kill2:
        print("  FINDING: Death is composition-specific BUT random pruning also works.")
        print("  Surprising: profiling adds no value despite composition being the cause.")

    if kill3:
        print("  NOTE: Most death is training-induced. MATH.md Assumption 6 needs revision.")
    else:
        print("  NOTE: Significant composition-induced death. MATH.md Assumption 6 holds.")


if __name__ == "__main__":
    main()
