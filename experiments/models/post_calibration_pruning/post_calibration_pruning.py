"""Post-Calibration Pruning Safety -- exp_post_calibration_pruning.

Context from prior experiments:
  - revival_under_composition: composed revival is only 2.9% at 100-step
    calibration. Dead capsules STAY dead under composition.
  - prune_before_compose: pre-prune-then-compose matches compose-then-prune
    within +0.01% (200x margin on 2% threshold).

This experiment tests a THIRD pipeline ordering:
  compose -> calibrate -> profile -> prune

The hypothesis: since revival is suppressed under composition (2.9% at
100 steps), post-calibration pruning should be even safer than
pre-calibration pruning. The calibration step "locks in" the dead set
before profiling, capturing any transient revival that would have occurred.

Four pipelines compared:

  Pipeline A (BASELINE -- pre-composition pruning):
    1. Fine-tune per domain -> profile -> prune -> compose -> calibrate 100 steps
    (Validated in prune_before_compose at +0.01% vs compose-then-prune)

  Pipeline B (NEW -- post-calibration pruning):
    1. Fine-tune per domain -> compose -> calibrate 100 steps -> profile -> prune

  Pipeline C (REFERENCE -- compose-then-prune, pre-calibration):
    1. Fine-tune per domain -> compose -> profile -> prune -> calibrate 100 steps

  Pipeline D (CONTROL -- compose + calibrate, no prune):
    1. Fine-tune per domain -> compose -> calibrate 100 steps (no pruning)

Kill criteria:
  1. Pipeline B quality degrades >2% vs Pipeline A (pre-composition pruning)
  2. Revival rate after 100-step calibration >5% (measured by comparing
     dead sets at compose-time vs post-calibration)
"""

import copy
import statistics

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..relu_router.test_composition import (
    compose_relu_models,
    _make_relu_model, _freeze_attention, _eval_domains,
    full_capsule_calibrate,
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE,
    CALIBRATION_STEPS, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
)
from ..prune_before_compose.prune_before_compose import compose_pruned_models
from ..capsule_revival.test_capsule_revival import get_dead_mask, transition_counts


def measure_revival_rate(dead_mask_before, dead_mask_after):
    """Compute revival rate: fraction of before-dead that became alive.

    Args:
        dead_mask_before: list of bool (True=dead)
        dead_mask_after: list of bool (True=dead)

    Returns:
        revival_rate: float (0 to 1)
        n_revived: int
        n_dead_before: int
        n_newly_dead: int
    """
    trans = transition_counts(dead_mask_before, dead_mask_after)
    n_dead_before = trans["dd"] + trans["da"]
    n_revived = trans["da"]
    n_newly_dead = trans["ad"]
    revival_rate = n_revived / n_dead_before if n_dead_before > 0 else 0.0
    return revival_rate, n_revived, n_dead_before, n_newly_dead


def profile_dead_mask(model, dataset, seed=42):
    """Profile model and return flat dead mask (True=dead) and stats.

    Returns:
        dead_mask: list of bool (True=dead)
        death_rate: float
        n_alive: int
        n_total: int
    """
    freqs = profile_activations(model, dataset, n_batches=20, batch_size=32, seed=seed)
    flat_mask, per_layer_masks = get_dead_mask(freqs)
    n_total = len(flat_mask)
    n_dead = sum(flat_mask)
    return flat_mask, n_dead / n_total, n_total - n_dead, n_total


def run_pipeline_experiment(seed=42):
    """Run all four pipelines for one seed and return results."""
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
    domain_names = sorted(domain_datasets.keys())
    results = {}
    meta = {}

    # ============================================================
    # 0. Joint training baseline (upper bound)
    # ============================================================
    print(f"  [0/7] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 1. Shared: Pretrain base + domain fine-tune
    # ============================================================
    print(f"  [1/7] Pretrain base + fine-tune per domain...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # ============================================================
    # 2. Pipeline A: Pre-composition pruning (validated baseline)
    #    profile -> prune -> compose -> calibrate
    # ============================================================
    print(f"  [2/7] Pipeline A: pre-composition pruning...")

    pruned_domain_models_A = {}
    prune_stats_A = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(domain_models[d_name])
        freqs_d = profile_activations(
            model_d, domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        masks_d = identify_dead_capsules(freqs_d, threshold=0.0)
        before = sum(m.shape[0] for m in masks_d)
        alive = sum(int(mx.sum(m).item()) for m in masks_d)
        prune_model(model_d, masks_d, verbose=False)
        pruned_domain_models_A[d_name] = model_d
        prune_stats_A[d_name] = {"before": before, "alive": alive}
        print(f"    {d_name}: {before} -> {alive} ({(before-alive)/before*100:.1f}% pruned)")

    composed_A = compose_pruned_models(base, [pruned_domain_models_A[d] for d in domain_names])
    full_capsule_calibrate(composed_A, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["A_pre_comp_prune"] = _eval_domains(composed_A, domain_datasets)

    meta["A_total_before"] = sum(s["before"] for s in prune_stats_A.values())
    meta["A_total_alive"] = sum(s["alive"] for s in prune_stats_A.values())

    # ============================================================
    # 3. Pipeline B: Post-calibration pruning (THE NEW PIPELINE)
    #    compose -> calibrate 100 steps -> profile -> prune
    # ============================================================
    print(f"  [3/7] Pipeline B: post-calibration pruning...")

    composed_B = compose_relu_models(base, [domain_models[d] for d in domain_names])

    # Profile BEFORE calibration (for revival measurement)
    dead_mask_pre_cal, death_rate_pre_cal, _, _ = profile_dead_mask(
        composed_B, joint_val, seed=seed)
    meta["B_death_rate_pre_cal"] = death_rate_pre_cal
    print(f"    Pre-calibration death rate: {death_rate_pre_cal:.1%}")

    # Calibrate 100 steps
    full_capsule_calibrate(composed_B, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)

    # Profile AFTER calibration
    dead_mask_post_cal, death_rate_post_cal, n_alive_B, n_total_B = profile_dead_mask(
        composed_B, joint_val, seed=seed)
    meta["B_death_rate_post_cal"] = death_rate_post_cal
    print(f"    Post-calibration death rate: {death_rate_post_cal:.1%}")

    # Compute revival rate during calibration
    revival_rate, n_revived, n_dead_before, n_newly_dead = measure_revival_rate(
        dead_mask_pre_cal, dead_mask_post_cal)
    meta["B_revival_rate"] = revival_rate
    meta["B_n_revived"] = n_revived
    meta["B_n_dead_before"] = n_dead_before
    meta["B_n_newly_dead"] = n_newly_dead
    print(f"    Revival during calibration: {revival_rate:.1%} "
          f"({n_revived}/{n_dead_before} dead revived, {n_newly_dead} newly dead)")

    # Evaluate BEFORE pruning (post-calibration, full model)
    results["B_post_cal_no_prune"] = _eval_domains(composed_B, domain_datasets)

    # Now prune based on post-calibration profile
    freqs_B = profile_activations(composed_B, joint_val, n_batches=20, batch_size=32, seed=seed)
    masks_B = identify_dead_capsules(freqs_B, threshold=0.0)
    n_alive_after_prune_B = sum(int(mx.sum(m).item()) for m in masks_B)
    prune_model(composed_B, masks_B, verbose=False)
    meta["B_total_before"] = n_total_B
    meta["B_total_alive"] = n_alive_after_prune_B
    print(f"    Pruned: {n_total_B} -> {n_alive_after_prune_B} "
          f"({(n_total_B - n_alive_after_prune_B)/n_total_B*100:.1f}% pruned)")

    results["B_post_cal_pruned"] = _eval_domains(composed_B, domain_datasets)

    # ============================================================
    # 4. Pipeline C: Compose-then-prune (pre-calibration prune)
    #    compose -> profile -> prune -> calibrate
    # ============================================================
    print(f"  [4/7] Pipeline C: compose-then-prune (pre-calibration)...")

    composed_C = compose_relu_models(base, [domain_models[d] for d in domain_names])
    freqs_C = profile_activations(composed_C, joint_val, n_batches=20, batch_size=32, seed=seed)
    masks_C = identify_dead_capsules(freqs_C, threshold=0.0)
    total_before_C = sum(m.shape[0] for m in masks_C)
    total_alive_C = sum(int(mx.sum(m).item()) for m in masks_C)
    prune_model(composed_C, masks_C, verbose=False)
    meta["C_total_before"] = total_before_C
    meta["C_total_alive"] = total_alive_C
    print(f"    Pruned: {total_before_C} -> {total_alive_C} "
          f"({(total_before_C - total_alive_C)/total_before_C*100:.1f}% pruned)")

    full_capsule_calibrate(composed_C, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["C_compose_then_prune"] = _eval_domains(composed_C, domain_datasets)

    # ============================================================
    # 5. Pipeline D: Compose + calibrate, no pruning (control)
    # ============================================================
    print(f"  [5/7] Pipeline D: compose + calibrate, no prune (control)...")
    composed_D = compose_relu_models(base, [domain_models[d] for d in domain_names])
    full_capsule_calibrate(composed_D, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["D_no_prune"] = _eval_domains(composed_D, domain_datasets)

    # ============================================================
    # 6. Pipeline B2: Post-calibration with EXTENDED calibration (200 steps)
    # ============================================================
    print(f"  [6/7] Pipeline B2: post-calibration pruning (200 steps cal)...")

    composed_B2 = compose_relu_models(base, [domain_models[d] for d in domain_names])

    # Profile before calibration
    dead_mask_pre_cal_200, _, _, _ = profile_dead_mask(composed_B2, joint_val, seed=seed)

    # Calibrate 200 steps
    full_capsule_calibrate(composed_B2, joint_train, steps=200,
                           lr=LR * 0.1, seed=seed)

    # Profile after calibration
    dead_mask_post_cal_200, death_rate_200, _, _ = profile_dead_mask(
        composed_B2, joint_val, seed=seed)

    revival_200, n_rev_200, n_dead_200, n_nd_200 = measure_revival_rate(
        dead_mask_pre_cal_200, dead_mask_post_cal_200)
    meta["B2_revival_rate_200"] = revival_200
    meta["B2_death_rate_200"] = death_rate_200
    print(f"    Revival during 200-step calibration: {revival_200:.1%} "
          f"({n_rev_200}/{n_dead_200} revived)")

    # Prune
    freqs_B2 = profile_activations(composed_B2, joint_val, n_batches=20, batch_size=32, seed=seed)
    masks_B2 = identify_dead_capsules(freqs_B2, threshold=0.0)
    prune_model(composed_B2, masks_B2, verbose=False)
    results["B2_post_cal_200"] = _eval_domains(composed_B2, domain_datasets)

    # ============================================================
    # 7. Direct revival measurement at compose time
    #    (compose -> profile immediately, then calibrate -> profile again)
    # ============================================================
    print(f"  [7/7] Direct revival measurement...")

    composed_revival = compose_relu_models(base, [domain_models[d] for d in domain_names])
    dead_mask_t0, dr_t0, _, _ = profile_dead_mask(composed_revival, joint_val, seed=seed)

    # Calibrate 50 steps
    composed_50 = copy.deepcopy(composed_revival)
    full_capsule_calibrate(composed_50, joint_train, steps=50, lr=LR * 0.1, seed=seed)
    dead_mask_50, dr_50, _, _ = profile_dead_mask(composed_50, joint_val, seed=seed)
    rev_50, n_rev_50, n_dead_50, _ = measure_revival_rate(dead_mask_t0, dead_mask_50)

    # Calibrate 100 steps
    composed_100 = copy.deepcopy(composed_revival)
    full_capsule_calibrate(composed_100, joint_train, steps=100, lr=LR * 0.1, seed=seed)
    dead_mask_100, dr_100, _, _ = profile_dead_mask(composed_100, joint_val, seed=seed)
    rev_100, n_rev_100, n_dead_100, _ = measure_revival_rate(dead_mask_t0, dead_mask_100)

    # Calibrate 200 steps
    composed_200 = copy.deepcopy(composed_revival)
    full_capsule_calibrate(composed_200, joint_train, steps=200, lr=LR * 0.1, seed=seed)
    dead_mask_200, dr_200, _, _ = profile_dead_mask(composed_200, joint_val, seed=seed)
    rev_200_direct, n_rev_200d, n_dead_200d, _ = measure_revival_rate(dead_mask_t0, dead_mask_200)

    meta["revival_trajectory"] = {
        0: {"death_rate": dr_t0, "revival_rate": 0.0},
        50: {"death_rate": dr_50, "revival_rate": rev_50, "n_revived": n_rev_50, "n_dead_anchor": n_dead_50},
        100: {"death_rate": dr_100, "revival_rate": rev_100, "n_revived": n_rev_100, "n_dead_anchor": n_dead_100},
        200: {"death_rate": dr_200, "revival_rate": rev_200_direct, "n_revived": n_rev_200d, "n_dead_anchor": n_dead_200d},
    }
    print(f"    Revival trajectory: t=0 death={dr_t0:.1%}, "
          f"t=50 rev={rev_50:.1%}, t=100 rev={rev_100:.1%}, t=200 rev={rev_200_direct:.1%}")

    return results, meta


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []
    all_meta = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        results, meta = run_pipeline_experiment(seed=seed)
        all_results.append(results)
        all_meta.append(meta)

        # Per-seed summary
        print(f"\n  Per-seed summary:")
        for method, vals in results.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.4f}" for d in domains)
            print(f"    {method:<25} avg={vals['avg']:.4f} ({detail})")

    # ============================================================
    # 3-Seed Aggregate
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])

    # Use Pipeline A as the baseline for kill criterion
    A_mean = statistics.mean([r["A_pre_comp_prune"]["avg"] for r in all_results])

    print(f"\n  {'Method':<30} {'Avg Loss':>10} {'Std':>8} {'vs Joint':>10} {'vs Pipe A':>10}")
    print("  " + "-" * 72)

    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_A = ((mean - A_mean) / A_mean) * 100
        print(f"  {method:<30} {mean:>10.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_A:>+9.1f}%")

    # ============================================================
    # Revival Rate Analysis (Kill Criterion 2)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Revival Rate During Calibration (Kill Criterion 2)")
    print(f"{'='*70}")

    revival_100 = [m["B_revival_rate"] for m in all_meta]
    revival_200 = [m["B2_revival_rate_200"] for m in all_meta]

    print(f"\n  Revival during 100-step calibration:")
    print(f"    Mean: {statistics.mean(revival_100):.1%} (per-seed: "
          f"{', '.join(f'{r:.1%}' for r in revival_100)})")
    if len(revival_100) > 1:
        print(f"    Std:  {statistics.stdev(revival_100):.1%}")

    print(f"\n  Revival during 200-step calibration:")
    print(f"    Mean: {statistics.mean(revival_200):.1%} (per-seed: "
          f"{', '.join(f'{r:.1%}' for r in revival_200)})")

    # Revival trajectory
    print(f"\n  Revival Trajectory (3-seed mean):")
    print(f"  {'Cal Steps':>10} | {'Death Rate':>12} | {'Revival Rate':>14}")
    print(f"  " + "-" * 45)

    for S in [0, 50, 100, 200]:
        drs = [m["revival_trajectory"][S]["death_rate"] for m in all_meta]
        if S == 0:
            print(f"  {S:>10} | {statistics.mean(drs):>11.1%} |       (anchor)")
        else:
            revs = [m["revival_trajectory"][S]["revival_rate"] for m in all_meta]
            print(f"  {S:>10} | {statistics.mean(drs):>11.1%} | {statistics.mean(revs):>13.1%}")

    # ============================================================
    # Pruning Statistics
    # ============================================================
    print(f"\n{'='*70}")
    print("  Pruning Statistics (3-seed mean)")
    print(f"{'='*70}")

    for label, key_before, key_alive in [
        ("A (pre-composition)", "A_total_before", "A_total_alive"),
        ("B (post-calibration, 100 steps)", "B_total_before", "B_total_alive"),
        ("C (compose-then-prune, pre-cal)", "C_total_before", "C_total_alive"),
    ]:
        befores = [m[key_before] for m in all_meta]
        alives = [m[key_alive] for m in all_meta]
        pcts = [(b - a) / b * 100 for b, a in zip(befores, alives)]
        print(f"\n  Pipeline {label}:")
        print(f"    Total before: {statistics.mean(befores):.0f}")
        print(f"    Alive after:  {statistics.mean(alives):.0f}")
        print(f"    Pruned:       {statistics.mean(pcts):.1f}%")

    # ============================================================
    # Kill Threshold Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    B_mean = statistics.mean([r["B_post_cal_pruned"]["avg"] for r in all_results])
    C_mean = statistics.mean([r["C_compose_then_prune"]["avg"] for r in all_results])
    D_mean = statistics.mean([r["D_no_prune"]["avg"] for r in all_results])
    B2_mean = statistics.mean([r["B2_post_cal_200"]["avg"] for r in all_results])

    delta_B = ((B_mean - A_mean) / A_mean) * 100
    delta_C = ((C_mean - A_mean) / A_mean) * 100
    delta_B2 = ((B2_mean - A_mean) / A_mean) * 100

    print(f"\n  Kill Criterion 1: post-calibration pruning degrades >2% vs pre-calibration pruning")
    print(f"\n  Pipeline A (pre-comp prune, baseline):   avg loss = {A_mean:.4f}")
    print(f"  Pipeline B (post-cal prune, 100 steps):   avg loss = {B_mean:.4f} ({delta_B:+.2f}% vs A)")
    print(f"  Pipeline C (compose-then-prune, pre-cal): avg loss = {C_mean:.4f} ({delta_C:+.2f}% vs A)")
    print(f"  Pipeline B2 (post-cal prune, 200 steps):  avg loss = {B2_mean:.4f} ({delta_B2:+.2f}% vs A)")
    print(f"  Pipeline D (no prune, control):            avg loss = {D_mean:.4f}")

    # Kill criterion 1 check
    kill_1_status = "KILL" if delta_B > 2.0 else "PASS"
    print(f"\n  {kill_1_status}: Pipeline B vs A: {delta_B:+.2f}% (threshold: 2%)")

    # Kill criterion 2 check
    mean_revival = statistics.mean(revival_100)
    kill_2_status = "KILL" if mean_revival > 0.05 else "PASS"
    print(f"  {kill_2_status}: Revival rate at 100-step cal: {mean_revival:.1%} (threshold: 5%)")

    # ============================================================
    # Overall Verdict
    # ============================================================
    print(f"\n{'='*70}")
    print("  Overall Verdict")
    print(f"{'='*70}")

    if kill_1_status == "PASS" and kill_2_status == "PASS":
        print(f"\n  PASS. Post-calibration pruning is safe.")
        print(f"  - Quality: {delta_B:+.2f}% vs pre-composition pruning (within 2% threshold)")
        print(f"  - Revival: {mean_revival:.1%} during 100-step calibration (below 5% threshold)")
        print(f"\n  The full pipeline compose -> calibrate -> profile -> prune is validated.")
        print(f"  Dead capsules stay dead during calibration (revival {mean_revival:.1%}),")
        print(f"  so the post-calibration dead set is accurate and pruning is safe.")
        print(f"\n  Three equivalent pruning orderings now validated:")
        print(f"    1. Pre-composition:  profile -> prune -> compose -> calibrate (Exp prune_before_compose)")
        print(f"    2. Pre-calibration:  compose -> profile -> prune -> calibrate (Exp dead_capsule_pruning)")
        print(f"    3. Post-calibration: compose -> calibrate -> profile -> prune (this experiment)")
    elif kill_1_status == "KILL":
        print(f"\n  KILL (criterion 1). Post-calibration pruning degrades quality by {delta_B:+.2f}%.")
        print(f"  Calibration changes the activation landscape enough that post-calibration")
        print(f"  profiling identifies a DIFFERENT dead set than pre-calibration profiling.")
    elif kill_2_status == "KILL":
        print(f"\n  KILL (criterion 2). Revival rate {mean_revival:.1%} exceeds 5% threshold.")
        print(f"  The 2.9% finding from revival_under_composition is NOT reproduced.")
        print(f"  Dead capsules are not stable during calibration.")


if __name__ == "__main__":
    main()
