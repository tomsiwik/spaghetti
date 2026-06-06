"""Pre-composition pruning pipeline at N=5 domains.

Scales the prune_before_compose experiment from N=2 to N=5 domains.
At N=2 the pipeline was validated with +0.01% delta (200x margin on 2%
threshold). At N=5, identity Jaccard drops from 0.895 to 0.792 (Exp
n5_identity_scaling). This experiment tests whether the quality
equivalence still holds at the lower identity overlap.

Two pipelines compared (same as prune_before_compose, but with 5 domains):

  Pipeline A (BASELINE -- compose-then-prune):
    1. Pretrain shared base on all data
    2. Fine-tune MLP per domain (attention frozen), 5 domains
    3. Compose by concatenating A/B weight matrices from all 5 domains
    4. Profile composed model on joint data
    5. Prune dead capsules (tau=0)
    6. Calibrate on joint data (100 steps)
    7. Evaluate

  Pipeline B (NEW -- prune-before-compose):
    1. Pretrain shared base on all data
    2. Fine-tune MLP per domain (attention frozen), 5 domains
    3. Profile each single-domain model on own-domain data (parallelizable)
    4. Prune dead capsules from each domain model independently
    5. Compose by concatenating the already-pruned A/B matrices
    6. Calibrate on joint data (100 steps, same budget as A)
    7. Evaluate

  Pipeline C (CONTROL -- compose, no prune, just calibrate):
    Steps 1-3 of Pipeline A, then calibrate without pruning.

Kill criterion:
  Pipeline B quality degrades >3% vs Pipeline A (compose-then-prune).
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


def run_pipeline_experiment_n5(seed=42):
    """Run all pipelines for one seed at N=5 domains and return results."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="quintary")

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
    assert len(domain_names) == 5, f"Expected 5 domains, got {len(domain_names)}"
    results = {}

    # ============================================================
    # 0. Joint training baseline (upper bound)
    # ============================================================
    print(f"  [0/5] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 5)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 1. Shared: Pretrain base + domain fine-tune (5 domains)
    # ============================================================
    print(f"  [1/5] Pretrain base + fine-tune per 5 domains...")
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
    # 2. Pipeline A: Compose -> Profile -> Prune -> Calibrate
    # ============================================================
    print(f"  [2/5] Pipeline A: compose-then-prune (N=5)...")
    composed_A = compose_relu_models(base, [domain_models[d] for d in domain_names])

    # Profile composed model on joint data
    freqs_A = profile_activations(composed_A, joint_val, n_batches=20, batch_size=32, seed=seed)
    masks_A = identify_dead_capsules(freqs_A, threshold=0.0)

    # Count stats before pruning
    total_before_A = sum(m.shape[0] for m in masks_A)
    total_alive_A = sum(int(mx.sum(m).item()) for m in masks_A)
    pct_pruned_A = (total_before_A - total_alive_A) / total_before_A * 100

    # Evaluate BEFORE pruning+calibration (zero-shot composed)
    results["A_before_prune"] = _eval_domains(composed_A, domain_datasets)

    # Prune
    prune_model(composed_A, masks_A, verbose=False)

    # Evaluate after pruning, before calibration
    results["A_after_prune"] = _eval_domains(composed_A, domain_datasets)

    # Calibrate
    full_capsule_calibrate(composed_A, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)

    results["A_final"] = _eval_domains(composed_A, domain_datasets)

    print(f"    Compose-then-prune: {total_before_A} -> {total_alive_A} capsules "
          f"({pct_pruned_A:.1f}% pruned)")

    # ============================================================
    # 3. Pipeline B: Profile -> Prune -> Compose -> Calibrate
    # ============================================================
    print(f"  [3/5] Pipeline B: prune-before-compose (N=5)...")

    pruned_domain_models = {}
    prune_stats_B = {}

    for d_name in domain_names:
        model_d = copy.deepcopy(domain_models[d_name])
        # Profile on own-domain validation data
        freqs_d = profile_activations(
            model_d, domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        masks_d = identify_dead_capsules(freqs_d, threshold=0.0)

        before = sum(m.shape[0] for m in masks_d)
        alive = sum(int(mx.sum(m).item()) for m in masks_d)
        pct = (before - alive) / before * 100

        prune_model(model_d, masks_d, verbose=False)
        pruned_domain_models[d_name] = model_d
        prune_stats_B[d_name] = {
            "before": before, "alive": alive, "pct_pruned": pct
        }
        print(f"    {d_name}: {before} -> {alive} capsules ({pct:.1f}% pruned)")

    # Compose the already-pruned models
    composed_B = compose_pruned_models(base, [pruned_domain_models[d] for d in domain_names])

    # Evaluate before calibration
    results["B_before_cal"] = _eval_domains(composed_B, domain_datasets)

    # Calibrate (same budget as Pipeline A)
    full_capsule_calibrate(composed_B, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)

    results["B_final"] = _eval_domains(composed_B, domain_datasets)

    total_before_B = sum(s["before"] for s in prune_stats_B.values())
    total_alive_B = sum(s["alive"] for s in prune_stats_B.values())
    pct_pruned_B = (total_before_B - total_alive_B) / total_before_B * 100

    # ============================================================
    # 4. Pipeline C: Compose -> Calibrate (no pruning, control)
    # ============================================================
    print(f"  [4/5] Pipeline C: compose + calibrate, no pruning (control)...")
    composed_C = compose_relu_models(base, [domain_models[d] for d in domain_names])
    full_capsule_calibrate(composed_C, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["C_no_prune"] = _eval_domains(composed_C, domain_datasets)

    # ============================================================
    # 5. Per-domain pruning ratio tracking
    # ============================================================
    print(f"  [5/5] Per-domain pruning ratio analysis...")

    # For pipeline A, decompose per-domain pruning
    prune_stats_A_per_domain = {}
    P = N_CAPSULES
    for k, d_name in enumerate(domain_names):
        offset = k * P
        # masks_A covers composed model; extract domain k's slice
        domain_alive = 0
        domain_total = 0
        for l_idx, m in enumerate(masks_A):
            layer_slice = m[offset:offset + P]
            domain_alive += int(mx.sum(layer_slice).item())
            domain_total += layer_slice.shape[0]
        pct_d = (domain_total - domain_alive) / domain_total * 100
        prune_stats_A_per_domain[d_name] = {
            "before": domain_total, "alive": domain_alive, "pct_pruned": pct_d
        }

    # Collect metadata
    meta = {
        "prune_stats_A": {
            "before": total_before_A, "alive": total_alive_A, "pct_pruned": pct_pruned_A
        },
        "prune_stats_A_per_domain": prune_stats_A_per_domain,
        "prune_stats_B": prune_stats_B,
        "prune_stats_B_total": {
            "before": total_before_B, "alive": total_alive_B, "pct_pruned": pct_pruned_B
        },
    }

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
        results, meta = run_pipeline_experiment_n5(seed=seed)
        all_results.append(results)
        all_meta.append(meta)

        # Print per-seed summary
        print(f"\n  Per-seed summary:")
        for method, vals in results.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.4f}" for d in domains)
            print(f"    {method:<25} avg={vals['avg']:.4f} ({detail})")

    # ============================================================
    # 3-Seed Aggregate
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results (N=5 Domains)")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<30} {'Avg Loss':>10} {'Std':>8} {'vs Joint':>10} {'vs Pipe A':>10}")
    print("  " + "-" * 72)

    A_mean = statistics.mean([r["A_final"]["avg"] for r in all_results])

    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_A = ((mean - A_mean) / A_mean) * 100
        print(f"  {method:<30} {mean:>10.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_A:>+9.1f}%")

    # ============================================================
    # Pruning statistics
    # ============================================================
    print(f"\n{'='*70}")
    print("  Pruning Statistics (3-seed mean)")
    print(f"{'='*70}")

    A_pcts = [m["prune_stats_A"]["pct_pruned"] for m in all_meta]
    B_pcts = [m["prune_stats_B_total"]["pct_pruned"] for m in all_meta]
    A_alive = [m["prune_stats_A"]["alive"] for m in all_meta]
    B_alive = [m["prune_stats_B_total"]["alive"] for m in all_meta]

    print(f"\n  Pipeline A (compose-then-prune, N=5):")
    print(f"    Capsules pruned: {statistics.mean(A_pcts):.1f}% (std={statistics.stdev(A_pcts):.1f}%)")
    print(f"    Alive capsules: {statistics.mean(A_alive):.0f}")

    print(f"\n  Pipeline B (prune-before-compose, N=5):")
    print(f"    Capsules pruned: {statistics.mean(B_pcts):.1f}% (std={statistics.stdev(B_pcts):.1f}%)")
    print(f"    Alive capsules: {statistics.mean(B_alive):.0f}")

    print(f"\n  Pruning gap (B - A): {statistics.mean(B_pcts) - statistics.mean(A_pcts):.1f}pp")

    # Per-domain pruning comparison
    print(f"\n  Per-domain pruning comparison (3-seed mean):")
    domain_names = list(all_meta[0]["prune_stats_B"].keys())
    print(f"  {'Domain':<8} | {'A pruned%':>10} | {'B pruned%':>10} | {'Gap':>6}")
    print("  " + "-" * 42)
    for d_name in domain_names:
        a_pcts_d = [m["prune_stats_A_per_domain"][d_name]["pct_pruned"] for m in all_meta]
        b_pcts_d = [m["prune_stats_B"][d_name]["pct_pruned"] for m in all_meta]
        a_mean = statistics.mean(a_pcts_d)
        b_mean = statistics.mean(b_pcts_d)
        print(f"  {d_name:<8} | {a_mean:>9.1f}% | {b_mean:>9.1f}% | {b_mean - a_mean:>+5.1f}pp")

    # ============================================================
    # Kill Threshold Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    B_mean = statistics.mean([r["B_final"]["avg"] for r in all_results])
    C_mean = statistics.mean([r["C_no_prune"]["avg"] for r in all_results])

    delta_B = ((B_mean - A_mean) / A_mean) * 100

    print(f"\n  Kill criterion: pre-prune-then-compose quality degrades >3% vs compose-then-prune at N=5")
    print(f"\n  Pipeline A (compose-then-prune, baseline): avg loss = {A_mean:.4f}")
    print(f"  Pipeline B (prune-before, own-domain prof):  avg loss = {B_mean:.4f} ({delta_B:+.2f}% vs A)")
    print(f"  Pipeline C (compose, no prune, calibrate):   avg loss = {C_mean:.4f} ({((C_mean - A_mean) / A_mean) * 100:+.2f}% vs A)")
    print(f"  Joint training upper bound:                  avg loss = {joint_mean:.4f}")

    status = "KILL" if delta_B > 3.0 else "PASS"
    print(f"\n  {status}: Pipeline B vs Pipeline A: {delta_B:+.2f}% (threshold: 3%)")

    # Pre-calibration comparison
    A_pre = statistics.mean([r["A_after_prune"]["avg"] for r in all_results])
    B_pre = statistics.mean([r["B_before_cal"]["avg"] for r in all_results])
    delta_pre = ((B_pre - A_pre) / A_pre) * 100
    print(f"\n  Pre-calibration comparison:")
    print(f"    Pipeline A (after prune, before cal): {A_pre:.4f}")
    print(f"    Pipeline B (after prune+compose, before cal): {B_pre:.4f} ({delta_pre:+.2f}%)")

    # Comparison with N=2 results
    print(f"\n{'='*70}")
    print("  Comparison with N=2 (prune_before_compose)")
    print(f"{'='*70}")
    print(f"\n  N=2 result: Pipeline B vs A = +0.01% (margin: 200x on 2% threshold)")
    print(f"  N=5 result: Pipeline B vs A = {delta_B:+.2f}% (threshold: 3%)")
    print(f"\n  Identity Jaccard:")
    print(f"    N=2: 0.895 (Exp 16)")
    print(f"    N=5: 0.792 (Exp n5_identity_scaling)")
    print(f"  Pruning gap (B - A):")
    print(f"    N=2: +6.0pp")
    print(f"    N=5: {statistics.mean(B_pcts) - statistics.mean(A_pcts):+.1f}pp")

    # ============================================================
    # Overall Verdict
    # ============================================================
    print(f"\n{'='*70}")
    print("  Overall Verdict")
    print(f"{'='*70}")

    if delta_B <= 3.0:
        print(f"\n  PASS. Pre-composition pruning at N=5 achieves {delta_B:+.2f}% vs")
        print(f"  compose-then-prune baseline (within 3% threshold).")
        print(f"\n  Despite lower identity Jaccard at N=5 (0.792 vs 0.895 at N=2),")
        print(f"  the pre-composition pruning pipeline remains viable. Calibration")
        print(f"  absorbs the pruning differences even at higher domain count.")
        print(f"\n  The validated protocol extends to N=5:")
        print(f"    1. Profile each single-domain model independently (parallelizable)")
        print(f"    2. Prune dead capsules pre-composition")
        print(f"    3. Compose the smaller, already-pruned models")
        print(f"    4. Calibrate on joint data (100 steps)")
    else:
        print(f"\n  KILL. Pre-composition pruning at N=5 degrades {delta_B:+.2f}% vs")
        print(f"  compose-then-prune baseline (exceeds 3% threshold).")
        print(f"\n  The lower identity Jaccard at N=5 (0.792) causes meaningful")
        print(f"  quality loss when pruning decisions are made pre-composition.")
        print(f"  At N>=5, post-composition profiling is required.")


if __name__ == "__main__":
    main()
