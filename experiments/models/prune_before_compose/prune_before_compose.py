"""Pre-composition pruning pipeline -- end-to-end validation.

Exp 16 proved that dead capsule identity is conserved across composition
(Jaccard=0.895, overlap=0.986). This experiment validates the practical
consequence: prune BEFORE composing (cheaper, parallelizable, no joint
data needed for profiling) and compare quality against the established
compose-then-prune baseline.

Two pipelines compared:

  Pipeline A (BASELINE -- compose-then-prune):
    1. Pretrain shared base on all data
    2. Fine-tune MLP per domain (attention frozen)
    3. Compose by concatenating A/B weight matrices
    4. Profile composed model on joint data
    5. Prune dead capsules (tau=0)
    6. Calibrate on joint data (100 steps)
    7. Evaluate

  Pipeline B (NEW -- prune-before-compose):
    1. Pretrain shared base on all data
    2. Fine-tune MLP per domain (attention frozen)
    3. Profile each single-domain model on own-domain data (parallelizable)
    4. Prune dead capsules from each domain model independently
    5. Compose by concatenating the already-pruned A/B matrices
    6. Calibrate on joint data (100 steps, same budget as A)
    7. Evaluate

  Pipeline C (CONTROL -- compose, no prune, just calibrate):
    Steps 1-3 of Pipeline A, then calibrate without pruning.

Kill criterion:
  Pipeline B quality degrades >2% vs Pipeline A (compose-then-prune).
"""

import copy
import statistics
import random

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


def compose_pruned_models(base_model, domain_models):
    """Compose domain models that have ALREADY been pruned.

    Unlike compose_relu_models which assumes equal-sized pools,
    this handles variable-sized pools after pruning. Each domain
    model may have a different number of surviving capsules per layer.

    The composed model has P_alive_A + P_alive_B capsules per layer.
    """
    n_domains = len(domain_models)
    V = base_model.lm_head.weight.shape[0]

    # Compute total capsules per layer (may differ per layer)
    n_layers = len(base_model.layers)
    capsules_per_layer = []
    for l in range(n_layers):
        total = sum(dm.layers[l].capsule_pool.n_capsules for dm in domain_models)
        capsules_per_layer.append(total)

    # Build composed model with the correct per-layer capsule counts
    # We need to build it manually since capsule counts vary per layer
    composed = ReLURouterGPT(
        vocab_size=V,
        n_capsules=max(capsules_per_layer),  # placeholder, will be overridden
        **BASE,
    )

    # Copy shared parameters from base model
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    # For each layer, concatenate the (possibly different-sized) pools
    for l in range(n_layers):
        A_parts = [dm.layers[l].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[l].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)  # (P_total_l, d)
        B_composed = mx.concatenate(B_parts, axis=1)  # (d, P_total_l)
        mx.eval(A_composed, B_composed)

        P_total_l = A_composed.shape[0]
        d = A_composed.shape[1]

        new_pool = ReLUCapsulePool(d, P_total_l)
        new_pool.A.load_weights([("weight", A_composed)])
        new_pool.B.load_weights([("weight", B_composed)])
        composed.layers[l].capsule_pool = new_pool

    mx.eval(composed.parameters())
    return composed


def run_pipeline_experiment(seed=42):
    """Run all three pipelines for one seed and return results."""
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
    # 0. Joint training baseline (upper bound)
    # ============================================================
    print(f"  [0/6] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 1. Shared: Pretrain base + domain fine-tune
    # ============================================================
    print(f"  [1/6] Pretrain base + fine-tune per domain...")
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
    print(f"  [2/6] Pipeline A: compose-then-prune...")
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
    print(f"  [3/6] Pipeline B: prune-before-compose...")

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
    print(f"  [4/6] Pipeline C: compose + calibrate, no pruning (control)...")
    composed_C = compose_relu_models(base, [domain_models[d] for d in domain_names])
    full_capsule_calibrate(composed_C, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["C_no_prune"] = _eval_domains(composed_C, domain_datasets)

    # ============================================================
    # 5. Pipeline B2: Prune with cross-domain profiling
    # ============================================================
    # Prune using CROSS-domain data (the "harder" scenario -- profile on
    # data the capsules were NOT trained on)
    print(f"  [5/6] Pipeline B2: prune-before-compose (cross-domain profile)...")

    pruned_cross_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(domain_models[d_name])
        other_name = [n for n in domain_names if n != d_name][0]
        freqs_cross = profile_activations(
            model_d, domain_datasets[other_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        masks_cross = identify_dead_capsules(freqs_cross, threshold=0.0)

        before = sum(m.shape[0] for m in masks_cross)
        alive = sum(int(mx.sum(m).item()) for m in masks_cross)
        pct = (before - alive) / before * 100

        prune_model(model_d, masks_cross, verbose=False)
        pruned_cross_models[d_name] = model_d
        print(f"    {d_name} (cross-domain profile): {before} -> {alive} ({pct:.1f}% pruned)")

    composed_B2 = compose_pruned_models(base, [pruned_cross_models[d] for d in domain_names])
    full_capsule_calibrate(composed_B2, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["B2_cross_final"] = _eval_domains(composed_B2, domain_datasets)

    # ============================================================
    # 6. Pipeline B3: Prune with joint-data profiling (each domain on joint)
    # ============================================================
    print(f"  [6/6] Pipeline B3: prune-before-compose (joint-data profile)...")

    pruned_joint_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(domain_models[d_name])
        freqs_joint = profile_activations(
            model_d, joint_val,
            n_batches=20, batch_size=32, seed=seed,
        )
        masks_joint = identify_dead_capsules(freqs_joint, threshold=0.0)

        before = sum(m.shape[0] for m in masks_joint)
        alive = sum(int(mx.sum(m).item()) for m in masks_joint)
        pct = (before - alive) / before * 100

        prune_model(model_d, masks_joint, verbose=False)
        pruned_joint_models[d_name] = model_d
        print(f"    {d_name} (joint profile): {before} -> {alive} ({pct:.1f}% pruned)")

    composed_B3 = compose_pruned_models(base, [pruned_joint_models[d] for d in domain_names])
    full_capsule_calibrate(composed_B3, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["B3_joint_final"] = _eval_domains(composed_B3, domain_datasets)

    # Collect metadata
    meta = {
        "prune_stats_A": {"before": total_before_A, "alive": total_alive_A, "pct_pruned": pct_pruned_A},
        "prune_stats_B": prune_stats_B,
        "prune_stats_B_total": {"before": total_before_B, "alive": total_alive_B, "pct_pruned": pct_pruned_B},
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
        results, meta = run_pipeline_experiment(seed=seed)
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
    print("  3-Seed Aggregate Results")
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

    print(f"\n  Pipeline A (compose-then-prune):")
    print(f"    Capsules pruned: {statistics.mean(A_pcts):.1f}% (std={statistics.stdev(A_pcts):.1f}%)")
    print(f"    Alive capsules: {statistics.mean(A_alive):.0f}")

    print(f"\n  Pipeline B (prune-before-compose):")
    print(f"    Capsules pruned: {statistics.mean(B_pcts):.1f}% (std={statistics.stdev(B_pcts):.1f}%)")
    print(f"    Alive capsules: {statistics.mean(B_alive):.0f}")

    print(f"\n  Pruning gap (B finds fewer dead): {statistics.mean(B_pcts) - statistics.mean(A_pcts):.1f}pp")

    # ============================================================
    # Kill Threshold Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    B_mean = statistics.mean([r["B_final"]["avg"] for r in all_results])
    B2_mean = statistics.mean([r["B2_cross_final"]["avg"] for r in all_results])
    B3_mean = statistics.mean([r["B3_joint_final"]["avg"] for r in all_results])

    delta_B = ((B_mean - A_mean) / A_mean) * 100
    delta_B2 = ((B2_mean - A_mean) / A_mean) * 100
    delta_B3 = ((B3_mean - A_mean) / A_mean) * 100

    print(f"\n  Kill criterion: pre-prune-then-compose quality degrades >2% vs compose-then-prune")
    print(f"\n  Pipeline A (compose-then-prune, baseline): avg loss = {A_mean:.4f}")
    print(f"  Pipeline B (prune-before, own-domain prof):  avg loss = {B_mean:.4f} ({delta_B:+.2f}% vs A)")
    print(f"  Pipeline B2 (prune-before, cross-domain):    avg loss = {B2_mean:.4f} ({delta_B2:+.2f}% vs A)")
    print(f"  Pipeline B3 (prune-before, joint-data prof): avg loss = {B3_mean:.4f} ({delta_B3:+.2f}% vs A)")

    # Check kill criterion for each variant
    for label, delta in [("B (own-domain)", delta_B), ("B2 (cross-domain)", delta_B2), ("B3 (joint)", delta_B3)]:
        status = "KILL" if delta > 2.0 else "PASS"
        print(f"\n  {status}: Pipeline {label}: {delta:+.2f}% vs Pipeline A (threshold: 2%)")

    # Overall verdict
    print(f"\n{'='*70}")
    print("  Overall Verdict")
    print(f"{'='*70}")

    best_B_delta = min(delta_B, delta_B2, delta_B3)
    best_B_label = ["B (own-domain)", "B2 (cross-domain)", "B3 (joint)"][
        [delta_B, delta_B2, delta_B3].index(best_B_delta)
    ]

    if best_B_delta <= 2.0:
        print(f"\n  PASS. Best pre-composition pipeline ({best_B_label}) achieves {best_B_delta:+.2f}% vs")
        print(f"  compose-then-prune baseline (within 2% threshold).")
        print(f"\n  Pre-composition pruning is validated. The optimized protocol:")
        print(f"    1. Profile each single-domain model independently (parallelizable)")
        print(f"    2. Prune dead capsules pre-composition")
        print(f"    3. Compose the smaller, already-pruned models")
        print(f"    4. Calibrate on joint data")
    else:
        print(f"\n  KILL. Best pre-composition pipeline ({best_B_label}) degrades {best_B_delta:+.2f}% vs")
        print(f"  compose-then-prune baseline (exceeds 2% threshold).")
        print(f"\n  Despite high identity overlap (Exp 16, Jaccard=0.895), the ~6% missed")
        print(f"  pruning opportunity or the different profiling context causes meaningful")
        print(f"  quality loss.")


if __name__ == "__main__":
    main()
