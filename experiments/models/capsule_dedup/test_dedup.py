"""Capsule Deduplication experiment.

Tests whether cosine-similarity-based deduplication of redundant capsules
after composition preserves quality while reducing parameter count.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose by concatenating A and B weight matrices from both domains
  4. Deduplicate: identify pairs with cos(a_i, a_j) > threshold, merge
  5. Evaluate: does deduplicated model match unmerged quality?

Controls:
  - Unmerged concatenation (the +4.3% zero-shot baseline)
  - Weight averaging (the +1.5% baseline)
  - Joint training (upper bound)

Sweep: threshold in {0.90, 0.95, 0.99}, cross-pool-only vs all-pairs
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
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE, BATCH_SIZE, LR,
)
from .capsule_dedup import (
    cosine_similarity_matrix,
    find_redundant_clusters,
    find_all_redundant_clusters,
    merge_capsules,
    deduplicate_composed_model,
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


def check_capsule_death(model, dataset, batch_size=32, n_batches=10) -> dict:
    """Check for dead capsules (never fire) after deduplication.

    Returns per-layer dead capsule count and percentage.
    """
    rng = random.Random(0)
    results = []

    for l_idx, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        P = pool.n_capsules

        # Run a bunch of batches and accumulate which capsules ever fire
        ever_fired = mx.zeros(P)

        for _ in range(n_batches):
            inputs, targets = dataset.get_batch(batch_size, rng)
            # Run forward up to this layer to get the input
            B_size, T = inputs.shape
            pos = mx.arange(T)
            x = model.wte(inputs) + model.wpe(pos)
            x = model.norm0(x)
            for prev_layer in model.layers[:l_idx]:
                x = prev_layer(x)

            # Get pre-activation at this layer
            x_norm = layer.norm2(x + layer.attn(layer.norm1(x)))
            h = nn.relu(pool.A(x_norm))  # (B, T, P)
            fired = (h > 0).astype(mx.float32)  # (B, T, P)
            ever_fired = ever_fired + mx.sum(fired, axis=(0, 1))
            mx.eval(ever_fired)

        n_dead = int(mx.sum(ever_fired == 0).item())
        results.append({
            "layer": l_idx,
            "P": P,
            "n_dead": n_dead,
            "pct_dead": n_dead / P * 100 if P > 0 else 0,
        })

    return results


def run_dedup_experiment(seed=42):
    """Run the full deduplication experiment.

    Returns dict of method -> {domain: loss, "avg": avg_loss} plus
    dedup statistics.
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
    dedup_stats = {}
    domain_names = list(domain_datasets.keys())

    # ============================================================
    # 1. Joint training baseline
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
    # 3. Unmerged concatenation (zero-shot baseline)
    # ============================================================
    print("  [3/7] Unmerged concatenation (zero-shot)...")
    composed = compose_relu_models(base, domain_models)
    results["concat_zero_shot"] = _eval_domains(composed, domain_datasets)

    # ============================================================
    # 4. Weight averaging baseline
    # ============================================================
    print("  [4/7] Weight averaging baseline...")
    averaged = weight_average_relu_models(base, domain_models)
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)

    # ============================================================
    # 5. Deduplication sweep: thresholds x modes
    # ============================================================
    thresholds = [0.90, 0.95, 0.99]
    modes = [
        ("cross_only", True),
        ("all_pairs", False),
    ]

    for tau in thresholds:
        for mode_name, cross_only in modes:
            key = f"dedup_{mode_name}_t{tau:.2f}"
            print(f"  [5/7] Dedup {mode_name} tau={tau:.2f}...")
            model_dedup = copy.deepcopy(composed)
            pool_sizes = [N_CAPSULES, N_CAPSULES]

            stats = deduplicate_composed_model(
                model_dedup,
                threshold=tau,
                pool_sizes=pool_sizes if cross_only else None,
                cross_pool_only=cross_only,
                verbose=True,
            )
            mx.eval(model_dedup.parameters())

            results[key] = _eval_domains(model_dedup, domain_datasets)
            dedup_stats[key] = stats

            # Check for capsule death
            death = check_capsule_death(model_dedup, joint_val)
            dedup_stats[key]["capsule_death"] = death
            total_dead = sum(d["n_dead"] for d in death)
            total_caps = sum(d["P"] for d in death)
            print(f"    Dead capsules: {total_dead}/{total_caps} "
                  f"({total_dead/total_caps*100:.1f}%)")

    # ============================================================
    # 6. Dedup + calibration (best threshold)
    # ============================================================
    print("  [6/7] Dedup (tau=0.95, cross-only) + full calibration...")
    model_dedup_cal = copy.deepcopy(composed)
    deduplicate_composed_model(
        model_dedup_cal,
        threshold=0.95,
        pool_sizes=[N_CAPSULES, N_CAPSULES],
        cross_pool_only=True,
        verbose=False,
    )
    mx.eval(model_dedup_cal.parameters())

    # Full capsule calibration: unfreeze capsule weights, train on joint data
    _freeze_attention(model_dedup_cal)
    train(model_dedup_cal, joint_train, steps=100,
          batch_size=BATCH_SIZE, lr=LR * 0.1, seed=seed, log_every=9999)
    model_dedup_cal.unfreeze()
    results["dedup_calibrated"] = _eval_domains(model_dedup_cal, domain_datasets)

    # ============================================================
    # 7. Parameter counts
    # ============================================================
    print("  [7/7] Parameter counts...")
    param_counts = {
        "joint": count_params(model_joint),
        "concat": count_params(composed),
        "weight_avg": count_params(averaged),
    }
    # Add dedup param counts
    for tau in thresholds:
        for mode_name, cross_only in modes:
            key = f"dedup_{mode_name}_t{tau:.2f}"
            # Reconstruct to count (we deepcopied for eval)
            m = copy.deepcopy(composed)
            deduplicate_composed_model(
                m, threshold=tau,
                pool_sizes=[N_CAPSULES, N_CAPSULES] if cross_only else None,
                cross_pool_only=cross_only,
                verbose=False,
            )
            mx.eval(m.parameters())
            param_counts[key] = count_params(m)
            param_counts[key + "_capsule"] = count_capsule_params(m)

    return results, dedup_stats, param_counts


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
        r, s, p = run_dedup_experiment(seed=seed)
        all_results.append(r)
        all_stats.append(s)
        all_params.append(p)

        # Per-seed summary
        for method, vals in r.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains)
            print(f"  {method:<35} avg={vals['avg']:.4f} ({detail})")

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])
    concat_mean = statistics.mean([r["concat_zero_shot"]["avg"] for r in all_results])
    wavg_mean = statistics.mean([r["weight_avg"]["avg"] for r in all_results])

    print(f"\n  {'Method':<35} {'avg':>8} {'std':>8} {'vs joint':>10} {'vs concat':>10}")
    print("  " + "-" * 80)
    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        print(f"  {method:<35} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_concat:>+9.1f}%")

    # ============================================================
    # Redundancy statistics
    # ============================================================
    print(f"\n  Deduplication Statistics (3-seed mean)")
    print("  " + "-" * 60)

    for key in sorted(all_stats[0].keys()):
        removals = [s[key]["pct_capsules_removed"] for s in all_stats]
        mean_removal = statistics.mean(removals)
        per_layer = []
        for l in range(4):
            layer_removals = [s[key]["per_layer"][l]["n_removed"] for s in all_stats]
            per_layer.append(statistics.mean(layer_removals))

        print(f"  {key:<35} removal={mean_removal:.1f}%  "
              f"per-layer=[{', '.join(f'{v:.0f}' for v in per_layer)}]")

        # Cross-pool similarity stats (if available)
        cross_sims = []
        for s in all_stats:
            for layer_s in s[key]["per_layer"]:
                if "cross_sim_mean" in layer_s:
                    cross_sims.append(layer_s["cross_sim_mean"])
        if cross_sims:
            print(f"    cross-pool cos mean: {statistics.mean(cross_sims):.4f}")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n  Kill Threshold Analysis")
    print("  " + "-" * 60)

    # Kill 1: dedup > 5% worse than unmerged concatenation
    # Kill 2: dedup worse than weight averaging
    # Kill 3: >20% capsule death
    # Kill 4: <10% of capsules have cos > 0.95

    best_dedup = None
    best_dedup_loss = 999

    for method in methods:
        if not method.startswith("dedup_"):
            continue
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        vs_concat = ((mean - concat_mean) / concat_mean) * 100
        vs_wavg = ((mean - wavg_mean) / wavg_mean) * 100
        vs_joint = ((mean - joint_mean) / joint_mean) * 100

        kill1 = "KILL" if vs_concat > 5.0 else "OK"
        kill2 = "KILL" if mean > wavg_mean else "OK"

        # Capsule death
        total_dead_pct = 0
        if method in all_stats[0]:
            deaths = []
            for s in all_stats:
                if "capsule_death" in s[method]:
                    total_d = sum(d["n_dead"] for d in s[method]["capsule_death"])
                    total_c = sum(d["P"] for d in s[method]["capsule_death"])
                    deaths.append(total_d / total_c * 100 if total_c > 0 else 0)
            if deaths:
                total_dead_pct = statistics.mean(deaths)
        kill3 = "KILL" if total_dead_pct > 20 else "OK"

        # Redundancy amount
        removal_pcts = []
        if method in all_stats[0]:
            removal_pcts = [s[method]["pct_capsules_removed"] for s in all_stats]
        avg_removal = statistics.mean(removal_pcts) if removal_pcts else 0
        kill4_result = "N/A"
        if "t0.95" in method:
            kill4_result = "KILL" if avg_removal < 2.5 else "OK"  # 10% pairs ~ 2.5% capsules removed

        print(f"  {method:<35}")
        print(f"    vs concat:    {vs_concat:>+6.1f}% {'  *** '+kill1 if kill1=='KILL' else '  '+kill1}")
        print(f"    vs weight_avg:{vs_wavg:>+6.1f}% {'  *** '+kill2 if kill2=='KILL' else '  '+kill2}")
        print(f"    dead capsules:{total_dead_pct:>5.1f}% {'  *** '+kill3 if kill3=='KILL' else '  '+kill3}")
        if kill4_result != "N/A":
            print(f"    removal rate: {avg_removal:>5.1f}% {'  *** '+kill4_result if kill4_result=='KILL' else '  '+kill4_result}")
        print(f"    vs joint:     {vs_joint:>+6.1f}%")

        if mean < best_dedup_loss and "calibrated" not in method:
            best_dedup_loss = mean
            best_dedup = method

    # ============================================================
    # Parameter savings summary
    # ============================================================
    print(f"\n  Parameter Counts (seed 42)")
    print("  " + "-" * 60)
    p = all_params[0]
    for key in sorted(p.keys()):
        if "_capsule" not in key:
            print(f"  {key:<35} {p[key]:>8,} params")

    print(f"\n  Best dedup method: {best_dedup}")
    print(f"  Best dedup loss:   {best_dedup_loss:.4f}")
    print(f"  Joint baseline:    {joint_mean:.4f}")
    print(f"  Concat baseline:   {concat_mean:.4f}")
    print(f"  Weight avg:        {wavg_mean:.4f}")


if __name__ == "__main__":
    main()
