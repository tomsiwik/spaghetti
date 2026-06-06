"""Shared Layer 0 Calibrated Experiment.

Tests whether the shared Layer 0 quality advantage (1.7-3.0% vs full
concat in zero-shot) persists after 200-step calibration.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose with full concatenation (control)
  4. Compose with shared Layer 0 (average strategy, recommended by parent)
  5. Calibrate BOTH composed models for 200 steps on mixed-domain data
  6. Track loss curves during calibration to see convergence dynamics
  7. Compare: is shared L0 still better after calibration?

Kill criterion: shared Layer 0 advantage over full concat disappears
(<0.5% difference) after 200-step calibration.

Additional analysis:
  - Loss curves during calibration (does concat catch up?)
  - Zero-shot vs calibrated comparison (is calibration needed at all?)
  - Per-domain quality impact after calibration
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from ..relu_router.relu_router import ReLURouterGPT
from ..shared_layer0_pool.shared_layer0_pool import (
    compose_shared_layer0,
    compose_full_concat,
    count_params,
)


# Shared experiment config (matches parent: shared_layer0_pool)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain (composed = 256)
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
STEPS_CALIBRATE = 200  # Extended calibration (the variable under test)
BATCH_SIZE = 32
LR = 3e-3
CALIBRATION_LR = 3e-3  # Same LR for calibration
CALIBRATION_LOG_INTERVAL = 10  # Log every N steps for loss curves


def _make_relu_model(vocab_size, n_capsules=N_CAPSULES):
    """Create a ReLURouterGPT model."""
    model = ReLURouterGPT(
        vocab_size=vocab_size, n_capsules=n_capsules, **BASE,
    )
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    """Freeze everything EXCEPT capsule pool / MLP weights."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    """Evaluate model on all domains, return dict with per-domain and avg loss."""
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def calibrate_with_curve(
    model,
    domain_datasets,
    val_datasets,
    steps=STEPS_CALIBRATE,
    lr=CALIBRATION_LR,
    batch_size=BATCH_SIZE,
    seed=42,
    log_interval=CALIBRATION_LOG_INTERVAL,
):
    """Calibrate composed model's MLP weights on mixed-domain data.

    Freezes everything except capsule pools (MLP weights).
    Alternates batches from each domain.
    Returns loss curve and periodic val evaluations.

    Args:
        model: Composed model to calibrate.
        domain_datasets: dict of {name: (train_ds, val_ds)} for training.
        val_datasets: dict of {name: val_ds} for periodic evaluation.
        steps: Number of calibration steps.
        lr: Learning rate for calibration.
        batch_size: Batch size.
        seed: Random seed.
        log_interval: How often to record val loss.

    Returns:
        dict with 'train_losses', 'val_curve' (list of (step, avg_val_loss)),
        'final_val' (dict of per-domain val losses).
    """
    # Freeze everything except capsule pools
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    domain_names = list(domain_datasets.keys())
    n_domains = len(domain_names)

    train_losses = []
    val_curve = []  # (step, avg_val_loss)

    for step in range(1, steps + 1):
        # Alternate domains
        d_idx = (step - 1) % n_domains
        d_name = domain_names[d_idx]
        train_ds = domain_datasets[d_name][0]

        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        train_losses.append(loss.item())

        # Periodic val evaluation
        if step % log_interval == 0 or step == 1 or step == steps:
            avg_val = 0.0
            for vd_name, vd_ds in val_datasets.items():
                avg_val += evaluate(model, vd_ds, batch_size, n_batches=5)
            avg_val /= len(val_datasets)
            val_curve.append((step, avg_val))

    # Final evaluation
    model.unfreeze()
    final_val = {}
    for d_name, d_ds in val_datasets.items():
        final_val[d_name] = evaluate(model, d_ds, batch_size)
    final_val["avg"] = sum(v for k, v in final_val.items() if k != "avg") / len(val_datasets)

    return {
        "train_losses": train_losses,
        "val_curve": val_curve,
        "final_val": final_val,
    }


def run_experiment(seed=42):
    """Run the full calibration persistence experiment for one seed."""
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
    val_datasets = {d: domain_datasets[d][1] for d in domain_names}
    results = {}

    # ============================================================
    # 1. Joint training baseline
    # ============================================================
    print("  [1/5] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)
    print(f"    joint avg: {results['joint']['avg']:.4f}")

    # ============================================================
    # 2. Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/5] Pretrain base + fine-tune per domain...")
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
    # 3. Full concatenation: zero-shot + calibrated
    # ============================================================
    print("  [3/5] Full concatenation: zero-shot + calibrated...")

    # Zero-shot evaluation
    composed_full = compose_full_concat(base, domain_models)
    results["full_concat_zeroshot"] = _eval_domains(composed_full, domain_datasets)
    print(f"    full_concat zero-shot avg: {results['full_concat_zeroshot']['avg']:.4f}")

    # Calibrated evaluation (200 steps)
    composed_full_cal = compose_full_concat(base, domain_models)
    cal_results_full = calibrate_with_curve(
        composed_full_cal, domain_datasets, val_datasets,
        steps=STEPS_CALIBRATE, seed=seed,
    )
    results["full_concat_calibrated"] = cal_results_full["final_val"]
    results["full_concat_cal_curve"] = cal_results_full["val_curve"]
    results["full_concat_cal_train_losses"] = cal_results_full["train_losses"]
    print(f"    full_concat calibrated avg: {results['full_concat_calibrated']['avg']:.4f}")

    # ============================================================
    # 4. Shared Layer 0 (average strategy): zero-shot + calibrated
    # ============================================================
    print("  [4/5] Shared Layer 0 (average): zero-shot + calibrated...")

    # Zero-shot evaluation
    composed_shared = compose_shared_layer0(base, domain_models, strategy="average")
    results["shared_L0_zeroshot"] = _eval_domains(composed_shared, domain_datasets)
    print(f"    shared_L0 zero-shot avg: {results['shared_L0_zeroshot']['avg']:.4f}")

    # Calibrated evaluation (200 steps)
    composed_shared_cal = compose_shared_layer0(base, domain_models, strategy="average")
    cal_results_shared = calibrate_with_curve(
        composed_shared_cal, domain_datasets, val_datasets,
        steps=STEPS_CALIBRATE, seed=seed,
    )
    results["shared_L0_calibrated"] = cal_results_shared["final_val"]
    results["shared_L0_cal_curve"] = cal_results_shared["val_curve"]
    results["shared_L0_cal_train_losses"] = cal_results_shared["train_losses"]
    print(f"    shared_L0 calibrated avg: {results['shared_L0_calibrated']['avg']:.4f}")

    # ============================================================
    # 5. Weight averaging baseline (for context)
    # ============================================================
    print("  [5/5] Weight averaging baseline...")
    n_domains = len(domain_models)
    averaged = _make_relu_model(V, n_capsules=N_CAPSULES)
    base_params = dict(nn.utils.tree_flatten(base.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    averaged.load_weights(shared_weights, strict=False)
    for layer_idx in range(len(averaged.layers)):
        A_avg = sum(dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models) / n_domains
        B_avg = sum(dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models) / n_domains
        averaged.layers[layer_idx].capsule_pool.A.load_weights([("weight", A_avg)])
        averaged.layers[layer_idx].capsule_pool.B.load_weights([("weight", B_avg)])
    mx.eval(averaged.parameters())
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)
    print(f"    weight_avg avg: {results['weight_avg']['avg']:.4f}")

    # Parameter counts
    results["full_concat_params"] = count_params(composed_full)
    results["shared_L0_params"] = count_params(composed_shared)

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
    # Aggregate results
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    methods = [
        ("joint", "Joint baseline"),
        ("full_concat_zeroshot", "Full concat (zero-shot)"),
        ("shared_L0_zeroshot", "Shared L0 (zero-shot)"),
        ("full_concat_calibrated", "Full concat (calibrated)"),
        ("shared_L0_calibrated", "Shared L0 (calibrated)"),
        ("weight_avg", "Weight avg (zero-shot)"),
    ]

    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<30} {'avg':>8} {'std':>8} {'vs joint':>10}")
    print("  " + "-" * 60)
    for key, label in methods:
        avgs = [r[key]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        print(f"  {label:<30} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}%")

    # ============================================================
    # The key comparison: shared vs concat AFTER calibration
    # ============================================================
    print(f"\n{'='*70}")
    print("  KEY COMPARISON: Shared L0 vs Full Concat AFTER 200-step Calibration")
    print(f"{'='*70}")

    full_cal_avgs = [r["full_concat_calibrated"]["avg"] for r in all_results]
    shared_cal_avgs = [r["shared_L0_calibrated"]["avg"] for r in all_results]

    full_cal_mean = statistics.mean(full_cal_avgs)
    shared_cal_mean = statistics.mean(shared_cal_avgs)
    full_cal_std = statistics.stdev(full_cal_avgs)
    shared_cal_std = statistics.stdev(shared_cal_avgs)

    delta_pct = ((shared_cal_mean - full_cal_mean) / full_cal_mean) * 100

    print(f"\n  Full concat calibrated:  {full_cal_mean:.4f} +/- {full_cal_std:.4f}")
    print(f"  Shared L0 calibrated:    {shared_cal_mean:.4f} +/- {shared_cal_std:.4f}")
    print(f"  Delta:                   {delta_pct:+.2f}%")

    # Per-seed detail
    print(f"\n  Per-seed breakdown:")
    for i, seed in enumerate(seeds):
        fc = full_cal_avgs[i]
        sl = shared_cal_avgs[i]
        d = ((sl - fc) / fc) * 100
        print(f"    Seed {seed}: full_concat={fc:.4f}, shared_L0={sl:.4f}, delta={d:+.2f}%")

    # ============================================================
    # Calibration improvement analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Calibration Improvement Analysis")
    print(f"{'='*70}")

    full_zs_mean = statistics.mean([r["full_concat_zeroshot"]["avg"] for r in all_results])
    shared_zs_mean = statistics.mean([r["shared_L0_zeroshot"]["avg"] for r in all_results])

    full_improvement = ((full_cal_mean - full_zs_mean) / full_zs_mean) * 100
    shared_improvement = ((shared_cal_mean - shared_zs_mean) / shared_zs_mean) * 100
    zs_delta = ((shared_zs_mean - full_zs_mean) / full_zs_mean) * 100

    print(f"\n  Zero-shot gap (shared vs full):    {zs_delta:+.2f}%")
    print(f"  Calibrated gap (shared vs full):   {delta_pct:+.2f}%")
    print(f"  Full concat calibration gain:      {full_improvement:+.2f}%")
    print(f"  Shared L0 calibration gain:        {shared_improvement:+.2f}%")
    print(f"  Gap change after calibration:      {(delta_pct - zs_delta):+.2f}pp")

    # ============================================================
    # Loss curve analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Loss Curves During Calibration (Seed 0 example)")
    print(f"{'='*70}")

    # Use first seed's curves
    fc_curve = all_results[0]["full_concat_cal_curve"]
    sl_curve = all_results[0]["shared_L0_cal_curve"]

    print(f"\n  {'Step':>6} {'Full Concat':>14} {'Shared L0':>14} {'Delta':>10}")
    print("  " + "-" * 48)
    for (fc_step, fc_val), (sl_step, sl_val) in zip(fc_curve, sl_curve):
        d = ((sl_val - fc_val) / fc_val) * 100
        print(f"  {fc_step:>6} {fc_val:>14.4f} {sl_val:>14.4f} {d:>+9.2f}%")

    # ============================================================
    # Kill criterion
    # ============================================================
    print(f"\n{'='*70}")
    print("  KILL CRITERION ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Kill criterion: shared Layer 0 advantage disappears (<0.5%)")
    print(f"  after 200-step calibration")
    print(f"\n  Post-calibration delta: {delta_pct:+.2f}%")

    # The advantage "disappears" if |delta| < 0.5% (shared is not meaningfully
    # better or worse). Since delta is negative when shared is better:
    if delta_pct < -0.5:
        # Shared is STILL better by more than 0.5%
        print(f"  Shared L0 advantage PERSISTS ({delta_pct:+.2f}% > 0.5% threshold)")
        print(f"  Verdict: PASS (advantage persists after calibration)")
    elif delta_pct > 0.5:
        # Full concat is better after calibration
        print(f"  Full concat is BETTER after calibration ({delta_pct:+.2f}%)")
        print(f"  Verdict: KILL (sharing becomes a disadvantage after calibration)")
    else:
        # Gap is within +/- 0.5% -- advantage disappears
        print(f"  Gap is within +/- 0.5% threshold")
        print(f"  Verdict: KILL (advantage disappears after calibration)")

    # Also report whether either condition reaches joint baseline
    print(f"\n  Context: vs joint baseline ({joint_mean:.4f})")
    print(f"    Full concat calibrated: {((full_cal_mean - joint_mean)/joint_mean)*100:+.2f}% vs joint")
    print(f"    Shared L0 calibrated:   {((shared_cal_mean - joint_mean)/joint_mean)*100:+.2f}% vs joint")

    # Parameter savings (always present regardless of quality)
    full_p = statistics.mean([r["full_concat_params"] for r in all_results])
    shared_p = statistics.mean([r["shared_L0_params"] for r in all_results])
    saving_pct = (full_p - shared_p) / full_p * 100
    print(f"\n  Parameter savings (always valid): {saving_pct:.1f}% ({full_p - shared_p:.0f} params)")


if __name__ == "__main__":
    main()
