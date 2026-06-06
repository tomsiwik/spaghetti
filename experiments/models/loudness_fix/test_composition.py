"""Composition experiment: loudness fix for zero-shot ReLU Router composition.

Tests three interventions to fix the +6.6% zero-shot composition degradation:

1. Per-pool RMSNorm: Normalize each pool's output independently before summing.
   True zero-shot -- no calibration data needed.

2. Scalar calibration: Train 1 scalar per pool per layer (8 params for 2 domains).
   Diagnostic: if this matches full calibration, loudness is the sole issue.

3. Matched-magnitude training: Auxiliary loss during fine-tuning to equalize
   output norms. Enables zero-shot composition by training-time constraint.

Controls:
  - ReLU Router joint training (upper bound)
  - ReLU Router zero-shot (lower bound, the +6.6% problem)
  - ReLU Router full calibration (continued training, unfair)
  - Weight averaging (standard model merging baseline)
  - Capsule MoE with router calibration (architecture comparison)

Protocol:
  1. Pretrain base model on ALL data (300 steps)
  2. Fine-tune only MLP weights per domain (200 steps, attention frozen)
  3. Compose using each method
  4. Evaluate all methods on both domains
"""

import copy
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models import get_model
from experiments.models.relu_router.relu_router import ReLURouterGPT
from experiments.models.relu_router.test_composition import (
    compose_relu_models,
    weight_average_relu_models,
    scalar_calibrate,
    full_capsule_calibrate,
    compose_capsule_moe,
    calibrate_capsule_moe_router,
)
from .loudness_fix import (
    MatchedMagnitudeGPT,
    compose_with_rmsnorm,
)


# Shared experiment config (matches relu_router experiment)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain (composed = 256)
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3

# Capsule MoE config
CAP_MOE = dict(n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def _make_relu_model(vocab_size, n_capsules=N_CAPSULES):
    model = ReLURouterGPT(vocab_size=vocab_size, n_capsules=n_capsules, **BASE)
    mx.eval(model.parameters())
    return model


def _make_matched_model(vocab_size, n_capsules=N_CAPSULES, mag_coeff=1.0):
    model = MatchedMagnitudeGPT(
        vocab_size=vocab_size, n_capsules=n_capsules,
        mag_coeff=mag_coeff, **BASE,
    )
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def measure_pool_rms(model, dataset, batch_size=32, n_batches=5, seed=42):
    """Measure per-layer output RMS of a model's capsule pools."""
    rng = random.Random(seed)
    layer_rms = [[] for _ in model.layers]

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)
            x_norm = layer.norm2(x)
            pool_out = layer.capsule_pool(x_norm)
            rms = mx.sqrt(mx.mean(pool_out * pool_out)).item()
            layer_rms[l_idx].append(rms)
            x = x + pool_out

        mx.eval(x)

    return [statistics.mean(rms_list) for rms_list in layer_rms]


def run_composition_experiment(seed=42):
    """Run the full composition experiment with all controls.

    Returns dict of {method_name: {domain: loss, "avg": avg_loss}}.
    Also returns metadata dict with diagnostic info.
    """
    t0 = time.time()
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

    V = tokenizer.vocab_size
    results = {}
    meta = {}
    domain_names = list(domain_datasets.keys())

    # ============================================================
    # 1. RELU ROUTER: Joint training baseline
    # ============================================================
    print("  [1/10] ReLU Router: joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["relu_joint"] = _eval_domains(model_joint, domain_datasets)

    # Measure joint model's per-layer RMS for reference
    joint_rms = measure_pool_rms(model_joint, joint_train, seed=seed)
    meta["joint_rms"] = joint_rms

    # ============================================================
    # 2. RELU ROUTER: Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/10] ReLU Router: pretrain base...")
    base_relu = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base_relu, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Measure pretrained base RMS
    base_rms = measure_pool_rms(base_relu, joint_train, seed=seed)
    meta["base_rms"] = base_rms

    print("        Fine-tuning per domain...")
    relu_domain_models = []
    domain_rms = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base_relu)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        relu_domain_models.append(model_d)
        domain_rms[d_name] = measure_pool_rms(model_d, domain_datasets[d_name][0], seed=seed)

    meta["domain_rms"] = domain_rms

    # Print RMS diagnostic
    print(f"        Base RMS per layer: {['%.4f' % r for r in base_rms]}")
    for d_name in domain_names:
        print(f"        {d_name} RMS per layer: {['%.4f' % r for r in domain_rms[d_name]]}")

    # ============================================================
    # 3. RELU ROUTER: Zero-shot composition (the +6.6% problem)
    # ============================================================
    print("  [3/10] ReLU Router: zero-shot composition...")
    composed_relu = compose_relu_models(base_relu, relu_domain_models)
    results["relu_zero_shot"] = _eval_domains(composed_relu, domain_datasets)

    # ============================================================
    # 4. INTERVENTION 1: Per-pool RMSNorm composition (TRUE zero-shot)
    # ============================================================
    print("  [4/10] Loudness Fix: per-pool RMSNorm (true zero-shot)...")
    composed_rmsnorm = compose_with_rmsnorm(
        base_relu, relu_domain_models, vocab_size=V,
        n_capsules_per_pool=N_CAPSULES,
        n_embd=BASE["n_embd"], n_head=BASE["n_head"],
        n_layer=BASE["n_layer"], block_size=BASE["block_size"],
    )
    results["rmsnorm_zero_shot"] = _eval_domains(composed_rmsnorm, domain_datasets)

    # Also try with different target_rms values
    for target_rms in [0.25, 0.5, 1.0]:
        key = f"rmsnorm_t{target_rms:.2f}"
        composed_t = compose_with_rmsnorm(
            base_relu, relu_domain_models, vocab_size=V,
            n_capsules_per_pool=N_CAPSULES, target_rms=target_rms,
            n_embd=BASE["n_embd"], n_head=BASE["n_head"],
            n_layer=BASE["n_layer"], block_size=BASE["block_size"],
        )
        results[key] = _eval_domains(composed_t, domain_datasets)

    # ============================================================
    # 5. INTERVENTION 2: Scalar calibration (diagnostic)
    # ============================================================
    print("  [5/10] Loudness Fix: scalar-only calibration (8 params)...")
    composed_scalar = compose_relu_models(base_relu, relu_domain_models)
    learned_scales = scalar_calibrate(
        composed_scalar,
        domain_datasets[domain_names[0]][0],
        domain_datasets[domain_names[1]][0],
        N_CAPSULES, steps=CALIBRATION_STEPS, lr=1e-2, seed=seed,
    )
    results["scalar_cal"] = _eval_domains(composed_scalar, domain_datasets)
    meta["learned_scales"] = learned_scales
    print(f"        Learned scales: {['%.3f' % s for s in learned_scales]}")

    # ============================================================
    # 6. RELU ROUTER: Full capsule calibration (unfair, for reference)
    # ============================================================
    print("  [6/10] ReLU Router: full capsule calibration (continued training)...")
    composed_full = compose_relu_models(base_relu, relu_domain_models)
    full_capsule_calibrate(composed_full, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["full_cal"] = _eval_domains(composed_full, domain_datasets)

    # ============================================================
    # 7. RELU ROUTER: Weight averaging baseline
    # ============================================================
    print("  [7/10] ReLU Router: weight averaging baseline...")
    averaged = weight_average_relu_models(base_relu, relu_domain_models)
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)

    # ============================================================
    # 8. INTERVENTION 3: Matched-magnitude training + zero-shot composition
    # ============================================================
    print("  [8/10] Matched-Magnitude: pretrain + set target + fine-tune...")
    base_matched = _make_matched_model(V, n_capsules=N_CAPSULES, mag_coeff=1.0)
    train(base_matched, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Set target RMS from pretrained base
    target_rms_values = base_matched.set_target_rms(joint_train, seed=seed)
    meta["matched_target_rms"] = target_rms_values
    print(f"        Target RMS per layer: {['%.4f' % r for r in target_rms_values]}")

    # Fine-tune per domain with magnitude loss active
    matched_domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base_matched)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        matched_domain_models.append(model_d)

    # Measure post-finetune RMS to see if magnitude loss worked
    matched_post_rms = {}
    for d_idx, d_name in enumerate(domain_names):
        matched_post_rms[d_name] = measure_pool_rms(
            matched_domain_models[d_idx], domain_datasets[d_name][0], seed=seed
        )
    meta["matched_post_rms"] = matched_post_rms
    for d_name in domain_names:
        print(f"        {d_name} post-FT RMS: {['%.4f' % r for r in matched_post_rms[d_name]]}")

    # Zero-shot composition of matched-magnitude models
    print("        Composing matched-magnitude models (zero-shot)...")
    # Use standard concatenation (same as relu_router)
    # Need to compose MatchedMagnitudeGPT models -- they have same structure as ReLURouterGPT
    composed_matched = _compose_matched_models(base_matched, matched_domain_models, V)
    results["matched_zero_shot"] = _eval_domains(composed_matched, domain_datasets)

    # Also try matched-magnitude + RMSNorm (belt and suspenders)
    print("        Composing matched-magnitude with RMSNorm...")
    composed_matched_rmsnorm = compose_with_rmsnorm(
        base_matched, matched_domain_models, vocab_size=V,
        n_capsules_per_pool=N_CAPSULES,
        n_embd=BASE["n_embd"], n_head=BASE["n_head"],
        n_layer=BASE["n_layer"], block_size=BASE["block_size"],
    )
    results["matched_rmsnorm"] = _eval_domains(composed_matched_rmsnorm, domain_datasets)

    # ============================================================
    # 9. CAPSULE MOE: Joint baseline
    # ============================================================
    print("  [9/10] Capsule MoE: joint training baseline...")
    cap_joint = get_model("capsule_moe", vocab_size=V,
                          n_groups=CAP_MOE["n_groups"] * 2,
                          n_capsules_per_group=CAP_MOE["n_capsules_per_group"],
                          top_k_groups=CAP_MOE["top_k_groups"] * 2,
                          **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer", "block_size"]})
    mx.eval(cap_joint.parameters())
    train(cap_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["cap_joint"] = _eval_domains(cap_joint, domain_datasets)

    # ============================================================
    # 10. CAPSULE MOE: Composition with router calibration
    # ============================================================
    print("  [10/10] Capsule MoE: composition + router calibration...")
    base_cap = get_model("capsule_moe", vocab_size=V, **CAP_MOE,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer", "block_size"]})
    mx.eval(base_cap.parameters())
    train(base_cap, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    cap_domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base_cap)
        model_d.freeze()
        for layer in model_d.layers:
            for group in layer.capsule_pool.groups:
                group.unfreeze()
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        cap_domain_models.append(model_d)

    composed_cap = compose_capsule_moe(base_cap, cap_domain_models, V)
    mx.eval(composed_cap.parameters())
    calibrate_capsule_moe_router(
        composed_cap,
        domain_datasets[domain_names[0]][0],
        domain_datasets[domain_names[1]][0],
        steps=CALIBRATION_STEPS, lr=LR, seed=seed,
    )
    results["cap_calibrated"] = _eval_domains(composed_cap, domain_datasets)

    elapsed = time.time() - t0
    meta["elapsed_s"] = elapsed
    return results, meta


def _compose_matched_models(base_model, domain_models, vocab_size):
    """Compose MatchedMagnitudeGPT models by standard weight concatenation.

    Same as compose_relu_models but creates a plain ReLURouterGPT as the
    composed model (the magnitude loss is only needed during training).
    """
    n_domains = len(domain_models)
    n_capsules_per_domain = domain_models[0].layers[0].capsule_pool.n_capsules
    n_capsules_total = n_capsules_per_domain * n_domains

    composed = ReLURouterGPT(
        vocab_size=vocab_size, n_capsules=n_capsules_total, **BASE,
    )
    mx.eval(composed.parameters())

    # Copy shared parameters from base
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    for layer_idx in range(len(composed.layers)):
        A_parts = [dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)
        B_composed = mx.concatenate(B_parts, axis=1)

        comp_pool = composed.layers[layer_idx].capsule_pool
        comp_pool.A.load_weights([("weight", A_composed)])
        comp_pool.B.load_weights([("weight", B_composed)])

    mx.eval(composed.parameters())
    return composed


def main():
    """Run composition experiment across 3 seeds and report."""
    seeds = [42, 123, 7]
    all_results = []
    all_meta = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r, m = run_composition_experiment(seed=seed)
        all_results.append(r)
        all_meta.append(m)

        # Print per-seed summary
        for method, vals in r.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains)
            print(f"  {method:<25} avg={vals['avg']:.4f} ({detail})")
        print(f"  Elapsed: {m['elapsed_s']:.0f}s")

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    relu_joint_mean = statistics.mean([r["relu_joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs joint':>10}")
    print("  " + "-" * 55)
    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - relu_joint_mean) / relu_joint_mean) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}%")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n  Kill Threshold Analysis")
    print("  " + "-" * 60)

    # Define methods and their kill criteria
    checks = [
        ("rmsnorm_zero_shot",   "RMSNorm zero-shot vs joint",              5.0, 10.0),
        ("matched_zero_shot",   "Matched-mag zero-shot vs joint",          5.0, 10.0),
        ("scalar_cal",          "Scalar cal vs joint",                     2.0,  5.0),
        ("relu_zero_shot",      "Plain zero-shot vs joint (reference)",    5.0, 10.0),
        ("full_cal",            "Full calibration vs joint (reference)",   2.0,  5.0),
        ("weight_avg",          "Weight averaging vs joint",               5.0, 10.0),
    ]

    for method, desc, target, kill in checks:
        if method not in all_results[0]:
            continue
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        delta = ((mean - relu_joint_mean) / relu_joint_mean) * 100
        if delta > kill:
            status = "KILLED"
        elif delta > target:
            status = "WARN  "
        else:
            status = "OK    "
        print(f"  {status} {desc:<45} {delta:>+.1f}% (target <{target}%, kill >{kill}%)")

    # Scalar vs full calibration comparison
    scalar_mean = statistics.mean([r["scalar_cal"]["avg"] for r in all_results])
    full_mean = statistics.mean([r["full_cal"]["avg"] for r in all_results])
    scalar_vs_full = ((scalar_mean - full_mean) / full_mean) * 100
    if abs(scalar_vs_full) < 2:
        print(f"\n  DIAGNOSTIC: Scalar cal within {scalar_vs_full:+.1f}% of full cal")
        print(f"             -> Loudness IS the sole issue (2 params/layer suffice)")
    else:
        print(f"\n  DIAGNOSTIC: Scalar cal {scalar_vs_full:+.1f}% vs full cal")
        print(f"             -> Loudness is NOT the sole issue (direction matters too)")

    # RMS diagnostic
    print(f"\n  RMS Diagnostic (seed 42):")
    if all_meta:
        m = all_meta[0]
        print(f"    Base RMS:  {['%.4f' % r for r in m.get('base_rms', [])]}")
        for d_name, rms in m.get('domain_rms', {}).items():
            print(f"    {d_name} RMS: {['%.4f' % r for r in rms]}")
        if 'matched_target_rms' in m:
            print(f"    Matched target: {['%.4f' % r for r in m['matched_target_rms']]}")
            for d_name, rms in m.get('matched_post_rms', {}).items():
                print(f"    Matched {d_name}: {['%.4f' % r for r in rms]}")

    # Learned scales diagnostic
    print(f"\n  Learned Scales (per seed):")
    for i, m in enumerate(all_meta):
        if 'learned_scales' in m:
            print(f"    Seed {seeds[i]}: {['%.3f' % s for s in m['learned_scales']]}")

    # Best zero-shot method
    zero_shot_methods = ["relu_zero_shot", "rmsnorm_zero_shot", "matched_zero_shot",
                         "matched_rmsnorm"]
    print(f"\n  Zero-Shot Methods Ranking:")
    rankings = []
    for method in zero_shot_methods:
        if method not in all_results[0]:
            continue
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        delta = ((mean - relu_joint_mean) / relu_joint_mean) * 100
        rankings.append((delta, method, mean))
    rankings.sort()
    for delta, method, mean in rankings:
        marker = " <-- BEST" if delta == rankings[0][0] else ""
        print(f"    {method:<25} {delta:>+.1f}% vs joint (avg={mean:.4f}){marker}")


if __name__ == "__main__":
    main()
