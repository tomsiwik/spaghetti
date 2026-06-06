"""Shared/Unique Decomposition Experiment.

Tests whether decomposing domain-specific fine-tuning deltas into shared
(always-on) + unique (routed) components improves composition robustness.

Compares:
1. Joint training (baseline)
2. Shared-base concatenation + calibrated router (validated approach)
3. Task arithmetic merge (4 groups, known to dilute)
4. Shared-only model (base + shared_delta, no unique)
5. Decomposed + calibrated routing of unique groups
6. Decomposed + uniform routing of unique groups (robustness test)

Key prediction: (6) should outperform concatenation + uniform because
shared groups absorb routing errors.
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CAL_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def compute_deltas(base_model, domain_model):
    """Compute weight deltas between domain model and base for capsule groups."""
    deltas = []  # per layer: list of (delta_A, delta_B) per group
    for l_idx in range(len(base_model.layers)):
        layer_deltas = []
        base_pool = base_model.layers[l_idx].capsule_pool
        domain_pool = domain_model.layers[l_idx].capsule_pool
        for g in range(base_pool.n_groups):
            delta_A = domain_pool.groups[g].A.weight - base_pool.groups[g].A.weight
            delta_B = domain_pool.groups[g].B.weight - base_pool.groups[g].B.weight
            layer_deltas.append((delta_A, delta_B))
        deltas.append(layer_deltas)
    return deltas


def decompose_deltas(deltas_A, deltas_B):
    """Decompose two sets of deltas into shared + unique components.

    Returns: shared_deltas, unique_A_deltas, unique_B_deltas
    Each has the same structure as input: list of layers, each a list of (dA, dB) per group.
    """
    shared = []
    unique_A = []
    unique_B = []
    for l_idx in range(len(deltas_A)):
        s_layer, uA_layer, uB_layer = [], [], []
        for g in range(len(deltas_A[l_idx])):
            dA_A, dA_B = deltas_A[l_idx][g]
            dB_A, dB_B = deltas_B[l_idx][g]
            # Shared = average of deltas
            s_A = (dA_A + dB_A) / 2
            s_B = (dA_B + dB_B) / 2
            # Unique = delta - shared
            uA_A = dA_A - s_A  # = (dA_A - dB_A) / 2
            uA_B = dA_B - s_B
            uB_A = dB_A - s_A  # = (dB_A - dA_A) / 2 = -uA_A
            uB_B = dB_B - s_B
            s_layer.append((s_A, s_B))
            uA_layer.append((uA_A, uA_B))
            uB_layer.append((uB_A, uB_B))
        shared.append(s_layer)
        unique_A.append(uA_layer)
        unique_B.append(uB_layer)
    return shared, unique_A, unique_B


def measure_decomposition(deltas_A, deltas_B, shared, unique_A, unique_B):
    """Measure shared/unique ratio and verify exact reconstruction."""
    total_delta_norm = 0.0
    total_shared_norm = 0.0
    total_unique_norm = 0.0
    max_recon_error = 0.0

    for l_idx in range(len(deltas_A)):
        for g in range(len(deltas_A[l_idx])):
            for mat_idx in range(2):  # 0=A, 1=B
                delta_A_mat = deltas_A[l_idx][g][mat_idx]
                delta_B_mat = deltas_B[l_idx][g][mat_idx]
                shared_mat = shared[l_idx][g][mat_idx]
                unique_A_mat = unique_A[l_idx][g][mat_idx]
                unique_B_mat = unique_B[l_idx][g][mat_idx]

                # Norms
                total_delta_norm += (mx.sum(delta_A_mat ** 2).item() +
                                     mx.sum(delta_B_mat ** 2).item())
                total_shared_norm += 2 * mx.sum(shared_mat ** 2).item()
                total_unique_norm += (mx.sum(unique_A_mat ** 2).item() +
                                      mx.sum(unique_B_mat ** 2).item())

                # Reconstruction check: shared + unique_k should = delta_k
                recon_A = shared_mat + unique_A_mat
                recon_B = shared_mat + unique_B_mat
                err_A = mx.max(mx.abs(recon_A - delta_A_mat)).item()
                err_B = mx.max(mx.abs(recon_B - delta_B_mat)).item()
                max_recon_error = max(max_recon_error, err_A, err_B)

    total_delta_norm = total_delta_norm ** 0.5
    total_shared_norm = total_shared_norm ** 0.5
    total_unique_norm = total_unique_norm ** 0.5

    shared_frac = total_shared_norm / (total_shared_norm + total_unique_norm) if (total_shared_norm + total_unique_norm) > 0 else 0

    return {
        "delta_norm": total_delta_norm,
        "shared_norm": total_shared_norm,
        "unique_norm": total_unique_norm,
        "shared_fraction": shared_frac,
        "max_reconstruction_error": max_recon_error,
    }


def build_task_arithmetic_model(base_model, deltas_A, deltas_B, vocab_size, lam=0.5):
    """Build a merged model using task arithmetic: base + λ(Δ_A + Δ_B)."""
    merged = get_model("capsule_moe", vocab_size=vocab_size, **CAP)
    mx.eval(merged.parameters())

    # Copy base weights
    merged.wte.weight = base_model.wte.weight
    merged.wpe.weight = base_model.wpe.weight
    merged.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(base_model.layers)):
        layer_m = merged.layers[l_idx]
        layer_b = base_model.layers[l_idx]

        # Copy attention from base
        layer_m.attn.wq.weight = layer_b.attn.wq.weight
        layer_m.attn.wk.weight = layer_b.attn.wk.weight
        layer_m.attn.wv.weight = layer_b.attn.wv.weight
        layer_m.attn.wo.weight = layer_b.attn.wo.weight

        # Merge capsule groups: base + λ(Δ_A + Δ_B)
        pool_m = layer_m.capsule_pool
        pool_b = layer_b.capsule_pool
        for g in range(pool_b.n_groups):
            dA_A, dA_B = deltas_A[l_idx][g]
            dB_A, dB_B = deltas_B[l_idx][g]
            pool_m.groups[g].A.weight = pool_b.groups[g].A.weight + lam * (dA_A + dB_A)
            pool_m.groups[g].B.weight = pool_b.groups[g].B.weight + lam * (dA_B + dB_B)

    mx.eval(merged.parameters())
    return merged


def build_shared_only_model(base_model, shared_deltas, vocab_size):
    """Build a model with only shared knowledge: base + shared_delta."""
    model = get_model("capsule_moe", vocab_size=vocab_size, **CAP)
    mx.eval(model.parameters())

    model.wte.weight = base_model.wte.weight
    model.wpe.weight = base_model.wpe.weight
    model.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(base_model.layers)):
        layer = model.layers[l_idx]
        layer_b = base_model.layers[l_idx]

        layer.attn.wq.weight = layer_b.attn.wq.weight
        layer.attn.wk.weight = layer_b.attn.wk.weight
        layer.attn.wv.weight = layer_b.attn.wv.weight
        layer.attn.wo.weight = layer_b.attn.wo.weight

        pool = layer.capsule_pool
        pool_b = layer_b.capsule_pool
        for g in range(pool_b.n_groups):
            s_A, s_B = shared_deltas[l_idx][g]
            pool.groups[g].A.weight = pool_b.groups[g].A.weight + s_A
            pool.groups[g].B.weight = pool_b.groups[g].B.weight + s_B

    mx.eval(model.parameters())
    return model


def build_decomposed_model(base_model, shared_deltas, unique_A_deltas, unique_B_deltas,
                            vocab_size, uniform_unique=False):
    """Build decomposed model: shared (always on) + unique (routed).

    4 shared groups + 8 unique groups (4 from A, 4 from B).
    """
    n_shared = len(shared_deltas[0])  # 4
    n_unique = n_shared * 2  # 8

    model = get_model("procrustes_decomp",
                       vocab_size=vocab_size, block_size=BASE["block_size"],
                       n_embd=BASE["n_embd"], n_head=BASE["n_head"],
                       n_layer=BASE["n_layer"],
                       n_shared=n_shared, n_unique=n_unique,
                       n_capsules_per_group=CAP["n_capsules_per_group"],
                       top_k_unique=n_shared,  # k=4 from 8 unique
                       uniform_unique=uniform_unique)
    mx.eval(model.parameters())

    # Copy base shared params
    model.wte.weight = base_model.wte.weight
    model.wpe.weight = base_model.wpe.weight
    model.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(base_model.layers)):
        layer = model.layers[l_idx]
        layer_b = base_model.layers[l_idx]

        # Copy attention from base
        layer.attn.wq.weight = layer_b.attn.wq.weight
        layer.attn.wk.weight = layer_b.attn.wk.weight
        layer.attn.wv.weight = layer_b.attn.wv.weight
        layer.attn.wo.weight = layer_b.attn.wo.weight

        pool = layer.pool
        pool_b = layer_b.capsule_pool

        # Shared groups: base + shared_delta
        for g in range(n_shared):
            s_A, s_B = shared_deltas[l_idx][g]
            pool.shared_groups[g].A.weight = pool_b.groups[g].A.weight + s_A
            pool.shared_groups[g].B.weight = pool_b.groups[g].B.weight + s_B

        # Unique groups: first n_shared from domain A, next from domain B
        for g in range(n_shared):
            uA_A, uA_B = unique_A_deltas[l_idx][g]
            pool.unique_groups[g].A.weight = uA_A
            pool.unique_groups[g].B.weight = uA_B

            uB_A, uB_B = unique_B_deltas[l_idx][g]
            pool.unique_groups[n_shared + g].A.weight = uB_A
            pool.unique_groups[n_shared + g].B.weight = uB_B

    mx.eval(model.parameters())
    return model


def freeze_except_unique_router(model):
    """Freeze all params except the decomposed pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.pool.router.unfreeze()


def calibrate_decomposed_router(model, train_ds_a, train_ds_b,
                                 steps=100, lr=3e-3, seed=42):
    """Calibrate the unique-group router on mixed-domain data."""
    freeze_except_unique_router(model)

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0 or step == steps:
            print(f"    decomp router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


def run_experiment(seed=42):
    """Run the full decomposition experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"PROCRUSTES DECOMPOSITION EXPERIMENT (seed={seed})")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    splits = domain_split(docs)
    all_train, all_val = train_val_split(docs, seed=seed)

    train_a_docs, val_a_docs = train_val_split(splits["a_m"], seed=seed)
    train_b_docs, val_b_docs = train_val_split(splits["n_z"], seed=seed)

    train_a = CharDataset(train_a_docs, tokenizer, BASE["block_size"])
    val_a = CharDataset(val_a_docs, tokenizer, BASE["block_size"])
    train_b = CharDataset(train_b_docs, tokenizer, BASE["block_size"])
    val_b = CharDataset(val_b_docs, tokenizer, BASE["block_size"])
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # === Baseline: Joint training ===
    print("\n--- Baseline: Joint training ---")
    model_joint = get_model("capsule_moe", vocab_size=V, **CAP)
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
        if step % 200 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    j_a = evaluate(model_joint, val_a, BATCH_SIZE)
    j_b = evaluate(model_joint, val_b, BATCH_SIZE)
    results["joint"] = {"a_m": j_a, "n_z": j_b, "avg": (j_a + j_b) / 2}
    print(f"  Joint: a_m={j_a:.4f}, n_z={j_b:.4f}, avg={(j_a+j_b)/2:.4f}")

    # === Shared-base protocol: pretrain + fine-tune domains ===
    print("\n--- Pretraining shared base ---")
    base_model = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    def copy_weights(src, dst):
        pairs = list(zip(
            [k for k, _ in nn.utils.tree_flatten(src.parameters())],
            [v for _, v in nn.utils.tree_flatten(src.parameters())]
        ))
        dst.load_weights(pairs)
        mx.eval(dst.parameters())

    # Fine-tune domain A
    print("\n--- Fine-tuning domain A capsules ---")
    model_a = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_a.parameters())
    copy_weights(base_model, model_a)
    model_a.freeze()
    for layer in model_a.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_a, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_a.unfreeze()

    # Fine-tune domain B
    print("\n--- Fine-tuning domain B capsules ---")
    model_b = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_b.parameters())
    copy_weights(base_model, model_b)
    model_b.freeze()
    for layer in model_b.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_b, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_b.unfreeze()

    # === Compute deltas and decompose ===
    print("\n--- Computing decomposition ---")
    deltas_A = compute_deltas(base_model, model_a)
    deltas_B = compute_deltas(base_model, model_b)
    shared, unique_A, unique_B = decompose_deltas(deltas_A, deltas_B)
    metrics = measure_decomposition(deltas_A, deltas_B, shared, unique_A, unique_B)

    print(f"  Delta norm:     {metrics['delta_norm']:.4f}")
    print(f"  Shared norm:    {metrics['shared_norm']:.4f}")
    print(f"  Unique norm:    {metrics['unique_norm']:.4f}")
    print(f"  Shared fraction: {metrics['shared_fraction']:.1%}")
    print(f"  Max recon error: {metrics['max_reconstruction_error']:.2e}")

    # === Model 1: Shared-base concatenation + calibrated router ===
    print("\n--- Concatenation + calibrated router ---")
    from experiments.models.capsule_moe.test_composition import (
        compose_from_shared_base, calibrate_router)
    groups_a_layers = [model_a.layers[l].capsule_pool.groups
                       for l in range(len(base_model.layers))]
    groups_b_layers = [model_b.layers[l].capsule_pool.groups
                       for l in range(len(base_model.layers))]
    concat_cal = compose_from_shared_base(base_model, groups_a_layers, groups_b_layers,
                                           V, BASE["block_size"])
    mx.eval(concat_cal.parameters())
    calibrate_router(concat_cal, train_a, train_b, steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    cc_a = evaluate(concat_cal, val_a, BATCH_SIZE)
    cc_b = evaluate(concat_cal, val_b, BATCH_SIZE)
    results["concat_cal"] = {"a_m": cc_a, "n_z": cc_b, "avg": (cc_a + cc_b) / 2}
    print(f"  Concat+cal: a_m={cc_a:.4f}, n_z={cc_b:.4f}, avg={(cc_a+cc_b)/2:.4f}")

    # === Model 2: Concatenation + uniform router ===
    print("\n--- Concatenation + uniform router ---")
    concat_uni = compose_from_shared_base(base_model, groups_a_layers, groups_b_layers,
                                           V, BASE["block_size"])
    for layer in concat_uni.layers:
        layer.capsule_pool.uniform_routing = True
    mx.eval(concat_uni.parameters())
    cu_a = evaluate(concat_uni, val_a, BATCH_SIZE)
    cu_b = evaluate(concat_uni, val_b, BATCH_SIZE)
    results["concat_uni"] = {"a_m": cu_a, "n_z": cu_b, "avg": (cu_a + cu_b) / 2}
    print(f"  Concat+uni: a_m={cu_a:.4f}, n_z={cu_b:.4f}, avg={(cu_a+cu_b)/2:.4f}")

    # === Model 3: Task arithmetic merge ===
    print("\n--- Task arithmetic merge ---")
    ta_model = build_task_arithmetic_model(base_model, deltas_A, deltas_B, V)
    ta_a = evaluate(ta_model, val_a, BATCH_SIZE)
    ta_b = evaluate(ta_model, val_b, BATCH_SIZE)
    results["task_arith"] = {"a_m": ta_a, "n_z": ta_b, "avg": (ta_a + ta_b) / 2}
    print(f"  Task arith: a_m={ta_a:.4f}, n_z={ta_b:.4f}, avg={(ta_a+ta_b)/2:.4f}")

    # === Model 4: Shared-only (no unique) ===
    print("\n--- Shared-only model ---")
    shared_model = build_shared_only_model(base_model, shared, V)
    so_a = evaluate(shared_model, val_a, BATCH_SIZE)
    so_b = evaluate(shared_model, val_b, BATCH_SIZE)
    results["shared_only"] = {"a_m": so_a, "n_z": so_b, "avg": (so_a + so_b) / 2}
    print(f"  Shared only: a_m={so_a:.4f}, n_z={so_b:.4f}, avg={(so_a+so_b)/2:.4f}")

    # === Model 5: Decomposed + calibrated router ===
    print("\n--- Decomposed + calibrated router ---")
    decomp_cal = build_decomposed_model(base_model, shared, unique_A, unique_B, V)
    calibrate_decomposed_router(decomp_cal, train_a, train_b,
                                 steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    dc_a = evaluate(decomp_cal, val_a, BATCH_SIZE)
    dc_b = evaluate(decomp_cal, val_b, BATCH_SIZE)
    results["decomp_cal"] = {"a_m": dc_a, "n_z": dc_b, "avg": (dc_a + dc_b) / 2}
    print(f"  Decomp+cal: a_m={dc_a:.4f}, n_z={dc_b:.4f}, avg={(dc_a+dc_b)/2:.4f}")

    # === Model 6: Decomposed + uniform routing (robustness test) ===
    print("\n--- Decomposed + uniform routing ---")
    decomp_uni = build_decomposed_model(base_model, shared, unique_A, unique_B, V,
                                         uniform_unique=True)
    du_a = evaluate(decomp_uni, val_a, BATCH_SIZE)
    du_b = evaluate(decomp_uni, val_b, BATCH_SIZE)
    results["decomp_uni"] = {"a_m": du_a, "n_z": du_b, "avg": (du_a + du_b) / 2}
    print(f"  Decomp+uni: a_m={du_a:.4f}, n_z={du_b:.4f}, avg={(du_a+du_b)/2:.4f}")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Shared fraction: {metrics['shared_fraction']:.1%}")
    print(f"\n{'Method':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} {'vs joint':>10}")
    print("-" * 60)
    for method, vals in results.items():
        delta = (vals["avg"] - results["joint"]["avg"]) / results["joint"]["avg"] * 100
        print(f"{method:<25} {vals['a_m']:>8.4f} {vals['n_z']:>8.4f} {vals['avg']:>8.4f} {delta:>+9.1f}%")

    return results, metrics


def run_multiseed(seeds=(42, 123, 7)):
    """Run across multiple seeds and aggregate."""
    all_results = {}
    all_metrics = {}

    for seed in seeds:
        results, metrics = run_experiment(seed)
        all_results[seed] = results
        all_metrics[seed] = metrics

    # Aggregate
    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    # Shared fraction
    shared_fracs = [all_metrics[s]["shared_fraction"] for s in seeds]
    print(f"  Shared fraction: {statistics.mean(shared_fracs):.1%} "
          f"(range: {min(shared_fracs):.1%}-{max(shared_fracs):.1%})")

    methods = list(all_results[seeds[0]].keys())
    joint_avgs = [all_results[s]["joint"]["avg"] for s in seeds]
    joint_mean = statistics.mean(joint_avgs)

    print(f"\n{'Method':<25} {'avg (mean)':>12} {'avg (range)':>14} {'vs joint':>10}")
    print("-" * 65)
    for method in methods:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        range_avg = max(avgs) - min(avgs)
        if method == "joint":
            delta_str = "baseline"
        else:
            delta = (mean_avg - joint_mean) / joint_mean * 100
            delta_str = f"{delta:+.1f}%"
        print(f"{method:<25} {mean_avg:>12.4f} {'+/- ' + f'{range_avg:.4f}':>14} {delta_str:>10}")

    # Kill threshold checks
    print(f"\n--- Kill Threshold Checks ---")
    for method in ["decomp_cal", "decomp_uni"]:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        delta_pct = (mean_avg - joint_mean) / joint_mean * 100
        if delta_pct > 5.0:
            print(f"  KILL: {method} {delta_pct:+.1f}% vs joint (threshold: 5%)")
        else:
            print(f"  OK: {method} {delta_pct:+.1f}% vs joint")

    # Robustness comparison: decomp_uni vs concat_uni
    decomp_uni_avgs = [all_results[s]["decomp_uni"]["avg"] for s in seeds]
    concat_uni_avgs = [all_results[s]["concat_uni"]["avg"] for s in seeds]
    du_mean = statistics.mean(decomp_uni_avgs)
    cu_mean = statistics.mean(concat_uni_avgs)
    improvement = (cu_mean - du_mean) / cu_mean * 100
    print(f"\n  Robustness test: decomp+uniform vs concat+uniform")
    print(f"    decomp+uniform avg: {du_mean:.4f}")
    print(f"    concat+uniform avg: {cu_mean:.4f}")
    if du_mean < cu_mean:
        print(f"    Decomposition is MORE robust ({improvement:+.1f}% better)")
    else:
        print(f"    Decomposition is LESS robust ({improvement:+.1f}% worse)")

    return all_results, all_metrics


if __name__ == "__main__":
    run_multiseed()
