"""Exp 4: Scale composition to N=5 experts.

Research question: Does the shared-base composition protocol scale from
N=2 to N=5 domains? Do capsule group subspaces remain orthogonal?

Protocol:
  1. Pretrain shared base on all data (300 steps)
  2. Fine-tune only capsule groups per domain (300 steps × 5 domains)
  3. Compose: concatenate 5×4=20 capsule groups + shared base
  4. Calibrate router (200 steps on mixed data)
  5. Compare vs joint training (G=20, k=10, 1500 steps)

Kill threshold: composition+calibrated vs joint > 5%.
"""

import copy
import math
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss

# Config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
G_PER_DOMAIN = 4
N_CAPSULES = 64
K_PER_DOMAIN = 2
STEPS_PER_DOMAIN = 300
PRETRAIN_STEPS = 300
ROUTER_CAL_STEPS = 200
BATCH_SIZE = 32
LR = 3e-3
N_DOMAINS = 5


def build_domain_datasets(tokenizer, docs, block_size=32, seed=42):
    """Build train/val datasets for each of 5 domains."""
    splits = domain_split(docs, method="quintary")
    datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        datasets[name] = (
            CharDataset(dtrain, tokenizer, block_size),
            CharDataset(dval, tokenizer, block_size),
        )
    return datasets


def compose_from_shared_base_n(base_model, domain_groups, vocab_size, block_size):
    """Compose N domain capsule groups onto shared base.

    domain_groups: list of N entries, each is list of [layer][group] CapsuleGroups
    """
    n_domains = len(domain_groups)
    composed_g = n_domains * G_PER_DOMAIN
    composed_k = n_domains * K_PER_DOMAIN

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_g,
                         n_capsules_per_group=N_CAPSULES,
                         top_k_groups=composed_k,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Copy base shared params
    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_base = base_model.layers[l_idx]

        # Copy attention from base
        layer_c.attn.wq.weight = layer_base.attn.wq.weight
        layer_c.attn.wk.weight = layer_base.attn.wk.weight
        layer_c.attn.wv.weight = layer_base.attn.wv.weight
        layer_c.attn.wo.weight = layer_base.attn.wo.weight

        # Slot domain capsule groups
        pool_c = layer_c.capsule_pool
        for d_idx, d_groups in enumerate(domain_groups):
            offset = d_idx * G_PER_DOMAIN
            for g in range(G_PER_DOMAIN):
                pool_c.groups[offset + g].A.weight = d_groups[l_idx][g].A.weight
                pool_c.groups[offset + g].B.weight = d_groups[l_idx][g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    """Freeze all parameters except the capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_datasets, steps=200, lr=3e-3, seed=42):
    """Train only router on rotating domain batches."""
    freeze_except_router(model)

    domain_list = list(train_datasets.values())
    n_domains = len(domain_list)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    for step in range(1, steps + 1):
        ds = domain_list[step % n_domains]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % 50 == 0 or step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()
    return losses


def compute_delta_orthogonality(base_model, domain_models):
    """Compute pairwise cosine similarity of capsule weight deltas across domains.

    Returns dict with per-layer and aggregate stats.
    """
    n_domains = len(domain_models)
    n_layers = len(base_model.layers)

    all_layer_sims = []

    for l_idx in range(n_layers):
        # Extract flattened capsule weight deltas per domain
        deltas = []
        base_groups = base_model.layers[l_idx].capsule_pool.groups
        for d_model in domain_models:
            d_groups = d_model.layers[l_idx].capsule_pool.groups
            # Flatten all group weights into a single vector
            delta_parts = []
            for g in range(G_PER_DOMAIN):
                delta_A = d_groups[g].A.weight - base_groups[g].A.weight
                delta_B = d_groups[g].B.weight - base_groups[g].B.weight
                delta_parts.append(delta_A.reshape(-1))
                delta_parts.append(delta_B.reshape(-1))
            delta_vec = mx.concatenate(delta_parts)
            deltas.append(delta_vec)

        # Pairwise cosine similarity
        layer_sims = []
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                dot = mx.sum(deltas[i] * deltas[j]).item()
                norm_i = mx.sqrt(mx.sum(deltas[i] * deltas[i])).item()
                norm_j = mx.sqrt(mx.sum(deltas[j] * deltas[j])).item()
                cos_sim = dot / (norm_i * norm_j + 1e-12)
                layer_sims.append(cos_sim)
        all_layer_sims.append(layer_sims)

    # Aggregate
    flat_sims = [s for layer in all_layer_sims for s in layer]
    return {
        "per_layer": all_layer_sims,
        "mean": statistics.mean(flat_sims),
        "max": max(flat_sims),
        "min": min(flat_sims),
        "all_sims": flat_sims,
    }


def task_arithmetic_compose(base_model, domain_models, vocab_size, block_size, scale=None):
    """Compose by averaging weight deltas (task arithmetic)."""
    n_domains = len(domain_models)
    if scale is None:
        scale = 1.0 / n_domains

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=G_PER_DOMAIN, n_capsules_per_group=N_CAPSULES,
                         top_k_groups=K_PER_DOMAIN,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Start from base
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    composed_params = dict(nn.utils.tree_flatten(composed.parameters()))

    for key in composed_params:
        base_val = base_params[key]
        delta_sum = mx.zeros_like(base_val)
        for d_model in domain_models:
            d_params = dict(nn.utils.tree_flatten(d_model.parameters()))
            delta_sum = delta_sum + (d_params[key] - base_val)
        composed_params[key] = base_val + scale * delta_sum

    composed.load_weights(list(composed_params.items()))
    mx.eval(composed.parameters())
    return composed


def analyze_router(model, datasets, n_batches=10):
    """Analyze router probability distribution on each domain's data."""
    n_groups = model.layers[0].capsule_pool.n_groups
    domain_names = list(datasets.keys())
    analysis = {}

    for d_name in domain_names:
        _, val_ds = datasets[d_name]
        rng = random.Random(0)
        all_probs = []

        for _ in range(n_batches):
            inputs, _ = val_ds.get_batch(BATCH_SIZE, rng)
            _ = model(inputs)
            for layer in model.layers:
                probs = layer.capsule_pool._gate_probs
                all_probs.append(probs.reshape(-1, n_groups))

        stacked = mx.concatenate(all_probs, axis=0)  # (N_tokens, G)
        mean_probs = mx.mean(stacked, axis=0)  # (G,)

        # Entropy
        p = mean_probs
        h = -mx.sum(p * mx.log(p + 1e-12)).item()
        h_max = math.log(n_groups)

        # Top group concentration
        sorted_p = mx.sort(p)[::-1]
        c1 = sorted_p[0].item()

        analysis[d_name] = {
            "mean_probs": mean_probs.tolist(),
            "entropy": h,
            "h_ratio": h / h_max if h_max > 0 else 0,
            "c1": c1,
        }

    return analysis


def run_experiment(seed=42):
    """Run the full N=5 scaling experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"EXP 4: N=5 EXPERT SCALING (seed={seed})")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    datasets = build_domain_datasets(tokenizer, docs, seed=seed)
    domain_names = list(datasets.keys())
    print(f"Domains: {domain_names}")
    for name, (tr, va) in datasets.items():
        print(f"  {name}: {len(tr)} train, {len(va)} val sequences")

    # Also build joint dataset
    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # --- Baseline: Joint training (G=20, k=10) ---
    print("\n--- Baseline: Joint training (G=20, k=10) ---")
    model_joint = get_model("capsule_moe", vocab_size=V,
                            n_groups=N_DOMAINS * G_PER_DOMAIN,
                            n_capsules_per_group=N_CAPSULES,
                            top_k_groups=N_DOMAINS * K_PER_DOMAIN,
                            **BASE)
    mx.eval(model_joint.parameters())

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = N_DOMAINS * STEPS_PER_DOMAIN
    for step in range(1, total_steps + 1):
        # Rotate through domains
        d_name = domain_names[step % N_DOMAINS]
        tr_ds = datasets[d_name][0]
        inputs, targets = tr_ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)
        if step % 300 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    joint_vals = {}
    for d_name, (_, val_ds) in datasets.items():
        joint_vals[d_name] = evaluate(model_joint, val_ds, BATCH_SIZE)
    joint_avg = statistics.mean(joint_vals.values())
    results["joint"] = {**joint_vals, "avg": joint_avg}
    print(f"  Joint: {' | '.join(f'{d}={v:.4f}' for d, v in joint_vals.items())} | avg={joint_avg:.4f}")

    # --- Phase 1: Pretrain shared base ---
    print("\n--- Phase 1: Pretrain shared base (G=4, k=2) ---")
    base_model = get_model("capsule_moe", vocab_size=V,
                           n_groups=G_PER_DOMAIN, n_capsules_per_group=N_CAPSULES,
                           top_k_groups=K_PER_DOMAIN, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)

    # --- Phase 2: Fine-tune capsule groups per domain ---
    domain_models = []
    domain_groups_all = []  # For composition

    for d_idx, d_name in enumerate(domain_names):
        print(f"\n--- Phase 2.{d_idx+1}: Fine-tune domain '{d_name}' ---")
        tr_ds, val_ds = datasets[d_name]

        d_model = get_model("capsule_moe", vocab_size=V,
                            n_groups=G_PER_DOMAIN, n_capsules_per_group=N_CAPSULES,
                            top_k_groups=K_PER_DOMAIN, **BASE)
        mx.eval(d_model.parameters())

        # Copy base weights
        d_model.load_weights(list(zip(
            [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
            [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
        )))
        mx.eval(d_model.parameters())

        # Freeze everything except capsule groups
        d_model.freeze()
        for layer in d_model.layers:
            for group in layer.capsule_pool.groups:
                group.unfreeze()

        train(d_model, tr_ds, val_ds, steps=STEPS_PER_DOMAIN,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
        d_model.unfreeze()

        domain_models.append(d_model)

        # Extract groups for composition
        d_groups = []
        for l_idx in range(len(base_model.layers)):
            d_groups.append(d_model.layers[l_idx].capsule_pool.groups)
        domain_groups_all.append(d_groups)

    # --- Orthogonality analysis ---
    print("\n--- Orthogonality analysis ---")
    ortho = compute_delta_orthogonality(base_model, domain_models)
    print(f"  Pairwise cosine similarity: mean={ortho['mean']:.4f}, max={ortho['max']:.4f}, min={ortho['min']:.4f}")
    for l_idx, layer_sims in enumerate(ortho["per_layer"]):
        print(f"  Layer {l_idx}: mean={statistics.mean(layer_sims):.4f}, max={max(layer_sims):.4f}")

    # --- Phase 3a: Compose with uniform routing ---
    print("\n--- Phase 3a: Composed (uniform routing) ---")
    composed_uniform = compose_from_shared_base_n(base_model, domain_groups_all, V, BASE["block_size"])
    for layer in composed_uniform.layers:
        layer.capsule_pool.uniform_routing = True
    mx.eval(composed_uniform.parameters())

    uniform_vals = {}
    for d_name, (_, val_ds) in datasets.items():
        uniform_vals[d_name] = evaluate(composed_uniform, val_ds, BATCH_SIZE)
    uniform_avg = statistics.mean(uniform_vals.values())
    results["composed_uniform"] = {**uniform_vals, "avg": uniform_avg}
    print(f"  Uniform: {' | '.join(f'{d}={v:.4f}' for d, v in uniform_vals.items())} | avg={uniform_avg:.4f}")

    # --- Phase 3b: Compose with calibrated router ---
    print("\n--- Phase 3b: Composed (calibrated router) ---")
    composed_cal = compose_from_shared_base_n(base_model, domain_groups_all, V, BASE["block_size"])
    mx.eval(composed_cal.parameters())

    train_datasets = {d_name: datasets[d_name][0] for d_name in domain_names}
    calibrate_router(composed_cal, train_datasets,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)

    cal_vals = {}
    for d_name, (_, val_ds) in datasets.items():
        cal_vals[d_name] = evaluate(composed_cal, val_ds, BATCH_SIZE)
    cal_avg = statistics.mean(cal_vals.values())
    results["composed_calibrated"] = {**cal_vals, "avg": cal_avg}
    print(f"  Calibrated: {' | '.join(f'{d}={v:.4f}' for d, v in cal_vals.items())} | avg={cal_avg:.4f}")

    # --- Task arithmetic baseline ---
    print("\n--- Baseline: Task arithmetic (mean of deltas) ---")
    ta_model = task_arithmetic_compose(base_model, domain_models, V, BASE["block_size"])
    ta_vals = {}
    for d_name, (_, val_ds) in datasets.items():
        ta_vals[d_name] = evaluate(ta_model, val_ds, BATCH_SIZE)
    ta_avg = statistics.mean(ta_vals.values())
    results["task_arithmetic"] = {**ta_vals, "avg": ta_avg}
    print(f"  Task arith: {' | '.join(f'{d}={v:.4f}' for d, v in ta_vals.items())} | avg={ta_avg:.4f}")

    # --- Router analysis ---
    print("\n--- Router analysis ---")
    router_info = analyze_router(composed_cal, datasets)
    for d_name, info in router_info.items():
        print(f"  {d_name}: H/H_max={info['h_ratio']:.3f}, C1={info['c1']:.3f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'avg':>8} {'vs joint':>10}")
    print("-" * 45)
    for method, vals in results.items():
        delta = (vals["avg"] - results["joint"]["avg"]) / results["joint"]["avg"] * 100
        print(f"{method:<25} {vals['avg']:>8.4f} {delta:>+9.1f}%")

    # Per-domain detail
    print(f"\n{'Method':<25}", end="")
    for d_name in domain_names:
        print(f" {d_name:>8}", end="")
    print(f" {'avg':>8}")
    print("-" * (25 + 9 * (len(domain_names) + 1)))
    for method, vals in results.items():
        print(f"{method:<25}", end="")
        for d_name in domain_names:
            print(f" {vals[d_name]:>8.4f}", end="")
        print(f" {vals['avg']:>8.4f}")

    return {
        "results": results,
        "orthogonality": ortho,
        "router_analysis": router_info,
    }


def run_multiseed(seeds=(42, 123, 7)):
    """Run across multiple seeds and aggregate."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_experiment(seed)

    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    methods = ["joint", "composed_uniform", "composed_calibrated", "task_arithmetic"]
    joint_avgs = [all_results[s]["results"]["joint"]["avg"] for s in seeds]
    joint_mean = statistics.mean(joint_avgs)

    print(f"{'Method':<25} {'avg (mean)':>12} {'avg (range)':>14} {'vs joint':>10}")
    print("-" * 65)
    for method in methods:
        avgs = [all_results[s]["results"][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        range_avg = max(avgs) - min(avgs)
        if method == "joint":
            delta_str = "baseline"
        else:
            delta = (mean_avg - joint_mean) / joint_mean * 100
            delta_str = f"{delta:+.1f}%"
        print(f"{method:<25} {mean_avg:>12.4f} {'+/- ' + f'{range_avg:.4f}':>14} {delta_str:>10}")

    # Orthogonality aggregate
    print(f"\nOrthogonality (pairwise cosine similarity of capsule deltas):")
    all_means = [all_results[s]["orthogonality"]["mean"] for s in seeds]
    all_maxes = [all_results[s]["orthogonality"]["max"] for s in seeds]
    print(f"  Mean cos sim: {statistics.mean(all_means):.4f} (range: {min(all_means):.4f} - {max(all_means):.4f})")
    print(f"  Max cos sim:  {statistics.mean(all_maxes):.4f} (range: {min(all_maxes):.4f} - {max(all_maxes):.4f})")

    # Kill threshold checks
    print(f"\nKill threshold checks:")
    comp_avgs = [all_results[s]["results"]["composed_calibrated"]["avg"] for s in seeds]
    comp_mean = statistics.mean(comp_avgs)
    delta_pct = (comp_mean - joint_mean) / joint_mean * 100
    status = "**KILL**" if delta_pct > 5.0 else "PASS"
    print(f"  Composition+cal vs joint: {delta_pct:+.1f}% (threshold: 5%) -> {status}")

    max_cos = statistics.mean(all_maxes)
    status2 = "CONCERN" if max_cos > 0.5 else "PASS"
    print(f"  Max pairwise cosine sim:  {max_cos:.4f} (threshold: 0.5) -> {status2}")

    # Per-domain worst case
    domain_names = list(all_results[seeds[0]]["results"]["joint"].keys())
    domain_names = [d for d in domain_names if d != "avg"]
    print(f"\n  Per-domain vs joint:")
    for d_name in domain_names:
        d_joints = [all_results[s]["results"]["joint"][d_name] for s in seeds]
        d_comps = [all_results[s]["results"]["composed_calibrated"][d_name] for s in seeds]
        d_delta = (statistics.mean(d_comps) - statistics.mean(d_joints)) / statistics.mean(d_joints) * 100
        status3 = "**PARTIAL KILL**" if d_delta > 10.0 else "PASS"
        print(f"    {d_name}: {d_delta:+.1f}% -> {status3}")

    return all_results


if __name__ == "__main__":
    run_multiseed()
