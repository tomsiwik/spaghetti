"""Delta rule interference ordering experiment.

Tests whether the delta rule's retrieval-and-correction mechanism reverses
the favorable interference ordering (linear < full, ratio 0.59x) found in
the simplified gated linear recurrence variant.

Protocol: identical to l2_norm_attention/run_l2_norm_experiment.py
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets
  Phase 6: Compute per-layer interference metrics

Three conditions:
  1. full_attn: all 4 layers full attention (control)
  2. l2_norm_3_1: L2-normalized simplified linear (baseline from prior experiment)
  3. delta_rule_3_1: delta rule linear with L2 norm (test condition)

7 seeds per condition (balancing statistical power and runtime).

Kill criteria:
  1. Delta rule reverses interference ordering: linear > full (ratio >1.0x)
  2. Delta rule composition gap exceeds +10% median across 5+ seeds
"""

import copy
import random
import statistics
import time
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


# Standard micro config (matches prior hybrid/l2_norm experiments)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def compose_models(model_a, model_b, model_name, vocab_size, block_size,
                   base_model=None, layer_types=None):
    """Create a composed model by concatenating capsule groups from A and B."""
    n_groups_a = model_a.layers[0].capsule_pool.n_groups
    n_groups_b = model_b.layers[0].capsule_pool.n_groups
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = model_a.layers[0].capsule_pool.top_k_groups + \
                     model_b.layers[0].capsule_pool.top_k_groups

    kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_groups=composed_groups,
        n_capsules_per_group=CAP["n_capsules_per_group"],
        top_k_groups=composed_top_k,
        n_embd=BASE["n_embd"], n_head=BASE["n_head"], n_layer=BASE["n_layer"],
    )
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    composed = get_model(model_name, **kwargs)

    source = base_model if base_model is not None else model_a
    composed.wte.weight = source.wte.weight
    composed.wpe.weight = source.wpe.weight
    composed.lm_head.weight = source.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_src = source.layers[l_idx]
        layer_a = model_a.layers[l_idx]
        layer_b = model_b.layers[l_idx]

        # Copy attention weights from base
        for name in ["wq", "wk", "wv", "wo"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        # Copy gate/delta-rule-specific params if they exist
        for name in ["wg", "w_a", "w_beta", "w_z"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        # Copy non-Linear params (dt_bias, A_log, out_norm)
        if hasattr(layer_src.attn, "dt_bias") and hasattr(layer_c.attn, "dt_bias"):
            layer_c.attn.dt_bias = layer_src.attn.dt_bias
        if hasattr(layer_src.attn, "A_log") and hasattr(layer_c.attn, "A_log"):
            layer_c.attn.A_log = layer_src.attn.A_log
        # Note: out_norm is parameter-free RMSNorm (no weights to copy)

        # Note: RMSNorm in micro GPT is parameter-free (no weights to copy)

        # Copy capsule groups
        pool_c = layer_c.capsule_pool
        pool_a = layer_a.capsule_pool
        pool_b = layer_b.capsule_pool

        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = pool_a.groups[g].A.weight
            pool_c.groups[g].B.weight = pool_a.groups[g].B.weight

        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = pool_b.groups[g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = pool_b.groups[g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    """Freeze all params except capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_ds_a, train_ds_b, steps=100, lr=3e-3, seed=42):
    """Train only router weights on mixed-domain data."""
    freeze_except_router(model)
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

    model.unfreeze()


def compute_per_layer_interference(model_a, model_b, base_model, val_ds, n_layers):
    """Compute per-layer interference as cosine distance between domain-specific outputs.

    For each layer, measure how much the capsule pool outputs differ between
    the two domain-specialized models when given the same input. Higher values
    mean more interference (more change from composition).
    """
    rng = random.Random(0)
    inputs, _ = val_ds.get_batch(BATCH_SIZE, rng)

    interference = {}
    for l_idx in range(n_layers):
        # Forward through base up to this layer, then through each domain's capsules
        # We measure the difference in capsule pool outputs
        x = model_a.wte(inputs) + model_a.wpe(mx.arange(inputs.shape[1]))
        x = model_a.norm0(x)

        for i in range(l_idx):
            x = base_model.layers[i](x)

        x_normed = model_a.layers[l_idx].norm2(
            x + model_a.layers[l_idx].attn(model_a.layers[l_idx].norm1(x))
        )

        out_a = model_a.layers[l_idx].capsule_pool(x_normed)
        out_b = model_b.layers[l_idx].capsule_pool(x_normed)
        mx.eval(out_a, out_b)

        # Cosine distance between capsule pool outputs
        dot = (out_a * out_b).sum()
        norm_a = mx.sqrt((out_a * out_a).sum())
        norm_b = mx.sqrt((out_b * out_b).sum())
        cos_sim = dot / (norm_a * norm_b + 1e-8)
        mx.eval(cos_sim)

        interference[l_idx] = 1.0 - cos_sim.item()  # cosine distance

    return interference


def run_single_seed(model_name, layer_types, docs, tokenizer, seed):
    """Run the full composition protocol for one seed, return metrics."""
    V = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        domain_datasets[name] = (
            CharDataset(dtrain, tokenizer, BASE["block_size"]),
            CharDataset(dval, tokenizer, BASE["block_size"]),
        )
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    kwargs = dict(vocab_size=V, **CAP)
    if layer_types is not None:
        kwargs["layer_types"] = layer_types

    # --- Joint training baseline ---
    model_joint = get_model(model_name, **kwargs)
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

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2

    # --- Pretrain base ---
    base_model = get_model(model_name, **kwargs)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # --- Fine-tune domain A ---
    model_a = get_model(model_name, **kwargs)
    mx.eval(model_a.parameters())
    model_a.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_a.parameters())
    model_a.freeze()
    for layer in model_a.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_a, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_a.unfreeze()

    # --- Fine-tune domain B ---
    model_b = get_model(model_name, **kwargs)
    mx.eval(model_b.parameters())
    model_b.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_b.parameters())
    model_b.freeze()
    for layer in model_b.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_b, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_b.unfreeze()

    # --- Compute per-layer interference ---
    # Use a mixed val set for interference measurement
    all_val_ds = CharDataset(all_val, tokenizer, BASE["block_size"])
    interference = compute_per_layer_interference(
        model_a, model_b, base_model, all_val_ds, BASE["n_layer"]
    )

    # --- Compose + calibrate ---
    composed = compose_models(model_a, model_b, model_name, V, BASE["block_size"],
                              base_model=base_model, layer_types=layer_types)
    calibrate_router(composed, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2

    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "gap_pct": gap_pct,
        "interference": interference,
    }


def run_experiment(n_seeds=7):
    """Run delta rule interference ordering experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    seeds = list(range(n_seeds))

    conditions = {
        "full_attn": {
            "model_name": "full_attn_capsule_moe",
            "layer_types": ["full", "full", "full", "full"],
        },
        "l2_norm_3_1": {
            "model_name": "l2_norm_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
        },
        "delta_rule_3_1": {
            "model_name": "delta_rule_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
        },
    }

    all_results = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"\n{'='*70}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*70}")

        cond_results = []
        for i, seed in enumerate(seeds):
            print(f"  Seed {seed} ({i+1}/{n_seeds})...", end=" ", flush=True)
            result = run_single_seed(
                model_name=cond_cfg["model_name"],
                layer_types=cond_cfg["layer_types"],
                docs=docs,
                tokenizer=tokenizer,
                seed=seed,
            )
            print(f"gap={result['gap_pct']:+.2f}%  "
                  f"interf=[" + ", ".join(f"{result['interference'][l]:.3f}" for l in range(4)) + "]")
            cond_results.append(result)

        all_results[cond_name] = cond_results

    # --- Analysis ---
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        print(f"\n--- {cond_name} ---")
        print(f"  Gap mean:   {statistics.mean(gaps):+.2f}%")
        print(f"  Gap median: {statistics.median(gaps):+.2f}%")
        print(f"  Gap std:    {statistics.stdev(gaps):.2f}%")
        print(f"  Gap min:    {min(gaps):+.2f}%")
        print(f"  Gap max:    {max(gaps):+.2f}%")

        # Per-layer interference
        for l_idx in range(BASE["n_layer"]):
            layer_interf = [r["interference"][l_idx] for r in results]
            print(f"  Layer {l_idx} interference: mean={statistics.mean(layer_interf):.4f}  "
                  f"std={statistics.stdev(layer_interf):.4f}")

    # --- Kill Criterion 1: Interference ordering ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 1: Interference ordering (threshold: ratio >1.0x)")
    print(f"{'='*70}")

    # Compute mean interference for linear layers (0, 1, 2) and full layer (3)
    # For delta_rule condition
    delta_results = all_results["delta_rule_3_1"]
    full_results = all_results["full_attn"]

    # Delta rule: linear interference = mean of layers 0, 1, 2 (excluding layer 0 if near-zero)
    delta_linear_interf = []
    delta_full_interf = []
    for r in delta_results:
        # Layers 0-2 are linear, layer 3 is full
        linear_layers = [r["interference"][l] for l in [1, 2]]  # exclude Layer 0 (shared base)
        full_layer = r["interference"][3]
        delta_linear_interf.extend(linear_layers)
        delta_full_interf.append(full_layer)

    # Also compute for l2_norm baseline
    l2_results = all_results["l2_norm_3_1"]
    l2_linear_interf = []
    l2_full_interf = []
    for r in l2_results:
        linear_layers = [r["interference"][l] for l in [1, 2]]
        full_layer = r["interference"][3]
        l2_linear_interf.extend(linear_layers)
        l2_full_interf.append(full_layer)

    delta_linear_mean = statistics.mean(delta_linear_interf)
    delta_full_mean = statistics.mean(delta_full_interf)
    delta_ratio = delta_linear_mean / (delta_full_mean + 1e-8)

    l2_linear_mean = statistics.mean(l2_linear_interf)
    l2_full_mean = statistics.mean(l2_full_interf)
    l2_ratio = l2_linear_mean / (l2_full_mean + 1e-8)

    print(f"\n  L2 norm (simplified, baseline):")
    print(f"    Linear layers (1,2) mean interference: {l2_linear_mean:.4f}")
    print(f"    Full layer (3) mean interference:      {l2_full_mean:.4f}")
    print(f"    Ratio (linear/full): {l2_ratio:.2f}x")

    print(f"\n  Delta rule:")
    print(f"    Linear layers (1,2) mean interference: {delta_linear_mean:.4f}")
    print(f"    Full layer (3) mean interference:      {delta_full_mean:.4f}")
    print(f"    Ratio (linear/full): {delta_ratio:.2f}x")

    if delta_ratio > 1.0:
        print(f"\n  ** KILL: Delta rule reverses ordering ({delta_ratio:.2f}x > 1.0x) **")
        k1_pass = False
    else:
        print(f"\n  PASS: Delta rule maintains ordering ({delta_ratio:.2f}x <= 1.0x)")
        k1_pass = True

    # --- Kill Criterion 2: Composition gap ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 2: Composition gap (threshold: +10% median)")
    print(f"{'='*70}")

    delta_gaps = [r["gap_pct"] for r in delta_results]
    delta_median = statistics.median(delta_gaps)

    print(f"  Delta rule median gap: {delta_median:+.2f}%")

    if delta_median > 10.0:
        print(f"  ** KILL: Median gap {delta_median:+.2f}% > +10% threshold **")
        k2_pass = False
    else:
        print(f"  PASS: Median gap {delta_median:+.2f}% <= +10% threshold")
        k2_pass = True

    # --- Per-seed comparison table ---
    print(f"\n{'='*70}")
    print("Per-seed composition gaps")
    print(f"{'='*70}")
    print(f"{'Seed':>6} {'Full Attn':>10} {'L2 Norm':>10} {'Delta Rule':>10}")
    print("-" * 40)
    for i, seed in enumerate(seeds):
        f_gap = full_results[i]["gap_pct"]
        l_gap = l2_results[i]["gap_pct"]
        d_gap = delta_results[i]["gap_pct"]
        print(f"{seed:>6} {f_gap:>+9.2f}% {l_gap:>+9.2f}% {d_gap:>+9.2f}%")

    # --- Per-layer interference comparison ---
    print(f"\n{'='*70}")
    print("Per-layer mean interference")
    print(f"{'='*70}")
    print(f"{'Layer':>6} {'Type':>10} {'Full Attn':>10} {'L2 Norm':>10} {'Delta Rule':>10}")
    print("-" * 50)
    for l_idx in range(BASE["n_layer"]):
        full_interf = statistics.mean([r["interference"][l_idx] for r in full_results])
        l2_interf = statistics.mean([r["interference"][l_idx] for r in l2_results])
        delta_interf = statistics.mean([r["interference"][l_idx] for r in delta_results])
        lt = "linear" if l_idx < 3 else "full"
        print(f"{l_idx:>6} {lt:>10} {full_interf:>10.4f} {l2_interf:>10.4f} {delta_interf:>10.4f}")

    # --- Summary verdict ---
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    if k1_pass and k2_pass:
        print("  PASS: Delta rule does NOT reverse interference ordering and gap is acceptable")
        print(f"  Delta rule ratio: {delta_ratio:.2f}x (simplified baseline: {l2_ratio:.2f}x)")
        print(f"  Delta rule median gap: {delta_median:+.2f}%")
    elif not k1_pass:
        print("  KILL (criterion 1): Delta rule REVERSES interference ordering")
        print(f"  Delta rule ratio: {delta_ratio:.2f}x > 1.0x threshold")
    elif not k2_pass:
        print("  KILL (criterion 2): Delta rule composition gap too large")
        print(f"  Delta rule median gap: {delta_median:+.2f}% > +10% threshold")

    # Save results to JSON
    output = {
        "n_seeds": n_seeds,
        "elapsed_seconds": elapsed,
        "conditions": {},
    }
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        output["conditions"][cond_name] = {
            "gap_mean": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_std": statistics.stdev(gaps),
            "gap_min": min(gaps),
            "gap_max": max(gaps),
            "per_seed": [
                {
                    "seed": seeds[i],
                    "gap_pct": r["gap_pct"],
                    "joint_avg": r["joint_avg"],
                    "composed_avg": r["composed_avg"],
                    "interference": {str(k): v for k, v in r["interference"].items()},
                }
                for i, r in enumerate(results)
            ],
        }
    output["kill_criteria"] = {
        "k1_delta_ratio": delta_ratio,
        "k1_l2_ratio": l2_ratio,
        "k1_threshold": 1.0,
        "k1_pass": k1_pass,
        "k2_delta_median_gap": delta_median,
        "k2_threshold": 10.0,
        "k2_pass": k2_pass,
        "overall_pass": k1_pass and k2_pass,
    }

    results_path = "experiments/models/delta_rule_attention/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=7)
