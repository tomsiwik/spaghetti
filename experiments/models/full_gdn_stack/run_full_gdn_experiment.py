"""Full GatedDeltaNet composition stack experiment.

Tests whether all GatedDeltaNet components combined (L2 norm + delta rule +
conv1d + per-dim beta + SiLU gate) create emergent composition interference
not present when components were tested individually.

Protocol: identical to delta_rule_attention/run_delta_rule_experiment.py
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets
  Phase 6: Compute per-layer interference metrics

Four conditions:
  1. full_attn: all 4 layers full attention (control)
  2. delta_rule_3_1: delta rule linear with L2 norm (baseline from prior exp)
  3. full_gdn_3_1: full GatedDeltaNet stack (test condition)
  4. full_gdn_3_1_no_conv: full stack minus conv1d (ablation)

7 seeds per condition (matching prior experiments).

Kill criteria:
  1. Full GDN composition gap >5% (median across seeds)
  2. Component interaction produces >2x interference vs delta-rule-only model
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


# Standard micro config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def compose_models(model_a, model_b, model_name, vocab_size, block_size,
                   base_model=None, layer_types=None, extra_kwargs=None):
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
    if extra_kwargs:
        kwargs.update(extra_kwargs)

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

        # Copy attention weights from base (all Linear layers)
        for name in ["wq", "wk", "wv", "wo", "wg", "w_a", "w_beta", "w_z"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        # Copy non-Linear params
        for name in ["dt_bias", "A_log"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                setattr(layer_c.attn, name, getattr(layer_src.attn, name))

        # Copy conv1d weights if they exist
        for conv_name in ["conv_q", "conv_k", "conv_v"]:
            if hasattr(layer_src.attn, conv_name) and hasattr(layer_c.attn, conv_name):
                src_conv = getattr(layer_src.attn, conv_name)
                dst_conv = getattr(layer_c.attn, conv_name)
                dst_conv.weight = src_conv.weight

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
    """Compute per-layer interference as cosine distance between domain-specific outputs."""
    rng = random.Random(0)
    inputs, _ = val_ds.get_batch(BATCH_SIZE, rng)

    interference = {}
    for l_idx in range(n_layers):
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

        dot = (out_a * out_b).sum()
        norm_a = mx.sqrt((out_a * out_a).sum())
        norm_b = mx.sqrt((out_b * out_b).sum())
        cos_sim = dot / (norm_a * norm_b + 1e-8)
        mx.eval(cos_sim)

        interference[l_idx] = 1.0 - cos_sim.item()

    return interference


def run_single_seed(model_name, layer_types, docs, tokenizer, seed,
                    extra_kwargs=None):
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
    if extra_kwargs:
        kwargs.update(extra_kwargs)

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
    all_val_ds = CharDataset(all_val, tokenizer, BASE["block_size"])
    interference = compute_per_layer_interference(
        model_a, model_b, base_model, all_val_ds, BASE["n_layer"]
    )

    # --- Compose + calibrate ---
    composed = compose_models(model_a, model_b, model_name, V, BASE["block_size"],
                              base_model=base_model, layer_types=layer_types,
                              extra_kwargs=extra_kwargs)
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
    """Run full GDN stack composition experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    seeds = list(range(n_seeds))

    conditions = {
        "full_attn": {
            "model_name": "full_attn_capsule_moe",
            "layer_types": ["full", "full", "full", "full"],
            "extra_kwargs": None,
        },
        "delta_rule_3_1": {
            "model_name": "delta_rule_hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "extra_kwargs": None,
        },
        "full_gdn_3_1": {
            "model_name": "full_gdn_stack_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "extra_kwargs": None,
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
                extra_kwargs=cond_cfg.get("extra_kwargs"),
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

        for l_idx in range(BASE["n_layer"]):
            layer_interf = [r["interference"][l_idx] for r in results]
            print(f"  Layer {l_idx} interference: mean={statistics.mean(layer_interf):.4f}  "
                  f"std={statistics.stdev(layer_interf):.4f}")

    # --- Kill Criterion 1: Composition gap ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 1: Full GDN composition gap (threshold: >5%)")
    print(f"{'='*70}")

    gdn_results = all_results["full_gdn_3_1"]
    gdn_gaps = [r["gap_pct"] for r in gdn_results]
    gdn_median = statistics.median(gdn_gaps)

    print(f"  Full GDN median gap: {gdn_median:+.2f}%")
    if gdn_median > 5.0:
        print(f"  ** KILL: Median gap {gdn_median:+.2f}% > +5% threshold **")
        k1_pass = False
    else:
        print(f"  PASS: Median gap {gdn_median:+.2f}% <= +5% threshold")
        k1_pass = True

    # --- Kill Criterion 2: Component interaction interference ---
    print(f"\n{'='*70}")
    print("KILL CRITERION 2: Interference ratio vs delta-rule-only (threshold: >2.0x)")
    print(f"{'='*70}")

    delta_results = all_results["delta_rule_3_1"]

    # Compute mean interference for linear layers 1,2 (excluding layer 0)
    gdn_linear_interf = []
    delta_linear_interf = []
    for r in gdn_results:
        gdn_linear_interf.extend([r["interference"][l] for l in [1, 2]])
    for r in delta_results:
        delta_linear_interf.extend([r["interference"][l] for l in [1, 2]])

    gdn_interf_mean = statistics.mean(gdn_linear_interf)
    delta_interf_mean = statistics.mean(delta_linear_interf)
    interf_ratio = gdn_interf_mean / (delta_interf_mean + 1e-8)

    print(f"  Delta rule linear (L1,2) mean interference: {delta_interf_mean:.4f}")
    print(f"  Full GDN linear (L1,2) mean interference:   {gdn_interf_mean:.4f}")
    print(f"  Ratio (full_gdn / delta_rule): {interf_ratio:.2f}x")

    if interf_ratio > 2.0:
        print(f"  ** KILL: Interference ratio {interf_ratio:.2f}x > 2.0x threshold **")
        k2_pass = False
    else:
        print(f"  PASS: Interference ratio {interf_ratio:.2f}x <= 2.0x threshold")
        k2_pass = True

    # --- Interference ordering check ---
    print(f"\n{'='*70}")
    print("INTERFERENCE ORDERING (linear < full maintained?)")
    print(f"{'='*70}")

    full_results = all_results["full_attn"]

    for cond_name in ["delta_rule_3_1", "full_gdn_3_1"]:
        results = all_results[cond_name]
        lin_interf = []
        full_interf = []
        for r in results:
            lin_interf.extend([r["interference"][l] for l in [1, 2]])
            full_interf.append(r["interference"][3])
        lin_mean = statistics.mean(lin_interf)
        full_mean = statistics.mean(full_interf)
        ratio = lin_mean / (full_mean + 1e-8)
        print(f"\n  {cond_name}:")
        print(f"    Linear (L1,2) mean: {lin_mean:.4f}")
        print(f"    Full (L3) mean:     {full_mean:.4f}")
        print(f"    Ratio: {ratio:.2f}x {'(PASS)' if ratio <= 1.0 else '(FAIL)'}")

    # --- Per-seed comparison table ---
    print(f"\n{'='*70}")
    print("Per-seed composition gaps")
    print(f"{'='*70}")
    print(f"{'Seed':>6} {'Full Attn':>10} {'Delta Rule':>12} {'Full GDN':>10}")
    print("-" * 42)
    for i, seed in enumerate(seeds):
        f_gap = full_results[i]["gap_pct"]
        d_gap = delta_results[i]["gap_pct"]
        g_gap = gdn_results[i]["gap_pct"]
        print(f"{seed:>6} {f_gap:>+9.2f}% {d_gap:>+11.2f}% {g_gap:>+9.2f}%")

    # --- Per-layer interference comparison ---
    print(f"\n{'='*70}")
    print("Per-layer mean interference")
    print(f"{'='*70}")
    print(f"{'Layer':>6} {'Type':>10} {'Full Attn':>10} {'Delta Rule':>12} {'Full GDN':>10}")
    print("-" * 52)
    for l_idx in range(BASE["n_layer"]):
        full_interf = statistics.mean([r["interference"][l_idx] for r in full_results])
        delta_interf = statistics.mean([r["interference"][l_idx] for r in delta_results])
        gdn_interf = statistics.mean([r["interference"][l_idx] for r in gdn_results])
        lt = "linear" if l_idx < 3 else "full"
        print(f"{l_idx:>6} {lt:>10} {full_interf:>10.4f} {delta_interf:>12.4f} {gdn_interf:>10.4f}")

    # --- Summary verdict ---
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    if k1_pass and k2_pass:
        print("  PASS: Full GatedDeltaNet stack is composition-compatible")
        print(f"  Composition gap: {gdn_median:+.2f}% (threshold: 5%)")
        print(f"  Interference ratio vs delta-rule: {interf_ratio:.2f}x (threshold: 2.0x)")
        print("  All GatedDeltaNet components can be used together for composition.")
    elif not k1_pass:
        print("  KILL (criterion 1): Full GDN composition gap too large")
        print(f"  Median gap: {gdn_median:+.2f}% > +5% threshold")
    elif not k2_pass:
        print("  KILL (criterion 2): Component interaction creates excessive interference")
        print(f"  Interference ratio: {interf_ratio:.2f}x > 2.0x threshold")

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
        "k1_gdn_median_gap": gdn_median,
        "k1_threshold": 5.0,
        "k1_pass": k1_pass,
        "k2_interference_ratio": interf_ratio,
        "k2_threshold": 2.0,
        "k2_pass": k2_pass,
        "overall_pass": k1_pass and k2_pass,
    }

    results_path = "experiments/models/full_gdn_stack/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=7)
