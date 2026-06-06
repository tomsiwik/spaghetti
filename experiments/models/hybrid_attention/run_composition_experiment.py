"""Composition experiment: hybrid attention vs full attention.

Compares capsule composition quality between:
  1. full_attn_capsule_moe: all 4 layers use standard causal self-attention
  2. hybrid_capsule_moe: 3 linear + 1 full attention (3:1 ratio like Qwen3.5)

Protocol (established in capsule_moe/test_composition.py):
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune only capsule groups per domain (freeze attention) -- 300 steps
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets

Kill criteria:
  1. Hybrid composition degrades >10% vs full attention composition
  2. Linear attention layers show >2x higher composition interference than full layers

Per-layer interference is measured by weight-space divergence of attention outputs
before vs after composition (cosine distance between pre/post composition hidden states).
"""

import copy
import random
import statistics
import time

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
                   base_model=None, layer_types=None):
    """Create a composed model by concatenating capsule groups from A and B.

    Uses shared base model for attention weights.
    Capsule groups from model_a go to slots 0..G-1, from model_b to slots G..2G-1.
    Router is randomly initialized (will be calibrated).
    """
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

    # Use base model for shared params (attention, embeddings)
    source = base_model if base_model is not None else model_a
    composed.wte.weight = source.wte.weight
    composed.wpe.weight = source.wpe.weight
    composed.lm_head.weight = source.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_src = source.layers[l_idx]
        layer_a = model_a.layers[l_idx]
        layer_b = model_b.layers[l_idx]

        # Copy attention from base/source
        for name in ["wq", "wk", "wv", "wo"]:
            src_param = getattr(layer_src.attn, name).weight
            getattr(layer_c.attn, name).weight = src_param
        # Copy gate weights for linear attention layers
        if hasattr(layer_src.attn, "wg") and hasattr(layer_c.attn, "wg"):
            layer_c.attn.wg.weight = layer_src.attn.wg.weight

        # Copy capsule groups: first G from A, next G from B
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

        if step % 50 == 0 or step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


def measure_per_layer_interference(base_model, composed_model, val_ds, n_batches=10):
    """Measure how much each layer's output changes between base and composed models.

    For each layer, compute the mean L2 distance (normalized) of attention output
    between the base model and the composed model when processing the same inputs.

    This isolates whether linear attention layers show more composition interference
    than full attention layers.

    Returns: dict mapping layer_idx -> {"type": str, "interference": float}
    """
    rng = random.Random(0)
    results = {}

    for l_idx in range(len(base_model.layers)):
        layer_type = base_model.layer_types[l_idx]
        total_diff = 0.0
        total_norm = 0.0
        count = 0

        for _ in range(n_batches):
            inputs, _ = val_ds.get_batch(BATCH_SIZE, rng)

            # Run through layers up to l_idx, capture attention output
            # Base model
            x_base = base_model.wte(inputs) + base_model.wpe(mx.arange(inputs.shape[1]))
            x_base = base_model.norm0(x_base)
            for i in range(l_idx):
                x_base = base_model.layers[i](x_base)
            # Get attention output at this layer (before residual + MLP)
            attn_out_base = base_model.layers[l_idx].attn(
                base_model.layers[l_idx].norm1(x_base))

            # Composed model
            x_comp = composed_model.wte(inputs) + composed_model.wpe(mx.arange(inputs.shape[1]))
            x_comp = composed_model.norm0(x_comp)
            for i in range(l_idx):
                x_comp = composed_model.layers[i](x_comp)
            attn_out_comp = composed_model.layers[l_idx].attn(
                composed_model.layers[l_idx].norm1(x_comp))

            mx.eval(attn_out_base, attn_out_comp)

            # Normalized L2 distance
            diff = mx.sqrt(mx.mean((attn_out_base - attn_out_comp) ** 2)).item()
            norm = mx.sqrt(mx.mean(attn_out_base ** 2)).item() + 1e-8
            total_diff += diff / norm
            count += 1

        results[l_idx] = {
            "type": layer_type,
            "interference": total_diff / count,
        }

    return results


def run_single_condition(model_name, layer_types, condition_name, docs, tokenizer, seed=42):
    """Run the full composition protocol for one attention condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name} (seed={seed})")
    print(f"{'='*70}")

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
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # --- Joint training baseline ---
    print(f"\n--- Joint training ({condition_name}) ---")
    kwargs = dict(vocab_size=V, **CAP)
    if layer_types is not None:
        kwargs["layer_types"] = layer_types
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
        if step % 200 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2
    results["joint"] = {"a_m": joint_val_a, "n_z": joint_val_b, "avg": joint_avg}
    print(f"  Joint: a_m={joint_val_a:.4f}, n_z={joint_val_b:.4f}, avg={joint_avg:.4f}")

    # --- Phase 1: Pretrain base ---
    print(f"\n--- Pretrain base ({condition_name}) ---")
    kwargs_base = dict(vocab_size=V, **CAP)
    if layer_types is not None:
        kwargs_base["layer_types"] = layer_types
    base_model = get_model(model_name, **kwargs_base)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    # --- Phase 2a: Fine-tune capsules on domain A ---
    print(f"\n--- Fine-tune domain A ({condition_name}) ---")
    model_a = get_model(model_name, **kwargs_base)
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
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_a.unfreeze()

    # --- Phase 2b: Fine-tune capsules on domain B ---
    print(f"\n--- Fine-tune domain B ({condition_name}) ---")
    model_b = get_model(model_name, **kwargs_base)
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
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_b.unfreeze()

    # --- Phase 3: Compose + calibrate ---
    print(f"\n--- Compose + calibrate ({condition_name}) ---")
    composed = compose_models(model_a, model_b, model_name, V, BASE["block_size"],
                              base_model=base_model, layer_types=layer_types)
    calibrate_router(composed, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2
    results["composed"] = {"a_m": comp_val_a, "n_z": comp_val_b, "avg": comp_avg}
    print(f"  Composed: a_m={comp_val_a:.4f}, n_z={comp_val_b:.4f}, avg={comp_avg:.4f}")

    # --- Composition gap ---
    gap_pct = (comp_avg - joint_avg) / joint_avg * 100
    results["gap_pct"] = gap_pct
    print(f"  Gap vs joint: {gap_pct:+.2f}%")

    # --- Per-layer interference measurement ---
    print(f"\n--- Per-layer interference ({condition_name}) ---")
    interference = measure_per_layer_interference(base_model, composed, joint_val)
    results["interference"] = interference
    for l_idx, info in interference.items():
        print(f"  Layer {l_idx} ({info['type']}): interference = {info['interference']:.4f}")

    # Single-domain eval (for completeness)
    spec_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_b = evaluate(model_b, val_b, BATCH_SIZE)
    single_avg = (spec_a + spec_b) / 2
    results["single_avg"] = single_avg
    print(f"\n  Single-domain avg: {single_avg:.4f}")

    return results


def run_experiment(seeds=(42, 123, 777)):
    """Run the full experiment across conditions and seeds."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    conditions = {
        "full_attn": {
            "model_name": "full_attn_capsule_moe",
            "layer_types": ["full"] * 4,
        },
        "hybrid_3_1": {
            "model_name": "hybrid_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
        },
    }

    all_results = {}
    for cond_name, cond_cfg in conditions.items():
        all_results[cond_name] = {}
        for seed in seeds:
            result = run_single_condition(
                model_name=cond_cfg["model_name"],
                layer_types=cond_cfg["layer_types"],
                condition_name=f"{cond_name} (seed={seed})",
                docs=docs,
                tokenizer=tokenizer,
                seed=seed,
            )
            all_results[cond_name][seed] = result

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT SUMMARY ({len(seeds)} seeds, {elapsed:.0f}s total)")
    print(f"{'='*70}")

    print(f"\n{'Condition':<16} {'Joint':>10} {'Composed':>10} {'Single':>10} {'Gap (%)':>10}")
    print("-" * 60)
    for cond_name in conditions:
        joints = [all_results[cond_name][s]["joint"]["avg"] for s in seeds]
        comps = [all_results[cond_name][s]["composed"]["avg"] for s in seeds]
        singles = [all_results[cond_name][s]["single_avg"] for s in seeds]
        gaps = [all_results[cond_name][s]["gap_pct"] for s in seeds]
        print(f"{cond_name:<16} {statistics.mean(joints):>10.4f} "
              f"{statistics.mean(comps):>10.4f} {statistics.mean(singles):>10.4f} "
              f"{statistics.mean(gaps):>+9.2f}%")
        if len(seeds) > 1:
            print(f"{'  (std)':<16} {statistics.stdev(joints):>10.4f} "
                  f"{statistics.stdev(comps):>10.4f} {statistics.stdev(singles):>10.4f} "
                  f"{statistics.stdev(gaps):>10.2f}")

    # --- Kill criterion 1: hybrid composition gap vs full composition gap ---
    print(f"\n--- KILL CRITERION 1: Hybrid composition gap vs full ---")
    full_gaps = [all_results["full_attn"][s]["gap_pct"] for s in seeds]
    hybrid_gaps = [all_results["hybrid_3_1"][s]["gap_pct"] for s in seeds]
    full_mean_gap = statistics.mean(full_gaps)
    hybrid_mean_gap = statistics.mean(hybrid_gaps)

    # "degrades >10%" means hybrid gap is >10pp worse than full gap
    full_comp_means = statistics.mean([all_results["full_attn"][s]["composed"]["avg"] for s in seeds])
    hybrid_comp_means = statistics.mean([all_results["hybrid_3_1"][s]["composed"]["avg"] for s in seeds])
    full_joint_means = statistics.mean([all_results["full_attn"][s]["joint"]["avg"] for s in seeds])
    hybrid_joint_means = statistics.mean([all_results["hybrid_3_1"][s]["joint"]["avg"] for s in seeds])

    full_rel = (full_comp_means - full_joint_means) / full_joint_means * 100
    hybrid_rel = (hybrid_comp_means - hybrid_joint_means) / hybrid_joint_means * 100

    # Also compute median gaps (robust to outliers)
    full_median_gap = statistics.median(full_gaps)
    hybrid_median_gap = statistics.median(hybrid_gaps)

    print(f"  Full attention:   mean gap = {full_rel:+.2f}%, median gap = {full_median_gap:+.2f}%")
    print(f"  Hybrid attention: mean gap = {hybrid_rel:+.2f}%, median gap = {hybrid_median_gap:+.2f}%")
    degradation_mean = hybrid_rel - full_rel
    degradation_median = hybrid_median_gap - full_median_gap
    print(f"  Degradation (mean):   {degradation_mean:+.2f}pp")
    print(f"  Degradation (median): {degradation_median:+.2f}pp")
    if len(seeds) >= 3:
        # Flag if any single seed drives >50% of the mean gap effect
        hybrid_gap_vals = [all_results["hybrid_3_1"][s]["gap_pct"] for s in seeds]
        for i, s in enumerate(seeds):
            leave_one_out = [g for j, g in enumerate(hybrid_gap_vals) if j != i]
            loo_mean = statistics.mean(leave_one_out)
            contribution = abs(hybrid_mean_gap - loo_mean) / (abs(hybrid_mean_gap) + 1e-8) * 100
            if contribution > 50:
                print(f"  WARNING: Seed {s} contributes {contribution:.0f}% of mean hybrid gap effect")
    if max(degradation_mean, degradation_median) > 10.0:
        print(f"  ** KILL: hybrid degrades >10pp threshold **")
    else:
        print(f"  PASS: within 10pp threshold (both mean and median)")

    # --- Kill criterion 2: per-layer interference comparison ---
    print(f"\n--- KILL CRITERION 2: Linear vs full attention layer interference ---")
    linear_intf_all = []
    linear_intf_excl0 = []  # excluding Layer 0
    full_intf_all = []
    for seed in seeds:
        intf = all_results["hybrid_3_1"][seed]["interference"]
        for l_idx, info in intf.items():
            if info["type"] == "linear":
                linear_intf_all.append(info["interference"])
                if l_idx != 0:
                    linear_intf_excl0.append(info["interference"])
            else:
                full_intf_all.append(info["interference"])

    if linear_intf_all and full_intf_all:
        mean_linear = statistics.mean(linear_intf_all)
        mean_linear_excl0 = statistics.mean(linear_intf_excl0) if linear_intf_excl0 else 0.0
        mean_full = statistics.mean(full_intf_all)
        ratio_inclusive = mean_linear / (mean_full + 1e-8)
        ratio_exclusive = mean_linear_excl0 / (mean_full + 1e-8)
        print(f"  Mean linear layer interference (all):      {mean_linear:.4f}")
        print(f"  Mean linear layer interference (excl L0):  {mean_linear_excl0:.4f}")
        print(f"  Mean full attention layer interference:     {mean_full:.4f}")
        print(f"  Ratio inclusive  (linear/full): {ratio_inclusive:.2f}x")
        print(f"  Ratio exclusive  (linear excl L0 / full): {ratio_exclusive:.2f}x")
        print(f"  NOTE: Layer 0 shows zero interference due to shared base weights,")
        print(f"        not attention type. The exclusive ratio is the honest metric.")
        if ratio_exclusive > 2.0:
            print(f"  ** KILL: exclusive ratio {ratio_exclusive:.2f}x > 2x threshold **")
        else:
            print(f"  PASS: within 2x threshold (exclusive ratio)")
    else:
        print("  WARNING: insufficient data for comparison")

    # --- Full attention per-layer interference (depth confound check) ---
    print(f"\n--- FULL ATTENTION per-layer interference (depth confound check) ---")
    for seed in seeds:
        if "interference" in all_results["full_attn"][seed]:
            print(f"  Seed {seed}:")
            intf = all_results["full_attn"][seed]["interference"]
            for l_idx in sorted(intf.keys()):
                info = intf[l_idx]
                print(f"    Layer {l_idx} ({info['type']:>6}): {info['interference']:.4f}")
    # Aggregate full_attn per-layer means
    full_attn_layer_means = {}
    for l_idx in range(BASE["n_layer"]):
        vals = []
        for seed in seeds:
            if "interference" in all_results["full_attn"][seed]:
                intf = all_results["full_attn"][seed]["interference"]
                if l_idx in intf:
                    vals.append(intf[l_idx]["interference"])
        if vals:
            full_attn_layer_means[l_idx] = statistics.mean(vals)
    if full_attn_layer_means:
        print(f"  Mean across seeds:")
        for l_idx in sorted(full_attn_layer_means.keys()):
            print(f"    Layer {l_idx}: {full_attn_layer_means[l_idx]:.4f}")
        if 2 in full_attn_layer_means and 3 in full_attn_layer_means:
            l2_l3_gap = full_attn_layer_means[3] - full_attn_layer_means[2]
            print(f"  Layer 3 - Layer 2 gap (full_attn): {l2_l3_gap:+.4f}")
            print(f"  If Layer 3 > Layer 2 in full_attn, depth confound is confirmed.")

    # --- Per-seed details ---
    print(f"\n--- Per-seed composition gaps ---")
    print(f"{'Seed':>6} {'Full gap':>10} {'Hybrid gap':>12}")
    for seed in seeds:
        fg = all_results["full_attn"][seed]["gap_pct"]
        hg = all_results["hybrid_3_1"][seed]["gap_pct"]
        print(f"{seed:>6} {fg:>+9.2f}% {hg:>+11.2f}%")

    # --- Per-layer interference details (all seeds) ---
    print(f"\n--- Per-layer interference (hybrid model, all seeds) ---")
    for seed in seeds:
        print(f"  Seed {seed}:")
        intf = all_results["hybrid_3_1"][seed]["interference"]
        for l_idx in sorted(intf.keys()):
            info = intf[l_idx]
            print(f"    Layer {l_idx} ({info['type']:>6}): {info['interference']:.4f}")

    return all_results


if __name__ == "__main__":
    run_experiment()
