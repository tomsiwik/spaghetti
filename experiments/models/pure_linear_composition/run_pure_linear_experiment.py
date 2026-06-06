"""Pure-linear composition control experiment.

Tests whether pure-linear attention (4:0 all GatedDeltaNet, no full attention
scaffolding) can compose capsule groups comparably to the validated 3:1
hybrid configuration.

This is a CONTROL experiment identified by adversarial review. The hybrid
experiment proved 3:1 linear:full is composition-compatible. But it left
ambiguous whether linear attention NEEDS at least one full attention layer
to provide "composition scaffolding" -- a global context layer that helps
domain-specific capsule outputs integrate.

Hypothesis: Pure-linear (4:0) composition degrades >5% vs hybrid (3:1).
Kill criteria: "pure-linear composition degrades >5% vs hybrid 3:1 composition"

Protocol: identical to full_gdn_stack/run_full_gdn_experiment.py
  Phase 1: Pretrain shared base on all data (300 steps)
  Phase 2: Fine-tune capsule groups per domain (300 steps, attention frozen)
  Phase 3: Compose by concatenating domain groups, double top-k
  Phase 4: Calibrate router on mixed data (100 steps)
  Phase 5: Evaluate on per-domain val sets
  Phase 6: Compute per-layer interference metrics

Three conditions:
  1. full_attn: all 4 layers full attention (control baseline)
  2. hybrid_3_1: 3 linear + 1 full (validated condition)
  3. pure_linear: all 4 layers GatedDeltaNet linear (test condition)

7 seeds per condition.
"""

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

        # Copy norms
        for norm_name in ["norm1", "norm2"]:
            src_norm = getattr(layer_src, norm_name)
            dst_norm = getattr(layer_c, norm_name)
            if hasattr(src_norm, "weight") and hasattr(dst_norm, "weight"):
                dst_norm.weight = src_norm.weight

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

    # Single-domain avg for reference
    spec_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_b = evaluate(model_b, val_b, BATCH_SIZE)
    single_avg = (spec_a + spec_b) / 2

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "single_avg": single_avg,
        "gap_pct": gap_pct,
        "interference": interference,
    }


def run_experiment(n_seeds=7):
    """Run pure-linear composition control experiment."""
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
        "hybrid_3_1": {
            "model_name": "full_gdn_stack_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "full"],
            "extra_kwargs": None,
        },
        "pure_linear": {
            "model_name": "full_gdn_stack_capsule_moe",
            "layer_types": ["linear", "linear", "linear", "linear"],
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
            try:
                result = run_single_seed(
                    model_name=cond_cfg["model_name"],
                    layer_types=cond_cfg["layer_types"],
                    docs=docs,
                    tokenizer=tokenizer,
                    seed=seed,
                    extra_kwargs=cond_cfg.get("extra_kwargs"),
                )
                print(f"gap={result['gap_pct']:+.2f}%  "
                      f"joint={result['joint_avg']:.4f}  "
                      f"comp={result['composed_avg']:.4f}  "
                      f"interf=[" + ", ".join(f"{result['interference'][l]:.3f}" for l in range(4)) + "]")
                cond_results.append(result)
            except RuntimeError as e:
                print(f"FAILED (RuntimeError: {e})")
                continue

        if len(cond_results) < 3:
            print(f"  WARNING: Only {len(cond_results)} seeds succeeded, need at least 3")
        all_results[cond_name] = cond_results

    # Check we have enough data
    for cond_name, results in all_results.items():
        if len(results) < 3:
            print(f"\nFATAL: {cond_name} has only {len(results)} results, need >= 3. Aborting.")
            return all_results

    # =========================================================================
    # Analysis
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    # --- Per-condition summary ---
    for cond_name, results in all_results.items():
        gaps = [r["gap_pct"] for r in results]
        joints = [r["joint_avg"] for r in results]
        comps = [r["composed_avg"] for r in results]
        singles = [r["single_avg"] for r in results]
        print(f"\n--- {cond_name} ---")
        print(f"  Joint:    mean={statistics.mean(joints):.4f}  std={statistics.stdev(joints):.4f}")
        print(f"  Composed: mean={statistics.mean(comps):.4f}  std={statistics.stdev(comps):.4f}")
        print(f"  Single:   mean={statistics.mean(singles):.4f}  std={statistics.stdev(singles):.4f}")
        print(f"  Gap mean:   {statistics.mean(gaps):+.2f}%")
        print(f"  Gap median: {statistics.median(gaps):+.2f}%")
        print(f"  Gap std:    {statistics.stdev(gaps):.2f}%")
        print(f"  Gap range:  [{min(gaps):+.2f}%, {max(gaps):+.2f}%]")

    # =========================================================================
    # KILL CRITERION: pure-linear gap vs hybrid gap
    # =========================================================================
    print(f"\n{'='*70}")
    print("KILL CRITERION: Pure-linear composition degrades >5% vs hybrid 3:1")
    print(f"{'='*70}")

    hybrid_results = all_results["hybrid_3_1"]
    pure_results = all_results["pure_linear"]
    full_results = all_results["full_attn"]

    hybrid_gaps = [r["gap_pct"] for r in hybrid_results]
    pure_gaps = [r["gap_pct"] for r in pure_results]
    full_gaps = [r["gap_pct"] for r in full_results]

    hybrid_median = statistics.median(hybrid_gaps)
    pure_median = statistics.median(pure_gaps)
    full_median = statistics.median(full_gaps)

    hybrid_mean = statistics.mean(hybrid_gaps)
    pure_mean = statistics.mean(pure_gaps)
    full_mean = statistics.mean(full_gaps)

    # The kill criterion is about RELATIVE degradation: pure vs hybrid
    # Method: compare composed quality directly (not gaps, which confound
    # with each condition's own joint baseline)
    hybrid_comp_vals = [r["composed_avg"] for r in hybrid_results]
    pure_comp_vals = [r["composed_avg"] for r in pure_results]

    hybrid_comp_mean = statistics.mean(hybrid_comp_vals)
    pure_comp_mean = statistics.mean(pure_comp_vals)

    # Degradation = (pure - hybrid) / hybrid * 100
    # If pure loss is HIGHER, that's degradation (positive %)
    degradation_pct = (pure_comp_mean - hybrid_comp_mean) / hybrid_comp_mean * 100

    print(f"\n  Condition       | Gap mean  | Gap median | Composed mean")
    print(f"  ----------------+-----------+------------+--------------")
    print(f"  full_attn       | {full_mean:+7.2f}%  | {full_median:+7.2f}%   | {statistics.mean([r['composed_avg'] for r in full_results]):.4f}")
    print(f"  hybrid_3_1      | {hybrid_mean:+7.2f}%  | {hybrid_median:+7.2f}%   | {hybrid_comp_mean:.4f}")
    print(f"  pure_linear     | {pure_mean:+7.2f}%  | {pure_median:+7.2f}%   | {pure_comp_mean:.4f}")
    print(f"\n  Degradation (pure vs hybrid composed loss): {degradation_pct:+.2f}%")

    # Also check median gap difference
    gap_diff_median = pure_median - hybrid_median
    gap_diff_mean = pure_mean - hybrid_mean
    print(f"  Gap difference (pure - hybrid), mean:   {gap_diff_mean:+.2f}pp")
    print(f"  Gap difference (pure - hybrid), median: {gap_diff_median:+.2f}pp")

    killed = degradation_pct > 5.0
    if killed:
        print(f"\n  ** KILL: Pure-linear composed loss is {degradation_pct:+.2f}% worse than hybrid **")
        print(f"  ** Threshold: >5%, actual: {degradation_pct:+.2f}% **")
        print(f"  Conclusion: Linear attention NEEDS full attention scaffolding for composition.")
    else:
        print(f"\n  PASS: Pure-linear composed loss is within 5% of hybrid ({degradation_pct:+.2f}%)")
        print(f"  Conclusion: Linear attention does NOT need full attention scaffolding.")

    # =========================================================================
    # Supplementary: absolute quality comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUPPLEMENTARY: Absolute quality (joint training)")
    print(f"{'='*70}")

    for cond_name, results in all_results.items():
        joints = [r["joint_avg"] for r in results]
        print(f"  {cond_name:16s}: joint mean={statistics.mean(joints):.4f}  "
              f"std={statistics.stdev(joints):.4f}")

    hybrid_joint_mean = statistics.mean([r["joint_avg"] for r in hybrid_results])
    pure_joint_mean = statistics.mean([r["joint_avg"] for r in pure_results])
    quality_diff = (pure_joint_mean - hybrid_joint_mean) / hybrid_joint_mean * 100
    print(f"\n  Pure vs hybrid JOINT quality: {quality_diff:+.2f}%")
    print(f"  (Positive = pure-linear has higher loss = worse quality)")

    # =========================================================================
    # Per-layer interference
    # =========================================================================
    print(f"\n{'='*70}")
    print("Per-layer interference (mean across seeds)")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} {'Full Attn':>10} {'Hybrid 3:1':>12} {'Pure Linear':>12}")
    print(f"  " + "-" * 44)
    for l_idx in range(BASE["n_layer"]):
        full_interf = statistics.mean([r["interference"][l_idx] for r in full_results])
        hybrid_interf = statistics.mean([r["interference"][l_idx] for r in hybrid_results])
        pure_interf = statistics.mean([r["interference"][l_idx] for r in pure_results])
        print(f"  {l_idx:>6} {full_interf:>10.4f} {hybrid_interf:>12.4f} {pure_interf:>12.4f}")

    # =========================================================================
    # Per-seed details
    # =========================================================================
    print(f"\n{'='*70}")
    print("Per-seed composition gaps")
    print(f"{'='*70}")
    n_show = min(len(full_results), len(hybrid_results), len(pure_results))
    print(f"  {'Idx':>6} {'Full Attn':>10} {'Hybrid 3:1':>12} {'Pure Linear':>12}")
    print(f"  " + "-" * 44)
    for i in range(n_show):
        f_gap = full_results[i]["gap_pct"]
        h_gap = hybrid_results[i]["gap_pct"]
        p_gap = pure_results[i]["gap_pct"]
        print(f"  {i:>6} {f_gap:>+9.2f}% {h_gap:>+11.2f}% {p_gap:>+11.2f}%")

    # =========================================================================
    # Catastrophic failure check
    # =========================================================================
    print(f"\n{'='*70}")
    print("Catastrophic failure check (gap > 20%)")
    print(f"{'='*70}")
    for cond_name, results in all_results.items():
        failures = [(seeds[i], r["gap_pct"]) for i, r in enumerate(results) if r["gap_pct"] > 20.0]
        if failures:
            print(f"  {cond_name}: {len(failures)}/{n_seeds} failures: {failures}")
        else:
            print(f"  {cond_name}: 0/{n_seeds} catastrophic failures")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "n_seeds": n_seeds,
        "elapsed_seconds": elapsed,
        "conditions": {},
        "kill_criterion": {
            "description": "pure-linear composition degrades >5% vs hybrid 3:1",
            "degradation_pct": degradation_pct,
            "threshold": 5.0,
            "killed": killed,
            "hybrid_comp_mean": hybrid_comp_mean,
            "pure_comp_mean": pure_comp_mean,
            "gap_diff_mean": gap_diff_mean,
            "gap_diff_median": gap_diff_median,
        },
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
                    "idx": i,
                    "gap_pct": r["gap_pct"],
                    "joint_avg": r["joint_avg"],
                    "composed_avg": r["composed_avg"],
                    "single_avg": r["single_avg"],
                    "interference": {str(k): v for k, v in r["interference"].items()},
                }
                for i, r in enumerate(results)
            ],
        }

    results_path = "experiments/models/pure_linear_composition/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=7)
