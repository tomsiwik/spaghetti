"""Pure-linear state capacity scaling experiment.

Tests whether the composition gap grows >3x when scaling d_h from 16 to 32 or 64.

Hypothesis (to falsify): composition gap grows >3x when scaling d_h from 64 to
128 or 256 (i.e., d_h from 16 to 32 or 64), indicating state capacity saturation.

Protocol: identical to pure_linear_composition experiment, but repeated at
three model dimensions:
  - d=64,  h=4, d_h=16   (baseline, already proven +1.02% gap)
  - d=128, h=4, d_h=32   (2x state capacity: 32x32 = 1024 elements per head)
  - d=256, h=4, d_h=64   (4x state capacity: 64x64 = 4096 elements per head)

At each dimension, we run pure-linear (4:0) vs full attention (4:0 full) to
compute the composition gap. The kill criterion is about gap GROWTH, not gap
magnitude: if gap(d=128)/gap(d=64) > 3 or gap(d=256)/gap(d=64) > 3, the
state capacity concern is confirmed.

We keep n_head=4 fixed to isolate d_h scaling from head-count effects.
Capsule groups and routing stay the same (G=4, P=64, k=2).
Training steps scale with model size to keep tokens-per-parameter constant:
  d=64:  base=300 steps, fine-tune=300, calibrate=100
  d=128: base=450 steps, fine-tune=450, calibrate=150  (1.5x)
  d=256: base=600 steps, fine-tune=600, calibrate=200  (2x)

5 seeds per condition (3 dimensions x 2 attention types = 6 conditions, 30 runs).
"""

import random
import statistics
import time
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


# ─── Dimension configurations ───────────────────────────────────────────────
CONFIGS = {
    "d64": {
        "n_embd": 64, "n_head": 4,   # d_h=16, state=16x16=256
        "base_steps": 300, "ft_steps": 300, "cal_steps": 100,
    },
    "d128": {
        "n_embd": 128, "n_head": 4,  # d_h=32, state=32x32=1024
        "base_steps": 450, "ft_steps": 450, "cal_steps": 150,
    },
    "d256": {
        "n_embd": 256, "n_head": 4,  # d_h=64, state=64x64=4096
        "base_steps": 600, "ft_steps": 600, "cal_steps": 200,
    },
}

N_LAYER = 4
BLOCK_SIZE = 32
N_GROUPS = 4
N_CAPSULES_PER_GROUP = 64
TOP_K = 2
BATCH_SIZE = 32
LR = 3e-3


def compose_models(model_a, model_b, model_name, vocab_size, n_embd, n_head,
                   base_model=None, layer_types=None):
    """Create a composed model by concatenating capsule groups from A and B."""
    n_groups_a = model_a.layers[0].capsule_pool.n_groups
    n_groups_b = model_b.layers[0].capsule_pool.n_groups
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = model_a.layers[0].capsule_pool.top_k_groups + \
                     model_b.layers[0].capsule_pool.top_k_groups

    kwargs = dict(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_groups=composed_groups,
        n_capsules_per_group=N_CAPSULES_PER_GROUP,
        top_k_groups=composed_top_k,
        n_embd=n_embd, n_head=n_head, n_layer=N_LAYER,
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
        for name in ["wq", "wk", "wv", "wo", "wg", "w_a", "w_beta", "w_z"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                src_param = getattr(layer_src.attn, name).weight
                getattr(layer_c.attn, name).weight = src_param

        for name in ["dt_bias", "A_log"]:
            if hasattr(layer_src.attn, name) and hasattr(layer_c.attn, name):
                setattr(layer_c.attn, name, getattr(layer_src.attn, name))

        for conv_name in ["conv_q", "conv_k", "conv_v"]:
            if hasattr(layer_src.attn, conv_name) and hasattr(layer_c.attn, conv_name):
                src_conv = getattr(layer_src.attn, conv_name)
                dst_conv = getattr(layer_c.attn, conv_name)
                dst_conv.weight = src_conv.weight

        for norm_name in ["norm1", "norm2"]:
            src_norm = getattr(layer_src, norm_name)
            dst_norm = getattr(layer_c, norm_name)
            if hasattr(src_norm, "weight") and hasattr(dst_norm, "weight"):
                dst_norm.weight = src_norm.weight

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


def run_single_seed(dim_cfg, attn_type, docs, tokenizer, seed):
    """Run the full composition protocol for one seed and config."""
    V = tokenizer.vocab_size
    n_embd = dim_cfg["n_embd"]
    n_head = dim_cfg["n_head"]
    base_steps = dim_cfg["base_steps"]
    ft_steps = dim_cfg["ft_steps"]
    cal_steps = dim_cfg["cal_steps"]

    splits = domain_split(docs)
    domain_datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        domain_datasets[name] = (
            CharDataset(dtrain, tokenizer, BLOCK_SIZE),
            CharDataset(dval, tokenizer, BLOCK_SIZE),
        )
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BLOCK_SIZE)

    if attn_type == "pure_linear":
        model_name = "full_gdn_stack_capsule_moe"
        layer_types = ["linear"] * N_LAYER
    else:
        model_name = "full_attn_capsule_moe"
        layer_types = ["full"] * N_LAYER

    kwargs = dict(
        vocab_size=V, block_size=BLOCK_SIZE,
        n_embd=n_embd, n_head=n_head, n_layer=N_LAYER,
        n_groups=N_GROUPS, n_capsules_per_group=N_CAPSULES_PER_GROUP,
        top_k_groups=TOP_K,
    )
    if attn_type == "pure_linear":
        kwargs["layer_types"] = layer_types

    # --- Joint training baseline ---
    model_joint = get_model(model_name, **kwargs)
    mx.eval(model_joint.parameters())

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = 2 * ft_steps
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
    train(base_model, joint_train, steps=base_steps,
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
    train(model_a, train_a, val_a, steps=ft_steps,
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
    train(model_b, train_b, val_b, steps=ft_steps,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_b.unfreeze()

    # --- Compose + calibrate ---
    composed = compose_models(
        model_a, model_b, model_name, V, n_embd, n_head,
        base_model=base_model, layer_types=layer_types if attn_type == "pure_linear" else None,
    )
    calibrate_router(composed, train_a, train_b,
                     steps=cal_steps, lr=LR, seed=seed)

    comp_val_a = evaluate(composed, val_a, BATCH_SIZE)
    comp_val_b = evaluate(composed, val_b, BATCH_SIZE)
    comp_avg = (comp_val_a + comp_val_b) / 2

    gap_pct = (comp_avg - joint_avg) / joint_avg * 100

    spec_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_b = evaluate(model_b, val_b, BATCH_SIZE)
    single_avg = (spec_a + spec_b) / 2

    d_h = n_embd // n_head
    state_size = d_h * d_h

    return {
        "joint_avg": joint_avg,
        "composed_avg": comp_avg,
        "single_avg": single_avg,
        "gap_pct": gap_pct,
        "d_h": d_h,
        "state_size": state_size,
    }


def run_experiment(n_seeds=5):
    """Run linear state capacity scaling experiment."""
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)

    seeds = list(range(n_seeds))

    all_results = {}

    for dim_name, dim_cfg in CONFIGS.items():
        for attn_type in ["pure_linear", "full_attn"]:
            cond_name = f"{dim_name}_{attn_type}"
            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name} (d={dim_cfg['n_embd']}, d_h={dim_cfg['n_embd']//dim_cfg['n_head']}, {attn_type})")
            print(f"{'='*70}")

            cond_results = []
            for i, seed in enumerate(seeds):
                print(f"  Seed {seed} ({i+1}/{n_seeds})...", end=" ", flush=True)
                try:
                    result = run_single_seed(dim_cfg, attn_type, docs, tokenizer, seed)
                    print(f"gap={result['gap_pct']:+.2f}%  "
                          f"joint={result['joint_avg']:.4f}  "
                          f"comp={result['composed_avg']:.4f}  "
                          f"state={result['state_size']}")
                    cond_results.append(result)
                except RuntimeError as e:
                    print(f"FAILED (RuntimeError: {e})")
                    continue

            if len(cond_results) < 3:
                print(f"  WARNING: Only {len(cond_results)} seeds succeeded")
            all_results[cond_name] = cond_results

    elapsed = time.time() - t0

    # =========================================================================
    # Analysis
    # =========================================================================
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_seeds} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")

    # Compute per-dimension composition gaps for pure-linear
    dim_gaps = {}
    for dim_name in CONFIGS:
        pl_key = f"{dim_name}_pure_linear"
        fa_key = f"{dim_name}_full_attn"

        if pl_key in all_results and fa_key in all_results:
            pl_results = all_results[pl_key]
            fa_results = all_results[fa_key]

            pl_gaps = [r["gap_pct"] for r in pl_results]
            fa_gaps = [r["gap_pct"] for r in fa_results]

            pl_comp = [r["composed_avg"] for r in pl_results]
            fa_comp = [r["composed_avg"] for r in fa_results]
            pl_joint = [r["joint_avg"] for r in pl_results]
            fa_joint = [r["joint_avg"] for r in fa_results]

            d_h = CONFIGS[dim_name]["n_embd"] // CONFIGS[dim_name]["n_head"]

            dim_gaps[dim_name] = {
                "d_h": d_h,
                "state_size": d_h * d_h,
                "pl_gap_mean": statistics.mean(pl_gaps),
                "pl_gap_median": statistics.median(pl_gaps),
                "pl_gap_std": statistics.stdev(pl_gaps) if len(pl_gaps) > 1 else 0,
                "fa_gap_mean": statistics.mean(fa_gaps),
                "fa_gap_median": statistics.median(fa_gaps),
                "fa_gap_std": statistics.stdev(fa_gaps) if len(fa_gaps) > 1 else 0,
                "pl_comp_mean": statistics.mean(pl_comp),
                "fa_comp_mean": statistics.mean(fa_comp),
                "pl_joint_mean": statistics.mean(pl_joint),
                "fa_joint_mean": statistics.mean(fa_joint),
                "pl_gaps": pl_gaps,
                "fa_gaps": fa_gaps,
            }

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("Per-dimension summary")
    print(f"{'='*70}")
    print(f"  {'Dim':>6} {'d_h':>4} {'State':>6} | {'PL gap mean':>11} {'PL gap med':>11} {'PL gap std':>11} | {'FA gap mean':>11}")
    print(f"  " + "-" * 80)
    for dim_name, dg in dim_gaps.items():
        print(f"  {dim_name:>6} {dg['d_h']:>4} {dg['state_size']:>6} | "
              f"{dg['pl_gap_mean']:>+10.2f}% {dg['pl_gap_median']:>+10.2f}% {dg['pl_gap_std']:>10.2f}% | "
              f"{dg['fa_gap_mean']:>+10.2f}%")

    # --- Absolute quality ---
    print(f"\n{'='*70}")
    print("Absolute quality (joint & composed means)")
    print(f"{'='*70}")
    print(f"  {'Dim':>6} | {'PL joint':>10} {'PL comp':>10} {'PL gap':>10} | {'FA joint':>10} {'FA comp':>10} {'FA gap':>10}")
    print(f"  " + "-" * 80)
    for dim_name, dg in dim_gaps.items():
        print(f"  {dim_name:>6} | "
              f"{dg['pl_joint_mean']:>10.4f} {dg['pl_comp_mean']:>10.4f} {dg['pl_gap_mean']:>+9.2f}% | "
              f"{dg['fa_joint_mean']:>10.4f} {dg['fa_comp_mean']:>10.4f} {dg['fa_gap_mean']:>+9.2f}%")

    # =========================================================================
    # KILL CRITERION: composition gap grows >3x
    # =========================================================================
    print(f"\n{'='*70}")
    print("KILL CRITERION: Composition gap grows >3x when scaling d_h")
    print(f"{'='*70}")

    baseline_dim = "d64"
    if baseline_dim in dim_gaps:
        baseline_gap = abs(dim_gaps[baseline_dim]["pl_gap_mean"])
        # Use a floor to avoid division by near-zero
        baseline_gap_floor = max(baseline_gap, 0.5)  # at least 0.5% reference

        print(f"\n  Baseline (d=64, d_h=16): pure-linear gap = {dim_gaps[baseline_dim]['pl_gap_mean']:+.2f}%")
        print(f"  Using floor of {baseline_gap_floor:.2f}% for ratio computation")

        killed = False
        for dim_name, dg in dim_gaps.items():
            if dim_name == baseline_dim:
                continue
            this_gap = abs(dg["pl_gap_mean"])
            ratio = this_gap / baseline_gap_floor
            kill = ratio > 3.0

            print(f"\n  {dim_name} (d_h={dg['d_h']}): pure-linear gap = {dg['pl_gap_mean']:+.2f}%")
            print(f"    Ratio vs baseline: {ratio:.2f}x (threshold: >3x)")
            if kill:
                print(f"    ** KILL: {ratio:.2f}x > 3x **")
                killed = True
            else:
                print(f"    PASS: {ratio:.2f}x <= 3x")

        # Also compute per-seed paired gap ratios for robustness
        print(f"\n{'='*70}")
        print("Per-seed gap details")
        print(f"{'='*70}")
        for dim_name in CONFIGS:
            pl_key = f"{dim_name}_pure_linear"
            if pl_key in all_results:
                results = all_results[pl_key]
                gaps = [r["gap_pct"] for r in results]
                print(f"\n  {dim_name} pure-linear: {[f'{g:+.2f}%' for g in gaps]}")

        # Also: check whether full_attn gap also grows (to distinguish
        # state capacity from general scaling effects)
        print(f"\n{'='*70}")
        print("Control: does full_attn gap also grow with dimension?")
        print(f"{'='*70}")
        if baseline_dim in dim_gaps:
            fa_baseline_gap = abs(dim_gaps[baseline_dim]["fa_gap_mean"])
            fa_baseline_floor = max(fa_baseline_gap, 0.5)
            print(f"  Baseline full_attn gap: {dim_gaps[baseline_dim]['fa_gap_mean']:+.2f}%")
            for dim_name, dg in dim_gaps.items():
                if dim_name == baseline_dim:
                    continue
                fa_this_gap = abs(dg["fa_gap_mean"])
                fa_ratio = fa_this_gap / fa_baseline_floor
                print(f"  {dim_name} full_attn: gap = {dg['fa_gap_mean']:+.2f}%, ratio = {fa_ratio:.2f}x")

        # Key comparison: is the gap growth SPECIFIC to linear attention?
        print(f"\n{'='*70}")
        print("LINEAR-SPECIFIC gap growth (pure_linear gap - full_attn gap)")
        print(f"{'='*70}")
        for dim_name, dg in dim_gaps.items():
            diff = dg["pl_gap_mean"] - dg["fa_gap_mean"]
            print(f"  {dim_name}: PL-FA diff = {diff:+.2f}pp")

        # Final verdict
        print(f"\n{'='*70}")
        if killed:
            print("VERDICT: KILL -- composition gap grows >3x with d_h scaling")
            print("State capacity saturation is a real concern for pure-linear at scale.")
        else:
            print("VERDICT: PASS -- composition gap does NOT grow >3x with d_h scaling")
            print("State capacity is not binding at d_h up to 64.")
        print(f"{'='*70}")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "n_seeds": n_seeds,
        "elapsed_seconds": elapsed,
        "configs": {k: {**v, "d_h": v["n_embd"] // v["n_head"]} for k, v in CONFIGS.items()},
        "dim_gaps": {},
        "per_condition": {},
    }
    for dim_name, dg in dim_gaps.items():
        output["dim_gaps"][dim_name] = {
            k: v for k, v in dg.items()
            if k not in ("pl_gaps", "fa_gaps")
        }
        output["dim_gaps"][dim_name]["pl_gaps"] = dg["pl_gaps"]
        output["dim_gaps"][dim_name]["fa_gaps"] = dg["fa_gaps"]

    for cond_name, results in all_results.items():
        output["per_condition"][cond_name] = [
            {
                "gap_pct": r["gap_pct"],
                "joint_avg": r["joint_avg"],
                "composed_avg": r["composed_avg"],
                "single_avg": r["single_avg"],
                "d_h": r["d_h"],
                "state_size": r["state_size"],
            }
            for r in results
        ]

    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment(n_seeds=5)
