"""Experiment: Minimal Gate Recalibration for Subtree Grafting.

Tests which gates need recalibration after grafting domain subtrees.
Compares three recalibration strategies at MATCHED step budgets:

    (a) root-only: gate 0 only (1/7 gates, 260 params)
    (b) root+graft-point: gates 0, 1, 2 (3/7 gates, 780 params)
    (c) all-gates: gates 0-6 (7/7 gates, 1820 params) -- baseline

All conditions use the same grafted model (same subtree fine-tuning),
differing ONLY in which gates are unfrozen during calibration.

Kill criteria:
1. root-only recalibration >3% worse than all-gates recalibration
2. selective (root+graft-point) >1.5% worse than all-gates recalibration

Protocol:
1. Pretrain base on all data (300 steps)
2. Fine-tune subtrees per domain (200 steps, only assigned subtree trainable)
3. Graft subtrees into one model
4. For each recalibration condition:
   - Start from the SAME grafted model state
   - Freeze everything except the specified gates
   - Calibrate for 100 steps on mixed data
   - Evaluate on joint + per-domain val sets

Additionally, we run weight averaging + all-gates calibration as the
reference point (from the parent experiment).
"""

import sys
import time
import random
import copy

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model
from experiments.models.subtree_grafting.run_experiment import (
    get_subtree_params, set_param_by_path, freeze_except_subtree,
    freeze_except_tree, get_tree_params,
)


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    t0 = time.time()
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == steps:
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {time.time()-t0:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"val_loss": val_loss, "elapsed_s": time.time() - t0}


# ── Selective freezing functions ──────────────────────────────────────────────

def freeze_all(model):
    """Freeze the entire model (base for selective unfreezing)."""
    model.freeze()


def unfreeze_gates(model, gate_indices):
    """Unfreeze specific gate indices across all layers.

    After calling freeze_all(), call this to selectively unfreeze gates.
    """
    for layer in model.layers:
        for gi in gate_indices:
            layer.tree.gates[gi].unfreeze()


def freeze_except_gates(model, gate_indices):
    """Freeze everything except the specified gate indices."""
    freeze_all(model)
    unfreeze_gates(model, gate_indices)


def freeze_except_all_gates(model):
    """Freeze everything except ALL tree gates (7 gates)."""
    freeze_all(model)
    for layer in model.layers:
        for gate in layer.tree.gates:
            gate.unfreeze()


# ── Recalibration conditions ─────────────────────────────────────────────────

RECAL_CONDITIONS = {
    "root_only": {
        "description": "Gate 0 only (root = domain router)",
        "gate_indices": [0],
        "n_gates": 1,
    },
    "root_graft_point": {
        "description": "Gates 0, 1, 2 (root + top of each subtree)",
        "gate_indices": [0, 1, 2],
        "n_gates": 3,
    },
    "all_gates": {
        "description": "Gates 0-6 (all internal gates)",
        "gate_indices": list(range(7)),
        "n_gates": 7,
    },
}


def run_experiment(seeds=(42, 123, 777), steps_pretrain=300,
                   steps_finetune=200, steps_calibrate=100):
    """Run the minimal graft recalibration experiment."""
    print("=" * 70)
    print("EXPERIMENT: Minimal Gate Recalibration for Subtree Grafting")
    print("=" * 70)
    print(f"  Pretrain: {steps_pretrain} steps")
    print(f"  Fine-tune: {steps_finetune} steps")
    print(f"  Calibrate: {steps_calibrate} steps (matched across all conditions)")
    print(f"  Seeds: {seeds}")
    print(f"  Conditions: {list(RECAL_CONDITIONS.keys())}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_names = list(splits.keys())

    # Result storage
    conditions = list(RECAL_CONDITIONS.keys())
    all_results = {"joint": [], "weight_avg": []}
    for cond in conditions:
        all_results[f"graft_{cond}"] = []
        all_results[f"graft_{cond}_domain_a"] = []
        all_results[f"graft_{cond}_domain_b"] = []
    all_results["joint_domain_a"] = []
    all_results["joint_domain_b"] = []
    all_results["weight_avg_domain_a"] = []
    all_results["weight_avg_domain_b"] = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # Prepare datasets
        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )
        all_train, all_val = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train, tokenizer, 32)
        joint_val = CharDataset(all_val, tokenizer, 32)

        tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                        n_capsules_per_leaf=32, beam_width=2)

        # ── Joint training baseline ───────────────────────────────────────
        print(f"\n  [JOINT] Full tree training on all data")
        mx.random.seed(seed)
        joint_model = get_model("hierarchical_tree", **tree_cfg)
        mx.eval(joint_model.parameters())
        joint_result = train_model(joint_model, joint_train, joint_val,
                                   steps=steps_pretrain + steps_finetune,
                                   seed=seed, log_every=200)
        all_results["joint"].append(joint_result["val_loss"])
        print(f"    joint val_loss: {joint_result['val_loss']:.4f}")

        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(joint_model, domain_datasets[d_name][1], 32)
            side = "a" if di == 0 else "b"
            all_results[f"joint_domain_{side}"].append(d_vl)
            print(f"    joint {d_name}: {d_vl:.4f}")

        # ── Pretrain shared base ──────────────────────────────────────────
        print(f"\n  [BASE] Pretrain shared base")
        mx.random.seed(seed)
        base_model = get_model("hierarchical_tree", **tree_cfg)
        mx.eval(base_model.parameters())
        _ = train_model(base_model, joint_train, joint_val,
                        steps=steps_pretrain, seed=seed, log_every=200)
        base_weights = {k: mx.array(v) for k, v in
                        nn.utils.tree_flatten(base_model.parameters())}

        # ── Weight averaging baseline ─────────────────────────────────────
        print(f"\n  [WEIGHT AVG] Fine-tune full tree per domain, then average")
        domain_tree_params = {}
        for d_name, (d_train, d_val) in domain_datasets.items():
            print(f"    fine-tune on {d_name}...")
            mx.random.seed(seed)
            d_model = get_model("hierarchical_tree", **tree_cfg)
            d_model.load_weights(list(base_weights.items()))
            mx.eval(d_model.parameters())
            freeze_except_tree(d_model)
            _ = train_model(d_model, d_train, d_val,
                            steps=steps_finetune, seed=seed, log_every=200)
            domain_tree_params[d_name] = get_tree_params(d_model)

        avg_model = get_model("hierarchical_tree", **tree_cfg)
        avg_model.load_weights(list(base_weights.items()))
        mx.eval(avg_model.parameters())
        for key in domain_tree_params[domain_names[0]]:
            avg_w = sum(domain_tree_params[d][key] for d in domain_names) / len(domain_names)
            set_param_by_path(avg_model, key, avg_w)
        mx.eval(avg_model.parameters())

        freeze_except_all_gates(avg_model)
        print(f"    calibrating all gates ({count_params(avg_model)} trainable)...")
        _ = train_model(avg_model, joint_train, joint_val,
                        steps=steps_calibrate, seed=seed, log_every=100)
        avg_vl = evaluate(avg_model, joint_val, 32)
        all_results["weight_avg"].append(avg_vl)
        print(f"    weight_avg val_loss: {avg_vl:.4f}")

        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(avg_model, domain_datasets[d_name][1], 32)
            side = "a" if di == 0 else "b"
            all_results[f"weight_avg_domain_{side}"].append(d_vl)
            print(f"    weight_avg {d_name}: {d_vl:.4f}")

        # ── Subtree grafting (shared fine-tuning) ─────────────────────────
        print(f"\n  [GRAFT] Fine-tune subtrees per domain")
        graft_subtree_params = {}
        for di, (d_name, (d_train, d_val)) in enumerate(domain_datasets.items()):
            side = "left" if di == 0 else "right"
            print(f"    fine-tune {side} subtree on {d_name}...")
            mx.random.seed(seed)
            d_model = get_model("hierarchical_tree", **tree_cfg)
            d_model.load_weights(list(base_weights.items()))
            mx.eval(d_model.parameters())
            freeze_except_subtree(d_model, side=side)
            _ = train_model(d_model, d_train, d_val,
                            steps=steps_finetune, seed=seed, log_every=200)
            graft_subtree_params[d_name] = get_subtree_params(d_model, side=side)

        # Build the grafted model (shared starting point for all conditions)
        graft_base = get_model("hierarchical_tree", **tree_cfg)
        graft_base.load_weights(list(base_weights.items()))
        mx.eval(graft_base.parameters())
        for d_name, params in graft_subtree_params.items():
            for key, val in params.items():
                set_param_by_path(graft_base, key, val)
        mx.eval(graft_base.parameters())

        # Save grafted weights for reuse across conditions
        graft_weights = {k: mx.array(v) for k, v in
                         nn.utils.tree_flatten(graft_base.parameters())}

        # ── Test each recalibration condition ─────────────────────────────
        for cond_name, cond_cfg in RECAL_CONDITIONS.items():
            print(f"\n  [GRAFT {cond_name.upper()}] {cond_cfg['description']}")

            # Fresh model from grafted weights (identical starting point)
            cond_model = get_model("hierarchical_tree", **tree_cfg)
            cond_model.load_weights(list(graft_weights.items()))
            mx.eval(cond_model.parameters())

            # Selective freezing
            freeze_except_gates(cond_model, cond_cfg["gate_indices"])
            n_trainable = count_params(cond_model)
            print(f"    trainable params: {n_trainable} "
                  f"({cond_cfg['n_gates']}/7 gates)")

            # Calibrate
            _ = train_model(cond_model, joint_train, joint_val,
                            steps=steps_calibrate, seed=seed, log_every=100)
            cond_vl = evaluate(cond_model, joint_val, 32)
            all_results[f"graft_{cond_name}"].append(cond_vl)
            print(f"    {cond_name} val_loss: {cond_vl:.4f}")

            # Per-domain evaluation
            for di, d_name in enumerate(domain_names):
                d_vl = evaluate(cond_model, domain_datasets[d_name][1], 32)
                side = "a" if di == 0 else "b"
                all_results[f"graft_{cond_name}_domain_{side}"].append(d_vl)
                print(f"    {cond_name} {d_name}: {d_vl:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")

    def mean(lst):
        return sum(lst) / len(lst)

    joint_mean = mean(all_results["joint"])
    avg_mean = mean(all_results["weight_avg"])

    print(f"\n  Joint (baseline):     mean={joint_mean:.4f}  "
          f"per-seed: {['%.4f' % v for v in all_results['joint']]}")
    print(f"  Weight averaging:     mean={avg_mean:.4f}  "
          f"per-seed: {['%.4f' % v for v in all_results['weight_avg']]}")

    # Recalibration results
    cond_means = {}
    for cond_name in conditions:
        cond_mean = mean(all_results[f"graft_{cond_name}"])
        cond_means[cond_name] = cond_mean
        gap_vs_joint = 100 * (cond_mean - joint_mean) / joint_mean
        gap_vs_avg = 100 * (cond_mean - avg_mean) / avg_mean
        print(f"  Graft {cond_name:20s}: mean={cond_mean:.4f}  "
              f"per-seed: {['%.4f' % v for v in all_results[f'graft_{cond_name}']]}  "
              f"vs joint: {gap_vs_joint:+.2f}%  vs wt_avg: {gap_vs_avg:+.2f}%")

    # ── Kill criteria assessment ──────────────────────────────────────────
    all_gates_mean = cond_means["all_gates"]

    print(f"\n  --- Kill Criteria Assessment ---")
    print(f"  Reference: all-gates recalibration = {all_gates_mean:.4f}")

    # Kill 1: root-only >3% worse than all-gates
    root_only_mean = cond_means["root_only"]
    root_gap = 100 * (root_only_mean - all_gates_mean) / all_gates_mean
    kill_1 = root_gap > 3.0
    print(f"\n  Kill 1: root-only vs all-gates: {root_gap:+.2f}%  "
          f"(threshold: 3%)  {'TRIGGERED' if kill_1 else 'PASSES'}")

    # Kill 2: root+graft-point >1.5% worse than all-gates
    rgp_mean = cond_means["root_graft_point"]
    rgp_gap = 100 * (rgp_mean - all_gates_mean) / all_gates_mean
    kill_2 = rgp_gap > 1.5
    print(f"  Kill 2: root+graft-point vs all-gates: {rgp_gap:+.2f}%  "
          f"(threshold: 1.5%)  {'TRIGGERED' if kill_2 else 'PASSES'}")

    # Per-domain analysis
    print(f"\n  --- Per-Domain Analysis ---")
    for cond_name in conditions:
        print(f"\n  {cond_name}:")
        for di, d_name in enumerate(domain_names):
            side = "a" if di == 0 else "b"
            joint_d = mean(all_results[f"joint_domain_{side}"])
            cond_d = mean(all_results[f"graft_{cond_name}_domain_{side}"])
            deg = 100 * (cond_d - joint_d) / joint_d
            print(f"    {d_name}: joint={joint_d:.4f}  "
                  f"graft_{cond_name}={cond_d:.4f}  ({deg:+.2f}%)")

    # Cost analysis
    print(f"\n  --- Cost Analysis ---")
    d_plus_1 = 65  # d=64 embedding dim + 1 bias
    n_layers = 4
    for cond_name, cond_cfg in RECAL_CONDITIONS.items():
        n_params = cond_cfg["n_gates"] * d_plus_1 * n_layers
        speedup = 7.0 / cond_cfg["n_gates"]
        print(f"  {cond_name:20s}: {cond_cfg['n_gates']}/7 gates, "
              f"{n_params:5d} cal params, {speedup:.1f}x fewer gate params")

    # Overall verdict
    killed = kill_1 or kill_2
    print(f"\n  OVERALL: {'KILLED' if killed else 'PASSES'}")
    if not kill_1 and not kill_2:
        print(f"  Root+graft-point recalibration is SUFFICIENT "
              f"({rgp_gap:+.2f}% vs all-gates, within 1.5% threshold)")
    elif kill_1 and not kill_2:
        print(f"  Root-only is insufficient (as expected from parent experiment)")
        print(f"  Root+graft-point is sufficient ({rgp_gap:+.2f}%)")
    elif kill_2:
        print(f"  Even root+graft-point is insufficient -- all gates needed")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment()
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
