"""Experiment: Persistent Expert Tree — version-aware cross-version composition.

REVISED: Fixes 6 issues from adversarial review.

Tests two kill criteria:
KC1: cross-version composition >5% worse than same-version composition
KC2: persistent structure overhead >15% memory vs mutable (leaf-only fine-tuning)

Key changes from v1:
- Fine-tunes ONLY tree leaves (freezes embeddings, attention, norms, lm_head)
- Uses the persistent tree API (update_leaves/compose_versions/set_active_version)
- Reports memory_report() from the tree API (measures structural sharing)
- Adds flat-dict baseline (naive full-copy checkpointing)
- Adds same-version cherry-pick control (isolates version-crossing factor)

Protocol:
1. Pretrain base model on all data (v0) -- 500 steps, all params trainable
2. Freeze non-tree params. Create version v1 by updating leaves 0-3, fine-tune
   on domain A -- 200 steps (only leaves trainable)
3. Restore to v0, create version v2 by updating leaves 4-7, fine-tune
   on domain B -- 200 steps (only leaves trainable)
4. Baselines:
   a. Same-version weight-avg: average v1 and v2 leaf params
   b. Same-version cherry-pick: cherry-pick A-leaves from v1, B-leaves from v1
      (same version, no cross-version mixing)
   c. Flat-dict: store full parameter snapshots in a plain dict
5. Cross-version: compose_versions({0-3: v1, 4-7: v2}) via tree API
6. Calibrate all composed versions -- 100 steps (gates only)
7. Compare quality and measure memory overhead via memory_report()
"""

import sys
import time
import random
import json
import os

os.environ["PYTHONUNBUFFERED"] = "1"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def freeze_non_tree_leaves(model):
    """Freeze everything except tree leaves."""
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        # Freeze gates too -- only leaves should be trainable during fine-tuning
        for gate in layer.tree.gates:
            gate.freeze()


def freeze_all_except_target_leaves(model, target_leaf_indices):
    """Freeze everything except the specified tree leaves.

    This is critical for persistent versioning: when v1 copies leaves 0-3,
    leaves 4-7 are shared references with v0. Training must NOT update
    shared leaves, or it would corrupt v0's data.
    """
    freeze_non_tree_leaves(model)
    for layer in model.layers:
        for i, leaf in enumerate(layer.tree.leaves):
            if i in target_leaf_indices:
                leaf.unfreeze()
            else:
                leaf.freeze()


def unfreeze_gates_only(model):
    """Unfreeze only gates for calibration. Leaves and everything else stay frozen."""
    for layer in model.layers:
        for gate in layer.tree.gates:
            gate.unfreeze()


def unfreeze_all_tree(model):
    """Unfreeze all tree params (gates + leaves) for calibration."""
    for layer in model.layers:
        for gate in layer.tree.gates:
            gate.unfreeze()
        for leaf in layer.tree.leaves:
            leaf.unfreeze()


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    """Train and return metrics."""
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
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"val_loss": val_loss, "elapsed_s": time.time() - t0}


def flat_dict_snapshot(model):
    """Naive flat-dict snapshot: store ALL tree parameters as tensors."""
    params = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        if "tree" in k:
            params[k] = mx.array(v)
    return params


def flat_dict_restore(model, params):
    """Restore tree parameters from a flat dict snapshot."""
    model.update(nn.utils.tree_unflatten(list(params.items())))
    mx.eval(model.parameters())


def flat_dict_memory(snapshots):
    """Compute memory for flat-dict approach (no structural sharing)."""
    total_params = 0
    for snap in snapshots.values():
        total_params += sum(v.size for v in snap.values())
    return total_params


def run_experiment(seed=42):
    """Run the full persistent expert tree experiment."""
    print(f"\n{'='*60}")
    print(f"  Persistent Expert Tree Experiment v2 (seed={seed})")
    print(f"{'='*60}")

    # ── Data setup ──────────────────────────────────────────────────
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    domains = domain_split(docs)
    domain_names = sorted(domains.keys())[:2]
    print(f"\nDomains: {domain_names}")

    all_train, all_val = [], []
    domain_datasets = {}
    for name in domain_names:
        tr, va = train_val_split(domains[name], seed=seed)
        all_train.extend(tr)
        all_val.extend(va)
        domain_datasets[name] = {
            "train": CharDataset(tr, tokenizer, block_size=32),
            "val": CharDataset(va, tokenizer, block_size=32),
        }
    joint_train = CharDataset(all_train, tokenizer, block_size=32)
    joint_val = CharDataset(all_val, tokenizer, block_size=32)

    results = {"seed": seed, "domains": domain_names}

    # ── 1. Joint training (upper bound) ──────────────────────────────
    print("\n--- Phase 1: Joint training (upper bound, 500 steps) ---")
    joint_model = get_model("persistent_expert_tree",
                            vocab_size=tokenizer.vocab_size)
    joint_metrics = train_model(joint_model, joint_train, joint_val,
                                steps=500, seed=seed)
    results["joint"] = joint_metrics
    print(f"  Joint val_loss: {joint_metrics['val_loss']:.4f}")

    # ── 2. Pretrain base model (v0, all params trainable) ─────────────
    print("\n--- Phase 2: Pretrain base model (v0, 300 steps, full model) ---")
    model = get_model("persistent_expert_tree",
                      vocab_size=tokenizer.vocab_size)
    pretrain_metrics = train_model(model, joint_train, joint_val,
                                   steps=300, seed=seed)
    results["pretrain"] = pretrain_metrics
    print(f"  Pretrain val_loss: {pretrain_metrics['val_loss']:.4f}")

    # Save v0 output for rollback test
    v0_output = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(v0_output)

    # Store flat-dict snapshot of v0 for baseline comparison
    flat_v0 = flat_dict_snapshot(model)

    # Count trainable leaf params vs total
    total_tree_params = sum(v.size for k, v in nn.utils.tree_flatten(model.parameters())
                            if "tree" in k)
    leaf_params_only = sum(v.size for k, v in nn.utils.tree_flatten(model.parameters())
                           if "tree.leaves" in k)
    gate_params_only = sum(v.size for k, v in nn.utils.tree_flatten(model.parameters())
                           if "tree.gates" in k)
    total_all_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))

    results["param_breakdown"] = {
        "total_all": total_all_params,
        "total_tree": total_tree_params,
        "leaf_params": leaf_params_only,
        "gate_params": gate_params_only,
        "leaf_pct_of_tree": leaf_params_only / total_tree_params * 100,
        "tree_pct_of_total": total_tree_params / total_all_params * 100,
    }
    print(f"  Total params: {total_all_params:,}")
    print(f"  Tree params:  {total_tree_params:,} ({total_tree_params/total_all_params*100:.1f}%)")
    print(f"  Leaf params:  {leaf_params_only:,} ({leaf_params_only/total_tree_params*100:.1f}% of tree)")
    print(f"  Gate params:  {gate_params_only:,} ({gate_params_only/total_tree_params*100:.1f}% of tree)")

    # ── 3a. Fine-tune leaves 0-3 on domain A (v1) ─────────────────────
    # Use the tree API: create v1 by path-copying leaves 0-3
    print(f"\n--- Phase 3a: Fine-tune {domain_names[0]} leaves 0-3 (-> v1) ---")

    # Create v1 via tree API (path-copy leaves 0-3 from v0)
    v1_leaves = [0, 1, 2, 3]
    v1_id = model.update_leaves(v1_leaves,
                                 description=f"fine-tune {domain_names[0]}")
    model.set_version(v1_id)

    # Freeze everything except target leaves (0-3).
    # Leaves 4-7 are shared references with v0 -- must NOT be trained.
    freeze_all_except_target_leaves(model, set(v1_leaves))

    trainable_during_ft = count_params(model)
    print(f"  Trainable params during fine-tuning: {trainable_during_ft:,}")

    ft_a_metrics = train_model(model,
                               domain_datasets[domain_names[0]]["train"],
                               domain_datasets[domain_names[0]]["val"],
                               steps=200, seed=seed)
    results["finetune_A"] = ft_a_metrics
    print(f"  {domain_names[0]} val_loss: {ft_a_metrics['val_loss']:.4f}")

    # Store flat-dict snapshot of v1
    flat_v1 = flat_dict_snapshot(model)

    # ── 3b. Fine-tune leaves 4-7 on domain B (v2) ─────────────────────
    print(f"\n--- Phase 3b: Fine-tune {domain_names[1]} leaves 4-7 (-> v2) ---")

    # Switch back to v0 and create v2
    model.set_version(0)
    v2_leaves = [4, 5, 6, 7]
    v2_id = model.update_leaves(v2_leaves,
                                 description=f"fine-tune {domain_names[1]}")
    model.set_version(v2_id)

    # Freeze everything except target leaves (4-7).
    # Leaves 0-3 are shared references with v0 -- must NOT be trained.
    freeze_all_except_target_leaves(model, set(v2_leaves))

    ft_b_metrics = train_model(model,
                               domain_datasets[domain_names[1]]["train"],
                               domain_datasets[domain_names[1]]["val"],
                               steps=200, seed=seed)
    results["finetune_B"] = ft_b_metrics
    print(f"  {domain_names[1]} val_loss: {ft_b_metrics['val_loss']:.4f}")

    # Store flat-dict snapshot of v2
    flat_v2 = flat_dict_snapshot(model)

    # ── Rollback test (must be before flat_dict_restore mutates module params) ──
    print("\n--- Rollback test (set_version(0)) ---")
    model.set_version(0)
    v0_output_after = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(v0_output_after)
    rollback_diff = mx.abs(v0_output - v0_output_after).max().item()
    results["rollback_max_diff"] = rollback_diff
    print(f"  Rollback max diff: {rollback_diff:.2e} (should be ~0)")

    # ── 4a. Same-version weight averaging (v1+v2 leaves averaged) ─────
    print("\n--- Phase 4a: Same-version weight averaging (v1+v2 leaf avg) ---")
    # Average the leaf params from v1 and v2 snapshots
    avg_params = {}
    for k in flat_v1:
        if "tree.leaves" in k:
            avg_params[k] = (flat_v1[k] + flat_v2[k]) / 2.0
        else:
            avg_params[k] = flat_v0[k]  # gates from base
    model.set_version(0)  # reset to base structure
    flat_dict_restore(model, avg_params)

    # Unfreeze gates for calibration
    freeze_non_tree_leaves(model)
    unfreeze_gates_only(model)
    same_avg_raw = evaluate(model, joint_val, batch_size=32)
    print(f"  Same-version avg raw val_loss: {same_avg_raw:.4f}")

    print("  Calibrating (100 steps, gates only)...")
    same_avg_cal = train_model(model, joint_train, joint_val,
                               steps=100, seed=seed, lr=3e-3)
    results["same_version_avg"] = {
        "raw_val_loss": same_avg_raw,
        "calibrated_val_loss": same_avg_cal["val_loss"],
    }
    print(f"  Same-version avg calibrated: {same_avg_cal['val_loss']:.4f}")

    # ── 4b. Same-version cherry-pick CONTROL ──────────────────────────
    # Cherry-pick from SAME version (v1 only) to isolate version-crossing
    print("\n--- Phase 4b: Same-version cherry-pick CONTROL (all from v1) ---")
    model.set_version(0)
    flat_dict_restore(model, flat_v1)  # load all v1 params

    freeze_non_tree_leaves(model)
    unfreeze_gates_only(model)
    same_pick_raw = evaluate(model, joint_val, batch_size=32)
    print(f"  Same-version cherry-pick raw val_loss: {same_pick_raw:.4f}")

    print("  Calibrating (100 steps, gates only)...")
    same_pick_cal = train_model(model, joint_train, joint_val,
                                steps=100, seed=seed, lr=3e-3)
    results["same_version_cherrypick"] = {
        "raw_val_loss": same_pick_raw,
        "calibrated_val_loss": same_pick_cal["val_loss"],
    }
    print(f"  Same-version cherry-pick calibrated: {same_pick_cal['val_loss']:.4f}")

    # ── 4c. Cross-version composition via tree API ────────────────────
    print("\n--- Phase 4c: Cross-version cherry-pick (A from v1, B from v2) via tree API ---")

    # Use compose_versions API: leaves 0-3 from v1, leaves 4-7 from v2
    model.set_version(0)
    leaf_version_map = {i: v1_id for i in range(4)}
    leaf_version_map.update({i: v2_id for i in range(4, 8)})
    v3_id = model.compose_versions(leaf_version_map,
                                    description="cross-version: A from v1, B from v2")
    model.set_version(v3_id)

    freeze_non_tree_leaves(model)
    unfreeze_gates_only(model)
    cross_raw = evaluate(model, joint_val, batch_size=32)
    print(f"  Cross-version raw val_loss: {cross_raw:.4f}")

    print("  Calibrating (100 steps, gates only)...")
    cross_cal = train_model(model, joint_train, joint_val,
                            steps=100, seed=seed, lr=3e-3)
    results["cross_version"] = {
        "raw_val_loss": cross_raw,
        "calibrated_val_loss": cross_cal["val_loss"],
    }
    print(f"  Cross-version calibrated: {cross_cal['val_loss']:.4f}")

    # ── 5. Memory overhead via tree API memory_report() ─────────────────
    print("\n--- Phase 6: Memory overhead via tree API ---")
    mem_report = model.memory_report()
    results["tree_memory_report"] = mem_report

    print(f"  Versions: {mem_report['n_versions']}")
    print(f"  Persistent params: {mem_report['total_persistent_params']:,}")
    print(f"  Full-copy params:  {mem_report['total_full_copy_params']:,}")
    print(f"  Base params:       {mem_report['base_params']:.0f}")
    print(f"  Overhead vs mutable: {mem_report['memory_overhead_pct']:.1f}%")
    print(f"  Savings vs full-copy: {mem_report['memory_savings_vs_full_copy_pct']:.1f}%")

    # ── 6. Flat-dict baseline comparison ──────────────────────────────
    print("\n--- Phase 6: Flat-dict baseline (naive checkpointing) ---")
    flat_snapshots = {"v0": flat_v0, "v1": flat_v1, "v2": flat_v2}
    flat_total_params = flat_dict_memory(flat_snapshots)
    flat_base_params = sum(v.size for v in flat_v0.values())

    results["flat_dict_memory"] = {
        "total_params": flat_total_params,
        "base_params": flat_base_params,
        "overhead_vs_mutable_pct": (flat_total_params - flat_base_params) / flat_base_params * 100,
        "n_snapshots": len(flat_snapshots),
    }

    print(f"  Flat-dict total: {flat_total_params:,} ({len(flat_snapshots)} snapshots)")
    print(f"  Flat-dict overhead: {(flat_total_params - flat_base_params) / flat_base_params * 100:.1f}%")
    print(f"  Tree overhead:      {mem_report['memory_overhead_pct']:.1f}%")
    print(f"  Tree savings vs flat: {(1 - mem_report['total_persistent_params'] / flat_total_params) * 100:.1f}%")

    results["tree_vs_flat"] = {
        "tree_persistent_params": mem_report["total_persistent_params"],
        "flat_total_params": flat_total_params,
        "tree_savings_vs_flat_pct": (1 - mem_report["total_persistent_params"] / flat_total_params) * 100,
    }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY (seed={seed})")
    print(f"{'='*60}")

    joint_val_loss = results["joint"]["val_loss"]
    same_avg_val = results["same_version_avg"]["calibrated_val_loss"]
    same_pick_val = results["same_version_cherrypick"]["calibrated_val_loss"]
    cross_val = results["cross_version"]["calibrated_val_loss"]

    cross_vs_same_pick_pct = (cross_val - same_pick_val) / same_pick_val * 100
    cross_vs_same_avg_pct = (cross_val - same_avg_val) / same_avg_val * 100
    cross_vs_joint_pct = (cross_val - joint_val_loss) / joint_val_loss * 100
    same_pick_vs_joint_pct = (same_pick_val - joint_val_loss) / joint_val_loss * 100
    same_avg_vs_joint_pct = (same_avg_val - joint_val_loss) / joint_val_loss * 100

    print(f"  Joint training:                 {joint_val_loss:.4f}")
    print(f"  Same-version avg (calibrated):  {same_avg_val:.4f} ({same_avg_vs_joint_pct:+.2f}% vs joint)")
    print(f"  Same-version pick (calibrated): {same_pick_val:.4f} ({same_pick_vs_joint_pct:+.2f}% vs joint)")
    print(f"  Cross-version (calibrated):     {cross_val:.4f} ({cross_vs_joint_pct:+.2f}% vs joint)")
    print(f"  Cross vs Same-pick:             {cross_vs_same_pick_pct:+.2f}%")
    print(f"  Cross vs Same-avg:              {cross_vs_same_avg_pct:+.2f}%")
    print(f"  Memory overhead (tree API):     {mem_report['memory_overhead_pct']:.1f}%")
    print(f"  Memory savings vs flat-dict:    {results['tree_vs_flat']['tree_savings_vs_flat_pct']:.1f}%")
    print(f"  Rollback fidelity:              {rollback_diff:.2e}")
    print()

    kc1_pass = abs(cross_vs_same_pick_pct) <= 5.0
    kc2_pass = mem_report["memory_overhead_pct"] <= 15.0
    print(f"  KC1 (cross-version <=5% worse vs same-pick): {'PASS' if kc1_pass else 'FAIL'} ({cross_vs_same_pick_pct:+.2f}%)")
    print(f"  KC2 (tree overhead <=15%):                   {'PASS' if kc2_pass else 'FAIL'} ({mem_report['memory_overhead_pct']:.1f}%)")

    results["summary"] = {
        "joint_val_loss": joint_val_loss,
        "same_version_avg_val_loss": same_avg_val,
        "same_version_cherrypick_val_loss": same_pick_val,
        "cross_version_val_loss": cross_val,
        "cross_vs_same_pick_pct": cross_vs_same_pick_pct,
        "cross_vs_same_avg_pct": cross_vs_same_avg_pct,
        "cross_vs_joint_pct": cross_vs_joint_pct,
        "same_pick_vs_joint_pct": same_pick_vs_joint_pct,
        "same_avg_vs_joint_pct": same_avg_vs_joint_pct,
        "memory_overhead_pct": mem_report["memory_overhead_pct"],
        "memory_savings_vs_full_copy_pct": mem_report["memory_savings_vs_full_copy_pct"],
        "tree_savings_vs_flat_pct": results["tree_vs_flat"]["tree_savings_vs_flat_pct"],
        "rollback_max_diff": rollback_diff,
        "kc1_pass": kc1_pass,
        "kc2_pass": kc2_pass,
    }

    return results


def main():
    all_results = []
    seeds = [42, 123, 7]

    for seed in seeds:
        result = run_experiment(seed=seed)
        all_results.append(result)

    # ── Aggregate across seeds ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*60}")

    joint_losses = [r["summary"]["joint_val_loss"] for r in all_results]
    same_avg_losses = [r["summary"]["same_version_avg_val_loss"] for r in all_results]
    same_pick_losses = [r["summary"]["same_version_cherrypick_val_loss"] for r in all_results]
    cross_losses = [r["summary"]["cross_version_val_loss"] for r in all_results]
    cross_vs_same_pick = [r["summary"]["cross_vs_same_pick_pct"] for r in all_results]
    overheads = [r["summary"]["memory_overhead_pct"] for r in all_results]
    savings = [r["summary"]["memory_savings_vs_full_copy_pct"] for r in all_results]
    tree_vs_flat = [r["summary"]["tree_savings_vs_flat_pct"] for r in all_results]
    rollbacks = [r["summary"]["rollback_max_diff"] for r in all_results]

    def mean(xs): return sum(xs) / len(xs)

    print(f"  Joint:              {mean(joint_losses):.4f}")
    print(f"  Same-version avg:   {mean(same_avg_losses):.4f}")
    print(f"  Same-version pick:  {mean(same_pick_losses):.4f}")
    print(f"  Cross-version:      {mean(cross_losses):.4f}")
    print(f"  Cross vs Same-pick: {mean(cross_vs_same_pick):+.2f}% (per seed: {[f'{x:+.2f}%' for x in cross_vs_same_pick]})")
    print(f"  Mem overhead:       {mean(overheads):.1f}%")
    print(f"  Savings vs full-copy: {mean(savings):.1f}%")
    print(f"  Tree vs flat-dict:  {mean(tree_vs_flat):.1f}%")
    print(f"  Rollback diff:      {mean(rollbacks):.2e}")

    mean_cross_vs_same = mean(cross_vs_same_pick)
    mean_overhead = mean(overheads)

    kc1_overall = abs(mean_cross_vs_same) <= 5.0
    kc2_overall = mean_overhead <= 15.0

    print()
    print(f"  KC1 (cross-version <=5% worse): {'PASS' if kc1_overall else 'FAIL'} (mean {mean_cross_vs_same:+.2f}%)")
    print(f"  KC2 (overhead <=15%):           {'PASS' if kc2_overall else 'FAIL'} (mean {mean_overhead:.1f}%)")

    # Save results
    output = {
        "experiment": "persistent_expert_tree_v2",
        "revision": "v2 -- uses tree API, leaf-only fine-tuning, flat-dict baseline, same-version cherry-pick control",
        "seeds": seeds,
        "per_seed": [r["summary"] for r in all_results],
        "aggregate": {
            "mean_joint_val_loss": mean(joint_losses),
            "mean_same_version_avg_val_loss": mean(same_avg_losses),
            "mean_same_version_cherrypick_val_loss": mean(same_pick_losses),
            "mean_cross_version_val_loss": mean(cross_losses),
            "mean_cross_vs_same_pick_pct": mean_cross_vs_same,
            "mean_memory_overhead_pct": mean_overhead,
            "mean_savings_vs_full_copy_pct": mean(savings),
            "mean_tree_savings_vs_flat_pct": mean(tree_vs_flat),
            "mean_rollback_diff": mean(rollbacks),
            "kc1_pass": kc1_overall,
            "kc2_pass": kc2_overall,
        },
        "param_breakdown": all_results[0].get("param_breakdown", {}),
        "tree_memory_report_seed42": all_results[0].get("tree_memory_report", {}),
        "flat_dict_memory_seed42": all_results[0].get("flat_dict_memory", {}),
        "tree_vs_flat_seed42": all_results[0].get("tree_vs_flat", {}),
    }

    out_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/persistent_expert_tree/results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
