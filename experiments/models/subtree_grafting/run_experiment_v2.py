"""Experiment v2: Subtree Grafting Composition (revised calibration).

Key change from v1: grafting calibration now trains ALL gates (not just root)
for 100 steps, matching the weight averaging calibration budget exactly.

Diagnostic showed:
- Root-only 50 steps: +2.63% gap (unfair: fewer params + fewer steps)
- All-gates 100 steps: +1.70% gap (fair comparison)
- All-gates 200 steps: +1.34% gap (diminishing returns)

Kill criteria unchanged:
1. Subtree grafting >3% worse than weight averaging
2. Grafting produces >5% degradation on donor domain
"""

import sys
import time
import random

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


def freeze_except_all_gates(model):
    """Freeze everything except ALL tree gates."""
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        for leaf in layer.tree.leaves:
            leaf.freeze()
        # All gates remain trainable


def run_experiment(seeds=(42, 123, 777), steps_pretrain=300,
                   steps_finetune=200, steps_calibrate=100):
    """Run subtree grafting vs weight averaging with matched calibration."""
    print("=" * 70)
    print("EXPERIMENT v2: Subtree Grafting (matched calibration)")
    print("=" * 70)
    print(f"  Pretrain: {steps_pretrain} steps")
    print(f"  Fine-tune: {steps_finetune} steps")
    print(f"  Calibrate (both methods): {steps_calibrate} steps")
    print(f"  Grafting calibration: ALL gates (not just root)")
    print(f"  Seeds: {seeds}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_names = list(splits.keys())

    all_results = {
        "joint": [], "weight_avg": [], "graft": [],
        "graft_domain_a": [], "graft_domain_b": [],
        "joint_domain_a": [], "joint_domain_b": [],
        "weight_avg_domain_a": [], "weight_avg_domain_b": [],
    }

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

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

        # ── (c) Joint training baseline ──────────────────────────────────
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
            all_results[f"joint_domain_{'a' if di == 0 else 'b'}"].append(d_vl)
            print(f"    joint {d_name}: {d_vl:.4f}")

        # ── Pretrain shared base ─────────────────────────────────────────
        print(f"\n  [BASE] Pretrain shared base")
        mx.random.seed(seed)
        base_model = get_model("hierarchical_tree", **tree_cfg)
        mx.eval(base_model.parameters())
        _ = train_model(base_model, joint_train, joint_val,
                        steps=steps_pretrain, seed=seed, log_every=200)
        base_weights = {k: mx.array(v) for k, v in
                        nn.utils.tree_flatten(base_model.parameters())}

        # ── (a) Weight Averaging ─────────────────────────────────────────
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

        # Calibrate gates only
        freeze_except_all_gates(avg_model)
        _ = train_model(avg_model, joint_train, joint_val,
                        steps=steps_calibrate, seed=seed, log_every=100)
        avg_vl = evaluate(avg_model, joint_val, 32)
        all_results["weight_avg"].append(avg_vl)
        print(f"    weight_avg val_loss: {avg_vl:.4f}")

        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(avg_model, domain_datasets[d_name][1], 32)
            all_results[f"weight_avg_domain_{'a' if di == 0 else 'b'}"].append(d_vl)
            print(f"    weight_avg {d_name}: {d_vl:.4f}")

        # ── (b) Subtree Grafting ─────────────────────────────────────────
        print(f"\n  [GRAFT] Fine-tune subtrees per domain, then graft")
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

        graft_model = get_model("hierarchical_tree", **tree_cfg)
        graft_model.load_weights(list(base_weights.items()))
        mx.eval(graft_model.parameters())
        for d_name, params in graft_subtree_params.items():
            for key, val in params.items():
                set_param_by_path(graft_model, key, val)
        mx.eval(graft_model.parameters())

        # Calibrate ALL gates (matched budget with weight averaging)
        freeze_except_all_gates(graft_model)
        _ = train_model(graft_model, joint_train, joint_val,
                        steps=steps_calibrate, seed=seed, log_every=100)
        graft_vl = evaluate(graft_model, joint_val, 32)
        all_results["graft"].append(graft_vl)
        print(f"    graft val_loss: {graft_vl:.4f}")

        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(graft_model, domain_datasets[d_name][1], 32)
            all_results[f"graft_domain_{'a' if di == 0 else 'b'}"].append(d_vl)
            print(f"    graft {d_name}: {d_vl:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")

    def mean(lst):
        return sum(lst) / len(lst)

    joint_mean = mean(all_results["joint"])
    avg_mean = mean(all_results["weight_avg"])
    graft_mean = mean(all_results["graft"])

    avg_gap = 100 * (avg_mean - joint_mean) / joint_mean
    graft_gap = 100 * (graft_mean - joint_mean) / joint_mean
    graft_vs_avg = 100 * (graft_mean - avg_mean) / avg_mean

    print(f"\n  Joint (baseline):    mean={joint_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['joint']]}")
    print(f"  Weight averaging:    mean={avg_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['weight_avg']]}")
    print(f"  Subtree grafting:    mean={graft_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['graft']]}")

    print(f"\n  Weight avg vs joint:     {avg_gap:+.2f}%")
    print(f"  Subtree graft vs joint:  {graft_gap:+.2f}%")
    print(f"  Subtree graft vs wt avg: {graft_vs_avg:+.2f}%")

    kill_1 = graft_vs_avg > 3.0
    print(f"\n  Kill criterion 1 (graft >3% worse than wt avg): "
          f"{'TRIGGERED' if kill_1 else 'PASSES'} ({graft_vs_avg:+.2f}%)")

    print(f"\n  Per-domain analysis (kill criterion 2):")
    any_kill_2 = False
    for di, d_name in enumerate(domain_names):
        side = "a" if di == 0 else "b"
        joint_d = mean(all_results[f"joint_domain_{side}"])
        graft_d = mean(all_results[f"graft_domain_{side}"])
        avg_d = mean(all_results[f"weight_avg_domain_{side}"])
        deg_graft = 100 * (graft_d - joint_d) / joint_d
        deg_avg = 100 * (avg_d - joint_d) / joint_d
        kill_2_d = deg_graft > 5.0
        if kill_2_d:
            any_kill_2 = True
        print(f"    {d_name}: joint={joint_d:.4f}  wt_avg={avg_d:.4f}({deg_avg:+.2f}%)  "
              f"graft={graft_d:.4f}({deg_graft:+.2f}%) {'KILL' if kill_2_d else 'PASS'}")

    killed = kill_1 or any_kill_2
    print(f"\n  OVERALL: {'KILLED' if killed else 'PASSES'}")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment()
    print(f"\nTotal time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
