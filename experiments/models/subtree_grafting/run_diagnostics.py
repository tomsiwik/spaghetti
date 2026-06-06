"""Diagnostic: Investigate subtree grafting calibration budget.

The main experiment triggered kill criterion 1 at +3.57% (graft vs weight avg).
This diagnostic tests whether more root gate calibration steps close the gap,
or whether the degradation is fundamental.

Also tests: grafting with more calibration freedom (train all gates, not just root).
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
    freeze_except_root_gate, freeze_except_tree, get_tree_params,
)


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
    return {"val_loss": val_loss}


def freeze_except_all_gates(model):
    """Freeze everything except ALL gates (not just root)."""
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


def run_calibration_sweep():
    """Sweep graft calibration steps and compare against weight averaging."""
    print("=" * 70)
    print("DIAGNOSTIC: Grafting Calibration Budget Sweep")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)

    seed = 42
    steps_pretrain = 300
    steps_finetune = 200

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

    # Pretrain base
    print(f"\n  [BASE] Pretrain shared base")
    mx.random.seed(seed)
    base_model = get_model("hierarchical_tree", **tree_cfg)
    mx.eval(base_model.parameters())
    _ = train_model(base_model, joint_train, joint_val, steps=steps_pretrain,
                    seed=seed, log_every=200)
    base_weights = {k: mx.array(v) for k, v in
                    nn.utils.tree_flatten(base_model.parameters())}

    # Weight averaging baseline (for comparison)
    print(f"\n  [WEIGHT AVG] Fine-tune + average + calibrate 100 steps")
    domain_tree_params = {}
    domain_names = list(splits.keys())
    for d_name, (d_train, d_val) in domain_datasets.items():
        mx.random.seed(seed)
        d_model = get_model("hierarchical_tree", **tree_cfg)
        d_model.load_weights(list(base_weights.items()))
        mx.eval(d_model.parameters())
        freeze_except_tree(d_model)
        _ = train_model(d_model, d_train, d_val, steps=steps_finetune,
                        seed=seed, log_every=200)
        domain_tree_params[d_name] = get_tree_params(d_model)

    avg_model = get_model("hierarchical_tree", **tree_cfg)
    avg_model.load_weights(list(base_weights.items()))
    mx.eval(avg_model.parameters())
    for key in domain_tree_params[domain_names[0]]:
        avg_w = sum(domain_tree_params[d][key] for d in domain_names) / len(domain_names)
        set_param_by_path(avg_model, key, avg_w)
    mx.eval(avg_model.parameters())
    avg_model.wte.freeze(); avg_model.wpe.freeze(); avg_model.norm0.freeze(); avg_model.lm_head.freeze()
    for layer in avg_model.layers:
        layer.norm1.freeze(); layer.attn.freeze(); layer.norm2.freeze()
        for leaf in layer.tree.leaves:
            leaf.freeze()
    _ = train_model(avg_model, joint_train, joint_val, steps=100, seed=seed, log_every=100)
    avg_vl = evaluate(avg_model, joint_val, 32)
    print(f"    weight_avg val_loss: {avg_vl:.4f}")

    # Prepare grafted subtrees
    graft_subtree_params = {}
    for di, (d_name, (d_train, d_val)) in enumerate(domain_datasets.items()):
        side = "left" if di == 0 else "right"
        mx.random.seed(seed)
        d_model = get_model("hierarchical_tree", **tree_cfg)
        d_model.load_weights(list(base_weights.items()))
        mx.eval(d_model.parameters())
        freeze_except_subtree(d_model, side=side)
        _ = train_model(d_model, d_train, d_val, steps=steps_finetune,
                        seed=seed, log_every=200)
        graft_subtree_params[d_name] = get_subtree_params(d_model, side=side)

    # Sweep calibration budgets with root-only vs all-gates
    calibration_configs = [
        ("root_only_50",   50,  "root"),
        ("root_only_100", 100,  "root"),
        ("root_only_200", 200,  "root"),
        ("all_gates_50",   50,  "all_gates"),
        ("all_gates_100", 100,  "all_gates"),
        ("all_gates_200", 200,  "all_gates"),
    ]

    print(f"\n  [GRAFT SWEEP]")
    results = {"weight_avg": avg_vl}

    for name, cal_steps, freeze_mode in calibration_configs:
        graft_model = get_model("hierarchical_tree", **tree_cfg)
        graft_model.load_weights(list(base_weights.items()))
        mx.eval(graft_model.parameters())
        for d_name, params in graft_subtree_params.items():
            for key, val in params.items():
                set_param_by_path(graft_model, key, val)
        mx.eval(graft_model.parameters())

        if freeze_mode == "root":
            freeze_except_root_gate(graft_model)
        else:
            freeze_except_all_gates(graft_model)

        _ = train_model(graft_model, joint_train, joint_val,
                        steps=cal_steps, seed=seed, log_every=cal_steps)
        vl = evaluate(graft_model, joint_val, 32)
        gap_vs_avg = 100 * (vl - avg_vl) / avg_vl
        results[name] = vl
        print(f"    {name:20s}: val_loss={vl:.4f}  vs_wt_avg={gap_vs_avg:+.2f}%")

    print(f"\n  Summary:")
    print(f"    Weight averaging (100 cal steps): {results['weight_avg']:.4f}")
    for name, vl in results.items():
        if name != "weight_avg":
            gap = 100 * (vl - results["weight_avg"]) / results["weight_avg"]
            print(f"    {name:20s}: {vl:.4f}  gap={gap:+.2f}%")


if __name__ == "__main__":
    t0 = time.time()
    run_calibration_sweep()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
