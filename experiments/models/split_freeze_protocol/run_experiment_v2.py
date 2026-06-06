"""V2: Diagnosis of KC2 failure -- is it calibration budget or structural?

The v1 experiment showed +31.28% degradation on frozen domain A after grafting.
The root gate (260 params, 100 steps) couldn't learn to route properly.

This v2 tries:
1. All-gates calibration (not just root) -- 1820 params, 200 steps
2. All-gates + leaves calibration (full tree except frozen subtree)
3. Larger calibration budget (400 steps)

If degradation drops below 2% with more calibration, the protocol works
but needs sufficient calibration budget. If it stays high, the issue is
structural: frozen branches interfere with new branches at a deeper level.
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
from experiments.models.split_freeze_protocol.split_freeze_protocol import (
    freeze_leaves,
    freeze_gates,
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
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"val_loss": val_loss, "elapsed_s": time.time() - t0}


def run_freeze_diagnostic(seeds=(42, 123, 777), steps_domain_a=400,
                           steps_graft_b=200):
    """Sweep calibration configurations to diagnose KC2 failure."""
    print("=" * 70)
    print("KC2 DIAGNOSTIC: Calibration budget sweep")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_names = list(splits.keys())

    left_gates = {1, 3, 4}
    left_leaves = {0, 1, 2, 3}
    right_gates = {2, 5, 6}
    right_leaves = {4, 5, 6, 7}

    calibration_configs = [
        ("root-only_100",   {"trainable_gates": {0}, "trainable_leaves": set(), "steps": 100}),
        ("root-only_200",   {"trainable_gates": {0}, "trainable_leaves": set(), "steps": 200}),
        ("all-gates_100",   {"trainable_gates": {0} | right_gates, "trainable_leaves": set(), "steps": 100}),
        ("all-gates_200",   {"trainable_gates": {0} | right_gates, "trainable_leaves": set(), "steps": 200}),
        ("right-tree_200",  {"trainable_gates": {0} | right_gates, "trainable_leaves": right_leaves, "steps": 200}),
        ("right-tree_400",  {"trainable_gates": {0} | right_gates, "trainable_leaves": right_leaves, "steps": 400}),
    ]

    all_results = {}

    for seed in seeds:
        print(f"\n  SEED {seed}")
        print(f"  {'='*60}")

        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )
        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train_docs, tokenizer, 32)
        joint_val = CharDataset(all_val_docs, tokenizer, 32)

        tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                        n_capsules_per_leaf=32, beam_width=2)

        d_a_name = domain_names[0]
        d_a_train, d_a_val = domain_datasets[d_a_name]
        d_b_name = domain_names[1]
        d_b_train, d_b_val = domain_datasets[d_b_name]

        # Train on domain A
        print(f"\n  [DOMAIN A] Training tree on {d_a_name} ({steps_domain_a} steps)")
        mx.random.seed(seed)
        base_model = get_model("split_freeze_protocol", **tree_cfg)
        mx.eval(base_model.parameters())
        _ = train_model(base_model, d_a_train, d_a_val,
                        steps=steps_domain_a, seed=seed, log_every=200)

        pre_graft_a = evaluate(base_model, d_a_val, 32)
        print(f"    pre-graft domain A val_loss: {pre_graft_a:.4f}")

        # Save base weights
        base_weights = {k: mx.array(v) for k, v in nn.utils.tree_flatten(base_model.parameters())}

        # Test each calibration config
        for config_name, config in calibration_configs:
            print(f"\n  [{config_name}] Calibration config")

            # Start from base model
            mx.random.seed(seed)
            model = get_model("split_freeze_protocol", **tree_cfg)
            model.load_weights(list(base_weights.items()))
            mx.eval(model.parameters())

            # Freeze left subtree
            freeze_leaves(model, left_leaves)
            freeze_gates(model, left_gates)

            # Reinitialize right subtree
            for layer in model.layers:
                for li in right_leaves:
                    d = layer.tree.leaves[li].A.weight.shape[1]
                    n_caps = layer.tree.leaves[li].A.weight.shape[0]
                    scale = (2.0 / (d + n_caps)) ** 0.5
                    new_A = mx.random.normal((n_caps, d)) * scale
                    new_B = mx.random.normal((d, n_caps)) * scale
                    layer.tree.leaves[li].A.load_weights([("weight", new_A)])
                    layer.tree.leaves[li].B.load_weights([("weight", new_B)])
                for gi in right_gates:
                    gate = layer.tree.gates[gi]
                    d_in = gate.proj.weight.shape[1]
                    s = (1.0 / d_in) ** 0.5
                    new_w = mx.random.normal(gate.proj.weight.shape) * s
                    new_b = mx.zeros(gate.proj.bias.shape)
                    gate.proj.load_weights([("weight", new_w), ("bias", new_b)])
            mx.eval(model.parameters())

            # Freeze shared params
            model.wte.freeze()
            model.wpe.freeze()
            model.norm0.freeze()
            model.lm_head.freeze()
            for layer in model.layers:
                layer.norm1.freeze()
                layer.attn.freeze()
                layer.norm2.freeze()

            # Train right subtree on domain B
            _ = train_model(model, d_b_train, d_b_val,
                            steps=steps_graft_b, seed=seed, log_every=200)

            # Now apply calibration config
            # Freeze everything, then selectively unfreeze
            for layer in model.layers:
                for gi in range(len(layer.tree.gates)):
                    layer.tree.gates[gi].freeze()
                for li in range(len(layer.tree.leaves)):
                    layer.tree.leaves[li].freeze()

            # Unfreeze calibration params
            for layer in model.layers:
                for gi in config["trainable_gates"]:
                    if gi < len(layer.tree.gates):
                        layer.tree.gates[gi].unfreeze()
                for li in config["trainable_leaves"]:
                    if li < len(layer.tree.leaves):
                        layer.tree.leaves[li].unfreeze()

            n_trainable = count_params(model)
            print(f"    trainable params: {n_trainable}, steps: {config['steps']}")

            _ = train_model(model, joint_train, joint_val,
                            steps=config["steps"], seed=seed, log_every=200)

            post_a = evaluate(model, d_a_val, 32)
            post_b = evaluate(model, d_b_val, 32)
            degradation = 100 * (post_a - pre_graft_a) / pre_graft_a

            print(f"    domain A: {post_a:.4f} (degradation: {degradation:+.2f}%)")
            print(f"    domain B: {post_b:.4f}")

            key = f"{config_name}"
            if key not in all_results:
                all_results[key] = {"degradation": [], "a": [], "b": [], "pre_a": []}
            all_results[key]["degradation"].append(degradation)
            all_results[key]["a"].append(post_a)
            all_results[key]["b"].append(post_b)
            all_results[key]["pre_a"].append(pre_graft_a)

    # Summary
    print(f"\n{'='*70}")
    print(f"CALIBRATION SWEEP RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"\n  {'Config':<20} | {'Mean Degr.':<12} | {'Per-seed degradations'}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*40}")

    for config_name, _ in calibration_configs:
        data = all_results[config_name]
        mean_deg = sum(data["degradation"]) / len(data["degradation"])
        per_seed = ", ".join(f"{d:+.2f}%" for d in data["degradation"])
        status = "PASS" if mean_deg <= 2.0 else "KILL"
        print(f"  {config_name:<20} | {mean_deg:+8.2f}%    | {per_seed}  [{status}]")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_freeze_diagnostic()
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
