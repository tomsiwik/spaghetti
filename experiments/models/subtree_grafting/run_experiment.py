"""Experiment: Subtree Grafting Composition vs Weight Averaging vs Joint Training.

Compares three composition methods on the hierarchical capsule tree:

(a) Weight averaging (baseline from hierarchical_tree experiment):
    Average all domain-specific tree parameters (gates + leaves).

(b) Subtree grafting:
    Split tree at root. Domain A owns left subtree (leaves 0-3, gates 1,3,4).
    Domain B owns right subtree (leaves 4-7, gates 2,5,6). Graft them together.
    Retrain only root gate (gate 0) to route between domain subtrees.

(c) Full joint training (upper bound):
    Train entire tree on all data from scratch.

Kill criteria:
1. Subtree grafting composition >3% worse than weight averaging
2. Grafting produces >5% degradation on donor subtree's original domain

Protocol:
1. Pretrain base model on all data (300 steps)
2. Fine-tune per domain (200 steps, attention frozen):
   - For grafting: constrain each domain to its assigned subtree
   - For weight averaging: fine-tune full tree per domain
3. Compose:
   - Grafting: combine subtrees, retrain root gate (50 steps)
   - Weight averaging: average domain params, calibrate gates (100 steps)
4. Evaluate on joint val set AND per-domain val sets
"""

import sys
import time
import random
import json
import copy

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


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


def get_tree_params(model):
    """Extract all tree parameters as a flat dict."""
    params = {}
    for layer_idx, layer in enumerate(model.layers):
        for k, v in nn.utils.tree_flatten(layer.tree.parameters()):
            full_key = f"layers.{layer_idx}.tree.{k}"
            params[full_key] = mx.array(v)
    return params


def get_subtree_params(model, side="left"):
    """Extract parameters for one subtree (left or right of root).

    In a depth-3 tree:
    - Internal gates: 0 (root), 1 (left child), 2 (right child),
                      3,4 (children of 1), 5,6 (children of 2)
    - Leaves: 0-3 (left subtree), 4-7 (right subtree)

    Left subtree: gates [1, 3, 4], leaves [0, 1, 2, 3]
    Right subtree: gates [2, 5, 6], leaves [4, 5, 6, 7]
    """
    if side == "left":
        gate_indices = [1, 3, 4]
        leaf_indices = [0, 1, 2, 3]
    else:
        gate_indices = [2, 5, 6]
        leaf_indices = [4, 5, 6, 7]

    params = {}
    for layer_idx, layer in enumerate(model.layers):
        tree = layer.tree
        # Gates
        for gi in gate_indices:
            for k, v in nn.utils.tree_flatten(tree.gates[gi].parameters()):
                full_key = f"layers.{layer_idx}.tree.gates.{gi}.{k}"
                params[full_key] = mx.array(v)
        # Leaves
        for li in leaf_indices:
            for k, v in nn.utils.tree_flatten(tree.leaves[li].parameters()):
                full_key = f"layers.{layer_idx}.tree.leaves.{li}.{k}"
                params[full_key] = mx.array(v)

    return params


def set_param_by_path(model, path, value):
    """Set a parameter in the model by dotted path string."""
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def freeze_except_root_gate(model):
    """Freeze everything except the root gate (gate[0]) in each layer.

    For subtree grafting: after grafting subtrees, only the root gate
    (domain router) needs retraining.
    """
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        # Freeze all gates except root (index 0)
        for gi in range(1, len(layer.tree.gates)):
            layer.tree.gates[gi].freeze()
        # Freeze all leaves
        for leaf in layer.tree.leaves:
            leaf.freeze()
        # Root gate (index 0) stays trainable


def freeze_except_subtree(model, side="left"):
    """Freeze everything except one subtree.

    For domain-specific fine-tuning: only train the assigned subtree.
    """
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()

    if side == "left":
        trainable_gates = {1, 3, 4}
        trainable_leaves = {0, 1, 2, 3}
    else:
        trainable_gates = {2, 5, 6}
        trainable_leaves = {4, 5, 6, 7}

    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        # Freeze root gate always during domain fine-tuning
        layer.tree.gates[0].freeze()
        # Freeze non-assigned gates
        for gi in range(len(layer.tree.gates)):
            if gi not in trainable_gates:
                layer.tree.gates[gi].freeze()
        # Freeze non-assigned leaves
        for li in range(len(layer.tree.leaves)):
            if li not in trainable_leaves:
                layer.tree.leaves[li].freeze()


def freeze_except_tree(model):
    """Freeze everything except tree params (gates + leaves)."""
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()


def run_experiment(seeds=(42, 123, 777), steps_pretrain=300,
                   steps_finetune=200, steps_calibrate=100,
                   steps_graft_calibrate=50):
    """Run the full subtree grafting comparison."""
    print("=" * 70)
    print("EXPERIMENT: Subtree Grafting Composition")
    print("=" * 70)
    print(f"  Pretrain: {steps_pretrain} steps")
    print(f"  Fine-tune: {steps_finetune} steps")
    print(f"  Calibrate (weight avg): {steps_calibrate} steps")
    print(f"  Calibrate (grafting): {steps_graft_calibrate} steps")
    print(f"  Seeds: {seeds}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_names = list(splits.keys())

    all_results = {
        "joint": [],           # full joint training val_loss
        "weight_avg": [],      # weight averaging composition val_loss
        "graft": [],           # subtree grafting composition val_loss
        "graft_domain_a": [],  # grafting: domain A val_loss (donor preservation)
        "graft_domain_b": [],  # grafting: domain B val_loss (donor preservation)
        "joint_domain_a": [],  # joint: domain A val_loss (reference)
        "joint_domain_b": [],  # joint: domain B val_loss (reference)
    }

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

        # ── (c) Joint training baseline ──────────────────────────────────
        print(f"\n  [JOINT] Full tree training on all data")
        mx.random.seed(seed)
        joint_model = get_model("hierarchical_tree", **tree_cfg)
        mx.eval(joint_model.parameters())
        joint_result = train_model(joint_model, joint_train, joint_val,
                                   steps=steps_pretrain + steps_finetune,
                                   seed=seed, log_every=200)
        joint_vl = joint_result["val_loss"]
        all_results["joint"].append(joint_vl)
        print(f"    joint val_loss: {joint_vl:.4f}")

        # Per-domain eval of joint model
        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(joint_model, domain_datasets[d_name][1], 32)
            all_results[f"joint_domain_{'a' if di == 0 else 'b'}"].append(d_vl)
            print(f"    joint {d_name} val_loss: {d_vl:.4f}")

        # ── Pretrain shared base ─────────────────────────────────────────
        print(f"\n  [BASE] Pretrain shared base on all data")
        mx.random.seed(seed)
        base_model = get_model("hierarchical_tree", **tree_cfg)
        mx.eval(base_model.parameters())
        _ = train_model(base_model, joint_train, joint_val,
                        steps=steps_pretrain, seed=seed, log_every=200)

        # Save base weights
        base_weights = {k: mx.array(v) for k, v in
                        nn.utils.tree_flatten(base_model.parameters())}

        # ── (a) Weight Averaging Composition ─────────────────────────────
        print(f"\n  [WEIGHT AVG] Fine-tune full tree per domain, then average")
        domain_tree_params = {}
        for d_name, (d_train, d_val) in domain_datasets.items():
            print(f"    fine-tune on {d_name}...")
            mx.random.seed(seed)
            d_model = get_model("hierarchical_tree", **tree_cfg)
            d_model.load_weights(list(base_weights.items()))
            mx.eval(d_model.parameters())

            # Freeze attention+embeddings, train only tree
            freeze_except_tree(d_model)
            _ = train_model(d_model, d_train, d_val,
                            steps=steps_finetune, seed=seed, log_every=200)
            domain_tree_params[d_name] = get_tree_params(d_model)

        # Average domain tree weights
        print(f"    composing via weight averaging...")
        avg_model = get_model("hierarchical_tree", **tree_cfg)
        avg_model.load_weights(list(base_weights.items()))
        mx.eval(avg_model.parameters())

        n_domains = len(domain_names)
        all_keys = list(domain_tree_params[domain_names[0]].keys())
        for key in all_keys:
            avg_w = sum(domain_tree_params[d][key] for d in domain_names) / n_domains
            set_param_by_path(avg_model, key, avg_w)
        mx.eval(avg_model.parameters())

        # Calibrate gates (freeze leaves)
        avg_model.wte.freeze()
        avg_model.wpe.freeze()
        avg_model.norm0.freeze()
        avg_model.lm_head.freeze()
        for layer in avg_model.layers:
            layer.norm1.freeze()
            layer.attn.freeze()
            layer.norm2.freeze()
            for leaf in layer.tree.leaves:
                leaf.freeze()

        _ = train_model(avg_model, joint_train, joint_val,
                        steps=steps_calibrate, seed=seed, log_every=100)
        avg_vl = evaluate(avg_model, joint_val, 32)
        all_results["weight_avg"].append(avg_vl)
        print(f"    weight_avg val_loss: {avg_vl:.4f}")

        # ── (b) Subtree Grafting Composition ─────────────────────────────
        print(f"\n  [GRAFT] Fine-tune subtrees per domain, then graft")

        # Fine-tune left subtree on domain A, right subtree on domain B
        graft_subtree_params = {}
        for di, (d_name, (d_train, d_val)) in enumerate(domain_datasets.items()):
            side = "left" if di == 0 else "right"
            print(f"    fine-tune {side} subtree on {d_name}...")
            mx.random.seed(seed)
            d_model = get_model("hierarchical_tree", **tree_cfg)
            d_model.load_weights(list(base_weights.items()))
            mx.eval(d_model.parameters())

            # Freeze everything except assigned subtree
            freeze_except_subtree(d_model, side=side)
            _ = train_model(d_model, d_train, d_val,
                            steps=steps_finetune, seed=seed, log_every=200)
            graft_subtree_params[d_name] = get_subtree_params(d_model, side=side)

        # Graft: combine subtrees into one model
        print(f"    grafting subtrees...")
        graft_model = get_model("hierarchical_tree", **tree_cfg)
        graft_model.load_weights(list(base_weights.items()))
        mx.eval(graft_model.parameters())

        # Set domain-specific subtree params
        for d_name, params in graft_subtree_params.items():
            for key, val in params.items():
                set_param_by_path(graft_model, key, val)
        mx.eval(graft_model.parameters())

        # Calibrate only root gate
        freeze_except_root_gate(graft_model)
        _ = train_model(graft_model, joint_train, joint_val,
                        steps=steps_graft_calibrate, seed=seed, log_every=50)
        graft_vl = evaluate(graft_model, joint_val, 32)
        all_results["graft"].append(graft_vl)
        print(f"    graft val_loss: {graft_vl:.4f}")

        # Per-domain eval of grafted model (kill criterion 2)
        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(graft_model, domain_datasets[d_name][1], 32)
            all_results[f"graft_domain_{'a' if di == 0 else 'b'}"].append(d_vl)
            print(f"    graft {d_name} val_loss: {d_vl:.4f}")

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

    print(f"\n  Joint (baseline):    mean val_loss = {joint_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['joint']]}")
    print(f"  Weight averaging:    mean val_loss = {avg_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['weight_avg']]}")
    print(f"  Subtree grafting:    mean val_loss = {graft_mean:.4f}  per-seed: {['%.4f' % v for v in all_results['graft']]}")

    print(f"\n  Weight avg vs joint:     {avg_gap:+.2f}%")
    print(f"  Subtree graft vs joint:  {graft_gap:+.2f}%")
    print(f"  Subtree graft vs wt avg: {graft_vs_avg:+.2f}%")

    # Kill criterion 1: grafting >3% worse than weight averaging
    kill_1 = graft_vs_avg > 3.0
    print(f"\n  Kill criterion 1 (graft >3% worse than weight avg): "
          f"{'TRIGGERED' if kill_1 else 'PASSES'} ({graft_vs_avg:+.2f}%)")

    # Kill criterion 2: grafting >5% degradation on donor domain
    print(f"\n  Per-domain analysis (kill criterion 2):")
    for di, d_name in enumerate(domain_names):
        side = "a" if di == 0 else "b"
        joint_d = mean(all_results[f"joint_domain_{side}"])
        graft_d = mean(all_results[f"graft_domain_{side}"])
        degradation = 100 * (graft_d - joint_d) / joint_d
        kill_2_d = degradation > 5.0
        print(f"    {d_name}: joint={joint_d:.4f} graft={graft_d:.4f} "
              f"degradation={degradation:+.2f}% "
              f"{'KILL' if kill_2_d else 'PASS'}")
        if kill_2_d:
            print(f"    >>> KILL CRITERION 2 TRIGGERED for {d_name}")

    # Overall kill assessment
    any_kill_2 = False
    for di, d_name in enumerate(domain_names):
        side = "a" if di == 0 else "b"
        joint_d = mean(all_results[f"joint_domain_{side}"])
        graft_d = mean(all_results[f"graft_domain_{side}"])
        if 100 * (graft_d - joint_d) / joint_d > 5.0:
            any_kill_2 = True

    killed = kill_1 or any_kill_2
    print(f"\n  OVERALL: {'KILLED' if killed else 'PASSES'}")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment()
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
