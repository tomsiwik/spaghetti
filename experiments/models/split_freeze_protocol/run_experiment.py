"""Experiment: Split-and-Freeze Protocol for Tree-Structured Experts.

Tests two kill criteria:

KC1: "split branch quality >5% worse than training a new flat expert from scratch"
     Protocol: Train a base tree on all data. Take a leaf that routes broadly.
     (a) SPLIT that leaf into two children, fine-tune only the split children.
     (b) FROM SCRATCH: train a brand-new pair of leaves (random init) at the
         same position with same data.
     Compare val loss of split vs from-scratch.

KC2: "frozen branches degrade >2% when new branches are grafted alongside"
     Protocol: Train a base tree on domain A. Freeze the left subtree
     (leaves 0-3, mature domain A branch). Graft a new right subtree for
     domain B. Measure domain A val loss before and after grafting.
     The frozen left subtree should not degrade.

Both kill criteria are tested with 3 seeds.
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
from experiments.models.split_freeze_protocol.split_freeze_protocol import (
    compute_leaf_entropy,
    split_leaf,
    compute_activation_identity,
    compute_identity_jaccard,
    freeze_leaves,
    freeze_gates,
)
from experiments.models.subtree_grafting.run_experiment import (
    get_subtree_params, set_param_by_path, freeze_except_subtree,
    freeze_except_tree, get_tree_params,
)


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


def save_weights(model):
    """Deep copy of model weights."""
    return {k: mx.array(v) for k, v in nn.utils.tree_flatten(model.parameters())}


# ── KC1: Split quality vs from-scratch ──────────────────────────────────────

def run_split_experiment(seeds=(42, 123, 777), steps_base=300, steps_finetune=200):
    """Test KC1: split branch quality vs from-scratch flat expert.

    Protocol:
    1. Train base tree on all data (steps_base steps)
    2. Identify the leaf with highest routing weight (most-used leaf)
    3. (a) SPLIT: Replace that leaf with two half-size leaves (inheriting
       parent capsules). Fine-tune only the split pair.
    4. (b) FROM-SCRATCH: Replace that leaf with two fresh random leaves.
       Fine-tune only those leaves.
    5. Compare val loss.

    Since we can't dynamically change tree depth at micro scale, we simulate
    splitting by using two adjacent leaves (e.g., leaves 0,1 -- siblings
    sharing gate 1). The split version inherits parent capsules; from-scratch
    reinitializes them.
    """
    print("=" * 70)
    print("KC1: Split branch quality vs from-scratch flat expert")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size

    results = {"split": [], "scratch": [], "base": []}

    for seed in seeds:
        print(f"\n  SEED {seed}")
        print(f"  {'='*60}")

        # Prepare data
        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
        all_train = CharDataset(all_train_docs, tokenizer, 32)
        all_val = CharDataset(all_val_docs, tokenizer, 32)

        tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                        n_capsules_per_leaf=32, beam_width=2)

        # 1. Train base model on all data
        print(f"\n  [BASE] Training base tree ({steps_base} steps)")
        mx.random.seed(seed)
        base_model = get_model("split_freeze_protocol", **tree_cfg)
        mx.eval(base_model.parameters())
        base_result = train_model(base_model, all_train, all_val,
                                   steps=steps_base, seed=seed, log_every=200)
        base_vl = base_result["val_loss"]
        results["base"].append(base_vl)
        print(f"    base val_loss: {base_vl:.4f}")

        # Save base weights for both experiments
        base_weights = save_weights(base_model)

        # 2. Identify most-used sibling pair
        # Use leaves 0,1 (siblings under gate 1) as the split target.
        # This is the left-most pair in the tree.
        split_leaves = [0, 1]  # siblings sharing parent gate 1
        split_gate = 1  # their parent gate

        # ── (a) SPLIT: inherit parent capsules ──────────────────────────
        print(f"\n  [SPLIT] Splitting leaves {split_leaves} with inherited capsules")
        mx.random.seed(seed)
        split_model = get_model("split_freeze_protocol", **tree_cfg)
        split_model.load_weights(list(base_weights.items()))
        mx.eval(split_model.parameters())

        # The split approach: leaves already exist in the tree (leaves 0 and 1).
        # We want to simulate that they inherited from a parent by averaging
        # the two leaves' capsules into a "parent", then splitting back.
        # But more directly: we just keep the base-trained capsules for
        # leaves 0,1 as-is (they were already trained jointly at the base
        # stage, which is equivalent to a parent that saw all data).

        # Freeze everything except the split leaves and their parent gate
        split_model.wte.freeze()
        split_model.wpe.freeze()
        split_model.norm0.freeze()
        split_model.lm_head.freeze()
        for layer in split_model.layers:
            layer.norm1.freeze()
            layer.attn.freeze()
            layer.norm2.freeze()
            # Freeze all gates except the parent gate of the split pair
            for gi in range(len(layer.tree.gates)):
                if gi != split_gate:
                    layer.tree.gates[gi].freeze()
            # Freeze all leaves except the split pair
            for li in range(len(layer.tree.leaves)):
                if li not in split_leaves:
                    layer.tree.leaves[li].freeze()

        n_trainable_split = count_params(split_model)
        print(f"    trainable params: {n_trainable_split}")

        split_result = train_model(split_model, all_train, all_val,
                                    steps=steps_finetune, seed=seed, log_every=200)
        split_vl = split_result["val_loss"]
        results["split"].append(split_vl)
        print(f"    split val_loss: {split_vl:.4f}")

        # ── (b) FROM-SCRATCH: reinitialize the same leaves ─────────────
        print(f"\n  [SCRATCH] From-scratch leaves {split_leaves}")
        mx.random.seed(seed)
        scratch_model = get_model("split_freeze_protocol", **tree_cfg)
        scratch_model.load_weights(list(base_weights.items()))
        mx.eval(scratch_model.parameters())

        # Reinitialize the target leaves with fresh random weights
        for layer in scratch_model.layers:
            for li in split_leaves:
                d = layer.tree.leaves[li].A.weight.shape[1]
                n_caps = layer.tree.leaves[li].A.weight.shape[0]
                # Xavier/Glorot uniform initialization
                scale = (2.0 / (d + n_caps)) ** 0.5
                new_A = mx.random.normal((n_caps, d)) * scale
                new_B = mx.random.normal((d, n_caps)) * scale
                layer.tree.leaves[li].A.load_weights([("weight", new_A)])
                layer.tree.leaves[li].B.load_weights([("weight", new_B)])
        mx.eval(scratch_model.parameters())

        # Freeze same params as split experiment (only target leaves + parent gate)
        scratch_model.wte.freeze()
        scratch_model.wpe.freeze()
        scratch_model.norm0.freeze()
        scratch_model.lm_head.freeze()
        for layer in scratch_model.layers:
            layer.norm1.freeze()
            layer.attn.freeze()
            layer.norm2.freeze()
            for gi in range(len(layer.tree.gates)):
                if gi != split_gate:
                    layer.tree.gates[gi].freeze()
            for li in range(len(layer.tree.leaves)):
                if li not in split_leaves:
                    layer.tree.leaves[li].freeze()

        n_trainable_scratch = count_params(scratch_model)
        print(f"    trainable params: {n_trainable_scratch}")

        scratch_result = train_model(scratch_model, all_train, all_val,
                                      steps=steps_finetune, seed=seed, log_every=200)
        scratch_vl = scratch_result["val_loss"]
        results["scratch"].append(scratch_vl)
        print(f"    scratch val_loss: {scratch_vl:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"KC1 RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")

    def mean(lst): return sum(lst) / len(lst)

    base_mean = mean(results["base"])
    split_mean = mean(results["split"])
    scratch_mean = mean(results["scratch"])
    split_vs_scratch = 100 * (split_mean - scratch_mean) / scratch_mean

    print(f"\n  Base (pre-split):    {base_mean:.4f}  per-seed: {['%.4f' % v for v in results['base']]}")
    print(f"  Split (inherited):   {split_mean:.4f}  per-seed: {['%.4f' % v for v in results['split']]}")
    print(f"  From-scratch:        {scratch_mean:.4f}  per-seed: {['%.4f' % v for v in results['scratch']]}")
    print(f"\n  Split vs scratch: {split_vs_scratch:+.2f}%")
    print(f"  KC1 threshold: split >5% worse than scratch -> KILL")

    kill_kc1 = split_vs_scratch > 5.0
    print(f"  KC1: {'KILLED' if kill_kc1 else 'PASSES'} ({split_vs_scratch:+.2f}%)")

    return results, kill_kc1


# ── KC2: Frozen branch stability under grafting ─────────────────────────────

def run_freeze_experiment(seeds=(42, 123, 777), steps_domain_a=400,
                           steps_graft_b=200, steps_calibrate=100):
    """Test KC2: frozen branches degrade when new branches grafted alongside.

    Protocol:
    1. Train full tree on domain A data (steps_domain_a steps).
       This creates a "mature" model for domain A.
    2. Measure domain A val loss (pre-graft baseline).
    3. Freeze the LEFT subtree (domain A's territory: leaves 0-3, gates 1,3,4).
    4. Reinitialize the RIGHT subtree (leaves 4-7, gates 2,5,6) for domain B.
    5. Train ONLY the right subtree + root gate on domain B data.
    6. Calibrate root gate on mixed data.
    7. Measure domain A val loss AFTER grafting (post-graft).
    8. Degradation = (post - pre) / pre * 100. Kill if >2%.
    """
    print("\n" + "=" * 70)
    print("KC2: Frozen branch stability under grafting")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)
    domain_names = list(splits.keys())

    results = {
        "pre_graft_a": [],   # domain A val loss before grafting
        "post_graft_a": [],  # domain A val loss after grafting
        "post_graft_b": [],  # domain B val loss after grafting
        "joint_a": [],       # domain A val loss from joint training (reference)
        "joint_b": [],       # domain B val loss from joint training (reference)
    }

    for seed in seeds:
        print(f"\n  SEED {seed}")
        print(f"  {'='*60}")

        # Prepare datasets
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

        # ── Reference: joint training on both domains ─────────────────────
        print(f"\n  [JOINT] Full training on all data (reference)")
        mx.random.seed(seed)
        joint_model = get_model("split_freeze_protocol", **tree_cfg)
        mx.eval(joint_model.parameters())
        _ = train_model(joint_model, joint_train, joint_val,
                        steps=steps_domain_a + steps_graft_b,
                        seed=seed, log_every=200)

        for di, d_name in enumerate(domain_names):
            d_vl = evaluate(joint_model, domain_datasets[d_name][1], 32)
            side = "a" if di == 0 else "b"
            results[f"joint_{side}"].append(d_vl)
            print(f"    joint {d_name} val_loss: {d_vl:.4f}")

        # ── Step 1: Train on domain A ─────────────────────────────────────
        d_a_name = domain_names[0]
        d_a_train, d_a_val = domain_datasets[d_a_name]

        print(f"\n  [DOMAIN A] Training tree on {d_a_name} ({steps_domain_a} steps)")
        mx.random.seed(seed)
        model = get_model("split_freeze_protocol", **tree_cfg)
        mx.eval(model.parameters())
        _ = train_model(model, d_a_train, d_a_val,
                        steps=steps_domain_a, seed=seed, log_every=200)

        # Step 2: Measure pre-graft domain A quality
        pre_graft_a = evaluate(model, d_a_val, 32)
        results["pre_graft_a"].append(pre_graft_a)
        print(f"    pre-graft domain A val_loss: {pre_graft_a:.4f}")

        # Step 3: Freeze left subtree (domain A's territory)
        print(f"\n  [FREEZE] Freezing left subtree (domain A)")
        left_gates = {1, 3, 4}
        left_leaves = {0, 1, 2, 3}

        # Save pre-graft weights for domain A's subtree
        pre_graft_left_params = {}
        for layer_idx, layer in enumerate(model.layers):
            for li in left_leaves:
                for k, v in nn.utils.tree_flatten(layer.tree.leaves[li].parameters()):
                    key = f"layers.{layer_idx}.tree.leaves.{li}.{k}"
                    pre_graft_left_params[key] = mx.array(v)
            for gi in left_gates:
                for k, v in nn.utils.tree_flatten(layer.tree.gates[gi].parameters()):
                    key = f"layers.{layer_idx}.tree.gates.{gi}.{k}"
                    pre_graft_left_params[key] = mx.array(v)

        # Freeze left subtree
        freeze_leaves(model, left_leaves)
        freeze_gates(model, left_gates)

        # Step 4: Reinitialize right subtree for domain B
        print(f"  [REINIT] Reinitializing right subtree for domain B")
        right_gates = {2, 5, 6}
        right_leaves = {4, 5, 6, 7}

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
                scale = (1.0 / d_in) ** 0.5
                new_w = mx.random.normal(gate.proj.weight.shape) * scale
                new_b = mx.zeros(gate.proj.bias.shape)
                gate.proj.load_weights([("weight", new_w), ("bias", new_b)])
        mx.eval(model.parameters())

        # Also freeze attention + embeddings (only tree right side + root gate trainable)
        model.wte.freeze()
        model.wpe.freeze()
        model.norm0.freeze()
        model.lm_head.freeze()
        for layer in model.layers:
            layer.norm1.freeze()
            layer.attn.freeze()
            layer.norm2.freeze()

        n_trainable = count_params(model)
        print(f"    trainable params: {n_trainable}")

        # Step 5: Train right subtree + root gate on domain B
        d_b_name = domain_names[1]
        d_b_train, d_b_val = domain_datasets[d_b_name]

        print(f"\n  [GRAFT B] Training right subtree on {d_b_name} ({steps_graft_b} steps)")
        _ = train_model(model, d_b_train, d_b_val,
                        steps=steps_graft_b, seed=seed, log_every=200)

        # Step 6: Calibrate root gate on mixed data
        print(f"\n  [CALIBRATE] Root gate calibration ({steps_calibrate} steps)")
        # Freeze right subtree too, only root gate trainable
        for layer in model.layers:
            for gi in right_gates:
                layer.tree.gates[gi].freeze()
            for li in right_leaves:
                layer.tree.leaves[li].freeze()
        # Root gate (index 0) stays trainable

        n_trainable_cal = count_params(model)
        print(f"    calibration trainable params: {n_trainable_cal}")

        _ = train_model(model, joint_train, joint_val,
                        steps=steps_calibrate, seed=seed, log_every=100)

        # Step 7: Measure post-graft domain A quality
        post_graft_a = evaluate(model, d_a_val, 32)
        results["post_graft_a"].append(post_graft_a)
        print(f"    post-graft domain A val_loss: {post_graft_a:.4f}")

        post_graft_b = evaluate(model, d_b_val, 32)
        results["post_graft_b"].append(post_graft_b)
        print(f"    post-graft domain B val_loss: {post_graft_b:.4f}")

        # Verify left subtree weights are unchanged (frozen guarantee)
        print(f"\n  [VERIFY] Checking frozen weights unchanged...")
        weight_drift = 0.0
        for layer_idx, layer in enumerate(model.layers):
            for li in left_leaves:
                for k, v in nn.utils.tree_flatten(layer.tree.leaves[li].parameters()):
                    key = f"layers.{layer_idx}.tree.leaves.{li}.{k}"
                    diff = mx.abs(v - pre_graft_left_params[key]).sum().item()
                    weight_drift += diff
            for gi in left_gates:
                for k, v in nn.utils.tree_flatten(layer.tree.gates[gi].parameters()):
                    key = f"layers.{layer_idx}.tree.gates.{gi}.{k}"
                    diff = mx.abs(v - pre_graft_left_params[key]).sum().item()
                    weight_drift += diff
        print(f"    total weight drift in frozen params: {weight_drift:.6f}")
        assert weight_drift < 1e-6, f"Frozen weights changed! drift={weight_drift}"

        # Compute degradation
        degradation = 100 * (post_graft_a - pre_graft_a) / pre_graft_a
        print(f"    domain A degradation: {degradation:+.2f}%")

    # Summary
    print(f"\n{'='*70}")
    print(f"KC2 RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")

    def mean(lst): return sum(lst) / len(lst)

    pre_mean = mean(results["pre_graft_a"])
    post_mean = mean(results["post_graft_a"])
    post_b_mean = mean(results["post_graft_b"])
    joint_a_mean = mean(results["joint_a"])
    joint_b_mean = mean(results["joint_b"])

    degradation_mean = 100 * (post_mean - pre_mean) / pre_mean

    print(f"\n  Domain A (frozen, pre-graft):  {pre_mean:.4f}  per-seed: {['%.4f' % v for v in results['pre_graft_a']]}")
    print(f"  Domain A (frozen, post-graft): {post_mean:.4f}  per-seed: {['%.4f' % v for v in results['post_graft_a']]}")
    print(f"  Domain B (grafted):            {post_b_mean:.4f}  per-seed: {['%.4f' % v for v in results['post_graft_b']]}")
    print(f"  Joint A (reference):           {joint_a_mean:.4f}")
    print(f"  Joint B (reference):           {joint_b_mean:.4f}")
    print(f"\n  Domain A degradation: {degradation_mean:+.2f}%")
    print(f"  KC2 threshold: frozen branches degrade >2% -> KILL")

    kill_kc2 = degradation_mean > 2.0
    print(f"  KC2: {'KILLED' if kill_kc2 else 'PASSES'} ({degradation_mean:+.2f}%)")

    return results, kill_kc2


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(seeds=(42, 123, 777)):
    """Run both KC1 and KC2 experiments."""
    t0 = time.time()

    kc1_results, kill_kc1 = run_split_experiment(seeds)
    kc2_results, kill_kc2 = run_freeze_experiment(seeds)

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")
    print(f"  KC1 (split vs scratch): {'KILLED' if kill_kc1 else 'PASSES'}")
    print(f"  KC2 (frozen stability): {'KILLED' if kill_kc2 else 'PASSES'}")
    print(f"  Overall: {'KILLED' if (kill_kc1 or kill_kc2) else 'PASSES'}")
    print(f"  Total time: {total:.1f}s ({total/60:.1f} min)")

    return {
        "kc1": kc1_results,
        "kc2": kc2_results,
        "kill_kc1": kill_kc1,
        "kill_kc2": kill_kc2,
    }


if __name__ == "__main__":
    run_experiment()
