"""Experiment: Hierarchical Capsule Tree vs Flat Capsule MoE.

Compares:
1. Single-domain quality: tree vs flat at matched params
2. Composition quality: shared-base composition for both

Kill criteria:
- Tree-routed composition degrades >5% vs flat softmax composition
- Tree routing quality (perplexity) worse than flat softmax at same active params

We match total capsule count: 8 groups * 32 caps/group = 256 total for both.
Flat: G=8, k=2, 32 capsules/group (softmax router)
Tree: depth=3, 8 leaves, 32 capsules/leaf, beam=2 (binary gates)
"""

import sys
import time
import random
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Ensure project root is importable
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3, seed=42, log_every=100):
    """Train a model and return train/val loss."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"losses": losses, "val_loss": val_loss, "elapsed_s": time.time() - t0}


def run_single_domain_comparison(seeds=(42, 123, 777), steps=500):
    """Compare single-domain quality: tree vs flat at matched params."""
    print("=" * 70)
    print("EXPERIMENT 1: Single-Domain Quality Comparison")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size

    results = {"flat": [], "tree": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tokenizer, 32)
        val_ds = CharDataset(docs_val, tokenizer, 32)

        # Flat capsule MoE: G=8, k=2, 32 caps/group = 256 total
        print(f"\n  [flat capsule_moe G=8, k=2]")
        mx.random.seed(seed)
        flat_model = get_model("capsule_moe", vocab_size=vs, block_size=32,
                               n_groups=8, n_capsules_per_group=32, top_k_groups=2)
        mx.eval(flat_model.parameters())
        n_flat = count_params(flat_model)
        print(f"    params: {n_flat:,}")
        flat_result = train_model(flat_model, train_ds, val_ds, steps=steps, seed=seed)
        results["flat"].append(flat_result["val_loss"])
        print(f"    val_loss: {flat_result['val_loss']:.4f}")

        # Tree: depth=3, beam=2, 32 caps/leaf = 256 total
        print(f"\n  [hierarchical_tree depth=3, beam=2]")
        mx.random.seed(seed)
        tree_model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                               tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
        mx.eval(tree_model.parameters())
        n_tree = count_params(tree_model)
        print(f"    params: {n_tree:,}")
        tree_result = train_model(tree_model, train_ds, val_ds, steps=steps, seed=seed)
        results["tree"].append(tree_result["val_loss"])
        print(f"    val_loss: {tree_result['val_loss']:.4f}")

    # Summary
    flat_mean = sum(results["flat"]) / len(results["flat"])
    tree_mean = sum(results["tree"]) / len(results["tree"])
    delta_pct = 100 * (tree_mean - flat_mean) / flat_mean

    print(f"\n{'='*70}")
    print(f"SINGLE-DOMAIN SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"  Flat (G=8, k=2):  mean val_loss = {flat_mean:.4f}  params = {n_flat:,}")
    print(f"  Tree (D=3, B=2):  mean val_loss = {tree_mean:.4f}  params = {n_tree:,}")
    print(f"  Delta: {delta_pct:+.2f}%")
    print(f"  Per-seed: {['%.4f' % v for v in results['flat']]} vs {['%.4f' % v for v in results['tree']]}")

    kill_2 = delta_pct > 0  # tree worse than flat = kill criterion 2 triggered
    print(f"\n  Kill criterion 2 (tree quality worse than flat): {'TRIGGERED' if kill_2 else 'PASSES'} ({delta_pct:+.2f}%)")

    return results, {"flat_mean": flat_mean, "tree_mean": tree_mean,
                      "delta_pct": delta_pct, "n_flat": n_flat, "n_tree": n_tree}


def run_composition_comparison(seeds=(42, 123, 777), steps_pretrain=300,
                                steps_finetune=200, steps_calibrate=100):
    """Compare composition quality: shared-base protocol for tree vs flat.

    Protocol (same for both):
    1. Pretrain base model on all data
    2. Fine-tune capsule groups on each domain (attention frozen)
    3. Compose by concatenating domain-specific groups
    4. Calibrate router on mixed data
    5. Evaluate
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Composition Quality Comparison")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)

    results = {"flat_joint": [], "flat_composed": [],
               "tree_joint": [], "tree_composed": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Prepare domain datasets
        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )

        # Also prepare joint dataset (all data)
        all_train, all_val = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train, tokenizer, 32)
        joint_val = CharDataset(all_val, tokenizer, 32)

        for model_type in ["flat", "tree"]:
            print(f"\n  === {model_type} ===")

            # 1. Joint training baseline
            print(f"  [joint baseline]")
            mx.random.seed(seed)
            if model_type == "flat":
                joint_model = get_model("capsule_moe", vocab_size=vs, block_size=32,
                                        n_groups=8, n_capsules_per_group=32, top_k_groups=2)
            else:
                joint_model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                                        tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
            mx.eval(joint_model.parameters())
            joint_result = train_model(joint_model, joint_train, joint_val,
                                        steps=steps_pretrain + steps_finetune,
                                        seed=seed, log_every=200)
            results[f"{model_type}_joint"].append(joint_result["val_loss"])
            print(f"    joint val_loss: {joint_result['val_loss']:.4f}")

            # 2. Shared-base composition
            print(f"  [shared-base composition]")

            # 2a. Pretrain base on all data
            mx.random.seed(seed)
            if model_type == "flat":
                base_model = get_model("capsule_moe", vocab_size=vs, block_size=32,
                                        n_groups=8, n_capsules_per_group=32, top_k_groups=2)
            else:
                base_model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                                        tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
            mx.eval(base_model.parameters())
            _ = train_model(base_model, joint_train, joint_val,
                           steps=steps_pretrain, seed=seed, log_every=200)

            # Save base weights
            base_weights = {k: mx.array(v) for k, v in
                           nn.utils.tree_flatten(base_model.parameters())}

            # 2b. Fine-tune per domain (freeze attention+embeddings, train only MLP/tree)
            domain_expert_weights = {}
            for d_name, (d_train, d_val) in domain_datasets.items():
                print(f"    fine-tune on {d_name}...")
                mx.random.seed(seed)
                if model_type == "flat":
                    d_model = get_model("capsule_moe", vocab_size=vs, block_size=32,
                                        n_groups=8, n_capsules_per_group=32, top_k_groups=2)
                else:
                    d_model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                                        tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
                # Load base weights
                d_model.load_weights(list(base_weights.items()))
                mx.eval(d_model.parameters())

                # Freeze everything except capsule/tree params
                d_model.wte.freeze()
                d_model.wpe.freeze()
                d_model.norm0.freeze()
                d_model.lm_head.freeze()
                for layer in d_model.layers:
                    layer.norm1.freeze()
                    layer.attn.freeze()
                    layer.norm2.freeze()

                _ = train_model(d_model, d_train, d_val,
                               steps=steps_finetune, seed=seed, log_every=200)

                # Save domain-specific weights
                if model_type == "flat":
                    key = "capsule_pool"
                else:
                    key = "tree"
                domain_expert_weights[d_name] = {}
                for layer_idx, layer in enumerate(d_model.layers):
                    expert_module = getattr(layer, key)
                    for k, v in nn.utils.tree_flatten(expert_module.parameters()):
                        full_key = f"layers.{layer_idx}.{key}.{k}"
                        domain_expert_weights[d_name][full_key] = mx.array(v)

            # 2c. Compose: average domain expert weights (weight averaging)
            # This is the zero-shot composition approach
            print(f"    composing via weight averaging...")
            composed_model = get_model(
                "capsule_moe" if model_type == "flat" else "hierarchical_tree",
                vocab_size=vs, block_size=32,
                **({"n_groups": 8, "n_capsules_per_group": 32, "top_k_groups": 2}
                   if model_type == "flat"
                   else {"tree_depth": 3, "n_capsules_per_leaf": 32, "beam_width": 2})
            )
            composed_model.load_weights(list(base_weights.items()))

            # Weight-average the domain expert modules
            domains = list(domain_expert_weights.keys())
            n_domains = len(domains)
            for key in domain_expert_weights[domains[0]]:
                avg_w = sum(domain_expert_weights[d][key] for d in domains) / n_domains
                # Set in composed model
                parts = key.split(".")
                obj = composed_model
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                setattr(obj, parts[-1], avg_w)

            mx.eval(composed_model.parameters())

            # 2d. Calibrate: brief training on mixed data (router/gates only)
            # Freeze everything except routing params
            composed_model.wte.freeze()
            composed_model.wpe.freeze()
            composed_model.norm0.freeze()
            composed_model.lm_head.freeze()
            for layer in composed_model.layers:
                layer.norm1.freeze()
                layer.attn.freeze()
                layer.norm2.freeze()
                if model_type == "flat":
                    # Only train router
                    for g in layer.capsule_pool.groups:
                        g.freeze()
                else:
                    # Only train gates
                    for leaf in layer.tree.leaves:
                        leaf.freeze()

            _ = train_model(composed_model, joint_train, joint_val,
                           steps=steps_calibrate, seed=seed, log_every=100)

            # Unfreeze for eval
            composed_val = evaluate(composed_model, joint_val, 32)
            results[f"{model_type}_composed"].append(composed_val)
            print(f"    composed val_loss: {composed_val:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"COMPOSITION SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")
    for mt in ["flat", "tree"]:
        joint_mean = sum(results[f"{mt}_joint"]) / len(results[f"{mt}_joint"])
        comp_mean = sum(results[f"{mt}_composed"]) / len(results[f"{mt}_composed"])
        gap = 100 * (comp_mean - joint_mean) / joint_mean
        print(f"  {mt:5s}: joint={joint_mean:.4f}  composed={comp_mean:.4f}  gap={gap:+.2f}%")
        print(f"         per-seed joint:    {['%.4f' % v for v in results[f'{mt}_joint']]}")
        print(f"         per-seed composed: {['%.4f' % v for v in results[f'{mt}_composed']]}")

    flat_gap = 100 * (sum(results["flat_composed"])/len(seeds) -
                       sum(results["flat_joint"])/len(seeds)) / (sum(results["flat_joint"])/len(seeds))
    tree_gap = 100 * (sum(results["tree_composed"])/len(seeds) -
                       sum(results["tree_joint"])/len(seeds)) / (sum(results["tree_joint"])/len(seeds))
    comp_delta = tree_gap - flat_gap

    print(f"\n  Flat composition gap: {flat_gap:+.2f}%")
    print(f"  Tree composition gap: {tree_gap:+.2f}%")
    print(f"  Tree vs Flat composition delta: {comp_delta:+.2f}pp")

    kill_1 = tree_gap > flat_gap + 5.0
    print(f"\n  Kill criterion 1 (tree composition >5% worse than flat): {'TRIGGERED' if kill_1 else 'PASSES'}")

    return results


def main():
    t0 = time.time()

    # Experiment 1: Single-domain quality
    single_results, single_summary = run_single_domain_comparison(
        seeds=(42, 123, 777), steps=500
    )

    # Experiment 2: Composition quality
    comp_results = run_composition_comparison(
        seeds=(42, 123, 777),
        steps_pretrain=300, steps_finetune=200, steps_calibrate=100
    )

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
