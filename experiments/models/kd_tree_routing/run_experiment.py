"""Run the KD-tree routing experiment.

Compares:
1. kd_tree_routing (this model) — KD-tree split with temperature annealing
2. hierarchical_tree (parent) — sigmoid binary gates
3. capsule_moe (grandparent) — flat softmax routing

Protocol:
- Single-domain quality: 500 steps, 3 seeds
- Composition quality: 300 pretrain + 200 finetune + 100 calibrate, 3 seeds
- Routing diagnostics after training
"""

import sys
import time
import random
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Ensure imports work
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import evaluate


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_with_temperature(model, dataset, val_dataset, steps=500,
                           batch_size=32, lr=3e-3, seed=42, log_every=50):
    """Train with temperature annealing for KD-tree models."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        # Temperature annealing for KD-tree models
        if hasattr(model, 'step_temperature'):
            model.step_temperature(step, steps)

        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            temp_str = ""
            if hasattr(model, 'layers') and hasattr(model.layers[0], 'tree'):
                tree = model.layers[0].tree
                if hasattr(tree, 'temperature'):
                    temp_str = f" | temp={tree.temperature:.2f}"
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{steps} | loss {loss_val:.4f}{temp_str}")

    val_loss = evaluate(model, val_dataset, batch_size) if val_dataset else None
    return {"val_loss": val_loss, "losses": losses, "elapsed": time.time() - t0}


def _ntp_loss(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )
    return loss + model.aux_loss()


def run_single_domain(model_name, seed=42, steps=500, **model_kwargs):
    """Train and evaluate on all names data."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    train_ds = CharDataset(docs_train, tokenizer, 32)
    val_ds = CharDataset(docs_val, tokenizer, 32)

    kwargs = dict(vocab_size=tokenizer.vocab_size, block_size=32, **model_kwargs)
    model = get_model(model_name, **kwargs)
    mx.eval(model.parameters())
    n_params = count_params(model)

    print(f"\n=== {model_name} (seed={seed}, {n_params:,} params) ===")
    result = train_with_temperature(model, train_ds, val_ds, steps=steps, seed=seed)
    print(f"  val_loss = {result['val_loss']:.4f}")

    # Get routing diagnostics if available
    diag = None
    if hasattr(model, 'get_routing_diagnostics'):
        # Run a forward pass to populate diagnostics
        inputs, _ = val_ds.get_batch(32, random.Random(0))
        _ = model(inputs)
        mx.eval(model.parameters())
        diag = model.get_routing_diagnostics()

    return {
        "model": model_name,
        "seed": seed,
        "n_params": n_params,
        "val_loss": result["val_loss"],
        "elapsed": result["elapsed"],
        "diagnostics": diag,
    }


def run_composition(model_name, seed=42, pretrain_steps=300,
                    finetune_steps=200, calibrate_steps=100, **model_kwargs):
    """Composition experiment: pretrain -> finetune per domain -> weight average -> calibrate."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)  # a-m, n-z

    # Prepare datasets
    all_train, all_val = train_val_split(docs, seed=seed)
    all_train_ds = CharDataset(all_train, tokenizer, 32)
    all_val_ds = CharDataset(all_val, tokenizer, 32)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, 32),
            CharDataset(d_val, tokenizer, 32),
        )

    kwargs = dict(vocab_size=tokenizer.vocab_size, block_size=32, **model_kwargs)

    # --- Joint training baseline ---
    print(f"\n=== {model_name} JOINT (seed={seed}) ===")
    joint_model = get_model(model_name, **kwargs)
    mx.eval(joint_model.parameters())
    joint_result = train_with_temperature(
        joint_model, all_train_ds, all_val_ds,
        steps=pretrain_steps + finetune_steps, seed=seed
    )
    joint_loss = joint_result["val_loss"]
    print(f"  joint val_loss = {joint_loss:.4f}")

    # --- Composition: pretrain + finetune per domain + weight average ---
    print(f"\n=== {model_name} PRETRAIN (seed={seed}) ===")
    base_model = get_model(model_name, **kwargs)
    mx.eval(base_model.parameters())
    _ = train_with_temperature(
        base_model, all_train_ds, None,
        steps=pretrain_steps, seed=seed
    )

    # Save base weights
    base_weights = {k: mx.array(v) for k, v in
                    nn.utils.tree_flatten(base_model.parameters())}

    # Fine-tune per domain
    domain_deltas = {}
    for d_name, (d_train, d_val) in domain_datasets.items():
        print(f"\n=== {model_name} FINETUNE {d_name} (seed={seed}) ===")
        ft_model = get_model(model_name, **kwargs)
        # Load base weights
        ft_flat = dict(nn.utils.tree_flatten(ft_model.parameters()))
        for k in base_weights:
            if k in ft_flat:
                ft_flat[k] = base_weights[k] * 1
        ft_model.load_weights(list(ft_flat.items()))
        mx.eval(ft_model.parameters())

        _ = train_with_temperature(
            ft_model, d_train, d_val,
            steps=finetune_steps, seed=seed
        )

        # Compute delta
        ft_weights = dict(nn.utils.tree_flatten(ft_model.parameters()))
        delta = {}
        for k in base_weights:
            if k in ft_weights:
                delta[k] = ft_weights[k] - base_weights[k]
        domain_deltas[d_name] = delta

    # Weight average deltas
    domains = list(domain_deltas.keys())
    avg_delta = {}
    for k in domain_deltas[domains[0]]:
        avg_delta[k] = sum(domain_deltas[d][k] for d in domains) / len(domains)

    # Apply averaged delta to base
    composed_model = get_model(model_name, **kwargs)
    composed_flat = dict(nn.utils.tree_flatten(composed_model.parameters()))
    for k in base_weights:
        if k in composed_flat and k in avg_delta:
            composed_flat[k] = base_weights[k] + avg_delta[k]
        elif k in composed_flat:
            composed_flat[k] = base_weights[k] * 1
    composed_model.load_weights(list(composed_flat.items()))
    mx.eval(composed_model.parameters())

    # Evaluate before calibration
    pre_cal_loss = evaluate(composed_model, all_val_ds, 32)
    print(f"  pre-calibration val_loss = {pre_cal_loss:.4f}")

    # Calibrate (train on mixed data)
    print(f"\n=== {model_name} CALIBRATE (seed={seed}) ===")
    cal_result = train_with_temperature(
        composed_model, all_train_ds, all_val_ds,
        steps=calibrate_steps, seed=seed
    )
    composed_loss = cal_result["val_loss"]
    print(f"  composed val_loss = {composed_loss:.4f}")

    gap = (composed_loss - joint_loss) / joint_loss * 100
    print(f"  gap = {gap:+.2f}%")

    return {
        "model": model_name,
        "seed": seed,
        "joint_loss": joint_loss,
        "composed_loss": composed_loss,
        "gap_pct": gap,
    }


def main():
    seeds = [42, 123, 7]
    models = ["kd_tree_routing", "hierarchical_tree"]

    print("=" * 70)
    print("EXPERIMENT: KD-Tree Routing vs Hierarchical Tree vs Flat MoE")
    print("=" * 70)

    # --- Single domain quality ---
    print("\n" + "=" * 70)
    print("PHASE 1: Single-Domain Quality (500 steps)")
    print("=" * 70)

    single_results = {}
    for model_name in models:
        single_results[model_name] = []
        for seed in seeds:
            r = run_single_domain(model_name, seed=seed, steps=500)
            single_results[model_name].append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("SINGLE-DOMAIN RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} | {'Params':>8} | {'Mean Val Loss':>13} | Per-seed losses")
    print("-" * 85)
    for model_name in models:
        results = single_results[model_name]
        losses = [r["val_loss"] for r in results]
        mean_loss = sum(losses) / len(losses)
        n_params = results[0]["n_params"]
        loss_str = ", ".join(f"{l:.4f}" for l in losses)
        print(f"{model_name:<25} | {n_params:>8,} | {mean_loss:>13.4f} | {loss_str}")

    # Print diagnostics for KD-tree
    print("\n" + "=" * 70)
    print("KD-TREE ROUTING DIAGNOSTICS (seed=42)")
    print("=" * 70)
    kd_result = single_results["kd_tree_routing"][0]
    if kd_result["diagnostics"]:
        for layer_name, layer_diag in kd_result["diagnostics"].items():
            print(f"\n{layer_name}:")
            print(f"  Temperature: {layer_diag['temperature']:.2f}")
            if "normalized_entropy" in layer_diag:
                print(f"  Normalized entropy: {layer_diag['normalized_entropy']:.3f}")
                print(f"  Mean max leaf prob: {layer_diag['mean_max_leaf_prob']:.3f}")
                print(f"  Leaf usage: {[f'{u:.3f}' for u in layer_diag['leaf_usage']]}")
            if "split_directions" in layer_diag:
                print(f"  Split directions:")
                for i, sd in enumerate(layer_diag["split_directions"]):
                    print(f"    node {i}: top_dim={sd['top_dim']}, "
                          f"concentration={sd['concentration']:.3f}")

    # --- Composition quality ---
    print("\n" + "=" * 70)
    print("PHASE 2: Composition Quality (300+200+100 steps)")
    print("=" * 70)

    comp_results = {}
    for model_name in models:
        comp_results[model_name] = []
        for seed in seeds:
            r = run_composition(model_name, seed=seed)
            comp_results[model_name].append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("COMPOSITION RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} | {'Mean Joint':>10} | {'Mean Composed':>13} | {'Mean Gap':>10}")
    print("-" * 70)
    for model_name in models:
        results = comp_results[model_name]
        mean_joint = sum(r["joint_loss"] for r in results) / len(results)
        mean_comp = sum(r["composed_loss"] for r in results) / len(results)
        mean_gap = sum(r["gap_pct"] for r in results) / len(results)
        print(f"{model_name:<25} | {mean_joint:>10.4f} | {mean_comp:>13.4f} | {mean_gap:>+10.2f}%")

    # Save all results
    all_results = {
        "single_domain": {
            model: [
                {k: v for k, v in r.items() if k != "diagnostics"}
                for r in results
            ]
            for model, results in single_results.items()
        },
        "composition": comp_results,
    }

    with open("/Users/tom/Code/tomsiwik/llm/experiments/models/kd_tree_routing/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to experiments/models/kd_tree_routing/results.json")


if __name__ == "__main__":
    main()
