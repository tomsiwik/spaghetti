"""Diagnostics: tree routing behavior, gate sharpness, leaf utilization."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time
import math

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import ntp_loss
from experiments.models import get_model


def analyze_routing(model, dataset, n_batches=20, batch_size=32):
    """Analyze tree routing patterns after training."""
    rng = random.Random(0)

    all_leaf_probs = []

    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        _ = model(inputs)

        for layer in model.layers:
            lp = layer.tree._leaf_probs  # (B, T, n_leaves)
            mx.eval(lp)
            all_leaf_probs.append(lp)

    # Stack all leaf probs
    all_lp = mx.concatenate(all_leaf_probs, axis=0)  # (total, T, n_leaves)
    mx.eval(all_lp)

    # Leaf utilization: mean probability per leaf
    mean_leaf_prob = mx.mean(all_lp, axis=(0, 1))  # (n_leaves,)
    mx.eval(mean_leaf_prob)

    # Entropy of leaf distribution (per token, then averaged)
    eps = 1e-8
    entropy = -mx.sum(all_lp * mx.log(all_lp + eps), axis=-1)  # (total, T)
    mean_entropy = mx.mean(entropy)
    max_entropy = math.log(all_lp.shape[-1])  # log(n_leaves) = uniform
    mx.eval(mean_entropy)

    # Top-1 leaf selection frequency
    top1_leaves = mx.argmax(all_lp, axis=-1)  # (total, T)
    mx.eval(top1_leaves)
    n_leaves = all_lp.shape[-1]
    leaf_counts = [0] * n_leaves
    flat_top1 = top1_leaves.reshape(-1).tolist()
    for l in flat_top1:
        leaf_counts[l] += 1
    total = sum(leaf_counts)
    leaf_freqs = [c / total for c in leaf_counts]

    return {
        "mean_leaf_prob": mean_leaf_prob.tolist(),
        "mean_entropy": mean_entropy.item(),
        "max_entropy": max_entropy,
        "normalized_entropy": mean_entropy.item() / max_entropy,
        "top1_leaf_freq": leaf_freqs,
    }


def main():
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tokenizer, 32)
    val_ds = CharDataset(docs_val, tokenizer, 32)

    # Train tree model
    print("Training hierarchical tree model (500 steps)...")
    mx.random.seed(42)
    model = get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                      tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    rng = random.Random(42)
    optimizer = optim.Adam(learning_rate=3e-3)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)
    for step in range(1, 501):
        inputs, targets = train_ds.get_batch(32, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step % 100 == 0:
            print(f"  step {step}/500 | loss {loss.item():.4f}")

    # Analyze routing
    print("\nRouting diagnostics:")
    diag = analyze_routing(model, val_ds)

    print(f"\n  Mean leaf probabilities: {['%.3f' % p for p in diag['mean_leaf_prob']]}")
    print(f"  Ideal uniform: {1/8:.3f}")
    print(f"  Mean routing entropy: {diag['mean_entropy']:.3f} / {diag['max_entropy']:.3f} (max)")
    print(f"  Normalized entropy: {diag['normalized_entropy']:.3f} (1.0 = uniform, 0.0 = deterministic)")
    print(f"  Top-1 leaf selection frequency: {['%.3f' % f for f in diag['top1_leaf_freq']]}")

    # Check gate sharpness per layer
    print("\n  Per-layer gate analysis:")
    rng = random.Random(0)
    inputs, targets = val_ds.get_batch(32, rng)
    _ = model(inputs)

    for layer_idx, layer in enumerate(model.layers):
        tree = layer.tree
        x = layer.norm2(layer.attn(layer.norm1(mx.random.normal((1, 8, 64)))))
        # Just use the stored leaf probs
        lp = tree._leaf_probs  # (B, T, 8)
        mx.eval(lp)

        # Compute how "sharp" the leaf distribution is
        entropy = -mx.sum(lp * mx.log(lp + 1e-8), axis=-1)
        mean_ent = mx.mean(entropy).item()
        max_prob = mx.max(lp, axis=-1)
        mean_max = mx.mean(max_prob).item()

        print(f"    Layer {layer_idx}: mean_entropy={mean_ent:.3f}, mean_max_prob={mean_max:.3f}")

    # Also measure: how many unique leaves are typically in beam-2 selection?
    print("\n  Beam-2 selection diversity:")
    _ = model(inputs)
    for layer_idx, layer in enumerate(model.layers):
        lp = layer.tree._leaf_probs
        mx.eval(lp)
        top2_vals = mx.topk(lp, 2, axis=-1)
        threshold = mx.min(top2_vals, axis=-1, keepdims=True)
        selected = (lp >= threshold).astype(mx.float32)
        n_selected = mx.sum(selected, axis=-1)
        mean_selected = mx.mean(n_selected).item()
        print(f"    Layer {layer_idx}: mean leaves selected = {mean_selected:.2f}")


if __name__ == "__main__":
    main()
