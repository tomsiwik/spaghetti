"""Run entropy-adaptive routing experiment.

Compares:
  1. capsule_moe (fixed k=2 baseline)
  2. entropy_adaptive_router (variable k based on routing entropy)
  3. capsule_moe with k=1 (to confirm catastrophic failure)

Kill criteria:
  1. Variable-k routing worse than fixed k=2 at same average compute
  2. Entropy-based k-selection doesn't reduce average k below 1.8
"""

import sys
import os
import time
import random
import json
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_and_eval(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                   seed=42, log_every=100):
    """Train and evaluate, return val loss + training stats."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        return loss + model.aux_loss()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            extra = ""
            if hasattr(model, "entropy_stats"):
                stats = model.entropy_stats()
                avg_k = model.avg_k()
                mean_h = sum(s.get("mean_entropy", 0) for s in stats) / len(stats)
                frac_k1 = sum(s.get("frac_k1", 0) for s in stats) / len(stats)
                extra = f" | avg_k={avg_k:.3f} | H={mean_h:.3f} | %k1={frac_k1:.1%}"
            print(f"  step {step:4d}/{steps} | loss {loss_val:.4f}{extra}")

    elapsed = time.time() - t0

    # Evaluate
    eval_rng = random.Random(0)
    total_val = 0.0
    n_batches = 20
    for _ in range(n_batches):
        inputs, targets = val_ds.get_batch(batch_size, eval_rng)
        logits = model(inputs)
        B, T, V = logits.shape
        val_loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        mx.eval(val_loss)
        total_val += val_loss.item()
    val_loss = total_val / n_batches

    return {
        "val_loss": val_loss,
        "final_train_loss": losses[-1],
        "elapsed": elapsed,
        "losses": losses,
    }


def run_experiment(seeds=(42, 123, 7), steps=500):
    """Full experiment: compare fixed k=2 vs entropy-adaptive."""

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    block_size = 32

    results = {"fixed_k2": [], "entropy_adaptive": [], "fixed_k1": []}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tokenizer, block_size)
        val_ds = CharDataset(docs_val, tokenizer, block_size)

        # 1. Fixed k=2 baseline (capsule_moe)
        print(f"\n--- Baseline: capsule_moe (fixed k=2) ---")
        model_k2 = get_model("capsule_moe", vocab_size=tokenizer.vocab_size,
                             block_size=block_size, n_groups=4,
                             n_capsules_per_group=64, top_k_groups=2)
        mx.eval(model_k2.parameters())
        print(f"  params: {count_params(model_k2):,}")
        r_k2 = train_and_eval(model_k2, train_ds, val_ds, steps=steps,
                              seed=seed)
        results["fixed_k2"].append(r_k2)
        print(f"  val_loss = {r_k2['val_loss']:.4f}")

        # 2. Entropy-adaptive k
        print(f"\n--- Entropy-adaptive router ---")
        model_ea = get_model("entropy_adaptive_router",
                             vocab_size=tokenizer.vocab_size,
                             block_size=block_size, n_groups=4,
                             n_capsules_per_group=64, tau_h=0.5,
                             learn_threshold=True)
        mx.eval(model_ea.parameters())
        print(f"  params: {count_params(model_ea):,}")
        r_ea = train_and_eval(model_ea, train_ds, val_ds, steps=steps,
                              seed=seed)
        results["entropy_adaptive"].append(r_ea)
        print(f"  val_loss = {r_ea['val_loss']:.4f}")

        # Log final entropy stats
        print("  Final entropy stats:")
        for i, s in enumerate(model_ea.entropy_stats()):
            print(f"    Layer {i}: H={s['mean_entropy']:.4f}, "
                  f"tau={s['tau_h']:.4f}, avg_k={s['avg_k']:.3f}, "
                  f"frac_k1={s['frac_k1']:.1%}")

        # 3. Fixed k=1 (to confirm catastrophic failure)
        print(f"\n--- Control: capsule_moe (fixed k=1) ---")
        model_k1 = get_model("capsule_moe", vocab_size=tokenizer.vocab_size,
                             block_size=block_size, n_groups=4,
                             n_capsules_per_group=64, top_k_groups=1)
        mx.eval(model_k1.parameters())
        r_k1 = train_and_eval(model_k1, train_ds, val_ds, steps=steps,
                              seed=seed)
        results["fixed_k1"].append(r_k1)
        print(f"  val_loss = {r_k1['val_loss']:.4f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (3-seed mean)")
    print(f"{'='*60}")

    for name, runs in results.items():
        mean_val = sum(r["val_loss"] for r in runs) / len(runs)
        print(f"  {name:25s}: val_loss = {mean_val:.4f}")

    mean_k2 = sum(r["val_loss"] for r in results["fixed_k2"]) / len(results["fixed_k2"])
    mean_ea = sum(r["val_loss"] for r in results["entropy_adaptive"]) / len(results["entropy_adaptive"])
    mean_k1 = sum(r["val_loss"] for r in results["fixed_k1"]) / len(results["fixed_k1"])

    gap_ea = (mean_ea - mean_k2) / mean_k2 * 100
    gap_k1 = (mean_k1 - mean_k2) / mean_k2 * 100

    print(f"\n  Entropy-adaptive vs fixed k=2: {gap_ea:+.2f}%")
    print(f"  Fixed k=1 vs fixed k=2:       {gap_k1:+.2f}%")

    # Kill criteria evaluation
    print(f"\n{'='*60}")
    print("KILL CRITERIA")
    print(f"{'='*60}")

    # KC1: variable-k worse than fixed k=2
    if gap_ea > 0:
        print(f"  KC1: FAIL - entropy-adaptive is {gap_ea:+.2f}% worse than k=2")
    else:
        print(f"  KC1: PASS - entropy-adaptive is {gap_ea:+.2f}% vs k=2")

    # KC2: avg_k doesn't drop below 1.8
    # We need to get this from the last run's model
    # For now, just note that we logged it
    print(f"  KC2: Check avg_k values in per-seed logs above")

    return results


if __name__ == "__main__":
    results = run_experiment()
