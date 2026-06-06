"""Experiment: Consistent hash routing for incremental expert add/remove.

Tests two kill criteria:
1. >5% quality degradation when adding expert without recalibration
2. Adding one expert displaces >30% of existing routing decisions

Protocol:
1. Train consistent-hash model with N=8 experts (500 steps)
2. Measure baseline val loss
3. Add 9th expert (random init, no recalibration)
4. Measure val loss after adding expert
5. Measure routing displacement (fraction of tokens routed differently)
6. Compare against softmax baseline with same protocol
7. Repeat for 3 seeds
"""

import json
import time
import random
import struct
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import train, evaluate
from experiments.models.consistent_hash_routing.consistent_hash_routing import (
    ConsistentHashRoutingGPT,
)
from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def measure_displacement(model, tokens, n_groups_before, n_groups_after):
    """Measure what fraction of tokens change their primary expert assignment.

    For consistent hash model: add expert to ring, compare assignments.
    """
    B, T = tokens.shape

    # Get assignments before adding expert
    # Forward pass to populate hidden states
    pos = mx.arange(T)
    x = model.wte(tokens) + model.wpe(pos)
    x = model.norm0(x)

    total_displaced = 0
    total_tokens = 0

    for li, layer in enumerate(model.layers):
        h = x + layer.attn(layer.norm1(x))
        h_normed = layer.norm2(h)

        # Get routing before
        pool = layer.capsule_pool
        weights_before = pool.router(h_normed, n_groups_override=n_groups_before)
        mx.eval(weights_before)
        assign_before = mx.argmax(weights_before, axis=-1)

        # Add expert to this layer's ring
        pool.router.add_expert(n_groups_before)

        # Get routing after
        weights_after = pool.router(h_normed, n_groups_override=n_groups_after)
        mx.eval(weights_after)
        assign_after = mx.argmax(weights_after, axis=-1)

        # Remove the expert we just added (reset for next measurement)
        pool.router.remove_expert(n_groups_before)
        pool.router.n_groups = n_groups_before

        # Count displaced
        displaced = mx.sum(assign_before != assign_after).item()
        total_displaced += displaced
        total_tokens += B * T

        # Advance hidden state
        x = layer(x)

    return total_displaced / total_tokens if total_tokens > 0 else 0


def measure_softmax_displacement(model, tokens, new_n_groups):
    """Measure displacement for softmax router when adding an expert.

    For softmax: extend router weight matrix, add new expert group.
    The router weight for the new expert is random, which disrupts existing routing.
    """
    B, T = tokens.shape
    pos = mx.arange(T)
    x = model.wte(tokens) + model.wpe(pos)
    x = model.norm0(x)

    total_displaced = 0
    total_tokens = 0

    for li, layer in enumerate(model.layers):
        h = x + layer.attn(layer.norm1(x))
        h_normed = layer.norm2(h)

        pool = layer.capsule_pool
        n_before = pool.n_groups

        # Get routing before
        scores_before = pool.router(h_normed)  # (B, T, G)
        mx.eval(scores_before)
        assign_before = mx.argmax(scores_before, axis=-1)

        # Simulate adding an expert: extend router weights
        old_weight = pool.router.weight  # (G, d)
        new_row = mx.random.normal((1, old_weight.shape[1])) * 0.02
        new_weight = mx.concatenate([old_weight, new_row], axis=0)

        # Compute new scores
        scores_after = h_normed @ new_weight.T  # (B, T, G+1)
        mx.eval(scores_after)
        assign_after = mx.argmax(scores_after, axis=-1)

        displaced = mx.sum(assign_before != assign_after).item()
        total_displaced += displaced
        total_tokens += B * T

        x = layer(x)

    return total_displaced / total_tokens if total_tokens > 0 else 0


def run_seed(seed: int, steps: int = 500):
    """Run full experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    # Data setup
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    vocab_size = tokenizer.vocab_size
    N = 8  # initial experts

    # ----- Train consistent hash model -----
    print(f"\n--- Consistent Hash Routing (N={N}) ---")
    ch_model = ConsistentHashRoutingGPT(
        vocab_size=vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4,
        n_groups=N, n_capsules_per_group=32, top_k=2,
    )
    mx.eval(ch_model.parameters())
    ch_params = count_params(ch_model)
    print(f"  Params: {ch_params:,}")

    ch_result = train(ch_model, train_ds, val_ds, steps=steps,
                      batch_size=32, lr=3e-3, seed=seed, log_every=100)
    ch_val_before = ch_result["val_loss"]
    print(f"  Val loss (N={N}): {ch_val_before:.4f}")

    # Measure displacement when adding expert
    rng = random.Random(seed)
    disp_inputs, _ = val_ds.get_batch(32, rng)
    ch_displacement = measure_displacement(ch_model, disp_inputs, N, N + 1)
    print(f"  Displacement (N={N} -> {N+1}): {ch_displacement:.1%}")

    # Add 9th expert and measure quality without recalibration
    new_groups = ch_model.add_expert_to_all_layers()
    mx.eval([g.parameters() for g in new_groups])
    ch_val_after = evaluate(ch_model, val_ds, batch_size=32)
    ch_degradation = (ch_val_after - ch_val_before) / ch_val_before * 100
    print(f"  Val loss (N={N+1}, no recal): {ch_val_after:.4f} ({ch_degradation:+.2f}%)")

    # ----- Train softmax baseline -----
    print(f"\n--- Softmax Routing Baseline (N={N}) ---")
    sm_model = CapsuleMoEGPT(
        vocab_size=vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4,
        n_groups=N, n_capsules_per_group=32, top_k_groups=2,
    )
    mx.eval(sm_model.parameters())
    sm_params = count_params(sm_model)
    print(f"  Params: {sm_params:,}")

    sm_result = train(sm_model, train_ds, val_ds, steps=steps,
                      batch_size=32, lr=3e-3, seed=seed, log_every=100)
    sm_val_before = sm_result["val_loss"]
    print(f"  Val loss (N={N}): {sm_val_before:.4f}")

    # Measure displacement for softmax
    sm_displacement = measure_softmax_displacement(sm_model, disp_inputs, N + 1)
    print(f"  Displacement (N={N} -> {N+1}): {sm_displacement:.1%}")

    # Add expert to softmax model: extend each layer
    for layer in sm_model.layers:
        pool = layer.capsule_pool
        # Add new group
        from experiments.models.capsule_moe.capsule_moe import CapsuleGroup
        new_group = CapsuleGroup(64, 32)
        mx.eval(new_group.parameters())
        pool.groups.append(new_group)

        # Extend router weights
        old_w = pool.router.weight  # (G, d)
        new_row = mx.random.normal((1, old_w.shape[1])) * 0.02
        pool.router.weight = mx.concatenate([old_w, new_row], axis=0)
        pool.n_groups = N + 1
        pool.top_k_groups = 2

    sm_val_after = evaluate(sm_model, val_ds, batch_size=32)
    sm_degradation = (sm_val_after - sm_val_before) / sm_val_before * 100
    print(f"  Val loss (N={N+1}, no recal): {sm_val_after:.4f} ({sm_degradation:+.2f}%)")

    return {
        "seed": seed,
        "consistent_hash": {
            "params": ch_params,
            "val_loss_before": ch_val_before,
            "val_loss_after": ch_val_after,
            "degradation_pct": ch_degradation,
            "displacement": ch_displacement,
            "tokens_per_sec": ch_result["tokens_per_sec"],
        },
        "softmax": {
            "params": sm_params,
            "val_loss_before": sm_val_before,
            "val_loss_after": sm_val_after,
            "degradation_pct": sm_degradation,
            "displacement": sm_displacement,
            "tokens_per_sec": sm_result["tokens_per_sec"],
        },
    }


def main():
    seeds = [42, 123, 777]
    results = []

    for seed in seeds:
        r = run_seed(seed, steps=500)
        results.append(r)

    # Aggregate
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS (3 seeds)")
    print(f"{'='*60}")

    ch_vals_before = [r["consistent_hash"]["val_loss_before"] for r in results]
    ch_vals_after = [r["consistent_hash"]["val_loss_after"] for r in results]
    ch_degs = [r["consistent_hash"]["degradation_pct"] for r in results]
    ch_disps = [r["consistent_hash"]["displacement"] for r in results]

    sm_vals_before = [r["softmax"]["val_loss_before"] for r in results]
    sm_vals_after = [r["softmax"]["val_loss_after"] for r in results]
    sm_degs = [r["softmax"]["degradation_pct"] for r in results]
    sm_disps = [r["softmax"]["displacement"] for r in results]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return (sum((x - m)**2 for x in xs) / len(xs)) ** 0.5

    print(f"\n{'Method':<25} {'Val Before':>10} {'Val After':>10} {'Degrad%':>10} {'Displace%':>10}")
    print("-" * 70)
    print(f"{'Consistent Hash':<25} {mean(ch_vals_before):>10.4f} {mean(ch_vals_after):>10.4f} {mean(ch_degs):>+10.2f} {mean(ch_disps)*100:>10.1f}")
    print(f"{'Softmax':<25} {mean(sm_vals_before):>10.4f} {mean(sm_vals_after):>10.4f} {mean(sm_degs):>+10.2f} {mean(sm_disps)*100:>10.1f}")

    print(f"\n--- Kill Criteria ---")
    ch_mean_deg = mean(ch_degs)
    ch_mean_disp = mean(ch_disps) * 100
    print(f"  CH degradation: {ch_mean_deg:+.2f}% (threshold: >5% kills)")
    print(f"  CH displacement: {ch_mean_disp:.1f}% (threshold: >30% kills)")
    print(f"  Kill criterion 1 (degradation): {'KILLED' if abs(ch_mean_deg) > 5 else 'PASSES'}")
    print(f"  Kill criterion 2 (displacement): {'KILLED' if ch_mean_disp > 30 else 'PASSES'}")

    print(f"\n--- Comparison ---")
    print(f"  CH vs SM training quality: {(mean(ch_vals_before) - mean(sm_vals_before)) / mean(sm_vals_before) * 100:+.2f}%")
    print(f"  CH vs SM add-expert degradation: CH {ch_mean_deg:+.2f}% vs SM {mean(sm_degs):+.2f}%")
    print(f"  CH vs SM displacement: CH {ch_mean_disp:.1f}% vs SM {mean(sm_disps)*100:.1f}%")

    # Per-seed breakdown
    print(f"\n--- Per-Seed Breakdown ---")
    print(f"{'Seed':>6} | {'CH val_b':>8} {'CH val_a':>8} {'CH deg%':>8} {'CH disp%':>8} | {'SM val_b':>8} {'SM val_a':>8} {'SM deg%':>8} {'SM disp%':>8}")
    for r in results:
        ch = r["consistent_hash"]
        sm = r["softmax"]
        print(f"{r['seed']:>6} | {ch['val_loss_before']:>8.4f} {ch['val_loss_after']:>8.4f} {ch['degradation_pct']:>+8.2f} {ch['displacement']*100:>8.1f} | {sm['val_loss_before']:>8.4f} {sm['val_loss_after']:>8.4f} {sm['degradation_pct']:>+8.2f} {sm['displacement']*100:>8.1f}")

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
