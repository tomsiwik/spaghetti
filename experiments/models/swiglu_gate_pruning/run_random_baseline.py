"""Random pruning baseline for SwiGLU gate-product pruning experiment.

Runs the same pruning fraction (66.5%) with randomly selected capsules
instead of gate-product-guided selection. This is the cheapest falsification
test: if random pruning yields comparable degradation, gate product profiling
adds no value.

3 random seeds x 3 training seeds = 9 evaluations.
"""

import copy
import json
import random

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from experiments.models.swiglu_gate_pruning.swiglu_gate_pruning import (
    SwiGLUGatePruningGPT,
    SwiGLUCapsulePool,
    profile_gate_products,
    identify_prunable_by_gate_product,
    prune_swiglu_model,
)


def random_prune_masks(model, target_fraction: float, rng_seed: int) -> list:
    """Create random alive masks that prune target_fraction of capsules per layer."""
    rng = random.Random(rng_seed)
    masks = []
    for layer in model.layers:
        P = layer.capsule_pool.n_capsules
        n_prune = int(P * target_fraction)
        n_alive = P - n_prune
        # Random selection of which capsules to keep
        indices = list(range(P))
        rng.shuffle(indices)
        alive_set = set(indices[:n_alive])
        mask = mx.array([i in alive_set for i in range(P)])
        masks.append(mask)
    return masks


def run_random_baseline(training_seeds=(42, 123, 7), random_seeds=(0, 1, 2),
                        target_fraction=0.665, steps=300, verbose=True):
    """Run random pruning baseline and compare against gate-product pruning."""

    results = {}

    for seed in training_seeds:
        print(f"\n{'='*60}")
        print(f"  TRAINING SEED {seed}")
        print(f"{'='*60}")

        docs = load_names()
        tokenizer = CharTokenizer(docs)
        splits = domain_split(docs, method="binary")
        vocab_size = tokenizer.vocab_size

        domain_data = {}
        for dname, ddocs in splits.items():
            dtrain, dval = train_val_split(ddocs, seed=seed)
            domain_data[dname] = {
                "train": CharDataset(dtrain, tokenizer, block_size=32),
                "val": CharDataset(dval, tokenizer, block_size=32),
            }

        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
        joint_val = CharDataset(all_val_docs, tokenizer, block_size=32)

        model_kwargs = dict(vocab_size=vocab_size, block_size=32, n_embd=64,
                            n_head=4, n_layer=4, n_capsules=128)

        # Train SwiGLU model
        print(f"\n--- Training SwiGLU model ---")
        mx.random.seed(seed)
        model = SwiGLUGatePruningGPT(**model_kwargs)
        mx.eval(model.parameters())

        train(model, domain_data["a_m"]["train"], domain_data["a_m"]["val"],
              steps=steps, batch_size=32, lr=3e-3, seed=seed, log_every=100)

        unpruned_loss = evaluate(model, joint_val, batch_size=32)
        print(f"  Unpruned val loss: {unpruned_loss:.4f}")

        # Gate-product guided pruning at tau=0.05 (for comparison)
        gp_profiles = profile_gate_products(model, joint_val,
                                             n_batches=20, batch_size=32, seed=seed)
        gp_masks = identify_prunable_by_gate_product(gp_profiles, threshold=0.05)

        model_gp = copy.deepcopy(model)
        gp_stats = prune_swiglu_model(model_gp, gp_masks, verbose=False)
        gp_loss = evaluate(model_gp, joint_val, batch_size=32)
        gp_delta = (gp_loss - unpruned_loss) / unpruned_loss * 100
        gp_pct = gp_stats["pct_pruned"]

        print(f"  Gate-product pruning: {gp_pct:.1f}% pruned, delta={gp_delta:+.2f}%")

        # Random pruning at the SAME fraction as gate-product
        # Use the actual per-seed pruning fraction, not a fixed 66.5%
        actual_fraction = gp_pct / 100.0

        random_results = []
        for rseed in random_seeds:
            model_rand = copy.deepcopy(model)
            rand_masks = random_prune_masks(model_rand, actual_fraction, rseed)
            rand_stats = prune_swiglu_model(model_rand, rand_masks, verbose=False)
            rand_loss = evaluate(model_rand, joint_val, batch_size=32)
            rand_delta = (rand_loss - unpruned_loss) / unpruned_loss * 100
            rand_pct = rand_stats["pct_pruned"]

            print(f"  Random pruning (rseed={rseed}): {rand_pct:.1f}% pruned, "
                  f"delta={rand_delta:+.2f}%")

            random_results.append({
                "random_seed": rseed,
                "pct_pruned": rand_pct,
                "pruned_loss": rand_loss,
                "delta_pct": rand_delta,
            })

        rand_deltas = [r["delta_pct"] for r in random_results]
        mean_rand_delta = sum(rand_deltas) / len(rand_deltas)

        results[str(seed)] = {
            "unpruned_loss": unpruned_loss,
            "gate_product": {
                "pct_pruned": gp_pct,
                "pruned_loss": gp_loss,
                "delta_pct": gp_delta,
            },
            "random": random_results,
            "random_mean_delta": mean_rand_delta,
            "actual_fraction": actual_fraction,
        }

        print(f"\n  Gate-product delta: {gp_delta:+.2f}%")
        print(f"  Random mean delta: {mean_rand_delta:+.2f}%")
        ratio = mean_rand_delta / gp_delta if gp_delta != 0 else float('inf')
        print(f"  Random/GP ratio: {ratio:.2f}x")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RANDOM vs GATE-PRODUCT")
    print(f"{'='*60}")

    all_gp_deltas = [results[str(s)]["gate_product"]["delta_pct"] for s in training_seeds]
    all_rand_deltas = []
    for s in training_seeds:
        for r in results[str(s)]["random"]:
            all_rand_deltas.append(r["delta_pct"])

    mean_gp = sum(all_gp_deltas) / len(all_gp_deltas)
    mean_rand = sum(all_rand_deltas) / len(all_rand_deltas)

    print(f"  Gate-product mean delta: {mean_gp:+.2f}%")
    print(f"  Random mean delta: {mean_rand:+.2f}%")
    print(f"  Random is {mean_rand/mean_gp:.1f}x the gate-product degradation"
          if mean_gp > 0 else "")

    return results


if __name__ == "__main__":
    results = run_random_baseline()
    outpath = "/Users/tom/Code/tomsiwik/llm/experiments/models/swiglu_gate_pruning/random_baseline_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
