"""Run the AIMD load balance experiment.

Three-way comparison:
  1. AIMD (additive increase / multiplicative decrease) on routing bias
  2. Aux loss (Switch Transformer style balance loss)
  3. No balance (pure softmax, control)

Metrics:
  - val_loss (quality)
  - load imbalance over training (convergence to fairness)
  - steps to reach fair allocation (convergence speed)

Seeds: 3 (42, 123, 7) for statistical reliability.
"""

import sys
import time
import random
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, ".")

from experiments.data import (
    load_names, CharTokenizer, CharDataset, train_val_split,
)


def ntp_loss(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean",
    )
    return loss + model.aux_loss()


def train_with_tracking(model, train_ds, val_ds, steps=500, batch_size=32,
                         lr=3e-3, seed=42, log_every=50, track_every=10):
    """Train model and track load balance metrics over time."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    imbalance_history = []  # (step, mean_imbalance)
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        # Track load balance every N steps
        if step % track_every == 0:
            stats = model.load_balance_stats()
            if stats:
                mean_imb = sum(s["imbalance"] for s in stats) / len(stats)
                imbalance_history.append((step, mean_imb))

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            stats = model.load_balance_stats()
            imb_str = ""
            if stats:
                mean_imb = sum(s["imbalance"] for s in stats) / len(stats)
                imb_str = f" | imbal {mean_imb:.4f}"
            print(f"  step {step:4d}/{steps} | loss {loss_val:.4f}{imb_str}")

    # Final evaluation
    val_loss = evaluate(model, val_ds, batch_size)
    elapsed = time.time() - t0

    return {
        "val_loss": val_loss,
        "final_loss": losses[-1],
        "losses": losses,
        "imbalance_history": imbalance_history,
        "elapsed_s": elapsed,
    }


def evaluate(model, dataset, batch_size=32, n_batches=10):
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean",
        )
        total += loss.item()
    return total / n_batches


def convergence_step(imbalance_history, threshold=0.05):
    """Find first step where imbalance drops below threshold and stays."""
    for i, (step, imb) in enumerate(imbalance_history):
        # Check if all subsequent are below threshold
        remaining = [im for _, im in imbalance_history[i:]]
        if all(im < threshold for im in remaining):
            return step
    return None  # never converged


def run_experiment():
    seeds = [42, 123, 7]
    steps = 500
    batch_size = 32
    lr = 3e-3
    n_groups = 4
    n_capsules_per_group = 64

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    model_configs = {
        "aimd_balance": {
            "module": "micro.models.aimd_load_balance.aimd_load_balance",
            "class": "AIMDLoadBalanceGPT",
            "kwargs": dict(
                vocab_size=tokenizer.vocab_size, block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
                alpha=0.05, beta=0.5, epsilon=0.02,
            ),
        },
        "aux_loss_balance": {
            "module": "micro.models.aimd_load_balance.aimd_load_balance",
            "class": "AuxLossBalanceGPT",
            "kwargs": dict(
                vocab_size=tokenizer.vocab_size, block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
                balance_coeff=0.01,
            ),
        },
        "no_balance": {
            "module": "micro.models.aimd_load_balance.aimd_load_balance",
            "class": "NoBalanceGPT",
            "kwargs": dict(
                vocab_size=tokenizer.vocab_size, block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
            ),
        },
    }

    all_results = {}

    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            mx.random.seed(seed)

            # Import and instantiate
            import importlib
            mod = importlib.import_module(config["module"])
            cls = getattr(mod, config["class"])
            model = cls(**config["kwargs"])
            mx.eval(model.parameters())

            n_params = sum(v.size for _, v in nn.utils.tree_flatten(
                model.trainable_parameters()))
            print(f"  Params: {n_params:,}")

            result = train_with_tracking(
                model, train_ds, val_ds, steps=steps, batch_size=batch_size,
                lr=lr, seed=seed, log_every=100, track_every=10,
            )

            # Convergence analysis
            conv_step = convergence_step(result["imbalance_history"], threshold=0.15)
            result["convergence_step"] = conv_step

            # Final balance stats
            final_stats = model.load_balance_stats()
            result["final_balance_stats"] = final_stats

            seed_results.append({
                "seed": seed,
                "val_loss": result["val_loss"],
                "final_loss": result["final_loss"],
                "convergence_step": conv_step,
                "elapsed_s": result["elapsed_s"],
                "imbalance_history": result["imbalance_history"],
                "final_balance_stats": final_stats,
            })

            print(f"  val_loss: {result['val_loss']:.4f}")
            print(f"  convergence_step: {conv_step}")
            if final_stats:
                for s in final_stats:
                    print(f"  layer {s['layer']}: imbal={s['imbalance']:.4f}, "
                          f"loads={[f'{l:.3f}' for l in s['load_fractions']]}")

        all_results[model_name] = seed_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} | {'Val Loss (mean)':<16} | {'Val Loss (std)':<15} | "
          f"{'Conv Step (mean)':<16}")
    print("-" * 70)

    for model_name, results in all_results.items():
        val_losses = [r["val_loss"] for r in results]
        conv_steps = [r["convergence_step"] for r in results
                      if r["convergence_step"] is not None]

        mean_vl = sum(val_losses) / len(val_losses)
        std_vl = (sum((v - mean_vl) ** 2 for v in val_losses) / len(val_losses)) ** 0.5
        mean_cs = sum(conv_steps) / len(conv_steps) if conv_steps else None
        cs_str = f"{mean_cs:.0f}" if mean_cs else "never"

        print(f"{model_name:<20} | {mean_vl:<16.4f} | {std_vl:<15.4f} | {cs_str:<16}")

    # Kill criteria evaluation
    print("\n" + "=" * 70)
    print("KILL CRITERIA EVALUATION")
    print("=" * 70)

    aimd_vl = [r["val_loss"] for r in all_results["aimd_balance"]]
    aux_vl = [r["val_loss"] for r in all_results["aux_loss_balance"]]
    no_vl = [r["val_loss"] for r in all_results["no_balance"]]

    mean_aimd = sum(aimd_vl) / len(aimd_vl)
    mean_aux = sum(aux_vl) / len(aux_vl)
    mean_no = sum(no_vl) / len(no_vl)

    quality_diff = (mean_aimd - mean_aux) / mean_aux * 100
    print(f"KC1: AIMD vs Aux Loss quality: {quality_diff:+.2f}%")
    if mean_aimd > mean_aux:
        print(f"  -> AIMD is WORSE by {quality_diff:+.2f}% (higher loss = worse)")
        print(f"  -> KC1 TRIGGERED: AIMD worse than aux loss")
    else:
        print(f"  -> AIMD is BETTER by {-quality_diff:.2f}%")
        print(f"  -> KC1 PASSES")

    aimd_cs = [r["convergence_step"] for r in all_results["aimd_balance"]
               if r["convergence_step"] is not None]
    aux_cs = [r["convergence_step"] for r in all_results["aux_loss_balance"]
              if r["convergence_step"] is not None]

    if aimd_cs and aux_cs:
        mean_aimd_cs = sum(aimd_cs) / len(aimd_cs)
        mean_aux_cs = sum(aux_cs) / len(aux_cs)
        ratio = mean_aimd_cs / mean_aux_cs
        print(f"\nKC2: Convergence speed ratio (AIMD/Aux): {ratio:.2f}x")
        if ratio > 2.0:
            print(f"  -> KC2 TRIGGERED: AIMD takes {ratio:.1f}x longer to converge")
        else:
            print(f"  -> KC2 PASSES: AIMD converges within 2x of aux loss")
    else:
        print("\nKC2: Cannot evaluate (one or both never converged)")
        if not aimd_cs:
            print("  -> AIMD never converged to fair allocation")
        if not aux_cs:
            print("  -> Aux loss never converged to fair allocation")

    # Also report imbalance at end of training
    print("\n" + "=" * 70)
    print("FINAL LOAD IMBALANCE (mean across seeds, layer 0)")
    print("=" * 70)

    for model_name, results in all_results.items():
        imbalances = []
        for r in results:
            if r["imbalance_history"]:
                _, final_imb = r["imbalance_history"][-1]
                imbalances.append(final_imb)
        if imbalances:
            mean_imb = sum(imbalances) / len(imbalances)
            print(f"  {model_name:<20}: {mean_imb:.4f}")

    # Save results
    output_path = "experiments/models/aimd_load_balance/results.json"
    # Convert to serializable format
    serializable = {}
    for model_name, results in all_results.items():
        serializable[model_name] = []
        for r in results:
            sr = dict(r)
            # Clean up non-serializable data
            if sr.get("final_balance_stats"):
                sr["final_balance_stats"] = [
                    {k: v for k, v in s.items()} for s in sr["final_balance_stats"]
                ]
            serializable[model_name].append(sr)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
