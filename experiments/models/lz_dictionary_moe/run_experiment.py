"""Run the LZ Dictionary MoE experiment.

Compares dictionary-composed MoE against standard independent-expert MoE.

Experiments:
1. Standard MoE baseline (independent experts)
2. Dictionary MoE (shared codebook + residuals) at SAME total params
3. Dictionary MoE at FEWER total params (the compression case)
4. Dictionary utilization analysis

Kill criteria:
- Dictionary experts >3% worse than independent experts at same total params
- Dictionary utilization <30% (most entries unused)
"""

import sys
import json
import math

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_and_eval(model, train_ds, val_ds, steps=500, lr=3e-3, batch_size=32, seed=42):
    """Train model and return val loss."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: (
        nn.losses.cross_entropy(
            m(x).reshape(-1, m(x).shape[-1]),
            y.reshape(-1),
            reduction="mean"
        ) + m.aux_loss()
    ))

    # Simpler loss function to avoid double forward pass
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        ) + model.aux_loss()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % 100 == 0 or step == steps:
            print(f"    step {step:4d}/{steps} | loss {losses[-1]:.4f}")

    # Evaluate
    val_rng = random.Random(0)
    val_losses = []
    for _ in range(20):
        inputs, targets = val_ds.get_batch(batch_size, val_rng)
        logits = model(inputs)
        B, T, V = logits.shape
        vloss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean"
        )
        val_losses.append(vloss.item())

    val_loss = sum(val_losses) / len(val_losses)
    return val_loss, losses


def run_seed(seed):
    """Run full comparison for one seed."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    vs = tokenizer.vocab_size
    steps = 500
    results = {}

    # --- 1. Standard MoE baseline ---
    print("\n--- Standard MoE (4 experts, independent) ---")
    mx.random.seed(seed)
    model_std = get_model("moe", vocab_size=vs, n_embd=64, n_layer=4,
                           n_experts=4, top_k=2)
    mx.eval(model_std.parameters())
    n_std = count_params(model_std)
    print(f"  Params: {n_std:,}")
    val_std, losses_std = train_and_eval(model_std, train_ds, val_ds,
                                          steps=steps, seed=seed)
    print(f"  Val loss: {val_std:.4f}")
    results["standard_moe"] = {"val_loss": val_std, "params": n_std}

    # --- 2. Dictionary MoE (default config -- FEWER params) ---
    print("\n--- Dictionary MoE (8 dict entries, rank=32, delta=16) ---")
    mx.random.seed(seed)
    model_dict = get_model("lz_dictionary_moe", vocab_size=vs, n_embd=64,
                            n_layer=4, n_experts=4, top_k=2,
                            n_dict=8, dict_rank=32, delta_rank=16)
    mx.eval(model_dict.parameters())
    n_dict = count_params(model_dict)
    print(f"  Params: {n_dict:,} ({n_dict/n_std:.1%} of standard)")
    val_dict, losses_dict = train_and_eval(model_dict, train_ds, val_ds,
                                            steps=steps, seed=seed)
    print(f"  Val loss: {val_dict:.4f}")

    # Dictionary utilization
    diag = model_dict.dictionary_diagnostics()
    for layer_name, layer_diag in diag.items():
        print(f"  {layer_name}: util={layer_diag['utilization_rate']:.2f}, "
              f"H_norm={layer_diag['normalized_entropy']:.3f}")
        alphas = layer_diag['per_entry_weight']
        print(f"    alpha weights: [{', '.join(f'{a:.3f}' for a in alphas)}]")

    results["dict_moe_small"] = {
        "val_loss": val_dict,
        "params": n_dict,
        "diagnostics": {k: {
            "utilization_rate": v["utilization_rate"],
            "normalized_entropy": v["normalized_entropy"],
        } for k, v in diag.items()},
    }

    # --- 3. Dictionary MoE (larger -- closer to same param budget) ---
    # Increase dict_rank and delta_rank to use more params
    print("\n--- Dictionary MoE (8 dict, rank=64, delta=48 -- larger) ---")
    mx.random.seed(seed)
    model_dict_lg = get_model("lz_dictionary_moe", vocab_size=vs, n_embd=64,
                               n_layer=4, n_experts=4, top_k=2,
                               n_dict=8, dict_rank=64, delta_rank=48)
    mx.eval(model_dict_lg.parameters())
    n_dict_lg = count_params(model_dict_lg)
    print(f"  Params: {n_dict_lg:,} ({n_dict_lg/n_std:.1%} of standard)")
    val_dict_lg, losses_dict_lg = train_and_eval(model_dict_lg, train_ds, val_ds,
                                                   steps=steps, seed=seed)
    print(f"  Val loss: {val_dict_lg:.4f}")

    diag_lg = model_dict_lg.dictionary_diagnostics()
    for layer_name, layer_diag in diag_lg.items():
        print(f"  {layer_name}: util={layer_diag['utilization_rate']:.2f}, "
              f"H_norm={layer_diag['normalized_entropy']:.3f}")

    results["dict_moe_large"] = {
        "val_loss": val_dict_lg,
        "params": n_dict_lg,
        "diagnostics": {k: {
            "utilization_rate": v["utilization_rate"],
            "normalized_entropy": v["normalized_entropy"],
        } for k, v in diag_lg.items()},
    }

    # --- 4. Dense GPT baseline (no MoE) ---
    print("\n--- Dense GPT baseline ---")
    mx.random.seed(seed)
    model_gpt = get_model("gpt", vocab_size=vs, n_embd=64, n_layer=4)
    mx.eval(model_gpt.parameters())
    n_gpt = count_params(model_gpt)
    print(f"  Params: {n_gpt:,}")
    val_gpt, losses_gpt = train_and_eval(model_gpt, train_ds, val_ds,
                                           steps=steps, seed=seed)
    print(f"  Val loss: {val_gpt:.4f}")
    results["dense_gpt"] = {"val_loss": val_gpt, "params": n_gpt}

    return results


def main():
    seeds = [42, 123, 7]
    all_results = {}

    for seed in seeds:
        all_results[seed] = run_seed(seed)

    # Aggregate results
    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS (3-seed mean)")
    print("=" * 70)

    models = ["dense_gpt", "standard_moe", "dict_moe_small", "dict_moe_large"]
    labels = ["Dense GPT", "Standard MoE", "Dict MoE (small)", "Dict MoE (large)"]

    print(f"\n{'Model':<22} {'Params':>10} {'Val Loss':>10} {'vs Std MoE':>12} {'vs Dense':>10}")
    print("-" * 70)

    for model_name, label in zip(models, labels):
        losses = [all_results[s][model_name]["val_loss"] for s in seeds]
        params = all_results[seeds[0]][model_name]["params"]
        mean_loss = sum(losses) / len(losses)
        std_loss = (sum((l - mean_loss)**2 for l in losses) / len(losses)) ** 0.5

        std_moe_losses = [all_results[s]["standard_moe"]["val_loss"] for s in seeds]
        std_moe_mean = sum(std_moe_losses) / len(std_moe_losses)
        gpt_losses = [all_results[s]["dense_gpt"]["val_loss"] for s in seeds]
        gpt_mean = sum(gpt_losses) / len(gpt_losses)

        vs_moe = (mean_loss - std_moe_mean) / std_moe_mean * 100
        vs_gpt = (mean_loss - gpt_mean) / gpt_mean * 100

        print(f"{label:<22} {params:>10,} {mean_loss:>8.4f}+/-{std_loss:.4f} "
              f"{vs_moe:>+10.1f}%  {vs_gpt:>+8.1f}%")

    # Dictionary utilization
    print("\n--- Dictionary Utilization ---")
    for config_name in ["dict_moe_small", "dict_moe_large"]:
        print(f"\n  {config_name}:")
        for layer in ["layer_0", "layer_1", "layer_2", "layer_3"]:
            utils = []
            entropies = []
            for s in seeds:
                d = all_results[s][config_name].get("diagnostics", {})
                if layer in d:
                    utils.append(d[layer]["utilization_rate"])
                    entropies.append(d[layer]["normalized_entropy"])
            if utils:
                mean_util = sum(utils) / len(utils)
                mean_ent = sum(entropies) / len(entropies)
                print(f"    {layer}: util={mean_util:.2f}, H_norm={mean_ent:.3f}")

    # Kill criteria check
    print("\n--- Kill Criteria Check ---")
    dict_small_losses = [all_results[s]["dict_moe_small"]["val_loss"] for s in seeds]
    dict_large_losses = [all_results[s]["dict_moe_large"]["val_loss"] for s in seeds]
    std_losses = [all_results[s]["standard_moe"]["val_loss"] for s in seeds]

    mean_small = sum(dict_small_losses) / len(dict_small_losses)
    mean_large = sum(dict_large_losses) / len(dict_large_losses)
    mean_std = sum(std_losses) / len(std_losses)

    degradation_small = (mean_small - mean_std) / mean_std * 100
    degradation_large = (mean_large - mean_std) / mean_std * 100

    print(f"  Dict small vs Std MoE: {degradation_small:+.1f}% (kill threshold: >3%)")
    print(f"  Dict large vs Std MoE: {degradation_large:+.1f}% (kill threshold: >3%)")

    # Utilization check
    all_utils = []
    for s in seeds:
        for config in ["dict_moe_small", "dict_moe_large"]:
            d = all_results[s][config].get("diagnostics", {})
            for layer_name, layer_diag in d.items():
                all_utils.append(layer_diag["utilization_rate"])

    mean_util = sum(all_utils) / len(all_utils) if all_utils else 0
    print(f"  Mean dictionary utilization: {mean_util:.1%} (kill threshold: <30%)")

    kill_quality = degradation_large > 3.0
    kill_util = mean_util < 0.30

    if kill_quality:
        print("\n  KILL: Dictionary experts >3% worse than standard MoE")
    elif kill_util:
        print("\n  KILL: Dictionary utilization <30%")
    else:
        print("\n  PASS: Both kill criteria satisfied")

    # Save results
    results_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/lz_dictionary_moe/results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
