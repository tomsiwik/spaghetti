"""
ReLoRA Merge Cycle Scaling: Does composition quality degrade with K=5..200 merge cycles?

Extends the proven micro ReLoRA composition test to stress-test at high merge cycle
counts. Production ReLoRA uses hundreds of cycles; the original micro test only
validated K=5.

Design:
  - Fixed total pretraining budget (2000 steps) for all K values
  - Vary K in {5, 25, 50, 100, 200}: steps_per_cycle = 2000/K
  - For each K: build ReLoRA base, build conventional base (same budget),
    train N=4 domain experts on each, measure cos_ratio and loss_ratio
  - Track trend: does cos_ratio or loss_ratio grow with K?

Kill criteria:
  K1: cos_ratio > 5.0 at K=200 (systematic weight bias from repeated merges)
  K2: loss_ratio > 1.50 at K=200 (cumulative merge error compounds)

Architecture: Reuses LoRA infrastructure from lora_procrustes via relora_composition_test.
All training uses the micro names dataset (character-level), d=64, r=8, 4-layer GPT.
"""

import math
import time
import random
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..gpt import GPT
from ..lora_procrustes.lora_procrustes import LoRAGPT
from ..relora_composition_test.relora_composition_test import (
    merge_lora_into_base,
    train_relora,
    train_conventional,
    train_lora_expert,
    compute_pairwise_cosine,
    _ntp_loss,
    _evaluate,
)


def run_single_K(
    K: int,
    total_pretrain_steps: int = 2000,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    block_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict:
    """Run the composition test for a single merge cycle count K.

    Returns dict with cos_ratio, loss_ratio, and detailed metrics.
    """
    from ...data import load_names, CharTokenizer, CharDataset, domain_split

    merge_every = total_pretrain_steps // K
    if merge_every < 1:
        raise ValueError(f"K={K} too large for total_steps={total_pretrain_steps}")

    print(f"\n{'='*72}")
    print(f"K={K}: merge_every={merge_every}, total_steps={total_pretrain_steps}")
    print(f"{'='*72}")

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:n_experts]

    rng_split = random.Random(seed)
    split_idx = int(len(docs) * 0.9)
    rng_split.shuffle(docs_copy := list(docs))
    train_docs = docs_copy[:split_idx]
    val_docs = docs_copy[split_idx:]

    train_ds = CharDataset(train_docs, tokenizer, block_size)
    val_ds = CharDataset(val_docs, tokenizer, block_size)

    t0 = time.time()

    # Phase 1a: ReLoRA pretraining
    print(f"  ReLoRA pretraining (K={K}, {merge_every} steps/cycle)...")
    relora_model = LoRAGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
    )
    mx.eval(relora_model.parameters())

    relora_result = train_relora(
        relora_model, train_ds,
        total_steps=total_pretrain_steps, merge_every=merge_every,
        batch_size=batch_size, lr=lr, seed=seed,
        log_every=max(merge_every, 200),
    )
    relora_val = _evaluate(relora_model, val_ds, batch_size)
    print(f"  ReLoRA val loss: {relora_val:.4f} ({relora_result['merges_done']} merges)")

    # Phase 1b: Conventional pretraining (same budget, same seed)
    print(f"  Conventional pretraining ({total_pretrain_steps} steps)...")
    conv_model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
    )
    mx.eval(conv_model.parameters())

    conv_result = train_conventional(
        conv_model, train_ds,
        total_steps=total_pretrain_steps, batch_size=batch_size,
        lr=lr, seed=seed, log_every=500,
    )
    conv_val = _evaluate(conv_model, val_ds, batch_size)
    print(f"  Conventional val loss: {conv_val:.4f}")

    # Phase 2: Train domain experts on both bases
    print(f"  Training {n_experts} experts on each base...")
    relora_deltas_all = []
    conv_deltas_all = []
    relora_expert_losses = []
    conv_expert_losses = []

    for i, domain in enumerate(domain_names):
        domain_docs = domains[domain]
        rng_domain = random.Random(seed + 1000 + i)
        domain_docs_shuffled = list(domain_docs)
        rng_domain.shuffle(domain_docs_shuffled)
        n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
        expert_train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, block_size)
        expert_val_ds = CharDataset(
            domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
            else domain_docs_shuffled, tokenizer, block_size
        )

        # Expert on ReLoRA base
        _, r_deltas, r_val = train_lora_expert(
            relora_model, expert_train_ds, expert_val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, is_relora_base=True,
        )
        relora_deltas_all.append(r_deltas)
        relora_expert_losses.append(r_val)

        # Expert on conventional base
        _, c_deltas, c_val = train_lora_expert(
            conv_model, expert_train_ds, expert_val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, is_relora_base=False,
        )
        conv_deltas_all.append(c_deltas)
        conv_expert_losses.append(c_val)

        print(f"    Expert {i} ({domain}): relora_val={r_val:.4f}, conv_val={c_val:.4f}")

    # Phase 3: Measure orthogonality
    relora_cosines = compute_pairwise_cosine(relora_deltas_all)
    conv_cosines = compute_pairwise_cosine(conv_deltas_all)

    relora_cos_vals = [abs(c) for (_, _, c) in relora_cosines]
    conv_cos_vals = [abs(c) for (_, _, c) in conv_cosines]

    relora_mean_cos = sum(relora_cos_vals) / len(relora_cos_vals) if relora_cos_vals else 0
    conv_mean_cos = sum(conv_cos_vals) / len(conv_cos_vals) if conv_cos_vals else 0

    cos_ratio = relora_mean_cos / (conv_mean_cos + 1e-12)

    relora_mean_loss = sum(relora_expert_losses) / len(relora_expert_losses)
    conv_mean_loss = sum(conv_expert_losses) / len(conv_expert_losses)
    loss_ratio = relora_mean_loss / (conv_mean_loss + 1e-12)

    base_ratio = relora_val / (conv_val + 1e-12)

    elapsed = time.time() - t0

    print(f"  K={K} results: cos_ratio={cos_ratio:.4f}, loss_ratio={loss_ratio:.4f}, "
          f"base_ratio={base_ratio:.4f}, time={elapsed:.0f}s")

    return {
        "K": K,
        "merge_every": merge_every,
        "merges_done": relora_result["merges_done"],
        "cos_ratio": float(cos_ratio),
        "loss_ratio": float(loss_ratio),
        "base_ratio": float(base_ratio),
        "relora_mean_cos": float(relora_mean_cos),
        "conv_mean_cos": float(conv_mean_cos),
        "relora_mean_expert_loss": float(relora_mean_loss),
        "conv_mean_expert_loss": float(conv_mean_loss),
        "relora_base_loss": float(relora_val),
        "conv_base_loss": float(conv_val),
        "relora_expert_losses": [float(x) for x in relora_expert_losses],
        "conv_expert_losses": [float(x) for x in conv_expert_losses],
        "relora_cosines": [(i, j, float(c)) for (i, j, c) in relora_cosines],
        "conv_cosines": [(i, j, float(c)) for (i, j, c) in conv_cosines],
        "elapsed_s": float(elapsed),
    }


def run_scaling_experiment(
    K_values: list = None,
    seeds: list = None,
    total_pretrain_steps: int = 2000,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    lora_rank: int = 8,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
) -> dict:
    """Run the full merge cycle scaling experiment across K values and seeds.

    Default K_values: [5, 25, 50, 100, 200]
    Default seeds: [42, 7] (2 seeds for speed; 3 seeds would add ~50% runtime)

    Returns comprehensive results dict with trend analysis.
    """
    if K_values is None:
        K_values = [5, 25, 50, 100, 200]
    if seeds is None:
        seeds = [42, 7]

    print("=" * 72)
    print("ReLoRA MERGE CYCLE SCALING EXPERIMENT")
    print(f"  K values: {K_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Total pretrain steps: {total_pretrain_steps}")
    print(f"  Config: d={n_embd}, r={lora_rank}, L={n_layer}, N={n_experts}")
    print("=" * 72)

    t0_total = time.time()
    all_results = {}

    for K in K_values:
        all_results[K] = {}
        for seed in seeds:
            result = run_single_K(
                K=K,
                total_pretrain_steps=total_pretrain_steps,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                lora_rank=lora_rank, lora_alpha=1.0,
                expert_train_steps=expert_train_steps,
                n_experts=n_experts,
                batch_size=batch_size, lr=lr, seed=seed,
            )
            all_results[K][seed] = result

    # Aggregate per-K
    summary = []
    for K in K_values:
        cos_ratios = [all_results[K][s]["cos_ratio"] for s in seeds]
        loss_ratios = [all_results[K][s]["loss_ratio"] for s in seeds]
        base_ratios = [all_results[K][s]["base_ratio"] for s in seeds]

        mean_cr = sum(cos_ratios) / len(cos_ratios)
        mean_lr = sum(loss_ratios) / len(loss_ratios)
        mean_br = sum(base_ratios) / len(base_ratios)

        summary.append({
            "K": K,
            "mean_cos_ratio": float(mean_cr),
            "std_cos_ratio": float(_std(cos_ratios)),
            "cos_ratios": cos_ratios,
            "mean_loss_ratio": float(mean_lr),
            "std_loss_ratio": float(_std(loss_ratios)),
            "loss_ratios": loss_ratios,
            "mean_base_ratio": float(mean_br),
            "base_ratios": base_ratios,
        })

    # Trend analysis: fit log-linear model cos_ratio = a * K^b
    # If b > 0, cos_ratio grows with K (bad)
    # If b ~ 0, cos_ratio is stable (good)
    Ks = [s["K"] for s in summary]
    cos_rs = [s["mean_cos_ratio"] for s in summary]
    loss_rs = [s["mean_loss_ratio"] for s in summary]

    cos_slope = _log_linear_slope(Ks, cos_rs)
    loss_slope = _log_linear_slope(Ks, loss_rs)

    # Kill criteria at K=200
    K200_entry = next((s for s in summary if s["K"] == 200), None)
    k1_result = K200_entry["mean_cos_ratio"] if K200_entry else None
    k2_result = K200_entry["mean_loss_ratio"] if K200_entry else None

    k1_killed = k1_result is not None and k1_result > 5.0
    k2_killed = k2_result is not None and k2_result > 1.50

    if k1_killed or k2_killed:
        verdict = "KILLED"
    elif k1_result is not None and k1_result < 3.0 and k2_result < 1.30:
        verdict = "SURVIVES"
    else:
        verdict = "INCONCLUSIVE"

    elapsed_total = time.time() - t0_total

    # Print summary
    print(f"\n{'='*72}")
    print("SUMMARY: ReLoRA Merge Cycle Scaling")
    print(f"{'='*72}")
    print(f"  {'K':>5} {'cos_ratio':>12} {'loss_ratio':>12} {'base_ratio':>12}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for s in summary:
        print(f"  {s['K']:>5} {s['mean_cos_ratio']:>12.4f} "
              f"{s['mean_loss_ratio']:>12.4f} {s['mean_base_ratio']:>12.4f}")

    print(f"\n  Trend (log-linear slope):")
    print(f"    cos_ratio  vs K: slope = {cos_slope:.4f} "
          f"({'GROWING' if cos_slope > 0.1 else 'STABLE' if cos_slope < 0.1 else 'SHRINKING'})")
    print(f"    loss_ratio vs K: slope = {loss_slope:.4f} "
          f"({'GROWING' if loss_slope > 0.1 else 'STABLE' if loss_slope < 0.1 else 'SHRINKING'})")

    print(f"\n  Kill Criteria at K=200:")
    print(f"    K1: cos_ratio = {k1_result:.4f} (threshold >5.0) -> "
          f"{'KILLED' if k1_killed else 'SURVIVES'}")
    print(f"    K2: loss_ratio = {k2_result:.4f} (threshold >1.50) -> "
          f"{'KILLED' if k2_killed else 'SURVIVES'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")

    # Build output
    output = {
        "experiment": "relora_merge_cycle_scaling",
        "config": {
            "K_values": K_values,
            "seeds": seeds,
            "total_pretrain_steps": total_pretrain_steps,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "lora_rank": lora_rank,
            "expert_train_steps": expert_train_steps,
            "n_experts": n_experts,
            "batch_size": batch_size,
            "lr": lr,
        },
        "summary": summary,
        "trend": {
            "cos_ratio_log_slope": float(cos_slope),
            "loss_ratio_log_slope": float(loss_slope),
        },
        "kill_criteria": {
            "K1_cos_ratio_at_K200": float(k1_result) if k1_result else None,
            "K1_threshold": 5.0,
            "K1_killed": k1_killed,
            "K2_loss_ratio_at_K200": float(k2_result) if k2_result else None,
            "K2_threshold": 1.50,
            "K2_killed": k2_killed,
        },
        "verdict": verdict,
        "per_K_per_seed": {
            str(K): {str(s): all_results[K][s] for s in seeds}
            for K in K_values
        },
        "elapsed_total_s": float(elapsed_total),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


def _std(values: list) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _log_linear_slope(xs: list, ys: list) -> float:
    """Fit y = a * x^b in log-log space, return b (slope).

    Positive b means y grows with x.
    b near 0 means y is constant.
    Negative b means y shrinks with x.
    """
    if len(xs) < 2:
        return 0.0
    # Filter out non-positive values
    pairs = [(math.log(x), math.log(max(y, 1e-12))) for x, y in zip(xs, ys) if x > 0]
    if len(pairs) < 2:
        return 0.0
    n = len(pairs)
    sx = sum(lx for lx, _ in pairs)
    sy = sum(ly for _, ly in pairs)
    sxy = sum(lx * ly for lx, ly in pairs)
    sxx = sum(lx * lx for lx, _ in pairs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


if __name__ == "__main__":
    run_scaling_experiment()
