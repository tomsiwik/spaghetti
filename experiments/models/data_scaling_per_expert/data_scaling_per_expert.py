#!/usr/bin/env python3
"""
Data Scaling Per Expert: Expert quality as function of training examples.

Hypothesis: LoRA expert quality (NTP loss / PPL) improves with more training
data but saturates due to adapter capacity (rank) limitations. There exists
a "sweet spot" where adding more data stops helping.

Kill criteria:
  K1: Quality does not improve beyond 200 examples (data scaling flat)
  K2: Quality still improving significantly at 5000 examples (need more data)

"Significantly" for K2: relative PPL improvement from 2000->5000 > 5%.

Experimental design:
  1. Pretrain a micro GPT on all names data
  2. Pick domain a_e (10K+ names, largest domain)
  3. For each N in {50, 100, 200, 500, 1000, 2000, 5000}:
     - Sample N training examples from domain
     - Train LoRA expert (rank=8) for fixed steps
     - Evaluate on held-out test set (500 examples)
  4. Fixed training steps across all N (same compute budget)
  5. 3 seeds for statistical significance
  6. Fit log-linear scaling law: PPL = a * N^(-alpha)

Reuses: GPT, LoRALinear, LoRAGPT, CharTokenizer, CharDataset from
  experiments/models/base_free_composition/base_free_composition.py
"""

import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import reusable infrastructure ────────────────────────────────────────
# Direct import to avoid experiments/models/__init__.py which loads all models
import importlib.util

_BFC_PATH = os.path.join(os.path.dirname(__file__), "..", "base_free_composition",
                         "base_free_composition.py")
_spec = importlib.util.spec_from_file_location("base_free_composition", _BFC_PATH)
_bfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bfc)

GPT = _bfc.GPT
LoRALinear = _bfc.LoRALinear
LoRAGPT = _bfc.LoRAGPT
CharTokenizer = _bfc.CharTokenizer
CharDataset = _bfc.CharDataset
load_names = _bfc.load_names
domain_split = _bfc.domain_split
train_gpt = _bfc.train_gpt
evaluate_model = _bfc.evaluate_model


# ── Experiment-specific LoRA training ─────────────────────────────────────


def train_lora_expert_fixed_steps(
    base_gpt: GPT,
    train_ds: CharDataset,
    val_ds: CharDataset,
    rank: int = 8,
    alpha: float = 1.0,
    steps: int = 300,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train LoRA expert for fixed number of steps, return metrics.

    Returns dict with:
      - val_loss: validation NTP loss
      - val_ppl: exp(val_loss)
      - train_losses: per-step training losses
      - final_train_loss: last training loss
    """
    base_copy = copy.deepcopy(base_gpt)
    lora_model = LoRAGPT(base_copy, rank=rank, alpha=alpha)
    lora_model.to(device)
    lora_model.train()

    rng = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)

    train_losses = []
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng, device)
        logits = lora_model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    val_loss = evaluate_model(lora_model, val_ds, batch_size, n_batches=20,
                              device=device)
    val_ppl = math.exp(min(val_loss, 20.0))  # Cap to avoid overflow

    return {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "final_train_loss": train_losses[-1] if train_losses else float("inf"),
        "train_losses": train_losses,
    }


# ── Scaling law fitting ──────────────────────────────────────────────────


def fit_power_law(n_values, ppl_values):
    """Fit PPL = a * N^(-alpha) via log-log linear regression.

    Returns: alpha (exponent), a (coefficient), r_squared.
    Negative alpha means PPL decreases with more data (expected).
    """
    log_n = np.log(np.array(n_values, dtype=float))
    log_ppl = np.log(np.array(ppl_values, dtype=float))

    valid = np.isfinite(log_n) & np.isfinite(log_ppl)
    log_n = log_n[valid]
    log_ppl = log_ppl[valid]

    if len(log_n) < 2:
        return {"alpha": float("nan"), "a": float("nan"), "r_squared": float("nan")}

    n = len(log_n)
    mean_x = np.mean(log_n)
    mean_y = np.mean(log_ppl)
    ss_xx = np.sum((log_n - mean_x) ** 2)
    ss_xy = np.sum((log_n - mean_x) * (log_ppl - mean_y))
    ss_yy = np.sum((log_ppl - mean_y) ** 2)

    if ss_xx < 1e-12:
        return {"alpha": float("nan"), "a": float("nan"), "r_squared": float("nan")}

    slope = ss_xy / ss_xx  # This is -alpha (since PPL decreases with N)
    log_a = mean_y - slope * mean_x
    a = np.exp(log_a)

    ss_res = np.sum((log_ppl - (log_a + slope * log_n)) ** 2)
    r_squared = 1 - ss_res / ss_yy if ss_yy > 1e-12 else float("nan")

    return {
        "alpha": float(-slope),  # alpha > 0 means PPL decreases with N
        "a": float(a),
        "slope": float(slope),
        "r_squared": float(r_squared),
        "n_points": int(n),
    }


def compute_saturation_point(n_values, ppl_values, threshold_pct=5.0):
    """Find the data size beyond which improvement drops below threshold.

    Computes relative improvement between consecutive data sizes.
    Saturation point = smallest N where all subsequent improvements < threshold%.

    Returns dict with saturation_n, improvements list, etc.
    """
    improvements = []
    for i in range(1, len(n_values)):
        rel_improvement = (ppl_values[i - 1] - ppl_values[i]) / ppl_values[i - 1] * 100
        improvements.append({
            "from_n": n_values[i - 1],
            "to_n": n_values[i],
            "ppl_from": ppl_values[i - 1],
            "ppl_to": ppl_values[i],
            "rel_improvement_pct": rel_improvement,
            "data_ratio": n_values[i] / n_values[i - 1],
        })

    # Find saturation: first point where ALL subsequent improvements < threshold
    saturation_n = None
    for i in range(len(improvements)):
        all_below = all(imp["rel_improvement_pct"] < threshold_pct
                        for imp in improvements[i:])
        if all_below:
            saturation_n = improvements[i]["from_n"]
            break

    return {
        "improvements": improvements,
        "saturation_n": saturation_n,
        "threshold_pct": threshold_pct,
    }


# ── Main Experiment ──────────────────────────────────────────────────────


def run_experiment(
    data_sizes=None,
    n_seeds=3,
    d_model=64,
    n_heads=4,
    n_layers=4,
    block_size=32,
    lora_rank=8,
    lora_alpha=1.0,
    pretrain_steps=1000,
    expert_steps=300,
    batch_size=32,
    lr=3e-3,
    n_test=500,
    domain="a_e",
    device="cpu",
):
    """Run the data scaling experiment.

    For each data size N, train a LoRA expert and measure quality.
    Same base model, same training steps, same rank -- only data size varies.
    """
    if data_sizes is None:
        data_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

    seeds = [42 + i * 100 for i in range(n_seeds)]

    print("=" * 72)
    print("DATA SCALING PER EXPERT EXPERIMENT")
    print("=" * 72)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}")
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
    print(f"  Pretrain: {pretrain_steps} steps, Expert: {expert_steps} steps")
    print(f"  Data sizes: {data_sizes}")
    print(f"  Domain: {domain}")
    print(f"  Test set: {n_test} examples (held out)")
    print(f"  Seeds: {seeds}")
    print("=" * 72)

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # Domain split
    domains = domain_split(docs, "quintary")
    domain_docs = domains[domain]
    print(f"  Domain '{domain}': {len(domain_docs)} names")
    assert len(domain_docs) >= max(data_sizes) + n_test, \
        f"Domain {domain} has {len(domain_docs)} names, need {max(data_sizes) + n_test}"

    t0 = time.time()

    all_results = {}  # {N: [{seed, val_loss, val_ppl, ...}, ...]}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*72}")
        print(f"  SEED {seed} ({seed_idx + 1}/{n_seeds})")
        print(f"{'='*72}")

        torch.manual_seed(seed)
        rng = random.Random(seed)

        # Shuffle domain docs for this seed
        domain_shuffled = list(domain_docs)
        rng.shuffle(domain_shuffled)

        # Hold out test set (FIXED across data sizes for fair comparison)
        test_docs = domain_shuffled[:n_test]
        available_train = domain_shuffled[n_test:]  # remaining for training
        assert len(available_train) >= max(data_sizes), \
            f"Not enough train data: {len(available_train)} < {max(data_sizes)}"

        test_ds = CharDataset(test_docs, tokenizer, block_size)

        # Pretrain base model on ALL data (not domain-specific)
        print(f"\n  --- Pretraining base model ---")
        torch.manual_seed(seed)
        base_model = GPT(vocab_size, block_size, d_model, n_heads, n_layers)

        # Use all docs for pretraining (base should be general)
        all_train_ds = CharDataset(docs, tokenizer, block_size)
        train_gpt(base_model, all_train_ds, steps=pretrain_steps,
                  batch_size=batch_size, lr=lr, seed=seed, device=device)

        base_val_loss = evaluate_model(base_model, test_ds, batch_size,
                                       n_batches=20, device=device)
        base_ppl = math.exp(min(base_val_loss, 20.0))
        print(f"  Base val loss on domain: {base_val_loss:.4f} (PPL={base_ppl:.2f})")

        # Train LoRA expert at each data size
        for N in data_sizes:
            print(f"\n  --- Training expert with N={N} examples ---")
            train_docs = available_train[:N]
            train_ds = CharDataset(train_docs, tokenizer, block_size)

            result = train_lora_expert_fixed_steps(
                base_model, train_ds, test_ds,
                rank=lora_rank, alpha=lora_alpha,
                steps=expert_steps, batch_size=min(batch_size, N),
                lr=lr, seed=seed, device=device,
            )

            # Relative improvement over base
            ppl_improvement = (base_ppl - result["val_ppl"]) / base_ppl * 100
            loss_improvement = (base_val_loss - result["val_loss"]) / base_val_loss * 100

            entry = {
                "seed": seed,
                "n_train": N,
                "val_loss": result["val_loss"],
                "val_ppl": result["val_ppl"],
                "final_train_loss": result["final_train_loss"],
                "base_val_loss": base_val_loss,
                "base_ppl": base_ppl,
                "ppl_improvement_pct": ppl_improvement,
                "loss_improvement_pct": loss_improvement,
            }

            if N not in all_results:
                all_results[N] = []
            all_results[N].append(entry)

            print(f"    N={N}: val_loss={result['val_loss']:.4f} "
                  f"PPL={result['val_ppl']:.2f} "
                  f"(base PPL={base_ppl:.2f}, improvement={ppl_improvement:+.1f}%)")

    total_time = time.time() - t0

    # ================================================================
    # Aggregate across seeds
    # ================================================================
    print(f"\n{'='*72}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*72}")

    print(f"\n  {'N':>6s} | {'PPL (mean)':>10s} | {'PPL (std)':>10s} | "
          f"{'Loss (mean)':>11s} | {'Improve%':>9s} | {'vs Base':>8s}")
    print(f"  {'-'*72}")

    agg_n = []
    agg_ppl_mean = []
    agg_ppl_std = []
    agg_loss_mean = []
    agg_improvement = []

    for N in data_sizes:
        entries = all_results[N]
        ppls = [e["val_ppl"] for e in entries]
        losses = [e["val_loss"] for e in entries]
        improvements = [e["ppl_improvement_pct"] for e in entries]

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls))
        mean_loss = float(np.mean(losses))
        mean_improvement = float(np.mean(improvements))

        # Mean base PPL for reference
        mean_base_ppl = float(np.mean([e["base_ppl"] for e in entries]))

        agg_n.append(N)
        agg_ppl_mean.append(mean_ppl)
        agg_ppl_std.append(std_ppl)
        agg_loss_mean.append(mean_loss)
        agg_improvement.append(mean_improvement)

        print(f"  {N:6d} | {mean_ppl:10.2f} | {std_ppl:10.2f} | "
              f"{mean_loss:11.4f} | {mean_improvement:+8.1f}% | "
              f"base={mean_base_ppl:.2f}")

    # ================================================================
    # Fit scaling law
    # ================================================================
    print(f"\n  --- Scaling Law Fit ---")
    fit = fit_power_law(agg_n, agg_ppl_mean)
    print(f"  PPL = {fit['a']:.2f} * N^({fit['slope']:.4f})")
    print(f"  alpha (decay rate): {fit['alpha']:.4f}")
    print(f"  R-squared: {fit['r_squared']:.4f}")

    # ================================================================
    # Saturation analysis
    # ================================================================
    print(f"\n  --- Saturation Analysis ---")
    sat = compute_saturation_point(agg_n, agg_ppl_mean, threshold_pct=5.0)

    print(f"\n  Step-by-step improvements:")
    for imp in sat["improvements"]:
        ratio = imp["data_ratio"]
        print(f"    {imp['from_n']:5d} -> {imp['to_n']:5d} "
              f"({ratio:.1f}x data): "
              f"PPL {imp['ppl_from']:.2f} -> {imp['ppl_to']:.2f} "
              f"({imp['rel_improvement_pct']:+.1f}%)")

    if sat["saturation_n"] is not None:
        print(f"\n  Saturation point: N={sat['saturation_n']} "
              f"(all subsequent improvements < {sat['threshold_pct']}%)")
    else:
        print(f"\n  No saturation detected at threshold {sat['threshold_pct']}%")

    # ================================================================
    # Marginal efficiency (PPL improvement per additional example)
    # ================================================================
    print(f"\n  --- Marginal Efficiency ---")
    print(f"  {'Interval':>15s} | {'PPL drop':>9s} | {'Per example':>11s} | "
          f"{'Per $0.01':>9s}")
    print(f"  {'-'*55}")

    for imp in sat["improvements"]:
        delta_n = imp["to_n"] - imp["from_n"]
        ppl_drop = imp["ppl_from"] - imp["ppl_to"]
        per_example = ppl_drop / delta_n if delta_n > 0 else 0
        # Cost model: ~$0.02/1000 examples (Groq batch pricing)
        cost = delta_n * 0.00002
        per_dollar_cent = ppl_drop / max(cost * 100, 1e-10)
        interval = f"{imp['from_n']}->{imp['to_n']}"
        print(f"  {interval:>15s} | {ppl_drop:9.2f} | {per_example:11.4f} | "
              f"{per_dollar_cent:9.2f}")

    # ================================================================
    # Kill criteria assessment
    # ================================================================
    print(f"\n{'='*72}")
    print("  KILL CRITERIA")
    print(f"{'='*72}")

    # K1: Quality does not improve beyond 200 examples
    # Check if PPL at 200 is essentially the same as at 5000
    ppl_200 = None
    ppl_5000 = None
    for i, N in enumerate(agg_n):
        if N == 200:
            ppl_200 = agg_ppl_mean[i]
        if N == 5000:
            ppl_5000 = agg_ppl_mean[i]

    if ppl_200 is not None and ppl_5000 is not None:
        improvement_200_to_5000 = (ppl_200 - ppl_5000) / ppl_200 * 100
        k1_killed = improvement_200_to_5000 < 2.0  # Less than 2% improvement
        print(f"\n  K1: Quality flat beyond N=200?")
        print(f"      PPL at N=200: {ppl_200:.2f}")
        print(f"      PPL at N=5000: {ppl_5000:.2f}")
        print(f"      Improvement 200->5000: {improvement_200_to_5000:.1f}%")
        print(f"      STATUS: {'KILL (flat beyond 200)' if k1_killed else 'PASS (still improving)'}")
    else:
        k1_killed = False
        print(f"\n  K1: Cannot evaluate (missing data points)")

    # K2: Quality still improving significantly at 5000
    # Check relative improvement from 2000 to 5000
    ppl_2000 = None
    for i, N in enumerate(agg_n):
        if N == 2000:
            ppl_2000 = agg_ppl_mean[i]

    if ppl_2000 is not None and ppl_5000 is not None:
        improvement_2000_to_5000 = (ppl_2000 - ppl_5000) / ppl_2000 * 100
        k2_killed = improvement_2000_to_5000 > 5.0  # More than 5% improvement
        print(f"\n  K2: Quality still improving significantly at N=5000?")
        print(f"      PPL at N=2000: {ppl_2000:.2f}")
        print(f"      PPL at N=5000: {ppl_5000:.2f}")
        print(f"      Improvement 2000->5000: {improvement_2000_to_5000:.1f}%")
        print(f"      STATUS: {'KILL (need >5000 examples)' if k2_killed else 'PASS (saturating)'}")
    else:
        k2_killed = False
        print(f"\n  K2: Cannot evaluate (missing data points)")

    overall_kill = k1_killed or k2_killed
    print(f"\n  OVERALL: {'KILL' if overall_kill else 'PASS'}")

    if not overall_kill:
        if sat["saturation_n"] is not None:
            print(f"  RECOMMENDATION: {sat['saturation_n']} examples per expert "
                  f"is the cost-optimal budget")
        else:
            print(f"  RECOMMENDATION: Curve still declining, consider "
                  f"N=1000-2000 as practical budget")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "experiment": "data_scaling_per_expert",
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "block_size": block_size,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "pretrain_steps": pretrain_steps,
            "expert_steps": expert_steps,
            "batch_size": batch_size,
            "lr": lr,
            "n_test": n_test,
            "domain": domain,
            "n_seeds": n_seeds,
            "data_sizes": data_sizes,
        },
        "aggregate": {
            "n_values": agg_n,
            "ppl_mean": agg_ppl_mean,
            "ppl_std": agg_ppl_std,
            "loss_mean": agg_loss_mean,
            "improvement_pct": agg_improvement,
        },
        "scaling_law": fit,
        "saturation": {
            "saturation_n": sat["saturation_n"],
            "threshold_pct": sat["threshold_pct"],
            "improvements": sat["improvements"],
        },
        "marginal_efficiency": [
            {
                "from_n": imp["from_n"],
                "to_n": imp["to_n"],
                "ppl_drop": imp["ppl_from"] - imp["ppl_to"],
                "per_example": (imp["ppl_from"] - imp["ppl_to"]) / max(imp["to_n"] - imp["from_n"], 1),
            }
            for imp in sat["improvements"]
        ],
        "kill_criteria": {
            "k1_flat_beyond_200": bool(k1_killed),
            "k1_improvement_200_5000_pct": float(improvement_200_to_5000) if ppl_200 and ppl_5000 else None,
            "k2_still_improving_at_5000": bool(k2_killed),
            "k2_improvement_2000_5000_pct": float(improvement_2000_to_5000) if ppl_2000 and ppl_5000 else None,
            "overall_kill": bool(overall_kill),
        },
        "per_size_per_seed": {str(N): entries for N, entries in all_results.items()},
        "total_time_s": total_time,
    }

    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  Results saved to {output_path}")
    print(f"  Total time: {total_time:.1f}s")

    return output


if __name__ == "__main__":
    import sys
    if "--fast" in sys.argv:
        run_experiment(
            data_sizes=[50, 200, 1000, 5000],
            n_seeds=2,
            pretrain_steps=500,
            expert_steps=150,
        )
    else:
        run_experiment()
