"""
Amplification Factor Scaling: Does the zero-shot amplification factor c
scale with model dimension d?

Parent: zero_shot_base_transfer (proven)
  - That experiment found c ~ 0.1 at d=64
  - This experiment measures c at d=64, d=128, d=256 to determine scaling

Hypothesis: The amplification factor c = (expert_loss_ratio - 1) / (base_loss_ratio - 1)
decreases with d, meaning zero-shot transfer gets SAFER at larger model dimensions.

Kill criteria:
  1. c grows linearly or faster with d (amplification worsens at scale)
  2. No measurable amplification trend across d=64, d=128, d=256

Architecture: Reuses parent's GPT, LoRA, and data infrastructure.
Scales model dimension while keeping other hyperparameters proportional.
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse parent's infrastructure
from experiments.models.base_free_composition.base_free_composition import (
    GPT,
    LoRALinear,
    LoRAGPT,
    CharTokenizer,
    CharDataset,
    load_names,
    domain_split,
    compute_delta,
    svd_truncate,
    reconstruct_with_delta,
    delta_reconstruction_error,
    effective_rank,
    train_gpt,
    evaluate_model,
    train_lora_expert,
)

from experiments.models.zero_shot_base_transfer.zero_shot_base_transfer import (
    apply_lora_deltas_to_base,
    evaluate_expert_zero_shot,
)


# ── Experiment Configuration per Dimension ─────────────────────────────────


def get_config_for_d(d: int) -> dict:
    """Return model and training config scaled for dimension d.

    We scale n_head proportionally (keeping head_dim=16),
    and scale training steps to ensure comparable convergence.
    LoRA rank stays fixed at 8 to isolate the d-scaling effect.
    """
    configs = {
        64: {
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 4,
            "block_size": 32,
            "lora_rank": 8,
            "lora_alpha": 1.0,
            "pretrain_steps": 1000,
            "expert_steps": 300,
            "n_experts": 4,
            "batch_size": 32,
            "lr": 3e-3,
            "delta_ranks": [32, 16, 8, 4],
        },
        128: {
            "n_embd": 128,
            "n_head": 8,
            "n_layer": 4,
            "block_size": 32,
            "lora_rank": 8,
            "lora_alpha": 1.0,
            "pretrain_steps": 1500,
            "expert_steps": 400,
            "n_experts": 4,
            "batch_size": 32,
            "lr": 2e-3,
            "delta_ranks": [64, 32, 16, 8],
        },
        256: {
            "n_embd": 256,
            "n_head": 16,
            "n_layer": 4,
            "block_size": 32,
            "lora_rank": 8,
            "lora_alpha": 1.0,
            "pretrain_steps": 2000,
            "expert_steps": 500,
            "n_experts": 4,
            "batch_size": 32,
            "lr": 1e-3,
            "delta_ranks": [128, 64, 32, 16],
        },
    }
    return configs[d]


# ── Core Measurement ───────────────────────────────────────────────────────


@dataclass
class AmplificationMeasurement:
    """Amplification factor measurement at one (d, SVD_rank, seed) point."""
    d: int
    svd_rank: int
    seed: int
    base_loss_pretrained: float
    base_loss_svd: float
    base_loss_ratio: float  # L_base_svd / L_base_pretrained
    expert_loss_pretrained: float  # mean across experts
    expert_loss_svd: float  # mean across experts, zero-shot
    expert_loss_ratio: float  # L_expert_svd / L_expert_pretrained
    amplification_factor: float  # (expert_loss_ratio - 1) / (base_loss_ratio - 1)
    expert_losses_pretrained: list  # per-expert
    expert_losses_svd: list  # per-expert


@dataclass
class DimensionResult:
    """Results for one model dimension d, aggregated across seeds."""
    d: int
    seeds: list
    measurements: list  # list of AmplificationMeasurement dicts
    # Per SVD rank: mean amplification factor across seeds
    rank_amplifications: dict  # {svd_rank: {"mean_c": ..., "std_c": ..., "values": [...]}}
    # Overall mean c for this d (across all ranks and seeds)
    mean_c: float
    std_c: float
    pretrain_val_loss: float  # mean pretrained val loss across seeds


def run_single_d(
    d: int,
    seeds: list = None,
    device: str = "cpu",
) -> DimensionResult:
    """Run the zero-shot amplification measurement at one dimension d."""
    if seeds is None:
        seeds = [42, 123, 7]

    cfg = get_config_for_d(d)
    print(f"\n{'='*72}")
    print(f"AMPLIFICATION FACTOR MEASUREMENT: d={d}")
    print(f"Config: h={cfg['n_head']}, L={cfg['n_layer']}, r={cfg['lora_rank']}")
    print(f"Training: {cfg['pretrain_steps']} pretrain, {cfg['expert_steps']} expert steps")
    print(f"Delta ranks: {cfg['delta_ranks']}")
    print(f"Seeds: {seeds}")
    print(f"{'='*72}")

    # Load data (shared across all dimensions)
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:cfg["n_experts"]]

    all_measurements = []
    pretrain_losses = []

    for seed in seeds:
        print(f"\n--- Seed {seed}, d={d} ---")
        torch.manual_seed(seed)

        # Global train/val split
        rng_split = random.Random(seed)
        docs_copy = list(docs)
        rng_split.shuffle(docs_copy)
        split_idx = int(len(docs_copy) * 0.9)
        pretrain_train_ds = CharDataset(docs_copy[:split_idx], tokenizer, cfg["block_size"])
        pretrain_val_ds = CharDataset(docs_copy[split_idx:], tokenizer, cfg["block_size"])

        # Phase 1: Pretrain
        torch.manual_seed(seed)
        model = GPT(vocab_size, cfg["block_size"], cfg["n_embd"],
                     cfg["n_head"], cfg["n_layer"])
        skeleton_state = {k: v.clone() for k, v in model.state_dict().items()}

        train_gpt(model, pretrain_train_ds,
                  steps=cfg["pretrain_steps"], batch_size=cfg["batch_size"],
                  lr=cfg["lr"], seed=seed, device=device)

        pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
        pretrained_val = evaluate_model(model, pretrain_val_ds,
                                        cfg["batch_size"], device=device)
        pretrain_losses.append(pretrained_val)
        print(f"  Pretrained val loss: {pretrained_val:.4f}")

        # Phase 2: Train experts
        domain_datasets = {}
        for i, domain_name in enumerate(domain_names):
            domain_docs = domains[domain_name]
            rng_domain = random.Random(seed + 1000 + i)
            domain_docs_shuffled = list(domain_docs)
            rng_domain.shuffle(domain_docs_shuffled)
            n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
            train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, cfg["block_size"])
            val_ds = CharDataset(
                domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
                else domain_docs_shuffled, tokenizer, cfg["block_size"]
            )
            domain_datasets[domain_name] = (train_ds, val_ds)

        expert_data = []
        for i, domain_name in enumerate(domain_names):
            train_ds, val_ds = domain_datasets[domain_name]
            expert_deltas, expert_val = train_lora_expert(
                model, train_ds, val_ds,
                rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
                steps=cfg["expert_steps"], batch_size=cfg["batch_size"],
                lr=cfg["lr"], seed=seed + i, device=device,
            )
            expert_data.append((domain_name, expert_deltas, expert_val, val_ds))
            print(f"  Expert {i} ({domain_name}): val_loss={expert_val:.4f}")

        ref_losses = [ed[2] for ed in expert_data]
        ref_mean = sum(ref_losses) / len(ref_losses)

        # Phase 3: SVD bases and zero-shot evaluation
        deltas = compute_delta(pretrained_state, skeleton_state)

        for svd_rank in cfg["delta_ranks"]:
            # Build SVD-reconstructed base
            svd_state = reconstruct_with_delta(
                skeleton_state, deltas, rank=svd_rank,
                pretrained_state=pretrained_state
            )

            # Evaluate base quality on SVD base
            base_model = GPT(vocab_size, cfg["block_size"], cfg["n_embd"],
                             cfg["n_head"], cfg["n_layer"])
            base_model.load_state_dict(svd_state)
            base_val_svd = evaluate_model(base_model, pretrain_val_ds,
                                          cfg["batch_size"], device=device)

            base_ratio = base_val_svd / (pretrained_val + 1e-12)

            # Zero-shot expert evaluation
            expert_losses_svd = []
            for i, (domain_name, expert_deltas, _, val_ds) in enumerate(expert_data):
                expert_loss = evaluate_expert_zero_shot(
                    svd_state, expert_deltas, val_ds,
                    vocab_size=vocab_size, block_size=cfg["block_size"],
                    n_embd=cfg["n_embd"], n_head=cfg["n_head"],
                    n_layer=cfg["n_layer"],
                    batch_size=cfg["batch_size"], device=device,
                )
                expert_losses_svd.append(expert_loss)

            mean_expert_svd = sum(expert_losses_svd) / len(expert_losses_svd)
            expert_ratio = mean_expert_svd / (ref_mean + 1e-12)

            # Compute amplification factor
            # c = (expert_loss_ratio - 1) / (base_loss_ratio - 1)
            # Only meaningful when base_loss_ratio > 1 (base quality degraded)
            base_excess = base_ratio - 1.0
            expert_excess = expert_ratio - 1.0

            if abs(base_excess) > 1e-6:
                c = expert_excess / base_excess
            else:
                c = float("nan")  # base unchanged, can't compute

            measurement = AmplificationMeasurement(
                d=d,
                svd_rank=svd_rank,
                seed=seed,
                base_loss_pretrained=pretrained_val,
                base_loss_svd=base_val_svd,
                base_loss_ratio=base_ratio,
                expert_loss_pretrained=ref_mean,
                expert_loss_svd=mean_expert_svd,
                expert_loss_ratio=expert_ratio,
                amplification_factor=c,
                expert_losses_pretrained=ref_losses,
                expert_losses_svd=expert_losses_svd,
            )
            all_measurements.append(measurement)

            print(f"  SVD rank {svd_rank:4d}: base_ratio={base_ratio:.4f}, "
                  f"expert_ratio={expert_ratio:.4f}, c={c:.4f}")

    # Aggregate per SVD rank
    rank_amplifications = {}
    for svd_rank in cfg["delta_ranks"]:
        c_values = [
            m.amplification_factor
            for m in all_measurements
            if m.svd_rank == svd_rank and not math.isnan(m.amplification_factor)
        ]
        if c_values:
            mean_c = sum(c_values) / len(c_values)
            std_c = (sum((v - mean_c) ** 2 for v in c_values) / len(c_values)) ** 0.5
        else:
            mean_c = float("nan")
            std_c = float("nan")
        rank_amplifications[svd_rank] = {
            "mean_c": mean_c,
            "std_c": std_c,
            "values": c_values,
        }

    # Overall mean c
    all_c = [
        m.amplification_factor
        for m in all_measurements
        if not math.isnan(m.amplification_factor)
    ]
    overall_mean_c = sum(all_c) / len(all_c) if all_c else float("nan")
    overall_std_c = (
        (sum((v - overall_mean_c) ** 2 for v in all_c) / len(all_c)) ** 0.5
        if all_c else float("nan")
    )

    result = DimensionResult(
        d=d,
        seeds=seeds,
        measurements=[asdict(m) for m in all_measurements],
        rank_amplifications={str(k): v for k, v in rank_amplifications.items()},
        mean_c=overall_mean_c,
        std_c=overall_std_c,
        pretrain_val_loss=sum(pretrain_losses) / len(pretrain_losses),
    )

    return result


# ── Scaling Law Analysis ───────────────────────────────────────────────────


def fit_power_law(d_values: list, c_values: list) -> dict:
    """Fit c = a * d^alpha via log-log linear regression.

    Returns dict with alpha (exponent), a (coefficient), r_squared.
    """
    log_d = np.log(np.array(d_values, dtype=float))
    log_c = np.log(np.array(c_values, dtype=float))

    # Filter out NaN/inf
    valid = np.isfinite(log_d) & np.isfinite(log_c)
    log_d = log_d[valid]
    log_c = log_c[valid]

    if len(log_d) < 2:
        return {"alpha": float("nan"), "a": float("nan"), "r_squared": float("nan")}

    # Linear regression: log(c) = log(a) + alpha * log(d)
    n = len(log_d)
    mean_x = np.mean(log_d)
    mean_y = np.mean(log_c)
    ss_xx = np.sum((log_d - mean_x) ** 2)
    ss_xy = np.sum((log_d - mean_x) * (log_c - mean_y))
    ss_yy = np.sum((log_c - mean_y) ** 2)

    if ss_xx < 1e-12:
        return {"alpha": float("nan"), "a": float("nan"), "r_squared": float("nan")}

    alpha = ss_xy / ss_xx
    log_a = mean_y - alpha * mean_x
    a = np.exp(log_a)

    # R-squared
    ss_res = np.sum((log_c - (log_a + alpha * log_d)) ** 2)
    r_squared = 1 - ss_res / ss_yy if ss_yy > 1e-12 else float("nan")

    return {
        "alpha": float(alpha),
        "a": float(a),
        "r_squared": float(r_squared),
        "n_points": int(n),
    }


def analyze_scaling(dim_results: list) -> dict:
    """Analyze amplification factor scaling across dimensions.

    Fits power law c(d) = a * d^alpha and evaluates kill criteria.
    """
    d_values = [r.d for r in dim_results]
    c_values = [r.mean_c for r in dim_results]

    # Fit overall power law
    overall_fit = fit_power_law(d_values, c_values)

    # Also fit per-SVD-rank power laws
    # Collect unique SVD ranks that appear at all dimensions
    # (ranks are dimension-specific, so we match by rank/d ratio)
    per_rank_fits = {}

    # Use matching SVD rank fractions: rank/d ratios
    # At d=64: {32, 16, 8, 4} -> ratios {0.5, 0.25, 0.125, 0.0625}
    # At d=128: {64, 32, 16, 8} -> ratios {0.5, 0.25, 0.125, 0.0625}
    # At d=256: {128, 64, 32, 16} -> ratios {0.5, 0.25, 0.125, 0.0625}
    rank_ratios = [0.5, 0.25, 0.125, 0.0625]

    for ratio in rank_ratios:
        ratio_d_vals = []
        ratio_c_vals = []
        for r in dim_results:
            svd_rank = int(r.d * ratio)
            rank_str = str(svd_rank)
            if rank_str in r.rank_amplifications:
                ra = r.rank_amplifications[rank_str]
                if not math.isnan(ra["mean_c"]):
                    ratio_d_vals.append(r.d)
                    ratio_c_vals.append(ra["mean_c"])

        if len(ratio_d_vals) >= 2:
            fit = fit_power_law(ratio_d_vals, ratio_c_vals)
            per_rank_fits[f"ratio_{ratio}"] = {
                "d_values": ratio_d_vals,
                "c_values": ratio_c_vals,
                "fit": fit,
            }

    # Kill criteria evaluation
    alpha = overall_fit["alpha"]

    # K1: c grows linearly or faster with d (alpha >= 1)
    k1_killed = alpha >= 1.0 if not math.isnan(alpha) else False

    # K2: no measurable trend (check if c values are essentially flat)
    # We define "no trend" as: all c values within 20% of each other
    if all(not math.isnan(c) for c in c_values) and len(c_values) >= 2:
        c_range = max(c_values) - min(c_values)
        c_mean = sum(c_values) / len(c_values)
        k2_killed = c_range / (c_mean + 1e-12) < 0.2  # less than 20% variation
    else:
        k2_killed = True  # can't measure

    # Determine verdict
    if k1_killed:
        verdict = "KILLED (K1: amplification grows >= linearly with d)"
    elif k2_killed:
        verdict = "KILLED (K2: no measurable trend)"
    else:
        if alpha < 0:
            verdict = f"SURVIVES: amplification DECREASES with d (alpha={alpha:.3f})"
        else:
            verdict = f"SURVIVES: amplification grows sub-linearly (alpha={alpha:.3f})"

    return {
        "d_values": d_values,
        "c_values": c_values,
        "overall_fit": overall_fit,
        "per_rank_fits": per_rank_fits,
        "k1_killed": k1_killed,
        "k2_killed": k2_killed,
        "verdict": verdict,
    }


# ── Main Experiment ────────────────────────────────────────────────────────


def run_experiment(
    dimensions: list = None,
    seeds: list = None,
    device: str = "cpu",
) -> dict:
    """Run the full amplification factor scaling experiment.

    For each d in {64, 128, 256}, runs zero-shot transfer at multiple
    SVD ranks and seeds, measures amplification factor c, fits scaling law.
    """
    if dimensions is None:
        dimensions = [64, 128, 256]
    if seeds is None:
        seeds = [42, 123, 7]

    t0 = time.time()

    print("=" * 72)
    print("AMPLIFICATION FACTOR SCALING EXPERIMENT")
    print(f"Dimensions: {dimensions}")
    print(f"Seeds: {seeds}")
    print("=" * 72)

    dim_results = []
    for d in dimensions:
        result = run_single_d(d, seeds=seeds, device=device)
        dim_results.append(result)

    total_time = time.time() - t0

    # Analyze scaling
    scaling = analyze_scaling(dim_results)

    # Print summary
    print(f"\n{'='*72}")
    print("AMPLIFICATION FACTOR SCALING SUMMARY")
    print(f"{'='*72}")

    print(f"\n{'d':>6s} | {'mean_c':>8s} | {'std_c':>8s} | {'pretrain_loss':>13s}")
    print("-" * 45)
    for r in dim_results:
        print(f"{r.d:6d} | {r.mean_c:8.4f} | {r.std_c:8.4f} | {r.pretrain_val_loss:13.4f}")

    print(f"\nPower law fit: c(d) = {scaling['overall_fit']['a']:.4f} * d^{scaling['overall_fit']['alpha']:.4f}")
    print(f"R-squared: {scaling['overall_fit']['r_squared']:.4f}")

    print(f"\nPer-rank-ratio fits:")
    for ratio_key, ratio_data in scaling["per_rank_fits"].items():
        fit = ratio_data["fit"]
        print(f"  {ratio_key}: alpha={fit['alpha']:.4f}, R2={fit['r_squared']:.4f}, "
              f"c_values={[f'{v:.4f}' for v in ratio_data['c_values']]}")

    print(f"\nKill criteria:")
    print(f"  K1 (c grows >= linearly): {'KILLED' if scaling['k1_killed'] else 'SURVIVES'}")
    print(f"  K2 (no measurable trend): {'KILLED' if scaling['k2_killed'] else 'SURVIVES'}")
    print(f"\nVERDICT: {scaling['verdict']}")
    print(f"\nTotal time: {total_time:.1f}s")

    # Save results
    output = {
        "dimensions": [asdict_safe(r) for r in dim_results],
        "scaling_analysis": scaling,
        "total_time": total_time,
        "verdict": scaling["verdict"],
    }

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


def asdict_safe(obj):
    """Convert dataclass to dict, handling nested structures."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


if __name__ == "__main__":
    run_experiment()
