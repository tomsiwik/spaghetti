"""
Few-Shot Adaptation Recovery: Can a few adaptation steps close the
zero-shot transfer gap?

Parent: zero_shot_base_transfer (proven)
  - That experiment showed 2.8% transfer gap at rank-16, growing to 22.6% at rank-4.
  - This experiment tests whether 1-50 LoRA adaptation steps on the new base
    can close that gap cheaply, without full expert retraining.

Hypothesis: A small number of adaptation steps (10-50) on the perturbed base
can reduce the zero-shot transfer gap by >50%, making base swapping nearly free:
deploy zero-shot immediately, then cheaply adapt in background.

Kill criteria:
  K1: 50 adaptation steps don't reduce transfer gap by >50% (adaptation too slow)
  K2: Adapted expert quality worse than zero-shot on original base (adaptation hurts)

Architecture: Reuses parent's GPT, LoRA, and data infrastructure.
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

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
    compute_pairwise_cosine,
    train_gpt,
    evaluate_model,
    train_lora_expert,
)

from experiments.models.zero_shot_base_transfer.zero_shot_base_transfer import (
    apply_lora_deltas_to_base,
    evaluate_expert_zero_shot,
)


# -- Adaptation: Fine-tune LoRA expert on new base for N steps ----------------


def inject_lora_deltas_as_trainable(
    base_gpt: GPT,
    lora_deltas: list,
    rank: int = 8,
    alpha: float = 1.0,
) -> LoRAGPT:
    """Create a LoRAGPT with A,B initialized from pre-trained deltas.

    Instead of standard Kaiming/zero init, we decompose each existing
    delta back into A,B via SVD so adaptation starts from the trained
    expert state rather than from scratch.

    Args:
        base_gpt: The NEW base model (perturbed)
        lora_deltas: List of (layer_idx, fc_name, delta_tensor) from training
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        A LoRAGPT where A,B are initialized to reproduce the original delta.
    """
    base_copy = copy.deepcopy(base_gpt)
    lora_model = LoRAGPT(base_copy, rank=rank, alpha=alpha)

    scale = alpha / rank

    # For each delta, decompose via SVD and initialize A, B
    for layer_idx, fc_name, delta in lora_deltas:
        # delta shape: (in_features, out_features)
        # LoRA computes: (x @ A @ B) * scale
        # So delta = A @ B * scale, meaning A @ B = delta / scale
        target = delta / scale

        # SVD decomposition: target = U @ S @ V^T
        # We want A (in, r) and B (r, out) such that A @ B approx= target
        U, S, Vt = torch.linalg.svd(target, full_matrices=False)

        # Take top-r components
        r = min(rank, U.shape[1])
        # A = U[:, :r] @ diag(sqrt(S[:r]))
        # B = diag(sqrt(S[:r])) @ Vt[:r, :]
        sqrt_s = torch.sqrt(S[:r])
        A_init = U[:, :r] * sqrt_s.unsqueeze(0)  # (in, r)
        B_init = Vt[:r, :] * sqrt_s.unsqueeze(1)  # (r, out)

        # Set the LoRA parameters
        lora_layer = lora_model.lora_layers[layer_idx]
        lora_linear = lora_layer[fc_name]
        with torch.no_grad():
            lora_linear.A.copy_(A_init)
            lora_linear.B.copy_(B_init)

    return lora_model


def adapt_expert(
    base_state: dict,
    lora_deltas: list,
    train_ds: CharDataset,
    val_ds: CharDataset,
    adapt_steps: int,
    rank: int = 8,
    alpha: float = 1.0,
    batch_size: int = 32,
    lr: float = 3e-3,
    vocab_size: int = 42,
    block_size: int = 32,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> tuple:
    """Adapt a pre-trained expert to a new base model for N steps.

    Returns: (adapted_deltas, adapted_val_loss, loss_curve)
    """
    # Build base model from state
    base_model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
    base_model.load_state_dict(base_state)

    # Create LoRA model initialized from pre-trained deltas
    lora_model = inject_lora_deltas_as_trainable(
        base_model, lora_deltas, rank=rank, alpha=alpha
    )
    lora_model.to(device)
    lora_model.train()

    rng = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)

    loss_curve = []

    for step in range(1, adapt_steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng, device)
        logits = lora_model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())

    # Evaluate
    val_loss = evaluate_model(lora_model, val_ds, batch_size, device=device)
    adapted_deltas = lora_model.get_all_deltas()

    return adapted_deltas, val_loss, loss_curve


# -- Experiment ---------------------------------------------------------------


@dataclass
class AdaptationResult:
    """Results for one expert at one adaptation budget."""
    domain: str
    adapt_steps: int
    # Losses
    zeroshot_loss: float       # expert on new base, zero-shot (no adaptation)
    adapted_loss: float        # expert on new base, after N adaptation steps
    retrained_loss: float      # expert retrained from scratch on new base
    original_loss: float       # expert on original base (reference)
    # Derived
    zs_gap: float              # zeroshot_loss - original_loss (the transfer gap)
    adapted_gap: float         # adapted_loss - original_loss
    gap_reduction_pct: float   # (zs_gap - adapted_gap) / zs_gap * 100
    # K2: quality on original base after adaptation
    adapted_on_original_loss: float  # adapted expert evaluated on ORIGINAL base


@dataclass
class ExperimentResults:
    """Complete experiment results."""
    per_expert: list           # list of AdaptationResult dicts
    aggregate: dict            # aggregated metrics
    kill_criteria: dict        # K1, K2 evaluation
    config: dict
    seed: int
    timing: dict


def run_experiment(
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    block_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    total_pretrain_steps: int = 1000,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
    adapt_lr: float = 1e-3,
    adapt_steps_list: list = None,
    delta_rank: int = 16,
    seed: int = 42,
    device: str = "cpu",
) -> ExperimentResults:
    """Run the few-shot adaptation recovery experiment.

    Protocol:
      1. Pretrain a micro GPT (the "pretrained base")
      2. Train N experts on the pretrained base (standard LoRA training)
      3. Build SVD-perturbed base at rank delta_rank
      4. Evaluate zero-shot transfer gap (baseline)
      5. For each adaptation budget (1, 5, 10, 25, 50 steps):
         a. Initialize LoRA from pre-trained deltas via SVD decomposition
         b. Fine-tune on new base for N steps
         c. Measure adapted quality on new base
         d. Measure adapted quality on ORIGINAL base (K2 check)
      6. Also retrain from scratch as upper-bound reference
    """
    if adapt_steps_list is None:
        adapt_steps_list = [1, 5, 10, 25, 50]

    print("=" * 72)
    print("FEW-SHOT ADAPTATION RECOVERY EXPERIMENT")
    print(f"Config: d={n_embd}, h={n_head}, L={n_layer}, r={lora_rank}")
    print(f"Pretrain: {total_pretrain_steps} steps, Expert train: {expert_train_steps} steps")
    print(f"Base perturbation: SVD rank-{delta_rank}")
    print(f"Adaptation budgets: {adapt_steps_list}")
    print(f"Adaptation LR: {adapt_lr}")
    print(f"Seed: {seed}")
    print("=" * 72)

    torch.manual_seed(seed)
    t0 = time.time()

    # -- Load data -----------------------------------------------------------
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:n_experts]
    print(f"\nDomains: {domain_names}")

    # Global train/val split
    rng_split = random.Random(seed)
    docs_copy = list(docs)
    rng_split.shuffle(docs_copy)
    split_idx = int(len(docs_copy) * 0.9)
    pretrain_train_ds = CharDataset(docs_copy[:split_idx], tokenizer, block_size)
    pretrain_val_ds = CharDataset(docs_copy[split_idx:], tokenizer, block_size)

    # -- Phase 1: Pretrain base model ----------------------------------------
    print("\n--- Phase 1: Pretraining ---")
    torch.manual_seed(seed)
    pretrained_model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
    skeleton_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    train_gpt(pretrained_model, pretrain_train_ds,
              steps=total_pretrain_steps, batch_size=batch_size,
              lr=lr, seed=seed, device=device)

    pretrained_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
    pretrained_val = evaluate_model(pretrained_model, pretrain_val_ds, batch_size, device=device)
    print(f"  Pretrained val loss: {pretrained_val:.4f}")

    pretrain_time = time.time() - t0

    # -- Phase 2: Train experts on pretrained base ---------------------------
    print("\n--- Phase 2: Training Experts ---")
    t1 = time.time()

    domain_datasets = {}
    for i, domain in enumerate(domain_names):
        domain_docs = domains[domain]
        rng_domain = random.Random(seed + 1000 + i)
        domain_docs_shuffled = list(domain_docs)
        rng_domain.shuffle(domain_docs_shuffled)
        n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
        train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, block_size)
        val_ds = CharDataset(
            domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
            else domain_docs_shuffled, tokenizer, block_size
        )
        domain_datasets[domain] = (train_ds, val_ds)

    expert_data = []  # (domain, deltas, val_loss, train_ds, val_ds)
    for i, domain in enumerate(domain_names):
        train_ds, val_ds = domain_datasets[domain]
        expert_deltas, expert_val = train_lora_expert(
            pretrained_model, train_ds, val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, device=device,
        )
        expert_data.append((domain, expert_deltas, expert_val, train_ds, val_ds))
        print(f"  Expert {i} ({domain}): val_loss={expert_val:.4f}")

    expert_train_time = time.time() - t1

    # -- Phase 3: Build SVD-perturbed base -----------------------------------
    print(f"\n--- Phase 3: SVD-Perturbed Base (rank-{delta_rank}) ---")
    deltas = compute_delta(pretrained_state, skeleton_state)
    perturbed_state = reconstruct_with_delta(
        skeleton_state, deltas, rank=delta_rank,
        pretrained_state=pretrained_state
    )

    errs = delta_reconstruction_error(deltas, delta_rank)
    rms_err = errs["_total"]["rms_relative_error"]
    print(f"  Base reconstruction error: {rms_err:.4f}")

    # Evaluate perturbed base
    perturbed_model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
    perturbed_model.load_state_dict(perturbed_state)
    perturbed_val = evaluate_model(perturbed_model, pretrain_val_ds, batch_size, device=device)
    print(f"  Perturbed base val loss: {perturbed_val:.4f} (ratio: {perturbed_val/pretrained_val:.4f})")

    # -- Phase 4: Zero-shot baseline -----------------------------------------
    print("\n--- Phase 4: Zero-Shot Transfer Baseline ---")
    zs_losses = []
    for i, (domain, expert_deltas, orig_val, train_ds, val_ds) in enumerate(expert_data):
        zs_loss = evaluate_expert_zero_shot(
            perturbed_state, expert_deltas, val_ds,
            vocab_size=vocab_size, block_size=block_size,
            n_embd=n_embd, n_head=n_head, n_layer=n_layer,
            batch_size=batch_size, device=device,
        )
        zs_losses.append(zs_loss)
        gap = zs_loss - orig_val
        gap_pct = gap / orig_val * 100
        print(f"  Expert {i} ({domain}): ZS loss={zs_loss:.4f}, "
              f"orig={orig_val:.4f}, gap={gap_pct:.2f}%")

    # -- Phase 5: Retrained upper bound --------------------------------------
    print("\n--- Phase 5: Retrained Upper Bound ---")
    retrained_losses = []
    for i, (domain, _, orig_val, train_ds, val_ds) in enumerate(expert_data):
        # Train fresh expert on perturbed base
        perturbed_base = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
        perturbed_base.load_state_dict(perturbed_state)
        _, retrained_val = train_lora_expert(
            perturbed_base, train_ds, val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, device=device,
        )
        retrained_losses.append(retrained_val)
        gap = retrained_val - orig_val
        gap_pct = gap / orig_val * 100
        print(f"  Expert {i} ({domain}): retrained loss={retrained_val:.4f}, "
              f"gap from orig={gap_pct:.2f}%")

    # -- Phase 6: Adaptation at various step budgets -------------------------
    print("\n--- Phase 6: Few-Shot Adaptation ---")
    all_results = []

    for adapt_steps in adapt_steps_list:
        print(f"\n  Adaptation budget: {adapt_steps} steps")
        for i, (domain, expert_deltas, orig_val, train_ds, val_ds) in enumerate(expert_data):
            # Adapt expert on perturbed base
            adapted_deltas, adapted_loss, loss_curve = adapt_expert(
                perturbed_state, expert_deltas, train_ds, val_ds,
                adapt_steps=adapt_steps,
                rank=lora_rank, alpha=lora_alpha,
                batch_size=batch_size, lr=adapt_lr,
                vocab_size=vocab_size, block_size=block_size,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                seed=seed + 500 + i,
                device=device,
            )

            # K2 check: evaluate adapted expert on ORIGINAL base
            adapted_on_original = evaluate_expert_zero_shot(
                pretrained_state, adapted_deltas, val_ds,
                vocab_size=vocab_size, block_size=block_size,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                batch_size=batch_size, device=device,
            )

            zs_gap = zs_losses[i] - orig_val
            adapted_gap = adapted_loss - orig_val
            gap_reduction = (zs_gap - adapted_gap) / (abs(zs_gap) + 1e-12) * 100

            result = AdaptationResult(
                domain=domain,
                adapt_steps=adapt_steps,
                zeroshot_loss=zs_losses[i],
                adapted_loss=adapted_loss,
                retrained_loss=retrained_losses[i],
                original_loss=orig_val,
                zs_gap=zs_gap,
                adapted_gap=adapted_gap,
                gap_reduction_pct=gap_reduction,
                adapted_on_original_loss=adapted_on_original,
            )
            all_results.append(result)

            print(f"    Expert {i} ({domain}): adapted={adapted_loss:.4f}, "
                  f"gap_reduction={gap_reduction:.1f}%, "
                  f"on_orig={adapted_on_original:.4f} "
                  f"(vs ZS orig={orig_val:.4f})")

    total_time = time.time() - t0

    # -- Phase 7: Aggregate and Kill Criteria --------------------------------
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Group by adaptation steps
    aggregate = {}
    for steps in adapt_steps_list:
        step_results = [r for r in all_results if r.adapt_steps == steps]
        mean_gap_reduction = sum(r.gap_reduction_pct for r in step_results) / len(step_results)
        mean_adapted_loss = sum(r.adapted_loss for r in step_results) / len(step_results)
        mean_zs_loss = sum(r.zeroshot_loss for r in step_results) / len(step_results)
        mean_retrained_loss = sum(r.retrained_loss for r in step_results) / len(step_results)
        mean_original_loss = sum(r.original_loss for r in step_results) / len(step_results)
        mean_adapted_on_orig = sum(r.adapted_on_original_loss for r in step_results) / len(step_results)

        # How much of the retrained-ZS gap does adaptation recover?
        total_recoverable = mean_zs_loss - mean_retrained_loss
        adapted_recovery = mean_zs_loss - mean_adapted_loss
        recovery_of_retrained_pct = (adapted_recovery / (total_recoverable + 1e-12)) * 100

        # Adapted loss ratio relative to original
        adapted_loss_ratio = mean_adapted_loss / mean_original_loss
        zs_loss_ratio = mean_zs_loss / mean_original_loss
        retrained_loss_ratio = mean_retrained_loss / mean_original_loss

        # K2: adapted on original vs original
        k2_ratio = mean_adapted_on_orig / mean_original_loss

        aggregate[steps] = {
            "mean_gap_reduction_pct": mean_gap_reduction,
            "mean_adapted_loss": mean_adapted_loss,
            "mean_zs_loss": mean_zs_loss,
            "mean_retrained_loss": mean_retrained_loss,
            "mean_original_loss": mean_original_loss,
            "adapted_loss_ratio": adapted_loss_ratio,
            "zs_loss_ratio": zs_loss_ratio,
            "retrained_loss_ratio": retrained_loss_ratio,
            "recovery_of_retrained_pct": recovery_of_retrained_pct,
            "k2_adapted_on_orig_ratio": k2_ratio,
        }

    # Print table
    print(f"\n{'Steps':>6s} | {'ZS Ratio':>9s} | {'Adapted Ratio':>14s} | "
          f"{'Retrained Ratio':>16s} | {'Gap Reduction':>14s} | "
          f"{'Recovery vs RT':>14s} | {'K2 (on orig)':>12s}")
    print("-" * 100)
    for steps in adapt_steps_list:
        a = aggregate[steps]
        print(f"{steps:6d} | {a['zs_loss_ratio']:9.4f} | {a['adapted_loss_ratio']:14.4f} | "
              f"{a['retrained_loss_ratio']:16.4f} | {a['mean_gap_reduction_pct']:13.1f}% | "
              f"{a['recovery_of_retrained_pct']:13.1f}% | {a['k2_adapted_on_orig_ratio']:12.4f}")

    # Kill Criteria
    best_steps = max(adapt_steps_list)
    best_agg = aggregate[best_steps]

    k1_gap_reduction = best_agg["mean_gap_reduction_pct"]
    k1_verdict = "KILLED" if k1_gap_reduction < 50.0 else "SURVIVES"

    # K2: any adapted expert worse than ZS on original base?
    k2_worst_ratio = max(
        r.adapted_on_original_loss / r.original_loss
        for r in all_results if r.adapt_steps == best_steps
    )
    # "Worse than zero-shot on original base" means ratio > 1.0 + some margin
    # The zero-shot loss on original base IS the original_loss (they were trained there)
    # So K2 asks: does adaptation on new base hurt quality on old base?
    k2_verdict = "KILLED" if k2_worst_ratio > 1.0 + 0.001 else "SURVIVES"

    # Also check at all step counts for K2
    k2_any_hurt = False
    for steps in adapt_steps_list:
        for r in all_results:
            if r.adapt_steps == steps:
                ratio = r.adapted_on_original_loss / r.original_loss
                if ratio > 1.05:  # 5% degradation threshold for "hurts"
                    k2_any_hurt = True

    print(f"\n{'='*72}")
    print("KILL CRITERIA EVALUATION")
    print(f"{'='*72}")
    print(f"\n  K1: 50 steps must reduce gap by >50%")
    print(f"      Gap reduction at {best_steps} steps: {k1_gap_reduction:.1f}%")
    print(f"      Verdict: {k1_verdict}")

    print(f"\n  K2: Adapted expert must not be worse on original base")
    print(f"      Worst adapted/original ratio at {best_steps} steps: {k2_worst_ratio:.4f}")
    print(f"      Any expert >5% worse on original at any step count: {k2_any_hurt}")
    print(f"      Verdict: {k2_verdict}")

    if k1_verdict == "KILLED" or k2_verdict == "KILLED":
        overall = "KILLED"
    else:
        overall = "SURVIVES"

    print(f"\n  OVERALL VERDICT: {overall}")

    kill_criteria = {
        "k1_gap_reduction_pct": k1_gap_reduction,
        "k1_threshold": 50.0,
        "k1_verdict": k1_verdict,
        "k2_worst_ratio": k2_worst_ratio,
        "k2_any_expert_hurt_5pct": k2_any_hurt,
        "k2_verdict": k2_verdict,
        "overall_verdict": overall,
    }

    config = {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "lora_rank": lora_rank,
        "lora_alpha": lora_alpha, "total_pretrain_steps": total_pretrain_steps,
        "expert_train_steps": expert_train_steps, "n_experts": n_experts,
        "delta_rank": delta_rank, "adapt_steps_list": adapt_steps_list,
        "adapt_lr": adapt_lr, "lr": lr,
    }

    timing = {
        "pretrain_time": pretrain_time,
        "expert_train_time": expert_train_time,
        "total_time": total_time,
    }

    results = ExperimentResults(
        per_expert=[asdict(r) for r in all_results],
        aggregate={str(k): v for k, v in aggregate.items()},
        kill_criteria=kill_criteria,
        config=config,
        seed=seed,
        timing=timing,
    )

    output_path = os.path.join(os.path.dirname(__file__), f"results_seed_{seed}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_multi_seed(seeds: list = None, **kwargs) -> dict:
    """Run experiment across multiple seeds and aggregate."""
    if seeds is None:
        seeds = [42, 123, 7]

    all_results = []
    per_seed = {}

    for s in seeds:
        print(f"\n{'='*72}")
        print(f"SEED {s}")
        print(f"{'='*72}")
        r = run_experiment(seed=s, **kwargs)
        all_results.append(r)
        per_seed[str(s)] = {
            "verdict": r.kill_criteria["overall_verdict"],
            "aggregate": r.aggregate,
            "kill_criteria": r.kill_criteria,
        }

    # Aggregate across seeds
    adapt_steps_list = all_results[0].config["adapt_steps_list"]
    agg_across_seeds = {}

    for steps in adapt_steps_list:
        key = str(steps)
        gap_reductions = [r.aggregate[key]["mean_gap_reduction_pct"] for r in all_results]
        adapted_ratios = [r.aggregate[key]["adapted_loss_ratio"] for r in all_results]
        recovery_pcts = [r.aggregate[key]["recovery_of_retrained_pct"] for r in all_results]
        k2_ratios = [r.aggregate[key]["k2_adapted_on_orig_ratio"] for r in all_results]

        mean_gap_red = sum(gap_reductions) / len(gap_reductions)
        std_gap_red = (sum((g - mean_gap_red) ** 2 for g in gap_reductions) / len(gap_reductions)) ** 0.5
        mean_adapted = sum(adapted_ratios) / len(adapted_ratios)
        mean_recovery = sum(recovery_pcts) / len(recovery_pcts)
        mean_k2 = sum(k2_ratios) / len(k2_ratios)

        agg_across_seeds[key] = {
            "mean_gap_reduction_pct": mean_gap_red,
            "std_gap_reduction_pct": std_gap_red,
            "mean_adapted_loss_ratio": mean_adapted,
            "mean_recovery_of_retrained_pct": mean_recovery,
            "mean_k2_ratio": mean_k2,
            "per_seed_gap_reductions": gap_reductions,
        }

    # Overall kill criteria across seeds
    verdicts = [r.kill_criteria["overall_verdict"] for r in all_results]
    if all(v == "SURVIVES" for v in verdicts):
        overall = "SURVIVES"
    elif any(v == "KILLED" for v in verdicts):
        overall = "KILLED"
    else:
        overall = "INCONCLUSIVE"

    best_steps = str(max(adapt_steps_list))
    k1_values = [r.kill_criteria["k1_gap_reduction_pct"] for r in all_results]
    k2_values = [r.kill_criteria["k2_worst_ratio"] for r in all_results]

    aggregate = {
        "seeds": per_seed,
        "aggregate_across_seeds": agg_across_seeds,
        "overall_verdict": overall,
        "k1_per_seed": k1_values,
        "k1_mean": sum(k1_values) / len(k1_values),
        "k2_per_seed": k2_values,
        "k2_mean": sum(k2_values) / len(k2_values),
        "config": all_results[0].config,
    }

    # Print aggregate summary
    print(f"\n{'='*72}")
    print("AGGREGATE RESULTS (across all seeds)")
    print(f"{'='*72}")

    print(f"\n{'Steps':>6s} | {'Gap Reduction':>14s} | {'Std':>8s} | "
          f"{'Recovery vs RT':>14s} | {'K2 Ratio':>10s}")
    print("-" * 65)
    for steps in adapt_steps_list:
        key = str(steps)
        a = agg_across_seeds[key]
        print(f"{steps:6d} | {a['mean_gap_reduction_pct']:13.1f}% | "
              f"{a['std_gap_reduction_pct']:8.1f} | "
              f"{a['mean_recovery_of_retrained_pct']:13.1f}% | "
              f"{a['mean_k2_ratio']:10.4f}")

    print(f"\nK1 gap reduction at {best_steps} steps per seed: {k1_values}")
    print(f"K1 mean: {aggregate['k1_mean']:.1f}% (threshold: >50%)")
    print(f"K2 worst ratio per seed: {k2_values}")
    print(f"K2 mean: {aggregate['k2_mean']:.4f}")
    print(f"\nOverall verdict: {overall}")

    output_path = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregate saved to {output_path}")

    return aggregate


if __name__ == "__main__":
    run_multi_seed()
