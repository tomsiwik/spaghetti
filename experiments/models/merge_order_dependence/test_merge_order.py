"""Merge Order Dependence Experiment for Gram-Schmidt LoRA Composition.

Tests whether the ordering of experts in Gram-Schmidt orthogonalization
affects final merged model quality. Two regimes:

Phase 1: Natural LoRA experts (d=64, N=5 and N=8) -- near-orthogonal regime
  - Train real LoRA experts on domain splits
  - Apply GS in 20 random orderings
  - Measure quality variance across orderings

Phase 2: Synthetic high-overlap experts -- stress test regime
  - Create synthetic expert deltas with controlled cosine similarity (0.1 to 0.7)
  - Apply GS in 20 random orderings
  - Measure how order dependence scales with overlap

Phase 3: Order-invariant alternatives
  - Compare GS against SVD-based simultaneous orthogonalization
  - Symmetric Gram-Schmidt (average of all orderings)
  - Determine if alternatives are needed

Kill Criteria (from HYPOTHESES.yml):
  K1: Quality variance across 10 random orderings >5%
  K2: Worst ordering >15% worse than best ordering
"""

import copy
import itertools
import random
import statistics
import time
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss
from experiments.models.lora_merging_bakeoff.merging_methods import (
    extract_deltas,
    merge_simple_average,
    apply_merged_deltas,
)
from experiments.models.lora_procrustes.test_lora_procrustes import (
    freeze_except_lora,
)
from experiments.models.gram_schmidt_composition.gram_schmidt import (
    gram_schmidt_orthogonalize,
    merge_gs_average,
    merge_naive_sum,
    cosine_sim,
    flatten_delta_dict,
    unflatten_delta_dict,
)

# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3
N_ORDERINGS = 20  # Number of random orderings to test


# ── Helpers ─────────────────────────────────────────────────────────────────

def pretrain_base(joint_train, vocab_size, seed):
    """Pretrain base GPT model on joint data."""
    base = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base.parameters())
    train(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
          lr=LR, seed=seed, log_every=300)
    return base


def finetune_lora(base_model, train_ds, val_ds, vocab_size, seed):
    """Fine-tune a LoRA model on a single domain."""
    lora = get_model("lora_gpt", vocab_size=vocab_size, **BASE,
                     lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora.parameters())

    # Copy base weights into LoRA model
    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora.layers[l_idx]
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora.wte.weight = base_model.wte.weight
    lora.wpe.weight = base_model.wpe.weight
    lora.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora.parameters())

    freeze_except_lora(lora)
    train(lora, train_ds, val_ds, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
    lora.unfreeze()
    return lora


def eval_merged_model(base_model, merged_deltas, val_datasets, vocab_size):
    """Evaluate a merged delta model on all domains."""
    model = apply_merged_deltas(base_model, merged_deltas, vocab_size)
    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(model, val_ds, BATCH_SIZE)
    results["avg"] = sum(v for k, v in results.items() if k != "avg") / len(val_datasets)
    return results


def gs_merge_with_ordering(delta_dicts_by_name, ordering, merge_type="average"):
    """Apply Gram-Schmidt merge in a specific ordering.

    Args:
        delta_dicts_by_name: dict mapping expert name -> delta dict
        ordering: list of expert names in desired order
        merge_type: "average" (1/N) or "sum"

    Returns:
        merged_delta: single delta dict
        report: GS diagnostics
    """
    ordered_deltas = [delta_dicts_by_name[name] for name in ordering]
    if merge_type == "average":
        return merge_gs_average(ordered_deltas, list(ordering))
    else:
        from experiments.models.gram_schmidt_composition.gram_schmidt import merge_with_gram_schmidt
        return merge_with_gram_schmidt(ordered_deltas, list(ordering))


# ── Phase 1: Natural LoRA Expert Order Dependence ──────────────────────────

def run_natural_order_experiment(seed=42, n_domains=5, n_orderings=N_ORDERINGS):
    """Test order dependence with real trained LoRA experts.

    Returns dict with all results for analysis.
    """
    split_method = {5: "quintary", 8: "octonary"}.get(n_domains)
    if split_method is None:
        raise ValueError(f"Unsupported n_domains={n_domains}")

    print(f"\n{'='*70}")
    print(f"PHASE 1: NATURAL LoRA EXPERTS (N={n_domains}, seed={seed}, {n_orderings} orderings)")
    print(f"{'='*70}")

    mx.random.seed(seed)
    t0 = time.time()

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # Domain splits
    splits = domain_split(docs, method=split_method)
    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    train_datasets = {}
    val_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_datasets[d_name] = CharDataset(d_train, tokenizer, BASE["block_size"])
        val_datasets[d_name] = CharDataset(d_val, tokenizer, BASE["block_size"])

    domain_names = list(splits.keys())

    # 1. Pretrain base model
    print("\n--- Pretraining base model ---")
    base_model = pretrain_base(joint_train, V, seed)

    # Evaluate base model
    base_results = {}
    for d_name, val_ds in val_datasets.items():
        base_results[d_name] = evaluate(base_model, val_ds, BATCH_SIZE)
    base_results["avg"] = sum(v for k, v in base_results.items() if k != "avg") / len(val_datasets)
    print(f"  Base avg loss: {base_results['avg']:.4f}")

    # 2. Fine-tune LoRA per domain
    delta_dicts_by_name = {}
    for i, d_name in enumerate(domain_names):
        print(f"  Fine-tuning LoRA for {d_name}...")
        lora = finetune_lora(base_model, train_datasets[d_name],
                             val_datasets[d_name], V, seed + i)
        delta_dicts_by_name[d_name] = extract_deltas(lora)

    # 3. Measure pairwise cosines
    flat_deltas = {d: flatten_delta_dict(delta_dicts_by_name[d]) for d in domain_names}
    pairwise_cosines = {}
    for i, d_i in enumerate(domain_names):
        for j in range(i + 1, len(domain_names)):
            d_j = domain_names[j]
            cos = cosine_sim(flat_deltas[d_i], flat_deltas[d_j])
            pairwise_cosines[(d_i, d_j)] = cos

    max_cos = max(abs(v) for v in pairwise_cosines.values())
    mean_cos = np.mean([abs(v) for v in pairwise_cosines.values()])
    print(f"\n  Pairwise cosines: max={max_cos:.4f}, mean={mean_cos:.4f}")

    # 4. Simple average (order-invariant baseline)
    all_deltas_list = [delta_dicts_by_name[d] for d in domain_names]
    merged_avg = merge_simple_average(all_deltas_list)
    avg_results = eval_merged_model(base_model, merged_avg, val_datasets, V)
    print(f"  Simple avg baseline: {avg_results['avg']:.4f}")

    # 5. Test N_ORDERINGS random orderings with GS average
    print(f"\n--- Testing {n_orderings} random orderings ---")
    rng = random.Random(seed * 1000)
    ordering_results = []

    for trial in range(n_orderings):
        ordering = list(domain_names)
        rng.shuffle(ordering)

        merged, report = gs_merge_with_ordering(delta_dicts_by_name, ordering, "average")
        results = eval_merged_model(base_model, merged, val_datasets, V)

        ordering_results.append({
            "ordering": ordering,
            "results": results,
            "signal_retention": report["signal_retention"],
            "signal_retention_min": report["signal_retention_min"],
        })

    # 6. Analyze order dependence
    avg_losses = [r["results"]["avg"] for r in ordering_results]
    best_loss = min(avg_losses)
    worst_loss = max(avg_losses)
    mean_loss = statistics.mean(avg_losses)
    std_loss = statistics.stdev(avg_losses)
    cv_pct = (std_loss / mean_loss) * 100  # coefficient of variation

    # Per-domain analysis
    per_domain_cv = {}
    for d_name in domain_names:
        domain_losses = [r["results"][d_name] for r in ordering_results]
        d_mean = statistics.mean(domain_losses)
        d_std = statistics.stdev(domain_losses)
        per_domain_cv[d_name] = (d_std / d_mean) * 100 if d_mean > 0 else 0

    # Signal retention analysis
    min_retentions = [r["signal_retention_min"] for r in ordering_results]
    retention_mean = statistics.mean(min_retentions)
    retention_std = statistics.stdev(min_retentions)

    # Kill criteria
    worst_vs_best_pct = ((worst_loss - best_loss) / best_loss) * 100
    max_domain_cv = max(per_domain_cv.values())

    print(f"\n--- Results Summary ---")
    print(f"  Avg loss across orderings: {mean_loss:.6f} +/- {std_loss:.6f}")
    print(f"  Best: {best_loss:.6f}, Worst: {worst_loss:.6f}")
    print(f"  CV (aggregate): {cv_pct:.4f}%")
    print(f"  Worst vs best gap: {worst_vs_best_pct:.4f}%")
    print(f"  Simple avg baseline: {avg_results['avg']:.6f}")
    print(f"  Min signal retention: {retention_mean:.4f} +/- {retention_std:.4f}")

    print(f"\n  Per-domain CV:")
    for d_name, cv in per_domain_cv.items():
        print(f"    {d_name:>12}: {cv:.4f}%")

    k1_pass = cv_pct <= 5.0
    k2_pass = worst_vs_best_pct <= 15.0
    print(f"\n  K1 (CV <= 5%): {'PASS' if k1_pass else 'KILL'} ({cv_pct:.4f}%)")
    print(f"  K2 (worst/best <= 15%): {'PASS' if k2_pass else 'KILL'} ({worst_vs_best_pct:.4f}%)")

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "seed": seed,
        "n_domains": n_domains,
        "n_orderings": n_orderings,
        "pairwise_cosines": {f"{a} vs {b}": v for (a, b), v in pairwise_cosines.items()},
        "max_cosine": max_cos,
        "mean_cosine": mean_cos,
        "base_avg": base_results["avg"],
        "simple_avg": avg_results["avg"],
        "ordering_avg_losses": avg_losses,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "cv_pct": cv_pct,
        "best_loss": best_loss,
        "worst_loss": worst_loss,
        "worst_vs_best_pct": worst_vs_best_pct,
        "per_domain_cv": per_domain_cv,
        "max_domain_cv": max_domain_cv,
        "retention_mean": retention_mean,
        "retention_std": retention_std,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "elapsed": elapsed,
    }


# ── Phase 2: Synthetic High-Overlap Experts ────────────────────────────────

def create_synthetic_experts_with_overlap(n_experts, dim, target_cosine, seed=42):
    """Create synthetic expert delta vectors with controlled pairwise cosine similarity.

    Uses a shared component + unique component to achieve target cosine:
      d_k = alpha * shared + beta * unique_k

    where alpha / (alpha + beta) ~ target_cosine approximately.

    Args:
        n_experts: number of experts
        dim: dimension of each delta vector
        target_cosine: desired average pairwise cosine similarity
        seed: random seed

    Returns:
        list of delta dicts (single-key for simplicity)
    """
    rng = np.random.RandomState(seed)

    # Create shared direction
    shared = rng.randn(dim)
    shared = shared / np.linalg.norm(shared)

    # The cosine between two vectors d_i = a*s + b*u_i and d_j = a*s + b*u_j
    # is approximately a^2 / (a^2 + b^2) when u_i, u_j are random orthogonal.
    # So we set a = sqrt(target_cosine), b = sqrt(1 - target_cosine)
    alpha = np.sqrt(max(target_cosine, 0.0))
    beta = np.sqrt(max(1.0 - target_cosine, 0.0))

    deltas = []
    for k in range(n_experts):
        unique = rng.randn(dim)
        unique = unique / np.linalg.norm(unique)
        # Remove shared component from unique to make them truly independent
        unique = unique - np.dot(unique, shared) * shared
        unique = unique / np.linalg.norm(unique)

        vec = alpha * shared + beta * unique
        # Wrap in delta dict format
        delta_dict = {(0, 'fc1'): mx.array(vec.reshape(1, -1).astype(np.float32))}
        deltas.append(delta_dict)

    # Verify actual cosines
    flat_vecs = [flatten_delta_dict(d) for d in deltas]
    actual_cosines = []
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            actual_cosines.append(cosine_sim(flat_vecs[i], flat_vecs[j]))
    actual_mean = np.mean(actual_cosines)

    return deltas, actual_mean


def run_synthetic_order_experiment(n_experts=10, dim=4096, target_cosines=None,
                                   n_orderings=N_ORDERINGS, seed=42):
    """Test order dependence with synthetic high-overlap experts.

    Instead of evaluating a model (no training), we measure how much
    the merged delta vector changes across orderings. The metric is:
    - Cosine similarity between merged vectors from different orderings
    - L2 norm variance of merged vectors
    - Per-expert signal retention variance across orderings

    Args:
        n_experts: number of synthetic experts
        dim: dimension of delta vectors
        target_cosines: list of target pairwise cosine similarities to test
        n_orderings: number of random orderings per condition
        seed: random seed
    """
    if target_cosines is None:
        target_cosines = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    print(f"\n{'='*70}")
    print(f"PHASE 2: SYNTHETIC HIGH-OVERLAP EXPERTS (N={n_experts}, D={dim})")
    print(f"{'='*70}")

    t0 = time.time()
    results_by_cos = {}

    for target_cos in target_cosines:
        print(f"\n--- Target cosine: {target_cos:.2f} ---")

        deltas, actual_cos = create_synthetic_experts_with_overlap(
            n_experts, dim, target_cos, seed)
        names = [f"expert_{i}" for i in range(n_experts)]
        delta_by_name = dict(zip(names, deltas))

        print(f"  Actual mean cosine: {actual_cos:.4f}")

        rng = random.Random(seed * 100 + int(target_cos * 1000))
        merged_vectors = []
        signal_retentions_all = []
        ordering_details = []

        for trial in range(n_orderings):
            ordering = list(names)
            rng.shuffle(ordering)

            ordered_deltas = [delta_by_name[n] for n in ordering]
            ortho_deltas, report = gram_schmidt_orthogonalize(ordered_deltas, list(ordering))

            # Merge via average
            N = len(ortho_deltas)
            keys = sorted(ortho_deltas[0].keys())
            merged = {}
            for k in keys:
                merged[k] = sum(d[k] for d in ortho_deltas) / N

            flat_merged = flatten_delta_dict(merged)
            merged_vectors.append(flat_merged)
            signal_retentions_all.append(report["signal_retention"])
            ordering_details.append({
                "ordering": ordering,
                "signal_retention_min": report["signal_retention_min"],
            })

        # Analyze merged vector variation
        # Pairwise cosines between all merged vectors (should all be ~1.0 if order-invariant)
        merged_cosines = []
        for i in range(len(merged_vectors)):
            for j in range(i + 1, len(merged_vectors)):
                merged_cosines.append(cosine_sim(merged_vectors[i], merged_vectors[j]))

        merged_cos_min = min(merged_cosines)
        merged_cos_mean = np.mean(merged_cosines)

        # L2 norm variation
        norms = [np.linalg.norm(v) for v in merged_vectors]
        norm_cv = (statistics.stdev(norms) / statistics.mean(norms)) * 100

        # Signal retention variation per expert position
        # For each expert, what's the range of signal retention across orderings?
        expert_retention_ranges = {}
        for name in names:
            rets = [sr[name] for sr in signal_retentions_all if name in sr]
            if len(rets) > 1:
                expert_retention_ranges[name] = {
                    "min": min(rets),
                    "max": max(rets),
                    "range": max(rets) - min(rets),
                    "mean": statistics.mean(rets),
                }

        max_retention_range = max(v["range"] for v in expert_retention_ranges.values())
        mean_retention_range = statistics.mean([v["range"] for v in expert_retention_ranges.values()])

        print(f"  Merged vector similarity: min={merged_cos_min:.6f}, mean={merged_cos_mean:.6f}")
        print(f"  Merged norm CV: {norm_cv:.4f}%")
        print(f"  Signal retention range: max={max_retention_range:.4f}, mean={mean_retention_range:.4f}")

        # Relative variation as proxy for quality variance
        # The merged vector variation directly translates to model output variation
        variation_pct = (1.0 - merged_cos_min) * 100
        print(f"  Max merged vector deviation: {variation_pct:.4f}%")

        results_by_cos[target_cos] = {
            "actual_cosine": actual_cos,
            "merged_cos_min": merged_cos_min,
            "merged_cos_mean": merged_cos_mean,
            "norm_cv_pct": norm_cv,
            "max_retention_range": max_retention_range,
            "mean_retention_range": mean_retention_range,
            "variation_pct": variation_pct,
        }

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*70}")
    print(f"PHASE 2 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Target cos':>12} {'Actual cos':>12} {'Merged cos min':>15} {'Norm CV%':>10} {'Ret range':>12} {'Variation%':>12}")
    print(f"  {'-'*75}")
    for tc in target_cosines:
        r = results_by_cos[tc]
        print(f"  {tc:>12.2f} {r['actual_cosine']:>12.4f} {r['merged_cos_min']:>15.6f} "
              f"{r['norm_cv_pct']:>10.4f} {r['max_retention_range']:>12.4f} {r['variation_pct']:>12.4f}")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    return results_by_cos


# ── Phase 3: Order-Invariant Alternatives ──────────────────────────────────

def svd_simultaneous_orthogonalize(delta_dicts, names=None):
    """SVD-based simultaneous (order-invariant) orthogonalization.

    Instead of sequential Gram-Schmidt, compute the column space of the
    matrix [d_1 | d_2 | ... | d_N] via SVD, then project each delta onto
    its nearest orthogonal basis vector.

    This is order-invariant by construction.

    Returns:
        orthogonalized: list of orthogonalized delta dicts
        report: dict with diagnostics
    """
    N = len(delta_dicts)
    if names is None:
        names = [f"expert_{i}" for i in range(N)]

    template = delta_dicts[0]
    flat_originals = [flatten_delta_dict(d) for d in delta_dicts]
    D = len(flat_originals[0])

    # Stack into matrix [D x N]
    M = np.column_stack(flat_originals)

    # SVD: M = U @ S @ V^T
    # U[:, :N] spans the same subspace as the deltas
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Project each delta onto the orthogonal basis
    # coords[k] = U^T @ d_k gives coordinates in orthogonal basis
    coords = U.T @ M  # [N x N] matrix of coordinates

    # Reconstruct each delta using only its component in the orthogonal basis
    # But keep each delta's projection separate
    flat_ortho = []
    for k in range(N):
        # Project delta k onto each basis vector
        proj = np.zeros(D)
        for j in range(N):
            proj += coords[j, k] * U[:, j]
        flat_ortho.append(proj)

    # Note: this doesn't actually orthogonalize the deltas!
    # The projections are just the original deltas expressed in the U basis.
    # For true orthogonalization, assign each delta to its most aligned basis vector.

    # Hungarian assignment: each delta gets a unique basis vector
    from scipy.optimize import linear_sum_assignment

    # Cost matrix: negative absolute correlation (we want max alignment)
    cost = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cost[i, j] = -abs(np.dot(flat_originals[i], U[:, j]))

    row_ind, col_ind = linear_sum_assignment(cost)

    # Each expert k gets basis vector col_ind[k]
    flat_ortho_assigned = []
    for k in range(N):
        basis_idx = col_ind[k]
        # Scale basis vector to match original delta's projection magnitude
        scale = np.dot(flat_originals[k], U[:, basis_idx])
        flat_ortho_assigned.append(scale * U[:, basis_idx])

    # Signal retention
    signal_retention = {}
    for k in range(N):
        orig_norm = np.linalg.norm(flat_originals[k])
        ortho_norm = np.linalg.norm(flat_ortho_assigned[k])
        signal_retention[names[k]] = float(ortho_norm / orig_norm) if orig_norm > 1e-12 else 0.0

    # Post-orthogonalization cosines (should be exactly 0)
    post_cosines = {}
    for i in range(N):
        for j in range(i + 1, N):
            cos = cosine_sim(flat_ortho_assigned[i], flat_ortho_assigned[j])
            post_cosines[f"{names[i]} vs {names[j]}"] = cos

    orthogonalized = [unflatten_delta_dict(flat_ortho_assigned[k], template)
                      for k in range(N)]

    report = {
        "n_experts": N,
        "names": names,
        "signal_retention": signal_retention,
        "signal_retention_min": min(signal_retention.values()),
        "post_cosines": post_cosines,
        "max_post_cosine": max(abs(v) for v in post_cosines.values()) if post_cosines else 0.0,
        "singular_values": S.tolist(),
    }

    return orthogonalized, report


def symmetric_gs_orthogonalize(delta_dicts, names=None, n_orderings=20, seed=42):
    """Symmetric Gram-Schmidt: average over many random orderings.

    This is an order-invariant approximation: run GS in many orderings,
    then average the results. In the limit of all N! orderings, this
    is perfectly symmetric.

    Returns:
        orthogonalized: list of averaged orthogonalized delta dicts
        report: diagnostics
    """
    N = len(delta_dicts)
    if names is None:
        names = [f"expert_{i}" for i in range(N)]

    rng = random.Random(seed)
    template = delta_dicts[0]

    # Accumulate flattened ortho deltas across orderings
    accum = {name: np.zeros(len(flatten_delta_dict(delta_dicts[0]))) for name in names}
    delta_by_name = dict(zip(names, delta_dicts))

    for trial in range(n_orderings):
        ordering = list(names)
        rng.shuffle(ordering)
        ordered_deltas = [delta_by_name[n] for n in ordering]
        ortho_deltas, _ = gram_schmidt_orthogonalize(ordered_deltas, list(ordering))

        # Map back to original name ordering
        for idx, name in enumerate(ordering):
            accum[name] += flatten_delta_dict(ortho_deltas[idx])

    # Average
    flat_ortho = [accum[name] / n_orderings for name in names]

    # Signal retention
    flat_originals = [flatten_delta_dict(d) for d in delta_dicts]
    signal_retention = {}
    for k in range(N):
        orig_norm = np.linalg.norm(flat_originals[k])
        ortho_norm = np.linalg.norm(flat_ortho[k])
        signal_retention[names[k]] = float(ortho_norm / orig_norm) if orig_norm > 1e-12 else 0.0

    # Post cosines
    post_cosines = {}
    for i in range(N):
        for j in range(i + 1, N):
            cos = cosine_sim(flat_ortho[i], flat_ortho[j])
            post_cosines[f"{names[i]} vs {names[j]}"] = cos

    orthogonalized = [unflatten_delta_dict(flat_ortho[k], template) for k in range(N)]

    report = {
        "n_experts": N,
        "signal_retention": signal_retention,
        "signal_retention_min": min(signal_retention.values()),
        "post_cosines": post_cosines,
        "max_post_cosine": max(abs(v) for v in post_cosines.values()) if post_cosines else 0.0,
    }

    return orthogonalized, report


def run_alternatives_comparison(n_experts=10, dim=4096, target_cosine=0.5,
                                 n_orderings=20, seed=42):
    """Compare GS vs SVD vs Symmetric GS for order invariance.

    Uses synthetic experts with controlled overlap to compare:
    1. Standard GS (single ordering) -- order-dependent baseline
    2. SVD simultaneous -- fully order-invariant
    3. Symmetric GS (average of orderings) -- approximately order-invariant
    """
    print(f"\n{'='*70}")
    print(f"PHASE 3: ORDER-INVARIANT ALTERNATIVES (N={n_experts}, cos={target_cosine})")
    print(f"{'='*70}")

    t0 = time.time()

    deltas, actual_cos = create_synthetic_experts_with_overlap(
        n_experts, dim, target_cosine, seed)
    names = [f"expert_{i}" for i in range(n_experts)]
    delta_by_name = dict(zip(names, deltas))
    flat_originals = [flatten_delta_dict(d) for d in deltas]

    print(f"  Actual mean cosine: {actual_cos:.4f}")

    # --- Standard GS: measure variance across orderings ---
    rng = random.Random(seed * 200)
    gs_merged_vectors = []
    gs_retentions = []

    for trial in range(n_orderings):
        ordering = list(names)
        rng.shuffle(ordering)
        ordered_deltas = [delta_by_name[n] for n in ordering]
        ortho, report = gram_schmidt_orthogonalize(ordered_deltas, list(ordering))

        N = len(ortho)
        keys = sorted(ortho[0].keys())
        merged = {}
        for k in keys:
            merged[k] = sum(d[k] for d in ortho) / N

        gs_merged_vectors.append(flatten_delta_dict(merged))
        gs_retentions.append(report["signal_retention_min"])

    # GS variation
    gs_cosines = []
    for i in range(len(gs_merged_vectors)):
        for j in range(i + 1, len(gs_merged_vectors)):
            gs_cosines.append(cosine_sim(gs_merged_vectors[i], gs_merged_vectors[j]))
    gs_cos_min = min(gs_cosines)
    gs_norms = [np.linalg.norm(v) for v in gs_merged_vectors]
    gs_norm_cv = (statistics.stdev(gs_norms) / statistics.mean(gs_norms)) * 100

    print(f"\n  Standard GS:")
    print(f"    Merged cos min: {gs_cos_min:.6f}")
    print(f"    Norm CV: {gs_norm_cv:.4f}%")
    print(f"    Min retention: {min(gs_retentions):.4f}")

    # --- SVD simultaneous ---
    svd_ortho, svd_report = svd_simultaneous_orthogonalize(deltas, names)
    svd_merged_keys = sorted(svd_ortho[0].keys())
    svd_merged = {}
    for k in svd_merged_keys:
        svd_merged[k] = sum(d[k] for d in svd_ortho) / len(svd_ortho)
    svd_merged_flat = flatten_delta_dict(svd_merged)

    print(f"\n  SVD Simultaneous:")
    print(f"    Min retention: {svd_report['signal_retention_min']:.4f}")
    print(f"    Max post-cosine: {svd_report['max_post_cosine']:.6f}")
    print(f"    Singular values: {svd_report['singular_values'][:5]}...")

    # --- Symmetric GS ---
    sym_ortho, sym_report = symmetric_gs_orthogonalize(deltas, names, n_orderings=50, seed=seed)
    sym_merged_keys = sorted(sym_ortho[0].keys())
    sym_merged = {}
    for k in sym_merged_keys:
        sym_merged[k] = sum(d[k] for d in sym_ortho) / len(sym_ortho)
    sym_merged_flat = flatten_delta_dict(sym_merged)

    print(f"\n  Symmetric GS (50 orderings):")
    print(f"    Min retention: {sym_report['signal_retention_min']:.4f}")
    print(f"    Max post-cosine: {sym_report['max_post_cosine']:.6f}")

    # Compare all methods against each other
    # Use mean of GS vectors as the GS reference
    gs_mean = np.mean(gs_merged_vectors, axis=0)

    cos_gs_svd = cosine_sim(gs_mean, svd_merged_flat)
    cos_gs_sym = cosine_sim(gs_mean, sym_merged_flat)
    cos_svd_sym = cosine_sim(svd_merged_flat, sym_merged_flat)

    # Also compare to naive average (no orthogonalization)
    naive_merged = merge_naive_sum(deltas)
    naive_merged_keys = sorted(naive_merged.keys())
    naive_avg = {}
    for k in naive_merged_keys:
        naive_avg[k] = naive_merged[k] / len(deltas)
    naive_flat = flatten_delta_dict(naive_avg)

    cos_gs_naive = cosine_sim(gs_mean, naive_flat)
    cos_svd_naive = cosine_sim(svd_merged_flat, naive_flat)

    print(f"\n  Cross-method cosine similarity:")
    print(f"    GS mean vs SVD:     {cos_gs_svd:.6f}")
    print(f"    GS mean vs Sym GS:  {cos_gs_sym:.6f}")
    print(f"    SVD vs Sym GS:      {cos_svd_sym:.6f}")
    print(f"    GS mean vs Naive:   {cos_gs_naive:.6f}")
    print(f"    SVD vs Naive:       {cos_svd_naive:.6f}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        "target_cosine": target_cosine,
        "actual_cosine": actual_cos,
        "gs_cos_min": gs_cos_min,
        "gs_norm_cv": gs_norm_cv,
        "svd_retention_min": svd_report["signal_retention_min"],
        "svd_max_post_cos": svd_report["max_post_cosine"],
        "sym_retention_min": sym_report["signal_retention_min"],
        "sym_max_post_cos": sym_report["max_post_cosine"],
        "cos_gs_svd": cos_gs_svd,
        "cos_gs_sym": cos_gs_sym,
        "cos_svd_sym": cos_svd_sym,
        "cos_gs_naive": cos_gs_naive,
        "elapsed": elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def run_full_experiment():
    """Run the complete merge order dependence experiment."""
    all_results = {}

    # Phase 1: Natural experts
    print("\n" + "=" * 70)
    print("PHASE 1: NATURAL LoRA EXPERTS -- ORDER DEPENDENCE")
    print("=" * 70)

    # N=5, two seeds
    for seed in [42, 7]:
        key = f"natural_N5_seed{seed}"
        all_results[key] = run_natural_order_experiment(seed=seed, n_domains=5,
                                                         n_orderings=N_ORDERINGS)

    # N=8, one seed (more domains = more ordering variation)
    all_results["natural_N8_seed42"] = run_natural_order_experiment(
        seed=42, n_domains=8, n_orderings=N_ORDERINGS)

    # Phase 2: Synthetic high-overlap
    print("\n" + "=" * 70)
    print("PHASE 2: SYNTHETIC HIGH-OVERLAP -- STRESS TEST")
    print("=" * 70)
    all_results["synthetic"] = run_synthetic_order_experiment(
        n_experts=10, dim=4096, n_orderings=N_ORDERINGS)

    # Phase 3: Alternatives comparison at high overlap
    print("\n" + "=" * 70)
    print("PHASE 3: ORDER-INVARIANT ALTERNATIVES")
    print("=" * 70)
    for cos_val in [0.1, 0.3, 0.5, 0.7]:
        key = f"alternatives_cos{cos_val}"
        all_results[key] = run_alternatives_comparison(
            n_experts=10, dim=4096, target_cosine=cos_val, n_orderings=N_ORDERINGS)

    # ── Final Summary ──
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY: MERGE ORDER DEPENDENCE")
    print("=" * 70)

    # Natural expert results
    print("\n  Phase 1: Natural LoRA Experts (near-orthogonal regime)")
    print(f"  {'Condition':>25} {'CV%':>8} {'Best/Worst%':>12} {'K1':>6} {'K2':>6}")
    print(f"  {'-'*60}")
    for key in sorted(all_results.keys()):
        if key.startswith("natural_"):
            r = all_results[key]
            print(f"  {key:>25} {r['cv_pct']:>8.4f} {r['worst_vs_best_pct']:>12.4f} "
                  f"{'PASS' if r['k1_pass'] else 'KILL':>6} "
                  f"{'PASS' if r['k2_pass'] else 'KILL':>6}")

    # Synthetic results
    if "synthetic" in all_results:
        print(f"\n  Phase 2: Synthetic Experts (controlled overlap)")
        print(f"  {'Target cos':>12} {'Merged cos min':>15} {'Norm CV%':>10} {'Variation%':>12}")
        print(f"  {'-'*55}")
        for tc, r in sorted(all_results["synthetic"].items()):
            if isinstance(tc, (int, float)):
                print(f"  {tc:>12.2f} {r['merged_cos_min']:>15.6f} "
                      f"{r['norm_cv_pct']:>10.4f} {r['variation_pct']:>12.4f}")

    # Kill criteria assessment
    print(f"\n  Kill Criteria Assessment:")

    # Aggregate K1 and K2 from natural experiments
    all_cvs = [all_results[k]["cv_pct"] for k in all_results if k.startswith("natural_")]
    all_gaps = [all_results[k]["worst_vs_best_pct"] for k in all_results if k.startswith("natural_")]

    max_cv = max(all_cvs) if all_cvs else 0
    max_gap = max(all_gaps) if all_gaps else 0

    k1_overall = max_cv <= 5.0
    k2_overall = max_gap <= 15.0

    print(f"    K1 (CV <= 5%): worst CV = {max_cv:.4f}% -> {'PASS' if k1_overall else 'KILL'}")
    print(f"    K2 (worst/best <= 15%): worst gap = {max_gap:.4f}% -> {'PASS' if k2_overall else 'KILL'}")
    print(f"    Overall: {'PASS' if (k1_overall and k2_overall) else 'KILL'}")

    return all_results


if __name__ == "__main__":
    results = run_full_experiment()
