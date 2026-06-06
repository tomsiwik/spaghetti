"""Gram-Schmidt Orthogonalization for LoRA Composition: Experiment.

Tests whether Gram-Schmidt projection preserves expert quality when merging
LoRA adapters that have non-trivial overlap. Uses micro-scale (d=64, rank=8)
character-level name generation to isolate the composition mechanism.

Protocol:
1. Pretrain base GPT model on all names
2. Fine-tune N LoRA experts on distinct alphabet-based domain splits
3. Measure pairwise cosine similarity between deltas
4. Merge two ways: (a) naive sum, (b) Gram-Schmidt then sum
5. Compare merged model PPL against base and individual experts
6. Verify kill criteria:
   - Orthogonalized expert loses >10% PPL improvement -> KILL
   - Signal retention <50% for any expert -> KILL

Key insight from lora_merging_bakeoff: simple average is the best zero-shot
merge method for orthogonal deltas. But it SUMS the overlapping components,
which can cause interference. Gram-Schmidt removes this overlap.
"""

import copy
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
    merge_with_gram_schmidt,
    merge_gs_average,
    merge_naive_sum,
    cosine_sim,
    flatten_delta_dict,
)

# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3


# ── Helpers ─────────────────────────────────────────────────────────────────

def pretrain_base(joint_train, vocab_size, seed):
    """Pretrain base GPT model on joint data."""
    base = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base.parameters())
    train(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
          lr=LR, seed=seed, log_every=150)
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


def eval_individual_expert(lora_model, val_ds):
    """Evaluate a single LoRA expert on its own domain."""
    return evaluate(lora_model, val_ds, BATCH_SIZE)


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_gram_schmidt_experiment(seed=42, n_domains=5):
    """Run the Gram-Schmidt orthogonalization experiment.

    Steps:
    1. Pretrain base model
    2. Fine-tune N LoRA experts on domain splits
    3. Measure pairwise cosine similarity
    4. Compare merge strategies: naive sum vs GS-orthogonalized sum
    5. Evaluate per-domain PPL for base, individual, naive-merged, GS-merged
    """
    print(f"\n{'='*70}")
    print(f"GRAM-SCHMIDT ORTHOGONALIZATION EXPERIMENT (seed={seed}, N={n_domains})")
    print(f"{'='*70}")

    mx.random.seed(seed)
    t0 = time.time()

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # Domain splits
    if n_domains == 2:
        splits = domain_split(docs, method="binary")
    elif n_domains == 5:
        splits = domain_split(docs, method="quintary")
    else:
        raise ValueError(f"Unsupported n_domains={n_domains}")

    # Prepare datasets
    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    train_datasets = {}
    val_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_datasets[d_name] = CharDataset(d_train, tokenizer, BASE["block_size"])
        val_datasets[d_name] = CharDataset(d_val, tokenizer, BASE["block_size"])

    domain_names = list(splits.keys())

    # === 1. Pretrain base model ===
    print("\n--- 1. Pretraining base model ---")
    base_model = pretrain_base(joint_train, V, seed)

    # Evaluate base model on all domains
    base_results = {}
    for d_name, val_ds in val_datasets.items():
        base_results[d_name] = evaluate(base_model, val_ds, BATCH_SIZE)
    base_results["avg"] = sum(v for k, v in base_results.items() if k != "avg") / len(val_datasets)
    print(f"  Base avg PPL: {base_results['avg']:.4f}")

    # === 2. Fine-tune LoRA per domain ===
    lora_models = {}
    delta_dicts = {}
    individual_results = {}

    for i, d_name in enumerate(domain_names):
        print(f"\n--- 2{chr(97+i)}. Fine-tuning LoRA for {d_name} ---")
        lora = finetune_lora(base_model, train_datasets[d_name],
                             val_datasets[d_name], V, seed + i)
        lora_models[d_name] = lora
        delta_dicts[d_name] = extract_deltas(lora)

        # Evaluate individual expert on its own domain
        individual_results[d_name] = eval_individual_expert(lora, val_datasets[d_name])
        ppl_improvement = (base_results[d_name] - individual_results[d_name]) / base_results[d_name] * 100
        print(f"  {d_name}: base={base_results[d_name]:.4f}, expert={individual_results[d_name]:.4f}, improvement={ppl_improvement:+.2f}%")

    all_delta_dicts = [delta_dicts[d] for d in domain_names]

    # === 3. Pairwise cosine similarity analysis ===
    print(f"\n--- 3. Pairwise Cosine Similarity ---")
    import numpy as np
    flat_deltas = {d: flatten_delta_dict(delta_dicts[d]) for d in domain_names}

    print(f"\n  {'':>12}", end="")
    for d in domain_names:
        print(f" {d[:8]:>8}", end="")
    print()

    max_cos = 0.0
    for i, d_i in enumerate(domain_names):
        print(f"  {d_i:>12}", end="")
        for j, d_j in enumerate(domain_names):
            if i == j:
                print(f" {'1.000':>8}", end="")
            elif j > i:
                cos = cosine_sim(flat_deltas[d_i], flat_deltas[d_j])
                print(f" {cos:>8.4f}", end="")
                max_cos = max(max_cos, abs(cos))
            else:
                cos = cosine_sim(flat_deltas[d_i], flat_deltas[d_j])
                print(f" {cos:>8.4f}", end="")
        print()

    print(f"\n  Max absolute cosine: {max_cos:.4f}")

    # === 4. Naive sum merge (no orthogonalization) ===
    print(f"\n--- 4. Naive Sum Merge ---")
    t_naive = time.time()
    merged_naive = merge_naive_sum(all_delta_dicts)
    t_naive_cost = time.time() - t_naive
    naive_results = eval_merged_model(base_model, merged_naive, val_datasets, V)
    print(f"  Naive sum avg: {naive_results['avg']:.4f} ({t_naive_cost*1000:.1f}ms)")

    # === 5. Simple average merge (1/N scaling, from bakeoff) ===
    print(f"\n--- 5. Simple Average Merge (1/N) ---")
    t_avg = time.time()
    merged_avg = merge_simple_average(all_delta_dicts)
    t_avg_cost = time.time() - t_avg
    avg_results = eval_merged_model(base_model, merged_avg, val_datasets, V)
    print(f"  Simple avg: {avg_results['avg']:.4f} ({t_avg_cost*1000:.1f}ms)")

    # === 6a. Gram-Schmidt orthogonalized SUM merge ===
    print(f"\n--- 6a. Gram-Schmidt Sum Merge ---")
    t_gs = time.time()
    merged_gs, gs_report = merge_with_gram_schmidt(all_delta_dicts, domain_names)
    t_gs_cost = time.time() - t_gs
    gs_results = eval_merged_model(base_model, merged_gs, val_datasets, V)
    print(f"  GS sum avg: {gs_results['avg']:.4f} ({t_gs_cost*1000:.1f}ms)")

    # === 6b. Gram-Schmidt orthogonalized AVERAGE merge (key comparison) ===
    print(f"\n--- 6b. Gram-Schmidt Average Merge (1/N) ---")
    t_gsa = time.time()
    merged_gsa, gs_avg_report = merge_gs_average(all_delta_dicts, domain_names)
    t_gsa_cost = time.time() - t_gsa
    gsa_results = eval_merged_model(base_model, merged_gsa, val_datasets, V)
    print(f"  GS avg avg: {gsa_results['avg']:.4f} ({t_gsa_cost*1000:.1f}ms)")

    # Print GS diagnostics
    print(f"\n  Signal Retention (||D_k'|| / ||D_k||):")
    for name, retention in gs_report["signal_retention"].items():
        status = "OK" if retention >= 0.5 else "KILL"
        print(f"    {name:>12}: {retention:.4f} ({retention*100:.1f}%) [{status}]")
    print(f"  Min retention: {gs_report['signal_retention_min']:.4f}")

    print(f"\n  Post-GS pairwise cosines (should be ~0):")
    for pair, cos in gs_report["post_cosines"].items():
        print(f"    {pair}: {cos:.6f}")

    # === 7. GS with different orderings (order sensitivity check) ===
    print(f"\n--- 7. Order Sensitivity Check ---")
    import itertools
    if n_domains <= 5:
        # Test reversed ordering
        reversed_names = list(reversed(domain_names))
        reversed_deltas = [delta_dicts[d] for d in reversed_names]
        merged_gs_rev, gs_rev_report = merge_with_gram_schmidt(reversed_deltas, reversed_names)
        gs_rev_results = eval_merged_model(base_model, merged_gs_rev, val_datasets, V)
        print(f"  Original order avg: {gs_results['avg']:.4f}")
        print(f"  Reversed order avg: {gs_rev_results['avg']:.4f}")
        print(f"  Order gap: {abs(gs_results['avg'] - gs_rev_results['avg']):.4f}")

        # Signal retention comparison
        print(f"\n  Signal retention by ordering:")
        print(f"  {'Expert':>12} {'Original':>10} {'Reversed':>10}")
        for d in domain_names:
            orig_ret = gs_report["signal_retention"].get(d, 0)
            rev_ret = gs_rev_report["signal_retention"].get(d, 0)
            print(f"  {d:>12} {orig_ret:>10.4f} {rev_ret:>10.4f}")

    # === Summary ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY (N={n_domains}, seed={seed}, elapsed={elapsed:.1f}s)")
    print(f"{'='*70}")

    print(f"\n  Per-domain validation loss:")
    print(f"  {'Domain':>12} {'Base':>8} {'Expert':>8} {'Naive':>8} {'Avg(1/N)':>8} {'GS Sum':>8} {'GS Avg':>8}")
    print(f"  {'-'*66}")
    for d in domain_names:
        base_v = base_results[d]
        expert_v = individual_results.get(d, float('nan'))
        naive_v = naive_results[d]
        avg_v = avg_results[d]
        gs_v = gs_results[d]
        gsa_v = gsa_results[d]
        print(f"  {d:>12} {base_v:>8.4f} {expert_v:>8.4f} {naive_v:>8.4f} {avg_v:>8.4f} {gs_v:>8.4f} {gsa_v:>8.4f}")

    print(f"\n  {'Avg':>12} {base_results['avg']:>8.4f} {'':>8} {naive_results['avg']:>8.4f} {avg_results['avg']:>8.4f} {gs_results['avg']:>8.4f} {gsa_results['avg']:>8.4f}")

    # PPL improvement analysis
    print(f"\n  PPL Improvement vs Base (lower = better):")
    print(f"  {'Method':>15} {'Avg Loss':>10} {'vs Base':>10}")
    print(f"  {'-'*38}")
    base_avg = base_results['avg']
    for method, res in [("Base", base_results), ("Naive Sum", naive_results),
                         ("Avg (1/N)", avg_results), ("GS Sum", gs_results),
                         ("GS Avg (1/N)", gsa_results)]:
        gap = (res['avg'] - base_avg) / base_avg * 100
        print(f"  {method:>15} {res['avg']:>10.4f} {gap:>+9.2f}%")

    if 'gs_rev_results' in dir() or 'gs_rev_results' in locals():
        gap_rev = (gs_rev_results['avg'] - base_avg) / base_avg * 100
        print(f"  {'GS Rev Sum':>15} {gs_rev_results['avg']:>10.4f} {gap_rev:>+9.2f}%")

    # === Kill Criteria ===
    print(f"\n--- Kill Criteria ---")

    # KC1: GS average loses >10% PPL improvement vs simple average
    print(f"\n  KC1: GS Avg loses >10% PPL improvement vs Simple Avg?")
    print(f"  (Comparing the fair apples-to-apples: both use 1/N scaling)")
    kc1_killed = False
    for d in domain_names:
        # Compare per-domain loss: GS avg vs simple avg (both use 1/N)
        # "PPL improvement over base" = base_loss - merged_loss (positive = better)
        avg_improvement = base_results[d] - avg_results[d]
        gsa_improvement = base_results[d] - gsa_results[d]
        if avg_improvement > 0:
            loss_pct = (avg_improvement - gsa_improvement) / avg_improvement * 100
        elif avg_improvement < 0:
            # Both are worse than base - compare how much worse
            loss_pct = 0.0 if abs(gsa_improvement) <= abs(avg_improvement) * 1.1 else 100.0
        else:
            loss_pct = 0.0
        status = "KILL" if loss_pct > 10.0 else "OK"
        if loss_pct > 10.0:
            kc1_killed = True
        print(f"    {d:>12}: avg_improv={avg_improvement:+.4f}, gsa_improv={gsa_improvement:+.4f}, loss={loss_pct:+.1f}% [{status}]")

    # KC2: signal retention <50%
    print(f"\n  KC2: Signal retention <50%?")
    kc2_killed = gs_report['signal_retention_min'] < 0.5
    for name, ret in gs_report['signal_retention'].items():
        status = "KILL" if ret < 0.5 else "OK"
        print(f"    {name:>12}: {ret:.4f} ({ret*100:.1f}%) [{status}]")

    print(f"\n  KC1 (>10% PPL loss): {'KILL' if kc1_killed else 'PASS'}")
    print(f"  KC2 (<50% signal):   {'KILL' if kc2_killed else 'PASS'}")
    print(f"  Overall:             {'KILL' if (kc1_killed or kc2_killed) else 'PASS'}")

    return {
        "base": base_results,
        "individual": individual_results,
        "naive_sum": naive_results,
        "simple_avg": avg_results,
        "gs_merge": gs_results,
        "gs_avg": gsa_results,
        "gs_report": gs_report,
        "gs_reversed": gs_rev_results if 'gs_rev_results' in locals() else None,
        "gs_rev_report": gs_rev_report if 'gs_rev_report' in locals() else None,
        "kc1_killed": kc1_killed,
        "kc2_killed": kc2_killed,
    }


def run_multiseed(seeds=(42, 123, 7), n_domains=5):
    """Run across multiple seeds and aggregate."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_gram_schmidt_experiment(seed, n_domains)

    print(f"\n\n{'='*70}")
    print(f"MULTI-SEED AGGREGATE (N={n_domains}, seeds={seeds})")
    print(f"{'='*70}")

    methods = ["base", "naive_sum", "simple_avg", "gs_merge", "gs_avg"]
    method_avgs = {m: [] for m in methods}

    for seed in seeds:
        for m in methods:
            if m == "base":
                method_avgs[m].append(all_results[seed]["base"]["avg"])
            else:
                method_avgs[m].append(all_results[seed][m]["avg"])

    base_mean = statistics.mean(method_avgs["base"])

    print(f"\n  {'Method':>15} {'Mean Loss':>10} {'Std':>8} {'vs Base':>10}")
    print(f"  {'-'*48}")
    for m in methods:
        mean_val = statistics.mean(method_avgs[m])
        std_val = statistics.stdev(method_avgs[m]) if len(method_avgs[m]) > 1 else 0.0
        gap = (mean_val - base_mean) / base_mean * 100
        gap_str = "baseline" if m == "base" else f"{gap:+.2f}%"
        print(f"  {m:>15} {mean_val:>10.4f} {std_val:>8.4f} {gap_str:>10}")

    # Aggregate signal retention
    print(f"\n  Signal Retention (aggregated across seeds):")
    all_retentions = {}
    for seed in seeds:
        report = all_results[seed]["gs_report"]
        for name, ret in report["signal_retention"].items():
            all_retentions.setdefault(name, []).append(ret)

    for name, rets in all_retentions.items():
        mean_ret = statistics.mean(rets)
        std_ret = statistics.stdev(rets) if len(rets) > 1 else 0.0
        print(f"    {name:>12}: {mean_ret:.4f} +/- {std_ret:.4f} ({mean_ret*100:.1f}%)")

    # Aggregate kill criteria
    kc1_any = any(all_results[s]["kc1_killed"] for s in seeds)
    kc2_any = any(all_results[s]["kc2_killed"] for s in seeds)
    print(f"\n  KC1 across seeds: {'KILL (at least one seed)' if kc1_any else 'PASS (all seeds)'}")
    print(f"  KC2 across seeds: {'KILL (at least one seed)' if kc2_any else 'PASS (all seeds)'}")

    return all_results


# ── Unit tests ──────────────────────────────────────────────────────────────

def test_gram_schmidt_orthogonality():
    """Verify that GS produces orthogonal vectors."""
    import numpy as np
    from experiments.models.gram_schmidt_composition.gram_schmidt import gram_schmidt_orthogonalize

    # Create 3 delta dicts with known overlap
    d1 = {(0, 'fc1'): mx.array([[1.0, 0.0, 0.0, 0.0]])}
    d2 = {(0, 'fc1'): mx.array([[0.7, 0.7, 0.0, 0.0]])}  # 45-degree angle to d1
    d3 = {(0, 'fc1'): mx.array([[0.5, 0.5, 0.5, 0.0]])}  # overlaps with both

    ortho, report = gram_schmidt_orthogonalize([d1, d2, d3], ["d1", "d2", "d3"])

    # Verify orthogonality
    flat = [flatten_delta_dict(d) for d in ortho]
    for i in range(3):
        for j in range(i + 1, 3):
            cos = cosine_sim(flat[i], flat[j])
            assert abs(cos) < 1e-6, f"Not orthogonal: {i} vs {j}, cos={cos}"

    # Verify first vector unchanged
    orig_flat = flatten_delta_dict(d1)
    assert np.allclose(flat[0], orig_flat), "First vector should be unchanged"

    # Verify signal retention
    assert report["signal_retention"]["d1"] > 0.99, "First expert should retain all signal"
    # d2 had 50% overlap with d1, so retention should be ~0.71 (sin(45deg))
    assert 0.5 < report["signal_retention"]["d2"] < 0.9, \
        f"d2 retention unexpected: {report['signal_retention']['d2']}"

    print("PASS: Gram-Schmidt orthogonality unit test")


def test_gram_schmidt_with_orthogonal_inputs():
    """When inputs are already orthogonal, GS should not change them."""
    d1 = {(0, 'fc1'): mx.array([[1.0, 0.0, 0.0]])}
    d2 = {(0, 'fc1'): mx.array([[0.0, 1.0, 0.0]])}
    d3 = {(0, 'fc1'): mx.array([[0.0, 0.0, 1.0]])}

    ortho, report = gram_schmidt_orthogonalize([d1, d2, d3])

    # All should retain full signal
    for name, ret in report["signal_retention"].items():
        assert ret > 0.99, f"{name} retention too low: {ret}"

    print("PASS: Gram-Schmidt with orthogonal inputs test")


def test_gram_schmidt_signal_kill_criterion():
    """Test that heavily overlapping experts trigger signal kill."""
    # d2 is almost identical to d1 -> GS should nearly zero it out
    d1 = {(0, 'fc1'): mx.array([[1.0, 0.0, 0.0]])}
    d2 = {(0, 'fc1'): mx.array([[0.99, 0.01, 0.0]])}  # 99% overlap

    ortho, report = gram_schmidt_orthogonalize([d1, d2])

    # d2 should have very low retention
    assert report["signal_retention"]["expert_1"] < 0.2, \
        f"Expected low retention for near-duplicate: {report['signal_retention']['expert_1']}"

    print("PASS: Gram-Schmidt signal kill criterion test")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run unit tests first
    print("Running unit tests...")
    test_gram_schmidt_orthogonality()
    test_gram_schmidt_with_orthogonal_inputs()
    test_gram_schmidt_signal_kill_criterion()
    print()

    # Run full experiment
    print("=" * 70)
    print("PHASE 1: N=5 DOMAINS (quintary split, 3 seeds)")
    print("=" * 70)
    results_n5 = run_multiseed(seeds=(42, 123, 7), n_domains=5)

    print("\n\n")
    print("=" * 70)
    print("PHASE 2: N=2 DOMAINS (binary split, 3 seeds)")
    print("=" * 70)
    results_n2 = run_multiseed(seeds=(42, 123, 7), n_domains=2)
