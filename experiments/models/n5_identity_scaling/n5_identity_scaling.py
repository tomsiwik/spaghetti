"""N=5 Identity Scaling: Capsule identity Jaccard at N=5 domains.

Exp 16 validated identity tracking at N=2 (Jaccard=0.895, well above 0.50).
Adversarial review flagged that perturbation grows linearly with N. This
experiment measures at N=5 to determine if pre-composition pruning remains
safe at higher fan-out.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen), 5 domains
  3. Profile each single-domain model: record per-capsule dead/alive set
  4. Compose all 5 domains by concatenating weight matrices
  5. Profile composed model: record per-capsule dead/alive set
  6. Compare dead sets: Jaccard, overlap coefficient, decomposition
  7. Additionally: measure Jaccard at intermediate N (2, 3, 4, 5) to see trajectory
  8. Repeat for 3 seeds

Kill criterion:
  Jaccard between single-domain and N=5 composed dead sets drops below 0.70
"""

import copy
import statistics
import itertools

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    compose_relu_models,
    _make_relu_model, _freeze_attention, _eval_domains,
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE,
    BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations
from ..capsule_identity.capsule_identity import (
    get_dead_set,
    jaccard_similarity,
    overlap_coefficient,
    per_layer_jaccard,
)


def compose_n_domains(base_model, domain_models):
    """Compose N domain-specific MLP pools by weight concatenation.

    Generalizes compose_relu_models to N >= 2 domains. Capsule indices
    in the composed model: domain k occupies [k*P, (k+1)*P - 1] per layer.
    """
    return compose_relu_models(base_model, domain_models)


def decompose_n_domain(single_dead_sets, dead_composed, n_capsules_per_domain, domain_names):
    """Decompose composed dead set for N domains.

    For each domain k, extract its capsule indices from the composed dead set
    and compute Jaccard against the single-domain dead set.

    Args:
        single_dead_sets: dict {domain_name: set of (layer, capsule_idx)}
        dead_composed: set of (layer, capsule_idx) in composed model
        n_capsules_per_domain: P
        domain_names: ordered list of domain names

    Returns:
        dict with per-domain and combined metrics
    """
    P = n_capsules_per_domain
    N = len(domain_names)

    per_domain = {}
    for k, d_name in enumerate(domain_names):
        offset = k * P
        # Extract domain k's half from composed dead set
        composed_k = {(l, c - offset) for (l, c) in dead_composed
                      if offset <= c < offset + P}

        single_k = single_dead_sets[d_name]

        dead_in_both = single_k & composed_k
        dead_only_single = single_k - composed_k
        dead_only_composed = composed_k - single_k

        per_domain[d_name] = {
            "dead_single": len(single_k),
            "dead_composed_half": len(composed_k),
            "dead_in_both": len(dead_in_both),
            "dead_only_single": len(dead_only_single),
            "dead_only_composed": len(dead_only_composed),
            "jaccard": jaccard_similarity(single_k, composed_k),
            "overlap_coeff": overlap_coefficient(single_k, composed_k),
        }

    # Combined: union of all single-domain dead sets vs composed
    union_single = set()
    for k, d_name in enumerate(domain_names):
        offset = k * P
        union_single |= {(l, c + offset) for (l, c) in single_dead_sets[d_name]}

    combined_jaccard = jaccard_similarity(union_single, dead_composed)
    combined_overlap = overlap_coefficient(union_single, dead_composed)

    return {
        "per_domain": per_domain,
        "combined_jaccard": combined_jaccard,
        "combined_overlap": combined_overlap,
    }


def run_n5_experiment(seed=42):
    """Run the N=5 capsule identity tracking experiment for one seed.

    Returns dict with all profiling results and identity comparisons,
    including intermediate N values (2, 3, 4, 5).
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="quintary")

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = list(domain_datasets.keys())
    assert len(domain_names) == 5, f"Expected 5 domains, got {len(domain_names)}"

    # ============================================================
    # 1. Pretrain base + domain fine-tune
    # ============================================================
    print(f"  [1/5] Pretrain base + fine-tune per 5 domains...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # ============================================================
    # 2. Profile single-domain models on own-domain data
    # ============================================================
    print(f"  [2/5] Profile single-domain models...")

    single_domain_dead = {}
    single_domain_stats = {}

    for d_name in domain_names:
        freqs = profile_activations(
            domain_models[d_name],
            domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        dead_set, dead_flat, per_layer = get_dead_set(freqs)
        single_domain_dead[d_name] = dead_set

        total_dead = sum(nd for nd, nt in per_layer)
        total_caps = sum(nt for nd, nt in per_layer)
        single_domain_stats[d_name] = {
            "dead_set_size": len(dead_set),
            "total_capsules": total_caps,
            "death_rate": total_dead / total_caps * 100,
            "per_layer": per_layer,
        }
        print(f"    {d_name}: {total_dead}/{total_caps} dead "
              f"({total_dead / total_caps * 100:.1f}%)")

    # ============================================================
    # 3. Compose at N=2,3,4,5 and profile each
    # ============================================================
    print(f"  [3/5] Compose at increasing N and profile...")

    n_sweep_results = {}
    for n in [2, 3, 4, 5]:
        # Use first n domains
        subset_names = domain_names[:n]
        subset_models = [domain_models[d] for d in subset_names]

        composed = compose_n_domains(base, subset_models)

        # Profile on joint validation data
        freqs_composed = profile_activations(
            composed, joint_val,
            n_batches=20, batch_size=32, seed=seed,
        )
        dead_composed, _, per_layer_composed = get_dead_set(freqs_composed)

        total_dead_c = sum(nd for nd, nt in per_layer_composed)
        total_caps_c = sum(nt for nd, nt in per_layer_composed)
        death_rate_c = total_dead_c / total_caps_c * 100

        # Decompose: per-domain Jaccard
        decomp = decompose_n_domain(
            {d: single_domain_dead[d] for d in subset_names},
            dead_composed,
            N_CAPSULES,
            subset_names,
        )

        # Per-layer Jaccard for each domain
        per_layer_jaccards = {}
        for k, d_name in enumerate(subset_names):
            offset = k * N_CAPSULES
            composed_k_dead = {(l, c - offset) for (l, c) in dead_composed
                               if offset <= c < offset + N_CAPSULES}
            plj = per_layer_jaccard(
                single_domain_dead[d_name],
                composed_k_dead,
                n_layers=BASE["n_layer"],
            )
            per_layer_jaccards[d_name] = plj

        n_sweep_results[n] = {
            "subset_names": subset_names,
            "dead_composed": dead_composed,
            "death_rate": death_rate_c,
            "decomposition": decomp,
            "per_layer_jaccards": per_layer_jaccards,
        }

        print(f"    N={n}: {total_dead_c}/{total_caps_c} dead ({death_rate_c:.1f}%), "
              f"combined Jaccard={decomp['combined_jaccard']:.3f}, "
              f"overlap={decomp['combined_overlap']:.3f}")

    # ============================================================
    # 4. Profile N=5 composed model per-domain
    # ============================================================
    print(f"  [4/5] Profile N=5 composed model per-domain...")

    composed_5 = compose_n_domains(base, [domain_models[d] for d in domain_names])
    composed_per_domain_dead = {}
    for d_name in domain_names:
        freqs_cd = profile_activations(
            composed_5, domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        dead_cd, _, per_layer_cd = get_dead_set(freqs_cd)
        composed_per_domain_dead[d_name] = dead_cd
        total_dead_cd = sum(nd for nd, nt in per_layer_cd)
        total_caps_cd = sum(nt for nd, nt in per_layer_cd)
        print(f"    Composed N=5 (on {d_name}): {total_dead_cd}/{total_caps_cd} dead "
              f"({total_dead_cd / total_caps_cd * 100:.1f}%)")

    # Per-domain Jaccard when profiling composed on own-domain data
    composed_on_own = {}
    for k, d_name in enumerate(domain_names):
        offset = k * N_CAPSULES
        composed_k_on_own = {(l, c - offset) for (l, c) in composed_per_domain_dead[d_name]
                             if offset <= c < offset + N_CAPSULES}
        j = jaccard_similarity(single_domain_dead[d_name], composed_k_on_own)
        o = overlap_coefficient(single_domain_dead[d_name], composed_k_on_own)
        composed_on_own[d_name] = {"jaccard": j, "overlap": o}
        print(f"    {d_name} single vs composed(own data): J={j:.3f}, OC={o:.3f}")

    # ============================================================
    # 5. Compile results
    # ============================================================
    print(f"  [5/5] Computing summary metrics...")

    result = {
        "domain_names": domain_names,
        "single_domain_stats": single_domain_stats,
        "single_domain_dead": single_domain_dead,
        "n_sweep_results": n_sweep_results,
        "composed_per_domain_dead": composed_per_domain_dead,
        "composed_on_own_domain": composed_on_own,
    }

    return result


def print_single_seed_summary(result):
    """Print summary for one seed."""
    domain_names = result["domain_names"]

    print(f"\n  --- N-sweep: Combined Jaccard trajectory ---")
    for n in [2, 3, 4, 5]:
        r = result["n_sweep_results"][n]
        print(f"    N={n}: combined_Jaccard={r['decomposition']['combined_jaccard']:.3f}, "
              f"combined_overlap={r['decomposition']['combined_overlap']:.3f}, "
              f"death_rate={r['death_rate']:.1f}%")

    print(f"\n  --- N=5 per-domain decomposition ---")
    decomp5 = result["n_sweep_results"][5]["decomposition"]
    for d_name, d in decomp5["per_domain"].items():
        print(f"    {d_name}: single_dead={d['dead_single']}, composed_half_dead={d['dead_composed_half']}, "
              f"J={d['jaccard']:.3f}, OC={d['overlap_coeff']:.3f}, "
              f"killed_by_comp={d['dead_only_composed']}, revived={d['dead_only_single']}")

    print(f"\n  --- N=5 composed on own-domain data ---")
    for d_name in domain_names:
        cod = result["composed_on_own_domain"][d_name]
        print(f"    {d_name}: J={cod['jaccard']:.3f}, OC={cod['overlap']:.3f}")


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        result = run_n5_experiment(seed=seed)
        all_results.append(result)
        print_single_seed_summary(result)

    domain_names = all_results[0]["domain_names"]

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    # Single-domain death rates
    print(f"\n  --- Single-Domain Death Rates ---")
    for d_name in domain_names:
        rates = [r["single_domain_stats"][d_name]["death_rate"] for r in all_results]
        print(f"    {d_name}: {statistics.mean(rates):.1f}% "
              f"(std={statistics.stdev(rates):.1f}%)")

    # ============================================================
    # KEY: N-sweep combined Jaccard trajectory
    # ============================================================
    print(f"\n{'='*70}")
    print("  KEY METRIC: Combined Jaccard vs N (3-seed mean)")
    print(f"{'='*70}")

    print(f"\n  {'N':>3} | {'Jaccard':>8} {'Std':>8} | {'Overlap':>8} {'Std':>8} | {'Death%':>7} {'Std':>7}")
    print("  " + "-" * 65)

    for n in [2, 3, 4, 5]:
        jac_vals = [r["n_sweep_results"][n]["decomposition"]["combined_jaccard"]
                    for r in all_results]
        ovl_vals = [r["n_sweep_results"][n]["decomposition"]["combined_overlap"]
                    for r in all_results]
        dr_vals = [r["n_sweep_results"][n]["death_rate"] for r in all_results]

        print(f"  {n:>3} | {statistics.mean(jac_vals):>7.3f} {statistics.stdev(jac_vals):>7.3f} "
              f"| {statistics.mean(ovl_vals):>7.3f} {statistics.stdev(ovl_vals):>7.3f} "
              f"| {statistics.mean(dr_vals):>6.1f} {statistics.stdev(dr_vals):>6.1f}")

    # ============================================================
    # N=5 per-domain Jaccard
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=5 Per-Domain Jaccard (3-seed mean)")
    print(f"{'='*70}")

    print(f"\n  {'Domain':<8} | {'Jaccard':>8} {'Std':>8} | {'Overlap':>8} {'Std':>8} | "
          f"{'Killed':>7} {'Revived':>8}")
    print("  " + "-" * 75)

    for d_name in domain_names:
        jacs = [r["n_sweep_results"][5]["decomposition"]["per_domain"][d_name]["jaccard"]
                for r in all_results]
        ovls = [r["n_sweep_results"][5]["decomposition"]["per_domain"][d_name]["overlap_coeff"]
                for r in all_results]
        killed = [r["n_sweep_results"][5]["decomposition"]["per_domain"][d_name]["dead_only_composed"]
                  for r in all_results]
        revived = [r["n_sweep_results"][5]["decomposition"]["per_domain"][d_name]["dead_only_single"]
                   for r in all_results]

        print(f"  {d_name:<8} | {statistics.mean(jacs):>7.3f} {statistics.stdev(jacs):>7.3f} "
              f"| {statistics.mean(ovls):>7.3f} {statistics.stdev(ovls):>7.3f} "
              f"| {statistics.mean(killed):>6.1f} {statistics.mean(revived):>7.1f}")

    # ============================================================
    # N=5 composed on own-domain
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=5 Composed on Own-Domain Data (3-seed mean)")
    print(f"{'='*70}")

    print(f"\n  {'Domain':<8} | {'Jaccard':>8} {'Std':>8} | {'Overlap':>8} {'Std':>8}")
    print("  " + "-" * 50)

    for d_name in domain_names:
        jacs = [r["composed_on_own_domain"][d_name]["jaccard"] for r in all_results]
        ovls = [r["composed_on_own_domain"][d_name]["overlap"] for r in all_results]
        print(f"  {d_name:<8} | {statistics.mean(jacs):>7.3f} {statistics.stdev(jacs):>7.3f} "
              f"| {statistics.mean(ovls):>7.3f} {statistics.stdev(ovls):>7.3f}")

    # ============================================================
    # Per-layer Jaccard at N=5
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=5 Per-Layer Jaccard (mean across all domains, 3 seeds)")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>5} | {'Mean J':>8} {'Std':>8}")
    print("  " + "-" * 30)

    for l in range(BASE["n_layer"]):
        all_layer_jacs = []
        for r in all_results:
            for d_name in domain_names:
                plj = r["n_sweep_results"][5]["per_layer_jaccards"][d_name]
                all_layer_jacs.append(plj[l][1])  # [1] is Jaccard
        print(f"  {l:>5} | {statistics.mean(all_layer_jacs):>7.3f} "
              f"{statistics.stdev(all_layer_jacs):>7.3f}")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Primary metric: combined Jaccard at N=5
    n5_jac = [r["n_sweep_results"][5]["decomposition"]["combined_jaccard"]
              for r in all_results]
    mean_n5_jac = statistics.mean(n5_jac)
    std_n5_jac = statistics.stdev(n5_jac)

    # All per-domain Jaccards at N=5
    all_per_domain_jacs = []
    for r in all_results:
        for d_name in domain_names:
            all_per_domain_jacs.append(
                r["n_sweep_results"][5]["decomposition"]["per_domain"][d_name]["jaccard"]
            )

    min_domain_jac = min(all_per_domain_jacs)
    max_domain_jac = max(all_per_domain_jacs)

    print(f"\n  Kill criterion: Jaccard < 0.70 at N=5")
    print(f"  (between single-domain and N=5 composed dead sets)")
    print()
    print(f"  Combined Jaccard (N=5):  {mean_n5_jac:.3f} +/- {std_n5_jac:.3f}")
    print(f"  Min per-domain Jaccard:  {min_domain_jac:.3f}")
    print(f"  Max per-domain Jaccard:  {max_domain_jac:.3f}")

    # Trajectory from N=2 to N=5
    n2_jac = statistics.mean([r["n_sweep_results"][2]["decomposition"]["combined_jaccard"]
                              for r in all_results])
    print(f"\n  Trajectory:")
    print(f"    N=2 combined Jaccard: {n2_jac:.3f} (Exp 16 reference: 0.895)")
    print(f"    N=5 combined Jaccard: {mean_n5_jac:.3f}")
    print(f"    Degradation N=2->N=5: {(n2_jac - mean_n5_jac):.3f}")

    # Verdict
    killed = mean_n5_jac < 0.70

    if killed:
        print(f"\n  VERDICT: KILL. Combined Jaccard {mean_n5_jac:.3f} < 0.70.")
        print("  Pre-composition profiling is NOT safe at N=5 domains.")
        print("  Post-composition profiling is required at higher fan-out.")
    else:
        if min_domain_jac < 0.70:
            print(f"\n  VERDICT: PASS (with caveat). Combined Jaccard {mean_n5_jac:.3f} >= 0.70")
            print(f"  BUT min per-domain Jaccard = {min_domain_jac:.3f} < 0.70.")
            print("  Some domains may have degraded identity preservation.")
        else:
            print(f"\n  VERDICT: PASS. Combined Jaccard {mean_n5_jac:.3f} >= 0.70.")
            print("  Pre-composition profiling remains safe at N=5 domains.")
            print("  Death identity is preserved despite 5-domain composition.")

    # ============================================================
    # Interpretation
    # ============================================================
    print(f"\n{'='*70}")
    print("  Interpretation")
    print(f"{'='*70}")

    # Compute degradation rate per additional domain
    jac_2 = statistics.mean([r["n_sweep_results"][2]["decomposition"]["combined_jaccard"]
                             for r in all_results])
    jac_5 = mean_n5_jac
    degradation_per_domain = (jac_2 - jac_5) / 3  # 3 additional domains

    print(f"""
  Exp 16 showed Jaccard=0.895 at N=2 (binary: a-m vs n-z).
  This experiment extends to N=5 (quintary: a-e, f-j, k-o, p-t, u-z).

  N-sweep results:
    - Combined Jaccard degrades from {jac_2:.3f} (N=2) to {jac_5:.3f} (N=5)
    - Degradation rate: ~{degradation_per_domain:.3f} per additional domain
    - At this rate, Jaccard would reach 0.70 at N ~ {2 + (jac_2 - 0.70) / max(degradation_per_domain, 0.001):.0f}

  The perturbation from composition scales with N (each additional domain
  adds its capsule residuals to the hidden state). The key question is
  whether this perturbation accumulates fast enough to cross the 0.70
  threshold at practical domain counts.
""")

    return all_results


if __name__ == "__main__":
    main()
