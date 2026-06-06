"""N=8 Identity Boundary: Capsule identity Jaccard at N=8 domains.

N=5 experiment (n5_identity_scaling) showed:
  - Combined Jaccard = 0.792 at N=5 (above 0.70 threshold)
  - Linear degradation ~0.026 Jaccard per additional domain
  - Extrapolated safe limit ~N=8 where J ~ 0.71

This experiment validates the linear extrapolation by measuring at N=8 using
an octonary split (a-c, d-f, g-i, j-l, m-o, p-r, s-u, v-z). It also
tests whether the degradation remains linear or shows a phase transition
at higher N.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen), 8 domains
  3. Profile each single-domain model: record per-capsule dead/alive set
  4. Compose at N=2,3,4,5,6,7,8 and profile each
  5. Compare dead sets: Jaccard, overlap coefficient, decomposition
  6. Repeat for 3 seeds

Kill criteria:
  - Combined Jaccard at N=8 drops below 0.70
  - Per-domain minimum Jaccard below 0.50 for any domain
"""

import copy
import statistics

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
from ..n5_identity_scaling.n5_identity_scaling import (
    compose_n_domains,
    decompose_n_domain,
)


def run_n8_experiment(seed=42):
    """Run the N=8 capsule identity tracking experiment for one seed.

    Returns dict with all profiling results and identity comparisons,
    including intermediate N values (2, 3, 4, 5, 6, 7, 8).
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="octonary")

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
    assert len(domain_names) == 8, f"Expected 8 domains, got {len(domain_names)}"

    # ============================================================
    # 1. Pretrain base + domain fine-tune
    # ============================================================
    print(f"  [1/5] Pretrain base + fine-tune per 8 domains...")
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
    # 3. Compose at N=2,3,4,5,6,7,8 and profile each
    # ============================================================
    print(f"  [3/5] Compose at increasing N and profile...")

    n_sweep_results = {}
    for n in range(2, 9):
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
    # 4. Profile N=8 composed model per-domain
    # ============================================================
    print(f"  [4/5] Profile N=8 composed model per-domain...")

    composed_8 = compose_n_domains(base, [domain_models[d] for d in domain_names])
    composed_per_domain_dead = {}
    for d_name in domain_names:
        freqs_cd = profile_activations(
            composed_8, domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        dead_cd, _, per_layer_cd = get_dead_set(freqs_cd)
        composed_per_domain_dead[d_name] = dead_cd
        total_dead_cd = sum(nd for nd, nt in per_layer_cd)
        total_caps_cd = sum(nt for nd, nt in per_layer_cd)
        print(f"    Composed N=8 (on {d_name}): {total_dead_cd}/{total_caps_cd} dead "
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
    for n in range(2, 9):
        r = result["n_sweep_results"][n]
        print(f"    N={n}: combined_Jaccard={r['decomposition']['combined_jaccard']:.3f}, "
              f"combined_overlap={r['decomposition']['combined_overlap']:.3f}, "
              f"death_rate={r['death_rate']:.1f}%")

    print(f"\n  --- N=8 per-domain decomposition ---")
    decomp8 = result["n_sweep_results"][8]["decomposition"]
    for d_name, d in decomp8["per_domain"].items():
        print(f"    {d_name}: single_dead={d['dead_single']}, composed_half_dead={d['dead_composed_half']}, "
              f"J={d['jaccard']:.3f}, OC={d['overlap_coeff']:.3f}, "
              f"killed_by_comp={d['dead_only_composed']}, revived={d['dead_only_single']}")

    print(f"\n  --- N=8 composed on own-domain data ---")
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
        result = run_n8_experiment(seed=seed)
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

    for n in range(2, 9):
        jac_vals = [r["n_sweep_results"][n]["decomposition"]["combined_jaccard"]
                    for r in all_results]
        ovl_vals = [r["n_sweep_results"][n]["decomposition"]["combined_overlap"]
                    for r in all_results]
        dr_vals = [r["n_sweep_results"][n]["death_rate"] for r in all_results]

        print(f"  {n:>3} | {statistics.mean(jac_vals):>7.3f} {statistics.stdev(jac_vals):>7.3f} "
              f"| {statistics.mean(ovl_vals):>7.3f} {statistics.stdev(ovl_vals):>7.3f} "
              f"| {statistics.mean(dr_vals):>6.1f} {statistics.stdev(dr_vals):>6.1f}")

    # ============================================================
    # N=8 per-domain Jaccard
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=8 Per-Domain Jaccard (3-seed mean)")
    print(f"{'='*70}")

    print(f"\n  {'Domain':<8} | {'Jaccard':>8} {'Std':>8} | {'Overlap':>8} {'Std':>8} | "
          f"{'Killed':>7} {'Revived':>8}")
    print("  " + "-" * 75)

    for d_name in domain_names:
        jacs = [r["n_sweep_results"][8]["decomposition"]["per_domain"][d_name]["jaccard"]
                for r in all_results]
        ovls = [r["n_sweep_results"][8]["decomposition"]["per_domain"][d_name]["overlap_coeff"]
                for r in all_results]
        killed = [r["n_sweep_results"][8]["decomposition"]["per_domain"][d_name]["dead_only_composed"]
                  for r in all_results]
        revived = [r["n_sweep_results"][8]["decomposition"]["per_domain"][d_name]["dead_only_single"]
                   for r in all_results]

        print(f"  {d_name:<8} | {statistics.mean(jacs):>7.3f} {statistics.stdev(jacs):>7.3f} "
              f"| {statistics.mean(ovls):>7.3f} {statistics.stdev(ovls):>7.3f} "
              f"| {statistics.mean(killed):>6.1f} {statistics.mean(revived):>7.1f}")

    # ============================================================
    # N=8 composed on own-domain
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=8 Composed on Own-Domain Data (3-seed mean)")
    print(f"{'='*70}")

    print(f"\n  {'Domain':<8} | {'Jaccard':>8} {'Std':>8} | {'Overlap':>8} {'Std':>8}")
    print("  " + "-" * 50)

    for d_name in domain_names:
        jacs = [r["composed_on_own_domain"][d_name]["jaccard"] for r in all_results]
        ovls = [r["composed_on_own_domain"][d_name]["overlap"] for r in all_results]
        print(f"  {d_name:<8} | {statistics.mean(jacs):>7.3f} {statistics.stdev(jacs):>7.3f} "
              f"| {statistics.mean(ovls):>7.3f} {statistics.stdev(ovls):>7.3f}")

    # ============================================================
    # Per-layer Jaccard at N=8
    # ============================================================
    print(f"\n{'='*70}")
    print("  N=8 Per-Layer Jaccard (mean across all domains, 3 seeds)")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>5} | {'Mean J':>8} {'Std':>8}")
    print("  " + "-" * 30)

    for l in range(BASE["n_layer"]):
        all_layer_jacs = []
        for r in all_results:
            for d_name in domain_names:
                plj = r["n_sweep_results"][8]["per_layer_jaccards"][d_name]
                all_layer_jacs.append(plj[l][1])  # [1] is Jaccard
        print(f"  {l:>5} | {statistics.mean(all_layer_jacs):>7.3f} "
              f"{statistics.stdev(all_layer_jacs):>7.3f}")

    # ============================================================
    # Linearity test: compare linear extrapolation to actual
    # ============================================================
    print(f"\n{'='*70}")
    print("  Linearity Test: Does the N=5 linear model hold?")
    print(f"{'='*70}")

    # N=5 experiment found: J(N=2) ~ 0.871, degradation ~ 0.026/domain
    # Predicted J(N=8) = 0.871 - 0.026 * 6 = 0.715
    # (or from N=5: 0.792 - 0.026 * 3 = 0.714)
    n5_predicted_j8 = 0.792 - 0.026 * 3

    # Fit our own linear model from our N-sweep
    jac_by_n = {}
    for n in range(2, 9):
        jac_by_n[n] = statistics.mean(
            [r["n_sweep_results"][n]["decomposition"]["combined_jaccard"]
             for r in all_results]
        )

    # Linear regression: J = a + b*N
    ns = list(range(2, 9))
    jacs = [jac_by_n[n] for n in ns]
    n_mean = statistics.mean(ns)
    j_mean = statistics.mean(jacs)
    numerator = sum((n - n_mean) * (j - j_mean) for n, j in zip(ns, jacs))
    denominator = sum((n - n_mean) ** 2 for n in ns)
    slope = numerator / denominator if denominator else 0
    intercept = j_mean - slope * n_mean

    # Residuals
    residuals = [j - (intercept + slope * n) for n, j in zip(ns, jacs)]
    rmse = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5

    print(f"\n  N=5 experiment prediction for N=8: J = {n5_predicted_j8:.3f}")
    print(f"  Actual N=8 combined Jaccard:       J = {jac_by_n[8]:.3f}")
    print(f"  Prediction error:                  {jac_by_n[8] - n5_predicted_j8:+.3f}")
    print(f"\n  Linear fit from this experiment: J = {intercept:.3f} + {slope:.4f} * N")
    print(f"  Degradation per domain: {slope:.4f}")
    print(f"  Linear fit RMSE: {rmse:.4f}")
    print(f"\n  N-sweep trajectory:")
    for n in ns:
        predicted = intercept + slope * n
        actual = jac_by_n[n]
        print(f"    N={n}: actual={actual:.3f}, linear_fit={predicted:.3f}, "
              f"residual={actual - predicted:+.4f}")

    # Extrapolate: when does J reach 0.70?
    if slope < 0:
        n_threshold = (0.70 - intercept) / slope
        print(f"\n  Extrapolated N at J=0.70: {n_threshold:.1f}")
    else:
        print(f"\n  Jaccard is NOT decreasing with N (slope={slope:.4f})")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Primary metric: combined Jaccard at N=8
    n8_jac = [r["n_sweep_results"][8]["decomposition"]["combined_jaccard"]
              for r in all_results]
    mean_n8_jac = statistics.mean(n8_jac)
    std_n8_jac = statistics.stdev(n8_jac)

    # All per-domain Jaccards at N=8
    all_per_domain_jacs = []
    for r in all_results:
        for d_name in domain_names:
            all_per_domain_jacs.append(
                r["n_sweep_results"][8]["decomposition"]["per_domain"][d_name]["jaccard"]
            )

    min_domain_jac = min(all_per_domain_jacs)
    max_domain_jac = max(all_per_domain_jacs)

    print(f"\n  Kill criterion 1: Combined Jaccard < 0.70 at N=8")
    print(f"  Combined Jaccard (N=8):  {mean_n8_jac:.3f} +/- {std_n8_jac:.3f}")
    killed_combined = mean_n8_jac < 0.70

    print(f"\n  Kill criterion 2: Per-domain minimum Jaccard < 0.50")
    print(f"  Min per-domain Jaccard:  {min_domain_jac:.3f}")
    print(f"  Max per-domain Jaccard:  {max_domain_jac:.3f}")
    killed_perdomain = min_domain_jac < 0.50

    print(f"\n  | Criterion                | Value  | Threshold | Result |")
    print(f"  |--------------------------|--------|-----------|--------|")
    r1 = "KILL" if killed_combined else "PASS"
    r2 = "KILL" if killed_perdomain else "PASS"
    print(f"  | Combined J at N=8        | {mean_n8_jac:.3f}  | <0.70     | {r1:>6} |")
    print(f"  | Min per-domain J (any)   | {min_domain_jac:.3f}  | <0.50     | {r2:>6} |")

    n_killed = sum([killed_combined, killed_perdomain])
    if n_killed > 0:
        print(f"\n  VERDICT: KILL. {n_killed}/2 kill criteria triggered.")
        if killed_combined:
            print(f"  Combined Jaccard {mean_n8_jac:.3f} < 0.70.")
            print("  Pre-composition profiling is NOT safe at N=8 domains.")
            print("  The linear degradation model from N=5 was (approximately) correct.")
        if killed_perdomain:
            print(f"  Per-domain minimum Jaccard {min_domain_jac:.3f} < 0.50.")
            print("  Some domain-seed combinations have severely degraded identity.")
    else:
        print(f"\n  VERDICT: PASS. 0/2 kill criteria triggered.")
        print(f"  Combined Jaccard {mean_n8_jac:.3f} >= 0.70.")
        if min_domain_jac < 0.70:
            print(f"  Note: min per-domain Jaccard = {min_domain_jac:.3f} < 0.70.")
            print("  Individual domain-seed combinations breach the combined threshold.")
        print("  Pre-composition profiling remains safe at N=8 domains.")
        print("  The N=5 linear extrapolation was conservative.")

    # ============================================================
    # Comparison with N=5 experiment
    # ============================================================
    print(f"\n{'='*70}")
    print("  Comparison with N=5 Experiment")
    print(f"{'='*70}")

    jac_2 = jac_by_n[2]
    jac_5 = jac_by_n[5]
    jac_8 = jac_by_n[8]

    degradation_2_to_5 = (jac_2 - jac_5) / 3
    degradation_5_to_8 = (jac_5 - jac_8) / 3
    degradation_2_to_8 = (jac_2 - jac_8) / 6

    print(f"""
  Cross-experiment comparison:
    N=5 exp (quintary):  J(N=2)={0.871:.3f}, J(N=5)={0.792:.3f}, rate={0.026:.3f}/domain
    This exp (octonary): J(N=2)={jac_2:.3f}, J(N=5)={jac_5:.3f}, J(N=8)={jac_8:.3f}

  Degradation rates:
    N=2 to N=5: {degradation_2_to_5:.4f}/domain
    N=5 to N=8: {degradation_5_to_8:.4f}/domain
    N=2 to N=8: {degradation_2_to_8:.4f}/domain (overall)

  Phase transition test:
    If linear: rate(5->8) should be similar to rate(2->5)
    Ratio: rate(5->8) / rate(2->5) = {degradation_5_to_8 / max(degradation_2_to_5, 0.001):.2f}
    (1.0 = perfectly linear, >1.5 = superlinear/phase transition, <0.7 = sublinear)
""")

    return all_results


if __name__ == "__main__":
    main()
