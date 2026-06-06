"""Flat MoE N=8 Boundary: Composition gap + identity tracking at N=8 domains.

REVISED per adversarial review. Changes:
  - Added post-calibration Jaccard measurements at N=2 and N=5
  - Updated verdict logic to distinguish PARTIAL KILL from FULL KILL
  - Standardized terminology: "PARTIAL KILL" when 1/2 criteria triggered

Combines the composition quality measurement (gap vs joint) with capsule
identity tracking (Jaccard) at N=8 domains using the flat MoE protocol.

The N=5 identity scaling experiment extrapolated ~N=8 as the safe limit.
The N=8 identity boundary experiment confirmed J=0.800 (sublinear degradation).
Sequential freeze-graft confirmed flat MoE is the only viable N>2 strategy.

This experiment answers: does the flat MoE protocol produce acceptable
composition QUALITY (not just identity preservation) at N=8?

Protocol:
  1. Pretrain base model on ALL data (300 steps)
  2. Fine-tune only MLP weights per domain (attention frozen, 200 steps), N domains
  3. Joint training baseline: train on round-robin mixed domain data (N*200 steps)
  4. Compose all N domains by concatenating weight matrices
  5. Calibrate router on mixed-domain data (scaled steps)
  6. Profile dead capsules for Jaccard measurement (post-calibration)
  7. Evaluate: composition gap = (composed_loss - joint_loss) / joint_loss
  8. Repeat for 3 seeds

Kill criteria:
  - composition gap >10% at N=8
  - combined Jaccard <0.60

Revision adds: same protocol at N=2 (100 cal steps) and N=5 (200 cal steps)
to disambiguate N-scaling from universal calibration effects on Jaccard.
"""

import copy
import random
import statistics
import time
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
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
)
from ..n5_identity_scaling.n5_identity_scaling import (
    compose_n_domains,
    decompose_n_domain,
)


# Calibration steps scaled per N (reviewer requested these exact values)
CALIBRATION_STEPS_BY_N = {
    2: 100,
    5: 200,
    8: 300,
}


def calibrate_router_relu(composed, domain_train_datasets, steps, lr=LR, seed=42):
    """Calibrate by training the full model (all params unfrozen) on mixed data.

    For ReLURouterGPT there is no explicit router -- the capsule pool is a
    single A/B matrix. Calibration fine-tunes all params on mixed domain data
    (round-robin across domains) to close the function-space gap.
    """
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(composed, ntp_loss)

    ds_list = list(domain_train_datasets.values())
    n_ds = len(ds_list)

    for step in range(1, steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(composed, inputs, targets)
        optimizer.update(composed, grads)
        mx.eval(composed.parameters(), optimizer.state)


def run_post_cal_jaccard(n_domains, split_method, cal_steps, seed=42):
    """Run composition + calibration + Jaccard profiling for a given N.

    This is the core measurement: pretrain, fine-tune per domain, compose,
    calibrate, then measure post-calibration Jaccard. Used for N=2, N=5, and N=8.

    Returns dict with composition gap, Jaccard, and all metrics.
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method=split_method)

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
    domain_trains = {d: domain_datasets[d][0] for d in domain_names}
    domain_vals = {d: domain_datasets[d][1] for d in domain_names}
    assert len(domain_names) == n_domains, f"Expected {n_domains} domains, got {len(domain_names)}"

    # ============================================================
    # 1. Joint training baseline (round-robin, same total steps)
    # ============================================================
    total_finetune_steps = n_domains * STEPS_FINETUNE
    print(f"    [1/6] Joint training baseline ({total_finetune_steps} steps)...")

    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * n_domains)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    ds_list = list(domain_trains.values())
    n_ds = len(ds_list)
    for step in range(1, total_finetune_steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    joint_per_domain = {}
    for d_name in domain_names:
        joint_per_domain[d_name] = evaluate(model_joint, domain_vals[d_name], BATCH_SIZE)
    joint_avg = sum(joint_per_domain.values()) / len(joint_per_domain)
    print(f"      Joint avg val loss: {joint_avg:.4f}")

    # ============================================================
    # 2. Pretrain base + fine-tune per domain
    # ============================================================
    print(f"    [2/6] Pretrain base ({STEPS_PRETRAIN} steps) + fine-tune per {n_domains} domains...")
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
    # 3. Compose all N domains (zero-shot)
    # ============================================================
    print(f"    [3/6] Compose {n_domains} domains (zero-shot)...")
    composed_zero = compose_n_domains(base, [domain_models[d] for d in domain_names])

    zero_per_domain = {}
    for d_name in domain_names:
        zero_per_domain[d_name] = evaluate(composed_zero, domain_vals[d_name], BATCH_SIZE)
    zero_avg = sum(zero_per_domain.values()) / len(zero_per_domain)
    zero_gap = (zero_avg - joint_avg) / joint_avg * 100
    print(f"      Zero-shot avg: {zero_avg:.4f}, gap: {zero_gap:+.2f}%")

    # ============================================================
    # 4. Calibrate (fine-tune on mixed data)
    # ============================================================
    print(f"    [4/6] Calibrate ({cal_steps} steps)...")
    composed_cal = compose_n_domains(base, [domain_models[d] for d in domain_names])
    calibrate_router_relu(composed_cal, domain_trains, steps=cal_steps,
                          lr=LR, seed=seed)

    cal_per_domain = {}
    for d_name in domain_names:
        cal_per_domain[d_name] = evaluate(composed_cal, domain_vals[d_name], BATCH_SIZE)
    cal_avg = sum(cal_per_domain.values()) / len(cal_per_domain)
    cal_gap = (cal_avg - joint_avg) / joint_avg * 100
    print(f"      Calibrated avg: {cal_avg:.4f}, gap: {cal_gap:+.2f}%")

    # ============================================================
    # 5. Profile capsule identity (Jaccard) -- POST-CALIBRATION
    # ============================================================
    print(f"    [5/6] Profile capsule identity (post-calibration)...")

    # Profile single-domain models on their own domain data
    # NOTE: This creates a distribution mismatch with the composed model
    # (profiled on joint data). See PAPER.md for discussion.
    single_domain_dead = {}
    for d_name in domain_names:
        freqs = profile_activations(
            domain_models[d_name],
            domain_datasets[d_name][1],
            n_batches=20, batch_size=32, seed=seed,
        )
        dead_set, _, _ = get_dead_set(freqs)
        single_domain_dead[d_name] = dead_set

    # Profile composed (calibrated) model on joint data
    freqs_composed = profile_activations(
        composed_cal, joint_val,
        n_batches=20, batch_size=32, seed=seed,
    )
    dead_composed, _, per_layer_composed = get_dead_set(freqs_composed)

    total_dead_c = sum(nd for nd, nt in per_layer_composed)
    total_caps_c = sum(nt for nd, nt in per_layer_composed)
    death_rate_c = total_dead_c / total_caps_c * 100

    # Decompose
    decomp = decompose_n_domain(
        single_domain_dead, dead_composed, N_CAPSULES, domain_names
    )

    combined_jaccard = decomp["combined_jaccard"]
    combined_overlap = decomp["combined_overlap"]
    print(f"      Combined Jaccard: {combined_jaccard:.3f}, Overlap: {combined_overlap:.3f}")
    print(f"      Death rate (composed): {death_rate_c:.1f}%")

    # Per-domain Jaccard
    per_domain_jaccards = {}
    for d_name, d_metrics in decomp["per_domain"].items():
        per_domain_jaccards[d_name] = d_metrics["jaccard"]

    # ============================================================
    # 6. Single-domain specialist average
    # ============================================================
    print(f"    [6/6] Evaluate single-domain specialists...")
    spec_per_domain = {}
    for d_name in domain_names:
        spec_per_domain[d_name] = evaluate(domain_models[d_name], domain_vals[d_name], BATCH_SIZE)
    spec_avg = sum(spec_per_domain.values()) / len(spec_per_domain)
    print(f"      Specialist avg: {spec_avg:.4f}")

    return {
        "seed": seed,
        "n_domains": n_domains,
        "cal_steps": cal_steps,
        "split_method": split_method,
        "joint_avg": joint_avg,
        "joint_per_domain": joint_per_domain,
        "zero_avg": zero_avg,
        "zero_gap": zero_gap,
        "zero_per_domain": zero_per_domain,
        "cal_avg": cal_avg,
        "cal_gap": cal_gap,
        "cal_per_domain": cal_per_domain,
        "spec_avg": spec_avg,
        "spec_per_domain": spec_per_domain,
        "combined_jaccard": combined_jaccard,
        "combined_overlap": combined_overlap,
        "per_domain_jaccards": per_domain_jaccards,
        "decomposition": {
            d_name: {
                "jaccard": d["jaccard"],
                "overlap_coeff": d["overlap_coeff"],
                "dead_single": d["dead_single"],
                "dead_composed_half": d["dead_composed_half"],
                "dead_only_composed": d["dead_only_composed"],
                "dead_only_single": d["dead_only_single"],
            }
            for d_name, d in decomp["per_domain"].items()
        },
        "death_rate_composed": death_rate_c,
    }


# Keep the original function name for backward compatibility
def run_single_seed(seed=42):
    """Run the full N=8 flat MoE boundary experiment for one seed."""
    return run_post_cal_jaccard(
        n_domains=8,
        split_method="octonary",
        cal_steps=CALIBRATION_STEPS_BY_N[8],
        seed=seed,
    )


def main():
    """Run across 3 seeds at N=2, N=5, and N=8. Report aggregate results."""
    t0 = time.time()
    seeds = [42, 123, 7]

    # Configuration for each N condition
    conditions = [
        {"n_domains": 2, "split_method": "binary",   "cal_steps": CALIBRATION_STEPS_BY_N[2]},
        {"n_domains": 5, "split_method": "quintary",  "cal_steps": CALIBRATION_STEPS_BY_N[5]},
        {"n_domains": 8, "split_method": "octonary",  "cal_steps": CALIBRATION_STEPS_BY_N[8]},
    ]

    all_results = {}  # {N: [results per seed]}

    for cond in conditions:
        n = cond["n_domains"]
        all_results[n] = []
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"  N={n}, Seed {seed} ({cond['split_method']}, {cond['cal_steps']} cal steps)")
            print(f"{'='*70}")
            result = run_post_cal_jaccard(
                n_domains=cond["n_domains"],
                split_method=cond["split_method"],
                cal_steps=cond["cal_steps"],
                seed=seed,
            )
            all_results[n].append(result)

    elapsed = time.time() - t0

    # ============================================================
    # Aggregate and compare across N
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  REVISED EXPERIMENT: Post-Calibration Jaccard at N=2, N=5, N=8")
    print(f"  ({elapsed:.0f}s total, 3 seeds each)")
    print(f"{'='*70}")

    print(f"\n  --- Post-Calibration Jaccard vs N ---")
    print(f"  {'N':>3} | {'Cal Steps':>9} | {'J mean':>8} {'J std':>8} | {'Gap mean':>8} {'Gap std':>8} | {'Death%':>7}")
    print("  " + "-" * 72)

    n_jaccard_means = {}
    for n in [2, 5, 8]:
        results = all_results[n]
        jacs = [r["combined_jaccard"] for r in results]
        gaps = [r["cal_gap"] for r in results]
        drs = [r["death_rate_composed"] for r in results]

        j_mean = statistics.mean(jacs)
        j_std = statistics.stdev(jacs)
        g_mean = statistics.mean(gaps)
        g_std = statistics.stdev(gaps)
        dr_mean = statistics.mean(drs)
        n_jaccard_means[n] = j_mean

        print(f"  {n:>3} | {CALIBRATION_STEPS_BY_N[n]:>9} | {j_mean:>7.3f} {j_std:>7.3f} "
              f"| {g_mean:>+7.2f}% {g_std:>7.2f}% | {dr_mean:>6.1f}%")

    # ============================================================
    # N=8 detailed results (primary experiment)
    # ============================================================
    n8_results = all_results[8]

    print(f"\n  --- N=8 Per-Seed Details ---")
    print(f"  {'Seed':>6} | {'Joint':>8} | {'Cal':>8} | {'Gap':>8} | {'Jaccard':>8}")
    print("  " + "-" * 50)
    for r in n8_results:
        print(f"  {r['seed']:>6} | {r['joint_avg']:>8.4f} | {r['cal_avg']:>8.4f} "
              f"| {r['cal_gap']:>+7.2f}% | {r['combined_jaccard']:>7.3f}")

    # Per-domain Jaccard at N=8
    domain_names_8 = list(n8_results[0]["per_domain_jaccards"].keys())
    all_per_domain_jacs = []
    print(f"\n  --- N=8 Per-Domain Jaccard ---")
    print(f"  {'Domain':<8} | {'J mean':>8} {'J std':>8} | {'Cal gap':>8}")
    print("  " + "-" * 42)
    for d_name in domain_names_8:
        jacs = [r["per_domain_jaccards"][d_name] for r in n8_results]
        cal_domain_gaps = []
        for r in n8_results:
            j_val = r["joint_per_domain"][d_name]
            c_val = r["cal_per_domain"][d_name]
            cal_domain_gaps.append((c_val - j_val) / j_val * 100)
        all_per_domain_jacs.extend(jacs)
        print(f"  {d_name:<8} | {statistics.mean(jacs):>7.3f} {statistics.stdev(jacs):>7.3f} "
              f"| {statistics.mean(cal_domain_gaps):>+7.2f}%")

    min_domain_jac = min(all_per_domain_jacs)
    max_domain_jac = max(all_per_domain_jacs)

    # Quality comparison
    print(f"\n  --- N=8 Quality Summary ---")
    joint_avgs = [r["joint_avg"] for r in n8_results]
    cal_avgs = [r["cal_avg"] for r in n8_results]
    zero_avgs = [r["zero_avg"] for r in n8_results]
    spec_avgs = [r["spec_avg"] for r in n8_results]
    print(f"  Joint:      {statistics.mean(joint_avgs):.4f} +/- {statistics.stdev(joint_avgs):.4f}")
    print(f"  Calibrated: {statistics.mean(cal_avgs):.4f} +/- {statistics.stdev(cal_avgs):.4f}")
    print(f"  Zero-shot:  {statistics.mean(zero_avgs):.4f} +/- {statistics.stdev(zero_avgs):.4f}")
    print(f"  Specialist: {statistics.mean(spec_avgs):.4f} +/- {statistics.stdev(spec_avgs):.4f}")

    # ============================================================
    # Kill threshold analysis (N=8, primary)
    # ============================================================
    print(f"\n{'='*70}")
    print("  KILL THRESHOLD ANALYSIS")
    print(f"{'='*70}")

    cal_gaps_8 = [r["cal_gap"] for r in n8_results]
    jaccards_8 = [r["combined_jaccard"] for r in n8_results]
    mean_cal_gap = statistics.mean(cal_gaps_8)
    mean_jaccard_8 = statistics.mean(jaccards_8)

    killed_gap = mean_cal_gap > 10.0
    killed_jaccard = mean_jaccard_8 < 0.60

    print(f"\n  | Criterion                | Value   | Threshold | Result |")
    print(f"  |--------------------------|---------|-----------|--------|")
    r1 = "KILL" if killed_gap else "PASS"
    r2 = "KILL" if killed_jaccard else "PASS"
    print(f"  | Composition gap (cal)    | {mean_cal_gap:>+6.2f}% | >10%      | {r1:>6} |")
    print(f"  | Combined Jaccard         | {mean_jaccard_8:>6.3f}  | <0.60     | {r2:>6} |")

    # Additional context
    print(f"\n  Min per-domain Jaccard (N=8):  {min_domain_jac:.3f}")
    print(f"  Max per-domain Jaccard (N=8):  {max_domain_jac:.3f}")

    n_killed = sum([killed_gap, killed_jaccard])

    # Verdict with PARTIAL KILL distinction (Fix #4)
    if n_killed == 2:
        verdict = "FULL KILL"
        print(f"\n  VERDICT: FULL KILL. 2/2 kill criteria triggered.")
    elif n_killed == 1:
        verdict = "PARTIAL KILL"
        print(f"\n  VERDICT: PARTIAL KILL. 1/2 kill criteria triggered.")
        if killed_jaccard:
            print("  Quality passes but identity tracking fails.")
        else:
            print("  Identity tracking passes but quality fails.")
    else:
        verdict = "PASS"
        print(f"\n  VERDICT: PASS. 0/2 kill criteria triggered.")
        print(f"  Flat MoE composition is viable at N=8 domains.")

    # ============================================================
    # KEY REVISION: Attribution analysis -- is J drop from N or calibration?
    # ============================================================
    print(f"\n{'='*70}")
    print("  ATTRIBUTION ANALYSIS: N-scaling vs Calibration Effect")
    print(f"{'='*70}")

    print(f"\n  Post-calibration Jaccard across N:")
    for n in [2, 5, 8]:
        jacs = [r["combined_jaccard"] for r in all_results[n]]
        print(f"    N={n}: J={statistics.mean(jacs):.3f} +/- {statistics.stdev(jacs):.3f} "
              f"({CALIBRATION_STEPS_BY_N[n]} cal steps)")

    # Compare with pre-calibration references
    print(f"\n  Pre-calibration Jaccard (from prior experiments, no calibration):")
    print(f"    N=2 (capsule_identity):      J=0.895")
    print(f"    N=5 (n5_identity_scaling):    J=0.792")
    print(f"    N=8 (n8_identity_boundary):   J=0.800")

    print(f"\n  Calibration-induced Jaccard drop (post-cal minus pre-cal reference):")
    pre_cal_refs = {2: 0.895, 5: 0.792, 8: 0.800}
    for n in [2, 5, 8]:
        post_cal = statistics.mean([r["combined_jaccard"] for r in all_results[n]])
        pre_cal = pre_cal_refs[n]
        drop = post_cal - pre_cal
        print(f"    N={n}: {post_cal:.3f} - {pre_cal:.3f} = {drop:+.3f}")

    # Determine if the Jaccard kill is N-specific or universal
    j2 = statistics.mean([r["combined_jaccard"] for r in all_results[2]])
    j5 = statistics.mean([r["combined_jaccard"] for r in all_results[5]])
    j8 = statistics.mean([r["combined_jaccard"] for r in all_results[8]])

    if j2 < 0.60:
        print(f"\n  FINDING: Post-cal Jaccard is below 0.60 even at N=2 (J={j2:.3f}).")
        print(f"  The Jaccard kill is a UNIVERSAL CALIBRATION EFFECT, not N=8-specific.")
        print(f"  The kill criterion threshold (0.60) is miscalibrated for post-cal measurement.")
    elif j5 < 0.60:
        print(f"\n  FINDING: Post-cal Jaccard drops below 0.60 between N=2 (J={j2:.3f}) and N=5 (J={j5:.3f}).")
        print(f"  The Jaccard kill onset is between N=2 and N=5, not at N=8.")
    else:
        print(f"\n  FINDING: Post-cal Jaccard remains above 0.60 at N=2 (J={j2:.3f}) and N=5 (J={j5:.3f}).")
        print(f"  The Jaccard kill IS N=8-specific (or at least N>5-specific).")

    # ============================================================
    # Comparison with prior experiments
    # ============================================================
    print(f"\n{'='*70}")
    print("  Comparison with Prior Experiments")
    print(f"{'='*70}")

    print(f"""
  Composition gap scaling:
    N=2 (this revision):   {statistics.mean([r['cal_gap'] for r in all_results[2]]):+.2f}%
    N=5 (this revision):   {statistics.mean([r['cal_gap'] for r in all_results[5]]):+.2f}%
    N=8 (this revision):   {mean_cal_gap:+.2f}%

  Identity (combined Jaccard) scaling -- POST-CALIBRATION:
    N=2: {j2:.3f}
    N=5: {j5:.3f}
    N=8: {j8:.3f}

  For comparison, PRE-CALIBRATION Jaccard (from prior experiments):
    N=2: 0.895
    N=5: 0.792
    N=8: 0.800
""")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "revision": "v2_post_cal_jaccard_at_all_N",
        "n_seeds": len(seeds),
        "elapsed_seconds": elapsed,
        "conditions": {
            str(n): {
                "n_domains": n,
                "cal_steps": CALIBRATION_STEPS_BY_N[n],
                "jaccard_mean": statistics.mean([r["combined_jaccard"] for r in all_results[n]]),
                "jaccard_std": statistics.stdev([r["combined_jaccard"] for r in all_results[n]]),
                "gap_mean": statistics.mean([r["cal_gap"] for r in all_results[n]]),
                "gap_std": statistics.stdev([r["cal_gap"] for r in all_results[n]]),
                "overlap_mean": statistics.mean([r["combined_overlap"] for r in all_results[n]]),
                "death_rate_mean": statistics.mean([r["death_rate_composed"] for r in all_results[n]]),
            }
            for n in [2, 5, 8]
        },
        "kill_criteria": {
            "gap_threshold": 10.0,
            "jaccard_threshold": 0.60,
            "mean_cal_gap_n8": mean_cal_gap,
            "mean_jaccard_n8": mean_jaccard_8,
            "killed_gap": killed_gap,
            "killed_jaccard": killed_jaccard,
            "verdict": verdict,
        },
        "n8_aggregate": {
            "cal_gap_mean": mean_cal_gap,
            "cal_gap_std": statistics.stdev(cal_gaps_8),
            "zero_gap_mean": statistics.mean([r["zero_gap"] for r in n8_results]),
            "zero_gap_std": statistics.stdev([r["zero_gap"] for r in n8_results]),
            "jaccard_mean": mean_jaccard_8,
            "jaccard_std": statistics.stdev(jaccards_8),
            "overlap_mean": statistics.mean([r["combined_overlap"] for r in n8_results]),
            "overlap_std": statistics.stdev([r["combined_overlap"] for r in n8_results]),
            "joint_mean": statistics.mean(joint_avgs),
            "cal_mean": statistics.mean(cal_avgs),
            "zero_mean": statistics.mean(zero_avgs),
            "spec_mean": statistics.mean(spec_avgs),
            "min_per_domain_jaccard": min_domain_jac,
            "max_per_domain_jaccard": max_domain_jac,
        },
        "per_n_per_seed": {
            str(n): [
                {
                    "seed": r["seed"],
                    "joint_avg": r["joint_avg"],
                    "cal_avg": r["cal_avg"],
                    "zero_avg": r["zero_avg"],
                    "spec_avg": r["spec_avg"],
                    "cal_gap": r["cal_gap"],
                    "zero_gap": r["zero_gap"],
                    "combined_jaccard": r["combined_jaccard"],
                    "combined_overlap": r["combined_overlap"],
                    "death_rate_composed": r["death_rate_composed"],
                }
                for r in all_results[n]
            ]
            for n in [2, 5, 8]
        },
    }

    results_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/flat_moe_n8_boundary/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
