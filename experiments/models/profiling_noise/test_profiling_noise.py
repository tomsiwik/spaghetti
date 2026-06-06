"""Profiling Noise Quantification experiment (Exp 12, VISION.md #12).

Quantifies the false-positive revival rate from sampling noise in Exp 18's
profiling protocol. The key insight: profile the SAME checkpoint with
DIFFERENT random batches. Any dead/alive disagreement between the two
profiling runs is definitionally noise (weights did not change).

Protocol:
  1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
  2. For each checkpoint S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
     a. Start from pretrained base (deepcopy)
     b. Freeze attention, fine-tune MLP only for S steps
     c. Profile per-capsule activation frequencies TWICE:
        - Run A: seed_profile=1000 (20 batches x 32)
        - Run B: seed_profile=2000 (20 batches x 32)
     d. Record binary dead/alive mask from each run
  3. At each checkpoint, measure:
     - Same-checkpoint disagreement: capsules classified differently by runs A and B
     - "Flickering capsules": capsules that are borderline (0 < f < 0.05 in at least one run)
  4. Recompute Exp 18's transition matrices using:
     - Run A masks (original-style, single profiling)
     - Consensus masks (dead only if dead in BOTH runs A and B)
  5. Compare: how many D->A transitions survive noise correction?

Kill criteria:
  1. Same-checkpoint disagreement >20% of capsules: profiling is unreliable
  2. Noise-attributable D->A transitions >50% of total D->A: revival is artifactual
  3. Noise-corrected revival rate <5%: true revival too weak to matter
"""

import copy
import statistics

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, STEPS_PRETRAIN, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations
from ..capsule_revival.test_capsule_revival import (
    get_dead_mask, jaccard_similarity, transition_counts,
    STEP_COUNTS, DOMAIN,
)


# Two different seeds for profiling the same checkpoint
PROFILE_SEED_A = 1000
PROFILE_SEED_B = 2000


def run_noise_experiment(seed=42, domain_name=DOMAIN):
    """Run profiling noise quantification for one training seed.

    For each checkpoint, profiles TWICE with different random batches.
    This isolates sampling noise from genuine state changes.

    Returns:
        dict with keys:
          masks_a, masks_b: per-checkpoint dead masks from runs A and B
          disagreements: per-checkpoint count of capsules that differ
          flickering: per-checkpoint count of borderline capsules
          transitions_a: Exp-18-style transitions using run A only
          transitions_consensus: transitions using consensus (dead = dead in both)
          per_layer_disagreements: disagreement broken down by layer
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, _ = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    val_ds = domain_datasets[domain_name][1]
    train_ds = domain_datasets[domain_name][0]

    # ============================================================
    # 1. Pretrain base model (300 steps on all data)
    # ============================================================
    print(f"  Pretraining base model (300 steps)...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # ============================================================
    # 2. Sweep checkpoints, profile TWICE each
    # ============================================================
    masks_a = {}
    masks_b = {}
    freqs_a = {}
    freqs_b = {}
    per_layer_a = {}
    per_layer_b = {}
    disagreements = {}
    flickering_counts = {}
    per_layer_disagreements = {}

    for S in STEP_COUNTS:
        print(f"  [S={S:>4d}] ", end="", flush=True)

        model = copy.deepcopy(base)

        if S > 0:
            _freeze_attention(model)
            train(model, train_ds, steps=S,
                  batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
            model.unfreeze()

        # Profile A (seed=1000)
        f_a = profile_activations(
            model, val_ds,
            n_batches=20, batch_size=32, seed=PROFILE_SEED_A,
        )
        flat_a, pl_a = get_dead_mask(f_a)

        # Profile B (seed=2000)
        f_b = profile_activations(
            model, val_ds,
            n_batches=20, batch_size=32, seed=PROFILE_SEED_B,
        )
        flat_b, pl_b = get_dead_mask(f_b)

        masks_a[S] = flat_a
        masks_b[S] = flat_b
        per_layer_a[S] = pl_a
        per_layer_b[S] = pl_b

        # Store raw frequencies for borderline analysis
        freqs_a[S] = [f.tolist() for f in f_a]
        freqs_b[S] = [f.tolist() for f in f_b]

        # Count disagreements (dead in one run, alive in the other)
        n_disagree = sum(1 for a, b in zip(flat_a, flat_b) if a != b)
        disagreements[S] = n_disagree

        # Count flickering capsules (0 < f < 0.05 in at least one run)
        n_flickering = 0
        for fa_layer, fb_layer in zip(f_a, f_b):
            mx.eval(fa_layer)
            mx.eval(fb_layer)
            for fa_val, fb_val in zip(fa_layer.tolist(), fb_layer.tolist()):
                if (0 < fa_val < 0.05) or (0 < fb_val < 0.05):
                    n_flickering += 1
        flickering_counts[S] = n_flickering

        # Per-layer disagreement
        layer_dis = []
        for l_idx in range(4):
            n_dis = sum(1 for a, b in zip(pl_a[l_idx], pl_b[l_idx]) if a != b)
            layer_dis.append(n_dis)
        per_layer_disagreements[S] = layer_dis

        n_total = len(flat_a)
        death_a = sum(flat_a) / n_total
        death_b = sum(flat_b) / n_total
        print(f"death_A={death_a:.1%}, death_B={death_b:.1%}, "
              f"disagree={n_disagree}/{n_total} ({n_disagree/n_total:.1%}), "
              f"flickering={n_flickering}")

    # ============================================================
    # 3. Compute transitions with single-run (A) and consensus masks
    # ============================================================

    # Consensus: dead only if dead in BOTH runs
    consensus_masks = {}
    for S in STEP_COUNTS:
        consensus = [a and b for a, b in zip(masks_a[S], masks_b[S])]
        consensus_masks[S] = consensus

    # Transitions using run A only (replicates Exp 18)
    transitions_a = []
    for i in range(1, len(STEP_COUNTS)):
        s_before = STEP_COUNTS[i - 1]
        s_after = STEP_COUNTS[i]
        trans = transition_counts(masks_a[s_before], masks_a[s_after])
        trans["s_before"] = s_before
        trans["s_after"] = s_after
        transitions_a.append(trans)

    # Transitions using consensus masks (noise-corrected)
    transitions_consensus = []
    for i in range(1, len(STEP_COUNTS)):
        s_before = STEP_COUNTS[i - 1]
        s_after = STEP_COUNTS[i]
        trans = transition_counts(consensus_masks[s_before], consensus_masks[s_after])
        trans["s_before"] = s_before
        trans["s_after"] = s_after
        transitions_consensus.append(trans)

    # ============================================================
    # 4. Cohort analysis with consensus masks
    # ============================================================
    dead_at_100_a = {i for i, d in enumerate(masks_a[100]) if d}
    dead_at_100_consensus = {i for i, d in enumerate(consensus_masks[100]) if d}

    cohort_a = {}
    cohort_consensus = {}
    for S in STEP_COUNTS:
        if S < 100:
            continue
        dead_s_a = {i for i, d in enumerate(masks_a[S]) if d}
        dead_s_c = {i for i, d in enumerate(consensus_masks[S]) if d}

        if dead_at_100_a:
            cohort_a[S] = {
                "pct_revived": len(dead_at_100_a - dead_s_a) / len(dead_at_100_a) * 100,
                "pct_still_dead": len(dead_at_100_a & dead_s_a) / len(dead_at_100_a) * 100,
            }
        if dead_at_100_consensus:
            cohort_consensus[S] = {
                "pct_revived": len(dead_at_100_consensus - dead_s_c) / len(dead_at_100_consensus) * 100,
                "pct_still_dead": len(dead_at_100_consensus & dead_s_c) / len(dead_at_100_consensus) * 100,
            }

    # ============================================================
    # 5. Jaccard with both methods
    # ============================================================
    jaccard_a = {}
    jaccard_consensus = {}
    for s1 in [100]:
        for s2 in [3200]:
            d1_a = {i for i, d in enumerate(masks_a[s1]) if d}
            d2_a = {i for i, d in enumerate(masks_a[s2]) if d}
            jaccard_a[(s1, s2)] = jaccard_similarity(d1_a, d2_a)

            d1_c = {i for i, d in enumerate(consensus_masks[s1]) if d}
            d2_c = {i for i, d in enumerate(consensus_masks[s2]) if d}
            jaccard_consensus[(s1, s2)] = jaccard_similarity(d1_c, d2_c)

    return {
        "masks_a": masks_a,
        "masks_b": masks_b,
        "consensus_masks": consensus_masks,
        "freqs_a": freqs_a,
        "freqs_b": freqs_b,
        "disagreements": disagreements,
        "flickering_counts": flickering_counts,
        "per_layer_disagreements": per_layer_disagreements,
        "transitions_a": transitions_a,
        "transitions_consensus": transitions_consensus,
        "cohort_a": cohort_a,
        "cohort_consensus": cohort_consensus,
        "cohort_a_size": len(dead_at_100_a),
        "cohort_consensus_size": len(dead_at_100_consensus),
        "jaccard_a": jaccard_a,
        "jaccard_consensus": jaccard_consensus,
    }


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        result = run_noise_experiment(seed=seed)
        all_results.append(result)

    n_total = len(all_results[0]["masks_a"][0])

    # ============================================================
    # 1. Same-checkpoint profiling disagreement
    # ============================================================
    print(f"\n{'='*70}")
    print("  Same-Checkpoint Profiling Disagreement (Run A vs Run B)")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>6} | {'Disagree':>10} {'%':>8} | "
          f"{'Flickering':>10} {'%':>8}")
    print("  " + "-" * 55)

    max_disagree_pct = 0
    for S in STEP_COUNTS:
        disagrees = [r["disagreements"][S] for r in all_results]
        flickers = [r["flickering_counts"][S] for r in all_results]
        mean_dis = statistics.mean(disagrees)
        mean_flk = statistics.mean(flickers)
        pct_dis = mean_dis / n_total * 100
        pct_flk = mean_flk / n_total * 100
        max_disagree_pct = max(max_disagree_pct, pct_dis)
        print(f"  {S:>6} | {mean_dis:>9.1f} {pct_dis:>7.1f}% | "
              f"{mean_flk:>9.1f} {pct_flk:>7.1f}%")

    # ============================================================
    # 2. Per-layer disagreement
    # ============================================================
    print(f"\n{'='*70}")
    print("  Per-Layer Disagreement (mean across checkpoints and seeds)")
    print(f"{'='*70}")

    for l_idx in range(4):
        all_dis = []
        for r in all_results:
            for S in STEP_COUNTS:
                all_dis.append(r["per_layer_disagreements"][S][l_idx])
        mean_dis = statistics.mean(all_dis)
        print(f"  Layer {l_idx}: {mean_dis:.1f} capsules disagree (of 128)")

    # ============================================================
    # 3. Transition comparison: single-run vs consensus
    # ============================================================
    print(f"\n{'='*70}")
    print("  Transition Comparison: Single-Run (A) vs Consensus")
    print(f"{'='*70}")

    print(f"\n  {'Interval':>14} | {'D->A (A)':>10} {'D->A (C)':>10} | "
          f"{'Reduction':>10} | {'RevA%':>8} {'RevC%':>8}")
    print("  " + "-" * 75)

    total_da_a_all = 0
    total_da_c_all = 0
    total_da_noise = 0

    for t_idx in range(len(STEP_COUNTS) - 1):
        s_before = STEP_COUNTS[t_idx]
        s_after = STEP_COUNTS[t_idx + 1]

        da_a_vals = [r["transitions_a"][t_idx]["da"] for r in all_results]
        da_c_vals = [r["transitions_consensus"][t_idx]["da"] for r in all_results]
        dd_a_vals = [r["transitions_a"][t_idx]["dd"] for r in all_results]
        dd_c_vals = [r["transitions_consensus"][t_idx]["dd"] for r in all_results]

        mean_da_a = statistics.mean(da_a_vals)
        mean_da_c = statistics.mean(da_c_vals)
        mean_dd_a = statistics.mean(dd_a_vals)
        mean_dd_c = statistics.mean(dd_c_vals)

        total_da_a_all += mean_da_a
        total_da_c_all += mean_da_c

        dead_before_a = mean_da_a + mean_dd_a
        dead_before_c = mean_da_c + mean_dd_c

        rev_a = mean_da_a / dead_before_a * 100 if dead_before_a > 0 else 0
        rev_c = mean_da_c / dead_before_c * 100 if dead_before_c > 0 else 0

        reduction = (mean_da_a - mean_da_c) / mean_da_a * 100 if mean_da_a > 0 else 0

        print(f"  {s_before:>5}->{s_after:<5} | {mean_da_a:>9.1f} {mean_da_c:>9.1f} | "
              f"{reduction:>8.1f}% | {rev_a:>7.1f}% {rev_c:>7.1f}%")

    total_da_noise = total_da_a_all - total_da_c_all
    noise_fraction = total_da_noise / total_da_a_all * 100 if total_da_a_all > 0 else 0

    print(f"\n  Total D->A (single-run):  {total_da_a_all:.1f}")
    print(f"  Total D->A (consensus):   {total_da_c_all:.1f}")
    print(f"  Noise-attributable D->A:  {total_da_noise:.1f} ({noise_fraction:.1f}%)")

    # ============================================================
    # 4. Cohort analysis comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  Cohort Analysis: Single-Run vs Consensus (capsules dead at S=100)")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>6} | {'Revived (A)':>12} {'Revived (C)':>12}")
    print("  " + "-" * 40)

    for S in STEP_COUNTS:
        if S < 100:
            continue
        rev_a_vals = [r["cohort_a"][S]["pct_revived"] for r in all_results]
        rev_c_vals = [r["cohort_consensus"][S]["pct_revived"] for r in all_results]
        mean_rev_a = statistics.mean(rev_a_vals)
        mean_rev_c = statistics.mean(rev_c_vals)
        print(f"  {S:>6} | {mean_rev_a:>10.1f}% {mean_rev_c:>10.1f}%")

    # Final cohort revival at S=3200
    cohort_rev_a_3200 = statistics.mean([r["cohort_a"][3200]["pct_revived"] for r in all_results])
    cohort_rev_c_3200 = statistics.mean([r["cohort_consensus"][3200]["pct_revived"] for r in all_results])

    # ============================================================
    # 5. Jaccard comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  Jaccard Comparison: Single-Run vs Consensus")
    print(f"{'='*70}")

    j_a_vals = [r["jaccard_a"][(100, 3200)] for r in all_results]
    j_c_vals = [r["jaccard_consensus"][(100, 3200)] for r in all_results]
    mean_j_a = statistics.mean(j_a_vals)
    mean_j_c = statistics.mean(j_c_vals)
    std_j_a = statistics.stdev(j_a_vals) if len(j_a_vals) > 1 else 0
    std_j_c = statistics.stdev(j_c_vals) if len(j_c_vals) > 1 else 0

    print(f"\n  Jaccard(dead_100, dead_3200):")
    print(f"    Single-run (A):  {mean_j_a:.3f} +/- {std_j_a:.3f}")
    print(f"    Consensus:       {mean_j_c:.3f} +/- {std_j_c:.3f}")

    # ============================================================
    # 6. Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Kill 1: Same-checkpoint disagreement > 20%
    kill1 = max_disagree_pct > 20.0
    print(f"\n  Kill 1: Max same-checkpoint disagreement = {max_disagree_pct:.1f}%")
    if kill1:
        print(f"    KILL: > 20%. Profiling is unreliable.")
    else:
        print(f"    PASS: <= 20%. Profiling is stable across batches.")

    # Kill 2: Noise-attributable D->A > 50% of total
    kill2 = noise_fraction > 50.0
    print(f"\n  Kill 2: Noise-attributable D->A = {noise_fraction:.1f}%")
    if kill2:
        print(f"    KILL: > 50%. Revival finding is artifactual.")
    else:
        print(f"    PASS: <= 50%. Most D->A transitions are genuine.")

    # Kill 3: Noise-corrected revival rate < 5%
    # Use the max per-interval consensus revival rate
    max_consensus_revival = 0
    for t_idx in range(len(STEP_COUNTS) - 1):
        da_c = statistics.mean([r["transitions_consensus"][t_idx]["da"] for r in all_results])
        dd_c = statistics.mean([r["transitions_consensus"][t_idx]["dd"] for r in all_results])
        dead_before_c = da_c + dd_c
        if dead_before_c > 0:
            rev_c = da_c / dead_before_c * 100
            max_consensus_revival = max(max_consensus_revival, rev_c)

    kill3 = max_consensus_revival < 5.0
    print(f"\n  Kill 3: Max noise-corrected revival rate = {max_consensus_revival:.1f}%")
    if kill3:
        print(f"    KILL: < 5%. True revival too weak to matter.")
    else:
        print(f"    PASS: >= 5%. Genuine revival mechanism confirmed.")

    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    # ============================================================
    # 7. Impact on Exp 18 findings
    # ============================================================
    print(f"\n{'='*70}")
    print("  Impact Assessment on Exp 18 Findings")
    print(f"{'='*70}")

    print(f"\n  Exp 18 reported:")
    print(f"    - 28.1% cohort revival (S=100 dead cohort revived by S=3200)")
    print(f"    - Jaccard(dead_100, dead_3200) = 0.669")
    print(f"    - Max revival rate per interval = 15.9%")

    print(f"\n  This experiment found:")
    print(f"    - Cohort revival (single-run):  {cohort_rev_a_3200:.1f}%")
    print(f"    - Cohort revival (consensus):   {cohort_rev_c_3200:.1f}%")
    print(f"    - Noise fraction of D->A:       {noise_fraction:.1f}%")
    print(f"    - Jaccard single-run:           {mean_j_a:.3f}")
    print(f"    - Jaccard consensus:            {mean_j_c:.3f}")

    if noise_fraction < 20:
        print(f"\n  CONCLUSION: Profiling noise accounts for <20% of D->A transitions.")
        print(f"  Exp 18's revival finding is ROBUST. The consensus-corrected metrics")
        print(f"  confirm genuine capsule revival at a rate close to the original report.")
    elif noise_fraction < 50:
        print(f"\n  CONCLUSION: Profiling noise accounts for {noise_fraction:.0f}% of D->A.")
        print(f"  Exp 18's revival finding is DIRECTIONALLY CORRECT but OVERSTATED.")
        print(f"  The true revival rate is ~{100-noise_fraction:.0f}% of what was reported.")
        print(f"  The 'prune after training' recommendation remains valid.")
    else:
        print(f"\n  CONCLUSION: Profiling noise accounts for >{noise_fraction:.0f}% of D->A.")
        print(f"  Exp 18's revival finding is UNRELIABLE. Borderline capsule flickering")
        print(f"  dominates over genuine revival. The profiling protocol needs more")
        print(f"  samples or a consensus requirement before revival claims can be made.")


if __name__ == "__main__":
    main()
