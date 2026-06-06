"""Per-Capsule Revival Tracking experiment (Exp 18).

Tracks per-capsule death/alive identity across training checkpoints to
determine whether the aggregate death decrease from Exp 17 reflects:
  (a) True revival of the SAME dead capsules (inter-layer coupling)
  (b) Population turnover (different capsules cycling through dead/alive)

Protocol:
  1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
  2. For each checkpoint S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
     a. Fine-tune for exactly S steps (same seed = same trajectory)
     b. Profile per-capsule activation frequencies
     c. Record binary dead/alive mask per capsule
  3. Compute transition matrices between consecutive checkpoints:
     alive->alive, alive->dead, dead->alive, dead->dead
  4. Track cohort survival: capsules dead at S=100, what fraction
     remain dead at S=200, 400, ..., 3200?
  5. Compute Jaccard similarity of dead sets across checkpoints

Kill criteria:
  1. Jaccard(dead_100, dead_3200) > 0.85: death is sticky, revival negligible
  2. Revival rate (dead->alive per interval) < 5% of dead pop: mechanism too weak
  3. Total turnover events < 10 per seed: dynamics too sparse
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


# Step counts (same as Exp 17 for comparability)
STEP_COUNTS = [0, 50, 100, 200, 400, 800, 1600, 3200]

# Domain for fine-tuning
DOMAIN = "a_m"


def get_dead_mask(freqs, threshold=0.0):
    """Convert per-layer frequency arrays to a flat binary dead mask.

    Args:
        freqs: list of (P_l,) arrays from profile_activations
        threshold: frequency threshold below which capsule is "dead"

    Returns:
        flat_mask: list of bool (True = dead) across all layers
        per_layer_masks: list of list of bool per layer
    """
    per_layer_masks = []
    flat_mask = []
    for freq in freqs:
        mx.eval(freq)
        layer_mask = [f <= threshold for f in freq.tolist()]
        per_layer_masks.append(layer_mask)
        flat_mask.extend(layer_mask)
    return flat_mask, per_layer_masks


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets: |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 1.0


def transition_counts(mask_before, mask_after):
    """Count transitions between dead/alive states.

    Args:
        mask_before, mask_after: list of bool (True = dead)

    Returns:
        dict with counts: dd (dead->dead), da (dead->alive),
        ad (alive->dead), aa (alive->alive)
    """
    assert len(mask_before) == len(mask_after)
    dd = sum(1 for a, b in zip(mask_before, mask_after) if a and b)
    da = sum(1 for a, b in zip(mask_before, mask_after) if a and not b)
    ad = sum(1 for a, b in zip(mask_before, mask_after) if not a and b)
    aa = sum(1 for a, b in zip(mask_before, mask_after) if not a and not b)
    return {"dd": dd, "da": da, "ad": ad, "aa": aa}


def run_revival_experiment(seed=42, domain_name=DOMAIN):
    """Run per-capsule revival tracking for one seed.

    Returns:
        checkpoint_masks: dict mapping step_count -> flat dead mask
        checkpoint_per_layer: dict mapping step_count -> per_layer dead masks
        checkpoint_freqs: dict mapping step_count -> per_layer freq arrays
        transition_data: list of transition dicts between consecutive checkpoints
        cohort_data: dict tracking the S=100 dead cohort across time
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
    train_ds = domain_datasets[domain_name][0]
    val_ds = domain_datasets[domain_name][1]

    # ============================================================
    # 1. Pretrain base model (300 steps on all data)
    # ============================================================
    print(f"  Pretraining base model (300 steps)...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # ============================================================
    # 2. Sweep fine-tuning steps, tracking per-capsule identity
    # ============================================================
    checkpoint_masks = {}
    checkpoint_per_layer = {}
    checkpoint_death_rates = {}

    for S in STEP_COUNTS:
        print(f"  [S={S:>4d}] ", end="", flush=True)

        # Start from pretrained base (same as Exp 17)
        model = copy.deepcopy(base)

        if S > 0:
            _freeze_attention(model)
            train(model, train_ds, steps=S,
                  batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
            model.unfreeze()

        # Profile activations
        freqs = profile_activations(
            model, val_ds,
            n_batches=20, batch_size=32, seed=seed,
        )

        flat_mask, per_layer_masks = get_dead_mask(freqs)
        checkpoint_masks[S] = flat_mask
        checkpoint_per_layer[S] = per_layer_masks

        death_rate = sum(flat_mask) / len(flat_mask)
        checkpoint_death_rates[S] = death_rate

        print(f"death={death_rate:.1%}, "
              f"n_dead={sum(flat_mask)}/{len(flat_mask)}")

    # ============================================================
    # 3. Compute transition matrices
    # ============================================================
    transition_data = []
    for i in range(1, len(STEP_COUNTS)):
        s_before = STEP_COUNTS[i - 1]
        s_after = STEP_COUNTS[i]
        trans = transition_counts(
            checkpoint_masks[s_before],
            checkpoint_masks[s_after],
        )
        trans["s_before"] = s_before
        trans["s_after"] = s_after
        transition_data.append(trans)

    # ============================================================
    # 4. Cohort tracking: capsules dead at S=100
    # ============================================================
    dead_at_100 = {i for i, d in enumerate(checkpoint_masks[100]) if d}
    cohort_data = {"anchor_step": 100, "anchor_size": len(dead_at_100)}
    cohort_data["survival"] = {}

    for S in STEP_COUNTS:
        if S < 100:
            continue
        dead_at_S = {i for i, d in enumerate(checkpoint_masks[S]) if d}
        still_dead = dead_at_100 & dead_at_S
        revived = dead_at_100 - dead_at_S

        cohort_data["survival"][S] = {
            "still_dead": len(still_dead),
            "revived": len(revived),
            "pct_still_dead": len(still_dead) / len(dead_at_100) * 100 if dead_at_100 else 0,
            "pct_revived": len(revived) / len(dead_at_100) * 100 if dead_at_100 else 0,
        }

    # ============================================================
    # 5. Jaccard similarity matrix
    # ============================================================
    jaccard_data = {}
    for s1 in STEP_COUNTS:
        dead_s1 = {i for i, d in enumerate(checkpoint_masks[s1]) if d}
        for s2 in STEP_COUNTS:
            if s2 <= s1:
                continue
            dead_s2 = {i for i, d in enumerate(checkpoint_masks[s2]) if d}
            jaccard_data[(s1, s2)] = jaccard_similarity(dead_s1, dead_s2)

    # ============================================================
    # 6. Per-layer transition analysis
    # ============================================================
    per_layer_transitions = []
    for l_idx in range(4):
        layer_trans = []
        for i in range(1, len(STEP_COUNTS)):
            s_before = STEP_COUNTS[i - 1]
            s_after = STEP_COUNTS[i]
            mask_before = checkpoint_per_layer[s_before][l_idx]
            mask_after = checkpoint_per_layer[s_after][l_idx]
            trans = transition_counts(mask_before, mask_after)
            trans["s_before"] = s_before
            trans["s_after"] = s_after
            layer_trans.append(trans)
        per_layer_transitions.append(layer_trans)

    return {
        "checkpoint_masks": checkpoint_masks,
        "checkpoint_per_layer": checkpoint_per_layer,
        "checkpoint_death_rates": checkpoint_death_rates,
        "transition_data": transition_data,
        "cohort_data": cohort_data,
        "jaccard_data": jaccard_data,
        "per_layer_transitions": per_layer_transitions,
    }


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        result = run_revival_experiment(seed=seed)
        all_results.append(result)

    # ============================================================
    # Aggregate: Death rates (should match Exp 17)
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate: Death Rate vs Training Steps")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>6} | {'Death Rate':>10} {'Std':>8}")
    print("  " + "-" * 35)

    for S in STEP_COUNTS:
        rates = [r["checkpoint_death_rates"][S] for r in all_results]
        mean_dr = statistics.mean(rates)
        std_dr = statistics.stdev(rates) if len(rates) > 1 else 0
        print(f"  {S:>6} | {mean_dr:>9.1%} {std_dr:>7.1%}")

    # ============================================================
    # Aggregate: Transition matrices
    # ============================================================
    print(f"\n{'='*70}")
    print("  Transition Analysis (consecutive checkpoints)")
    print(f"{'='*70}")

    print(f"\n  {'Interval':>14} | {'D->D':>6} {'D->A':>6} {'A->D':>6} {'A->A':>6} | "
          f"{'Revival%':>8} {'NewDeath%':>10}")
    print("  " + "-" * 75)

    total_turnover_events = []
    total_revivals = []

    for t_idx in range(len(STEP_COUNTS) - 1):
        s_before = STEP_COUNTS[t_idx]
        s_after = STEP_COUNTS[t_idx + 1]

        dds = [r["transition_data"][t_idx]["dd"] for r in all_results]
        das = [r["transition_data"][t_idx]["da"] for r in all_results]
        ads = [r["transition_data"][t_idx]["ad"] for r in all_results]
        aas = [r["transition_data"][t_idx]["aa"] for r in all_results]

        mean_dd = statistics.mean(dds)
        mean_da = statistics.mean(das)
        mean_ad = statistics.mean(ads)
        mean_aa = statistics.mean(aas)

        total_dead_before = mean_dd + mean_da
        total_alive_before = mean_ad + mean_aa
        revival_pct = mean_da / total_dead_before * 100 if total_dead_before > 0 else 0
        new_death_pct = mean_ad / total_alive_before * 100 if total_alive_before > 0 else 0

        total_turnover_events.extend([sum(das) + sum(ads)])
        total_revivals.extend(das)

        print(f"  {s_before:>5}->{s_after:<5} | {mean_dd:>5.0f} {mean_da:>5.0f} "
              f"{mean_ad:>5.0f} {mean_aa:>5.0f} | "
              f"{revival_pct:>7.1f}% {new_death_pct:>9.1f}%")

    # ============================================================
    # Aggregate: Cohort analysis (S=100 dead cohort)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Cohort Tracking: Capsules Dead at S=100")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>6} | {'Still Dead':>10} {'Std':>8} | {'Revived':>8} {'Std':>8}")
    print("  " + "-" * 55)

    for S in STEP_COUNTS:
        if S < 100:
            continue
        still_deads = [r["cohort_data"]["survival"][S]["pct_still_dead"] for r in all_results]
        reviveds = [r["cohort_data"]["survival"][S]["pct_revived"] for r in all_results]

        mean_sd = statistics.mean(still_deads)
        std_sd = statistics.stdev(still_deads) if len(still_deads) > 1 else 0
        mean_rv = statistics.mean(reviveds)
        std_rv = statistics.stdev(reviveds) if len(reviveds) > 1 else 0

        print(f"  {S:>6} | {mean_sd:>9.1f}% {std_sd:>7.1f}% | "
              f"{mean_rv:>7.1f}% {std_rv:>7.1f}%")

    # ============================================================
    # Aggregate: Jaccard similarity
    # ============================================================
    print(f"\n{'='*70}")
    print("  Jaccard Similarity of Dead Sets")
    print(f"{'='*70}")

    # Key comparisons
    key_pairs = [(100, 200), (100, 400), (100, 800), (100, 1600), (100, 3200),
                 (50, 3200), (200, 3200)]

    print(f"\n  {'Pair':>14} | {'Jaccard':>8} {'Std':>8}")
    print("  " + "-" * 35)

    for s1, s2 in key_pairs:
        jaccards = [r["jaccard_data"][(s1, s2)] for r in all_results]
        mean_j = statistics.mean(jaccards)
        std_j = statistics.stdev(jaccards) if len(jaccards) > 1 else 0
        print(f"  {s1:>5}->{s2:<5} | {mean_j:>7.3f} {std_j:>7.3f}")

    # ============================================================
    # Aggregate: Per-layer transitions (focus on revival)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Per-Layer Revival Rates (D->A transitions, S=100->3200)")
    print(f"{'='*70}")

    for l_idx in range(4):
        # Sum all D->A transitions from S=100 onward
        total_da_per_seed = []
        total_dead_at_100_per_seed = []
        for r in all_results:
            da_sum = 0
            for t in r["per_layer_transitions"][l_idx]:
                if t["s_before"] >= 100:
                    da_sum += t["da"]
            # Dead at S=100 for this layer
            dead_100 = sum(1 for d in r["checkpoint_per_layer"][100][l_idx] if d)
            total_da_per_seed.append(da_sum)
            total_dead_at_100_per_seed.append(dead_100)

        mean_da = statistics.mean(total_da_per_seed)
        mean_dead_100 = statistics.mean(total_dead_at_100_per_seed)

        # Note: cumulative D->A counts total transitions, not unique capsules
        print(f"  Layer {l_idx}: {mean_da:.0f} D->A transitions "
              f"(of {mean_dead_100:.0f} dead at S=100)")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Kill 1: Jaccard(dead_100, dead_3200) > 0.85
    jaccards_100_3200 = [r["jaccard_data"][(100, 3200)] for r in all_results]
    mean_j = statistics.mean(jaccards_100_3200)
    kill1 = mean_j > 0.85
    print(f"\n  Kill 1: Jaccard(dead_100, dead_3200) = {mean_j:.3f}")
    if kill1:
        print(f"    KILL: > 0.85. Death is STICKY. Revival negligible.")
    else:
        print(f"    PASS: <= 0.85. Significant capsule identity change.")

    # Kill 2: Revival rate < 5% of dead population per interval
    max_revival_pct = 0
    for t_idx in range(len(STEP_COUNTS) - 1):
        das = [r["transition_data"][t_idx]["da"] for r in all_results]
        dds = [r["transition_data"][t_idx]["dd"] for r in all_results]
        mean_da = statistics.mean(das)
        mean_dead_before = statistics.mean(das) + statistics.mean(dds)
        if mean_dead_before > 0:
            revival_pct = mean_da / mean_dead_before * 100
            max_revival_pct = max(max_revival_pct, revival_pct)

    kill2 = max_revival_pct < 5.0
    print(f"\n  Kill 2: Max revival rate per interval = {max_revival_pct:.1f}%")
    if kill2:
        print(f"    KILL: < 5%. Inter-layer coupling revival too weak.")
    else:
        print(f"    PASS: >= 5%. Revival mechanism is meaningful.")

    # Kill 3: Total turnover events < 10 per seed
    min_turnover = float("inf")
    for r in all_results:
        turnover = sum(t["da"] + t["ad"] for t in r["transition_data"])
        min_turnover = min(min_turnover, turnover)

    kill3 = min_turnover < 10
    print(f"\n  Kill 3: Min turnover events per seed = {min_turnover:.0f}")
    if kill3:
        print(f"    KILL: < 10. Dynamics too sparse to study.")
    else:
        print(f"    PASS: >= 10. Sufficient dynamics for analysis.")

    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    # ============================================================
    # Summary: Revival vs Turnover
    # ============================================================
    print(f"\n{'='*70}")
    print("  Summary: Revival vs Population Turnover")
    print(f"{'='*70}")

    # Compute: of the aggregate death decrease (S=100 -> S=3200),
    # how much is due to the S=100 dead cohort reviving vs new deaths
    # being fewer than in the spike phase?

    cohort_revival_pcts = [r["cohort_data"]["survival"][3200]["pct_revived"]
                           for r in all_results]
    mean_cohort_revival = statistics.mean(cohort_revival_pcts)

    death_100 = statistics.mean([r["checkpoint_death_rates"][100] for r in all_results])
    death_3200 = statistics.mean([r["checkpoint_death_rates"][3200] for r in all_results])
    aggregate_decrease_pp = (death_100 - death_3200) * 100

    # Total capsules
    n_total = len(all_results[0]["checkpoint_masks"][0])
    cohort_sizes = [r["cohort_data"]["anchor_size"] for r in all_results]
    mean_cohort_size = statistics.mean(cohort_sizes)
    revived_count = mean_cohort_revival / 100 * mean_cohort_size
    revival_contribution_pp = revived_count / n_total * 100

    print(f"\n  Aggregate death decrease (S=100 -> S=3200): {aggregate_decrease_pp:.1f} pp")
    print(f"  S=100 dead cohort size: {mean_cohort_size:.0f} / {n_total}")
    print(f"  Of S=100 dead cohort, revived by S=3200: {mean_cohort_revival:.1f}%")
    print(f"  Revival contribution to decrease: {revival_contribution_pp:.1f} pp")
    print(f"  Remaining decrease (new deaths avoided): {aggregate_decrease_pp - revival_contribution_pp:.1f} pp")

    if mean_cohort_revival > 20:
        print(f"\n  FINDING: True revival of SAME capsules is a major contributor.")
        print(f"  The inter-layer coupling revival mechanism is empirically confirmed")
        print(f"  at the per-capsule identity level.")
    elif mean_cohort_revival > 5:
        print(f"\n  FINDING: Mixed dynamics -- both revival and turnover contribute.")
        print(f"  Some capsules genuinely revive, but turnover is also significant.")
    else:
        print(f"\n  FINDING: Death is STICKY. The aggregate decrease is mainly from")
        print(f"  fewer NEW deaths in later training, not revival of old dead capsules.")
        print(f"  Population turnover dominates over true revival.")


if __name__ == "__main__":
    main()
