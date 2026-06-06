"""Freeze-Then-Prune Protocol -- compare post-training vs mid-training pruning.

Experimental design:
  Protocol A (FREEZE-THEN-PRUNE):
    1. Pretrain base (300 steps)
    2. Fine-tune all MLP layers for FULL training (S_total steps)
    3. Freeze all weights
    4. Profile dead capsules -> prune -> evaluate quality

  Protocol B (MID-TRAINING-PRUNE):
    1. Pretrain base (300 steps)
    2. Fine-tune for S_mid steps (S_mid < S_total)
    3. Profile dead capsules -> prune
    4. Continue training for remaining (S_total - S_mid) steps
    5. Evaluate quality

  Protocol C (MID-TRAINING-PRUNE-AND-FREEZE):
    1. Same as B but after pruning at S_mid, freeze and don't continue
    2. Tests whether continued training after pruning helps

  Control (NO-PRUNE):
    1. Pretrain + fine-tune for S_total steps, no pruning
    2. Evaluate quality (upper bound)

Measurements:
  - Death rate at each protocol's profiling point
  - Post-pruning model quality (val loss)
  - Quality gap vs no-prune control
  - Revival-after-prune rate (for Protocol B: how many pruned capsules
    would have revived if not pruned)

Kill criteria:
  1. Post-freeze death rate < mid-training death rate + 5pp
     (freeze-then-prune does NOT yield meaningfully more dead capsules)
  2. Post-freeze pruned quality > 3% worse than mid-training pruned quality
     (freeze-then-prune hurts quality despite higher yield)
"""

import copy
import statistics
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..relu_router.test_composition import (
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, STEPS_PRETRAIN, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations, identify_dead_capsules, prune_model,
)
from ..capsule_revival.test_capsule_revival import get_dead_mask, transition_counts


# Experiment config
DOMAIN = "a_m"
N_LAYERS = 4
S_TOTAL = 3200        # Full training duration
S_MID_POINTS = [100, 400, 800, 1600]  # Mid-training prune points


def profile_death_rate(model, val_ds, seed=42):
    """Profile and return death rate and per-layer info.

    Returns:
        death_rate: overall fraction of dead capsules
        per_layer_rates: list of per-layer death rates
        per_layer_masks: list of list of bool (True=dead) per layer
        flat_mask: list of bool across all layers
    """
    freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
    flat_mask, per_layer_masks = get_dead_mask(freqs)
    overall_rate = sum(flat_mask) / len(flat_mask) if flat_mask else 0
    per_layer_rates = []
    for layer_mask in per_layer_masks:
        rate = sum(layer_mask) / len(layer_mask) if layer_mask else 0
        per_layer_rates.append(rate)
    return overall_rate, per_layer_rates, per_layer_masks, flat_mask


def prune_dead_capsules(model, val_ds, seed=42, verbose=False):
    """Profile and prune dead capsules from model in-place.

    Returns:
        prune_stats: dict with pruning statistics
        death_rate: overall death rate before pruning
        per_layer_rates: per-layer death rates before pruning
    """
    freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
    alive_masks = identify_dead_capsules(freqs, threshold=0.0)
    flat_mask, per_layer_masks = get_dead_mask(freqs)
    death_rate = sum(flat_mask) / len(flat_mask) if flat_mask else 0
    per_layer_rates = [
        sum(m) / len(m) if m else 0 for m in per_layer_masks
    ]
    prune_stats = prune_model(model, alive_masks, verbose=verbose)
    return prune_stats, death_rate, per_layer_rates


def run_protocol_A(base_model, train_ds, val_ds, seed=42):
    """Protocol A: Freeze-then-prune.

    Train fully -> freeze -> profile -> prune -> evaluate.
    """
    model = copy.deepcopy(base_model)
    _freeze_attention(model)
    train(model, train_ds, steps=S_TOTAL, batch_size=BATCH_SIZE, lr=LR,
          seed=seed, log_every=9999)

    # Profile val loss BEFORE pruning (fully trained, no prune)
    model.unfreeze()
    val_loss_before = evaluate(model, val_ds, BATCH_SIZE)

    # Profile death and prune
    death_rate, per_layer_rates, per_layer_masks, flat_mask = profile_death_rate(
        model, val_ds, seed=seed
    )
    n_dead_before = sum(flat_mask)

    prune_stats, _, _ = prune_dead_capsules(model, val_ds, seed=seed)
    val_loss_after = evaluate(model, val_ds, BATCH_SIZE)

    return {
        "protocol": "A_freeze_then_prune",
        "val_loss_before_prune": val_loss_before,
        "val_loss_after_prune": val_loss_after,
        "death_rate": death_rate,
        "per_layer_rates": per_layer_rates,
        "n_dead": n_dead_before,
        "n_total": len(flat_mask),
        "prune_stats": prune_stats,
        "quality_change_pct": (val_loss_after - val_loss_before) / val_loss_before * 100,
    }


def run_protocol_B(base_model, train_ds, val_ds, s_mid, seed=42):
    """Protocol B: Mid-training prune then continue.

    Train to s_mid -> profile -> prune -> train remaining -> evaluate.
    """
    model = copy.deepcopy(base_model)
    _freeze_attention(model)

    # Phase 1: Train to mid-point
    train(model, train_ds, steps=s_mid, batch_size=BATCH_SIZE, lr=LR,
          seed=seed, log_every=9999)

    # Profile death at mid-point (before pruning)
    model.unfreeze()
    death_rate_mid, per_layer_rates_mid, per_layer_masks_mid, flat_mask_mid = \
        profile_death_rate(model, val_ds, seed=seed)
    n_dead_mid = sum(flat_mask_mid)

    # Prune at mid-point
    prune_stats, _, _ = prune_dead_capsules(model, val_ds, seed=seed)
    val_loss_mid_pruned = evaluate(model, val_ds, BATCH_SIZE)

    # Phase 2: Continue training after pruning
    _freeze_attention(model)
    remaining_steps = S_TOTAL - s_mid
    if remaining_steps > 0:
        train(model, train_ds, steps=remaining_steps, batch_size=BATCH_SIZE,
              lr=LR, seed=seed + 1000, log_every=9999)

    model.unfreeze()
    val_loss_final = evaluate(model, val_ds, BATCH_SIZE)

    # Profile death at end (after continued training post-prune)
    death_rate_final, per_layer_rates_final, _, flat_mask_final = \
        profile_death_rate(model, val_ds, seed=seed)

    return {
        "protocol": f"B_mid_prune_S{s_mid}",
        "s_mid": s_mid,
        "death_rate_at_prune": death_rate_mid,
        "per_layer_rates_at_prune": per_layer_rates_mid,
        "n_dead_at_prune": n_dead_mid,
        "n_total": len(flat_mask_mid),
        "val_loss_at_prune": val_loss_mid_pruned,
        "death_rate_final": death_rate_final,
        "per_layer_rates_final": per_layer_rates_final,
        "val_loss_final": val_loss_final,
        "prune_stats": prune_stats,
    }


def run_no_prune_control(base_model, train_ds, val_ds, seed=42):
    """Control: Full training, no pruning."""
    model = copy.deepcopy(base_model)
    _freeze_attention(model)
    train(model, train_ds, steps=S_TOTAL, batch_size=BATCH_SIZE, lr=LR,
          seed=seed, log_every=9999)
    model.unfreeze()
    val_loss = evaluate(model, val_ds, BATCH_SIZE)

    death_rate, per_layer_rates, _, flat_mask = profile_death_rate(
        model, val_ds, seed=seed
    )

    return {
        "protocol": "control_no_prune",
        "val_loss": val_loss,
        "death_rate": death_rate,
        "per_layer_rates": per_layer_rates,
        "n_dead": sum(flat_mask),
        "n_total": len(flat_mask),
    }


def run_full_experiment(seed=42):
    """Run all protocols for one seed."""
    # Setup data
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
    train_ds = domain_datasets[DOMAIN][0]
    val_ds = domain_datasets[DOMAIN][1]

    # Pretrain base model
    print(f"  Pretraining base model (300 steps, seed={seed})...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    results = {}

    # Control: no pruning
    print(f"\n  === CONTROL (no pruning, {S_TOTAL} steps) ===")
    results["control"] = run_no_prune_control(base, train_ds, val_ds, seed=seed)
    print(f"    val_loss={results['control']['val_loss']:.4f}, "
          f"death_rate={results['control']['death_rate']:.1%}")

    # Protocol A: Freeze-then-prune (train fully, then prune)
    print(f"\n  === PROTOCOL A: Freeze-then-prune ({S_TOTAL} steps, then prune) ===")
    results["proto_A"] = run_protocol_A(base, train_ds, val_ds, seed=seed)
    print(f"    death_rate={results['proto_A']['death_rate']:.1%}, "
          f"val_before={results['proto_A']['val_loss_before_prune']:.4f}, "
          f"val_after={results['proto_A']['val_loss_after_prune']:.4f}, "
          f"change={results['proto_A']['quality_change_pct']:+.2f}%")

    # Protocol B: Mid-training prune at various checkpoints
    for s_mid in S_MID_POINTS:
        key = f"proto_B_S{s_mid}"
        print(f"\n  === PROTOCOL B: Mid-prune at S={s_mid}, continue to {S_TOTAL} ===")
        results[key] = run_protocol_B(base, train_ds, val_ds, s_mid, seed=seed)
        r = results[key]
        print(f"    death_at_prune={r['death_rate_at_prune']:.1%}, "
              f"val_at_prune={r['val_loss_at_prune']:.4f}, "
              f"val_final={r['val_loss_final']:.4f}, "
              f"death_final={r['death_rate_final']:.1%}")

    return results


def analyze_results(all_seeds_results):
    """Analyze and print aggregate results across seeds."""
    seeds = list(all_seeds_results.keys())
    n_seeds = len(seeds)

    print(f"\n{'='*80}")
    print(f"  FREEZE-THEN-PRUNE PROTOCOL ANALYSIS ({n_seeds} seeds)")
    print(f"{'='*80}")

    # Collect per-protocol aggregates
    # Control
    ctrl_losses = [all_seeds_results[s]["control"]["val_loss"] for s in seeds]
    ctrl_deaths = [all_seeds_results[s]["control"]["death_rate"] for s in seeds]
    ctrl_mean_loss = statistics.mean(ctrl_losses)

    print(f"\n  Control (no pruning, S={S_TOTAL}):")
    print(f"    val_loss: {statistics.mean(ctrl_losses):.4f} +/- {statistics.stdev(ctrl_losses):.4f}" if n_seeds > 1 else f"    val_loss: {ctrl_losses[0]:.4f}")
    print(f"    death_rate: {statistics.mean(ctrl_deaths):.1%}")

    # Protocol A: Freeze-then-prune
    a_deaths = [all_seeds_results[s]["proto_A"]["death_rate"] for s in seeds]
    a_losses_before = [all_seeds_results[s]["proto_A"]["val_loss_before_prune"] for s in seeds]
    a_losses_after = [all_seeds_results[s]["proto_A"]["val_loss_after_prune"] for s in seeds]
    a_mean_loss_after = statistics.mean(a_losses_after)
    a_vs_ctrl = (a_mean_loss_after - ctrl_mean_loss) / ctrl_mean_loss * 100

    print(f"\n  Protocol A: Freeze-then-prune (train {S_TOTAL} steps, then prune):")
    print(f"    death_rate: {statistics.mean(a_deaths):.1%}" + (f" +/- {statistics.stdev(a_deaths):.1%}" if n_seeds > 1 else ""))
    print(f"    val_loss (before prune): {statistics.mean(a_losses_before):.4f}")
    print(f"    val_loss (after prune):  {statistics.mean(a_losses_after):.4f}")
    print(f"    vs control: {a_vs_ctrl:+.2f}%")

    # Protocol B: Mid-training prune at each checkpoint
    print(f"\n  Protocol B: Mid-training prune (prune at S_mid, continue to {S_TOTAL}):")
    print(f"    {'S_mid':>6} | {'death_at_prune':>15} | {'val_at_prune':>13} | {'val_final':>10} | {'death_final':>12} | {'vs ctrl':>8} | {'vs proto_A':>11}")
    print(f"    {'-'*6}-+-{'-'*15}-+-{'-'*13}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*11}")

    b_results = {}
    for s_mid in S_MID_POINTS:
        key = f"proto_B_S{s_mid}"
        b_deaths_at = [all_seeds_results[s][key]["death_rate_at_prune"] for s in seeds]
        b_vals_at = [all_seeds_results[s][key]["val_loss_at_prune"] for s in seeds]
        b_vals_final = [all_seeds_results[s][key]["val_loss_final"] for s in seeds]
        b_deaths_final = [all_seeds_results[s][key]["death_rate_final"] for s in seeds]

        mean_death_at = statistics.mean(b_deaths_at)
        mean_val_final = statistics.mean(b_vals_final)
        mean_death_final = statistics.mean(b_deaths_final)
        vs_ctrl = (mean_val_final - ctrl_mean_loss) / ctrl_mean_loss * 100
        vs_proto_a = (mean_val_final - a_mean_loss_after) / a_mean_loss_after * 100

        b_results[s_mid] = {
            "death_at_prune": mean_death_at,
            "val_final": mean_val_final,
            "death_final": mean_death_final,
            "vs_ctrl": vs_ctrl,
            "vs_proto_a": vs_proto_a,
        }

        print(f"    {s_mid:>6} | {mean_death_at:>14.1%} | {statistics.mean(b_vals_at):>12.4f} | {mean_val_final:>9.4f} | {mean_death_final:>11.1%} | {vs_ctrl:>+7.2f}% | {vs_proto_a:>+10.2f}%")

    # Key comparison: death yield difference
    print(f"\n{'='*80}")
    print(f"  KEY COMPARISON: Pruning Yield (death rate at profile time)")
    print(f"{'='*80}")

    a_mean_death = statistics.mean(a_deaths)
    print(f"\n  Protocol A (freeze-then-prune) death rate: {a_mean_death:.1%}")
    for s_mid in S_MID_POINTS:
        key = f"proto_B_S{s_mid}"
        b_death = statistics.mean([all_seeds_results[s][key]["death_rate_at_prune"] for s in seeds])
        diff_pp = (a_mean_death - b_death) * 100
        print(f"  Protocol B S={s_mid:>4d} death rate:          {b_death:.1%}  (A - B = {diff_pp:+.1f}pp)")

    # Kill criterion 1: post-freeze yields <5pp more dead capsules than mid-training
    print(f"\n{'='*80}")
    print(f"  KILL CRITERION 1: Does freeze-then-prune yield >=5pp more dead capsules?")
    print(f"{'='*80}")

    max_b_death = max(
        statistics.mean([all_seeds_results[s][f"proto_B_S{sm}"]["death_rate_at_prune"] for s in seeds])
        for sm in S_MID_POINTS
    )
    yield_diff_pp = (a_mean_death - max_b_death) * 100
    kill1 = yield_diff_pp < 5.0
    print(f"\n  Proto A death: {a_mean_death:.1%}")
    print(f"  Best Proto B death (max across S_mid): {max_b_death:.1%}")
    print(f"  Difference: {yield_diff_pp:+.1f}pp")
    print(f"  Threshold: >=5pp required")
    if kill1:
        print(f"  KILL: Freeze-then-prune does NOT yield meaningfully more dead capsules")
    else:
        print(f"  PASS: Freeze-then-prune yields {yield_diff_pp:.1f}pp more dead capsules")

    # Kill criterion 2: post-freeze pruned quality >3% worse than mid-training pruned
    print(f"\n{'='*80}")
    print(f"  KILL CRITERION 2: Does freeze-then-prune quality degrade >3% vs mid-prune?")
    print(f"{'='*80}")

    best_b_final_loss = min(
        statistics.mean([all_seeds_results[s][f"proto_B_S{sm}"]["val_loss_final"] for s in seeds])
        for sm in S_MID_POINTS
    )
    quality_diff = (a_mean_loss_after - best_b_final_loss) / best_b_final_loss * 100
    kill2 = quality_diff > 3.0
    print(f"\n  Proto A val_loss (after prune): {a_mean_loss_after:.4f}")
    print(f"  Best Proto B val_loss (final):  {best_b_final_loss:.4f}")
    print(f"  Quality difference: {quality_diff:+.2f}%")
    print(f"  Threshold: >3% = KILL")
    if kill2:
        print(f"  KILL: Freeze-then-prune quality degrades too much")
    else:
        print(f"  PASS: Freeze-then-prune quality within tolerance")

    # Per-layer analysis
    print(f"\n{'='*80}")
    print(f"  PER-LAYER DEATH RATES")
    print(f"{'='*80}")

    print(f"\n  {'Protocol':<30} | " + " | ".join(f"  Layer {l}" for l in range(N_LAYERS)))
    print(f"  {'-'*30}-+-" + "-+-".join(f"{'-'*8}" for _ in range(N_LAYERS)))

    # Control
    ctrl_plr = [statistics.mean([all_seeds_results[s]["control"]["per_layer_rates"][l] for s in seeds]) for l in range(N_LAYERS)]
    row = f"  {'control (no prune)':<30} | " + " | ".join(f" {r:>5.1%} " for r in ctrl_plr)
    print(row)

    # Proto A
    a_plr = [statistics.mean([all_seeds_results[s]["proto_A"]["per_layer_rates"][l] for s in seeds]) for l in range(N_LAYERS)]
    row = f"  {'A: freeze-then-prune':<30} | " + " | ".join(f" {r:>5.1%} " for r in a_plr)
    print(row)

    # Proto B at each checkpoint
    for s_mid in S_MID_POINTS:
        key = f"proto_B_S{s_mid}"
        b_plr = [statistics.mean([all_seeds_results[s][key]["per_layer_rates_at_prune"][l] for s in seeds]) for l in range(N_LAYERS)]
        row = f"  {'B: mid-prune S=' + str(s_mid):<30} | " + " | ".join(f" {r:>5.1%} " for r in b_plr)
        print(row)

    # Revival-that-would-have-happened analysis
    # Compare mid-prune death set to final death set (what would have happened without pruning)
    print(f"\n{'='*80}")
    print(f"  REVIVAL ANALYSIS: What fraction of mid-prune dead set would have revived?")
    print(f"  (Comparing mid-checkpoint death set to control's final death set)")
    print(f"{'='*80}")

    for s_mid in S_MID_POINTS:
        key = f"proto_B_S{s_mid}"
        revival_rates = []
        false_positive_rates = []
        for s in seeds:
            mid_masks = all_seeds_results[s][key]
            # Dead at mid-point (would have been pruned)
            # Compare against control's death set at end
            # We can't directly compare masks since model states differ
            # Instead, the death_rate_at_prune vs control death_rate tells us
            # the "excess" dead capsules at mid-training
            mid_death = mid_masks["death_rate_at_prune"]
            ctrl_death = all_seeds_results[s]["control"]["death_rate"]
            # Capsules dead at mid but alive at end = false positives
            excess = mid_death - ctrl_death
            revival_rates.append(excess)

        mean_excess = statistics.mean(revival_rates)
        print(f"\n  S_mid={s_mid:>4d}: mid_death={statistics.mean([all_seeds_results[s][key]['death_rate_at_prune'] for s in seeds]):.1%}, "
              f"final_death={statistics.mean(ctrl_deaths):.1%}, "
              f"excess (would-revive estimate): {mean_excess:+.1%}")

    # Overall verdict
    print(f"\n{'='*80}")
    print(f"  OVERALL VERDICT")
    print(f"{'='*80}")

    if kill1 and kill2:
        verdict = "KILL"
        print(f"\n  KILL: Both criteria triggered. Freeze-then-prune offers no advantage.")
    elif kill1:
        verdict = "KILL"
        print(f"\n  KILL (criterion 1): Freeze-then-prune does not yield meaningfully")
        print(f"  more dead capsules than mid-training pruning.")
    elif kill2:
        verdict = "KILL"
        print(f"\n  KILL (criterion 2): Freeze-then-prune quality degrades too much")
        print(f"  compared to mid-training pruning.")
    else:
        verdict = "PASS"
        print(f"\n  PASS: Freeze-then-prune yields higher pruning yield with")
        print(f"  acceptable quality. Recommended as the pruning protocol.")
        print(f"  Implication: profile dead capsules AFTER training completes,")
        print(f"  not during training when revival can invalidate the profiling.")

    return verdict


def main():
    """Run across 3 seeds and analyze."""
    seeds = [42, 123, 7]
    all_seeds_results = {}

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"  SEED {seed}")
        print(f"{'='*80}")
        all_seeds_results[seed] = run_full_experiment(seed=seed)

    verdict = analyze_results(all_seeds_results)
    return verdict


if __name__ == "__main__":
    main()
