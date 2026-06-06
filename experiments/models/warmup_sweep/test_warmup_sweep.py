"""Warmup Fraction Sensitivity experiment (Exp 20).

Measures how the warmup fraction in a warmup+cosine LR schedule affects
the ReLU capsule death trajectory during fine-tuning. Extends Exp 19
(which used a fixed 10% warmup) with a 5-point sweep over warmup fraction.

Protocol:
  1. Pretrain base model on ALL data (300 steps, constant LR -- same as Exp 19)
  2. For each warmup fraction in {1%, 2%, 5%, 10%, 20%}:
     For each checkpoint S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
       a. Start from the pretrained base (deepcopy)
       b. Freeze attention, fine-tune MLP only for S steps using
          warmup+cosine schedule with the given warmup fraction
       c. Profile activation frequencies on domain validation data
       d. Record: death rate per layer, aggregate death rate, val loss
  3. Also run constant (no warmup) and cosine-only as controls.
  4. Analyze: identify minimum effective warmup fraction.

Kill criteria:
  1. All fractions >= 1% produce death rates within 5pp at S=50: warmup
     fraction does not matter.
  2. f_w=0.01 provides >90% of f_w=0.10 spike suppression: question is
     moot for all practical LLM recipes.
  3. Non-monotonic: some f_w shows MORE death than a smaller f_w: the
     linear death-vs-LR model is wrong.
"""

import copy
import math
import statistics
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, STEPS_PRETRAIN, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations
from ..lr_schedule_death.test_lr_schedule_death import (
    make_lr_schedule, train_with_schedule, _compute_death_stats,
    TOTAL_STEPS,
)


# Step counts to sweep (same as Exp 19)
STEP_COUNTS = [0, 50, 100, 200, 400, 800, 1600, 3200]

# Domain for fine-tuning
DOMAIN = "a_m"

# Warmup fractions to sweep
WARMUP_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]

# Condition labels
CONDITION_NAMES = ["constant", "cosine_only"] + [f"wc_{int(f*100):02d}" for f in WARMUP_FRACTIONS]


def make_warmup_cosine_schedule(warmup_frac, peak_lr=LR, total_steps=TOTAL_STEPS):
    """Create a warmup+cosine LR schedule with given warmup fraction.

    Args:
        warmup_frac: fraction of total steps for linear warmup (0 to 1)
        peak_lr: maximum learning rate
        total_steps: total fine-tuning steps

    Returns:
        schedule: MLX LR schedule callable
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))

    warmup = optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)
    cosine = optim.cosine_decay(peak_lr,
                                decay_steps=total_steps - warmup_steps,
                                end=0.0)
    return optim.join_schedules([warmup, cosine], [warmup_steps])


def run_warmup_sweep(seed=42, domain_name=DOMAIN):
    """Run the warmup fraction sweep for one seed.

    Returns:
        results: dict mapping condition_name -> list of checkpoint dicts
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
    # 1. Pretrain base model (300 steps, constant LR)
    # ============================================================
    print(f"  Pretraining base model (300 steps, constant LR)...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Profile S=0 baseline (shared across all conditions)
    freqs_0 = profile_activations(base, val_ds, n_batches=20, batch_size=32, seed=seed)
    base_death = _compute_death_stats(freqs_0)
    base_val_loss = evaluate(base, val_ds, batch_size=BATCH_SIZE)

    results = {}

    # ============================================================
    # 2. Control: constant LR (no warmup, no cosine)
    # ============================================================
    print(f"\n  Condition: constant (control)")
    print(f"  {'='*60}")

    step_data = []
    step_data.append({
        "steps": 0,
        **base_death,
        "val_loss": base_val_loss,
        "lr_at_checkpoint": LR,
    })
    print(f"  [S=   0] death={base_death['death_rate']:.1%}")

    for S in STEP_COUNTS:
        if S == 0:
            continue
        print(f"  [S={S:>4d}] ", end="", flush=True)
        model = copy.deepcopy(base)
        _freeze_attention(model)
        train_result = train_with_schedule(model, train_ds, steps=S, schedule=LR, seed=seed)
        model.unfreeze()
        freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
        death_stats = _compute_death_stats(freqs)
        val_loss = evaluate(model, val_ds, batch_size=BATCH_SIZE)
        entry = {"steps": S, **death_stats, "val_loss": val_loss, "lr_at_checkpoint": LR}
        step_data.append(entry)
        print(f"death={death_stats['death_rate']:.1%}, val_loss={val_loss:.4f}")
    results["constant"] = step_data

    # ============================================================
    # 3. Control: cosine-only (no warmup)
    # ============================================================
    print(f"\n  Condition: cosine_only (control)")
    print(f"  {'='*60}")

    step_data = []
    step_data.append({
        "steps": 0,
        **base_death,
        "val_loss": base_val_loss,
        "lr_at_checkpoint": LR,
    })
    print(f"  [S=   0] death={base_death['death_rate']:.1%}")

    for S in STEP_COUNTS:
        if S == 0:
            continue
        print(f"  [S={S:>4d}] ", end="", flush=True)
        model = copy.deepcopy(base)
        _freeze_attention(model)
        cosine_schedule = optim.cosine_decay(LR, decay_steps=TOTAL_STEPS, end=0.0)
        train_result = train_with_schedule(model, train_ds, steps=S, schedule=cosine_schedule, seed=seed)
        model.unfreeze()
        freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
        death_stats = _compute_death_stats(freqs)
        val_loss = evaluate(model, val_ds, batch_size=BATCH_SIZE)
        lr_at_s = train_result["lr_trajectory"][-1] if train_result["lr_trajectory"] else LR
        entry = {"steps": S, **death_stats, "val_loss": val_loss, "lr_at_checkpoint": lr_at_s}
        step_data.append(entry)
        print(f"death={death_stats['death_rate']:.1%}, val_loss={val_loss:.4f}")
    results["cosine_only"] = step_data

    # ============================================================
    # 4. Warmup fraction sweep (warmup+cosine for each fraction)
    # ============================================================
    for f_w in WARMUP_FRACTIONS:
        cond_name = f"wc_{int(f_w*100):02d}"
        s_w = max(1, int(TOTAL_STEPS * f_w))
        print(f"\n  Condition: {cond_name} (warmup={f_w:.0%}, S_w={s_w})")
        print(f"  {'='*60}")

        step_data = []
        step_data.append({
            "steps": 0,
            **base_death,
            "val_loss": base_val_loss,
            "lr_at_checkpoint": 0.0,  # warmup starts at 0
        })
        print(f"  [S=   0] death={base_death['death_rate']:.1%}")

        for S in STEP_COUNTS:
            if S == 0:
                continue
            print(f"  [S={S:>4d}] ", end="", flush=True)
            model = copy.deepcopy(base)
            _freeze_attention(model)

            schedule = make_warmup_cosine_schedule(f_w, peak_lr=LR, total_steps=TOTAL_STEPS)
            train_result = train_with_schedule(model, train_ds, steps=S, schedule=schedule, seed=seed)
            model.unfreeze()

            freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
            death_stats = _compute_death_stats(freqs)
            val_loss = evaluate(model, val_ds, batch_size=BATCH_SIZE)
            lr_at_s = train_result["lr_trajectory"][-1] if train_result["lr_trajectory"] else 0.0

            entry = {"steps": S, **death_stats, "val_loss": val_loss, "lr_at_checkpoint": lr_at_s}
            step_data.append(entry)
            print(f"death={death_stats['death_rate']:.1%}, val_loss={val_loss:.4f}, lr={lr_at_s:.2e}")
        results[cond_name] = step_data

    return results


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        results = run_warmup_sweep(seed=seed)
        all_results.append(results)

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate: Death Rate by Condition and Step Count")
    print(f"{'='*70}")

    # Aggregate data structure
    agg = {}  # condition -> list of {"steps": S, "death_rate": mean, "std": std}
    for cond in CONDITION_NAMES:
        agg[cond] = []

    # Table header
    header = f"  {'Steps':>6} |"
    for cond in CONDITION_NAMES:
        header += f" {cond:>10} |"
    print(f"\n{header}")
    print("  " + "-" * (10 + 13 * len(CONDITION_NAMES)))

    for s_idx, S in enumerate(STEP_COUNTS):
        line = f"  {S:>6} |"
        for cond in CONDITION_NAMES:
            rates = [r[cond][s_idx]["death_rate"] for r in all_results]
            mean_dr = statistics.mean(rates)
            std_dr = statistics.stdev(rates) if len(rates) > 1 else 0
            agg[cond].append({"steps": S, "death_rate": mean_dr, "std": std_dr})
            line += f" {mean_dr:>5.1%}+/-{std_dr:>3.1%} |"
        print(line)

    # ============================================================
    # Val loss table
    # ============================================================
    print(f"\n  Val Loss at S=3200 (3-seed mean):")
    print(f"  {'Condition':>14} | {'Val Loss':>10} | {'Death Rate':>10}")
    print("  " + "-" * 45)
    s3200_idx = STEP_COUNTS.index(3200)
    for cond in CONDITION_NAMES:
        losses = [r[cond][s3200_idx]["val_loss"] for r in all_results]
        deaths = [r[cond][s3200_idx]["death_rate"] for r in all_results]
        print(f"  {cond:>14} | {statistics.mean(losses):>10.4f} | {statistics.mean(deaths):>9.1%}")

    # ============================================================
    # Spike analysis at S=50
    # ============================================================
    print(f"\n{'='*70}")
    print("  Death Spike Analysis at S=50")
    print(f"{'='*70}")

    s50_idx = STEP_COUNTS.index(50)
    const_50 = agg["constant"][s50_idx]["death_rate"]
    base_rate = agg["constant"][0]["death_rate"]  # S=0 baseline

    print(f"\n  S=0 baseline: {base_rate:.1%}")
    print(f"  Constant LR at S=50: {const_50:.1%} (spike = +{const_50 - base_rate:.1%}pp)")
    print()
    print(f"  {'Condition':>14} | {'Death@50':>10} | {'Spike':>8} | {'Suppression':>12} | {'% of 10% benefit':>18}")
    print("  " + "-" * 80)

    # Compute suppression relative to constant
    suppression_10 = const_50 - agg["wc_10"][s50_idx]["death_rate"]

    for cond in CONDITION_NAMES:
        death_50 = agg[cond][s50_idx]["death_rate"]
        spike = death_50 - base_rate
        suppression = const_50 - death_50
        pct_of_10 = (suppression / suppression_10 * 100) if suppression_10 > 0 else 0
        print(f"  {cond:>14} | {death_50:>9.1%} | {spike:>+7.1%} | {suppression:>+11.1%} | {pct_of_10:>16.0f}%")

    # ============================================================
    # Warmup steps vs spike timescale
    # ============================================================
    print(f"\n{'='*70}")
    print("  Warmup Steps vs Death Spike Timescale")
    print(f"{'='*70}")

    T_spike = 50  # empirical from Exp 17
    print(f"\n  T_spike ~ {T_spike} steps (from Exp 17)")
    print()
    print(f"  {'f_w':>6} | {'S_w':>6} | {'R=S_w/T':>8} | {'Death@50':>10} | {'Death@3200':>12}")
    print("  " + "-" * 60)

    for f_w in WARMUP_FRACTIONS:
        cond = f"wc_{int(f_w*100):02d}"
        s_w = max(1, int(TOTAL_STEPS * f_w))
        R = s_w / T_spike
        death_50 = agg[cond][s50_idx]["death_rate"]
        death_3200 = agg[cond][s3200_idx]["death_rate"]
        print(f"  {f_w:>5.0%} | {s_w:>6d} | {R:>8.2f} | {death_50:>9.1%} | {death_3200:>11.1%}")

    # ============================================================
    # Per-layer analysis at S=3200
    # ============================================================
    print(f"\n{'='*70}")
    print("  Per-Layer Death at S=3200")
    print(f"{'='*70}")

    print(f"\n  {'Condition':>14} | {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5} | {'Agg':>5}")
    print("  " + "-" * 55)
    for cond in CONDITION_NAMES:
        layer_deaths = []
        for l in range(4):
            layer_rates = [r[cond][s3200_idx]["per_layer_death"][l] for r in all_results]
            layer_deaths.append(statistics.mean(layer_rates))
        agg_rate = agg[cond][s3200_idx]["death_rate"]
        print(f"  {cond:>14} | {layer_deaths[0]:>4.0%} {layer_deaths[1]:>4.0%} "
              f"{layer_deaths[2]:>4.0%} {layer_deaths[3]:>4.0%} | {agg_rate:>4.1%}")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Kill 1: All fractions >= 1% within 5pp at S=50
    wc_deaths_50 = {f"wc_{int(f*100):02d}": agg[f"wc_{int(f*100):02d}"][s50_idx]["death_rate"]
                    for f in WARMUP_FRACTIONS}
    max_wc_50 = max(wc_deaths_50.values())
    min_wc_50 = min(wc_deaths_50.values())
    range_50 = max_wc_50 - min_wc_50

    kill1 = range_50 < 0.05
    print(f"\n  Kill 1: All warmup fractions within 5pp at S=50?")
    for cond, death in wc_deaths_50.items():
        print(f"    {cond}: {death:.1%}")
    print(f"    Range: {range_50:.1%}")
    if kill1:
        print(f"    KILL: Range {range_50:.1%} < 5pp. Warmup fraction does not matter.")
    else:
        print(f"    PASS: Range {range_50:.1%} >= 5pp. Warmup fraction matters.")

    # Kill 2: f_w=0.01 provides >90% of f_w=0.10 benefit
    supp_01 = const_50 - agg["wc_01"][s50_idx]["death_rate"]
    supp_10 = const_50 - agg["wc_10"][s50_idx]["death_rate"]
    ratio_01_10 = supp_01 / supp_10 if supp_10 > 0 else 0

    kill2 = ratio_01_10 > 0.90
    print(f"\n  Kill 2: f_w=0.01 provides >90% of f_w=0.10 spike suppression?")
    print(f"    Suppression at f_w=0.01: {supp_01:+.1%}")
    print(f"    Suppression at f_w=0.10: {supp_10:+.1%}")
    print(f"    Ratio: {ratio_01_10:.1%}")
    if kill2:
        print(f"    KILL: 1% warmup captures {ratio_01_10:.0%} of 10% warmup benefit. Question moot.")
    else:
        print(f"    PASS: 1% warmup captures only {ratio_01_10:.0%} of 10% warmup benefit. Fraction matters.")

    # Kill 3: Non-monotonic (any inversion)
    inversions = []
    for i in range(len(WARMUP_FRACTIONS) - 1):
        cond_a = f"wc_{int(WARMUP_FRACTIONS[i]*100):02d}"
        cond_b = f"wc_{int(WARMUP_FRACTIONS[i+1]*100):02d}"
        death_a = agg[cond_a][s50_idx]["death_rate"]
        death_b = agg[cond_b][s50_idx]["death_rate"]
        if death_b > death_a + 0.02:  # 2pp tolerance for noise
            inversions.append((WARMUP_FRACTIONS[i], WARMUP_FRACTIONS[i+1], death_a, death_b))

    kill3 = len(inversions) > 0
    print(f"\n  Kill 3: Non-monotonic at S=50 (more warmup = more death)?")
    if kill3:
        print(f"    KILL: {len(inversions)} inversion(s) found:")
        for f_a, f_b, d_a, d_b in inversions:
            print(f"      f_w={f_a:.0%} ({d_a:.1%}) -> f_w={f_b:.0%} ({d_b:.1%})")
    else:
        print(f"    PASS: Monotonically decreasing death with increasing warmup.")

    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    # ============================================================
    # Critical threshold estimation
    # ============================================================
    print(f"\n{'='*70}")
    print("  Critical Warmup Fraction Estimation")
    print(f"{'='*70}")

    # Find where suppression crosses 50% and 80% of max benefit
    supp_20 = const_50 - agg["wc_20"][s50_idx]["death_rate"]
    max_suppression = supp_20  # largest warmup = most suppression

    print(f"\n  Max suppression (f_w=20%): {max_suppression:+.1%}")
    print()

    for threshold_pct in [50, 80, 90]:
        threshold = threshold_pct / 100 * max_suppression
        # Linear interpolation to find crossing point
        found = False
        for i, f_w in enumerate(WARMUP_FRACTIONS):
            cond = f"wc_{int(f_w*100):02d}"
            supp = const_50 - agg[cond][s50_idx]["death_rate"]
            if supp >= threshold:
                if i == 0:
                    print(f"  {threshold_pct}% of max benefit: achieved at f_w <= {f_w:.0%} (S_w <= {max(1, int(TOTAL_STEPS * f_w))})")
                else:
                    prev_f = WARMUP_FRACTIONS[i-1]
                    prev_cond = f"wc_{int(prev_f*100):02d}"
                    prev_supp = const_50 - agg[prev_cond][s50_idx]["death_rate"]
                    # Linear interpolation
                    frac = (threshold - prev_supp) / (supp - prev_supp) if supp != prev_supp else 0.5
                    interp_f = prev_f + frac * (f_w - prev_f)
                    interp_sw = max(1, int(TOTAL_STEPS * interp_f))
                    print(f"  {threshold_pct}% of max benefit: ~f_w={interp_f:.1%} (S_w~{interp_sw}), R~{interp_sw/T_spike:.1f}")
                found = True
                break
        if not found:
            print(f"  {threshold_pct}% of max benefit: not reached by f_w=20%")

    # ============================================================
    # Equilibrium sensitivity
    # ============================================================
    print(f"\n{'='*70}")
    print("  Equilibrium Death Rate Sensitivity (S=3200)")
    print(f"{'='*70}")

    print(f"\n  {'f_w':>6} | {'Death@3200':>12} | {'vs constant':>12} | {'vs f_w=10%':>12}")
    print("  " + "-" * 55)

    const_3200 = agg["constant"][s3200_idx]["death_rate"]
    wc10_3200 = agg["wc_10"][s3200_idx]["death_rate"]

    for cond in CONDITION_NAMES:
        death_3200 = agg[cond][s3200_idx]["death_rate"]
        vs_const = death_3200 - const_3200
        vs_10 = death_3200 - wc10_3200
        print(f"  {cond:>14} | {death_3200:>11.1%} | {vs_const:>+11.1%} | {vs_10:>+11.1%}")

    # ============================================================
    # MATH.md prediction validation
    # ============================================================
    print(f"\n{'='*70}")
    print("  MATH.md Prediction Validation")
    print(f"{'='*70}")

    # Compare predicted vs actual suppression factors
    print(f"\n  {'f_w':>6} | {'S_w':>4} | {'Predicted F':>12} | {'Predicted death@50':>20} | {'Actual death@50':>16}")
    print("  " + "-" * 80)

    delta_0 = base_rate
    delta_spike = const_50 - delta_0

    for f_w in WARMUP_FRACTIONS:
        s_w = max(1, int(TOTAL_STEPS * f_w))
        # Suppression factor from MATH.md Section 3.2
        if s_w >= 50:
            F = 25.5 / s_w
        else:
            F = 1 - (s_w - 1) / 100
        predicted_death = delta_0 + F * delta_spike
        cond = f"wc_{int(f_w*100):02d}"
        actual_death = agg[cond][s50_idx]["death_rate"]
        print(f"  {f_w:>5.0%} | {s_w:>4d} | {F:>11.3f} | {predicted_death:>19.1%} | {actual_death:>15.1%}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    print(f"\n  Kill criteria: {n_kills}/3 triggered")
    if not kill1:
        print("  - Warmup fraction DOES affect the death spike (range > 5pp)")
    else:
        print("  - Warmup fraction does NOT meaningfully affect the death spike")

    if not kill2:
        print(f"  - 1% warmup captures only {ratio_01_10:.0%} of 10% warmup benefit")
        print("    -> Macro prediction (20% dead) requires warmup > 1%")
    else:
        print(f"  - 1% warmup captures {ratio_01_10:.0%} of 10% warmup benefit")
        print("    -> Macro prediction (20% dead) holds for typical LLM warmup fractions")

    if kill3:
        print("  - WARNING: Non-monotonic relationship -- linear LR-death model is wrong")

    # Practical recommendation
    print(f"\n  Practical recommendation:")
    # Find minimum fraction that achieves 80% of max suppression
    for f_w in WARMUP_FRACTIONS:
        cond = f"wc_{int(f_w*100):02d}"
        supp = const_50 - agg[cond][s50_idx]["death_rate"]
        if supp >= 0.80 * max_suppression:
            s_w = max(1, int(TOTAL_STEPS * f_w))
            print(f"  Minimum warmup for 80% spike suppression: f_w >= {f_w:.0%} (S_w >= {s_w})")
            print(f"  At this fraction: death@50 = {agg[cond][s50_idx]['death_rate']:.1%}, "
                  f"death@3200 = {agg[cond][s3200_idx]['death_rate']:.1%}")
            break


if __name__ == "__main__":
    main()
