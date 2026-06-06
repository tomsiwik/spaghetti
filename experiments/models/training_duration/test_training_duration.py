"""Training Duration vs Death Rate experiment (Exp 17).

Measures how ReLU capsule death rate changes with training duration.
Sweeps fine-tuning steps from 0 (pretrained base) to 3200 while keeping
all other hyperparameters identical to Exp 10.

Protocol:
  1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
  2. For each step count S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
     a. Start from the pretrained base (deepcopy)
     b. Freeze attention, fine-tune MLP only for S steps on domain data
     c. Profile activation frequencies on domain validation data
     d. Record: death rate per layer, aggregate death rate, val loss
  3. Analyze trajectory: monotonic? equilibrium? recovery?

Kill criteria:
  1. If death rate decreases from 200 to 3200 steps by >5pp: early death
     is transient, pruning needs re-evaluation for macro
  2. If death rate at 3200 < 30%: pruning opportunity shrinks at macro
     training durations
  3. If death rate std across seeds > 20pp at any step count: unreliable
"""

import copy
import statistics
import math

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


# Step counts to sweep (geometric spacing on log scale)
STEP_COUNTS = [0, 50, 100, 200, 400, 800, 1600, 3200]

# Domain for fine-tuning (use first domain: a_m)
DOMAIN = "a_m"


def run_duration_experiment(seed=42, domain_name=DOMAIN):
    """Run the training duration sweep for one seed.

    Returns:
        step_data: list of dicts, one per step count, containing:
            - steps: number of fine-tuning steps
            - death_rate: aggregate death rate (fraction)
            - per_layer_death: list of per-layer death rates
            - val_loss: validation loss on the domain
            - per_layer_alive_mean_freq: mean freq of alive capsules per layer
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
    # 2. Sweep fine-tuning steps
    # ============================================================
    step_data = []

    for S in STEP_COUNTS:
        print(f"  [S={S:>4d}] ", end="", flush=True)

        # Start from pretrained base
        model = copy.deepcopy(base)

        if S > 0:
            # Fine-tune MLP only
            _freeze_attention(model)
            train(model, train_ds, steps=S,
                  batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
            model.unfreeze()

        # Profile activations on domain validation data
        freqs = profile_activations(
            model, val_ds,
            n_batches=20, batch_size=32, seed=seed,
        )

        # Compute death statistics
        per_layer_death = []
        per_layer_alive_freq = []
        total_dead = 0
        total_caps = 0

        for l_idx, freq in enumerate(freqs):
            mx.eval(freq)
            freq_list = freq.tolist()
            P = len(freq_list)
            n_dead = sum(1 for f in freq_list if f == 0.0)
            alive_freqs = [f for f in freq_list if f > 0.0]
            mean_alive = sum(alive_freqs) / len(alive_freqs) if alive_freqs else 0.0

            per_layer_death.append(n_dead / P)
            per_layer_alive_freq.append(mean_alive)
            total_dead += n_dead
            total_caps += P

        death_rate = total_dead / total_caps

        # Evaluate val loss
        val_loss = evaluate(model, val_ds, batch_size=BATCH_SIZE)

        entry = {
            "steps": S,
            "death_rate": death_rate,
            "per_layer_death": per_layer_death,
            "val_loss": val_loss,
            "per_layer_alive_freq": per_layer_alive_freq,
        }
        step_data.append(entry)

        print(f"death={death_rate:.1%}, val_loss={val_loss:.4f}, "
              f"per_layer=[{', '.join(f'{d:.0%}' for d in per_layer_death)}]")

    return step_data


def fit_exponential(step_data):
    """Fit delta(S) = delta_0 + delta_inf * (1 - exp(-S/tau)) to data.

    Uses simple grid search (no scipy dependency).

    Returns:
        best_params: dict with delta_0, delta_inf, tau, r_squared
    """
    # Extract data points
    steps = [d["steps"] for d in step_data]
    deaths = [d["death_rate"] for d in step_data]

    # Grid search for best fit
    best_r2 = -float("inf")
    best_params = {}

    delta_0_candidates = [deaths[0]]  # Use measured S=0 death rate
    delta_inf_range = [i * 0.05 for i in range(1, 20)]  # 0.05 to 0.95
    tau_range = [25, 50, 100, 150, 200, 300, 500, 800, 1200, 2000]

    # Mean for R^2 computation
    mean_death = sum(deaths) / len(deaths)
    ss_tot = sum((d - mean_death) ** 2 for d in deaths)

    for d0 in delta_0_candidates:
        for dinf in delta_inf_range:
            for tau in tau_range:
                predicted = [d0 + dinf * (1 - math.exp(-s / tau)) if s > 0 else d0
                             for s in steps]
                ss_res = sum((a - p) ** 2 for a, p in zip(deaths, predicted))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        "delta_0": d0,
                        "delta_inf": dinf,
                        "tau": tau,
                        "r_squared": r2,
                        "asymptotic_death": d0 + dinf,
                    }

    return best_params


def check_monotonicity(step_data):
    """Check if death rate is monotonically non-decreasing.

    Returns:
        is_monotonic: True if death never decreases
        violations: list of (S_from, S_to, delta_from, delta_to) for decreases
    """
    violations = []
    for i in range(1, len(step_data)):
        if step_data[i]["death_rate"] < step_data[i-1]["death_rate"] - 0.005:
            # Allow 0.5pp tolerance for measurement noise
            violations.append((
                step_data[i-1]["steps"],
                step_data[i]["steps"],
                step_data[i-1]["death_rate"],
                step_data[i]["death_rate"],
            ))
    return len(violations) == 0, violations


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_step_data = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        data = run_duration_experiment(seed=seed)
        all_step_data.append(data)

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate: Death Rate vs Training Steps")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>6} | {'Death Rate':>10} {'Std':>8} | {'Val Loss':>10} {'Std':>8} | "
          f"{'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5}")
    print("  " + "-" * 80)

    for s_idx, S in enumerate(STEP_COUNTS):
        death_rates = [d[s_idx]["death_rate"] for d in all_step_data]
        val_losses = [d[s_idx]["val_loss"] for d in all_step_data]
        per_layer = [[d[s_idx]["per_layer_death"][l] for d in all_step_data]
                      for l in range(4)]

        mean_dr = statistics.mean(death_rates)
        std_dr = statistics.stdev(death_rates) if len(death_rates) > 1 else 0
        mean_vl = statistics.mean(val_losses)
        std_vl = statistics.stdev(val_losses) if len(val_losses) > 1 else 0
        layer_means = [statistics.mean(pl) for pl in per_layer]

        print(f"  {S:>6} | {mean_dr:>9.1%} {std_dr:>7.1%} | {mean_vl:>10.4f} {std_vl:>7.4f} | "
              f"{layer_means[0]:>4.0%} {layer_means[1]:>4.0%} "
              f"{layer_means[2]:>4.0%} {layer_means[3]:>4.0%}")

    # ============================================================
    # Monotonicity check
    # ============================================================
    print(f"\n{'='*70}")
    print("  Monotonicity Analysis")
    print(f"{'='*70}")

    for seed_idx, seed in enumerate(seeds):
        is_mono, violations = check_monotonicity(all_step_data[seed_idx])
        if is_mono:
            print(f"  Seed {seed}: MONOTONIC (death never decreases)")
        else:
            print(f"  Seed {seed}: NON-MONOTONIC ({len(violations)} violations)")
            for s_from, s_to, d_from, d_to in violations:
                print(f"    S={s_from}->{s_to}: {d_from:.1%} -> {d_to:.1%} "
                      f"(decreased {d_from - d_to:.1%})")

    # Aggregate monotonicity
    agg_data = []
    for s_idx in range(len(STEP_COUNTS)):
        mean_dr = statistics.mean([d[s_idx]["death_rate"] for d in all_step_data])
        agg_data.append({"steps": STEP_COUNTS[s_idx], "death_rate": mean_dr})

    is_agg_mono, agg_violations = check_monotonicity(agg_data)
    if is_agg_mono:
        print(f"\n  Aggregate: MONOTONIC (3-seed mean death never decreases)")
    else:
        print(f"\n  Aggregate: NON-MONOTONIC")
        for s_from, s_to, d_from, d_to in agg_violations:
            print(f"    S={s_from}->{s_to}: {d_from:.1%} -> {d_to:.1%}")

    # ============================================================
    # Curve fitting
    # ============================================================
    print(f"\n{'='*70}")
    print("  Curve Fitting (Saturating Exponential)")
    print(f"{'='*70}")

    # Fit to aggregate data
    params = fit_exponential(agg_data)
    print(f"\n  delta(S) = {params['delta_0']:.3f} + {params['delta_inf']:.3f} * (1 - exp(-S/{params['tau']:.0f}))")
    print(f"  Asymptotic death rate: {params['asymptotic_death']:.1%}")
    print(f"  Time constant (tau): {params['tau']:.0f} steps")
    print(f"  R-squared: {params['r_squared']:.4f}")

    # Per-seed fits
    for seed_idx, seed in enumerate(seeds):
        p = fit_exponential(all_step_data[seed_idx])
        print(f"  Seed {seed}: asymptotic={p['asymptotic_death']:.1%}, "
              f"tau={p['tau']:.0f}, R2={p['r_squared']:.4f}")

    # ============================================================
    # Exp 10 replication check
    # ============================================================
    print(f"\n{'='*70}")
    print("  Exp 10 Replication Check (S=200)")
    print(f"{'='*70}")

    s200_rates = [d[STEP_COUNTS.index(200)]["death_rate"] for d in all_step_data]
    mean_s200 = statistics.mean(s200_rates)
    exp10_ref = 0.543  # From Exp 10

    print(f"\n  Exp 10 reference: {exp10_ref:.1%}")
    print(f"  This experiment (S=200): {mean_s200:.1%}")
    print(f"  Difference: {abs(mean_s200 - exp10_ref):.1%}")
    if abs(mean_s200 - exp10_ref) < 0.10:
        print("  PASS: Within 10pp of Exp 10 (expected: some variance from domain-specific vs full profiling)")
    else:
        print("  WARNING: Large discrepancy with Exp 10")

    # ============================================================
    # Val loss vs death rate correlation
    # ============================================================
    print(f"\n{'='*70}")
    print("  Val Loss vs Death Rate Correlation")
    print(f"{'='*70}")

    # Compute Pearson correlation
    all_deaths = []
    all_losses = []
    for data in all_step_data:
        for entry in data:
            if entry["steps"] > 0:  # Exclude S=0 (no fine-tuning)
                all_deaths.append(entry["death_rate"])
                all_losses.append(entry["val_loss"])

    if len(all_deaths) > 2:
        mean_d = sum(all_deaths) / len(all_deaths)
        mean_l = sum(all_losses) / len(all_losses)
        cov = sum((d - mean_d) * (l - mean_l) for d, l in zip(all_deaths, all_losses)) / (len(all_deaths) - 1)
        std_d = (sum((d - mean_d)**2 for d in all_deaths) / (len(all_deaths) - 1)) ** 0.5
        std_l = (sum((l - mean_l)**2 for l in all_losses) / (len(all_losses) - 1)) ** 0.5
        corr = cov / (std_d * std_l) if std_d > 0 and std_l > 0 else 0

        print(f"\n  Pearson r(death_rate, val_loss): {corr:.3f}")
        if abs(corr) > 0.7:
            print("  Strong correlation: death rate and quality are coupled")
        elif abs(corr) > 0.3:
            print("  Moderate correlation: death and quality partially coupled")
        else:
            print("  Weak correlation: death and quality are largely independent")

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    death_200 = statistics.mean([d[STEP_COUNTS.index(200)]["death_rate"] for d in all_step_data])
    death_3200 = statistics.mean([d[STEP_COUNTS.index(3200)]["death_rate"] for d in all_step_data])

    # Kill 1: Death decreases from 200 to 3200
    kill1 = death_3200 < death_200 - 0.05
    print(f"\n  Kill 1: Death at 3200 ({death_3200:.1%}) vs 200 ({death_200:.1%}) = "
          f"delta {death_3200 - death_200:+.1%}")
    if kill1:
        print(f"    KILL: Death DECREASED by >{5}pp. Early death is transient.")
    else:
        print(f"    PASS: Death did not decrease significantly.")

    # Kill 2: Death at 3200 < 30%
    kill2 = death_3200 < 0.30
    print(f"  Kill 2: Death at 3200 = {death_3200:.1%}")
    if kill2:
        print(f"    KILL: Death below 30%. Pruning opportunity shrinks at macro.")
    else:
        print(f"    PASS: Death rate remains substantial ({death_3200:.1%} >= 30%).")

    # Kill 3: High variance
    max_std = 0
    for s_idx in range(len(STEP_COUNTS)):
        rates = [d[s_idx]["death_rate"] for d in all_step_data]
        std = statistics.stdev(rates) if len(rates) > 1 else 0
        max_std = max(max_std, std)

    kill3 = max_std > 0.20
    print(f"  Kill 3: Max std across step counts = {max_std:.1%}")
    if kill3:
        print(f"    KILL: High variance (>20pp). Measurement unreliable.")
    else:
        print(f"    PASS: Measurement stable across seeds.")

    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    if is_agg_mono and not kill1 and not kill2:
        print("\n  FINDING: Death is MONOTONICALLY INCREASING and stabilizes")
        print(f"  at {params['asymptotic_death']:.1%} (tau={params['tau']:.0f} steps).")
        print("  The 200-step measurement from Exp 10 is directionally correct.")
        print("  At macro scale, expect HIGHER death rates with longer training.")
        print("  Pruning opportunity is robust to training duration.")
    elif kill1:
        print("\n  FINDING: Death is TRANSIENT (decreases with more training).")
        print("  The Exp 10 measurement overestimates long-term death.")
        print("  Pruning opportunity may be weaker at macro scale.")
    elif kill2:
        print("\n  FINDING: Death at long training is LOW (<30%).")
        print("  Pruning opportunity shrinks substantially at macro scale.")
    else:
        print("\n  FINDING: Mixed results. See detailed analysis above.")

    if not kill1 and not kill2:
        print(f"\n  MACRO PREDICTION: With {params['tau']:.0f}-step time constant,")
        print(f"  expect ~{params['asymptotic_death']:.0%} death rate regardless of")
        print(f"  training duration at macro scale (100K+ steps >> tau).")


if __name__ == "__main__":
    main()
