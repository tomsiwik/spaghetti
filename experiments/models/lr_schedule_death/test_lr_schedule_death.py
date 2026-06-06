"""LR Schedule Impact on Death Trajectory experiment (Exp 19).

Measures how different learning rate schedules affect the ReLU capsule
death trajectory during fine-tuning. Extends Exp 17 (constant LR) with
three additional schedules: warmup-only, cosine-only, warmup+cosine.

Protocol:
  1. Pretrain base model on ALL data (300 steps, constant LR -- same as Exp 17)
  2. For each LR schedule in {constant, warmup, cosine, warmup+cosine}:
     For each checkpoint S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
       a. Start from the pretrained base (deepcopy)
       b. Freeze attention, fine-tune MLP only for S steps using the schedule
       c. Profile activation frequencies on domain validation data
       d. Record: death rate per layer, aggregate death rate, val loss
  3. Compare death trajectories across schedules

LR Schedules (all peak at LR=3e-3, total_steps used for schedule shape):
  - constant: LR = 3e-3 at all steps (Exp 17 baseline)
  - warmup: linear 0 -> 3e-3 over first 10% of total_steps, then constant
  - cosine: 3e-3 -> 0 cosine decay over total_steps
  - warmup_cosine: linear warmup (10%) then cosine decay (standard macro)

Kill criteria:
  1. Warmup death spike at S=50 differs from constant by <3pp: warmup has
     no meaningful effect on the critical spike phase
  2. Final death rate (S=3200) differs by <3pp across ALL schedules: LR
     schedule does not affect equilibrium death
  3. Cosine decay revival rate (death decrease S=200 to S=3200) is not higher
     than constant LR: Gurbuzbalaban revival prediction fails
"""

import copy
import math
import statistics
import time
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


# Step counts to sweep (geometric spacing, matching Exp 17)
STEP_COUNTS = [0, 50, 100, 200, 400, 800, 1600, 3200]

# Total steps for schedule computation (always 3200, the max)
TOTAL_STEPS = 3200

# Warmup fraction (10% of total steps)
WARMUP_FRACTION = 0.10

# Domain for fine-tuning
DOMAIN = "a_m"

# LR schedule names
SCHEDULES = ["constant", "warmup", "cosine", "warmup_cosine"]


def make_lr_schedule(schedule_name, peak_lr=LR, total_steps=TOTAL_STEPS,
                     warmup_frac=WARMUP_FRACTION):
    """Create an MLX LR schedule callable.

    Args:
        schedule_name: one of "constant", "warmup", "cosine", "warmup_cosine"
        peak_lr: maximum learning rate
        total_steps: total fine-tuning steps
        warmup_frac: fraction of steps for warmup

    Returns:
        schedule: callable that returns LR for a given step, or float for constant
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))

    if schedule_name == "constant":
        return peak_lr

    elif schedule_name == "warmup":
        # Linear warmup 0 -> peak_lr, then constant
        warmup = optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)
        constant = optim.linear_schedule(peak_lr, peak_lr, steps=total_steps)
        return optim.join_schedules([warmup, constant], [warmup_steps])

    elif schedule_name == "cosine":
        # Cosine decay from peak_lr -> 0
        return optim.cosine_decay(peak_lr, decay_steps=total_steps, end=0.0)

    elif schedule_name == "warmup_cosine":
        # Linear warmup then cosine decay
        warmup = optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)
        cosine = optim.cosine_decay(peak_lr,
                                    decay_steps=total_steps - warmup_steps,
                                    end=0.0)
        return optim.join_schedules([warmup, cosine], [warmup_steps])

    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


def train_with_schedule(model, dataset, steps, schedule, seed=42):
    """Train model with a given LR schedule.

    Similar to micro.train.train() but supports LR schedule callables.

    Args:
        model: the model to train
        dataset: training dataset
        steps: number of training steps
        schedule: LR schedule (float for constant, callable for dynamic)
        seed: random seed

    Returns:
        dict with final_loss, lr_trajectory (list of LRs at each step)
    """
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=schedule)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    lr_trajectory = []

    for step in range(1, steps + 1):
        inputs, targets = dataset.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())

        # Record current LR
        current_lr = optimizer.learning_rate
        if hasattr(current_lr, 'item'):
            current_lr = current_lr.item()
        lr_trajectory.append(float(current_lr))

    return {
        "final_loss": losses[-1] if losses else 0.0,
        "losses": losses,
        "lr_trajectory": lr_trajectory,
    }


def run_schedule_experiment(seed=42, domain_name=DOMAIN):
    """Run the LR schedule sweep for one seed.

    Returns:
        results: dict mapping schedule_name -> list of checkpoint dicts
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
    # 1. Pretrain base model (300 steps, constant LR -- same as Exp 17)
    # ============================================================
    print(f"  Pretraining base model (300 steps, constant LR)...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Profile S=0 baseline (shared across all schedules)
    freqs_0 = profile_activations(base, val_ds, n_batches=20, batch_size=32, seed=seed)
    base_death = _compute_death_stats(freqs_0)

    results = {}

    # ============================================================
    # 2. For each LR schedule, sweep checkpoints
    # ============================================================
    for sched_name in SCHEDULES:
        print(f"\n  Schedule: {sched_name}")
        print(f"  {'='*60}")

        schedule = make_lr_schedule(sched_name, peak_lr=LR, total_steps=TOTAL_STEPS)
        step_data = []

        # S=0 is shared
        step_data.append({
            "steps": 0,
            **base_death,
            "val_loss": evaluate(base, val_ds, batch_size=BATCH_SIZE),
            "lr_at_checkpoint": LR,
        })
        print(f"  [S=   0] death={base_death['death_rate']:.1%}, "
              f"per_layer=[{', '.join(f'{d:.0%}' for d in base_death['per_layer_death'])}]")

        for S in STEP_COUNTS:
            if S == 0:
                continue

            print(f"  [S={S:>4d}] ", end="", flush=True)

            # Start from pretrained base
            model = copy.deepcopy(base)
            _freeze_attention(model)

            # Create schedule for this specific run
            # Important: schedule is based on TOTAL_STEPS (3200), not S
            run_schedule = make_lr_schedule(sched_name, peak_lr=LR,
                                           total_steps=TOTAL_STEPS)

            # Train for exactly S steps with the schedule
            train_result = train_with_schedule(
                model, train_ds, steps=S, schedule=run_schedule, seed=seed,
            )

            model.unfreeze()

            # Profile activations
            freqs = profile_activations(
                model, val_ds, n_batches=20, batch_size=32, seed=seed,
            )
            death_stats = _compute_death_stats(freqs)

            # Val loss
            val_loss = evaluate(model, val_ds, batch_size=BATCH_SIZE)

            # LR at this checkpoint
            lr_at_s = train_result["lr_trajectory"][-1] if train_result["lr_trajectory"] else LR

            entry = {
                "steps": S,
                **death_stats,
                "val_loss": val_loss,
                "lr_at_checkpoint": lr_at_s,
            }
            step_data.append(entry)

            print(f"death={death_stats['death_rate']:.1%}, "
                  f"val_loss={val_loss:.4f}, lr={lr_at_s:.2e}, "
                  f"per_layer=[{', '.join(f'{d:.0%}' for d in death_stats['per_layer_death'])}]")

        results[sched_name] = step_data

    return results


def _compute_death_stats(freqs):
    """Compute death statistics from activation frequency tensors.

    Args:
        freqs: list of tensors, one per layer, each of shape (P,)

    Returns:
        dict with death_rate, per_layer_death, per_layer_alive_freq
    """
    per_layer_death = []
    per_layer_alive_freq = []
    total_dead = 0
    total_caps = 0

    for freq in freqs:
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

    return {
        "death_rate": total_dead / total_caps,
        "per_layer_death": per_layer_death,
        "per_layer_alive_freq": per_layer_alive_freq,
    }


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []  # list of dicts: schedule -> step_data

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        results = run_schedule_experiment(seed=seed)
        all_results.append(results)

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate: Death Rate by Schedule and Step Count")
    print(f"{'='*70}")

    # Table header
    print(f"\n  {'Steps':>6} | ", end="")
    for sched in SCHEDULES:
        print(f" {sched:>14} |", end="")
    print()
    print("  " + "-" * (10 + 17 * len(SCHEDULES)))

    # Collect aggregate data for analysis
    agg = {}  # schedule -> list of {"steps": S, "death_rate": mean, "std": std, ...}

    for sched in SCHEDULES:
        agg[sched] = []

    for s_idx, S in enumerate(STEP_COUNTS):
        print(f"  {S:>6} | ", end="")
        for sched in SCHEDULES:
            rates = [r[sched][s_idx]["death_rate"] for r in all_results]
            mean_dr = statistics.mean(rates)
            std_dr = statistics.stdev(rates) if len(rates) > 1 else 0
            agg[sched].append({
                "steps": S,
                "death_rate": mean_dr,
                "std": std_dr,
            })
            print(f" {mean_dr:>5.1%} +/-{std_dr:>4.1%} |", end="")
        print()

    # ============================================================
    # Val loss table
    # ============================================================
    print(f"\n  {'Steps':>6} | ", end="")
    for sched in SCHEDULES:
        print(f" {sched:>14} |", end="")
    print("  (val loss)")
    print("  " + "-" * (10 + 17 * len(SCHEDULES)))

    for s_idx, S in enumerate(STEP_COUNTS):
        print(f"  {S:>6} | ", end="")
        for sched in SCHEDULES:
            losses = [r[sched][s_idx]["val_loss"] for r in all_results]
            mean_vl = statistics.mean(losses)
            print(f"        {mean_vl:.4f} |", end="")
        print()

    # ============================================================
    # LR at each checkpoint
    # ============================================================
    print(f"\n  LR at each checkpoint:")
    print(f"  {'Steps':>6} | ", end="")
    for sched in SCHEDULES:
        print(f" {sched:>14} |", end="")
    print()
    print("  " + "-" * (10 + 17 * len(SCHEDULES)))

    for s_idx, S in enumerate(STEP_COUNTS):
        print(f"  {S:>6} | ", end="")
        for sched in SCHEDULES:
            lrs = [r[sched][s_idx]["lr_at_checkpoint"] for r in all_results]
            mean_lr = statistics.mean(lrs)
            print(f"       {mean_lr:.2e} |", end="")
        print()

    # ============================================================
    # Kill threshold analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Kill Threshold Analysis")
    print(f"{'='*70}")

    # Kill 1: Warmup effect on death spike at S=50
    const_50 = agg["constant"][STEP_COUNTS.index(50)]["death_rate"]
    warmup_50 = agg["warmup"][STEP_COUNTS.index(50)]["death_rate"]
    wc_50 = agg["warmup_cosine"][STEP_COUNTS.index(50)]["death_rate"]

    spike_diff_warmup = abs(warmup_50 - const_50)
    spike_diff_wc = abs(wc_50 - const_50)
    max_spike_diff = max(spike_diff_warmup, spike_diff_wc)

    kill1 = max_spike_diff < 0.03
    print(f"\n  Kill 1: Warmup effect on S=50 death spike")
    print(f"    constant:      {const_50:.1%}")
    print(f"    warmup:        {warmup_50:.1%} (diff: {spike_diff_warmup:.1%})")
    print(f"    warmup_cosine: {wc_50:.1%} (diff: {spike_diff_wc:.1%})")
    if kill1:
        print(f"    KILL: Max diff {max_spike_diff:.1%} < 3pp. Warmup has no meaningful effect.")
    else:
        print(f"    PASS: Max diff {max_spike_diff:.1%} >= 3pp. Warmup affects spike.")

    # Kill 2: Equilibrium death rate at S=3200
    s3200_idx = STEP_COUNTS.index(3200)
    deaths_3200 = {s: agg[s][s3200_idx]["death_rate"] for s in SCHEDULES}
    max_3200 = max(deaths_3200.values())
    min_3200 = min(deaths_3200.values())
    range_3200 = max_3200 - min_3200

    kill2 = range_3200 < 0.03
    print(f"\n  Kill 2: Equilibrium death rate at S=3200")
    for sched in SCHEDULES:
        print(f"    {sched:>14}: {deaths_3200[sched]:.1%}")
    print(f"    Range: {range_3200:.1%}")
    if kill2:
        print(f"    KILL: Range {range_3200:.1%} < 3pp. Schedule does not affect equilibrium.")
    else:
        print(f"    PASS: Range {range_3200:.1%} >= 3pp. Schedule affects equilibrium.")

    # Kill 3: Cosine decay revival (compare death decrease 200->3200)
    s200_idx = STEP_COUNTS.index(200)
    revival_const = agg["constant"][s200_idx]["death_rate"] - agg["constant"][s3200_idx]["death_rate"]
    revival_cosine = agg["cosine"][s200_idx]["death_rate"] - agg["cosine"][s3200_idx]["death_rate"]
    revival_wc = agg["warmup_cosine"][s200_idx]["death_rate"] - agg["warmup_cosine"][s3200_idx]["death_rate"]

    # Cosine/warmup_cosine should show MORE revival (larger decrease)
    best_cosine_revival = max(revival_cosine, revival_wc)
    kill3 = best_cosine_revival <= revival_const
    print(f"\n  Kill 3: Cosine decay revival (death decrease S=200 -> S=3200)")
    print(f"    constant:      {revival_const:+.1%} (Exp 17 baseline)")
    print(f"    cosine:        {revival_cosine:+.1%}")
    print(f"    warmup_cosine: {revival_wc:+.1%}")
    if kill3:
        print(f"    KILL: Cosine revival ({best_cosine_revival:+.1%}) <= constant ({revival_const:+.1%}).")
        print(f"    Gurbuzbalaban prediction fails: LR decay does not boost revival.")
    else:
        print(f"    PASS: Cosine revival ({best_cosine_revival:+.1%}) > constant ({revival_const:+.1%}).")
        print(f"    Gurbuzbalaban prediction confirmed: LR decay boosts revival.")

    n_kills = sum([kill1, kill2, kill3])
    print(f"\n  VERDICT: {n_kills}/3 kill criteria triggered")

    # ============================================================
    # Per-layer analysis at key checkpoints
    # ============================================================
    print(f"\n{'='*70}")
    print("  Per-Layer Death Rates at Key Checkpoints")
    print(f"{'='*70}")

    for S in [50, 200, 3200]:
        s_idx = STEP_COUNTS.index(S)
        print(f"\n  S={S}:")
        print(f"  {'Schedule':>14} | {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5} | {'Agg':>5}")
        print("  " + "-" * 55)
        for sched in SCHEDULES:
            layer_deaths = []
            for l in range(4):
                layer_rates = [r[sched][s_idx]["per_layer_death"][l] for r in all_results]
                layer_deaths.append(statistics.mean(layer_rates))
            agg_rate = agg[sched][s_idx]["death_rate"]
            print(f"  {sched:>14} | {layer_deaths[0]:>4.0%} {layer_deaths[1]:>4.0%} "
                  f"{layer_deaths[2]:>4.0%} {layer_deaths[3]:>4.0%} | {agg_rate:>4.1%}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    print(f"\n  Kill criteria: {n_kills}/3 triggered")
    if kill1:
        print("  - Warmup has NO effect on the initial death spike")
    else:
        print("  - Warmup DOES affect the initial death spike")
    if kill2:
        print("  - LR schedule does NOT affect equilibrium death rate")
    else:
        print("  - LR schedule DOES affect equilibrium death rate")
    if kill3:
        print("  - Cosine decay does NOT boost neural revival")
    else:
        print("  - Cosine decay DOES boost neural revival (Gurbuzbalaban confirmed)")

    # Practical implications
    print(f"\n  Practical implications for macro:")
    if kill1 and kill2:
        print("  - LR schedule does not meaningfully change pruning dynamics")
        print("  - Exp 17's constant-LR results transfer directly to macro")
        print("  - No schedule-specific pruning timing needed")
    elif not kill1:
        print(f"  - Warmup changes the spike: {warmup_50:.1%} vs {const_50:.1%} constant")
        if warmup_50 < const_50:
            print("  - Warmup REDUCES initial death -> fewer neurons to prune early")
        else:
            print("  - Warmup INCREASES initial death -> more neurons to prune early")
    if not kill2:
        best_sched = min(deaths_3200, key=deaths_3200.get)
        worst_sched = max(deaths_3200, key=deaths_3200.get)
        print(f"  - Best equilibrium: {best_sched} ({deaths_3200[best_sched]:.1%})")
        print(f"  - Worst equilibrium: {worst_sched} ({deaths_3200[worst_sched]:.1%})")


if __name__ == "__main__":
    main()
