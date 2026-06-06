"""Calibration LR/steps scaling law as function of N.

Parent (discriminability_n_gt_2) proved:
  - Router gradients are 5-7x smaller at N=8 vs N=2 (k/N dilution)
  - Phase transition softens with N but discriminability still predicts quality
  - At real scale (cos~0.0002), all experts maximally discriminable

Killed predecessor (dense_backprop_calibration) showed:
  - Dense backprop doesn't close gradient gap or speed convergence
  - k/N dilution is NOT the gradient bottleneck
  - Quality is similar despite 2-3x gradient difference
  - LR scaling is the cost-effective fix (0x overhead vs 4x for dense)

This experiment tests: does scaling calibration LR as N/k and/or steps as N/k
compensate for gradient attenuation and produce a usable scaling law?

Kill criteria:
  KC1: No monotonic relationship between N and optimal calibration steps
  KC2: LR scaling does not compensate for gradient attenuation (quality gap persists)
       Specifically: best LR-scaled config at N=8 does not close >50% of the
       quality gap between N=2(default_LR) and N=8(default_LR).
"""

import random
import statistics
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models.lora_procrustes.lora_procrustes import LoRAGPT
from experiments.models.lora_procrustes.test_lora_procrustes import (
    copy_weights, count_params, freeze_except_lora, reset_lora,
    get_deltas, apply_deltas_to_base,
    RoutedDeltaGPT,
)
from experiments.models.gap_as_signal.test_gap_as_signal import (
    flatten_deltas, unflatten_deltas, project_to_target_cosine,
    compute_r_squared,
)
from experiments.models.discriminability_n_gt_2.test_discriminability_n_gt_2 import (
    generate_n_experts_at_cosine,
)

# -- Config -----------------------------------------------------------------

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
BASE_LR = 3e-3

# Experiment parameters
TOP_K = 2
TARGET_COS = 0.0  # maximally discriminable (practical regime)

# Sweep dimensions
N_VALUES = [2, 4, 8, 16]
LR_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0, 8.0]
MAX_CAL_STEPS = 600  # enough headroom for larger N
CAL_EVAL_EVERY = 10


# -- Calibration with convergence tracking ----------------------------------

def calibrate_and_track(model, train_datasets, val_ds, steps, lr, seed=42):
    """Calibrate router and record loss curve for convergence analysis.

    Returns:
        loss_curve: list of (step, val_loss)
        final_val_loss: final validation loss (10-batch eval)
    """
    model.freeze()
    for router in model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    loss_curve = []
    n_domains = len(train_datasets)

    for step in range(1, steps + 1):
        domain_idx = step % n_domains
        inputs, targets = train_datasets[domain_idx].get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % CAL_EVAL_EVERY == 0 or step == 1:
            val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=5)
            loss_curve.append((step, val_loss))

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)
    model.unfreeze()

    return loss_curve, final_val_loss


# -- Convergence Step Finder ------------------------------------------------

def find_convergence_step(loss_curve, target_loss, patience=3):
    """Find first step where val loss drops below target and stays there.

    Args:
        loss_curve: list of (step, val_loss)
        target_loss: target to reach
        patience: number of consecutive evaluations below target

    Returns:
        convergence_step: step number, or None if never converged
    """
    consecutive = 0
    for step, val_loss in loss_curve:
        if val_loss <= target_loss:
            consecutive += 1
            if consecutive >= patience:
                # Return the step where it first dropped below
                return loss_curve[loss_curve.index((step, val_loss)) - patience + 1][0]
        else:
            consecutive = 0
    return None


def find_best_loss_at_step(loss_curve, step_budget):
    """Find the best (lowest) val loss achieved within step budget."""
    best = float('inf')
    for step, val_loss in loss_curve:
        if step <= step_budget:
            best = min(best, val_loss)
    return best


# -- Single Trial -----------------------------------------------------------

def run_lr_sweep_trial(n_experts, lr_mult, base_model, deltas_a, deltas_b,
                       train_datasets, val_ds, joint_val_loss, V, seed=42):
    """Run one calibration trial with specific N and LR multiplier."""
    actual_lr = BASE_LR * lr_mult

    # Generate experts at cos=0.0
    if n_experts == 2:
        deltas_b_proj, actual_cos = project_to_target_cosine(
            deltas_a, deltas_b, TARGET_COS)
        all_delta_sets = [deltas_a, deltas_b_proj]
    else:
        all_delta_sets, actual_cos, _, _ = generate_n_experts_at_cosine(
            deltas_a, deltas_b, TARGET_COS, n_experts)

    # Create routed model
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    model = RoutedDeltaGPT(base_copy, all_delta_sets, V, top_k=TOP_K)
    mx.eval(model.parameters())

    # Calibrate
    loss_curve, final_val = calibrate_and_track(
        model, train_datasets, val_ds,
        steps=MAX_CAL_STEPS, lr=actual_lr, seed=seed
    )

    vs_joint = (final_val - joint_val_loss) / joint_val_loss * 100

    # Find best loss at various step budgets
    best_at_100 = find_best_loss_at_step(loss_curve, 100)
    best_at_200 = find_best_loss_at_step(loss_curve, 200)
    best_at_300 = find_best_loss_at_step(loss_curve, 300)
    best_at_600 = find_best_loss_at_step(loss_curve, 600)

    return {
        'n_experts': n_experts,
        'lr_mult': lr_mult,
        'actual_lr': actual_lr,
        'final_val_loss': final_val,
        'vs_joint_pct': vs_joint,
        'loss_curve': loss_curve,
        'best_at_100': best_at_100,
        'best_at_200': best_at_200,
        'best_at_300': best_at_300,
        'best_at_600': best_at_600,
    }


# -- Full Experiment --------------------------------------------------------

def run_experiment(seed=42, verbose=True):
    """Run the LR scaling experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"CALIBRATION LR SCALING EXPERIMENT (seed={seed})")
        print(f"{'='*70}")

    mx.random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    splits = domain_split(docs)
    all_train, all_val = train_val_split(docs, seed=seed)

    train_a_docs, val_a_docs = train_val_split(splits["a_m"], seed=seed)
    train_b_docs, val_b_docs = train_val_split(splits["n_z"], seed=seed)

    train_a = CharDataset(train_a_docs, tokenizer, BASE["block_size"])
    val_a = CharDataset(val_a_docs, tokenizer, BASE["block_size"])
    train_b = CharDataset(train_b_docs, tokenizer, BASE["block_size"])
    val_b = CharDataset(val_b_docs, tokenizer, BASE["block_size"])
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    train_datasets = [train_a, train_b]

    # === 1. Joint training baseline ===
    if verbose:
        print("\n--- Joint training baseline ---")
    model_joint = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(model_joint.parameters())
    total_steps = 2 * FINETUNE_STEPS
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=BASE_LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    joint_val_loss = evaluate(model_joint, joint_val, BATCH_SIZE, n_batches=10)
    if verbose:
        print(f"  Joint val loss: {joint_val_loss:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=BASE_LR, seed=seed, log_every=300)

    # === 3. Fine-tune 2 LoRA experts ===
    def finetune_lora(domain_train, domain_val, domain_name):
        if verbose:
            print(f"\n--- Fine-tuning LoRA for {domain_name} ---")
        lora_model = get_model("lora_gpt", vocab_size=V, **BASE,
                               lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        mx.eval(lora_model.parameters())

        for l_idx in range(BASE['n_layer']):
            bl = base_model.layers[l_idx]
            ll = lora_model.layers[l_idx]
            ll.attn.wq.weight = bl.attn.wq.weight
            ll.attn.wk.weight = bl.attn.wk.weight
            ll.attn.wv.weight = bl.attn.wv.weight
            ll.attn.wo.weight = bl.attn.wo.weight
            ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
            ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
        lora_model.wte.weight = base_model.wte.weight
        lora_model.wpe.weight = base_model.wpe.weight
        lora_model.lm_head.weight = base_model.lm_head.weight
        mx.eval(lora_model.parameters())

        freeze_except_lora(lora_model)
        train(lora_model, domain_train, domain_val, steps=FINETUNE_STEPS,
              batch_size=BATCH_SIZE, lr=BASE_LR, seed=seed, log_every=300)
        lora_model.unfreeze()
        return lora_model

    lora_a = finetune_lora(train_a, val_a, "A (a-m)")
    lora_b = finetune_lora(train_b, val_b, "B (n-z)")

    # === 4. Extract deltas ===
    deltas_a = get_deltas(lora_a)
    deltas_b = get_deltas(lora_b)

    # === 5. LR sweep for each N ===
    results = {
        'seed': seed,
        'joint_val_loss': joint_val_loss,
        'trials': [],
    }

    total_configs = len(N_VALUES) * len(LR_MULTIPLIERS)
    config_idx = 0

    for n_exp in N_VALUES:
        for lr_mult in LR_MULTIPLIERS:
            config_idx += 1
            if verbose:
                print(f"\n[{config_idx}/{total_configs}] N={n_exp}, LR_mult={lr_mult:.1f}x "
                      f"(LR={BASE_LR * lr_mult:.4f})")

            trial = run_lr_sweep_trial(
                n_exp, lr_mult, base_model, deltas_a, deltas_b,
                train_datasets, joint_val, joint_val_loss, V,
                seed=seed
            )
            results['trials'].append(trial)

            if verbose:
                print(f"  Final: {trial['final_val_loss']:.4f} "
                      f"(vs joint: {trial['vs_joint_pct']:+.2f}%)")

    return results


# -- Analysis ---------------------------------------------------------------

def analyze_results(all_experiments):
    """Analyze: is there a monotonic scaling law for LR and steps?"""
    print(f"\n\n{'='*80}")
    print("CALIBRATION LR SCALING ANALYSIS")
    print(f"{'='*80}")

    # Aggregate trials across seeds
    # Key: (n_experts, lr_mult) -> list of trials
    by_config = {}
    joint_losses = [exp['joint_val_loss'] for exp in all_experiments]
    mean_joint = statistics.mean(joint_losses)

    for exp in all_experiments:
        for trial in exp['trials']:
            key = (trial['n_experts'], trial['lr_mult'])
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(trial)

    # === 1. Full Grid: Final Val Loss ===
    print(f"\n{'='*80}")
    print(f"FULL GRID: FINAL VAL LOSS (vs joint) [mean over {len(all_experiments)} seeds]")
    print(f"{'='*80}")
    print(f"\nJoint baseline: {mean_joint:.4f}")

    header = f"{'N':>4} | " + " | ".join(f"LR*{m:.1f}" for m in LR_MULTIPLIERS)
    print(f"\n{header}")
    print("-" * len(header))

    # Track best config per N
    best_per_n = {}  # n -> (lr_mult, mean_vs_joint)

    for n_exp in N_VALUES:
        row = f"{n_exp:>4} | "
        best_vj = float('inf')
        best_lr = 1.0
        for lr_mult in LR_MULTIPLIERS:
            key = (n_exp, lr_mult)
            if key in by_config:
                trials = by_config[key]
                mean_vj = statistics.mean([t['vs_joint_pct'] for t in trials])
                row += f"{mean_vj:>+7.2f}% | "
                if abs(mean_vj) < abs(best_vj):
                    best_vj = mean_vj
                    best_lr = lr_mult
            else:
                row += f"{'N/A':>8} | "
        print(row)
        best_per_n[n_exp] = (best_lr, best_vj)

    # === 2. Best LR per N ===
    print(f"\n{'='*80}")
    print("OPTIMAL LR MULTIPLIER PER N")
    print(f"{'='*80}")

    print(f"\n{'N':>4} | {'Best LR*':>8} | {'vs Joint':>9} | {'Predicted LR*':>13} | {'Match?':>6}")
    print("-" * 55)

    lr_scaling_matches = []
    for n_exp in N_VALUES:
        best_lr, best_vj = best_per_n[n_exp]
        predicted_lr = n_exp / TOP_K  # N/k scaling prediction
        match = abs(best_lr - predicted_lr) / predicted_lr < 0.5  # within 50%
        lr_scaling_matches.append(match)
        print(f"{n_exp:>4} | {best_lr:>8.1f}x | {best_vj:>+8.2f}% | {predicted_lr:>12.1f}x | "
              f"{'YES' if match else 'NO':>6}")

    # === 3. Convergence at Fixed Step Budgets ===
    print(f"\n{'='*80}")
    print("BEST QUALITY AT FIXED STEP BUDGETS (default LR)")
    print(f"{'='*80}")

    print(f"\n{'N':>4} | {'S=100':>8} | {'S=200':>8} | {'S=300':>8} | {'S=600':>8}")
    print("-" * 50)

    for n_exp in N_VALUES:
        key = (n_exp, 1.0)  # default LR
        if key in by_config:
            trials = by_config[key]
            b100 = statistics.mean([t['best_at_100'] for t in trials])
            b200 = statistics.mean([t['best_at_200'] for t in trials])
            b300 = statistics.mean([t['best_at_300'] for t in trials])
            b600 = statistics.mean([t['best_at_600'] for t in trials])
            print(f"{n_exp:>4} | {b100:>8.4f} | {b200:>8.4f} | {b300:>8.4f} | {b600:>8.4f}")

    # === 4. Optimal Steps at Scaled LR ===
    print(f"\n{'='*80}")
    print("CONVERGENCE WITH OPTIMAL LR SCALING")
    print(f"{'='*80}")

    # For each N, use the N/k-scaled LR and measure quality at each step budget
    print(f"\n{'N':>4} | {'LR*':>5} | {'S=100':>8} | {'S=200':>8} | {'S=300':>8} | {'S=600':>8}")
    print("-" * 55)

    step_scaling_data = {}
    for n_exp in N_VALUES:
        # Use the LR closest to N/k prediction
        predicted_lr = n_exp / TOP_K
        # Find closest available multiplier
        closest_lr = min(LR_MULTIPLIERS, key=lambda m: abs(m - predicted_lr))
        key = (n_exp, closest_lr)
        if key in by_config:
            trials = by_config[key]
            b100 = statistics.mean([t['best_at_100'] for t in trials])
            b200 = statistics.mean([t['best_at_200'] for t in trials])
            b300 = statistics.mean([t['best_at_300'] for t in trials])
            b600 = statistics.mean([t['best_at_600'] for t in trials])
            print(f"{n_exp:>4} | {closest_lr:>4.1f}x | {b100:>8.4f} | {b200:>8.4f} | "
                  f"{b300:>8.4f} | {b600:>8.4f}")

            # Find step where quality matches N=2 default LR at S=300
            step_scaling_data[n_exp] = {
                'lr_mult': closest_lr,
                'budgets': {100: b100, 200: b200, 300: b300, 600: b600},
            }

    # === 5. Monotonicity Check (KC1) ===
    print(f"\n{'='*80}")
    print("KC1: MONOTONIC RELATIONSHIP BETWEEN N AND OPTIMAL STEPS")
    print(f"{'='*80}")

    # At default LR, the optimal step budget should increase with N
    # "Optimal steps" = step budget at which quality stops improving meaningfully
    # Use the step budget that achieves within 0.5% of final (S=600) quality

    optimal_steps_per_n = {}
    for n_exp in N_VALUES:
        key = (n_exp, 1.0)  # default LR
        if key in by_config:
            trials = by_config[key]
            final_loss = statistics.mean([t['best_at_600'] for t in trials])
            target = final_loss * 1.005  # within 0.5% of final
            # Find earliest budget that achieves this
            for budget, budget_key in [(100, 'best_at_100'), (200, 'best_at_200'),
                                        (300, 'best_at_300'), (600, 'best_at_600')]:
                budget_loss = statistics.mean([t[budget_key] for t in trials])
                if budget_loss <= target:
                    optimal_steps_per_n[n_exp] = budget
                    break
            else:
                optimal_steps_per_n[n_exp] = 600

    print(f"\n  At default LR (1.0x), steps to reach within 0.5% of S=600 quality:")
    kc1_monotonic = True
    prev_steps = 0
    for n_exp in N_VALUES:
        if n_exp in optimal_steps_per_n:
            steps = optimal_steps_per_n[n_exp]
            predicted_steps = int(300 * n_exp / TOP_K / (N_VALUES[0] / TOP_K))
            is_monotonic = steps >= prev_steps
            if not is_monotonic and prev_steps > 0:
                kc1_monotonic = False
            print(f"    N={n_exp:>2}: {steps:>4} steps (predicted: {predicted_steps:>4})")
            prev_steps = steps

    # Also check: with N/k-scaled LR, do optimal steps DECREASE (compensated)?
    print(f"\n  With N/k-scaled LR, steps to reach N=2(default_LR, S=300) quality:")
    n2_default_key = (2, 1.0)
    n2_target = None
    if n2_default_key in by_config:
        n2_target = statistics.mean([t['best_at_300'] for t in by_config[n2_default_key]])
        print(f"    Target loss (N=2 at LR*1.0, S=300): {n2_target:.4f}")

    compensated_steps = {}
    if n2_target is not None:
        for n_exp in N_VALUES:
            predicted_lr = n_exp / TOP_K
            closest_lr = min(LR_MULTIPLIERS, key=lambda m: abs(m - predicted_lr))
            key = (n_exp, closest_lr)
            if key in by_config:
                trials = by_config[key]
                # Find which step budget reaches n2_target
                found_step = None
                for budget, budget_key in [(100, 'best_at_100'), (200, 'best_at_200'),
                                            (300, 'best_at_300'), (600, 'best_at_600')]:
                    budget_loss = statistics.mean([t[budget_key] for t in trials])
                    if budget_loss <= n2_target:
                        found_step = budget
                        break
                compensated_steps[n_exp] = found_step
                status = f"{found_step:>4} steps" if found_step else "not reached"
                print(f"    N={n_exp:>2} (LR*{closest_lr:.1f}x): {status}")

    # KC1 verdict
    n_values_with_steps = sorted(optimal_steps_per_n.keys())
    if len(n_values_with_steps) >= 3:
        steps_sequence = [optimal_steps_per_n[n] for n in n_values_with_steps]
        # Check if non-decreasing (monotonic)
        is_non_decreasing = all(steps_sequence[i] <= steps_sequence[i+1]
                                for i in range(len(steps_sequence)-1))
        # Also check: correlation between N and optimal steps
        r2_steps, r_steps, _ = compute_r_squared(
            list(n_values_with_steps),
            [float(optimal_steps_per_n[n]) for n in n_values_with_steps])
        print(f"\n  Monotonicity: {'YES' if is_non_decreasing else 'NO'}")
        print(f"  r^2(N, optimal_steps): {r2_steps:.4f}")
        kc1_pass = is_non_decreasing or r2_steps >= 0.5
    else:
        kc1_pass = False

    print(f"\n  KC1: {'PASS' if kc1_pass else 'KILL'} (monotonic: {kc1_monotonic})")

    # === 6. Quality Gap Closure (KC2) ===
    print(f"\n{'='*80}")
    print("KC2: LR SCALING CLOSES QUALITY GAP")
    print(f"{'='*80}")

    # Gap = quality(N=2, default_LR) - quality(N=8, default_LR)
    # Closure = 1 - [quality(N=2) - quality(N=8, best_scaled_LR)] / [quality(N=2) - quality(N=8, default_LR)]
    n2_default = None
    if (2, 1.0) in by_config:
        n2_default = statistics.mean([t['vs_joint_pct'] for t in by_config[(2, 1.0)]])

    kc2_closures = {}
    for n_exp in [4, 8, 16]:
        if n2_default is None:
            continue

        # Default LR quality
        default_key = (n_exp, 1.0)
        if default_key not in by_config:
            continue
        default_vj = statistics.mean([t['vs_joint_pct'] for t in by_config[default_key]])

        # Best scaled LR quality
        best_vj = float('inf')
        best_lr_m = 1.0
        for lr_mult in LR_MULTIPLIERS:
            key = (n_exp, lr_mult)
            if key in by_config:
                vj = statistics.mean([t['vs_joint_pct'] for t in by_config[key]])
                if abs(vj) < abs(best_vj):
                    best_vj = vj
                    best_lr_m = lr_mult

        # Gap and closure (all values are % above joint, so lower is better)
        gap_original = abs(default_vj) - abs(n2_default) if abs(default_vj) > abs(n2_default) else 0
        gap_scaled = abs(best_vj) - abs(n2_default) if abs(best_vj) > abs(n2_default) else 0
        closure = 1 - gap_scaled / gap_original if gap_original > 0.01 else 1.0

        kc2_closures[n_exp] = closure

        print(f"\n  N={n_exp}:")
        print(f"    N=2 (LR*1.0): {n2_default:+.2f}% vs joint")
        print(f"    N={n_exp} (LR*1.0): {default_vj:+.2f}% vs joint")
        print(f"    N={n_exp} (LR*{best_lr_m:.1f}): {best_vj:+.2f}% vs joint")
        print(f"    Gap closure: {closure:.1%}")

    mean_closure = statistics.mean(list(kc2_closures.values())) if kc2_closures else 0.0
    kc2_pass = mean_closure > 0.50  # >50% gap closure
    n8_closure = kc2_closures.get(8, 0.0)

    print(f"\n  Mean gap closure: {mean_closure:.1%}")
    print(f"  N=8 gap closure: {n8_closure:.1%}")
    print(f"  KC2 threshold: >50%")
    print(f"  KC2: {'PASS' if kc2_pass else 'KILL'}")

    # === 7. Scaling Law Fit ===
    print(f"\n{'='*80}")
    print("SCALING LAW FIT")
    print(f"{'='*80}")

    # Fit: LR_opt(N) = a * (N/k)^b
    # If b~1, the simple N/k scaling works
    # If b<1, sublinear (diminishing returns)
    # If b>1, superlinear (gradient attenuation compounds)
    opt_lr_per_n = [(n, best_per_n[n][0]) for n in N_VALUES]
    n_over_k = [n / TOP_K for n, _ in opt_lr_per_n]
    opt_lrs = [lr for _, (lr, _) in best_per_n.items()]

    # Log-log fit: log(LR_opt) = b * log(N/k) + log(a)
    if len(n_over_k) >= 3:
        log_nk = [math.log(x) for x in n_over_k]
        log_lr = [math.log(x) for x in opt_lrs]
        r2_scaling, r_scaling, _ = compute_r_squared(log_nk, log_lr)

        # Simple linear regression for slope b
        mean_x = statistics.mean(log_nk)
        mean_y = statistics.mean(log_lr)
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_nk, log_lr))
        var_x = sum((x - mean_x) ** 2 for x in log_nk)
        b_slope = cov_xy / var_x if var_x > 1e-10 else 0
        a_intercept = math.exp(mean_y - b_slope * mean_x) if abs(b_slope) < 10 else 1.0

        print(f"\n  Log-log fit: LR_opt = {a_intercept:.2f} * (N/k)^{b_slope:.2f}")
        print(f"  r^2 = {r2_scaling:.4f}")
        print(f"  If b~1.0: simple N/k scaling works")
        print(f"  If b~0.0: LR should not scale with N")
        print(f"  Actual b = {b_slope:.2f}")

        # Practical scaling law
        print(f"\n  Practical scaling law for contribution protocol:")
        for n in [2, 4, 8, 16, 32, 64]:
            predicted_lr_mult = a_intercept * (n / TOP_K) ** b_slope
            predicted_steps = int(300 * max(1, (n / TOP_K) ** max(0, b_slope)))
            print(f"    N={n:>3}: LR_mult = {predicted_lr_mult:>6.2f}x, "
                  f"steps ~ {predicted_steps:>4}")

    # === 8. Overall Verdict ===
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if kc1_pass and kc2_pass:
        print(f"\n  PROVEN: Calibration LR scales monotonically with N and closes quality gap")
        status = "proven"
    elif kc1_pass and not kc2_pass:
        print(f"\n  PARTIAL: Monotonic relationship exists but LR scaling doesn't close gap")
        status = "partial"
    elif not kc1_pass and kc2_pass:
        print(f"\n  PARTIAL: LR scaling helps quality but relationship isn't monotonic")
        status = "partial"
    else:
        print(f"\n  KILLED: No usable scaling law found")
        status = "killed"

    return {
        'kc1_pass': kc1_pass,
        'kc2_pass': kc2_pass,
        'best_per_n': best_per_n,
        'mean_closure': mean_closure,
        'status': status,
        'optimal_steps_per_n': optimal_steps_per_n,
        'kc2_closures': kc2_closures,
    }


# -- Multi-Seed Runner -----------------------------------------------------

def run_multiseed(seeds=(42, 123, 7)):
    """Run the full experiment across multiple seeds."""
    t0 = time.time()
    all_experiments = []

    for seed in seeds:
        result = run_experiment(seed=seed)
        all_experiments.append(result)

    analysis = analyze_results(all_experiments)
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return all_experiments, analysis


if __name__ == "__main__":
    run_multiseed()
