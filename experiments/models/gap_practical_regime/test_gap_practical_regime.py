"""Gap-as-Signal in Practical Cosine Regime (cos < 0.3).

Zooms into the practical regime where real LoRA adapters live.
The parent experiment showed r^2=0.74 across [0.0, 0.9], but the reviewer
identified that cos>=0.7 drives most of the signal. Quality differences
in [0.0, 0.3] were only 0.2pp (+2.1% to +2.3%).

This experiment:
1. Fine-grained cosine sweep: cos = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
2. Also includes cos=0.5 and cos=0.9 as anchors for comparison
3. 5 seeds for statistical power (small effect sizes need more replicates)
4. Measures within-regime r^2, effect sizes, and whether gap variation exceeds noise
5. Cohen's d for cos=0.0 vs cos=0.3 effect size

Kill criteria:
- quality difference between cos=0.0 and cos=0.3 is < 0.5pp (not meaningful)
- gap magnitude variation in [0.0, 0.3] is within noise (F-test or CV)
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
    RoutedDeltaGPT, calibrate_router,
)
from experiments.models.gap_as_signal.test_gap_as_signal import (
    flatten_deltas, unflatten_deltas, project_to_target_cosine,
    measure_function_space_gap, calibrate_router_tracked,
    compute_r_squared,
)


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3

# Fine-grained cosine sweep in the practical regime + anchors
PRACTICAL_COSINES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ANCHOR_COSINES = [0.50, 0.90]
ALL_COSINES = PRACTICAL_COSINES + ANCHOR_COSINES

# Calibration config (same as parent)
MAX_CAL_STEPS = 300
CAL_EVAL_EVERY = 5
CONVERGENCE_THRESHOLD = 0.005

# More seeds for statistical power
SEEDS = [42, 123, 7, 2024, 999]


# ── Effect Size Computation ───────────────────────────────────────────────

def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    s1, s2 = statistics.stdev(group1), statistics.stdev(group2)
    # Pooled standard deviation
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp < 1e-12:
        return float('inf') if abs(m1 - m2) > 1e-12 else 0.0
    return (m2 - m1) / sp


def coefficient_of_variation(values):
    """Compute coefficient of variation (std/mean)."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values)
    if abs(m) < 1e-12:
        return float('inf')
    return s / abs(m)


def noise_floor_bootstrap(all_trials_by_cos, n_bootstrap=1000, seed=42):
    """Estimate noise floor: within-seed variance at a fixed cosine level.

    Returns the mean within-cosine-level standard deviation of vs_joint_pct.
    If the between-cosine variation is within this noise floor, the gap
    signal is not informative.
    """
    rng = random.Random(seed)
    # Compute within-level std for each cosine level
    within_stds = []
    for cos_val, trials in all_trials_by_cos.items():
        if cos_val > 0.30:  # only practical regime
            continue
        values = [t['vs_joint_pct'] for t in trials]
        if len(values) >= 2:
            within_stds.append(statistics.stdev(values))

    if not within_stds:
        return 0.0
    return statistics.mean(within_stds)


# ── Single Cosine Trial ────────────────────────────────────────────────────

def run_single_trial(target_cos, base_model, deltas_a, deltas_b_original,
                     joint_model, train_a, train_b, val_a, val_b, joint_val,
                     joint_val_loss, V, seed=42):
    """Run one trial at a specific target cosine similarity.

    Reuses the parent experiment's infrastructure with identical protocol.
    """
    # Project expert B's deltas to achieve target cosine
    deltas_b_proj, actual_cos = project_to_target_cosine(
        deltas_a, deltas_b_original, target_cos
    )

    # Create task-arithmetic model (simple average of deltas)
    ta_deltas = []
    for m_idx in range(len(deltas_a)):
        l_idx, name, d_a = deltas_a[m_idx]
        _, _, d_b = deltas_b_proj[m_idx]
        ta_deltas.append((l_idx, name, (d_a + d_b) / 2))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)

    # Measure function-space gap: composed (task arithmetic) vs joint
    l2_gap, ce_gap, kl_gap, prob_l1 = measure_function_space_gap(
        ta_model, joint_model, joint_val
    )

    # Create routed model and calibrate
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    routed_model = RoutedDeltaGPT(base_copy, [deltas_a, deltas_b_proj], V, top_k=2)
    mx.eval(routed_model.parameters())

    # Measure gap for routed model BEFORE calibration
    _, ce_gap_pre, kl_gap_pre, _ = measure_function_space_gap(
        routed_model, joint_model, joint_val
    )

    # Calibrate with tracking
    loss_curve, steps_to_converge, final_val = calibrate_router_tracked(
        routed_model, train_a, train_b, joint_val,
        joint_val_loss, steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    # Measure gap AFTER calibration
    _, ce_gap_post, kl_gap_post, _ = measure_function_space_gap(
        routed_model, joint_model, joint_val
    )

    val_a_loss = evaluate(routed_model, val_a, BATCH_SIZE)
    val_b_loss = evaluate(routed_model, val_b, BATCH_SIZE)
    avg_val = (val_a_loss + val_b_loss) / 2

    # AUC of calibration curve
    if loss_curve:
        auc = sum(vl for _, vl in loss_curve) / len(loss_curve)
    else:
        auc = float('inf')

    return {
        'target_cos': target_cos,
        'actual_cos': actual_cos,
        'ce_gap_ta': ce_gap,
        'kl_gap_ta': kl_gap,
        'prob_l1_ta': prob_l1,
        'ce_gap_pre': ce_gap_pre,
        'kl_gap_pre': kl_gap_pre,
        'ce_gap_post': ce_gap_post,
        'kl_gap_post': kl_gap_post,
        'final_val_loss': final_val,
        'avg_domain_val': avg_val,
        'vs_joint_pct': (avg_val - joint_val_loss) / joint_val_loss * 100,
        'auc': auc,
        'steps_to_converge': steps_to_converge,
    }


# ── Full Experiment ────────────────────────────────────────────────────────

def run_experiment(seed=42, verbose=True):
    """Run the experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAP PRACTICAL REGIME (seed={seed})")
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

    # === 1. Joint training baseline ===
    if verbose:
        print("\n--- Joint training baseline ---")
    model_joint = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(model_joint.parameters())
    total_steps = 2 * FINETUNE_STEPS
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    j_a = evaluate(model_joint, val_a, BATCH_SIZE)
    j_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_val_loss = (j_a + j_b) / 2
    joint_on_joint = evaluate(model_joint, joint_val, BATCH_SIZE)
    if verbose:
        print(f"  Joint val loss: avg={joint_val_loss:.4f} (a_m={j_a:.4f}, n_z={j_b:.4f})")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)

    # === 3. Fine-tune LoRA experts ===
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
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
        lora_model.unfreeze()
        return lora_model

    lora_a = finetune_lora(train_a, val_a, "A (a-m)")
    lora_b = finetune_lora(train_b, val_b, "B (n-z)")

    # === 4. Extract deltas ===
    deltas_a = get_deltas(lora_a)
    deltas_b = get_deltas(lora_b)

    # Natural cosine
    flat_a = flatten_deltas(deltas_a)
    flat_b = flatten_deltas(deltas_b)
    natural_cos = (mx.sum(flat_a * flat_b) /
                   (mx.sqrt(mx.sum(flat_a**2)) * mx.sqrt(mx.sum(flat_b**2)) + 1e-12)).item()
    if verbose:
        print(f"\n  Natural cosine similarity: {natural_cos:.4f}")

    # === 5. Run trials at each target cosine ===
    if verbose:
        print(f"\n{'='*70}")
        print("RUNNING FINE-GRAINED COSINE SWEEP")
        print(f"{'='*70}")

    trials = []
    for target_cos in ALL_COSINES:
        if verbose:
            print(f"\n  --- cos={target_cos:.2f} ---", end="", flush=True)
        trial = run_single_trial(
            target_cos, base_model, deltas_a, deltas_b,
            model_joint, train_a, train_b, val_a, val_b, joint_val,
            joint_on_joint, V, seed=seed
        )
        if verbose:
            print(f" -> vs_joint={trial['vs_joint_pct']:+.2f}%, "
                  f"CE_gap={trial['ce_gap_ta']:.5f}, "
                  f"KL_gap={trial['kl_gap_ta']:.5f}")
        trials.append(trial)

    return {
        'seed': seed,
        'joint_val_loss': joint_val_loss,
        'joint_on_joint': joint_on_joint,
        'natural_cos': natural_cos,
        'trials': trials,
    }


# ── Analysis ──────────────────────────────────────────────────────────────

def analyze_results(all_experiments):
    """Analyze the practical regime with statistical rigor."""
    print(f"\n\n{'='*80}")
    print("GAP-AS-SIGNAL: PRACTICAL REGIME ANALYSIS")
    print(f"{'='*80}")

    # Aggregate trials across seeds
    by_cos = {}
    for exp in all_experiments:
        for trial in exp['trials']:
            cos = trial['target_cos']
            if cos not in by_cos:
                by_cos[cos] = []
            by_cos[cos].append(trial)

    # ── Summary Table ──
    print(f"\n{'Cos':>6} | {'CE Gap':>10} | {'KL Gap':>10} | {'Final VL':>10} | "
          f"{'vs Joint':>10} | {'Std(vJ)':>8} | {'N':>3}")
    print("-" * 75)

    practical_cos_list = []
    practical_quality_list = []
    practical_ce_gaps = []
    practical_kl_gaps = []
    all_cos_list = []
    all_quality_list = []
    all_ce_gaps = []

    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        ce_gaps = [t['ce_gap_ta'] for t in trials]
        kl_gaps = [t['kl_gap_ta'] for t in trials]
        final_vls = [t['final_val_loss'] for t in trials]
        vs_joints = [t['vs_joint_pct'] for t in trials]

        mean_ce = statistics.mean(ce_gaps)
        mean_kl = statistics.mean(kl_gaps)
        mean_vl = statistics.mean(final_vls)
        mean_vj = statistics.mean(vs_joints)
        std_vj = statistics.stdev(vs_joints) if len(vs_joints) >= 2 else 0.0

        print(f"{cos:>6.2f} | {mean_ce:>10.5f} | {mean_kl:>10.5f} | {mean_vl:>10.4f} | "
              f"{mean_vj:>+9.2f}% | {std_vj:>8.3f} | {len(trials):>3}")

        for t in trials:
            all_cos_list.append(cos)
            all_quality_list.append(t['vs_joint_pct'])
            all_ce_gaps.append(t['ce_gap_ta'])
            if cos <= 0.30:
                practical_cos_list.append(cos)
                practical_quality_list.append(t['vs_joint_pct'])
                practical_ce_gaps.append(t['ce_gap_ta'])
                practical_kl_gaps.append(t['kl_gap_ta'])

    # ── Kill Criterion 1: Quality difference cos=0.0 vs cos=0.3 ──
    print(f"\n{'='*80}")
    print("KILL CRITERION 1: Quality difference cos=0.0 vs cos=0.3 >= 0.5pp")
    print(f"{'='*80}")

    q_00 = [t['vs_joint_pct'] for t in by_cos.get(0.00, [])]
    q_30 = [t['vs_joint_pct'] for t in by_cos.get(0.30, [])]

    if q_00 and q_30:
        mean_00 = statistics.mean(q_00)
        mean_30 = statistics.mean(q_30)
        diff_pp = mean_30 - mean_00
        d = cohens_d(q_00, q_30)

        print(f"  cos=0.00 quality: {mean_00:+.3f}% vs joint (std={statistics.stdev(q_00):.3f})")
        print(f"  cos=0.30 quality: {mean_30:+.3f}% vs joint (std={statistics.stdev(q_30):.3f})")
        print(f"  Difference: {diff_pp:+.3f}pp")
        print(f"  Cohen's d: {d:.3f} ({'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})")

        kc1_pass = diff_pp > 0.5
        if kc1_pass:
            print(f"  PASS: difference {diff_pp:+.3f}pp > 0.5pp threshold")
        else:
            print(f"  KILL: difference {diff_pp:+.3f}pp < 0.5pp threshold")
    else:
        kc1_pass = False
        diff_pp = 0.0
        d = 0.0
        print("  INSUFFICIENT DATA")

    # ── Kill Criterion 2: Gap magnitude variation exceeds noise ──
    print(f"\n{'='*80}")
    print("KILL CRITERION 2: Gap magnitude variation in [0.0, 0.3] exceeds noise")
    print(f"{'='*80}")

    # Compute between-level variation of gap magnitudes
    practical_cos_means_ce = {}
    practical_cos_means_q = {}
    for cos in PRACTICAL_COSINES:
        if cos in by_cos:
            practical_cos_means_ce[cos] = statistics.mean([t['ce_gap_ta'] for t in by_cos[cos]])
            practical_cos_means_q[cos] = statistics.mean([t['vs_joint_pct'] for t in by_cos[cos]])

    if len(practical_cos_means_ce) >= 3:
        between_ce_values = list(practical_cos_means_ce.values())
        between_q_values = list(practical_cos_means_q.values())

        # Between-level range and CV
        ce_range = max(between_ce_values) - min(between_ce_values)
        ce_cv = coefficient_of_variation(between_ce_values)
        q_range = max(between_q_values) - min(between_q_values)

        # Noise floor: average within-level standard deviation
        noise_floor = noise_floor_bootstrap(by_cos)

        print(f"  CE gap range across [0.0, 0.3]: {ce_range:.6f}")
        print(f"  CE gap CV: {ce_cv:.3f}")
        print(f"  Quality range across [0.0, 0.3]: {q_range:.3f}pp")
        print(f"  Noise floor (mean within-level std): {noise_floor:.3f}pp")

        # Signal-to-noise: between-level range vs within-level noise
        snr = q_range / noise_floor if noise_floor > 1e-6 else float('inf')
        print(f"  Signal-to-noise ratio (range/noise): {snr:.2f}")

        # F-test analog: is between-group variance significantly larger than within-group?
        # Between-level variance of quality means
        between_var = statistics.variance(between_q_values) if len(between_q_values) >= 2 else 0.0
        within_var = noise_floor ** 2 if noise_floor > 0 else 1e-12
        f_ratio = between_var / within_var if within_var > 1e-12 else float('inf')
        print(f"  F-ratio (between/within variance): {f_ratio:.2f}")

        kc2_pass = snr > 1.0 and q_range > noise_floor
        if kc2_pass:
            print(f"  PASS: quality range ({q_range:.3f}pp) > noise floor ({noise_floor:.3f}pp)")
        else:
            print(f"  KILL: quality range ({q_range:.3f}pp) within noise ({noise_floor:.3f}pp)")
    else:
        kc2_pass = False
        snr = 0.0
        f_ratio = 0.0
        ce_range = 0.0
        q_range = 0.0
        noise_floor = 0.0
        print("  INSUFFICIENT DATA")

    # ── Correlation Analysis ──
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")

    # Within practical regime only
    r2_practical, r_practical, slope_practical = compute_r_squared(
        practical_cos_list, practical_quality_list
    )
    r2_prac_ce, r_prac_ce, _ = compute_r_squared(
        practical_cos_list, practical_ce_gaps
    )

    print(f"\n  Within practical regime [0.0, 0.3] ({len(practical_cos_list)} points):")
    print(f"    Cosine vs Quality: r={r_practical:.4f}, r^2={r2_practical:.4f}")
    print(f"    Cosine vs CE Gap:  r={r_prac_ce:.4f}, r^2={r2_prac_ce:.4f}")
    print(f"    Slope (quality/cos): {slope_practical:.3f} pp per 0.1 cos")

    # Full range for comparison
    r2_full, r_full, _ = compute_r_squared(all_cos_list, all_quality_list)
    r2_full_ce, r_full_ce, _ = compute_r_squared(all_cos_list, all_ce_gaps)
    print(f"\n  Full range [0.0, 0.9] ({len(all_cos_list)} points):")
    print(f"    Cosine vs Quality: r={r_full:.4f}, r^2={r2_full:.4f}")
    print(f"    Cosine vs CE Gap:  r={r_full_ce:.4f}, r^2={r2_full_ce:.4f}")

    # CE gap vs quality within practical regime
    r2_gap_q_prac, r_gap_q_prac, _ = compute_r_squared(
        practical_ce_gaps, practical_quality_list
    )
    print(f"\n  Gap as predictor within [0.0, 0.3]:")
    print(f"    CE Gap vs Quality: r={r_gap_q_prac:.4f}, r^2={r2_gap_q_prac:.4f}")

    # ── Monotonicity Check ──
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK (practical regime)")
    print(f"{'='*80}")

    if len(practical_cos_means_q) >= 2:
        sorted_cos = sorted(practical_cos_means_q.keys())
        monotonic_steps = 0
        total_steps = len(sorted_cos) - 1
        for i in range(total_steps):
            if practical_cos_means_q[sorted_cos[i+1]] >= practical_cos_means_q[sorted_cos[i]]:
                monotonic_steps += 1
                direction = "+"
            else:
                direction = "-"
            print(f"  cos {sorted_cos[i]:.2f} -> {sorted_cos[i+1]:.2f}: "
                  f"{practical_cos_means_q[sorted_cos[i]]:+.3f}% -> "
                  f"{practical_cos_means_q[sorted_cos[i+1]]:+.3f}% [{direction}]")

        mono_frac = monotonic_steps / total_steps if total_steps > 0 else 0.0
        print(f"\n  Monotonic transitions: {monotonic_steps}/{total_steps} ({mono_frac:.0%})")
        print(f"  (100% = perfectly monotonic increasing quality gap with cosine)")

    # ── Anchor Comparison ──
    print(f"\n{'='*80}")
    print("ANCHOR COMPARISON: practical vs high-cosine")
    print(f"{'='*80}")

    for anchor_cos in ANCHOR_COSINES:
        if anchor_cos in by_cos:
            anchor_q = statistics.mean([t['vs_joint_pct'] for t in by_cos[anchor_cos]])
            anchor_ce = statistics.mean([t['ce_gap_ta'] for t in by_cos[anchor_cos]])
            print(f"  cos={anchor_cos:.1f}: quality={anchor_q:+.2f}%, CE_gap={anchor_ce:.5f}")

    if q_00:
        print(f"  cos=0.0: quality={mean_00:+.2f}%, CE_gap={practical_cos_means_ce.get(0.0, 0):.5f}")
        if 0.50 in by_cos:
            q_50 = statistics.mean([t['vs_joint_pct'] for t in by_cos[0.50]])
            print(f"\n  Range [0.0, 0.3] quality spread: {q_range:.3f}pp")
            print(f"  Range [0.0, 0.5] quality spread: {q_50 - mean_00:.3f}pp")
            if 0.90 in by_cos:
                q_90 = statistics.mean([t['vs_joint_pct'] for t in by_cos[0.90]])
                print(f"  Range [0.0, 0.9] quality spread: {q_90 - mean_00:.3f}pp")
                practical_fraction = q_range / (q_90 - mean_00) if (q_90 - mean_00) > 1e-6 else 0.0
                print(f"  Practical regime accounts for {practical_fraction:.1%} of total quality range")

    # ── Overall Verdict ──
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if kc1_pass and kc2_pass:
        print(f"\n  HYPOTHESIS PROVEN: Both kill criteria pass")
        print(f"    KC1: quality difference cos=0.0 vs cos=0.3 = {diff_pp:+.3f}pp (>0.5pp)")
        print(f"    KC2: gap variation exceeds noise (SNR={snr:.2f})")
        print(f"    Gap-as-signal IS informative in the practical regime")
        verdict = "PROVEN"
    elif not kc1_pass and not kc2_pass:
        print(f"\n  HYPOTHESIS KILLED: Both kill criteria fail")
        print(f"    KC1: quality difference = {diff_pp:+.3f}pp (<0.5pp)")
        print(f"    KC2: gap variation within noise (SNR={snr:.2f})")
        print(f"    Gap-as-signal is NOT informative in the practical regime")
        print(f"    Implication: the diagnostic only rejects obviously correlated adapters (cos>0.3)")
        verdict = "KILLED"
    else:
        print(f"\n  PARTIAL: 1/2 kill criteria fail")
        print(f"    KC1 {'PASS' if kc1_pass else 'FAIL'}: {diff_pp:+.3f}pp vs 0.5pp threshold")
        print(f"    KC2 {'PASS' if kc2_pass else 'FAIL'}: SNR={snr:.2f}")
        verdict = "PARTIAL"

    print(f"\n  Within-regime r^2 = {r2_practical:.4f} (parent full-range r^2 = 0.74)")
    print(f"  Cohen's d (cos=0.0 vs 0.3) = {d:.3f}")

    return {
        'verdict': verdict,
        'diff_pp': diff_pp,
        'cohens_d': d,
        'snr': snr,
        'f_ratio': f_ratio,
        'noise_floor': noise_floor,
        'q_range': q_range,
        'ce_range': ce_range,
        'r2_practical': r2_practical,
        'r2_full': r2_full,
        'r2_gap_quality_practical': r2_gap_q_prac,
        'kc1_pass': kc1_pass,
        'kc2_pass': kc2_pass,
        'practical_cos_means_q': practical_cos_means_q,
        'practical_cos_means_ce': practical_cos_means_ce,
        'by_cos': by_cos,
    }


# ── Multi-Seed Runner ─────────────────────────────────────────────────────

def run_multiseed(seeds=SEEDS):
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
