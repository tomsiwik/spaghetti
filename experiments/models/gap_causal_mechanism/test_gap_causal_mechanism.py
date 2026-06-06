"""Gap Causal Mechanism: does gap magnitude CAUSE larger router gradients?

Follow-up to gap_as_signal (proven, r^2=0.74). The parent showed correlation
between gap magnitude and calibration quality. This experiment establishes the
CAUSAL MECHANISM: larger gap -> larger router gradient norms -> faster learning.

Protocol:
1. Reuse gap_as_signal infrastructure (shared base, LoRA experts, controlled
   cosine projection)
2. During calibration, extract per-step router gradient norms
3. Correlate: gap_magnitude vs mean_router_gradient_magnitude
4. Kill criteria:
   - r^2(gap, gradient_magnitude) < 0.3
   - equal gradient magnitudes at cos=0.0 and cos=0.9

Key insight: In the RoutedDeltaGPT forward pass, the router produces softmax
weights that multiply expert outputs. The gradient of the loss w.r.t. router
weights is proportional to the DISCRIMINABILITY of expert outputs: when experts
produce different outputs (high gap), the gradient pushes the router to pick
the better one. When experts produce similar outputs (low gap), gradients vanish.
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
    measure_function_space_gap, compute_r_squared,
)


# -- Config -----------------------------------------------------------------

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3

TARGET_COSINES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

MAX_CAL_STEPS = 300
CAL_EVAL_EVERY = 5


# -- Router Gradient Measurement -------------------------------------------

def extract_router_gradient_norms(grads, model):
    """Extract gradient norms for all router parameters.

    Returns:
        total_norm: L2 norm across all router weight gradients
        per_layer_norms: list of per-layer router gradient norms
    """
    per_layer_norms = []

    # Navigate the gradient tree to find router gradients
    # The model structure has model.routers[i].weight
    grad_tree = grads
    if hasattr(grad_tree, '__getitem__'):
        # grads is a nested dict matching model parameter structure
        pass

    # Direct approach: iterate over model.routers and find matching gradients
    # MLX returns gradients as a nested structure matching the model
    flat_grads = dict(nn.utils.tree_flatten(grads))

    total_sq = 0.0
    for i in range(len(model.routers)):
        key = f"routers.{i}.weight"
        if key in flat_grads:
            g = flat_grads[key]
            norm_sq = mx.sum(g * g).item()
            per_layer_norms.append(math.sqrt(norm_sq))
            total_sq += norm_sq
        else:
            per_layer_norms.append(0.0)

    total_norm = math.sqrt(total_sq)
    return total_norm, per_layer_norms


def calibrate_with_gradient_tracking(model, train_ds_a, train_ds_b, val_ds,
                                      joint_val_loss, steps=300, lr=3e-3, seed=42):
    """Calibrate router and record per-step gradient norms.

    This is the key measurement: we record the L2 norm of the router weight
    gradients at EVERY step, not just loss values.

    Returns:
        loss_curve: list of (step, val_loss)
        grad_norms: list of (step, total_grad_norm, per_layer_norms)
        final_val_loss: final validation loss
    """
    model.freeze()
    for router in model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    loss_curve = []
    grad_norms = []

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)

        # Extract router gradient norms BEFORE optimizer step
        total_norm, per_layer = extract_router_gradient_norms(grads, model)
        grad_norms.append((step, total_norm, per_layer))

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % CAL_EVAL_EVERY == 0 or step == 1:
            val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=5)
            loss_curve.append((step, val_loss))

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)
    model.unfreeze()

    return loss_curve, grad_norms, final_val_loss


# -- Single Cosine Trial ---------------------------------------------------

def run_single_trial(target_cos, base_model, deltas_a, deltas_b_original,
                     joint_model, train_a, train_b, val_a, val_b, joint_val,
                     joint_val_loss, V, seed=42):
    """Run one trial at a specific cosine, recording gradient norms."""
    print(f"\n  --- Target cosine = {target_cos:.1f} ---")

    # Project expert B to target cosine
    deltas_b_proj, actual_cos = project_to_target_cosine(
        deltas_a, deltas_b_original, target_cos
    )
    print(f"    Achieved cosine: {actual_cos:.4f}")

    # Create task-arithmetic model
    ta_deltas = []
    for m_idx in range(len(deltas_a)):
        l_idx, name, d_a = deltas_a[m_idx]
        _, _, d_b = deltas_b_proj[m_idx]
        ta_deltas.append((l_idx, name, (d_a + d_b) / 2))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)

    # Measure function-space gap
    l2_gap, ce_gap, kl_gap, prob_l1 = measure_function_space_gap(
        ta_model, joint_model, joint_val
    )
    print(f"    Function-space gap: CE={ce_gap:.4f}, KL={kl_gap:.4f}")

    # Create routed model
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    routed_model = RoutedDeltaGPT(base_copy, [deltas_a, deltas_b_proj], V, top_k=2)
    mx.eval(routed_model.parameters())

    # Calibrate WITH gradient tracking
    loss_curve, grad_norms, final_val = calibrate_with_gradient_tracking(
        routed_model, train_a, train_b, joint_val,
        joint_val_loss, steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    # Compute gradient statistics
    all_total_norms = [gn[1] for gn in grad_norms]
    mean_grad_norm = statistics.mean(all_total_norms) if all_total_norms else 0.0
    max_grad_norm = max(all_total_norms) if all_total_norms else 0.0

    # Early gradient norms (first 50 steps) -- most informative for "signal strength"
    early_norms = [gn[1] for gn in grad_norms if gn[0] <= 50]
    mean_early_grad = statistics.mean(early_norms) if early_norms else 0.0

    # Late gradient norms (last 50 steps) -- should decay as router converges
    late_norms = [gn[1] for gn in grad_norms if gn[0] > 250]
    mean_late_grad = statistics.mean(late_norms) if late_norms else 0.0

    # Per-layer gradient norms (averaged over all steps)
    n_layers = len(grad_norms[0][2]) if grad_norms else 0
    per_layer_means = []
    for l in range(n_layers):
        layer_norms = [gn[2][l] for gn in grad_norms]
        per_layer_means.append(statistics.mean(layer_norms))

    # Domain-specific val losses
    val_a_loss = evaluate(routed_model, val_a, BATCH_SIZE)
    val_b_loss = evaluate(routed_model, val_b, BATCH_SIZE)
    avg_val = (val_a_loss + val_b_loss) / 2
    vs_joint = (avg_val - joint_val_loss) / joint_val_loss * 100

    print(f"    Mean grad norm: {mean_grad_norm:.6f} (early: {mean_early_grad:.6f}, late: {mean_late_grad:.6f})")
    print(f"    Final val: {final_val:.4f} (vs joint: {vs_joint:+.1f}%)")
    print(f"    Per-layer grad norms: {[f'{n:.6f}' for n in per_layer_means]}")

    return {
        'target_cos': target_cos,
        'actual_cos': actual_cos,
        'ce_gap': ce_gap,
        'kl_gap': kl_gap,
        'prob_l1': prob_l1,
        'mean_grad_norm': mean_grad_norm,
        'max_grad_norm': max_grad_norm,
        'mean_early_grad': mean_early_grad,
        'mean_late_grad': mean_late_grad,
        'per_layer_grad_means': per_layer_means,
        'final_val_loss': final_val,
        'avg_domain_val': avg_val,
        'vs_joint_pct': vs_joint,
        'all_grad_norms': all_total_norms,  # full trajectory
        'loss_curve': loss_curve,
    }


# -- Full Experiment -------------------------------------------------------

def run_experiment(seed=42, verbose=True):
    """Run the causal mechanism experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAP CAUSAL MECHANISM EXPERIMENT (seed={seed})")
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
    if verbose:
        print(f"  Joint val loss: avg={joint_val_loss:.4f}")

    joint_on_joint = evaluate(model_joint, joint_val, BATCH_SIZE)

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

    natural_cos = (mx.sum(flatten_deltas(deltas_a) * flatten_deltas(deltas_b)) /
                   (mx.sqrt(mx.sum(flatten_deltas(deltas_a)**2)) *
                    mx.sqrt(mx.sum(flatten_deltas(deltas_b)**2)) + 1e-12)).item()
    if verbose:
        print(f"\n  Natural cosine similarity: {natural_cos:.4f}")

    # === 5. Run trials at each target cosine ===
    if verbose:
        print(f"\n{'='*70}")
        print("COSINE SWEEP WITH GRADIENT TRACKING")
        print(f"{'='*70}")

    trials = []
    for target_cos in TARGET_COSINES:
        trial = run_single_trial(
            target_cos, base_model, deltas_a, deltas_b,
            model_joint, train_a, train_b, val_a, val_b, joint_val,
            joint_on_joint, V, seed=seed
        )
        trials.append(trial)

    return {
        'seed': seed,
        'joint_val_loss': joint_val_loss,
        'joint_on_joint': joint_on_joint,
        'natural_cos': natural_cos,
        'trials': trials,
    }


# -- Analysis ---------------------------------------------------------------

def analyze_results(all_experiments):
    """Analyze causal mechanism: gap -> gradient magnitude -> quality."""
    print(f"\n\n{'='*80}")
    print("GAP CAUSAL MECHANISM ANALYSIS")
    print(f"{'='*80}")

    # Aggregate trials across seeds
    by_cos = {}
    for exp in all_experiments:
        for trial in exp['trials']:
            cos = trial['target_cos']
            if cos not in by_cos:
                by_cos[cos] = []
            by_cos[cos].append(trial)

    # Summary table
    print(f"\n{'Cos':>5} | {'CE Gap':>8} | {'MeanGrad':>10} | {'EarlyGrad':>10} | "
          f"{'LateGrad':>10} | {'Final VL':>9} | {'vs Joint':>9}")
    print("-" * 85)

    cos_list = []
    ce_gaps = []
    mean_grads = []
    early_grads = []
    late_grads = []
    vs_joints = []
    kl_gaps = []

    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        m_ce = statistics.mean([t['ce_gap'] for t in trials])
        m_grad = statistics.mean([t['mean_grad_norm'] for t in trials])
        m_early = statistics.mean([t['mean_early_grad'] for t in trials])
        m_late = statistics.mean([t['mean_late_grad'] for t in trials])
        m_vl = statistics.mean([t['final_val_loss'] for t in trials])
        m_vj = statistics.mean([t['vs_joint_pct'] for t in trials])

        print(f"{cos:>5.1f} | {m_ce:>8.4f} | {m_grad:>10.6f} | {m_early:>10.6f} | "
              f"{m_late:>10.6f} | {m_vl:>9.4f} | {m_vj:>+8.1f}%")

        for t in trials:
            cos_list.append(cos)
            ce_gaps.append(t['ce_gap'])
            mean_grads.append(t['mean_grad_norm'])
            early_grads.append(t['mean_early_grad'])
            late_grads.append(t['mean_late_grad'])
            vs_joints.append(t['vs_joint_pct'])
            kl_gaps.append(t['kl_gap'])

    # === Correlation Analysis ===
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS: GAP -> GRADIENT -> QUALITY")
    print(f"{'='*80}")

    # Chain: gap -> gradient magnitude (the causal link)
    r2_ce_grad, r_ce_grad, slope_ce_grad = compute_r_squared(ce_gaps, mean_grads)
    print(f"\n1. CE Gap vs Mean Router Gradient Norm:")
    print(f"   r = {r_ce_grad:.4f}, r^2 = {r2_ce_grad:.4f}, slope = {slope_ce_grad:.6f}")
    print(f"   Direction: {'positive (expected: larger gap -> larger gradients)' if r_ce_grad > 0 else 'NEGATIVE (unexpected)'}")

    r2_kl_grad, r_kl_grad, _ = compute_r_squared(kl_gaps, mean_grads)
    print(f"\n2. KL Gap vs Mean Router Gradient Norm:")
    print(f"   r = {r_kl_grad:.4f}, r^2 = {r2_kl_grad:.4f}")

    # Chain: gradient magnitude -> quality (the consequence)
    r2_grad_quality, r_grad_quality, _ = compute_r_squared(mean_grads, vs_joints)
    print(f"\n3. Mean Router Gradient Norm vs Final Quality (% above joint):")
    print(f"   r = {r_grad_quality:.4f}, r^2 = {r2_grad_quality:.4f}")

    # Early gradients (most diagnostic of initial signal strength)
    r2_ce_early, r_ce_early, _ = compute_r_squared(ce_gaps, early_grads)
    print(f"\n4. CE Gap vs Early Gradient Norm (first 50 steps):")
    print(f"   r = {r_ce_early:.4f}, r^2 = {r2_ce_early:.4f}")

    r2_early_quality, r_early_quality, _ = compute_r_squared(early_grads, vs_joints)
    print(f"\n5. Early Gradient Norm vs Final Quality:")
    print(f"   r = {r_early_quality:.4f}, r^2 = {r2_early_quality:.4f}")

    # Cosine -> gradient (direct)
    r2_cos_grad, r_cos_grad, _ = compute_r_squared(cos_list, mean_grads)
    print(f"\n6. Cosine vs Mean Router Gradient Norm:")
    print(f"   r = {r_cos_grad:.4f}, r^2 = {r2_cos_grad:.4f}")

    # Gradient decay ratio (early/late) -- should be higher for low cosine
    # because stronger initial signal + faster convergence
    decay_ratios = []
    for e, l in zip(early_grads, late_grads):
        if l > 1e-10:
            decay_ratios.append(e / l)
        else:
            decay_ratios.append(float('inf'))
    finite_decay = [(c, d) for c, d in zip(cos_list, decay_ratios) if d < float('inf')]
    if finite_decay:
        r2_cos_decay, r_cos_decay, _ = compute_r_squared(
            [x[0] for x in finite_decay],
            [x[1] for x in finite_decay]
        )
        print(f"\n7. Cosine vs Gradient Decay Ratio (early/late):")
        print(f"   r = {r_cos_decay:.4f}, r^2 = {r2_cos_decay:.4f}")

    # Per-layer analysis
    print(f"\n{'='*80}")
    print("PER-LAYER GRADIENT ANALYSIS")
    print(f"{'='*80}")

    n_layers = len(by_cos[0.0][0]['per_layer_grad_means']) if by_cos.get(0.0) else 0
    for l in range(n_layers):
        layer_grads = []
        for exp in all_experiments:
            for trial in exp['trials']:
                if l < len(trial['per_layer_grad_means']):
                    layer_grads.append((trial['ce_gap'], trial['per_layer_grad_means'][l]))

        if layer_grads:
            ces = [x[0] for x in layer_grads]
            lgs = [x[1] for x in layer_grads]
            r2_l, r_l, _ = compute_r_squared(ces, lgs)
            print(f"  Layer {l}: CE_gap vs grad_norm: r={r_l:.4f}, r^2={r2_l:.4f}")

    # === Kill Criteria ===
    print(f"\n{'='*80}")
    print("KILL CRITERIA")
    print(f"{'='*80}")

    # KC1: r^2(gap, gradient_magnitude) >= 0.3
    # Use the best of CE gap and KL gap correlations with gradient norms
    best_gap_grad_r2 = max(r2_ce_grad, r2_kl_grad, r2_ce_early)
    gap_grad_map = {
        r2_ce_grad: "CE_gap_vs_mean_grad",
        r2_kl_grad: "KL_gap_vs_mean_grad",
        r2_ce_early: "CE_gap_vs_early_grad",
    }
    best_gap_grad_metric = gap_grad_map[best_gap_grad_r2]

    print(f"\n1. Gap magnitude correlates with router gradient magnitude?")
    print(f"   Best r^2 = {best_gap_grad_r2:.4f} (metric: {best_gap_grad_metric})")
    if best_gap_grad_r2 >= 0.3:
        print(f"   PASS: r^2 = {best_gap_grad_r2:.4f} >= 0.3")
        kc1_pass = True
    else:
        print(f"   KILL: r^2 = {best_gap_grad_r2:.4f} < 0.3")
        kc1_pass = False

    # KC2: gradient magnitudes differ between cos=0.0 and cos=0.9
    grad_cos0 = [t['mean_grad_norm'] for t in by_cos.get(0.0, [])]
    grad_cos9 = [t['mean_grad_norm'] for t in by_cos.get(0.9, [])]

    if grad_cos0 and grad_cos9:
        mean_g0 = statistics.mean(grad_cos0)
        mean_g9 = statistics.mean(grad_cos9)
        # Also check early gradients
        early_g0 = statistics.mean([t['mean_early_grad'] for t in by_cos[0.0]])
        early_g9 = statistics.mean([t['mean_early_grad'] for t in by_cos[0.9]])

        ratio = mean_g0 / mean_g9 if mean_g9 > 1e-10 else float('inf')
        early_ratio = early_g0 / early_g9 if early_g9 > 1e-10 else float('inf')

        print(f"\n2. Gradient magnitudes differ between cos=0.0 and cos=0.9?")
        print(f"   cos=0.0 mean grad: {mean_g0:.6f} (early: {early_g0:.6f})")
        print(f"   cos=0.9 mean grad: {mean_g9:.6f} (early: {early_g9:.6f})")
        print(f"   Ratio (0.0/0.9): {ratio:.3f} (early: {early_ratio:.3f})")

        # Kill if gradient magnitudes are equal (within 10%)
        if abs(ratio - 1.0) > 0.10 or abs(early_ratio - 1.0) > 0.10:
            print(f"   PASS: gradient magnitudes differ (ratio = {ratio:.3f})")
            kc2_pass = True
        else:
            print(f"   KILL: gradient magnitudes are equal (ratio = {ratio:.3f})")
            kc2_pass = False
    else:
        print(f"\n2. Cannot evaluate KC2: missing cos=0.0 or cos=0.9 data")
        kc2_pass = False

    # === Monotonicity check ===
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK")
    print(f"{'='*80}")

    cos_sorted = sorted(by_cos.keys())
    grad_means_by_cos = [statistics.mean([t['mean_grad_norm'] for t in by_cos[c]]) for c in cos_sorted]
    early_means_by_cos = [statistics.mean([t['mean_early_grad'] for t in by_cos[c]]) for c in cos_sorted]

    # Check if gradient norms are monotonically related to cosine
    # We expect: gradient norms DECREASE with increasing cosine (less gap -> less gradient)
    # OR gradient norms INCREASE (if the interpretation is that higher gap = harder problem = larger gradients)
    # The hypothesis predicts: orthogonal (cos=0) -> larger gap -> BUT the gap is between COMPOSED
    # and JOINT, not between the experts. Need to check empirically.

    monotonic_decreasing = all(grad_means_by_cos[i] >= grad_means_by_cos[i+1]
                               for i in range(len(grad_means_by_cos)-1))
    monotonic_increasing = all(grad_means_by_cos[i] <= grad_means_by_cos[i+1]
                               for i in range(len(grad_means_by_cos)-1))

    print(f"  Mean grad norms by cosine: {[f'{g:.6f}' for g in grad_means_by_cos]}")
    print(f"  Early grad norms by cosine: {[f'{g:.6f}' for g in early_means_by_cos]}")
    print(f"  Monotonically decreasing: {monotonic_decreasing}")
    print(f"  Monotonically increasing: {monotonic_increasing}")

    if monotonic_decreasing:
        print(f"  Pattern: HIGHER cosine -> LOWER gradient norm")
        print(f"  Interpretation: more correlated experts -> smaller gap -> weaker gradient signal")
    elif monotonic_increasing:
        print(f"  Pattern: HIGHER cosine -> HIGHER gradient norm")
        print(f"  Interpretation: more correlated experts -> larger gap -> stronger gradient")
        print(f"  BUT the router has a HARDER problem, not a stronger signal")
    else:
        print(f"  Pattern: NON-MONOTONIC")

    # === Full causal chain test ===
    print(f"\n{'='*80}")
    print("CAUSAL CHAIN: gap -> gradient -> quality")
    print(f"{'='*80}")

    # The full causal chain: cos -> gap -> gradient -> quality
    # If gap causes gradients and gradients cause quality, then:
    # r^2(gap, gradient) and r^2(gradient, quality) should both be > 0.3
    # AND their signs should be consistent

    chain_holds = (best_gap_grad_r2 >= 0.3 and r2_grad_quality >= 0.3)
    print(f"\n  Gap -> Gradient: r^2 = {best_gap_grad_r2:.4f} ({'PASS' if best_gap_grad_r2 >= 0.3 else 'FAIL'})")
    print(f"  Gradient -> Quality: r^2 = {r2_grad_quality:.4f} ({'PASS' if r2_grad_quality >= 0.3 else 'FAIL'})")
    print(f"  Full chain: {'HOLDS' if chain_holds else 'BROKEN'}")

    # === Overall Verdict ===
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if kc1_pass and kc2_pass:
        print(f"\n  HYPOTHESIS PROVEN: Both kill criteria pass")
        print(f"  Gap magnitude causally drives router gradient magnitude")
        status = "proven"
    elif not kc1_pass and not kc2_pass:
        print(f"\n  HYPOTHESIS KILLED: Both kill criteria fail")
        status = "killed"
    else:
        print(f"\n  PARTIAL: KC1={'PASS' if kc1_pass else 'FAIL'}, KC2={'PASS' if kc2_pass else 'FAIL'}")
        status = "partial"

    return {
        'r2_ce_grad': r2_ce_grad,
        'r2_kl_grad': r2_kl_grad,
        'r2_ce_early_grad': r2_ce_early,
        'r2_grad_quality': r2_grad_quality,
        'r2_cos_grad': r2_cos_grad,
        'best_gap_grad_r2': best_gap_grad_r2,
        'best_gap_grad_metric': best_gap_grad_metric,
        'kc1_pass': kc1_pass,
        'kc2_pass': kc2_pass,
        'status': status,
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
