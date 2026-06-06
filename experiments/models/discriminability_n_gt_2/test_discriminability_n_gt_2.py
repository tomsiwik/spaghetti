"""Discriminability at N>2 with top-k selection.

Does expert discriminability predict router gradient magnitude when the router
must BOTH select AND mix experts (N=8, top_k=2)?

Parent experiment (gap_causal_mechanism) proved at N=2, top_k=2 (mixing only):
  - Discriminability drives gradient magnitude (r^2=0.63 on mean curve)
  - Phase transition at cos~0.5-0.7
  - 15.5x gradient ratio between cos=0.0 and cos=0.9

This experiment tests whether the mechanism generalizes to N=8, top_k=2,
where the router must:
  1. SELECT which 2 of 8 experts to use (hard routing decision)
  2. MIX the selected experts' outputs (soft weight assignment)

Key measurement: mean pairwise expert discriminability across ALL 8 experts
vs total router gradient norm during calibration.

Kill criteria:
  KC1: r^2(discriminability, gradient) < 0.3 at N=8, top_k=2
  KC2: selection gradients behave qualitatively differently from mixing gradients
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
from experiments.models.gap_causal_mechanism.test_gap_causal_mechanism import (
    extract_router_gradient_norms,
)


# -- Config -----------------------------------------------------------------

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3

# N=8 experts, top_k=2 (the selection+mixing regime)
N_EXPERTS = 8
TOP_K = 2

# Cosine sweep: same levels as parent for comparison
TARGET_COSINES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

MAX_CAL_STEPS = 300
CAL_EVAL_EVERY = 5


# -- Multi-Expert Generation ------------------------------------------------

def generate_n_experts_at_cosine(deltas_a, deltas_b_original, target_cos, n_experts):
    """Generate N experts with controlled mean pairwise cosine similarity.

    Strategy: Generate N experts as random linear combinations of two orthogonal
    basis vectors (a and perp(a,b)), each with the target cosine to expert A,
    but mutually diverse by rotating in the perpendicular subspace.

    For target_cos=0.0: All experts are random directions orthogonal to each other.
    For target_cos=0.9: All experts are close to expert A (highly correlated).

    Args:
        deltas_a: reference delta set (expert 0)
        deltas_b_original: second trained delta set (used for orthogonal basis)
        target_cos: target mean pairwise cosine similarity
        n_experts: number of experts to generate (including expert A)

    Returns:
        list of n_experts delta sets, each with controlled cosine relationships
        actual_mean_cos: measured mean pairwise cosine
    """
    flat_a = flatten_deltas(deltas_a)
    flat_b = flatten_deltas(deltas_b_original)

    norm_a = mx.sqrt(mx.sum(flat_a * flat_a)).item()
    norm_b = mx.sqrt(mx.sum(flat_b * flat_b)).item()

    # Create orthonormal basis: e1 = a/||a||, e2 = perp(b,a)/||perp||
    a_hat = flat_a / (norm_a + 1e-12)

    # Gram-Schmidt: project out a-component from b
    proj = mx.sum(flat_b * a_hat) * a_hat
    perp = flat_b - proj
    norm_perp = mx.sqrt(mx.sum(perp * perp)).item()
    e2 = perp / (norm_perp + 1e-12)

    # For more than 2 basis directions, we need additional random orthogonal vectors
    # We create a small subspace by adding random noise orthogonal to both e1 and e2
    D = flat_a.shape[0]

    # Generate experts as: alpha * e1 + beta_i * e2 + gamma_i * noise_i
    # where alpha = target_cos * norm_b controls alignment with expert A
    # and beta_i, gamma_i are chosen so that pairwise cosine ~ target_cos

    all_delta_sets = [deltas_a]  # Expert 0 is always the reference

    # For the remaining N-1 experts, create synthetic experts
    # Each has cosine ~ target_cos with expert A
    # and cosine ~ target_cos with each other (controlled by angular spread)
    for i in range(1, n_experts):
        # Rotate in the (e1, e2) plane plus small random perturbations
        # to create diverse experts at the target cosine
        angle_to_a = math.acos(min(max(target_cos, -1.0), 1.0))
        # Spread the N-1 experts uniformly around the cone at angle angle_to_a from e1
        phase = 2 * math.pi * i / (n_experts - 1) if n_experts > 2 else 0.0

        # For 2D case (enough for cosine control):
        # v_i = cos(angle_to_a) * e1 + sin(angle_to_a) * [cos(phase)*e2 + sin(phase)*e3]
        # But we only have e1 and e2 as meaningful directions.
        # Adding small random noise in high-D space for diversity among the N-1 experts

        # Generate random direction orthogonal to e1 and e2
        noise = mx.random.normal(shape=flat_a.shape)
        noise = noise - mx.sum(noise * a_hat) * a_hat  # remove e1 component
        noise = noise - mx.sum(noise * e2) * e2  # remove e2 component
        noise_norm = mx.sqrt(mx.sum(noise * noise)).item()
        e_noise = noise / (noise_norm + 1e-12)

        # Mix e2 and e_noise based on phase to create angular diversity
        cos_phase = math.cos(phase)
        sin_phase = math.sin(phase)
        perp_dir = cos_phase * e2 + sin_phase * e_noise
        # Renormalize (e2 and e_noise are orthogonal, so already unit norm)
        perp_dir_norm = mx.sqrt(mx.sum(perp_dir * perp_dir)).item()
        perp_dir = perp_dir / (perp_dir_norm + 1e-12)

        # Construct expert: target_cos * e1 + sqrt(1-target_cos^2) * perp_dir
        flat_expert = (target_cos * a_hat +
                      math.sqrt(max(1 - target_cos ** 2, 0)) * perp_dir)
        # Scale to match norm of expert B
        flat_expert = flat_expert * norm_b

        expert_deltas = unflatten_deltas(flat_expert, deltas_b_original)
        all_delta_sets.append(expert_deltas)

    # Measure actual pairwise cosines
    all_flat = [flatten_deltas(ds) for ds in all_delta_sets]
    pairwise_cosines = []
    for i in range(len(all_flat)):
        for j in range(i + 1, len(all_flat)):
            cos_ij = (mx.sum(all_flat[i] * all_flat[j]) /
                      (mx.sqrt(mx.sum(all_flat[i]**2)) *
                       mx.sqrt(mx.sum(all_flat[j]**2)) + 1e-12)).item()
            pairwise_cosines.append(cos_ij)

    actual_mean_cos = statistics.mean(pairwise_cosines) if pairwise_cosines else 0.0
    actual_std_cos = statistics.stdev(pairwise_cosines) if len(pairwise_cosines) > 1 else 0.0

    return all_delta_sets, actual_mean_cos, actual_std_cos, pairwise_cosines


# -- Discriminability Measurement ------------------------------------------

def measure_discriminability(models_or_deltas, base_model, dataset, V,
                             n_batches=10):
    """Measure mean pairwise expert discriminability.

    Discriminability D(i,j) = E_x[||f_i(x) - f_j(x)||_2] for each expert pair.

    For N experts, computes N*(N-1)/2 pairwise discriminabilities.

    Returns:
        mean_discriminability: mean across all pairs
        per_pair: list of (i, j, discriminability)
    """
    rng = random.Random(0)
    n_experts = len(models_or_deltas)

    # Build single-expert models to get per-expert outputs
    expert_models = []
    for delta_set in models_or_deltas:
        model = apply_deltas_to_base(base_model, delta_set, V)
        expert_models.append(model)

    per_pair = []
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            total_l2 = 0.0
            n_tokens = 0
            rng_pair = random.Random(0)  # consistent batches per pair
            for _ in range(n_batches):
                inputs, targets = dataset.get_batch(BATCH_SIZE, rng_pair)
                logits_i = expert_models[i](inputs)
                logits_j = expert_models[j](inputs)
                diff = logits_i - logits_j
                l2 = mx.sqrt(mx.sum(diff * diff, axis=-1))  # (B, T)
                total_l2 += mx.sum(l2).item()
                n_tokens += inputs.shape[0] * inputs.shape[1]
            mean_d = total_l2 / max(n_tokens, 1)
            per_pair.append((i, j, mean_d))

    mean_disc = statistics.mean([p[2] for p in per_pair]) if per_pair else 0.0
    return mean_disc, per_pair


# -- Router Gradient Extraction (extended for N>2) --------------------------

def calibrate_with_gradient_tracking_n_experts(
        model, train_datasets, val_ds, steps=300, lr=3e-3, seed=42):
    """Calibrate router for N experts and record per-step gradient norms.

    Args:
        model: RoutedDeltaGPT with N experts
        train_datasets: list of CharDatasets (one per domain)
        val_ds: validation dataset
        steps: calibration steps
        lr: learning rate
        seed: random seed

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
    n_domains = len(train_datasets)

    for step in range(1, steps + 1):
        # Cycle through domains
        domain_idx = step % n_domains
        inputs, targets = train_datasets[domain_idx].get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)

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


# -- Single Cosine Trial (N=8) ---------------------------------------------

def run_n8_trial(target_cos, base_model, deltas_a, deltas_b_original,
                 joint_model, train_datasets, val_ds, joint_val_loss, V,
                 seed=42):
    """Run one trial at N=8, top_k=2 with specific mean pairwise cosine."""
    print(f"\n  --- Target cosine = {target_cos:.1f} (N={N_EXPERTS}, k={TOP_K}) ---")

    # Generate N experts at target cosine
    all_delta_sets, actual_mean_cos, cos_std, pairwise_cos = \
        generate_n_experts_at_cosine(deltas_a, deltas_b_original, target_cos, N_EXPERTS)
    print(f"    Mean pairwise cosine: {actual_mean_cos:.4f} (std={cos_std:.4f})")

    # Measure discriminability
    mean_disc, per_pair_disc = measure_discriminability(
        all_delta_sets, base_model, val_ds, V, n_batches=5)
    print(f"    Mean discriminability: {mean_disc:.4f}")

    # Create routed model with N experts, top_k routing
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    routed_model = RoutedDeltaGPT(base_copy, all_delta_sets, V, top_k=TOP_K)
    mx.eval(routed_model.parameters())

    # Calibrate with gradient tracking
    loss_curve, grad_norms, final_val = calibrate_with_gradient_tracking_n_experts(
        routed_model, train_datasets, val_ds,
        steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    # Compute gradient statistics
    all_total_norms = [gn[1] for gn in grad_norms]
    mean_grad_norm = statistics.mean(all_total_norms) if all_total_norms else 0.0
    early_norms = [gn[1] for gn in grad_norms if gn[0] <= 50]
    mean_early_grad = statistics.mean(early_norms) if early_norms else 0.0
    late_norms = [gn[1] for gn in grad_norms if gn[0] > 250]
    mean_late_grad = statistics.mean(late_norms) if late_norms else 0.0

    # Per-layer gradient norms
    n_layers = len(grad_norms[0][2]) if grad_norms else 0
    per_layer_means = []
    for l in range(n_layers):
        layer_norms = [gn[2][l] for gn in grad_norms]
        per_layer_means.append(statistics.mean(layer_norms))

    vs_joint = (final_val - joint_val_loss) / joint_val_loss * 100

    print(f"    Mean grad norm: {mean_grad_norm:.6f} (early: {mean_early_grad:.6f})")
    print(f"    Final val: {final_val:.4f} (vs joint: {vs_joint:+.1f}%)")

    return {
        'target_cos': target_cos,
        'actual_mean_cos': actual_mean_cos,
        'cos_std': cos_std,
        'pairwise_cos': pairwise_cos,
        'mean_discriminability': mean_disc,
        'per_pair_disc': per_pair_disc,
        'mean_grad_norm': mean_grad_norm,
        'mean_early_grad': mean_early_grad,
        'mean_late_grad': mean_late_grad,
        'per_layer_grad_means': per_layer_means,
        'final_val_loss': final_val,
        'vs_joint_pct': vs_joint,
        'all_grad_norms': all_total_norms,
        'loss_curve': loss_curve,
    }


# -- N=2 Baseline Trial (for direct comparison) ----------------------------

def run_n2_trial(target_cos, base_model, deltas_a, deltas_b_original,
                 joint_model, train_datasets, val_ds, joint_val_loss, V,
                 seed=42):
    """Run one trial at N=2, top_k=2 (mixing only, reproducing parent)."""
    print(f"\n  --- Target cosine = {target_cos:.1f} (N=2, k=2 baseline) ---")

    # Project expert B
    deltas_b_proj, actual_cos = project_to_target_cosine(
        deltas_a, deltas_b_original, target_cos)
    print(f"    Achieved cosine: {actual_cos:.4f}")

    # Measure discriminability for N=2
    mean_disc, per_pair_disc = measure_discriminability(
        [deltas_a, deltas_b_proj], base_model, val_ds, V, n_batches=5)
    print(f"    Discriminability: {mean_disc:.4f}")

    # Create routed model
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    routed_model = RoutedDeltaGPT(base_copy, [deltas_a, deltas_b_proj], V, top_k=2)
    mx.eval(routed_model.parameters())

    # Calibrate with gradient tracking
    loss_curve, grad_norms, final_val = calibrate_with_gradient_tracking_n_experts(
        routed_model, train_datasets, val_ds,
        steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    all_total_norms = [gn[1] for gn in grad_norms]
    mean_grad_norm = statistics.mean(all_total_norms) if all_total_norms else 0.0
    early_norms = [gn[1] for gn in grad_norms if gn[0] <= 50]
    mean_early_grad = statistics.mean(early_norms) if early_norms else 0.0

    vs_joint = (final_val - joint_val_loss) / joint_val_loss * 100
    print(f"    Mean grad norm: {mean_grad_norm:.6f} (early: {mean_early_grad:.6f})")
    print(f"    Final val: {final_val:.4f} (vs joint: {vs_joint:+.1f}%)")

    return {
        'target_cos': target_cos,
        'actual_cos': actual_cos,
        'mean_discriminability': mean_disc,
        'mean_grad_norm': mean_grad_norm,
        'mean_early_grad': mean_early_grad,
        'vs_joint_pct': vs_joint,
    }


# -- Full Experiment -------------------------------------------------------

def run_experiment(seed=42, verbose=True):
    """Run the N>2 discriminability experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"DISCRIMINABILITY N>2 EXPERIMENT (seed={seed})")
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

    # Both domains for cycling during calibration
    train_datasets = [train_a, train_b]

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

    joint_val_loss = evaluate(model_joint, joint_val, BATCH_SIZE, n_batches=10)
    if verbose:
        print(f"  Joint val loss: {joint_val_loss:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)

    # === 3. Fine-tune 2 LoRA experts (basis for all synthetic experts) ===
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

    # === 5. Run N=8 trials at each target cosine ===
    if verbose:
        print(f"\n{'='*70}")
        print(f"N=8 TOP_K=2 COSINE SWEEP WITH GRADIENT TRACKING")
        print(f"{'='*70}")

    n8_trials = []
    for target_cos in TARGET_COSINES:
        trial = run_n8_trial(
            target_cos, base_model, deltas_a, deltas_b,
            model_joint, train_datasets, joint_val, joint_val_loss, V,
            seed=seed
        )
        n8_trials.append(trial)

    # === 6. Run N=2 baseline trials for direct comparison ===
    if verbose:
        print(f"\n{'='*70}")
        print(f"N=2 TOP_K=2 BASELINE (MIXING ONLY)")
        print(f"{'='*70}")

    n2_trials = []
    for target_cos in TARGET_COSINES:
        trial = run_n2_trial(
            target_cos, base_model, deltas_a, deltas_b,
            model_joint, train_datasets, joint_val, joint_val_loss, V,
            seed=seed
        )
        n2_trials.append(trial)

    return {
        'seed': seed,
        'joint_val_loss': joint_val_loss,
        'natural_cos': natural_cos,
        'n8_trials': n8_trials,
        'n2_trials': n2_trials,
    }


# -- Analysis ---------------------------------------------------------------

def analyze_results(all_experiments):
    """Analyze: does discriminability predict gradients at N=8?"""
    print(f"\n\n{'='*80}")
    print("DISCRIMINABILITY N>2 ANALYSIS")
    print(f"{'='*80}")

    # === N=8 Analysis ===
    print(f"\n{'='*80}")
    print("N=8, TOP_K=2 (SELECTION + MIXING)")
    print(f"{'='*80}")

    n8_by_cos = {}
    for exp in all_experiments:
        for trial in exp['n8_trials']:
            cos = trial['target_cos']
            if cos not in n8_by_cos:
                n8_by_cos[cos] = []
            n8_by_cos[cos].append(trial)

    print(f"\n{'Cos':>5} | {'MeanCos':>8} | {'Discrim':>8} | {'MeanGrad':>10} | "
          f"{'EarlyGrad':>10} | {'vs Joint':>9}")
    print("-" * 75)

    n8_cos_list = []
    n8_disc_list = []
    n8_grad_list = []
    n8_early_list = []
    n8_quality_list = []

    for cos in sorted(n8_by_cos.keys()):
        trials = n8_by_cos[cos]
        m_cos = statistics.mean([t['actual_mean_cos'] for t in trials])
        m_disc = statistics.mean([t['mean_discriminability'] for t in trials])
        m_grad = statistics.mean([t['mean_grad_norm'] for t in trials])
        m_early = statistics.mean([t['mean_early_grad'] for t in trials])
        m_vj = statistics.mean([t['vs_joint_pct'] for t in trials])

        print(f"{cos:>5.1f} | {m_cos:>8.4f} | {m_disc:>8.4f} | {m_grad:>10.6f} | "
              f"{m_early:>10.6f} | {m_vj:>+8.1f}%")

        for t in trials:
            n8_cos_list.append(cos)
            n8_disc_list.append(t['mean_discriminability'])
            n8_grad_list.append(t['mean_grad_norm'])
            n8_early_list.append(t['mean_early_grad'])
            n8_quality_list.append(t['vs_joint_pct'])

    # === N=2 Analysis ===
    print(f"\n{'='*80}")
    print("N=2, TOP_K=2 BASELINE (MIXING ONLY)")
    print(f"{'='*80}")

    n2_by_cos = {}
    for exp in all_experiments:
        for trial in exp['n2_trials']:
            cos = trial['target_cos']
            if cos not in n2_by_cos:
                n2_by_cos[cos] = []
            n2_by_cos[cos].append(trial)

    print(f"\n{'Cos':>5} | {'Discrim':>8} | {'MeanGrad':>10} | {'EarlyGrad':>10} | {'vs Joint':>9}")
    print("-" * 60)

    n2_cos_list = []
    n2_disc_list = []
    n2_grad_list = []
    n2_early_list = []
    n2_quality_list = []

    for cos in sorted(n2_by_cos.keys()):
        trials = n2_by_cos[cos]
        m_disc = statistics.mean([t['mean_discriminability'] for t in trials])
        m_grad = statistics.mean([t['mean_grad_norm'] for t in trials])
        m_early = statistics.mean([t['mean_early_grad'] for t in trials])
        m_vj = statistics.mean([t['vs_joint_pct'] for t in trials])

        print(f"{cos:>5.1f} | {m_disc:>8.4f} | {m_grad:>10.6f} | {m_early:>10.6f} | {m_vj:>+8.1f}%")

        for t in trials:
            n2_cos_list.append(cos)
            n2_disc_list.append(t['mean_discriminability'])
            n2_grad_list.append(t['mean_grad_norm'])
            n2_early_list.append(t['mean_early_grad'])
            n2_quality_list.append(t['vs_joint_pct'])

    # === Correlation Comparison ===
    print(f"\n{'='*80}")
    print("CORRELATION COMPARISON: N=8 vs N=2")
    print(f"{'='*80}")

    # N=8 correlations (pooled)
    r2_n8_disc_grad, r_n8_disc_grad, _ = compute_r_squared(n8_disc_list, n8_grad_list)
    r2_n8_cos_grad, r_n8_cos_grad, _ = compute_r_squared(n8_cos_list, n8_grad_list)
    r2_n8_disc_quality, r_n8_disc_quality, _ = compute_r_squared(n8_disc_list, n8_quality_list)
    r2_n8_grad_quality, r_n8_grad_quality, _ = compute_r_squared(n8_grad_list, n8_quality_list)

    # N=2 correlations (pooled)
    r2_n2_disc_grad, r_n2_disc_grad, _ = compute_r_squared(n2_disc_list, n2_grad_list)
    r2_n2_cos_grad, r_n2_cos_grad, _ = compute_r_squared(n2_cos_list, n2_grad_list)
    r2_n2_disc_quality, r_n2_disc_quality, _ = compute_r_squared(n2_disc_list, n2_quality_list)
    r2_n2_grad_quality, r_n2_grad_quality, _ = compute_r_squared(n2_grad_list, n2_quality_list)

    print(f"\n  Pooled correlations (all seeds x cosine levels):")
    print(f"  {'Metric':<30} | {'N=8 r^2':>8} | {'N=2 r^2':>8} | {'Delta':>8}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'Discriminability vs Gradient':<30} | {r2_n8_disc_grad:>8.4f} | {r2_n2_disc_grad:>8.4f} | {r2_n8_disc_grad - r2_n2_disc_grad:>+8.4f}")
    print(f"  {'Cosine vs Gradient':<30} | {r2_n8_cos_grad:>8.4f} | {r2_n2_cos_grad:>8.4f} | {r2_n8_cos_grad - r2_n2_cos_grad:>+8.4f}")
    print(f"  {'Discriminability vs Quality':<30} | {r2_n8_disc_quality:>8.4f} | {r2_n2_disc_quality:>8.4f} | {r2_n8_disc_quality - r2_n2_disc_quality:>+8.4f}")
    print(f"  {'Gradient vs Quality':<30} | {r2_n8_grad_quality:>8.4f} | {r2_n2_grad_quality:>8.4f} | {r2_n8_grad_quality - r2_n2_grad_quality:>+8.4f}")

    # Mean-curve correlations (more robust, fewer points)
    n8_mean_disc_by_cos = [statistics.mean([t['mean_discriminability'] for t in n8_by_cos[c]])
                           for c in sorted(n8_by_cos.keys())]
    n8_mean_grad_by_cos = [statistics.mean([t['mean_grad_norm'] for t in n8_by_cos[c]])
                           for c in sorted(n8_by_cos.keys())]
    n8_mean_quality_by_cos = [statistics.mean([t['vs_joint_pct'] for t in n8_by_cos[c]])
                              for c in sorted(n8_by_cos.keys())]
    cos_sorted = sorted(n8_by_cos.keys())

    r2_n8_mc_disc_grad, r_n8_mc_disc_grad, _ = compute_r_squared(n8_mean_disc_by_cos, n8_mean_grad_by_cos)
    r2_n8_mc_cos_grad, r_n8_mc_cos_grad, _ = compute_r_squared(list(cos_sorted), n8_mean_grad_by_cos)
    r2_n8_mc_grad_quality, _, _ = compute_r_squared(n8_mean_grad_by_cos, n8_mean_quality_by_cos)

    n2_mean_disc_by_cos = [statistics.mean([t['mean_discriminability'] for t in n2_by_cos[c]])
                           for c in sorted(n2_by_cos.keys())]
    n2_mean_grad_by_cos = [statistics.mean([t['mean_grad_norm'] for t in n2_by_cos[c]])
                           for c in sorted(n2_by_cos.keys())]
    n2_mean_quality_by_cos = [statistics.mean([t['vs_joint_pct'] for t in n2_by_cos[c]])
                              for c in sorted(n2_by_cos.keys())]

    r2_n2_mc_disc_grad, _, _ = compute_r_squared(n2_mean_disc_by_cos, n2_mean_grad_by_cos)
    r2_n2_mc_cos_grad, _, _ = compute_r_squared(list(sorted(n2_by_cos.keys())), n2_mean_grad_by_cos)
    r2_n2_mc_grad_quality, _, _ = compute_r_squared(n2_mean_grad_by_cos, n2_mean_quality_by_cos)

    print(f"\n  Mean-curve correlations ({len(cos_sorted)} cosine levels):")
    print(f"  {'Metric':<30} | {'N=8 r^2':>8} | {'N=2 r^2':>8} | {'Delta':>8}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'Discriminability vs Gradient':<30} | {r2_n8_mc_disc_grad:>8.4f} | {r2_n2_mc_disc_grad:>8.4f} | {r2_n8_mc_disc_grad - r2_n2_mc_disc_grad:>+8.4f}")
    print(f"  {'Cosine vs Gradient':<30} | {r2_n8_mc_cos_grad:>8.4f} | {r2_n2_mc_cos_grad:>8.4f} | {r2_n8_mc_cos_grad - r2_n2_mc_cos_grad:>+8.4f}")
    print(f"  {'Gradient vs Quality':<30} | {r2_n8_mc_grad_quality:>8.4f} | {r2_n2_mc_grad_quality:>8.4f} | {r2_n8_mc_grad_quality - r2_n2_mc_grad_quality:>+8.4f}")

    # === Phase Transition Check ===
    print(f"\n{'='*80}")
    print("PHASE TRANSITION CHECK")
    print(f"{'='*80}")

    # Check if phase transition still exists at N=8
    if 0.0 in n8_by_cos and 0.9 in n8_by_cos:
        g_cos0 = statistics.mean([t['mean_grad_norm'] for t in n8_by_cos[0.0]])
        g_cos9 = statistics.mean([t['mean_grad_norm'] for t in n8_by_cos[0.9]])
        ratio_n8 = g_cos0 / g_cos9 if g_cos9 > 1e-10 else float('inf')

        g2_cos0 = statistics.mean([t['mean_grad_norm'] for t in n2_by_cos[0.0]])
        g2_cos9 = statistics.mean([t['mean_grad_norm'] for t in n2_by_cos[0.9]])
        ratio_n2 = g2_cos0 / g2_cos9 if g2_cos9 > 1e-10 else float('inf')

        print(f"\n  N=8: grad(cos=0.0)/grad(cos=0.9) = {ratio_n8:.1f}x")
        print(f"  N=2: grad(cos=0.0)/grad(cos=0.9) = {ratio_n2:.1f}x")
        print(f"  Parent experiment: 15.5x")

    # Check regime split
    if 0.5 in n8_by_cos and 0.7 in n8_by_cos:
        regime_a = []  # cos <= 0.5
        regime_b = []  # cos >= 0.7
        for cos in sorted(n8_by_cos.keys()):
            trials = n8_by_cos[cos]
            grads = [t['mean_grad_norm'] for t in trials]
            if cos <= 0.5:
                regime_a.extend(grads)
            if cos >= 0.7:
                regime_b.extend(grads)

        if regime_a and regime_b:
            mean_a = statistics.mean(regime_a)
            mean_b = statistics.mean(regime_b)
            ratio_ab = mean_a / mean_b if mean_b > 1e-10 else float('inf')
            print(f"\n  N=8 regime split:")
            print(f"    Regime A (cos<=0.5): mean grad = {mean_a:.6f}")
            print(f"    Regime B (cos>=0.7): mean grad = {mean_b:.6f}")
            print(f"    Ratio A/B: {ratio_ab:.1f}x")

    # === KC2: Selection vs Mixing Gradient Comparison ===
    print(f"\n{'='*80}")
    print("KC2: SELECTION vs MIXING GRADIENT DYNAMICS")
    print(f"{'='*80}")

    # Compare gradient patterns N=8 vs N=2
    # If qualitatively different: gradient-cosine relationship shape differs
    # Check: correlation SIGN should be the same (both negative)
    print(f"\n  Gradient-cosine correlation signs:")
    print(f"    N=8: r = {r_n8_cos_grad:+.4f} ({'negative' if r_n8_cos_grad < 0 else 'positive'})")
    print(f"    N=2: r = {r_n2_cos_grad:+.4f} ({'negative' if r_n2_cos_grad < 0 else 'positive'})")

    same_sign = (r_n8_cos_grad < 0) == (r_n2_cos_grad < 0)
    print(f"    Same sign: {same_sign}")

    # Check: relative gradient magnitudes across cosine levels should be similar
    # Normalize by max gradient to compare shapes
    if n8_mean_grad_by_cos and n2_mean_grad_by_cos:
        max_n8 = max(n8_mean_grad_by_cos)
        max_n2 = max(n2_mean_grad_by_cos)
        if max_n8 > 0 and max_n2 > 0:
            norm_n8 = [g / max_n8 for g in n8_mean_grad_by_cos]
            norm_n2 = [g / max_n2 for g in n2_mean_grad_by_cos]
            # Compute correlation between normalized shapes
            r2_shape, r_shape, _ = compute_r_squared(norm_n8, norm_n2)
            print(f"\n  Shape correlation (normalized gradient profiles):")
            print(f"    r^2 = {r2_shape:.4f} (1.0 = identical shape)")
            print(f"    N=8 normalized: {[f'{g:.3f}' for g in norm_n8]}")
            print(f"    N=2 normalized: {[f'{g:.3f}' for g in norm_n2]}")

    # === Kill Criteria ===
    print(f"\n{'='*80}")
    print("KILL CRITERIA")
    print(f"{'='*80}")

    # KC1: r^2(discriminability, gradient) >= 0.3 at N=8
    # Use mean-curve r^2 (more appropriate for small N)
    kc1_r2 = r2_n8_mc_disc_grad
    kc1_pass = kc1_r2 >= 0.3
    print(f"\n  KC1: Discriminability predicts gradient at N=8, top_k=2?")
    print(f"    r^2(discriminability, gradient) = {kc1_r2:.4f} (mean curve)")
    print(f"    r^2(discriminability, gradient) = {r2_n8_disc_grad:.4f} (pooled)")
    print(f"    Threshold: >= 0.3")
    print(f"    {'PASS' if kc1_pass else 'KILL'}")

    # KC2: selection gradients NOT qualitatively different from mixing gradients
    # Same sign AND similar shape (r^2 > 0.5)
    kc2_same_sign = same_sign
    kc2_shape = r2_shape if 'r2_shape' in dir() else 0.0
    kc2_pass = kc2_same_sign and kc2_shape >= 0.5
    print(f"\n  KC2: Selection gradients similar to mixing gradients?")
    print(f"    Same correlation sign: {kc2_same_sign}")
    print(f"    Shape correlation r^2: {kc2_shape:.4f}")
    print(f"    Threshold: same sign AND shape r^2 >= 0.5")
    print(f"    {'PASS' if kc2_pass else 'KILL'}")

    # === Overall Verdict ===
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if kc1_pass and kc2_pass:
        print(f"\n  PROVEN: Discriminability mechanism generalizes to N=8, top_k=2")
        print(f"  Selection+mixing gradients follow the same discriminability dynamics")
        status = "proven"
    elif kc1_pass and not kc2_pass:
        print(f"\n  PARTIAL: Discriminability predicts gradients BUT dynamics differ")
        print(f"  Selection creates qualitatively different gradient patterns")
        status = "partial"
    elif not kc1_pass and kc2_pass:
        print(f"\n  KILLED: Discriminability does NOT predict gradients at N=8")
        print(f"  Even though dynamics shape is similar, predictive power is lost")
        status = "killed"
    else:
        print(f"\n  KILLED: Both criteria fail")
        status = "killed"

    return {
        'r2_n8_disc_grad_pooled': r2_n8_disc_grad,
        'r2_n8_disc_grad_mc': r2_n8_mc_disc_grad,
        'r2_n8_cos_grad_pooled': r2_n8_cos_grad,
        'r2_n8_cos_grad_mc': r2_n8_mc_cos_grad,
        'r2_n2_disc_grad_pooled': r2_n2_disc_grad,
        'r2_n2_disc_grad_mc': r2_n2_mc_disc_grad,
        'r_n8_cos_grad': r_n8_cos_grad,
        'r_n2_cos_grad': r_n2_cos_grad,
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
