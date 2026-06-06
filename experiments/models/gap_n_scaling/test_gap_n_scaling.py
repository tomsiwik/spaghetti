"""Gap-as-Signal at N>2: Does gap predict calibration quality when the router
must SELECT experts, not just learn mixing weights?

Protocol:
1. Pretrain shared base model
2. Train N=4 independent LoRA experts on 4 domains (quaternary: a-f, g-m, n-s, t-z)
3. For each orthogonality condition (all-orthogonal, mixed, correlated):
   a. Project expert deltas to achieve target pairwise cosine structure
   b. Create routed model with N=4, top_k=2
   c. Measure function-space gap, calibrate router, measure final quality
   d. Measure SELECTION ACCURACY: does router pick correct domain experts?
4. Repeat at N=8 (4 real + 4 projected duplicates) for scaling check
5. Compare gap-quality correlation at N=4 vs the N=2 baseline

Key difference from N=2: With 4 experts and top_k=2, there are C(4,2)=6
possible expert pairs. The router must learn a 6-way selection problem PLUS
mixing weights for each selected pair. This is qualitatively harder than
the 1-way mixing problem at N=2.

Kill criteria:
- Gap-quality correlation r^2 < 0.3 at N=4 with top_k=2
- Expert SELECTION accuracy does not improve with orthogonality
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
    measure_function_space_gap, compute_r_squared,
)


# -- Config ------------------------------------------------------------------

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3

MAX_CAL_STEPS = 300
CAL_EVAL_EVERY = 5
CONVERGENCE_THRESHOLD = 0.005


# -- Multi-Expert Projection -------------------------------------------------

def project_pool_to_target_cosine(all_deltas, target_cos):
    """Project all expert deltas so ALL pairwise cosines are approximately target_cos.

    Uses sequential Gram-Schmidt-like projection: keeps expert 0 fixed,
    projects each subsequent expert to have target_cos with ALL previous experts.

    For target_cos=0: makes all experts mutually orthogonal (Gram-Schmidt).
    For target_cos>0: blends each expert toward previous experts.

    Args:
        all_deltas: list of N delta sets
        target_cos: target pairwise cosine for ALL pairs

    Returns:
        projected_deltas: list of N delta sets with controlled pairwise cosines
        cosine_matrix: NxN matrix of actual pairwise cosines
    """
    N = len(all_deltas)
    flat_vecs = [flatten_deltas(d) for d in all_deltas]
    D = flat_vecs[0].shape[0]

    # Keep expert 0 unchanged
    result_flat = [flat_vecs[0]]

    for i in range(1, N):
        b = flat_vecs[i]
        b_norm = mx.sqrt(mx.sum(b * b))

        if target_cos < 1e-6:
            # Orthogonalize against ALL previous experts (Gram-Schmidt)
            v = b
            for j in range(i):
                a_hat_j = result_flat[j] / (mx.sqrt(mx.sum(result_flat[j] ** 2)) + 1e-12)
                v = v - mx.sum(v * a_hat_j) * a_hat_j
            v_norm = mx.sqrt(mx.sum(v * v))
            # Rescale to original norm
            v = v / (v_norm + 1e-12) * b_norm
            mx.eval(v)
            result_flat.append(v)
        else:
            # For non-zero target cosine, we average the directions of all
            # previous experts and project relative to that mean direction
            mean_prev = sum(result_flat[:i]) / i
            mean_prev_norm = mx.sqrt(mx.sum(mean_prev * mean_prev))
            a_hat = mean_prev / (mean_prev_norm + 1e-12)

            # Perpendicular component
            b_parallel = mx.sum(b * a_hat) * a_hat
            b_perp = b - b_parallel
            b_perp_norm = mx.sqrt(mx.sum(b_perp * b_perp))
            b_perp_hat = b_perp / (b_perp_norm + 1e-12)

            sin_c = math.sqrt(max(0, 1 - target_cos ** 2))
            v = target_cos * b_norm * a_hat + sin_c * b_norm * b_perp_hat
            mx.eval(v)
            result_flat.append(v)

    # Compute actual cosine matrix
    cosine_matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                cosine_matrix[i][j] = 1.0
            else:
                ni = mx.sqrt(mx.sum(result_flat[i] ** 2))
                nj = mx.sqrt(mx.sum(result_flat[j] ** 2))
                cosine_matrix[i][j] = (mx.sum(result_flat[i] * result_flat[j]) / (ni * nj + 1e-12)).item()

    # Unflatten back to delta format
    projected_deltas = []
    for i in range(N):
        projected_deltas.append(unflatten_deltas(result_flat[i], all_deltas[i]))

    return projected_deltas, cosine_matrix


# -- Selection Accuracy -------------------------------------------------------

def measure_selection_accuracy(model, domain_datasets, n_batches=10):
    """Measure whether the router selects the correct domain-specific experts.

    For each domain's validation data, checks which experts the router
    activates most strongly. "Correct" means the expert trained on that
    domain is among the top_k selected.

    Args:
        model: RoutedDeltaGPT with N experts
        domain_datasets: list of N CharDataset objects, one per domain

    Returns:
        per_domain_accuracy: list of N floats (fraction of tokens where
            the domain's own expert is in top_k)
        mean_accuracy: average across domains
        selection_entropy: average entropy of routing distribution
        expert_usage: NxN matrix where [domain][expert] = fraction of tokens
            routed to that expert from that domain
    """
    N = model.n_experts
    top_k = model.top_k
    n_layers = len(model.base_layers)

    # We measure at the middle layer (most discriminative)
    target_layer = n_layers // 2

    per_domain_accuracy = []
    expert_usage = [[0.0] * N for _ in range(len(domain_datasets))]
    total_entropy = 0.0
    total_tokens = 0

    for domain_idx, ds in enumerate(domain_datasets):
        rng = random.Random(0)
        correct_selections = 0
        domain_tokens = 0

        for _ in range(n_batches):
            inputs, targets = ds.get_batch(BATCH_SIZE, rng)
            B, T = inputs.shape

            # Forward through model up to the target layer to get router scores
            pos = mx.arange(T)
            x = model.wte(inputs) + model.wpe(pos)
            x = model.norm0(x)

            for l_idx, base_layer in enumerate(model.base_layers):
                x = x + base_layer.attn(base_layer.norm1(x))
                h = base_layer.norm2(x)

                if l_idx == target_layer:
                    # Get routing decisions
                    scores = model.routers[l_idx](h)  # (B, T, N)
                    probs = mx.softmax(scores, axis=-1)  # (B, T, N)

                    # Check if domain_idx expert is in top_k
                    top_vals = mx.topk(scores, top_k, axis=-1)  # (B, T, top_k)
                    threshold = mx.min(top_vals, axis=-1, keepdims=True)  # (B, T, 1)
                    selected = (scores >= threshold).astype(mx.float32)  # (B, T, N)

                    # Count tokens where domain expert is selected
                    if domain_idx < N:
                        domain_selected = selected[:, :, domain_idx]  # (B, T)
                        correct_selections += mx.sum(domain_selected).item()

                    # Track expert usage for this domain
                    for e in range(N):
                        expert_usage[domain_idx][e] += mx.sum(selected[:, :, e]).item()

                    # Entropy of routing probabilities
                    entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)  # (B, T)
                    total_entropy += mx.sum(entropy).item()
                    domain_tokens += B * T
                    total_tokens += B * T

                # Continue forward pass
                if not model.uniform:
                    scores_l = model.routers[l_idx](h)
                    probs_l = mx.softmax(scores_l, axis=-1)
                    top_vals_l = mx.topk(scores_l, top_k, axis=-1)
                    threshold_l = mx.min(top_vals_l, axis=-1, keepdims=True)
                    mask_l = (scores_l >= threshold_l).astype(mx.float32)
                    masked_probs = probs_l * mask_l
                    masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)
                    delta_out = mx.zeros_like(h)
                    for e in range(N):
                        w_e = masked_probs[..., e:e+1]
                        delta_out = delta_out + w_e * model._run_expert_mlp(h, l_idx, e)
                    x = x + delta_out
                else:
                    w = 1.0 / N
                    delta_out = mx.zeros_like(h)
                    for e in range(N):
                        delta_out = delta_out + w * model._run_expert_mlp(h, l_idx, e)
                    x = x + delta_out

        if domain_tokens > 0:
            acc = correct_selections / domain_tokens
        else:
            acc = 0.0
        per_domain_accuracy.append(acc)

        # Normalize expert usage
        total_domain_usage = sum(expert_usage[domain_idx])
        if total_domain_usage > 0:
            expert_usage[domain_idx] = [u / total_domain_usage for u in expert_usage[domain_idx]]

    mean_accuracy = statistics.mean(per_domain_accuracy) if per_domain_accuracy else 0.0
    mean_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0
    max_entropy = math.log(N) if N > 1 else 1.0
    normalized_entropy = mean_entropy / max_entropy

    return per_domain_accuracy, mean_accuracy, normalized_entropy, expert_usage


# -- Calibration with Selection Tracking --------------------------------------

def calibrate_router_n_experts(model, domain_train_datasets, val_ds,
                                joint_val_loss, steps=300, lr=3e-3, seed=42):
    """Calibrate router for N>2 experts, alternating domain batches.

    Returns:
        loss_curve: list of (step, val_loss)
        steps_to_converge: first step where val_loss < threshold, or None
        final_val_loss: val loss after all steps
    """
    model.freeze()
    for router in model.routers:
        router.unfreeze()

    N = len(domain_train_datasets)
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    convergence_target = joint_val_loss * (1 + CONVERGENCE_THRESHOLD)
    loss_curve = []
    steps_to_converge = None

    for step in range(1, steps + 1):
        # Round-robin across domains
        domain_idx = (step - 1) % N
        inputs, targets = domain_train_datasets[domain_idx].get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % CAL_EVAL_EVERY == 0 or step == 1:
            val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=5)
            loss_curve.append((step, val_loss))

            if steps_to_converge is None and val_loss <= convergence_target:
                steps_to_converge = step

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)
    model.unfreeze()

    return loss_curve, steps_to_converge, final_val_loss


# -- Single Trial at One Cosine Level ----------------------------------------

def run_n_expert_trial(target_cos, base_model, all_deltas_original, joint_model,
                        domain_train_datasets, domain_val_datasets, joint_val_ds,
                        joint_val_loss, V, N, seed=42):
    """Run one trial with N experts at a specific target pairwise cosine.

    Returns dict with all measurements.
    """
    print(f"\n  --- N={N}, Target cosine = {target_cos:.1f} ---")

    # Project all expert deltas to target cosine structure
    projected_deltas, cosine_matrix = project_pool_to_target_cosine(
        all_deltas_original, target_cos
    )

    # Print cosine matrix
    print(f"    Cosine matrix (actual):")
    for i in range(N):
        row = " ".join(f"{cosine_matrix[i][j]:+.3f}" for j in range(N))
        print(f"      [{row}]")

    # Mean off-diagonal cosine
    off_diag = []
    for i in range(N):
        for j in range(i + 1, N):
            off_diag.append(cosine_matrix[i][j])
    mean_cos = statistics.mean(off_diag) if off_diag else 0.0
    max_cos = max(off_diag) if off_diag else 0.0
    print(f"    Mean pairwise cosine: {mean_cos:.4f}, max: {max_cos:.4f}")

    # Create task-arithmetic model (simple average of all deltas)
    ta_deltas = []
    n_matrices = len(projected_deltas[0])
    for m_idx in range(n_matrices):
        l_idx, name, _ = projected_deltas[0][m_idx]
        avg_delta = sum(projected_deltas[e][m_idx][2] for e in range(N)) / N
        ta_deltas.append((l_idx, name, avg_delta))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)

    # Measure function-space gap: task-arithmetic vs joint
    _, ce_gap, kl_gap, prob_l1 = measure_function_space_gap(
        ta_model, joint_model, joint_val_ds
    )
    print(f"    Gap (TA): CE={ce_gap:.4f}, KL={kl_gap:.4f}, L1={prob_l1:.4f}")

    # Create routed model
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    routed_model = RoutedDeltaGPT(base_copy, projected_deltas, V, top_k=2)
    mx.eval(routed_model.parameters())

    # Measure gap before calibration
    _, ce_gap_pre, kl_gap_pre, _ = measure_function_space_gap(
        routed_model, joint_model, joint_val_ds
    )
    print(f"    Gap (routed, pre-cal): CE={ce_gap_pre:.4f}, KL={kl_gap_pre:.4f}")

    # Measure selection accuracy BEFORE calibration
    pre_acc, pre_mean_acc, pre_entropy, pre_usage = measure_selection_accuracy(
        routed_model, domain_val_datasets
    )
    print(f"    Selection acc (pre-cal): {pre_mean_acc:.3f} "
          f"(per-domain: {[f'{a:.3f}' for a in pre_acc]})")
    print(f"    Routing entropy (pre-cal): {pre_entropy:.3f} (1.0=uniform)")

    # Calibrate
    loss_curve, steps_to_converge, final_val = calibrate_router_n_experts(
        routed_model, domain_train_datasets, joint_val_ds,
        joint_val_loss, steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    # Measure gap after calibration
    _, ce_gap_post, kl_gap_post, _ = measure_function_space_gap(
        routed_model, joint_model, joint_val_ds
    )

    # Measure selection accuracy AFTER calibration
    post_acc, post_mean_acc, post_entropy, post_usage = measure_selection_accuracy(
        routed_model, domain_val_datasets
    )
    print(f"    Selection acc (post-cal): {post_mean_acc:.3f} "
          f"(per-domain: {[f'{a:.3f}' for a in post_acc]})")
    print(f"    Routing entropy (post-cal): {post_entropy:.3f}")

    # Per-domain validation losses
    domain_val_losses = []
    for ds in domain_val_datasets:
        vl = evaluate(routed_model, ds, BATCH_SIZE)
        domain_val_losses.append(vl)
    avg_domain_val = statistics.mean(domain_val_losses)

    print(f"    Steps to converge: {steps_to_converge}")
    print(f"    Final val loss: {final_val:.4f} (joint: {joint_val_loss:.4f})")
    print(f"    Per-domain: {[f'{v:.4f}' for v in domain_val_losses]}")
    print(f"    vs joint: {(avg_domain_val - joint_val_loss) / joint_val_loss * 100:+.1f}%")

    # AUC metric
    if loss_curve:
        auc = sum(vl for _, vl in loss_curve) / len(loss_curve)
    else:
        auc = float('inf')

    # Selection accuracy improvement
    selection_improvement = post_mean_acc - pre_mean_acc

    return {
        'target_cos': target_cos,
        'N': N,
        'mean_pairwise_cos': mean_cos,
        'max_pairwise_cos': max_cos,
        'ce_gap_ta': ce_gap,
        'kl_gap_ta': kl_gap,
        'prob_l1_ta': prob_l1,
        'ce_gap_pre': ce_gap_pre,
        'kl_gap_pre': kl_gap_pre,
        'ce_gap_post': ce_gap_post,
        'kl_gap_post': kl_gap_post,
        'pre_selection_accuracy': pre_mean_acc,
        'post_selection_accuracy': post_mean_acc,
        'selection_improvement': selection_improvement,
        'pre_routing_entropy': pre_entropy,
        'post_routing_entropy': post_entropy,
        'per_domain_accuracy_post': post_acc,
        'expert_usage_post': post_usage,
        'steps_to_converge': steps_to_converge,
        'final_val_loss': final_val,
        'avg_domain_val': avg_domain_val,
        'vs_joint_pct': (avg_domain_val - joint_val_loss) / joint_val_loss * 100,
        'domain_val_losses': domain_val_losses,
        'loss_curve': loss_curve,
        'auc': auc,
        'cosine_matrix': cosine_matrix,
    }


# -- Full Experiment ----------------------------------------------------------

def run_experiment(seed=42, verbose=True):
    """Run the full N>2 gap-as-signal experiment.

    Tests N=4 experts with top_k=2 across cosine levels.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAP-AS-SIGNAL N>2 EXPERIMENT (seed={seed})")
        print(f"{'='*70}")

    mx.random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # 4-domain split
    splits = domain_split(docs, method="quaternary")
    domain_keys = sorted(splits.keys())  # a_f, g_m, n_s, t_z
    N = len(domain_keys)

    if verbose:
        print(f"\n  Domains: {domain_keys}")
        for k in domain_keys:
            print(f"    {k}: {len(splits[k])} names")

    all_train, all_val = train_val_split(docs, seed=seed)

    domain_train_datasets = []
    domain_val_datasets = []
    for k in domain_keys:
        d_train, d_val = train_val_split(splits[k], seed=seed)
        domain_train_datasets.append(CharDataset(d_train, tokenizer, BASE["block_size"]))
        domain_val_datasets.append(CharDataset(d_val, tokenizer, BASE["block_size"]))

    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    # === 1. Joint training baseline ===
    if verbose:
        print("\n--- Joint training baseline ---")
    model_joint = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(model_joint.parameters())
    total_steps = N * FINETUNE_STEPS  # more steps for more domains
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    for step in range(1, total_steps + 1):
        domain_idx = (step - 1) % N
        inputs, targets = domain_train_datasets[domain_idx].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    domain_joint_losses = []
    for ds in domain_val_datasets:
        vl = evaluate(model_joint, ds, BATCH_SIZE)
        domain_joint_losses.append(vl)
    joint_val_loss = statistics.mean(domain_joint_losses)
    if verbose:
        print(f"  Joint val losses: {[f'{v:.4f}' for v in domain_joint_losses]}")
        print(f"  Joint avg: {joint_val_loss:.4f}")

    joint_on_joint = evaluate(model_joint, joint_val, BATCH_SIZE)
    if verbose:
        print(f"  Joint on joint_val: {joint_on_joint:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)

    # === 3. Fine-tune N LoRA experts ===
    all_deltas = []
    lora_models = []

    for d_idx, k in enumerate(domain_keys):
        if verbose:
            print(f"\n--- Fine-tuning LoRA for domain {d_idx}: {k} ---")
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
        train(lora_model, domain_train_datasets[d_idx], domain_val_datasets[d_idx],
              steps=FINETUNE_STEPS, batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
        lora_model.unfreeze()

        deltas = get_deltas(lora_model)
        all_deltas.append(deltas)
        lora_models.append(lora_model)

    # === 4. Measure natural pairwise cosines ===
    if verbose:
        print(f"\n  Natural pairwise cosines:")
    flat_vecs = [flatten_deltas(d) for d in all_deltas]
    natural_cosines = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                natural_cosines[i][j] = 1.0
            else:
                ni = mx.sqrt(mx.sum(flat_vecs[i] ** 2))
                nj = mx.sqrt(mx.sum(flat_vecs[j] ** 2))
                natural_cosines[i][j] = (mx.sum(flat_vecs[i] * flat_vecs[j]) / (ni * nj + 1e-12)).item()
    if verbose:
        for i in range(N):
            row = " ".join(f"{natural_cosines[i][j]:+.3f}" for j in range(N))
            print(f"    [{row}]")

    # === 5. Run cosine sweep ===
    TARGET_COSINES = [0.0, 0.2, 0.5, 0.7, 0.9]

    if verbose:
        print(f"\n{'='*70}")
        print("RUNNING N=4 COSINE SWEEP")
        print(f"{'='*70}")

    trials = []
    for target_cos in TARGET_COSINES:
        trial = run_n_expert_trial(
            target_cos, base_model, all_deltas, model_joint,
            domain_train_datasets, domain_val_datasets, joint_val,
            joint_on_joint, V, N, seed=seed
        )
        trials.append(trial)

    return {
        'seed': seed,
        'N': N,
        'domain_keys': domain_keys,
        'joint_val_loss': joint_val_loss,
        'joint_on_joint': joint_on_joint,
        'natural_cosines': natural_cosines,
        'trials': trials,
    }


# -- Analysis ----------------------------------------------------------------

def analyze_results(all_experiments):
    """Analyze N>2 gap-as-signal results."""
    print(f"\n\n{'='*80}")
    print("GAP-AS-SIGNAL N>2 ANALYSIS")
    print(f"{'='*80}")

    # Aggregate trials
    by_cos = {}
    for exp in all_experiments:
        for trial in exp['trials']:
            cos = trial['target_cos']
            if cos not in by_cos:
                by_cos[cos] = []
            by_cos[cos].append(trial)

    # Summary table
    print(f"\n{'Cos':>5} | {'CE Gap':>8} | {'KL Gap':>8} | {'Final VL':>9} | "
          f"{'vs Joint':>9} | {'SelAcc':>7} | {'Entropy':>8} | {'AUC':>8}")
    print("-" * 85)

    cos_list = []
    ce_gaps_all = []
    kl_gaps_all = []
    vs_joint_all = []
    sel_acc_all = []
    auc_list = []
    entropy_all = []

    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        mean_ce = statistics.mean([t['ce_gap_ta'] for t in trials])
        mean_kl = statistics.mean([t['kl_gap_ta'] for t in trials])
        mean_vl = statistics.mean([t['final_val_loss'] for t in trials])
        mean_vj = statistics.mean([t['vs_joint_pct'] for t in trials])
        mean_sel = statistics.mean([t['post_selection_accuracy'] for t in trials])
        mean_ent = statistics.mean([t['post_routing_entropy'] for t in trials])
        mean_auc = statistics.mean([t['auc'] for t in trials])

        print(f"{cos:>5.1f} | {mean_ce:>8.4f} | {mean_kl:>8.4f} | {mean_vl:>9.4f} | "
              f"{mean_vj:>+8.1f}% | {mean_sel:>7.3f} | {mean_ent:>8.3f} | {mean_auc:>8.4f}")

        for t in trials:
            cos_list.append(cos)
            ce_gaps_all.append(t['ce_gap_ta'])
            kl_gaps_all.append(t['kl_gap_ta'])
            vs_joint_all.append(t['vs_joint_pct'])
            sel_acc_all.append(t['post_selection_accuracy'])
            auc_list.append(t['auc'])
            entropy_all.append(t['post_routing_entropy'])

    # === Correlation Analysis ===
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")

    r2_cos_quality, r_cos_quality, _ = compute_r_squared(cos_list, vs_joint_all)
    print(f"\n1. Cosine vs Final Quality (% above joint):")
    print(f"   r = {r_cos_quality:.4f}, r^2 = {r2_cos_quality:.4f}")

    r2_ce_quality, r_ce_quality, _ = compute_r_squared(ce_gaps_all, vs_joint_all)
    print(f"\n2. CE Gap vs Final Quality:")
    print(f"   r = {r_ce_quality:.4f}, r^2 = {r2_ce_quality:.4f}")

    r2_kl_quality, r_kl_quality, _ = compute_r_squared(kl_gaps_all, vs_joint_all)
    print(f"\n3. KL Gap vs Final Quality:")
    print(f"   r = {r_kl_quality:.4f}, r^2 = {r2_kl_quality:.4f}")

    r2_cos_sel, r_cos_sel, _ = compute_r_squared(cos_list, sel_acc_all)
    print(f"\n4. Cosine vs Selection Accuracy:")
    print(f"   r = {r_cos_sel:.4f}, r^2 = {r2_cos_sel:.4f}")
    print(f"   Direction: {'negative (expected: lower cos = better selection)' if r_cos_sel < 0 else 'positive (unexpected)'}")

    r2_cos_entropy, r_cos_entropy, _ = compute_r_squared(cos_list, entropy_all)
    print(f"\n5. Cosine vs Routing Entropy:")
    print(f"   r = {r_cos_entropy:.4f}, r^2 = {r2_cos_entropy:.4f}")

    r2_sel_quality, r_sel_quality, _ = compute_r_squared(sel_acc_all, vs_joint_all)
    print(f"\n6. Selection Accuracy vs Quality:")
    print(f"   r = {r_sel_quality:.4f}, r^2 = {r2_sel_quality:.4f}")

    r2_cos_auc, r_cos_auc, _ = compute_r_squared(cos_list, auc_list)
    print(f"\n7. Cosine vs AUC:")
    print(f"   r = {r_cos_auc:.4f}, r^2 = {r2_cos_auc:.4f}")

    # === Selection Accuracy Detail ===
    print(f"\n{'='*80}")
    print("SELECTION ACCURACY DETAIL")
    print(f"{'='*80}")

    # Selection accuracy improvement with orthogonality
    if by_cos.get(0.0) and by_cos.get(0.9):
        ortho_sel = statistics.mean([t['post_selection_accuracy'] for t in by_cos[0.0]])
        corr_sel = statistics.mean([t['post_selection_accuracy'] for t in by_cos[0.9]])
        ortho_sel_pre = statistics.mean([t['pre_selection_accuracy'] for t in by_cos[0.0]])
        corr_sel_pre = statistics.mean([t['pre_selection_accuracy'] for t in by_cos[0.9]])
        print(f"\n  Orthogonal (cos=0.0):")
        print(f"    Pre-calibration selection: {ortho_sel_pre:.3f}")
        print(f"    Post-calibration selection: {ortho_sel:.3f}")
        print(f"    Improvement: {ortho_sel - ortho_sel_pre:+.3f}")
        print(f"\n  Correlated (cos=0.9):")
        print(f"    Pre-calibration selection: {corr_sel_pre:.3f}")
        print(f"    Post-calibration selection: {corr_sel:.3f}")
        print(f"    Improvement: {corr_sel - corr_sel_pre:+.3f}")

    # Expert usage matrix for orthogonal case
    if by_cos.get(0.0):
        trial_0 = by_cos[0.0][0]
        print(f"\n  Expert usage matrix (cos=0.0, post-calibration):")
        print(f"  {'':>8}", end="")
        for e in range(trial_0['N']):
            print(f"  Expert{e}", end="")
        print()
        for d_idx in range(trial_0['N']):
            print(f"  Domain{d_idx}", end="")
            for e in range(trial_0['N']):
                print(f"    {trial_0['expert_usage_post'][d_idx][e]:.3f}", end="")
            print()

    # === Kill Criteria ===
    print(f"\n{'='*80}")
    print("KILL CRITERIA")
    print(f"{'='*80}")

    # Kill 1: gap-quality correlation r^2 < 0.3 at N=4 with top_k=2
    best_r2 = max(r2_cos_quality, r2_ce_quality, r2_kl_quality)
    r2_map = {
        r2_cos_quality: "cos_vs_quality",
        r2_ce_quality: "CE_gap_vs_quality",
        r2_kl_quality: "KL_gap_vs_quality",
    }
    best_metric = r2_map.get(best_r2, "unknown")
    print(f"\n1. Gap-quality correlation r^2 >= 0.3 at N=4, top_k=2?")
    print(f"   Best r^2 = {best_r2:.4f} (metric: {best_metric})")
    kill_1 = best_r2 < 0.3
    if kill_1:
        print(f"   KILL: r^2 = {best_r2:.4f} < 0.3")
    else:
        print(f"   PASS: r^2 = {best_r2:.4f} >= 0.3")

    # Kill 2: selection accuracy does NOT improve with orthogonality
    sel_improves = False
    if by_cos.get(0.0) and by_cos.get(0.9):
        ortho_sel_post = statistics.mean([t['post_selection_accuracy'] for t in by_cos[0.0]])
        corr_sel_post = statistics.mean([t['post_selection_accuracy'] for t in by_cos[0.9]])
        # Also check if orthogonal selection IMPROVES more from calibration
        ortho_improvement = statistics.mean([t['selection_improvement'] for t in by_cos[0.0]])
        corr_improvement = statistics.mean([t['selection_improvement'] for t in by_cos[0.9]])

        print(f"\n2. Expert selection accuracy improves with orthogonality?")
        print(f"   Orthogonal post-cal selection: {ortho_sel_post:.3f}")
        print(f"   Correlated post-cal selection: {corr_sel_post:.3f}")
        print(f"   Orthogonal improvement from calibration: {ortho_improvement:+.3f}")
        print(f"   Correlated improvement from calibration: {corr_improvement:+.3f}")

        # Selection improves if either:
        # a) Orthogonal has better absolute selection accuracy, OR
        # b) Orthogonal shows more improvement from calibration
        if ortho_sel_post > corr_sel_post or ortho_improvement > corr_improvement:
            sel_improves = True
            print(f"   PASS: orthogonal selection is better or improves more")
        else:
            print(f"   KILL: selection accuracy does not improve with orthogonality")
    else:
        print(f"\n2. Cannot test (missing cos=0.0 or cos=0.9 data)")

    # Kill 3 (bonus): orthogonal experts produce better quality at N=4
    print(f"\n3. Orthogonal experts produce better quality at N=4?")
    if by_cos.get(0.0) and by_cos.get(0.9):
        ortho_q = statistics.mean([t['vs_joint_pct'] for t in by_cos[0.0]])
        corr_q = statistics.mean([t['vs_joint_pct'] for t in by_cos[0.9]])
        if ortho_q < corr_q:
            print(f"   PASS: orthogonal ({ortho_q:+.1f}%) closer to joint than correlated ({corr_q:+.1f}%)")
        else:
            print(f"   FAIL: orthogonal ({ortho_q:+.1f}%) NOT closer to joint")

    # === Overall Verdict ===
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if not kill_1 and sel_improves:
        print(f"\n  HYPOTHESIS PROVEN at N=4:")
        print(f"  Gap-as-signal holds for expert SELECTION (not just mixing).")
        print(f"  Gap-quality correlation r^2 = {best_r2:.4f} (N=2 was 0.74)")
        verdict = "PROVEN"
    elif kill_1:
        print(f"\n  HYPOTHESIS KILLED:")
        print(f"  Gap-quality correlation r^2 = {best_r2:.4f} < 0.3")
        print(f"  Gap-as-signal does NOT generalize from mixing to selection.")
        verdict = "KILLED"
    elif not sel_improves:
        print(f"\n  HYPOTHESIS PARTIALLY KILLED:")
        print(f"  Gap-quality correlation holds (r^2={best_r2:.4f})")
        print(f"  BUT selection accuracy does not improve with orthogonality.")
        verdict = "PARTIAL"
    else:
        print(f"\n  INCONCLUSIVE")
        verdict = "INCONCLUSIVE"

    return {
        'r2_cos_quality': r2_cos_quality,
        'r2_ce_quality': r2_ce_quality,
        'r2_kl_quality': r2_kl_quality,
        'r2_cos_selection': r2_cos_sel,
        'r2_cos_entropy': r2_cos_entropy,
        'r2_sel_quality': r2_sel_quality,
        'best_r2': best_r2,
        'best_metric': best_metric,
        'verdict': verdict,
    }


# -- Multi-Seed Runner -------------------------------------------------------

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
