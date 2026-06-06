"""Dense backpropagation for MoE calibration at N>2.

Parent experiment (discriminability_n_gt_2) proved:
  - At N=8, top_k=2: router gradients are 5-7x smaller than at N=2
  - This is consistent with k/N = 25% dilution (only 2 of 8 experts get gradients)
  - Discriminability mechanism works (r^2=0.46) but is attenuated

This experiment tests: can dense backpropagation restore N=2-level gradient
magnitude while keeping forward pass sparse?

Dense backprop: forward uses top-k selection, backward computes gradients
through ALL expert outputs. Implemented via straight-through estimator:
  forward: sparse_weights (top-k masked + renormalized)
  backward: dense_weights (full softmax, all N experts)

Kill criteria:
  KC1: dense backprop does NOT close gradient magnitude gap vs N=2 by >50%
  KC2: calibration convergence speed improvement <2x vs standard top-k backprop
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
from experiments.models.gap_causal_mechanism.test_gap_causal_mechanism import (
    extract_router_gradient_norms,
)
from experiments.models.discriminability_n_gt_2.test_discriminability_n_gt_2 import (
    generate_n_experts_at_cosine,
    measure_discriminability,
    BASE, LORA_RANK, LORA_ALPHA, PRETRAIN_STEPS, FINETUNE_STEPS,
    BATCH_SIZE, LR, N_EXPERTS, TOP_K, TARGET_COSINES,
    MAX_CAL_STEPS, CAL_EVAL_EVERY,
)


# -- Dense Backprop RoutedDeltaGPT -----------------------------------------

class DenseBackpropRoutedDeltaGPT(nn.Module):
    """RoutedDeltaGPT with dense backpropagation.

    Key difference from RoutedDeltaGPT:
    - Forward: computes ALL expert outputs, but the forward VALUE uses only top-k
    - Backward: gradients flow through all N experts via straight-through estimator

    The straight-through trick:
        sparse_weights = top_k_mask(softmax(scores))  # used in forward
        dense_weights = softmax(scores)                # used in backward
        weights = dense_weights + stop_gradient(sparse_weights - dense_weights)

    This means:
        forward value = sum_k(sparse_weight_k * expert_k(x))   [sparse, efficient]
        backward grad = d/d_theta sum_N(dense_weight_i * expert_i(x))  [dense, informative]

    NOTE: forward still computes all N expert outputs (for backward). This is fine
    for calibration (small number of steps). At inference, switch to standard top-k.
    """

    def __init__(self, base_model, delta_sets, vocab_size,
                 top_k: int = 1, dense_backprop: bool = True):
        super().__init__()
        self.n_experts = len(delta_sets)
        self.top_k = min(top_k, self.n_experts)
        self.dense_backprop = dense_backprop
        n_embd = BASE['n_embd']

        # Copy base model weights
        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.base_layers = base_model.layers
        self.lm_head = base_model.lm_head

        # Pre-build expert weight matrices
        n_layer = len(base_model.layers)
        self._expert_fc1_weights = []
        self._expert_fc2_weights = []

        for l_idx in range(n_layer):
            base_fc1_w = base_model.layers[l_idx].mlp.fc1.weight
            base_fc2_w = base_model.layers[l_idx].mlp.fc2.weight

            fc1_list = []
            fc2_list = []
            for expert_idx, deltas in enumerate(delta_sets):
                for dl_idx, name, delta in deltas:
                    if dl_idx == l_idx and name == 'fc1':
                        fc1_list.append(base_fc1_w + delta.T)
                    elif dl_idx == l_idx and name == 'fc2':
                        fc2_list.append(base_fc2_w + delta.T)

            self._expert_fc1_weights.append(mx.stack(fc1_list))
            self._expert_fc2_weights.append(mx.stack(fc2_list))

        mx.eval(self._expert_fc1_weights + self._expert_fc2_weights)

        # Per-layer router
        self.routers = [nn.Linear(n_embd, self.n_experts, bias=False)
                        for _ in range(n_layer)]

    def _run_expert_mlp(self, h, l_idx, expert_idx):
        """Run MLP with expert-specific weights."""
        fc1_w = self._expert_fc1_weights[l_idx][expert_idx]
        fc2_w = self._expert_fc2_weights[l_idx][expert_idx]
        h_fc1 = h @ fc1_w.T
        h_relu = nn.relu(h_fc1)
        return h_relu @ fc2_w.T

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        for l_idx, base_layer in enumerate(self.base_layers):
            # Attention (from base, unmodified)
            x = x + base_layer.attn(base_layer.norm1(x))

            # MLP with routed LoRA deltas
            h = base_layer.norm2(x)
            scores = self.routers[l_idx](h)  # (B, T, N)
            dense_probs = mx.softmax(scores, axis=-1)  # (B, T, N)

            if self.dense_backprop:
                # Dense backprop: straight-through estimator
                # Forward value: top-k masked weights
                # Backward: gradients through all N experts via dense_probs

                # Compute top-k mask
                top_vals = mx.topk(scores, self.top_k, axis=-1)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                mask = (scores >= threshold).astype(mx.float32)
                sparse_probs = dense_probs * mask
                sparse_probs = sparse_probs / (mx.sum(sparse_probs, axis=-1, keepdims=True) + 1e-8)

                # Straight-through: forward uses sparse, backward uses dense
                # weights = dense_probs + stop_gradient(sparse_probs - dense_probs)
                # This evaluates to sparse_probs in forward but has dense_probs gradient
                weights = dense_probs + mx.stop_gradient(sparse_probs - dense_probs)

                # Compute ALL expert outputs (needed for dense backward)
                delta_out = mx.zeros_like(h)
                for e in range(self.n_experts):
                    w_e = weights[..., e:e+1]
                    delta_out = delta_out + w_e * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out
            else:
                # Standard sparse backprop (parent behavior)
                top_vals = mx.topk(scores, self.top_k, axis=-1)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                mask = (scores >= threshold).astype(mx.float32)
                masked_probs = dense_probs * mask
                masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

                delta_out = mx.zeros_like(h)
                for e in range(self.n_experts):
                    w_e = masked_probs[..., e:e+1]
                    delta_out = delta_out + w_e * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out

        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        return mx.array(0.0)

    def on_domain_switch(self, domain: str):
        pass


# -- Calibration with gradient tracking -----------------------------------

def calibrate_with_gradient_tracking(
        model, train_datasets, val_ds, steps=300, lr=3e-3, seed=42,
        convergence_threshold=None, baseline_loss=None):
    """Calibrate router and record per-step gradient norms.

    Args:
        model: DenseBackpropRoutedDeltaGPT or RoutedDeltaGPT
        train_datasets: list of CharDatasets
        val_ds: validation dataset
        steps: max calibration steps
        lr: learning rate
        seed: random seed
        convergence_threshold: if set, stop when val_loss < this value
        baseline_loss: reference loss for convergence measurement

    Returns:
        loss_curve: list of (step, val_loss)
        grad_norms: list of (step, total_grad_norm, per_layer_norms)
        final_val_loss: final validation loss
        convergence_step: step at which convergence_threshold was reached (or None)
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
    convergence_step = None

    for step in range(1, steps + 1):
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

            # Check convergence
            if convergence_threshold is not None and convergence_step is None:
                if val_loss <= convergence_threshold:
                    convergence_step = step

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)
    model.unfreeze()

    return loss_curve, grad_norms, final_val_loss, convergence_step


# -- Single Trial ----------------------------------------------------------

def run_trial(target_cos, base_model, deltas_a, deltas_b_original,
              train_datasets, val_ds, joint_val_loss, V,
              dense_backprop=True, n_experts=N_EXPERTS, top_k=TOP_K,
              seed=42):
    """Run one trial with either dense or sparse backprop."""
    mode = "dense" if dense_backprop else "sparse"
    print(f"\n  --- cos={target_cos:.1f}, N={n_experts}, k={top_k}, {mode} backprop ---")

    if n_experts == 2:
        # N=2 case: project expert B to target cosine
        deltas_b_proj, actual_cos = project_to_target_cosine(
            deltas_a, deltas_b_original, target_cos)
        all_delta_sets = [deltas_a, deltas_b_proj]
        actual_mean_cos = actual_cos
    else:
        # N>2: generate synthetic experts at target cosine
        all_delta_sets, actual_mean_cos, _, _ = generate_n_experts_at_cosine(
            deltas_a, deltas_b_original, target_cos, n_experts)

    print(f"    Actual cosine: {actual_mean_cos:.4f}")

    # Measure discriminability
    mean_disc, _ = measure_discriminability(
        all_delta_sets, base_model, val_ds, V, n_batches=5)
    print(f"    Discriminability: {mean_disc:.4f}")

    # Create model
    base_copy = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    if dense_backprop or n_experts > 2:
        # Use our DenseBackpropRoutedDeltaGPT for both dense and sparse comparison
        model = DenseBackpropRoutedDeltaGPT(
            base_copy, all_delta_sets, V, top_k=top_k,
            dense_backprop=dense_backprop)
    else:
        # N=2 sparse: use standard RoutedDeltaGPT for perfect comparison with parent
        model = RoutedDeltaGPT(base_copy, all_delta_sets, V, top_k=top_k)

    mx.eval(model.parameters())

    # Set convergence threshold: N=2 sparse final loss as the target
    # (We'll compute this from N=2 results in the analysis)
    loss_curve, grad_norms, final_val, convergence_step = \
        calibrate_with_gradient_tracking(
            model, train_datasets, val_ds,
            steps=MAX_CAL_STEPS, lr=LR, seed=seed)

    # Gradient statistics
    all_total_norms = [gn[1] for gn in grad_norms]
    mean_grad_norm = statistics.mean(all_total_norms)
    early_norms = [gn[1] for gn in grad_norms if gn[0] <= 50]
    mean_early_grad = statistics.mean(early_norms)
    late_norms = [gn[1] for gn in grad_norms if gn[0] > 250]
    mean_late_grad = statistics.mean(late_norms) if late_norms else 0.0

    # Per-layer
    n_layers = len(grad_norms[0][2])
    per_layer_means = []
    for l in range(n_layers):
        layer_norms = [gn[2][l] for gn in grad_norms]
        per_layer_means.append(statistics.mean(layer_norms))

    vs_joint = (final_val - joint_val_loss) / joint_val_loss * 100

    print(f"    Mean grad: {mean_grad_norm:.6f} (early: {mean_early_grad:.6f})")
    print(f"    Final val: {final_val:.4f} (vs joint: {vs_joint:+.1f}%)")

    return {
        'target_cos': target_cos,
        'actual_cos': actual_mean_cos,
        'n_experts': n_experts,
        'top_k': top_k,
        'dense_backprop': dense_backprop,
        'mean_discriminability': mean_disc,
        'mean_grad_norm': mean_grad_norm,
        'mean_early_grad': mean_early_grad,
        'mean_late_grad': mean_late_grad,
        'per_layer_grad_means': per_layer_means,
        'final_val_loss': final_val,
        'vs_joint_pct': vs_joint,
        'all_grad_norms': all_total_norms,
        'loss_curve': loss_curve,
    }


# -- Full Experiment -------------------------------------------------------

def run_experiment(seed=42, verbose=True):
    """Run the dense backprop calibration experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"DENSE BACKPROP CALIBRATION EXPERIMENT (seed={seed})")
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
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
        lora_model.unfreeze()
        return lora_model

    lora_a = finetune_lora(train_a, val_a, "A (a-m)")
    lora_b = finetune_lora(train_b, val_b, "B (n-z)")

    # === 4. Extract deltas ===
    deltas_a = get_deltas(lora_a)
    deltas_b = get_deltas(lora_b)

    # === 5. Run comparison at key cosine levels ===
    # Use fewer cosine levels than parent (focus on 0.0 and 0.5 for clarity)
    test_cosines = [0.0, 0.3, 0.7]

    results = {
        'seed': seed,
        'joint_val_loss': joint_val_loss,
        'trials': {},
    }

    # Run 6 configurations: {N=2, N=4, N=8} x {dense, sparse}
    # But focus on N=8 dense vs N=8 sparse vs N=2 sparse (the core comparison)
    configs = [
        # (n_experts, top_k, dense_backprop, label)
        (2, 2, False, "N2_sparse"),       # baseline: N=2, sparse (parent regime)
        (8, 2, False, "N8_sparse"),       # N=8 with standard top-k backprop
        (8, 2, True, "N8_dense"),         # N=8 with dense backprop (hypothesis)
        (4, 2, False, "N4_sparse"),       # intermediate: N=4, sparse
        (4, 2, True, "N4_dense"),         # intermediate: N=4, dense
    ]

    for n_exp, k, dense, label in configs:
        if verbose:
            print(f"\n{'='*70}")
            print(f"CONFIG: {label} (N={n_exp}, k={k}, {'dense' if dense else 'sparse'} backprop)")
            print(f"{'='*70}")

        trials = []
        for target_cos in test_cosines:
            trial = run_trial(
                target_cos, base_model, deltas_a, deltas_b,
                train_datasets, joint_val, joint_val_loss, V,
                dense_backprop=dense, n_experts=n_exp, top_k=k,
                seed=seed
            )
            trials.append(trial)
        results['trials'][label] = trials

    return results


# -- Analysis ---------------------------------------------------------------

def analyze_results(all_experiments):
    """Analyze: does dense backprop restore N=2-level gradients?"""
    print(f"\n\n{'='*80}")
    print("DENSE BACKPROP CALIBRATION ANALYSIS")
    print(f"{'='*80}")

    # Aggregate across seeds per config
    configs = {}
    for exp in all_experiments:
        for label, trials in exp['trials'].items():
            if label not in configs:
                configs[label] = {}
            for trial in trials:
                cos = trial['target_cos']
                if cos not in configs[label]:
                    configs[label][cos] = []
                configs[label][cos].append(trial)

    # === Summary Table ===
    print(f"\n{'='*80}")
    print("GRADIENT MAGNITUDE COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Config':<15} | {'Cos':>5} | {'MeanGrad':>10} | {'EarlyGrad':>10} | "
          f"{'FinalVL':>8} | {'vs Joint':>9}")
    print("-" * 75)

    for label in ["N2_sparse", "N4_sparse", "N4_dense", "N8_sparse", "N8_dense"]:
        if label not in configs:
            continue
        for cos in sorted(configs[label].keys()):
            trials = configs[label][cos]
            m_grad = statistics.mean([t['mean_grad_norm'] for t in trials])
            m_early = statistics.mean([t['mean_early_grad'] for t in trials])
            m_val = statistics.mean([t['final_val_loss'] for t in trials])
            m_vj = statistics.mean([t['vs_joint_pct'] for t in trials])
            print(f"{label:<15} | {cos:>5.1f} | {m_grad:>10.6f} | {m_early:>10.6f} | "
                  f"{m_val:>8.4f} | {m_vj:>+8.1f}%")
        print()

    # === KC1: Gradient magnitude gap closure ===
    print(f"\n{'='*80}")
    print("KC1: GRADIENT MAGNITUDE GAP CLOSURE")
    print(f"{'='*80}")

    # For each cosine level, compute:
    # gap_sparse = |grad_N2 - grad_N8_sparse|
    # gap_dense = |grad_N2 - grad_N8_dense|
    # closure = 1 - gap_dense / gap_sparse
    # Need >50% closure

    closure_ratios = []
    for cos in sorted(set(configs.get("N2_sparse", {}).keys()) &
                      set(configs.get("N8_sparse", {}).keys()) &
                      set(configs.get("N8_dense", {}).keys())):
        n2_grad = statistics.mean([t['mean_grad_norm'] for t in configs["N2_sparse"][cos]])
        n8s_grad = statistics.mean([t['mean_grad_norm'] for t in configs["N8_sparse"][cos]])
        n8d_grad = statistics.mean([t['mean_grad_norm'] for t in configs["N8_dense"][cos]])

        gap_sparse = abs(n2_grad - n8s_grad)
        gap_dense = abs(n2_grad - n8d_grad)
        closure = 1 - gap_dense / gap_sparse if gap_sparse > 1e-10 else 0.0

        n2_early = statistics.mean([t['mean_early_grad'] for t in configs["N2_sparse"][cos]])
        n8s_early = statistics.mean([t['mean_early_grad'] for t in configs["N8_sparse"][cos]])
        n8d_early = statistics.mean([t['mean_early_grad'] for t in configs["N8_dense"][cos]])

        early_gap_sparse = abs(n2_early - n8s_early)
        early_gap_dense = abs(n2_early - n8d_early)
        early_closure = 1 - early_gap_dense / early_gap_sparse if early_gap_sparse > 1e-10 else 0.0

        print(f"\n  cos={cos:.1f}:")
        print(f"    N=2 sparse grad:  {n2_grad:.6f}  (early: {n2_early:.6f})")
        print(f"    N=8 sparse grad:  {n8s_grad:.6f}  (early: {n8s_early:.6f})")
        print(f"    N=8 dense grad:   {n8d_grad:.6f}  (early: {n8d_early:.6f})")
        print(f"    Gap closure (mean): {closure:.1%}")
        print(f"    Gap closure (early): {early_closure:.1%}")

        closure_ratios.append(closure)

    mean_closure = statistics.mean(closure_ratios) if closure_ratios else 0.0
    kc1_pass = mean_closure > 0.50

    print(f"\n  Mean gap closure: {mean_closure:.1%}")
    print(f"  KC1 threshold: >50%")
    print(f"  KC1: {'PASS' if kc1_pass else 'KILL'}")

    # === KC2: Convergence Speed ===
    print(f"\n{'='*80}")
    print("KC2: CONVERGENCE SPEED COMPARISON")
    print(f"{'='*80}")

    # Measure convergence: how many steps to reach a target quality?
    # Target: N=8 sparse final loss (achieved at step 300)
    # Dense should reach this in fewer steps

    speedup_ratios = []
    for cos in sorted(set(configs.get("N8_sparse", {}).keys()) &
                      set(configs.get("N8_dense", {}).keys())):
        sparse_trials = configs["N8_sparse"][cos]
        dense_trials = configs["N8_dense"][cos]

        # Target: mean final loss of sparse
        target_loss = statistics.mean([t['final_val_loss'] for t in sparse_trials])

        # Find step at which dense reaches this loss
        dense_conv_steps = []
        sparse_conv_steps = []

        for t in dense_trials:
            conv_step = None
            for step, val_loss in t['loss_curve']:
                if val_loss <= target_loss:
                    conv_step = step
                    break
            dense_conv_steps.append(conv_step if conv_step else MAX_CAL_STEPS)

        for t in sparse_trials:
            conv_step = None
            for step, val_loss in t['loss_curve']:
                if val_loss <= target_loss:
                    conv_step = step
                    break
            sparse_conv_steps.append(conv_step if conv_step else MAX_CAL_STEPS)

        mean_dense_steps = statistics.mean(dense_conv_steps)
        mean_sparse_steps = statistics.mean(sparse_conv_steps)
        speedup = mean_sparse_steps / mean_dense_steps if mean_dense_steps > 0 else 1.0

        print(f"\n  cos={cos:.1f}:")
        print(f"    Target loss (N8 sparse final): {target_loss:.4f}")
        print(f"    N8 sparse convergence: {mean_sparse_steps:.0f} steps")
        print(f"    N8 dense convergence:  {mean_dense_steps:.0f} steps")
        print(f"    Speedup: {speedup:.2f}x")

        speedup_ratios.append(speedup)

    mean_speedup = statistics.mean(speedup_ratios) if speedup_ratios else 1.0
    kc2_pass = mean_speedup >= 2.0

    print(f"\n  Mean convergence speedup: {mean_speedup:.2f}x")
    print(f"  KC2 threshold: >=2x")
    print(f"  KC2: {'PASS' if kc2_pass else 'KILL'}")

    # === Gradient Profile Shape ===
    print(f"\n{'='*80}")
    print("GRADIENT PROFILE SHAPE COMPARISON")
    print(f"{'='*80}")

    # Does dense backprop at N=8 restore the N=2 gradient-vs-cosine shape?
    for label_pair in [("N2_sparse", "N8_dense"), ("N2_sparse", "N8_sparse")]:
        a_label, b_label = label_pair
        if a_label not in configs or b_label not in configs:
            continue

        common_cos = sorted(set(configs[a_label].keys()) & set(configs[b_label].keys()))
        if len(common_cos) < 3:
            continue

        a_grads = [statistics.mean([t['mean_grad_norm'] for t in configs[a_label][c]])
                   for c in common_cos]
        b_grads = [statistics.mean([t['mean_grad_norm'] for t in configs[b_label][c]])
                   for c in common_cos]

        # Normalize for shape comparison
        max_a = max(a_grads) if a_grads else 1
        max_b = max(b_grads) if b_grads else 1
        norm_a = [g / max_a for g in a_grads]
        norm_b = [g / max_b for g in b_grads]

        r2_shape, r_shape, _ = compute_r_squared(norm_a, norm_b)
        print(f"\n  {a_label} vs {b_label}:")
        print(f"    Cosines: {common_cos}")
        print(f"    {a_label} normalized: {[f'{g:.3f}' for g in norm_a]}")
        print(f"    {b_label} normalized: {[f'{g:.3f}' for g in norm_b]}")
        print(f"    Shape r^2: {r2_shape:.4f}")

    # === N=4 comparison ===
    if "N4_sparse" in configs and "N4_dense" in configs:
        print(f"\n{'='*80}")
        print("N=4 DENSE vs SPARSE (intermediate check)")
        print(f"{'='*80}")

        for cos in sorted(set(configs["N4_sparse"].keys()) & set(configs["N4_dense"].keys())):
            n4s = statistics.mean([t['mean_grad_norm'] for t in configs["N4_sparse"][cos]])
            n4d = statistics.mean([t['mean_grad_norm'] for t in configs["N4_dense"][cos]])
            ratio = n4d / n4s if n4s > 1e-10 else 0
            print(f"  cos={cos:.1f}: sparse={n4s:.6f}, dense={n4d:.6f}, ratio={ratio:.2f}x")

    # === Overall Verdict ===
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print(f"{'='*80}")

    if kc1_pass and kc2_pass:
        print(f"\n  PROVEN: Dense backprop restores gradient strength AND speeds convergence")
        status = "proven"
    elif kc1_pass and not kc2_pass:
        print(f"\n  PARTIAL: Dense backprop restores gradients but convergence speedup < 2x")
        print(f"  Gradients are larger but don't translate to proportionally faster learning")
        status = "partial"
    elif not kc1_pass and kc2_pass:
        print(f"\n  PARTIAL: Convergence is faster but gradient gap not closed by >50%")
        status = "partial"
    else:
        print(f"\n  KILLED: Dense backprop does not sufficiently restore gradient strength")
        print(f"  or speed convergence. The k/N dilution is not the gradient bottleneck.")
        status = "killed"

    return {
        'mean_closure': mean_closure,
        'mean_speedup': mean_speedup,
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
