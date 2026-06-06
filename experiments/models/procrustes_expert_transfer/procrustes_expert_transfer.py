"""
Procrustes Expert Transfer: Can LoRA experts trained on one base model
be transferred to an independently-trained base model via Procrustes alignment?

Parent: zero_shot_base_transfer (proven -- same skeleton, SVD perturbation)
  - That experiment tested transfer within the same weight space lineage.
  - This experiment tests transfer across INDEPENDENTLY trained models.

Hypothesis: Procrustes alignment finds per-layer rotations R_l such that
W_B_l ~ R_l @ W_A_l, enabling expert transfer: dW_B = R_l @ dW_A @ R_l^T.
Cost: O(d^3) once for alignment, then O(d*r) per expert.

Kill criteria:
  K1: Transferred expert PPL >20% worse than natively-trained expert
  K2: Procrustes alignment error >5% (weight spaces too different)

Architecture: Reuses base_free_composition GPT/LoRA infrastructure.
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse parent infrastructure
from experiments.models.base_free_composition.base_free_composition import (
    GPT,
    LoRALinear,
    LoRAGPT,
    CharTokenizer,
    CharDataset,
    load_names,
    domain_split,
    compute_pairwise_cosine,
    train_gpt,
    evaluate_model,
    train_lora_expert,
)


def _json_default(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Procrustes Alignment ─────────────────────────────────────────────────────


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> tuple:
    """Solve the orthogonal Procrustes problem: find R = argmin ||R @ A - B||_F
    subject to R^T @ R = I.

    Uses scipy's implementation which computes SVD of B @ A^T.

    Args:
        A: Source matrix, shape (d_out, d_in) or (d, d)
        B: Target matrix, shape (d_out, d_in) or (d, d)

    Returns:
        R: Orthogonal rotation matrix, shape (d_out, d_out)
        scale: Not used (set to 1.0 for orthogonal Procrustes)
        residual: ||R @ A - B||_F / ||B||_F (relative alignment error)
    """
    # SVD of B @ A^T
    M = B @ A.T
    U, S, Vt = np.linalg.svd(M)

    # R = U @ V^T (closest orthogonal matrix)
    R = U @ Vt

    # Ensure proper rotation (det = +1), not reflection
    if np.linalg.det(R) < 0:
        # Flip the sign of the last column of U
        U[:, -1] *= -1
        R = U @ Vt

    # Compute relative residual
    residual_abs = np.linalg.norm(R @ A - B, 'fro')
    norm_B = np.linalg.norm(B, 'fro')
    residual_rel = residual_abs / (norm_B + 1e-12)

    return R, residual_rel


def compute_per_layer_alignment(
    state_A: dict,
    state_B: dict,
    layer_keys: list = None,
) -> dict:
    """Compute per-layer Procrustes alignment from model A to model B.

    For each weight matrix W, find R such that R @ W_A ~ W_B.

    For MLP layers with shapes (d_out, d_in):
        - fc1: (4d, d) -> left rotation R_out (4d x 4d) and right rotation R_in (d x d)
        - fc2: (d, 4d) -> left rotation R_out (d x d) and right rotation R_in (4d x 4d)

    We use a simplified approach: align using the full weight matrix.
    For W_B ~ R_left @ W_A @ R_right^T, we solve two coupled Procrustes:
        1. Fix R_right = I, solve for R_left
        2. Fix R_left, solve for R_right
        3. Iterate (alternating Procrustes)

    For simplicity in this micro experiment, we use the approach from
    Git Re-Basin: align activation spaces, not individual weight matrices.
    We align the hidden representations layer-by-layer.

    Simplified approach: For each linear layer with shape (d_out, d_in),
    compute R_left such that R_left @ W_A ~ W_B (left-multiply alignment).
    This aligns the output space. The input space alignment is inherited
    from the previous layer.

    Returns:
        dict mapping layer key -> {R: rotation matrix, residual: alignment error}
    """
    if layer_keys is None:
        # Auto-detect weight matrices
        layer_keys = [k for k in state_A if 'weight' in k
                      and state_A[k].dim() == 2
                      and state_A[k].shape == state_B[k].shape]

    alignments = {}
    for key in layer_keys:
        W_A = state_A[key].numpy()
        W_B = state_B[key].numpy()

        R, residual = orthogonal_procrustes(W_A, W_B)
        alignments[key] = {
            'R': R,
            'residual': residual,
            'shape': W_A.shape,
        }

    return alignments


def compute_activation_alignment(
    model_A: GPT,
    model_B: GPT,
    dataset: CharDataset,
    n_batches: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Compute per-layer activation-space Procrustes alignment.

    This is more principled than weight-space alignment because it
    accounts for the actual data distribution. For each layer l, we
    collect activations H_A^l and H_B^l, then find R_l such that
    R_l @ H_A^l ~ H_B^l.

    This follows the approach in model stitching / representation
    similarity analysis.

    Returns:
        dict mapping layer_idx -> {R: (d, d), residual: float}
    """
    model_A.eval()
    model_B.eval()

    # Collect activations per layer using hooks
    acts_A = {}
    acts_B = {}

    def make_hook(storage, layer_idx):
        def hook(module, input, output):
            # output shape: (B, T, d)
            if layer_idx not in storage:
                storage[layer_idx] = []
            # Flatten to (B*T, d)
            storage[layer_idx].append(output.detach().reshape(-1, output.shape[-1]))
        return hook

    hooks_A = []
    hooks_B = []
    for i, layer in enumerate(model_A.layers):
        hooks_A.append(layer.register_forward_hook(make_hook(acts_A, i)))
    for i, layer in enumerate(model_B.layers):
        hooks_B.append(layer.register_forward_hook(make_hook(acts_B, i)))

    rng = random.Random(0)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = dataset.get_batch(batch_size, rng=rng, device=device)
            model_A(x)
            model_B(x)

    for h in hooks_A + hooks_B:
        h.remove()

    # Compute Procrustes alignment per layer
    alignments = {}
    for layer_idx in sorted(acts_A.keys()):
        H_A = torch.cat(acts_A[layer_idx], dim=0).numpy()  # (N_samples, d)
        H_B = torch.cat(acts_B[layer_idx], dim=0).numpy()

        # Procrustes on activation matrices (transposed for our convention)
        # We want R such that H_A @ R^T ~ H_B, equivalently R @ H_A^T ~ H_B^T
        R, residual = orthogonal_procrustes(H_A.T, H_B.T)

        alignments[layer_idx] = {
            'R': R,  # shape (d, d)
            'residual': residual,
            'n_samples': H_A.shape[0],
        }

    return alignments


def transform_expert_deltas(
    expert_deltas: list,
    alignments: dict,
    n_embd: int,
) -> list:
    """Transform LoRA expert deltas from model A's space to model B's space.

    For a LoRA delta dW with shape (d_in, d_out) applied as:
        y = (W_base + dW^T) @ x  (note: dW stored as (in, out), weight is (out, in))

    The transformation depends on where in the network the weight sits:
    - fc1 (d -> 4d): input space is hidden dim d, output is intermediate 4d
    - fc2 (4d -> d): input space is intermediate 4d, output is hidden dim d

    For activation-space alignment at layer l with rotation R_l (d x d):
    The hidden state at layer l is rotated: h_B = R_l @ h_A
    So for fc1 at layer l: input rotated by R_l, output rotated by identity
    (intermediate space is private to the layer)
    For fc2 at layer l: input is identity, output rotated by R_{l+1}
    (or R_l if we treat the residual connection)

    Simplified approach for MLP-only LoRA:
    Since the residual stream is rotated by R_l at layer l,
    and the MLP computes: mlp(x) = fc2(relu(fc1(x))),
    the full transformation for the MLP delta is:

        dW_fc1_B = dW_fc1_A @ R_l^T        (rotate input)
        dW_fc2_B = R_l @ dW_fc2_A          (rotate output, intermediate unchanged)

    Wait -- we need to be more careful. The LoRA delta is stored as (in_feat, out_feat)
    and applied as x @ delta (matrix multiply on the right). The linear weight W has
    shape (out, in) and computes W @ x^T.

    Let's trace through carefully:
    - fc1.weight: (4d, d), computes z = fc1.weight @ x (x is d-dim)
    - fc2.weight: (d, 4d), computes y = fc2.weight @ relu(z)

    If hidden state rotates by R (d x d): x_B = R @ x_A
    Then for fc1: z = W_fc1_A @ x_A = W_fc1_A @ R^T @ x_B
    So W_fc1_B = W_fc1_A @ R^T -> delta_fc1_B = delta_fc1_A @ R^T (in weight space)

    But delta is stored as (in, out) and added as delta^T to weight.
    weight_B = weight_A @ R^T, and delta^T is added to weight.
    So: (W + delta^T)_B = W_A @ R^T + delta^T_A @ R^T
    delta^T_B = delta^T_A @ R^T
    delta_B = R @ delta_A  (for fc1, in the (in, out) storage)

    Hmm, let me reconsider. The delta is stored as (in_features, out_features).
    It's applied as: output = base(x) + x @ A @ B * scale
    where A: (in, rank), B: (rank, out).
    The combined delta (in, out) gets transposed and added to weight:
        effective_weight = weight + delta.T  (since weight is (out, in))

    If the residual stream at layer l rotates by R_l:
    - fc1 input space rotated by R_l (d x d)
    - fc1 output space is intermediate (4d), unaligned (different for each model)
    - fc2 input space is intermediate (4d), same caveat
    - fc2 output contributes to residual, rotated by R_l (via residual connection)

    The intermediate space between fc1 and fc2 is model-specific and NOT
    aligned by our rotation. This is a fundamental limitation.

    For the Procrustes transfer to work well, we need the intermediate
    representations to be similar enough. Two approaches:
    1. Weight-space alignment: align each weight matrix independently
    2. End-to-end MLP alignment: treat the MLP as a unit

    We implement BOTH and compare:
    - Method 1 (per-weight): R_left @ W_A for each weight independently
    - Method 2 (activation): align hidden states, transform fc1 input and fc2 output

    Args:
        expert_deltas: list of (layer_idx, fc_name, delta_tensor) from model A
        alignments: dict from compute_activation_alignment or per-weight alignment
        n_embd: model embedding dimension

    Returns:
        transformed deltas in the same format
    """
    transformed = []

    for layer_idx, fc_name, delta in expert_deltas:
        # delta shape: (in_features, out_features)
        d = delta.numpy().copy()

        if isinstance(list(alignments.keys())[0], int):
            # Activation-space alignment: R per layer
            R_l = alignments[layer_idx]['R']  # (d, d)

            if fc_name == "fc1":
                # fc1: (d, 4d) -- rotate input space
                # delta stored as (in=d, out=4d)
                # In weight space: W_B = W_A @ R^T means delta.T -> delta.T @ R^T
                # So delta -> R @ delta
                d_transformed = R_l @ d
            elif fc_name == "fc2":
                # fc2: (4d, d) -- rotate output space
                # The output contributes to residual stream via x + mlp(x)
                # So the output of fc2 should be rotated by R_l
                # W_fc2 has shape (d, 4d), computes y = W @ z
                # y_B = R_l @ y_A = R_l @ W_A @ z_A
                # So W_B = R_l @ W_A, delta.T_B = R_l @ delta.T_A
                # delta_B = delta_A @ R_l^T
                d_transformed = d @ R_l.T
            else:
                d_transformed = d
        else:
            # Per-weight alignment
            key = f"layers.{layer_idx}.mlp.{fc_name}.weight"
            if key in alignments:
                R = alignments[key]['R']
                # R @ W_A ~ W_B, so R @ delta.T ~ delta_B.T
                # delta_B = delta_A @ R.T
                d_transformed = d @ R.T
            else:
                d_transformed = d

        transformed.append((layer_idx, fc_name, torch.tensor(d_transformed, dtype=delta.dtype)))

    return transformed


def apply_deltas_to_base(
    base_gpt: GPT,
    deltas: list,
) -> nn.Module:
    """Apply expert deltas to a base model (same as zero_shot_base_transfer)."""
    model = copy.deepcopy(base_gpt)
    model.eval()
    for layer_idx, fc_name, delta in deltas:
        layer = model.layers[layer_idx]
        linear = getattr(layer.mlp, fc_name)
        with torch.no_grad():
            linear.weight.add_(delta.T)
    return model


# ── Experiment ────────────────────────────────────────────────────────────────


@dataclass
class TransferResult:
    """Results for one transfer method."""
    method: str
    expert_losses: list          # per-expert loss on model B
    mean_expert_loss: float
    native_expert_losses: list   # per-expert loss of natively-trained on B
    mean_native_loss: float
    loss_ratios: list            # per-expert transferred/native
    mean_loss_ratio: float
    alignment_errors: list       # per-layer Procrustes residuals
    mean_alignment_error: float
    k1_violated: bool            # >20% worse
    k2_violated: bool            # alignment error >5%


@dataclass
class ExperimentResults:
    """Complete Procrustes expert transfer results."""
    seed: int
    config: dict

    # Baselines
    model_A_val_loss: float
    model_B_val_loss: float
    native_A_expert_losses: list   # experts trained on A, eval on A
    native_B_expert_losses: list   # experts trained on B, eval on B

    # Transfer methods
    naive_transfer: dict           # zero-shot (no alignment)
    procrustes_weight: dict        # per-weight Procrustes
    procrustes_activation: dict    # activation-based Procrustes

    # Timing
    pretrain_time: float
    expert_train_time: float
    alignment_time: float
    eval_time: float

    verdict: str


def run_experiment(
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    block_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    pretrain_steps: int = 1000,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    device: str = "cpu",
) -> ExperimentResults:
    """Run the Procrustes expert transfer experiment.

    Protocol:
      1. Train Model A (seed) and Model B (seed+1000) independently
      2. Train N experts on Model A
      3. Train N experts on Model B (native baseline)
      4. Compute Procrustes alignment A -> B (weight-space and activation-space)
      5. Transform A's experts to B's space via Procrustes
      6. Evaluate: transferred experts on B vs native experts on B

    Three transfer methods compared:
      - Naive: apply A's deltas directly to B (no alignment)
      - Procrustes (weight): per-weight orthogonal alignment
      - Procrustes (activation): activation-space alignment
    """
    seed_A = seed
    seed_B = seed + 1000  # Independent model

    print("=" * 72)
    print("PROCRUSTES EXPERT TRANSFER EXPERIMENT")
    print(f"Config: d={n_embd}, h={n_head}, L={n_layer}, r={lora_rank}")
    print(f"Model A seed: {seed_A}, Model B seed: {seed_B}")
    print(f"Experts: {n_experts} x {expert_train_steps} steps")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:n_experts]
    print(f"\nDomains: {domain_names}")

    # Global train/val split
    rng_split = random.Random(seed)
    docs_copy = list(docs)
    rng_split.shuffle(docs_copy)
    split_idx = int(len(docs_copy) * 0.9)
    pretrain_train_ds = CharDataset(docs_copy[:split_idx], tokenizer, block_size)
    pretrain_val_ds = CharDataset(docs_copy[split_idx:], tokenizer, block_size)

    # Domain datasets (shared across both models)
    domain_datasets = {}
    for i, domain in enumerate(domain_names):
        domain_docs = domains[domain]
        rng_domain = random.Random(seed + 2000 + i)
        domain_docs_shuffled = list(domain_docs)
        rng_domain.shuffle(domain_docs_shuffled)
        n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
        train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, block_size)
        val_ds = CharDataset(
            domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
            else domain_docs_shuffled, tokenizer, block_size
        )
        domain_datasets[domain] = (train_ds, val_ds)

    # ── Phase 1: Train two independent base models ─────────────────────
    print("\n--- Phase 1: Training Independent Base Models ---")
    t0_pretrain = time.time()

    torch.manual_seed(seed_A)
    model_A = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
    train_gpt(model_A, pretrain_train_ds, steps=pretrain_steps,
              batch_size=batch_size, lr=lr, seed=seed_A, device=device)
    val_A = evaluate_model(model_A, pretrain_val_ds, batch_size, device=device)
    print(f"  Model A val loss: {val_A:.4f}")

    torch.manual_seed(seed_B)
    model_B = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
    train_gpt(model_B, pretrain_train_ds, steps=pretrain_steps,
              batch_size=batch_size, lr=lr, seed=seed_B, device=device)
    val_B = evaluate_model(model_B, pretrain_val_ds, batch_size, device=device)
    print(f"  Model B val loss: {val_B:.4f}")

    pretrain_time = time.time() - t0_pretrain

    state_A = {k: v.clone() for k, v in model_A.state_dict().items()}
    state_B = {k: v.clone() for k, v in model_B.state_dict().items()}

    # ── Phase 2: Train experts on both models ──────────────────────────
    print("\n--- Phase 2: Training Experts ---")
    t0_expert = time.time()

    experts_A = []  # (domain, deltas, val_loss, val_ds)
    experts_B = []

    for i, domain in enumerate(domain_names):
        train_ds, val_ds = domain_datasets[domain]

        # Expert on Model A
        deltas_A, val_A_exp = train_lora_expert(
            model_A, train_ds, val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed_A + i, device=device,
        )
        experts_A.append((domain, deltas_A, val_A_exp, val_ds))
        print(f"  Expert A.{i} ({domain}): val={val_A_exp:.4f}")

        # Expert on Model B (native baseline)
        deltas_B, val_B_exp = train_lora_expert(
            model_B, train_ds, val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed_B + i, device=device,
        )
        experts_B.append((domain, deltas_B, val_B_exp, val_ds))
        print(f"  Expert B.{i} ({domain}): val={val_B_exp:.4f}")

    expert_train_time = time.time() - t0_expert

    native_A_losses = [e[2] for e in experts_A]
    native_B_losses = [e[2] for e in experts_B]

    # ── Phase 3: Compute Procrustes alignments ─────────────────────────
    print("\n--- Phase 3: Procrustes Alignment ---")
    t0_align = time.time()

    # Method 1: Per-weight Procrustes
    weight_keys = [k for k in state_A if 'weight' in k
                   and state_A[k].dim() == 2
                   and 'mlp' in k]  # Only MLP weights (where LoRA is)
    weight_alignments = compute_per_layer_alignment(state_A, state_B, weight_keys)

    print("  Per-weight alignment residuals:")
    for key, info in weight_alignments.items():
        print(f"    {key}: residual = {info['residual']:.4f} ({info['shape']})")

    # Method 2: Activation-space Procrustes
    act_alignments = compute_activation_alignment(
        model_A, model_B, pretrain_train_ds,
        n_batches=10, batch_size=batch_size, device=device,
    )

    print("  Activation-space alignment residuals:")
    for layer_idx, info in act_alignments.items():
        print(f"    Layer {layer_idx}: residual = {info['residual']:.4f} "
              f"(n_samples={info['n_samples']})")

    alignment_time = time.time() - t0_align

    # ── Phase 4: Transfer and evaluate ─────────────────────────────────
    print("\n--- Phase 4: Transfer and Evaluate ---")
    t0_eval = time.time()

    def evaluate_transfer(method_name, transform_fn):
        """Evaluate a transfer method across all experts."""
        losses = []
        ratios = []
        for i, (domain, deltas_A, _, val_ds) in enumerate(experts_A):
            # Transform deltas
            transformed = transform_fn(deltas_A)
            # Apply to model B
            model_with_expert = apply_deltas_to_base(model_B, transformed)
            model_with_expert.to(device)
            loss = evaluate_model(model_with_expert, val_ds, batch_size, device=device)
            losses.append(loss)
            ratio = loss / (native_B_losses[i] + 1e-12)
            ratios.append(ratio)
            print(f"    Expert {i} ({domain}): loss={loss:.4f}, "
                  f"native={native_B_losses[i]:.4f}, ratio={ratio:.3f}")
        return losses, ratios

    # Method 0: Naive (no alignment)
    print(f"\n  Naive transfer (no alignment):")
    naive_losses, naive_ratios = evaluate_transfer(
        "naive", lambda deltas: deltas)  # No transformation

    # Method 1: Per-weight Procrustes
    print(f"\n  Per-weight Procrustes transfer:")
    pw_losses, pw_ratios = evaluate_transfer(
        "procrustes_weight",
        lambda deltas: transform_expert_deltas(deltas, weight_alignments, n_embd))

    # Method 2: Activation-space Procrustes
    print(f"\n  Activation-space Procrustes transfer:")
    act_losses, act_ratios = evaluate_transfer(
        "procrustes_activation",
        lambda deltas: transform_expert_deltas(deltas, act_alignments, n_embd))

    eval_time = time.time() - t0_eval

    # ── Phase 5: Kill Criteria ─────────────────────────────────────────
    print("\n--- Phase 5: Kill Criteria ---")

    def assess_method(name, losses, ratios, alignment_errors):
        mean_loss = sum(losses) / len(losses)
        mean_native = sum(native_B_losses) / len(native_B_losses)
        mean_ratio = sum(ratios) / len(ratios)
        mean_align_err = sum(alignment_errors) / len(alignment_errors) if alignment_errors else 0

        k1 = mean_ratio > 1.20  # >20% worse
        k2 = mean_align_err > 0.05  # >5% alignment error

        print(f"\n  {name}:")
        print(f"    Mean transferred loss: {mean_loss:.4f}")
        print(f"    Mean native loss:      {mean_native:.4f}")
        print(f"    Mean loss ratio:       {mean_ratio:.4f} "
              f"({'KILLED' if k1 else 'SURVIVES'} -- threshold 1.20)")
        if alignment_errors:
            print(f"    Mean alignment error:  {mean_align_err:.4f} "
                  f"({'KILLED' if k2 else 'SURVIVES'} -- threshold 0.05)")

        return TransferResult(
            method=name,
            expert_losses=losses,
            mean_expert_loss=mean_loss,
            native_expert_losses=native_B_losses,
            mean_native_loss=mean_native,
            loss_ratios=ratios,
            mean_loss_ratio=mean_ratio,
            alignment_errors=alignment_errors,
            mean_alignment_error=mean_align_err,
            k1_violated=k1,
            k2_violated=k2,
        )

    naive_result = assess_method(
        "Naive (no alignment)", naive_losses, naive_ratios, [])

    pw_align_errs = [info['residual'] for info in weight_alignments.values()]
    pw_result = assess_method(
        "Procrustes (per-weight)", pw_losses, pw_ratios, pw_align_errs)

    act_align_errs = [info['residual'] for info in act_alignments.values()]
    act_result = assess_method(
        "Procrustes (activation)", act_losses, act_ratios, act_align_errs)

    # Overall verdict
    # The hypothesis succeeds if at least one Procrustes method beats naive
    # AND passes kill criteria
    best_procrustes = min(pw_result.mean_loss_ratio, act_result.mean_loss_ratio)
    improvement_over_naive = naive_result.mean_loss_ratio - best_procrustes

    if pw_result.k1_violated and act_result.k1_violated:
        verdict = "KILLED (K1: both Procrustes methods >20% worse than native)"
    elif pw_result.k2_violated and act_result.k2_violated:
        verdict = "KILLED (K2: alignment error >5% for both methods)"
    elif best_procrustes <= 1.20 and improvement_over_naive > 0.01:
        verdict = "SURVIVES (Procrustes transfer works and improves over naive)"
    elif best_procrustes <= 1.20:
        verdict = "SURVIVES (Procrustes transfer within threshold, minimal improvement over naive)"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  Improvement of Procrustes over naive: {improvement_over_naive:.4f}")
    print(f"  VERDICT: {verdict}")

    # ── Summary Table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print(f"{'='*72}")
    print(f"\n{'Method':30s} | {'Mean Loss':>10s} | {'Ratio':>8s} | {'Align Err':>10s} | K1  | K2")
    print("-" * 80)
    print(f"{'Native on B (baseline)':30s} | {sum(native_B_losses)/len(native_B_losses):10.4f} | {'1.000':>8s} | {'N/A':>10s} | N/A | N/A")
    for r in [naive_result, pw_result, act_result]:
        ae_str = f"{r.mean_alignment_error:.4f}" if r.alignment_errors else "N/A"
        k1_str = "KILL" if r.k1_violated else "ok"
        k2_str = "KILL" if r.k2_violated else ("ok" if r.alignment_errors else "N/A")
        print(f"{r.method:30s} | {r.mean_expert_loss:10.4f} | {r.mean_loss_ratio:8.3f} | {ae_str:>10s} | {k1_str:3s} | {k2_str:3s}")

    # ── Save results ───────────────────────────────────────────────────
    config = {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "lora_rank": lora_rank,
        "lora_alpha": lora_alpha, "pretrain_steps": pretrain_steps,
        "expert_train_steps": expert_train_steps, "n_experts": n_experts,
        "seed_A": seed_A, "seed_B": seed_B,
    }

    results = ExperimentResults(
        seed=seed,
        config=config,
        model_A_val_loss=evaluate_model(model_A, pretrain_val_ds, batch_size, device=device),
        model_B_val_loss=evaluate_model(model_B, pretrain_val_ds, batch_size, device=device),
        native_A_expert_losses=native_A_losses,
        native_B_expert_losses=native_B_losses,
        naive_transfer=asdict(naive_result),
        procrustes_weight=asdict(pw_result),
        procrustes_activation=asdict(act_result),
        pretrain_time=pretrain_time,
        expert_train_time=expert_train_time,
        alignment_time=alignment_time,
        eval_time=eval_time,
        verdict=verdict,
    )

    output_path = os.path.join(os.path.dirname(__file__), f"results_seed_{seed}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2, default=_json_default)
    print(f"\nResults saved to {output_path}")

    return results


def run_multi_seed(seeds: list = None, **kwargs) -> dict:
    """Run experiment across multiple seeds."""
    if seeds is None:
        seeds = [42, 123, 7]

    all_results = []

    for s in seeds:
        print(f"\n{'='*72}")
        print(f"SEED {s}")
        print(f"{'='*72}")
        r = run_experiment(seed=s, **kwargs)
        all_results.append(r)

    # Aggregate
    print(f"\n{'='*72}")
    print("AGGREGATE RESULTS")
    print(f"{'='*72}")

    for method_name in ["naive_transfer", "procrustes_weight", "procrustes_activation"]:
        ratios = [getattr(r, method_name)["mean_loss_ratio"] for r in all_results]
        mean_ratio = sum(ratios) / len(ratios)
        std_ratio = (sum((x - mean_ratio)**2 for x in ratios) / len(ratios)) ** 0.5
        k1_any = any(getattr(r, method_name)["k1_violated"] for r in all_results)
        print(f"  {method_name:30s}: ratio = {mean_ratio:.4f} +/- {std_ratio:.4f} "
              f"{'KILLED' if k1_any else 'SURVIVES'}")

    verdicts = [r.verdict for r in all_results]
    survives = sum(1 for v in verdicts if "SURVIVES" in v)
    killed = sum(1 for v in verdicts if "KILLED" in v)

    if killed > 0 and survives == 0:
        overall = "KILLED"
    elif survives == len(verdicts):
        overall = "SURVIVES"
    elif survives > killed:
        overall = "SURVIVES (majority)"
    else:
        overall = "INCONCLUSIVE"

    print(f"\n  Per-seed verdicts: {verdicts}")
    print(f"  Overall: {overall}")

    aggregate = {
        "seeds": seeds,
        "per_seed_verdicts": verdicts,
        "overall_verdict": overall,
        "methods": {},
    }
    for method_name in ["naive_transfer", "procrustes_weight", "procrustes_activation"]:
        ratios = [getattr(r, method_name)["mean_loss_ratio"] for r in all_results]
        align_errs = [getattr(r, method_name)["mean_alignment_error"] for r in all_results]
        aggregate["methods"][method_name] = {
            "mean_ratio": sum(ratios) / len(ratios),
            "std_ratio": (sum((x - sum(ratios)/len(ratios))**2 for x in ratios) / len(ratios)) ** 0.5,
            "ratios_per_seed": ratios,
            "mean_alignment_error": sum(align_errs) / len(align_errs),
        }

    output_path = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=_json_default)
    print(f"\nAggregate saved to {output_path}")

    return aggregate


if __name__ == "__main__":
    run_multi_seed()
