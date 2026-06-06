"""
Brainstacks — Loss Functions (MLX)

Paper: https://arxiv.org/abs/2604.01152
Implements: Task loss + auxiliary MoE load balancing (§3.1),
            Meta-router BCE + confidence margin (§3.6)
"""

import mlx.core as mx
import mlx.nn as nn


def brainstacks_loss(
    logits: mx.array,
    targets: mx.array,
    aux_loss: mx.array,
    aux_coeff: float = 0.01,
) -> mx.array:
    """Combined task loss + MoE auxiliary load balancing loss.

    §3.1 — "The total training loss is L = L_task + λ_aux · L_aux with coefficient λ_aux = 0.01"

    Args:
        logits: model output logits — shape: (batch, seq_len, vocab)
        targets: target token ids — shape: (batch, seq_len)
        aux_loss: load balancing loss from MoE routing — scalar
        aux_coeff: §3.1 — λ_aux = 0.01

    Returns:
        total_loss: L_task + λ_aux · L_aux — scalar
    """
    # Standard cross-entropy for language modeling
    task_loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )

    return task_loss + aux_coeff * aux_loss


def router_loss(
    predictions: mx.array,
    targets: mx.array,
    confidence_margin: float = 0.1,
) -> mx.array:
    """Meta-router training loss.

    §3.6 — "BCE loss with a confidence margin penalty that pushes predictions
    toward clear yes/no decisions"

    Args:
        predictions: sigmoid probabilities — shape: (batch, num_domains)
        targets: discovered + label blend targets — shape: (batch, num_domains)
        confidence_margin: [UNSPECIFIED] — penalty strength for uncertain predictions

    Returns:
        loss: BCE + confidence penalty — scalar
    """
    # §3.6 — BCE loss
    eps = 1e-7
    bce = -(targets * mx.log(predictions + eps) +
            (1 - targets) * mx.log(1 - predictions + eps))
    bce = mx.mean(bce)

    # §3.6 — "confidence margin penalty that pushes predictions toward clear yes/no decisions"
    # Penalize predictions near 0.5 (uncertain)
    confidence_penalty = mx.mean(
        mx.maximum(0.0, confidence_margin - mx.abs(predictions - 0.5))
    )

    return bce + confidence_penalty


def blend_router_targets(
    discovered: mx.array,
    label: mx.array,
    blend_discovered: float = 0.8,
    blend_label: float = 0.2,
    reasoning_idx: int = 4,
    reasoning_boost: float = 0.5,
) -> mx.array:
    """Blend discovered and label targets for meta-router training.

    §3.6 — "The final training target blends the discovered target (80%)
    with the prior label target (20%)"
    §3.6 — "if adding reasoning reduces loss at all, set its target to 0.5"

    Args:
        discovered: empirically discovered targets — shape: (batch, num_domains)
        label: prior domain label targets — shape: (batch, num_domains)
        blend_discovered: §3.6 — 0.8
        blend_label: §3.6 — 0.2
        reasoning_idx: index of reasoning domain (default 4)
        reasoning_boost: §3.6 — 0.5 soft boost value

    Returns:
        blended: final training targets — shape: (batch, num_domains)
    """
    blended = blend_discovered * discovered + blend_label * label

    # §3.6 — Reasoning soft-boost
    reasoning_col = blended[:, reasoning_idx:reasoning_idx + 1]
    boosted = mx.where(reasoning_col > 0, reasoning_boost, reasoning_col)
    blended = mx.concatenate([
        blended[:, :reasoning_idx],
        boosted,
        blended[:, reasoning_idx + 1:],
    ], axis=-1)

    return blended
