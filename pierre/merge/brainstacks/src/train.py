"""
Brainstacks — Training Loops (MLX)

Paper: https://arxiv.org/abs/2604.01152
Implements: Algorithm 1 (Inner Loop: Residual Boosting, §3.3),
            Algorithm 2 (Outer Loop: Continual Domain Learning, §3.4)

This is a minimal training scaffold demonstrating the core algorithms.
Full reproduction requires a base model (e.g., TinyLlama or Gemma) loaded
via mlx-lm and domain-specific datasets.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model import BrainstacksConfig, MoELoRADelta, NullSpaceProjector, StackedMoELoRALayer


@dataclass
class TrainingConfig:
    """Training configuration from paper.

    All defaults from §4.3 unless marked [UNSPECIFIED].
    """

    batch_size: int = 4          # §4.3 — "batch=4"
    grad_accum: int = 4          # §4.3 — "grad_accum=4 (effective batch 16)"
    seq_len: int = 512           # §4.3 — "seq_len=512"
    lr: float = 1e-4             # §4.3 — "lr=1×10^-4"
    max_steps: int = 500         # §4.3 — "max_steps=500 per domain"
    max_steps_reasoning: int = 600  # §4.3 — "600 for reasoning"
    weight_decay: float = 0.01   # [UNSPECIFIED] — AdamW default
    aux_loss_coeff: float = 0.01 # §3.1 — "λ_aux = 0.01"


def inner_loop(
    stacked_layers: list[StackedMoELoRALayer],
    train_data,  # iterator yielding (input_ids, targets) batches
    val_data,    # iterator yielding (input_ids, targets) batches
    model_forward_fn,  # callable: (input_ids) -> (logits, total_aux_loss)
    max_rounds: int = 2,
    min_loss_delta: float = 0.002,
    spike_threshold: float = 0.1,
    patience: int = 4,
    training_config: Optional[TrainingConfig] = None,
) -> float:
    """Algorithm 1: Brainstacks Inner Loop — Residual Boosting.

    §3.3 — "Stack 1 learns the primary correction for the domain. After Stack 1
    is frozen, Stack 2 is added and trained on the same data, but now the loss
    landscape has changed because Stack 1's frozen contribution alters the
    model's output."

    Args:
        stacked_layers: all StackedMoELoRALayer instances in the model
        train_data: training data iterator
        val_data: validation data iterator
        model_forward_fn: forward pass through full model
        max_rounds: §3.3 — "up to 2 rounds"
        min_loss_delta: §3.3 — "minimum loss delta threshold (0.002)"
        spike_threshold: §3.3 — "spike_threshold=0.1"
        patience: §3.3 — "patience=4 evaluation steps"
        training_config: training hyperparameters

    Returns:
        final_val_loss: validation loss after all rounds
    """
    cfg = training_config or TrainingConfig()

    # Algorithm 1, line 1 — "ℓ_prev ← evaluate(M, D_val)"
    l_prev = _evaluate(model_forward_fn, val_data, cfg)

    # Algorithm 1, line 2 — "for m = 1 to R do"
    for m in range(max_rounds):
        # Algorithm 1, line 3 — "Add new trainable MoELoRADelta stack to each layer"
        for layer in stacked_layers:
            layer.add_new_stack()

        # Collect trainable parameters from active stacks only
        trainable_params = []
        for layer in stacked_layers:
            if layer.active_stack is not None:
                trainable_params.extend(layer.active_stack.trainable_parameters())

        optimizer = optim.AdamW(
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # §3.3 — BestStackCallback state
        best_loss = float("inf")
        best_weights = None
        patience_counter = 0

        # Algorithm 1, line 4 — "Train active stack on D for T steps"
        def loss_fn(model_params, x, y):
            logits, aux_loss = model_forward_fn(x, training=True)
            task_loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="mean",
            )
            return task_loss + cfg.aux_loss_coeff * aux_loss

        max_steps = cfg.max_steps
        step = 0
        for x_batch, y_batch in train_data:
            if step >= max_steps:
                break

            # Forward + backward + update
            loss, grads = nn.value_and_grad(loss_fn)(None, x_batch, y_batch)
            optimizer.update(trainable_params, grads)
            mx.eval(loss)
            step += 1

            # §3.3 — BestStackCallback: periodic validation
            if step % (max_steps // 10) == 0:
                val_loss = _evaluate(model_forward_fn, val_data, cfg)

                if val_loss < best_loss:
                    best_loss = val_loss
                    # Snapshot active stack weights
                    best_weights = [
                        layer.active_stack.parameters()
                        for layer in stacked_layers
                        if layer.active_stack is not None
                    ]
                    patience_counter = 0

                # §3.3 — "spike_threshold=0.1"
                elif val_loss > best_loss + spike_threshold:
                    # Restore best weights
                    if best_weights is not None:
                        for layer, weights in zip(stacked_layers, best_weights):
                            if layer.active_stack is not None:
                                layer.active_stack.load_weights(list(weights.items()))
                    break

                else:
                    patience_counter += 1
                    # §3.3 — "patience=4 evaluation steps"
                    if patience_counter >= patience:
                        break

        # Algorithm 1, line 5 — "ℓ_m ← evaluate(M, D_val)"
        l_m = _evaluate(model_forward_fn, val_data, cfg)

        # Algorithm 1, line 6 — "Freeze active stack → move to frozen stacks"
        for layer in stacked_layers:
            layer.freeze_active_stack()

        # Algorithm 1, lines 7-8 — "if ℓ_prev − ℓ_m < δ_min then break"
        if l_prev - l_m < min_loss_delta:
            break

        # Algorithm 1, line 9 — "ℓ_prev ← ℓ_m"
        l_prev = l_m

    return l_prev


def outer_loop(
    stacked_layers: list[StackedMoELoRALayer],
    domains: list[dict],
    model_forward_fn,
    null_space_projector: NullSpaceProjector,
    brainstacks_config: BrainstacksConfig,
    training_config: Optional[TrainingConfig] = None,
) -> dict:
    """Algorithm 2: Brainstacks Outer Loop — Continual Domain Learning.

    §3.4 — Sequential domain training with null-space projection from domain 2 onward.

    Args:
        stacked_layers: all StackedMoELoRALayer instances
        domains: list of dicts with 'name', 'train_data', 'val_data', 'rounds'
        model_forward_fn: forward pass function
        null_space_projector: NullSpaceProjector instance
        brainstacks_config: model config
        training_config: training hyperparameters

    Returns:
        results: dict of domain → final validation loss
    """
    cfg = training_config or TrainingConfig()
    results = {}

    for d_idx, domain in enumerate(domains):
        domain_name = domain["name"]

        # Algorithm 2, line 2 — "Compute null space projectors from all frozen stacks (if d > 1)"
        if d_idx > 0:
            _compute_and_set_null_space(
                stacked_layers, domain, null_space_projector
            )

        # Algorithm 2, line 3 — "Run inner loop on domain D_d with max rounds R_d"
        final_loss = inner_loop(
            stacked_layers=stacked_layers,
            train_data=domain["train_data"],
            val_data=domain["val_data"],
            model_forward_fn=model_forward_fn,
            max_rounds=domain.get("rounds", 2),
            training_config=cfg,
        )

        results[domain_name] = final_loss

        # Algorithm 2, lines 6-7 — "Evaluate on D_val_d' // forgetting check"
        for prev_d_idx in range(d_idx + 1):
            prev_domain = domains[prev_d_idx]
            prev_loss = _evaluate(model_forward_fn, prev_domain["val_data"], cfg)
            results[f"{prev_domain['name']}_after_{domain_name}"] = prev_loss

    return results


def _evaluate(model_forward_fn, val_data, cfg: TrainingConfig) -> float:
    """Evaluate model on validation data, returning mean loss."""
    total_loss = mx.array(0.0)
    count = 0
    for x, y in val_data:
        logits, aux = model_forward_fn(x, training=False)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="mean",
        )
        total_loss = total_loss + loss
        count += 1
        if count >= 10:  # Quick eval on subset
            break
    mx.eval(total_loss)
    return (total_loss / max(count, 1)).item()


def _compute_and_set_null_space(
    stacked_layers: list[StackedMoELoRALayer],
    domain: dict,
    projector: NullSpaceProjector,
) -> None:
    """Compute and set null-space projectors for all layers.

    §3.5 — "Run n_samples validation examples from previous domains,
    collecting the frozen stacks' aggregate output delta at each layer"
    """
    # TODO: Collect deltas by hooking into frozen stack forward passes
    # and running n_samples through the model. This requires the actual
    # base model forward pass with hooks, which is model-specific.
    #
    # Pseudocode:
    # for each layer:
    #   deltas = []
    #   for batch in prev_domain_val_data[:n_samples]:
    #     x = get_input_to_layer(batch)
    #     delta = sum(frozen_stack(x) for frozen_stack in layer.frozen_stacks)
    #     deltas.append(delta.mean(axis=1))  # average over seq_len
    #   D = stack(deltas)  # (n_samples, h_dim)
    #   P = projector.compute_projector(D)
    #   layer._null_space_P = P
    pass
