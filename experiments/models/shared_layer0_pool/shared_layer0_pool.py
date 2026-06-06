"""Shared Layer 0 Capsule Pool Across Domains.

Exp behavioral_dedup showed that Layer 0 has massive cross-domain
co-activation (J=0.527) while deeper layers specialize (J<0.05).
This experiment tests the natural follow-up: share a single Layer 0
capsule pool across all domains, training per-domain pools only for
layers 1+.

The model class itself is identical to ReLURouterGPT. The sharing
is implemented in the COMPOSITION PROTOCOL, not the architecture.
Three sharing strategies for Layer 0:

  1. "base": Use the pretrained base model's Layer 0 (no domain fine-tuning)
  2. "average": Average the domain-specific Layer 0 weights
  3. "first": Use the first domain's Layer 0 (arbitrary choice, tests
     whether Layer 0 is truly domain-invariant)

For layers 1+, standard concatenation composition is used (the validated
protocol from relu_router).

Kill criterion: shared Layer 0 pool degrades quality >2% vs per-domain
Layer 0 pools (i.e., standard full concatenation).
"""

import copy

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


def compose_shared_layer0(
    base_model: ReLURouterGPT,
    domain_models: list[ReLURouterGPT],
    strategy: str = "average",
) -> ReLURouterGPT:
    """Compose domain models with a SHARED Layer 0 capsule pool.

    Layer 0: single pool (not concatenated), determined by strategy.
    Layers 1+: standard concatenation (per-domain pools composed).

    Args:
        base_model: The pretrained base model (before domain fine-tuning).
        domain_models: List of domain-specialized models.
        strategy: How to construct the shared Layer 0.
            "base": use pretrained base Layer 0 (no domain adaptation)
            "average": average domain-specific Layer 0 weights
            "first": use first domain's Layer 0

    Returns:
        Composed ReLURouterGPT with shared Layer 0.
    """
    n_domains = len(domain_models)
    n_capsules_per_domain = domain_models[0].layers[0].capsule_pool.n_capsules
    n_capsules_concat = n_capsules_per_domain * n_domains

    # For Layer 0: shared pool has n_capsules_per_domain capsules (NOT concatenated)
    # For Layers 1+: concatenated pool has n_capsules_concat capsules
    # We need a model where Layer 0 has different capsule count than other layers.
    # Simplest approach: build a model with concat capsules everywhere,
    # then replace Layer 0 with the shared pool.

    vocab_size = base_model.lm_head.weight.shape[0]

    # Build composed model with full concatenation first
    composed = ReLURouterGPT(
        vocab_size=vocab_size,
        n_capsules=n_capsules_concat,
        n_embd=base_model.layers[0].attn.wq.weight.shape[0],
        n_head=base_model.layers[0].attn.n_head,
        n_layer=len(base_model.layers),
        block_size=base_model.wpe.weight.shape[0],
    )
    mx.eval(composed.parameters())

    # Copy shared parameters (embeddings, attention, norms, lm_head)
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    # Layers 1+: standard concatenation
    for layer_idx in range(1, len(composed.layers)):
        A_parts = [dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)  # (P*D, d)
        B_composed = mx.concatenate(B_parts, axis=1)  # (d, P*D)

        comp_pool = composed.layers[layer_idx].capsule_pool
        comp_pool.A.load_weights([("weight", A_composed)])
        comp_pool.B.load_weights([("weight", B_composed)])

    # Layer 0: shared pool (NOT concatenated)
    n_embd = base_model.layers[0].attn.wq.weight.shape[0]
    shared_pool = ReLUCapsulePool(n_embd, n_capsules_per_domain)

    if strategy == "base":
        # Use the pretrained base Layer 0 (before domain fine-tuning)
        shared_pool.A.load_weights([("weight", base_model.layers[0].capsule_pool.A.weight)])
        shared_pool.B.load_weights([("weight", base_model.layers[0].capsule_pool.B.weight)])

    elif strategy == "average":
        # Average domain-specific Layer 0 weights
        A_avg = sum(dm.layers[0].capsule_pool.A.weight for dm in domain_models) / n_domains
        B_avg = sum(dm.layers[0].capsule_pool.B.weight for dm in domain_models) / n_domains
        shared_pool.A.load_weights([("weight", A_avg)])
        shared_pool.B.load_weights([("weight", B_avg)])

    elif strategy == "first":
        # Use first domain's Layer 0 (tests domain-invariance)
        shared_pool.A.load_weights([("weight", domain_models[0].layers[0].capsule_pool.A.weight)])
        shared_pool.B.load_weights([("weight", domain_models[0].layers[0].capsule_pool.B.weight)])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Replace Layer 0's capsule pool with the shared one
    composed.layers[0].capsule_pool = shared_pool
    mx.eval(composed.parameters())

    return composed


def compose_full_concat(
    base_model: ReLURouterGPT,
    domain_models: list[ReLURouterGPT],
) -> ReLURouterGPT:
    """Standard full concatenation composition (control condition).

    All layers get concatenated per-domain pools.
    This is identical to compose_relu_models from test_composition.py,
    duplicated here for self-contained experiment.
    """
    n_domains = len(domain_models)
    n_capsules_per_domain = domain_models[0].layers[0].capsule_pool.n_capsules
    n_capsules_total = n_capsules_per_domain * n_domains

    vocab_size = base_model.lm_head.weight.shape[0]
    composed = ReLURouterGPT(
        vocab_size=vocab_size,
        n_capsules=n_capsules_total,
        n_embd=base_model.layers[0].attn.wq.weight.shape[0],
        n_head=base_model.layers[0].attn.n_head,
        n_layer=len(base_model.layers),
        block_size=base_model.wpe.weight.shape[0],
    )
    mx.eval(composed.parameters())

    # Copy shared parameters
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    # ALL layers: concatenation
    for layer_idx in range(len(composed.layers)):
        A_parts = [dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)
        B_composed = mx.concatenate(B_parts, axis=1)

        comp_pool = composed.layers[layer_idx].capsule_pool
        comp_pool.A.load_weights([("weight", A_composed)])
        comp_pool.B.load_weights([("weight", B_composed)])

    mx.eval(composed.parameters())
    return composed


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


@register("shared_layer0_pool", parent="behavioral_dedup")
class SharedLayer0GPT(ReLURouterGPT):
    """ReLURouterGPT with shared Layer 0 capsule pool composition protocol.

    The model architecture is identical to ReLURouterGPT. The innovation
    is in the composition function (compose_shared_layer0) which shares
    Layer 0 instead of concatenating it.

    Registered for lineage tracking. All forward pass logic inherited.
    """
    pass
