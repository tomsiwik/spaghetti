"""Shared Layer 0 Capsule Pool at N=5 Domains.

Extends the shared_layer0_pool experiment from N=2 to N=5 domains.
The parent experiment showed shared Layer 0 IMPROVES quality 1.7-3.0%
at N=2 with cross-pool Jaccard=0.544. This experiment tests whether
that benefit persists at N=5, where Layer 0 pools may have diverged
more due to different character distributions.

The composition functions are adapted from shared_layer0_pool.py to
handle N=5 domains with the quintary split.

Kill criteria:
  1. shared Layer 0 pool degrades quality >2% vs full concat at N=5
  2. Layer 0 cross-domain Jaccard drops below 0.40 at N=5
"""

import copy
import random

import mlx.core as mx
import mlx.nn as nn

from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


def compose_shared_layer0_n5(
    base_model: ReLURouterGPT,
    domain_models: list[ReLURouterGPT],
    strategy: str = "average",
) -> ReLURouterGPT:
    """Compose N domain models with a SHARED Layer 0 capsule pool.

    Layer 0: single pool (not concatenated), determined by strategy.
    Layers 1+: standard concatenation (per-domain pools composed).

    Args:
        base_model: The pretrained base model (before domain fine-tuning).
        domain_models: List of N domain-specialized models.
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

    vocab_size = base_model.lm_head.weight.shape[0]

    # Build composed model with full concatenation first (for layers 1+)
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
        shared_pool.A.load_weights([("weight", base_model.layers[0].capsule_pool.A.weight)])
        shared_pool.B.load_weights([("weight", base_model.layers[0].capsule_pool.B.weight)])

    elif strategy == "average":
        A_avg = sum(dm.layers[0].capsule_pool.A.weight for dm in domain_models) / n_domains
        B_avg = sum(dm.layers[0].capsule_pool.B.weight for dm in domain_models) / n_domains
        shared_pool.A.load_weights([("weight", A_avg)])
        shared_pool.B.load_weights([("weight", B_avg)])

    elif strategy == "first":
        shared_pool.A.load_weights([("weight", domain_models[0].layers[0].capsule_pool.A.weight)])
        shared_pool.B.load_weights([("weight", domain_models[0].layers[0].capsule_pool.B.weight)])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Replace Layer 0's capsule pool with the shared one
    composed.layers[0].capsule_pool = shared_pool
    mx.eval(composed.parameters())

    return composed


def compose_full_concat_n5(
    base_model: ReLURouterGPT,
    domain_models: list[ReLURouterGPT],
) -> ReLURouterGPT:
    """Standard full concatenation composition (control condition).

    All layers get concatenated per-domain pools. N-domain generalization
    of compose_full_concat from shared_layer0_pool.
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

    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

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


def profile_layer0_cross_domain_jaccard(
    domain_models: list[ReLURouterGPT],
    base_model: ReLURouterGPT,
    dataset,
    domain_names: list[str],
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = 0,
):
    """Compute pairwise Layer 0 co-activation Jaccard across all domain pairs.

    For each domain model, profile which Layer 0 capsules fire on joint data.
    Then compute pairwise Jaccard between all domain pairs.

    Returns:
        dict with pairwise Jaccards, mean, min, and per-domain fire patterns.
    """
    rng = random.Random(seed)
    n_domains = len(domain_models)
    P = domain_models[0].layers[0].capsule_pool.n_capsules

    # Collect fire patterns for each domain's Layer 0
    fire_patterns = {}  # domain_name -> (P,) total fire counts

    for d_idx, d_name in enumerate(domain_names):
        model = domain_models[d_idx]
        layer = model.layers[0]
        fire_count = mx.zeros((P,))

        rng_local = random.Random(seed)  # same data for all domains
        for batch_idx in range(n_batches):
            inputs, _ = dataset.get_batch(batch_size, rng_local)
            B_size, T = inputs.shape

            pos = mx.arange(T)
            x = model.wte(inputs) + model.wpe(pos)
            x = model.norm0(x)
            x_norm1 = layer.norm1(x)
            x = x + layer.attn(x_norm1)
            x_norm2 = layer.norm2(x)

            pool = layer.capsule_pool
            h = nn.relu(pool.A(x_norm2))  # (B, T, P)
            fired = (h > 0).astype(mx.float32)  # (B, T, P)
            fired_flat = fired.reshape(-1, P)  # (N, P)

            fire_count = fire_count + mx.sum(fired_flat, axis=0)

            if batch_idx % 5 == 0:
                mx.eval(fire_count)

        mx.eval(fire_count)
        # Binary: capsule fires at least once -> alive
        fire_binary = (fire_count > 0).astype(mx.float32)
        fire_patterns[d_name] = fire_binary

    # Compute pairwise Jaccard
    pairwise_jaccards = {}
    all_jacs = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            d_i = domain_names[i]
            d_j = domain_names[j]
            fi = fire_patterns[d_i]
            fj = fire_patterns[d_j]

            intersection = mx.sum(fi * fj)
            union = mx.sum(mx.clip(fi + fj, 0, 1))
            mx.eval(intersection, union)

            jac = (intersection / (union + 1e-8)).item()
            pairwise_jaccards[(d_i, d_j)] = jac
            all_jacs.append(jac)

    return {
        "pairwise_jaccards": pairwise_jaccards,
        "mean_jaccard": sum(all_jacs) / len(all_jacs) if all_jacs else 0.0,
        "min_jaccard": min(all_jacs) if all_jacs else 0.0,
        "max_jaccard": max(all_jacs) if all_jacs else 0.0,
        "n_pairs": len(all_jacs),
    }


def profile_layer0_coactivation_jaccard(
    composed_model: ReLURouterGPT,
    dataset,
    n_capsules_per_domain: int,
    n_domains: int,
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = 0,
):
    """Compute cross-pool co-activation Jaccard in composed model Layer 0.

    This matches the profile_layer0_jaccard from shared_layer0_pool but
    generalized to N domains: computes average cross-pool Jaccard between
    all domain pairs in the concatenated Layer 0.

    Returns:
        dict with mean, p50, p90, max Jaccard across all cross-pool pairs.
    """
    rng = random.Random(seed)
    layer = composed_model.layers[0]
    P_total = layer.capsule_pool.n_capsules
    P = n_capsules_per_domain

    co_fire = mx.zeros((P_total, P_total))
    fire_count = mx.zeros((P_total,))

    for batch_idx in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B_size, T = inputs.shape

        pos = mx.arange(T)
        x = composed_model.wte(inputs) + composed_model.wpe(pos)
        x = composed_model.norm0(x)
        x_norm1 = layer.norm1(x)
        x = x + layer.attn(x_norm1)
        x_norm2 = layer.norm2(x)

        pool = layer.capsule_pool
        h = nn.relu(pool.A(x_norm2))  # (B, T, P_total)
        fired = (h > 0).astype(mx.float32)
        fired_flat = fired.reshape(-1, P_total)

        co_fire = co_fire + (fired_flat.T @ fired_flat)
        fire_count = fire_count + mx.sum(fired_flat, axis=0)

        if batch_idx % 5 == 0:
            mx.eval(co_fire, fire_count)

    mx.eval(co_fire, fire_count)

    # Compute Jaccard matrix
    fc_sum = fire_count[:, None] + fire_count[None, :]
    union = fc_sum - co_fire
    J = co_fire / (union + 1e-8)
    mx.eval(J)
    J_np = J.tolist()

    # Compute cross-pool Jaccard: between capsules from different domains
    cross_jacs = []
    for di in range(n_domains):
        for dj in range(di + 1, n_domains):
            for ci in range(P):
                for cj in range(P):
                    cross_jacs.append(J_np[di * P + ci][dj * P + cj])

    if cross_jacs:
        cross_jacs.sort()
        n = len(cross_jacs)
        return {
            "mean": sum(cross_jacs) / len(cross_jacs),
            "p50": cross_jacs[n // 2],
            "p90": cross_jacs[int(n * 0.90)],
            "max": cross_jacs[-1],
            "n_pairs": n,
        }
    else:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "n_pairs": 0}


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
