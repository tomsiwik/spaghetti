"""Capsule Identity Tracking Across Composition -- Exp 16.

Exp 10 (pruning_controls) showed 87% of composed death is training-induced
(aggregate). Exp 18 (capsule_revival) showed Jaccard=0.669 for dead cohort
stability over training steps. This experiment completes the picture by
comparing ACROSS settings (single vs composed) rather than across time.

Core question: When you profile which capsules are dead in single-domain
vs composed settings, what is the Jaccard overlap of the dead sets? Are
the SAME capsules dead, or does composition create novel death patterns?

The model class is a thin wrapper for lineage tracking. All profiling
and analysis functions are defined here as standalone utilities.
"""

import mlx.core as mx

from ..relu_router.relu_router import ReLURouterGPT
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations


def get_dead_set(freqs, threshold=0.0):
    """Convert per-layer frequency arrays to a flat set of dead capsule indices.

    Args:
        freqs: list of (P_l,) arrays from profile_activations
        threshold: frequency threshold at or below which capsule is "dead"

    Returns:
        dead_set: set of (layer_idx, capsule_idx) tuples
        dead_flat: flat list of bool (True = dead, for all layers concatenated)
        per_layer_counts: list of (n_dead, n_total) tuples per layer
    """
    dead_set = set()
    dead_flat = []
    per_layer_counts = []

    for l_idx, freq in enumerate(freqs):
        mx.eval(freq)
        freq_list = freq.tolist()
        n_dead = 0
        for c_idx, f in enumerate(freq_list):
            is_dead = f <= threshold
            dead_flat.append(is_dead)
            if is_dead:
                dead_set.add((l_idx, c_idx))
                n_dead += 1
        per_layer_counts.append((n_dead, len(freq_list)))

    return dead_set, dead_flat, per_layer_counts


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity: |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 1.0


def overlap_coefficient(set_a, set_b):
    """Overlap coefficient: |A & B| / min(|A|, |B|).

    Measures what fraction of the smaller set is contained in the larger.
    Robust to sets of different sizes (unlike Jaccard).
    """
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def dice_coefficient(set_a, set_b):
    """Dice coefficient: 2|A & B| / (|A| + |B|).

    Alternative to Jaccard, less sensitive to size differences.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    return 2 * len(intersection) / (len(set_a) + len(set_b)) if (set_a or set_b) else 1.0


def composition_death_decomposition(
    dead_single_A, dead_single_B, dead_composed, n_capsules_per_domain
):
    """Decompose composed dead set into sources.

    In the composed model, capsules [0..P-1] come from domain A and
    capsules [P..2P-1] come from domain B. This function tracks which
    capsules in the composed dead set were already dead in single-domain.

    Args:
        dead_single_A: set of (layer, capsule_idx) dead in domain A model
        dead_single_B: set of (layer, capsule_idx) dead in domain B model
        dead_composed: set of (layer, capsule_idx) dead in composed model
        n_capsules_per_domain: P (capsules per domain per layer)

    Returns:
        dict with decomposition statistics
    """
    # In composed model, A's capsules are at indices [0..P-1],
    # B's capsules are at indices [P..2P-1]
    P = n_capsules_per_domain

    # Split composed dead set into A-half and B-half
    composed_dead_A_half = {(l, c) for (l, c) in dead_composed if c < P}
    composed_dead_B_half = {(l, c - P) for (l, c) in dead_composed if c >= P}

    # Categories for A's capsules
    dead_in_both_A = dead_single_A & composed_dead_A_half
    dead_only_single_A = dead_single_A - composed_dead_A_half
    dead_only_composed_A = composed_dead_A_half - dead_single_A

    # Categories for B's capsules
    dead_in_both_B = dead_single_B & composed_dead_B_half
    dead_only_single_B = dead_single_B - composed_dead_B_half
    dead_only_composed_B = composed_dead_B_half - dead_single_B

    # Jaccard per domain half
    jaccard_A = jaccard_similarity(dead_single_A, composed_dead_A_half)
    jaccard_B = jaccard_similarity(dead_single_B, composed_dead_B_half)
    overlap_A = overlap_coefficient(dead_single_A, composed_dead_A_half)
    overlap_B = overlap_coefficient(dead_single_B, composed_dead_B_half)

    return {
        "domain_A": {
            "dead_single": len(dead_single_A),
            "dead_composed_half": len(composed_dead_A_half),
            "dead_in_both": len(dead_in_both_A),
            "dead_only_single": len(dead_only_single_A),
            "dead_only_composed": len(dead_only_composed_A),
            "jaccard": jaccard_A,
            "overlap_coeff": overlap_A,
        },
        "domain_B": {
            "dead_single": len(dead_single_B),
            "dead_composed_half": len(composed_dead_B_half),
            "dead_in_both": len(dead_in_both_B),
            "dead_only_single": len(dead_only_single_B),
            "dead_only_composed": len(dead_only_composed_B),
            "jaccard": jaccard_B,
            "overlap_coeff": overlap_B,
        },
        "combined_jaccard": jaccard_similarity(
            dead_single_A | {(l, c + P) for (l, c) in dead_single_B},
            dead_composed,
        ),
        "combined_overlap": overlap_coefficient(
            dead_single_A | {(l, c + P) for (l, c) in dead_single_B},
            dead_composed,
        ),
    }


def per_layer_jaccard(dead_set_a, dead_set_b, n_layers):
    """Compute Jaccard similarity per layer.

    Args:
        dead_set_a, dead_set_b: sets of (layer, capsule_idx) tuples
        n_layers: number of layers

    Returns:
        list of (layer_idx, jaccard, n_dead_a, n_dead_b, n_intersection) tuples
    """
    results = []
    for l in range(n_layers):
        layer_a = {c for (ll, c) in dead_set_a if ll == l}
        layer_b = {c for (ll, c) in dead_set_b if ll == l}
        j = jaccard_similarity(layer_a, layer_b)
        intersection = len(layer_a & layer_b)
        results.append((l, j, len(layer_a), len(layer_b), intersection))
    return results
