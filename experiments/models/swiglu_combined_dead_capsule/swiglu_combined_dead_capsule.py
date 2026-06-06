"""Combined dead capsule + gate-product pruning for SwiGLU MLPs.

Two pruning criteria applied to SwiGLU MLP capsules:

1. **Dead capsule pruning** (activation frequency = 0): A capsule whose gate
   product |SiLU(W_gate @ x) * (W_up @ x)| is ALWAYS below a tiny epsilon
   across all calibration data. Removing such capsules is exact (zero output
   change). This is the SwiGLU analogue of ReLU dead neuron pruning.

2. **Gate-product magnitude pruning** (mean |gate*up| < tau): A capsule whose
   AVERAGE gate product magnitude is below threshold tau. Removing these
   capsules introduces bounded error: ||delta_y|| <= tau * ||b_i|| per position.

The hypothesis: these criteria identify COMPLEMENTARY parameter sets.
Dead capsules are those that never contribute (frequency-based).
Gate-product-low capsules contribute rarely but may still fire occasionally.
If the overlap is low, combining both criteria yields higher pruning rates
than either alone without additional quality cost.

Kill criteria:
  - Combined pruning does not exceed either method alone by >5pp
  - Quality degrades >3% vs no pruning
"""

import copy
import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..swiglu_gate_pruning.swiglu_gate_pruning import (
    SwiGLUGatePruningGPT,
    SwiGLUCapsulePool,
    SwiGLUCapsuleBlock,
    profile_gate_products,
    identify_prunable_by_gate_product,
    prune_swiglu_model,
)


def profile_dead_capsules_swiglu(model: SwiGLUGatePruningGPT,
                                  dataset,
                                  n_batches: int = 20,
                                  batch_size: int = 32,
                                  seed: int = 0,
                                  dead_epsilon: float = 1e-6
                                  ) -> list[dict]:
    """Profile per-capsule activation frequency for SwiGLU models.

    A capsule is "dead" if its gate product |SiLU(W_gate @ x) * (W_up @ x)|
    is below dead_epsilon for ALL positions in the calibration data.

    Returns per-layer dicts with:
      - fire_frequency: fraction of positions where |gate_product| > dead_epsilon
      - mean_abs: mean |gate_product| per capsule
      - max_abs: max |gate_product| per capsule
      - n_dead: count of capsules with fire_frequency = 0
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Accumulators
    fire_counts = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    sum_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    max_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]

    total_positions = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_positions += B * T

        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            x_norm = layer.norm2(x)
            pool = layer.capsule_pool

            gate_out = nn.silu(pool.W_gate(x_norm))  # (B, T, P)
            up_out = pool.W_up(x_norm)                # (B, T, P)
            gate_product = gate_out * up_out           # (B, T, P)

            gp_abs = mx.abs(gate_product)

            # Count fires: positions where |gate_product| > epsilon
            fired = (gp_abs > dead_epsilon).astype(mx.float32)
            fire_counts[l_idx] = fire_counts[l_idx] + mx.sum(fired, axis=(0, 1))

            # Accumulate magnitude stats
            sum_abs[l_idx] = sum_abs[l_idx] + mx.sum(gp_abs, axis=(0, 1))
            max_abs[l_idx] = mx.maximum(max_abs[l_idx], mx.max(gp_abs, axis=(0, 1)))

            mx.eval(fire_counts[l_idx], sum_abs[l_idx], max_abs[l_idx])

            # Complete layer forward
            x = x + pool.B(gate_product)

    results = []
    for l_idx in range(n_layers):
        P = model.layers[l_idx].capsule_pool.n_capsules
        freq = fire_counts[l_idx] / total_positions
        mean = sum_abs[l_idx] / total_positions
        mx.eval(freq, mean, max_abs[l_idx])

        freq_list = freq.tolist()
        n_dead = sum(1 for f in freq_list if f <= 0.0)

        results.append({
            "fire_frequency": freq,
            "mean_abs": mean,
            "max_abs": max_abs[l_idx],
            "n_capsules": P,
            "n_dead": n_dead,
            "pct_dead": n_dead / P * 100 if P > 0 else 0,
        })

    return results


def identify_dead_capsules_swiglu(profiles: list[dict],
                                   threshold: float = 0.0
                                   ) -> list[mx.array]:
    """Identify dead capsules: those that fire <= threshold fraction of the time.

    Args:
        profiles: Per-layer profiles from profile_dead_capsules_swiglu.
        threshold: Frequency threshold. 0.0 = truly dead (never fires).

    Returns:
        List of boolean masks (True = alive, False = dead).
    """
    masks = []
    for prof in profiles:
        alive = prof["fire_frequency"] > threshold
        mx.eval(alive)
        masks.append(alive)
    return masks


def combined_pruning_masks(dead_masks: list[mx.array],
                            gate_masks: list[mx.array]
                            ) -> list[mx.array]:
    """Combine dead capsule and gate-product masks via intersection (AND).

    A capsule survives only if it is alive in BOTH criteria.
    Equivalently: prune if dead OR gate-product-low.

    Returns:
        List of boolean masks (True = alive if alive in both).
    """
    combined = []
    for dm, gm in zip(dead_masks, gate_masks):
        # alive = alive_dead AND alive_gate
        mask = dm * gm  # element-wise AND for boolean-like float masks
        # Ensure boolean
        mask = mask.astype(mx.bool_)
        mx.eval(mask)
        combined.append(mask)
    return combined


def compute_set_overlap(dead_masks: list[mx.array],
                         gate_masks: list[mx.array],
                         n_capsules_per_layer: list[int]
                         ) -> dict:
    """Compute overlap statistics between dead and gate-product pruning sets.

    Returns dict with:
      - jaccard: |dead AND gate| / |dead OR gate|
      - overlap_coeff: |dead AND gate| / min(|dead|, |gate|)
      - pct_gate_also_dead: fraction of gate-prunable that are also dead
      - pct_dead_also_gate: fraction of dead that are also gate-prunable
      - per_layer: detailed per-layer breakdown
    """
    total_dead = 0
    total_gate = 0
    total_both = 0
    total_either = 0
    total_caps = 0
    per_layer = []

    for l_idx, (dm, gm) in enumerate(zip(dead_masks, gate_masks)):
        P = n_capsules_per_layer[l_idx]
        mx.eval(dm, gm)

        # Prune sets (inverted masks: False = prunable)
        dead_prune = ~dm.astype(mx.bool_)  # True = dead/prunable
        gate_prune = ~gm.astype(mx.bool_)  # True = gate-prunable

        n_dead = int(mx.sum(dead_prune.astype(mx.float32)).item())
        n_gate = int(mx.sum(gate_prune.astype(mx.float32)).item())

        both = dead_prune * gate_prune  # AND: prunable by both
        either = mx.maximum(dead_prune.astype(mx.float32),
                            gate_prune.astype(mx.float32))  # OR

        n_both = int(mx.sum(both.astype(mx.float32)).item())
        n_either = int(mx.sum(either).item())

        # Dead-only (dead but not gate-prunable)
        dead_only = dead_prune * (~gate_prune)
        n_dead_only = int(mx.sum(dead_only.astype(mx.float32)).item())

        # Gate-only (gate-prunable but not dead)
        gate_only = gate_prune * (~dead_prune)
        n_gate_only = int(mx.sum(gate_only.astype(mx.float32)).item())

        layer_info = {
            "layer": l_idx,
            "n_capsules": P,
            "n_dead": n_dead,
            "n_gate_prunable": n_gate,
            "n_both": n_both,
            "n_either": n_either,
            "n_dead_only": n_dead_only,
            "n_gate_only": n_gate_only,
            "pct_dead": n_dead / P * 100 if P > 0 else 0,
            "pct_gate": n_gate / P * 100 if P > 0 else 0,
            "pct_combined": n_either / P * 100 if P > 0 else 0,
        }

        if n_gate > 0:
            layer_info["pct_gate_also_dead"] = n_both / n_gate * 100
        else:
            layer_info["pct_gate_also_dead"] = 0.0

        if n_dead > 0:
            layer_info["pct_dead_also_gate"] = n_both / n_dead * 100
        else:
            layer_info["pct_dead_also_gate"] = 0.0

        per_layer.append(layer_info)
        total_dead += n_dead
        total_gate += n_gate
        total_both += n_both
        total_either += n_either
        total_caps += P

    # Aggregate
    jaccard = total_both / total_either if total_either > 0 else 0.0
    overlap_coeff = (total_both / min(total_dead, total_gate)
                     if min(total_dead, total_gate) > 0 else 0.0)
    pct_gate_also_dead = total_both / total_gate * 100 if total_gate > 0 else 0.0
    pct_dead_also_gate = total_both / total_dead * 100 if total_dead > 0 else 0.0

    return {
        "total_capsules": total_caps,
        "total_dead": total_dead,
        "total_gate_prunable": total_gate,
        "total_both": total_both,
        "total_either": total_either,
        "jaccard": jaccard,
        "overlap_coeff": overlap_coeff,
        "pct_gate_also_dead": pct_gate_also_dead,
        "pct_dead_also_gate": pct_dead_also_gate,
        "pct_dead_only": (total_dead - total_both) / total_caps * 100 if total_caps > 0 else 0,
        "pct_gate_only": (total_gate - total_both) / total_caps * 100 if total_caps > 0 else 0,
        "pct_combined": total_either / total_caps * 100 if total_caps > 0 else 0,
        "per_layer": per_layer,
    }


@register("swiglu_combined_dead_capsule", parent="swiglu_gate_pruning")
class SwiGLUCombinedDeadCapsuleGPT(SwiGLUGatePruningGPT):
    """SwiGLU GPT with combined dead capsule + gate-product pruning.

    Inherits all SwiGLU architecture from SwiGLUGatePruningGPT.
    The pruning is applied as post-processing, not architectural change.
    """
    pass
