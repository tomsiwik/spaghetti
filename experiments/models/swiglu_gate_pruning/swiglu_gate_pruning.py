"""SwiGLU Gate-Aware Pruning -- gate-product profiling for SwiGLU capsule MLPs.

SiLU pruning was killed (Exp 15): SiLU(z) has a floor at ~0.046 mean absolute
activation, making 0% of capsules prunable at safe thresholds.

But Qwen's actual MLP is SwiGLU:
    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

The PRODUCT can be near-zero even when SiLU is not -- the up-projection acts
as a learned binary mask. This experiment profiles |gate_output * up_output|
per capsule and prunes where the product is near-zero.

Key differences from SiLU pruning (Exp 15):
  - SiLU profiled: |SiLU(A @ x)| per capsule (floor ~0.046)
  - SwiGLU profiles: |SiLU(W_gate @ x) * (W_up @ x)| per capsule
  - The multiplicative interaction creates additional sparsity

Kill criteria:
  - <10% of capsules prunable at gate-product threshold
  - Pruning by gate product >3% worse than no pruning
"""

import copy
import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class SwiGLUCapsulePool(nn.Module):
    """SwiGLU MLP capsule pool, matching Qwen/Llama architecture.

    Forward: y = B @ (SiLU(W_gate @ x) * (W_up @ x))

    Where:
      - W_gate: (P, d) -- gate projection, SiLU-activated
      - W_up:   (P, d) -- up projection, acts as learned gate
      - B:      (d, P) -- down projection
      - P = n_capsules (the intermediate dimension)

    This is architecturally identical to:
      fc1 = gate_proj, fc2 = up_proj, fc3 = down_proj
    in miniqwen.py / Qwen3.5's FeedForward.

    Composition: same as SiLU capsules -- concatenate gate/up vertically
    and B horizontally. B is zero-initialized for identity at composition.
    """

    NEAR_ZERO_THRESHOLD = 0.01

    def __init__(self, n_embd: int, n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.n_capsules = n_capsules
        self.l1_target_sparsity = l1_target_sparsity
        self.l1_coeff = l1_coeff
        self.balance_coeff = balance_coeff

        # W_gate: (P, d) -- gate projection (SiLU-activated)
        self.W_gate = nn.Linear(n_embd, n_capsules, bias=False)
        # W_up: (P, d) -- up projection (linear, acts as gate)
        self.W_up = nn.Linear(n_embd, n_capsules, bias=False)
        # B: (d, P) -- down projection
        self.B = nn.Linear(n_capsules, n_embd, bias=False)

        self._running_sparsity = 0.3
        self._activation_counts = None
        self._last_gate_product = None  # for profiling

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d)"""
        gate_out = nn.silu(self.W_gate(x))  # (B, T, P) -- SiLU-activated gate
        up_out = self.W_up(x)               # (B, T, P) -- linear up projection
        h = gate_out * up_out               # (B, T, P) -- gate product (SwiGLU)
        self._store_stats(h)
        return self.B(h)                    # (B, T, d)

    def _store_stats(self, h):
        """Store gate product statistics."""
        self._last_gate_product = h

        near_zero = (mx.abs(h) < self.NEAR_ZERO_THRESHOLD).astype(mx.float32)
        sparsity = mx.mean(near_zero).item()
        alpha = 0.1
        self._running_sparsity = (1 - alpha) * self._running_sparsity + alpha * sparsity

        self._activation_counts = mx.mean(mx.abs(h), axis=(0, 1))  # (P,)

    def sparsity_loss(self) -> mx.array:
        if self._activation_counts is None:
            return mx.array(0.0)
        sparsity_gap = self.l1_target_sparsity - self._running_sparsity
        adaptive_coeff = self.l1_coeff * (1.0 + 5.0 * max(-0.5, min(0.5, sparsity_gap)))
        mean_activation = mx.mean(self._activation_counts)
        return adaptive_coeff * mean_activation

    def balance_loss(self) -> mx.array:
        if self._activation_counts is None:
            return mx.array(0.0)
        mean_act = mx.mean(self._activation_counts)
        variance = mx.mean((self._activation_counts - mean_act) ** 2)
        return self.balance_coeff * self.n_capsules * variance

    def aux_loss(self) -> mx.array:
        return self.sparsity_loss() + self.balance_loss()


class SwiGLUCapsuleBlock(nn.Module):
    """Transformer block with SwiGLU capsule pool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = SwiGLUCapsulePool(
            n_embd, n_capsules,
            l1_target_sparsity=l1_target_sparsity,
            l1_coeff=l1_coeff,
            balance_coeff=balance_coeff,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("swiglu_gate_pruning", parent="silu_capsule")
class SwiGLUGatePruningGPT(nn.Module):
    """GPT with SwiGLU MLP for gate-product pruning experiments.

    Architecture matches Qwen3.5's FeedForward:
      output = B @ (SiLU(W_gate @ x) * (W_up @ x))

    The key insight: even though SiLU(W_gate @ x) is never zero (floor ~0.046),
    the PRODUCT SiLU(W_gate @ x) * (W_up @ x) can be near-zero when W_up learns
    to suppress specific capsules for specific inputs.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [SwiGLUCapsuleBlock(
            n_embd, n_head, n_capsules,
            l1_target_sparsity=l1_target_sparsity,
            l1_coeff=l1_coeff,
            balance_coeff=balance_coeff,
        ) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.aux_loss()
        return total

    def on_domain_switch(self, domain: str):
        pass

    def capsule_stats(self) -> dict:
        """Gate product statistics from last forward pass."""
        stats = {"eff_sparsity": [], "mean_act": [], "min_act": [],
                 "max_act": [], "n_near_dead": []}

        for layer in self.layers:
            pool = layer.capsule_pool
            if pool._activation_counts is None:
                for k in stats:
                    stats[k].append(None)
                continue

            counts = pool._activation_counts
            mx.eval(counts)

            threshold = SwiGLUCapsulePool.NEAR_ZERO_THRESHOLD
            stats["mean_act"].append(mx.mean(counts).item())
            stats["min_act"].append(mx.min(counts).item())
            stats["max_act"].append(mx.max(counts).item())
            stats["n_near_dead"].append(int(mx.sum(counts < threshold).item()))

            if pool._last_gate_product is not None:
                h = pool._last_gate_product
                eff_sp = mx.mean((mx.abs(h) < threshold).astype(mx.float32)).item()
                stats["eff_sparsity"].append(eff_sp)
            else:
                stats["eff_sparsity"].append(pool._running_sparsity)

        return stats


# ========================================================================
# Profiling and pruning functions
# ========================================================================

def profile_gate_products(model: SwiGLUGatePruningGPT,
                          dataset,
                          n_batches: int = 20,
                          batch_size: int = 32,
                          seed: int = 0) -> list[dict]:
    """Profile per-capsule gate product magnitudes.

    For each layer, computes:
      - gate_product_mean_abs: mean |SiLU(W_gate @ x) * (W_up @ x)| per capsule
      - gate_only_mean_abs: mean |SiLU(W_gate @ x)| per capsule (SiLU component)
      - up_only_mean_abs: mean |W_up @ x| per capsule (linear component)
      - product_max_abs: max absolute gate product per capsule

    Returns list of dicts (one per layer).
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Accumulators for gate product
    sum_gp_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    max_gp_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]

    # Accumulators for individual components (diagnostic)
    sum_gate_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    sum_up_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]

    # Frequency above thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    freq_above = {tau: [mx.zeros(layer.capsule_pool.n_capsules)
                        for layer in model.layers]
                  for tau in thresholds}

    total_positions = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_positions += B * T

        # Forward through model layer by layer
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            # Attention
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            # Capsule pool -- compute gate product components
            x_norm = layer.norm2(x)
            pool = layer.capsule_pool

            gate_out = nn.silu(pool.W_gate(x_norm))  # (B, T, P)
            up_out = pool.W_up(x_norm)                # (B, T, P)
            gate_product = gate_out * up_out           # (B, T, P)

            # Accumulate gate product statistics
            gp_abs = mx.abs(gate_product)
            sum_gp_abs[l_idx] = sum_gp_abs[l_idx] + mx.sum(gp_abs, axis=(0, 1))
            max_gp_abs[l_idx] = mx.maximum(max_gp_abs[l_idx],
                                            mx.max(gp_abs, axis=(0, 1)))

            # Individual component statistics
            sum_gate_abs[l_idx] = sum_gate_abs[l_idx] + mx.sum(mx.abs(gate_out), axis=(0, 1))
            sum_up_abs[l_idx] = sum_up_abs[l_idx] + mx.sum(mx.abs(up_out), axis=(0, 1))

            # Frequency above thresholds (for gate product)
            for tau in thresholds:
                above = (gp_abs > tau).astype(mx.float32)
                freq_above[tau][l_idx] = freq_above[tau][l_idx] + mx.sum(above, axis=(0, 1))

            mx.eval(sum_gp_abs[l_idx], max_gp_abs[l_idx],
                    sum_gate_abs[l_idx], sum_up_abs[l_idx])
            for tau in thresholds:
                mx.eval(freq_above[tau][l_idx])

            # Complete the layer forward pass
            x = x + pool.B(gate_product)

    # Compute final statistics
    results = []
    for l_idx in range(n_layers):
        P = model.layers[l_idx].capsule_pool.n_capsules
        gp_mean_abs = sum_gp_abs[l_idx] / total_positions
        gate_mean_abs = sum_gate_abs[l_idx] / total_positions
        up_mean_abs = sum_up_abs[l_idx] / total_positions
        mx.eval(gp_mean_abs, gate_mean_abs, up_mean_abs, max_gp_abs[l_idx])

        freq_dict = {}
        for tau in thresholds:
            f = freq_above[tau][l_idx] / total_positions
            mx.eval(f)
            freq_dict[tau] = f

        results.append({
            "gate_product_mean_abs": gp_mean_abs,
            "gate_only_mean_abs": gate_mean_abs,
            "up_only_mean_abs": up_mean_abs,
            "product_max_abs": max_gp_abs[l_idx],
            "freq_above": freq_dict,
            "n_capsules": P,
        })

    return results


def identify_prunable_by_gate_product(profiles: list[dict],
                                       threshold: float = 0.01,
                                       method: str = "gate_product_mean_abs"
                                       ) -> list[mx.array]:
    """Identify capsules to prune based on gate product magnitude.

    Args:
        profiles: Per-layer profiles from profile_gate_products.
        threshold: Gate product magnitude threshold.
        method: "gate_product_mean_abs" or "product_max_abs".

    Returns:
        List of boolean mask arrays (True = alive, False = prune).
    """
    masks = []
    for prof in profiles:
        metric = prof[method]
        alive = metric > threshold
        mx.eval(alive)
        masks.append(alive)
    return masks


def prune_swiglu_model(model: SwiGLUGatePruningGPT,
                        alive_masks: list[mx.array],
                        verbose: bool = True) -> dict:
    """Prune capsules from SwiGLU model in-place.

    Removes capsule i by removing row i from W_gate and W_up,
    and column i from B.

    Error bound: for capsule i with gate_product_mean_abs < tau,
    ||delta_y|| <= tau * ||b_i|| per position (approximate, not lossless).
    """
    stats = {
        "per_layer": [],
        "total_before": 0,
        "total_after": 0,
        "total_pruned": 0,
    }

    for l_idx, (layer, mask) in enumerate(zip(model.layers, alive_masks)):
        pool = layer.capsule_pool
        W_gate = pool.W_gate.weight  # (P, d)
        W_up = pool.W_up.weight      # (P, d)
        B = pool.B.weight            # (d, P)
        P_before = W_gate.shape[0]
        d = W_gate.shape[1]

        mx.eval(mask)
        alive_indices = mx.array([i for i in range(P_before) if mask[i].item()])
        n_alive = alive_indices.shape[0]

        if n_alive == P_before:
            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": P_before,
                "n_pruned": 0,
                "pct_pruned": 0.0,
            }
        elif n_alive == 0:
            # Keep at least 1 capsule
            alive_indices = mx.array([0])
            n_alive = 1
            new_pool = SwiGLUCapsulePool(d, 1)
            new_pool.W_gate.load_weights([("weight", W_gate[0:1])])
            new_pool.W_up.load_weights([("weight", W_up[0:1])])
            new_pool.B.load_weights([("weight", B[:, 0:1])])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())

            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": 1,
                "n_pruned": P_before - 1,
                "pct_pruned": (P_before - 1) / P_before * 100,
            }
        else:
            W_gate_new = W_gate[alive_indices]  # (n_alive, d)
            W_up_new = W_up[alive_indices]      # (n_alive, d)
            B_new = B[:, alive_indices]          # (d, n_alive)
            mx.eval(W_gate_new, W_up_new, B_new)

            new_pool = SwiGLUCapsulePool(d, n_alive)
            new_pool.W_gate.load_weights([("weight", W_gate_new)])
            new_pool.W_up.load_weights([("weight", W_up_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())

            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": n_alive,
                "n_pruned": P_before - n_alive,
                "pct_pruned": (P_before - n_alive) / P_before * 100,
            }

        stats["per_layer"].append(layer_stats)
        stats["total_before"] += P_before
        stats["total_after"] += layer_stats["P_after"]
        stats["total_pruned"] += layer_stats["n_pruned"]

        if verbose:
            print(f"  Layer {l_idx}: {P_before} -> {layer_stats['P_after']} capsules "
                  f"({layer_stats['n_pruned']} pruned, {layer_stats['pct_pruned']:.1f}%)")

    total_pct = (stats["total_pruned"] / stats["total_before"] * 100
                 if stats["total_before"] > 0 else 0)
    stats["pct_pruned"] = total_pct

    if verbose:
        print(f"  Total: {stats['total_before']} -> {stats['total_after']} capsules "
              f"({total_pct:.1f}% pruned)")

    return stats
