"""Sparse Router — top-k sweep instrumentation for capsule MoE composition."""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..capsule_moe.capsule_moe import CapsuleMoEGPT


@register("sparse_router", parent="capsule_moe")
class SparseRouterGPT(CapsuleMoEGPT):
    """CapsuleMoEGPT with runtime top-k control and router diagnostics.

    No new parameters — identical architecture to capsule_moe.
    Adds methods for sweeping top_k and analyzing router behavior
    (entropy, concentration, group frequency).
    """

    def set_top_k(self, k: int):
        """Change top-k group selection for all layers."""
        for layer in self.layers:
            layer.capsule_pool.top_k_groups = k

    def get_top_k(self) -> int:
        return self.layers[0].capsule_pool.top_k_groups

    def router_stats(self) -> dict:
        """Router statistics from last forward pass.

        Returns per-layer: entropy, entropy_ratio, concentration_1, group_freqs.
        Call after a forward pass to populate _gate_probs.
        """
        n_groups = self.layers[0].capsule_pool.n_groups
        h_max = math.log(n_groups)

        stats = {"entropy": [], "entropy_ratio": [],
                 "concentration_1": [], "group_freqs": []}

        for layer in self.layers:
            probs = layer.capsule_pool._gate_probs  # (B, T, G) raw softmax
            if probs is None:
                for k in stats:
                    stats[k].append(None)
                continue
            mx.eval(probs)

            # Shannon entropy H(p) = -sum(p * log(p))
            log_p = mx.where(probs > 1e-10, mx.log(probs), mx.array(0.0))
            h = -mx.sum(probs * log_p, axis=-1)  # (B, T)
            mean_h = mx.mean(h).item()
            stats["entropy"].append(mean_h)
            stats["entropy_ratio"].append(mean_h / h_max if h_max > 0 else 0)

            # Top-1 concentration C_1 = max(p)
            top1 = mx.max(probs, axis=-1)  # (B, T)
            stats["concentration_1"].append(mx.mean(top1).item())

            # Group activation frequency (top-1 selection)
            top1_idx = mx.argmax(probs, axis=-1)  # (B, T)
            mx.eval(top1_idx)
            freqs = []
            for g in range(n_groups):
                freq = mx.mean((top1_idx == g).astype(mx.float32)).item()
                freqs.append(freq)
            stats["group_freqs"].append(freqs)

        return stats
