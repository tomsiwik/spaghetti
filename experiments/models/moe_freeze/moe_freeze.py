"""MoE + freeze-on-domain-switch: freeze best expert, recycle worst."""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..moe import MoEGPT, MoELayer, ExpertMLP


def _expert_weight_norm(expert) -> float:
    """Sum of squared norms of an expert's parameters — proxy for specialization."""
    total = 0.0
    for _, v in nn.utils.tree_flatten(expert.parameters()):
        total += mx.sum(v * v).item()
    return total


@register("moe_freeze", parent="moe")
class MoEFreezeGPT(MoEGPT):
    """MoE with lifecycle: freeze best expert, recycle worst on domain switch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frozen_experts: set[tuple[int, int]] = set()  # (layer_idx, expert_idx)

    def on_domain_switch(self, domain: str):
        """Freeze the most specialized expert per layer, recycle the least."""
        for li, layer in enumerate(self.layers):
            moe: MoELayer = layer.moe
            norms = [_expert_weight_norm(e) for e in moe.experts]
            best = max(range(len(norms)), key=lambda i: norms[i])
            worst = min(
                (i for i in range(len(norms)) if (li, i) not in self._frozen_experts),
                key=lambda i: norms[i],
                default=None,
            )

            # Freeze best
            if (li, best) not in self._frozen_experts:
                moe.experts[best].freeze()
                self._frozen_experts.add((li, best))

            # Recycle worst (reinitialize weights) if it's not the one we just froze
            if worst is not None and worst != best:
                n_embd = moe.experts[worst].fc1.weight.shape[1]
                moe.experts[worst] = ExpertMLP(n_embd)
                mx.eval(moe.experts[worst].parameters())
