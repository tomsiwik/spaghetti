"""Discriminability at N>2: does expert discriminability predict router gradients
when the router must BOTH select (top_k<N) AND mix experts?

Extends exp_gap_causal_mechanism (N=2, top_k=2, mixing only) to N=8, top_k=2
(selection + mixing).
"""

from .. import register
from ..gpt import GPT


@register("discriminability_n_gt_2", parent="gap_causal_mechanism")
class DiscriminabilityNGt2GPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
