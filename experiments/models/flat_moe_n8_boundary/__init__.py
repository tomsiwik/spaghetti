"""Flat MoE N=8 Boundary: Composition quality + identity at N=8 domains.

Tests the flat MoE composition protocol (concatenation + pruning + calibration)
at the N=8 extrapolated safe limit. Measures both composition gap and Jaccard.

Kill criteria:
  - composition gap >10% at N=8
  - combined Jaccard <0.60
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("flat_moe_n8_boundary", parent="n8_identity_boundary")
class FlatMoeN8BoundaryGPT(ReLURouterGPT):
    """ReLURouterGPT used for N=8 flat MoE boundary experiment.

    This model IS a ReLURouterGPT. The experiment combines:
    1. Composition gap measurement (quality vs joint training)
    2. Capsule identity Jaccard tracking

    Registered for lineage tracking only.
    """
    pass
