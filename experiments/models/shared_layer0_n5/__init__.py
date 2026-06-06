"""Shared Layer 0 at N=5 domains.

Tests whether the shared Layer 0 capsule pool composition protocol
scales from N=2 to N=5 domains. Parent experiment (shared_layer0_pool)
proved 1.7-3.0% improvement at N=2. This experiment tests the same
protocol with the quintary split (a-e, f-j, k-o, p-t, u-z).

Kill criteria:
  - shared Layer 0 pool degrades quality >2% vs per-domain Layer 0 pools at N=5
  - Layer 0 cross-domain Jaccard drops below 0.40 at N=5
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("shared_layer0_n5", parent="shared_layer0_pool")
class SharedLayer0N5GPT(ReLURouterGPT):
    """ReLURouterGPT with shared Layer 0 composition at N=5 domains.

    Architecture is identical to ReLURouterGPT. The experiment tests
    the composition protocol from shared_layer0_pool at N=5 scale.
    Registered for lineage tracking only.
    """
    pass
