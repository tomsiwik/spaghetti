"""N=8 Identity Boundary: Capsule identity Jaccard at N=8 (predicted safe limit).

N=5 experiment measured combined Jaccard = 0.792 with linear degradation
~0.026 per additional domain. Extrapolated safe limit ~N=8 where J ~ 0.71.
This experiment validates that extrapolation by scaling to 8 domains.

Kill criteria:
  - Combined Jaccard at N=8 drops below 0.70
  - Per-domain minimum Jaccard below 0.50 for any domain
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("n8_identity_boundary", parent="n5_identity_scaling")
class N8IdentityBoundaryGPT(ReLURouterGPT):
    """ReLURouterGPT used for N=8 capsule identity tracking.

    This model IS a ReLURouterGPT. The experiment profiles per-capsule
    death/alive status in N=8 single-domain models and in their 8-way
    composition, then compares identity of dead sets. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
