"""Pre-composition pruning at N=5 domains.

Scales prune_before_compose (validated at N=2) to N=5 domains.
N=5 identity Jaccard is 0.792 (vs 0.895 at N=2). Tests whether the
pre-prune-then-compose pipeline still matches compose-then-prune
at this lower identity overlap.

Kill criterion: pre-prune-then-compose quality degrades >3% vs
compose-then-prune at N=5.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("prune_compose_n5", parent="prune_before_compose")
class PruneComposeN5GPT(ReLURouterGPT):
    """ReLURouterGPT used for N=5 pre-composition pruning pipeline.

    This model IS a ReLURouterGPT. The experiment validates that the
    pre-composition pruning pipeline (profile->prune->compose->calibrate)
    still works at N=5 domains where identity Jaccard has degraded to 0.792.

    Registered for lineage tracking only.
    """
    pass
