"""Pre-composition pruning pipeline -- Exp prune_before_compose_e2e."""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("prune_before_compose", parent="capsule_identity")
class PruneBeforeComposeGPT(ReLURouterGPT):
    """ReLURouterGPT with pre-composition pruning pipeline.

    This model IS a ReLURouterGPT. The experiment validates a different
    pruning ORDER: prune each domain model independently BEFORE composing,
    rather than composing first then pruning.

    Enabled by Exp 16's finding that dead capsule identity is preserved
    across composition (Jaccard=0.895, overlap coefficient=0.986).
    """
    pass
