"""Capsule Identity Tracking Across Composition (Exp 16).

Compares per-capsule dead/alive identity between single-domain and
composed settings to determine whether the SAME capsules die in both,
or whether composition creates novel death patterns.

Kill criterion: Jaccard overlap of dead sets < 50% between single-domain
and composed settings.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("capsule_identity", parent="capsule_revival")
class CapsuleIdentityGPT(ReLURouterGPT):
    """ReLURouterGPT used for capsule identity tracking across composition.

    This model IS a ReLURouterGPT. The experiment profiles per-capsule
    death/alive status in single-domain models and in composed models,
    then compares the identity of dead sets. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
