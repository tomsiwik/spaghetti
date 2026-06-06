"""Death Recovery Mechanism -- Exp 20.

Isolate the inter-layer coupling revival mechanism by selectively
freezing layers during fine-tuning. If revival in layer l is driven
by weight updates in layers 0..l-1 shifting the input distribution,
then freezing those upstream layers should suppress revival in layer l.

The model class is a thin wrapper for lineage tracking.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("death_recovery_mechanism", parent="pruning_controls")
class DeathRecoveryMechanismGPT(ReLURouterGPT):
    """ReLURouterGPT used for layer-freeze revival isolation experiments.

    This model IS a ReLURouterGPT. The experiment selectively freezes
    MLP layers during fine-tuning to isolate which layers drive revival
    in downstream layers. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
