"""Training Duration vs Death Rate -- Exp 17.

Measures how ReLU capsule death rate evolves as a function of fine-tuning
steps. The 54% death rate from Exp 10 was measured after 200 fine-tuning
steps. This experiment sweeps step counts from 50 to 3200 to determine
whether death is:
  (a) monotonically increasing (death accumulates, irreversible)
  (b) stabilizing at an equilibrium
  (c) recovering (early death is transient)

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("training_duration", parent="pruning_controls")
class TrainingDurationGPT(ReLURouterGPT):
    """ReLURouterGPT used for training duration vs death rate experiments.

    This model IS a ReLURouterGPT. The experiment varies fine-tuning
    step counts and profiles death rates at each checkpoint. No
    architectural changes.

    Registered for lineage tracking only.
    """
    pass
