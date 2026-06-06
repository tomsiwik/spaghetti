"""LR Schedule Impact on Death Trajectory -- Exp 19.

Measures how different learning rate schedules (constant, warmup-only,
cosine-only, warmup+cosine) affect the ReLU capsule death trajectory.

Exp 17 established the "spike and slow decay" death pattern under constant
LR (3e-3). Macro-scale training universally uses warmup + cosine decay.
This experiment tests whether the death trajectory changes qualitatively
under standard macro LR schedules.

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("lr_schedule_death", parent="training_duration")
class LRScheduleDeathGPT(ReLURouterGPT):
    """ReLURouterGPT used for LR schedule vs death rate experiments.

    This model IS a ReLURouterGPT. The experiment varies the learning
    rate schedule during fine-tuning and profiles death rates at
    checkpoints. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
