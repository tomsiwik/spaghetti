"""Warmup Fraction Sensitivity -- Exp 20.

Measures how the warmup fraction in a warmup+cosine LR schedule affects
the ReLU capsule death spike and equilibrium death rate.

Exp 19 used 10% warmup and found it eliminates 74% of the death spike.
This experiment sweeps warmup fraction {1%, 2%, 5%, 10%, 20%} to find
the minimum effective warmup for death prevention.

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("warmup_sweep", parent="lr_schedule_death")
class WarmupSweepGPT(ReLURouterGPT):
    """ReLURouterGPT used for warmup fraction sensitivity experiments.

    This model IS a ReLURouterGPT. The experiment varies the warmup
    fraction in a warmup+cosine LR schedule during fine-tuning and
    profiles death rates at checkpoints. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
