"""Per-Capsule Revival Tracking -- Exp 18.

Exp 17 showed aggregate death rate decreases over training (55% -> 47%),
but did not track WHICH capsules died and revived. This experiment tracks
per-capsule identity across training checkpoints to distinguish:

  (a) True revival: the SAME capsules that were dead at S=100 become alive
      at later checkpoints (inter-layer coupling revival)
  (b) Population turnover: DIFFERENT capsules die and revive at similar
      rates, giving the appearance of aggregate decrease

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("capsule_revival", parent="pruning_controls")
class CapsuleRevivalGPT(ReLURouterGPT):
    """ReLURouterGPT used for per-capsule revival tracking experiments.

    This model IS a ReLURouterGPT. The experiment profiles per-capsule
    death/alive status at each training checkpoint and tracks identity
    persistence. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
