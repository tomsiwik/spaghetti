"""Freeze-Then-Prune Protocol -- Exp freeze_then_prune.

Compare two pruning protocols:
  Protocol A (freeze-then-prune): Train fully, freeze ALL layers, profile
    dead capsules (now permanently dead per exp20), then prune.
  Protocol B (mid-training prune): Profile and prune at intermediate
    checkpoints while training is ongoing (revival still active).

The key insight from exp20 is that inter-layer coupling drives 79-94% of
revival. After training completes and weights are frozen, dead neurons are
permanently dead -- no upstream changes can revive them. This makes
post-training profiling the ground truth for which capsules are truly dead.

Mid-training pruning risks:
  1. False positives: pruning capsules that would have revived
  2. Lower yield: fewer capsules appear dead when revival is active
  3. Quality loss: removing capsules that downstream layers depend on

Kill criteria:
  1. Post-freeze pruning yields <5pp more dead capsules than mid-training
  2. Post-freeze pruned model quality degrades >3% vs mid-training pruned
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("freeze_then_prune", parent="death_recovery_mechanism")
class FreezeThenPruneGPT(ReLURouterGPT):
    """ReLURouterGPT used for freeze-then-prune protocol experiments.

    This model IS a ReLURouterGPT. The experiment compares pruning protocols
    (post-training vs mid-training). No architectural changes.

    Registered for lineage tracking only.
    """
    pass
