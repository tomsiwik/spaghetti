"""Minimal Graft Recalibration — which gates need recalibration after grafting?

Extends SubtreeGraftingGPT. Architecturally identical; this model exists
to track the selective recalibration experiment independently.

The question: subtree grafting showed root-only calibration is insufficient
(+2.42% vs weight averaging) while all-gates calibration works (+0.67%).
Which specific gates need recalibration?

Gate topology in a depth-3 binary tree (7 internal gates):
    Gate 0: root (domain router)
    Gates 1, 2: graft-point gates (top of each domain subtree)
    Gates 3, 4: children of gate 1 (deep left subtree)
    Gates 5, 6: children of gate 2 (deep right subtree)

Recalibration conditions tested:
    (a) root-only: gate 0 only (1 of 7 gates)
    (b) root+graft-point: gates 0, 1, 2 (3 of 7 gates)
    (c) all-gates: gates 0-6 (7 of 7 gates, baseline)
"""

from .. import register
from ..subtree_grafting.subtree_grafting import SubtreeGraftingGPT


@register("minimal_graft_recal", parent="subtree_grafting")
class MinimalGraftRecalGPT(SubtreeGraftingGPT):
    """SubtreeGraftingGPT with selective gate recalibration experiment.

    Architecturally identical. Exists as a separate registry entry to track
    selective recalibration results independently.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
