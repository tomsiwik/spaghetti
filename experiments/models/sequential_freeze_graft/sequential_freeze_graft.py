"""Sequential Freeze-Graft-Calibrate Protocol for N>2 Domains.

Extends split_freeze_protocol to test the sequential contribution protocol
at realistic domain counts (N=2,3,4). Each graft step:
  1. Freeze the most recently trained subtree
  2. Reinitialize the next subtree for a new domain
  3. Train the new subtree on domain data
  4. Calibrate all unfrozen parameters on mixed data
  5. Measure degradation on ALL previously frozen domains

The tree uses depth-3 (8 leaves, 7 gates) with a progressive halving
allocation:
  - Graft 1 (N=2): domain A gets leaves 0-3, domain B gets leaves 4-7
  - Graft 2 (N=3): freeze B's 4-5, domain C gets leaves 6-7
  - Graft 3 (N=4): freeze C's leaf 6, domain D gets leaf 7

Kill criteria:
  - Cumulative degradation grows >2x between N=2 and N=4
  - Calibration cost per graft grows superlinearly with N
"""

from .. import register
from ..hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT


@register("sequential_freeze_graft", parent="split_freeze_protocol")
class SequentialFreezeGraftGPT(HierarchicalTreeGPT):
    """HierarchicalTreeGPT with sequential freeze-graft protocol.

    Architecturally identical to HierarchicalTreeGPT. Exists as a separate
    registry entry to track the sequential freeze-graft experiment results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
