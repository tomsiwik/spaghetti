"""Skip-List Composition Test: skip-list routing under shared-base composition.

Reuses SkipListRoutingGPT architecture exactly. This module simply re-registers
it under a new name for the composition experiment so the original model's
registration stays clean.

The actual experiment logic is in run_experiment.py. This module exists only
to satisfy the arena registration convention.
"""

from .. import register
from ..skip_list_routing.skip_list_routing import SkipListRoutingGPT


@register("skip_list_composition_test", parent="skip_list_routing")
class SkipListCompositionTestGPT(SkipListRoutingGPT):
    """SkipListRoutingGPT re-registered for composition testing.

    Identical architecture. The experiment tests whether skip-list adaptive
    depth survives the shared-base composition protocol.
    """
    pass
