"""Combined parallel+pure-linear composition at N=5 domains.

Scales the parallel_pure_linear_combined experiment (validated at N=2 with
+1.48% degradation, threshold 5%) to N=5 domains using the quintary
character split. Tests whether the combined architecture's composition gap
remains within 8% at higher domain count.

Kill criterion: N=5 composition gap >8% for parallel+pure-linear architecture.

Dependencies: exp_parallel_pure_linear_combined (proven at N=2).
"""

from .. import register
from ..parallel_pure_linear_combined.parallel_pure_linear_combined import (  # noqa: F401
    ParallelPureLinearCapsuleMoEGPT,
    SequentialHybridCapsuleMoEGPT,
)


@register("combined_n5_scaling", parent="parallel_pure_linear_capsule_moe")
class CombinedN5ScalingGPT(ParallelPureLinearCapsuleMoEGPT):
    """ParallelPureLinearCapsuleMoEGPT used for N=5 scaling validation.

    Identical architecture to the N=2 experiment. The experiment tests
    whether composition gap stays within 8% at N=5 domains. No
    architectural changes -- only the composition protocol scales.

    Registered for lineage tracking only.
    """
    pass
