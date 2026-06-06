"""Profiling Noise Quantification -- Exp 12 (VISION.md #12).

Exp 18 found that 28.1% of capsules dead at S=100 revive by S=3200, with
D->A transitions accelerating over training. The adversarial review flagged
that profiling with only 640 samples (20 batches x 32) could produce
false-positive D->A transitions for borderline capsules near f=0.

This experiment quantifies the noise floor by profiling each checkpoint
TWICE with DIFFERENT random batches. Any D->A transitions that appear
within a single checkpoint (same weights, different samples) are
definitionally noise -- the capsule did not actually change state.

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT.
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("profiling_noise", parent="capsule_revival")
class ProfilingNoiseGPT(ReLURouterGPT):
    """ReLURouterGPT used for profiling noise quantification experiments.

    This model IS a ReLURouterGPT. The experiment profiles the same
    checkpoint with different random batches to quantify noise in
    dead/alive classification. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
