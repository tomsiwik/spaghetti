"""Gap-as-Signal at N>2: Expert Selection vs Mixing Weights.

At N=2, top_k=2, the router only learns mixing weights (both experts always
selected). At N=4+, top_k=2, the router must also SELECT which experts to
activate -- a fundamentally harder problem.

This experiment tests whether gap-as-signal holds when routing involves actual
expert selection, not just mixing weight calibration.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import GPT


@register("gap_n_scaling", parent="gpt")
class GapNScalingGPT(GPT):
    """Placeholder for registry. Actual experiment runs standalone."""
    pass
