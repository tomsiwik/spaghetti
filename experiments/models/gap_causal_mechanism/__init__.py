"""Gap Causal Mechanism: does gap magnitude drive router gradient magnitude?

Follow-up to gap_as_signal (proven, r^2=0.74 correlation). This experiment
establishes CAUSATION: larger gap -> larger per-token router gradients -> faster learning.

The key new measurement: extract router gradient norms during calibration at each
cosine level, then correlate gap_magnitude vs mean_router_gradient_magnitude.
"""

from .. import register
from ..gpt import GPT


@register("gap_causal_mechanism", parent="gap_as_signal")
class GapCausalMechanismGPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
