"""Gap-as-Signal: function-space gap magnitude predicts calibration speed.

The field treats the function-space gap f(x)+g(x) != h(x) as a problem
to minimize (TIES, DARE, Model Soups). We claim: the gap IS the routing
signal. Larger gap = stronger signal = faster router calibration.

Formal prediction: calibration_steps proportional to 1/gap_magnitude.
Orthogonal experts (cos~0.0) produce maximal gap -> minimal calibration steps.
Correlated experts (cos~0.5+) produce small gap -> slow/impossible calibration.

This model is a placeholder registration; the actual experiment runs standalone.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import GPT


@register("gap_as_signal", parent="gpt")
class GapAsSignalGPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
