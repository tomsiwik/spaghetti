"""Gap-as-Signal in Practical Cosine Regime (cos < 0.3).

The parent experiment (gap_as_signal) showed r^2=0.74 across cos={0.0..0.9},
but the adversarial review identified a leverage effect: cos>=0.7 drives most
of the correlation. In the practical regime where real LoRA adapters live
(cos < 0.3, since natural cos ~ 0.000), quality differences were only ~0.2pp.

This experiment zooms into [0.0, 0.3] with finer granularity (7 levels at
0.05 increments) and more seeds (5) to determine if gap-as-signal provides
meaningful discrimination in the regime that matters.

Kill criteria:
- quality difference between cos=0.0 and cos=0.3 is < 0.5pp
- gap magnitude variation in [0.0, 0.3] is within noise
"""

from .. import register
from ..gpt import GPT


@register("gap_practical_regime", parent="gap_as_signal")
class GapPracticalRegimeGPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
