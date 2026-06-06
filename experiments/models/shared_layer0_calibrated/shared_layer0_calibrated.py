"""Shared Layer 0 Calibrated: Does the sharing benefit persist after calibration?

The parent experiment (shared_layer0_pool) showed that shared Layer 0
IMPROVES quality 1.7-3.0% over full concatenation in ZERO-SHOT
composition. The adversarial review identified that calibration might
eliminate this advantage: if the router learns to downweight redundant
Layer 0 contributions, the double-counting problem that sharing solves
structurally might be absorbed by calibration.

This experiment tests whether the shared Layer 0 advantage persists
after 200-step calibration of the composed model's MLP weights.

Kill criterion: shared Layer 0 advantage over full concat disappears
(<0.5% difference) after 200-step calibration.
"""

from .. import register
from ..shared_layer0_pool.shared_layer0_pool import (
    SharedLayer0GPT,
    compose_shared_layer0,
    compose_full_concat,
    count_params,
)


@register("shared_layer0_calibrated", parent="shared_layer0_pool")
class SharedLayer0CalibratedGPT(SharedLayer0GPT):
    """SharedLayer0GPT evaluated after extended calibration.

    Architecture is identical. The experiment is in the calibration
    protocol: 200 steps of MLP fine-tuning on mixed-domain data after
    composition, for both shared and full-concat conditions.
    """
    pass
