"""Calibration LR/steps scaling law as function of N.

If router gradients scale as k/N, optimal calibration LR should scale as N/k
and steps as N/k. Derives the scaling law for the contribution protocol:
how many calibration steps and what LR when adding the Nth expert?
"""

from .. import register
from ..gpt import GPT


@register("calibration_lr_scaling", parent="discriminability_n_gt_2")
class CalibrationLRScalingGPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
