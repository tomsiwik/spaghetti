"""Dense backpropagation to restore gradient strength at N>2.

Forward pass: top-k expert selection (sparse, efficient).
Backward pass: gradients flow through ALL experts (dense, informative).

This should restore N=2-level gradient magnitude during calibration while
keeping inference efficient.
"""

from .. import register
from ..gpt import GPT


@register("dense_backprop_calibration", parent="discriminability_n_gt_2")
class DenseBackpropCalibrationGPT(GPT):
    """Placeholder for registry. Actual experiment uses LoRA composition pipeline."""
    pass
