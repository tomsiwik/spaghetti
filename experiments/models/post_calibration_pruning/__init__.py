"""Post-Calibration Pruning Safety (exp_post_calibration_pruning).

Tests the full pipeline: compose -> calibrate 100 steps -> profile -> prune.
Compares against pre-composition pruning (from prune_before_compose).

Kill criteria:
1. post-calibration pruning degrades quality >2% vs pre-calibration pruning
2. revival rate after 100-step calibration >5% (contradicts 2.9% finding)
"""

from .. import register
from ..relu_router.relu_router import ReLURouterGPT


@register("post_calibration_pruning", parent="revival_under_composition")
class PostCalibrationPruningGPT(ReLURouterGPT):
    """ReLURouterGPT for post-calibration pruning experiments.

    Thin wrapper for lineage tracking. No architectural changes.
    The experiment validates a pipeline ordering: compose first,
    calibrate, THEN profile and prune.
    """
    pass
