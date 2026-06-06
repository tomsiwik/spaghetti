"""Training-time composition compatibility (Exp 11).

Tests whether auxiliary losses during domain fine-tuning can reduce the
composition gap between independently-composed and jointly-trained models.
"""

from .training_compat import TrainingCompatGPT  # noqa: F401
