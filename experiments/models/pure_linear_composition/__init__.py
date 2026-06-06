"""Pure-linear composition control experiment.

Tests 4:0 all-linear-attention as a control against the 3:1 hybrid.
Reuses full_gdn_stack model with layer_types=["linear"]*4.
"""

from ..full_gdn_stack.full_gdn_stack import FullGDNStackCapsuleMoEGPT  # noqa: F401
