"""Pure-linear state capacity scaling experiment.

Tests whether the composition gap from pure-linear attention grows >3x
when scaling d_h from 16 (d=64) to 32 (d=128) or 64 (d=256).

Reuses full_gdn_stack model with layer_types=["linear"]*4 at varied n_embd.
"""

from ..full_gdn_stack.full_gdn_stack import FullGDNStackCapsuleMoEGPT  # noqa: F401
