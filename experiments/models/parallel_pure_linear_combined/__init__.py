"""Combined parallel blocks + pure-linear attention for composition.

Tests the combined effect of two individually-proven architectural modifications:
  1. Parallel blocks (attention + capsule pool from same normalized input)
  2. Pure-linear attention (all layers GatedDeltaNet, no full attention)

Dependencies: exp_parallel_block_capsules (proven), exp_pure_linear_composition (proven).
"""

from .parallel_pure_linear_combined import (  # noqa: F401
    ParallelPureLinearCapsuleMoEGPT,
    SequentialHybridCapsuleMoEGPT,
)
