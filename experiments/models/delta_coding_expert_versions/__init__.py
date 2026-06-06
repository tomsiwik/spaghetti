"""Delta Coding for Expert Version Management.

Treats expert weight updates like video compression: keyframes (I-frames) store
full snapshots, deltas (P-frames) store only inter-version differences.
Reconstructs any version via: v_n = keyframe + sum(deltas_1..n).

Extends LoRA delta coding from single-shot adaptation to sequential version chains.
"""

from .. import register
from .delta_coding_expert_versions import DeltaCodedLoRAGPT

# Register for the arena
register("delta_coded_lora_gpt", parent="lora_gpt")(DeltaCodedLoRAGPT)
