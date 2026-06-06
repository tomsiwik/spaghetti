"""Combined Parallel Block + Pure-Linear Attention CapsuleMoE.

This model combines two individually-proven architectural modifications:

  1. **Parallel blocks** (from exp_parallel_block_capsules):
     Single normalization, attention and capsule pool from same input.
     x_{l+1} = x_l + attn(norm(x_l)) + capsule_pool(norm(x_l))
     Proven: -0.39pp composition gap vs sequential (3 seeds).

  2. **Pure-linear attention** (from exp_pure_linear_composition):
     All layers use GatedDeltaNet linear attention, no full attention.
     Proven: +1.02% degradation vs hybrid 3:1 (7 seeds, within 5% threshold).

Combined architecture per layer:
  x -> norm -> [gdn_attn(norm_x), capsule_pool(norm_x)] -> +residual

This is the maximally-simplified composition-safe architecture:
  - No sequential norm1->attn->norm2->mlp chain
  - No full attention layers at all
  - Single norm per layer, parallel branches

Control condition: sequential blocks + hybrid 3:1 attention (the validated baseline).

Kill criterion: combined degrades >5% vs sequential+hybrid baseline.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool
from ..full_gdn_stack.full_gdn_stack import (
    FullGatedDeltaNetAttention,
    FullGDNStackBlock,
)
from ..hybrid_attention.hybrid_attention import CausalSelfAttention


# =============================================================================
# Parallel Block with GatedDeltaNet linear attention
# =============================================================================

class ParallelGDNBlock(nn.Module):
    """Parallel transformer block with GatedDeltaNet or full attention + CapsulePool.

    Single normalization, both branches from same normalized input:
        x_{l+1} = x_l + attn(norm(x_l)) + capsule_pool(norm(x_l))

    Combines parallel execution (from parallel_block_capsules) with
    GatedDeltaNet linear attention (from full_gdn_stack).
    """

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "linear", conv_kernel: int = 4):
        super().__init__()
        self.layer_type = layer_type
        self.norm = RMSNorm(n_embd)  # single norm for both branches
        if layer_type == "linear":
            self.attn = FullGatedDeltaNetAttention(n_embd, n_head, conv_kernel)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        normed = self.norm(x)
        # Parallel branches: both from same normalized input
        attn_out = self.attn(normed)
        mlp_out = self.capsule_pool(normed)
        return x + attn_out + mlp_out


# =============================================================================
# Combined: Parallel + Pure-Linear GPT
# =============================================================================

@register("parallel_pure_linear_capsule_moe", parent="full_gdn_stack_capsule_moe")
class ParallelPureLinearCapsuleMoEGPT(nn.Module):
    """GPT with parallel blocks and pure-linear (all GatedDeltaNet) attention.

    Maximum architectural simplification for composition:
      - Parallel block execution (one norm, branches from same input)
      - All layers GatedDeltaNet linear attention (no full attention scaffolding)
      - CapsulePool MLP in all layers

    This is the test condition: the simplest composition-safe architecture
    if both parallel and pure-linear modifications are compatible when combined.

    Also supports mixed layer_types for controlled comparison.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_types: list[str] | None = None,
                 conv_kernel: int = 4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)

        if layer_types is None:
            layer_types = ["linear"] * n_layer  # default: all linear

        if len(layer_types) != n_layer:
            raise ValueError(f"len(layer_types)={len(layer_types)} != n_layer={n_layer}")

        self.layer_types = layer_types
        self.layers = [
            ParallelGDNBlock(
                n_embd, n_head, n_groups, n_capsules_per_group,
                top_k_groups, uniform_routing, layer_type=lt,
                conv_kernel=conv_kernel,
            )
            for lt in layer_types
        ]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass


# =============================================================================
# Control: Sequential + Hybrid 3:1 GPT (re-export for clean comparison)
# =============================================================================

@register("sequential_hybrid_capsule_moe", parent="full_gdn_stack_capsule_moe")
class SequentialHybridCapsuleMoEGPT(nn.Module):
    """GPT with sequential blocks and hybrid 3:1 attention.

    This is the validated baseline condition:
      - Sequential block execution (norm1->attn->norm2->mlp)
      - 3:1 linear:full attention pattern
      - CapsulePool MLP in all layers

    Architecturally identical to FullGDNStackCapsuleMoEGPT with default
    layer_types. Re-registered here for clarity in the experiment.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_types: list[str] | None = None,
                 conv_kernel: int = 4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)

        if layer_types is None:
            layer_types = ["linear"] * (n_layer - 1) + ["full"]  # 3:1 hybrid

        if len(layer_types) != n_layer:
            raise ValueError(f"len(layer_types)={len(layer_types)} != n_layer={n_layer}")

        self.layer_types = layer_types
        self.layers = [
            FullGDNStackBlock(
                n_embd, n_head, n_groups, n_capsules_per_group,
                top_k_groups, uniform_routing, layer_type=lt,
                conv_kernel=conv_kernel,
            )
            for lt in layer_types
        ]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
