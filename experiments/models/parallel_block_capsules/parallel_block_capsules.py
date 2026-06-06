"""Parallel Block CapsuleMoE — attention and capsule pool compute from same normalized input.

Architecture comparison:

  Sequential (standard):
    x -> norm1 -> attn -> +residual -> norm2 -> capsule_pool -> +residual
    Capsule pool sees POST-attention representation.

  Parallel (Tiny Aya style):
    x -> norm -> [attn, capsule_pool] -> +residual (both outputs summed)
    Capsule pool sees PRE-attention representation (same input as attention).

The parallel formulation changes the composition dynamics:
- In sequential blocks, capsule adapters operate on attention-processed features.
  Since shared attention is the composition bottleneck (Exp 4), capsules receive
  an already-interfered signal.
- In parallel blocks, capsules operate on the same normalized input as attention.
  The adapter pathway is independent of attention, potentially reducing the
  bottleneck effect.

This mirrors Cohere's Tiny Aya architecture (3.35B) which uses a single LayerNorm
per block with parallel attention + MLP branches.

Key difference: Tiny Aya uses a single norm (CohereLayerNorm, which is mean-centering
LayerNorm without bias). We use a single RMSNorm per block for consistency with
the micro codebase.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsulePool, CapsuleGroup


# =============================================================================
# Parallel Block: attention and capsule pool from same normalized input
# =============================================================================

class ParallelCapsuleBlock(nn.Module):
    """Transformer block with PARALLEL attention and capsule pool.

    Both attention and capsule pool receive the same norm(x) input.
    Their outputs are summed into a single residual update:

        x_{l+1} = x_l + attn(norm(x_l)) + capsule_pool(norm(x_l))

    This is the Tiny Aya / Cohere parallel block pattern.
    Uses a single normalization layer (not two).
    """

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.norm = RMSNorm(n_embd)  # single norm for both branches
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
# Sequential Block (control): standard pre-norm with two norms
# =============================================================================

class SequentialCapsuleBlock(nn.Module):
    """Standard transformer block with SEQUENTIAL attention then capsule pool.

    This is the standard pre-norm architecture used in GPT, Llama, Qwen, etc.:

        h = x + attn(norm1(x))
        x_{l+1} = h + capsule_pool(norm2(h))

    Capsule pool sees post-attention features (attention output + residual, re-normed).
    Uses two normalization layers.
    """

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


# =============================================================================
# Parallel CapsuleMoE GPT
# =============================================================================

@register("parallel_capsule_moe", parent="capsule_moe")
class ParallelCapsuleMoEGPT(nn.Module):
    """GPT with parallel blocks (Tiny Aya style) and CapsulePool MLP.

    All layers use parallel attention + capsule pool from the same normalized input.
    This tests whether parallel block composition reduces the shared attention
    bottleneck identified in Exp 4.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [
            ParallelCapsuleBlock(n_embd, n_head, n_groups, n_capsules_per_group,
                                 top_k_groups, uniform_routing)
            for _ in range(n_layer)
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
# Sequential CapsuleMoE GPT (explicit control — same as CapsuleMoEGPT)
# =============================================================================

@register("sequential_capsule_moe", parent="capsule_moe")
class SequentialCapsuleMoEGPT(nn.Module):
    """GPT with sequential blocks (standard pre-norm) and CapsulePool MLP.

    Explicit control condition for the parallel block experiment.
    Architecturally identical to CapsuleMoEGPT but uses the SequentialCapsuleBlock
    class for code symmetry with the parallel variant.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [
            SequentialCapsuleBlock(n_embd, n_head, n_groups, n_capsules_per_group,
                                   top_k_groups, uniform_routing)
            for _ in range(n_layer)
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
