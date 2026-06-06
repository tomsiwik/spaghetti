"""Softmax collision quantification and mitigation for MoE routing.

Quantifies the softmax routing "collision" problem: when top-k expert scores
are within epsilon of each other, the router is uncertain and routing quality
degrades. Measures collision rate scaling with N experts.

Mitigations tested:
1. Temperature scaling: scores / T before softmax (T < 1 sharpens, T > 1 flattens)
2. Margin-based auxiliary loss: penalize small gaps between top-1 and top-2

Kill criteria:
- KC1: Collision rate does not increase with N (contradicts scaling concern)
- KC2: Collision-aware routing provides <0.5% quality improvement over naive softmax

Prior art:
- Switch Transformers (Fedus et al. 2021): load balancing auxiliary loss
- DeepSeek-V3: auxiliary-loss-free load balancing via bias term
- Cuckoo experiment (this project): measured 57.4% collision rate at N=8
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class TempScaledCapsulePool(nn.Module):
    """CapsulePool with temperature-scaled softmax routing.

    Temperature T controls softmax sharpness:
    - T < 1: sharper distribution, more decisive routing (fewer collisions)
    - T = 1: standard softmax (baseline)
    - T > 1: flatter distribution, more uniform routing (more collisions)
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32, top_k: int = 2,
                 temperature: float = 1.0):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k
        self.temperature = temperature

        self.router = nn.Linear(n_embd, n_groups, bias=False)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]
        self._gate_probs = None
        self._raw_scores = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        scores = self.router(x)  # (B, T, G)
        self._raw_scores = scores

        # Temperature-scaled softmax
        scaled_scores = scores / self.temperature
        probs = mx.softmax(scaled_scores, axis=-1)
        self._gate_probs = probs

        # Top-k selection
        top_vals = mx.topk(scaled_scores, self.top_k, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scaled_scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1,
                                              keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)
        return out

    def balance_loss(self) -> mx.array:
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_groups * mx.sum(mean_probs * mean_probs)


class MarginCapsulePool(nn.Module):
    """CapsulePool with margin-based auxiliary loss.

    Adds a penalty when the gap between top-1 and top-2 softmax scores
    is below a target margin. This encourages the router to make more
    decisive selections, reducing collisions.

    Loss: margin_loss = mean(max(0, target_margin - (top1 - top2)))
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32, top_k: int = 2,
                 target_margin: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k
        self.target_margin = target_margin

        self.router = nn.Linear(n_embd, n_groups, bias=False)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]
        self._gate_probs = None
        self._margin_loss = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        scores = self.router(x)  # (B, T, G)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Compute margin loss: penalize small top1-top2 gaps
        sorted_probs = mx.sort(probs, axis=-1)
        top1 = sorted_probs[..., -1]  # (B, T)
        top2 = sorted_probs[..., -2]  # (B, T)
        gap = top1 - top2  # (B, T)
        # Hinge loss: penalize when gap < target_margin
        self._margin_loss = mx.mean(mx.maximum(
            mx.array(0.0), self.target_margin - gap
        ))

        # Top-k selection (standard)
        top_vals = mx.topk(scores, self.top_k, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1,
                                              keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)
        return out

    def balance_loss(self) -> mx.array:
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_groups * mx.sum(mean_probs * mean_probs)

    def margin_loss(self) -> mx.array:
        if self._margin_loss is None:
            return mx.array(0.0)
        return self._margin_loss


class TempScaledBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = TempScaledCapsulePool(
            n_embd, n_groups, n_capsules_per_group, top_k, temperature
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


class MarginBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2, target_margin: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = MarginCapsulePool(
            n_embd, n_groups, n_capsules_per_group, top_k, target_margin
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("temp_scaled_moe", parent="capsule_moe")
class TempScaledMoEGPT(nn.Module):
    """GPT with temperature-scaled softmax routing."""

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.n_embd = n_embd
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [TempScaledBlock(n_embd, n_head, n_groups,
                                       n_capsules_per_group, top_k, temperature)
                       for _ in range(n_layer)]
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


@register("margin_loss_moe", parent="capsule_moe")
class MarginLossMoEGPT(nn.Module):
    """GPT with margin-based collision-aware routing."""

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2, target_margin: float = 0.1,
                 margin_weight: float = 0.1):
        super().__init__()
        self.n_embd = n_embd
        self.margin_weight = margin_weight
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [MarginBlock(n_embd, n_head, n_groups,
                                   n_capsules_per_group, top_k, target_margin)
                       for _ in range(n_layer)]
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
            pool = layer.capsule_pool
            total = total + pool.balance_loss()
            total = total + self.margin_weight * pool.margin_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
