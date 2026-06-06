"""MoE GPT — MLX port of micromoe.py. Replaces MLP with Mixture of Experts."""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention, Block


class ExpertMLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, n_embd: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = [ExpertMLP(n_embd) for _ in range(n_experts)]
        self._gate_probs = None  # stored for aux_loss

    def __call__(self, x):
        # x: (B, T, D)
        scores = self.router(x)  # (B, T, N)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k selection: zero out non-top-k
        top_vals = mx.topk(scores, self.top_k, axis=-1)  # (B, T, k)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)  # (B, T, 1)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Run all experts, weight by masked probs (cheaper than sparse at small N)
        out = mx.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * expert(x)
        return out

    def balance_loss(self) -> mx.array:
        """L = N * sum(mean_prob_i^2). Minimized at uniform 1/N."""
        if self._gate_probs is None:
            return mx.array(0.0)
        # mean probability per expert across batch and time
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (N,)
        return self.n_experts * mx.sum(mean_probs * mean_probs)


class MoEBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.moe = MoELayer(n_embd, n_experts, top_k)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


@register("moe", parent="gpt")
class MoEGPT(nn.Module):
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [MoEBlock(n_embd, n_head, n_experts, top_k) for _ in range(n_layer)]
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
            total = total + layer.moe.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
