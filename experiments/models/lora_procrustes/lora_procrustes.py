"""LoRA-augmented GPT for linear Procrustes decomposition experiments.

LoRA adapters inject low-rank deltas dW = (alpha/r) * A @ B into frozen MLP layers.
These deltas are PURE LINEAR -- no activation function in the delta path.
This enables exact shared/unique decomposition in both weight and function space:
  shared_dW @ x + unique_dW @ x = dW @ x  (exactly)

Unlike CapsuleGroups with ReLU (where decomposition is approximate), LoRA delta
decomposition is mathematically exact. This is the "linear resurrection" of the
killed Procrustes experiment (exp3_procrustes_decomp).
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter: out = W @ x + (alpha/r) * A @ B @ x.

    The LoRA delta dW = (alpha/r) * A @ B is a rank-r matrix (r << d).
    A: (in_dim, r), B: (r, out_dim). Only A, B are trainable.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        scale = (2.0 / in_dim) ** 0.5  # Kaiming-like init for A
        self.A = mx.random.normal((in_dim, rank)) * scale
        self.B = mx.zeros((rank, out_dim))
        self.A = self.A  # ensure tracked

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return base_out + lora_out

    def get_delta(self) -> mx.array:
        """Return the full LoRA delta matrix: (alpha/r) * A @ B, shape (in_dim, out_dim)."""
        return (self.alpha / self.rank) * (self.A @ self.B)


class LoRAMLP(nn.Module):
    """Two-layer MLP with LoRA adapters on both layers."""

    def __init__(self, n_embd: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.fc1 = LoRALinear(n_embd, 4 * n_embd, rank, alpha)
        self.fc2 = LoRALinear(4 * n_embd, n_embd, rank, alpha)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class LoRABlock(nn.Module):
    """Transformer block with LoRA-augmented MLP."""

    def __init__(self, n_embd: int, n_head: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = LoRAMLP(n_embd, rank, alpha)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register("lora_gpt", parent="gpt")
class LoRAGPT(nn.Module):
    """GPT with LoRA adapters on MLP layers.

    Architecture identical to GPT but MLP fc1/fc2 layers have LoRA adapters.
    During fine-tuning, only LoRA A/B matrices are trained (base frozen).
    The LoRA deltas are pure linear: dW = (alpha/r) * A @ B.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 lora_rank: int = 8, lora_alpha: float = 1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [LoRABlock(n_embd, n_head, lora_rank, lora_alpha)
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
        return mx.array(0.0)

    def on_domain_switch(self, domain: str):
        pass

    def freeze_base(self):
        """Freeze everything except LoRA A/B parameters."""
        self.freeze()
        for layer in self.layers:
            # Unfreeze LoRA params (A and B are direct attributes, not nn.Linear)
            layer.mlp.fc1.A = layer.mlp.fc1.A
            layer.mlp.fc1.B = layer.mlp.fc1.B
            layer.mlp.fc2.A = layer.mlp.fc2.A
            layer.mlp.fc2.B = layer.mlp.fc2.B

    def get_lora_params(self) -> list:
        """Get list of (name, param) for all LoRA A/B matrices."""
        params = []
        for l_idx, layer in enumerate(self.layers):
            params.append((f"layers.{l_idx}.mlp.fc1.A", layer.mlp.fc1.A))
            params.append((f"layers.{l_idx}.mlp.fc1.B", layer.mlp.fc1.B))
            params.append((f"layers.{l_idx}.mlp.fc2.A", layer.mlp.fc2.A))
            params.append((f"layers.{l_idx}.mlp.fc2.B", layer.mlp.fc2.B))
        return params

    def get_all_deltas(self) -> list:
        """Get all LoRA delta matrices.

        Returns list of (layer_idx, 'fc1'|'fc2', delta_matrix).
        """
        deltas = []
        for l_idx, layer in enumerate(self.layers):
            deltas.append((l_idx, 'fc1', layer.mlp.fc1.get_delta()))
            deltas.append((l_idx, 'fc2', layer.mlp.fc2.get_delta()))
        return deltas
