"""GPT with LoRA adapters on BOTH MLP and Attention (Wq/Wk) layers.

Extends the existing LoRAGPT (MLP-only LoRA) by adding rank-r_attn LoRA adapters
to the Wq and Wk projections. The hypothesis: shared attention is the proven
composition bottleneck; adapting it alongside MLP should close the composition gap.

Key design choices:
- Attention LoRA on Wq/Wk only (not Wv/Wo): modify attention ROUTING, not value content
- Lower rank for attention (default r=4) vs MLP (default r=8): minimal intervention
- LoRA deltas are pure linear: dW = (alpha/r) * A @ B, enabling exact composition
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter: out = W @ x + (alpha/r) * A @ B @ x."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        scale = (2.0 / in_dim) ** 0.5
        self.A = mx.random.normal((in_dim, rank)) * scale
        self.B = mx.zeros((rank, out_dim))

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return base_out + lora_out

    def get_delta(self) -> mx.array:
        """Return full LoRA delta: (alpha/r) * A @ B, shape (in_dim, out_dim)."""
        return (self.alpha / self.rank) * (self.A @ self.B)


class LoRACausalSelfAttention(nn.Module):
    """Causal self-attention with LoRA adapters on Wq and Wk."""

    def __init__(self, n_embd: int, n_head: int,
                 attn_rank: int = 4, attn_alpha: float = 1.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # Wq and Wk get LoRA adapters
        self.wq = LoRALinear(n_embd, n_embd, attn_rank, attn_alpha)
        self.wk = LoRALinear(n_embd, n_embd, attn_rank, attn_alpha)
        # Wv and Wo remain standard (no adaptation)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class LoRAMLP(nn.Module):
    """Two-layer MLP with LoRA adapters on both layers."""

    def __init__(self, n_embd: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.fc1 = LoRALinear(n_embd, 4 * n_embd, rank, alpha)
        self.fc2 = LoRALinear(4 * n_embd, n_embd, rank, alpha)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class AttnLoRABlock(nn.Module):
    """Transformer block with LoRA on both attention (Wq/Wk) and MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 mlp_rank: int = 8, mlp_alpha: float = 1.0,
                 attn_rank: int = 4, attn_alpha: float = 1.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = LoRACausalSelfAttention(n_embd, n_head, attn_rank, attn_alpha)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = LoRAMLP(n_embd, mlp_rank, mlp_alpha)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register("attn_lora_gpt", parent="lora_gpt")
class AttnLoRAGPT(nn.Module):
    """GPT with LoRA adapters on BOTH attention (Wq/Wk) and MLP (fc1/fc2).

    Architecture extends LoRAGPT by adding low-rank adapters to the attention
    query and key projections. This targets the proven composition bottleneck
    (shared attention) with minimal intervention.

    Attention adapters use lower rank (default 4) than MLP adapters (default 8)
    following the control theory principle of minimal intervention at the bottleneck.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 mlp_rank: int = 8, mlp_alpha: float = 1.0,
                 attn_rank: int = 4, attn_alpha: float = 1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [AttnLoRABlock(n_embd, n_head, mlp_rank, mlp_alpha,
                                      attn_rank, attn_alpha)
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

    def get_all_deltas(self) -> list:
        """Get all LoRA delta matrices (both MLP and attention).

        Returns list of (layer_idx, sublayer_name, delta_matrix).
        sublayer_name is one of: 'fc1', 'fc2', 'wq', 'wk'
        """
        deltas = []
        for l_idx, layer in enumerate(self.layers):
            deltas.append((l_idx, 'fc1', layer.mlp.fc1.get_delta()))
            deltas.append((l_idx, 'fc2', layer.mlp.fc2.get_delta()))
            deltas.append((l_idx, 'wq', layer.attn.wq.get_delta()))
            deltas.append((l_idx, 'wk', layer.attn.wk.get_delta()))
        return deltas

    def get_mlp_deltas(self) -> list:
        """Get only MLP LoRA deltas (for comparison with MLP-only baseline)."""
        deltas = []
        for l_idx, layer in enumerate(self.layers):
            deltas.append((l_idx, 'fc1', layer.mlp.fc1.get_delta()))
            deltas.append((l_idx, 'fc2', layer.mlp.fc2.get_delta()))
        return deltas

    def get_attn_deltas(self) -> list:
        """Get only attention LoRA deltas."""
        deltas = []
        for l_idx, layer in enumerate(self.layers):
            deltas.append((l_idx, 'wq', layer.attn.wq.get_delta()))
            deltas.append((l_idx, 'wk', layer.attn.wk.get_delta()))
        return deltas
