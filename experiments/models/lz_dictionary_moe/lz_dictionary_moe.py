"""LZ Dictionary MoE -- shared dictionary of MLP sub-modules.

Inspired by LZ77/LZ78 compression: factor expert MLPs into references to a
shared "dictionary" of low-rank sub-modules + per-expert residual deltas.

Each expert is composed as:
    expert_i(x) = sum_j alpha_{i,j} * dict_j(x) + delta_i(x)

where:
    dict_j: shared low-rank MLP sub-module (d -> r -> d), the "codebook entry"
    alpha_{i,j}: learned per-expert coefficients selecting dictionary entries
    delta_i: small per-expert residual (d -> r_delta -> d), the "literal"

The dictionary is shared across all experts in a layer, analogous to how
LZ compression maintains a dictionary of previously seen patterns.

Prior art:
    - StructMoE (ICML 2024): Low-Rank Experts dynamically added per-expert
    - L-MoE (2025): Soft weighted average of LoRA expert parameters
    - AoE (ICLR 2025): Low-rank factorization for self-routing
    - Union-of-Experts (2024): Expert SVD reveals shared structure
    - DeepSeek-MoE: Shared experts + routed experts

Key difference from prior work:
    We decompose WITHIN each expert using a SHARED codebook, rather than
    sharing entire experts (DeepSeek) or dynamically generating parameters
    (HyperMoE). This is closer to VQ-based weight compression but applied
    at the functional level (sub-networks, not individual weights).
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class DictionaryEntry(nn.Module):
    """A single codebook entry: a low-rank MLP sub-module.

    Computes: x -> down(x) -> relu -> up(result) -> output
    Shape: (B, T, d) -> (B, T, r) -> (B, T, d)

    These are SHARED across all experts in a layer.
    """

    def __init__(self, n_embd: int, rank: int):
        super().__init__()
        self.down = nn.Linear(n_embd, rank, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)

    def __call__(self, x):
        return self.up(nn.relu(self.down(x)))


class DictionaryExpert(nn.Module):
    """An expert composed from dictionary entries + residual.

    expert_i(x) = sum_j alpha_{i,j} * dict_j(x) + delta_i(x)

    Parameters:
        alpha: (D,) learned coefficients for D dictionary entries
        delta: low-rank residual MLP (d -> r_delta -> d)

    The alpha coefficients are softmax-normalized for stability, but NOT
    sparsified -- all dictionary entries contribute (soft composition).
    Utilization is measured post-hoc via alpha magnitude.
    """

    def __init__(self, n_embd: int, n_dict: int, delta_rank: int):
        super().__init__()
        self.n_dict = n_dict
        # Raw alpha logits -- softmax gives composition weights
        self.alpha_logits = mx.random.normal((n_dict,)) * 0.01
        # Per-expert residual (low-rank)
        self.delta_down = nn.Linear(n_embd, delta_rank, bias=False)
        self.delta_up = nn.Linear(delta_rank, n_embd, bias=False)

    def __call__(self, x, dictionary):
        """Forward pass using shared dictionary.

        Args:
            x: (B, T, d) input hidden states
            dictionary: list of DictionaryEntry modules (shared)

        Returns:
            (B, T, d) expert output
        """
        # Composition weights via softmax over alpha logits
        alpha = mx.softmax(self.alpha_logits)  # (D,)

        # Weighted sum of dictionary entry outputs
        out = mx.zeros_like(x)
        for j in range(self.n_dict):
            out = out + alpha[j] * dictionary[j](x)

        # Add per-expert residual
        delta = self.delta_up(nn.relu(self.delta_down(x)))
        out = out + delta

        return out

    def get_alpha_weights(self):
        """Return normalized alpha for diagnostics."""
        return mx.softmax(self.alpha_logits)


class DictionaryMoELayer(nn.Module):
    """MoE layer where experts share a dictionary of sub-modules.

    Architecture:
        - D shared dictionary entries (low-rank MLPs), each d -> r -> d
        - N experts, each composed as: sum_j(alpha_j * dict_j) + delta
        - Standard softmax top-k routing over experts

    Parameter count comparison with standard MoE:
        Standard: N * (d * 4d + 4d * d) = 8Nd^2
        Dictionary: D * 2dr + N * (D + 2d*r_delta) = 2Ddr + ND + 2Nd*r_delta

        With D=4, r=d, r_delta=d/4, N=4:
        Standard: 32d^2
        Dictionary: 8d^2 + 16 + 2d^2 = 10d^2 + 16
        Savings: ~69% (but lower effective rank per expert)
    """

    def __init__(self, n_embd: int, n_experts: int = 4, top_k: int = 2,
                 n_dict: int = 8, dict_rank: int = 32, delta_rank: int = 16):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_dict = n_dict

        # Token-level router (same as standard MoE)
        self.router = nn.Linear(n_embd, n_experts, bias=False)

        # Shared dictionary of low-rank sub-modules
        self.dictionary = [DictionaryEntry(n_embd, dict_rank)
                           for _ in range(n_dict)]

        # Experts composed from dictionary + residual
        self.experts = [DictionaryExpert(n_embd, n_dict, delta_rank)
                        for _ in range(n_experts)]

        self._gate_probs = None

    def __call__(self, x):
        """Forward: route tokens to dictionary-composed experts.

        x: (B, T, d) -> (B, T, d)
        """
        # Standard softmax routing
        scores = self.router(x)  # (B, T, N)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k selection
        top_vals = mx.topk(scores, self.top_k, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Weighted expert outputs
        out = mx.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * expert(x, self.dictionary)

        return out

    def balance_loss(self) -> mx.array:
        """Standard MoE load-balancing loss."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_experts * mx.sum(mean_probs * mean_probs)

    def dictionary_utilization(self) -> dict:
        """Measure how much each dictionary entry is used across experts.

        Returns dict with:
            - per_entry_weight: mean alpha weight across experts for each dict entry
            - utilization_rate: fraction of entries with mean weight > 1/(2*D)
            - entropy: Shannon entropy of mean alpha distribution
        """
        # Gather alpha weights from all experts
        all_alphas = mx.stack([e.get_alpha_weights() for e in self.experts])  # (N, D)
        mean_alpha = mx.mean(all_alphas, axis=0)  # (D,)
        mx.eval(mean_alpha)

        # Utilization: entry is "used" if mean alpha > uniform/2
        threshold = 0.5 / self.n_dict
        utilized = mx.sum((mean_alpha > threshold).astype(mx.float32)).item()
        utilization_rate = utilized / self.n_dict

        # Shannon entropy of mean alpha distribution
        eps = 1e-8
        entropy = -mx.sum(mean_alpha * mx.log(mean_alpha + eps)).item()
        import math
        max_entropy = math.log(self.n_dict)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "per_entry_weight": mean_alpha.tolist(),
            "utilization_rate": utilization_rate,
            "normalized_entropy": normalized_entropy,
            "expert_alphas": all_alphas.tolist(),
        }


class DictionaryMoEBlock(nn.Module):
    """Transformer block with dictionary-composed MoE layer."""

    def __init__(self, n_embd: int, n_head: int, n_experts: int = 4,
                 top_k: int = 2, n_dict: int = 8, dict_rank: int = 32,
                 delta_rank: int = 16):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.moe = DictionaryMoELayer(n_embd, n_experts, top_k,
                                       n_dict, dict_rank, delta_rank)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


@register("lz_dictionary_moe", parent="moe")
class DictionaryMoEGPT(nn.Module):
    """GPT with dictionary-composed MoE experts.

    Each expert MLP is a weighted combination of shared dictionary entries
    (low-rank sub-modules) plus a small per-expert residual. The dictionary
    is shared within each layer.

    Default config: d=64, N=4 experts, k=2, D=8 dict entries,
                    dict_rank=32 (d/2), delta_rank=16 (d/4).

    vs MoEGPT baseline:
        MoEGPT: 4 independent experts, each d->4d->d = 8d^2 params per expert
        DictMoE: 8 shared dict entries (d->32->d) + 4 experts with delta (d->16->d)
                 + 4*8 alpha params
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_experts: int = 4, top_k: int = 2,
                 n_dict: int = 8, dict_rank: int = 32, delta_rank: int = 16):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [DictionaryMoEBlock(n_embd, n_head, n_experts, top_k,
                                           n_dict, dict_rank, delta_rank)
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
            total = total + layer.moe.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def dictionary_diagnostics(self) -> dict:
        """Get per-layer dictionary utilization diagnostics."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            diagnostics[f"layer_{li}"] = layer.moe.dictionary_utilization()
        return diagnostics
