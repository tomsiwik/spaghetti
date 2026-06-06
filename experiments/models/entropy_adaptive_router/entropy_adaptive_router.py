"""Entropy-Adaptive Router — variable-k routing per token.

Hypothesis: Softmax routers allocate equal compute (k=2) to all tokens.
Some tokens are easy (one expert highly confident, low entropy), others
are ambiguous (high entropy). We adapt k per token based on the Shannon
entropy of the routing probability distribution:

  H(p) = -sum_g p_g * log(p_g)

Low H -> k=1 (confident, cheap). High H -> k=2 (uncertain, accurate).

The entropy threshold tau_H splits tokens into "confident" (k=1) and
"uncertain" (k=2) groups. tau_H is calibrated to target an average k.

Connection to arithmetic coding: arithmetic coding allocates variable
precision (bits) per symbol based on symbol probability. Here we allocate
variable compute (number of experts) per token based on routing confidence.

Kill criteria:
  1. Variable-k worse than fixed k=2 at same average compute
  2. Entropy-based k-selection doesn't reduce average k below 1.8
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class EntropyAdaptivePool(nn.Module):
    """CapsulePool with entropy-adaptive per-token k selection.

    For each token:
      1. Compute routing scores s = W_r @ x, probabilities p = softmax(s)
      2. Compute entropy H = -sum(p * log(p))
      3. If H < tau_H: use k=1 (confident). Else: use k=2 (uncertain).
      4. Weight selected groups by renormalized probabilities.

    The threshold tau_H can be:
      - Fixed (set via constructor)
      - Learned (a scalar parameter optimized with the router)
      - Percentile-based (target a fraction of tokens at k=1)

    We implement the learned threshold variant for differentiability.
    """

    def __init__(self, n_embd: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64,
                 tau_h: float = 0.5,
                 learn_threshold: bool = True,
                 min_k: int = 1, max_k: int = 2,
                 sparsity_coeff: float = 0.0):
        super().__init__()
        self.n_groups = n_groups
        self.min_k = min_k
        self.max_k = max_k
        self.sparsity_coeff = sparsity_coeff

        # Router
        self.router = nn.Linear(n_embd, n_groups, bias=False)

        # Capsule groups (same as CapsuleMoEGPT)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        # Entropy threshold -- learned via sigmoid parametrization
        # tau_H = sigmoid(raw_tau) * H_max, where H_max = log(G)
        self.h_max = math.log(n_groups)
        if learn_threshold:
            # Initialize raw_tau so that sigmoid(raw_tau) * H_max ~ tau_h
            # sigmoid(x) = tau_h / H_max => x = log(tau_h / (H_max - tau_h))
            init_val = math.log(max(tau_h, 0.01) / max(self.h_max - tau_h, 0.01))
            self.raw_tau = mx.array([init_val])
        else:
            self.raw_tau = None
            self._fixed_tau = tau_h

        # Statistics for logging
        self._gate_probs = None
        self._entropies = None
        self._k_per_token = None
        self._avg_k = None
        self._alpha = None  # soft k-selection variable

    @property
    def tau_h(self) -> float:
        """Current entropy threshold."""
        if self.raw_tau is not None:
            return (mx.sigmoid(self.raw_tau) * self.h_max).item()
        return self._fixed_tau

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape
        G = self.n_groups

        # Step 1: Routing scores and probabilities
        scores = self.router(x)  # (B, T, G)
        probs = mx.softmax(scores, axis=-1)  # (B, T, G)
        self._gate_probs = probs

        # Step 2: Per-token entropy H = -sum(p * log(p))
        # Add eps to avoid log(0)
        log_probs = mx.log(probs + 1e-10)
        entropies = -mx.sum(probs * log_probs, axis=-1)  # (B, T)
        self._entropies = entropies

        # Step 3: Determine k per token via entropy threshold
        # Use soft threshold for differentiability during training:
        # alpha = sigmoid((H - tau_H) / temperature)
        # effective_k = k_min + alpha * (k_max - k_min)
        # At inference, this becomes a hard threshold.
        if self.raw_tau is not None:
            tau = mx.sigmoid(self.raw_tau) * self.h_max  # scalar
        else:
            tau = mx.array([self._fixed_tau])

        # Soft interpolation: temperature controls sharpness
        temperature = 0.1  # sharp but differentiable
        alpha = mx.sigmoid((entropies - tau) / temperature)  # (B, T), 0=confident, 1=uncertain
        self._alpha = alpha
        # alpha ~ 0 -> k=1 (use only top-1 expert)
        # alpha ~ 1 -> k=2 (use top-2 experts)

        # For logging
        hard_k = mx.where(entropies < tau, 1.0, 2.0)
        self._k_per_token = hard_k
        self._avg_k = mx.mean(hard_k).item()

        # Step 4: Compute outputs with adaptive weighting
        # For k=1 tokens: only top-1 expert contributes
        # For k=2 tokens: top-2 experts contribute
        # We implement this via soft masking:

        # Get top-2 indices and create masks
        top2_vals = mx.topk(scores, min(2, G), axis=-1)  # (B, T, 2)
        thresh_2 = mx.min(top2_vals, axis=-1, keepdims=True)  # (B, T, 1)
        mask_top2 = (scores >= thresh_2).astype(mx.float32)  # (B, T, G)

        # Get top-1 mask
        top1_vals = mx.topk(scores, 1, axis=-1)  # (B, T, 1)
        mask_top1 = (scores >= top1_vals).astype(mx.float32)  # (B, T, G)

        # Interpolate between top-1 and top-2 masks based on alpha
        alpha_expanded = alpha[..., None]  # (B, T, 1)
        mask = mask_top1 * (1.0 - alpha_expanded) + mask_top2 * alpha_expanded  # (B, T, G)

        # Apply mask and renormalize
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Step 5: Weighted sum of group outputs
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)

        return out

    def balance_loss(self) -> mx.array:
        """Standard balance loss (same as CapsulePool)."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        return self.n_groups * mx.sum(mean_probs * mean_probs)

    def sparsity_loss(self) -> mx.array:
        """Compute efficiency loss: penalize using k=2 when k=1 would suffice.

        L_sparse = sparsity_coeff * mean(alpha)
        where alpha is the soft k-selection variable (0=k1, 1=k2).
        Minimizing this encourages the router to be more confident (lower entropy).
        """
        if self._alpha is None or self.sparsity_coeff == 0.0:
            return mx.array(0.0)
        return self.sparsity_coeff * mx.mean(self._alpha)

    def entropy_stats(self) -> dict:
        """Return entropy statistics from last forward pass."""
        if self._entropies is None:
            return {}
        ent = self._entropies
        mx.eval(ent)
        return {
            "mean_entropy": mx.mean(ent).item(),
            "std_entropy": mx.std(ent).item() if ent.size > 1 else 0.0,
            "min_entropy": mx.min(ent).item(),
            "max_entropy": mx.max(ent).item(),
            "tau_h": self.tau_h,
            "avg_k": self._avg_k,
            "frac_k1": mx.mean((self._k_per_token == 1.0).astype(mx.float32)).item()
                        if self._k_per_token is not None else 0.0,
        }


class EntropyAdaptiveBlock(nn.Module):
    """Transformer block with EntropyAdaptivePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 tau_h: float = 0.5, learn_threshold: bool = True,
                 sparsity_coeff: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = EntropyAdaptivePool(
            n_embd, n_groups, n_capsules_per_group,
            tau_h=tau_h, learn_threshold=learn_threshold,
            sparsity_coeff=sparsity_coeff,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("entropy_adaptive_router", parent="capsule_moe")
class EntropyAdaptiveRouterGPT(nn.Module):
    """GPT with entropy-adaptive per-token k routing.

    Extends CapsuleMoEGPT by replacing fixed top-k routing with entropy-based
    adaptive k selection. High-confidence tokens use k=1, low-confidence tokens
    use k=2. The entropy threshold is learned during training.

    Parameter overhead vs capsule_moe: 1 scalar per layer (the threshold).
    Negligible.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 tau_h: float = 0.5, learn_threshold: bool = True,
                 sparsity_coeff: float = 0.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [EntropyAdaptiveBlock(
            n_embd, n_head, n_groups, n_capsules_per_group,
            tau_h=tau_h, learn_threshold=learn_threshold,
            sparsity_coeff=sparsity_coeff,
        ) for _ in range(n_layer)]
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
        """Balance loss + sparsity loss across all layers."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
            total = total + layer.capsule_pool.sparsity_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def entropy_stats(self) -> list[dict]:
        """Per-layer entropy statistics."""
        return [layer.capsule_pool.entropy_stats() for layer in self.layers]

    def avg_k(self) -> float:
        """Average k across all layers and tokens from last forward pass."""
        stats = self.entropy_stats()
        ks = [s.get("avg_k", 2.0) for s in stats if s]
        return sum(ks) / len(ks) if ks else 2.0
