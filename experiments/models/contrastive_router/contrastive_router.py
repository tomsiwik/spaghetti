"""Contrastive Router — InfoNCE-trained routing keys for capsule MoE composition."""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class ContrastiveCapsulePool(nn.Module):
    """Capsule pool with contrastive key routing: s_i = ||x @ K_i||^2.

    Each group i has a routing key K_i in R^{d x d_key}. The routing score
    is the squared L2 norm of the projection: s_i = ||x @ K_i||^2.
    Top-k selection and weighting identical to CapsulePool.
    """

    def __init__(self, n_embd: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64, top_k_groups: int = 2,
                 d_key: int = 8):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups
        self.d_key = d_key
        self.uniform_routing = False

        # Contrastive routing keys: K_i in R^{d x d_key} per group
        self.routing_keys = [nn.Linear(n_embd, d_key, bias=False)
                             for _ in range(n_groups)]
        # Capsule groups (same as CapsulePool)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]
        self._gate_probs = None

    def routing_scores(self, x):
        """s_i = ||x @ K_i||^2. x: (..., d) -> (..., N)"""
        return mx.stack([mx.sum(k(x) * k(x), axis=-1)
                         for k in self.routing_keys], axis=-1)

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        if self.uniform_routing:
            w = 1.0 / self.n_groups
            out = mx.zeros_like(x)
            for group in self.groups:
                out = out + w * group(x)
            self._gate_probs = mx.full((*x.shape[:-1], self.n_groups), w)
            return out

        scores = self.routing_scores(x)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (
            mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)
        return out

    def balance_loss(self):
        """L = G * sum(mean_prob_g^2). Minimized at uniform 1/G."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_groups * mx.sum(mean_probs * mean_probs)


class ContrastiveBlock(nn.Module):
    """Transformer block with ContrastiveCapsulePool."""

    def __init__(self, n_embd: int, n_head: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64, top_k_groups: int = 2,
                 d_key: int = 8):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = ContrastiveCapsulePool(
            n_embd, n_groups, n_capsules_per_group, top_k_groups, d_key)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("contrastive_router", parent="capsule_moe")
class ContrastiveRouterGPT(nn.Module):
    """Capsule MoE with contrastive routing keys (K_i per group).

    Each group has a routing key K_i in R^{d x d_key}. Routing score:
    s_i = ||x @ K_i||^2. Keys are trained with InfoNCE loss on labeled
    hidden states for domain discrimination.

    Total params = capsule_moe params + n_layer * N * d * d_key routing keys.
    Default: 203,136 + 4 * 8 * 64 * 8 = 219,520 total.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, d_key: int = 8):
        super().__init__()
        self.d_key = d_key
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [ContrastiveBlock(n_embd, n_head, n_groups,
                                        n_capsules_per_group, top_k_groups,
                                        d_key)
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

    def aux_loss(self):
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass


# --- Calibration utilities ---

def extract_hidden_states(model, tokens):
    """Extract per-layer hidden states (input to capsule pool).

    Works with CapsuleMoEGPT and ContrastiveRouterGPT since both share
    the same block structure (norm1 -> attn -> norm2 -> capsule_pool).

    Returns: list of (B, T, d) arrays, one per layer.
    """
    B, T = tokens.shape
    pos = mx.arange(T)
    x = model.wte(tokens) + model.wpe(pos)
    x = model.norm0(x)
    hiddens = []
    for layer in model.layers:
        x = x + layer.attn(layer.norm1(x))
        h = layer.norm2(x)
        hiddens.append(h)
        x = x + layer.capsule_pool(h)
    return hiddens


def infonce_loss(pool, h, labels, groups_per_domain, tau=0.1):
    """InfoNCE loss for one layer's contrastive routing keys.

    pool: ContrastiveCapsulePool (differentiable through pool.routing_keys)
    h: (B, d) flattened token hidden states
    labels: (B,) integer domain labels (0-indexed)
    groups_per_domain: G (groups belonging to each domain)
    tau: temperature scalar
    """
    scores = pool.routing_scores(h)  # (B, N)
    n_domains = pool.n_groups // groups_per_domain

    # Domain-level scores: max over groups within each domain
    ds = []
    for d in range(n_domains):
        s = d * groups_per_domain
        ds.append(mx.max(scores[:, s:s + groups_per_domain], axis=-1))
    domain_scores = mx.stack(ds, axis=-1) / tau  # (B, D)

    return nn.losses.cross_entropy(domain_scores, labels, reduction="mean")


def routing_accuracy(pool, h, labels, groups_per_domain):
    """Fraction of tokens routed to the correct domain."""
    scores = pool.routing_scores(h)  # (B, N)
    n_domains = pool.n_groups // groups_per_domain

    ds = []
    for d in range(n_domains):
        s = d * groups_per_domain
        ds.append(mx.max(scores[:, s:s + groups_per_domain], axis=-1))
    preds = mx.argmax(mx.stack(ds, axis=-1), axis=-1)
    return (mx.sum(preds == labels) / labels.size).item()
