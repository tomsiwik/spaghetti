"""LSH Capsule Routing — locality-sensitive hashing replaces learned softmax router.

Random-projection LSH maps each token to expert buckets in O(T*d) time,
independent of the number of experts N. Multiple hash tables (T tables)
provide top-k selection: each table votes for one expert, top-k by vote
count (or by accumulated hash-score) are selected.

Key design:
- Hash functions are FIXED random Gaussian projections (no learned params)
- Each hash table: h_t(x) = argmax over buckets of (R_t @ x)
  where R_t in R^{n_buckets x d} is a fixed random matrix
- With T hash tables, each token gets T votes for experts
- Top-k experts by vote count are selected (ties broken by accumulated score)
- No routing parameters to train -- zero calibration cost

Architecture:
  Token x in R^d
    |
    v
  [Hash Table 1: R_1 @ x -> bucket b_1 -> expert e_1]
  [Hash Table 2: R_2 @ x -> bucket b_2 -> expert e_2]
  ...
  [Hash Table T: R_T @ x -> bucket b_T -> expert e_T]
    |
    v
  Top-k experts by vote count
    |
    v
  Weighted sum of selected expert outputs

Prior art:
- Hash Layers (NeurIPS 2021): competitive with Switch Transformer
- PEER (DeepMind 2024): 1M+ experts via product-key retrieval
- SimHash / random-projection LSH (Charikar 2002)
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class LSHRouter(nn.Module):
    """Locality-sensitive hash router using random projections.

    Each hash table is a fixed random Gaussian matrix R_t in R^{n_groups x d}.
    The hash function: h_t(x) = argmax(R_t @ x), mapping each token to one of
    n_groups buckets.

    With T tables, each token gets T expert votes. Top-k experts by vote count
    are selected. When votes tie, accumulated projection scores break ties.

    Parameters:
        n_embd: embedding dimension d
        n_groups: number of expert groups (buckets per hash table)
        n_tables: number of hash tables T
        top_k: number of experts to select
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_tables: int = 4, top_k: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.n_groups = n_groups
        self.n_tables = n_tables
        self.top_k = top_k

        # Fixed random projection matrices (NOT trainable)
        # Each R_t in R^{n_groups x d}, initialized from N(0, 1/sqrt(d))
        # We store them as a single tensor (n_tables, n_groups, n_embd)
        scale = 1.0 / math.sqrt(n_embd)
        self._projections = mx.random.normal((n_tables, n_groups, n_embd)) * scale

        # Cache for diagnostics
        self._gate_probs = None

    def __call__(self, x):
        """Compute routing weights via LSH.

        x: (B, T, d) -> weights: (B, T, n_groups), sparse with top_k nonzero
        """
        B, T, d = x.shape

        # Compute projection scores for all tables at once
        # x: (B, T, d), projections: (n_tables, n_groups, d)
        # Result: (B, T, n_tables, n_groups)
        # Use matmul: x @ projections^T -> (B, T, 1, d) @ (n_tables, d, n_groups)
        # Reshape projections to (n_tables, d, n_groups) and use broadcast matmul
        proj_t = mx.transpose(self._projections, axes=(0, 2, 1))  # (n_tables, d, n_groups)
        # x: (B, T, d) -> (B, T, 1, d)
        # proj_t: (n_tables, d, n_groups) -> (1, 1, n_tables*d... no, just loop)
        # Simplest correct approach: loop over tables
        score_list = []
        for t in range(self.n_tables):
            # self._projections[t]: (n_groups, d)
            # x @ proj[t]^T: (B, T, d) @ (d, n_groups) -> (B, T, n_groups)
            s = x @ self._projections[t].T
            score_list.append(s)
        # Stack: (B, T, n_tables, n_groups)
        scores = mx.stack(score_list, axis=2)

        # For each table, find the winning bucket (argmax)
        # winners: (B, T, n_tables) - index of winning group per table
        # But for vote counting, we need to accumulate votes across tables

        # Approach: compute vote counts by summing one-hot winners across tables
        # Then select top-k by vote count, break ties with total score

        # One-hot encode winners per table
        # scores: (B, T, n_tables, n_groups)
        # For each table, the winner gets score 1
        table_max = mx.max(scores, axis=-1, keepdims=True)  # (B, T, n_tables, 1)
        is_winner = (scores >= table_max).astype(mx.float32)  # (B, T, n_tables, n_groups)

        # Vote count per group: sum across tables
        votes = mx.sum(is_winner, axis=2)  # (B, T, n_groups)

        # Accumulated score per group: sum of projection scores across tables
        # (used for tie-breaking and as routing weight)
        total_score = mx.sum(scores, axis=2)  # (B, T, n_groups)

        # Combined ranking: votes * large_constant + normalized_score
        # This ensures vote count dominates, with scores breaking ties
        combined = votes * 1000.0 + total_score

        # Top-k selection
        top_vals = mx.topk(combined, self.top_k, axis=-1)  # (B, T, k)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)  # (B, T, 1)
        mask = (combined >= threshold).astype(mx.float32)  # (B, T, n_groups)

        # Routing weights: use softmax over total_score for selected experts
        # (LSH selects WHICH experts, but we need smooth weights for combination)
        # Using votes as weights would give uniform 1/k; using scores gives
        # data-dependent weights without learned parameters.
        masked_scores = total_score * mask + (1 - mask) * (-1e9)
        weights = mx.softmax(masked_scores, axis=-1)  # (B, T, n_groups)
        weights = weights * mask  # zero out non-selected

        self._gate_probs = weights
        return weights


class LSHCapsulePool(nn.Module):
    """Pool of capsule groups with LSH-based routing.

    Replaces the learned softmax router in CapsulePool with fixed
    random-projection LSH. The capsule groups themselves are still
    learned (same as CapsulePool).
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32,
                 n_tables: int = 4, top_k: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.n_tables = n_tables
        self.top_k = top_k

        # LSH router (no learned parameters)
        self.router = LSHRouter(n_embd, n_groups, n_tables, top_k)

        # Capsule groups (learned, same as CapsulePool)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        # Get routing weights from LSH
        weights = self.router(x)  # (B, T, n_groups)
        self._gate_probs = weights

        # Compute weighted expert outputs
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = weights[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)

        return out

    def balance_loss(self) -> mx.array:
        """Balance loss: same formula as CapsulePool.

        L = G * sum(mean_prob_g^2). Minimized at uniform 1/G.
        Even though routing is not learned, balance loss can be used
        as a diagnostic or to tune projection matrices.
        """
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        return self.n_groups * mx.sum(mean_probs * mean_probs)


class LSHBlock(nn.Module):
    """Transformer block with LSH-routed CapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 n_tables: int = 4, top_k: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = LSHCapsulePool(
            n_embd, n_groups, n_capsules_per_group, n_tables, top_k
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("lsh_capsule_routing", parent="capsule_moe")
class LSHCapsuleRoutingGPT(nn.Module):
    """GPT with LSH-routed capsule groups replacing learned softmax routing.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - LSHCapsulePool: random-projection LSH selects top-k groups
    - Language model head (same as GPT)

    Key differences from CapsuleMoEGPT:
    - Router is FIXED random projections (no learned routing params)
    - Routing cost is O(T*d) independent of N (vs O(N*d) for softmax)
    - No routing parameters to train or calibrate
    - Multiple hash tables (T) provide robustness and top-k selection

    Default config: d=64, G=8, 32 caps/group, T=4 tables, k=2.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 n_tables: int = 4, top_k: int = 2):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [LSHBlock(n_embd, n_head, n_groups,
                                n_capsules_per_group, n_tables, top_k)
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
        """Balance loss for diagnostics (routing is not learned)."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def get_routing_diagnostics(self) -> dict:
        """Return diagnostic info about LSH routing behavior."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            pool = layer.capsule_pool
            if pool._gate_probs is not None:
                gp = pool._gate_probs  # (B, T, G)
                # Which experts are selected (nonzero weight)
                selected = (gp > 0).astype(mx.float32)
                # Mean experts selected per token
                mean_selected = mx.mean(mx.sum(selected, axis=-1)).item()
                # Expert utilization (how often each expert is selected)
                util = mx.mean(selected, axis=(0, 1))
                mx.eval(util)
                # Routing entropy
                eps = 1e-8
                entropy = -mx.sum(gp * mx.log(gp + eps), axis=-1)
                max_entropy = math.log(pool.n_groups)
                norm_entropy = mx.mean(entropy).item() / max_entropy if max_entropy > 0 else 0

                diagnostics[f"layer_{li}"] = {
                    "mean_selected": mean_selected,
                    "expert_utilization": util.tolist(),
                    "normalized_entropy": norm_entropy,
                    "n_tables": pool.n_tables,
                }
            else:
                diagnostics[f"layer_{li}"] = {"status": "no forward pass yet"}
        return diagnostics
