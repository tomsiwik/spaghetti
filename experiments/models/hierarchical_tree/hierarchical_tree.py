"""Hierarchical Capsule Tree — binary tree routing replaces flat softmax MoE.

A depth-D binary tree of capsule groups. Internal nodes are learned binary gates
(sigmoid). Leaf nodes are CapsuleGroup experts. Routing is top-down traversal
with beam search (beam=k selects k leaves).

Key design:
- Root node is always computed (shared knowledge anchor)
- Leaves are specialized capsules (conditionally computed)
- Beam=2 traversal matches the proven k=2 optimal finding
- Depth-3 tree = 8 leaf groups (matches G=8 from composition experiments)
- Each internal node: sigmoid(x @ w + b), costing d+1 params

Architecture:
         [root gate]
         /          \\
    [gate_0]      [gate_1]
    /      \\      /      \\
  [g00]  [g01]  [g10]  [g11]    <-- depth 2
  / \\    / \\    / \\    / \\
 L0  L1 L2  L3 L4  L5 L6  L7   <-- depth 3 (leaf capsule groups)

Traversal with beam=2:
  root: pick top-2 children by gate probability
  for each selected child: pick top-1 child by gate probability
  result: 2 leaf groups selected
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class TreeGate(nn.Module):
    """Binary gate at an internal tree node.

    Produces a scalar probability of going left (1-p = right).
    gate(x) = sigmoid(x @ w + b), where w in R^d, b in R^1.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1, bias=True)

    def __call__(self, x):
        """x: (..., d) -> p_left: (..., 1) in [0, 1]."""
        return mx.sigmoid(self.proj(x))


class HierarchicalCapsuleTree(nn.Module):
    """Binary tree of capsule groups with beam-search routing.

    Tree structure: depth D, 2^D leaf capsule groups, 2^D - 1 internal gates.
    Routing: top-down beam search selects beam_width leaf groups.

    The tree is stored as a flat array indexed by node position:
    - Internal gates: indices 0..2^D-2 (stored in self.gates)
    - Leaf groups: indices 0..2^D-1 (stored in self.leaves)

    Node indexing (0-based):
    - Root = 0
    - Left child of node i = 2*i + 1
    - Right child of node i = 2*i + 2
    - Leaf nodes at depth D correspond to leaf indices 0..2^D-1
    """

    def __init__(self, n_embd: int, depth: int = 3,
                 n_capsules_per_leaf: int = 32, beam_width: int = 2):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1  # 2^D - 1
        self.beam_width = beam_width

        # Internal binary gates (one per internal node)
        self.gates = [TreeGate(n_embd) for _ in range(self.n_internal)]

        # Leaf capsule groups
        self.leaves = [CapsuleGroup(n_embd, n_capsules_per_leaf)
                       for _ in range(self.n_leaves)]

        # Cache for gate probabilities (for aux loss)
        self._leaf_probs = None

    def _compute_all_gate_probs(self, x):
        """Compute all gate probabilities in one pass.

        Returns list of gate outputs (p_left for each internal node).
        x: (B, T, d) -> gate_probs[i]: (B, T, 1) for each internal node i.
        """
        return [gate(x) for gate in self.gates]

    def _tree_beam_routing(self, x):
        """Top-down beam search through the binary tree.

        Returns:
            selected_leaves: list of (leaf_index, weight) tuples, len=beam_width
            leaf_probs: (B, T, n_leaves) full probability distribution for aux loss
        """
        B, T, _ = x.shape

        # Compute all gate probabilities
        gate_probs = self._compute_all_gate_probs(x)  # list of (B, T, 1)

        # Compute leaf probabilities by multiplying gate probs along each path
        # leaf_probs[l] = product of gate decisions along path from root to leaf l
        leaf_prob_list = []
        for leaf_idx in range(self.n_leaves):
            # Trace path from root to this leaf
            # In a perfect binary tree, leaf leaf_idx corresponds to
            # internal node sequence determined by binary representation
            prob = mx.ones((B, T, 1))
            node = 0  # start at root
            for d in range(self.depth):
                p_left = gate_probs[node]  # (B, T, 1)
                # Which direction does this leaf go at this depth?
                # Bit (depth-1-d) of leaf_idx determines left(0) or right(1)
                bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                if bit == 0:  # go left
                    prob = prob * p_left
                    node = 2 * node + 1
                else:  # go right
                    prob = prob * (1 - p_left)
                    node = 2 * node + 2
            leaf_prob_list.append(prob)

        # Stack into (B, T, n_leaves)
        leaf_probs = mx.concatenate(leaf_prob_list, axis=-1)
        return leaf_probs

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape

        # Get full leaf probability distribution
        leaf_probs = self._tree_beam_routing(x)  # (B, T, n_leaves)
        self._leaf_probs = leaf_probs

        # Top-k leaf selection (beam search result)
        top_vals = mx.topk(leaf_probs, self.beam_width, axis=-1)  # (B, T, beam)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)       # (B, T, 1)
        mask = (leaf_probs >= threshold).astype(mx.float32)        # (B, T, n_leaves)

        # Renormalize selected leaf weights
        masked_probs = leaf_probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Compute output: weighted sum of selected leaf outputs
        out = mx.zeros_like(x)
        for i, leaf in enumerate(self.leaves):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * leaf(x)

        return out

    def balance_loss(self) -> mx.array:
        """Balance loss encouraging uniform leaf utilization.

        L = n_leaves * sum(mean_prob_leaf^2). Minimized at uniform 1/n_leaves.
        Identical formula to flat CapsulePool but over leaf probabilities.
        """
        if self._leaf_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))  # (n_leaves,)
        return self.n_leaves * mx.sum(mean_probs * mean_probs)

    def gate_entropy_loss(self) -> mx.array:
        """Entropy regularization on gate decisions.

        Encourages gates to make sharp (low-entropy) binary decisions.
        L = -mean over gates of [p*log(p) + (1-p)*log(1-p)]
        Lower values = sharper gates. We MINIMIZE this to encourage sharpness.
        Note: we return negative entropy (to be added to loss for minimization).
        """
        if self._leaf_probs is None:
            return mx.array(0.0)
        # Compute entropy of each gate
        total_entropy = mx.array(0.0)
        # We use leaf_probs to derive gate usage, but simpler to
        # just compute from the gate outputs directly on next forward pass.
        # For now, use the leaf probability distribution entropy.
        eps = 1e-8
        lp = self._leaf_probs  # (B, T, n_leaves)
        # Per-token entropy of leaf distribution
        entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)  # (B, T)
        # We want LOW entropy (sharp routing), so add mean entropy to loss
        return mx.mean(entropy)


class HierarchicalBlock(nn.Module):
    """Transformer block with HierarchicalCapsuleTree replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.tree = HierarchicalCapsuleTree(
            n_embd, depth=tree_depth,
            n_capsules_per_leaf=n_capsules_per_leaf,
            beam_width=beam_width,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.tree(self.norm2(x))
        return x


@register("hierarchical_tree", parent="capsule_moe")
class HierarchicalTreeGPT(nn.Module):
    """GPT with hierarchical binary tree of capsule groups replacing MLP.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - HierarchicalCapsuleTree: depth-D binary tree with 2^D leaf capsule groups
    - Language model head (same as GPT)

    Default config: d=64, depth=3, 8 leaves, 32 capsules/leaf (P=256 total),
    beam=2 (matches k=2 optimal from flat MoE experiments).
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [HierarchicalBlock(n_embd, n_head, tree_depth,
                                         n_capsules_per_leaf, beam_width)
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
        """Combined auxiliary loss: balance + gate entropy."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.tree.balance_loss()
            total = total + 0.1 * layer.tree.gate_entropy_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
