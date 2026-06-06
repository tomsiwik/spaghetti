"""Huffman-shaped Capsule Tree -- activation-frequency-optimal expert structure.

Replaces the balanced binary tree (depth D, all leaves at same depth) with a
Huffman-coded tree where frequently-activated capsule groups are near the root
(fewer routing decisions) and rarely-activated groups are deeper (more decisions).

Minimizes expected routing depth: E[depth] = sum_l freq_l * depth_l, which is
exactly the expected codeword length in Huffman coding.

Two operating modes:
1. HuffmanCapsuleTree: build tree from given frequencies, variable-depth leaves
2. HuffmanTreeGPT: full model that profiles a balanced tree then reshapes to Huffman

Architecture example (8 leaves, non-uniform frequencies):
    If frequencies are [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]:
    Huffman tree (one possible shape):
                    [root]
                   /      \\
              [g0]          L0(0.30)        <-- depth 1 (most frequent)
             /    \\
         [g1]      L1(0.25)                 <-- depth 2
        /    \\
    [g2]      [g3]
    / \\       / \\
   L2  L3   [g4]  L4                       <-- depth 3-4
            / \\
           L5  [g5]
              / \\
             L6  L7                         <-- depth 5 (rarest)

    Balanced tree: all leaves at depth 3, E[depth] = 3.0
    Huffman tree: E[depth] = sum(freq*depth) < 3.0
"""

import heapq
import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


# ── Huffman tree construction ────────────────────────────────────────────────

class HuffmanNode:
    """Node in a Huffman tree (used only for construction, not for forward pass)."""

    def __init__(self, freq: float, leaf_id: int | None = None,
                 left: "HuffmanNode | None" = None,
                 right: "HuffmanNode | None" = None):
        self.freq = freq
        self.leaf_id = leaf_id  # None for internal nodes
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

    def is_leaf(self):
        return self.leaf_id is not None


def build_huffman_tree(frequencies: list[float]) -> HuffmanNode:
    """Build a Huffman tree from leaf activation frequencies.

    Args:
        frequencies: list of N activation frequencies (must sum to ~1.0)

    Returns:
        Root HuffmanNode of the constructed tree
    """
    n = len(frequencies)
    if n == 1:
        return HuffmanNode(freq=frequencies[0], leaf_id=0)

    # Build min-heap of leaf nodes
    heap = [HuffmanNode(freq=f, leaf_id=i) for i, f in enumerate(frequencies)]
    heapq.heapify(heap)

    # Repeatedly merge two lowest-frequency nodes
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)

    return heap[0]


def get_huffman_codes(root: HuffmanNode) -> dict[int, list[int]]:
    """Extract Huffman codes (paths) for each leaf.

    Returns:
        dict mapping leaf_id -> list of bits (0=left, 1=right) from root to leaf
    """
    codes = {}

    def traverse(node, path):
        if node.is_leaf():
            codes[node.leaf_id] = list(path)
            return
        if node.left:
            traverse(node.left, path + [0])
        if node.right:
            traverse(node.right, path + [1])

    traverse(root, [])

    # Handle edge case: single leaf gets code [0]
    if len(codes) == 1 and codes[list(codes.keys())[0]] == []:
        codes[list(codes.keys())[0]] = [0]

    return codes


def huffman_expected_depth(frequencies: list[float], codes: dict[int, list[int]]) -> float:
    """Compute expected routing depth: E[depth] = sum_i freq_i * depth_i."""
    total = 0.0
    for i, freq in enumerate(frequencies):
        total += freq * len(codes[i])
    return total


def count_internal_nodes(root: HuffmanNode) -> int:
    """Count internal (non-leaf) nodes in the tree."""
    if root.is_leaf():
        return 0
    count = 1
    if root.left:
        count += count_internal_nodes(root.left)
    if root.right:
        count += count_internal_nodes(root.right)
    return count


def max_depth(root: HuffmanNode) -> int:
    """Maximum depth of the tree."""
    if root.is_leaf():
        return 0
    d_left = max_depth(root.left) if root.left else 0
    d_right = max_depth(root.right) if root.right else 0
    return 1 + max(d_left, d_right)


# ── Huffman-shaped Capsule Tree module ────────────────────────────────────────

class HuffmanGate(nn.Module):
    """Binary gate at an internal Huffman tree node.

    Identical to TreeGate but with a unique node_id for tracking.
    gate(x) = sigmoid(w^T x + b).
    """

    def __init__(self, n_embd: int, node_id: int = 0):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1, bias=True)
        self.node_id = node_id

    def __call__(self, x):
        return mx.sigmoid(self.proj(x))


class HuffmanCapsuleTree(nn.Module):
    """Variable-depth binary tree of capsule groups shaped by Huffman coding.

    Unlike the balanced HierarchicalCapsuleTree where all leaves are at the same
    depth D, this tree has variable-depth leaves based on activation frequencies.
    Frequently-activated leaves are near the root (few gate decisions), while
    rarely-activated leaves are deeper (more gate decisions).

    The tree structure is defined by Huffman codes: each leaf has a binary code
    (path from root), and internal gates are placed at each branching point.

    For forward pass, we compute leaf probabilities by multiplying gate decisions
    along each leaf's path, then select top-k leaves (same beam search as balanced).
    """

    def __init__(self, n_embd: int, n_leaves: int = 8,
                 n_capsules_per_leaf: int = 32, beam_width: int = 2,
                 frequencies: list[float] | None = None):
        super().__init__()
        self.n_leaves = n_leaves
        self.beam_width = beam_width
        self.n_embd = n_embd

        # Build Huffman tree from frequencies (or use uniform -> balanced)
        if frequencies is None:
            frequencies = [1.0 / n_leaves] * n_leaves
        assert len(frequencies) == n_leaves, \
            f"Expected {n_leaves} frequencies, got {len(frequencies)}"

        # Normalize frequencies
        total = sum(frequencies)
        self.frequencies = [f / total for f in frequencies]

        # Build Huffman tree and extract codes
        huff_root = build_huffman_tree(self.frequencies)
        self.codes = get_huffman_codes(huff_root)
        self.n_internal = count_internal_nodes(huff_root)
        self._max_depth = max_depth(huff_root)

        # Expected depth metrics
        self.balanced_depth = _balanced_depth_for_n(n_leaves)
        self.expected_depth = huffman_expected_depth(self.frequencies, self.codes)

        # Create gates: one per internal node
        # We need to map internal nodes to gate indices. Use BFS ordering.
        self._gate_map = {}  # maps (code_prefix_tuple) -> gate_index
        self._build_gate_map(huff_root)
        self.gates = [HuffmanGate(n_embd, node_id=i)
                      for i in range(self.n_internal)]

        # Leaf capsule groups
        self.leaves = [CapsuleGroup(n_embd, n_capsules_per_leaf)
                       for _ in range(n_leaves)]

        # Cache for aux loss
        self._leaf_probs = None

    def _build_gate_map(self, root: HuffmanNode, prefix: tuple = ()):
        """Map each internal node (identified by its path prefix) to a gate index."""
        if root.is_leaf():
            return
        gate_idx = len(self._gate_map)
        self._gate_map[prefix] = gate_idx
        if root.left:
            self._build_gate_map(root.left, prefix + (0,))
        if root.right:
            self._build_gate_map(root.right, prefix + (1,))

    def _compute_leaf_probs(self, x):
        """Compute leaf probabilities by following each leaf's Huffman code path.

        For each leaf, multiply gate decisions along its code path.
        Gate decision: if code bit = 0 (left), use p_left; if 1 (right), use 1-p_left.

        Returns: (B, T, n_leaves) probability distribution over leaves
        """
        B, T, _ = x.shape

        # Pre-compute all gate outputs
        gate_outputs = [gate(x) for gate in self.gates]  # list of (B, T, 1)

        leaf_prob_list = []
        for leaf_id in range(self.n_leaves):
            code = self.codes[leaf_id]
            prob = mx.ones((B, T, 1))

            # Walk down the code path, multiplying gate probabilities
            prefix = ()
            for bit in code:
                gate_idx = self._gate_map[prefix]
                p_left = gate_outputs[gate_idx]  # (B, T, 1)
                if bit == 0:
                    prob = prob * p_left
                    prefix = prefix + (0,)
                else:
                    prob = prob * (1 - p_left)
                    prefix = prefix + (1,)

            leaf_prob_list.append(prob)

        # Stack into (B, T, n_leaves)
        leaf_probs = mx.concatenate(leaf_prob_list, axis=-1)
        return leaf_probs

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape

        # Get full leaf probability distribution
        leaf_probs = self._compute_leaf_probs(x)  # (B, T, n_leaves)
        self._leaf_probs = leaf_probs

        # Top-k leaf selection (beam search result)
        top_vals = mx.topk(leaf_probs, self.beam_width, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (leaf_probs >= threshold).astype(mx.float32)

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
        """Balance loss encouraging utilization proportional to target frequencies.

        For Huffman trees, we do NOT want uniform utilization -- we want utilization
        proportional to the Huffman frequencies. So we measure KL divergence between
        actual utilization and target frequencies.

        L = sum_l f_actual_l * log(f_actual_l / f_target_l)
        """
        if self._leaf_probs is None:
            return mx.array(0.0)

        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))  # (n_leaves,)
        target = mx.array(self.frequencies)

        # KL divergence: actual || target
        eps = 1e-8
        kl = mx.sum(mean_probs * mx.log((mean_probs + eps) / (target + eps)))
        return self.n_leaves * kl

    def gate_entropy_loss(self) -> mx.array:
        """Entropy regularization encouraging sharp gate decisions."""
        if self._leaf_probs is None:
            return mx.array(0.0)
        eps = 1e-8
        lp = self._leaf_probs
        entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)
        return mx.mean(entropy)

    def avg_routing_depth(self) -> float:
        """Compute average routing depth using current leaf probabilities.

        This is the key metric: E[depth] = sum_l P(leaf=l) * depth(l).
        For a balanced tree of 8 leaves, this is always 3.0.
        For a Huffman tree, this should be < 3.0 when frequencies are non-uniform.
        """
        if self._leaf_probs is None:
            return -1.0
        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))  # (n_leaves,)
        depths = mx.array([float(len(self.codes[i])) for i in range(self.n_leaves)])
        return (mx.sum(mean_probs * depths)).item()


def _balanced_depth_for_n(n: int) -> float:
    """Depth of a balanced binary tree with n leaves."""
    import math
    return math.ceil(math.log2(n)) if n > 1 else 1


# ── Transformer block and full model ─────────────────────────────────────────

class HuffmanBlock(nn.Module):
    """Transformer block with HuffmanCapsuleTree replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_leaves: int = 8, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2, frequencies: list[float] | None = None):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.tree = HuffmanCapsuleTree(
            n_embd, n_leaves=n_leaves,
            n_capsules_per_leaf=n_capsules_per_leaf,
            beam_width=beam_width,
            frequencies=frequencies,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.tree(self.norm2(x))
        return x


@register("huffman_tree", parent="hierarchical_tree")
class HuffmanTreeGPT(nn.Module):
    """GPT with Huffman-shaped binary tree of capsule groups.

    The tree shape is determined by leaf activation frequencies. When frequencies
    are uniform, this degenerates to a balanced binary tree (equivalent to
    HierarchicalTreeGPT). When frequencies are non-uniform, frequent leaves get
    shorter paths and the expected routing depth decreases.

    Default config: d=64, 8 leaves, 32 capsules/leaf (P=256 total), beam=2.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_leaves: int = 8, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2,
                 frequencies: list[float] | None = None):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [HuffmanBlock(n_embd, n_head, n_leaves,
                                     n_capsules_per_leaf, beam_width,
                                     frequencies)
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

    def avg_routing_depth(self) -> float:
        """Average routing depth across all layers."""
        depths = [layer.tree.avg_routing_depth() for layer in self.layers]
        valid = [d for d in depths if d >= 0]
        return sum(valid) / len(valid) if valid else -1.0
