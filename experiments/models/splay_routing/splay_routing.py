"""Splay-Tree Adaptive Routing -- self-adjusting expert structure for MoE.

After each routing decision, the selected expert's path toward the root is
"splayed" -- its gate biases are adjusted to make it cheaper to reach next time.
This is analogous to splay tree rotations but implemented as soft bias updates
on the existing binary tree gates, preserving learned weights.

Key mechanism:
- Maintain exponential moving average (EMA) of per-leaf access frequencies
- Convert frequencies to gate bias corrections: frequently-used leaves get
  positive bias along their path, making them more likely to be selected
- Use a temperature-scaled decay so that when distribution shifts, old
  frequencies decay and new patterns take over automatically
- The bias correction is ADDITIVE and NON-PARAMETRIC (not learned) -- it's
  a runtime optimization on top of the learned tree

This gives the splay tree's key property (working-set optimality) without
actually restructuring the tree topology, which would destroy learned weights.

Kill criteria:
1. Splay restructuring does NOT reduce routing cost on non-stationary data
2. Splay overhead (restructuring cost) exceeds routing savings
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class SplayTreeGate(nn.Module):
    """Binary gate with splay bias correction.

    gate(x) = sigmoid(x @ w + b + splay_bias)

    The splay_bias is a non-parametric correction derived from leaf access
    frequencies. It is NOT learned by gradient descent -- it is updated at
    runtime based on which leaves are selected.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1, bias=True)
        # Splay bias: positive = favor left child, negative = favor right
        # This is a buffer, not a parameter (no gradient)
        self._splay_bias = 0.0

    def __call__(self, x):
        """x: (..., d) -> p_left: (..., 1) in [0, 1]."""
        logit = self.proj(x) + self._splay_bias
        return mx.sigmoid(logit)


class SplayCapsuleTree(nn.Module):
    """Binary tree of capsule groups with splay-adaptive routing.

    Extends HierarchicalCapsuleTree with runtime splay bias adjustment.
    After each forward pass, the selected leaves' paths get bias corrections
    that make them cheaper to reach in subsequent calls.

    Splay mechanism:
    1. Track per-leaf selection frequency via EMA: f_l = decay * f_l + (1-decay) * selected_l
    2. For each internal gate i, compute the total frequency of leaves in its
       left vs right subtree: f_left_i = sum(f_l for l in left_subtree(i))
    3. Set splay_bias_i = alpha * log(f_left_i / f_right_i) (log-odds of subtree frequency)
    4. This shifts the gate toward the more frequently used subtree

    The log-odds formulation is natural because sigmoid(logit + log(p/q)) = sigmoid(logit) * p/q
    (approximately, for small corrections), so we're multiplicatively boosting the
    probability of the more active subtree.
    """

    def __init__(self, n_embd: int, depth: int = 3,
                 n_capsules_per_leaf: int = 32, beam_width: int = 2,
                 splay_alpha: float = 1.0, splay_decay: float = 0.95):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.beam_width = beam_width
        self.splay_alpha = splay_alpha
        self.splay_decay = splay_decay

        # Internal binary gates (one per internal node)
        self.gates = [SplayTreeGate(n_embd) for _ in range(self.n_internal)]

        # Leaf capsule groups
        self.leaves = [CapsuleGroup(n_embd, n_capsules_per_leaf)
                       for _ in range(self.n_leaves)]

        # Splay state: per-leaf EMA frequency (not a parameter)
        self._leaf_freq = [1.0 / self.n_leaves] * self.n_leaves

        # Cache for auxiliary losses
        self._leaf_probs = None

        # Precompute subtree membership for each internal node
        # _subtree_left[i] = list of leaf indices in left subtree of node i
        # _subtree_right[i] = list of leaf indices in right subtree of node i
        self._subtree_left = {}
        self._subtree_right = {}
        self._precompute_subtrees()

    def _precompute_subtrees(self):
        """Precompute which leaves belong to left/right subtree of each gate."""
        for node_idx in range(self.n_internal):
            left_leaves = []
            right_leaves = []
            for leaf_idx in range(self.n_leaves):
                # Trace path: determine if this leaf goes left or right at node_idx
                node = 0
                for d in range(self.depth):
                    if node == node_idx:
                        bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                        if bit == 0:
                            left_leaves.append(leaf_idx)
                        else:
                            right_leaves.append(leaf_idx)
                        break
                    bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                    if bit == 0:
                        node = 2 * node + 1
                    else:
                        node = 2 * node + 2
            self._subtree_left[node_idx] = left_leaves
            self._subtree_right[node_idx] = right_leaves

    def _update_splay_biases(self):
        """Update gate splay biases based on current leaf frequencies.

        For each gate, compute log-odds of left vs right subtree frequency
        and set as additive bias correction.
        """
        eps = 1e-8
        for node_idx in range(self.n_internal):
            left_freq = sum(self._leaf_freq[l] for l in self._subtree_left[node_idx])
            right_freq = sum(self._leaf_freq[l] for l in self._subtree_right[node_idx])
            # Log-odds: positive = favor left, negative = favor right
            log_odds = float(mx.log(mx.array(left_freq + eps) / mx.array(right_freq + eps)).item())
            self.gates[node_idx]._splay_bias = self.splay_alpha * log_odds

    def _update_leaf_frequencies(self, leaf_probs):
        """Update EMA leaf frequencies based on actual routing decisions.

        leaf_probs: (B, T, n_leaves) -- probabilities from current batch
        """
        # Average selection frequency across batch and time
        mean_probs = mx.mean(leaf_probs, axis=(0, 1))  # (n_leaves,)
        mean_probs_list = mean_probs.tolist()

        # EMA update
        for l in range(self.n_leaves):
            self._leaf_freq[l] = (self.splay_decay * self._leaf_freq[l] +
                                  (1 - self.splay_decay) * mean_probs_list[l])

        # Renormalize to sum to 1
        total = sum(self._leaf_freq)
        if total > 0:
            self._leaf_freq = [f / total for f in self._leaf_freq]

    def _compute_all_gate_probs(self, x):
        """Compute all gate probabilities (with splay bias applied)."""
        return [gate(x) for gate in self.gates]

    def _tree_beam_routing(self, x):
        """Top-down beam search through the splay-adjusted binary tree."""
        B, T, _ = x.shape

        gate_probs = self._compute_all_gate_probs(x)

        leaf_prob_list = []
        for leaf_idx in range(self.n_leaves):
            prob = mx.ones((B, T, 1))
            node = 0
            for d in range(self.depth):
                p_left = gate_probs[node]
                bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                if bit == 0:
                    prob = prob * p_left
                    node = 2 * node + 1
                else:
                    prob = prob * (1 - p_left)
                    node = 2 * node + 2
            leaf_prob_list.append(prob)

        leaf_probs = mx.concatenate(leaf_prob_list, axis=-1)
        return leaf_probs

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape

        # Get leaf probabilities (with splay bias applied in gates)
        leaf_probs = self._tree_beam_routing(x)
        self._leaf_probs = leaf_probs

        # Top-k selection
        top_vals = mx.topk(leaf_probs, self.beam_width, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (leaf_probs >= threshold).astype(mx.float32)

        masked_probs = leaf_probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Compute output
        out = mx.zeros_like(x)
        for i, leaf in enumerate(self.leaves):
            w = masked_probs[..., i:i+1]
            out = out + w * leaf(x)

        # Update splay state (after forward pass, does not affect gradients)
        self._update_leaf_frequencies(leaf_probs)
        self._update_splay_biases()

        return out

    def reset_splay(self):
        """Reset splay state to uniform -- used for controlled experiments."""
        self._leaf_freq = [1.0 / self.n_leaves] * self.n_leaves
        for gate in self.gates:
            gate._splay_bias = 0.0

    def get_splay_state(self) -> dict:
        """Return current splay state for diagnostics."""
        return {
            "leaf_freq": list(self._leaf_freq),
            "gate_biases": [g._splay_bias for g in self.gates],
        }

    def balance_loss(self) -> mx.array:
        """Balance loss (same as hierarchical tree)."""
        if self._leaf_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))
        return self.n_leaves * mx.sum(mean_probs * mean_probs)

    def gate_entropy_loss(self) -> mx.array:
        """Gate entropy loss (same as hierarchical tree)."""
        if self._leaf_probs is None:
            return mx.array(0.0)
        eps = 1e-8
        lp = self._leaf_probs
        entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)
        return mx.mean(entropy)


class SplayBlock(nn.Module):
    """Transformer block with SplayCapsuleTree replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2, splay_alpha: float = 1.0,
                 splay_decay: float = 0.95):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.tree = SplayCapsuleTree(
            n_embd, depth=tree_depth,
            n_capsules_per_leaf=n_capsules_per_leaf,
            beam_width=beam_width,
            splay_alpha=splay_alpha,
            splay_decay=splay_decay,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.tree(self.norm2(x))
        return x


@register("splay_routing", parent="hierarchical_tree")
class SplayTreeGPT(nn.Module):
    """GPT with splay-adaptive binary tree routing.

    Extends HierarchicalTreeGPT with runtime splay bias adjustment.
    The tree structure is identical (depth-D binary tree with 2^D leaf
    capsule groups), but gates receive additive bias corrections based
    on leaf access frequency EMA. This implements the splay tree's
    "move to root" property without topology changes.

    Key difference from parent:
    - Gates have _splay_bias field updated at runtime
    - on_domain_switch resets splay state (simulating distribution shift)
    - Splay biases are NOT learned by gradient descent
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2, splay_alpha: float = 1.0,
                 splay_decay: float = 0.95):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [SplayBlock(n_embd, n_head, tree_depth,
                                   n_capsules_per_leaf, beam_width,
                                   splay_alpha, splay_decay)
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
        """Reset splay biases on domain switch -- allows fast adaptation."""
        for layer in self.layers:
            layer.tree.reset_splay()

    def reset_all_splay(self):
        """Reset all splay state to uniform."""
        for layer in self.layers:
            layer.tree.reset_splay()

    def get_splay_diagnostics(self) -> list[dict]:
        """Return splay state for all layers."""
        return [layer.tree.get_splay_state() for layer in self.layers]
