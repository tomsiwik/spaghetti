"""KD-Tree Routing — axis-aligned feature-space partitioning for expert routing.

Each internal node learns:
  1. A projection vector v in R^d (normalized to unit norm) — the split dimension
  2. A threshold tau in R — the split point
  3. A temperature T > 0 — controls soft/hard boundary

The routing decision at node i:
  p_left(x) = sigmoid(T * (v_i^T x - tau_i))

This is structurally similar to the hierarchical_tree's sigmoid gates but with a
key difference: the split is along a SINGLE learned direction (the projection vector
is constrained to unit norm, making it an axis in a rotated coordinate system).

During training: temperature starts low (soft splits, gradients flow everywhere)
and anneals upward (sharper splits, approaching hard partitioning).

During inference: T -> infinity gives hard binary splits (each token falls in
exactly one leaf polytope, zero routing ambiguity).

Key properties:
- Non-overlapping leaf territories by construction (binary space partition)
- Forced sharp routing (temperature annealing, not learned)
- Each leaf owns a convex polytope in embedding space
- Routing cost: O(depth * d) for projection + comparison
- Compatible with beam search for top-k selection

Architecture:
         [split_0: v0^T x > tau0]
         /                        \\
    [split_1]                  [split_2]
    /        \\                /         \\
  [s3]     [s4]           [s5]       [s6]
  / \\      / \\            / \\         / \\
 L0  L1  L2  L3         L4  L5     L6  L7
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class KDSplitNode(nn.Module):
    """KD-tree split node: learns a projection direction + threshold.

    Split decision: p_left(x) = sigmoid(temperature * (v^T x - tau))

    The projection vector v is learned freely (not constrained to unit norm
    during training -- the temperature handles the sharpness). At inference
    time, the hard split is sign(v^T x - tau), which is norm-invariant.

    Parameters:
        proj: Linear(d, 1, bias=False) -- the projection direction v
        threshold: scalar -- the split point tau
    """

    def __init__(self, n_embd: int):
        super().__init__()
        # Projection direction: d -> 1 (learned hyperplane normal)
        self.proj = nn.Linear(n_embd, 1, bias=False)
        # Threshold (split point along the projection direction)
        self.threshold = mx.zeros((1,))

    def __call__(self, x, temperature: float = 1.0):
        """Compute soft split probability.

        x: (..., d) -> p_left: (..., 1) in [0, 1]
        temperature: higher = sharper splits
        """
        # Project input onto learned direction
        projected = self.proj(x)  # (..., 1)
        # Soft binary decision with temperature
        return mx.sigmoid(temperature * (projected - self.threshold))


class KDTreeCapsuleRouter(nn.Module):
    """KD-tree of capsule groups with temperature-annealed routing.

    Tree structure: depth D, 2^D leaf capsule groups, 2^D - 1 split nodes.
    Each split node partitions along a learned projection direction.

    Training regime:
    - Start with temperature=1.0 (soft, smooth gradients)
    - Anneal to temperature=10.0+ (sharp, approaching hard partitioning)
    - At inference: hard splits (argmax, no softmax bleeding)
    """

    def __init__(self, n_embd: int, depth: int = 3,
                 n_capsules_per_leaf: int = 32, beam_width: int = 2,
                 init_temperature: float = 1.0, max_temperature: float = 10.0):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.beam_width = beam_width
        self.temperature = init_temperature
        self.max_temperature = max_temperature

        # KD-tree split nodes (one per internal node)
        self.splits = [KDSplitNode(n_embd) for _ in range(self.n_internal)]

        # Leaf capsule groups
        self.leaves = [CapsuleGroup(n_embd, n_capsules_per_leaf)
                       for _ in range(self.n_leaves)]

        # Cache for diagnostics
        self._leaf_probs = None
        self._split_sharpness = None

    def _compute_leaf_probs(self, x):
        """Compute probability of each leaf via product of split decisions.

        Identical structure to hierarchical_tree but using KD split nodes
        with temperature control instead of plain sigmoid gates.

        x: (B, T, d) -> leaf_probs: (B, T, n_leaves)
        """
        B, T, _ = x.shape

        # Compute all split probabilities
        split_probs = [split(x, self.temperature) for split in self.splits]

        # Compute leaf probabilities by tracing paths
        leaf_prob_list = []
        for leaf_idx in range(self.n_leaves):
            prob = mx.ones((B, T, 1))
            node = 0
            for d in range(self.depth):
                p_left = split_probs[node]  # (B, T, 1)
                bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                if bit == 0:  # go left
                    prob = prob * p_left
                    node = 2 * node + 1
                else:  # go right
                    prob = prob * (1 - p_left)
                    node = 2 * node + 2
            leaf_prob_list.append(prob)

        leaf_probs = mx.concatenate(leaf_prob_list, axis=-1)

        # Record split sharpness for diagnostics
        # Sharpness = how far from 0.5 the average split probability is
        sharpness_vals = []
        for sp in split_probs:
            mean_p = mx.mean(sp).item()
            sharpness_vals.append(abs(mean_p - 0.5))
        self._split_sharpness = sharpness_vals

        return leaf_probs

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape

        # Get full leaf probability distribution
        leaf_probs = self._compute_leaf_probs(x)  # (B, T, n_leaves)
        self._leaf_probs = leaf_probs

        # Top-k leaf selection
        top_vals = mx.topk(leaf_probs, self.beam_width, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (leaf_probs >= threshold).astype(mx.float32)

        # Renormalize selected leaf weights
        masked_probs = leaf_probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Compute weighted output
        out = mx.zeros_like(x)
        for i, leaf in enumerate(self.leaves):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * leaf(x)

        return out

    def set_temperature(self, temp: float):
        """Set routing temperature (call during training for annealing)."""
        self.temperature = min(temp, self.max_temperature)

    def balance_loss(self) -> mx.array:
        """Balance loss: L = n_leaves * sum(mean_prob^2). Min at uniform."""
        if self._leaf_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))
        return self.n_leaves * mx.sum(mean_probs * mean_probs)

    def entropy_loss(self) -> mx.array:
        """Leaf distribution entropy (we MINIMIZE this for sharp routing)."""
        if self._leaf_probs is None:
            return mx.array(0.0)
        eps = 1e-8
        lp = self._leaf_probs
        entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)
        return mx.mean(entropy)

    def split_diversity_loss(self) -> mx.array:
        """Encourage split directions to be diverse (not all splitting same dim).

        Penalizes high cosine similarity between split projection vectors
        at the same tree depth. Siblings should split along different directions.

        L = mean of |cos(v_i, v_j)|^2 for sibling pairs.
        """
        total = mx.array(0.0)
        n_pairs = 0

        # Group splits by depth level
        for depth_level in range(self.depth):
            start = (2 ** depth_level) - 1
            end = (2 ** (depth_level + 1)) - 1
            if end - start < 2:
                continue

            # Collect projection vectors at this level
            vecs = []
            for i in range(start, min(end, self.n_internal)):
                v = self.splits[i].proj.weight  # (1, d)
                v = v.reshape(-1)  # (d,)
                vecs.append(v)

            if len(vecs) < 2:
                continue

            # Pairwise cosine similarity
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    cos_sim = mx.sum(vecs[i] * vecs[j]) / (
                        mx.sqrt(mx.sum(vecs[i] * vecs[i]) + 1e-8) *
                        mx.sqrt(mx.sum(vecs[j] * vecs[j]) + 1e-8)
                    )
                    total = total + cos_sim * cos_sim
                    n_pairs += 1

        if n_pairs == 0:
            return mx.array(0.0)
        return total / n_pairs


class KDTreeBlock(nn.Module):
    """Transformer block with KDTreeCapsuleRouter replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2, init_temperature: float = 1.0,
                 max_temperature: float = 10.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.tree = KDTreeCapsuleRouter(
            n_embd, depth=tree_depth,
            n_capsules_per_leaf=n_capsules_per_leaf,
            beam_width=beam_width,
            init_temperature=init_temperature,
            max_temperature=max_temperature,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.tree(self.norm2(x))
        return x


@register("kd_tree_routing", parent="hierarchical_tree")
class KDTreeRoutingGPT(nn.Module):
    """GPT with KD-tree feature-space partitioning for expert routing.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - KDTreeCapsuleRouter: depth-D KD-tree with learned split directions
    - Language model head (same as GPT)

    Key difference from hierarchical_tree:
    - Split nodes learn a projection direction + threshold (KD-tree style)
    - Temperature annealing: soft splits during training, hard during inference
    - Split diversity loss encourages different split directions at each level
    - Produces non-overlapping expert territories by construction

    Default config: d=64, depth=3, 8 leaves, 32 capsules/leaf (P=256 total),
    beam=2 (matches k=2 optimal from composition experiments).
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2, init_temperature: float = 1.0,
                 max_temperature: float = 10.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [KDTreeBlock(n_embd, n_head, tree_depth,
                                    n_capsules_per_leaf, beam_width,
                                    init_temperature, max_temperature)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._step = 0
        self._total_steps = 500  # default, set by caller

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def step_temperature(self, step: int, total_steps: int):
        """Anneal temperature: linear ramp from init to max over training.

        Schedule: T(s) = T_init + (T_max - T_init) * s / S_total
        First 20% of training: keep temperature at init (warm-up for soft routing)
        Remaining 80%: linear ramp to max temperature
        """
        self._step = step
        self._total_steps = total_steps
        warmup_frac = 0.2
        if step < warmup_frac * total_steps:
            temp = self.layers[0].tree.temperature  # keep current
        else:
            progress = (step - warmup_frac * total_steps) / ((1 - warmup_frac) * total_steps)
            progress = min(max(progress, 0.0), 1.0)
            init_t = 1.0
            max_t = self.layers[0].tree.max_temperature
            temp = init_t + (max_t - init_t) * progress

        for layer in self.layers:
            layer.tree.set_temperature(temp)

    def aux_loss(self) -> mx.array:
        """Combined auxiliary loss: balance + entropy + split diversity."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.tree.balance_loss()
            total = total + 0.1 * layer.tree.entropy_loss()
            total = total + 0.05 * layer.tree.split_diversity_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def get_routing_diagnostics(self) -> dict:
        """Return diagnostic info about learned split structure."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            tree = layer.tree
            layer_diag = {
                "temperature": tree.temperature,
                "split_sharpness": tree._split_sharpness,
            }
            if tree._leaf_probs is not None:
                lp = tree._leaf_probs
                # Routing entropy
                eps = 1e-8
                entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)
                max_entropy = math.log(tree.n_leaves)
                norm_entropy = mx.mean(entropy).item() / max_entropy

                # Max leaf probability (how confident the routing is)
                max_prob = mx.mean(mx.max(lp, axis=-1)).item()

                # Leaf utilization
                leaf_usage = mx.mean(lp, axis=(0, 1))
                mx.eval(leaf_usage)
                usage_list = leaf_usage.tolist()

                layer_diag["normalized_entropy"] = norm_entropy
                layer_diag["mean_max_leaf_prob"] = max_prob
                layer_diag["leaf_usage"] = usage_list

            # Split direction analysis
            split_dirs = []
            for split in tree.splits:
                w = split.proj.weight.reshape(-1)
                mx.eval(w)
                # Find the dimension with highest absolute weight
                abs_w = mx.abs(w)
                top_dim = mx.argmax(abs_w).item()
                concentration = mx.max(abs_w).item() / (mx.sum(abs_w).item() + 1e-8)
                split_dirs.append({
                    "top_dim": top_dim,
                    "concentration": concentration,
                    "threshold": split.threshold.tolist(),
                })
            layer_diag["split_directions"] = split_dirs

            diagnostics[f"layer_{li}"] = layer_diag
        return diagnostics
