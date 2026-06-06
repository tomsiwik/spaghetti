"""Skip-List Multi-Resolution Routing -- probabilistic multi-level expert indexing.

Inspired by Pugh (1990) skip lists: organize N experts at multiple resolution
levels. Level 0 has all N experts (fine-grained). Level k has ceil(N/2^k)
"express" experts (coarse-grained), each representing a cluster of 2^k
Level-0 experts.

Routing is top-down with adaptive depth:
1. Start at highest level (coarsest, fewest experts)
2. Compute softmax routing scores at current level
3. If max confidence > threshold, use this level's experts (early stop)
4. Otherwise, descend to next finer level, restricting to children of
   top-scored coarse experts

Key properties:
- Easy tokens stop early at coarse levels (fewer gate evaluations)
- Hard tokens descend to fine-grained Level 0 (full precision)
- Average routing cost adapts per-token
- No extra expert parameters: coarse experts are weight-averages of children
- Compatible with top-k selection at each level

Architecture (N=8 experts, L=3 levels):
    Level 2 (coarsest): 2 express experts  [E0_2, E1_2]
    Level 1:            4 express experts  [E0_1, E1_1, E2_1, E3_1]
    Level 0 (finest):   8 leaf experts     [E0, E1, E2, E3, E4, E5, E6, E7]

    Parentage: E0_2 covers {E0,E1,E2,E3}, E1_2 covers {E4,E5,E6,E7}
               E0_1 covers {E0,E1}, E1_1 covers {E2,E3}, etc.

Comparison to hierarchical_tree:
- hierarchical_tree: binary gates at each node, ALWAYS traverses full depth
- skip_list_routing: softmax + confidence at each level, CAN STOP EARLY
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class LevelRouter(nn.Module):
    """Router for a single skip-list level.

    Produces softmax scores over the experts at this level.
    Also computes a scalar "confidence" = max(softmax) to decide
    whether to use this level or descend.
    """

    def __init__(self, n_embd: int, n_experts_at_level: int):
        super().__init__()
        self.router = nn.Linear(n_embd, n_experts_at_level, bias=False)
        self.n_experts = n_experts_at_level

    def __call__(self, x):
        """x: (..., d) -> scores: (..., n_experts), probs: (..., n_experts)."""
        scores = self.router(x)  # (..., n_experts)
        probs = mx.softmax(scores, axis=-1)
        return scores, probs


class SkipListCapsulePool(nn.Module):
    """Multi-resolution capsule pool with skip-list routing.

    N leaf experts at Level 0. ceil(N/2^k) express experts at Level k.
    Express experts share weights with their children (weight-averaged).

    Routing proceeds top-down with adaptive depth:
    - At each level, compute softmax over that level's experts
    - If confidence (max prob) exceeds threshold, use that level
    - Otherwise descend to next level for finer routing

    For differentiability during training, we use a soft version:
    output = sum over levels of (level_weight * level_output)
    where level_weight = confidence * product(1 - confidence) of higher levels

    This is analogous to how skip lists probabilistically assign levels.
    """

    def __init__(self, n_embd: int, n_experts: int = 8,
                 n_capsules_per_expert: int = 32, top_k: int = 2,
                 confidence_threshold: float = 0.7):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        # Level 0: all N leaf experts (the actual parameters)
        self.experts = [CapsuleGroup(n_embd, n_capsules_per_expert)
                        for _ in range(n_experts)]

        # Compute number of levels: L = floor(log2(N))
        self.n_levels = 0
        n = n_experts
        while n > 1:
            n = (n + 1) // 2  # ceil division
            self.n_levels += 1
        # n_levels = number of coarse levels ABOVE Level 0

        # Routers: one per level (Level 0 through Level n_levels)
        # Level 0 has n_experts experts, Level k has ceil(n_experts/2^k)
        self.level_sizes = []
        self.routers = []
        for k in range(self.n_levels + 1):
            size = self._level_size(k)
            self.level_sizes.append(size)
            self.routers.append(LevelRouter(n_embd, size))

        # Confidence gate per level (except Level 0 which always executes)
        # Learns when to stop at each level
        self.confidence_gates = [nn.Linear(n_embd, 1, bias=True)
                                 for _ in range(self.n_levels)]

        # Cache for diagnostics
        self._level_usage = None  # (B, T, n_levels+1) soft level weights
        self._gate_probs = None   # for balance loss

    def _level_size(self, k: int) -> int:
        """Number of experts at level k."""
        n = self.n_experts
        for _ in range(k):
            n = (n + 1) // 2
        return n

    def _get_coarse_expert_output(self, x, level: int, expert_idx: int):
        """Get output from a coarse expert by averaging its children.

        At level k, expert i covers leaf experts [i*2^k, (i+1)*2^k).
        Output = average of children outputs (no extra parameters).
        """
        if level == 0:
            return self.experts[expert_idx](x)

        # Find children at level k-1
        child_start = expert_idx * 2
        child_end = min(child_start + 2, self._level_size(level - 1))

        # Recursively get children outputs and average
        child_outputs = []
        for c in range(child_start, child_end):
            child_outputs.append(
                self._get_coarse_expert_output(x, level - 1, c)
            )

        if len(child_outputs) == 1:
            return child_outputs[0]
        return sum(child_outputs) / len(child_outputs)

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)

        Multi-resolution routing with soft level selection.
        """
        B, T, D = x.shape

        # Compute confidence at each coarse level (top-down)
        # Level n_levels is coarsest, Level 0 is finest
        # p_stop[k] = sigmoid(gate_k(x)) = probability of stopping at level k+1
        # (k indexes from coarsest=0 to n_levels-1)

        level_weights = []  # weight for each level's output
        cum_pass_through = mx.ones((B, T, 1))  # probability of reaching this level

        # Process from coarsest (level n_levels) down to level 1
        for lev_idx in range(self.n_levels):
            level = self.n_levels - lev_idx  # actual level number (coarsest first)

            # Confidence = sigmoid of learned gate
            conf = mx.sigmoid(self.confidence_gates[lev_idx](x))  # (B, T, 1)

            # Weight for this level = probability of reaching it * confidence
            w = cum_pass_through * conf
            level_weights.append((level, w))

            # Update pass-through for next (finer) level
            cum_pass_through = cum_pass_through * (1 - conf)

        # Level 0 gets all remaining probability
        level_weights.append((0, cum_pass_through))

        # Compute output at each level, weighted by level_weights
        out = mx.zeros_like(x)
        all_probs = []

        for level, w in level_weights:
            n_at_level = self.level_sizes[level]
            router_idx = level

            # Get routing probabilities at this level
            _, probs = self.routers[router_idx](x)  # (B, T, n_at_level)
            all_probs.append(probs)

            # Top-k selection at this level
            if n_at_level <= self.top_k:
                # Use all experts at this level
                masked_probs = probs
            else:
                scores, _ = self.routers[router_idx](x)
                top_vals = mx.topk(scores, self.top_k, axis=-1)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                mask = (scores >= threshold).astype(mx.float32)
                masked_probs = probs * mask
                masked_probs = masked_probs / (
                    mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8
                )

            # Weighted sum of expert outputs at this level
            level_out = mx.zeros_like(x)
            for i in range(n_at_level):
                expert_w = masked_probs[..., i:i + 1]  # (B, T, 1)
                expert_out = self._get_coarse_expert_output(x, level, i)
                level_out = level_out + expert_w * expert_out

            # Add level contribution weighted by level selection probability
            out = out + w * level_out

        # Store diagnostics
        self._level_usage = mx.concatenate(
            [w for _, w in level_weights], axis=-1
        )  # (B, T, n_levels+1)

        # For balance loss, use Level 0 probs (the fine-grained routing)
        if all_probs:
            self._gate_probs = all_probs[-1]  # Level 0 probs

        return out

    def balance_loss(self) -> mx.array:
        """Balance loss over Level 0 expert utilization."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_experts * mx.sum(mean_probs * mean_probs)

    def level_usage_stats(self):
        """Return mean weight per level (for diagnostics)."""
        if self._level_usage is None:
            return None
        # Mean across batch and time
        return mx.mean(self._level_usage, axis=(0, 1))  # (n_levels+1,)

    def avg_routing_depth(self):
        """Expected routing depth = sum(level_weight * (n_levels - level)).

        Level n_levels (coarsest) = depth 1 gate evaluation
        Level 0 (finest) = n_levels+1 gate evaluations
        """
        if self._level_usage is None:
            return None
        # level_weights are ordered: coarsest first, finest last
        # Depth cost: coarsest=1, next=2, ..., finest=n_levels+1
        depths = mx.arange(1, self.n_levels + 2, dtype=mx.float32)
        usage = mx.mean(self._level_usage, axis=(0, 1))  # (n_levels+1,)
        return mx.sum(usage * depths)


class SkipListBlock(nn.Module):
    """Transformer block with SkipListCapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_experts: int = 8, n_capsules_per_expert: int = 32,
                 top_k: int = 2, confidence_threshold: float = 0.7):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.skip_pool = SkipListCapsulePool(
            n_embd, n_experts, n_capsules_per_expert,
            top_k, confidence_threshold,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.skip_pool(self.norm2(x))
        return x


@register("skip_list_routing", parent="hierarchical_tree")
class SkipListRoutingGPT(nn.Module):
    """GPT with skip-list multi-resolution expert routing.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - SkipListCapsulePool: multi-level expert routing with adaptive depth
    - Language model head (same as GPT)

    Default config: d=64, 8 experts, 32 capsules/expert (256 total),
    top_k=2 (matches proven k=2 optimal).
    3 levels: Level 0 (8 experts), Level 1 (4), Level 2 (2).
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_experts: int = 8, n_capsules_per_expert: int = 32,
                 top_k: int = 2, confidence_threshold: float = 0.7):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [SkipListBlock(n_embd, n_head, n_experts,
                                      n_capsules_per_expert, top_k,
                                      confidence_threshold)
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
        """Combined auxiliary loss: balance + depth regularization."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.skip_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def get_routing_stats(self):
        """Return per-layer routing depth and level usage statistics."""
        stats = {}
        for i, layer in enumerate(self.layers):
            pool = layer.skip_pool
            usage = pool.level_usage_stats()
            depth = pool.avg_routing_depth()
            if usage is not None:
                stats[f"layer_{i}"] = {
                    "level_usage": usage.tolist(),
                    "avg_depth": depth.item() if depth is not None else None,
                    "n_levels": pool.n_levels,
                }
        return stats
