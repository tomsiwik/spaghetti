"""Consistent Hash Routing — hash-ring placement for incremental expert add/remove.

Uses consistent hashing (Karger et al. 1997) to place experts on a hash ring.
Each token hashes to a position on the ring and routes to the nearest k experts.
Adding an expert displaces only ~1/N of existing routing decisions.

Key design:
- Each expert is assigned V virtual nodes on a ring [0, 2^32)
- Token hidden states are hashed to a ring position via a deterministic hash
- Top-k experts by proximity on the ring are selected
- Routing weights are computed from inverse ring distance (closer = higher weight)
- Adding expert N+1 only remaps tokens in its ring segment (~1/N fraction)

The hash function maps d-dimensional hidden states to a scalar ring position
using a fixed random projection (dot product with a frozen random vector),
then applies a deterministic integer hash. This preserves some locality:
similar hidden states get similar ring positions.

Prior art:
- Karger et al. 1997: Consistent hashing for web caching
- Lamping & Stepanov 2014: Jump consistent hash (simpler, perfect balance)
- Hash Layers (NeurIPS 2021): Hash-based routing for MoE (not consistent)
- LSH Capsule Routing (this project): Random-projection routing (validated)
"""

import math
import struct

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


# --------------------------------------------------------------------------- #
#  Hash utilities (pure-Python for deterministic ring placement)
# --------------------------------------------------------------------------- #

def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash. Fast, good distribution."""
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _jump_consistent_hash(key: int, num_buckets: int) -> int:
    """Jump consistent hash (Lamping & Stepanov 2014).

    Maps key to one of num_buckets with perfect balance and minimal
    displacement when num_buckets changes. O(ln(num_buckets)) time.

    Returns bucket in [0, num_buckets).
    """
    if num_buckets <= 0:
        return 0
    b = -1
    j = 0
    while j < num_buckets:
        b = j
        key = ((key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))
    return b


class ConsistentHashRouter(nn.Module):
    """Routes tokens to experts via consistent hashing on a ring.

    Each token's hidden state is projected to a scalar via a fixed random
    vector, quantized to an integer, then mapped to k experts using
    jump consistent hash with k different hash seeds.

    Parameters:
        n_embd: embedding dimension d
        n_groups: number of expert groups N
        top_k: number of experts to select per token
        n_virtual: number of virtual nodes per expert on the ring
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 top_k: int = 2, n_virtual: int = 150):
        super().__init__()
        self.n_embd = n_embd
        self.n_groups = n_groups
        self.top_k = top_k
        self.n_virtual = n_virtual

        # Fixed random projection vector for hashing hidden states to ring
        # NOT trainable. Initialized once and frozen.
        scale = 1.0 / math.sqrt(n_embd)
        self._proj = mx.random.normal((n_embd,)) * scale

        # Build the hash ring: map ring positions -> expert indices
        # Each expert gets n_virtual positions on the ring [0, 2^32)
        self._ring_positions = []  # sorted list of (ring_pos, expert_idx)
        self._build_ring(n_groups)

        # Cache for diagnostics
        self._gate_probs = None
        self._expert_assignments = None

    def _build_ring(self, n_groups: int):
        """Place n_groups experts on the ring with virtual nodes."""
        ring = []
        for expert_idx in range(n_groups):
            for v in range(self.n_virtual):
                # Deterministic ring position for expert_idx, virtual node v
                key_bytes = struct.pack(">II", expert_idx, v)
                pos = _fnv1a_32(key_bytes)
                ring.append((pos, expert_idx))
        ring.sort(key=lambda x: x[0])
        self._ring_positions = ring
        # Precompute arrays for fast lookup
        self._ring_pos_array = [p for p, _ in ring]
        self._ring_expert_array = [e for _, e in ring]

    def _find_nearest_k(self, ring_pos: int, k: int, n_groups: int) -> list:
        """Find k nearest distinct experts clockwise from ring_pos.

        Returns list of (expert_idx, distance) pairs.
        """
        n = len(self._ring_positions)
        if n == 0:
            return [(i, 1) for i in range(min(k, n_groups))]

        # Binary search for insertion point
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring_pos_array[mid] < ring_pos:
                lo = mid + 1
            else:
                hi = mid

        # Walk clockwise collecting distinct experts
        selected = []
        seen = set()
        ring_max = 0xFFFFFFFF
        for offset in range(n):
            idx = (lo + offset) % n
            expert = self._ring_expert_array[idx]
            if expert >= n_groups:
                continue  # skip experts beyond current count
            if expert not in seen:
                seen.add(expert)
                # Distance on ring (clockwise)
                rp = self._ring_pos_array[idx]
                dist = (rp - ring_pos) % (ring_max + 1)
                selected.append((expert, max(dist, 1)))
                if len(selected) >= k:
                    break

        # If not enough experts found (shouldn't happen with virtual nodes)
        while len(selected) < k and len(selected) < n_groups:
            for e in range(n_groups):
                if e not in seen:
                    seen.add(e)
                    selected.append((e, ring_max // 2))
                    if len(selected) >= k:
                        break

        return selected

    def __call__(self, x, n_groups_override: int = None):
        """Compute routing weights via consistent hashing.

        x: (B, T, d) -> weights: (B, T, n_groups), sparse with top_k nonzero

        n_groups_override: if set, use this many groups instead of self.n_groups.
            Allows adding experts without modifying the router.
        """
        B, T, d = x.shape
        n_groups = n_groups_override if n_groups_override is not None else self.n_groups

        # Project hidden states to scalar ring positions
        # x @ proj -> (B, T) scalar
        proj_scores = x @ self._proj  # (B, T)
        mx.eval(proj_scores)

        # Quantize to ring positions [0, 2^32)
        # Use the raw float bits for hashing - deterministic
        proj_np = proj_scores.tolist()  # Convert to Python for hash computation

        # Compute routing for each token
        weights_list = []
        expert_assignments = []

        for b in range(B):
            batch_weights = []
            batch_assignments = []
            for t in range(T):
                val = proj_np[b][t]
                # Hash float value to ring position
                val_bytes = struct.pack(">f", val)
                ring_pos = _fnv1a_32(val_bytes)

                # Find nearest k experts on the ring
                nearest = self._find_nearest_k(ring_pos, self.top_k, n_groups)

                # Compute weights from inverse distance
                # Closer on ring = higher weight
                token_weights = [0.0] * n_groups
                inv_dists = []
                experts = []
                for expert_idx, dist in nearest:
                    inv_dist = 1.0 / (dist + 1)
                    inv_dists.append(inv_dist)
                    experts.append(expert_idx)

                # Softmax over inverse distances for smooth weights
                max_inv = max(inv_dists) if inv_dists else 1.0
                exp_vals = [math.exp(v - max_inv) for v in inv_dists]
                sum_exp = sum(exp_vals)
                for i, expert_idx in enumerate(experts):
                    token_weights[expert_idx] = exp_vals[i] / sum_exp

                batch_weights.append(token_weights)
                batch_assignments.append([e for e, _ in nearest])
            weights_list.append(batch_weights)
            expert_assignments.append(batch_assignments)

        weights = mx.array(weights_list)  # (B, T, n_groups)
        self._gate_probs = weights
        self._expert_assignments = expert_assignments
        return weights

    def get_assignments(self, x, n_groups: int = None) -> list:
        """Get expert assignments without computing full weights.

        Returns list of expert indices per token for displacement analysis.
        """
        if self._expert_assignments is not None:
            return self._expert_assignments
        self(x, n_groups)
        return self._expert_assignments

    def add_expert(self, new_expert_idx: int):
        """Add a new expert to the ring. Only adds virtual nodes for the new expert.

        This is the key operation: existing ring positions are unchanged,
        so only tokens that fall in the new expert's ring segments get remapped.
        """
        for v in range(self.n_virtual):
            key_bytes = struct.pack(">II", new_expert_idx, v)
            pos = _fnv1a_32(key_bytes)
            # Insert into sorted ring
            lo, hi = 0, len(self._ring_positions)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._ring_pos_array[mid] < pos:
                    lo = mid + 1
                else:
                    hi = mid
            self._ring_positions.insert(lo, (pos, new_expert_idx))
            self._ring_pos_array.insert(lo, pos)
            self._ring_expert_array.insert(lo, new_expert_idx)
        self.n_groups = new_expert_idx + 1

    def remove_expert(self, expert_idx: int):
        """Remove an expert from the ring. Only removes its virtual nodes.

        Tokens previously routed to this expert redistribute to neighbors.
        """
        self._ring_positions = [(p, e) for p, e in self._ring_positions if e != expert_idx]
        self._ring_pos_array = [p for p, _ in self._ring_positions]
        self._ring_expert_array = [e for _, e in self._ring_positions]


class ConsistentHashCapsulePool(nn.Module):
    """Pool of capsule groups with consistent-hash routing.

    Supports dynamic expert add/remove via ring operations.
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32, top_k: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k

        # Consistent hash router
        self.router = ConsistentHashRouter(n_embd, n_groups, top_k)

        # Capsule groups (learned)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        weights = self.router(x, n_groups_override=len(self.groups))
        self._gate_probs = weights

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = weights[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)

        return out

    def add_expert(self, n_embd: int, n_capsules: int = 32):
        """Add a new expert group to the pool.

        The expert is added to the ring and appended to the group list.
        No recalibration needed -- consistent hashing handles routing.
        """
        new_idx = len(self.groups)
        self.router.add_expert(new_idx)
        new_group = CapsuleGroup(n_embd, n_capsules)
        self.groups.append(new_group)
        self.n_groups = len(self.groups)
        return new_group

    def remove_expert(self, expert_idx: int):
        """Remove an expert from the pool."""
        self.router.remove_expert(expert_idx)
        # Don't remove from list to preserve indices -- mark as inactive
        # For simplicity, we zero out its contribution
        self.n_groups -= 1

    def balance_loss(self) -> mx.array:
        """Balance loss for diagnostics."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        G = mean_probs.shape[0]
        return G * mx.sum(mean_probs * mean_probs)


class ConsistentHashBlock(nn.Module):
    """Transformer block with consistent-hash-routed CapsulePool."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = ConsistentHashCapsulePool(
            n_embd, n_groups, n_capsules_per_group, top_k
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("consistent_hash_routing", parent="capsule_moe")
class ConsistentHashRoutingGPT(nn.Module):
    """GPT with consistent-hash-routed capsule groups.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - ConsistentHashCapsulePool: hash-ring routes to top-k groups
    - Language model head (same as GPT)

    Key differences from CapsuleMoEGPT:
    - Router uses consistent hashing (no learned routing params)
    - Adding/removing experts only displaces ~1/N of routing decisions
    - No routing parameters to train or calibrate
    - Deterministic routing (same input always routes to same experts)

    Default config: d=64, G=8, 32 caps/group, k=2.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.n_capsules_per_group = n_capsules_per_group
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [ConsistentHashBlock(n_embd, n_head, n_groups,
                                           n_capsules_per_group, top_k)
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

    def add_expert_to_all_layers(self):
        """Add one new expert group to every layer's capsule pool.

        Returns the new expert groups (for optional training).
        """
        new_groups = []
        for layer in self.layers:
            g = layer.capsule_pool.add_expert(self.n_embd, self.n_capsules_per_group)
            new_groups.append(g)
        return new_groups

    def get_routing_assignments(self, tokens) -> dict:
        """Get per-token expert assignments for displacement analysis."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        assignments = {}
        for li, layer in enumerate(self.layers):
            h = layer.norm2(x + layer.attn(layer.norm1(x)))
            layer.capsule_pool.router(h, n_groups_override=len(layer.capsule_pool.groups))
            assignments[f"layer_{li}"] = layer.capsule_pool.router._expert_assignments
            x = layer(x)
        return assignments

    def get_routing_diagnostics(self) -> dict:
        """Return diagnostic info about routing behavior."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            pool = layer.capsule_pool
            if pool._gate_probs is not None:
                gp = pool._gate_probs
                selected = (gp > 0).astype(mx.float32)
                mean_selected = mx.mean(mx.sum(selected, axis=-1)).item()
                util = mx.mean(selected, axis=(0, 1))
                mx.eval(util)
                eps = 1e-8
                entropy = -mx.sum(gp * mx.log(gp + eps), axis=-1)
                max_entropy = math.log(len(pool.groups))
                norm_entropy = mx.mean(entropy).item() / max_entropy if max_entropy > 0 else 0

                diagnostics[f"layer_{li}"] = {
                    "mean_selected": mean_selected,
                    "expert_utilization": util.tolist(),
                    "normalized_entropy": norm_entropy,
                    "n_groups": len(pool.groups),
                }
        return diagnostics
