"""Persistent Expert Tree — version-aware expert composition via path copying.

Extends HierarchicalTreeGPT with Okasaki-style persistent binary tree semantics.
When an expert leaf is updated, only the root-to-leaf path is copied (O(log L)
nodes); all other nodes are shared with the previous version via structural sharing.

Key concepts:
- TreeVersion: immutable snapshot of the tree at a point in time
- Path copying: update_leaf() creates new version with O(D) copied nodes
- Cross-version composition: compose experts from different versions
- Structural sharing: versions share most of their nodes in memory

Architecture (same as hierarchical_tree, different composition protocol):
         [root gate]
         /          \\
    [gate_0]      [gate_1]
    /      \\      /      \\
  [g00]  [g01]  [g10]  [g11]
  / \\    / \\    / \\    / \\
 L0  L1 L2  L3 L4  L5 L6  L7

With persistent versioning:
  v0: base tree (all nodes shared)
  v1: update L0 -> copy L0, g00, gate_0, root (4 nodes), share 11 others
  v2: update L5 -> copy L5, g10, gate_1, root (4 nodes), share 11 others
  Cross-version: use L0 from v1, L5 from v2 -> builds v3 by composing paths
"""

import copy
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..hierarchical_tree.hierarchical_tree import (
    HierarchicalTreeGPT,
    HierarchicalCapsuleTree,
    HierarchicalBlock,
    TreeGate,
)
from ..capsule_moe.capsule_moe import CapsuleGroup


@dataclass
class TreeVersion:
    """Immutable snapshot of tree node references at a point in time.

    Stores references (not copies) to gate and leaf modules. Multiple versions
    can reference the same module instances (structural sharing). A new version
    created by path copying will have new module instances only along the
    modified path.

    Fields:
        version_id: unique identifier for this version
        parent_id: version this was derived from (None for base)
        gates: list of gate module references (length = 2^D - 1)
        leaves: list of leaf module references (length = 2^D)
        description: human-readable description of what changed
    """
    version_id: int
    parent_id: int | None
    gates: list  # list of TreeGate references
    leaves: list  # list of CapsuleGroup references
    description: str = ""
    metadata: dict = field(default_factory=dict)


class PersistentCapsuleTree(nn.Module):
    """HierarchicalCapsuleTree with persistent versioning via path copying.

    Wraps the tree's gates and leaves in a version-tracking system. The current
    active version determines which gate/leaf modules are used during forward pass.
    Creating a new version (via update_leaf or compose_versions) uses path copying
    to share unchanged nodes.
    """

    def __init__(self, n_embd: int, depth: int = 3,
                 n_capsules_per_leaf: int = 32, beam_width: int = 2):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.beam_width = beam_width
        self.n_embd = n_embd
        self.n_capsules_per_leaf = n_capsules_per_leaf

        # Create base gates and leaves
        base_gates = [TreeGate(n_embd) for _ in range(self.n_internal)]
        base_leaves = [CapsuleGroup(n_embd, n_capsules_per_leaf)
                       for _ in range(self.n_leaves)]

        # Version 0 = base tree
        self._versions: list[TreeVersion] = [
            TreeVersion(
                version_id=0,
                parent_id=None,
                gates=base_gates,
                leaves=base_leaves,
                description="base",
            )
        ]
        self._active_version_id = 0
        self._next_version_id = 1

        # Expose current gates/leaves as module attributes for MLX parameter tracking
        self.gates = base_gates
        self.leaves = base_leaves
        self._leaf_probs = None

    @property
    def active_version(self) -> TreeVersion:
        return self._versions[self._active_version_id]

    @property
    def n_versions(self) -> int:
        return len(self._versions)

    def set_active_version(self, version_id: int):
        """Switch which version is used for forward pass."""
        if version_id < 0 or version_id >= len(self._versions):
            raise ValueError(f"Version {version_id} does not exist (have {len(self._versions)})")
        self._active_version_id = version_id
        v = self._versions[version_id]
        self.gates = v.gates
        self.leaves = v.leaves

    def _ancestor_path(self, leaf_idx: int) -> list[int]:
        """Return internal node indices on root-to-leaf path (excluding leaf).

        For a depth-D tree, leaf leaf_idx has D ancestors (including root).
        Node indexing: root=0, left_child(i)=2i+1, right_child(i)=2i+2.
        Leaf leaf_idx maps to internal node (n_internal + leaf_idx) in the
        full-tree indexing, but leaves are stored separately.
        """
        # Work backwards from the leaf's parent
        # The leaf's parent in internal-node indexing:
        # leaf_idx's parent = (n_internal + leaf_idx - 1) // 2
        # But we need to trace from root to leaf, not leaf to root.
        path = []
        node = 0  # start at root
        for d in range(self.depth):
            path.append(node)
            bit = (leaf_idx >> (self.depth - 1 - d)) & 1
            if bit == 0:
                node = 2 * node + 1
            else:
                node = 2 * node + 2
        return path

    def _deep_copy_module(self, module):
        """Create a deep copy of an nn.Module with independent parameters.

        Uses MLX array copying instead of Python deepcopy (which is very slow
        on MLX modules due to graph tracing).
        """
        if isinstance(module, TreeGate):
            new_mod = TreeGate(self.n_embd)
        elif isinstance(module, CapsuleGroup):
            new_mod = CapsuleGroup(self.n_embd, self.n_capsules_per_leaf)
        else:
            raise TypeError(f"Cannot copy module of type {type(module)}")

        # Copy parameter values
        src_params = dict(nn.utils.tree_flatten(module.parameters()))
        new_params = dict(nn.utils.tree_flatten(new_mod.parameters()))
        updates = {}
        for key in new_params:
            if key in src_params:
                updates[key] = mx.array(src_params[key])
        new_mod.update(nn.utils.tree_unflatten(list(updates.items())))
        return new_mod

    def update_leaf(self, leaf_idx: int, new_leaf: CapsuleGroup | None = None,
                    description: str = "") -> int:
        """Create new version with updated leaf, using path copying.

        Only copies the O(D) ancestor gates and the target leaf.
        All other nodes are shared references from the current version.

        Args:
            leaf_idx: which leaf to update (0..n_leaves-1)
            new_leaf: replacement leaf module (if None, deep-copies current)
            description: human-readable note

        Returns:
            new version_id
        """
        current = self.active_version
        new_gates = list(current.gates)  # shallow copy of reference list
        new_leaves = list(current.leaves)  # shallow copy of reference list

        # Path copy: deep-copy each gate on the root-to-leaf path
        path = self._ancestor_path(leaf_idx)
        for gate_idx in path:
            new_gates[gate_idx] = self._deep_copy_module(current.gates[gate_idx])

        # Update the leaf
        if new_leaf is not None:
            new_leaves[leaf_idx] = new_leaf
        else:
            new_leaves[leaf_idx] = self._deep_copy_module(current.leaves[leaf_idx])

        vid = self._next_version_id
        self._next_version_id += 1

        new_version = TreeVersion(
            version_id=vid,
            parent_id=self._active_version_id,
            gates=new_gates,
            leaves=new_leaves,
            description=description or f"update leaf {leaf_idx}",
        )
        self._versions.append(new_version)
        return vid

    def update_leaves(self, leaf_indices: list[int],
                      description: str = "") -> int:
        """Create new version with multiple updated leaves in a single operation.

        Path-copies all ancestor gates (union of paths) and all specified leaves.
        This is more efficient and correct than calling update_leaf() multiple times,
        because it creates a single version where ALL specified leaves are independent
        copies (avoiding the shared-reference mutation bug).

        Args:
            leaf_indices: list of leaf indices to copy (0..n_leaves-1)
            description: human-readable note

        Returns:
            new version_id
        """
        current = self.active_version
        new_gates = list(current.gates)
        new_leaves = list(current.leaves)

        # Collect union of all paths
        gates_to_copy = set()
        for leaf_idx in leaf_indices:
            path = self._ancestor_path(leaf_idx)
            for gate_idx in path:
                gates_to_copy.add(gate_idx)

        # Path-copy all necessary gates
        for gate_idx in gates_to_copy:
            new_gates[gate_idx] = self._deep_copy_module(current.gates[gate_idx])

        # Copy all specified leaves
        for leaf_idx in leaf_indices:
            new_leaves[leaf_idx] = self._deep_copy_module(current.leaves[leaf_idx])

        vid = self._next_version_id
        self._next_version_id += 1

        new_version = TreeVersion(
            version_id=vid,
            parent_id=self._active_version_id,
            gates=new_gates,
            leaves=new_leaves,
            description=description or f"update leaves {leaf_indices}",
        )
        self._versions.append(new_version)
        return vid

    def compose_versions(self, leaf_version_map: dict[int, int],
                         description: str = "") -> int:
        """Create new version by cherry-picking leaves from different versions.

        This is the key operation: cross-version composition. Take leaf i from
        version v_i, leaf j from version v_j, etc. Path-copy all necessary
        gates to create a consistent new tree.

        Args:
            leaf_version_map: {leaf_idx: version_id} for leaves to pick.
                              Leaves not specified use current active version.
            description: human-readable note

        Returns:
            new version_id
        """
        current = self.active_version
        new_gates = list(current.gates)  # start with current version's gates
        new_leaves = list(current.leaves)

        # Collect all gates that need path-copying (union of paths)
        gates_to_copy = set()
        for leaf_idx, src_version_id in leaf_version_map.items():
            if src_version_id < 0 or src_version_id >= len(self._versions):
                raise ValueError(f"Version {src_version_id} does not exist")
            # Get the leaf from the source version
            src = self._versions[src_version_id]
            new_leaves[leaf_idx] = src.leaves[leaf_idx]

            # Mark ancestor gates for copying from source version
            path = self._ancestor_path(leaf_idx)
            for gate_idx in path:
                gates_to_copy.add(gate_idx)

        # For shared gates (on paths of multiple leaves from different versions),
        # we need to create fresh copies and let calibration learn the right values.
        # For gates on a single leaf's path, we can take from that leaf's source version.
        # Simple approach: deep-copy all gates on any modified path from current version.
        # This is conservative but correct -- calibration will fix the gate values.
        for gate_idx in gates_to_copy:
            new_gates[gate_idx] = self._deep_copy_module(current.gates[gate_idx])

        vid = self._next_version_id
        self._next_version_id += 1

        new_version = TreeVersion(
            version_id=vid,
            parent_id=self._active_version_id,
            gates=new_gates,
            leaves=new_leaves,
            description=description or f"compose from {leaf_version_map}",
            metadata={"leaf_version_map": leaf_version_map},
        )
        self._versions.append(new_version)
        return vid

    def memory_report(self) -> dict:
        """Report memory usage showing structural sharing.

        Returns dict with:
        - total_unique_nodes: unique module instances across all versions
        - total_node_refs: total references across all versions
        - sharing_ratio: refs/unique (higher = more sharing)
        - per_version_unique: list of unique nodes per version
        """
        all_gate_ids = set()
        all_leaf_ids = set()
        per_version = []

        for v in self._versions:
            v_gate_ids = {id(g) for g in v.gates}
            v_leaf_ids = {id(l) for l in v.leaves}
            all_gate_ids.update(v_gate_ids)
            all_leaf_ids.update(v_leaf_ids)
            per_version.append({
                "version": v.version_id,
                "unique_gates": len(v_gate_ids),
                "unique_leaves": len(v_leaf_ids),
            })

        total_unique = len(all_gate_ids) + len(all_leaf_ids)
        total_refs = sum(len(v.gates) + len(v.leaves) for v in self._versions)

        # Memory: count parameters in unique modules
        unique_modules = set()
        for v in self._versions:
            for g in v.gates:
                unique_modules.add(id(g))
            for l in v.leaves:
                unique_modules.add(id(l))

        # Actual parameter count for unique modules
        param_count = 0
        seen = set()
        for v in self._versions:
            for g in v.gates:
                if id(g) not in seen:
                    seen.add(id(g))
                    param_count += sum(v.size for _, v in nn.utils.tree_flatten(g.parameters()))
            for l in v.leaves:
                if id(l) not in seen:
                    seen.add(id(l))
                    param_count += sum(v.size for _, v in nn.utils.tree_flatten(l.parameters()))

        # Compare to non-persistent (full copy per version)
        base_params = sum(v.size for _, v in nn.utils.tree_flatten(
            self._versions[0].gates[0].parameters())) * self.n_internal
        base_params += sum(v.size for _, v in nn.utils.tree_flatten(
            self._versions[0].leaves[0].parameters())) * self.n_leaves
        full_copy_params = base_params * len(self._versions)

        return {
            "n_versions": len(self._versions),
            "total_unique_nodes": total_unique,
            "total_node_refs": total_refs,
            "sharing_ratio": total_refs / max(total_unique, 1),
            "persistent_params": param_count,
            "full_copy_params": full_copy_params,
            "memory_overhead_pct": ((param_count - base_params) / base_params * 100)
                                   if base_params > 0 else 0,
            "per_version": per_version,
        }

    # ── Forward pass (delegates to active version) ────────────────────────

    def _compute_all_gate_probs(self, x):
        return [gate(x) for gate in self.gates]

    def _tree_beam_routing(self, x):
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
        B, T, D = x.shape
        leaf_probs = self._tree_beam_routing(x)
        self._leaf_probs = leaf_probs

        top_vals = mx.topk(leaf_probs, self.beam_width, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (leaf_probs >= threshold).astype(mx.float32)

        masked_probs = leaf_probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, leaf in enumerate(self.leaves):
            w = masked_probs[..., i:i+1]
            out = out + w * leaf(x)

        return out

    def balance_loss(self) -> mx.array:
        if self._leaf_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._leaf_probs, axis=(0, 1))
        return self.n_leaves * mx.sum(mean_probs * mean_probs)

    def gate_entropy_loss(self) -> mx.array:
        if self._leaf_probs is None:
            return mx.array(0.0)
        eps = 1e-8
        lp = self._leaf_probs
        entropy = -mx.sum(lp * mx.log(lp + eps), axis=-1)
        return mx.mean(entropy)


class PersistentBlock(nn.Module):
    """Transformer block with PersistentCapsuleTree."""

    def __init__(self, n_embd: int, n_head: int,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2):
        super().__init__()
        from ..gpt import RMSNorm, CausalSelfAttention
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.tree = PersistentCapsuleTree(
            n_embd, depth=tree_depth,
            n_capsules_per_leaf=n_capsules_per_leaf,
            beam_width=beam_width,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.tree(self.norm2(x))
        return x


@register("persistent_expert_tree", parent="hierarchical_tree")
class PersistentExpertTreeGPT(nn.Module):
    """GPT with persistent versioned binary tree of capsule groups.

    Architecturally identical to HierarchicalTreeGPT but with version tracking
    on the tree structure. Each composition or expert update creates a new
    version via path copying, sharing unchanged nodes with previous versions.

    Default config matches hierarchical_tree: d=64, depth=3, 8 leaves,
    32 capsules/leaf, beam=2.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 tree_depth: int = 3, n_capsules_per_leaf: int = 32,
                 beam_width: int = 2):
        super().__init__()
        from ..gpt import RMSNorm
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [PersistentBlock(n_embd, n_head, tree_depth,
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
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.tree.balance_loss()
            total = total + 0.1 * layer.tree.gate_entropy_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    # ── Version management API ──────────────────────────────────────────

    def set_version(self, version_id: int):
        """Set all layers to use the specified version."""
        for layer in self.layers:
            layer.tree.set_active_version(version_id)

    def update_leaf(self, leaf_idx: int, description: str = "") -> int:
        """Create new version across all layers with updated leaf."""
        vid = None
        for layer in self.layers:
            vid = layer.tree.update_leaf(leaf_idx, description=description)
        return vid

    def update_leaves(self, leaf_indices: list[int], description: str = "") -> int:
        """Create new version across all layers with multiple updated leaves."""
        vid = None
        for layer in self.layers:
            vid = layer.tree.update_leaves(leaf_indices, description=description)
        return vid

    def compose_versions(self, leaf_version_map: dict[int, int],
                         description: str = "") -> int:
        """Create cross-version composition across all layers."""
        vid = None
        for layer in self.layers:
            vid = layer.tree.compose_versions(leaf_version_map, description=description)
        return vid

    def memory_report(self) -> dict:
        """Aggregate memory report across all layers."""
        reports = [layer.tree.memory_report() for layer in self.layers]
        total_persistent = sum(r["persistent_params"] for r in reports)
        total_full_copy = sum(r["full_copy_params"] for r in reports)
        base_params = sum(r["full_copy_params"] / r["n_versions"] for r in reports)
        overhead = ((total_persistent - base_params) / base_params * 100) if base_params > 0 else 0

        return {
            "n_versions": reports[0]["n_versions"],
            "n_layers": len(reports),
            "total_persistent_params": total_persistent,
            "total_full_copy_params": total_full_copy,
            "base_params": base_params,
            "memory_overhead_pct": overhead,
            "memory_savings_vs_full_copy_pct": (
                (1 - total_persistent / total_full_copy) * 100
                if total_full_copy > 0 else 0
            ),
            "per_layer": reports,
        }
