"""Subtree Grafting Composition — leverage tree topology for expert merging.

Extends HierarchicalTreeGPT with subtree grafting composition. Instead of
weight-averaging domain-specific parameters (which destroys learned routing
decisions), grafting attaches each domain's trained subtree onto the shared
root, preserving internal routing structure.

Key idea: In a depth-3 binary tree with 2 domains:
- Domain A owns the left subtree (leaves 0-3, gates 1,3,4)
- Domain B owns the right subtree (leaves 4-7, gates 2,5,6)
- The root gate (gate 0) becomes a domain router

After domain-specific fine-tuning:
- Grafting takes domain A's left-subtree params + domain B's right-subtree params
- Only the root gate needs retraining (to route between the two domain subtrees)
- All internal routing decisions within subtrees are preserved exactly

This is registered as a separate model but uses the same architecture as
hierarchical_tree. The difference is purely in the composition protocol,
not the model architecture.
"""

from .. import register
from ..hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT


@register("subtree_grafting", parent="hierarchical_tree")
class SubtreeGraftingGPT(HierarchicalTreeGPT):
    """HierarchicalTreeGPT with subtree grafting composition protocol.

    Architecturally identical to HierarchicalTreeGPT. The model exists as a
    separate registry entry to track the subtree grafting experiment results
    independently from the weight-averaging composition results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
