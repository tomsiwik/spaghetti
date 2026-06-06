# Okasaki - Purely Functional Data Structures (1998)

## Source
- Book: Chris Okasaki, "Purely Functional Data Structures", Cambridge University Press, 1998
- Based on PhD thesis, Carnegie Mellon University, 1996
- Wikipedia: https://en.wikipedia.org/wiki/Persistent_data_structure

## Key Insight
Persistent data structures preserve all previous versions when modified. The key technique for trees is **path copying**: updating a leaf copies only the O(log N) ancestor nodes on the root-to-leaf path, sharing all other nodes with the previous version.

## Relevance to Our Work
The HierarchicalCapsuleTree is a binary tree of expert groups. When experts are updated (fine-tuned, composed), path copying enables:
1. Version-aware composition (compose expert-v3 with expert-v1)
2. O(log L) memory per version (vs O(L) for full copy)
3. Instant rollback by swapping tree roots
4. Composition history as a DAG of tree versions

## Key Techniques
- **Path copying**: Copy root-to-modified-leaf path, share everything else. O(log N) space per update.
- **Fat node**: Store version stamps in nodes (O(1) space but O(log V) access time per version query).
- **Structural sharing**: Immutable nodes can be safely referenced by multiple tree versions.

## Connection to exp_persistent_expert_tree
Direct application of path-copying persistent binary tree to the HierarchicalCapsuleTree architecture.
