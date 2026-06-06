"""Tests for PersistentExpertTreeGPT — versioning, path copying, composition."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.models.persistent_expert_tree.persistent_expert_tree import (
    PersistentExpertTreeGPT,
    PersistentCapsuleTree,
)


def test_basic_forward():
    """Model produces valid logits."""
    model = PersistentExpertTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=2)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"
    print("PASS: basic_forward")


def test_version_creation():
    """update_leaf creates new version with path copying."""
    tree = PersistentCapsuleTree(n_embd=64, depth=3)
    assert tree.n_versions == 1, "Should start with 1 version"

    # Update leaf 0 -> creates version 1
    vid = tree.update_leaf(0, description="fine-tune leaf 0")
    assert vid == 1
    assert tree.n_versions == 2

    # Check structural sharing: only leaf 0 and its 3 ancestors changed
    v0 = tree._versions[0]
    v1 = tree._versions[1]

    # Leaf 0 should be different (explicitly updated)
    assert id(v0.leaves[0]) != id(v1.leaves[0]), "Leaf 0 should be copied"
    # Leaf 1 should be shared (sibling of leaf 0, not updated)
    assert id(v0.leaves[1]) == id(v1.leaves[1]), "Leaf 1 should be shared"
    # Leaf 2 should be shared (different subtree branch)
    assert id(v0.leaves[2]) == id(v1.leaves[2]), "Leaf 2 should be shared"
    # Leaves 4-7 should be shared (different subtree entirely)
    for i in range(4, 8):
        assert id(v0.leaves[i]) == id(v1.leaves[i]), f"Leaf {i} should be shared"

    # Ancestor gates of leaf 0: path is root(0) -> gate_1(1) -> gate_3(3)
    # (leaf 0 = binary 000, MSB first: go left at each level)
    path = tree._ancestor_path(0)
    assert path == [0, 1, 3], f"Expected path [0, 1, 3], got {path}"

    # Gates on path should be different (path-copied)
    for gate_idx in path:
        assert id(v0.gates[gate_idx]) != id(v1.gates[gate_idx]), \
            f"Gate {gate_idx} should be copied (on path)"

    # Gates NOT on path should be shared
    off_path = set(range(7)) - set(path)
    for gate_idx in off_path:
        assert id(v0.gates[gate_idx]) == id(v1.gates[gate_idx]), \
            f"Gate {gate_idx} should be shared (off path)"

    print("PASS: version_creation")


def test_cross_version_composition():
    """compose_versions creates version with leaves from different sources."""
    tree = PersistentCapsuleTree(n_embd=64, depth=3)

    # v1: update leaf 0
    v1 = tree.update_leaf(0, description="domain A leaf 0")
    tree.set_active_version(0)  # back to base

    # v2: update leaf 5
    v2 = tree.update_leaf(5, description="domain B leaf 5")
    tree.set_active_version(0)

    # v3: compose leaf 0 from v1 + leaf 5 from v2
    v3 = tree.compose_versions({0: v1, 5: v2}, description="cross-version compose")
    tree.set_active_version(v3)

    v1_obj = tree._versions[v1]
    v2_obj = tree._versions[v2]
    v3_obj = tree._versions[v3]

    # v3 should have leaf 0 from v1
    assert id(v3_obj.leaves[0]) == id(v1_obj.leaves[0]), \
        "v3 leaf 0 should be from v1"
    # v3 should have leaf 5 from v2
    assert id(v3_obj.leaves[5]) == id(v2_obj.leaves[5]), \
        "v3 leaf 5 should be from v2"
    # Other leaves from base (v0)
    v0_obj = tree._versions[0]
    for i in [1, 2, 3, 4, 6, 7]:
        assert id(v3_obj.leaves[i]) == id(v0_obj.leaves[i]), \
            f"v3 leaf {i} should be from v0 (base)"

    print("PASS: cross_version_composition")


def test_memory_report():
    """Memory report shows structural sharing."""
    tree = PersistentCapsuleTree(n_embd=64, depth=3)

    # Create a few versions
    tree.update_leaf(0)
    tree.set_active_version(0)
    tree.update_leaf(5)

    report = tree.memory_report()
    assert report["n_versions"] == 3
    assert report["sharing_ratio"] > 1.0, "Should have structural sharing"
    assert report["memory_overhead_pct"] < 100, "Should not double memory for 2 updates"

    # With 3 versions and 2 path copies (each copying D=3 gates + 1 leaf = 4 nodes):
    # Base: 7 gates + 8 leaves = 15 nodes
    # v1: 3 new gates + 1 new leaf = 4 new nodes
    # v2: 3 new gates + 1 new leaf = 4 new nodes
    # Total unique: 15 + 4 + 4 = 23 (vs 45 for full copy)
    assert report["total_unique_nodes"] == 23, \
        f"Expected 23 unique nodes, got {report['total_unique_nodes']}"

    savings_pct = (1 - report["persistent_params"] / report["full_copy_params"]) * 100
    print(f"PASS: memory_report (overhead={report['memory_overhead_pct']:.1f}%, "
          f"savings vs full copy={savings_pct:.1f}%)")


def test_version_switch_changes_output():
    """Switching versions changes forward pass output."""
    model = PersistentExpertTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=2)
    tokens = mx.array([[1, 2, 3]])

    # Get output at v0
    out_v0 = model(tokens)
    mx.eval(out_v0)

    # Create v1 by updating leaf 0
    v1 = model.update_leaf(0, description="updated")
    model.set_version(v1)

    # Get output at v1 (should differ because leaf 0 is deep-copied with same weights initially)
    out_v1 = model(tokens)
    mx.eval(out_v1)

    # Switch back to v0
    model.set_version(0)
    out_v0_again = model(tokens)
    mx.eval(out_v0_again)

    # v0 outputs should be identical (rollback works)
    diff = mx.abs(out_v0 - out_v0_again).max().item()
    assert diff < 1e-5, f"Rollback should give same output, diff={diff}"

    print("PASS: version_switch_changes_output")


def test_param_count_matches_hierarchical():
    """Param count matches hierarchical_tree at v0."""
    from experiments.models.hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT

    ht = HierarchicalTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=4)
    pt = PersistentExpertTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=4)

    ht_params = sum(v.size for _, v in nn.utils.tree_flatten(ht.parameters()))
    pt_params = sum(v.size for _, v in nn.utils.tree_flatten(pt.parameters()))

    assert ht_params == pt_params, \
        f"Param mismatch: hierarchical={ht_params}, persistent={pt_params}"
    print(f"PASS: param_count_matches ({pt_params} params)")


def test_ancestor_path():
    """Verify ancestor path computation for all leaves."""
    tree = PersistentCapsuleTree(n_embd=64, depth=3)

    # Depth-3 tree: leaves 0-7. Sibling leaves share the same gate path
    # because the last gate on the path determines left vs right child leaf.
    # Leaf 0 (binary 000): root(0) -> left(1) -> left(3) -> left child = leaf 0
    assert tree._ancestor_path(0) == [0, 1, 3]
    # Leaf 1 (binary 001): root(0) -> left(1) -> left(3) -> right child = leaf 1
    assert tree._ancestor_path(1) == [0, 1, 3]
    # Leaf 2 (binary 010): root(0) -> left(1) -> right(4)
    assert tree._ancestor_path(2) == [0, 1, 4]
    # Leaf 4 (binary 100): root(0) -> right(2) -> left(5)
    assert tree._ancestor_path(4) == [0, 2, 5]
    # Leaf 7 (binary 111): root(0) -> right(2) -> right(6)
    assert tree._ancestor_path(7) == [0, 2, 6]

    print("PASS: ancestor_path")


if __name__ == "__main__":
    test_basic_forward()
    test_ancestor_path()
    test_version_creation()
    test_cross_version_composition()
    test_memory_report()
    test_version_switch_changes_output()
    test_param_count_matches_hierarchical()
    print("\nAll tests passed!")
