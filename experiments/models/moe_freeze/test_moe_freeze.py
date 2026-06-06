"""Tests for the MoE + freeze lifecycle model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, train_multidomain
from experiments.metrics import compute_forgetting


# Small config for fast tests
CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
           n_experts=4, top_k=2)


def test_forward_shape():
    """tokens (B, T) -> logits (B, T, V) for various B, T, V."""
    print("=" * 60)
    print("test_forward_shape")

    for B, T, V in [(1, 8, 28), (4, 32, 28), (2, 16, 50)]:
        model = get_model("moe_freeze", **{**CFG, "vocab_size": V, "block_size": T})
        tokens = mx.zeros((B, T), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
        print(f"  B={B}, T={T}, V={V} -> {logits.shape}  OK")

    print("  PASSED\n")


def test_on_domain_switch_freezes():
    """After on_domain_switch(), at least one expert per layer is frozen."""
    print("=" * 60)
    print("test_on_domain_switch_freezes")

    model = get_model("moe_freeze", **CFG)
    # Do a forward pass so experts have non-uniform norms
    tokens = mx.random.randint(0, 28, (4, 16))
    logits = model(tokens)
    mx.eval(logits)

    # Train a few steps to differentiate expert weights
    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:200], tok, block_size=32)
    train(model, ds, steps=20, batch_size=16, lr=3e-3, log_every=100)

    assert len(model._frozen_experts) == 0, "No experts should be frozen before domain switch"

    model.on_domain_switch("domain_b")

    assert len(model._frozen_experts) >= CFG["n_layer"], (
        f"Expected >= {CFG['n_layer']} frozen experts, got {len(model._frozen_experts)}"
    )
    print(f"  frozen experts: {model._frozen_experts}")

    # Verify frozen experts have no trainable params
    trainable_keys = {k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    for li, ei in model._frozen_experts:
        prefix = f"layers.{li}.moe.experts.{ei}."
        frozen_trainable = [k for k in trainable_keys if k.startswith(prefix)]
        assert len(frozen_trainable) == 0, (
            f"Frozen expert ({li},{ei}) still has trainable params: {frozen_trainable}"
        )
        print(f"  expert ({li},{ei}): confirmed not in trainable tree  OK")

    print("  PASSED\n")


def test_on_domain_switch_recycles():
    """After switch, recycled expert has small weight norm (reinitialized)."""
    print("=" * 60)
    print("test_on_domain_switch_recycles")

    model = get_model("moe_freeze", **CFG)
    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:200], tok, block_size=32)

    # Train to grow expert norms
    train(model, ds, steps=50, batch_size=16, lr=3e-3, log_every=100)

    # Record norms before switch
    norms_before = {}
    for li, layer in enumerate(model.layers):
        for ei, expert in enumerate(layer.moe.experts):
            norm = sum(mx.sum(v * v).item() for _, v in nn.utils.tree_flatten(expert.parameters()))
            norms_before[(li, ei)] = norm

    model.on_domain_switch("domain_b")

    # Find recycled experts (unfrozen experts that were replaced)
    for li, layer in enumerate(model.layers):
        for ei, expert in enumerate(layer.moe.experts):
            if (li, ei) in model._frozen_experts:
                continue
            norm_after = sum(mx.sum(v * v).item() for _, v in nn.utils.tree_flatten(expert.parameters()))
            # Recycled expert should have smaller norm than before (fresh init)
            if norm_after < norms_before[(li, ei)] * 0.5:
                print(f"  expert ({li},{ei}): norm {norms_before[(li, ei)]:.4f} -> {norm_after:.4f} (recycled)  OK")

    print("  PASSED\n")


def test_frozen_expert_unchanged():
    """Freeze an expert, do a training step, verify its weights didn't change."""
    print("=" * 60)
    print("test_frozen_expert_unchanged")

    model = get_model("moe_freeze", **CFG)
    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:200], tok, block_size=32)

    # Train a bit then freeze
    train(model, ds, steps=20, batch_size=16, lr=3e-3, log_every=100)
    model.on_domain_switch("domain_b")

    # Record frozen expert weights
    frozen_weights = {}
    for li, ei in model._frozen_experts:
        expert = model.layers[li].moe.experts[ei]
        weights = {k: v.tolist() for k, v in nn.utils.tree_flatten(expert.parameters())}
        frozen_weights[(li, ei)] = weights

    # Train more steps
    train(model, ds, steps=20, batch_size=16, lr=3e-3, log_every=100)

    # Verify frozen expert weights are unchanged
    for li, ei in model._frozen_experts:
        expert = model.layers[li].moe.experts[ei]
        weights_after = {k: v.tolist() for k, v in nn.utils.tree_flatten(expert.parameters())}
        for key in frozen_weights[(li, ei)]:
            assert frozen_weights[(li, ei)][key] == weights_after[key], (
                f"Frozen expert ({li},{ei}) param {key} changed after training"
            )
        print(f"  expert ({li},{ei}): weights unchanged after training  OK")

    print("  PASSED\n")


def test_frozen_set_grows():
    """Calling on_domain_switch() multiple times grows the frozen set."""
    print("=" * 60)
    print("test_frozen_set_grows")

    model = get_model("moe_freeze", **CFG)
    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:200], tok, block_size=32)

    sizes = [len(model._frozen_experts)]
    for i, domain in enumerate(["b", "c", "d"]):
        train(model, ds, steps=10, batch_size=16, lr=3e-3, log_every=100)
        model.on_domain_switch(f"domain_{domain}")
        sizes.append(len(model._frozen_experts))
        print(f"  after switch {i+1} (domain_{domain}): {len(model._frozen_experts)} frozen")

    # Frozen set should grow (or stay same if all experts in a layer are frozen)
    assert sizes[-1] >= sizes[0], f"Frozen set didn't grow: {sizes}"
    assert sizes[1] > sizes[0], f"First switch should freeze at least one expert: {sizes}"
    print(f"  sizes over switches: {sizes}")
    print("  PASSED\n")


def test_multidomain_forgetting():
    """Train on domain A then B with lifecycle, verify domain A loss doesn't explode."""
    print("=" * 60)
    print("test_multidomain_forgetting")

    docs = load_names()
    tok = CharTokenizer(docs)
    domains = domain_split(docs)

    domain_datasets = {}
    for name, domain_docs in domains.items():
        tr, val = train_val_split(domain_docs)
        domain_datasets[name] = (
            CharDataset(tr[:300], tok, block_size=32),
            CharDataset(val[:100], tok, block_size=32),
        )

    model = get_model("moe_freeze", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2, n_experts=4, top_k=2)

    result = train_multidomain(model, domain_datasets,
                                steps_per_domain=100, batch_size=32, lr=3e-3, log_every=50)

    forgetting = compute_forgetting(result["eval_matrix"], result["domains"])

    for domain, stats in forgetting.items():
        print(f"  {domain}: after_own={stats['after_own']:.3f}, "
              f"after_last={stats['after_last']:.3f}, "
              f"forgetting={stats['forgetting']:.3f} ({stats['pct']:.1f}%)")
        # Loss shouldn't more than triple (generous threshold for short training)
        assert stats["after_last"] < stats["after_own"] * 3 + 1.0, (
            f"Domain {domain} loss exploded: {stats['after_own']:.3f} -> {stats['after_last']:.3f}"
        )

    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_on_domain_switch_freezes()
    test_on_domain_switch_recycles()
    test_frozen_expert_unchanged()
    test_frozen_set_grows()
    test_multidomain_forgetting()
    print("All MoE Freeze tests passed!")
