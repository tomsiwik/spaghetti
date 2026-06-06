"""Tests for the dense GPT model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


# Small config for fast tests
CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2)


def test_forward_shape():
    """tokens (B, T) -> logits (B, T, V) for various B, T, V."""
    print("=" * 60)
    print("test_forward_shape")

    for B, T, V in [(1, 8, 28), (4, 32, 28), (2, 16, 50)]:
        model = get_model("gpt", **{**CFG, "vocab_size": V, "block_size": T})
        tokens = mx.zeros((B, T), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
        print(f"  B={B}, T={T}, V={V} -> {logits.shape}  OK")

    print("  PASSED\n")


def test_aux_loss_zero():
    """GPT.aux_loss() always returns 0.0."""
    print("=" * 60)
    print("test_aux_loss_zero")

    model = get_model("gpt", **CFG)
    # Before forward
    assert model.aux_loss().item() == 0.0, "aux_loss should be 0 before forward"
    # After forward
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    model(tokens)
    assert model.aux_loss().item() == 0.0, "aux_loss should be 0 after forward"
    print("  PASSED\n")


def test_on_domain_switch_noop():
    """on_domain_switch() doesn't error or change params."""
    print("=" * 60)
    print("test_on_domain_switch_noop")

    model = get_model("gpt", **CFG)
    params_before = {k: v.tolist() for k, v in nn.utils.tree_flatten(model.parameters())}

    model.on_domain_switch("domain_a")
    model.on_domain_switch("domain_b")

    params_after = {k: v.tolist() for k, v in nn.utils.tree_flatten(model.parameters())}
    assert params_before == params_after, "on_domain_switch changed parameters"
    print("  PASSED\n")


def test_causal_masking():
    """Future tokens don't influence past logits."""
    print("=" * 60)
    print("test_causal_masking")

    model = get_model("gpt", **CFG)
    T = 8
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    # Perturb a future token (position 5) and check positions 0-4 unchanged
    tokens_perturbed = mx.array([[1, 2, 3, 4, 5, 20, 7, 8]])
    logits_perturbed = model(tokens_perturbed)
    mx.eval(logits_perturbed)

    for t in range(5):
        diff = mx.max(mx.abs(logits_orig[0, t] - logits_perturbed[0, t])).item()
        assert diff == 0.0, f"Position {t} changed (max diff={diff}) when future token modified"
        print(f"  position {t}: diff={diff}  OK")

    # Position 5 itself should change (different input token)
    diff_at_5 = mx.max(mx.abs(logits_orig[0, 5] - logits_perturbed[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change when its own token changes"
    print(f"  position 5: diff={diff_at_5:.6f} (changed as expected)  OK")
    print("  PASSED\n")


def test_param_count():
    """Parameter count matches formula: 2*V*d + T*d + 12*L*d^2."""
    print("=" * 60)
    print("test_param_count")

    V, T, d, L = 28, 32, 32, 2
    expected = 2 * V * d + T * d + 12 * L * d**2

    model = get_model("gpt", vocab_size=V, block_size=T, n_embd=d, n_head=4, n_layer=L)
    actual = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))

    print(f"  expected: {expected}")
    print(f"  actual:   {actual}")
    assert actual == expected, f"Param count mismatch: {actual} != {expected}"
    print("  PASSED\n")


def test_learns_names():
    """Train on CharDataset for ~200 steps, loss decreases below threshold."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("gpt", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss, f"Loss didn't decrease: {first_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


def test_generates_names():
    """Train, then generate names and check plausibility."""
    print("=" * 60)
    print("test_generates_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("gpt", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2)
    train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    # Generate 10 names
    generated = []
    bos = tok.bos
    for i in range(10):
        tokens = model.generate(mx.array([bos]), max_new=30, temperature=0.8)
        mx.eval(mx.array(tokens))  # force evaluation
        name = tok.decode(tokens).strip()
        generated.append(name)
        print(f"  [{i}] {repr(name)} (len={len(name)})")

    # Check plausibility
    valid_lens = sum(1 for n in generated if 2 <= len(n) <= 15)
    all_alpha = all(c.isalpha() for n in generated for c in n)
    terminated = sum(1 for n in generated if len(tok.encode(n)) < 28)

    print(f"  valid lengths (2-15): {valid_lens}/10")
    print(f"  all alpha chars: {all_alpha}")
    print(f"  terminated early: {terminated}/10")

    assert valid_lens >= 7, f"Only {valid_lens}/10 names have valid length"
    assert all_alpha, "Generated non-alpha characters"
    assert terminated >= 5, f"Only {terminated}/10 terminated (all hit max_new)"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_aux_loss_zero()
    test_on_domain_switch_noop()
    test_causal_masking()
    test_param_count()
    test_learns_names()
    test_generates_names()
    print("All GPT tests passed!")
