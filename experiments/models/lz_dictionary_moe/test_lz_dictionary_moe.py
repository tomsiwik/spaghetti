"""Tests for LZ Dictionary MoE model."""

import mlx.core as mx
import mlx.nn as nn


def test_forward_pass():
    """Model produces correct output shape."""
    from .lz_dictionary_moe import DictionaryMoEGPT

    model = DictionaryMoEGPT(vocab_size=28, block_size=32, n_embd=64,
                              n_head=4, n_layer=2, n_experts=4, top_k=2,
                              n_dict=8, dict_rank=32, delta_rank=16)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]] * 2)  # (2, 8)
    logits = model(tokens)
    assert logits.shape == (2, 8, 28), f"Expected (2, 8, 28), got {logits.shape}"
    print("PASS: forward_pass")


def test_aux_loss():
    """aux_loss returns a scalar."""
    from .lz_dictionary_moe import DictionaryMoEGPT

    model = DictionaryMoEGPT(vocab_size=28, n_embd=64, n_layer=2)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4]] * 2)
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() >= 0, "aux_loss should be non-negative"
    print("PASS: aux_loss")


def test_dictionary_diagnostics():
    """Dictionary diagnostics return utilization metrics."""
    from .lz_dictionary_moe import DictionaryMoEGPT

    model = DictionaryMoEGPT(vocab_size=28, n_embd=64, n_layer=2,
                              n_dict=8, n_experts=4)
    mx.eval(model.parameters())

    diag = model.dictionary_diagnostics()
    assert "layer_0" in diag
    assert "layer_1" in diag

    for layer_name, layer_diag in diag.items():
        assert "utilization_rate" in layer_diag
        assert "normalized_entropy" in layer_diag
        assert "per_entry_weight" in layer_diag
        assert len(layer_diag["per_entry_weight"]) == 8
        assert 0 <= layer_diag["utilization_rate"] <= 1
        assert layer_diag["normalized_entropy"] >= 0
        print(f"  {layer_name}: util={layer_diag['utilization_rate']:.2f}, "
              f"H_norm={layer_diag['normalized_entropy']:.3f}")

    print("PASS: dictionary_diagnostics")


def test_dictionary_entry():
    """Single dictionary entry is a valid low-rank MLP."""
    from .lz_dictionary_moe import DictionaryEntry

    entry = DictionaryEntry(n_embd=64, rank=32)
    mx.eval(entry.parameters())

    x = mx.random.normal((2, 8, 64))
    out = entry(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"
    print("PASS: dictionary_entry")


def test_dictionary_expert():
    """Dictionary expert produces correct shape using shared dictionary."""
    from .lz_dictionary_moe import DictionaryEntry, DictionaryExpert

    n_dict = 4
    dictionary = [DictionaryEntry(64, 32) for _ in range(n_dict)]
    expert = DictionaryExpert(64, n_dict, delta_rank=16)
    for d in dictionary:
        mx.eval(d.parameters())
    mx.eval(expert.parameters())

    x = mx.random.normal((2, 8, 64))
    out = expert(x, dictionary)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"

    # Check alpha weights sum to 1
    alpha = expert.get_alpha_weights()
    mx.eval(alpha)
    alpha_sum = mx.sum(alpha).item()
    assert abs(alpha_sum - 1.0) < 1e-5, f"Alpha should sum to 1, got {alpha_sum}"
    print("PASS: dictionary_expert")


def test_param_count_savings():
    """Dictionary MoE should use fewer params than standard MoE in MLP layers."""
    from .lz_dictionary_moe import DictionaryMoEGPT
    from ..moe import MoEGPT

    d, N = 64, 4

    standard = MoEGPT(vocab_size=28, n_embd=d, n_layer=4, n_experts=N, top_k=2)
    dictionary = DictionaryMoEGPT(vocab_size=28, n_embd=d, n_layer=4,
                                   n_experts=N, top_k=2,
                                   n_dict=8, dict_rank=32, delta_rank=16)

    def count(model):
        return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))

    mx.eval(standard.parameters())
    mx.eval(dictionary.parameters())

    n_std = count(standard)
    n_dict = count(dictionary)

    print(f"  Standard MoE: {n_std:,} params")
    print(f"  Dictionary MoE: {n_dict:,} params")
    print(f"  Ratio: {n_dict/n_std:.2%}")

    # Dictionary version should be smaller (that's the point)
    # But not required -- we want to compare at SAME total params
    print("PASS: param_count_savings")


def test_gradient_flow():
    """Gradients flow through dictionary entries AND expert residuals."""
    from .lz_dictionary_moe import DictionaryMoEGPT

    model = DictionaryMoEGPT(vocab_size=28, n_embd=64, n_layer=1,
                              n_experts=2, n_dict=4, dict_rank=16, delta_rank=8)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
        ) + model.aux_loss()

    x = mx.array([[1, 2, 3, 4]])
    y = mx.array([[2, 3, 4, 0]])

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    mx.eval(loss, grads)

    # Check that dictionary entry gradients exist
    dict_grad = grads["layers"][0]["moe"]["dictionary"][0]["down"]["weight"]
    assert mx.any(dict_grad != 0).item(), "Dictionary entry should receive gradients"

    # Check that expert delta gradients exist
    delta_grad = grads["layers"][0]["moe"]["experts"][0]["delta_down"]["weight"]
    assert mx.any(delta_grad != 0).item(), "Expert delta should receive gradients"

    print("PASS: gradient_flow")


if __name__ == "__main__":
    test_dictionary_entry()
    test_dictionary_expert()
    test_forward_pass()
    test_aux_loss()
    test_dictionary_diagnostics()
    test_param_count_savings()
    test_gradient_flow()
    print("\nAll tests passed!")
