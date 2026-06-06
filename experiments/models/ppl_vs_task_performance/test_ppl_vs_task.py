#!/usr/bin/env python3
"""Tests for PPL vs Task Performance experiment."""

import random
import pytest


def test_data_generators():
    """Test that all domain generators produce valid data."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import DOMAIN_GENERATORS

    rng = random.Random(42)
    for domain, gen in DOMAIN_GENERATORS.items():
        data = gen(10, rng)
        assert len(data) == 10, f"{domain}: expected 10, got {len(data)}"
        for s in data:
            assert isinstance(s, str) and len(s) > 0, f"{domain}: invalid entry {s!r}"


def test_tokenizer():
    """Test character tokenizer encode/decode roundtrip."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import CharTokenizer

    tok = CharTokenizer()
    test_strings = ["2+3=5", "abc>cba", "ab*3=ababab", "bca>abc", "101>odd"]

    for s in test_strings:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        assert decoded == s, f"Roundtrip failed: {s!r} -> {ids} -> {decoded!r}"


def test_arithmetic_correctness():
    """Test that arithmetic data is actually correct."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import _make_arithmetic_data

    data = _make_arithmetic_data(100, random.Random(42))
    for entry in data:
        parts = entry.split("=")
        assert len(parts) == 2
        ab = parts[0].split("+")
        assert len(ab) == 2
        a, b = int(ab[0]), int(ab[1])
        c = int(parts[1])
        assert a + b == c, f"Wrong: {entry}"


def test_parity_correctness():
    """Test that parity data is actually correct."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import _make_parity_data

    data = _make_parity_data(100, random.Random(42))
    for entry in data:
        bits, parity = entry.split(">")
        count = bits.count("1")
        expected = "even" if count % 2 == 0 else "odd"
        assert parity == expected, f"Wrong: {entry}"


def test_model_builds():
    """Test that the model builds and does a forward pass."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import (
        _build_model, CharTokenizer
    )
    import torch

    tok = CharTokenizer()
    model = _build_model(tok.vocab_size, d_model=32, n_heads=2, n_layers=2,
                         max_seq_len=32, device="cpu")
    x = torch.randint(0, tok.vocab_size, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, tok.vocab_size)


def test_lora_apply_and_restore():
    """Test LoRA delta application and base restoration."""
    from experiments.models.ppl_vs_task_performance.ppl_vs_task import (
        _build_model, CharTokenizer, apply_expert_deltas, restore_base
    )
    import torch

    tok = CharTokenizer()
    model = _build_model(tok.vocab_size, d_model=32, n_heads=2, n_layers=2,
                         max_seq_len=32, device="cpu")

    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Create fake deltas
    deltas = {}
    for name, param in model.named_parameters():
        if "fc1.weight" in name or "fc2.weight" in name:
            deltas[name] = torch.randn_like(param) * 0.01

    # Apply
    apply_expert_deltas(model, deltas, base_state)
    for name in deltas:
        assert not torch.allclose(model.state_dict()[name], base_state[name])

    # Restore
    restore_base(model, base_state)
    for name in deltas:
        assert torch.allclose(model.state_dict()[name], base_state[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
