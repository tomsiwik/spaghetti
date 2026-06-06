#!/usr/bin/env python3
"""Tests for Answer-Conditioned Scoring experiment."""

import random
import numpy as onp
import pytest


def test_data_generators():
    """Test that all domain generators produce valid data."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import DOMAIN_GENERATORS
    rng = random.Random(42)
    for domain, gen in DOMAIN_GENERATORS.items():
        data = gen(10, rng)
        assert len(data) == 10, f"{domain}: expected 10, got {len(data)}"
        for s in data:
            assert isinstance(s, str) and len(s) > 0


def test_tokenizer():
    """Test character tokenizer encode/decode roundtrip."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import CharTokenizer
    tok = CharTokenizer()
    for s in ["2+3=5", "abc>cba", "ab*3=ababab", "bca>abc", "101>odd"]:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        assert decoded == s, f"Roundtrip failed: {s!r} -> {ids} -> {decoded!r}"


def test_model_forward():
    """Test model forward pass produces correct shape."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
        init_model, forward, CharTokenizer
    )
    tok = CharTokenizer()
    params = init_model(tok.vocab_size, d=32, H=2, L=2, max_T=32, seed=42)
    x = onp.array([[2, 3, 4, 5]], dtype=onp.int32)
    logits = forward(params, x, tok.pad_id)
    assert logits.shape == (1, 4, tok.vocab_size)


def test_loss_decreases():
    """Test that training reduces loss."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
        init_model, forward, compute_loss, CharTokenizer, _prepare_batch, train_model,
        _make_parity_data
    )
    tok = CharTokenizer()
    params = init_model(tok.vocab_size, d=32, H=2, L=2, max_T=32, seed=42)

    data = _make_parity_data(100, random.Random(42))
    data_enc = [tok.encode(s) for s in data]

    # Compute initial loss
    inp, tgt, mask = _prepare_batch(data_enc[:16], tok.pad_id, max_len=32)
    param_vals = [params[k] for k in sorted(params.keys()) if k != '_config']
    loss_before = float(compute_loss(params, inp, tgt, mask, tok.pad_id))

    # Train briefly
    params = train_model(params, data_enc, tok.pad_id, epochs=3, lr=0.001,
                         batch_size=16, verbose=False)

    inp, tgt, mask = _prepare_batch(data_enc[:16], tok.pad_id, max_len=32)
    loss_after = float(compute_loss(params, inp, tgt, mask, tok.pad_id))

    assert loss_after < loss_before, f"Loss should decrease: {loss_before:.4f} -> {loss_after:.4f}"


def test_answer_only_ppl_differs():
    """Test that answer-only PPL differs from full-sequence PPL."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
        init_model, CharTokenizer, compute_full_sequence_ppl, compute_answer_only_ppl
    )
    tok = CharTokenizer()
    params = init_model(tok.vocab_size, d=32, H=2, L=2, max_T=32, seed=42)

    data_str = ["abc>cba", "xyz>zyx", "ab>ba"]
    data_enc = [tok.encode(s) for s in data_str]

    full_ppl = compute_full_sequence_ppl(params, data_enc, tok.pad_id)
    answer_ppl = compute_answer_only_ppl(params, data_str, data_enc, ">", tok.pad_id)

    assert full_ppl != answer_ppl, f"Should differ: full={full_ppl}, answer={answer_ppl}"
    assert 1.0 < full_ppl < 10000.0
    assert 1.0 < answer_ppl < 10000.0


def test_answer_mask_positions():
    """Test that answer-only PPL uses correct token positions."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
        init_model, CharTokenizer, compute_per_token_loss_np
    )
    tok = CharTokenizer()
    params = init_model(tok.vocab_size, d=32, H=2, L=2, max_T=32, seed=42)

    # "abc>cba" encoded: [a, b, c, >, c, b, a, <EOS>]
    # inp: [a, b, c, >, c, b, a]
    # tgt: [b, c, >, c, b, a, <EOS>]
    # delimiter ">" at string pos 3
    # answer tgt starts at index 3: tgt[3]="c" (first answer char)
    s = "abc>cba"
    enc = tok.encode(s)
    losses, mask = compute_per_token_loss_np(params, enc, tok.pad_id)

    assert len(losses) == len(enc) - 1  # T-1 positions
    delim_pos = s.rfind(">")
    assert delim_pos == 3

    # Answer tokens are at indices 3, 4, 5, 6 in tgt
    # (corresponding to chars c, b, a, <EOS>)
    assert mask[3] > 0  # 'c' after delimiter
    assert mask[4] > 0  # 'b'
    assert mask[5] > 0  # 'a'
    assert mask[6] > 0  # <EOS>


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
