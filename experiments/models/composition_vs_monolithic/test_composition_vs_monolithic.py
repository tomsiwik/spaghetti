#!/usr/bin/env python3
"""Tests for composition_vs_monolithic experiment.

Run with: uv run python experiments/models/composition_vs_monolithic/test_composition_vs_monolithic.py
"""

import numpy as onp


def _load_module():
    """Load the module directly (bypass micro.models __init__ which needs mlx)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'cvm',
        '/Users/tom/Code/tomsiwik/llm/.worktrees/chirpy-lily/experiments/models/composition_vs_monolithic/composition_vs_monolithic.py'
    )
    cvm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvm)
    return cvm


def test_forward_shape():
    cvm = _load_module()
    tok = cvm.CharTokenizer()
    params = cvm.init_model(tok.vocab_size, d=16, H=2, L=1, max_T=16, seed=42)
    inp = onp.array([[1, 2, 3, 4]], dtype=onp.int32)
    logits = cvm.forward(params, inp)
    assert logits.shape == (1, 4, tok.vocab_size), f"Bad shape: {logits.shape}"
    print("PASS: forward shape")


def test_training_reduces_loss():
    cvm = _load_module()
    tok = cvm.CharTokenizer()
    params = cvm.init_model(tok.vocab_size, d=16, H=2, L=1, max_T=16, seed=42)

    rng = onp.random.RandomState(42)
    data = cvm._make_arithmetic_data(50, rng)
    data_enc = [tok.encode(s) for s in data]

    loss_before = cvm.eval_loss(params, data_enc, tok.pad_id)
    params = cvm.train_model(params, data_enc, tok.pad_id, epochs=5,
                              lr=0.001, batch_size=8, verbose=False)
    loss_after = cvm.eval_loss(params, data_enc, tok.pad_id)

    assert loss_after < loss_before, f"Training failed: {loss_after} >= {loss_before}"
    print(f"PASS: training ({loss_before:.3f} -> {loss_after:.3f})")


def test_delta_computation():
    cvm = _load_module()
    tok = cvm.CharTokenizer()
    base = cvm.init_model(tok.vocab_size, d=16, H=2, L=1, max_T=16, seed=42)

    # Create a "trained" model by adding noise
    trained = {}
    for k, v in base.items():
        if k == '_config':
            trained[k] = v
        else:
            trained[k] = v + onp.random.randn(*v.shape) * 0.01

    delta = cvm.compute_delta(base, trained)
    for k in delta:
        expected = trained[k] - base[k]
        assert onp.allclose(delta[k], expected), f"Delta mismatch for {k}"

    print("PASS: delta computation")


def test_svd_truncation():
    cvm = _load_module()
    delta = {
        'W': onp.random.randn(16, 16).astype(onp.float32),
        'b': onp.random.randn(16).astype(onp.float32),
    }

    trunc, sig_ret = cvm.svd_truncate_delta(delta, rank=4)
    assert trunc['W'].shape == delta['W'].shape
    assert trunc['b'].shape == delta['b'].shape
    assert 0 < sig_ret['W'] <= 1.0
    assert sig_ret['b'] == 1.0  # 1D kept as-is

    # Truncated should have rank <= 4
    U, S, Vt = onp.linalg.svd(trunc['W'], full_matrices=False)
    assert onp.sum(S > 1e-6) <= 4

    print("PASS: SVD truncation")


def test_merge_sum_vs_avg():
    cvm = _load_module()
    d1 = {'W': onp.ones((4, 4))}
    d2 = {'W': onp.ones((4, 4)) * 2}

    merged_sum = cvm.merge_deltas([d1, d2], mode='sum')
    assert onp.allclose(merged_sum['W'], 3 * onp.ones((4, 4)))

    merged_avg = cvm.merge_deltas([d1, d2], mode='avg')
    assert onp.allclose(merged_avg['W'], 1.5 * onp.ones((4, 4)))

    print("PASS: merge sum vs avg")


def test_domains_are_distinct():
    cvm = _load_module()
    rng = onp.random.RandomState(42)

    arith = cvm._make_arithmetic_data(100, rng)
    rev = cvm._make_reverse_data(100, rng)

    # Check that data is actually different
    assert arith[0] != rev[0]
    assert '+' in arith[0] or '=' in arith[0]
    assert '>' in rev[0]

    print("PASS: domains are distinct")


def test_param_budget_match():
    """5 x rank-4 has same total params as 1 x rank-20."""
    d_in, d_out = 32, 32
    r_expert = 4
    r_mono = 20
    n_experts = 5

    params_expert = r_expert * (d_in + d_out)
    params_mono = r_mono * (d_in + d_out)
    total_composed = n_experts * params_expert

    assert total_composed == params_mono, \
        f"Budget mismatch: {total_composed} != {params_mono}"

    print("PASS: param budget match")


if __name__ == '__main__':
    test_forward_shape()
    test_training_reduces_loss()
    test_delta_computation()
    test_svd_truncation()
    test_merge_sum_vs_avg()
    test_domains_are_distinct()
    test_param_budget_match()
    print("\nAll tests passed!")
