"""Tests for delta rank scaling experiment (v2 revised)."""

import math
import torch
import pytest
from .delta_rank_scaling import (
    GPT, CharTokenizer, CharDataset, RMSNorm, MLP, CausalSelfAttention,
    effective_rank, rank_at_threshold, singular_value_spectrum,
    analyze_deltas, fit_power_law, train_gpt, evaluate_model, load_names,
    train_with_checkpoints, _fit_log_log,
)


# ── Unit tests for effective rank ────────────────────────────────────────

def test_effective_rank_identity():
    """Identity matrix should have effective rank = n."""
    m = torch.eye(10)
    r = effective_rank(m)
    assert abs(r - 10.0) < 0.01, f"Identity rank should be 10, got {r}"


def test_effective_rank_rank1():
    """Rank-1 matrix should have effective rank = 1."""
    m = torch.randn(10, 1) @ torch.randn(1, 10)
    r = effective_rank(m)
    assert abs(r - 1.0) < 0.1, f"Rank-1 matrix should have r_eff ~1, got {r}"


def test_effective_rank_1d_returns_zero():
    """1D tensors should return 0."""
    assert effective_rank(torch.randn(10)) == 0.0


def test_effective_rank_zero_matrix():
    """Zero matrix should return 0."""
    assert effective_rank(torch.zeros(5, 5)) == 0.0


def test_effective_rank_random_matrix():
    """Random Gaussian matrix should have effective rank close to min(m, n)."""
    torch.manual_seed(42)
    m = torch.randn(50, 30)
    r = effective_rank(m)
    # Random matrices have near-maximal rank
    assert r > 20, f"Random matrix should have high effective rank, got {r}"


# ── Unit tests for rank_at_threshold ─────────────────────────────────────

def test_rank_at_threshold_identity():
    """Identity matrix needs all dims for 99% energy."""
    m = torch.eye(10)
    r = rank_at_threshold(m, 0.99)
    assert r == 10


def test_rank_at_threshold_rank1():
    """Rank-1 matrix needs 1 dim for any threshold."""
    m = torch.randn(10, 1) @ torch.randn(1, 10)
    r = rank_at_threshold(m, 0.99)
    assert r == 1


def test_rank_at_threshold_1d():
    """1D tensor returns 0."""
    assert rank_at_threshold(torch.randn(5), 0.99) == 0


def test_rank_at_threshold_monotonic():
    """Higher threshold needs more rank."""
    torch.manual_seed(42)
    m = torch.randn(20, 20)
    r90 = rank_at_threshold(m, 0.90)
    r95 = rank_at_threshold(m, 0.95)
    r99 = rank_at_threshold(m, 0.99)
    assert r90 <= r95 <= r99


# ── Unit tests for singular_value_spectrum ───────────────────────────────

def test_spectrum_normalized():
    """First singular value should be 1.0 (normalized)."""
    torch.manual_seed(42)
    m = torch.randn(10, 10)
    s = singular_value_spectrum(m)
    assert abs(s[0] - 1.0) < 1e-6


def test_spectrum_sorted_descending():
    """Spectrum should be descending."""
    torch.manual_seed(42)
    m = torch.randn(10, 10)
    s = singular_value_spectrum(m)
    for i in range(len(s) - 1):
        assert s[i] >= s[i + 1]


def test_spectrum_1d_empty():
    """1D tensor returns empty."""
    assert singular_value_spectrum(torch.randn(5)) == []


# ── Model tests ──────────────────────────────────────────────────────────

def test_gpt_forward():
    """GPT forward pass produces correct output shape."""
    model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
    x = torch.randint(0, 28, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 28)


def test_gpt_different_dims():
    """GPT works at different embedding dimensions."""
    for d in [32, 64, 128]:
        model = GPT(vocab_size=28, block_size=16, n_embd=d, n_head=max(1, d // 16), n_layer=2)
        x = torch.randint(0, 28, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 28)


# ── Delta analysis tests ────────────────────────────────────────────────

def test_analyze_deltas_basic():
    """analyze_deltas produces analysis for each 2D weight."""
    torch.manual_seed(42)
    m1 = GPT(vocab_size=28, n_embd=32, n_head=4, n_layer=2)
    s1 = {k: v.clone() for k, v in m1.state_dict().items()}
    # Perturb slightly (simulate training)
    s2 = {k: v + 0.01 * torch.randn_like(v) if v.dim() == 2 and torch.isfinite(v).all()
           else v.clone() for k, v in s1.items()}
    analyses = analyze_deltas(s2, s1)
    assert len(analyses) > 0
    for a in analyses:
        assert a.ratio >= 0
        assert a.ratio <= 1.0 + 1e-6  # ratio can't exceed 1


def test_analyze_deltas_trained_has_structure():
    """Trained deltas should have lower effective rank ratio than random deltas."""
    torch.manual_seed(42)
    docs = ["alice", "bob", "carol", "dave", "eve"] * 100
    tok = CharTokenizer(docs)
    ds = CharDataset(docs, tok, block_size=16)

    model = GPT(tok.vocab_size, 16, 32, 4, 2)
    skeleton = {k: v.clone() for k, v in model.state_dict().items()}
    train_gpt(model, ds, steps=50, batch_size=8, seed=42)
    trained = {k: v.clone() for k, v in model.state_dict().items()}

    analyses = analyze_deltas(trained, skeleton)
    ratios = [a.ratio for a in analyses]
    mean_ratio = sum(ratios) / len(ratios)

    # Trained deltas should have some structure (ratio < 1.0)
    # Random perturbation would give ratio close to 1.0
    assert mean_ratio < 0.95, f"Trained delta should have structured rank, got ratio={mean_ratio}"


# ── Power law fit tests ──────────────────────────────────────────────────

def test_fit_power_law_perfect():
    """Perfect power law data should give R^2 = 1."""
    dims = [64, 128, 256]
    # ratio = 2.0 * d^(-0.5)
    ratios = [2.0 * d ** (-0.5) for d in dims]
    fit = fit_power_law(dims, ratios)
    assert abs(fit["r_squared"] - 1.0) < 0.01
    assert abs(fit["b"] - (-0.5)) < 0.01


def test_fit_power_law_extrapolation():
    """Extrapolations should be present."""
    fit = fit_power_law([64, 128, 256], [0.6, 0.5, 0.4])
    assert "extrapolations" in fit
    assert "4096" in fit["extrapolations"]
    assert "predicted_ratio" in fit["extrapolations"]["4096"]


def test_fit_power_law_bootstrap_ci():
    """Bootstrap CI should produce intervals containing point estimate."""
    dims = [64, 128, 256]
    ratios = [0.66, 0.63, 0.54]
    per_seed = {"64": [0.60, 0.66, 0.72], "128": [0.57, 0.63, 0.69], "256": [0.48, 0.54, 0.60]}
    fit = fit_power_law(dims, ratios, per_seed_ratios=per_seed, n_bootstrap=1000)
    assert "b_ci_95" in fit
    assert fit["b_ci_95"][0] <= fit["b"] <= fit["b_ci_95"][1]
    # CI should be nonzero
    ci_width = fit["b_ci_95"][1] - fit["b_ci_95"][0]
    assert ci_width > 0.0, f"CI has zero width"
    # Extrapolations should have CI
    assert "ratio_ci_95" in fit["extrapolations"]["4096"]


def test_fit_log_log():
    """Internal log-log fit helper should work."""
    import numpy as np
    log_d = np.log([64.0, 128.0, 256.0])
    log_r = np.log([0.25, 0.177, 0.125])  # ratio = 2 * d^{-0.5}
    a, b, r2 = _fit_log_log(log_d, log_r)
    assert abs(b - (-0.5)) < 0.01
    assert r2 > 0.99


# ── Integration test ─────────────────────────────────────────────────────

def test_mini_experiment_fixed_steps():
    """Run a tiny version without convergence control."""
    from .delta_rank_scaling import run_experiment
    results = run_experiment(
        dimensions=[32, 64],
        n_layer=2,
        pretrain_steps=50,
        batch_size=8,
        seed=42,
        convergence_control=False,
    )
    assert len(results.dimension_results) == 2
    for dr in results.dimension_results:
        assert 0 < dr["mean_ratio"] < 1.0
        assert 0 < dr["ffn_attn_mean_ratio"] < 1.0
        assert len(dr["checkpoint_rho"]) > 0  # Multi-checkpoint data present


def test_mini_experiment_convergence():
    """Run a tiny version with convergence control."""
    from .delta_rank_scaling import run_experiment
    results = run_experiment(
        dimensions=[32, 64],
        n_layer=2,
        pretrain_steps=50,
        batch_size=8,
        seed=42,
        convergence_control=True,
    )
    assert len(results.dimension_results) == 2
    for dr in results.dimension_results:
        assert 0 < dr["ffn_attn_mean_ratio"] < 1.0
        assert dr["convergence_controlled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
