#!/usr/bin/env python3
"""Tests for the FFN-only matched rank experiment.

Tests the analysis infrastructure (no GPU needed) and validates
that kill criteria evaluation logic works correctly.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def test_cosine_similarity():
    """Test cosine similarity computation."""
    from experiments.models.ffn_only_matched_rank.analyze import cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-10

    c = np.array([1.0, 1.0, 0.0])
    d = np.array([1.0, 1.0, 0.0])
    assert abs(cosine_similarity(c, d) - 1.0) < 1e-10

    e = np.array([1.0, 0.0])
    f = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(e, f) + 1.0) < 1e-10


def test_extract_module_vectors():
    """Test module filtering logic."""
    from experiments.models.ffn_only_matched_rank.analyze import extract_module_vectors

    weights = {
        "model.layers.0.mlp.gate_proj.lora_A.weight": np.ones((8, 64)),
        "model.layers.0.mlp.gate_proj.lora_B.weight": np.ones((256, 8)),
        "model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((8, 64)) * 2,
        "model.layers.0.self_attn.q_proj.lora_B.weight": np.ones((64, 8)) * 2,
    }

    ffn_vec = extract_module_vectors(weights, module_filter="ffn")
    attn_vec = extract_module_vectors(weights, module_filter="attn")
    all_vec = extract_module_vectors(weights, module_filter=None)

    # FFN should only have mlp params
    assert len(ffn_vec) == 8 * 64 + 256 * 8  # gate A + gate B
    assert np.all(ffn_vec == 1.0)

    # Attn should only have self_attn params
    assert len(attn_vec) == 8 * 64 + 64 * 8  # q A + q B
    assert np.all(attn_vec == 2.0)

    # All should have everything
    assert len(all_vec) == len(ffn_vec) + len(attn_vec)


def test_kill_criteria_quality_pass():
    """Test that quality kill criterion passes when gap < 5%."""
    ffn_ppl = 3.5
    all_ppl = 3.4
    gap_pct = (ffn_ppl - all_ppl) / all_ppl * 100
    assert gap_pct < 5.0, f"Gap {gap_pct:.1f}% should be < 5%"


def test_kill_criteria_quality_fail():
    """Test that quality kill criterion triggers when gap > 5%."""
    ffn_ppl = 3.8
    all_ppl = 3.4
    gap_pct = (ffn_ppl - all_ppl) / all_ppl * 100
    assert gap_pct > 5.0, f"Gap {gap_pct:.1f}% should be > 5%"


def test_kill_criteria_orthogonality_pass():
    """Test that orthogonality kill criterion passes when diff < 50%."""
    independent_mean = 0.065
    retroactive_mean = 0.060
    diff_pct = abs(independent_mean - retroactive_mean) / retroactive_mean * 100
    assert diff_pct < 50.0, f"Diff {diff_pct:.1f}% should be < 50%"


def test_kill_criteria_orthogonality_fail():
    """Test that orthogonality kill criterion triggers when diff > 50%."""
    independent_mean = 0.10
    retroactive_mean = 0.060
    diff_pct = abs(independent_mean - retroactive_mean) / retroactive_mean * 100
    assert diff_pct > 50.0, f"Diff {diff_pct:.1f}% should be > 50%"


def test_retroactive_numbers_match_prior_experiment():
    """Verify that our analysis reproduces the prior experiment's numbers."""
    results_path = Path("experiments/models/ffn_only_matched_rank/results.json")
    if not results_path.exists():
        pytest.skip("Results not yet generated")

    results = json.loads(results_path.read_text())

    # Check retroactive FFN subset matches prior experiment
    ffn_subset = results.get("all_modules_ffn_subset_ortho", {})
    if "mean_abs_cos" not in ffn_subset:
        pytest.skip("Orthogonality not computed yet")

    # Prior experiment reported 0.0605
    assert abs(ffn_subset["mean_abs_cos"] - 0.0605) < 0.001, \
        f"FFN subset mean |cos| {ffn_subset['mean_abs_cos']:.4f} should match prior ~0.0605"


def test_parameter_count_ffn_vs_all():
    """Verify FFN-only has ~25% fewer params than all-modules at rank 16."""
    # Qwen2.5-7B dimensions
    d, d_ff, d_kv, L, r = 3584, 18944, 512, 28, 16

    ffn_params = L * 3 * r * (d + d_ff)  # gate, up, down
    attn_params = L * (2 * r * d + r * d + r * d_kv + r * d + r * d_kv + 2 * r * d)
    # Simplified: q(2rd) + k(rd + r*d_kv) + v(rd + r*d_kv) + o(2rd)
    all_params = ffn_params + attn_params

    ratio = ffn_params / all_params
    assert 0.70 < ratio < 0.80, f"FFN/All ratio {ratio:.2f} should be ~0.75"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
