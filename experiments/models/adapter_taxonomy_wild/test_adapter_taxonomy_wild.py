"""Tests for adapter taxonomy analysis."""

import json
import os
import pytest

from experiments.models.adapter_taxonomy_wild.adapter_taxonomy_wild import (
    build_adapter_taxonomy,
    score_adapter,
    compute_capacity_bounds,
    evaluate_kill_criteria,
    run_analysis,
)


def test_taxonomy_completeness():
    """All 15 adapter types specified in hypothesis are present."""
    taxonomy = build_adapter_taxonomy()
    required = {
        "lora", "qlora", "dora", "rslora",  # (a) low-rank additive
        "ia3",  # (b) rescaling
        "houlsby",  # (c) bottleneck
        "prefix_tuning", "prompt_tuning",  # (d) virtual token
        "bitfit",  # (e) bias
        "full_rank",  # (f) full-rank
        "molora",  # (g) MoE-native
        "lora_xs", "vera", "tied_lora",  # (h) compressed
        "relora", "lte",  # bonus: base-free methods
    }
    assert required.issubset(set(taxonomy.keys())), (
        f"Missing: {required - set(taxonomy.keys())}"
    )


def test_lora_is_additively_composable():
    """LoRA must compose additively -- this is our core architecture property."""
    taxonomy = build_adapter_taxonomy()
    for name in ["lora", "qlora", "rslora"]:
        assert taxonomy[name].composes_additively is True
        assert taxonomy[name].composition_mode == "additive"


def test_ia3_is_not_additively_composable():
    """IA3 is multiplicative -- incompatible with our additive protocol."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["ia3"].composes_additively is False
    assert taxonomy["ia3"].composition_mode == "multiplicative"


def test_dora_has_composition_caveats():
    """DoRA's magnitude scaling breaks naive additive composition."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["dora"].composes_additively is False


def test_houlsby_is_sequential():
    """Houlsby adapters use sequential composition (nonlinear)."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["houlsby"].composition_mode == "sequential"
    assert taxonomy["houlsby"].can_merge_into_base is False


def test_prefix_tuning_is_concatenative():
    """Prefix tuning composes via concatenation, not addition."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["prefix_tuning"].composition_mode == "concatenative"
    assert taxonomy["prompt_tuning"].composition_mode == "concatenative"


def test_relora_is_base_free():
    """ReLoRA can train from scratch without a frozen base."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["relora"].requires_frozen_base is False
    assert taxonomy["relora"].can_encode_base_knowledge is True


def test_lte_is_base_free():
    """LTE can train from scratch without a frozen base."""
    taxonomy = build_adapter_taxonomy()
    assert taxonomy["lte"].requires_frozen_base is False
    assert taxonomy["lte"].can_encode_base_knowledge is True


def test_kill_criteria_both_survive():
    """Both kill criteria should be disproven (hypothesis survives)."""
    taxonomy = build_adapter_taxonomy()
    results = evaluate_kill_criteria(taxonomy)
    assert results["kill_1_base_knowledge"]["verdict"] == "SURVIVES"
    assert results["kill_2_frozen_base"]["verdict"] == "SURVIVES"


def test_capacity_ordering():
    """LoRA-XS < LoRA < full_rank in parameter count."""
    capacity = compute_capacity_bounds()
    assert capacity["lora_xs"]["total_params"] < capacity["lora"]["total_params"]
    assert capacity["lora"]["total_params"] < capacity["full_rank"]["total_params"]


def test_relora_achieves_full_rank():
    """ReLoRA achieves full rank via accumulation."""
    capacity = compute_capacity_bounds()
    assert capacity["relora"]["rank_fraction"] == 1.0


def test_composability_fit_ordering():
    """LoRA should score highest on composability fit for our architecture."""
    taxonomy = build_adapter_taxonomy()
    scores = {name: score_adapter(a) for name, a in taxonomy.items()}

    lora_fit = scores["lora"].composability_fit
    houlsby_fit = scores["houlsby"].composability_fit
    ia3_fit = scores["ia3"].composability_fit

    assert lora_fit > houlsby_fit, "LoRA should score higher than Houlsby"
    assert lora_fit > ia3_fit, "LoRA should score higher than IA3"


def test_run_analysis_produces_results():
    """Full analysis should complete and produce results.json."""
    results = run_analysis()
    assert "taxonomy" in results
    assert "scores" in results
    assert "kill_criteria" in results
    assert "summary" in results
    assert results["summary"]["kill_criterion_1_survives"] is True
    assert results["summary"]["kill_criterion_2_survives"] is True

    # Check results file was written
    results_path = os.path.join(
        os.path.dirname(__file__), "results.json"
    )
    assert os.path.exists(results_path)
    with open(results_path) as f:
        saved = json.load(f)
    assert saved["summary"]["total_types_surveyed"] >= 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
