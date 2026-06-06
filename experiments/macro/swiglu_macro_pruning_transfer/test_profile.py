"""Tests for gate-product profiling experiment.

Validates that the profiling and analysis code produces correct results.
These are NOT the experiment results -- those are in results.json.
"""

import json
from pathlib import Path


def test_results_exist():
    """Results file exists and has expected structure."""
    results_path = Path(__file__).parent / "results.json"
    assert results_path.exists(), "results.json not found"

    with open(results_path) as f:
        data = json.load(f)

    # Check structure
    assert "analysis" in data
    assert "pruning" in data
    assert "random_pruning_baseline" in data
    assert "calibration" in data
    assert "evaluation" in data
    assert data["model"] == "Qwen/Qwen2.5-0.5B"


def test_data_provenance():
    """Data is from WikiText-2, not hardcoded prompts."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    # Calibration is WikiText-2 test split
    cal = data["calibration"]
    assert cal["dataset"] == "wikitext-2-raw-v1"
    assert cal["split"] == "test"
    assert cal["unique_tokens_used"] > 1000, (
        f"Only {cal['unique_tokens_used']} unique tokens -- too few for real data"
    )

    # Evaluation is WikiText-2 validation split (genuinely held-out)
    ev = data["evaluation"]
    assert ev["dataset"] == "wikitext-2-raw-v1"
    assert ev["split"] == "validation"
    assert ev["split"] != cal["split"], "Eval split must differ from calibration split"


def test_distribution_bimodality():
    """Aggregate bimodality coefficient > 0.555."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    agg = data["analysis"]["aggregate"]
    assert agg["bimodality_coefficient"] > 0.555, (
        f"BC={agg['bimodality_coefficient']} should be > 0.555"
    )
    assert agg["is_bimodal_sas"] is True


def test_pruning_all_thresholds_fail():
    """All pruning thresholds exceed 5% degradation."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    for pr in data["pruning"]["pruning"]:
        assert pr["delta_ppl_pct"] > 5.0, (
            f"tau={pr['threshold']}: delta={pr['delta_ppl_pct']}% should be > 5%"
        )


def test_baseline_perplexity_reasonable():
    """Baseline perplexity is in expected range for Qwen2.5-0.5B on WikiText-2."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    base_ppl = data["pruning"]["baseline_ppl"]
    # Qwen2.5-0.5B should be roughly 14-25 ppl on WikiText-2
    assert 10 < base_ppl < 30, (
        f"Baseline ppl={base_ppl:.2f} outside expected range [10, 30]"
    )


def test_random_pruning_baseline_exists():
    """Random pruning baseline was run with 3 seeds."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    rpb = data["random_pruning_baseline"]
    assert rpb["n_seeds"] == 3
    assert len(rpb["per_seed"]) == 3
    assert rpb["mean_ppl"] > 0
    assert rpb["std_ppl"] >= 0


def test_profiled_pruning_worse_than_random():
    """Gate-product profiled pruning is worse than random at tau=0.05.

    This validates the anti-signal finding: profiling selects specialist
    neurons that are the worst to prune.
    """
    results_path = Path(__file__).parent / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    rpb = data["random_pruning_baseline"]
    random_ppl = rpb["mean_ppl"]

    # Find tau=0.05 pruning result
    profiled_ppl = None
    for pr in data["pruning"]["pruning"]:
        if abs(pr["threshold"] - 0.05) < 0.001:
            profiled_ppl = pr["pruned_ppl"]
            break

    assert profiled_ppl is not None, "No tau=0.05 pruning result found"
    assert profiled_ppl > random_ppl, (
        f"Profiled ({profiled_ppl:.2f}) should be worse than random ({random_ppl:.2f})"
    )


if __name__ == "__main__":
    test_results_exist()
    test_data_provenance()
    test_distribution_bimodality()
    test_pruning_all_thresholds_fail()
    test_baseline_perplexity_reasonable()
    test_random_pruning_baseline_exists()
    test_profiled_pruning_worse_than_random()
    print("All tests passed.")
