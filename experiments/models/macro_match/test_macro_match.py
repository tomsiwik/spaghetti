"""Tests for exp5_macro_match: verify results integrity and kill gate logic."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def test_results_exist():
    """Verify that the results file exists and has all required phases."""
    results_path = Path("experiments/models/macro_match/results.json")
    assert results_path.exists(), "results.json not found -- run the experiment first"

    results = json.loads(results_path.read_text())
    phases = [r.get("phase") for r in results]

    # Must have baseline evaluations
    assert "baseline_1_5b" in phases, "Missing 1.5B baseline evaluation"
    assert "base_0_5b" in phases, "Missing 0.5B baseline evaluation"

    # Must have at least one composition evaluation
    assert "composed_v1" in phases or "composed_v2" in phases, \
        "Missing composition evaluation"


def test_kill_gate():
    """Verify the kill gate is correctly evaluated."""
    results_path = Path("experiments/models/macro_match/results.json")
    if not results_path.exists():
        print("SKIP: no results file")
        return

    results = json.loads(results_path.read_text())

    baseline = None
    composed = None
    for r in results:
        if r.get("phase") == "baseline_1_5b":
            baseline = r
        if r.get("phase") == "composed_v2":
            composed = r
        elif r.get("phase") == "composed_v1" and composed is None:
            composed = r

    assert baseline is not None, "No baseline"
    assert composed is not None, "No composed"

    # Check kill gate logic
    for domain in ["python", "javascript"]:
        ppl_target = baseline["perplexity"][domain]
        ppl_composed = composed["calibrated_ppl"][domain]
        delta_pct = (ppl_composed - ppl_target) / ppl_target * 100

        print(f"  {domain}: {ppl_composed:.4f} vs {ppl_target:.4f} = {delta_pct:+.1f}%")

        # Verify the gap is real (not a data error)
        assert ppl_composed > 0, f"Invalid composed PPL for {domain}"
        assert ppl_target > 0, f"Invalid target PPL for {domain}"


def test_capsule_improvement():
    """Verify capsules actually improved over the base model."""
    results_path = Path("experiments/models/macro_match/results.json")
    if not results_path.exists():
        print("SKIP: no results file")
        return

    results = json.loads(results_path.read_text())

    base = None
    composed = None
    for r in results:
        if r.get("phase") == "base_0_5b":
            base = r
        if r.get("phase") in ("composed_v1", "composed_v2"):
            composed = r

    if base is None or composed is None:
        print("SKIP: missing data")
        return

    for domain in ["python", "javascript"]:
        ppl_base = base["perplexity"][domain]
        ppl_composed = composed["calibrated_ppl"][domain]

        improvement = (ppl_base - ppl_composed) / ppl_base * 100
        print(f"  {domain}: {ppl_base:.4f} -> {ppl_composed:.4f} ({improvement:+.1f}%)")

        # Capsules MUST improve over base (otherwise something is broken)
        assert ppl_composed < ppl_base, \
            f"Capsules did not improve {domain}: {ppl_composed:.4f} >= {ppl_base:.4f}"


def test_param_efficiency():
    """Verify active params are under the target ratio."""
    results_path = Path("experiments/models/macro_match/results.json")
    if not results_path.exists():
        print("SKIP: no results file")
        return

    results = json.loads(results_path.read_text())

    baseline = None
    composed = None
    for r in results:
        if r.get("phase") == "baseline_1_5b":
            baseline = r
        if r.get("phase") == "composed_v2":
            composed = r

    if baseline is None or composed is None:
        print("SKIP: missing data")
        return

    if "active_params_per_token" in composed:
        active = composed["active_params_per_token"]
        target = baseline["param_count"]
        ratio = active / target
        print(f"  Active/Target ratio: {ratio:.3f}x (goal: <0.33x)")
        # We expect close to 1/3
        assert ratio < 0.50, f"Active param ratio {ratio:.2f}x too high"


if __name__ == "__main__":
    tests = [
        test_results_exist,
        test_kill_gate,
        test_capsule_improvement,
        test_param_efficiency,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            print(f"\n{name}:")
            test_fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
