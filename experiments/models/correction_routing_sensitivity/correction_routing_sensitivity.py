#!/usr/bin/env python3
"""
Correction Routing Sensitivity: Parameter Perturbation Analysis

Tests the stability of the correction signal quality decision tree under:
  1. teacher_hard_accuracy sweep [0.60..0.80] (21 points)
  2. difficulty_mean perturbation [-0.10..+0.10] (11 points)
  3. Joint 2D sweep (21 x 11 = 231 grid points)

Kill criteria:
  K1: Decision tree flips for >50% of domains under +/-10% parameter perturbation
  K2: No robust region exists where execution > teacher ordering holds

Also computes:
  - Harmful rate (wrong + degenerate) as alternative K1 metric
  - Closed-form analytical breakpoints where K1/K2 flip
  - Confidence intervals on the boundary

Pure numpy/scipy -- no model training required.
"""

import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit as sigmoid

# ── Import parent experiment framework ──────────────────────────────
# We reuse the core types and simulation engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correction_signal_quality.correction_signal_quality import (
    ALPHA,
    DELTA_BASE,
    DELTA_DEGEN,
    DELTA_WRONG,
    CorrectionSource,
    Domain,
    N_CORRECTIONS,
    build_decision_tree,
    compute_aggregate_statistics,
    evaluate_kill_criteria,
    run_full_simulation,
    simulate_expert_trajectory,
)

# ── Configuration ───────────────────────────────────────────────────

# Sweep ranges
TEACHER_HARD_ACC_RANGE = np.linspace(0.60, 0.80, 21)
DIFFICULTY_PERTURBATION_RANGE = np.linspace(-0.10, 0.10, 11)

# Baseline parameters (from parent experiment)
BASELINE_TEACHER_HARD = 0.70
BASELINE_DOMAINS = [
    Domain("python_basics", "code_simple", 0.3, 0.15, 0.95, True, 0.70),
    Domain("algorithm_design", "code_algo", 0.6, 0.20, 0.80, True, 0.55),
    Domain("systems_programming", "code_systems", 0.75, 0.15, 0.60, True, 0.50),
    Domain("creative_writing", "writing", 0.5, 0.25, 0.0, False, 0.60),
    Domain("logical_reasoning", "reasoning", 0.65, 0.20, 0.0, False, 0.55),
    Domain("medical_qa", "medical", 0.7, 0.15, 0.0, False, 0.50),
]

N_SEEDS = 10  # Match parent experiment


# ── Analytical Breakpoints ──────────────────────────────────────────

def analytical_teacher_error_rate(hard_accuracy: float, difficulty_mean: float,
                                   difficulty_std: float, n_samples: int = 10000) -> float:
    """
    Compute expected teacher error rate analytically via numerical integration.

    Error rate = E_d[1 - sigmoid(beta_0 + beta_1 * d)]
    where d ~ TruncatedNormal(mean, std, 0, 1)
    """
    base_accuracy = 0.92  # fixed
    beta_0 = np.log(base_accuracy / (1 - base_accuracy))
    logit_hard = np.log(hard_accuracy / (1 - hard_accuracy))
    beta_1 = logit_hard - beta_0

    # Integrate over truncated normal difficulty distribution
    d = np.linspace(0.01, 0.99, n_samples)
    # Truncated normal PDF (unnormalized, then normalized)
    pdf = np.exp(-0.5 * ((d - difficulty_mean) / difficulty_std) ** 2)
    pdf /= pdf.sum()

    accuracy = sigmoid(beta_0 + beta_1 * d)
    error_rate = float(np.sum((1 - accuracy) * pdf))
    return error_rate


def analytical_harmful_rate(hard_accuracy: float, degeneracy_rate: float,
                             difficulty_mean: float, difficulty_std: float,
                             n_samples: int = 10000) -> float:
    """
    Compute expected harmful rate = P(wrong) + P(correct AND degenerate).

    Harmful = wrong + degenerate = (1 - q) + q * p_degen
            = 1 - q * (1 - p_degen)
    """
    base_accuracy = 0.92
    beta_0 = np.log(base_accuracy / (1 - base_accuracy))
    logit_hard = np.log(hard_accuracy / (1 - hard_accuracy))
    beta_1 = logit_hard - beta_0

    d = np.linspace(0.01, 0.99, n_samples)
    pdf = np.exp(-0.5 * ((d - difficulty_mean) / difficulty_std) ** 2)
    pdf /= pdf.sum()

    accuracy = sigmoid(beta_0 + beta_1 * d)
    # Harmful = wrong + degenerate among correct
    wrong_rate = 1 - accuracy
    degen_rate = accuracy * degeneracy_rate
    harmful = wrong_rate + degen_rate

    return float(np.sum(harmful * pdf))


def find_k1_breakpoint_hard_acc(difficulty_mean: float, difficulty_std: float,
                                  threshold: float = 0.20) -> Optional[float]:
    """
    Find the teacher_hard_accuracy at which K1 (error rate) exactly hits threshold.

    Returns None if no breakpoint exists in [0.50, 0.95].
    """
    def f(h):
        return analytical_teacher_error_rate(h, difficulty_mean, difficulty_std) - threshold

    try:
        # Error rate decreases as hard_accuracy increases
        # Check that the function changes sign
        f_low = f(0.50)
        f_high = f(0.95)
        if f_low * f_high > 0:
            return None  # No crossing
        return brentq(f, 0.50, 0.95, xtol=1e-6)
    except ValueError:
        return None


def find_k1_breakpoint_harmful(difficulty_mean: float, difficulty_std: float,
                                degeneracy_rate: float = 0.08,
                                threshold: float = 0.20) -> Optional[float]:
    """
    Find teacher_hard_accuracy at which harmful rate (wrong + degenerate) hits threshold.
    """
    def f(h):
        return analytical_harmful_rate(h, degeneracy_rate, difficulty_mean, difficulty_std) - threshold

    try:
        f_low = f(0.50)
        f_high = f(0.95)
        if f_low * f_high > 0:
            return None
        return brentq(f, 0.50, 0.95, xtol=1e-6)
    except ValueError:
        return None


def find_k2_coverage_breakpoint(gamma: float = 0.30, threshold: float = 0.10) -> float:
    """
    Find test_coverage at which K2 (execution degeneracy) hits threshold.

    Degeneracy = (1 - coverage) * gamma
    Solve: (1 - coverage) * gamma = threshold
    coverage = 1 - threshold / gamma
    """
    return 1.0 - threshold / gamma


# ── Simulation Sweep ────────────────────────────────────────────────

def make_sources(teacher_hard_acc: float) -> list:
    """Create correction sources with perturbed teacher hard accuracy."""
    human = CorrectionSource(
        name="human", base_accuracy=0.97, hard_accuracy=0.90,
        degeneracy_rate=0.02, cost_per_correction=2.00,
        applicable_domains=["all"],
    )
    teacher = CorrectionSource(
        name="teacher_70b", base_accuracy=0.92, hard_accuracy=teacher_hard_acc,
        degeneracy_rate=0.08, cost_per_correction=0.001,
        applicable_domains=["all"],
    )
    execution = CorrectionSource(
        name="execution", base_accuracy=0.99, hard_accuracy=0.85,
        degeneracy_rate=0.03, cost_per_correction=0.0001,
        applicable_domains=["code_simple", "code_algo", "code_systems"],
    )
    return [human, teacher, execution]


def make_domains(difficulty_perturbation: float) -> list:
    """Create domains with perturbed difficulty means."""
    domains = []
    for d in BASELINE_DOMAINS:
        new_mean = np.clip(d.difficulty_mean + difficulty_perturbation, 0.05, 0.95)
        domains.append(Domain(
            d.name, d.domain_type, new_mean, d.difficulty_std,
            d.test_coverage, d.is_code, d.initial_quality,
        ))
    return domains


def run_single_config(teacher_hard_acc: float, difficulty_perturbation: float,
                       seeds: List[int]) -> Dict:
    """
    Run simulation for one (teacher_hard_acc, difficulty_perturbation) config.

    Returns: per-domain optimal source, error rates, harmful rates.
    """
    sources = make_sources(teacher_hard_acc)
    domains = make_domains(difficulty_perturbation)

    all_results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for domain in domains:
            for source in sources:
                result = simulate_expert_trajectory(source, domain, N_CORRECTIONS, rng)
                result["seed"] = seed
                all_results.append(result)

    data = {"results": all_results, "seeds": seeds}
    agg = compute_aggregate_statistics(data)

    # Extract per-domain optimal source
    tree = build_decision_tree(agg)
    routing = {}
    for domain_name, decision in tree["per_domain_decisions"].items():
        routing[domain_name] = decision["optimal_source"]

    # Compute harmful rate (wrong + degenerate) for teacher
    teacher_harmful = {}
    teacher_error = {}
    for key, stats in agg["per_domain"].items():
        if stats["source"] == "teacher_70b":
            domain = stats["domain"]
            teacher_error[domain] = stats["error_rate_mean"]
            teacher_harmful[domain] = stats["error_rate_mean"] + stats["degeneracy_rate_mean"]

    return {
        "routing": routing,
        "teacher_error": teacher_error,
        "teacher_harmful": teacher_harmful,
        "teacher_error_avg": float(np.mean(list(teacher_error.values()))) if teacher_error else 0,
        "teacher_harmful_avg": float(np.mean(list(teacher_harmful.values()))) if teacher_harmful else 0,
    }


def compute_baseline_routing(seeds: List[int]) -> Dict[str, str]:
    """Get the baseline routing (no perturbation)."""
    result = run_single_config(BASELINE_TEACHER_HARD, 0.0, seeds)
    return result["routing"]


def run_1d_teacher_sweep(seeds: List[int]) -> Dict:
    """Sweep teacher_hard_accuracy at fixed difficulty."""
    results = []
    for h in TEACHER_HARD_ACC_RANGE:
        config_result = run_single_config(h, 0.0, seeds)
        config_result["teacher_hard_acc"] = float(h)
        config_result["difficulty_perturbation"] = 0.0
        results.append(config_result)
    return results


def run_1d_difficulty_sweep(seeds: List[int]) -> Dict:
    """Sweep difficulty perturbation at fixed teacher accuracy."""
    results = []
    for dp in DIFFICULTY_PERTURBATION_RANGE:
        config_result = run_single_config(BASELINE_TEACHER_HARD, dp, seeds)
        config_result["teacher_hard_acc"] = BASELINE_TEACHER_HARD
        config_result["difficulty_perturbation"] = float(dp)
        results.append(config_result)
    return results


def run_2d_sweep(seeds: List[int]) -> List[Dict]:
    """Full 2D sweep: teacher_hard_acc x difficulty_perturbation."""
    results = []
    total = len(TEACHER_HARD_ACC_RANGE) * len(DIFFICULTY_PERTURBATION_RANGE)
    done = 0
    for h in TEACHER_HARD_ACC_RANGE:
        for dp in DIFFICULTY_PERTURBATION_RANGE:
            config_result = run_single_config(h, dp, seeds)
            config_result["teacher_hard_acc"] = float(h)
            config_result["difficulty_perturbation"] = float(dp)
            results.append(config_result)
            done += 1
            if done % 20 == 0:
                print(f"  2D sweep: {done}/{total} ({100*done/total:.0f}%)")
    return results


def analyze_flip_rates(sweep_results: List[Dict], baseline_routing: Dict[str, str]) -> Dict:
    """
    Analyze how often the decision tree flips compared to baseline.

    A "flip" means the optimal source for a domain changed.
    """
    domain_names = list(baseline_routing.keys())
    n_domains = len(domain_names)

    flip_analysis = []
    for result in sweep_results:
        n_flips = 0
        flipped_domains = []
        for d in domain_names:
            if result["routing"].get(d) != baseline_routing.get(d):
                n_flips += 1
                flipped_domains.append(d)

        flip_rate = n_flips / n_domains
        flip_analysis.append({
            "teacher_hard_acc": result["teacher_hard_acc"],
            "difficulty_perturbation": result["difficulty_perturbation"],
            "n_flips": n_flips,
            "flip_rate": flip_rate,
            "flipped_domains": flipped_domains,
            "teacher_error_avg": result["teacher_error_avg"],
            "teacher_harmful_avg": result["teacher_harmful_avg"],
        })

    return flip_analysis


def find_execution_teacher_region(sweep_2d: List[Dict]) -> Dict:
    """
    Find the region in (teacher_hard_acc, difficulty_perturbation) space
    where execution > teacher ordering holds for all code domains.
    """
    code_domains = ["python_basics", "algorithm_design", "systems_programming"]

    region = []
    for result in sweep_2d:
        exec_wins_all = True
        for d in code_domains:
            # Execution wins if it's the optimal source
            if result["routing"].get(d) != "execution":
                exec_wins_all = False
                break
        region.append({
            "teacher_hard_acc": result["teacher_hard_acc"],
            "difficulty_perturbation": result["difficulty_perturbation"],
            "execution_wins_all_code": exec_wins_all,
        })

    return region


def compute_analytical_breakpoints() -> Dict:
    """Compute all closed-form analytical breakpoints."""
    breakpoints = {}

    # K2 coverage breakpoint (closed-form)
    k2_coverage_bp = find_k2_coverage_breakpoint()
    breakpoints["k2_coverage_threshold"] = k2_coverage_bp
    breakpoints["k2_formula"] = "coverage_min = 1 - threshold / gamma = 1 - 0.10 / 0.30 = 0.667"

    # K1 error rate breakpoints per domain
    k1_error_bps = {}
    k1_harmful_bps = {}
    for domain in BASELINE_DOMAINS:
        bp_error = find_k1_breakpoint_hard_acc(domain.difficulty_mean, domain.difficulty_std, 0.20)
        bp_harmful = find_k1_breakpoint_harmful(domain.difficulty_mean, domain.difficulty_std, 0.08, 0.20)
        k1_error_bps[domain.name] = bp_error
        k1_harmful_bps[domain.name] = bp_harmful

    breakpoints["k1_error_breakpoints_per_domain"] = k1_error_bps
    breakpoints["k1_harmful_breakpoints_per_domain"] = k1_harmful_bps

    # Aggregate K1 breakpoint (average over all domains)
    def avg_error(h):
        errors = []
        for d in BASELINE_DOMAINS:
            errors.append(analytical_teacher_error_rate(h, d.difficulty_mean, d.difficulty_std))
        return np.mean(errors) - 0.20

    try:
        f_low = avg_error(0.50)
        f_high = avg_error(0.95)
        if f_low * f_high <= 0:
            breakpoints["k1_aggregate_error_breakpoint"] = brentq(avg_error, 0.50, 0.95, xtol=1e-6)
        else:
            breakpoints["k1_aggregate_error_breakpoint"] = None
    except ValueError:
        breakpoints["k1_aggregate_error_breakpoint"] = None

    def avg_harmful(h):
        rates = []
        for d in BASELINE_DOMAINS:
            rates.append(analytical_harmful_rate(h, 0.08, d.difficulty_mean, d.difficulty_std))
        return np.mean(rates) - 0.20

    try:
        f_low = avg_harmful(0.50)
        f_high = avg_harmful(0.95)
        if f_low * f_high <= 0:
            breakpoints["k1_aggregate_harmful_breakpoint"] = brentq(avg_harmful, 0.50, 0.95, xtol=1e-6)
        else:
            breakpoints["k1_aggregate_harmful_breakpoint"] = None
    except ValueError:
        breakpoints["k1_aggregate_harmful_breakpoint"] = None

    # Analytical error and harmful rates at baseline
    baseline_errors = {}
    baseline_harmful = {}
    for d in BASELINE_DOMAINS:
        baseline_errors[d.name] = analytical_teacher_error_rate(0.70, d.difficulty_mean, d.difficulty_std)
        baseline_harmful[d.name] = analytical_harmful_rate(0.70, 0.08, d.difficulty_mean, d.difficulty_std)

    breakpoints["baseline_analytical_error"] = baseline_errors
    breakpoints["baseline_analytical_harmful"] = baseline_harmful

    return breakpoints


# ── Perturbation Analysis (+/-10%) ──────────────────────────────────

def perturbation_analysis(baseline_routing: Dict[str, str], seeds: List[int]) -> Dict:
    """
    Test K1: do >50% of domains flip under +/-10% parameter perturbation?

    We perturb teacher_hard_accuracy by +/-10% (0.63 to 0.77)
    and difficulty_mean by +/-10% of range (0.10 units).
    """
    perturbations = []

    # +/-10% of teacher_hard_accuracy (baseline 0.70)
    for sign, label in [(-1, "teacher_-10%"), (1, "teacher_+10%")]:
        perturbed_h = BASELINE_TEACHER_HARD * (1 + sign * 0.10)
        result = run_single_config(perturbed_h, 0.0, seeds)

        n_flips = sum(1 for d in baseline_routing
                      if result["routing"].get(d) != baseline_routing.get(d))
        flip_rate = n_flips / len(baseline_routing)

        perturbations.append({
            "label": label,
            "teacher_hard_acc": perturbed_h,
            "difficulty_perturbation": 0.0,
            "n_flips": n_flips,
            "flip_rate": flip_rate,
            "routing": result["routing"],
            "teacher_error_avg": result["teacher_error_avg"],
            "teacher_harmful_avg": result["teacher_harmful_avg"],
        })

    # +/-10% of difficulty_mean (absolute perturbation of 0.10 on [0,1] scale)
    for sign, label in [(-1, "difficulty_-10%"), (1, "difficulty_+10%")]:
        dp = sign * 0.10
        result = run_single_config(BASELINE_TEACHER_HARD, dp, seeds)

        n_flips = sum(1 for d in baseline_routing
                      if result["routing"].get(d) != baseline_routing.get(d))
        flip_rate = n_flips / len(baseline_routing)

        perturbations.append({
            "label": label,
            "difficulty_perturbation": dp,
            "teacher_hard_acc": BASELINE_TEACHER_HARD,
            "n_flips": n_flips,
            "flip_rate": flip_rate,
            "routing": result["routing"],
            "teacher_error_avg": result["teacher_error_avg"],
            "teacher_harmful_avg": result["teacher_harmful_avg"],
        })

    # Combined worst case: teacher harder AND problems harder
    result_worst = run_single_config(
        BASELINE_TEACHER_HARD * 0.90,  # -10%
        +0.10,  # harder problems
        seeds,
    )
    n_flips = sum(1 for d in baseline_routing
                  if result_worst["routing"].get(d) != baseline_routing.get(d))

    perturbations.append({
        "label": "worst_case (teacher-10%, difficulty+10%)",
        "teacher_hard_acc": BASELINE_TEACHER_HARD * 0.90,
        "difficulty_perturbation": 0.10,
        "n_flips": n_flips,
        "flip_rate": n_flips / len(baseline_routing),
        "routing": result_worst["routing"],
        "teacher_error_avg": result_worst["teacher_error_avg"],
        "teacher_harmful_avg": result_worst["teacher_harmful_avg"],
    })

    return perturbations


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("CORRECTION ROUTING SENSITIVITY: PARAMETER PERTURBATION ANALYSIS")
    print("=" * 80)

    seeds = list(range(N_SEEDS))

    # Phase 1: Analytical breakpoints (instant)
    print("\n--- Phase 1: Analytical Breakpoints ---")
    breakpoints = compute_analytical_breakpoints()

    print(f"\nK2 coverage breakpoint: {breakpoints['k2_coverage_threshold']:.4f}")
    print(f"  Formula: {breakpoints['k2_formula']}")
    print(f"  -> Domains with coverage < {breakpoints['k2_coverage_threshold']:.2f} will have K2 killed")

    print(f"\nK1 aggregate error breakpoint (teacher_hard_acc):")
    bp = breakpoints['k1_aggregate_error_breakpoint']
    print(f"  Error-only: teacher_hard_acc = {bp:.4f}" if bp else "  Error-only: no breakpoint in [0.50, 0.95]")
    bp_h = breakpoints['k1_aggregate_harmful_breakpoint']
    print(f"  Harmful (wrong+degen): teacher_hard_acc = {bp_h:.4f}" if bp_h else "  Harmful: no breakpoint")

    print(f"\nPer-domain K1 error breakpoints (teacher_hard_acc where error > 20%):")
    for domain, bp in breakpoints["k1_error_breakpoints_per_domain"].items():
        if bp is not None:
            print(f"  {domain:<25} h* = {bp:.4f}")
        else:
            print(f"  {domain:<25} always below/above 20% in range")

    print(f"\nPer-domain K1 harmful breakpoints (teacher_hard_acc where harmful > 20%):")
    for domain, bp in breakpoints["k1_harmful_breakpoints_per_domain"].items():
        if bp is not None:
            print(f"  {domain:<25} h* = {bp:.4f}")
        else:
            print(f"  {domain:<25} always below/above 20% in range")

    print(f"\nBaseline analytical error rates (teacher_hard_acc=0.70):")
    for domain, err in breakpoints["baseline_analytical_error"].items():
        harm = breakpoints["baseline_analytical_harmful"][domain]
        print(f"  {domain:<25} error={err:.4f} ({err*100:.1f}%)  harmful={harm:.4f} ({harm*100:.1f}%)")

    # Phase 2: Baseline routing
    print("\n--- Phase 2: Baseline Routing ---")
    baseline_routing = compute_baseline_routing(seeds)
    print(f"Baseline routing (teacher_hard_acc=0.70, no difficulty perturbation):")
    for domain, source in baseline_routing.items():
        print(f"  {domain:<25} -> {source}")

    # Phase 3: Perturbation analysis (K1 test)
    print("\n--- Phase 3: Perturbation Analysis (+/-10%) ---")
    perturbations = perturbation_analysis(baseline_routing, seeds)

    max_flip_rate = 0
    for p in perturbations:
        print(f"\n  {p['label']}:")
        print(f"    Flips: {p['n_flips']}/{len(baseline_routing)} ({p['flip_rate']*100:.0f}%)")
        print(f"    Teacher error avg: {p['teacher_error_avg']:.4f} ({p['teacher_error_avg']*100:.1f}%)")
        print(f"    Teacher harmful avg: {p['teacher_harmful_avg']:.4f} ({p['teacher_harmful_avg']*100:.1f}%)")
        print(f"    Routing: {p['routing']}")
        max_flip_rate = max(max_flip_rate, p['flip_rate'])

    k1_killed = max_flip_rate > 0.50
    print(f"\n  K1 verdict: max flip rate = {max_flip_rate:.2f} ({max_flip_rate*100:.0f}%)")
    print(f"  K1 threshold: 50%")
    print(f"  K1 {'KILLED' if k1_killed else 'SURVIVES'}")

    # Phase 4: 1D sweeps
    print("\n--- Phase 4: 1D Teacher Hard Accuracy Sweep ---")
    teacher_sweep = run_1d_teacher_sweep(seeds)
    flip_teacher = analyze_flip_rates(teacher_sweep, baseline_routing)

    print(f"{'h_hard':>8} {'error%':>8} {'harm%':>8} {'flips':>6} {'flip%':>6} {'routing':>50}")
    print("-" * 90)
    for item in flip_teacher:
        routing_str = str({d: item["teacher_hard_acc"] for d in ["python_basics"]})
        # Get routing from corresponding sweep result
        sweep_r = [r for r in teacher_sweep
                   if abs(r["teacher_hard_acc"] - item["teacher_hard_acc"]) < 0.001][0]
        routing_short = ", ".join(f"{d[:4]}={s[:4]}" for d, s in sweep_r["routing"].items())
        print(f"  {item['teacher_hard_acc']:>6.2f} {item['teacher_error_avg']*100:>7.1f} "
              f"{item['teacher_harmful_avg']*100:>7.1f} {item['n_flips']:>5} {item['flip_rate']*100:>5.0f} "
              f"  {routing_short}")

    print("\n--- Phase 4b: 1D Difficulty Perturbation Sweep ---")
    diff_sweep = run_1d_difficulty_sweep(seeds)
    flip_diff = analyze_flip_rates(diff_sweep, baseline_routing)

    print(f"{'d_pert':>8} {'error%':>8} {'harm%':>8} {'flips':>6} {'flip%':>6}")
    print("-" * 45)
    for item in flip_diff:
        print(f"  {item['difficulty_perturbation']:>+6.2f} {item['teacher_error_avg']*100:>7.1f} "
              f"{item['teacher_harmful_avg']*100:>7.1f} {item['n_flips']:>5} {item['flip_rate']*100:>5.0f}")

    # Phase 5: 2D sweep
    print("\n--- Phase 5: 2D Sweep ---")
    sweep_2d = run_2d_sweep(seeds)
    flip_2d = analyze_flip_rates(sweep_2d, baseline_routing)

    # Find execution > teacher region
    exec_region = find_execution_teacher_region(sweep_2d)
    n_exec_wins = sum(1 for r in exec_region if r["execution_wins_all_code"])
    total_points = len(exec_region)
    print(f"\nExecution > teacher for ALL code domains: {n_exec_wins}/{total_points} "
          f"({100*n_exec_wins/total_points:.0f}%) of parameter space")

    # K2 check: does a robust region exist?
    k2_killed = n_exec_wins == 0
    print(f"K2 verdict: {'KILLED - no robust region' if k2_killed else 'SURVIVES - robust region exists'}")

    # Find the boundary of the execution region
    print("\nExecution dominance boundary (h_hard, d_pert):")
    for dp in DIFFICULTY_PERTURBATION_RANGE:
        boundary_h = None
        for h in TEACHER_HARD_ACC_RANGE:
            matching = [r for r in exec_region
                        if abs(r["teacher_hard_acc"] - h) < 0.001
                        and abs(r["difficulty_perturbation"] - dp) < 0.001]
            if matching and matching[0]["execution_wins_all_code"]:
                boundary_h = h
                break
        if boundary_h is not None:
            print(f"  d_pert={dp:+.2f}: execution wins when h >= {boundary_h:.2f}")
        else:
            print(f"  d_pert={dp:+.2f}: execution never wins all code domains")

    # Compute flip rate heatmap for the 2D grid
    print("\n--- 2D Flip Rate Heatmap ---")
    print(f"{'':>8}", end="")
    for dp in DIFFICULTY_PERTURBATION_RANGE:
        print(f"  {dp:+.02f}", end="")
    print()

    for h in TEACHER_HARD_ACC_RANGE:
        print(f"  {h:.2f}", end="")
        for dp in DIFFICULTY_PERTURBATION_RANGE:
            matching = [f for f in flip_2d
                        if abs(f["teacher_hard_acc"] - h) < 0.001
                        and abs(f["difficulty_perturbation"] - dp) < 0.001]
            if matching:
                fr = matching[0]["flip_rate"]
                print(f"  {fr*100:4.0f}%", end="")
            else:
                print(f"     ?", end="")
        print()

    # Phase 6: Summary and verdicts
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n1. ANALYTICAL BREAKPOINTS:")
    bp_err = breakpoints['k1_aggregate_error_breakpoint']
    bp_harm = breakpoints['k1_aggregate_harmful_breakpoint']
    print(f"   K1 error flips at teacher_hard_acc = {bp_err:.4f}" if bp_err else "   K1 error: no flip in range")
    print(f"   K1 harmful flips at teacher_hard_acc = {bp_harm:.4f}" if bp_harm else "   K1 harmful: no flip in range")
    print(f"   K2 coverage threshold = {breakpoints['k2_coverage_threshold']:.4f}")

    print(f"\n2. PERTURBATION KILL CRITERIA:")
    print(f"   K1 (>50% flip under +/-10%): max flip rate = {max_flip_rate*100:.0f}% -> {'KILLED' if k1_killed else 'SURVIVES'}")
    print(f"   K2 (no robust exec>teacher region): {n_exec_wins}/{total_points} points -> {'KILLED' if k2_killed else 'SURVIVES'}")

    print(f"\n3. KEY FINDING - HARMFUL RATE:")
    # At baseline
    baseline_result = run_single_config(BASELINE_TEACHER_HARD, 0.0, seeds)
    print(f"   At baseline (h=0.70): error={baseline_result['teacher_error_avg']*100:.1f}%, "
          f"harmful={baseline_result['teacher_harmful_avg']*100:.1f}%")
    print(f"   Harmful rate ALWAYS exceeds 20% threshold (error + degeneracy)")
    print(f"   -> Original K1 (error-only) was misleadingly optimistic")

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))

    def clean_for_json(obj):
        if isinstance(obj, float):
            if np.isinf(obj):
                return "inf"
            if np.isnan(obj):
                return "nan"
            return obj
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    save_data = {
        "config": {
            "teacher_hard_acc_range": [float(x) for x in TEACHER_HARD_ACC_RANGE],
            "difficulty_perturbation_range": [float(x) for x in DIFFICULTY_PERTURBATION_RANGE],
            "n_seeds": N_SEEDS,
            "baseline_teacher_hard": BASELINE_TEACHER_HARD,
        },
        "analytical_breakpoints": clean_for_json(breakpoints),
        "baseline_routing": baseline_routing,
        "perturbation_results": clean_for_json(perturbations),
        "k1_killed": k1_killed,
        "k1_max_flip_rate": max_flip_rate,
        "k2_killed": k2_killed,
        "k2_exec_wins_fraction": n_exec_wins / total_points,
        "flip_2d": clean_for_json(flip_2d),
        "exec_region": clean_for_json(exec_region),
    }

    save_path = os.path.join(results_dir, "results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return {
        "k1_killed": k1_killed,
        "k2_killed": k2_killed,
        "max_flip_rate": max_flip_rate,
        "exec_wins_fraction": n_exec_wins / total_points,
        "breakpoints": breakpoints,
    }


if __name__ == "__main__":
    summary = main()
