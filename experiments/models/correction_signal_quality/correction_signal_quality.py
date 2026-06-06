#!/usr/bin/env python3
"""
Correction Signal Quality: Simulation Study

Compares three correction sources for SOLE expert evolution:
  (a) Human corrections (gold standard, expensive)
  (b) 70B teacher corrections (automated, cheaper)
  (c) Execution feedback (pass/fail, code-only, free)

Measures: correction accuracy, expert improvement per correction,
cost per useful correction. Builds a decision tree for domain routing.

Pure numpy/scipy simulation -- no model training required.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import expit as sigmoid  # logistic sigmoid


# ── Configuration ────────────────────────────────────────────────

@dataclass
class CorrectionSource:
    """A source of corrections with error and cost characteristics."""
    name: str
    base_accuracy: float          # baseline accuracy on easy problems
    hard_accuracy: float          # accuracy on hardest problems
    degeneracy_rate: float        # P(degenerate | accepted as correct)
    cost_per_correction: float    # USD
    applicable_domains: List[str] # which domains this source works on

    # Sigmoid parameters fit to (easy, hard) accuracy endpoints
    beta_0: float = 0.0
    beta_1: float = 0.0

    def __post_init__(self):
        # Fit sigmoid: accuracy(d) = sigmoid(beta_0 + beta_1 * d)
        # where d in [0,1] is difficulty
        # At d=0: accuracy = base_accuracy -> beta_0 = logit(base_accuracy)
        # At d=1: accuracy = hard_accuracy -> beta_0 + beta_1 = logit(hard_accuracy)
        if self.base_accuracy > 0 and self.base_accuracy < 1:
            self.beta_0 = np.log(self.base_accuracy / (1 - self.base_accuracy))
        else:
            self.beta_0 = 5.0 if self.base_accuracy >= 1 else -5.0

        if self.hard_accuracy > 0 and self.hard_accuracy < 1:
            logit_hard = np.log(self.hard_accuracy / (1 - self.hard_accuracy))
        else:
            logit_hard = 5.0 if self.hard_accuracy >= 1 else -5.0
        self.beta_1 = logit_hard - self.beta_0

    def accuracy(self, difficulty: np.ndarray) -> np.ndarray:
        """Correction accuracy as a function of problem difficulty."""
        return sigmoid(self.beta_0 + self.beta_1 * difficulty)

    def avg_accuracy(self, n_samples: int = 10000) -> float:
        """Average accuracy over uniform difficulty distribution."""
        d = np.linspace(0, 1, n_samples)
        return float(np.mean(self.accuracy(d)))


@dataclass
class Domain:
    """A domain with specific difficulty distribution and feedback properties."""
    name: str
    domain_type: str              # "code_simple", "code_algo", "code_systems", "writing", "reasoning", "medical"
    difficulty_mean: float        # mean difficulty [0, 1]
    difficulty_std: float         # std of difficulty
    test_coverage: float          # for code: test suite coverage [0, 1]; 0 for non-code
    is_code: bool                 # whether execution feedback is applicable
    initial_quality: float        # q_0 from distillation


# ── Define Correction Sources ────────────────────────────────────

HUMAN = CorrectionSource(
    name="human",
    base_accuracy=0.97,
    hard_accuracy=0.90,
    degeneracy_rate=0.02,
    cost_per_correction=2.00,
    applicable_domains=["all"],
)

TEACHER = CorrectionSource(
    name="teacher_70b",
    base_accuracy=0.92,
    hard_accuracy=0.70,
    degeneracy_rate=0.08,
    cost_per_correction=0.001,
    applicable_domains=["all"],
)

EXECUTION = CorrectionSource(
    name="execution",
    base_accuracy=0.99,       # if tests pass, answer is almost certainly correct
    hard_accuracy=0.85,       # complex systems have integration gaps
    degeneracy_rate=0.03,     # base rate; adjusted per domain by coverage
    cost_per_correction=0.0001,
    applicable_domains=["code_simple", "code_algo", "code_systems"],
)


# ── Define Domains ───────────────────────────────────────────────

DOMAINS = [
    Domain("python_basics", "code_simple", 0.3, 0.15, 0.95, True, 0.70),
    Domain("algorithm_design", "code_algo", 0.6, 0.20, 0.80, True, 0.55),
    Domain("systems_programming", "code_systems", 0.75, 0.15, 0.60, True, 0.50),
    Domain("creative_writing", "writing", 0.5, 0.25, 0.0, False, 0.60),
    Domain("logical_reasoning", "reasoning", 0.65, 0.20, 0.0, False, 0.55),
    Domain("medical_qa", "medical", 0.7, 0.15, 0.0, False, 0.50),
]

SOURCES = [HUMAN, TEACHER, EXECUTION]

# ── Simulation Parameters ────────────────────────────────────────

N_CORRECTIONS = 200       # corrections per domain per source per seed
N_CYCLES = 20             # self-learning evolution cycles
N_SEEDS = 10              # Monte Carlo seeds
DELTA_BASE = 0.02         # quality improvement per correct non-degenerate correction
DELTA_DEGEN = 0.01        # quality degradation per degenerate correction
DELTA_WRONG = 0.015       # quality degradation per wrong correction
ALPHA = 0.7               # diminishing returns exponent


# ── Core Simulation ──────────────────────────────────────────────

def generate_problems(domain: Domain, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n problems with difficulties drawn from domain's distribution."""
    difficulties = rng.normal(domain.difficulty_mean, domain.difficulty_std, n)
    return np.clip(difficulties, 0.01, 0.99)


def simulate_corrections(
    source: CorrectionSource,
    domain: Domain,
    difficulties: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate corrections from a source on problems of given difficulties.

    Returns:
        is_correct: bool array, whether the correction is correct
        is_degenerate: bool array, whether the correction is degenerate (among correct ones)
        is_applicable: bool array, whether the source can provide signal
    """
    n = len(difficulties)

    # Check applicability
    if "all" not in source.applicable_domains and domain.domain_type not in source.applicable_domains:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    is_applicable = np.ones(n, dtype=bool)

    # Accuracy varies with difficulty
    acc = source.accuracy(difficulties)
    is_correct = rng.random(n) < acc

    # Degeneracy rate (for execution: modulated by test coverage)
    if source.name == "execution":
        # Higher degeneracy when test coverage is low
        degen_rate = (1 - domain.test_coverage) * 0.30  # gamma = 0.30
    else:
        degen_rate = source.degeneracy_rate

    is_degenerate = is_correct & (rng.random(n) < degen_rate)

    return is_correct, is_degenerate, is_applicable


def simulate_expert_trajectory(
    source: CorrectionSource,
    domain: Domain,
    n_corrections: int,
    rng: np.random.Generator,
) -> Dict:
    """
    Simulate expert quality trajectory under corrections from one source.

    Returns dict with trajectory data and summary statistics.
    """
    # Generate problem difficulties
    difficulties = generate_problems(domain, n_corrections, rng)

    # Simulate corrections
    is_correct, is_degenerate, is_applicable = simulate_corrections(
        source, domain, difficulties, rng
    )

    # Track expert quality over corrections
    q_current = domain.initial_quality
    quality_trajectory = [q_current]

    n_correct_nondegen = 0
    n_degenerate = 0
    n_wrong = 0
    n_no_signal = 0

    for i in range(n_corrections):
        if not is_applicable[i]:
            n_no_signal += 1
            quality_trajectory.append(q_current)
            continue

        if is_correct[i] and not is_degenerate[i]:
            # Correct, non-degenerate: improve with diminishing returns
            delta = DELTA_BASE * (1 - q_current) ** ALPHA
            q_current += delta
            n_correct_nondegen += 1
        elif is_correct[i] and is_degenerate[i]:
            # Degenerate: technically correct but harmful
            q_current -= DELTA_DEGEN
            n_degenerate += 1
        else:
            # Wrong correction
            q_current -= DELTA_WRONG
            n_wrong += 1

        q_current = max(0.0, min(1.0, q_current))
        quality_trajectory.append(q_current)

    total_cost = n_corrections * source.cost_per_correction
    if n_no_signal == n_corrections:
        total_cost = 0  # no signal means no cost incurred

    final_quality = quality_trajectory[-1]
    quality_delta = final_quality - domain.initial_quality

    # Effective Improvement Rate (quality per dollar)
    eir = quality_delta / total_cost if total_cost > 0 else (
        float('inf') if quality_delta > 0 else 0.0
    )

    return {
        "source": source.name,
        "domain": domain.name,
        "domain_type": domain.domain_type,
        "initial_quality": domain.initial_quality,
        "final_quality": final_quality,
        "quality_delta": quality_delta,
        "n_correct_nondegen": n_correct_nondegen,
        "n_degenerate": n_degenerate,
        "n_wrong": n_wrong,
        "n_no_signal": n_no_signal,
        "total_cost": total_cost,
        "eir": eir,
        "accuracy_empirical": n_correct_nondegen / max(1, n_corrections - n_no_signal),
        "error_rate_empirical": n_wrong / max(1, n_corrections - n_no_signal),
        "degeneracy_rate_empirical": n_degenerate / max(1, n_corrections - n_no_signal),
        "trajectory": quality_trajectory,
    }


def run_full_simulation(seeds: List[int] = None) -> Dict:
    """Run the full simulation across all sources, domains, and seeds."""
    if seeds is None:
        seeds = list(range(N_SEEDS))

    all_results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for domain in DOMAINS:
            for source in SOURCES:
                result = simulate_expert_trajectory(
                    source, domain, N_CORRECTIONS, rng
                )
                result["seed"] = seed
                all_results.append(result)

    return {"results": all_results, "seeds": seeds}


def compute_aggregate_statistics(data: Dict) -> Dict:
    """Compute aggregate statistics from simulation results."""
    results = data["results"]

    # Group by (source, domain)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["source"], r["domain"])].append(r)

    # Also group by (source, domain_type) for type-level analysis
    type_groups = defaultdict(list)
    for r in results:
        type_groups[(r["source"], r["domain_type"])].append(r)

    aggregates = {}
    for (source, domain), runs in groups.items():
        key = f"{source}_{domain}"
        n = len(runs)
        aggregates[key] = {
            "source": source,
            "domain": domain,
            "n_seeds": n,
            "quality_delta_mean": float(np.mean([r["quality_delta"] for r in runs])),
            "quality_delta_std": float(np.std([r["quality_delta"] for r in runs])),
            "accuracy_mean": float(np.mean([r["accuracy_empirical"] for r in runs])),
            "accuracy_std": float(np.std([r["accuracy_empirical"] for r in runs])),
            "error_rate_mean": float(np.mean([r["error_rate_empirical"] for r in runs])),
            "degeneracy_rate_mean": float(np.mean([r["degeneracy_rate_empirical"] for r in runs])),
            "total_cost_mean": float(np.mean([r["total_cost"] for r in runs])),
            "eir_mean": float(np.mean([r["eir"] for r in runs])) if all(r["eir"] != float('inf') for r in runs) else float('inf'),
        }

    # Source-level averages (across all applicable domains)
    source_agg = {}
    for source in SOURCES:
        applicable_runs = [r for r in results if r["source"] == source.name and r["n_no_signal"] < N_CORRECTIONS]
        if applicable_runs:
            source_agg[source.name] = {
                "accuracy_mean": float(np.mean([r["accuracy_empirical"] for r in applicable_runs])),
                "accuracy_std": float(np.std([r["accuracy_empirical"] for r in applicable_runs])),
                "error_rate_mean": float(np.mean([r["error_rate_empirical"] for r in applicable_runs])),
                "degeneracy_rate_mean": float(np.mean([r["degeneracy_rate_empirical"] for r in applicable_runs])),
                "quality_delta_mean": float(np.mean([r["quality_delta"] for r in applicable_runs])),
                "cost_mean": float(np.mean([r["total_cost"] for r in applicable_runs])),
                "n_applicable_domains": len(set(r["domain"] for r in applicable_runs)),
            }

    return {
        "per_domain": aggregates,
        "per_source": source_agg,
    }


def evaluate_kill_criteria(agg: Dict) -> Dict:
    """Evaluate K1 and K2 kill criteria against simulation data."""

    # K1: Teacher corrections are wrong >20% of the time
    teacher_stats = agg["per_source"].get("teacher_70b", {})
    teacher_error_rate = teacher_stats.get("error_rate_mean", 0)
    k1_killed = teacher_error_rate > 0.20

    # Also check per-domain: are there domains where teacher exceeds 20%?
    teacher_domain_errors = {}
    for key, stats in agg["per_domain"].items():
        if stats["source"] == "teacher_70b":
            teacher_domain_errors[stats["domain"]] = stats["error_rate_mean"]

    k1_worst_domain = max(teacher_domain_errors.items(), key=lambda x: x[1]) if teacher_domain_errors else ("none", 0)

    # K2: Execution feedback produces degenerate solutions (>10% accepted are degenerate)
    exec_degeneracy = {}
    for key, stats in agg["per_domain"].items():
        if stats["source"] == "execution" and stats["accuracy_mean"] > 0:
            # Degenerate rate among accepted (correct) solutions
            exec_degeneracy[stats["domain"]] = stats["degeneracy_rate_mean"]

    k2_worst = max(exec_degeneracy.items(), key=lambda x: x[1]) if exec_degeneracy else ("none", 0)
    k2_killed = k2_worst[1] > 0.10 if exec_degeneracy else False

    return {
        "K1_teacher_error_rate_avg": teacher_error_rate,
        "K1_threshold": 0.20,
        "K1_killed": k1_killed,
        "K1_per_domain": teacher_domain_errors,
        "K1_worst_domain": k1_worst_domain[0],
        "K1_worst_error_rate": k1_worst_domain[1],
        "K2_exec_degeneracy_per_domain": exec_degeneracy,
        "K2_worst_domain": k2_worst[0],
        "K2_worst_degeneracy": k2_worst[1],
        "K2_threshold": 0.10,
        "K2_killed": k2_killed,
    }


def build_decision_tree(agg: Dict) -> Dict:
    """Build optimal correction source routing per domain."""
    decisions = {}

    for domain in DOMAINS:
        domain_results = {}
        for source in SOURCES:
            key = f"{source.name}_{domain.name}"
            if key in agg["per_domain"]:
                stats = agg["per_domain"][key]
                if stats["quality_delta_mean"] > -999:  # has signal
                    domain_results[source.name] = {
                        "quality_delta": stats["quality_delta_mean"],
                        "cost": stats["total_cost_mean"],
                        "eir": stats["eir_mean"],
                        "accuracy": stats["accuracy_mean"],
                    }

        # Pick best by EIR (cost-effectiveness), then by quality if EIR is infinite
        valid = {k: v for k, v in domain_results.items() if v["quality_delta"] > 0}

        if not valid:
            best = "human"  # fallback
            reason = "no source improves quality"
        else:
            # Primary: best EIR
            finite_eir = {k: v for k, v in valid.items() if v["eir"] != float('inf')}
            if finite_eir:
                best = max(finite_eir, key=lambda k: finite_eir[k]["eir"])
                reason = f"highest EIR ({finite_eir[best]['eir']:.1f} quality/dollar)"
            else:
                best = max(valid, key=lambda k: valid[k]["quality_delta"])
                reason = f"best quality delta ({valid[best]['quality_delta']:.4f})"

        decisions[domain.name] = {
            "optimal_source": best,
            "reason": reason,
            "all_sources": domain_results,
            "domain_type": domain.domain_type,
            "is_code": domain.is_code,
        }

    # Build simplified routing rules
    routing_rules = {
        "code_with_good_tests": "execution (coverage > 0.8)",
        "code_with_poor_tests": "teacher_70b (coverage < 0.6: degeneracy risk)",
        "non_code": "teacher_70b (only automated option)",
        "critical_accuracy": "human (when >95% accuracy required)",
    }

    return {
        "per_domain_decisions": decisions,
        "routing_rules": routing_rules,
    }


def print_results(agg: Dict, kill: Dict, tree: Dict):
    """Print formatted results to stdout."""
    print("=" * 80)
    print("CORRECTION SIGNAL QUALITY: SIMULATION RESULTS")
    print("=" * 80)

    # Source-level summary
    print("\n--- Source-Level Summary ---")
    print(f"{'Source':<15} {'Accuracy':>10} {'Error Rate':>12} {'Degen Rate':>12} {'Quality Delta':>15} {'Cost':>10}")
    print("-" * 75)
    for source_name, stats in agg["per_source"].items():
        print(f"{source_name:<15} {stats['accuracy_mean']:>9.3f} {stats['error_rate_mean']:>11.3f} "
              f"{stats['degeneracy_rate_mean']:>11.3f} {stats['quality_delta_mean']:>14.4f} "
              f"${stats['cost_mean']:>8.4f}")

    # Per-domain results
    print("\n--- Per-Domain Quality Delta (mean over seeds) ---")
    print(f"{'Domain':<25} {'Human':>10} {'Teacher':>10} {'Execution':>10} {'Best':>12}")
    print("-" * 70)
    for domain in DOMAINS:
        vals = {}
        for source in SOURCES:
            key = f"{source.name}_{domain.name}"
            if key in agg["per_domain"]:
                vals[source.name] = agg["per_domain"][key]["quality_delta_mean"]

        human_v = vals.get("human", 0)
        teacher_v = vals.get("teacher_70b", 0)
        exec_v = vals.get("execution", 0)

        valid = {k: v for k, v in vals.items() if v != 0}
        best = max(valid, key=lambda k: valid[k]) if valid else "none"

        print(f"{domain.name:<25} {human_v:>+9.4f} {teacher_v:>+9.4f} {exec_v:>+9.4f} {best:>12}")

    # Per-domain cost
    print("\n--- Per-Domain Cost (USD, for 200 corrections) ---")
    print(f"{'Domain':<25} {'Human':>10} {'Teacher':>10} {'Execution':>10}")
    print("-" * 55)
    for domain in DOMAINS:
        costs = {}
        for source in SOURCES:
            key = f"{source.name}_{domain.name}"
            if key in agg["per_domain"]:
                costs[source.name] = agg["per_domain"][key]["total_cost_mean"]

        print(f"{domain.name:<25} ${costs.get('human', 0):>8.2f} ${costs.get('teacher_70b', 0):>8.4f} ${costs.get('execution', 0):>8.4f}")

    # Kill criteria
    print("\n--- Kill Criteria Assessment ---")
    print(f"\nK1: Teacher corrections wrong >20% of the time")
    print(f"  Average teacher error rate: {kill['K1_teacher_error_rate_avg']:.3f} ({kill['K1_teacher_error_rate_avg']*100:.1f}%)")
    print(f"  Threshold: 0.200 (20%)")
    print(f"  Worst domain: {kill['K1_worst_domain']} at {kill['K1_worst_error_rate']:.3f} ({kill['K1_worst_error_rate']*100:.1f}%)")
    print(f"  Per-domain errors:")
    for domain, err in sorted(kill["K1_per_domain"].items(), key=lambda x: -x[1]):
        marker = " <-- EXCEEDS" if err > 0.20 else ""
        print(f"    {domain:<25} {err:.3f} ({err*100:.1f}%){marker}")
    print(f"  VERDICT: {'KILLED' if kill['K1_killed'] else 'SURVIVES'}")

    print(f"\nK2: Execution feedback produces degenerate solutions (>10% of accepted)")
    print(f"  Per-domain degeneracy rates:")
    for domain, deg in sorted(kill["K2_exec_degeneracy_per_domain"].items(), key=lambda x: -x[1]):
        marker = " <-- EXCEEDS" if deg > 0.10 else ""
        print(f"    {domain:<25} {deg:.3f} ({deg*100:.1f}%){marker}")
    print(f"  Worst domain: {kill['K2_worst_domain']} at {kill['K2_worst_degeneracy']:.3f} ({kill['K2_worst_degeneracy']*100:.1f}%)")
    print(f"  VERDICT: {'KILLED' if kill['K2_killed'] else 'SURVIVES'}")

    # Decision tree
    print("\n--- Optimal Correction Routing (Decision Tree) ---")
    for domain_name, decision in tree["per_domain_decisions"].items():
        print(f"  {domain_name:<25} -> {decision['optimal_source']:<15} ({decision['reason']})")

    print("\n--- Simplified Routing Rules ---")
    for rule, source in tree["routing_rules"].items():
        print(f"  {rule:<30} -> {source}")

    # Cost-effectiveness analysis
    print("\n--- Cost-Effectiveness Analysis (quality improvement per $1) ---")
    for domain in DOMAINS:
        print(f"\n  {domain.name}:")
        for source in SOURCES:
            key = f"{source.name}_{domain.name}"
            if key in agg["per_domain"]:
                stats = agg["per_domain"][key]
                if stats["eir_mean"] != float('inf') and stats["total_cost_mean"] > 0:
                    print(f"    {source.name:<15} EIR = {stats['eir_mean']:>10.2f} q/$")
                elif stats["quality_delta_mean"] > 0:
                    print(f"    {source.name:<15} EIR = inf (free signal)")
                else:
                    print(f"    {source.name:<15} N/A (no signal)")


def main():
    """Main entry point."""
    print("Running correction signal quality simulation...")
    print(f"  Domains: {len(DOMAINS)}")
    print(f"  Sources: {len(SOURCES)}")
    print(f"  Corrections per (source, domain): {N_CORRECTIONS}")
    print(f"  Seeds: {N_SEEDS}")
    print()

    # Run simulation
    data = run_full_simulation()

    # Compute aggregates
    agg = compute_aggregate_statistics(data)

    # Evaluate kill criteria
    kill = evaluate_kill_criteria(agg)

    # Build decision tree
    tree = build_decision_tree(agg)

    # Print results
    print_results(agg, kill, tree)

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))

    # Save aggregates (without trajectories -- too large)
    save_data = {
        "config": {
            "n_corrections": N_CORRECTIONS,
            "n_seeds": N_SEEDS,
            "delta_base": DELTA_BASE,
            "delta_degen": DELTA_DEGEN,
            "delta_wrong": DELTA_WRONG,
            "alpha": ALPHA,
            "domains": [{"name": d.name, "type": d.domain_type, "difficulty_mean": d.difficulty_mean,
                         "test_coverage": d.test_coverage, "is_code": d.is_code, "initial_quality": d.initial_quality}
                        for d in DOMAINS],
            "sources": [{"name": s.name, "base_accuracy": s.base_accuracy, "hard_accuracy": s.hard_accuracy,
                         "degeneracy_rate": s.degeneracy_rate, "cost": s.cost_per_correction}
                        for s in SOURCES],
        },
        "aggregates": agg,
        "kill_criteria": kill,
        "decision_tree": tree,
    }

    # Handle inf/nan for JSON
    def clean_for_json(obj):
        if isinstance(obj, float):
            if np.isinf(obj):
                return "inf"
            if np.isnan(obj):
                return "nan"
            return obj
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    save_path = os.path.join(results_dir, "results.json")
    with open(save_path, "w") as f:
        json.dump(clean_for_json(save_data), f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Return summary for programmatic use
    return {
        "K1_killed": kill["K1_killed"],
        "K2_killed": kill["K2_killed"],
        "teacher_avg_error": kill["K1_teacher_error_rate_avg"],
        "exec_worst_degeneracy": kill["K2_worst_degeneracy"],
        "optimal_routing": {d: v["optimal_source"] for d, v in tree["per_domain_decisions"].items()},
    }


if __name__ == "__main__":
    summary = main()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"K1 (teacher error >20%): {'KILLED' if summary['K1_killed'] else 'SURVIVES'} (avg error: {summary['teacher_avg_error']:.1%})")
    print(f"K2 (exec degeneracy >10%): {'KILLED' if summary['K2_killed'] else 'SURVIVES'} (worst: {summary['exec_worst_degeneracy']:.1%})")
    print(f"Optimal routing: {summary['optimal_routing']}")
