#!/usr/bin/env python3
"""
Execution-Based Self-Learning Loop Simulation
=============================================

Simulates the dynamics of a code expert self-improving via execution feedback:
  1. Expert generates K solutions per problem
  2. Execute each against test cases (binary pass/fail oracle)
  3. SFT variant: passing solutions -> new training data
     DPO variant: (pass, fail) pairs -> preference training data
  4. Retrain expert on combined original + new data
  5. Measure pass@1 improvement and diversity
  6. Repeat for T cycles

Calibrated from published results:
  - SPIN (Chen et al., 2024): 3 iterations, ~5pp improvement, convergence
  - ReST-EM (Singh et al., 2024): 2-3 iterations, 5-15pp on code
  - CodeRL (Le et al., 2022): +5-12% on MBPP with execution feedback
  - Self-play (Haluptzok et al., 2022): accuracy doubles with verified data
  - Model collapse (Shumailov et al., 2023): 5-10 iter before degradation
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """Self-learning loop simulation parameters."""
    # Problem set
    n_problems: int = 200          # number of problems in the evaluation set
    n_train_problems: int = 500    # number of problems available for training data generation

    # Generation
    K: int = 10                     # solutions generated per problem per cycle
    temperature: float = 0.8        # sampling temperature (affects diversity)

    # Expert model parameters (abstract)
    initial_pass_rate: float = 0.30  # initial pass@1 (calibrated: small code models ~30% on MBPP)
    skill_dim: int = 50             # dimensionality of the latent skill vector
    noise_scale: float = 0.05       # noise in skill update

    # Learning dynamics
    sft_learning_rate: float = 0.15  # how much pass@1 improves per unit of good signal (SFT)
    dpo_learning_rate: float = 0.22  # DPO learns faster due to contrastive signal (calibrated: ~1.5x SFT)
    dpo_negative_weight: float = 0.3 # how much negative examples contribute

    # Diversity / collapse dynamics
    initial_diversity: float = 1.0   # normalized diversity (1.0 = maximum)
    diversity_decay_self: float = 0.03  # diversity lost per cycle from self-training (matches MATH.md gamma_SFT)
    diversity_decay_dpo: float = 0.015  # DPO preserves diversity better (matches MATH.md gamma_DPO)
    diversity_recovery_fresh: float = 0.01  # recovery from fresh data mixing (per % fresh)
    fresh_data_fraction: float = 0.0  # fraction of fresh data mixed in (0 = pure self-play)

    # Collapse threshold
    collapse_diversity_threshold: float = 0.3  # below this, model has collapsed
    collapse_pass_rate_penalty: float = 0.5    # pass@1 multiplier when collapsed

    # Simulation
    n_cycles: int = 15              # total self-learning cycles
    n_seeds: int = 20               # Monte Carlo seeds

    # Problem difficulty distribution
    difficulty_mean: float = 0.5    # mean difficulty (0=easy, 1=hard)
    difficulty_std: float = 0.25    # spread of difficulties


# ============================================================
# Self-Learning Loop Model
# ============================================================

def problem_difficulty_distribution(config: Config, rng: np.random.Generator) -> np.ndarray:
    """Generate problem difficulties ~ truncated normal on [0, 1]."""
    d = rng.normal(config.difficulty_mean, config.difficulty_std, config.n_problems)
    return np.clip(d, 0.01, 0.99)


def pass_probability(skill: float, difficulty: np.ndarray, diversity: float) -> np.ndarray:
    """
    Probability of generating a correct solution.

    p(pass | skill, difficulty, diversity) = sigmoid(skill_logit - difficulty_logit)

    Diversity modulates effective skill: collapsed model can't explore solution space,
    so even if nominal skill is high, pass@1 drops.
    """
    # Effective skill accounts for diversity
    effective_skill = skill * (0.5 + 0.5 * diversity)  # at diversity=0.3, skill is 65% effective

    # Logit model: higher skill -> higher pass rate, higher difficulty -> lower
    skill_logit = np.log(effective_skill / (1 - np.clip(effective_skill, 0.01, 0.99)))
    diff_logit = np.log(difficulty / (1 - difficulty))

    logit = skill_logit - diff_logit
    return 1.0 / (1.0 + np.exp(-logit))


def generate_solutions(pass_probs: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate K solutions per problem, returns (n_problems, K) binary array.
    1 = pass, 0 = fail.
    """
    # Each solution is an independent Bernoulli trial
    return rng.random((len(pass_probs), K)) < pass_probs[:, None]


def compute_pass_at_1(pass_probs: np.ndarray) -> float:
    """Expected pass@1 = mean probability of passing on first try."""
    return float(np.mean(pass_probs))


def compute_pass_at_k(solutions: np.ndarray, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al., 2021 Codex paper).
    pass@k = E[1 - C(n-c, k) / C(n, k)] where c = number passing out of n.
    """
    n = solutions.shape[1]
    c = solutions.sum(axis=1)  # number of correct per problem

    # For each problem, compute 1 - C(n-c,k)/C(n,k)
    # Use log to avoid overflow
    pass_k = np.zeros(len(c))
    for i in range(len(c)):
        ci = int(c[i])
        if ci == 0:
            pass_k[i] = 0.0
        elif ci >= k:
            # Almost certainly pass
            if n - ci < k:
                pass_k[i] = 1.0
            else:
                # 1 - prod_{j=0}^{k-1} (n-c-j)/(n-j)
                log_ratio = sum(np.log(max(n - ci - j, 1e-10)) - np.log(n - j) for j in range(k))
                pass_k[i] = 1.0 - np.exp(log_ratio)
        else:
            pass_k[i] = 1.0 - np.exp(
                sum(np.log(max(n - ci - j, 1e-10)) - np.log(n - j) for j in range(min(k, n)))
            )

    return float(np.mean(pass_k))


def sft_update(skill: float, solutions: np.ndarray, difficulty: np.ndarray,
               lr: float, diversity: float) -> float:
    """
    SFT update: learn from passing solutions only.

    Signal strength = fraction of problems with at least one passing solution,
    weighted by difficulty (harder passing solutions teach more).
    """
    has_pass = solutions.any(axis=1)  # problems with >= 1 passing solution
    if not has_pass.any():
        return skill

    # Signal: how much new information the passing solutions provide
    # Harder problems that pass provide more signal
    signal = np.mean(difficulty[has_pass])  # average difficulty of solved problems

    # But signal quality degrades with low diversity (same solutions over and over)
    effective_signal = signal * diversity

    # Diminishing returns: improvement decreases as skill approaches 1
    headroom = 1.0 - skill
    delta = lr * effective_signal * headroom

    return min(skill + delta, 0.99)


def dpo_update(skill: float, solutions: np.ndarray, difficulty: np.ndarray,
               lr: float, neg_weight: float, diversity: float) -> float:
    """
    DPO update: learn from (pass, fail) pairs.

    Contrastive signal: for each problem, if we have both passing AND failing solutions,
    the model learns both what to do AND what to avoid.
    """
    has_pass = solutions.any(axis=1)
    has_fail = (~solutions.astype(bool)).any(axis=1)
    has_both = has_pass & has_fail  # problems with contrastive pairs

    if not has_both.any():
        # Fall back to SFT-like update if no contrastive pairs
        return sft_update(skill, solutions, difficulty, lr * 0.7, diversity)

    # Positive signal (same as SFT)
    pos_signal = np.mean(difficulty[has_both])

    # Negative signal: harder failed problems teach more about what to avoid
    # But we need to be careful not to overweight negatives
    neg_signal = np.mean(difficulty[has_both]) * neg_weight

    # Combined signal
    total_signal = (pos_signal + neg_signal) * diversity

    headroom = 1.0 - skill
    delta = lr * total_signal * headroom

    return min(skill + delta, 0.99)


def update_diversity(diversity: float, solutions: np.ndarray,
                     decay_rate: float, fresh_fraction: float,
                     recovery_rate: float, rng: np.random.Generator) -> float:
    """
    Update diversity based on self-training dynamics.

    Uses the clean geometric decay model from MATH.md:
        d_{t+1} = d_t * (1 - gamma)

    This is the constant-decay model (no acceleration). The stress test
    separately evaluates the accelerating decay model. Both scripts now use
    the same diversity model to ensure results are directly comparable.

    Fresh data mixing partially counteracts decay:
        d_{t+1} = d_t * (1 - gamma) + f * r
    """
    # Clean geometric decay (matches MATH.md exactly)
    diversity *= (1.0 - decay_rate)

    # Recovery from fresh data
    if fresh_fraction > 0:
        diversity += fresh_fraction * recovery_rate

    # Small random perturbation
    diversity += rng.normal(0, 0.005)

    return np.clip(diversity, 0.05, 1.0)


def compute_unique_ngram_ratio(solutions: np.ndarray) -> float:
    """
    Proxy for output diversity: fraction of unique solution patterns.
    In real setting, this would be unique n-grams in generated code.
    Here, we measure the fraction of unique binary patterns across solutions.
    """
    n_problems, K = solutions.shape
    unique_patterns = set()
    for i in range(n_problems):
        pattern = tuple(solutions[i])
        unique_patterns.add(pattern)
    return len(unique_patterns) / n_problems


def run_self_learning_loop(config: Config, method: str, seed: int,
                           fresh_fraction: float = 0.0) -> Dict:
    """
    Run one self-learning loop simulation.

    method: 'sft' or 'dpo'
    Returns trajectory of metrics across cycles.
    """
    rng = np.random.default_rng(seed)

    # Initialize
    skill = config.initial_pass_rate
    diversity = config.initial_diversity
    difficulties = problem_difficulty_distribution(config, rng)

    # Track metrics
    trajectory = {
        'pass_at_1': [],
        'pass_at_5': [],
        'pass_at_10': [],
        'diversity': [],
        'unique_pattern_ratio': [],
        'skill': [],
        'n_passing_problems': [],
        'n_contrastive_pairs': [],
        'collapsed': [],
    }

    for cycle in range(config.n_cycles):
        # 1. Compute current pass probabilities
        pass_probs = pass_probability(skill, difficulties, diversity)

        # 2. Generate K solutions per problem
        solutions = generate_solutions(pass_probs, config.K, rng)

        # 3. Compute metrics
        pass1 = compute_pass_at_1(pass_probs)
        pass5 = compute_pass_at_k(solutions, 5)
        pass10 = compute_pass_at_k(solutions, 10)
        unique_ratio = compute_unique_ngram_ratio(solutions)
        n_passing = int(solutions.any(axis=1).sum())
        n_contrastive = int((solutions.any(axis=1) & (~solutions.astype(bool)).any(axis=1)).sum())
        is_collapsed = diversity < config.collapse_diversity_threshold

        trajectory['pass_at_1'].append(pass1)
        trajectory['pass_at_5'].append(pass5)
        trajectory['pass_at_10'].append(pass10)
        trajectory['diversity'].append(diversity)
        trajectory['unique_pattern_ratio'].append(unique_ratio)
        trajectory['skill'].append(skill)
        trajectory['n_passing_problems'].append(n_passing)
        trajectory['n_contrastive_pairs'].append(n_contrastive)
        trajectory['collapsed'].append(is_collapsed)

        # 4. Update skill based on training method
        if method == 'sft':
            lr = config.sft_learning_rate
            decay = config.diversity_decay_self
            skill = sft_update(skill, solutions, difficulties, lr, diversity)
        elif method == 'dpo':
            lr = config.dpo_learning_rate
            decay = config.diversity_decay_dpo
            skill = dpo_update(skill, solutions, difficulties, lr,
                              config.dpo_negative_weight, diversity)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 5. Update diversity
        diversity = update_diversity(diversity, solutions, decay,
                                    fresh_fraction, config.diversity_recovery_fresh, rng)

        # 6. Apply collapse penalty if diversity too low
        if diversity < config.collapse_diversity_threshold:
            skill *= config.collapse_pass_rate_penalty

    return trajectory


# ============================================================
# Analysis
# ============================================================

def analyze_trajectories(trajectories: List[Dict], method: str) -> Dict:
    """Compute summary statistics across seeds."""
    n_seeds = len(trajectories)
    n_cycles = len(trajectories[0]['pass_at_1'])

    # Stack into arrays
    pass1 = np.array([t['pass_at_1'] for t in trajectories])
    pass5 = np.array([t['pass_at_5'] for t in trajectories])
    diversity = np.array([t['diversity'] for t in trajectories])
    unique_ratio = np.array([t['unique_pattern_ratio'] for t in trajectories])
    collapsed = np.array([t['collapsed'] for t in trajectories])

    # K1: Does pass@1 improve after 5 cycles?
    initial_pass1 = pass1[:, 0]
    cycle5_pass1 = pass1[:, min(4, n_cycles-1)]
    final_pass1 = pass1[:, -1]

    improvement_5 = cycle5_pass1 - initial_pass1
    improvement_final = final_pass1 - initial_pass1

    k1_pass = float(np.mean(improvement_5) > 0)  # fraction of seeds that improve
    k1_mean_improvement = float(np.mean(improvement_5))

    # K2: Does diversity degrade? (model collapse)
    initial_div = diversity[:, 0]
    final_div = diversity[:, -1]
    div_change = (final_div - initial_div) / initial_div

    k2_collapse_fraction = float(np.mean(collapsed[:, -1]))  # fraction collapsed at end
    k2_mean_div_change = float(np.mean(div_change))

    # Peak pass@1 and which cycle
    peak_pass1_per_seed = np.max(pass1, axis=1)
    peak_cycle_per_seed = np.argmax(pass1, axis=1)

    # Convergence: cycle where improvement < 1pp
    convergence_cycles = []
    for s in range(n_seeds):
        for c in range(1, n_cycles):
            if abs(pass1[s, c] - pass1[s, c-1]) < 0.01:
                convergence_cycles.append(c)
                break
        else:
            convergence_cycles.append(n_cycles)

    return {
        'method': method,
        'n_seeds': n_seeds,
        'n_cycles': n_cycles,
        # Pass@1 trajectory
        'pass1_mean': pass1.mean(axis=0).tolist(),
        'pass1_std': pass1.std(axis=0).tolist(),
        'pass1_initial_mean': float(np.mean(initial_pass1)),
        'pass1_cycle5_mean': float(np.mean(cycle5_pass1)),
        'pass1_final_mean': float(np.mean(final_pass1)),
        'pass1_peak_mean': float(np.mean(peak_pass1_per_seed)),
        'pass1_peak_cycle_mean': float(np.mean(peak_cycle_per_seed)),
        # Pass@5 trajectory
        'pass5_mean': pass5.mean(axis=0).tolist(),
        'pass5_std': pass5.std(axis=0).tolist(),
        # Diversity trajectory
        'diversity_mean': diversity.mean(axis=0).tolist(),
        'diversity_std': diversity.std(axis=0).tolist(),
        'unique_ratio_mean': unique_ratio.mean(axis=0).tolist(),
        # Kill criteria
        'k1_mean_improvement_5cycles': k1_mean_improvement,
        'k1_fraction_improving': k1_pass,
        'k2_collapse_fraction': k2_collapse_fraction,
        'k2_mean_diversity_change': k2_mean_div_change,
        # Convergence
        'convergence_cycle_mean': float(np.mean(convergence_cycles)),
        'convergence_cycle_std': float(np.std(convergence_cycles)),
        # Improvement trajectory
        'improvement_mean': improvement_final,
    }


def run_fresh_data_sweep(config: Config) -> Dict:
    """
    Sweep fresh data fraction to find minimum needed to prevent collapse.
    This is the key practical question: how much fresh data do we need?
    """
    fractions = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = {}

    for frac in fractions:
        trajectories_sft = []
        trajectories_dpo = []
        for seed in range(config.n_seeds):
            t_sft = run_self_learning_loop(config, 'sft', seed, fresh_fraction=frac)
            t_dpo = run_self_learning_loop(config, 'dpo', seed, fresh_fraction=frac)
            trajectories_sft.append(t_sft)
            trajectories_dpo.append(t_dpo)

        results[f'sft_fresh_{frac:.2f}'] = analyze_trajectories(trajectories_sft, f'sft_fresh={frac}')
        results[f'dpo_fresh_{frac:.2f}'] = analyze_trajectories(trajectories_dpo, f'dpo_fresh={frac}')

    return results


def run_k_sweep(config: Config) -> Dict:
    """Sweep number of solutions K to understand sample efficiency."""
    k_values = [1, 3, 5, 10, 20, 50]
    results = {}

    for k in k_values:
        cfg = Config(**{**asdict(config), 'K': k})
        trajectories = []
        for seed in range(config.n_seeds):
            t = run_self_learning_loop(cfg, 'dpo', seed)
            trajectories.append(t)
        results[f'dpo_K={k}'] = analyze_trajectories(trajectories, f'dpo_K={k}')

    return results


# ============================================================
# Main
# ============================================================

def main():
    config = Config()
    results = {}

    print("=" * 70)
    print("EXECUTION-BASED SELF-LEARNING LOOP SIMULATION")
    print("=" * 70)
    print(f"Config: {config.n_problems} problems, K={config.K} solutions/problem, "
          f"{config.n_cycles} cycles, {config.n_seeds} seeds")
    print()

    # ── Experiment 1: SFT vs DPO comparison (pure self-play, no fresh data) ──
    print("Experiment 1: SFT vs DPO (pure self-play)")
    print("-" * 50)

    sft_trajectories = []
    dpo_trajectories = []
    for seed in range(config.n_seeds):
        sft_trajectories.append(run_self_learning_loop(config, 'sft', seed))
        dpo_trajectories.append(run_self_learning_loop(config, 'dpo', seed))

    sft_results = analyze_trajectories(sft_trajectories, 'sft')
    dpo_results = analyze_trajectories(dpo_trajectories, 'dpo')

    results['sft_pure'] = sft_results
    results['dpo_pure'] = dpo_results

    print(f"  SFT: pass@1 {sft_results['pass1_initial_mean']:.3f} -> "
          f"{sft_results['pass1_cycle5_mean']:.3f} (cycle 5) -> "
          f"{sft_results['pass1_final_mean']:.3f} (final)")
    print(f"       peak {sft_results['pass1_peak_mean']:.3f} at cycle "
          f"{sft_results['pass1_peak_cycle_mean']:.1f}, "
          f"convergence at cycle {sft_results['convergence_cycle_mean']:.1f}")
    print(f"       diversity: {sft_results['diversity_mean'][0]:.3f} -> "
          f"{sft_results['diversity_mean'][-1]:.3f} "
          f"({sft_results['k2_mean_diversity_change']*100:.1f}%)")
    print(f"       collapse fraction: {sft_results['k2_collapse_fraction']*100:.1f}%")
    print()
    print(f"  DPO: pass@1 {dpo_results['pass1_initial_mean']:.3f} -> "
          f"{dpo_results['pass1_cycle5_mean']:.3f} (cycle 5) -> "
          f"{dpo_results['pass1_final_mean']:.3f} (final)")
    print(f"       peak {dpo_results['pass1_peak_mean']:.3f} at cycle "
          f"{dpo_results['pass1_peak_cycle_mean']:.1f}, "
          f"convergence at cycle {dpo_results['convergence_cycle_mean']:.1f}")
    print(f"       diversity: {dpo_results['diversity_mean'][0]:.3f} -> "
          f"{dpo_results['diversity_mean'][-1]:.3f} "
          f"({dpo_results['k2_mean_diversity_change']*100:.1f}%)")
    print(f"       collapse fraction: {dpo_results['k2_collapse_fraction']*100:.1f}%")

    # ── Experiment 2: Fresh data mixing sweep ──
    print()
    print("Experiment 2: Fresh data mixing sweep")
    print("-" * 50)

    fresh_results = run_fresh_data_sweep(config)
    results['fresh_sweep'] = fresh_results

    for key, res in fresh_results.items():
        method = key.split('_fresh_')[0]
        frac = key.split('_fresh_')[1]
        print(f"  {method} fresh={frac}: pass@1 final={res['pass1_final_mean']:.3f}, "
              f"diversity={res['diversity_mean'][-1]:.3f}, "
              f"collapse={res['k2_collapse_fraction']*100:.0f}%")

    # ── Experiment 3: K sweep (sample efficiency) ──
    print()
    print("Experiment 3: Solutions per problem (K) sweep")
    print("-" * 50)

    k_results = run_k_sweep(config)
    results['k_sweep'] = k_results

    for key, res in k_results.items():
        print(f"  {key}: pass@1 final={res['pass1_final_mean']:.3f}, "
              f"peak={res['pass1_peak_mean']:.3f} at cycle {res['pass1_peak_cycle_mean']:.1f}")

    # ── Kill Criteria Assessment ──
    print()
    print("=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: pass@1 improves after 5 cycles
    sft_k1_improvement = sft_results['k1_mean_improvement_5cycles']
    dpo_k1_improvement = dpo_results['k1_mean_improvement_5cycles']

    print(f"\nK1: Self-learning loop improves pass@1 after 5 cycles?")
    print(f"  SFT improvement: {sft_k1_improvement*100:.2f}pp -> {'PASS' if sft_k1_improvement > 0 else 'FAIL'}")
    print(f"  DPO improvement: {dpo_k1_improvement*100:.2f}pp -> {'PASS' if dpo_k1_improvement > 0 else 'FAIL'}")

    # K2: training data quality doesn't degrade (model collapse)
    sft_k2 = sft_results['k2_collapse_fraction']
    dpo_k2 = dpo_results['k2_collapse_fraction']

    print(f"\nK2: No model collapse (diversity stays above {config.collapse_diversity_threshold})?")
    print(f"  SFT collapse rate: {sft_k2*100:.1f}% -> {'FAIL' if sft_k2 > 0.5 else 'PASS'}")
    print(f"  DPO collapse rate: {dpo_k2*100:.1f}% -> {'FAIL' if dpo_k2 > 0.5 else 'PASS'}")

    # Find minimum fresh data to prevent collapse
    min_fresh_sft = None
    min_fresh_dpo = None
    for frac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        key_sft = f'sft_fresh_{frac:.2f}'
        key_dpo = f'dpo_fresh_{frac:.2f}'
        if key_sft in fresh_results and fresh_results[key_sft]['k2_collapse_fraction'] == 0 and min_fresh_sft is None:
            min_fresh_sft = frac
        if key_dpo in fresh_results and fresh_results[key_dpo]['k2_collapse_fraction'] == 0 and min_fresh_dpo is None:
            min_fresh_dpo = frac

    print(f"\nMinimum fresh data to prevent collapse:")
    print(f"  SFT: {min_fresh_sft if min_fresh_sft is not None else '>50%'}")
    print(f"  DPO: {min_fresh_dpo if min_fresh_dpo is not None else '>50%'}")

    # DPO advantage
    dpo_advantage = dpo_results['pass1_peak_mean'] - sft_results['pass1_peak_mean']
    print(f"\nDPO advantage over SFT (peak pass@1): {dpo_advantage*100:.2f}pp")

    # Best configuration
    best_key = None
    best_pass1 = 0
    for key, res in fresh_results.items():
        if res['k2_collapse_fraction'] == 0 and res['pass1_peak_mean'] > best_pass1:
            best_pass1 = res['pass1_peak_mean']
            best_key = key

    if best_key:
        print(f"\nBest non-collapsing config: {best_key}")
        print(f"  Peak pass@1: {best_pass1:.3f}")
        print(f"  Final pass@1: {fresh_results[best_key]['pass1_final_mean']:.3f}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    overall_k1 = "PASS" if max(sft_k1_improvement, dpo_k1_improvement) > 0 else "FAIL"
    overall_k2_pure = "FAIL" if max(sft_k2, dpo_k2) > 0.5 else "PASS"

    print(f"K1 (pass@1 improves after 5 cycles): {overall_k1}")
    print(f"  SFT: +{sft_k1_improvement*100:.2f}pp, DPO: +{dpo_k1_improvement*100:.2f}pp")
    print(f"K2 (no model collapse in pure self-play): {overall_k2_pure}")
    print(f"  SFT: {sft_k2*100:.0f}% collapse, DPO: {dpo_k2*100:.0f}% collapse")
    if min_fresh_dpo is not None:
        print(f"K2 (collapse prevented with fresh data): PASS at >={min_fresh_dpo*100:.0f}% fresh (DPO)")

    # Save results
    outdir = Path(__file__).parent

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(outdir / 'results.json', 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to {outdir / 'results.json'}")

    return results


if __name__ == '__main__':
    main()
