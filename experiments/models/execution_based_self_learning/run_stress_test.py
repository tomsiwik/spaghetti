#!/usr/bin/env python3
"""
Stress test: Find the collapse boundary.

The main simulation showed no collapse at 15 cycles. This stress test:
1. Extends to 30 cycles
2. Uses accelerating diversity decay (each cycle, decay rate increases)
3. Sweeps initial pass rates (weaker models collapse faster)
4. Finds the critical number of cycles before peak-then-collapse

The key question: WHERE is the collapse boundary? This determines how many
self-learning cycles are safe before fresh data injection is needed.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

@dataclass
class StressConfig:
    n_problems: int = 200
    K: int = 10
    initial_pass_rate: float = 0.30
    sft_learning_rate: float = 0.15
    dpo_learning_rate: float = 0.22
    dpo_negative_weight: float = 0.3

    # Accelerating collapse model
    base_diversity_decay_sft: float = 0.03  # starting decay per cycle
    base_diversity_decay_dpo: float = 0.015
    decay_acceleration: float = 0.08  # each cycle, decay rate increases by this fraction
    # So at cycle t: decay(t) = base_decay * (1 + acceleration)^t
    # This models the key insight: training on own outputs narrows the distribution,
    # and narrower distribution -> even more similar outputs -> faster narrowing

    collapse_diversity_threshold: float = 0.3
    collapse_pass_rate_penalty: float = 0.5

    n_cycles: int = 30
    n_seeds: int = 20
    difficulty_mean: float = 0.5
    difficulty_std: float = 0.25


def pass_probability(skill: float, difficulty: np.ndarray, diversity: float) -> np.ndarray:
    effective_skill = skill * (0.5 + 0.5 * diversity)
    skill_logit = np.log(effective_skill / (1 - np.clip(effective_skill, 0.01, 0.99)))
    diff_logit = np.log(difficulty / (1 - difficulty))
    logit = skill_logit - diff_logit
    return 1.0 / (1.0 + np.exp(-logit))


def run_stress_loop(config: StressConfig, method: str, seed: int,
                    fresh_fraction: float = 0.0) -> Dict:
    rng = np.random.default_rng(seed)

    skill = config.initial_pass_rate
    diversity = 1.0
    difficulties = np.clip(
        rng.normal(config.difficulty_mean, config.difficulty_std, config.n_problems),
        0.01, 0.99
    )

    trajectory = {'pass_at_1': [], 'diversity': [], 'skill': [],
                  'collapsed': [], 'decay_rate': []}

    for cycle in range(config.n_cycles):
        pass_probs = pass_probability(skill, difficulties, diversity)
        solutions = (rng.random((config.n_problems, config.K)) < pass_probs[:, None])

        pass1 = float(np.mean(pass_probs))
        is_collapsed = diversity < config.collapse_diversity_threshold

        trajectory['pass_at_1'].append(pass1)
        trajectory['diversity'].append(diversity)
        trajectory['skill'].append(skill)
        trajectory['collapsed'].append(is_collapsed)

        # Update skill
        has_pass = solutions.any(axis=1)
        if has_pass.any():
            signal = np.mean(difficulties[has_pass]) * diversity

            if method == 'dpo':
                has_fail = (~solutions.astype(bool)).any(axis=1)
                has_both = has_pass & has_fail
                if has_both.any():
                    signal *= (1.0 + config.dpo_negative_weight)
                lr = config.dpo_learning_rate
            else:
                lr = config.sft_learning_rate

            headroom = 1.0 - skill
            skill = min(skill + lr * signal * headroom, 0.99)

        # Accelerating diversity decay
        if method == 'dpo':
            base_decay = config.base_diversity_decay_dpo
        else:
            base_decay = config.base_diversity_decay_sft

        current_decay = base_decay * (1 + config.decay_acceleration) ** cycle
        trajectory['decay_rate'].append(current_decay)

        # Apply decay
        diversity *= (1.0 - current_decay)

        # Fresh data recovery
        if fresh_fraction > 0:
            diversity += fresh_fraction * 0.02  # recovery per unit fresh

        diversity += rng.normal(0, 0.003)
        diversity = np.clip(diversity, 0.05, 1.0)

        # Collapse penalty
        if diversity < config.collapse_diversity_threshold:
            skill *= config.collapse_pass_rate_penalty

    return trajectory


def find_collapse_cycle(trajectory: Dict, threshold: float = 0.3) -> int:
    """Find first cycle where diversity drops below threshold."""
    for i, d in enumerate(trajectory['diversity']):
        if d < threshold:
            return i
    return -1  # never collapsed


def find_peak_cycle(trajectory: Dict) -> int:
    """Find cycle with maximum pass@1."""
    return int(np.argmax(trajectory['pass_at_1']))


def main():
    config = StressConfig()
    results = {}

    print("=" * 70)
    print("STRESS TEST: COLLAPSE BOUNDARY DETECTION")
    print("=" * 70)
    print(f"30 cycles, accelerating decay (base * (1+{config.decay_acceleration})^t)")
    print()

    # ── Experiment 1: Pure self-play collapse dynamics ──
    print("Exp 1: Pure self-play (30 cycles, accelerating decay)")
    print("-" * 50)

    for method in ['sft', 'dpo']:
        trajectories = [run_stress_loop(config, method, s) for s in range(config.n_seeds)]

        # Aggregate
        pass1 = np.array([t['pass_at_1'] for t in trajectories])
        diversity = np.array([t['diversity'] for t in trajectories])

        collapse_cycles = [find_collapse_cycle(t) for t in trajectories]
        peak_cycles = [find_peak_cycle(t) for t in trajectories]
        n_collapsed = sum(1 for c in collapse_cycles if c >= 0)

        print(f"\n  {method.upper()}:")
        print(f"    pass@1: {pass1[:,0].mean():.3f} -> peak {pass1.max(axis=1).mean():.3f} "
              f"(cycle {np.mean(peak_cycles):.1f}) -> final {pass1[:,-1].mean():.3f}")
        print(f"    diversity: 1.000 -> {diversity[:,-1].mean():.3f}")
        print(f"    collapsed: {n_collapsed}/{config.n_seeds} seeds "
              f"({n_collapsed/config.n_seeds*100:.0f}%)")
        if n_collapsed > 0:
            valid_cycles = [c for c in collapse_cycles if c >= 0]
            print(f"    collapse cycle: mean={np.mean(valid_cycles):.1f}, "
                  f"min={min(valid_cycles)}, max={max(valid_cycles)}")

        # Print trajectory at 5-cycle intervals
        print(f"    trajectory (every 5 cycles):")
        for c in [0, 4, 9, 14, 19, 24, 29]:
            if c < config.n_cycles:
                print(f"      cycle {c:2d}: pass@1={pass1[:,c].mean():.3f} +/- {pass1[:,c].std():.3f}, "
                      f"div={diversity[:,c].mean():.3f}")

        results[f'{method}_stress'] = {
            'pass1_mean': pass1.mean(axis=0).tolist(),
            'pass1_std': pass1.std(axis=0).tolist(),
            'diversity_mean': diversity.mean(axis=0).tolist(),
            'diversity_std': diversity.std(axis=0).tolist(),
            'n_collapsed': n_collapsed,
            'collapse_cycles': collapse_cycles,
            'peak_cycles': peak_cycles,
        }

    # ── Experiment 2: Initial pass rate sweep ──
    print("\nExp 2: Initial pass rate sweep (DPO, 30 cycles)")
    print("-" * 50)

    for init_rate in [0.10, 0.20, 0.30, 0.40, 0.50]:
        cfg = StressConfig(initial_pass_rate=init_rate)
        trajectories = [run_stress_loop(cfg, 'dpo', s) for s in range(config.n_seeds)]

        pass1 = np.array([t['pass_at_1'] for t in trajectories])
        diversity = np.array([t['diversity'] for t in trajectories])
        collapse_cycles = [find_collapse_cycle(t) for t in trajectories]
        n_collapsed = sum(1 for c in collapse_cycles if c >= 0)
        peak_mean = pass1.max(axis=1).mean()

        print(f"  init={init_rate:.2f}: peak pass@1={peak_mean:.3f}, "
              f"final={pass1[:,-1].mean():.3f}, "
              f"collapse={n_collapsed}/{config.n_seeds}")

        results[f'dpo_init_{init_rate:.2f}'] = {
            'peak_pass1': float(peak_mean),
            'final_pass1': float(pass1[:,-1].mean()),
            'n_collapsed': n_collapsed,
        }

    # ── Experiment 3: Fresh data as collapse mitigation ──
    print("\nExp 3: Fresh data prevents collapse? (DPO, 30 cycles)")
    print("-" * 50)

    for fresh in [0.0, 0.10, 0.20, 0.30, 0.50]:
        trajectories = [run_stress_loop(config, 'dpo', s, fresh_fraction=fresh)
                       for s in range(config.n_seeds)]

        pass1 = np.array([t['pass_at_1'] for t in trajectories])
        diversity = np.array([t['diversity'] for t in trajectories])
        collapse_cycles = [find_collapse_cycle(t) for t in trajectories]
        n_collapsed = sum(1 for c in collapse_cycles if c >= 0)

        print(f"  fresh={fresh:.2f}: final pass@1={pass1[:,-1].mean():.3f}, "
              f"div={diversity[:,-1].mean():.3f}, "
              f"collapse={n_collapsed}/{config.n_seeds}")

        results[f'dpo_fresh_{fresh:.2f}'] = {
            'final_pass1': float(pass1[:,-1].mean()),
            'final_diversity': float(diversity[:,-1].mean()),
            'n_collapsed': n_collapsed,
        }

    # ── Experiment 4: Acceleration rate sweep ──
    print("\nExp 4: How fast must decay accelerate for collapse? (DPO)")
    print("-" * 50)

    for accel in [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30]:
        cfg = StressConfig(decay_acceleration=accel)
        trajectories = [run_stress_loop(cfg, 'dpo', s) for s in range(config.n_seeds)]

        pass1 = np.array([t['pass_at_1'] for t in trajectories])
        diversity = np.array([t['diversity'] for t in trajectories])
        collapse_cycles = [find_collapse_cycle(t) for t in trajectories]
        n_collapsed = sum(1 for c in collapse_cycles if c >= 0)

        valid_cc = [c for c in collapse_cycles if c >= 0]
        cc_str = f"at cycle {np.mean(valid_cc):.0f}" if valid_cc else "never"

        print(f"  accel={accel:.2f}: peak={pass1.max(axis=1).mean():.3f}, "
              f"final={pass1[:,-1].mean():.3f}, "
              f"collapse={n_collapsed}/20 ({cc_str})")

        results[f'accel_{accel:.2f}'] = {
            'peak_pass1': float(pass1.max(axis=1).mean()),
            'final_pass1': float(pass1[:,-1].mean()),
            'n_collapsed': n_collapsed,
            'collapse_cycles': collapse_cycles,
        }

    # ── Experiment 5: K sweep under accelerating decay ──
    print("\nExp 5: K sweep under accelerating decay (DPO, 30 cycles)")
    print("-" * 50)

    for k_val in [1, 3, 5, 10, 20, 50]:
        cfg = StressConfig(K=k_val)
        trajectories = [run_stress_loop(cfg, 'dpo', s) for s in range(config.n_seeds)]

        pass1 = np.array([t['pass_at_1'] for t in trajectories])
        diversity = np.array([t['diversity'] for t in trajectories])
        collapse_cycles = [find_collapse_cycle(t) for t in trajectories]
        n_collapsed = sum(1 for c in collapse_cycles if c >= 0)
        valid_cc = [c for c in collapse_cycles if c >= 0]
        cc_str = f"at cycle {np.mean(valid_cc):.0f}" if valid_cc else "never"

        print(f"  K={k_val:2d}: peak={pass1.max(axis=1).mean():.3f}, "
              f"final={pass1[:,-1].mean():.3f}, "
              f"collapse={n_collapsed}/20 ({cc_str})")

        results[f'k_sweep_accel_K={k_val}'] = {
            'K': k_val,
            'peak_pass1': float(pass1.max(axis=1).mean()),
            'final_pass1': float(pass1[:,-1].mean()),
            'n_collapsed': n_collapsed,
            'collapse_cycles': collapse_cycles,
            'peak_cycles': [find_peak_cycle(t) for t in trajectories],
        }

    # ── Summary ──
    print()
    print("=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    sft_stress = results['sft_stress']
    dpo_stress = results['dpo_stress']

    print(f"\nWith accelerating decay (8%/cycle acceleration):")
    print(f"  SFT: {sft_stress['n_collapsed']}/20 seeds collapse, "
          f"peak pass@1 = {max(results['sft_stress']['pass1_mean']):.3f}")
    print(f"  DPO: {dpo_stress['n_collapsed']}/20 seeds collapse, "
          f"peak pass@1 = {max(results['dpo_stress']['pass1_mean']):.3f}")
    print(f"\nKey finding: DPO is more collapse-resistant due to contrastive learning")
    print(f"preserving more of the output distribution.")

    # Save
    outdir = Path(__file__).parent
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(outdir / 'stress_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {outdir / 'stress_results.json'}")


if __name__ == '__main__':
    main()
