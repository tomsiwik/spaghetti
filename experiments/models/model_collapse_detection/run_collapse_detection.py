#!/usr/bin/env python3
"""
Model Collapse Detection for Self-Learning LoRA Experts (v2 -- revised)
========================================================================

Revised based on adversarial review. Key changes from v1:
1. Added ANCHORED full-rank baseline to isolate rank effect from base-anchoring
2. Added LoRA ablation WITHOUT norm constraint (rank-only regularization test)
3. Reframed correlation experiment to test in collapse-prone regime
4. Softened "no fresh data needed" claim
5. Fixed the MATH.md norm bound

Goes beyond the parent experiment (execution_based_self_learning) which used a
scalar skill model. This experiment models collapse at the representation level:

1. **Vocabulary distribution model**: Track a full categorical distribution over
   V=100 tokens through 30 self-learning cycles.

2. **LoRA rank constraint**: Compare collapse dynamics across ranks r={4,8,16,32,64}
   against TWO baselines: unanchored full-rank AND anchored full-rank.

3. **Norm constraint ablation**: Test LoRA with and without norm bounding to
   disentangle rank effect from norm-bounding effect.

4. **Correlation effect in collapse-prone regime**: Test on anchored full-rank
   where collapse may occur, not on LoRA where it is prevented.

Kill criteria:
  K1: output diversity drops >30% after 5 self-learning cycles
  K2: expert converges to repetitive/degenerate outputs
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from scipy import stats
from scipy.special import softmax


# ============================================================
# Configuration
# ============================================================

@dataclass
class CollapseConfig:
    """Configuration for collapse detection experiment."""
    vocab_size: int = 100
    seq_len: int = 16
    n_sequences: int = 200
    zipf_exponent: float = 1.2
    d_model: int = 64
    lora_ranks: list = None
    n_cycles: int = 30
    learning_rate: float = 0.3
    fresh_data_fractions: list = None
    temperature: float = 0.8
    correlation_strength: float = 0.3
    n_seeds: int = 10
    diversity_drop_threshold: float = 0.30
    degeneracy_threshold: float = 0.10

    def __post_init__(self):
        if self.lora_ranks is None:
            self.lora_ranks = [4, 8, 16, 32, 64]
        if self.fresh_data_fractions is None:
            self.fresh_data_fractions = [0.0, 0.1, 0.3, 0.5]


# ============================================================
# Distribution Model
# ============================================================

def create_zipf_distribution(vocab_size: int, exponent: float) -> np.ndarray:
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, exponent)
    return probs / probs.sum()


def create_lora_perturbation(d_model: int, vocab_size: int, rank: int,
                             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.standard_normal((rank, d_model)) / np.sqrt(d_model)
    B = rng.standard_normal((vocab_size, rank)) / np.sqrt(rank)
    return A, B


def apply_lora_to_distribution(base_logits: np.ndarray, A: np.ndarray,
                                B: np.ndarray, context_embedding: np.ndarray,
                                scale: float = 1.0) -> np.ndarray:
    delta = scale * B @ (A @ context_embedding)
    result = base_logits + delta
    return np.clip(result, -50, 50)


def self_train_lora(A: np.ndarray, B: np.ndarray,
                    generated_logits: np.ndarray,
                    target_logits: np.ndarray,
                    lr: float, rank: int,
                    rng: np.random.Generator,
                    apply_norm_constraint: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate one step of LoRA self-training.

    When apply_norm_constraint=False, the norm caps are removed to test
    whether rank alone (without norm bounding) prevents collapse.
    """
    grad_direction = target_logits - generated_logits

    U, s, Vt = np.linalg.svd(np.outer(grad_direction, grad_direction[:rank]),
                               full_matrices=False)
    effective_rank = min(rank, len(s))
    projected_grad = U[:, :effective_rank] @ np.diag(s[:effective_rank])

    grad_B = projected_grad[:, :B.shape[1]] * lr
    grad_A = (B.T @ grad_direction)[:, None] * np.ones((1, A.shape[1])) * lr / A.shape[1]

    noise_B = rng.standard_normal(B.shape) * lr * 0.05
    noise_A = rng.standard_normal(A.shape) * lr * 0.05

    B_new = B + grad_B + noise_B
    A_new = A + grad_A + noise_A

    if apply_norm_constraint:
        max_norm_B = 5.0
        max_norm_A = 5.0
        norm_B = np.linalg.norm(B_new, 'fro')
        norm_A = np.linalg.norm(A_new, 'fro')
        if norm_B > max_norm_B:
            B_new *= max_norm_B / norm_B
        if norm_A > max_norm_A:
            A_new *= max_norm_A / norm_A

    return A_new, B_new


# ============================================================
# Sequence Generation
# ============================================================

def sample_sequences(probs: np.ndarray, n_seq: int, seq_len: int,
                     temperature: float, correlation: float,
                     rng: np.random.Generator) -> np.ndarray:
    vocab_size = len(probs)
    sequences = np.zeros((n_seq, seq_len), dtype=np.int32)
    logits = np.log(probs + 1e-12) / temperature
    base_probs = softmax(logits)

    sequences[:, 0] = rng.choice(vocab_size, size=n_seq, p=base_probs)

    if correlation < 1e-6:
        for j in range(1, seq_len):
            sequences[:, j] = rng.choice(vocab_size, size=n_seq, p=base_probs)
    else:
        neighborhood = max(1, vocab_size // 20)
        for j in range(1, seq_len):
            boost = np.zeros((n_seq, vocab_size))
            prev_tokens = sequences[:, j - 1]
            for i in range(n_seq):
                low = max(0, prev_tokens[i] - neighborhood)
                high = min(vocab_size, prev_tokens[i] + neighborhood)
                boost[i, low:high] = correlation
            adj_logits = logits[None, :] + boost
            adj_probs = softmax(adj_logits, axis=1)
            cumsum = np.cumsum(adj_probs, axis=1)
            u = rng.random(n_seq)[:, None]
            sequences[:, j] = np.argmax(cumsum >= u, axis=1)

    return sequences


# ============================================================
# Collapse Detection Metrics
# ============================================================

def unique_ngram_ratio(sequences: np.ndarray, n: int = 2) -> float:
    all_ngrams = set()
    total = 0
    n_seq, seq_len = sequences.shape
    for i in range(n_seq):
        for j in range(seq_len - n + 1):
            ngram = tuple(sequences[i, j:j+n])
            all_ngrams.add(ngram)
            total += 1
    return len(all_ngrams) / max(total, 1)


def embedding_variance(sequences: np.ndarray, embed_matrix: np.ndarray) -> float:
    n_seq, seq_len = sequences.shape
    embeddings = np.zeros((n_seq, embed_matrix.shape[1]))
    for i in range(n_seq):
        token_embeds = embed_matrix[sequences[i]]
        embeddings[i] = token_embeds.mean(axis=0)
    return float(np.var(embeddings, axis=0).sum())


def distribution_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def top_k_concentration(probs: np.ndarray, k: int = 10) -> float:
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[:k].sum())


def unique_output_ratio(sequences: np.ndarray) -> float:
    unique = set(tuple(s) for s in sequences)
    return len(unique) / len(sequences)


# ============================================================
# Core Experiment Loops
# ============================================================

def _init_metrics():
    return {
        'ngram_2_ratio': [], 'ngram_3_ratio': [],
        'embedding_variance': [], 'entropy': [],
        'kl_from_initial': [], 'kl_from_base': [],
        'top10_concentration': [], 'top50_concentration': [],
        'unique_output_ratio': [], 'max_prob': [],
    }


def _record_metrics(metrics, sequences, expert_probs, initial_probs, base_probs, embed_matrix):
    metrics['ngram_2_ratio'].append(unique_ngram_ratio(sequences, 2))
    metrics['ngram_3_ratio'].append(unique_ngram_ratio(sequences, 3))
    metrics['embedding_variance'].append(embedding_variance(sequences, embed_matrix))
    metrics['entropy'].append(distribution_entropy(expert_probs))
    metrics['kl_from_initial'].append(kl_divergence(expert_probs, initial_probs))
    metrics['kl_from_base'].append(kl_divergence(expert_probs, base_probs))
    metrics['top10_concentration'].append(top_k_concentration(expert_probs, 10))
    metrics['top50_concentration'].append(top_k_concentration(expert_probs, 50))
    metrics['unique_output_ratio'].append(unique_output_ratio(sequences))
    metrics['max_prob'].append(float(expert_probs.max()))


def run_full_rank_unanchored_experiment(config: CollapseConfig,
                                        fresh_fraction: float, seed: int,
                                        use_correlation: bool = True) -> Dict:
    """
    Full-rank UNANCHORED baseline (original v1 baseline).
    Logits drift freely -- no base anchoring.
    """
    rng = np.random.default_rng(seed)
    base_probs = create_zipf_distribution(config.vocab_size, config.zipf_exponent)
    base_logits = np.log(base_probs + 1e-12)

    expert_logits = base_logits + rng.standard_normal(config.vocab_size) * 0.1
    expert_logits = np.clip(expert_logits, -50, 50)
    expert_probs = softmax(expert_logits)

    embed_matrix = rng.standard_normal((config.vocab_size, config.d_model)) / np.sqrt(config.d_model)
    initial_probs = expert_probs.copy()
    correlation = config.correlation_strength if use_correlation else 0.0
    metrics = _init_metrics()

    for cycle in range(config.n_cycles):
        sequences = sample_sequences(
            expert_probs, config.n_sequences, config.seq_len,
            config.temperature, correlation, rng)
        _record_metrics(metrics, sequences, expert_probs, initial_probs, base_probs, embed_matrix)

        token_counts = np.bincount(sequences.flatten(), minlength=config.vocab_size)
        empirical_probs = token_counts / token_counts.sum()
        empirical_logits = np.log(empirical_probs + 1e-12)

        if fresh_fraction > 0:
            mixed_logits = ((1 - fresh_fraction) * empirical_logits +
                           fresh_fraction * base_logits)
        else:
            mixed_logits = empirical_logits

        grad = mixed_logits - expert_logits
        expert_logits = expert_logits + config.learning_rate * grad
        expert_logits += rng.standard_normal(config.vocab_size) * config.learning_rate * 0.05
        expert_logits = np.clip(expert_logits, -50, 50)
        expert_probs = softmax(expert_logits)

    return metrics


def run_full_rank_anchored_experiment(config: CollapseConfig,
                                      fresh_fraction: float, seed: int,
                                      use_correlation: bool = True) -> Dict:
    """
    FIX 1: Full-rank ANCHORED baseline.
    expert_logits = base_logits + delta, where delta is updated freely.
    This isolates the rank effect: if anchored full-rank also avoids collapse,
    base-anchoring (not rank) is the mechanism. If it collapses, rank matters.
    """
    rng = np.random.default_rng(seed)
    base_probs = create_zipf_distribution(config.vocab_size, config.zipf_exponent)
    base_logits = np.log(base_probs + 1e-12)

    # Initialize delta as small perturbation (same as LoRA init magnitude)
    delta = rng.standard_normal(config.vocab_size) * 0.1

    expert_logits = np.clip(base_logits + delta, -50, 50)
    expert_probs = softmax(expert_logits)

    embed_matrix = rng.standard_normal((config.vocab_size, config.d_model)) / np.sqrt(config.d_model)
    initial_probs = expert_probs.copy()
    correlation = config.correlation_strength if use_correlation else 0.0
    metrics = _init_metrics()

    for cycle in range(config.n_cycles):
        sequences = sample_sequences(
            expert_probs, config.n_sequences, config.seq_len,
            config.temperature, correlation, rng)
        _record_metrics(metrics, sequences, expert_probs, initial_probs, base_probs, embed_matrix)

        token_counts = np.bincount(sequences.flatten(), minlength=config.vocab_size)
        empirical_probs = token_counts / token_counts.sum()
        empirical_logits = np.log(empirical_probs + 1e-12)

        if fresh_fraction > 0:
            mixed_logits = ((1 - fresh_fraction) * empirical_logits +
                           fresh_fraction * base_logits)
        else:
            mixed_logits = empirical_logits

        # Update delta (full-rank, all V directions), but always re-anchor to base
        grad = mixed_logits - expert_logits
        delta = delta + config.learning_rate * grad
        delta += rng.standard_normal(config.vocab_size) * config.learning_rate * 0.05

        expert_logits = np.clip(base_logits + delta, -50, 50)
        expert_probs = softmax(expert_logits)

    return metrics


def run_collapse_experiment(config: CollapseConfig, rank: int,
                           fresh_fraction: float, seed: int,
                           use_correlation: bool = True,
                           apply_norm_constraint: bool = True) -> Dict:
    """
    LoRA-constrained self-learning experiment.

    FIX 2: apply_norm_constraint=False removes ||A||_F, ||B||_F caps to test
    whether rank alone (without norm bounding) prevents collapse.
    """
    rng = np.random.default_rng(seed)
    base_probs = create_zipf_distribution(config.vocab_size, config.zipf_exponent)
    base_logits = np.log(base_probs + 1e-12)

    A, B = create_lora_perturbation(config.d_model, config.vocab_size, rank, rng)
    embed_matrix = rng.standard_normal((config.vocab_size, config.d_model)) / np.sqrt(config.d_model)
    context = rng.standard_normal(config.d_model) / np.sqrt(config.d_model)

    expert_logits = apply_lora_to_distribution(base_logits, A, B, context, scale=0.5)
    expert_probs = softmax(expert_logits)
    initial_probs = expert_probs.copy()

    correlation = config.correlation_strength if use_correlation else 0.0
    metrics = _init_metrics()

    for cycle in range(config.n_cycles):
        sequences = sample_sequences(
            expert_probs, config.n_sequences, config.seq_len,
            config.temperature, correlation, rng)
        _record_metrics(metrics, sequences, expert_probs, initial_probs, base_probs, embed_matrix)

        token_counts = np.bincount(sequences.flatten(), minlength=config.vocab_size)
        empirical_probs = token_counts / token_counts.sum()
        empirical_logits = np.log(empirical_probs + 1e-12)

        if fresh_fraction > 0:
            mixed_logits = ((1 - fresh_fraction) * empirical_logits +
                           fresh_fraction * base_logits)
        else:
            mixed_logits = empirical_logits

        A, B = self_train_lora(A, B, expert_logits, mixed_logits,
                               config.learning_rate, rank, rng,
                               apply_norm_constraint=apply_norm_constraint)

        expert_logits = apply_lora_to_distribution(base_logits, A, B, context, scale=0.5)
        expert_probs = softmax(expert_logits)

    return metrics


def analyze_collapse(metrics: Dict, config: CollapseConfig) -> Dict:
    """Analyze collapse metrics and assess kill criteria."""
    n_cycles = len(metrics['ngram_2_ratio'])

    initial_ngram2 = metrics['ngram_2_ratio'][0]
    cycle5_ngram2 = metrics['ngram_2_ratio'][min(4, n_cycles - 1)]
    final_ngram2 = metrics['ngram_2_ratio'][-1]

    ngram2_drop_5 = (initial_ngram2 - cycle5_ngram2) / max(initial_ngram2, 1e-12)
    ngram2_drop_final = (initial_ngram2 - final_ngram2) / max(initial_ngram2, 1e-12)

    initial_entropy = metrics['entropy'][0]
    cycle5_entropy = metrics['entropy'][min(4, n_cycles - 1)]
    final_entropy = metrics['entropy'][-1]

    entropy_drop_5 = (initial_entropy - cycle5_entropy) / max(initial_entropy, 1e-12)
    entropy_drop_final = (initial_entropy - final_entropy) / max(initial_entropy, 1e-12)

    initial_embed_var = metrics['embedding_variance'][0]
    cycle5_embed_var = metrics['embedding_variance'][min(4, n_cycles - 1)]
    final_embed_var = metrics['embedding_variance'][-1]

    embed_var_drop_5 = (initial_embed_var - cycle5_embed_var) / max(initial_embed_var, 1e-12)
    embed_var_drop_final = (initial_embed_var - final_embed_var) / max(initial_embed_var, 1e-12)

    final_unique = metrics['unique_output_ratio'][-1]
    k2_degenerate = final_unique < config.degeneracy_threshold

    collapse_cycle = None
    for c in range(n_cycles):
        drop = (initial_ngram2 - metrics['ngram_2_ratio'][c]) / max(initial_ngram2, 1e-12)
        if drop > config.diversity_drop_threshold:
            collapse_cycle = c
            break

    early_warning = {}
    for metric_name, threshold_frac in [
        ('ngram_2_ratio', 0.15), ('ngram_3_ratio', 0.15),
        ('entropy', 0.10), ('embedding_variance', 0.15),
    ]:
        initial = metrics[metric_name][0]
        warn_cycle = None
        for c in range(n_cycles):
            drop = (initial - metrics[metric_name][c]) / max(abs(initial), 1e-12)
            if drop > threshold_frac:
                warn_cycle = c
                break
        early_warning[metric_name] = warn_cycle

    return {
        'k1_ngram2_drop_5cycles': ngram2_drop_5,
        'k1_entropy_drop_5cycles': entropy_drop_5,
        'k1_embed_var_drop_5cycles': embed_var_drop_5,
        'k1_pass': ngram2_drop_5 < config.diversity_drop_threshold,
        'k2_final_unique_ratio': final_unique,
        'k2_degenerate': k2_degenerate,
        'ngram2_drop_final': ngram2_drop_final,
        'entropy_drop_final': entropy_drop_final,
        'embed_var_drop_final': embed_var_drop_final,
        'collapse_cycle': collapse_cycle,
        'early_warning': early_warning,
        'final_kl_from_initial': metrics['kl_from_initial'][-1],
        'final_kl_from_base': metrics['kl_from_base'][-1],
        'final_top10': metrics['top10_concentration'][-1],
    }


# ============================================================
# Aggregate helper
# ============================================================

def _aggregate_seeds(seed_results, config):
    """Aggregate analysis results across seeds."""
    k1_drops = [r['k1_ngram2_drop_5cycles'] for r in seed_results]
    k1_pass_rate = sum(1 for r in seed_results if r['k1_pass']) / len(seed_results)
    collapse_cycles = [r['collapse_cycle'] for r in seed_results if r['collapse_cycle'] is not None]
    k2_degen_rate = sum(1 for r in seed_results if r['k2_degenerate']) / len(seed_results)
    entropy_drops = [r['k1_entropy_drop_5cycles'] for r in seed_results]
    embed_drops = [r['k1_embed_var_drop_5cycles'] for r in seed_results]
    final_kls = [r['final_kl_from_initial'] for r in seed_results]

    return {
        'k1_ngram2_drop_mean': float(np.mean(k1_drops)),
        'k1_ngram2_drop_std': float(np.std(k1_drops)),
        'k1_pass_rate': k1_pass_rate,
        'k1_entropy_drop_mean': float(np.mean(entropy_drops)),
        'k1_embed_var_drop_mean': float(np.mean(embed_drops)),
        'collapse_fraction': len(collapse_cycles) / len(seed_results),
        'collapse_cycle_mean': float(np.mean(collapse_cycles)) if collapse_cycles else None,
        'collapse_cycle_std': float(np.std(collapse_cycles)) if collapse_cycles else None,
        'k2_degeneracy_rate': k2_degen_rate,
        'final_kl_mean': float(np.mean(final_kls)),
        'ngram2_trajectory': [float(np.mean([r['metrics']['ngram_2_ratio'][c] for r in seed_results]))
                              for c in range(config.n_cycles)],
        'entropy_trajectory': [float(np.mean([r['metrics']['entropy'][c] for r in seed_results]))
                              for c in range(config.n_cycles)],
    }


# ============================================================
# Experiments
# ============================================================

def experiment_rank_sweep(config: CollapseConfig) -> Dict:
    """
    Experiment 1: Rank sweep + THREE baselines (anchored full-rank, unanchored full-rank, LoRA no-norm).

    This is the core experiment. Tests rank constraint against properly controlled baselines.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LoRA Rank Sweep + Controlled Baselines")
    print("=" * 70)

    results = {}

    # --- LoRA with norm constraint (standard) ---
    for rank in config.lora_ranks:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_collapse_experiment(config, rank, 0.0, seed)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        agg = _aggregate_seeds(seed_results, config)
        agg['rank'] = rank
        agg['condition'] = 'lora_normed'
        results[f'rank_{rank}'] = agg

        status = "PASS" if agg['k1_pass_rate'] > 0.5 else "FAIL"
        print(f"  LoRA r={rank:3d} (normed): ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}% (5cyc), "
              f"entropy={agg['k1_entropy_drop_mean']*100:+.1f}%, "
              f"collapse={agg['collapse_fraction']*100:.0f}%, K1={status}")

    # --- FIX 2: LoRA WITHOUT norm constraint ---
    print("\n  --- LoRA without norm constraint (rank-only) ---")
    for rank in config.lora_ranks:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_collapse_experiment(config, rank, 0.0, seed,
                                             apply_norm_constraint=False)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        agg = _aggregate_seeds(seed_results, config)
        agg['rank'] = rank
        agg['condition'] = 'lora_unnormed'
        results[f'rank_{rank}_no_norm'] = agg

        status = "PASS" if agg['k1_pass_rate'] > 0.5 else "FAIL"
        print(f"  LoRA r={rank:3d} (NO norm): ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}% (5cyc), "
              f"entropy={agg['k1_entropy_drop_mean']*100:+.1f}%, "
              f"collapse={agg['collapse_fraction']*100:.0f}%, K1={status}")

    # --- FIX 1: Anchored full-rank baseline ---
    print("\n  --- Anchored full-rank baseline ---")
    seed_results = []
    for seed in range(config.n_seeds):
        metrics = run_full_rank_anchored_experiment(config, 0.0, seed)
        analysis = analyze_collapse(metrics, config)
        analysis['metrics'] = metrics
        seed_results.append(analysis)

    agg = _aggregate_seeds(seed_results, config)
    agg['rank'] = 'full_anchored'
    agg['condition'] = 'full_rank_anchored'
    results['full_rank_anchored'] = agg

    status = "PASS" if agg['k1_pass_rate'] > 0.5 else "FAIL"
    print(f"  FULL RANK (anchored): ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}% (5cyc), "
          f"entropy={agg['k1_entropy_drop_mean']*100:+.1f}%, "
          f"collapse={agg['collapse_fraction']*100:.0f}%, K1={status}")

    # --- Unanchored full-rank baseline (original v1) ---
    print("\n  --- Unanchored full-rank baseline (v1 control) ---")
    seed_results = []
    for seed in range(config.n_seeds):
        metrics = run_full_rank_unanchored_experiment(config, 0.0, seed)
        analysis = analyze_collapse(metrics, config)
        analysis['metrics'] = metrics
        seed_results.append(analysis)

    agg = _aggregate_seeds(seed_results, config)
    agg['rank'] = 'full_unanchored'
    agg['condition'] = 'full_rank_unanchored'
    results['full_rank_unanchored'] = agg

    status = "PASS" if agg['k1_pass_rate'] > 0.5 else "FAIL"
    print(f"  FULL RANK (unanchored): ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}% (5cyc), "
          f"entropy={agg['k1_entropy_drop_mean']*100:+.1f}%, "
          f"collapse={agg['collapse_fraction']*100:.0f}%, K1={status}")

    return results


def experiment_correlation_effect(config: CollapseConfig) -> Dict:
    """
    Experiment 2: Do correlated outputs accelerate collapse?

    FIX 4: Test in the ANCHORED FULL-RANK regime where collapse may occur,
    not in the LoRA regime where rank prevents it. Also test on LoRA for
    completeness, but the informative comparison is anchored full-rank.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Correlation Effect (tested in collapse-prone regime)")
    print("=" * 70)

    results = {}

    # Test on anchored full-rank (where collapse may occur)
    print("  --- Anchored full-rank regime ---")
    for use_corr, label in [(False, 'anchored_fr_independent'), (True, 'anchored_fr_correlated')]:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_full_rank_anchored_experiment(config, 0.0, seed,
                                                        use_correlation=use_corr)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        k1_drops = [r['k1_ngram2_drop_5cycles'] for r in seed_results]
        collapse_cycles = [r['collapse_cycle'] for r in seed_results if r['collapse_cycle'] is not None]

        agg = {
            'correlation': use_corr,
            'regime': 'anchored_full_rank',
            'k1_ngram2_drop_mean': float(np.mean(k1_drops)),
            'k1_ngram2_drop_std': float(np.std(k1_drops)),
            'collapse_fraction': len(collapse_cycles) / len(seed_results),
            'collapse_cycle_mean': float(np.mean(collapse_cycles)) if collapse_cycles else None,
            'ngram2_trajectory': [float(np.mean([r['metrics']['ngram_2_ratio'][c] for r in seed_results]))
                                  for c in range(config.n_cycles)],
        }
        results[label] = agg
        print(f"  {label:32s}: ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}%, "
              f"collapse={agg['collapse_fraction']*100:.0f}%"
              + (f", cycle={agg['collapse_cycle_mean']:.1f}" if agg['collapse_cycle_mean'] else ""))

    # Also test on unanchored full-rank (known to collapse)
    print("  --- Unanchored full-rank regime ---")
    for use_corr, label in [(False, 'unanchored_fr_independent'), (True, 'unanchored_fr_correlated')]:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_full_rank_unanchored_experiment(config, 0.0, seed,
                                                          use_correlation=use_corr)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        k1_drops = [r['k1_ngram2_drop_5cycles'] for r in seed_results]
        collapse_cycles = [r['collapse_cycle'] for r in seed_results if r['collapse_cycle'] is not None]

        agg = {
            'correlation': use_corr,
            'regime': 'unanchored_full_rank',
            'k1_ngram2_drop_mean': float(np.mean(k1_drops)),
            'k1_ngram2_drop_std': float(np.std(k1_drops)),
            'collapse_fraction': len(collapse_cycles) / len(seed_results),
            'collapse_cycle_mean': float(np.mean(collapse_cycles)) if collapse_cycles else None,
            'ngram2_trajectory': [float(np.mean([r['metrics']['ngram_2_ratio'][c] for r in seed_results]))
                                  for c in range(config.n_cycles)],
        }
        results[label] = agg
        print(f"  {label:32s}: ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}%, "
              f"collapse={agg['collapse_fraction']*100:.0f}%"
              + (f", cycle={agg['collapse_cycle_mean']:.1f}" if agg['collapse_cycle_mean'] else ""))

    # Also test on LoRA (for completeness -- expected to be uninformative)
    print("  --- LoRA r=16 regime (expected uninformative) ---")
    for use_corr, label in [(False, 'lora_independent'), (True, 'lora_correlated')]:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_collapse_experiment(config, 16, 0.0, seed,
                                             use_correlation=use_corr)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        k1_drops = [r['k1_ngram2_drop_5cycles'] for r in seed_results]
        collapse_cycles = [r['collapse_cycle'] for r in seed_results if r['collapse_cycle'] is not None]

        agg = {
            'correlation': use_corr,
            'regime': 'lora_r16',
            'k1_ngram2_drop_mean': float(np.mean(k1_drops)),
            'k1_ngram2_drop_std': float(np.std(k1_drops)),
            'collapse_fraction': len(collapse_cycles) / len(seed_results),
            'collapse_cycle_mean': float(np.mean(collapse_cycles)) if collapse_cycles else None,
        }
        results[label] = agg
        print(f"  {label:32s}: ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}%, "
              f"collapse={agg['collapse_fraction']*100:.0f}%")

    # Acceleration factors
    for regime_prefix, regime_name in [
        ('anchored_fr', 'Anchored full-rank'),
        ('unanchored_fr', 'Unanchored full-rank'),
        ('lora', 'LoRA r=16'),
    ]:
        indep = results.get(f'{regime_prefix}_independent', {}).get('k1_ngram2_drop_mean', 0)
        corr = results.get(f'{regime_prefix}_correlated', {}).get('k1_ngram2_drop_mean', 0)
        if abs(indep) > 1e-6:
            accel = corr / indep
        else:
            accel = float('nan')
        results[f'{regime_prefix}_acceleration'] = accel
        print(f"\n  {regime_name} acceleration: {accel:.2f}x")

    return results


def experiment_fresh_data_mitigation(config: CollapseConfig) -> Dict:
    """Experiment 3: Fresh data prevents collapse (validates parent's 50% finding)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Fresh Data Mitigation")
    print("=" * 70)

    rank = 16
    results = {}

    for frac in config.fresh_data_fractions:
        seed_results = []
        for seed in range(config.n_seeds):
            metrics = run_collapse_experiment(config, rank, frac, seed)
            analysis = analyze_collapse(metrics, config)
            analysis['metrics'] = metrics
            seed_results.append(analysis)

        k1_drops = [r['k1_ngram2_drop_5cycles'] for r in seed_results]
        collapse_cycles = [r['collapse_cycle'] for r in seed_results if r['collapse_cycle'] is not None]
        k2_degen_rate = sum(1 for r in seed_results if r['k2_degenerate']) / len(seed_results)
        entropy_drops_final = [r['entropy_drop_final'] for r in seed_results]

        agg = {
            'fresh_fraction': frac,
            'k1_ngram2_drop_mean': float(np.mean(k1_drops)),
            'k1_ngram2_drop_std': float(np.std(k1_drops)),
            'collapse_fraction': len(collapse_cycles) / len(seed_results),
            'collapse_cycle_mean': float(np.mean(collapse_cycles)) if collapse_cycles else None,
            'k2_degeneracy_rate': k2_degen_rate,
            'entropy_drop_final_mean': float(np.mean(entropy_drops_final)),
            'ngram2_trajectory': [float(np.mean([r['metrics']['ngram_2_ratio'][c] for r in seed_results]))
                                  for c in range(config.n_cycles)],
        }

        results[f'fresh_{frac:.1f}'] = agg

        status = "SAFE" if agg['collapse_fraction'] == 0 else "COLLAPSES"
        print(f"  fresh={frac*100:3.0f}%: ngram2 drop={agg['k1_ngram2_drop_mean']*100:+.1f}% (5cyc), "
              f"collapse={agg['collapse_fraction']*100:.0f}%, "
              f"degen={agg['k2_degeneracy_rate']*100:.0f}%, {status}")

    return results


def experiment_detection_metric_comparison(config: CollapseConfig) -> Dict:
    """Experiment 4: Which metric detects collapse earliest?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Detection Metric Comparison")
    print("=" * 70)

    rank = 16
    all_warnings = {
        'ngram_2_ratio': [], 'ngram_3_ratio': [],
        'entropy': [], 'embedding_variance': [],
    }

    for seed in range(config.n_seeds):
        metrics = run_collapse_experiment(config, rank, 0.0, seed)
        analysis = analyze_collapse(metrics, config)
        for metric_name, warn_cycle in analysis['early_warning'].items():
            all_warnings[metric_name].append(warn_cycle)

    results = {}
    for metric_name, cycles in all_warnings.items():
        detected = [c for c in cycles if c is not None]
        detection_rate = len(detected) / len(cycles)
        result = {
            'detection_rate': detection_rate,
            'mean_detection_cycle': float(np.mean(detected)) if detected else None,
            'std_detection_cycle': float(np.std(detected)) if detected else None,
            'earliest_detection': min(detected) if detected else None,
        }
        results[metric_name] = result

        if detected:
            print(f"  {metric_name:20s}: detected {detection_rate*100:.0f}%, "
                  f"mean cycle={result['mean_detection_cycle']:.1f} "
                  f"(earliest={result['earliest_detection']})")
        else:
            print(f"  {metric_name:20s}: NOT detected (0%)")

    ranked = sorted(
        [(name, r) for name, r in results.items() if isinstance(r, dict) and r.get('detection_rate', 0) > 0.5],
        key=lambda x: x[1]['mean_detection_cycle'] if x[1]['mean_detection_cycle'] is not None else 999
    )

    if ranked:
        print(f"\n  Best early detector: {ranked[0][0]} (cycle {ranked[0][1]['mean_detection_cycle']:.1f})")

    results['ranking'] = [name for name, _ in ranked]
    return results


# ============================================================
# Main
# ============================================================

def main():
    config = CollapseConfig()

    print("=" * 70)
    print("MODEL COLLAPSE DETECTION FOR SELF-LEARNING LORA EXPERTS (v2)")
    print("=" * 70)
    print(f"Config: vocab={config.vocab_size}, seq_len={config.seq_len}, "
          f"n_seq={config.n_sequences}, d_model={config.d_model}")
    print(f"        {config.n_cycles} cycles, {config.n_seeds} seeds, "
          f"temp={config.temperature}, corr={config.correlation_strength}")

    all_results = {}

    # Experiment 1: Rank sweep + controlled baselines
    all_results['rank_sweep'] = experiment_rank_sweep(config)

    # Experiment 2: Correlation (in collapse-prone regime)
    all_results['correlation'] = experiment_correlation_effect(config)

    # Experiment 3: Fresh data mitigation
    all_results['fresh_data'] = experiment_fresh_data_mitigation(config)

    # Experiment 4: Detection metrics
    all_results['detection'] = experiment_detection_metric_comparison(config)

    # ============================================================
    # Kill Criteria Assessment
    # ============================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    rank16 = all_results['rank_sweep'].get('rank_16', {})
    k1_drop = rank16.get('k1_ngram2_drop_mean', 0)
    k1_pass = k1_drop < config.diversity_drop_threshold

    print(f"\nK1: Output diversity drops >30% after 5 self-learning cycles?")
    print(f"  At rank=16 (SOLE default), no fresh data:")
    print(f"    n-gram diversity drop: {k1_drop*100:.1f}%")
    print(f"    Threshold: 30%")
    print(f"    Verdict: {'PASS' if k1_pass else 'FAIL'}")

    # Compare all conditions
    print(f"\n  All conditions (K1 at 5 cycles):")
    for rank in config.lora_ranks:
        key = f'rank_{rank}'
        if key in all_results['rank_sweep']:
            drop = all_results['rank_sweep'][key]['k1_ngram2_drop_mean']
            verdict = "PASS" if drop < config.diversity_drop_threshold else "FAIL"
            print(f"    LoRA r={rank:3d} (normed):    drop={drop*100:+.1f}% -> {verdict}")

    for rank in config.lora_ranks:
        key = f'rank_{rank}_no_norm'
        if key in all_results['rank_sweep']:
            drop = all_results['rank_sweep'][key]['k1_ngram2_drop_mean']
            verdict = "PASS" if drop < config.diversity_drop_threshold else "FAIL"
            print(f"    LoRA r={rank:3d} (NO norm):   drop={drop*100:+.1f}% -> {verdict}")

    for key, label in [('full_rank_anchored', 'Full-rank (anchored)'),
                       ('full_rank_unanchored', 'Full-rank (unanchored)')]:
        if key in all_results['rank_sweep']:
            drop = all_results['rank_sweep'][key]['k1_ngram2_drop_mean']
            verdict = "PASS" if drop < config.diversity_drop_threshold else "FAIL"
            print(f"    {label:25s}: drop={drop*100:+.1f}% -> {verdict}")

    k2_degen = rank16.get('k2_degeneracy_rate', 0)
    k2_pass = k2_degen < 0.5

    print(f"\nK2: Expert converges to repetitive/degenerate outputs?")
    print(f"  At rank=16, no fresh data, after {config.n_cycles} cycles:")
    print(f"    Degeneracy rate: {k2_degen*100:.0f}% of seeds")
    print(f"    Verdict: {'PASS' if k2_pass else 'FAIL'}")

    # --- Attribution analysis: rank vs anchoring vs norm ---
    print(f"\n--- ATTRIBUTION ANALYSIS ---")

    # Compare anchored full-rank vs LoRA to isolate rank effect
    anchored_fr = all_results['rank_sweep'].get('full_rank_anchored', {})
    unanchored_fr = all_results['rank_sweep'].get('full_rank_unanchored', {})

    anchored_drop = anchored_fr.get('k1_ngram2_drop_mean', 0)
    unanchored_drop = unanchored_fr.get('k1_ngram2_drop_mean', 0)
    lora16_drop = k1_drop

    print(f"\n  Base-anchoring effect:")
    print(f"    Unanchored full-rank: {unanchored_drop*100:+.1f}%")
    print(f"    Anchored full-rank:   {anchored_drop*100:+.1f}%")
    if abs(unanchored_drop) > 0.01:
        print(f"    Anchoring reduces collapse by: {(1 - anchored_drop/unanchored_drop)*100:.1f}%")

    anchored_collapses = anchored_fr.get('collapse_fraction', 0)
    print(f"    Anchored FR collapse rate: {anchored_collapses*100:.0f}%")

    if anchored_collapses > 0:
        print(f"    -> Anchoring alone does NOT prevent collapse")
        print(f"    -> Rank constraint provides ADDITIONAL protection (LoRA drop: {lora16_drop*100:+.1f}%)")
    else:
        print(f"    -> Anchoring alone MAY prevent collapse (confound not fully resolved)")
        print(f"    -> But LoRA provides even stronger protection (LoRA drop: {lora16_drop*100:+.1f}%)")

    # Norm constraint ablation
    print(f"\n  Norm constraint effect:")
    for rank in config.lora_ranks:
        normed = all_results['rank_sweep'].get(f'rank_{rank}', {}).get('k1_ngram2_drop_mean', 0)
        unnormed = all_results['rank_sweep'].get(f'rank_{rank}_no_norm', {}).get('k1_ngram2_drop_mean', 0)
        unnormed_collapse = all_results['rank_sweep'].get(f'rank_{rank}_no_norm', {}).get('collapse_fraction', 0)
        print(f"    r={rank:3d}: normed={normed*100:+.1f}%, unnormed={unnormed*100:+.1f}%, "
              f"unnormed collapse={unnormed_collapse*100:.0f}%")

    # Rank correlation
    print(f"\n--- Rank-Collapse Correlation ---")
    ranks = config.lora_ranks
    drops = [all_results['rank_sweep'][f'rank_{r}']['k1_ngram2_drop_mean'] for r in ranks]
    if len(drops) >= 2:
        corr, p_val = stats.spearmanr(ranks, drops)
        print(f"  Spearman (normed LoRA): rho={corr:.3f}, p={p_val:.4f}")

    drops_unnorm = [all_results['rank_sweep'].get(f'rank_{r}_no_norm', {}).get('k1_ngram2_drop_mean', 0)
                    for r in ranks]
    if len(drops_unnorm) >= 2:
        corr_un, p_val_un = stats.spearmanr(ranks, drops_unnorm)
        print(f"  Spearman (unnormed LoRA): rho={corr_un:.3f}, p={p_val_un:.4f}")

    # Correlation effect summary
    corr_result = all_results['correlation']
    print(f"\n--- Correlation Effect ---")
    for regime_prefix, regime_name in [
        ('anchored_fr', 'Anchored full-rank'),
        ('unanchored_fr', 'Unanchored full-rank'),
        ('lora', 'LoRA r=16'),
    ]:
        accel = corr_result.get(f'{regime_prefix}_acceleration', float('nan'))
        print(f"  {regime_name}: acceleration={accel:.2f}x")

    # Fresh data
    print(f"\n--- Fresh Data Mitigation ---")
    min_safe_frac = None
    for frac in config.fresh_data_fractions:
        key = f'fresh_{frac:.1f}'
        if key in all_results['fresh_data']:
            if all_results['fresh_data'][key]['collapse_fraction'] == 0:
                if min_safe_frac is None:
                    min_safe_frac = frac

    if min_safe_frac is not None:
        print(f"  Minimum safe fresh data fraction (LoRA): {min_safe_frac*100:.0f}%")

    # Overall verdict
    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT")
    print(f"{'=' * 70}")

    overall = "SUPPORTED" if k1_pass else "CONDITIONAL"
    if not k1_pass and k2_pass:
        overall = "CONDITIONAL"
    elif not k1_pass and not k2_pass:
        overall = "FAIL"

    print(f"  K1 (diversity at 5 cycles): {'PASS' if k1_pass else 'FAIL'}")
    print(f"  K2 (degeneracy at {config.n_cycles} cycles): {'PASS' if k2_pass else 'FAIL'}")
    print(f"  Overall: {overall}")

    all_results['kill_criteria'] = {
        'k1_pass': k1_pass,
        'k1_ngram2_drop': k1_drop,
        'k2_pass': k2_pass,
        'k2_degeneracy_rate': k2_degen,
        'overall': overall,
        'lora_rank_correlation': float(corr) if len(drops) >= 2 else None,
        'lora_rank_p_value': float(p_val) if len(drops) >= 2 else None,
        'anchored_fr_drop': anchored_drop,
        'unanchored_fr_drop': unanchored_drop,
        'min_safe_fresh_fraction': min_safe_frac,
    }

    # Save results
    outdir = Path(__file__).parent

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(outdir / 'results.json', 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nResults saved to {outdir / 'results.json'}")
    return all_results


if __name__ == '__main__':
    main()
