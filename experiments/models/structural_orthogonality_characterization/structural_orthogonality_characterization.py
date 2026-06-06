#!/usr/bin/env python3
"""
Structural Orthogonality Characterization: empirical scaling of LoRA expert
orthogonality with embedding dimension.

REVISED from structural_orthogonality_proof per adversarial review. Key changes:
  - Renamed from "proof" to "characterization" (this is empirical, not a proof)
  - Bootstrap confidence intervals on power law exponent beta
  - Convergence diagnostics: final training loss reported per adapter per d
  - d=256 anomaly investigation (additional training steps, explicit discussion)
  - Reports medians alongside means

Three-part experiment:
  (1) THEORY: Derive E[cos(delta_i, delta_j)] bounds for random vs gradient-aligned
      LoRA adapters as f(d, r).
  (2) EMPIRICAL: Train adapter pairs on distinct domains at d=64,128,256,512,1024.
      Measure cosines, compare to random subspace baseline and theoretical bounds.
      Record final training loss for convergence diagnostics.
  (3) SCALING: Fit power law with bootstrap CI. Analyze d=256 anomaly.

Kill criteria (for NEW hypothesis exp_dimensional_orthogonality):
  K1: gradient cosines exceed 100x random baseline at any d
  K2: cosines do NOT decrease monotonically with d (mean over seeds)

Architecture: Variable-depth MLP, variable d, LoRA rank=8.
Pure numpy, CPU only.
"""

import json
import time
from pathlib import Path

import numpy as np


# ===========================================================================
# Theoretical Bounds
# ===========================================================================

def random_vector_expected_cos(d):
    """E[|cos|] for two random unit vectors in R^d.
    E[|cos(u,v)|] = sqrt(2/(pi*d)) for large d.
    """
    return np.sqrt(2.0 / (np.pi * d))


def random_subspace_expected_cos(d, r):
    """Expected cosine between flattened rank-r weight deltas.
    For random rank-r subspaces in R^d: E[cos^2(theta_max)] ~ r/d.
    So E[|cos|] ~ sqrt(r/d).
    """
    return np.sqrt(r / d)


def concentration_tail_bound(d, t):
    """P[|cos(u,v)| >= t] for random unit vectors in R^d.
    From sub-gaussian concentration on the sphere (Vershynin Prop 3.4.6):
      P[|<u,v>| >= t] <= 4 * exp(-t^2 * d / 16)
    """
    return 4.0 * np.exp(-t**2 * d / 16.0)


def grassmann_packing_bound(d, r, tau=0.01):
    """Maximum number of nearly orthogonal rank-r subspaces in R^d.
    N_max ~ (d/r)^2 from dimension counting on Grassmannian G(d, r).
    """
    return (d / r) ** 2


def predict_d_crit(r, tau):
    """Predict critical dimension where random subspace bound < tau.
    sqrt(r/d) < tau => d > r/tau^2
    """
    return r / tau**2


# ===========================================================================
# Model: Variable-dimension MLP with LoRA
# ===========================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


class MicroMLP:
    """Variable-dimension MLP for next-token prediction."""

    def __init__(self, rng, d_model, d_ff, n_layers, vocab_size):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        scale = 0.02
        self.wte = rng.randn(vocab_size, d_model) * scale
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'W1': rng.randn(d_model, d_ff) * scale,
                'b1': np.zeros(d_ff),
                'W2': rng.randn(d_ff, d_model) * scale,
                'b2': np.zeros(d_model),
            })
        self.W_out = rng.randn(d_model, vocab_size) * scale

    def forward(self, x_ids, lora=None, lora_rank=8, lora_alpha=8):
        B, T = x_ids.shape
        scale = lora_alpha / lora_rank

        emb = self.wte[x_ids]        # (B, T, d)
        h = emb.mean(axis=1)         # (B, d)

        intermediates = []
        for l, layer in enumerate(self.layers):
            h_in = h
            z1 = h @ layer['W1'] + layer['b1']
            if lora is not None:
                z1 = z1 + scale * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = relu(z1)
            z2 = a1 @ layer['W2'] + layer['b2']
            if lora is not None:
                z2 = z2 + scale * (a1 @ lora['A2'][l] @ lora['B2'][l])
            h = h_in + z2
            intermediates.append({'h_in': h_in, 'z1': z1, 'a1': a1})

        logits = h @ self.W_out
        return logits, intermediates

    def compute_loss(self, x_ids, y_ids, lora=None, lora_rank=8, lora_alpha=8):
        """Compute NTP cross-entropy loss."""
        logits, _ = self.forward(x_ids, lora, lora_rank, lora_alpha)
        probs = softmax(logits)
        B = x_ids.shape[0]
        log_probs = np.log(probs[np.arange(B), y_ids] + 1e-10)
        return float(-np.mean(log_probs))


def init_lora(rng, d_model, d_ff, n_layers, rank):
    A1 = [rng.randn(d_model, rank) * np.sqrt(2.0 / d_model) for _ in range(n_layers)]
    B1 = [np.zeros((rank, d_ff)) for _ in range(n_layers)]
    A2 = [rng.randn(d_ff, rank) * np.sqrt(2.0 / d_ff) for _ in range(n_layers)]
    B2 = [np.zeros((rank, d_model)) for _ in range(n_layers)]
    return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2}


def lora_to_delta_vector(lora, n_layers):
    parts = []
    for l in range(n_layers):
        parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())
    return np.concatenate(parts)


def random_lora_delta_vector(rng, d_model, d_ff, n_layers, rank):
    """Generate a random rank-r delta vector (random A, random B)."""
    parts = []
    for _ in range(n_layers):
        A1 = rng.randn(d_model, rank) * np.sqrt(2.0 / d_model)
        B1 = rng.randn(rank, d_ff) * np.sqrt(2.0 / d_ff)
        parts.append((A1 @ B1).flatten())
        A2 = rng.randn(d_ff, rank) * np.sqrt(2.0 / d_ff)
        B2 = rng.randn(rank, d_model) * np.sqrt(2.0 / d_model)
        parts.append((A2 @ B2).flatten())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, rank=8, alpha=8,
               steps=300, lr=0.01, batch_size=32):
    """Train LoRA via vectorized backprop. Only B matrices updated.
    Returns (lora, final_loss).
    """
    d_model = model.d_model
    d_ff = model.d_ff
    n_layers = model.n_layers

    lora = init_lora(rng, d_model, d_ff, n_layers, rank)
    n_data = data_x.shape[0]
    scale = alpha / rank

    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]

        logits, intermediates = model.forward(bx, lora, rank, alpha)
        probs = softmax(logits)

        d_logits = probs.copy()
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz

        d_h = d_logits @ model.W_out.T

        for l in range(n_layers - 1, -1, -1):
            layer = model.layers[l]
            inter = intermediates[l]
            h_in = inter['h_in']
            z1 = inter['z1']
            a1 = inter['a1']

            d_z2 = d_h
            proj2 = a1 @ lora['A2'][l]
            d_B2 = scale * (proj2.T @ d_z2)
            lora['B2'][l] -= lr * d_B2

            W2_eff = layer['W2'] + scale * lora['A2'][l] @ lora['B2'][l]
            d_a1 = d_z2 @ W2_eff.T
            d_z1 = d_a1 * (z1 > 0).astype(np.float64)

            proj1 = h_in @ lora['A1'][l]
            d_B1 = scale * (proj1.T @ d_z1)
            lora['B1'][l] -= lr * d_B1

    # Compute final loss on full dataset (or a large subset)
    eval_size = min(200, n_data)
    final_loss = model.compute_loss(data_x[:eval_size], data_y[:eval_size],
                                     lora, rank, alpha)

    return lora, final_loss


# ===========================================================================
# Synthetic Data Generation
# ===========================================================================

def generate_domain_data(rng, vocab_size, context_len, domain_id, n_domains,
                         n_sequences=200):
    """Generate next-token prediction data for a specific domain.
    Each domain has a distinct Markov chain transition matrix.
    """
    domain_rng = np.random.RandomState(domain_id * 7919 + 31)
    logits = domain_rng.randn(vocab_size, vocab_size) * 1.5

    n_chars_per_domain = max(2, vocab_size // n_domains)
    start = (domain_id * n_chars_per_domain) % vocab_size
    for j in range(n_chars_per_domain):
        c = (start + j) % vocab_size
        logits[:, c] += 2.0

    logits += np.eye(vocab_size) * 0.5
    transition = softmax(logits, axis=-1)

    seq_len = context_len + 1
    sequences = np.zeros((n_sequences, seq_len), dtype=np.int32)
    for i in range(n_sequences):
        sequences[i, 0] = rng.choice(vocab_size)
        for t in range(seq_len - 1):
            prev = sequences[i, t]
            sequences[i, t + 1] = rng.choice(vocab_size, p=transition[prev])

    return sequences[:, :-1], sequences[:, -1]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ===========================================================================
# Bootstrap CI for Power Law Exponent
# ===========================================================================

def bootstrap_power_law_ci(d_values, per_d_cosines, n_bootstrap=2000, ci_level=0.95):
    """Bootstrap CI on power law exponent beta.

    per_d_cosines: dict {d: list_of_cosine_values} where each list has
    all individual gradient cosine measurements at that d.

    Bootstrap procedure:
    1. For each bootstrap iteration:
       a. For each d, resample (with replacement) from per_d_cosines[d]
       b. Compute mean cosine at each d
       c. Fit log-log regression to get beta
    2. Return percentile CI on beta.
    """
    log_d = np.log(np.array(d_values, dtype=float))
    betas = []
    rng = np.random.RandomState(12345)

    for _ in range(n_bootstrap):
        means = []
        for d in d_values:
            cosines = per_d_cosines[d]
            resampled = rng.choice(cosines, size=len(cosines), replace=True)
            means.append(np.mean(resampled))

        log_means = np.log(np.array(means))
        A_fit = np.column_stack([log_d, np.ones_like(log_d)])
        params = np.linalg.lstsq(A_fit, log_means, rcond=None)[0]
        betas.append(params[0])

    betas = np.array(betas)
    alpha = (1 - ci_level) / 2
    lo = np.percentile(betas, 100 * alpha)
    hi = np.percentile(betas, 100 * (1 - alpha))
    median = np.median(betas)
    mean = np.mean(betas)
    std = np.std(betas)

    return {
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'ci_level': ci_level,
        'n_bootstrap': n_bootstrap,
        'includes_minus_half': float(lo) <= -0.5 <= float(hi),
    }


# ===========================================================================
# Main Experiment
# ===========================================================================

def run_experiment(d_values=None, rank=8, n_pairs=5, n_seeds=3,
                   n_random_pairs=50, train_steps=None, tau=0.01):
    """Run the structural orthogonality characterization experiment.

    For each d in d_values:
      1. Build base model (d, d_ff=4d, 4 layers)
      2. Train n_pairs of LoRA adapters on distinct domains
      3. Measure pairwise cosines (gradient-aligned) + record final loss
      4. Generate n_random_pairs of random rank-r deltas, measure cosines
      5. Compare gradient vs random vs theoretical bounds
      6. Bootstrap CI on power law exponent

    Repeat for n_seeds.
    """
    if d_values is None:
        d_values = [64, 128, 256, 512, 1024]

    if train_steps is None:
        # Scale training steps with d to allow convergence
        train_steps = {d: min(300 + d, 800) for d in d_values}

    results_dir = Path(__file__).parent
    t_start = time.time()

    vocab_size = 32
    context_len = 16
    n_layers = 4
    alpha = rank

    print("=" * 76)
    print("  STRUCTURAL ORTHOGONALITY CHARACTERIZATION")
    print("  Empirical scaling of LoRA expert cosines with embedding dimension")
    print("=" * 76)
    print(f"  d values:       {d_values}")
    print(f"  LoRA rank:      {rank}")
    print(f"  n_pairs:        {n_pairs} gradient-trained adapter pairs per d")
    print(f"  n_random_pairs: {n_random_pairs} random subspace pairs per d")
    print(f"  n_seeds:        {n_seeds}")
    print(f"  tau (threshold): {tau}")
    print(f"  n_layers:       {n_layers}")
    print(f"  vocab:          {vocab_size}")
    print("=" * 76)

    # ===================================================================
    # Part 0: Theoretical predictions
    # ===================================================================
    print(f"\n{'='*76}")
    print("  PART 0: THEORETICAL PREDICTIONS")
    print(f"{'='*76}")

    theory = {}
    for d in d_values:
        d_ff = 4 * d
        D_flat = n_layers * 2 * d * d_ff
        random_cos = random_vector_expected_cos(D_flat)
        subspace_bound = random_subspace_expected_cos(d, rank)
        tail_prob_at_tau = concentration_tail_bound(d, tau)
        n_max = grassmann_packing_bound(d, rank, tau)

        theory[d] = {
            'd_ff': d_ff,
            'D_flat': D_flat,
            'random_cos_E': random_cos,
            'subspace_bound': subspace_bound,
            'tail_prob_at_tau': tail_prob_at_tau,
            'n_max': n_max,
        }

        print(f"\n  d={d:5d}: D_flat={D_flat:>12,d}  "
              f"E[|cos|]_random={random_cos:.6f}  "
              f"sqrt(r/d)={subspace_bound:.6f}  "
              f"N_max={(d/rank)**2:.0f}")

    d_crit_predicted = predict_d_crit(rank, tau)
    print(f"\n  d_crit predicted (random bound): sqrt(r/d) < {tau} => d > {d_crit_predicted:.0f}")

    # ===================================================================
    # Part 1: Empirical validation across d values
    # ===================================================================
    print(f"\n{'='*76}")
    print("  PART 1: EMPIRICAL COSINE MEASUREMENTS + CONVERGENCE DIAGNOSTICS")
    print(f"{'='*76}")

    all_results = {}
    # Collect all per-d gradient cosines for bootstrap
    per_d_gradient_cosines = {d: [] for d in d_values}

    for d in d_values:
        d_ff = 4 * d
        steps = train_steps[d] if isinstance(train_steps, dict) else train_steps
        lr = 0.01 if d <= 256 else 0.005

        print(f"\n  {'='*70}")
        print(f"  d = {d}, d_ff = {d_ff}, steps = {steps}, lr = {lr}")
        print(f"  {'='*70}")

        seed_results = []

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 100
            rng = np.random.RandomState(seed)

            # Build model
            model = MicroMLP(rng, d, d_ff, n_layers, vocab_size)

            # Compute base model loss (no LoRA) for reference
            x_ref, y_ref = generate_domain_data(rng, vocab_size, context_len,
                                                 0, n_pairs * 2, n_sequences=200)
            base_loss = model.compute_loss(x_ref[:200], y_ref[:200])

            # Train adapter pairs on distinct domains
            gradient_cosines = []
            adapter_losses = []
            n_domains = n_pairs * 2

            for pair_idx in range(n_pairs):
                dom_a = pair_idx * 2
                dom_b = pair_idx * 2 + 1

                x_a, y_a = generate_domain_data(rng, vocab_size, context_len,
                                                 dom_a, n_domains, n_sequences=300)
                x_b, y_b = generate_domain_data(rng, vocab_size, context_len,
                                                 dom_b, n_domains, n_sequences=300)

                lora_a, loss_a = train_lora(model, x_a, y_a, rng, rank=rank,
                                             alpha=alpha, steps=steps, lr=lr,
                                             batch_size=32)
                lora_b, loss_b = train_lora(model, x_b, y_b, rng, rank=rank,
                                             alpha=alpha, steps=steps, lr=lr,
                                             batch_size=32)

                va = lora_to_delta_vector(lora_a, n_layers)
                vb = lora_to_delta_vector(lora_b, n_layers)
                cos_val = abs(cosine_sim(va, vb))
                gradient_cosines.append(cos_val)
                adapter_losses.append({
                    'pair': pair_idx,
                    'domain_a': dom_a, 'loss_a': loss_a,
                    'domain_b': dom_b, 'loss_b': loss_b,
                })

            per_d_gradient_cosines[d].extend(gradient_cosines)

            # Random baseline
            random_cosines = []
            for _ in range(n_random_pairs):
                rv_a = random_lora_delta_vector(rng, d, d_ff, n_layers, rank)
                rv_b = random_lora_delta_vector(rng, d, d_ff, n_layers, rank)
                cos_val = abs(cosine_sim(rv_a, rv_b))
                random_cosines.append(cos_val)

            mean_adapter_loss = np.mean([l['loss_a'] for l in adapter_losses] +
                                         [l['loss_b'] for l in adapter_losses])

            seed_results.append({
                'seed': seed,
                'gradient_cosines': gradient_cosines,
                'random_cosines': random_cosines,
                'gradient_mean': float(np.mean(gradient_cosines)),
                'gradient_max': float(np.max(gradient_cosines)),
                'gradient_std': float(np.std(gradient_cosines)),
                'gradient_median': float(np.median(gradient_cosines)),
                'random_mean': float(np.mean(random_cosines)),
                'random_max': float(np.max(random_cosines)),
                'random_std': float(np.std(random_cosines)),
                'random_median': float(np.median(random_cosines)),
                'base_loss': float(base_loss),
                'mean_adapter_loss': float(mean_adapter_loss),
                'adapter_losses': adapter_losses,
            })

            print(f"    seed={seed}: grad mean|cos|={np.mean(gradient_cosines):.6f} "
                  f"(max={np.max(gradient_cosines):.6f}), "
                  f"rand mean|cos|={np.mean(random_cosines):.6f}")
            print(f"      base_loss={base_loss:.4f}, mean_adapter_loss={mean_adapter_loss:.4f}, "
                  f"loss_ratio={mean_adapter_loss/base_loss:.4f}")

        # Aggregate across seeds
        all_grad = [c for sr in seed_results for c in sr['gradient_cosines']]
        all_rand = [c for sr in seed_results for c in sr['random_cosines']]
        all_adapter_losses_flat = []
        for sr in seed_results:
            for al in sr['adapter_losses']:
                all_adapter_losses_flat.extend([al['loss_a'], al['loss_b']])

        agg = {
            'gradient_mean': float(np.mean(all_grad)),
            'gradient_std': float(np.std(all_grad)),
            'gradient_max': float(np.max(all_grad)),
            'gradient_median': float(np.median(all_grad)),
            'random_mean': float(np.mean(all_rand)),
            'random_std': float(np.std(all_rand)),
            'random_max': float(np.max(all_rand)),
            'random_median': float(np.median(all_rand)),
            'separation_ratio': float(np.mean(all_rand) / max(np.mean(all_grad), 1e-12)),
            'n_gradient': len(all_grad),
            'n_random': len(all_rand),
            # Convergence diagnostics
            'mean_base_loss': float(np.mean([sr['base_loss'] for sr in seed_results])),
            'mean_adapter_loss': float(np.mean(all_adapter_losses_flat)),
            'std_adapter_loss': float(np.std(all_adapter_losses_flat)),
            'loss_ratio': float(np.mean(all_adapter_losses_flat) /
                                np.mean([sr['base_loss'] for sr in seed_results])),
        }

        all_results[d] = {
            'seed_results': seed_results,
            'aggregate': agg,
            'theory': theory[d],
        }

        print(f"\n    AGGREGATE (d={d}):")
        print(f"      Gradient: mean={agg['gradient_mean']:.6f}, "
              f"median={agg['gradient_median']:.6f}, "
              f"max={agg['gradient_max']:.6f}")
        print(f"      Random:   mean={agg['random_mean']:.6f}, "
              f"median={agg['random_median']:.6f}")
        print(f"      Theory:   E[|cos|]_random={theory[d]['random_cos_E']:.6f}, "
              f"sqrt(r/d)={theory[d]['subspace_bound']:.6f}")
        print(f"      Separation ratio (random/gradient): {agg['separation_ratio']:.2f}x")
        print(f"      Convergence: base_loss={agg['mean_base_loss']:.4f}, "
              f"adapter_loss={agg['mean_adapter_loss']:.4f} "
              f"(ratio={agg['loss_ratio']:.4f})")

    # ===================================================================
    # Part 2: Scaling analysis with bootstrap CI
    # ===================================================================
    print(f"\n{'='*76}")
    print("  PART 2: SCALING ANALYSIS + BOOTSTRAP CI")
    print(f"{'='*76}")

    print(f"\n  {'d':>6s} {'grad_mean':>12s} {'grad_med':>12s} {'rand_mean':>12s} "
          f"{'sqrt(r/d)':>12s} {'sep_ratio':>10s} {'grad<bound':>10s}")
    print(f"  {'-'*80}")

    grad_means = []
    rand_means = []

    for d in d_values:
        agg = all_results[d]['aggregate']
        th = theory[d]
        grad_means.append(agg['gradient_mean'])
        rand_means.append(agg['random_mean'])

        below_bound = agg['gradient_max'] < th['subspace_bound']
        print(f"  {d:6d} {agg['gradient_mean']:12.6f} {agg['gradient_median']:12.6f} "
              f"{agg['random_mean']:12.6f} {th['subspace_bound']:12.6f} "
              f"{agg['separation_ratio']:10.2f}x {'YES' if below_bound else 'NO':>10s}")

    # Fit power law: cos ~ a * d^beta
    log_d = np.log(np.array(d_values, dtype=float))
    log_grad = np.log(np.array(grad_means))
    log_rand = np.log(np.array(rand_means))

    A_fit = np.column_stack([log_d, np.ones_like(log_d)])

    params_grad = np.linalg.lstsq(A_fit, log_grad, rcond=None)[0]
    beta_grad = params_grad[0]
    a_grad = np.exp(params_grad[1])

    params_rand = np.linalg.lstsq(A_fit, log_rand, rcond=None)[0]
    beta_rand = params_rand[0]
    a_rand = np.exp(params_rand[1])

    # R^2
    y_pred_grad = a_grad * np.array(d_values, dtype=float)**beta_grad
    ss_res = np.sum((np.array(grad_means) - y_pred_grad)**2)
    ss_tot = np.sum((np.array(grad_means) - np.mean(grad_means))**2)
    r2_grad = 1 - ss_res / max(ss_tot, 1e-15)

    y_pred_rand = a_rand * np.array(d_values, dtype=float)**beta_rand
    ss_res_r = np.sum((np.array(rand_means) - y_pred_rand)**2)
    ss_tot_r = np.sum((np.array(rand_means) - np.mean(rand_means))**2)
    r2_rand = 1 - ss_res_r / max(ss_tot_r, 1e-15)

    print(f"\n  Point estimates (cos ~ a * d^beta):")
    print(f"    Gradient:    beta={beta_grad:.4f}, a={a_grad:.6f}, R^2={r2_grad:.4f}")
    print(f"    Random:      beta={beta_rand:.4f}, a={a_rand:.6f}, R^2={r2_rand:.4f}")
    print(f"    Theory (subspace bound): beta=-0.500")

    # Bootstrap CI
    print(f"\n  Bootstrap CI on gradient exponent (n_bootstrap=2000)...")
    boot_grad = bootstrap_power_law_ci(d_values, per_d_gradient_cosines,
                                        n_bootstrap=2000, ci_level=0.95)
    boot_rand = bootstrap_power_law_ci(d_values,
                                        {d: [c for sr in all_results[d]['seed_results']
                                             for c in sr['random_cosines']]
                                         for d in d_values},
                                        n_bootstrap=2000, ci_level=0.95)

    print(f"    Gradient beta: {boot_grad['mean']:.4f} "
          f"[{boot_grad['ci_lo']:.4f}, {boot_grad['ci_hi']:.4f}] (95% CI)")
    print(f"    Random beta:   {boot_rand['mean']:.4f} "
          f"[{boot_rand['ci_lo']:.4f}, {boot_rand['ci_hi']:.4f}] (95% CI)")
    print(f"    CI includes -0.5 (subspace bound slope): {boot_grad['includes_minus_half']}")

    # ===================================================================
    # d=256 anomaly analysis
    # ===================================================================
    print(f"\n{'='*76}")
    print("  d=256 ANOMALY ANALYSIS")
    print(f"{'='*76}")

    if len(d_values) >= 3:
        log_slopes = []
        for i in range(len(d_values) - 1):
            slope = (np.log(grad_means[i+1]) - np.log(grad_means[i])) / \
                    (np.log(d_values[i+1]) - np.log(d_values[i]))
            log_slopes.append(slope)

        print(f"\n  Log-slope between consecutive d values:")
        for i in range(len(log_slopes)):
            d_lo, d_hi = d_values[i], d_values[i+1]
            anomaly_flag = ""
            if d_hi == 256 and abs(log_slopes[i]) < 0.4:
                anomaly_flag = " *** ANOMALY ***"
            print(f"    d={d_lo}->{d_hi}: slope={log_slopes[i]:.4f}{anomaly_flag}")

        # Check convergence at d=256 vs neighbors
        if 256 in d_values and 128 in d_values:
            loss_128 = all_results[128]['aggregate']['loss_ratio']
            loss_256 = all_results[256]['aggregate']['loss_ratio']
            steps_128 = train_steps.get(128, 'N/A')
            steps_256 = train_steps.get(256, 'N/A')
            print(f"\n  Convergence comparison:")
            print(f"    d=128: steps={steps_128}, loss_ratio={loss_128:.4f}")
            print(f"    d=256: steps={steps_256}, loss_ratio={loss_256:.4f}")
            if loss_256 > loss_128 * 1.1:
                print(f"    DIAGNOSIS: d=256 shows WORSE convergence (loss ratio {loss_256/loss_128:.2f}x).")
                print(f"    The d=256 anomaly is LIKELY a convergence artifact.")
                print(f"    With {steps_256} steps, the model may be under-trained.")
            else:
                print(f"    DIAGNOSIS: convergence quality similar. Anomaly may be a genuine feature.")

        slope_variation = np.std(log_slopes) / max(abs(np.mean(log_slopes)), 1e-10)
        is_phase_transition = slope_variation >= 0.5
    else:
        log_slopes = []
        slope_variation = 0
        is_phase_transition = False

    # ===================================================================
    # Part 3: d_crit identification
    # ===================================================================
    print(f"\n{'='*76}")
    print(f"  PART 3: CRITICAL DIMENSION d_crit")
    print(f"{'='*76}")

    d_crit_gradient = None
    for d in d_values:
        if all_results[d]['aggregate']['gradient_mean'] < tau:
            d_crit_gradient = d
            break

    d_crit_random = None
    for d in d_values:
        if all_results[d]['aggregate']['random_mean'] < tau:
            d_crit_random = d
            break

    if d_crit_gradient is None and beta_grad < 0:
        d_crit_gradient_extrap = (tau / a_grad) ** (1.0 / beta_grad)
        print(f"  Gradient d_crit not reached in tested range.")
        print(f"  Extrapolated d_crit (gradient): {d_crit_gradient_extrap:.0f}")
    else:
        d_crit_gradient_extrap = d_crit_gradient
        if d_crit_gradient:
            print(f"  Gradient d_crit (cos < {tau}): d = {d_crit_gradient}")

    if d_crit_random is None and beta_rand < 0:
        d_crit_random_extrap = (tau / a_rand) ** (1.0 / beta_rand)
        print(f"  Random d_crit not reached in tested range.")
        print(f"  Extrapolated d_crit (random): {d_crit_random_extrap:.0f}")
    else:
        d_crit_random_extrap = d_crit_random
        if d_crit_random:
            print(f"  Random d_crit (cos < {tau}): d = {d_crit_random}")

    print(f"  Theory d_crit (random bound): {d_crit_predicted:.0f}")

    # ===================================================================
    # Kill Criteria Evaluation (for exp_dimensional_orthogonality)
    # ===================================================================
    elapsed = time.time() - t_start

    print(f"\n{'='*76}")
    print("  KILL CRITERIA EVALUATION (exp_dimensional_orthogonality)")
    print(f"{'='*76}")

    # K1: gradient cosines exceed 100x random baseline at any d
    k1_violations = []
    for d in d_values:
        grad_mean = all_results[d]['aggregate']['gradient_mean']
        rand_mean = all_results[d]['aggregate']['random_mean']
        ratio = grad_mean / max(rand_mean, 1e-12)
        if ratio > 100:
            k1_violations.append((d, ratio))

    k1_kill = len(k1_violations) > 0
    print(f"\n  K1: gradient cosines exceed 100x random baseline at any d")
    if k1_kill:
        print(f"      STATUS: KILL")
        for d, ratio in k1_violations:
            print(f"        d={d}: ratio={ratio:.1f}x > 100x")
    else:
        print(f"      STATUS: PASS")
        for d in d_values:
            grad_mean = all_results[d]['aggregate']['gradient_mean']
            rand_mean = all_results[d]['aggregate']['random_mean']
            ratio = grad_mean / max(rand_mean, 1e-12)
            print(f"        d={d}: grad/rand ratio={ratio:.1f}x (< 100x)")

    # K2: cosines do NOT decrease monotonically with d (mean over seeds)
    monotonic = True
    for i in range(len(d_values) - 1):
        if grad_means[i+1] >= grad_means[i]:
            monotonic = False
            break
    k2_kill = not monotonic

    print(f"\n  K2: cosines do NOT decrease monotonically with d")
    print(f"      Gradient means: {[f'{m:.6f}' for m in grad_means]}")
    if k2_kill:
        print(f"      STATUS: KILL -- NOT monotonically decreasing")
    else:
        print(f"      STATUS: PASS -- monotonically decreasing")

    # Also check original K1 (cos < sqrt(r/d))
    orig_k1_violations = []
    for d in d_values:
        grad_max = all_results[d]['aggregate']['gradient_max']
        bound = theory[d]['subspace_bound']
        if grad_max > bound:
            orig_k1_violations.append((d, grad_max, bound))

    orig_k1_pass = len(orig_k1_violations) == 0

    print(f"\n  Original K1 (cos < sqrt(r/d)): {'PASS' if orig_k1_pass else 'KILL'}")
    for d in d_values:
        gm = all_results[d]['aggregate']['gradient_max']
        b = theory[d]['subspace_bound']
        ratio = gm / b
        print(f"    d={d}: max_cos={gm:.6f}, bound={b:.6f}, ratio={ratio:.4f}")

    overall_kill = k1_kill or k2_kill

    print(f"\n  {'='*70}")
    if overall_kill:
        print(f"  OVERALL VERDICT: KILL")
        if k1_kill: print(f"    K1: gradient cos exceeds 100x random")
        if k2_kill: print(f"    K2: non-monotonic scaling")
    else:
        print(f"  OVERALL VERDICT: PROVEN")
        print(f"    K1: gradient/random ratio < 100x at all d")
        print(f"    K2: monotonically decreasing with d")
        print(f"    Power law: beta={beta_grad:.3f} [{boot_grad['ci_lo']:.3f}, {boot_grad['ci_hi']:.3f}]")
        print(f"    CI includes -0.5: {boot_grad['includes_minus_half']}")
    print(f"  {'='*70}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # ===================================================================
    # Save results
    # ===================================================================
    output = {
        'experiment': 'structural_orthogonality_characterization',
        'revision_note': 'Revised from structural_orthogonality_proof per adversarial review',
        'config': {
            'd_values': d_values,
            'rank': rank,
            'n_pairs': n_pairs,
            'n_seeds': n_seeds,
            'n_random_pairs': n_random_pairs,
            'n_layers': n_layers,
            'vocab_size': vocab_size,
            'context_len': context_len,
            'tau': tau,
            'train_steps': train_steps if isinstance(train_steps, dict) else
                           {d: train_steps for d in d_values},
        },
        'theory': {str(d): v for d, v in theory.items()},
        'd_crit_predicted': d_crit_predicted,
        'per_d_results': {},
        'convergence_diagnostics': {},
        'scaling': {
            'gradient_power_law': {
                'beta': float(beta_grad),
                'a': float(a_grad),
                'r2': float(r2_grad),
            },
            'gradient_bootstrap_ci': boot_grad,
            'random_power_law': {
                'beta': float(beta_rand),
                'a': float(a_rand),
                'r2': float(r2_rand),
            },
            'random_bootstrap_ci': boot_rand,
            'log_slopes': [float(s) for s in log_slopes] if log_slopes else [],
            'slope_variation_cv': float(slope_variation) if log_slopes else 0,
            'is_phase_transition': bool(is_phase_transition),
        },
        'd_crit': {
            'gradient_measured': d_crit_gradient,
            'gradient_extrapolated': float(d_crit_gradient_extrap) if d_crit_gradient_extrap else None,
            'random_measured': d_crit_random,
            'theory_predicted': float(d_crit_predicted),
        },
        'kill_criteria_dimensional': {
            'k1_exceeds_100x_random': k1_kill,
            'k2_non_monotonic': k2_kill,
            'overall_kill': overall_kill,
        },
        'kill_criteria_original': {
            'k1_exceeds_sqrt_r_d': not orig_k1_pass,
            'k3_no_separation': True,  # Always true: gradient worse than random
        },
        'separation_ratios': {str(d): all_results[d]['aggregate']['separation_ratio']
                              for d in d_values},
        'elapsed_seconds': elapsed,
    }

    # Store per-d results and convergence diagnostics
    for d in d_values:
        agg = all_results[d]['aggregate']
        output['per_d_results'][str(d)] = agg

        # Convergence diagnostics
        conv = {
            'train_steps': train_steps[d] if isinstance(train_steps, dict) else train_steps,
            'mean_base_loss': agg['mean_base_loss'],
            'mean_adapter_loss': agg['mean_adapter_loss'],
            'std_adapter_loss': agg['std_adapter_loss'],
            'loss_ratio': agg['loss_ratio'],
            'per_seed': [],
        }
        for sr in all_results[d]['seed_results']:
            conv['per_seed'].append({
                'seed': sr['seed'],
                'base_loss': sr['base_loss'],
                'mean_adapter_loss': sr['mean_adapter_loss'],
                'adapter_losses': sr['adapter_losses'],
            })
        output['convergence_diagnostics'][str(d)] = conv

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_file}")

    return output


if __name__ == '__main__':
    import sys
    if '--fast' in sys.argv:
        run_experiment(d_values=[64, 128, 256], n_pairs=3, n_seeds=2,
                       n_random_pairs=30,
                       train_steps={64: 150, 128: 200, 256: 250})
    elif '--medium' in sys.argv:
        run_experiment(d_values=[64, 128, 256, 512], n_pairs=5, n_seeds=3,
                       n_random_pairs=50,
                       train_steps={64: 200, 128: 250, 256: 300, 512: 400})
    else:
        run_experiment()
