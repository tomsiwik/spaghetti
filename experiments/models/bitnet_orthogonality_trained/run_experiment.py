#!/usr/bin/env python3
"""
BitNet Orthogonality Trained: Does a ternary base produce more orthogonal
trained LoRA adapter weight deltas than an FP16 base?

DIAGNOSTIC experiment building on exp_bitnet_composition_stability (SUPPORTED).
That experiment found composition ratio 0.63 on ternary base. This experiment
measures WHY -- specifically whether ternary base creates more separable feature
channels as measured by pairwise cosine similarity of trained adapter deltas.

Hypothesis: Ternary {-1, 0, 1} base weights constrain gradient flow through
discrete channels, causing trained adapters to settle in more orthogonal
subspaces than on continuous FP16 base.

Kill criteria:
  K1: mean |cos| of trained adapters on BitNet >= mean |cos| on FP16
      (ternary base does not improve orthogonality)
  K2: arithmetic-sort pair cos on BitNet >= 0.5
      (high-overlap domains still collide -- these are the proxy for
       math-medical which showed cos=0.703 on FP16 at macro scale)

Design:
  - Same micro architecture as bitnet_composition_stability: d=64, r=4, L=2, H=2
  - Same 5 synthetic domains: arithmetic, reverse, repeat, sort, parity
  - Train FP16 base, quantize to ternary, train adapters on BOTH bases
  - Compute FULL pairwise cosine matrix (10 pairs from 5 domains)
  - 3 seeds for statistical confidence
  - Report: mean |cos|, max |cos|, per-pair breakdown, paired t-test

Reuses data/model/training infrastructure from bitnet_composition_stability.

Micro scale: d=64, r=4, L=2. CPU only, ~5 min.
"""

import json
import time
import sys
from pathlib import Path
from itertools import combinations

import autograd.numpy as np
from autograd import grad
import numpy as onp

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bitnet_composition_stability"))
from bitnet_composition_stability import (
    DOMAIN_GENERATORS, CharTokenizer,
    init_model, quantize_model_to_ternary, forward, compute_loss,
    init_lora, apply_lora, train_lora, lora_to_delta,
    eval_loss, _prepare_batch, flatten_delta, cosine_sim
)


def train_base(params, data, pad_id, epochs=30, lr=0.001, batch_size=32,
               clip_grad=1.0, verbose=True):
    """Train base model (all params). Copied from composition_stability."""
    cfg = params['_config']
    trainable_keys = [k for k in sorted(params.keys()) if k != '_config']

    def loss_fn(param_vals, inp, tgt, mask):
        p = dict(zip(trainable_keys, param_vals))
        p['_config'] = cfg
        return compute_loss(p, inp, tgt, mask, pad_id)

    grad_fn = grad(loss_fn)
    m_state = [onp.zeros_like(params[k]) for k in trainable_keys]
    v_state = [onp.zeros_like(params[k]) for k in trainable_keys]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(data)
    rng = onp.random.RandomState(42)

    for epoch in range(epochs):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [data[j] for j in batch_idx]
            inp, tgt, mask = _prepare_batch(batch, pad_id)
            if onp.sum(mask) == 0:
                continue
            param_vals = [params[k] for k in trainable_keys]
            loss_val = float(loss_fn(param_vals, inp, tgt, mask))
            grads = grad_fn(param_vals, inp, tgt, mask)
            grad_norm = onp.sqrt(sum(float(onp.sum(g**2)) for g in grads))
            if grad_norm > clip_grad:
                sc = clip_grad / grad_norm
                grads = [g * sc for g in grads]
            step += 1
            for k_idx, key in enumerate(trainable_keys):
                g = onp.array(grads[k_idx])
                m_state[k_idx] = beta1 * m_state[k_idx] + (1 - beta1) * g
                v_state[k_idx] = beta2 * v_state[k_idx] + (1 - beta2) * g**2
                m_hat = m_state[k_idx] / (1 - beta1**step)
                v_hat = v_state[k_idx] / (1 - beta2**step)
                params[key] = params[key] - lr * m_hat / (onp.sqrt(v_hat) + eps)
            epoch_loss += loss_val
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}: loss={avg_loss:.4f}")

    return params


def compute_pairwise_cosines(deltas, domain_names):
    """Compute full pairwise |cos| matrix for all domain adapter pairs.

    Returns:
      matrix: NxN |cos| matrix (symmetric, diagonal=1.0)
      pairs: list of (name_i, name_j, |cos|) for all i<j
    """
    N = len(domain_names)
    flat = {}
    for name in domain_names:
        flat[name] = flatten_delta(deltas[name])

    matrix = onp.zeros((N, N))
    pairs = []

    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                cos = abs(cosine_sim(flat[domain_names[i]], flat[domain_names[j]]))
                matrix[i, j] = cos
                matrix[j, i] = cos
                pairs.append((domain_names[i], domain_names[j], cos))

    return matrix, pairs


def compute_delta_norms(deltas, domain_names):
    """Compute Frobenius norms of adapter deltas."""
    norms = {}
    for name in domain_names:
        flat = flatten_delta(deltas[name])
        norms[name] = float(onp.linalg.norm(flat))
    return norms


def compute_effective_rank(deltas, domain_names):
    """Compute effective rank (Shannon entropy of singular values) for each adapter."""
    ranks = {}
    for name in domain_names:
        flat = flatten_delta(deltas[name])
        # Reshape to approximate matrix form for SVD
        # Use the largest weight matrix's delta as representative
        for k in sorted(deltas[name].keys()):
            W = deltas[name][k]
            if W.ndim == 2 and W.shape[0] >= 4 and W.shape[1] >= 4:
                s = onp.linalg.svd(W, compute_uv=False)
                s = s[s > 1e-10]
                if len(s) > 0:
                    p = s / s.sum()
                    entropy = -onp.sum(p * onp.log(p + 1e-10))
                    ranks[name] = float(onp.exp(entropy))
                    break
        if name not in ranks:
            ranks[name] = 0.0
    return ranks


def run_seed(seed, d=64, r=4, L=2, H=2, n_data=300, n_eval=100,
             base_epochs=30, lora_epochs=30, verbose=True):
    """Run one seed: train base, quantize, train adapters, measure cosines."""

    print(f"\n{'='*70}")
    print(f"  BitNet Orthogonality Trained (seed={seed}, d={d}, r={r})")
    print(f"{'='*70}")

    tok = CharTokenizer()
    V = tok.vocab_size
    rng = onp.random.RandomState(seed)
    domain_names = list(DOMAIN_GENERATORS.keys())

    # Step 1: Generate data
    print("\n[1/5] Generating domain data...")
    domain_data = {}
    domain_eval = {}
    mixed_train = []
    for name, gen_fn in DOMAIN_GENERATORS.items():
        train = gen_fn(n_data, rng)
        test = gen_fn(n_eval, rng)
        domain_data[name] = [tok.encode(s) for s in train]
        domain_eval[name] = [tok.encode(s) for s in test]
        mixed_train.extend(train[:n_data // len(DOMAIN_GENERATORS)])
    mixed_encoded = [tok.encode(s) for s in mixed_train]

    # Step 2: Train FP16 base
    print("\n[2/5] Training FP16 base model...")
    fp16_base = init_model(V, d=d, H=H, L=L, seed=seed)
    fp16_base = train_base(fp16_base, mixed_encoded, tok.pad_id,
                           epochs=base_epochs, verbose=verbose)

    # Step 3: Create ternary base
    print("\n[3/5] Quantizing base to ternary (BitNet absmean)...")
    ternary_base, scales = quantize_model_to_ternary(fp16_base)

    # Step 4: Train LoRA adapters on both bases
    print("\n[4/5] Training LoRA adapters on both bases...")
    fp16_deltas = {}
    ternary_deltas = {}
    fp16_final_losses = {}
    ternary_final_losses = {}

    for di, name in enumerate(domain_names):
        lora_seed = seed * 100 + di

        # FP16 base adapter
        print(f"  FP16 adapter: {name}...")
        lora_fp16 = init_lora(fp16_base, rank=r, seed=lora_seed)
        lora_fp16 = train_lora(fp16_base, lora_fp16, domain_data[name], tok.pad_id,
                               epochs=lora_epochs, lr=0.003, verbose=False)
        fp16_deltas[name] = lora_to_delta(lora_fp16, fp16_base)

        # Record final training loss
        effective = apply_lora(fp16_base, lora_fp16)
        fp16_final_losses[name] = eval_loss(effective, domain_eval[name], tok.pad_id)

        # Ternary base adapter
        print(f"  Ternary adapter: {name}...")
        lora_ternary = init_lora(ternary_base, rank=r, seed=lora_seed)
        lora_ternary = train_lora(ternary_base, lora_ternary, domain_data[name], tok.pad_id,
                                  epochs=lora_epochs, lr=0.003, verbose=False)
        ternary_deltas[name] = lora_to_delta(lora_ternary, ternary_base)
        effective_t = apply_lora(ternary_base, lora_ternary)
        ternary_final_losses[name] = eval_loss(effective_t, domain_eval[name], tok.pad_id)

    # Step 5: Compute pairwise cosines
    print("\n[5/5] Computing pairwise cosine similarities...")

    fp16_matrix, fp16_pairs = compute_pairwise_cosines(fp16_deltas, domain_names)
    ternary_matrix, ternary_pairs = compute_pairwise_cosines(ternary_deltas, domain_names)

    fp16_norms = compute_delta_norms(fp16_deltas, domain_names)
    ternary_norms = compute_delta_norms(ternary_deltas, domain_names)

    # Identify high-overlap pair (arithmetic-sort proxy for math-medical)
    arith_sort_fp16 = None
    arith_sort_ternary = None
    for name_i, name_j, cos in fp16_pairs:
        if (name_i == 'arithmetic' and name_j == 'sort') or \
           (name_i == 'sort' and name_j == 'arithmetic'):
            arith_sort_fp16 = cos
    for name_i, name_j, cos in ternary_pairs:
        if (name_i == 'arithmetic' and name_j == 'sort') or \
           (name_i == 'sort' and name_j == 'arithmetic'):
            arith_sort_ternary = cos

    # Summary stats
    fp16_cos_values = [cos for _, _, cos in fp16_pairs]
    ternary_cos_values = [cos for _, _, cos in ternary_pairs]

    fp16_mean_cos = float(onp.mean(fp16_cos_values))
    ternary_mean_cos = float(onp.mean(ternary_cos_values))
    fp16_max_cos = float(onp.max(fp16_cos_values))
    ternary_max_cos = float(onp.max(ternary_cos_values))
    fp16_median_cos = float(onp.median(fp16_cos_values))
    ternary_median_cos = float(onp.median(ternary_cos_values))

    # Print results
    print(f"\n  === Pairwise |cos| Summary ===")
    print(f"  FP16:    mean={fp16_mean_cos:.4f}, max={fp16_max_cos:.4f}, "
          f"median={fp16_median_cos:.4f}")
    print(f"  Ternary: mean={ternary_mean_cos:.4f}, max={ternary_max_cos:.4f}, "
          f"median={ternary_median_cos:.4f}")
    print(f"  Delta:   mean={ternary_mean_cos - fp16_mean_cos:+.4f} "
          f"({'ternary worse' if ternary_mean_cos >= fp16_mean_cos else 'ternary better'})")

    print(f"\n  Arithmetic-Sort pair (math-medical proxy):")
    print(f"  FP16:    |cos|={arith_sort_fp16:.4f}")
    print(f"  Ternary: |cos|={arith_sort_ternary:.4f}")

    print(f"\n  Full pairwise breakdown:")
    for i, ((n_i, n_j, cos_fp), (_, _, cos_t)) in enumerate(zip(fp16_pairs, ternary_pairs)):
        delta = cos_t - cos_fp
        print(f"    {n_i:12s}-{n_j:12s}: FP16={cos_fp:.4f}, Ternary={cos_t:.4f}, "
              f"delta={delta:+.4f}")

    # Build result dict
    result = {
        'seed': seed,
        'd': d,
        'r': r,
        'L': L,
        'fp16': {
            'mean_cos': fp16_mean_cos,
            'max_cos': fp16_max_cos,
            'median_cos': fp16_median_cos,
            'pairwise': {f"{n_i}_{n_j}": cos for n_i, n_j, cos in fp16_pairs},
            'arithmetic_sort_cos': arith_sort_fp16,
            'delta_norms': fp16_norms,
            'final_losses': {k: float(v) for k, v in fp16_final_losses.items()},
            'cos_matrix': fp16_matrix.tolist(),
        },
        'ternary': {
            'mean_cos': ternary_mean_cos,
            'max_cos': ternary_max_cos,
            'median_cos': ternary_median_cos,
            'pairwise': {f"{n_i}_{n_j}": cos for n_i, n_j, cos in ternary_pairs},
            'arithmetic_sort_cos': arith_sort_ternary,
            'delta_norms': ternary_norms,
            'final_losses': {k: float(v) for k, v in ternary_final_losses.items()},
            'cos_matrix': ternary_matrix.tolist(),
        },
        'domain_names': domain_names,
        'kill_criteria': {
            'K1_fp16_mean_cos': fp16_mean_cos,
            'K1_ternary_mean_cos': ternary_mean_cos,
            'K1_pass': bool(ternary_mean_cos < fp16_mean_cos),
            'K2_ternary_arith_sort_cos': arith_sort_ternary,
            'K2_threshold': 0.5,
            'K2_pass': bool(arith_sort_ternary < 0.5),
        },
    }

    return result


def main():
    t0 = time.time()

    seeds = [42, 123, 314]
    all_results = []

    for seed in seeds:
        result = run_seed(seed, verbose=True)
        all_results.append(result)

    # Aggregate across seeds
    fp16_means = [r['fp16']['mean_cos'] for r in all_results]
    ternary_means = [r['ternary']['mean_cos'] for r in all_results]
    fp16_maxes = [r['fp16']['max_cos'] for r in all_results]
    ternary_maxes = [r['ternary']['max_cos'] for r in all_results]

    fp16_arith_sort = [r['fp16']['arithmetic_sort_cos'] for r in all_results]
    ternary_arith_sort = [r['ternary']['arithmetic_sort_cos'] for r in all_results]

    # Paired differences for statistical test
    pair_diffs_mean = [t - f for t, f in zip(ternary_means, fp16_means)]
    pair_diffs_arith = [t - f for t, f in zip(ternary_arith_sort, fp16_arith_sort)]

    # One-sample t-test on paired differences (H0: diff = 0)
    def paired_t_stat(diffs):
        n = len(diffs)
        mean_d = onp.mean(diffs)
        std_d = onp.std(diffs, ddof=1)
        if std_d < 1e-10:
            return float('inf') if mean_d > 0 else float('-inf'), 0.0
        t_stat = mean_d / (std_d / onp.sqrt(n))
        # Two-tailed p-value approximation (t-distribution with n-1 df)
        # For n=3, df=2: critical values 4.303 (p=0.05), 9.925 (p=0.01)
        return float(t_stat), float(mean_d)

    t_mean, diff_mean = paired_t_stat(pair_diffs_mean)
    t_arith, diff_arith = paired_t_stat(pair_diffs_arith)

    # Per-pair aggregation across seeds
    pair_names = list(all_results[0]['fp16']['pairwise'].keys())
    per_pair_agg = {}
    for pname in pair_names:
        fp16_vals = [r['fp16']['pairwise'][pname] for r in all_results]
        ternary_vals = [r['ternary']['pairwise'][pname] for r in all_results]
        per_pair_agg[pname] = {
            'fp16_mean': float(onp.mean(fp16_vals)),
            'fp16_std': float(onp.std(fp16_vals)),
            'ternary_mean': float(onp.mean(ternary_vals)),
            'ternary_std': float(onp.std(ternary_vals)),
            'delta': float(onp.mean(ternary_vals) - onp.mean(fp16_vals)),
            'ternary_lower': bool(onp.mean(ternary_vals) < onp.mean(fp16_vals)),
        }

    n_ternary_lower = sum(1 for v in per_pair_agg.values() if v['ternary_lower'])
    n_total_pairs = len(per_pair_agg)

    # Print aggregate
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")

    print(f"\n  Mean |cos| (across all pairs):")
    print(f"    FP16:    {onp.mean(fp16_means):.4f} +/- {onp.std(fp16_means):.4f}")
    print(f"    Ternary: {onp.mean(ternary_means):.4f} +/- {onp.std(ternary_means):.4f}")
    print(f"    Paired diff: {diff_mean:+.4f} (t={t_mean:.3f}, n=3)")
    print(f"    Direction: {'ternary worse (KILLS K1)' if diff_mean >= 0 else 'ternary better (K1 survives)'}")

    print(f"\n  Max |cos| (worst pair):")
    print(f"    FP16:    {onp.mean(fp16_maxes):.4f} +/- {onp.std(fp16_maxes):.4f}")
    print(f"    Ternary: {onp.mean(ternary_maxes):.4f} +/- {onp.std(ternary_maxes):.4f}")

    print(f"\n  Arithmetic-Sort pair (math-medical proxy):")
    print(f"    FP16:    {onp.mean(fp16_arith_sort):.4f} +/- {onp.std(fp16_arith_sort):.4f}")
    print(f"    Ternary: {onp.mean(ternary_arith_sort):.4f} +/- {onp.std(ternary_arith_sort):.4f}")
    print(f"    Paired diff: {diff_arith:+.4f} (t={t_arith:.3f})")
    print(f"    K2 threshold: 0.5, ternary value: {onp.mean(ternary_arith_sort):.4f}")

    print(f"\n  Per-pair ternary vs FP16 (mean across seeds):")
    for pname, agg in per_pair_agg.items():
        direction = "better" if agg['ternary_lower'] else "WORSE"
        print(f"    {pname:25s}: FP16={agg['fp16_mean']:.4f}, "
              f"Ternary={agg['ternary_mean']:.4f}, delta={agg['delta']:+.4f} ({direction})")

    print(f"\n  Pairs where ternary is lower: {n_ternary_lower}/{n_total_pairs}")

    # Kill criteria assessment
    k1_pass_all = all(r['kill_criteria']['K1_pass'] for r in all_results)
    k1_pass_aggregate = onp.mean(ternary_means) < onp.mean(fp16_means)
    k2_pass_all = all(r['kill_criteria']['K2_pass'] for r in all_results)

    print(f"\n  === KILL CRITERIA ===")
    print(f"  K1 (mean |cos| ternary < FP16): {'PASS' if k1_pass_aggregate else 'KILLED'}")
    print(f"     Per-seed: {sum(1 for r in all_results if r['kill_criteria']['K1_pass'])}/3")
    print(f"     Aggregate: FP16={onp.mean(fp16_means):.4f} vs Ternary={onp.mean(ternary_means):.4f}")
    print(f"  K2 (arith-sort ternary < 0.5): {'PASS' if k2_pass_all else 'KILLED'}")
    print(f"     Per-seed: {sum(1 for r in all_results if r['kill_criteria']['K2_pass'])}/3")
    print(f"     Aggregate: {onp.mean(ternary_arith_sort):.4f} (threshold 0.5)")

    elapsed = time.time() - t0

    output = {
        'config': {
            'd': 64, 'r': 4, 'L': 2, 'H': 2,
            'seeds': seeds, 'n_domains': 5,
            'base_epochs': 30, 'lora_epochs': 30,
        },
        'per_seed': all_results,
        'aggregate': {
            'fp16_mean_cos': {
                'mean': float(onp.mean(fp16_means)),
                'std': float(onp.std(fp16_means)),
            },
            'ternary_mean_cos': {
                'mean': float(onp.mean(ternary_means)),
                'std': float(onp.std(ternary_means)),
            },
            'fp16_max_cos': {
                'mean': float(onp.mean(fp16_maxes)),
                'std': float(onp.std(fp16_maxes)),
            },
            'ternary_max_cos': {
                'mean': float(onp.mean(ternary_maxes)),
                'std': float(onp.std(ternary_maxes)),
            },
            'fp16_arith_sort_cos': {
                'mean': float(onp.mean(fp16_arith_sort)),
                'std': float(onp.std(fp16_arith_sort)),
            },
            'ternary_arith_sort_cos': {
                'mean': float(onp.mean(ternary_arith_sort)),
                'std': float(onp.std(ternary_arith_sort)),
            },
            'paired_t_test_mean_cos': {
                't_stat': t_mean,
                'mean_diff': diff_mean,
                'direction': 'ternary_worse' if diff_mean >= 0 else 'ternary_better',
            },
            'paired_t_test_arith_sort': {
                't_stat': t_arith,
                'mean_diff': diff_arith,
            },
            'per_pair': per_pair_agg,
            'n_pairs_ternary_lower': n_ternary_lower,
            'n_total_pairs': n_total_pairs,
        },
        'kill_criteria': {
            'K1_pass': bool(k1_pass_aggregate),
            'K1_per_seed_pass_rate': f"{sum(1 for r in all_results if r['kill_criteria']['K1_pass'])}/3",
            'K1_direction': 'ternary_worse' if not k1_pass_aggregate else 'ternary_better',
            'K2_pass': bool(k2_pass_all),
            'K2_per_seed_pass_rate': f"{sum(1 for r in all_results if r['kill_criteria']['K2_pass'])}/3",
            'K2_ternary_arith_sort_mean': float(onp.mean(ternary_arith_sort)),
        },
        'runtime_seconds': elapsed,
    }

    # Save
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")

    return output


if __name__ == '__main__':
    main()
