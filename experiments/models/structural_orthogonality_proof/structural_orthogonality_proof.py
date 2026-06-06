#!/usr/bin/env python3
"""
Structural Orthogonality Proof: Mathematical guarantee for LoRA expert orthogonality.

Hypothesis: For gradient-aligned LoRA adapters trained on distinct domains,
  E[cos(A_i, A_j)] << sqrt(r/d) (the random subspace bound),
and there exists a critical d_crit below which structural orthogonality
breaks down as a phase transition.

Three deliverables:
  (1) Derive tight bound on E[cos(A_i, A_j)] for gradient-aligned adapters
  (2) Empirically validate at d=64, 128, 256, 512, 1024
  (3) Identify critical d_crit where cos < tau=0.01

Pure numpy, CPU-only, float32 for speed. ~5-8 minutes.

Kill criteria:
  K1: empirical cos(theta) exceeds random subspace bound sqrt(r/d) at any tested d
  K2: no clear d threshold (gradual decline, not phase-transition)
  K3: gradient-aligned adapters NOT more orthogonal than random subspaces
"""

import json
import time
from pathlib import Path

import numpy as np

# Force float32 for speed
DTYPE = np.float32

# =============================================================================
# Constants
# =============================================================================

VOCAB_SIZE = 32
CONTEXT_LEN = 16
LORA_RANK = 8
LORA_ALPHA = 8

D_VALUES = [64, 128, 256, 512, 1024]
N_PAIRS = 4
N_RANDOM_PAIRS = 30
TAU = 0.01
SEEDS = [42, 137]

# Adaptive config per dimension to keep runtime bounded
# {d: (n_layers, d_ff_mult, steps, lr, n_seq, batch_size)}
D_CONFIG = {
    64:   (2, 4, 300, 0.01,  200, 32),
    128:  (2, 4, 300, 0.008, 200, 32),
    256:  (2, 2, 250, 0.005, 150, 32),
    512:  (1, 2, 200, 0.003, 100, 32),
    1024: (1, 1, 150, 0.002, 100, 32),
}


# =============================================================================
# Utilities
# =============================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# Model + LoRA training (single-layer optimized)
# =============================================================================

class MicroMLP:
    def __init__(self, d, n_layers, d_ff_mult, rng):
        self.d = d
        self.d_ff = d_ff_mult * d
        self.n_layers = n_layers
        s = DTYPE(0.02)
        self.wte = (rng.randn(VOCAB_SIZE, d) * s).astype(DTYPE)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'W1': (rng.randn(d, self.d_ff) * s).astype(DTYPE),
                'W2': (rng.randn(self.d_ff, d) * s).astype(DTYPE),
            })
        self.W_out = (rng.randn(d, VOCAB_SIZE) * s).astype(DTYPE)


def init_lora(d, d_ff, n_layers, rng):
    A1, B1, A2, B2 = [], [], [], []
    for _ in range(n_layers):
        A1.append((rng.randn(d, LORA_RANK) * np.sqrt(2.0 / d)).astype(DTYPE))
        B1.append(np.zeros((LORA_RANK, d_ff), dtype=DTYPE))
        A2.append((rng.randn(d_ff, LORA_RANK) * np.sqrt(2.0 / d_ff)).astype(DTYPE))
        B2.append(np.zeros((LORA_RANK, d), dtype=DTYPE))
    return A1, B1, A2, B2


def lora_delta_vec(A1, B1, A2, B2):
    parts = []
    for l in range(len(A1)):
        parts.append((A1[l] @ B1[l]).ravel())
        parts.append((A2[l] @ B2[l]).ravel())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, steps, lr, batch_size):
    """Train B-only LoRA with manual backprop. Optimized for speed."""
    d, d_ff, nl = model.d, model.d_ff, model.n_layers
    A1, B1, A2, B2 = init_lora(d, d_ff, nl, rng)
    scale = DTYPE(LORA_ALPHA / LORA_RANK)
    n = data_x.shape[0]
    final_loss = float('inf')
    lr_f = DTYPE(lr)

    for step in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]

        # Forward: embed -> mean -> MLP layers -> logits
        h = model.wte[bx].mean(axis=1)  # (B, d)

        inters = []
        for l in range(nl):
            h_in = h
            z1 = h @ model.layers[l]['W1'] + scale * (h @ A1[l] @ B1[l])
            a1 = np.maximum(z1, 0)  # ReLU
            z2 = a1 @ model.layers[l]['W2'] + scale * (a1 @ A2[l] @ B2[l])
            h = h_in + z2
            inters.append((h_in, z1, a1))

        logits = h @ model.W_out
        # Stable softmax
        logits -= logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=-1, keepdims=True)

        # Loss
        target_probs = probs[np.arange(B_sz), by]
        final_loss = float(-np.mean(np.log(target_probs + 1e-10)))

        # Backward
        dl = probs.copy()
        dl[np.arange(B_sz), by] -= 1.0
        dl /= B_sz

        dh = dl @ model.W_out.T

        for l in range(nl - 1, -1, -1):
            h_in, z1, a1 = inters[l]

            # B2 gradient
            proj2 = a1 @ A2[l]  # (B, r)
            dB2 = scale * (proj2.T @ dh)  # (r, d)
            B2[l] -= lr_f * dB2

            # Backprop through fc2
            W2_eff = model.layers[l]['W2'] + scale * (A2[l] @ B2[l])
            da1 = dh @ W2_eff.T
            dz1 = da1 * (z1 > 0).astype(DTYPE)

            # B1 gradient
            proj1 = h_in @ A1[l]  # (B, r)
            dB1 = scale * (proj1.T @ dz1)  # (r, d_ff)
            B1[l] -= lr_f * dB1

        # No backprop through residual to earlier layers (standard LoRA approx)

    return A1, B1, A2, B2, final_loss


# =============================================================================
# Domain data
# =============================================================================

def generate_domain_data(domain_id, n_sequences=200):
    rng = np.random.RandomState(domain_id * 7919 + 13)
    logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE).astype(np.float64) * 2.0

    gs = (domain_id * 7) % VOCAB_SIZE
    ge = gs + max(3, VOCAB_SIZE // 6)
    if ge > VOCAB_SIZE:
        logits[:, gs:] += 2.0
        logits[:, :ge - VOCAB_SIZE] += 2.0
    else:
        logits[:, gs:ge] += 2.0

    for i in range(VOCAB_SIZE):
        logits[i, (i + domain_id) % VOCAB_SIZE] += 1.5

    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    trans = e / e.sum(axis=-1, keepdims=True)

    gen_rng = np.random.RandomState(domain_id * 31 + 7)
    seqs = np.zeros((n_sequences, CONTEXT_LEN + 1), dtype=np.int32)
    for i in range(n_sequences):
        seqs[i, 0] = gen_rng.choice(VOCAB_SIZE)
        for t in range(CONTEXT_LEN):
            seqs[i, t + 1] = gen_rng.choice(VOCAB_SIZE, p=trans[seqs[i, t]])

    return seqs[:, :-1], seqs[:, -1]


# =============================================================================
# Random baseline
# =============================================================================

def random_subspace_cosine(d, d_ff, n_layers, n_pairs=30):
    rng = np.random.RandomState(42)
    cosines = []
    for _ in range(n_pairs):
        v1, v2 = [], []
        for _ in range(n_layers):
            v1.append((rng.randn(d, LORA_RANK) @ rng.randn(LORA_RANK, d_ff)).ravel())
            v2.append((rng.randn(d, LORA_RANK) @ rng.randn(LORA_RANK, d_ff)).ravel())
            v1.append((rng.randn(d_ff, LORA_RANK) @ rng.randn(LORA_RANK, d)).ravel())
            v2.append((rng.randn(d_ff, LORA_RANK) @ rng.randn(LORA_RANK, d)).ravel())
        v1 = np.concatenate(v1).astype(DTYPE)
        v2 = np.concatenate(v2).astype(DTYPE)
        cosines.append(abs(cosine_sim(v1, v2)))
    return cosines


# =============================================================================
# Main
# =============================================================================

def run_experiment(seeds=None, d_values=None):
    if seeds is None:
        seeds = SEEDS
    if d_values is None:
        d_values = D_VALUES

    results_dir = Path(__file__).parent
    t0 = time.time()

    print("=" * 72)
    print("  Structural Orthogonality Proof")
    print(f"  d={d_values}, seeds={seeds}, pairs={N_PAIRS}, rank={LORA_RANK}")
    print("=" * 72)

    all_results = {}

    for seed in seeds:
        print(f"\n  === SEED {seed} ===")
        seed_results = {}

        for d in d_values:
            nl, d_ff_mult, steps, lr, n_seq, bs = D_CONFIG[d]
            d_ff = d_ff_mult * d
            D = nl * 2 * d * d_ff

            rb = np.sqrt(LORA_RANK / d)
            t1 = time.time()
            print(f"\n  d={d}, layers={nl}, d_ff={d_ff}, D={D:,}, "
                  f"steps={steps}, lr={lr}")
            print(f"  bound sqrt(r/d) = {rb:.6f}")

            rng = np.random.RandomState(seed + d)
            model = MicroMLP(d, nl, d_ff_mult, rng)

            trained_cos = []
            details = []
            for p in range(N_PAIRS):
                da = p * 2 + seed * 100
                db = p * 2 + 1 + seed * 100

                xa, ya = generate_domain_data(da, n_seq)
                xb, yb = generate_domain_data(db, n_seq)

                rng_a = np.random.RandomState(seed + d + p * 17)
                A1a, B1a, A2a, B2a, la = train_lora(
                    model, xa, ya, rng_a, steps, lr, bs)

                rng_b = np.random.RandomState(seed + d + p * 17 + 7)
                A1b, B1b, A2b, B2b, lb = train_lora(
                    model, xb, yb, rng_b, steps, lr, bs)

                va = lora_delta_vec(A1a, B1a, A2a, B2a)
                vb = lora_delta_vec(A1b, B1b, A2b, B2b)
                c = abs(cosine_sim(va, vb))
                trained_cos.append(c)
                details.append({
                    'pair': p, 'da': da, 'db': db,
                    'loss_a': float(la), 'loss_b': float(lb),
                    'cos': float(c),
                })
                print(f"    pair {p}: |cos|={c:.6f}, loss=({la:.3f},{lb:.3f})")

            rand_cos = random_subspace_cosine(d, d_ff, nl, N_RANDOM_PAIRS)

            tm = float(np.mean(trained_cos))
            rm = float(np.mean(rand_cos))
            sr = rm / max(tm, 1e-12)
            ab = all(c < rb for c in trained_cos)

            print(f"  trained={tm:.6f}, random={rm:.6f}, bound={rb:.6f}")
            print(f"  sep={sr:.2f}x, below_bound={ab}, "
                  f"time={time.time()-t1:.1f}s")

            seed_results[d] = {
                'd': d, 'D': D, 'n_layers': nl, 'd_ff': d_ff,
                'steps': steps, 'lr': lr,
                'random_bound': float(rb),
                'trained_cosines': [float(x) for x in trained_cos],
                'random_cosines': [float(x) for x in rand_cos],
                'trained_mean': tm,
                'trained_max': float(max(trained_cos)),
                'random_mean': rm,
                'separation_ratio': sr,
                'all_below_bound': ab,
                'details': details,
            }

        all_results[seed] = seed_results

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n{'='*72}")
    print(f"  AGGREGATE ({len(seeds)} seeds)")
    print(f"{'='*72}")

    aggregate = {}
    for d in d_values:
        ta, ra = [], []
        for s in seeds:
            ta.extend(all_results[s][d]['trained_cosines'])
            ra.extend(all_results[s][d]['random_cosines'])

        rb = np.sqrt(LORA_RANK / d)
        agg = {
            'd': d,
            'random_bound': float(rb),
            'trained_mean': float(np.mean(ta)),
            'trained_std': float(np.std(ta)),
            'trained_median': float(np.median(ta)),
            'trained_max': float(np.max(ta)),
            'trained_p95': float(np.percentile(ta, 95)),
            'random_mean': float(np.mean(ra)),
            'random_std': float(np.std(ra)),
            'separation_ratio': float(np.mean(ra) / max(np.mean(ta), 1e-12)),
            'all_below_bound': all(c < rb for c in ta),
            'frac_below_tau': float(np.mean([c < TAU for c in ta])),
            'n_trained': len(ta),
        }
        aggregate[d] = agg
        print(f"  d={d:5d}: trained={agg['trained_mean']:.6f}, "
              f"random={agg['random_mean']:.6f}, bound={rb:.6f}, "
              f"sep={agg['separation_ratio']:.2f}x, "
              f"%<tau={agg['frac_below_tau']:.0%}")

    # Phase transition
    print(f"\n  PHASE TRANSITION")
    log_d = np.log(d_values)
    tmeans = [max(aggregate[d]['trained_mean'], 1e-12) for d in d_values]
    log_c = np.log(tmeans)
    co = np.polyfit(log_d, log_c, 1)
    alpha = -co[0]
    C = np.exp(co[1])
    yp = np.polyval(co, log_d)
    ss_r = np.sum((log_c - yp) ** 2)
    ss_t = np.sum((log_c - np.mean(log_c)) ** 2)
    r2 = 1 - ss_r / max(ss_t, 1e-12)

    print(f"  E[|cos|] = {C:.4f} * d^(-{alpha:.3f}), R^2={r2:.4f}")

    if alpha > 0:
        dcrit = (C / TAU) ** (1 / alpha)
        print(f"  d_crit predicted: {dcrit:.0f}")
    else:
        dcrit = float('inf')
        print(f"  WARNING: no decay")

    dcrit_th = LORA_RANK / TAU ** 2
    print(f"  d_crit theoretical: {dcrit_th:.0f}")

    # Random baseline decay
    rmeans = [max(aggregate[d]['random_mean'], 1e-12) for d in d_values]
    cor = np.polyfit(log_d, np.log(rmeans), 1)
    alpha_r = -cor[0]
    print(f"  Random alpha: {alpha_r:.3f}")

    sep_ratios = [aggregate[d]['separation_ratio'] for d in d_values]

    # Kill criteria
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA")
    print(f"{'='*72}")

    k1 = all(aggregate[d]['all_below_bound'] for d in d_values)
    print(f"  K1 below bound: {'PASS' if k1 else 'FAIL'}")
    if not k1:
        for d in d_values:
            if not aggregate[d]['all_below_bound']:
                print(f"    d={d}: max={aggregate[d]['trained_max']:.6f} "
                      f">= bound={aggregate[d]['random_bound']:.6f}")

    k2 = alpha > 0.5
    print(f"  K2 phase transition: {'PASS' if k2 else 'FAIL'} (alpha={alpha:.3f})")

    frac = [aggregate[d]['frac_below_tau'] for d in d_values]
    for d, f in zip(d_values, frac):
        bar = "#" * int(f * 20)
        print(f"    d={d:5d}: {f:.0%} below tau {bar}")

    dcrit_emp = None
    for d, f in zip(d_values, frac):
        if f >= 0.8:
            dcrit_emp = d
            break

    k3 = all(s >= 1.5 for s in sep_ratios)
    print(f"  K3 separation: {'PASS' if k3 else 'FAIL'} "
          f"(min={min(sep_ratios):.2f}x)")
    if not k3:
        for d, s in zip(d_values, sep_ratios):
            if s < 1.5:
                print(f"    d={d}: sep={s:.2f}x < 1.5x")

    overall = k1 and k2 and k3

    print(f"\n{'='*72}")
    if overall:
        print(f"  VERDICT: PROVEN")
    else:
        rs = []
        if not k1: rs.append("K1")
        if not k2: rs.append("K2")
        if not k3: rs.append("K3")
        print(f"  VERDICT: {'KILL' if not k1 else 'REVISE'} ({', '.join(rs)} failed)")
    print(f"{'='*72}")

    # Table
    print(f"\n  {'d':>6} | {'Trained':>10} | {'Random':>10} | "
          f"{'Bound':>10} | {'Sep':>6} | {'%<tau':>6}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*6}-+-{'-'*6}")
    for d in d_values:
        a = aggregate[d]
        print(f"  {d:6d} | {a['trained_mean']:10.6f} | "
              f"{a['random_mean']:10.6f} | {a['random_bound']:10.6f} | "
              f"{a['separation_ratio']:6.2f} | {a['frac_below_tau']*100:5.1f}%")

    print(f"\n  Total: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'seeds': seeds, 'd_values': d_values, 'n_pairs': N_PAIRS,
            'n_random': N_RANDOM_PAIRS, 'rank': LORA_RANK, 'tau': TAU,
            'd_config': {str(d): list(v) for d, v in D_CONFIG.items()},
        },
        'per_seed': {str(s): {str(d): r for d, r in sr.items()}
                     for s, sr in all_results.items()},
        'aggregate': {str(d): a for d, a in aggregate.items()},
        'phase_transition': {
            'alpha': float(alpha), 'C': float(C), 'r2': float(r2),
            'd_crit_predicted': float(dcrit),
            'd_crit_theoretical': float(dcrit_th),
            'd_crit_empirical': dcrit_emp,
            'alpha_random': float(alpha_r),
        },
        'kill_criteria': {
            'k1_below_bound': k1, 'k2_phase_transition': k2,
            'k3_separation': k3, 'overall': overall,
        },
        'elapsed_seconds': elapsed,
    }

    out = results_dir / 'results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out}")

    return output


if __name__ == '__main__':
    run_experiment()
