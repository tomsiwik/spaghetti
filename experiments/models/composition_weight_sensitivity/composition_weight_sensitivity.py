#!/usr/bin/env python3
"""
Composition Weight Sensitivity: Pre-merge dilution test across N experts.

Hypothesis: Pre-merged SOLE quality degrades gracefully as expert count N
grows. Each expert's marginal contribution (delta loss from adding/removing
one expert) remains detectable above the noise floor up to N=100.

Method:
  1. Train a base MLP on mixed-domain Markov chain data (4 layers, d=64)
  2. Train N domain-specific LoRA experts (rank-8) independently
  3. Pre-merge all N via averaging: W + (1/N) * sum(B_i @ A_i)
  4. Measure aggregate quality: loss_composed vs loss_base
  5. Measure per-expert marginal contribution:
       delta_loss_i = loss(all N merged) - loss(all except i merged)
  6. Establish noise floor via random perturbation control
  7. Sweep N = 2, 5, 10, 20, 50, 100

Kill criteria:
  K1: Per-expert signal (delta_loss_i) drops below noise floor at N < 50
  K2: Pre-merge quality degrades >20% at N=100 vs N=10

Architecture reused from lora_flow_comparison.py (same MLP + LoRA).
Runtime: ~3-5 minutes on Apple Silicon (numpy, CPU only).
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

# Force unbuffered output
def log(msg):
    print(msg, flush=True)

# =============================================================================
# Constants
# =============================================================================
VOCAB_SIZE = 32
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8
SEEDS = [42, 137, 2024]

BASE_LR = 0.05
EXPERT_LR = 0.02
BASE_STEPS = 300
EXPERT_STEPS = 150  # reduced for speed; sufficient for signal at micro
BATCH_SIZE = 64
SEQ_LEN = 16
EVAL_SAMPLES = 128

N_VALUES = [2, 5, 10, 20, 50, 100]
MAX_MARGINAL_SAMPLES = 10
RESULTS_DIR = Path(__file__).parent


# =============================================================================
# Data generation (vectorized)
# =============================================================================
def make_transition_matrix(domain_idx, n_domains, rng):
    V = VOCAB_SIZE
    chunk = max(1, V // max(n_domains, 4))
    base = rng.dirichlet(np.ones(V) * 0.4, size=V).astype(np.float32)
    start = (domain_idx * chunk) % V
    end = min(start + chunk, V)
    base[:, start:end] *= 2.5
    noise = rng.dirichlet(np.ones(V) * 1.0, size=V).astype(np.float32)
    alpha = 0.1 + 0.03 * (domain_idx % 10)
    mixed = (1 - alpha) * base + alpha * noise
    return mixed / mixed.sum(axis=1, keepdims=True)


def generate_data(tm, n_samples, seq_len, rng):
    """Vectorized Markov chain data generation."""
    data = np.zeros((n_samples, seq_len), dtype=np.int32)
    data[:, 0] = rng.integers(0, VOCAB_SIZE, size=n_samples)
    cum_tm = np.cumsum(tm, axis=1)
    for t in range(1, seq_len):
        u = rng.random(n_samples).astype(np.float32)
        prev = data[:, t - 1]
        cum_rows = cum_tm[prev]
        data[:, t] = (u[:, None] < cum_rows).argmax(axis=1).astype(np.int32)
    return data


# =============================================================================
# Model
# =============================================================================
def silu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return x * s

def silu_deriv(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return s + x * s * (1 - s)

def softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    def __init__(self, rng):
        sc = lambda fi, fo: np.sqrt(2.0 / (fi + fo))
        self.embed = rng.standard_normal((VOCAB_SIZE, D_MODEL)).astype(np.float32) * 0.02
        self.W_out = rng.standard_normal((D_MODEL, VOCAB_SIZE)).astype(np.float32) * sc(D_MODEL, VOCAB_SIZE)
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)
        self.W_up, self.b_up, self.W_down, self.b_down = [], [], [], []
        for _ in range(N_LAYERS):
            self.W_up.append(rng.standard_normal((D_MODEL, D_FF)).astype(np.float32) * sc(D_MODEL, D_FF))
            self.b_up.append(np.zeros(D_FF, dtype=np.float32))
            self.W_down.append(rng.standard_normal((D_FF, D_MODEL)).astype(np.float32) * sc(D_FF, D_MODEL))
            self.b_down.append(np.zeros(D_MODEL, dtype=np.float32))

    def forward(self, x_ids, lora_deltas=None):
        x = self.embed[x_ids]
        self._hs = [x.copy()]
        h = x
        for l in range(N_LAYERS):
            Wu = self.W_up[l] + (lora_deltas[l][0] if lora_deltas else 0)
            Wd = self.W_down[l] + (lora_deltas[l][1] if lora_deltas else 0)
            pre = h @ Wu + self.b_up[l]
            act = silu(pre)
            h = h + act @ Wd + self.b_down[l]
            self._hs.append(h.copy())
        return h @ self.W_out + self.b_out

    def loss_and_grads(self, x_ids, targets, lora_deltas=None, grad_lora=False, grad_base=True):
        B = x_ids.shape[0]
        logits = self.forward(x_ids, lora_deltas)
        probs = softmax(logits)
        loss = -np.log(np.clip(probs[np.arange(B), targets], 1e-10, 1.0)).mean()
        dlogits = probs.copy()
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B
        grads = {}
        if grad_base:
            grads['W_out'] = self._hs[N_LAYERS].T @ dlogits
            grads['b_out'] = dlogits.sum(0)
        dh = dlogits @ self.W_out.T
        lora_grads = []
        for l in range(N_LAYERS - 1, -1, -1):
            Wu = self.W_up[l] + (lora_deltas[l][0] if lora_deltas else 0)
            Wd = self.W_down[l] + (lora_deltas[l][1] if lora_deltas else 0)
            h_in = self._hs[l]
            pre = h_in @ Wu + self.b_up[l]
            act = silu(pre)
            dact = dh @ Wd.T
            dpre = dact * silu_deriv(pre)
            if grad_base:
                grads[f'W_up_{l}'] = h_in.T @ dpre
                grads[f'b_up_{l}'] = dpre.sum(0)
                grads[f'W_down_{l}'] = act.T @ dh
                grads[f'b_down_{l}'] = dh.sum(0)
            if grad_lora and lora_deltas is not None:
                lora_grads.append((h_in.T @ dpre, act.T @ dh))
            else:
                lora_grads.append(None)
            dh = dh + dpre @ Wu.T
        if grad_base:
            grads['embed'] = np.zeros_like(self.embed)
            np.add.at(grads['embed'], x_ids, dh)
        lora_grads.reverse()
        return loss, grads, lora_grads

    def ntp_loss(self, data, lora_deltas=None):
        B, T = data.shape[0], data.shape[1] - 1
        total = 0.0
        for t in range(T):
            logits = self.forward(data[:, t], lora_deltas)
            probs = softmax(logits)
            total += -np.log(np.clip(probs[np.arange(B), data[:, t + 1]], 1e-10, 1.0)).mean()
        return total / T


class LoRA:
    def __init__(self, rng):
        self.A_up, self.B_up, self.A_down, self.B_down = [], [], [], []
        for _ in range(N_LAYERS):
            self.A_up.append(rng.standard_normal((D_MODEL, LORA_RANK)).astype(np.float32) * 0.01)
            self.B_up.append(rng.standard_normal((LORA_RANK, D_FF)).astype(np.float32) * 0.001)
            self.A_down.append(rng.standard_normal((D_FF, LORA_RANK)).astype(np.float32) * 0.01)
            self.B_down.append(rng.standard_normal((LORA_RANK, D_MODEL)).astype(np.float32) * 0.001)

    def get_deltas(self, scale=LORA_ALPHA / LORA_RANK):
        return [(scale * (self.A_up[l] @ self.B_up[l]),
                 scale * (self.A_down[l] @ self.B_down[l])) for l in range(N_LAYERS)]

    def flatten(self):
        parts = []
        for dW_up, dW_down in self.get_deltas():
            parts.extend([dW_up.ravel(), dW_down.ravel()])
        return np.concatenate(parts)

    def train_step(self, model, x_ids, targets, lr):
        deltas = self.get_deltas()
        loss, _, lg = model.loss_and_grads(x_ids, targets, deltas, grad_lora=True, grad_base=False)
        s = LORA_ALPHA / LORA_RANK
        for l in range(N_LAYERS):
            if lg[l] is None: continue
            gu, gd = lg[l]
            self.A_up[l] -= lr * s * (gu @ self.B_up[l].T)
            self.B_up[l] -= lr * s * (self.A_up[l].T @ gu)
            self.A_down[l] -= lr * s * (gd @ self.B_down[l].T)
            self.B_down[l] -= lr * s * (self.A_down[l].T @ gd)
        return loss


# =============================================================================
# Composition
# =============================================================================
def compose_sum(deltas_list):
    out = []
    for l in range(N_LAYERS):
        u = sum(d[l][0] for d in deltas_list)
        v = sum(d[l][1] for d in deltas_list)
        out.append((u, v))
    return out

def compose_avg(deltas_list):
    N = len(deltas_list)
    out = []
    for l in range(N_LAYERS):
        u = sum(d[l][0] for d in deltas_list) / N
        v = sum(d[l][1] for d in deltas_list) / N
        out.append((u, v))
    return out

def compose_leave_one_out(deltas_list, exclude_idx, mode='avg'):
    remaining = [d for i, d in enumerate(deltas_list) if i != exclude_idx]
    return compose_avg(remaining) if mode == 'avg' else compose_sum(remaining)

def make_noise_delta(deltas_list, rng):
    expert_norms = []
    for deltas in deltas_list:
        norm_sq = sum(float(np.sum(u**2) + np.sum(d**2)) for u, d in deltas)
        expert_norms.append(np.sqrt(norm_sq))
    avg_norm = np.mean(expert_norms)
    N = len(deltas_list)
    noise = []
    noise_sq = 0.0
    for l in range(N_LAYERS):
        u = rng.standard_normal(deltas_list[0][l][0].shape).astype(np.float32)
        d = rng.standard_normal(deltas_list[0][l][1].shape).astype(np.float32)
        noise_sq += float(np.sum(u**2) + np.sum(d**2))
        noise.append((u, d))
    scale = (avg_norm / N) / max(np.sqrt(noise_sq), 1e-12)
    return [(u * scale, d * scale) for u, d in noise]


# =============================================================================
# Main
# =============================================================================
def run_experiment(seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    log(f"\n{'='*70}")
    log(f"  SEED {seed}")
    log(f"{'='*70}")

    max_n = max(N_VALUES)

    # Generate data
    log(f"  Generating data for {max_n} domains...")
    train_data, eval_data = {}, {}
    for i in range(max_n):
        tm = make_transition_matrix(i, max_n, np.random.default_rng(seed + i * 17))
        train_data[i] = generate_data(tm, 300, SEQ_LEN + 1, np.random.default_rng(seed + i * 31))
        eval_data[i] = generate_data(tm, EVAL_SAMPLES, SEQ_LEN + 1, np.random.default_rng(seed + i * 31 + 1))
    log(f"  Data generated.")

    # Train base
    log(f"  Training base model...")
    model = MLP(rng)
    all_train = np.concatenate([train_data[i] for i in range(max_n)], axis=0)
    n_all = len(all_train)
    for step in range(BASE_STEPS):
        idx = rng.integers(0, n_all, size=BATCH_SIZE)
        t = rng.integers(0, SEQ_LEN)
        loss, grads, _ = model.loss_and_grads(all_train[idx, t], all_train[idx, t + 1])
        model.embed -= BASE_LR * grads['embed']
        model.W_out -= BASE_LR * grads['W_out']
        model.b_out -= BASE_LR * grads['b_out']
        for l in range(N_LAYERS):
            model.W_up[l] -= BASE_LR * grads[f'W_up_{l}']
            model.b_up[l] -= BASE_LR * grads[f'b_up_{l}']
            model.W_down[l] -= BASE_LR * grads[f'W_down_{l}']
            model.b_down[l] -= BASE_LR * grads[f'b_down_{l}']

    base_losses = {i: model.ntp_loss(eval_data[i]) for i in range(max_n)}
    overall_base = float(np.mean(list(base_losses.values())))
    log(f"    Base mean NTP loss: {overall_base:.4f}")

    # Train all experts
    log(f"  Training {max_n} LoRA experts ({EXPERT_STEPS} steps each)...")
    experts = {}
    individual_losses = {}
    for i in range(max_n):
        lora = LoRA(np.random.default_rng(seed + i * 137 + 7))
        td = train_data[i]
        for step in range(EXPERT_STEPS):
            idx = rng.integers(0, len(td), size=min(BATCH_SIZE, len(td)))
            t_pos = rng.integers(0, SEQ_LEN)
            lora.train_step(model, td[idx, t_pos], td[idx, t_pos + 1], EXPERT_LR)
        experts[i] = lora
        individual_losses[i] = model.ntp_loss(eval_data[i], lora.get_deltas())
        if (i + 1) % 25 == 0:
            log(f"    Trained {i+1}/{max_n}")

    ind_mean = float(np.mean(list(individual_losses.values())))
    log(f"    Expert mean individual loss: {ind_mean:.4f}")
    log(f"    Individual improvement: {(overall_base - ind_mean)/overall_base*100:.2f}%")

    # Orthogonality
    sample = [experts[i] for i in range(min(20, max_n))]
    vecs = [l.flatten() for l in sample]
    cos_vals = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            ni, nj = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[j])
            if ni > 1e-12 and nj > 1e-12:
                cos_vals.append(abs(float(np.dot(vecs[i], vecs[j]) / (ni * nj))))
    mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
    log(f"    Mean |cos|: {mean_cos:.6f}")

    # N sweep
    results = {
        'seed': seed,
        'base_mean_loss': overall_base,
        'individual_mean_loss': ind_mean,
        'individual_improvement_pct': float((overall_base - ind_mean)/overall_base*100),
        'orthogonality': {'mean_abs_cos': mean_cos},
        'n_sweep': {},
    }

    for N in N_VALUES:
        log(f"\n  --- N = {N} ---")
        active_deltas = [experts[i].get_deltas() for i in range(N)]
        mixed_eval = np.concatenate([eval_data[i] for i in range(N)], axis=0)
        base_loss_mixed = model.ntp_loss(mixed_eval)

        # SUM and AVG composition
        sum_d = compose_sum(active_deltas)
        sum_loss = model.ntp_loss(mixed_eval, sum_d)
        avg_d = compose_avg(active_deltas)
        avg_loss = model.ntp_loss(mixed_eval, avg_d)

        # Marginal contributions
        n_marginal = min(MAX_MARGINAL_SAMPLES, N)
        marginal_sum, marginal_avg = [], []
        for idx in range(n_marginal):
            loo_s = compose_leave_one_out(active_deltas, idx, 'sum')
            marginal_sum.append(float(model.ntp_loss(mixed_eval, loo_s) - sum_loss))
            loo_a = compose_leave_one_out(active_deltas, idx, 'avg')
            marginal_avg.append(float(model.ntp_loss(mixed_eval, loo_a) - avg_loss))

        # Noise floor (20 samples for statistical reliability)
        # SYMMETRY FIX: Both signal and noise use the same operation.
        #
        # Signal (marginal_avg[i]):
        #   loss(compose_avg(N-1 real, excl i)) - loss(compose_avg(N real))
        #   = LOO removal with reweighting 1/N -> 1/(N-1)
        #
        # Noise (symmetric): replace expert with random noise, then do LOO
        # removal of the noise expert. This gives:
        #   loss(compose_avg(N-1 real)) - loss(compose_avg(N-1 real + 1 noise))
        #   = LOO removal of noise expert, same 1/N -> 1/(N-1) reweighting
        #
        # Both operations measure "what changes when we remove one slot from
        # N-expert averaged composition?" Signal removes a real expert;
        # noise removes a random perturbation. Same reweighting in both.
        NOISE_SAMPLES = 20
        noise_losses = []
        # Pre-compute LOO of last real expert (the "without" baseline)
        loo_last = compose_leave_one_out(active_deltas, N - 1, 'avg')
        loo_last_loss = model.ntp_loss(mixed_eval, loo_last)
        for _ in range(NOISE_SAMPLES):
            nd = make_noise_delta(active_deltas, rng)
            # Compose N-1 real + 1 noise, averaged at 1/N each
            noisy_set = active_deltas[:-1] + [nd]
            noisy_loss = model.ntp_loss(mixed_eval, compose_avg(noisy_set))
            # LOO of noise expert = compose_avg(N-1 real) = loo_last_loss
            # So noise marginal = loo_last_loss - noisy_loss
            # (same structure as signal: loss_without - loss_with)
            noise_losses.append(float(loo_last_loss - noisy_loss))
        noise_floor = float(np.std(noise_losses)) if noise_losses else 0.0

        # Per-domain eval
        per_dom = []
        for i in range(min(n_marginal, N)):
            bl = base_losses[i]
            cl = model.ntp_loss(eval_data[i], avg_d)
            per_dom.append(float((cl - bl) / bl * 100))

        sum_gap = (sum_loss - base_loss_mixed) / base_loss_mixed * 100
        avg_gap = (avg_loss - base_loss_mixed) / base_loss_mixed * 100
        snr = abs(np.mean(marginal_avg)) / max(noise_floor, 1e-12)

        log(f"    Base: {base_loss_mixed:.4f}  SUM: {sum_loss:.4f} ({sum_gap:+.2f}%)  "
            f"AVG: {avg_loss:.4f} ({avg_gap:+.2f}%)")
        log(f"    Marginal(AVG): mean={np.mean(marginal_avg):.6f} std={np.std(marginal_avg):.6f}")
        log(f"    Noise floor: {noise_floor:.6f}  SNR: {snr:.2f}")
        log(f"    Per-domain gaps: mean={np.mean(per_dom):+.2f}% max={np.max(per_dom):+.2f}%")

        results['n_sweep'][N] = {
            'base_loss_mixed': float(base_loss_mixed),
            'sum_loss': float(sum_loss), 'avg_loss': float(avg_loss),
            'sum_gap_pct': float(sum_gap), 'avg_gap_pct': float(avg_gap),
            'marginal_sum': marginal_sum, 'marginal_avg': marginal_avg,
            'marginal_sum_mean': float(np.mean(marginal_sum)),
            'marginal_avg_mean': float(np.mean(marginal_avg)),
            'marginal_sum_std': float(np.std(marginal_sum)),
            'marginal_avg_std': float(np.std(marginal_avg)),
            'noise_floor': noise_floor,
            'signal_noise_ratio': float(snr),
            'per_domain_gaps': per_dom,
            'per_domain_gap_mean': float(np.mean(per_dom)),
            'per_domain_gap_max': float(np.max(per_dom)),
        }

    results['time_s'] = time.perf_counter() - t0
    log(f"\n  Seed {seed} time: {results['time_s']:.1f}s")
    return results


def run_all():
    all_results = {}
    t0 = time.perf_counter()
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed)
    total_time = time.perf_counter() - t0

    # Aggregate
    log(f"\n{'='*70}")
    log(f"  AGGREGATED ({len(SEEDS)} seeds)")
    log(f"{'='*70}")

    agg = {}
    for N in N_VALUES:
        vals = {k: [all_results[s]['n_sweep'][N][k] for s in SEEDS]
                for k in ['sum_gap_pct', 'avg_gap_pct', 'marginal_avg_mean',
                           'noise_floor', 'signal_noise_ratio', 'per_domain_gap_mean']}
        agg[N] = {f'{k}_mean': float(np.mean(v)) for k, v in vals.items()}
        agg[N].update({f'{k}_std': float(np.std(v)) for k, v in vals.items()})

        log(f"\n  N={N:3d}:")
        log(f"    SUM gap: {agg[N]['sum_gap_pct_mean']:+.3f}% +/- {agg[N]['sum_gap_pct_std']:.3f}%")
        log(f"    AVG gap: {agg[N]['avg_gap_pct_mean']:+.3f}% +/- {agg[N]['avg_gap_pct_std']:.3f}%")
        log(f"    Marginal(AVG): {agg[N]['marginal_avg_mean_mean']:.6f}")
        log(f"    Noise: {agg[N]['noise_floor_mean']:.6f}  SNR: {agg[N]['signal_noise_ratio_mean']:.2f}")

    # Kill criteria
    log(f"\n{'='*70}")
    log(f"  KILL CRITERIA")
    log(f"{'='*70}")

    k1_kill = False
    log("\n  K1: Per-expert signal below noise at N<50?")
    for N in N_VALUES:
        if N >= 50: continue
        snr = agg[N]['signal_noise_ratio_mean']
        status = "BELOW NOISE" if snr < 1.0 else "DETECTABLE"
        log(f"    N={N:3d}: SNR={snr:.2f} -> {status}")
        if snr < 1.0:
            k1_kill = True

    k2_kill = False
    if 100 in agg and 10 in agg:
        deg = agg[100]['avg_gap_pct_mean'] - agg[10]['avg_gap_pct_mean']
        k2_kill = abs(deg) > 20.0
        log(f"\n  K2: AVG degradation N=100 vs N=10 > 20%?")
        log(f"    N=10:  {agg[10]['avg_gap_pct_mean']:+.3f}%")
        log(f"    N=100: {agg[100]['avg_gap_pct_mean']:+.3f}%")
        log(f"    Degradation: {deg:+.3f}%  -> {'KILL' if k2_kill else 'PASS'}")

    overall = k1_kill or k2_kill
    log(f"\n  K1: {'KILL' if k1_kill else 'PASS'}  K2: {'KILL' if k2_kill else 'PASS'}  "
        f"OVERALL: {'KILL' if overall else 'PASS'}")

    # Scaling analysis
    log(f"\n{'='*70}")
    log(f"  SCALING ANALYSIS")
    log(f"{'='*70}")

    ns = np.array(N_VALUES, dtype=float)
    marginals = np.array([abs(agg[N]['marginal_avg_mean_mean']) for N in N_VALUES])
    valid = marginals > 1e-12
    power = None
    n_critical = None
    if valid.sum() >= 2:
        coeffs = np.polyfit(np.log(ns[valid]), np.log(marginals[valid]), 1)
        power = float(coeffs[0])
        const = float(np.exp(coeffs[1]))
        log(f"  Marginal ~ {const:.6f} * N^({power:.3f})")
        log(f"  Expected -1.0 for pure 1/N dilution")
        mean_noise = np.mean([agg[N]['noise_floor_mean'] for N in N_VALUES])
        if mean_noise > 1e-12 and power < 0:
            n_critical = float((mean_noise / const) ** (1.0 / power))
            log(f"  N_critical (marginal = noise): {n_critical:.0f}")

    # SUM scaling
    sum_gaps = np.array([agg[N]['sum_gap_pct_mean'] for N in N_VALUES])
    log(f"\n  SUM gap vs N: {list(zip(N_VALUES, [f'{g:+.2f}%' for g in sum_gaps]))}")
    avg_gaps = np.array([agg[N]['avg_gap_pct_mean'] for N in N_VALUES])
    log(f"  AVG gap vs N: {list(zip(N_VALUES, [f'{g:+.2f}%' for g in avg_gaps]))}")

    # Save
    output = {
        'experiment': 'composition_weight_sensitivity',
        'config': {
            'd_model': D_MODEL, 'd_ff': D_FF, 'n_layers': N_LAYERS,
            'vocab_size': VOCAB_SIZE, 'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA, 'n_values': N_VALUES, 'seeds': SEEDS,
            'base_steps': BASE_STEPS, 'expert_steps': EXPERT_STEPS,
        },
        'aggregate': {str(k): v for k, v in agg.items()},
        'kill_criteria': {
            'k1_signal_below_noise_at_n_lt_50': k1_kill,
            'k2_degradation_gt_20pct': k2_kill,
            'overall_kill': overall,
        },
        'scaling': {
            'power_law_exponent': power,
            'n_critical': n_critical,
        },
        'per_seed': {str(s): _ser(all_results[s]) for s in SEEDS},
        'total_time_s': total_time,
    }

    out = RESULTS_DIR / 'results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\n  Saved to {out}")
    log(f"  Total time: {total_time:.1f}s")
    return output


def _ser(d):
    if isinstance(d, dict): return {str(k): _ser(v) for k, v in d.items()}
    if isinstance(d, list): return [_ser(v) for v in d]
    if isinstance(d, (np.floating, np.integer)): return float(d)
    if isinstance(d, np.ndarray): return d.tolist()
    return d


if __name__ == '__main__':
    run_all()
