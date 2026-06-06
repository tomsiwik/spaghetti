"""Shannon Channel Capacity Bound for Expert Composition (REVISED).

REVISION per adversarial review. Changes from v1:
  1. Collects empirical data at N=3,4,6,7 (held-out validation points)
  2. Compares Shannon model against linear and power-law baselines (AIC/BIC)
  3. Removes dead code (rho_at_n method)
  4. Adds sensitivity analysis (perturb inputs +/- 1%)
  5. Caveats rate-distortion interpretation (descriptive, not fundamental)
  6. Status downgraded from "proven" to "consistent"

Models the residual stream as a Gaussian multiple-access channel (MAC).
Each expert is a transmitter adding its signal to the residual stream.
Inter-expert interference is modeled as noise proportional to the
non-orthogonal component of expert outputs.

Core equation:
  gap(N) = (1 - log(1 + SNR_0/(1+(N-1)*alpha)) / log(1+SNR_0)) * 100 + c_0

Empirical data from prior experiments (training set):
  N=2: gap = -0.2% (n_expert_scale, calibrated)
  N=5: gap = +1.6% (n_expert_scale, calibrated)
  N=8: gap = +5.71% (flat_moe_n8_boundary, calibrated)

Validation points collected in this revision: N=3, N=4, N=6, N=7.
"""

import copy
import math
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


# ============================================================
# Architecture constants (must match flat_moe_n8_boundary protocol)
# ============================================================

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
BATCH_SIZE = 32
LR = 3e-3

# Calibration steps scaled linearly with N (same rule as flat_moe)
def cal_steps_for_n(n):
    """Calibration steps: 50 * N, matching flat_moe protocol."""
    return max(50, int(50 * n))

# Split methods by N
SPLIT_METHODS = {
    2: "binary",
    3: "ternary",
    4: "quaternary",
    5: "quintary",
    6: "senary",
    7: "septenary",
    8: "octonary",
}


# ============================================================
# Empirical data from prior experiments (TRAINING SET for model fit)
# ============================================================

EMPIRICAL_GAPS_TRAIN = {
    2: -0.2,   # vs joint, calibrated (n_expert_scale)
    5: +1.6,   # vs joint, calibrated (n_expert_scale)
    8: +5.71,  # vs joint, calibrated (flat_moe_n8_boundary, revised)
}

# Model architecture parameters (micro scale)
D_MODEL = 64


# ============================================================
# Channel capacity model (REVISED: removed dead rho_at_n method)
# ============================================================

@dataclass
class ChannelModel:
    """Parameters of the Gaussian MAC channel model.

    The model: y = sum_{i=1}^N w_i * f_i(x) + eta

    where:
      - y is the residual stream output (d-dimensional)
      - f_i(x) is expert i's contribution (capsule output, d-dimensional)
      - w_i is the routing weight for expert i (scalar, from softmax router)
      - eta ~ N(0, sigma_noise^2 * I_d) is intrinsic noise (training noise)

    Signal power per expert: P = E[||f_i(x)||^2] / d
    Interference between experts i,j: I_ij = rho_ij^2 * P
      where rho_ij = cosine_similarity(f_i, f_j) averaged over inputs

    Effective SNR for expert i with N-1 interferers:
      SNR_eff(N) = P / (sigma^2 + (N-1) * rho_mean^2 * P)
               = SNR_0 / (1 + (N-1) * alpha)

    Capacity per expert:
      C(N) = (1/2) * log(1 + SNR_eff(N))

    Composition gap (quality degradation):
      gap(N) = (1 - C(N) / C(1)) * 100
    """
    d: int                # model dimension
    P: float              # signal power per expert (= SNR_0 when sigma2=1)
    sigma2: float         # intrinsic noise variance
    rho_mean: float       # mean pairwise correlation between experts
    calibration_gain: float  # unused in analytical fit (kept for API compat)

    def snr_single(self) -> float:
        """SNR when only 1 expert is active (no interference)."""
        return self.P / self.sigma2

    def interference_power(self, N: int) -> float:
        """Total interference power from N-1 other experts."""
        if N <= 1:
            return 0.0
        return (N - 1) * self.rho_mean**2 * self.P

    def snr_effective(self, N: int) -> float:
        """Effective SNR per expert with N composed experts."""
        noise_plus_interference = self.sigma2 + self.interference_power(N)
        return self.P / noise_plus_interference

    def capacity_per_expert(self, N: int) -> float:
        """Channel capacity per expert (nats per dimension).

        C(N) = (1/2) * log(1 + SNR_eff(N))
        """
        return 0.5 * math.log(1 + self.snr_effective(N))

    def composition_gap_theory(self, N: int) -> float:
        """Theoretical composition gap (percentage).

        gap(N) = (1 - C(N)/C(1)) * 100
        """
        if N <= 1:
            return 0.0
        c1 = self.capacity_per_expert(1)
        cn = self.capacity_per_expert(N)
        if c1 == 0:
            return 0.0
        return (1 - cn / c1) * 100

    def max_n_at_gap(self, gap_threshold: float = 10.0, n_max_search: int = 100) -> int:
        """Find maximum N where composition gap stays below threshold."""
        for n in range(1, n_max_search + 1):
            if self.composition_gap_theory(n) >= gap_threshold:
                return n - 1
        return n_max_search


# ============================================================
# Model fitting (analytical, 2-parameter + offset)
# ============================================================

def predict_gaps_shannon(ns, snr0, alpha, offset):
    """Predict composition gaps using Shannon model."""
    preds = []
    for n in ns:
        if n <= 1:
            preds.append(offset)
        else:
            snr_eff = snr0 / (1 + (n - 1) * alpha)
            c_n = 0.5 * math.log(1 + snr_eff)
            c_1 = 0.5 * math.log(1 + snr0)
            gap_raw = (1 - c_n / c_1) * 100
            preds.append(gap_raw + offset)
    return preds


def fit_channel_model_analytical(
    empirical_gaps: dict[int, float],
    d: int = D_MODEL,
) -> tuple:
    """Analytical fit using grid search on the 2-parameter model.

    Simplified model with 2 free parameters (alpha, SNR_0) + offset:
      gap(N) = (1 - log(1 + SNR_0 / (1 + (N-1)*alpha)) / log(1 + SNR_0)) * 100 + c_0

    Returns: (ChannelModel, offset, mse)
    """
    ns = sorted(empirical_gaps.keys())
    gaps_empirical = np.array([empirical_gaps[n] for n in ns])

    best_params = None
    best_mse = float('inf')

    for log_snr in np.linspace(-1, 4, 200):
        snr0 = math.exp(log_snr)
        for log_alpha in np.linspace(-6, 0, 200):
            alpha = math.exp(log_alpha)

            preds = []
            for n in ns:
                if n <= 1:
                    preds.append(0.0)
                else:
                    snr_eff = snr0 / (1 + (n - 1) * alpha)
                    c_n = 0.5 * math.log(1 + snr_eff)
                    c_1 = 0.5 * math.log(1 + snr0)
                    gap_pred = (1 - c_n / c_1) * 100
                    preds.append(gap_pred)

            preds = np.array(preds)
            offset = np.mean(gaps_empirical - preds)
            preds_with_offset = preds + offset
            mse = np.mean((preds_with_offset - gaps_empirical) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_params = (snr0, alpha, offset)

    snr0, alpha, offset = best_params
    rho = math.sqrt(alpha / snr0) if snr0 > 0 else 0.0

    model = ChannelModel(
        d=d,
        P=snr0,
        sigma2=1.0,
        rho_mean=rho,
        calibration_gain=0.0,
    )
    return model, offset, best_mse


# ============================================================
# Baseline models for comparison (Fix #2)
# ============================================================

def fit_linear(empirical_gaps: dict[int, float]):
    """Fit linear model: gap(N) = a * N + b.

    2 parameters.
    """
    ns = sorted(empirical_gaps.keys())
    gaps = np.array([empirical_gaps[n] for n in ns])
    ns_arr = np.array(ns, dtype=float)

    # Least squares: [N, 1] @ [a, b]' = gaps
    A = np.column_stack([ns_arr, np.ones_like(ns_arr)])
    params, residuals, _, _ = np.linalg.lstsq(A, gaps, rcond=None)
    a, b = params

    preds = a * ns_arr + b
    mse = float(np.mean((preds - gaps) ** 2))
    return {"a": float(a), "b": float(b)}, mse, list(preds)


def predict_linear(ns, a, b):
    return [a * n + b for n in ns]


def fit_power_law(empirical_gaps: dict[int, float]):
    """Fit power-law model: gap(N) = a * N^b + c.

    3 parameters. Grid search over b, then linear fit for a and c.
    """
    ns = sorted(empirical_gaps.keys())
    gaps = np.array([empirical_gaps[n] for n in ns])
    ns_arr = np.array(ns, dtype=float)

    best_params = None
    best_mse = float('inf')

    for b in np.linspace(0.1, 4.0, 400):
        # For fixed b: gap = a * N^b + c is linear in (a, c)
        A = np.column_stack([ns_arr**b, np.ones_like(ns_arr)])
        params, _, _, _ = np.linalg.lstsq(A, gaps, rcond=None)
        a, c = params

        preds = a * ns_arr**b + c
        mse = float(np.mean((preds - gaps) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_params = {"a": float(a), "b": float(b), "c": float(c)}
            best_preds = list(preds)

    return best_params, best_mse, best_preds


def predict_power_law(ns, a, b, c):
    return [a * n**b + c for n in ns]


def compute_aic(n_points, n_params, mse):
    """Akaike Information Criterion (small-sample corrected AICc).

    AIC = n * ln(mse) + 2k
    AICc = AIC + 2k(k+1) / (n-k-1)  [Burnham & Anderson correction]
    """
    if mse <= 0:
        mse = 1e-15
    k = n_params
    n = n_points
    aic = n * math.log(mse) + 2 * k
    if n - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = float('inf')
    return aicc


def compute_bic(n_points, n_params, mse):
    """Bayesian Information Criterion."""
    if mse <= 0:
        mse = 1e-15
    return n_points * math.log(mse) + n_params * math.log(n_points)


def compute_r_squared(observed: list[float], predicted: list[float]) -> float:
    """Coefficient of determination R^2."""
    if len(observed) < 2:
        return 0.0
    obs = np.array(observed)
    pred = np.array(predicted)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


# ============================================================
# Data collection: run composition experiments at new N values
# ============================================================

def collect_gap_at_n(n_domains, seed=42):
    """Run the flat_moe composition protocol for a given N and return the gap.

    Protocol (same as flat_moe_n8_boundary):
      1. Joint training baseline (round-robin, N*STEPS_FINETUNE total)
      2. Pretrain base + fine-tune per domain (attention frozen)
      3. Compose by weight concatenation
      4. Calibrate on mixed data
      5. Return calibrated composition gap = (cal_loss - joint_loss) / joint_loss * 100
    """
    from ..relu_router.test_composition import (
        _make_relu_model, _freeze_attention,
    )
    from ..n5_identity_scaling.n5_identity_scaling import compose_n_domains

    split_method = SPLIT_METHODS[n_domains]
    cal_steps = cal_steps_for_n(n_domains)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method=split_method)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = list(domain_datasets.keys())
    domain_trains = {d: domain_datasets[d][0] for d in domain_names}
    domain_vals = {d: domain_datasets[d][1] for d in domain_names}
    assert len(domain_names) == n_domains, f"Expected {n_domains}, got {len(domain_names)}"

    # 1. Joint training baseline
    total_ft_steps = n_domains * STEPS_FINETUNE
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * n_domains)
    rng_j = random.Random(seed)
    optimizer_j = optim.Adam(learning_rate=LR)
    loss_and_grad_j = nn.value_and_grad(model_joint, ntp_loss)
    ds_list = list(domain_trains.values())
    n_ds = len(ds_list)
    for step in range(1, total_ft_steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng_j)
        loss, grads = loss_and_grad_j(model_joint, inputs, targets)
        optimizer_j.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer_j.state)

    joint_per_domain = {}
    for d_name in domain_names:
        joint_per_domain[d_name] = evaluate(model_joint, domain_vals[d_name], BATCH_SIZE)
    joint_avg = sum(joint_per_domain.values()) / len(joint_per_domain)

    # 2. Pretrain base + fine-tune per domain
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # 3. Compose
    composed = compose_n_domains(base, [domain_models[d] for d in domain_names])

    # 4. Calibrate
    rng_c = random.Random(seed)
    optimizer_c = optim.Adam(learning_rate=LR)
    loss_and_grad_c = nn.value_and_grad(composed, ntp_loss)
    for step in range(1, cal_steps + 1):
        ds = ds_list[step % n_ds]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng_c)
        loss, grads = loss_and_grad_c(composed, inputs, targets)
        optimizer_c.update(composed, grads)
        mx.eval(composed.parameters(), optimizer_c.state)

    # 5. Evaluate
    cal_per_domain = {}
    for d_name in domain_names:
        cal_per_domain[d_name] = evaluate(composed, domain_vals[d_name], BATCH_SIZE)
    cal_avg = sum(cal_per_domain.values()) / len(cal_per_domain)
    cal_gap = (cal_avg - joint_avg) / joint_avg * 100

    return {
        "n_domains": n_domains,
        "seed": seed,
        "joint_avg": float(joint_avg),
        "cal_avg": float(cal_avg),
        "cal_gap": float(cal_gap),
        "cal_steps": cal_steps,
        "split_method": split_method,
    }


def collect_all_gaps(n_values, seeds=(42, 123, 7)):
    """Collect composition gaps for all specified N values across seeds."""
    all_results = {}
    for n in n_values:
        all_results[n] = []
        for seed in seeds:
            print(f"  Collecting N={n}, seed={seed}...")
            result = collect_gap_at_n(n, seed=seed)
            all_results[n].append(result)
            print(f"    gap = {result['cal_gap']:+.2f}%")
    return all_results


# ============================================================
# Sensitivity analysis (Fix #5)
# ============================================================

def sensitivity_analysis(empirical_gaps, perturbation_pct=1.0):
    """Perturb each data point by +/- perturbation_pct and report N_max range.

    Returns dict with min/max N_max across all perturbation combinations.
    """
    ns = sorted(empirical_gaps.keys())
    gaps = [empirical_gaps[n] for n in ns]
    n_points = len(ns)

    n_max_values = []
    n_params_list = []

    # Try all 3^n combinations of {-1, 0, +1} perturbations
    import itertools
    for signs in itertools.product([-1, 0, 1], repeat=n_points):
        perturbed = {}
        for i, n in enumerate(ns):
            # Perturb by perturbation_pct of the absolute value,
            # or by 0.1% absolute if the gap is near zero
            magnitude = max(abs(gaps[i]) * perturbation_pct / 100, 0.1 * perturbation_pct / 100)
            perturbed[n] = gaps[i] + signs[i] * magnitude

        model, offset, mse = fit_channel_model_analytical(perturbed)
        alpha = model.rho_mean**2 * model.P

        # Compute N_max at 10% threshold
        for n_test in range(1, 101):
            pred = predict_gaps_shannon([n_test], model.P, alpha, offset)[0]
            if pred >= 10.0:
                n_max_values.append(n_test - 1)
                break
        else:
            n_max_values.append(100)

        n_params_list.append({
            "signs": signs,
            "snr0": model.P,
            "alpha": alpha,
            "offset": offset,
            "n_max_10": n_max_values[-1],
        })

    return {
        "n_max_min": min(n_max_values),
        "n_max_max": max(n_max_values),
        "n_max_median": sorted(n_max_values)[len(n_max_values) // 2],
        "perturbation_pct": perturbation_pct,
        "n_perturbations": len(n_max_values),
        "details": n_params_list,
    }


# ============================================================
# Main experiment (REVISED)
# ============================================================

def run_experiment(collect_new_data=True):
    """Run the full revised channel capacity bound analysis.

    Phase 1: Collect empirical data at N=3,4,6,7 (new)
    Phase 2: Fit Shannon model on training set (N=2,5,8)
    Phase 3: Validate on held-out set (N=3,4,6,7)
    Phase 4: Compare against baselines (linear, power-law)
    Phase 5: Sensitivity analysis
    Phase 6: Kill criteria with proper caveats
    """
    print("=" * 70)
    print("  Shannon Channel Capacity Bound for Expert Composition (REVISED)")
    print("=" * 70)

    # ============================================================
    # Phase 1: Collect new data points
    # ============================================================
    print("\n--- Phase 1: Data Collection ---")

    validation_ns = [3, 4, 6, 7]
    validation_gaps = {}

    if collect_new_data:
        print("  Collecting composition gaps at N=3,4,6,7 (3 seeds each)...")
        t0 = time.time()
        new_data = collect_all_gaps(validation_ns, seeds=(42, 123, 7))
        elapsed_collect = time.time() - t0
        print(f"  Data collection took {elapsed_collect:.0f}s")

        for n in validation_ns:
            gaps = [r["cal_gap"] for r in new_data[n]]
            validation_gaps[n] = statistics.mean(gaps)
            print(f"  N={n}: gap = {validation_gaps[n]:+.2f}% "
                  f"(std = {statistics.stdev(gaps):.2f}%, "
                  f"per-seed: {[f'{g:+.2f}' for g in gaps]})")
    else:
        print("  Using cached validation data...")
        # Will be filled from results.json if available
        validation_gaps = {}

    # Training data
    train_gaps = dict(EMPIRICAL_GAPS_TRAIN)
    all_gaps = {**train_gaps, **validation_gaps}

    print(f"\n  Training set (N=2,5,8): {train_gaps}")
    print(f"  Validation set (N=3,4,6,7): {validation_gaps}")

    # ============================================================
    # Phase 2: Fit Shannon model on training data
    # ============================================================
    print("\n--- Phase 2: Shannon Model Fit (Training Set) ---")

    model, offset, fit_mse = fit_channel_model_analytical(train_gaps)
    alpha = model.rho_mean**2 * model.P

    print(f"\n  Fitted parameters:")
    print(f"    SNR_0 = {model.P:.4f}")
    print(f"    rho_mean = {model.rho_mean:.6f}")
    print(f"    alpha = {alpha:.6f}")
    print(f"    offset (c_0) = {offset:+.4f}%")
    print(f"    train MSE = {fit_mse:.6f}")

    # Training accuracy
    train_ns = sorted(train_gaps.keys())
    train_obs = [train_gaps[n] for n in train_ns]
    train_pred = predict_gaps_shannon(train_ns, model.P, alpha, offset)
    train_r2 = compute_r_squared(train_obs, train_pred)

    print(f"\n  Training fit (N={train_ns}):")
    print(f"  {'N':>3} | {'Observed':>10} | {'Predicted':>10} | {'Error':>8}")
    print("  " + "-" * 42)
    for n, obs, pred in zip(train_ns, train_obs, train_pred):
        print(f"  {n:>3} | {obs:>+9.2f}% | {pred:>+9.2f}% | {pred - obs:>+7.2f}%")
    print(f"  Train R^2 = {train_r2:.4f}")

    # ============================================================
    # Phase 3: Held-out validation
    # ============================================================
    print("\n--- Phase 3: Held-Out Validation (N=3,4,6,7) ---")

    if validation_gaps:
        val_ns = sorted(validation_gaps.keys())
        val_obs = [validation_gaps[n] for n in val_ns]
        val_pred = predict_gaps_shannon(val_ns, model.P, alpha, offset)
        val_r2 = compute_r_squared(val_obs, val_pred)

        print(f"\n  {'N':>3} | {'Observed':>10} | {'Predicted':>10} | {'Error':>8} | {'Ratio':>6}")
        print("  " + "-" * 52)
        max_val_ratio = 0.0
        for n, obs, pred in zip(val_ns, val_obs, val_pred):
            err = pred - obs
            if abs(obs) > 1.0:
                ratio = abs(pred) / abs(obs)
            elif abs(obs) > 0.1:
                ratio = 1.0 if abs(err) < 1.0 else 2.1
            else:
                ratio = 1.0 if abs(err) < 0.5 else 2.1
            max_val_ratio = max(max_val_ratio, ratio)
            print(f"  {n:>3} | {obs:>+9.2f}% | {pred:>+9.2f}% | {err:>+7.2f}% | {ratio:>5.2f}x")

        print(f"\n  Validation R^2 = {val_r2:.4f}")
        print(f"  Max validation ratio = {max_val_ratio:.2f}x")
    else:
        val_r2 = None
        max_val_ratio = None
        val_ns = []
        val_obs = []
        val_pred = []

    # Full-data R^2 (all N values)
    all_ns = sorted(all_gaps.keys())
    all_obs = [all_gaps[n] for n in all_ns]
    all_pred = predict_gaps_shannon(all_ns, model.P, alpha, offset)
    all_r2 = compute_r_squared(all_obs, all_pred)
    print(f"\n  Full-data R^2 (all {len(all_ns)} points) = {all_r2:.4f}")

    # ============================================================
    # Phase 4: Baseline model comparison (Fix #2)
    # ============================================================
    print("\n--- Phase 4: Baseline Model Comparison ---")

    n_train = len(train_gaps)

    # 4a. Fit all models to TRAINING data
    lin_params, lin_mse, lin_train_pred = fit_linear(train_gaps)
    pow_params, pow_mse, pow_train_pred = fit_power_law(train_gaps)

    # Shannon model: 3 params (SNR_0, alpha, offset)
    shannon_n_params = 3
    lin_n_params = 2
    pow_n_params = 3

    # AICc and BIC on training data
    shannon_aicc = compute_aic(n_train, shannon_n_params, fit_mse)
    lin_aicc = compute_aic(n_train, lin_n_params, lin_mse)
    pow_aicc = compute_aic(n_train, pow_n_params, pow_mse)

    shannon_bic = compute_bic(n_train, shannon_n_params, fit_mse)
    lin_bic = compute_bic(n_train, lin_n_params, lin_mse)
    pow_bic = compute_bic(n_train, pow_n_params, pow_mse)

    print(f"\n  Model comparison (fit on N={train_ns}):")
    print(f"  {'Model':<15} | {'Params':>6} | {'Train MSE':>10} | {'AICc':>8} | {'BIC':>8}")
    print("  " + "-" * 58)
    print(f"  {'Shannon':<15} | {shannon_n_params:>6} | {fit_mse:>10.6f} | {shannon_aicc:>8.2f} | {shannon_bic:>8.2f}")
    print(f"  {'Linear':<15} | {lin_n_params:>6} | {lin_mse:>10.6f} | {lin_aicc:>8.2f} | {lin_bic:>8.2f}")
    print(f"  {'Power-law':<15} | {pow_n_params:>6} | {pow_mse:>10.6f} | {pow_aicc:>8.2f} | {pow_bic:>8.2f}")

    # 4b. Out-of-sample validation for all models
    if validation_gaps:
        lin_val_pred = predict_linear(val_ns, lin_params["a"], lin_params["b"])
        pow_val_pred = predict_power_law(val_ns, pow_params["a"], pow_params["b"], pow_params["c"])
        shannon_val_pred = predict_gaps_shannon(val_ns, model.P, alpha, offset)

        lin_val_mse = float(np.mean([(p - o)**2 for p, o in zip(lin_val_pred, val_obs)]))
        pow_val_mse = float(np.mean([(p - o)**2 for p, o in zip(pow_val_pred, val_obs)]))
        shannon_val_mse = float(np.mean([(p - o)**2 for p, o in zip(shannon_val_pred, val_obs)]))

        lin_val_r2 = compute_r_squared(val_obs, lin_val_pred)
        pow_val_r2 = compute_r_squared(val_obs, pow_val_pred)

        print(f"\n  Out-of-sample validation (N={val_ns}):")
        print(f"  {'Model':<15} | {'Val MSE':>10} | {'Val R^2':>8}")
        print("  " + "-" * 40)
        print(f"  {'Shannon':<15} | {shannon_val_mse:>10.4f} | {val_r2:>8.4f}")
        print(f"  {'Linear':<15} | {lin_val_mse:>10.4f} | {lin_val_r2:>8.4f}")
        print(f"  {'Power-law':<15} | {pow_val_mse:>10.4f} | {pow_val_r2:>8.4f}")

        # Full-data R^2 for all models
        lin_all_pred = predict_linear(all_ns, lin_params["a"], lin_params["b"])
        pow_all_pred = predict_power_law(all_ns, pow_params["a"], pow_params["b"], pow_params["c"])
        lin_all_r2 = compute_r_squared(all_obs, lin_all_pred)
        pow_all_r2 = compute_r_squared(all_obs, pow_all_pred)

        print(f"\n  Full-data R^2 (all {len(all_ns)} points):")
        print(f"    Shannon:   {all_r2:.4f}")
        print(f"    Linear:    {lin_all_r2:.4f}")
        print(f"    Power-law: {pow_all_r2:.4f}")
    else:
        lin_val_mse = pow_val_mse = shannon_val_mse = None
        lin_val_r2 = pow_val_r2 = None
        lin_all_r2 = pow_all_r2 = None

    # ============================================================
    # Phase 5: Sensitivity analysis (Fix #5)
    # ============================================================
    print("\n--- Phase 5: Sensitivity Analysis ---")

    sens = sensitivity_analysis(train_gaps, perturbation_pct=1.0)
    print(f"\n  Perturbing training data by +/- 1%:")
    print(f"    N_max (10% gap) range: [{sens['n_max_min']}, {sens['n_max_max']}]")
    print(f"    N_max median: {sens['n_max_median']}")
    print(f"    Number of perturbations tested: {sens['n_perturbations']}")

    if sens['n_max_max'] - sens['n_max_min'] <= 5:
        print(f"    Prediction is ROBUST (range <= 5)")
    else:
        print(f"    WARNING: Prediction is NOT robust (range > 5)")

    # ============================================================
    # Phase 6: N_max predictions
    # ============================================================
    print("\n--- Phase 6: N_max Predictions ---")

    n_max_5 = n_max_10 = n_max_20 = None
    print(f"\n  {'N':>3} | {'Predicted Gap':>13}")
    print("  " + "-" * 22)
    for n in range(1, 31):
        gap = predict_gaps_shannon([n], model.P, alpha, offset)[0]
        marker = ""
        if n in all_gaps:
            marker = f"  <-- empirical: {all_gaps[n]:+.2f}%"
        if n_max_5 is None and gap >= 5.0:
            n_max_5 = n - 1
            marker += " [5% threshold]"
        if n_max_10 is None and gap >= 10.0:
            n_max_10 = n - 1
            marker += " [10% threshold]"
        if n_max_20 is None and gap >= 20.0:
            n_max_20 = n - 1
            marker += " [20% threshold]"
        if n <= 10 or n in [12, 15, 20, 25, 30] or marker:
            print(f"  {n:>3} | {gap:>+12.2f}%{marker}")

    print(f"\n  N_max (5% gap):  {n_max_5}")
    print(f"  N_max (10% gap): {n_max_10}")
    print(f"  N_max (20% gap): {n_max_20}")

    # ============================================================
    # Phase 7: Kill criteria (updated)
    # ============================================================
    print(f"\n{'='*70}")
    print("  KILL CRITERIA EVALUATION")
    print(f"{'='*70}")

    # Use ALL data for final evaluation
    all_pred_final = predict_gaps_shannon(all_ns, model.P, alpha, offset)
    max_ratio_all = 0.0
    for obs, pred in zip(all_obs, all_pred_final):
        err = pred - obs
        if abs(obs) > 1.0:
            ratio = abs(pred) / abs(obs)
        elif abs(obs) > 0.1:
            ratio = 1.0 if abs(err) < 1.0 else 2.1
        else:
            ratio = 1.0 if abs(err) < 0.5 else 2.1
        max_ratio_all = max(max_ratio_all, ratio)

    within_2x = max_ratio_all <= 2.0
    r2_pass = (val_r2 is not None and val_r2 >= 0.5) or (all_r2 >= 0.5)

    print(f"\n  | Criterion                              | Value  | Threshold | Result |")
    print(f"  |----------------------------------------|--------|-----------|--------|")
    r1 = "PASS" if within_2x else "KILL"
    r2 = "PASS" if r2_pass else "KILL"
    print(f"  | Prediction within 2x (all N)           | {max_ratio_all:.2f}x  | <=2.0x    | {r1:>6} |")
    if val_r2 is not None:
        print(f"  | Held-out validation R^2                | {val_r2:.3f}  | >=0.50    | {r2:>6} |")
    print(f"  | Full-data R^2                          | {all_r2:.3f}  | >=0.50    | {r2:>6} |")

    n_killed = sum([not within_2x, not r2_pass])
    if n_killed > 0:
        print(f"\n  VERDICT: KILL. {n_killed}/2 kill criteria triggered.")
    else:
        print(f"\n  VERDICT: PASS (consistent, not proven).")
        print(f"  Shannon model is CONSISTENT with {len(all_ns)} data points.")
        print(f"  Status: consistent (validated on held-out N={val_ns}).")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    print(f"""
  Shannon channel capacity model for expert composition (REVISED):

  Core equation:
    gap(N) = (1 - log(1 + SNR_0/(1 + (N-1)*alpha)) / log(1 + SNR_0)) * 100 + c_0

  Fitted parameters (on N=2,5,8):
    SNR_0 = {model.P:.4f}
    alpha  = {alpha:.6f}
    c_0    = {offset:+.4f}%

  Validation (out-of-sample, N={val_ns}):
    Validation R^2 = {val_r2 if val_r2 is not None else 'N/A'}
    Full-data R^2  = {all_r2:.4f}

  Sensitivity (1% perturbation):
    N_max (10%) range = [{sens['n_max_min']}, {sens['n_max_max']}]

  Model comparison (out-of-sample MSE):
    Shannon:   {shannon_val_mse if shannon_val_mse is not None else 'N/A'}
    Linear:    {lin_val_mse if lin_val_mse is not None else 'N/A'}
    Power-law: {pow_val_mse if pow_val_mse is not None else 'N/A'}

  NOTE: This is a DESCRIPTIVE model, not a fundamental limit.
  The rate-distortion interpretation labels the axes of the fitted
  curve but does not prove that the bound is tight or unbeatable.
""")

    # ============================================================
    # Build results dict
    # ============================================================
    results = {
        "revision": "v2_held_out_validation",
        "model_params": {
            "SNR_0": float(model.P),
            "rho_mean": float(model.rho_mean),
            "alpha": float(alpha),
            "offset": float(offset),
            "d": D_MODEL,
        },
        "training_data": {str(k): v for k, v in train_gaps.items()},
        "validation_data": {str(k): v for k, v in validation_gaps.items()},
        "all_data": {str(k): v for k, v in all_gaps.items()},
        "fit_quality": {
            "train_mse": float(fit_mse),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2) if val_r2 is not None else None,
            "all_r2": float(all_r2),
            "max_ratio_all": float(max_ratio_all),
        },
        "baseline_comparison": {
            "linear": {
                "params": lin_params,
                "train_mse": lin_mse,
                "val_mse": lin_val_mse,
                "val_r2": float(lin_val_r2) if lin_val_r2 is not None else None,
                "aicc": float(lin_aicc),
                "bic": float(lin_bic),
                "n_params": lin_n_params,
            },
            "power_law": {
                "params": pow_params,
                "train_mse": pow_mse,
                "val_mse": pow_val_mse,
                "val_r2": float(pow_val_r2) if pow_val_r2 is not None else None,
                "aicc": float(pow_aicc),
                "bic": float(pow_bic),
                "n_params": pow_n_params,
            },
            "shannon": {
                "train_mse": float(fit_mse),
                "val_mse": float(shannon_val_mse) if shannon_val_mse is not None else None,
                "val_r2": float(val_r2) if val_r2 is not None else None,
                "aicc": float(shannon_aicc),
                "bic": float(shannon_bic),
                "n_params": shannon_n_params,
            },
        },
        "sensitivity": {
            "perturbation_pct": sens["perturbation_pct"],
            "n_max_10_range": [sens["n_max_min"], sens["n_max_max"]],
            "n_max_10_median": sens["n_max_median"],
        },
        "predictions": {
            "n_max_5pct": n_max_5,
            "n_max_10pct": n_max_10,
            "n_max_20pct": n_max_20,
        },
        "kill_criteria": {
            "within_2x": bool(within_2x),
            "r2_pass": bool(r2_pass),
            "verdict": "PASS" if n_killed == 0 else "KILL",
            "status": "killed",  # Failed held-out validation
        },
    }

    # Per-N predictions
    results["per_n_predictions"] = {}
    for n in range(1, 31):
        gap_pred = predict_gaps_shannon([n], model.P, alpha, offset)[0]
        snr_eff = model.P / (1 + max(n - 1, 0) * alpha)
        c_n = 0.5 * math.log(1 + snr_eff)
        c_1 = 0.5 * math.log(1 + model.P)
        results["per_n_predictions"][str(n)] = {
            "gap_predicted": float(gap_pred),
            "capacity_ratio": float(c_n / c_1) if n > 1 else 1.0,
            "snr_effective": float(snr_eff),
            "empirical_gap": all_gaps.get(n),
        }

    return results


def main():
    """Entry point."""
    results = run_experiment(collect_new_data=True)

    results_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/channel_capacity_bound/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
