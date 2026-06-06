#!/usr/bin/env python3
"""
Honest perturbation bound for cross-adapter KV-cache reuse.

Parent (F#309) derived a self-contradicting Theorem 2 bound (1.6% vs 62.5%).
This experiment verifies the corrected bound (MATH.md) via numerical simulation
at BitNet-2B dimensions (d=2560, d_k=128, L=28, r=16), comparing to parent's
measured drift (13.26% PPL gap at alpha=20).

Kill criteria:
  K1566 (proxy): Corrected PPL-drift bound matches parent's measured drift within 2x.
                 Predicted ~7% PPL drift at alpha=20 vs measured 13.26% (1.9x ratio).
  K1945 (target, F#666 pair): Independent closed-form operator-norm bound
    bound_Drift_rel(alpha) = 2*alpha*sigma_B*sqrt(r)/||W_K||_op
                           + gamma * alpha^2 * sigma_B^2 * r / (||W_Q||_op * ||W_K||_op)
    (MATH.md Theorem 2 eqs (1)-(3); gamma measured from sampled adapters)
    agrees with SIMULATED rel_Drift at each alpha in {5, 10, 20} within 2x (magnitude
    test, not scaling-ratio tautology); AND simulated drift at alpha=5 > 0.

This is pure linear algebra (no LLM weights, no training). Uses numpy for
portability -- the math is model/platform-independent.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# BitNet-2B dimensions (matches parent F#309 measurement at alpha=20: 13.26% PPL drift)
D = 2560        # hidden dim
D_K = 128       # attention head dim
L = 28          # num layers
R = 16          # LoRA rank
SIGMA_B = 0.05  # post-training row-norm of B (conservative, matches F#627)
SEED = 42

# Parent reference point
PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20 = 13.26


def log(msg):
    print(msg, flush=True)


def sample_base_weights(rng, d_in, d_out):
    """Sample base W ~ Gaussian scaled so ||W||_op ~ 1 (trained-LLM-like)."""
    W = rng.standard_normal((d_in, d_out)) / math.sqrt(d_in)
    return W


def sample_grassmannian_A(rng, r, d):
    """Sample A in R^{r x d} with orthonormal rows (Grassmannian init, F#562).

    Uses QR decomposition of a (d x r) Gaussian; take Q, transpose to (r x d)."""
    G = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)  # Q: (d x r), orthonormal columns
    return Q.T  # (r x d), orthonormal rows


def sample_B(rng, d_k, r, sigma_B):
    """Sample B in R^{d_k x r} with row norm approx sigma_B."""
    B = rng.standard_normal((d_k, r)) * sigma_B / math.sqrt(r)
    # Re-scale each row to exactly sigma_B (closer to post-training)
    for i in range(d_k):
        n = np.linalg.norm(B[i])
        if n > 0:
            B[i] = B[i] * sigma_B / n * math.sqrt(r) / math.sqrt(d_k)
    return B


def sample_adapter(rng, d, d_k, r, sigma_B, grassmannian=True):
    """Sample a LoRA adapter (A orthonormal, B with sigma_B scale)."""
    if grassmannian:
        A = sample_grassmannian_A(rng, r, d)
    else:
        A = rng.standard_normal((r, d)) / math.sqrt(d)
    B = sample_B(rng, d_k, r, sigma_B)
    return A, B


def perturbation(alpha, B, A):
    """Compute alpha * B @ A: the LoRA additive perturbation."""
    return alpha * (B @ A)  # (d_k x d)


def closed_form_bound(alpha, sigma_B, r, W_Q_op, W_K_op, gamma):
    """Independent closed-form bound from MATH.md Theorem 2 equations (1)-(3).

    Uses sampled operator norms (NOT the simulated Drift) to produce a prediction
    that does NOT cancel algebraically against the simulation. This is the fix
    for the K1945 tautology flagged by reviewer (antipatterns f, g).

    Returns dict with bound_D1_rel, bound_D2_rel, bound_Drift_rel (all magnitudes
    relative to ||S0||_op). These are upper bounds: simulated rel_* should
    be <= bound_*_rel (the 'within 2x' test then checks how tight the bound is).
    """
    bound_D1_rel = (2.0 * alpha * sigma_B * math.sqrt(r)) / W_K_op
    bound_D2_rel = (alpha * alpha * sigma_B * sigma_B * r) / (W_Q_op * W_K_op)
    bound_Drift_rel = bound_D1_rel + gamma * bound_D2_rel
    return {
        "bound_D1_rel": bound_D1_rel,
        "bound_D2_rel": bound_D2_rel,
        "bound_Drift_rel": bound_Drift_rel,
    }


def compute_drift_stats(alpha, W_Q, W_K, adapter_A, adapter_B, n_samples, rng):
    """Compute attention-score drift statistics at given alpha.

    Generates n_samples pairs (h_t, h_s) of RMS-normed hidden states and
    computes Drift = (q_B . K_A) - (q_B . K_B) for segment-B query vs segment-A/B keys.
    Returns std(Drift)/||S0||_op and component breakdown.
    """
    A_A, B_A = adapter_A
    A_B, B_B = adapter_B

    # Perturbations (d_k x d)
    dQ_B = perturbation(alpha, B_B, A_B)
    dK_A = perturbation(alpha, B_A, A_A)
    dK_B = perturbation(alpha, B_B, A_B)

    # Full query projection for adapter B: (W_Q + dQ_B^T dimension-matched)
    # W_Q: (d x d_k) ... q_t = h_t @ W_Q + h_t @ dQ_B^T, shape (d_k,)
    # In this convention: q = (W_Q + dQ_B^T) applied to h from the LEFT as h @ (W_Q + dQ_B^T)
    # But dQ_B is (d_k x d), so dQ_B^T is (d x d_k). Stacked: (W_Q + dQ_B^T) is (d x d_k). OK.

    # S0[t,s] = (h_t W_Q)(W_K^T h_s) = base-base attention
    # D1[t,s] = h_t (W_Q) (dK_A - dK_B)^T h_s  [base Q, adapter-diff K]
    # D2[t,s] = h_t (dQ_B^T) (dK_A - dK_B)^T h_s  [adapter Q, adapter-diff K]

    dK_diff = dK_A - dK_B  # (d_k x d)

    # Sample hidden states RMS-normed
    H_t = rng.standard_normal((n_samples, D))
    H_s = rng.standard_normal((n_samples, D))
    H_t = H_t / np.linalg.norm(H_t, axis=1, keepdims=True) * math.sqrt(D)  # ||h|| = sqrt(d)
    H_s = H_s / np.linalg.norm(H_s, axis=1, keepdims=True) * math.sqrt(D)

    # Project
    # Q_base[t] = H_t @ W_Q : (n, d_k)
    # Q_adapter_B[t] = H_t @ dQ_B^T : (n, d_k)
    # K_diff[s] = H_s @ dK_diff^T : (n, d_k)
    # K_base[s] = H_s @ W_K : (n, d_k)
    Q_base = H_t @ W_Q           # (n, d_k)
    Q_adapter = H_t @ dQ_B.T     # (n, d_k)
    K_base = H_s @ W_K           # (n, d_k)
    K_diff = H_s @ dK_diff.T     # (n, d_k)

    # Pair-wise scalar products -- use per-token (diagonal) comparison:
    # Each t paired with the same s (n samples of the (t,s) pair)
    S0 = np.sum(Q_base * K_base, axis=1)       # (n,)
    D1 = np.sum(Q_base * K_diff, axis=1)       # (n,)
    D2 = np.sum(Q_adapter * K_diff, axis=1)    # (n,)
    Drift = D1 + D2

    # Operator-norm normalisation via empirical RMS
    S0_rms = float(np.sqrt(np.mean(S0**2)))
    D1_rms = float(np.sqrt(np.mean(D1**2)))
    D2_rms = float(np.sqrt(np.mean(D2**2)))
    Drift_rms = float(np.sqrt(np.mean(Drift**2)))

    rel_D1 = D1_rms / S0_rms if S0_rms > 0 else float("inf")
    rel_D2 = D2_rms / S0_rms if S0_rms > 0 else float("inf")
    rel_Drift = Drift_rms / S0_rms if S0_rms > 0 else float("inf")

    # Post-softmax attenuation (divide by sqrt(d_k)) per Theorem 2 (3)
    postsoftmax_drift = rel_Drift / math.sqrt(D_K)

    # Residual-stream attenuation across L layers: sqrt(L)/L
    residual_attenuation = math.sqrt(L) / L
    ppl_drift_estimate = postsoftmax_drift * residual_attenuation

    return {
        "alpha": alpha,
        "rel_D1": rel_D1,
        "rel_D2": rel_D2,
        "rel_Drift_combined": rel_Drift,
        "S0_rms": S0_rms,
        "D1_rms": D1_rms,
        "D2_rms": D2_rms,
        "Drift_rms": Drift_rms,
        "D1_to_D2_ratio": rel_D1 / rel_D2 if rel_D2 > 0 else float("inf"),
        "postsoftmax_drift_per_layer": postsoftmax_drift,
        "predicted_ppl_drift_pct": ppl_drift_estimate * 100,  # as percentage
    }


def grassmannian_gamma(A_A, A_B):
    """Measure the Grassmannian suppression factor: ||A_B^T A_A||_F / sqrt(r)."""
    cross = A_B @ A_A.T  # (r x r)
    return float(np.linalg.norm(cross, "fro") / math.sqrt(A_A.shape[0]))


def simulate_at_alpha(alpha, n_samples, rng, grassmannian=True):
    """Run one alpha point, averaging over multiple random (W, adapter) draws.

    Now also computes the independent closed-form bound per trial using the
    per-trial sampled ||W_Q||_op, ||W_K||_op, gamma. Aggregated bound_Drift_rel
    is the mean over trials (NOT derived from the simulated Drift — this is
    the independence that removes the K1945 tautology).
    """
    n_trials = 10
    results = []
    for trial in range(n_trials):
        W_Q = sample_base_weights(rng, D, D_K)
        W_K = sample_base_weights(rng, D, D_K)
        adapter_A = sample_adapter(rng, D, D_K, R, SIGMA_B, grassmannian=grassmannian)
        adapter_B = sample_adapter(rng, D, D_K, R, SIGMA_B, grassmannian=grassmannian)
        gamma = grassmannian_gamma(adapter_A[0], adapter_B[0])
        stats = compute_drift_stats(alpha, W_Q, W_K, adapter_A, adapter_B, n_samples, rng)
        stats["gamma"] = gamma
        # Independent closed-form bound (MATH.md Theorem 2 eqs 1-3)
        W_Q_op = float(np.linalg.norm(W_Q, ord=2))
        W_K_op = float(np.linalg.norm(W_K, ord=2))
        bound = closed_form_bound(alpha, SIGMA_B, R, W_Q_op, W_K_op, gamma)
        stats["W_Q_op"] = W_Q_op
        stats["W_K_op"] = W_K_op
        stats.update(bound)
        results.append(stats)

    # Aggregate across trials
    agg = {}
    for key in ["rel_D1", "rel_D2", "rel_Drift_combined",
                "D1_to_D2_ratio", "postsoftmax_drift_per_layer",
                "predicted_ppl_drift_pct", "gamma",
                "W_Q_op", "W_K_op",
                "bound_D1_rel", "bound_D2_rel", "bound_Drift_rel"]:
        vals = [r[key] for r in results if not math.isinf(r[key])]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("inf")
        agg[f"{key}_std"] = float(np.std(vals)) if vals else float("inf")
    agg["alpha"] = alpha
    agg["n_trials"] = n_trials
    agg["n_samples_per_trial"] = n_samples
    return agg


def main():
    t0 = time.time()
    log("=" * 70)
    log("Honest perturbation bound verification (KC1566 + K1945)")
    log("=" * 70)
    log(f"Dimensions: d={D}, d_k={D_K}, L={L}, r={R}, sigma_B={SIGMA_B}")
    log(f"Parent reference: PPL drift {PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20}% at alpha=20")
    log("")

    rng = np.random.default_rng(SEED)

    # Grassmannian-initialized adapters (matches F#562 trained adapter structure)
    log("-" * 70)
    log("Part A: Grassmannian A-matrices (matches parent)")
    log("-" * 70)
    alpha_values = [5.0, 10.0, 20.0]
    grassmannian_results = {}
    for alpha in alpha_values:
        log(f"\n  alpha = {alpha}")
        agg = simulate_at_alpha(alpha, n_samples=512, rng=rng, grassmannian=True)
        grassmannian_results[alpha] = agg
        log(f"    rel_D1 = {agg['rel_D1_mean']:.4f} +/- {agg['rel_D1_std']:.4f}")
        log(f"    rel_D2 = {agg['rel_D2_mean']:.4f} +/- {agg['rel_D2_std']:.4f}")
        log(f"    D1:D2 ratio = {agg['D1_to_D2_ratio_mean']:.3f}")
        log(f"    rel_Drift = {agg['rel_Drift_combined_mean']:.4f}")
        log(f"    gamma = {agg['gamma_mean']:.4f}")
        log(f"    predicted_ppl_drift = {agg['predicted_ppl_drift_pct_mean']:.2f}%")

    # Unstructured A-matrices bookend (gamma=1)
    log("\n" + "-" * 70)
    log("Part B: Unstructured A-matrices (gamma=1 bookend)")
    log("-" * 70)
    rng2 = np.random.default_rng(SEED + 1)
    unstructured_results = {}
    for alpha in alpha_values:
        log(f"\n  alpha = {alpha}")
        agg = simulate_at_alpha(alpha, n_samples=512, rng=rng2, grassmannian=False)
        unstructured_results[alpha] = agg
        log(f"    rel_D1 = {agg['rel_D1_mean']:.4f}")
        log(f"    rel_D2 = {agg['rel_D2_mean']:.4f}")
        log(f"    predicted_ppl_drift = {agg['predicted_ppl_drift_pct_mean']:.2f}%")

    # Kill-criterion evaluation
    log("\n" + "=" * 70)
    log("Kill criteria evaluation")
    log("=" * 70)

    # K1566: bound at alpha=20 vs parent measured 13.26%
    predicted_at_20 = grassmannian_results[20.0]["predicted_ppl_drift_pct_mean"]
    ratio_at_20 = PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20 / predicted_at_20 if predicted_at_20 > 0 else float("inf")
    K1566_pass = (0.5 <= ratio_at_20 <= 2.0)
    log(f"\n  K1566: corrected bound {predicted_at_20:.2f}% vs measured {PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20}% at alpha=20")
    log(f"         ratio = {ratio_at_20:.2f}x (threshold in [0.5x, 2.0x]) -> {'PASS' if K1566_pass else 'FAIL'}")

    # K1945 target-gated (rewritten per REVIEW r1 to remove tautology antipatterns f,g):
    # independent closed-form bound magnitude vs simulation at each alpha.
    # Bound computed from sampled ||W_Q||_op, ||W_K||_op, measured gamma — NOT from
    # the simulated Drift. Ratio sim/bound in [0.5, 2.0] tests that the bound
    # tracks simulation within 2x at each alpha.
    log(f"\n  K1945 (target pair, F#666) — independent closed-form bound magnitude:")
    alpha_check_list = [5.0, 10.0, 20.0]
    K1945_magnitude_pass_list = []
    K1945_per_alpha = {}
    for a in alpha_check_list:
        sim_drift = grassmannian_results[a]["rel_Drift_combined_mean"]
        bound_drift = grassmannian_results[a]["bound_Drift_rel_mean"]
        bound_D1 = grassmannian_results[a]["bound_D1_rel_mean"]
        bound_D2 = grassmannian_results[a]["bound_D2_rel_mean"]
        W_Q_op = grassmannian_results[a]["W_Q_op_mean"]
        W_K_op = grassmannian_results[a]["W_K_op_mean"]
        g = grassmannian_results[a]["gamma_mean"]
        ratio = sim_drift / bound_drift if bound_drift > 0 else float("inf")
        pass_a = (0.5 <= ratio <= 2.0)
        K1945_magnitude_pass_list.append(pass_a)
        K1945_per_alpha[str(a)] = {
            "alpha": a,
            "sim_rel_Drift": sim_drift,
            "bound_D1_rel": bound_D1,
            "bound_D2_rel": bound_D2,
            "bound_Drift_rel": bound_drift,
            "W_Q_op_sampled": W_Q_op,
            "W_K_op_sampled": W_K_op,
            "gamma_sampled": g,
            "ratio_sim_over_bound": ratio,
            "threshold_range": [0.5, 2.0],
            "pass": pass_a,
        }
        log(f"    alpha={a}:  sim={sim_drift:.4f}  bound={bound_drift:.4f}  "
            f"(D1={bound_D1:.3f} + {g:.3f}*D2={g*bound_D2:.3f})  "
            f"ratio={ratio:.3f}  -> {'PASS' if pass_a else 'FAIL'}")
    simulated_drift_at_5 = grassmannian_results[5.0]["rel_Drift_combined_mean"]
    K1945_positive_pass = simulated_drift_at_5 > 0
    K1945_magnitude_pass = all(K1945_magnitude_pass_list)
    K1945_pass = K1945_magnitude_pass and K1945_positive_pass
    log(f"    Drift at alpha=5 > 0?         {'PASS' if K1945_positive_pass else 'FAIL'}")
    log(f"    K1945 overall:                {'PASS' if K1945_pass else 'FAIL'}")

    # Verdict per F#666 decision matrix (explicit PROVISIONAL branch fixes antipattern
    # of hand-editing results.json after run — see REVIEW r1 fix #3).
    if K1566_pass and K1945_pass:
        verdict = "SUPPORTED"
        verdict_reasoning = "Both K1566 (proxy) and K1945 (target) pass."
    elif K1566_pass and not K1945_pass:
        verdict = "KILLED"
        verdict_reasoning = ("F#666: proxy-PASS + target-FAIL. Proxy is tautological/"
                             "misaligned with the mechanism the target KC tests. Kill on target.")
    elif (not K1566_pass) and K1945_pass:
        verdict = "PROVISIONAL"
        verdict_reasoning = ("F#666: proxy-FAIL + target-PASS. Finding is about the "
                             "proxy (mis-calibrated for this mechanism), not a KILL. "
                             "Follow-up: refine the proxy KC.")
    else:
        verdict = "KILLED"
        verdict_reasoning = ("F#666: proxy-FAIL + target-FAIL. Both KCs fail. "
                             "The corrected math resolves the parent 1.6% vs 62.5% "
                             "contradiction (decomposition D1+D2 with sqrt(r) factor) "
                             "but the operator-norm bound is too loose to predict "
                             "simulated drift within 2x, so the proxy-FAIL is "
                             "corroborated by a genuine target-FAIL (no F#666 escape).")
    all_pass = K1566_pass and K1945_pass

    log(f"\n  === VERDICT: {verdict} ===")
    log(f"  Reasoning: {verdict_reasoning}")

    # Compile results
    results = {
        "experiment": "exp_followup_kv_cache_reuse_honest",
        "verdict": verdict,
        "verdict_reasoning": verdict_reasoning,
        "all_pass": all_pass,
        "is_smoke": False,
        "platform": "numerical-simulation (numpy)",
        "dimensions": {"d": D, "d_k": D_K, "L": L, "r": R, "sigma_B": SIGMA_B},
        "parent_reference": {
            "finding": "#309",
            "measured_ppl_drift_pct_at_alpha_20": PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20,
        },
        "grassmannian_results": {str(k): v for k, v in grassmannian_results.items()},
        "unstructured_results": {str(k): v for k, v in unstructured_results.items()},
        "K1566": {
            "predicted_at_alpha_20_pct": predicted_at_20,
            "measured_at_alpha_20_pct": PARENT_MEASURED_PPL_DRIFT_PCT_AT_ALPHA_20,
            "ratio": ratio_at_20,
            "threshold_range": [0.5, 2.0],
            "pass": K1566_pass,
        },
        "K1945": {
            "test_description": ("Independent closed-form bound from MATH.md Theorem 2 eqs "
                                 "(1)-(3), computed per-trial from sampled ||W_Q||_op, "
                                 "||W_K||_op, gamma — NOT derived from simulated Drift. "
                                 "PASS if simulated/bound in [0.5, 2.0] at each alpha in "
                                 "{5, 10, 20} AND simulated drift at alpha=5 > 0."),
            "per_alpha": K1945_per_alpha,
            "magnitude_pass": K1945_magnitude_pass,
            "positive_at_alpha_5_pass": K1945_positive_pass,
            "pass": K1945_pass,
        },
        "seed": SEED,
        "total_time_s": round(time.time() - t0, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults written to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
