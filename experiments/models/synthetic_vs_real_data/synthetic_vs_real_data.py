#!/usr/bin/env python3
"""
Synthetic vs Real Data Quality: Simulation Study

Compares three training data regimes for SOLE expert distillation:
  (a) Synthetic only (LLM-generated, Groq/Phi-1 style)
  (b) Real only (HuggingFace codeparrot-clean style)
  (c) Mixed (sweep from 10% to 90% synthetic)

MATHEMATICAL MODEL
==================
Ground-truth task: W* in R^{d x d} with effective rank = r.
Each data source generates (x, y) pairs where y = x @ W* + noise.

Key differences between sources:
  SYNTHETIC: Low label noise (sigma_s = 0.05), concentrated inputs (5 modes,
    Dirichlet alpha=0.5), systematic bias. Models LLM-generated "textbook" data.
  REAL: High label noise (sigma_r = 0.30), diverse inputs (20 modes,
    Dirichlet alpha=2.0), no bias. Models real-world codeparrot-clean data.

The tension: synthetic has cleaner SUPERVISION but narrower COVERAGE.
Real has noisier supervision but broader coverage. For LoRA training
(which learns a low-rank approximation), coverage matters because the
LoRA must generalize to the full input space at test time.

We evaluate on THREE held-out distributions:
  1. Uniform (unbiased coverage of R^d)
  2. Synthetic-like (concentrated, same modes)
  3. Real-like (diverse, same modes)

Calibration from literature:
  - Phi-1 (Gunasekar et al. 2023): synthetic-only matches real on HumanEval
  - Orca (Mukherjee et al. 2023): synthetic instruction-following competitive
  - Shumailov et al. 2024: model collapse from recursive synthetic generation

Pure numpy/scipy simulation. Runs in <60 seconds on Apple Silicon.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import subspace_angles

# ── Configuration ────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1337]
D_MODEL = 64
RANK = 8
N_TRAIN = 1000
N_EVAL = 500
N_EXPERTS = 4
MIXING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_GRADIENT_STEPS = 500
LR = 0.01
BATCH_SIZE = 64


# ── Data Source Models ───────────────────────────────────────────

@dataclass
class DataSourceConfig:
    name: str
    n_input_modes: int
    mode_concentration: float   # Dirichlet alpha
    input_noise_scale: float    # spread around mode centers
    label_noise_std: float      # label noise sigma
    systematic_bias: float      # input bias magnitude
    benchmark_overlap: float    # P(overlap with benchmark)


SYNTHETIC_CFG = DataSourceConfig(
    name="synthetic",
    n_input_modes=5,
    mode_concentration=0.5,
    input_noise_scale=0.3,
    label_noise_std=0.05,
    systematic_bias=0.25,
    benchmark_overlap=0.10,
)

REAL_CFG = DataSourceConfig(
    name="real",
    n_input_modes=20,
    mode_concentration=2.0,
    input_noise_scale=0.8,
    label_noise_std=0.30,
    systematic_bias=0.02,
    benchmark_overlap=0.02,
)


# ── Data Generation ──────────────────────────────────────────────

def make_mode_centers(n_modes: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-norm mode centers in R^d."""
    centers = rng.standard_normal((n_modes, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def generate_inputs(cfg: DataSourceConfig, n: int, d: int,
                    rng: np.random.Generator,
                    mode_centers: np.ndarray,
                    bias_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate input vectors from mixture of Gaussians."""
    alpha = np.ones(cfg.n_input_modes) * cfg.mode_concentration
    weights = rng.dirichlet(alpha)
    assignments = rng.choice(cfg.n_input_modes, size=n, p=weights)

    X = np.zeros((n, d))
    for i in range(n):
        X[i] = mode_centers[assignments[i]] + \
               rng.standard_normal(d) * cfg.input_noise_scale

    if cfg.systematic_bias > 0 and bias_dir is not None:
        X += cfg.systematic_bias * bias_dir[np.newaxis, :]
    return X


def generate_labels(X: np.ndarray, W_star: np.ndarray,
                    noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return X @ W_star + rng.standard_normal(X.shape) * noise_std


# ── LoRA Training ────────────────────────────────────────────────

def train_lora(X: np.ndarray, y: np.ndarray, d: int, rank: int,
               n_steps: int, lr: float,
               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Train B (learned) with A frozen. Returns (A, B)."""
    n = X.shape[0]
    A = rng.standard_normal((rank, d)) * (1.0 / np.sqrt(d))
    B = np.zeros((d, rank))

    for step in range(n_steps):
        idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
        xb, yb = X[idx], y[idx]
        y_pred = xb @ A.T @ B.T
        error = y_pred - yb
        grad_B = error.T @ xb @ A.T / len(idx)
        # Cosine decay schedule
        decay = 0.5 * (1 + np.cos(np.pi * step / n_steps))
        B -= lr * (0.1 + 0.9 * decay) * grad_B

    return A, B


# ── Metrics ──────────────────────────────────────────────────────

def quality(A, B, X_eval, W_star):
    """1 - ||pred - true||_F / ||true||_F on held-out data."""
    y_pred = X_eval @ A.T @ B.T
    y_true = X_eval @ W_star
    return max(0, 1 - np.linalg.norm(y_pred - y_true, 'fro') /
               (np.linalg.norm(y_true, 'fro') + 1e-10))


def effective_rank(X):
    """Effective rank = exp(entropy of normalized singular values)."""
    S = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = S / (S.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def mean_cos(experts):
    """Mean pairwise |cos| of flattened B@A."""
    n = len(experts)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            wi = (experts[i][1] @ experts[i][0]).flatten()
            wj = (experts[j][1] @ experts[j][0]).flatten()
            vals.append(abs(np.dot(wi, wj) /
                           (np.linalg.norm(wi) * np.linalg.norm(wj) + 1e-10)))
    return float(np.mean(vals))


def min_principal_angle_deg(experts):
    """Mean minimum principal angle (deg) between B subspaces."""
    n = len(experts)
    if n < 2:
        return 90.0
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            theta = subspace_angles(experts[i][1], experts[j][1])
            angles.append(np.min(theta))
    return float(np.degrees(np.mean(angles)))


def contamination_stats(overlap_rate, n_train, bench_size=164):
    expected = n_train * overlap_rate
    p_any = min(1.0, 1 - (1 - overlap_rate) ** n_train)
    boost = (expected / bench_size) * 0.30 * 100  # percent
    return {"expected_overlap": round(expected, 1),
            "p_any": round(p_any, 4),
            "boost_pct": round(boost, 2)}


# ── Main Experiment ──────────────────────────────────────────────

def run_seed(seed: int) -> Dict:
    rng = np.random.default_rng(seed)

    # Shared ground truth: rank-r task
    U = rng.standard_normal((D_MODEL, RANK))
    V = rng.standard_normal((RANK, D_MODEL))
    W_star = U @ V * 0.1

    bias_dir = rng.standard_normal(D_MODEL)
    bias_dir /= np.linalg.norm(bias_dir)

    # Pre-generate mode centers for each source (stable across experts)
    synth_modes = make_mode_centers(SYNTHETIC_CFG.n_input_modes, D_MODEL, rng)
    real_modes = make_mode_centers(REAL_CFG.n_input_modes, D_MODEL, rng)

    # Evaluation data: uniform
    X_eval_uniform = rng.standard_normal((N_EVAL, D_MODEL)) * 0.5

    # Evaluation data: synthetic-like
    eval_rng = np.random.default_rng(seed + 10000)
    X_eval_synth = generate_inputs(SYNTHETIC_CFG, N_EVAL, D_MODEL,
                                   eval_rng, synth_modes, bias_dir)
    # Evaluation data: real-like
    X_eval_real = generate_inputs(REAL_CFG, N_EVAL, D_MODEL,
                                  eval_rng, real_modes, None)

    result = {"seed": seed}

    # ── Part 1: Pure regimes ──────────────────────────────────────
    regimes = {}
    expert_bank = {}

    for cfg, modes, bias in [(SYNTHETIC_CFG, synth_modes, bias_dir),
                              (REAL_CFG, real_modes, None)]:
        experts = []
        qs_uniform, qs_synth, qs_real, divs = [], [], [], []

        for _ in range(N_EXPERTS):
            X = generate_inputs(cfg, N_TRAIN, D_MODEL, rng, modes, bias)
            y = generate_labels(X, W_star, cfg.label_noise_std, rng)
            divs.append(effective_rank(X))

            A, B = train_lora(X, y, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
            experts.append((A, B))

            qs_uniform.append(quality(A, B, X_eval_uniform, W_star))
            qs_synth.append(quality(A, B, X_eval_synth, W_star))
            qs_real.append(quality(A, B, X_eval_real, W_star))

        regimes[cfg.name] = {
            "quality_uniform": {"mean": float(np.mean(qs_uniform)),
                                "std": float(np.std(qs_uniform))},
            "quality_synth_eval": {"mean": float(np.mean(qs_synth)),
                                   "std": float(np.std(qs_synth))},
            "quality_real_eval": {"mean": float(np.mean(qs_real)),
                                  "std": float(np.std(qs_real))},
            "effective_rank": {"mean": float(np.mean(divs)),
                               "std": float(np.std(divs))},
            "cos_sim": mean_cos(experts),
            "angle_deg": min_principal_angle_deg(experts),
            "contamination": contamination_stats(cfg.benchmark_overlap, N_TRAIN),
        }
        expert_bank[cfg.name] = experts

    result["regimes"] = regimes

    # ── Part 2: Mixing ratio sweep ────────────────────────────────
    sweep = {}
    for ratio in MIXING_RATIOS:
        n_s = int(N_TRAIN * ratio)
        n_r = N_TRAIN - n_s
        parts_X, parts_y = [], []
        if n_s > 0:
            Xs = generate_inputs(SYNTHETIC_CFG, n_s, D_MODEL, rng, synth_modes, bias_dir)
            ys = generate_labels(Xs, W_star, SYNTHETIC_CFG.label_noise_std, rng)
            parts_X.append(Xs); parts_y.append(ys)
        if n_r > 0:
            Xr = generate_inputs(REAL_CFG, n_r, D_MODEL, rng, real_modes, None)
            yr = generate_labels(Xr, W_star, REAL_CFG.label_noise_std, rng)
            parts_X.append(Xr); parts_y.append(yr)

        Xm = np.vstack(parts_X)
        ym = np.vstack(parts_y)
        perm = rng.permutation(len(Xm))
        Xm, ym = Xm[perm], ym[perm]

        A, B = train_lora(Xm, ym, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
        sweep[f"{ratio:.1f}"] = {
            "ratio": ratio,
            "q_uniform": quality(A, B, X_eval_uniform, W_star),
            "q_synth": quality(A, B, X_eval_synth, W_star),
            "q_real": quality(A, B, X_eval_real, W_star),
            "eff_rank": effective_rank(Xm),
            "contamination": contamination_stats(
                ratio * SYNTHETIC_CFG.benchmark_overlap +
                (1 - ratio) * REAL_CFG.benchmark_overlap, N_TRAIN),
        }
    result["mixing_sweep"] = sweep

    # ── Part 3: Cross-regime orthogonality ────────────────────────
    mixed_experts = []
    for _ in range(N_EXPERTS):
        Xs = generate_inputs(SYNTHETIC_CFG, N_TRAIN // 2, D_MODEL, rng, synth_modes, bias_dir)
        ys = generate_labels(Xs, W_star, SYNTHETIC_CFG.label_noise_std, rng)
        Xr = generate_inputs(REAL_CFG, N_TRAIN // 2, D_MODEL, rng, real_modes, None)
        yr = generate_labels(Xr, W_star, REAL_CFG.label_noise_std, rng)
        Xm = np.vstack([Xs, Xr]); ym = np.vstack([ys, yr])
        p = rng.permutation(len(Xm))
        Xm, ym = Xm[p], ym[p]
        A, B = train_lora(Xm, ym, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
        mixed_experts.append((A, B))

    result["orthogonality"] = {
        "within_synthetic": mean_cos(expert_bank["synthetic"]),
        "within_real": mean_cos(expert_bank["real"]),
        "within_mixed": mean_cos(mixed_experts),
        "cross_synth_real": mean_cos(
            expert_bank["synthetic"][:2] + expert_bank["real"][:2]),
        "angle_synth": min_principal_angle_deg(expert_bank["synthetic"]),
        "angle_real": min_principal_angle_deg(expert_bank["real"]),
        "angle_mixed": min_principal_angle_deg(mixed_experts),
    }

    return result


def aggregate(all_results: List[Dict]) -> Dict:
    n = len(all_results)
    agg = {
        "n_seeds": n,
        "config": {"d": D_MODEL, "r": RANK, "n_train": N_TRAIN, "n_eval": N_EVAL,
                    "n_experts": N_EXPERTS, "steps": N_GRADIENT_STEPS, "lr": LR},
        "regimes": {},
        "mixing_sweep": {},
        "orthogonality": {},
        "kill_criteria": {},
    }

    # Regime aggregation
    for regime in ["synthetic", "real"]:
        def v(key):
            return [r["regimes"][regime][key] for r in all_results]

        q_uni = [r["regimes"][regime]["quality_uniform"]["mean"] for r in all_results]
        q_syn = [r["regimes"][regime]["quality_synth_eval"]["mean"] for r in all_results]
        q_rea = [r["regimes"][regime]["quality_real_eval"]["mean"] for r in all_results]
        er = [r["regimes"][regime]["effective_rank"]["mean"] for r in all_results]
        cs = [r["regimes"][regime]["cos_sim"] for r in all_results]
        an = [r["regimes"][regime]["angle_deg"] for r in all_results]

        def stats(vals):
            return {"mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci95": float(1.96 * np.std(vals) / np.sqrt(n))}

        agg["regimes"][regime] = {
            "quality_uniform": stats(q_uni),
            "quality_synth_eval": stats(q_syn),
            "quality_real_eval": stats(q_rea),
            "effective_rank": stats(er),
            "cos_sim": stats(cs),
            "angle_deg": stats(an),
            "contamination": all_results[0]["regimes"][regime]["contamination"],
        }

    # K1: synthetic >15% worse on UNIFORM eval
    sq = agg["regimes"]["synthetic"]["quality_uniform"]["mean"]
    rq = agg["regimes"]["real"]["quality_uniform"]["mean"]
    gap = (rq - sq) / (rq + 1e-10) * 100
    agg["kill_criteria"]["K1"] = {
        "desc": "synthetic-only >15% worse than real on uniform eval",
        "synth_q": round(sq, 5), "real_q": round(rq, 5),
        "gap_pct": round(gap, 1), "threshold": 15,
        "killed": gap > 15,
        "verdict": "KILLED" if gap > 15 else "SURVIVES",
    }

    # Mixing sweep aggregation
    for rs in all_results[0]["mixing_sweep"]:
        qu = [r["mixing_sweep"][rs]["q_uniform"] for r in all_results]
        qs = [r["mixing_sweep"][rs]["q_synth"] for r in all_results]
        qr = [r["mixing_sweep"][rs]["q_real"] for r in all_results]
        er = [r["mixing_sweep"][rs]["eff_rank"] for r in all_results]
        agg["mixing_sweep"][rs] = {
            "ratio": float(rs),
            "q_uniform": {"mean": float(np.mean(qu)), "std": float(np.std(qu))},
            "q_synth": {"mean": float(np.mean(qs)), "std": float(np.std(qs))},
            "q_real": {"mean": float(np.mean(qr)), "std": float(np.std(qr))},
            "eff_rank": {"mean": float(np.mean(er)), "std": float(np.std(er))},
            "contamination": all_results[0]["mixing_sweep"][rs]["contamination"],
        }

    # K2: mixed NOT better than either alone (on uniform eval)
    best_pure = max(sq, rq)
    mixed = {k: v["q_uniform"]["mean"] for k, v in agg["mixing_sweep"].items()
             if 0 < float(k) < 1}
    if mixed:
        best_k = max(mixed, key=mixed.get)
        best_v = mixed[best_k]
    else:
        best_k, best_v = "N/A", 0
    imp = (best_v - best_pure) / (best_pure + 1e-10) * 100
    agg["kill_criteria"]["K2"] = {
        "desc": "mixed NOT better than either alone on uniform eval",
        "best_pure": round(best_pure, 5),
        "best_pure_regime": "synthetic" if sq >= rq else "real",
        "best_mixed": round(best_v, 5),
        "best_ratio": best_k,
        "improvement_pct": round(imp, 1),
        "killed": best_v <= best_pure,
        "verdict": "KILLED" if best_v <= best_pure else "SURVIVES",
    }

    # Orthogonality aggregation
    for key in ["within_synthetic", "within_real", "within_mixed",
                "cross_synth_real", "angle_synth", "angle_real", "angle_mixed"]:
        vals = [r["orthogonality"][key] for r in all_results]
        agg["orthogonality"][key] = {"mean": float(np.mean(vals)),
                                      "std": float(np.std(vals))}

    return agg


def print_report(agg):
    print("=" * 74)
    print("SYNTHETIC vs REAL DATA QUALITY: SIMULATION RESULTS")
    print("=" * 74)
    c = agg["config"]
    print(f"\nConfig: d={c['d']}, r={c['r']}, N_train={c['n_train']}, "
          f"steps={c['steps']}, seeds={agg['n_seeds']}")

    # Quality table
    print("\n--- Quality (reconstruction accuracy on held-out data) ---")
    print(f"{'Regime':<11} {'Uniform eval':<20} {'Synth eval':<20} "
          f"{'Real eval':<20} {'Eff. rank':<12}")
    for regime in ["synthetic", "real"]:
        r = agg["regimes"][regime]
        qu = r["quality_uniform"]
        qs = r["quality_synth_eval"]
        qr = r["quality_real_eval"]
        er = r["effective_rank"]
        print(f"{regime:<11} "
              f"{qu['mean']:.4f} +/- {qu['ci95']:.4f}   "
              f"{qs['mean']:.4f} +/- {qs['ci95']:.4f}   "
              f"{qr['mean']:.4f} +/- {qr['ci95']:.4f}   "
              f"{er['mean']:.1f}")

    # Kill criteria
    print("\n--- Kill Criteria ---")
    k1 = agg["kill_criteria"]["K1"]
    print(f"K1: gap = {k1['gap_pct']:.1f}% (synth={k1['synth_q']:.4f}, "
          f"real={k1['real_q']:.4f}, threshold=15%) -> {k1['verdict']}")
    k2 = agg["kill_criteria"]["K2"]
    print(f"K2: best_mixed={k2['best_mixed']:.4f} at ratio={k2['best_ratio']}, "
          f"best_pure={k2['best_pure']:.4f} ({k2['best_pure_regime']}), "
          f"improvement={k2['improvement_pct']:+.1f}% -> {k2['verdict']}")

    # Mixing sweep
    print("\n--- Mixing Ratio Sweep ---")
    print(f"{'Ratio':<7} {'Q(uniform)':<14} {'Q(synth)':<14} "
          f"{'Q(real)':<14} {'Eff.rank':<10} {'Contam%':<8}")
    for rs in sorted(agg["mixing_sweep"], key=float):
        m = agg["mixing_sweep"][rs]
        print(f"{float(rs):<7.1f} "
              f"{m['q_uniform']['mean']:<14.4f} "
              f"{m['q_synth']['mean']:<14.4f} "
              f"{m['q_real']['mean']:<14.4f} "
              f"{m['eff_rank']['mean']:<10.1f} "
              f"{m['contamination']['boost_pct']:<8.2f}")

    # Orthogonality
    print("\n--- Orthogonality ---")
    print(f"{'Group':<22} {'|cos| mean':<14} {'std':<10} {'angle(deg)'}")
    pairs = [("within_synthetic", "angle_synth"),
             ("within_real", "angle_real"),
             ("within_mixed", "angle_mixed")]
    for ck, ak in pairs:
        co = agg["orthogonality"][ck]
        ao = agg["orthogonality"][ak]
        print(f"{ck:<22} {co['mean']:<14.6f} {co['std']:<10.6f} "
              f"{ao['mean']:.2f}")
    co = agg["orthogonality"]["cross_synth_real"]
    print(f"{'cross_synth_real':<22} {co['mean']:<14.6f} {co['std']:<10.6f}")

    # Contamination
    print("\n--- Contamination Risk ---")
    for regime in ["synthetic", "real"]:
        ct = agg["regimes"][regime]["contamination"]
        print(f"{regime}: overlap={ct['expected_overlap']}, "
              f"P(any)={ct['p_any']}, boost={ct['boost_pct']}%")

    # Key findings
    print("\n" + "=" * 74)
    print("KEY FINDINGS")
    print("=" * 74)

    sq = agg["regimes"]["synthetic"]["quality_uniform"]["mean"]
    rq = agg["regimes"]["real"]["quality_uniform"]["mean"]
    er_s = agg["regimes"]["synthetic"]["effective_rank"]["mean"]
    er_r = agg["regimes"]["real"]["effective_rank"]["mean"]

    if sq > rq:
        print(f"1. QUALITY: Synthetic beats real on uniform eval "
              f"({sq:.4f} vs {rq:.4f})")
    else:
        gap = (rq - sq) / (rq + 1e-10) * 100
        print(f"1. QUALITY: Real beats synthetic on uniform eval by {gap:.1f}% "
              f"({rq:.4f} vs {sq:.4f})")

    # Within-distribution quality
    sq_s = agg["regimes"]["synthetic"]["quality_synth_eval"]["mean"]
    rq_r = agg["regimes"]["real"]["quality_real_eval"]["mean"]
    print(f"   Within-distribution: synth-on-synth={sq_s:.4f}, "
          f"real-on-real={rq_r:.4f}")

    print(f"2. DIVERSITY: Real eff.rank={er_r:.1f} vs synthetic={er_s:.1f} "
          f"({er_r/er_s:.1f}x)")

    cs = agg["orthogonality"]["within_synthetic"]["mean"]
    cr = agg["orthogonality"]["within_real"]["mean"]
    if cs < cr:
        print(f"3. ORTHOGONALITY: Synthetic experts MORE orthogonal "
              f"(|cos|={cs:.5f} vs {cr:.5f})")
    else:
        print(f"3. ORTHOGONALITY: Real experts MORE orthogonal "
              f"(|cos|={cr:.5f} vs {cs:.5f})")

    ct_s = agg["regimes"]["synthetic"]["contamination"]
    ct_r = agg["regimes"]["real"]["contamination"]
    print(f"4. CONTAMINATION: Synthetic {ct_s['boost_pct']/ct_r['boost_pct']:.0f}x "
          f"higher risk ({ct_s['boost_pct']}% vs {ct_r['boost_pct']}%)")

    k2 = agg["kill_criteria"]["K2"]
    if not k2["killed"]:
        print(f"5. MIXING: Best at {k2['best_ratio']} synthetic "
              f"({k2['improvement_pct']:+.1f}% over best pure)")
    else:
        print(f"5. MIXING: No ratio beats pure {k2['best_pure_regime']} "
              f"(best at {k2['best_ratio']}, {k2['improvement_pct']:+.1f}%)")


def main():
    print("Running synthetic vs real data quality simulation...")
    all_results = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        all_results.append(run_seed(seed))
        print("done")

    agg = aggregate(all_results)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved to {out}")

    print_report(agg)
    return agg


if __name__ == "__main__":
    main()
