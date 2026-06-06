#!/usr/bin/env python3
"""
Followup: Crystallize + flywheel with heterogeneous real-user adapters.

MATH: experiments/models/exp_followup_m2p_crystallize_real_users/MATH.md

Tests K1564: mean cos(B_crystal, B_D*) >= 0.95 under heterogeneous (LR, steps,
seed) user construction — NOT the iid B* + ε_u tautology used in parent
exp_p1_t6_crystallize_domain.

Parent experiment's synthetic-by-construction: every user had identical σ, zero
drift → averaging trivially returns to B*. This experiment constructs each user
as B_u = convergence_u · B* + (1-convergence_u) · drift_u + η_u where convergence
depends on LR and steps per user — giving non-zero μ̄.

Infrastructure note: parent adapter weights
(exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors
and exp_p1_t2_multi_domain_5/adapters/{legal,finance}/adapters.safetensors) are
gitignored and not on disk. If REAL_ADAPTERS_AVAILABLE=False, we draw B* synthetic
with the parent-reported norm/std, generate heterogeneous users, and flag the
limitation in the verdict. This is the actual on-disk state as of 2026-04-19.
"""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Adapter paths (same as parent T6.2)
T21 = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training/adapters"
T26 = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5/adapters"

ADAPTER_PATHS = {
    "math":    T21 / "math/adapters.safetensors",
    "code":    T21 / "code/adapters.safetensors",
    "medical": T21 / "medical/adapters.safetensors",
    "legal":   T26 / "legal/adapters.safetensors",
    "finance": T26 / "finance/adapters.safetensors",
}

# Parent-reported statistics (from exp_p1_t6_crystallize_domain MATH.md quantitative
# derivations section). Used as fallback when real adapter weights are absent.
PARENT_STATS = {
    "math":    {"norm": 5.7618, "std": 0.007425, "dim": 602_112},
    "code":    {"norm": 5.81,   "std": 0.00745,  "dim": 602_112},
    "medical": {"norm": 5.73,   "std": 0.00738,  "dim": 602_112},
    "legal":   {"norm": 5.76,   "std": 0.00742,  "dim": 602_112},
    "finance": {"norm": 5.78,   "std": 0.00744,  "dim": 602_112},
}

USERS_PER_DOMAIN = 2 if IS_SMOKE else 5
RANDOM_SEED = 42
K_THRESHOLD = 0.95
RANK_R = 8  # Gemma-4 parent LoRA rank

# Heterogeneity distribution parameters
LR_MIN, LR_MAX = 1e-5, 1e-3
STEPS_CHOICES = (100, 200, 400, 800, 1600)
TAU = 1.0  # convergence timescale
SIGMA_SCALE_EXP = 0.3  # σ_u ∝ (lr_u / 1e-4) ** 0.3
SIGMA_FRAC_BASE = 0.5  # base multiplier on std(B*)


# ─────────────────────────────────────────────────────────────────────
# Phase 0: Load or synthesize canonical B*
# ─────────────────────────────────────────────────────────────────────

def load_or_synthesize_canonical(rng: np.random.Generator) -> tuple[dict, bool]:
    """Return dict[domain → B*_vector_float32]. Real if available, else synthetic."""
    vectors = {}
    real_available = all(p.exists() for p in ADAPTER_PATHS.values())

    if real_available:
        print("Phase 0: loading REAL canonical adapters from disk", flush=True)
        for name, path in ADAPTER_PATHS.items():
            weights = mx.load(str(path))
            b_keys = sorted(k for k in weights.keys() if "lora_b" in k)
            if not b_keys:
                raise RuntimeError(f"No lora_b keys in {path}")
            b_parts = [weights[k].flatten() for k in b_keys]
            b_flat = mx.concatenate(b_parts)
            mx.eval(b_flat)
            v = np.array(b_flat, dtype=np.float32)
            vectors[name] = v
            del weights, b_flat
        mx.clear_cache()
    else:
        print("Phase 0: real adapters NOT on disk — synthesizing B* from parent stats", flush=True)
        for name, stats in PARENT_STATS.items():
            d = stats["dim"]
            norm_target = stats["norm"]
            # Draw a random direction and scale to parent-reported norm.
            raw = rng.standard_normal(d).astype(np.float32)
            raw *= (norm_target / np.linalg.norm(raw))
            vectors[name] = raw
    return vectors, real_available


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Generate heterogeneous users per domain
# ─────────────────────────────────────────────────────────────────────

def sample_user_hparams(n_users: int, rng: np.random.Generator) -> list[dict]:
    lr = np.exp(rng.uniform(np.log(LR_MIN), np.log(LR_MAX), size=n_users))
    steps = rng.choice(STEPS_CHOICES, size=n_users)
    seeds = rng.integers(0, 2**31 - 1, size=n_users)
    return [
        {"lr": float(lr[i]), "steps": int(steps[i]), "seed": int(seeds[i])}
        for i in range(n_users)
    ]


def convergence_factor(lr: float, steps: int) -> float:
    # Calibrated so (lr=1e-4, steps=400) → ~0.96
    # 1 - exp(-lr * steps / tau_scaled). Choose tau such that 1e-4*400/tau = 3.2
    tau_scaled = 1e-4 * 400 / 3.2
    return float(1.0 - np.exp(-lr * steps / tau_scaled))


def generate_heterogeneous_users(
    canonical: dict,
    n_users: int,
    rng: np.random.Generator,
) -> dict:
    """Returns dict[domain → {users:[vec...], hparams:[...], stats:...}]."""
    out = {}
    for name, B_star in canonical.items():
        d = B_star.shape[0]
        norm_star = float(np.linalg.norm(B_star))
        std_star = float(B_star.std())
        hparams = sample_user_hparams(n_users, rng)
        users = []
        for h in hparams:
            u_rng = np.random.default_rng(h["seed"])
            conv = convergence_factor(h["lr"], h["steps"])
            sigma_u = SIGMA_FRAC_BASE * std_star * (h["lr"] / 1e-4) ** SIGMA_SCALE_EXP
            # bias_drift: random unit direction scaled to ||B*|| (represents where
            # under-converged/overshoot users end up — NOT aligned to B*)
            bias_raw = u_rng.standard_normal(d).astype(np.float32)
            bias_drift = bias_raw * (norm_star / np.linalg.norm(bias_raw))
            # rank-r projection is implicit because LoRA updates live in low-rank;
            # we approximate rank-structure by scaling bias_drift by sqrt(RANK_R/d)
            # for the directional component that exits B*'s subspace.
            # For simplicity (and conservatism), keep full-rank bias.
            eta = u_rng.standard_normal(d).astype(np.float32) * sigma_u
            B_u = conv * B_star + (1.0 - conv) * bias_drift + eta
            users.append(B_u)
        out[name] = {
            "B_star": B_star,
            "users": users,
            "hparams": [
                {**h, "convergence": convergence_factor(h["lr"], h["steps"])}
                for h in hparams
            ],
            "norm_star": norm_star,
            "std_star": std_star,
            "dim": d,
            "sigma_u": [
                SIGMA_FRAC_BASE * std_star * (h["lr"] / 1e-4) ** SIGMA_SCALE_EXP
                for h in hparams
            ],
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Crystallize and measure
# ─────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def crystallize_and_measure(user_data: dict) -> dict:
    print("Phase 2: crystallize and compute cosines", flush=True)
    rows = []
    per_domain = {}
    for name, bundle in user_data.items():
        B_star = bundle["B_star"]
        users = bundle["users"]
        # Crystal = mean of users
        B_crystal = np.mean(np.stack(users, axis=0), axis=0)
        # μ̄ = B_crystal - B*
        mu_bar = B_crystal - B_star
        cos_crystal = cosine(B_crystal, B_star)
        cos_users = [cosine(u, B_star) for u in users]
        mean_cos_users = float(np.mean(cos_users))
        crystal_gain = cos_crystal - mean_cos_users
        mu_bar_norm = float(np.linalg.norm(mu_bar))
        mu_bar_norm_frac = mu_bar_norm / bundle["norm_star"]
        sigma_spread = (max(bundle["sigma_u"]) / (min(bundle["sigma_u"]) + 1e-12))
        per_domain[name] = {
            "cos_crystal": cos_crystal,
            "mean_cos_users": mean_cos_users,
            "crystal_gain": crystal_gain,
            "mu_bar_norm": mu_bar_norm,
            "mu_bar_norm_frac": mu_bar_norm_frac,
            "sigma_spread": sigma_spread,
            "per_user_cos": cos_users,
            "convergences": [h["convergence"] for h in bundle["hparams"]],
            "lrs": [h["lr"] for h in bundle["hparams"]],
            "steps": [h["steps"] for h in bundle["hparams"]],
        }
        rows.append(cos_crystal)
    mean_cos_crystal = float(np.mean(rows))
    return {
        "per_domain": per_domain,
        "mean_cos_crystal_across_domains": mean_cos_crystal,
        "per_domain_cos": rows,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    canonical, real_available = load_or_synthesize_canonical(rng)

    user_data = generate_heterogeneous_users(canonical, USERS_PER_DOMAIN, rng)

    measurements = crystallize_and_measure(user_data)

    mean_cos = measurements["mean_cos_crystal_across_domains"]

    # K1564: pre-registered KC. Threshold 0.95. No relaxation allowed.
    k1564_pass = mean_cos >= K_THRESHOLD

    # K_vacate clause: if real adapters absent, the test is *heterogeneous-synthetic*,
    # not heterogeneous-real. The KC result still stands for the simulation — we do
    # NOT silently relax the threshold. But we record the limitation explicitly so
    # the reviewer can distinguish "killed under heterogeneous synthetic" from
    # "killed under real users" vs. "K_vacate for lack of real users".
    k_vacate_reason = (
        None if real_available else
        "Real parent adapters not on disk (infra blocker #2: "
        "exp_p1_t2_single_domain_training adapters gitignored). "
        "Test run against heterogeneous SYNTHETIC users only."
    )

    verdict = "PASS" if k1564_pass else "KILLED"
    all_pass = k1564_pass

    results = {
        "experiment": "exp_followup_m2p_crystallize_real_users",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": IS_SMOKE,
        "real_adapters_available": real_available,
        "k_vacate_reason": k_vacate_reason,
        "kill_criteria": [
            {
                "id": 1564,
                "text": (
                    "Flywheel crystal cos(crystal, B*) >= 0.95 with heterogeneous "
                    "LR/steps/seeds (not 5-canonical+epsilon)"
                ),
                "threshold": K_THRESHOLD,
                "measured_mean_cos": mean_cos,
                "result": "pass" if k1564_pass else "fail",
            }
        ],
        "measurements": measurements,
        "config": {
            "users_per_domain": USERS_PER_DOMAIN,
            "random_seed": RANDOM_SEED,
            "lr_range": [LR_MIN, LR_MAX],
            "steps_choices": list(STEPS_CHOICES),
            "tau": TAU,
            "sigma_scale_exp": SIGMA_SCALE_EXP,
            "sigma_frac_base": SIGMA_FRAC_BASE,
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n=== RESULT: {verdict} ===", flush=True)
    print(f"mean cos(crystal, B*) = {mean_cos:.4f} (threshold {K_THRESHOLD})", flush=True)
    if k_vacate_reason:
        print(f"K_vacate-note: {k_vacate_reason}", flush=True)
    for name, m in measurements["per_domain"].items():
        print(
            f"  {name}: cos_crystal={m['cos_crystal']:.4f} "
            f"mean_user={m['mean_cos_users']:.4f} "
            f"||μ̄||/||B*||={m['mu_bar_norm_frac']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
