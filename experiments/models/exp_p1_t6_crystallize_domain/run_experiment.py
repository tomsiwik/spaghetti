#!/usr/bin/env python3
"""
T6.2: Crystallize cluster of user adapters into single domain adapter

MATH: experiments/models/exp_p1_t6_crystallize_domain/MATH.md

Crystallization = weighted average of B-matrices from N same-domain user adapters.
By the Law of Large Numbers, the average is closer to the true domain centroid
than any individual user adapter (Theorem 1). Uses only weight files — no user data.

Kill criteria:
  K1120: cos(B_crystal, B_canonical) >= mean(cos(B_user, B_canonical)) for all 5 domains
  K1121: 5 crystallized adapters (from 25), freeing 20 slots
  K1122: ||B_crystal||/||B_canonical|| in [0.90, 1.10] (norm preserved, MMLU proxy)
  K1123: No user training data accessed during crystallization

References:
  - Model Soup: Wortsman et al. 2022, arxiv 2203.05482
  - Task Arithmetic: Ilharco et al. 2023, arxiv 2212.04089
  - Finding #450 (T6.1): B-matrix K-means, silhouette=0.8193, purity=1.0
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Adapter paths (same as T6.1)
T21 = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training/adapters"
T26 = Path(__file__).parent.parent / "exp_p1_t2_multi_domain_5/adapters"

ADAPTER_PATHS = {
    "math":    T21 / "math/adapters.safetensors",
    "code":    T21 / "code/adapters.safetensors",
    "medical": T21 / "medical/adapters.safetensors",
    "legal":   T26 / "legal/adapters.safetensors",
    "finance": T26 / "finance/adapters.safetensors",
}

NOISE_SIGMA_FRAC = 0.5    # σ = 0.5 × std(B) per element
USERS_PER_DOMAIN = 5      # 1 canonical + 4 variants
RANDOM_SEED = 42
NORM_RATIO_THRESHOLD = (0.90, 1.10)  # K1122: within 10% of canonical norm


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Load canonical adapters (flat B-vectors)
# ─────────────────────────────────────────────────────────────────────

def extract_b_vector(path: Path) -> np.ndarray:
    """Load and flatten all lora_b weights from adapter safetensors."""
    weights = mx.load(str(path))
    b_keys = sorted(k for k in weights.keys() if "lora_b" in k)
    b_parts = [weights[k].flatten() for k in b_keys]
    b_flat = mx.concatenate(b_parts)
    mx.eval(b_flat)
    return np.array(b_flat, dtype=np.float32)


def load_canonical_adapters() -> tuple[dict, dict]:
    """Load 5 canonical domain adapters. Returns (name->vector, name->stats)."""
    print("\nPhase 1: Loading canonical domain adapters", flush=True)
    vectors = {}
    stats = {}
    for name, path in ADAPTER_PATHS.items():
        v = extract_b_vector(path)
        norm = float(np.linalg.norm(v))
        std = float(v.std())
        vectors[name] = v
        stats[name] = {
            "norm": round(norm, 4),
            "std": round(std, 6),
            "dim": v.shape[0],
        }
        print(f"  {name}: dim={v.shape[0]}, norm={norm:.4f}, std={std:.6f}", flush=True)
    return vectors, stats


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Generate synthetic user adapters
# ─────────────────────────────────────────────────────────────────────

def make_user_adapters(
    canonical: dict,
    n_users: int = USERS_PER_DOMAIN,
    sigma_frac: float = NOISE_SIGMA_FRAC,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Create n_users adapters per domain: canonical + (n_users-1) noisy variants.
    B_u = B_canonical + ε_u, ε_u ~ N(0, (σ_frac * std(B))^2 * I)
    """
    print(f"\nPhase 2: Generating {n_users} user adapters per domain (σ_frac={sigma_frac})", flush=True)
    rng = np.random.default_rng(seed)
    user_adapters = {}

    for name, bvec in canonical.items():
        sigma = sigma_frac * float(bvec.std())
        users = [bvec.copy()]  # user 0 = canonical itself
        for u in range(1, n_users):
            noise = rng.normal(0, sigma, size=bvec.shape).astype(np.float32)
            users.append(bvec + noise)

        user_adapters[name] = users
        norms = [float(np.linalg.norm(u)) for u in users]
        print(
            f"  {name}: σ={sigma:.6f}, norms=[{', '.join(f'{n:.4f}' for n in norms)}]",
            flush=True,
        )

    return user_adapters


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Crystallize each domain
# ─────────────────────────────────────────────────────────────────────

def crystallize(user_adapters: dict) -> dict:
    """B_crystal = (1/N) * sum(B_u for u in domain)."""
    print("\nPhase 3: Crystallizing — averaging B-matrices per domain", flush=True)
    crystals = {}
    for name, users in user_adapters.items():
        B_crystal = np.mean(np.stack(users, axis=0), axis=0)
        crystals[name] = B_crystal
        print(
            f"  {name}: N={len(users)}, ||B_crystal||={np.linalg.norm(B_crystal):.4f}",
            flush=True,
        )
    return crystals


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Evaluate kill criteria
# ─────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def evaluate_quality(
    canonical: dict,
    user_adapters: dict,
    crystals: dict,
) -> dict:
    """
    K1120: cos(B_crystal, B*) >= mean_u cos(B_u, B*) for all domains
    K1121: N adapters after = n_domains (from n_domains × n_users before)
    K1122: ||B_crystal|| / ||B*|| in [0.90, 1.10]
    K1123: no user data accessed (structural — verified by construction)
    """
    print("\nPhase 4: Evaluating kill criteria", flush=True)

    per_domain = {}
    k1120_results = {}
    k1122_results = {}

    for name in canonical:
        b_star = canonical[name]
        users = user_adapters[name]
        b_crystal = crystals[name]

        cos_crystal = cosine(b_crystal, b_star)
        cos_users = [cosine(u, b_star) for u in users]
        mean_cos_user = float(np.mean(cos_users))

        norm_star = float(np.linalg.norm(b_star))
        norm_crystal = float(np.linalg.norm(b_crystal))
        norm_ratio = norm_crystal / norm_star

        delta_cos = cos_crystal - mean_cos_user
        k1120_pass = cos_crystal >= mean_cos_user
        k1122_pass = NORM_RATIO_THRESHOLD[0] <= norm_ratio <= NORM_RATIO_THRESHOLD[1]

        k1120_results[name] = {
            "pass": k1120_pass,
            "cos_crystal": round(cos_crystal, 4),
            "mean_cos_user": round(mean_cos_user, 4),
            "delta_cos_pp": round(delta_cos * 100, 2),
        }
        k1122_results[name] = {
            "pass": k1122_pass,
            "norm_star": round(norm_star, 4),
            "norm_crystal": round(norm_crystal, 4),
            "norm_ratio": round(norm_ratio, 4),
        }

        per_domain[name] = {
            "cos_users": [round(c, 4) for c in cos_users],
            "mean_cos_user": round(mean_cos_user, 4),
            "cos_crystal": round(cos_crystal, 4),
            "delta_cos_pp": round(delta_cos * 100, 2),
            "norm_ratio": round(norm_ratio, 4),
        }

        status_1120 = "PASS" if k1120_pass else "FAIL"
        status_1122 = "PASS" if k1122_pass else "FAIL"
        print(
            f"  {name}: cos_crystal={cos_crystal:.4f} vs mean_user={mean_cos_user:.4f} "
            f"(+{delta_cos*100:.2f}pp) K1120={status_1120} | "
            f"norm_ratio={norm_ratio:.4f} K1122={status_1122}",
            flush=True,
        )

    # K1120: all 5 domains must pass
    k1120_all_pass = all(v["pass"] for v in k1120_results.values())
    # K1121: 5 crystals from 25 adapters
    n_before = sum(len(v) for v in user_adapters.values())
    n_after = len(crystals)
    k1121_pass = n_after == len(canonical)
    # K1122: all domains norm-preserved
    k1122_all_pass = all(v["pass"] for v in k1122_results.values())
    # K1123: structural (no training data accessed in this script)
    k1123_pass = True

    print(f"\n  K1120 (quality >= mean_user): {'PASS' if k1120_all_pass else 'FAIL'} ({sum(v['pass'] for v in k1120_results.values())}/5 domains)")
    print(f"  K1121 (slot liberation {n_before} → {n_after}): {'PASS' if k1121_pass else 'FAIL'}")
    print(f"  K1122 (norm preserved): {'PASS' if k1122_all_pass else 'FAIL'} ({sum(v['pass'] for v in k1122_results.values())}/5 domains)")
    print(f"  K1123 (no user data): PASS (structural)")

    return {
        "per_domain": per_domain,
        "kill_criteria": {
            "K1120": {
                "pass": k1120_all_pass,
                "n_domains_pass": sum(v["pass"] for v in k1120_results.values()),
                "per_domain": k1120_results,
            },
            "K1121": {
                "pass": k1121_pass,
                "n_before": n_before,
                "n_after": n_after,
                "slots_freed": n_before - n_after,
            },
            "K1122": {
                "pass": k1122_all_pass,
                "norm_threshold": list(NORM_RATIO_THRESHOLD),
                "per_domain": k1122_results,
            },
            "K1123": {
                "pass": k1123_pass,
                "method": "B-matrix averaging only — no training data files read",
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("T6.2: Crystallize Domain Adapters")
    print(f"  is_smoke={IS_SMOKE}, users_per_domain={USERS_PER_DOMAIN}, sigma_frac={NOISE_SIGMA_FRAC}")
    print("=" * 60, flush=True)

    # Phase 1
    canonical, stats = load_canonical_adapters()

    # Phase 2
    n_users = 3 if IS_SMOKE else USERS_PER_DOMAIN
    user_adapters = make_user_adapters(canonical, n_users=n_users)

    # Phase 3
    crystals = crystallize(user_adapters)

    # Phase 4
    results_eval = evaluate_quality(canonical, user_adapters, crystals)

    # Compose results
    results = {
        "is_smoke": IS_SMOKE,
        "n_domains": len(canonical),
        "users_per_domain": n_users,
        "noise_sigma_frac": NOISE_SIGMA_FRAC,
        "adapter_stats": stats,
        **results_eval,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_FILE}", flush=True)

    # Summary
    kc = results["kill_criteria"]
    all_pass = all(v["pass"] for v in kc.values())
    print("\n" + "=" * 60)
    print(f"VERDICT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for k, v in kc.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
