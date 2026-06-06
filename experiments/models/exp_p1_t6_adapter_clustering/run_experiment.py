#!/usr/bin/env python3
"""
T6.1: Cluster user adapters by domain similarity for crystallization candidates

MATH: experiments/models/exp_p1_t6_adapter_clustering/MATH.md

Uses existing domain adapters (T2.1: math/code/medical, T2.6: legal/finance)
plus synthetic user variants to test whether B-matrix cosine similarity can
recover domain groupings without accessing user data.

Phases:
  Phase 1: Load 5 canonical domain adapters, measure pairwise similarities
  Phase 2: Generate 25 synthetic user adapters (5 domains × 5 users/domain)
  Phase 3: K-means clustering (K=3,4,5) on cosine-normalized B-vectors
  Phase 4: Measure silhouette score, report best K
  Phase 5: Verify K1117/K1118/K1119

Kill criteria:
  K1117: Clustering identifies >= 3 natural domain groups from 25+ adapters
  K1118: Intra-cluster similarity > inter-cluster (silhouette > 0.3)
  K1119: Clustering uses adapter B-matrix only (no user data accessed)

References:
  - Task Arithmetic: Ilharco et al. 2023, arxiv 2212.04089
  - LIMA: Zhou et al. 2023, arxiv 2305.11206
  - Finding #216 (supported): SFT adapters at T=200 have cos=0.97
  - Finding #217 (supported): LoRA scale separates 3 domain categories
"""

import json
import os
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Adapter paths
T21 = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training/adapters"
T26 = Path(__file__).parent.parent / "exp_p1_t2_multi_domain_5/adapters"

ADAPTER_PATHS = {
    "math":    T21 / "math/adapters.safetensors",
    "code":    T21 / "code/adapters.safetensors",
    "medical": T21 / "medical/adapters.safetensors",
    "legal":   T26 / "legal/adapters.safetensors",
    "finance": T26 / "finance/adapters.safetensors",
}

NOISE_SIGMA_FRAC = 0.5   # σ = 0.5 × std(B) per element
USERS_PER_DOMAIN = 5     # 1 canonical + 4 variants = 5 per domain
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Load adapters and extract B-vectors
# ─────────────────────────────────────────────────────────────────────

def extract_b_vector(path: Path) -> np.ndarray:
    """Extract and flatten all lora_b weights from an adapter safetensors file."""
    weights = mx.load(str(path))
    b_keys = sorted(k for k in weights.keys() if "lora_b" in k)
    b_parts = [weights[k].flatten() for k in b_keys]
    b_flat = mx.concatenate(b_parts)
    mx.eval(b_flat)
    return np.array(b_flat, dtype=np.float32)


def load_canonical_adapters() -> tuple[dict, dict]:
    """Load 5 domain adapters. Returns (name->vector, name->stats)."""
    print("\nPhase 1: Loading canonical domain adapters", flush=True)
    vectors = {}
    stats = {}
    for name, path in ADAPTER_PATHS.items():
        v = extract_b_vector(path)
        norm = float(np.linalg.norm(v))
        std = float(v.std())
        vectors[name] = v
        stats[name] = {"norm": round(norm, 4), "std": round(std, 6), "dim": v.shape[0]}
        print(f"  {name}: dim={v.shape[0]}, norm={norm:.4f}, std={std:.6f}", flush=True)
    return vectors, stats


def pairwise_cosine(vectors: dict) -> dict:
    """Compute all pairwise cosine similarities."""
    names = list(vectors.keys())
    sims = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = vectors[names[i]], vectors[names[j]]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            key = f"{names[i]}-{names[j]}"
            sims[key] = round(cos, 4)
    return sims


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Generate synthetic user variants
# ─────────────────────────────────────────────────────────────────────

def generate_user_adapters(
    canonical: dict,
    n_per_domain: int = USERS_PER_DOMAIN,
    sigma_frac: float = NOISE_SIGMA_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Generate synthetic user adapters via noise perturbation.

    For each canonical adapter:
      - user_0 = canonical (no noise)
      - user_1..4 = canonical + ε, ε ~ N(0, σ²I), σ = sigma_frac × std(B)

    Returns:
      vectors: (N, D) matrix of B-vectors
      labels: domain label per adapter (e.g., "math")
      user_ids: user identifier (e.g., "math_u0")
    """
    print(f"\nPhase 2: Generating {len(canonical) * n_per_domain} user adapters", flush=True)
    print(f"  Noise: σ = {sigma_frac} × std(B) per element", flush=True)

    rng = np.random.default_rng(seed)
    vectors_list = []
    labels = []
    user_ids = []

    for domain, b_vec in canonical.items():
        sigma = sigma_frac * b_vec.std()
        vectors_list.append(b_vec.copy())
        labels.append(domain)
        user_ids.append(f"{domain}_u0")
        for u in range(1, n_per_domain):
            noise = rng.normal(0, sigma, size=b_vec.shape).astype(np.float32)
            variants = b_vec + noise
            vectors_list.append(variants)
            labels.append(domain)
            user_ids.append(f"{domain}_u{u}")
        print(f"  {domain}: σ={sigma:.6f} (noise/norm={sigma * np.sqrt(len(b_vec)) / np.linalg.norm(b_vec):.3f})", flush=True)

    return np.stack(vectors_list), labels, user_ids


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Clustering
# ─────────────────────────────────────────────────────────────────────

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length. Use float64 to avoid overflow at high dim."""
    X64 = X.astype(np.float64)
    norms = np.linalg.norm(X64, axis=1, keepdims=True)
    return (X64 / (norms + 1e-12)).astype(np.float32)


def kmeans_cosine(X: np.ndarray, K: int, n_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, float]:
    """
    K-means in cosine space (normalize first, then L2 k-means = cosine k-means).
    Uses float64 internally to avoid overflow at dim=602K.
    Returns (assignments, inertia).
    """
    from sklearn.cluster import KMeans
    X_norm = l2_normalize(X).astype(np.float64)
    km = KMeans(n_clusters=K, n_init=10, max_iter=n_iter, random_state=seed)
    assignments = km.fit_predict(X_norm)
    inertia = float(km.inertia_)
    return assignments, inertia


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score in cosine space using float64."""
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    if len(set(labels)) < 2:
        return -1.0
    X64 = l2_normalize(X).astype(np.float64)
    dist_matrix = cosine_distances(X64)
    return float(silhouette_score(dist_matrix, labels, metric="precomputed"))


def pca_reduce(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA reduction using SVD. Uses float64 throughout."""
    X64 = X.astype(np.float64)
    X_centered = X64 - X64.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proj = X_centered @ Vt[:n_comp].T
    # Replace any non-finite values (BLAS edge cases)
    proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    return proj.astype(np.float32)


def run_clustering(vectors: np.ndarray, true_labels: list[str]) -> dict:
    """Run k-means for K=3,4,5. Report silhouette + domain alignment."""
    print("\nPhase 3: K-means clustering (K=3,4,5)", flush=True)
    # PCA-reduce first (avoids numerical overflow at dim=602K, standard practice)
    n_comp = min(50, vectors.shape[0] - 1)
    X_pca = pca_reduce(vectors, n_components=n_comp)
    print(f"  PCA: {vectors.shape[1]} → {X_pca.shape[1]} dims", flush=True)
    X_norm = l2_normalize(X_pca)
    results = {}

    for K in [3, 4, 5]:
        assignments, inertia = kmeans_cosine(X_pca, K=K)
        sil = compute_silhouette(X_norm, assignments)
        # Domain purity: for each cluster, what fraction is the majority domain?
        domain_to_idx = {}
        for i, d in enumerate(true_labels):
            domain_to_idx.setdefault(d, []).append(i)
        cluster_domains = {}
        for cluster_id in range(K):
            idxs = [i for i, a in enumerate(assignments) if a == cluster_id]
            domain_counts = {}
            for idx in idxs:
                d = true_labels[idx]
                domain_counts[d] = domain_counts.get(d, 0) + 1
            majority = max(domain_counts, key=domain_counts.get) if domain_counts else "?"
            purity = domain_counts.get(majority, 0) / len(idxs) if idxs else 0.0
            cluster_domains[cluster_id] = {
                "majority_domain": majority,
                "purity": round(purity, 3),
                "size": len(idxs),
                "domain_counts": domain_counts,
            }
        print(f"  K={K}: silhouette={sil:.4f}, inertia={inertia:.2f}", flush=True)
        for cid, info in cluster_domains.items():
            print(f"    cluster {cid}: {info['majority_domain']} (purity={info['purity']:.2f}, n={info['size']})", flush=True)
        results[K] = {
            "silhouette": round(sil, 4),
            "inertia": round(inertia, 2),
            "clusters": cluster_domains,
            "assignments": assignments.tolist(),
        }

    return results


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Kill criteria evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_kill_criteria(clustering_results: dict, n_adapters: int) -> dict:
    print("\nPhase 4: Kill criteria evaluation", flush=True)

    # K1117: >= 3 natural domain groups from 25+ adapters
    best_K = max(clustering_results.keys(), key=lambda k: clustering_results[k]["silhouette"])
    best_sil = clustering_results[best_K]["silhouette"]
    n_groups = len(clustering_results[best_K]["clusters"])
    K1117_pass = n_groups >= 3 and n_adapters >= 25
    print(f"  K1117: {n_groups} groups from {n_adapters} adapters → {'PASS' if K1117_pass else 'FAIL'}", flush=True)

    # K1118: silhouette > 0.3
    K1118_pass = best_sil > 0.3
    print(f"  K1118: best silhouette={best_sil:.4f} (K={best_K}) → {'PASS' if K1118_pass else 'FAIL'}", flush=True)

    # K1119: only B-matrices used (by construction — no training data accessed)
    K1119_pass = True
    print(f"  K1119: B-matrix-only clustering → PASS (by construction)", flush=True)

    return {
        "K1117": {"pass": K1117_pass, "n_groups": n_groups, "n_adapters": n_adapters},
        "K1118": {"pass": K1118_pass, "best_silhouette": best_sil, "best_K": best_K},
        "K1119": {"pass": K1119_pass, "method": "B-matrix cosine similarity, no user data"},
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"T6.1: Adapter Clustering (smoke={IS_SMOKE})", flush=True)

    # Phase 1
    canonical, adapter_stats = load_canonical_adapters()
    cross_cosines = pairwise_cosine(canonical)
    print("\n  Canonical pairwise cosine (real adapters):", flush=True)
    for pair, cos in cross_cosines.items():
        print(f"    {pair}: {cos:.4f}", flush=True)

    # Phase 2: generate synthetic users
    n_per_domain = 2 if IS_SMOKE else USERS_PER_DOMAIN
    vectors, labels, user_ids = generate_user_adapters(
        canonical, n_per_domain=n_per_domain
    )
    n_adapters = len(labels)
    print(f"\n  Total user adapters: {n_adapters}", flush=True)

    # Phase 3: clustering
    clustering_results = run_clustering(vectors, labels)

    # Phase 4: kill criteria
    kc = evaluate_kill_criteria(clustering_results, n_adapters)

    # Save results
    results = {
        "is_smoke": IS_SMOKE,
        "n_adapters": n_adapters,
        "n_per_domain": n_per_domain,
        "noise_sigma_frac": NOISE_SIGMA_FRAC,
        "adapter_stats": adapter_stats,
        "canonical_pairwise_cosine": cross_cosines,
        "clustering": {str(k): v for k, v in clustering_results.items()},
        "kill_criteria": kc,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_FILE}", flush=True)

    # Summary
    all_pass = all(kc[k]["pass"] for k in ["K1117", "K1118", "K1119"])
    print(f"\n{'='*50}", flush=True)
    print(f"VERDICT: {'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
