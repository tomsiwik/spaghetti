#!/usr/bin/env python3
"""Room Model: adapter gradient ∇H as free routing signal.

Tests whether the spatial gradient of adapter B-matrices (like edge detection
on a heightmap) correlates with domain similarity.

Kill criteria:
  K825: Gradient similarity uncorrelated with behavioral similarity (r < 0.3)
"""

import gc, json, math, os, time, itertools
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

DOMAINS = ["medical", "code", "math", "legal", "finance"]
MAX_SEQ = 256
SEED = 42


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)


def compute_adapter_gradient(adapter_b):
    """Compute discrete gradient of adapter B-matrices.

    Treats each B-matrix as a 2D image and computes Sobel-like gradients.
    Returns the gradient magnitude per key.
    """
    gradients = {}
    for key, val in adapter_b.items():
        B = np.array(val)
        if B.ndim != 2: continue
        # Discrete gradient: difference between adjacent rows and columns
        grad_row = np.diff(B, axis=0)  # (r-1, out)
        grad_col = np.diff(B, axis=1)  # (r, out-1)
        # Gradient magnitude (average)
        mag = (np.mean(np.abs(grad_row)) + np.mean(np.abs(grad_col))) / 2
        gradients[key] = {
            "magnitude": float(mag),
            "row_grad_norm": float(np.linalg.norm(grad_row)),
            "col_grad_norm": float(np.linalg.norm(grad_col)),
        }
    return gradients


def gradient_similarity(grad_a, grad_b):
    """Compute similarity between two adapter gradient profiles."""
    common_keys = set(grad_a.keys()) & set(grad_b.keys())
    if not common_keys: return 0.0

    mags_a = [grad_a[k]["magnitude"] for k in common_keys]
    mags_b = [grad_b[k]["magnitude"] for k in common_keys]

    a = np.array(mags_a)
    b = np.array(mags_b)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def behavioral_similarity(domain_a, domain_b):
    """Heuristic behavioral similarity based on known domain relationships.

    From our experiments: medical-legal are somewhat related, code-math are related,
    finance-legal are related. Code-medical are unrelated.
    """
    # Based on cross-domain PPL experiments and Finding #186
    similarity_matrix = {
        ("medical", "legal"): 0.3,
        ("medical", "finance"): 0.2,
        ("code", "math"): 0.5,
        ("legal", "finance"): 0.4,
        ("medical", "code"): 0.1,
        ("medical", "math"): 0.2,
        ("code", "legal"): 0.1,
        ("code", "finance"): 0.1,
        ("math", "legal"): 0.15,
        ("math", "finance"): 0.2,
    }
    key = tuple(sorted([domain_a, domain_b]))
    return similarity_matrix.get(key, similarity_matrix.get((key[1], key[0]), 0.0))


def main():
    t0 = time.time()
    log("Room Model: Adapter Gradient Analysis")
    log("=" * 60)

    # Load all adapters
    adapters = {}
    for d in DOMAINS:
        path = SFT_SOURCE / d / "adapter.npz"
        if path.exists():
            adapters[d] = dict(mx.load(str(path)))
            log(f"  Loaded {d}: {len(adapters[d])} params")

    # Phase 1: Compute gradients for each adapter
    log("\n=== Phase 1: Adapter Gradients ===")
    all_gradients = {}
    for d, adapter in adapters.items():
        grads = compute_adapter_gradient(adapter)
        all_gradients[d] = grads
        mean_mag = float(np.mean([g["magnitude"] for g in grads.values()]))
        log(f"  {d}: {len(grads)} modules, mean gradient magnitude={mean_mag:.6f}")

    # Phase 2: Pairwise gradient similarity vs behavioral similarity
    log("\n=== Phase 2: Gradient vs Behavioral Similarity ===")
    pairs = list(itertools.combinations(DOMAINS, 2))
    grad_sims = []
    behav_sims = []
    pair_data = []

    for d_a, d_b in pairs:
        if d_a not in all_gradients or d_b not in all_gradients:
            continue
        gs = gradient_similarity(all_gradients[d_a], all_gradients[d_b])
        bs = behavioral_similarity(d_a, d_b)
        grad_sims.append(gs)
        behav_sims.append(bs)
        pair_data.append({"pair": f"{d_a}-{d_b}", "gradient_sim": round(gs, 4), "behavioral_sim": round(bs, 2)})
        log(f"  {d_a}-{d_b}: grad_sim={gs:.4f} behav_sim={bs:.2f}")

    # Compute correlation
    if len(grad_sims) >= 3:
        correlation = float(np.corrcoef(grad_sims, behav_sims)[0, 1])
    else:
        correlation = 0.0
    log(f"\n  Correlation(gradient_sim, behavioral_sim) = {correlation:.4f}")

    # Phase 3: Per-module gradient analysis
    log("\n=== Phase 3: Module-level patterns ===")
    # Which modules have the most variable gradients across domains?
    module_variance = {}
    sample_keys = list(all_gradients[DOMAINS[0]].keys())[:20]
    for key in sample_keys:
        mags = [all_gradients[d][key]["magnitude"] for d in DOMAINS if key in all_gradients.get(d, {})]
        if len(mags) >= 2:
            module_variance[key] = float(np.var(mags))

    # Top 5 most variable modules (these differentiate domains most)
    sorted_modules = sorted(module_variance.items(), key=lambda x: -x[1])
    log("  Most domain-differentiating modules (highest gradient variance):")
    for key, var in sorted_modules[:5]:
        log(f"    {key}: variance={var:.8f}")

    results = {
        "experiment": "room_gradient_analysis",
        "total_time_s": round(time.time() - t0, 1),
        "gradient_correlation": round(correlation, 4),
        "pairs": pair_data,
        "module_variance_top5": {k: round(v, 8) for k, v in sorted_modules[:5]},
        "n_modules_analyzed": len(sample_keys),
    }

    k825 = abs(correlation) >= 0.3
    results["kill_criteria"] = {
        "K825": {"pass": k825, "value": round(correlation, 4), "threshold": 0.3},
    }
    results["all_pass"] = k825

    log(f"\n{'='*60}")
    log(f"Correlation: {correlation:.4f} (threshold: |r| >= 0.3)")
    log(f"  {'PASS' if k825 else 'FAIL'}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
