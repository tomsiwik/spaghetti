#!/usr/bin/env python3
"""Sweep quality capture as function of specialization and N/domain ratio.

Answers: Under what conditions does routing QUALITY justify routing COST?
"""

import json
import numpy as np
from pathlib import Path
from experiments.models.inference_routing_strategies.routing_strategies import (
    generate_quality_matrix, generate_queries, compute_expert_centroids,
    OracleRouter, PreMergeRouter, HashRingRouter, EmbeddingSimilarityRouter,
    TinyClassifierRouter, HierarchicalRouter, measure_quality,
)


def run_sweep():
    embed_dim = 64
    n_clusters = 3
    n_seeds = 3

    # Sweep parameters
    specialization_values = [0.2, 0.5, 0.8, 1.0, 2.0]
    # Ratio of experts per domain
    expert_ratio_configs = [
        {"n_domains": 15, "n_experts": 15, "label": "1:1"},
        {"n_domains": 15, "n_experts": 30, "label": "2:1"},
        {"n_domains": 15, "n_experts": 60, "label": "4:1"},
        {"n_domains": 15, "n_experts": 100, "label": "6.7:1"},
        {"n_domains": 5, "n_experts": 5, "label": "1:1 (N=5)"},
        {"n_domains": 5, "n_experts": 50, "label": "10:1 (N=50)"},
    ]

    results = []

    for spec in specialization_values:
        for cfg in expert_ratio_configs:
            n_dom = cfg["n_domains"]
            n_exp = cfg["n_experts"]
            label = cfg["label"]

            captures = {s: [] for s in ["premerge", "hash_ring", "embedding_sim",
                                         "tiny_classifier", "hierarchical"]}
            oracle_agrs = {s: [] for s in captures}

            for seed in range(n_seeds):
                rng = np.random.default_rng(42 + seed)

                qm = generate_quality_matrix(n_dom, n_exp, n_clusters, spec, rng)
                Q = qm["Q"]
                queries = generate_queries(2000, n_dom, embed_dim,
                                           qm["domain_to_cluster"], rng)
                test_queries = generate_queries(1000, n_dom, embed_dim,
                                                qm["domain_to_cluster"],
                                                np.random.default_rng(1000 + seed))

                expert_centroids = compute_expert_centroids(
                    qm["expert_to_domain"], queries["domain_centroids"],
                    n_exp, embed_dim, rng)
                oracle = OracleRouter(Q)

                # Train labels
                train_labels = np.array([oracle.route_for_domain(d)
                                         for d in queries["domains"]])

                # Build routers
                routers = {
                    "premerge": PreMergeRouter(Q),
                    "hash_ring": HashRingRouter(n_exp),
                    "embedding_sim": EmbeddingSimilarityRouter(expert_centroids),
                    "hierarchical": HierarchicalRouter(
                        queries["cluster_centroids"], qm["expert_to_cluster"], n_exp),
                }

                tc = TinyClassifierRouter(n_exp, embed_dim, hidden_dim=32, rng=rng)
                tc.train(queries["embeddings"], train_labels, epochs=100)
                routers["tiny_classifier"] = tc

                for name, router in routers.items():
                    qual = measure_quality(router, Q, test_queries["domains"],
                                           test_queries["embeddings"], oracle)
                    captures[name].append(qual["quality_capture"])
                    oracle_agrs[name].append(qual["oracle_agreement"])

            row = {
                "specialization": spec,
                "config": label,
                "n_domains": n_dom,
                "n_experts": n_exp,
                "ratio": n_exp / n_dom,
            }
            for name in captures:
                row[f"{name}_capture_mean"] = float(np.mean(captures[name]))
                row[f"{name}_capture_std"] = float(np.std(captures[name]))
                row[f"{name}_oracle_agr_mean"] = float(np.mean(oracle_agrs[name]))

            results.append(row)

            # Print
            best_name = max(captures.keys(),
                            key=lambda n: np.mean(captures[n]))
            best_val = np.mean(captures[best_name])
            pm_val = np.mean(captures["premerge"])
            print(f"  spec={spec:.1f}, {label:12s}: "
                  f"best={best_name}({best_val:.3f}), "
                  f"premerge={pm_val:.3f}, "
                  f"gap={best_val - pm_val:+.3f}")

    # Summary
    print("\n=== Key Finding ===")
    print("When is routing quality worth the cost?")
    print()

    # Find configs where best router quality > premerge by significant margin
    for row in results:
        best_capture = max(row[f"{s}_capture_mean"]
                          for s in ["embedding_sim", "tiny_classifier", "hierarchical"])
        pm_capture = row["premerge_capture_mean"]
        gap = best_capture - pm_capture

        if gap > 0.05:
            print(f"  WORTH IT: spec={row['specialization']}, {row['config']}: "
                  f"best routing={best_capture:.3f} vs premerge={pm_capture:.3f} "
                  f"(+{gap:.3f})")
        elif gap > 0.01:
            print(f"  MARGINAL: spec={row['specialization']}, {row['config']}: "
                  f"best routing={best_capture:.3f} vs premerge={pm_capture:.3f} "
                  f"(+{gap:.3f})")

    # K3 check: does ANY config get >90% quality capture?
    print("\n=== K3 Check: Quality Capture >90%? ===")
    k3_passes = []
    for row in results:
        for s in ["premerge", "hash_ring", "embedding_sim",
                  "tiny_classifier", "hierarchical"]:
            cap = row[f"{s}_capture_mean"]
            if cap > 0.90:
                k3_passes.append({
                    "strategy": s,
                    "config": row["config"],
                    "specialization": row["specialization"],
                    "capture": cap,
                })

    if k3_passes:
        print(f"  {len(k3_passes)} configs pass K3:")
        for p in k3_passes:
            print(f"    {p['strategy']} at spec={p['specialization']}, "
                  f"{p['config']}: {p['capture']:.3f}")
    else:
        print("  NO config passes K3 (>90% oracle quality capture)")
        print("  This is expected when N_experts >> N_domains")
        print()
        # Check at 1:1 ratio
        one_to_one = [r for r in results
                     if abs(r["ratio"] - 1.0) < 0.01]
        if one_to_one:
            print("  At 1:1 expert:domain ratio:")
            for row in one_to_one:
                for s in ["embedding_sim", "tiny_classifier"]:
                    cap = row[f"{s}_capture_mean"]
                    print(f"    spec={row['specialization']}, {s}: {cap:.3f}")

    out_path = Path(__file__).parent / "sweep_results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    run_sweep()
