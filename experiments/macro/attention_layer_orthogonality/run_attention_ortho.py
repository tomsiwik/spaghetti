#!/usr/bin/env python3
"""
Attention-Layer Orthogonality at Macro Scale (memory-efficient version)

Tests whether attention-layer LoRA adapters maintain structural orthogonality
for dissimilar domains on real Qwen2.5-7B adapters (d=3584, rank=16).

Memory-efficient: loads adapters one at a time, extracts vectors, releases.
"""

import json
import sys
import time
import gc
from pathlib import Path
import numpy as np

ADAPTER_DIR = Path("/workspace/llm/adapters")
RESULTS_DIR = Path("/workspace/llm/results/attention_ortho")

CLUSTERS = {
    "stem": ["astronomy", "biology", "chemistry", "genetics", "geology",
             "neuroscience", "physics", "statistics", "math", "abstract-math"],
    "programming": ["bash", "cpp", "go", "java", "javascript", "python",
                    "rust", "sql", "swift", "typescript"],
    "writing": ["academic-writing", "copywriting", "creative-fiction",
                "documentation", "grant-writing", "journalism", "marketing",
                "poetry", "screenplay", "speechwriting", "technical-writing"],
    "reasoning": ["analogical-reasoning", "causal-reasoning", "critical-analysis",
                  "logic-puzzles", "spatial-reasoning", "systems-thinking"],
    "professional": ["accounting", "cybersecurity", "data-engineering", "debate",
                     "devops", "ethics", "finance", "game-theory", "hr", "legal",
                     "medical", "project-management", "ecology"],
}


def get_cluster(name):
    for cluster, members in CLUSTERS.items():
        if name in members:
            return cluster
    return "unknown"


def extract_vectors(adapter_path):
    """Load adapter, extract flattened attn and mlp vectors as float32, release weights."""
    from safetensors.torch import load_file
    weights = load_file(str(adapter_path / "adapter_model.safetensors"))

    attn_parts = []
    mlp_parts = []
    for key in sorted(weights.keys()):
        arr = weights[key].float().numpy().flatten()
        if "self_attn" in key:
            attn_parts.append(arr)
        elif "mlp" in key:
            mlp_parts.append(arr)

    attn_vec = np.concatenate(attn_parts).astype(np.float32)
    mlp_vec = np.concatenate(mlp_parts).astype(np.float32)
    del weights, attn_parts, mlp_parts
    gc.collect()
    return attn_vec, mlp_vec


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(abs(np.dot(a, b) / (na * nb)))


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    adapters = sorted([d.name for d in ADAPTER_DIR.iterdir() if d.is_dir()
                       and (d / "adapter_model.safetensors").exists()])
    print(f"Found {len(adapters)} adapters", flush=True)

    clusters = {a: get_cluster(a) for a in adapters}
    unknown = [a for a, c in clusters.items() if c == "unknown"]
    if unknown:
        print(f"Warning: unclassified: {unknown}", flush=True)

    # Phase 1: Extract all vectors (one at a time, keep only float32 arrays)
    print("Phase 1: Extracting vectors...", flush=True)
    attn_vecs = {}
    mlp_vecs = {}
    for i, name in enumerate(adapters):
        attn_vecs[name], mlp_vecs[name] = extract_vectors(ADAPTER_DIR / name)
        print(f"  [{i+1}/{len(adapters)}] {name}: attn={attn_vecs[name].shape}, mlp={mlp_vecs[name].shape}", flush=True)

    attn_dim = attn_vecs[adapters[0]].shape[0]
    mlp_dim = mlp_vecs[adapters[0]].shape[0]
    print(f"Attention vector dim: {attn_dim}", flush=True)
    print(f"MLP vector dim: {mlp_dim}", flush=True)

    # Phase 2: Pairwise cosines
    print("\nPhase 2: Pairwise cosines...", flush=True)
    results = {"attention_dissimilar": [], "attention_similar": [],
               "mlp_dissimilar": [], "mlp_similar": []}
    pair_details = []

    n = len(adapters)
    done = 0
    total = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            a, b = adapters[i], adapters[j]
            ptype = "similar" if clusters[a] == clusters[b] else "dissimilar"

            ac = cosine_sim(attn_vecs[a], attn_vecs[b])
            mc = cosine_sim(mlp_vecs[a], mlp_vecs[b])

            results[f"attention_{ptype}"].append(ac)
            results[f"mlp_{ptype}"].append(mc)
            pair_details.append({
                "a": a, "b": b, "ca": clusters[a], "cb": clusters[b],
                "type": ptype, "attn_cos": ac, "mlp_cos": mc
            })
            done += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed adapter {i+1}/{n} ({done}/{total} pairs)", flush=True)

    # Phase 3: Statistics
    r, d = 16, 3584
    bound = np.sqrt(r / d)
    print(f"\nBound sqrt({r}/{d}) = {bound:.6f}", flush=True)

    stats = {}
    for key, vals in results.items():
        if not vals:
            continue
        arr = np.array(vals)
        stats[key] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "frac_above_bound": float(np.mean(arr > bound)),
            "frac_above_0.1": float(np.mean(arr > 0.1)),
            "frac_above_0.01": float(np.mean(arr > 0.01)),
        }
        print(f"\n{key}:", flush=True)
        for k, v in stats[key].items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    # Kill criteria
    ad = stats.get("attention_dissimilar", {})
    k1 = ad.get("frac_above_bound", 0) > 0.20
    k2 = ad.get("max", 0) > 0.1

    print(f"\n{'='*60}", flush=True)
    print(f"KILL CRITERIA", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"K1 (>20% dissimilar pairs above bound): {'FAIL' if k1 else 'PASS'} ({ad.get('frac_above_bound',0):.4f})", flush=True)
    print(f"K2 (any dissimilar pair > 0.1): {'FAIL' if k2 else 'PASS'} (max={ad.get('max',0):.6f})", flush=True)
    print(f"Overall: {'KILLED' if (k1 or k2) else 'SURVIVES'}", flush=True)

    # Phase 4: Per-layer analysis (sampled)
    print(f"\nPhase 4: Per-layer analysis (30 sampled dissimilar pairs)...", flush=True)
    per_layer = per_layer_analysis(adapters, clusters)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    # Save
    output = {
        "config": {"n_adapters": len(adapters), "rank": r, "d_model": d,
                    "bound": float(bound), "attn_dim": attn_dim, "mlp_dim": mlp_dim},
        "statistics": stats,
        "kill_criteria": {"k1": k1, "k1_val": float(ad.get("frac_above_bound", 0)),
                          "k2": k2, "k2_val": float(ad.get("max", 0)),
                          "killed": k1 or k2},
        "per_layer": per_layer,
        "pair_details": pair_details,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'results.json'}", flush=True)

    # Top-20 highest attention cosines (dissimilar)
    dissim = sorted([p for p in pair_details if p["type"] == "dissimilar"],
                    key=lambda x: x["attn_cos"], reverse=True)
    print(f"\nTop-20 highest attention cos (dissimilar):", flush=True)
    for p in dissim[:20]:
        print(f"  {p['a']:25s} <-> {p['b']:25s}  attn={p['attn_cos']:.6f}  mlp={p['mlp_cos']:.6f}  ({p['ca']}/{p['cb']})", flush=True)


def per_layer_analysis(adapters, clusters):
    """Per-transformer-layer attention cosines for sampled dissimilar pairs."""
    from safetensors.torch import load_file

    # Get layer indices from first adapter
    w0 = load_file(str(ADAPTER_DIR / adapters[0] / "adapter_model.safetensors"))
    layer_nums = sorted(set(
        int(k.split(".")[4]) for k in w0.keys() if "self_attn" in k
    ))
    del w0
    gc.collect()
    print(f"  Found {len(layer_nums)} layers", flush=True)

    # Sample dissimilar pairs
    dissim = [(a, b) for i, a in enumerate(adapters)
              for b in adapters[i+1:] if clusters[a] != clusters[b]]
    np.random.seed(42)
    if len(dissim) > 30:
        idx = np.random.choice(len(dissim), 30, replace=False)
        sampled = [dissim[i] for i in idx]
    else:
        sampled = dissim

    per_layer_cos = {l: [] for l in layer_nums}

    for pi, (a_name, b_name) in enumerate(sampled):
        wa = load_file(str(ADAPTER_DIR / a_name / "adapter_model.safetensors"))
        wb = load_file(str(ADAPTER_DIR / b_name / "adapter_model.safetensors"))

        for l in layer_nums:
            prefix = f"base_model.model.model.layers.{l}.self_attn"
            a_parts = []
            b_parts = []
            for k in sorted(wa.keys()):
                if k.startswith(prefix):
                    a_parts.append(wa[k].float().numpy().flatten())
                    b_parts.append(wb[k].float().numpy().flatten())
            if a_parts:
                cos = cosine_sim(np.concatenate(a_parts), np.concatenate(b_parts))
                per_layer_cos[l].append(cos)

        del wa, wb
        gc.collect()
        if (pi + 1) % 10 == 0:
            print(f"  Per-layer: {pi+1}/{len(sampled)} pairs done", flush=True)

    # Report
    print(f"\n  Per-layer attention cos (dissimilar, {len(sampled)} pairs):", flush=True)
    print(f"  {'Layer':>6s} {'Mean':>10s} {'Max':>10s} {'P95':>10s}", flush=True)
    stats = {}
    for l in layer_nums:
        arr = np.array(per_layer_cos[l])
        if len(arr) > 0:
            s = {"mean": float(np.mean(arr)), "max": float(np.max(arr)),
                 "p95": float(np.percentile(arr, 95)), "std": float(np.std(arr))}
            stats[str(l)] = s
            print(f"  {l:6d} {s['mean']:10.6f} {s['max']:10.6f} {s['p95']:10.6f}", flush=True)

    return stats


if __name__ == "__main__":
    main()
