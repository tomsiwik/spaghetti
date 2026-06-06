#!/usr/bin/env python3
"""Expert redundancy clustering: identify prunable adapters via cosine similarity.

Computes 50x50 pairwise cosine matrix, clusters by similarity, and tests whether
pruning redundant experts improves composed model quality (less dilution).

Uses existing pilot50 adapters. Mostly weight-space analysis + targeted eval.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/expert_redundancy_clustering")
SEED = 42
REDUNDANCY_THRESHOLD = 0.10  # cos > 0.10 = potentially redundant


def extract_adapter_vector(adapter_path):
    """Extract flattened adapter weight vector."""
    from safetensors.torch import load_file

    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        # Try bin format
        bin_path = adapter_path / "adapter_model.bin"
        if bin_path.exists():
            weights = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
    else:
        weights = load_file(str(safetensors_path))

    parts = []
    for key in sorted(weights.keys()):
        parts.append(weights[key].float().numpy().flatten())
    return np.concatenate(parts).astype(np.float32)


def cosine_sim(a, b):
    """Compute absolute cosine similarity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(abs(np.dot(a, b) / (na * nb)))


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load all adapter vectors
    print("=== Phase 1: Loading adapter vectors ===")
    adapter_dirs = sorted(
        [d for d in ADAPTER_DIR.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]
    )
    print(f"Found {len(adapter_dirs)} adapters")

    vectors = {}
    for d in adapter_dirs:
        try:
            vec = extract_adapter_vector(d)
            vectors[d.name] = vec
            print(f"  {d.name}: dim={len(vec)}")
        except Exception as e:
            print(f"  SKIP {d.name}: {e}")

    names = sorted(vectors.keys())
    n = len(names)
    print(f"\nLoaded {n} adapter vectors, dimension={len(vectors[names[0]])}")

    # Phase 2: Pairwise cosine matrix
    print("\n=== Phase 2: Pairwise cosine similarity ===")
    cos_matrix = np.zeros((n, n))
    pair_details = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = cosine_sim(vectors[names[i]], vectors[names[j]])
            cos_matrix[i, j] = cos
            cos_matrix[j, i] = cos
            pair_details.append({
                "a": names[i],
                "b": names[j],
                "cosine": round(cos, 6),
            })

    # Statistics
    upper_tri = cos_matrix[np.triu_indices(n, k=1)]
    print(f"Cosine stats: mean={np.mean(upper_tri):.6f}, "
          f"std={np.std(upper_tri):.6f}, "
          f"max={np.max(upper_tri):.6f}, "
          f"min={np.min(upper_tri):.6f}")

    # Sort by similarity
    pair_details.sort(key=lambda x: x["cosine"], reverse=True)
    print(f"\nTop-10 most similar pairs:")
    for p in pair_details[:10]:
        print(f"  {p['a']} <-> {p['b']}: cos={p['cosine']:.6f}")

    n_redundant_pairs = sum(1 for p in pair_details if p["cosine"] > REDUNDANCY_THRESHOLD)
    print(f"\nPairs with cos > {REDUNDANCY_THRESHOLD}: {n_redundant_pairs}/{len(pair_details)}")

    # Phase 3: Hierarchical clustering
    print("\n=== Phase 3: Hierarchical clustering ===")
    dist_matrix = 1.0 - cos_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")

    # Cut at threshold
    clusters = fcluster(Z, t=1.0 - REDUNDANCY_THRESHOLD, criterion="distance")
    cluster_map = {}
    for i, c in enumerate(clusters):
        cluster_map.setdefault(int(c), []).append(names[i])

    print(f"Clusters at cos>{REDUNDANCY_THRESHOLD}: {len(cluster_map)}")
    for cid, members in sorted(cluster_map.items()):
        if len(members) > 1:
            print(f"  Cluster {cid} ({len(members)} members): {members}")

    # Phase 4: Select representatives (highest L2 norm = strongest signal)
    print("\n=== Phase 4: Pruning analysis ===")
    representatives = []
    pruned = []
    for cid, members in cluster_map.items():
        if len(members) == 1:
            representatives.append(members[0])
        else:
            # Pick member with highest L2 norm (strongest adapter)
            norms = {m: np.linalg.norm(vectors[m]) for m in members}
            best = max(norms, key=norms.get)
            representatives.append(best)
            pruned.extend([m for m in members if m != best])
            print(f"  Cluster {cid}: keep {best} (norm={norms[best]:.2f}), "
                  f"prune {[m for m in members if m != best]}")

    n_pruned = len(pruned)
    n_kept = len(representatives)
    print(f"\nPruning result: {n_kept} kept, {n_pruned} pruned "
          f"({n_pruned / n * 100:.1f}% reduction)")

    # Phase 5: Composition quality comparison (if pruning happened)
    composition_comparison = None
    if n_pruned > 0 and n_kept >= 2:
        print("\n=== Phase 5: Composition quality comparison ===")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            base_model.eval()

            # Load eval data (use a few MMLU subjects)
            from datasets import load_dataset

            eval_subjects = ["abstract_algebra", "college_physics", "machine_learning",
                             "medical_genetics", "philosophy"]
            eval_data = {}
            for subj in eval_subjects:
                try:
                    ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                    eval_data[subj] = ds.select(range(min(50, len(ds))))
                except Exception:
                    pass

            def eval_composition(model, tokenizer, adapter_list, name):
                """Compose adapters and evaluate MMLU accuracy."""
                from peft import PeftModel
                composed = PeftModel.from_pretrained(
                    model, str(ADAPTER_DIR / adapter_list[0]),
                    adapter_name=adapter_list[0])
                for a in adapter_list[1:]:
                    composed.load_adapter(str(ADAPTER_DIR / a), adapter_name=a)
                composed.add_weighted_adapter(
                    adapters=list(adapter_list),
                    weights=[1.0 / len(adapter_list)] * len(adapter_list),
                    adapter_name="composed",
                    combination_type="linear",
                )
                composed.set_adapter("composed")
                composed.eval()

                total_correct, total_count = 0, 0
                choice_tokens = {}
                for letter in "ABCD":
                    ids = tokenizer.encode(letter, add_special_tokens=False)
                    choice_tokens[letter] = ids[0]
                    ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
                    if ids_space:
                        choice_tokens[f" {letter}"] = ids_space[-1]

                for subj, ds in eval_data.items():
                    correct = 0
                    for ex in ds:
                        q = ex["question"]
                        prompt = f"{q}\n"
                        for i, c in enumerate(ex["choices"]):
                            prompt += f"{'ABCD'[i]}. {c}\n"
                        prompt += "Answer:"
                        inputs = tokenizer(prompt, return_tensors="pt",
                                           truncation=True, max_length=512).to(model.device)
                        with torch.no_grad():
                            out = composed(**inputs)
                            logits = out.logits[0, -1]
                            lp = torch.log_softmax(logits, dim=-1)
                        scores = {}
                        for letter in "ABCD":
                            tid = choice_tokens[letter]
                            tid_s = choice_tokens.get(f" {letter}", tid)
                            scores[letter] = max(lp[tid].item(), lp[tid_s].item())
                        pred = max(scores, key=scores.get)
                        gold = "ABCD"[ex["answer"]]
                        correct += int(pred == gold)
                    total_correct += correct
                    total_count += len(ds)

                acc = total_correct / max(1, total_count)
                del composed
                torch.cuda.empty_cache()
                return {"accuracy": round(acc, 4), "correct": total_correct, "total": total_count}

            print(f"Evaluating full composition (N={n})...")
            full_result = eval_composition(base_model, tokenizer, names, "full")
            print(f"  Full (N={n}): {full_result['accuracy']:.4f}")

            print(f"Evaluating pruned composition (N={n_kept})...")
            pruned_result = eval_composition(base_model, tokenizer, representatives, "pruned")
            print(f"  Pruned (N={n_kept}): {pruned_result['accuracy']:.4f}")

            delta = (pruned_result["accuracy"] - full_result["accuracy"]) * 100
            composition_comparison = {
                "full_n": n,
                "full_accuracy": full_result["accuracy"],
                "pruned_n": n_kept,
                "pruned_accuracy": pruned_result["accuracy"],
                "delta_pp": round(delta, 4),
                "pruning_helps": delta > 0,
            }
            print(f"\nPruned vs Full: {delta:+.2f}pp")

        except Exception as e:
            print(f"Composition comparison failed: {e}")
            import traceback
            traceback.print_exc()

    # Kill criteria
    k1_pass = np.max(upper_tri) >= REDUNDANCY_THRESHOLD  # redundancy exists
    k2_pass = True
    if composition_comparison:
        k2_pass = composition_comparison["delta_pp"] >= -3.0  # pruning doesn't hurt >3%

    verdict = "PASS" if k1_pass and k2_pass else "FAIL"

    # Save results
    results = {
        "n_adapters": n,
        "cosine_stats": {
            "mean": round(float(np.mean(upper_tri)), 6),
            "std": round(float(np.std(upper_tri)), 6),
            "max": round(float(np.max(upper_tri)), 6),
            "min": round(float(np.min(upper_tri)), 6),
            "median": round(float(np.median(upper_tri)), 6),
        },
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "n_redundant_pairs": n_redundant_pairs,
        "n_clusters": len(cluster_map),
        "multi_member_clusters": {
            str(cid): members for cid, members in cluster_map.items() if len(members) > 1
        },
        "representatives": representatives,
        "pruned": pruned,
        "n_kept": n_kept,
        "n_pruned": n_pruned,
        "pruning_pct": round(n_pruned / n * 100, 1),
        "top_20_pairs": pair_details[:20],
        "composition_comparison": composition_comparison,
        "kill_criteria": {
            "k1_redundancy_exists": k1_pass,
            "k2_pruning_safe": k2_pass,
            "verdict": verdict,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "redundancy_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
