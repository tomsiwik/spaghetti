#!/usr/bin/env python3
"""Expert Removal Safety at Macro Scale — validates micro safety bound on Qwen2.5-7B.

Micro proven: D = sum_eps * alpha where alpha=0.022, yielding 0.023% max output
deviation at d=896, N=50. This experiment measures the ACTUAL output deviation when
removing individual experts from composed pilot-50 adapters on Qwen2.5-7B.

Approach:
1. Compose N adapters (N=5,10,25,50) into a single merged adapter
2. Generate reference outputs from composed model
3. Remove one adapter at a time and measure output deviation
4. Compare empirical deviation against micro-predicted bound

Kill criteria:
- K1: empirical output deviation at N=50 exceeds micro-predicted bound by >5x
- K2: removal causes >1% perplexity regression on remaining experts' domains
- K3: deviation is NOT monotonic with adapter norm (unpredictable)

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "expert_removal_macro_safety"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

MAX_SEQ_LEN = 512 if not IS_SMOKE else 128
EVAL_SAMPLES = 20 if not IS_SMOKE else 3
N_REMOVAL_CANDIDATES = 10 if not IS_SMOKE else 2
COMPOSE_NS = [5, 10, 25, 50] if not IS_SMOKE else [3, 5]
SEED = 42

# Micro-predicted bound: D = sum_eps * alpha
# alpha = 0.022, sum_eps = sum of per-expert cosine contributions
# At d=896, SOLE cosines ~ 0.00025 (from structural_orthogonality)
# D ~ N * 0.00025 * 0.022 ~ 0.000275% at N=50
MICRO_PREDICTED_ALPHA = 0.022
MICRO_PREDICTED_COS = 0.00025

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


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def get_cluster(name):
    for cluster, members in CLUSTERS.items():
        if name in members:
            return cluster
    return "unknown"


def discover_adapters():
    """Find valid adapters with safetensors."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def compose_adapters_on_cpu(adapter_names, weights=None):
    """Compose adapters on CPU, return as temp dir with single adapter."""
    from safetensors.torch import load_file, save_file
    import torch

    if weights is None:
        weights = [1.0] * len(adapter_names)

    composed = {}
    for i, name in enumerate(adapter_names):
        path = ADAPTER_DIR / name / "adapter_model.safetensors"
        tensors = load_file(str(path), device="cpu")
        w = weights[i]
        for key, val in tensors.items():
            if key in composed:
                composed[key] = composed[key] + w * val.float()
            else:
                composed[key] = w * val.float()
        del tensors
        gc.collect()

    # Save to temp dir
    tmpdir = tempfile.mkdtemp(prefix="composed_")
    out_path = os.path.join(tmpdir, "adapter_model.safetensors")
    save_file({k: v.to(torch.bfloat16) for k, v in composed.items()}, out_path)

    # Copy adapter_config from first adapter
    cfg_src = ADAPTER_DIR / adapter_names[0] / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy(str(cfg_src), os.path.join(tmpdir, "adapter_config.json"))

    del composed
    gc.collect()
    return tmpdir


def measure_adapter_norms():
    """Compute Frobenius norm of each adapter's delta weights."""
    from safetensors.torch import load_file

    adapters = discover_adapters()
    norms = {}
    for name in adapters:
        path = ADAPTER_DIR / name / "adapter_model.safetensors"
        tensors = load_file(str(path), device="cpu")
        total = sum(t.float().norm().item() ** 2 for t in tensors.values())
        norms[name] = math.sqrt(total)
        del tensors
    gc.collect()
    return norms


def load_eval_data(domain, n_samples):
    """Load evaluation texts for a domain."""
    domain_dir = DATA_DIR / domain
    if not domain_dir.exists():
        return []

    texts = []
    for f in sorted(domain_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", obj.get("output", "")))
                    if text and len(text) > 50:
                        texts.append(text[:MAX_SEQ_LEN * 4])  # rough char limit
                except Exception:
                    continue
                if len(texts) >= n_samples:
                    break
        if len(texts) >= n_samples:
            break
    return texts[:n_samples]


def compute_logits_and_ppl(model, tokenizer, texts):
    """Compute mean logits and perplexity over texts."""
    import torch

    all_logits = []
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Per-token NLL for PPL
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
            n_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n_tokens

            # Mean logit vector (last token) for deviation measurement
            last_logits = logits[0, -1, :].float().cpu().numpy()
            all_logits.append(last_logits)

        del inputs, outputs, logits
        torch.cuda.empty_cache()

    mean_logits = np.mean(all_logits, axis=0) if all_logits else np.array([])
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return mean_logits, ppl


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def relative_deviation(ref, test):
    """Relative L2 deviation between logit vectors."""
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(test - ref) / ref_norm)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log("=" * 60)
    log("EXPERT REMOVAL SAFETY — MACRO VALIDATION")
    log("=" * 60)

    # Discover adapters
    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")

    if len(all_adapters) < max(COMPOSE_NS):
        log(f"WARN: only {len(all_adapters)} adapters, adjusting COMPOSE_NS")
        COMPOSE_NS[:] = [n for n in COMPOSE_NS if n <= len(all_adapters)]

    # Measure adapter norms
    log("Measuring adapter norms...")
    norms = measure_adapter_norms()
    norm_sorted = sorted(norms.items(), key=lambda x: -x[1])
    log(f"Norm range: {norm_sorted[-1][1]:.3f} to {norm_sorted[0][1]:.3f}")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"experiments": [], "summary": {}}

    for N in COMPOSE_NS:
        log(f"\n{'='*40}")
        log(f"TESTING N={N}")
        log(f"{'='*40}")

        # Select N adapters (diverse clusters)
        selected = []
        by_cluster = {}
        for a in all_adapters:
            c = get_cluster(a)
            by_cluster.setdefault(c, []).append(a)
        # Round-robin across clusters
        clusters = list(by_cluster.keys())
        random.shuffle(clusters)
        ci = 0
        while len(selected) < N:
            c = clusters[ci % len(clusters)]
            remaining = [a for a in by_cluster[c] if a not in selected]
            if remaining:
                selected.append(remaining[0])
            ci += 1
            if ci > N * 5:
                # Fallback: just take first N
                selected = all_adapters[:N]
                break

        log(f"Selected {len(selected)} adapters across clusters")

        # Compose all N adapters
        log(f"Composing {N} adapters...")
        composed_dir = compose_adapters_on_cpu(selected)
        log(f"Composed adapter saved to {composed_dir}")

        # Load base model + composed adapter
        log("Loading model with composed adapter...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, composed_dir)
        model.eval()

        # Collect eval data from selected domains
        eval_data = {}
        for domain in selected[:5]:  # eval on first 5 domains for speed
            texts = load_eval_data(domain, EVAL_SAMPLES)
            if texts:
                eval_data[domain] = texts
        if not eval_data:
            log("WARN: no eval data found, using synthetic prompts")
            eval_data["synthetic"] = [f"Explain the concept of {d} in detail." for d in selected[:5]]

        # Reference: compute logits/PPL with all N composed
        log("Computing reference outputs (all N composed)...")
        ref_logits = {}
        ref_ppl = {}
        for domain, texts in eval_data.items():
            logits, ppl = compute_logits_and_ppl(model, tokenizer, texts)
            ref_logits[domain] = logits
            ref_ppl[domain] = ppl
            log(f"  {domain}: PPL={ppl:.2f}")

        # Cleanup full model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # For each removal candidate: compose N-1, measure deviation
        n_to_remove = min(N_REMOVAL_CANDIDATES, N)
        # Pick diverse removal candidates: highest norm, lowest norm, random
        candidates_by_norm = sorted(selected, key=lambda x: norms.get(x, 0))
        removal_candidates = []
        if n_to_remove >= 3:
            removal_candidates.append(candidates_by_norm[-1])  # highest norm
            removal_candidates.append(candidates_by_norm[0])   # lowest norm
            removal_candidates.append(candidates_by_norm[len(candidates_by_norm)//2])  # median
            # Fill rest randomly
            remaining_pool = [a for a in selected if a not in removal_candidates]
            random.shuffle(remaining_pool)
            removal_candidates.extend(remaining_pool[:n_to_remove - 3])
        else:
            removal_candidates = selected[:n_to_remove]

        removal_results = []
        for removed in removal_candidates:
            log(f"\n  Removing: {removed} (norm={norms.get(removed, 0):.4f})")
            remaining = [a for a in selected if a != removed]

            # Compose N-1
            minus_one_dir = compose_adapters_on_cpu(remaining)

            # Load model with N-1 adapter
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(model, minus_one_dir)
            model.eval()

            # Measure deviation
            deviations = {}
            ppl_changes = {}
            for domain, texts in eval_data.items():
                test_logits, test_ppl = compute_logits_and_ppl(model, tokenizer, texts)
                dev = relative_deviation(ref_logits[domain], test_logits)
                ppl_change = (test_ppl - ref_ppl[domain]) / ref_ppl[domain] * 100
                deviations[domain] = dev
                ppl_changes[domain] = ppl_change
                log(f"    {domain}: deviation={dev:.6f}, PPL_change={ppl_change:+.3f}%")

            # Predicted deviation from micro bound
            removed_norm_frac = norms.get(removed, 0) / sum(norms.get(a, 0) for a in selected)
            predicted_dev = MICRO_PREDICTED_ALPHA * removed_norm_frac

            removal_results.append({
                "removed_adapter": removed,
                "removed_norm": norms.get(removed, 0),
                "cluster": get_cluster(removed),
                "per_domain_deviation": deviations,
                "per_domain_ppl_change_pct": ppl_changes,
                "mean_deviation": float(np.mean(list(deviations.values()))),
                "max_deviation": float(max(deviations.values())) if deviations else 0,
                "mean_ppl_change_pct": float(np.mean(list(ppl_changes.values()))),
                "max_ppl_change_pct": float(max(ppl_changes.values())) if ppl_changes else 0,
                "predicted_deviation": predicted_dev,
                "empirical_vs_predicted_ratio": (
                    float(np.mean(list(deviations.values()))) / predicted_dev
                    if predicted_dev > 0 else float("inf")
                ),
            })

            del model
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(minus_one_dir)

        # Cleanup composed dir
        shutil.rmtree(composed_dir)

        # Per-N summary
        mean_devs = [r["mean_deviation"] for r in removal_results]
        max_devs = [r["max_deviation"] for r in removal_results]
        ppl_changes_all = [r["max_ppl_change_pct"] for r in removal_results]
        norms_removed = [r["removed_norm"] for r in removal_results]

        # K3: check monotonicity (deviation vs norm correlation)
        if len(norms_removed) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(norms_removed, mean_devs)
        else:
            rho, p = 0.0, 1.0

        n_result = {
            "N": N,
            "n_adapters_evaluated": len(selected),
            "n_removals": len(removal_results),
            "mean_deviation": float(np.mean(mean_devs)),
            "max_deviation": float(np.max(max_devs)),
            "mean_max_ppl_change_pct": float(np.mean(ppl_changes_all)),
            "worst_ppl_change_pct": float(np.max(ppl_changes_all)),
            "norm_deviation_spearman_rho": float(rho),
            "norm_deviation_spearman_p": float(p),
            "micro_predicted_bound": MICRO_PREDICTED_ALPHA * N * MICRO_PREDICTED_COS,
            "empirical_vs_predicted_ratio": (
                float(np.mean(mean_devs)) / (MICRO_PREDICTED_ALPHA * N * MICRO_PREDICTED_COS)
                if MICRO_PREDICTED_ALPHA * N * MICRO_PREDICTED_COS > 0 else float("inf")
            ),
            "removal_details": removal_results,
            "ref_ppl": {k: float(v) for k, v in ref_ppl.items()},
        }
        results["experiments"].append(n_result)

        log(f"\n  N={N} summary:")
        log(f"    Mean deviation: {n_result['mean_deviation']:.8f}")
        log(f"    Max deviation:  {n_result['max_deviation']:.8f}")
        log(f"    Worst PPL change: {n_result['worst_ppl_change_pct']:+.3f}%")
        log(f"    Micro predicted bound: {n_result['micro_predicted_bound']:.8f}")
        log(f"    Empirical/predicted ratio: {n_result['empirical_vs_predicted_ratio']:.2f}x")
        log(f"    Norm-deviation Spearman rho: {rho:.3f} (p={p:.4f})")

    # Kill criteria assessment
    if results["experiments"]:
        largest_N = results["experiments"][-1]
        k1_ratio = largest_N["empirical_vs_predicted_ratio"]
        k1_pass = k1_ratio <= 5.0
        k2_worst_ppl = largest_N["worst_ppl_change_pct"]
        k2_pass = abs(k2_worst_ppl) <= 1.0
        k3_rho = largest_N["norm_deviation_spearman_rho"]
        k3_p = largest_N["norm_deviation_spearman_p"]
        k3_pass = k3_rho > 0.3 and k3_p < 0.1  # monotonic = predictable

        results["summary"] = {
            "K1_empirical_vs_predicted_ratio": k1_ratio,
            "K1_threshold": 5.0,
            "K1_pass": k1_pass,
            "K2_worst_ppl_change_pct": k2_worst_ppl,
            "K2_threshold_pct": 1.0,
            "K2_pass": k2_pass,
            "K3_norm_deviation_rho": k3_rho,
            "K3_norm_deviation_p": k3_p,
            "K3_monotonic": k3_pass,
            "overall_pass": k1_pass and k2_pass,
            "micro_alpha": MICRO_PREDICTED_ALPHA,
            "micro_cos_estimate": MICRO_PREDICTED_COS,
        }

        log(f"\n{'='*60}")
        log("KILL CRITERIA ASSESSMENT")
        log(f"{'='*60}")
        log(f"K1: empirical/predicted = {k1_ratio:.2f}x (threshold <5x): {'PASS' if k1_pass else 'FAIL'}")
        log(f"K2: worst PPL change = {k2_worst_ppl:+.3f}% (threshold <1%): {'PASS' if k2_pass else 'FAIL'}")
        log(f"K3: norm-deviation rho = {k3_rho:.3f} (p={k3_p:.4f}): {'MONOTONIC' if k3_pass else 'NOT MONOTONIC'}")
        log(f"Overall: {'PASS — micro safety bound transfers to macro' if k1_pass and k2_pass else 'FAIL'}")

    # Save results
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
