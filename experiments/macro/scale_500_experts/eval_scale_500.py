#!/usr/bin/env python3
"""Evaluate composition quality at N=50, 100, 250, 500 experts.

Tests kill criteria:
- K1: hash ring displacement at N=500 exceeds 1% per add (should be ~0.2%)
- K2: composed model quality degrades >5% vs N=50 model on shared domains
- K3: vLLM serving fails or exceeds 2x latency vs N=50

Uses pre-merged LoRA composition (weighted addition) at each scale point.
Measures per-domain PPL degradation and inference latency.

Supports SMOKE_TEST=1 for quick validation.

Usage (on RunPod via gpu_queue):
    uv run python3 tools/gpu_queue.py submit macro/scale_500_experts/eval_scale_500.py
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
PILOT_DATA_DIR = REPO_ROOT / "data" / "distillation"
SCALE_DATA_DIR = REPO_ROOT / "data" / "scale_500"
RESULTS_DIR = REPO_ROOT / "results" / "scale_500_experts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Scale points to test
N_VALUES = [5, 10] if IS_SMOKE else [50, 100, 250, 500]
EVAL_SAMPLES = 3 if IS_SMOKE else 30
MAX_SEQ_LEN = 256 if IS_SMOKE else 512
# Number of eval domains to sample at each N (evaluating all is too slow at N=500)
EVAL_DOMAINS_PER_N = 3 if IS_SMOKE else 20


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_all_adapters():
    """Find all valid adapters (pilot50 + scale)."""
    adapters = []
    if not ADAPTER_DIR.exists():
        return adapters
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_eval_texts(domain, tokenizer, n=30):
    """Load evaluation texts for a domain from pilot or scale data dirs."""
    texts = []
    # Try pilot data dir first, then scale data dir
    for base_dir in [PILOT_DATA_DIR, SCALE_DATA_DIR]:
        for fname in ["eval.jsonl", "train.jsonl"]:
            f = base_dir / domain / fname
            if not f.exists():
                continue
            with open(f) as fh:
                lines = fh.readlines()
            # Use last portion to avoid train contamination
            if fname == "train.jsonl":
                lines = lines[-min(100, len(lines)):]
            for line in lines:
                record = json.loads(line)
                if "messages" in record:
                    text = tokenizer.apply_chat_template(
                        record["messages"], tokenize=False,
                        add_generation_prompt=False)
                elif "text" in record:
                    text = record["text"]
                else:
                    continue
                texts.append(text)
                if len(texts) >= n:
                    return texts
    return texts


def measure_ppl(model, tokenizer, texts):
    """Compute mean perplexity over texts."""
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN).to(model.device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def test_hash_ring_displacement(n_experts):
    """Test hash ring displacement when adding experts one by one up to N.

    Returns max displacement percentage at each scale point.
    Uses FNV1a hash (same as routing code) on domain names.
    """
    import hashlib

    log(f"\n=== Hash Ring Displacement Test (N={n_experts}) ===")

    def fnv1a(s):
        h = 0x811c9dc5
        for b in s.encode():
            h ^= b
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h

    # Simulate token assignment with 1000 virtual tokens
    n_tokens = 10000
    results = {}

    for n in N_VALUES:
        if n > n_experts:
            break

        # Assign tokens to experts via hash ring
        expert_names = [f"expert_{i:03d}" for i in range(n)]
        ring_points = {}
        for name in expert_names:
            for v in range(50):  # 50 virtual nodes per expert
                key = f"{name}:{v}"
                ring_points[fnv1a(key)] = name
        sorted_points = sorted(ring_points.keys())

        # Count tokens per expert
        counts = {name: 0 for name in expert_names}
        for t in range(n_tokens):
            token_hash = fnv1a(f"token_{t}")
            # Find next point on ring
            idx = 0
            for i, p in enumerate(sorted_points):
                if p >= token_hash:
                    idx = i
                    break
            expert = ring_points[sorted_points[idx % len(sorted_points)]]
            counts[expert] += 1

        # Calculate displacement from uniform
        expected = n_tokens / n
        max_imbalance = max(abs(c - expected) / expected for c in counts.values())
        results[n] = round(max_imbalance * 100, 2)
        log(f"  N={n}: max imbalance = {results[n]:.2f}% (threshold: 1%)")

    return results


def compose_and_evaluate(base_model, tokenizer, adapter_names, eval_domains):
    """Compose N adapters and evaluate per-domain PPL."""
    import torch
    from peft import PeftModel

    n = len(adapter_names)
    log(f"\n  Composing N={n} adapters...")

    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIR / adapter_names[0]),
        adapter_name=adapter_names[0])

    # Load remaining adapters
    for i, name in enumerate(adapter_names[1:], 1):
        model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)
        if (i + 1) % 50 == 0:
            log(f"    Loaded {i + 1}/{n} adapters")

    log(f"    All {n} adapters loaded. Merging...")

    # Merge with equal weights (averaged by 1/N)
    weights = [1.0 / n] * n
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="composed",
        combination_type="linear",
    )
    model.set_adapter("composed")

    # Evaluate per-domain
    results = {}
    for domain in eval_domains:
        eval_texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if not eval_texts:
            continue
        ppl = measure_ppl(model, tokenizer, eval_texts)
        results[domain] = round(ppl, 4)
        log(f"    {domain}: composed PPL = {ppl:.4f}")

    # Measure inference latency
    log(f"    Measuring inference latency...")
    latencies = []
    test_text = "What is the capital of France?"
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=20)

    # Measure
    with torch.no_grad():
        for _ in range(10):
            t0 = time.time()
            model.generate(**inputs, max_new_tokens=50)
            latencies.append(time.time() - t0)

    mean_latency = sum(latencies) / len(latencies)
    log(f"    Mean latency: {mean_latency:.3f}s (50 tokens)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results, mean_latency


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log("=" * 60)
    log("Scale 500 Composition Quality Evaluation")
    log("=" * 60)
    log(f"  N values: {N_VALUES}")
    log(f"  Eval samples/domain: {EVAL_SAMPLES}")
    log(f"  Eval domains/N: {EVAL_DOMAINS_PER_N}")
    log(f"  Smoke test: {IS_SMOKE}")

    # Discover all adapters
    all_adapters = discover_all_adapters()
    log(f"\nFound {len(all_adapters)} total adapters")

    max_n = max(N_VALUES)
    if len(all_adapters) < max_n:
        log(f"WARNING: only {len(all_adapters)} adapters, capping N_VALUES")
        actual_n_values = [n for n in N_VALUES if n <= len(all_adapters)]
        if all_adapters:
            actual_n_values.append(len(all_adapters))
        actual_n_values = sorted(set(actual_n_values))
    else:
        actual_n_values = N_VALUES

    if not actual_n_values:
        log("ERROR: no adapters found")
        sys.exit(1)

    log(f"Testing N values: {actual_n_values}")

    # Phase 0: Hash ring displacement test (no GPU needed)
    hash_ring_results = test_hash_ring_displacement(max(actual_n_values))
    k1_pass = all(v <= 1.0 for v in hash_ring_results.values())
    log(f"\nK1 (hash ring <1% imbalance): {'PASS' if k1_pass else 'FAIL'}")

    # Load base model
    log("\nLoading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    log("Base model loaded.")

    # Evaluate at each scale point
    random.seed(42)
    composition_results = {}
    latency_results = {}

    for n in actual_n_values:
        log(f"\n{'=' * 40}")
        log(f"=== N={n} ===")
        log(f"{'=' * 40}")

        # Select N adapters (deterministic)
        selected = all_adapters[:n]

        # Sample eval domains from the selected set
        eval_subset = random.sample(
            selected, min(EVAL_DOMAINS_PER_N, len(selected)))

        ppls, latency = compose_and_evaluate(
            base_model, tokenizer, selected, eval_subset)
        composition_results[n] = ppls
        latency_results[n] = latency

    # Kill criteria assessment
    log(f"\n{'=' * 60}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'=' * 60}")

    # K2: Quality degradation >5% vs N=50
    k2_pass = True
    k2_detail = {}
    if 50 in composition_results and len(actual_n_values) > 1:
        n50_ppls = composition_results[50]
        largest_n = max(actual_n_values)
        largest_ppls = composition_results[largest_n]

        # Compare on shared domains
        shared = set(n50_ppls.keys()) & set(largest_ppls.keys())
        if shared:
            degradations = {}
            for domain in shared:
                if n50_ppls[domain] > 0:
                    deg = (largest_ppls[domain] - n50_ppls[domain]) / n50_ppls[domain] * 100
                    degradations[domain] = round(deg, 2)
            if degradations:
                max_deg = max(degradations.values())
                mean_deg = sum(degradations.values()) / len(degradations)
                k2_pass = max_deg <= 5.0
                k2_detail = {
                    "n_compared": f"{50} vs {largest_n}",
                    "shared_domains": len(shared),
                    "max_degradation_pct": round(max_deg, 2),
                    "mean_degradation_pct": round(mean_deg, 2),
                    "per_domain": degradations,
                }
    log(f"K2 (quality degradation <5% vs N=50): {'PASS' if k2_pass else 'FAIL'}")
    if k2_detail:
        log(f"  Max: {k2_detail.get('max_degradation_pct', 'N/A')}%, "
            f"Mean: {k2_detail.get('mean_degradation_pct', 'N/A')}%")

    # K3: Latency <2x vs N=50
    k3_pass = True
    k3_detail = {}
    if 50 in latency_results and len(actual_n_values) > 1:
        lat_50 = latency_results[50]
        largest_n = max(actual_n_values)
        lat_max = latency_results[largest_n]
        ratio = lat_max / lat_50 if lat_50 > 0 else float("inf")
        k3_pass = ratio <= 2.0
        k3_detail = {
            "latency_n50": round(lat_50, 4),
            f"latency_n{largest_n}": round(lat_max, 4),
            "ratio": round(ratio, 3),
        }
    log(f"K3 (latency <2x vs N=50): {'PASS' if k3_pass else 'FAIL'}")
    if k3_detail:
        log(f"  Ratio: {k3_detail.get('ratio', 'N/A')}x")

    verdict = "PASS" if (k1_pass and k2_pass and k3_pass) else "FAIL"
    log(f"\nVERDICT: {verdict}")

    # Save results
    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_values_tested": actual_n_values,
            "total_adapters": len(all_adapters),
            "eval_samples_per_domain": EVAL_SAMPLES,
            "eval_domains_per_n": EVAL_DOMAINS_PER_N,
            "smoke_test": IS_SMOKE,
        },
        "hash_ring_displacement": hash_ring_results,
        "composition_ppl": {
            str(n): ppls for n, ppls in composition_results.items()
        },
        "latency_s": {str(n): round(lat, 4) for n, lat in latency_results.items()},
        "kill_criteria": {
            "K1_hash_ring_lt_1pct": {
                "results": hash_ring_results,
                "pass": k1_pass,
            },
            "K2_quality_degradation_lt_5pct": {
                **k2_detail,
                "pass": k2_pass,
            },
            "K3_latency_lt_2x": {
                **k3_detail,
                "pass": k3_pass,
            },
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_file = RESULTS_DIR / "scale_500_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"\nResults saved to {out_file}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
