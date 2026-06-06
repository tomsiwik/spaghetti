#!/usr/bin/env python3
"""Pre-merge composition quality at N=5,10,25,50 pilot experts.

Tests whether individual expert quality translates to composed model quality
by pre-merging increasing numbers of LoRA adapters via weighted addition
and measuring per-domain PPL degradation vs single-expert baselines.

Kill criteria:
- K1: pre-merged model at N=50 degrades >10% on any individual domain vs single-expert
- K2: quality degrades superlinearly with N (composition doesn't scale)
- K3: N=50 pre-merge worse than base on >20% of domains

Supports SMOKE_TEST=1 for <60s validation.
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
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "pilot50_composition_quality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Composition sizes to test
N_VALUES = [3, 5] if IS_SMOKE else [5, 10, 25, 50]
EVAL_SAMPLES = 5 if IS_SMOKE else 50
MAX_SEQ_LEN = 256 if IS_SMOKE else 512


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid adapters in the adapter directory."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_eval_texts(domain, tokenizer, n=50):
    """Load evaluation texts for a domain."""
    texts = []
    for fname in ["eval.jsonl", "train.jsonl"]:
        f = DATA_DIR / domain / fname
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        # Use last portion for eval to avoid train contamination
        if fname == "train.jsonl":
            lines = lines[-min(200, len(lines)):]
        for line in lines:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue
            texts.append(text)
            if len(texts) >= n:
                return texts
    return texts


def measure_ppl(model, tokenizer, texts):
    """Compute mean perplexity over a set of texts."""
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
        return float("inf"), []
    return math.exp(sum(losses) / len(losses)), losses


def measure_single_expert_baselines(base_model, tokenizer, domains):
    """Measure PPL for each domain with its own single expert adapter."""
    import torch
    from peft import PeftModel

    baselines = {}
    for domain in domains:
        adapter_path = ADAPTER_DIR / domain
        if not adapter_path.exists():
            log(f"  SKIP {domain}: no adapter")
            continue
        eval_texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if not eval_texts:
            log(f"  SKIP {domain}: no eval data")
            continue

        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        ppl, _ = measure_ppl(model, tokenizer, eval_texts)
        baselines[domain] = {"ppl": round(ppl, 4), "n_eval": len(eval_texts)}
        log(f"  {domain}: single-expert PPL = {ppl:.4f} ({len(eval_texts)} samples)")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return baselines


def measure_base_model_ppl(base_model, tokenizer, domains):
    """Measure PPL for each domain using the base model (no adapters)."""
    base_ppls = {}
    for domain in domains:
        eval_texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if not eval_texts:
            continue
        ppl, _ = measure_ppl(base_model, tokenizer, eval_texts)
        base_ppls[domain] = round(ppl, 4)
        log(f"  {domain}: base PPL = {ppl:.4f}")
    return base_ppls


def compose_and_evaluate(base_model, tokenizer, adapter_names, eval_domains):
    """Compose N adapters via weighted addition and evaluate per-domain PPL.

    Uses PEFT's add_weighted_adapter with combination_type='linear' and
    equal weights (SOLE: unit-weight addition, scaled by 1/N for stability).
    """
    import torch
    from peft import PeftModel

    n = len(adapter_names)
    log(f"\n  Composing N={n} adapters: {adapter_names[:5]}{'...' if n > 5 else ''}")

    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIR / adapter_names[0]),
        adapter_name=adapter_names[0])

    # Load remaining adapters
    for name in adapter_names[1:]:
        model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)

    # Merge with equal weights (SOLE addition, averaged by 1/N)
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
        ppl, _ = measure_ppl(model, tokenizer, eval_texts)
        results[domain] = round(ppl, 4)
        log(f"    {domain}: composed PPL = {ppl:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log("=" * 60)
    log("Pilot 50 Composition Quality Experiment")
    log("=" * 60)
    log(f"Smoke test: {IS_SMOKE}")

    # Discover available adapters
    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters: {all_adapters[:10]}...")

    if len(all_adapters) < max(N_VALUES):
        log(f"WARNING: only {len(all_adapters)} adapters, capping N_VALUES")
        N_VALUES_ACTUAL = [n for n in N_VALUES if n <= len(all_adapters)]
        if not N_VALUES_ACTUAL:
            N_VALUES_ACTUAL = [len(all_adapters)]
    else:
        N_VALUES_ACTUAL = N_VALUES

    log(f"Testing N values: {N_VALUES_ACTUAL}")

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

    # Identify evaluation domains (adapters that have eval data)
    eval_domains = []
    for name in all_adapters:
        data_dir = DATA_DIR / name
        if data_dir.exists() and any(data_dir.iterdir()):
            eval_domains.append(name)
    log(f"Evaluation domains ({len(eval_domains)}): {eval_domains[:10]}...")

    # Phase 1: Base model PPL (no adapters)
    log("\n=== Phase 1: Base Model PPL ===")
    base_ppls = measure_base_model_ppl(base_model, tokenizer, eval_domains)

    # Phase 2: Single-expert baselines
    log("\n=== Phase 2: Single Expert Baselines ===")
    single_baselines = measure_single_expert_baselines(
        base_model, tokenizer, eval_domains)
    log(f"Got baselines for {len(single_baselines)} domains")

    # Phase 3: Composition at each N
    log("\n=== Phase 3: Composition Quality at Varying N ===")
    random.seed(42)
    composition_results = {}

    for n in N_VALUES_ACTUAL:
        # Select N adapters (deterministic, include domains we can evaluate)
        selected = list(single_baselines.keys())[:n]
        if len(selected) < n:
            # Pad with additional adapters
            remaining = [a for a in all_adapters if a not in selected]
            selected.extend(remaining[:n - len(selected)])

        # Only evaluate domains that are in the composed set AND have baselines
        eval_subset = [d for d in selected if d in single_baselines]

        composed_ppls = compose_and_evaluate(
            base_model, tokenizer, selected, eval_subset)
        composition_results[n] = {
            "adapters": selected,
            "per_domain_ppl": composed_ppls,
        }

    # Phase 4: Kill criteria assessment
    log("\n=== Phase 4: Kill Criteria Assessment ===")

    # K1: Max degradation at largest N vs single-expert
    largest_n = max(composition_results.keys())
    largest_result = composition_results[largest_n]
    max_degradation = 0.0
    degradations = {}
    for domain, composed_ppl in largest_result["per_domain_ppl"].items():
        if domain in single_baselines:
            single_ppl = single_baselines[domain]["ppl"]
            if single_ppl > 0:
                deg_pct = (composed_ppl - single_ppl) / single_ppl * 100
                degradations[domain] = round(deg_pct, 2)
                max_degradation = max(max_degradation, deg_pct)

    k1_pass = max_degradation <= 10.0

    # K2: Superlinear degradation check
    # Fit mean degradation vs N — if exponent > 1.2, it's superlinear
    mean_degs = {}
    for n, res in sorted(composition_results.items()):
        degs = []
        for domain, composed_ppl in res["per_domain_ppl"].items():
            if domain in single_baselines:
                single_ppl = single_baselines[domain]["ppl"]
                if single_ppl > 0:
                    degs.append((composed_ppl - single_ppl) / single_ppl * 100)
        if degs:
            mean_degs[n] = sum(degs) / len(degs)

    superlinear = False
    if len(mean_degs) >= 2:
        ns = sorted(mean_degs.keys())
        # Check if degradation growth rate accelerates
        if len(ns) >= 3:
            # Compare growth rate between first half and second half
            mid = len(ns) // 2
            first_rate = (mean_degs[ns[mid]] - mean_degs[ns[0]]) / (ns[mid] - ns[0]) if ns[mid] != ns[0] else 0
            last_rate = (mean_degs[ns[-1]] - mean_degs[ns[mid]]) / (ns[-1] - ns[mid]) if ns[-1] != ns[mid] else 0
            superlinear = last_rate > first_rate * 1.5 and last_rate > 0
        else:
            # With only 2 points, check if rate is excessive
            rate = (mean_degs[ns[-1]] - mean_degs[ns[0]]) / (ns[-1] - ns[0])
            superlinear = rate > 0.5  # >0.5% per additional expert is concerning
    k2_pass = not superlinear

    # K3: Domains worse than base at N=largest
    n_worse_than_base = 0
    n_evaluated = 0
    worse_domains = []
    for domain, composed_ppl in largest_result["per_domain_ppl"].items():
        if domain in base_ppls:
            n_evaluated += 1
            if composed_ppl > base_ppls[domain]:
                n_worse_than_base += 1
                worse_domains.append(domain)
    worse_pct = (n_worse_than_base / max(n_evaluated, 1)) * 100
    k3_pass = worse_pct <= 20.0

    # Build aggregate results
    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_values": N_VALUES_ACTUAL,
            "eval_samples_per_domain": EVAL_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "total_adapters": len(all_adapters),
            "eval_domains": len(eval_domains),
            "smoke_test": IS_SMOKE,
        },
        "base_model_ppl": base_ppls,
        "single_expert_baselines": single_baselines,
        "composition_results": {
            str(n): {
                "adapters": res["adapters"],
                "per_domain_ppl": res["per_domain_ppl"],
                "mean_degradation_pct": round(mean_degs.get(n, 0), 2),
            }
            for n, res in composition_results.items()
        },
        "kill_criteria": {
            "K1_max_degradation_lt_10pct": {
                "value": round(max_degradation, 2),
                "threshold": 10.0,
                "per_domain": degradations,
                "pass": k1_pass,
            },
            "K2_not_superlinear": {
                "mean_degradation_by_n": {str(k): round(v, 2) for k, v in mean_degs.items()},
                "superlinear_detected": superlinear,
                "pass": k2_pass,
            },
            "K3_worse_than_base_lt_20pct": {
                "value": round(worse_pct, 1),
                "threshold": 20.0,
                "n_worse": n_worse_than_base,
                "n_evaluated": n_evaluated,
                "worse_domains": worse_domains,
                "pass": k3_pass,
            },
        },
        "verdict": "PASS" if (k1_pass and k2_pass and k3_pass) else "FAIL",
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Print summary
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    for n, res in sorted(composition_results.items()):
        mean_deg = mean_degs.get(n, 0)
        log(f"  N={n}: mean degradation = {mean_deg:+.2f}%")

    log(f"\nK1 (max degradation <10%): {'PASS' if k1_pass else 'FAIL'} — {max_degradation:.2f}%")
    log(f"K2 (not superlinear): {'PASS' if k2_pass else 'FAIL'} — superlinear={superlinear}")
    log(f"K3 (worse than base <20%): {'PASS' if k3_pass else 'FAIL'} — {worse_pct:.0f}% ({n_worse_than_base}/{n_evaluated})")
    log(f"\nVERDICT: {aggregate['verdict']}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")

    # Save results
    out_file = RESULTS_DIR / "composition_quality_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
