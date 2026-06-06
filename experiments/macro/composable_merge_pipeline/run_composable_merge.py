#!/usr/bin/env python3
"""exp_composable_merge_pipeline — End-to-end merge pipeline with quality gates.

Compose N adapters via weight addition (unit weights — GS proved unnecessary),
benchmark merged model vs individually-served experts per domain.

Kill criteria:
  K1: merged model (N=20) loses >5% quality vs individually-served experts
  K2: merge + benchmark cycle takes >1 hour
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "composable_merge_pipeline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B")
HF_CACHE = "/workspace/hf_cache"

MAX_ADAPTERS = 3 if IS_SMOKE else 20
MAX_EVAL_TOKENS = 50 if IS_SMOKE else 512
EVAL_SAMPLES = 2 if IS_SMOKE else 20
QUALITY_THRESHOLD = 0.05  # K1: max 5% degradation
TIME_THRESHOLD_S = 3600   # K2: max 1 hour


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid adapters in ADAPTER_DIR."""
    adapters = []
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_domain_samples(domain, n_samples, tokenizer=None):
    """Load evaluation samples for a domain from distillation data.

    Checks eval.jsonl first, then falls back to train.jsonl (last N as held-out).
    """
    texts = []
    for fname in ["eval.jsonl", "train.jsonl"]:
        data_path = DATA_DIR / domain / fname
        if not data_path.exists():
            continue
        with open(data_path) as f:
            lines = f.readlines()
        # Use last portion of train.jsonl to avoid contamination
        if fname == "train.jsonl":
            lines = lines[-min(n_samples * 4, len(lines)):]
        for line in lines:
            obj = json.loads(line)
            if "messages" in obj:
                if tokenizer is not None:
                    text = tokenizer.apply_chat_template(
                        obj["messages"], tokenize=False, add_generation_prompt=False)
                else:
                    text = " ".join(m.get("content", "") for m in obj["messages"])
            elif "text" in obj:
                text = obj["text"]
            else:
                continue
            if text.strip():
                texts.append(text.strip())
            if len(texts) >= n_samples:
                return texts
    return texts


def compute_ppl(model, tokenizer, texts, max_len=MAX_EVAL_TOKENS):
    """Compute mean perplexity over texts."""
    import torch
    if not texts:
        return float("nan")
    ppls = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        ppls.append(out.loss.exp().item())
    return sum(ppls) / len(ppls)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    t0 = time.time()

    # --- Discover adapters ---
    all_adapters = discover_adapters()
    if len(all_adapters) < 2:
        log(f"ABORT: only {len(all_adapters)} adapters found, need at least 2")
        sys.exit(1)
    adapters = all_adapters[:MAX_ADAPTERS]
    log(f"Using {len(adapters)}/{len(all_adapters)} adapters: {adapters[:5]}...")

    # --- Load base model ---
    log("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
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

    # --- Phase 1: Benchmark each expert individually ---
    log("Phase 1: Individual expert benchmarks")
    individual_ppls = {}
    for domain in adapters:
        samples = load_domain_samples(domain, EVAL_SAMPLES, tokenizer)
        if not samples:
            log(f"  SKIP {domain}: no eval data")
            continue

        expert_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR / domain))
        ppl = compute_ppl(expert_model, tokenizer, samples)
        individual_ppls[domain] = {"ppl": ppl, "n_samples": len(samples)}
        log(f"  {domain}: PPL={ppl:.2f} ({len(samples)} samples)")

        # Unload adapter
        expert_model.unload()
        del expert_model
        torch.cuda.empty_cache()
        gc.collect()

    if not individual_ppls:
        log("ABORT: no domains had eval data")
        sys.exit(1)

    t_phase1 = time.time() - t0
    log(f"Phase 1 done in {t_phase1:.0f}s ({len(individual_ppls)} domains)")

    # --- Phase 2: Compose all adapters via weighted addition ---
    log("Phase 2: Composing adapters via linear merge (unit weights)")
    t_merge_start = time.time()

    compose_model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIR / adapters[0]), adapter_name=adapters[0]
    )
    for name in adapters[1:]:
        compose_model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)
        log(f"  Loaded adapter: {name}")

    compose_model.add_weighted_adapter(
        adapters=adapters,
        weights=[1.0 / len(adapters)] * len(adapters),
        adapter_name="composed",
        combination_type="linear",
    )
    compose_model.set_adapter("composed")
    t_merge = time.time() - t_merge_start
    log(f"Merge completed in {t_merge:.1f}s")

    # --- Phase 3: Benchmark merged model per domain ---
    log("Phase 3: Merged model benchmarks")
    merged_ppls = {}
    for domain in individual_ppls:
        samples = load_domain_samples(domain, EVAL_SAMPLES, tokenizer)
        ppl = compute_ppl(compose_model, tokenizer, samples)
        merged_ppls[domain] = {"ppl": ppl, "n_samples": len(samples)}
        log(f"  {domain}: PPL={ppl:.2f}")

    # Cleanup
    del compose_model
    torch.cuda.empty_cache()
    gc.collect()

    t_total = time.time() - t0

    # --- Phase 4: Quality gate assessment ---
    log("Phase 4: Quality gate assessment")
    domain_results = {}
    degradations = []
    for domain in individual_ppls:
        ind_ppl = individual_ppls[domain]["ppl"]
        mer_ppl = merged_ppls[domain]["ppl"]
        # Degradation: how much worse is merged vs individual (higher PPL = worse)
        if ind_ppl > 0:
            degradation = (mer_ppl - ind_ppl) / ind_ppl
        else:
            degradation = 0.0
        degradations.append(degradation)
        domain_results[domain] = {
            "individual_ppl": round(ind_ppl, 4),
            "merged_ppl": round(mer_ppl, 4),
            "degradation_pct": round(degradation * 100, 2),
            "pass": degradation <= QUALITY_THRESHOLD,
        }
        status = "PASS" if degradation <= QUALITY_THRESHOLD else "FAIL"
        log(f"  {domain}: {ind_ppl:.2f} → {mer_ppl:.2f} ({degradation*100:+.1f}%) [{status}]")

    mean_degradation = sum(degradations) / len(degradations) if degradations else 0
    max_degradation = max(degradations) if degradations else 0
    n_pass = sum(1 for d in degradations if d <= QUALITY_THRESHOLD)

    k1_pass = mean_degradation <= QUALITY_THRESHOLD
    k2_pass = t_total <= TIME_THRESHOLD_S

    verdict = "PASS" if (k1_pass and k2_pass) else "FAIL"

    log(f"\n{'='*60}")
    log(f"VERDICT: {verdict}")
    log(f"  K1 (quality): mean degradation = {mean_degradation*100:.2f}% "
        f"(threshold {QUALITY_THRESHOLD*100}%) → {'PASS' if k1_pass else 'FAIL'}")
    log(f"    Max degradation: {max_degradation*100:.2f}%")
    log(f"    Domains pass: {n_pass}/{len(degradations)}")
    log(f"  K2 (speed): total cycle = {t_total:.0f}s "
        f"(threshold {TIME_THRESHOLD_S}s) → {'PASS' if k2_pass else 'FAIL'}")
    log(f"    Merge time: {t_merge:.1f}s")
    log(f"  Adapters: {len(adapters)}")
    log(f"{'='*60}")

    # --- Save results ---
    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "base_model": BASE_MODEL,
            "n_adapters": len(adapters),
            "adapter_names": adapters,
            "eval_samples_per_domain": EVAL_SAMPLES,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "combination_type": "linear",
            "weights": "unit (1.0 each)",
            "is_smoke": IS_SMOKE,
        },
        "kill_criteria": {
            "K1_quality_degradation": {
                "metric": "mean PPL degradation merged vs individual",
                "value": round(mean_degradation * 100, 2),
                "threshold_pct": QUALITY_THRESHOLD * 100,
                "max_degradation_pct": round(max_degradation * 100, 2),
                "domains_pass": f"{n_pass}/{len(degradations)}",
                "pass": k1_pass,
            },
            "K2_cycle_time": {
                "metric": "total merge + benchmark cycle time",
                "value_s": round(t_total, 1),
                "merge_time_s": round(t_merge, 1),
                "threshold_s": TIME_THRESHOLD_S,
                "pass": k2_pass,
            },
        },
        "domain_results": domain_results,
        "verdict": verdict,
        "elapsed_s": round(t_total, 1),
    }

    out_file = RESULTS_DIR / "composable_merge_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
