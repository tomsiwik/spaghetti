#!/usr/bin/env python3
"""End-to-end inference throughput of composed SOLE model.

Measures real-world serving performance of SOLE on RunPod A5000.
Tests pre-merged mode (fastest, recommended by micro findings: -4% overhead).

Key questions:
- Does pre-merged SOLE match base Qwen2.5-7B throughput?
- How does batch size affect throughput?
- What are p50/p99 latencies for first-token and full generation?

Kill criteria:
- K1: Pre-merged SOLE throughput <80% of base model (composition overhead too high)
- K2: Pre-merged SOLE first-token latency >2x base model (unacceptable UX)
- K3: Throughput degrades >10% at batch_size=32 vs batch_size=1 (scaling problem)

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
RESULTS_DIR = REPO_ROOT / "results" / "sole_inference_throughput"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Benchmark parameters
BATCH_SIZES = [1, 4] if IS_SMOKE else [1, 4, 16, 32]
PROMPT_LENGTH = 64 if IS_SMOKE else 128  # tokens
GEN_LENGTH = 32 if IS_SMOKE else 128  # tokens to generate
N_WARMUP = 2 if IS_SMOKE else 5
N_ITERATIONS = 3 if IS_SMOKE else 20


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid adapters."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def build_prompts(tokenizer, n_prompts, prompt_length):
    """Build diverse prompts from domain data."""
    prompts = []
    domains = ["python", "math", "medical", "bash", "legal", "creative-fiction"]
    for domain in domains:
        f = DATA_DIR / domain / "train.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                record = json.loads(line)
                if "messages" in record:
                    text = tokenizer.apply_chat_template(
                        record["messages"][:1], tokenize=False, add_generation_prompt=True)
                elif "text" in record:
                    text = record["text"][:200]
                else:
                    continue
                prompts.append(text)
                if len(prompts) >= n_prompts * 2:
                    break
        if len(prompts) >= n_prompts * 2:
            break
    random.shuffle(prompts)
    return prompts[:n_prompts]


def benchmark_model(model, tokenizer, prompts, batch_sizes, gen_length,
                    n_warmup, n_iterations, label="model"):
    """Run throughput and latency benchmarks."""
    import torch
    import numpy as np

    results = {}
    model.eval()

    for bs in batch_sizes:
        log(f"  Benchmarking {label} @ batch_size={bs}")
        batch_prompts = prompts[:bs]

        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(model.device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=gen_length,
                               do_sample=False, use_cache=True)
            torch.cuda.synchronize()

        # Benchmark: first-token latency
        first_token_latencies = []
        for _ in range(n_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1,
                               do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            first_token_latencies.append((time.perf_counter() - t0) * 1000)  # ms

        # Benchmark: full generation throughput
        gen_latencies = []
        total_tokens = []
        for _ in range(n_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=gen_length,
                                        do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            gen_latencies.append(elapsed * 1000)  # ms
            # Count generated tokens
            n_gen = output.shape[1] - inputs.input_ids.shape[1]
            total_tokens.append(n_gen * bs)

        # Compute metrics
        first_token_p50 = float(np.percentile(first_token_latencies, 50))
        first_token_p99 = float(np.percentile(first_token_latencies, 99))
        gen_p50 = float(np.percentile(gen_latencies, 50))
        gen_p99 = float(np.percentile(gen_latencies, 99))
        avg_tokens = sum(total_tokens) / len(total_tokens)
        avg_gen_time = sum(gen_latencies) / len(gen_latencies) / 1000  # seconds
        throughput = avg_tokens / avg_gen_time if avg_gen_time > 0 else 0

        results[bs] = {
            "batch_size": bs,
            "first_token_p50_ms": first_token_p50,
            "first_token_p99_ms": first_token_p99,
            "generation_p50_ms": gen_p50,
            "generation_p99_ms": gen_p99,
            "throughput_tok_per_s": throughput,
            "avg_tokens_per_batch": avg_tokens,
        }
        log(f"    FT p50={first_token_p50:.1f}ms p99={first_token_p99:.1f}ms "
            f"Gen p50={gen_p50:.1f}ms Throughput={throughput:.1f} tok/s")

    return results


def run_experiment():
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log("=" * 70)
    log("SOLE Inference Throughput Experiment")
    log("=" * 70)

    adapters = discover_adapters()
    log(f"Found {len(adapters)} adapters")
    if IS_SMOKE:
        adapters = adapters[:5]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build test prompts
    max_prompts = max(BATCH_SIZES) * 2
    prompts = build_prompts(tokenizer, max_prompts, PROMPT_LENGTH)
    log(f"Built {len(prompts)} test prompts")

    # ============================================================
    # PHASE 1: Benchmark BASE model
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 1: Base model throughput")
    log("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE,
        torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )

    base_results = benchmark_model(model, tokenizer, prompts, BATCH_SIZES,
                                   GEN_LENGTH, N_WARMUP, N_ITERATIONS, "base")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE 2: Benchmark SOLE pre-merged model
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 2: SOLE pre-merged model throughput")
    log("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE,
        torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )

    log(f"Pre-merging {len(adapters)} adapters...")
    merge_start = time.time()
    n_merged = 0
    for adapter_name in adapters:
        adapter_path = ADAPTER_DIR / adapter_name
        try:
            peft_model = PeftModel.from_pretrained(model, str(adapter_path))
            peft_model.merge_and_unload()
            model = peft_model.base_model
            n_merged += 1
        except Exception as e:
            log(f"  WARN: failed to merge {adapter_name}: {e}")
    merge_time = time.time() - merge_start
    log(f"Merged {n_merged} adapters in {merge_time:.1f}s")

    sole_results = benchmark_model(model, tokenizer, prompts, BATCH_SIZES,
                                   GEN_LENGTH, N_WARMUP, N_ITERATIONS, "SOLE")

    # GPU memory stats
    gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
    gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # K1: Throughput comparison at bs=1
    base_tp_1 = base_results.get(1, {}).get("throughput_tok_per_s", 1)
    sole_tp_1 = sole_results.get(1, {}).get("throughput_tok_per_s", 0)
    tp_ratio = sole_tp_1 / base_tp_1 if base_tp_1 > 0 else 0

    # K2: First-token latency comparison at bs=1
    base_ft = base_results.get(1, {}).get("first_token_p50_ms", 1)
    sole_ft = sole_results.get(1, {}).get("first_token_p50_ms", 0)
    ft_ratio = sole_ft / base_ft if base_ft > 0 else float("inf")

    # K3: Throughput scaling bs=1 vs bs=32
    sole_tp_max = sole_results.get(max(BATCH_SIZES), {}).get("throughput_tok_per_s", 0)
    scaling = sole_tp_max / sole_tp_1 if sole_tp_1 > 0 else 0
    # "degrades" means per-request throughput at high batch < low batch
    # Actually throughput should increase with batch size; the kill is if it drops
    tp_per_req_1 = sole_tp_1
    tp_per_req_max = sole_tp_max / max(BATCH_SIZES) if max(BATCH_SIZES) > 0 else 0
    degradation = (tp_per_req_1 - tp_per_req_max) / tp_per_req_1 * 100 if tp_per_req_1 > 0 else 0

    k1_result = "KILLED" if tp_ratio < 0.80 else "SURVIVES"
    k2_result = "KILLED" if ft_ratio > 2.0 else "SURVIVES"
    k3_result = "KILLED" if degradation > 10 else "SURVIVES"

    log(f"\nKILL CRITERIA:")
    log(f"  K1 (SOLE throughput <80% of base @ bs=1): {k1_result} "
        f"(ratio={tp_ratio:.2f}, base={base_tp_1:.1f} sole={sole_tp_1:.1f} tok/s)")
    log(f"  K2 (first-token latency >2x base @ bs=1): {k2_result} "
        f"(ratio={ft_ratio:.2f}, base={base_ft:.1f}ms sole={sole_ft:.1f}ms)")
    log(f"  K3 (per-request throughput degrades >10% at max batch): {k3_result} "
        f"(degradation={degradation:.1f}%)")

    overall = "KILLED" if "KILLED" in [k1_result, k2_result, k3_result] else "SURVIVES"
    log(f"\n  OVERALL: {overall}")

    # Per-batch-size comparison table
    log("\n  Batch | Base tok/s | SOLE tok/s | Ratio  | SOLE FT p50")
    log("  ------|-----------|-----------|--------|------------")
    for bs in BATCH_SIZES:
        b = base_results.get(bs, {})
        s = sole_results.get(bs, {})
        ratio = s.get("throughput_tok_per_s", 0) / b.get("throughput_tok_per_s", 1) if b.get("throughput_tok_per_s", 1) > 0 else 0
        log(f"  {bs:5d} | {b.get('throughput_tok_per_s', 0):9.1f} | {s.get('throughput_tok_per_s', 0):9.1f} | {ratio:6.2f} | {s.get('first_token_p50_ms', 0):.1f}ms")

    results = {
        "experiment": "sole_inference_throughput",
        "n_adapters_merged": n_merged,
        "merge_time_s": merge_time,
        "base_model": BASE_MODEL,
        "gpu_memory_gb": {"allocated": gpu_mem_allocated, "reserved": gpu_mem_reserved},
        "batch_sizes_tested": BATCH_SIZES,
        "gen_length": GEN_LENGTH,
        "n_iterations": N_ITERATIONS,
        "base_results": {str(k): v for k, v in base_results.items()},
        "sole_results": {str(k): v for k, v in sole_results.items()},
        "kill_criteria": {
            "K1_throughput_ratio": tp_ratio,
            "K1_threshold": 0.80,
            "K1_result": k1_result,
            "K2_first_token_ratio": ft_ratio,
            "K2_threshold": 2.0,
            "K2_result": k2_result,
            "K3_degradation_pct": degradation,
            "K3_threshold": 10.0,
            "K3_result": k3_result,
        },
        "overall": overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_experiment()
