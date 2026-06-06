#!/usr/bin/env python3
"""GPU Latency Benchmark: Validate N-independent latency on production GPU.

Tests:
1. Base model latency (baseline)
2. Pre-merge latency at N=5,10,20,50 (should be ~0% overhead)
3. Dynamic LoRA application at N=5,10,20,50, k=1,2 (overhead should not scale with N)

Uses REAL trained adapters from the distillation pilot (not synthetic weights).
All measurements on RTX 4090 with CUDA, proper warmup and timing.

Usage (on RunPod):
    cd /workspace/llm
    python scripts/gpu_latency_bench.py
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
import torch.cuda

REPO_ROOT = Path(__file__).parent.parent.parent
ADAPTER_DIR = REPO_ROOT / "adapters"
RESULTS_DIR = REPO_ROOT / "results"
HF_CACHE = "/workspace/hf_cache"

# Measurement parameters
WARMUP_ITERS = 20
MEASURE_ITERS = 100
SEQ_LENGTHS = [64, 256]
BATCH_SIZE = 1
N_VALUES = [5, 10, 20, 50]
K_VALUES = [1, 2]


def get_adapter_dirs(n: int) -> list[Path]:
    """Get exactly n adapter directories, sorted for reproducibility."""
    all_adapters = sorted(
        d for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )
    if len(all_adapters) < n:
        raise ValueError(f"Need {n} adapters, found {len(all_adapters)}")
    return all_adapters[:n]


def measure_latency(model, tokenizer, seq_len: int,
                    warmup: int = WARMUP_ITERS,
                    iters: int = MEASURE_ITERS) -> dict:
    """Measure forward pass latency with proper CUDA synchronization."""
    # Create fixed input
    input_ids = torch.randint(
        100, tokenizer.vocab_size - 100,
        (BATCH_SIZE, seq_len), device=model.device
    )

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Measure with CUDA events for precise GPU timing
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(input_ids)
            end_event.record()

            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # ms

    return {
        "mean_ms": round(sum(times) / len(times), 3),
        "median_ms": round(sorted(times)[len(times) // 2], 3),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "std_ms": round((sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5, 3),
    }


def measure_merge_time(base_model, adapter_dirs: list[Path]) -> float:
    """Measure time to merge N adapters into base weights."""
    from peft import PeftModel

    torch.cuda.synchronize()
    start = time.perf_counter()

    # Load first adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_dirs[0]))
    # Load remaining as additional adapters
    for i, ad in enumerate(adapter_dirs[1:], 1):
        adapter_name = f"expert_{i}"
        model.load_adapter(str(ad), adapter_name=adapter_name)

    # Set equal weights and merge
    adapters = list(model.peft_config.keys())
    weights = [1.0 / len(adapters)] * len(adapters)

    model.add_weighted_adapter(
        adapters=adapters,
        weights=weights,
        adapter_name="merged",
        combination_type="linear",
    )
    model.set_adapter("merged")
    merged_model = model.merge_and_unload()

    torch.cuda.synchronize()
    merge_time = (time.perf_counter() - start) * 1000  # ms

    return merge_time, merged_model


def measure_dynamic_latency(base_model, tokenizer, adapter_dirs: list[Path],
                            k: int, seq_len: int) -> dict:
    """Measure dynamic top-k LoRA application latency.

    Strategy: load k adapters, apply them at inference time via PEFT.
    This simulates the dynamic composition path without fused kernels.
    """
    from peft import PeftModel

    # Pick k random adapters (fixed seed for reproducibility)
    random.seed(42)
    selected = random.sample(adapter_dirs, k)

    # Load base + k adapters
    model = PeftModel.from_pretrained(base_model, str(selected[0]))
    for i, ad in enumerate(selected[1:], 1):
        model.load_adapter(str(ad), adapter_name=f"expert_{i}")

    # Create merged adapter from selected k
    adapters = list(model.peft_config.keys())
    weights = [1.0 / k] * k
    model.add_weighted_adapter(
        adapters=adapters,
        weights=weights,
        adapter_name="active",
        combination_type="linear",
    )
    model.set_adapter("active")

    # Now measure forward pass with the active LoRA adapter (NOT merged into weights)
    result = measure_latency(model, tokenizer, seq_len)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="GPU Latency Benchmark")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", default=str(RESULTS_DIR / "gpu_latency_benchmark.json"))
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=MEASURE_ITERS)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"GPU Latency Benchmark")
    print(f"  Base model: {args.base}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Warmup: {args.warmup}, Measure: {args.iters}")
    print(f"  N values: {N_VALUES}")
    print(f"  Sequence lengths: {SEQ_LENGTHS}")
    print()

    # Load base model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, cache_dir=HF_CACHE, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, device_map={"": 0},
        cache_dir=HF_CACHE, trust_remote_code=True)

    results = {
        "base_model": args.base,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "warmup": args.warmup,
            "iters": args.iters,
            "batch_size": BATCH_SIZE,
            "n_values": N_VALUES,
            "k_values": K_VALUES,
            "seq_lengths": SEQ_LENGTHS,
        },
        "base_latency": {},
        "premerge": {},
        "dynamic": {},
    }

    # 1. Base model latency
    print("=" * 60)
    print("PHASE 1: Base model latency")
    for seq_len in SEQ_LENGTHS:
        print(f"  seq_len={seq_len}...")
        lat = measure_latency(base_model, tokenizer, seq_len,
                              warmup=args.warmup, iters=args.iters)
        results["base_latency"][str(seq_len)] = lat
        print(f"    mean={lat['mean_ms']:.1f}ms, median={lat['median_ms']:.1f}ms, "
              f"p95={lat['p95_ms']:.1f}ms")

    # 2. Pre-merge latency at each N
    print()
    print("=" * 60)
    print("PHASE 2: Pre-merge latency (merge N adapters, then measure)")
    for n in N_VALUES:
        print(f"\n  N={n}:")
        adapter_dirs = get_adapter_dirs(n)

        # Measure merge time and get merged model
        merge_time_ms, merged_model = measure_merge_time(base_model, adapter_dirs)
        print(f"    Merge time: {merge_time_ms:.0f}ms")

        results["premerge"][str(n)] = {"merge_time_ms": round(merge_time_ms, 1)}

        for seq_len in SEQ_LENGTHS:
            lat = measure_latency(merged_model, tokenizer, seq_len,
                                  warmup=args.warmup, iters=args.iters)
            base_lat = results["base_latency"][str(seq_len)]["median_ms"]
            overhead_pct = (lat["median_ms"] - base_lat) / base_lat * 100
            results["premerge"][str(n)][str(seq_len)] = {
                **lat,
                "overhead_pct": round(overhead_pct, 2),
            }
            print(f"    seq_len={seq_len}: median={lat['median_ms']:.1f}ms "
                  f"(overhead: {overhead_pct:+.1f}%)")

        # Cleanup: base_model was consumed by PeftModel wrapping, delete both
        del base_model
        del merged_model
        gc.collect()
        torch.cuda.empty_cache()

        # Re-load base model for next iteration (merge_and_unload is destructive)
        print("    Reloading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float16, device_map={"": 0},
            cache_dir=HF_CACHE, trust_remote_code=True)

    # 3. Dynamic top-k latency
    print()
    print("=" * 60)
    print("PHASE 3: Dynamic LoRA latency (PEFT adapter, NOT merged into weights)")
    for n in N_VALUES:
        results["dynamic"][str(n)] = {}
        adapter_dirs = get_adapter_dirs(n)

        for k in K_VALUES:
            if k > n:
                continue
            print(f"\n  N={n}, k={k}:")
            results["dynamic"][str(n)][str(k)] = {}

            for seq_len in SEQ_LENGTHS:
                lat = measure_dynamic_latency(
                    base_model, tokenizer, adapter_dirs, k, seq_len)
                base_lat = results["base_latency"][str(seq_len)]["median_ms"]
                overhead_pct = (lat["median_ms"] - base_lat) / base_lat * 100
                results["dynamic"][str(n)][str(k)][str(seq_len)] = {
                    **lat,
                    "overhead_pct": round(overhead_pct, 2),
                }
                print(f"    seq_len={seq_len}: median={lat['median_ms']:.1f}ms "
                      f"(overhead: {overhead_pct:+.1f}%)")

                # Re-load base model (PeftModel wrapping is destructive to state)
                del base_model
                gc.collect()
                torch.cuda.empty_cache()
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base, dtype=torch.float16, device_map={"": 0},
                    cache_dir=HF_CACHE, trust_remote_code=True)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print()

    # Pre-merge summary table
    print("Pre-merge overhead vs base (median, seq_len=256):")
    print(f"  {'N':>5} | {'Latency (ms)':>12} | {'Overhead':>10} | {'Merge (ms)':>10}")
    print(f"  {'---':>5}-+-{'---':>12}-+-{'---':>10}-+-{'---':>10}")
    base_256 = results["base_latency"]["256"]["median_ms"]
    print(f"  {'base':>5} | {base_256:>12.1f} | {'--':>10} | {'--':>10}")
    for n in N_VALUES:
        pm = results["premerge"][str(n)]
        lat = pm["256"]["median_ms"]
        ovh = pm["256"]["overhead_pct"]
        mt = pm["merge_time_ms"]
        print(f"  {n:>5} | {lat:>12.1f} | {ovh:>+9.1f}% | {mt:>10.0f}")

    # Kill criteria assessment
    print()
    max_premerge_overhead = max(
        results["premerge"][str(n)]["256"]["overhead_pct"] for n in N_VALUES
    )
    premerge_pass = max_premerge_overhead <= 5.0

    # Check if dynamic overhead scales with N
    dynamic_overheads_k1 = []
    for n in N_VALUES:
        if "1" in results["dynamic"][str(n)]:
            ovh = results["dynamic"][str(n)]["1"]["256"]["overhead_pct"]
            dynamic_overheads_k1.append((n, ovh))

    if len(dynamic_overheads_k1) >= 2:
        # Linear regression slope: if positive and significant, overhead scales with N
        n_vals = [x[0] for x in dynamic_overheads_k1]
        ovh_vals = [x[1] for x in dynamic_overheads_k1]
        n_mean = sum(n_vals) / len(n_vals)
        ovh_mean = sum(ovh_vals) / len(ovh_vals)
        numerator = sum((n - n_mean) * (o - ovh_mean) for n, o in zip(n_vals, ovh_vals))
        denominator = sum((n - n_mean) ** 2 for n in n_vals)
        slope = numerator / denominator if denominator > 0 else 0
        # slope = overhead_pct per additional expert
        dynamic_scales = slope > 0.1  # >0.1% per expert = significant scaling
    else:
        dynamic_scales = False

    dynamic_pass = not dynamic_scales

    verdict = "PASS" if (premerge_pass and dynamic_pass) else "KILL"

    results["kill_criteria"] = {
        "K1_premerge_max_overhead_pct": round(max_premerge_overhead, 2),
        "K1_threshold": 5.0,
        "K1_pass": premerge_pass,
        "K2_dynamic_scales_with_N": dynamic_scales,
        "K2_slope_pct_per_expert": round(slope, 4) if len(dynamic_overheads_k1) >= 2 else None,
        "K2_pass": dynamic_pass,
        "verdict": verdict,
    }

    print(f"Kill Criteria:")
    print(f"  K1: Pre-merge max overhead = {max_premerge_overhead:+.1f}% "
          f"(threshold: <=5%) -> {'PASS' if premerge_pass else 'KILL'}")
    print(f"  K2: Dynamic scales with N = {dynamic_scales} "
          f"(slope: {slope:.4f}%/expert) -> {'PASS' if dynamic_pass else 'KILL'}")
    print(f"  Verdict: {verdict}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
