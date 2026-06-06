#!/usr/bin/env python3
"""Run ONLY the cold-start ternary no-weight-decay condition and merge into results.json.

This is a targeted runner to avoid re-running the 4 existing conditions (~35 min).
It imports the model/training code from run_experiment.py and runs only the new
phase_cold_start_ternary_no_wd, then merges results into the existing results.json.
"""

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx

# Memory limits (MANDATORY per CODING_GUIDELINES)
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Import everything needed from run_experiment
from run_experiment import (
    phase_load_data,
    phase_cold_start_ternary_no_wd,
    log_memory,
)


def main():
    t0 = time.time()
    log_memory("start")

    # Load data
    train_tokens, val_tokens = phase_load_data()
    log_memory("after-data")

    # Run only the new condition
    cold_nowd_results = phase_cold_start_ternary_no_wd(train_tokens, val_tokens)
    log_memory("after-cold-ternary-nowd")

    elapsed = time.time() - t0
    print(f"\nNew condition completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Load existing results and merge
    existing = json.loads(RESULTS_FILE.read_text())
    existing["cold_start_ternary_no_wd"] = cold_nowd_results

    # Compute ablation analysis
    fp32_ppl = existing["fp32_baseline"]["ppl"]
    cold_ppl = existing["cold_start_ternary"]["ppl"]
    cold_nowd_ppl = cold_nowd_results["ppl"]
    best_warm_ppl = existing["best_warm_start"]["ppl"]

    cold_nowd_ratio = cold_nowd_ppl / fp32_ppl
    wd_effect = (cold_ppl - cold_nowd_ppl) / cold_ppl * 100
    improvement_vs_cold = (cold_ppl - best_warm_ppl) / cold_ppl * 100
    improvement_vs_cold_nowd = (cold_nowd_ppl - best_warm_ppl) / cold_nowd_ppl * 100

    existing["weight_decay_ablation"] = {
        "cold_wd001_ppl": cold_ppl,
        "cold_wd000_ppl": cold_nowd_ppl,
        "cold_wd000_ratio_vs_fp32": round(cold_nowd_ratio, 4),
        "warm_best_ppl": best_warm_ppl,
        "wd_effect_on_cold_pct": round(wd_effect, 2),
        "warm_improvement_vs_cold_wd001_pct": round(improvement_vs_cold, 2),
        "warm_improvement_vs_cold_wd000_pct": round(improvement_vs_cold_nowd, 2),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("WEIGHT DECAY ABLATION RESULTS")
    print("=" * 60)
    print(f"  FP32 baseline:                {fp32_ppl:.2f} (1.000x)")
    print(f"  Cold-start ternary (wd=0.01): {cold_ppl:.2f} ({cold_ppl / fp32_ppl:.3f}x)")
    print(f"  Cold-start ternary (wd=0.0):  {cold_nowd_ppl:.2f} ({cold_nowd_ratio:.3f}x)")
    print(f"  Best warm-start:              {best_warm_ppl:.2f} ({best_warm_ppl / fp32_ppl:.3f}x)")
    print(f"\n  Weight decay effect on cold-start PPL: {wd_effect:+.1f}%")
    print(f"  Warm-start improvement vs cold (wd=0.01): {improvement_vs_cold:.1f}%")
    print(f"  Warm-start improvement vs cold (wd=0.0):  {improvement_vs_cold_nowd:.1f}%")

    if abs(wd_effect) < 3.0:
        print("\n  INTERPRETATION: Weight decay has minimal effect (<3% PPL change).")
        print("  The warm-start advantage is NOT explained by weight decay removal.")
    elif wd_effect > 0 and cold_nowd_ppl < cold_ppl * 0.97:
        print(f"\n  INTERPRETATION: Removing weight decay improves cold-start by {wd_effect:.1f}%.")
        if improvement_vs_cold_nowd > 3.0:
            print("  But warm-start STILL beats cold-start-no-wd -- advantage is real.")
        else:
            print("  WARNING: Warm-start advantage may be mostly from weight decay removal!")
    else:
        print(f"\n  INTERPRETATION: Removing weight decay changes cold-start PPL by {wd_effect:+.1f}%.")

    # Save merged results
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nResults merged into {RESULTS_FILE}")


if __name__ == "__main__":
    main()
