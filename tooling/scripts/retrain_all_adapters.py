#!/usr/bin/env python3
"""Retrain all 50 pilot adapters that are missing on RunPod.

Runs on RunPod. Checks /workspace/llm/adapters/ for existing adapters,
trains missing ones using pilot50_train.py subprocess isolation.

Usage (submitted via gpu_queue):
    uv run python3 tools/gpu_queue.py submit scripts/retrain_all_adapters.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "distillation"
ADAPTER_DIR = REPO_ROOT / "adapters"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "pilot50_train.py"

BASE_MODEL = "Qwen/Qwen2.5-7B"
RANK = 16
STEPS = 300
LR = 2e-4


def adapter_is_complete(domain: str) -> bool:
    """Check if adapter has both config and weights."""
    d = ADAPTER_DIR / domain
    return (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists()


def main():
    # Find all domains with training data
    train_files = sorted(DATA_DIR.glob("*/train.jsonl"))
    all_domains = [f.parent.name for f in train_files]

    complete = [d for d in all_domains if adapter_is_complete(d)]
    missing = [d for d in all_domains if not adapter_is_complete(d)]

    print(f"=" * 70)
    print(f"Adapter Retrain — Full 50-Domain Recovery")
    print(f"=" * 70)
    print(f"  Total domains: {len(all_domains)}")
    print(f"  Already complete: {len(complete)}")
    print(f"  Need training: {len(missing)}")
    print(f"  Complete: {', '.join(complete)}")
    print(f"  Missing: {', '.join(missing)}")

    if not missing:
        print("\nAll adapters present! Nothing to do.")
        return

    est_min = len(missing) * 15
    print(f"\n  Estimated time: {est_min} min ({est_min / 60:.1f} hr)")
    print(f"  Estimated cost: ${est_min / 60 * 0.34:.2f}")
    print()

    trained = 0
    failed = []
    total_time = 0.0

    for i, domain in enumerate(missing):
        print(f"\n[{i + 1}/{len(missing)}] Training {domain}...", flush=True)
        start = time.time()

        # Use pilot50_train.py in single-domain subprocess mode
        result = subprocess.run(
            [
                sys.executable, str(TRAIN_SCRIPT),
                "--base", BASE_MODEL,
                "--rank", str(RANK),
                "--steps", str(STEPS),
                "--lr", str(LR),
                "--single",
                "--domains", domain,
            ],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            timeout=10800,  # 3 hr timeout per domain (300 steps @ ~29s/step ≈ 2.4hr)
        )

        elapsed = time.time() - start
        total_time += elapsed

        if result.returncode == 0 and adapter_is_complete(domain):
            trained += 1
            print(f"  {domain}: OK in {elapsed:.0f}s", flush=True)
        else:
            print(f"  {domain}: FAILED (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
            failed.append(domain)

    # Final report
    print(f"\n{'=' * 70}")
    print(f"RETRAIN COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Trained: {trained}/{len(missing)}")
    print(f"  Failed: {len(failed)}: {', '.join(failed)}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Cost: ${total_time / 3600 * 0.34:.2f}")

    # Full verification
    final_complete = sum(1 for d in all_domains if adapter_is_complete(d))
    print(f"\n  Final adapter count: {final_complete}/{len(all_domains)}")

    # Save results
    results = {
        "total_domains": len(all_domains),
        "previously_complete": len(complete),
        "trained": trained,
        "failed": failed,
        "final_complete": final_complete,
        "total_time_s": total_time,
        "cost_usd": total_time / 3600 * 0.34,
    }
    results_path = REPO_ROOT / "results" / "retrain_all_adapters.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
