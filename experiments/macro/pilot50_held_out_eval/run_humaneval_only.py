#!/usr/bin/env python3
"""Run ONLY the HumanEval evaluation (MMLU already completed).

Standalone script for resubmission after the first run_all_eval failed on HumanEval.
Uses base cache if available, otherwise evaluates base model first.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = Path("/workspace/llm/results/held_out_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_CACHE = RESULTS_DIR / "humaneval_results_base_cache.json"


def main():
    t0 = time.time()
    print(f"=== HumanEval-Only Resubmission ===")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_humaneval.py"),
        "--adapter", "python",
        "--out", str(RESULTS_DIR / "humaneval_results.json"),
    ]

    # Use cached base if previous run produced it
    if BASE_CACHE.exists():
        cmd += ["--base-cache", str(BASE_CACHE)]
        print(f"Using cached base results from {BASE_CACHE}")

    print(f"CMD: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd="/workspace/llm")

    elapsed = time.time() - t0
    status = "OK" if r.returncode == 0 else "FAILED"
    print(f"\nHumanEval: {status} in {elapsed:.0f}s")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
