#!/usr/bin/env python3
"""Run all held-out evaluations: MMLU + HumanEval.

Wraps eval_mmlu.py and eval_humaneval.py into one script for gpu_queue.
Supports SMOKE_TEST=1 env var to validate in <60s.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = Path("/workspace/llm/results/held_out_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd, label, fatal=True):
    print(f"\n=== {label} ===")
    print(f"CMD: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd="/workspace/llm")
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"{label}: {'OK' if ok else 'FAILED'} in {elapsed:.0f}s")
    if not ok and fatal:
        sys.exit(r.returncode)
    return ok


def main():
    t0 = time.time()
    print(f"=== Pilot 50 Held-Out Evaluation ===")
    print(f"Smoke test: {IS_SMOKE}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # Phase 1: MMLU
    mmlu_args = [sys.executable, str(SCRIPT_DIR / "eval_mmlu.py")]
    if IS_SMOKE:
        # Just test one adapter to validate the pipeline
        mmlu_args += ["--adapter", "physics", "--out", str(RESULTS_DIR / "mmlu_smoke.json")]
    else:
        mmlu_args += ["--all", "--out", str(RESULTS_DIR / "mmlu_results.json")]
    run(mmlu_args, "MMLU Evaluation")

    # Phase 2: HumanEval (non-fatal: MMLU results are preserved even if this fails)
    he_args = [sys.executable, str(SCRIPT_DIR / "eval_humaneval.py"),
               "--adapter", "python", "--out",
               str(RESULTS_DIR / ("humaneval_smoke.json" if IS_SMOKE else "humaneval_results.json"))]
    if IS_SMOKE:
        he_args += ["--max-problems", "3"]
    he_ok = run(he_args, "HumanEval Evaluation", fatal=False)

    elapsed = time.time() - t0
    print(f"\n=== ALL {'DONE' if he_ok else 'PARTIAL (HumanEval failed)'} in {elapsed:.0f}s ===")

    # Print summary
    for name in ["mmlu_results.json", "humaneval_results.json",
                  "mmlu_smoke.json", "humaneval_smoke.json"]:
        p = RESULTS_DIR / name
        if p.exists():
            try:
                d = json.load(open(p))
                agg = d.get("aggregate", {})
                print(f"\n{name}: {json.dumps(agg, indent=2)}")
            except Exception as e:
                print(f"{name}: error reading: {e}")


    if not he_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
