#!/usr/bin/env python3
"""
HumanEval pass@1 evaluation using bigcode-evaluation-harness.

Uses the standard bigcode-evaluation-harness framework for proper:
- Batched generation with correct stop sequences
- Sandboxed code execution with timeouts
- pass@k calculation per Chen et al. (2021)
- PEFT adapter and 4-bit quantization support

Reference: https://github.com/bigcode-project/bigcode-evaluation-harness

Usage:
    python eval_humaneval_v2.py                        # base + all available adapters
    python eval_humaneval_v2.py --adapter python       # base + single adapter
    SMOKE_TEST=1 python eval_humaneval_v2.py           # quick 5-problem test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BIGCODE_REPO = "https://github.com/bigcode-project/bigcode-evaluation-harness.git"
BIGCODE_DIR = Path("/workspace/bigcode-evaluation-harness")

BASE_MODEL = "/workspace/models/Qwen2.5-7B"
ADAPTER_DIR = Path("/workspace/llm/adapters")
RESULTS_DIR = Path("/workspace/llm/results/held_out_eval")

# Code adapters to evaluate on HumanEval (Python code generation benchmark)
CODE_ADAPTERS = [
    "python", "javascript", "rust", "go", "cpp",
    "java", "typescript", "bash", "swift", "sql",
]

MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", 7200))  # 2hr default
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"


def install_bigcode_harness():
    """Install bigcode-evaluation-harness if not already present."""
    try:
        import bigcode_eval  # noqa: F401
        print("bigcode-evaluation-harness already installed")
        return True
    except ImportError:
        pass

    print("Installing bigcode-evaluation-harness...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/bigcode-project/bigcode-evaluation-harness.git"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"pip install failed: {r.stderr[-500:]}")
        # Fallback: clone and install
        if not BIGCODE_DIR.exists():
            subprocess.run(
                ["git", "clone", "--depth=1", BIGCODE_REPO, str(BIGCODE_DIR)],
                timeout=120,
            )
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(BIGCODE_DIR)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"Fallback install failed: {r.stderr[-500:]}")
            return False

    print("bigcode-evaluation-harness installed successfully")
    return True


def find_bigcode_main():
    """Locate the main.py entry point for bigcode-evaluation-harness."""
    # Primary location: cloned repo
    main_py = BIGCODE_DIR / "main.py"
    if main_py.exists():
        return str(main_py)

    # Fallback: clone if missing
    print(f"main.py not found at {main_py}, cloning repo...")
    subprocess.run(
        ["git", "clone", "--depth=1", BIGCODE_REPO, str(BIGCODE_DIR)],
        timeout=120,
    )
    if main_py.exists():
        return str(main_py)

    return None


def run_humaneval(model_path, peft_model=None, output_prefix="base", limit=None):
    """Run HumanEval evaluation via bigcode-evaluation-harness CLI.

    Args:
        model_path: Path to base model
        peft_model: Optional path to PEFT/LoRA adapter
        output_prefix: Prefix for output files
        limit: Max problems to evaluate (None = all 164)

    Returns:
        dict with results or None on failure
    """
    main_py = find_bigcode_main()
    if not main_py:
        print("ERROR: Cannot find bigcode-evaluation-harness main.py")
        return None

    metric_path = str(RESULTS_DIR / f"humaneval_{output_prefix}_metrics.json")
    gen_path = str(RESULTS_DIR / f"humaneval_{output_prefix}_generations.json")

    cmd = [
        sys.executable, main_py,
        "--model", model_path,
        "--tasks", "humaneval",
        "--trust_remote_code",
        "--do_sample", "False",
        "--n_samples", "1",
        "--batch_size", "1",
        "--max_length_generation", "512",
        "--temperature", "0.0",
        "--precision", "bf16",
        "--allow_code_execution",
        "--save_generations",
        "--save_generations_path", gen_path,
        "--metric_output_path", metric_path,
    ]

    if peft_model:
        cmd.extend(["--peft_model", peft_model])

    if limit:
        cmd.extend(["--limit", str(limit)])

    label = output_prefix if not peft_model else f"adapter:{output_prefix}"
    print(f"\n{'='*60}")
    print(f"Running HumanEval: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            cwd="/workspace/llm",
            timeout=MAX_RUNTIME // 2,  # each run gets half the budget
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.0f}s (rc={r.returncode})")

        if r.returncode != 0:
            return {"error": f"exit code {r.returncode}", "elapsed_s": elapsed}

        # Parse results
        if os.path.exists(metric_path):
            with open(metric_path) as f:
                metrics = json.load(f)
            metrics["elapsed_s"] = round(elapsed, 1)
            return metrics
        else:
            return {"error": "no metrics file produced", "elapsed_s": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"TIMEOUT after {elapsed:.0f}s")
        return {"error": "timeout", "elapsed_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e}")
        return {"error": str(e), "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="HumanEval eval via bigcode-evaluation-harness")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--adapter", type=str, default=None,
                        help="Single adapter to evaluate (default: all available code adapters)")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model evaluation")
    parser.add_argument("--out", default=str(RESULTS_DIR / "humaneval_v2_results.json"))
    args = parser.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== HumanEval Evaluation (bigcode-evaluation-harness) ===")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"SMOKE_TEST: {SMOKE_TEST}")

    # Step 1: Install framework
    if not install_bigcode_harness():
        print("FATAL: Could not install bigcode-evaluation-harness")
        sys.exit(1)

    limit = 5 if SMOKE_TEST else None

    # Step 2: Determine which adapters to evaluate
    adapter_dir = Path(args.adapter_dir)
    if args.adapter:
        adapters = [args.adapter]
    else:
        adapters = [
            name for name in CODE_ADAPTERS
            if (adapter_dir / name).exists()
        ]
    print(f"Adapters to evaluate: {adapters}")
    print(f"Available on disk: {[d.name for d in adapter_dir.iterdir() if d.is_dir()]}")

    results = {
        "experiment": "pilot50_held_out_humaneval_v2",
        "framework": "bigcode-evaluation-harness",
        "base_model": args.base_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "smoke_test": SMOKE_TEST,
        "base_results": None,
        "adapter_results": {},
        "comparisons": {},
    }

    # Step 3: Evaluate base model
    if not args.skip_base:
        base_res = run_humaneval(args.base_model, output_prefix="base", limit=limit)
        results["base_results"] = base_res
        if base_res and "error" not in base_res:
            # bigcode-eval returns results under the task key
            he_res = base_res.get("humaneval", base_res)
            print(f"Base: {json.dumps(he_res, indent=2)}")

    # Step 4: Evaluate each adapter
    for adapter_name in adapters:
        adapter_path = str(adapter_dir / adapter_name)
        if not os.path.exists(adapter_path):
            print(f"SKIP {adapter_name}: not found at {adapter_path}")
            continue

        res = run_humaneval(
            args.base_model,
            peft_model=adapter_path,
            output_prefix=adapter_name,
            limit=limit,
        )
        results["adapter_results"][adapter_name] = res

        # Compare with base
        if results["base_results"] and res and "error" not in res:
            base_he = results["base_results"].get("humaneval", results["base_results"])
            adapter_he = res.get("humaneval", res)

            # Extract pass@1 from bigcode-eval format
            base_p1 = base_he.get("pass@1", base_he.get("pass_at_1"))
            adapter_p1 = adapter_he.get("pass@1", adapter_he.get("pass_at_1"))

            if base_p1 is not None and adapter_p1 is not None:
                results["comparisons"][adapter_name] = {
                    "base_pass_at_1": base_p1,
                    "adapter_pass_at_1": adapter_p1,
                    "delta_pp": adapter_p1 - base_p1,
                    "adapter_wins": adapter_p1 > base_p1,
                }

    # Step 5: Aggregate
    if results["comparisons"]:
        wins = sum(1 for c in results["comparisons"].values() if c["adapter_wins"])
        total = len(results["comparisons"])
        avg_delta = sum(c["delta_pp"] for c in results["comparisons"].values()) / total

        results["aggregate"] = {
            "adapters_evaluated": total,
            "adapters_beating_base": wins,
            "win_rate_pct": wins / total * 100,
            "avg_delta_pp": avg_delta,
        }

        results["kill_criteria"] = {
            "python_below_base": results["comparisons"].get("python", {}).get("adapter_wins") is False,
        }

        print(f"\n=== AGGREGATE ===")
        print(f"Adapters beating base: {wins}/{total}")
        print(f"Average delta: {avg_delta:+.4f}")

    elapsed = time.time() - t0
    results["total_elapsed_s"] = round(elapsed, 1)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
