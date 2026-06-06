#!/usr/bin/env python3
"""
BENCH: LiveCodeBench v6 Baseline + Pierre Code Adapter
Google target: 52.0% (float Gemma 4 E4B)

Phase 1: Base 4-bit model — establishes quantization gap vs 52.0%
Phase 2: CodeAlpaca code adapter — tests domain transfer to competitive programming

Kill criteria:
  K1420: base pass@1 >= 42% (within 10pp of 52.0%)
  K1421: code adapter pass@1 >= base + 5pp (domain transfer)
  K1422: total eval < 8h

MATH.md Theorem 1 prediction: base ~39-47%
MATH.md Theorem 2 prediction: adapter delta ~1-3pp (EXPECTED FAIL K1421)
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
# registry.json: code-codealpaca-knowledge-v0 lives in exp_p1_t2_single_domain_training
CODE_ADAPTER_PATH = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training" / "adapters" / "code"
LCB_DIR = EXPERIMENT_DIR.parent / "reference_implementations" / "LiveCodeBench"
PORT = 8321
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"


def log(msg):
    print(msg, flush=True)


def start_mlx_server(model_path, port=PORT, adapter_path=None):
    cmd = [sys.executable, "-m", "mlx_lm.server",
           "--model", model_path, "--port", str(port)]
    if adapter_path:
        cmd += ["--adapter-path", str(adapter_path)]
    log(f"  Starting MLX server: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=os.setsid)
    import urllib.request
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
            log(f"  MLX server ready on port {port}")
            return proc
        except Exception:
            pass
    raise RuntimeError("MLX server failed to start after 120s")


def stop_server(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
            log("  MLX server stopped")
        except Exception as e:
            log(f"  Warning: stop_server: {e}")
            proc.kill()


def install_lcb():
    log("[Setup] Installing LiveCodeBench...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(LCB_DIR)],
        capture_output=True, timeout=300
    )
    if result.returncode != 0:
        log(f"  pip install warning: {result.stderr[-200:]}")


def parse_score_from_stdout(stdout: str) -> dict:
    """Parse pass@k scores from LCB stdout output."""
    scores = {}
    # Overall Pass@k: e.g., "Pass@1 =  0.423"
    for m in re.finditer(r"Pass@(\d+)\s*=\s*([\d.]+)", stdout):
        scores[f"pass_at_{m.group(1)}"] = float(m.group(2))
    # Per-difficulty: e.g., "Easy Pass@1: 0.65"
    for m in re.finditer(r"(Easy|Medium|Hard)\s+Pass@(\d+):\s*([\d.]+)", stdout):
        scores[f"{m.group(1).lower()}_pass_at_{m.group(2)}"] = float(m.group(3))
    return scores


def run_livecodebench(model_name: str, port: int = PORT) -> dict:
    """Run LiveCodeBench v6 against local mlx_lm.server."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "not-needed"
    env["OPENAI_KEY"] = "not-needed"  # LCB reads OPENAI_KEY for api_key param
    env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"

    # Always use --n 1 (pass@1 from single sample — sufficient for directional signal).
    # --n 10 means 10 samples PER PROBLEM (not 10 problems). With 500+ problems in v6
    # that would be 5000+ generations (~100h). Date filter caps to ~50-100 recent problems.
    if IS_SMOKE:
        start_date, end_date = "2025-03-01", "2025-04-30"
    else:
        start_date, end_date = "2025-01-01", "2025-04-30"

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", model_name,
        "--scenario", "codegeneration",
        "--release_version", "release_v6",
        "--evaluate",
        "--n", "1",
        "--temperature", "0.2",
        "--start_date", start_date,
        "--end_date", end_date,
    ]

    log(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=10 * 3600,
        env=env,
        cwd=str(LCB_DIR)
    )
    elapsed = time.time() - t0

    log(f"  Completed in {elapsed:.0f}s (exit={result.returncode})")
    log(f"  STDOUT tail:\n{result.stdout[-2000:]}")
    if result.returncode != 0:
        log(f"  STDERR:\n{result.stderr[-500:]}")

    scores = parse_score_from_stdout(result.stdout)
    log(f"  Parsed scores: {scores}")

    return {
        "scores": scores,
        "stdout": result.stdout[-3000:],
        "stderr": result.stderr[-500:] if result.returncode != 0 else "",
        "elapsed_s": elapsed,
        "exit_code": result.returncode,
    }


def main():
    log("=" * 70)
    log("BENCH: LiveCodeBench v6 — Gemma 4 E4B 4-bit (base + code adapter)")
    log(f"SMOKE={IS_SMOKE}")
    log("=" * 70)

    results = {
        "experiment": "exp_bench_livecodebench_v6",
        "smoke": IS_SMOKE,
        "model": MODEL_ID,
        "google_target": 0.520,
        "kill_criteria": {
            "K1420": "base pass@1 >= 0.42 (within 10pp of 52.0%)",
            "K1421": "code adapter pass@1 >= base + 0.05",
            "K1422": "total eval < 8h",
        }
    }
    t_start = time.time()

    install_lcb()

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B 4-bit (no adapter)")
    server = start_mlx_server(MODEL_ID)
    try:
        base_results = run_livecodebench("gemma-4-e4b-base")
        results["base"] = base_results
    finally:
        stop_server(server)
        time.sleep(3)  # Let port free up

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n[Checkpoint] Base results saved to {RESULTS_FILE}")

    # Phase 2: Code adapter
    log("\n[Phase 2] Code adapter (CodeAlpaca, q_proj r=6)")
    if CODE_ADAPTER_PATH.exists():
        server = start_mlx_server(MODEL_ID, adapter_path=CODE_ADAPTER_PATH)
        try:
            adapter_results = run_livecodebench("gemma-4-e4b-code")
            results["code_adapter"] = adapter_results
        finally:
            stop_server(server)
    else:
        log(f"  Code adapter not found at {CODE_ADAPTER_PATH} — skipping Phase 2")
        results["code_adapter"] = {"error": f"adapter not found: {CODE_ADAPTER_PATH}"}

    total_elapsed = time.time() - t_start
    results["total_elapsed_s"] = total_elapsed

    # Kill criteria evaluation
    base_score = results.get("base", {}).get("scores", {}).get("pass_at_1", None)
    adapter_score = results.get("code_adapter", {}).get("scores", {}).get("pass_at_1", None)

    k_results = {}
    if base_score is not None:
        k_results["K1420"] = "PASS" if base_score >= 0.42 else "FAIL"
        log(f"\n[K1420] Base LCB pass@1 = {base_score:.3f} → {k_results['K1420']}")
    if base_score is not None and adapter_score is not None:
        delta = adapter_score - base_score
        k_results["K1421"] = "PASS" if delta >= 0.05 else "FAIL"
        log(f"[K1421] Code adapter delta = {delta:.3f} → {k_results['K1421']}")
    k_results["K1422"] = "PASS" if total_elapsed < 8 * 3600 else "FAIL"
    log(f"[K1422] Total time = {total_elapsed/3600:.1f}h → {k_results['K1422']}")

    results["kill_results"] = k_results
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nFinal results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
