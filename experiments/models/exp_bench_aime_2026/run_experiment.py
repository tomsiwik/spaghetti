#!/usr/bin/env python3
"""
BENCH: AIME 2026 Baseline + Pierre Adapted
Google target: 42.5% pass@k

Runs AIME 2026 (30 math competition problems) via MathArena harness.
Uses mlx_lm.server as OpenAI-compatible backend (port 8321).

n=2 seeds per problem (not 4) to fit within 2h budget (Theorem 3 in MATH.md).
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
MATHARENA_DIR = EXPERIMENT_DIR.parent / "reference_implementations" / "matharena"
PORT = 8321
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
# n=2 to fit in 2h (Theorem 3: n=4 would need ~3h)
N_SAMPLES = 1 if IS_SMOKE else 2
MODEL_NAME = "local/gemma4-e4b-4bit"
MODEL_OUTPUT_DIR = EXPERIMENT_DIR / "matharena_output"


def log(msg):
    print(msg, flush=True)


def write_model_config():
    """Write Gemma 4 E4B local server config into MathArena's config dir."""
    local_dir = MATHARENA_DIR / "configs" / "models" / "local"
    local_dir.mkdir(exist_ok=True)
    config_path = local_dir / "gemma4-e4b-4bit.yaml"
    config_path.write_text(f"""model: gemma-4-e4b-4bit
api: custom
api_key_env: null
base_url: http://localhost:{PORT}/v1
max_tokens: 8192
read_cost: 0.0
write_cost: 0.0
concurrent_requests: 1
human_readable_id: gemma4-e4b-4bit-local
date: "2026-04-14"
temperature: 0.7
top_p: 0.95
other_params:
  open: true
  creator: Google/MLX
  parameters: 4
  active_parameters: 4
""")
    log(f"  Model config written: {config_path}")
    return config_path


def start_mlx_server(model_path, adapter_path=None, port=PORT):
    cmd = [sys.executable, "-m", "mlx_lm.server",
           "--model", model_path, "--port", str(port),
           "--log-level", "WARNING"]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]
    log(f"  Starting MLX server: {' '.join(cmd)}")
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
            continue
    raise RuntimeError("MLX server failed to start within 120s")


def stop_server(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
            log("  MLX server stopped")
        except Exception as e:
            log(f"  Warning: server stop failed: {e}")


def parse_score_from_status(status_file: Path) -> dict:
    """Parse MathArena status file: lines like 'Problem  0: OOX- (1 left)'."""
    if not status_file.exists():
        return {"pass_at_n": None, "n_correct": None, "n_problems": None, "raw": ""}
    text = status_file.read_text()
    n_problems = 0
    n_solved = 0  # at least 1 correct in n runs
    total_runs = 0
    total_correct = 0
    for line in text.splitlines():
        m = re.match(r"Problem\s+\d+:\s+([OX-]+)", line)
        if m:
            n_problems += 1
            runs = m.group(1).replace("-", "")  # ignore pending
            correct_runs = runs.count("O")
            total_correct += correct_runs
            total_runs += len(runs)
            if correct_runs > 0:
                n_solved += 1
    pass_at_n = n_solved / n_problems if n_problems > 0 else None
    avg_correct = total_correct / total_runs if total_runs > 0 else None
    return {
        "pass_at_n": pass_at_n,
        "n_correct": n_solved,
        "n_problems": n_problems,
        "avg_per_run": avg_correct,
        "total_correct_runs": total_correct,
        "total_runs": total_runs,
        "raw": text[:500],
    }


def run_matharena(comp="aime/aime_2026", n_samples=N_SAMPLES, output_subdir="base") -> dict:
    """Run MathArena against the local MLX server."""
    output_dir = MODEL_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Problems to run: smoke test uses only first 3 problems
    cmd = [
        sys.executable,
        str(MATHARENA_DIR / "scripts" / "run.py"),
        "--comp", comp,
        "--models", MODEL_NAME,
        "--n", str(n_samples),
        "--model-configs-dir", str(MATHARENA_DIR / "configs" / "models"),
        "--comp-configs-dir", str(MATHARENA_DIR / "configs" / "competitions"),
        "--output-dir", str(output_dir),
    ]
    if IS_SMOKE:
        cmd += ["--problems", "1", "2", "3"]

    log(f"  Running MathArena ({comp}, n={n_samples}): {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(3 * 3600),
        cwd=str(MATHARENA_DIR),
    )
    elapsed = time.time() - t0

    log(f"  MathArena done in {elapsed:.0f}s (exit={result.returncode})")
    if result.stdout:
        log(f"  STDOUT (tail): {result.stdout[-2000:]}")
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[-500:]}")

    # Parse score from status file (glob to find it regardless of nested path)
    status_files = list((MATHARENA_DIR / "logs" / "status").glob("**/*2026*"))
    status_file = status_files[0] if status_files else Path("/dev/null")
    score = parse_score_from_status(status_file)
    pct = f"{score['pass_at_n']:.1%}" if score['pass_at_n'] is not None else "N/A"
    log(f"  Score: pass@{n_samples}={pct} ({score['n_correct']}/{score['n_problems']})")

    return {
        "elapsed_s": elapsed,
        "exit_code": result.returncode,
        "score": score,
        "stdout_tail": result.stdout[-2000:],
        "stderr": result.stderr[-300:] if result.returncode != 0 else "",
    }


def find_math_adapter() -> Path | None:
    """Find math adapter path from registry."""
    repo_root = EXPERIMENT_DIR.parent.parent.parent
    registry_path = repo_root / "data" / "adapters" / "registry.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text())
    for entry in registry:
        if entry.get("domain") == "math" or "math" in entry.get("id", ""):
            adapter_path = Path(entry.get("path", ""))
            if adapter_path.exists():
                return adapter_path
    return None


def main():
    log("=" * 70)
    log("BENCH: AIME 2026 — Base E4B-4bit vs Pierre Math Adapter")
    log(f"SMOKE={IS_SMOKE}, N_SAMPLES={N_SAMPLES}, MODEL={MODEL_ID}")
    log("=" * 70)

    t_start = time.time()
    results = {
        "experiment": "exp_bench_aime_2026",
        "smoke": IS_SMOKE,
        "model": MODEL_ID,
        "n_samples": N_SAMPLES,
        "google_target": 0.425,
    }

    # Write model config for MathArena
    write_model_config()

    # --- Phase 1: Base model ---
    log("\n[Phase 1] Base Gemma 4 E4B-4bit on AIME 2026")
    server = start_mlx_server(MODEL_ID)
    try:
        base_result = run_matharena(output_subdir="base")
        results["base"] = base_result
        results["base_pass_at_n"] = base_result["score"]["pass_at_n"]
    finally:
        stop_server(server)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nPhase 1 saved → {RESULTS_FILE}")

    # --- Phase 2: Math adapter ---
    log("\n[Phase 2] Pierre Math Adapter on AIME 2026")
    adapter_path = find_math_adapter()
    if adapter_path:
        log(f"  Found math adapter: {adapter_path}")
        server = start_mlx_server(MODEL_ID, adapter_path=str(adapter_path))
        try:
            adp_result = run_matharena(output_subdir="math_adapter")
            results["math_adapter"] = adp_result
            results["adapter_pass_at_n"] = adp_result["score"]["pass_at_n"]
        finally:
            stop_server(server)
    else:
        log("  No math adapter found in registry — skipping Phase 2")
        results["math_adapter"] = {"skipped": True, "reason": "adapter not found"}

    # --- K-criteria evaluation ---
    base_acc = results.get("base_pass_at_n")
    adp_acc = results.get("adapter_pass_at_n")
    google_target = 0.425
    t_total = time.time() - t_start

    results["elapsed_total_s"] = t_total
    results["kill_criteria"] = {
        "K1417_base_within_10pp": (
            abs(base_acc - google_target) <= 0.10
            if base_acc is not None else None
        ),
        "K1418_adapter_plus_10pp": (
            adp_acc >= (base_acc or 0) + 0.10
            if adp_acc is not None and base_acc is not None else None
        ),
        "K1419_under_2h": t_total < 7200,
    }

    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log(f"  Base pass@{N_SAMPLES}: {base_acc:.1%}" if base_acc is not None else "  Base: N/A")
    log(f"  Adapter pass@{N_SAMPLES}: {adp_acc:.1%}" if adp_acc is not None else "  Adapter: N/A")
    log(f"  Google target: {google_target:.1%}")
    log(f"  Total time: {t_total/60:.1f} min")
    log(f"  K1417 (within 10pp): {results['kill_criteria']['K1417_base_within_10pp']}")
    log(f"  K1418 (adapter +10pp): {results['kill_criteria']['K1418_adapter_plus_10pp']}")
    log(f"  K1419 (<2h): {results['kill_criteria']['K1419_under_2h']}")
    log("=" * 70)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nFinal results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
