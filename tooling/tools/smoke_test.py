#!/usr/bin/env python3
"""Smoke test wrapper — runs any experiment script for max 60s to catch crashes early.

Reports: import success, CUDA availability, memory baseline, first training step,
peak memory, and whether the script survives 60s without error.

Usage:
    python tools/smoke_test.py <script.py> [-- --args]

Exit codes:
    0 = smoke test passed (script ran 60s or completed without error)
    1 = script crashed within 60s (would waste hours of GPU time)
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TIMEOUT = 60  # seconds


def get_gpu_stats():
    """Get GPU memory stats via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            return {
                "gpu_mem_used_mb": int(parts[0]),
                "gpu_mem_total_mb": int(parts[1]),
                "gpu_util_pct": int(parts[2]),
            }
    except Exception:
        pass
    return {}


def get_system_stats():
    """Get CPU/RAM stats."""
    stats = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        stats["ram_used_mb"] = int(mem.used / 1024 / 1024)
        stats["ram_total_mb"] = int(mem.total / 1024 / 1024)
        stats["ram_pct"] = mem.percent
        stats["cpu_pct"] = psutil.cpu_percent(interval=0.5)
    except ImportError:
        # fallback: read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f}
            total = int(lines["MemTotal"].split()[0]) // 1024
            avail = int(lines["MemAvailable"].split()[0]) // 1024
            stats["ram_used_mb"] = total - avail
            stats["ram_total_mb"] = total
            stats["ram_pct"] = round((total - avail) / total * 100, 1)
        except Exception:
            pass
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/smoke_test.py <script.py> [-- --args]")
        sys.exit(1)

    script = sys.argv[1]
    # Parse args after --
    script_args = []
    if "--" in sys.argv[2:]:
        idx = sys.argv.index("--", 2)
        script_args = sys.argv[idx + 1:]
    elif len(sys.argv) > 2:
        script_args = sys.argv[2:]

    if not Path(script).exists():
        print(f"SMOKE FAIL: script not found: {script}")
        sys.exit(1)

    result = {
        "script": script,
        "args": script_args,
        "timeout_s": TIMEOUT,
        "status": "unknown",
    }

    # Pre-run stats
    print(f"=== SMOKE TEST: {script} (max {TIMEOUT}s) ===")
    pre_gpu = get_gpu_stats()
    pre_sys = get_system_stats()
    result["pre_gpu"] = pre_gpu
    result["pre_system"] = pre_sys
    print(f"PRE  GPU: {pre_gpu.get('gpu_mem_used_mb', '?')}/{pre_gpu.get('gpu_mem_total_mb', '?')} MB")
    print(f"PRE  RAM: {pre_sys.get('ram_used_mb', '?')}/{pre_sys.get('ram_total_mb', '?')} MB ({pre_sys.get('ram_pct', '?')}%)")

    # Run script with timeout
    cmd = [sys.executable, script] + script_args
    env = os.environ.copy()
    env["SMOKE_TEST"] = "1"  # scripts can check this to reduce iterations

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(Path(script).parent.parent.parent),
        )

        output_lines = []
        peak_gpu_mem = pre_gpu.get("gpu_mem_used_mb", 0)

        # Read output line by line, check GPU every 10s
        last_gpu_check = t0
        while True:
            elapsed = time.time() - t0
            if elapsed > TIMEOUT:
                # Script survived — that's a pass
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                result["status"] = "passed"
                result["reason"] = f"survived {TIMEOUT}s without crash"
                break

            # Non-blocking read
            import select
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    output_lines.append(decoded)
                    # Keep last 30 lines
                    if len(output_lines) > 50:
                        output_lines = output_lines[-30:]

            # Check GPU periodically
            if time.time() - last_gpu_check > 10:
                gpu = get_gpu_stats()
                if gpu.get("gpu_mem_used_mb", 0) > peak_gpu_mem:
                    peak_gpu_mem = gpu["gpu_mem_used_mb"]
                last_gpu_check = time.time()

            # Check if process died
            ret = proc.poll()
            if ret is not None:
                # Process exited
                remaining = proc.stdout.read()
                if remaining:
                    for l in remaining.decode("utf-8", errors="replace").splitlines():
                        output_lines.append(l)
                    if len(output_lines) > 50:
                        output_lines = output_lines[-30:]

                if ret == 0:
                    result["status"] = "passed"
                    result["reason"] = f"completed successfully in {elapsed:.1f}s"
                else:
                    result["status"] = "FAILED"
                    result["reason"] = f"crashed with rc={ret} after {elapsed:.1f}s"
                    result["returncode"] = ret
                break

    except Exception as e:
        result["status"] = "FAILED"
        result["reason"] = str(e)
        output_lines = [str(e)]

    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)

    # Post-run stats
    post_gpu = get_gpu_stats()
    post_sys = get_system_stats()
    result["post_gpu"] = post_gpu
    result["post_system"] = post_sys
    result["peak_gpu_mem_mb"] = max(
        peak_gpu_mem,
        post_gpu.get("gpu_mem_used_mb", 0),
    )
    result["last_output"] = output_lines[-20:] if output_lines else []

    # Summary
    print(f"\n=== SMOKE TEST RESULT: {result['status']} ===")
    print(f"Elapsed: {result['elapsed_s']}s")
    print(f"POST GPU: {post_gpu.get('gpu_mem_used_mb', '?')}/{post_gpu.get('gpu_mem_total_mb', '?')} MB")
    print(f"POST RAM: {post_sys.get('ram_used_mb', '?')}/{post_sys.get('ram_total_mb', '?')} MB ({post_sys.get('ram_pct', '?')}%)")
    print(f"Peak GPU: {result['peak_gpu_mem_mb']} MB")
    print(f"Reason: {result['reason']}")

    if result["status"] == "FAILED":
        print(f"\n--- Last output ---")
        for line in result["last_output"][-15:]:
            print(f"  {line}")

    # Write result JSON
    out_path = Path("/workspace/gpu_queue") / "smoke_results" / f"{Path(script).stem}.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved: {out_path}")
    except Exception:
        # Print to stdout as fallback
        print(json.dumps(result, indent=2))

    sys.exit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
