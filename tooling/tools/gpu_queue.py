#!/usr/bin/env python3
"""
GPU Task Queue — submit, monitor, and retrieve GPU jobs on RunPod.

The worker runs persistently on RunPod, processing tasks with zero idle time.
This tool manages the queue from the local machine via SSH.

Usage:
    python tools/gpu_queue.py start                    # start worker on RunPod
    python tools/gpu_queue.py submit script.py [args]  # add task to queue
    python tools/gpu_queue.py status                   # show queue + active task
    python tools/gpu_queue.py results [task_id]        # fetch results
    python tools/gpu_queue.py log [N]                  # tail worker log
    python tools/gpu_queue.py stop                     # graceful stop after current task
    python tools/gpu_queue.py drain                    # list completed tasks
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SSH_ALIAS = "runpod"
REMOTE_DIR = "/workspace/llm"
QUEUE_DIR = "/workspace/gpu_queue"
QUEUE_FILE = f"{QUEUE_DIR}/pending.jsonl"
ACTIVE_FILE = f"{QUEUE_DIR}/active.json"
DONE_DIR = f"{QUEUE_DIR}/done"
LOG_FILE = f"{QUEUE_DIR}/worker.log"
PID_FILE = f"{QUEUE_DIR}/worker.pid"
STOP_FILE = f"{QUEUE_DIR}/stop"

REPO_ROOT = Path(__file__).parent.parent


def ssh(cmd, timeout=30, capture=True):
    """Run command on RunPod via SSH."""
    full = ["ssh", SSH_ALIAS, cmd]
    if capture:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    else:
        return subprocess.run(full, timeout=timeout).returncode


def ssh_check():
    """Verify SSH connectivity."""
    try:
        rc, out, _ = ssh("echo ok", timeout=10)
        return rc == 0 and "ok" in out
    except Exception:
        return False


# ── Worker script (runs on RunPod) ────────────────────────────────────────────

WORKER_SCRIPT = r'''#!/usr/bin/env python3
"""GPU Worker — persistent task processor. Runs on RunPod."""
import json, os, sys, time, signal, subprocess, shutil, gc
from pathlib import Path
from datetime import datetime, timezone

QUEUE_DIR = Path("/workspace/gpu_queue")
QUEUE_FILE = QUEUE_DIR / "pending.jsonl"
ACTIVE_FILE = QUEUE_DIR / "active.json"
DONE_DIR = QUEUE_DIR / "done"
LOG_FILE = QUEUE_DIR / "worker.log"
PID_FILE = QUEUE_DIR / "worker.pid"
STOP_FILE = QUEUE_DIR / "stop"
WORK_DIR = Path("/workspace/llm")

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def pop_task():
    """Atomically pop first task from queue."""
    if not QUEUE_FILE.exists():
        return None
    with open(QUEUE_FILE, "r") as f:
        lines = [l for l in f.readlines() if l.strip()]
    if not lines:
        return None
    task = json.loads(lines[0])
    with open(QUEUE_FILE, "w") as f:
        f.writelines(lines[1:])
    return task

def mark_active(task):
    task["started_at"] = datetime.now(timezone.utc).isoformat()
    with open(ACTIVE_FILE, "w") as f:
        json.dump(task, f, indent=2)

def mark_done(task, returncode, elapsed):
    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    task["returncode"] = returncode
    task["elapsed_s"] = round(elapsed, 1)
    task["status"] = "ok" if returncode == 0 else "failed"
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    done_file = DONE_DIR / f"{task['id']}.json"
    with open(done_file, "w") as f:
        json.dump(task, f, indent=2)
    if ACTIVE_FILE.exists():
        ACTIVE_FILE.unlink()
    return task

def _kill_process_group(proc):
    """Kill a process and all its children via process group."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    except (ProcessLookupError, PermissionError, OSError):
        # Process already exited
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

def run_smoke_test(task, run_env):
    """Run script for max 60s to catch crashes before committing GPU hours.
    Returns True if smoke test passes, False if it fails."""
    script = task["script"]
    args = task.get("args", [])
    smoke_env = run_env.copy()
    smoke_env["SMOKE_TEST"] = "1"

    smoke_log = QUEUE_DIR / f"{task['id']}.smoke.log"
    log(f"SMOKE: {task['id']} — running {script} for max 60s...")

    cmd = [sys.executable, script] + args
    t0 = time.time()
    try:
        with open(smoke_log, "w") as logf:
            # start_new_session=True creates a process group so we can
            # kill the entire tree (prevents orphaned child processes
            # that leak memory — root cause of OOM crash 2026-03-16)
            proc = subprocess.Popen(
                cmd, cwd=str(WORK_DIR), env=smoke_env,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                elapsed = time.time() - t0
                log(f"SMOKE PASS: {task['id']} — survived {elapsed:.0f}s without crash")
                return True

        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"SMOKE PASS: {task['id']} — completed in {elapsed:.0f}s")
            return True
        else:
            # Read last 15 lines of smoke log for error context
            try:
                lines = open(smoke_log).readlines()
                tail = "".join(lines[-15:])
                log(f"SMOKE FAIL: {task['id']} — crashed rc={proc.returncode} in {elapsed:.0f}s\n{tail}")
            except Exception:
                log(f"SMOKE FAIL: {task['id']} — crashed rc={proc.returncode} in {elapsed:.0f}s")
            return False
    except Exception as e:
        log(f"SMOKE FAIL: {task['id']} — exception: {e}")
        return False

def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return -1

def _get_rss_mb():
    """Get current worker RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1

def run_task(task):
    """Execute a task. Returns (returncode, elapsed_seconds)."""
    script = task["script"]
    args = task.get("args", [])
    env = task.get("env", {})
    max_runtime = int(env.get("MAX_RUNTIME", 0)) or 28800  # 8hr default

    cmd = [sys.executable, script] + args
    run_env = os.environ.copy()
    run_env["HF_HOME"] = "/workspace/hf_cache"
    run_env["OMP_NUM_THREADS"] = "4"
    run_env["MKL_NUM_THREADS"] = "4"
    run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_env.update(env)

    # Smoke test first — catch crashes in 60s instead of wasting hours
    skip_smoke = task.get("skip_smoke", False)
    if not skip_smoke:
        if not run_smoke_test(task, run_env):
            log(f"SKIPPED: {task['id']} — smoke test failed, not wasting GPU hours")
            return 1, 0.0

    task_log = QUEUE_DIR / f"{task['id']}.log"
    log(f"START: {task['id']} — {script} {' '.join(args)}")
    mark_active(task)

    t0 = time.time()
    util_log_interval = 60  # Log GPU util every 60s
    last_util_log = 0
    with open(task_log, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(WORK_DIR), env=run_env,
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,  # process group for clean kill
        )
        # Poll so we can log GPU utilization
        while proc.poll() is None:
            elapsed = time.time() - t0
            # Hard timeout — kill entire process group
            if elapsed > max_runtime:
                log(f"TIMEOUT: {task['id']} — killing after {elapsed:.0f}s (limit {max_runtime}s)")
                _kill_process_group(proc)
                break
            # Log GPU utilization periodically after 120s warmup
            if elapsed > 120 and elapsed - last_util_log > util_log_interval:
                util = get_gpu_utilization()
                rss = _get_rss_mb()
                if util >= 0:
                    log(f"UTIL: {task['id']} — GPU {util}%, worker RSS {rss}MB at {elapsed:.0f}s")
                last_util_log = elapsed
            time.sleep(5)

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    log(f"DONE: {task['id']} — {status} in {elapsed:.0f}s")

    # Force memory cleanup between tasks
    gc.collect()

    return proc.returncode, elapsed

def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Clean stop file
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    log("=== GPU Worker started ===")
    log(f"PID: {os.getpid()}")
    log(f"Queue: {QUEUE_FILE}")

    idle_logged = False
    while True:
        # Check for stop signal
        if STOP_FILE.exists():
            log("Stop signal received. Exiting after current task.")
            STOP_FILE.unlink()
            break

        task = pop_task()
        if task is None:
            if not idle_logged:
                log("Queue empty — waiting for tasks...")
                idle_logged = True
            time.sleep(3)
            continue

        idle_logged = False
        rc, elapsed = run_task(task)
        result = mark_done(task, rc, elapsed)

        # Brief pause to let filesystem sync
        time.sleep(1)

    log("=== GPU Worker stopped ===")
    if PID_FILE.exists():
        PID_FILE.unlink()

if __name__ == "__main__":
    main()
'''


# ── Local commands ────────────────────────────────────────────────────────────

def cmd_start(args):
    """Start the persistent GPU worker on RunPod."""
    # Check if already running
    rc, out, _ = ssh(f"cat {PID_FILE} 2>/dev/null && ps -p $(cat {PID_FILE} 2>/dev/null) -o pid= 2>/dev/null", timeout=10)
    if rc == 0 and out.strip():
        print(f"Worker already running (PID {out.strip().splitlines()[-1].strip()})")
        return 0

    # Upload worker script
    worker_path = f"{QUEUE_DIR}/worker.py"
    ssh(f"mkdir -p {QUEUE_DIR} {DONE_DIR}", timeout=10)

    # Write worker script via heredoc
    proc = subprocess.run(
        ["ssh", SSH_ALIAS, f"cat > {worker_path}"],
        input=WORKER_SCRIPT, text=True, timeout=15,
    )
    if proc.returncode != 0:
        print("Failed to upload worker script")
        return 1

    # Start worker as detached background process
    # Use setsid + redirect + disown to fully detach from SSH session
    rc, _, err = ssh(
        f"cd {REMOTE_DIR} && setsid python3 {worker_path} >> {LOG_FILE} 2>&1 < /dev/null &",
        timeout=15,
    )

    time.sleep(2)
    rc, out, _ = ssh(f"cat {PID_FILE} 2>/dev/null", timeout=10)
    if rc == 0 and out.strip():
        print(f"Worker started (PID {out.strip()})")
    else:
        print("Worker may have failed to start. Check: python tools/gpu_queue.py log")
    return 0


def cmd_submit(args):
    """Submit a task to the GPU queue."""
    # Sync repo first
    print("Syncing repo...")
    sync_cmd = [
        "rsync", "-rlptz", "--delete",
        "--exclude", ".venv", "--exclude", "__pycache__",
        "--exclude", ".git", "--exclude", "*.pyc",
        "--exclude", ".ralph/events-*", "--exclude", ".worktrees",
        "--exclude", "references",
        "--exclude", "results",
        "--exclude", "adapters/*/adapter_model.safetensors",
        "--exclude", "**/checkpoints/",
        "--exclude", "**/results_*.json",
        "-e", "ssh",
        f"{REPO_ROOT}/", f"{SSH_ALIAS}:{REMOTE_DIR}/",
    ]
    subprocess.run(sync_cmd, timeout=120)

    # Build task
    task_id = f"{Path(args.script).stem}_{int(time.time())}"
    task = {
        "id": task_id,
        "script": args.script,
        "args": args.args or [],
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    task_json = json.dumps(task)

    # Append to remote queue
    rc, _, err = ssh(f"echo '{task_json}' >> {QUEUE_FILE}", timeout=10)
    if rc == 0:
        print(f"Submitted: {task_id}")
        print(f"  Script: {args.script}")
        if args.args:
            print(f"  Args: {' '.join(args.args)}")
    else:
        print(f"Failed to submit: {err}")
    return rc


def cmd_status(args):
    """Show queue status."""
    # Active task
    rc, out, _ = ssh(f"cat {ACTIVE_FILE} 2>/dev/null || echo 'none'", timeout=10)
    if out and out != "none":
        active = json.loads(out)
        print(f"ACTIVE: {active['id']} — {active['script']}")
        print(f"  Started: {active.get('started_at', '?')}")
    else:
        print("ACTIVE: (none)")

    # Pending queue
    rc, out, _ = ssh(f"cat {QUEUE_FILE} 2>/dev/null || echo ''", timeout=10)
    pending = [json.loads(l) for l in out.strip().splitlines() if l.strip()]
    print(f"\nPENDING: {len(pending)} tasks")
    for t in pending:
        print(f"  - {t['id']}: {t['script']} {' '.join(t.get('args', []))}")

    # Recent done
    rc, out, _ = ssh(f"ls -t {DONE_DIR}/*.json 2>/dev/null | head -5", timeout=10)
    if out:
        print(f"\nRECENT COMPLETED:")
        for f in out.splitlines():
            rc2, content, _ = ssh(f"cat {f}", timeout=10)
            if rc2 == 0:
                d = json.loads(content)
                status = "OK" if d.get("status") == "ok" else "FAILED"
                print(f"  {status}: {d['id']} ({d.get('elapsed_s', '?')}s)")

    # Worker alive?
    rc, out, _ = ssh(f"cat {PID_FILE} 2>/dev/null && ps -p $(cat {PID_FILE} 2>/dev/null) -o pid= 2>/dev/null", timeout=10)
    alive = rc == 0 and out.strip()
    print(f"\nWORKER: {'RUNNING' if alive else 'STOPPED'}")

    # GPU utilization
    rc, out, _ = ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null", timeout=10)
    if rc == 0:
        print(f"GPU: {out}")
    return 0


def cmd_results(args):
    """Fetch results for a specific task or latest."""
    if args.task_id:
        pattern = f"{DONE_DIR}/{args.task_id}*.json"
    else:
        pattern = f"{DONE_DIR}/*.json"

    rc, out, _ = ssh(f"ls -t {pattern} 2>/dev/null | head -1", timeout=10)
    if not out:
        print("No results found")
        return 1

    rc, content, _ = ssh(f"cat {out}", timeout=10)
    if rc == 0:
        d = json.loads(content)
        print(json.dumps(d, indent=2))

        # Also show task log tail
        task_log = f"{QUEUE_DIR}/{d['id']}.log"
        rc, log_out, _ = ssh(f"tail -20 {task_log} 2>/dev/null", timeout=10)
        if log_out:
            print(f"\n--- Last 20 lines of task log ---")
            print(log_out)
    return 0


def cmd_log(args):
    """Tail the worker log."""
    n = args.lines or 30
    rc = ssh(f"tail -{n} {LOG_FILE} 2>/dev/null || echo 'No log found'", timeout=10, capture=False)
    return rc


def cmd_stop(args):
    """Signal worker to stop after current task."""
    rc, _, _ = ssh(f"touch {STOP_FILE}", timeout=10)
    if rc == 0:
        print("Stop signal sent. Worker will exit after current task completes.")
    return rc


def cmd_drain(args):
    """List all completed tasks."""
    rc, out, _ = ssh(f"ls -t {DONE_DIR}/*.json 2>/dev/null", timeout=10)
    if not out:
        print("No completed tasks")
        return 0
    for f in out.splitlines():
        rc2, content, _ = ssh(f"cat {f}", timeout=10)
        if rc2 == 0:
            d = json.loads(content)
            status = "OK" if d.get("status") == "ok" else "FAIL"
            print(f"[{status}] {d['id']}  {d.get('elapsed_s','?')}s  {d['script']}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="GPU Task Queue for RunPod")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start persistent worker on RunPod")

    p_sub = sub.add_parser("submit", help="Submit task to queue")
    p_sub.add_argument("script", help="Script path (relative to repo root)")
    p_sub.add_argument("args", nargs=argparse.REMAINDER, help="Script arguments (use -- before flags)")

    sub.add_parser("status", help="Show queue status")

    p_res = sub.add_parser("results", help="Fetch task results")
    p_res.add_argument("task_id", nargs="?", help="Task ID (default: latest)")

    p_log = sub.add_parser("log", help="Tail worker log")
    p_log.add_argument("lines", nargs="?", type=int, help="Number of lines (default: 30)")

    sub.add_parser("stop", help="Graceful stop after current task")
    sub.add_parser("drain", help="List all completed tasks")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "start": cmd_start, "submit": cmd_submit, "status": cmd_status,
        "results": cmd_results, "log": cmd_log, "stop": cmd_stop,
        "drain": cmd_drain,
    }
    sys.exit(handlers[args.command](args) or 0)


if __name__ == "__main__":
    main()
