#!/usr/bin/env python3
"""
Watch pueue queue and auto-complete experiments in DB based on results.json.

For each experiment task:
  - Wait for it to finish in pueue
  - Read its results.json
  - Update KC results in DB (kill-update)
  - Set experiment status (supported|killed) based on all_pass + verdict
  - Log structured summary

Run: uv run python scripts/auto_complete_experiments.py <task_id> [<task_id>...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Map task_id → experiment_id (extracted from pueue label)
def get_pueue_task(task_id: int) -> dict:
    out = subprocess.run(["pueue", "status", "--json", "--group", "experiments"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    data = json.loads(out.stdout)
    tasks = data.get("tasks", {})
    return tasks.get(str(task_id))


def parse_results(exp_id: str) -> dict | None:
    p = REPO_ROOT / "micro" / "models" / exp_id / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def map_kc_to_db(results: dict) -> dict[int, str]:
    """results['kill_criteria'] keys look like 'K2069_...'. Extract the integer."""
    kc_results = {}
    for key, val in results.get("kill_criteria", {}).items():
        if not key.startswith("K") or "_" not in key:
            continue
        try:
            kc_id = int(key.split("_")[0][1:])
        except ValueError:
            continue
        result = val.get("pass") if isinstance(val, dict) else None
        if result is True:
            kc_results[kc_id] = "pass"
        elif result is False:
            kc_results[kc_id] = "fail"
        elif result is None or val.get("result") == "inconclusive":
            kc_results[kc_id] = "inconclusive"
        else:
            kc_results[kc_id] = "inconclusive"
    return kc_results


def update_db(exp_id: str, results: dict) -> bool:
    """Update kill criteria + experiment status. Returns True on success."""
    kc_map = map_kc_to_db(results)
    print(f"  KC results: {kc_map}", flush=True)

    for kc_id, result in kc_map.items():
        out = subprocess.run(
            ["experiment", "kill-update", exp_id, "--criterion", str(kc_id), "--result", result],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"  WARN: kill-update K{kc_id} failed: {out.stderr.strip()}", flush=True)

    verdict = results.get("verdict", "INCONCLUSIVE")
    all_pass = results.get("all_pass", False)
    is_smoke = results.get("is_smoke", False)

    if is_smoke or verdict == "PROVISIONAL":
        status = "provisional"
    elif verdict == "SUPPORTED" and all_pass:
        status = "supported"
    elif verdict == "KILLED":
        status = "killed"
    else:
        status = "open"

    summary_lines = [f"verdict={verdict}", f"all_pass={all_pass}"]
    for kc_id, result in kc_map.items():
        summary_lines.append(f"K{kc_id}={result}")
    summary = "; ".join(summary_lines)

    out = subprocess.run(
        ["experiment", "complete", exp_id,
         "--status", status,
         "--dir", f"experiments/models/{exp_id}",
         "--evidence", summary],
        capture_output=True, text=True,
    )
    print(f"  experiment complete → status={status}: {out.stdout.strip() or out.stderr.strip()}", flush=True)
    return out.returncode == 0


def watch_task(task_id: int, poll_interval: int = 30) -> bool:
    """Poll until task is done, then complete in DB. Returns True if processed."""
    while True:
        task = get_pueue_task(task_id)
        if task is None:
            print(f"[task {task_id}] not found in pueue", flush=True)
            return False
        status = task.get("status")
        if isinstance(status, dict):
            status_name = list(status.keys())[0] if status else "Unknown"
        else:
            status_name = str(status)

        if status_name in ("Success", "Done"):
            label = task.get("label", f"task_{task_id}")
            exp_id = label  # label is the experiment id by convention
            print(f"\n[task {task_id} = {exp_id}] DONE → reading results", flush=True)
            results = parse_results(exp_id)
            if results is None:
                print(f"  results.json MISSING", flush=True)
                return False
            return update_db(exp_id, results)

        if status_name in ("Failed", "Killed"):
            label = task.get("label", f"task_{task_id}")
            print(f"\n[task {task_id} = {label}] {status_name} — checking partial results", flush=True)
            results = parse_results(label)
            if results:
                update_db(label, results)
            else:
                # Mark experiment killed without results
                subprocess.run(
                    ["experiment", "complete", label, "--status", "killed",
                     "--evidence", f"pueue task {task_id} {status_name}, no results.json produced"],
                )
            return False

        # Still running or queued — poll
        elapsed = task.get("start") and "running" or "queued"
        print(f"[task {task_id}] status={status_name}, sleeping {poll_interval}s...", flush=True)
        time.sleep(poll_interval)


def main():
    if len(sys.argv) < 2:
        print("Usage: auto_complete_experiments.py <task_id> [<task_id>...]")
        sys.exit(1)

    task_ids = [int(t) for t in sys.argv[1:]]
    print(f"Watching pueue tasks: {task_ids}", flush=True)

    for tid in task_ids:
        ok = watch_task(tid)
        print(f"task {tid} processed: {ok}", flush=True)


if __name__ == "__main__":
    main()
