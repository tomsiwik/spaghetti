#!/usr/bin/env python3
"""
RunPod SSH executor — run scripts on a pre-created RunPod instance.

Uses SSH config alias 'runpod' (see ~/.ssh/config). You create the pod manually.

Usage:
    python -m composer.runpod_exec test                    # verify connection
    python -m composer.runpod_exec setup                   # install deps + cache model
    python -m composer.runpod_exec sync                    # rsync repo to pod
    python -m composer.runpod_exec run macro/script.py     # sync + run script
    python -m composer.runpod_exec exec "nvidia-smi"       # run arbitrary command
    python -m composer.runpod_exec pull macro/results.json  # pull file back
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SSH_ALIAS = "runpod"
REMOTE_DIR = "/workspace/llm"


def ssh_exec(command, timeout=600, stream=True):
    """Execute command on pod via SSH."""
    cmd = ["ssh", SSH_ALIAS, command]
    if stream:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr


def cmd_test(args):
    """Quick connectivity test."""
    print("Testing SSH connection to RunPod...")
    rc, out, err = ssh_exec(
        "echo 'SSH OK' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python3 --version",
        timeout=15, stream=False,
    )
    if rc == 0:
        print(f"Connected:\n{out.strip()}")
    else:
        print(f"Failed (rc={rc}):\n{err.strip()}")
    return rc


SETUP_SCRIPT = r"""set -euo pipefail
echo "=== RunPod Setup ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"

pip install -q 'unsloth[colab-new]' 'trl>=0.15' 'transformers>=4.45.0' 'peft>=0.13.0' datasets accelerate bitsandbytes scipy scikit-learn pyyaml tqdm 2>&1 | tail -3

python3 -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
hf = os.environ.get('HF_HOME', '/workspace/hf_cache')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', cache_dir=hf)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', cache_dir=hf)
print('Qwen2.5-7B cached.')
"

echo "=== Setup Complete ==="
"""


def cmd_setup(args):
    """First-time setup: install deps, cache model."""
    print("Running setup on RunPod pod...")
    return ssh_exec(SETUP_SCRIPT, timeout=600)


def cmd_sync(args):
    """Sync repo to pod via rsync."""
    ssh_exec(f"mkdir -p {REMOTE_DIR}", timeout=10, stream=False)

    cmd = [
        "rsync", "-rlptz", "--delete",
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "*.pyc",
        "--exclude", ".ralph/diagnostics/logs",
        "--exclude", ".ralph/events-*",
        "--exclude", "macro/capsule_states",
        "--exclude", ".worktrees",
        "--exclude", "references",
        "-e", "ssh",
        f"{REPO_ROOT}/",
        f"{SSH_ALIAS}:{REMOTE_DIR}/",
    ]
    print(f"Syncing {REPO_ROOT} → {SSH_ALIAS}:{REMOTE_DIR}/")
    return subprocess.run(cmd).returncode


def cmd_run(args):
    """Sync repo then run a script on the pod."""
    print("Syncing repo...")
    rc = cmd_sync(args)
    if rc != 0:
        print("Sync failed!")
        return rc

    print(f"Running: {args.script}")
    run_cmd = f"cd {REMOTE_DIR} && export HF_HOME=/workspace/hf_cache && python3 {args.script}"
    return ssh_exec(run_cmd, timeout=args.timeout)


def cmd_exec(args):
    """Run arbitrary command on pod."""
    return ssh_exec(args.command, timeout=args.timeout)


def cmd_pull(args):
    """Pull a file or directory from the pod."""
    remote = args.remote_path
    local = args.local_path or str(REPO_ROOT / Path(remote).name)
    if not remote.startswith("/"):
        remote = f"{REMOTE_DIR}/{remote}"
    print(f"Pulling {remote} → {local}")
    cmd = ["scp", "-r", f"{SSH_ALIAS}:{remote}", str(local)]
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="RunPod SSH executor")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("test", help="Test SSH connection")
    sub.add_parser("setup", help="First-time setup (deps + model cache)")
    sub.add_parser("sync", help="Rsync repo to pod")

    p_run = sub.add_parser("run", help="Sync + run a script")
    p_run.add_argument("script", help="Script path relative to repo root")
    p_run.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")

    p_exec = sub.add_parser("exec", help="Run arbitrary command")
    p_exec.add_argument("command", help="Command to execute")
    p_exec.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    p_pull = sub.add_parser("pull", help="Pull file from pod")
    p_pull.add_argument("remote_path", help="Path on pod (relative to /workspace/llm/ or absolute)")
    p_pull.add_argument("local_path", nargs="?", help="Local destination (default: current dir)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "test": cmd_test, "setup": cmd_setup, "sync": cmd_sync,
        "run": cmd_run, "exec": cmd_exec, "pull": cmd_pull,
    }
    sys.exit(handlers[args.command](args) or 0)


if __name__ == "__main__":
    main()
