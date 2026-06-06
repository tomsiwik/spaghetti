#!/usr/bin/env python3
"""Runner for MMLU held-out eval with dependency fixes.

Applies set_submodule monkey-patch for RunPod PyTorch compatibility,
then runs eval_mmlu.py --all.

Usage (via gpu_queue):
    uv run python3 tools/gpu_queue.py submit macro/pilot50_held_out_eval/run_mmlu_eval.py
"""

import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_deps():
    """Install missing deps and apply torch patches."""
    deps = ["peft", "bitsandbytes", "datasets"]
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        log(f"Installing missing deps: {missing}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", *missing,
        ])

    import torch
    if not hasattr(torch.nn.Module, "set_submodule"):
        log(f"torch {torch.__version__} missing set_submodule — adding monkey-patch")
        def _set_submodule(self, target, module):
            atoms = target.split(".")
            mod = self
            for item in atoms[:-1]:
                mod = getattr(mod, item)
            setattr(mod, atoms[-1], module)
        torch.nn.Module.set_submodule = _set_submodule
        log("set_submodule patched.")


def main():
    log("=" * 72)
    log("MMLU HELD-OUT EVAL — PILOT 50 EXPERTS")
    log("=" * 72)

    ensure_deps()

    smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "eval_mmlu.py"),
        "--all",
        "--out", "results/held_out_eval/mmlu_all.json",
    ]
    if smoke:
        cmd += ["--max-per-subset", "5"]

    log(f"Running: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        log(f"MMLU eval failed with exit {rc}")
        sys.exit(rc)

    log("MMLU eval complete.")


if __name__ == "__main__":
    main()
