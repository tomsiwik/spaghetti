#!/usr/bin/env python3
"""
Pierre L1-Reflex Classifier — implementation skeleton.

This is the orchestration script. The CORE work — NEON C kernels — needs to be
written separately as bench.c + model.c (modeled on ../talos-vs-macbook/bench_c.c).

Status: SKELETON. Run will fail until bench.c is implemented. Documents
the architecture so anyone can pick up later.

Phases:
  1. Build training data (Pierre-needed vs base-suffices binary task)
  2. Train microGPT-scale model in MLX (16-dim embed, 1-head, 2-layer)
  3. Export weights as packed fp32 binary (model.bin)
  4. Build bench.c with `make` (C + NEON intrinsics, references talos-vs-macbook)
  5. Run benchmark via subprocess, parse latency + accuracy
  6. Compute KCs
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
TALOS_REF = REPO_ROOT.parent / "talos-vs-macbook"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42


def log(m): print(m, flush=True)


def main():
    t0 = time.time()
    log(f"=== Pierre L1-Reflex Classifier (SKELETON; SMOKE={IS_SMOKE}) ===")

    log("\n[Phase 1] Build training data")
    log(f"  positive class: beehive prompts (data/beehive_snapshot/approved.jsonl, 2069 rows)")
    log(f"  positive class: GSM8K-train + HumanEval-train + MedQA-train ({600 if not IS_SMOKE else 30} mixed)")
    log(f"  negative class: trivial conversational, single-fact, greetings ({3000 if not IS_SMOKE else 100} target)")
    log(f"  TODO: write `prepare_training_data.py` — outputs train.jsonl + valid.jsonl with binary labels")

    log("\n[Phase 2] Train microGPT-scale binary classifier in MLX")
    log("  config: 16-dim embed, 1-head, 2-layer transformer (~4K params)")
    log(f"  TODO: write `train.py` — char-level tokenizer, 2K iters Adam, exports weights.bin")

    log("\n[Phase 3] Export weights")
    log("  format: packed fp32 in layer order: [embed][block_0_attn][block_0_mlp][block_1_attn][block_1_mlp][output_proj]")
    log("  size: ~16KB")
    log("  TODO: `export_weights.py` — read MLX checkpoint, write little-endian fp32 binary")

    log("\n[Phase 4] Build C/NEON benchmark")
    if TALOS_REF.exists():
        log(f"  reference: {TALOS_REF}/bench_c.c (microGPT NEON pattern)")
        log(f"  reference: {TALOS_REF}/model.py (loader + sampler logic, port to C)")
    else:
        log(f"  WARN: talos-vs-macbook reference not found at {TALOS_REF}")
    log("  TODO: write `bench.c` with NEON intrinsics — see MATH.md for kernel structure")
    log("  TODO: write `Makefile` — clang -O3 -march=native -ffast-math")

    log("\n[Phase 5] Benchmark")
    log("  TODO: subprocess call to ./bench, parse stdout (n_correct, p50_us, p95_us, peak_kb)")

    log("\n[Phase 6] Latency-with-Pierre measurement")
    log("  TODO: integrate reflex into Pierre serving path, measure end-to-end p95")
    log("  reference: latency methodology from exp_pierre_phase1_e2e_viability/run_experiment.py::measure_p95_first_token_latency_ms")

    # Skeleton results
    results = {
        "is_smoke": IS_SMOKE,
        "status": "SKELETON — implementation pending; see MATH.md for architecture",
        "phases_implemented": 0,
        "phases_total": 6,
        "blocking_files": [
            "prepare_training_data.py",
            "train.py",
            "export_weights.py",
            "bench.c",
            "model.c",
            "Makefile",
        ],
        "verdict": "SKELETON",
        "all_pass": False,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nVERDICT: SKELETON (no execution; see MATH.md)")


if __name__ == "__main__":
    main()
