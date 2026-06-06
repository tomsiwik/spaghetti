#!/usr/bin/env python3
"""
exp_jepa_adapter_residual_stream — JEPA residual-stream predictor vs token-space LoRA.

Pre-registered KCs (canonical DB text — do not edit):
  K#1766 structural: SIGReg Epps-Pulley rejection rate < 5% on adapter output activations at step 500 (no collapse)
  K#1767 proxy:      prediction loss L_pred(step 500) / L_pred(step 50) < 0.5
  K#1768 target:     GSM8K-Hard accuracy >= token-space r=16 LoRA baseline at matched param budget, n>=200, greedy
  K#1769 ablation:   removing SIGReg (lambda=0) degrades K1768 by >= 5pp

Skills invoked: /mlx-dev + /fast-mlx (documented in MATH.md §0).

Runtime structure (3 phases + ablation = 4 training runs):
  Phase A — token-space r=16 LoRA baseline (mlx_lm.lora subprocess)
  Phase B — JEPA adapter with SIGReg (custom MLX training loop)
  Phase C — JEPA ablation (lambda=0, custom MLX training loop)
  Phase D — GSM8K-Hard evaluation (n=200 greedy) for all three adapters

Full budget: ~4-6h on M5 Pro 48GB. Exceeds single-iteration researcher cap (30 min / 40 tool calls).
This scaffold supports SMOKE_TEST=1 (fast plumbing check) and full runs.
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 10 if IS_SMOKE else 200
N_STEPS = 50 if IS_SMOKE else 500
SEED = 42

ADAPTER_RANK = 16
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # <= 8 per F#328/F#330
PREDICT_LAYER = 21                      # middle of 42-layer Gemma 4 E4B
PREDICT_SHIFT = 1                       # predict h_{t+1} from h_t
SIGREG_M = 1024                         # random projections
SIGREG_LAMBDAS = [0.0, 0.1, 1.0, 10.0]  # bisection grid


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def prepare_gsm8k_train(data_dir: Path, n_train: int):
    """GSM8K training split as chat-formatted JSONL."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    records = []
    for ex in ds:
        msg = {
            "messages": [
                {"role": "user", "content": f"Solve the following math problem step by step.\n\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }
        records.append(json.dumps(msg))

    n_val = max(1, len(records) // 10)
    train_file.write_text("\n".join(records[n_val:]))
    valid_file.write_text("\n".join(records[:n_val]))
    print(f"GSM8K train prepared: {len(records)-n_val} train / {n_val} val", flush=True)


# ─────────────────────────────────────────────
# Phase A: token-space LoRA baseline
# ─────────────────────────────────────────────

def write_lora_config(config_path: Path, data_dir: Path, adapter_path: Path, rank: int, steps: int):
    import yaml

    config = {
        "model": MODEL_ID,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,
        "iters": steps,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": rank,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": [f"self_attn.{t}" for t in ADAPTER_TARGETS],
        },
        "adapter_path": str(adapter_path),
        "save_every": steps,
        "val_batches": 5,
        "steps_per_report": max(10, steps // 10),
        "steps_per_eval": steps,
        "max_seq_length": 512,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "seed": SEED,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def train_token_lora(data_dir: Path, adapter_path: Path) -> dict:
    """Phase A: token-space r=16 LoRA baseline via mlx_lm.lora."""
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = EXPERIMENT_DIR / "lora_config_baseline.yaml"
    write_lora_config(config_path, data_dir, adapter_path, ADAPTER_RANK, N_STEPS)

    print(f"\n=== Phase A: Token-space LoRA r={ADAPTER_RANK} baseline ({N_STEPS} steps) ===", flush=True)
    t0 = time.time()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    result = subprocess.run(cmd, text=True, cwd=str(EXPERIMENT_DIR))

    elapsed = time.time() - t0
    print(f"Phase A done in {elapsed:.1f}s (exit={result.returncode})", flush=True)
    return {"train_time_s": round(elapsed, 1), "exit_code": result.returncode}


# ─────────────────────────────────────────────
# Phase B/C: JEPA custom training loop
# ─────────────────────────────────────────────
#
# NOT-YET-IMPLEMENTED. Requires:
# 1. Loading mlx-community/gemma-4-e4b-it-4bit via mlx_lm.load.
# 2. Attaching LoRA on v_proj + o_proj at rank 16 (mlx_lm.tuner.lora.LoRALinear.from_linear).
# 3. Hook on model.layers[21].residual stream to capture h_ℓ(t) for t in batch.
# 4. Prediction head P_θ: 2-layer MLP hidden_dim=2304, trained end-to-end with adapter.
# 5. Forward pass on batch B of token sequences; compute h_ℓ(t) and h_ℓ(t+1)=stopgrad(h_ℓ shifted by 1).
# 6. Loss = MSE(P_θ(h_ℓ(t)), stopgrad(h_ℓ(t+1))) + λ · SIGReg(Z) where Z = P_θ output batch.
# 7. SIGReg: sample M=1024 unit vectors u_m, project Z · u_m, compute Epps-Pulley statistic vs N(0,1).
# 8. Gradient step via nn.value_and_grad + mlx.optimizers.AdamW.
# 9. mx.eval(model.parameters(), loss) at step boundary; mx.clear_cache() between batches.
# 10. Save adapter weights compatible with mlx_lm adapter loading (prediction head discarded at inference).
#
# Scope-preservation (researcher guardrail): do NOT silently fall back to standard LoRA training
# if JEPA training fails. Report failure honestly, mark K1766/K1767 as UNMEASURABLE.

def train_jepa_adapter(data_dir: Path, adapter_path: Path, lam: float) -> dict:
    """Phase B/C: JEPA residual-stream predictor with SIGReg lambda=lam.

    NOT YET IMPLEMENTED. Returns NotImplementedError with structured marker
    so the pipeline can record a PROVISIONAL verdict without silent degradation.
    """
    raise NotImplementedError(
        "JEPA adapter training loop not yet implemented. "
        "Requires custom MLX training: layer-21 hook + prediction head + SIGReg Epps-Pulley. "
        "See PAPER.md §Measurement blockers."
    )


def measure_sigreg_rejection(adapter_path: Path) -> float:
    """K#1766: fraction of M=1024 projections where Epps-Pulley rejects N(0,1) at α=0.05.

    NOT YET IMPLEMENTED. Requires forward-pass activation capture from trained adapter.
    """
    raise NotImplementedError("SIGReg measurement depends on Phase B completing.")


# ─────────────────────────────────────────────
# Phase D: GSM8K-Hard evaluation
# ─────────────────────────────────────────────

def eval_gsm8k_hard(adapter_path, n_eval: int) -> float:
    """GSM8K test split, greedy, n=N_EVAL. Returns accuracy 0-100.

    'Hard' is operationalized as the test split with max_tokens=1024 (per F#1629 recovery).
    """
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    correct = 0
    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = correct / len(ds) * 100
    print(f"GSM8K accuracy: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")

    print(f"JEPA residual-stream adapter vs token-space LoRA", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "gsm8k"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    baseline_adapter = adapters_dir / "baseline_lora_r16"
    jepa_adapter = adapters_dir / "jepa_sigreg"
    jepa_ablation_adapter = adapters_dir / "jepa_ablation"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
        "model_id": MODEL_ID,
        "adapter_rank": ADAPTER_RANK,
        "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE,
        "predict_layer": PREDICT_LAYER,
        "sigreg_M": SIGREG_M,
        "sigreg_lambdas": SIGREG_LAMBDAS,
        "phase_a_baseline": None,
        "phase_b_jepa": None,
        "phase_c_ablation": None,
        "phase_d_eval": {},
        "kc": {
            "K1766_sigreg_rejection_lt_5pct": "untested",
            "K1767_loss_ratio_lt_0_5": "untested",
            "K1768_gsm8k_hard_ge_baseline": "untested",
            "K1769_ablation_gap_ge_5pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: dataset ────────────────────────────────────
    print("\n=== Phase 0: Dataset ===", flush=True)
    prepare_gsm8k_train(data_dir, N_TRAIN)

    # ── Phase A: token-space LoRA baseline ─────────────────
    try:
        phase_a = train_token_lora(data_dir, baseline_adapter)
        results["phase_a_baseline"] = phase_a
    except Exception as exc:
        results["phase_a_baseline"] = {"error": str(exc)}
        results["blockers"].append(f"Phase A failed: {exc}")

    # ── Phase B: JEPA adapter with SIGReg ──────────────────
    try:
        phase_b = train_jepa_adapter(data_dir, jepa_adapter, lam=1.0)
        results["phase_b_jepa"] = phase_b
    except NotImplementedError as exc:
        results["phase_b_jepa"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B not implemented: {exc}")
    except Exception as exc:
        results["phase_b_jepa"] = {"error": str(exc)}
        results["blockers"].append(f"Phase B failed: {exc}")

    # ── Phase C: JEPA ablation (lambda=0) ──────────────────
    try:
        phase_c = train_jepa_adapter(data_dir, jepa_ablation_adapter, lam=0.0)
        results["phase_c_ablation"] = phase_c
    except NotImplementedError as exc:
        results["phase_c_ablation"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase C not implemented: {exc}")
    except Exception as exc:
        results["phase_c_ablation"] = {"error": str(exc)}
        results["blockers"].append(f"Phase C failed: {exc}")

    # ── Phase D: GSM8K-Hard eval ──────────────────────────
    if results["phase_a_baseline"] and results["phase_a_baseline"].get("exit_code") == 0:
        try:
            baseline_acc = eval_gsm8k_hard(baseline_adapter, N_EVAL)
            results["phase_d_eval"]["baseline_acc_pct"] = round(baseline_acc, 1)
        except Exception as exc:
            results["blockers"].append(f"Phase D baseline eval failed: {exc}")
    else:
        results["blockers"].append("Phase D baseline eval skipped: Phase A did not complete")

    # JEPA + ablation eval skipped if training didn't produce adapters
    if jepa_adapter.exists():
        try:
            jepa_acc = eval_gsm8k_hard(jepa_adapter, N_EVAL)
            results["phase_d_eval"]["jepa_acc_pct"] = round(jepa_acc, 1)
        except Exception as exc:
            results["blockers"].append(f"Phase D JEPA eval failed: {exc}")

    if jepa_ablation_adapter.exists():
        try:
            abl_acc = eval_gsm8k_hard(jepa_ablation_adapter, N_EVAL)
            results["phase_d_eval"]["ablation_acc_pct"] = round(abl_acc, 1)
        except Exception as exc:
            results["blockers"].append(f"Phase D ablation eval failed: {exc}")

    # ── KC resolution ──────────────────────────────────────
    # With Phase B not implemented, K#1766, K#1767, K#1768, K#1769 remain 'untested'.
    # When Phase B lands: compute KCs from results["phase_b_jepa"] loss trace + activation stats
    # and results["phase_d_eval"] accuracy comparisons.

    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Blockers: {results['blockers']}", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
