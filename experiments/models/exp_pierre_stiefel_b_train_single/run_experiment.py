#!/usr/bin/env python3
"""
Stiefel-aware single-adapter training: feasibility test.

Trains a math adapter (GSM8K) under two conditions:
  1. Standard: retract both A and B every 20 steps (existing behavior)
  2. Stiefel-B: retract B every step via QR, A every 20 steps

Compares accuracy, orthogonality, and per-step cost.

Kill criteria:
  K1 CONVERGENCE: Stiefel-trained accuracy >= unconstrained - 5pp
  K2 ORTHOGONALITY: ||B B^T - I||_F <= 1e-3 avg over layers post-training
  K3 STEP COST: per-step time increase <= 25%
  K4 VS POSTPROJ: Stiefel-trained accuracy >= post-hoc projected (40.7% avg)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tooling.scripts.polar_train import (
    PoLARLinear, _get_layers, inject_polar_adapters,
    tokenize_record, collate, loss_fn, _grad_clip,
    eval_gsm8k, cleanup,
    SEED, RANK, SCALE, LR, GRAD_CLIP, BATCH_SIZE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 5 if IS_SMOKE else 50
N_STEPS = 20 if IS_SMOKE else 1000


def prepare_math_data(n_train: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))
    records = []
    for ex in ds:
        records.append({
            "messages": [
                {"role": "user", "content": f"Solve the following math problem step by step.\n\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
        })
    return records


def retract_b_only(modules) -> float:
    """Retract only B to Stiefel manifold (row-orthonormal). Returns max ||BB^T - I||_F."""
    max_dist = 0.0
    I_r = np.eye(RANK)
    for m in modules:
        B_np = np.array(m.lora_b.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(B_np)) or np.sum(B_np ** 2) < 1e-12:
            max_dist = max(max_dist, float(np.sqrt(RANK)))
            continue
        W, _, Vh = np.linalg.svd(B_np, full_matrices=False)
        B_ret = W @ Vh
        m.lora_b = mx.array(B_ret.astype(np.float32))
        dist = float(np.sqrt(np.sum((B_ret @ B_ret.T - I_r) ** 2)))
        max_dist = max(max_dist, dist)
    return max_dist


def retract_a_only(modules) -> float:
    """Retract only A to Stiefel manifold (col-orthonormal). Returns max ||A^T A - I||_F."""
    max_dist = 0.0
    I_r = np.eye(RANK)
    for m in modules:
        A_np = np.array(m.lora_a.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
            max_dist = max(max_dist, float(np.sqrt(RANK)))
            continue
        W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
        A_ret = W @ Vh
        m.lora_a = mx.array(A_ret.astype(np.float32))
        dist = float(np.sqrt(np.sum((A_ret.T @ A_ret - I_r) ** 2)))
        max_dist = max(max_dist, dist)
    return max_dist


def measure_b_orthogonality(modules) -> dict:
    """Measure B orthogonality across all layers."""
    I_r = np.eye(RANK)
    dists = []
    for m in modules:
        B_np = np.array(m.lora_b.tolist(), dtype=np.float64)
        dist = float(np.sqrt(np.sum((B_np @ B_np.T - I_r) ** 2)))
        dists.append(dist)
    return {
        "avg": float(np.mean(dists)),
        "max": float(np.max(dists)),
        "min": float(np.min(dists)),
        "per_layer": dists,
    }


def train_with_config(model, tokenizer, records, modules, n_steps: int,
                      retract_b_every: int, retract_a_every: int = 20,
                      label: str = "train") -> dict:
    """Train with configurable retraction frequencies for A and B."""
    optimizer = optim.Adam(learning_rate=LR)
    rng = np.random.default_rng(SEED)
    tokenized = [tokenize_record(tokenizer, r) for r in records]
    n_data = len(tokenized)
    grad_fn = nn.value_and_grad(model, loss_fn)
    losses = []
    step_times = []

    print(f"\n=== {label}: retract_b_every={retract_b_every}, retract_a_every={retract_a_every} ===", flush=True)

    for step in range(n_steps):
        idx = rng.choice(n_data, size=min(BATCH_SIZE, n_data), replace=(n_data < BATCH_SIZE))
        batch = [tokenized[i] for i in idx]
        inputs, labels_batch = collate(batch)

        t0 = time.perf_counter()
        loss, grads = grad_fn(model, inputs, labels_batch)
        grads = _grad_clip(grads, GRAD_CLIP)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if retract_b_every > 0 and (step + 1) % retract_b_every == 0:
            retract_b_only(modules)
            mx.eval(model.parameters())

        if retract_a_every > 0 and (step + 1) % retract_a_every == 0:
            retract_a_only(modules)
            mx.eval(model.parameters())

        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            losses.append(loss_val)
            print(f"[{label} step {step}] DIVERGED loss={loss_val}", flush=True)
            break
        losses.append(loss_val)

        if step % 100 == 0:
            print(f"[{label} step {step}/{n_steps}] loss={loss_val:.4f} step_time={step_times[-1]:.3f}s", flush=True)

    # Final retraction
    retract_b_only(modules)
    retract_a_only(modules)
    mx.eval(model.parameters())

    orth = measure_b_orthogonality(modules)
    avg_step_time = float(np.mean(step_times[10:])) if len(step_times) > 10 else float(np.mean(step_times))

    return {
        "final_loss": losses[-1] if losses else float("nan"),
        "first_loss": losses[0] if losses else float("nan"),
        "n_steps_completed": len(losses),
        "any_nan": any(not math.isfinite(l) for l in losses),
        "avg_step_time_s": avg_step_time,
        "b_orthogonality": orth,
        "losses_sampled": losses[::max(1, len(losses)//20)],
    }


def load_model():
    from mlx_lm import load
    print(f"Loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def main():
    print("=== Stiefel-B single-adapter training experiment ===", flush=True)
    print(f"SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}", flush=True)

    # Phase 1: Prepare data
    print("\n--- Phase 1: Data preparation ---", flush=True)
    records = prepare_math_data(N_TRAIN)
    print(f"Prepared {len(records)} training records", flush=True)

    # Phase 2: Train STANDARD (retract both A/B every 20 steps)
    print("\n--- Phase 2: Standard training ---", flush=True)
    model, tokenizer = load_model()
    modules_std = inject_polar_adapters(model, rank=RANK, scale=SCALE, seed=SEED)
    std_result = train_with_config(
        model, tokenizer, records, modules_std, N_STEPS,
        retract_b_every=20, retract_a_every=20, label="standard"
    )

    # Eval standard adapter
    print("\n--- Phase 2b: Eval standard ---", flush=True)
    std_gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"Standard GSM8K: {std_gsm8k:.1f}%", flush=True)
    std_result["gsm8k_accuracy"] = std_gsm8k

    # Cleanup
    cleanup(model, tokenizer)

    # Phase 3: Train STIEFEL-B (retract B every step, A every 20)
    print("\n--- Phase 3: Stiefel-B training ---", flush=True)
    model, tokenizer = load_model()
    modules_stf = inject_polar_adapters(model, rank=RANK, scale=SCALE, seed=SEED)
    stf_result = train_with_config(
        model, tokenizer, records, modules_stf, N_STEPS,
        retract_b_every=1, retract_a_every=20, label="stiefel_b"
    )

    # Eval Stiefel adapter
    print("\n--- Phase 3b: Eval Stiefel-B ---", flush=True)
    stf_gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"Stiefel-B GSM8K: {stf_gsm8k:.1f}%", flush=True)
    stf_result["gsm8k_accuracy"] = stf_gsm8k

    cleanup(model, tokenizer)

    # Phase 4: Evaluate kill criteria
    print("\n--- Phase 4: Kill criteria evaluation ---", flush=True)

    delta_accuracy = stf_gsm8k - std_gsm8k
    k1_pass = delta_accuracy >= -5.0
    print(f"K1 CONVERGENCE: Stiefel={stf_gsm8k:.1f}% vs Standard={std_gsm8k:.1f}% "
          f"(Δ={delta_accuracy:+.1f}pp, threshold=-5pp) → {'PASS' if k1_pass else 'FAIL'}", flush=True)

    b_orth_avg = stf_result["b_orthogonality"]["avg"]
    k2_pass = b_orth_avg <= 1e-3
    print(f"K2 ORTHOGONALITY: avg ||BB^T-I||_F = {b_orth_avg:.6f} "
          f"(threshold=1e-3) → {'PASS' if k2_pass else 'FAIL'}", flush=True)

    time_ratio = stf_result["avg_step_time_s"] / std_result["avg_step_time_s"]
    time_increase_pct = (time_ratio - 1.0) * 100
    k3_pass = time_increase_pct <= 25.0
    print(f"K3 STEP COST: Stiefel={stf_result['avg_step_time_s']:.4f}s vs "
          f"Standard={std_result['avg_step_time_s']:.4f}s "
          f"(+{time_increase_pct:.1f}%, threshold=25%) → {'PASS' if k3_pass else 'FAIL'}", flush=True)

    postproj_avg = 40.7  # from exp_pierre_stiefel_b_postproj
    k4_pass = stf_gsm8k >= postproj_avg
    print(f"K4 VS POSTPROJ: Stiefel-trained={stf_gsm8k:.1f}% vs "
          f"post-hoc-projected avg={postproj_avg:.1f}% → {'PASS' if k4_pass else 'FAIL'}", flush=True)

    all_pass = k1_pass and k2_pass and k3_pass and k4_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    print(f"\n=== VERDICT: {verdict} ===", flush=True)

    results = {
        "verdict": verdict,
        "all_pass": all_pass,
        "config": {
            "model": MODEL_ID,
            "rank": RANK,
            "scale": SCALE,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "n_steps": N_STEPS,
            "is_smoke": IS_SMOKE,
            "seed": SEED,
        },
        "standard": std_result,
        "stiefel_b": stf_result,
        "kill_criteria": {
            "K1_convergence": {
                "pass": k1_pass,
                "stiefel_gsm8k": stf_gsm8k,
                "standard_gsm8k": std_gsm8k,
                "delta_pp": delta_accuracy,
                "threshold_pp": -5.0,
            },
            "K2_orthogonality": {
                "pass": k2_pass,
                "avg_frobenius": b_orth_avg,
                "max_frobenius": stf_result["b_orthogonality"]["max"],
                "threshold": 1e-3,
            },
            "K3_step_cost": {
                "pass": k3_pass,
                "stiefel_step_time_s": stf_result["avg_step_time_s"],
                "standard_step_time_s": std_result["avg_step_time_s"],
                "increase_pct": time_increase_pct,
                "threshold_pct": 25.0,
            },
            "K4_vs_postproj": {
                "pass": k4_pass,
                "stiefel_trained_gsm8k": stf_gsm8k,
                "postproj_avg": postproj_avg,
            },
        },
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
